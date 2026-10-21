"""R87 matched shared/private residuals using the frozen R84 training engine."""

import argparse
import sys
from pathlib import Path

import torch

from code.unified_selector.registry import GLOBAL_SOLVERS, PROBLEMS
from . import r84_experiment as engine
from .r76_experiment import RDropObjective, file_hash, read_json
from .r77_data import parameter_hash
from .r84_prepare import DEFAULT_DATA_ROOT, DEFAULT_PREPARED_ROOT, SPLIT_ROWS
from .r85_experiment import ensure_test_cache, evaluate, selection_counts, zero_summary
from .train import set_seed


V4 = Path(__file__).resolve().parent
DEFAULT_ROOT = V4 / 'runs/R87_private_method_residual'
ARMS = ('A_shared_residual', 'B_private_residual')
BATCH_SIZE, EVAL_EVERY = engine.BATCH_SIZE, engine.EVAL_EVERY
WARMUP_UPDATES, MAX_UPDATES = engine.WARMUP_UPDATES, engine.MAX_UPDATES
PEAK_LR, END_LR = engine.PEAK_LR, engine.END_LR
WINNER_RULE, MODEL_FACTORY = engine.WINNER_RULE, 'code.V4.r87_experiment.load_checkpoint'
learning_rate, consistency_weight = engine.learning_rate, engine.consistency_weight
micro_caps, task_order, json_hash = engine.micro_caps, engine.task_order, engine.json_hash
prepared_config, status_update = engine.prepared_config, engine.status_update
RESIDUAL_HIDDEN = 32


def make_model(params, arm, state_dict=None):
    if arm not in ARMS:
        raise ValueError('Unknown R87 arm')
    from .r87_model import make_model as factory
    return factory(params, arm, state_dict=state_dict)


def initialize_shared(initial, geometry, arm, device='cpu'):
    reference = initial['trainable']
    if (initial.get('seed') != 2 or initial.get('initialization') != 'scratch'
            or parameter_hash(reference) != initial.get('trainable_sha256')):
        raise ValueError('R87 requires the verified fresh seed2 R84/R77 parameter source')
    from .r87_model import initialize_from_fresh_common
    set_seed(2)
    model = initialize_from_fresh_common(initial, initial['model_params'], arm)
    actual = dict(model.named_parameters())
    if set(reference) - set(actual) or any(actual[name].shape != value.shape for name, value in reference.items()):
        raise ValueError('R87 must retain every original parameter name and shape')
    new_names = sorted(set(actual) - set(reference))
    dim = int(initial['model_params'].get('embedding_dim', 128))
    expected_added = len(GLOBAL_SOLVERS) * RESIDUAL_HIDDEN * (3 * dim + 1)
    if (new_names != ['residual.W', 'residual.v']
            or sum(actual[name].numel() for name in new_names) != expected_added
            or torch.count_nonzero(actual['residual.v']).item() != 0):
        raise ValueError('Only the matched bias-free residual parameter budget may be added')
    with torch.no_grad():
        branch = model.instance_encoder.geometry_residual
        branch.mean.copy_(torch.as_tensor(geometry['mean'], dtype=branch.mean.dtype))
        branch.std.copy_(torch.as_tensor(geometry['std'], dtype=branch.std.dtype))
    common = {name: actual[name] for name in reference}
    if parameter_hash(common) != initial['trainable_sha256']:
        raise ValueError('The full common fresh initialization changed before R87 update1')
    receipt = dict(seed=2, initialization='scratch_shared_r77', historical_checkpoint_loaded=False,
        r77_fresh_trainable_sha256=initial['trainable_sha256'],
        common_parameter_names=sorted(reference), common_parameter_sha256=parameter_hash(common),
        common_parameter_count=sum(value.numel() for value in common.values()),
        new_parameter_names=new_names, new_parameter_count=expected_added,
        new_parameter_sha256=parameter_hash({name: actual[name] for name in new_names}),
        trainable_sha256=parameter_hash(actual), trainable_parameter_count=sum(value.numel() for value in actual.values()),
        removed_parameter_names=[], geometry_fitted_on='frozen R84 new81000 TRAIN only',
        residual_input='original 3*d pre-score features', residual_bias=False,
        residual_hidden=RESIDUAL_HIDDEN, residual_output_initialization='verified all zeros',
        residual_parameter_report=model.parameter_report())
    return model.to(device), dict(model.params), receipt


def make_objective(selector, arm):
    if arm not in ARMS:
        raise ValueError('Unknown R87 arm')
    return RDropObjective(selector)


def checkpoint_extras(objective):
    if type(objective) is not RDropObjective:
        raise TypeError('R87 uses the original stateless R77/R-Drop objective')
    return {}


def restore_objective(objective, checkpoint):
    if type(objective) is not RDropObjective or 'objective_state' in checkpoint:
        raise ValueError('R85 mutable objective state cannot be resumed into R87')


def update_batch(objective, optimizer, scaler, batch, weight, runtime, cap,
                 max_retries=20, *, problem):
    if problem not in PROBLEMS:
        raise ValueError('R87 requires a registered single-problem logical batch')
    parts, retries, _ = engine.update_batch(objective, optimizer, scaler, batch, weight,
        runtime, cap, max_retries=max_retries)
    return parts, retries, {}


def protocol_lock(args):
    path = args.protocol_lock or args.root / 'protocol_lock.json'
    record = read_json(path)
    expected = dict(experiment='R87', locked=True, arms=list(ARMS), seed=2,
        batch_size=BATCH_SIZE, max_updates=MAX_UPDATES, warmup_updates=WARMUP_UPDATES,
        eval_every=EVAL_EVERY)
    if any(record.get(name) != value for name, value in expected.items()):
        raise ValueError('R87 requires main\'s unchanged two-arm protocol lock')
    numeric = dict(peak_lr=PEAK_LR, end_lr=END_LR, weight_decay=1e-4, dropout=.1,
        grad_clip=1., large_micro_cap=352, eval_batch_size=256, private_hidden_dim=32)
    if any(name in record and record[name] != value for name, value in numeric.items()):
        raise ValueError('R87 lock disagrees with the fixed matched training controls')
    for name in ('data_root', 'prepared_root'):
        if name in record and Path(record[name]).resolve() != getattr(args, name).resolve():
            raise ValueError('R87 lock names a different ' + name)
    if record.get('rows_per_problem', SPLIT_ROWS) != SPLIT_ROWS:
        raise ValueError('R87 must retain the frozen NEW90k rows')
    return dict(path=str(path.resolve()), sha256=file_hash(path), record=record)


def protocol(args, ready, manifest, geometry, baselines, params, order, caps):
    if (args.gpu_type != '4090' or args.eval_batch_size != 256
            or caps['TSP'] != 352 or caps['CVRP'] != 352):
        raise ValueError('Both R87 arms use the prelocked 4090 microcap352/eval256 execution')
    config = engine.protocol(args, ready, manifest, geometry, baselines, params, order, caps)
    config.update(experiment='R87', r87_schema_version=1, model_factory=MODEL_FACTORY,
        main_protocol_lock=protocol_lock(args),
        objective=dict(classification='original native CE', extra_loss=False,
            coefficients=dict(ce=.35, comparison=.35, risk=.02),
            consistency='original R-Drop ramp to .35', mutable_objective_state=False),
        residual=dict(kind='shared' if args.arm == ARMS[0] else 'private_global_method',
            input='original 3*d score features', hidden_per_method=32,
            shared_hidden=len(GLOBAL_SOLVERS) * 32, bias=False, dropout=0.,
            activation='GELU', output_initialization='zero', added_parameter_counts='exactly matched'),
        historical_reference=dict(model='R84A', same_frozen_new90k_split=True,
            run=str((engine.DEFAULT_ROOT / 'A_baseline').resolve()), retrained=False),
        objective_state='none: original RDropObjective; no EQL/queue/subset state',
        selection_head='original_score_mlp_plus_residual_maximin',
        inference_rule='eval mode, original shared utility plus residual, original pair differences/maximin; stable ascending-global-solver-ID tie break',
        nss_reference='reuse exact4500 R84 predictions from locked R79 weights; no NSS inference/retraining',
        source_sha256={name: file_hash(V4 / name) for name in
            ('r87_experiment.py', 'r87_model.py', 'r87_reporting.py',
             'r84_experiment.py', 'r84_prepare.py', 'r84_reporting.py',
             'r85_experiment.py', 'r85_reporting.py', 'r80_reporting.py',
             'r76_experiment.py', 'r77_experiment.py', 'pairwise_selector.py',
             'pairwise_objective.py', 'solver_code_encoder.py')})
    config['loss'].pop('focal_extra', None)
    return config


def load_checkpoint(path, device='cpu'):
    state = torch.load(path, map_location='cpu', weights_only=False)
    config = state['args']
    if (config.get('experiment') != 'R87' or config.get('r87_schema_version') != 1
            or config.get('arm') not in ARMS or config.get('seed') != 2
            or config.get('initialization') != 'scratch_shared_r77'
            or config.get('solver_names') != GLOBAL_SOLVERS
            or config.get('max_successful_updates') != MAX_UPDATES
            or state.get('state_hash') != json_hash(config) or 'objective_state' in state):
        raise ValueError('Checkpoint is not an unchanged fresh R87 model with the original objective')
    model = make_model(config['model_params'], config['arm'], state_dict=state['model'])
    return model.to(device), state


load_model = load_checkpoint
from_checkpoint = load_checkpoint
authorize_test = engine.authorize_test


def train(args, device):
    protocol_lock(args)
    return engine.train(args, device, integration=sys.modules[__name__])


def test(args, device):
    return engine.test(args, device, integration=sys.modules[__name__])


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('command', choices=('train', 'test', 'describe', 'status'))
    result.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    result.add_argument('--arm', choices=ARMS, required=True)
    result.add_argument('--prepared-root', type=Path, default=DEFAULT_PREPARED_ROOT)
    result.add_argument('--data-root', type=Path, default=DEFAULT_DATA_ROOT)
    result.add_argument('--cache-root', type=Path)
    result.add_argument('--protocol-lock', type=Path)
    result.add_argument('--device', default='cuda:0')
    result.add_argument('--gpu-type', choices=('4090',), default='4090')
    result.add_argument('--large-micro-cap', type=int, default=352)
    result.add_argument('--eval-batch-size', type=int, default=256)
    result.add_argument('--log-every', type=int, default=18)
    result.add_argument('--wandb-mode', choices=('auto', 'online', 'offline'), default='auto')
    result.add_argument('--resume', action='store_true')
    result.add_argument('--state-hash', '--expected-state-hash')
    result.set_defaults(experiment='R87')
    return result


def describe(args):
    return dict(experiment='R87', arm=args.arm, rows=SPLIT_ROWS, seed=2,
        batch_size=BATCH_SIZE, max_updates=MAX_UPDATES, warmup_updates=WARMUP_UPDATES,
        eval_every=EVAL_EVERY, model_factory=MODEL_FACTORY,
        objective='unchanged R77 CE/comparison/risk/R-Drop',
        inference='original shared utility plus matched residual then maximin',
        data_root=str(args.data_root), prepared_root=str(args.prepared_root),
        microbatch_caps=micro_caps(args.gpu_type, args.large_micro_cap),
        training_requires='main-locked R87 protocol; launcher-owned assigned GPU')


def main(argv=None):
    return engine.main(argv, integration=sys.modules[__name__])


if __name__ == '__main__':
    main()
