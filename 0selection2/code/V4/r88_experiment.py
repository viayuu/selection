"""R88 full18 controlled TSP execution-role inputs, no new parameters/losses."""

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


V4 = Path(__file__).resolve().parent
DEFAULT_ROOT = V4 / 'runs/R88_execution_role_inputs'
ARMS = ('A_canonical_anchor', 'B_execution_anchor')
BATCH_SIZE, EVAL_EVERY = engine.BATCH_SIZE, engine.EVAL_EVERY
WARMUP_UPDATES, MAX_UPDATES = engine.WARMUP_UPDATES, engine.MAX_UPDATES
PEAK_LR, END_LR = engine.PEAK_LR, engine.END_LR
WINNER_RULE, MODEL_FACTORY = engine.WINNER_RULE, 'code.V4.r88_experiment.load_checkpoint'
learning_rate, consistency_weight = engine.learning_rate, engine.consistency_weight
micro_caps, task_order, json_hash = engine.micro_caps, engine.task_order, engine.json_hash
prepared_config, status_update = engine.prepared_config, engine.status_update
ROLE_SLOT = 3
SELECTION_HEAD = 'original_score_mlp_maximin'


def make_model(params, arm, state_dict=None):
    if arm not in ARMS:
        raise ValueError('Unknown R88 arm')
    if any(name.startswith('residual_') for name in params):
        raise ValueError('R87 residual reconstruction fields do not belong to R88')
    from .r88_model import make_model as factory
    model = factory(params, arm, state_dict=state_dict)
    if model.params.get('selection_head', SELECTION_HEAD) != SELECTION_HEAD:
        raise ValueError('R88 retains the original score MLP and maximin selection head')
    return model


def initialize_shared(initial, geometry, arm, device='cpu'):
    model, params, receipt = engine.initialize_shared(initial, geometry, arm, device,
        model_factory=make_model)
    actual = dict(model.named_parameters())
    if (set(actual) != set(initial['trainable'])
            or parameter_hash(actual) != initial['trainable_sha256']):
        raise ValueError('R88 must retain every original fresh trainable with zero added parameters')
    receipt.update(common_parameter_count=sum(value.numel() for value in actual.values()),
        new_parameter_count=0, trainable_parameter_count=sum(value.numel() for value in actual.values()),
        role_arm=arm, role_spec=params.get('role_spec'), role_slot=ROLE_SLOT,
        geometry_fitted_on='frozen R84 new81000 TRAIN only')
    return model, params, receipt


def make_objective(selector, arm):
    if arm not in ARMS:
        raise ValueError('Unknown R88 arm')
    return RDropObjective(selector)


def checkpoint_extras(objective):
    if type(objective) is not RDropObjective:
        raise TypeError('R88 uses the original stateless R77/R-Drop objective')
    return {}


def restore_objective(objective, checkpoint):
    if type(objective) is not RDropObjective or 'objective_state' in checkpoint:
        raise ValueError('R85 mutable objective state cannot be resumed into R88')


def update_batch(objective, optimizer, scaler, batch, weight, runtime, cap,
                 max_retries=20, *, problem):
    if problem not in PROBLEMS:
        raise ValueError('R88 requires a registered single-task logical batch')
    parts, retries, _ = engine.update_batch(objective, optimizer, scaler, batch, weight,
        runtime, cap, max_retries=max_retries)
    return parts, retries, {}


def protocol_lock(args):
    path = args.protocol_lock or args.root / 'protocol_lock.json'
    record = read_json(path)
    expected = dict(experiment='R88', locked=True, arms=list(ARMS), seed=2,
        batch_size=BATCH_SIZE, max_updates=MAX_UPDATES, warmup_updates=WARMUP_UPDATES,
        eval_every=EVAL_EVERY)
    if any(record.get(name) != value for name, value in expected.items()):
        raise ValueError('R88 requires main\'s unchanged controlled two-arm protocol lock')
    fixed = dict(peak_lr=PEAK_LR, end_lr=END_LR, weight_decay=1e-4, dropout=.1,
        grad_clip=1., large_micro_cap=352, eval_batch_size=256, role_slot=ROLE_SLOT,
        role_feature_slot=ROLE_SLOT, new_parameter_count=0, new_trainable_parameters=0,
        gpu_type='3090')
    if any(name in record and record[name] != value for name, value in fixed.items()):
        raise ValueError('R88 lock disagrees with the fixed training/input controls')
    for name in ('data_root', 'prepared_root'):
        if name in record and Path(record[name]).resolve() != getattr(args, name).resolve():
            raise ValueError('R88 lock names a different ' + name)
    if record.get('rows_per_problem', SPLIT_ROWS) != SPLIT_ROWS:
        raise ValueError('R88 retains the complete frozen NEW90k data split')
    return dict(path=str(path.resolve()), sha256=file_hash(path), record=record)


def protocol(args, ready, manifest, geometry, baselines, params, order, caps):
    if (args.gpu_type != '3090' or args.eval_batch_size != 256
            or caps['TSP'] != 352 or caps['CVRP'] != 352):
        raise ValueError('Both R88 arms use the prelocked 3090 microcap352/eval256 execution')
    config = engine.protocol(args, ready, manifest, geometry, baselines, params, order, caps)
    config.update(experiment='R88', r88_schema_version=1, model_factory=MODEL_FACTORY,
        main_protocol_lock=protocol_lock(args),
        objective=dict(classification='original native CE', extra_loss=False,
            coefficients=dict(ce=.35, comparison=.35, risk=.02),
            consistency='original R-Drop ramp to .35', mutable_objective_state=False),
        input_roles=dict(arm=args.arm, problem='TSP', node_column=ROLE_SLOT,
            assignment='input-only min x then min y, all exact coordinate ties' if args.arm == ARMS[0] else 'physical input node0',
            model_role_spec=params.get('role_spec'), label_dependent=False,
            other17_tasks='original inputs and mathematical forward',
            new_parameter_count=0, preprocessing=False),
        historical_reference=dict(model='R84A', same_frozen_new90k_split=True,
            run=str((engine.DEFAULT_ROOT / 'A_baseline').resolve()), retrained=False),
        objective_state='none: original stateless RDropObjective',
        selection_head=SELECTION_HEAD,
        inference_rule='eval mode, input-derived TSP role in slot3, original R77 score MLP/pair differences/maximin; stable ascending-global-solver-ID tie break',
        nss_reference='reuse exact4500 R84 predictions from locked R79 weights; no NSS inference/retraining',
        evidence_boundary='controlled known-role input experiment; not proof of the entire historical Top1 plateau cause',
        source_sha256={name: file_hash(V4 / name) for name in
            ('r88_experiment.py', 'r88_model.py', 'r88_reporting.py',
             'r84_experiment.py', 'r84_prepare.py', 'r84_reporting.py',
             'r85_experiment.py', 'r85_reporting.py', 'r87_experiment.py', 'r87_reporting.py',
             'r80_reporting.py', 'r76_experiment.py', 'r77_experiment.py',
             'pairwise_selector.py', 'pairwise_objective.py', 'solver_code_encoder.py')})
    config['loss'].pop('focal_extra', None)
    return config


def load_checkpoint(path, device='cpu'):
    state = torch.load(path, map_location='cpu', weights_only=False)
    config = state['args']
    if (config.get('experiment') != 'R88' or config.get('r88_schema_version') != 1
            or config.get('arm') not in ARMS or config.get('seed') != 2
            or config.get('initialization') != 'scratch_shared_r77'
            or config.get('solver_names') != GLOBAL_SOLVERS
            or config.get('max_successful_updates') != MAX_UPDATES
            or state.get('state_hash') != json_hash(config) or 'objective_state' in state):
        raise ValueError('Checkpoint is not an unchanged fresh R88 input-role model')
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
    result.add_argument('--gpu-type', choices=('3090',), default='3090')
    result.add_argument('--large-micro-cap', type=int, default=352)
    result.add_argument('--eval-batch-size', type=int, default=256)
    result.add_argument('--log-every', type=int, default=18)
    result.add_argument('--wandb-mode', choices=('auto', 'online', 'offline'), default='auto')
    result.add_argument('--resume', action='store_true')
    result.add_argument('--state-hash', '--expected-state-hash')
    result.set_defaults(experiment='R88')
    return result


def describe(args):
    return dict(experiment='R88', arm=args.arm, rows=SPLIT_ROWS, seed=2,
        batch_size=BATCH_SIZE, max_updates=MAX_UPDATES, warmup_updates=WARMUP_UPDATES,
        eval_every=EVAL_EVERY, model_factory=MODEL_FACTORY, new_parameter_count=0,
        role_slot=ROLE_SLOT, objective='unchanged full18 R77 CE/comparison/risk/R-Drop',
        inference='original score MLP and maximin; TSP input-derived role only',
        data_root=str(args.data_root), prepared_root=str(args.prepared_root),
        microbatch_caps=micro_caps(args.gpu_type, args.large_micro_cap),
        training_requires='main-locked R88 protocol; launcher waits for the R85A/B deliveries')


def main(argv=None):
    return engine.main(argv, integration=sys.modules[__name__])


if __name__ == '__main__':
    main()
