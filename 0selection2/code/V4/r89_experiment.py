"""R89 matched AdamW/Muon updates on one identical full18 execution-role model."""

import argparse
import sys
from pathlib import Path

import torch

from code.unified_selector.registry import GLOBAL_SOLVERS, PROBLEMS
from . import r84_experiment as shared_engine
from .r76_experiment import RDropObjective, file_hash, read_json
from .r77_data import parameter_hash
from .r84_prepare import DEFAULT_DATA_ROOT, DEFAULT_PREPARED_ROOT, SPLIT_ROWS
from .r85_experiment import ensure_test_cache, evaluate, selection_counts, zero_summary
from .r89_engine_adapter import PrivateEngineAdapter, TrainingHooks


V4 = Path(__file__).resolve().parent
DEFAULT_ROOT = V4 / 'runs/R89_optimizer_geometry'
ARMS = ('A_adamw', 'B_muon_hybrid')
MODEL_ROLE_ARM, ROLE_SLOT = 'B_execution_anchor', 3
SELECTION_HEAD = 'original_score_mlp_maximin'
BATCH_SIZE, EVAL_EVERY = shared_engine.BATCH_SIZE, shared_engine.EVAL_EVERY
WARMUP_UPDATES, MAX_UPDATES = shared_engine.WARMUP_UPDATES, shared_engine.MAX_UPDATES
PEAK_LR, END_LR = shared_engine.PEAK_LR, shared_engine.END_LR
WINNER_RULE, MODEL_FACTORY = shared_engine.WINNER_RULE, 'code.V4.r89_experiment.load_checkpoint'
learning_rate, consistency_weight = shared_engine.learning_rate, shared_engine.consistency_weight
micro_caps, task_order, json_hash = shared_engine.micro_caps, shared_engine.task_order, shared_engine.json_hash
prepared_config, status_update = shared_engine.prepared_config, shared_engine.status_update


def make_model(params, arm, state_dict=None):
    if arm not in ARMS:
        raise ValueError('Unknown R89 optimizer arm')
    if any(name.startswith('residual_') for name in params):
        raise ValueError('R89 does not add or load private scoring residuals')
    from .r88_model import make_model as factory
    return factory(params, MODEL_ROLE_ARM, state_dict=state_dict)


def initialize_shared(initial, geometry, arm, device='cpu'):
    model, params, receipt = shared_engine.initialize_shared(initial, geometry, arm, device,
        model_factory=make_model)
    actual = dict(model.named_parameters())
    if (set(actual) != set(initial['trainable'])
            or parameter_hash(actual) != initial['trainable_sha256']
            or params.get('role_arm') != MODEL_ROLE_ARM):
        raise ValueError('Both R89 arms must use identical original fresh parameters and the fixed execution role')
    receipt.update(common_parameter_count=sum(value.numel() for value in actual.values()),
        trainable_parameter_count=sum(value.numel() for value in actual.values()), new_parameter_count=0,
        optimizer_arm=arm, model_role_arm=MODEL_ROLE_ARM, role_slot=ROLE_SLOT,
        geometry_fitted_on='frozen R84 new81000 TRAIN only')
    return model, params, receipt


def make_objective(selector, arm):
    if arm not in ARMS:
        raise ValueError('Unknown R89 optimizer arm')
    return RDropObjective(selector)


def checkpoint_extras(objective):
    if type(objective) is not RDropObjective:
        raise TypeError('R89 retains the original stateless R77 objective')
    return {}


def restore_objective(objective, checkpoint):
    if type(objective) is not RDropObjective or 'objective_state' in checkpoint:
        raise ValueError('Mutable auxiliary objective state does not belong to R89')


def update_batch(objective, optimizer, scaler, batch, weight, runtime, cap,
                 max_retries=20, *, problem):
    if problem not in PROBLEMS:
        raise ValueError('R89 requires a complete registered-task logical batch')
    parts, retries, _ = shared_engine.update_batch(objective, optimizer, scaler, batch,
        weight, runtime, cap, max_retries=max_retries)
    return parts, retries, {}


def protocol_lock(args):
    path = args.protocol_lock or args.root / 'protocol_lock.json'
    record = read_json(path)
    expected = dict(experiment='R89', locked=True, arms=list(ARMS), seed=2,
        batch_size=BATCH_SIZE, max_updates=MAX_UPDATES, warmup_updates=WARMUP_UPDATES,
        eval_every=EVAL_EVERY)
    if any(record.get(name) != value for name, value in expected.items()):
        raise ValueError('R89 requires main\'s fixed two-optimizer protocol lock')
    fixed = dict(peak_lr=PEAK_LR, end_lr=END_LR, weight_decay=1e-4, dropout=.1,
        grad_clip=1., large_micro_cap=352, eval_batch_size=256, role_feature_slot=ROLE_SLOT,
        new_trainable_parameters=0, model_role_arm=MODEL_ROLE_ARM, gpu_type='4090')
    if any(name in record and record[name] != value for name, value in fixed.items()):
        raise ValueError('R89 lock disagrees with the fixed model/training controls')
    muon = record.get('B_muon_hybrid', {})
    expected_muon = dict(momentum=.95, nesterov=True, newton_schulz_steps=5,
        lr_multiplier=1., lr=PEAK_LR, weight_decay=1e-4, keller_aspect_scale_also_applied=False)
    if any(name in muon and muon[name] != value for name, value in expected_muon.items()):
        raise ValueError('R89 Muon settings changed from the same-LR Moonshot protocol')
    for name in ('data_root', 'prepared_root'):
        if name in record and Path(record[name]).resolve() != getattr(args, name).resolve():
            raise ValueError('R89 lock names a different ' + name)
    if record.get('rows_per_problem', SPLIT_ROWS) != SPLIT_ROWS:
        raise ValueError('R89 must retain the frozen full18 NEW90k rows')
    return dict(path=str(path.resolve()), sha256=file_hash(path), record=record)


def protocol(args, ready, manifest, geometry, baselines, params, order, caps, *, optimizer_receipt=None):
    if (args.gpu_type != '4090' or args.eval_batch_size != 256
            or caps['TSP'] != 352 or caps['CVRP'] != 352):
        raise ValueError('Both R89 arms use the prelocked4090 microcap352/eval256 execution')
    if optimizer_receipt is None or optimizer_receipt.get('arm') != args.arm:
        raise ValueError('R89 protocol must describe an actually constructed optimizer, not a placeholder')
    config = shared_engine.protocol(args, ready, manifest, geometry, baselines, params, order, caps)
    config.update(experiment='R89', r89_schema_version=1, model_factory=MODEL_FACTORY,
        main_protocol_lock=protocol_lock(args), optimizer=optimizer_receipt['optimizer_class'],
        optimizer_configuration=optimizer_receipt,
        optimizer_schedule='every parameter group receives the same successful-update LR; no multipliers',
        optimizer_runtime_metrics='read actual whole-step application/direction statistics only after optimizer.step',
        objective=dict(classification='original native CE', extra_loss=False,
            coefficients=dict(ce=.35, comparison=.35, risk=.02),
            consistency='original R-Drop ramp to .35', mutable_objective_state=False),
        input_roles=dict(arm=MODEL_ROLE_ARM, problem='TSP', node_column=ROLE_SLOT,
            model_role_spec=params.get('role_spec'), label_dependent=False, new_parameter_count=0,
            reason='known execution protocol; not selected from R88 interim or TEST results'),
        selection_head=SELECTION_HEAD,
        objective_state='none; optimizer native state and provenance saved atomically with the model',
        inference_rule='eval mode, fixed physical-node0 TSP slot3 role, original R77 score MLP/pair differences/maximin; stable ascending-global-solver-ID tie break',
        historical_reference=dict(model='R84A', same_frozen_new90k_split=True,
            run=str((shared_engine.DEFAULT_ROOT / 'A_baseline').resolve()), retrained=False),
        nss_reference='reuse exact4500 R84 predictions from locked R79 weights; no NSS inference/retraining',
        source_sha256={name: file_hash(V4 / name) for name in
            ('r89_experiment.py', 'r89_engine_adapter.py', 'r89_optim.py', 'r89_reporting.py',
             'r88_model.py', 'r88_reporting.py', 'r87_experiment.py', 'r87_reporting.py',
             'r84_experiment.py', 'r84_prepare.py', 'r84_reporting.py',
             'r85_experiment.py', 'r85_reporting.py', 'r80_reporting.py',
             'r76_experiment.py', 'r77_experiment.py', 'pairwise_selector.py',
             'pairwise_objective.py', 'solver_code_encoder.py')})
    config['loss'].pop('focal_extra', None)
    return config


def load_checkpoint(path, device='cpu'):
    state = torch.load(path, map_location='cpu', weights_only=False)
    config = state['args']
    if (config.get('experiment') != 'R89' or config.get('r89_schema_version') != 1
            or config.get('arm') not in ARMS or config.get('seed') != 2
            or config.get('initialization') != 'scratch_shared_r77'
            or config.get('solver_names') != GLOBAL_SOLVERS
            or config.get('max_successful_updates') != MAX_UPDATES
            or config.get('model_params', {}).get('role_arm') != MODEL_ROLE_ARM
            or state.get('state_hash') != json_hash(config) or 'objective_state' in state):
        raise ValueError('Checkpoint is not an unchanged fresh R89 fixed-role selector')
    model = make_model(config['model_params'], config['arm'], state_dict=state['model'])
    return model.to(device), state


load_model = load_checkpoint
from_checkpoint = load_checkpoint
authorize_test = shared_engine.authorize_test


def train(args, device):
    protocol_lock(args)
    adapter = PrivateEngineAdapter(args.arm, args.root / args.arm)
    hooks = TrainingHooks(sys.modules[__name__], adapter)
    return adapter.engine.train(args, device, integration=hooks)


def test(args, device):
    return shared_engine.test(args, device, integration=sys.modules[__name__])


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
    result.set_defaults(experiment='R89')
    return result


def describe(args):
    return dict(experiment='R89', arm=args.arm, model_role_arm=MODEL_ROLE_ARM,
        role_slot=ROLE_SLOT, new_parameter_count=0, rows=SPLIT_ROWS, seed=2,
        batch_size=BATCH_SIZE, max_updates=MAX_UPDATES, warmup_updates=WARMUP_UPDATES,
        eval_every=EVAL_EVERY, model_factory=MODEL_FACTORY,
        objective='unchanged original full18 R77 CE/comparison/risk/R-Drop',
        optimizer='actual Torch AdamW' if args.arm == ARMS[0] else 'Muon hidden Linear matrices plus AdamW auxiliary parameters',
        engine='private immutable R84 module with local optimizer-constructor facade',
        inference='same fixed execution-role model and original score MLP/maximin in both arms',
        microbatch_caps=micro_caps(args.gpu_type, args.large_micro_cap),
        training_requires='main-locked R89 protocol and launcher-assigned4090')


def main(argv=None):
    return shared_engine.main(argv, integration=sys.modules[__name__])


if __name__ == '__main__':
    main()
