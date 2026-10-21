"""R93 full18 plain-classification engineering control, differing only in dropout."""

import argparse
import copy
import importlib.util
import sys
from functools import lru_cache
from pathlib import Path

import torch

from code.unified_selector.registry import GLOBAL_SOLVERS, PROBLEMS
from . import r84_experiment as engine
from .r76_experiment import PART_KEYS, file_hash, read_json, write_json
from .r77_data import parameter_hash
from .r84_prepare import DEFAULT_DATA_ROOT, DEFAULT_PREPARED_ROOT, SPLIT_ROWS
from .r85_experiment import ensure_test_cache, evaluate as evaluate_original
from .r85_experiment import selection_counts, zero_summary
from .r93_model import ARMS, DROPOUT, MODEL_ROLE_ARM, ROLE_SLOT, SELECTION_HEAD, SPEC, make_model
from .r93_objective import PlainClassificationObjective


V4 = Path(__file__).resolve().parent
DEFAULT_ROOT = V4 / 'runs/R93_plain_classification'
NEURAL_ARMS = ARMS
BATCH_SIZE, MAX_UPDATES = engine.BATCH_SIZE, engine.MAX_UPDATES
WARMUP_UPDATES, EVAL_EVERY = engine.WARMUP_UPDATES, engine.EVAL_EVERY
PEAK_LR, END_LR = engine.PEAK_LR, engine.END_LR
BUDGETS = dict.fromkeys(ARMS, MAX_UPDATES)
MODEL_FACTORY, WINNER_RULE = 'code.V4.r93_experiment.load_checkpoint', engine.WINNER_RULE
prepared_config, micro_caps = engine.prepared_config, engine.micro_caps
json_hash, task_order, learning_rate = engine.json_hash, engine.task_order, engine.learning_rate
status_update = engine.status_update
INFERENCE_RULE = ('eval mode; original score MLP, pair differences and maximin; '
                  'stable ascending-global-solver-ID tie break; no bias or temperature')


def consistency_weight(update):
    return 0.


def initialize_shared(initial, geometry, arm, device='cpu'):
    if arm not in ARMS:
        raise ValueError('Unknown R93 arm')
    model, params, receipt = engine.initialize_shared(initial, geometry, arm, device,
                                                     model_factory=make_model)
    actual = dict(model.named_parameters())
    if (set(actual) != set(initial['trainable'])
            or parameter_hash(actual) != initial['trainable_sha256']
            or params.get('dropout') != DROPOUT[arm]
            or params.get('role_arm') != MODEL_ROLE_ARM):
        raise ValueError('R93 requires every original fresh R84 parameter and the fixed execution role')
    receipt.update(trainable_parameter_count=sum(value.numel() for value in actual.values()),
        new_parameter_count=0, model_role_arm=MODEL_ROLE_ARM, role_slot=ROLE_SLOT,
        dropout=DROPOUT[arm], geometry_fitted_on='frozen R84 new81000 TRAIN only')
    return model, params, receipt


def protocol_lock(args):
    path = args.protocol_lock or args.root / 'protocol_lock.json'
    record = read_json(path)
    required = dict(experiment='R93', locked=True, arms=list(ARMS), seed=2,
        task_scope='full18', rows_per_problem=SPLIT_ROWS, batch_size=BATCH_SIZE,
        logical_global_batch=BATCH_SIZE, max_updates=MAX_UPDATES,
        warmup_updates=WARMUP_UPDATES, eval_every=EVAL_EVERY, dropout=DROPOUT,
        large_micro_cap=352, eval_batch_size=256)
    if any(record.get(name) != value for name, value in required.items()):
        raise ValueError('R93 requires the locked two-arm full18 pure-classification protocol')
    fixed = dict(arm_budgets=BUDGETS, optimizer='AdamW', peak_lr=PEAK_LR, end_lr=END_LR,
        weight_decay=1e-4, grad_clip=1., role_arm=MODEL_ROLE_ARM, role_slot=ROLE_SLOT,
        selection_head=SELECTION_HEAD)
    if any(name in record and record[name] != value for name, value in fixed.items()):
        raise ValueError('R93 fixed model, optimizer or budget controls changed')
    for name in ('data_root', 'prepared_root'):
        if Path(record.get(name, '')).resolve() != getattr(args, name).resolve():
            raise ValueError('R93 lock names a different ' + name)
    return dict(path=str(path.resolve()), sha256=file_hash(path), record=record)


def objective_spec(arm):
    if arm not in ARMS:
        raise ValueError('Unknown R93 arm')
    return dict(main='unweighted CE on legal-masked raw utility', target='saved native ind only',
        coefficients=dict(ce=1., comparison=0., risk=0., kl=0.), forward_passes=1,
        rdrop=False, extra_loss=False, class_rebalancing=False, logit_bias=False,
        temperature=False, cross_instance_objective=False, mutable_objective_state=False,
        training_ce='raw utility CE', evaluation_ce='original maximin CE diagnostic; not training loss')


def protocol(args, ready, manifest, geometry, baselines, params, order, caps):
    if (args.arm not in ARMS or args.large_micro_cap != 352 or args.eval_batch_size != 256
            or caps.get('TSP') != 352 or caps.get('CVRP') != 352
            or params.get('r93_arm') != args.arm or params.get('r93_spec') != SPEC
            or params.get('dropout') != DROPOUT[args.arm]):
        raise ValueError('R93 retains locked arm dropout, logical1024, micro352 and eval256')
    config = engine.protocol(args, ready, manifest, geometry, baselines, params, order, caps)
    config.update(experiment='R93', r93_schema_version=1, model_factory=MODEL_FACTORY,
        main_protocol_lock=protocol_lock(args), objective=objective_spec(args.arm),
        model_spec=copy.deepcopy(SPEC), dropout=DROPOUT[args.arm], rdrop_weight=0.,
        rdrop_ramp_updates=0, loss=dict(ce=1., comparison=0., risk=0., kl=0., dropout_passes=1),
        objective_state='none: stateless single-forward raw-utility CE',
        training_unit='successful_neural_updates', selection_head=SELECTION_HEAD,
        input_roles=dict(arm=MODEL_ROLE_ARM, problem='TSP', node_column=ROLE_SLOT,
            assignment='physical input node0', label_dependent=False, new_parameter_count=0),
        inference_rule=INFERENCE_RULE,
        gradients='actual micro rows / logical1024; one forward per micro; one clip/step; replay RNG on AMP retry',
        ce_diagnostics=dict(training='raw utility CE', evaluation='original maximin CE; diagnostic only'),
        historical_reference=dict(primary='R89B', secondary='R84A', retrained=False,
            matched_original_loss_baseline=False,
            primary_run=str((V4 / 'runs/R89_optimizer_geometry/B_muon_hybrid').resolve()),
            secondary_run=str((engine.DEFAULT_ROOT / 'A_baseline').resolve())),
        evidence_boundary='Broad engineering learning control, not novelty or causal loss diagnosis. A/B differ only in dropout. B retains weight decay. R89B/R84A are historical, not newly matched full-loss controls.',
        source_sha256={name: file_hash(V4 / name) for name in
            ('r93_experiment.py', 'r93_objective.py', 'r93_model.py', 'r84_experiment.py',
             'r84_prepare.py', 'r85_experiment.py', 'r88_model.py', 'pairwise_selector.py',
             'pairwise_objective.py', 'r76_experiment.py', 'solver_code_encoder.py')})
    return config


def make_objective(selector, arm):
    objective_spec(arm)
    return PlainClassificationObjective(selector)


def checkpoint_extras(objective):
    if type(objective) is not PlainClassificationObjective:
        raise TypeError('R93 requires the stateless pure CE objective')
    return {}


def restore_objective(objective, checkpoint):
    checkpoint_extras(objective)
    if 'objective_state' in checkpoint:
        raise ValueError('R93 cannot restore mutable or auxiliary objective state')


def update_batch(objective, optimizer, scaler, batch, weight, runtime, cap,
                 max_retries=20, *, problem):
    expected_cap = micro_caps('3090', 352).get(problem)
    if problem not in PROBLEMS or len(batch['ind']) != BATCH_SIZE or cap != expected_cap:
        raise ValueError('R93 requires logical1024 rows and the locked per-task microcap: TSP/CVRP352, other tasks1024')
    parts, retries, _ = engine.update_batch(objective, optimizer, scaler, batch, 0., runtime,
                                           cap, max_retries=max_retries)
    return parts, retries, dict(train_utility_ce=parts[PART_KEYS.index('ce')].detach())


def validate_checkpoint_config(config):
    arm = config.get('arm')
    if arm not in ARMS:
        raise ValueError('Unknown R93 checkpoint arm')
    required = dict(experiment='R93', r93_schema_version=1, seed=2,
        initialization='scratch_shared_r77', model_factory=MODEL_FACTORY,
        solver_names=list(GLOBAL_SOLVERS), rows_per_problem=SPLIT_ROWS,
        global_batch_size=BATCH_SIZE, logical_global_batch=BATCH_SIZE,
        max_successful_updates=MAX_UPDATES, warmup_updates=WARMUP_UPDATES,
        eval_every_updates=EVAL_EVERY, eval_batch_size=256, optimizer='AdamW',
        weight_decay=1e-4, grad_clip=1., dropout=DROPOUT[arm],
        rdrop_weight=0., rdrop_ramp_updates=0, objective=objective_spec(arm),
        loss=dict(ce=1., comparison=0., risk=0., kl=0., dropout_passes=1),
        model_spec=SPEC, selection_head=SELECTION_HEAD, inference_rule=INFERENCE_RULE,
        winner_rule=WINNER_RULE)
    if any(config.get(name) != value for name, value in required.items()):
        raise ValueError('R93 checkpoint model, objective or fixed training controls changed')
    params = config.get('model_params', {})
    if (params.get('r93_arm') != arm or params.get('r93_spec') != SPEC
            or params.get('dropout') != DROPOUT[arm]):
        raise ValueError('R93 checkpoint arm/dropout parameters changed')


def load_checkpoint(path, device='cpu'):
    state = torch.load(path, map_location='cpu', weights_only=False)
    config = state['args']
    validate_checkpoint_config(config)
    if state.get('state_hash') != json_hash(config):
        raise ValueError('R93 checkpoint protocol hash changed')
    if 'objective_state' in state:
        raise ValueError('R93 checkpoints have no mutable objective state')
    model = make_model(config['model_params'], config['arm'], state_dict=state['model'])
    return model.to(device), state


load_model = load_checkpoint
from_checkpoint = load_checkpoint


def evaluate(selector, args, manifest, baselines, split, tag, step, state_hash, checkpoint_sha256=None):
    result = evaluate_original(selector, args, manifest, baselines, split, tag, step,
                               state_hash, checkpoint_sha256)
    result['ce_diagnostic'] = 'original maximin CE; R93 training uses raw utility CE'
    write_json(args.root / args.arm / 'evaluation' / split / tag / 'metrics.json', result)
    return result


def authorize_test(args):
    lock, config = engine.authorize_test(args)
    validate_checkpoint_config(config)
    if lock.get('inference_rule') != config['inference_rule'] or lock.get('winner_rule') != WINNER_RULE:
        raise PermissionError('R93 TEST requires unchanged original inference and native winner labels')
    return lock, config


@lru_cache(maxsize=1)
def training_engine():
    """Zero the legacy logged R-Drop ramp without modifying shared R84 globals."""
    name = 'code.V4._r93_training_engine'
    spec = importlib.util.spec_from_file_location(name, engine.__file__)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    module.consistency_weight = consistency_weight
    return module


def train(args, device):
    protocol_lock(args)
    return training_engine().train(args, device, integration=sys.modules[__name__])


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
    result.add_argument('--gpu-type', choices=('3090', '4090'), default='3090')
    result.add_argument('--large-micro-cap', type=int, default=352)
    result.add_argument('--eval-batch-size', type=int, default=256)
    result.add_argument('--device', default='cuda:0')
    result.add_argument('--wandb-mode', choices=('auto', 'online', 'offline'), default='offline')
    result.add_argument('--log-every', type=int, default=18)
    result.add_argument('--resume', action='store_true')
    result.add_argument('--state-hash', '--expected-state-hash')
    result.set_defaults(experiment='R93')
    return result


def describe(args):
    return dict(experiment='R93', arm=args.arm, seed=2, rows=SPLIT_ROWS,
        training_unit='successful_neural_updates', max_updates=MAX_UPDATES,
        batch_size=BATCH_SIZE, logical_batch=BATCH_SIZE, warmup_updates=WARMUP_UPDATES,
        eval_every=EVAL_EVERY, dropout=DROPOUT[args.arm], objective=objective_spec(args.arm),
        model_factory=MODEL_FACTORY, selection_head=SELECTION_HEAD,
        data_root=str(args.data_root), prepared_root=str(args.prepared_root))


def main(argv=None):
    return engine.main(argv, integration=sys.modules[__name__])


if __name__ == '__main__':
    main()
