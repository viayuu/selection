"""R91 full18 probability-link control on the identical R88 execution-role model."""

import argparse
import sys
from pathlib import Path

import torch

from code.unified_selector.registry import GLOBAL_SOLVERS, PROBLEMS
from . import r84_experiment as engine
from .r76_experiment import PART_KEYS, file_hash, read_json
from .r77_data import parameter_hash
from .r84_prepare import DEFAULT_DATA_ROOT, DEFAULT_PREPARED_ROOT, SPLIT_ROWS
from .r85_experiment import ensure_test_cache, evaluate, selection_counts, zero_summary
from .r91_objective import ProbabilityLinkObjective, probability_links


V4 = Path(__file__).resolve().parent
DEFAULT_ROOT = V4 / 'runs/R91_probability_link_control'
ARMS = ('A_utility_ce', 'B_utility_probabilities')
MODEL_ROLE_ARM, ROLE_SLOT = 'B_execution_anchor', 3
SELECTION_HEAD = 'original_score_mlp_maximin'
BATCH_SIZE, EVAL_EVERY = engine.BATCH_SIZE, engine.EVAL_EVERY
WARMUP_UPDATES, MAX_UPDATES = engine.WARMUP_UPDATES, engine.MAX_UPDATES
PEAK_LR, END_LR = engine.PEAK_LR, engine.END_LR
WINNER_RULE, MODEL_FACTORY = engine.WINNER_RULE, 'code.V4.r91_experiment.load_checkpoint'
learning_rate, consistency_weight = engine.learning_rate, engine.consistency_weight
micro_caps, task_order, json_hash = engine.micro_caps, engine.task_order, engine.json_hash
prepared_config, status_update = engine.prepared_config, engine.status_update


def make_model(params, arm, state_dict=None):
    if arm not in ARMS:
        raise ValueError('Unknown R91 probability-link arm')
    if any(name.startswith('residual_') for name in params):
        raise ValueError('R91 cannot add or reconstruct residual parameters')
    from .r88_model import make_model as factory
    return factory(params, MODEL_ROLE_ARM, state_dict=state_dict)


def initialize_shared(initial, geometry, arm, device='cpu'):
    model, params, receipt = engine.initialize_shared(initial, geometry, arm, device,
        model_factory=make_model)
    actual = dict(model.named_parameters())
    if (set(actual) != set(initial['trainable'])
            or parameter_hash(actual) != initial['trainable_sha256']
            or params.get('role_arm') != MODEL_ROLE_ARM):
        raise ValueError('Both R91 arms require every identical fresh R84 parameter and the fixed execution role')
    count = sum(value.numel() for value in actual.values())
    receipt.update(common_parameter_count=count, trainable_parameter_count=count,
        new_parameter_count=0, objective_arm=arm, role_arm=MODEL_ROLE_ARM,
        model_role_arm=MODEL_ROLE_ARM, role_slot=ROLE_SLOT,
        geometry_fitted_on='frozen R84 new81000 TRAIN only')
    return model, params, receipt


def make_objective(selector, arm):
    if arm not in ARMS:
        raise ValueError('Unknown R91 probability-link arm')
    return ProbabilityLinkObjective(selector, arm)


def checkpoint_extras(objective):
    if type(objective) is not ProbabilityLinkObjective or objective.arm not in ARMS:
        raise TypeError('R91 requires the stateless ProbabilityLinkObjective')
    return {}


def restore_objective(objective, checkpoint):
    checkpoint_extras(objective)
    if ('objective_state' in checkpoint
            or checkpoint.get('args', {}).get('arm', objective.arm) != objective.arm):
        raise ValueError('Mutable or other-arm objective state cannot be resumed into R91')


def update_batch(objective, optimizer, scaler, batch, weight, runtime, cap,
                 max_retries=20, *, problem):
    if problem not in PROBLEMS:
        raise ValueError('R91 requires a registered single-task logical batch')
    parts, retries, _ = engine.update_batch(objective, optimizer, scaler, batch,
        weight, runtime, cap, max_retries=max_retries)
    return parts, retries, {}


def protocol_lock(args):
    path = args.protocol_lock or args.root / 'protocol_lock.json'
    record = read_json(path)
    expected = dict(experiment='R91', locked=True, arms=list(ARMS), seed=2,
        batch_size=BATCH_SIZE, max_updates=MAX_UPDATES, warmup_updates=WARMUP_UPDATES,
        eval_every=EVAL_EVERY)
    if any(record.get(name) != value for name, value in expected.items()):
        raise ValueError("R91 requires main's unchanged controlled probability-link protocol lock")
    fixed = dict(peak_lr=PEAK_LR, end_lr=END_LR, weight_decay=1e-4,
        dropout=.1, grad_clip=1., large_micro_cap=352, eval_batch_size=256,
        role_slot=ROLE_SLOT, role_feature_slot=ROLE_SLOT, role_arm=MODEL_ROLE_ARM,
        model_role_arm=MODEL_ROLE_ARM, new_parameter_count=0, new_trainable_parameters=0,
        gpu_type='3090', selection_head=SELECTION_HEAD,
        optimizer='actual AdamW lr1e-4 weight_decay1e-4; original successful-update warmup then cosine to1e-6')
    if any(name in record and record[name] != value for name, value in fixed.items()):
        raise ValueError('R91 lock disagrees with the fixed model/training controls')
    for name in ('data_root', 'prepared_root'):
        if name in record and Path(record[name]).resolve() != getattr(args, name).resolve():
            raise ValueError('R91 lock names a different ' + name)
    if record.get('rows_per_problem', SPLIT_ROWS) != SPLIT_ROWS:
        raise ValueError('R91 retains the frozen full18 NEW90k data split')
    for arm in ARMS:
        links = probability_links(arm)
        expected_links = dict(ce='legal-masked raw utility',
            risk='legal-masked raw utility probabilities' if links['risk'] == 'utility' else 'original maximin probabilities',
            rdrop='legal-masked raw utility symmetric KL' if links['rdrop'] == 'utility' else 'original maximin symmetric KL')
        if record.get(arm) != expected_links:
            raise ValueError('R91 lock changes the probability links for ' + arm)
    return dict(path=str(path.resolve()), sha256=file_hash(path), record=record)


def objective_spec(arm):
    return dict(classification='legal-masked utility CE', probability_links=probability_links(arm),
        coefficients=dict(ce=.35, comparison=.35, risk=.02), extra_loss=False,
        consistency='original R-Drop ramp to .35', mutable_objective_state=False,
        comparison='original pair margins and comparison loss; unchanged averaged-maximin Top3 focus',
        part_keys=list(PART_KEYS))


def protocol(args, ready, manifest, geometry, baselines, params, order, caps):
    if (args.gpu_type != '3090' or args.large_micro_cap != 352
            or args.eval_batch_size != 256 or caps['TSP'] != 352 or caps['CVRP'] != 352
            or params.get('role_arm') != MODEL_ROLE_ARM):
        raise ValueError('Both R91 arms require the fixed execution role and 3090 microcap352/eval256')
    config = engine.protocol(args, ready, manifest, geometry, baselines, params, order, caps)
    config.update(experiment='R91', r91_schema_version=1, model_factory=MODEL_FACTORY,
        main_protocol_lock=protocol_lock(args), role_arm=MODEL_ROLE_ARM,
        role_slot=ROLE_SLOT, new_parameter_count=0,
        train_ce_probability_link='utility', evaluation_ce_probability_link='maximin',
        objective=objective_spec(args.arm), objective_state='none: stateless ProbabilityLinkObjective',
        input_roles=dict(arm=MODEL_ROLE_ARM, problem='TSP', node_column=ROLE_SLOT,
            assignment='physical input node0', model_role_spec=params.get('role_spec'),
            label_dependent=False, other17_tasks='original inputs and mathematical forward',
            new_parameter_count=0, preprocessing=False),
        historical_reference=dict(model='R84A', same_frozen_new90k_split=True,
            run=str((engine.DEFAULT_ROOT / 'A_baseline').resolve()), retrained=False,
            evidence='main historical reference; not a newly matched original-loss control'),
        same_role_reference=dict(model='R88B', same_frozen_new90k_split=True,
            run=str((V4 / 'runs/R88_execution_role_inputs/B_execution_anchor').resolve()),
            role_arm=MODEL_ROLE_ARM, original_loss=True, retrained=False,
            evidence='historical same-role old-loss reference; not a newly matched original-loss control'),
        selection_head=SELECTION_HEAD,
        inference_rule='eval mode, fixed physical-node0 TSP slot3 role, original R77 score MLP/pair differences/maximin; stable ascending-global-solver-ID tie break',
        nss_reference='reuse exact4500 R84 predictions from locked R79 weights; no NSS inference/retraining',
        evidence_boundary='A/B controls probability links only; historical R88B/R84A are not newly matched original-loss controls',
        source_sha256={name: file_hash(V4 / name) for name in
            ('r91_experiment.py', 'r91_objective.py', 'r91_reporting.py', 'r88_model.py',
             'r84_experiment.py', 'r84_prepare.py', 'r84_reporting.py',
             'r85_experiment.py', 'r85_reporting.py', 'r87_reporting.py',
             'r80_reporting.py', 'r76_experiment.py', 'r77_experiment.py',
             'pairwise_selector.py', 'pairwise_objective.py', 'solver_code_encoder.py')})
    config['loss'].pop('focal_extra', None)
    return config


def validate_checkpoint_config(config):
    arm = config.get('arm')
    if (config.get('experiment') != 'R91' or config.get('r91_schema_version') != 1
            or arm not in ARMS or config.get('seed') != 2
            or config.get('initialization') != 'scratch_shared_r77'
            or config.get('solver_names') != GLOBAL_SOLVERS
            or config.get('max_successful_updates') != MAX_UPDATES
            or config.get('model_factory') != MODEL_FACTORY
            or config.get('role_arm') != MODEL_ROLE_ARM
            or config.get('model_params', {}).get('role_arm') != MODEL_ROLE_ARM
            or config.get('selection_head') != SELECTION_HEAD
            or config.get('objective') != objective_spec(arm)):
        raise ValueError('Checkpoint is not an unchanged fresh R91 fixed-role probability-link selector')


def load_checkpoint(path, device='cpu'):
    state = torch.load(path, map_location='cpu', weights_only=False)
    config = state['args']
    validate_checkpoint_config(config)
    if state.get('state_hash') != json_hash(config) or 'objective_state' in state:
        raise ValueError('R91 checkpoint protocol hash changed or mutable objective state was added')
    model = make_model(config['model_params'], config['arm'], state_dict=state['model'])
    return model.to(device), state


load_model = load_checkpoint
from_checkpoint = load_checkpoint


def authorize_test(args):
    lock, config = engine.authorize_test(args)
    validate_checkpoint_config(config)
    return lock, config


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
    result.set_defaults(experiment='R91')
    return result


def describe(args):
    return dict(experiment='R91', arm=args.arm, role_arm=MODEL_ROLE_ARM,
        model_role_arm=MODEL_ROLE_ARM, role_slot=ROLE_SLOT, new_parameter_count=0,
        rows=SPLIT_ROWS, seed=2, batch_size=BATCH_SIZE, max_updates=MAX_UPDATES,
        warmup_updates=WARMUP_UPDATES, eval_every=EVAL_EVERY, model_factory=MODEL_FACTORY,
        objective=objective_spec(args.arm), optimizer='AdamW',
        train_ce_probability_link='utility', evaluation_ce_probability_link='maximin',
        inference='identical R88 B_execution_anchor model; original score MLP/maximin in both arms',
        data_root=str(args.data_root), prepared_root=str(args.prepared_root),
        microbatch_caps=micro_caps(args.gpu_type, args.large_micro_cap),
        training_requires='main-locked R91 protocol; external launcher owns execution')


def main(argv=None):
    return engine.main(argv, integration=sys.modules[__name__])


if __name__ == '__main__':
    main()
