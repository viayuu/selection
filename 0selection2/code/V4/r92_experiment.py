"""R92: full18 hierarchical graph selectors and independent GPU tree selectors."""

import argparse
import copy
import importlib.util
import json
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch

from code.unified_selector.registry import GLOBAL_SOLVERS, PROBLEMS
from . import r84_experiment as engine
from .r76_experiment import file_hash, read_json, write_json
from .r77_data import parameter_hash
from .r84_prepare import DEFAULT_DATA_ROOT, DEFAULT_PREPARED_ROOT, SPLIT_ROWS
from .r85_experiment import ensure_test_cache, selection_counts, zero_summary
from .r92_model import ARCHITECTURE, make_model
from .r92_objective import HierarchyObjective, update_batch


V4 = Path(__file__).resolve().parent
DEFAULT_ROOT = V4 / 'runs/R92_broad_learning'
ARMS, NEURAL_ARMS, TREE_ARMS = ('A', 'B', 'C', 'D'), ('A', 'B'), ('C', 'D')
BATCH_SIZE, MAX_UPDATES = engine.BATCH_SIZE, engine.MAX_UPDATES
WARMUP_UPDATES, EVAL_EVERY = engine.WARMUP_UPDATES, engine.EVAL_EVERY
TREE_ITERATIONS, TREE_EVAL_EVERY = 3000, 100
BUDGETS = dict(A=MAX_UPDATES, B=MAX_UPDATES, C=TREE_ITERATIONS, D=TREE_ITERATIONS)
MODEL_FACTORY, WINNER_RULE = 'code.V4.r92_experiment.load_checkpoint', engine.WINNER_RULE
SELECTION_HEAD = 'original_score_mlp_maximin'
prepared_config, micro_caps = engine.prepared_config, engine.micro_caps
json_hash, task_order = engine.json_hash, engine.task_order


def initialize_shared(initial, geometry, arm, device='cpu'):
    if (arm not in NEURAL_ARMS or initial.get('seed') != 2
            or initial.get('initialization') != 'scratch'
            or parameter_hash(initial['trainable']) != initial.get('trainable_sha256')):
        raise ValueError('Use only the frozen fresh R84 seed2 initialization, not a trained checkpoint')
    # Build a common superset once per construction. Assignment-only parameters
    # exist only in B; every shared new backbone parameter has the same value.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(920002)
        template = make_model(initial['model_params'], 'B')
        fresh = {name: value.detach().clone() for name, value in template.named_parameters()}
        del template
        model = make_model(initial['model_params'], arm)
    actual, reference = dict(model.named_parameters()), initial['trainable']
    retained = {name: value for name, value in actual.items()
                if name in reference and value.shape == reference[name].shape}
    with torch.no_grad():
        for name, value in actual.items():
            if name not in fresh or fresh[name].shape != value.shape:
                raise ValueError('New R92 parameter absent from the common fresh architecture template: ' + name)
            value.copy_(reference[name] if name in retained else fresh[name])
        branch = model.instance_encoder.geometry_residual
        branch.mean.copy_(torch.as_tensor(geometry['mean'], dtype=branch.mean.dtype))
        branch.std.copy_(torch.as_tensor(geometry['std'], dtype=branch.std.dtype))
    shared_new = {name: value for name, value in actual.items()
                  if name not in retained and not name.startswith('instance_encoder.assignment_nets.')}
    receipt = dict(seed=2, initialization='scratch_shared_retained_plus_new_hierarchy',
        historical_checkpoint_loaded=False, r77_fresh_trainable_sha256=initial['trainable_sha256'],
        common_parameter_names=sorted(retained), common_parameter_sha256=parameter_hash(retained),
        new_shared_parameter_sha256=parameter_hash(shared_new),
        trainable_sha256=parameter_hash(actual),
        retained_parameter_count=sum(value.numel() for value in retained.values()),
        trainable_parameter_count=sum(value.numel() for value in actual.values()),
        new_parameter_names=sorted(set(actual) - set(retained)),
        removed_parameter_names=sorted(set(reference) - set(retained)),
        new_common_seed=920002, geometry_fitted_on='frozen R84 new81000 TRAIN only')
    return model.to(device), copy.deepcopy(model.params), receipt


def protocol_lock(args):
    path = args.protocol_lock or args.root / 'protocol_lock.json'
    record = read_json(path)
    required = dict(experiment='R92', locked=True, arms=list(ARMS), seed=2,
        logical_global_batch=1024, task_scope='full18', rows_per_problem=SPLIT_ROWS)
    if any(record.get(name) != value for name, value in required.items()):
        raise ValueError('R92 requires the final full18 broad-learning protocol lock')
    if (record.get('arm_budgets') != dict(A=MAX_UPDATES, B=MAX_UPDATES,
                                        C=TREE_ITERATIONS, D=TREE_ITERATIONS)
            or record.get('warmup_updates') != WARMUP_UPDATES
            or record.get('eval_every') != EVAL_EVERY
            or record.get('tree_eval_every') != TREE_EVAL_EVERY):
        raise ValueError('R92 neural and tree budgets differ from the locked protocol')
    for name in ('data_root', 'prepared_root'):
        if Path(record.get(name, '')).resolve() != getattr(args, name).resolve():
            raise ValueError('R92 uses a different locked ' + name)
    return dict(path=str(path.resolve()), sha256=file_hash(path), record=record)


def objective_spec(arm):
    if arm in NEURAL_ARMS:
        return dict(main='original R77 CE/comparison/risk plus original symmetric R-Drop',
            coefficients=dict(ce=.35, comparison=.35, risk=.02),
            rdrop='linear ramp to .35 over4320 updates',
            hierarchy_auxiliary=dict(link=.1 if arm == 'B' else 0.,
                entropy=.01 if arm == 'B' else 0.,
                target='input relations and assignment entropy only; no routes or performance labels'),
            cross_instance_objective=False, mutable_objective_state=False)
    if arm in TREE_ARMS:
        return dict(main='MultiClass native winner' if arm == 'C' else 'MultiRMSE complete legal-pool relative regret',
            class_rebalancing=False, performance_input=False, neural_loss=False)
    raise ValueError('Unknown R92 arm')


def _base_config(args, ready, manifest, geometry, baselines, params, order, caps):
    config = engine.protocol(args, ready, manifest, geometry, baselines, params, order, caps)
    config.update(experiment='R92', r92_schema_version=1, model_factory=MODEL_FACTORY,
        main_protocol_lock=protocol_lock(args), objective=objective_spec(args.arm),
        architecture_family='hierarchical_graph' if args.arm in NEURAL_ARMS else 'independent_tree_bank',
        historical_reference=dict(model='R89B', retrained=False,
            run=str((V4 / 'runs/R89_optimizer_geometry/B_muon_hybrid').resolve()),
            evidence='historical same-data reference, not equal-parameter/equal-compute control'),
        source_sha256={name: file_hash(V4 / name) for name in
            ('r92_experiment.py', 'r92_objective.py', 'r92_model.py', 'r92_tree.py',
             'r92_tree_features.py', 'r92_reporting.py', 'r84_experiment.py',
             'r84_prepare.py', 'r88_model.py', 'pairwise_selector.py', 'pairwise_objective.py')})
    config['loss'].pop('focal_extra', None)
    return config


def protocol(args, ready, manifest, geometry, baselines, params, order, caps):
    if (args.arm not in NEURAL_ARMS or args.large_micro_cap != 352
            or args.eval_batch_size != 256):
        raise ValueError('R92 neural arms retain logical1024, micro352 and eval256')
    config = _base_config(args, ready, manifest, geometry, baselines, params, order, caps)
    config.update(initialization='scratch_shared_retained_plus_new_hierarchy',
        training_unit='successful_neural_updates', hierarchy=copy.deepcopy(ARCHITECTURE),
        input_roles=dict(problem='TSP', node_column=3, assignment='physical input node0', label_dependent=False),
        inference_rule='eval mode; input-only32/8 graph hierarchy; original score MLP/pair differences/maximin; stable ascending-global-solver-ID tie break',
        selection_head=SELECTION_HEAD,
        gradients='per-graph means weighted by actual micro rows / logical1024; both dropout forwards; one clip/step',
        evidence_boundary='A hard versus B learned graph coarsening; B includes fixed input-only link/entropy regularization. Not an exact DiffPool reproduction.')
    return config


def tree_protocol(args):
    """Common data/metric metadata, not a claim of matched neural optimization."""
    if args.arm not in TREE_ARMS:
        raise ValueError('Only C/D use tree_protocol')
    ready, manifest, geometry, baselines = prepared_config(args)
    initial = torch.load(args.prepared_root / 'identicalinit_seed2.pt', map_location='cpu', weights_only=False)
    config = _base_config(args, ready, manifest, geometry, baselines,
        initial['model_params'], [], micro_caps(args.gpu_type, args.large_micro_cap))
    config.update(initialization='scratch_tree_seed2', training_unit='trees_per_task',
        common_trainable_sha256=None,
        max_successful_updates=TREE_ITERATIONS, trees_per_task=TREE_ITERATIONS,
        updates_per_problem=None, total_sample_exposures=None,
        global_batch_size=None, logical_global_batch=None, warmup_updates=0,
        rdrop_ramp_updates=0, rdrop_weight=0., dropout=None, grad_clip=None,
        optimizer='CatBoost Plain GPU boosting, not SGD', weight_decay=None,
        eval_every_updates=TREE_EVAL_EVERY, loss=objective_spec(args.arm),
        lr=dict(rule='constant boosting learning rate', value=.03),
        model_params=dict(tree_arm=args.arm), selection_head='native multiclass' if args.arm == 'C' else 'minimum predicted relative regret',
        inference_rule='one global VAL-selected tree count for the complete18 bank; native legal pool; stable ascending-global-solver-ID ties',
        train_eval='evaluate TRAIN/VAL at common100-tree checkpoints; full selected TRAIN/TEST',
        source_plan='full TRAIN table per task for boosting; no shuffled SGD minibatches or neural initialization',
        checkpoint_rule='strict minimum full VAL macro_vs_sbs_pct over a single common tree count for all18 tasks; first wins ties',
        gradients='not applicable: independent GPU tree fitting, no neural minibatches',
        tree_settings=dict(iterations=3000, depth=8, learning_rate=.03, l2_leaf_reg=3.,
            border_count=128, random_seed=2, boosting_type='Plain', task_type='GPU',
            devices='0', thread_count=2, gpu_ram_part=.85),
        evidence_boundary='Independent input-derived tree selector family. Eighteen task-specific banks share one global VAL-selected tree count, not per-task best assembly. Compute budget is not matched to neural arms.')
    return config


def make_objective(selector, arm):
    return HierarchyObjective(selector, arm)


def checkpoint_extras(objective):
    if type(objective) is not HierarchyObjective:
        raise TypeError('Unexpected mutable R92 objective')
    return {}


def restore_objective(objective, checkpoint):
    checkpoint_extras(objective)
    if 'objective_state' in checkpoint:
        raise ValueError('R92 hierarchy objectives have no mutable training state')


def validate_checkpoint_config(config):
    arm = config.get('arm')
    if (config.get('experiment') != 'R92' or config.get('r92_schema_version') != 1
            or arm not in ARMS or config.get('seed') != 2
            or config.get('max_successful_updates') != BUDGETS[arm]
            or config.get('solver_names') != GLOBAL_SOLVERS
            or config.get('model_factory') != MODEL_FACTORY):
        raise ValueError('Not a locked fresh R92 checkpoint')
    if arm in NEURAL_ARMS and (config.get('objective') != objective_spec(arm)
            or config.get('hierarchy') != ARCHITECTURE):
        raise ValueError('Neural architecture/objective changed after the R92 lock')


def load_checkpoint(path, device='cpu'):
    state = torch.load(path, map_location='cpu', weights_only=False)
    config = state['args']
    validate_checkpoint_config(config)
    if state.get('state_hash') != json_hash(config):
        raise ValueError('R92 checkpoint protocol hash changed')
    if config['arm'] in TREE_ARMS:
        from .r92_tree import load_checkpoint as load_tree
        return load_tree(path, device)
    model = make_model(config['model_params'], config['arm'], state_dict=state['model'])
    return model.to(device), state


load_model = load_checkpoint
from_checkpoint = load_checkpoint


@lru_cache(maxsize=2)
def evaluation_engine(arm):
    if arm in NEURAL_ARMS:
        return engine
    name = 'code.V4._r92_tree_evaluation_' + arm
    spec = importlib.util.spec_from_file_location(name, engine.__file__)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    module.MAX_UPDATES, module.BATCH_SIZE = TREE_ITERATIONS, 0
    return module


def evaluate(selector, args, manifest, baselines, split, tag, step, state_hash, checkpoint_sha256=None):
    result = evaluation_engine(args.arm).evaluate(selector, args, manifest, baselines,
        split, tag, step, state_hash, checkpoint_sha256)
    if 'zero_selection' not in result:
        rows = []
        with np.load(result['prediction_archive'], allow_pickle=False) as archive:
            for problem in PROBLEMS:
                rows.extend(selection_counts(archive, problem + '__', problem, split))
        result['zero_selection'] = zero_summary(rows)
        directory = args.root / args.arm / 'evaluation' / split / tag
        write_json(directory / 'method_selection.json', dict(methods=rows, summary=result['zero_selection']))
        if args.arm in TREE_ARMS:
            result.update(training_unit='trees_per_task', sample_exposures=None)
        write_json(directory / 'metrics.json', result)
    return result


def authorize_test(args):
    root = args.root / args.arm
    complete, lock, config = (read_json(root / name) for name in
        ('training_complete.json', 'final_test_lock.json', 'protocol.json'))
    validate_checkpoint_config(config)
    if (complete.get('successful_updates') != BUDGETS[args.arm]
            or not lock.get('selected_without_test') or lock.get('per_problem_assembly') is not False
            or lock.get('experiment') != 'R92' or lock.get('arm') != args.arm
            or lock.get('state_hash') != json_hash(config)
            or Path(lock['checkpoint']).resolve() != (root / 'best.pt').resolve()
            or lock['checkpoint_sha256'] != file_hash(root / 'best.pt')
            or Path(config['data_root']).resolve() != args.data_root.resolve()):
        raise PermissionError('TEST needs the complete R92 budget and one immutable global VAL lock')
    if args.state_hash is not None and args.state_hash != lock['state_hash']:
        raise ValueError('Expected protocol hash differs from the unique lock')
    return lock, config


def train(args, device):
    protocol_lock(args)
    if args.arm in TREE_ARMS:
        from .r92_tree import train as train_tree
        return train_tree(args, device)
    return engine.train(args, device, integration=sys.modules[__name__])


def test(args, device):
    if args.arm in TREE_ARMS:
        from .r92_tree import test as test_tree
        return test_tree(args, device)
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
    result.add_argument('--gpu-type', choices=('3090', '4090'), default='3090')
    result.add_argument('--large-micro-cap', type=int, default=352)
    result.add_argument('--eval-batch-size', type=int, default=256)
    result.add_argument('--log-every', type=int, default=18)
    result.add_argument('--wandb-mode', choices=('auto', 'online', 'offline'), default='offline')
    result.add_argument('--resume', action='store_true')
    result.add_argument('--state-hash', '--expected-state-hash')
    result.set_defaults(experiment='R92')
    return result


def describe(args):
    return dict(experiment='R92', arm=args.arm, seed=2, rows=SPLIT_ROWS,
        training_unit='successful_neural_updates' if args.arm in NEURAL_ARMS else 'trees_per_task',
        max_updates=BUDGETS[args.arm], logical_batch=1024 if args.arm in NEURAL_ARMS else None,
        objective=objective_spec(args.arm), model_factory=MODEL_FACTORY,
        data_root=str(args.data_root), prepared_root=str(args.prepared_root))


def main(argv=None):
    return engine.main(argv, integration=sys.modules[__name__])


if __name__ == '__main__':
    main()
