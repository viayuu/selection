"""Finite R117 whole18 sidecar; capacity, allocation and wall timeout belong to main.

TRAIN/VAL features are prepared once by train. TEST reads only an existing R84
source cache, and native TEST features require the immutable full-budget VAL lock.
Historical modules are loaded privately, never monkeypatched or edited.
"""

import argparse
import copy
import fcntl
from functools import lru_cache
import importlib
import inspect
import json
import math
from pathlib import Path
import sys
import time
import types

import torch

from code.unified_selector.registry import GLOBAL_SOLVERS, PROBLEMS
from . import r84_experiment as engine
from .r76_experiment import file_hash, read_json, utc, write_json
from .r77_data import parameter_hash
from .r84_prepare import DEFAULT_DATA_ROOT, DEFAULT_PREPARED_ROOT, SPLIT_ROWS
from .r117_objective import NativeObjective, LOSS_RECIPE, PART_KEYS, inputs_only, raw_inputs


V4 = Path(__file__).resolve().parent
DEFAULT_ROOT = V4 / 'runs/R117_native_computation'
DEFAULT_CACHE_ROOT = Path('/tmp/shiys_R84_new90k_cache')
DEFAULT_COMMON_ROOT = Path('/tmp/shiys_R117_common_base')
ARMS = ('A', 'B')
ARM_VARIANTS = dict(A='added_random', B='added_pretrained')
BATCH_SIZE, MAX_UPDATES, WARMUP_UPDATES = 1024, 28440, 4266
EVAL_EVERY, TRAIN_EVERY = 1422, 7110
PEAK_LR, END_LR = 1e-4, 1e-6
BUDGETS = dict.fromkeys(ARMS, MAX_UPDATES)
ORIGINAL_PARAMETERS, NEW_PARAMETERS, PARAMETER_COUNT = 1842953, 98304, 1941257
MODEL_FACTORY = 'code.V4.r117_experiment.load_checkpoint'
MODEL_ROLE_ARM = 'B_execution_anchor'
INITIALIZATION = 'fresh_r84_plus_frozen_native_computation'
CHECKPOINT_RULE = ('strict maximum full18 VAL macro_top1; exact Top1 ties: minimum '
                   'macro_actual_regret_pct; exact double ties: first checkpoint')
INFERENCE_RULE = ('eval frozen native computation; original R77 score-difference maximin; '
                  'stable ascending-global-solver-ID ties; no calibration or task stitching')
WINNER_RULE = engine.WINNER_RULE
VAL_GATE = dict(top1_gain_pp=.20, macro_regret_harm_pp=0., max_task_regret_harm_pp=.05,
                vs_sbs_must_not_worsen=True, evaluated_on='locked whole18 VAL only')
PROJECTION_SHAPES = {'instance_encoder.nco_projection.weight': (128, 128), **{
    'instance_encoder.native_projections.' + p + '.weight': (128, 256 if p == 'ATSP' else 128)
    for p in ('TSP', 'CVRP', 'VRP', 'ATSP')}}
SOURCE_FILES = ('r117_experiment.py', 'r117_objective.py', 'r117_model.py', 'r117_native.py',
    'r117_features.py', 'r117_reporting.py', 'r117_finish.py', 'test_r117_experiment.py',
    'r117_capacity.py', 'r117_launch.py', 'run_v4_r117.sh',
    'test_r117_native.py', 'test_r117_model.py', 'test_r117_features.py',
    'r112_experiment.py', 'r112_reporting.py', 'r112_finish.py', 'r116_mechanism_finish.py',
    'r113_finish.py', 'r84_experiment.py', 'r84_prepare.py', 'r84_reporting.py',
    'r80_experiment.py', 'r80_reporting.py', 'r77_experiment.py', 'r77_data.py',
    'r76_experiment.py', 'r76_cache.py', 'r75_cache.py', 'r88_model.py', 'r94_donor.py',
    'r94_model.py', 'r102_donor.py', 'solver_code_encoder.py', 'V4Model.py',
    'pairwise_selector.py', 'pairwise_objective.py', 'r42_experiment.py')
FILES = SOURCE_FILES
json_hash, task_order = engine.json_hash, engine.task_order


def private_module(source, name):
    path = Path(source.__file__ if hasattr(source, '__file__') else source)
    module = types.ModuleType('code.V4._r117_' + name)
    module.__file__, module.__package__ = str(path), 'code.V4'
    sys.modules[module.__name__] = module
    text = path.read_text(encoding='utf-8')
    for old, new in (('r112', 'r117'), ('R112', 'R117'), ('A_collapsed', 'A'),
                     ('B_interfaces', 'B'), ('R117_region_interfaces', 'R117_native_computation')):
        text = text.replace(old, new)
    exec(compile(text, str(path) + ':R117_private', 'exec'), module.__dict__)
    return module


def adapt_function(module, original, replacements):
    text = inspect.getsource(original)
    for old, new in replacements:
        if text.count(old) != 1:
            raise RuntimeError('Historical template drift: ' + old)
        text = text.replace(old, new)
    exec(compile(text, original.__code__.co_filename + ':R117_private', 'exec'), module.__dict__)


_base = private_module(V4 / 'r112_experiment.py', 'runner_callbacks')
selection_key = _base.selection_key
selected_capture_proof = _base.selected_capture_proof


def _models():
    module = importlib.import_module('code.V4.r117_model')
    if tuple(module.ARMS) != ARMS or module.INITIALIZATION != INITIALIZATION:
        raise ValueError('R117 model/runner contract differs')
    if module.PROJECTION_SHAPES != PROJECTION_SHAPES:
        raise ValueError('Only the five declared bias-free projections may be added')
    return module


def make_model(params, arm, state_dict=None, *, donor=None):
    return _models().make_model(params, arm, state_dict=state_dict, donor=donor)


def feature_root(args):
    value = getattr(args, 'feature_root', None)
    return Path(value) if value is not None else Path('/tmp/shiys_R117_' + args.arm + '_native_features')


def common_root(args):
    return Path(getattr(args, 'common_root', None) or DEFAULT_COMMON_ROOT)


def donor_record(donor, arm):
    value = donor.provenance() if callable(donor.provenance) else donor.provenance
    if (arm not in ARMS or not isinstance(value, dict) or value.get('arm') != arm
            or any(p.requires_grad for p in donor.parameters()) or donor.training):
        raise ValueError('R117 donor must expose arm-specific provenance and remain frozen/eval')
    from .r94_donor import CHECKPOINT_SHA256
    components = value.get('components', {})
    added_variant = 'pretrained' if arm == 'B' else 'random'
    if (value.get('variant') != ARM_VARIANTS[arm]
            or set(components) != {'base', 'TSP', 'CVRP', 'VRP', 'ATSP'}
            or components['base'].get('variant') != 'pretrained'
            or components['base'].get('checkpoint_sha256') != CHECKPOINT_SHA256
            or any(components[p].get('variant') != added_variant for p in ('TSP', 'CVRP', 'VRP', 'ATSP'))
            or value.get('external_pretraining_by_component') != dict(base=True,
                TSP=arm == 'B', CVRP=arm == 'B', VRP=arm == 'B', ATSP=arm == 'B')):
        raise ValueError('Both arms require the same pinned pretrained ReLD base; only native additions differ')
    return copy.deepcopy(value)


def initialize_shared(initial, geometry, arm, device='cpu'):
    reference = initial['trainable']
    if (arm not in ARMS or initial.get('seed') != 2 or initial.get('initialization') != 'scratch'
            or parameter_hash(reference) != initial.get('trainable_sha256')
            or sum(p.numel() for p in reference.values()) != ORIGINAL_PARAMETERS):
        raise ValueError('Use fresh R84 seed2 TRAIN-only preparation, never a trained selector')
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(2)
        model = make_model(initial['model_params'], arm)
    actual = {n: p for n, p in model.named_parameters() if p.requires_grad}
    retained = {n: p for n, p in actual.items() if n in reference and p.shape == reference[n].shape}
    added = {n: p for n, p in actual.items() if n not in retained}
    if (set(retained) != set(reference) or set(added) != set(PROJECTION_SHAPES)
            or any(tuple(added[n].shape) != s for n, s in PROJECTION_SHAPES.items())
            or sum(p.numel() for p in actual.values()) != PARAMETER_COUNT):
        raise ValueError('R117 must retain all R84 tensors and add exactly 98,304 projection parameters')
    spec = _models().SPEC
    for name, shape in PROJECTION_SHAPES.items():
        seed = (spec['common_projection_seed'] if name.endswith('nco_projection.weight')
                else spec['projection_seeds'][name.split('.')[-2]])
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            expected = torch.nn.Linear(shape[1], shape[0], bias=False).weight
        if not torch.equal(added[name].detach().cpu(), expected.detach()):
            raise ValueError('Projection must use the isolated common initialization: ' + name)
    provenance = donor_record(model.donor, arm)
    with torch.no_grad():
        for name, tensor in retained.items():
            tensor.copy_(reference[name])
        branch = model.instance_encoder.geometry_residual
        branch.mean.copy_(torch.as_tensor(geometry['mean'], dtype=branch.mean.dtype))
        branch.std.copy_(torch.as_tensor(geometry['std'], dtype=branch.std.dtype))
    if parameter_hash(retained) != initial['trainable_sha256']:
        raise ValueError('Retained R84 initialization changed')
    params = copy.deepcopy(model.params)
    params.update(r117_arm=arm, r117_fresh_trainable_sha256=parameter_hash(actual))
    receipt = dict(seed=2, initialization=INITIALIZATION, historical_selector_checkpoint_loaded=False,
        historical_checkpoint_loaded=False, external_solver_pretraining=True,
        common_base_pretrained_in_both_arms=True, added_native_pretrained=arm == 'B',
        fresh_reference_sha256=initial['trainable_sha256'],
        common_parameter_sha256=parameter_hash(retained), common_parameter_names=sorted(retained),
        common_parameter_count=ORIGINAL_PARAMETERS, new_parameter_names=sorted(added),
        new_parameter_shapes={n: list(p.shape) for n, p in added.items()},
        new_parameter_sha256=parameter_hash(added), new_parameter_count=NEW_PARAMETERS,
        trainable_sha256=parameter_hash(actual), trainable_parameter_count=PARAMETER_COUNT,
        parameter_layout_sha256=json_hash({n: list(p.shape) for n, p in actual.items()}),
        frozen_parameter_count=sum(p.numel() for p in model.donor.parameters()),
        frozen_donor=provenance, removed_parameter_names=[],
        geometry_fitted_on='frozen R84 new81000 TRAIN only')
    return model.to(device), params, receipt


def required_lock_fields():
    return dict(experiment='R117', locked=True, arms=list(ARMS), seed=2, task_scope='full18',
        rows_per_problem=dict(SPLIT_ROWS), logical_global_batch=BATCH_SIZE,
        max_successful_updates=MAX_UPDATES, warmup_updates=WARMUP_UPDATES,
        eval_every_updates=EVAL_EVERY, train_eval_every_updates=TRAIN_EVERY,
        optimizer='AdamW', peak_lr=PEAK_LR, end_lr=END_LR, weight_decay=1e-4,
        grad_clip=1., dropout=.1, checkpoint_rule=CHECKPOINT_RULE, inference_rule=INFERENCE_RULE,
        loss=copy.deepcopy(LOSS_RECIPE), rdrop_weight=.35, rdrop_ramp_updates=WARMUP_UPDATES,
        val_gate=copy.deepcopy(VAL_GATE), max_train_epochs=20,
        native_extraction_batch_size=128,
        per_arm_gpu_hours_cap=14, combined_gpu_hours_cap=28, gpu_type='3090', wandb_mode='offline')


def locked_caps(record):
    caps, evaluation = record.get('microbatch_caps', {}), record.get('eval_batch_size')
    if (set(caps) != set(PROBLEMS) or any(type(v) is not int or not 1 <= v <= BATCH_SIZE for v in caps.values())
            or type(evaluation) is not int or not 1 <= evaluation <= SPLIT_ROWS['val']):
        raise ValueError('Main witness must lock positive all18 microbatch_caps and eval_batch_size')
    return dict(caps), evaluation


def protocol_lock(args):
    path = Path(args.protocol_lock or args.root / 'protocol_lock.json').expanduser().resolve()
    record = read_json(path)
    mismatch = [k for k, v in required_lock_fields().items() if record.get(k) != v]
    if mismatch:
        raise ValueError('R117 main lock differs: ' + ', '.join(mismatch))
    caps, evaluation = locked_caps(record)
    for field, name in (('data_root', 'data_root'), ('prepared_root', 'prepared_root'), ('primary_cache', 'cache_root')):
        if Path(record[field]).resolve() != Path(getattr(args, name)).resolve():
            raise ValueError('R117 main lock names another ' + field)
    if (args.gpu_type != '3090' or args.wandb_mode != 'offline'
            or args.large_micro_cap is not None and args.large_micro_cap != caps['TSP']
            or args.eval_batch_size is not None and args.eval_batch_size != evaluation):
        raise ValueError('No override of witness-locked 3090/caps/eval/offline W&B')
    lib_batch = record.get('lib_batch_size', min(evaluation, caps['TSP'], caps['CVRP']))
    if type(lib_batch) is not int or not 1 <= lib_batch <= min(evaluation, caps['TSP'], caps['CVRP']):
        raise ValueError('LIB batch must fit the fixed witness caps')
    if args.lib_batch_size is not None and args.lib_batch_size != lib_batch:
        raise ValueError('CLI cannot change the fixed LIB batch')
    args.large_micro_cap, args.eval_batch_size, args.lib_batch_size = caps['TSP'], evaluation, lib_batch
    if (record.get('feature_roots', {}).get(args.arm) != str(feature_root(args).resolve())
            or record.get('common_feature_root') != str(common_root(args).resolve())):
        raise ValueError('Native/common feature roots must match the main-owned protocol lock')
    witness = Path(record.get('capacity_witness', ''))
    if not witness.is_file() or file_hash(witness) != record.get('capacity_witness_sha256'):
        raise ValueError('Main lock must bind the one completed capacity witness')
    capacity = read_json(witness)
    if (capacity.get('complete') is not True or capacity.get('microbatch_caps') != caps
            or capacity.get('eval_batch_size') != evaluation or capacity.get('native_extraction_batch_size') != 128):
        raise ValueError('Capacity witness differs from the frozen execution caps')
    return dict(path=str(path), sha256=file_hash(path), record=record)


def prepared_config(args):
    protocol_lock(args)
    ready, manifest, geometry, baselines = engine.prepared_config(args)
    for split in ('train', 'val'):
        for p in PROBLEMS:
            row, reference = manifest['splits'][split][p], manifest['splits']['train'][p]
            if (row.get('complete') is not True or row['pool'] != reference['pool']
                    or row['pool_ids'] != reference['pool_ids'] or row['pool_ids'] != sorted(set(row['pool_ids']))
                    or baselines[p].get('selection_split') != 'train'):
                raise ValueError('Require complete frozen native pools and TRAIN-only SBS')
    return ready, manifest, geometry, baselines


@lru_cache(maxsize=8)
def _private_engine(binding):
    bound = json.loads(binding)
    caps, _ = locked_caps(bound['record'])
    local = private_module(engine, 'engine_' + json_hash(bound)[:12])
    for name in ('BATCH_SIZE', 'MAX_UPDATES', 'WARMUP_UPDATES', 'EVAL_EVERY', 'PEAK_LR', 'END_LR', 'PART_KEYS'):
        setattr(local, name, globals()[name])
    local.SPLIT_ROWS, local.MODEL_FACTORY = dict(SPLIT_ROWS), MODEL_FACTORY
    local.LOSS_RECIPE = copy.deepcopy(LOSS_RECIPE)
    local.inputs_only, local.selection_key = inputs_only, selection_key
    local.write_json, local.wandb_run = _base._engine_write_json, wandb_run
    local.lock_best = lock_best

    def open_cache(cache_root, split, problem, manifest, data_root):
        from .r117_features import NativeFeatureCache
        base = engine.open_cache(cache_root, split, problem, manifest, data_root)
        try:
            return NativeFeatureCache(base, Path(bound['feature_root']) / split / problem,
                Path(bound['common_root']) / split / problem, expected_provenance=bound['provenance'])
        except BaseException:
            base.close()
            raise

    def micro_caps(gpu_type='3090', large_cap=None):
        if gpu_type != '3090' or large_cap is not None and large_cap != caps['TSP']:
            raise ValueError('No GPU override, capacity retry or automatic shrinking')
        return dict(caps)

    local.open_cache, local.micro_caps = open_cache, micro_caps
    adapt_function(local, engine.train, (
        ('params, order, caps)', 'params, order, caps, model=selector, initialization=initialization)'),
        ("cost = validation['macro']['macro_vs_sbs_pct']", 'cost = selection_key(validation)'),
        ('if not math.isfinite(cost):', 'if not all(math.isfinite(v) for v in cost):'),
        ("cost < best['val']['macro']['macro_vs_sbs_pct']", "cost < selection_key(best['val'])"),
        ('(40 - len(history))', '(MAX_UPDATES // EVAL_EVERY - len(history))')))
    return local


def private_engine(args=None, *, config=None, provenance=None):
    if config is not None:
        record = config['main_protocol_lock']['record']
        features, common = config['feature_cache_root'], config['common_feature_cache_root']
        provenance = config['model_params']['frozen_donor']['provenance']
    else:
        record = protocol_lock(args)['record']
        features, common = str(feature_root(args).resolve()), str(common_root(args).resolve())
        if provenance is None:
            provenance = read_json(args.root / args.arm / 'input_features_ready.json')['donor_provenance']
    return _private_engine(json.dumps(dict(record=record, feature_root=features,
        common_root=common, provenance=provenance), sort_keys=True))


def objective_spec(arm):
    if arm not in ARMS:
        raise ValueError('Unknown R117 arm')
    return dict(main='unchanged original R77 CE/comparison/risk plus symmetric R-Drop',
        coefficients=dict(ce=.35, comparison=.35, risk=.02), forward_passes=2,
        rdrop_weight=.35, rdrop_ramp_updates=WARMUP_UPDATES, donor_preparations_per_pair=1,
        auxiliary_loss=False, route_targets=False, cross_instance_objective=False,
        mutable_objective_state=False)


def lock_best(root, best, config, state_hash):
    record = dict(experiment='R117', arm=config['arm'], checkpoint=str((root / 'best.pt').resolve()),
        checkpoint_sha256=file_hash(root / 'best.pt'), state_hash=state_hash,
        epoch=best['epoch'], selected_update=best['successful_updates'], rule=CHECKPOINT_RULE,
        selected_without_test=True, selection_split='val', per_problem_assembly=False,
        val=best['val']['macro'], inference_rule=INFERENCE_RULE, winner_rule=WINNER_RULE)
    path = root / 'final_test_lock.json'
    if path.exists():
        prior = read_json(path)
        if any(prior.get(k) != v for k, v in record.items()):
            raise PermissionError('The unique whole18 VAL checkpoint/inference lock cannot change')
        return prior
    record['locked_utc'] = utc()
    write_json(path, record)
    return record


def protocol(args, ready, manifest, geometry, baselines, params, order, caps, *, model, initialization):
    main_lock = protocol_lock(args)
    config = private_engine(args).protocol(args, ready, manifest, geometry, baselines, params, order, caps)
    config.pop('common_trainable_sha256', None)
    config.update(experiment='R117', r117_schema_version=1, model_factory=MODEL_FACTORY,
        initialization=INITIALIZATION, main_protocol_lock=main_lock, loss=copy.deepcopy(LOSS_RECIPE),
        objective=objective_spec(args.arm), model_spec=copy.deepcopy(_models().SPEC), role_arm=MODEL_ROLE_ARM,
        frozen_donor=donor_record(model.donor, args.arm),
        feature_cache_root=str(feature_root(args).resolve()), common_feature_cache_root=str(common_root(args).resolve()),
        feature_preparation_batch_size=128, feature_precision='FP32 native, FP16 cache and online quantization',
        initial_trainable_sha256=initialization['trainable_sha256'],
        initial_parameter_layout_sha256=initialization['parameter_layout_sha256'],
        original_fresh_reference_sha256=ready['trainable_sha256'],
        common_parameter_sha256=initialization['common_parameter_sha256'],
        projection_initialization_sha256=initialization['new_parameter_sha256'],
        trainable_parameter_count=PARAMETER_COUNT, train_eval_every_updates=TRAIN_EVERY,
        checkpoint_rule=CHECKPOINT_RULE, inference_rule=INFERENCE_RULE, val_gate=copy.deepcopy(VAL_GATE),
        selected_val_binding_rule='exact_model_protocol_and_update',
        cross_arm_initialization='identical fresh R84 retained tensors and isolated five projections; frozen additions differ',
        common_base_pretrained_in_both_arms=True, added_native_pretrained=args.arm == 'B',
        train_eval=f'full TRAIN every{TRAIN_EVERY} successful updates plus selected checkpoint',
        max_train_epochs=20, train_exposures_per_problem=MAX_UPDATES // len(PROBLEMS) * BATCH_SIZE / SPLIT_ROWS['train'],
        current_query_labels_in_forward=False, historical_TRAIN_label_context_used=False,
        cache_policy='existing R84 raw source caches; new full18 input-only native extraction; TEST only after VAL lock',
        historical_comparison='R94B is historical, not a matched pretraining/control comparison',
        native_scope='finite LEHD/ReLD/MatNet reference computation; no actions, route prefixes or DIFUSCO claim',
        no_actions=True, complete_solutions=False,
        per_arm_gpu_hours_cap=14, combined_gpu_hours_cap=28,
        timeout_policy='main hard wall timeout includes feature extraction, train, TEST, LIB and timing; partial is incomplete',
        oom_policy='fail; no automatic cap shrink',
        plan_sha256=file_hash(args.root / 'PLAN.md'),
        source_sha256={name: file_hash(V4 / name) for name in SOURCE_FILES})
    config = json.loads(json.dumps(config, allow_nan=False))
    validate_checkpoint_config(config)
    return config


def validate_checkpoint_config(config):
    record = config.get('main_protocol_lock', {}).get('record', {})
    caps, evaluation = locked_caps(record)
    params = config.get('model_params', {})
    expected = dict(experiment='R117', r117_schema_version=1, model_factory=MODEL_FACTORY,
        initialization=INITIALIZATION, seed=2, solver_names=list(GLOBAL_SOLVERS), rows_per_problem=dict(SPLIT_ROWS),
        global_batch_size=BATCH_SIZE, logical_global_batch=BATCH_SIZE, max_successful_updates=MAX_UPDATES,
        warmup_updates=WARMUP_UPDATES, eval_every_updates=EVAL_EVERY, train_eval_every_updates=TRAIN_EVERY,
        eval_batch_size=evaluation, microbatch_caps=caps, optimizer='AdamW', weight_decay=1e-4,
        grad_clip=1., dropout=.1, rdrop_weight=.35, rdrop_ramp_updates=WARMUP_UPDATES,
        loss=LOSS_RECIPE, checkpoint_rule=CHECKPOINT_RULE, inference_rule=INFERENCE_RULE,
        val_gate=VAL_GATE, trainable_parameter_count=PARAMETER_COUNT, role_arm=MODEL_ROLE_ARM,
        selection_head='original_score_mlp_maximin', current_query_labels_in_forward=False,
        historical_TRAIN_label_context_used=False, common_base_pretrained_in_both_arms=True)
    spec = json.loads(json.dumps(_models().SPEC))
    if (config.get('arm') not in ARMS or any(config.get(k) != v for k, v in expected.items())
            or any(record.get(k) != v for k, v in required_lock_fields().items())
            or config.get('model_spec') != spec or params.get('model_spec') != spec
            or params.get('r117_arm') != config['arm'] or params.get('dropout') != .1
            or params.get('frozen_donor', {}).get('provenance') != config.get('frozen_donor')
            or config.get('objective') != objective_spec(config['arm'])
            or config.get('lr') != dict(rule='successful-update warmup then cosine, independent of VAL',
                peak=PEAK_LR, end=END_LR, warmup_updates=WARMUP_UPDATES, max_updates=MAX_UPDATES)
            or set(config.get('source_sha256', {})) != set(SOURCE_FILES)):
        raise ValueError('R117 checkpoint differs from fixed whole18 model/training/selection protocol')
    for field, locked in (('data_root', 'data_root'), ('prepared_root', 'prepared_root'), ('cache_root', 'primary_cache')):
        if Path(config[field]).resolve() != Path(record[locked]).resolve():
            raise ValueError('R117 embedded path differs: ' + field)


def load_checkpoint(path, device='cpu'):
    state = torch.load(path, map_location='cpu', weights_only=False)
    config = state['args']
    validate_checkpoint_config(config)
    if state.get('state_hash') != json_hash(config) or 'objective_state' in state:
        raise ValueError('R117 requires an immutable protocol and stateless objective')
    for name, expected in config['source_sha256'].items():
        if file_hash(V4 / name) != expected:
            raise ValueError('Frozen R117 source changed: ' + name)
    # The factory's restore=True donor path restores embedded weights, not external checkpoints.
    model = make_model(config['model_params'], config['arm'], state_dict=state['model'])
    if (sum(p.numel() for p in model.parameters() if p.requires_grad) != PARAMETER_COUNT
            or parameter_hash(model.state_dict()) != parameter_hash(state['model'])
            or donor_record(model.donor, config['arm']) != config['frozen_donor']):
        raise ValueError('Restore all trainable/frozen tensors and buffers exactly')
    return model.to(device), state


load_model = from_checkpoint = load_checkpoint


def make_objective(selector, arm):
    objective_spec(arm)
    return NativeObjective(selector)


def checkpoint_extras(objective):
    if type(objective) is not NativeObjective:
        raise TypeError('Use only the stateless original R77 native-input objective')
    return {}


def restore_objective(objective, checkpoint):
    checkpoint_extras(objective)
    if 'objective_state' in checkpoint:
        raise ValueError('R117 has no mutable objective state')


def update_batch(objective, optimizer, scaler, batch, weight, runtime, cap, *, problem, max_retries=20):
    if problem not in PROBLEMS or len(batch['ind']) != BATCH_SIZE:
        raise ValueError('Require exactly1024 real rows from one full18 task')
    return engine.update_batch(objective, optimizer, scaler, batch, weight, runtime, cap, max_retries)


def wandb_run(args, config):
    if args.wandb_mode != 'offline':
        raise ValueError('This bounded protocol requires offline W&B')
    return engine.wandb_run(args, config)


def feature_receipts(record, split, args, provenance):
    tasks = record.get('splits', {}).get(split, record)
    if (not isinstance(tasks, dict) or set(tasks) != set(PROBLEMS) or record.get('complete') is not True
            or record.get('format') != 'r117_native_node_features_v1'
            or record.get('labels_opened') is not False or record.get('actions_selected') is not False):
        raise ValueError('R117 requires complete all18 native feature receipts')
    for problem, receipt in tasks.items():
        identity = receipt.get('identity', receipt)
        if (receipt.get('complete') is not True
                or identity.get('problem') != problem or identity.get('split') != split
                or identity.get('rows') != SPLIT_ROWS[split] or receipt.get('rows_written') != SPLIT_ROWS[split]
                or identity.get('input_only') is not True or identity.get('labels_opened') is not False
                or identity.get('donor') != provenance or identity.get('dtype') != 'float16'
                or receipt.get('width') != (256 if problem == 'ATSP' else 128)):
            raise ValueError('Incomplete or unsafe native receipt: ' + split + '/' + problem)
        if Path(receipt.get('directory', '')).resolve() != (feature_root(args) / split / problem).resolve():
            raise ValueError('Native receipt belongs to another arm/root')
    if record.get('donor') != provenance:
        raise ValueError('Native preparation donor provenance changed')
    return tasks


def prepare_input_features(args, donor, device, *, split=None, test_lock=None):
    from .r117_features import prepare_features
    if split not in (None, 'train', 'val', 'test') or split == 'test' and test_lock is None:
        raise PermissionError('TEST native features require the genuine unique VAL lock')
    provenance, records = donor_record(donor, args.arm), {}
    for name in ((split,) if split is not None else ('train', 'val')):
        record = prepare_features(args.cache_root, feature_root(args), name, donor, device,
            common_root=common_root(args), batch_size=128,
            manifest=args.prepared_root / 'source_cache_manifest.json', data_root=args.data_root,
            test_lock=test_lock)
        records[name] = feature_receipts(record, name, args, provenance)
    return records


def prepare(args, device):
    from .r117_native import NativeComputation
    lock = protocol_lock(args)
    prepared_config(args)
    started = time.monotonic()
    with torch.random.fork_rng(devices=[]):
        donor = NativeComputation(args.arm).to(device).requires_grad_(False).eval()
    provenance = donor_record(donor, args.arm)
    receipts = prepare_input_features(args, donor, device)
    value = dict(experiment='R117', arm=args.arm, complete=True, completed_utc=utc(),
        main_protocol_lock_sha256=lock['sha256'], splits=['train', 'val'], preparation_batch_size=128,
        cache_root=str(args.cache_root.resolve()), feature_root=str(feature_root(args).resolve()),
        common_feature_root=str(common_root(args).resolve()), donor_provenance=provenance,
        feature_receipts=receipts, labels_used=False, solver_executed=False, no_actions=True,
        complete_solutions=False, seconds=time.monotonic() - started)
    path = args.root / args.arm / 'input_features_ready.json'
    if path.exists():
        prior = read_json(path)
        if any(prior.get(k) != v for k, v in value.items() if k not in ('seconds', 'completed_utc')):
            raise ValueError('R117 preparation identity changed; preserve existing run')
        return prior
    write_json(path, value)
    return value


def train(args, device):
    if not (args.root / args.arm / 'training_complete.json').exists():
        prepare(args, device)
    return private_engine(args).train(args, device, integration=sys.modules[__name__])


def authorize_test(args):
    config = read_json(args.root / args.arm / 'protocol.json')
    lock, checked = private_engine(config=config).authorize_test(args)
    validate_checkpoint_config(config)
    if (checked != config or protocol_lock(args) != config['main_protocol_lock']
            or lock.get('rule') != CHECKPOINT_RULE or lock.get('inference_rule') != INFERENCE_RULE
            or lock.get('selection_split') != 'val'
            or feature_root(args).resolve() != Path(config['feature_cache_root']).resolve()
            or common_root(args).resolve() != Path(config['common_feature_cache_root']).resolve()):
        raise PermissionError('TEST requires unchanged unique whole18 VAL/data/cache/inference locks')
    return lock, config


def ensure_test_cache(args, lock):
    verified, config = authorize_test(args)
    if verified != lock:
        raise PermissionError('TEST source request differs from the unique VAL lock')
    manifest = read_json(Path(config['prepared_root']) / 'source_cache_manifest.json')
    tasks = manifest.get('splits', {}).get('test', {})
    if (set(tasks) != set(PROBLEMS) or any(tasks[p].get('complete') is not True
            or tasks[p].get('rows') != SPLIT_ROWS['test']
            or tasks[p].get('pool') != config['candidate_pools'][p]
            or tasks[p].get('pool_ids') != config['candidate_pool_ids'][p] for p in PROBLEMS)):
        raise ValueError('Existing full18 R84 TEST source cache required; never silently generate/relabel')
    return manifest


def evaluate(selector, args, manifest, baselines, split, tag, step, state_hash, checkpoint_sha256=None):
    provenance = donor_record(selector.donor, args.arm)
    if split == 'test':
        lock, _ = authorize_test(args)
        if (step != lock['selected_update'] or state_hash != lock['state_hash']
                or checkpoint_sha256 != lock['checkpoint_sha256']):
            raise PermissionError('TEST prediction must use exactly the locked checkpoint')
        receipts = prepare_input_features(args, selector.donor, next(selector.parameters()).device,
            split='test', test_lock=lock)
        write_json(args.root / args.arm / 'test_input_features_ready.json', dict(experiment='R117',
            arm=args.arm, complete=True, checkpoint_sha256=lock['checkpoint_sha256'],
            state_hash=state_hash, feature_receipts=receipts, labels_used=False))
    return private_engine(args, provenance=provenance).evaluate(selector, args, manifest, baselines,
        split, tag, step, state_hash, checkpoint_sha256)


def test(args, device):
    config = read_json(args.root / args.arm / 'protocol.json')
    lock, _ = authorize_test(args)
    from .r117_reporting import bind_device
    bind_device(args, device, lock)
    return private_engine(config=config).test(args, device, integration=sys.modules[__name__])


def cached_input_batch(config, manifest, split, problem, device, count=None):
    return _base.cached_input_batch(config, manifest, split, problem, device, count)


def timing(args, device):
    from .r117_reporting import measure_timing
    return measure_timing(args, device)


def finalize(args, device):
    result = test(args, device)
    from . import r117_reporting
    r117_reporting.lib_stage(args)
    measured = timing(args, device)
    return dict(test=result, timing=measured, report=r117_reporting.report(args))


def describe(args):
    path = args.protocol_lock or args.root / 'protocol_lock.json'
    lock = protocol_lock(args) if path.exists() else None
    return dict(experiment='R117', arm=args.arm, required_protocol_lock=required_lock_fields(),
        required_execution_fields=['data_root', 'prepared_root', 'primary_cache',
            'microbatch_caps (all18; one bounded capacity witness)', 'eval_batch_size', 'lib_batch_size'],
        main_protocol_lock=lock, model_factory=MODEL_FACTORY, trainable_parameter_count=PARAMETER_COUNT,
        feature_root=str(feature_root(args)), common_root=str(common_root(args)),
        train_prepares_TRAIN_VAL=True, TEST_native_features_require_unique_VAL_lock=True,
        launcher_contract='one3090/arm; hard14h including features+all inference; no allocation/retry/shrinking',
        source_files=list(SOURCE_FILES), val_gate=VAL_GATE)


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('command', choices=('train', 'test', 'timing', 'finalize', 'describe', 'status'))
    result.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    result.add_argument('--arm', choices=ARMS, required=True)
    result.add_argument('--data-root', type=Path, default=DEFAULT_DATA_ROOT)
    result.add_argument('--prepared-root', type=Path, default=DEFAULT_PREPARED_ROOT)
    result.add_argument('--cache-root', type=Path, default=DEFAULT_CACHE_ROOT)
    result.add_argument('--feature-root', type=Path)
    result.add_argument('--common-root', type=Path, default=DEFAULT_COMMON_ROOT)
    result.add_argument('--protocol-lock', type=Path)
    result.add_argument('--device', default='cuda:0')
    result.add_argument('--gpu-type', choices=('3090',), default='3090')
    result.add_argument('--large-micro-cap', type=int)
    result.add_argument('--eval-batch-size', type=int)
    result.add_argument('--lib-batch-size', type=int)
    result.add_argument('--log-every', type=int, default=18)
    result.add_argument('--wandb-mode', choices=('offline',), default='offline')
    result.add_argument('--resume', action='store_true')
    result.add_argument('--state-hash', '--expected-state-hash')
    result.add_argument('--nss-root', type=Path, default=V4 / 'runs/R84_new90k_attention/nss_reference')
    result.set_defaults(experiment='R117')
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    for name in ('root', 'data_root', 'prepared_root', 'cache_root', 'feature_root', 'common_root',
                 'protocol_lock', 'nss_root'):
        value = getattr(args, name)
        if value is not None:
            setattr(args, name, value.expanduser().resolve())
    if args.command in ('describe', 'status'):
        path = args.root / args.arm / 'status.json'
        value = describe(args) if args.command == 'describe' else (
            read_json(path) if path.exists() else dict(experiment='R117', arm=args.arm, phase='not_started'))
        print(json.dumps(value, indent=2))
        return value
    protocol_lock(args)
    if args.log_every < 1:
        raise ValueError('Log interval must be positive')
    run = args.root / args.arm
    run.mkdir(parents=True, exist_ok=True)
    with (run / 'runner.lock').open('a') as claim:
        fcntl.flock(claim, fcntl.LOCK_EX | fcntl.LOCK_NB)
        torch.set_num_threads(2)
        engine.configure_torch()
        device = torch.device(args.device)
        if device.type != 'cuda':
            raise ValueError('Formal execution requires a main-assigned RTX3090')
        try:
            return globals()[args.command](args, device)
        except BaseException as error:
            write_json(run / 'error.json', dict(experiment='R117', arm=args.arm, updated_utc=utc(),
                command=args.command, error_type=type(error).__name__, message=str(error)))
            raise


if __name__ == '__main__':
    main()
