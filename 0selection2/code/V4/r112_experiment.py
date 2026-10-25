"""R112 full18 runner, adapting a private copy of the frozen R84 engine.

Integration contract: r112_model supplies ARMS, make_model and initialize_shared.
Only TRAIN supervision reaches the original R77 two-forward objective. Caps are
required in the main thread's protocol_lock.json (microbatch_caps, eval_batch_size),
not inferred from a GPU name. `train` includes locked TEST; `finalize` additionally
finishes input-only timing, LIB and saved-prediction PDF/CSV reporting.
"""

import argparse
import copy
import fcntl
import importlib
import importlib.util
import inspect
import json
import math
import sys
import time
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch

from code.unified_selector.registry import GLOBAL_SOLVERS, PROBLEMS
from . import r84_experiment as engine
from .r76_experiment import RDropObjective, file_hash, read_json, utc, write_json
from .r77_data import parameter_hash
from .r84_prepare import DEFAULT_DATA_ROOT, DEFAULT_PREPARED_ROOT, SPLIT_ROWS


V4 = Path(__file__).resolve().parent
DEFAULT_ROOT = V4 / 'runs/R112_region_interfaces'
DEFAULT_CACHE_ROOT = Path('/tmp/shiys_R84_new90k_cache')
ARMS = ('A_collapsed', 'B_interfaces')
BATCH_SIZE, MAX_UPDATES, WARMUP_UPDATES = 512, 57600, 4320
EVAL_EVERY, TRAIN_EVERY = 2880, 14400
PEAK_LR, END_LR = 1e-4, 1e-6
BUDGETS = dict.fromkeys(ARMS, MAX_UPDATES)
MODEL_FACTORY = 'code.V4.r112_experiment.load_checkpoint'
MODEL_ROLE_ARM = 'B_execution_anchor'
INITIALIZATION = 'scratch_seed2_identical_R112_parameters_R84_subset_reuse'
CHECKPOINT_RULE = ('strict maximum full18 VAL macro_top1; exact Top1 ties: minimum '
                   'macro_actual_regret_pct; exact double ties: first checkpoint')
INFERENCE_RULE = 'eval mode, original R77 score-difference maximin; stable ascending-global-solver-ID ties'
WINNER_RULE = engine.WINNER_RULE
json_hash, task_order = engine.json_hash, engine.task_order


def _models():
    module = importlib.import_module('code.V4.r112_model')
    if tuple(module.ARMS) != ARMS:
        raise ValueError('R112 model and runner arms differ')
    return module


def make_model(params, arm, state_dict=None):
    return _models().make_model(params, arm, state_dict=state_dict)


def initialize_shared(initial, geometry, arm, device='cpu'):
    if (arm not in ARMS or initial.get('seed') != 2 or initial.get('initialization') != 'scratch'
            or parameter_hash(initial['trainable']) != initial.get('trainable_sha256')):
        raise ValueError('Use the prepared fresh seed2 tensors, never a trained checkpoint')
    model, params, receipt = _models().initialize_shared(initial, geometry, arm, device)
    if (params.get('dropout') != .1 or params.get('role_arm', MODEL_ROLE_ARM) != MODEL_ROLE_ARM
            or params.get('r112_spec', {}).get('role_arm', MODEL_ROLE_ARM) != MODEL_ROLE_ARM):
        raise ValueError('R112 requires dropout .1 and the unchanged R88B execution role')
    weights = dict(model.named_parameters())
    receipt = dict(receipt, seed=2, historical_checkpoint_loaded=False,
        fresh_reference_sha256=initial['trainable_sha256'],
        trainable_sha256=parameter_hash(weights),
        trainable_parameter_count=sum(p.numel() for p in weights.values() if p.requires_grad),
        parameter_layout_sha256=json_hash({k: list(v.shape) for k, v in weights.items()}),
        geometry_fitted_on='frozen R84 81000 TRAIN only')
    return model, params, receipt


def required_lock_fields():
    return dict(experiment='R112', locked=True, arms=list(ARMS), seed=2, task_scope='full18',
        rows_per_problem=dict(SPLIT_ROWS), logical_global_batch=BATCH_SIZE,
        max_successful_updates=MAX_UPDATES, warmup_updates=WARMUP_UPDATES,
        eval_every_updates=EVAL_EVERY, train_eval_every_updates=TRAIN_EVERY,
        optimizer='AdamW', peak_lr=PEAK_LR, end_lr=END_LR, weight_decay=1e-4,
        grad_clip=1., dropout=.1, checkpoint_rule=CHECKPOINT_RULE, inference_rule=INFERENCE_RULE)


def locked_caps(record):
    caps, evaluation = record.get('microbatch_caps', {}), record.get('eval_batch_size')
    if (set(caps) != set(PROBLEMS) or any(type(v) is not int or not 1 <= v <= BATCH_SIZE
                                        for v in caps.values())
            or type(evaluation) is not int or not 1 <= evaluation <= SPLIT_ROWS['val']):
        raise ValueError('Main lock must fix all18 positive microbatch_caps and eval_batch_size')
    return dict(caps), evaluation


def protocol_lock(args):
    path = Path(args.protocol_lock or args.root / 'protocol_lock.json').expanduser().resolve()
    record = read_json(path)
    mismatches = [k for k, v in required_lock_fields().items() if record.get(k) != v]
    if mismatches:
        raise ValueError('R112 main lock differs: ' + ', '.join(mismatches))
    caps, evaluation = locked_caps(record)
    for name, field in (('data_root', 'data_root'), ('prepared_root', 'prepared_root'),
                        ('cache_root', 'primary_cache')):
        if Path(record[field]).resolve() != Path(getattr(args, name)).resolve():
            raise ValueError('R112 main lock names another ' + field)
    if (args.large_micro_cap is not None and args.large_micro_cap != caps['TSP']
            or args.eval_batch_size is not None and args.eval_batch_size != evaluation):
        raise ValueError('CLI cannot override witness-locked caps/evaluation batch')
    args.large_micro_cap, args.eval_batch_size = caps['TSP'], evaluation
    return dict(path=str(path), sha256=file_hash(path), record=record)


def prepared_config(args):
    protocol_lock(args)
    return engine.prepared_config(args)


def selection_key(validation):
    """Minimize this tuple; no tolerance-based ties or cost-ratio proxy."""
    macro = validation['macro']
    top1, regret = float(macro['macro_top1']), float(macro['macro_actual_regret_pct'])
    if not math.isfinite(top1) or not 0 <= top1 <= 1 or not math.isfinite(regret):
        raise ValueError('VAL Top1 and actual regret must be finite native metrics')
    return -top1, regret


def private_module(source, name):
    spec = importlib.util.spec_from_file_location('code.V4._r112_' + name, source.__file__)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def adapt_function(module, original, replacements):
    """Small checked source substitutions keep the old engine itself unchanged."""
    source = inspect.getsource(original)
    for old, new in replacements:
        if source.count(old) != 1:
            raise RuntimeError('Historical template drift; inspect adaptation: ' + old)
        source = source.replace(old, new)
    exec(compile(source, original.__code__.co_filename + ':R112_private', 'exec'), module.__dict__)


@lru_cache(maxsize=8)
def _private_engine(binding):
    record = json.loads(binding)
    caps, _ = locked_caps(record)
    module = private_module(engine, 'engine_' + json_hash(record)[:12])
    module.SPLIT_ROWS = copy.deepcopy(SPLIT_ROWS)
    for name in ('BATCH_SIZE', 'MAX_UPDATES', 'WARMUP_UPDATES', 'EVAL_EVERY', 'PEAK_LR', 'END_LR'):
        setattr(module, name, globals()[name])
    if TRAIN_EVERY != 5 * EVAL_EVERY or MAX_UPDATES % EVAL_EVERY:
        raise ValueError('Private R84 cadence requires five VAL intervals per TRAIN evaluation')

    def micro_caps(gpu_type='4090', large_cap=None):
        if large_cap is not None and large_cap != caps['TSP']:
            raise ValueError('No automatic cap shrink after the main lock')
        return dict(caps)

    module.micro_caps, module.selection_key = micro_caps, selection_key
    module.write_json = _engine_write_json
    adapt_function(module, engine.train, (
        ('params, order, caps)', 'params, order, caps, model=selector, initialization=initialization)'),
        ("cost = validation['macro']['macro_vs_sbs_pct']", 'cost = selection_key(validation)'),
        ('if not math.isfinite(cost):', 'if not all(math.isfinite(v) for v in cost):'),
        ("cost < best['val']['macro']['macro_vs_sbs_pct']", "cost < selection_key(best['val'])"),
        ('(40 - len(history))', '(MAX_UPDATES // EVAL_EVERY - len(history))'),
        ('exactly1024 natural rows', 'exactly the locked logical batch of natural rows')))
    return module


def private_engine(args=None, *, config=None):
    record = config['main_protocol_lock']['record'] if config is not None else protocol_lock(args)['record']
    return _private_engine(json.dumps(record, sort_keys=True))


def protocol(args, ready, manifest, geometry, baselines, params, order, caps, *, model, initialization):
    main_lock = protocol_lock(args)
    local = private_engine(args)
    config = local.protocol(args, ready, manifest, geometry, baselines, params, order, caps)
    config['loss'].pop('focal_extra', None)
    config.pop('common_trainable_sha256', None)
    config.update(experiment='R112', r112_schema_version=1, model_factory=MODEL_FACTORY,
        initialization=INITIALIZATION, main_protocol_lock=main_lock,
        model_spec=copy.deepcopy(params.get('r112_spec')),
        trainable_parameter_count=sum(p.numel() for p in model.parameters() if p.requires_grad),
        initial_trainable_sha256=initialization['trainable_sha256'],
        initial_parameter_layout_sha256=initialization['parameter_layout_sha256'],
        original_fresh_reference_sha256=ready['trainable_sha256'],
        common_parameter_sha256=initialization.get('common_parameter_sha256'),
        cross_arm_initialization='all named parameters identical; original R84 reuse is a subset',
        train_eval_every_updates=TRAIN_EVERY, role_arm=MODEL_ROLE_ARM,
        selection_head='original_score_mlp_maximin', checkpoint_rule=CHECKPOINT_RULE,
        inference_rule=INFERENCE_RULE, selected_val_binding_rule='exact_model_protocol_and_update',
        gradients=f'micro rows / logical{BATCH_SIZE}; one clip/step; same-batch/RNG AMP replay',
        source_plan=f'common seed2 full18 natural stream; exact{BATCH_SIZE} rows; shuffled tails carried across cycles',
        train_eval=f'full TRAIN every{TRAIN_EVERY} successful updates plus selected checkpoint',
        objective=dict(classification='original R77 native CE', forward_passes=2,
            comparison=True, relative_risk=True, symmetric_rdrop=True, mutable_objective_state=False),
        current_query_labels_in_forward=False, historical_TRAIN_label_context_used=False,
        cache_policy='read-only reuse of frozen R84 prepared/raw cache; no implicit generation',
        oom_policy='fail; no automatic cap shrink',
        source_sha256={name: file_hash(V4 / name) for name in
            ('r112_experiment.py', 'r112_model.py', 'r84_experiment.py', 'r84_prepare.py',
             'r76_experiment.py', 'r77_experiment.py', 'pairwise_objective.py', 'r88_model.py')})
    # Model SPEC contains tuples; persisted JSON and resume must compare exactly.
    config = json.loads(json.dumps(config, allow_nan=False))
    validate_checkpoint_config(config)
    return config


def validate_checkpoint_config(config):
    record = config.get('main_protocol_lock', {}).get('record', {})
    caps, evaluation = locked_caps(record)
    expected = dict(experiment='R112', r112_schema_version=1, model_factory=MODEL_FACTORY,
        initialization=INITIALIZATION, seed=2, solver_names=list(GLOBAL_SOLVERS),
        rows_per_problem=SPLIT_ROWS, global_batch_size=BATCH_SIZE, logical_global_batch=BATCH_SIZE,
        max_successful_updates=MAX_UPDATES, warmup_updates=WARMUP_UPDATES,
        eval_every_updates=EVAL_EVERY, train_eval_every_updates=TRAIN_EVERY,
        eval_batch_size=evaluation, microbatch_caps=caps, optimizer='AdamW',
        weight_decay=1e-4, grad_clip=1., dropout=.1, rdrop_weight=.35,
        rdrop_ramp_updates=WARMUP_UPDATES, role_arm=MODEL_ROLE_ARM,
        loss=engine.LOSS_RECIPE, checkpoint_rule=CHECKPOINT_RULE, inference_rule=INFERENCE_RULE)
    if (config.get('arm') not in ARMS or any(config.get(k) != v for k, v in expected.items())
            or any(record.get(k) != v for k, v in required_lock_fields().items())
            or config.get('lr') != dict(rule='successful-update warmup then cosine, independent of VAL',
                peak=PEAK_LR, end=END_LR, warmup_updates=WARMUP_UPDATES, max_updates=MAX_UPDATES)
            or config.get('model_params', {}).get('role_arm', MODEL_ROLE_ARM) != MODEL_ROLE_ARM
            or config.get('model_params', {}).get('r112_arm') != config.get('arm')
            or config.get('model_spec') != config.get('model_params', {}).get('r112_spec')
            or config.get('model_params', {}).get('dropout') != .1):
        raise ValueError('R112 checkpoint differs from frozen data/training/selection protocol')
    for field, locked in (('data_root', 'data_root'), ('prepared_root', 'prepared_root'), ('cache_root', 'primary_cache')):
        if Path(config[field]).resolve() != Path(record[locked]).resolve():
            raise ValueError('R112 embedded path differs: ' + field)


def load_checkpoint(path, device='cpu'):
    state = torch.load(path, map_location='cpu', weights_only=False)
    config = state['args']
    validate_checkpoint_config(config)
    if state.get('state_hash') != json_hash(config) or 'objective_state' in state:
        raise ValueError('R112 checkpoint protocol/stateless objective identity differs')
    for name, expected in config['source_sha256'].items():
        if file_hash(V4 / name) != expected:
            raise ValueError('Frozen R112 inference source changed: ' + name)
    model = make_model(config['model_params'], config['arm'], state_dict=state['model'])
    if (sum(p.numel() for p in model.parameters() if p.requires_grad) != config['trainable_parameter_count']
            or parameter_hash(model.state_dict()) != parameter_hash(state['model'])):
        raise ValueError('R112 factory must restore all parameters and buffers exactly')
    return model.to(device), state


load_model = from_checkpoint = load_checkpoint


def make_objective(selector, arm):
    if arm not in ARMS:
        raise ValueError('Unknown R112 arm')
    return RDropObjective(selector)


def checkpoint_extras(objective):
    if type(objective) is not RDropObjective:
        raise TypeError('R112 uses the original stateless two-forward RDropObjective')
    return {}


def restore_objective(objective, checkpoint):
    checkpoint_extras(objective)
    if 'objective_state' in checkpoint:
        raise ValueError('Unexpected mutable objective state')


def update_batch(objective, optimizer, scaler, batch, weight, runtime, cap, *, problem, max_retries=20):
    if problem not in PROBLEMS or len(batch['ind']) != BATCH_SIZE:
        raise ValueError('R112 requires a complete natural logical batch from one of all18 tasks')
    return engine.update_batch(objective, optimizer, scaler, batch, weight, runtime, cap, max_retries)


def selected_capture_proof(run, result, lock, split, state=None):
    run = Path(run)
    if file_hash(run / 'best.pt') != lock['checkpoint_sha256']:
        raise ValueError('Selected capture requires the immutable checkpoint')
    if state is None:
        state = torch.load(run / 'best.pt', map_location='cpu', weights_only=False)
    capture = result.get('archive_capture_identity', result['identity'])
    path = Path(result['prediction_archive']).resolve()
    expected = dict(experiment='R112', arm=lock['arm'], split=split,
        model_state_sha256=parameter_hash(state['model']), state_hash=lock['state_hash'],
        successful_updates=lock['selected_update'])
    if (not path.is_relative_to((run / 'evaluation' / split).resolve())
            or not path.is_file() or read_json(path.parent / 'identity.json') != capture
            or any(capture.get(k) != v for k, v in expected.items())
            or capture.get('checkpoint_sha256') not in (None, lock['checkpoint_sha256'])
            or state['args'].get('experiment') != 'R112' or state['args'].get('arm') != lock['arm']
            or state.get('state_hash') != lock['state_hash'] or json_hash(state['args']) != lock['state_hash']
            or state.get('successful_updates') != lock['selected_update']):
        raise ValueError('Selected prediction capture differs from checkpoint/model/protocol/update')
    return copy.deepcopy(capture)


def _engine_write_json(path, value):
    path = Path(path)
    if path.name == 'initialization.json':
        sibling = next(a for a in ARMS if a != path.parent.name)
        other = path.parent.parent / sibling / path.name
        if other.exists():
            receipt = read_json(other)
            if any(receipt.get(k) != value.get(k) for k in
                   ('fresh_reference_sha256', 'trainable_sha256', 'parameter_layout_sha256')):
                raise ValueError('R112 A/B must start from identical complete parameter tensors')
    if path.name == 'final_metrics.json':
        value = copy.deepcopy(value)
        lock = read_json(path.parent / 'final_test_lock.json')
        if any(value.get(k) != lock[k] for k in ('state_hash', 'selected_update', 'checkpoint_sha256')):
            raise ValueError('Final metrics differ from the unique VAL checkpoint')
        capture = selected_capture_proof(path.parent, value['val'], lock, 'val')
        value['val'].update(archive_capture_identity=capture,
            identity=dict(capture, checkpoint_sha256=lock['checkpoint_sha256']))
    return write_json(path, value)


def evaluate(selector, args, *positional, **keywords):
    return private_engine(args).evaluate(selector, args, *positional, **keywords)


def authorize_test(args):
    lock, config = private_engine(args).authorize_test(args)
    validate_checkpoint_config(config)
    if (protocol_lock(args) != config['main_protocol_lock']
            or lock.get('rule') != CHECKPOINT_RULE or lock.get('inference_rule') != INFERENCE_RULE):
        raise PermissionError('R112 TEST must use the unchanged unique VAL/main/inference lock')
    return lock, config


def ensure_test_cache(args, lock):
    verified, config = authorize_test(args)
    if verified != lock:
        raise PermissionError('TEST cache request names another checkpoint')
    manifest = read_json(Path(config['prepared_root']) / 'source_cache_manifest.json')
    tasks = manifest.get('splits', {}).get('test', {})
    if set(tasks) != set(PROBLEMS) or any(tasks[p].get('rows') != SPLIT_ROWS['test']
                                        or tasks[p].get('complete') is not True for p in PROBLEMS):
        raise ValueError('Complete preexisting full18 R84 TEST cache required; no implicit generation')
    return manifest


def train(args, device):
    return private_engine(args).train(args, device, integration=sys.modules[__name__])


def test(args, device):
    return private_engine(args).test(args, device, integration=sys.modules[__name__])


# Adapted from the R109 input-only timing helpers; local for independent publication.
def raw_inputs(batch):
    names = ('node', 'matrix', 'node_mask', 'n', 'node_geom', 'kind', 'problem_id',
             'coord_dist', 'cbits', 'problem_desc', 'd_in', 'pool_ids', 'solver_mask', 'mask')
    return {key: batch[key] for key in names if key in batch}


def cached_input_batch(config, manifest, split, problem, device, count=None):
    """Read only frozen input arrays, without opening supervision or targets."""
    from .r75_cache import _close, _open_arrays
    from .r76_cache import _validate_record
    count = config['eval_batch_size'] if count is None else count
    if not isinstance(count, int) or count < 1:
        raise ValueError('Timing requires a positive input batch size')
    root = Path(config['cache_root'])
    directory = next((p for p in (root / split / problem, root / (problem + split)) if p.is_dir()), None)
    if directory is None:
        raise FileNotFoundError('Existing frozen input cache required: ' + problem + '/' + split)
    record = read_json(directory / 'metadata.json')
    specs = _validate_record(record, directory)
    expected = manifest['splits'][split][problem]
    if (record['problem'] != problem or record['split'] != split or record['rows'] != SPLIT_ROWS[split]
            or record['rows'] != expected['rows'] or record['rows'] < count
            or record['pool'] != config['candidate_pools'][problem]
            or record['pool_ids'] != config['candidate_pool_ids'][problem]
            or record['pool'] != expected['pool'] or record['pool_ids'] != expected['pool_ids']
            or Path(record['data_root']).resolve() != Path(config['data_root']).resolve()):
        raise ValueError('R112 input-only cache provenance differs')
    keys = ('n', 'coord_dist', 'node', 'node_geom') if record['kind'] == 'coord' else ('n', 'coord_dist', 'matrix')
    arrays = _open_arrays(directory, {k: specs[k] for k in keys})
    try:
        n = torch.from_numpy(np.array(arrays['n'][:count], copy=True))
        maximum = int(n.max())
        batch = dict(problem_id=record['problem_id'], kind=record['kind'], n=n,
            node_mask=torch.arange(maximum)[None] < n[:, None],
            cbits=torch.tensor(record['cbits'], dtype=torch.float32).repeat(count, 1),
            problem_desc=torch.tensor(record['problem_desc'], dtype=torch.float32).repeat(count, 1),
            coord_dist=torch.from_numpy(np.array(arrays['coord_dist'][:count], copy=True)),
            pool_ids=torch.tensor(record['pool_ids'], dtype=torch.long),
            mask=torch.tensor(record['mask'], dtype=torch.float32).repeat(count, 1),
            solver_mask=torch.ones(count, len(record['pool']), dtype=torch.bool))
        if record['kind'] == 'coord':
            batch['d_in'] = record['d_in']
            for key in ('node', 'node_geom'):
                batch[key] = torch.from_numpy(np.array(arrays[key][:count, :maximum], copy=True))
        else:
            batch['matrix'] = torch.from_numpy(np.array(arrays['matrix'][:count, :maximum, :maximum], copy=True))
    finally:
        _close(arrays)
    return {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}


@torch.no_grad()
def measure_inputs(model, batch, device, repetitions=3):
    raw = raw_inputs(batch)

    def sync():
        if device.type == 'cuda':
            torch.cuda.synchronize(device)

    model(raw)
    sync()
    samples = []
    for _ in range(repetitions):
        sync()
        started = time.perf_counter()
        output = model(raw)
        sync()
        samples.append(time.perf_counter() - started)
        if not torch.isfinite(output['logits'][output['solver_mask']]).all():
            raise ValueError('Nonfinite R112 timing prediction')
    return dict(batch_seconds=samples,
        mean_seconds_per_instance=sum(samples) / (repetitions * len(raw['node_mask'])))


@torch.no_grad()
def timing(args, device):
    if device.type != 'cuda':
        raise ValueError('Standalone timing requires an explicitly assigned CUDA device')
    lock, config = authorize_test(args)
    run = args.root / args.arm
    saved = read_json(run / 'test_results.json')
    if (saved.get('complete') is not True or saved.get('checkpoint_sha256') != lock['checkpoint_sha256']
            or saved.get('state_hash') != lock['state_hash']):
        raise PermissionError('Timing follows complete locked TEST')
    from .r112_reporting import inference_timing
    if (run / 'inference_timing.json').exists():
        return inference_timing(run, config, lock, saved['result']['identity']['model_state_sha256'])
    model, state = load_checkpoint(run / 'best.pt', device)
    model_hash = parameter_hash(state['model'])
    if model_hash != saved['result']['identity']['model_state_sha256']:
        raise ValueError('Timing and TEST must use the same complete model')
    del state
    model.eval()
    manifest = ensure_test_cache(args, lock)
    count, rows = config['eval_batch_size'], []
    for problem in PROBLEMS:
        batch = cached_input_batch(config, manifest, 'test', problem, device, count=count)
        rows.append(dict(problem=problem, instances=count,
            standalone_model=measure_inputs(model, batch, device, repetitions=3)))
        del batch
    if file_hash(run / 'best.pt') != lock['checkpoint_sha256'] or parameter_hash(model.state_dict()) != model_hash:
        raise ValueError('Timing changed the locked checkpoint/model')
    value = dict(experiment='R112', arm=args.arm, complete=True, completed_utc=utc(),
        checkpoint_sha256=lock['checkpoint_sha256'], state_hash=lock['state_hash'],
        selected_update=lock['selected_update'], model_state_sha256=model_hash,
        model_factory=MODEL_FACTORY, inference_rule=INFERENCE_RULE, device=str(device),
        device_name=torch.cuda.get_device_name(device), repetitions=3, warmup_batches=1,
        eval_batch_size=count, parameter_count=config['trainable_parameter_count'],
        scope='complete_standalone_model_original_GPU_resident_inputs',
        sample_policy=f'first{count} frozen TEST inputs per problem', all_current_model_included=True,
        end_to_end_measured=False, input_labels_read=False,
        excludes='dataset I/O, CPU-to-GPU transfer, training and solver execution',
        per_problem=rows, model_reselection=False)
    write_json(run / 'inference_timing.json', value)
    return inference_timing(run, config, lock, model_hash)


def finalize(args, device):
    """Idempotent same-checkpoint TEST -> timing -> LIB -> CPU reports."""
    result = test(args, device)
    timing(args, device)
    from . import r112_reporting
    r112_reporting.lib_stage(args)
    return dict(test=result, report=r112_reporting.report(args))


def describe(args):
    path = args.protocol_lock or args.root / 'protocol_lock.json'
    locked = protocol_lock(args) if path.exists() else None
    return dict(experiment='R112', arm=args.arm, required_protocol_lock=required_lock_fields(),
        required_execution_fields=['data_root', 'prepared_root', 'primary_cache',
                                   'microbatch_caps (all18)', 'eval_batch_size'],
        main_protocol_lock=locked, model_factory=MODEL_FACTORY,
        model_api='ARMS; make_model(params,arm,state_dict=None); initialize_shared(initial,geometry,arm,device)',
        cache_root=str(args.cache_root), caps_status='locked' if locked else 'awaiting_main_GPU_witness',
        wandb_mode=args.wandb_mode)


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('command', choices=('train', 'test', 'timing', 'finalize', 'describe', 'status'))
    result.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    result.add_argument('--arm', choices=ARMS, required=True)
    result.add_argument('--data-root', type=Path, default=DEFAULT_DATA_ROOT)
    result.add_argument('--prepared-root', type=Path, default=DEFAULT_PREPARED_ROOT)
    result.add_argument('--cache-root', type=Path, default=DEFAULT_CACHE_ROOT)
    result.add_argument('--protocol-lock', type=Path)
    result.add_argument('--device', default='cuda:0')
    result.add_argument('--gpu-type', choices=('3090', '4090'), default='4090')
    result.add_argument('--large-micro-cap', type=int)
    result.add_argument('--eval-batch-size', type=int)
    result.add_argument('--log-every', type=int, default=18)
    result.add_argument('--wandb-mode', choices=('auto', 'online', 'offline'), default='auto')
    result.add_argument('--resume', action='store_true')
    result.add_argument('--state-hash', '--expected-state-hash')
    result.add_argument('--lib-batch-size', type=int)
    result.add_argument('--nss-root', type=Path, default=V4 / 'runs/R84_new90k_attention/nss_reference')
    result.set_defaults(experiment='R112')
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    for name in ('root', 'data_root', 'prepared_root', 'cache_root', 'protocol_lock', 'nss_root'):
        value = getattr(args, name)
        if value is not None:
            setattr(args, name, value.expanduser().resolve())
    if args.command in ('describe', 'status'):
        path = args.root / args.arm / 'status.json'
        value = describe(args) if args.command == 'describe' else (
            read_json(path) if path.exists() else dict(experiment='R112', arm=args.arm, phase='not_started'))
        print(json.dumps(value, indent=2))
        return value
    protocol_lock(args)
    if args.log_every < 1 or args.lib_batch_size is not None and args.lib_batch_size < 1:
        raise ValueError('Log interval and LIB batch must be positive')
    run = args.root / args.arm
    run.mkdir(parents=True, exist_ok=True)
    with (run / 'runner.lock').open('a') as claim:
        fcntl.flock(claim, fcntl.LOCK_EX | fcntl.LOCK_NB)
        torch.set_num_threads(2)
        engine.configure_torch()
        device = torch.device(args.device)
        if device.type != 'cuda':
            raise ValueError('Formal train/TEST/timing requires a launcher-assigned CUDA device')
        try:
            return globals()[args.command](args, device)
        except BaseException as error:
            write_json(run / 'error.json', dict(experiment='R112', arm=args.arm, updated_utc=utc(),
                command=args.command, error_type=type(error).__name__, message=str(error)))
            raise


if __name__ == '__main__':
    main()
