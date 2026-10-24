"""Full18 R107 connected-path content via the private original R84 engine."""

import argparse
import copy
import fcntl
import importlib.util
import json
import sys
import time
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch

from code.unified_selector.registry import GLOBAL_SOLVERS, PROBLEMS
from . import r84_experiment as engine
from . import r107_model as models
from .r76_experiment import RDropObjective, file_hash, read_json, utc, write_json
from .r77_data import parameter_hash
from .r84_prepare import DEFAULT_DATA_ROOT, DEFAULT_PREPARED_ROOT, SPLIT_ROWS
from .r85_experiment import selection_counts, zero_summary


V4 = Path(__file__).resolve().parent
DEFAULT_ROOT = V4 / 'runs/R107_explicit_path_relations'
DEFAULT_CACHE_ROOT = Path('/tmp/shiys_R84_new90k_cache')
ARMS = NEURAL_ARMS = models.ARMS
BATCH_SIZE, MAX_UPDATES, WARMUP_UPDATES, EVAL_EVERY = 1024, 57600, 4320, 1440
TRAIN_EVERY, PEAK_LR, END_LR = 7200, 1e-4, 1e-6
PATH_INIT_SEED = models.PATH_INIT_SEED
BUDGETS = dict.fromkeys(ARMS, MAX_UPDATES)
PARAMETER_COUNTS = models.PARAMETER_COUNTS
ORIGINAL_FRESH_SHA256 = 'dad87d8598f7560e6f170ef048ba1bf515edeb03cef7ec42e16839a36af4ef93'
DIMENSIONS = dict(embedding_dim=128, encoder_layer_num=4, joint_layer_num=2,
    head_num=4, qkv_dim=32, ff_hidden_dim=512, head_hidden_dim=256, solver_feature_hidden=128)

CHECKPOINT_RULE = 'strict minimum whole18 VAL macro_vs_sbs_pct; first exact tie; common full-TRAIN SBS'
INFERENCE_RULE = 'original score MLP utility differences and maximin; no calibration or solver quotas'
VAL_BINDING_RULE = 'exact_selected_checkpoint_model_state_protocol_state_and_successful_update'
MODEL_FACTORY = 'code.V4.r107_experiment.load_checkpoint'
INITIALIZATION = 'scratch_seed2_common_plus_isolated_path_relations'
MODEL_ROLE_ARM, SELECTION_HEAD = 'B_execution_anchor', 'original_score_mlp_maximin'
WINNER_RULE = engine.WINNER_RULE
learning_rate, consistency_weight = engine.learning_rate, engine.consistency_weight
task_order, json_hash, status_update = engine.task_order, engine.json_hash, engine.status_update


def micro_caps(gpu_type='3090', large_cap=None):
    if gpu_type not in ('3090', '4090') or large_cap not in (None, 256):
        raise ValueError('R107 fixes TSP/CVRP256 and other tasks1024; no automatic shrink')
    return {p: 256 if p in ('TSP', 'CVRP') else 1024 for p in PROBLEMS}


def required_lock_fields():
    return dict(experiment='R107', locked=True, arms=list(ARMS), seed=2, task_scope='full18',
        rows_per_problem=dict(SPLIT_ROWS), batch_size=BATCH_SIZE, logical_global_batch=BATCH_SIZE,
        max_updates=MAX_UPDATES, warmup_updates=WARMUP_UPDATES, eval_every=EVAL_EVERY,
        train_eval_every=TRAIN_EVERY, large_micro_cap=256, other_micro_cap=1024, eval_batch_size=128,
        model_role_arm=MODEL_ROLE_ARM, role_feature_slot=3, embedding_dim=128,
        encoder_layer_num=4, joint_layer_num=2, head_num=4,
        path_spec=copy.deepcopy(models.PATH_SPEC), path_init_seed=PATH_INIT_SEED,
        trainable_parameters=PARAMETER_COUNTS.copy(),
        identical_complete_fresh_parameters=True, original_fresh_trainable_sha256=ORIGINAL_FRESH_SHA256,
        current_query_labels_in_forward=False, historical_TRAIN_label_context_used=False,
        external_pretraining=False, new_targets=False, auxiliary_loss=False, optimizer='AdamW',
        peak_lr=PEAK_LR, end_lr=END_LR, weight_decay=1e-4, grad_clip=1., dropout=.1,
        loss=dict(CE=.35, comparison=.35, relative_risk=.02, RDrop_final=.35),
        checkpoint_rule=CHECKPOINT_RULE, inference_rule=INFERENCE_RULE, R58_operations=False)

def protocol_lock(args):
    path = Path(args.protocol_lock or args.root / 'protocol_lock.json').expanduser().resolve()
    record = read_json(path)
    mismatches = [k for k, v in required_lock_fields().items() if record.get(k) != v]
    if mismatches:
        raise ValueError('R107 main scientific lock differs: ' + ', '.join(mismatches))
    for name in ('data_root', 'prepared_root'):
        if Path(record[name]).resolve() != Path(getattr(args, name)).resolve():
            raise ValueError('R107 main lock names another ' + name)
    if Path(record['primary_cache']).resolve() != Path(args.cache_root or DEFAULT_CACHE_ROOT).resolve():
        raise ValueError('R107 must reuse the main-locked R84 cache in place')
    return dict(path=str(path), sha256=file_hash(path), record=record)


def prepared_config(args):
    protocol_lock(args)
    if args.cache_root is None:
        args.cache_root = DEFAULT_CACHE_ROOT
    return engine.prepared_config(args)


make_model = models.make_model


def initialize_shared(initial, geometry, arm, device='cpu'):
    if arm not in ARMS or initial.get('trainable_sha256') != ORIGINAL_FRESH_SHA256:
        raise ValueError('R107 requires main-locked original fresh R84 seed2 tensors')
    model, params, receipt = models.initialize_shared(initial, geometry, arm, device)
    other = ARMS[1 - ARMS.index(arm)]
    control, _, _ = models.initialize_shared(initial, geometry, other, 'cpu')
    common_state = parameter_hash(model.state_dict())
    if (sum(p.numel() for p in model.parameters()) != PARAMETER_COUNTS[arm]
            or params.get('dropout') != .1 or common_state != parameter_hash(control.state_dict())):
        raise ValueError('R107 count, dropout or complete cross-arm fresh tensors differ')
    del control
    receipt.update(trainable_parameter_count=PARAMETER_COUNTS[arm],
        geometry_fitted_on='frozen R84 new81000 TRAIN only',
        identical_complete_fresh_parameters=True, common_fresh_model_state_sha256=common_state,
        cross_arm_complete_state_verified=True, path_init_seed=PATH_INIT_SEED,
        active_functional_freedom_matched=False,
        comparison_boundary='shared complete tensors, distinct factorized versus joint path computation')
    return model, params, receipt


def make_objective(selector, arm):
    if arm not in ARMS:
        raise ValueError('Unknown R107 arm')
    return RDropObjective(selector)


def checkpoint_extras(objective):
    if type(objective) is not RDropObjective:
        raise TypeError('R107 requires original stateless two-forward R-Drop')
    return {}


def restore_objective(objective, checkpoint):
    checkpoint_extras(objective)
    if 'objective_state' in checkpoint:
        raise ValueError('No mutable objective/teacher state belongs to R107')


def update_batch(objective, optimizer, scaler, batch, weight, runtime, cap, *, problem,
                 max_retries=20):
    if (problem not in PROBLEMS or len(batch['ind']) != BATCH_SIZE or cap != micro_caps()[problem]
            or any(key.startswith(('teacher_', 'donor_', 'nco_', 'route_target', 'context_')) for key in batch)):
        raise ValueError('R107 requires original logical1024 inputs and fixed microcaps')
    return private_engine().update_batch(objective, optimizer, scaler, batch, weight,
                                        runtime, cap, max_retries=max_retries)


def protocol(args, ready, manifest, geometry, baselines, params, order, caps):
    if args.large_micro_cap != 256 or args.eval_batch_size != 128 or caps != micro_caps(args.gpu_type, 256):
        raise ValueError('Both R107 arms require common256/1024/eval128 caps')
    config = engine.protocol(args, ready, manifest, geometry, baselines, params, order, caps)
    config['loss'].pop('focal_extra', None)
    config.update(experiment='R107', r107_schema_version=1, model_factory=MODEL_FACTORY,
        initialization=INITIALIZATION, main_protocol_lock=protocol_lock(args),
        model_spec=copy.deepcopy(models.SPEC), path_spec=copy.deepcopy(models.PATH_SPEC),
        path_init_seed=PATH_INIT_SEED, trainable_parameter_count=PARAMETER_COUNTS[args.arm],
        original_parameter_count=PARAMETER_COUNTS['original'],
        identical_complete_fresh_parameters=True, total_parameter_matched=True,
        active_parameter_matched=False, equal_flops_claim=False,
        train_eval_every_updates=TRAIN_EVERY, role_arm=MODEL_ROLE_ARM,
        input_roles=dict(problem='TSP', node_column=3, assignment='physical input node0', label_dependent=False),
        selection_head=SELECTION_HEAD, checkpoint_rule=CHECKPOINT_RULE, inference_rule=INFERENCE_RULE,
        objective=dict(classification='original R77 native CE', forward_passes=2,
            comparison=True, relative_risk=True, symmetric_rdrop=True, mutable_objective_state=False),
        external_pretraining=False, teacher_inputs=False, new_features=True, source_vectors=False,
        historical_TRAIN_label_context_used=False, route_targets=False, new_targets=False,
        auxiliary_loss=False, solver_quotas=False, routing_calibration=False,
        feature_origin='deterministic current raw inputs only; on-device directed path/constraint composition',
        cache_policy='direct frozen R84 reuse; no new feature cache and no solver calls',
        cache_input_node_maxima=dict(TSP=499, CVRP=500, other_tasks_at_most=101),
        oom_policy='fail without automatic cap shrink; main resolves witness before formal training',
        selected_val_binding_rule=VAL_BINDING_RULE,
        historical_references=['R89A original AdamW', 'R89B Muon', 'R84A'],
        evidence_boundary='Same complete tensors and total counts; factorized versus connected computation, not equal active functional freedom or FLOPs; historical references not rerun; TEST development-exposed',
        source_sha256={name: file_hash(V4 / name) for name in
            ('r107_experiment.py', 'r107_model.py', 'r107_reporting.py', 'r84_experiment.py',
             'r84_prepare.py', 'r88_model.py', 'pairwise_selector.py', 'pairwise_objective.py',
             'dual_stream.py', 'V4Model.py', 'r76_experiment.py', 'r77_experiment.py')})
    validate_checkpoint_config(config)
    return config


def validate_checkpoint_config(config):
    arm = config.get('arm')
    expected = dict(experiment='R107', r107_schema_version=1, model_factory=MODEL_FACTORY,
        initialization=INITIALIZATION, seed=2, solver_names=list(GLOBAL_SOLVERS), rows_per_problem=SPLIT_ROWS,
        global_batch_size=1024, logical_global_batch=1024, max_successful_updates=57600,
        warmup_updates=4320, eval_every_updates=1440, train_eval_every_updates=7200,
        eval_batch_size=128, microbatch_caps=micro_caps(), optimizer='AdamW',
        weight_decay=1e-4, grad_clip=1., dropout=.1, rdrop_weight=.35, rdrop_ramp_updates=4320,
        selection_head=SELECTION_HEAD, inference_rule=INFERENCE_RULE, checkpoint_rule=CHECKPOINT_RULE,
        role_arm=MODEL_ROLE_ARM, path_init_seed=PATH_INIT_SEED, model_spec=models.SPEC,
        path_spec=models.PATH_SPEC, loss=engine.LOSS_RECIPE, selected_val_binding_rule=VAL_BINDING_RULE,
        identical_complete_fresh_parameters=True, total_parameter_matched=True,
        active_parameter_matched=False, equal_flops_claim=False,
        external_pretraining=False, teacher_inputs=False, new_features=True, source_vectors=False,
        historical_TRAIN_label_context_used=False, route_targets=False, new_targets=False,
        auxiliary_loss=False, solver_quotas=False, routing_calibration=False)
    if arm not in ARMS or any(config.get(k) != v for k, v in expected.items()):
        raise ValueError('R107 full18 path/objective/execution protocol changed')
    record, params = config.get('main_protocol_lock', {}).get('record', {}), config.get('model_params', {})
    if (config.get('trainable_parameter_count') != PARAMETER_COUNTS[arm]
            or config.get('original_parameter_count') != PARAMETER_COUNTS['original']
            or any(params.get(k) != v for k, v in DIMENSIONS.items())
            or params.get('path_arm') != arm or params.get('path_spec') != models.PATH_SPEC
            or params.get('role_arm') != MODEL_ROLE_ARM or params.get('dropout') != .1
            or config.get('lr') != dict(rule='successful-update warmup then cosine, independent of VAL',
                peak=PEAK_LR, end=END_LR, warmup_updates=WARMUP_UPDATES, max_updates=MAX_UPDATES)
            or any(record.get(k) != v for k, v in required_lock_fields().items())):
        raise ValueError('R107 actual dimensions, LR or embedded main lock differ')
    for field, locked in (('data_root', 'data_root'), ('prepared_root', 'prepared_root'), ('cache_root', 'primary_cache')):
        if Path(config[field]).resolve() != Path(record[locked]).resolve():
            raise ValueError('R107 embedded paths differ: ' + field)

def load_checkpoint(path, device='cpu'):
    state = torch.load(path, map_location='cpu', weights_only=False)
    config = state['args']
    validate_checkpoint_config(config)
    if state.get('state_hash') != json_hash(config) or 'objective_state' in state:
        raise ValueError('R107 checkpoint protocol or stateless objective identity differs')
    model = make_model(config['model_params'], config['arm'], state['model'])
    if (sum(p.numel() for p in model.parameters()) != PARAMETER_COUNTS[config['arm']]
            or parameter_hash(model.state_dict()) != parameter_hash(state['model'])):
        raise ValueError('R107 factory must restore the complete model and buffers exactly')
    return model.to(device), state


load_model = from_checkpoint = load_checkpoint


def selected_capture_proof(run, summary, lock, split, state=None):
    """Null capture file SHA is safe only with this exact, immutable-state proof."""
    run, checkpoint = Path(run), Path(run) / 'best.pt'
    if file_hash(checkpoint) != lock['checkpoint_sha256']:
        raise ValueError('Selected capture requires the exact immutable checkpoint file')
    if state is None:
        state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    capture = copy.deepcopy(summary.get('archive_capture_identity', summary['identity']))
    path = Path(summary['prediction_archive']).resolve()
    if not path.is_relative_to((run / 'evaluation' / split).resolve()):
        raise ValueError('Selected capture archive is outside its own run/split')
    archived = read_json(path.parent / 'identity.json')
    model_hash = parameter_hash(state['model'])
    if (capture != archived or capture.get('experiment') != 'R107'
            or capture.get('arm') != lock.get('arm') or lock.get('arm') not in ARMS
            or state['args'].get('experiment') != 'R107' or state['args'].get('arm') != lock['arm']
            or capture.get('split') != split
            or capture.get('checkpoint_sha256') not in (None, lock['checkpoint_sha256'])
            or capture.get('model_state_sha256') != model_hash
            or capture.get('state_hash') != lock['state_hash']
            or state.get('state_hash') != lock['state_hash'] or json_hash(state['args']) != lock['state_hash']
            or state.get('successful_updates') != lock['selected_update']
            or capture.get('successful_updates') != lock['selected_update']):
        raise ValueError('Selected capture must exactly match model-state, protocol-state and update')
    return capture, dict(rule=VAL_BINDING_RULE, checkpoint_sha256=lock['checkpoint_sha256'],
        model_state_sha256=model_hash, state_hash=lock['state_hash'], selected_update=lock['selected_update'],
        archive_capture_checkpoint_sha256=capture.get('checkpoint_sha256'), archive_identity_unchanged=True)


def bind_selected_val(run, final):
    """Bind the official summary, never rewrite capture identities or archives."""
    lock = read_json(Path(run) / 'final_test_lock.json')
    if (final.get('state_hash') != lock['state_hash'] or final.get('selected_update') != lock['selected_update']
            or final.get('checkpoint_sha256') != lock['checkpoint_sha256']):
        raise ValueError('Final summary differs from the unique VAL checkpoint lock')
    result = copy.deepcopy(final)
    val = result['val']
    capture, binding = selected_capture_proof(run, val, lock, 'val')
    official = dict(capture, checkpoint_sha256=lock['checkpoint_sha256'])
    if ('checkpoint_binding' in val and val['checkpoint_binding'] != binding
            or val['identity'] not in (capture, official)):
        raise ValueError('Existing selected VAL binding differs from its capture proof')
    val.update(archive_capture_identity=capture, identity=official, checkpoint_binding=binding)
    return result


def _engine_write_json(path, value):
    path = Path(path)
    if path.name == 'final_metrics.json' and 'val' in value:
        value = bind_selected_val(path.parent, value)
    return write_json(path, value)


def private_module(source, name):
    spec = importlib.util.spec_from_file_location('code.V4._r107_' + name, source.__file__)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@lru_cache(maxsize=1)
def private_engine():
    module = private_module(engine, 'private_engine')
    # Both train and evaluate resolve these in their own module globals.
    module.micro_caps, module.write_json = micro_caps, _engine_write_json
    return module


def evaluate(*args, **kwargs):
    return private_engine().evaluate(*args, **kwargs)


def authorize_test(args):
    lock, config = private_engine().authorize_test(args)
    validate_checkpoint_config(config)
    if protocol_lock(args) != config['main_protocol_lock'] or lock['inference_rule'] != INFERENCE_RULE:
        raise PermissionError('R107 TEST differs from the unique VAL/main/inference lock')
    if Path(args.cache_root or config['cache_root']).resolve() != Path(config['cache_root']).resolve():
        raise PermissionError('R107 TEST cannot change the frozen cache')
    return lock, config


def ensure_test_cache(args, lock):
    verified, config = authorize_test(args)
    if verified != lock:
        raise PermissionError('R107 TEST cache request uses another lock')
    manifest = read_json(Path(config['prepared_root']) / 'source_cache_manifest.json')
    tasks = manifest.get('splits', {}).get('test', {})
    if set(tasks) != set(PROBLEMS) or any(tasks[p].get('rows') != SPLIT_ROWS['test']
                                        or tasks[p].get('complete') is not True for p in PROBLEMS):
        raise PermissionError('Complete preexisting full18 R84 TEST cache required; never build/copy')
    return manifest


def train(args, device):
    protocol_lock(args)
    return private_engine().train(args, device, integration=sys.modules[__name__])


def test(args, device):
    return private_engine().test(args, device, integration=sys.modules[__name__])


def raw_inputs(batch):
    names = ('node', 'matrix', 'node_mask', 'n', 'node_geom', 'kind', 'problem_id',
             'coord_dist', 'cbits', 'problem_desc', 'd_in', 'pool_ids', 'solver_mask', 'mask')
    return {key: batch[key] for key in names if key in batch}


def cached_input_batch(config, manifest, split, problem, device, count=128):
    """Read only frozen input arrays, without opening supervision or targets."""
    from .r75_cache import _close, _open_arrays
    from .r76_cache import _validate_record
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
        raise ValueError('R107 input-only cache provenance differs')
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
            raise ValueError('Nonfinite R107 timing prediction')
    return dict(batch_seconds=samples,
        mean_seconds_per_instance=sum(samples) / (repetitions * len(raw['node_mask'])))


@torch.no_grad()
def timing(args, device):
    if device.type != 'cuda':
        raise ValueError('R107 standalone timing requires the launcher-assigned CUDA device')
    lock, config = authorize_test(args)
    run = args.root / args.arm
    record = read_json(run / 'test_results.json')
    if (record.get('complete') is not True or record.get('checkpoint_sha256') != lock['checkpoint_sha256']
            or record.get('state_hash') != lock['state_hash']):
        raise PermissionError('R107 timing requires completed locked full TEST')
    destination = run / 'inference_timing.json'
    if destination.exists():
        from .r107_reporting import inference_timing
        return inference_timing(run, lock)
    model, state = load_checkpoint(run / 'best.pt', device)
    model_hash = parameter_hash(state['model'])
    if model_hash != record['result']['identity']['model_state_sha256']:
        raise ValueError('R107 timing model differs from locked TEST')
    del state
    model.eval()
    manifest = ensure_test_cache(args, lock)
    rows = []
    for problem in PROBLEMS:
        raw = cached_input_batch(config, manifest, 'test', problem, device)
        rows.append(dict(problem=problem, instances=128, standalone_model=measure_inputs(model, raw, device)))
    if file_hash(run / 'best.pt') != lock['checkpoint_sha256'] or parameter_hash(model.state_dict()) != model_hash:
        raise ValueError('Timing changed the locked R107 checkpoint/model')
    value = dict(experiment='R107', arm=args.arm, complete=True, completed_utc=utc(),
        checkpoint_sha256=lock['checkpoint_sha256'], state_hash=lock['state_hash'],
        selected_update=lock['selected_update'], model_state_sha256=model_hash, model_factory=MODEL_FACTORY,
        inference_rule=INFERENCE_RULE, device=str(device), repetitions=3, warmup_batches=1,
        sample_policy='first128 frozen TEST inputs per problem', parameter_count=PARAMETER_COUNTS[args.arm],
        original_parameter_count=PARAMETER_COUNTS['original'],
        scope='complete_standalone_model_original_GPU_resident_inputs',
        all_current_model_included=True, explicit_path_computation_included=True,
        donor_present=False, cached_feature_latency_used=False, end_to_end_measured=False,
        excludes='dataset I/O, CPU-to-GPU transfer, training and solver execution',
        input_labels_read=False, per_problem=rows, model_reselection=False, routing_calibration=False)
    write_json(destination, value)
    return value


def describe(args):
    return dict(experiment='R107', arm=args.arm, required_protocol_lock=required_lock_fields(),
        model_spec=models.SPEC, path_spec=models.PATH_SPEC, trainable_parameters=PARAMETER_COUNTS[args.arm],
        microbatch_caps=micro_caps(), eval_batch_size=128, cache_root=str(args.cache_root or DEFAULT_CACHE_ROOT),
        model_factory=MODEL_FACTORY, inference_rule=INFERENCE_RULE,
        wandb_mode=args.wandb_mode)


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('command', choices=('train', 'test', 'timing', 'describe', 'status'))
    result.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    result.add_argument('--arm', choices=ARMS, required=True)
    result.add_argument('--prepared-root', type=Path, default=DEFAULT_PREPARED_ROOT)
    result.add_argument('--data-root', type=Path, default=DEFAULT_DATA_ROOT)
    result.add_argument('--cache-root', type=Path, default=DEFAULT_CACHE_ROOT)
    result.add_argument('--protocol-lock', type=Path)
    result.add_argument('--device', default='cuda:0')
    result.add_argument('--gpu-type', choices=('3090', '4090'), default='3090')
    result.add_argument('--large-micro-cap', type=int, choices=(256,), default=256)
    result.add_argument('--eval-batch-size', type=int, choices=(128,), default=128)
    result.add_argument('--log-every', type=int, default=18)
    result.add_argument('--wandb-mode', choices=('auto', 'online', 'offline'), default='offline')
    result.add_argument('--resume', action='store_true')
    result.add_argument('--state-hash', '--expected-state-hash')
    result.set_defaults(experiment='R107')
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    for name in ('root', 'prepared_root', 'data_root', 'cache_root', 'protocol_lock'):
        value = getattr(args, name)
        if value is not None:
            setattr(args, name, value.expanduser().resolve())
    if args.command in ('describe', 'status'):
        value = describe(args) if args.command == 'describe' else (
            read_json(args.root / args.arm / 'status.json')
            if (args.root / args.arm / 'status.json').exists() else dict(experiment='R107', phase='not_started'))
        print(json.dumps(value, indent=2))
        return value
    if args.command in ('train', 'test'):
        return private_engine().main(argv, integration=sys.modules[__name__])
    run = args.root / args.arm
    run.mkdir(parents=True, exist_ok=True)
    with (run / 'runner.lock').open('a') as claim:
        fcntl.flock(claim, fcntl.LOCK_EX | fcntl.LOCK_NB)
        torch.set_num_threads(2)
        engine.configure_torch()
        return timing(args, torch.device(args.device))


if __name__ == '__main__':
    main()

