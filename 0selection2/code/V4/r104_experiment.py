"""Fresh full18 R104 capacity axes via an isolated, original R84 train engine."""

import argparse
import copy
import fcntl
import importlib.util
import json
import sys
import time
from functools import lru_cache
from pathlib import Path

import torch

from code.unified_selector.registry import GLOBAL_SOLVERS, PROBLEMS
from . import r84_experiment as engine
from . import r104_model as models
from .r76_experiment import RDropObjective, file_hash, read_json, utc, write_json
from .r77_data import parameter_hash
from .r84_prepare import DEFAULT_DATA_ROOT, DEFAULT_PREPARED_ROOT, SPLIT_ROWS
from .r85_experiment import selection_counts, zero_summary


V4 = Path(__file__).resolve().parent
DEFAULT_ROOT = V4 / 'runs/R104_capacity_scaling'
DEFAULT_CACHE_ROOT = Path('/tmp/shiys_R84_new90k_cache')
ARMS = NEURAL_ARMS = models.ARMS
BATCH_SIZE, MAX_UPDATES, WARMUP_UPDATES, EVAL_EVERY = 1024, 57600, 4320, 1440
TRAIN_EVERY, PEAK_LR, END_LR = 7200, 1e-4, 1e-6
BUDGETS = dict.fromkeys(ARMS, MAX_UPDATES)
PARAMETER_COUNTS = dict(original=1842953, A_width256=7288969, B_depth8=3426057)
COPIED_COUNTS = dict(A_width256=1417, B_depth8=1842953)
MODEL_FACTORY = 'code.V4.r104_experiment.load_checkpoint'
INITIALIZATION = 'scratch_seed2_common_plus_isolated_capacity'
MODEL_ROLE_ARM, SELECTION_HEAD = models.ROLE_ARM, models.SPEC['selection_head']
WINNER_RULE = engine.WINNER_RULE
learning_rate, consistency_weight = engine.learning_rate, engine.consistency_weight
task_order, json_hash, status_update = engine.task_order, engine.json_hash, engine.status_update
INFERENCE_RULE = ('eval mode, complete R104 capacity model; original R88B physical TSP node0 '
    'slot3, methodID+34, synchronous two-way interaction, original score MLP, FP32 utility '
    'differences/maximin; stable ascending-global-solver-ID ties; original inputs only')
VAL_BINDING_RULE = 'exact_selected_checkpoint_model_state_protocol_state_and_successful_update'


def micro_caps(gpu_type='3090', large_cap=None):
    if gpu_type not in ('3090', '4090') or large_cap not in (None, 128):
        raise ValueError('R104 fixes TSP/CVRP128 and every other task512; no automatic shrink')
    return {p: 128 if p in ('TSP', 'CVRP') else 512 for p in PROBLEMS}


def required_lock_fields():
    """Subset of the existing main-owned scientific lock required by this runner."""
    return dict(experiment='R104', locked=True, arms=list(ARMS), seed=2, task_scope='full18',
        rows_per_problem=dict(SPLIT_ROWS), batch_size=BATCH_SIZE, logical_global_batch=BATCH_SIZE,
        max_updates=MAX_UPDATES, warmup_updates=WARMUP_UPDATES, eval_every=EVAL_EVERY,
        train_eval_every=TRAIN_EVERY, large_micro_cap=128, other_micro_cap=512, eval_batch_size=128,
        capacities=copy.deepcopy(models.CAPACITIES), trainable_parameters=PARAMETER_COUNTS.copy(),
        capacity_init_seed=models.CAPACITY_INIT_SEED, model_role_arm=MODEL_ROLE_ARM, role_feature_slot=3,
        current_query_labels_in_forward=False, historical_TRAIN_label_context_used=False,
        external_pretraining=False, optimizer='AdamW', peak_lr=PEAK_LR, end_lr=END_LR,
        weight_decay=1e-4, grad_clip=1., dropout=.1,
        loss=dict(CE=.35, comparison=.35, relative_risk=.02, RDrop_final=.35))


def protocol_lock(args):
    path = Path(args.protocol_lock or args.root / 'protocol_lock.json').expanduser().resolve()
    record = read_json(path)
    if any(record.get(key) != value for key, value in required_lock_fields().items()):
        raise ValueError('R104 main scientific lock differs from the fixed capacity/full18 controls')
    for name in ('data_root', 'prepared_root'):
        if Path(record[name]).resolve() != Path(getattr(args, name)).resolve():
            raise ValueError('R104 main lock names another ' + name)
    if Path(record['primary_cache']).resolve() != Path(args.cache_root or DEFAULT_CACHE_ROOT).resolve():
        raise ValueError('R104 must directly reuse the main-locked R84 cache, without copying')
    return dict(path=str(path), sha256=file_hash(path), record=record)


def prepared_config(args):
    protocol_lock(args)
    if args.cache_root is None:
        args.cache_root = DEFAULT_CACHE_ROOT
    return engine.prepared_config(args)


make_model = models.make_model


def initialize_shared(initial, geometry, arm, device='cpu'):
    model, params, receipt = models.initialize_shared(initial, geometry, arm, device)
    if (sum(p.numel() for p in model.parameters()) != PARAMETER_COUNTS[arm]
            or params['dropout'] != .1):
        raise ValueError('R104 instantiated capacity/dropout differs from the fixed model')
    receipt.update(trainable_parameter_count=PARAMETER_COUNTS[arm],
        geometry_fitted_on='frozen R84 new81000 TRAIN only', identical_full_initialization=False)
    return model, params, receipt


def make_objective(selector, arm):
    if arm not in ARMS:
        raise ValueError('Unknown R104 arm')
    return RDropObjective(selector)


def checkpoint_extras(objective):
    if type(objective) is not RDropObjective:
        raise TypeError('R104 requires the original stateless two-view RDropObjective')
    return {}


def restore_objective(objective, checkpoint):
    checkpoint_extras(objective)
    if 'objective_state' in checkpoint:
        raise ValueError('No mutable objective/teacher state belongs to R104')


def update_batch(objective, optimizer, scaler, batch, weight, runtime, cap, *, problem,
                 max_retries=20):
    if (problem not in PROBLEMS or len(batch['ind']) != BATCH_SIZE
            or cap != micro_caps()[problem]
            or any(key.startswith(('teacher_', 'donor_', 'nco_', 'route_target')) for key in batch)):
        raise ValueError('R104 requires original logical1024 inputs and the fixed task microcap')
    return private_engine().update_batch(objective, optimizer, scaler, batch, weight,
                                        runtime, cap, max_retries=max_retries)


def protocol(args, ready, manifest, geometry, baselines, params, order, caps):
    if args.large_micro_cap != 128 or args.eval_batch_size != 128 or caps != micro_caps(args.gpu_type, 128):
        raise ValueError('Both R104 capacity axes require the same128/512/eval128 execution caps')
    config = engine.protocol(args, ready, manifest, geometry, baselines, params, order, caps)
    config['loss'].pop('focal_extra', None)
    config.update(experiment='R104', r104_schema_version=1, model_factory=MODEL_FACTORY,
        initialization=INITIALIZATION, main_protocol_lock=protocol_lock(args),
        model_spec=copy.deepcopy(models.SPEC), capacity=copy.deepcopy(models.CAPACITIES[args.arm]),
        capacity_init_seed=models.CAPACITY_INIT_SEED, trainable_parameter_count=PARAMETER_COUNTS[args.arm],
        original_parameter_count=PARAMETER_COUNTS['original'], copied_parameter_count=COPIED_COUNTS[args.arm],
        independent_capacity_parameter_count=PARAMETER_COUNTS[args.arm] - COPIED_COUNTS[args.arm],
        identical_full_initialization=False, parameter_matched=False, equal_flops=False,
        train_eval_every_updates=TRAIN_EVERY, role_arm=MODEL_ROLE_ARM,
        input_roles=dict(problem='TSP', node_column=3, assignment='physical input node0', label_dependent=False),
        selection_head=SELECTION_HEAD, inference_rule=INFERENCE_RULE,
        objective=dict(classification='original R77 native CE', forward_passes=2,
            comparison=True, relative_risk=True, symmetric_rdrop=True, mutable_objective_state=False),
        external_pretraining=False, teacher_inputs=False, new_features=False, route_targets=False,
        new_targets=False, auxiliary_loss=False, cache_policy='direct frozen R84 reuse; never build or copy',
        cache_input_node_maxima=dict(TSP=499, CVRP=500, other_tasks_at_most=101),
        oom_policy='fail without automatic cap shrink; main resolves witness before formal training',
        selected_val_binding_rule=VAL_BINDING_RULE,
        historical_references=['R89A original AdamW', 'R89B Muon', 'R84A'],
        evidence_boundary='Two capacity axes, not same-parameter/equal-FLOPs/identical-full-init controls; historical references not rerun; TEST development-exposed',
        source_sha256={name: file_hash(V4 / name) for name in
            ('r104_experiment.py', 'r104_model.py', 'r104_reporting.py', 'r84_experiment.py',
             'r84_prepare.py', 'r88_model.py', 'pairwise_selector.py', 'pairwise_objective.py',
             'dual_stream.py', 'V4Model.py', 'r76_experiment.py', 'r77_experiment.py')})
    validate_checkpoint_config(config)
    return config


def validate_checkpoint_config(config):
    arm = config.get('arm')
    expected = dict(experiment='R104', r104_schema_version=1, model_factory=MODEL_FACTORY,
        initialization=INITIALIZATION, seed=2, solver_names=list(GLOBAL_SOLVERS), rows_per_problem=SPLIT_ROWS,
        global_batch_size=1024, logical_global_batch=1024, max_successful_updates=57600,
        warmup_updates=4320, eval_every_updates=1440, train_eval_every_updates=7200,
        eval_batch_size=128, microbatch_caps=micro_caps(), optimizer='AdamW',
        weight_decay=1e-4, grad_clip=1., dropout=.1, rdrop_weight=.35, rdrop_ramp_updates=4320,
        selection_head=SELECTION_HEAD, inference_rule=INFERENCE_RULE, role_arm=MODEL_ROLE_ARM,
        capacity_init_seed=models.CAPACITY_INIT_SEED, model_spec=models.SPEC,
        loss=engine.LOSS_RECIPE, selected_val_binding_rule=VAL_BINDING_RULE,
        identical_full_initialization=False, parameter_matched=False, equal_flops=False,
        external_pretraining=False, teacher_inputs=False, new_features=False, route_targets=False,
        new_targets=False, auxiliary_loss=False)
    if arm not in ARMS or any(config.get(k) != v for k, v in expected.items()):
        raise ValueError('R104 full18 capacity/objective/execution protocol changed')
    if (config.get('capacity') != models.CAPACITIES[arm]
            or config.get('trainable_parameter_count') != PARAMETER_COUNTS[arm]
            or any(config.get('model_params', {}).get(k) != v for k, v in models.CAPACITIES[arm].items())
            or config['model_params'].get('role_arm') != MODEL_ROLE_ARM
            or config.get('lr') != dict(rule='successful-update warmup then cosine, independent of VAL',
                peak=PEAK_LR, end=END_LR, warmup_updates=WARMUP_UPDATES, max_updates=MAX_UPDATES)
            or any(config.get('main_protocol_lock', {}).get('record', {}).get(k) != v
                   for k, v in required_lock_fields().items())):
        raise ValueError('R104 actual dimensions, LR or embedded main lock differ')


def load_checkpoint(path, device='cpu'):
    state = torch.load(path, map_location='cpu', weights_only=False)
    config = state['args']
    validate_checkpoint_config(config)
    if state.get('state_hash') != json_hash(config) or 'objective_state' in state:
        raise ValueError('R104 checkpoint protocol or stateless loss identity differs')
    return make_model(config['model_params'], config['arm'], state['model']).to(device), state


load_model = from_checkpoint = load_checkpoint


def bind_selected_val(run, final):
    """Bind only the final summary; never rewrite capture-time null archive SHAs."""
    lock = read_json(run / 'final_test_lock.json')
    checkpoint = run / 'best.pt'
    if file_hash(checkpoint) != lock['checkpoint_sha256']:
        raise ValueError('Selected VAL binding requires the exact immutable checkpoint')
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    model_hash = parameter_hash(state['model'])
    result = copy.deepcopy(final)
    val = result['val']
    capture = copy.deepcopy(val.get('archive_capture_identity', val['identity']))
    archived = read_json(Path(val['prediction_archive']).parent / 'identity.json')
    if (capture != archived or capture.get('experiment') != 'R104'
            or capture.get('arm') != lock['arm'] or capture.get('split') != 'val'
            or capture.get('checkpoint_sha256') not in (None, lock['checkpoint_sha256'])
            or capture.get('model_state_sha256') != model_hash
            or capture.get('state_hash') != lock['state_hash']
            or state.get('state_hash') != lock['state_hash']
            or json_hash(state['args']) != lock['state_hash']
            or state.get('successful_updates') != lock['selected_update']
            or capture.get('successful_updates') != lock['selected_update']
            or result.get('state_hash') != lock['state_hash']
            or result.get('selected_update') != lock['selected_update']
            or result.get('checkpoint_sha256') != lock['checkpoint_sha256']):
        raise ValueError('Selected VAL summary must exactly match model-state, protocol-state and update')
    val['archive_capture_identity'] = capture
    val['identity'] = dict(capture, checkpoint_sha256=lock['checkpoint_sha256'])
    val['checkpoint_binding'] = dict(rule=VAL_BINDING_RULE, checkpoint_sha256=lock['checkpoint_sha256'],
        model_state_sha256=model_hash, state_hash=lock['state_hash'], selected_update=lock['selected_update'],
        archive_capture_checkpoint_sha256=capture.get('checkpoint_sha256'), archive_identity_unchanged=True)
    return result


def _engine_write_json(path, value):
    path = Path(path)
    if path.name == 'final_metrics.json' and 'val' in value:
        value = bind_selected_val(path.parent, value)
    return write_json(path, value)


@lru_cache(maxsize=1)
def private_engine():
    name = 'code.V4._r104_private_engine'
    spec = importlib.util.spec_from_file_location(name, engine.__file__)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    # R84 reads these globals inside BOTH train() and evaluate(), not hooks.
    module.micro_caps, module.write_json = micro_caps, _engine_write_json
    return module


def evaluate(*args, **kwargs):
    return private_engine().evaluate(*args, **kwargs)


def authorize_test(args):
    lock, config = private_engine().authorize_test(args)
    validate_checkpoint_config(config)
    if protocol_lock(args) != config['main_protocol_lock'] or lock['inference_rule'] != INFERENCE_RULE:
        raise PermissionError('R104 TEST differs from the unique VAL/main/inference lock')
    if Path(args.cache_root or config['cache_root']).resolve() != Path(config['cache_root']).resolve():
        raise PermissionError('R104 TEST cannot change the frozen cache')
    return lock, config


def ensure_test_cache(args, lock):
    """Consume only already complete frozen R84 TEST arrays; never prepare them."""
    verified, config = authorize_test(args)
    if verified != lock:
        raise PermissionError('R104 TEST cache request uses another lock')
    manifest = read_json(Path(config['prepared_root']) / 'source_cache_manifest.json')
    tasks = manifest.get('splits', {}).get('test', {})
    if set(tasks) != set(PROBLEMS) or any(tasks[p].get('rows') != SPLIT_ROWS['test']
                                        or tasks[p].get('complete') is not True for p in PROBLEMS):
        raise PermissionError('Preexisting complete full18 R84 TEST cache required; no automatic preparation')
    return manifest


def train(args, device):
    protocol_lock(args)
    return private_engine().train(args, device, integration=sys.modules[__name__])


def test(args, device):
    return private_engine().test(args, device, integration=sys.modules[__name__])


def raw_inputs(batch):
    names = ('node', 'matrix', 'node_mask', 'n', 'node_geom', 'kind', 'problem_id',
             'coord_dist', 'cbits', 'problem_desc', 'd_in', 'pool_ids', 'solver_mask')
    return {key: batch[key] for key in names if key in batch}


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
            raise ValueError('Nonfinite R104 timing prediction')
    return dict(batch_seconds=samples,
        mean_seconds_per_instance=sum(samples) / (repetitions * len(raw['node_mask'])))


@torch.no_grad()
def timing(args, device):
    lock, config = authorize_test(args)
    run = args.root / args.arm
    test_record = read_json(run / 'test_results.json')
    if (test_record.get('complete') is not True or test_record.get('checkpoint_sha256') != lock['checkpoint_sha256']
            or test_record.get('state_hash') != lock['state_hash']):
        raise PermissionError('R104 timing requires completed locked TEST')
    destination = run / 'inference_timing.json'
    if destination.exists():
        from .r104_reporting import inference_timing
        return inference_timing(run, lock)
    model, state = load_checkpoint(run / 'best.pt', device)
    if parameter_hash(model.state_dict()) != test_record['result']['identity']['model_state_sha256']:
        raise ValueError('R104 timing model differs from locked TEST state')
    del state
    model.eval()
    manifest = ensure_test_cache(args, lock)
    rows = []
    for problem in PROBLEMS:
        cache = private_engine().open_cache(Path(config['cache_root']), 'test', problem,
                                            manifest, Path(config['data_root']))
        try:
            if cache.size < 128:
                raise ValueError('R104 timing needs first128 original TEST inputs per problem')
            batch = cache.batch(torch.arange(128), device=device)
            rows.append(dict(problem=problem, instances=128, standalone_model=measure_inputs(model, batch, device)))
        finally:
            cache.close()
    value = dict(experiment='R104', arm=args.arm, complete=True, completed_utc=utc(),
        checkpoint_sha256=lock['checkpoint_sha256'], state_hash=lock['state_hash'],
        inference_rule=INFERENCE_RULE, device=str(device), repetitions=3, warmup_batches=1,
        parameter_count=PARAMETER_COUNTS[args.arm], original_parameter_count=PARAMETER_COUNTS['original'],
        scope='complete_standalone_model_original_GPU_resident_inputs',
        all_current_model_included=True, donor_present=False, cached_feature_latency_used=False,
        end_to_end_measured=False, excludes='dataset I/O, host transfer, training and solver execution',
        per_problem=rows, model_reselection=False)
    write_json(destination, value)
    return value


def describe(args):
    return dict(experiment='R104', arm=args.arm, required_protocol_lock=required_lock_fields(),
        capacity=models.CAPACITIES[args.arm], model_spec=models.SPEC,
        trainable_parameters=PARAMETER_COUNTS[args.arm], microbatch_caps=micro_caps(),
        eval_batch_size=128, cache_root=str(args.cache_root or DEFAULT_CACHE_ROOT),
        model_factory=MODEL_FACTORY, inference_rule=INFERENCE_RULE)


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
    result.add_argument('--large-micro-cap', type=int, choices=(128,), default=128)
    result.add_argument('--eval-batch-size', type=int, choices=(128,), default=128)
    result.add_argument('--log-every', type=int, default=18)
    result.add_argument('--wandb-mode', choices=('auto', 'online', 'offline'), default='offline')
    result.add_argument('--resume', action='store_true')
    result.add_argument('--state-hash', '--expected-state-hash')
    result.set_defaults(experiment='R104')
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
            if (args.root / args.arm / 'status.json').exists() else dict(phase='not_started'))
        print(json.dumps(value, indent=2))
        return value
    if args.command in ('train', 'test'):
        return private_engine().main(argv, integration=sys.modules[__name__])
    run = args.root / args.arm
    run.mkdir(parents=True, exist_ok=True)
    with (run / 'runner.lock').open('a') as claim:
        fcntl.flock(claim, fcntl.LOCK_EX | fcntl.LOCK_NB)
        device = torch.device(args.device)
        if device.type != 'cuda':
            raise ValueError('Formal R104 timing requires the launcher-assigned CUDA device')
        torch.set_num_threads(2)
        engine.configure_torch()
        return timing(args, device)


if __name__ == '__main__':
    main()
