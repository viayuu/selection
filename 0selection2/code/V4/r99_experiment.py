"""Matched full18 input competition, using a private unchanged R84 train loop.

The main-owned lock freezes the experiment, not this runner. Feature roots are
arm-local runtime locations; all other initialization/training controls match.
Online latency starts from original GPU-resident inputs, never cached messages.
"""

import argparse
import copy
import fcntl
import importlib.util
import json
import math
import sys
import time
from functools import lru_cache
from pathlib import Path

import torch

from code.unified_selector.registry import GLOBAL_SOLVERS, PROBLEMS
from . import r84_experiment as engine
from .r76_experiment import PART_KEYS, file_hash, read_json, utc, write_json
from .r77_data import parameter_hash
from .r84_prepare import DEFAULT_DATA_ROOT, DEFAULT_PREPARED_ROOT, SPLIT_ROWS
from .r85_experiment import ensure_test_cache, selection_counts, zero_summary


V4 = Path(__file__).resolve().parent
DEFAULT_ROOT = V4 / 'runs/R99_global_competition_input'
ARMS = NEURAL_ARMS = ('A', 'B')
VARIANTS = dict(A='row', B='balanced')
BATCH_SIZE, MAX_UPDATES = engine.BATCH_SIZE, engine.MAX_UPDATES
WARMUP_UPDATES, EVAL_EVERY = engine.WARMUP_UPDATES, engine.EVAL_EVERY
TRAIN_EVERY, PROJECTION_SEED = 7200, 990003
PEAK_LR, END_LR = engine.PEAK_LR, engine.END_LR
BUDGETS = dict.fromkeys(ARMS, MAX_UPDATES)
WINNER_RULE = engine.WINNER_RULE
MODEL_FACTORY = 'code.V4.r99_experiment.load_checkpoint'
INITIALIZATION = 'scratch_shared_r77_plus_input_competition'
MODEL_ROLE_ARM, ROLE_SLOT = 'B_execution_anchor', 3
NEW_PARAMETER = 'instance_encoder.relation_projection.weight'
SELECTION_HEAD = 'original_score_mlp_maximin'
prepared_config, micro_caps = engine.prepared_config, engine.micro_caps
json_hash, task_order, learning_rate = engine.json_hash, engine.task_order, engine.learning_rate
status_update = engine.status_update
INFERENCE_RULE = (
    'eval input-only distance messages, A row-softmax or B fixed64 Sinkhorn; '
    'FP32 transform then FP16 quantization; outgoing/incoming one/two-hop raw8; '
    'Linear32to128 before original four instance layers; R88B physical TSP node0; '
    'hand34 methodID, joint2, original score MLP/pair differences/maximin; '
    'stable ascending-global-solver-ID ties; all18 including directed ATSP; no calibration')
TIMING_SCOPES = ('transform_only', 'transform_plus_selector', 'cache_hit_selector')


def feature_root(args):
    value = getattr(args, 'feature_root', None)
    return Path(value).expanduser() if value is not None else Path('/tmp/shiys_R99_' + args.arm + '_relation_features')


def required_lock_fields():
    """Exact main-owned lock schema; extra provenance keys are allowed."""
    return dict(experiment='R99', locked=True, arms=list(ARMS), seed=2,
        task_scope='full18', rows_per_problem=dict(SPLIT_ROWS), batch_size=BATCH_SIZE,
        logical_global_batch=BATCH_SIZE, max_updates=MAX_UPDATES,
        warmup_updates=WARMUP_UPDATES, eval_every=EVAL_EVERY,
        train_eval_every=TRAIN_EVERY, large_micro_cap=352, eval_batch_size=256,
        relation_variants=VARIANTS.copy(), projection_seed=PROJECTION_SEED)


def protocol_lock(args):
    path = (getattr(args, 'protocol_lock', None) or args.root / 'protocol_lock.json').expanduser().resolve()
    record = read_json(path)
    if any(record.get(key) != value for key, value in required_lock_fields().items()):
        raise ValueError('R99 requires the final matched input-only full18 protocol lock')
    for name in ('data_root', 'prepared_root'):
        if not record.get(name) or Path(record[name]).resolve() != getattr(args, name).resolve():
            raise ValueError('R99 lock names a different ' + name)
    return dict(path=str(path.resolve()), sha256=file_hash(path), record=record)


def objective_spec(arm):
    if arm not in ARMS:
        raise ValueError('Unknown R99 arm')
    return dict(main='unchanged R77 CE/comparison/risk plus symmetric R-Drop',
        classification='original R77 CE .35 / comparison .35 / risk .02 / R-Drop .35',
        coefficients=dict(ce=.35, comparison=.35, risk=.02),
        rdrop='linear ramp to .35 over4320 updates', forward_passes=2,
        new_target=False, auxiliary_loss=False, route_targets=False,
        cross_instance_objective=False, mutable_objective_state=False)


def initialize_shared(initial, geometry, arm, device='cpu'):
    from .r99_model import make_model
    reference = initial['trainable']
    if (arm not in ARMS or initial.get('seed') != 2
            or initial.get('initialization') != 'scratch'
            or parameter_hash(reference) != initial.get('trainable_sha256')):
        raise ValueError('R99 requires the frozen fresh R84 seed2 initialization, never trained weights')
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(2)
        model = make_model(initial['model_params'], arm)
    actual = dict(model.named_parameters())
    retained = {name: value for name, value in actual.items()
                if name in reference and value.shape == reference[name].shape}
    added = {name: value for name, value in actual.items() if name not in retained}
    if (set(retained) != set(reference) or set(added) != {NEW_PARAMETER}
            or any(not value.requires_grad for value in actual.values())):
        raise ValueError('R99 must retain all original trainables and add only relation_projection.weight')
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(PROJECTION_SEED)
        expected = torch.nn.Linear(32, initial['model_params']['embedding_dim'], bias=False).weight
    if not torch.equal(added[NEW_PARAMETER].detach().cpu(), expected.detach()):
        raise ValueError('R99 shared new projection must use seed990003')
    with torch.no_grad():
        for name, value in retained.items():
            value.copy_(reference[name])
        branch = model.instance_encoder.geometry_residual
        branch.mean.copy_(torch.as_tensor(geometry['mean'], dtype=branch.mean.dtype))
        branch.std.copy_(torch.as_tensor(geometry['std'], dtype=branch.std.dtype))
    common_hash, new_hash, full_hash = map(parameter_hash, (retained, added, actual))
    if common_hash != initial['trainable_sha256']:
        raise ValueError('Original R84 initialization changed before the first R99 update')
    params = copy.deepcopy(model.params)
    params.update(r99_fresh_trainable_sha256=full_hash,
        r99_common_parameter_sha256=common_hash, r99_new_parameter_sha256=new_hash)
    receipt = dict(seed=2, initialization=INITIALIZATION, historical_checkpoint_loaded=False,
        external_pretraining=False, r77_fresh_trainable_sha256=initial['trainable_sha256'],
        common_parameter_names=sorted(retained), common_parameter_sha256=common_hash,
        common_parameter_count=sum(value.numel() for value in retained.values()),
        new_parameter_names=sorted(added), new_parameter_sha256=new_hash,
        new_parameter_count=sum(value.numel() for value in added.values()),
        new_common_seed=PROJECTION_SEED, trainable_sha256=full_hash,
        trainable_parameter_count=sum(value.numel() for value in actual.values()),
        removed_parameter_names=[], frozen_parameter_count=0,
        geometry_fitted_on='frozen R84 new81000 TRAIN only')
    return model.to(device), params, receipt


def protocol(args, ready, manifest, geometry, baselines, params, order, caps):
    from .r99_model import SPEC
    from .r99_relations import SPEC as transform_spec
    if (args.large_micro_cap != 352 or args.eval_batch_size != 256
            or caps != micro_caps(args.gpu_type, 352)):
        raise ValueError('R99 retains TSP/CVRP352, others1024; eval256')
    config = engine.protocol(args, ready, manifest, geometry, baselines, params, order, caps)
    config['loss'].pop('focal_extra', None)
    config.update(experiment='R99', r99_schema_version=1, model_factory=MODEL_FACTORY,
        initialization=INITIALIZATION, main_protocol_lock=protocol_lock(args),
        objective=objective_spec(args.arm), model_spec=copy.deepcopy(SPEC),
        relation_transform=copy.deepcopy(transform_spec), relation_variant=VARIANTS[args.arm],
        relation_variants=VARIANTS.copy(), role_arm=MODEL_ROLE_ARM,
        projection_seed=PROJECTION_SEED, train_eval_every_updates=TRAIN_EVERY,
        relation_initialization_sha256=params['r99_fresh_trainable_sha256'],
        retained_parameter_sha256=params['r99_common_parameter_sha256'],
        new_parameter_sha256=params['r99_new_parameter_sha256'],
        feature_cache_root=str(feature_root(args).resolve()),
        feature_precision='FP32 input transform then FP16 per-node messages; identical online quantization',
        feature_cache_identity='original input cache identity + fixed transform + arm; no outcomes',
        dropout=.1, selection_head=SELECTION_HEAD, inference_rule=INFERENCE_RULE,
        input_roles=dict(arm=MODEL_ROLE_ARM, problem='TSP', node_column=ROLE_SLOT,
            assignment='physical input node0', label_dependent=False),
        historical_reference=dict(primary='R89B', secondary='R84A', retrained=False,
            matched_control='R99A row-softmax distance input messages'),
        external_pretraining=False, new_target=False, auxiliary_loss=False,
        evidence_boundary=('Input-only representational accessibility, not new information, an assignment '
            'solver, a VRP bound, feasibility, solver execution or B-semantics repair. Finite64 marginal '
            'residuals are not exact convergence. A/B matches everything except column competition. '
            'Development-exposed TEST; one minimum full18 VAL macro_vs_sbs_pct lock; no LIB/TEST tuning.'),
        source_sha256={name: file_hash(V4 / name) for name in
            ('r99_experiment.py', 'r99_model.py', 'r99_objective.py', 'r99_relations.py',
             'r99_features.py', 'r84_experiment.py', 'r84_prepare.py', 'r88_model.py',
             'V4Model.py', 'pairwise_selector.py', 'pairwise_objective.py', 'r76_experiment.py')})
    validate_checkpoint_config(config)
    return config


def make_objective(selector, arm):
    from .r99_objective import CompetitionObjective
    objective_spec(arm)
    return CompetitionObjective(selector)


def checkpoint_extras(objective):
    return {}


def restore_objective(objective, checkpoint):
    if 'objective_state' in checkpoint:
        raise ValueError('R99 has no mutable objective state')


def update_batch(objective, optimizer, scaler, batch, weight, runtime, cap, *, problem,
                 max_retries=20):
    if (problem not in PROBLEMS or len(batch['ind']) != BATCH_SIZE
            or cap != micro_caps('3090', 352)[problem]):
        raise ValueError('R99 logical batch or per-task microbatch changed')
    return engine.update_batch(objective, optimizer, scaler, batch, weight, runtime,
                               cap, max_retries=max_retries)


@lru_cache(maxsize=4)
def private_engine(arm, feature_directory):
    from .r99_features import CompetitionFeatureCache
    from .r99_model import inputs_with_relations
    if arm not in ARMS:
        raise ValueError('Unknown R99 arm')
    # Neither historical modules nor the other arm's closure are patched.
    name = 'code.V4._r99_engine_' + arm
    spec = importlib.util.spec_from_file_location(name, engine.__file__)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)

    def open_with_features(cache_root, split, problem, manifest, data_root, **kwargs):
        base = engine.open_cache(cache_root, split, problem, manifest, data_root, **kwargs)
        try:
            return CompetitionFeatureCache(base, Path(feature_directory) / split / problem,
                                           expected_variant=VARIANTS[arm])
        except BaseException:
            base.close()
            raise

    module.open_cache, module.inputs_only = open_with_features, inputs_with_relations
    return module


def prepare_input_features(args, device, *, split=None, test_lock=None):
    from .r99_features import prepare_features
    if split == 'test' and test_lock is None:
        raise PermissionError('R99 TEST input transform requires the unique completed VAL lock')
    records = {name: prepare_features(cache_root=args.cache_root, feature_root=feature_root(args),
        split=name, variant=VARIANTS[args.arm], device=device, batch_size=128,
        manifest=args.prepared_root / 'source_cache_manifest.json',
        data_root=args.data_root, test_lock=test_lock)
        for name in ((split,) if split is not None else ('train', 'val'))}
    for name, tasks in records.items():
        if set(tasks) != set(PROBLEMS):
            raise ValueError('R99 feature preparation must complete every task, including matrix ATSP')
        for problem, receipt in tasks.items():
            expected = dict(problem=problem, split=name, variant=VARIANTS[args.arm],
                            rows=SPLIT_ROWS[name], input_only=True, labels_opened=False)
            residual = receipt.get('max_marginal_residual')
            if (receipt.get('complete') is not True or receipt.get('rows_written') != SPLIT_ROWS[name]
                    or any(receipt.get('identity', {}).get(k) != v for k, v in expected.items())
                    or receipt.get('dtype') != 'float16' or receipt.get('width') != 32
                    or isinstance(residual, bool) or not isinstance(residual, (int, float))
                    or not math.isfinite(residual) or residual < 0):
                raise ValueError('Incomplete or incompatible R99 per-task feature receipt')
    return records


def evaluate(selector, args, manifest, baselines, split, tag, step, state_hash,
             checkpoint_sha256=None):
    if split == 'test':
        lock, _ = authorize_test(args)
        started = time.monotonic()
        receipts = prepare_input_features(args, next(selector.parameters()).device, split='test', test_lock=lock)
        path = args.root / args.arm / 'test_input_features_ready.json'
        if not path.exists():
            write_json(path, dict(experiment='R99', arm=args.arm, variant=VARIANTS[args.arm], complete=True,
                splits=['test'], completed_utc=utc(), checkpoint_sha256=lock['checkpoint_sha256'],
                state_hash=lock['state_hash'], seconds=time.monotonic() - started, feature_receipts=receipts))
    return private_engine(args.arm, str(feature_root(args).resolve())).evaluate(
        selector, args, manifest, baselines, split, tag, step, state_hash, checkpoint_sha256)


def validate_checkpoint_config(config):
    from .r99_model import SPEC
    from .r99_relations import SPEC as transform_spec
    arm = config.get('arm')
    expected = dict(experiment='R99', r99_schema_version=1, seed=2,
        initialization=INITIALIZATION, model_factory=MODEL_FACTORY, solver_names=list(GLOBAL_SOLVERS),
        rows_per_problem=SPLIT_ROWS, global_batch_size=BATCH_SIZE, logical_global_batch=BATCH_SIZE,
        max_successful_updates=MAX_UPDATES, warmup_updates=WARMUP_UPDATES,
        eval_every_updates=EVAL_EVERY, train_eval_every_updates=TRAIN_EVERY, eval_batch_size=256,
        optimizer='AdamW', weight_decay=1e-4, grad_clip=1., dropout=.1, rdrop_weight=.35,
        rdrop_ramp_updates=WARMUP_UPDATES, model_spec=SPEC, relation_transform=transform_spec,
        relation_variants=VARIANTS, projection_seed=PROJECTION_SEED, selection_head=SELECTION_HEAD,
        inference_rule=INFERENCE_RULE, winner_rule=WINNER_RULE, role_arm=MODEL_ROLE_ARM,
        external_pretraining=False, new_target=False, auxiliary_loss=False,
        microbatch_caps=micro_caps('3090', 352))
    if arm not in ARMS or any(config.get(key) != value for key, value in expected.items()):
        raise ValueError('Not an unchanged fresh full18 R99 checkpoint')
    params = config.get('model_params', {})
    if (config.get('relation_variant') != VARIANTS[arm]
            or config.get('objective') != objective_spec(arm)
            or config.get('loss') != engine.LOSS_RECIPE
            or params.get('r99_arm') != arm or params.get('r99_spec') != SPEC
            or config.get('lr') != dict(rule='successful-update warmup then cosine, independent of VAL',
                peak=PEAK_LR, end=END_LR, warmup_updates=WARMUP_UPDATES, max_updates=MAX_UPDATES)):
        raise ValueError('R99 input transform, original objective or LR changed')
    for config_key, param_key in (
            ('relation_initialization_sha256', 'r99_fresh_trainable_sha256'),
            ('retained_parameter_sha256', 'r99_common_parameter_sha256'),
            ('new_parameter_sha256', 'r99_new_parameter_sha256')):
        if not config.get(config_key) or config[config_key] != params.get(param_key):
            raise ValueError('R99 initialization hashes are missing or disagree')
    if config['retained_parameter_sha256'] != config.get('common_trainable_sha256'):
        raise ValueError('R99 retained parameters must equal the full fresh R84 initialization')


def load_checkpoint(path, device='cpu'):
    from .r99_model import make_model
    state = torch.load(path, map_location='cpu', weights_only=False)
    config = state['args']
    validate_checkpoint_config(config)
    if state.get('state_hash') != json_hash(config) or 'objective_state' in state:
        raise ValueError('R99 checkpoint protocol or stateless objective changed')
    model = make_model(config['model_params'], config['arm'], state_dict=state['model'])
    return model.to(device), state


load_model = from_checkpoint = load_checkpoint


def authorize_test(args):
    lock, config = engine.authorize_test(args)
    validate_checkpoint_config(config)
    if lock.get('inference_rule') != config['inference_rule']:
        raise PermissionError('R99 TEST inference differs from the unique VAL lock')
    if protocol_lock(args) != config.get('main_protocol_lock'):
        raise PermissionError('R99 main protocol lock changed after training')
    if feature_root(args).resolve() != Path(config['feature_cache_root']).resolve():
        raise PermissionError('R99 arm feature location changed after training')
    return lock, config


def prepare(args, device):
    lock = protocol_lock(args)
    prepared_config(args)
    started = time.monotonic()
    status_update(args, 'input_feature_preparation', 0, feature_root=str(feature_root(args)),
                  splits=['train', 'val'], labels_used=False)
    records = prepare_input_features(args, device)
    value = dict(experiment='R99', arm=args.arm, variant=VARIANTS[args.arm], complete=True,
        completed_utc=utc(), feature_root=str(feature_root(args).resolve()),
        cache_root=str(args.cache_root.resolve()), main_protocol_lock_sha256=lock['sha256'],
        splits=['train', 'val'], seconds=time.monotonic() - started,
        feature_receipts=records, labels_used=False, solver_executed=False, new_target=False)
    write_json(args.root / args.arm / 'input_features_ready.json', value)
    status_update(args, 'input_features_complete', 0, seconds=value['seconds'])
    return value


def train(args, device):
    lock = protocol_lock(args)
    prepared_config(args)
    ready = read_json(args.root / args.arm / 'input_features_ready.json')
    expected = dict(experiment='R99', arm=args.arm, variant=VARIANTS[args.arm], complete=True,
        feature_root=str(feature_root(args).resolve()), cache_root=str(args.cache_root.resolve()),
        splits=['train', 'val'], main_protocol_lock_sha256=lock['sha256'])
    if any(ready.get(key) != value for key, value in expected.items()):
        raise ValueError('Prepare matching arm-local R99 TRAIN/VAL input messages first')
    return private_engine(args.arm, str(feature_root(args).resolve())).train(
        args, device, integration=sys.modules[__name__])


def test(args, device):
    return engine.test(args, device, integration=sys.modules[__name__])


def raw_inputs(batch):
    """An allowlist as well as explicit pops prevents the historical timing bug."""
    from .r99_model import inputs_with_relations
    raw = dict(batch)
    raw.pop('relation_nodes', None)
    raw.pop('relation_node_mask', None)
    return inputs_with_relations(raw)


def _synchronize(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


@torch.no_grad()
def measure_inputs(selector, base_batch, cached_batch, device, repetitions=3):
    """CPU-testable timing spine; real CLI requires an allocated CUDA device."""
    from .r99_model import inputs_with_relations
    from .r99_relations import transport
    raw, cached = raw_inputs(base_batch), inputs_with_relations(cached_batch)
    if ('relation_nodes' not in cached or 'relation_node_mask' not in cached
            or not torch.equal(raw['node_mask'], cached['node_mask'])):
        raise ValueError('Timing needs aligned original inputs and arm-local cached messages')
    online = selector.prepare_inputs(raw)
    if (not torch.equal(online['relation_node_mask'], cached['relation_node_mask'])
            or not torch.equal(online['relation_nodes'], cached['relation_nodes'])):
        raise ValueError('Cached and online FP16 input messages differ on the fixed timing rows')
    variant = VARIANTS[selector.r99_arm]
    measurements = {}
    calls = dict(transform_only=lambda: transport(raw, variant),
        transform_plus_selector=lambda: selector(raw), cache_hit_selector=lambda: selector(cached))
    for scope, call in calls.items():
        call()
        _synchronize(device)
        samples = []
        for _ in range(repetitions):
            _synchronize(device)
            start = time.perf_counter()
            result = call()
            _synchronize(device)
            samples.append(time.perf_counter() - start)
            if scope != 'transform_only' and not torch.isfinite(result['logits'][result['solver_mask']]).all():
                raise ValueError('Nonfinite locked R99 timing prediction')
        measurements[scope] = dict(batch_seconds=samples,
            mean_seconds_per_instance=sum(samples) / (repetitions * len(raw['node_mask'])))
    measurements['max_marginal_residual'] = float(transport(raw, variant)[1])
    return measurements


@torch.no_grad()
def timing(args, device):
    lock, config = authorize_test(args)
    saved_test = read_json(args.root / args.arm / 'test_results.json')
    if (saved_test.get('complete') is not True
            or saved_test.get('state_hash') != lock['state_hash']
            or saved_test.get('checkpoint_sha256') != lock['checkpoint_sha256']):
        raise PermissionError('R99 timing follows complete locked TEST, never model selection')
    destination = args.root / args.arm / 'inference_timing.json'
    if destination.exists():
        from .r99_reporting import inference_timing
        inference_timing(destination.parent, lock)
        return read_json(destination)
    selector, state = load_checkpoint(args.root / args.arm / 'best.pt', device)
    del state
    selector.eval()
    args.cache_root = Path(config['cache_root']) if args.cache_root is None else args.cache_root
    manifest = read_json(args.prepared_root / 'source_cache_manifest.json')
    module = private_engine(args.arm, str(feature_root(args).resolve()))
    rows = []
    for problem in PROBLEMS:
        # Separate original cache: never pass the feature-wrapped batch as raw.
        base = engine.open_cache(args.cache_root, 'test', problem, manifest, args.data_root)
        cached = None
        try:
            cached = module.open_cache(args.cache_root, 'test', problem, manifest, args.data_root)
            if base.size < 128 or cached.size != base.size:
                raise ValueError('R99 timing requires first128 aligned TEST rows per task')
            indices = torch.arange(128)
            rows.append(dict(problem=problem, instances=128,
                **measure_inputs(selector, base.batch(indices, device=device),
                                 cached.batch(indices, device=device), device)))
        finally:
            base.close()
            if cached is not None:
                cached.close()
    value = dict(experiment='R99', arm=args.arm, variant=VARIANTS[args.arm], complete=True,
        completed_utc=utc(), checkpoint_sha256=lock['checkpoint_sha256'], state_hash=lock['state_hash'],
        inference_rule=config['inference_rule'], device=str(device), repetitions=3, warmup_batches=1,
        scope='same first128 TEST input rows/task, inputs already on GPU; synchronized wall time',
        raw_input_source='original_base_cache_without_relation_fields', online_transform_verified=True,
        online_fp16_matches_cache=True, includes_input_transform=True, includes_solver_decoder=False,
        full_system_measured=False, end_to_end_measured=False,
        excludes='dataset I/O, host transfer and one-time feature-cache preparation; cache-hit is not full inference',
        preprocessing=read_json(args.root / args.arm / 'input_features_ready.json'),
        test_preprocessing=read_json(args.root / args.arm / 'test_input_features_ready.json'),
        per_problem=rows, model_reselection=False)
    write_json(destination, value)
    return value


def describe(args):
    return dict(experiment='R99', arm=args.arm, variant=VARIANTS[args.arm], scope='full18',
        feature_root=str(feature_root(args)), required_protocol_lock=required_lock_fields(),
        required_lock_paths=['data_root', 'prepared_root'], objective=objective_spec(args.arm),
        inference_rule=INFERENCE_RULE, max_updates=MAX_UPDATES, seed=2,
        batch_size=BATCH_SIZE, new_solver_runs=False, new_performance_labels=False)


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('command', choices=('prepare', 'train', 'test', 'timing', 'describe', 'status'))
    result.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    result.add_argument('--arm', choices=ARMS, required=True)
    result.add_argument('--prepared-root', type=Path, default=DEFAULT_PREPARED_ROOT)
    result.add_argument('--data-root', type=Path, default=DEFAULT_DATA_ROOT)
    result.add_argument('--cache-root', type=Path)
    result.add_argument('--feature-root', type=Path)
    result.add_argument('--protocol-lock', type=Path)
    result.add_argument('--gpu-type', choices=('3090', '4090'), default='3090')
    result.add_argument('--large-micro-cap', type=int, default=352)
    result.add_argument('--eval-batch-size', type=int, default=256)
    result.add_argument('--log-every', type=int, default=18)
    result.add_argument('--device', default='cuda:0')
    result.add_argument('--wandb-mode', choices=('auto', 'online', 'offline'), default='offline')
    result.add_argument('--resume', action='store_true')
    result.add_argument('--state-hash', '--expected-state-hash')
    result.set_defaults(experiment='R99')
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    if args.command not in ('prepare', 'timing'):
        return engine.main(argv, integration=sys.modules[__name__])
    for name in ('root', 'prepared_root', 'data_root', 'cache_root', 'feature_root', 'protocol_lock'):
        value = getattr(args, name)
        if value is not None:
            setattr(args, name, value.expanduser().resolve())
    run = args.root / args.arm
    run.mkdir(parents=True, exist_ok=True)
    with (run / 'runner.lock').open('a') as claim:
        fcntl.flock(claim, fcntl.LOCK_EX | fcntl.LOCK_NB)
        device = torch.device(args.device)
        if device.type != 'cuda':
            raise ValueError('Formal R99 preparation/timing requires a launcher-authorized CUDA device')
        torch.set_num_threads(2)
        torch.set_num_interop_threads(2)
        return prepare(args, device) if args.command == 'prepare' else timing(args, device)


if __name__ == '__main__':
    main()
