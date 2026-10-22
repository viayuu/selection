"""R101: fixed TRAIN hint pretraining, then unchanged original full18 learning.

The main-owned pretraining endpoint is an initialization, not an online branch.
Only the instance encoder is transferred; final checkpoints are self-contained.
"""

import argparse
import copy
import fcntl
import hashlib
import importlib
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import torch

from code.unified_selector.registry import GLOBAL_SOLVERS, PROBLEMS
from . import r84_experiment as engine
from .r76_experiment import RDropObjective, file_hash, read_json, utc, write_json
from .r77_data import parameter_hash
from .r84_prepare import DEFAULT_DATA_ROOT, DEFAULT_PREPARED_ROOT, SPLIT_ROWS
from .r85_experiment import ensure_test_cache, selection_counts, zero_summary


V4 = Path(__file__).resolve().parent
DEFAULT_ROOT = V4 / 'runs/R101_hint_pretraining'
ARMS = NEURAL_ARMS = ('A', 'B')
VARIANTS = dict(A='random', B='pretrained')
MODEL_FACTORY = 'code.V4.r101_experiment.load_checkpoint'
MODEL_ROLE_ARM, SELECTION_HEAD = 'B_execution_anchor', 'original_score_mlp_maximin'
INITIALIZATION = 'fresh_R84_with_TRAIN_hint_pretrained_instance_encoder'
PARAMETER_COUNT, PRETRAIN_UPDATES, TRAIN_EVERY = 1842953, 12800, 7200
BATCH_SIZE, MAX_UPDATES = engine.BATCH_SIZE, engine.MAX_UPDATES
WARMUP_UPDATES, EVAL_EVERY = engine.WARMUP_UPDATES, engine.EVAL_EVERY
PEAK_LR, END_LR, WINNER_RULE = engine.PEAK_LR, engine.END_LR, engine.WINNER_RULE
BUDGETS = dict.fromkeys(ARMS, MAX_UPDATES)
prepared_config, micro_caps = engine.prepared_config, engine.micro_caps
task_order, learning_rate, json_hash = engine.task_order, engine.learning_rate, engine.json_hash
status_update = engine.status_update
INFERENCE_RULE = ('eval mode, original R88B physical TSP node0 slot3; original R77 '
    'hand34/methodID, joint2, score MLP/pair differences/maximin; stable ascending '
    'global-solver-ID ties; original inputs only; no teacher, hint, adapter or calibration')
TEACHER_CHECKPOINT_SHA256 = '2ce5fb49950c4c9f51d2015a125744be9960aa0c60c609cfe235e74d44ad4be3'


def pretrain_spec():
    from .r101_pretrain import SPEC
    return copy.deepcopy(SPEC)


def model_spec():
    """The model/schema owner supplies this pure metadata dictionary."""
    return copy.deepcopy(importlib.import_module('code.V4.r101_model').SPEC)


def required_lock_fields():
    """Main can import this before freezing the shared A/B lock."""
    return dict(experiment='R101', locked=True, arms=list(ARMS), seed=2,
        task_scope='full18', rows_per_problem=dict(SPLIT_ROWS), batch_size=BATCH_SIZE,
        logical_global_batch=BATCH_SIZE, max_updates=MAX_UPDATES,
        warmup_updates=WARMUP_UPDATES, eval_every=EVAL_EVERY,
        train_eval_every=TRAIN_EVERY, large_micro_cap=352, eval_batch_size=256,
        teacher_variants=VARIANTS.copy(), pretrain_updates=PRETRAIN_UPDATES,
        pretrain_warmup=960, hint_pretraining=pretrain_spec())


def protocol_lock(args, *, require_teacher=False):
    path = Path(getattr(args, 'protocol_lock', None) or args.root / 'protocol_lock.json').expanduser().resolve()
    record = read_json(path)
    if any(record.get(key) != value for key, value in required_lock_fields().items()):
        raise ValueError('R101 requires the final matched two-stage full18 protocol lock')
    for name in ('data_root', 'prepared_root'):
        if not record.get(name) or Path(record[name]).resolve() != Path(getattr(args, name)).resolve():
            raise ValueError('R101 lock names a different ' + name)
    if require_teacher:
        root = getattr(args, 'teacher_root', None)
        if root is None:
            raise ValueError('R101 pretraining/transfer requires --teacher-root provenance')
        roots = record.get('teacher_roots')
        if roots is not None and (args.arm not in roots or Path(roots[args.arm]).resolve() != Path(root).resolve()):
            raise ValueError('R101 arm teacher root differs from the main lock')
    return dict(path=str(path), sha256=file_hash(path), record=record)


def objective_spec(arm):
    if arm not in ARMS:
        raise ValueError('Unknown R101 arm')
    return dict(classification='original R77 native CE',
        coefficients=dict(ce=.35, comparison=.35, risk=.02),
        consistency='original symmetric R-Drop ramp to .35 over4320 updates',
        forward_passes=2, stageII_hint_loss=False, extra_loss=False,
        mutable_objective_state=False)


def validate_teacher_identity(identity, arm):
    covered = {p for p in PROBLEMS if p not in ('TSP', 'ATSP')}
    if (not isinstance(identity, dict) or identity.get('variant') != VARIANTS[arm]
            or identity.get('split') != 'train' or identity.get('rows_per_task') != 81000
            or identity.get('no_teacher_in_final_model') is not True
            or set(identity.get('metadata_sha256', {})) != covered
            or set(identity.get('donor', {})) != covered):
        raise ValueError('R101 lineage requires aligned full TRAIN-only 16-task teacher provenance')
    for problem in covered:
        donor = identity['donor'][problem]
        if (donor.get('variant') != VARIANTS[arm] or donor.get('output_dim') != 128
                or not _digest(identity['metadata_sha256'][problem])):
            raise ValueError('R101 teacher width/variant/cache identity changed')
        if arm == 'A':
            if donor.get('checkpoint_sha256') is not None or donor.get('random_seed') != 940002:
                raise ValueError('R101A requires the original random R94A teacher')
        elif donor.get('checkpoint_sha256') != TEACHER_CHECKPOINT_SHA256:
            raise ValueError('R101B requires the pinned pretrained R94B teacher')


def _digest(value):
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def validate_lineage(lineage, arm):
    expected = dict(experiment='R101', arm=arm, complete=True,
        successful_updates=PRETRAIN_UPDATES, pretrain_spec=pretrain_spec())
    if not isinstance(lineage, dict) or any(lineage.get(k) != v for k, v in expected.items()):
        raise ValueError('R101 needs a newly completed fixed12800 TRAIN hint endpoint')
    for key in ('checkpoint_sha256', 'encoder_sha256', 'initial_trainable_sha256',
                'unchanged_non_encoder_sha256', 'protocol_sha256'):
        if not _digest(lineage.get(key)):
            raise ValueError('R101 pretraining lineage is missing ' + key)
    validate_teacher_identity(lineage.get('teacher_identity'), arm)


def read_pretraining(args):
    """Called only at Stage II initialization, never by a deployed factory."""
    lock = protocol_lock(args)
    run = args.root / args.arm
    path = run / 'pretrain_end.pt'
    receipt = read_json(run / 'pretrain_complete.json')
    state = torch.load(path, map_location='cpu', weights_only=False)
    keys = ('experiment', 'arm', 'complete', 'successful_updates', 'encoder_sha256',
            'initial_trainable_sha256', 'unchanged_non_encoder_sha256',
            'teacher_identity', 'pretrain_spec', 'protocol_sha256')
    lineage = {key: copy.deepcopy(state.get(key)) for key in keys}
    lineage['checkpoint_sha256'] = file_hash(path)
    validate_lineage(lineage, args.arm)
    if (lineage['protocol_sha256'] != lock['sha256']
            or receipt.get('complete') is not True
            or receipt.get('successful_updates') != PRETRAIN_UPDATES
            or receipt.get('checkpoint_sha256') != lineage['checkpoint_sha256']
            or receipt.get('protocol_sha256') != lock['sha256']):
        raise ValueError('R101 pretraining endpoint/completion/main lock disagree')
    for key in keys:
        if key in receipt and receipt[key] != lineage[key]:
            raise ValueError('R101 pretraining receipt differs: ' + key)
    encoder = state.get('instance_encoder')
    if (not isinstance(encoder, dict) or not encoder
            or any(not torch.is_tensor(t) or t.device.type != 'cpu' for t in encoder.values())
            or parameter_hash(encoder) != lineage['encoder_sha256']):
        raise ValueError('R101 pretrained encoder state/hash differs')
    return state, lineage


def make_model(params, arm, state_dict=None):
    from .r88_model import make_model as original
    if arm not in ARMS or params.get('r101_arm', arm) != arm:
        raise ValueError('Unknown or mismatched R101 arm')
    if params.get('r101_spec', model_spec()) != model_spec():
        raise ValueError('R101 model metadata changed')
    values = copy.deepcopy(params)
    values.update(r101_arm=arm, r101_spec=model_spec())
    model = original(values, MODEL_ROLE_ARM, state_dict=state_dict)
    if (sum(t.numel() for t in model.parameters()) != PARAMETER_COUNT
            or any(not p.requires_grad for p in model.parameters())):
        raise ValueError('R101 retains exactly the original1842953 trainable student parameters')
    return model


def initialize_shared(initial, geometry, arm, device='cpu', *, pretrain_state):
    from .r101_pretrain import initialize_fresh, non_encoder_hash
    if arm not in ARMS:
        raise ValueError('Unknown R101 arm')
    model = initialize_fresh(initial, geometry, 'cpu')
    other_before = non_encoder_hash(model)
    if (pretrain_state.get('arm') != arm or pretrain_state.get('experiment') != 'R101'
            or pretrain_state.get('complete') is not True
            or pretrain_state.get('successful_updates') != PRETRAIN_UPDATES
            or pretrain_state.get('initial_trainable_sha256') != initial['trainable_sha256']
            or pretrain_state.get('unchanged_non_encoder_sha256') != other_before
            or pretrain_state.get('pretrain_spec') != pretrain_spec()
            or parameter_hash(pretrain_state['instance_encoder']) != pretrain_state.get('encoder_sha256')):
        raise ValueError('R101 endpoint must descend from this exact fresh R84 initialization')
    validate_teacher_identity(pretrain_state.get('teacher_identity'), arm)
    fresh_buffers = {name: value.clone() for name, value in model.instance_encoder.named_buffers()}
    model.instance_encoder.load_state_dict(pretrain_state['instance_encoder'], strict=True)
    for name, tensor in model.instance_encoder.named_buffers():
        if not torch.equal(tensor, fresh_buffers[name]):
            raise ValueError('R101 pretraining must retain the frozen R84 geometry/buffers')
    actual = dict(model.named_parameters())
    if (set(actual) != set(initial['trainable']) or non_encoder_hash(model) != other_before
            or sum(t.numel() for t in actual.values()) != PARAMETER_COUNT
            or any(not p.requires_grad for p in actual.values())):
        raise ValueError('R101 transfer changed original non-encoder parameters or student capacity')
    model.params.update(r101_arm=arm, r101_spec=model_spec())
    receipt = dict(seed=2, initialization=INITIALIZATION,
        r77_fresh_trainable_sha256=initial['trainable_sha256'],
        initial_trainable_sha256=initial['trainable_sha256'],
        unchanged_non_encoder_sha256=other_before,
        pretrained_encoder_sha256=parameter_hash(model.instance_encoder.state_dict()),
        trainable_sha256=parameter_hash(actual), trainable_parameter_count=PARAMETER_COUNT,
        transferred_parameter_prefix='instance_encoder.',
        historical_selector_checkpoint_loaded=False, optimizer_reset=True, scaler_reset=True,
        rng_reset=True, task_sample_streams_reset=True, geometry_fitted_on='frozen R84 TRAIN81000 only',
        role_arm=MODEL_ROLE_ARM)
    return model.to(device), copy.deepcopy(model.params), receipt


def protocol(args, ready, manifest, geometry, baselines, params, order, caps, *, lineage):
    if args.large_micro_cap != 352 or args.eval_batch_size != 256 or caps != micro_caps('4090', 352):
        raise ValueError('R101 retains TSP/CVRP352, others1024; eval256')
    validate_lineage(lineage, args.arm)
    config = engine.protocol(args, ready, manifest, geometry, baselines, params, order, caps)
    config['loss'].pop('focal_extra', None)
    config.update(experiment='R101', r101_schema_version=1, model_factory=MODEL_FACTORY,
        initialization=INITIALIZATION, main_protocol_lock=protocol_lock(args),
        hint_pretraining=pretrain_spec(), pretraining_lineage=copy.deepcopy(lineage),
        model_spec=model_spec(), objective=objective_spec(args.arm),
        train_eval_every_updates=TRAIN_EVERY, role_arm=MODEL_ROLE_ARM,
        input_roles=dict(arm=MODEL_ROLE_ARM, problem='TSP', node_column=3,
            assignment='physical input node0', label_dependent=False),
        unchanged_non_encoder_sha256=lineage['unchanged_non_encoder_sha256'],
        pretrained_encoder_sha256=lineage['encoder_sha256'],
        trainable_parameter_count=PARAMETER_COUNT, new_trainable_parameters=0,
        selection_head=SELECTION_HEAD, inference_rule=INFERENCE_RULE,
        downstream_teacher_inputs=False, downstream_hint_loss=False,
        external_pretraining=args.arm == 'B', deployment_teacher=False,
        stageII_reset=dict(optimizer=True, scaler=True, rng_seed=2, task_order=True, sample_order=True),
        historical_reference=dict(primary='R89B', secondary='R84A', budget_matched=False,
            matched_control='R101A random teacher, identical12800+57600 updates',
            pretrained_teacher_history='R94B has additional external solver pretraining history'),
        evidence_boundary=('Paper memo is conditional motivation, not measured support. A/B matches '
            '12800 TRAIN hint +57600 full18 updates; teacher knowledge differs. Historical R89B/R84A '
            'are not total-budget-matched. R94B donor has extra external history. No teacher at forward '
            'or deployment. TEST is development-exposed; unique full18 VAL actual-cost lock; no TEST/LIB selection.'),
        source_sha256={name: file_hash(V4 / name) for name in
            ('r101_experiment.py', 'r101_model.py', 'r101_pretrain.py', 'r101_features.py',
             'r84_experiment.py', 'r84_prepare.py', 'r88_model.py', 'pairwise_selector.py',
             'pairwise_objective.py', 'r76_experiment.py')})
    validate_checkpoint_config(config)
    return config


def validate_checkpoint_config(config):
    from .r88_model import ROLE_SPEC
    arm = config.get('arm')
    expected = dict(experiment='R101', r101_schema_version=1, seed=2, model_factory=MODEL_FACTORY,
        initialization=INITIALIZATION, solver_names=list(GLOBAL_SOLVERS), rows_per_problem=SPLIT_ROWS,
        global_batch_size=BATCH_SIZE, logical_global_batch=BATCH_SIZE,
        max_successful_updates=MAX_UPDATES, warmup_updates=WARMUP_UPDATES,
        eval_every_updates=EVAL_EVERY, train_eval_every_updates=TRAIN_EVERY, eval_batch_size=256,
        optimizer='AdamW', weight_decay=1e-4, grad_clip=1., dropout=.1, rdrop_weight=.35,
        rdrop_ramp_updates=WARMUP_UPDATES, selection_head=SELECTION_HEAD, inference_rule=INFERENCE_RULE,
        role_arm=MODEL_ROLE_ARM, model_spec=model_spec(), hint_pretraining=pretrain_spec(),
        trainable_parameter_count=PARAMETER_COUNT, new_trainable_parameters=0,
        downstream_teacher_inputs=False, downstream_hint_loss=False, deployment_teacher=False,
        microbatch_caps=micro_caps('4090', 352), loss=engine.LOSS_RECIPE,
        stageII_reset=dict(optimizer=True, scaler=True, rng_seed=2, task_order=True, sample_order=True))
    if arm not in ARMS or any(config.get(k) != v for k, v in expected.items()):
        raise ValueError('R101 original student or fixed Stage II controls changed')
    lineage = config.get('pretraining_lineage')
    validate_lineage(lineage, arm)
    params = config.get('model_params', {})
    lock = config.get('main_protocol_lock', {})
    fixed_params = dict(dropout=.1, solver_representation='handcrafted',
        pair_mode='score_difference', selection_head=SELECTION_HEAD,
        role_arm=MODEL_ROLE_ARM, role_spec=ROLE_SPEC)
    if (config.get('objective') != objective_spec(arm)
            or params.get('r101_arm') != arm or params.get('r101_spec') != model_spec()
            or any(params.get(k) != v for k, v in fixed_params.items())
            or config.get('external_pretraining') != (arm == 'B')
            or config.get('common_trainable_sha256') != lineage['initial_trainable_sha256']
            or config.get('pretrained_encoder_sha256') != lineage['encoder_sha256']
            or config.get('unchanged_non_encoder_sha256') != lineage['unchanged_non_encoder_sha256']
            or lock.get('sha256') != lineage['protocol_sha256']
            or config.get('lr') != dict(rule='successful-update warmup then cosine, independent of VAL',
                peak=PEAK_LR, end=END_LR, warmup_updates=WARMUP_UPDATES, max_updates=MAX_UPDATES)
            or config.get('task_order_sha256') != hashlib.sha256(
                '\n'.join(task_order(MAX_UPDATES)).encode('ascii')).hexdigest()
            or config.get('checkpoint_rule') != 'strict minimum full VAL macro_vs_sbs_pct with common new81000 TRAIN-fitted SBS; first wins ties'
            or config.get('early_stop') is not False or config.get('fixed_budget') is not True):
        raise ValueError('R101 encoder lineage, fresh remainder or original VAL/LR protocol changed')
    if any(lock.get('record', {}).get(k) != v for k, v in required_lock_fields().items()):
        raise ValueError('R101 embedded main lock changed')


def load_checkpoint(path, device='cpu'):
    """No teacher/cache/pretrain file access, even when lineage paths are gone."""
    state = torch.load(path, map_location='cpu', weights_only=False)
    config = state['args']
    validate_checkpoint_config(config)
    if state.get('state_hash') != json_hash(config) or 'objective_state' in state:
        raise ValueError('R101 checkpoint protocol/stateless objective changed')
    return make_model(config['model_params'], config['arm'], state['model']).to(device), state


load_model = from_checkpoint = load_checkpoint


def make_objective(selector, arm):
    objective_spec(arm)
    return RDropObjective(selector)


def checkpoint_extras(objective):
    return {}


def restore_objective(objective, checkpoint):
    if 'objective_state' in checkpoint:
        raise ValueError('R101 Stage II has no mutable hint/objective state')


def update_batch(objective, optimizer, scaler, batch, weight, runtime, cap, *, problem, max_retries=20):
    if (problem not in PROBLEMS or len(batch['ind']) != BATCH_SIZE
            or cap != micro_caps('4090', 352)[problem]
            or any(key.startswith(('teacher_', 'donor_', 'nco_')) for key in batch)):
        raise ValueError('R101 Stage II requires original full1024 batches without teacher inputs')
    parts, retries, _ = engine.update_batch(objective, optimizer, scaler, batch, weight,
        runtime, cap, max_retries=max_retries)
    return parts, retries, {}


def evaluate(*args, **kwargs):
    return engine.evaluate(*args, **kwargs)


def authorize_test(args):
    lock, config = engine.authorize_test(args)
    validate_checkpoint_config(config)
    if (lock.get('inference_rule') != config['inference_rule']
            or protocol_lock(args) != config['main_protocol_lock']):
        raise PermissionError('R101 TEST differs from its unique full18 VAL/protocol lock')
    if args.cache_root is not None and Path(args.cache_root).resolve() != Path(config['cache_root']).resolve():
        raise PermissionError('R101 TEST must use the locked original input cache')
    return lock, config


def pretrain(args, device):
    protocol_lock(args, require_teacher=True)
    prepared_config(args)
    from .r101_pretrain import pretrain as implementation
    return implementation(args, device)


def train(args, device):
    # Completion/reports/deployment never reopen a teacher or pretraining file.
    if (args.root / args.arm / 'training_complete.json').exists():
        return test(args, device)
    state, lineage = read_pretraining(args)
    hooks = SimpleNamespace(**vars(sys.modules[__name__]))
    hooks.initialize_shared = lambda initial, geometry, arm, device='cpu': initialize_shared(
        initial, geometry, arm, device, pretrain_state=state)
    hooks.protocol = lambda *values: protocol(*values, lineage=lineage)
    return engine.train(args, device, integration=hooks)


def test(args, device):
    return engine.test(args, device, integration=sys.modules[__name__])


def raw_inputs(batch):
    names = ('node', 'matrix', 'node_mask', 'n', 'node_geom', 'kind', 'problem_id', 'coord_dist',
             'cbits', 'problem_desc', 'd_in', 'pool_ids', 'solver_mask')
    return {key: batch[key] for key in names if key in batch}


@torch.no_grad()
def measure_inputs(selector, batch, device, repetitions=3):
    raw = raw_inputs(batch)
    def synchronize():
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
    selector(raw)
    synchronize()
    samples = []
    for _ in range(repetitions):
        synchronize()
        started = time.perf_counter()
        result = selector(raw)
        synchronize()
        samples.append(time.perf_counter() - started)
        if not torch.isfinite(result['logits'][result['solver_mask']]).all():
            raise ValueError('Nonfinite R101 timing prediction')
    return dict(batch_seconds=samples,
        mean_seconds_per_instance=sum(samples) / (repetitions * len(raw['node_mask'])))


@torch.no_grad()
def timing(args, device):
    lock, config = authorize_test(args)
    run = args.root / args.arm
    saved = read_json(run / 'test_results.json')
    if (saved.get('complete') is not True or saved.get('checkpoint_sha256') != lock['checkpoint_sha256']
            or saved.get('state_hash') != lock['state_hash']):
        raise PermissionError('R101 timing follows complete locked TEST')
    destination = run / 'inference_timing.json'
    if destination.exists():
        from .r101_reporting import inference_timing
        inference_timing(run, lock)
        return read_json(destination)
    selector, state = load_checkpoint(run / 'best.pt', device)
    del state
    selector.eval()
    manifest = read_json(Path(config['prepared_root']) / 'source_cache_manifest.json')
    rows = []
    for problem in PROBLEMS:
        base = engine.open_cache(Path(config['cache_root']), 'test', problem, manifest, Path(config['data_root']))
        try:
            if base.size < 128:
                raise ValueError('R101 timing uses the first128 original TEST inputs per task')
            rows.append(dict(problem=problem, instances=128,
                student=measure_inputs(selector, base.batch(torch.arange(128), device=device), device)))
        finally:
            base.close()
    value = dict(experiment='R101', arm=args.arm, complete=True, completed_utc=utc(),
        checkpoint_sha256=lock['checkpoint_sha256'], state_hash=lock['state_hash'],
        inference_rule=config['inference_rule'], device=str(device), repetitions=3, warmup_batches=1,
        scope='first128 locked TEST input rows/task; GPU-resident original inputs; synchronized wall time',
        raw_input_source='original_base_cache', student_only=True, teacher_arrays_opened=False,
        online_teacher=False, full_system_measured=False, end_to_end_measured=False,
        excludes='dataset I/O, host transfer, TRAIN hint pretraining and solver execution',
        r94_cached_latency_used=False, per_problem=rows, model_reselection=False)
    write_json(destination, value)
    return value


def describe(args):
    return dict(experiment='R101', arm=args.arm, teacher_variant=VARIANTS[args.arm], scope='full18',
        required_protocol_lock=required_lock_fields(), required_lock_paths=['data_root', 'prepared_root'],
        optional_lock_paths='teacher_roots: arm->teacher-root', objective=objective_spec(args.arm),
        inference_rule=INFERENCE_RULE, pretrain_api='code.V4.r101_pretrain.pretrain(args, device)',
        pretraining_updates=PRETRAIN_UPDATES, max_updates=MAX_UPDATES,
        batch_size=BATCH_SIZE, seed=2, trainable_parameters=PARAMETER_COUNT, deployment_teacher=False)


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('command', choices=('pretrain', 'train', 'test', 'timing', 'describe', 'status'))
    result.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    result.add_argument('--arm', choices=ARMS, required=True)
    result.add_argument('--prepared-root', type=Path, default=DEFAULT_PREPARED_ROOT)
    result.add_argument('--data-root', type=Path, default=DEFAULT_DATA_ROOT)
    result.add_argument('--cache-root', type=Path)
    result.add_argument('--teacher-root', type=Path)
    result.add_argument('--protocol-lock', type=Path)
    result.add_argument('--gpu-type', choices=('4090', '3090'), default='4090')
    result.add_argument('--large-micro-cap', type=int, default=352)
    result.add_argument('--eval-batch-size', type=int, default=256)
    result.add_argument('--log-every', type=int, default=18)
    result.add_argument('--device', default='cuda:0')
    result.add_argument('--wandb-mode', choices=('auto', 'online', 'offline'), default='offline')
    result.add_argument('--resume', action='store_true')
    result.add_argument('--state-hash', '--expected-state-hash')
    result.set_defaults(experiment='R101')
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    for name in ('root', 'prepared_root', 'data_root', 'cache_root', 'teacher_root', 'protocol_lock'):
        value = getattr(args, name)
        if value is not None:
            setattr(args, name, value.expanduser().resolve())
    if args.command == 'describe':
        value = describe(args)
        print(json.dumps(value, indent=2))
        return value
    if args.command == 'status':
        run = args.root / args.arm
        value = {stage: read_json(run / name) if (run / name).exists() else dict(phase='not_started')
            for stage, name in (('pretraining', 'pretrain_status.json'), ('downstream', 'status.json'))}
        print(json.dumps(value, indent=2))
        return value
    if args.command in ('train', 'test'):
        return engine.main(argv, integration=sys.modules[__name__])
    run = args.root / args.arm
    run.mkdir(parents=True, exist_ok=True)
    with (run / 'runner.lock').open('a') as claim:
        fcntl.flock(claim, fcntl.LOCK_EX | fcntl.LOCK_NB)
        device = torch.device(args.device)
        if device.type != 'cuda':
            raise ValueError('Formal R101 pretraining/timing requires an allocated CUDA device')
        torch.set_num_threads(2)
        torch.set_num_interop_threads(2)
        engine.configure_torch()
        return pretrain(args, device) if args.command == 'pretrain' else timing(args, device)


if __name__ == '__main__':
    main()
