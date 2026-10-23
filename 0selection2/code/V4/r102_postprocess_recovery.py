"""One-shot R102 delivery continuation, never training or resource dispatch.

Only the main-owned cross-node array proof authorizes header compatibility.
All writes are private; the original failed experiment remains failed forever.
"""

import argparse
import copy
from contextlib import contextmanager, ExitStack
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import traceback
from types import SimpleNamespace
from unittest.mock import patch
from uuid import UUID

from . import r102_transfer as transfer


V4 = Path(__file__).resolve().parent
DEFAULT_ROOT = V4 / 'runs/R102_full_type_pretrained_fusion'
RELATIVE = Path('preparation/postprocess_recovery')
SCHEMA = 'r102_postprocess_recovery_v1'
PROOF_SCHEMA = 'r102_cross_node_test_array_identity_v1'
BUILD_FIELDS = frozenset(('created_utc', 'completed_utc', 'seconds', 'test_lock_root'))
PROBLEMS = transfer.PROBLEMS
LOCKED = dict(
    A=(33120, '10694b4d0fed80f1baec0d5e7de15cceec210eca5dfa846623783c4b9d6e7a2c'),
    B=(20160, '7df434c762a0e0b7726ab67211f63f51b55d1f86d108ec307a0aede0060f921f'))
POST_TEST_KIND = 'post_test_original_timing_observation'
KNOWN_TIMING_FAILURES = dict(
    A=('2503d4ff835e10dcf3809918e1ab9893824cc31d61bb81943b2b953389af4d62',
       '903bb3f17a6547f99e89197d9aaef3b59ddb0403a1e6d202645f54b4751030ca',
       'bd3bbb8469411a7d7939215035013ee0a75c56ff49e8624f3c90d9173556fb91',
       'c9d899d459006f37ca7e7149e2ba2b3d4a85a5eda94e55de029d26334b2b94b3'),
    B=('6f3fea8148f550d018281ecdceafc49277aa36ca6dd4e46bff0c385c2b364017',
       '76fdbbe4970c73ef87792fbd2212670c76fe6df238888a7b0cdbdd3dab5f922a',
       '52e18da11d0e958e0345b8996d1fbd0d3b1e10a8e2541bd4122b6f2d77d92a80',
       'eff9aef46cfef415e38e83eac1a2e927c94003b17ecf5e95dfb1b2da463612ac'))
NUMERICAL_BOUNDARY = (
    'Full TEST metrics use the preserved cached-feature first TEST. They are NOT '
    'certified as bitwise-equivalent online-donor efficacy. Original all18 timing '
    'measures online donor+selector and cached selector separately. Differences '
    'and changed picks concern only first128 rows/task, not whole-TEST agreement. '
    'No inference, grouping, TF32 policy, weights or selection was changed.')
FROZEN_VALIDATE = transfer.validate_frozen_header
FROZEN_R94_VALIDATE = transfer._validate_r94
read_json, file_hash, digest = transfer.read_json, transfer.file_hash, transfer.digest


def utc():
    return datetime.now(timezone.utc).isoformat()


def base(root):
    return Path(root).resolve() / RELATIVE


def view(root):
    return base(root) / 'report_view'


def private(path, root):
    path = Path(path).absolute()
    if not path.resolve().is_relative_to(base(root)) or path.is_symlink():
        raise ValueError('Writes must remain in the real private recovery tree')
    return path


def save(path, record, root, *, exclusive=False):
    path = private(path, root)
    path.parent.mkdir(parents=True, exist_ok=True)
    if exclusive:
        transfer.write_new_json(path, record)
    else:
        temporary = private(path.with_name('.' + path.name + '.' + str(os.getpid())), root)
        transfer.write_new_json(temporary, record)
        temporary.replace(path)


def artifact(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=file_hash(path))


def checked_artifact(record):
    path = Path(record['path'])
    if file_hash(path) != record['sha256']:
        raise ValueError('Evidence changed: ' + str(path))
    return path


def _without_runtime(record):
    return {k: v for k, v in record.items() if k not in transfer.RUNTIME_FIELDS}


def validate_proof(path, manifest_path):
    """Validate the existing concrete proof; do not hash/open array payloads."""
    proof = read_json(path)
    manifest = read_json(manifest_path)
    headers = manifest['splits']['test']
    if (proof.get('schema') != PROOF_SCHEMA or proof.get('complete') is not True
            or proof.get('all_arrays_equal') is not True or proof.get('problem_count') != 18
            or proof.get('gpu03_header_matches_frozen_reference') is not True
            or proof.get('other_gpu05_header_fields_equal') is not True
            or set(proof.get('allowed_build_receipt_fields', [])) != BUILD_FIELDS
            or set(proof.get('nodes', {})) != {'gpu03', 'gpu05'}
            or set(proof.get('per_problem', {})) != set(PROBLEMS)
            or set(headers) != set(PROBLEMS)):
        raise ValueError('Require the complete main-owned all18 TEST byte-equality proof')
    nodes = {}
    for node, receipt in proof['nodes'].items():
        raw = read_json(checked_artifact(receipt))
        if (raw.get('node') != node or raw.get('mode') != 'hashes'
                or raw.get('arrays_hashed') is not True or raw.get('CPU_only') is not True
                or raw.get('no_GPU_context') is not True or raw.get('cache_modified') is not False
                or raw.get('raw_values_interpreted') is not False
                or set(raw.get('problems', {})) != set(PROBLEMS)):
            raise ValueError('Invalid original cross-node hash manifest: ' + node)
        nodes[node] = raw
    pairs, total_count, total_bytes = {}, 0, 0
    for problem in PROBLEMS:
        reference = copy.deepcopy(headers[problem])
        FROZEN_VALIDATE(reference)
        if reference['split'] != 'test' or reference['rows'] != 4500:
            raise ValueError('Proof must cover the frozen new4500 TEST, not another split')
        left, right = (nodes[n]['problems'][problem] for n in ('gpu03', 'gpu05'))
        differences = right.get('header_differences', {})
        if left.get('header_differences') != {} or set(differences) != BUILD_FIELDS:
            raise ValueError('Only the exact four observed build-receipt differences are authorized')
        actual = copy.deepcopy(reference)
        for key, values in differences.items():
            if (set(values) != {'actual', 'reference'} or values['reference'] != reference[key]
                    or values['actual'] == values['reference']):
                raise ValueError('Header difference does not match the frozen reference')
            actual[key] = values['actual']
        FROZEN_VALIDATE(actual)
        arrays = left.get('arrays', {})
        if set(arrays) != set(reference['arrays']) or arrays != right.get('arrays'):
            raise ValueError('Every TEST array, including FP64 costs/native winners, must match')
        size = 0
        for key, entry in arrays.items():
            sha = entry.get('sha256', '')
            if (entry.get('file') != reference['arrays'][key]['file']
                    or type(entry.get('bytes')) is not int or entry['bytes'] <= 0
                    or len(sha) != 64 or any(c not in '0123456789abcdef' for c in sha)):
                raise ValueError('Malformed concrete array byte/hash record')
            size += entry['bytes']
        summary = proof['per_problem'][problem]
        if (summary.get('arrays_equal') is not True or summary.get('array_count') != len(arrays)
                or summary.get('bytes') != size
                or set(summary.get('gpu05_different_header_fields', [])) != BUILD_FIELDS):
            raise ValueError('Proof summary disagrees with its raw hash manifests')
        total_count += len(arrays)
        total_bytes += size
        pairs[problem] = dict(reference=reference, actual=actual, arrays=arrays)
    if proof.get('array_count') != total_count or proof.get('total_bytes_per_node') != total_bytes:
        raise ValueError('Cross-node TEST proof totals disagree')
    return dict(proof=artifact(path), manifest=artifact(manifest_path), nodes=proof['nodes'],
                array_count=total_count, total_bytes_per_node=total_bytes, pairs=pairs)


class HeaderCompatibility:
    """Fail-closed, process-local adapters for precisely two proven TEST headers."""

    def __init__(self, evidence):
        self.pairs = copy.deepcopy(evidence['pairs'])
        self.uses = []

    def validate(self, record, reference=None):
        try:
            return FROZEN_VALIDATE(record, reference)
        except ValueError:
            if reference is None:
                raise
            FROZEN_VALIDATE(record)
            FROZEN_VALIDATE(reference)
            pair = self.pairs.get(record['problem'])
            if pair is None or record['split'] != 'test' or reference['split'] != 'test':
                raise ValueError('Compatibility is restricted to the proven TEST header pair')
            known = [_without_runtime(pair[k]) for k in ('reference', 'actual')]
            if _without_runtime(record) not in known or _without_runtime(reference) not in known:
                raise ValueError('Unproven header change; preserve first failure')
            differing = {k for k in record.keys() | reference.keys()
                         if k not in transfer.RUNTIME_FIELDS and record.get(k) != reference.get(k)}
            if differing != BUILD_FIELDS:
                raise ValueError('Not the exact four proven build-receipt differences')
            self.uses.append(dict(problem=record['problem'], fields=sorted(differing)))
            return record

    def validate_r94(self, metadata, original, donor):
        if original.get('split') != 'test':
            return FROZEN_R94_VALIDATE(metadata, original, donor)
        pair = self.pairs.get(original.get('problem'))
        if pair is None:
            raise ValueError('R94 routing header is not covered by the all18 proof')
        self.validate(original, pair['reference'])
        bound = metadata.get('source_cache', {}).get('metadata_sha256')
        for key in ('reference', 'actual'):
            candidate = pair[key]
            if digest(candidate) == bound:
                # Keep the original R94 digest, donor, lock and direct input-path checks.
                return FROZEN_R94_VALIDATE(metadata, candidate, donor)
        raise ValueError('R94 routing digest binds neither proven original TEST header')


def validate_local_cache(cache_root, evidence):
    """Headers/NPY headers/stat only; byte payload equality was proved by main."""
    signatures = {}
    for problem, pair in evidence['pairs'].items():
        directory = Path(cache_root) / 'test' / problem
        metadata_path = directory / 'metadata.json'
        record = read_json(metadata_path)
        FROZEN_VALIDATE(record)
        if record != pair['actual']:
            raise ValueError('Live gpu05 header differs from the concrete proved header: ' + problem)
        for key, entry in pair['arrays'].items():
            path = directory / entry['file']
            spec = record['arrays'][key]
            header = transfer.npy_header(path)
            dtype = {'float32': '<f4', 'float64': '<f8', 'int64': '<i8'}[spec['dtype']]
            if (path.is_symlink() or not path.is_file() or path.stat().st_size != entry['bytes']
                    or header['shape'] != spec['shape'] or header['dtype'] != dtype):
                raise ValueError('Live proven TEST array header/size changed: ' + str(path))
            signatures[str(path)] = signature(path)
        signatures[str(metadata_path)] = signature(metadata_path)
    return signatures


def signature(path):
    stat = Path(path).stat()
    return [stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns]


def original_records(root, arm):
    """Validate both terminal failures before accepting either continuation."""
    from . import r102_experiment as experiment
    root = Path(root).resolve()
    frozen, absent, workers = {}, [], {}
    for side in ('A', 'B'):
        run = root / side
        dispatch = root.parent / 'R98_parallel_dispatch' / ('R102_' + side)
        config = read_json(run / 'protocol.json')
        lock = read_json(run / 'final_test_lock.json')
        train = read_json(run / 'training_complete.json')
        error = read_json(run / 'error.json')
        worker = read_json(dispatch / 'worker.json')
        controller = read_json(dispatch / 'controller.json')
        experiment.validate_checkpoint_config(config)
        update, sha = LOCKED[side]
        if (lock.get('selected_update') != update or lock.get('checkpoint_sha256') != sha
                or file_hash(run / 'best.pt') != sha or lock.get('arm') != side
                or lock.get('state_hash') != experiment.json_hash(config)
                or lock.get('checkpoint') != str((run / 'best.pt').resolve())
                or lock.get('inference_rule') != config['inference_rule']
                or lock.get('rule') != config['checkpoint_rule']
                or train.get('successful_updates') != 57600
                or train.get('state_hash') != lock['state_hash'] or train.get('selected_best') != lock
                or train.get('updates_by_problem') != dict.fromkeys(PROBLEMS, 3200)
                or read_json(run / 'status.json').get('phase') != 'failed'
                or worker.get('state') != 'failed' or controller.get('state') != 'failed'
                or worker.get('automatic_retry') is not False or controller.get('automatic_retry') is not False
                or error.get('message') != 'Full frozen R84 header identity differs'
                or 'r102_features.py' not in error.get('traceback', '')
                or 'r102_transfer.py' not in error.get('traceback', '')):
            raise ValueError('Both original locked TRAIN wrappers/controllers must be terminal with the actual header failure')
        transfer.authorize_test(lock, side)
        for name, sha in config['source_sha256'].items():
            if file_hash(V4 / name) != sha:
                raise ValueError('Original source pin changed: ' + name)
        for relative in ('test_results.json', 'test_complete.json', 'test_input_features_ready.json', 'evaluation/test'):
            path = run / relative
            if path.exists() or path.is_symlink():
                raise ValueError('This recovery is only for the not-yet-completed first TEST: ' + str(path))
            absent.append(str(path))
        names = ('best.pt', 'protocol.json', 'training_complete.json', 'final_test_lock.json',
                 'test_started.json', 'error.json', 'status.json', 'result.json', 'history.json',
                 'final_metrics.json', 'best_eval.json', 'input_features_ready.json', 'initialization.json')
        paths = [run / name for name in names]
        for split in ('train', 'val'):
            paths.extend(run / 'evaluation' / split / 'selected_best' / name
                         for name in ('identity.json', 'metrics.json', 'rank000.npz'))
        paths.extend(dispatch / name for name in ('worker.json', 'controller.json'))
        for path in paths:
            frozen[str(path)] = file_hash(path)
        if read_json(run / 'test_started.json').get('checkpoint_sha256') != lock['checkpoint_sha256']:
            raise ValueError('Original first-TEST marker names a different checkpoint')
        workers[side] = dict(gpu_uuid=worker['gpu_uuid'], cpus=worker['cpus'],
                             container_id=worker['container_id'], node=controller['node'])
    watcher = root / 'goal_watch_receipt.json'
    if read_json(watcher).get('state') != 'stopped_on_error':
        raise ValueError('Preserve the stopped original watcher; do not restart it')
    for path in (watcher, root / 'goal_progress.json', root / 'protocol_lock.json'):
        frozen[str(path)] = file_hash(path)
    run = root / arm
    config, lock = read_json(run / 'protocol.json'), read_json(run / 'final_test_lock.json')
    final, best = read_json(run / 'final_metrics.json'), read_json(run / 'best_eval.json')
    val = final['val']['identity']
    selected = read_json(run / 'evaluation/val/selected_best/identity.json')
    training = read_json(run / 'evaluation/train/selected_best/identity.json')
    if (selected != val or best['val']['identity'] != val or best['successful_updates'] != lock['selected_update']
            or final.get('selected_update') != lock['selected_update']
            or final.get('checkpoint_sha256') != lock['checkpoint_sha256']
            or final.get('state_hash') != lock['state_hash']
            or final['val']['macro'] != lock['val'] or final.get('test_read') is not False):
        raise ValueError('Original VAL summary/archive/lock do not bind the same selected update')
    for identity in (val, training):
        if (identity.get('state_hash') != lock['state_hash']
                or identity.get('successful_updates') != lock['selected_update']
                or not identity.get('model_state_sha256')
                or identity['model_state_sha256'] != val['model_state_sha256']
                or identity.get('checkpoint_sha256') not in (None, lock['checkpoint_sha256'])):
            raise ValueError('Selected TRAIN/VAL archive identity differs')
    if config['main_protocol_lock']['sha256'] != file_hash(root / 'protocol_lock.json'):
        raise ValueError('Original main protocol lock changed')
    return config, lock, dict(schema=SCHEMA, original_root=str(root), arm=arm,
        original_files=frozen, original_absent=absent, assignments=workers,
        selected_update=lock['selected_update'], checkpoint_sha256=lock['checkpoint_sha256'],
        state_hash=lock['state_hash'], model_state_sha256=val['model_state_sha256'],
        val_capture_checkpoint_sha256=val.get('checkpoint_sha256'), val_archive_modified=False,
        val_binding_basis='original selected update/protocol/full model state, verified again on strict restore',
        original_status='failed', original_watcher='stopped_on_error', training=False,
        optimizer_constructed=False, model_reselection=False, full_test_previously_completed=False)


def verify_origin(origin, *, check_cache=False):
    for path, sha in origin['original_files'].items():
        if file_hash(path) != sha:
            raise ValueError('Original immutable evidence changed: ' + path)
    for path in origin['original_absent']:
        if Path(path).exists() or Path(path).is_symlink():
            raise ValueError('Unexpected original TEST output: ' + path)
    for path, before in (origin.get('array_signatures', {}) if check_cache else {}).items():
        if signature(path) != before:
            raise ValueError('Original TEST cache changed during continuation: ' + path)


@contextmanager
def quiescent(root):
    with ExitStack() as stack:
        for arm in ('A', 'B'):
            handle = stack.enter_context((Path(root) / arm / 'runner.lock').open('rb'))
            fcntl.flock(handle, fcntl.LOCK_SH | fcntl.LOCK_NB)
        yield


def shadow(root, arm):
    destination = private(view(root) / arm, root)
    if destination.exists():
        raise FileExistsError('Preserve the existing first-TEST shadow, no implicit resume')
    destination.mkdir(parents=True)
    original = Path(root) / arm
    for name in ('protocol.json', 'training_complete.json', 'final_test_lock.json', 'final_metrics.json',
                 'best_eval.json', 'history.json', 'input_features_ready.json', 'initialization.json'):
        content = (original / name).read_bytes()
        if len(content) > 32 * 1024 * 1024:
            raise ValueError('Shadow permits only bounded metadata copies')
        with private(destination / name, root).open('xb') as stream:
            stream.write(content)
    (destination / 'best.pt').symlink_to((original / 'best.pt').resolve())
    (destination / 'evaluation').mkdir()
    for split in ('train', 'val'):
        (destination / 'evaluation' / split).symlink_to((original / 'evaluation' / split).resolve(), target_is_directory=True)
    return destination


def runtime_args(args, config, *, shadow_paths=False):
    from . import r102_experiment as experiment
    canonical = Path(args.root).resolve()
    values = dict(cache_root=config['cache_root'], feature_root=config['feature_cache_root'],
                  routing_feature_root=config['routing_feature_cache_root'], protocol_lock=str(canonical / 'protocol_lock.json'))
    for key, expected in values.items():
        supplied = getattr(args, key, None)
        if supplied is not None and Path(supplied).resolve() != Path(expected).resolve():
            raise ValueError('CLI must retain the original fixed ' + key)
    runtime = experiment.parser().parse_args(['test', '--root', str(view(canonical)), '--arm', args.arm])
    for key, value in values.items():
        setattr(runtime, key, Path(value).resolve())
    runtime.data_root, runtime.prepared_root = Path(config['data_root']), Path(config['prepared_root'])
    runtime.device, runtime.state_hash = args.device, experiment.json_hash(config)
    runtime.gpu_type, runtime.eval_batch_size = config['gpu_type'], config['eval_batch_size']
    runtime.large_micro_cap = config['microbatch_caps']['TSP']
    if shadow_paths:
        runtime.feature_root = base(canonical) / args.arm / 'test_features'
    return runtime


def normalize_gpu_uuid(value):
    return UUID(str(value).removeprefix('GPU-'))


def require_gpu(args, assignment):
    if args.device != 'cuda:0' or os.environ.get('CUDA_VISIBLE_DEVICES') != assignment['gpu_uuid']:
        raise ValueError('Expose only the ORIGINAL assigned UUID as --device cuda:0')
    if getattr(args, 'expected_uuid', None) not in (None, assignment['gpu_uuid']):
        raise ValueError('--expected-uuid differs from the original assignment')
    if set(os.sched_getaffinity(0)) != set(assignment['cpus']):
        raise ValueError('Retain the original assigned CPU affinity')
    if assignment['container_id'] not in Path('/proc/self/cgroup').read_text():
        raise ValueError('Use the original assigned allocation container')
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError('Exactly one original assigned GPU must be visible')
    props = torch.cuda.get_device_properties(0)
    raw_uuid = str(props.uuid)
    if normalize_gpu_uuid(raw_uuid) != normalize_gpu_uuid(assignment['gpu_uuid']):
        raise ValueError('Actual CUDA UUID differs from the original assignment')
    return torch.device(args.device), dict(uuid=assignment['gpu_uuid'], torch_uuid_raw=raw_uuid, name=props.name,
        cpus=sorted(os.sched_getaffinity(0)), torch_version=torch.__version__, cuda_version=torch.version.cuda)


def bitwise(left, right):
    import torch
    return (left.shape == right.shape and left.dtype == right.dtype
            and torch.equal(left.contiguous().view(torch.uint8), right.contiguous().view(torch.uint8)))


def timing_comparison(selector, raw_batch, cached_batch):
    """Observe the original forward path, never regroup or replace its inputs."""
    import torch
    from . import r102_experiment as experiment
    from .pairwise_selector import rank_by_solver_ids
    with torch.no_grad():
        raw, cached = experiment.raw_inputs(raw_batch), experiment.inputs_with_donor(cached_batch)
        online = selector.prepare_inputs(raw)
        left, right = selector(raw), selector(cached)
        masks = torch.equal(left['solver_mask'], right['solver_mask'])
        a, b = left['logits'][left['solver_mask']], right['logits'][right['solver_mask']]
        ids = raw['pool_ids']
        lp = ids[rank_by_solver_ids(left['logits'], ids)[:, 0]]
        rp = ids[rank_by_solver_ids(right['logits'], ids)[:, 0]]
        finite = bool(torch.isfinite(a).all() and torch.isfinite(b).all())
        return dict(nco_fp16_bitwise_exact=bitwise(online['nco_nodes'], cached['nco_nodes']),
            node_masks_exact=bitwise(online['nco_node_mask'], cached['nco_node_mask']),
            solver_masks_exact=masks, valid_scores_bitwise_exact=bitwise(a, b), finite=finite,
            prediction_changed_count=int((lp != rp).sum()),
            nco_unequal_elements=int((online['nco_nodes'] != cached['nco_nodes']).sum()),
            max_abs_nco_difference=float((online['nco_nodes'].float() - cached['nco_nodes'].float()).abs().max()),
            valid_score_unequal_elements=int((a != b).sum()) if a.shape == b.shape else None,
            max_abs_score_difference=float((a - b).abs().max()) if finite and a.shape == b.shape else None,
            inference_changed=False, grouping_changed=False)


def require_alignment(record):
    if (any(record.get(key) is not True for key in ('nco_fp16_bitwise_exact', 'node_masks_exact',
            'solver_masks_exact', 'valid_scores_bitwise_exact', 'finite'))
            or record.get('prediction_changed_count') != 0):
        raise ValueError('R102 cached-vs-online drift; preserve first error, do not change inference/grouping')


@contextmanager
def compatibility(args, config, lock, origin, evidence, *, observational_timing=False):
    from . import r102_experiment as experiment, r102_features as features
    from .r77_data import parameter_hash
    adapter = HeaderCompatibility(evidence)
    authorize, load, evaluate, measure = (experiment.authorize_test, experiment.load_checkpoint,
                                         experiment.evaluate, experiment.measure_inputs)
    runtime = runtime_args(args, config, shadow_paths=True)
    original_features = Path(config['feature_cache_root']).resolve()

    def authorized(actual):
        if (actual.root.resolve() != view(args.root) or actual.arm != args.arm
                or experiment.feature_root(actual).resolve() != runtime.feature_root.resolve()):
            raise PermissionError('Path adapter is valid only for this private arm shadow')
        original = copy.copy(actual)
        original.feature_root = original_features
        value = authorize(original)
        if value != (lock, config):
            raise PermissionError('Original R102 authorization changed')
        return value

    def strict_load(path, device='cpu'):
        if Path(path).resolve() != Path(lock['checkpoint']).resolve():
            raise PermissionError('Only the original unique VAL-selected checkpoint may be loaded')
        model, state = load(path, device)
        if (state['args'] != config or state['successful_updates'] != lock['selected_update']
                or parameter_hash(model.state_dict()) != origin['model_state_sha256']):
            raise PermissionError('Strict original restore differs from selected VAL model/update/buffers')
        model.requires_grad_(False).eval()
        return model, state

    def unchanged_evaluate(model, *values, **kwargs):
        before = parameter_hash(model.state_dict())
        result = evaluate(model, *values, **kwargs)
        if before != origin['model_state_sha256'] or parameter_hash(model.state_dict()) != before:
            raise ValueError('Model or buffers changed during frozen TEST')
        return result

    def checked_measure(selector, raw, cached, device, repetitions=3):
        problem = PROBLEMS[int(raw['problem_id'])]
        comparison = timing_comparison(selector, raw, cached)
        target = 'timing_observations' if observational_timing else 'timing_checks'
        if observational_timing:
            comparison.update(problem=problem, instances=128, scope='first128 TEST rows/task only',
                              numerical_boundary=NUMERICAL_BOUNDARY)
        save(base(args.root) / args.arm / target / (problem + '.json'), comparison, args.root, exclusive=True)
        if not observational_timing:
            require_alignment(comparison)
        return measure(selector, raw, cached, device, repetitions)

    with ExitStack() as stack:
        for module, name, value in (
                (transfer, 'validate_frozen_header', adapter.validate),
                (features, 'validate_frozen_header', adapter.validate),
                (transfer, '_validate_r94', adapter.validate_r94),
                (features, '_validate_r94', adapter.validate_r94),
                (experiment, 'authorize_test', authorized), (experiment, 'load_checkpoint', strict_load),
                (experiment, 'evaluate', unchanged_evaluate), (experiment, 'measure_inputs', checked_measure)):
            stack.enter_context(patch.object(module, name, value))
        yield runtime, adapter


def reporter_args(args, command):
    from . import r102_reporting as reporting
    argv = [command, '--root', str(view(args.root)), '--device', args.device, '--lib-batch-size', '2']
    if args.arm is not None:
        argv += ['--arm', args.arm]
    for key in ('nss_root', 'historical_root', 'r94_root'):
        value = getattr(args, key, None)
        if value is not None:
            argv += ['--' + key.replace('_', '-'), str(value)]
    return reporting.parser().parse_args(argv)


def first_error(root, arm, stage, error, *, filename='first_error.json'):
    path = base(root) / arm / filename
    if not path.exists():
        save(path, dict(schema=SCHEMA, stage=stage, complete=False, exit_status=1,
            type=type(error).__name__, message=str(error), traceback=traceback.format_exc(),
            failed_utc=utc(), automatic_retry=False, original_receipt_writes_by_helper=False), root, exclusive=True)


def stage(root, arm, name, call, origin, *, check_cache=False, allowed_failure=None):
    directory = base(root) / arm
    if (directory / 'first_error.json').exists():
        if allowed_failure is None or allowed_failure != known_timing_failure(root, arm):
            raise PermissionError('Only the explicitly declared known timing-gate failure may continue')
    if (directory / 'post_test_error.json').exists() or (directory / (name + '_started.json')).exists():
        raise PermissionError('One-shot stage already attempted/failed; no automatic retry')
    verify_origin(origin, check_cache=check_cache)
    save(directory / (name + '_started.json'), dict(schema=SCHEMA, started_utc=utc(),
        checkpoint_sha256=origin['checkpoint_sha256'], automatic_retry=False), root, exclusive=True)
    try:
        value = call()
        verify_origin(origin, check_cache=check_cache)
        return value
    except BaseException as error:
        first_error(root, arm, name, error, filename='post_test_error.json' if allowed_failure else 'first_error.json')
        raise


def known_timing_failure(root, arm):
    """The explicit exception is SHA-bound to these two actual timing failures."""
    directory = base(root) / arm
    pins = KNOWN_TIMING_FAILURES[arm]
    failure, observation = artifact(directory / 'first_error.json'), artifact(directory / 'timing_checks/TSP.json')
    if failure['sha256'] != pins[0] or observation['sha256'] != pins[1]:
        raise PermissionError('Unknown/changed prior failure; no post-TEST continuation authorized')
    error, check = read_json(failure['path']), read_json(observation['path'])
    if (error.get('stage') != 'timing' or error.get('type') != 'ValueError'
            or error.get('message') != 'R102 cached-vs-online drift; preserve first error, do not change inference/grouping'
            or error.get('exit_status') != 1 or error.get('complete') is not False
            or any(check.get(k) is not True for k in ('finite', 'node_masks_exact', 'solver_masks_exact'))
            or check.get('prediction_changed_count') != 0 or check.get('nco_fp16_bitwise_exact') is not False
            or check.get('valid_scores_bitwise_exact') is not False):
        raise PermissionError('Not the observed extra bitwise-gate failure')
    entries = {}
    for name in ('gpu_entry.json', 'gpu_uuid_format_correction_entry.json'):
        path = directory / name
        entry = read_json(path)
        if entry.get('state') != 'failed' or entry.get('exit_status') != 1:
            raise PermissionError('Preserve both failed original recovery entries')
        entries[name] = artifact(path)
    return dict(failure=failure, observation=observation, preserved_entries=entries)


def post_test_state(root, arm):
    from . import r102_reporting as reporting
    directory = base(root) / arm
    prior = known_timing_failure(root, arm)
    origin = read_json(directory / 'origin.json')
    verify_origin(origin)
    run, config, lock, test, val = reporting.locked_inputs(view(root), arm)
    complete = read_json(run / 'test_complete.json')
    if (LOCKED[arm] != (lock['selected_update'], lock['checkpoint_sha256'])
            or complete.get('full_tests') != 1 or complete.get('checkpoint_sha256') != lock['checkpoint_sha256']
            or test['identity']['model_state_sha256'] != origin['model_state_sha256']
            or val['identity']['model_state_sha256'] != origin['model_state_sha256']
            or any(test['per_problem'][p]['n'] != 4500 for p in PROBLEMS)):
        raise PermissionError('Require the actual completed first full TEST of the unchanged selected model')
    for name, sha in zip(('test_results.json', 'evaluation/test/selected_best/rank000.npz'), KNOWN_TIMING_FAILURES[arm][2:]):
        if file_hash(run / name) != sha:
            raise PermissionError('Completed first TEST changed before continuation: ' + name)
    for name, sha in config['source_sha256'].items():
        if file_hash(V4 / name) != sha:
            raise PermissionError('Original R102 source pin changed: ' + name)
        origin['original_files'][str(V4 / name)] = sha
    protected = [run / name for name in ('test_results.json', 'test_complete.json', 'test_started.json',
        'test_input_features_ready.json', 'final_metrics.json', 'protocol.json', 'final_test_lock.json')]
    protected += [p for p in (run / 'evaluation/test').rglob('*') if p.is_file()]
    protected += [directory / name for name in ('first_error.json', 'timing_checks/TSP.json',
        'timing_started.json', 'test_started.json', 'gpu_entry.json', 'gpu_uuid_format_correction_entry.json', 'origin.json')]
    origin['original_files'].update({str(p): file_hash(p) for p in protected})
    origin.update(prior_origin=artifact(directory / 'origin.json'), prior_timing_failure=prior,
                  completion_kind=POST_TEST_KIND, full_test_previously_completed=True,
                  test_rerun=False, numerical_boundary=NUMERICAL_BOUNDARY)
    return config, lock, origin


def finish_gpu(args):
    """Explicitly authorized timing/LIB only; never call TEST, prepare or train."""
    if args.arm is None or args.device != 'cuda:0':
        raise ValueError('finish-gpu requires --arm A|B --device cuda:0')
    from . import r102_experiment as experiment, r102_reporting as reporting
    import torch
    root, directory = args.root, base(args.root) / args.arm
    with quiescent(root):
        config, lock, origin = post_test_state(root, args.arm)
        runtime_args(args, config)
        device, hardware = require_gpu(args, origin['assignments'][args.arm])
        verify_origin(origin, check_cache=True)
        proof = origin['array_proof']
        evidence = validate_proof(checked_artifact(proof['proof']), checked_artifact(proof['manifest']))
        if args.proof is not None and artifact(args.proof) != proof['proof']:
            raise PermissionError('Retain the original cross-node TEST array proof')
        with private(directory / 'finish_gpu.lock', root).open('a') as claim:
            fcntl.flock(claim, fcntl.LOCK_EX | fcntl.LOCK_NB)
            for path in (view(root) / args.arm / 'inference_timing.json', view(root) / args.arm / 'lib_evaluation',
                         directory / 'gpu_complete.json', directory / 'post_test_error.json'):
                if path.exists():
                    raise FileExistsError('Preserve existing post-TEST attempt/output: ' + str(path))
            save(directory / 'finish_gpu_provenance.json', origin, root, exclusive=True)
            try:
                torch.set_num_threads(2)
                torch.set_num_interop_threads(2)
                with compatibility(args, config, lock, origin, evidence, observational_timing=True) as (runtime, adapter):
                    stage(root, args.arm, 'timing_observation', lambda: experiment.timing(runtime, device), origin,
                          check_cache=True, allowed_failure=origin['prior_timing_failure'])
                    stage(root, args.arm, 'lib_completion', lambda: reporting.lib_stage(reporter_args(args, 'lib')), origin,
                          check_cache=True, allowed_failure=origin['prior_timing_failure'])
                run = view(root) / args.arm
                reporting.inference_timing(run, lock)
                reporting._completed_lib(run, lock)
                observations = {p: artifact(directory / 'timing_observations' / (p + '.json')) for p in PROBLEMS}
                verify_origin(origin, check_cache=True)
                receipt = dict(schema=SCHEMA, experiment='R102', arm=args.arm, complete=True, exit_status=0,
                    completed_utc=utc(), completion_kind=POST_TEST_KIND, stages=['timing_observation', 'lib_completion'],
                    stage_exit_statuses=dict(timing_observation=0, lib_completion=0),
                    origin=artifact(directory / 'finish_gpu_provenance.json'), prior_timing_failure=origin['prior_timing_failure'],
                    report_view=str(view(root)), run=str(run), checkpoint_sha256=lock['checkpoint_sha256'],
                    state_hash=lock['state_hash'], selected_update=lock['selected_update'],
                    model_state_sha256=origin['model_state_sha256'], hardware=hardware, compatibility_uses=adapter.uses,
                    training=False, test_rerun=False, model_reselection=False, original_failures_preserved=True,
                    numerical_boundary=NUMERICAL_BOUNDARY, online_efficacy_bitwise_certified=False,
                    timing_observations=observations, outputs={name: artifact(run / name) for name in
                        ('test_results.json', 'test_complete.json', 'evaluation/test/selected_best/identity.json',
                         'evaluation/test/selected_best/rank000.npz', 'inference_timing.json', 'lib_stage_complete.json')})
                save(directory / 'gpu_complete.json', receipt, root, exclusive=True)
                return receipt
            except BaseException as error:
                first_error(root, args.arm, 'finish-gpu', error, filename='post_test_error.json')
                raise


def gpu(args):
    if args.arm is None or args.device != 'cuda:0':
        raise ValueError('gpu requires --arm A|B --device cuda:0; no GPU allocation occurs here')
    root = args.root
    with quiescent(root):
        config, lock, origin = original_records(root, args.arm)
        runtime_args(args, config)
        device, hardware = require_gpu(args, origin['assignments'][args.arm])
        proof_path = args.proof or base(root) / 'array_equality_proof.json'
        evidence = validate_proof(proof_path, Path(config['prepared_root']) / 'source_cache_manifest.json')
        origin.update(array_proof={k: v for k, v in evidence.items() if k != 'pairs'},
            array_signatures=validate_local_cache(config['cache_root'], evidence), hardware=hardware,
            report_view=str(view(root)), original_protocol_lock=str(root / 'protocol_lock.json'),
            header_compatibility=dict(only_fields=sorted(BUILD_FIELDS), process_local=True,
                frozen_headers_rewritten=False, arrays_rewritten=False,
                r94_digest_policy='only the exact proven actual/reference digests; all other original checks retained'),
            original_feature_root=config['feature_cache_root'],
            recovery_test_feature_root=str(base(root) / args.arm / 'test_features'))
        directory = base(root) / args.arm
        private(directory, root).mkdir(parents=True, exist_ok=True)
        with (directory / 'continuation.lock').open('a') as claim:
            fcntl.flock(claim, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if (directory / 'origin.json').exists() or (directory / 'first_error.json').exists():
                raise FileExistsError('Preserve original continuation attempt; no implicit resume/retry')
            save(directory / 'origin.json', origin, root, exclusive=True)
            try:
                run = shadow(root, args.arm)
                from . import r102_experiment as experiment, r102_reporting as reporting
                import torch
                torch.set_num_threads(2)
                torch.set_num_interop_threads(2)
                with compatibility(args, config, lock, origin, evidence) as (runtime, adapter):
                    # Mirror the original TEST entry setup, not a new numerical recipe.
                    initial = (torch.get_float32_matmul_precision(), torch.backends.cuda.matmul.allow_tf32,
                               torch.backends.cudnn.allow_tf32, torch.backends.cudnn.benchmark)
                    try:
                        experiment.engine.configure_torch()
                        stage(root, args.arm, 'test', lambda: experiment.test(runtime, device), origin, check_cache=True)
                    finally:
                        torch.set_float32_matmul_precision(initial[0])
                        torch.backends.cuda.matmul.allow_tf32 = initial[1]
                        torch.backends.cudnn.allow_tf32, torch.backends.cudnn.benchmark = initial[2:]
                    stage(root, args.arm, 'timing', lambda: experiment.timing(runtime, device), origin, check_cache=True)
                    stage(root, args.arm, 'lib', lambda: reporting.lib_stage(reporter_args(args, 'lib')), origin, check_cache=True)
                reporting.inference_timing(run, lock)
                reporting._completed_lib(run, lock)
                receipt = dict(schema=SCHEMA, experiment='R102', arm=args.arm, complete=True,
                    exit_status=0, completed_utc=utc(), stages=['first_completed_TEST', 'timing', 'LIB'],
                    origin=artifact(directory / 'origin.json'), report_view=str(view(root)),
                    run=str(run), checkpoint_sha256=lock['checkpoint_sha256'], state_hash=lock['state_hash'],
                    selected_update=lock['selected_update'], model_state_sha256=origin['model_state_sha256'],
                    original_failures_preserved=True, training=False, model_reselection=False,
                    compatibility_uses=adapter.uses,
                    outputs={name: artifact(run / name) for name in ('test_results.json', 'test_complete.json',
                        'evaluation/test/selected_best/identity.json', 'evaluation/test/selected_best/rank000.npz',
                        'inference_timing.json', 'lib_stage_complete.json')})
                verify_origin(origin, check_cache=True)
                save(directory / 'gpu_complete.json', receipt, root, exclusive=True)
                return receipt
            except BaseException as error:
                first_error(root, args.arm, 'gpu', error)
                raise


def checked_completion(root, arm, name):
    directory = base(root) / arm
    if (directory / 'post_test_error.json').exists():
        raise PermissionError('Post-TEST continuation failed; no later stages')
    receipt = read_json(directory / (name + '_complete.json'))
    if (directory / 'first_error.json').exists():
        if (receipt.get('completion_kind') != POST_TEST_KIND
                or receipt.get('prior_timing_failure') != known_timing_failure(root, arm)
                or receipt.get('stage_exit_statuses') != dict(timing_observation=0, lib_completion=0)
                or receipt.get('test_rerun') is not False
                or receipt.get('online_efficacy_bitwise_certified') is not False
                or receipt.get('numerical_boundary') != NUMERICAL_BOUNDARY
                or set(receipt.get('timing_observations', {})) != set(PROBLEMS)):
            raise PermissionError('Only declared successful original-timing completion may follow the known failure')
        for problem, item in receipt['timing_observations'].items():
            if checked_artifact(item) != directory / 'timing_observations' / (problem + '.json'):
                raise PermissionError('Timing observation path differs from the explicit continuation')
    lock = read_json(Path(root) / arm / 'final_test_lock.json')
    if (receipt.get('schema') != SCHEMA or receipt.get('complete') is not True or receipt.get('exit_status') != 0
            or receipt.get('arm') != arm or receipt.get('checkpoint_sha256') != lock['checkpoint_sha256']
            or receipt.get('state_hash') != lock['state_hash'] or receipt.get('report_view') != str(view(root))):
        raise ValueError('Require real same-lock recovery completion, not original failed receipts')
    origin = read_json(checked_artifact(receipt['origin']))
    verify_origin(origin)
    for record in receipt['outputs'].values():
        checked_artifact(record)
    return receipt, origin


def numerical_boundary_report(root, arm, receipt):
    """Separate CPU report page; never rewrite the archived TEST or its metrics."""
    import textwrap
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    output = view(root) / arm / 'full_model_statistics'
    lines = [NUMERICAL_BOUNDARY, '', 'Problem | FP16 H exact | scores exact | sampled changed picks | max abs score delta']
    for problem in PROBLEMS:
        row = read_json(checked_artifact(receipt['timing_observations'][problem]))
        lines.append(f"{problem}: {row['nco_fp16_bitwise_exact']} | {row['valid_scores_bitwise_exact']} | "
                     f"{row['prediction_changed_count']}/128 | {row['max_abs_score_difference']}")
    note, pdf = (private(output / name, root) for name in ('numerical_boundary.md', 'numerical_boundary.pdf'))
    with note.open('x', encoding='utf-8') as stream:
        stream.write('# R102 Numerical Boundary\n\n' + '\n'.join(lines) + '\n')
    if pdf.exists():
        raise FileExistsError('Preserve existing numerical boundary PDF')
    fig = plt.figure(figsize=(12, 11))
    try:
        fig.text(.06, .94, 'R102/' + arm + ': Cached TEST vs Original Online Timing', fontsize=16)
        fig.text(.06, .88, '\n'.join(textwrap.wrap(lines[0], 110)), fontsize=10, va='top')
        fig.text(.06, .73, '\n'.join(lines[2:]), family='monospace', fontsize=9, va='top', linespacing=1.6)
        with PdfPages(pdf) as pages:
            pages.savefig(fig)
    finally:
        plt.close(fig)
    return dict(note=artifact(note), pdf=artifact(pdf))


def report(args):
    if args.device != 'cpu' or args.arm is None:
        raise ValueError('report requires --device cpu --arm A|B')
    receipt, origin = checked_completion(args.root, args.arm, 'gpu')
    config = read_json(args.root / args.arm / 'protocol.json')
    runtime_args(args, config)
    from . import r102_reporting as reporting
    run = view(args.root) / args.arm

    def finish():
        reporting.report(reporter_args(args, 'report'))
        reporting.completed_arm_report(run, read_json(run / 'final_test_lock.json'))
        boundary = numerical_boundary_report(args.root, args.arm, receipt) if receipt.get('completion_kind') == POST_TEST_KIND else {}
        if boundary:
            path = run / 'full_model_statistics/protocol.json'
            protocol = read_json(path)
            protocol.update(recovery_numerical_boundary=NUMERICAL_BOUNDARY,
                            online_efficacy_bitwise_certified=False, numerical_boundary_artifacts=boundary,
                            timing_observations=receipt['timing_observations'])
            save(path, protocol, args.root)
        record = dict(receipt, stages=['CPU_report'], completed_utc=utc(),
            gpu_completion=artifact(base(args.root) / args.arm / 'gpu_complete.json'),
            outputs={name: artifact(run / name) for name in ('postprocess_status.json', 'full_model_statistics/protocol.json')})
        record['outputs'].update({'numerical_boundary_' + k: v for k, v in boundary.items()})
        for name in reporting.REQUIRED_PDFS:
            record['outputs'][name] = artifact(run / 'full_model_statistics' / name)
        return record

    receipt = stage(args.root, args.arm, 'report', finish, origin, allowed_failure=receipt.get('prior_timing_failure'))
    save(base(args.root) / args.arm / 'report_complete.json', receipt, args.root, exclusive=True)
    return receipt


def compare(args):
    if args.device != 'cpu':
        raise ValueError('compare requires --device cpu')
    origins = [checked_completion(args.root, arm, 'report')[1] for arm in ('A', 'B')]
    from . import r102_reporting as reporting
    for arm in ('A', 'B'):
        runtime_args(SimpleNamespace(**dict(vars(args), arm=arm)), read_json(args.root / arm / 'protocol.json'))
    result = stage(args.root, 'comparison', 'compare', lambda: reporting.compare(reporter_args(args, 'compare')), origins[0])
    for origin in origins:
        verify_origin(origin)
    receipt = dict(schema=SCHEMA, experiment='R102', complete=True, exit_status=0,
        completed_utc=utc(), report_view=str(view(args.root)), training=False, inference=False,
        reports={arm: artifact(base(args.root) / arm / 'report_complete.json') for arm in ('A', 'B')}, result=result)
    save(base(args.root) / 'compare_complete.json', receipt, args.root, exclusive=True)
    return receipt


def describe(args):
    proof = args.proof or base(args.root) / 'array_equality_proof.json'
    manifest = V4 / 'runs/R84_new90k_attention/prepared/source_cache_manifest.json'
    evidence = validate_proof(proof, manifest)
    return dict(schema=SCHEMA, proof_valid=True, proof=evidence['proof'],
        array_count=evidence['array_count'], problem_count=18, allowed_header_fields=sorted(BUILD_FIELDS),
        base=str(base(args.root)), report_view=str(view(args.root)), commands=['gpu', 'finish-gpu', 'report', 'compare', 'describe'],
        gpu_stages=['first_completed_TEST', 'timing', 'LIB'], CPU_stages=['report', 'compare'],
        finish_gpu_stages=['timing_observation', 'lib_completion'], finish_gpu_test_rerun=False,
        gpu_receipt='BASE/{arm}/gpu_complete.json', report_receipt='BASE/{arm}/report_complete.json',
        compare_receipt='BASE/compare_complete.json', original_protocol_lock=str(args.root / 'protocol_lock.json'),
        training=False, inference=False, cuda_initialized=False, array_payloads_opened=False)


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('command', choices=('gpu', 'finish-gpu', 'report', 'compare', 'describe'))
    result.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    result.add_argument('--arm', choices=('A', 'B'))
    result.add_argument('--device', default='cpu')
    for key in ('cache-root', 'feature-root', 'routing-feature-root', 'protocol-lock',
                'proof', 'nss-root', 'historical-root', 'r94-root'):
        result.add_argument('--' + key, type=Path)
    result.add_argument('--expected-uuid')
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    args.root = args.root.expanduser().resolve()
    value = {'gpu': gpu, 'finish-gpu': finish_gpu, 'report': report, 'compare': compare, 'describe': describe}[args.command](args)
    print(json.dumps(value, indent=2, allow_nan=False))
    return value


if __name__ == '__main__':
    main()
