"""Disjoint R99 postprocessing recovery. Never train, rebuild features, or launch GPUs.

The launcher owns allocation, container and CUDA visibility. All new artifacts
belong below R99/preparation/postprocess_recovery; original receipts stay intact.
"""

import argparse
from contextlib import contextmanager
import fcntl
import importlib.util
import json
import math
import os
from pathlib import Path
import signal
import sys
import time

import numpy as np
import torch

from .r76_experiment import file_hash, read_json, utc
from . import r99_experiment as experiment
from .r99_features import FORMAT, digest, source_identity, close_array
from .r99_model import inputs_with_relations
from .r99_relations import SPEC, WIDTH, transport
from .pairwise_selector import rank_by_solver_ids


DEFAULT_ROOT = experiment.DEFAULT_ROOT
RECOVERY_RELATIVE = Path('preparation/postprocess_recovery')
SCHEMA = 'r99_postprocess_recovery_v1'
PATHS = ('mixed', 'exact_n')


def save_new(path, value):
    """Atomic replacement is allowed only for private recovery artifacts."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name('.' + path.name + '.' + str(os.getpid()))
    with temporary.open('x', encoding='utf-8') as output:
        json.dump(value, output, indent=2, allow_nan=False)
        output.write('\n')
    temporary.replace(path)


def recovery_root(root):
    return Path(root).resolve() / RECOVERY_RELATIVE


def assert_private(path, root):
    path = Path(path)
    if not path.resolve().is_relative_to(recovery_root(root)):
        raise ValueError('Writes must stay inside preparation/postprocess_recovery')
    return path


@contextmanager
def claimed(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def provenance(root, arm):
    """Hash original failure evidence; never copy launcher argv or credentials."""
    run = Path(root) / arm
    config = read_json(run / 'protocol.json')
    lock = read_json(run / 'final_test_lock.json')
    train = read_json(run / 'training_complete.json')
    test = read_json(run / 'test_results.json')
    experiment.validate_checkpoint_config(config)
    if (train.get('successful_updates') != 57600 or test.get('complete') is not True
            or test.get('checkpoint_sha256') != lock.get('checkpoint_sha256')
            or test.get('state_hash') != lock.get('state_hash')
            or lock.get('arm') != arm or config.get('arm') != arm
            or lock.get('selected_without_test') is not True
            or lock.get('per_problem_assembly') is not False
            or lock.get('state_hash') != experiment.json_hash(config)
            or lock.get('inference_rule') != config.get('inference_rule')
            or file_hash(run / 'best.pt') != lock.get('checkpoint_sha256')):
        raise ValueError('Recovery requires completed TRAIN and unchanged locked full TEST/checkpoint')
    dispatch = Path(root).parent / 'R98_parallel_dispatch' / ('R99_' + arm)
    failures = {}
    for name in ('worker.json', 'controller.json', 'timing.log', 'allocation_GPU_pipeline.log'):
        path = dispatch / name
        if not path.is_file():
            raise FileNotFoundError('Missing original failure evidence: ' + str(path))
        failures[name] = dict(path=str(path.resolve()), sha256=file_hash(path))
    worker = read_json(dispatch / 'worker.json')
    stages = {item['name']: item for item in worker.get('stages', [])}
    if (worker.get('state') != 'failed' or stages.get('TEST', {}).get('exit_status') != 0
            or stages.get('timing', {}).get('exit_status') != 1):
        raise ValueError('Expected original rc0 TEST followed by failed timing, not a healthy worker')
    frozen = {}
    for name in ('protocol.json', 'training_complete.json', 'final_test_lock.json',
                 'test_results.json', 'test_complete.json', 'final_metrics.json',
                 'input_features_ready.json', 'test_input_features_ready.json', 'best_eval.json',
                 'evaluation/train/selected_best/identity.json',
                 'evaluation/train/selected_best/rank000.npz',
                 'evaluation/val/selected_best/identity.json',
                 'evaluation/val/selected_best/rank000.npz',
                 'evaluation/test/selected_best/identity.json',
                 'evaluation/test/selected_best/rank000.npz'):
        path = run / name
        frozen[name] = dict(path=str(path.resolve()), sha256=file_hash(path))
    identity = dict(schema=SCHEMA, experiment='R99', arm=arm,
        checkpoint_sha256=lock['checkpoint_sha256'], state_hash=lock['state_hash'],
        lock=lock, lock_sha256=frozen['final_test_lock.json']['sha256'],
        original_failures=failures, original_failure_sha256=digest(failures),
        original_frozen=frozen, selected_update=lock['selected_update'],
        training_rerun=False, full_test_rerun=False, feature_rebuild=False,
        feature_copy=False, decision_rule_changed=False)
    return config, lock, identity


class InputReader:
    """Read only input arrays, not ScenarioCache's labels or comparison targets."""

    def __init__(self, config, problem, lock):
        self.arrays = {}
        self.problem = problem
        self.directory = Path(config['cache_root']) / 'test' / problem
        self.feature_directory = Path(config['feature_cache_root']) / 'test' / problem
        try:
            self.metadata = record = read_json(self.directory / 'metadata.json')
            manifest = read_json(Path(config['prepared_root']) / 'source_cache_manifest.json')
            expected = manifest['splits']['test'][problem]
            if (record.get('complete') is not True or record.get('rows_written') != 4500
                    or record.get('rows') != 4500 or record.get('problem') != problem
                    or record.get('split') != 'test'
                    or Path(record['data_root']).resolve() != Path(config['data_root']).resolve()
                    or any(record[k] != expected.get(k) for k in record
                           if k not in ('directory', 'cache_root'))):
                raise ValueError('Input cache disagrees with the frozen source manifest')
            feature = read_json(self.feature_directory / 'metadata.json')
            identity = dict(format=FORMAT, problem=problem, split='test',
                variant=experiment.VARIANTS[config['arm']], transform=SPEC,
                source_identity=source_identity(record), rows=4500, input_only=True,
                labels_opened=False, test_lock_sha256=digest(lock))
            if (feature.get('complete') is not True or feature.get('identity') != identity
                    or feature.get('dtype') != 'float16' or feature.get('width') != WIDTH):
                raise ValueError('Existing messages do not match the unchanged original cache/lock')
            keys = ['n', 'coord_dist'] + (['node', 'node_geom'] if record['kind'] == 'coord' else ['matrix'])
            for key in keys:
                spec = record['arrays'][key]
                array = np.load(self.directory / spec['file'], mmap_mode='r', allow_pickle=False)
                self.arrays[key] = array
                if list(array.shape) != spec['shape'] or str(array.dtype) != spec['dtype']:
                    raise ValueError('Input array schema differs: ' + key)
            self.lengths = self.arrays['n']
            self.offsets = np.load(self.feature_directory / 'offsets.npy', mmap_mode='r', allow_pickle=False)
            self.messages = np.load(self.feature_directory / 'messages.npy', mmap_mode='r', allow_pickle=False)
            if (self.offsets.dtype != np.int64 or self.offsets.shape != (4501,)
                    or not np.array_equal(np.diff(self.offsets), self.lengths)
                    or self.messages.dtype != np.float16
                    or self.messages.shape != (int(self.offsets[-1]), WIDTH)):
                raise ValueError('Existing ragged FP16 cache is not aligned with original real N')
        except BaseException:
            self.close()
            raise

    def batch(self, rows, device, *, cached=False):
        rows = np.asarray(rows, dtype=np.int64)
        record = self.metadata
        lengths = np.array(self.lengths[rows], copy=True)
        count, maximum = len(rows), int(lengths.max())
        mask = np.arange(maximum)[None] < lengths[:, None]
        batch = dict(problem_id=record['problem_id'], kind=record['kind'],
            n=torch.from_numpy(lengths), node_mask=torch.from_numpy(mask),
            cbits=torch.tensor(record['cbits'], dtype=torch.float32).repeat(count, 1),
            problem_desc=torch.tensor(record['problem_desc'], dtype=torch.float32).repeat(count, 1),
            coord_dist=torch.from_numpy(np.array(self.arrays['coord_dist'][rows], copy=True)),
            pool_ids=torch.tensor(record['pool_ids'], dtype=torch.long),
            mask=torch.tensor(record['mask'], dtype=torch.float32).repeat(count, 1),
            solver_mask=torch.ones(count, len(record['pool']), dtype=torch.bool))
        if record['kind'] == 'coord':
            batch['d_in'] = record['d_in']
            for key in ('node', 'node_geom'):
                batch[key] = torch.from_numpy(np.ascontiguousarray(self.arrays[key][:, :maximum][rows]))
        else:
            batch['matrix'] = torch.from_numpy(np.ascontiguousarray(self.arrays['matrix'][:, :maximum, :maximum][rows]))
        if cached:
            positions = self.offsets[rows, None] + np.arange(maximum)[None]
            values = np.ascontiguousarray(self.messages[np.where(mask, positions, 0)])
            values[~mask] = 0
            batch['relation_nodes'] = torch.from_numpy(values)
            batch['relation_node_mask'] = batch['node_mask']
        return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}

    def close(self):
        for array in [*self.arrays.values(), getattr(self, 'offsets', None), getattr(self, 'messages', None)]:
            close_array(array)
        self.arrays = {}


def numerical_policy():
    return dict(matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
        cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
        float32_matmul_precision=torch.get_float32_matmul_precision())


@contextmanager
def tf32_policy(policy):
    previous = numerical_policy()
    try:
        if policy != 'default':
            torch.set_float32_matmul_precision('high' if policy == 'on' else 'highest')
            torch.backends.cuda.matmul.allow_tf32 = policy == 'on'
            torch.backends.cudnn.allow_tf32 = policy == 'on'
        yield numerical_policy()
    finally:
        torch.set_float32_matmul_precision(previous['float32_matmul_precision'])
        torch.backends.cuda.matmul.allow_tf32 = previous['matmul_allow_tf32']
        torch.backends.cudnn.allow_tf32 = previous['cudnn_allow_tf32']


def difference(actual, reference, *, rtol=1e-3, atol=1e-5):
    if actual.shape != reference.shape or actual.dtype != reference.dtype:
        raise ValueError('Comparison requires identical shape and dtype')
    actual, reference = actual.detach().cpu().contiguous(), reference.detach().cpu().contiguous()
    finite = torch.isfinite(actual) & torch.isfinite(reference)
    delta = (actual.double() - reference.double()).abs()
    relative = delta / reference.double().abs().clamp_min(1e-12)
    bits = actual.view(torch.uint8).ne(reference.view(torch.uint8))
    return dict(elements=actual.numel(), value_exact=torch.equal(actual, reference),
        bitwise_exact=not bool(bits.any()), unequal_elements=int((actual != reference).sum()),
        unequal_bytes=int(bits.sum()), nonfinite_count=int((~finite).sum()),
        max_abs=float(delta[finite].max()) if finite.any() else None,
        max_relative=float(relative[finite].max()) if finite.any() else None,
        absolute_bad_count=int((delta > atol).sum()), relative_bad_count=int((relative > rtol).sum()),
        tolerance_bad_count=int((delta > atol + rtol * reference.double().abs()).sum()),
        relative_denominator_floor=1e-12,
        diagnostic_tolerances_only=dict(rtol=rtol, atol=atol),
        approximate_equality_is_not_exact=True)


def real_n_groups(raw):
    n = raw['n']
    mask = raw['node_mask']
    expected = torch.arange(mask.shape[1], device=mask.device)[None] < n[:, None]
    if not torch.equal(mask, expected):
        raise ValueError('Only original contiguous real-node masks are supported')
    return [(int(value), torch.where(n == value)[0]) for value in torch.unique(n, sorted=True)]


def crop_rows(raw, indices, count):
    batch = {}
    for key, value in raw.items():
        if torch.is_tensor(value) and key != 'pool_ids' and value.ndim and value.shape[0] == len(raw['n']):
            value = value.index_select(0, indices)
            if key in ('node_mask', 'node', 'node_geom'):
                value = value[:, :count]
            elif key == 'matrix':
                value = value[:, :count, :count]
        batch[key] = value
    return batch


@torch.no_grad()
def transform_path(raw, variant, path):
    if path == 'mixed':
        return transport(raw, variant)
    if path != 'exact_n':
        raise ValueError('Unknown diagnostic path')
    messages = torch.zeros((*raw['node_mask'].shape, WIDTH), dtype=torch.float16, device=raw['n'].device)
    residual = messages.new_zeros((), dtype=torch.float32)
    for count, indices in real_n_groups(raw):
        values, error = transport(crop_rows(raw, indices, count), variant)
        messages[indices, :count] = values
        residual = torch.maximum(residual, error)
    return messages, residual


def prediction(logits, pool_ids):
    return pool_ids[rank_by_solver_ids(logits, pool_ids)[:, 0]].detach().cpu().tolist()


@torch.no_grad()
def online_forward(selector, raw, variant, path):
    if path == 'mixed':
        return selector(raw)
    messages, _ = transform_path(raw, variant, path)
    return selector(dict(raw, relation_nodes=messages, relation_node_mask=raw['node_mask']))


def aligned_comparison(selector, raw, cached, variant, path):
    messages, residual = transform_path(raw, variant, path)
    if path == 'mixed':
        messages = selector.prepare_inputs(raw)['relation_nodes']
    mask_equal = torch.equal(raw['node_mask'], cached['relation_node_mask'])
    if not mask_equal:
        raise ValueError('Original input masks and cached messages differ')
    actual, reference = online_forward(selector, raw, variant, path), selector(cached)
    if not torch.equal(actual['solver_mask'], reference['solver_mask']):
        raise ValueError('Online and cached candidate masks differ')
    mask = reference['solver_mask']
    if not torch.isfinite(actual['logits'][mask]).all() or not torch.isfinite(reference['logits'][mask]).all():
        raise ValueError('Nonfinite diagnostic score')
    left, right = prediction(actual['logits'], raw['pool_ids']), prediction(reference['logits'], raw['pool_ids'])
    return dict(messages=difference(messages, cached['relation_nodes']),
        scores=difference(actual['logits'][mask], reference['logits'][mask]),
        prediction_changed_count=sum(a != b for a, b in zip(left, right)),
        online_predictions_global_ids=left, cached_predictions_global_ids=right,
        max_marginal_residual=float(residual), masks_exact=mask_equal)


@torch.no_grad()
def measure(selector, raw, cached, variant, path, device, repetitions=3):
    if repetitions != 3:
        raise ValueError('Recovery preserves three timing repetitions')
    calls = dict(transform_only=lambda: transform_path(raw, variant, path),
        transform_plus_selector=lambda: online_forward(selector, raw, variant, path),
        cache_hit_selector=lambda: selector(cached))
    rows = {}
    for scope, call in calls.items():
        call()
        experiment._synchronize(device)
        samples = []
        for _ in range(repetitions):
            experiment._synchronize(device)
            start = time.perf_counter()
            value = call()
            experiment._synchronize(device)
            samples.append(time.perf_counter() - start)
            if scope != 'transform_only' and not torch.isfinite(value['logits'][value['solver_mask']]).all():
                raise ValueError('Nonfinite locked timing prediction')
        rows[scope] = dict(batch_seconds=samples,
            mean_seconds_per_instance=sum(samples) / (repetitions * len(raw['n'])))
    return rows


def stage_receipt_path(root, arm, stage):
    return recovery_root(root) / arm / (stage + '_stage.json')


@torch.no_grad()
def timing(args):
    device, hardware = require_gpu(args)
    output = assert_private(args.output or recovery_root(args.root) / args.arm / 'inference_timing.json', args.root)
    receipt = stage_receipt_path(args.root, args.arm, 'timing')
    with claimed(output.with_suffix('.lock')), bounded(args.max_seconds):
        if output.exists() or receipt.exists():
            raise FileExistsError('Preserve previous timing attempt; no automatic retry')
        config, lock, origin = provenance(args.root, args.arm)
        validate_runtime_paths(args, config)
        value = dict(**origin, command='timing', recovery_schema=SCHEMA, complete=False,
            started_utc=utc(), hardware=hardware, variant=experiment.VARIANTS[args.arm],
            inference_rule=lock['inference_rule'], device=str(device), repetitions=3, warmup_batches=1,
            timing_path=args.timing_path, requested_tf32_policy=args.timing_tf32_policy,
            scope='same first128 TEST input rows/task, GPU-resident original inputs; synchronized wall time',
            raw_input_source='original_base_cache_without_relation_fields',
            online_transform_verified=True, includes_input_transform=True, includes_solver_decoder=False,
            full_system_measured=False, end_to_end_measured=False, model_reselection=False,
            labels_opened=False, efficacy_reused_as_exact_online=False,
            excludes='dataset I/O, host transfer and feature-cache preparation; cache-hit is not full inference',
            equivalence_scope='first128 rows/task only; cached full TEST efficacy is not online efficacy',
            preprocessing=read_json(args.root / args.arm / 'input_features_ready.json'),
            test_preprocessing=read_json(args.root / args.arm / 'test_input_features_ready.json'), per_problem=[])
        save_new(output, value)
        save_new(receipt, dict(**origin, stage='timing', complete=False, started_utc=utc()))
        try:
            selector, state = experiment.load_checkpoint(args.root / args.arm / 'best.pt', device)
            if state['args'] != config or state.get('successful_updates') != lock['selected_update']:
                raise ValueError('Timing factory returned another checkpoint/configuration')
            del state
            selector.eval().requires_grad_(False)
            with tf32_policy(args.timing_tf32_policy) as effective:
                value['effective_policy'] = effective
                for problem in experiment.PROBLEMS:
                    reader = InputReader(config, problem, lock)
                    try:
                        raw = reader.batch(np.arange(128), device)
                        cached = reader.batch(np.arange(128), device, cached=True)
                        comparison = aligned_comparison(selector, raw, cached, value['variant'], args.timing_path)
                        original = comparison if args.timing_path == 'mixed' else aligned_comparison(
                            selector, raw, cached, value['variant'], 'mixed')
                        measurements = measure(selector, raw, cached, value['variant'], args.timing_path, device)
                        value['per_problem'].append(dict(problem=problem, instances=128,
                            **measurements, max_marginal_residual=comparison['max_marginal_residual'],
                            comparison=comparison, original_mixed_comparison=original))
                        save_new(output, value)
                        print(json.dumps(dict(stage='timing', problem=problem,
                            online_fp16_matches_cache=comparison['messages']['bitwise_exact'],
                            original_mixed_matches_cache=original['messages']['bitwise_exact'],
                            prediction_changed_count=comparison['prediction_changed_count'])), flush=True)
                    finally:
                        reader.close()
            value.update(complete=True, exit_status=0, completed_utc=utc(),
                online_fp16_matches_cache=all(r['comparison']['messages']['bitwise_exact'] for r in value['per_problem']),
                original_inference_equivalence=all(r['original_mixed_comparison']['messages']['bitwise_exact']
                    for r in value['per_problem']) if args.timing_tf32_policy == 'default' else False)
            value['numerical_disclosure'] = (
                'Unchanged locked weights/decision rule. Measured path=' + args.timing_path
                + ', TF32 policy=' + args.timing_tf32_policy
                + '. Bitwise FP16 equality on first128 rows/task=' + str(value['online_fp16_matches_cache'])
                + '; original mixed-online equivalence=' + str(value['original_inference_equivalence'])
                + '. Full TEST efficacy comes from untouched cached predictions, not a demonstrated exact online run.'
                + (' Exact-N partitioning and scatter overhead are included.' if args.timing_path == 'exact_n' else ''))
            verify_origin(origin, args.root, args.arm)
            save_new(output, value)
            save_new(receipt, dict(**origin, stage='timing', complete=True, exit_status=0, completed_utc=utc(),
                output=dict(path=str(output), sha256=file_hash(output)),
                original_inference_equivalence=value['original_inference_equivalence'],
                online_fp16_matches_cache=value['online_fp16_matches_cache']))
            return value
        except BaseException as error:
            value.update(complete=False, exit_status=1, error=str(error), failed_utc=utc())
            save_new(output, value)
            save_new(receipt, dict(**origin, stage='timing', complete=False, exit_status=1,
                error=str(error), failed_utc=utc(), automatic_retry=False))
            raise


def verify_origin(origin, root, arm):
    _, _, current = provenance(root, arm)
    for key in ('checkpoint_sha256', 'state_hash', 'lock_sha256', 'original_failure_sha256'):
        if current[key] != origin[key]:
            raise ValueError('Original identity changed during recovery: ' + key)
    for name, item in origin['original_frozen'].items():
        if file_hash(Path(item['path'])) != item['sha256']:
            if name != 'final_metrics.json' or not authorized_val_binding(root, arm, item['sha256']):
                raise ValueError('Frozen original changed during recovery: ' + name)
    return current


def validate_timing_record(record, lock, *, path=None):
    """Private revision: exact flags must match evidence, never an allclose override."""
    required = dict(experiment='R99', recovery_schema=SCHEMA, complete=True, exit_status=0,
        repetitions=3, warmup_batches=1, includes_input_transform=True,
        raw_input_source='original_base_cache_without_relation_fields', online_transform_verified=True,
        full_system_measured=False, end_to_end_measured=False, model_reselection=False,
        efficacy_reused_as_exact_online=False, labels_opened=False,
        checkpoint_sha256=lock['checkpoint_sha256'], state_hash=lock['state_hash'],
        inference_rule=lock['inference_rule'])
    arm = record.get('arm')
    if (arm != lock['arm'] or record.get('variant') != experiment.VARIANTS[arm]
            or any(record.get(k) != v for k, v in required.items())
            or record.get('timing_path') not in PATHS or not record.get('scope')
            or not record.get('excludes') or not record.get('numerical_disclosure')):
        raise ValueError('Incomplete or foreign recovery timing revision')
    items = record.get('per_problem', [])
    if (len(items) != len(experiment.PROBLEMS)
            or {r.get('problem') for r in items} != set(experiment.PROBLEMS)):
        raise ValueError('Timing must cover each of all18 problems exactly once')

    def positive(value, zero=False):
        return (not isinstance(value, bool) and isinstance(value, (float, int))
                and math.isfinite(value) and (value >= 0 if zero else value > 0))

    rows, exact, original = [], [], []
    for item in items:
        if item.get('instances') != 128 or not positive(item.get('max_marginal_residual'), zero=True):
            raise ValueError('Require first128 inputs and a finite marginal residual')
        row = dict(problem=item['problem'], instances=128, device=record['hardware']['device_name'],
            repetitions=3, max_marginal_residual=item['max_marginal_residual'])
        for scope in experiment.TIMING_SCOPES:
            value = item.get(scope, {})
            samples, mean = value.get('batch_seconds'), value.get('mean_seconds_per_instance')
            if (not isinstance(samples, list) or len(samples) != 3 or not all(positive(x) for x in samples)
                    or not positive(mean) or not math.isclose(mean, sum(samples)/(3*128), rel_tol=1e-6, abs_tol=1e-12)):
                raise ValueError('Require three actual synchronized measurements: ' + scope)
            row[scope + '_ms_per_instance'] = 1000*mean
        for key in ('comparison', 'original_mixed_comparison'):
            comparison = item.get(key, {})
            messages = comparison.get('messages', {})
            bitwise = messages.get('bitwise_exact')
            if (type(bitwise) is not bool or comparison.get('masks_exact') is not True
                    or not isinstance(messages.get('unequal_bytes'), int)
                    or messages['unequal_bytes'] < 0 or messages.get('nonfinite_count') != 0
                    or bitwise != (messages['unequal_bytes'] == 0)
                    or not positive(messages.get('max_abs'), zero=True)
                    or comparison.get('scores', {}).get('nonfinite_count') != 0
                    or len(comparison.get('online_predictions_global_ids', [])) != 128
                    or len(comparison.get('cached_predictions_global_ids', [])) != 128
                    or comparison.get('prediction_changed_count') != sum(a != b for a, b in zip(
                        comparison['online_predictions_global_ids'], comparison['cached_predictions_global_ids']))):
                raise ValueError('Exact/nonexact timing flags require honest per-row comparison evidence')
        exact.append(item['comparison']['messages']['bitwise_exact'])
        original.append(item['original_mixed_comparison']['messages']['bitwise_exact'])
        rows.append(row)
    original_exact = all(original) and record.get('requested_tf32_policy') == 'default'
    if (type(record.get('online_fp16_matches_cache')) is not bool
            or record['online_fp16_matches_cache'] != all(exact)
            or type(record.get('original_inference_equivalence')) is not bool
            or record['original_inference_equivalence'] != original_exact):
        raise ValueError('Do not relabel approximate/changed-numerics evidence as exact original inference')
    prep, test = record.get('preprocessing', {}), record.get('test_preprocessing', {})
    for value, splits in ((prep, ['train', 'val']), (test, ['test'])):
        if (value.get('complete') is not True or value.get('experiment') != 'R99'
                or value.get('arm') != arm or value.get('variant') != record['variant']
                or value.get('splits') != splits or not positive(value.get('seconds'), zero=True)):
            raise ValueError('Require distinct original TRAIN/VAL and TEST preparation receipts')
    if test.get('checkpoint_sha256') != lock['checkpoint_sha256'] or test.get('state_hash') != lock['state_hash']:
        raise ValueError('TEST preparation belongs to another lock')
    rows.sort(key=lambda row: experiment.PROBLEMS.index(row['problem']))
    return dict(path=str(path) if path else None, sha256=file_hash(path) if path else None,
        status='complete_private_numerical_revision', measurements=rows,
        preprocessing_seconds=prep['seconds'], preprocessing=prep,
        test_preprocessing_seconds=test['seconds'], test_preprocessing=test,
        transform_plus_selector_measured=True, full_system_measured=False, end_to_end_measured=False,
        pdf_panel=True, scope=record['scope'], excludes=record['excludes'],
        note=record['numerical_disclosure'], recovery_schema=SCHEMA,
        online_fp16_matches_cache=all(exact), original_inference_equivalence=original_exact,
        timing_path=record['timing_path'], requested_tf32_policy=record['requested_tf32_policy'],
        efficacy_reused_as_exact_online=False, numerical_disclosure=record['numerical_disclosure'])


def completed_original_lib(root, arm, lock, origin):
    run = Path(root) / arm
    path = Path(root) / 'preparation/manual_stages' / (arm + '_lib.json')
    manual, complete = read_json(path), read_json(run / 'lib_stage_complete.json')
    for key, name in (('original_worker_sha256', 'worker.json'), ('original_controller_sha256', 'controller.json')):
        if manual.get(key) != origin['original_failures'][name]['sha256']:
            raise ValueError('LIB recovery lost original failure linkage')
    if (manual.get('stage') != 'lib' or manual.get('arm') != arm or manual.get('state') != 'complete'
            or manual.get('exit_status') != 0 or manual.get('checkpoint_lock') != lock
            or complete.get('complete') is not True or complete.get('exit_status') != 0
            or complete.get('checkpoint_sha256') != lock['checkpoint_sha256']
            or complete.get('state_hash') != lock['state_hash']
            or set(complete.get('datasets', [])) != {'TSPLIB', 'CVRPLIB'}):
        raise ValueError('Require main-owned rc0 original reporter LIB, never rerun it here')
    return dict(manual_receipt=dict(path=str(path), sha256=file_hash(path)),
        completion=dict(path=str(run / 'lib_stage_complete.json'), sha256=file_hash(run / 'lib_stage_complete.json')),
        checkpoint_sha256=lock['checkpoint_sha256'], datasets=complete['datasets'])


def ensure_link(path, target, root):
    assert_private(path.parent, root)
    if path.is_symlink():
        # A symlink may intentionally resolve to immutable original input files.
        if path.resolve() != Path(target).resolve():
            raise ValueError('Recovery view has a different input link: ' + str(path))
    elif path.exists():
        raise FileExistsError('Preserve unexpected recovery view input: ' + str(path))
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.symlink_to(Path(target).resolve(), target_is_directory=Path(target).is_dir())


def reporting_view(root, arm):
    """Link read-only inputs, not source/feature copies; every report write is private."""
    original = Path(root) / arm
    view_root = recovery_root(root) / 'report_view'
    run = view_root / arm
    assert_private(run, root)
    run.mkdir(parents=True, exist_ok=True)
    for name in ('best.pt', 'protocol.json', 'training_complete.json', 'final_test_lock.json',
                 'test_results.json', 'test_complete.json', 'final_metrics.json', 'best_eval.json',
                 'history.json', 'input_features_ready.json', 'test_input_features_ready.json',
                 'evaluation', 'lib_evaluation', 'lib_stage_complete.json'):
        source = original / name
        if name == 'final_metrics.json':
            binding_path = stage_receipt_path(root, arm, 'bind_val')
            if binding_path.is_file() and read_json(binding_path).get('complete') is True:
                source = Path(read_json(binding_path)['backup']['path'])
                existing = run / name
                if existing.is_symlink() and existing.resolve() == (original / name).resolve():
                    existing.unlink()
        if not source.exists():
            raise FileNotFoundError('Missing original report input: ' + str(source))
        ensure_link(run / name, source, root)
    timing_receipt = read_json(stage_receipt_path(root, arm, 'timing'))
    source = Path(timing_receipt['output']['path'])
    if timing_receipt.get('complete') is not True or file_hash(source) != timing_receipt['output']['sha256']:
        raise ValueError('Require successful immutable private timing receipt')
    ensure_link(run / 'inference_timing.json', source, root)
    return view_root


def private_reporter():
    from . import r99_reporting as frozen
    name = 'code.V4._r99_recovery_reporting_' + str(os.getpid())
    spec = importlib.util.spec_from_file_location(name, frozen.__file__)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    def private_timing(run, lock):
        path = Path(run) / 'inference_timing.json'
        return validate_timing_record(read_json(path), lock, path=path)

    original_panel = module._timing_panel

    def panel(pdf, helper, model, lock, timings, page, *, locks=None):
        original_panel(pdf, helper, model, lock, timings, page, locks=locks)
        import matplotlib.pyplot as plt
        import textwrap
        fig = module.common._page(helper, model + ' | Timing recovery numerical boundary',
            'Private validator revision; approximate messages are NEVER declared bitwise equivalent.', model, lock, page + len(timings))
        for index, (arm, timing) in enumerate(timings.items()):
            fig.text(.05, .83 - .23*index,
                '\n'.join(textwrap.wrap(arm + ': ' + timing['numerical_disclosure'], 115)), fontsize=10)
        fig.text(.05, .20, 'TEST efficacy: original locked cached predictions only. No TRAIN/TEST rerun or model selection.', fontsize=10)
        fig.text(.05, .14, 'LIB: unchanged original reporter, same checkpoint; its native online computation is disclosed separately.', fontsize=10)
        pdf.savefig(fig)
        plt.close(fig)

    module.inference_timing = private_timing
    module._timing_panel = panel
    module.REFERENCE_BOUNDARY += (
        ' RECOVERY REVISION: cached TEST efficacy is not evidence of exact online efficacy. '
        'See private timing schema, explicit FP16/score/prediction differences and numerical-boundary page. '
        'Original failed worker/controller receipts remain failed and unchanged.')
    module.BOUNDARY = module.REFERENCE_BOUNDARY
    return module


def report_arguments(args, module, root, command):
    argv = [command, '--root', str(root), '--device', 'cpu']
    if args.arm:
        argv += ['--arm', args.arm]
    for key in ('nss_root', 'historical_root', 'r89_root'):
        value = getattr(args, key, None)
        if value is not None:
            argv += ['--' + key.replace('_', '-'), str(value)]
    return module.parser().parse_args(argv)


def report(args):
    if args.device != 'cpu':
        raise ValueError('CPU-only recovery reporting requires --device cpu')
    config, lock, origin = provenance(args.root, args.arm)
    lib = completed_original_lib(args.root, args.arm, lock, origin)
    root = reporting_view(args.root, args.arm)
    module = private_reporter()
    timing_info = module.inference_timing(root / args.arm, lock)
    if timing_info['timing_path'] != 'mixed' or timing_info['requested_tf32_policy'] != 'default':
        raise ValueError('Approved recovery keeps the original mixed/default online path')
    if not timing_info['original_inference_equivalence'] and args.timing_revision != 'disclosed-online-v1':
        raise ValueError('Nonexact online timing requires explicit --timing-revision disclosed-online-v1; not old-schema equivalence')
    path = stage_receipt_path(args.root, args.arm, 'report')
    with claimed(path.with_suffix('.lock')), bounded(args.max_seconds):
        if path.exists():
            raise FileExistsError('Preserve previous CPU report attempt; no automatic retry')
        receipt = dict(**origin, stage='report', complete=False, started_utc=utc(),
            private_validator_schema=SCHEMA, original_reporter_sources_modified=False,
            timing_revision=args.timing_revision, timing=timing_info, lib=lib,
            cli=[sys.executable, '-B', '-m', 'code.V4.r99_postprocess_recovery', *sys.argv[1:]],
            report_root=str(root))
        save_new(path, receipt)
        try:
            module.report(report_arguments(args, module, root, 'report'))
            module.completed_arm_report(root / args.arm, lock)
            verify_origin(origin, args.root, args.arm)
            output = root / args.arm / 'full_model_statistics'
            pdfs = {name: dict(path=str(output/name), sha256=file_hash(output/name)) for name in module.REQUIRED_PDFS}
            receipt.update(complete=True, exit_status=0, completed_utc=utc(), pdfs=pdfs,
                report_protocol=dict(path=str(output/'protocol.json'), sha256=file_hash(output/'protocol.json')))
            save_new(path, receipt)
            return receipt
        except BaseException as error:
            receipt.update(complete=False, exit_status=1, failed_utc=utc(), error=str(error), automatic_retry=False)
            save_new(path, receipt)
            raise


def check_report(root, arm):
    path = stage_receipt_path(root, arm, 'report')
    record = read_json(path)
    if record.get('complete') is not True or record.get('exit_status') != 0:
        raise ValueError('Same-lock full CPU report must complete first')
    verify_origin(record, root, arm)
    for item in [*record['pdfs'].values(), record['report_protocol']]:
        if file_hash(Path(item['path'])) != item['sha256']:
            raise ValueError('Completed report artifact changed')
    module = private_reporter()
    view = Path(record['report_root'])
    module.completed_arm_report(view / arm, record['lock'])
    return record


def json_value_span(text, keys, start=0):
    """Find one JSON value without reformatting any other original byte."""
    decoder = json.JSONDecoder()

    def whitespace(index):
        while index < len(text) and text[index].isspace():
            index += 1
        return index

    index = whitespace(start)
    if text[index] != '{':
        raise ValueError('Expected an object on the metadata binding path')
    index = whitespace(index + 1)
    while text[index] != '}':
        key, index = decoder.raw_decode(text, index)
        index = whitespace(index)
        if text[index] != ':':
            raise ValueError('Invalid JSON metadata')
        value_start = whitespace(index + 1)
        _, value_end = decoder.raw_decode(text, value_start)
        if key == keys[0]:
            return json_value_span(text, keys[1:], value_start) if len(keys) > 1 else (value_start, value_end)
        index = whitespace(value_end)
        if text[index] == ',':
            index = whitespace(index + 1)
        elif text[index] != '}':
            raise ValueError('Invalid JSON metadata separator')
    raise ValueError('Missing JSON metadata member: ' + '.'.join(keys))


def bound_val_bytes(original, checkpoint_sha256):
    text = original.decode('utf-8')
    begin, end = json_value_span(text, ('val', 'identity', 'checkpoint_sha256'))
    if text[begin:end] != 'null':
        raise ValueError('Only present-null val.identity.checkpoint_sha256 may be bound')
    return (text[:begin] + json.dumps(checkpoint_sha256) + text[end:]).encode('utf-8')


def authorized_val_binding(root, arm, original_sha256):
    path = stage_receipt_path(root, arm, 'bind_val')
    if not path.is_file():
        return False
    receipt = read_json(path)
    target = Path(root) / arm / 'final_metrics.json'
    backup = Path(receipt.get('backup', {}).get('path', '/nonexistent'))
    lock = read_json(Path(root) / arm / 'final_test_lock.json')
    if (receipt.get('complete') is not True or receipt.get('exit_status') != 0
            or receipt.get('checkpoint_sha256') != lock['checkpoint_sha256']
            or receipt.get('backup', {}).get('sha256') != original_sha256
            or file_hash(backup) != original_sha256
            or file_hash(target) != receipt.get('after_sha256')):
        return False
    original = backup.read_bytes()
    return target.read_bytes() == bound_val_bytes(original, lock['checkpoint_sha256'])


def bind_val(args):
    if args.device != 'cpu':
        raise ValueError('Metadata-only binding requires --device cpu')
    path = stage_receipt_path(args.root, args.arm, 'bind_val')
    with claimed(path.with_suffix('.lock')):
        report_record = check_report(args.root, args.arm)
        original_sha = report_record['original_frozen']['final_metrics.json']['sha256']
        if path.is_file():
            if authorized_val_binding(args.root, args.arm, original_sha):
                return read_json(path)
            raise ValueError('Preserve incomplete/conflicting metadata binding attempt')
        run = args.root / args.arm
        lock = report_record['lock']
        target = run / 'final_metrics.json'
        original = target.read_bytes()
        if file_hash(target) != original_sha or target.is_symlink():
            raise ValueError('Metadata changed before explicit binding')
        final = json.loads(original)
        archived_path = run / 'evaluation/val/selected_best/identity.json'
        archived = read_json(archived_path)
        val_identity = final['val']['identity']
        test_identity = read_json(run / 'test_results.json')['result']['identity']
        if (val_identity.get('checkpoint_sha256', 'missing') is not None
                or archived != val_identity
                or not val_identity.get('model_state_sha256')
                or val_identity['model_state_sha256'] != test_identity.get('model_state_sha256')
                or test_identity.get('checkpoint_sha256') != lock['checkpoint_sha256']
                or any(identity.get('state_hash') != lock['state_hash']
                       or identity.get('successful_updates') != lock['selected_update']
                       for identity in (val_identity, test_identity))):
            raise ValueError('Require original present-null VAL and same update/state/modelState as locked TEST')
        backup = recovery_root(args.root) / args.arm / 'final_metrics.before_val_sha_binding.json'
        with backup.open('xb') as handle:
            handle.write(original)
        updated = bound_val_bytes(original, lock['checkpoint_sha256'])
        proof = dict(schema=SCHEMA, experiment='R99', arm=args.arm, stage='bind_val', complete=False,
            checkpoint_sha256=lock['checkpoint_sha256'], state_hash=lock['state_hash'],
            selected_update=lock['selected_update'], model_state_sha256=val_identity['model_state_sha256'],
            original_failure_sha256=report_record['original_failure_sha256'],
            full_report_receipt=dict(path=str(stage_receipt_path(args.root, args.arm, 'report')),
                sha256=file_hash(stage_receipt_path(args.root, args.arm, 'report'))),
            target=str(target), changed_field='val.identity.checkpoint_sha256', old_value=None,
            new_value=lock['checkpoint_sha256'], metadata_only=True,
            backup=dict(path=str(backup), sha256=file_hash(backup)),
            original_archived_val_identity=dict(path=str(archived_path), sha256=file_hash(archived_path)),
            numeric_values_changed=False, prediction_arrays_changed=False, weights_changed=False,
            original_failure_receipts_changed=False, started_utc=utc())
        save_new(path, proof)
        temporary = backup.with_name('.final_metrics.bound.' + str(os.getpid()))
        with temporary.open('xb') as handle:
            handle.write(updated)
        temporary.chmod(target.stat().st_mode & 0o777)
        temporary.replace(target)
        proof.update(complete=True, exit_status=0, completed_utc=utc(), after_sha256=file_hash(target))
        save_new(path, proof)
        if not authorized_val_binding(args.root, args.arm, original_sha):
            raise ValueError('Metadata-only byte-span verification failed')
        reporting_view(args.root, args.arm)
        verify_origin(report_record, args.root, args.arm)
        return proof


def compare(args):
    if args.device != 'cpu':
        raise ValueError('Comparison recovery is CPU-only')
    records = {arm: check_report(args.root, arm) for arm in experiment.ARMS}
    module = private_reporter()
    root = recovery_root(args.root) / 'report_view'
    path = recovery_root(args.root) / 'comparison_stage.json'
    with claimed(path.with_suffix('.lock')), bounded(args.max_seconds):
        if path.exists():
            raise FileExistsError('Preserve previous comparison attempt')
        receipt = dict(schema=SCHEMA, experiment='R99', stage='compare', complete=False, started_utc=utc(),
            checkpoint_locks={arm: r['lock'] for arm, r in records.items()},
            original_failure_sha256={arm: r['original_failure_sha256'] for arm, r in records.items()})
        save_new(path, receipt)
        try:
            module.compare(report_arguments(args, module, root, 'compare'))
            for arm, record in records.items():
                verify_origin(record, args.root, arm)
            pdf = root / 'RESULTS.pdf'
            receipt.update(complete=True, exit_status=0, completed_utc=utc(),
                comparison_pdf=dict(path=str(pdf), sha256=file_hash(pdf)),
                protocol=dict(path=str(root/'comparison_artifacts/protocol.json'),
                    sha256=file_hash(root/'comparison_artifacts/protocol.json')))
            save_new(path, receipt)
            return receipt
        except BaseException as error:
            receipt.update(complete=False, exit_status=1, error=str(error), automatic_retry=False)
            save_new(path, receipt)
            raise


def checked_artifact(item):
    if file_hash(Path(item['path'])) != item['sha256']:
        raise ValueError('Recovery artifact changed: ' + item['path'])
    return Path(item['path'])


def validate_completed_arm(root, arm):
    """Read-only, CPU-only evidence gate usable directly by the next launcher."""
    path = recovery_root(root) / arm / 'recovery_complete.json'
    value = read_json(path)
    record = check_report(root, arm)
    lock = record['lock']
    expected = dict(schema=SCHEMA, experiment='R99', arm=arm, complete=True, exit_status=0,
        state='complete', same_checkpoint_verified=True, cli_module='code.V4.r99_postprocess_recovery',
        checkpoint_lock=lock, checkpoint_sha256=lock['checkpoint_sha256'], state_hash=lock['state_hash'],
        original_failure_sha256=record['original_failure_sha256'],
        original_failure_receipts_preserved=True, training_rerun=False, full_test_rerun=False,
        feature_copy=False, feature_rebuild=False, decision_rule_changed=False,
        efficacy_reused_as_exact_online=False, pdfs=record['pdfs'])
    if any(value.get(key) != expected_value for key, expected_value in expected.items()):
        raise ValueError('Incomplete or foreign same-lock arm recovery gate: ' + arm)
    if value.get('stages_completed') != [
            'original_locked_full_TEST', 'recovery_timing', 'original_reporter_LIB',
            'private_CPU_full_report', 'metadata_only_VAL_checkpoint_binding']:
        raise ValueError('Arm recovery stages are incomplete')
    expected_report = stage_receipt_path(root, arm, 'report')
    if checked_artifact(value['report_receipt']).resolve() != expected_report.resolve():
        raise ValueError('Recovery gate points to a different report receipt')
    original_lib = completed_original_lib(root, arm, lock, record)
    if value.get('lib') != original_lib:
        raise ValueError('Same-lock original LIB receipt changed')
    stage = read_json(stage_receipt_path(root, arm, 'timing'))
    if stage.get('complete') is not True or stage.get('exit_status') != 0:
        raise ValueError('Timing stage is incomplete')
    timing_file = checked_artifact(stage['output'])
    timing_info = validate_timing_record(read_json(timing_file), lock, path=timing_file)
    if (value.get('timing') != timing_info or timing_info['timing_path'] != 'mixed'
            or timing_info['requested_tf32_policy'] != 'default'
            or value.get('original_inference_equivalence') != timing_info['original_inference_equivalence']
            or (not timing_info['original_inference_equivalence']
                and record.get('timing_revision') != 'disclosed-online-v1')):
        raise ValueError('Recovery gate must retain the approved honest mixed/default timing revision')
    if checked_artifact(value['val_binding']).resolve() != stage_receipt_path(root, arm, 'bind_val').resolve():
        raise ValueError('Recovery gate points to a different VAL binding')
    if (read_json(Path(root)/arm/'final_metrics.json')['val']['identity'].get('checkpoint_sha256')
            != lock['checkpoint_sha256'] or not authorized_val_binding(root, arm,
                record['original_frozen']['final_metrics.json']['sha256'])):
        raise ValueError('Report-backed metadata-only VAL binding is no longer valid')
    binding = read_json(value['val_binding']['path'])
    checked_artifact(binding['original_archived_val_identity'])
    checked_artifact(binding['full_report_receipt'])
    return value


def validate_recovery_complete(root):
    """R104 contract: recursively validate evidence, not old failed dispatch status."""
    path = recovery_root(root) / 'recovery_complete.json'
    value = read_json(path)
    expected = dict(schema=SCHEMA, experiment='R99', complete=True, exit_status=0, state='complete',
        same_checkpoint_verified=True, cli_module='code.V4.r99_postprocess_recovery',
        original_failure_receipts_preserved=True, training_rerun=False, full_test_rerun=False,
        feature_copy=False, feature_rebuild=False, decision_rule_changed=False,
        efficacy_reused_as_exact_online=False)
    if (any(value.get(k) != v for k, v in expected.items())
            or set(value.get('arms', {})) != set(experiment.ARMS)
            or value.get('stages_completed') != [
                'A_timing_LIB_PDF_VAL_binding', 'B_timing_LIB_PDF_VAL_binding', 'CPU_AB_comparison']):
        raise ValueError('R99 global recovery pipeline is incomplete')
    arms = {arm: validate_completed_arm(root, arm) for arm in experiment.ARMS}
    for arm, child in arms.items():
        item = value['arms'][arm]
        expected_path = recovery_root(root) / arm / 'recovery_complete.json'
        if (checked_artifact(item).resolve() != expected_path.resolve()
                or item.get('checkpoint_lock') != child['checkpoint_lock']
                or item.get('original_failure_sha256') != child['original_failure_sha256']):
            raise ValueError('R99 global recovery links a foreign arm')
    locks = {arm: r['checkpoint_lock'] for arm, r in arms.items()}
    failures = {arm: r['original_failure_sha256'] for arm, r in arms.items()}
    if value.get('checkpoint_locks') != locks or value.get('original_failure_sha256') != failures:
        raise ValueError('R99 global same-lock/original-failure identities differ')
    if value.get('original_inference_equivalence') != all(r['original_inference_equivalence'] for r in arms.values()):
        raise ValueError('R99 global numerical-equivalence disclosure differs')
    comparison_path = checked_artifact(value['comparison'])
    if comparison_path.resolve() != (recovery_root(root)/'comparison_stage.json').resolve():
        raise ValueError('Foreign comparison receipt')
    comparison = read_json(comparison_path)
    if (comparison.get('schema') != SCHEMA or comparison.get('complete') is not True
            or comparison.get('exit_status') != 0 or comparison.get('checkpoint_locks') != locks
            or comparison.get('original_failure_sha256') != failures):
        raise ValueError('R99 comparison is incomplete or from different checkpoints/failures')
    checked_artifact(comparison['comparison_pdf'])
    protocol = read_json(checked_artifact(comparison['protocol']))
    if protocol.get('complete') is not True or protocol.get('checkpoints') != locks:
        raise ValueError('R99 comparison protocol differs from locked arms')
    return value


def finalize(args):
    if args.device != 'cpu':
        raise ValueError('Completion gate is CPU-only')
    if args.arm is not None:
        if args.bind_val_sha:
            bind_val(args)
        report_record = check_report(args.root, args.arm)
        _, lock, _ = provenance(args.root, args.arm)
        lib = completed_original_lib(args.root, args.arm, lock, report_record)
        timing_path = stage_receipt_path(args.root, args.arm, 'timing')
        timing_record = read_json(timing_path)
        timing_file = Path(timing_record['output']['path'])
        if timing_record.get('complete') is not True or file_hash(timing_file) != timing_record['output']['sha256']:
            raise ValueError('Private timing stage is incomplete or changed')
        timing_info = validate_timing_record(read_json(timing_file), lock, path=timing_file)
        binding = stage_receipt_path(args.root, args.arm, 'bind_val')
        val = read_json(args.root / args.arm / 'final_metrics.json')['val']['identity']
        if (val.get('checkpoint_sha256') != lock['checkpoint_sha256']
                or not binding.is_file()
                or not authorized_val_binding(args.root, args.arm,
                    report_record['original_frozen']['final_metrics.json']['sha256'])):
            raise ValueError('Finish explicit report-backed VAL metadata binding first: finalize --bind-val-sha')
        value = dict(schema=SCHEMA, experiment='R99', arm=args.arm, complete=True, exit_status=0,
            state='complete', completed_utc=utc(), checkpoint_lock=lock, same_checkpoint_verified=True,
            cli_module='code.V4.r99_postprocess_recovery',
            checkpoint_sha256=lock['checkpoint_sha256'], state_hash=lock['state_hash'],
            original_failure_sha256=report_record['original_failure_sha256'],
            original_failures=report_record['original_failures'],
            original_failure_receipts_preserved=True, original_worker_state='failed',
            original_controller_state='failed', training_rerun=False, full_test_rerun=False,
            feature_copy=False, feature_rebuild=False, decision_rule_changed=False,
            stages_completed=['original_locked_full_TEST', 'recovery_timing', 'original_reporter_LIB',
                'private_CPU_full_report', 'metadata_only_VAL_checkpoint_binding'],
            timing=timing_info, lib=lib, pdfs=report_record['pdfs'],
            report_receipt=dict(path=str(stage_receipt_path(args.root, args.arm, 'report')),
                sha256=file_hash(stage_receipt_path(args.root, args.arm, 'report'))),
            val_binding=dict(path=str(binding), sha256=file_hash(binding)),
            original_inference_equivalence=timing_info['original_inference_equivalence'],
            efficacy_reused_as_exact_online=False)
        output = recovery_root(args.root) / args.arm / 'recovery_complete.json'
        if output.exists():
            return validate_completed_arm(args.root, args.arm)
        save_new(output, value)
        return validate_completed_arm(args.root, args.arm)
    if args.bind_val_sha:
        raise ValueError('--bind-val-sha requires --arm')
    arms = {}
    children = {}
    for arm in experiment.ARMS:
        path = recovery_root(args.root) / arm / 'recovery_complete.json'
        value = validate_completed_arm(args.root, arm)
        children[arm] = value
        arms[arm] = dict(path=str(path), sha256=file_hash(path),
            checkpoint_lock=value['checkpoint_lock'], original_failure_sha256=value['original_failure_sha256'])
    comparison_path = recovery_root(args.root) / 'comparison_stage.json'
    comparison = read_json(comparison_path)
    locks = {arm: r['checkpoint_lock'] for arm, r in children.items()}
    failures = {arm: r['original_failure_sha256'] for arm, r in children.items()}
    if (comparison.get('complete') is not True or comparison.get('exit_status') != 0
            or comparison.get('checkpoint_locks') != locks
            or comparison.get('original_failure_sha256') != failures):
        raise ValueError('Both full reports plus rc0 comparison are required')
    for item in (comparison['comparison_pdf'], comparison['protocol']):
        if file_hash(Path(item['path'])) != item['sha256']:
            raise ValueError('Comparison delivery changed')
    comparison_protocol = read_json(comparison['protocol']['path'])
    if comparison_protocol.get('complete') is not True or comparison_protocol.get('checkpoints') != locks:
        raise ValueError('Comparison PDF/protocol must use the two completed recovery locks')
    value = dict(schema=SCHEMA, experiment='R99', complete=True, exit_status=0, state='complete',
        completed_utc=utc(), arms=arms, same_checkpoint_verified=True,
        cli_module='code.V4.r99_postprocess_recovery',
        checkpoint_locks={arm: r['checkpoint_lock'] for arm, r in children.items()},
        original_failure_sha256={arm: r['original_failure_sha256'] for arm, r in children.items()},
        original_inference_equivalence=all(r['original_inference_equivalence'] for r in children.values()),
        stages_completed=['A_timing_LIB_PDF_VAL_binding', 'B_timing_LIB_PDF_VAL_binding', 'CPU_AB_comparison'],
        comparison=dict(path=str(comparison_path), sha256=file_hash(comparison_path)),
        original_failure_receipts_preserved=True, training_rerun=False, full_test_rerun=False,
        feature_rebuild=False, feature_copy=False, decision_rule_changed=False,
        efficacy_reused_as_exact_online=False)
    output = recovery_root(args.root) / 'recovery_complete.json'
    if output.exists():
        return validate_recovery_complete(args.root)
    save_new(output, value)
    return validate_recovery_complete(args.root)


def require_gpu(args):
    if not str(args.device).startswith('cuda:'):
        raise ValueError('GPU commands require a launcher-allocated explicit --device cuda:N')
    if args.expected_uuid:
        visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
        if visible != args.expected_uuid:
            raise ValueError('Launcher must expose exactly --expected-uuid via CUDA_VISIBLE_DEVICES')
    if args.expected_cpus:
        expected = {int(value) for value in args.expected_cpus.split(',')}
        if set(os.sched_getaffinity(0)) != expected:
            raise ValueError('Task CPU affinity differs from --expected-cpus')
    device = torch.device(args.device)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1 or device.index != 0:
        raise ValueError('Expose exactly one launcher-owned GPU as cuda:0')
    props = torch.cuda.get_device_properties(device)
    return device, dict(device=str(device), device_name=props.name,
        visible_uuid=os.environ.get('CUDA_VISIBLE_DEVICES'), cpus=sorted(os.sched_getaffinity(0)),
        torch_version=torch.__version__, cuda_version=torch.version.cuda)


@contextmanager
def bounded(seconds):
    def expired(signum, frame):
        raise TimeoutError('Bounded R99 postprocessing exceeded --max-seconds')
    previous = signal.signal(signal.SIGALRM, expired)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)


@torch.no_grad()
def diagnose(args):
    device, hardware = require_gpu(args)
    output = assert_private(args.output or recovery_root(args.root) / args.arm / 'diagnosis.json', args.root)
    with claimed(output.with_suffix('.lock')), bounded(args.max_seconds):
        if output.exists():
            raise FileExistsError('Preserve existing diagnosis; choose a new private --output')
        config, lock, origin = provenance(args.root, args.arm)
        validate_runtime_paths(args, config)
        selector, state = experiment.load_checkpoint(args.root / args.arm / 'best.pt', device)
        if state['args'] != config or state.get('successful_updates') != lock['selected_update']:
            raise ValueError('Factory returned another checkpoint/configuration')
        del state
        selector.eval().requires_grad_(False)
        result = dict(**origin, command='diagnose', complete=False, started_utc=utc(),
            hardware=hardware, paths=list(PATHS), policies=args.tf32_policies,
            rows_per_problem=128, labels_opened=False, predictions_are_diagnostic_only=True,
            cache_recipe='stable exact-real-N buckets, chunks of 128; this first probe uses only selected rows',
            original_inference_equivalence=None, per_problem=[])
        save_new(output, result)
        try:
            for problem in args.problems:
                reader = InputReader(config, problem, lock)
                try:
                    raw = reader.batch(np.arange(128), device)
                    cached = reader.batch(np.arange(128), device, cached=True)
                    reference = selector(cached)
                    logits = reference['logits'][reference['solver_mask']]
                    if not torch.isfinite(logits).all():
                        raise ValueError('Nonfinite cached-reference diagnostic score')
                    reference_picks = prediction(reference['logits'], raw['pool_ids'])
                    row = dict(problem=problem, instances=128,
                        real_n_counts={str(n): len(indices) for n, indices in real_n_groups(raw)},
                        cached_predictions_global_ids=reference_picks, reference_policy=numerical_policy(),
                        measurements=[])
                    for policy in args.tf32_policies:
                        with tf32_policy(policy) as effective:
                            for path in PATHS:
                                started = time.monotonic()
                                messages, residual = transform_path(raw, experiment.VARIANTS[args.arm], path)
                                if path == 'mixed':
                                    messages = selector.prepare_inputs(raw)['relation_nodes']
                                messages_diff = difference(messages, cached['relation_nodes'])
                                online = dict(raw, relation_nodes=messages, relation_node_mask=raw['node_mask'])
                                scores = selector(raw) if path == 'mixed' else selector(online)
                                if not torch.equal(scores['solver_mask'], reference['solver_mask']):
                                    raise ValueError('Candidate masks differ')
                                score_diff = difference(scores['logits'][scores['solver_mask']], logits)
                                picks = prediction(scores['logits'], raw['pool_ids'])
                                row['measurements'].append(dict(path=path, requested_tf32_policy=policy,
                                    effective_policy=effective, messages=messages_diff, scores=score_diff,
                                    predictions_global_ids=picks,
                                    prediction_changed_count=sum(a != b for a, b in zip(picks, reference_picks)),
                                    max_marginal_residual=float(residual), seconds=time.monotonic()-started,
                                    original_inference_equivalence=messages_diff['bitwise_exact']))
                    result['per_problem'].append(row)
                    save_new(output, result)
                    print(json.dumps(dict(problem=problem, cases=[dict(path=m['path'], policy=m['requested_tf32_policy'],
                        exact=m['messages']['bitwise_exact'], unequal=m['messages']['unequal_elements'],
                        max_abs=m['messages']['max_abs'], changed_picks=m['prediction_changed_count'])
                        for m in row['measurements']])), flush=True)
                finally:
                    reader.close()
            result.update(complete=True, exit_status=0, completed_utc=utc(),
                exact_paths_across_measured_problems=[dict(path=path, tf32_policy=policy) for path in PATHS
                    for policy in args.tf32_policies if all(any(m['path'] == path
                        and m['requested_tf32_policy'] == policy and m['messages']['bitwise_exact']
                        for m in row['measurements']) for row in result['per_problem'])],
                original_inference_equivalence=all(any(m['path'] == 'mixed'
                    and m['requested_tf32_policy'] == 'default' and m['messages']['bitwise_exact']
                    for m in row['measurements']) for row in result['per_problem']),
                equivalence_scope='Only these 128 input rows per measured task, not full TEST or LIB')
            save_new(output, result)
            return result
        except BaseException as error:
            result.update(complete=False, exit_status=1, error=str(error), failed_utc=utc())
            save_new(output, result)
            raise


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('command', choices=('diagnose', 'timing', 'report', 'compare', 'bind-val', 'finalize', 'validate'))
    result.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    result.add_argument('--arm', choices=experiment.ARMS)
    result.add_argument('--device', default='cuda:0')
    result.add_argument('--output', type=Path)
    result.add_argument('--problems', choices=experiment.PROBLEMS, nargs='+', default=list(experiment.PROBLEMS))
    result.add_argument('--tf32-policies', choices=('default', 'off', 'on'), nargs='+', default=['default', 'off', 'on'])
    result.add_argument('--max-seconds', type=int, default=300)
    result.add_argument('--expected-uuid')
    result.add_argument('--expected-cpus')
    result.add_argument('--cache-root', type=Path)
    result.add_argument('--feature-root', type=Path)
    result.add_argument('--timing-path', choices=PATHS, default='mixed')
    result.add_argument('--timing-tf32-policy', choices=('default', 'off', 'on'), default='default')
    result.add_argument('--timing-revision', choices=('disclosed-online-v1',))
    result.add_argument('--nss-root', type=Path)
    result.add_argument('--historical-root', type=Path)
    result.add_argument('--r89-root', type=Path)
    result.add_argument('--bind-val-sha', action='store_true')
    return result


def validate_runtime_paths(args, config):
    for argument, key in (('cache_root', 'cache_root'), ('feature_root', 'feature_cache_root')):
        value = getattr(args, argument, None)
        if value is not None and value.expanduser().resolve() != Path(config[key]).resolve():
            raise ValueError('Recovery cannot relocate or change original ' + argument)


def main(argv=None):
    args = parser().parse_args(argv)
    args.root = args.root.expanduser().resolve()
    if args.command not in ('compare', 'finalize', 'validate') and args.arm is None:
        raise ValueError('--arm is required for this stage')
    if args.command == 'compare' and args.arm is not None:
        raise ValueError('Comparison always uses both locked arms; omit --arm')
    if args.max_seconds < 1 or args.max_seconds > 1800:
        raise ValueError('--max-seconds must be in [1, 1800]')
    if len(set(args.problems)) != len(args.problems) or len(set(args.tf32_policies)) != len(args.tf32_policies):
        raise ValueError('Duplicate tasks or policies are not allowed')
    torch.set_num_threads(2)
    if args.command == 'validate':
        if args.device != 'cpu':
            raise ValueError('Completion validation requires --device cpu')
        result = validate_completed_arm(args.root, args.arm) if args.arm else validate_recovery_complete(args.root)
        print(json.dumps(dict(command='validate', complete=True, exit_status=0, schema=SCHEMA)), flush=True)
        return result
    commands = {'diagnose': diagnose, 'timing': timing, 'report': report,
                'compare': compare, 'bind-val': bind_val, 'finalize': finalize}
    result = commands[args.command](args)
    print(json.dumps(dict(command=args.command, arm=args.arm, complete=result.get('complete'),
        exit_status=result.get('exit_status'), schema=SCHEMA)), flush=True)
    return result


if __name__ == '__main__':
    main()
