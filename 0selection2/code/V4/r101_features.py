"""R101 TRAIN-only, input-only hints from already completed R94 node caches.

No ScenarioCache, donor construction, extraction, source labels, or weights are
opened here. R94's path/mtime-sensitive validator is deliberately unchanged.
R101 authenticates its original donor receipt and immutable shared R84 metadata,
checks input headers and the full small n vector in original row order, and
records transport stat differences separately from content identity.

teacher_root accepts either the sidecar root or its train directory. Explicit
stage_teacher copies only metadata.json and nco_nodes.npy for all 16 routing
tasks. A None target_root records read-only reuse (B); it never copies implicitly.
"""

import argparse
import copy
import hashlib
import json
import mmap
import os
import shutil
import socket
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from .r94_donor import OUTPUT_DIM, PROBLEM_NAMES, SUPPORTED_PROBLEMS
from .r94_features import EXTRACT_ORDER, FEATURE_FILE, FEATURE_FORMAT, _digest, _indices, _source_record


RUNS_ROOT = Path(__file__).resolve().parent / 'runs'
R84_CACHE_ROOT = RUNS_ROOT / 'R84_new90k_attention/input_cache'
R94_ROOT = RUNS_ROOT / 'R94_pretrained_node_fusion'
TRAIN_ROWS = 81000
STAGE_FORMAT = 'r101_existing_teacher_train_v1'
MAX_BUFFER_BYTES = 8 * 1024 * 1024
INPUT_KEYS = ('n', 'node', 'node_geom', 'coord_dist')
TRANSPORT_POLICY = (
    'Path and mtime are transport observations, not content hashes. Identity is '
    'original R94 provenance, immutable original R84 metadata, input schema and '
    'full TRAIN origin, plus actual n in original row order. R94 validation is unchanged.')


def _json(path):
    return json.loads(Path(path).read_text())


def _variant(value):
    variants = {'A': ('A', 'random'), 'B': ('B', 'pretrained'),
                'random': ('A', 'random'), 'pretrained': ('B', 'pretrained')}
    if value not in variants:
        raise ValueError('variant must be A/random or B/pretrained')
    return variants[value]


def _train_root(root):
    root = Path(root).expanduser().resolve()
    if any(part.lower() in ('val', 'validation', 'test', 'lib', 'tsplib', 'cvrplib')
           for part in root.parts):
        raise ValueError('R101 teacher access is TRAIN only')
    return root if root.name == 'train' else root / 'train'


def _stat(path):
    stat = Path(path).stat()
    return dict(bytes=stat.st_size, mtime_ns=stat.st_mtime_ns)


def _signature(path):
    stat = Path(path).stat()
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns


def _write_new_json(path, record):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(record, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())


def _original_donor(variant):
    arm, name = _variant(variant)
    receipt = _json(R94_ROOT / arm / 'input_features_ready.json')
    donor = receipt.get('frozen_donor', {})
    if (receipt.get('experiment') != 'R94' or receipt.get('arm') != arm
            or receipt.get('complete') is not True or 'train' not in receipt.get('splits', [])
            or receipt.get('labels_used') is not False or receipt.get('decoder_executed') is not False
            or donor.get('variant') != name or donor.get('output_dim') != OUTPUT_DIM
            or donor.get('supported_problems') != list(SUPPORTED_PROBLEMS)):
        raise ValueError('Original frozen R94 teacher provenance receipt is invalid')
    return donor


def _load_receipt(receipt):
    if receipt is None:
        return None
    record = _json(receipt) if isinstance(receipt, (str, Path)) else copy.deepcopy(receipt)
    if (not isinstance(record, dict) or record.get('format') != STAGE_FORMAT
            or record.get('complete') is not True or record.get('split') != 'train'
            or record.get('problems') != list(SUPPORTED_PROBLEMS)
            or set(record.get('tasks', {})) != set(SUPPORTED_PROBLEMS)):
        raise ValueError('Invalid or incomplete TRAIN teacher source receipt')
    return record


def _open_array(path, shape, dtype):
    array = np.load(path, mmap_mode='r', allow_pickle=False)
    if (array.shape != tuple(shape) or array.dtype != np.dtype(dtype)
            or not array.flags.c_contiguous):
        array._mmap.close()
        raise ValueError(f'{path}: input/teacher array shape, dtype or row layout differs')
    array._mmap.madvise(mmap.MADV_RANDOM)
    return array


def _contract(cache_root, teacher_root, problem, variant, source_receipt=None):
    if problem not in SUPPORTED_PROBLEMS:
        raise ValueError('Dense hints cover only the 16 routing tasks, not TSP/ATSP')
    arm, variant = _variant(variant)
    directory, record = _source_record(cache_root, 'train', problem)
    original_dir, original = _source_record(R84_CACHE_ROOT, 'train', problem)
    immutable_sha = _digest(original)
    if record['rows'] != TRAIN_ROWS or _digest(record) != immutable_sha:
        raise ValueError(f'{problem}: immutable original full-TRAIN R84 metadata hash differs')
    rows, maximum, width = record['rows'], record['true_length_max'], record['d_in']
    expected = dict(n=dict(file='n.npy', shape=[rows], dtype='int64'),
                    node=dict(file='node.npy', shape=[rows, maximum, width], dtype='float32'),
                    node_geom=dict(file='node_geom.npy', shape=[rows, maximum, 12], dtype='float32'),
                    coord_dist=dict(file='coord_dist.npy', shape=[rows], dtype='int64'))
    if any(record.get('arrays', {}).get(key) != spec for key, spec in expected.items()):
        raise ValueError(f'{problem}: original R84 input schema differs')
    pool, ids, mask = record.get('pool'), record.get('pool_ids'), record.get('mask')
    if (not pool or len(set(pool)) != len(pool) or not isinstance(ids, list)
            or len(ids) != len(pool) or ids != sorted(set(ids))
            or not isinstance(mask, list) or any(not 0 <= item < len(mask) for item in ids)
            or mask != [int(index in ids) for index in range(len(mask))]
            or not record.get('cbits') or not record.get('problem_desc')):
        raise ValueError(f'{problem}: frozen legal pool/condition input metadata differs')
    feature_dir = _train_root(teacher_root) / problem
    metadata = _json(feature_dir / 'metadata.json')
    source = metadata.get('source_cache', {})
    donor = _original_donor(variant)
    if (metadata.get('format') != FEATURE_FORMAT or metadata.get('complete') is not True
            or metadata.get('rows') != rows or metadata.get('rows_written') != rows
            or metadata.get('shape') != [rows, maximum, OUTPUT_DIM]
            or metadata.get('file') != FEATURE_FILE or metadata.get('dtype') != 'float16'
            or metadata.get('problem') != problem or metadata.get('split') != 'train'
            or metadata.get('input_only') is not True or metadata.get('test_lock') is not None
            or metadata.get('extract_order') != EXTRACT_ORDER or metadata.get('donor') != donor
            or source.get('metadata_sha256') != immutable_sha):
        raise ValueError(f'{problem}: original R94 variant/provenance or metadata hash differs')
    origin = Path(source.get('directory', ''))
    if not origin.is_absolute() or origin.parts[-2:] != ('train', problem):
        raise ValueError(f'{problem}: teacher source must be original full TRAIN in original row order')
    observations = {}
    for key in ('n', 'node'):
        saved = source.get('inputs', {}).get(key, {})
        path = directory / expected[key]['file']
        actual = _stat(path)
        if (saved.get('spec') != expected[key] or saved.get('bytes') != actual['bytes']
                or saved.get('path') != str(origin / expected[key]['file'])
                or not isinstance(saved.get('mtime_ns'), int)):
            raise ValueError(f'{problem}: teacher source input schema/origin differs')
        observations[key] = dict(original=saved, runtime=dict(path=str(path), **actual),
                                 mtime_differs=saved['mtime_ns'] != actual['mtime_ns'])
    receipt = _load_receipt(source_receipt)
    if receipt is not None:
        entry = receipt['tasks'][problem]
        if (receipt.get('arm') != arm or receipt.get('variant') != variant
                or receipt.get('root') != str(_train_root(teacher_root))
                or entry.get('teacher_metadata_sha256') != _digest(metadata)
                or entry.get('r84_metadata_sha256') != immutable_sha
                or entry.get('donor_sha256') != _digest(donor)):
            raise ValueError(f'{problem}: source receipt metadata hash/variant differs')
        for filename in ('metadata.json', FEATURE_FILE):
            if entry.get('files', {}).get(filename, {}).get('target_stat') != _stat(feature_dir / filename):
                raise ValueError(f'{problem}: staged teacher changed since source receipt')
    return directory, record, feature_dir, metadata, dict(
        arm=arm, variant=variant, r84_metadata_sha256=immutable_sha,
        teacher_metadata_sha256=_digest(metadata), donor_sha256=_digest(donor),
        origin=str(origin), transport_policy=TRANSPORT_POLICY, transport_observations=observations,
        canonical_directory=str(original_dir))


class HintInputCache:
    """Input-only original R84 TRAIN rows plus frozen FP16 teacher nodes.

    batch(rows, device='cpu') preserves duplicates and caller row order. Returned
    teacher_node_mask is the exact same tensor as node_mask. All arrays are
    read-only mmaps, and close releases mappings and prefetch descriptors.
    """

    def __init__(self, cache_root, teacher_root, problem, variant, source_receipt=None):
        self._arrays, self._fds, self._closed = {}, {}, False
        self.lengths = None
        (self.directory, self.metadata, self.teacher_directory,
         self.teacher_metadata, self.identity) = _contract(
            cache_root, teacher_root, problem, variant, source_receipt)
        self.problem, self.split, self.kind = problem, 'train', 'coord'
        self.size = self.metadata['rows']
        self.pool, self.pool_ids = list(self.metadata['pool']), list(self.metadata['pool_ids'])
        try:
            for key in (*INPUT_KEYS, 'teacher_nodes'):
                if key == 'teacher_nodes':
                    path = self.teacher_directory / FEATURE_FILE
                    shape, dtype = self.teacher_metadata['shape'], 'float16'
                else:
                    spec = self.metadata['arrays'][key]
                    path, shape, dtype = self.directory / spec['file'], spec['shape'], spec['dtype']
                self._arrays[key] = _open_array(path, shape, dtype)
                self._fds[key] = os.open(path, os.O_RDONLY)
                os.posix_fadvise(self._fds[key], 0, 0, os.POSIX_FADV_RANDOM)
            self.lengths = self._arrays['n']
            original_path = Path(self.identity['canonical_directory']) / 'n.npy'
            original = _open_array(original_path, [self.size], 'int64')
            try:
                if (int(self.lengths.min()) != self.metadata['true_length_min']
                        or int(self.lengths.max()) != self.metadata['true_length_max']
                        or not np.array_equal(self.lengths, original)):
                    raise ValueError(f'{problem}: actual n/original TRAIN row alignment differs')
            finally:
                original._mmap.close()
            self.identity['actual_n_sha256'] = hashlib.sha256(self.lengths.tobytes()).hexdigest()
            self.identity['row_alignment'] = 'original R84 rows; exact full n vector; R94 writes original row slots'
        except BaseException:
            self.close()
            raise

    def _gather(self, key, rows, maximum=None):
        array, fd = self._arrays[key], self._fds[key]
        view = array if maximum is None else array[:, :maximum]
        wanted = array.dtype.itemsize if array.ndim == 1 else maximum * array.strides[1]
        for row in np.unique(rows):
            os.posix_fadvise(fd, int(array.offset) + int(row) * int(array.strides[0]),
                             int(wanted), os.POSIX_FADV_WILLNEED)
        return torch.from_numpy(np.array(view[rows], copy=True, order='C'))

    def batch(self, rows, device='cpu'):
        if self._closed:
            raise RuntimeError('HintInputCache is closed')
        rows = _indices(rows, self.size)
        n, record, count = self._gather('n', rows), self.metadata, len(rows)
        maximum = int(n.max())
        mask = torch.arange(maximum)[None] < n[:, None]
        result = dict(problem_id=record['problem_id'], problem_name=self.problem,
                      kind=self.kind, d_in=record['d_in'], n=n, node_mask=mask,
                      cbits=torch.tensor(record['cbits'], dtype=torch.float32).repeat(count, 1),
                      problem_desc=torch.tensor(record['problem_desc'], dtype=torch.float32).repeat(count, 1),
                      coord_dist=self._gather('coord_dist', rows),
                      pool_ids=torch.tensor(self.pool_ids, dtype=torch.long),
                      mask=torch.tensor(record['mask'], dtype=torch.float32).repeat(count, 1),
                      solver_mask=torch.ones(count, len(self.pool), dtype=torch.bool))
        for key in ('node', 'node_geom', 'teacher_nodes'):
            result[key] = self._gather(key, rows, maximum)
            result[key].masked_fill_(~mask[..., None], 0)
        result = {key: value.to(device) if torch.is_tensor(value) else value
                  for key, value in result.items()}
        result['teacher_node_mask'] = result['node_mask']
        return result

    def close(self):
        if not self._closed:
            for fd in self._fds.values():
                os.close(fd)
            for array in self._arrays.values():
                array._mmap.close()
            self._arrays, self._fds, self.lengths = {}, {}, None
            self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def inspect_teacher(teacher_root, variant, *, cache_root=R84_CACHE_ROOT, source_receipt=None,
                    target_root=None):
    """Read small JSON, array headers and actual n only, never teacher payloads."""
    started = time.monotonic()
    tasks, total = {}, 0
    for problem in SUPPORTED_PROBLEMS:
        with HintInputCache(cache_root, teacher_root, problem, variant, source_receipt) as cache:
            files = {name: dict(source_stat=_stat(cache.teacher_directory / name))
                     for name in ('metadata.json', FEATURE_FILE)}
            tasks[problem] = dict(cache.identity, rows=cache.size,
                                  shape=cache.teacher_metadata['shape'], files=files)
            total += sum(entry['source_stat']['bytes'] for entry in files.values())
    arm, name = _variant(variant)
    result = dict(format=STAGE_FORMAT, complete=True, split='train', arm=arm, variant=name,
                root=str(_train_root(teacher_root)), problems=list(SUPPORTED_PROBLEMS), tasks=tasks,
                bytes=total, gib=total / 1024 ** 3, inspection_seconds=time.monotonic() - started,
                labels_read=False, encoder_executed=False, teacher_payload_read=False,
                teacher_provenance=_original_donor(variant), transport_policy=TRANSPORT_POLICY)
    result['source_disk_available_bytes'] = shutil.disk_usage(_train_root(teacher_root)).free
    if target_root is not None:
        target = _train_root(target_root)
        ancestor = target.parent
        while not ancestor.exists():
            ancestor = ancestor.parent
        result['destination'] = dict(root=str(target), exists=target.exists(),
                                     disk_available_bytes=shutil.disk_usage(ancestor).free)
    result['unmeasured_time_projection'] = dict(
        note='Illustrations only, not measured throughput; include one source read, write and target read',
        seconds_at_combined_io_mib_s={str(rate): 3 * total / (rate * 1024 ** 2) for rate in (50, 100, 200)})
    return result


def inspect_teacher_metadata(teacher_root, variant):
    """Small discovery receipt only: no array open, stat, hash, or copy.

    This is not a full stage receipt and is not accepted as source_receipt.
    The authorized worker still checks actual input n and headers on cache open.
    """
    started = time.monotonic()
    arm, variant = _variant(variant)
    donor, tasks = _original_donor(variant), {}
    root = _train_root(teacher_root)
    for problem in SUPPORTED_PROBLEMS:
        _, raw = _source_record(R84_CACHE_ROOT, 'train', problem)
        metadata = _json(root / problem / 'metadata.json')
        source = metadata.get('source_cache', {})
        shape = [TRAIN_ROWS, raw['true_length_max'], OUTPUT_DIM]
        if (raw['rows'] != TRAIN_ROWS or metadata.get('format') != FEATURE_FORMAT
                or metadata.get('problem') != problem or metadata.get('split') != 'train'
                or metadata.get('complete') is not True or metadata.get('rows') != TRAIN_ROWS
                or metadata.get('rows_written') != TRAIN_ROWS or metadata.get('shape') != shape
                or metadata.get('file') != FEATURE_FILE or metadata.get('dtype') != 'float16'
                or metadata.get('input_only') is not True or metadata.get('test_lock') is not None
                or metadata.get('extract_order') != EXTRACT_ORDER or metadata.get('donor') != donor
                or source.get('metadata_sha256') != _digest(raw)):
            raise ValueError(f'{problem}: original TRAIN teacher metadata/provenance differs')
        origin = Path(source.get('directory', ''))
        if not origin.is_absolute() or origin.parts[-2:] != ('train', problem):
            raise ValueError(f'{problem}: teacher metadata source is not original TRAIN')
        for key in ('n', 'node'):
            saved = source.get('inputs', {}).get(key, {})
            if (saved.get('spec') != raw['arrays'][key]
                    or saved.get('path') != str(origin / raw['arrays'][key]['file'])):
                raise ValueError(f'{problem}: teacher metadata input schema/origin differs')
        tasks[problem] = dict(teacher_metadata_sha256=_digest(metadata),
                              r84_metadata_sha256=_digest(raw), donor_sha256=_digest(donor),
                              rows=TRAIN_ROWS, shape=shape, origin=str(origin),
                              pool=raw['pool'], pool_ids=raw['pool_ids'])
    return dict(format='r101_teacher_metadata_only_v1', complete=True,
                validation_scope='metadata_only_not_array_or_row_validation',
                root=str(root), teacher_root=str(root.parent), arm=arm, variant=variant,
                split='train', problems=list(SUPPORTED_PROBLEMS), tasks=tasks,
                teacher_provenance=donor, seconds=time.monotonic() - started,
                created_utc=datetime.now(timezone.utc).isoformat(), host=socket.gethostname(),
                arrays_opened=False, array_hashes_computed=False, copied=False,
                labels_read=False, encoder_executed=False, weights_opened=False,
                worker_checks_required='HintInputCache validates actual n, headers and immutable input identity')


def _hash_file(path, buffer_bytes):
    digest = hashlib.sha256()
    with Path(path).open('rb', buffering=0) as stream:
        os.posix_fadvise(stream.fileno(), 0, 0, os.POSIX_FADV_SEQUENTIAL)
        buffer = bytearray(buffer_bytes)
        offset = 0
        while count := stream.readinto(buffer):
            digest.update(memoryview(buffer)[:count])
            os.posix_fadvise(stream.fileno(), offset, count, os.POSIX_FADV_DONTNEED)
            offset += count
    return digest.hexdigest()


def _copy_file(source, target, buffer_bytes, expected_sha=None):
    """One source pass including SHA; one target verification pass; no retry."""
    started, before, digest = time.monotonic(), _signature(source), hashlib.sha256()
    with Path(source).open('rb', buffering=0) as src, Path(target).open('xb', buffering=0) as dst:
        os.posix_fadvise(src.fileno(), 0, 0, os.POSIX_FADV_SEQUENTIAL)
        buffer = bytearray(buffer_bytes)
        offset = 0
        while count := src.readinto(buffer):
            view = memoryview(buffer)[:count]
            digest.update(view)
            while view:
                written = dst.write(view)
                if not written:
                    raise OSError('Teacher staging write made no progress')
                view = view[written:]
            view.release()
            os.posix_fadvise(src.fileno(), offset, count, os.POSIX_FADV_DONTNEED)
            offset += count
        os.fsync(dst.fileno())
    del buffer
    source_sha = digest.hexdigest()
    if _signature(source) != before:
        raise ValueError(f'{source}: source changed during copy; partial target preserved, no retry')
    target_sha = _hash_file(target, buffer_bytes)
    if _signature(source) != before:
        raise ValueError(f'{source}: source changed during target verification; no retry')
    if target_sha != source_sha or (expected_sha is not None and source_sha != expected_sha):
        raise ValueError(f'{source}: teacher source/target SHA256 mismatch')
    return dict(source_sha256=source_sha, target_sha256=target_sha, bytes=before[2],
                source_stat=_stat(source), target_stat=_stat(target), seconds=time.monotonic() - started,
                source_passes=1, target_verification_passes=1)


def stage_teacher(teacher_root, target_root=None, variant='A', *, cache_root=R84_CACHE_ROOT,
                  source_receipt=None, receipt_path=None, buffer_bytes=MAX_BUFFER_BYTES):
    """Explicit copy or reuse; returns identity/root receipt for the launcher.

    Existing targets are never overwritten or automatically resumed. For B reuse,
    no large file is read or hashed. Unknown external pretraining provenance is
    retained verbatim from the frozen original R94 receipt.
    """
    if not isinstance(buffer_bytes, int) or not 0 < buffer_bytes <= MAX_BUFFER_BYTES:
        raise ValueError('CPU staging buffer must be in (0, 8 MiB]')
    started = time.monotonic()
    plan = inspect_teacher(teacher_root, variant, cache_root=cache_root, source_receipt=source_receipt)
    source = _train_root(teacher_root)
    if target_root is None:
        receipt = copy.deepcopy(plan)
        for problem in SUPPORTED_PROBLEMS:
            for entry in receipt['tasks'][problem]['files'].values():
                entry['target_stat'] = dict(entry['source_stat'])
        receipt.update(mode='reuse_existing_local', copied=False, payload_sha256_verified=False)
    else:
        target = _train_root(target_root)
        if target.exists():
            raise FileExistsError(f'{target}: preserve existing/partial target; no automatic retry')
        if source == target or source in target.parents or target in source.parents:
            raise ValueError('Teacher source and target must be disjoint')
        parent = target.parent
        while not parent.exists():
            parent = parent.parent
        free = shutil.disk_usage(parent).free
        if free < plan['bytes'] + MAX_BUFFER_BYTES:
            raise OSError(f'Teacher staging needs {plan["bytes"]} bytes; available={free}')
        if receipt_path is not None and Path(receipt_path).exists():
            raise FileExistsError(f'{receipt_path}: never overwrite an existing stage receipt')
        print(json.dumps(dict(event='before_teacher_copy', bytes=plan['bytes'], gib=plan['gib'],
                              disk_available_bytes=free, buffer_bytes=buffer_bytes,
                              time_projection=plan['unmeasured_time_projection'],
                              source=str(source), target=str(target)), sort_keys=True), flush=True)
        target.mkdir(parents=True)
        prior = _load_receipt(source_receipt)
        receipt = copy.deepcopy(plan)
        receipt['tasks'] = {}
        for problem in SUPPORTED_PROBLEMS:
            task = plan['tasks'][problem]
            (target / problem).mkdir()
            copied = {}
            for filename in ('metadata.json', FEATURE_FILE):
                expected = None if prior is None else prior['tasks'][problem]['files'][filename].get('target_sha256')
                copied[filename] = _copy_file(source / problem / filename, target / problem / filename,
                                              buffer_bytes, expected)
            # The small metadata may have changed since inspection even if it
            # stayed stable during its own copy. Never certify that mismatch.
            if _digest(_json(target / problem / 'metadata.json')) != task['teacher_metadata_sha256']:
                raise ValueError(f'{problem}: teacher metadata changed after inspection')
            receipt['tasks'][problem] = dict(task, files=copied)
            print(json.dumps(dict(event='teacher_task_copied', problem=problem,
                                  bytes=sum(entry['bytes'] for entry in copied.values())), sort_keys=True), flush=True)
        receipt.update(root=str(target), source_root=str(source), mode='stream_copy_verified',
                       copied=True, payload_sha256_verified=True, buffer_bytes=buffer_bytes,
                       disk_available_before_bytes=free)
    receipt.update(created_utc=datetime.now(timezone.utc).isoformat(), host=socket.gethostname(),
                   seconds=time.monotonic() - started, automatic_retries=False,
                   source_files_modified=False, weights_opened=False)
    if receipt_path is not None:
        _write_new_json(receipt_path, receipt)
    return receipt


def prepare_local(args):
    """Launcher adapter; the caller explicitly supplies source and optional target."""
    return stage_teacher(args.teacher_root, getattr(args, 'target_root', None), args.variant,
                         cache_root=args.cache_root, source_receipt=getattr(args, 'source_receipt', None),
                         receipt_path=getattr(args, 'receipt_path', None),
                         buffer_bytes=getattr(args, 'buffer_bytes', MAX_BUFFER_BYTES))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    for command in ('inspect', 'metadata', 'stage'):
        sub = commands.add_parser(command)
        sub.add_argument('--teacher-root', type=Path, required=True)
        sub.add_argument('--variant', choices=('A', 'B', 'random', 'pretrained'), required=True)
        sub.add_argument('--cache-root', type=Path, default=R84_CACHE_ROOT)
        sub.add_argument('--source-receipt', type=Path)
        sub.add_argument('--receipt-path', type=Path)
        sub.add_argument('--target-root', type=Path)
        if command == 'stage':
            sub.add_argument('--buffer-bytes', type=int, default=MAX_BUFFER_BYTES)
    args = parser.parse_args(argv)
    if args.command in ('inspect', 'metadata'):
        if args.command == 'metadata':
            result = inspect_teacher_metadata(args.teacher_root, args.variant)
        else:
            result = inspect_teacher(args.teacher_root, args.variant, cache_root=args.cache_root,
                                     source_receipt=args.source_receipt, target_root=args.target_root)
        if args.receipt_path is not None:
            _write_new_json(args.receipt_path, result)
    else:
        result = prepare_local(args)
    print(json.dumps(result, sort_keys=True, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
