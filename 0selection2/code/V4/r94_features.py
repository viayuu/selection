"""Input-only FP16 node sidecars for existing R76/R84 ScenarioCaches.

No source regeneration, labels, costs, winners, geometry, statistics or pooling.
Preparation opens ONLY metadata.json, n.npy and node.npy. Do not instantiate a
ScenarioCache for preparation: its constructor reads supervision. Complete
tasks are reusable and partial tasks resume at a flushed stable-N-order cursor.
TRAIN/VAL identity is per-task, never a hash of a mutable aggregate manifest.

FrozenFeatureCache(base_cache, feature_directory, expected_variant=None,
                   expected_provenance=None) retains the ScenarioCache
interface and adds nco_nodes / nco_node_mask on supported tasks. TSP/ATSP are
raw-only pass-throughs. A missing supported sidecar is an error, not an implicit
extraction job. LIB can instead use the attached model.donor on raw inputs.
"""

import argparse
import copy
import fcntl
import hashlib
import json
import mmap
import operator
import os
from pathlib import Path

import numpy as np
import torch

from .r94_donor import OUTPUT_DIM, PROBLEM_NAMES, SUPPORTED_PROBLEMS, StaticReLDDonor


FEATURE_FORMAT = 'r94_frozen_nodes_v1'
FEATURE_FILE = 'nco_nodes.npy'
PARTIAL_FILE = '.nco_nodes.partial.npy'
EXTRACT_ORDER = 'stable_true_n_ascending_then_original_row_v1'


def _read(path):
    return json.loads(Path(path).read_text())


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode('utf-8')).hexdigest()


def _atomic_json(path, value):
    temporary = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    try:
        with temporary.open('w') as stream:
            json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _split(split):
    value = 'val' if str(split).lower() == 'validation' else str(split).lower()
    if value not in ('train', 'val', 'test'):
        raise ValueError('Feature split must be train, val or test')
    return value


def _test_lock(test_lock):
    """The caller supplies R94's formal VAL-only checkpoint/inference lock.

    Uses the actual R80/R84 engine lock: checkpoint is its absolute path and
    checkpoint_sha256 is top-level. The selector checkpoint is authenticated
    by the runner; feature extraction does not load or hash its weights.
    """
    if test_lock is None:
        raise PermissionError('R94 TEST features require a formal R94 VAL lock')
    record = _read(test_lock) if isinstance(test_lock, (str, Path)) else copy.deepcopy(test_lock)
    checkpoint = record.get('checkpoint') if isinstance(record, dict) else None
    sha = record.get('checkpoint_sha256') if isinstance(record, dict) else None
    rule = record.get('inference_rule') if isinstance(record, dict) else None
    if (not isinstance(record, dict) or record.get('experiment') != 'R94'
            or record.get('selected_without_test') is not True
            or record.get('selection_split', 'val') not in ('val', 'validation')
            or record.get('per_problem_assembly') is not False
            or record.get('arm') not in ('A', 'B')
            or not isinstance(record.get('selected_update'), int) or record['selected_update'] <= 0
            or not isinstance(checkpoint, str) or not Path(checkpoint).is_absolute()
            or not isinstance(sha, str) or len(sha) != 64
            or any(char not in '0123456789abcdefABCDEF' for char in sha)
            or not isinstance(rule, (str, dict)) or not rule):
        raise PermissionError('Invalid R94 VAL lock: require VAL-only selection, checkpoint pin and inference_rule')
    return dict(record=record, sha256=_digest(record))


def _manifest(manifest):
    if manifest is None:
        return None
    value = _read(manifest) if isinstance(manifest, (str, Path)) else manifest
    if not isinstance(value, dict) or not isinstance(value.get('splits'), dict):
        raise ValueError('Manifest must contain splits[split][problem] records')
    return value


def _source_record(cache_root, split, problem, manifest=None, data_root=None):
    directory = Path(cache_root).expanduser().resolve() / split / problem
    entry = None
    if manifest is not None:
        entry = manifest.get('splits', {}).get(split, {}).get(problem)
        if not isinstance(entry, dict):
            raise ValueError(f'{split}/{problem}: missing per-task source manifest record')
        # The manifest describes immutable source content. It must not redirect
        # node-local runtime reads back to its original shared-cache directory.
    record = _read(directory / 'metadata.json')
    rows, maximum = record.get('rows'), record.get('true_length_max')
    width = 4 if problem == 'CVRP' else 8
    if (record.get('cache_format') != 'r76_scenario_v1' or record.get('cache_version') != 1
            or record.get('complete') is not True or record.get('rows_written') != rows
            or record.get('problem') != problem or record.get('split') != split
            or record.get('problem_id') != PROBLEM_NAMES.index(problem)
            or record.get('kind') != 'coord' or record.get('d_in') != width
            or not isinstance(rows, int) or rows <= 0 or not isinstance(maximum, int) or maximum < 2
            or not isinstance(record.get('true_length_min'), int)
            or not 2 <= record['true_length_min'] <= maximum):
        raise ValueError(f'{directory}: require a complete audited coordinate ScenarioCache')
    expected = dict(n=dict(file='n.npy', shape=[rows], dtype='int64'),
                    node=dict(file='node.npy', shape=[rows, maximum, width], dtype='float32'))
    if any(record.get('arrays', {}).get(key) != spec for key, spec in expected.items()):
        raise ValueError(f'{directory}: raw input array contract changed')
    if entry is not None:
        for key in ('problem', 'problem_id', 'split', 'rows', 'true_length_min', 'true_length_max',
                    'kind', 'd_in', 'scenario_id'):
            if key in entry and entry[key] != record.get(key):
                raise ValueError(f'{directory}: per-task manifest {key} differs from source cache')
    if data_root is not None and Path(record.get('data_root', '')).expanduser().resolve() != Path(data_root).expanduser().resolve():
        raise ValueError(f'{directory}: data_root identity differs from requested scenario')
    return directory, record


def _source_identity(directory, record):
    inputs = {}
    for key in ('n', 'node'):
        path = directory / record['arrays'][key]['file']
        stat = path.stat()
        inputs[key] = dict(path=str(path.resolve()), bytes=stat.st_size, mtime_ns=stat.st_mtime_ns,
                           spec=record['arrays'][key])
    return dict(directory=str(directory), metadata_sha256=_digest(record), inputs=inputs)


def _open_input(directory, spec):
    array = np.load(directory / spec['file'], mmap_mode='r', allow_pickle=False)
    if (array.dtype != np.dtype(spec['dtype']) or array.shape != tuple(spec['shape'])
            or not array.flags.c_contiguous):
        array._mmap.close()
        raise ValueError(f'{directory}: input shape/dtype/layout changed')
    return array


def _open_features(path, shape, mode='r'):
    array = np.load(path, mmap_mode=mode, allow_pickle=False)
    if array.dtype != np.float16 or array.shape != tuple(shape) or not array.flags.c_contiguous:
        array._mmap.close()
        raise ValueError(f'{path}: invalid FP16 per-node sidecar')
    if mode == 'r':
        array._mmap.madvise(mmap.MADV_RANDOM)
    return array


def _indices(indices, size):
    if torch.is_tensor(indices):
        if indices.device.type != 'cpu' or indices.ndim != 1:
            raise ValueError('Cache indices must be a one-dimensional CPU sequence')
        indices = indices.tolist()
    if isinstance(indices, str):
        indices = [indices]
    if isinstance(indices, np.ndarray) and indices.ndim != 1:
        raise ValueError('Cache indices must be one-dimensional')
    rows = []
    for index in indices:
        if isinstance(index, str):
            row = int(index)
            if str(row) != index:
                raise KeyError(f'Unknown instance ID: {index}')
        else:
            row = operator.index(index)
        if not 0 <= row < size:
            raise IndexError(f'Cache row {row} outside [0, {size})')
        rows.append(row)
    if not rows:
        raise ValueError('Cannot gather an empty feature batch')
    return np.asarray(rows, dtype=np.int64)


def estimate_features(cache_root, splits=('train', 'val'), manifest=None, data_root=None):
    """Metadata-only rectangular FP16 payload estimate, per donor variant.

    Does not stat/open arrays, construct a donor or require TEST access. NPY
    headers and small JSON/lock files are excluded from the byte estimate.
    """
    if isinstance(splits, str):
        splits = (splits,)
    splits = tuple(dict.fromkeys(_split(value) for value in splits))
    if not splits:
        raise ValueError('At least one split is required')
    manifest = _manifest(manifest)
    tasks, total = [], 0
    for split in splits:
        for problem in SUPPORTED_PROBLEMS:
            _, record = _source_record(cache_root, split, problem, manifest, data_root)
            size = record['rows'] * record['true_length_max'] * OUTPUT_DIM * np.dtype('float16').itemsize
            tasks.append(dict(problem=problem, split=split, rows=record['rows'],
                              padded_n=record['true_length_max'], payload_bytes=size))
            total += size
    return dict(format=FEATURE_FORMAT, dtype='float16', output_dim=OUTPUT_DIM,
                payload_bytes=total, payload_gib=total / 1024 ** 3, per_variant=True,
                excluded='NPY headers, metadata and lock files', tasks=tasks,
                raw_only_problems=['TSP', 'ATSP'])


def _prepare_task(directory, record, target, donor, provenance, device, batch_size, lock):
    source = _source_identity(directory, record)
    shape = [record['rows'], record['true_length_max'], OUTPUT_DIM]
    contract = dict(format=FEATURE_FORMAT, problem=record['problem'], split=record['split'],
                    rows=record['rows'], shape=shape, dtype='float16', source_cache=source,
                    donor=provenance, input_only=True, test_lock=lock, extract_order=EXTRACT_ORDER)
    target.mkdir(parents=True, exist_ok=True)
    with (target / '.prepare.lock').open('a') as guard:
        try:
            fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f'{target}: another feature writer is active') from error
        metadata_path, final, partial = target / 'metadata.json', target / FEATURE_FILE, target / PARTIAL_FILE
        prior = _read(metadata_path) if metadata_path.is_file() else None
        if prior is not None:
            if any(prior.get(key) != value for key, value in contract.items()):
                raise ValueError(f'{target}: source/donor/lock identity changed; preserve and use a new feature_root')
            cursor = prior.get('rows_written')
            if not isinstance(cursor, int) or not 0 <= cursor <= record['rows']:
                raise ValueError(f'{target}: invalid stable-N-order completion cursor')
            if prior.get('complete'):
                if cursor != record['rows'] or prior.get('file') != FEATURE_FILE:
                    raise ValueError(f'{target}: inconsistent complete metadata')
                features = _open_features(final, shape)
                features._mmap.close()
                return prior
        else:
            if partial.exists() or final.exists():
                raise ValueError(f'{target}: untracked feature arrays preserved; use a new directory')
            cursor = 0
            prior = dict(contract, rows_written=0, complete=False, file=FEATURE_FILE)
            _atomic_json(metadata_path, prior)
        if final.exists():
            # Recover a crash between array rename and the atomic complete marker.
            if cursor != record['rows'] or partial.exists():
                raise ValueError(f'{target}: unexpected final array preserved for recovery')
            features = _open_features(final, shape)
            features._mmap.close()
        else:
            if partial.exists():
                features = _open_features(partial, shape, 'r+')
            elif cursor == 0:
                features = np.lib.format.open_memmap(partial, mode='w+', dtype=np.float16, shape=tuple(shape))
            else:
                raise ValueError(f'{target}: missing partial feature array')
            lengths = nodes = None
            try:
                lengths = _open_input(directory, record['arrays']['n'])
                nodes = _open_input(directory, record['arrays']['node'])
                if (int(lengths.min()) != record['true_length_min']
                        or int(lengths.max()) != record['true_length_max']):
                    raise ValueError(f'{directory}: true lengths disagree with metadata bounds')
                # Only the small length/order vectors enter RAM, never all H or
                # all raw nodes. Group globally, not inside random 128-row chunks.
                order = np.argsort(lengths, kind='stable')
                sorted_lengths = lengths[order]
                start = cursor
                while start < record['rows']:
                    group_end = int(np.searchsorted(sorted_lengths, sorted_lengths[start], side='right'))
                    stop = min(start + batch_size, group_end)
                    rows = order[start:stop]
                    n = torch.from_numpy(np.array(lengths[rows], copy=True))
                    maximum = int(n[0])
                    batch = dict(problem_id=record['problem_id'], kind='coord', d_in=record['d_in'],
                                 node=torch.from_numpy(np.array(nodes[rows, :maximum], copy=True)).to(device),
                                 n=n.to(device), node_mask=(torch.arange(maximum)[None] < n[:, None]).to(device))
                    with torch.no_grad():
                        hidden = donor(batch)
                    if (hidden.shape != (stop - start, maximum, OUTPUT_DIM)
                            or not torch.isfinite(hidden).all()
                            or hidden[~batch['node_mask']].count_nonzero()):
                        raise ValueError('Donor must return finite node-aligned features with zero padding')
                    values = hidden.detach().to(device='cpu', dtype=torch.float16).numpy()
                    if not np.isfinite(values).all():
                        raise ValueError('Donor features overflowed the FP16 sidecar')
                    features[rows, :maximum] = values
                    features[rows, maximum:] = 0
                    features.flush()
                    prior['rows_written'] = stop
                    _atomic_json(metadata_path, prior)
                    start = stop
            finally:
                features._mmap.close()
                for array in (lengths, nodes):
                    if array is not None:
                        array._mmap.close()
            if _source_identity(directory, _read(directory / 'metadata.json')) != source:
                raise ValueError(f'{directory}: source changed during extraction; sidecar remains incomplete')
            os.replace(partial, final)
        prior['complete'] = True
        _atomic_json(metadata_path, prior)
        return prior


def prepare_features(cache_root, feature_root, split, donor, device, batch_size=128,
                     test_lock=None, manifest=None, data_root=None):
    """Sequential supported16 extraction into feature_root/split/problem.

    Donor is an already constructed StaticReLDDonor (random or pretrained).
    Caller explicitly chooses device; no jobs, GPU discovery or source caches
    are created. TEST is forbidden until _test_lock's formal R94 lock exists.
    Within each task, stable-sort by true N and use exact-N chunks no larger
    than batch_size, scattering H back to original rows. rows_written counts
    positions in this deterministic extraction order, not an original prefix.
    Returns a JSON-ready per-task manifest, without writing a shared manifest.
    """
    split = _split(split)
    lock = _test_lock(test_lock) if split == 'test' else None
    batch_size = operator.index(batch_size)
    if isinstance(batch_size, bool) or batch_size <= 0:
        raise ValueError('batch_size must be a positive integer')
    manifest = _manifest(manifest)
    sources = {problem: _source_record(cache_root, split, problem, manifest, data_root)
               for problem in SUPPORTED_PROBLEMS}
    provenance = donor.provenance()
    if (provenance.get('variant') not in ('random', 'pretrained')
            or provenance.get('output_dim') != OUTPUT_DIM
            or provenance.get('supported_problems') != list(SUPPORTED_PROBLEMS)):
        raise ValueError('Donor provenance does not match the static ReLD feature contract')
    donor.to(torch.device(device)).eval()
    feature_root = Path(feature_root).expanduser().resolve()
    records = {}
    for problem, (directory, record) in sources.items():
        target = feature_root / split / problem
        print(f'[R94 features] {split}/{problem}: preparing {record["rows"]} exact-N input rows', flush=True)
        result = _prepare_task(directory, record, target, donor, provenance, device, batch_size, lock)
        records[problem] = dict(result, directory=str(target))
        print(f'[R94 features] {split}/{problem}: complete {result["rows_written"]} rows', flush=True)
    return dict(format=FEATURE_FORMAT, complete=True, splits={split: records},
                donor=provenance, raw_only_problems=['TSP', 'ATSP'])


class FrozenFeatureCache:
    """Lazy FP16 sidecar over a ScenarioCache; close() closes both mappings.

    raw/pool/pool_ids/size/lengths/metadata/directory and other base attributes
    are delegated unchanged. nco_node_mask is the very same node_mask tensor.
    No normalization/stats or extraction is performed on the batch hot path.
    """

    def __init__(self, base_cache, feature_directory, expected_variant=None, expected_provenance=None):
        self.base_cache = base_cache
        self.feature_directory = Path(feature_directory).expanduser().resolve()
        self.feature_metadata, self._features, self._closed = None, None, False
        self._feature_fd = None
        if expected_variant is not None and expected_variant not in ('random', 'pretrained'):
            raise ValueError('Unknown expected donor variant')
        if expected_provenance is not None and not isinstance(expected_provenance, dict):
            raise TypeError('expected_provenance must be the dict from model.donor.provenance()')
        if base_cache.problem in ('TSP', 'ATSP'):
            return
        if base_cache.problem not in SUPPORTED_PROBLEMS:
            raise ValueError(f'Unsupported R94 problem {base_cache.problem}')
        directory, record = _source_record(base_cache.directory.parent.parent,
                                           base_cache.split, base_cache.problem,
                                           manifest=dict(splits={base_cache.split: {
                                               base_cache.problem: dict(directory=str(base_cache.directory))}}))
        metadata = _read(self.feature_directory / 'metadata.json')
        shape = [base_cache.size, record['true_length_max'], OUTPUT_DIM]
        if (metadata.get('format') != FEATURE_FORMAT or metadata.get('complete') is not True
                or metadata.get('rows_written') != base_cache.size or metadata.get('rows') != base_cache.size
                or metadata.get('file') != FEATURE_FILE or metadata.get('dtype') != 'float16'
                or metadata.get('shape') != shape or metadata.get('problem') != base_cache.problem
                or metadata.get('split') != base_cache.split or metadata.get('input_only') is not True
                or metadata.get('extract_order') != EXTRACT_ORDER
                or metadata.get('source_cache') != _source_identity(directory, record)
                or base_cache.metadata != record):
            raise ValueError(f'{self.feature_directory}: incomplete or mismatched source feature cache')
        if base_cache.split == 'test':
            saved = metadata.get('test_lock')
            if not isinstance(saved, dict) or _test_lock(saved.get('record')) != saved:
                raise PermissionError('TEST sidecar lacks an intact formal R94 VAL lock')
        donor = metadata.get('donor', {})
        if (donor.get('variant') not in ('random', 'pretrained') or donor.get('output_dim') != OUTPUT_DIM
                or donor.get('supported_problems') != list(SUPPORTED_PROBLEMS)):
            raise ValueError('Sidecar donor provenance is invalid')
        if expected_variant is not None and donor['variant'] != expected_variant:
            raise ValueError('Feature donor variant differs from the requested arm')
        if expected_provenance is not None and donor != expected_provenance:
            raise ValueError('Feature donor provenance differs from the attached model donor')
        self._features = _open_features(self.feature_directory / FEATURE_FILE, shape)
        try:
            self._feature_fd = os.open(self.feature_directory / FEATURE_FILE, os.O_RDONLY)
            os.posix_fadvise(self._feature_fd, 0, 0, os.POSIX_FADV_RANDOM)
        except BaseException:
            if self._feature_fd is not None:
                os.close(self._feature_fd)
                self._feature_fd = None
            self._features._mmap.close()
            self._features = None
            raise
        self.feature_metadata = metadata

    def __getattr__(self, name):
        return getattr(self.base_cache, name)

    def batch(self, indices, device='cpu'):
        if self._closed:
            raise RuntimeError('FrozenFeatureCache is closed')
        rows = _indices(indices, self.base_cache.size)
        batch = self.base_cache.batch(rows, device=device)
        if self._features is not None:
            mask = batch['node_mask']
            maximum = mask.shape[1]
            # Queue just the requested row spans asynchronously. RANDOM alone
            # saves bytes but serializes page faults; implicit read-around was
            # reading gigabytes for a ~26MB batch under the job's RAM limit.
            row_stride = int(self._features.strides[0])
            wanted = maximum * OUTPUT_DIM * self._features.dtype.itemsize
            for row in np.unique(rows):
                os.posix_fadvise(self._feature_fd,
                    int(self._features.offset) + int(row) * row_stride,
                    wanted, os.POSIX_FADV_WILLNEED)
            # Slice a view first, then gather ONLY requested rows and nmax.
            values = np.ascontiguousarray(self._features[:, :maximum][rows])
            nodes = torch.from_numpy(values).to(device=mask.device)
            nodes.masked_fill_(~mask[..., None], 0)
            batch['nco_nodes'], batch['nco_node_mask'] = nodes, mask
        return batch

    def close(self):
        if not self._closed:
            if self._feature_fd is not None:
                os.close(self._feature_fd)
                self._feature_fd = None
            if self._features is not None:
                self._features._mmap.close()
                self._features = None
            self.base_cache.close()
            self._closed = True


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    estimate = commands.add_parser('estimate', help='Read metadata only; never extract')
    prepare = commands.add_parser('prepare', help='Explicitly requested sequential extraction only')
    for command in (estimate, prepare):
        command.add_argument('--cache-root', type=Path, required=True)
        command.add_argument('--manifest', type=Path)
        command.add_argument('--data-root', type=Path)
    estimate.add_argument('--splits', nargs='+', default=['train', 'val'], choices=['train', 'val', 'test'])
    prepare.add_argument('--feature-root', type=Path, required=True)
    prepare.add_argument('--split', choices=['train', 'val', 'test'], required=True)
    prepare.add_argument('--variant', choices=['random', 'pretrained'], required=True)
    prepare.add_argument('--checkpoint-path', type=Path)
    prepare.add_argument('--device', required=True)
    prepare.add_argument('--batch-size', type=int, default=128)
    prepare.add_argument('--test-lock', type=Path)
    args = parser.parse_args(argv)
    if args.command == 'estimate':
        result = estimate_features(args.cache_root, args.splits, args.manifest, args.data_root)
    else:
        if args.split == 'test':
            _test_lock(args.test_lock)
        result = prepare_features(args.cache_root, args.feature_root, args.split,
                                  StaticReLDDonor(args.variant, args.checkpoint_path),
                                  args.device, args.batch_size, args.test_lock, args.manifest, args.data_root)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
