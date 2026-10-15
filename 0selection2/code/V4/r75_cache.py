"""Contiguous, resumable R75 caches; only selected rows ever leave CPU storage.

``prepare_cache`` returns ``splits[split][problem]`` metadata records. Open a
record's ``directory`` with ``ProblemCache``; its ``raw`` arrays are read-only
FP64 costs and native int64 winners, while ``batch`` follows the unified FP32
collator contract. No model, solver, augmentation, or relabeling is involved.
"""

import gc
import json
import operator
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from code.unified_selector import data as unified_data
from code.unified_selector.registry import POOLS, PROBLEMS, problem_to_pool_mask
from .local_geometry import FIELDS, SCALES, local_geometry
from .r75_data import (_check_split, _file_identity, _source_paths,
                       build_batch, load_dataset)


DEFAULT_CACHE_ROOT = Path('/tmp/shiys_r75_large_complete_labels_cache')
CACHE_VERSION = 1
COLLATE_CHUNK = 128
GEOMETRY_CHUNK = 32


def _utc():
    return datetime.now(timezone.utc).isoformat()


def _json(path, value):
    path = Path(path)
    temporary = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    os.replace(temporary, path)


def _read(path):
    return json.loads(Path(path).read_text())


def _progress(root, phase, problem, split, done, total, started):
    _json(root / 'cache_progress.json', dict(
        phase=phase, problem=problem, split=split, done=int(done), total=int(total),
        heartbeatUTC=_utc(), seconds=time.monotonic() - started))


def _feature_sources():
    return [_file_identity(path) for path in (
        Path(unified_data.__file__).resolve(),
        Path(__file__).with_name('r75_data.py'),
        Path(__file__).with_name('local_geometry.py'))]


def _specs(size, maximum, kind, d_in, candidates):
    arrays = dict(n=((size,), 'int64'), costs=((size, candidates), 'float64'),
                  winner=((size,), 'int64'), coord_dist=((size,), 'int64'))
    if kind == 'coord':
        arrays.update(node=((size, maximum, d_in), 'float32'),
                      node_geom=((size, maximum, len(FIELDS)), 'float32'))
    else:
        arrays['matrix'] = ((size, maximum, maximum), 'float32')
    return {key: dict(file=key + '.npy', shape=list(shape), dtype=dtype)
            for key, (shape, dtype) in arrays.items()}


def _close(arrays):
    for array in arrays.values():
        array._mmap.close()


def _open_arrays(directory, specs, writable=False, done=0):
    arrays = {}
    try:
        for key, spec in specs.items():
            path = directory / spec['file']
            if path.is_file():
                array = np.load(path, mmap_mode='r+' if writable else 'r', allow_pickle=False)
            elif writable and done == 0:
                array = np.lib.format.open_memmap(
                    path, mode='w+', dtype=spec['dtype'], shape=tuple(spec['shape']))
            else:
                raise ValueError(f'{directory}: missing cache array {key}; preserve this partial cache')
            arrays[key] = array
            if (array.shape != tuple(spec['shape']) or array.dtype != np.dtype(spec['dtype'])
                    or not array.flags.c_contiguous):
                raise ValueError(f'{path}: cache shape/dtype/layout changed')
    except BaseException:
        _close(arrays)
        raise
    return arrays


def _row_indices(indices, size):
    if torch.is_tensor(indices):
        if indices.device.type != 'cpu' or indices.ndim != 1:
            raise ValueError('Cache indices must be a one-dimensional CPU sequence')
        indices = indices.tolist()
    elif isinstance(indices, str):
        indices = [indices]
    elif isinstance(indices, np.ndarray) and indices.ndim != 1:
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
        raise ValueError('Cannot gather an empty cache batch')
    return np.asarray(rows, dtype=np.int64)


def _transfer(batch, device):
    device = torch.device(device)
    if device.type == 'cpu':
        return batch
    return {key: (value.pin_memory().to(device, non_blocking=True)
                  if device.type == 'cuda' else value.to(device))
            if torch.is_tensor(value) else value for key, value in batch.items()}


class ProblemCache:
    """Lazy NumPy memmaps for one problem/split, never a full-device cache."""

    def __init__(self, directory):
        self.directory = Path(directory).expanduser().resolve()
        self.metadata = _read(self.directory / 'metadata.json')
        record = self.metadata
        self.problem, self.split = record['problem'], record['split']
        _check_split(self.split, record.get('test_lock_root'))
        self.size, self.kind = record['rows'], record['kind']
        self.pool, self.pool_ids = list(record['pool']), list(record['pool_ids'])
        if (not record['complete'] or record.get('rows_written') != self.size
                or record.get('cache_version') != CACHE_VERSION):
            raise ValueError(f'{self.directory}: incomplete or incompatible cache')
        if (self.pool != list(POOLS[self.problem])
                or self.pool_ids != list(problem_to_pool_mask(self.problem)[0])):
            raise ValueError(f'{self.directory}: registry pool identity changed')
        expected = _specs(self.size, record['true_length_max'], self.kind,
                          record['d_in'], len(self.pool))
        if record['arrays'] != expected:
            raise ValueError(f'{self.directory}: typed array specification changed')
        self._arrays = _open_arrays(self.directory, expected)
        self.lengths = self._arrays['n']
        self.raw = dict(costs=self._arrays['costs'], winner=self._arrays['winner'],
                        pool=self.pool, pool_ids=self.pool_ids)

    def batch(self, indices, device='cuda:0'):
        """Gather and crop on CPU, then pin/transfer only the current batch."""
        rows = _row_indices(indices, self.size)
        n = torch.from_numpy(self.lengths[rows])
        maximum, count = int(n.max()), len(rows)
        record = self.metadata
        batch = dict(
            problem_id=record['problem_id'], kind=self.kind, n=n,
            node_mask=torch.arange(maximum)[None, :] < n[:, None],
            cbits=torch.tensor(record['cbits'], dtype=torch.float32).repeat(count, 1),
            problem_desc=torch.tensor(record['problem_desc'], dtype=torch.float32).repeat(count, 1),
            coord_dist=torch.from_numpy(self._arrays['coord_dist'][rows]),
            pool_ids=torch.tensor(self.pool_ids, dtype=torch.long),
            mask=torch.tensor(record['mask'], dtype=torch.float32).repeat(count, 1),
            costs=torch.from_numpy(self.raw['costs'][rows].astype(np.float32)),
            ind=torch.from_numpy(self.raw['winner'][rows]))
        # Slice before advanced indexing so high-n padding is never gathered.
        if self.kind == 'coord':
            batch.update(d_in=record['d_in'])
            for key in ('node', 'node_geom'):
                batch[key] = torch.from_numpy(
                    np.ascontiguousarray(self._arrays[key][:, :maximum][rows]))
        else:
            batch['matrix'] = torch.from_numpy(np.ascontiguousarray(
                self._arrays['matrix'][:, :maximum, :maximum][rows]))
        return _transfer(batch, device)

    def close(self):
        """Release mappings when this cache and its raw views are no longer used."""
        _close(self._arrays)
        self._arrays = {}


def _compatible(record, source_files, feature_sources, problem, split):
    return (record.get('cache_version') == CACHE_VERSION
            and record.get('source_files') == source_files
            and record.get('feature_sources') == feature_sources
            and record.get('problem') == problem and record.get('split') == split
            and record.get('pool') == list(POOLS[problem])
            and record.get('pool_ids') == list(problem_to_pool_mask(problem)[0]))


def _geometry(batch, arrays, start, split, moments, device, root, problem, total, started):
    maximum = arrays['node_geom'].shape[1]
    for offset in range(0, len(batch['n']), GEOMETRY_CHUNK):
        stop = min(offset + GEOMETRY_CHUNK, len(batch['n']))
        chunk_max = int(batch['n'][offset:stop].max())
        chunk = _transfer(dict(
            xy=batch['node'][offset:stop, :chunk_max, :2].contiguous(),
            mask=batch['node_mask'][offset:stop, :chunk_max].contiguous()), device)
        with torch.no_grad(), torch.autocast(device_type=torch.device(device).type, enabled=False):
            features = local_geometry(chunk['xy'].double(), chunk['mask']).float().cpu().numpy()
        if not np.isfinite(features).all():
            raise ValueError(f'{problem}/{split}: nonfinite cached geometry')
        section = slice(start + offset, start + stop)
        arrays['node_geom'][section, :chunk_max] = features
        arrays['node_geom'][section, chunk_max:maximum] = 0
        if split == 'train':
            valid = features[batch['node_mask'][offset:stop, :chunk_max].numpy()].astype(np.float64)
            moments['count'] += len(valid)
            moments['sum'] += valid.sum(axis=0)
            moments['squared_sum'] += np.square(valid).sum(axis=0)
        _progress(root, 'geometry', problem, split, start + stop, total, started)


def _prepare_problem(root, data_root, cache_root, problem, split, device, feature_sources, started):
    source_paths = _source_paths(problem, split, data_root, root)
    source_files = [_file_identity(path) for path in source_paths]
    directory = cache_root / split / problem
    path = directory / 'metadata.json'
    previous = _read(path) if path.is_file() else None
    if previous is not None:
        if not _compatible(previous, source_files, feature_sources, problem, split):
            raise ValueError(f'{directory}: source identities changed; preserved cache needs a new cache_root')
        if previous['complete']:
            cache = ProblemCache(directory)
            cache.close()
            _progress(root, 'reused', problem, split, previous['rows'], previous['rows'], started)
            return previous
    elif directory.exists() and any(directory.iterdir()):
        raise ValueError(f'{directory}: cache files without metadata; preserved for recovery')

    _progress(root, 'load_cpu', problem, split,
              0 if previous is None else previous['rows_written'],
              0 if previous is None else previous['rows'], started)
    dataset = load_dataset(problem, split=split, data_root=data_root, root=root)
    if dataset.metadata['source_files'] != source_files:
        raise ValueError(f'{directory}: source files changed while opening CPU data')
    specs = _specs(dataset.size, int(dataset.lengths.max()), dataset.kind,
                   dataset.d_in, dataset.K_p)
    if previous is not None and (previous['rows'] != dataset.size or previous['arrays'] != specs):
        raise ValueError(f'{directory}: partial cache shape changed; preserve it and use a new cache_root')
    record = previous if previous is not None else dict(
        dataset.metadata, directory=str(directory), cache_root=str(cache_root),
        cache_version=CACHE_VERSION, complete=False, rows_written=0,
        created_utc=_utc(), seconds=0., feature_sources=feature_sources, arrays=specs,
        cbits=list(dataset.cbits), problem_desc=list(dataset.problem_desc), mask=list(dataset.mask),
        instance_ids='original string row positions: str(index); no reordering',
        geometry_calculation_dtype='float64', geometry_cache_dtype='float32',
        geometry_fields=list(FIELDS), geometry_scales=list(SCALES),
        geometry_moments=dict(count=0, sum=[0.] * len(FIELDS), squared_sum=[0.] * len(FIELDS)),
        test_lock_root=str(root) if split == 'test' else None)
    done = operator.index(record['rows_written'])
    if not 0 <= done <= dataset.size:
        raise ValueError(f'{directory}: invalid partial-cache cursor')
    moments = dict(count=record['geometry_moments']['count'],
                   sum=np.asarray(record['geometry_moments']['sum'], dtype=np.float64),
                   squared_sum=np.asarray(record['geometry_moments']['squared_sum'], dtype=np.float64))
    directory.mkdir(parents=True, exist_ok=True)
    _json(path, record)
    arrays = _open_arrays(directory, specs, writable=True, done=done)
    before, prior_seconds = time.monotonic(), record['seconds']
    try:
        for start in range(done, dataset.size, COLLATE_CHUNK):
            stop = min(start + COLLATE_CHUNK, dataset.size)
            batch = build_batch(dataset, range(start, stop), device='cpu', include_geometry=False)
            if not np.array_equal(batch['n'].numpy(), dataset.lengths[start:stop]):
                raise ValueError(f'{problem}/{split}: original parser and node counts disagree')
            key = 'node' if dataset.kind == 'coord' else 'matrix'
            if not torch.isfinite(batch[key]).all() or not torch.isfinite(batch['costs']).all():
                raise ValueError(f'{problem}/{split}: nonfinite unified features/costs')
            maximum = batch[key].shape[1]
            if key == 'node':
                arrays[key][start:stop, :maximum] = batch[key].numpy()
                arrays[key][start:stop, maximum:] = 0
                _geometry(batch, arrays, start, split, moments, device, root,
                          problem, dataset.size, started)
            else:
                arrays[key][start:stop] = 0
                arrays[key][start:stop, :maximum, :maximum] = batch[key].numpy()
            arrays['n'][start:stop] = dataset.lengths[start:stop]
            arrays['coord_dist'][start:stop] = batch['coord_dist'].numpy()
            arrays['costs'][start:stop] = dataset.raw_costs[start:stop]
            arrays['winner'][start:stop] = dataset.raw_winner[start:stop]
            # Publish the cursor only after all arrays and TRAIN moments agree.
            for array in arrays.values():
                array.flush()
            record.update(rows_written=stop, updated_utc=_utc(),
                          seconds=prior_seconds + time.monotonic() - before,
                          geometry_moments=dict(count=moments['count'], sum=moments['sum'].tolist(),
                                                squared_sum=moments['squared_sum'].tolist()))
            _json(path, record)
            _progress(root, 'cached', problem, split, stop, dataset.size, started)
            del batch
        if source_files != [_file_identity(path) for path in source_paths]:
            raise ValueError(f'{directory}: source changed during preparation; cache remains incomplete')
        if (split == 'train' and dataset.kind == 'coord'
                and moments['count'] != int(dataset.lengths.sum())):
            raise ValueError(f'{directory}: TRAIN geometry must count only all true valid nodes')
        record.update(complete=True, completed_utc=_utc(),
                      storage_bytes=sum(array.nbytes for array in arrays.values()))
        _json(path, record)
    finally:
        _close(arrays)
        del dataset
        gc.collect()
    return record


def _geometry_stats(tasks, feature_sources):
    if any(problem not in tasks or not tasks[problem]['complete'] for problem in PROBLEMS):
        return None
    counts, sources = {}, {}
    sums, squares = np.zeros(len(FIELDS), dtype=np.float64), np.zeros(len(FIELDS), dtype=np.float64)
    for problem in PROBLEMS:
        record = tasks[problem]
        if record['kind'] != 'coord':
            continue
        moments = record['geometry_moments']
        counts[problem] = int(moments['count'])
        sums += np.asarray(moments['sum'], dtype=np.float64)
        squares += np.asarray(moments['squared_sum'], dtype=np.float64)
        sources[f'train/{problem}'] = record['source_files'][0]
    count = sum(counts.values())
    if count <= 0:
        raise ValueError('Geometry normalization requires true TRAIN coordinate nodes')
    mean = sums / count
    std = np.maximum(np.sqrt(np.maximum(squares / count - np.square(mean), 0)), 1e-6)
    return dict(fields=list(FIELDS), scales=list(SCALES), mean=mean.tolist(), std=std.tolist(),
                fitted_on='all coordinate training valid nodes only', valid_train_nodes=count,
                valid_nodes_by_problem=counts, sources=sources, feature_sources=feature_sources,
                calculation_dtype='float64', cache_dtype='float32', statistics_dtype='float64',
                statistics_input='cached float32 geometry converted to float64', std_floor=1e-6,
                sum=sums.tolist(), squared_sum=squares.tolist(),
                neighbor_policy='exclude self/padding; include all kth-distance ties; use all when fewer than k')


def prepare_cache(root: Path, data_root: Path, cache_root: Path, device='cuda:0',
                  splits=('train', 'val')) -> dict:
    """Prepare one CPU-loaded problem at a time and return persistent metadata.

    Normalization lives at ``root/geometry_stats.json`` and in the returned
    ``geometry_stats`` key. Requested splits are appended without discarding
    existing split metadata. TEST is gated before even statting its sources.
    Matching partial caches resume at their last flushed CPU chunk; mismatched
    caches are preserved and rejected, never silently overwritten or reused.
    """
    root = Path(root).expanduser().resolve()
    data_root = Path(data_root).expanduser().resolve()
    cache_root = Path(cache_root).expanduser().resolve()
    if isinstance(splits, str):
        splits = (splits,)
    splits = tuple(dict.fromkeys(str(split).lower() for split in splits))
    if not splits:
        raise ValueError('At least one cache split is required')
    for split in splits:
        _check_split(split, root)
    root.mkdir(parents=True, exist_ok=True)
    cache_root.mkdir(parents=True, exist_ok=True)
    path = root / 'cache_manifest.json'
    manifest = _read(path) if path.is_file() else dict(
        cache_version=CACHE_VERSION, data_root=str(data_root), cache_root=str(cache_root),
        created_utc=_utc(), splits={}, geometry_stats=None,
        geometry_stats_path=str(root / 'geometry_stats.json'))
    if (manifest.get('cache_version') != CACHE_VERSION
            or manifest.get('data_root') != str(data_root)
            or manifest.get('cache_root') != str(cache_root)):
        raise ValueError('Existing manifest belongs to a different cache; preserve it and use a new root')
    started, feature_sources = time.monotonic(), _feature_sources()
    manifest.update(complete=False, updated_utc=_utc())
    _json(path, manifest)
    for split in splits:
        tasks = manifest['splits'].setdefault(split, {})
        for problem in PROBLEMS:
            tasks[problem] = _prepare_problem(root, data_root, cache_root, problem, split,
                                               device, feature_sources, started)
            manifest.update(updated_utc=_utc(), seconds=time.monotonic() - started)
            _json(path, manifest)
        if split == 'train':
            stats = _geometry_stats(tasks, feature_sources)
            _json(root / 'geometry_stats.json', stats)
            manifest['geometry_stats'] = stats
            _json(path, manifest)
    if manifest.get('geometry_stats') is not None:
        _json(root / 'geometry_stats.json', manifest['geometry_stats'])
    manifest.update(
        complete=(manifest.get('geometry_stats') is not None and all(
            len(tasks) == len(PROBLEMS) and all(record['complete'] for record in tasks.values())
            for tasks in manifest['splits'].values())),
        updated_utc=_utc(), seconds=time.monotonic() - started,
        rows_by_split={split: sum(record['rows'] for record in tasks.values())
                       for split, tasks in manifest['splits'].items()})
    _json(path, manifest)
    _progress(root, 'complete', None, None,
              sum(manifest['rows_by_split'].values()), sum(manifest['rows_by_split'].values()), started)
    return manifest
