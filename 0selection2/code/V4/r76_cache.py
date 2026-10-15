"""Scenario-aware R76 input caches, prepared by disjoint independent workers.

Run ``python -m code.V4.r76_cache --help`` for the preparation/finalization CLI.
The on-disk arrays follow R75; R43 targets are derived once at cache open from
original FP64 costs. Use ``inputs_only(batch)`` before every model forward.
Cache directories can be copied to node-local storage without source data.
"""

import argparse
import gc
import operator
import time
from pathlib import Path

import numpy as np
import torch

from code.unified_selector import data as unified_data, registry
from code.unified_selector.registry import (
    GLOBAL_SOLVERS, P2I, PROBLEMS, constraint_bits, problem_descriptor,
)
from .local_geometry import FIELDS, SCALES, local_geometry
from .new_split_data import build_batch, load_dataset, scenario_solver_ids
from .pairwise_objective import comparison_targets, inputs_only
from .r75_cache import (
    COLLATE_CHUNK, GEOMETRY_CHUNK, _close, _geometry_stats, _json,
    _open_arrays, _read, _row_indices, _specs, _transfer, _utc,
)
from .r75_data import _check_split, _file_identity


CACHE_VERSION = 1
CACHE_FORMAT = 'r76_scenario_v1'
CHECKPOINT_ROWS = 8192
CHECKPOINT_SECONDS = 120.
PROGRESS_SECONDS = 5.
DEFAULT_DATA_ROOT = Path(
    '/public/home/shiys/selection_data_new_only_B_filtered_single_reld_mt_seed2_20261014')


def _splits(splits, root):
    if isinstance(splits, str):
        splits = (splits,)
    result = tuple(dict.fromkeys(
        'val' if str(split).lower() == 'validation' else str(split).lower()
        for split in splits))
    if not result:
        raise ValueError('At least one cache split is required')
    for split in result:
        _check_split(split, root)
    return result


def _feature_sources():
    names = ('r75_data.py', 'r75_cache.py', 'new_split_data.py',
             'local_geometry.py', 'pairwise_objective.py', 'r76_cache.py')
    return [_file_identity(path) for path in (
        Path(unified_data.__file__).resolve(), Path(registry.__file__).resolve(),
        *(Path(__file__).with_name(name) for name in names))]


class _Scenario:
    """Explicit source/pool adapter; never replaces legacy module globals."""

    def __init__(self, data_root):
        self.data_root = data_root
        self.path = data_root / 'scenario_manifest.json'
        self.identity = _file_identity(self.path)
        self.manifest = manifest = _read(self.path)
        if (manifest.get('status') != 'complete' or manifest.get('format_version') != 1
                or set(manifest['problems']) != set(PROBLEMS)):
            raise ValueError('R76 requires a complete format1 scenario with all 18 problems')
        self.ids = scenario_solver_ids(manifest)
        self.header = dict(
            scenario_id=manifest['scenario_id'],
            scenario_global_solvers=manifest['global_solvers'],
            scenario_global_solver_ids=self.ids,
            model_global_solvers=list(GLOBAL_SOLVERS),
            scenario_solver_aliases=manifest.get('solver_aliases', {}))
        for problem in PROBLEMS:
            self.pool(problem)

    def pool(self, problem):
        record = self.manifest['problems'][problem]
        names = record['methods']
        if (not names or len(set(names)) != len(names)
                or any(name not in self.ids for name in names)):
            raise ValueError(f'{problem}: invalid scenario candidate names')
        ids = [self.ids[name] for name in names]
        if ids != sorted(ids) or record['global_solver_ids'] != ids:
            raise ValueError(f'{problem}: candidate columns disagree with global IDs')
        return list(names), ids

    def sources(self, problem, split, root):
        _check_split(split, root)
        names, ids = self.pool(problem)
        directory = self.data_root / (problem + split)
        metadata_path = directory / 'metadata.json'
        metadata = _read(metadata_path)
        if metadata['methods'] != names or metadata['global_solver_ids'] != ids:
            raise ValueError(f'{directory}: split and scenario candidate identities disagree')
        results = sorted((directory / 'results').glob('result_*.txt'))
        if ({path.name for path in results} != {'result_' + name + '.txt' for name in names}
                or not all(path.is_file() for path in results)):
            raise ValueError(f'{directory}: result filenames disagree with scenario candidates')
        paths = [self.path, metadata_path, directory / 'dataset.pkl',
                 directory / 'raw_label.pkl', directory / 'input_metadata.npz', *results]
        identities = [_file_identity(path) for path in paths]
        if identities[0] != self.identity:
            raise ValueError('Scenario manifest changed during cache preparation')
        return paths, identities


def _validate_record(record, directory):
    if (record.get('cache_format') != CACHE_FORMAT
            or record.get('cache_version') != CACHE_VERSION
            or not record.get('complete') or record.get('rows_written') != record.get('rows')):
        raise ValueError(f'{directory}: incomplete or incompatible R76 cache')
    problem, split = record['problem'], record['split']
    if problem not in P2I or split not in ('train', 'val', 'test') or record['problem_id'] != P2I[problem]:
        raise ValueError(f'{directory}: invalid problem/split identity')
    ids = scenario_solver_ids(dict(
        global_solvers=record['scenario_global_solvers'],
        global_solver_ids=record['scenario_global_solver_ids'],
        model_global_solvers=record['model_global_solvers'],
        solver_aliases=record['scenario_solver_aliases']))
    pool, pool_ids = record['pool'], record['pool_ids']
    if (not record.get('scenario_id') or not pool or len(set(pool)) != len(pool)
            or any(name not in ids for name in pool)
            or pool_ids != [ids[name] for name in pool] or pool_ids != sorted(pool_ids)
            or record['solver_aliases'] != {
                name: record['scenario_solver_aliases'][name]
                for name in pool if name in record['scenario_solver_aliases']}):
        raise ValueError(f'{directory}: scenario pool identity changed')
    kind = 'matrix' if problem == 'ATSP' else 'coord'
    d_in = None if kind == 'matrix' else 2 if problem == 'TSP' else 4 if problem == 'CVRP' else 8
    mask = [int(index in pool_ids) for index in range(len(GLOBAL_SOLVERS))]
    if (record['kind'] != kind or record['d_in'] != d_in
            or record['mask'] != mask or record['cbits'] != constraint_bits(problem)
            or record['problem_desc'] != problem_descriptor(problem)
            or record['rows'] <= 0 or record['true_length_min'] <= 0
            or record['true_length_max'] < record['true_length_min']
            or record['geometry_fields'] != list(FIELDS)
            or record['geometry_scales'] != list(SCALES)
            or record['geometry_calculation_dtype'] != 'float64'
            or record['geometry_cache_dtype'] != 'float32'):
        raise ValueError(f'{directory}: cached input contract changed')
    expected = _specs(record['rows'], record['true_length_max'], kind, d_in, len(pool))
    if record['arrays'] != expected:
        raise ValueError(f'{directory}: typed array specification changed')
    return expected


class ScenarioCache:
    """Read-only filemapped features and once-per-open FP64-derived R43 labels.

    ``ind`` is the native legal-pool slot, ``mask`` is the 22-ID global mask,
    and ``solver_mask`` is the local candidate mask. CPU gathers own contiguous
    arrays, so an external prefetcher can pin them and transfer asynchronously.
    Raw costs are supervision only; ``inputs_only`` removes all label fields.
    """

    def __init__(self, directory):
        self.directory = Path(directory).expanduser().resolve()
        self._arrays, self._targets, self.raw = {}, {}, {}
        self.lengths = None
        self.metadata = record = _read(self.directory / 'metadata.json')
        _check_split(record['split'], record.get('test_lock_root'))
        specs = _validate_record(record, self.directory)
        self.problem, self.split = record['problem'], record['split']
        self.size, self.kind = record['rows'], record['kind']
        self.pool, self.pool_ids = list(record['pool']), list(record['pool_ids'])
        try:
            self._arrays = _open_arrays(self.directory, specs)
            self.lengths = self._arrays['n']
            if (int(self.lengths.min()) != record['true_length_min']
                    or int(self.lengths.max()) != record['true_length_max']):
                raise ValueError(f'{self.directory}: cached true lengths disagree with metadata')
            self.raw = dict(costs=self._arrays['costs'], winner=self._arrays['winner'],
                            pool=self.pool, pool_ids=self.pool_ids)
            if not np.array_equal(self.raw['winner'], self.raw['costs'].argmin(axis=1)):
                raise ValueError(f'{self.directory}: native winner disagrees with FP64 costs')
            self._targets = {key: value.contiguous().numpy()
                             for key, value in comparison_targets(self.raw['costs']).items()}
        except BaseException:
            self.close()
            raise

    def batch(self, indices, device='cpu'):
        """Gather/crop cached rows only; no parsing, targets, or geometry here."""
        if not self._arrays:
            raise RuntimeError('ScenarioCache is closed')
        rows = _row_indices(indices, self.size)
        n = torch.from_numpy(self.lengths[rows])
        maximum, count, record = int(n.max()), len(rows), self.metadata
        batch = dict(
            problem_id=record['problem_id'], kind=self.kind, n=n,
            node_mask=torch.arange(maximum)[None, :] < n[:, None],
            cbits=torch.tensor(record['cbits'], dtype=torch.float32).repeat(count, 1),
            problem_desc=torch.tensor(record['problem_desc'], dtype=torch.float32).repeat(count, 1),
            coord_dist=torch.from_numpy(self._arrays['coord_dist'][rows]),
            pool_ids=torch.tensor(self.pool_ids, dtype=torch.long),
            mask=torch.tensor(record['mask'], dtype=torch.float32).repeat(count, 1),
            solver_mask=torch.ones(count, len(self.pool), dtype=torch.bool),
            costs=torch.from_numpy(self.raw['costs'][rows].astype(np.float32)),
            ind=torch.from_numpy(self.raw['winner'][rows]))
        if self.kind == 'coord':
            batch['d_in'] = record['d_in']
            for key in ('node', 'node_geom'):
                batch[key] = torch.from_numpy(np.ascontiguousarray(
                    self._arrays[key][:, :maximum][rows]))
        else:
            batch['matrix'] = torch.from_numpy(np.ascontiguousarray(
                self._arrays['matrix'][:, :maximum, :maximum][rows]))
        batch.update({key: torch.from_numpy(value[rows]) for key, value in self._targets.items()})
        return _transfer(batch, device)

    def close(self):
        """Release mappings; callers must stop using external raw views first."""
        _close(self._arrays)
        self._arrays, self._targets, self.raw = {}, {}, {}
        self.lengths = None


class _Progress:
    def __init__(self, root, worker_rank):
        self.path = root / f'cache_progress_worker_{worker_rank}.json'
        self.rank, self.started, self.last = worker_rank, time.monotonic(), 0.

    def update(self, phase, problem=None, split=None, done=0, total=0, force=False, **extra):
        now = time.monotonic()
        if force or now - self.last >= PROGRESS_SECONDS:
            _json(self.path, dict(
                worker_rank=self.rank, phase=phase, problem=problem, split=split,
                done=int(done), total=int(total), heartbeatUTC=_utc(),
                seconds=now - self.started, **extra))
            self.last = now


def _compatible(record, scenario, sources, feature_sources, problem, split):
    pool, ids = scenario.pool(problem)
    return (record.get('cache_format') == CACHE_FORMAT
            and record.get('cache_version') == CACHE_VERSION
            and record.get('problem') == problem and record.get('split') == split
            and record.get('pool') == pool and record.get('pool_ids') == ids
            and record.get('source_files') == sources
            and record.get('feature_sources') == feature_sources
            and all(record.get(key) == value for key, value in scenario.header.items()))


def _geometry(batch, arrays, start, split, moments, device, problem):
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


def _checkpoint(path, record, arrays, moments, stop, before, prior_seconds):
    # Flush only coarse checkpoints; the cursor and moments describe these rows.
    for array in arrays.values():
        array.flush()
    record.update(rows_written=stop, updated_utc=_utc(),
                  seconds=prior_seconds + time.monotonic() - before,
                  geometry_moments=dict(count=moments['count'], sum=moments['sum'].tolist(),
                                        squared_sum=moments['squared_sum'].tolist()))
    _json(path, record)


def _prepare_problem(root, scenario, cache_root, problem, split, device, feature_sources, progress):
    source_paths, source_files = scenario.sources(problem, split, root)
    directory = cache_root / split / problem
    path = directory / 'metadata.json'
    previous = _read(path) if path.is_file() else None
    if previous is not None:
        if not _compatible(previous, scenario, source_files, feature_sources, problem, split):
            raise ValueError(f'{directory}: incompatible scenario/source; preserve it and use a new cache_root')
        if previous['complete']:
            _close(_open_arrays(directory, _validate_record(previous, directory)))
            progress.update('reused', problem, split, previous['rows'], previous['rows'], force=True)
            return previous
    elif directory.exists() and any(directory.iterdir()):
        raise ValueError(f'{directory}: cache files without metadata; preserved for recovery')
    progress.update('load_cpu', problem, split,
                    0 if previous is None else previous['rows_written'],
                    0 if previous is None else previous['rows'], force=True)
    dataset = load_dataset(problem, split=split, data_root=scenario.data_root, root=root)
    if dataset.metadata['source_files'] != source_files:
        raise ValueError(f'{directory}: source changed while opening CPU data')
    if dataset.size <= 0 or np.any(dataset.lengths <= 0) or np.any(dataset.raw_costs <= 0):
        raise ValueError(f'{directory}: R43 requires positive lengths and original solver costs')
    specs = _specs(dataset.size, int(dataset.lengths.max()), dataset.kind, dataset.d_in, dataset.K_p)
    if previous is not None and (previous['rows'] != dataset.size or previous['arrays'] != specs):
        raise ValueError(f'{directory}: partial-cache shape changed; use a new cache_root')
    record = previous if previous is not None else dict(
        dataset.metadata, **scenario.header, directory=str(directory), cache_root=str(cache_root),
        cache_format=CACHE_FORMAT, cache_version=CACHE_VERSION, complete=False, rows_written=0,
        created_utc=_utc(), seconds=0., feature_sources=feature_sources, arrays=specs,
        cbits=list(dataset.cbits), problem_desc=list(dataset.problem_desc), mask=list(dataset.mask),
        instance_ids='original string row positions: str(index); no reordering',
        geometry_calculation_dtype='float64', geometry_cache_dtype='float32',
        geometry_fields=list(FIELDS), geometry_scales=list(SCALES),
        geometry_moments=dict(count=0, sum=[0.] * len(FIELDS), squared_sum=[0.] * len(FIELDS)),
        comparison_targets='pairwise_objective.comparison_targets(original FP64 costs) once at open',
        checkpoint_rows=CHECKPOINT_ROWS, checkpoint_seconds=CHECKPOINT_SECONDS,
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
    before, last_checkpoint, prior_seconds = time.monotonic(), time.monotonic(), record['seconds']
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
                _geometry(batch, arrays, start, split, moments, device, problem)
            else:
                arrays[key][start:stop] = 0
                arrays[key][start:stop, :maximum, :maximum] = batch[key].numpy()
            arrays['n'][start:stop] = dataset.lengths[start:stop]
            arrays['coord_dist'][start:stop] = batch['coord_dist'].numpy()
            arrays['costs'][start:stop] = dataset.raw_costs[start:stop]
            arrays['winner'][start:stop] = dataset.raw_winner[start:stop]
            if (stop - record['rows_written'] >= CHECKPOINT_ROWS
                    or time.monotonic() - last_checkpoint >= CHECKPOINT_SECONDS):
                _checkpoint(path, record, arrays, moments, stop, before, prior_seconds)
                last_checkpoint = time.monotonic()
            progress.update('cached', problem, split, stop, dataset.size)
            del batch
        if source_files != [_file_identity(source) for source in source_paths]:
            raise ValueError(f'{directory}: source changed during preparation; cache remains incomplete')
        if (split == 'train' and dataset.kind == 'coord'
                and moments['count'] != int(dataset.lengths.sum())):
            raise ValueError(f'{directory}: TRAIN geometry must count all true valid nodes only')
        _checkpoint(path, record, arrays, moments, dataset.size, before, prior_seconds)
        record.update(complete=True, completed_utc=_utc(),
                      storage_bytes=sum(array.nbytes for array in arrays.values()))
        _json(path, record)
        progress.update('complete_problem', problem, split, dataset.size, dataset.size, force=True)
    finally:
        _close(arrays)
        del dataset
        gc.collect()
    return record


def _stats(tasks, feature_sources):
    stats = _geometry_stats(tasks, feature_sources)
    if stats is not None:
        stats['sources'] = {
            f'train/{problem}': next(source for source in record['source_files']
                                    if Path(source['path']).name == 'dataset.pkl')
            for problem, record in tasks.items() if record['kind'] == 'coord'}
    return stats


def _manifest(path, root, scenario, cache_root):
    expected = dict(cache_format=CACHE_FORMAT, cache_version=CACHE_VERSION,
                    data_root=str(scenario.data_root), cache_root=str(cache_root), **scenario.header)
    manifest = _read(path) if path.is_file() else dict(
        expected, created_utc=_utc(), splits={}, geometry_stats=None,
        geometry_stats_path=str(root / 'geometry_stats.json'))
    if any(manifest.get(key) != value for key, value in expected.items()):
        raise ValueError(f'{path}: different scenario/cache identity; preserve it and use a new root')
    return manifest


def _totals(manifest):
    manifest['rows_by_split'] = {
        split: sum(record['rows'] for record in tasks.values())
        for split, tasks in manifest['splits'].items()}
    manifest['updated_utc'] = _utc()


def prepare_worker(root, data_root, cache_root, worker_rank=0, workers=1,
                   device='cuda:0', splits=('train', 'val')):
    """Prepare only PROBLEMS[worker_rank::workers], one problem at a time.

    Worker manifests/progress are rank-specific; only ``finalize_cache`` writes
    the shared manifest/statistics. A partial cache resumes at its last coarse
    flushed checkpoint. TEST is gated before any TEST source access.
    """
    root, data_root, cache_root = (Path(path).expanduser().resolve()
                                   for path in (root, data_root, cache_root))
    splits = _splits(splits, root)
    worker_rank, workers = operator.index(worker_rank), operator.index(workers)
    if workers <= 0 or not 0 <= worker_rank < workers:
        raise ValueError('Require workers > 0 and 0 <= worker_rank < workers')
    device = torch.device(device)
    scenario, feature_sources = _Scenario(data_root), _feature_sources()
    root.mkdir(parents=True, exist_ok=True)
    cache_root.mkdir(parents=True, exist_ok=True)
    path = root / f'cache_manifest_worker_{worker_rank}.json'
    manifest = _manifest(path, root, scenario, cache_root)
    problems = list(PROBLEMS[worker_rank::workers])
    if ('workers' in manifest and (manifest['workers'] != workers
                                  or manifest['problems'] != problems)):
        raise ValueError(f'{path}: worker assignment changed')
    manifest.update(worker_rank=worker_rank, workers=workers, problems=problems, complete=False)
    _json(path, manifest)
    progress = _Progress(root, worker_rank)
    try:
        for problem in problems:
            for split in splits:
                record = _prepare_problem(root, scenario, cache_root, problem, split,
                                          device, feature_sources, progress)
                manifest['splits'].setdefault(split, {})[problem] = record
                _totals(manifest)
                _json(path, manifest)
        manifest['geometry_stats'] = _stats(manifest['splits'].get('train', {}), feature_sources)
        manifest['complete'] = all(
            set(tasks) == set(problems) and all(record['complete'] for record in tasks.values())
            for tasks in manifest['splits'].values())
        _totals(manifest)
        _json(path, manifest)
        total = sum(manifest['rows_by_split'].values())
        progress.update('complete', done=total, total=total, force=True)
    except BaseException as error:
        progress.update('failed', force=True, error=f'{type(error).__name__}: {error}')
        raise
    return manifest


def finalize_cache(root, data_root, cache_root, splits=('train', 'val')):
    """Combine complete per-problem records after all preparation workers exit.

    Append requested splits to the manifest, preserving TRAIN/VAL when adding
    locked TEST later. Do not open a ScenarioCache or compute targets/geometry.
    """
    root, data_root, cache_root = (Path(path).expanduser().resolve()
                                   for path in (root, data_root, cache_root))
    splits = _splits(splits, root)
    scenario, feature_sources = _Scenario(data_root), _feature_sources()
    path = root / 'cache_manifest.json'
    manifest = _manifest(path, root, scenario, cache_root)
    combined = {split: dict(tasks) for split, tasks in manifest['splits'].items()}
    for split in dict.fromkeys((*combined, *splits)):
        _check_split(split, root)
        tasks = {}
        for problem in PROBLEMS:
            directory = cache_root / split / problem
            record = _read(directory / 'metadata.json')
            _, sources = scenario.sources(problem, split, root)
            if not _compatible(record, scenario, sources, feature_sources, problem, split):
                raise ValueError(f'{directory}: incompatible scenario/source identity')
            if split == 'test' and record.get('test_lock_root') != str(root):
                raise ValueError(f'{directory}: TEST belongs to a different run lock')
            specs = _validate_record(record, directory)
            _close(_open_arrays(directory, specs))
            moments = record['geometry_moments']
            if (len(moments['sum']) != len(FIELDS) or len(moments['squared_sum']) != len(FIELDS)
                    or not np.isfinite(moments['sum']).all()
                    or not np.isfinite(moments['squared_sum']).all()
                    or (moments['count'] <= 0 if split == 'train' and record['kind'] == 'coord'
                        else moments['count'] != 0)):
                raise ValueError(f'{directory}: invalid TRAIN-only geometry moments')
            tasks[problem] = record
        combined[split] = tasks
    manifest['splits'] = combined
    manifest['geometry_stats'] = _stats(combined.get('train', {}), feature_sources)
    manifest['complete'] = manifest['geometry_stats'] is not None
    _totals(manifest)
    root.mkdir(parents=True, exist_ok=True)
    if manifest['geometry_stats'] is not None:
        _json(root / 'geometry_stats.json', manifest['geometry_stats'])
    _json(path, manifest)
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--data-root', type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument('--cache-root', type=Path, help='Defaults to RUN/cache')
    parser.add_argument('--worker-rank', type=int, default=0)
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--splits', nargs='+', choices=('train', 'val', 'test'), default=['train', 'val'])
    parser.add_argument('--finalize', action='store_true')
    args = parser.parse_args(argv)
    cache_root = args.cache_root if args.cache_root is not None else args.root / 'cache'
    if args.finalize:
        manifest = finalize_cache(args.root, args.data_root, cache_root, splits=args.splits)
        path = args.root / 'cache_manifest.json'
    else:
        manifest = prepare_worker(args.root, args.data_root, cache_root,
                                  worker_rank=args.worker_rank, workers=args.workers,
                                  device=args.device, splits=args.splits)
        path = args.root / f'cache_manifest_worker_{args.worker_rank}.json'
    print(f'{path.resolve()}: complete={manifest["complete"]}, rows={manifest["rows_by_split"]}', flush=True)


if __name__ == '__main__':
    main()
