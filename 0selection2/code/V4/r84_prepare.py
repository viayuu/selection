"""Prepare the new90k scenario once, reusing matching R76 cached input rows.

Only TRAIN supplies geometry moments, SBS identities, and fresh initialization.
TEST feature preparation requires a completed arm's unique VAL checkpoint lock.
No global loader/source variables are replaced in R75/R76 modules.
"""

import argparse
import fcntl
import gc
import hashlib
import os
import shutil
import time
from pathlib import Path

import numpy as np
import torch

from code.unified_selector.registry import PROBLEMS
from .local_geometry import FIELDS, SCALES
from .new_split_data import build_batch, load_dataset
from .r75_cache import COLLATE_CHUNK, _close, _open_arrays, _specs
from .r75_data import _file_identity
from .r76_cache import (
    CACHE_FORMAT, CACHE_VERSION, _Scenario, _feature_sources, _geometry, _stats,
    _validate_record,
)
from .r76_experiment import file_hash, read_json, utc, write_json
from .r77_data import fit_sbs, parameter_hash


V4 = Path(__file__).resolve().parent
DEFAULT_ROOT = V4 / 'runs/R84_new90k_attention'
DEFAULT_DATA_ROOT = Path('/public/home/shiys/selection_data_new90k_B_filtered_single_reld_mt_seed2_20261018')
DEFAULT_PREPARED_ROOT = DEFAULT_ROOT / 'prepared'
DEFAULT_FRESH_INIT = V4 / 'runs/R77_sample_size_batch1024/identicalinit_seed2.pt'
OLD_DATA_ROOT = Path('/public/home/shiys/selection_data_new_only_B_filtered_single_reld_mt_seed2_20261014')
SPLIT_ROWS = dict(train=81000, val=4500, test=4500)


def feature_sources():
    return [*_feature_sources(), _file_identity(Path(__file__).resolve())]


def test_permission(root):
    if root is None:
        raise PermissionError('TEST preparation requires --test-lock-root RUN/ARM')
    root = Path(root).resolve()
    lock = read_json(root / 'final_test_lock.json')
    if (lock.get('experiment') != 'R84' or lock.get('selected_without_test') is not True
            or lock.get('per_problem_assembly') is not False
            or Path(lock['checkpoint']).resolve() != (root / 'best.pt').resolve()
            or lock['checkpoint_sha256'] != file_hash(root / 'best.pt')):
        raise PermissionError('TEST feature preparation requires an unchanged R84 unique VAL lock')
    return root


class ReusableInputs:
    """Old input/geometry arrays only; never reuse labels or normalization."""

    def __init__(self, problem, roots, old_data_root=OLD_DATA_ROOT):
        self.sources = []
        for cache_root in roots:
            for split in ('train', 'val', 'test'):
                directory = Path(cache_root) / split / problem
                inputs_path = Path(old_data_root) / (problem + split) / 'input_metadata.npz'
                if not (directory / 'metadata.json').is_file() or not inputs_path.is_file():
                    continue
                record = read_json(directory / 'metadata.json')
                if (not record.get('complete') or record.get('rows_written') != record.get('rows')
                        or record.get('geometry_fields') != list(FIELDS)
                        or record.get('geometry_scales') != list(SCALES)):
                    continue
                with np.load(inputs_path, allow_pickle=False) as metadata:
                    ids = metadata['source_index'].astype(np.int64)
                if len(ids) != record['rows'] or len(np.unique(ids)) != len(ids):
                    raise ValueError('Old cached source IDs must be distinct and aligned')
                fields = ('node', 'node_geom', 'n', 'coord_dist') if record['kind'] == 'coord' else (
                    'matrix', 'n', 'coord_dist')
                specs = {key: record['arrays'][key] for key in fields}
                arrays = _open_arrays(directory, specs)
                self.sources.append((ids, np.argsort(ids, kind='stable'), arrays, directory))

    def copy_geometry(self, batch, ids, target):
        """Reuse only rows whose actual collated fields and sizes are identical."""
        reused = np.zeros(len(ids), dtype=bool)
        key = 'node' if batch['kind'] == 'coord' else 'matrix'
        current = batch[key].numpy()
        lengths = batch['n'].numpy()
        dist = batch['coord_dist'].numpy()
        for source_ids, order, arrays, _ in self.sources:
            wanted = np.flatnonzero(~reused)
            if not len(wanted):
                break
            sorted_ids = source_ids[order]
            positions = np.searchsorted(sorted_ids, ids[wanted])
            good = positions < len(sorted_ids)
            positions = positions.clip(0, len(sorted_ids) - 1)
            good &= sorted_ids[positions] == ids[wanted]
            rows, old_rows = wanted[good], order[positions[good]]
            for row, old in zip(rows, old_rows):
                n = int(lengths[row])
                if (int(arrays['n'][old]) != n or int(arrays['coord_dist'][old]) != int(dist[row])):
                    continue
                previous = arrays[key][old, :n] if key == 'node' else arrays[key][old, :n, :n]
                actual = current[row, :n] if key == 'node' else current[row, :n, :n]
                if not np.array_equal(previous, actual):
                    continue
                if key == 'node':
                    target[row, :n] = arrays['node_geom'][old, :n]
                    target[row, n:] = 0
                reused[row] = True
        return reused

    def close(self):
        for _, _, arrays, _ in self.sources:
            _close(arrays)
        self.sources.clear()


def add_moments(geometry, lengths, moments):
    maximum = geometry.shape[1]
    values = geometry[np.arange(maximum)[None] < lengths[:, None]].astype(np.float64)
    moments['count'] += len(values)
    moments['sum'] += values.sum(0)
    moments['squared_sum'] += np.square(values).sum(0)


def prepare_problem(root, scenario, cache_root, problem, split, device, reuse_roots,
                    old_data_root, test_lock_root=None):
    directory = cache_root / split / problem
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / 'metadata.json'
    access_root = test_permission(test_lock_root) if split == 'test' else root
    _, source_files = scenario.sources(problem, split, access_root)
    sources = feature_sources()
    if path.exists():
        record = read_json(path)
        if (record.get('data_root') != str(scenario.data_root)
                or record.get('source_files') != source_files or record.get('feature_sources') != sources):
            raise ValueError(f'{directory}: existing cache identity changed; preserve this cache')
        if record.get('complete'):
            _close(_open_arrays(directory, _validate_record(record, directory)))
            return record
        if record.get('rows_written', 0):
            raise ValueError(f'{directory}: incomplete cache retained; restart preparation with a new cache root')
    started = time.monotonic()
    dataset = load_dataset(problem, split=split, data_root=scenario.data_root, root=access_root)
    if dataset.size != SPLIT_ROWS[split]:
        raise ValueError(f'{problem}/{split}: expected {SPLIT_ROWS[split]} rows, got {dataset.size}')
    specs = _specs(dataset.size, int(dataset.lengths.max()), dataset.kind, dataset.d_in, dataset.K_p)
    record = dict(dataset.metadata, **scenario.header, directory=str(directory), cache_root=str(cache_root),
        cache_format=CACHE_FORMAT, cache_version=CACHE_VERSION, complete=False, rows_written=0,
        created_utc=utc(), seconds=0., feature_sources=sources, arrays=specs,
        cbits=list(dataset.cbits), problem_desc=list(dataset.problem_desc), mask=list(dataset.mask),
        geometry_calculation_dtype='float64', geometry_cache_dtype='float32',
        geometry_fields=list(FIELDS), geometry_scales=list(SCALES),
        geometry_moments=dict(count=0, sum=[0.] * len(FIELDS), squared_sum=[0.] * len(FIELDS)),
        test_lock_root=str(access_root) if split == 'test' else None,
        reuse_policy='source ID plus exact actual input/n/coord_dist equality; labels never copied',
        reused_input_rows=0, newly_computed_geometry_rows=0)
    write_json(path, record)
    arrays = _open_arrays(directory, specs, writable=True)
    reuse = ReusableInputs(problem, reuse_roots, old_data_root)
    moments = dict(count=0, sum=np.zeros(len(FIELDS), dtype=np.float64),
                   squared_sum=np.zeros(len(FIELDS), dtype=np.float64))
    last_report = 0.
    try:
        for start in range(0, dataset.size, COLLATE_CHUNK):
            stop = min(start + COLLATE_CHUNK, dataset.size)
            batch = build_batch(dataset, range(start, stop), device='cpu', include_geometry=False)
            key = 'node' if dataset.kind == 'coord' else 'matrix'
            width = batch[key].shape[1]
            if key == 'node':
                arrays[key][start:stop] = 0
                arrays[key][start:stop, :width] = batch[key].numpy()
                target = np.zeros((stop - start, width, len(FIELDS)), dtype=np.float32)
                reused = reuse.copy_geometry(batch, dataset.source_indices[start:stop], target)
                missing = np.flatnonzero(~reused)
                if len(missing):
                    selected = {k: v[missing] if torch.is_tensor(v) and v.ndim > 0
                                and v.shape[0] == stop - start and k not in ('pool_ids',) else v
                                for k, v in batch.items()}
                    local_arrays = {'node_geom': np.zeros((len(missing), width, len(FIELDS)), np.float32)}
                    unused = dict(count=0, sum=np.zeros(len(FIELDS)), squared_sum=np.zeros(len(FIELDS)))
                    _geometry(selected, local_arrays, 0, 'val', unused, device, problem)
                    target[missing] = local_arrays['node_geom']
                arrays['node_geom'][start:stop] = 0
                arrays['node_geom'][start:stop, :width] = target
                if split == 'train':
                    add_moments(target, batch['n'].numpy(), moments)
                record['reused_input_rows'] += int(reused.sum())
                record['newly_computed_geometry_rows'] += len(missing)
            else:
                arrays[key][start:stop] = 0
                arrays[key][start:stop, :width, :width] = batch[key].numpy()
            arrays['n'][start:stop] = dataset.lengths[start:stop]
            arrays['coord_dist'][start:stop] = batch['coord_dist'].numpy()
            arrays['costs'][start:stop] = dataset.raw_costs[start:stop]
            arrays['winner'][start:stop] = dataset.raw_winner[start:stop]
            if time.monotonic() - last_report >= 10 or stop == dataset.size:
                write_json(root / f'cache_progress_{problem}_{split}.json', dict(
                    phase='cache', problem=problem, split=split, done=stop, total=dataset.size,
                    reused=record['reused_input_rows'], seconds=time.monotonic() - started, updated_utc=utc()))
                print(f'[R84 prepare] {problem}/{split} {stop}/{dataset.size} reused={record["reused_input_rows"]}', flush=True)
                last_report = time.monotonic()
        for array in arrays.values():
            array.flush()
        record.update(complete=True, rows_written=dataset.size, completed_utc=utc(),
            seconds=time.monotonic() - started, storage_bytes=sum(a.nbytes for a in arrays.values()),
            geometry_moments=dict(count=moments['count'], sum=moments['sum'].tolist(),
                                  squared_sum=moments['squared_sum'].tolist()))
        write_json(path, record)
    finally:
        reuse.close()
        _close(arrays)
        del dataset
        gc.collect()
    return record


def finalize(root, data_root, cache_root, fresh_init=DEFAULT_FRESH_INIT, splits=('train', 'val')):
    root, data_root, cache_root = [Path(v).resolve() for v in (root, data_root, cache_root)]
    root.mkdir(parents=True, exist_ok=True)
    with (root / 'preparation_finalize.lock').open('a') as claim:
        fcntl.flock(claim, fcntl.LOCK_EX)
        scenario, tasks = _Scenario(data_root), {}
        manifest_path = root / 'source_cache_manifest.json'
        if manifest_path.exists():
            previous = read_json(manifest_path)
            if Path(previous['data_root']).resolve() != data_root:
                raise ValueError('Preparation belongs to another release')
            tasks.update(previous['splits'])
        for split in splits:
            tasks[split] = {}
            for problem in PROBLEMS:
                directory = cache_root / split / problem
                record = read_json(directory / 'metadata.json')
                _validate_record(record, directory)
                if (record['rows'] != SPLIT_ROWS[split] or record['data_root'] != str(data_root)
                        or record['scenario_id'] != scenario.header['scenario_id']):
                    raise ValueError('R84 cache row/scenario mismatch')
                tasks[split][problem] = record
        manifest = dict(experiment='R84', cache_format=CACHE_FORMAT, cache_version=CACHE_VERSION,
            data_root=str(data_root), cache_root=str(cache_root), **scenario.header,
            splits=tasks, complete=all(p in tasks.get('train', {}) for p in PROBLEMS), updated_utc=utc())
        write_json(manifest_path, manifest)
        if not all(p in tasks.get('train', {}) and p in tasks.get('val', {}) for p in PROBLEMS):
            return manifest
        ready_path = root / 'prepared.json'
        if ready_path.exists():
            ready = read_json(ready_path)
            # TEST appends feature metadata after checkpoint lock; TRAIN artifacts stay fixed.
            for name, digest in ready['artifact_sha256'].items():
                if file_hash(root / name) != digest:
                    raise ValueError(f'Locked preparation changed: {name}')
            return ready
        geometry = _stats(tasks['train'], feature_sources())
        write_json(root / 'geometry_stats.json', geometry)
        baselines = {}
        for problem in PROBLEMS:
            directory = cache_root / 'train' / problem
            costs = np.load(directory / 'costs.npy', mmap_mode='r', allow_pickle=False)
            winner = np.load(directory / 'winner.npy', mmap_mode='r', allow_pickle=False)
            try:
                baselines[problem] = fit_sbs(costs, tasks['train'][problem]['pool'], winner)
            finally:
                costs._mmap.close()
                winner._mmap.close()
        write_json(root / 'common_full_train_sbs.json', baselines)
        initial = torch.load(fresh_init, map_location='cpu', weights_only=False)
        if (initial.get('seed') != 2 or initial.get('initialization') != 'scratch'
                or parameter_hash(initial['trainable']) != initial.get('trainable_sha256')):
            raise ValueError('Only the locked fresh seed2 initialization may be reused')
        destination = root / 'identicalinit_seed2.pt'
        if not destination.exists():
            shutil.copyfile(fresh_init, destination)
        if file_hash(destination) != file_hash(fresh_init):
            raise ValueError('Fresh initialization copy differs')
        artifacts = ('geometry_stats.json', 'common_full_train_sbs.json', 'identicalinit_seed2.pt')
        ready = dict(experiment='R84', seed=2, created_utc=utc(), data_root=str(data_root),
            scenario_sha256=file_hash(data_root / 'scenario_manifest.json'), cache_source=str(cache_root),
            rows_per_problem=SPLIT_ROWS, trainable_sha256=initial['trainable_sha256'],
            model_params=initial['model_params'], fresh_initialization_source=str(Path(fresh_init).resolve()),
            artifact_sha256={name: file_hash(root / name) for name in artifacts},
            train_val_manifest_sha256=hashlib.sha256(str({s: tasks[s] for s in ('train', 'val')}).encode()).hexdigest(),
            fitted_on='new81000 TRAIN only; TEST features prepared after unique VAL lock')
        write_json(ready_path, ready)
        return ready


def prepare(args):
    args.root.mkdir(parents=True, exist_ok=True)
    scenario = _Scenario(args.data_root)
    if not 0 <= args.worker_rank < args.workers:
        raise ValueError('Invalid preparation worker assignment')
    for problem in PROBLEMS[args.worker_rank::args.workers]:
        for split in args.splits:
            directory = args.cache_root / split / problem
            directory.mkdir(parents=True, exist_ok=True)
            with (directory / 'prepare.lock').open('a') as claim:
                fcntl.flock(claim, fcntl.LOCK_EX)
                prepare_problem(args.root, scenario, args.cache_root, problem, split,
                    torch.device(args.device), args.reuse_cache_root, args.reuse_data_root, args.test_lock_root)
    if args.workers == 1:
        return finalize(args.root, args.data_root, args.cache_root, args.fresh_init, args.splits)


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('command', choices=('prepare', 'finalize', 'status'))
    result.add_argument('--root', type=Path, default=DEFAULT_PREPARED_ROOT)
    result.add_argument('--data-root', type=Path, default=DEFAULT_DATA_ROOT)
    result.add_argument('--cache-root', type=Path, required=True)
    result.add_argument('--device', default='cuda:0')
    result.add_argument('--splits', nargs='+', choices=tuple(SPLIT_ROWS), default=['train', 'val'])
    result.add_argument('--reuse-cache-root', action='append', type=Path, default=[])
    result.add_argument('--reuse-data-root', type=Path, default=OLD_DATA_ROOT)
    result.add_argument('--fresh-init', type=Path, default=DEFAULT_FRESH_INIT)
    result.add_argument('--test-lock-root', type=Path)
    result.add_argument('--worker-rank', type=int, default=0)
    result.add_argument('--workers', type=int, default=1)
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    for name in ('root', 'data_root', 'cache_root', 'reuse_data_root', 'fresh_init'):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    if args.command == 'status':
        path = args.root / 'prepared.json'
        print(path.read_text() if path.exists() else 'preparation pending')
        return
    torch.set_num_threads(2)
    if args.command == 'finalize':
        return finalize(args.root, args.data_root, args.cache_root, args.fresh_init, args.splits)
    if not args.reuse_cache_root:
        args.reuse_cache_root = [Path('/tmp/shiys_R76_R45A_single_reld_mt_cache')]
    return prepare(args)


if __name__ == '__main__':
    main()
