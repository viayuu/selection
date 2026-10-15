"""TRAIN-only R77 sampling and read-only views of the existing R76 caches."""

import argparse
import copy
import hashlib
from pathlib import Path

import numpy as np
import torch

from code.unified_selector.registry import PROBLEMS
from .multitask_probe import task_seed
from .r45_experiment import initialize as initialize_r45
from .r76_experiment import DATA_ROOT, file_hash, read_json, save_checkpoint, utc, write_json


V4 = Path(__file__).resolve().parent
DEFAULT_ROOT = V4 / 'runs/R77_sample_size_batch1024'
DEFAULT_SOURCE = V4 / 'runs/R76_R45A_single_reld_mt'
GROUPS = dict(A='A_10k_seed2', B='B_full_seed2')
FULL_ROWS, SUBSET_ROWS, HELDOUT_ROWS = 70114, 10000, 3895


def proportional_counts(counts, total):
    counts = np.asarray(counts, dtype=np.int64)
    if counts.ndim != 1 or not len(counts) or np.any(counts <= 0) or not 0 <= total <= counts.sum():
        raise ValueError('Invalid proportional allocation')
    numerator = counts * int(total)
    quota = numerator // counts.sum()
    order = np.argsort(-(numerator % counts.sum()), kind='stable')
    quota[order[:int(total - quota.sum())]] += 1
    if quota.sum() != total or np.any(quota > counts):
        raise ValueError('Stratum quota exceeds its eligible population')
    return quota


def hierarchical_sample(metadata, size=SUBSET_ROWS, seed=2):
    """Allocate at each existing input-only level, then sample leaf rows uniformly."""
    if 'scale' not in metadata:
        raise ValueError('Sampling requires real input scale metadata')
    fields = ['scale'] + [key for key in ('distribution', 'capacity_type') if key in metadata]
    arrays = {key: np.asarray(metadata[key]) for key in fields}
    n = len(arrays['scale'])
    if not 0 < size <= n or any(value.shape != (n,) for value in arrays.values()):
        raise ValueError('Input metadata must be aligned with all eligible TRAIN rows')
    rng, selected, strata = np.random.default_rng(seed), [], []

    def visit(rows, level, quota, labels):
        if level == len(fields):
            if quota:
                selected.append(rng.choice(rows, size=int(quota), replace=False))
            strata.append(dict(labels, eligible=int(len(rows)), selected=int(quota)))
            return
        key = fields[level]
        values, inverse, counts = np.unique(arrays[key][rows], return_inverse=True, return_counts=True)
        quotas = proportional_counts(counts, quota)
        for index, (value, take) in enumerate(zip(values, quotas)):
            value = value.item() if isinstance(value, np.generic) else value
            visit(rows[inverse == index], level + 1, int(take), dict(labels, **{key: value}))

    visit(np.arange(n, dtype=np.int64), 0, int(size), {})
    indices = np.sort(np.concatenate(selected)).astype(np.int64)
    if len(indices) != size or len(np.unique(indices)) != size:
        raise ValueError('Sampling must retain exactly the requested distinct source rows')
    return indices, dict(fields=fields, strata=strata, seed=int(seed), eligible=n, selected=int(size),
        rule='hierarchical proportional largest-remainder quotas; uniform leaf sampling without replacement',
        winner_used=False)


class IndexedCache:
    """An arm's local row space; gathers still use unmodified R76 cached inputs."""

    def __init__(self, source, indices=None):
        self.source = source
        self.indices = None if indices is None else np.asarray(indices, dtype=np.int64)
        if self.indices is not None and (self.indices.ndim != 1 or not len(self.indices)
                or len(np.unique(self.indices)) != len(self.indices)
                or self.indices.min() < 0 or self.indices.max() >= source.size):
            raise ValueError('An indexed cache needs distinct eligible source rows')
        self.size = source.size if self.indices is None else len(self.indices)
        self.problem, self.split, self.kind = source.problem, source.split, source.kind
        self.pool, self.pool_ids, self.metadata = source.pool, source.pool_ids, source.metadata
        self.lengths = source.lengths if self.indices is None else source.lengths[self.indices]
        self.raw = dict(pool=self.pool, pool_ids=self.pool_ids,
            **{key: source.raw[key] if self.indices is None else source.raw[key][self.indices]
               for key in ('costs', 'winner')})

    def source_rows(self, rows):
        rows = np.asarray(rows, dtype=np.int64)
        if rows.ndim != 1 or not len(rows) or rows.min() < 0 or rows.max() >= self.size:
            raise IndexError('Batch rows are outside this arm\'s TRAIN population')
        return rows if self.indices is None else self.indices[rows]

    def batch(self, rows, device='cpu'):
        if torch.is_tensor(rows):
            rows = rows.cpu().numpy()
        return self.source.batch(self.source_rows(rows), device=device)

    def close(self):
        self.raw, self.lengths = {}, None
        self.source.close()


def parameter_hash(parameters):
    digest = hashlib.sha256()
    for name, tensor in sorted(parameters.items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode('ascii'))
        digest.update(str((tuple(value.shape), value.dtype)).encode('ascii'))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def fit_sbs(costs, pool, winner):
    costs = np.asarray(costs, dtype=np.float64)
    mean = costs.mean(axis=0, dtype=np.float64)
    regret = ((costs - costs.min(axis=1, keepdims=True)) / costs.min(axis=1, keepdims=True)).mean(axis=0)
    slot = int(mean.argmin())
    return dict(sbs_solver=pool[slot], sbs_slot=slot, selection_split='train',
        selection_rule='minimum mean original FP64 cost on this arm\'s TRAIN rows',
        mean_cost_by_solver=dict(zip(pool, mean.tolist())),
        mean_relative_regret_by_solver=dict(zip(pool, regret.tolist())),
        minimum_mean_relative_regret_solver=pool[int(regret.argmin())],
        majority_winner_solver=pool[int(np.bincount(winner, minlength=len(pool)).argmax())],
        train_rows=len(costs))


def subset_geometry(cache_root, manifest, indices, template):
    sums = np.zeros(len(template['mean']), dtype=np.float64)
    squares, counts = np.zeros_like(sums), {}
    for problem in PROBLEMS:
        record = manifest['splits']['train'][problem]
        if record['kind'] != 'coord':
            continue
        directory = cache_root / 'train' / problem
        geometry = np.load(directory / 'node_geom.npy', mmap_mode='r', allow_pickle=False)
        lengths = np.load(directory / 'n.npy', mmap_mode='r', allow_pickle=False)
        count = 0
        try:
            for start in range(0, len(indices[problem]), 256):
                rows = indices[problem][start:start + 256]
                n = lengths[rows]
                maximum = int(n.max())
                values = geometry[:, :maximum][rows]
                valid = values[np.arange(maximum)[None, :] < n[:, None]].astype(np.float64)
                count += len(valid)
                sums += valid.sum(axis=0)
                squares += np.square(valid).sum(axis=0)
        finally:
            geometry._mmap.close()
            lengths._mmap.close()
        counts[problem] = count
        print(f'[R77 prepare] A geometry {problem}: {count} valid TRAIN nodes', flush=True)
    count = sum(counts.values())
    mean = sums / count
    std = np.maximum(np.sqrt(np.maximum(squares / count - np.square(mean), 0.)), 1e-6)
    result = copy.deepcopy(template)
    result.update(mean=mean.tolist(), std=std.tolist(), sum=sums.tolist(), squared_sum=squares.tolist(),
        fitted_on='A: selected 10000 TRAIN rows/problem, true valid coordinate nodes only',
        valid_train_nodes=count, valid_nodes_by_problem=counts)
    return result


def prepare(root=DEFAULT_ROOT, data_root=DATA_ROOT, source_root=DEFAULT_SOURCE, cache_root=None):
    root, data_root, source_root = (Path(path).expanduser().resolve() for path in (root, data_root, source_root))
    cache_root = Path(cache_root or source_root / 'cache').expanduser().resolve()
    ready = root / 'prepared.json'
    if ready.exists():
        record = read_json(ready)
        if record['data_root'] != str(data_root) or record['source_root'] != str(source_root):
            raise ValueError('Existing R77 preparation belongs to a different data/source root')
        for path, digest in record['artifact_sha256'].items():
            if file_hash(root / path) != digest:
                raise ValueError(f'Prepared artifact changed: {path}')
        print('[R77 prepare] Reusing locked TRAIN-only preparation', flush=True)
        return record
    root.mkdir(parents=True, exist_ok=True)
    manifest = read_json(source_root / 'cache_manifest.json')
    if Path(manifest['data_root']).resolve() != data_root or not manifest.get('complete'):
        raise ValueError('R77 requires the completed R76 cache for this exact release')
    for split, size in (('train', FULL_ROWS), ('val', HELDOUT_ROWS), ('test', HELDOUT_ROWS)):
        if set(manifest['splits'][split]) != set(PROBLEMS) or any(
                manifest['splits'][split][p]['rows'] != size for p in PROBLEMS):
            raise ValueError(f'Keep all 18 original {split} populations unchanged')
    indices, sampling, own_sbs = {}, {}, {}
    artifacts = {}
    directory = root / 'indices/A'
    directory.mkdir(parents=True, exist_ok=True)
    for problem in PROBLEMS:
        path = data_root / (problem + 'train') / 'input_metadata.npz'
        with np.load(path, allow_pickle=False) as saved:
            metadata = {key: saved[key] for key in ('scale', 'distribution', 'capacity_type') if key in saved}
            rows, receipt = hierarchical_sample(metadata, seed=task_seed(2, problem))
            if receipt['eligible'] != FULL_ROWS:
                raise ValueError(f'{problem}: sample from all {FULL_ROWS} TRAIN inputs, not a winner-filtered pool')
            receipt['input_metadata_sha256'] = file_hash(path)
            if 'source_index' in saved:
                receipt['selected_source_index_sha256'] = hashlib.sha256(saved['source_index'][rows].tobytes()).hexdigest()
        indices[problem], sampling[problem] = rows, receipt
        index_path = directory / (problem + '.npy')
        np.save(index_path, rows, allow_pickle=False)
        artifacts[str(index_path.relative_to(root))] = file_hash(index_path)
        cache_directory = cache_root / 'train' / problem
        cached = read_json(cache_directory / 'metadata.json')
        if (not cached.get('complete') or cached['rows'] != FULL_ROWS or cached['problem'] != problem
                or Path(cached['data_root']).resolve() != data_root
                or cached['pool'] != manifest['splits']['train'][problem]['pool']):
            raise ValueError(f'{problem}: incompatible existing TRAIN cache')
        costs = np.load(cache_directory / 'costs.npy', mmap_mode='r', allow_pickle=False)
        winner = np.load(cache_directory / 'winner.npy', mmap_mode='r', allow_pickle=False)
        try:
            own_sbs[problem] = fit_sbs(costs[rows], cached['pool'], winner[rows])
        finally:
            costs._mmap.close()
            winner._mmap.close()
        print(f'[R77 prepare] {problem}: A {len(rows)} subset of B {FULL_ROWS}; input strata only', flush=True)
    common_sbs = read_json(data_root / 'baselines.json')
    geometry_b = read_json(source_root / 'geometry_stats.json')
    geometry_a = subset_geometry(cache_root, manifest, indices, geometry_b)
    for group, geometry, baselines in (('A', geometry_a, own_sbs), ('B', geometry_b, common_sbs)):
        arm = root / GROUPS[group]
        arm.mkdir(exist_ok=True)
        geometry = dict(geometry, r77_group=group, train_rows_per_problem=SUBSET_ROWS if group == 'A' else FULL_ROWS)
        write_json(arm / 'geometry_stats.json', geometry)
        write_json(arm / 'own_train_sbs.json', baselines)
        for name in ('geometry_stats.json', 'own_train_sbs.json'):
            path = arm / name
            artifacts[str(path.relative_to(root))] = file_hash(path)
    # Persist parameters, not geometry buffers: both arms load these exact trainable values.
    model, params = initialize_r45('A', geometry_a, device='cpu', seed=2)
    parameters = {name: tensor.detach().cpu().clone() for name, tensor in model.named_parameters()}
    digest = parameter_hash(parameters)
    params = dict(params, sdpa=True)
    save_checkpoint(root / 'identicalinit_seed2.pt', dict(trainable=parameters, model_params=params,
        seed=2, initialization='scratch', trainable_sha256=digest, geometry_buffers_included=False))
    del model, parameters
    for name, value in (('source_cache_manifest.json', manifest), ('sampling.json', sampling),
                        ('common_full_train_sbs.json', common_sbs)):
        write_json(root / name, value)
        artifacts[name] = file_hash(root / name)
    artifacts['identicalinit_seed2.pt'] = file_hash(root / 'identicalinit_seed2.pt')
    record = dict(experiment='R77', prepared_utc=utc(), data_root=str(data_root), source_root=str(source_root),
        cache_source=str(cache_root), seed=2, groups=GROUPS, sizes=dict(A=SUBSET_ROWS, B=FULL_ROWS),
        heldout_rows_per_problem=HELDOUT_ROWS, trainable_sha256=digest, model_params=params,
        artifact_sha256=artifacts, test_inputs_read=False, costs_recomputed=False,
        sampling='scale > existing distribution > existing capacity_type; never winner',
        checkpoint_sbs='common FULL 70114 TRAIN selected SBS; own TRAIN SBS is reporting-only')
    write_json(ready, record)
    write_json(root / 'initialization.json', dict(seed=2, identicalinit=True,
        common_trainable_sha256=digest, trainable_parameters_only=True,
        geometry_normalization='separate TRAIN-only mean/std per arm', historical_checkpoint_loaded=False))
    print('[R77 prepare] Ready; no GPU training or TEST input reads', flush=True)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    parser.add_argument('--data-root', type=Path, default=DATA_ROOT)
    parser.add_argument('--source-root', type=Path, default=DEFAULT_SOURCE)
    parser.add_argument('--cache-root', type=Path)
    args = parser.parse_args()
    torch.set_num_threads(2)
    prepare(args.root, args.data_root, args.source_root, args.cache_root)


if __name__ == '__main__':
    main()
