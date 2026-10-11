"""Read-only, CPU/mmap expansion inputs with two observed ReLD cost columns."""

import csv
import json
import os
import pickle
import tempfile
from pathlib import Path

import numpy as np
import torch

from code.unified_selector.registry import (P2I, POOLS, PROBLEMS, constraint_bits,
                                            problem_descriptor, problem_to_pool_mask)


SOURCE_ROOT = Path('/public/home/shiys') / '\u8bad\u7ec3\u6570\u636e_\u6bcf\u7c7b10\u4e07'
CACHE_ROOT = Path(__file__).resolve().parent / 'runs/R60_expanded_partial_supervision/data_cache'
PAIR_SOLVERS = ('RELD_MOEL', 'RELD_MTL')
MVRP_PROBLEMS = tuple(PROBLEMS[3:])
RECIPE_RECORD = '\u6807\u7b7e\u914d\u65b9\u4e0e\u6821\u51c6\u8bb0\u5f55.csv'
PROGRESS_RECORD = '\u6807\u7b7e\u8fd0\u884c\u8fdb\u5ea6.csv'
GENERATION_RECORD = '\u751f\u6210\u4fe1\u606f.json'
FIELDS = dict(problem='\u95ee\u9898', solver='\u65b9\u6cd5',
              added='\u65b0\u589e\u4ee3\u4ef7\u5df2\u4fdd\u5b58',
              calibration='\u6821\u51c6\u72b6\u6001', calibrated='\u6821\u51c6\u5b9e\u4f8b\u6570',
              source_count='\u6570\u636e\u5b9e\u4f8b\u6570', prefix='\u539f\u6709\u6807\u7b7e\u6570',
              new_count='\u65b0\u589e\u6807\u7b7e\u6570', status='\u72b6\u6001')
CALIBRATED = '\u5206\u5c42\u6837\u672c\u56de\u653e\u4e00\u81f4'
COMPLETE = '\u5b8c\u6210'
LIMITATIONS = [
    'Calibration agreement covers only 4 or 8 recorded examples per column, not full historical proof.',
    'Legacy signed-load scenario; this is not a repaired common classic-backhaul benchmark.',
    'Existing data-only no-overlap report predates the recorded VRPLTW repair; no new overlap audit.',
    'Only two observed costs are available here; no full-pool winner, costs or oracle are inferred.',
]


def _recorded_pairs(source_root, problems, source_count, prefix_count):
    tables = []
    for filename in (RECIPE_RECORD, PROGRESS_RECORD):
        with (source_root / filename).open(encoding='utf-8-sig', newline='') as stream:
            rows = csv.DictReader(stream)
            table = {}
            for row in rows:
                key = row[FIELDS['problem']], row[FIELDS['solver']]
                if key[0] in problems and key[1] in PAIR_SOLVERS:
                    if key in table:
                        raise ValueError('Duplicate recorded deployment: ' + str(key))
                    table[key] = row
            tables.append(table)
    recipe, progress = tables
    result = {}
    for problem in problems:
        result[problem] = []
        for solver in PAIR_SOLVERS:
            r, p = recipe[(problem, solver)], progress[(problem, solver)]
            if (int(r[FIELDS['added']]) != source_count - prefix_count or
                    r[FIELDS['calibration']] != CALIBRATED or
                    int(p[FIELDS['source_count']]) != source_count or
                    int(p[FIELDS['prefix']]) != prefix_count or
                    int(p[FIELDS['new_count']]) != source_count - prefix_count or
                    p[FIELDS['status']] != COMPLETE):
                raise ValueError('Incomplete expansion pair column: ' + str((problem, solver)))
            result[problem].append(dict(solver=solver, recorded_additions=source_count - prefix_count,
                calibration_examples=int(r[FIELDS['calibrated']]), calibration_status=r[FIELDS['calibration']]))
    return result


def _file_identity(path):
    stat = path.stat()
    return dict(path=str(path.resolve()), bytes=stat.st_size, mtime_ns=stat.st_mtime_ns)


def _node_input(instance, bits):
    """Exactly the current UnifiedProblemDataset MVRP field layout, without labels."""
    xy = np.asarray(instance['node_xy'], dtype=np.float32)
    demand = np.asarray(instance['node_demand'], dtype=np.float32)
    if xy.ndim != 2 or xy.shape[1] != 2 or demand.shape != (len(xy),) or not len(xy):
        raise ValueError('Invalid MVRP coordinates/demand shapes')
    node = np.zeros((len(xy) + 1, 8), dtype=np.float32)
    node[0, :2] = np.asarray(instance['depot_xy'], dtype=np.float32).reshape(1, 2)[0]
    node[0, 3] = 1.
    node[1:, :2], node[1:, 2] = xy, demand
    if bits[3]:
        node[1:, 4] = float(instance.get('route_limit', 0.))
    if bits[4] and instance.get('service_time') is not None:
        for column, name in ((5, 'service_time'), (6, 'tw_start'), (7, 'tw_end')):
            node[1:, column] = np.asarray(instance[name], dtype=np.float32)
    if not np.isfinite(node).all():
        raise ValueError('Nonfinite original MVRP input')
    return node


def _build_cache(directory, context):
    problem = context['problem']
    source_count, prefix = context['source_count'], context['prefix_count']
    columns = []
    for identity in context['sources'][1:]:
        values = np.loadtxt(identity['path'], delimiter=',', dtype=np.float64, ndmin=2)
        if (values.shape != (source_count, 2) or
                not np.array_equal(values[:, 0], np.arange(source_count)) or
                not np.isfinite(values[:, 1]).all() or not (values[:, 1] > 0).all()):
            raise ValueError('Pair column count/order/finite-cost check failed: ' + identity['path'])
        columns.append(values[prefix:, 1].copy())
    pair_costs = np.stack(columns, axis=1)
    with Path(context['sources'][0]['path']).open('rb') as stream:
        instances = pickle.load(stream)
    if not isinstance(instances, (list, tuple)) or len(instances) != source_count:
        raise ValueError('Expansion input count must match named pair columns')
    nodes = np.fromiter((len(instances[i]['node_xy']) + 1 for i in range(prefix, source_count)),
                        dtype=np.int32, count=source_count - prefix)
    offsets = np.concatenate((np.array([0], dtype=np.int64), np.cumsum(nodes, dtype=np.int64)))
    with tempfile.TemporaryDirectory(prefix='.' + problem + '-', dir=directory.parent) as temporary:
        staging = Path(temporary)
        values = np.lib.format.open_memmap(staging / 'node_values.npy', mode='w+', dtype=np.float32,
                                          shape=(int(offsets[-1]), 8))
        bits = constraint_bits(problem)
        for row, source_index in enumerate(range(prefix, source_count)):
            values[offsets[row]:offsets[row + 1]] = _node_input(instances[source_index], bits)
        values.flush()
        del values, instances
        arrays = dict(nodes=nodes, offsets=offsets, pair_costs=pair_costs,
                      source_indices=np.arange(prefix, source_count, dtype=np.int64))
        for name, array in arrays.items():
            np.save(staging / (name + '.npy'), array, allow_pickle=False)
        if context['sources'] != [_file_identity(Path(item['path'])) for item in context['sources']]:
            raise ValueError('Source inputs/costs changed during cache preparation')
        manifest = dict(context, count=len(nodes), total_nodes=int(offsets[-1]),
                        node_min=int(nodes.min()), node_max=int(nodes.max()),
                        exact_pair_ties=int(np.count_nonzero(pair_costs[:, 0] == pair_costs[:, 1])),
                        cost_dtype='float64', node_dtype='float32', limitations=LIMITATIONS)
        (staging / 'manifest.json').write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')
        os.replace(staging, directory)


class ExpandedPairDataset:
    """Ragged inputs stay read-only on disk; only the requested batch moves to device."""

    def __init__(self, directory):
        self.directory = Path(directory)
        self.manifest = json.loads((self.directory / 'manifest.json').read_text())
        self.problem = self.manifest['problem']
        self.pool = tuple(POOLS[self.problem])
        self.pair_solvers = PAIR_SOLVERS
        if self.manifest['pair_solvers'] != list(PAIR_SOLVERS) or self.manifest['pool'] != list(self.pool):
            raise ValueError('Cache solver name/order does not match the registry')
        for name in ('node_values', 'nodes', 'offsets', 'pair_costs', 'source_indices'):
            setattr(self, name, np.load(self.directory / (name + '.npy'), mmap_mode='r', allow_pickle=False))
        self.size = self.manifest['count']
        self.lengths = self.nodes
        if (self.nodes.shape != (self.size,) or self.source_indices.shape != (self.size,) or
                self.offsets.shape != (self.size + 1,) or self.pair_costs.shape != (self.size, 2) or
                self.pair_costs.dtype != np.float64 or
                self.node_values.shape != (self.manifest['total_nodes'], 8)):
            raise ValueError('Incomplete expansion mmap arrays')

    def __len__(self):
        return self.size

    def gather(self, indices, device='cpu'):
        if torch.is_tensor(indices):
            if indices.device.type != 'cpu':
                raise ValueError('Expansion row indices must remain on CPU')
            indices = indices.numpy()
        indices = np.asarray(indices)
        if (indices.ndim != 1 or not len(indices) or not np.issubdtype(indices.dtype, np.integer) or
                indices.min() < 0 or indices.max() >= self.size):
            raise ValueError('Expected a nonempty vector of valid expansion row indices')
        lengths = np.asarray(self.nodes[indices], dtype=np.int64)
        batch_size, max_n = len(indices), int(lengths.max())
        node = np.zeros((batch_size, max_n, 8), dtype=np.float32)
        for target, row in enumerate(indices):
            node[target, :lengths[target]] = self.node_values[self.offsets[row]:self.offsets[row + 1]]
        pool_ids, mask = problem_to_pool_mask(self.problem)
        batch = dict(problem_id=P2I[self.problem], kind='coord', d_in=8,
            node=torch.from_numpy(node), node_mask=torch.from_numpy(np.arange(max_n)[None] < lengths[:, None]),
            n=torch.from_numpy(lengths), pool_ids=torch.tensor(pool_ids, dtype=torch.long),
            mask=torch.tensor(mask, dtype=torch.float32)[None].expand(batch_size, -1).clone(),
            cbits=torch.tensor(constraint_bits(self.problem), dtype=torch.float32)[None].expand(batch_size, -1).clone(),
            problem_desc=torch.tensor(problem_descriptor(self.problem), dtype=torch.float32)[None].expand(batch_size, -1).clone(),
            coord_dist=torch.full((batch_size,), 4, dtype=torch.long))
        batch = {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}
        costs = torch.from_numpy(np.array(self.pair_costs[indices], dtype=np.float64, copy=True)).to(device)
        return batch, costs


def prepare_expanded(cache_root=CACHE_ROOT, source_root=SOURCE_ROOT, *, problems=MVRP_PROBLEMS,
                     source_count=100000, prefix_count=10000):
    """One-time conversion, one source pickle at a time; reuse complete mmap caches."""
    cache_root, source_root = Path(cache_root), Path(source_root)
    problems = tuple(problems)
    if not problems or len(set(problems)) != len(problems) or any(p not in MVRP_PROBLEMS for p in problems):
        raise ValueError('Expansion is restricted to the original 15 MVRP tasks')
    if not 0 < prefix_count < source_count:
        raise ValueError('Expected an excluded original prefix and nonempty expansion')
    records = _recorded_pairs(source_root, problems, source_count, prefix_count)
    cache_root.mkdir(parents=True, exist_ok=True)
    datasets = {}
    for problem in problems:
        if len(POOLS[problem]) != 7 or any(s not in POOLS[problem] for s in PAIR_SOLVERS):
            raise ValueError('R60 must retain the complete seven-solver MVRP registry pool')
        source = source_root / (problem + 'train')
        paths = [source / 'dataset.pkl'] + [source / 'results' / ('result_' + s + '.txt') for s in PAIR_SOLVERS]
        generation = source / GENERATION_RECORD
        repair_info = json.loads(generation.read_text()).get('data_repairs', []) if generation.exists() else []
        context = dict(version=1, problem=problem, source_count=source_count, prefix_count=prefix_count,
            pair_solvers=list(PAIR_SOLVERS), pool=list(POOLS[problem]),
            sources=[_file_identity(path) for path in paths], recorded_calibration=records[problem],
            recorded_input_repairs=repair_info, coord_dist='unknown=4; never inferred from source index')
        directory = cache_root / problem
        if directory.exists():
            manifest = json.loads((directory / 'manifest.json').read_text())
            if any(manifest.get(key) != value for key, value in context.items()):
                raise ValueError('Expansion cache inputs/protocol changed; use a new cache root: ' + problem)
        else:
            _build_cache(directory, context)
        datasets[problem] = ExpandedPairDataset(directory)
        print('[R60 expansion] ' + problem + ': ' + str(len(datasets[problem])) + ' read-only rows', flush=True)
    return datasets


def paired_auxiliary_schedule(expanded, original_nodes, epochs=40, epoch_size=10000, seed=2):
    """B traverses expansions; A replays original rows with the exact same true sizes."""
    plans = {arm: {} for arm in ('A', 'B')}
    for problem, dataset in expanded.items():
        old = original_nodes[problem]
        if torch.is_tensor(old):
            old = old.cpu().numpy()
        old = np.asarray(old)
        if old.ndim != 1 or len(old) != epoch_size or dataset.size % epoch_size:
            raise ValueError('Auxiliary schedule requires complete equal-sized original epochs')
        buckets = {int(n): np.flatnonzero(old == n) for n in np.unique(dataset.nodes)}
        if any(not len(rows) for rows in buckets.values()):
            raise ValueError('Expanded true size has no original-TRAIN replay bucket: ' + problem)
        rng_b = np.random.default_rng(np.random.SeedSequence([seed, 60, P2I[problem], 0]))
        rng_a = np.random.default_rng(np.random.SeedSequence([seed, 60, P2I[problem], 1]))
        plans['A'][problem], plans['B'][problem] = [], []
        order, cursor = None, dataset.size
        for _ in range(epochs):
            if cursor == dataset.size:
                order, cursor = rng_b.permutation(dataset.size), 0
            indices_b = order[cursor:cursor + epoch_size].copy()
            cursor += epoch_size
            sizes = dataset.nodes[indices_b]
            indices_a = np.empty(epoch_size, dtype=np.int64)
            for size, rows in buckets.items():
                positions = np.flatnonzero(sizes == size)
                indices_a[positions] = rng_a.choice(rows, size=len(positions), replace=True)
            plans['A'][problem].append(indices_a)
            plans['B'][problem].append(indices_b)
    return plans
