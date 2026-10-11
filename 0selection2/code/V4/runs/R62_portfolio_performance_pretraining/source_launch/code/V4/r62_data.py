"""Frozen auxiliary portfolio observations and nested, size-matched CPU datasets."""

import csv
import hashlib
import io
import json
import os
import pickle
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

from code.unified_selector.registry import (GLOBAL_SOLVERS, P2I, POOLS, PROBLEMS,
                                            constraint_bits, problem_descriptor,
                                            problem_to_pool_mask)
from .r60_data import (CACHE_ROOT as EXPANDED_ROOT, FIELDS, GENERATION_RECORD,
                       PROGRESS_RECORD, RECIPE_RECORD, SOURCE_ROOT, _file_identity,
                       _node_input)


CACHE_ROOT = Path(__file__).resolve().parent / 'runs/R62_portfolio_performance_pretraining/data_cache'
PRODUCER_ROOT = Path('/public/home/shiys/easynco_v3_bridge/exports/train_100k_20261002')
LIMITATIONS = [
    'Auxiliary expansion deployments, NOT legacy labels or a common-feasible-domain relabeling.',
    'Historical calibration covers only 4/8 samples; matches and differences do not certify equivalence.',
    'Recorded needs-investigation, sampling, budget and TW/backhaul differences remain unresolved.',
    'Existing no-overlap report predates the recorded VRPLTW repair; no new overlap audit.',
    'Missing observations are NaN plus false masks; no full-pool winner, oracle or rank is inferred.',
    'Snapshot ignores subsequent queue arrivals; only TRAIN source IDs outside the original prefix.',
]


def _json(path):
    return json.loads(Path(path).read_text())


def _write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp-' + str(os.getpid()))
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    os.replace(temporary, path)


def _records(source_root, filename):
    with (source_root / filename).open(encoding='utf-8-sig', newline='') as stream:
        return {tuple(row[FIELDS[k]] for k in ('problem', 'solver')): row for row in csv.DictReader(stream)}


def _freeze_inventory(root, source_root, producer_root, problems, source_count, prefix, subset_count, seed):
    path = root / 'inventory.json'
    config = dict(problems=list(problems), source_root=str(source_root.resolve()),
                  producer_root=str(producer_root.resolve()), source_count=source_count,
                  prefix_count=prefix, subset_count=subset_count, seed=seed,
                  global_solvers=list(GLOBAL_SOLVERS))
    if path.exists():
        inventory = _json(path)
        if inventory['config'] != config:
            raise ValueError('Frozen portfolio configuration changed; use a new cache root')
        return inventory
    frozen = time.time_ns()
    recipe, progress = _records(source_root, RECIPE_RECORD), _records(source_root, PROGRESS_RECORD)
    inventory = dict(version=1, frozen_ns=frozen, config=config, tasks={}, limitations=LIMITATIONS)
    for problem in problems:
        source = source_root / (problem + 'train')
        generation = _json(source / GENERATION_RECORD)
        if (generation['problem'] != problem or generation['target_instances'] != source_count or
                generation['original_instances'] != prefix or generation['methods'] != list(POOLS[problem])):
            raise ValueError('Expansion input/pool identity mismatch: ' + problem)
        task = dict(generation=generation, generation_file=_file_identity(source / GENERATION_RECORD),
                    input_file=_file_identity(source / 'dataset.pkl'), columns=[])
        for solver in POOLS[problem]:
            key = problem, solver
            deployment = producer_root / 'labels' / problem / solver
            column = dict(solver=solver, solver_id=GLOBAL_SOLVERS.index(solver),
                          recipe_record=recipe.get(key), progress_record=progress.get(key),
                          mode='missing', ignored=[])
            try:
                profile = _json(deployment / 'profile.json')
                profile_hash = hashlib.sha256(json.dumps(profile, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
                if profile.get('problem') != problem or profile.get('method') != solver:
                    raise ValueError('Profile problem/method mismatch')
                column.update(profile=profile, profile_sha256=profile_hash)
                complete_path = deployment / 'complete.json'
                table = source / 'results' / ('result_' + solver + '.txt')
                if complete_path.exists() and complete_path.stat().st_mtime_ns <= frozen:
                    complete = _json(complete_path)
                    if (complete['problem'] != problem or complete['method'] != solver or
                            complete['profile_sha256'] != profile_hash or complete['instances'] != source_count):
                        raise ValueError('Published column identity mismatch')
                    column.update(mode='table', table=_file_identity(table), complete=complete)
                else:
                    metas = sorted(p for p in (deployment / 'shards').glob('*.meta.json')
                                   if p.stat().st_mtime_ns <= frozen)
                    column.update(mode='shards', metas=[_file_identity(p) for p in metas])
            except (OSError, ValueError, KeyError, TypeError) as error:
                column['ignored'].append(str(error))
            task['columns'].append(column)
        inventory['tasks'][problem] = task
    _write_json(path, inventory)
    return inventory


def _read_column(column, generation, problem, prefix, source_count):
    values = np.full(source_count - prefix, np.nan, dtype=np.float64)
    ignored = list(column['ignored'])
    links = []
    if column['mode'] == 'table':
        try:
            raw = Path(column['table']['path']).read_bytes()
            if hashlib.sha256(raw).hexdigest() != column['complete']['result_sha256']:
                raise ValueError('Published cost table no longer matches its saved identity')
            table = np.loadtxt(io.BytesIO(raw), delimiter=',', dtype=np.float64, ndmin=2)
            if table.shape != (source_count, 2) or not np.array_equal(table[:, 0], np.arange(source_count)):
                raise ValueError('Published cost row IDs/count do not match inputs')
            values[:] = table[prefix:, 1]
        except (OSError, ValueError, KeyError) as error:
            ignored.append(str(error))
    elif column['mode'] == 'shards':
        inputs = {(int(s['start_index']), int(s['count'])): s for s in generation['shards']}
        seen = np.zeros(len(values), dtype=bool)
        for identity in column['metas']:
            try:
                if _file_identity(Path(identity['path'])) != identity:
                    raise ValueError('Frozen shard metadata changed')
                meta = _json(identity['path'])
                start, count = int(meta['start_index']), int(meta['count'])
                shard = inputs.get((start, count))
                if (shard is None or start < prefix or start + count > source_count or
                        meta['problem'] != problem or meta['method'] != column['solver'] or
                        meta['input_sha256'] != shard['sha256'] or
                        meta['profile_sha256'] != column['profile_sha256']):
                    raise ValueError('Saved observation has no matching input/row/deployment identity')
                result = _json(meta['result_path'])
                keys = ('problem', 'method', 'start_index', 'count', 'input_sha256', 'profile_sha256')
                if any(result.get(key) != meta[key] for key in keys):
                    raise ValueError('Saved result and metadata identities differ')
                if result.get('audit', {}).get('feasible') is False:
                    raise ValueError('Producer explicitly marked saved observations failed')
                costs = np.asarray(result['costs'], dtype=np.float64)
                if costs.shape != (count,):
                    raise ValueError('Saved cost row count differs from input shard')
                positions = slice(start - prefix, start + count - prefix)
                if seen[positions].any():
                    raise ValueError('Duplicate observed source IDs')
                values[positions] = costs
                seen[positions] = True
                links.append(dict(meta=identity, result_path=meta['result_path'],
                                  result_sha256=meta.get('result_sha256'), start_index=start, count=count,
                                  input_sha256=meta['input_sha256']))
            except (OSError, ValueError, KeyError, TypeError) as error:
                ignored.append(dict(meta=identity['path'], reason=str(error)))
    nonfinite = int((~np.isfinite(values)).sum())
    values[~np.isfinite(values)] = np.nan
    return values, dict(column, observed=int(np.isfinite(values).sum()), missing_or_nonfinite=nonfinite,
                        ignored=ignored, result_links=links)


def _coverage(costs, observed):
    comparisons = np.zeros(len(costs), dtype=np.int32)
    for i in range(costs.shape[1]):
        for j in range(i + 1, costs.shape[1]):
            comparisons += observed[:, i] & observed[:, j] & (costs[:, i] != costs[:, j])
    return dict(rows=len(costs), observed_cells=int(observed.sum()), missing_cells=int((~observed).sum()),
                rows_at_least_two_observed=int((observed.sum(1) >= 2).sum()),
                comparison_bearing_rows=int((comparisons > 0).sum()),
                nontied_observed_pairs=int(comparisons.sum()), observed_per_column=observed.sum(0).tolist())


def _freeze_bank(root, inventory):
    prefix, count = inventory['config']['prefix_count'], inventory['config']['source_count']
    for problem, task in inventory['tasks'].items():
        directory = root / 'bank' / problem
        if directory.exists():
            continue
        directory.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='.' + problem, dir=directory.parent) as temporary:
            staging = Path(temporary)
            costs, columns = [], []
            for column in task['columns']:
                values, record = _read_column(column, task['generation'], problem, prefix, count)
                costs.append(values)
                columns.append(record)
            costs = np.stack(costs, axis=1)
            observed = np.isfinite(costs)
            np.save(staging / 'raw_costs.npy', costs, allow_pickle=False)
            np.save(staging / 'observed.npy', observed, allow_pickle=False)
            _write_json(staging / 'manifest.json', dict(problem=problem, pool=list(POOLS[problem]),
                frozen_ns=inventory['frozen_ns'], columns=columns, coverage=_coverage(costs, observed),
                limitations=LIMITATIONS))
            os.replace(staging, directory)
        print('[R62 bank] ' + problem + ': ' + str(int(observed.sum())) + ' observed cells', flush=True)


def _native_input(problem, instance):
    if problem == 'TSP':
        result = np.asarray(instance[0], dtype=np.float32)
        if result.ndim != 2 or result.shape[1] != 2:
            raise ValueError('Expected native TSP (1,N,2) input')
    elif problem == 'CVRP':
        loc = np.asarray(instance['loc'][0], dtype=np.float32)
        demand = np.asarray(instance['demand'][0], dtype=np.float32)
        depot = np.asarray(instance['depot'][0], dtype=np.float32)
        if loc.ndim != 2 or loc.shape[1] != 2 or demand.shape != (len(loc),) or depot.shape != (1, 2):
            raise ValueError('Expected native CVRP loc/demand/depot fields')
        result = np.zeros((len(loc) + 1, 4), dtype=np.float32)
        result[0, :2], result[0, 3] = depot[0], 1.
        result[1:, :2], result[1:, 2] = loc, demand
    elif problem == 'ATSP':
        array = np.asarray(instance)
        result = np.asarray(array[0] if array.ndim == 3 else array, dtype=np.float32)
        if result.ndim != 2 or result.shape[0] != result.shape[1]:
            raise ValueError('Expected native directed ATSP square matrix')
    else:
        result = _node_input(instance, constraint_bits(problem))
    if not len(result) or not np.isfinite(result).all():
        raise ValueError('Empty/nonfinite expansion input')
    return result


def _prepare_features(root, problem, task, config, expanded_root):
    prefix, count = config['prefix_count'], config['source_count']
    if problem not in ('TSP', 'CVRP', 'ATSP'):
        directory = expanded_root / problem
        manifest = _json(directory / 'manifest.json')
        if (manifest['problem'] != problem or manifest['pool'] != list(POOLS[problem]) or
                manifest['sources'][0] != task['input_file'] or manifest['source_count'] != count or
                manifest['prefix_count'] != prefix or
                manifest.get('recorded_input_repairs', []) != task['generation'].get('data_repairs', [])):
            raise ValueError('Existing input mmap does not match frozen expansion source: ' + problem)
        return directory, dict(kind='coord', d_in=8, original_manifest=str(directory / 'manifest.json'))
    directory = root / 'features' / problem
    kind, width = ('matrix', None) if problem == 'ATSP' else ('coord', 2 if problem == 'TSP' else 4)
    if not directory.exists():
        directory.parent.mkdir(parents=True, exist_ok=True)
        nodes = np.zeros(count - prefix, dtype=np.int32)
        shards = task['generation']['shards']
        for shard in shards:
            start, size = int(shard['start_index']), int(shard['count'])
            if start < prefix or start + size > count or nodes[start - prefix:start + size - prefix].any():
                raise ValueError('Overlapping/out-of-range expansion input IDs')
            with Path(shard['path']).open('rb') as stream:
                instances = pickle.load(stream)
            if len(instances) != size:
                raise ValueError('Expansion shard input count mismatch')
            nodes[start - prefix:start + size - prefix] = [len(_native_input(problem, x)) for x in instances]
            del instances
        if (nodes <= 0).any():
            raise ValueError('Missing expansion input positions: ' + problem)
        elements = nodes.astype(np.int64) ** 2 if kind == 'matrix' else nodes
        offsets = np.concatenate(([0], np.cumsum(elements, dtype=np.int64)))
        with tempfile.TemporaryDirectory(prefix='.' + problem, dir=directory.parent) as temporary:
            staging = Path(temporary)
            name = 'matrix_values' if kind == 'matrix' else 'node_values'
            shape = (int(offsets[-1]),) if kind == 'matrix' else (int(offsets[-1]), width)
            values = np.lib.format.open_memmap(staging / (name + '.npy'), mode='w+', dtype=np.float32, shape=shape)
            for shard in shards:
                with Path(shard['path']).open('rb') as stream:
                    instances = pickle.load(stream)
                for local, instance in enumerate(instances):
                    row = int(shard['start_index']) - prefix + local
                    native = _native_input(problem, instance)
                    values[offsets[row]:offsets[row + 1]] = native.reshape(-1) if kind == 'matrix' else native
                del instances
            values.flush()
            del values
            for name, array in dict(nodes=nodes, offsets=offsets,
                    source_indices=np.arange(prefix, count, dtype=np.int64)).items():
                np.save(staging / (name + '.npy'), array, allow_pickle=False)
            _write_json(staging / 'manifest.json', dict(problem=problem, kind=kind, d_in=width,
                generation_file=task['generation_file'], input_file=task['input_file'], count=len(nodes)))
            os.replace(staging, directory)
    return directory, dict(kind=kind, d_in=width, original_manifest=str(directory / 'manifest.json'))


def _quotas(counts, total, ensure_each=False):
    counts = np.asarray(counts, dtype=np.int64)
    if total > counts.sum() or total < (len(counts) if ensure_each else 0):
        raise ValueError('Subset cannot cover all actual-size strata')
    ideal = counts * (total / counts.sum())
    result = np.floor(ideal).astype(np.int64)
    if ensure_each:
        result = np.maximum(result, 1)
    while result.sum() != total:
        delta = total - int(result.sum())
        valid = result < counts if delta > 0 else result > (1 if ensure_each else 0)
        score = ideal - result if delta > 0 else result - ideal
        result[int(np.argmax(np.where(valid, score, -np.inf)))] += 1 if delta > 0 else -1
    return result


def _subset(nodes, observed, count, seed, problem):
    rng = np.random.default_rng(np.random.SeedSequence([seed, 62, P2I[problem], 2]))
    sizes, size_counts = np.unique(nodes, return_counts=True)
    allocations = _quotas(size_counts, count, ensure_each=True)
    pattern = (observed.astype(np.int64) * (1 << np.arange(observed.shape[1]))).sum(1)
    selected = []
    for size, allocation in zip(sizes, allocations):
        rows = np.flatnonzero(nodes == size)
        patterns, counts = np.unique(pattern[rows], return_counts=True)
        for key, number in zip(patterns, _quotas(counts, int(allocation))):
            selected.extend(rng.choice(rows[pattern[rows] == key], size=int(number), replace=False).tolist())
    return np.sort(np.asarray(selected, dtype=np.int64))


class PortfolioDataset:
    """Full legal candidate inputs; observed FP64 costs and masks are separate outputs."""

    def __init__(self, directory, subset=False):
        self.directory = Path(directory)
        self.manifest = _json(self.directory / 'manifest.json')
        self.problem, self.pool = self.manifest['problem'], tuple(self.manifest['pool'])
        if self.pool != tuple(POOLS[self.problem]):
            raise ValueError('Portfolio cost columns differ from registry pool order')
        self.indices = np.load(self.directory / 'subset_indices.npy', mmap_mode='r') if subset else np.arange(self.manifest['count'])
        features = Path(self.manifest['feature_directory'])
        self.kind, self.d_in = self.manifest['kind'], self.manifest['d_in']
        self._nodes = np.load(features / 'nodes.npy', mmap_mode='r')
        self._sources = np.load(features / 'source_indices.npy', mmap_mode='r')
        self.offsets = np.load(features / 'offsets.npy', mmap_mode='r')
        name = 'matrix_values' if self.kind == 'matrix' else 'node_values'
        self.values = np.load(features / (name + '.npy'), mmap_mode='r')
        bank = Path(self.manifest['bank_directory'])
        self._costs = np.load(bank / 'raw_costs.npy', mmap_mode='r')
        self._observed = np.load(bank / 'observed.npy', mmap_mode='r')
        self.size = len(self.indices)
        self.nodes = self._nodes[self.indices] if subset else self._nodes
        self.source_indices = self._sources[self.indices] if subset else self._sources
        self.raw_costs = self._costs[self.indices] if subset else self._costs
        self.observed = self._observed[self.indices] if subset else self._observed
        self.lengths = self.nodes
        expected_sources = np.arange(self.manifest['prefix_count'], self.manifest['source_count'])
        if (not np.array_equal(self._sources, expected_sources) or
                self._costs.shape != (len(expected_sources), len(self.pool)) or self._costs.dtype != np.float64 or
                self._observed.shape != self._costs.shape):
            raise ValueError('Frozen input/cost row identity or FP64 shape mismatch')

    def __len__(self):
        return self.size

    def gather(self, indices, device='cpu'):
        if torch.is_tensor(indices):
            if indices.device.type != 'cpu':
                raise ValueError('Portfolio row indices must stay on CPU')
            indices = indices.numpy()
        indices = np.asarray(indices)
        if (indices.ndim != 1 or not len(indices) or not np.issubdtype(indices.dtype, np.integer) or
                indices.min() < 0 or indices.max() >= self.size):
            raise ValueError('Expected nonempty valid portfolio row indices')
        rows = self.indices[indices]
        sizes = np.asarray(self._nodes[rows], dtype=np.int64)
        batch_size, maximum = len(rows), int(sizes.max())
        pool_ids, mask = problem_to_pool_mask(self.problem)
        inputs = dict(problem_id=P2I[self.problem], kind=self.kind, n=torch.from_numpy(sizes),
            node_mask=torch.from_numpy(np.arange(maximum)[None] < sizes[:, None]),
            pool_ids=torch.tensor(pool_ids, dtype=torch.long),
            mask=torch.tensor(mask, dtype=torch.float32)[None].expand(batch_size, -1).clone(),
            cbits=torch.tensor(constraint_bits(self.problem), dtype=torch.float32)[None].expand(batch_size, -1).clone(),
            problem_desc=torch.tensor(problem_descriptor(self.problem), dtype=torch.float32)[None].expand(batch_size, -1).clone(),
            coord_dist=torch.full((batch_size,), 4, dtype=torch.long))
        shape = (batch_size, maximum, maximum) if self.kind == 'matrix' else (batch_size, maximum, self.d_in)
        values = np.zeros(shape, dtype=np.float32)
        for local, row in enumerate(rows):
            saved = self.values[self.offsets[row]:self.offsets[row + 1]]
            if self.kind == 'matrix':
                values[local, :sizes[local], :sizes[local]] = saved.reshape(sizes[local], sizes[local])
            else:
                values[local, :sizes[local]] = saved
        if self.kind == 'coord':
            inputs.update(node=torch.from_numpy(values), d_in=self.d_in)
        else:
            inputs['matrix'] = torch.from_numpy(values)
        inputs = {key: value.to(device) if torch.is_tensor(value) else value for key, value in inputs.items()}
        costs = torch.from_numpy(np.array(self._costs[rows], copy=True)).to(device)
        observed = torch.from_numpy(np.array(self._observed[rows], copy=True)).to(device)
        return inputs, costs, observed


def prepare_portfolio(cache_root=CACHE_ROOT, source_root=SOURCE_ROOT, *, producer_root=PRODUCER_ROOT,
                      expanded_root=EXPANDED_ROOT, problems=PROBLEMS, source_count=100000,
                      prefix_count=10000, subset_count=10000, seed=2):
    """Freeze observations once, convert native inputs sequentially, publish nested views."""
    root, source_root = Path(cache_root), Path(source_root)
    problems = tuple(problems)
    if (not problems or len(set(problems)) != len(problems) or any(p not in PROBLEMS for p in problems) or
            not 0 < prefix_count < source_count or not 0 < subset_count <= source_count - prefix_count):
        raise ValueError('Invalid TRAIN-only portfolio dimensions/tasks')
    root.mkdir(parents=True, exist_ok=True)
    inventory = _freeze_inventory(root, source_root, Path(producer_root), problems, source_count,
                                  prefix_count, subset_count, seed)
    _freeze_bank(root, inventory)
    data = {'A': {}, 'B': {}}
    for problem, task in inventory['tasks'].items():
        directory = root / 'datasets' / problem
        if not directory.exists():
            features, metadata = _prepare_features(root, problem, task, inventory['config'], Path(expanded_root))
            nodes = np.load(features / 'nodes.npy', mmap_mode='r')
            costs = np.load(root / 'bank' / problem / 'raw_costs.npy', mmap_mode='r')
            observed = np.load(root / 'bank' / problem / 'observed.npy', mmap_mode='r')
            if len(nodes) != source_count - prefix_count:
                raise ValueError('Expanded feature count differs from frozen observation positions')
            indices = _subset(nodes, observed, subset_count, seed, problem)
            directory.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(prefix='.' + problem, dir=directory.parent) as temporary:
                staging = Path(temporary)
                np.save(staging / 'subset_indices.npy', indices, allow_pickle=False)
                np.save(staging / 'subset_source_indices.npy', np.asarray(np.arange(prefix_count, source_count)[indices]), allow_pickle=False)
                _write_json(staging / 'manifest.json', dict(metadata, problem=problem, pool=list(POOLS[problem]),
                    feature_directory=str(features.resolve()), bank_directory=str((root / 'bank' / problem).resolve()),
                    bank_manifest=str(root / 'bank' / problem / 'manifest.json'), count=len(nodes),
                    source_count=source_count, prefix_count=prefix_count, subset_count=len(indices),
                    full_coverage=_coverage(costs, observed), subset_coverage=_coverage(costs[indices], observed[indices]),
                    size_counts={str(n): int((nodes == n).sum()) for n in np.unique(nodes)},
                    subset_size_counts={str(n): int((nodes[indices] == n).sum()) for n in np.unique(nodes)},
                    subset_rule='seeded actual-size and observed-availability-pattern stratification; no cost values',
                    limitations=LIMITATIONS))
                os.replace(staging, directory)
        data['A'][problem], data['B'][problem] = PortfolioDataset(directory, True), PortfolioDataset(directory)
        print('[R62 inputs] ' + problem + ': A=' + str(len(data['A'][problem])) +
              ' B=' + str(len(data['B'][problem])) + ' CPU rows', flush=True)
    return data


def portfolio_schedule(data, passes=2, seed=2):
    """Two full B traversals; fixed nested A subset replay matches actual batch sizes."""
    plans = {'A': {}, 'B': {}}
    for problem, full in data['B'].items():
        subset = data['A'][problem]
        buckets = {int(n): np.flatnonzero(subset.nodes == n) for n in np.unique(full.nodes)}
        if any(not len(rows) for rows in buckets.values()):
            raise ValueError('A subset does not cover a B actual-size stratum')
        rng_b = np.random.default_rng(np.random.SeedSequence([seed, 62, P2I[problem], 0]))
        rng_a = np.random.default_rng(np.random.SeedSequence([seed, 62, P2I[problem], 1]))
        plans['A'][problem], plans['B'][problem] = [], []
        for _ in range(passes):
            b = rng_b.permutation(len(full))
            a = np.empty(len(b), dtype=np.int64)
            for size, rows in buckets.items():
                positions = np.flatnonzero(full.nodes[b] == size)
                a[positions] = rng_a.choice(rows, size=len(positions), replace=True)
            plans['A'][problem].append(a)
            plans['B'][problem].append(b)
    return plans
