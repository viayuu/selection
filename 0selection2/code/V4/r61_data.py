"""CPU-only saved-TRAIN-route caches for dense behavior pretraining."""

import json
import os
import pickle
import tempfile
from pathlib import Path

import numpy as np
import torch

from code.unified_selector.registry import (DATA_ROOT, P2I, POOLS, constraint_bits,
                                            problem_descriptor, problem_to_pool_mask)
from .r54_routes import route_successors
from .r60_data import (CACHE_ROOT as EXPANDED_ROOT, GENERATION_RECORD, LIMITATIONS,
                       MVRP_PROBLEMS, PAIR_SOLVERS, SOURCE_ROOT, _file_identity, _node_input)


DIRECTORY = Path(__file__).resolve().parent
CACHE_ROOT = DIRECTORY / 'runs/R61_behavior_pretraining/data_cache'
OLD_ROUTE_ROOT = DIRECTORY / 'runs/R54_route_supervision/route_cache'
PRODUCER_ROOT = Path('/public/home/shiys/easynco_v3_bridge/exports/train_100k_20261002')
ARRAY_NAMES = ('node_values', 'nodes', 'offsets', 'source_indices')


def _feature_arrays(directory):
    return {name: np.load(Path(directory) / (name + '.npy'), mmap_mode='r', allow_pickle=False)
            for name in ARRAY_NAMES}


def _validate_targets(target, nodes):
    customers = np.asarray(nodes) - 1
    width = target.shape[-1]
    valid = np.arange(width)[None, None, :] < customers[:, None, None]
    self_id = np.arange(1, width + 1)[None, None, :]
    if (target.ndim != 3 or target.shape[:2] != (len(nodes), 2) or
            np.any(valid & ((target < 0) | (target > customers[:, None, None]) | (target == self_id))) or
            np.any(~valid & (target != -100))):
        raise ValueError('Invalid successor/END/padding targets')


def _old_context(problem, original_root, route_root, count):
    return dict(version=1, group='old', problem=problem, split='train', count=count,
        pair_solvers=list(PAIR_SOLVERS), pool=list(POOLS[problem]), source_first=0, source_last=count - 1,
        sources=[_file_identity(original_root / (problem + 'train/dataset.pkl')),
                 _file_identity(route_root / (problem + '_train.npz'))],
        target_convention='original customer IDs 1..N; 0=END; -100=padding',
        conversion='reuse existing R54 route_successors outputs from TRAIN only', limitations=LIMITATIONS)


def _prepare_old(directory, context):
    with Path(context['sources'][0]['path']).open('rb') as stream:
        instances = pickle.load(stream)
    count, problem = context['count'], context['problem']
    if len(instances) != count:
        raise ValueError('Original TRAIN input count changed: ' + problem)
    nodes = np.asarray([len(item['node_xy']) + 1 for item in instances], dtype=np.int32)
    with np.load(context['sources'][1]['path'], allow_pickle=False) as archive:
        if (archive['pool'].tolist() != list(PAIR_SOLVERS) or
                not np.array_equal(archive['selected_indices'], np.arange(count)) or
                not np.array_equal(archive['customers'], nodes - 1) or
                archive['complete'].shape != (count, 2) or not archive['complete'].all()):
            raise ValueError('R54 original TRAIN route identities/counts do not match: ' + problem)
        successor = archive['successor']
        recorded_data_hash = str(archive['data_hash'])
    _validate_targets(successor, nodes)
    offsets = np.concatenate(([0], np.cumsum(nodes, dtype=np.int64)))
    with tempfile.TemporaryDirectory(prefix='.' + problem + '-', dir=directory.parent) as temporary:
        staging = Path(temporary)
        features = staging / 'features'
        features.mkdir()
        values = np.lib.format.open_memmap(features / 'node_values.npy', mode='w+', dtype=np.float32,
                                          shape=(int(offsets[-1]), 8))
        bits = constraint_bits(problem)
        for row, item in enumerate(instances):
            values[offsets[row]:offsets[row + 1]] = _node_input(item, bits)
        values.flush()
        del values, instances
        for name, array in dict(nodes=nodes, offsets=offsets, source_indices=np.arange(count, dtype=np.int64)).items():
            np.save(features / (name + '.npy'), array, allow_pickle=False)
        np.save(staging / 'successor.npy', successor, allow_pickle=False)
        manifest = dict(context, feature_directory=str((directory / 'features').resolve()),
                        recorded_r54_input_hash=recorded_data_hash, resolved_method_routes=2 * count,
                        max_customers=int(nodes.max()) - 1, cost_reconstruction=False)
        _publish(staging, directory, manifest)


def _new_context(problem, expanded_root, source_root, producer_root, count, prefix):
    features = expanded_root / problem
    feature_manifest_path = features / 'manifest.json'
    feature_manifest = json.loads(feature_manifest_path.read_text())
    arrays = _feature_arrays(features)
    if (feature_manifest['problem'] != problem or feature_manifest['count'] != count or
            feature_manifest['prefix_count'] != prefix or
            feature_manifest['pair_solvers'] != list(PAIR_SOLVERS) or
            feature_manifest['pool'] != list(POOLS[problem]) or
            not np.array_equal(arrays['source_indices'], np.arange(prefix, prefix + count))):
        raise ValueError('R60 expansion feature identities/counts changed: ' + problem)
    generation_path = source_root / (problem + 'train') / GENERATION_RECORD
    generation = json.loads(generation_path.read_text())
    if generation.get('data_repairs', []) != feature_manifest['recorded_input_repairs']:
        raise ValueError('Repaired input and R60 feature cache no longer agree: ' + problem)
    sources, bindings = [_file_identity(feature_manifest_path), _file_identity(generation_path)], []
    for solver in PAIR_SOLVERS:
        for shard in generation['shards']:
            meta_path = producer_root / 'labels' / problem / solver / 'shards' / (shard['id'] + '.meta.json')
            meta = json.loads(meta_path.read_text())
            expected = dict(problem=problem, method=solver, start_index=shard['start_index'],
                            count=shard['count'], input_sha256=shard['sha256'])
            if any(meta.get(key) != value for key, value in expected.items()):
                raise ValueError('Saved tour metadata/input identity mismatch: ' + str(meta_path))
            result_path = Path(meta['result_path'])
            sources.extend((_file_identity(meta_path), _file_identity(result_path)))
            bindings.append(dict(expected, shard_id=shard['id'], metadata_path=str(meta_path.resolve()),
                result_path=str(result_path.resolve()), recorded_result_sha256=meta['result_sha256'],
                input_path=shard['path'], profile_sha256=meta['profile_sha256']))
    return dict(version=1, group='new', problem=problem, split='train', count=count,
        pair_solvers=list(PAIR_SOLVERS), pool=list(POOLS[problem]), source_first=prefix,
        source_last=prefix + count - 1, feature_directory=str(features.resolve()), sources=sources,
        route_sources=bindings, recorded_input_repairs=feature_manifest['recorded_input_repairs'],
        target_convention='original customer IDs 1..N; 0=END; -100=padding',
        conversion='R54 route_successors applied to each producer-selected best native tour',
        limitations=LIMITATIONS)


def _prepare_new(directory, context):
    nodes = _feature_arrays(context['feature_directory'])['nodes']
    count, prefix, problem = context['count'], context['source_first'], context['problem']
    maximum = int(nodes.max()) - 1
    seen = np.zeros((count, 2), dtype=bool)
    with tempfile.TemporaryDirectory(prefix='.' + problem + '-', dir=directory.parent) as temporary:
        staging = Path(temporary)
        successor = np.lib.format.open_memmap(staging / 'successor.npy', mode='w+', dtype=np.int16,
                                             shape=(count, 2, maximum))
        successor[:] = -100
        witness = None
        for binding in context['route_sources']:
            payload = json.loads(Path(binding['result_path']).read_text())
            keys = ('problem', 'method', 'start_index', 'count', 'input_sha256', 'shard_id', 'profile_sha256')
            if any(payload.get(key) != binding[key] for key in keys):
                raise ValueError('Saved tour body/metadata identity mismatch: ' + binding['result_path'])
            role = PAIR_SOLVERS.index(binding['method'])
            start = binding['start_index'] - prefix
            end = start + binding['count']
            if (start < 0 or end > count or seen[start:end, role].any() or
                    len(payload.get('tours', [])) != end - start or
                    len(payload.get('costs', [])) != end - start):
                raise ValueError('Missing/overlapping saved tour rows: ' + binding['result_path'])
            for offset, route in enumerate(payload['tours']):
                row = start + offset
                customers = int(nodes[row]) - 1
                try:
                    successor[row, role, :customers] = route_successors(route, customers)
                except (ValueError, TypeError, OverflowError) as error:
                    raise ValueError('Unusable saved tour: ' + str((problem, binding['method'], prefix + row))) from error
            seen[start:end, role] = True
            if witness is None:
                witness = dict(result_path=binding['result_path'], metadata_path=binding['metadata_path'],
                    source_index=prefix + start, method=binding['method'], customers=int(nodes[start]) - 1,
                    tour_count=len(payload['tours']), tour_shape=list(np.asarray(payload['tours'][0]).shape),
                    route_prefix=payload['tours'][0][:12], best_route_only=True,
                    selection_source='/public/home/shiys/easynco_v3_bridge/scripts/train_100k_backends.py:248')
        if not seen.all():
            missing = np.count_nonzero(~seen, axis=0).tolist()
            raise ValueError('Missing saved route coverage: ' + str((problem, list(PAIR_SOLVERS), missing)))
        _validate_targets(successor, nodes)
        successor.flush()
        del successor
        manifest = dict(context, resolved_method_routes=int(seen.sum()), max_customers=maximum,
                        artifact_witness=witness, cost_reconstruction=False)
        _publish(staging, directory, manifest)


def _publish(staging, directory, manifest):
    if manifest['sources'] != [_file_identity(Path(item['path'])) for item in manifest['sources']]:
        raise ValueError('Saved TRAIN inputs/routes changed during conversion')
    (staging / 'manifest.json').write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')
    os.replace(staging, directory)


class BehaviorDataset:
    """Read-only packed input features and two method-specific successor targets."""

    def __init__(self, directory):
        self.directory = Path(directory)
        self.manifest = json.loads((self.directory / 'manifest.json').read_text())
        self.problem = self.manifest['problem']
        self.size = self.manifest['count']
        self.pair_solvers = PAIR_SOLVERS
        if (self.manifest['split'] != 'train' or self.manifest['pair_solvers'] != list(PAIR_SOLVERS) or
                self.manifest['pool'] != list(POOLS[self.problem])):
            raise ValueError('Behavior cache must retain TRAIN and named full-pool identities')
        for name, array in _feature_arrays(self.manifest['feature_directory']).items():
            setattr(self, name, array)
        self.lengths = self.nodes
        self.successor = np.load(self.directory / 'successor.npy', mmap_mode='r', allow_pickle=False)
        if (self.nodes.shape != (self.size,) or self.source_indices.shape != (self.size,) or
                self.successor.shape != (self.size, 2, self.manifest['max_customers']) or
                self.successor.dtype != np.int16):
            raise ValueError('Incomplete behavior cache arrays')

    def __len__(self):
        return self.size

    def gather(self, indices, device='cpu'):
        if torch.is_tensor(indices):
            if indices.device.type != 'cpu':
                raise ValueError('Behavior row indices must remain on CPU')
            indices = indices.numpy()
        indices = np.asarray(indices)
        if (indices.ndim != 1 or not len(indices) or not np.issubdtype(indices.dtype, np.integer) or
                indices.min() < 0 or indices.max() >= self.size):
            raise ValueError('Expected nonempty valid behavior row indices')
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
        targets = torch.from_numpy(np.array(self.successor[indices, :, :max_n - 1], dtype=np.int64, copy=True))
        clean = {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}
        return clean, targets.to(device)


def prepare_behavior(cache_root=CACHE_ROOT, *, expanded_root=EXPANDED_ROOT, original_root=DATA_ROOT,
                     route_root=OLD_ROUTE_ROOT, source_root=SOURCE_ROOT, producer_root=PRODUCER_ROOT,
                     problems=MVRP_PROBLEMS, original_count=10000, expanded_count=90000):
    """Resolve and convert existing TRAIN behavior, one problem/shard at a time."""
    cache_root = Path(cache_root)
    problems = tuple(problems)
    if not problems or len(set(problems)) != len(problems) or any(p not in MVRP_PROBLEMS for p in problems):
        raise ValueError('Behavior pretraining is restricted to the original 15 MVRP tasks')
    result = {'old': {}, 'new': {}}
    for group in result:
        (cache_root / group).mkdir(parents=True, exist_ok=True)
    for problem in problems:
        contexts = dict(old=_old_context(problem, Path(original_root), Path(route_root), original_count),
            new=_new_context(problem, Path(expanded_root), Path(source_root), Path(producer_root), expanded_count, original_count))
        for group, context in contexts.items():
            directory = cache_root / group / problem
            if directory.exists():
                previous = json.loads((directory / 'manifest.json').read_text())
                if any(previous.get(key) != value for key, value in context.items()):
                    raise ValueError('Behavior source identities changed; use a new cache root: ' + str((problem, group)))
            else:
                (_prepare_old if group == 'old' else _prepare_new)(directory, context)
            result[group][problem] = BehaviorDataset(directory)
            print('[R61 behavior] ' + problem + '/' + group + ': ' + str(context['count']) +
                  ' TRAIN queries / ' + str(2 * context['count']) + ' resolved method routes', flush=True)
    return result


def behavior_schedule(datasets, passes=2, seed=2):
    """Two complete B traversals, with replacement A replay matching every true size."""
    plans = {'A': {}, 'B': {}}
    for problem, new in datasets['new'].items():
        old = datasets['old'][problem]
        buckets = {int(n): np.flatnonzero(old.nodes == n) for n in np.unique(new.nodes)}
        if any(not len(rows) for rows in buckets.values()):
            raise ValueError('New true size has no original-TRAIN replay bucket: ' + problem)
        rng_b = np.random.default_rng(np.random.SeedSequence([seed, 61, P2I[problem], 0]))
        rng_a = np.random.default_rng(np.random.SeedSequence([seed, 61, P2I[problem], 1]))
        plans['A'][problem], plans['B'][problem] = [], []
        for _ in range(passes):
            b = rng_b.permutation(new.size)
            a = np.empty(new.size, dtype=np.int64)
            sizes = new.nodes[b]
            for size, rows in buckets.items():
                positions = np.flatnonzero(sizes == size)
                a[positions] = rng_a.choice(rows, size=len(positions), replace=True)
            plans['A'][problem].append(a)
            plans['B'][problem].append(b)
    return plans
