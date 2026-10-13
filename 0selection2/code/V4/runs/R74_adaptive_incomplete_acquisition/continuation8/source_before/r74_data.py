"""Original native TRAIN/VAL facade and lossless factorized acquisition caches."""

import math
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch

from code.unified_selector.data import UnifiedProblemDataset, collate_single_problem
from code.unified_selector.registry import P2I
from .r59_experiment import atomic_json
from .r74_acquisition import quarter_increment


@lru_cache(maxsize=128)
def mapped_array(path):
    return np.load(path, mmap_mode='r', allow_pickle=False)


def probe_ids(problem):
    return {'TSP': (0,), 'CVRP': (1,), 'ATSP': (2,)}.get(problem, (3, 4))


def geometry_of_instance(problem, sample):
    if problem == 'TSP':
        return int(sample.shape[-2]), 1, 0
    if problem == 'ATSP':
        n = int(sample.shape[-1])
        return n, n, 0
    if problem == 'CVRP':
        n = int(sample['loc'].shape[-2])
        return n, min(n, 100), 0
    n = len(sample['node_xy'])
    positive = int((np.asarray(sample['node_demand']) > 0).sum())
    starts = int(n * (1-.2)) if 'B' in problem else n
    if 'B' in problem and positive < starts:
        raise ValueError('Native backhaul start subset cannot supply the locked environment starts')
    return n, starts, positive if 'B' in problem else 0


class NativeInputs:
    def __init__(self, problem, split, geometry_directory):
        if split not in ('train', 'val'):
            raise ValueError('R74 implementation authorizes original TRAIN/VAL only')
        self.problem, self.split = problem, split
        self.dataset = UnifiedProblemDataset(problem, split, coord_augment=0)
        self.instances = self.dataset.instances
        self.size = len(self.dataset)
        expected = 10000 if split == 'train' else 1000
        if self.size != expected:
            raise ValueError('Native full task coverage changed')
        self.geometry = None if problem == 'ATSP' else torch.load(
            Path(geometry_directory) / f'{problem}_{split}.pt', map_location='cpu', weights_only=True)
        self.groups = {}
        for row, sample in enumerate(self.instances):
            key = geometry_of_instance(problem, sample)
            quarter_increment(key[0])
            self.groups.setdefault(key, []).append(row)

    def gather(self, rows, device):
        batch = collate_single_problem([self.dataset[int(row)] for row in rows])
        if self.geometry is not None:
            batch['node_geom'] = self.geometry[torch.as_tensor(rows, dtype=torch.long), :batch['node'].shape[1]]
        return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}


class GroupWriter:
    def __init__(self, directory, rows, n, starts, pool_ids, problem):
        directory.mkdir(parents=True, exist_ok=False)
        self.directory, self.arrays = directory, {}
        self.rows, self.n, self.starts = np.asarray(rows, dtype=np.int64), n, starts
        self.node_count = n + int(problem not in ('TSP', 'ATSP'))
        count, candidates = len(rows), len(pool_ids)
        specs = dict(nodes=((count, self.node_count, 128), np.float32),
            features=((count, candidates, 384), np.float32), h=((count, 128), np.float32),
            base_logits=((count, candidates), np.float32))
        for probe in probe_ids(problem):
            for checkpoint in (0, 1):
                prefix = f'p{probe}_c{checkpoint}_'
                specs.update({prefix+'start': ((count, starts), np.int32),
                    prefix+'current': ((count, starts), np.int32),
                    prefix+'visited_mean': ((count, starts, 128), np.float32),
                    prefix+'scalars': ((count, starts, 12), np.float32),
                    prefix+'visited_bits': ((count, starts, math.ceil(self.node_count/8)), np.uint8)})
        for key, (shape, dtype) in specs.items():
            self.arrays[key] = np.lib.format.open_memmap(directory / (key+'.npy'), mode='w+', dtype=dtype, shape=shape)
        np.save(directory / 'source_indices.npy', self.rows, allow_pickle=False)
        self.metadata = dict(directory=str(directory), rows=count, true_n=n, node_count=self.node_count,
            starts=starts, pool_ids=list(pool_ids), original_source_indices='source_indices.npy',
            representation='exact FP32 base nodes/features/utilities/summary; integer start/current; '
                'lossless packed distinct-visited identities; exact FP32 visited mean/scalars; no token quantization',
            specs={key: dict(shape=list(shape), dtype=np.dtype(dtype).name) for key, (shape, dtype) in specs.items()},
            complete=False)
        atomic_json(directory / 'metadata.json', self.metadata)

    def write_base(self, start, stop, values):
        for key in ('nodes', 'features', 'h', 'base_logits'):
            self.arrays[key][start:stop] = values[key].detach().cpu().numpy()

    def write_observation(self, start, stop, probe, checkpoint, value):
        for key in ('start', 'current', 'visited_mean', 'scalars', 'visited_bits'):
            array = value[key]
            self.arrays[f'p{probe}_c{checkpoint}_{key}'][start:stop] = (
                array.detach().cpu().numpy() if torch.is_tensor(array) else array)

    def flush(self):
        for array in self.arrays.values():
            array.flush()

    def finish(self):
        self.flush()
        self.metadata['complete'] = True
        self.metadata['storage_bytes'] = sum(array.nbytes for array in self.arrays.values())
        atomic_json(self.directory / 'metadata.json', self.metadata)
        return self.metadata


class CachedTask:
    def __init__(self, metadata, raw):
        self.problem, self.split = metadata['problem'], metadata['split']
        self.size, self.raw, self.groups = metadata['rows'], raw, []
        if not metadata['complete']:
            raise ValueError('A partial acquisition cache cannot train/evaluate a policy')
        self.locations = np.full((self.size, 2), -1, dtype=np.int64)
        for group_id, record in enumerate(metadata['groups']):
            if not record['complete']:
                raise ValueError('Incomplete source group')
            directory = Path(record['directory'])
            source = np.load(directory / 'source_indices.npy', allow_pickle=False)
            if (self.locations[source] != -1).any():
                raise ValueError('Duplicate source identity in acquisition cache')
            self.locations[source, 0] = group_id
            self.locations[source, 1] = np.arange(len(source))
            self.groups.append(dict(record=record, directory=directory))
        if (self.locations < 0).any():
            raise ValueError('Acquisition cache lost original native rows')

    def take(self, keys, indices):
        indices = np.asarray(indices, dtype=np.int64)
        locations = self.locations[indices]
        selected_groups = np.unique(locations[:, 0])
        result = {}
        for key in keys:
            shapes = [tuple(self.groups[int(g)]['record']['specs'][key]['shape'][1:]) for g in selected_groups]
            shape = tuple(max(s[axis] for s in shapes) for axis in range(len(shapes[0])))
            dtype = self.groups[int(selected_groups[0])]['record']['specs'][key]['dtype']
            out = np.zeros((len(indices), *shape), dtype=dtype)
            for group_id in selected_groups:
                positions = np.flatnonzero(locations[:, 0] == group_id)
                values = mapped_array(self.groups[int(group_id)]['directory'] / (key+'.npy'))[locations[positions, 1]]
                section = (positions,) + tuple(slice(0, size) for size in values.shape[1:])
                out[section] = values
            result[key] = out
        return result

    def gather(self, indices, device):
        indices = indices.numpy() if torch.is_tensor(indices) else np.asarray(indices, dtype=np.int64)
        values = self.take(('nodes', 'features', 'h', 'base_logits'), indices)
        values = {key: torch.from_numpy(value).to(device) for key, value in values.items()}
        values.update(problem_id=P2I[self.problem], mvrp=self.problem not in ('TSP', 'CVRP', 'ATSP'),
            probe_id=probe_ids(self.problem)[0], pool_ids=torch.tensor(self.raw['pool_ids'], dtype=torch.long, device=device),
            solver_mask=torch.ones(len(indices), len(self.raw['pool_ids']), dtype=torch.bool, device=device),
            winner=torch.tensor(self.raw['winner'][indices], dtype=torch.long, device=device))
        values['observations'] = Observations(self, indices, values['nodes'], device)
        del values['nodes']
        return values


class Observations:
    def __init__(self, task, indices, nodes, device):
        self.task, self.indices, self.nodes, self.device = task, indices, nodes, device
        self.cache = {}

    def __call__(self, probe, checkpoint, positions=None):
        selected = np.arange(len(self.indices)) if positions is None else positions.detach().cpu().numpy()
        key = (probe, checkpoint, tuple(selected.tolist()))
        if key in self.cache:
            return self.cache[key]
        source = self.indices[selected]
        prefix = f'p{probe}_c{checkpoint}_'
        names = ('start', 'current', 'visited_mean', 'scalars')
        raw = self.task.take(tuple(prefix+name for name in names), source)
        arrays = {name: torch.from_numpy(raw[prefix+name]).to(self.device) for name in names}
        starts = [self.task.groups[int(g)]['record']['starts'] for g in self.task.locations[source, 0]]
        mask = torch.arange(arrays['start'].shape[1], device=self.device)[None] < torch.tensor(starts, device=self.device)[:, None]
        nodes = self.nodes if positions is None else self.nodes[positions]
        first = nodes.gather(1, arrays['start'].long()[..., None].expand(-1, -1, 128))
        current = nodes.gather(1, arrays['current'].long()[..., None].expand(-1, -1, 128))
        identity = torch.zeros(*mask.shape, 7, device=self.device)
        identity[..., probe] = 1.
        identity[..., 5+checkpoint] = 1.
        tokens = torch.cat((first, current, arrays['visited_mean'], arrays['scalars'], identity), -1)
        if tokens.shape[-1] != 403 or not torch.isfinite(tokens[mask]).all():
            raise ValueError('Lossless token reconstruction failed')
        self.cache[key] = tokens, mask
        return tokens, mask
