"""Explicit scenario data loader; never changes legacy DATA_ROOT or pool globals."""

import json
import pickle

import numpy as np
import torch

from ..unified_selector.data import UnifiedProblemDataset, collate_single_problem
from ..unified_selector.registry import M_GLOBAL, P2I, PROBLEMS, S2I, constraint_bits, problem_descriptor
from .pairwise_objective import comparison_targets
from .performance_experiment import prepare_geometry
from .r58_labels import read_lock
from .r58_scenario import file_hash
from .tensor_loader import TensorBatchLoader


class ScenarioDataset(UnifiedProblemDataset):
    def __init__(self, root, problem, split, pool):
        directory = root / 'scenario_v2' / f'{problem}{split}'
        manifest = json.loads((directory / 'manifest.json').read_text())
        if manifest['pool'] != pool:
            raise ValueError('Scenario solver names/order differ from locked pool')
        if (file_hash(directory / 'dataset.pkl') != manifest['input_sha256'] or
                file_hash(directory / 'raw_label.pkl') != manifest['label_sha256']):
            raise ValueError('Published scenario data changed')
        with (directory / 'dataset.pkl').open('rb') as stream:
            self.instances = pickle.load(stream)
        with (directory / 'raw_label.pkl').open('rb') as stream:
            self.labels = pickle.load(stream)
        self.problem, self.split, self.coord_augment = problem, split, 0
        self.base_N = self.N = len(self.instances)
        self.pool_order = [S2I[name] for name in pool]
        self.K_p = len(pool)
        self.mask = np.zeros(M_GLOBAL, np.float32)
        self.mask[self.pool_order] = 1.
        self.cbits, self.problem_desc = constraint_bits(problem), problem_descriptor(problem)
        self.pid = P2I[problem]
        if set(self.labels) != set(map(str, range(self.N))):
            raise ValueError('Non-contiguous labels or dropped instances')
        costs = np.array([self.labels[str(i)]['cost'] for i in range(self.N)], np.float64)
        native = np.array([self.labels[str(i)]['ind'] for i in range(self.N)])
        if costs.shape != (self.N, self.K_p) or not np.isfinite(costs).all():
            raise ValueError('Incomplete scenario cost vector')
        np.testing.assert_array_equal(costs.argmin(1), native)
        self.raw = dict(costs=costs, winner=native, pool=pool, pool_ids=self.pool_order,
                        data_hash=manifest['input_sha256'], label_hash=manifest['label_sha256'])

    def __getitem__(self, index):
        sample = super().__getitem__(index)
        sample['coord_dist'] = 4  # No distribution metadata inferred from row index.
        return sample


def make_loader(root, problem, split, pool, device, batch_size=128):
    dataset = ScenarioDataset(root, problem, split, pool)
    batch = collate_single_problem([dataset[i] for i in range(len(dataset))])
    batch.update(comparison_targets(dataset.raw['costs']))
    batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
    np.testing.assert_array_equal(batch['pool_ids'].cpu().numpy(), dataset.raw['pool_ids'])
    np.testing.assert_array_equal(batch['ind'].cpu().numpy(), dataset.raw['winner'])
    return TensorBatchLoader(batch, batch_size, False, False), dataset.raw


def release_check(root):
    locked = read_lock(root)
    release = json.loads((root / 'scenario_v2' / 'release.json').read_text())
    if not release['ready'] or release['lock_sha256'] != locked['lock_sha256'] or len(release['datasets']) != 54:
        raise ValueError('Training requires a complete validated scenario_v2 release')
    found = set()
    for record in release['datasets']:
        pair = record['problem'], record['split']
        if pair in found or pair[0] not in PROBLEMS or pair[1] not in ('train', 'val', 'test'):
            raise ValueError('Invalid or duplicated release dataset')
        found.add(pair)
        path = root / 'scenario_v2' / (pair[0] + pair[1]) / 'manifest.json'
        if json.loads(path.read_text()) != record or record['pool'] != locked['pools'][pair[0]]:
            raise ValueError('Published dataset manifest differs from release')
    return locked, release


def prepare(root, device, batch_size=128):
    locked, _ = release_check(root)
    loaders, raw = {s: {} for s in ('train', 'val')}, {s: {} for s in ('train', 'val')}
    for split in loaders:
        for problem in PROBLEMS:
            loader, record = make_loader(root, problem, split, locked['pools'][problem], device, batch_size)
            loaders[split][problem], raw[split][problem] = loader, record
    geometry = prepare_geometry(loaders, raw, root / 'geometry_cache')
    return loaders, raw, geometry


def attach_test_geometry(loader):
    from .local_geometry import local_geometry
    from .multitask_probe import gather_batch
    if loader.batch['kind'] != 'coord':
        return
    batch = loader.batch
    features = torch.zeros(loader.size, batch['node'].shape[1], 12, device=batch['node'].device)
    for start in range(0, loader.size, 16):
        indices = torch.arange(start, min(start + 16, loader.size))
        selected = gather_batch(loader, indices)
        values = local_geometry(selected['node'][..., :2], selected['node_mask'])
        features[start:start + len(indices), :values.shape[1]] = values
    loader.batch['node_geom'] = features
