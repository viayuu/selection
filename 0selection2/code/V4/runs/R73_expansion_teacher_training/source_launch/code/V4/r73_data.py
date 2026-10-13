"""R73 expansion inputs and immutable hard teacher targets; no cost-bank reader."""

import json
from pathlib import Path

import numpy as np
import torch

from code.unified_selector.registry import (GLOBAL_SOLVERS, P2I, POOLS, PROBLEMS,
                                            constraint_bits, problem_descriptor,
                                            problem_to_pool_mask)
from .local_geometry import local_geometry


CACHE_ROOT = Path(__file__).resolve().parent / 'runs/R62_portfolio_performance_pretraining/data_cache'


class ExpansionInputs:
    def __init__(self, problem):
        self.problem = problem
        manifest_path = CACHE_ROOT / 'datasets' / problem / 'manifest.json'
        manifest = json.loads(manifest_path.read_text())
        self.pool = tuple(manifest['pool'])
        self.kind, self.d_in = manifest['kind'], manifest['d_in']
        self.size = int(manifest['count'])
        if (manifest['problem'] != problem or self.pool != tuple(POOLS[problem])
                or self.size != 90000 or manifest['prefix_count'] != 10000
                or manifest['source_count'] != 100000):
            raise ValueError('R73 requires the complete original expansion input range and pool')
        features = Path(manifest['feature_directory'])
        names = ('nodes.npy', 'source_indices.npy', 'offsets.npy',
                 'matrix_values.npy' if self.kind == 'matrix' else 'node_values.npy')
        self.opened_input_files = [str(features / name) for name in names]
        self.nodes, self.sources, self.offsets, self.values = [
            np.load(path, mmap_mode='r', allow_pickle=False) for path in self.opened_input_files]
        if (self.nodes.shape != (90000,) or self.sources.shape != (90000,)
                or self.offsets.shape != (90001,)
                or not np.array_equal(self.sources, np.arange(10000, 100000))
                or self.values.dtype != np.float32 or int(self.offsets[-1]) != len(self.values)):
            raise ValueError('Expansion input rows, offsets or source IDs changed')
        self.metadata = dict(problem=problem, problem_id=P2I[problem], count=self.size,
            source_first=10000, source_last=99999, pool=list(self.pool),
            pool_ids=[GLOBAL_SOLVERS.index(name) for name in self.pool], kind=self.kind, d_in=self.d_in,
            manifest=str(manifest_path), feature_directory=str(features),
            opened_input_files=self.opened_input_files, auxiliary_costs_opened=False,
            observation_masks_opened=False, geometry='existing label-free local_geometry; original buffers retained')

    def gather(self, indices, device):
        if torch.is_tensor(indices):
            if indices.device.type != 'cpu':
                raise ValueError('Expansion row indices must remain on CPU')
            indices = indices.numpy()
        indices = np.asarray(indices)
        if (indices.ndim != 1 or not len(indices) or not np.issubdtype(indices.dtype, np.integer)
                or indices.min() < 0 or indices.max() >= self.size):
            raise ValueError('Invalid expansion row indices')
        sizes = np.asarray(self.nodes[indices], dtype=np.int64)
        rows, maximum = len(indices), int(sizes.max())
        pool_ids, global_mask = problem_to_pool_mask(self.problem)
        inputs = dict(problem_id=P2I[self.problem], kind=self.kind, n=torch.from_numpy(sizes),
            node_mask=torch.from_numpy(np.arange(maximum)[None] < sizes[:, None]),
            pool_ids=torch.tensor(pool_ids, dtype=torch.long),
            mask=torch.tensor(global_mask, dtype=torch.float32)[None].expand(rows, -1).clone(),
            cbits=torch.tensor(constraint_bits(self.problem), dtype=torch.float32)[None].expand(rows, -1).clone(),
            problem_desc=torch.tensor(problem_descriptor(self.problem), dtype=torch.float32)[None].expand(rows, -1).clone(),
            coord_dist=torch.full((rows,), 4, dtype=torch.long))
        shape = (rows, maximum, maximum) if self.kind == 'matrix' else (rows, maximum, self.d_in)
        values = np.zeros(shape, dtype=np.float32)
        for local, row in enumerate(indices):
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
        if self.kind == 'coord':
            inputs['node_geom'] = local_geometry(inputs['node'][..., :2], inputs['node_mask']).float()
        return inputs


def global_to_local(global_ids, pool_ids, mask=None):
    if global_ids.ndim != 1 or global_ids.dtype != torch.long or pool_ids.dtype != torch.long:
        raise ValueError('Pseudo-targets and pool IDs must be integer global IDs')
    pool_ids = pool_ids.to(global_ids.device)
    if pool_ids.ndim == 1:
        pool_ids = pool_ids[None].expand(len(global_ids), -1)
    if pool_ids.ndim != 2 or len(pool_ids) != len(global_ids):
        raise ValueError('Pseudo-target task/row alignment mismatch')
    legal = torch.ones_like(pool_ids, dtype=torch.bool) if mask is None else mask.to(
        device=global_ids.device, dtype=torch.bool)
    if legal.shape != pool_ids.shape:
        raise ValueError('Legal mask and candidate columns differ')
    matches = (pool_ids == global_ids[:, None]) & legal
    if not (matches.sum(-1) == 1).all():
        raise ValueError('Every teacher target must map to exactly one legal local column')
    return matches.long().argmax(-1)


def expansion_order(problem_id):
    generator = np.random.default_rng(730002 + problem_id)
    return np.concatenate([generator.permutation(90000) for _ in range(5)])[:400000].astype(np.int32)


def expansion_manifest():
    records = {}
    for problem in PROBLEMS:
        path = CACHE_ROOT / 'datasets' / problem / 'manifest.json'
        manifest = json.loads(path.read_text())
        records[problem] = {key: manifest[key] for key in
            ('problem', 'pool', 'feature_directory', 'count', 'source_count', 'prefix_count', 'kind', 'd_in')}
        records[problem]['manifest'] = str(path)
    return records
