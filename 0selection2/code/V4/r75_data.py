"""CPU-backed native rows for R75 capacity measurements and disk caches.

Raw FP64 costs stay on the dataset; batches retain the unified loader's FP32
costs and native integer winners. No sampling, augmentation, or global cache.
"""

import operator
import pickle
from collections.abc import Mapping
from io import BytesIO
from pathlib import Path

import numpy as np
import torch

from code.unified_selector.data import UnifiedProblemDataset, collate_single_problem
from code.unified_selector.registry import (
    P2I, POOLS, constraint_bits, problem_descriptor, problem_to_pool_mask,
)
from .local_geometry import FIELDS, local_geometry


DEFAULT_DATA_ROOT = Path(
    '/public/home/shiys/'
    '\u8bad\u7ec3\u6570\u636e_\u516c\u5171\u5b8c\u6574\u6807\u7b7e_20261013_1923'
)


def _file_identity(path):
    stat = path.stat()
    return dict(path=str(path), size_bytes=stat.st_size, mtime_ns=stat.st_mtime_ns)


def _check_split(split, root=None):
    if split not in ('train', 'val', 'test'):
        raise ValueError(f'Unknown R75 split: {split}')
    if split == 'test' and (root is None or not (Path(root) / 'final_test_lock.json').is_file()):
        raise PermissionError('R75 TEST requires root/final_test_lock.json before opening data')


def _source_paths(problem, split, data_root, root=None):
    _check_split(split, root)
    if problem not in POOLS:
        raise ValueError(f'Unknown problem: {problem}')
    directory = Path(data_root).expanduser().resolve() / f'{problem}{split}'
    result_files = sorted((directory / 'results').glob('result_*.txt'))
    expected_names = [f'result_{name}.txt' for name in POOLS[problem]]
    if ([path.name for path in result_files] != expected_names
            or not all(path.is_file() for path in result_files)):
        raise ValueError(f'{directory}: sorted result filenames must exactly match registry POOLS')
    return [directory / 'dataset.pkl', directory / 'raw_label.pkl', *result_files]


def _cpu_storage(data):
    return torch.load(BytesIO(data), map_location='cpu', weights_only=False)


class _CPUUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return _cpu_storage
        return super().find_class(module, name)


def _node_count(problem, instance):
    if problem == 'TSP':
        return len(instance[0])
    if problem == 'ATSP':
        return int(instance.shape[-2])
    if problem == 'CVRP':
        return len(instance['loc'][0]) + 1
    return len(instance['node_xy']) + 1


class RealTrainDataset(UnifiedProblemDataset):
    """Use inherited feature extraction without changing either DATA_ROOT.

    ``raw_costs`` is an aligned CPU NumPy float64 array, indexed by real row
    position. String instance IDs are accepted by ``__getitem__`` as well.
    """

    def __init__(self, problem, split='train', data_root=DEFAULT_DATA_ROOT, *, root=None):
        self.problem, self.split = str(problem).upper(), str(split).lower()
        self.data_root = Path(data_root).expanduser().resolve()
        source_files = _source_paths(self.problem, self.split, self.data_root, root)
        source_identities = [_file_identity(path) for path in source_files]
        directory = source_files[0].parent
        self.pool = tuple(POOLS[self.problem])

        with (directory / 'dataset.pkl').open('rb') as stream:
            self.instances = _CPUUnpickler(stream).load()
        with (directory / 'raw_label.pkl').open('rb') as stream:
            raw = _CPUUnpickler(stream).load()
        self.base_N = self.N = self.size = len(self.instances)
        self.instance_ids = tuple(str(i) for i in range(self.size))
        if (not self.size or len(raw) != self.size
                or any(instance_id not in raw for instance_id in self.instance_ids)):
            raise ValueError(f'{directory}: nonempty rows and string label IDs must align')
        self.labels = {key: dict(cost=raw[key]['cost'], ind=raw[key]['ind'])
                       for key in self.instance_ids}
        del raw

        self.pool_order, self.mask = problem_to_pool_mask(self.problem)
        self.cbits = constraint_bits(self.problem)
        self.problem_desc = problem_descriptor(self.problem)
        self.pid, self.K_p, self.coord_augment = P2I[self.problem], len(self.pool), 0
        self.raw_costs = np.asarray([self.labels[key]['cost'] for key in self.instance_ids],
                                   dtype=np.float64)
        native_ind = np.asarray([self.labels[key]['ind'] for key in self.instance_ids])
        if (self.raw_costs.shape != (self.size, self.K_p)
                or not np.isfinite(self.raw_costs).all()):
            raise ValueError(f'{directory}: each label must contain exactly {self.K_p} finite costs')
        if (native_ind.shape != (self.size,)
                or not np.issubdtype(native_ind.dtype, np.integer)
                or np.any(native_ind < 0) or np.any(native_ind >= self.K_p)):
            raise ValueError(f'{directory}: native ind must be an integer legal-pool slot')
        # Never derive winners from FP32 costs, including exact/rounding ties.
        self.raw_costs.setflags(write=False)
        self.raw_winner = native_ind.astype(np.int64, copy=False)
        self.raw_winner.setflags(write=False)
        self.lengths = np.fromiter((_node_count(self.problem, inst) for inst in self.instances),
                                   dtype=np.int64, count=self.size)
        if np.any(self.lengths <= 0):
            raise ValueError(f'{directory}: node counts must be positive')
        self.lengths.setflags(write=False)
        self.kind = 'matrix' if self.problem == 'ATSP' else 'coord'
        self.d_in = None if self.kind == 'matrix' else (
            2 if self.problem == 'TSP' else 4 if self.problem == 'CVRP' else 8)
        self.metadata = dict(
            problem=self.problem, problem_id=self.pid, split=self.split,
            data_root=str(self.data_root), rows=self.size,
            true_length_min=int(self.lengths.min()), true_length_max=int(self.lengths.max()),
            length_definition='model node count; includes depot for CVRP/MVRP',
            kind=self.kind, d_in=self.d_in, pool=list(self.pool), pool_ids=list(self.pool_order),
            raw_cost_dtype=str(self.raw_costs.dtype), winner_source='native raw_label ind; no relabeling',
            coord_augment=0,
            coord_dist='inherited index-based approximation; unchanged, not measured metadata',
            source_files=source_identities,
        )

    def _row_index(self, index):
        if isinstance(index, str):
            if index not in self.labels:
                raise KeyError(f'Unknown {self.split.upper()} instance ID: {index}')
            index = int(index)
        index = operator.index(index)
        if not 0 <= index < self.size:
            raise IndexError(f'{self.split.upper()} row {index} outside [0, {self.size})')
        return index

    def __getitem__(self, index):
        return super().__getitem__(self._row_index(index))


def load_dataset(problem, split='train', data_root=DEFAULT_DATA_ROOT, *, root=None):
    """Load one CPU problem; TEST additionally requires the run's ``root`` lock.

    Existing capacity calls still default to TRAIN without opening other splits.
    """
    return RealTrainDataset(problem, split=split, data_root=data_root, root=root)


def lengths(dataset):
    """Read-only NumPy int64 true model node counts, without sample decoding."""
    return dataset.lengths


def dataset_sizes(datasets):
    """Metadata for one dataset, or metadata by problem for a mapping/iterable.

    Uses only already loaded datasets; does not open additional splits/tasks.
    """
    if isinstance(datasets, RealTrainDataset):
        return dict(datasets.metadata)
    if isinstance(datasets, Mapping):
        datasets = datasets.values()
    return {dataset.problem: dict(dataset.metadata) for dataset in datasets}


def build_batch(dataset, indices, device='cpu', include_geometry=True, geometry_batch_size=32):
    """Collate only requested real rows, then transfer only this batch.

    ``indices`` is a sequence of integer positions or string instance IDs.
    Coordinate geometry is computed on CPU in bounded chunks, using the same
    FP64 local_geometry API and FP32 node_geom cache format as R65B.
    """
    if torch.is_tensor(indices):
        if indices.device.type != 'cpu' or indices.ndim != 1:
            raise ValueError('Row indices must be a one-dimensional CPU sequence')
        indices = indices.tolist()
    elif isinstance(indices, str):
        indices = [indices]
    rows = [dataset._row_index(index) for index in indices]
    if not rows:
        raise ValueError('Cannot collate an empty batch')
    if include_geometry:
        geometry_batch_size = operator.index(geometry_batch_size)
        if geometry_batch_size <= 0:
            raise ValueError('geometry_batch_size must be positive')

    # The existing collator pads to this selection's actual max n, not the pool's.
    batch = collate_single_problem([dataset[index] for index in rows])
    if include_geometry and batch['kind'] == 'coord':
        maximum = batch['node'].shape[1]
        batch['node_geom'] = torch.zeros(len(rows), maximum, len(FIELDS), dtype=torch.float32)
        for start in range(0, len(rows), geometry_batch_size):
            stop = min(start + geometry_batch_size, len(rows))
            chunk_max = int(batch['n'][start:stop].max())
            batch['node_geom'][start:stop, :chunk_max] = local_geometry(
                batch['node'][start:stop, :chunk_max, :2],
                batch['node_mask'][start:stop, :chunk_max],
            ).float()
    return {key: value.to(device) if torch.is_tensor(value) else value
            for key, value in batch.items()}
