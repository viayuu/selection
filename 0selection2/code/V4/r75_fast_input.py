"""Lossless, compact anonymous-RAM input for existing complete R75 caches.

``estimate_resident_bytes`` scans only lengths and includes packed features,
scalar arrays and int64 offsets. It excludes bounded loading scratch space,
batch tensors, pinned staging and Python/allocator overhead. No global cache is
kept. Close old instances before opening another selected-training cache.

``batch_cpu(indices)``, ``pin_batch(cpu_batch)`` and
``transfer_pinned(pinned_batch, device)`` expose separate profiling boundaries.
``batch_gpu(cpu_batch, device)`` combines pin/transfer; ``batch(indices, device)``
combines all three. CUDA staging is allocated once per key at its configured
maximum capacity, never for the full dataset.
Calls on an individual instance must be serialized, as with ProblemCache.
"""

import operator

import numpy as np
import torch

from .r75_cache import ProblemCache, _row_indices


DEFAULT_CHUNK_BYTES = 32 * 1024 * 1024
DEFAULT_MAX_BATCH_SIZE = 384
_SCALAR_KEYS = ('n', 'costs', 'winner', 'coord_dist')


def _positive(value, name):
    value = operator.index(value)
    if value <= 0:
        raise ValueError(f'{name} must be positive')
    return value


def _chunks(array, chunk_bytes):
    """Read consecutive file rows without faulting the source mmap into RSS."""
    row_elements = int(np.prod(array.shape[1:], dtype=np.int64))
    row_bytes = row_elements * array.dtype.itemsize
    chunk_rows = max(1, chunk_bytes // max(1, row_bytes))
    with open(array.filename, 'rb') as stream:
        stream.seek(array.offset)
        for start in range(0, len(array), chunk_rows):
            stop = min(start + chunk_rows, len(array))
            count = (stop - start) * row_elements
            block = np.fromfile(stream, dtype=array.dtype, count=count)
            if block.size != count:
                raise ValueError(f'{array.filename}: truncated cache array')
            block = block.reshape((stop - start,) + array.shape[1:])
            yield start, block
            del block


def _units(lengths, source):
    if np.any(lengths < 0) or np.any(lengths > source.metadata['true_length_max']):
        raise ValueError(f'{source.directory}: invalid true lengths')
    return lengths if source.kind == 'coord' else np.square(lengths)


def _resident_size(source, units):
    scalar_bytes = sum(source._arrays[key].nbytes for key in _SCALAR_KEYS)
    offset_bytes = (source.size + 1) * np.dtype('int64').itemsize
    if source.kind == 'coord':
        unit_bytes = sum(source._arrays[key].shape[2] * source._arrays[key].dtype.itemsize
                         for key in ('node', 'node_geom'))
    else:
        unit_bytes = source._arrays['matrix'].dtype.itemsize
    return int(scalar_bytes + offset_bytes + units * unit_bytes)


def estimate_resident_bytes(directory, *, chunk_bytes=DEFAULT_CHUNK_BYTES):
    """Estimate owned NumPy bytes before loading features; only read ``n.npy``."""
    chunk_bytes = _positive(chunk_bytes, 'chunk_bytes')
    source = ProblemCache(directory)
    try:
        units = 0
        for _, lengths in _chunks(source.lengths, chunk_bytes):
            units += int(_units(lengths, source).sum(dtype=np.int64))
            del lengths
        return _resident_size(source, units)
    finally:
        source.close()


def _copy_array(array, chunk_bytes):
    result = np.empty(array.shape, dtype=array.dtype)
    for start, block in _chunks(array, chunk_bytes):
        result[start:start + len(block)] = block
        del block
    return result


class FastProblemCache:
    """ProblemCache-compatible batches backed by losslessly packed CPU arrays.

    ``max_resident_bytes`` is an optional *per-instance* allocation guard. The
    parent must sum estimates across TRAIN/VAL itself. GPU batches are bounded
    by ``max_batch_size``; CPU gathers retain the original unrestricted API.
    """

    estimate_resident_bytes = staticmethod(estimate_resident_bytes)

    def __init__(self, directory, *, chunk_bytes=DEFAULT_CHUNK_BYTES,
                 max_resident_bytes=None, max_batch_size=DEFAULT_MAX_BATCH_SIZE):
        chunk_bytes = _positive(chunk_bytes, 'chunk_bytes')
        self.max_batch_size = _positive(max_batch_size, 'max_batch_size')
        if max_resident_bytes is not None:
            max_resident_bytes = operator.index(max_resident_bytes)
            if max_resident_bytes < 0:
                raise ValueError('max_resident_bytes must be nonnegative')
        self._arrays, self.raw, self._pinned = {}, {}, {}
        self._offsets = self.lengths = self._copy_event = self._copy_device = None
        self.resident_bytes = 0
        source = ProblemCache(directory)
        try:
            self.directory, self.metadata = source.directory, source.metadata
            self.problem, self.split = source.problem, source.split
            self.size, self.kind = source.size, source.kind
            self.pool, self.pool_ids = source.pool, source.pool_ids
            self.lengths = _copy_array(source.lengths, chunk_bytes)
            self._arrays['n'] = self.lengths
            self._offsets = np.empty(self.size + 1, dtype=np.int64)
            self._offsets[0] = 0
            chunk_rows = max(1, chunk_bytes // np.dtype('int64').itemsize)
            cursor = 0
            for start in range(0, self.size, chunk_rows):
                stop = min(start + chunk_rows, self.size)
                units = _units(self.lengths[start:stop], source)
                np.cumsum(units, dtype=np.int64, out=self._offsets[start + 1:stop + 1])
                self._offsets[start + 1:stop + 1] += cursor
                cursor = int(self._offsets[stop])
            estimated = _resident_size(source, cursor)
            if max_resident_bytes is not None and estimated > max_resident_bytes:
                raise MemoryError(f'{self.directory}: {estimated} resident bytes exceed '
                                  f'per-cache limit {max_resident_bytes}')
            for key in _SCALAR_KEYS[1:]:
                self._arrays[key] = _copy_array(source._arrays[key], chunk_bytes)
            keys = ('node', 'node_geom') if self.kind == 'coord' else ('matrix',)
            for key in keys:
                original = source._arrays[key]
                shape = (cursor, original.shape[2]) if self.kind == 'coord' else (cursor,)
                packed = np.empty(shape, dtype=original.dtype)
                self._arrays[key] = packed
                for start, block in _chunks(original, chunk_bytes):
                    for local, row in enumerate(block):
                        index = start + local
                        n = int(self.lengths[index])
                        left, right = self._offsets[index:index + 2]
                        # Refuse to discard nonzero padding, including NaN/Inf.
                        if np.any(row[n:]) or (self.kind != 'coord' and np.any(row[:n, n:])):
                            raise ValueError(f'{self.directory}: {key} row {index} has nonzero padding')
                        if self.kind == 'coord':
                            packed[left:right] = row[:n]
                        else:
                            packed[left:right].reshape(n, n)[:] = row[:n, :n]
                    del row, block
            self.raw = dict(costs=self._arrays['costs'], winner=self._arrays['winner'],
                            pool=self.pool, pool_ids=self.pool_ids)
            self.resident_bytes = sum(array.nbytes for array in self._arrays.values()) + self._offsets.nbytes
            if self.resident_bytes != estimated:
                raise RuntimeError('Packed allocation disagrees with resident estimate')
            self._pinned_capacities = self._staging_capacities()
        except BaseException:
            self.close()
            raise
        finally:
            source.close()

    def _staging_capacities(self):
        b, n, record = self.max_batch_size, self.metadata['true_length_max'], self.metadata
        capacities = dict(n=b, node_mask=b * n, cbits=b * len(record['cbits']),
                          problem_desc=b * len(record['problem_desc']), coord_dist=b,
                          pool_ids=len(self.pool_ids), mask=b * len(record['mask']),
                          costs=b * len(self.pool), ind=b)
        if self.kind == 'coord':
            capacities.update(node=b * n * record['d_in'],
                              node_geom=b * n * self._arrays['node_geom'].shape[1])
        else:
            capacities['matrix'] = b * n * n
        return capacities

    @property
    def pinned_bytes(self):
        """Live owned staging bytes, separate from ``resident_bytes``."""
        return sum(value.numel() * value.element_size() for value in self._pinned.values())

    def batch_cpu(self, indices):
        """Gather original-order rows and restore strictly zero CPU padding."""
        if self._offsets is None:
            raise RuntimeError('FastProblemCache is closed')
        rows = _row_indices(indices, self.size)
        n = torch.from_numpy(self.lengths[rows])
        maximum, count = int(n.max()), len(rows)
        record = self.metadata
        batch = dict(
            problem_id=record['problem_id'], kind=self.kind, n=n,
            node_mask=torch.arange(maximum)[None, :] < n[:, None],
            cbits=torch.tensor(record['cbits'], dtype=torch.float32).repeat(count, 1),
            problem_desc=torch.tensor(record['problem_desc'], dtype=torch.float32).repeat(count, 1),
            coord_dist=torch.from_numpy(self._arrays['coord_dist'][rows]),
            pool_ids=torch.tensor(self.pool_ids, dtype=torch.long),
            mask=torch.tensor(record['mask'], dtype=torch.float32).repeat(count, 1),
            costs=torch.from_numpy(self.raw['costs'][rows].astype(np.float32)),
            ind=torch.from_numpy(self.raw['winner'][rows]))
        if self.kind == 'coord':
            batch['d_in'] = record['d_in']
            keys = ('node', 'node_geom')
        else:
            keys = ('matrix',)
        for key in keys:
            packed = self._arrays[key]
            shape = ((count, maximum, packed.shape[1]) if self.kind == 'coord'
                     else (count, maximum, maximum))
            restored = np.zeros(shape, dtype=packed.dtype)
            for position, index in enumerate(rows):
                length = int(self.lengths[index])
                left, right = self._offsets[index:index + 2]
                if self.kind == 'coord':
                    restored[position, :length] = packed[left:right]
                else:
                    restored[position, :length, :length] = packed[left:right].reshape(length, length)
            batch[key] = torch.from_numpy(restored)
        return batch

    def pin_batch(self, cpu_batch):
        """Copy a gathered batch into bounded, reusable pinned host views.

        Wait on the previous H2D event before overwriting any staging buffer.
        Returned views are valid only until the next pin_batch/close call;
        transfer them before pinning another batch on this instance.
        """
        if self._offsets is None:
            raise RuntimeError('FastProblemCache is closed')
        if len(cpu_batch['n']) > self.max_batch_size:
            raise ValueError(f'GPU batch exceeds max_batch_size={self.max_batch_size}')
        for key, value in cpu_batch.items():
            if torch.is_tensor(value):
                if value.device.type != 'cpu' or value.numel() > self._pinned_capacities.get(key, -1):
                    raise ValueError(f'{key}: not a CPU batch within configured staging capacity')
                if key in self._pinned and self._pinned[key].dtype != value.dtype:
                    raise ValueError(f'{key}: staging dtype changed')
        if self._copy_event is not None:
            self._copy_event.synchronize()
        result = {}
        for key, value in cpu_batch.items():
            if not torch.is_tensor(value):
                result[key] = value
                continue
            if key not in self._pinned:
                self._pinned[key] = torch.empty(
                    self._pinned_capacities[key], dtype=value.dtype, device='cpu', pin_memory=True)
            staging = self._pinned[key][:value.numel()].view(value.shape)
            staging.copy_(value)
            result[key] = staging
        return result

    def transfer_pinned(self, pinned_batch, device='cuda:0'):
        """Submit H2D only; return independent device tensors and record completion.

        The event also protects staging after a partial-transfer exception.
        No gather, host copy or CPU synchronization occurs in the CUDA path.
        """
        if self._offsets is None:
            raise RuntimeError('FastProblemCache is closed')
        device = torch.device(device)
        if device.type == 'cpu':
            return pinned_batch
        if device.type != 'cuda':
            return {key: value.to(device) if torch.is_tensor(value) else value
                    for key, value in pinned_batch.items()}
        stream = torch.cuda.current_stream(device)
        if self._copy_event is None or self._copy_device != stream.device:
            self._copy_event = torch.cuda.Event(enable_timing=False)
            self._copy_device = stream.device
        result = {}
        try:
            for key, value in pinned_batch.items():
                result[key] = value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        finally:
            self._copy_event.record(stream)
        return result

    def batch_gpu(self, cpu_batch, device='cuda:0'):
        """Combine pin/transfer without gathering an already-built CPU batch."""
        if self._offsets is None:
            raise RuntimeError('FastProblemCache is closed')
        device = torch.device(device)
        if device.type == 'cuda':
            return self.transfer_pinned(self.pin_batch(cpu_batch), device)
        return self.transfer_pinned(cpu_batch, device)

    def batch(self, indices, device='cuda:0'):
        """ProblemCache-compatible gather followed by batch-only transfer."""
        return self.batch_gpu(self.batch_cpu(indices), device)

    def close(self):
        """Release owned RAM/staging references, waiting only for own H2D work.

        External references to raw arrays must also be dropped by their owner.
        PyTorch may retain freed pinned blocks in its bounded allocator cache.
        """
        try:
            if self._copy_event is not None:
                self._copy_event.synchronize()
        finally:
            self._copy_event = self._copy_device = None
            self._pinned.clear()
            self.raw.clear()
            self._arrays.clear()
            self._offsets = self.lengths = None
            self.resident_bytes = 0

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
