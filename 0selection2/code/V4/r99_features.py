"""Ragged FP16 input-only competition sidecars for all18 frozen tasks.

Preparation reads n.npy and node.npy/matrix.npy, never a ScenarioCache or a
cost/winner array. Runtime wrapping retains the existing supervised cache.
"""

import fcntl
import hashlib
import json
import mmap
import os
from pathlib import Path
import time

import numpy as np
import torch

from code.unified_selector.registry import PROBLEMS
from .r76_experiment import read_json, write_json, utc
from .r99_relations import SPEC, WIDTH, VARIANTS, transport


FORMAT = 'r99_ragged_input_messages_v1'


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def source_identity(record):
    key = 'node' if record['kind'] == 'coord' else 'matrix'
    return dict(metadata_sha256=digest(record), arrays={
        name: record['arrays'][name] for name in ('n', key)})


def close_array(array):
    if getattr(array, '_mmap', None) is not None:
        array._mmap.close()


def authorize_test(record):
    if isinstance(record, (str, Path)):
        record = read_json(record)
    if (not isinstance(record, dict) or record.get('experiment') != 'R99'
            or record.get('arm') not in ('A', 'B')
            or record.get('selected_without_test') is not True
            or record.get('per_problem_assembly') is not False
            or not record.get('checkpoint_sha256')):
        raise PermissionError('R99 TEST messages require the unique VAL-only checkpoint lock')
    return record


def _prepare_task(cache_root, feature_root, split, problem, variant, device,
                  batch_size, manifest, data_root, test_lock):
    source = Path(cache_root) / split / problem
    record = read_json(source / 'metadata.json')
    entry = manifest['splits'][split][problem] if manifest is not None else record
    # The frozen manifest extends the original metadata with runtime locations.
    # Bind every original metadata field, not just task/shape identifiers.
    for name in record:
        if name in ('directory', 'cache_root'):
            continue
        if record.get(name) != entry.get(name):
            raise ValueError('Frozen input source identity changed: ' + name)
    if (record.get('complete') is not True or record.get('rows_written') != record.get('rows')
            or record.get('problem') != problem or record.get('split') != split
            or (data_root is not None and Path(record['data_root']).resolve() != Path(data_root).resolve())):
        raise ValueError('Use the complete frozen R84 input-only source')
    target = Path(feature_root) / split / problem
    target.mkdir(parents=True, exist_ok=True)
    identity = dict(format=FORMAT, problem=problem, split=split, variant=variant,
        transform=SPEC, source_identity=source_identity(record),
        rows=record['rows'], input_only=True, labels_opened=False,
        test_lock_sha256=digest(test_lock) if test_lock is not None else None)
    with (target / 'prepare.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        receipt = target / 'metadata.json'
        if receipt.exists():
            saved = read_json(receipt)
            if saved.get('identity') != identity or saved.get('complete') is not True:
                raise ValueError('Preserve incompatible or incomplete R99 sidecar')
            for name in ('messages.npy', 'offsets.npy'):
                if (target/name).stat().st_size != saved['files'][name]:
                    raise ValueError('R99 sidecar size changed')
            return saved
        if any((target/name).exists() for name in ('messages.npy', 'messages.partial.npy', 'offsets.npy')):
            raise RuntimeError('Preserve partial extraction; no automatic replacement')
        started = time.monotonic()
        lengths = array = output = None
        try:
            lengths = np.load(source/'n.npy', mmap_mode='r', allow_pickle=False)
            key = 'node' if record['kind'] == 'coord' else 'matrix'
            array = np.load(source/record['arrays'][key]['file'], mmap_mode='r', allow_pickle=False)
            if (lengths.dtype != np.int64 or lengths.shape != (record['rows'],)
                    or array.shape != tuple(record['arrays'][key]['shape'])
                    or array.dtype != np.float32 or (lengths < 2).any()):
                raise ValueError('Input arrays disagree with the frozen source schema')
            offsets = np.r_[np.int64(0), np.cumsum(lengths, dtype=np.int64)]
            np.save(target/'offsets.npy', offsets, allow_pickle=False)
            output = np.lib.format.open_memmap(target/'messages.partial.npy', mode='w+',
                         dtype=np.float16, shape=(int(offsets[-1]), WIDTH))
            order = np.argsort(lengths, kind='stable')
            sorted_n = lengths[order]
            start, largest_residual = 0, 0.
            while start < len(order):
                count = int(sorted_n[start])
                end = int(np.searchsorted(sorted_n, count, side='right'))
                while start < end:
                    stop = min(start+batch_size, end)
                    rows = order[start:stop]
                    batch = dict(kind=record['kind'], problem_id=record['problem_id'],
                        n=torch.full((len(rows),), count, dtype=torch.long, device=device),
                        node_mask=torch.ones(len(rows), count, dtype=torch.bool, device=device))
                    raw = array[:, :count][rows] if key == 'node' else array[:, :count, :count][rows]
                    batch[key] = torch.from_numpy(np.ascontiguousarray(raw)).to(device)
                    values, residual = transport(batch, variant)
                    values = values.cpu().numpy()
                    largest_residual = max(largest_residual, float(residual))
                    for row, value in zip(rows, values):
                        output[offsets[row]:offsets[row+1]] = value
                    start = stop
                print('[R99 inputs]', variant, split, problem, start, '/', len(order), flush=True)
            output.flush()
            close_array(output)
            output = None
            (target/'messages.partial.npy').replace(target/'messages.npy')
            value = dict(identity=identity, complete=True, completed_utc=utc(),
                source_cache_root=str(Path(cache_root)), rows_written=record['rows'],
                total_real_nodes=int(offsets[-1]), dtype='float16', width=WIDTH,
                shape=[int(offsets[-1]), WIDTH], max_marginal_residual=largest_residual,
                finite_iteration_convergence_not_assumed=True, seconds=time.monotonic()-started,
                files={name:(target/name).stat().st_size for name in ('messages.npy', 'offsets.npy')})
            write_json(receipt, value)
            return value
        finally:
            for value in (lengths, array, output):
                close_array(value)


def prepare_features(cache_root, feature_root, split, variant, device, batch_size=128,
                     manifest=None, data_root=None, test_lock=None):
    if variant not in VARIANTS or split not in ('train', 'val', 'test') or batch_size < 1:
        raise ValueError('Use a declared variant, split and positive extraction batch')
    test_lock = authorize_test(test_lock) if split == 'test' else None
    if isinstance(manifest, (str, Path)):
        manifest = read_json(manifest)
    return {problem: _prepare_task(cache_root, feature_root, split, problem, variant,
        device, batch_size, manifest, data_root, test_lock) for problem in PROBLEMS}


class CompetitionFeatureCache:
    def __init__(self, base_cache, feature_directory, expected_variant):
        self.base_cache = base_cache
        self.directory_features = Path(feature_directory)
        self.metadata_features = read_json(self.directory_features/'metadata.json')
        identity = self.metadata_features.get('identity', {})
        expected = dict(format=FORMAT, problem=base_cache.problem, split=base_cache.split,
                        variant=expected_variant, transform=SPEC,
                        source_identity=source_identity(base_cache.metadata), rows=base_cache.size,
                        input_only=True, labels_opened=False)
        if (self.metadata_features.get('complete') is not True
                or any(identity.get(key) != value for key, value in expected.items())):
            raise ValueError('R99 messages do not belong to this input cache/variant')
        if base_cache.split == 'test' and not identity.get('test_lock_sha256'):
            raise PermissionError('TEST input messages were not built after a VAL lock')
        self.offsets = np.load(self.directory_features/'offsets.npy', mmap_mode='r', allow_pickle=False)
        self.messages = np.load(self.directory_features/'messages.npy', mmap_mode='r', allow_pickle=False)
        if (self.offsets.dtype != np.int64 or self.offsets.shape != (base_cache.size+1,)
                or not np.array_equal(np.diff(self.offsets), base_cache.lengths)
                or self.messages.dtype != np.float16
                or self.messages.shape != (int(self.offsets[-1]), WIDTH)):
            raise ValueError('Ragged feature offsets do not align with the original nodes')
        self.messages._mmap.madvise(mmap.MADV_RANDOM)
        self.fd = os.open(self.directory_features/'messages.npy', os.O_RDONLY)
        os.posix_fadvise(self.fd, 0, 0, os.POSIX_FADV_RANDOM)
        self.closed = False

    def __getattr__(self, name):
        return getattr(self.base_cache, name)

    def batch(self, indices, device='cpu'):
        rows = np.asarray(indices, dtype=np.int64)
        if self.closed or rows.ndim != 1 or len(rows) == 0:
            raise ValueError('Expected a live feature cache and nonempty CPU row indices')
        batch = self.base_cache.batch(rows, device=device)
        maximum = batch['node_mask'].shape[1]
        for row in np.unique(rows):
            os.posix_fadvise(self.fd, int(self.messages.offset)+int(self.offsets[row])*WIDTH*2,
                            int(self.offsets[row+1]-self.offsets[row])*WIDTH*2,
                            os.POSIX_FADV_WILLNEED)
        valid = np.arange(maximum)[None] < np.asarray(self.lengths[rows])[:, None]
        positions = self.offsets[rows, None] + np.arange(maximum)[None]
        values = np.ascontiguousarray(self.messages[np.where(valid, positions, 0)])
        values[~valid] = 0.
        batch['relation_nodes'] = torch.from_numpy(values).to(batch['node_mask'].device)
        batch['relation_node_mask'] = batch['node_mask']
        return batch

    def close(self):
        if not self.closed:
            os.close(self.fd)
            close_array(self.offsets)
            close_array(self.messages)
            self.base_cache.close()
            self.closed = True
