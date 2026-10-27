"""Input-only native sidecars, with one shared pretrained routing base cache.

Only metadata, lengths and raw node/matrix arrays are opened while extracting.
Native rows keep their original identities; TEST extraction requires a VAL lock.
Partial extraction is retained as incomplete, never silently regenerated.
"""

import copy
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
from .r117_model import WIDTH_BY_PROBLEM
from .r76_experiment import read_json, write_json, utc
from .r94_donor import CHECKPOINT_SHA256, SPEC as BASE_SPEC


FORMAT = 'r117_native_node_features_v1'
COMMON_FORMAT = 'r117_shared_pretrained_reld_nodes_v1'
FEATURE_FILE = 'native_nodes.npy'
BASE_FILE = 'nco_nodes.npy'
BASE_IDENTITY = dict(checkpoint_sha256=CHECKPOINT_SHA256, spec=BASE_SPEC,
    precision='FP32 native encoder quantized to FP16 before extra decoder and selector')


def _digest(record):
    return hashlib.sha256(json.dumps(record, sort_keys=True,
        separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def _test_permission(split, lock):
    if split != 'test':
        return None
    if (not isinstance(lock, dict) or lock.get('experiment') != 'R117'
            or lock.get('selected_without_test') is not True
            or lock.get('per_problem_assembly') is not False
            or lock.get('selection_split') != 'val'
            or not lock.get('checkpoint_sha256') or not lock.get('inference_rule')):
        raise PermissionError('TEST native features require this arm\'s unique R117 VAL lock')
    return _digest(lock)


def _open(path, shape, dtype='float16'):
    value = np.load(path, mmap_mode='r', allow_pickle=False)
    if value.shape != tuple(shape) or value.dtype != np.dtype(dtype):
        value._mmap.close()
        raise ValueError('Frozen native feature shape/dtype mismatch: ' + str(path))
    value._mmap.madvise(mmap.MADV_RANDOM)
    return value


def _source(cache_root, split, problem, manifest, data_root):
    directory = Path(cache_root) / split / problem
    record = read_json(directory / 'metadata.json')
    reference = manifest['splits'][split][problem]
    for field in ('problem', 'split', 'rows', 'kind', 'd_in', 'true_length_max',
                  'true_length_min', 'scenario_id', 'pool', 'pool_ids', 'arrays'):
        if record.get(field) != reference.get(field):
            raise ValueError('Frozen input header mismatch: ' + problem + '/' + field)
    if (record.get('complete') is not True or record.get('rows_written') != record['rows']
            or Path(record['data_root']).resolve() != Path(data_root).resolve()):
        raise ValueError('Require existing complete frozen source inputs')
    return directory, record


def _contract(record, provenance, test_sha, common=False):
    return dict(format=COMMON_FORMAT if common else FORMAT, problem=record['problem'],
        split=record['split'], rows=record['rows'], shape=[record['rows'],
            record['true_length_max'], 128 if common else WIDTH_BY_PROBLEM[record['problem']]],
        dtype='float16', input_only=True, labels_opened=False,
        file=BASE_FILE if common else FEATURE_FILE, source_metadata_sha256=_digest(record),
        donor=copy.deepcopy(BASE_IDENTITY if common else provenance),
        test_lock_sha256=None if common else test_sha,
        extraction='stable valid-N order, then original row order; physical node identities retained')


def _reuse(directory, contract):
    path = directory / 'metadata.json'
    if not path.exists():
        return None
    prior = read_json(path)
    if (any(prior.get(key) != value for key, value in contract.items())
            or prior.get('complete') is not True or prior.get('rows_written') != contract['rows']):
        raise RuntimeError('Incomplete or different native sidecar retained: ' + str(directory))
    array = _open(directory / contract['file'], contract['shape'])
    array._mmap.close()
    return prior


def _create(directory, contract):
    directory.mkdir(parents=True, exist_ok=True)
    partial = directory / ('.' + contract['file'] + '.partial')
    if partial.exists() or (directory / contract['file']).exists():
        raise RuntimeError('Untracked native array retained: ' + str(directory))
    write_json(directory / 'metadata.json', dict(contract, complete=False, rows_written=0))
    return np.lib.format.open_memmap(partial, mode='w+', dtype=np.float16, shape=tuple(contract['shape']))


def _finish(directory, contract, array):
    array.flush()
    array._mmap.close()
    os.replace(directory / ('.' + contract['file'] + '.partial'), directory / contract['file'])
    value = dict(contract, complete=True, rows_written=contract['rows'], completed_utc=utc())
    write_json(directory / 'metadata.json', value)
    return value


def _extra(donor, batch, base):
    value = donor.forward_extra(batch, base_nodes=base)
    return value['native_nodes'] if isinstance(value, dict) else value


def _prepare_task(source, record, target, common, donor, device, batch_size, provenance, test_sha):
    contract = _contract(record, provenance, test_sha)
    target.mkdir(parents=True, exist_ok=True)
    routing = record['problem'] not in ('TSP', 'ATSP')
    with (target / '.prepare.lock').open('a') as own_lock:
        fcntl.flock(own_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        prior = _reuse(target, contract)
        if prior is not None:
            return prior
        common.mkdir(parents=True, exist_ok=True)
        with (common / '.prepare.lock').open('a') as base_lock:
            fcntl.flock(base_lock, fcntl.LOCK_EX)
            base_contract = _contract(record, provenance, test_sha, common=True)
            base_prior = _reuse(common, base_contract) if routing else None
            base = (_open(common / BASE_FILE, base_contract['shape']) if base_prior else None)
            base_out = _create(common, base_contract) if routing and base is None else None
            # An existing shared base is immutable; release its writer lock now.
            if base_out is None:
                fcntl.flock(base_lock, fcntl.LOCK_UN)
            output = _create(target, contract)
            lengths, values = None, None
            try:
                lengths = _open(source / 'n.npy', [record['rows']], 'int64')
                key = 'matrix' if record['kind'] == 'matrix' else 'node'
                spec = record['arrays'][key]
                values = _open(source / spec['file'], spec['shape'], spec['dtype'])
                order = np.argsort(lengths, kind='stable')
                sorted_lengths = lengths[order]
                cursor, logged = 0, 0
                while cursor < len(order):
                    n = int(sorted_lengths[cursor])
                    stop = min(cursor + batch_size,
                        int(np.searchsorted(sorted_lengths, n, side='right')))
                    rows = order[cursor:stop]
                    raw = values[rows, :n, :n] if key == 'matrix' else values[rows, :n]
                    batch = dict(problem_id=record['problem_id'], problem_name=record['problem'],
                        kind=record['kind'], d_in=record['d_in'],
                        n=torch.full((len(rows),), n, dtype=torch.long, device=device),
                        node_mask=torch.ones(len(rows), n, dtype=torch.bool, device=device),
                        **{key: torch.from_numpy(np.array(raw, copy=True)).to(device)})
                    with torch.no_grad():
                        if base is not None:
                            base_values = torch.from_numpy(np.array(base[rows, :n], copy=True)).to(device).float()
                            native = _extra(donor, batch, base_values)
                        else:
                            extracted = donor(batch)
                            native = extracted['native_nodes']
                            if base_out is not None:
                                base_out[rows] = 0.
                                base_out[rows, :n] = extracted['base_nodes'].float().cpu().numpy().astype(np.float16)
                    if (native.shape != (len(rows), n, contract['shape'][-1])
                            or not torch.isfinite(native).all()):
                        raise ValueError('Nonfinite or misaligned frozen native output')
                    output[rows] = 0.
                    output[rows, :n] = native.float().cpu().numpy().astype(np.float16)
                    cursor = stop
                    if cursor - logged >= 8192 or cursor == len(order):
                        logged = cursor
                        print(f'[R117 native] {record["split"]}/{record["problem"]} {cursor}/{len(order)}', flush=True)
                if base_out is not None:
                    _finish(common, base_contract, base_out)
                    base_out = None
                result = _finish(target, contract, output)
                output = None
                return result
            finally:
                for array in (lengths, values, base, base_out, output):
                    if array is not None:
                        array._mmap.close()


def prepare_features(cache_root, feature_root, split, donor, device, *, common_root,
                     batch_size=128, manifest=None, data_root=None, test_lock=None):
    if split not in ('train', 'val', 'test') or not 1 <= batch_size <= 256:
        raise ValueError('Use a frozen split and bounded native extraction batch')
    test_sha = _test_permission(split, test_lock)
    manifest = read_json(manifest) if isinstance(manifest, (str, Path)) else manifest
    if not isinstance(manifest, dict) or data_root is None:
        raise ValueError('Native preparation requires the frozen source manifest and data root')
    provenance, records = donor.provenance(), {}
    donor.to(device).eval()
    started = time.monotonic()
    for problem in PROBLEMS:
        source, record = _source(cache_root, split, problem, manifest, data_root)
        target, common = Path(feature_root) / split / problem, Path(common_root) / split / problem
        result = _prepare_task(source, record, target, common, donor, device,
            batch_size, provenance, test_sha)
        records[problem] = dict(result, directory=str(target), width=WIDTH_BY_PROBLEM[problem])
    return dict(format=FORMAT, complete=True, splits={split: records}, donor=provenance,
        labels_opened=False, actions_selected=False, seconds=time.monotonic() - started)


class NativeFeatureCache:
    """Original ScenarioCache supervision plus input-only aligned native features."""

    def __init__(self, base_cache, feature_directory, common_directory, *, expected_provenance):
        self.base_cache = base_cache
        self._maps, self._fds, self._closed = {}, {}, False
        record = base_cache.metadata
        width = WIDTH_BY_PROBLEM[base_cache.problem]
        directory = Path(feature_directory)
        metadata = read_json(directory / 'metadata.json')
        expected = _contract(record, expected_provenance, metadata.get('test_lock_sha256'))
        if base_cache.split == 'test' and not metadata.get('test_lock_sha256'):
            raise PermissionError('TEST native sidecar has no unique VAL authorization')
        _reuse(directory, expected)
        self.feature_metadata = metadata
        specs = [('native_nodes', directory / FEATURE_FILE, [base_cache.size, record['true_length_max'], width])]
        if base_cache.problem not in ('TSP', 'ATSP'):
            common = Path(common_directory)
            common_expected = _contract(record, expected_provenance, None, common=True)
            _reuse(common, common_expected)
            specs.append(('nco_nodes', common / BASE_FILE,
                [base_cache.size, record['true_length_max'], 128]))
        try:
            for key, path, shape in specs:
                self._maps[key] = _open(path, shape)
                self._fds[key] = os.open(path, os.O_RDONLY)
                os.posix_fadvise(self._fds[key], 0, 0, os.POSIX_FADV_RANDOM)
        except BaseException:
            self.close()
            raise

    def __getattr__(self, name):
        return getattr(self.base_cache, name)

    def batch(self, indices, device='cpu'):
        if self._closed:
            raise RuntimeError('Closed native feature cache')
        rows = np.asarray(indices.cpu() if torch.is_tensor(indices) else indices, dtype=np.int64)
        result = self.base_cache.batch(rows, device=device)
        mask = result['node_mask']
        for key, array in self._maps.items():
            wanted = mask.shape[1] * array.shape[-1] * array.dtype.itemsize
            for row in np.unique(rows):
                os.posix_fadvise(self._fds[key], int(array.offset) + int(row) * array.strides[0],
                    wanted, os.POSIX_FADV_WILLNEED)
            value = torch.from_numpy(np.ascontiguousarray(array[:, :mask.shape[1]][rows])).to(mask.device)
            result[key] = value.masked_fill_(~mask[..., None], 0.)
        if 'nco_nodes' not in result:
            result['nco_nodes'] = torch.zeros(*mask.shape, 128, dtype=torch.float16, device=mask.device)
        result['nco_node_mask'] = result['native_node_mask'] = mask
        return result

    def close(self):
        if not self._closed:
            for fd in self._fds.values():
                os.close(fd)
            for array in self._maps.values():
                array._mmap.close()
            self._fds.clear()
            self._maps.clear()
            self.base_cache.close()
            self._closed = True
