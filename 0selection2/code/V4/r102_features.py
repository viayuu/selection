"""Frozen full18 R102 nodes: native TSP/ATSP and read-only R94B routing reuse.

Preparation opens only cache metadata/n and native node/matrix inputs. It
never constructs a ScenarioCache or opens datasets, costs, winners or labels.
No routing encoder is run here. Incomplete or incompatible outputs are kept
and rejected, not automatically resumed, replaced or regenerated.
"""

import argparse
import copy
import fcntl
import hashlib
import mmap
import operator
import os
from pathlib import Path

import numpy as np
import torch

from .r102_donor import (FullTypeDonor, PROBLEM_NAMES, ROUTING_PROBLEMS,
                        SPEC, SUPPORTED_PROBLEMS, VARIANTS, WIDTH_BY_PROBLEM)
from .r102_transfer import (COPY_RECEIPT, FEATURE_FILE,
    _validate_r94, authorize_test, digest, file_hash, npy_header, read_json,
    validate_copy_receipt, validate_frozen_header, write_new_json)


FEATURE_FORMAT = 'r102_frozen_full_type_nodes_v1'
PARTIAL_FILE = '.nco_nodes.partial.npy'
EXTRACT_ORDER = 'stable_true_n_ascending_then_original_row_v1'
RUNS_ROOT = Path(__file__).resolve().parent / 'runs'
R84_CACHE_ROOT = RUNS_ROOT / 'R84_new90k_attention/input_cache'
R84_MANIFEST = RUNS_ROOT / 'R84_new90k_attention/prepared/source_cache_manifest.json'
R94_READY = RUNS_ROOT / 'R94_pretrained_node_fusion/B/input_features_ready.json'
DEFAULT_ROUTING_ROOT = Path('/tmp/shiys_R94_B_node_features')


def _split(value):
    value = 'val' if str(value).lower() == 'validation' else str(value).lower()
    if value not in ('train', 'val', 'test'):
        raise ValueError('Feature split must be train, val or test')
    return value


def _manifest(value):
    if isinstance(value, (str, Path)):
        value = read_json(value)
    if value is not None and (not isinstance(value, dict) or not isinstance(value.get('splits'), dict)):
        raise ValueError('Manifest requires full frozen splits[split][problem] records')
    return value


def _provenance(value):
    if (not isinstance(value, dict) or value.get('format') != SPEC['format']
            or value.get('arm') not in VARIANTS or value.get('variant') != VARIANTS[value['arm']]
            or value.get('supported_problems') != list(SUPPORTED_PROBLEMS)
            or value.get('widths') != WIDTH_BY_PROBLEM
            or value.get('external_pretraining_by_component') != dict(
                TSP=value.get('arm') == 'B', ATSP=value.get('arm') == 'B', VRP=True)
            or set(value.get('components', {})) != {'TSP', 'ATSP', 'VRP'}):
        raise ValueError('Require the full R102 donor component provenance')
    for key, width in (('TSP', 128), ('ATSP', 512), ('VRP', 128)):
        component = value['components'][key]
        variant = 'pretrained' if key == 'VRP' or value['arm'] == 'B' else 'random'
        if component.get('output_dim') != width or component.get('variant') != variant:
            raise ValueError('R102 component width/pretraining identity differs')
    return value


def _source_record(cache_root, split, problem, manifest=None, data_root=None):
    directory = Path(cache_root).expanduser().resolve() / split / problem
    record = read_json(directory / 'metadata.json')
    canonical = R84_CACHE_ROOT / split / problem / 'metadata.json'
    entry = None if manifest is None else manifest.get('splits', {}).get(split, {}).get(problem)
    if manifest is not None and not isinstance(entry, dict):
        raise ValueError(f'{split}/{problem}: missing full frozen manifest entry')
    if canonical.is_file():
        reference = read_json(canonical)
        if entry is not None:
            validate_frozen_header(entry, reference)
    elif R84_MANIFEST.is_file():
        # R84's original TEST was prepared only in /tmp; its immutable full
        # header is retained in the shared manifest, not input_cache/test.
        reference = read_json(R84_MANIFEST).get('splits', {}).get(split, {}).get(problem)
        if not isinstance(reference, dict):
            raise ValueError('Frozen R84 manifest lacks the original full task header')
        if entry is not None:
            validate_frozen_header(entry, reference)
    elif entry is not None:
        reference = copy.deepcopy(entry)
    else:
        raise ValueError('Missing immutable original R84 header or full frozen manifest')
    validate_frozen_header(record, reference)
    if record['problem'] != problem or record['split'] != split:
        raise ValueError('Frozen R84 task/split identity differs')
    if data_root is not None and Path(record['data_root']).expanduser().resolve() != Path(data_root).expanduser().resolve():
        raise ValueError('Frozen R84 data_root identity differs')
    return directory, record, reference


def _open_array(path, shape, dtype):
    array = np.load(path, mmap_mode='r', allow_pickle=False)
    try:
        if (array.shape != tuple(shape) or array.dtype != np.dtype(dtype)
                or not array.flags.c_contiguous):
            raise ValueError(f'{path}: input/feature shape, dtype or row layout differs')
        array._mmap.madvise(mmap.MADV_RANDOM)
    except BaseException:
        array._mmap.close()
        raise
    return array


def _lengths(directory, record, reference, witness=None):
    array = _open_array(directory / 'n.npy', [record['rows']], 'int64')
    try:
        values = np.array(array, copy=True)
    finally:
        array._mmap.close()
    original_path = (Path(witness) if witness is not None else
                     R84_CACHE_ROOT / record['split'] / record['problem'] / 'n.npy')
    if not original_path.is_file():
        original_path = Path(reference['directory']) / 'n.npy'
    original = _open_array(original_path, [record['rows']], 'int64')
    try:
        if (int(values.min()) != record['true_length_min'] or int(values.max()) != record['true_length_max']
                or not np.array_equal(values, original)):
            raise ValueError('Actual full n vector/original row alignment differs')
    finally:
        original._mmap.close()
    return values


def _source_identity(record, reference, lengths):
    key = 'matrix' if record['kind'] == 'matrix' else 'node'
    return dict(metadata_sha256=digest(reference),
                actual_n_sha256=hashlib.sha256(lengths.tobytes()).hexdigest(),
                inputs={name: record['arrays'][name] for name in ('n', key)},
                row_alignment='original frozen R84 rows and node order; full n vector equality')


def _routing_source(directory, record, reference, routing_directory, provenance, lengths):
    routing_directory = Path(routing_directory).expanduser().resolve()
    donor = provenance['components']['VRP']
    copied = (routing_directory / COPY_RECEIPT).is_file()
    if copied:
        receipt, original, metadata = validate_copy_receipt(routing_directory, donor)
        validate_frozen_header(record, original)
        witness = _lengths(directory, record, reference, routing_directory / 'n.npy')
        if not np.array_equal(witness, lengths):
            raise ValueError('Transported routing full n vector differs')
        payload = dict(mode='transported_payload_receipt', copy_receipt_sha256=digest(receipt),
                       files=receipt['files'])
    else:
        ready = read_json(R94_READY)
        if (ready.get('experiment') != 'R94' or ready.get('arm') != 'B'
                or ready.get('complete') is not True or ready.get('frozen_donor') != donor
                or ready.get('labels_used') is not False or ready.get('decoder_executed') is not False
                or not {'train', 'val'} <= set(ready.get('splits', []))
                or routing_directory != Path(ready.get('feature_root', '')).resolve() / record['split'] / record['problem']):
            raise ValueError('Direct routing reuse requires the known original completed R94B receipt')
        metadata = read_json(routing_directory / 'metadata.json')
        _validate_r94(metadata, reference, donor)
        source = metadata['source_cache']
        if Path(source.get('directory', '')).resolve() != directory:
            raise ValueError('Direct routing inputs must retain the original raw cache path; transport needs a receipt')
        for key in ('n', 'node'):
            saved, path = source['inputs'][key], directory / record['arrays'][key]['file']
            if saved.get('path') != str(path) or saved['bytes'] != path.stat().st_size:
                raise ValueError('Original direct R94B input file/schema differs')
        payload = dict(mode='original_direct', files=None,
                       note='No copy occurred; original R94B provenance and full R84 n are retained')
    shape = [record['rows'], record['true_length_max'], 128]
    header = npy_header(routing_directory / FEATURE_FILE)
    size = (routing_directory / FEATURE_FILE).stat().st_size
    if (header['shape'] != shape or header['dtype'] != '<f2'
            or size != header['offset'] + int(np.prod(shape)) * 2):
        raise ValueError('Complete original R94B FP16 feature header differs')
    return dict(r94_metadata_sha256=digest(metadata),
                r84_metadata_sha256=digest(reference), donor_sha256=digest(donor),
                actual_n_sha256=hashlib.sha256(lengths.tobytes()).hexdigest(),
                feature_bytes=size, payload=payload)


def _contract(record, reference, lengths, provenance, lock, source_kind):
    return dict(format=FEATURE_FORMAT, problem=record['problem'], split=record['split'],
        rows=record['rows'], shape=[record['rows'], record['true_length_max'], WIDTH_BY_PROBLEM[record['problem']]],
        dtype='float16', file=FEATURE_FILE, variant=provenance['variant'], donor=provenance,
        input_only=True, labels_opened=False, source_kind=source_kind,
        source_cache=_source_identity(record, reference, lengths), test_lock=lock,
        extract_order=EXTRACT_ORDER)


def _complete(target, contract):
    if not (target / 'metadata.json').is_file():
        raise ValueError(f'{target}: incomplete/untracked sidecar preserved; use a new root')
    metadata = read_json(target / 'metadata.json')
    if (any(metadata.get(key) != value for key, value in contract.items())
            or metadata.get('complete') is not True or metadata.get('rows_written') != contract['rows']):
        raise ValueError(f'{target}: incomplete or identity-changing sidecar preserved; use a new root')
    return metadata


def _atomic_complete(path, value):
    temporary = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    write_new_json(temporary, value)
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _signature(path):
    value = path.stat()
    return value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns


def _prepare_native(directory, record, reference, lengths, target, contract, donor, device, batch_size):
    if target.exists():
        metadata = _complete(target, contract)
        array = _open_array(target / FEATURE_FILE, contract['shape'], 'float16')
        array._mmap.close()
        entry = metadata.get('files', {}).get(FEATURE_FILE, {})
        if (entry.get('bytes') != (target / FEATURE_FILE).stat().st_size
                or not isinstance(entry.get('sha256'), str) or len(entry['sha256']) != 64):
            raise ValueError('Native feature payload receipt differs')
        return metadata
    target.mkdir(parents=True, exist_ok=False)
    with (target / '.prepare.lock').open('a') as guard:
        fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
        metadata = dict(contract, complete=False, rows_written=0)
        write_new_json(target / 'metadata.json', metadata)
        spec = record['arrays']['matrix' if record['kind'] == 'matrix' else 'node']
        input_path = directory / spec['file']
        before = {name: _signature(directory / name) for name in ('metadata.json', 'n.npy', spec['file'])}
        inputs = output = None
        try:
            inputs = _open_array(input_path, spec['shape'], spec['dtype'])
            output = np.lib.format.open_memmap(target / PARTIAL_FILE, mode='w+', dtype=np.float16,
                                                shape=tuple(contract['shape']))
            order = np.argsort(lengths, kind='stable')
            sorted_n, start = lengths[order], 0
            while start < len(order):
                count = int(sorted_n[start])
                end = int(np.searchsorted(sorted_n, count, side='right'))
                stop = min(start + batch_size, end)
                rows = order[start:stop]
                view = inputs[:, :count, :count] if record['kind'] == 'matrix' else inputs[:, :count]
                raw = torch.from_numpy(np.array(view[rows], copy=True, order='C')).to(device)
                key = 'matrix' if record['kind'] == 'matrix' else 'node'
                batch = dict(problem_id=record['problem_id'], problem_name=record['problem'],
                    kind=record['kind'], d_in=record['d_in'],
                    n=torch.full((len(rows),), count, dtype=torch.long, device=device),
                    node_mask=torch.ones(len(rows), count, dtype=torch.bool, device=device))
                batch[key] = raw
                with torch.no_grad():
                    hidden = donor(batch)
                width = WIDTH_BY_PROBLEM[record['problem']]
                if (hidden.shape != (len(rows), count, width) or hidden.dtype != torch.float32
                        or hidden.requires_grad or not torch.isfinite(hidden).all()):
                    raise ValueError('Native donor must return finite frozen FP32 node-aligned features')
                values = hidden.to(device='cpu', dtype=torch.float16).numpy()
                if not np.isfinite(values).all():
                    raise ValueError('Native features overflowed the FP16 sidecar')
                output[rows, :count], output[rows, count:] = values, 0
                metadata['rows_written'] = stop
                start = stop
            output.flush()
        finally:
            for array in (inputs, output):
                if array is not None:
                    array._mmap.close()
        if before != {name: _signature(directory / name) for name in before}:
            raise ValueError('Native inputs changed during extraction; preserve incomplete sidecar')
        validate_frozen_header(read_json(directory / 'metadata.json'), reference)
        if not np.array_equal(_lengths(directory, record, reference), lengths):
            raise ValueError('Full n changed during native extraction; preserve incomplete sidecar')
        partial = target / PARTIAL_FILE
        metadata['files'] = {FEATURE_FILE: dict(bytes=partial.stat().st_size, sha256=file_hash(partial))}
        partial.rename(target / FEATURE_FILE)
        metadata['complete'] = True
        _atomic_complete(target / 'metadata.json', metadata)
        return metadata


def prepare_features(cache_root, feature_root, split, donor, device, *, routing_feature_root,
                     batch_size=128, manifest=None, data_root=None, test_lock=None):
    """Prepare only native TSP/ATSP; validate/reference R94B for all other16."""
    split = _split(split)
    provenance = _provenance(donor.provenance())
    lock = authorize_test(test_lock, provenance['arm']) if split == 'test' else None
    if isinstance(batch_size, bool) or operator.index(batch_size) < 1:
        raise ValueError('batch_size must be a positive integer')
    batch_size, manifest = operator.index(batch_size), _manifest(manifest)
    root = Path(feature_root).expanduser().resolve()
    routing_root = Path(routing_feature_root).expanduser().resolve()
    cache_root = Path(cache_root).expanduser().resolve()
    if (root == cache_root or root == routing_root or root in cache_root.parents
            or cache_root in root.parents or root in routing_root.parents or routing_root in root.parents):
        raise ValueError('New feature root must be disjoint from immutable input/routing roots')
    sources = {}
    # Preflight every existing source before writing either native task.
    for problem in SUPPORTED_PROBLEMS:
        directory, record, reference = _source_record(cache_root, split, problem, manifest, data_root)
        witness = routing_root / split / problem / 'n.npy'
        witness = witness if problem in ROUTING_PROBLEMS and (routing_root / split / problem / COPY_RECEIPT).is_file() else None
        lengths = _lengths(directory, record, reference, witness)
        source_kind = 'r94b_reference' if problem in ROUTING_PROBLEMS else 'native_extra'
        contract = _contract(record, reference, lengths, provenance, lock, source_kind)
        if problem in ROUTING_PROBLEMS:
            route = routing_root / split / problem
            contract['routing_identity'] = _routing_source(directory, record, reference, route, provenance, lengths)
        sources[problem] = (directory, record, reference, lengths, contract)
    donor.to(torch.device(device)).eval()
    records = {}
    for problem, (directory, record, reference, lengths, contract) in sources.items():
        target = root / split / problem
        if problem in ROUTING_PROBLEMS:
            if target.exists():
                result = _complete(target, contract)
            else:
                target.mkdir(parents=True, exist_ok=False)
                result = dict(contract, complete=True, rows_written=record['rows'],
                              routing_directory=str(routing_root / split / problem))
                write_new_json(target / 'metadata.json', result)
        else:
            result = _prepare_native(directory, record, reference, lengths, target, contract,
                                     donor, device, batch_size)
        records[problem] = dict(result, directory=str(target))
    return dict(format=FEATURE_FORMAT, complete=True, donor=provenance, splits={split: records},
                routing_reextracted=False, source_files_opened=False, labels_opened=False)


def _indices(values, size):
    if torch.is_tensor(values):
        if values.device.type != 'cpu' or values.ndim != 1:
            raise ValueError('Cache indices must be a one-dimensional CPU sequence')
        values = values.tolist()
    if isinstance(values, str):
        values = [values]
    if isinstance(values, np.ndarray) and values.ndim != 1:
        raise ValueError('Cache indices must be one-dimensional')
    rows = []
    for value in values:
        if isinstance(value, str):
            row = int(value)
            if str(row) != value:
                raise KeyError('Unknown instance ID')
        else:
            if isinstance(value, (bool, np.bool_)):
                raise ValueError('Boolean row indices are not instance IDs')
            row = operator.index(value)
        if not 0 <= row < size:
            raise IndexError('Feature row outside original cache')
        rows.append(row)
    if not rows:
        raise ValueError('Cannot gather an empty feature batch')
    return np.asarray(rows, dtype=np.int64)


class FrozenFeatureCache:
    """Read-only lazy mmap plus precise row-span prefetch; base API unchanged."""

    def __init__(self, base_cache, feature_directory, expected_variant=None, expected_provenance=None,
                 *, routing_feature_directory=None):
        self.base_cache = base_cache
        self.feature_directory = Path(feature_directory).expanduser().resolve()
        self._features, self._feature_fd, self._closed = None, None, False
        saved = read_json(self.feature_directory / 'metadata.json')
        provenance = _provenance(saved.get('donor'))
        if expected_variant is not None and expected_variant not in VARIANTS.values():
            raise ValueError('Unknown R102 expected_variant')
        if expected_variant is not None and expected_variant != provenance['variant']:
            raise ValueError('Feature variant differs from the attached R102 arm')
        if expected_provenance is not None and provenance != expected_provenance:
            raise ValueError('Full feature provenance differs from the attached donor')
        split, problem = base_cache.split, base_cache.problem
        lock = None
        if split == 'test':
            authorization = saved.get('test_lock')
            if not isinstance(authorization, dict):
                raise PermissionError('TEST consumption requires a new intact R102 VAL lock')
            lock = authorize_test(authorization.get('record'), provenance['arm'])
            if authorization != lock:
                raise PermissionError('R102 TEST authorization receipt changed')
        directory, record, reference = _source_record(base_cache.directory.parent.parent, split, problem)
        if base_cache.metadata != record or base_cache.size != record['rows']:
            raise ValueError('Base cache full frozen R84 metadata differs')
        routing = problem in ROUTING_PROBLEMS
        route = None
        if routing:
            route = Path(routing_feature_directory or saved.get('routing_directory', '')).expanduser().resolve()
        witness = route / 'n.npy' if routing and (route / COPY_RECEIPT).is_file() else None
        lengths = _lengths(directory, record, reference, witness)
        if not np.array_equal(np.asarray(base_cache.lengths), lengths):
            raise ValueError('Base cache actual full n/original row alignment differs')
        contract = _contract(record, reference, lengths, provenance, lock,
                             'r94b_reference' if routing else 'native_extra')
        if routing:
            contract['routing_identity'] = _routing_source(directory, record, reference, route, provenance, lengths)
        self.feature_metadata = _complete(self.feature_directory, contract)
        path = (route if routing else self.feature_directory) / FEATURE_FILE
        try:
            self._features = _open_array(path, contract['shape'], 'float16')
            self._feature_fd = os.open(path, os.O_RDONLY)
            os.posix_fadvise(self._feature_fd, 0, 0, os.POSIX_FADV_RANDOM)
        except BaseException:
            if self._feature_fd is not None:
                os.close(self._feature_fd)
                self._feature_fd = None
            if self._features is not None:
                self._features._mmap.close()
                self._features = None
            raise

    def __getattr__(self, name):
        return getattr(self.base_cache, name)

    def batch(self, indices, device='cpu'):
        if self._closed:
            raise RuntimeError('FrozenFeatureCache is closed')
        rows = _indices(indices, self.base_cache.size)
        batch = self.base_cache.batch(rows, device=device)
        mask, maximum = batch['node_mask'], batch['node_mask'].shape[1]
        width = WIDTH_BY_PROBLEM[self.base_cache.problem]
        if mask.dtype != torch.bool or maximum > self._features.shape[1]:
            raise ValueError('Base batch node mask differs from original feature rows')
        for row in np.unique(rows):
            os.posix_fadvise(self._feature_fd,
                int(self._features.offset) + int(row) * int(self._features.strides[0]),
                maximum * width * self._features.dtype.itemsize, os.POSIX_FADV_WILLNEED)
        values = np.ascontiguousarray(self._features[:, :maximum][rows])
        nodes = torch.from_numpy(values).to(device=mask.device)
        nodes.masked_fill_(~mask[..., None], 0)
        if not torch.isfinite(nodes).all():
            raise ValueError('Nonfinite valid frozen feature payload')
        batch['nco_nodes'], batch['nco_node_mask'] = nodes, mask
        return batch

    def close(self):
        if not self._closed:
            if self._feature_fd is not None:
                os.close(self._feature_fd)
                self._feature_fd = None
            if self._features is not None:
                self._features._mmap.close()
                self._features = None
            self.base_cache.close()
            self._closed = True


def estimate_features(cache_root, splits=('train', 'val'), manifest=None, data_root=None):
    """Metadata only: new native bytes vs direct-reused routing bytes, no arrays."""
    splits = (splits,) if isinstance(splits, str) else splits
    manifest, tasks = _manifest(manifest), []
    for split in tuple(dict.fromkeys(_split(s) for s in splits)):
        for problem in SUPPORTED_PROBLEMS:
            _, record, _ = _source_record(cache_root, split, problem, manifest, data_root)
            size = record['rows'] * record['true_length_max'] * WIDTH_BY_PROBLEM[problem] * 2
            tasks.append(dict(problem=problem, split=split, payload_bytes=size,
                              new_extraction=problem not in ROUTING_PROBLEMS))
    return dict(format=FEATURE_FORMAT, metadata_only=True, tasks=tasks,
        new_payload_bytes=sum(t['payload_bytes'] for t in tasks if t['new_extraction']),
        reused_payload_bytes=sum(t['payload_bytes'] for t in tasks if not t['new_extraction']),
        routing_reextracted=False)


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    commands = result.add_subparsers(dest='command', required=True)
    for name in ('estimate', 'prepare'):
        command = commands.add_parser(name)
        command.add_argument('--cache-root', type=Path, required=True)
        command.add_argument('--manifest', type=Path)
        command.add_argument('--data-root', type=Path)
        if name == 'estimate':
            command.add_argument('--splits', nargs='+', choices=('train', 'val', 'test'), default=['train', 'val'])
        else:
            command.add_argument('--feature-root', type=Path, required=True)
            command.add_argument('--routing-feature-root', type=Path, required=True)
            command.add_argument('--split', choices=('train', 'val', 'test'), required=True)
            command.add_argument('--arm', choices=('A', 'B'), required=True)
            command.add_argument('--device', required=True)
            command.add_argument('--batch-size', type=int, default=128)
            command.add_argument('--test-lock', type=Path)
    return result


def main(argv=None):
    import json
    args = vars(parser().parse_args(argv))
    command = args.pop('command')
    if command == 'estimate':
        value = estimate_features(**args)
    else:
        if args['split'] == 'test':
            authorize_test(args['test_lock'], args['arm'])
        args['donor'] = FullTypeDonor(args.pop('arm'))
        value = prepare_features(**args)
    print(json.dumps(value, sort_keys=True, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
