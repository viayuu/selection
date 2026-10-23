"""Explicit future CPU-only R94B routing export/stage; standard library only.

No implicit transport, extraction, solver execution or source data access.
Original metadata is copied byte-for-byte. Incomplete targets are preserved
and rejected, never automatically resumed or regenerated. All payload copies
and hashes stream through at most an 8 MiB buffer.
"""

import argparse
import ast
import copy
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import struct


MAX_BUFFER_BYTES = 8 * 1024 * 1024
COPY_FORMAT = 'r102_r94b_routing_copy_v1'
TRANSPORT_FORMAT = 'r102_routing_transport_v1'
COPY_RECEIPT = 'copy_receipt.json'
FEATURE_FILE = 'nco_nodes.npy'
PROBLEMS = ('TSP', 'CVRP', 'ATSP', 'OVRP', 'VRPB', 'VRPL', 'VRPTW', 'OVRPTW',
            'OVRPB', 'OVRPL', 'VRPBL', 'VRPBTW', 'VRPLTW', 'OVRPBL', 'OVRPBTW',
            'OVRPLTW', 'VRPBLTW', 'OVRPBLTW')
ROUTING_PROBLEMS = tuple(p for p in PROBLEMS if p not in ('TSP', 'ATSP'))
RELD_SHA256 = '2ce5fb49950c4c9f51d2015a125744be9960aa0c60c609cfe235e74d44ad4be3'
PAYLOAD_FILES = ('metadata.json', FEATURE_FILE, 'n.npy', 'r84_metadata.json')
FROZEN_FIELDS = frozenset((
    'problem', 'problem_id', 'split', 'data_root', 'rows', 'true_length_min',
    'true_length_max', 'length_definition', 'kind', 'd_in', 'pool', 'pool_ids',
    'raw_cost_dtype', 'winner_source', 'coord_augment', 'coord_dist', 'solver_aliases',
    'scenario_id', 'source_files', 'scenario_global_solvers', 'scenario_global_solver_ids',
    'model_global_solvers', 'scenario_solver_aliases', 'directory', 'cache_root',
    'cache_format', 'cache_version', 'complete', 'rows_written', 'created_utc',
    'seconds', 'feature_sources', 'arrays', 'cbits', 'problem_desc', 'mask',
    'geometry_calculation_dtype', 'geometry_cache_dtype', 'geometry_fields',
    'geometry_scales', 'geometry_moments', 'test_lock_root', 'reuse_policy',
    'reused_input_rows', 'newly_computed_geometry_rows', 'completed_utc', 'storage_bytes'))
RUNTIME_FIELDS = frozenset(('directory', 'cache_root'))


def read_json(path):
    path = Path(path)
    if not 0 < path.stat().st_size <= 4 * 1024 * 1024:
        raise ValueError(f'{path}: require a small bounded JSON receipt')
    return json.loads(path.read_text())


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode('utf-8')).hexdigest()


def file_hash(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(MAX_BUFFER_BYTES), b''):
            result.update(block)
    return result.hexdigest()


def write_new_json(path, record):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(record, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())


def authorize_test(record, arm=None):
    if isinstance(record, (str, Path)):
        record = read_json(record)
    record = copy.deepcopy(record)
    checkpoint = record.get('checkpoint') if isinstance(record, dict) else None
    sha = record.get('checkpoint_sha256') if isinstance(record, dict) else None
    if (not isinstance(record, dict) or record.get('experiment') != 'R102'
            or record.get('arm') not in ('A', 'B') or (arm is not None and record['arm'] != arm)
            or record.get('selection_split') not in ('val', 'validation')
            or record.get('selected_without_test') is not True
            or record.get('per_problem_assembly') is not False
            or type(record.get('selected_update')) is not int or record['selected_update'] <= 0
            or not isinstance(checkpoint, str) or not Path(checkpoint).is_absolute()
            or not isinstance(sha, str) or len(sha) != 64
            or any(char not in '0123456789abcdef' for char in sha.lower())
            or not isinstance(record.get('inference_rule'), (str, dict)) or not record['inference_rule']):
        raise PermissionError('TEST requires this arm\'s unique R102 VAL-only checkpoint/inference lock')
    return dict(record=record, sha256=digest(record))


def validate_frozen_header(record, reference=None):
    """Check every frozen header field, never open any specified arrays."""
    if not isinstance(record, dict) or not FROZEN_FIELDS <= set(record):
        raise ValueError('Frozen R84 header has missing fields; no defensive bypass')
    if reference is not None:
        validate_frozen_header(reference)
        actual = {k: v for k, v in record.items() if k not in RUNTIME_FIELDS}
        frozen = {k: v for k, v in reference.items() if k not in RUNTIME_FIELDS}
        if actual != frozen:
            raise ValueError('Full frozen R84 header identity differs')
    problem, split, rows = record['problem'], record['split'], record['rows']
    maximum, minimum = record['true_length_max'], record['true_length_min']
    kind = 'matrix' if problem == 'ATSP' else 'coord'
    width = None if kind == 'matrix' else 2 if problem == 'TSP' else 4 if problem == 'CVRP' else 8
    if (problem not in PROBLEMS or record['problem_id'] != PROBLEMS.index(problem)
            or split not in ('train', 'val', 'test') or type(rows) is not int or rows <= 0
            or type(maximum) is not int or type(minimum) is not int or not 2 <= minimum <= maximum
            or record['cache_format'] != 'r76_scenario_v1' or record['cache_version'] != 1
            or record['complete'] is not True or record['rows_written'] != rows
            or record['kind'] != kind or record['d_in'] != width
            or record['raw_cost_dtype'] != 'float64' or not record['winner_source']
            or record['geometry_calculation_dtype'] != 'float64'
            or record['geometry_cache_dtype'] != 'float32'
            or record['geometry_scales'] != [4, 8, 16]
            or record['geometry_fields'] != [f'k{k}_{name}' for k in (4, 8, 16)
                for name in ('radius', 'mean_distance', 'distance_cv', 'anisotropy')]
            or not record['scenario_id'] or not record['source_files'] or not record['feature_sources']
            or not record['cbits'] or not record['problem_desc']):
        raise ValueError('Frozen R84 header schema differs')
    pool, ids, global_ids = record['pool'], record['pool_ids'], record['scenario_global_solver_ids']
    if (not pool or len(pool) != len(set(pool)) or not isinstance(ids, list)
            or ids != sorted(set(ids)) or len(pool) != len(ids)
            or not isinstance(global_ids, dict) or any(global_ids.get(p) != i for p, i in zip(pool, ids))
            or record['mask'] != [int(i in ids) for i in range(len(record['model_global_solvers']))]
            or any(type(i) is not int or not 0 <= i < len(record['mask']) for i in ids)
            or 'RELD_MTL' in pool or 'RELD_MOEL' in pool
            or ('B' in problem and problem != 'ATSP'
                and any(p in pool for p in ('RouteFinder', 'MoSES_RF', 'MoSES_CaDA')))):
        raise ValueError('Frozen R84 legal candidate pool differs')
    arrays = dict(n=((rows,), 'int64'), costs=((rows, len(pool)), 'float64'),
                  winner=((rows,), 'int64'), coord_dist=((rows,), 'int64'))
    if kind == 'matrix':
        arrays['matrix'] = ((rows, maximum, maximum), 'float32')
    else:
        arrays.update(node=((rows, maximum, width), 'float32'),
                      node_geom=((rows, maximum, 12), 'float32'))
    expected = {key: dict(file=key + '.npy', shape=list(shape), dtype=dtype)
                for key, (shape, dtype) in arrays.items()}
    if record['arrays'] != expected:
        raise ValueError('Frozen R84 full typed-array header differs')
    return record


def npy_header(path):
    with Path(path).open('rb') as stream:
        if stream.read(6) != b'\x93NUMPY':
            raise ValueError('Expected a native NPY file')
        version = stream.read(2)
        if version == b'\x01\x00':
            raw = stream.read(2)
            size = struct.unpack('<H', raw)[0] if len(raw) == 2 else 0
        elif version in (b'\x02\x00', b'\x03\x00'):
            raw = stream.read(4)
            size = struct.unpack('<I', raw)[0] if len(raw) == 4 else 0
        else:
            raise ValueError('Unsupported NPY version')
        if not 0 < size <= 65536:
            raise ValueError('Unbounded or truncated NPY header')
        raw = stream.read(size)
        if len(raw) != size:
            raise ValueError('Truncated NPY header')
        header = ast.literal_eval(raw.decode('utf-8'))
        if (not isinstance(header, dict) or set(header) != {'descr', 'fortran_order', 'shape'}
                or header['fortran_order'] is not False or not isinstance(header['shape'], tuple)
                or any(type(n) is not int or n <= 0 for n in header['shape'])):
            raise ValueError('Invalid C-order NPY header')
        return dict(shape=list(header['shape']), dtype=header['descr'], offset=stream.tell())


def n_identity(path, record):
    header = npy_header(path)
    rows = record['rows']
    if (header['shape'] != [rows] or header['dtype'] != '<i8'
            or Path(path).stat().st_size != header['offset'] + rows * 8):
        raise ValueError('Original full n vector header differs')
    result, minimum, maximum = hashlib.sha256(), None, None
    with Path(path).open('rb') as stream:
        stream.seek(header['offset'])
        for block in iter(lambda: stream.read(MAX_BUFFER_BYTES), b''):
            result.update(block)
            for (value,) in struct.iter_unpack('<q', block):
                minimum = value if minimum is None else min(minimum, value)
                maximum = value if maximum is None else max(maximum, value)
    if (minimum, maximum) != (record['true_length_min'], record['true_length_max']):
        raise ValueError('Full n vector disagrees with original bounds')
    return result.hexdigest()


def _validate_r94(metadata, original, donor):
    problem, split = original['problem'], original['split']
    source = metadata.get('source_cache', {})
    if (metadata.get('format') != 'r94_frozen_nodes_v1' or metadata.get('complete') is not True
            or metadata.get('problem') != problem or metadata.get('split') != split
            or metadata.get('rows') != original['rows'] or metadata.get('rows_written') != original['rows']
            or metadata.get('shape') != [original['rows'], original['true_length_max'], 128]
            or metadata.get('dtype') != 'float16' or metadata.get('file') != FEATURE_FILE
            or metadata.get('input_only') is not True
            or metadata.get('extract_order') != 'stable_true_n_ascending_then_original_row_v1'
            or metadata.get('donor') != donor or donor.get('variant') != 'pretrained'
            or donor.get('checkpoint_sha256') != RELD_SHA256
            or donor.get('supported_problems') != list(ROUTING_PROBLEMS)
            or source.get('metadata_sha256') != digest(original)):
        raise ValueError('Original complete R94B provenance/header identity differs')
    for key in ('n', 'node'):
        saved = source.get('inputs', {}).get(key, {})
        if saved.get('spec') != original['arrays'][key] or type(saved.get('bytes')) is not int:
            raise ValueError('Original R94 input schema differs')
    if split == 'test':
        lock = metadata.get('test_lock')
        if (not isinstance(lock, dict) or lock.get('sha256') != digest(lock.get('record'))
                or lock.get('record', {}).get('experiment') != 'R94'
                or lock['record'].get('arm') != 'B'
                or lock['record'].get('selected_without_test') is not True):
            raise ValueError('Original R94B TEST provenance lock differs')
    elif metadata.get('test_lock') is not None:
        raise ValueError('TRAIN/VAL routing must not carry TEST authorization')


def validate_copy_receipt(directory, expected_provenance=None, *, verify_payload=False):
    """Small receipts/headers/n by default; stream H only in explicit transport."""
    directory = Path(directory)
    receipt = read_json(directory / COPY_RECEIPT)
    original = validate_frozen_header(read_json(directory / 'r84_metadata.json'))
    metadata = read_json(directory / 'metadata.json')
    donor = metadata.get('donor', {})
    _validate_r94(metadata, original, donor)
    if expected_provenance is not None and donor != expected_provenance:
        raise ValueError('Routing must retain the identical original R94B donor in both arms')
    if (receipt.get('format') != COPY_FORMAT or receipt.get('complete') is not True
            or receipt.get('problem') != original['problem'] or receipt.get('split') != original['split']
            or receipt.get('variant') != 'pretrained' or receipt.get('rows') != original['rows']
            or receipt.get('r84_metadata_sha256') != digest(original)
            or receipt.get('r94_metadata_sha256') != digest(metadata)
            or receipt.get('donor_sha256') != digest(donor)
            or set(receipt.get('files', {})) != set(PAYLOAD_FILES)
            or receipt.get('actual_n_sha256') != n_identity(directory / 'n.npy', original)):
        raise ValueError('Routing copy receipt identity differs')
    shape = [original['rows'], original['true_length_max'], 128]
    header = npy_header(directory / FEATURE_FILE)
    if (header['shape'] != shape or header['dtype'] != '<f2'
            or (directory / FEATURE_FILE).stat().st_size != header['offset'] + math.prod(shape) * 2):
        raise ValueError('Routing FP16 feature header differs')
    for name in PAYLOAD_FILES:
        entry = receipt['files'][name]
        sha = entry.get('sha256')
        if (entry.get('bytes') != (directory / name).stat().st_size
                or not isinstance(sha, str) or len(sha) != 64
                or any(c not in '0123456789abcdef' for c in sha)):
            raise ValueError('Routing payload size/hash receipt differs')
        if (name != FEATURE_FILE or verify_payload) and file_hash(directory / name) != sha:
            raise ValueError('Routing payload content hash differs')
    return receipt, original, metadata


def _copy_new(source, target, expected=None):
    before = source.stat()
    sha = hashlib.sha256()
    with source.open('rb') as reader, target.open('xb') as writer:
        for block in iter(lambda: reader.read(MAX_BUFFER_BYTES), b''):
            sha.update(block)
            writer.write(block)
        writer.flush()
        os.fsync(writer.fileno())
    after = source.stat()
    if (before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (
            after.st_size, after.st_mtime_ns, after.st_ctime_ns):
        raise ValueError('Source changed during transport; preserve incomplete target')
    entry = dict(bytes=target.stat().st_size, sha256=sha.hexdigest())
    if (expected is not None and entry != expected) or file_hash(target) != entry['sha256']:
        raise ValueError('Transport payload checksum mismatch; preserve incomplete target')
    return entry


def _splits(values):
    values = tuple(dict.fromkeys(values))
    if not values or any(v not in ('train', 'val', 'test') for v in values):
        raise ValueError('Explicit train/val/test splits required')
    return values


def _receipt_name(splits):
    return 'routing_transport.json' if splits == ('train', 'val') else 'routing_transport_' + '_'.join(splits) + '.json'


def transport(action, source_root, target_root, *, splits=('train', 'val'),
              source_receipt, cache_root=None, test_lock=None):
    """Future explicit export/stage, with no overwrites and a shared stage lock."""
    if action not in ('export', 'stage'):
        raise ValueError('Transport action must be export or stage')
    splits = _splits(splits)
    authorization = authorize_test(test_lock) if 'test' in splits else None
    source, target = Path(source_root).expanduser().resolve(), Path(target_root).expanduser().resolve()
    if source == target or source in target.parents or target in source.parents:
        raise ValueError('Source/target must be disjoint; original files are immutable')
    origin = read_json(source_receipt)
    if action == 'export':
        if (cache_root is None or origin.get('experiment') != 'R94' or origin.get('arm') != 'B'
                or origin.get('complete') is not True or origin.get('labels_used') is not False
                or origin.get('decoder_executed') is not False
                or not set(splits) <= set(origin.get('splits', []))):
            raise ValueError('Export needs the original completed R94B receipt and input cache root')
        donor = origin['frozen_donor']
    else:
        if (origin.get('format') != TRANSPORT_FORMAT or origin.get('complete') is not True
                or origin.get('variant') != 'pretrained' or origin.get('problems') != list(ROUTING_PROBLEMS)
                or not set(splits) <= set(origin.get('splits', []))):
            raise ValueError('Stage requires a complete R102 export transport receipt')
        donor = origin['donor']
    if donor.get('checkpoint_sha256') != RELD_SHA256 or donor.get('variant') != 'pretrained':
        raise ValueError('Only the original pinned R94B donor may be transported')
    target.parent.mkdir(parents=True, exist_ok=True)
    lock_path = target.with_name(target.name + '.r102_stage.lock')
    with lock_path.open('a') as guard:
        fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
        receipt_path = target / _receipt_name(splits)
        if receipt_path.exists():
            saved = read_json(receipt_path)
            if (saved.get('source_receipt_sha256') != digest(origin) or saved.get('splits') != list(splits)
                    or saved.get('complete') is not True or saved.get('donor') != donor):
                raise ValueError('Completed transport identity differs; preserve target')
            for split in splits:
                for problem in ROUTING_PROBLEMS:
                    task, _, _ = validate_copy_receipt(target / split / problem, donor, verify_payload=True)
                    if digest(task) != saved['tasks'][split][problem]:
                        raise ValueError('Completed transport task receipt changed')
            return saved
        if any((target / split / problem).exists() for split in splits for problem in ROUTING_PROBLEMS):
            raise ValueError('Incomplete transport target preserved; no automatic retry or regeneration')
        tasks = {}
        for split in splits:
            tasks[split] = {}
            for problem in ROUTING_PROBLEMS:
                source_dir, target_dir = source / split / problem, target / split / problem
                target_dir.mkdir(parents=True, exist_ok=False)
                if action == 'export':
                    input_dir = Path(cache_root) / split / problem
                    original = validate_frozen_header(read_json(input_dir / 'metadata.json'))
                    metadata = read_json(source_dir / 'metadata.json')
                    _validate_r94(metadata, original, donor)
                    paths = {'metadata.json': source_dir / 'metadata.json', FEATURE_FILE: source_dir / FEATURE_FILE,
                             'n.npy': input_dir / 'n.npy', 'r84_metadata.json': input_dir / 'metadata.json'}
                    files = {name: _copy_new(paths[name], target_dir / name) for name in PAYLOAD_FILES}
                    task = dict(format=COPY_FORMAT, complete=True, variant='pretrained',
                        problem=problem, split=split, rows=original['rows'], files=files,
                        r84_metadata_sha256=digest(original), r94_metadata_sha256=digest(metadata),
                        donor_sha256=digest(donor), actual_n_sha256=n_identity(target_dir / 'n.npy', original))
                    write_new_json(target_dir / COPY_RECEIPT, task)
                else:
                    task, _, _ = validate_copy_receipt(source_dir, donor)
                    if digest(task) != origin.get('tasks', {}).get(split, {}).get(problem):
                        raise ValueError('Exported task identity differs from pinned root receipt')
                    for name in PAYLOAD_FILES:
                        _copy_new(source_dir / name, target_dir / name, task['files'][name])
                    _copy_new(source_dir / COPY_RECEIPT, target_dir / COPY_RECEIPT)
                validate_copy_receipt(target_dir, donor)
                tasks[split][problem] = digest(task)
        result = dict(format=TRANSPORT_FORMAT, experiment='R102', complete=True,
            variant='pretrained', problems=list(ROUTING_PROBLEMS), splits=list(splits), tasks=tasks,
            donor=donor, source_receipt_sha256=digest(origin), action=action,
            test_access_authorization=authorization,
            source_path_observation=str(source), target_path_observation=str(target),
            identity_policy='payload SHA256 and original metadata/full n; paths and mtimes are not content identities')
        write_new_json(receipt_path, result)
        return result


def inspect_staged(target_root, source_receipt, splits=('train', 'val')):
    """Small JSON/stat readiness only; never open NPY payloads or weights.

    This authenticates the completed transfer's receipts, not a fresh audit
    of large payload bytes. Explicit stage already streamed those checksums.
    """
    splits = _splits(splits)
    if 'test' in splits:
        raise PermissionError('Metadata-only launcher inspection is TRAIN/VAL only')
    origin = read_json(source_receipt)
    target = Path(target_root).expanduser().resolve()
    saved = read_json(target / _receipt_name(splits))
    if (origin.get('format') != TRANSPORT_FORMAT or origin.get('complete') is not True
            or saved.get('format') != TRANSPORT_FORMAT or saved.get('complete') is not True
            or saved.get('action') != 'stage' or saved.get('variant') != 'pretrained'
            or saved.get('source_receipt_sha256') != digest(origin)
            or saved.get('splits') != list(splits) or saved.get('donor') != origin.get('donor')
            or saved.get('problems') != list(ROUTING_PROBLEMS)):
        raise ValueError('Completed staged routing manifest identity differs')
    for split in splits:
        for problem in ROUTING_PROBLEMS:
            directory = target / split / problem
            task = read_json(directory / COPY_RECEIPT)
            if (task.get('complete') is not True or task.get('format') != COPY_FORMAT
                    or digest(task) != saved.get('tasks', {}).get(split, {}).get(problem)
                    or digest(task) != origin.get('tasks', {}).get(split, {}).get(problem)):
                raise ValueError('Completed staged task receipt differs')
            original = validate_frozen_header(read_json(directory / 'r84_metadata.json'))
            metadata = read_json(directory / 'metadata.json')
            _validate_r94(metadata, original, saved['donor'])
            if (digest(original) != task.get('r84_metadata_sha256')
                    or digest(metadata) != task.get('r94_metadata_sha256')
                    or set(task.get('files', {})) != set(PAYLOAD_FILES)):
                raise ValueError('Completed staged small metadata hashes differ')
            for name in PAYLOAD_FILES:
                if (directory / name).stat().st_size != task['files'][name].get('bytes'):
                    raise ValueError('Completed staged payload size differs')
    return dict(format=TRANSPORT_FORMAT, complete=True, metadata_only=True,
                payload_opened=False, target_root=str(target), splits=list(splits),
                manifest_sha256=digest(saved), source_receipt_sha256=digest(origin),
                tasks=saved['tasks'], variant='pretrained')


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    commands = result.add_subparsers(dest='action', required=True)
    for action in ('export', 'stage'):
        command = commands.add_parser(action)
        command.add_argument('--source-root', type=Path, required=True)
        command.add_argument('--target-root', type=Path, required=True)
        command.add_argument('--source-receipt', type=Path, required=True)
        command.add_argument('--splits', nargs='+', choices=('train', 'val', 'test'), default=['train', 'val'])
        command.add_argument('--test-lock', type=Path)
        if action == 'export':
            command.add_argument('--cache-root', type=Path, required=True)
    return result


def main(argv=None):
    args = vars(parser().parse_args(argv))
    print(json.dumps(transport(**args), sort_keys=True, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
