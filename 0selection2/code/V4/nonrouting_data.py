"""Native, label-isolated nonrouting records and an immutable CPU export.

No solver modules are imported. Routing data is referenced, never loaded here.
See runs/R120_multidomain_selector/data_preparation.md for the interface contract.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np


SOURCE_ROOT = Path('/public/home/shiys/nonrouting_selector_20261023')
ROUTING_ROOT = Path('/public/home/shiys/selection_data_new90k_B_filtered_single_reld_mt_seed2_20261018')
PREPARED_ROOT = Path('/public/home/shiys/selection_data_multidomain_seed2_20261029')
SCHEMA_VERSION = 'nonrouting_raw_v1'
PREPARED_VERSION = 'multidomain_seed2_20261029_v1'
BASE_COMMIT = '04866414831d25330b0d2e575e4b63c68b4196d0'
PROBLEMS = ('MIS', 'JSSP', 'FJSP', 'BPP', 'MKP', 'SMTWTP')
READY_PROBLEMS = ('MIS', 'JSSP', 'FJSP')
DIRECTIONS = {p: 'max' if p in ('MIS', 'MKP') else 'min' for p in PROBLEMS}
_SCHEDULE_FIELDS = ('num_jobs', 'num_machines', 'job_ptr', 'operation_job',
                    'operation_index', 'precedence_edge_index')
INPUT_FIELDS = {
    'MIS': ('num_nodes', 'edge_index'),
    'JSSP': _SCHEDULE_FIELDS + ('duration', 'machine'),
    'FJSP': _SCHEDULE_FIELDS + ('processing_time', 'compatibility'),
    'BPP': ('sizes', 'capacity'),
    'MKP': ('profits', 'resources', 'capacities'),
    'SMTWTP': ('p', 'w', 'd'),
}
LABEL_DTYPES = {
    'native_objective': np.float64, 'observed_mask': np.bool_,
    'all_best_mask': np.bool_, 'canonical_winner_index': np.int64,
    'runtime_s': np.float64,
}


def _json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def _dump(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _integers(value):
    array = np.asarray(value)
    if array.size and (array.dtype.kind not in 'iu' or array.min() < -(2**31)
                       or array.max() >= 2**31):
        raise ValueError('Native inputs must contain int32-representable integers')
    return np.asarray(value, dtype=np.int32)


def _edges(pairs):
    return _integers(pairs).reshape(-1, 2).T.copy()


def typed_inputs(instance: Mapping) -> dict[str, np.ndarray]:
    """Encode only raw problem data, preserving node/item/job/machine numbering."""
    problem, data = instance['problem'], instance['data']
    if problem == 'MIS':
        return {'num_nodes': _integers(data['n']), 'edge_index': _edges(data['edges'])}
    if problem in ('JSSP', 'FJSP'):
        jobs, machines = data['jobs'], data['machines']
        ptr = np.concatenate(([0], np.cumsum([len(job) for job in jobs]))).astype(np.int32)
        operation_job, operation_index, precedences, options = [], [], [], []
        for job_id, job in enumerate(jobs):
            for operation_id, alternatives in enumerate(job):
                operation_job.append(job_id)
                operation_index.append(operation_id)
                if operation_id:
                    current = int(ptr[job_id]) + operation_id
                    precedences.append((current - 1, current))
                options.append(alternatives)
        result = dict(num_jobs=_integers(len(jobs)), num_machines=_integers(machines),
                      job_ptr=ptr, operation_job=_integers(operation_job),
                      operation_index=_integers(operation_index),
                      precedence_edge_index=_edges(precedences))
        if problem == 'JSSP':
            if any(len(alternatives) != 1 for alternatives in options):
                raise ValueError('JSSP operation must have exactly one machine')
            result.update(duration=_integers([a[0][1] for a in options]),
                          machine=_integers([a[0][0] for a in options]))
        else:
            times = np.full((len(options), machines), -1, dtype=np.int32)
            compatibility = np.zeros(times.shape, dtype=np.bool_)
            for operation, alternatives in enumerate(options):
                if not alternatives:
                    raise ValueError('FJSP operation has no compatible machine')
                for machine, duration in alternatives:
                    if not 0 <= machine < machines or duration <= 0 or compatibility[operation, machine]:
                        raise ValueError('Invalid or duplicate FJSP machine option')
                    times[operation, machine] = duration
                    compatibility[operation, machine] = True
            result.update(processing_time=times, compatibility=compatibility)
        return result
    if problem == 'BPP':
        return dict(sizes=_integers(data['sizes']), capacity=_integers(data['capacity']))
    if problem == 'MKP':
        return dict(profits=_integers(data['profits']), resources=_integers(data['weights']).T.copy(),
                    capacities=_integers(data['capacities']))
    if problem == 'SMTWTP':
        return dict(p=_integers(data['processing']), w=_integers(data['weights']),
                    d=_integers(data['due_dates']))
    raise ValueError(f'Unsupported problem: {problem}')


def native_labels(labels: Iterable[Mapping], candidate_solver_ids, objective_direction):
    """Use native FP64 objectives and exact ties; unobserved values stay NaN."""
    solver_ids = tuple(candidate_solver_ids)
    if not solver_ids or tuple(sorted(set(solver_ids))) != solver_ids:
        raise ValueError('Candidate solver IDs must be unique and lexicographically sorted')
    if objective_direction not in ('min', 'max'):
        raise ValueError('Direction must be min or max')
    columns = {solver_id: column for column, solver_id in enumerate(solver_ids)}
    objective = np.full(len(solver_ids), np.nan, dtype=np.float64)
    runtime = np.full(len(solver_ids), np.nan, dtype=np.float64)
    observed = np.zeros(len(solver_ids), dtype=np.bool_)
    seen = set()
    for label in labels:
        solver_id = label['solver_id']
        if solver_id not in columns or solver_id in seen:
            raise ValueError(f'Unknown or duplicate solver ID: {solver_id}')
        seen.add(solver_id)
        if label['objective_direction'] != objective_direction:
            raise ValueError('Native objective direction differs')
        if not label.get('feasible', False) or label.get('objective') is None:
            continue
        column = columns[solver_id]
        value = float(label['objective'])
        if not np.isfinite(value):
            raise ValueError('An observed native objective must be finite')
        objective[column], observed[column] = value, True
        if label.get('runtime_s') is not None:
            runtime[column] = float(label['runtime_s'])
    best = np.zeros_like(observed)
    if observed.any():
        optimum = (np.min if objective_direction == 'min' else np.max)(objective[observed])
        best = observed & (objective == optimum)
    return dict(native_objective=objective, observed_mask=observed, all_best_mask=best,
                canonical_winner_index=np.asarray(np.flatnonzero(best)[0] if best.any() else -1,
                                                  dtype=np.int64),
                runtime_s=runtime, objective_direction=objective_direction)


def instance_key(instance: Mapping):
    """No bare global-index joins; legacy neural rows encode version in their ID."""
    instance_id = instance['instance_id']
    version = instance.get('dataset_version', instance_id.split('/', 1)[0])
    key = (version, instance['problem'], instance['split'], instance_id)
    expected_id = f'{version}/{instance["problem"]}/{instance["split"]}/'
    if not instance_id.startswith(expected_id):
        raise ValueError('Instance ID does not encode the declared version/problem/split')
    return key


class IdentityRegistry:
    """Reject changed IDs, duplicate content across splits, and base-group leakage."""

    def __init__(self):
        self.keys, self.identities, self.groups = {}, {}, {}

    def add(self, instance):
        key, identity = instance_key(instance), instance['input_identity']
        if key in self.keys:
            raise ValueError(f'Duplicate or changed instance key: {key}')
        identity_key = (instance['problem'], identity)
        if identity_key in self.identities:
            raise ValueError(f'Duplicate input identity across records/splits: {identity}')
        group = instance.get('base_group_id', instance.get('base_instance_id', identity))
        group_key = (key[0], instance['problem'], str(group))
        if group_key in self.groups and self.groups[group_key] != instance['split']:
            raise ValueError('Base group crosses frozen splits')
        self.keys[key], self.identities[identity_key] = identity, key
        self.groups[group_key] = instance['split']


def make_sample(instance, labels, candidate_solver_ids, *, source_provenance):
    """Shared six-domain record constructor for accepted labels, never solutions."""
    instance_key(instance)
    metadata = {key: value for key, value in instance.items() if key != 'data'}
    metadata['candidate_solver_ids'] = tuple(candidate_solver_ids)
    metadata['source_provenance'] = source_provenance
    if 'base_group_id' not in metadata:
        metadata['base_group_id'] = metadata.get('base_instance_id', instance['input_identity'])
        metadata['base_group_source'] = ('base_instance_id' if 'base_instance_id' in metadata
                                         else 'input_identity_fallback')
    return dict(inputs=typed_inputs(instance),
                labels=native_labels(labels, candidate_solver_ids, DIRECTIONS[instance['problem']]),
                metadata=metadata)


def _pad(values):
    first = values[0]
    if first.ndim == 0:
        return np.stack(values)
    shape = tuple(max(value.shape[axis] for value in values) for axis in range(first.ndim))
    result = np.full((len(values),) + shape, False if first.dtype.kind == 'b' else -1,
                     dtype=first.dtype)
    for row, value in enumerate(values):
        result[(row,) + tuple(slice(0, size) for size in value.shape)] = value
    return result


def collate_nonrouting(samples):
    """Pad a single-domain batch. Only batch['inputs'] may enter the model."""
    import torch

    if not samples:
        raise ValueError('Cannot collate an empty batch')
    problem = samples[0]['metadata']['problem']
    solver_ids = tuple(samples[0]['metadata']['candidate_solver_ids'])
    for sample in samples:
        if sample['metadata']['problem'] != problem:
            raise ValueError('A batch must contain a single problem')
        if tuple(sample['metadata']['candidate_solver_ids']) != solver_ids:
            raise ValueError('Candidate order differs within batch')
        if set(sample['inputs']) != set(INPUT_FIELDS[problem]):
            raise ValueError('Unexpected input fields; labels/metadata must stay separate')
        if sample['labels']['objective_direction'] != DIRECTIONS[problem]:
            raise ValueError('Direction differs within batch')
    arrays = {key: _pad([s['inputs'][key] for s in samples]) for key in INPUT_FIELDS[problem]}

    def mask(name, lengths):
        arrays[name] = np.arange(max(lengths))[None, :] < np.asarray(lengths)[:, None]

    if problem == 'MIS':
        mask('node_mask', [int(s['inputs']['num_nodes']) for s in samples])
        mask('edge_mask', [s['inputs']['edge_index'].shape[1] for s in samples])
    elif problem in ('JSSP', 'FJSP'):
        mask('job_mask', [int(s['inputs']['num_jobs']) for s in samples])
        mask('machine_mask', [int(s['inputs']['num_machines']) for s in samples])
        mask('operation_mask', [len(s['inputs']['operation_job']) for s in samples])
        mask('precedence_mask', [s['inputs']['precedence_edge_index'].shape[1] for s in samples])
        mask('job_ptr_mask', [len(s['inputs']['job_ptr']) for s in samples])
    elif problem == 'BPP':
        mask('item_mask', [len(s['inputs']['sizes']) for s in samples])
    elif problem == 'MKP':
        mask('item_mask', [len(s['inputs']['profits']) for s in samples])
        mask('resource_mask', [len(s['inputs']['capacities']) for s in samples])
    else:
        mask('job_mask', [len(s['inputs']['p']) for s in samples])
    inputs = {key: torch.from_numpy(value).to(torch.bool if value.dtype.kind == 'b' else torch.int64)
              for key, value in arrays.items()}
    labels = {key: torch.from_numpy(np.stack([s['labels'][key] for s in samples]))
              for key in LABEL_DTYPES}
    labels['objective_direction'] = DIRECTIONS[problem]
    return dict(inputs=inputs, labels=labels, metadata=[s['metadata'] for s in samples])


def _check_labels(labels, columns, direction):
    objective = np.asarray(labels['native_objective'])
    observed, best = (np.asarray(labels[key]) for key in ('observed_mask', 'all_best_mask'))
    if (objective.dtype != np.float64 or objective.shape != (columns,)
            or observed.dtype != np.bool_ or observed.shape != (columns,)
            or best.dtype != np.bool_ or best.shape != (columns,)
            or np.asarray(labels['runtime_s']).shape != (columns,)
            or np.asarray(labels['canonical_winner_index']).shape != ()):
        raise ValueError('Label dtype/shape differs from the candidate columns')
    if (not np.isfinite(objective[observed]).all() or not np.isnan(objective[~observed]).all()):
        raise ValueError('Invalid missing/observed objective masks')
    expected_best = np.zeros(columns, dtype=np.bool_)
    if observed.any():
        optimum = (np.min if direction == 'min' else np.max)(objective[observed])
        expected_best = observed & (objective == optimum)
    expected_first = int(np.flatnonzero(expected_best)[0]) if expected_best.any() else -1
    if not np.array_equal(best, expected_best) or int(labels['canonical_winner_index']) != expected_first:
        raise ValueError('Native all-best mask/canonical-first policy differs')


def write_prepared_split(directory, samples, *, count, problem, split, candidate_solver_ids):
    """Stream shared records into mmap-able ragged arrays, without changing splits.

    Returns a descriptor suitable for manifest['nonrouting'][problem]['splits'][split].
    Publication is exclusive. Caller supplies accepted source identities/provenance.
    """
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    solver_ids = tuple(candidate_solver_ids)
    if count < 1 or problem not in PROBLEMS or not solver_ids or tuple(sorted(set(solver_ids))) != solver_ids:
        raise ValueError('Invalid split dimensions or candidate ordering')
    label_arrays = {}
    for key, dtype in LABEL_DTYPES.items():
        shape = (count,) if key == 'canonical_winner_index' else (count, len(solver_ids))
        label_arrays[key] = np.lib.format.open_memmap(directory / f'{key}.npy', mode='w+',
                                                     dtype=dtype, shape=shape)
    fields, handles, offsets, shapes, metadata_offsets = {}, {}, {}, {}, [0]
    versions, tie_count, observed_count = set(), 0, 0
    iterator = iter(samples)
    try:
        with (directory / 'metadata.jsonl').open('xb') as metadata_stream:
            for row in range(count):
                try:
                    sample = next(iterator)
                except StopIteration as error:
                    raise ValueError(f'Missing tail: {problem}/{split} {row}/{count}') from error
                meta, inputs, labels = sample['metadata'], sample['inputs'], sample['labels']
                if (meta['problem'] != problem or meta['split'] != split
                        or tuple(meta['candidate_solver_ids']) != solver_ids):
                    raise ValueError('Record does not match the split/candidate contract')
                if labels['objective_direction'] != DIRECTIONS[problem]:
                    raise ValueError('Record objective direction differs')
                _check_labels(labels, len(solver_ids), DIRECTIONS[problem])
                if set(inputs) != set(INPUT_FIELDS[problem]):
                    raise ValueError('Raw input schema differs')
                versions.add(instance_key(meta)[0])
                for key in INPUT_FIELDS[problem]:
                    value = np.ascontiguousarray(inputs[key]) if inputs[key].ndim else inputs[key]
                    if key not in handles:
                        fields[key] = dict(dtype=value.dtype.str, ndim=value.ndim)
                        handles[key] = (directory / f'input_{key}.bin').open('xb')
                        offsets[key], shapes[key] = [0], []
                    if value.dtype.str != fields[key]['dtype'] or value.ndim != fields[key]['ndim']:
                        raise ValueError('Ragged input dtype/rank changed')
                    handles[key].write(value.tobytes())
                    offsets[key].append(offsets[key][-1] + value.size)
                    shapes[key].append(value.shape)
                for key in LABEL_DTYPES:
                    label_arrays[key][row] = labels[key]
                observed = labels['observed_mask']
                tie_count += int(np.count_nonzero(labels['all_best_mask']) > 1)
                observed_count += int(np.count_nonzero(observed))
                encoded = (json.dumps(meta, sort_keys=True, separators=(',', ':'), allow_nan=False)
                           + '\n').encode('utf-8')
                metadata_stream.write(encoded)
                metadata_offsets.append(metadata_offsets[-1] + len(encoded))
            if next(iterator, None) is not None:
                raise ValueError(f'Unexpected records after {problem}/{split} tail')
    finally:
        for handle in handles.values():
            handle.close()
        for array in label_arrays.values():
            array.flush()
    for key in fields:
        np.save(directory / f'input_{key}_offsets.npy', np.asarray(offsets[key], dtype=np.int64))
        np.save(directory / f'input_{key}_shapes.npy', np.asarray(shapes[key], dtype=np.int32).reshape(count, fields[key]['ndim']))
        fields[key]['elements'] = offsets[key][-1]
    np.save(directory / 'metadata_offsets.npy', np.asarray(metadata_offsets, dtype=np.int64))
    descriptor = dict(schema_version=SCHEMA_VERSION, directory=str(directory), count=count,
                      problem=problem, split=split, dataset_versions=sorted(versions),
                      candidate_solver_ids=list(solver_ids), objective_direction=DIRECTIONS[problem],
                      input_fields=fields, observed_labels=observed_count, tied_instances=tie_count,
                      measured_pool_complete=observed_count == count * len(solver_ids),
                      canonical_winner_policy='first_best_in_fixed_solver_id_order_not_native_unique',
                      full_pool_complete=False)
    descriptor['files'] = {p.name: dict(bytes=p.stat().st_size, sha256=_sha256(p))
                           for p in sorted(directory.iterdir()) if p.is_file()}
    _dump(directory / 'split_manifest.json', descriptor)
    return descriptor


class NonRoutingDataset:
    """Lazy metadata and mmap raw arrays. Does not open any routing data."""

    def __init__(self, root, problem, split, *, allow_incomplete=False):
        self.manifest = load_manifest(root)
        if self.manifest['schema_version'] != SCHEMA_VERSION:
            raise ValueError('Unknown prepared schema')
        entry = self.manifest['nonrouting'][problem]
        if entry['status'] == 'pending':
            raise ValueError(f'{problem} is pending: {entry["pending_reason"]}')
        if not entry['ready_for_selector_training'] and not allow_incomplete:
            raise ValueError(f'{problem} is not a ready formal selector pool')
        self.descriptor = entry['splits'][split]
        self.directory = Path(self.descriptor['directory'])
        self.problem, self.split = problem, split
        self.candidate_solver_ids = tuple(entry['candidate_solver_ids'])
        self.objective_direction = DIRECTIONS[problem]
        if (self.descriptor['schema_version'] != SCHEMA_VERSION
                or self.descriptor['problem'] != problem or self.descriptor['split'] != split
                or tuple(self.descriptor['candidate_solver_ids']) != self.candidate_solver_ids
                or self.descriptor['objective_direction'] != self.objective_direction
                or set(self.descriptor['input_fields']) != set(INPUT_FIELDS[problem])):
            raise ValueError('Split descriptor and problem registry differ')
        self._open_arrays()
        self._stream, self._pid = None, None

    def _open_arrays(self):
        self._raw, self._offsets, self._shapes = {}, {}, {}
        for key, spec in self.descriptor['input_fields'].items():
            path = self.directory / f'input_{key}.bin'
            self._raw[key] = (np.memmap(path, mode='r', dtype=spec['dtype'], shape=(spec['elements'],))
                              if spec['elements'] else np.empty(0, dtype=spec['dtype']))
            self._offsets[key] = np.load(self.directory / f'input_{key}_offsets.npy', mmap_mode='r')
            self._shapes[key] = np.load(self.directory / f'input_{key}_shapes.npy', mmap_mode='r')
        self.label_arrays = {key: np.load(self.directory / f'{key}.npy', mmap_mode='r')
                             for key in LABEL_DTYPES}
        for key, array in self.label_arrays.items():
            shape = (len(self),) if key == 'canonical_winner_index' else (len(self), len(self.candidate_solver_ids))
            if array.shape != shape or array.dtype != LABEL_DTYPES[key]:
                raise ValueError('Prepared label array dtype/shape differs')
        self._metadata_offsets = np.load(self.directory / 'metadata_offsets.npy', mmap_mode='r')

    def __len__(self):
        return self.descriptor['count']

    def __getitem__(self, index):
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        if self._stream is None or self._pid != os.getpid():
            self.close()
            self._stream = (self.directory / 'metadata.jsonl').open('rb')
            self._pid = os.getpid()
        start, stop = map(int, self._metadata_offsets[index:index + 2])
        self._stream.seek(start)
        metadata = json.loads(self._stream.read(stop - start))
        inputs = {}
        for key in self._raw:
            first, last = map(int, self._offsets[key][index:index + 2])
            inputs[key] = self._raw[key][first:last].reshape(tuple(self._shapes[key][index]))
        labels = {key: np.asarray(array[index]) for key, array in self.label_arrays.items()}
        labels['objective_direction'] = self.objective_direction
        return dict(inputs=inputs, labels=labels, metadata=metadata)

    def close(self):
        if self._stream is not None:
            self._stream.close()
        self._stream, self._pid = None, None

    def __getstate__(self):
        state = self.__dict__.copy()
        state['_stream'], state['_pid'] = None, None
        # NumPy memmap pickle otherwise serializes the whole raw input buffer.
        for key in ('_raw', '_offsets', '_shapes', 'label_arrays', '_metadata_offsets'):
            state.pop(key)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._open_arrays()


def load_manifest(root=PREPARED_ROOT):
    """Read the registry without touching any instance arrays."""
    root = Path(root)
    return _json(root / 'manifest.json' if root.is_dir() else root)


def load_nonrouting_datasets(root=PREPARED_ROOT, *, problems=None,
                            splits=('train', 'val', 'test'), allow_incomplete=False):
    """Return datasets[problem][split]; default selection excludes pending pools."""
    manifest = load_manifest(root)
    if problems is None:
        problems = [p for p, entry in manifest['nonrouting'].items()
                    if entry['status'] != 'pending' and entry['ready_for_selector_training']]
    return {problem: {split: NonRoutingDataset(root, problem, split,
                                              allow_incomplete=allow_incomplete)
                      for split in splits} for problem in problems}


class _SourceCatalogue:
    def __init__(self, expected=None):
        self.expected = expected or {}
        self.files, self.ids = [], {}

    def identify(self, path):
        path = Path(path).absolute()
        name = str(path)
        if name not in self.ids:
            self.ids[name] = len(self.files)
            self.files.append(dict(path=name, resolved_path=str(path.resolve())))
        return self.ids[name]

    def finish(self, source_id, digest, size):
        record = self.files[source_id]
        expected = self.expected.get(record['path'], self.expected.get(record['resolved_path']))
        if expected is not None and digest != expected:
            raise ValueError(f'Accepted source hash changed: {record["path"]}')
        if 'sha256' in record and record['sha256'] != digest:
            raise ValueError('Source changed while exporting')
        record.update(sha256=digest, bytes=size, accepted_hash_checked=expected is not None)

    def add_file(self, path):
        source_id = self.identify(path)
        self.finish(source_id, _sha256(path), Path(path).stat().st_size)
        return source_id

    def records(self, paths):
        for path in paths:
            source_id, digest, size = self.identify(path), hashlib.sha256(), 0
            with Path(path).open('rb') as stream:
                for line_number, line in enumerate(stream, 1):
                    if not line.endswith(b'\n'):
                        raise ValueError(f'Partial source row: {path}:{line_number}')
                    digest.update(line)
                    size += len(line)
                    yield json.loads(line), [source_id, line_number]
            self.finish(source_id, digest.hexdigest(), size)


def _accepted_sources(source_root):
    first = source_root / 'runs/20261025_构造法首批'
    current = source_root / 'runs/20261026_持续标签'
    acceptance = _json(current / '最终验收.json')
    if not acceptance['passed'] or acceptance['full_pool_complete']:
        raise ValueError('Expected the accepted, explicitly partial source campaign')
    expected = dict(acceptance['neural_output_files'])
    expected.update(acceptance['audit_receipts_and_profiles'])
    catalogue = _SourceCatalogue(expected)
    for name in ('最终验收.json', '运行验收清单.csv', '结果与使用说明.md'):
        catalogue.add_file(current / name)
    for path in acceptance['audit_receipts_and_profiles']:
        catalogue.add_file(path)
    for name in ('首批审计.json', '冻结协议.json'):
        catalogue.add_file(first / name)
    catalogue.add_file(source_root / 'refine-logs/20261028_新增问题总览/问题与方法总览.md')
    for path in (first / '首批审计.json', current / '经典续跑审计.json',
                 current / 'MIS验证测试补充审计.json'):
        if not _json(path)['passed']:
            raise ValueError(f'Source audit not passed: {path}')
    frozen = _json(first / '冻结协议.json')
    for name in ('core.py', 'generate.py', 'production_v2/data.py', 'production_v2/protocol.py'):
        catalogue.add_file(source_root / name)
    inventory = {problem: [] for problem in PROBLEMS}
    with (current / '运行验收清单.csv').open(encoding='utf-8-sig', newline='') as stream:
        for row in csv.DictReader(stream):
            if row['audit_passed'] != 'True':
                raise ValueError('Unaccepted solver in source inventory')
            for split, count in acceptance['counts_by_problem'][row['problem']].items():
                if int(row[f'{split}_count']) != count:
                    raise ValueError('Accepted inventory split counts disagree')
            inventory[row['problem']].append(row)
    for problem in inventory:
        inventory[problem].sort(key=lambda row: row['solver_id'])
    neural_directories = {}
    for name in acceptance['neural_output_files']:
        directory = Path(name).parent.parent
        if directory in neural_directories:
            continue
        with Path(name).open() as stream:
            label = json.loads(next(stream))
        neural_directories[directory] = (label['problem'], label['solver_id'])
    return first, current, acceptance, frozen, catalogue, inventory, neural_directories


def _accepted_samples(problem, split, count, first, current, catalogue, inventory,
                      neural_directories, identities):
    solver_ids = tuple(row['solver_id'] for row in inventory[problem])
    protocols = {row['solver_id']: row['protocol_identity'] for row in inventory[problem]}
    paths = sorted((first / '实例标签' / problem / split).glob('*.jsonl'))
    paths += sorted((current / '经典标签' / problem / split).glob('*.jsonl'))
    classical = catalogue.records(paths)
    streams = []
    for directory, (domain, solver_id) in sorted(neural_directories.items()):
        if domain == problem:
            neural_paths = sorted((directory / split).glob('*.jsonl'))
            if any(str(path) not in catalogue.expected for path in neural_paths):
                raise ValueError('Neural chunk is not in the accepted frozen inventory')
            streams.append((solver_id, catalogue.records(neural_paths)))
    for index in range(count):
        try:
            record, source_ref = next(classical)
        except StopIteration as error:
            raise ValueError(f'Missing classical tail {problem}/{split}/{index}') from error
        instance = record['instance']
        if (instance['problem'] != problem or instance['split'] != split
                or instance['global_index'] != index):
            raise ValueError('Classical input order/split differs from frozen source')
        # Hash the saved native payload, not a regenerated or summarized input.
        encoded = json.dumps(dict(problem=problem, data=instance['data']),
                             sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
        if hashlib.sha256(encoded).hexdigest() != instance['input_identity']:
            raise ValueError('Saved raw input identity differs')
        identities.add(instance)
        labels = list(record['labels'])
        references = {label['solver_id']: source_ref for label in labels}
        for solver_id, stream in streams:
            try:
                label, reference = next(stream)
            except StopIteration as error:
                raise ValueError(f'Missing neural tail {solver_id}/{split}/{index}') from error
            if (instance_key(label) != instance_key(instance)
                    or label['input_identity'] != instance['input_identity']
                    or label['global_index'] != index or label['solver_id'] != solver_id
                    or label['protocol_identity'] != protocols[solver_id]):
                raise ValueError('Neural/classical source identity or deployment collision')
            labels.append(label)
            references[solver_id] = reference
        sample = make_sample(instance, labels, solver_ids, source_provenance=dict(
            input=source_ref, labels=[references.get(solver_id) for solver_id in solver_ids],
            reference_format='[source_files_index, one_based_jsonl_line]'))
        if not sample['labels']['observed_mask'].all():
            raise ValueError('Accepted measured pool has missing/nonfeasible labels')
        yield sample
    if next(classical, None) is not None:
        raise ValueError('Classical stream contains an unexpected tail')
    for solver_id, stream in streams:
        if next(stream, None) is not None:
            raise ValueError(f'Neural stream contains an unexpected tail: {solver_id}')


def build_prepared(output_root=PREPARED_ROOT, *, source_root=SOURCE_ROOT, routing_root=ROUTING_ROOT):
    """Build ready three domains, six explicit schema entries, and routing references."""
    output_root, source_root, routing_root = map(Path, (output_root, source_root, routing_root))
    output_root.mkdir(parents=True, exist_ok=True)
    if (output_root / 'manifest.json').exists():
        raise FileExistsError('Prepared manifest already exists; choose a new version')
    _dump(output_root / 'ready3_build_claim.json', dict(prepared_version=PREPARED_VERSION,
                                                      base_commit=BASE_COMMIT))
    first, current, acceptance, frozen, catalogue, inventory, neural = _accepted_sources(source_root)
    routing_split = _json(routing_root / 'split_manifest.json')
    routing_refs = {}
    for name in ('split_manifest.json', 'scenario_manifest.json', 'readiness.json', 'split_indices.npz'):
        path = routing_root / name
        routing_refs[name] = dict(path=str(path), bytes=path.stat().st_size, sha256=_sha256(path))
    manifest = dict(schema_version=SCHEMA_VERSION, prepared_version=PREPARED_VERSION,
                    base_commit=BASE_COMMIT, selector_seed=2, source_data_seed=frozen['data_seed'],
                    source_dataset_version=frozen['dataset_version'],
                    source_acceptance_identity=acceptance['acceptance_identity'],
                    full_pool_complete=False, all_six_training_pools_ready=False,
                    supported_nonrouting_types=list(PROBLEMS),
                    routing=dict(root=str(routing_root), reference_only=True, files=routing_refs,
                                 problems=list(routing_split['problems']),
                                 split_sizes_per_problem=routing_split['split_sizes_per_problem'],
                                 existing_loader='code.V4.new_split_data.load_dataset'),
                    nonrouting={})
    pending = {
        'BPP': 'Only two accepted all-tied candidates; no formal validation/test; replacement work pending.',
        'MKP': '270 benchmark-only inputs and one accepted candidate; synthetic formal pools pending.',
        'SMTWTP': '375 benchmark-only inputs and three accepted candidates; synthetic formal pools pending.',
    }
    identities = IdentityRegistry()
    for problem in PROBLEMS:
        solver_ids = [row['solver_id'] for row in inventory[problem]]
        entry = dict(status='ready' if problem in READY_PROBLEMS else 'pending',
                     ready_for_selector_training=problem in READY_PROBLEMS,
                     candidate_solver_ids=solver_ids, candidate_inventory=inventory[problem],
                     objective_direction=DIRECTIONS[problem],
                     source_available_counts=acceptance['counts_by_problem'][problem], splits={},
                     deferred_candidates=[row for row in frozen['registry'][problem]
                                          if row['status'] != 'active_constructive'],
                     full_pool_complete=False)
        manifest['nonrouting'][problem] = entry
        if problem not in READY_PROBLEMS:
            entry['pending_reason'] = pending[problem]
            continue
        for split, count in acceptance['counts_by_problem'][problem].items():
            samples = _accepted_samples(problem, split, count, first, current, catalogue,
                                        inventory, neural, identities)
            descriptor = write_prepared_split(output_root / 'nonrouting' / problem / split,
                                              samples, count=count, problem=problem, split=split,
                                              candidate_solver_ids=solver_ids)
            entry['splits'][split] = descriptor
            print(f'EXPORTED {problem}/{split}: inputs={count}, labels={descriptor["observed_labels"]}',
                  flush=True)
    if any('sha256' not in record for record in catalogue.files):
        raise ValueError('A source stream was not consumed through its audited tail')
    manifest['source_files'] = catalogue.files
    manifest['exported_inputs'] = len(identities.keys)
    manifest['exported_labels'] = sum(s['observed_labels'] for entry in manifest['nonrouting'].values()
                                       for s in entry['splits'].values())
    _dump(output_root / 'manifest.json', manifest)
    (output_root / 'manifest.sha256').write_text(_sha256(output_root / 'manifest.json') + '\n')
    # Only seal our published files, not parallel workers' pending-domain assets.
    owned_files = [output_root / name for name in ('manifest.json', 'manifest.sha256',
                                                 'ready3_build_claim.json')]
    for entry in manifest['nonrouting'].values():
        for descriptor in entry['splits'].values():
            directory = Path(descriptor['directory'])
            owned_files += [directory / name for name in descriptor['files']]
            owned_files.append(directory / 'split_manifest.json')
    for path in owned_files:
        path.chmod(0o444)
    print(f'COMPLETE {output_root / "manifest.json"}: {manifest["exported_inputs"]} inputs, '
          f'{manifest["exported_labels"]} labels; BPP/MKP/SMTWTP PENDING', flush=True)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('build',))
    parser.add_argument('--output-root', type=Path, default=PREPARED_ROOT)
    parser.add_argument('--source-root', type=Path, default=SOURCE_ROOT)
    parser.add_argument('--routing-root', type=Path, default=ROUTING_ROOT)
    args = parser.parse_args()
    build_prepared(args.output_root, source_root=args.source_root, routing_root=args.routing_root)


if __name__ == '__main__':
    main()
