"""Identity-bound strict FP64 TRAIN pair relations for frozen R100 binary ICL.

R97 X/source IDs/constant-column drop are read-only. Only TRAIN raw labels are
decoded here. -1 means an exact FP64 pair tie, 0 first wins, 1 second wins.
Pair orientation and iteration follow ascending stable global method IDs.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import fcntl
import hashlib
import os
from pathlib import Path
import shutil
import tempfile

import numpy as np

from .r97_data import (
    DEFAULT_DATA_ROOT, _atomic_json, _check_source_stamps, _close,
    _file_identity, _json_sha, _names, _positive_int, _problem_name,
    _read_json, _source_ids, _write_array, load_context,
)


SCHEMA = 'r100_train_strict_pair_relations_v1'
ROW_ORDER = 'natural frozen TRAIN row order'
LABEL_RULE = 'native FP64: 0 if cost[first]<cost[second], 1 if second<first, -1 exact tie excluded for this pair only'


def source_ids_sha256(values):
    return hashlib.sha256(_source_ids(values).astype('<i8', copy=False).tobytes()).hexdigest()


def legal_pairs(pool_ids):
    ids = np.asarray(pool_ids)
    if (ids.ndim != 1 or len(ids) < 2 or ids.dtype.kind not in 'iu'
            or np.any(ids < 0) or len(np.unique(ids)) != len(ids)):
        raise ValueError('Need at least two distinct nonnegative global method IDs')
    order = np.argsort(ids, kind='stable').tolist()
    return tuple((i, j) for a, i in enumerate(order) for j in order[a+1:])


def strict_pair_labels(costs, pool_ids):
    costs = np.asarray(costs)
    pairs = legal_pairs(pool_ids)
    if (costs.dtype != np.float64 or costs.ndim != 2 or not len(costs)
            or costs.shape[1] != len(pool_ids) or not np.isfinite(costs).all()):
        raise ValueError('Strict relations require original finite FP64 TRAIN costs')
    labels = np.full((len(costs), len(pairs)), -1, dtype=np.int8)
    for slot, (i, j) in enumerate(pairs):
        labels[costs[:, i] < costs[:, j], slot] = 0
        labels[costs[:, j] < costs[:, i], slot] = 1
    return labels


def pair_support(labels, pool, pool_ids):
    labels = np.asarray(labels)
    pairs = legal_pairs(pool_ids)
    if (labels.dtype != np.int8 or labels.ndim != 2 or not len(labels)
            or labels.shape[1] != len(pairs) or len(pool) != len(pool_ids)
            or not np.isin(labels, [-1, 0, 1]).all()):
        raise ValueError('Invalid strict binary relation table')
    records = []
    for slot, (i, j) in enumerate(pairs):
        first, second, ties = (int((labels[:, slot] == value).sum()) for value in (0, 1, -1))
        constant = .5 if first + second == 0 else 1. if second == 0 else 0. if first == 0 else None
        records.append(dict(pair_slot=slot, first_slot=i, second_slot=j,
            first_method=pool[i], second_method=pool[j],
            first_global_id=int(pool_ids[i]), second_global_id=int(pool_ids[j]),
            full_train_rows=len(labels), strict_rows=first+second, tie_rows=ties,
            label0_rows=first, label1_rows=second, constant_probability_first=constant,
            shortcut='no_strict_rows' if first+second == 0 else 'one_class' if constant is not None else 'none'))
    return records


def _validate_metadata(metadata, problem=None):
    identity = metadata['identity']
    pool = _names(identity['pool'])
    pairs = legal_pairs(identity['pool_ids'])
    rows = _positive_int(identity['rows'], 'TRAIN rows')
    if (metadata.get('complete') is not True or identity.get('schema') != SCHEMA
            or metadata.get('identity_sha256') != _json_sha(identity)
            or (problem is not None and identity['problem'] != problem)
            or identity.get('label_splits') != ['train'] or identity.get('row_order') != ROW_ORDER
            or identity.get('label_rule') != LABEL_RULE
            or identity.get('context_subsampling') is not False
            or identity.get('full_pool_winner_filter') is not False
            or len(identity['pool_ids']) != len(pool)
            or identity.get('pairs') != [list(pair) for pair in pairs]
            or identity.get('pair_support_sha256') != _json_sha(metadata['pair_support'])):
        raise ValueError('Incomplete or changed R100 strict-label identity')
    if identity['data_root'] == str(DEFAULT_DATA_ROOT) and rows != 81000:
        raise ValueError('NEW90k must retain all81000 TRAIN rows before per-pair tie filtering')
    return identity


def validate_relation_binding(context, metadata):
    identity = _validate_metadata(metadata)
    old = context.metadata['identity']
    if (identity['context_metadata_sha256'] != _json_sha(context.metadata)
            or identity['context_identity_sha256'] != context.metadata['identity_sha256']
            or identity['problem'] != old['problem'] or identity['data_root'] != old['data_root']
            or identity['rows'] != len(context.x) or identity['pool'] != list(context.pool)
            or identity['pool_ids'] != list(context.pool_ids)
            or identity['source_ids_values_sha256'] != source_ids_sha256(context.source_ids)
            or identity['kept_columns_sha256'] != _json_sha(context.kept_columns.tolist())
            or old.get('label_splits') != ['train']):
        raise ValueError('Strict TRAIN relations disagree with exact R97 context binding')
    return identity


def _prepare_problem(data_root, context_root, output_root, problem):
    from .r75_data import _CPUUnpickler

    with load_context(context_root, problem) as context:
        old, rows, candidates = context.metadata['identity'], len(context.x), len(context.pool)
        if (old['data_root'] != str(data_root) or old['row_order'] != ROW_ORDER
                or old['label_splits'] != ['train'] or old['context_subsampling'] is not False
                or old['class_quotas'] is not False):
            raise ValueError('Require the complete natural-frequency TRAIN-only R97 context')
        directory = data_root / f'{problem}train'
        paths = dict(context_metadata=context_root / problem / 'identity.json',
            native_metadata=directory / 'metadata.json', raw_label=directory / 'raw_label.pkl',
            train_input_metadata=directory / 'input_metadata.npz',
            train_cache_metadata=Path(old['source_files']['train_cache_metadata']['path']))
        sources = {key: _file_identity(path) for key, path in paths.items()}
        native, cache = _read_json(paths['native_metadata']), _read_json(paths['train_cache_metadata'])
        cached_raw = [s for s in cache['source_files'] if s['path'] == str(paths['raw_label'])]
        if (native.get('problem') != problem or native.get('split') != 'train'
                or native.get('instances') != rows or native.get('cost_dtype') != 'float64'
                or native.get('methods') != list(context.pool)
                or native.get('global_solver_ids') != list(context.pool_ids)
                or native['files']['raw_label.pkl']['sha256'] != sources['raw_label']['sha256']
                or native['files']['input_metadata.npz']['sha256'] != sources['train_input_metadata']['sha256']
                or sources['train_input_metadata']['sha256'] != old['source_files']['train_input_metadata']['sha256']
                or sources['train_cache_metadata']['sha256'] != old['source_files']['train_cache_metadata']['sha256']
                or cache.get('pool') != list(context.pool) or cache.get('pool_ids') != list(context.pool_ids)
                or cache.get('raw_cost_dtype') != 'float64' or len(cached_raw) != 1
                or any(cached_raw[0][key] != sources['raw_label'][key] for key in ('size_bytes', 'mtime_ns'))):
            raise ValueError('Native FP64 TRAIN labels/pool no longer match frozen cache/context identity')
        with np.load(paths['train_input_metadata'], allow_pickle=False) as inputs:
            ids = _source_ids(inputs['source_index'])
        if not np.array_equal(ids, context.source_ids):
            raise ValueError('Native TRAIN base-ID order differs from R97 context')
        with paths['raw_label'].open('rb') as stream:
            raw = _CPUUnpickler(stream).load()
        if not isinstance(raw, Mapping) or len(raw) != rows or set(raw) != {str(i) for i in range(rows)}:
            raise ValueError('TRAIN labels must cover every natural-order row exactly')
        costs = np.empty((rows, candidates), dtype=np.float64)
        for row in range(rows):
            record = raw[str(row)]
            values = np.asarray(record['cost'])
            if values.shape != (candidates,) or values.dtype != np.float64:
                raise ValueError('Original TRAIN cost rows must be the exact FP64 legal pool')
            if int(record['ind']) != context.y[row] or values[context.y[row]] != values.min():
                raise ValueError('Raw native winner and frozen TRAIN row identity disagree')
            costs[row] = values
        del raw
        labels = strict_pair_labels(costs, context.pool_ids)
        del costs
        support = pair_support(labels, context.pool, context.pool_ids)
        identity = dict(schema=SCHEMA, problem=problem, data_root=str(data_root), rows=rows,
            pool=list(context.pool), pool_ids=list(context.pool_ids),
            pairs=[list(pair) for pair in legal_pairs(context.pool_ids)],
            label_splits=['train'], row_order=ROW_ORDER, label_rule=LABEL_RULE,
            context_subsampling=False, full_pool_winner_filter=False,
            context_metadata_sha256=_json_sha(context.metadata),
            context_identity_sha256=context.metadata['identity_sha256'],
            source_ids_values_sha256=source_ids_sha256(ids),
            kept_columns_sha256=_json_sha(context.kept_columns.tolist()),
            raw_label_sha256=sources['raw_label']['sha256'], source_files=sources,
            pair_support_sha256=_json_sha(support))
        destination = output_root / problem
        if os.path.lexists(destination):
            if destination.is_symlink() or not destination.is_dir():
                raise ValueError('Preserve existing relation output; use a new root')
            with load_relations(output_root, problem) as table:
                if table.metadata['identity'] != identity:
                    raise ValueError('Relation identity changed; preserve old output')
                validate_relation_binding(context, table.metadata)
            _check_source_stamps(sources)
            return destination / 'identity.json'
        stage = Path(tempfile.mkdtemp(prefix=f'.{problem}.', dir=output_root))
        try:
            artifact = _write_array(stage / 'relations.npy', labels.shape, np.int8, [(0, labels)])
            _check_source_stamps(sources)
            _atomic_json(stage / 'identity.json', dict(complete=True, identity=identity,
                identity_sha256=_json_sha(identity), artifacts=dict(labels=artifact), pair_support=support))
            if os.path.lexists(destination):
                raise ValueError('Output appeared during preparation; preserve it')
            stage.rename(destination)
        finally:
            if stage.exists():
                shutil.rmtree(stage)
        return destination / 'identity.json'


def prepare_relations(data_root, context_root, output_root, problems=None):
    data_root, context_root, output_root = (Path(p).resolve() for p in (data_root, context_root, output_root))
    if output_root == context_root or output_root.is_relative_to(context_root):
        raise ValueError('Never write into historical R97 context')
    available = tuple(sorted(p.name for p in context_root.iterdir()
        if p.is_dir() and not p.name.startswith('.') and (p / 'identity.json').is_file()))
    if isinstance(problems, (str, bytes)):
        raise ValueError('Problems must be a sequence')
    problems = available if problems is None else tuple(problems)
    if not problems or len(set(problems)) != len(problems) or any(_problem_name(p) not in available for p in problems):
        raise ValueError('Require distinct available TRAIN problems')
    output_root.mkdir(parents=True, exist_ok=True)
    with (output_root / '.r100_prepare.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return {p: _prepare_problem(data_root, context_root, output_root, p) for p in problems}


@dataclass(frozen=True)
class RelationTable:
    labels: np.ndarray
    metadata: dict

    def close(self):
        _close(self.labels)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def load_relations(output_root, problem):
    directory = Path(output_root) / _problem_name(problem)
    metadata = _read_json(directory / 'identity.json')
    identity = _validate_metadata(metadata, problem)
    shape = (identity['rows'], len(identity['pairs']))
    path, spec = directory / 'relations.npy', metadata['artifacts']['labels']
    if (spec['file'] != path.name or spec['shape'] != list(shape) or spec['dtype'] != 'int8'
            or path.stat().st_size != spec['size_bytes'] or _file_identity(path)['sha256'] != spec['sha256']):
        raise ValueError('Strict TRAIN relation artifact schema/SHA changed')
    labels = np.load(path, mmap_mode='r', allow_pickle=False)
    try:
        if labels.shape != shape or pair_support(labels, identity['pool'], identity['pool_ids']) != metadata['pair_support']:
            raise ValueError('Strict relation support/tie counts changed')
        return RelationTable(labels, metadata)
    except BaseException:
        _close(labels)
        raise
