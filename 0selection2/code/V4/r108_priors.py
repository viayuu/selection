"""Fit fixed R108 priors from R84 TRAIN metadata, n.npy and native winner.npy.

CLI: python -m code.V4.r108_priors prepare --cache-root CACHE --root RUN
No source datasets, costs, solver results, VAL or TEST arrays are opened.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from code.unified_selector.registry import GLOBAL_SOLVERS, P2I, PROBLEMS


DEFAULT_ROOT = Path(__file__).resolve().parent / 'runs/R108_conditional_density_ratio'
DEFAULT_DATA_ROOT = Path('/public/home/shiys/selection_data_new90k_B_filtered_single_reld_mt_seed2_20261018')
FORMAT = 'r108_conditional_prior_v1'
PRIOR_SPEC = dict(
    version=1, fitted_on='frozen R84 new81000 TRAIN only',
    condition=['problem', 'exact_true_n'], winner='native FP64 filtered-pool winner.npy',
    global_laplace=1., conditional_shrink=20., unseen_n='task_prior',
    global_ids='original model IDs; reld_mt aliases RELD_MOEL',
    semantics='fixed aggregate learned classifier buffers, not per-query context',
)


def json_hash(record):
    return hashlib.sha256(json.dumps(record, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def _array_hash(array):
    digest = hashlib.sha256(str((array.dtype.str, array.shape)).encode())
    digest.update(array.tobytes(order='C'))
    return digest.hexdigest()


def _pool_check(metadata, problem):
    """Check the cached vocabulary, including the deliberate gap at RELD_MTL."""
    names, ids = metadata['pool'], metadata['pool_ids']
    aliases = metadata['scenario_solver_aliases']
    vocabulary = metadata['scenario_global_solver_ids']
    expected_names = [s for s in GLOBAL_SOLVERS if s != 'RELD_MTL']
    expected_names = ['reld_mt' if s == 'RELD_MOEL' else s for s in expected_names]
    expected_ids = {s: GLOBAL_SOLVERS.index(aliases.get(s, s)) for s in expected_names}
    if (metadata['model_global_solvers'] != list(GLOBAL_SOLVERS)
            or aliases != {'reld_mt': 'RELD_MOEL'}
            or metadata['scenario_global_solvers'] != expected_names
            or vocabulary != expected_ids or not names or len(set(names)) != len(names)
            or any(s not in vocabulary for s in names)
            or ids != [vocabulary[s] for s in names] or len(set(ids)) != len(ids)
            or ids != sorted(ids)):
        raise ValueError(f'{problem}: cached candidate/global ID identity changed')
    if problem not in ('TSP', 'CVRP', 'ATSP'):
        expected = ['MTPOMO', 'MVMOE', 'reld_mt'] if 'B' in problem else [
            'MTPOMO', 'MVMOE', 'MoSES_CaDA', 'MoSES_RF', 'reld_mt', 'RouteFinder']
        if names != expected:
            raise ValueError(f'{problem}: frozen filtered candidate pool changed')
    elif 'RELD_MTL' in names or 'RELD_MOEL' in names or 'reld_mt' in names:
        raise ValueError(f'{problem}: unexpected ReLD candidate')
    expected_mask = [int(i in ids) for i in range(len(GLOBAL_SOLVERS))]
    if metadata['mask'] != expected_mask:
        raise ValueError(f'{problem}: pool and global mask disagree')


def build_priors(cache_root, *, data_root=DEFAULT_DATA_ROOT, expected_rows=81000):
    """Sequentially aggregate all 18 TRAIN tasks; optional row count is for tests.

    Source identities are checked from frozen cache metadata, without opening
    any source files referenced by that metadata. Only two small arrays/task
    enter memory, and native winner columns are never recomputed from costs.
    """
    cache_root, data_root = Path(cache_root).resolve(), Path(data_root).resolve()
    if expected_rows <= 0:
        raise ValueError('Expected a positive TRAIN row count')
    tasks, shared_identity = {}, None
    for problem in PROBLEMS:
        directory = cache_root / 'train' / problem
        raw = (directory / 'metadata.json').read_bytes()
        metadata = json.loads(raw)
        if (metadata.get('problem') != problem or metadata.get('problem_id') != P2I[problem]
                or metadata.get('split') != 'train' or metadata.get('rows') != expected_rows
                or metadata.get('rows_written') != expected_rows or metadata.get('complete') is not True
                or metadata.get('cache_format') != 'r76_scenario_v1' or metadata.get('cache_version') != 1
                or Path(metadata['data_root']).resolve() != data_root
                or metadata.get('raw_cost_dtype') != 'float64'
                or metadata.get('winner_source') != 'recomputed FP64 winner in filtered candidate pool'):
            raise ValueError(f'{problem}: expected complete frozen R84 TRAIN cache identity')
        _pool_check(metadata, problem)
        sources = metadata.get('source_files', [])
        if (not metadata.get('scenario_id') or len(sources) < 2
                or Path(sources[0]['path']) != data_root / 'scenario_manifest.json'
                or Path(sources[1]['path']) != data_root / (problem + 'train') / 'metadata.json'
                or any(not isinstance(s.get('size_bytes'), int) or s['size_bytes'] <= 0
                       or not isinstance(s.get('mtime_ns'), int) for s in sources[:2])):
            raise ValueError(f'{problem}: missing frozen source identity')
        identity = dict(data_root=str(data_root), scenario_id=metadata['scenario_id'],
            scenario_source=sources[0], model_global_solvers=metadata['model_global_solvers'],
            scenario_global_solver_ids=metadata['scenario_global_solver_ids'],
            scenario_solver_aliases=metadata['scenario_solver_aliases'])
        if shared_identity is not None and identity != shared_identity:
            raise ValueError(f'{problem}: TRAIN tasks do not share one frozen source identity')
        shared_identity = identity
        arrays = {}
        for key in ('n', 'winner'):
            spec = metadata['arrays'][key]
            if spec != dict(file=key + '.npy', shape=[expected_rows], dtype='int64'):
                raise ValueError(f'{problem}: {key} array specification changed')
            array = np.load(directory / (key + '.npy'), allow_pickle=False)
            if array.shape != (expected_rows,) or array.dtype != np.dtype('int64'):
                raise ValueError(f'{problem}: {key} array shape/dtype changed')
            arrays[key] = array
        n, winner = arrays['n'], arrays['winner']
        k = len(metadata['pool'])
        if (n <= 0).any() or (winner < 0).any() or (winner >= k).any():
            raise ValueError(f'{problem}: invalid exact N or native winner column')
        if (n.min() != metadata['true_length_min'] or n.max() != metadata['true_length_max']):
            raise ValueError(f'{problem}: exact N disagrees with cached source identity')
        values, inverse = np.unique(n, return_inverse=True)
        counts = np.zeros((len(values), k), dtype=np.int64)
        np.add.at(counts, (inverse, winner), 1)
        global_counts = counts.sum(0)
        task_prior = (global_counts + 1.) / (expected_rows + k)
        conditional = (counts + 20. * task_prior) / (counts.sum(1, keepdims=True) + 20.)
        tasks[problem] = dict(problem_id=P2I[problem], rows=expected_rows,
            pool=metadata['pool'], pool_ids=metadata['pool_ids'], n_values=values.tolist(),
            counts_by_n=counts.tolist(), global_counts=global_counts.tolist(),
            task_prior=task_prior.tolist(), conditional_prior=conditional.tolist(),
            source=dict(metadata_sha256=hashlib.sha256(raw).hexdigest(),
                        n_sha256=_array_hash(n), winner_sha256=_array_hash(winner)))
        del arrays, n, winner, array
    record = dict(format=FORMAT, prior_spec=PRIOR_SPEC.copy(), split='train',
        rows_per_problem=expected_rows, source_identity=shared_identity, tasks=tasks)
    record['identity_sha256'] = json_hash(record)
    return record


def validate_priors(record):
    """Validate an in-memory record; factories never need the original files."""
    if (record.get('format') != FORMAT or record.get('prior_spec') != PRIOR_SPEC
            or record.get('split') != 'train' or set(record.get('tasks', {})) != set(PROBLEMS)
            or record.get('identity_sha256') != json_hash({k: v for k, v in record.items()
                                                        if k != 'identity_sha256'})):
        raise ValueError('R108 prior specification, TRAIN scope or content identity changed')
    for problem in PROBLEMS:
        task = record['tasks'][problem]
        pool, ids, values = task['pool'], task['pool_ids'], task['n_values']
        if (task['problem_id'] != P2I[problem] or not values or not pool
                or values != sorted(set(values)) or min(values) <= 0
                or len(ids) != len(pool) or len(set(ids)) != len(ids)
                or any(not isinstance(i, int) or not 0 <= i < len(GLOBAL_SOLVERS) for i in ids)
                or [GLOBAL_SOLVERS[i] for i in ids] != [
                    'RELD_MOEL' if name == 'reld_mt' else name for name in pool]):
            raise ValueError(f'{problem}: invalid prior dimensions or global ID alignment')
        counts = np.asarray(task['counts_by_n'])
        if (counts.shape != (len(values), len(pool)) or counts.dtype.kind not in 'iu'
                or (counts < 0).any() or (counts.sum(1) <= 0).any()
                or counts.sum() != task['rows'] or task['rows'] != record['rows_per_problem']
                or task['global_counts'] != counts.sum(0).tolist()):
            raise ValueError(f'{problem}: invalid TRAIN aggregate counts')
        global_prior = (counts.sum(0) + 1.) / (task['rows'] + len(pool))
        conditional = (counts + 20. * global_prior) / (counts.sum(1, keepdims=True) + 20.)
        if (not np.array_equal(np.asarray(task['task_prior']), global_prior)
                or not np.array_equal(np.asarray(task['conditional_prior']), conditional)):
            raise ValueError(f'{problem}: prior is not Laplace1 then shrink20')
    return record


def prepare(cache_root, root=DEFAULT_ROOT, *, output=None, **kwargs):
    """Write a compact generated JSON, retaining any different existing record."""
    record = build_priors(cache_root, **kwargs)
    destination = Path(output) if output is not None else Path(root) / 'prepared/priors.json'
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if json.loads(destination.read_text()) != record:
            raise ValueError('Existing prior differs; preserve it and use a new output path')
    else:
        with destination.open('x') as stream:
            json.dump(record, stream, sort_keys=True, separators=(',', ':'), allow_nan=False)
            stream.write('\n')
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', nargs='?', choices=('prepare',), default='prepare')
    parser.add_argument('--cache-root', type=Path, required=True)
    parser.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args(argv)
    record = prepare(args.cache_root, args.root, output=args.output)
    print(json.dumps(dict(identity_sha256=record['identity_sha256'], tasks=len(record['tasks']),
                          rows_per_problem=record['rows_per_problem'], split=record['split'])))


if __name__ == '__main__':
    main()
