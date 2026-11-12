"""CPU-only R126 overlay: remove MoSES_RF, never BQ/LEHD or frozen rows.

Inputs/geometry stay in the original R84 cache. ``prepared_config(args)`` and
``open_cache`` plug into a private R84/R83 runner without changing core globals.
Only pure file helpers are shared with the frozen R125 implementation.
"""

import argparse
import copy
import json
import pickle
from pathlib import Path

import numpy as np

from code.unified_selector.registry import GLOBAL_SOLVERS, PROBLEMS, S2I
from .export_new_split_data import B_PROBLEMS, derive_targets, write_pickle
from .new_split_data import scenario_solver_ids
from .pairwise_objective import comparison_targets
from .r75_cache import _open_arrays
from .r75_data import _check_split
from .r76_cache import ScenarioCache, _validate_record
from .r76_experiment import file_hash, read_json, utc, write_json
from .r77_data import fit_sbs
from .r125_data import _cache_directory, _identity, _link, _path, _source_costs, _write_csv


V4 = Path(__file__).resolve().parent
DEFAULT_ROOT = V4 / 'runs/R126_R83_no_moses_rf'
DEFAULT_DATA_ROOT = Path('/public/home/shiys/selection_data_new90k_no_moses_rf_seed2_20261112')
DEFAULT_PREPARED_ROOT = DEFAULT_ROOT / 'prepared'
SOURCE_DATA_ROOT = Path('/public/home/shiys/selection_data_new90k_B_filtered_single_reld_mt_seed2_20261018')
SOURCE_PREPARED_ROOT = V4 / 'runs/R84_new90k_attention/prepared'
DEFAULT_CACHE_ROOT = Path('/tmp/shiys_R84_new90k_cache')
SPLIT_ROWS = dict(train=81000, val=4500, test=4500)
REMOVED = ('MoSES_RF',)
CHANGED_PROBLEMS = ('CVRP', 'OVRP', 'VRPL', 'VRPTW', 'OVRPTW', 'OVRPL', 'VRPLTW', 'OVRPLTW')
SOURCE_SCENARIO_ID = 'new90k_seed2_B_filtered_MOEL_as_reld_mt_v1'
SCENARIO_ID = 'R126_new90k_seed2_B_filtered_single_reld_mt_no_moses_rf_v1'
WINNER_RULE = 'FP64 argmin in retained pool; exact ties use lowest stable global solver ID'
ALIASES = {'reld_mt': 'RELD_MOEL'}


def filter_pool(problem, methods, ids):
    """Remove only MoSES_RF, retaining native column order and global IDs."""
    methods, ids = list(methods), list(ids)
    if (not methods or len(set(methods)) != len(methods)
            or ids != sorted(set(ids)) or len(methods) != len(ids)
            or ids != [S2I[ALIASES.get(name, name)] for name in methods]):
        raise ValueError('Source candidate columns must retain their stable global identities')
    if ('MoSES_RF' in methods) != (problem in CHANGED_PROBLEMS):
        raise ValueError('Expected MoSES_RF only in CVRP and the seven non-B MVRP pools')
    if problem in ('TSP', 'CVRP') and not {'BQ', 'LEHD'}.issubset(methods):
        raise ValueError('R126 must start from the original pool with BQ and LEHD, not R125')
    kept = [i for i, name in enumerate(methods) if name not in REMOVED]
    return [methods[i] for i in kept], [ids[i] for i in kept], kept


def _validate_source(source):
    if (source.get('status') != 'complete' or source.get('scenario_id') != SOURCE_SCENARIO_ID
            or set(source['problems']) != set(PROBLEMS)
            or source['split_sizes_per_problem'] != SPLIT_ROWS):
        raise ValueError('R126 requires the original frozen full18 new90k release, never R125')
    scenario_solver_ids(source)
    if source.get('solver_aliases') != ALIASES:
        raise ValueError('Retain the original single reld_mt deployment')
    for problem in PROBLEMS:
        record = source['problems'][problem]
        filter_pool(problem, record['methods'], record['global_solver_ids'])
        forbidden = {'RELD_MOEL', 'RELD_MTL'}
        if problem in B_PROBLEMS:
            forbidden |= {'RouteFinder', 'MoSES_RF', 'MoSES_CaDA'}
        if forbidden.intersection(record['methods']):
            raise ValueError('Keep the original B8 filtering and reld_mt rules')


def _migration(original, targets, old_methods, methods, baseline, problem, split):
    old_winner, new_winner = original.argmin(axis=1), targets['winner']
    old_oracle = original[np.arange(len(original)), old_winner]
    new_oracle = targets['oracle_cost']
    counts = np.zeros((len(old_methods), len(methods)), dtype=np.int64)
    np.add.at(counts, (old_winner, new_winner), 1)
    transitions = []
    for old_slot, old_method in enumerate(old_methods):
        old_count = int(counts[old_slot].sum())
        for new_slot, method in enumerate(methods):
            count = int(counts[old_slot, new_slot])
            transitions.append(dict(problem=problem, split=split, old_winner=old_method,
                new_winner=method, old_method_removed=old_method in REMOVED,
                instances=len(original), old_winner_count=old_count, count=count,
                pct_all=count / len(original) * 100,
                pct_given_old_winner=count / old_count * 100 if old_count else 0.))
    removed_count = int(sum(counts[i].sum() for i, name in enumerate(old_methods) if name in REMOVED))
    new_counts = np.bincount(new_winner, minlength=len(methods))
    sbs_slot, majority_slot = baseline['sbs_slot'], methods.index(baseline['majority_winner_solver'])
    summary = dict(problem=problem, split=split, instances=len(original),
        old_candidates=len(old_methods), new_candidates=len(methods),
        removed_old_winner_count=removed_count, removed_old_winner_pct=removed_count / len(original) * 100,
        old_oracle_mean_cost=float(old_oracle.mean()), new_oracle_mean_cost=float(new_oracle.mean()),
        oracle_mean_cost_delta=float((new_oracle - old_oracle).mean()),
        oracle_mean_relative_increase_pct=float(((new_oracle - old_oracle) / old_oracle).mean() * 100),
        train_sbs_solver=baseline['sbs_solver'], train_sbs_winner_count=int(new_counts[sbs_slot]),
        train_sbs_winner_pct=float(new_counts[sbs_slot] / len(original) * 100),
        train_majority_solver=baseline['majority_winner_solver'],
        train_majority_winner_count=int(new_counts[majority_slot]),
        train_majority_winner_pct=float(new_counts[majority_slot] / len(original) * 100))
    return transitions, summary


def _export_changed(source, destination, problem, split, methods, ids, kept, baseline, original=None):
    destination.mkdir()
    metadata = read_json(source / 'metadata.json')
    original = _source_costs(source) if original is None else original
    targets = derive_targets(np.ascontiguousarray(original[:, kept]))
    costs, winner = targets['costs'], targets['winner']
    if len(costs) != metadata['instances']:
        raise ValueError('Candidate filtering must preserve every frozen source row')
    targets['sbs_train_cost'] = costs[:, baseline['sbs_slot']]
    transitions, summary = _migration(original, targets, metadata['methods'], methods, baseline, problem, split)
    np.save(destination / 'costs.npy', costs, allow_pickle=False)
    np.save(destination / 'winner.npy', winner, allow_pickle=False)
    np.savez_compressed(destination / 'targets.npz', **targets)
    with (source / 'raw_label.pkl').open('rb') as stream:
        labels = pickle.load(stream)
    raw = {str(row): dict(cost=costs[row].tolist(),
        time=[labels[str(row)]['time'][slot] for slot in kept], ind=int(winner[row]),
        gap=targets['gap_pct'][row].tolist(), rank=targets['rank'][row].tolist(),
        ranking=targets['ranking'][row].tolist(), oracle_cost=float(targets['oracle_cost'][row]),
        relative_regret=targets['relative_regret'][row].tolist()) for row in range(len(costs))}
    write_pickle(destination / 'raw_label.pkl', raw)
    for name in ('dataset.pkl', 'input_metadata.npz', 'index_mapping.csv'):
        _link(source / name, destination / name)
    (destination / 'results').mkdir()
    for method in methods:
        name = 'result_' + method + '.txt'
        _link(source / 'results' / name, destination / 'results' / name)
    posthoc_slot = int(costs.mean(axis=0, dtype=np.float64).argmin())
    metadata.update(scenario_id=SCENARIO_ID, source_methods=metadata['methods'],
        methods=methods, global_solver_ids=ids, retained_source_slots=kept,
        source_release_split=str(source), removed_methods=list(REMOVED),
        candidates=len(methods), finite_cost_values=int(costs.size), nonfinite_cost_values=0,
        cost_dtype='float64', winner_policy=WINNER_RULE,
        oracle_mean_cost=summary['new_oracle_mean_cost'],
        train_sbs_solver=baseline['sbs_solver'], train_sbs_mean_cost=float(targets['sbs_train_cost'].mean()),
        train_sbs_actual_regret_pct=float(targets['relative_regret'][:, baseline['sbs_slot']].mean() * 100),
        train_majority_solver=baseline['majority_winner_solver'],
        train_majority_top1_pct=summary['train_majority_winner_pct'],
        posthoc_split_sbs_solver=methods[posthoc_slot],
        posthoc_split_sbs_mean_cost=float(costs[:, posthoc_slot].mean()),
        exact_min_cost_ties=int(np.count_nonzero((costs == targets['oracle_cost'][:, None]).sum(axis=1) > 1)),
        winner_changed_from_source_pool=int(np.count_nonzero(
            original.argmin(axis=1) != np.asarray(kept)[winner])),
        inputs_reused=True, split_membership_and_order_unchanged=True)
    old_files = metadata['files']
    metadata['files'] = {name: _identity(destination / name)
        for name in ('raw_label.pkl', 'targets.npz', 'costs.npy', 'winner.npy')}
    # Existing identities suffice for linked inputs; never hash/read big datasets.
    metadata['files'].update({name: old_files[name]
        for name in ('dataset.pkl', 'input_metadata.npz', 'index_mapping.csv')})
    metadata['result_files'] = {name: value for name, value in metadata['result_files'].items()
        if name in {'result_' + method + '.txt' for method in methods}}
    write_json(destination / 'metadata.json', metadata)
    return dict(rows=len(costs), pool=methods, pool_ids=ids, artifacts=metadata['files'],
        source_index_sha256=metadata['source_index_sha256'], migration_summary=summary), transitions


def build_release(data_root=SOURCE_DATA_ROOT, output=DEFAULT_DATA_ROOT):
    """Build once from the original R84 release; never write into source/R125."""
    data_root, output = _path(data_root), _path(output)
    if output == data_root or data_root in output.parents:
        raise ValueError('Write an independent overlay, never inside the source release')
    source_path = data_root / 'scenario_manifest.json'
    source = read_json(source_path)
    _validate_source(source)
    digest = file_hash(source_path)
    if output.exists():
        saved = read_json(output / 'scenario_manifest.json')
        if (saved.get('experiment') != 'R126' or saved.get('status') != 'complete'
                or saved.get('parent_data_root') != str(data_root)
                or saved.get('parent_manifest_sha256') != digest):
            raise ValueError('Preserve the existing incomplete or differently bound scenario')
        return saved
    manifest = copy.deepcopy(source)
    manifest.update(experiment='R126', scenario_id=SCENARIO_ID, status='building', created_utc=utc(),
        parent_data_root=str(data_root), source_data_root=str(data_root), parent_manifest_sha256=digest,
        removed_methods=sorted(set(source.get('removed_methods', [])) | set(REMOVED)),
        candidate_filter=dict(removed=list(REMOVED), problems=list(CHANGED_PROBLEMS),
            protected_methods=['BQ', 'LEHD'], all_instances_retained=True,
            source_split_order_preserved=True, other_candidate_rules_unchanged=True),
        original_files_modified=False, inputs_modified=False, solver_runs=0, winner_rule=WINNER_RULE,
        test_label_conversion='mechanical only; not used for fitting or model selection')
    manifest['global_solvers'] = [name for name in source['global_solvers'] if name not in REMOVED]
    manifest['global_solver_ids'] = {name: source['global_solver_ids'][name] for name in manifest['global_solvers']}
    manifest['reserved_inactive_global_solver_ids'] = sorted(set(
        source.get('reserved_inactive_global_solver_ids', [])) | {source['global_solver_ids'][name] for name in REMOVED})
    scenario_solver_ids(manifest)
    output.mkdir(parents=True)
    write_json(output / 'scenario_manifest.json', manifest)
    for name in ('split_indices.npz', 'split_manifest.json', 'holdout_extensions.npz'):
        if (data_root / name).exists():
            _link(data_root / name, output / name)
    baselines = copy.deepcopy(read_json(data_root / 'baselines.json'))
    receipt = dict(experiment='R126', parent_data_root=str(data_root), parent_manifest_sha256=digest,
        prepared_utc=utc(), source_inputs_read=False, gpu_used=False, training_started=False,
        sbs_refitted_problems=list(CHANGED_PROBLEMS),
        sbs_unchanged_train_fit_reused=[p for p in PROBLEMS if p not in CHANGED_PROBLEMS],
        split_membership='all rows and order retained by input/index symlinks', splits={})
    transitions, summaries = [], []
    for problem in PROBLEMS:
        record = manifest['problems'][problem]
        methods, ids, kept = filter_pool(problem, record['methods'], record['global_solver_ids'])
        original = None
        if problem in CHANGED_PROBLEMS:
            original = _source_costs(data_root / (problem + 'train'))
            targets = derive_targets(np.ascontiguousarray(original[:, kept]))
            baselines[problem] = fit_sbs(targets['costs'], methods, targets['winner'])
            record.update(methods=methods, global_solver_ids=ids,
                parent_methods=source['problems'][problem]['methods'], retained_parent_slots=kept,
                retained_source_slots=[source['problems'][problem]['retained_source_slots'][i] for i in kept],
                removed_methods=sorted(set(record.get('removed_methods', [])) | set(REMOVED)),
                train_sbs=baselines[problem])
        if baselines[problem]['selection_split'] != 'train':
            raise ValueError('SBS must be selected only on frozen TRAIN rows')
        for split, rows in SPLIT_ROWS.items():
            src, dst = data_root / (problem + split), output / (problem + split)
            if problem in CHANGED_PROBLEMS:
                result, migration = _export_changed(src, dst, problem, split, methods, ids, kept,
                    baselines[problem], original=original if split == 'train' else None)
                if result['rows'] != rows:
                    raise ValueError('Frozen split row count changed')
                transitions.extend(migration)
                summaries.append(result['migration_summary'])
                receipt['splits'][problem + split] = result
            else:
                _link(src, dst)
        print(f'[R126 data] {problem}: {len(methods)} candidates; all split rows retained', flush=True)
    write_json(output / 'baselines.json', baselines)
    _write_csv(output / 'winner_migration.csv', transitions)
    _write_csv(output / 'candidate_filter_summary.csv', summaries)
    write_json(output / 'data_preparation.json', receipt)
    manifest.update(status='complete', completed_utc=utc(),
        data_preparation_sha256=file_hash(output / 'data_preparation.json'))
    write_json(output / 'scenario_manifest.json', manifest)
    return manifest


def _overlay_record(source, scenario, data_root, directory):
    record = copy.deepcopy(source)
    problem = record['problem']
    methods, ids, _ = filter_pool(problem, source['pool'], source['pool_ids'])
    if (scenario['problems'][problem]['methods'] != methods
            or scenario['problems'][problem]['global_solver_ids'] != ids or not source.get('complete')):
        raise ValueError('R126 must be exactly the original R84 pool minus MoSES_RF')
    record.update(data_root=str(data_root), scenario_id=scenario['scenario_id'],
        scenario_global_solvers=scenario['global_solvers'], scenario_global_solver_ids=scenario['global_solver_ids'],
        model_global_solvers=list(GLOBAL_SOLVERS), scenario_solver_aliases=scenario.get('solver_aliases', {}),
        pool=methods, pool_ids=ids, mask=[int(i in ids) for i in range(len(GLOBAL_SOLVERS))],
        winner_source=WINNER_RULE, candidate_overlay='r126', input_cache_reused=True,
        source_cache_directory=str(directory.resolve()), source_cache_data_root=source['data_root'],
        source_metadata_sha256=file_hash(directory / 'metadata.json') if (directory / 'metadata.json').is_file() else None)
    record['arrays']['costs']['shape'] = [record['rows'], len(methods)]
    return record


def prepare(root=DEFAULT_PREPARED_ROOT, data_root=DEFAULT_DATA_ROOT,
            source_prepared=SOURCE_PREPARED_ROOT, cache_root=DEFAULT_CACHE_ROOT):
    """Bind TRAIN-only state and input cache references, without GPU work."""
    root, data_root, source_prepared, cache_root = map(_path, (root, data_root, source_prepared, cache_root))
    if not (data_root / 'scenario_manifest.json').exists():
        build_release(output=data_root)
    scenario = read_json(data_root / 'scenario_manifest.json')
    digest = file_hash(data_root / 'scenario_manifest.json')
    if scenario.get('experiment') != 'R126' or scenario.get('status') != 'complete':
        raise ValueError('Prepare a completed R126 overlay first')
    ready_path = root / 'prepared.json'
    if ready_path.exists():
        ready = read_json(ready_path)
        expected = dict(data_root=str(data_root), cache_source=str(cache_root),
            source_prepared_root=str(source_prepared), scenario_sha256=digest, experiment='R126')
        if any(ready.get(key) != value for key, value in expected.items()):
            raise ValueError('Preserve existing preparation bound to another release/cache')
        for name, value in ready['artifact_sha256'].items():
            if file_hash(root / name) != value:
                raise ValueError('Prepared artifact changed: ' + name)
        return ready
    source_ready = read_json(source_prepared / 'prepared.json')
    source_manifest = read_json(source_prepared / 'source_cache_manifest.json')
    if (source_ready.get('experiment') != 'R84' or source_ready.get('seed') != 2
            or source_ready['data_root'] != scenario['parent_data_root']
            or source_ready['scenario_sha256'] != scenario['parent_manifest_sha256']
            or source_ready['rows_per_problem'] != SPLIT_ROWS):
        raise ValueError('Reuse only the original R84 frozen new90k TRAIN preparation')
    root.mkdir(parents=True, exist_ok=True)
    splits, missing_inputs = {split: {} for split in SPLIT_ROWS}, []
    for split, rows in SPLIT_ROWS.items():
        for problem in PROBLEMS:
            recorded = source_manifest['splits'][split][problem]
            fallback = Path(source_ready['cache_source']) / split / problem
            try:
                directory = _cache_directory(cache_root, split, problem, fallback)
                source = read_json(directory / 'metadata.json')
            except FileNotFoundError:
                if split != 'test':
                    raise
                directory, source = Path(recorded['directory']), recorded
                missing_inputs.append(problem + '/' + split)
            _validate_record(source, directory)
            if (source['rows'] != rows or source['problem'] != problem or source['split'] != split
                    or _path(source['data_root']) != _path(scenario['parent_data_root'])
                    or source['scenario_id'] != SOURCE_SCENARIO_ID):
                raise ValueError(f'Original R84 cache rows/release mismatch: {problem}/{split}')
            splits[split][problem] = _overlay_record(source, scenario, data_root, directory)
    manifest = dict(experiment='R126', complete=True, candidate_overlay='r126',
        data_root=str(data_root), source_data_root=scenario['parent_data_root'], cache_root=str(cache_root),
        scenario_id=scenario['scenario_id'], scenario_sha256=digest,
        scenario_global_solvers=scenario['global_solvers'], scenario_global_solver_ids=scenario['global_solver_ids'],
        scenario_solver_aliases=scenario.get('solver_aliases', {}), model_global_solvers=list(GLOBAL_SOLVERS),
        splits=splits, input_arrays_copied=False, missing_input_caches=missing_inputs, test_inputs_ready=not missing_inputs)
    write_json(root / 'source_cache_manifest.json', manifest)
    for name in ('geometry_stats.json', 'identicalinit_seed2.pt'):
        source = source_prepared / name
        if file_hash(source) != source_ready['artifact_sha256'][name]:
            raise ValueError('Locked R84 TRAIN artifact changed: ' + name)
        _link(source, root / name)
    _link(data_root / 'baselines.json', root / 'common_full_train_sbs.json')
    names = ('source_cache_manifest.json', 'geometry_stats.json', 'identicalinit_seed2.pt', 'common_full_train_sbs.json')
    ready = dict(experiment='R126', seed=2, created_utc=utc(), data_root=str(data_root),
        source_data_root=scenario['parent_data_root'], source_prepared_root=str(source_prepared),
        scenario_sha256=digest, cache_source=str(cache_root), rows_per_problem=SPLIT_ROWS,
        model_params=source_ready['model_params'], trainable_sha256=source_ready['trainable_sha256'],
        fresh_initialization_source=str((source_prepared / 'identicalinit_seed2.pt').resolve()),
        initialization='scratch', historical_checkpoint_loaded=False,
        geometry_source=str((source_prepared / 'geometry_stats.json').resolve()),
        artifact_sha256={name: file_hash(root / name) for name in names}, winner_rule=WINNER_RULE,
        fitted_on='same frozen 81000 TRAIN rows; changed-pool SBS refitted after filtering',
        test_labels='mechanical candidate conversion only; no TEST fitting or selection',
        input_arrays_copied=False, gpu_used=False, training_started=False,
        missing_input_caches=missing_inputs, test_inputs_ready=not missing_inputs)
    write_json(ready_path, ready)
    return ready


def prepared_config(args):
    """Return ready, manifest, geometry, baselines; small checks only at startup."""
    root = _path(getattr(args, 'prepared_root', None) or DEFAULT_PREPARED_ROOT)
    cache_root = getattr(args, 'cache_root', None) or DEFAULT_CACHE_ROOT
    ready = prepare(root=root, data_root=getattr(args, 'data_root', None) or DEFAULT_DATA_ROOT,
        source_prepared=getattr(args, 'source_prepared_root', None) or SOURCE_PREPARED_ROOT, cache_root=cache_root)
    if getattr(args, 'cache_root', None) is None:
        args.cache_root = Path(ready['cache_source'])
    return (ready, read_json(root / 'source_cache_manifest.json'),
            read_json(root / 'geometry_stats.json'), read_json(root / 'common_full_train_sbs.json'))


class CandidateCache(ScenarioCache):
    """R76 batch implementation, original feature memmaps, R126 supervision."""

    def __init__(self, directory, metadata, data_root):
        self.directory, self.metadata = Path(directory), copy.deepcopy(metadata)
        self._arrays, self._targets, self.raw, self.lengths = {}, {}, {}, None
        self.problem, self.split = metadata['problem'], metadata['split']
        self.size, self.kind = metadata['rows'], metadata['kind']
        self.pool, self.pool_ids = list(metadata['pool']), list(metadata['pool_ids'])
        source = read_json(self.directory / 'metadata.json')
        if (metadata['source_metadata_sha256'] is not None
                and file_hash(self.directory / 'metadata.json') != metadata['source_metadata_sha256']):
            raise ValueError('Original R84 cache metadata changed since R126 preparation')
        _check_split(self.split, source.get('test_lock_root'))
        specs = _validate_record(source, self.directory)
        expected, expected_ids, _ = filter_pool(self.problem, source['pool'], source['pool_ids'])
        if (self.pool != expected or self.pool_ids != expected_ids
                or source['rows'] != self.size or source['split'] != self.split):
            raise ValueError('R126 cache candidate/row identity mismatch')
        changed = self.problem in CHANGED_PROBLEMS
        try:
            self._arrays = _open_arrays(self.directory, {key: spec for key, spec in specs.items()
                if not changed or key not in ('costs', 'winner')})
            if changed:
                labels = Path(data_root) / (self.problem + self.split)
                self._arrays.update(_open_arrays(labels, {key: metadata['arrays'][key] for key in ('costs', 'winner')}))
            self.lengths = self._arrays['n']
            self.raw = dict(costs=self._arrays['costs'], winner=self._arrays['winner'], pool=self.pool, pool_ids=self.pool_ids)
            if (self.raw['costs'].dtype != np.float64 or self.raw['winner'].dtype != np.int64
                    or self.raw['costs'].shape != (self.size, len(self.pool))
                    or len(self.lengths) != self.size
                    or not np.array_equal(self.raw['winner'], self.raw['costs'].argmin(axis=1))):
                raise ValueError('Preserve FP64 costs and rebuilt int64 native-slot winners')
            self._targets = {key: value.contiguous().numpy() for key, value in comparison_targets(self.raw['costs']).items()}
        except BaseException:
            self.close()
            raise


def open_cache(cache_root, split, problem, manifest, data_root):
    """R76-compatible API; cost/pool overlays never copy or re-encode inputs."""
    split = 'val' if str(split).lower() == 'validation' else str(split).lower()
    problem = str(problem).upper()
    if manifest.get('candidate_overlay') != 'r126' or _path(manifest['data_root']) != _path(data_root):
        raise ValueError('Cache opening requires this R126 overlay manifest')
    metadata = manifest['splits'][split][problem]
    directory = _cache_directory(cache_root, split, problem, metadata['source_cache_directory'])
    return CandidateCache(directory, metadata, data_root)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('build', 'prepare'), nargs='?', default='prepare')
    parser.add_argument('--data-root', type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument('--source-data-root', type=Path, default=SOURCE_DATA_ROOT)
    parser.add_argument('--prepared-root', type=Path, default=DEFAULT_PREPARED_ROOT)
    parser.add_argument('--source-prepared-root', type=Path, default=SOURCE_PREPARED_ROOT)
    parser.add_argument('--cache-root', type=Path, default=DEFAULT_CACHE_ROOT)
    args = parser.parse_args(argv)
    build_release(args.source_data_root, args.data_root)
    if args.command == 'prepare':
        prepare(args.prepared_root, args.data_root, args.source_prepared_root, args.cache_root)
    print(json.dumps(dict(data_root=str(args.data_root), prepared_root=str(args.prepared_root)), indent=2))


if __name__ == '__main__':
    import torch
    torch.set_num_threads(2)
    main()
