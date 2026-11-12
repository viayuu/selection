"""CPU-only R125 candidate overlay; source rows and input caches stay untouched.

``prepared_config(args)`` returns ready, manifest, geometry, baselines. Bind
``open_cache`` in the private runner, keeping the original R84 cache root.
TEST labels are mechanically converted, never used to fit preparation state.
"""

import argparse
import copy
import csv
import json
from pathlib import Path

import numpy as np

from code.unified_selector.registry import GLOBAL_SOLVERS, PROBLEMS
from .export_new_split_data import derive_targets, write_pickle
from .new_split_data import scenario_solver_ids
from .pairwise_objective import comparison_targets
from .r75_cache import _open_arrays
from .r75_data import _check_split
from .r76_cache import ScenarioCache, _validate_record
from .r76_experiment import file_hash, read_json, utc, write_json
from .r77_data import fit_sbs


V4 = Path(__file__).resolve().parent
DEFAULT_ROOT = V4 / 'runs/R125_no_bq_lehd_source_embedding'
DEFAULT_DATA_ROOT = Path('/public/home/shiys/selection_data_new90k_no_bq_lehd_seed2_20261112')
DEFAULT_PREPARED_ROOT = DEFAULT_ROOT / 'prepared'
SOURCE_DATA_ROOT = Path('/public/home/shiys/selection_data_new90k_B_filtered_single_reld_mt_seed2_20261018')
SOURCE_PREPARED_ROOT = V4 / 'runs/R84_new90k_attention/prepared'
SOURCE_TEST_CACHE_ROOT = V4 / 'runs/R120_multidomain_selector/existing_test_cache'
DEFAULT_CACHE_ROOT = Path('/tmp/shiys_R84_new90k_cache')
SPLIT_ROWS = dict(train=81000, val=4500, test=4500)
REMOVED = ('BQ', 'LEHD')
CHANGED_PROBLEMS = ('TSP', 'CVRP')
SCENARIO_ID = 'R125_new90k_seed2_B_filtered_single_reld_mt_no_bq_lehd_v1'
WINNER_RULE = 'FP64 argmin in retained pool; exact ties use lowest stable global solver ID'


def _path(value):
    return Path(value).expanduser().resolve()


def _link(source, destination):
    source, destination = _path(source), Path(destination)
    if not source.exists():
        raise FileNotFoundError(source)
    if destination.is_symlink() and destination.resolve() == source:
        return
    if destination.exists() or destination.is_symlink():
        raise FileExistsError('Preserve existing artifact: ' + str(destination))
    destination.symlink_to(source, target_is_directory=source.is_dir())


def _identity(path):
    path = Path(path)
    return dict(path=str(path.resolve()), bytes=path.stat().st_size, sha256=file_hash(path))


def _write_csv(path, rows):
    with Path(path).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _migration(original, targets, old_methods, methods, baseline, problem, split):
    old_winner = original.argmin(axis=1)
    new_winner, new_oracle = targets['winner'], targets['oracle_cost']
    old_oracle = original[np.arange(len(original)), old_winner]
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
    sbs_slot = baseline['sbs_slot']
    majority_slot = methods.index(baseline['majority_winner_solver'])
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


def filter_pool(problem, methods, ids):
    """Keep source order/global IDs, removing deployments only where legal."""
    methods, ids = list(methods), list(ids)
    if len(methods) != len(ids) or len(set(methods)) != len(methods) or ids != sorted(set(ids)):
        raise ValueError('Source candidate columns need distinct, stable global IDs')
    present = set(methods).intersection(REMOVED)
    if (problem in CHANGED_PROBLEMS and present != set(REMOVED)) or (
            problem not in CHANGED_PROBLEMS and present):
        raise ValueError('BQ/LEHD must occur only in both TSP/CVRP source pools')
    kept = [i for i, name in enumerate(methods) if name not in REMOVED]
    if not kept:
        raise ValueError('Cannot remove every candidate')
    return [methods[i] for i in kept], [ids[i] for i in kept], kept


def _source_costs(directory):
    with np.load(directory / 'targets.npz', allow_pickle=False) as source:
        costs = source['costs']
    if costs.dtype != np.float64:
        raise ValueError('Do not recover FP64 costs from FP32 labels')
    return costs


def _export_changed(source, destination, problem, split, methods, ids, kept, baseline, scenario_id):
    destination.mkdir()
    metadata = read_json(source / 'metadata.json')
    original = _source_costs(source)
    targets = derive_targets(np.ascontiguousarray(original[:, kept]))
    costs, winner = targets['costs'], targets['winner']
    if len(costs) != metadata['instances']:
        raise ValueError('Filtering must preserve every source row')
    targets['sbs_train_cost'] = costs[:, baseline['sbs_slot']]
    transitions, migration_summary = _migration(original, targets, metadata['methods'],
        methods, baseline, problem, split)
    np.save(destination / 'costs.npy', costs, allow_pickle=False)
    np.save(destination / 'winner.npy', winner, allow_pickle=False)
    np.savez_compressed(destination / 'targets.npz', **targets)
    # Only labels are small materializations. Never unpickle/copy input datasets.
    import pickle
    with (source / 'raw_label.pkl').open('rb') as stream:
        labels = pickle.load(stream)
    raw = {}
    for row in range(len(costs)):
        raw[str(row)] = dict(cost=costs[row].tolist(),
            time=[labels[str(row)]['time'][slot] for slot in kept], ind=int(winner[row]),
            gap=targets['gap_pct'][row].tolist(), rank=targets['rank'][row].tolist(),
            ranking=targets['ranking'][row].tolist(), oracle_cost=float(targets['oracle_cost'][row]),
            relative_regret=targets['relative_regret'][row].tolist())
    write_pickle(destination / 'raw_label.pkl', raw)
    for name in ('dataset.pkl', 'input_metadata.npz', 'index_mapping.csv'):
        _link(source / name, destination / name)
    (destination / 'results').mkdir()
    for method in methods:
        _link(source / 'results' / ('result_' + method + '.txt'),
              destination / 'results' / ('result_' + method + '.txt'))
    majority = baseline['majority_winner_solver']
    posthoc_slot = int(costs.mean(axis=0, dtype=np.float64).argmin())
    metadata.update(scenario_id=scenario_id, methods=methods, global_solver_ids=ids,
        source_methods=metadata['methods'], retained_source_slots=kept,
        source_release_split=str(source), removed_methods=list(REMOVED),
        candidates=len(methods), finite_cost_values=int(costs.size), nonfinite_cost_values=0,
        cost_dtype='float64', winner_policy=WINNER_RULE,
        oracle_mean_cost=float(targets['oracle_cost'].mean()),
        train_sbs_solver=baseline['sbs_solver'],
        train_sbs_mean_cost=float(targets['sbs_train_cost'].mean()),
        train_sbs_actual_regret_pct=float(targets['relative_regret'][:, baseline['sbs_slot']].mean() * 100),
        train_majority_solver=majority,
        train_majority_top1_pct=float(np.mean(winner == methods.index(majority)) * 100),
        posthoc_split_sbs_solver=methods[posthoc_slot],
        posthoc_split_sbs_mean_cost=float(costs[:, posthoc_slot].mean()),
        exact_min_cost_ties=int(np.count_nonzero((costs == targets['oracle_cost'][:, None]).sum(axis=1) > 1)),
        winner_changed_from_source_pool=int(np.count_nonzero(
            original.argmin(axis=1) != np.asarray(kept)[winner])),
        inputs_reused=True, split_membership_and_order_unchanged=True)
    metadata['files'] = {
        name: _identity(destination / name) for name in
        ('raw_label.pkl', 'targets.npz', 'costs.npy', 'winner.npy')}
    # Keep the original recorded identities of linked, potentially large inputs.
    old_files = read_json(source / 'metadata.json')['files']
    metadata['files'].update({name: old_files[name] for name in
        ('dataset.pkl', 'input_metadata.npz', 'index_mapping.csv')})
    metadata['result_files'] = {name: value for name, value in metadata['result_files'].items()
        if name in {'result_' + method + '.txt' for method in methods}}
    write_json(destination / 'metadata.json', metadata)
    return dict(rows=len(costs), pool=methods, pool_ids=ids,
        source_index_sha256=metadata['source_index_sha256'],
        oracle_mean_cost=metadata['oracle_mean_cost'],
        artifacts=metadata['files'], winner_migration=transitions, migration_summary=migration_summary)


def build_release(data_root=SOURCE_DATA_ROOT, output=DEFAULT_DATA_ROOT):
    """Create one immutable, input-linked scenario; safely reuse a finished one."""
    data_root, output = _path(data_root), _path(output)
    if output == data_root or data_root in output.parents:
        raise ValueError('Write an independent scenario, never into the source release')
    source_manifest = data_root / 'scenario_manifest.json'
    source = read_json(source_manifest)
    if source.get('status') != 'complete' or set(source['problems']) != set(PROBLEMS):
        raise ValueError('Expected a complete frozen full18 release')
    scenario_solver_ids(source)
    if source['split_sizes_per_problem'] != SPLIT_ROWS:
        raise ValueError('Keep the frozen 81000/4500/4500 populations')
    source_digest = file_hash(source_manifest)
    if output.exists():
        saved = read_json(output / 'scenario_manifest.json')
        if (saved.get('status') != 'complete' or saved.get('experiment') != 'R125'
                or saved.get('parent_data_root') != str(data_root)
                or saved.get('parent_manifest_sha256') != source_digest):
            raise ValueError('Existing scenario is incomplete or belongs to another source; preserved')
        return saved
    manifest = copy.deepcopy(source)
    manifest.update(experiment='R125', scenario_id=SCENARIO_ID, status='building', created_utc=utc(),
        parent_data_root=str(data_root), parent_manifest_sha256=source_digest,
        removed_methods=sorted(set(source.get('removed_methods', [])) | set(REMOVED)),
        candidate_filter=dict(removed=list(REMOVED), problems=list(CHANGED_PROBLEMS),
            all_instances_retained=True, source_split_order_preserved=True,
            other_candidate_rules_unchanged=True),
        original_files_modified=False, inputs_modified=False, solver_runs=0,
        source_data_root=str(data_root),
        winner_rule=WINNER_RULE, test_label_conversion='mechanical only; not used for fitting or model selection')
    manifest['global_solvers'] = [name for name in source['global_solvers'] if name not in REMOVED]
    manifest['global_solver_ids'] = {name: source['global_solver_ids'][name]
        for name in manifest['global_solvers']}
    manifest['reserved_inactive_global_solver_ids'] = sorted(set(
        source.get('reserved_inactive_global_solver_ids', [])) | {
            source['global_solver_ids'][name] for name in REMOVED})
    scenario_solver_ids(manifest)
    baselines = copy.deepcopy(read_json(data_root / 'baselines.json'))
    plans = {}
    transitions, summaries = [], []
    for problem in PROBLEMS:
        record = source['problems'][problem]
        methods, ids, kept = filter_pool(problem, record['methods'], record['global_solver_ids'])
        plans[problem] = methods, ids, kept
        if problem in CHANGED_PROBLEMS:
            costs = _source_costs(data_root / (problem + 'train'))[:, kept]
            baselines[problem] = fit_sbs(costs, methods, derive_targets(costs)['winner'])
        if baselines[problem]['selection_split'] != 'train':
            raise ValueError('All SBS identities must be fitted on TRAIN only')
    output.mkdir(parents=True)
    write_json(output / 'scenario_manifest.json', manifest)
    for name in ('split_indices.npz', 'split_manifest.json', 'holdout_extensions.npz'):
        if (data_root / name).exists():
            _link(data_root / name, output / name)
    receipt = dict(experiment='R125', parent_data_root=str(data_root),
        parent_manifest_sha256=source_digest, prepared_utc=utc(),
        source_inputs_read=False, gpu_used=False, training_started=False,
        sbs_refitted_problems=list(CHANGED_PROBLEMS),
        sbs_unchanged_train_fit_reused=[p for p in PROBLEMS if p not in CHANGED_PROBLEMS],
        split_membership='all rows and order retained by input/index symlinks', splits={})
    for problem in PROBLEMS:
        methods, ids, kept = plans[problem]
        record = manifest['problems'][problem]
        if problem in CHANGED_PROBLEMS:
            record.update(methods=methods, global_solver_ids=ids,
                parent_methods=source['problems'][problem]['methods'], retained_parent_slots=kept,
                retained_source_slots=[source['problems'][problem]['retained_source_slots'][i] for i in kept],
                removed_methods=sorted(set(record.get('removed_methods', [])) | set(REMOVED)),
                train_sbs=baselines[problem])
        for split, rows in SPLIT_ROWS.items():
            src, dst = data_root / (problem + split), output / (problem + split)
            if problem in CHANGED_PROBLEMS:
                result = _export_changed(src, dst, problem, split, methods, ids, kept,
                                         baselines[problem], SCENARIO_ID)
                if result['rows'] != rows:
                    raise ValueError('Frozen split row count changed')
                transitions.extend(result.pop('winner_migration'))
                summaries.append(result['migration_summary'])
                receipt['splits'][problem + split] = result
            else:
                _link(src, dst)
        print(f'[R125 data] {problem}: {len(methods)} candidates; all split rows retained', flush=True)
    write_json(output / 'baselines.json', baselines)
    _write_csv(output / 'winner_migration.csv', transitions)
    _write_csv(output / 'candidate_filter_summary.csv', summaries)
    write_json(output / 'data_preparation.json', receipt)
    manifest.update(status='complete', completed_utc=utc(),
        data_preparation_sha256=file_hash(output / 'data_preparation.json'))
    write_json(output / 'scenario_manifest.json', manifest)
    return manifest


def _cache_directory(cache_root, split, problem, fallback=None):
    candidates = (Path(cache_root) / split / problem, Path(cache_root) / (problem + split))
    if fallback is not None:
        candidates += (Path(fallback),)
    directory = next((path for path in candidates if (path / 'metadata.json').is_file()), None)
    if directory is None:
        raise FileNotFoundError(f'Missing original R84 input cache: {problem}/{split} under {cache_root}')
    return directory


def _overlay_record(source, scenario, data_root, directory):
    record = copy.deepcopy(source)
    problem = record['problem']
    methods = scenario['problems'][problem]['methods']
    ids = scenario['problems'][problem]['global_solver_ids']
    expected, expected_ids, _ = filter_pool(problem, source['pool'], source['pool_ids'])
    if methods != expected or ids != expected_ids or not source.get('complete'):
        raise ValueError('R125 pool must be exactly the requested R84 candidate filter')
    record.update(data_root=str(data_root), scenario_id=scenario['scenario_id'],
        scenario_global_solvers=scenario['global_solvers'],
        scenario_global_solver_ids=scenario['global_solver_ids'],
        model_global_solvers=list(GLOBAL_SOLVERS),
        scenario_solver_aliases=scenario.get('solver_aliases', {}),
        pool=methods, pool_ids=ids, mask=[int(i in ids) for i in range(len(GLOBAL_SOLVERS))],
        winner_source=WINNER_RULE, source_cache_directory=str(directory.resolve()),
        source_cache_data_root=source['data_root'], input_cache_reused=True,
        candidate_overlay='r125', source_metadata_sha256=(file_hash(directory / 'metadata.json')
            if (directory / 'metadata.json').is_file() else None))
    record['arrays']['costs']['shape'] = [record['rows'], len(methods)]
    return record


def prepare(root=DEFAULT_PREPARED_ROOT, data_root=DEFAULT_DATA_ROOT,
            source_prepared=SOURCE_PREPARED_ROOT, cache_root=DEFAULT_CACHE_ROOT):
    """Bind the overlay to source input metadata, TRAIN moments and scratch init."""
    root, data_root, source_prepared, cache_root = map(_path,
        (root, data_root, source_prepared, cache_root))
    if not (data_root / 'scenario_manifest.json').exists():
        build_release(output=data_root)
    scenario = read_json(data_root / 'scenario_manifest.json')
    scenario_digest = file_hash(data_root / 'scenario_manifest.json')
    if scenario.get('experiment') != 'R125' or scenario.get('status') != 'complete':
        raise ValueError('Prepare a complete R125 overlay first')
    ready_path = root / 'prepared.json'
    if ready_path.exists():
        ready = read_json(ready_path)
        if any(ready.get(key) != value for key, value in dict(
                data_root=str(data_root), cache_source=str(cache_root),
                source_prepared_root=str(source_prepared), scenario_sha256=scenario_digest).items()):
            raise ValueError('Preserve prepared artifacts bound to another source/configuration')
        for name, digest in ready['artifact_sha256'].items():
            if file_hash(root / name) != digest:
                raise ValueError('Prepared artifact changed: ' + name)
        return ready
    source_ready = read_json(source_prepared / 'prepared.json')
    source_manifest = read_json(source_prepared / 'source_cache_manifest.json')
    if (source_ready['data_root'] != scenario['parent_data_root']
            or source_ready['scenario_sha256'] != scenario['parent_manifest_sha256']
            or source_ready['rows_per_problem'] != SPLIT_ROWS or source_ready.get('seed') != 2):
        raise ValueError('Source preparation must use this exact R84 frozen release and seed2')
    root.mkdir(parents=True, exist_ok=True)
    splits = {split: {} for split in SPLIT_ROWS}
    missing_inputs = []
    for split, rows in SPLIT_ROWS.items():
        for problem in PROBLEMS:
            recorded = source_manifest['splits'][split][problem]
            fallback = Path(source_ready['cache_source']) / split / problem
            if split == 'test' and not (fallback / 'metadata.json').is_file():
                fallback = SOURCE_TEST_CACHE_ROOT / split / problem
            try:
                directory = _cache_directory(cache_root, split, problem, fallback)
                source = read_json(directory / 'metadata.json')
            except FileNotFoundError:
                if split != 'test':
                    raise
                # Missing disposable TEST features must not block TRAIN/VAL.
                # Preserve historical metadata, explicitly marking it unavailable.
                directory = Path(recorded['directory'])
                source = recorded
                missing_inputs.append(problem + '/' + split)
            _validate_record(source, directory)
            if (source['rows'] != rows or source['problem'] != problem or source['split'] != split
                    or _path(source['data_root']) != _path(scenario['parent_data_root'])):
                raise ValueError(f'Original R84 cache rows/source mismatch: {problem}/{split}')
            splits[split][problem] = _overlay_record(source, scenario, data_root, directory)
    manifest = dict(experiment='R125', complete=True, candidate_overlay='r125',
        data_root=str(data_root), cache_root=str(cache_root),
        scenario_id=scenario['scenario_id'], scenario_sha256=scenario_digest,
        scenario_solver_aliases=scenario.get('solver_aliases', {}),
        model_global_solvers=list(GLOBAL_SOLVERS), splits=splits,
        source_data_root=scenario['parent_data_root'], input_arrays_copied=False,
        missing_input_caches=missing_inputs, test_inputs_ready=not missing_inputs)
    write_json(root / 'source_cache_manifest.json', manifest)
    for name in ('geometry_stats.json', 'identicalinit_seed2.pt'):
        source = source_prepared / name
        if file_hash(source) != source_ready['artifact_sha256'][name]:
            raise ValueError('Locked R84 preparation artifact changed: ' + name)
        _link(source, root / name)
    _link(data_root / 'baselines.json', root / 'common_full_train_sbs.json')
    names = ('source_cache_manifest.json', 'geometry_stats.json',
             'identicalinit_seed2.pt', 'common_full_train_sbs.json')
    ready = dict(experiment='R125', seed=2, created_utc=utc(), data_root=str(data_root),
        source_data_root=scenario['parent_data_root'],
        source_prepared_root=str(source_prepared), scenario_sha256=scenario_digest,
        cache_source=str(cache_root), rows_per_problem=SPLIT_ROWS,
        model_params=source_ready['model_params'], trainable_sha256=source_ready['trainable_sha256'],
        fresh_initialization_source=str((source_prepared / 'identicalinit_seed2.pt').resolve()),
        initialization='scratch', historical_checkpoint_loaded=False,
        geometry_source=str((source_prepared / 'geometry_stats.json').resolve()),
        artifact_sha256={name: file_hash(root / name) for name in names},
        winner_rule=WINNER_RULE, fitted_on='same frozen 81000 TRAIN rows; SBS refitted after filtering',
        test_labels='mechanical candidate conversion only; no TEST fitting or selection',
        input_arrays_copied=False, gpu_used=False, training_started=False,
        missing_input_caches=missing_inputs, test_inputs_ready=not missing_inputs)
    write_json(ready_path, ready)
    return ready


def prepared_config(args):
    """R84-compatible four-tuple, validated once at runner startup, not per step."""
    root = _path(getattr(args, 'prepared_root', DEFAULT_PREPARED_ROOT))
    ready = prepare(root=root, data_root=getattr(args, 'data_root', DEFAULT_DATA_ROOT),
        source_prepared=getattr(args, 'source_prepared_root', SOURCE_PREPARED_ROOT),
        cache_root=getattr(args, 'cache_root', DEFAULT_CACHE_ROOT))
    return (ready, read_json(root / 'source_cache_manifest.json'),
            read_json(root / 'geometry_stats.json'), read_json(root / 'common_full_train_sbs.json'))


class CandidateCache(ScenarioCache):
    """Reuse R76 gathers with original feature memmaps and new FP64 supervision."""

    def __init__(self, directory, metadata, data_root):
        self.directory, self.metadata = Path(directory), copy.deepcopy(metadata)
        self._arrays, self._targets, self.raw, self.lengths = {}, {}, {}, None
        self.problem, self.split = metadata['problem'], metadata['split']
        self.size, self.kind = metadata['rows'], metadata['kind']
        self.pool, self.pool_ids = list(metadata['pool']), list(metadata['pool_ids'])
        source = read_json(self.directory / 'metadata.json')
        if (metadata['source_metadata_sha256'] is not None
                and file_hash(self.directory / 'metadata.json') != metadata['source_metadata_sha256']):
            raise ValueError('Original cache metadata changed since preparation')
        _check_split(self.split, source.get('test_lock_root'))
        specs = _validate_record(source, self.directory)
        changed = self.problem in CHANGED_PROBLEMS
        try:
            self._arrays = _open_arrays(self.directory, {key: spec for key, spec in specs.items()
                if not changed or key not in ('costs', 'winner')})
            if changed:
                label_root = Path(data_root) / (self.problem + self.split)
                self._arrays.update(_open_arrays(label_root, {key: metadata['arrays'][key]
                    for key in ('costs', 'winner')}))
            self.lengths = self._arrays['n']
            self.raw = dict(costs=self._arrays['costs'], winner=self._arrays['winner'],
                pool=self.pool, pool_ids=self.pool_ids)
            if (self.raw['costs'].dtype != np.float64 or self.raw['winner'].dtype != np.int64
                    or self.raw['costs'].shape != (self.size, len(self.pool))
                    or not np.array_equal(self.raw['winner'], self.raw['costs'].argmin(axis=1))):
                raise ValueError('Overlay labels must retain FP64 costs and recomputed int64 winners')
            self._targets = {key: value.contiguous().numpy()
                for key, value in comparison_targets(self.raw['costs']).items()}
        except BaseException:
            self.close()
            raise


def open_cache(cache_root, split, problem, manifest, data_root):
    """The R76 five-argument API; no input copies, re-encoding, or global patches."""
    split = 'val' if str(split).lower() == 'validation' else str(split).lower()
    problem = str(problem).upper()
    if (manifest.get('candidate_overlay') != 'r125'
            or _path(manifest['data_root']) != _path(data_root)):
        raise ValueError('Cache opening requires this R125 scenario manifest')
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
