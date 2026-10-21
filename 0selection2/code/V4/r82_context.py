"""TRAIN-fitted context groups, native regret labels, and small pair coverage."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from code.unified_selector.registry import PROBLEMS
from .multitask_probe import task_seed
from .r76_experiment import DATA_ROOT, file_hash, read_json, utc, write_json
from .r77_data import DEFAULT_ROOT as R77_ROOT, FULL_ROWS
from .r77_experiment import ShuffleStream
from .r80_experiment import prepared_config as r77_sources, save_npz


DEFAULT_ROOT = Path(__file__).resolve().parent / 'runs/R82_context_matched_ranking'
GROUP = 'B_context_matched_rank_seed2'
ALIAS = 'Requested_R80B_context_matched_residual_ranking'
TARGET_RULE = 'original FP64 relative regret; unequal pairs inside identical TRAIN-fitted context only'
FIELDS = ('scale', 'distribution', 'capacity_type')


def array_hash(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def relative_regret(costs):
    if costs.dtype != np.float64 or costs.ndim != 2 or not all(costs.shape):
        raise ValueError('Require original FP64 costs [N,K], not upcast FP32')
    if not np.isfinite(costs).all() or np.any(costs <= 0):
        raise ValueError('All legal costs must be finite and positive')
    minimum = costs.min(1, keepdims=True)
    return (costs - minimum) / minimum


def metadata_values(metadata, problem):
    if any(key not in metadata for key in FIELDS):
        raise ValueError('Context requires recorded scale/distribution/capacity_type, never guessed IDs')
    size = np.asarray(metadata['scale'])
    distribution = np.asarray(metadata['distribution']).astype(str)
    capacity = np.asarray(metadata['capacity_type']).astype(str)
    if (size.ndim != 1 or distribution.shape != size.shape or capacity.shape != size.shape
            or not len(size) or not np.isfinite(size).all() or (size <= 0).any()
            or (size != np.floor(size)).any() or (distribution == '').any() or (capacity == '').any()):
        raise ValueError('Context metadata must have complete aligned real sizes and named conditions')
    if problem in ('TSP', 'ATSP'):
        capacity = np.full(len(size), 'not_applicable')
    return size.astype(np.int64), distribution, capacity


def _keys(size, distribution, capacity, cuts):
    bins = np.searchsorted(np.asarray(cuts, dtype=np.float64), np.log(size), side='right')
    return [(int(b), str(d), str(c)) for b, d, c in zip(bins, distribution, capacity)]


def fit_context(metadata, problem, bins=10):
    if bins < 1 or int(bins) != bins:
        raise ValueError('Quantile bin count must be a positive integer')
    size, distribution, capacity = metadata_values(metadata, problem)
    raw = np.quantile(np.log(size), np.linspace(0., 1., int(bins)+1)[1:-1])
    # Never split identical physical sizes just to obtain exactly equal bin counts.
    cuts = np.unique(raw).tolist()
    keys = _keys(size, distribution, capacity, cuts)
    axis = sorted(set(keys))
    lookup = {key: i for i, key in enumerate(axis)}
    groups = np.asarray([lookup[key] for key in keys], dtype=np.int64)
    records = []
    for group_id, (bin_id, dist, cap) in enumerate(axis):
        member = groups == group_id
        records.append(dict(group_id=group_id, size_bin=bin_id, distribution=dist, capacity_type=cap,
            train_rows=int(member.sum()), min_size=int(size[member].min()), max_size=int(size[member].max())))
    spec = dict(problem=problem, fitted_on='TRAIN only', requested_bins=int(bins),
        log_size_cutpoints=cuts, actual_size_bins=len({key[0] for key in axis}),
        quantile_interval_count=len(cuts)+1, group_count=len(axis), groups=records,
        fields=['problem', 'log_size_quantile_bin', 'distribution'] + ([] if problem in ('TSP','ATSP') else ['capacity_type']),
        group_rule='identical named distribution/capacity and quantile interval; never relax conditions',
        quantile_rule='np.quantile linear on TRAIN log(real_size); duplicate cutpoints merged; searchsorted side=right',
        unknown_heldout_group=-1, index_inference=False)
    return groups, spec


def apply_context(metadata, spec):
    size, distribution, capacity = metadata_values(metadata, spec['problem'])
    lookup = {(r['size_bin'], r['distribution'], r['capacity_type']): r['group_id'] for r in spec['groups']}
    return np.asarray([lookup.get(key, -1) for key in _keys(size, distribution, capacity,
        spec['log_size_cutpoints'])], dtype=np.int64)


def batch_coverage(groups, regrets):
    covered = np.zeros(len(groups), dtype=bool)
    methods = np.zeros(regrets.shape[1], dtype=bool)
    active_groups = pairs = matched_pairs = 0
    for group in np.unique(groups):
        rows = np.flatnonzero(groups == group)
        n = len(rows)
        if n < 2:
            continue
        total = n*(n-1)//2
        matched_pairs += total
        for method in range(regrets.shape[1]):
            _, counts = np.unique(regrets[rows, method], return_counts=True)
            unequal = total - int(np.sum(counts*(counts-1)//2))
            if unequal:
                covered[rows] = True
                methods[method] = True
                active_groups += 1
                pairs += unequal
    return dict(row_coverage=float(covered.mean()), effective_methods=int(methods.sum()),
        effective_group_method_cells=active_groups, unequal_method_pairs=pairs,
        matched_instance_pairs=matched_pairs, effective_method_mask=methods)


def coverage_draws(problem, groups, regrets, draws=8):
    stream = ShuffleStream(len(groups), task_seed(2, problem))
    rows, participation = [], np.zeros(regrets.shape[1], dtype=np.int64)
    for draw in range(draws):
        indices = stream.take(1024).numpy()
        record = batch_coverage(groups[indices], regrets[indices])
        participation += record.pop('effective_method_mask')
        rows.append(dict(problem=problem, draw=draw+1, batch_size=1024, **record))
    return rows, participation


def _cache_directory(root, problem):
    candidates = (root / 'train' / problem, root / (problem+'train'))
    return next((path for path in candidates if path.is_dir()), candidates[0])


def prepare_training_context(args):
    destination = args.root / 'prepared/prepared.json'
    if destination.exists():
        saved = read_json(destination)
        if (saved.get('experiment') != 'R82' or saved.get('group') != GROUP
                or saved.get('train_only') is not True or saved['data_root'] != str(args.data_root)
                or saved['r77_root'] != str(args.prepared_root)):
            raise ValueError('Existing preparation belongs to a different R82 protocol')
        for name, digest in saved['artifact_sha256'].items():
            if file_hash(args.root/name) != digest:
                raise ValueError(f'Frozen TRAIN context artifact changed: {name}')
        return saved
    reference, manifest, geometry, baselines = r77_sources(args)
    support, artifacts, coverage, cost_bytes = {}, {}, [], 0
    for problem in PROBLEMS:
        directory = _cache_directory(args.cache_root, problem)
        source = read_json(directory/'metadata.json')
        expected = manifest['splits']['train'][problem]
        if (not source.get('complete') or source.get('problem') != problem or source.get('split') != 'train'
                or source.get('rows') != FULL_ROWS or source['pool'] != expected['pool']
                or source['pool_ids'] != expected['pool_ids']
                or Path(source['data_root']).resolve() != args.data_root.resolve()):
            raise ValueError(f'Incorrect frozen TRAIN cache: {problem}')
        costs = np.load(directory/'costs.npy', allow_pickle=False)
        winner = np.load(directory/'winner.npy', allow_pickle=False)
        if costs.shape != (FULL_ROWS, len(source['pool'])) or not np.array_equal(winner, costs.argmin(1)):
            raise ValueError('Native TRAIN winner/candidate axis changed')
        regrets = relative_regret(costs)
        metadata_path = args.data_root/(problem+'train')/'input_metadata.npz'
        with np.load(metadata_path, allow_pickle=False) as archive:
            metadata = {key: archive[key] for key in FIELDS}
        size, _, _ = metadata_values(metadata, problem)
        n = np.load(directory/'n.npy', allow_pickle=False)
        if len(size) != FULL_ROWS or not np.array_equal(n, size + int(problem not in ('TSP','ATSP'))):
            raise ValueError(f'Recorded scale is not aligned with the real cached node count: {problem}')
        groups, context = fit_context(metadata, problem)
        group_count = context['group_count']
        means = np.zeros((group_count, costs.shape[1]), dtype=np.float64)
        np.add.at(means, groups, regrets)
        means /= np.bincount(groups, minlength=group_count)[:, None]
        min_cost = costs.min(1, keepdims=True)
        minima = costs == min_cost
        positive = minima & (minima.sum(1, keepdims=True) == 1)
        negative = costs > min_cost
        continuous = np.zeros(costs.shape[1], dtype=bool)
        for group in range(group_count):
            continuous |= np.ptp(regrets[groups == group], axis=0) > 0
        methods = [dict(method=name, solver_id=solver_id, slot=slot,
            positive_count=int(positive[:, slot].sum()), negative_count=int(negative[:, slot].sum()),
            eligible=bool(continuous[slot]), continuous_supported=bool(continuous[slot]))
            for slot, (name, solver_id) in enumerate(zip(source['pool'], source['pool_ids']))]
        target = args.root/'support'/(problem+'.npz')
        save_npz(target, dict(groups=groups, regrets=regrets, group_means=means, pool=np.asarray(source['pool']),
            pool_ids=np.asarray(source['pool_ids'], dtype=np.int64), source_indices=np.arange(FULL_ROWS, dtype=np.int64)))
        support[problem] = dict(path=str(target.relative_to(args.root)), rows=FULL_ROWS,
            pool=source['pool'], pool_ids=source['pool_ids'], context=context,
            native_costs_sha256=array_hash(costs), native_winner_sha256=array_hash(winner),
            groups_sha256=array_hash(groups), regrets_sha256=array_hash(regrets), group_means_sha256=array_hash(means),
            metadata_sha256=file_hash(metadata_path), cache_metadata_sha256=file_hash(directory/'metadata.json'),
            eligible_count=int(continuous.sum()), methods=methods, targets_rule=TARGET_RULE,
            centering='TRAIN means saved; same-group differences are exactly the uncentered regret differences')
        artifacts[str(target.relative_to(args.root))] = file_hash(target)
        samples, participation = coverage_draws(problem, groups, regrets)
        coverage.extend(samples)
        for method, times in zip(methods, participation):
            method['coverage_probe_active_batches'] = int(times)
        cost_bytes += costs.nbytes
        print(f'[R82 prepare] {problem}: {group_count} contexts, coverage '
              f'{100*np.mean([row["row_coverage"] for row in samples]):.2f}%', flush=True)
    support_path = args.root/'support/manifest.json'
    write_json(support_path, dict(experiment='R82', train_only=True, problems=support,
        targets_rule=TARGET_RULE, eligible_rule='diagnostic continuous variation only; actual CMR uses active batch methods',
        original_data_modified=False, raw_fp64_upcast_from_fp32=False))
    artifacts[str(support_path.relative_to(args.root))] = file_hash(support_path)
    coverage_path = args.root/'prepared/pair_coverage.json'
    enough = all(row['row_coverage'] >= .90 and row['unequal_method_pairs'] > 0 for row in coverage)
    write_json(coverage_path, dict(experiment='R82', seed=2, batch_size=1024, draws_per_problem=8,
        train_only=True, unchanged_natural_stream=True, feature_queue_used=False, sufficient=enough,
        readiness_rule='all 8 natural batches/problem have >=90% rows in an unequal-regret same-context pair',
        threshold_is_engineering_not_a_performance_claim=True, rows=coverage))
    csv_path = coverage_path.with_suffix('.csv')
    with csv_path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(coverage[0]))
        writer.writeheader()
        writer.writerows(coverage)
    for path in (coverage_path, csv_path):
        artifacts[str(path.relative_to(args.root))] = file_hash(path)
    record = dict(experiment='R82', alias=ALIAS, group=GROUP, schema_version=1, prepared_utc=utc(),
        train_only=True, data_root=str(args.data_root), r77_root=str(args.prepared_root), cache_root=str(args.cache_root),
        r77_prepared_sha256=file_hash(args.prepared_root/'prepared.json'),
        r77_artifact_sha256={key: reference['artifact_sha256'][key] for key in
            ('source_cache_manifest.json','identicalinit_seed2.pt','common_full_train_sbs.json','B_full_seed2/geometry_stats.json')},
        trainable_sha256=reference['trainable_sha256'], model_params=reference['model_params'],
        support_manifest=str(support_path.resolve()), artifact_sha256=artifacts,
        native_train_rows=FULL_ROWS*len(PROBLEMS), native_train_cost_bytes=cost_bytes,
        pair_coverage_sufficient=enough, pair_coverage=str(coverage_path.resolve()),
        training_started=False, test_inputs_read=False, test_costs_read=False, val_used_for_training=False,
        historical_checkpoint_loaded=False, cache_rebuilt=False, geometry_refitted=False, sbs_refitted=False,
        geometry_fitted_on=geometry.get('fitted_on','R77 FULL TRAIN'),
        full_train_sbs={p: baselines[p]['sbs_solver'] for p in PROBLEMS}, seed=2)
    write_json(destination, record)
    if not enough:
        raise RuntimeError('Matched pair coverage insufficient; saved evidence, no feature queue or training started')
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    parser.add_argument('--prepared-root', type=Path, default=R77_ROOT)
    parser.add_argument('--data-root', type=Path, default=DATA_ROOT)
    parser.add_argument('--cache-root', type=Path)
    args = parser.parse_args()
    for name in ('root','prepared_root','data_root','cache_root'):
        if getattr(args, name) is not None:
            setattr(args, name, getattr(args,name).expanduser().resolve())
    print(json.dumps(prepare_training_context(args), indent=2))


if __name__ == '__main__':
    main()
