"""Train-only, CPU-only interpretable advantage-region diagnosis (not a selector).

Run with ``python -B -m code.V4.plateau_regions run`` in easynco.
Only this module and the regions output directory belong to this diagnostic.
"""

import os
import sys

os.environ['CUDA_VISIBLE_DEVICES'] = ''
for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
              'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[_name] = '4'
sys.dont_write_bytecode = True
if hasattr(os, 'sched_getaffinity'):
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:4])

import argparse
import csv
from functools import lru_cache
import hashlib
import inspect
import json
import pickle
from pathlib import Path
import time

import numpy as np

from code.unified_selector.registry import DATA_ROOT, POOLS, PROBLEMS, problem_to_pool_mask

ROOT = Path(__file__).resolve().parent / 'runs/Plateau_diagnosis_R70/regions'
SEED = 2
PSEUDOCOUNT = 20.0
MAX_GEOMETRY_CUSTOMERS = 128
MIN_HELDOUT_SUPPORT = 30
GLOBAL_FIELDS = ('capacity', 'route_limit', 'speed', 'backhaul_class',
                 'open_route', 'depot_tw_end')
STAT_SUFFIXES = ('mean', 'std', 'q10', 'q50', 'q90', 'max')


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def file_hash(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def directory(problem, split):
    if split in ('TSPLIB', 'CVRPLIB'):
        if (problem, split) not in (('TSP', 'TSPLIB'), ('CVRP', 'CVRPLIB')):
            raise ValueError('LIB problem mismatch')
        return DATA_ROOT / split
    if split not in ('train', 'val', 'test'):
        raise ValueError('Unsupported split')
    return DATA_ROOT / (problem + split)


def load_raw(problem, split):
    source = directory(problem, split)
    with (source / 'raw_label.pkl').open('rb') as stream:
        labels = pickle.load(stream)
    pool = list(POOLS[problem])
    names = [p.stem[len('result_'):] for p in sorted((source / 'results').glob('result_*.txt'))]
    if names and names != pool:
        raise ValueError(f'{problem}/{split}: result filenames differ from registry')
    records = [labels[str(i)] for i in range(len(labels))]
    if any(len(r['cost']) != len(pool) for r in records):
        raise ValueError(f'{problem}/{split}: raw_label length differs from registry')
    costs = np.asarray([r['cost'] for r in records], dtype=np.float64)
    winner = np.asarray([r['ind'] for r in records], dtype=np.int64)
    if not np.isfinite(costs).all() or (costs <= 0).any():
        raise ValueError('Invalid original costs')
    if (winner < 0).any() or (winner >= len(pool)).any():
        raise ValueError('Invalid native winner')
    minimum = costs.min(1)
    return dict(costs=costs, winner=winner, pool=pool,
                pool_ids=np.asarray(problem_to_pool_mask(problem)[0], dtype=np.int64),
                gaps=(costs - minimum[:, None]) / minimum[:, None],
                data_hash=file_hash(source / 'dataset.pkl'),
                label_hash=file_hash(source / 'raw_label.pkl'),
                source=str(source), native_nonminimal_costs=int(np.count_nonzero(
                    costs[np.arange(len(winner)), winner] != minimum)))


def smoke(root):
    rows = []
    for problem in PROBLEMS:
        raw = load_raw(problem, 'train')
        rows.append(dict(problem=problem, n=len(raw['winner']), pool=raw['pool'],
                         dtype=str(raw['costs'].dtype), label_hash=raw['label_hash'],
                         native_nonminimal_costs=raw['native_nonminimal_costs']))
    dump(root / 'implement-stage/loader_smoke.json', dict(status='passed', problems=rows))
    print('Real FP64 loader smoke passed for all 18 problems.', flush=True)


def unpack_instance(problem, instance, split):
    result = dict(removed_embedded_depot=False)
    if problem == 'ATSP':
        matrix = np.asarray(instance, dtype=np.float64).reshape(instance.shape[-2:])
        result.update(matrix=matrix, n=len(matrix), raw_node_count=len(matrix))
        return result
    if problem == 'TSP':
        xy = np.asarray(instance, dtype=np.float64).reshape(-1, 2)
        result.update(xy=xy, n=len(xy), raw_node_count=len(xy))
        return result
    if problem == 'CVRP':
        xy = np.asarray(instance['loc'], dtype=np.float64).reshape(-1, 2)
        depot = np.asarray(instance['depot'], dtype=np.float64).reshape(2)
        demand = np.asarray(instance['demand'], dtype=np.float64).reshape(-1)
        result['raw_node_count'] = len(xy)
        if split == 'CVRPLIB' and demand[0] == 0 and np.array_equal(xy[0], depot):
            xy, demand = xy[1:], demand[1:]
            result['removed_embedded_depot'] = True
        result.update(xy=xy, depot=depot, demand=demand, n=len(xy))
        return result
    xy = np.asarray(instance['node_xy'], dtype=np.float64).reshape(-1, 2)
    result.update(xy=xy, depot=np.asarray(instance['depot_xy'], dtype=np.float64).reshape(2),
                  demand=np.asarray(instance['node_demand'], dtype=np.float64).reshape(-1),
                  n=len(xy), raw_node_count=len(xy))
    for field in ('service_time', 'tw_start', 'tw_end'):
        if field in instance:
            result[field] = np.asarray(instance[field], dtype=np.float64).reshape(-1)
    return result


def customer_subset(xy, limit=MAX_GEOMETRY_CUSTOMERS, attributes=()):
    if len(xy) <= limit:
        return np.arange(len(xy))
    bits = np.ascontiguousarray(xy, dtype=np.float64).view(np.uint64).reshape(-1, 2)
    with np.errstate(over='ignore'):
        h = bits[:, 0] ^ (bits[:, 1] * np.uint64(0x9E3779B97F4A7C15))
        h = (h ^ (h >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        h = (h ^ (h >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        h ^= h >> np.uint64(31)
    # Hash coordinates, not instance/customer IDs; raw attributes break coincident-node ties.
    return np.lexsort((*attributes, xy[:, 1], xy[:, 0], h))[:limit]


def add_stats(result, name, values, cached):
    keys = [name + '_' + suffix for suffix in STAT_SUFFIXES]
    if all(k in cached for k in keys):
        result.update({k: float(cached[k]) for k in keys})
        return
    values = np.asarray(values, dtype=np.float64).ravel()
    if not len(values):
        raise ValueError('Empty pre-solve statistic: ' + name)
    numbers = (values.mean(), values.std(), *np.quantile(values, [.1, .5, .9]), values.max())
    result.update({k: float(v) for k, v in zip(keys, numbers)})


def instance_features(problem, instance, split, cached=None):
    cached = {} if cached is None else cached
    data = unpack_instance(problem, instance, split)
    n = data['n']
    result = dict(customers=float(n), log_customers=float(np.log1p(n)))
    if isinstance(instance, dict):
        for field in GLOBAL_FIELDS:
            present = field in instance
            value = np.asarray(instance[field]) if present else np.asarray(0.)
            if value.size != 1:
                raise ValueError('Non-scalar global input: ' + field)
            result['has_' + field] = float(present)
            result[field] = float(value.item())
    if problem == 'ATSP':
        matrix = data['matrix']
        off = ~np.eye(n, dtype=bool)
        add_stats(result, 'matrix_edge', matrix[off], cached)
        add_stats(result, 'matrix_asymmetry', np.abs(matrix - matrix.T)[off], cached)
        add_stats(result, 'matrix_out_min', np.where(off, matrix, np.inf).min(1), cached)
        add_stats(result, 'matrix_in_min', np.where(off, matrix, np.inf).min(0), cached)
        add_stats(result, 'matrix_out_mean', (matrix * off).sum(1) / (n - 1), cached)
        add_stats(result, 'matrix_in_mean', (matrix * off).sum(0) / (n - 1), cached)
        result['matrix_zero_offdiag_fraction'] = float((matrix[off] == 0).mean())
        sampled = n
    else:
        xy = data['xy']
        center = xy.mean(0)
        centered = xy - center
        result.update(bbox_x=float(np.ptp(xy[:, 0])), bbox_y=float(np.ptp(xy[:, 1])),
                      centroid_x=float(center[0]), centroid_y=float(center[1]))
        eig = np.linalg.eigvalsh(centered.T @ centered / n)
        result['geometry_anisotropy'] = float(eig[-1] / max(eig.sum(), 1e-15))
        result['geometry_rms_radius'] = float(np.sqrt(np.square(centered).sum(1).mean()))
        add_stats(result, 'centroid_distance', np.linalg.norm(centered, axis=1), cached)
        attributes = tuple(data[k] for k in ('demand', 'tw_start', 'tw_end') if k in data)
        subset = customer_subset(xy, attributes=attributes)
        sampled = len(subset)
        off = ~np.eye(sampled, dtype=bool)
        need_distance = 'distance_mean' not in cached or 'nn1_mean' not in cached
        need_tw = 'tw_start' in data and 'directed_slack_mean' not in cached
        if need_distance or need_tw:
            sample_xy = xy[subset]
            distance = np.linalg.norm(sample_xy[:, None] - sample_xy[None], axis=-1)
        if need_distance:
            add_stats(result, 'distance', distance[off], cached)
            nearest = np.sort(np.where(off, distance, np.inf), axis=1)
            add_stats(result, 'nn1', nearest[:, 0], cached)
            add_stats(result, 'nn8_mean', nearest[:, :min(8, sampled - 1)].mean(1), cached)
        else:
            for name in ('distance', 'nn1', 'nn8_mean'):
                add_stats(result, name, [0.], cached)
        if 'depot' in data:
            dep_distance = np.linalg.norm(xy - data['depot'], axis=-1)
            result['centroid_depot_distance'] = float(np.linalg.norm(center - data['depot']))
            add_stats(result, 'depot_distance', dep_distance, cached)
        if 'demand' in data:
            demand = data['demand']
            if len(demand) != n:
                raise ValueError('Demand/customer count mismatch')
            result.update(total_demand=float(demand.sum()),
                          positive_demand_total=float(np.maximum(demand, 0).sum()),
                          negative_demand_abs_total=float(np.maximum(-demand, 0).sum()),
                          negative_demand_fraction=float((demand < 0).mean()),
                          zero_demand_fraction=float((demand == 0).mean()))
            add_stats(result, 'demand', demand, cached)
            add_stats(result, 'demand_abs', np.abs(demand), cached)
            if 'capacity' in instance and float(instance['capacity']) > 0:
                result['total_abs_demand_over_capacity'] = float(np.abs(demand).sum() / float(instance['capacity']))
        if 'tw_start' in data:
            start, end = data['tw_start'], data['tw_end']
            service = data.get('service_time', np.zeros(n))
            result['total_service'] = float(service.sum())
            for name, values in (('service', service), ('tw_start', start),
                                 ('tw_end', end), ('tw_width', end - start)):
                add_stats(result, name, values, cached)
            if need_tw:
                a, b, s = start[subset], end[subset], service[subset]
                speed = float(instance.get('speed', 1.))
                if speed <= 0:
                    raise ValueError('Invalid routing speed')
                overlap = np.minimum(b[:, None], b[None]) - np.maximum(a[:, None], a[None])
                slack = b[None] - (a[:, None] + s[:, None] + distance / speed)
                add_stats(result, 'directed_slack', slack[off], cached)
                result['tw_overlap_fraction'] = float((overlap[off] >= 0).mean())
                result['directed_tw_pressure_fraction'] = float((slack[off] < -1e-5).mean())
            else:
                add_stats(result, 'directed_slack', [0.], cached)
                for name in ('tw_overlap_fraction', 'directed_tw_pressure_fraction'):
                    result[name] = float(cached[name])
            speed = float(instance.get('speed', 1.))
            add_stats(result, 'depot_tw_slack', end - dep_distance / speed, cached)
    if not np.isfinite(list(result.values())).all():
        raise ValueError('Nonfinite raw feature')
    return result, data, sampled


def feature_definition():
    functions = (unpack_instance, customer_subset, add_stats, instance_features)
    definition = ''.join(inspect.getsource(f) for f in functions) + str(MAX_GEOMETRY_CUSTOMERS)
    return hashlib.sha256(definition.encode()).hexdigest()


def load_features(problem, split, raw, root):
    start = time.monotonic()
    path = root / 'feature_cache' / f'{problem}_{split}.npz'
    if path.exists():
        with np.load(path, allow_pickle=False) as saved:
            if str(saved['data_hash']) == raw['data_hash']:
                values = {k: saved[k] for k in saved.files}
                np.testing.assert_array_equal(values['indices'], np.arange(len(raw['winner'])))
                return values, dict(cache=str(path), reused=True, seconds=time.monotonic() - start,
                                    sampled_instances=int((values['geometry_sample_size'] < values['sizes']).sum()),
                                    removed_embedded_depots=int(values['removed_embedded_depot'].sum()))
    definition = feature_definition()
    import torch
    torch.set_num_threads(4)
    with (directory(problem, split) / 'dataset.pkl').open('rb') as stream:
        instances = pickle.load(stream)
    if len(instances) != len(raw['winner']):
        raise ValueError('Input/cost row mismatch')
    old_cache, old_fields, reused_source = None, None, None
    old_path = ROOT.parents[1] / 'R49_existing_data_diagnosis/input_statistics.npz'
    manifest_path = old_path.parent / 'split_manifest.json'
    if problem == 'OVRPTW' and split in ('train', 'val') and old_path.exists() and manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest['sources'][split]['data_hash'] == raw['data_hash']:
            with np.load(old_path, allow_pickle=False) as old:
                old_cache, old_fields = old[split], list(old['fields'])
            if len(old_cache) != len(instances) or not np.isfinite(old_cache).all():
                raise ValueError('Invalid r49 feature cache')
            for i in (0, len(instances) // 2, len(instances) - 1):
                fresh = instance_features(problem, instances[i], split)[0]
                np.testing.assert_allclose([fresh[k] for k in old_fields], old_cache[i], rtol=1e-10, atol=1e-12)
            reused_source = str(old_path)
    records, sizes, raw_counts, sample_counts, removed = [], [], [], [], []
    for i, instance in enumerate(instances):
        cached = {} if old_cache is None else dict(zip(old_fields, old_cache[i]))
        row, data, sampled = instance_features(problem, instance, split, cached)
        records.append(row)
        sizes.append(data['n'])
        raw_counts.append(data['raw_node_count'])
        sample_counts.append(sampled)
        removed.append(data['removed_embedded_depot'])
        if (i + 1) % 2500 == 0:
            print(f'[features] {problem}/{split} {i+1}/{len(instances)} {time.monotonic()-start:.1f}s', flush=True)
    names = sorted(records[0])
    if any(set(row) != set(names) for row in records):
        raise ValueError('Feature fields differ across inputs')
    values = dict(X=np.asarray([[row[k] for k in names] for row in records], dtype=np.float64),
                  names=np.asarray(names), sizes=np.asarray(sizes, np.int64),
                  indices=np.arange(len(instances), dtype=np.int64), split=np.asarray(split),
                  raw_node_counts=np.asarray(raw_counts, np.int64),
                  geometry_sample_size=np.asarray(sample_counts, np.int64),
                  removed_embedded_depot=np.asarray(removed, bool),
                  data_hash=np.asarray(raw['data_hash']), definition=np.asarray(definition))
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **values)
    meta = dict(cache=str(path), reused=False, reused_source=reused_source,
                reused_statistics_per_instance=0 if old_fields is None else len(old_fields),
                seconds=time.monotonic() - start, n=len(instances), feature_count=len(names),
                sampled_instances=int((values['geometry_sample_size'] < values['sizes']).sum()),
                max_geometry_sample_size=int(values['geometry_sample_size'].max()),
                removed_embedded_depots=int(values['removed_embedded_depot'].sum()))
    del instances
    return values, meta


def train_size_cuts(sizes):
    cuts = np.unique(np.quantile(sizes, [.25, .5, .75]))
    return cuts[(cuts > sizes.min()) & (cuts < sizes.max())]


def size_membership(sizes, cuts, minimum, maximum):
    result = np.searchsorted(cuts, sizes, side='left').astype(np.int64)
    result[(sizes < minimum) | (sizes > maximum)] = -1
    return result


def summarize_training(raw, rows, parent=None):
    rows = np.asarray(rows, np.int64)
    k = len(raw['pool'])
    counts = np.bincount(raw['winner'][rows], minlength=k).astype(np.float64)
    total = raw['gaps'][rows].sum(0)
    if parent is None:
        # A tiny symmetric floor makes log-probabilities finite; native counts remain unchanged.
        prob = (counts + 1e-12) / (len(rows) + k * 1e-12)
        gaps = total / len(rows)
    else:
        prob = (counts + PSEUDOCOUNT * parent['prob']) / (len(rows) + PSEUDOCOUNT)
        gaps = (total + PSEUDOCOUNT * parent['mean_gap']) / (len(rows) + PSEUDOCOUNT)
    return dict(n=len(rows), prob=prob, mean_gap=gaps,
                majority=int(prob.argmax()), min_mean_gap=int(gaps.argmin()))


def fit_priors(raw, sizes):
    cuts = train_size_cuts(sizes)
    minimum, maximum = int(sizes.min()), int(sizes.max())
    bins = size_membership(sizes, cuts, minimum, maximum)
    global_value = summarize_training(raw, np.arange(len(sizes)))
    per_bin = {int(b): summarize_training(raw, np.flatnonzero(bins == b), global_value)
               for b in np.unique(bins)}
    per_size = {int(n): summarize_training(raw, np.flatnonzero(sizes == n), per_bin[int(bins[sizes == n][0])])
                for n in np.unique(sizes)}
    return dict(cuts=cuts, minimum=minimum, maximum=maximum, global_value=global_value,
                bins=per_bin, exact_sizes=per_size,
                global_sbs=int(raw['costs'].mean(0).argmin()))


def apply_priors(prior, sizes):
    bins = size_membership(sizes, prior['cuts'], prior['minimum'], prior['maximum'])
    probs, gaps, levels, supports = [], [], [], []
    for n, b in zip(sizes, bins):
        if int(n) in prior['exact_sizes']:
            value, level = prior['exact_sizes'][int(n)], 'exact_size_shrunk_to_bin'
        elif int(b) in prior['bins']:
            value, level = prior['bins'][int(b)], 'unseen_size_bin_shrunk_to_global'
        else:
            value, level = prior['global_value'], 'out_of_train_size_global'
        probs.append(value['prob'])
        gaps.append(value['mean_gap'])
        levels.append(level)
        supports.append(value['n'])
    return dict(prob=np.asarray(probs), mean_gap=np.asarray(gaps), size_bin=bins,
                fallback=np.asarray(levels), train_support=np.asarray(supports, np.int64))


def ks_effect(first, second):
    first, second = np.sort(first), np.sort(second)
    grid = np.unique(np.concatenate((first, second)))
    return float(np.max(np.abs(np.searchsorted(first, grid, side='right') / len(first)
                               - np.searchsorted(second, grid, side='right') / len(second))))


def feature_shift(problem, split, train, values):
    if not np.array_equal(train['names'], values['names']):
        raise ValueError('Train/eval feature schema mismatch')
    reference, X = train['X'], values['X']
    lower, upper = reference.min(0), reference.max(0)
    outside = (X < lower) | (X > upper)
    rows = []
    for j, name in enumerate(train['names']):
        a, b = reference[:, j], X[:, j]
        difference = float(b.mean() - a.mean())
        sd = float(np.sqrt((a.var() + b.var()) / 2))
        smd = difference / sd if sd > 0 else (0. if difference == 0 else None)
        rows.append(dict(problem=problem, split=split, feature=str(name), n_train=len(a), n=len(b),
                         train_mean=float(a.mean()), heldout_mean=float(b.mean()),
                         train_std=float(a.std()), heldout_std=float(b.std()),
                         signed_pooled_standardized_mean_difference=smd,
                         ks_ecdf_distance=ks_effect(a, b), train_min=float(lower[j]),
                         train_max=float(upper[j]), heldout_min=float(b.min()), heldout_max=float(b.max()),
                         fraction_outside_train_range=float(outside[:, j].mean()),
                         train_constant=bool(lower[j] == upper[j]),
                         train_constant_changed=bool(lower[j] == upper[j] and outside[:, j].any())))
    nonsize = ~np.isin(train['names'], ['customers', 'log_customers'])
    size_out = (values['sizes'] < train['sizes'].min()) | (values['sizes'] > train['sizes'].max())
    summary = dict(problem=problem, split=split, n=len(X), features=X.shape[1],
                   train_min_size=int(train['sizes'].min()), train_max_size=int(train['sizes'].max()),
                   min_size=int(values['sizes'].min()), max_size=int(values['sizes'].max()),
                   extrapolated_size_count=int(size_out.sum()), extrapolated_size_fraction=float(size_out.mean()),
                   fraction_any_feature_outside_train=float(outside.any(1).mean()),
                   fraction_any_non_size_feature_outside_train=float(outside[:, nonsize].any(1).mean()),
                   mean_feature_out_of_train_fraction=float(outside.mean()),
                   max_non_size_ks=max(row['ks_ecdf_distance'] for row in rows if row['feature'] not in ('customers', 'log_customers')),
                   max_non_size_abs_smd=max(abs(row['signed_pooled_standardized_mean_difference']) for row in rows
                       if row['feature'] not in ('customers', 'log_customers') and row['signed_pooled_standardized_mean_difference'] is not None))
    return rows, summary, outside, size_out


def self_test(root):
    xy = np.random.default_rng(SEED).random((300, 2))
    order = np.random.default_rng(3).permutation(len(xy))
    first, second = xy[customer_subset(xy)], xy[order][customer_subset(xy[order])]
    np.testing.assert_array_equal(first, second)
    n = 50
    inst = dict(node_xy=xy[:n], depot_xy=np.array([[.5, .5]]), node_demand=np.ones(n) / n,
                capacity=1., service_time=np.ones(n) * .2,
                tw_start=np.linspace(0, 1, n), tw_end=np.linspace(.5, 1.5, n))
    features = instance_features('OVRPTW', inst, 'train')[0]
    from .r48_common import tree_features
    original = tree_features(inst)
    np.testing.assert_allclose([features[k] for k in original], list(original.values()), rtol=1e-12, atol=1e-12)
    a = np.array([10., 20., 30., 40.])
    np.testing.assert_array_equal(size_membership(np.array([9., 10., 25., 40., 41.]),
        train_size_cuts(a), 10, 40), [-1, 0, 1, 3, -1])
    raw = dict(costs=np.array([[1., 2.], [2., 1.], [1., 3.], [3., 1.]]),
               gaps=np.array([[0., 1.], [1., 0.], [0., 2.], [2., 0.]]),
               winner=np.array([0, 1, 0, 1]), pool=['a', 'b'])
    prior = fit_priors(raw, a)
    fitted = apply_priors(prior, np.array([10, 25, 50]))
    assert fitted['fallback'].tolist() == ['exact_size_shrunk_to_bin',
        'unseen_size_bin_shrunk_to_global', 'out_of_train_size_global']
    np.testing.assert_allclose(fitted['prob'].sum(1), 1.)
    assert ks_effect([1., 2.], [3., 4.]) == 1.
    lib = dict(loc=np.array([[[.5, .5], [.2, .3]]]), demand=np.array([[0., .1]]),
               depot=np.array([[[.5, .5]]]))
    assert unpack_instance('CVRP', lib, 'CVRPLIB')['n'] == 1
    dump(root / 'implement-stage/self_test.json', dict(status='passed', tests=[
        'coordinate-derived subset permutation invariance', 'r48 feature equivalence',
        'train quartiles and outside-size membership', 'hierarchical fallback',
        'finite normalized probabilities', 'KS effect', 'CVRPLIB embedded depot']))
    print('Feature/prior self-tests passed.', flush=True)


def extract(root):
    start = time.monotonic()
    completed = []
    jobs = [(p, split) for p in PROBLEMS for split in ('train', 'val', 'test')]
    jobs += [('TSP', 'TSPLIB'), ('CVRP', 'CVRPLIB')]
    for problem, split in jobs:
        raw = load_raw(problem, split)
        values, meta = load_features(problem, split, raw, root)
        completed.append(dict(problem=problem, split=split, **meta))
        dump(root / 'extraction_status.json', dict(status='running', completed=completed,
            total_splits=len(jobs), elapsed_seconds=time.monotonic() - start))
        print(f'[extracted] {problem}/{split}: {len(values["sizes"])} rows; '
              f'{meta["seconds"]:.1f}s; total {time.monotonic()-start:.1f}s', flush=True)
    dump(root / 'extraction_status.json', dict(status='complete', completed=completed,
        total_splits=len(jobs), elapsed_seconds=time.monotonic() - start))
    print('All 18 task train/val/test features and both LIB caches are complete.', flush=True)


@lru_cache(maxsize=10000)
def critical_value(n, family):
    from scipy.stats import t
    return float(t.ppf(1 - .025 / family, n - 1))


def mean_stats(values, prefix, family=1):
    values = np.asarray(values, dtype=np.float64)
    if not len(values):
        return {prefix + k: None for k in ('mean', 'se', 'ci95_low', 'ci95_high', 'sim_low', 'sim_high')}
    mean = float(values.mean())
    se = float(values.std(ddof=1) / np.sqrt(len(values))) if len(values) > 1 else None
    result = dict(mean=mean, se=se)
    for key, multiplier in (('ci95', 1), ('sim', family)):
        half = critical_value(len(values), multiplier) * se if se is not None else None
        result[key + '_low'] = mean - half if half is not None else None
        result[key + '_high'] = mean + half if half is not None else None
    return {prefix + k: v for k, v in result.items()}


def win_stats(hits, prefix):
    n = len(hits)
    if not n:
        return {prefix + k: None for k in ('rate', 'se', 'ci95_low', 'ci95_high')}
    p = float(np.mean(hits))
    z = 1.959963984540054
    scale = 1 + z * z / n
    center = (p + z * z / (2 * n)) / scale
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / scale
    return {prefix + 'rate': p, prefix + 'se': float(np.sqrt(p * (1 - p) / n)),
            prefix + 'ci95_low': float(max(0., center - half)),
            prefix + 'ci95_high': float(min(1., center + half))}


def region_feature_definitions(train):
    names = list(train['names'])
    geometry = 'matrix_asymmetry_mean' if 'matrix_asymmetry_mean' in names else 'distance_mean'
    candidates = dict(geometry_quantile=geometry, demand_quantile='demand_abs_mean',
                      tw_quantile='directed_tw_pressure_fraction')
    result = {}
    for name, feature in candidates.items():
        if feature in names:
            x = train['X'][:, names.index(feature)]
            if name == 'geometry_quantile' or x.min() < x.max():
                result[name] = dict(feature=feature, column=names.index(feature),
                                    threshold=float(np.median(x)), fitted_on='train_inputs_only')
    return result


def tree_rules(model, names):
    result = {}
    def visit(node, clauses):
        feature = model.tree_.feature[node]
        if feature < 0:
            result[int(node)] = ' AND '.join(clauses) if clauses else 'ALL'
        else:
            value = model.tree_.threshold[node]
            field = names[feature]
            visit(model.tree_.children_left[node], clauses + [f'{field} <= {value:.10g}'])
            visit(model.tree_.children_right[node], clauses + [f'{field} > {value:.10g}'])
    visit(0, [])
    return result


def fit_diagnostic_regions(raw, train, prior, root, problem):
    from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
    classifier = DecisionTreeClassifier(max_depth=3, min_samples_leaf=200, random_state=SEED)
    regressor = DecisionTreeRegressor(max_depth=3, min_samples_leaf=200, random_state=SEED)
    classifier.fit(train['X'], raw['winner'])
    regressor.fit(train['X'], raw['gaps'])
    definitions = region_feature_definitions(train)
    fitted = dict(classifier=classifier, regressor=regressor, features=definitions,
                  rules=dict(winner_tree_leaf=tree_rules(classifier, list(train['names'])),
                             gap_tree_leaf=tree_rules(regressor, list(train['names']))))
    public = dict(problem=problem, pool=raw['pool'], pool_ids=raw['pool_ids'].tolist(),
                  train_n=len(train['sizes']), train_size_min=prior['minimum'],
                  train_size_max=prior['maximum'], size_cuts=prior['cuts'].tolist(),
                  size_rule='searchsorted(cuts, size, side=left); out-of-training range -> -1/global fallback',
                  pseudocount=PSEUDOCOUNT,
                  global_sbs=raw['pool'][prior['global_sbs']],
                  predeclared_feature=definitions['geometry_quantile']['feature'],
                  predeclared_feature_threshold=definitions['geometry_quantile']['threshold'],
                  feature_quantiles=definitions,
                  global_prob=prior['global_value']['prob'].tolist(),
                  global_mean_gap=prior['global_value']['mean_gap'].tolist(),
                  bins={str(b): dict(n=v['n'], prob=v['prob'].tolist(), mean_gap=v['mean_gap'].tolist())
                        for b, v in prior['bins'].items()},
                  exact_sizes={str(n): dict(n=v['n'], prob=v['prob'].tolist(), mean_gap=v['mean_gap'].tolist())
                               for n, v in prior['exact_sizes'].items()},
                  data_hash=raw['data_hash'], label_hash=raw['label_hash'], trees={})
    for name, model in (('winner_tree_leaf', classifier), ('gap_tree_leaf', regressor)):
        public['trees'][name] = dict(max_depth=3, min_samples_leaf=200, random_state=SEED,
            feature_names=train['names'].tolist(), children_left=model.tree_.children_left.tolist(),
            children_right=model.tree_.children_right.tolist(), features=model.tree_.feature.tolist(),
            thresholds=model.tree_.threshold.tolist(), train_support=model.tree_.n_node_samples.tolist(),
            values=model.tree_.value.tolist(), rules={str(k): v for k, v in fitted['rules'][name].items()})
    # Persist the complete train-only definition before reading evaluation labels/features.
    dump(root / 'fitted' / (problem + '.json'), public)
    return fitted


def make_memberships(train, values, prior, fitted):
    np.testing.assert_array_equal(train['names'], values['names'])
    b = size_membership(values['sizes'], prior['cuts'], prior['minimum'], prior['maximum'])
    result = dict(indices=values['indices'], sizes=values['sizes'], split=values['split'],
                  all=np.zeros(len(b), np.int64), size_bin=b,
                  winner_tree_leaf=fitted['classifier'].apply(values['X']),
                  gap_tree_leaf=fitted['regressor'].apply(values['X']))
    for name, definition in fitted['features'].items():
        q = (values['X'][:, definition['column']] > definition['threshold']).astype(np.int64)
        result[name] = np.where(b < 0, -1, 2 * b + q)
        if name == 'geometry_quantile':
            result['predeclared_feature_quantile'] = q
    return result


def region_rule(name, region, fitted, prior):
    if name == 'all':
        return 'ALL'
    if name in ('winner_tree_leaf', 'gap_tree_leaf'):
        return fitted['rules'][name][region]
    if region == -1:
        return f'customers outside [{prior["minimum"]}, {prior["maximum"]}]; TRAIN-GLOBAL FALLBACK'
    b = region if name == 'size_bin' else region // 2
    low = prior['minimum'] if b == 0 else float(prior['cuts'][b - 1])
    high = prior['maximum'] if b == len(prior['cuts']) else float(prior['cuts'][b])
    text = f'customers {">=" if b == 0 else ">"} {low:g} AND customers <= {high:g}'
    if name != 'size_bin':
        d = fitted['features'][name]
        text += f' AND {d["feature"]} {"<=" if region % 2 == 0 else ">"} {d["threshold"]:.10g}'
    return text


def decision_metrics(raw, pred, scores, sbs, global_mean_gap):
    n = len(pred)
    chosen = raw['costs'][np.arange(n), pred]
    gaps = raw['gaps'][np.arange(n), pred] * 100
    order = np.argsort(-scores, axis=1, kind='stable')
    result = dict(n=n, top1=float((pred == raw['winner']).mean()), mean_cost=float(chosen.mean()),
                  mean_relative_gap_pct=float(gaps.mean()),
                  **mean_stats(gaps, 'relative_gap_pct_'),
                  **mean_stats((raw['gaps'][np.arange(n), sbs] * 100 - gaps), 'advantage_vs_train_sbs_pp_'),
                  **mean_stats(raw['gaps'][:, global_mean_gap] * 100 - gaps, 'advantage_vs_train_global_mean_gap_pp_'),
                  **mean_stats(raw['gaps'][:, int(raw['gaps'].mean(0).argmin())] * 100 - gaps,
                               'advantage_vs_hindsight_constant_mean_gap_pp_'))
    for k in (2, 3):
        result['top' + str(k)] = float((order[:, :min(k, scores.shape[1])] == raw['winner'][:, None]).any(1).mean())
    return result


def emit_solver_statistics(rows, problem, split, raw, scope, region, indices, train_n, fallback):
    if not len(indices):
        return
    for j, solver in enumerate(raw['pool']):
        rows.append(dict(problem=problem, split=split, scope=scope, region=int(region), n=len(indices),
            train_support=train_n, heldout_support_below30=split != 'train' and len(indices) < MIN_HELDOUT_SUPPORT,
            fallback=fallback, solver=solver, solver_index=j,
            native_win_count=int((raw['winner'][indices] == j).sum()),
            native_win_frequency=float((raw['winner'][indices] == j).mean()),
            mean_cost=float(raw['costs'][indices, j].mean()),
            mean_relative_gap_pct=float(raw['gaps'][indices, j].mean() * 100),
            **win_stats(raw['winner'][indices] == j, 'win_'),
            **mean_stats(raw['gaps'][indices, j] * 100, 'gap_pct_')))


def analyze_problem(problem, root, tables):
    raw_train = load_raw(problem, 'train')
    train, meta = load_features(problem, 'train', raw_train, root)
    prior = fit_priors(raw_train, train['sizes'])
    fitted = fit_diagnostic_regions(raw_train, train, prior, root, problem)
    datasets = dict(train=dict(raw=raw_train, features=train, feature_meta=meta))
    splits = ['val', 'test'] + (['TSPLIB'] if problem == 'TSP' else ['CVRPLIB'] if problem == 'CVRP' else [])
    for split in splits:
        raw = load_raw(problem, split)
        values, meta = load_features(problem, split, raw, root)
        datasets[split] = dict(raw=raw, features=values, feature_meta=meta)
    names = ['all', 'size_bin', *fitted['features'], 'winner_tree_leaf', 'gap_tree_leaf']
    for split, data in datasets.items():
        data['membership'] = make_memberships(train, data['features'], prior, fitted)
        path = root / 'memberships' / f'{problem}_{split}.npz'
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **data['membership'])
        shifts, coverage, outside, size_out = feature_shift(problem, split, train, data['features'])
        tables['feature_shift'].extend(shifts)
        tables['shift_summary'].append(coverage)
        data['outside'], data['size_out'] = outside, size_out
        tables['data_inventory'].append({**data['feature_meta'], 'problem': problem, 'split': split,
            'n': len(data['raw']['winner']), 'pool': json.dumps(data['raw']['pool']),
            'data_hash': data['raw']['data_hash'], 'label_hash': data['raw']['label_hash'],
            'native_nonminimal_costs': data['raw']['native_nonminimal_costs']})
    global_value = prior['global_value']
    group_values = {}
    for name in names:
        group_values[name] = {}
        for region in np.unique(datasets['train']['membership'][name]):
            ix = np.flatnonzero(datasets['train']['membership'][name] == region)
            group_values[name][int(region)] = summarize_training(raw_train, ix,
                None if name == 'all' else global_value)
    counts = np.bincount(raw_train['winner'], minlength=len(raw_train['pool']))
    top3 = np.argsort(-counts, kind='stable')[:3].tolist()
    ref_a, ref_b = top3[:2]
    pairs = [(a, b) for a in range(len(raw_train['pool'])) for b in range(a + 1, len(raw_train['pool']))]
    train_pair_gaps = np.asarray([(raw_train['gaps'][:, b] - raw_train['gaps'][:, a]).mean() for a, b in pairs])
    large_cut = float(np.quantile(np.abs(train_pair_gaps), .75))
    reference_family = len(PROBLEMS) * 6 * 8 * 2
    pair_family = sum(len(POOLS[p]) * (len(POOLS[p]) - 1) // 2 for p in PROBLEMS) * 6 * 8 * 2
    regional_scores = {split: {} for split in datasets}
    for name in names:
        all_regions = sorted(set(int(r) for data in datasets.values() for r in data['membership'][name]))
        for split, data in datasets.items():
            n, k = data['raw']['costs'].shape
            regional_scores[split][name] = dict(prob=np.empty((n, k)), mean_gap=np.empty((n, k)))
        for region in all_regions:
            value = group_values[name].get(region, global_value)
            support = {s: int((d['membership'][name] == region).sum()) for s, d in datasets.items()}
            train_n = support['train']
            fallback = 'global_no_train_region' if train_n == 0 else 'train_global_smoothed' if name != 'all' else 'problem_global'
            majority, mean_gap = value['majority'], value['min_mean_gap']
            train_ix = np.flatnonzero(datasets['train']['membership'][name] == region)
            transfer = dict(problem=problem, scope=name, region=region,
                            rule=region_rule(name, region, fitted, prior),
                            train_support=train_n, val_support=support['val'], test_support=support['test'],
                            ref_a=raw_train['pool'][ref_a], ref_b=raw_train['pool'][ref_b],
                            label_free_definition=name not in ('winner_tree_leaf', 'gap_tree_leaf'))
            for split, data in datasets.items():
                raw = data['raw']
                ix = np.flatnonzero(data['membership'][name] == region)
                n = len(ix)
                regional_scores[split][name]['prob'][ix] = value['prob']
                regional_scores[split][name]['mean_gap'][ix] = value['mean_gap']
                preferred = majority if name == 'winner_tree_leaf' else mean_gap
                advantage = (raw['gaps'][ix, ref_b] - raw['gaps'][ix, ref_a]) * 100
                ref_stats = mean_stats(advantage, 'ref_advantage_pp_', reference_family)
                row = dict(problem=problem, split=split, scope=name, region=region,
                    rule=transfer['rule'], n=n, support_train=train_n, support_val=support['val'],
                    support_test=support['test'], support_LIB=support.get('TSPLIB', support.get('CVRPLIB', 0)),
                    heldout_support_below30=split != 'train' and n < MIN_HELDOUT_SUPPORT,
                    fallback=fallback, train_estimation_support=value['n'],
                    majority_solver=raw['pool'][majority], majority_solver_index=majority,
                    mean_gap_solver=raw['pool'][mean_gap], mean_gap_solver_index=mean_gap,
                    preferred_solver=raw['pool'][preferred], action_disagreement=majority != mean_gap,
                    majority_top1=float((raw['winner'][ix] == majority).mean()) if n else None,
                    mean_gap_top1=float((raw['winner'][ix] == mean_gap).mean()) if n else None,
                    majority_mean_relative_gap_pct=float(raw['gaps'][ix, majority].mean() * 100) if n else None,
                    mean_gap_mean_relative_gap_pct=float(raw['gaps'][ix, mean_gap].mean() * 100) if n else None,
                    majority_mean_cost=float(raw['costs'][ix, majority].mean()) if n else None,
                    mean_gap_mean_cost=float(raw['costs'][ix, mean_gap].mean()) if n else None,
                    ref_a=raw['pool'][ref_a], ref_b=raw['pool'][ref_b],
                    ref_advantage_sign=int(np.sign(advantage.mean())) if n else None,
                    ci_family_size=reference_family,
                    training_interval_adaptive=name in ('winner_tree_leaf', 'gap_tree_leaf') and split == 'train',
                    **win_stats(raw['winner'][ix] == preferred, 'preferred_win_'), **ref_stats,
                    **mean_stats((raw['gaps'][ix, majority] - raw['gaps'][ix, mean_gap]) * 100,
                                 'mean_gap_advantage_vs_majority_pp_'))
                tables['regions'].append(row)
                if split in ('train', 'val', 'test'):
                    for key in ('mean', 'ci95_low', 'ci95_high', 'sim_low', 'sim_high', 'se'):
                        transfer[split + '_' + key] = ref_stats['ref_advantage_pp_' + key]
                emit_solver_statistics(tables['solver_statistics'], problem, split, raw, name, region, ix, train_n, fallback)
                if n:
                    probability = value['prob']
                    # An independent draw from the training region distribution has this exact expectation.
                    expected_hit = probability[raw['winner'][ix]]
                    expected_gap = raw['gaps'][ix] @ probability * 100
                    random_variance = ((raw['gaps'][ix] * 100 - expected_gap[:, None]) ** 2 @ probability).sum() / n ** 2
                    tables['independence_baseline'].append(dict(problem=problem, split=split, scope=name,
                        region=region, n=n, train_support=train_n, fallback=fallback,
                        expected_native_top1=float(expected_hit.mean()),
                        expected_relative_gap_pct=float(expected_gap.mean()),
                        random_draw_mean_gap_sd_pct=float(np.sqrt(max(0., random_variance))),
                        heldout_support_below30=split != 'train' and n < MIN_HELDOUT_SUPPORT,
                        interpretation='analytic independent solver draw conditional on identical region; not a permutation p-value'))
                for a, b in pairs:
                    if not n:
                        continue
                    diff = raw['costs'][ix, b] - raw['costs'][ix, a]
                    stats = mean_stats((raw['gaps'][ix, b] - raw['gaps'][ix, a]) * 100, 'advantage_a_pp_', pair_family)
                    training_diff = raw_train['costs'][train_ix, b] - raw_train['costs'][train_ix, a]
                    strict = diff != 0
                    train_strict = training_diff != 0
                    pa = float((training_diff[train_strict] > 0).mean()) if train_strict.any() else None
                    qa = float((diff[strict] > 0).mean()) if strict.any() else None
                    tables['pair_statistics'].append(dict(problem=problem, split=split, scope=name, region=region,
                        n=n, train_support=train_n, heldout_support_below30=split != 'train' and n < 30,
                        fallback=fallback, solver_a=raw['pool'][a], solver_b=raw['pool'][b],
                        a_strict_wins=int((diff > 0).sum()), b_strict_wins=int((diff < 0).sum()), exact_ties=int((diff == 0).sum()),
                        a_strict_win_rate=qa, train_a_strict_win_rate=pa,
                        independent_train_heldout_pair_sign_agreement=(pa * qa + (1 - pa) * (1 - qa)) if pa is not None and qa is not None else None,
                        both_train_top3_winners=a in top3 and b in top3,
                        large_train_expected_gap=abs(float((raw_train['gaps'][:, b] - raw_train['gaps'][:, a]).mean())) >= large_cut,
                        ci_family_size=pair_family, **stats))
            sign = int(np.sign(transfer['train_mean'])) if transfer['train_mean'] is not None else 0
            enough = train_n >= 200 and support['val'] >= 30 and support['test'] >= 30
            same = enough and sign != 0 and all(np.sign(transfer[s + '_mean']) == sign for s in ('val', 'test'))
            simultaneous = same and all(transfer[s + ('_sim_low' if sign > 0 else '_sim_high')] * sign > 0 for s in ('val', 'test'))
            transfer.update(sufficient_support=enough, consistent_train_val_test_sign=bool(same),
                            simultaneous_heldout_sign_supported=bool(simultaneous))
            tables['region_transfer'].append(transfer)
    for split, data in datasets.items():
        raw, features = data['raw'], data['features']
        n = len(raw['winner'])
        posterior = apply_priors(prior, features['sizes'])
        policies = dict(majority=posterior['prob'], mean_gap=-posterior['mean_gap'],
            global_sbs=np.tile(-raw_train['costs'].mean(0), (n, 1)),
            global_majority=np.tile(global_value['prob'], (n, 1)),
            global_mean_gap=np.tile(-global_value['mean_gap'], (n, 1)))
        for name in names:
            if name == 'all':
                continue
            scores = regional_scores[split][name]
            policies[name + '_majority'] = scores['prob']
            policies[name + '_mean_gap'] = -scores['mean_gap']
            pm, pg = scores['prob'].argmax(1), scores['mean_gap'].argmin(1)
            ix = np.arange(n)
            disagreement = pm != pg
            tables['decision_tradeoffs'].append(dict(problem=problem, split=split, scope=name, n=n,
                disagreement_count=int(disagreement.sum()), disagreement_fraction=float(disagreement.mean()),
                majority_top1=float((pm == raw['winner']).mean()), mean_gap_top1=float((pg == raw['winner']).mean()),
                mean_gap_accuracy_minus_majority_pp=float(((pg == raw['winner']).astype(float) - (pm == raw['winner'])).mean() * 100),
                majority_relative_gap_pct=float(raw['gaps'][ix, pm].mean() * 100),
                mean_gap_relative_gap_pct=float(raw['gaps'][ix, pg].mean() * 100),
                **mean_stats((raw['gaps'][ix, pm] - raw['gaps'][ix, pg]) * 100, 'mean_gap_advantage_pp_')))
        for j, solver in enumerate(raw['pool']):
            scores = np.zeros_like(raw['costs'])
            scores[:, j] = 1
            policies['fixed:' + solver] = scores
        hindsight = int(raw['gaps'].mean(0).argmin())
        hindsight_scores = np.tile(-raw['gaps'].mean(0), (n, 1))
        for name, scores in [*policies.items(), ('hindsight_best_constant_relative_gap', hindsight_scores)]:
            pred = scores.argmax(1)
            tables['baselines'].append(dict(problem=problem, split=split, policy=name,
                train_fitted=name != 'hindsight_best_constant_relative_gap',
                hindsight_solver=raw['pool'][hindsight] if name == 'hindsight_best_constant_relative_gap' else '',
                **decision_metrics(raw, pred, scores, prior['global_sbs'], global_value['min_mean_gap'])))
        for size in np.unique(features['sizes']):
            ix = np.flatnonzero(features['sizes'] == size)
            original_support = int((train['sizes'] == size).sum())
            emit_solver_statistics(tables['solver_statistics'], problem, split, raw, 'exact_size', int(size), ix,
                original_support, str(posterior['fallback'][ix[0]]))
            tables['size_decisions'].append(dict(problem=problem, split=split, size=int(size), n=len(ix),
                train_exact_size_support=original_support, train_estimation_support=int(posterior['train_support'][ix[0]]),
                size_bin=int(posterior['size_bin'][ix[0]]), fallback=str(posterior['fallback'][ix[0]]),
                majority_solver=raw['pool'][int(posterior['prob'][ix[0]].argmax())],
                mean_gap_solver=raw['pool'][int(posterior['mean_gap'][ix[0]].argmin())],
                heldout_support_below30=split != 'train' and len(ix) < 30))
        if split != 'train':
            folder = 'val_predictions' if split == 'val' else 'test_predictions' if split == 'test' else 'LIB_predictions'
            path = root / folder / (problem + '.npz')
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(path, indices=features['indices'], truth=raw['winner'], winner=raw['winner'],
                costs=raw['costs'], pool=np.asarray(raw['pool']), pool_ids=raw['pool_ids'],
                sizes=features['sizes'], true_size=features['sizes'], raw_node_counts=features['raw_node_counts'],
                split=np.asarray(split), pred_majority=posterior['prob'].argmax(1),
                pred_mean_gap=posterior['mean_gap'].argmin(1), pred_global_sbs=np.full(n, prior['global_sbs'], np.int64),
                pred_winner_tree=policies['winner_tree_leaf_majority'].argmax(1),
                pred_gap_tree=policies['gap_tree_leaf_mean_gap'].argmax(1),
                logits_majority=np.log(posterior['prob']), logits_mean_gap=-posterior['mean_gap'],
                train_prob=posterior['prob'], train_mean_gaps=posterior['mean_gap'],
                size_bin=posterior['size_bin'], fallback=posterior['fallback'], train_support=posterior['train_support'],
                size_extrapolated=data['size_out'], any_feature_outside_train=data['outside'].any(1),
                winner_tree_leaf=data['membership']['winner_tree_leaf'],
                gap_tree_leaf=data['membership']['gap_tree_leaf'])
    print(f'[regions] {problem}: priors, rules, same-region actions, all {len(pairs)} native pairs, '
          f'and {len(datasets)-1} heldout splits complete.', flush=True)


def methodology():
    return dict(purpose='Scientific train-to-heldout solver preference diagnosis; not competitive new selector training',
        problems=PROBLEMS, splits=['train', 'val', 'test', 'TSPLIB(TSP)', 'CVRPLIB(CVRP)'],
        hardware=dict(gpu=False, threads=4, affinity=sorted(os.sched_getaffinity(0))),
        labels='Original raw_label FP64 costs, unchanged native local winner; names/order from current registry and result filenames, not README',
        gap='(solver cost - original full native pool minimum) / minimum; CSV gap/advantage numbers in percent or percentage points',
        tie_break='first registry position, without modifying native winner',
        priors=dict(size='unique interior train quartiles', pseudocount=PSEUDOCOUNT,
            hierarchy=['exact training size shrunk to bin', 'bin shrunk to global', 'out-of-train-size global fallback'],
            global_sbs='training minimum mean absolute cost, distinct from training minimum mean relative gap'),
        trees=dict(winner='DecisionTreeClassifier gini', cost='DecisionTreeRegressor multioutput relative gap squared_error',
                   max_depth=3, min_samples_leaf=200, random_state=SEED, tuning=False, eval_threshold_selection=False),
        regions='Same frozen regions have BOTH majority and minmeanrelativegap actions; extra label-free train-median feature x size-bin regions',
        feature_only_regions=dict(geometry='distance_mean, or ATSP matrix_asymmetry_mean',
            demand='demand_abs_mean when varying in train', TW='directed_tw_pressure_fraction when varying in train',
            threshold='global training input median; no label threshold selection'),
        features=dict(source='raw pre-solve coordinates/matrix/demand/service/TW/explicit global fields only',
            index_distribution_inference=False, label_features=False,
            absent_globals='presence flag and zero; no inferred backhaul class/depot deadline',
            geometry=dict(max_customers=MAX_GEOMETRY_CUSTOMERS,
                sampling='smallest deterministic uint64 coordinate hashes with raw-attribute tie breaks; no instance IDs',
                all_customers_when_small=True, demand_tw_scalars='all actual customers',
                caution='nn1/nn8 are subset nearest distances for sampled instances, not estimates of exact full-set nearest neighbors'),
            reused_cache='OVRPTW r49 75-field input statistics if matching existing source provenance; all others computed/cached here',
            CVRPLIB='Remove first depot from loc only if zero demand and exactly matching depot; keep raw_node_counts; costs/winners unchanged'),
        uncertainty=dict(empirical_se='sample SD/sqrt(N)', mean_ci='two-sided Student-t 95%; Wilson native win-rate CI',
            support_min_heldout=30, unsupported='flagged, never merged',
            tree_train_ci='adaptive and descriptive, not honest estimation; feature-only regions avoid target-selected thresholds',
            reference_pair='top two native winner frequencies in TRAIN; registry tie break',
            reference_family_size=len(PROBLEMS) * 6 * 8 * 2,
            pair_family_size=sum(len(POOLS[p]) * (len(POOLS[p]) - 1) // 2 for p in PROBLEMS) * 6 * 8 * 2,
            simultaneous='Bonferroni Student-t intervals across conservative maximum val/test region families; descriptive iid-instance assumption',
            multiplicity_caution='No p-value search; nominal mean CIs also retained. Multiple routing variants may share base instances, so macro uncertainty is not computed as independent instances.'),
        pairs='All unordered pairs in each actual native pool, not only ReLD; train-top3 and large train expected-gap pairs flagged, never prune coverage',
        independence='Analytic expected native accuracy and relative gap of independent draws from train-smoothed same-region distribution; no duplicated model/error/permutation statistics',
        hindsight='Heldout best constant mean relative gap is an evaluation-label diagnostic lower reference among constants ONLY, not a fitted deployable policy or lower bound for region policies',
        feature_shift='Signed mean difference / pooled population SD; empirical two-sample ECDF KS distance; per-feature and per-instance train min/max exceedance. No KS p-values.',
        arrays=dict(feature_cache='feature_cache/{problem}_{split}.npz: X FP64[N,F], names, sizes, indices, split, raw_node_counts, geometry_sample_size',
            memberships='memberships/{problem}_{split}.npz: indices,sizes,split,size_bin,winner_tree_leaf,gap_tree_leaf,predeclared_feature_quantile,geometry_quantile plus demand/TW regions when available',
            baselines='{val_predictions,test_predictions,LIB_predictions}/{problem}.npz: indices,truth,winner,costs,pool,pool_ids,sizes,pred_majority,pred_mean_gap,pred_global_sbs,logits_majority,logits_mean_gap,train_prob,train_mean_gaps,range/fallback/leaf metadata',
            truth='native local registry winner, identical to winner',
            logits_majority='log smoothed train probabilities; argmax unchanged', logits_mean_gap='negative smoothed mean relative gaps, fractional units'),
        limits=['Historical solver costs/contracts are taken as provided, not recertified.',
                'No neural selector training, GPU, tuning, solver reruns, commits, or R70 residual contrast here.',
                'Failure of these shallow fixed diagnostics does not demonstrate feature insufficiency or conditional independence.',
                'LIB is an out-of-distribution supplemental holdout, never merged into the 18-task macro.',
                'Mean advantage sign alone is not certainty; support and empirical intervals are reported.'])


def write_findings(root, tables):
    base = {(r['problem'], r['split'], r['policy']): r for r in tables['baselines']}
    shift = {(r['problem'], r['split']): r for r in tables['shift_summary']}
    lines = ['# Train-Only Advantage Regions: All 18 Tasks', '',
        'Fixed diagnostic, not a new competitive selector. All thresholds, smoothing, actions and trees are fitted on training only. '
        'Original FP64 costs and native winners use current registry pools; README solver lists are not used. '
        'Train/val/test total 216,000 rows; LIB is supplemental. No GPU, neural training, solver reruns, tuning or commits.', '',
        '## Conditional priors and fixed constants', '',
        'Relative gap is 100*(selected cost - instance native-pool minimum)/minimum. '
        'SBS minimizes TRAIN mean absolute cost; hindsight constant minimizes HELDOUT mean relative gap and is a lower reference among constants only. '
        'Size prior uses exact training sizes shrunk to train-quartile bins, then problem-global (pseudocount 20 at each level). '
        'An unseen in-range size uses its bin; an out-of-train size uses the global prior.', '',
        '| Task | Majority val/test Top1 % | Mean-gap val/test Top1 % | Mean-gap val/test gap % | Train SBS val/test gap % | Hindsight constant val/test gap % |',
        '|---|---:|---:|---:|---:|---:|']
    for p in PROBLEMS:
        a = [base[p, s, 'majority'] for s in ('val', 'test')]
        b = [base[p, s, 'mean_gap'] for s in ('val', 'test')]
        c = [base[p, s, 'global_sbs'] for s in ('val', 'test')]
        h = [base[p, s, 'hindsight_best_constant_relative_gap'] for s in ('val', 'test')]
        pair = lambda rows, key, scale=1.: '/'.join(f'{r[key]*scale:.3f}' for r in rows)
        lines.append(f'| {p} | {pair(a,"top1",100)} | {pair(b,"top1",100)} | '
                     f'{pair(b,"mean_relative_gap_pct")} | {pair(c,"mean_relative_gap_pct")} | {pair(h,"mean_relative_gap_pct")} |')
    macro = {}
    for split in ('val', 'test'):
        macro[split] = {}
        for policy in ('majority', 'mean_gap', 'global_sbs', 'global_mean_gap',
                       'size_bin_majority', 'size_bin_mean_gap', 'winner_tree_leaf_majority', 'gap_tree_leaf_mean_gap',
                       'hindsight_best_constant_relative_gap'):
            group = [base[p, split, policy] for p in PROBLEMS]
            macro[split][policy] = {k: float(np.mean([r[k] for r in group]))
                                  for k in ('top1', 'mean_relative_gap_pct')}
    lines += ['', 'Equal-task macro (18 tasks, no LIB):', '',
        '| Policy | Val Top1 % | Test Top1 % | Val gap % | Test gap % |',
        '|---|---:|---:|---:|---:|']
    for policy in macro['val']:
        a, b = macro['val'][policy], macro['test'][policy]
        lines.append(f'| {policy} | {a["top1"]*100:.3f} | {b["top1"]*100:.3f} | '
                     f'{a["mean_relative_gap_pct"]:.4f} | {b["mean_relative_gap_pct"]:.4f} |')
    lines += ['', '## Identical-Region Accuracy/Cost Tradeoff', '',
        'Every frozen size/feature/tree region has BOTH train-majority and train-minmeanrelativegap decisions. '
        'The following uses the SAME label-free four size regions for both actions; their disagreement is the fraction of instances receiving different decisions. '
        'Positive gap advantage favors the mean-gap action. Full rule-level supports, uncertainty and tradeoffs are in regions.csv and decision_tradeoffs.csv.', '',
        '| Task | Val/test disagreement % | Mean-gap minus majority Top1 pp (val/test) | Mean-gap advantage pp (val/test) |',
        '|---|---:|---:|---:|']
    trade = {(r['problem'], r['split'], r['scope']): r for r in tables['decision_tradeoffs']}
    for p in PROBLEMS:
        a, b = trade[p, 'val', 'size_bin'], trade[p, 'test', 'size_bin']
        lines.append(f'| {p} | {a["disagreement_fraction"]*100:.1f}/{b["disagreement_fraction"]*100:.1f} | '
            f'{a["mean_gap_accuracy_minus_majority_pp"]:+.2f}/{b["mean_gap_accuracy_minus_majority_pp"]:+.2f} | '
            f'{a["mean_gap_advantage_pp_mean"]:+.4f}/{b["mean_gap_advantage_pp_mean"]:+.4f} |')
    transferable = [r for r in tables['region_transfer'] if r['simultaneous_heldout_sign_supported']]
    label_free = [r for r in transferable if r['label_free_definition']]
    eligible = [r for r in tables['region_transfer'] if r['sufficient_support']]
    lines += ['', '## Supported Region Trends', '',
        f'{len(eligible)} regions have at least 200 training and 30 validation/test instances; '
        f'{len(transferable)} have the same TRAIN/val/test primary-pair advantage sign and simultaneous heldout CIs excluding zero, '
        f'including {len(label_free)} label-free input-quantile regions. Overlapping regions are NOT independent discoveries. '
        'Primary pair is fixed from the top two TRAIN winner frequencies; positive advantage favors A. '
        'Intervals on target-derived TRAIN tree leaves are adaptive/descriptive, not independent estimation.', '',
        '| Task / region | Pair A vs B | Train/val/test support | Train/val/test advantage pp | Val simultaneous CI | Test simultaneous CI |',
        '|---|---|---:|---:|---:|---:|']
    for row in label_free:
        lines.append(f'| {row["problem"]} / {row["scope"]}:{row["region"]} | {row["ref_a"]} vs {row["ref_b"]} | '
            f'{row["train_support"]}/{row["val_support"]}/{row["test_support"]} | '
            f'{row["train_mean"]:+.3f}/{row["val_mean"]:+.3f}/{row["test_mean"]:+.3f} | '
            f'[{row["val_sim_low"]:+.3f}, {row["val_sim_high"]:+.3f}] | '
            f'[{row["test_sim_low"]:+.3f}, {row["test_sim_high"]:+.3f}] |')
    if not label_free:
        lines.append('| No label-free region clears this conservative criterion | | | | | |')
    lines += ['', 'All native method pairs, including rarely winning methods, are retained in pair_statistics.csv. '
        'Top3 training-winner pairs and large TRAIN expected-gap pairs are flagged rather than selected using heldout data. '
        'independence_baseline.csv gives analytic same-region probability-matching expectations, not a permutation p-value or a performance ceiling.', '',
        '## Distribution and LIB Extrapolation', '',
        'KS is ECDF distance, not a p-value. SMD is signed mean change divided by pooled population SD. '
        'Any-feature range exceedance is sensitive to feature count; per-feature effects are available in feature_shift.csv. '
        'These are empirical distribution contrasts, not inferred generation-distribution labels.', '',
        '| Task/split | N | Actual size range | Size extrapolation % | Any non-size feature out of train % | Max non-size abs SMD | Max non-size KS |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for p in PROBLEMS:
        for split in ('val', 'test'):
            r = shift[p, split]
            lines.append(f'| {p}/{split} | {r["n"]} | {r["min_size"]}..{r["max_size"]} | '
                f'{r["extrapolated_size_fraction"]*100:.1f} | {r["fraction_any_non_size_feature_outside_train"]*100:.1f} | '
                f'{r["max_non_size_abs_smd"]:.3f} | {r["max_non_size_ks"]:.3f} |')
    for p, split in (('TSP', 'TSPLIB'), ('CVRP', 'CVRPLIB')):
        r = shift[p, split]
        lines.append(f'| {p}/{split} | {r["n"]} | {r["min_size"]}..{r["max_size"]} | '
            f'{r["extrapolated_size_fraction"]*100:.1f} | {r["fraction_any_non_size_feature_outside_train"]*100:.1f} | '
            f'{r["max_non_size_abs_smd"]:.3f} | {r["max_non_size_ks"]:.3f} |')
        lines += ['', f'{split}: train-fixed majority Top1={base[p,split,"majority"]["top1"]*100:.2f}%, '
            f'gap={base[p,split,"majority"]["mean_relative_gap_pct"]:.4f}%; '
            f'train-fixed mean-gap Top1={base[p,split,"mean_gap"]["top1"]*100:.2f}%, '
            f'gap={base[p,split,"mean_gap"]["mean_relative_gap_pct"]:.4f}%; '
            f'train-SBS gap={base[p,split,"global_sbs"]["mean_relative_gap_pct"]:.4f}%; '
            f'hindsight-constant gap={base[p,split,"hindsight_best_constant_relative_gap"]["mean_relative_gap_pct"]:.4f}%.', '']
    lines += ['CVRPLIB geometry/actual size excludes an explicitly duplicated zero-demand first depot; '
        'raw_node_counts remains available. Train/test candidate costs and native labels are unchanged.', '',
        '## Scope and Limits', '',
        'Small heldout supports (<30) are flagged, never merged. Out-of-train sizes fall back to training global priors; '
        'trees still route extrapolated inputs, explicitly flagged in the arrays. '
        'Geometry distances are bounded to 128 deterministically coordinate-selected customers; '
        'subset nearest-neighbor values are not exact full-instance nearest neighbors when sampled.', '',
        'A null result for these fixed shallow regions does NOT prove feature insufficiency, label randomness, '
        'or conditional independence. The historical solver deployments/contracts are not recertified here. '
        'This report makes no architecture or competitive-selector claim. '
        'The main agent separately owns frozen-model/error/permutation and R70 score-conditioned residual statistics.', '',
        '## Files', '',
        '- methodology.json: fixed definitions, confidence/support caveats, schemas and ownership.',
        '- solver_statistics.csv / size_decisions.csv: all tasks, exact actual sizes, training bins and solver frequencies/gaps.',
        '- baselines.csv: train-fixed priors, every fixed solver, and separately labeled hindsight-constant references.',
        '- regions.csv / decision_tradeoffs.csv / region_transfer.csv: same-region decisions, supports, rules and replicated pair signs.',
        '- pair_statistics.csv / independence_baseline.csv: full native-pool pair coverage and analytic same-region reference.',
        '- feature_shift.csv / shift_summary.csv: train-to-val/test/LIB SMD, KS and train-range effects.',
        '- feature_cache/ / memberships/ / fitted/: per-instance raw statistics, frozen memberships and full train-only rules.',
        '- val_predictions/ / test_predictions/ / LIB_predictions/: original-order aggregation arrays with registry pools.']
    (root / 'findings.md').write_text('\n'.join(lines) + '\n')
    dump(root / 'summary.json', dict(macro=macro, supported_reference_regions=len(transferable),
        label_free_supported_reference_regions=len(label_free), eligible_regions=len(eligible),
        lib={split: shift[p, split] for p, split in (('TSP', 'TSPLIB'), ('CVRP', 'CVRPLIB'))}))


def run(root, wait_extraction=False):
    if wait_extraction:
        while not (root / 'extraction_status.json').exists() or json.loads((root / 'extraction_status.json').read_text())['status'] != 'complete':
            time.sleep(2)
    start = time.monotonic()
    root.mkdir(parents=True, exist_ok=True)
    dump(root / 'methodology.json', methodology())
    table_names = ('baselines', 'regions', 'region_transfer', 'solver_statistics', 'size_decisions',
                   'pair_statistics', 'decision_tradeoffs', 'independence_baseline',
                   'feature_shift', 'shift_summary', 'data_inventory')
    tables = {name: [] for name in table_names}
    completed = []
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=4):
        for problem in PROBLEMS:
            analyze_problem(problem, root, tables)
            completed.append(problem)
            dump(root / 'analysis_status.json', dict(status='running', completed=completed,
                elapsed_seconds=time.monotonic() - start))
    for name, rows in tables.items():
        write_csv(root / (name + '.csv'), rows)
    write_findings(root, tables)
    dump(root / 'analysis_status.json', dict(status='complete', completed=completed,
        elapsed_seconds=time.monotonic() - start, table_rows={k: len(v) for k, v in tables.items()},
        script_sha256=file_hash(Path(__file__))))
    print(f'Complete: all 18 tasks and both LIB splits; {time.monotonic()-start:.1f}s analysis. {root}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['smoke', 'self-test', 'extract', 'run'])
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--wait-extraction', action='store_true')
    args = parser.parse_args()
    if args.command == 'smoke':
        smoke(args.root)
    elif args.command == 'self-test':
        self_test(args.root)
    elif args.command == 'extract':
        extract(args.root)
    else:
        run(args.root, args.wait_extraction)


if __name__ == '__main__':
    main()
