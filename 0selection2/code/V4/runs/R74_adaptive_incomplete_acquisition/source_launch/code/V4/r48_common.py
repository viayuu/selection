"""R48 two-solver diagnostics: raw labels, input permutations and independent routes."""

import copy
import csv
import json
import pickle
from pathlib import Path

import numpy as np
import torch

from code.unified_selector.registry import DATA_ROOT
from .multitask_probe import dump, file_hash
from .performance_targets import read_raw_costs


ROOT = Path('code/V4/runs/R48_core_diagnosis')
R41 = Path('code/V4/runs/R41_signal_audit')
R47 = Path('code/V4/runs/R47_behavior_memory/B_behavior_memory_seed2/best.pt')
PAIR = ('RELD_MOEL', 'RELD_MTL')
FIELDS = ('node_xy', 'node_demand', 'service_time', 'tw_start', 'tw_end')
VARIANTS = ('original_0', 'original_1', 'permutation_0', 'permutation_1',
            'permutation_2', 'permutation_3')
COST_ATOL, COST_RTOL, FEASIBILITY_TOL = 2e-5, 2e-6, 1e-5


def write_csv(path, rows):
    rows = list(rows)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fields)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path):
    with Path(path).open(newline='') as stream:
        return list(csv.DictReader(stream))


def read_indexed_costs(path):
    values = []
    with Path(path).open(newline='') as stream:
        for row in csv.reader(stream):
            if not row:
                continue
            if len(row) != 2 or int(row[0]) != len(values):
                raise ValueError('Original cost CSV instance indices are not contiguous: '+str(path))
            values.append(float(row[1]))
    return np.asarray(values, dtype=np.float64)


def load_split(split):
    if split not in ('train', 'val'):
        raise ValueError('R48 only permits OVRPTW train/val')
    raw = read_raw_costs('OVRPTW', split)
    with (DATA_ROOT / ('OVRPTW' + split) / 'dataset.pkl').open('rb') as stream:
        instances = pickle.load(stream)
    if len(instances) != len(raw['costs']):
        raise ValueError('Instance/label count mismatch')
    costs = raw['costs'][:, [raw['pool'].index(name) for name in PAIR]]
    nodes = np.array([len(x['node_xy']) for x in instances], dtype=np.int64)
    return instances, raw, costs, nodes


def binary_labels(costs):
    if costs.dtype != np.float64 or costs.ndim != 2 or costs.shape[1] != 2:
        raise ValueError('Binary labels require original two-column FP64 costs')
    return np.where(costs[:, 0] == costs[:, 1], -1, (costs[:, 1] < costs[:, 0]).astype(int))


def pair_winner(first, second):
    return -1 if first == second else int(second < first)


def binary_metrics(scores, costs):
    scores = np.asarray(scores, dtype=np.float64)
    labels = binary_labels(costs)
    keep = labels >= 0
    if scores.shape != costs.shape or not np.isfinite(scores).all() or not keep.any():
        raise ValueError('Invalid binary scores or no strict comparisons')
    stable = scores - scores.max(1, keepdims=True)
    log_probability = stable - np.log(np.exp(stable).sum(1, keepdims=True))
    pred = scores.argmax(1)
    picked = costs[np.arange(len(costs)), pred]
    oracle = costs.min(1)
    regret = (picked - oracle) / oracle * 100
    return dict(n_total=len(labels), n=int(keep.sum()), n_ties=int((~keep).sum()),
                class0=int((labels == 0).sum()), class1=int((labels == 1).sum()),
                accuracy=float((pred[keep] == labels[keep]).mean()),
                ce=float(-log_probability[keep, labels[keep]].mean()),
                mean_cost=float(picked[keep].mean()), pair_oracle_cost=float(oracle[keep].mean()),
                pair_regret_pct=float(regret[keep].mean()),
                vs_pair_oracle_pct=float((picked[keep].mean()/oracle[keep].mean()-1)*100))


def permutation(instance, order):
    result = copy.deepcopy(instance)
    order = np.asarray(order, dtype=np.int64)
    if sorted(order.tolist()) != list(range(len(instance['node_xy']))):
        raise ValueError('Customer permutation must be a bijection')
    for field in FIELDS:
        value = instance[field]
        if torch.is_tensor(value):
            result[field] = value[torch.from_numpy(order)].clone()
        elif isinstance(value, np.ndarray):
            result[field] = value[order].copy()
        else:
            result[field] = [value[int(i)] for i in order]
    return result


def original_route(route, order):
    mapping = np.concatenate(([0], np.asarray(order, dtype=np.int64) + 1))
    return mapping[np.asarray(route, dtype=np.int64)].tolist()


def check_open_tw_route(instance, route, runtime_float32=False):
    """Independent NumPy checker: capacity=1, speed=1, no return/depot deadline."""
    def values(name):
        value = np.asarray(instance[name])
        if runtime_float32:
            value = value.astype(np.float32)
        return value.astype(np.float64)

    depot = values('depot_xy').reshape(2)
    xy, demand = values('node_xy'), values('node_demand')
    service, start, end = [values(name) for name in ('service_time', 'tw_start', 'tw_end')]
    capacity = float(instance.get('capacity', 1.))
    n = len(xy)
    route = list(map(int, route))
    seen = np.zeros(n, dtype=np.int64)
    violations = []
    if not route or route[0] != 0:
        violations.append('missing_initial_depot')
    time_value = load = cost = max_capacity_excess = max_tw_lateness = 0.
    previous = depot
    routes, on_route = 0, False
    for node in route:
        if node == 0:
            previous, time_value, load, on_route = depot, 0., 0., False
            continue
        if not 1 <= node <= n:
            violations.append('node_out_of_range')
            continue
        j = node - 1
        if not on_route:
            routes += 1
            on_route = True
        distance = float(np.linalg.norm(xy[j] - previous))
        cost += distance
        arrival = max(time_value + distance, start[j])
        max_tw_lateness = max(max_tw_lateness, float(arrival - end[j]))
        load += demand[j]
        max_capacity_excess = max(max_capacity_excess, float(load - capacity))
        time_value, previous = arrival + service[j], xy[j]
        seen[j] += 1
    if (seen == 0).any():
        violations.append('missing_customers')
    if (seen > 1).any():
        violations.append('duplicate_customers')
    if max_capacity_excess > FEASIBILITY_TOL:
        violations.append('capacity')
    if max_tw_lateness > FEASIBILITY_TOL:
        violations.append('time_window')
    if (demand < 0).any() or capacity != 1.:
        violations.append('unexpected_OVRPTW_input_contract')
    return dict(recomputed_cost=cost, feasible=not violations, violations=';'.join(violations),
                missing_customers=int((seen == 0).sum()), duplicate_customers=int((seen > 1).sum()),
                max_capacity_excess=max_capacity_excess, max_tw_lateness=max_tw_lateness,
                routes=routes, customers=n, route_available=True)


def close_cost(first, second):
    return bool(np.isclose(first, second, atol=COST_ATOL, rtol=COST_RTOL))


def prepare_plan(root):
    root.mkdir(parents=True, exist_ok=True)
    path = root / 'audit_plan.json'
    if path.exists():
        return json.loads(path.read_text())
    prior = json.loads((R41 / 'audit_indices.json').read_text())['problems']['OVRPTW']
    old_manifest = json.loads((R41 / 'solver_manifest.json').read_text())
    solvers = {name: old_manifest['solvers']['OVRPTW/' + name] for name in PAIR}
    plan = dict(seed=2, sample_source=str((R41/'audit_indices.json').resolve()),
                sample_sha256=file_hash(R41/'audit_indices.json'), problem='OVRPTW', pair=list(PAIR),
                variants=list(VARIANTS), solvers=solvers, splits={})
    for split in ('train', 'val'):
        instances, raw, _, _ = load_split(split)
        sample = prior['splits'][split]
        if sample['dataset_hash'] != raw['data_hash'] or sample['label_hash'] != raw['label_hash']:
            raise ValueError('R41 sample sources changed')
        rng = np.random.default_rng(np.random.SeedSequence([2, 48, int(split == 'val')]))
        orders = {str(index): [rng.permutation(len(instances[index]['node_xy'])).tolist()
                              for _ in range(4)] for index in sample['indices']}
        plan['splits'][split] = dict(sample, permutations=orders)
    dump(path, plan)
    return plan


def plan_order(plan, split, index, variant):
    n = len(plan['splits'][split]['permutations'][str(index)][0])
    return (np.arange(n) if variant.startswith('original') else
            np.asarray(plan['splits'][split]['permutations'][str(index)][int(variant.rsplit('_', 1)[1])]))


def tree_features(instance):
    """Label-free, permutation-invariant geometry/demand/time statistics, speed=1."""
    xy = np.asarray(instance['node_xy'], dtype=np.float64)
    depot = np.asarray(instance['depot_xy'], dtype=np.float64).reshape(2)
    demand, service, start, end = [np.asarray(instance[k], dtype=np.float64)
                                   for k in ('node_demand', 'service_time', 'tw_start', 'tw_end')]
    n = len(xy)
    distance = np.linalg.norm(xy[:, None]-xy[None], axis=-1)
    off = ~np.eye(n, dtype=bool)
    dep_distance = np.linalg.norm(xy-depot, axis=-1)
    result = dict(customers=float(n), log_customers=float(np.log1p(n)),
                  total_demand=float(demand.sum()), total_service=float(service.sum()),
                  bbox_x=float(np.ptp(xy[:, 0])), bbox_y=float(np.ptp(xy[:, 1])),
                  centroid_depot_distance=float(np.linalg.norm(xy.mean(0)-depot)))
    def stats(name, values):
        values = np.asarray(values).ravel()
        for suffix, value in zip(('mean', 'std', 'q10', 'q50', 'q90', 'max'),
                                 (values.mean(), values.std(), *np.quantile(values, [.1, .5, .9]), values.max())):
            result[name+'_'+suffix] = float(value)
    for name, value in (('distance', distance[off]), ('depot_distance', dep_distance),
                        ('demand', demand), ('service', service), ('tw_start', start),
                        ('tw_end', end), ('tw_width', end-start)):
        stats(name, value)
    nearest = np.sort(np.where(off, distance, np.inf), axis=1)
    stats('nn1', nearest[:, 0])
    stats('nn8_mean', nearest[:, :min(8, n-1)].mean(1))
    overlap = np.minimum(end[:, None], end[None])-np.maximum(start[:, None], start[None])
    slack = end[None] - (start[:, None]+service[:, None]+distance)
    stats('directed_slack', slack[off])
    result['tw_overlap_fraction'] = float((overlap[off] >= 0).mean())
    result['directed_tw_pressure_fraction'] = float((slack[off] < -FEASIBILITY_TOL).mean())
    stats('depot_tw_slack', end-dep_distance)
    if not np.isfinite(list(result.values())).all():
        raise ValueError('Nonfinite independent tree inputs')
    return result
