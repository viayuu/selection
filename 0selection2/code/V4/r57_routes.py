"""Independent route checks: no native environment masks or solver code are used."""

import argparse
import csv
import gzip
import json
from pathlib import Path

import numpy as np

from .multitask_probe import dump, file_hash
from .performance_targets import read_raw_costs
from .r48_common import PAIR, write_csv
from .r53_data import MVRP, read_baseline
from .r57_signal import ROOT, load_instances


CACHE = Path('code/V4/runs/R54_route_supervision/route_cache')
TOL = 2e-5


def check_route(instance, route, problem, tolerance=TOL):
    """Check classical backhaul explicitly; also expose signed-load compatibility.

    Raw data do not specify a depot closing time. The depot=3 result is a
    separately named sensitivity check, never silently part of feasibility.
    """
    xy = np.concatenate((np.asarray(instance['depot_xy']).reshape(1, 2), np.asarray(instance['node_xy'])))
    xy = xy.astype(np.float32).astype(np.float64)
    n = len(xy) - 1
    route = np.asarray(route, np.int64)
    valid_ids = route.ndim == 1 and len(route) > 0 and route[0] == 0 and np.all((route >= 0) & (route <= n))
    visited = bool(valid_ids and np.array_equal(np.sort(route[route > 0]), np.arange(1, n + 1)))
    result = dict(visit_once=visited, delivery_capacity_ok=False, pickup_capacity_ok=False,
                  backhaul_precedence_ok=False, customer_tw_ok=False, route_limit_ok=False,
                  signed_load_ok=False, closed_depot_horizon3_ok=False,
                  explicit_classical_feasible=False, cost=float('nan'),
                  depot_deadline_in_raw='depot_tw_end' in instance)
    if not visited:
        return result
    if route[-1] != 0:
        route = np.append(route, 0)
    distance = np.linalg.norm(xy[route[1:]] - xy[route[:-1]], axis=1)
    opened, backhaul, tw = problem.startswith('O'), 'B' in problem, 'TW' in problem
    result['cost'] = float((distance * (route[1:] != 0) if opened else distance).sum())
    demand = np.r_[0., np.asarray(instance['node_demand'], dtype=np.float32).astype(np.float64)]
    capacity = float(instance.get('capacity', 1.))
    limit = float(instance.get('route_limit', np.inf))
    service = np.r_[0., np.asarray(instance.get('service_time', np.zeros(n)), dtype=np.float32).astype(np.float64)]
    start = np.r_[0., np.asarray(instance.get('tw_start', np.zeros(n)), dtype=np.float32).astype(np.float64)]
    end = np.r_[np.inf, np.asarray(instance.get('tw_end', np.full(n, np.inf)), dtype=np.float32).astype(np.float64)]
    speed = float(instance.get('speed', 1.))
    delivery_ok = pickup_ok = precedence_ok = tw_ok = length_ok = horizon_ok = signed_ok = True
    delivery = pickup = length = clock = 0.
    seen_pickup = False
    signed_remaining, linehauls_left = capacity, int((demand > 0).sum())
    for position in range(1, len(route)):
        node, travel = int(route[position]), float(distance[position - 1])
        if node == 0:
            if not opened:
                length += travel
                clock += travel / speed
                horizon_ok &= clock <= float(instance.get('depot_tw_end', 3.)) + tolerance
            length_ok &= length <= limit + tolerance
            delivery_ok &= delivery <= capacity + tolerance
            pickup_ok &= pickup <= capacity + tolerance
            delivery = pickup = length = clock = 0.
            seen_pickup = False
            signed_remaining = capacity if linehauls_left else 0.
            continue
        length += travel
        arrival = max(clock + travel / speed, float(start[node]))
        if tw:
            tw_ok &= arrival <= end[node] + tolerance
        clock = arrival + service[node]
        value = float(demand[node])
        if value > 0:
            precedence_ok &= not seen_pickup
            delivery += value
            linehauls_left -= 1
        elif value < 0:
            seen_pickup = True
            pickup -= value
        signed_remaining -= value
        signed_ok &= -tolerance <= signed_remaining <= capacity + tolerance
    result.update(delivery_capacity_ok=bool(delivery_ok), pickup_capacity_ok=bool(pickup_ok),
        backhaul_precedence_ok=bool(precedence_ok if backhaul else True),
        customer_tw_ok=bool(tw_ok), route_limit_ok=bool(length_ok), signed_load_ok=bool(signed_ok),
        closed_depot_horizon3_ok=bool(horizon_ok if tw and not opened else True))
    result['explicit_classical_feasible'] = bool(delivery_ok and pickup_ok and
        (precedence_ok if backhaul else True) and tw_ok and length_ok)
    return result


def run(root=ROOT):
    root.mkdir(parents=True, exist_ok=True)
    protocol = dict(source=str(CACHE.resolve()), route_generation=False, splits=['train', 'val'],
        checker='independent scalar route simulation; does not import native solver environments',
        tolerance=TOL, fp32_inputs_then_fp64_arithmetic=True,
        checks=['visit every customer exactly once', 'per-vehicle delivery sum <= capacity',
                'per-vehicle pickup sum <= capacity', 'classical B: no delivery after a pickup within a route',
                'service starts inside each customer window, waiting and service included',
                'per-vehicle distance <= route_limit; open excludes depot return',
                'closed depot horizon=3 is reported separately because raw data omit it'],
        ambiguity='Raw signed-demand fields do not specify backhaul_class; classical B and native signed-load semantics are separately reported.',
        validation='only the fixed 64 R54 diagnostic routes per problem; no new labels or test',
        attribution='Flags concern the saved selected route, not proof that every possible route/label is infeasible.')
    dump(root / 'route_check_protocol.json', protocol)
    raw_val, baseline = read_baseline('val', root)
    summaries, examples, affected = [], [], []
    flags = ('visit_once', 'delivery_capacity_ok', 'pickup_capacity_ok', 'backhaul_precedence_ok',
             'customer_tw_ok', 'route_limit_ok', 'signed_load_ok', 'closed_depot_horizon3_ok',
             'explicit_classical_feasible')
    output = root / 'route_checks.csv.gz'
    fields = ['problem', 'split', 'index', 'solver', 'true_size', 'historical_cost',
              'independent_cost', 'abs_cost_error', 'cost_matches', *flags]
    with gzip.open(output, 'wt', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for problem in MVRP:
            for split in ('train', 'val'):
                path = CACHE / (problem + '_' + split + '.npz')
                instances = load_instances(problem, split)
                raw = read_raw_costs(problem, split) if split == 'train' else raw_val[problem]
                with np.load(path, allow_pickle=False) as archive:
                    if str(archive['data_hash']) != raw['data_hash'] or str(archive['label_hash']) != raw['label_hash']:
                        raise ValueError('Existing route cache is not bound to these inputs/labels')
                    np.testing.assert_array_equal(archive['pool'], PAIR)
                    rows = archive['selected_indices']
                    if not archive['complete'][rows].all():
                        raise ValueError('Partial R54 route cache')
                    routes, route_lengths = archive['routes'], archive['route_lengths']
                    masks = np.zeros((len(instances), 2), bool)
                    for method, name in enumerate(PAIR):
                        counts = {flag: 0 for flag in flags}
                        errors, example_count = [], 0
                        for i in rows:
                            route = routes[i, method, :int(route_lengths[i, method])]
                            checked = check_route(instances[i], route, problem)
                            old = float(raw['costs'][i, raw['pool'].index(name)])
                            error = abs(checked['cost'] - old)
                            match = bool(np.isclose(checked['cost'], old, atol=5e-5, rtol=2e-6))
                            errors.append(error)
                            masks[i, method] = not checked['explicit_classical_feasible']
                            for flag in flags:
                                counts[flag] += int(not checked[flag])
                            writer.writerow(dict(problem=problem, split=split, index=int(i), solver=name,
                                true_size=len(instances[i]['node_xy']), historical_cost=old,
                                independent_cost=checked['cost'], abs_cost_error=error,
                                cost_matches=match, **{flag: checked[flag] for flag in flags}))
                            if (masks[i, method] or not match) and example_count < 3:
                                examples.append(dict(problem=problem, split=split, index=int(i), solver=name,
                                    route=route.tolist(), demand=np.asarray(instances[i]['node_demand']).tolist(),
                                    capacity=float(instances[i].get('capacity', 1.)),
                                    route_limit=instances[i].get('route_limit'), historical_cost=old,
                                    independent_check=checked))
                                example_count += 1
                        summaries.append(dict(problem=problem, split=split, solver=name, n=len(rows),
                            max_abs_cost_error=float(max(errors)),
                            **{flag + '_failures': value for flag, value in counts.items()},
                            route_source=str(path.resolve()), route_sha256=file_hash(path)))
                    if split == 'val':
                        pred = baseline[problem]['order'][:, 0]
                        winner = raw['winner']
                        regret = (raw['costs'][np.arange(len(pred)), pred] - raw['costs'].min(1)) / raw['costs'].min(1)
                        selected_flag = np.zeros(len(pred), bool)
                        winner_flag = np.zeros(len(pred), bool)
                        for method, name in enumerate(PAIR):
                            slot = raw['pool'].index(name)
                            selected_flag |= masks[:, method] & (pred == slot)
                            winner_flag |= masks[:, method] & (winner == slot)
                        affected.append(dict(problem=problem, n_val=len(pred), inspected_instances=len(rows),
                            routes_classical_flagged_instances=int(masks[rows].any(1).sum()),
                            R45A_selected_route_flagged=int(selected_flag.sum()),
                            native_winner_route_flagged=int(winner_flag.sum()),
                            selected_flagged_observed_regret_contribution_all_pp=float((regret * selected_flag).mean() * 100 / 18),
                            any_pair_route_flagged_observed_regret_contribution_all_pp=float((regret[rows] * masks[rows].any(1)).sum() / len(pred) * 100 / 18),
                            no_full_val_extrapolation=True))
                print(f'[R57 routes] {problem}/{split}: {2 * len(rows)} independent checks', flush=True)
    write_csv(root / 'route_check_summary.csv', summaries)
    write_csv(root / 'route_affected_validation.csv', affected)
    dump(root / 'route_violation_examples.json', examples)
    return summaries


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    run(parser.parse_args().root)
