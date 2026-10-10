"""Executable scenario_v2 contract and solver-independent route validation."""

import hashlib
import json
from pathlib import Path

import numpy as np

from ..unified_selector.registry import GLOBAL_SOLVERS, POOLS, PROBLEMS


ROOT = Path(__file__).resolve().parent / 'runs/R58_unified_scenario'
SCENARIO = ROOT / 'scenario_v2'
INPUT_ROOT = Path(__file__).resolve().parents[2] / 'data'
CONTRACT = {
    'scenario': 'scenario_v2',
    'revision': 1,
    'decision_source': 'R58 explicit new benchmark; not a reconstruction of historical deployments',
    'coordinates': 'original inputs converted to FP32, distances/checking accumulated in FP64',
    'objective': 'sum Euclidean travel distance; ATSP uses the original directed matrix',
    'capacity': 'raw capacity when present, otherwise normalized capacity=1; unlimited vehicles',
    'backhaul': {
        'class': 'classical_1',
        'order': 'within each vehicle route all positive deliveries precede negative pickups',
        'capacity': 'sum of deliveries <= Q and sum of absolute pickups <= Q separately',
        'pure_pickup_routes': True,
        'mixed_backhaul': False,
    },
    'time_windows': {
        'constraint': 'service START lies in [tw_start,tw_end]; early arrival waits',
        'service': 'raw per-customer service_time; depot service=0',
        'speed': 'raw speed if supplied, otherwise explicitly fixed at 1',
        'departure': 'every vehicle starts at depot time 0 (raw depot_tw_start if supplied)',
        'depot_close': 'raw depot_tw_end if supplied; otherwise +infinity, NOT 3',
        'closed_return': 'return travel takes time and must meet an explicitly supplied depot close',
    },
    'length': 'per-vehicle travel DISTANCE <= raw route_limit; excludes waiting and service',
    'open': 'a depot delimiter ends the route without return distance or return time; next vehicle starts afresh',
    'visits': 'every customer exactly once, depot can delimit any number of nonempty routes',
    'tolerance': {'feasibility_abs': 2e-5, 'reported_cost_atol': 5e-5, 'reported_cost_rtol': 2e-6},
    'eligibility': 'locked per problem/deployment before validation/test solving, never per instance',
    'failures': 'stop publication; retain instance and failure evidence; no dropping or silently repairing tours',
    'ties': 'FP64 exact cost ties; native ind regenerated using first minimum in fixed global-ID pool order',
    'test_policy': 'same prelocked contract/deployments as train/val; no protocol changes from test results',
    'old_data': 'read-only; old costs/winners are not authoritative in scenario_v2',
    'sources': [
        'https://arxiv.org/html/2406.15007v3',
        'https://epubs.siam.org/doi/10.1137/1.9780898718515.ch8',
        'https://arxiv.org/abs/1506.02465',
    ],
}


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    temporary.replace(path)


def as_numpy(value):
    if hasattr(value, 'detach'):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float32).astype(np.float64)


def fields(problem, item):
    """Canonical physical fields; no labels, native masks, or environment imports."""
    if problem == 'TSP':
        xy = as_numpy(item).reshape(-1, 2)
        if len(xy) < 2 or not np.isfinite(xy).all():
            raise ValueError('Invalid TSP coordinates')
        return {'xy': xy, 'n': len(xy)}
    if problem == 'ATSP':
        matrix = as_numpy(item).squeeze(0)
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or not np.isfinite(matrix).all() or (matrix < 0).any():
            raise ValueError('Invalid directed ATSP matrix')
        return {'matrix': matrix, 'n': len(matrix)}
    if problem == 'CVRP':
        customers = as_numpy(item['loc']).reshape(-1, 2)
        depot = as_numpy(item['depot']).reshape(1, 2)
        demand = as_numpy(item['demand']).reshape(-1)
    else:
        customers = as_numpy(item['node_xy']).reshape(-1, 2)
        depot = as_numpy(item['depot_xy']).reshape(1, 2)
        demand = as_numpy(item['node_demand']).reshape(-1)
    n = len(customers)
    has_tw, has_l = 'TW' in problem, 'L' in problem
    service = as_numpy(item.get('service_time', np.zeros(n))).reshape(-1) if has_tw else np.zeros(n)
    start = as_numpy(item.get('tw_start', np.zeros(n))).reshape(-1) if has_tw else np.zeros(n)
    end = as_numpy(item.get('tw_end', np.full(n, np.inf))).reshape(-1) if has_tw else np.full(n, np.inf)
    if has_tw and any(key not in item for key in ('service_time', 'tw_start', 'tw_end')):
        raise ValueError('Missing required customer time-window fields')
    if has_l and 'route_limit' not in item:
        raise ValueError('Missing required distance limit')
    output = dict(xy=np.concatenate((depot, customers)), n=n, demand=np.r_[0., demand],
        capacity=float(item.get('capacity', 1.)), speed=float(item.get('speed', 1.)),
        service=np.r_[0., service], start=np.r_[float(item.get('depot_tw_start', 0.)), start],
        end=np.r_[float(item.get('depot_tw_end', np.inf)), end],
        limit=float(item['route_limit']) if has_l else np.inf,
        opened=problem.startswith('O'), backhaul='B' in problem, tw=has_tw)
    if len(demand) != n or any(len(a) != n for a in (service, start, end)):
        raise ValueError('Node/attribute alignment error')
    if not np.isfinite([output['capacity'], output['speed']]).all() or output['capacity'] <= 0 or output['speed'] <= 0:
        raise ValueError('Nonpositive capacity or travel speed')
    if (not np.isfinite(output['xy']).all() or not np.isfinite(demand).all() or
            not np.isfinite(service).all() or (service < 0).any() or
            not np.isfinite(output['start']).all() or (output['start'] < 0).any() or np.isnan(output['end']).any() or
            np.isneginf(output['end']).any() or (output['end'] < output['start']).any()):
        raise ValueError('Invalid physical coordinates, demand, service or time windows')
    if has_tw and not np.isfinite(end).all():
        raise ValueError('Customer TW endpoints must be finite')
    if has_l and (not np.isfinite(output['limit']) or output['limit'] <= 0):
        raise ValueError('Invalid finite route distance limit')
    if not output['backhaul'] and (demand < 0).any():
        raise ValueError('Pickup demand in a task without B')
    return output


def check_route(problem, item, route, reported_cost=None):
    """Independent scalar simulation of the published task contract."""
    spec = fields(problem, item)
    route = np.asarray(route)
    errors = []
    if route.ndim != 1 or not len(route) or not np.issubdtype(route.dtype, np.integer):
        return dict(feasible=False, errors=['invalid_route_encoding'], cost=None, cost_matches=False)
    route = route.astype(np.int64)
    n = spec['n']
    if problem in ('TSP', 'ATSP'):
        if len(route) == n + 1 and route[0] == route[-1]:
            route = route[:-1]
        if not np.array_equal(np.sort(route), np.arange(n)):
            errors.append('visit_once')
            cost = None
        elif problem == 'ATSP':
            cost = float(spec['matrix'][route, np.roll(route, -1)].sum(dtype=np.float64))
        else:
            cost = float(np.linalg.norm(spec['xy'][route] - spec['xy'][np.roll(route, -1)], axis=1).sum(dtype=np.float64))
    else:
        if route[0] != 0 or (route < 0).any() or (route > n).any():
            errors.append('depot_or_node_id')
        if not np.array_equal(np.sort(route[route > 0]), np.arange(1, n + 1)):
            errors.append('visit_once')
        if errors:
            return dict(feasible=False, errors=errors, cost=None, cost_matches=False)
        if route[-1] != 0:
            route = np.r_[route, 0]
        tol = CONTRACT['tolerance']['feasibility_abs']
        cost = delivery = pickup = length = 0.
        time = spec['start'][0]
        seen_pickup = False
        for previous, node in zip(route[:-1], route[1:]):
            distance = float(np.linalg.norm(spec['xy'][previous] - spec['xy'][node]))
            if node == 0:
                if not spec['opened']:
                    length += distance
                    cost += distance
                    time += distance / spec['speed']
                    if spec['tw'] and time > spec['end'][0] + tol:
                        errors.append('depot_time_window')
                if length > spec['limit'] + tol:
                    errors.append('route_distance_limit')
                if delivery > spec['capacity'] + tol:
                    errors.append('delivery_capacity')
                if pickup > spec['capacity'] + tol:
                    errors.append('pickup_capacity')
                delivery = pickup = length = 0.
                time, seen_pickup = spec['start'][0], False
                continue
            length += distance
            cost += distance
            service_start = max(time + distance / spec['speed'], spec['start'][node])
            if spec['tw'] and service_start > spec['end'][node] + tol:
                errors.append('customer_time_window')
            time = service_start + spec['service'][node]
            demand = spec['demand'][node]
            if demand > 0:
                if spec['backhaul'] and seen_pickup:
                    errors.append('backhaul_precedence')
                delivery += demand
            elif demand < 0:
                seen_pickup = True
                pickup -= demand
    tolerance = CONTRACT['tolerance']
    matches = cost is not None and (reported_cost is None or bool(np.isclose(
        cost, reported_cost, atol=tolerance['reported_cost_atol'], rtol=tolerance['reported_cost_rtol'])))
    return dict(feasible=not errors, errors=sorted(set(errors)), cost=cost, cost_matches=matches)


def write_contract(root=ROOT):
    path = Path(root) / 'task_contract.json'
    payload = dict(contract=CONTRACT, contract_sha256=canonical_hash(CONTRACT),
                   global_solver_order=list(GLOBAL_SOLVERS), problems=list(PROBLEMS),
                   original_pools={p: list(POOLS[p]) for p in PROBLEMS})
    if path.exists() and json.loads(path.read_text()) != payload:
        raise ValueError('Contract changed in an existing run; create a new scenario revision')
    save_json(path, payload)
    return payload


if __name__ == '__main__':
    print(json.dumps(write_contract(), indent=2, ensure_ascii=False))
