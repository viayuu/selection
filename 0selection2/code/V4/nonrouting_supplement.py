"""Versioned integer nonrouting data and certified labels; no selector or GPU launch."""

import argparse
import ast
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from functools import lru_cache
import hashlib
import importlib.util
import itertools
import json
import math
import os
from pathlib import Path
import platform
import time
import warnings

import numpy as np


ROOT = Path('/public/home/shiys/nonrouting_supplement_20261029')
LEGACY = Path('/public/home/shiys/nonrouting_selector_20261023')
VERSION = 'nonrouting_supplement_v1_20261029'
SEED = 20261029
COUNTS = {'SMTWTP': {'train': 10000, 'val': 3750, 'test': 3750},
          'MKP': {'train': 10000, 'val': 1350, 'test': 1350},
          'BPP': {'train': 10000, 'val': 1000, 'test': 1000}}
SPLITS = {'train': 101, 'val': 202, 'test': 303, 'witness': 991}
PROBLEM_IDS = {'SMTWTP': 31, 'MKP': 32, 'BPP': 33}
LEGACY_SHA = '181e4d657aabcaf0c7ac57e59c6d2d2a6f78b8335e7916782becb319d3bf4550'
NATIVE_AS_SHA = '7da5f26b10dcae34d5f9e37f967a7acce4056e38f53fd1e68b6344d9dd5aedd9'
AS_ID = 'MKP/AntSystem/deepaco_native20x100_fp64_v1'
MBS_ID = 'BPP/MBS_prime/fleszar_hindi2002_v1'
LP_ID = 'MKP/LPDualGreedy/highs_ds_v1'
CLASSIC = {'SMTWTP': ('EDD', 'WSPT', 'ATC'), 'MKP': ('SurrogateGreedy',),
           'BPP': ('FFD', 'BFD')}


class NativeCertificateError(ValueError):
    def __init__(self, message, evidence):
        super().__init__(message)
        self.evidence = evidence


def utc():
    return datetime.now(timezone.utc).isoformat()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def file_sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value, immutable=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if immutable and path.exists():
        if read(path) != value:
            raise ValueError('Refuse to overwrite immutable artifact: ' + str(path))
        return
    temporary = path.with_name('.' + path.name + '.' + str(os.getpid()))
    with temporary.open('x') as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    if immutable:
        try:
            os.link(temporary, path)
        finally:
            temporary.unlink()
    else:
        temporary.replace(path)


def owned(root):
    root = Path(root).resolve()
    if root != ROOT and ROOT not in root.parents:
        raise ValueError('Output must stay in the owned supplement root')
    return root


def cells(problem):
    if problem == 'SMTWTP':
        return [dict(n=n, TF_numerator=t, RDD_numerator=r,
                     distribution='orlibrary_style_integer_interval_then_zero_clip_v1')
                for n, t, r in itertools.product((40, 50, 100), range(1, 6), range(1, 6))]
    if problem == 'MKP':
        return [dict(n=n, dimensions=m, tightness_numerator=a, tightness_denominator=4,
                     distribution='chu_beasley_style_integer_floor_v1')
                for n, m, a in itertools.product((100, 250, 500), (5, 10, 30), (1, 2, 3))]
    if problem == 'BPP':
        return [dict(n=n, distribution=d) for n, d in itertools.product(
            (120, 240, 480), ('falkenauer_uniform_style', 'triplets_style'))]
    raise ValueError('Unsupported problem: ' + str(problem))


def signed_due_bounds(total, tf_numerator, rdd_numerator):
    return (-(-(total * (10 - 2 * tf_numerator - rdd_numerator)) // 10),
            total * (10 - 2 * tf_numerator + rdd_numerator) // 10)


def _make_instance(problem, split, index, seed=SEED):
    if problem not in PROBLEM_IDS or split not in SPLITS or index < 0:
        raise ValueError('Invalid problem/split/index')
    if problem == 'BPP' and split == 'train':
        raise ValueError('BPP TRAIN must be a read-only original view, never regenerated')
    catalogue = cells(problem)
    prefix = [seed, PROBLEM_IDS[problem], SPLITS[split]]
    order = np.random.default_rng(np.random.SeedSequence(prefix + [0])).permutation(len(catalogue))
    cell_index = int(order[index % len(order)])
    parameters = dict(catalogue[cell_index])
    attempts, witness = [], None
    for attempt in range(1000):
        subseed = prefix + [index + 1, attempt]
        rng = np.random.default_rng(np.random.SeedSequence(subseed))
        n = parameters['n']
        if problem == 'SMTWTP':
            p = rng.integers(1, 101, size=n)
            w = rng.integers(1, 11, size=n)
            low, high = signed_due_bounds(int(p.sum()), parameters['TF_numerator'], parameters['RDD_numerator'])
            signed = rng.integers(low, high + 1, size=n)
            data = dict(processing=p.tolist(), weights=w.tolist(), due_dates=np.maximum(0, signed).tolist())
            witness = dict(signed_due_draws=signed.tolist(), signed_interval=[low, high])
        elif problem == 'MKP':
            m = parameters['dimensions']
            weights = rng.integers(1, 1000, size=(m, n))
            noise = rng.integers(0, 500, size=n)
            profits = weights.sum(0) // m + noise
            capacities = parameters['tightness_numerator'] * weights.sum(1) // 4
            reasons = []
            if not np.all(weights <= capacities[:, None]):
                reasons.append('item_not_individually_feasible')
            if not np.all((capacities > 0) & (capacities < weights.sum(1))):
                reasons.append('capacity_not_strictly_between_zero_and_total')
            if reasons:
                attempts.append(dict(attempt=attempt, seed_sequence=subseed, reasons=reasons))
                continue
            data = dict(weights=weights.tolist(), profits=profits.tolist(), capacities=capacities.tolist())
            witness = dict(profit_noise=noise.tolist())
        elif parameters['distribution'] == 'falkenauer_uniform_style':
            data = dict(capacity=150, sizes=rng.integers(20, 101, size=n).tolist())
        else:
            sizes, groups = [], []
            while len(sizes) < n:
                first, second = map(int, rng.integers(251, 499, size=2))
                third = 1000 - first - second
                if 251 <= third <= 498:
                    groups.append(list(range(len(sizes), len(sizes) + 3)))
                    sizes.extend((first, second, third))
            permutation = rng.permutation(n)
            inverse = np.argsort(permutation)
            data = dict(capacity=1000, sizes=np.asarray(sizes)[permutation].tolist())
            witness = dict(hidden_triplet_bins=[[int(inverse[j]) for j in group] for group in groups],
                           known_optimum=n // 3, selector_or_solver_access=False)
        identity = digest(dict(problem=problem, data=data))
        source_id = f'{VERSION}/{problem}/{split}/{index:06d}'
        instance = dict(schema_version=3, dataset_version=VERSION, generator_version=VERSION,
            problem=problem, split=split, global_index=index, instance_id=source_id,
            source_id='synthetic://' + source_id, input_identity=identity, base_group=identity,
            cell_index=cell_index, parameters=parameters, seed_sequence=subseed, data=data,
            generation=dict(accepted_attempt=attempt, rejections=attempts,
                            style_not_exact_historical_generator=True))
        validate_instance(instance)
        return instance, witness
    raise RuntimeError('Deterministic rejection limit reached; row not silently dropped')


def make_instance(problem, split, index, seed=SEED):
    return _make_instance(problem, split, index, seed)[0]


def _integer_list(values, minimum=0):
    return isinstance(values, list) and all(type(v) is int and v >= minimum for v in values)


def validate_instance(instance):
    problem, data = instance['problem'], instance['data']
    if problem == 'SMTWTP':
        n = len(data['processing'])
        if not n or not _integer_list(data['processing'], 1) or not _integer_list(data['weights'], 1):
            raise ValueError('Invalid positive integer p/w')
        if len(data['weights']) != n or len(data['due_dates']) != n or not _integer_list(data['due_dates']):
            raise ValueError('Invalid nonnegative integer due dates')
    elif problem == 'BPP':
        if type(data['capacity']) is not int or data['capacity'] < 1 or not data['sizes']:
            raise ValueError('Invalid capacity/items')
        if not _integer_list(data['sizes'], 1) or max(data['sizes']) > data['capacity']:
            raise ValueError('Invalid integer item sizes')
    elif problem == 'MKP':
        n, m = len(data['profits']), len(data['capacities'])
        if not n or not m or not _integer_list(data['profits'], 1) or not _integer_list(data['capacities'], 1):
            raise ValueError('Invalid integer profits/capacities')
        if len(data['weights']) != m or any(len(row) != n or not _integer_list(row, 1) for row in data['weights']):
            raise ValueError('Invalid resource-by-item matrix')
        if any(max(row) > capacity or sum(row) <= capacity
               for row, capacity in zip(data['weights'], data['capacities'])):
            raise ValueError('Require well-stated MKP: every item fits, each capacity below total')
    else:
        raise ValueError(problem)
    if instance.get('input_identity') != digest(dict(problem=problem, data=data)):
        raise ValueError('Input identity mismatch')


def validate_solution(instance, solution):
    validate_instance(instance)
    data, problem = instance['data'], instance['problem']
    if problem == 'SMTWTP':
        order = solution['order']
        if not _integer_list(order) or sorted(order) != list(range(len(data['processing']))):
            raise ValueError('Certificate is not an exact job permutation')
        elapsed = objective = 0
        for j in order:
            elapsed += data['processing'][j]
            objective += data['weights'][j] * max(0, elapsed - data['due_dates'][j])
        return objective
    if problem == 'BPP':
        bins = solution['bins']
        items = [j for group in bins for j in group]
        if not _integer_list(items) or sorted(items) != list(range(len(data['sizes']))) or any(not g for g in bins):
            raise ValueError('Every item must appear once in a nonempty bin')
        if any(sum(data['sizes'][j] for j in group) > data['capacity'] for group in bins):
            raise ValueError('Integer bin capacity exceeded')
        return len(bins)
    selected = solution['selected']
    if not _integer_list(selected) or len(set(selected)) != len(selected) or any(j >= len(data['profits']) for j in selected):
        raise ValueError('Invalid integer subset certificate')
    if any(sum(row[j] for j in selected) > capacity for row, capacity in zip(data['weights'], data['capacities'])):
        raise ValueError('Original integer capacity exceeded; no repair authorized')
    return sum(data['profits'][j] for j in selected)


@lru_cache(maxsize=1)
def legacy_functions():
    path = LEGACY / 'solvers.py'
    if file_sha(path) != LEGACY_SHA:
        raise ValueError('Frozen constructive source changed')
    tree = ast.parse(path.read_text(), filename=str(path))
    names = ('single_machine', 'surrogate_greedy', 'bin_packing')
    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    if {n.name for n in functions} != set(names):
        raise ValueError('Missing frozen constructive function')
    namespace = dict(math=math, np=np)
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(path), 'exec'), namespace)
    return namespace


def mbs_one_bin(sizes, remaining, capacity):
    """Exact MBS-prime packing, including the paper's three safe pruning rules."""
    seed, tail = remaining[0], remaining[1:]
    suffix = [0] * (len(tail) + 1)
    for i in range(len(tail) - 1, -1, -1):
        suffix[i] = suffix[i + 1] + sizes[tail[i]]
    best, best_slack, current = [seed], capacity - sizes[seed], [seed]
    nodes = 0

    def search(start, slack):
        nonlocal best, best_slack, nodes
        nodes += 1
        if slack < best_slack:
            best, best_slack = list(current), slack
        if best_slack == 0 or start == len(tail) or slack < sizes[tail[-1]]:
            return
        previous_size = None
        for r in range(start, len(tail)):
            if slack - suffix[r] >= best_slack:
                break
            size = sizes[tail[r]]
            if size == previous_size:
                continue
            previous_size = size
            if size <= slack:
                current.append(tail[r])
                search(r + 1, slack - size)
                current.pop()
                if best_slack == 0:
                    return

    search(0, best_slack)
    return best, nodes


def mbs_prime(data):
    sizes, capacity = data['sizes'], data['capacity']
    remaining = sorted(range(len(sizes)), key=lambda j: (-sizes[j], j))
    bins, nodes = [], 0
    while remaining:
        group, count = mbs_one_bin(sizes, remaining, capacity)
        bins.append(group)
        nodes += count
        used = set(group)
        remaining = [j for j in remaining if j not in used]
    return dict(bins=bins), dict(search_nodes=nodes, truncation=False, seed='largest_remaining',
                                search='exact descending-size subset enumeration; first strict best')


def lp_dual_greedy(data):
    from scipy.optimize import linprog, OptimizeWarning
    weights = np.asarray(data['weights'], dtype=np.float64)
    capacities = np.asarray(data['capacities'], dtype=np.float64)
    profits = np.asarray(data['profits'], dtype=np.float64)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always', OptimizeWarning)
        result = linprog(-profits, A_ub=weights, b_ub=capacities, bounds=(0, 1),
                         method='highs-ds', options={'threads': 1, 'parallel': False})
    if not result.success or result.status != 0:
        raise RuntimeError('LP failed without fallback: ' + str(result.message))
    prices = -np.asarray(result.ineqlin.marginals, dtype=np.float64)
    if not np.isfinite(prices).all() or np.any(prices < -1e-9):
        raise ValueError('Invalid maximization-resource dual sign')
    prices = np.maximum(prices, 0.)
    denominator = prices @ weights
    priority = np.divide(profits, denominator, out=np.full(len(profits), np.inf), where=denominator > 0)
    order = sorted(range(len(profits)), key=lambda j: (-priority[j], j))
    residual, selected = list(data['capacities']), []
    for j in order:
        if all(row[j] <= cap for row, cap in zip(data['weights'], residual)):
            selected.append(j)
            residual = [cap - row[j] for row, cap in zip(data['weights'], residual)]
    lp_objective = float(-result.fun)
    dual_bound = float(prices @ capacities + np.maximum(0., profits - denominator).sum())
    if not math.isclose(dual_bound, lp_objective, rel_tol=1e-7, abs_tol=1e-6):
        raise ValueError('Resource dual prices do not reconstruct the bounded LP objective')
    return dict(selected=selected), dict(lp_status=int(result.status), lp_objective=lp_objective,
        resource_shadow_prices=prices.tolist(), bounded_lp_dual_objective=dual_bound,
        zero_denominator_count=int((denominator == 0).sum()), lp_iterations=int(result.nit),
        backend_warnings=[str(w.message) for w in caught], threads=1, lp_method='highs-ds',
        denominator_rule='exact zero -> +infinity priority; no epsilon', bound_is_not_integer_optimality=True)


def solver_ids(problem):
    ids = [f'{problem}/{name}/deterministic_v1' for name in CLASSIC[problem]]
    return sorted(ids + ([MBS_ID] if problem == 'BPP' else [LP_ID, AS_ID] if problem == 'MKP' else []))


@lru_cache(maxsize=1)
def candidate_protocols():
    result = {}
    for problem in CLASSIC:
        for name in CLASSIC[problem]:
            sid = f'{problem}/{name}/deterministic_v1'
            result[sid] = dict(solver_id=sid, source=str(LEGACY / 'solvers.py'), source_sha256=LEGACY_SHA,
                function='single_machine' if problem == 'SMTWTP' else 'bin_packing' if problem == 'BPP' else 'surrogate_greedy',
                parameters=dict(k=2., mean_processing='remaining_jobs') if name == 'ATC' else {},
                execution='one_complete_construction', objective_direction='max' if problem == 'MKP' else 'min',
                tie_break='stable item/job ID')
    result[MBS_ID] = dict(solver_id=MBS_ID, source='Fleszar-Hindi 2002 sections 2 and 3.1',
        doi='10.1016/S0305-0548(00)00082-4', implementation_sha256=file_sha(__file__),
        fixed_largest_seed=True, complete_enumeration=True, stop_at_zero_slack=True,
        pruning=['equal-size siblings', 'remaining-sum bound', 'smallest-item bound'],
        tie_break='first strict best in descending size / stable ID order', budget='no short cap, no VNS')
    result[LP_ID] = dict(solver_id=LP_ID, source='https://www.ac.tuwien.ac.at/files/pub/raidl-04a.pdf',
        section='3.5 equation 7, unbiassed profits', implementation_sha256=file_sha(__file__),
        backend='scipy.optimize.linprog highs-ds', options=dict(threads=1, parallel=False),
        dual_sign='negative inequality marginals of min(-profit)', negative_roundoff_tolerance=1e-9,
        zero_denominator='+infinity priority; stable item ID', repair=False, restarts=0)
    result[AS_ID] = dict(solver_id=AS_ID,
        source='https://github.com/henry-yeh/DeepACO/blob/9a756a3fe6cf9627ecc166110a02fdbb61f2a0d7/mkp/aco.py',
        source_sha256=NATIVE_AS_SHA, n_ants=20, iterations=100, alpha=1, beta=1, decay=.9,
        elitist=False, min_max=False, learned_heuristic=False,
        capacity_mapping='original W.T * (n//2) / original capacities; default and input dtype float64',
        seed_policy='first 32 SHA256 bits of [20261029,input_identity,solver_id]',
        native_output='unmodified alltime_best_sol; original integer certificate; no repair',
        rng_scope='per instance torch.fork_rng; CPU/GPU streams not claimed identical')
    return result


def native_ant_system(instance, device='cpu', iterations=100):
    if instance['problem'] != 'MKP' or iterations <= 0:
        raise ValueError('MKP only, positive iteration count required')
    validate_instance(instance)
    path = ROOT / 'sources/deepaco_mkp_aco.py'
    if file_sha(path) != NATIVE_AS_SHA:
        raise ValueError('Native Ant System source drift')
    import torch
    spec = importlib.util.spec_from_file_location('_supplement_native_mkp_as', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    data = instance['data']
    n = len(data['profits'])
    seed = int(digest([SEED, instance['input_identity'], AS_ID])[:8], 16)
    torch_device = torch.device(device)
    devices = [] if torch_device.type == 'cpu' else [torch_device.index if torch_device.index is not None else torch.cuda.current_device()]
    previous_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        with torch.random.fork_rng(devices=devices), torch.no_grad():
            torch.manual_seed(seed)
            profits = torch.tensor(data['profits'], dtype=torch.float64, device=torch_device)
            original = torch.tensor(data['weights'], dtype=torch.float64, device=torch_device).T
            capacities = torch.tensor(data['capacities'], dtype=torch.float64, device=torch_device)
            weights = original * (n // 2) / capacities
            native = module.ACO(profits, weights, n_ants=20, decay=.9, alpha=1, beta=1,
                                elitist=False, min_max=False, device=str(torch_device))
            objective, path_tensor = native.run(iterations)
            path_ids = [int(j) for j in path_tensor.cpu().tolist()]
            if any(j < 0 or j > n for j in path_ids):
                raise ValueError('Invalid native path item')
            solution = dict(selected=[j for j in path_ids if j < n])
            try:
                integer_objective = validate_solution(instance, solution)
            except ValueError as error:
                raise NativeCertificateError(str(error), dict(solution=solution, native_path=path_ids,
                    native_objective=float(objective.item()), input_identity=instance['input_identity'],
                    seed=seed, source_sha256=NATIVE_AS_SHA, device=str(torch_device))) from error
            if float(objective.item()) != integer_objective:
                raise ValueError('Native AS output objective differs from original integer certificate')
            loads = [sum(row[j] for j in solution['selected']) for row in data['weights']]
            normalized_loads = weights[solution['selected']].sum(0).cpu().tolist()
            extra = dict(native_objective=float(objective.item()), native_path=path_ids, seed=seed,
                dtype='float64', default_dtype='float64', device=str(torch_device), iterations=iterations,
                n_ants=20, original_integer_loads=loads, original_integer_capacities=data['capacities'],
                normalized_loads=normalized_loads, normalized_capacity=n // 2,
                minimum_integer_slack=min(c - load for c, load in zip(data['capacities'], loads)),
                rounding_boundary_rule='native floating comparisons retained; exact integer certificate must pass; never repair',
                source_sha256=NATIVE_AS_SHA, production_budget=iterations == 100)
            return solution, extra
    finally:
        torch.set_default_dtype(previous_dtype)


def solve_instance(instance, candidate, device='cpu'):
    validate_instance(instance)
    problem = instance['problem']
    sid = candidate if '/' in candidate else next((s for s in solver_ids(problem) if s.split('/')[1] == candidate), None)
    if sid not in solver_ids(problem):
        raise ValueError('Candidate not in frozen problem pool')
    method, extra = sid.split('/')[1], {}
    started = time.perf_counter()
    if sid == AS_ID:
        solution, extra = native_ant_system(instance, device)
    elif sid == LP_ID:
        solution, extra = lp_dual_greedy(instance['data'])
    elif sid == MBS_ID:
        solution, extra = mbs_prime(instance['data'])
    else:
        functions = legacy_functions()
        if problem == 'SMTWTP':
            solution = functions['single_machine'](instance['data'], method)
        elif problem == 'BPP':
            solution = functions['bin_packing'](instance['data'], method == 'BFD')
        else:
            solution = functions['surrogate_greedy'](instance['data'])
    runtime = time.perf_counter() - started
    validation_started = time.perf_counter()
    objective = validate_solution(instance, solution)
    protocol = candidate_protocols()[sid]
    return dict(solver_id=sid, family_id=method, method=method, status='feasible', feasible=True,
        valid=True, objective=objective, native_objective=objective,
        objective_direction='max' if problem == 'MKP' else 'min', solution=solution,
        input_identity=instance['input_identity'], source_id=instance.get('source_id', instance['instance_id']),
        input_dataset_version=instance['dataset_version'], instance_id=instance['instance_id'], split=instance['split'],
        runtime_s=runtime, validation_s=time.perf_counter() - validation_started,
        search_budget_s=None, deterministic=sid != AS_ID, seed=extra.get('seed'),
        inference_mode='native_100_iteration_search' if sid == AS_ID else 'one_complete_construction',
        label_protocol_identity=digest(protocol), candidate_protocol=protocol, execution_details=extra,
        bound=None, optimality_proven=False)


def _label_or_failure(instance, sid):
    try:
        return solve_instance(instance, sid)
    except Exception as error:
        return dict(solver_id=sid, status='failed', feasible=False, valid=False, objective=None,
            native_objective=None, input_identity=instance['input_identity'], error_type=type(error).__name__,
            error=str(error), label_protocol_identity=digest(candidate_protocols()[sid]))


def bpp_source_paths():
    paths = sorted((LEGACY / 'runs/20261025_\u6784\u9020\u6cd5\u9996\u6279/\u5b9e\u4f8b\u6807\u7b7e/BPP/train').glob('*.jsonl'))
    paths += sorted((LEGACY / 'runs/20261026_\u6301\u7eed\u6807\u7b7e/\u7ecf\u5178\u6807\u7b7e/BPP/train').glob('*.jsonl'))
    if len(paths) != 100:
        raise ValueError('Expected all100 original BPP TRAIN shards')
    return paths


def excluded_source_paths():
    directory = LEGACY / 'production_v2/reference_data'
    paths = [directory / ('mknapcb' + str(i) + '.txt') for i in range(1, 10)]
    paths += [directory / ('wt' + str(n) + '.txt') for n in (40, 50, 100)]
    first = LEGACY / 'runs/20261025_\u6784\u9020\u6cd5\u9996\u6279/\u5b9e\u4f8b\u6807\u7b7e/BPP'
    paths += sorted(first.glob('pilot_*/*.jsonl'))
    return paths


def excluded_identities():
    identities = set()
    for path in excluded_source_paths():
        if path.suffix == '.jsonl':
            with path.open() as stream:
                identities.update(json.loads(line)['instance']['input_identity'] for line in stream)
            continue
        values = list(map(int, path.read_text().split()))
        if path.name.startswith('wt'):
            n = int(path.stem[2:])
            if len(values) != 125 * 3 * n:
                raise ValueError('Unexpected benchmark count')
            for pos in range(0, len(values), 3 * n):
                data = dict(processing=values[pos:pos + n], weights=values[pos + n:pos + 2 * n],
                            due_dates=values[pos + 2 * n:pos + 3 * n])
                identities.add(digest(dict(problem='SMTWTP', data=data)))
        else:
            count, pos = values[0], 1
            if count != 30:
                raise ValueError('Unexpected MKP benchmark count')
            for _ in range(count):
                n, m, _ = values[pos:pos + 3]
                pos += 3
                profits = values[pos:pos + n]
                pos += n
                weights = [values[pos + i * n:pos + (i + 1) * n] for i in range(m)]
                pos += m * n
                capacities = values[pos:pos + m]
                pos += m
                identities.add(digest(dict(problem='MKP', data=dict(profits=profits, weights=weights, capacities=capacities))))
            if pos != len(values):
                raise ValueError('Unexpected MKP benchmark tail')
    return identities


@lru_cache(maxsize=4)
def _source_records(path, sha):
    path = Path(path)
    if file_sha(path) != sha:
        raise ValueError('Original BPP source changed: ' + str(path))
    with path.open() as stream:
        return [json.loads(line) for line in stream]


def resolve_record(record):
    if 'instance_ref' not in record:
        return record
    ref = record['instance_ref']
    source = _source_records(ref['path'], ref['sha256'])[ref['line'] - 1]
    instance = source['instance']
    if (instance['instance_id'] != ref['instance_id'] or instance['input_identity'] != ref['input_identity']
            or instance['global_index'] != ref['global_index']):
        raise ValueError('Read-only BPP reference identity/line mismatch')
    if 'instance' in record:
        if record['instance'] != instance or record['labels'][:len(source['labels'])] != source['labels']:
            raise ValueError('An already resolved BPP view changed its original input/labels')
        return record
    return dict(record, instance=instance, labels=[*source['labels'], *record['labels']])


def as_sample(record, solver_ids=None):
    record = resolve_record(record)
    instance = record['instance']
    problem, data = instance['problem'], instance['data']
    validate_instance(instance)
    if problem == 'BPP':
        inputs = dict(sizes=np.asarray(data['sizes'], dtype=np.int64), capacity=np.int64(data['capacity']))
    elif problem == 'MKP':
        inputs = dict(profits=np.asarray(data['profits'], dtype=np.int64),
                      resources=np.asarray(data['weights'], dtype=np.int64).T.copy(),
                      capacities=np.asarray(data['capacities'], dtype=np.int64))
    else:
        inputs = {k: np.asarray(data[v], dtype=np.int64) for k, v in
                  (('p', 'processing'), ('w', 'weights'), ('d', 'due_dates'))}
    ids = list(solver_ids) if solver_ids is not None else globals()['solver_ids'](problem)
    if ids != sorted(set(ids)):
        raise ValueError('Candidate columns must have fixed unique sorted solver IDs')
    values, observed = np.full(len(ids), np.nan, dtype=np.float64), np.zeros(len(ids), dtype=bool)
    lookup = {}
    for label in record['labels']:
        sid = label['solver_id']
        if sid in lookup:
            raise ValueError('Duplicate candidate label')
        lookup[sid] = label
        if sid in ids and label.get('feasible') is True and label.get('status') in ('feasible', 'optimal'):
            if label.get('input_identity', instance['input_identity']) != instance['input_identity']:
                raise ValueError('Candidate label belongs to another input')
            objective = validate_solution(instance, label['solution'])
            if objective != label['objective'] or label['objective_direction'] != ('max' if problem == 'MKP' else 'min'):
                raise ValueError('Native objective/direction mismatch')
            values[ids.index(sid)], observed[ids.index(sid)] = objective, True
    best = np.zeros(len(ids), dtype=bool)
    if observed.any():
        optimum = (np.max if problem == 'MKP' else np.min)(values[observed])
        best = observed & (values == optimum)
    runtimes = np.asarray([lookup[s].get('runtime_s', np.nan) if s in lookup else np.nan for s in ids], dtype=np.float64)
    labels = dict(native_objective=values, observed_mask=observed, all_best_mask=best,
                  canonical_winner_index=np.asarray(int(np.flatnonzero(best)[0]) if best.any() else -1, dtype=np.int64),
                  runtime_s=runtimes, objective_direction='max' if problem == 'MKP' else 'min')
    metadata = dict(problem=problem, split=instance['split'], dataset_version=instance['dataset_version'],
        supplement_version=VERSION, instance_id=instance['instance_id'], input_identity=instance['input_identity'],
        source_id=instance.get('source_id', instance['instance_id']), base_group=instance.get('base_group', instance['input_identity']),
        base_group_fallback='input_identity' if 'base_group' not in instance else None,
        solver_ids=ids, candidate_solver_ids=ids, base_group_id=instance.get('base_group', instance['input_identity']),
        parameters=instance['parameters'], seed_sequence=instance.get('seed_sequence'),
        source_reference=record.get('instance_ref'), labels_with_certificates=lookup,
        candidate_status={sid: lookup[sid]['status'] if sid in lookup else 'pending_gpu' if sid == AS_ID else 'missing' for sid in ids},
        canonical_winner_rule='first exact best in fixed sorted solver-ID order; not a unique native winner',
        full_pool_complete=False)
    return dict(inputs=inputs, labels=labels, metadata=metadata)


def protocol_spec():
    import scipy
    plan = LEGACY / 'refine-logs/20261028_\u4e09\u7c7b\u95ee\u9898\u8865\u9f50\u51c6\u5907/\u6570\u636e\u751f\u6210\u7ea6\u5b9a.json'
    sources = [Path(__file__), Path(__file__).with_name('test_nonrouting_supplement.py'),
               LEGACY / 'solvers.py', LEGACY / 'production_v2/data.py', LEGACY / 'production_v2/protocol.py',
               LEGACY / 'validation.py', plan, ROOT / 'sources/deepaco_mkp_aco.py']
    if file_sha(LEGACY / 'solvers.py') != LEGACY_SHA or file_sha(sources[-1]) != NATIVE_AS_SHA:
        raise ValueError('Frozen external source mismatch')
    return dict(schema_version=1, dataset_version=VERSION, data_seed=SEED, counts=COUNTS,
        cells={p: cells(p) for p in COUNTS}, candidates={p: solver_ids(p) for p in COUNTS},
        candidate_protocols=candidate_protocols(), input_identity='SHA256 canonical JSON of problem+native data, v2 compatible',
        generation_conventions=dict(SMTWTP='exact signed closed integer interval, draw then clip0; p/w positive integers',
            MKP='CB style floor-mean integer profit and rational floor capacity; deterministic recorded rejections',
            BPP='original10k read-only TRAIN reference; new independent6-cell VAL/TEST; hidden new triplet witnesses isolated'),
        benchmark_train_prohibited=True, old_data_mutated=False, full_pool_complete=False,
        missing_candidate='MKP AntSystem dispatched separately by main; unobserved until native certificate completes',
        sources_sha256={str(p.resolve()): file_sha(p) for p in sources},
        bpp_readonly_sources={str(p): file_sha(p) for p in bpp_source_paths()},
        excluded_readonly_sources={str(p): file_sha(p) for p in excluded_source_paths()},
        environment=dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__),
        relative_regret='SMTWTP oracle0: undefined; use absolute regret and zero masks, no epsilon')


def freeze_protocol(root):
    root = owned(root)
    spec = protocol_spec()
    write(root / 'protocol.json', spec, immutable=True)
    return dict(path=str(root / 'protocol.json'), sha256=file_sha(root / 'protocol.json'), identity=digest(spec))


def _store_shard(path, records):
    temporary = path.with_name('.' + path.name + '.partial.' + str(os.getpid()))
    path.parent.mkdir(parents=True, exist_ok=True)
    with temporary.open('x') as stream:
        for record in records:
            stream.write(json.dumps(record, separators=(',', ':'), allow_nan=False) + '\n')
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.link(temporary, path)
    finally:
        temporary.unlink()
    return dict(path=str(path), sha256=file_sha(path), count=len(records), bytes=path.stat().st_size)


def _records_for_bpp_train(start, stop, source_index):
    for index in range(start, stop):
        path, sha, line = source_index[index]
        source = _source_records(path, sha)[line - 1]
        instance = source['instance']
        if (instance['problem'], instance['split'], instance['global_index']) != ('BPP', 'train', index):
            raise ValueError('Original TRAIN is not an exact ordered10000 prefix')
        for label in source['labels']:
            if validate_solution(instance, label['solution']) != label['objective']:
                raise ValueError('Original BPP label certificate failed')
        if {l['solver_id'] for l in source['labels']} != set(solver_ids('BPP')) - {MBS_ID}:
            raise ValueError('Original BPP candidate set changed')
        ref = dict(path=path, sha256=sha, line=line, global_index=index,
                   instance_id=instance['instance_id'], input_identity=instance['input_identity'])
        yield dict(instance_ref=ref, labels=[_label_or_failure(instance, MBS_ID)],
                   original_labels_reused_readonly=True, full_pool_complete=False), None


def _domain_prepare(root, problem, counts, shard_size, protocol):
    import resource
    # Three CPU workers stay below6GiB of private virtual address space in total.
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    bound = 2 * 1024 ** 3
    resource.setrlimit(resource.RLIMIT_AS, (min(soft, bound) if soft > 0 else bound, hard))
    root = Path(root)
    receipt = root / 'domain_receipts' / (problem + '.json')
    if receipt.exists():
        previous = read(receipt)
        if previous['protocol'] != protocol or previous['counts'] != counts:
            raise ValueError('Existing domain receipt uses another protocol/scope')
        return previous
    source_index = []
    if problem == 'BPP':
        for path, sha in read(root / 'protocol.json')['bpp_readonly_sources'].items():
            source_index.extend((path, sha, i + 1) for i in range(len(_source_records(path, sha))))
        source_index.sort(key=lambda r: _source_records(r[0], r[1])[r[2] - 1]['instance']['global_index'])
        if len(source_index) != 10000:
            raise ValueError('Original BPP TRAIN count differs')
    shards, failures, seen, cell_counts = [], Counter(), set(), {}
    for split, count in counts.items():
        cell_count = Counter()
        for start in range(0, count, shard_size):
            stop = min(start + shard_size, count)
            path = root / 'records' / problem / split / f'{start:06d}_{stop:06d}.jsonl'
            if path.exists():
                with path.open() as stream:
                    records = [json.loads(line) for line in stream]
                if len(records) != stop - start:
                    raise ValueError('Refuse partial advertised shard')
                info = dict(path=str(path), sha256=file_sha(path), count=len(records), bytes=path.stat().st_size)
            else:
                records, witnesses = [], []
                if problem == 'BPP' and split == 'train':
                    generated = _records_for_bpp_train(start, stop, source_index)
                else:
                    generated = []
                    for index in range(start, stop):
                        instance, witness = _make_instance(problem, split, index)
                        labels = [_label_or_failure(instance, sid) for sid in solver_ids(problem) if sid != AS_ID]
                        record = dict(instance=instance, labels=labels, full_pool_complete=False,
                                      pending_candidates=[AS_ID] if problem == 'MKP' else [])
                        generated.append((record, witness))
                for record, witness in generated:
                    records.append(record)
                    if witness is not None:
                        instance = record['instance']
                        witnesses.append(dict(instance_id=instance['instance_id'], input_identity=instance['input_identity'], **witness))
                if witnesses:
                    witness_path = root / 'private_generation_witnesses' / problem / split / path.name
                    if witness_path.exists():
                        with witness_path.open() as stream:
                            if [json.loads(line) for line in stream] != witnesses:
                                raise ValueError('Generation witness changed')
                    else:
                        _store_shard(witness_path, witnesses)
                info = _store_shard(path, records)
            for offset, record in enumerate(records):
                resolved = resolve_record(record)
                instance = resolved['instance']
                if (instance['problem'], instance['split'], instance['global_index']) != (problem, split, start + offset):
                    raise ValueError('Shard instance order differs')
                as_sample(resolved)
                identity = instance['input_identity']
                if identity in seen:
                    raise ValueError('Duplicate input across supplement splits')
                seen.add(identity)
                cell_count[instance['cell_index']] += 1
                for label in resolved['labels']:
                    if label['status'] == 'failed':
                        failures[label['solver_id']] += 1
            shards.append(dict(info, problem=problem, split=split, first_index=start, stop_index=stop))
            print(json.dumps(dict(event='shard_complete', problem=problem, split=split, completed=stop,
                total=count, failed_labels=dict(failures), utc=utc())), flush=True)
            write(root / 'progress' / (problem + '.json'), dict(state='running', split=split, completed=stop,
                total=count, failed_labels=dict(failures), updated_utc=utc(), pid=os.getpid()))
        if set(cell_count) != set(range(len(cells(problem)))) and count >= len(cells(problem)):
            raise ValueError('Missing stratification cell')
        if cell_count and max(cell_count.values()) - min(cell_count.values()) > 1:
            raise ValueError('Stratification balance differs by more than one')
        cell_counts[split] = dict(cell_count)
    result = dict(problem=problem, state='complete', cpu_stage_complete=True, protocol=protocol,
        counts=counts, cell_counts=cell_counts, shards=shards, failures=dict(failures),
        full_pool_complete=False, candidate_complete=not failures and problem != 'MKP', completed_utc=utc())
    result['private_generation_witnesses'] = {str(p): file_sha(p) for p in
        sorted((root / 'private_generation_witnesses' / problem).glob('*/*.jsonl'))}
    write(root / 'domain_receipts' / (problem + '.json'), result, immutable=True)
    write(root / 'progress' / (problem + '.json'), dict(state='complete', pid=os.getpid(), updated_utc=utc()))
    return result


def prepare(root=ROOT, problems=None, workers=3, limit=None, shard_size=100):
    root = owned(root)
    if not 1 <= workers <= 3 or shard_size < 1 or (limit is not None and limit < 1):
        raise ValueError('Bounded positive worker/shard/row counts required')
    protocol = freeze_protocol(root)
    if (root / 'manifest.json').exists():
        previous = read(root / 'manifest.json')
        if previous['protocol'] != protocol:
            raise ValueError('Existing manifest binds another protocol')
        return previous
    problems = list(problems or COUNTS)
    if len(set(problems)) != len(problems) or any(p not in COUNTS for p in problems):
        raise ValueError('Invalid problems')
    results = []
    with ProcessPoolExecutor(max_workers=min(workers, len(problems))) as pool:
        futures = []
        for problem in problems:
            counts = {split: min(count, limit) if limit is not None else count for split, count in COUNTS[problem].items()}
            futures.append(pool.submit(_domain_prepare, root, problem, counts, shard_size, protocol))
        while futures:
            from concurrent.futures import wait, FIRST_COMPLETED
            done, _ = wait(futures, timeout=5., return_when=FIRST_COMPLETED)
            try:
                import psutil
                parent = psutil.Process()
                rss = sum(p.memory_info().rss for p in [parent, *parent.children(recursive=True)] if p.is_running())
                if rss >= 7 * 1024 ** 3:
                    raise MemoryError('Aggregate supplement RSS reached7GiB; no further work admitted')
            except ImportError:
                raise RuntimeError('psutil RSS guard unavailable')
            for future in done:
                results.append(future.result())
                futures.remove(future)
    manifest = dict(schema_version=1, dataset_version=VERSION, created_utc=utc(), protocol=protocol,
        cpu_stage_complete=True, full_pool_complete=False, requested_full_counts=limit is None,
        problems={r['problem']: r for r in results}, candidate_protocols=candidate_protocols(),
        status='complete_cpu_labels_pending_native_AS' if not any(r['failures'] for r in results) else 'cpu_stage_has_failed_labels',
        consumer_api='code.V4.nonrouting_supplement.iter_records/as_sample',
        hidden_generation_witnesses_are_not_inputs=True, benchmark_is_not_training=True)
    write(root / 'manifest.json', manifest, immutable=True)
    return manifest


def iter_records(root, problem, split, include_as=False):
    root = Path(root)
    manifest = read(root / 'manifest.json')
    domain = manifest['problems'][problem]
    expected = 0
    for shard in domain['shards']:
        if shard['split'] != split:
            continue
        path = Path(shard['path'])
        if file_sha(path) != shard['sha256']:
            raise ValueError('Completed shard changed')
        with path.open() as stream:
            for line in stream:
                record = resolve_record(json.loads(line))
                if record['instance']['global_index'] != expected:
                    raise ValueError('Noncontiguous source index')
                expected += 1
                if include_as and problem == 'MKP':
                    label_path = root / 'native_as_labels' / split / f'{expected - 1:06d}.json'
                    if label_path.exists():
                        label = read(label_path)
                        if label['solver_id'] != AS_ID or label['input_identity'] != record['instance']['input_identity']:
                            raise ValueError('Native AS sidecar input mismatch')
                        record = dict(record, labels=[*record['labels'], label])
                yield record
    if expected != domain['counts'][split]:
        raise ValueError('Incomplete completed-manifest split')


def audit(root=ROOT):
    root = owned(root)
    manifest = read(root / 'manifest.json')
    spec = read(root / 'protocol.json')
    for path, sha in spec['sources_sha256'].items():
        if file_sha(path) != sha:
            raise ValueError('Source drift during generation: ' + path)
    for path, sha in spec['bpp_readonly_sources'].items():
        if file_sha(path) != sha:
            raise ValueError('Old BPP source mutated')
    for path, sha in spec['excluded_readonly_sources'].items():
        if file_sha(path) != sha:
            raise ValueError('Excluded benchmark/pilot source changed')
    excluded = excluded_identities()
    seen, summaries = set(), []
    for problem, domain in manifest['problems'].items():
        for split in domain['counts']:
            objectives, zero_due, labels_count, failed, n = Counter(), 0, 0, Counter(), 0
            cells_seen = Counter()
            for record in iter_records(root, problem, split):
                instance = record['instance']
                identity = instance['input_identity']
                if identity in seen:
                    raise ValueError('Input repeated across formal splits')
                if identity in excluded:
                    raise ValueError('Public benchmark or legacy pilot leaked into a formal split')
                seen.add(identity)
                sample = as_sample(record)
                n += 1
                cells_seen[instance['cell_index']] += 1
                if problem == 'SMTWTP':
                    zero_due += instance['data']['due_dates'].count(0)
                best = sample['labels']['all_best_mask']
                objectives['tied_best_instances'] += int(best.sum() > 1)
                for label in record['labels']:
                    if label.get('feasible'):
                        labels_count += 1
                        objectives[label['solver_id']] += 1
                    else:
                        failed[label['solver_id']] += 1
            summaries.append(dict(problem=problem, split=split, instances=n, labels=labels_count,
                failed_labels=dict(failed), zero_due_dates=zero_due, cell_counts=dict(cells_seen),
                candidate_counts_and_ties=dict(objectives)))
    report = dict(passed=not any(r['failed_labels'] for r in summaries), created_utc=utc(),
        manifest_sha256=file_sha(root / 'manifest.json'), total_instances=len(seen),
        total_cpu_labels=sum(r['labels'] for r in summaries), summaries=summaries,
        native_integer_certificates_validated=True, old_bpp_sources_unchanged=True,
        excluded_public_benchmark_and_legacy_pilot_identities=len(excluded),
        benchmark_or_legacy_pilot_overlap=0, cross_split_duplicates=0, full_pool_complete=False,
        not_done='Formal native MKP AntSystem GPU labeling is main-thread owned')
    write(root / 'audit.json', report)
    return report


def _write_prepared_split(directory, records, count, problem, split, ids):
    """Independent writer of the agreed nonrouting_raw_v1 on-disk schema."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    dtypes = dict(native_objective=np.float64, observed_mask=np.bool_, all_best_mask=np.bool_,
                  canonical_winner_index=np.int64, runtime_s=np.float64)
    labels = {k: np.lib.format.open_memmap(directory / (k + '.npy'), mode='w+', dtype=dtype,
              shape=(count,) if k == 'canonical_winner_index' else (count, len(ids))) for k, dtype in dtypes.items()}
    fields, handles, offsets, shapes = {}, {}, {}, {}
    metadata_offsets, versions, observed_count, tied_count = [0], set(), 0, 0
    iterator = iter(records)
    try:
        with (directory / 'metadata.jsonl').open('xb') as stream:
            for index in range(count):
                record = next(iterator)
                sample = as_sample(record, ids)
                instance = record['instance']
                if (instance['problem'], instance['split'], instance['global_index']) != (problem, split, index):
                    raise ValueError('Prepared source order changed')
                versions.add(instance['dataset_version'])
                for name, value in sample['inputs'].items():
                    value = np.asarray(value)
                    if name not in fields:
                        fields[name] = dict(dtype=value.dtype.str, ndim=value.ndim)
                        handles[name] = (directory / ('input_' + name + '.bin')).open('xb')
                        offsets[name], shapes[name] = [0], []
                    if value.dtype.str != fields[name]['dtype'] or value.ndim != fields[name]['ndim']:
                        raise ValueError('Input field dtype/rank changed')
                    handles[name].write(value.tobytes())
                    offsets[name].append(offsets[name][-1] + value.size)
                    shapes[name].append(value.shape)
                for name in labels:
                    labels[name][index] = sample['labels'][name]
                observed_count += int(sample['labels']['observed_mask'].sum())
                tied_count += int(sample['labels']['all_best_mask'].sum() > 1)
                meta = dict(sample['metadata'])
                # Certificates remain in hashed JSONL label sources, not large model metadata.
                meta.pop('labels_with_certificates')
                meta['source_provenance'] = dict(input=record.get('instance_ref', instance.get('source_id')),
                    certificates='See immutable raw record and candidate protocol identities in supplement manifest')
                encoded = (json.dumps(meta, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode()
                stream.write(encoded)
                metadata_offsets.append(metadata_offsets[-1] + len(encoded))
            if next(iterator, None) is not None:
                raise ValueError('Prepared record tail exceeds count')
    finally:
        for handle in handles.values():
            handle.close()
        for array in labels.values():
            array.flush()
    for name, spec in fields.items():
        np.save(directory / ('input_' + name + '_offsets.npy'), np.asarray(offsets[name], dtype=np.int64))
        np.save(directory / ('input_' + name + '_shapes.npy'), np.asarray(shapes[name], dtype=np.int32).reshape(count, spec['ndim']))
        spec['elements'] = offsets[name][-1]
    np.save(directory / 'metadata_offsets.npy', np.asarray(metadata_offsets, dtype=np.int64))
    descriptor = dict(schema_version='nonrouting_raw_v1', directory=str(directory.resolve()),
        count=count, problem=problem, split=split, dataset_versions=sorted(versions),
        candidate_solver_ids=ids, objective_direction='max' if problem == 'MKP' else 'min',
        input_fields=fields, observed_labels=observed_count, tied_instances=tied_count,
        measured_pool_complete=observed_count == count * len(ids), full_pool_complete=False,
        canonical_winner_policy='first_best_in_fixed_solver_id_order_not_native_unique')
    descriptor['files'] = {p.name: dict(bytes=p.stat().st_size, sha256=file_sha(p))
                           for p in sorted(directory.iterdir()) if p.is_file()}
    write(directory / 'split_manifest.json', descriptor, immutable=True)
    return descriptor


def export_prepared(root=ROOT, output=None, include_as=True):
    root = owned(root)
    output = owned(output or root / ('prepared_with_as' if include_as else 'prepared_cpu'))
    if (output / 'manifest.json').exists():
        return read(output / 'manifest.json')
    raw = read(root / 'manifest.json')
    entries = {}
    for problem, domain in raw['problems'].items():
        ids = solver_ids(problem)
        splits = {}
        for split, count in domain['counts'].items():
            splits[split] = _write_prepared_split(output / 'nonrouting' / problem / split,
                iter_records(root, problem, split, include_as=include_as), count, problem, split, ids)
        complete = (domain['counts'] == COUNTS[problem] and
                    all(s['measured_pool_complete'] for s in splits.values()))
        entries[problem] = dict(status='ready' if complete else 'incomplete',
            ready_for_selector_training=complete, candidate_solver_ids=ids,
            candidate_inventory=[candidate_protocols()[sid] for sid in ids],
            objective_direction='max' if problem == 'MKP' else 'min', splits=splits,
            source_available_counts=domain['counts'], full_pool_complete=False,
            pending_reason=None if complete else 'Native AS or failed labels/formal counts not complete',
            deferred_candidates=[])
    manifest = dict(schema_version='nonrouting_raw_v1', prepared_version=VERSION,
        nonrouting=entries, source_manifest=dict(path=str(root / 'manifest.json'), sha256=file_sha(root / 'manifest.json')),
        data_seed=SEED, full_pool_complete=False, include_native_AS=include_as, created_utc=utc(),
        sources_sha256=read(root / 'protocol.json')['sources_sha256'])
    write(output / 'manifest.json', manifest, immutable=True)
    return manifest


def profile(root=ROOT, per_problem=2, as_demo=False):
    root = owned(root)
    rows = []
    for problem in COUNTS:
        for index in range(per_problem):
            instance, witness = _make_instance(problem, 'witness', index)
            labels = [solve_instance(instance, sid) for sid in solver_ids(problem) if sid != AS_ID]
            record = dict(instance=instance, labels=labels, witness_only=True, full_pool_complete=False)
            as_sample(record)
            rows.append(record)
            if witness is not None:
                write(root / 'witnesses' / f'{problem}_{index:03d}_generation.json', witness)
    if as_demo:
        # A real100-item/5-resource witness; no shortened search enters formal labels.
        cell = next(i for i in range(len(cells('MKP'))) if
                    make_instance('MKP', 'witness', i)['parameters'] == cells('MKP')[0])
        instance = make_instance('MKP', 'witness', cell)
        label = solve_instance(instance, AS_ID, 'cpu')
        rows.append(dict(instance=instance, labels=[label], witness_only=True, full_pool_complete=False))
    _store_shard(root / 'witnesses/real_profiles.jsonl', rows)
    summary = {}
    for sid in candidate_protocols():
        times = [l['runtime_s'] for r in rows for l in r['labels'] if l['solver_id'] == sid]
        if times:
            summary[sid] = dict(count=len(times), mean_s=float(np.mean(times)),
                p50_s=float(np.median(times)), p95_s=float(np.percentile(times, 95)), max_s=max(times))
    report = dict(created_utc=utc(), real_instances=len(rows), candidates=summary,
        certificates_passed=True, gpu_work_started=False, witnesses_not_formal_labels=True)
    write(root / 'witnesses/profile.json', report, immutable=True)
    return report


def mkp_as_cli(args):
    root = owned(args.root)
    if args.device != 'cpu' and not args.main_authorized_gpu:
        raise PermissionError('GPU scheduling is main-owned; require --main-authorized-gpu')
    manifest = read(root / 'manifest.json')
    stop = args.stop if args.stop is not None else manifest['problems']['MKP']['counts'][args.split]
    if args.start < 0 or stop <= args.start or stop > manifest['problems']['MKP']['counts'][args.split]:
        raise ValueError('Invalid native-AS interval')
    completed, started = 0, time.perf_counter()
    for record in iter_records(root, 'MKP', args.split):
        instance = record['instance']
        index = instance['global_index']
        if index < args.start:
            continue
        if index >= stop:
            break
        path = root / 'native_as_labels' / args.split / f'{index:06d}.json'
        if path.exists():
            saved = read(path)
            if saved.get('input_identity') != instance['input_identity'] or saved.get('solver_id') != AS_ID:
                raise ValueError('Existing native-AS label does not bind this input/candidate')
            validate_solution(instance, saved['solution'])
        else:
            try:
                label = solve_instance(instance, AS_ID, args.device)
            except Exception as error:
                write(root / 'native_as_failures' / args.split / (f'{index:06d}.json'),
                    dict(status='failed', complete=False, solver_id=AS_ID,
                         input_identity=instance['input_identity'], instance_id=instance['instance_id'],
                         error_type=type(error).__name__, first_error=str(error),
                         evidence=getattr(error, 'evidence', None), automatic_retry=False, created_utc=utc()))
                raise
            write(path, label, immutable=True)
        completed += 1
        if completed % 10 == 0:
            print(json.dumps(dict(event='native_as_label', split=args.split, index=index,
                                  completed=completed, seconds=time.perf_counter() - started)), flush=True)
    receipt = dict(complete=completed == stop - args.start, split=args.split, start=args.start, stop=stop,
        completed=completed, solver_id=AS_ID, manifest_sha256=file_sha(root / 'manifest.json'),
        device=args.device, iterations=100, ants=20, elapsed_s=time.perf_counter() - started, completed_utc=utc())
    write(root / 'native_as_receipts' / f'{args.split}_{args.start:06d}_{stop:06d}.json', receipt, immutable=True)
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'audit', 'profile', 'mkp-as', 'export'))
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--problems', nargs='+', choices=tuple(COUNTS))
    parser.add_argument('--workers', type=int, default=3)
    parser.add_argument('--limit', type=int)
    parser.add_argument('--shard-size', type=int, default=100)
    parser.add_argument('--per-problem', type=int, default=2)
    parser.add_argument('--as-demo', action='store_true')
    parser.add_argument('--split', choices=('train', 'val', 'test'), default='train')
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--stop', type=int)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--main-authorized-gpu', action='store_true')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--include-as', action='store_true')
    args = parser.parse_args(argv)
    if args.command == 'prepare':
        result = prepare(args.root, args.problems, args.workers, args.limit, args.shard_size)
    elif args.command == 'audit':
        result = audit(args.root)
    elif args.command == 'profile':
        result = profile(args.root, args.per_problem, args.as_demo)
    elif args.command == 'export':
        result = export_prepared(args.root, args.output, args.include_as)
    else:
        result = mkp_as_cli(args)
    print(json.dumps(result, sort_keys=True, allow_nan=False), flush=True)
    return result


if __name__ == '__main__':
    main()
