"""D1 full18 structural-family preparation; solver execution is separately gated."""

from __future__ import annotations

import argparse
import copy
import gc
import importlib
import importlib.util
import itertools
import json
import os
from pathlib import Path
import pickle
import socket
import sys
import time
from collections import defaultdict

import numpy as np

from . import r114_response_data as r114

ROOT = Path(__file__).resolve().parent / 'runs/R121_structural_families/preparation'
VERSION = 'r121_d1_structural_native_recipe_v1_seed2_20261030'
SCRIPTS = r114.BRIDGE / 'scripts'
PROBLEMS = r114.PROBLEMS
PARENT_LOCK = r114.PREPARATION / 'production_execution/deployments.production.lock.json'
CVRP_LOCK = r114.PREPARATION / 'tensor_boundary_recovery_20261026/acquisition_v1/cold.lock.json'
WINNER_POLICY = r114.WINNER_POLICY
NODES_PER_FAMILY = 5


def owned(path):
    path = Path(path).resolve()
    if path != ROOT.resolve() and ROOT.resolve() not in path.parents:
        raise ValueError('New artifacts must stay under the owned R121 preparation directory')
    return path


def seal(payload, key):
    payload = dict(payload)
    payload[key] = r114.canonical_hash(payload)
    return payload


def verify_seal(payload, key):
    value = dict(payload)
    expected = value.pop(key)
    if r114.canonical_hash(value) != expected:
        raise ValueError('Immutable identity mismatch: ' + key)


def axes(problem):
    if problem == 'TSP':
        return ('cluster_connectivity', 'anisotropy')
    if problem == 'ATSP':
        return ('block_connectivity', 'directed_cycle_competition')
    second = ('window_compatibility' if 'TW' in problem else
              'signed_role_spatial_segregation' if 'B' in problem else
              'fixed_route_budget_radial_exposure' if 'L' in problem else
              'whole_population_load_spatial_coupling')
    return ('cluster_connectivity', second)


def cells(problem):
    first, second = axes(problem)
    return [('original', {})] + [
        (f'z{a}{b}', {first: a, second: b}) for a, b in itertools.product((0, 1), repeat=2)]


def validate_input(problem, item):
    item = r114.plain(item)
    if problem not in ('TSP', 'ATSP'):
        proof = r114.validate_input(problem, item)
        values = r114.fields(problem, item)
        if np.any(values['xy'] < 0) or np.any(values['xy'] > 1):
            raise ValueError('Native routing coordinates must stay in [0,1]')
        return proof
    value = np.asarray(item, dtype=np.float64)
    if value.ndim != 3 or value.shape[0] != 1 or value.shape[1] < 4 or not np.isfinite(value).all():
        raise ValueError('Invalid native singleton TSP/ATSP tensor')
    if problem == 'TSP':
        if value.shape[2] != 2 or (value < 0).any() or (value > 1).any():
            raise ValueError('Invalid TSP coordinate domain')
    elif value.shape[2] != value.shape[1] or (value < 0).any() or np.any(np.diag(value[0]) != 0):
        raise ValueError('Invalid ATSP nonnegative directed matrix/diagonal')
    return dict(legal=True, n=value.shape[1], certificate='complete graph; native tensor shape/domain')


def base_group(problem, item):
    """Exact common geometry aliases share a partition even across routing tasks."""
    if problem == 'ATSP':
        kind, values = 'directed_matrix', np.asarray(item)[0]
    elif problem == 'TSP':
        kind, values = 'coordinate_graph', np.asarray(item)[0]
    else:
        f = r114.fields(problem, item)
        kind, values = 'coordinate_graph', np.concatenate((f['depot'][None], f['xy']))
    # The native generators and backend inputs use FP32, not a new feature transform.
    return r114.canonical_hash(dict(kind=kind, values=np.asarray(values, dtype=np.float32).tolist()))


def freeze_selection(data_root=r114.DATA_ROOT, count=256, seed=2):
    if count < 1:
        raise ValueError('Positive TRAIN base budget required')
    root = Path(data_root).resolve()
    scenario = r114.read_json(root / 'scenario_manifest.json')
    bases, pools, sources = [], {}, {}
    for problem in PROBLEMS:
        pools[problem] = r114.load_pool(root, problem)
        rows, meta = r114.read_mapping(root, problem, 'train')
        heldout = set()
        for split in ('val', 'test'):
            mapping, _ = r114.read_mapping(root, problem, split)
            heldout.update(int(row['source_index']) for row in mapping)
        if heldout & {int(row['source_index']) for row in rows} or count > len(rows):
            raise ValueError('Invalid TRAIN-only base membership/budget: ' + problem)
        groups = defaultdict(list)
        for row in rows:
            groups[(row['distribution'], row['capacity_type'], int(row['scale']) // 25)].append(row)
        rng = np.random.default_rng(r114.seed_for(VERSION, seed, problem, 'input_strata'))
        keys = sorted(groups)
        rng.shuffle(keys)
        for group in groups.values():
            rng.shuffle(group)
        chosen = []
        while len(chosen) < count:
            for key in keys:
                if groups[key]:
                    chosen.append(groups[key].pop())
                    if len(chosen) == count:
                        break
        for slot, row in enumerate(chosen):
            index = int(row['source_index'])
            bases.append(dict(problem=problem, split='train', slot=slot,
                base_id=f"{scenario['scenario_id']}:{problem}:source{index}",
                source_base_id=f"{scenario['source_data']}:{problem}:{index}",
                local_index=int(row['local_index']), source_index=index,
                scale=int(row['scale']), distribution=row['distribution'], capacity_type=row['capacity_type']))
        sources[problem] = dict(path=str(root / f'{problem}train/dataset.pkl'),
            expected_sha256=meta['files']['dataset.pkl']['sha256'],
            bytes=meta['files']['dataset.pkl']['bytes'], metadata_sha256=r114.file_hash(root / f'{problem}train/metadata.json'),
            mapping_sha256=meta['files']['index_mapping.csv']['sha256'], holdout_source_exclusion=True)
    return seal(dict(schema_version=1, version=VERSION, seed=seed, split='train', data_root=str(root),
        scenario_id=scenario['scenario_id'], scenario_sha256=r114.file_hash(root / 'scenario_manifest.json'),
        bases_per_problem=count, pools=pools, sources=sources, bases=bases,
        base_selection='input-only scale/distribution/capacity strata round-robin',
        partition='hash of exact native base geometry/matrix mod5; 0=family_val, others=family_train; both TRAIN-only',
        individual_historical_labels_used=False, natural_holdouts_modified=False), 'selection_sha256')


def native_generators():
    path = SCRIPTS / 'generate_train_100k.py'
    sys.path.insert(0, str(SCRIPTS))
    spec = importlib.util.spec_from_file_location('_r121_native_generation', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, module.load_generators()


def native_proposal(base, seed, generators):
    module, loaded = generators
    key = (base['scale'], base['distribution'], base['capacity_type'])
    item = module.build_shard(base['problem'], key, 1, seed, loaded)[0]
    return r114.plain(item)


def ranks(value):
    order = np.argsort(np.asarray(value), kind='stable')
    result = np.empty(len(order), dtype=np.int64)
    result[order] = np.arange(len(order))
    return result


def closure(matrix):
    matrix = np.asarray(matrix, dtype=np.float64).copy()
    np.fill_diagonal(matrix, 0.)
    for k in range(len(matrix)):
        matrix = np.minimum(matrix, matrix[:, k, None] + matrix[None, k, :])
    return matrix


def _cluster_xy(base_xy, proposal_xy, level, groups=None):
    n = len(base_xy)
    if groups is None:
        groups = ranks(base_xy[:, 0] + .381966 * base_xy[:, 1]) * 4 // n
    centers = np.asarray([[-1, -1], [-1, 1], [1, -1], [1, 1]], dtype=np.float64)
    jitter = .5 * (base_xy + proposal_xy) - .5
    separation = (.08, .30)[level]
    return .5 + separation * centers[groups] + .12 * jitter, groups


def structural_endpoint(problem, original, proposal, settings):
    """Native input only. Axis state/lineage never becomes a model input field."""
    if not settings:
        return copy.deepcopy(r114.plain(original))
    first, second = axes(problem)
    a, b = settings[first], settings[second]
    original, proposal = r114.plain(original), r114.plain(proposal)
    if problem == 'ATSP':
        old, latent = np.asarray(original)[0], np.asarray(proposal)[0]
        n = len(old)
        order = ranks(old.sum(1) + latent.sum(0))
        block = order * 4 // n
        positive = old[old > 0]
        if not len(positive):
            raise ValueError('Degenerate zero ATSP base; do not epsilon-fill')
        scale = float(np.median(positive))
        value = .5 * (old + latent)
        value += (.15, .85)[a] * scale * (block[:, None] != block[None, :])
        for group in range(4):
            nodes = np.flatnonzero(block == group)
            nodes = nodes[np.argsort(latent[nodes].sum(1), kind='stable')]
            if len(nodes) < 2:
                raise ValueError('At least two nodes per directed block are required')
            forward = np.roll(nodes, -1)
            value[nodes, forward] = np.minimum(value[nodes, forward], (.08, .16)[b] * scale)
            if b:
                backward = np.roll(nodes, 1)
                value[nodes, backward] = np.minimum(value[nodes, backward], .16 * scale)
        return closure(value).astype(np.float32)[None].tolist()
    if problem == 'TSP':
        xy, _ = _cluster_xy(np.asarray(original)[0], np.asarray(proposal)[0], a)
        xy[:, 1] = .5 + (1., .18)[b] * (xy[:, 1] - .5)
        return xy.astype(np.float32)[None].tolist()
    old = r114.fields(problem, original)
    latent = r114.fields(problem, proposal)
    q, n = old['q'], len(old['q'])
    groups = None
    if second == 'signed_role_spatial_segregation':
        if not (q > 0).any() or not (q < 0).any():
            raise ValueError('Missing one signed role; no fake role-structure coverage')
        groups = np.empty(n, dtype=np.int64)
        for role in (False, True):
            ix = np.flatnonzero((q < 0) == role)
            local = ranks(old['xy'][ix, 1])
            groups[ix] = (2 * int(role) + local % 2) if b else local % 4
    xy, groups = _cluster_xy(old['xy'], latent['xy'], a, groups)
    # Construct within original per-node reachability, rather than repair a solver route.
    delta = xy - old['depot']
    distance = np.linalg.norm(delta, axis=1)
    old_distance = np.linalg.norm(old['xy'] - old['depot'], axis=1)
    factor = np.minimum(1., np.divide(old_distance, distance, out=np.ones(n), where=distance > 0))
    if second == 'fixed_route_budget_radial_exposure':
        factor *= (.25, .95)[b]
    xy = (old['depot'] + delta * factor[:, None]).astype(np.float32)
    result = copy.deepcopy(original)
    result['loc' if problem == 'CVRP' else 'node_xy'] = (xy[None] if problem == 'CVRP' else xy).tolist()
    if second == 'whole_population_load_spatial_coupling':
        order = np.argsort(xy[:, 0] + .381966 * xy[:, 1], kind='stable')
        demand = np.empty(n, dtype=np.float64)
        demand[order] = np.sort(q) if b else np.sort(q)[::-1]
        result['demand' if problem == 'CVRP' else 'node_demand'] = (demand[None] if problem == 'CVRP' else demand).tolist()
    if second == 'window_compatibility':
        distance = np.linalg.norm(xy.astype(np.float64) - old['depot'], axis=1)
        service = np.asarray(original['service_time'], dtype=np.float64)
        lower, upper = distance, 3. - service - distance
        if np.any(upper < lower):
            raise ValueError('No native singleton TW interval; preserve this failure')
        priority = groups * (n + 1) + ranks(latent['xy'][:, 1]) if not b else ranks(latent['xy'][:, 0])
        position = ranks(priority) / max(1, n - 1)
        width = .18 * (upper - lower)
        start = lower + position * (upper - lower - width)
        result['tw_start'], result['tw_end'] = start.tolist(), (start + width).tolist()
    return result


def make_family(problem, original, proposal):
    validate_input(problem, original)
    validate_input(problem, proposal)
    records = []
    for cell, settings in cells(problem):
        item = structural_endpoint(problem, original, proposal, settings)
        proof = validate_input(problem, item)
        if problem not in ('TSP', 'ATSP'):
            before, after = r114.fields(problem, original), r114.fields(problem, item)
            if (not np.array_equal(np.sign(before['q']), np.sign(after['q']))
                    or not np.array_equal(np.sort(before['q']), np.sort(after['q']))):
                raise ValueError('Demand multiset or per-node historical B role changed')
            for key in ('capacity', 'route_limit', 'service_time'):
                if key in original and item[key] != r114.plain(original[key]):
                    raise ValueError('Native problem definition changed: ' + key)
        records.append(dict(cell=cell, axis_state=settings, item=item, input_check=proof,
            item_sha256=r114.canonical_hash(item)))
    if len({r['item_sha256'] for r in records}) != NODES_PER_FAMILY:
        raise ValueError('Collapsed structural axis; do not claim five distinct endpoints')
    return records


def deployment_inventory(pools):
    parents = {str(path): r114.read_json(path) for path in (PARENT_LOCK, CVRP_LOCK)}
    result, source_hashes, cache = {}, {}, {}
    for path in parents:
        source_hashes[path] = r114.file_hash(path)
    for problem, pool in pools.items():
        path = CVRP_LOCK if problem == 'CVRP' else PARENT_LOCK
        parent = parents[str(path)]
        for candidate in pool:
            key = problem + '/' + candidate['method']
            recorded = parent['deployments'].get(key)
            row = dict(problem=problem, **candidate, parent_lock=str(path),
                current_runtime_verified=False, historical_scalar_replay=False,
                historical_role_draws_restored=False if problem == 'ATSP' else None,
                qualification='R121 runtime/profile/route gate pending', expected_artifacts={})
            if recorded is None:
                row.update(status='blocked_missing_deployment_recipe', missing=[key], changed_small_sources=[])
                result[key] = row
                continue
            profile_path = Path(recorded['profile_path'])
            expected = {p: h for p, h in parent['artifacts'].items() if
                p.startswith(parent['scripts_root']) or p.startswith(parent['overlay_root'])}
            expected[str(profile_path)] = parent['artifacts'].get(str(profile_path))
            missing, changed = [], []
            if profile_path.is_file():
                profile = r114.read_json(profile_path)
                if r114.canonical_hash(profile) != recorded['profile_sha256']:
                    changed.append(str(profile_path))
                dependency = profile_path.parent / 'dependency_contract.json'
                if dependency.is_file():
                    expected.update(r114.read_json(dependency).get('artifacts', {}))
                else:
                    missing.append(str(dependency))
            else:
                missing.append(str(profile_path))
            for checkpoint in recorded['checkpoints']:
                expected[checkpoint['path']] = checkpoint['sha256']
            for artifact, digest in expected.items():
                p = Path(artifact)
                if not p.is_file() or digest is None:
                    missing.append(artifact)
                elif p.suffix in ('.py', '.json', '.yaml', '.yml'):
                    if artifact not in cache:
                        cache[artifact] = r114.file_hash(p)
                    if cache[artifact] != digest:
                        changed.append(artifact)
            row.update(recorded, scripts_root=parent['scripts_root'], overlay_root=parent['overlay_root'],
                legacy_root=parent['legacy_root'], expected_artifacts=expected,
                missing=sorted(set(missing)), changed_small_sources=sorted(set(changed)),
                status='blocked_source_drift' if changed else 'blocked_missing_artifacts' if missing else 'recipe_present_runtime_pending',
                parent_runtime_scope='not in R114 response collection' if problem in ('TSP', 'ATSP') else 'R11416-task old runtime evidence, not R121 qualification',
                new_rng_protocol='seeded native role/random draws; common initial seed, not pathwise alignment')
            row['deployment_id'] = key + '@' + r114.canonical_hash(dict(version=VERSION,
                profile=recorded['profile_sha256'], budget=recorded['budget'], checkpoints=recorded['checkpoints']))[:16]
            result[key] = row
    return seal(dict(schema_version=1, version=VERSION, deployments=result,
        parent_sources_sha256=source_hashes, weight_content_verification='deferred to authorized runtime gate',
        solver_execution=False, historical_reproduction=False), 'inventory_sha256')


def prepare_plan(output, data_root=r114.DATA_ROOT, count=256, seed=2):
    output = owned(output)
    selection = freeze_selection(data_root, count, seed)
    inventory = deployment_inventory(selection['pools'])
    sources = [Path(__file__), Path(r114.__file__), Path(__file__).with_name('r114_group_worker.py'),
        SCRIPTS / 'generate_train_100k.py',
        SCRIPTS / 'generate_mvrp_diverse_v4.py', SCRIPTS / 'train_100k_common.py',
        r114.WORKSPACE / 'EasyNCO/data/ATSPGenerator.py', r114.WORKSPACE / 'EasyNCO/data/MVRPGenerator.py',
        r114.WORKSPACE / '0selection2/literature/nss代码/neural-solver-selection/datasets/data_utils.py']
    protocol = seal(dict(schema_version=1, version=VERSION, seed=seed,
        axes={p: list(axes(p)) for p in PROBLEMS}, endpoints_per_family=NODES_PER_FAMILY,
        source_sha256={str(p.resolve()): r114.file_hash(p) for p in sources},
        winner_policy=WINNER_POLICY, native_directions='all18 minimize', B_semantics='unchanged signed-load; R58 deferred',
        labels='fresh whole legal pool on original AND changed endpoints; no historical cost substitution',
        inference='current native input only; no family relations, labels, costs or IDs as forward inputs',
        scientific_status='engineering population proposal; candidate-advantage coverage unobserved',
        randomization='common initial solver seed per base/method; not conditional-expectation or historical replay',
        knobs=dict(cluster_half_separation=[.08, .30], tsp_y_scale=[1., .18],
            route_radial_exposure=[.25, .95], TW_fraction_of_singleton_interval=.18,
            atsp_interblock_penalty=[.15, .85], atsp_cycle_discount=[.08, .16])), 'protocol_sha256')
    groups = [dict(problem=p, bases=count, endpoints=count * NODES_PER_FAMILY,
        methods=[c['method'] for c in selection['pools'][p]],
        labels=count * NODES_PER_FAMILY * len(selection['pools'][p])) for p in PROBLEMS]
    for name, value in (('selection.json', selection), ('deployments.json', inventory), ('protocol.json', protocol)):
        r114.write_new_json(output / name, value)
    plan = seal(dict(schema_version=1, version=VERSION, selection_path=str(output / 'selection.json'),
        inventory_path=str(output / 'deployments.json'), protocol_path=str(output / 'protocol.json'),
        selection_sha256=selection['selection_sha256'], inventory_sha256=inventory['inventory_sha256'],
        protocol_sha256=protocol['protocol_sha256'], groups=groups,
        full18=True, legal_deployments=sum(len(pool) for pool in selection['pools'].values()),
        planned_bases=len(selection['bases']), planned_endpoints=sum(g['endpoints'] for g in groups),
        planned_label_tasks=sum(g['labels'] for g in groups), advantage_coverage='UNKNOWN until fresh complete labels',
        solver_execution=False, automatic_launch=False, R120_release_required=True,
        execution=dict(batch_size=1, fresh_backend_per_task=True, original_rerun=True,
            model_seed='R114 seed_for(2,problem,method,model)', solver_seed='version/seed/base_id/method/common_solver',
            warm_grouping='not qualified for new full18; require separate fresh/warm evidence before changing protocol'),
        qualification_pending=list(inventory['deployments']),
        blocked_deployments=[k for k, row in inventory['deployments'].items() if row['status'].startswith('blocked')]), 'plan_sha256')
    r114.write_new_json(output / 'collection_plan.json', plan)
    return plan


def load_plan(path):
    plan = r114.read_json(path)
    verify_seal(plan, 'plan_sha256')
    values = []
    for name, key in (('selection', 'selection_sha256'), ('inventory', 'inventory_sha256'), ('protocol', 'protocol_sha256')):
        value = r114.read_json(plan[name + '_path'])
        verify_seal(value, key)
        if value[key] != plan[key]:
            raise ValueError('Plan binding changed')
        values.append(value)
    return plan, *values


def prepare_inputs(plan_path, output, problems=None, limit=1):
    plan, selection, inventory, protocol = load_plan(plan_path)
    output = owned(output)
    wanted = list(problems or PROBLEMS)
    if not set(wanted) <= set(PROBLEMS) or len(set(wanted)) != len(wanted) or not 1 <= limit <= selection['bases_per_problem']:
        raise ValueError('Invalid problem subset or prepared base prefix')
    for path, digest in protocol['source_sha256'].items():
        if r114.file_hash(path) != digest:
            raise ValueError('Preparation source changed; choose a new plan version: ' + path)
    generators = native_generators()
    endpoints, tasks, families, seen_groups = [], [], [], {}
    for problem in wanted:
        source = selection['sources'][problem]
        if r114.file_hash(source['path']) != source['expected_sha256']:
            raise ValueError('Frozen TRAIN input pickle changed')
        with Path(source['path']).open('rb') as stream:
            items = pickle.load(stream)
        for base in (b for b in selection['bases'] if b['problem'] == problem and b['slot'] < limit):
            original = r114.plain(items[base['local_index']])
            if validate_input(problem, original)['n'] != base['scale']:
                raise ValueError('TRAIN input/mapping scale differs')
            group_id = base_group(problem, original)
            fold = int(group_id[:8], 16) % 5
            partition = 'family_val' if fold == 0 else 'family_train'
            seen_groups.setdefault(group_id, partition)
            if seen_groups[group_id] != partition:
                raise ValueError('Shared base geometry crossed a family partition')
            seed = r114.seed_for(VERSION, selection['seed'], base['base_id'], 'native_proposal')
            proposal = native_proposal(base, seed, generators)
            family = make_family(problem, original, proposal)
            family_id = problem + '_src' + str(base['source_index'])
            ids = []
            for row in family:
                endpoint_id = family_id + '_' + row['cell']
                ids.append(endpoint_id)
                payload = dict(schema_version=1, version=VERSION, endpoint_id=endpoint_id,
                    problem=problem, split='train', base_id=base['base_id'], item=row['item'])
                path = output / 'inputs' / (endpoint_id + '.json')
                r114.write_new_json(path, payload)
                endpoint = dict(endpoint_id=endpoint_id, problem=problem, base_id=base['base_id'],
                    source_base_id=base['source_base_id'], source_index=base['source_index'],
                    local_index=base['local_index'], scale=base['scale'], family_id=family_id,
                    family_partition=partition, group_id=group_id, cell=row['cell'], axis_state=row['axis_state'],
                    input_path=str(path), input_sha256=r114.file_hash(path), item_sha256=row['item_sha256'],
                    input_check=row['input_check'], proposal_seed=seed, proposal_sha256=r114.canonical_hash(proposal))
                endpoints.append(endpoint)
                for candidate in selection['pools'][problem]:
                    deployment = inventory['deployments'][problem + '/' + candidate['method']]
                    tasks.append(dict(task_id=endpoint_id + '__' + candidate['method'], endpoint_id=endpoint_id,
                        problem=problem, base_id=base['base_id'], family_id=family_id, group_id=group_id,
                        family_partition=partition, input_sha256=endpoint['input_sha256'], **candidate,
                        deployment_id=deployment.get('deployment_id'),
                        solver_seed=r114.seed_for(VERSION, selection['seed'], base['base_id'], candidate['method'], 'common_solver'),
                        batch_size=1, rerun_original=True, status='not_run',
                        deployment_preparation_status=deployment['status']))
            families.append(dict(family_id=family_id, base_id=base['base_id'], group_id=group_id,
                problem=problem, partition=partition, original_id=ids[0], endpoint_ids=ids,
                response_edges=[[ids[0], x] for x in ids[1:]], independent_bases=1))
        del items
        gc.collect()
    manifest = seal(dict(schema_version=1, version=VERSION, split='train', plan_sha256=plan['plan_sha256'],
        plan_path=str(Path(plan_path).resolve()), endpoints=endpoints, tasks=tasks, families=families,
        pools=selection['pools'], prepared_problems=wanted, prepared_bases_per_problem=limit,
        full18_prepared=set(wanted) == set(PROBLEMS), full_planned_population_prepared=limit == selection['bases_per_problem'] and set(wanted) == set(PROBLEMS),
        solver_execution=False, actual_label_count=0, candidate_advantage_coverage='UNKNOWN'), 'manifest_sha256')
    r114.write_new_json(output / 'manifest.json', manifest)
    return manifest


def check_tour(problem, item, tour, cost):
    if problem not in ('TSP', 'ATSP'):
        return r114.check_route(problem, item, tour, cost)
    validate_input(problem, item)
    route = np.asarray(r114.plain(tour))
    value = np.asarray(item, dtype=np.float64)[0]
    n = len(value)
    if (route.shape != (n,) or not np.isfinite(route).all() or not np.equal(route, np.floor(route)).all()
            or sorted(route.tolist()) != list(range(n))):
        raise ValueError('TSP/ATSP tour must be an exact native-node permutation')
    route = route.astype(np.int64)
    nxt = np.roll(route, -1)
    total = float(value[route, nxt].sum()) if problem == 'ATSP' else float(np.linalg.norm(value[route] - value[nxt], axis=1).sum())
    if not np.isfinite(cost) or cost <= 0 or abs(total - cost) > 2e-4 + 2e-5 * abs(total):
        raise ValueError('Native reported cost disagrees with original directed/geometric input')
    return dict(verified=True, native_cost=float(cost), recomputed_cost_fp64=total)


def full_pool_targets(pool, records):
    """Saved-only new endpoint supervision; never FP32 argmin or a missing=zero fill."""
    ordered = sorted(pool, key=lambda c: c['global_id'])
    lookup = {}
    for row in records:
        if row['method'] in lookup:
            raise ValueError('Duplicate candidate receipt')
        lookup[row['method']] = row
    if set(lookup) != {c['method'] for c in ordered}:
        raise ValueError('A full legal candidate vector is required')
    identities = {(r['endpoint_id'], r['input_sha256'], r['base_id'], r['version']) for r in records}
    if len(identities) != 1 or next(iter(identities))[-1] != VERSION:
        raise ValueError('Cannot mix endpoint or deployment versions')
    for candidate in ordered:
        row = lookup[candidate['method']]
        if (row['status'] != 'ok' or row['stage'] != 'label' or row['route_audit'].get('verified') is not True
                or row.get('global_id') != candidate['global_id'] or not row.get('deployment_id')):
            raise ValueError('Preflight/missing/failed cost is not a label')
    costs = np.asarray([lookup[c['method']]['cost'] for c in ordered], dtype=np.float64)
    if not np.isfinite(costs).all() or (costs <= 0).any():
        raise ValueError('Log advantage undefined for zero/nonpositive/nonfinite native cost')
    best = costs == costs.min()
    log = np.log(costs)
    return dict(global_ids=[c['global_id'] for c in ordered], native_cost_fp64=costs.tolist(),
        observed_mask=[True] * len(costs), all_best_mask=best.tolist(),
        native_winner_global_id=ordered[int(np.flatnonzero(best)[0])]['global_id'],
        centered_log_advantage=(log - log.mean()).tolist(), winner_policy=WINNER_POLICY)


def require_release(path, plan_sha256):
    if path is None:
        raise PermissionError('R120 main-owned TEST/LIB/report resource release is required; no solver loaded')
    release = r114.read_json(path)
    required = ('TEST_complete', 'LIB_complete', 'report_complete', 'authorized_GPUs_released')
    if (release.get('owner') != 'main' or release.get('r121_plan_sha256') != plan_sha256
            or not all(release.get(k) is True for k in required) or not release.get('evidence')):
        raise PermissionError('Incomplete main-owned release receipt; no automatic probing/waiting')
    for ref in release['evidence']:
        if r114.file_hash(ref['path']) != ref['sha256']:
            raise ValueError('R120 release evidence changed')
    return release


def require_binding(path, release, device):
    if path is None:
        raise PermissionError('Main-owned explicit host/physical-GPU/CPU binding required')
    binding = r114.read_json(path)
    if (binding.get('host') != socket.gethostname() or device != 'cuda:0'
            or binding.get('gpu_uuid') not in release.get('authorized_gpu_uuids', [])
            or os.environ.get('CUDA_VISIBLE_DEVICES') != binding.get('gpu_uuid')
            or not binding.get('cpu_ids') or set(os.sched_getaffinity(0)) != set(binding['cpu_ids'])):
        raise PermissionError('Actual process affinity/UUID environment does not match main binding')
    return r114.canonical_hash(binding)


def run_task(manifest_path, task_id, output, execute=False, device=None, release=None,
             acknowledge_new_protocol=False, stage='preflight', qualification=None, binding=None):
    manifest = r114.read_json(manifest_path)
    verify_seal(manifest, 'manifest_sha256')
    plan, selection, inventory, protocol = load_plan(manifest['plan_path'])
    if manifest['plan_sha256'] != plan['plan_sha256']:
        raise ValueError('Prepared manifest belongs to another plan')
    task = next((t for t in manifest['tasks'] if t['task_id'] == task_id), None)
    if task is None:
        raise ValueError('Unknown task ID')
    deployment = inventory['deployments'][task['problem'] + '/' + task['method']]
    if not execute:
        return dict(task=task, deployment=deployment, solver_execution=False,
            required='explicit execute/device/new-protocol acknowledgement, R120 release, then per-deployment preflight')
    release_data = require_release(release, plan['plan_sha256'])
    if not acknowledge_new_protocol or device is None or deployment['status'].startswith('blocked'):
        raise PermissionError('Explicit new protocol/device and unblocked recorded deployment required')
    binding_sha = require_binding(binding, release_data, device)
    if stage not in ('preflight', 'label'):
        raise ValueError('Unknown collection stage')
    if stage == 'label':
        if qualification is None:
            raise PermissionError('Successful same-deployment/device/plan preflight receipt required')
        q = r114.read_json(qualification)
        if (q.get('stage') != 'preflight' or q.get('status') != 'ok' or q.get('plan_sha256') != plan['plan_sha256']
                or q.get('deployment_id') != task['deployment_id'] or q.get('device') != device):
            raise PermissionError('Preflight does not qualify this deployment/device/plan')
        if q.get('binding_sha256') != binding_sha or q.get('manifest_sha256') != manifest['manifest_sha256']:
            raise PermissionError('Preflight physical binding/input-manifest mismatch')
    output = owned(output)
    if output.exists():
        raise ValueError('Preserve existing attempt; select a new receipt path')
    for path, digest in {**protocol['source_sha256'], **deployment['expected_artifacts']}.items():
        if r114.file_hash(path) != digest:
            raise ValueError('Locked source/checkpoint changed; do not use defaults: ' + path)
    endpoint = next(e for e in manifest['endpoints'] if e['endpoint_id'] == task['endpoint_id'])
    if r114.file_hash(endpoint['input_path']) != task['input_sha256']:
        raise ValueError('Prepared endpoint changed')
    # Lazy imports: no backend, torch model, or CUDA call is reached by plan/prepare/dry-run.
    from .r114_group_worker import NativeSession, reset_rng
    lock = dict(scripts_root=deployment['scripts_root'], overlay_root=deployment['overlay_root'],
        legacy_root=deployment['legacy_root'], lock_sha256=inventory['inventory_sha256'],
        deployments={task['problem'] + '/' + task['method']: deployment})
    result = dict(task, version=VERSION, plan_sha256=plan['plan_sha256'], stage=stage,
        device=device, status='error', cost=None, route_audit=None, source='fresh_solver',
        historical_scalar_replay=False, automatic_retry=False, binding_sha256=binding_sha,
        manifest_sha256=manifest['manifest_sha256'])
    started = time.monotonic()
    try:
        session = NativeSession(manifest, lock, task['problem'], task['method'], device)
        if task['problem'] not in ('TSP', 'ATSP'):
            result.update(session.solve(task_id))
        else:
            torch = session.torch
            payload = r114.read_json(endpoint['input_path'])
            native = torch.tensor(payload['item'], dtype=torch.float32)
            reset_rng(torch, task['solver_seed'], device)
            if device.startswith('cuda'):
                torch.cuda.synchronize()
            solved = session.backend.run([native])
            if device.startswith('cuda'):
                torch.cuda.synchronize()
            if len(solved.costs) != 1 or len(solved.tours) != 1:
                raise ValueError('Missing native cost/tour')
            audit = check_tour(task['problem'], payload['item'], solved.tours[0], solved.costs[0])
            result.update(status='ok', cost=float(solved.costs[0]), tour=r114.plain(solved.tours[0]),
                route_audit=audit, profile_sha256=session.profile_sha256, backend_audit=r114.plain(solved.audit))
    except Exception as error:
        result.update(status='error', cost=None, first_error=str(error), error_type=type(error).__name__)
    result.update(stage=stage, wall_seconds_including_load=time.monotonic() - started,
        deployment_id=task['deployment_id'], model_seed=r114.seed_for(2, task['problem'], task['method'], 'model'),
        seed_semantics='common initial seed, not historical replay or pathwise coupling')
    r114.write_new_json(output, result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    plan = commands.add_parser('plan')
    plan.add_argument('--output', type=Path, default=ROOT / 'planned_v1')
    plan.add_argument('--data-root', type=Path, default=r114.DATA_ROOT)
    plan.add_argument('--bases-per-problem', type=int, default=256)
    plan.add_argument('--seed', type=int, default=2)
    prepare = commands.add_parser('prepare')
    prepare.add_argument('--plan', type=Path, required=True)
    prepare.add_argument('--output', type=Path, required=True)
    prepare.add_argument('--problems', nargs='+', choices=PROBLEMS)
    prepare.add_argument('--limit', type=int, default=1)
    task = commands.add_parser('task')
    task.add_argument('--manifest', type=Path, required=True)
    task.add_argument('--task-id', required=True)
    task.add_argument('--output', type=Path, required=True)
    task.add_argument('--execute', action='store_true')
    task.add_argument('--acknowledge-new-protocol', action='store_true')
    task.add_argument('--release', type=Path)
    task.add_argument('--device')
    task.add_argument('--stage', choices=('preflight', 'label'), default='preflight')
    task.add_argument('--qualification', type=Path)
    task.add_argument('--binding', type=Path)
    args = parser.parse_args(argv)
    if args.command == 'plan':
        result = prepare_plan(args.output, args.data_root, args.bases_per_problem, args.seed)
    elif args.command == 'prepare':
        result = prepare_inputs(args.plan, args.output, args.problems, args.limit)
    else:
        result = run_task(args.manifest, args.task_id, args.output, args.execute, args.device,
            args.release, args.acknowledge_new_protocol, args.stage, args.qualification, args.binding)
    print(json.dumps({k: v for k, v in result.items() if k not in ('endpoints', 'tasks', 'families', 'deployment')}, sort_keys=True))
    return 1 if result.get('status') == 'error' else 0


if __name__ == '__main__':
    raise SystemExit(main())
