"""Replay pinned native ReLD deployments and retain best routes for R54 only."""

import argparse
import json
import pickle
from pathlib import Path
import time

import numpy as np
import torch

from code.unified_selector.registry import DATA_ROOT
from .multitask_probe import dump, file_hash
from .performance_targets import read_raw_costs
from .r41_label_stability import proportional_indices
from .r48_common import PAIR, write_csv
from .r50_solver_encoder import RELD_ROOT, source_provenance
from .r51_probe import build_solvers
from .r53_data import MVRP


ROOT = Path('code/V4/runs/R54_route_supervision')
ATOL, RTOL = 5e-5, 2e-6


def route_successors(route, customers):
    route = np.asarray(route, dtype=np.int64)
    if route.ndim != 1 or not len(route) or route[0] != 0 or (route < 0).any() or (route > customers).any():
        raise ValueError('Invalid native route/depot IDs')
    visited = route[route != 0]
    if not np.array_equal(np.sort(visited), np.arange(1, customers + 1)):
        raise ValueError('Best route must visit every customer exactly once')
    target = np.zeros(customers, dtype=np.int16)
    for position, node in enumerate(route):
        if node and position + 1 < len(route):
            target[node - 1] = route[position + 1]
    return target


def route_cost(instance, route, opened):
    xy = np.concatenate((np.asarray(instance['depot_xy']).reshape(1, 2), np.asarray(instance['node_xy'])))
    # Replay converts native input to FP32, then an independent FP64 sum is used.
    xy = xy.astype(np.float32).astype(np.float64)
    route = np.asarray(route, dtype=np.int64)
    previous, following = route, np.roll(route, -1)
    segments = np.linalg.norm(xy[following] - xy[previous], axis=-1)
    if opened:
        segments = segments * (following != 0)
    return float(segments.sum())


def load_instances(problem, split):
    with (DATA_ROOT / (problem + split) / 'dataset.pkl').open('rb') as stream:
        instances = pickle.load(stream)
    return instances, np.asarray([len(x['node_xy']) for x in instances], dtype=np.int64)


def prepare(root):
    root.mkdir(parents=True, exist_ok=True)
    provenance = source_provenance()
    provenance['source_files'].update({str(RELD_ROOT / 'envs' / (p + 'Env.py')):
        file_hash(RELD_ROOT / 'envs' / (p + 'Env.py')) for p in MVRP})
    inventory, datasets = [], {}
    for p in MVRP:
        train, sizes = load_instances(p, 'train')
        cuts = np.quantile(sizes, [.25, .5, .75]).astype(int)
        datasets[p] = {}
        for split in ('train', 'val'):
            instances, ns = (train, sizes) if split == 'train' else load_instances(p, split)
            raw = read_raw_costs(p, split)
            rows = np.arange(len(instances)) if split == 'train' else proportional_indices(ns, cuts, 64, 2)[0]
            datasets[p][split] = dict(indices=rows.tolist(), customers=ns[rows].tolist(),
                data_hash=raw['data_hash'], label_hash=raw['label_hash'])
        for solver in PAIR:
            existing = Path('code/V4/runs/R48_core_diagnosis/solver_runs/train') / solver / 'run_original_0.routes.jsonl'
            n = sum(1 for _ in existing.open()) if p == 'OVRPTW' and existing.exists() else 0
            inventory.append(dict(problem=p, solver=solver, required_train=10000, existing_original_train=n,
                missing_train=10000 - n, reusable_path=str(existing.resolve()) if n else ''))
    plan = dict(experiment='R54', seed=2, provenance=provenance, datasets=datasets,
        training_solver_instance_routes=300000, validation_solver_instance_routes=15 * 64 * 2,
        validation_route_use='structure diagnostics only; never train or checkpoint selection',
        authorization='User explicitly approved full training-route replay plus fixed validation subset in this conversation.',
        recipe='Pinned R48 weights and original native bridge; argmax, aug=1, sample=1, batch<=128; native POMO start restrictions retained',
        historical_cost_atol=ATOL, historical_cost_rtol=RTOL,
        route_selection='native minimum-cost POMO trajectory, first index on exact tie',
        target='each customer -> next customer in the same vehicle route, or 0=END; padding=-100',
        independence='R52 validation trajectories are not imported; no test or new instances; original scalar labels immutable')
    path = root / 'route_plan.json'
    if path.exists():
        previous = json.loads(path.read_text())
        if previous != plan:
            raise ValueError('Route plan changed; do not overwrite prior annotations')
    dump(path, plan)
    write_csv(root / 'route_inventory.csv', inventory)
    return plan


def native_batch(module, instances, rows, problem, split, device):
    batch, size = module.collate_problem_batch(problem + split, instances, rows)
    fields = ['depot_xy', 'node_xy', 'node_demand']
    if 'route_limit' in instances[rows[0]]:
        fields += ['route_limit']
    if 'service_time' in instances[rows[0]]:
        fields += ['service_time', 'tw_start', 'tw_end']
    for field, tensor in zip(fields, batch):
        expected = torch.stack([torch.as_tensor(instances[i][field], dtype=torch.float32) for i in rows])
        torch.testing.assert_close(tensor, expected, rtol=0, atol=0)
    if len(fields) != len(batch) or any(float(instances[i].get('capacity', 1.)) != 1. for i in rows):
        raise ValueError('Native field/capacity contract changed')
    return tuple(x.to(device) for x in batch), size


@torch.no_grad()
def solve(model, module, batch, problem):
    count, n = batch[0].shape[0], batch[1].shape[1]
    env = module.get_env_class(problem)(problem_size=n, pomo_size=n, device=batch[0].device)
    env.load_problems(count, problems=batch, aug_factor=1)
    reset, _, _ = env.reset()
    model.pre_forward(reset)
    state, reward, done = env.pre_step()
    start_nodes = state.START_NODE.clone()
    step = 0
    while not done:
        selected, _ = model(state)
        if step == 0 and torch.any(selected != 0):
            raise ValueError('Forced initial depot changed')
        if step == 1 and not torch.equal(selected, start_nodes):
            raise ValueError('Native forced POMO starts changed')
        if step >= 2 and not torch.isfinite(state.ninf_mask.gather(2, selected[..., None])).all():
            raise ValueError('Native solver selected a masked/infeasible action')
        state, reward, done = env.step(selected)
        step += 1
        if step > 2 * n + 5:
            raise ValueError('Native route construction did not finish')
    costs, best = (-reward).min(1)
    routes = env.selected_node_list[torch.arange(count, device=best.device), best]
    starts = start_nodes[torch.arange(count, device=best.device), best]
    return costs.double().cpu().numpy(), routes.cpu().numpy(), best.cpu().numpy(), starts.cpu().numpy(), env.pomo_size


def reuse_r48(problem, instances, historical, selected, arrays):
    if problem != 'OVRPTW':
        return 0
    reused = 0
    for method, name in enumerate(PAIR):
        path = Path('code/V4/runs/R48_core_diagnosis/solver_runs/train') / name / 'run_original_0.routes.jsonl'
        if not path.exists():
            continue
        for line in path.open():
            item = json.loads(line)
            i = item['instance_index']
            if i not in selected:
                continue
            route = item['route_original_ids']
            cost = route_cost(instances[i], route, True)
            if not item['independent_check']['feasible'] or not np.isclose(cost, historical[i, method], atol=ATOL, rtol=RTOL):
                raise ValueError('R48 reusable route failed cost/feasibility replay')
            put_route(arrays, i, method, route, item['scalar_cost'], cost, item['best_pomo_index'], route[1], len(instances[i]['node_xy']))
            reused += 1
    return reused


def put_route(arrays, i, method, route, native_cost, recomputed_cost, best, start, pomo):
    n = int(arrays['customers'][i])
    target = route_successors(route, n)
    route = np.asarray(route, dtype=np.int16)
    while len(route) > 1 and route[-1] == 0 and route[-2] == 0:
        route = route[:-1]
    arrays['successor'][i, method, :n] = target
    arrays['routes'][i, method, :len(route)] = route
    arrays['route_lengths'][i, method] = len(route)
    arrays['native_costs'][i, method] = native_cost
    arrays['recomputed_costs'][i, method] = recomputed_cost
    arrays['best_pomo_index'][i, method] = best
    arrays['best_start_customer'][i, method] = start
    arrays['effective_pomo'][i, method] = pomo
    arrays['complete'][i, method] = True


def replay(root, precheck=False, allow=False):
    plan = prepare(root)
    if not precheck and not allow:
        raise RuntimeError('Full route replay requires explicit --allow-route-generation')
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1 or '3090' not in torch.cuda.get_device_name(0):
        raise RuntimeError('Bind only the authorized idle RTX3090')
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    models, module = build_solvers(plan['provenance'], 'cuda:0')
    rows, started = [], time.monotonic()
    folder = root / ('route_precheck' if precheck else 'route_cache')
    folder.mkdir(parents=True, exist_ok=True)
    for p in MVRP:
        for split in (('train',) if precheck else ('train', 'val')):
            instances, sizes = load_instances(p, split)
            raw = read_raw_costs(p, split)
            source = plan['datasets'][p][split]
            if (raw['data_hash'], raw['label_hash']) != (source['data_hash'], source['label_hash']):
                raise ValueError('Route input/label source changed')
            selected = np.asarray(source['indices'])
            if precheck:
                selected = proportional_indices(sizes, np.quantile(sizes, [.25, .5, .75]), 8, 2)[0]
            total, maximum = len(instances), int(sizes.max())
            path = folder / (p + '_' + split + '.npz')
            if path.exists():
                with np.load(path, allow_pickle=False) as archive:
                    arrays = {k: archive[k] for k in archive.files}
                if str(arrays['data_hash']) != raw['data_hash'] or str(arrays['label_hash']) != raw['label_hash']:
                    raise ValueError('Cached route hashes do not match current inputs')
                np.testing.assert_array_equal(arrays['selected_indices'], selected)
            else:
                arrays = dict(successor=np.full((total, 2, maximum), -100, np.int16),
                    routes=np.full((total, 2, 2 * maximum + 5), -1, np.int16),
                    route_lengths=np.zeros((total, 2), np.int16), complete=np.zeros((total, 2), bool),
                    native_costs=np.full((total, 2), np.nan), recomputed_costs=np.full((total, 2), np.nan),
                    best_pomo_index=np.full((total, 2), -1, np.int16), best_start_customer=np.full((total, 2), -1, np.int16),
                    effective_pomo=np.zeros((total, 2), np.int16), customers=sizes, selected_indices=selected,
                    data_hash=np.asarray(raw['data_hash']), label_hash=np.asarray(raw['label_hash']), pool=np.asarray(PAIR))
            historical = raw['costs'][:, [raw['pool'].index(name) for name in PAIR]]
            reused = reuse_r48(p, instances, historical, set(selected.tolist()), arrays) if split == 'train' else 0
            begin = time.monotonic()
            solved = 0
            for size in np.unique(sizes[selected]):
                group = selected[sizes[selected] == size]
                for offset in range(0, len(group), 128):
                    group_rows = group[offset:offset + 128]
                    for method, model in enumerate(models):
                        pending = group_rows[~arrays['complete'][group_rows, method]]
                        if not len(pending):
                            continue
                        batch, actual = native_batch(module, instances, pending.tolist(), p, split, 'cuda:0')
                        costs, routes, best, starts, pomo = solve(model, module, batch, p)
                        for k, i in enumerate(pending):
                            independent = route_cost(instances[i], routes[k], p.startswith('O'))
                            if not np.isclose(costs[k], historical[i, method], atol=ATOL, rtol=RTOL) or not np.isclose(independent, costs[k], atol=ATOL, rtol=RTOL):
                                dump(root / 'route_mismatch.json', dict(problem=p, split=split, index=int(i), solver=PAIR[method],
                                    historical=float(historical[i, method]), native=float(costs[k]), independent=independent))
                                raise ValueError('Historical cost mismatch: stop annotation and training')
                            put_route(arrays, i, method, routes[k], costs[k], independent, best[k], starts[k], pomo)
                            solved += 1
                if not precheck:
                    np.savez_compressed(path, **arrays)
                    print(f'[R54 routes] {p}/{split} size={size} complete={int(arrays["complete"][selected].sum())}/{2*len(selected)}', flush=True)
            if not arrays['complete'][selected].all():
                raise ValueError('Incomplete selected route coverage')
            np.savez_compressed(path, **arrays)
            record = dict(problem=p, split=split, n=len(selected), solver_instance_routes=2 * len(selected),
                newly_solved=solved, reused_r48=reused, seconds=time.monotonic() - begin,
                max_abs_historical_error=float(np.max(np.abs(arrays['native_costs'][selected] - historical[selected]))),
                sha256=file_hash(path), path=str(path.resolve()))
            rows.append(record)
            write_csv(root / ('route_precheck.csv' if precheck else 'route_generation.csv'), rows)
            print('[R54 routes complete] ' + json.dumps(record), flush=True)
    dump(root / ('route_precheck_receipt.json' if precheck else 'route_receipt.json'),
        dict(total_seconds=time.monotonic() - started, rows=rows, device=torch.cuda.get_device_name(0),
             all_selected_routes_reproduced_historical_cost=True, test_read=False,
             dataset='precheck 8 train instances/task' if precheck else 'all train + fixed 64 val/task'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--stage', choices=('prepare', 'precheck', 'generate'), default='prepare')
    parser.add_argument('--allow-route-generation', action='store_true')
    args = parser.parse_args()
    if args.stage == 'prepare':
        prepare(args.root)
    else:
        replay(args.root, args.stage == 'precheck', args.allow_route_generation)


if __name__ == '__main__':
    main()
