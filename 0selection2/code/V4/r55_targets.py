"""Convert already saved R54 routes into billed undirected edge-count differences."""

import argparse
import json
import pickle
from pathlib import Path
import time

import numpy as np
import torch

from code.unified_selector.registry import DATA_ROOT
from .multitask_probe import dump, file_hash
from .pair_specialist import PAIR
from .performance_targets import read_raw_costs
from .r48_common import write_csv
from .r53_data import MVRP


ROOT = Path('code/V4/runs/R55_route_cost_difference')
ROUTES = Path('code/V4/runs/R54_route_supervision')
ATOL, RTOL = 5e-5, 2e-6


def billed_edges(route, customers, opened):
    """Upper triangle stores counts; omit open-route returns before merging directions."""
    route = np.asarray(route, dtype=np.int64)
    if route.ndim != 1 or not len(route) or route[0] != 0 or (route < 0).any() or (route > customers).any():
        raise ValueError('Invalid native route/depot IDs')
    if not np.array_equal(np.sort(route[route != 0]), np.arange(1, customers + 1)):
        raise ValueError('A saved route must serve every customer exactly once')
    start, end = route, np.roll(route, -1)
    keep = start != end
    if opened:
        keep &= end != 0
    low, high = np.minimum(start[keep], end[keep]), np.maximum(start[keep], end[keep])
    counts = np.zeros((customers + 1, customers + 1), dtype=np.int16)
    np.add.at(counts, (low, high), 1)
    return counts


def distances(instance):
    xy = np.concatenate((np.asarray(instance['depot_xy']).reshape(1, 2), np.asarray(instance['node_xy'])))
    xy = xy.astype(np.float32).astype(np.float64)
    return np.linalg.norm(xy[:, None] - xy[None, :], axis=-1)


def prepare(root=ROOT):
    root.mkdir(parents=True, exist_ok=True)
    receipt = json.loads((ROUTES / 'route_receipt.json').read_text())
    source = {(r['problem'], r['split']): r for r in receipt['rows']}
    previous = json.loads((root / 'target_receipt.json').read_text()) if (root / 'target_receipt.json').exists() else None
    definition = file_hash(Path(__file__))
    prior = {(r['problem'], r['split']): r for r in previous['rows']} if previous else {}
    checks, rows = [], []
    begin = time.monotonic()
    for split in ('train', 'val'):
        for p in MVRP:
            raw = read_raw_costs(p, split)
            route_path = ROUTES / 'route_cache' / (p + '_' + split + '.npz')
            route_hash = file_hash(route_path)
            if route_hash != source[(p, split)]['sha256']:
                raise ValueError('R54 route archive changed after its replay receipt')
            path = root / 'target_cache' / (p + '_' + split + '.npz')
            old = prior.get((p, split))
            if old and path.exists() and previous['definition_sha256'] == definition:
                if (old['route_sha256'] != route_hash or old['data_hash'] != raw['data_hash'] or
                        old['label_hash'] != raw['label_hash'] or file_hash(path) != old['sha256']):
                    raise ValueError('R55 cached targets changed')
                rows.append(old)
                checks.append(old['check'])
                continue
            with (DATA_ROOT / (p + split) / 'dataset.pkl').open('rb') as stream:
                instances = pickle.load(stream)
            with np.load(route_path, allow_pickle=False) as saved:
                if str(saved['data_hash']) != raw['data_hash'] or str(saved['label_hash']) != raw['label_hash']:
                    raise ValueError('Routes and original inputs/costs have different hashes')
                np.testing.assert_array_equal(saved['pool'], PAIR)
                selected = saved['selected_indices'].copy()
                if not saved['complete'][selected].all():
                    raise ValueError('Incomplete original route annotation')
                if split == 'train':
                    np.testing.assert_array_equal(selected, np.arange(len(instances)))
                else:
                    if len(selected) != 64:
                        raise ValueError('R54 fixed diagnostic validation subset changed')
                customers = saved['customers']
                routes = saved['routes']
                lengths = saved['route_lengths']
                recomputed = saved['recomputed_costs']
                width = int(customers.max()) + 1
                target = np.zeros((len(instances), width, width), dtype=np.int8)
                errors, identity_errors, count_errors, nonzero, positive, negative = [], [], [], [], [], []
                a, b = [raw['pool'].index(name) for name in PAIR]
                for i in selected:
                    n = len(instances[i]['node_xy'])
                    if n != customers[i]:
                        raise ValueError('Customer count/input order mismatch')
                    counts = [billed_edges(routes[i, m, :lengths[i, m]], n, p.startswith('O'))
                              for m in range(2)]
                    distance = distances(instances[i])
                    costs = np.asarray([(count * distance).sum() for count in counts])
                    historical = raw['costs'][i, [a, b]]
                    if not np.allclose(costs, historical, atol=ATOL, rtol=RTOL):
                        raise ValueError(f'{p}/{split}/{i}: billed counts do not reproduce historical costs')
                    np.testing.assert_allclose(costs, recomputed[i], atol=1e-10, rtol=0)
                    difference = counts[1] - counts[0]
                    if np.abs(difference).max() > np.iinfo(np.int8).max:
                        raise ValueError('Count range exceeds lossless target storage')
                    target[i, :n + 1, :n + 1] = difference
                    delta = float((difference * distance).sum())
                    historical_delta = float(historical[1] - historical[0])
                    tolerance = 2 * ATOL + RTOL * np.abs(historical).sum()
                    if abs(delta - historical_delta) > tolerance:
                        raise ValueError('Difference supervision fails historical cost-difference identity')
                    errors.append(abs(delta - historical_delta))
                    identity_errors.append(abs(delta - (costs[1] - costs[0])))
                    count_errors.append(float(np.abs(costs - historical).max()))
                    nonzero.append(int(np.count_nonzero(difference)))
                    positive.append(int((difference > 0).sum()))
                    negative.append(int((difference < 0).sum()))
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(path, difference=target, selected_indices=selected, customers=saved_customers(instances),
                data_hash=raw['data_hash'], label_hash=raw['label_hash'], pool=np.asarray(PAIR), route_sha256=route_hash)
            check = dict(problem=p, split=split, n=len(selected), max_abs_cost_error=max(count_errors),
                max_abs_delta_historical_error=max(errors), mean_abs_delta_historical_error=float(np.mean(errors)),
                max_abs_algebra_identity_error=max(identity_errors), mean_nonzero_edges=float(np.mean(nonzero)),
                mean_positive_edges=float(np.mean(positive)), mean_negative_edges=float(np.mean(negative)),
                passed=True)
            checks.append(check)
            rows.append(dict(problem=p, split=split, n=len(selected), path=str(path.resolve()), sha256=file_hash(path),
                route_path=str(route_path.resolve()), route_sha256=route_hash,
                data_hash=raw['data_hash'], label_hash=raw['label_hash'], check=check))
            print(f'[R55 targets] {p}/{split} {len(selected)}: max delta error={max(errors):.3g}', flush=True)
    write_csv(root / 'target_identity_checks.csv', checks)
    result = dict(definition_sha256=definition, route_receipt_sha256=file_hash(ROUTES / 'route_receipt.json'), rows=rows,
        storage='int8 counts difference in upper triangle only; lower triangle/diagonal/padding zero',
        target='D=A_MTL-A_MOEL; positive analytic distance sum means MOEL cheaper',
        historical_cost_atol=ATOL, historical_cost_rtol=RTOL,
        delta_tolerance='2*ATOL+RTOL*(abs(historical MOEL)+abs(historical MTL))',
        no_solver_executed=True, new_annotations=0, test_read=False,
        all_original_costs_preserved=True, seconds=time.monotonic() - begin)
    if previous and previous['definition_sha256'] == definition and rows == previous['rows']:
        result['seconds'] = previous['seconds']
    dump(root / 'target_receipt.json', result)
    return result


def saved_customers(instances):
    return np.asarray([len(x['node_xy']) for x in instances], dtype=np.int64)


def attach_targets(loaders, raw, split, root=ROOT):
    receipt = json.loads((root / 'target_receipt.json').read_text())
    rows = {(r['problem'], r['split']): r for r in receipt['rows']}
    result = {}
    for p in MVRP:
        path = root / 'target_cache' / (p + '_' + split + '.npz')
        if file_hash(path) != rows[(p, split)]['sha256']:
            raise ValueError('Converted edge targets changed')
        with np.load(path, allow_pickle=False) as saved:
            if str(saved['data_hash']) != raw[p]['data_hash'] or str(saved['label_hash']) != raw[p]['label_hash']:
                raise ValueError('Target source mismatch')
            np.testing.assert_array_equal(saved['pool'], PAIR)
            loader = loaders[p]
            np.testing.assert_array_equal(saved['customers'] + 1, loader.lengths)
            device = loader.batch['node'].device
            loader.batch['edge_difference_target'] = torch.from_numpy(saved['difference'].copy()).to(device)
            result[p] = torch.from_numpy(saved['selected_indices'].copy())
        a, b = [raw[p]['pool'].index(name) for name in PAIR]
        loader.batch['cost_difference_target'] = torch.as_tensor(raw[p]['costs'][:, b] - raw[p]['costs'][:, a],
                                                                 dtype=torch.float32, device=device)
        loader.batch['oracle_cost_target'] = torch.as_tensor(raw[p]['costs'].min(1), dtype=torch.float32, device=device)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    prepare(parser.parse_args().root)
