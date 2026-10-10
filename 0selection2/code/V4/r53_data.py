"""Immutable R45A rankings, raw-cost pair labels, and paired R53 data plans."""

import json
from pathlib import Path

import numpy as np
import torch

from code.unified_selector.registry import PROBLEMS
from .local_geometry import FIELDS, local_geometry
from .multitask_probe import dump, file_hash, state_hash
from .pair_specialist import PAIR, gate_mask, ordinal_scores, pair_targets, stable_order
from .performance_evaluation import decision_metrics
from .performance_targets import read_raw_costs
from .r42_experiment import paired_schedule
from .r48_common import write_csv
from .train import make_loader


ROOT = Path('code/V4/runs/R53_pair_specialist')
BASELINE = Path('code/V4/runs/R45_solver_code_embeddings/A_handcrafted_seed2')
GEOMETRY = Path('code/V4/runs/R39_performance_model/geometry_cache')
MVRP = tuple(PROBLEMS[3:])
METRIC_KEYS = ('top1', 'top2', 'top3', 'mean_cost', 'vs_sbs_pct', 'vs_oracle_pct', 'actual_regret_pct')


def aggregate(results):
    return {key: float(np.mean([row[key] for row in results.values()])) for key in METRIC_KEYS}


def grouped(results):
    return dict(ALL=aggregate(results), MVRP=aggregate({p: results[p] for p in MVRP}),
                **{p: results[p] for p in PROBLEMS[:3]})


def read_baseline(split, root=ROOT):
    best = json.loads((BASELINE/'best_eval.json').read_text())
    folder = (BASELINE/'val_predictions'/f"epoch{best['epoch']:03d}" if split == 'val'
              else BASELINE.parent/'test_predictions'/'A')
    raw, predictions, manifest = {}, {}, {}
    for p in PROBLEMS:
        r = read_raw_costs(p, split)
        path = folder/(p+'.npz')
        with np.load(path, allow_pickle=False) as saved:
            indices = saved['indices']
            np.testing.assert_array_equal(np.sort(indices), np.arange(len(r['winner'])))
            np.testing.assert_array_equal(saved['pool_ids'], r['pool_ids'])
            if 'pool' in saved and list(saved['pool']) != r['pool']:
                raise ValueError(f'{p}: solver names changed')
            np.testing.assert_array_equal(saved['costs'], r['costs'][indices])
            np.testing.assert_array_equal(saved['winner'], r['winner'][indices])
            inverse = np.argsort(indices)
            logits = saved['logits'][inverse]
            order = stable_order(logits, r['pool_ids'])
            np.testing.assert_array_equal(order[:, 0], saved['pred'][inverse])
            nodes = saved['nodes'][inverse] if 'nodes' in saved else None
        raw[p] = r
        predictions[p] = dict(logits=logits, order=order, nodes=nodes)
        manifest[p] = dict(path=str(path.resolve()), sha256=file_hash(path),
            data_hash=r['data_hash'], label_hash=r['label_hash'], pool=r['pool'],
            pool_ids=list(map(int, r['pool_ids'])), n=len(r['winner']))
    dump(root/('baseline_'+split+'_manifest.json'), manifest)
    return raw, predictions


def baseline_metrics(raw, predictions):
    per_problem = {p: decision_metrics(ordinal_scores(predictions[p]['order']), raw[p])[0] for p in PROBLEMS}
    return dict(per_problem=per_problem, groups=grouped(per_problem))


def gate_budget(raw, predictions, root=ROOT):
    rows = []
    total_regret = baseline_metrics(raw, predictions)['groups']['ALL']['actual_regret_pct']
    for p in PROBLEMS:
        r, order = raw[p], predictions[p]['order']
        cost, winner = r['costs'], r['winner']
        pred, n = order[:, 0], len(winner)
        gate = gate_mask(order, r['pool'])
        chosen = cost[np.arange(n), pred]
        minimum = cost.min(1)
        regret = (chosen-minimum)/minimum
        pair_errors = np.zeros(n, dtype=bool)
        upper = chosen.copy()
        if p in MVRP:
            a, b = [r['pool'].index(name) for name in PAIR]
            pair_errors = np.isin(pred, [a, b]) & np.isin(winner, [a, b]) & (pred != winner)
            upper[gate] = cost[:, [a, b]].min(1)[gate]
        correctable = gate & (pred != winner) & (order[:, :2] == winner[:, None]).any(1)
        reduction = float(((chosen-upper)/minimum*100).mean()/len(PROBLEMS))
        rows.append(dict(problem=p, n=n, triggered=int(gate.sum()), trigger_pct=float(gate.mean()*100),
            gated_errors=int((gate & (pred != winner)).sum()),
            correctable_native_winner_errors=int(correctable.sum()),
            all_moel_mtl_mutual_errors=int(pair_errors.sum()),
            gated_moel_mtl_mutual_errors=int((pair_errors & gate).sum()),
            recoverable_all_top1_pp=float(correctable.mean()*100/len(PROBLEMS)),
            gated_regret_contribution_pp=float((regret*gate).mean()*100/len(PROBLEMS)),
            diagnostic_recoverable_regret_pp=reduction,
            diagnostic_recoverable_total_regret_pct=100*reduction/total_regret))
    write_csv(root/'gate_budget.csv', rows)
    baseline = baseline_metrics(raw, predictions)['groups']['ALL']
    summary = dict(baseline=baseline, triggered=sum(r['triggered'] for r in rows),
        n=sum(r['n'] for r in rows), correctable_errors=sum(r['correctable_native_winner_errors'] for r in rows),
        all_pair_errors=sum(r['all_moel_mtl_mutual_errors'] for r in rows),
        gated_pair_errors=sum(r['gated_moel_mtl_mutual_errors'] for r in rows),
        diagnostic_top1_recovery_pp=sum(r['recoverable_all_top1_pp'] for r in rows),
        diagnostic_regret_recovery_pp=sum(r['diagnostic_recoverable_regret_pp'] for r in rows),
        diagnostic_only=True, gate_uses_labels=False)
    summary['diagnostic_regret_recovery_relative_pct'] = summary['diagnostic_regret_recovery_pp']/total_regret*100
    dump(root/'gate_budget.json', summary)
    return summary


def raw_training(root=ROOT):
    raw = {p: read_raw_costs(p, 'train') for p in MVRP}
    count, total, rows = 0, 0., []
    for p, r in raw.items():
        y, w = pair_targets(r['costs'], r['pool'])
        valid = y >= 0
        count += int(valid.sum())
        total += float(w[valid].sum())
        a, b = [r['pool'].index(name) for name in PAIR]
        rows.append(dict(problem=p, n=len(y), non_ties=int(valid.sum()), ties=int((~valid).sum()),
            moel_wins=int((y == 1).sum()), mtl_wins=int((y == 0).sum()),
            non_ties_with_other_full_pool_winner=int((valid & ~np.isin(r['winner'], [a, b])).sum()),
            weight_mean=float(w[valid].mean()), weight_max=float(w.max()),
            weight_q50=float(np.quantile(w[valid], .5)), weight_q95=float(np.quantile(w[valid], .95)),
            data_hash=r['data_hash'], label_hash=r['label_hash']))
    mean = total/count
    if not mean > 0:
        raise ValueError('No nonzero training comparison costs')
    write_csv(root/'training_counts.csv', rows)
    return raw, mean


def build_loaders(raw, split, device, batch_size=128):
    stats = json.loads((GEOMETRY/'stats.json').read_text())
    definition = file_hash(Path(__file__).parent/'local_geometry.py')
    result = {}
    for p in MVRP:
        r = raw[p]
        loader = make_loader(p, split, batch_size, 0, shuffle=False, cache_device=device)
        loader.drop_last = False
        np.testing.assert_array_equal(loader.batch['ind'].cpu(), r['winner'])
        np.testing.assert_array_equal(loader.batch['pool_ids'].cpu(), r['pool_ids'])
        np.testing.assert_array_equal(loader.batch['costs'].cpu(), r['costs'].astype(np.float32))
        if loader.size != len(r['winner']):
            raise ValueError('Row coverage mismatch')
        path = GEOMETRY/f'{p}_{split}.pt'
        if path.exists() and stats['sources'].get(split+'/'+p) == r['data_hash'] and stats['definition_hash'] == definition:
            geom = torch.load(path, map_location='cpu', weights_only=True)
        else:
            if split != 'test':
                raise ValueError(f'{p}: train/val geometry cache provenance changed')
            values = []
            for start in range(0, loader.size, batch_size):
                values.append(local_geometry(loader.batch['node'][start:start+batch_size, :, :2],
                                             loader.batch['node_mask'][start:start+batch_size]).cpu())
            geom = torch.cat(values)
        if geom.shape != (*loader.batch['node'].shape[:2], len(FIELDS)):
            raise ValueError('Geometry padding/row correspondence mismatch')
        loader.batch['node_geom'] = geom.to(device)
        y, w = pair_targets(r['costs'], r['pool'])
        loader.batch['binary_label'] = torch.as_tensor(y, device=device)
        loader.batch['cost_weight'] = torch.as_tensor(w, dtype=torch.float32, device=device)
        result[p] = loader
    return result


def fit_geometry(loaders, root=ROOT):
    count, total, square = 0, torch.zeros(12, dtype=torch.float64), torch.zeros(12, dtype=torch.float64)
    for loader in loaders.values():
        for start in range(0, loader.size, 512):
            keep = loader.batch['binary_label'][start:start+512] >= 0
            values = loader.batch['node_geom'][start:start+512][keep]
            mask = loader.batch['node_mask'][start:start+512][keep]
            values = values[mask].cpu().double()
            count += len(values)
            total += values.sum(0)
            square += values.square().sum(0)
    mean = total/count
    std = (square/count-mean.square()).clamp_min(0).sqrt().clamp_min(1e-6)
    result = dict(mean=mean.tolist(), std=std.tolist(), count=count, fields=list(FIELDS),
                  fitted_on='valid nodes of all 15 MVRP non-tied training queries only')
    dump(root/'geometry_stats.json', result)
    return result


def make_plan(raw, root=ROOT, epochs=40, batch=128):
    valid = {p: torch.from_numpy(np.flatnonzero(pair_targets(r['costs'], r['pool'])[0] >= 0))
             for p, r in raw.items()}
    plan = paired_schedule({p: len(v) for p, v in valid.items()}, epochs, batch, 2)
    plan['valid_indices'] = valid
    path = root/'sampling_plan.pt'
    torch.save(plan, path)
    summary = dict(path=str(path.resolve()), sha256=file_hash(path), seed=2, epochs=epochs,
        batch=batch, drop_last=False, label_resampling=False,
        updates_per_epoch=len(plan['orders'][0]),
        epochs_hashes=[state_hash(dict(permutations=permutation, orders=order))
                       for permutation, order in zip(plan['permutations'], plan['orders'])])
    dump(root/'sampling_plan.json', summary)
    return plan, summary
