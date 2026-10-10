"""Frozen R45A validation-only error budgets, with equal problem weights."""

import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np

from code.unified_selector.registry import PROBLEMS
from .multitask_probe import dump, file_hash
from .performance_targets import read_raw_costs
from .r48_common import write_csv


ROOT = Path('code/V4/runs/R52_error_and_winner_mechanism')
BASELINE = Path('code/V4/runs/R45_solver_code_embeddings/A_handcrafted_seed2')
CATEGORIES = ('winner_rank2', 'winner_rank3', 'winner_outside_top3')


def ordered_predictions(logits, pool_ids):
    """The historical evaluator breaks equal scores by ascending global ID."""
    ids = np.argsort(pool_ids, kind='stable')
    return ids[np.argsort(-logits[:, ids], axis=1, kind='stable')]


def instance_records(problem, costs, winner, order, pool, problem_count, indices=None):
    count = len(winner)
    indices = np.arange(count) if indices is None else indices
    pred = order[:, 0]
    rank = (order == winner[:, None]).argmax(1) + 1
    minimum = costs.min(1)
    regret = (costs[np.arange(count), pred] - minimum) / minimum
    residual = (costs[np.arange(count), winner] - minimum) / minimum
    weight = 1. / (count * problem_count)
    records = []
    for i in range(count):
        category = 'correct' if rank[i] == 1 else CATEGORIES[min(int(rank[i]), 4) - 2]
        records.append(dict(problem=problem, index=int(indices[i]), selected=pool[pred[i]],
            winner=pool[winner[i]], winner_rank=int(rank[i]), category=category,
            selected_cost=float(costs[i, pred[i]]), winner_cost=float(costs[i, winner[i]]),
            oracle_cost=float(minimum[i]), regret=float(regret[i]),
            native_winner_regret=float(residual[i]), macro_weight=weight))
    return records


def error_summary(rows, all_rows, label):
    errors = [r for r in all_rows if r['category'] != 'correct']
    error_mass = sum(r['macro_weight'] for r in errors)
    loss_mass = sum(r['regret'] * r['macro_weight'] for r in all_rows)
    error_loss = sum(r['regret'] * r['macro_weight'] for r in errors)
    selected_mass = sum(r['macro_weight'] for r in rows)
    loss = sum(r['regret'] * r['macro_weight'] for r in rows)
    return dict(scope=label, count=len(rows), error_count=len(errors),
        share_of_errors_pct=100 * len(rows) / max(len(errors), 1),
        macro_share_of_errors_pct=100 * selected_mass / error_mass if error_mass else 0.,
        macro_top1_recovery_pp=100 * selected_mass,
        mean_actual_regret_pct=100 * np.mean([r['regret'] for r in rows]) if rows else 0.,
        macro_regret_contribution_pp=100 * loss,
        share_of_total_regret_pct=100 * loss / loss_mass if loss_mass else 0.,
        share_of_error_regret_pct=100 * loss / error_loss if error_loss else 0.,
        recoverable_macro_regret_pp=100 * sum(
            (r['regret'] - r['native_winner_regret']) * r['macro_weight'] for r in rows))


def group_errors(records, keys):
    groups = defaultdict(list)
    for row in records:
        if row['category'] != 'correct':
            groups[tuple(row[key] for key in keys)].append(row)
    result = []
    for values, rows in groups.items():
        result.append(dict(zip(keys, values), **error_summary(rows, records, 'ALL')))
    return sorted(result, key=lambda r: (-r['macro_regret_contribution_pp'], -r['count']))


def correction_bounds(records, ranked, keys):
    baseline_top1 = 100 * sum(r['macro_weight'] for r in records if r['category'] == 'correct')
    baseline_regret = 100 * sum(r['macro_weight'] * r['regret'] for r in records)
    result = []
    for k in sorted(set([1, 3, 5, 10, len(ranked)])):
        if not 0 < k <= len(ranked):
            continue
        selected = {tuple(r[key] for key in keys) for r in ranked[:k]}
        fixed = [r for r in records if r['category'] != 'correct' and
                 tuple(r[key] for key in keys) in selected]
        summary = error_summary(fixed, records, 'ALL')
        result.append(dict(pair_scope='+'.join(keys), number_of_pairs=k,
            corrected_errors=len(fixed), baseline_macro_top1_pct=baseline_top1,
            diagnostic_macro_top1_pct=baseline_top1 + summary['macro_top1_recovery_pp'],
            baseline_macro_actual_regret_pct=baseline_regret,
            diagnostic_macro_actual_regret_pct=baseline_regret - summary['recoverable_macro_regret_pp'],
            recovered_regret_share_pct=summary['share_of_total_regret_pct']))
    return result


def run(output=ROOT):
    output.mkdir(parents=True, exist_ok=True)
    best = json.loads((BASELINE / 'best_eval.json').read_text())
    epoch = int(best['epoch'])
    directory = BASELINE / 'val_predictions' / f'epoch{epoch:03d}'
    records, manifest, metrics = [], {}, []
    for problem in PROBLEMS:
        raw = read_raw_costs(problem, 'val')
        path = directory / f'{problem}.npz'
        with np.load(path, allow_pickle=False) as saved:
            indices = saved['indices']
            if not np.array_equal(np.sort(indices), np.arange(len(raw['winner']))):
                raise ValueError(f'{problem}: incomplete or duplicate validation prediction indices')
            if list(saved['pool']) != raw['pool'] or list(saved['pool_ids']) != list(raw['pool_ids']):
                raise ValueError(f'{problem}: candidate name/ID order mismatch')
            costs, winner = raw['costs'][indices], raw['winner'][indices]
            if not np.array_equal(costs, saved['costs']) or not np.array_equal(winner, saved['winner']):
                raise ValueError(f'{problem}: saved predictions do not match original FP64 labels')
            order = ordered_predictions(saved['logits'], saved['pool_ids'])
            if not np.array_equal(order[:, 0], saved['pred']):
                raise ValueError(f'{problem}: greedy decision replay mismatch')
            rows = instance_records(problem, costs, winner, order, raw['pool'], len(PROBLEMS), indices)
        for row in rows:
            row['solver_a'], row['solver_b'] = sorted((row['selected'], row['winner']))
        records.extend(rows)
        ranks = np.array([r['winner_rank'] for r in rows])
        metrics.append(dict(problem=problem, instances=len(rows),
            top1_pct=100 * np.mean(ranks <= 1), top2_pct=100 * np.mean(ranks <= 2),
            top3_pct=100 * np.mean(ranks <= 3),
            mean_cost=np.mean([r['selected_cost'] for r in rows]),
            actual_regret_pct=100 * np.mean([r['regret'] for r in rows]),
            native_winner_above_cost_min_count=sum(r['native_winner_regret'] > 0 for r in rows),
            max_native_winner_regret_pct=100 * max(r['native_winner_regret'] for r in rows)))
        manifest[problem] = dict(predictions=str(path), predictions_sha256=file_hash(path),
            raw_label_sha256=raw['label_hash'], dataset_sha256=raw['data_hash'],
            pool=raw['pool'], pool_ids=list(map(int, raw['pool_ids'])), instances=len(rows))

    topk = []
    for scope in ['ALL'] + list(PROBLEMS):
        scoped = records if scope == 'ALL' else [r for r in records if r['problem'] == scope]
        for category in CATEGORIES:
            summary = error_summary([r for r in scoped if r['category'] == category], scoped, scope)
            topk.append(dict(category=category, **summary))
    directed = group_errors(records, ('problem', 'selected', 'winner'))
    paired = group_errors(records, ('problem', 'solver_a', 'solver_b'))
    global_pairs = group_errors(records, ('solver_a', 'solver_b'))
    bounds = correction_bounds(records, paired, ('problem', 'solver_a', 'solver_b'))
    bounds += correction_bounds(records, global_pairs, ('solver_a', 'solver_b'))
    for filename, data in [('error_budget_by_problem_pair.csv', directed),
                           ('ranked_problem_pairs.csv', paired), ('ranked_global_pairs.csv', global_pairs),
                           ('topk_error_decomposition.csv', topk), ('correction_bounds.csv', bounds),
                           ('baseline_by_problem.csv', metrics), ('validation_instance_errors.csv', records)]:
        write_csv(output / filename, data)
    observed = dict(macro_top1=np.mean([r['top1_pct'] for r in metrics]) / 100,
        macro_top2=np.mean([r['top2_pct'] for r in metrics]) / 100,
        macro_top3=np.mean([r['top3_pct'] for r in metrics]) / 100,
        macro_mean_cost=np.mean([r['mean_cost'] for r in metrics]),
        macro_actual_regret_pct=np.mean([r['actual_regret_pct'] for r in metrics]))
    for key, value in observed.items():
        if not np.isclose(value, best['val']['macro'][key], rtol=0, atol=1e-10):
            raise ValueError(f'Frozen validation metric replay differs: {key}')
    dump(output / 'frozen_baseline.json', dict(name='R45A', checkpoint=str(BASELINE / 'best.pt'),
        checkpoint_sha256=file_hash(BASELINE / 'best.pt'), human_epoch=epoch,
        original_selection_metric=best['val']['macro']['macro_vs_sbs_pct'],
        metrics=observed, data=manifest, split='val', test_read=False))
    dump(output / 'protocol.json', dict(seed=2, selector_training=False, test_read=False,
        baseline='R45A validation-best', cost_precision='original raw_label.pkl FP64',
        weighting='equal weight for each problem, equal weight for instances within each problem',
        error_label='native ind', rank_tie_break='ascending global solver ID',
        pair_correction='diagnostic only: change existing errors in the selected pairs to native winner',
        mechanism=dict(instances=1000, sampling='size-stratified, independent of errors and winner',
            precheck=8, sparse_support_max_starts=2, quantiles=[.10, .25, .50],
            retain_fraction=.8, retain_rounding='ceil', subset_repeats=20,
            subset_seed=2, subset_rule='identical retained customer-start IDs for both methods',
            winner_ties='exact native costs; ties reported separately, never called strict flips',
            sensitive='at least one strict reversal in 20 offline subsets'),
        paper='https://arxiv.org/html/2405.13947v1'))
    print(json.dumps(dict(metrics=observed, top_problem_pairs=paired[:10],
        top_global_pairs=global_pairs[:5]), indent=2), flush=True)
    return records, paired


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT)
    run(parser.parse_args().output)
