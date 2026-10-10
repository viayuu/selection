"""Scenario progress, train-only references, and within-scenario result tables."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from ..unified_selector.registry import POOLS, PROBLEMS
from .performance_evaluation import decision_metrics
from .r58_labels import implementation_hashes, load_instances, size
from .r58_scenario import ROOT, save_json


def write_csv(path, rows):
    if not rows:
        return
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def progress(root):
    rows, estimate = [], 0.
    for problem in PROBLEMS:
        for method in POOLS[problem]:
            path = root / 'preflight' / f'{problem}__{method}.json'
            record = json.loads(path.read_text()) if path.exists() else {}
            current = record.get('implementation_sha256') == implementation_hashes()
            samples = record.get('results', [])
            batched = record.get('batch_probe', {})
            seconds = batched.get('seconds_per_instance', np.mean([r['seconds'] for r in samples]) if samples else 0.)
            qualified = bool(current and record.get('passed'))
            if qualified:
                estimate += float(seconds) * 12000
            rows.append(dict(problem=problem, solver=method, preflight_current=current,
                preflight_passed=qualified, seconds_per_instance=float(seconds) if qualified else None,
                error=record.get('error', 'not yet probed' if not record else '')))
    write_csv(root / 'deployment_preflight.csv', rows)
    release_path = root / 'scenario_v2/release.json'
    release = json.loads(release_path.read_text()) if release_path.exists() else {'ready': False}
    status = dict(qualified=sum(row['preflight_passed'] for row in rows), total=len(rows),
        complete_preflight=all(row['preflight_passed'] for row in rows), scenario_released=release['ready'],
        provisional_solve_hours=estimate / 3600,
        estimate_scope='current passing deployments only; no estimate for remaining ones; '
                       'point-sample timing, excluding deferred initialization when recorded',
        training_completed=(root / 'R45A_scenario_v2_seed2/result.json').exists(),
        test_completed=(root / 'test_results.json').exists())
    save_json(root / 'progress.json', status)
    if not status['test_completed']:
        text = [
            '# R58: Unified scenario_v2', '',
            '**Status: not complete. No new model result is claimed.**', '',
            f'- Qualified deployment preflights: {status["qualified"]}/{status["total"]}.',
            f'- Full independently checked scenario published: {status["scenario_released"]}.',
            f'- Full baseline training completed: {status["training_completed"]}.',
            f'- Locked checkpoint test completed: {status["test_completed"]}.', '',
            '## Locked semantic decisions',
            '- Classical backhaul, separate delivery/pickup capacity, delivery-before-pickup within each route.',
            '- Pure pickup routes are allowed. No implicit finite depot deadline.',
            '- L limits travel distance, not waiting/service. Open routes exclude depot return distance/time.',
            '- Fixed original weights with explicitly NEW adapted deployments, not historical reproduction.',
            '- Every saved route must pass a solver-independent simulation before a cost can be published.',
            '- All old instances, labels, results and solver source trees remain unchanged.', '',
            '## Completion gates',
            'All candidate preflights must pass before deployment lock; all54 original train/val/test datasets',
            'must have complete feasible cost vectors before training. Missing or invalid rows stop publication;',
            'they do not remove difficult instances or change per-instance candidate eligibility.', '',
            'See task_contract.json, deployment_preflight.csv, preflight/*.json, and pipeline_state.json.',
            'The provisional timing sum includes only probed deployments and is not a total-budget guarantee.', '',
        ]
        (root / 'RESULTS.md').write_text('\n'.join(text))
    return status


def read_performance(root, problem, split):
    with np.load(root / 'scenario_v2' / f'{problem}{split}' / 'performance.npz') as saved:
        return dict(costs=saved['costs'].copy(), winner=saved['winner'].copy(), pool=saved['pool'].tolist())


def fit_prior(costs, winner, scales):
    gap = (costs - costs.min(1, keepdims=True)) / costs.min(1, keepdims=True)
    bounds = np.unique(np.quantile(scales, np.linspace(0, 1, 9)[1:-1]))
    buckets = np.searchsorted(bounds, scales, side='right')
    global_gap = gap.mean(0)
    global_wins = (np.bincount(winner, minlength=costs.shape[1]) + 1.) / (len(winner) + costs.shape[1])
    records = []
    for bucket in range(len(bounds) + 1):
        selected = buckets == bucket
        n = int(selected.sum())
        local_gap = (gap[selected].sum(0) + 16 * global_gap) / (n + 16)
        local_wins = (np.bincount(winner[selected], minlength=costs.shape[1]) + 16 * global_wins) / (n + 16)
        records.append(dict(n=n, mean_gap=local_gap.tolist(), probabilities=local_wins.tolist()))
    return dict(size_bounds=bounds.tolist(), buckets=records, global_gap=global_gap.tolist(),
        train_fixed_sbs_slot=int(costs.mean(0).argmin()),
        rule='train-only size quantiles, at most8 bins; 16 global pseudocounts; global fallback if bin empty')


def baselines(root):
    references, fitted, rows = {}, {}, []
    for problem in PROBLEMS:
        raw = read_performance(root, problem, 'train')
        items, _ = load_instances(problem, 'train')
        scales = np.array([size(problem, item) for item in items])
        prior = fit_prior(raw['costs'], raw['winner'], scales)
        fitted[problem] = dict(prior, pool=raw['pool'])
        for split in ('val', 'test'):
            if split == 'test' and not (root / 'locked_checkpoint.json').exists():
                continue
            labels = read_performance(root, problem, split)
            items, _ = load_instances(problem, split)
            scales = np.array([size(problem, item) for item in items])
            buckets = np.searchsorted(prior['size_bounds'], scales, side='right')
            constant = np.zeros_like(labels['costs'])
            constant[:, prior['train_fixed_sbs_slot']] = 1
            cost_scores = -np.array([prior['buckets'][i]['mean_gap'] for i in buckets]) / .01
            win_scores = np.log(np.array([prior['buckets'][i]['probabilities'] for i in buckets]))
            for name, scores in (('train_fixed_SBS', constant), ('conditional_majority', win_scores),
                                 ('conditional_mean_gap', cost_scores)):
                metrics, pred = decision_metrics(scores, labels)
                references.setdefault(split, {}).setdefault(name, {})[problem] = metrics
                rows.append(dict(split=split, policy=name, problem=problem, n=metrics['n'],
                    top1=metrics['top1'], actual_regret_pct=metrics['actual_regret_pct'], mean_cost=metrics['mean_cost']))
    save_json(root / 'train_fitted_priors.json', fitted)
    save_json(root / 'reference_results.json', references)
    write_csv(root / 'reference_comparison.csv', rows)
    return references


def summarize(root):
    status = progress(root)
    if not status['test_completed']:
        return status
    references = baselines(root)
    run = root / 'R45A_scenario_v2_seed2'
    result = json.loads((run / 'result.json').read_text())
    test = json.loads((root / 'test_results.json').read_text())
    rows = []
    def append(split, policy, per):
        for problem, row in per.items():
            rows.append(dict(split=split, policy=policy, problem=problem, n=row['n'],
                top1=row['top1'], top2=row['top2'], top3=row['top3'], mean_cost=row['mean_cost'],
                vs_sbs_pct=row['vs_sbs_pct'], vs_oracle_pct=row['vs_oracle_pct'],
                actual_regret_pct=row['actual_regret_pct']))
        rows.append(dict(split=split, policy=policy, problem='ALL', n=sum(row['n'] for row in per.values()),
            **{key: float(np.mean([row[key] for row in per.values()])) for key in
               ('top1', 'top2', 'top3', 'mean_cost', 'vs_sbs_pct', 'vs_oracle_pct', 'actual_regret_pct')}))
    append('val', 'R58_full_baseline', result['best']['val']['per_problem'])
    append('test', 'R58_full_baseline', test['per_problem'])
    for split, policies in references.items():
        for name, per in policies.items():
            append(split, name, per)
    write_csv(root / 'comparison.csv', rows)
    text = ['# R58: Unified scenario_v2', '',
        'The full scenario release and one from-scratch full18-task baseline are complete.',
        'Old scenario scores are not a matched baseline and are not presented as model improvement.', '',
        '## Within-Scenario Results', '',
        '| Split | Policy | Top1 (%) | Actual regret (%) | Mean cost |',
        '|---|---|---:|---:|---:|']
    for row in rows:
        if row['problem'] == 'ALL':
            text.append(f'| {row["split"]} | {row["policy"]} | {100 * row["top1"]:.4f} | '
                        f'{row["actual_regret_pct"]:.6f} | {row["mean_cost"]:.6f} |')
    text += ['', '## Interpretation',
        '- The primary achievement is a common independently validated feasible domain, not a promised Top1 gain.',
        '- All references are fitted on the new training labels only; conditional priors use true size, never row-index metadata.',
        '- Reported SBS/Oracle ratios use the same evaluation pool; percentages are macro-averaged by problem.',
        '- Fixed baseline selection uses training mean raw cost; conditional cost selection uses mean relative gap.',
        '- Adapted decoders use original frozen weights; these are new deployments, not unmodified paper results.',
        '- Validation actual regret selects one checkpoint; test is evaluated once after lock.', '',
        'Task and deployment details: task_contract.json, deployments.lock.json, scenario_v2/release.json.', '']
    (root / 'RESULTS.md').write_text('\n'.join(text))
    plot_curves(root, result)
    return status


def plot_curves(root, result):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    history = json.loads((root / 'R45A_scenario_v2_seed2/history.json').read_text())
    figure, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    for axis, metric in zip(axes.flat, ('macro_top1', 'macro_ce', 'macro_actual_regret_pct', 'macro_mean_cost')):
        for split in ('train', 'val'):
            points = [row for row in history if row[split] is not None]
            axis.plot([row['successful_updates'] for row in points],
                      [row[split]['macro'][metric] for row in points], label=split)
        axis.set_xlabel('Successful optimizer updates')
        axis.set_ylabel(metric)
        axis.legend()
        axis.grid(alpha=.2)
    figure.savefig(root / 'learning_curves.png', dpi=160)
    plt.close(figure)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    summarize(parser.parse_args().root)
