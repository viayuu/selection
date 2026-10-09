"""R45 reports: the same selection metrics, capacity control, and paired decisions."""

import argparse
import json
from pathlib import Path

import numpy as np

from code.unified_selector.registry import PROBLEMS
from .multitask_probe import dump
from .performance_analysis import write_csv
from .r43_analysis import metric_table, plot_methods, plot_run, plotting, rows_for
from .r43_experiment import policy_metrics
from .r45_experiment import ARMS, ROOT, RUN_NAMES, R43_ARGS


def decision_changes(root, before_group, after_group):
    rows = []
    for p in PROBLEMS:
        before_path = root / 'test_predictions' / before_group / f'{p}.npz'
        after_path = root / 'test_predictions' / after_group / f'{p}.npz'
        with np.load(before_path) as before, np.load(after_path) as after:
            for key in ('indices', 'winner', 'costs', 'pool_ids', 'nodes'):
                np.testing.assert_array_equal(before[key], after[key])
            winner, costs = before['winner'], before['costs']
            a, b = before['pred'], after['pred']
            idx = np.arange(len(a))
            delta = costs[idx, b] - costs[idx, a]
            both_wrong = (a != winner) & (b != winner) & (a != b)
            categories = dict(corrected=(a != winner) & (b == winner), harmed=(a == winner) & (b != winner),
                              both_wrong_cheaper=both_wrong & (delta < 0), both_wrong_dearer=both_wrong & (delta > 0),
                              both_wrong_equal=both_wrong & (delta == 0), all=np.ones(len(a), dtype=bool))
            for category, mask in categories.items():
                rows.append(dict(before=before_group, after=after_group, problem=p, category=category,
                                 n=int(mask.sum()), cost_delta_sum=float(delta[mask].sum()),
                                 mean_cost_delta=float(delta[mask].sum() / len(a)),
                                 actual_regret_delta_pct=float((delta[mask] / costs.min(-1)[mask]).sum() / len(a) * 100)))
    for category in categories:
        selected = [r for r in rows if r['category'] == category]
        rows.append(dict(before=before_group, after=after_group, problem='ALL', category=category,
                         n=sum(r['n'] for r in selected), cost_delta_sum=sum(r['cost_delta_sum'] for r in selected),
                         mean_cost_delta=float(np.mean([r['mean_cost_delta'] for r in selected])),
                         actual_regret_delta_pct=float(np.mean([r['actual_regret_delta_pct'] for r in selected]))))
    return rows


def replay(root, results):
    rows = []
    for group in ARMS:
        for p in PROBLEMS:
            with np.load(root / 'test_predictions' / group / f'{p}.npz') as saved:
                raw = {key: saved[key] for key in ('costs', 'winner', 'pool_ids', 'pool')}
                raw['pool_ids'], raw['pool'] = raw['pool_ids'].tolist(), raw['pool'].tolist()
                actual, pred = policy_metrics(saved['logits'], raw)
                np.testing.assert_array_equal(pred, saved['pred'])
                expected = results[group]['per_problem'][p]
                error = max(abs(actual[k] - expected[k]) for k in ('top1', 'top2', 'top3', 'mean_cost', 'actual_regret_pct'))
                if error > 1e-10:
                    raise ValueError('Saved R45 predictions do not replay the reported metrics')
                rows.append(dict(group=group, problem=p, maximum_metric_error=error))
    dump(root / 'prediction_replay.json', rows)


def size_metrics(root, protocol):
    rows = []
    for group in ARMS:
        for p in PROBLEMS:
            with np.load(root / 'test_predictions' / group / f'{p}.npz') as saved:
                buckets = np.digitize(saved['nodes'], protocol['size_boundaries'][p], right=True)
                for bucket in np.unique(buckets):
                    take = buckets == bucket
                    raw = dict(costs=saved['costs'][take], winner=saved['winner'][take],
                               pool=saved['pool'].tolist(), pool_ids=saved['pool_ids'].tolist())
                    result, _ = policy_metrics(saved['logits'][take], raw)
                    rows.append(dict(group=group, problem=p, bucket=int(bucket), n=int(take.sum()),
                                     **{key: result[key] for key in ('top1', 'ce', 'mean_cost', 'actual_regret_pct')}))
    return rows


def plot_comparison(root):
    plt = plotting()
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fields = ('top1', 'ce', 'mean_cost', 'actual_regret_pct', 'vs_sbs_pct', 'true3_accuracy')
    histories = {g: json.loads((root / name / 'history.json').read_text()) for g, name in RUN_NAMES.items()}
    common = min(len(history) for history in histories.values())
    common_rows = []
    for group, history in histories.items():
        for field, ax in zip(fields, axes.flat):
            for split, style in (('train', '--'), ('val', '-')):
                points = [r for r in history if r[split] is not None]
                ax.plot([r['epoch'] for r in points], [r[split]['macro']['macro_' + field] for r in points],
                        linestyle=style, label=f'{group} {split}')
            ax.set(xlabel='Epoch', title=field)
            ax.grid(alpha=.2)
            ax.legend(fontsize=8)
        for row in history[:common]:
            common_rows.append(dict(group=group, epoch=row['epoch'], **row['val']['macro']))
    write_csv(root / 'common_epoch_validation.csv', common_rows)
    fig.tight_layout()
    fig.savefig(root / 'curves/comparison.png', dpi=150)
    plt.close(fig)


def summarize(root=ROOT):
    root = Path(root)
    results = json.loads((root / 'test_results.json').read_text())
    protocol = json.loads((root / 'protocol.json').read_text())
    replay(root, results)
    rows = [row for group, value in results.items() for row in rows_for('R45' + group, value)]
    historical_root = R43_ARGS.parents[1]
    if json.loads((historical_root / 'test_data_manifest.json').read_text()) != json.loads((root / 'test_data_manifest.json').read_text()):
        raise ValueError('Historical R43A test uses a different split or candidate order')
    historical = json.loads((historical_root / 'test_results.json').read_text())['A']
    rows = rows_for('R43A historical', historical) + rows
    write_csv(root / 'comparison.csv', rows)
    family_rows = [dict(model='R45' + group, family=family, **value)
                   for group, result in results.items() for family, value in result['families'].items()]
    write_csv(root / 'family_comparison.csv', family_rows)
    write_csv(root / 'arm_distributions.csv', [dict(model='R45' + group, problem=p, solver=s,
              pick_rate=r['arm_distribution'][s], winner_rate=r['method_top1'][s],
              mean_cost=r['method_mean_cost'][s]) for group, value in results.items()
              for p, r in value['per_problem'].items() for s in r['pool']])
    changes = [row for before, after in (('A', 'B'), ('A', 'C'), ('C', 'B'))
               for row in decision_changes(root, before, after)]
    write_csv(root / 'decision_changes.csv', changes)
    write_csv(root / 'size_comparison.csv', size_metrics(root, protocol))
    selected = [r for r in rows if r['problem'] == 'ALL']
    lines = ['# R45 source-embedding comparison', '',
             'One seed=2, three from-scratch arms. B/C differ only in fixed semantic ownership at initialization.',
             'A/B are not parameter-count-matched. Identical scheduling/stopping rules do not imply identical realized LR trajectories.',
             'Test is read only after all validation-best checkpoints are locked. Percentages use per-problem macro means.', '']
    provenance = protocol.get('embedding_provenance', {})
    if provenance.get('uses_assumptions'):
        lines += ['## Source configuration limitation', '',
                  'B/C use explicitly assumed local defaults for missing solver parameters. Historical configurations are NOT fully verified.',
                  'These results test an approximate source representation, not a certified reconstruction of every label-generating solver deployment.',
                  f"Preserved unverified deployments: {len(provenance['provenance_warnings'])}.", '']
    lines += ['## Test', ''] + metric_table(selected)
    lines += ['', '## Capacity and validation selection', '',
              '| Arm | Parameters | Stop | Best | Best val Top1 | Final val Top1 | Last5 val Top1 |',
              '|---|---:|---:|---:|---:|---:|---:|']
    for group, name in RUN_NAMES.items():
        record = json.loads((root / name / 'result.json').read_text())
        best, final = record['best'], record['final']
        count = protocol['initialization_checks'][group]['trainable_parameters']
        lines.append(f'| {group}: {ARMS[group]} | {count} | {record["epochs_completed"]} | {best["epoch"]} | '
                     f'{best["val"]["macro"]["macro_top1"]:.4f} | {final["val"]["macro"]["macro_top1"]:.4f} | '
                     f'{record["last5_val"]["macro_top1"]:.4f} |')
        plot_run(root / name)
        plot_methods('R45' + group, results[group], root / 'curves' / group / 'per_problem')
    lines += ['', '## Final and last-five validation', '',
              '| Arm | Point | Top1 | CE | Mean cost | vs_SBS | Actual regret |',
              '|---|---|---:|---:|---:|---:|---:|']
    for group, name in RUN_NAMES.items():
        record = json.loads((root / name / 'result.json').read_text())
        for point, metrics in (('Final', record['final']['val']['macro']), ('Last5', record['last5_val'])):
            lines.append(f'| {group} | {point} | {metrics["macro_top1"]:.4f} | {metrics["macro_ce"]:.4f} | '
                         f'{metrics["macro_mean_cost"]:.6f} | {metrics["macro_vs_sbs_pct"]:+.4f}% | '
                         f'{metrics["macro_actual_regret_pct"]:.4f}% |')
    lines += ['', '## Test families', '',
              '| Arm | Family | Top1 | Mean cost | vs_SBS | Actual regret |',
              '|---|---|---:|---:|---:|---:|']
    for row in family_rows:
        lines.append(f'| {row["model"]} | {row["family"]} | {row["macro_top1"]:.4f} | '
                     f'{row["macro_mean_cost"]:.6f} | {row["macro_vs_sbs_pct"]:+.4f}% | '
                     f'{row["macro_actual_regret_pct"]:.4f}% |')
    lines += ['', '## Paired test decisions', '',
              '| Before | After | Change | Instances | Delta mean cost | Delta actual regret |',
              '|---|---|---|---:|---:|---:|---:|---:|']
    for row in changes:
        if row['problem'] == 'ALL':
            lines.append(f'| {row["before"]} | {row["after"]} | {row["category"]} | {row["n"]} | '
                         f'{row["mean_cost_delta"]:+.6f} | {row["actual_regret_delta_pct"]:+.5f}% |')
    lines += ['', '## Representation contrasts', '']
    for before, after in (('A', 'B'), ('C', 'B')):
        a, b = results[before]['macro'], results[after]['macro']
        lines.append(f'- {after} minus {before}: Top1 {(b["macro_top1"]-a["macro_top1"])*100:+.3f} percentage points; '
                     f'mean cost {b["macro_mean_cost"]-a["macro_mean_cost"]:+.6f}; '
                     f'actual regret {b["macro_actual_regret_pct"]-a["macro_actual_regret_pct"]:+.5f} percentage points.')
    lines += ['', '## Interpretation limits', '',
              'Use B versus A for practical effect; B versus C tests this one fixed source-to-solver mapping control.',
              'Only B beating both A and C with lower selection cost supports useful source-semantic information here.',
              'Top1 gains with worse cost are a trade-off, not an unqualified improvement.', '', '## Per problem', '']
    lines += metric_table([r for r in rows if r['problem'] != 'ALL'])
    (root / 'comparison.md').write_text('\n'.join(lines) + '\n')
    (root / 'curves').mkdir(exist_ok=True)
    plot_comparison(root)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=ROOT)
    summarize(parser.parse_args().root)
