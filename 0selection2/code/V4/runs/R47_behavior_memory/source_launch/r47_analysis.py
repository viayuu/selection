"""Frozen-reference reporting for R47B; R47A is intentionally not trained."""

import argparse
import json
from pathlib import Path

import numpy as np

from .multitask_probe import dump
from .performance_analysis import write_csv
from .r43_analysis import metric_table, plot_methods, plot_run, plotting, rows_for
from .r43_experiment import policy_metrics
from .r47_experiment import ROOT, R45_ROOT, RUN_NAME


def decision_changes(root):
    rows = []
    for p in json.loads((root / 'test_results.json').read_text())['B']['per_problem']:
        with np.load(R45_ROOT / 'test_predictions/A' / f'{p}.npz') as old, np.load(root / 'test_predictions/B' / f'{p}.npz') as new:
            for key in ('indices', 'winner', 'costs', 'pool_ids', 'nodes'):
                np.testing.assert_array_equal(old[key], new[key])
            a, b, y, costs = old['pred'], new['pred'], old['winner'], old['costs']
            delta = costs[np.arange(len(a)), b] - costs[np.arange(len(a)), a]
            wrong = (a != y) & (b != y) & (a != b)
            sets = dict(corrected=(a != y) & (b == y), harmed=(a == y) & (b != y),
                        both_wrong_cheaper=wrong & (delta < 0), both_wrong_dearer=wrong & (delta > 0),
                        both_wrong_equal=wrong & (delta == 0), all=np.ones(len(a), dtype=bool))
            for name, mask in sets.items():
                rows.append(dict(problem=p, category=name, n=int(mask.sum()),
                    mean_cost_delta=float(delta[mask].sum()/len(a)),
                    actual_regret_delta_pct=float((delta[mask]/costs.min(-1)[mask]).sum()/len(a)*100)))
    for name in sets:
        selected = [r for r in rows if r['category'] == name]
        rows.append(dict(problem='ALL', category=name, n=sum(r['n'] for r in selected),
            mean_cost_delta=float(np.mean([r['mean_cost_delta'] for r in selected])),
            actual_regret_delta_pct=float(np.mean([r['actual_regret_delta_pct'] for r in selected]))))
    write_csv(root / 'decision_changes.csv', rows)
    return rows


def summarize(root=ROOT):
    root = Path(root)
    result = json.loads((root / 'test_results.json').read_text())['B']
    baseline = json.loads((R45_ROOT / 'test_results.json').read_text())['A']
    if json.loads((root / 'test_data_manifest.json').read_text()) != json.loads((R45_ROOT / 'test_data_manifest.json').read_text()):
        raise ValueError('Historical reference and R47 must retain identical test data and candidate identities')
    prior = json.loads((root / 'size_prior_test.json').read_text())
    results = dict(R45A_historical=baseline, R47B=result, size_prior=prior)
    rows = [row for name, value in results.items() for row in rows_for(name, value)]
    write_csv(root / 'comparison.csv', rows)
    family_rows = [dict(model=name, family=family, **metrics)
        for name, value in results.items() for family, metrics in value['families'].items()]
    write_csv(root / 'family_comparison.csv', family_rows)
    changes = decision_changes(root)
    record = json.loads((root / RUN_NAME / 'result.json').read_text())
    mechanism = json.loads((root / 'mechanism_check.json').read_text())
    check_rows = [dict(condition=name, **value['macro']) for name, value in
                  (('real', mechanism['real']), ('shuffled', mechanism['shuffled']))]
    write_csv(root / 'mechanism_check.csv', check_rows)
    replay, observations, distributions = [], [], []
    for p, expected in result['per_problem'].items():
        with np.load(root / 'test_predictions/B' / f'{p}.npz') as saved:
            raw = dict(costs=saved['costs'], winner=saved['winner'], pool_ids=saved['pool_ids'].tolist(), pool=saved['pool'].tolist())
            actual, pred = policy_metrics(saved['logits'], raw)
            np.testing.assert_array_equal(pred, saved['pred'])
            error = max(abs(actual[k]-expected[k]) for k in ('top1', 'top2', 'top3', 'ce', 'mean_cost', 'actual_regret_pct'))
            if error > 1e-10 or saved['costs'].dtype != np.float64:
                raise ValueError('Saved prediction replay or original cost precision differs')
            replay.append(dict(problem=p, instances=len(pred), maximum_metric_error=error,
                               memory_self_matches=expected['memory_self_matches']))
        for solver in expected['pool']:
            distributions.append(dict(problem=p, solver=solver, pick_rate=expected['arm_distribution'][solver],
                winner_rate=expected['method_top1'][solver], mean_cost=expected['method_mean_cost'][solver]))
        observations.append(dict(problem=p, selector_top1=expected['top1'], selector_top2=expected['top2'],
            selector_top3=expected['top3'], selector_mean_cost=expected['mean_cost'], sbs=expected['sbs_name'],
            sbs_top1=expected['method_top1'][expected['sbs_name']], sbs_mean_cost=expected['sbs_cost'],
            best_top1=expected['best_top1'], best_top2=expected['best_top2'], best_top3=expected['best_top3'],
            beat_top1=';'.join(s for s in expected['pool'] if expected['top1'] > expected['method_top1'][s]),
            not_beat_top1=';'.join(s for s in expected['pool'] if expected['top1'] <= expected['method_top1'][s]),
            beat_cost=';'.join(s for s in expected['pool'] if expected['mean_cost'] < expected['method_mean_cost'][s]),
            not_beat_cost=';'.join(s for s in expected['pool'] if expected['mean_cost'] >= expected['method_mean_cost'][s])))
    dump(root / 'prediction_replay.json', replay)
    write_csv(root / 'observations.csv', observations)
    write_csv(root / 'arm_distributions.csv', distributions)
    a, b = baseline['macro'], result['macro']
    top1_delta = (b['macro_top1']-a['macro_top1'])*100
    regret_relative = (b['macro_actual_regret_pct']/a['macro_actual_regret_pct']-1)*100
    screening = (top1_delta >= 3 and regret_relative <= 0) or (regret_relative <= -10 and top1_delta >= -1)
    real, shuffled = mechanism['real']['macro'], mechanism['shuffled']['macro']
    lines = ['# R47B: instance-conditioned performance memory', '',
        'Only B, seed2, was trained from scratch. R47A was not trained.',
        'R45A is a frozen historical reference, not a freshly matched R47A control.',
        'Train-only references; original FP64 costs; native winners; ALL is a per-problem macro average.', '',
        '## Test', ''] + metric_table([r for r in rows if r['problem'] == 'ALL'])
    lines += ['', f'R47B minus historical R45A: Top1 {top1_delta:+.4f} percentage points; '
              f'mean_cost {b["macro_mean_cost"]-a["macro_mean_cost"]:+.6f}; '
              f'actual regret relative change {regret_relative:+.3f}%.',
              f'Engineering screening target reached: {screening}. For the regret target, a Top1 drop greater than1pp is considered material.',
              'This is a single-seed direction screen, not a statistical or causal claim about retrieval.', '',
              '## Training', '', f'Stop epoch: {record["epochs_completed"]}; validation-best epoch: {record["best"]["epoch"]}; '
              f'successful updates: {record["successful_updates"]:,}.',
              '| Point | Top1 | CE | mean_cost | vs_SBS | actual regret |', '|---|---:|---:|---:|---:|---:|']
    for name, metrics in (('Best val', record['best']['val']['macro']), ('Final train', record['final']['train']['macro']),
                          ('Final val', record['final']['val']['macro']), ('Last5 val', record['last5_val'])):
        lines.append(f'| {name} | {metrics["macro_top1"]:.4f} | {metrics["macro_ce"]:.4f} | '
            f'{metrics["macro_mean_cost"]:.6f} | {metrics["macro_vs_sbs_pct"]:+.4f}% | {metrics["macro_actual_regret_pct"]:.4f}% |')
    lines += ['', '## Validation mechanism check', '',
              'The fixed best model is reevaluated without retraining. Whole behavior rows (gap vector and native winner) are permuted together within exact problem and node count. Keys stay unchanged.', '',
              '| Memory | Top1 | mean_cost | actual regret |', '|---|---:|---:|---:|',
              f'| Real | {real["macro_top1"]:.4f} | {real["macro_mean_cost"]:.6f} | {real["macro_actual_regret_pct"]:.4f}% |',
              f'| Shuffled | {shuffled["macro_top1"]:.4f} | {shuffled["macro_mean_cost"]:.6f} | {shuffled["macro_actual_regret_pct"]:.4f}% |',
              'Sensitivity shows reliance on the stored association, not by itself generalizable usefulness or a complete causal mechanism.', '',
              '## Families', '', '| Model | Family | Top1 | mean_cost | actual regret |', '|---|---|---:|---:|---:|']
    for row in family_rows:
        lines.append(f'| {row["model"]} | {row["family"]} | {row["macro_top1"]:.4f} | {row["macro_mean_cost"]:.6f} | {row["macro_actual_regret_pct"]:.4f}% |')
    lines += ['', '## Paired decisions versus historical R45A', '',
              '| Change | Instances | Contribution to mean_cost change | Contribution to regret change |', '|---|---:|---:|---:|']
    for row in changes:
        if row['problem'] == 'ALL':
            lines.append(f'| {row["category"]} | {row["n"]} | {row["mean_cost_delta"]:+.6f} | {row["actual_regret_delta_pct"]:+.5f}pp |')
    lines += ['', '## Implementation boundary', '',
        'The reference graph summaries are detached epoch snapshots, refreshed in eval mode at epoch start and before every formal evaluation. Query Encoder, shared key map on both sides, behavior value/correction, and the main selector are jointly trainable.',
        'Every train/train_eval query excludes its base-instance group; val/test only read train memory and also exclude input duplicates. Identity grouping covers coordinate permutations/D4 copies to1e-6 and identical ordered matrices; no external lineage is inferred.',
        'No source embeddings, label regeneration, extra seed, new loss, candidate cropping, or test-driven checkpoint selection.',
        'The average-gap reference is a train-only problem/size-quartile prior, not the R47 decision rule.', '',
        '## Per problem', ''] + metric_table([r for r in rows if r['problem'] != 'ALL'])
    (root / 'comparison.md').write_text('\n'.join(lines)+'\n')
    plot_run(root / RUN_NAME)
    plot_methods('R47B', result, root / 'curves/test_per_problem')
    plot_methods('R47B val-best', record['best']['val'], root / 'curves/val_best_per_problem')
    history = json.loads((root / RUN_NAME / 'history.json').read_text())
    plt = plotting()
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    baseline_val = json.loads((R45_ROOT / 'A_handcrafted_seed2/best_eval.json').read_text())['val']['macro']
    for ax, key in zip(axes, ('top1', 'mean_cost', 'actual_regret_pct')):
        for split, style in (('train', '--'), ('val', '-')):
            points = [r for r in history if r[split] is not None]
            ax.plot([r['epoch'] for r in points], [r[split]['macro']['macro_'+key] for r in points], style, label='R47B '+split)
        ax.axhline(baseline_val['macro_'+key], linestyle=':', label='Historical R45A best val')
        ax.set(title=key, xlabel='Epoch')
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    fig.tight_layout()
    (root / 'curves').mkdir(exist_ok=True)
    fig.savefig(root / 'curves/comparison.png', dpi=150)
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=ROOT)
    summarize(parser.parse_args().root)
