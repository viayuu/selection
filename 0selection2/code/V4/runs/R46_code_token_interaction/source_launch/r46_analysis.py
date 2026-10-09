"""R46A versus the already frozen R45A/R45B; no R46B claims or extra training."""

import argparse
import json
from pathlib import Path

import numpy as np

from code.unified_selector.registry import PROBLEMS
from .multitask_probe import dump
from .performance_analysis import write_csv
from .r43_analysis import metric_table, plot_methods, plot_run, plotting, rows_for
from .r43_experiment import policy_metrics
from .r46_experiment import ROOT, R45_ROOT, RUN_NAME


def decision_changes(root, baseline):
    rows = []
    for p in PROBLEMS:
        with np.load(R45_ROOT / 'test_predictions' / baseline / f'{p}.npz') as before, np.load(root / 'test_predictions/A' / f'{p}.npz') as after:
            for key in ('indices', 'winner', 'costs', 'pool_ids', 'nodes'):
                np.testing.assert_array_equal(before[key], after[key])
            a, b, winner, costs = before['pred'], after['pred'], before['winner'], before['costs']
            delta = costs[np.arange(len(a)), b] - costs[np.arange(len(a)), a]
            wrong = (a != winner) & (b != winner) & (a != b)
            categories = dict(corrected=(a != winner) & (b == winner), harmed=(a == winner) & (b != winner),
                              both_wrong_cheaper=wrong & (delta < 0), both_wrong_dearer=wrong & (delta > 0),
                              both_wrong_equal=wrong & (delta == 0), all=np.ones(len(a), dtype=bool))
            for name, mask in categories.items():
                rows.append(dict(before='R45'+baseline, after='R46A', problem=p, category=name, n=int(mask.sum()),
                                 mean_cost_delta=float(delta[mask].sum()/len(a)),
                                 actual_regret_delta_pct=float((delta[mask]/costs.min(-1)[mask]).sum()/len(a)*100)))
    for name in categories:
        selected = [r for r in rows if r['category'] == name]
        rows.append(dict(before='R45'+baseline, after='R46A', problem='ALL', category=name,
                         n=sum(r['n'] for r in selected), mean_cost_delta=float(np.mean([r['mean_cost_delta'] for r in selected])),
                         actual_regret_delta_pct=float(np.mean([r['actual_regret_delta_pct'] for r in selected]))))
    return rows


def summarize(root=ROOT):
    root = Path(root)
    result = json.loads((root / 'test_results.json').read_text())['A']
    protocol = json.loads((root / 'protocol.json').read_text())
    saved = json.loads((R45_ROOT / 'test_results.json').read_text())
    if json.loads((root / 'test_data_manifest.json').read_text()) != json.loads((R45_ROOT / 'test_data_manifest.json').read_text()):
        raise ValueError('R45 and R46 must use the same held-out data and candidate order')
    results = dict(R45A=saved['A'], R45B=saved['B'], R46A=result)
    rows = [r for name, value in results.items() for r in rows_for(name, value)]
    write_csv(root / 'comparison.csv', rows)
    family_rows = [dict(model=name, family=family, **scores)
                   for name, value in results.items() for family, scores in value['families'].items()]
    write_csv(root / 'family_comparison.csv', family_rows)
    replay, observations, distributions = [], [], []
    for p, expected in result['per_problem'].items():
        with np.load(root / 'test_predictions/A' / f'{p}.npz') as predictions:
            raw = {k: predictions[k] for k in ('costs', 'winner', 'pool_ids', 'pool')}
            raw.update(pool_ids=raw['pool_ids'].tolist(), pool=raw['pool'].tolist())
            actual, pred = policy_metrics(predictions['logits'], raw)
            np.testing.assert_array_equal(pred, predictions['pred'])
            error = max(abs(actual[k]-expected[k]) for k in ('top1', 'top2', 'top3', 'ce', 'mean_cost', 'actual_regret_pct'))
            if error > 1e-10 or predictions['costs'].dtype != np.float64:
                raise ValueError('Prediction replay or original cost precision differs')
            replay.append(dict(problem=p, instances=len(pred), maximum_metric_error=error))
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
    changes = decision_changes(root, 'A') + decision_changes(root, 'B')
    write_csv(root / 'decision_changes.csv', changes)
    record = json.loads((root / RUN_NAME / 'result.json').read_text())
    lines = ['# R46A: source views interact before aggregation', '',
             'Only R46A was trained, seed=2, from scratch on all18 tasks. R46B is NOT run.',
             'R45A/B are frozen historical references. ALL percentages are per-problem macro averages.',
             'Original raw-label FP64 costs and native winners; no API calls, new embeddings, test-driven selection or augmentation.', '',
             '## Model', '',
             'Four-layer instance Encoder -> two synchronous node/source-token layers -> instance-conditioned per-solver source pooling.',
             'No trainable solver ID, solver-specific parameter or handcrafted solver branch. Original shared decoder, score head, maximin and R43A objective are retained.',
             f"Source views/deployment slots: up to {protocol['source_package_audit']['global_slots']} per solver; no pre-interaction averaging.",
             'Legacy role summaries are saved only for embedding provenance validation; forward reads the complete-view component bank.', '',
             '## Fixed-input limitations', '',
             'Identical full source packages: ' + '; '.join(' / '.join(v) for v in protocol['source_package_audit']['identical_packages']) + '.',
             'Without an ID these methods cannot be distinguished by the model. They remain separate candidates; exact score ties use ascending global solver ID.',
             'Previously mixed content inside one external vector cannot be recovered. The original26 assumed historical deployment records remain unverified.',
             'Without R46B, this run cannot establish that correct source-to-solver correspondence is better than a fixed mismapping.', '',
             '## Test', '']
    lines += metric_table([r for r in rows if r['problem'] == 'ALL'])
    lines += ['', '## Training and validation', '',
              f"Parameters: {protocol['trainable_parameters']:,}. Stop epoch: {record['epochs_completed']}; validation-best epoch: {record['best']['epoch']}.",
              f"Successful updates: {record['successful_updates']:,}; {record['updates_by_problem']['TSP']} per task; full coverage, retained tail batches.",
              '| Point | Top1 | CE | mean_cost | vs_SBS | actual regret |', '|---|---:|---:|---:|---:|---:|']
    for name, metrics in (('Best val', record['best']['val']['macro']), ('Final train', record['final']['train']['macro']),
                          ('Final val', record['final']['val']['macro']), ('Last5 val', record['last5_val'])):
        lines.append(f'| {name} | {metrics["macro_top1"]:.4f} | {metrics["macro_ce"]:.4f} | '
                     f'{metrics["macro_mean_cost"]:.6f} | {metrics["macro_vs_sbs_pct"]:+.4f}% | {metrics["macro_actual_regret_pct"]:.4f}% |')
    lines += ['', '## Test families', '', '| Model | Family | Top1 | mean_cost | vs_SBS | actual regret |', '|---|---|---:|---:|---:|---:|']
    for row in family_rows:
        lines.append(f'| {row["model"]} | {row["family"]} | {row["macro_top1"]:.4f} | {row["macro_mean_cost"]:.6f} | '
                     f'{row["macro_vs_sbs_pct"]:+.4f}% | {row["macro_actual_regret_pct"]:.4f}% |')
    lines += ['', '## Contrasts', '']
    current = result['macro']
    for name in ('R45A', 'R45B'):
        before = results[name]['macro']
        lines.append(f'- R46A minus {name}: Top1 {(current["macro_top1"]-before["macro_top1"])*100:+.3f} percentage points; '
                     f'mean cost {current["macro_mean_cost"]-before["macro_mean_cost"]:+.6f}; '
                     f'actual regret {current["macro_actual_regret_pct"]-before["macro_actual_regret_pct"]:+.5f} percentage points.')
    lines += ['', '## Decision changes', '', '| Reference | Change | Instances | Delta mean cost | Delta regret |', '|---|---|---:|---:|---:|']
    for row in changes:
        if row['problem'] == 'ALL':
            lines.append(f'| {row["before"]} | {row["category"]} | {row["n"]} | {row["mean_cost_delta"]:+.6f} | '
                         f'{row["actual_regret_delta_pct"]:+.5f}% |')
    lines += ['', '## Per problem', ''] + metric_table([r for r in rows if r['problem'] != 'ALL'])
    (root / 'comparison.md').write_text('\n'.join(lines)+'\n')
    plot_run(root / RUN_NAME)
    plot_methods('R46A', result, root / 'curves/test_per_problem')
    plot_methods('R46A val-best', record['best']['val'], root / 'curves/val_best_per_problem')
    plt = plotting()
    history = json.loads((root / RUN_NAME / 'history.json').read_text())
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, key in zip(axes, ('top1', 'mean_cost', 'actual_regret_pct')):
        for split, style in (('train', '--'), ('val', '-')):
            points = [r for r in history if r[split] is not None]
            ax.plot([r['epoch'] for r in points], [r[split]['macro']['macro_'+key] for r in points], style, label='R46A '+split)
        for name, directory in (('R45A', 'A_handcrafted_seed2'), ('R45B', 'B_code_seed2')):
            metrics = json.loads((R45_ROOT / directory / 'best_eval.json').read_text())['val']['macro']
            ax.axhline(metrics['macro_'+key], linestyle=':', label=name+' best val')
        ax.set(title=key, xlabel='Epoch')
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(root / 'curves/comparison.png', dpi=150)
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=ROOT)
    summarize(parser.parse_args().root)
