"""R54 integrated results versus frozen R45A and historical R53A."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from code.unified_selector.registry import PROBLEMS
from .multitask_probe import dump
from .pair_specialist import ordinal_scores
from .performance_evaluation import decision_metrics
from .r48_common import write_csv
from .r53_analysis import changes
from .r53_data import MVRP, baseline_metrics, read_baseline
from .r54_experiment import ARM
from .r54_routes import ROOT


REFERENCE = Path('code/V4/runs/R53_pair_specialist/A_ce_seed2')


def plot(root, history):
    fig, axes = plt.subplots(3, 3, figsize=(15, 11), constrained_layout=True)
    x = [r['successful_updates'] for r in history]
    evaluated = [r for r in history if r['binary_train'] is not None]
    tx = [r['successful_updates'] for r in evaluated]
    for axis, key, title, factor in ((axes[0, 0], 'ce', 'Binary BCE', 1.),
            (axes[0, 1], 'accuracy', 'Binary accuracy (%)', 100.),
            (axes[0, 2], 'pair_regret_pct', 'Two-method regret (%)', 1.)):
        axis.plot(x, [r['binary_val']['macro'][key] * factor for r in history], label='Val')
        axis.plot(tx, [r['binary_train']['macro'][key] * factor for r in evaluated], '--', label='Train eval')
        axis.set_title(title)
    axes[1, 0].plot(x, [r['val']['groups']['ALL']['top1'] * 100 for r in history], label='R54 val')
    axes[1, 0].set_title('Integrated 18-task Top1 (%)')
    axes[1, 1].plot(x, [r['val']['groups']['ALL']['actual_regret_pct'] for r in history], label='R54 val')
    axes[1, 1].set_title('Integrated 18-task regret (%)')
    axes[1, 2].plot(x, [r['learning_rate'] for r in history], label='LR')
    axes[1, 2].set_title('Learning rate')
    for method in ('moel', 'mtl'):
        axes[2, 0].plot(x, [r['structure_val']['macro'][method + '_loss'] for r in history], label=method + ' val')
        axes[2, 0].plot(tx, [r['structure_train']['macro'][method + '_loss'] for r in evaluated], '--', label=method + ' train')
        axes[2, 1].plot(x, [r['structure_val']['macro'][method + '_accuracy'] * 100 for r in history], label=method + ' val')
        axes[2, 1].plot(tx, [r['structure_train']['macro'][method + '_accuracy'] * 100 for r in evaluated], '--', label=method + ' train')
    axes[2, 0].set_title('Normalized successor CE')
    axes[2, 1].set_title('Successor accuracy (%)')
    for key in ('loss', 'binary_loss', 'successor_loss'):
        axes[2, 2].plot(x, [r[key] for r in history], label=key)
    axes[2, 2].set_title('Training objective components')
    for axis in axes.flat:
        axis.set_xlabel('Successful optimizer updates')
        axis.grid(alpha=.2)
        axis.legend(fontsize=8)
    fig.savefig(root / 'learning_curves.png', dpi=160)
    plt.close(fig)


def analyze(root=ROOT):
    directory = root / ARM
    result = json.loads((directory / 'result.json').read_text())
    history = json.loads((directory / 'history.json').read_text())
    reference = json.loads((REFERENCE / 'result.json').read_text())
    annotation = json.loads((root / 'route_receipt.json').read_text())
    initial_structure = json.loads((directory / 'initial_structure_val.json').read_text())
    rows, correction_rows, binary_rows, structure_rows, screens = [], [], [], [], {}
    evaluated = {}
    for split in ('val', 'test'):
        raw, baseline = read_baseline(split, root)
        values = dict(R45A=baseline_metrics(raw, baseline),
            R53A=reference['best_validation'] if split == 'val' else json.loads((REFERENCE / 'test_result.json').read_text())['integrated'],
            R54=result['best_validation'] if split == 'val' else json.loads((directory / 'test_result.json').read_text())['integrated'])
        evaluated[split] = values
        for name, metrics in values.items():
            for scope, r in dict(ALL=metrics['groups']['ALL'], MVRP=metrics['groups']['MVRP'], **metrics['per_problem']).items():
                rows.append(dict(split=split, model=name, scope=scope, top1_pct=r['top1'] * 100, top2_pct=r['top2'] * 100,
                    top3_pct=r['top3'] * 100, mean_cost=r['mean_cost'], vs_sbs_pct=r['vs_sbs_pct'],
                    vs_oracle_pct=r['vs_oracle_pct'], actual_regret_pct=r['actual_regret_pct']))
        for p in PROBLEMS:
            with np.load(directory / 'predictions' / split / (p + '.npz'), allow_pickle=False) as saved:
                metrics, pred = decision_metrics(ordinal_scores(saved['order']), raw[p])
                np.testing.assert_array_equal(pred, saved['pred'])
                for key in ('top1', 'mean_cost', 'actual_regret_pct'):
                    np.testing.assert_allclose(metrics[key], values['R54']['per_problem'][p][key], atol=1e-10, rtol=0)
                correction_rows.extend(changes(saved, p, split, 'R54_vs_R45A'))
                with np.load(REFERENCE / 'predictions' / split / (p + '.npz'), allow_pickle=False) as older:
                    compared = dict(saved)
                    compared['baseline_pred'] = older['pred']
                    correction_rows.extend(changes(compared, p, split, 'R54_vs_R53A'))
        baseline_all, current = values['R45A']['groups']['ALL'], values['R54']['groups']['ALL']
        delta = (current['top1'] - baseline_all['top1']) * 100
        reduction = 100 * (baseline_all['actual_regret_pct'] - current['actual_regret_pct']) / baseline_all['actual_regret_pct']
        screens[split] = dict(top1_delta_pp=delta, regret_relative_reduction_pct=reduction,
            meets_predeclared_screen=bool((delta >= 5 and reduction >= 0) or (delta >= 0 and reduction >= 15)))
    for split, binary in (('train', result['best_binary_train']), ('val', result['best_binary_val']),
            ('test', json.loads((directory / 'test_result.json').read_text())['binary'])):
        for scope, r in dict(MVRP=binary['macro'], **binary['per_problem']).items():
            binary_rows.append(dict(model='R54', split=split, scope=scope, **r))
    for split in ('train', 'val'):
        for scope, metrics in dict(MVRP=result['best_structure_' + split]['macro'],
                                   **result['best_structure_' + split]['per_problem']).items():
            structure_rows.append(dict(split=split, scope=scope, **metrics))
    for p in MVRP:
        with np.load(directory / 'structure_predictions' / 'val' / (p + '.npz'), allow_pickle=False) as saved:
            valid = saved['customer_mask'][:, None]
            accuracy = ((saved['predicted_successor'] == saved['target_successor']) & valid).sum((0, 2)) / saved['customer_mask'].sum()
            expected = result['best_structure_val']['per_problem'][p]
            np.testing.assert_allclose(accuracy, [expected['moel_accuracy'], expected['mtl_accuracy']], rtol=0, atol=1e-7)
    write_csv(root / 'comparison.csv', rows)
    write_csv(root / 'corrections_and_harms.csv', correction_rows)
    write_csv(root / 'binary_comparison.csv', binary_rows)
    write_csv(root / 'successor_comparison.csv', structure_rows)
    write_csv(root / 'learning_curves.csv', [dict(epoch=r['epoch'], successful_updates=r['successful_updates'],
        loss=r['loss'], binary_loss=r['binary_loss'], successor_loss=r['successor_loss'], learning_rate=r['learning_rate'],
        val_binary_accuracy=r['binary_val']['macro']['accuracy'], val_binary_ce=r['binary_val']['macro']['ce'],
        val_top1=r['val']['groups']['ALL']['top1'], val_regret=r['val']['groups']['ALL']['actual_regret_pct'],
        val_moel_successor_accuracy=r['structure_val']['macro']['moel_accuracy'],
        val_mtl_successor_accuracy=r['structure_val']['macro']['mtl_accuracy']) for r in history])
    plot(root, history)
    dump(root / 'screen.json', screens)
    lines = ['# R54: Route-structure supervision for the pair specialist', '',
        '## Design',
        'The R53A four-layer, 128-dimensional instance encoder and unweighted BCE head are initialized from scratch. Two training-only directed successor heads supervise the native best MOEL and MTL routes. Every valid customer predicts another customer or END; self and padded destinations are excluded. Loss is BCE plus the mean of the two per-instance successor losses, divided by log(valid candidate count).', '',
        'Inference skips both auxiliary heads and executes no solver. Frozen R45A supplies the full ranking; only its original Top2={MOEL,MTL} decisions can change. Validation checkpoint selection remains integrated full18 actual regret, never route accuracy.', '',
        '[Joshi et al. (2019), section 4.1](https://arxiv.org/html/1906.01227v2) motivates solution-edge supervision. The directed VRP successor/END objective is our adaptation, not a reproduction of their undirected weighted edge classifier or TSP decoding.', '',
        '## Annotation Cost and Boundaries',
        f"Stored training routes: 300000 solver-instance pairs (150000 original instances), plus 1920 routes on a fixed 64-instance validation subset per task. Replay elapsed {annotation['total_seconds']:.1f} seconds. Historical costs were preserved and checked, not replaced.",
        'Existing R48 original training routes were reused after input/cost checks. R52 validation trajectories never entered training. Native argmax, augmentation=1, sample=1 and task-specific POMO start restrictions were retained; independent FP64 route-distance sums checked native costs. Native legal-action masks were checked during decoding. Missing historical CLI/source provenance remains as documented in route_plan.json.', '',
        'Only one seed and one new run were used. R53A is a historical control, with matching common initialization/sample order but a validation-driven schedule that can differ. No claim of a strict same-run auxiliary-loss ablation is made.', '',
        '## Integrated Results',
        '| Split | Model | Top1 % | Mean cost | Actual regret % |',
        '|---|---|---:|---:|---:|']
    for split, models in evaluated.items():
        for name, metrics in models.items():
            r = metrics['groups']['ALL']
            lines.append(f"| {split} | {name} | {r['top1']*100:.4f} | {r['mean_cost']:.6f} | {r['actual_regret_pct']:.6f} |")
    lines += ['', '## Structure Generalization',
        'Structure validation is measured on 960 untouched validation instances, not on POMO starts treated as independent examples. These routes are diagnostics only, not additional classifier inputs or checkpoint criteria.']
    init = initial_structure['macro']
    lines.append(f"Random-initialization validation successor accuracy: {init['moel_accuracy']*100:.2f}%/{init['mtl_accuracy']*100:.2f}% (MOEL/MTL). Saved best-checkpoint successor predictions reproduce the reported validation accuracies.")
    for split in ('train', 'val'):
        r = result['best_structure_' + split]['macro']
        lines.append(f"- {split}: MOEL/MTL successor accuracy {r['moel_accuracy']*100:.2f}%/{r['mtl_accuracy']*100:.2f}%; normalized CE {r['moel_loss']:.5f}/{r['mtl_loss']:.5f}.")
    lines += ['', '## Corrections and Harms']
    for split in ('val', 'test'):
        for ref in ('R45A', 'R53A'):
            subset = [r for r in correction_rows if r['split'] == split and r['arm'] == 'R54_vs_' + ref]
            counts = {k: sum(r['count'] for r in subset if r['category'] == k) for k in
                ('corrected', 'harmed', 'both_wrong_cheaper', 'both_wrong_more_expensive')}
            net = sum(r['macro_regret_delta_contribution_pp'] for r in subset)
            improved = sum(evaluated[split]['R54']['per_problem'][p]['actual_regret_pct'] <
                           evaluated[split][ref]['per_problem'][p]['actual_regret_pct'] for p in MVRP)
            lines.append(f"- {split} vs {ref}: corrected {counts['corrected']}, harmed {counts['harmed']}; both-wrong cheaper/more expensive {counts['both_wrong_cheaper']}/{counts['both_wrong_more_expensive']}; net regret change {net:+.6f} pp; regret improved on {improved}/15 MVRP tasks.")
    lines += ['', '## Conclusion',
        f"Best epoch {result['best_epoch']}; stopped at {result['final_epoch']}; successful updates {result['successful_updates']}. Last5 validation: Top1 {result['last5_val']['top1']*100:.4f}%, actual regret {result['last5_val']['actual_regret_pct']:.6f}%.",
        ('The prespecified screen is met on both validation and test; a strict matched follow-up is needed before attributing the gain to route supervision.'
         if all(s['meets_predeclared_screen'] for s in screens.values()) else
         'The prespecified system-level breakthrough screen is not met on both splits. Learning route connectivity alone has not demonstrated the required transferable solver-selection improvement. This does not establish a noise ceiling or prove that all structural supervision is ineffective.'), '',
        'Full task/family metrics, binary metrics and structural metrics are separate CSVs. Checkpoints, full route archives and W&B logs remain local; hashes and recipes are committed. Test used one locked validation-best checkpoint, without auxiliary heads or test-time solver execution.']
    (root / 'RESULTS.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps(screens, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    analyze(parser.parse_args().root)
