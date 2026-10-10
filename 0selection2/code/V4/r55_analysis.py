"""Replay R55 decisions and separately assess its supervised edge bottleneck."""

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
from .r55_experiment import ARM
from .r55_targets import ROOT


REFERENCES = dict(R53A=Path('code/V4/runs/R53_pair_specialist/A_ce_seed2'),
                  R54=Path('code/V4/runs/R54_route_supervision/route_supervised_seed2'))


def plot(root, history):
    fig, axes = plt.subplots(3, 3, figsize=(15, 11), constrained_layout=True)
    x = [r['successful_updates'] for r in history]
    trained = [r for r in history if r['pair_train'] is not None]
    tx = [r['successful_updates'] for r in trained]
    for ax, key, title, factor in ((axes[0, 0], 'accuracy', 'Two-method sign accuracy (%)', 100),
            (axes[0, 1], 'relative_cost_mse', 'Relative cost-difference MSE', 1),
            (axes[0, 2], 'pair_regret_pct', 'Two-method regret (%)', 1)):
        ax.plot(x, [r['pair_val']['macro'][key] * factor for r in history], label='Val')
        ax.plot(tx, [r['pair_train']['macro'][key] * factor for r in trained], '--', label='Train eval')
        ax.set_title(title)
    axes[1, 0].plot(x, [r['val']['groups']['ALL']['top1'] * 100 for r in history], label='R55 val')
    axes[1, 1].plot(x, [r['val']['groups']['ALL']['actual_regret_pct'] for r in history], label='R55 val')
    baseline = json.loads((root / 'baseline_val_metrics.json').read_text())['groups']['ALL']
    axes[1, 0].axhline(baseline['top1'] * 100, color='#777777', linestyle=':', label='R45A')
    axes[1, 1].axhline(baseline['actual_regret_pct'], color='#777777', linestyle=':', label='R45A')
    axes[1, 0].set_title('Integrated 18-task Top1 (%)')
    axes[1, 1].set_title('Integrated 18-task regret (%)')
    axes[1, 2].plot(x, [r['learning_rate'] for r in history], label='LR')
    axes[1, 2].set_title('Learning rate')
    axes[2, 0].plot(x, [r['structure_val']['macro']['edge_loss'] for r in history], label='Val diagnostic')
    axes[2, 0].plot(tx, [r['structure_train']['macro']['edge_loss'] for r in trained], '--', label='Train eval')
    axes[2, 0].plot(x, [r['structure_val']['macro']['zero_prediction_edge_loss'] for r in history], ':', label='All-zero edge reference')
    axes[2, 0].set_title('Group-balanced difference-edge MSE')
    for key in ('positive_edge_mse', 'negative_edge_mse', 'zero_edge_mse'):
        axes[2, 1].plot(x, [r['structure_val']['macro'][key] for r in history], label=key)
    axes[2, 1].set_title('Validation edge groups')
    for key in ('loss', 'edge_loss', 'relative_cost_loss'):
        axes[2, 2].plot(x, [r[key] for r in history], label=key)
    axes[2, 2].set_title('Training objectives')
    for axis in axes.flat:
        axis.set_xlabel('Successful optimizer updates')
        axis.grid(alpha=.2)
        axis.legend(fontsize=7)
    fig.savefig(root / 'learning_curves.png', dpi=160)
    plt.close(fig)


def analyze(root=ROOT):
    directory = root / ARM
    result = json.loads((directory / 'result.json').read_text())
    history = json.loads((directory / 'history.json').read_text())
    test = json.loads((directory / 'test_result.json').read_text())
    receipt = json.loads((root / 'target_receipt.json').read_text())
    initial = json.loads((directory / 'initial_structure_val.json').read_text())
    comparison, corrections, binary_rows, structural_rows, replay, screens = [], [], [], [], [], {}
    evaluated = {}
    for split in ('val', 'test'):
        raw, baseline = read_baseline(split, root)
        values = dict(R45A=baseline_metrics(raw, baseline))
        for name, path in REFERENCES.items():
            prior = json.loads((path / ('result.json' if split == 'val' else 'test_result.json')).read_text())
            values[name] = prior['best_validation'] if split == 'val' else prior['integrated']
        values['R55'] = result['best_validation'] if split == 'val' else test['integrated']
        evaluated[split] = values
        for name, metrics in values.items():
            for scope, r in dict(ALL=metrics['groups']['ALL'], MVRP=metrics['groups']['MVRP'], **metrics['per_problem']).items():
                comparison.append(dict(split=split, model=name, scope=scope, top1_pct=r['top1'] * 100,
                    top2_pct=r['top2'] * 100, top3_pct=r['top3'] * 100, mean_cost=r['mean_cost'],
                    vs_sbs_pct=r['vs_sbs_pct'], vs_oracle_pct=r['vs_oracle_pct'], actual_regret_pct=r['actual_regret_pct']))
        for p in PROBLEMS:
            with np.load(directory / 'predictions' / split / (p + '.npz'), allow_pickle=False) as saved:
                metrics, predicted = decision_metrics(ordinal_scores(saved['order']), raw[p])
                np.testing.assert_array_equal(predicted, saved['pred'])
                np.testing.assert_array_equal(raw[p]['costs'], saved['costs'])
                for key in ('top1', 'mean_cost', 'actual_regret_pct'):
                    np.testing.assert_allclose(metrics[key], values['R55']['per_problem'][p][key], atol=1e-10, rtol=0)
                corrections.extend(changes(saved, p, split, 'R55_vs_R45A'))
                for name, path in REFERENCES.items():
                    with np.load(path / 'predictions' / split / (p + '.npz'), allow_pickle=False) as older:
                        compared = dict(saved)
                        compared['baseline_pred'] = older['pred']
                        corrections.extend(changes(compared, p, split, 'R55_vs_' + name))
        base, current = values['R45A']['groups']['ALL'], values['R55']['groups']['ALL']
        delta = 100 * (current['top1'] - base['top1'])
        reduction = 100 * (base['actual_regret_pct'] - current['actual_regret_pct']) / base['actual_regret_pct']
        screens[split] = dict(top1_delta_pp=delta, regret_relative_reduction_pct=reduction,
            meets_predeclared_screen=bool((delta >= 5 and reduction >= 0) or (delta >= 0 and reduction >= 15)))
    for split, pair in (('train', result['best_pair_train']), ('val', result['best_pair_val']), ('test', test['pair'])):
        for scope, r in dict(MVRP=pair['macro'], **pair['per_problem']).items():
            binary_rows.append(dict(model='R55', split=split, scope=scope, **r))
    for split in ('train', 'val'):
        for scope, r in dict(MVRP=result['best_structure_' + split]['macro'], **result['best_structure_' + split]['per_problem']).items():
            structural_rows.append(dict(split=split, scope=scope, **r))
    for p in MVRP:
        with np.load(directory / 'bottleneck_predictions' / 'val' / (p + '.npz'), allow_pickle=False) as saved:
            pred, truth, valid = saved['edge_difference'], saved['target'].astype(np.float32), saved['edge_mask']
            reconstructed = (pred * saved['edge_distance'] * valid).sum(-1)
            np.testing.assert_allclose(reconstructed, saved['cost_difference'], atol=2e-5, rtol=2e-5)
            groups = [valid & test for test in (truth > 0, truth < 0, truth == 0)]
            counts = np.stack([g.sum(-1) for g in groups], -1)
            errors = np.stack([(((pred - truth) ** 2) * g).sum(-1) / np.maximum(count, 1)
                               for g, count in zip(groups, counts.T)], -1)
            active = counts > 0
            mse = ((errors * active).sum(-1) / np.maximum(active.sum(-1), 1)).mean()
            np.testing.assert_allclose(mse, result['best_structure_val']['per_problem'][p]['edge_loss'], atol=1e-6, rtol=0)
            replay.append(dict(problem=p, n=len(pred), edge_mse_replayed=float(mse),
                max_abs_analytic_replay_error=float(np.abs(reconstructed - saved['cost_difference']).max()),
                sign_disagreements=int(((reconstructed >= 0) != (saved['cost_difference'] >= 0)).sum())))
    write_csv(root / 'comparison.csv', comparison)
    write_csv(root / 'corrections_and_harms.csv', corrections)
    write_csv(root / 'pair_comparison.csv', binary_rows)
    write_csv(root / 'difference_edge_comparison.csv', structural_rows)
    write_csv(root / 'bottleneck_replay.csv', replay)
    write_csv(root / 'learning_curves.csv', [dict(epoch=r['epoch'], successful_updates=r['successful_updates'],
        loss=r['loss'], edge_loss=r['edge_loss'], relative_cost_loss=r['relative_cost_loss'],
        learning_rate=r['learning_rate'], val_pair_accuracy=r['pair_val']['macro']['accuracy'],
        val_relative_cost_mse=r['pair_val']['macro']['relative_cost_mse'],
        val_top1=r['val']['groups']['ALL']['top1'], val_regret=r['val']['groups']['ALL']['actual_regret_pct'],
        val_edge_loss=r['structure_val']['macro']['edge_loss']) for r in history])
    dump(root / 'screen.json', screens)
    plot(root, history)
    lines = ['# R55: Billed-edge cost-difference bottleneck', '',
        '## Design and Boundaries',
        'One scratch seed=2 model reuses the original four-layer, 128-dimensional instance Encoder. There is no graph-level binary head. A shared symmetric decoder predicts signed billed-count differences on every valid undirected edge, including depot edges. The only decision is the FP32 sum of predicted differences times input Euclidean distances; positive/zero selects MOEL, negative selects MTL.', '',
        'D = A_MTL - A_MOEL. Counts retain multiplicity; a closed single-customer route charges the same undirected depot edge twice. Open-route directed returns to the depot are removed before merging directions. Self/padding edges are excluded. No true route, cost, Oracle, winner or true edge mask enters forward.', '',
        'The objective is group-balanced edge MSE plus relative cost-difference MSE with the fixed 1% full-pool Oracle scale. Positive/negative/zero groups are averaged per instance and then equally combined among nonempty groups. There is no BCE, risk, successor CE, maximin, R-Drop, solver ID or post-sum learned classifier. Raw cost differences are not reported as calibrated classification logits.', '',
        '[Koh et al., Concept Bottleneck Models, ICML 2020, sections 2-3](https://proceedings.mlr.press/v119/koh20a.html) motivates requiring final predictions to pass through supervised intermediate predictions rather than attaching an auxiliary task. Our signed route-count concepts and fixed distance readout are project-specific, not a replication of their medical/bird experiments.', '',
        '## Data Conversion',
        f"Reused 300000 saved R54 training routes and 1920 fixed diagnostic validation routes. No new solver execution or annotation. Converted {sum(r['n'] for r in receipt['rows'])} instances. The largest historical cost-difference reconstruction error is {max(r['check']['max_abs_delta_historical_error'] for r in receipt['rows']):.9g}; the largest purely algebraic identity error is {max(r['check']['max_abs_algebra_identity_error'] for r in receipt['rows']):.9g}.",
        'Original cost/winner files were not modified. Historical cost tolerance remains ATOL=5e-5, RTOL=2e-6 per method; difference tolerance is the sum of the two cost tolerances. Independent geometry uses FP64 sums after native FP32 coordinate conversion. Target counts are stored losslessly as int8 in the upper triangle.', '',
        'Training uses the same 149976 non-tied queries (24 exact ties excluded), fresh shared Encoder initialization, complete shuffled coverage including tails, and interleaved task plan as R53/R54. Validation routes cover only a fixed 64-instance subset/task and are diagnostic-only; full18 validation selection uses original scalar costs and predicted signs. All normalizers are fit on training inputs only.', '',
        '## Integrated Results',
        '| Split | Model | Top1 % | Mean cost | Actual regret % |',
        '|---|---|---:|---:|---:|']
    for split, models in evaluated.items():
        for name, metrics in models.items():
            r = metrics['groups']['ALL']
            lines.append(f"| {split} | {name} | {r['top1']*100:.4f} | {r['mean_cost']:.6f} | {r['actual_regret_pct']:.6f} |")
    lines += ['', f"Best epoch {result['best_epoch']}; final epoch {result['final_epoch']}; {result['successful_updates']} successful updates. Best was the strict minimum integrated full18 validation actual regret. Last-five validation Top1 {result['last5_val']['top1']*100:.4f}%, regret {result['last5_val']['actual_regret_pct']:.6f}%.",
        'The fixed R45A Top2 trigger is unchanged. TSP/CVRP/ATSP decisions are untouched. Per-problem and MVRP macro results are in comparison.csv. All percentages preserve equal problem weights and native winner labels; raw costs are accumulated from original FP64 tables.', '',
        '## Bottleneck Generalization',
        f"Initial diagnostic validation edge loss: {initial['macro']['edge_loss']:.6f}; all-zero prediction reference: {initial['macro']['zero_prediction_edge_loss']:.6f}."]
    for split in ('train', 'val'):
        pair, structure = result['best_pair_' + split]['macro'], result['best_structure_' + split]['macro']
        lines.append(f"- {split}: pair sign accuracy {pair['accuracy']*100:.2f}%, pair regret {pair['pair_regret_pct']:.6f}%; edge MSE {structure['edge_loss']:.6f}, zero-edge reference {structure['zero_prediction_edge_loss']:.6f}; relative cost-difference MSE {pair['relative_cost_mse']:.6f}, zero-delta reference {pair['zero_delta_relative_mse']:.6f}.")
    lines += ['Saved diagnostic edge matrices reconstruct the final scalar outputs and grouped edge losses (bottleneck_replay.csv). Edge reconstruction alone is not evidence of better method choice. The predicted matrix is not constrained to a feasible route and is not a new solver.', '', '## Corrections and Harms']
    for split in ('val', 'test'):
        for reference in ('R45A', 'R53A', 'R54'):
            subset = [r for r in corrections if r['split'] == split and r['arm'] == 'R55_vs_' + reference]
            count = {k: sum(r['count'] for r in subset if r['category'] == k) for k in
                     ('corrected', 'harmed', 'both_wrong_cheaper', 'both_wrong_more_expensive')}
            net = sum(r['macro_regret_delta_contribution_pp'] for r in subset)
            improved = sum(evaluated[split]['R55']['per_problem'][p]['actual_regret_pct'] <
                           evaluated[split][reference]['per_problem'][p]['actual_regret_pct'] for p in MVRP)
            lines.append(f"- {split} vs {reference}: corrected {count['corrected']}, harmed {count['harmed']}; both-wrong cheaper {count['both_wrong_cheaper']}, more expensive {count['both_wrong_more_expensive']}; net macro regret change {net:+.6f} pp; {improved}/15 MVRP tasks have lower regret.")
    timing = test['benchmark']
    overhead = (timing['r55_encoder_and_edge_sum_ms'] / timing['r53a_encoder_and_binary_ms'] - 1) * 100
    lines += ['', '## Inference Cost',
        f"All 15000 MVRP test-instance expert forwards (including Encoder and all-edge readout): {test['specialist_15000_inference_seconds']:.3f} seconds. This includes ungated examples for reporting, not just deployed trigger calls.",
        f"Synchronized FP32 GPU benchmark, batch128/max nodes {timing['maximum_nodes']}: R53A Encoder+binary head {timing['r53a_encoder_and_binary_ms']:.3f} ms; R55 Encoder+edge-sum {timing['r55_encoder_and_edge_sum_ms']:.3f} ms ({overhead:+.1f}%); isolated edge decoder {timing['edge_decoder_only_ms']:.3f} ms.",
        'Warmup5/repeats30 on the same idle RTX3090. The common R45A ranking and CPU data loading are excluded, so these are expert/additional-selection costs, not complete route-solving or full selector system latency. No solver is executed at inference.', '',
        '## Conclusion']
    for split in ('val', 'test'):
        s = screens[split]
        lines.append(f"- {split} vs R45A: Top1 {s['top1_delta_pp']:+.4f} pp; regret relative reduction {s['regret_relative_reduction_pct']:+.3f}%; predeclared advancement screen passed: {s['meets_predeclared_screen']}.")
    if not all(s['meets_predeclared_screen'] for s in screens.values()):
        lines.append('This single-seed configuration did not achieve the predeclared selection improvement. Do not replace R45A based on intermediate edge metrics or small isolated differences; no additional weight, seed or architecture search was started.')
    else:
        lines.append('This configuration passed the prespecified screening target. Historical controls and one seed still limit causal and reproducibility claims.')
    lines += ['This tests the full supervised-difference bottleneck, not the hypothesis that labels are random or that selection accuracy has a hard ceiling. A good edge loss does not ensure that signed errors cancel accurately enough to recover small true cost differences.', '',
        '## Reproduction and Artifacts',
        '`bash code/V4/run_v4_r55.sh all` with the authorized single idle GPU binding. Source and exact Slurm/tmux launch are in R55_README.md and execution.json. Code, configs, checks, curves, per-instance predictions and reports are versioned; large target caches, initialization, best/last weights and offline W&B logs remain local.',
        'Validation-best SHA256 was locked before the one complete test inference. Saved full18 predictions reproduce all final metrics. Existing R45A/R53A/R54 checkpoints and results were preserved.']
    (root / 'RESULTS.md').write_text('\n'.join(lines) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    analyze(parser.parse_args().root)
