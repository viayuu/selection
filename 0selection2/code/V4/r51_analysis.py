"""R51 development-locked metrics, learning curves and total probe costs."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from .multitask_probe import dump, file_hash
from .r48_common import binary_metrics, write_csv
from .r49_analysis import bootstrap_difference, instance_metrics
from .r51_experiment import ROOT, R50, MODELS, SETS


def analyze(root):
    config = json.loads((root / 'config.json').read_text())
    timing = json.loads((root / 'inference_timing.json').read_text())
    selections = {name: json.loads((root / name / 'selection.json').read_text()) for name in MODELS}
    histories = {name: json.loads((root / name / 'history.json').read_text()) for name in MODELS}
    names = ('R50B_static_reference',) + MODELS
    archives = {name: {split: dict(np.load((R50 / 'B_pretrained_seed2' if name == names[0] else root / name)
                                    / (split + '_predictions.npz'))) for split in SETS} for name in names}
    selected_steps = {names[0]: json.loads((R50 / 'B_pretrained_seed2/selection.json').read_text())['best_updates']}
    selected_steps.update({name: selections[name]['best_updates'] for name in MODELS})
    plan = np.load(root / 'sampling_plan.npz')['indices']
    np.testing.assert_array_equal(plan, np.load(R50 / 'sampling_plan.npz')['indices'])
    unique, counts = np.unique(plan, return_counts=True)
    if plan.shape != (4000, 128) or len(unique) != 8000 or not (counts == 64).all():
        raise ValueError('Training data coverage differs from R50')
    results, checks = [], []
    for name in names:
        for split in SETS:
            reference, archive = archives[names[0]][split], archives[name][split]
            for field in ('indices', 'labels', 'costs', 'customers'):
                np.testing.assert_array_equal(archive[field], reference[field])
            metrics = binary_metrics(archive['scores'], archive['costs'])
            if name in MODELS:
                history = histories[name]
                if [r['updates'] for r in history] != list(range(0, 4001, 200)):
                    raise ValueError('Missing full-set evaluation point')
                chosen = min(history, key=lambda row: row['metrics']['development']['ce'])
                if chosen['updates'] != selected_steps[name]:
                    raise ValueError('Selection differs from development CE rule')
                error = max(abs(float(metrics[k]) - float(chosen['metrics'][split][k])) for k in metrics)
                if error > 1e-10:
                    raise ValueError('Saved predictions do not reproduce selected metrics')
                checks.append(dict(model=name, split=split, replay_error=error))
            results.append(dict(model=name, split=split, best_updates=selected_steps[name], **metrics))
    write_csv(root / 'comparison.csv', results)
    replication_error = max(float(np.abs(archives[MODELS[0]][split]['scores']
        - archives[names[0]][split]['scores']).max()) for split in SETS)
    indexed = {(r['model'], r['split']): r for r in results}
    changes, uncertainty = [], []
    for split, filename in (('internal', 'internal_predictions.csv'), ('original_val', 'val_predictions.csv')):
        reference = archives[names[0]][split]
        rows = []
        for i, index in enumerate(reference['indices']):
            row = dict(source_split='train' if split == 'internal' else 'val', original_index=int(index),
                customers=int(reference['customers'][i]), label=int(reference['labels'][i]),
                MOEL_cost=float(reference['costs'][i, 0]), MTL_cost=float(reference['costs'][i, 1]))
            for name in names:
                scores = archives[name][split]['scores'][i].astype(np.float64)
                p = np.exp(scores - scores.max())
                row.update({name + '_prediction': int(scores.argmax()), name + '_score0': float(scores[0]),
                            name + '_score1': float(scores[1]), name + '_probability1': float(p[1] / p.sum())})
            rows.append(row)
        write_csv(root / filename, rows)
        new, old = archives[MODELS[1]][split], archives[MODELS[0]][split]
        truth = new['labels']
        pred, original = new['scores'].argmax(1), old['scores'].argmax(1)
        costs = new['costs']
        delta = costs[np.arange(len(truth)), pred] - costs[np.arange(len(truth)), original]
        changes.append(dict(comparison='B_behavior minus A_static', split=split,
            corrected=int(((original != truth) & (pred == truth)).sum()),
            harmed=int(((original == truth) & (pred != truth)).sum()), changed=int((pred != original).sum()),
            mean_cost_delta=float(delta.mean()), pair_regret_delta_pp=float((delta / costs.min(1) * 100).mean())))
        for value in bootstrap_difference(instance_metrics(new), instance_metrics(old), True, 5102):
            uncertainty.append(dict(comparison='B_behavior minus A_static', split=split, **value))
    write_csv(root / 'decision_changes.csv', changes)
    write_csv(root / 'paired_uncertainty.csv', uncertainty)
    screen = {}
    for split in ('internal', 'original_val'):
        a, b = [indexed[(name, split)] for name in MODELS]
        screen[split] = dict(accuracy_gain_pp=(b['accuracy'] - a['accuracy']) * 100,
            regret_delta_pp=b['pair_regret_pct'] - a['pair_regret_pct'],
            regret_relative_reduction_pct=(a['pair_regret_pct'] - b['pair_regret_pct']) / a['pair_regret_pct'] * 100,
            meets_predictive_screen=b['accuracy'] >= a['accuracy'] + .05 and b['pair_regret_pct'] < a['pair_regret_pct'])
    predictive_pass = all(v['meets_predictive_screen'] for v in screen.values())
    passed = predictive_pass and timing['overhead_acceptable']
    dump(root / 'advancement_screen.json', dict(passed=passed, predictive_pass=predictive_pass,
        by_set=screen, overhead_acceptable=timing['overhead_acceptable'],
        total_time_ratio=timing['ratio_behavior_to_static'], max_time_ratio=1.25,
        artificial_engineering_screen=True, no_parameter_search_or_extra_seed=True))
    dump(root / 'execution_checks.json', dict(prediction_replay=checks,
        same_r50_split_and_batch_sequence=True, updates_each=4000, presentations_each=512000,
        unique_training_rows=8000, presentations_per_row=64, common_initialization_equal=True,
        solver_only_inputs='Original native depot/xy/demand/service/TW fields, no costs or winner',
        labels='Original raw_label FP64 costs unchanged', test_read=False,
        static_control_vs_r50_pretrained_max_logit_error=replication_error,
        post_training_code_changes='Only timing: native continuation skips unused post-probe diagnostic tracking; feature caches, models, training, best selections and predictions unchanged',
        source_hashes={name: file_hash(Path(__file__).parent / name) for name in config['sources']}))
    labels = dict(train='Train 8000', development='Development (selection)', internal='Internal holdout', original_val='Original validation')
    colors = ('#2879b9', '#d99b22', '#21836c', '#ce4e4b')
    fig, axes = plt.subplots(2, 3, figsize=(16, 8), sharex=True)
    for i, name in enumerate(MODELS):
        history = histories[name]
        for j, (field, title, scale) in enumerate((('ce', 'Unweighted CE', 1),
                ('accuracy', 'Binary accuracy (%)', 100), ('pair_regret_pct', 'Two-method regret (%)', 1))):
            ax = axes[i, j]
            for (split, label), color in zip(labels.items(), colors):
                ax.plot([r['updates'] for r in history], [r['metrics'][split][field] * scale for r in history],
                        label=label, color=color, lw=1.6)
            ax.axvline(selected_steps[name], color='#444444', ls='--', lw=.8, label='Best development CE')
            ax.set_title(name + ': ' + title, fontsize=11)
            ax.set_xlabel('Successful readout optimizer updates')
            ax.grid(alpha=.2)
            ax.legend(fontsize=8)
    fig.suptitle('R51: frozen pretrained static nodes vs native ten-action behavior', fontsize=13)
    fig.tight_layout()
    fig.savefig(root / 'learning_curves.png', dpi=180)
    plt.close(fig)
    lines = ['# R51: cost-limited native solver behavior probes', '',
        '## Main result', '',
        f'Predeclared continuation screen passed: **{passed}**; predictive screen: **{predictive_pass}**. '
        'Binary prediction is MOEL vs MTL only, not seven-method-pool Top1.', '',
        '| Model | Selected update | Train accuracy | Development CE | Internal accuracy / regret | Original validation accuracy / regret |',
        '|---|---:|---:|---:|---:|---:|']
    for name in names:
        train, dev, internal, val = [indexed[(name, split)] for split in SETS]
        lines.append(f'| {name} | {selected_steps[name]} | {train["accuracy"]*100:.2f}% | {dev["ce"]:.6f} | '
            f'{internal["accuracy"]*100:.2f}% / {internal["pair_regret_pct"]:.6f}% | '
            f'{val["accuracy"]*100:.2f}% / {val["pair_regret_pct"]:.6f}% |')
    for split, value in screen.items():
        lines.append(f'- {split}, B minus A: accuracy {value["accuracy_gain_pp"]:+.2f}pp; '
            f'regret {value["regret_delta_pp"]:+.6f}pp; relative regret reduction {value["regret_relative_reduction_pct"]:+.2f}%.')
    if predictive_pass:
        lines += ['', 'The fixed ten-action behavior features meet the predictive screen on both held-out sets. '
            'This is single-seed, two-method OVRPTW evidence, not an18-task result or an independent multi-seed confirmation.']
    else:
        lines += ['', 'This fixed behavior/readout configuration does not meet the predictive screen on both held-out sets. '
            'It does not establish random labels, an accuracy ceiling, or that all trajectory representations are ineffective. '
            'No probe-length, seed, pooling or classifier search was added.']
    lines += ['', '## Locked protocol', '',
        'Exact R49/R50 splits and4000x128 batch plan; all8000 training rows appear64 times. '
        'All21 evaluation points u0,200,...,4000 evaluate complete train/development/internal/original validation sets with eval() and tails retained. '
        'Strict minimum development CE alone locks one checkpoint per model; internal/original validation never select weights. No test read.',
        'Both full original R48-pinned epoch5000 Train_ALL solvers are frozen/eval, FP32/TF32-off, greedy, no augmentation. '
        'Original FP64 full-solve cost labels are not changed. R50 pretrained static node caches are reused with SHA checks. '
        'The native six-field input is used, not selector8D input. Neither probing nor readout receives current cost/winner.',
        'Both groups train fresh readouts with explicit identical common parameter values, seed2 and paired sample order. '
        'A is the exact R50 pretrained static readout architecture; B adds270->64->64 behavior MLP and graph-level concatenation. '
        'The shared385 columns of B first head layer and all other common layers copy A initialization; B extra columns are normally initialized. '
        'A/B are not parameter-count matched. Dropout call streams differ because B has an extra branch.',
        f'Fresh static A versus historical R50 pretrained B: maximum saved-logit difference across all four sets is {replication_error:.8g}.',
        f'Parameters: A {config["parameter_counts"][MODELS[0]]}; B {config["parameter_counts"][MODELS[1]]}. '
        'Fresh AdamW LR1e-3 WD1e-4, dropout0.1, gradient clip1, CE only;4000 updates, no scheduler/early stop.', '',
        '## Behavior evidence', '',
        'Two forced native moves (depot, each customer POMO start), then exactly10 network-selected greedy moves per solver. '
        'All customer starting nodes remain; no trajectory is completed or selected by cost while generating features. '
        'At steps1/5/10, each of9 fields is summarized across all POMO starts by mean/population std/q10/q50/q90:270 dimensions. '
        'Progress, pressure and accumulated cost are measured after the action; entropy/margin use that action\'s original pre-move distribution.',
        'Features: served customer/demand fractions, depot returns, effective open distance, waiting, remaining capacity, '
        'feasible unserved fraction, normalized legal-action entropy and top1-top2 probability margin. '
        'Distance excludes every return-to-depot segment, includes the forced depot-to-start edge, and uses input mean off-diagonal distance. '
        'Waiting uses input mean travel time (native speed1). No final cost/oracle/gap normalization. '
        'Behavior standardization uses the8000 training rows only. Full definitions, means/std and row lists are saved.', '',
        '## Actual full-pipeline timing', '',
        'The selected solver actually resumes the live prefix environment and decoder K/V at native selected_count12; '
        'the prefix is not rerun. An uninterrupted decoding witness produces exactly identical routes and costs. '
        'Unselected solver probing and both encoders are charged. All POMO starts remain through completion.',
        f'Seed5102 locks12 training instances at min/median/max input size,4 each; batch1,3 repeats, synchronized wall time. '
        'Time includes CPU native collation/H2D, both encoders/probes, feature aggregation, selection, and chosen solver completion. '
        'Actual methods may differ between A/B; this is the cost of the deployed pipelines, not a matched-solver microbenchmark.', '',
        '| Model | Both-encoder/probe feature ms | Selection ms | Chosen completion ms | Total ms |',
        '|---|---:|---:|---:|---:|']
    for name in MODELS:
        values = timing['summary'][name]
        lines.append(f'| {name} | {values["feature_ms"]["mean"]:.3f} | {values["selection_ms"]["mean"]:.3f} | '
            f'{values["completion_ms"]["mean"]:.3f} | {values["total_ms"]["mean"]:.3f} |')
    lines += ['', f'B/A total latency ratio: **{timing["ratio_behavior_to_static"]:.4f}x**. '
        f'Acceptable under predeclared <=1.25x engineering screen: **{timing["overhead_acceptable"]}**. '
        'The25% threshold is a practical screen chosen before training, not a universal application requirement. '
        'This is batch1 latency on one RTX3090, not a claim about batched throughput or other hardware. '
        'Timing full solves only provide runtime witnesses, never new labels used for feature fitting or training.', '',
        'The timed completion uses only original native model/env steps after the probe. '
        'Independent accumulated-distance bookkeeping remains enabled in the route-continuation preflight, '
        'but is disabled during completion timing because no downstream feature uses it. '
        'This timing-only code change was made after training; cached features, weights, selected updates and saved predictions did not change.', '',
        '## Corrections and harms', '', '| Set | Corrected | Harmed | Net changed cost | Regret delta (pp) |', '|---|---:|---:|---:|---:|']
    for row in changes:
        lines.append(f'| {row["split"]} | {row["corrected"]} | {row["harmed"]} | {row["mean_cost_delta"]:+.7f} | {row["pair_regret_delta_pp"]:+.6f} |')
    lines += ['', 'Paired fixed-prediction bootstrap intervals are in paired_uncertainty.csv. '
        'They measure holdout sampling uncertainty, not training-seed uncertainty.', '',
        '## Final-update observations', '',
        '| Model | Train accuracy / CE | Development accuracy / CE | Internal accuracy / CE | Original validation accuracy / CE |',
        '|---|---:|---:|---:|---:|']
    for name in MODELS:
        values = [histories[name][-1]['metrics'][split] for split in SETS]
        lines.append('| ' + name + ' | ' + ' | '.join(f'{v["accuracy"]*100:.2f}% / {v["ce"]:.6f}' for v in values) + ' |')
    lines += ['', '## Provenance and interpretation limits', '',
        'The solver checkpoint hashes, source hashes and training-source limitations are retained in solver_provenance.json. '
        'Available ReLD training source uses fresh online generated instances, but exact historical instance overlap cannot be certified. '
        'No selector weights are used to initialize frozen solvers or readouts. External solver pretraining is not equal-data-budget training.',
        'Reference: [Renau and Hart, On the Utility of Probing Trajectories for Algorithm-Selection,2024](https://arxiv.org/html/2401.12745v1). '
        'Section3.2 motivates short current-instance trajectories and combining solvers; section6 requires accounting for portfolio probing '
        'and discusses continuing the chosen run. Their experiments are continuous black-box optimization. '
        'The ten routing actions, native progress/constraint/probability features, POMO summaries and binary neural readout are this project\'s adaptation.', '',
        '## Artifacts and reproduction', '',
        'config.json, feature_definition.json, behavior_normalization.json, solver_provenance.json, '
        'preflight.json, comparison.csv, decision_changes.csv, paired_uncertainty.csv, learning_curves.png, '
        'internal_predictions.csv, val_predictions.csv, inference_timing.csv/json; per-group args, history, logs, predictions and selection. '
        'Large .pt static/behavior caches and best/last checkpoints remain local under existing repository ignore rules. '
        'W&B offline run files remain local. Execution command is in execution_command.md.', '']
    (root / 'RESULTS.md').write_text('\n'.join(lines))
    print('\n'.join(lines[:18]), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=ROOT)
    args = parser.parse_args()
    analyze(args.root)


if __name__ == '__main__':
    main()
