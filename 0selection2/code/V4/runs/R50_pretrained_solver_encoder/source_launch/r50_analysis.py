"""Same-checkpoint R50 comparisons, paired predictions and full-inference timing."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from .multitask_probe import dump, file_hash
from .r48_common import binary_metrics, write_csv
from .r50_experiment import ROOT, R49, MODELS, SETS


def analyze(root):
    config = json.loads((root / 'config.json').read_text())
    selections = {name: json.loads((root / name / 'selection.json').read_text()) for name in MODELS}
    histories = {name: json.loads((root / name / 'history.json').read_text()) for name in MODELS}
    names = ('R49B_8000',) + MODELS
    archives = {name: {split: dict(np.load((R49 / 'B_8000_seed2' if name == names[0] else root / name)
                                         / (split + '_predictions.npz'))) for split in SETS} for name in names}
    reference_selection = json.loads((R49 / 'B_8000_seed2/selection.json').read_text())
    selected_steps = {names[0]: reference_selection['best_updates']}
    selected_steps.update({name: selections[name]['best_updates'] for name in MODELS})
    if file_hash(root / 'split_manifest.json') != config['split_manifest_sha256']:
        raise ValueError('Locked R49 split changed')
    plan = np.load(root / 'sampling_plan.npz')['indices']
    np.testing.assert_array_equal(plan, np.load(R49 / 'B_8000_seed2/sampling_plan.npz')['indices'])
    unique, counts = np.unique(plan, return_counts=True)
    if plan.shape != (4000, 128) or len(unique) != 8000 or not (counts == 64).all():
        raise ValueError('Training coverage/budget differs from protocol')
    results, replay = [], []
    for name in names:
        for split in SETS:
            reference = archives[names[0]][split]
            archive = archives[name][split]
            for field in ('indices', 'labels', 'costs', 'customers'):
                np.testing.assert_array_equal(archive[field], reference[field])
            metrics = binary_metrics(archive['scores'], archive['costs'])
            if name in MODELS:
                history = histories[name]
                if [r['updates'] for r in history] != list(range(0, 4001, 200)):
                    raise ValueError('Missing full-set evaluation points')
                chosen = min(history, key=lambda row: row['metrics']['development']['ce'])
                if chosen['updates'] != selected_steps[name]:
                    raise ValueError('Checkpoint was not selected by development CE only')
                error = max(abs(float(metrics[k]) - float(chosen['metrics'][split][k])) for k in metrics)
                if error > 1e-10:
                    raise ValueError('Saved prediction metrics do not replay')
                replay.append(dict(model=name, split=split, maximum_metric_error=error))
                if selections[name]['initial_readout_sha256'] != config['initial_readout_sha256']:
                    raise ValueError('A/B readout initialization mismatch')
            results.append(dict(model=name, split=split, best_updates=selected_steps[name], **metrics))
    write_csv(root / 'comparison.csv', results)
    dump(root / 'execution_checks.json', dict(same_r49b_split_and_batch_sequence=True,
        updates_each=4000, batch_size=128, unique_training_rows=8000,
        presentations_per_training_row=64, initialization_equal=True, prediction_replay=replay,
        test_read=False, current_instance_labels_not_encoder_or_readout_inputs=True))
    indexed = {(r['model'], r['split']): r for r in results}
    changes = []
    for split, filename in (('internal', 'internal_predictions.csv'), ('original_val', 'val_predictions.csv')):
        reference = archives[names[0]][split]
        rows = []
        for i, index in enumerate(reference['indices']):
            row = dict(source_split='train' if split == 'internal' else 'val', original_index=int(index),
                customers=int(reference['customers'][i]), binary_label=int(reference['labels'][i]),
                MOEL_cost=float(reference['costs'][i, 0]), MTL_cost=float(reference['costs'][i, 1]))
            for name in names:
                scores = archives[name][split]['scores'][i].astype(np.float64)
                probability = np.exp(scores - scores.max())
                row.update({name + '_prediction': int(scores.argmax()), name + '_score0': float(scores[0]),
                            name + '_score1': float(scores[1]), name + '_probability1': float(probability[1] / probability.sum())})
            rows.append(row)
        write_csv(root / filename, rows)
        for first, second in ((MODELS[1], names[0]), (MODELS[1], MODELS[0])):
            left, right = archives[first][split], archives[second][split]
            truth = left['labels']
            pred, old = left['scores'].argmax(1), right['scores'].argmax(1)
            costs = left['costs']
            selected_cost = costs[np.arange(len(truth)), pred]
            old_cost = costs[np.arange(len(truth)), old]
            changes.append(dict(comparison=first + ' minus ' + second, split=split,
                corrected=int(((old != truth) & (pred == truth)).sum()),
                harmed=int(((old == truth) & (pred != truth)).sum()),
                changed=int((pred != old).sum()), mean_cost_delta=float((selected_cost - old_cost).mean()),
                actual_regret_delta_pp=float(((selected_cost - old_cost) / costs.min(1) * 100).mean())))
    write_csv(root / 'decision_changes.csv', changes)
    labels = dict(train='Train 8000', development='Development (selection)',
                  internal='Internal holdout', original_val='Original validation')
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
    fig.suptitle('R50: random vs pretrained frozen ReLD node representations', fontsize=13)
    fig.tight_layout()
    fig.savefig(root / 'learning_curves.png', dpi=180)
    plt.close(fig)
    screen = {}
    for split in ('internal', 'original_val'):
        base, random, pretrained = [indexed[(name, split)] for name in names]
        screen[split] = dict(
            accuracy_gain_vs_r49b_pp=(pretrained['accuracy'] - base['accuracy']) * 100,
            regret_delta_vs_r49b_pp=pretrained['pair_regret_pct'] - base['pair_regret_pct'],
            accuracy_gain_vs_random_pp=(pretrained['accuracy'] - random['accuracy']) * 100,
            regret_delta_vs_random_pp=pretrained['pair_regret_pct'] - random['pair_regret_pct'],
            meets_screen=(pretrained['accuracy'] >= base['accuracy'] + .05
                and pretrained['pair_regret_pct'] < base['pair_regret_pct']
                and pretrained['accuracy'] > random['accuracy']
                and pretrained['pair_regret_pct'] < random['pair_regret_pct']))
    passed = all(value['meets_screen'] for value in screen.values())
    dump(root / 'advancement_screen.json', dict(passed=passed, by_set=screen,
        artificial_engineering_screen=True, not_a_statistical_or_theoretical_threshold=True))
    timing = json.loads((root / 'inference_timing.json').read_text())
    cache = json.loads((root / 'cache_manifest.json').read_text())
    lines = ['# R50: frozen solver-encoder transfer', '',
        'Scope: OVRPTW, RELD_MOEL (0) vs RELD_MTL (1), seed2. Original FP64 cost labels; exact ties excluded. '
        'No test, new instances, new costs, selector continuation, solver decoding, or extra experiment branches.', '',
        '## Fixed protocol', '',
        'The exact R49 split and R49B 4000x128 sample plan are reused. All 8000 training instances appear64 times each. '
        'Development:999 non-ties of1000; internal:1000; original validation:1000. '
        'All 21 evaluation points (u0,200,...,4000) evaluate complete sets in eval mode, with tails retained. '
        'Strict minimum development CE alone locks a single checkpoint per model.',
        'A freezes original randomly initialized MOEL/MTL architectures; B freezes R48-pinned epoch5000 Train_ALL solver weights. '
        'Encoders stay eval/FP32/TF32-off and never enter the optimizer. Both groups copy identical fresh readout parameters and use paired dropout/sample streams.',
        'Each node retains both128-channel representations. Separate trainable LayerNorms precede concatenation with7 original node fields '
        '(xy, demand, service, start/end TW, depot flag). Shared263->128->128 MLP, customer-only masked mean/max, '
        'separate depot vector and log1p(n)/6, then385->128->2 MLP. No embedding subtraction or early global pooling. '
        f'Readout parameters:{config["trainable_parameters"]}; AdamW LR1e-3 WD1e-4, dropout0.1, clip1.0; unweighted CE only; no schedule/early stop.',
        'Native encoder input is depot xy and customer xy/demand/start/end TW. Service time is not used by the original Encoder; '
        'it is retained identically as a readout input in A/B. Native capacity1 and OVRPTW constraints are unchanged. '
        'Graphs are encoded in equal-size batches without padding; padding is introduced only after encoding and masked in readout.', '',
        '## Development-selected results', '',
        '| Model | Update | Train accuracy | Development CE | Internal accuracy / regret | Original validation accuracy / regret |',
        '|---|---:|---:|---:|---:|---:|']
    for name in names:
        train, dev, internal, val = [indexed[(name, split)] for split in SETS]
        lines.append(f'| {name} | {selected_steps[name]} | {train["accuracy"]*100:.2f}% | {dev["ce"]:.6f} | '
            f'{internal["accuracy"]*100:.2f}% / {internal["pair_regret_pct"]:.6f}% | '
            f'{val["accuracy"]*100:.2f}% / {val["pair_regret_pct"]:.6f}% |')
    lines += ['', 'All following values use the same development-selected checkpoint:', '',
        '| Model | Set | N | Accuracy | CE | Mean cost | Pair regret |', '|---|---|---:|---:|---:|---:|---:|']
    for row in results:
        lines.append(f'| {row["model"]} | {row["split"]} | {row["n"]} | {row["accuracy"]*100:.2f}% | '
            f'{row["ce"]:.6f} | {row["mean_cost"]:.7f} | {row["pair_regret_pct"]:.6f}% |')
    lines += ['', '## Final-update observations', '',
        'Update4000 is not substituted for the development-selected checkpoint:', '',
        '| Model | Train accuracy / CE | Development accuracy / CE | Internal accuracy / CE | Original validation accuracy / CE |',
        '|---|---:|---:|---:|---:|']
    for name in MODELS:
        values = [histories[name][-1]['metrics'][split] for split in SETS]
        lines.append('| ' + name + ' | ' + ' | '.join(f'{v["accuracy"]*100:.2f}% / {v["ce"]:.6f}' for v in values) + ' |')
    lines += ['', '## Interpretation', '',
        'Predeclared screen: B gains at least5pp accuracy over R49B on BOTH held-out sets, reduces BOTH pair regrets, '
        'and is better than A. This is an engineering continuation screen, not a guaranteed outcome or significance test.',
        f'Screen passed: **{passed}**.', '']
    for split, values in screen.items():
        lines.append(f'- {split}: B minus R49B accuracy {values["accuracy_gain_vs_r49b_pp"]:+.2f}pp; '
            f'regret {values["regret_delta_vs_r49b_pp"]:+.6f}pp. B minus random A accuracy '
            f'{values["accuracy_gain_vs_random_pp"]:+.2f}pp; regret {values["regret_delta_vs_random_pp"]:+.6f}pp.')
    if passed:
        lines += ['', 'This fixed-seed probe supports useful transfer from these pretrained solver encoders under this readout. '
            'It does not establish18-task gains or separate the contributions of the two encoders.']
    else:
        lines += ['', 'This configuration did not achieve the predeclared useful-transfer screen. '
            'The two frozen solver representations under this readout do not establish the requested clear generalization improvement. '
            'This does not prove random labels, an accuracy ceiling, or that every form of solver-representation transfer fails. '
            'No automatic fine-tuning, pooling redesign, extra seed, or18-task experiment was started.']
    lines += ['', 'Decision changes are in decision_changes.csv; corrected and newly harmed instances are reported together. '
        'Regret always refers to the two-method Oracle, not the original seven-candidate Oracle.', '',
        '## Full inference cost', '',
        'Measured on the one allocated idle RTX3090, with five warmups and30 synchronized repetitions. '
        'Input-derived median customer count is fixed before timing. Full pipeline includes native CPU collation, host-to-GPU transfer, '
        'BOTH encoders and readout; excludes disk reads/model loading/route solving. Cached readout is a separate lower-bound component, not deployment cost. '
        'Latency may be affected by another process on the other GPU/shared CPU.', '',
        '| Model | Component | Batch | Customers | Mean ms / batch | Mean ms / instance | p95 ms / batch |',
        '|---|---|---:|---:|---:|---:|---:|']
    for name in MODELS:
        for row in timing[name]['records']:
            lines.append(f'| {name} | {row["component"]} | {row["batch_size"]} | {row["customers"]} | '
                f'{row["mean_ms"]:.3f} | {row["per_instance_mean_ms"]:.3f} | {row["p95_ms"]:.3f} |')
    lines += ['', '## Provenance and boundaries', '',
        '[ReLD, ICLR2025](https://arxiv.org/html/2503.00753v1), sections2.1-2.3, analyzes the information represented by '
        'static node embeddings during route construction. R50 borrows the trained instance encoders, not its decoder modification; '
        'the paper does not establish that these representations predict solver-vs-solver outcomes.',
        'Exact local weight/source hashes, native parameters and limitations are in pretraining_provenance.json. '
        'Both current epoch5000 checkpoints identify Train_ALL. Available Trainer source generates fresh online routing instances '
        'and trains using POMO/REINFORCE; the task list includes OVRPTW. '
        'Historical complete CLI, generator seeds and training-instance manifest are not available, so no certified no-overlap claim is made. '
        'This introduces external solver pretraining and is not equal total pretraining-data budget.',
        'Necessary witnesses: both encoder outputs match original pre_forward exactly; no routes decoded; frozen state hashes unchanged; '
        'readout blocks receive finite nonzero gradients; masks/labels/initialization/sample order and prediction replay checked. '
        'See preflight.json, cache_manifest.json, execution_checks.json and each group gradient_witness.json.',
        f'Cache generation seconds: A={cache[MODELS[0]]["seconds"]:.2f}, B={cache[MODELS[1]]["seconds"]:.2f}. '
        'Large FP32 node caches and weight files remain local/ignored by Git; hashes and source/config/predictions are versioned.', '',
        '## Reproduction', '',
        '```bash',
        'R50_GPU_UUID=<currently-idle-3090-uuid> bash code/V4/run_v4_r50.sh all',
        'bash code/V4/run_v4_r50.sh analysis', '```',
        'The all stage intentionally refuses to overwrite existing runs. Only train/val are permitted by the underlying data loader.']
    (root / 'RESULTS.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines[lines.index('## Development-selected results'):lines.index('## Final-update observations')]), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=ROOT)
    analyze(parser.parse_args().root)
