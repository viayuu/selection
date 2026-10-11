"""Full-pool validation results for expanded, partially observed TRAIN data."""

import argparse
import json
from pathlib import Path

from code.unified_selector.registry import PROBLEMS
from .multitask_probe import dump
from .performance_analysis import write_csv
from .performance_targets import read_raw_costs
from .r43_analysis import plot_methods, plotting
from .r59_analysis import (REFERENCE, advancement_screen as transfer_screen,
                           changes, comparison_rows, load_json, plot_histories,
                           replay_directory, table)


ROOT = Path('code/V4/runs/R60_expanded_partial_supervision')
RUN_NAMES = dict(A='A_replay_seed2', B='B_expanded_seed2')


def advancement_screen(reference, replay, expanded):
    screen = transfer_screen(reference, replay, expanded)
    for old, new in (
        ('top1_gain_vs_random_pp', 'top1_gain_vs_replay_pp'),
        ('regret_no_worse_than_random', 'regret_no_worse_than_replay'),
        ('required_gain_vs_random_pp', 'required_gain_vs_replay_pp'),
    ):
        screen[new] = screen.pop(old)
    return screen


def plot_auxiliary(root, histories):
    plt = plotting()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for arm, history in histories.items():
        for ax, key in zip(axes, ('extra_aux_bce', 'extra_aux_accuracy')):
            points = [row for row in history if key in row['optimization']]
            if points:
                ax.plot([row['epoch'] for row in points],
                        [row['optimization'][key] for row in points], label=arm)
            ax.set(xlabel='Epoch', title=key)
            ax.grid(alpha=.2)
    for ax in axes:
        if ax.lines:
            ax.legend()
    fig.tight_layout()
    fig.savefig(root / 'curves' / 'auxiliary_comparison.png', dpi=150)
    plt.close(fig)


def summarize(root=ROOT):
    root = Path(root)
    records = {arm: load_json(root / name / 'result.json') for arm, name in RUN_NAMES.items()}
    histories = {arm: load_json(root / name / 'history.json') for arm, name in RUN_NAMES.items()}
    raw = {problem: read_raw_costs(problem, 'val') for problem in PROBLEMS}
    historical = load_json(REFERENCE / 'best_eval.json')
    directory = REFERENCE / 'val_predictions' / f"epoch{historical['epoch']:03d}"
    reference, reference_predictions = replay_directory(directory, raw)
    results, predictions = {}, {}
    for arm, name in RUN_NAMES.items():
        results[arm], predictions[arm] = replay_directory(
            root / name / 'val_predictions' / 'best', raw)
    dump(root / 'validation_results.json', results)
    all_results = {'R45A historical': reference,
                   **{'R60' + arm: result for arm, result in results.items()}}
    rows = [row for name, result in all_results.items() for row in comparison_rows(name, result)]
    write_csv(root / 'comparison.csv', rows)
    write_csv(root / 'family_comparison.csv', [dict(model=name, family=family, **scores)
              for name, result in all_results.items() for family, scores in result['families'].items()])
    distributions, single_methods = [], []
    for name, result in all_results.items():
        plot_methods(name, result, root / 'curves' / name.replace(' ', '_') / 'per_problem')
        for problem, metrics in result['per_problem'].items():
            for solver in metrics['pool']:
                distributions.append(dict(model=name, problem=problem, solver=solver,
                    pick_fraction=metrics['arm_distribution'][solver],
                    native_winner_fraction=metrics['method_top1'][solver]))
                single_methods.append(dict(model=name, problem=problem, solver=solver,
                    fixed_solver_top1=metrics['method_top1'][solver],
                    fixed_solver_mean_cost=metrics['method_mean_cost'][solver],
                    selector_top1=metrics['top1'], selector_mean_cost=metrics['mean_cost'],
                    selector_top1_exceeds=solver in metrics['top1_exceeds_methods'],
                    selector_cost_exceeds=solver in metrics['cost_exceeds_methods']))
    write_csv(root / 'solver_distributions.csv', distributions)
    write_csv(root / 'single_solver_comparison.csv', single_methods)
    changed = changes(reference_predictions, predictions['A'], 'R45A', 'R60A')
    changed += changes(reference_predictions, predictions['B'], 'R45A', 'R60B')
    changed += changes(predictions['A'], predictions['B'], 'R60A', 'R60B')
    write_csv(root / 'decision_changes.csv', changed)
    screen = advancement_screen(reference['macro'], results['A']['macro'], results['B']['macro'])
    dump(root / 'advancement_screen.json', screen)
    plot_histories(root, histories, reference['macro'])
    plot_auxiliary(root, histories)

    lines = ['# R60: Expanded Partial Comparison Supervision', '',
        'Seed2; original18 tasks and complete legal pools. Validation only; no test access.',
        'R58 remains deferred. Legacy B feasible-domain discrepancies are not repaired.', '',
        '## Validation-selected outcomes', '']
    lines += table([row for row in rows if row['problem'] == 'ALL'])
    lines += ['', 'Both arms train the original R45A/R43A scoring architecture from scratch.',
        'A adds old-TRAIN MOEL/MTL comparison replay; B uses additional TRAIN instances.',
        'The original full-pool objective, native winner and R-Drop remain in both arms.',
        'Extra data supervise only the observed utility difference with coefficient0.35.',
        'No unobserved solver cost, full-pool winner, Oracle or risk target is invented.',
        'No separate binary expert or candidate pruning is used at inference.', '',
        '## Training and selection', '',
        '| Arm | Best epoch | Stop epoch | Successful updates | Run wall hours | Last5 val Top1 % |',
        '|---|---:|---:|---:|---:|---:|']
    for arm, record in records.items():
        lines.append(f"| R60{arm} | {record['best']['epoch']} | {record['epochs_completed']} | "
                     f"{record['successful_updates']} | {record['wall_seconds'] / 3600:.4f} | "
                     f"{100 * record['last5_val']['macro_top1']:.4f} |")
    lines += ['', 'Checkpoint selection: strict minimum original-validation macro_vs_sbs_pct.',
        'Shared initialization, original batches and auxiliary task/scale plan. Replay A sets',
        'the common LR trace and training horizon; B independently selects its best. Extra ties are',
        'excluded from comparison loss, not relabeled as a full-pool winner.', '',
        '## Selected-checkpoint train and validation', '',
        '| Arm | Split | Top1 % | CE | Mean cost | Actual regret % |',
        '|---|---|---:|---:|---:|---:|']
    for arm, record in records.items():
        for split in ('train', 'val'):
            metrics = record['best'][split]['macro']
            lines.append(f"| R60{arm} | {split} | {100 * metrics['macro_top1']:.4f} | "
                         f"{metrics['macro_ce']:.6f} | {metrics['macro_mean_cost']:.6f} | "
                         f"{metrics['macro_actual_regret_pct']:.6f} |")
    lines += ['', 'Train evaluation covers the original full-pool training set in eval mode.',
        'Auxiliary comparison curves are online training diagnostics, not full-pool Top1.', '',
        '## Decision changes', '',
        '| Before | After | Category | N | Macro cost delta | Macro regret delta pp |',
        '|---|---|---|---:|---:|---:|']
    for row in changed:
        if row['problem'] == 'ALL':
            lines.append(f"| {row['before']} | {row['after']} | {row['category']} | {row['n']} | "
                         f"{row['mean_cost_delta']:+.8f} | {row['actual_regret_delta_pct']:+.6f} |")
    lines += ['', '## Precommitted decision', '',
        f"Advancement screen passed: **{screen['passed']}**.",
        f"B minus R45A Top1: {screen['top1_gain_vs_R45A_pp']:+.4f}pp; required>=3.",
        f"B minus replay A: {screen['top1_gain_vs_replay_pp']:+.4f}pp; required>=2.",
        f"Regret no worse than R45A/A: {screen['regret_no_worse_than_R45A']}/"
        f"{screen['regret_no_worse_than_replay']}.", '',
        'This screen is not proof of70% accuracy or performance on unseen test data.',
        'A negative result rejects this data/optimization configuration, not a learning ceiling.', '',
        '## Evidence limits', '',
        'One seed. R45A is historical; A is the matched update/replay control.',
        'Only MOEL/MTL receive expanded observations; do not generalize to all solver pairs.',
        'Existing expansion provenance/calibration supports the selected columns; small',
        'stratified replay is not proof of all-instance historical deployment equivalence.',
        'No new solver runs or cost-table reconstruction occur in R60. Original prefix rows',
        'are excluded from B auxiliary data. Source identities/counts are recorded in the',
        'data manifest and protocol. Legacy task-semantics uncertainty remains explicit.', '',
        '## Per problem', '']
    lines += table([row for row in rows if row['problem'] != 'ALL'])
    (root / 'RESULTS.md').write_text('\n'.join(lines) + '\n')
    return screen


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    print(json.dumps(summarize(parser.parse_args().root), indent=2))
