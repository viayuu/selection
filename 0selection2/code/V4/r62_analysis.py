"""Validation-only outcomes after source-separated portfolio pretraining."""

import argparse
import json
from pathlib import Path

from code.unified_selector.registry import PROBLEMS
from .multitask_probe import dump
from .performance_analysis import write_csv
from .performance_targets import read_raw_costs
from .r43_analysis import plot_methods, plotting
from .r59_analysis import (REFERENCE, changes, comparison_rows, load_json,
                           plot_histories, replay_directory, table)


ROOT = Path('code/V4/runs/R62_portfolio_performance_pretraining')
INCUMBENT = Path('code/V4/runs/R60_expanded_partial_supervision/B_expanded_seed2')
RUN_NAMES = dict(A='A_subset_seed2', B='B_full_seed2')


def advancement_screen(incumbent, subset, full):
    incumbent_gain = 100 * (full['macro_top1'] - incumbent['macro_top1'])
    subset_gain = 100 * (full['macro_top1'] - subset['macro_top1'])
    no_worse_incumbent = (full['macro_actual_regret_pct']
                         <= incumbent['macro_actual_regret_pct'] + 1e-10)
    no_worse_subset = (full['macro_actual_regret_pct']
                      <= subset['macro_actual_regret_pct'] + 1e-10)
    return dict(passed=bool(incumbent_gain >= 3. - 1e-10
                           and subset_gain >= 2. - 1e-10
                           and no_worse_incumbent and no_worse_subset),
                top1_gain_vs_incumbent_pp=incumbent_gain,
                top1_gain_vs_subset_pp=subset_gain,
                regret_no_worse_than_incumbent=bool(no_worse_incumbent),
                regret_no_worse_than_subset=bool(no_worse_subset),
                minimum_top1=.5236111111111112,
                required_gain_vs_incumbent_pp=3., required_gain_vs_subset_pp=2.,
                incumbent='R60B validation-cost-best epoch39',
                scope='validation direction screen; no test or 70% achievement claim')


def plot_pretraining(root, histories):
    plt = plotting()
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    rows = []
    for arm, history in histories.items():
        for row in history:
            rows.append(dict(arm=arm, epoch=row['epoch'],
                             successful_updates=row['successful_updates'],
                             **row['optimization']))
        for ax, field in zip(axes, ('loss', 'pair_accuracy', 'lr')):
            ax.plot([row['successful_updates'] for row in history],
                    [row['optimization'][field] for row in history], label=arm)
            ax.set(title=field, xlabel='Successful pretraining updates')
            ax.grid(alpha=.2)
    for ax in axes:
        ax.legend()
    fig.tight_layout()
    fig.savefig(root / 'curves' / 'pretraining.png', dpi=150)
    plt.close(fig)
    write_csv(root / 'pretraining_curves.csv', rows)


def summarize(root=ROOT):
    root = Path(root)
    records = {arm: load_json(root / name / 'result.json')
               for arm, name in RUN_NAMES.items()}
    histories = {arm: load_json(root / name / 'history.json')
                 for arm, name in RUN_NAMES.items()}
    pretrained = {arm: load_json(root / name / 'pretrain' / 'result.json')
                  for arm, name in RUN_NAMES.items()}
    pretrain_histories = {arm: load_json(root / name / 'pretrain' / 'history.json')
                         for arm, name in RUN_NAMES.items()}
    raw = {problem: read_raw_costs(problem, 'val') for problem in PROBLEMS}
    historical = load_json(REFERENCE / 'best_eval.json')
    reference, _ = replay_directory(
        REFERENCE / 'val_predictions' / f"epoch{historical['epoch']:03d}", raw)
    incumbent, incumbent_predictions = replay_directory(
        INCUMBENT / 'val_predictions' / 'best', raw)
    results, predictions = {}, {}
    for arm, name in RUN_NAMES.items():
        results[arm], predictions[arm] = replay_directory(
            root / name / 'val_predictions' / 'best', raw)
    all_results = {'R45A historical': reference, 'R60B incumbent': incumbent,
                   **{'R62' + arm: result for arm, result in results.items()}}
    dump(root / 'validation_results.json', results)
    rows = [row for name, result in all_results.items()
            for row in comparison_rows(name, result)]
    write_csv(root / 'comparison.csv', rows)
    write_csv(root / 'family_comparison.csv', [dict(model=name, family=family, **scores)
              for name, result in all_results.items()
              for family, scores in result['families'].items()])
    distributions = []
    for name, result in all_results.items():
        plot_methods(name, result, root / 'curves' / name.replace(' ', '_') / 'per_problem')
        for problem, metrics in result['per_problem'].items():
            for solver in metrics['pool']:
                distributions.append(dict(model=name, problem=problem, solver=solver,
                    pick_fraction=metrics['arm_distribution'][solver],
                    native_winner_fraction=metrics['method_top1'][solver]))
    write_csv(root / 'solver_distributions.csv', distributions)
    changed = changes(incumbent_predictions, predictions['A'], 'R60B', 'R62A')
    changed += changes(incumbent_predictions, predictions['B'], 'R60B', 'R62B')
    changed += changes(predictions['A'], predictions['B'], 'R62A', 'R62B')
    write_csv(root / 'decision_changes.csv', changed)
    screen = advancement_screen(incumbent['macro'], results['A']['macro'], results['B']['macro'])
    dump(root / 'advancement_screen.json', screen)
    plot_histories(root, histories, reference['macro'])
    plot_pretraining(root, pretrain_histories)

    lines = ['# R62: Source-Separated Portfolio-Performance Pretraining', '',
        'Seed2; original18 tasks, complete legal pools and native winners. Validation only.',
        'No test access, new solver execution, new annotations or R58 operations.', '',
        '## Validation-selected outcomes', '']
    lines += table([row for row in rows if row['problem'] == 'ALL'])
    lines += ['', 'A and B pretrain on the same available auxiliary deployment source.',
        'A replays a fixed nested new10k/task subset; B uses the complete available new-input bank.',
        'Missing auxiliary observations restrict pair supervision, never the forward candidate pool.',
        'Stage1 learns observed non-tied utility comparisons without constructing partial-pool winners.',
        'The auxiliary score head is reset; all other original selector parameters transfer.',
        'Stage2 uses only unchanged legacy full-pool labels and the original R43 objective/R-Drop.',
        'Inference uses the original one-pass single-solver selector, not a pretraining branch.', '',
        '## Matched training', '',
        '| Arm | Pretrain updates | Finetune updates | Best epoch | Stop epoch | Total run hours |',
        '|---|---:|---:|---:|---:|---:|']
    for arm, record in records.items():
        pre = pretrained[arm]
        hours = (pre['wall_seconds'] + record['wall_seconds']) / 3600
        lines.append(f"| R62{arm} | {pre['successful_updates']} | {record['successful_updates']} | "
                     f"{record['best']['epoch']} | {record['epochs_completed']} | {hours:.4f} |")
    lines += ['', 'Both arms transfer the final fixed-budget pretraining state, not a validation-picked state.',
        'A controls the common fine-tuning LR trace/horizon; each selects strict minimum macro_vs_sbs_pct.',
        'Scheduled updates and actual input-size exposure are matched, not observed pair counts.',
        'Actual source/observation coverage is recorded by ordinary conversion in data_cache/preparation.json.', '',
        '## Selected-checkpoint train and validation', '',
        '| Arm | Split | Top1 % | CE | Mean cost | Actual regret % |',
        '|---|---|---:|---:|---:|---:|']
    for arm, record in records.items():
        for split in ('train', 'val'):
            metrics = record['best'][split]['macro']
            lines.append(f"| R62{arm} | {split} | {100 * metrics['macro_top1']:.4f} | "
                         f"{metrics['macro_ce']:.6f} | {metrics['macro_mean_cost']:.6f} | "
                         f"{metrics['macro_actual_regret_pct']:.6f} |")
    lines += ['', 'Train evaluation uses eval mode, original full-pool TRAIN and all tail batches.',
        'Auxiliary training pair accuracy is not full-pool selection accuracy.', '',
        '## Decision changes', '',
        '| Before | After | Category | N | Macro cost delta | Macro regret delta pp |',
        '|---|---|---|---:|---:|---:|']
    for row in changed:
        if row['problem'] == 'ALL':
            lines.append(f"| {row['before']} | {row['after']} | {row['category']} | {row['n']} | "
                         f"{row['mean_cost_delta']:+.8f} | {row['actual_regret_delta_pct']:+.6f} |")
    lines += ['', '## Precommitted decision', '',
        f"Advancement screen passed: **{screen['passed']}**.",
        f"B minus incumbent Top1: {screen['top1_gain_vs_incumbent_pp']:+.4f}pp; required>=3.",
        f"B minus subset A: {screen['top1_gain_vs_subset_pp']:+.4f}pp; required>=2.",
        f"Regret no worse than incumbent/A: {screen['regret_no_worse_than_incumbent']}/"
        f"{screen['regret_no_worse_than_subset']}.", '',
        'This is a direction screen, not unseen-test evidence or achievement of70% Top1.',
        'Preserve partial gains even when the substantial-gain screen fails.',
        'Failure rejects this transfer recipe, not the possibility of better selection or an accuracy ceiling.', '',
        '## Evidence limits', '',
        'One seed. R45A/R60B are historical references, not newly trained matched controls.',
        'Expansion deployments can differ in weights, budgets, stochastic execution and constraint semantics.',
        'They provide explicitly different-source auxiliary knowledge, not certified legacy labels.',
        'Observed cells are never interpreted as a complete-pool winner or used to replace original costs.',
        'Both matched arms use the same frozen auxiliary source; normal loading records actual coverage.',
        'Existing source calibration and VRPLTW input-repair limitations remain disclosed.',
        'The legacy B feasible-domain mismatch remains unresolved; this is not a repaired common-task benchmark.', '',
        '## Per problem', '']
    lines += table([row for row in rows if row['problem'] != 'ALL'])
    (root / 'RESULTS.md').write_text('\n'.join(lines) + '\n')
    return screen


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    print(json.dumps(summarize(parser.parse_args().root), indent=2))
