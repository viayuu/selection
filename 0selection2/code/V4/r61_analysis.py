"""Full-pool outcomes after dense saved-route representation pretraining."""

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
from .r60_analysis import advancement_screen as comparison_screen


ROOT = Path('code/V4/runs/R61_behavior_pretraining')
INCUMBENT = Path('code/V4/runs/R60_expanded_partial_supervision/B_expanded_seed2')
RUN_NAMES = dict(A='A_replay_seed2', B='B_expanded_seed2')


def advancement_screen(incumbent, replay, expanded):
    screen = comparison_screen(incumbent, replay, expanded)
    screen = {key.replace('R45A', 'incumbent'): value for key, value in screen.items()}
    screen.update(incumbent='R60B validation-cost-best epoch39',
                  minimum_top1=incumbent['macro_top1'] + .03)
    return screen


def plot_pretraining(root, histories):
    plt = plotting()
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    rows = []
    for arm, history in histories.items():
        for row in history:
            rows.append(dict(arm=arm, epoch=row['epoch'],
                             successful_updates=row['successful_updates'], **row['optimization']))
        for ax, field in zip(axes, ('loss', 'successor_accuracy', 'lr')):
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
    records = {arm: load_json(root / name / 'result.json') for arm, name in RUN_NAMES.items()}
    histories = {arm: load_json(root / name / 'history.json') for arm, name in RUN_NAMES.items()}
    pretrained = {arm: load_json(root / name / 'pretrain' / 'result.json')
                  for arm, name in RUN_NAMES.items()}
    pretrain_histories = {arm: load_json(root / name / 'pretrain' / 'history.json')
                         for arm, name in RUN_NAMES.items()}
    raw = {problem: read_raw_costs(problem, 'val') for problem in PROBLEMS}
    historical = load_json(REFERENCE / 'best_eval.json')
    reference, _ = replay_directory(REFERENCE / 'val_predictions' / f"epoch{historical['epoch']:03d}", raw)
    incumbent, incumbent_predictions = replay_directory(INCUMBENT / 'val_predictions' / 'best', raw)
    results, predictions = {}, {}
    for arm, name in RUN_NAMES.items():
        results[arm], predictions[arm] = replay_directory(root / name / 'val_predictions' / 'best', raw)
    all_results = {'R45A historical': reference, 'R60B incumbent': incumbent,
                   **{'R61' + arm: result for arm, result in results.items()}}
    dump(root / 'validation_results.json', results)
    rows = [row for name, result in all_results.items() for row in comparison_rows(name, result)]
    write_csv(root / 'comparison.csv', rows)
    write_csv(root / 'family_comparison.csv', [dict(model=name, family=family, **scores)
              for name, result in all_results.items() for family, scores in result['families'].items()])
    distributions = []
    for name, result in all_results.items():
        plot_methods(name, result, root / 'curves' / name.replace(' ', '_') / 'per_problem')
        for problem, metrics in result['per_problem'].items():
            for solver in metrics['pool']:
                distributions.append(dict(model=name, problem=problem, solver=solver,
                    pick_fraction=metrics['arm_distribution'][solver],
                    native_winner_fraction=metrics['method_top1'][solver]))
    write_csv(root / 'solver_distributions.csv', distributions)
    changed = changes(incumbent_predictions, predictions['A'], 'R60B', 'R61A')
    changed += changes(incumbent_predictions, predictions['B'], 'R60B', 'R61B')
    changed += changes(predictions['A'], predictions['B'], 'R61A', 'R61B')
    write_csv(root / 'decision_changes.csv', changed)
    screen = advancement_screen(incumbent['macro'], results['A']['macro'], results['B']['macro'])
    dump(root / 'advancement_screen.json', screen)
    plot_histories(root, histories, incumbent['macro'])
    plot_pretraining(root, pretrain_histories)

    lines = ['# R61: Dense Saved-Route Behavior Pretraining', '',
        'Seed2; original18 tasks, complete legal pools and native winners. Validation only.',
        'No test access, new solver execution, new annotations or R58 operations.', '',
        '## Validation-selected outcomes', '']
    lines += table([row for row in rows if row['problem'] == 'ALL'])
    lines += ['', 'A pretrains on original-TRAIN route replay; B uses additional saved TRAIN routes.',
        'Both transfer only the final instance encoder into fresh complete-pool selectors.',
        'Fine-tuning uses the unchanged original R43 objective and R-Drop; route heads are discarded.',
        'There is no route prediction, solver probing, binary expert or candidate gate at inference.', '',
        '## Matched training', '',
        '| Arm | Pretrain updates | Finetune updates | Best epoch | Stop epoch | Total run hours |',
        '|---|---:|---:|---:|---:|---:|']
    for arm, record in records.items():
        pre = pretrained[arm]
        hours = (pre['wall_seconds'] + record['wall_seconds']) / 3600
        lines.append(f"| R61{arm} | {pre['successful_updates']} | {record['successful_updates']} | "
                     f"{record['best']['epoch']} | {record['epochs_completed']} | {hours:.4f} |")
    lines += ['', 'Stage1 transfers the final encoder after two fixed complete passes; no validation selection.',
        'Stage2 selects strict minimum full18 validation macro_vs_sbs_pct, not best Top1.',
        'Replay A controls the common fine-tuning LR trace/horizon; B selects its own best checkpoint.',
        'Initialization, source split, actual-size schedule and update budgets are matched.', '',
        '## Selected-checkpoint train and validation', '',
        '| Arm | Split | Top1 % | CE | Mean cost | Actual regret % |',
        '|---|---|---:|---:|---:|---:|']
    for arm, record in records.items():
        for split in ('train', 'val'):
            metrics = record['best'][split]['macro']
            lines.append(f"| R61{arm} | {split} | {100 * metrics['macro_top1']:.4f} | "
                         f"{metrics['macro_ce']:.6f} | {metrics['macro_mean_cost']:.6f} | "
                         f"{metrics['macro_actual_regret_pct']:.6f} |")
    lines += ['', 'Train evaluation covers original full-pool TRAIN in eval mode.',
        'Pretraining curves show online successor learning, not full-pool selection accuracy.', '',
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
        f"B minus replay A: {screen['top1_gain_vs_replay_pp']:+.4f}pp; required>=2.",
        f"Regret no worse than incumbent/A: {screen['regret_no_worse_than_incumbent']}/"
        f"{screen['regret_no_worse_than_replay']}.", '',
        'The screen is a direction decision, not proof of70% or unseen-test performance.',
        'Better successor learning alone cannot establish better solver selection.',
        'A negative result rejects this pretraining/transfer recipe, not an accuracy ceiling.', '',
        '## Evidence limits', '',
        'One seed. R45A and the locked R60B epoch39 incumbent are historical references.',
        'Only the two archived ReLD route behaviors provide expanded structural targets.',
        'No additional winner or complete-pool label is invented for expanded rows.',
        'Actual route availability/counts and source pairing are recorded during target conversion.',
        'Small existing calibration is not complete historical deployment certification.',
        'Legacy B feasible-domain inconsistencies and the existing VRPLTW repair provenance',
        'remain disclosed; this experiment does not claim a repaired common-task benchmark.', '',
        '## Per problem', '']
    lines += table([row for row in rows if row['problem'] != 'ALL'])
    (root / 'RESULTS.md').write_text('\n'.join(lines) + '\n')
    return screen


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    print(json.dumps(summarize(parser.parse_args().root), indent=2))
