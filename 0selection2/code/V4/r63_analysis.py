"""Validation-controlled R63 outcomes and optional completed, locked TEST replay."""

import argparse
import json
from pathlib import Path

import numpy as np

from code.unified_selector.registry import PROBLEMS
from .multitask_probe import dump
from .performance_analysis import write_csv
from .performance_targets import read_raw_costs
from .r43_analysis import plot_methods, plotting
from .r59_analysis import (REFERENCE, aggregate, changes, comparison_rows, grouped,
                           load_json, replay_directory, replay_prediction, table)


ROOT = Path('code/V4/runs/R63_recurrent_workspaces')
INCUMBENT = Path('code/V4/runs/R60_expanded_partial_supervision/B_expanded_seed2')
RUN_NAMES = dict(A='A_static_memory_seed2', B='B_communicating_memory_seed2')


def direction_decision(incumbent, results):
    best_arm = 'B' if results['B']['macro_top1'] > results['A']['macro_top1'] else 'A'
    best = results[best_arm]
    gain = 100 * (best['macro_top1'] - incumbent['macro_top1'])
    contrast = 100 * (results['B']['macro_top1'] - results['A']['macro_top1'])
    return dict(best_arm_by_native_top1=best_arm,
                best_validation_native_top1=best['macro_top1'],
                validation_target_hit=bool(best['macro_top1'] >= .70 - 1e-12),
                target_native_top1=.70, goal_achieved=False, heldout_required=True,
                best_gain_vs_incumbent_pp=gain,
                substantial_progress_milestone=bool(gain >= 3. - 1e-10),
                B_minus_A_top1_pp=contrast,
                communicating_advantage_ge_2pp=bool(contrast >= 2. - 1e-10),
                best_regret_delta_vs_incumbent_pp=(best['macro_actual_regret_pct']
                    - incumbent['macro_actual_regret_pct']),
                regret_is_target_veto=False,
                scope='original full18/fullpool native Top1; validation controls direction, never TEST')


def saved_test_predictions(directory):
    per_problem, predictions = {}, {}
    for problem in PROBLEMS:
        per_problem[problem], predictions[problem] = replay_prediction(directory / (problem + '.npz'))
    return dict(per_problem=per_problem, macro=aggregate(per_problem),
                families=grouped(per_problem)), predictions


def completed_test_outputs(root):
    """Read Pascal's completed receipts/predictions only; never start an evaluation."""
    receipt_path = root / 'test_protocol.json'
    lock_path = root / 'final_test_lock.json'
    locks = load_json(lock_path) if lock_path.exists() else {}
    info = dict(status='pending', receipt_path=str(receipt_path), lock_path=str(lock_path),
                locked_models=locks, historical_R45A_status='pending completed final TEST')
    if not receipt_path.exists():
        return {}, {}, info
    receipt = load_json(receipt_path)
    info.update(status=receipt['status'], receipt=receipt)
    if receipt['status'] != 'complete':
        return {}, {}, info
    if not receipt.get('selection_frozen_before_test') or receipt['locked_models'] != locks:
        raise ValueError('Completed TEST receipt must refer to the parent-frozen validation checkpoints')
    reported = load_json(root / 'test_results.json')
    if set(locks) != {'A', 'B', 'R60B'} or set(reported) != set(locks):
        raise ValueError('Completed final TEST must contain both R63 arms and the fixed R60B reference')
    directories = {**{'R63' + arm: root / name / 'test_predictions' for arm, name in RUN_NAMES.items()},
                   'R60B incumbent': root / 'reference/R60B/test_predictions'}
    results, predictions = {}, {}
    for name, directory in directories.items():
        results[name], saved = saved_test_predictions(directory)
        predictions[name.split()[0]] = saved
    historical_root = REFERENCE.parent
    historical_directory = historical_root / 'test_predictions/A'
    if ((historical_root / 'test_results.json').exists() and
            all((historical_directory / (p + '.npz')).exists() for p in PROBLEMS)):
        results['R45A historical'], predictions['R45A'] = saved_test_predictions(historical_directory)
        historical_locks = historical_root / 'locked_checkpoints.json'
        info.update(historical_R45A_status='reused original official full18 TEST; no rerun',
                    historical_R45A_checkpoint=load_json(historical_locks)['A'] if historical_locks.exists() else None,
                    historical_R45A_predictions=str(historical_directory))
    else:
        info['historical_R45A_status'] = 'missing original official full18 TEST predictions; not rerun'
    return results, predictions, info


def checkpoint_rows(root, records, historical_epoch, test_info):
    training_lock_path = root / 'locked_checkpoints.json'
    training_locks = load_json(training_lock_path) if training_lock_path.exists() else {}
    historical_lock_path = REFERENCE.parent / 'locked_checkpoints.json'
    historical_locks = load_json(historical_lock_path) if historical_lock_path.exists() else {}
    incumbent_record = load_json(INCUMBENT / 'result.json')
    defaults = {
        'R45A historical': dict(path=str(REFERENCE / 'best.pt'), epoch=historical_epoch),
        'R60B incumbent': dict(path=str(INCUMBENT / 'best.pt'), epoch=incumbent_record['best']['epoch'],
                              sha256=incumbent_record.get('best_checkpoint_sha256')),
        **{'R63' + arm: dict(path=str(root / name / 'best.pt'), epoch=records[arm]['best']['epoch'],
                            sha256=records[arm].get('best_checkpoint_sha256'))
           for arm, name in RUN_NAMES.items()}}
    locks = test_info['locked_models']
    rows = []
    for name, selected in defaults.items():
        arm = name[-1] if name in ('R63A', 'R63B') else 'R60B'
        lock = historical_locks.get('A') if name == 'R45A historical' else locks.get(arm)
        status = 'final TEST locked' if lock is not None else 'validation selected; final TEST lock pending'
        source = str(historical_lock_path if name == 'R45A historical' else root / 'final_test_lock.json')
        if lock is None and name in ('R63A', 'R63B') and arm in training_locks:
            lock, status, source = training_locks[arm], 'validation locked; final TEST lock pending', str(training_lock_path)
        if lock is not None:
            selected = lock
        rows.append(dict(model=name, checkpoint=selected['path'], checkpoint_epoch=selected['epoch'],
                         checkpoint_sha256=selected.get('sha256'), lock_status=status,
                         lock_source=source if lock is not None else None,
                         selection_rule='strict maximum validation native Top1; epoch ties keep earlier'
                            if name in ('R63A', 'R63B') else 'historical validation-cost selection; unchanged'))
    return rows


def family_rows(model, split, result):
    rows = []
    for family, metrics in [('ALL', result['macro']), *result['families'].items()]:
        tasks = (list(PROBLEMS) if family == 'ALL' else list(PROBLEMS[3:]) if family == 'MVRP' else [family])
        rows.append(dict(model=model, split=split, family=family,
                         n=sum(result['per_problem'][p]['n'] for p in tasks), **metrics))
    return rows


def paired_decisions(predictions, split):
    rows, family_changes = [], []
    pairs = [('R45A', 'R63A'), ('R45A', 'R63B'), ('R60B', 'R63A'),
             ('R60B', 'R63B'), ('R63A', 'R63B')]
    for before, after in pairs:
        if before not in predictions or after not in predictions:
            continue
        paired = changes(predictions[before], predictions[after], before, after)
        rows.extend(dict(split=split, **row) for row in paired)
        for family, tasks in [('TSP', ['TSP']), ('CVRP', ['CVRP']), ('ATSP', ['ATSP']),
                              ('MVRP', list(PROBLEMS[3:]))]:
            categories = dict.fromkeys(row['category'] for row in paired)
            for category in categories:
                selected = [row for row in paired if row['problem'] in tasks and row['category'] == category]
                family_changes.append(dict(split=split, before=before, after=after, family=family,
                    category=category, n=sum(row['n'] for row in selected),
                    mean_cost_delta=float(np.mean([row['mean_cost_delta'] for row in selected])),
                    actual_regret_delta_pct=float(np.mean([row['actual_regret_delta_pct'] for row in selected]))))
    return rows, family_changes


def family_table(rows):
    lines = ['| Model | Family | N | Top1 % | Top2 % | Top3 % | CE | Mean cost | vs SBS % | Regret % |',
             '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for row in rows:
        lines.append(f"| {row['model']} | {row['family']} | {row['n']} | "
                     f"{100 * row['macro_top1']:.4f} | {100 * row['macro_top2']:.4f} | "
                     f"{100 * row['macro_top3']:.4f} | {row['macro_ce']:.6f} | "
                     f"{row['macro_mean_cost']:.6f} | {row['macro_vs_sbs_pct']:+.4f} | "
                     f"{row['macro_actual_regret_pct']:.6f} |")
    return lines


def plot_training(root, histories, incumbent):
    plt = plotting()
    curves = root / 'curves'
    curves.mkdir(exist_ok=True)
    fields = ('top1', 'ce', 'mean_cost', 'actual_regret_pct', 'vs_sbs_pct', 'top2')
    rows = []
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for arm, history in histories.items():
        for row in history:
            for split in ('train', 'val'):
                if row.get(split) is not None:
                    rows.append(dict(arm=arm, epoch=row['epoch'], split=split,
                                     **row[split]['macro']))
        for ax, field in zip(axes.flat, fields):
            for split, style in (('train', '--'), ('val', '-')):
                points = [row for row in history if row.get(split) is not None]
                ax.plot([row['epoch'] for row in points],
                        [row[split]['macro']['macro_' + field] for row in points],
                        linestyle=style, label=arm + ' ' + split)
    for ax, field in zip(axes.flat, fields):
        ax.axhline(incumbent['macro']['macro_' + field], color='black',
                   linestyle=':', label='R60B selected val')
        ax.set(title=field, xlabel='Epoch')
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(root / 'learning_curves.png', dpi=150)
    plt.close(fig)
    write_csv(root / 'learning_curves.csv', rows)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    optimization_rows = []
    for arm, history in histories.items():
        for row in history:
            optimization_rows.append(dict(arm=arm, epoch=row['epoch'], **row['optimization']))
        for ax, field in zip(axes, ('ce', 'kl', 'grad_norm', 'lr')):
            points = [row for row in history if field in row['optimization']]
            ax.plot([row['epoch'] for row in points],
                    [row['optimization'][field] for row in points], label=arm)
    for ax, field in zip(axes, ('ce', 'kl', 'grad_norm', 'lr')):
        ax.set(title=field, xlabel='Epoch')
        ax.grid(alpha=.2)
        ax.legend()
    fig.tight_layout()
    fig.savefig(curves / 'optimization.png', dpi=150)
    plt.close(fig)
    write_csv(root / 'optimization_curves.csv', optimization_rows)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, family in zip(axes.flat, ('TSP', 'CVRP', 'ATSP', 'MVRP')):
        for arm, history in histories.items():
            for split, style in (('train', '--'), ('val', '-')):
                points = [row for row in history if row.get(split) is not None]
                ax.plot([row['epoch'] for row in points],
                        [row[split]['families'][family]['macro_top1'] for row in points],
                        linestyle=style, label=arm + ' ' + split)
        ax.set(title=family, xlabel='Epoch', ylabel='Native Top1')
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(curves / 'family_top1.png', dpi=150)
    plt.close(fig)
    common = min(len(history) for history in histories.values())
    write_csv(root / 'common_epoch_validation.csv',
              [dict(arm=arm, epoch=row['epoch'], **row['val']['macro'])
               for arm, history in histories.items() for row in history[:common]])


def inference_rows(root):
    rows = []
    for arm, name in RUN_NAMES.items():
        timing = load_json(root / name / 'inference_timing.json')
        problems = timing['per_problem']
        for problem, groups in [(key, [value]) for key, value in problems.items()] + [
                ('ALL', list(problems.values()))]:
            first = [group['single_queries'][0]['total_seconds'] for group in groups]
            subsequent = [row['total_seconds'] for group in groups
                          for row in group['single_queries'][1:]]
            batched_queries = sum(group['remaining_batched']['queries'] for group in groups)
            batched_seconds = sum(group['remaining_batched']['total_seconds'] for group in groups)
            rows.append(dict(arm=arm, problem=problem, checkpoint_epoch=timing['checkpoint_epoch'],
                queries=sum(group['queries'] for group in groups),
                first_single_queries=len(first), first_single_median_ms=1000 * float(np.median(first)),
                subsequent_single_queries=len(subsequent),
                subsequent_single_median_ms=1000 * float(np.median(subsequent)),
                subsequent_single_p95_ms=1000 * float(np.quantile(subsequent, .95)),
                remaining_batched_queries=batched_queries,
                batch_size=groups[0]['remaining_batched']['batch_size'],
                remaining_batched_seconds=batched_seconds,
                batched_queries_per_second=batched_queries / batched_seconds,
                graph_construction_seconds=sum(group['graph_construction_seconds'] for group in groups),
                selector_forward_wall_seconds=sum(group['selector_forward_wall_seconds'] for group in groups),
                total_selection_seconds=sum(group['total_seconds'] for group in groups)))
    write_csv(root / 'inference_timing.csv', rows)
    return rows


def summarize(root=ROOT):
    root = Path(root)
    records = {arm: load_json(root / name / 'result.json')
               for arm, name in RUN_NAMES.items()}
    histories = {arm: load_json(root / name / 'history.json')
                 for arm, name in RUN_NAMES.items()}
    raw = {problem: read_raw_costs(problem, 'val') for problem in PROBLEMS}
    historical = load_json(REFERENCE / 'best_eval.json')
    reference, reference_predictions = replay_directory(
        REFERENCE / 'val_predictions' / f"epoch{historical['epoch']:03d}", raw)
    incumbent, incumbent_predictions = replay_directory(
        INCUMBENT / 'val_predictions' / 'best', raw)
    results, predictions = {}, {}
    for arm, name in RUN_NAMES.items():
        results[arm], predictions[arm] = replay_directory(
            root / name / 'val_predictions' / 'best', raw)
    all_results = {'R45A historical': reference, 'R60B incumbent': incumbent,
                   **{'R63' + arm: result for arm, result in results.items()}}
    test_results, test_predictions, test_info = completed_test_outputs(root)
    split_results = {'val': all_results}
    if test_results:
        split_results['test'] = test_results
        dump(root / 'test_replay_results.json', test_results)
    checkpoints = checkpoint_rows(root, records, historical['epoch'], test_info)
    checkpoint_by_model = {row['model']: row for row in checkpoints}
    rows = [dict(row, split=split,
                 checkpoint=checkpoint_by_model[name]['checkpoint'],
                 checkpoint_epoch=checkpoint_by_model[name]['checkpoint_epoch'],
                 checkpoint_sha256=checkpoint_by_model[name]['checkpoint_sha256'])
            for split, evaluations in split_results.items() for name, result in evaluations.items()
            for row in comparison_rows(name, result)]
    families = [row for split, evaluations in split_results.items() for name, result in evaluations.items()
                for row in family_rows(name, split, result)]
    dump(root / 'validation_results.json', results)
    dump(root / 'final_test_status.json', test_info)
    write_csv(root / 'selected_checkpoints.csv', checkpoints)
    write_csv(root / 'comparison.csv', rows)
    write_csv(root / 'family_comparison.csv', families)
    distributions = []
    for split, evaluations in split_results.items():
        curve_root = root / 'curves' if split == 'val' else root / 'curves/test'
        for name, result in evaluations.items():
            plot_methods(name, result, curve_root / name.replace(' ', '_') / 'per_problem')
            for problem, metrics in result['per_problem'].items():
                for solver in metrics['pool']:
                    distributions.append(dict(model=name, split=split, problem=problem, solver=solver,
                        pick_fraction=metrics['arm_distribution'][solver],
                        native_winner_fraction=metrics['method_top1'][solver]))
    write_csv(root / 'solver_distributions.csv', distributions)
    validation_predictions = dict(R45A=reference_predictions, R60B=incumbent_predictions,
                                 **{'R63' + arm: saved for arm, saved in predictions.items()})
    changed, family_changed = paired_decisions(validation_predictions, 'val')
    if test_predictions:
        test_changed, test_family_changed = paired_decisions(test_predictions, 'test')
        changed += test_changed
        family_changed += test_family_changed
    write_csv(root / 'decision_changes.csv', changed)
    write_csv(root / 'family_decision_changes.csv', family_changed)
    decision = direction_decision(incumbent['macro'],
                                 {arm: result['macro'] for arm, result in results.items()})
    budgets = {arm: dict(epochs=record['epochs_completed'],
                        successful_updates=record['successful_updates'])
               for arm, record in records.items()}
    paired_horizon = budgets['A'] == budgets['B']
    amendment_path = root / 'budget_amendment.json'
    decision.update(training_budgets=budgets, paired_horizon=paired_horizon,
                    budget_amendment=load_json(amendment_path) if amendment_path.exists() else None,
                    B_minus_A_contrast_scope=('matched training horizon' if paired_horizon else
                        'descriptive unequal-budget A40/B15 contrast; not a matched-horizon mechanism claim'))
    official_arm = decision['best_arm_by_native_top1']
    decision.update(official_arm=official_arm,
                    official_arm_selection='maximum validation native Top1; ties A; never TEST',
                    final_test_status=test_info['status'], locked_checkpoints=checkpoints,
                    official_test_native_top1=None, official_test_target_hit=None,
                    official_test_gain_vs_incumbent_pp=None, test_B_minus_A_top1_pp=None,
                    goal_claim='Not declared from validation or auxiliary metrics; locked TEST target is reported separately.')
    if test_results:
        official_test = test_results['R63' + official_arm]['macro']['macro_top1']
        decision.update(heldout_required=False, official_test_native_top1=official_test,
                        official_test_target_hit=bool(official_test >= .70 - 1e-12),
                        official_test_gain_vs_incumbent_pp=100 * (
                            official_test - test_results['R60B incumbent']['macro']['macro_top1']),
                        test_B_minus_A_top1_pp=100 * (test_results['R63B']['macro']['macro_top1']
                                                     - test_results['R63A']['macro']['macro_top1']))
    dump(root / 'direction_decision.json', decision)
    plot_training(root, histories, incumbent)
    latency = inference_rows(root)

    lines = ['# R63: Native-Winner Classification With Recurrent Node Workspaces', '',
        'Seed2; original18 tasks, all legal candidates and original native winners.',
        'Training, checkpoint selection and the official arm are controlled by validation only.',
        'Completed locked TEST is reported separately; it never reselects a checkpoint or arm.',
        'No auxiliary data, solver execution, new annotation or R58 operations.', '',
        '## Validation Top1-selected outcomes', '']
    lines += table([row for row in rows if row['split'] == 'val' and row['problem'] == 'ALL'])
    lines += ['', '## Checkpoint identities', '',
        '| Model | Checkpoint | Epoch | SHA256 | Lock status |',
        '|---|---|---:|---|---|']
    for row in checkpoints:
        lines.append(f"| {row['model']} | `{row['checkpoint']}` | {row['checkpoint_epoch']} | "
                     f"{row['checkpoint_sha256'] or 'not recorded'} | {row['lock_status']} |")
    lines += ['', f'Official R63 arm: **R63{official_arm}**, chosen by validation native Top1; ties A.',
        'Each R63 arm retains its own unique validation-best checkpoint within its actual completed budget.',
        'R60B remains its existing validation-cost-best checkpoint (epoch39); it is not reselected.',
        'R45A uses its historical official TEST and original checkpoint lock; no rerun.', '',
        'Numerical tie caveat: A epochs32 and34 both selected8729/18000 native winners',
        'correctly. Floating-point macro averaging produced0.4849444444444444 versus',
        '0.4849444444444445, so the original strict-float comparison selected34 rather',
        'than honoring the intended earliest-on-exact-count-tie rule. A34 was locked',
        'before TEST and remains the reported checkpoint; no epoch32 TEST or reselection',
        'is performed. R64 will fix this prospectively using integer correct counts.',
        'This affects the tie rule, not the reported R63 Top1 or its negative outcome.', '',
        '## Locked final TEST', '', f"Final TEST receipt status: **{test_info['status']}**.",
        'Both training arms must finish before the parent freezes final_test_lock.json and authorizes TEST.',
        'Exactly one complete original18 TEST per R63A/R63B/R60B model; no TEST-based re-selection.',
        'The analysis reads only saved FP64 predictions and cannot initiate or repeat inference.']
    if test_results:
        lines += [''] + table([row for row in rows if row['split'] == 'test' and row['problem'] == 'ALL'])
        lines += ['', f"Official R63{official_arm} TEST native Top1: {100 * decision['official_test_native_top1']:.4f}%.",
            f"Official locked TEST70% target hit: **{decision['official_test_target_hit']}**.",
            f"Official minus R60B TEST Top1: {decision['official_test_gain_vs_incumbent_pp']:+.4f}pp.",
            f"R63B minus R63A TEST Top1: {decision['test_B_minus_A_top1_pp']:+.4f}pp.",
            f"R45A reference: {test_info['historical_R45A_status']}."]
    else:
        lines += ['', 'TEST metrics and contrasts are unavailable, not zero and not copied from validation.',
            'Pending/failed TEST does not authorize an automatic retry or alter validation selection.']
    lines += ['', 'The fixed R60B reference TEST is the explicit addition; other R59--R62 no-test histories remain unchanged.',
        'Costs/regret are disclosed outcomes, never a veto on native Top1 or a basis for TEST re-selection.', '',
        '## Validation families', '']
    lines += family_table([row for row in families if row['split'] == 'val'])
    if test_results:
        lines += ['', '## TEST families', ''] + family_table([row for row in families if row['split'] == 'test'])
    lines += ['', 'Each candidate has its own persistent node workspace and eight tied recurrent rounds.',
        'A reads initial neighbor states; B reads previous-round neighbor states. Both update their own states.',
        'Graphs use input distances/direction, valid-node masks, depot and special-token connections only.',
        'Both train native CE plus symmetric R-Drop and select strict maximum full18 native Top1.',
        'No maximin, comparison/risk loss, route output or independent binary expert is used.', '',
        '## Actual training budgets', '',
        '| Arm | Best epoch | Stop epoch | Successful updates | Run hours |',
        '|---|---:|---:|---:|---:|']
    for arm, record in records.items():
        lines.append(f"| R63{arm} | {record['best']['epoch']} | {record['epochs_completed']} | "
                     f"{record['successful_updates']} | {record['wall_seconds'] / 3600:.4f} |")
    lines += ['', ('A controlled the common native-Top1 learning-rate trace and stopping horizon.'
                   if paired_horizon else
                   'User amendment: A completed40 epochs; B stopped after15 complete epochs and full validation.'),
        ('Both arms completed the common horizon.' if paired_horizon else
         'Only the first15 epochs share batches and the A-controlled LR trace; A40/B15 is not a matched-compute contrast.'),
        'Effective batch128 uses fixed microbatch16 with sample-weighted accumulation.',
        'The two dropout forwards and all microbatches produce one optimizer update per original batch.',
        'Architecture parameters, initialization, common-prefix batches, graph construction and recurrence count are matched.', '',
        '## Common-prefix validation (descriptive only)', '',
        '| Arm | Prefix epochs | Last-prefix Top1 % | Best-prefix epoch | Best-prefix Top1 % |',
        '|---|---:|---:|---:|---:|']
    prefix_epochs = min(record['epochs_completed'] for record in records.values())
    for arm, history in histories.items():
        prefix = history[:prefix_epochs]
        prefix_best = max(prefix, key=lambda row: row['val']['macro']['macro_top1'])
        lines.append(f"| R63{arm} | {prefix_epochs} | {100 * prefix[-1]['val']['macro']['macro_top1']:.4f} | "
                     f"{prefix_best['epoch']} | {100 * prefix_best['val']['macro']['macro_top1']:.4f} |")
    lines += ['', 'These existing history values do not reselect, replace or evaluate any checkpoint.', '',
        '## Selected-checkpoint train and validation', '',
        '| Arm | Split | Top1 % | CE | Mean cost | Actual regret % |',
        '|---|---|---:|---:|---:|---:|']
    for arm, record in records.items():
        for split in ('train', 'val'):
            metrics = record['best'][split]['macro']
            lines.append(f"| R63{arm} | {split} | {100 * metrics['macro_top1']:.4f} | "
                         f"{metrics['macro_ce']:.6f} | {metrics['macro_mean_cost']:.6f} | "
                         f"{metrics['macro_actual_regret_pct']:.6f} |")
    lines += ['', 'Training evaluation uses eval mode and complete original TRAIN, not online batch accuracy.', '',
        '## Decision and limits', '',
        f"Best selected arm by native Top1: R63{decision['best_arm_by_native_top1']}.",
        f"Validation70% target hit: **{decision['validation_target_hit']}**; validation alone does not establish the target.",
        f"Best minus incumbent Top1: {decision['best_gain_vs_incumbent_pp']:+.4f}pp.",
        f"B minus A Top1: {decision['B_minus_A_top1_pp']:+.4f}pp.",
        f"Contrast scope: {decision['B_minus_A_contrast_scope']}.",
        f"Best minus incumbent regret: {decision['best_regret_delta_vs_incumbent_pp']:+.6f}pp.", '',
        'Costs/regret are disclosed, not an added veto on the user native-Top1 target.',
        'The+3pp progress and B-A+2pp mechanism milestones do not replace70%.',
        'Either arm may satisfy the validation target; unequal A40/B15 results do not establish a matched-budget communication advantage.',
        'The official arm is frozen by validation before TEST; a better contrast-arm TEST does not change it.',
        'The70% goal is not declared achieved by validation alone; official locked TEST target status is separate.',
        'Comparison with historical cost-trained models changes objective/readout as well as computation.',
        'This single-seed experiment with disclosed actual budgets cannot establish a universal accuracy ceiling.',
        'Legacy B semantic mismatches remain unresolved; this is not a repaired common-task benchmark.', '',
        '## Selection latency', '',
        'Per-arm inference_timing.json records normal final-validation timing, including graph construction.',
        'Inputs and geometry are already cached. Disk decoding, cache preparation and checkpoint loading are excluded.',
        'Each original query is evaluated once: first16 per task at batch1, then the remainder batched.',
        'Single-query times include gather, graph construction, selector forward and output transfer.',
        'ALL single-query quantiles pool the same number of observations per task, not per scale.',
        'No solver is run for selection; selector time is not full solver or end-to-end solve time.', '',
        '| Arm | First query median ms | Later batch1 median / p95 ms | Batched queries/s | Graph seconds | Forward seconds |',
        '|---|---:|---:|---:|---:|---:|']
    for row in latency:
        if row['problem'] == 'ALL':
            lines.append(f"| R63{row['arm']} | {row['first_single_median_ms']:.3f} | "
                         f"{row['subsequent_single_median_ms']:.3f} / {row['subsequent_single_p95_ms']:.3f} | "
                         f"{row['batched_queries_per_second']:.2f} | {row['graph_construction_seconds']:.3f} | "
                         f"{row['selector_forward_wall_seconds']:.3f} |")
    lines += ['', 'Full per-task timing is saved in inference_timing.csv; no standalone timing pass was added.', '',
        '## Decision changes', '',
        '| Split | Before | After | Category | N | Macro cost delta | Macro regret delta pp |',
        '|---|---|---|---|---:|---:|---:|']
    for row in changed:
        if row['problem'] == 'ALL':
            lines.append(f"| {row['split']} | {row['before']} | {row['after']} | {row['category']} | {row['n']} | "
                         f"{row['mean_cost_delta']:+.8f} | {row['actual_regret_delta_pct']:+.6f} |")
    lines += ['', 'Cost and regret deltas retain equal-problem macro weights, not raw-cost or sample-count weighting.',
        'Positive cost/regret deltas mean the after-model is worse. Both-wrong cheaper/dearer/equal changes are included.',
        'Per-task changes are in decision_changes.csv; TSP/CVRP/ATSP/MVRP changes are in family_decision_changes.csv.', '',
        '## Validation per problem', '']
    lines += table([row for row in rows if row['split'] == 'val' and row['problem'] != 'ALL'])
    if test_results:
        lines += ['', '## TEST per problem', '']
        lines += table([row for row in rows if row['split'] == 'test' and row['problem'] != 'ALL'])
    (root / 'RESULTS.md').write_text('\n'.join(lines) + '\n')
    return decision


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    print(json.dumps(summarize(parser.parse_args().root), indent=2))
