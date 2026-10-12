"""Future ONCE CPU reporting from saved JSON/updates; one new B, reused R68A."""

import argparse
import json
import os
import time
import traceback
from collections import Counter
from pathlib import Path

import numpy as np

from code.unified_selector.registry import PROBLEMS
from .multitask_probe import file_hash
from .r59_experiment import atomic_json
from .r64_analysis import METRICS, write_csv
from .r65_analysis import metrics, now, read
from .r66_analysis import metric_rows
from .r67_analysis import normalize_counts, read_updates
from .r68_analysis import fixed_references as historical_references, numeric_summary, selected_figures
from .r69_experiment import CONTROL_ROOT, ROOT, RUN_NAME
from .r69_frozen_evaluation import CONTROL_RUN, completed_test


MODEL_NAMES = {'A': 'Reused_R68A_shared', 'B': 'R69B_task_specific_new'}
LABELS = {'A': 'Reused R68A shared', 'B': 'New R69B task-specific'}


def fixed_references(root):
    selected, provenance = historical_references(root)
    lock, test = read(CONTROL_ROOT/'final_test_lock.json'), read(CONTROL_ROOT/'test_results.json')
    path = CONTROL_ROOT/'B_sam_adamw_seed2/result.json'
    result = read(path)
    selected['R68B'] = dict(epoch=lock['models']['B']['epoch'], train=result['best']['train'],
                           val=result['best']['val'], test=test['models']['B'])
    provenance['R68B'] = dict(result_path=str(path), result_sha256=file_hash(path),
        test_path=str(CONTROL_ROOT/'test_results.json'), test_json_pointer='/models/B',
        lock_path=str(CONTROL_ROOT/'final_test_lock.json'), model_json_pointer='/models/B',
        checkpoint_policy='unchanged original R68 validation lock; historical SAM candidate',
        parameter_mode='ordinary AdamW weights', historical_test_exposure=True,
        new_inference=False, reselection=False)
    normalize_counts(selected)
    return selected, provenance


def actual_compute(root, protocol, results, updates, test, test_terminal):
    rows, result = updates['B'], results['B']
    active, selected_rows, unused = Counter(), Counter(), Counter()
    for row in rows:
        for task, count in zip(row['tasks'], row['task_rows']):
            active[task] += 1
            selected_rows[task] += count
        for branch_id, names in row['unused_parameters_active_branches'].items():
            unused.update(f'branches.{branch_id}.{name}' for name in names)
    counters = read(root/RUN_NAME/'parameter_step_counts.json')
    branch_table = []
    for index, task in enumerate(PROBLEMS):
        values = {name: step for name, step in counters['steps'].items() if name.startswith(f'branches.{index}.')}
        branch_table.append(dict(problem=task, branch_id=index, active_updates=active[task], selected_rows=selected_rows[task],
            successful_view_rows=2*selected_rows[task], parameter_tensors=len(values),
            adam_step_min=min(values.values()), adam_step_max=max(values.values()),
            zero_step_tensors=sum(step == 0 for step in values.values()),
            exact_active_counter_match=all(step == active[task]-unused[name] for name, step in values.items())))
    counts = read(root/'parameter_counts.json')
    initial, bank_initial = read(root/'initialization.json'), read(root/RUN_NAME/'initialization.json')
    historical_initial = read(CONTROL_ROOT/'initialization.json')
    historical_schedule = read(CONTROL_ROOT/'schedule_seed2.json')
    shared_schedule = all(protocol['schedule_identity'][key] == historical_schedule[key] for key in
        ('original80epoch_sha256', 'selected_stage2_sha256', 'global_epochs', 'dropout_successful_offset'))
    log_schedule = len(rows) == len(updates['A']) == 9600 and all(
        all(left[key] == right[key] for key in ('tasks', 'task_rows', 'batch_indices_sha256',
                                               'dropout_seed', 'lr', 'consistency_weight'))
        for left, right in zip(rows, updates['A']))
    inherited_path = Path(protocol['inherited_identity']['terminal']['path']).parent/'stage1_result.json'
    inherited = read(inherited_path)
    success_views = sum(row['successful_view_rows'] for row in rows)
    attempted_views = sum(row['attempted_view_rows'] for row in rows)
    retries = sum(row['retries'] for row in rows)
    checks = dict(
        exact_budget=len(rows) == result['successful_updates'] == 9600
            and sum(row['rows'] for row in rows) == result['presentations'] == 7200000
            and success_views == result['successful_view_rows'] == 14400000
            and attempted_views == result['attempted_view_rows'] and retries == result['amp_retries']
            and [row['successful_update'] for row in rows] == list(range(1, 9601)),
        finite_updates=all(row['finite_loss'] and row['finite_gradient'] for row in rows),
        one_step_and_clip=all(row['base_optimizer_steps'] == row['active_parameter_weight_decay_steps']
                              == row['global_clip_calls'] == 1 for row in rows),
        inactive_grad_and_steps=all(row['inactive_grad_none'] and row['inactive_parameter_steps'] == 0
            and row['active_branch_ids'] == [PROBLEMS.index(task) for task in row['tasks']]
            and set(row['inactive_branch_ids']) == set(range(18))-set(row['active_branch_ids']) for row in rows),
        two_views_and_tails=all(row['successful_view_rows'] == 2*row['rows']
            and row['task_rows'] == ([256]*3 if row['update_in_epoch'] <= 234 else [16]*3)
            and row['attempted_view_rows'] >= row['successful_view_rows'] for row in rows),
        branch_budget=all(item['active_updates'] == 1600 and item['selected_rows'] == 400000
            and item['exact_active_counter_match'] for item in branch_table)
            and counters['branch_activations'] == dict(active),
        identical_prefinetuning_state=initial['starting_selector_hash'] == historical_initial['starting_selector_hash']
            and bank_initial['branch_hashes'] == [initial['starting_selector_hash']]*18
            and initial['stage2_head_seed'] == 670003 and initial['entire_fresh_head_copied'] is True,
        same_saved_schedule=shared_schedule, same_actual_task_rows_dropout_lr=log_schedule,
        parameter_counts=counts['B'] == counts['trainable_B'] == 33173154 and counts['shared'] == 1842953,
        inherited_once=inherited['successful_updates'] == 9600 and inherited['presentations'] == 7200000,
        focused_check_record=read(root/'cpu_check.json')['passed'] is True)
    new_B = dict(successful_updates=len(rows), selected_rows=sum(row['rows'] for row in rows),
        successful_view_rows=success_views, attempted_view_rows=attempted_views, amp_retries=retries,
        wall_seconds=result['wall_seconds'], logged_update_seconds=sum(row['seconds'] for row in rows),
        training_epoch_seconds=sum(row['optimization']['seconds'] for row in read(root/RUN_NAME/'history.json')),
        peak_allocated_gib=max(row['peak_memory_gib'] for row in rows),
        peak_reserved_gib=max(row['peak_reserved_gib'] for row in rows),
        peak_physical_gib=max(row['physical_used_gib'] for row in rows),
        grad_norm=numeric_summary([row['grad_norm'] for row in rows]),
        clip_coefficient=numeric_summary([row['clip_coefficient'] for row in rows]),
        amp_scale=numeric_summary([row['scale'] for row in rows]),
        complete_test_evaluation_seconds=test['models']['B']['complete_evaluation_seconds'], branches=branch_table)
    control = results['A']
    reused_A = dict(successful_updates=control['successful_updates'], selected_rows=control['presentations'],
        successful_view_rows=control['successful_view_rows'], attempted_view_rows=control['attempted_view_rows'],
        wall_seconds=control['wall_seconds'], amp_retries=control['amp_retries'],
        new_R69_training_compute=0, new_R69_test_compute=0, accounting='historical reused R68A; not a newly executed R69 arm',
        original_test_already_executed=1, new_R69_tests=0,
        historical_test_evaluation_seconds=test['models']['A']['complete_evaluation_seconds'])
    if (reused_A['successful_updates'], reused_A['selected_rows'], reused_A['amp_retries']) != (9600, 7200000, 5) or not np.isclose(reused_A['wall_seconds'], 3002.024237, rtol=0, atol=1e-5):
        raise ValueError('Retain actual fixed R68A historical expenditure separately')
    compute = dict(new_arms=['B'], reused_arms=['A'], new_B=new_B, historical_reused_A=reused_A,
        inherited_stage1_once=dict(successful_updates=inherited['successful_updates'], selected_rows=inherited['presentations'],
            wall_seconds=inherited['wall_seconds'], amp_retries=inherited['amp_retries'], counted_once=True,
            source='R67B terminal stage1 averaged-x trunk; entire fresh head670003', result_path=str(inherited_path),
            result_sha256=file_hash(inherited_path), new_compute=0),
        planned_new_B=dict(epochs=40, updates=9600, selected_rows=7200000, successful_view_rows=14400000,
                           active_updates_per_branch=1600, selected_rows_per_branch=400000),
        parameter_counts=counts, parameter_capacity_matched=False, per_parameter_optimizer_histories_matched=False,
        saved_initialization=initial, saved_schedule_identity=protocol['schedule_identity'], checks=checks,
        training_terminal=read(root/'training_terminal_receipt.json'), test_terminal=test_terminal,
        inherited_once_plus_new_B_updates=inherited['successful_updates']+len(rows),
        inherited_once_plus_new_B_selected_rows=inherited['presentations']+new_B['selected_rows'],
        new_test_wall_seconds=test['wall_seconds'], solver_executions_during_delivery=0,
        deployment_selector_forwards_per_query=1, deployment_completed_solvers_per_query=1,
        timing_scope='ordinary complete evaluation walltime, not singleton latency; A time is historical',
        pure_task_interference_causal_claim=False, fresh_encoders=False)
    return compute, branch_table, counters


def export_curves(root, histories, results, updates):
    learning, tasks, families, optimization, task_opt, update_rows = [], [], [], [], [], []
    for arm, history in histories.items():
        for row in history:
            identity = dict(arm=arm, model=MODEL_NAMES[arm], execution='reused R68A' if arm == 'A' else 'new R69B',
                            stage=row['stage'], epoch=row['epoch'], global_epoch=row['global_epoch'])
            optimization.append(dict(identity, objective='native CE plus legal RDrop KL',
                successful_updates=row['successful_updates'], presentations=row['presentations'],
                cumulative_successful_view_rows=row['successful_view_rows'],
                cumulative_attempted_view_rows=row['attempted_view_rows'], **row['optimization']))
            task_opt += [dict(identity, problem=task, **entry) for task, entry in row['task_optimization'].items()]
            for split in ('train', 'val'):
                value = row.get(split)
                if value is None:
                    continue
                learning += [dict(identity, evaluation='actual epoch', **entry)
                             for entry in metric_rows(MODEL_NAMES[arm], split, row['epoch'], value)]
                tasks += [dict(identity, split=split, problem=task, **metrics(entry)) for task, entry in value['per_problem'].items()]
                families += [dict(identity, split=split, family=family, **metrics(entry, 'macro_'))
                             for family, entry in value['families'].items()]
        best = results[arm]['best']
        learning += [dict(entry, arm=arm, execution='reused R68A' if arm == 'A' else 'new R69B',
                         evaluation='actual selected best', global_epoch=best['global_epoch'])
                     for entry in metric_rows(MODEL_NAMES[arm], 'train', best['epoch'], best['train'])]
        for row in updates[arm]:
            entry = dict(row, model=MODEL_NAMES[arm], execution='reused R68A' if arm == 'A' else 'new R69B')
            update_rows.append({key: json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else value
                                for key, value in entry.items()})
    for name, rows in (('learning_curves', learning), ('task_curves', tasks), ('family_curves', families),
                       ('optimization_curves', optimization), ('task_optimization', task_opt), ('update_curves', update_rows)):
        write_csv(root/(name+'.csv'), rows)
    return tasks, families


def validation_summaries(root, results, histories):
    summaries, last5, common = [], [], []
    for arm, history in histories.items():
        for label in ('best', 'final'):
            record = results[arm][label]
            for split in ('train', 'val'):
                if record.get(split) is not None:
                    summaries += [dict(arm=arm, execution='reused R68A' if arm == 'A' else 'new R69B', summary=label,
                        global_epoch=record['global_epoch'], **entry)
                        for entry in metric_rows(MODEL_NAMES[arm], split, record['epoch'], record[split])]
        grouped = {}
        for row in history[-5:]:
            for entry in metric_rows(MODEL_NAMES[arm], 'val', row['epoch'], row['val']):
                last5.append(dict(arm=arm, **entry))
                grouped.setdefault((entry['kind'], entry['scope']), []).append(entry)
        for (kind, scope), rows in grouped.items():
            for label, function in (('last5_mean', np.mean), ('last5_population_sd', np.std), ('last5_min', np.min), ('last5_max', np.max)):
                summaries.append(dict(arm=arm, model=MODEL_NAMES[arm], split='val', kind=kind, scope=scope,
                    summary=label, actual_epochs='36,37,38,39,40', not_seed_uncertainty=True,
                    **{key: float(function([row[key] for row in rows])) for key in (*METRICS, 'native_correct', 'native_total')}))
    for split in ('train', 'val'):
        available = {arm: {row['epoch']: row[split] for row in history if row.get(split) is not None}
                     for arm, history in histories.items()}
        for arm in ('A', 'B'):
            record = results[arm]['best']
            if record.get(split) is not None:
                available[arm].setdefault(record['epoch'], record[split])
        for epoch in sorted(set(available['A']) & set(available['B'])):
            left = {(row['kind'], row['scope']): row for row in metric_rows(MODEL_NAMES['A'], split, epoch, available['A'][epoch])}
            for right in metric_rows(MODEL_NAMES['B'], split, epoch, available['B'][epoch]):
                a = left[right['kind'], right['scope']]
                common.append(dict(epoch=epoch, global_epoch=40+epoch, split=split, kind=right['kind'], scope=right['scope'],
                    A_model=MODEL_NAMES['A'], B_model=MODEL_NAMES['B'], A_native_correct=a['native_correct'],
                    B_native_correct=right['native_correct'], native_total=right['native_total'],
                    **{arm+'_'+key: row[key] for arm, row in (('A', a), ('B', right)) for key in METRICS},
                    **{key+('_B_minus_A_pp' if key.startswith('top') else '_B_minus_A'):
                       (right[key]-a[key])*(100 if key.startswith('top') else 1) for key in METRICS}))
    write_csv(root/'stage2_summary.csv', summaries)
    write_csv(root/'last5_validation.csv', last5)
    write_csv(root/'matched_common_epochs.csv', common)


def figures(plt, curves, histories, updates, task_rows, family_rows):
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    for ax, metric in zip(axes.flat, METRICS):
        for arm, history in histories.items():
            for split, style in (('train', '--'), ('val', '-')):
                points = [row for row in history if row.get(split) is not None]
                ax.plot([row['epoch'] for row in points], [row[split]['macro']['macro_'+metric] for row in points],
                        linestyle=style, marker='.' if split == 'train' else None, label=LABELS[arm]+' '+split)
        ax.set(title=metric, xlabel='Fine-tuning epoch (global41-80)')
        ax.grid(alpha=.2)
        ax.legend(fontsize=7)
    axes.flat[-1].axis('off')
    fig.tight_layout()
    fig.savefig(curves/'learning_curves.png', dpi=150)
    plt.close(fig)
    keys = ('loss', 'ce', 'kl', 'lr', 'grad_norm', 'clip_coefficient', 'amp_retries', 'peak_memory_gib',
            'successful_view_rows', 'attempted_view_rows', 'seconds', 'scale')
    for resolution in ('epoch', 'update'):
        source = {arm: [dict(row['optimization'], epoch=row['epoch']) for row in history]
                  if resolution == 'epoch' else updates[arm] for arm, history in histories.items()}
        xkey = 'epoch' if resolution == 'epoch' else 'successful_update'
        fig, axes = plt.subplots(3, 4, figsize=(18, 11))
        for ax, key in zip(axes.flat, keys):
            actual_key = 'retries' if resolution == 'update' and key == 'amp_retries' else key
            for arm, rows in source.items():
                points = [row for row in rows if actual_key in row]
                if points:
                    ax.plot([row[xkey] for row in points], [row[actual_key] for row in points], label=LABELS[arm], linewidth=.8)
            ax.set(title=key, xlabel=xkey)
            ax.grid(alpha=.2)
            if ax.lines:
                ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(curves/('optimization_'+resolution+'.png'), dpi=150)
        plt.close(fig)
    for rows, field, columns in ((task_rows, 'problem', 3), (family_rows, 'family', 2)):
        names = list(dict.fromkeys(row[field] for row in rows))
        fig, axes = plt.subplots((len(names)+columns-1)//columns, columns,
                                 figsize=(6*columns, 3*((len(names)+columns-1)//columns)), squeeze=False)
        for ax, name in zip(axes.flat, names):
            for arm in ('A', 'B'):
                for split, style in (('train', '--'), ('val', '-')):
                    points = [row for row in rows if row[field] == name and row['arm'] == arm and row['split'] == split]
                    ax.plot([row['epoch'] for row in points], [row['top1'] for row in points],
                            linestyle=style, label=LABELS[arm]+' '+split)
            ax.set(title=name, xlabel='Actual epoch', ylabel='Native Top1')
            ax.legend(fontsize=6)
            ax.grid(alpha=.2)
        for ax in list(axes.flat)[len(names):]:
            ax.axis('off')
        fig.tight_layout()
        fig.savefig(curves/(field+'_top1.png'), dpi=150)
        plt.close(fig)


def comparison_tables(root, selected):
    comparison = [row for model, record in selected.items() for split in ('train', 'val', 'test')
        if record.get(split) is not None for row in metric_rows(model, split, record['epoch'], record[split])]
    write_csv(root/'comparison.csv', comparison)
    write_csv(root/'selected_metrics.csv', [row for row in comparison if row['model'] in MODEL_NAMES.values()])
    write_csv(root/'metric_availability.csv', [dict(model=model, split=split, available=record.get(split) is not None,
        policy='actual fixed saved reports only; missing TRAIN stays missing; no interpolation/inference/replay')
        for model, record in selected.items() for split in ('train', 'val', 'test')])
    index = {(row['model'], row['split'], row['kind'], row['scope']): row for row in comparison}
    pairs = [(MODEL_NAMES['B'], MODEL_NAMES['A'])]+[(model, ref) for model in MODEL_NAMES.values()
        for ref in selected if ref not in MODEL_NAMES.values()]
    deltas, gains = [], []
    for left, right in pairs:
        for split in ('train', 'val', 'test'):
            task_signs = Counter()
            for kind, scope in [('aggregate', 'ALL')]+[('family', family) for family in ('TSP', 'CVRP', 'ATSP', 'MVRP')]+[('task', task) for task in PROBLEMS]:
                if (left, split, kind, scope) not in index or (right, split, kind, scope) not in index:
                    continue
                a, b = index[left, split, kind, scope], index[right, split, kind, scope]
                delta = dict(comparison=left+' minus '+right, left_model=left, right_model=right,
                    split=split, kind=kind, scope=scope, native_correct_delta=a['native_correct']-b['native_correct'],
                    **{key+('_delta_pp' if key.startswith('top') else '_delta'):
                       (a[key]-b[key])*(100 if key.startswith('top') else 1) for key in METRICS})
                deltas.append(delta)
                if kind == 'task':
                    task_signs['gains' if delta['native_correct_delta'] > 0 else 'regressions' if delta['native_correct_delta'] < 0 else 'ties'] += 1
            if task_signs:
                gains.append(dict(left_model=left, right_model=right, split=split,
                    **{key: task_signs[key] for key in ('gains', 'regressions', 'ties')}))
    write_csv(root/'comparison_deltas.csv', deltas)
    write_csv(root/'task_gain_regression.csv', gains)
    return comparison, deltas, gains


def report(root, selected, lock, decision, compute, gains):
    official = lock['official_arm']
    lines = ['# R69 Locked Delivery', '',
        f"Validation-frozen official model: {LABELS[official]}, epoch{lock['models'][official]['epoch']}, global{lock['models'][official]['global_epoch']}. Original18 TEST: {decision['official_test_correct']}/18000 ({100*decision['official_test_top1']:.6f}% native Top1).", '',
        'R69 executed ONE new arm: B, the complete18-branch ordinary AdamW bank. A is explicitly reused R68A28/global68, fixed at8644/18000 validation correct. A received no new training, weights load, recount, inference, reselection or TEST. Its one original historical TEST and saved TRAIN/VAL result/curves are reused unchanged. Reuse establishes protocol equivalence, not an independent replication.', '',
        'B is selected by maximum INTEGER full18 validation native correct/18000 with the earliest exacttie. All18 branches come from ONE common epoch. Official B requires strictly>8644 correct; tie/lower retains fixed A. Costs and TEST do not select; individual task gains do not choose branch checkpoints. R60B is a descriptive fixed historical incumbent, not the matched control or a reselected model.', '',
        '## Selected Metrics', '',
        '| Model | Split | Top1 % | Top2 % | Top3 % | CE | Mean Cost | vsSBS % | Actual Regret % |',
        '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for model, record in selected.items():
        for split in ('train', 'val', 'test'):
            if record.get(split) is not None:
                value = metrics(record[split]['macro'], 'macro_')
                lines.append(f"| {model} | {split} | {100*value['top1']:.4f} | {100*value['top2']:.4f} | {100*value['top3']:.4f} | {value['ce']:.6f} | {value['mean_cost']:.6f} | {value['vs_sbs_pct']:.6f} | {value['actual_regret_pct']:.6f} |")
    lines += ['',
        'comparison.csv/selected_metrics.csv contain full task/family/ALL selected TRAIN/VAL/TEST. ALL is the original equal18-task macro; MVRP includes15 tasks. vsSBS is100*(selector mean cost/SBS cost-1); actual regret is mean per-query relative excess over the original legal-pool oracle. Cross-task mean costs mix task units and are descriptive. Positive accuracy deltas are improvements; positive CE/cost/regret deltas are worse.', '',
        'comparison_deltas.csv and task_gain_regression.csv compare new B with matched reused A and all fixed historical references, including R60B. Task and family gains/regressions are descriptive, never a checkpoint-selection rule.', '']
    for row in gains:
        if row['left_model'] == MODEL_NAMES['B'] and row['right_model'] in (MODEL_NAMES['A'], 'R60B'):
            lines.append(f"- {row['split']} B versus {row['right_model']}: {row['gains']} task gains, {row['regressions']} regressions, {row['ties']} ties.")
    lines += ['',
        'stage2_summary.csv reports selected-best/final TRAIN/VAL and last5 validation mean/population SD/min/max for epochs36-40. Last5 is neither seed uncertainty nor an averaged checkpoint. matched_common_epochs.csv uses only actually evaluated common epochs, including actual selected-best TRAIN. Every VAL epoch is full18; TRAIN is every5/final/selected-best. Curves contain only actual measurements, with missing historical TRAIN disclosed in metric_availability.csv. No dense TRAIN interpolation or reference inference fills gaps.', '',
        '## Compute and Interpretation', '',
        '| Component | Updates | Selected Rows | Successful View Rows | Attempted View Rows | AMP Retries | Wall s |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for label, key in (('NEW R69B', 'new_B'), ('HISTORICAL reused R68A', 'historical_reused_A')):
        value = compute[key]
        lines.append(f"| {label} | {value['successful_updates']} | {value['selected_rows']} | {value['successful_view_rows']} | {value['attempted_view_rows']} | {value['amp_retries']} | {value['wall_seconds']:.6f} |")
    prior, new = compute['inherited_stage1_once'], compute['new_B']
    lines += ['',
        f"Inherited R67B terminal stage1 pretraining counted ONCE separately: {prior['successful_updates']} updates/{prior['selected_rows']} rows/{prior['wall_seconds']:.6f}s. New B only: {new['successful_updates']} updates/{new['selected_rows']} rows/{new['successful_view_rows']} successful views. Each B branch:1600 active updates/400000 selected rows. New B peak allocated/reserved memory: {new['peak_allocated_gib']:.6f}/{new['peak_reserved_gib']:.6f}GiB. Reused A's3002.024237s and5 retries are historical, not new R69 expenditure.", '',
        'Actual B capacity is33173154 parameters versus1842953 shared A parameters. Capacity and per-parameter optimizer histories are NOT matched. Per-parameter Adam steps advance only on active gradients; unused input paths may remain at0 steps. branch_compute.csv and parameter_optimizer_steps.csv preserve actual activation/row/counter evidence. compute_summary.json verifies saved initialization and actual task/row/dropout/LR order against the reused control; it does not rerun historical outputs.', '',
        'All18 copies start identically from R67B terminal stage1 averaged-x trunk plus the entire fresh head670003, through the exact saved R68 pre-fine-tuning state. These are not from-scratch encoders. B trains all parameters, with no SAM/projector/averaging/optimizer mode transition. Same global41-80 schedule, native CE and FP32 legal symmetric RDrop, real256x3/tails16x3, global successful-update warmup clock, per-task FP32 gradient/3, one combined clip1 and one AdamW step. Whole-update AMP retries retain the same rows/RNG and do not advance the cursor or successful counter.', '',
        'A gain supports this unsharing recipe; it does not isolate harmful task-gradient interference from increased capacity, fewer per-parameter updates, changed Adam histories or fitting18 models with10000 fine-tuning examples each. No result establishes an accuracy ceiling, label noise, or a70% promise. Historical TEST exposure makes cross-version comparisons descriptive; the new B TEST never changes architecture, budget, selection or planning.', '',
        f"B validation correct minus matched A: {decision['validation_count_B_minus_A']}. Numeric official validation-frozen original18 TEST70% threshold reached: {decision['numeric_threshold_reached']}. Parent whole-goal audit passed: {decision['whole_objective_audit_passed']}. Supported70% claim: {decision['claim_70pct_supported']}. Parent owns whole-goal audit and closure.", '',
        '## Deployment, Provenance and Assets', '',
        'Known original problem identity activates ONE complete branch, one full-pool direct-utility forward, ascending-global-ID exact ties, then ONE selected solver completes per deployed query. Delivery evaluation executes ZERO solvers. test_results.json distinguishes newly evaluated B from unchanged saved A metrics and provenance; A has one historical original TEST, zero new R69 tests. No reference inference, NPZ prediction replay, solver runs or separate singleton latency benchmark occur.', '',
        f"New B ordinary full evaluation time: {new['complete_test_evaluation_seconds']:.6f}s. This includes transfer/metrics/NPZ saving and is NOT singleton latency. Other reference timings retain historical scope. Selected weights and complete NPZ logits/predictions/native winners/rawFP64 costs/indices/sizes/global pool IDs/names stay LOCAL with raw updates/W&B/schedules/cache. Git alone is not all inference assets; nothing is force-added.", '',
        'The eight legacy B-task feasible-domain mismatches stay disclosed/unfixed and are not a native70 veto. R58 is fully deferred. Parent owns reviews, goal audit, memory/state and one Git commit per completed version. This report creates none of those approvals.', '']
    (root/'REPORT.md').write_text('\n'.join(lines))


def summarize(root, authorized):
    if not authorized or root.resolve() != ROOT.resolve() or os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('Future separately authorized ONCE analysis requires --authorize-analysis, canonical root and CUDA_VISIBLE_DEVICES=')
    protocol, lock, test, test_terminal = completed_test(root)
    with (root/'analysis_started.json').open('x') as stream:
        json.dump(dict(started_utc=now(), cpu_only=True, executions=1, module_sha256=file_hash(Path(__file__)),
            final_test_lock_sha256=file_hash(root/'final_test_lock.json'),
            test_terminal_receipt_sha256=file_hash(root/'test_terminal_receipt.json'),
            inference=False, prediction_replay=False, automatic_retry=False), stream, indent=2)
        stream.write('\n')
    started = time.monotonic()
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        plt.rcParams.update({'font.size': 9})
        curves = root/'curves/delivery'
        curves.mkdir(parents=True, exist_ok=True)
        directories = {'A': CONTROL_RUN, 'B': root/RUN_NAME}
        results = {arm: read(directory/'result.json') for arm, directory in directories.items()}
        histories = {arm: read(directory/'history.json') for arm, directory in directories.items()}
        updates = {arm: read_updates(directory/'updates.jsonl') for arm, directory in directories.items()}
        if results['A']['best']['epoch'] != 28 or results['A']['best']['val']['native_correct'] != 8644:
            raise ValueError('Keep fixed reused R68A result identity; no new A selection')
        compute, branches, counters = actual_compute(root, protocol, results, updates, test, test_terminal)
        atomic_json(root/'compute_summary.json', compute)
        write_csv(root/'branch_compute.csv', branches)
        write_csv(root/'parameter_optimizer_steps.csv', [dict(parameter=name, adam_steps=steps)
                  for name, steps in counters['steps'].items()])
        task_rows, family_rows = export_curves(root, histories, results, updates)
        validation_summaries(root, results, histories)
        figures(plt, curves, histories, updates, task_rows, family_rows)
        selected = {MODEL_NAMES[arm]: dict(epoch=lock['models'][arm]['epoch'],
            train=result['best']['train'], val=result['best']['val'], test=test['models'][arm]) for arm, result in results.items()}
        methods, diagnostics = selected_figures(plt, curves, selected)
        write_csv(root/'method_metrics.csv', methods)
        write_csv(root/'task_diagnostics.csv', diagnostics)
        historical, provenance = fixed_references(root)
        selected.update(historical)
        comparison, deltas, gains = comparison_tables(root, selected)
        for split in ('train', 'val', 'test'):
            points = [row for row in deltas if row['left_model'] == MODEL_NAMES['B']
                      and row['right_model'] in (MODEL_NAMES['A'], 'R60B') and row['split'] == split and row['kind'] == 'task']
            fig, ax = plt.subplots(figsize=(12, 5))
            positions = np.arange(18)
            for offset, ref in enumerate((MODEL_NAMES['A'], 'R60B')):
                values = {row['scope']: row['top1_delta_pp'] for row in points if row['right_model'] == ref}
                if set(values) == set(PROBLEMS):
                    ax.bar(positions+(offset-.5)*.38, [values[task] for task in PROBLEMS], width=.38, label='B minus '+ref)
            ax.axhline(0, color='black', linewidth=.8)
            ax.set_xticks(positions, PROBLEMS, rotation=55, ha='right')
            ax.set(title='Selected '+split+' task deltas', ylabel='Native Top1 delta (percentage points)')
            ax.legend(fontsize=8)
            fig.tight_layout()
            fig.savefig(curves/('task_deltas_'+split+'.png'), dpi=150)
            plt.close(fig)
        official = lock['official_arm']
        correct = test['models'][official]['native_correct']
        audit_path = root/'objective_audit.json'
        audit = read(audit_path) if audit_path.exists() else {}
        audit_passed = (audit.get('passed') is True and audit.get('scope') == 'whole objective'
            and audit.get('final_test_lock_sha256') == file_hash(root/'final_test_lock.json')
            and audit.get('test_results_sha256') == file_hash(root/'test_results.json'))
        budget_consistent = all(compute['checks'].values())
        decision = dict(official_arm=official, official_model=MODEL_NAMES[official],
            selection='INTEGER VAL only; B earliest common-epoch maximum; official B strictly>8644, tie/lower fixed A',
            validation_count_B_minus_A=lock['models']['B']['native_correct']-8644,
            B_vs_matched_A_test_delta_pp=100*(test['models']['B']['macro']['macro_top1']-test['models']['A']['macro']['macro_top1']),
            B_vs_fixed_R60B_test_delta_pp=100*(test['models']['B']['macro']['macro_top1']-selected['R60B']['test']['macro']['macro_top1']),
            official_test_correct=correct, official_test_total=18000, official_test_top1=correct/18000,
            numeric_threshold_reached=10*correct >= 7*18000, whole_objective_audit_passed=audit_passed,
            actual_budget_consistent=budget_consistent,
            claim_70pct_supported=10*correct >= 7*18000 and audit_passed and budget_consistent,
            goal_status='ACTIVE; parent whole-goal audit/closure required', one_new_arm=True, one_seed=True,
            parameter_capacity_matched=False, per_parameter_optimizer_histories_matched=False,
            pure_task_interference_causal_claim=False, accuracy_ceiling_or_noise_claim=False,
            historical_test_exposure=True, new_B_test_used_for_planning=False, test_reselection=False, cost_veto=False)
        provenance[MODEL_NAMES['A']] = lock['control_reuse']
        provenance[MODEL_NAMES['B']] = dict(result_path=str(root/RUN_NAME/'result.json'),
            result_sha256=file_hash(root/RUN_NAME/'result.json'), test_path=str(root/'test_results.json'),
            test_json_pointer='/models/B', checkpoint=lock['models']['B'], evaluation='ONE new original18 TEST')
        atomic_json(root/'reference_provenance.json', dict(recorded_utc=now(), references=provenance,
            historical_test_exposure=True, reference_inference=False, NPZ_replay=False, solver_executions=0,
            extraction='saved official JSON/update records only; no missing TRAIN inference'))
        report(root, selected, lock, decision, compute, gains)
        atomic_json(root/'analysis.json', dict(recorded_utc=now(), decision=decision, actual_compute=compute,
            selected={model: dict(epoch=record['epoch'], **{split: record[split]['macro'] for split in ('train', 'val', 'test')
                if record.get(split) is not None}) for model, record in selected.items()},
            new_arms=['B'], reused_control=MODEL_NAMES['A'], legacy_B_semantics='eight mismatches disclosed/unfixed; R58 deferred'))
        receipt = dict(recorded_utc=now(), returncode=0, cpu_only=True, executions=1, wall_seconds=time.monotonic()-started,
            module_sha256=file_hash(Path(__file__)), comparison_rows=len(comparison), png_count=len(list(curves.rglob('*.png'))),
            inference=False, NPZ_replay=False, new_arms=['B'], reused_arms=['A'])
        atomic_json(root/'analysis_receipt.json', receipt)
        atomic_json(root/'delivery_receipt.json', dict(recorded_utc=now(), training=compute['training_terminal'],
            new_B_test=test_terminal, historical_A_test_reused=lock['control_reuse'], analysis=receipt,
            freeze_sha256=file_hash(root/'final_test_lock.json'), decision=decision,
            parent_reviews_goal_audit_memory_and_git=True, git_alone_contains_all_inference_assets=False,
            worker_git_actions=False, r58='fully deferred; eight legacy B-task mismatches disclosed'))
        print(json.dumps(dict(analysis=receipt, decision=decision)), flush=True)
    except BaseException as error:
        atomic_json(root/'analysis_failure.json', dict(recorded_utc=now(), exception_type=type(error).__name__,
            message=str(error), traceback=traceback.format_exc(), marker_preserved=True, automatic_retry=False))
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--authorize-analysis', action='store_true')
    args = parser.parse_args(argv)
    summarize(args.root.resolve(), args.authorize_analysis)


if __name__ == '__main__':
    main()
