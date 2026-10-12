"""ONCE CPU R70 delivery analysis from saved JSON/updates, never NPZ replay."""

import argparse
import json
import os
import time
import traceback
from collections import Counter
from pathlib import Path

import numpy as np

from code.unified_selector.registry import PROBLEMS
from .r59_experiment import atomic_json
from .r64_analysis import METRICS, write_csv
from .r65_analysis import historical_records, metrics
from .r67_analysis import read_updates
from .r68_analysis import numeric_summary, selected_figures
from .r70_experiment import CONTROL_ROOT, ROOT, RUN_NAME
from .r70_frozen_evaluation import CONTROL_RUN, completed_test, exclusive_json, now, read


NAMES = {'A': 'R70A_reused_R69B20', 'B': 'R70B_latent_mixup'}
LABELS = {'A': 'Reused R69B20', 'B': 'New R70B latent mixup'}


def metric_rows(model, split, epoch, value):
    common = dict(model=model, split=split, epoch=epoch)
    per = value['per_problem']
    rows = [dict(common, kind='aggregate', scope='ALL', native_correct=value.get('native_correct'),
        native_total=value.get('native_total', sum(entry['n'] for entry in per.values())), **metrics(value['macro'], 'macro_'))]
    for family, entry in value['families'].items():
        tasks = [family] if family != 'MVRP' else [task for task in per if task not in ('TSP', 'CVRP', 'ATSP')]
        counts = [per[task].get('native_correct') for task in tasks]
        rows.append(dict(common, kind='family', scope=family,
            native_correct=sum(counts) if all(type(count) is int for count in counts) else None,
            native_total=sum(per[task]['n'] for task in tasks), **metrics(entry, 'macro_')))
    rows += [dict(common, kind='task', scope=task, native_correct=entry.get('native_correct'),
                  native_total=entry['n'], **metrics(entry)) for task, entry in per.items()]
    return rows


def export_curves(histories, results, updates):
    learning, optimization, task_optimization = [], [], []
    for arm, history in histories.items():
        for row in history:
            identity = dict(arm=arm, execution='historical reuse' if arm == 'A' else 'new formal training',
                            global_epoch=row['global_epoch'])
            optimization.append(dict(identity, epoch=row['epoch'], successful_updates=row['successful_updates'],
                selected_rows=row['presentations'], cumulative_encoder_view_rows=row['successful_view_rows'],
                cumulative_mixed_head_rows=row.get('successful_mixed_head_rows'), **row['optimization']))
            task_optimization += [dict(identity, epoch=row['epoch'], problem=task, **entry)
                                  for task, entry in row['task_optimization'].items()]
            for split in ('train', 'val'):
                if row.get(split) is not None:
                    learning += [dict(identity, measurement='actual epoch', **entry)
                                 for entry in metric_rows(NAMES[arm], split, row['epoch'], row[split])]
        best = results[arm]['best']
        learning += [dict(arm=arm, global_epoch=best['global_epoch'], measurement='actual selected-best TRAIN', **entry)
                     for entry in metric_rows(NAMES[arm], 'train', best['epoch'], best['train'])]
    write_csv(ROOT / 'learning_curves.csv', learning)
    write_csv(ROOT / 'task_curves.csv', [row for row in learning if row['kind'] == 'task'])
    write_csv(ROOT / 'family_curves.csv', [row for row in learning if row['kind'] == 'family'])
    write_csv(ROOT / 'optimization_curves.csv', optimization)
    write_csv(ROOT / 'task_optimization.csv', task_optimization)
    write_csv(ROOT / 'update_curves.csv', [{key: json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else value
        for key, value in dict(row, model=NAMES['B']).items()} for row in updates])
    return learning


def summaries(histories, results):
    output, common, availability = [], [], []
    for arm, history in histories.items():
        for label in ('best', 'final'):
            record = results[arm][label]
            for split in ('train', 'val'):
                if record.get(split) is not None:
                    output += [dict(arm=arm, summary=label, global_epoch=record['global_epoch'], **entry)
                               for entry in metric_rows(NAMES[arm], split, record['epoch'], record[split])]
        last = [metric_rows(NAMES[arm], 'val', row['epoch'], row['val']) for row in history[-5:]]
        for index, entry in enumerate(last[0]):
            for label, function in (('last5_mean', np.mean), ('last5_population_sd', np.std), ('last5_min', np.min), ('last5_max', np.max)):
                output.append(dict(model=NAMES[arm], arm=arm, split='val', kind=entry['kind'], scope=entry['scope'],
                    summary=label, actual_epochs='36,37,38,39,40', not_seed_uncertainty=True,
                    **{key: float(function([rows[index][key] for rows in last])) for key in METRICS}))
        actual_train_epochs = {row['epoch'] for row in history[-5:] if row.get('train') is not None}
        if 36 <= results[arm]['best']['epoch'] <= 40:
            actual_train_epochs.add(results[arm]['best']['epoch'])
        availability.append(dict(model=NAMES[arm], scope='last five epochs36-40 TRAIN mean', available=False,
            actual_train_epochs=','.join(str(epoch) for epoch in sorted(actual_train_epochs)),
            reason='Only actual sparse TRAIN records exist; no interpolation or five-epoch mean'))
    for split in ('train', 'val'):
        available = {arm: {row['epoch']: row[split] for row in history if row.get(split) is not None}
                     for arm, history in histories.items()}
        for arm in ('A', 'B'):
            record = results[arm]['best']
            if record.get(split) is not None:
                available[arm].setdefault(record['epoch'], record[split])
        for epoch in sorted(set(available['A']) & set(available['B'])):
            left = metric_rows(NAMES['A'], split, epoch, available['A'][epoch])
            right = metric_rows(NAMES['B'], split, epoch, available['B'][epoch])
            for a, b in zip(left, right):
                common.append(dict(epoch=epoch, global_epoch=40 + epoch, split=split, kind=a['kind'], scope=a['scope'],
                    A_native_correct=a['native_correct'], B_native_correct=b['native_correct'],
                    native_correct_delta=b['native_correct'] - a['native_correct'],
                    **{arm + '_' + key: entry[key] for arm, entry in (('A', a), ('B', b)) for key in METRICS},
                    **{key + '_B_minus_A': b[key] - a[key] for key in METRICS}))
    write_csv(ROOT / 'stage2_summary.csv', output)
    write_csv(ROOT / 'matched_common_epochs.csv', common)
    write_csv(ROOT / 'summary_availability.csv', availability)


def comparison_tables(selected):
    rows = [row for model, record in selected.items() for split in ('train', 'val', 'test')
            if record.get(split) is not None for row in metric_rows(model, split, record['epoch'], record[split])]
    write_csv(ROOT / 'comparison.csv', rows)
    write_csv(ROOT / 'selected_metrics.csv', [row for row in rows if row['model'] in NAMES.values()])
    write_csv(ROOT / 'metric_availability.csv', [dict(model=model, split=split, available=record.get(split) is not None,
        policy='saved actual reports only; missing TRAIN/counts remain missing; no NPZ replay or inference')
        for model, record in selected.items() for split in ('train', 'val', 'test')])
    indexed = {(row['model'], row['split'], row['kind'], row['scope']): row for row in rows}
    deltas, gains = [], []
    for reference in [model for model in selected if model != NAMES['B']]:
        for split in ('train', 'val', 'test'):
            signs = Counter()
            for key, right in indexed.items():
                if key[0] != reference or key[1] != split:
                    continue
                left = indexed.get((NAMES['B'], split, key[2], key[3]))
                if left is None:
                    continue
                counts = (left['native_correct'], right['native_correct'])
                deltas.append(dict(left_model=NAMES['B'], right_model=reference, split=split, kind=key[2], scope=key[3],
                    native_correct_delta=counts[0] - counts[1] if all(type(count) is int for count in counts) else None,
                    **{metric + ('_delta_pp' if metric.startswith('top') else '_delta'):
                       (left[metric] - right[metric]) * (100 if metric.startswith('top') else 1) for metric in METRICS}))
                if key[2] == 'task':
                    change = left['top1'] - right['top1']
                    signs['gains' if change > 0 else 'regressions' if change < 0 else 'ties'] += 1
            if signs:
                gains.append(dict(left_model=NAMES['B'], right_model=reference, split=split,
                    **{key: signs[key] for key in ('gains', 'regressions', 'ties')}))
    write_csv(ROOT / 'comparison_deltas.csv', deltas)
    write_csv(ROOT / 'task_gain_regression.csv', gains)
    return rows, deltas, gains


def figures(plt, curves, histories, updates, learning):
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    for ax, metric in zip(axes.flat, METRICS):
        for arm, history in histories.items():
            for split in ('train', 'val'):
                points = [row for row in history if row.get(split) is not None]
                ax.plot([row['epoch'] for row in points], [row[split]['macro']['macro_' + metric] for row in points],
                    linestyle='None' if split == 'train' else '-', marker='.' if split == 'train' else None,
                    label=LABELS[arm] + ' ' + split)
        ax.set(title=metric, xlabel='Actual fine-tuning epoch/global41-80')
        ax.grid(alpha=.2)
        ax.legend(fontsize=7)
    axes.flat[-1].axis('off')
    fig.tight_layout()
    fig.savefig(curves / 'learning_curves.png', dpi=150)
    plt.close(fig)
    keys = ('loss', 'clean_ce', 'mix_ce', 'kl', 'lr', 'grad_norm', 'clip_coefficient', 'retries', 'scale',
        'successful_view_rows', 'attempted_view_rows', 'successful_mixed_head_rows', 'attempted_mixed_head_rows',
        'seconds', 'peak_memory_gib', 'peak_reserved_gib', 'physical_used_gib')
    for resolution in ('update', 'epoch'):
        rows = updates if resolution == 'update' else [dict(row['optimization'], epoch=row['epoch']) for row in histories['B']]
        xkey = 'successful_update' if resolution == 'update' else 'epoch'
        fig, axes = plt.subplots(5, 4, figsize=(18, 16))
        for ax, key in zip(axes.flat, keys):
            actual = 'amp_retries' if resolution == 'epoch' and key == 'retries' else key
            points = [row for row in rows if row.get(actual) is not None]
            if points:
                ax.plot([row[xkey] for row in points], [row[actual] for row in points], linewidth=.7)
            ax.set(title=key, xlabel=xkey)
            ax.grid(alpha=.2)
        for ax in list(axes.flat)[len(keys):]:
            ax.axis('off')
        fig.tight_layout()
        fig.savefig(curves / ('optimization_' + resolution + '.png'), dpi=150)
        plt.close(fig)
    for kind, columns in (('task', 3), ('family', 2)):
        names = list(dict.fromkeys(row['scope'] for row in learning if row['kind'] == kind))
        fig, axes = plt.subplots((len(names) + columns - 1) // columns, columns,
            figsize=(6 * columns, 3 * ((len(names) + columns - 1) // columns)), squeeze=False)
        for ax, scope in zip(axes.flat, names):
            for arm in ('A', 'B'):
                for split in ('train', 'val'):
                    points = [row for row in learning if row['kind'] == kind and row['scope'] == scope
                              and row['arm'] == arm and row['split'] == split and row['measurement'] == 'actual epoch']
                    ax.plot([row['epoch'] for row in points], [row['top1'] for row in points],
                        linestyle='None' if split == 'train' else '-', marker='.' if split == 'train' else None,
                        label=LABELS[arm] + ' ' + split)
            ax.set(title=scope, xlabel='Actual epoch', ylabel='Native Top1')
            ax.legend(fontsize=6)
            ax.grid(alpha=.2)
        for ax in list(axes.flat)[len(names):]:
            ax.axis('off')
        fig.tight_layout()
        fig.savefig(curves / (kind + '_top1.png'), dpi=150)
        plt.close(fig)


def actual_compute(protocol, results, updates, test, terminal):
    result, prior = results['B'], results['A']
    sums = dict(successful_updates=len(updates), selected_rows=sum(row['rows'] for row in updates),
        successful_view_rows=sum(row['successful_view_rows'] for row in updates),
        attempted_view_rows=sum(row['attempted_view_rows'] for row in updates),
        successful_mixed_head_rows=sum(row['successful_mixed_head_rows'] for row in updates),
        attempted_mixed_head_rows=sum(row['attempted_mixed_head_rows'] for row in updates),
        amp_retries=sum(row['retries'] for row in updates))
    if any(sums[key] != result['presentations' if key == 'selected_rows' else key] for key in sums):
        raise ValueError('Actual new-B raw-update accounting must equal its published final result')
    if (sums['successful_updates'], sums['selected_rows'], sums['successful_view_rows'], sums['successful_mixed_head_rows']) != (9600, 7200000, 14400000, 14400000):
        raise ValueError('Keep exact formal new-B budget, excluding capacity')
    new = dict(sums, wall_seconds=result['wall_seconds'], logged_update_seconds=sum(row['seconds'] for row in updates),
        peak_allocated_gib=max(row['peak_memory_gib'] for row in updates),
        peak_reserved_gib=max(row['peak_reserved_gib'] for row in updates),
        peak_physical_gib=max(row['physical_used_gib'] for row in updates),
        grad_norm=numeric_summary([row['grad_norm'] for row in updates]),
        clip_coefficient=numeric_summary([row['clip_coefficient'] for row in updates]),
        amp_scale=numeric_summary([row['scale'] for row in updates]),
        complete_test_evaluation_seconds=test['models']['B']['complete_evaluation_seconds'])
    historical = dict(successful_updates=prior['successful_updates'], selected_rows=prior['presentations'],
        successful_view_rows=prior['successful_view_rows'], attempted_view_rows=prior['attempted_view_rows'],
        amp_retries=prior['amp_retries'], wall_seconds=prior['wall_seconds'], mixed_head_rows=0,
        original_test_already_executed=1, new_R70_tests=0, new_R70_compute=0)
    if (historical['successful_updates'], historical['selected_rows'], historical['amp_retries']) != (9600, 7200000, 5) or abs(historical['wall_seconds'] - 3554.574466) > 1e-5:
        raise ValueError('Reused A expenditure remains historical R69B, unchanged')
    inherited_path = Path(protocol['inherited_identity']['terminal']['path']).parent / 'stage1_result.json'
    inherited = read(inherited_path)
    capacity = read(ROOT / 'capacity_receipt.json')
    compute = dict(new_B=new, historical_reused_A=historical,
        inherited_stage1_once=dict(successful_updates=inherited['successful_updates'], selected_rows=inherited['presentations'],
            wall_seconds=inherited['wall_seconds'], counted_once=True, new_compute=0, result_path=str(inherited_path)),
        discarded_capacity=dict(updates=1, selected_rows=768, wall_seconds=capacity['wall_seconds'],
            optimizer_moment_gpu_bytes=capacity['optimizer_moment_gpu_bytes'], storage=capacity['optimizer_storage'],
            retries=capacity['values']['retries'], excluded_from_formal_clock_budget_and_runtime=True),
        parameter_counts=read(ROOT / 'parameter_counts.json'), parameter_capacity_matched=True,
        per_parameter_update_recipe_matched=True, head_compute_matched=False, extra_mixed_head_rows=14400000,
        fresh_encoders=False, independent_control_replication=False,
        training_terminal=read(ROOT / 'training_terminal_receipt.json'), test_terminal=terminal,
        new_test_wall_seconds=test['wall_seconds'], solver_executions_during_delivery=0,
        timing_scope='whole ordinary full18 evaluation; not singleton latency or solver runtime')
    atomic_json(ROOT / 'compute_summary.json', compute)
    counters = read(ROOT / RUN_NAME / 'parameter_step_counts.json')
    write_csv(ROOT / 'parameter_optimizer_steps.csv', [dict(parameter=name, adam_steps=steps) for name, steps in counters['steps'].items()])
    write_csv(ROOT / 'branch_compute.csv', [dict(problem=task, branch_id=index,
        active_updates=result['updates_by_problem'][task], selected_rows=result['rows_by_problem'][task],
        successful_encoder_view_rows=2 * result['rows_by_problem'][task],
        successful_mixed_head_rows=2 * result['rows_by_problem'][task]) for index, task in enumerate(PROBLEMS)])
    return compute


def report(selected, lock, decision, compute, gains):
    lines = ['# R70 Locked Delivery', '',
        f"VAL-frozen official arm: {decision['official_arm']}. Official original18 TEST: {decision['official_test_correct']}/18000 = {100 * decision['official_test_top1']:.6f}% native Top1. The numeric70% threshold is12600/18000. Goal70 remains ACTIVE/unachieved; parent whole-objective audit and reviews remain required.", '',
        'Exactly ONE new R70 B was trained and tested. Fixed A is the unchanged R69B20/global60 complete18 bank with8604/18000 VAL correct and one already-executed original TEST. New TEST counts A0/B1; original A already1. A weights were not loaded, inferred, recounted, retrained or reselected. Its existing test_results.json/models/B object is reused unchanged. Protocol equivalence is not independent replication.', '',
        f"New B selected common epoch{lock['models']['B']['epoch']}/global{lock['models']['B']['global_epoch']}, VAL{lock['models']['B']['native_correct']}/18000: earliest maximum INTEGER full18 VAL count. Official B requires strictly>8604; tie/lower fixed A. All18 branches come from ONE complete bank checkpoint. Costs, task gains and TEST do not select.", '',
        '| Model | Split | Top1% | Top2% | Top3% | CE | Mean Cost | vsSBS% | Actual Regret% |',
        '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for model, record in selected.items():
        for split in ('train', 'val', 'test'):
            if record.get(split) is not None:
                value = metrics(record[split]['macro'], 'macro_')
                lines.append(f"| {model} | {split} | {100 * value['top1']:.6f} | {100 * value['top2']:.6f} | {100 * value['top3']:.6f} | {value['ce']:.6f} | {value['mean_cost']:.6f} | {value['vs_sbs_pct']:.6f} | {value['actual_regret_pct']:.6f} |")
    lines += ['',
        'comparison.csv and selected_metrics.csv contain full18 equal macros, TSP/CVRP/ATSP/MVRP families and all per-problem TRAIN/VAL/TEST Top1/2/3, CE, mean cost, vsSBS and actual regret. Cross-task costs mix task units and are descriptive. comparison_deltas.csv/task_gain_regression.csv preserve task/family gains. Fixed R45A/R60B/R64 references use saved official JSON only; missing TRAIN/counts stay missing, with no recount/inference/NPZ replay.', '',
        'TRAIN measurements are only every5/final and selected best. VAL measurements are every epoch. stage2_summary.csv contains best/final and epochs36-40 VAL mean/population SD/min/max; matched_common_epochs.csv uses actual common epochs only. A full last-five TRAIN mean is unavailable; summary_availability.csv lists the sparse actual TRAIN epochs, including a selected-best point in that interval when present. These are not seed uncertainty or averaged checkpoints.', '',
        '| Compute | Updates | Selected Rows | Encoder Views Successful/Attempted | Mixed Heads Successful/Attempted | Retries | Wall s |',
        '| --- | ---: | ---: | --- | --- | ---: | ---: |']
    for label, key in (('NEW formal B', 'new_B'), ('HISTORICAL reused A', 'historical_reused_A')):
        item = compute[key]
        lines.append(f"| {label} | {item['successful_updates']} | {item['selected_rows']} | {item['successful_view_rows']}/{item['attempted_view_rows']} | {item.get('successful_mixed_head_rows', 0)}/{item.get('attempted_mixed_head_rows', 0)} | {item['amp_retries']} | {item['wall_seconds']:.6f} |")
    inherited, capacity, new = compute['inherited_stage1_once'], compute['discarded_capacity'], compute['new_B']
    lines += ['',
        f"Inherited stage1 counted ONCE separately: {inherited['successful_updates']} updates/{inherited['selected_rows']} rows/{inherited['wall_seconds']:.6f}s. The one discarded capacity update adds768 TRAIN rows/{capacity['wall_seconds']:.6f}s, with{capacity['optimizer_moment_gpu_bytes']} bytes of full-bank Adam moments explicitly materialized. It is extra work, excluded from the formal9600-update clock; its mutated state was discarded before formal recreation.", '',
        f"Both A/B have33173154 parameters and the same per-parameter update recipe:1600 active updates/400000 selected rows per branch. B adds14.4M mixed-head rows; head compute is unmatched. New B peak allocated/reserved/physical memory: {new['peak_allocated_gib']:.6f}/{new['peak_reserved_gib']:.6f}/{new['peak_physical_gib']:.6f}GiB. New complete TEST evaluation time: {new['complete_test_evaluation_seconds']:.6f}s; this is not singleton latency or solver runtime.", '',
        'B starts from the ORIGINAL pre-fine-tuning R67B terminal stage1 averaged-x trunk plus entire fresh head670003, never trained A20. It is not a from-scratch encoder. The unchanged fixed40E global41-80 schedule uses real256x3/tails16x3, fresh AdamW, FP32 active disjoint task gradients/3, one clip1 and one step. Inactive grad=None means no decay/moment/counter advancement. Whole-update AMP retries preserve rows/RNG and successful clock.', '',
        'The sole objective package is0.5*CEclean+0.5*CEmix+.35*min(u/720,1)*KLclean. Both clean forwards occur first; one private seeded pi/lambda is shared across views, mixed features backpropagate through both partners, and auxiliary-head dropout restores ordinary CPU/CUDA RNG. No mixed KL, labels in features, coefficient search, new head or changed architecture. optimization/update CSVs and plots preserve actual component/LR/gradient/clip/AMP/retry/row/wall/memory records.', '',
        'Native inference remains ordinary clean FP32/full original legal pool/global-ID exact ties, one active branch and one forward/query. Delivery executed zero solvers, exactly144 wrapper/active-branch batches with104-row tails. A deployed query completes one selected solver. No mixed-feature TEST, augmentation, sampling, pruning, ensemble or singleton benchmark.', '',
        'Historical TEST exposure is disclosed. New B TEST does not change model, selection, budget, rules or successor planning; parent future planning explicitly excludes NEW R70 TEST. No result here establishes a70% claim, noise diagnosis or accuracy ceiling.', '',
        'Weights, NPZ logits/predictions/native winners/raw FP64 costs/indices/sizes/global pool IDs/names, raw updates, W&B, cache and regenerable schedule stay LOCAL. Git alone does not contain all inference assets. This executor performs no Git operations.', '',
        'R58 is fully deferred. Eight legacy B feasible-domain mismatches remain disclosed/unfixed and are not a native70% veto. Parent owns actual Astra reviews, fresh review, whole-objective audit, memory/state and one completed-version publication.', '']
    for entry in gains:
        if entry['right_model'] in (NAMES['A'], 'R60B'):
            lines.append(f"{entry['split']} B versus{entry['right_model']}: {entry['gains']} task gains/{entry['regressions']} regressions/{entry['ties']} ties.")
    text = '\n'.join(lines) + '\n'
    (ROOT / 'REPORT.md').write_text(text)
    (ROOT / 'RESULTS.md').write_text(text)


def summarize():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('ONCE analysis requires CPU-only CUDA_VISIBLE_DEVICES=')
    protocol, lock, test, terminal = completed_test()
    exclusive_json(ROOT / 'analysis_started.json', dict(started_utc=now(), cpu_only=True, invocations=1,
        inference=False, NPZ_replay=False, automatic_retry=False))
    started = time.monotonic()
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        plt.rcParams.update({'font.size': 9})
        curves = ROOT / 'curves/delivery'
        curves.mkdir(parents=True, exist_ok=True)
        results = {'A': read(CONTROL_RUN / 'result.json'), 'B': read(ROOT / RUN_NAME / 'result.json')}
        histories = {'A': read(ROOT / 'reused_A_history.json'), 'B': read(ROOT / RUN_NAME / 'history.json')}
        if results['A']['best']['epoch'] != 20 or results['A']['best']['val']['native_correct'] != 8604:
            raise ValueError('Preserve fixed R69B20 control metadata')
        updates = read_updates(ROOT / RUN_NAME / 'updates.jsonl')
        compute = actual_compute(protocol, results, updates, test, terminal)
        learning = export_curves(histories, results, updates)
        summaries(histories, results)
        figures(plt, curves, histories, updates, learning)
        selected = {NAMES[arm]: dict(epoch=lock['models'][arm]['epoch'], train=result['best']['train'],
            val=result['best']['val'], test=test['models'][arm]) for arm, result in results.items()}
        methods, diagnostics = selected_figures(plt, curves, selected)
        write_csv(ROOT / 'method_metrics.csv', methods)
        write_csv(ROOT / 'task_diagnostics.csv', diagnostics)
        historical, provenance = historical_records(ROOT)
        selected.update(historical)
        comparison, deltas, gains = comparison_tables(selected)
        official = lock['official_arm']
        correct = test['models'][official]['native_correct']
        decision = dict(official_arm=official, official_model=NAMES[official],
            official_test_correct=correct, official_test_total=18000, official_test_top1=correct / 18000,
            native_70pct_threshold_correct=12600, numeric_threshold_reached=correct >= 12600,
            validation_count_B_minus_A=lock['models']['B']['native_correct'] - 8604,
            B_vs_matched_A_test_delta_pp=100 * (test['models']['B']['macro']['macro_top1'] - test['models']['A']['macro']['macro_top1']),
            B_vs_fixed_R60B_test_delta_pp=100 * (test['models']['B']['macro']['macro_top1'] - selected['R60B']['test']['macro']['macro_top1']),
            whole_objective_audit_passed=False, claim_70pct_supported=False,
            goal_status='ACTIVE/unachieved; parent whole-objective audit and actual reviews pending',
            one_new_arm=True, one_seed=True, parameter_capacity_matched=True, extra_head_compute=True,
            historical_test_exposure=True, new_B_test_used_for_planning=False, test_reselection=False, cost_veto=False)
        provenance[NAMES['A']] = lock['control_reuse']
        provenance[NAMES['B']] = dict(result_path=str(ROOT / RUN_NAME / 'result.json'),
            test_path=str(ROOT / 'test_results.json'), test_json_pointer='/models/B',
            checkpoint=lock['models']['B'], evaluation='ONE new original full18 TEST')
        atomic_json(ROOT / 'reference_provenance.json', dict(recorded_utc=now(), references=provenance,
            historical_test_exposure=True, reference_inference=False, NPZ_replay=False, solver_executions=0))
        report(selected, lock, decision, compute, gains)
        atomic_json(ROOT / 'analysis.json', dict(recorded_utc=now(), decision=decision, actual_compute=compute,
            selected={model: dict(epoch=record['epoch'], **{split: record[split]['macro'] for split in ('train', 'val', 'test')
                if record.get(split) is not None}) for model, record in selected.items()},
            new_arms=['B'], reused_control=NAMES['A'], legacy_B_semantics='eight mismatches disclosed/unfixed; R58 deferred'))
        receipt = dict(recorded_utc=now(), returncode=0, cpu_only=True, invocations=1,
            wall_seconds=time.monotonic() - started, comparison_rows=len(comparison),
            png_count=len(list(curves.rglob('*.png'))), inference=False, NPZ_replay=False,
            new_arms=['B'], reused_arms=['A'])
        atomic_json(ROOT / 'analysis_receipt.json', receipt)
        atomic_json(ROOT / 'delivery_receipt.json', dict(recorded_utc=now(), training=compute['training_terminal'],
            new_B_test=terminal, historical_A_test_reused=lock['control_reuse'], analysis=receipt,
            lock_path=str(ROOT / 'final_test_lock.json'), decision=decision,
            parent_reviews_goal_audit_memory_and_Git_pending=True, worker_Git_actions=False,
            Git_alone_contains_all_inference_assets=False, R58='fully deferred; eight legacy B semantics disclosed'))
        print(json.dumps(dict(analysis=receipt, decision=decision)), flush=True)
    except BaseException as error:
        atomic_json(ROOT / 'analysis_failure.json', dict(recorded_utc=now(), error_type=type(error).__name__,
            message=str(error), traceback=traceback.format_exc(), marker_preserved=True, automatic_retry=False))
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--authorize-analysis', action='store_true')
    args = parser.parse_args(argv)
    if not args.authorize_analysis:
        parser.error('Require explicit conditional parent authorization for ONCE CPU analysis')
    summarize()


if __name__ == '__main__':
    main()
