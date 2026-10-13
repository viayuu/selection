"""ONCE CPU R71 delivery analysis of saved JSON/update logs; no NPZ replay."""

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
from .r65_analysis import historical_records
from .r67_analysis import read_updates
from .r68_analysis import numeric_summary, selected_figures
from .r70_analysis import metric_rows as saved_metric_rows
from .r71_experiment import ROOT, RUN_NAME
from .r71_frozen_evaluation import completed_test, exclusive_json, now, read


NAMES = {'A': 'R71A_reused_R70B23', 'B': 'R71B_joint_mixture'}
LABELS = {'A': 'Reused R70B23', 'B': 'New R71B joint mixture'}
DENSITY_METRICS = ('density_nll_per_dimension', 'density_joint_nll')


def metric_rows(model, split, epoch, value):
    rows = saved_metric_rows(model, split, epoch, value)
    for row in rows:
        tasks = list(value['per_problem']) if row['kind'] == 'aggregate' else (
            [task for task in value['per_problem'] if task not in ('TSP', 'CVRP', 'ATSP')]
            if row['scope'] == 'MVRP' else [row['scope']])
        entries = [value['per_problem'][task] for task in tasks]
        available = all(entry.get('density_nll_per_dimension') is not None for entry in entries)
        row['density_nll_per_dimension'] = float(np.mean([entry['density_nll_per_dimension'] for entry in entries])) if available else None
        row['density_joint_nll'] = float(np.mean([entry['density_nll_per_dimension'] * (len(entry['pool']) - 1)
                                                for entry in entries])) if available else None
        row['density_joint_nll_source'] = 'derived from saved task NLL/D and fixed D' if available else None
    return rows


def export_curves(histories, results, updates):
    learning, optimization, tasks = [], [], []
    for arm, history in histories.items():
        for row in history:
            identity = dict(arm=arm, model=NAMES[arm], execution='historical reuse' if arm == 'A' else 'new formal training',
                global_epoch=row['global_epoch'])
            optimization.append(dict(identity, epoch=row['epoch'], successful_updates=row['successful_updates'],
                selected_rows=row['presentations'], cumulative_encoder_view_rows=row['successful_view_rows'],
                cumulative_attempted_encoder_view_rows=row['attempted_view_rows'],
                historical_mixed_head_rows=row.get('successful_mixed_head_rows') if arm == 'A' else None,
                **row['optimization']))
            tasks += [dict(identity, epoch=row['epoch'], problem=task, **entry) for task, entry in row['task_optimization'].items()]
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
    write_csv(ROOT / 'task_optimization.csv', tasks)
    cumulative = 0.
    update_rows = []
    for row in updates:
        cumulative += row['seconds']
        update_rows.append({key: json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else value
            for key, value in dict(row, model=NAMES['B'], cumulative_logged_update_seconds=cumulative).items()})
    write_csv(ROOT / 'update_curves.csv', update_rows)
    return learning


def summaries(histories, results):
    rows, common, availability = [], [], []
    for arm, history in histories.items():
        for label in ('best', 'final'):
            record = results[arm][label]
            for split in ('train', 'val'):
                if record.get(split) is not None:
                    rows += [dict(arm=arm, summary=label, global_epoch=record['global_epoch'], **entry)
                             for entry in metric_rows(NAMES[arm], split, record['epoch'], record[split])]
        last = [metric_rows(NAMES[arm], 'val', row['epoch'], row['val']) for row in history[-5:]]
        for index, entry in enumerate(last[0]):
            for label, function in (('last5_mean', np.mean), ('last5_population_sd', np.std), ('last5_min', np.min), ('last5_max', np.max)):
                summary = dict(model=NAMES[arm], arm=arm, split='val', kind=entry['kind'], scope=entry['scope'],
                    summary=label, actual_epochs='36,37,38,39,40', not_seed_uncertainty=True)
                for key in METRICS + DENSITY_METRICS:
                    values = [record[index].get(key) for record in last]
                    summary[key] = float(function(values)) if all(value is not None for value in values) else None
                rows.append(summary)
        actual = {row['epoch'] for row in history[-5:] if row.get('train') is not None}
        if 36 <= results[arm]['best']['epoch'] <= 40:
            actual.add(results[arm]['best']['epoch'])
        availability.append(dict(model=NAMES[arm], scope='last-five36-40 TRAIN mean', available=False,
            actual_train_epochs=','.join(map(str, sorted(actual))), reason='Sparse actual TRAIN records only; no interpolation'))
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
                    **{arm + '_' + key: entry.get(key) for arm, entry in (('A', a), ('B', b)) for key in METRICS + DENSITY_METRICS},
                    **{key + '_B_minus_A': b[key] - a[key] for key in METRICS}))
    write_csv(ROOT / 'stage2_summary.csv', rows)
    write_csv(ROOT / 'matched_common_epochs.csv', common)
    write_csv(ROOT / 'summary_availability.csv', availability)


def comparison_tables(selected):
    rows = [row for model, record in selected.items() for split in ('train', 'val', 'test')
            if record.get(split) is not None for row in metric_rows(model, split, record['epoch'], record[split])]
    write_csv(ROOT / 'comparison.csv', rows)
    write_csv(ROOT / 'selected_metrics.csv', [row for row in rows if row['model'] in NAMES.values()])
    write_csv(ROOT / 'metric_availability.csv', [dict(model=model, split=split, available=record.get(split) is not None,
        policy='saved actual JSON only; missing records remain missing; no NPZ replay or inference')
        for model, record in selected.items() for split in ('train', 'val', 'test')])
    indexed = {(row['model'], row['split'], row['kind'], row['scope']): row for row in rows}
    deltas, gains = [], []
    for reference in [model for model in selected if model != NAMES['B']]:
        for split in ('train', 'val', 'test'):
            signs = Counter()
            for key, reference_row in indexed.items():
                if key[0] != reference or key[1] != split:
                    continue
                new = indexed.get((NAMES['B'], split, key[2], key[3]))
                if new is None:
                    continue
                counts = (new['native_correct'], reference_row['native_correct'])
                deltas.append(dict(left_model=NAMES['B'], right_model=reference, split=split, kind=key[2], scope=key[3],
                    native_correct_delta=counts[0] - counts[1] if all(type(count) is int for count in counts) else None,
                    **{metric + ('_delta_pp' if metric.startswith('top') else '_delta'):
                       (new[metric] - reference_row[metric]) * (100 if metric.startswith('top') else 1) for metric in METRICS}))
                if key[2] == 'task':
                    change = new['top1'] - reference_row['top1']
                    signs['gains' if change > 0 else 'regressions' if change < 0 else 'ties'] += 1
            if signs:
                gains.append(dict(left_model=NAMES['B'], right_model=reference, split=split,
                    **{key: signs[key] for key in ('gains', 'regressions', 'ties')}))
    write_csv(ROOT / 'comparison_deltas.csv', deltas)
    write_csv(ROOT / 'task_gain_regression.csv', gains)
    return rows, gains


def figures(plt, curves, histories, updates, learning):
    fig, axes = plt.subplots(3, 3, figsize=(16, 11))
    for ax, metric in zip(axes.flat, METRICS + DENSITY_METRICS):
        for arm in ('A', 'B'):
            for split in ('train', 'val'):
                points = [row for row in learning if row['kind'] == 'aggregate' and row['arm'] == arm
                          and row['split'] == split and row['measurement'] == 'actual epoch' and row.get(metric) is not None]
                if points:
                    ax.plot([row['epoch'] for row in points], [row[metric] for row in points],
                        linestyle='None' if split == 'train' else '-', marker='.' if split == 'train' else None,
                        label=LABELS[arm] + ' ' + split)
        ax.set(title=metric, xlabel='Actual fine-tuning epoch/global41-80')
        ax.grid(alpha=.2)
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(curves / 'learning_curves.png', dpi=150)
    plt.close(fig)
    keys = ('loss', 'nll_per_dimension', 'joint_nll', 'lr', 'grad_norm', 'clip_coefficient', 'retries', 'scale',
        'successful_view_rows', 'attempted_view_rows', 'view_presentations', 'attempted_view_presentations',
        'seconds', 'peak_memory_gib', 'peak_reserved_gib', 'physical_used_gib')
    for resolution in ('update', 'epoch'):
        records = updates if resolution == 'update' else [dict(row['optimization'], epoch=row['epoch']) for row in histories['B']]
        xkey = 'successful_update' if resolution == 'update' else 'epoch'
        fig, axes = plt.subplots(4, 4, figsize=(17, 13))
        for ax, key in zip(axes.flat, keys):
            actual = 'amp_retries' if resolution == 'epoch' and key == 'retries' else key
            points = [row for row in records if row.get(actual) is not None]
            if points:
                ax.plot([row[xkey] for row in points], [row[actual] for row in points], linewidth=.7)
            ax.set(title=key, xlabel=xkey)
            ax.grid(alpha=.2)
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
            ax.set(title=scope, xlabel='Actual fine-tuning epoch', ylabel='Native Top1')
            ax.grid(alpha=.2)
            ax.legend(fontsize=7)
        for ax in list(axes.flat)[len(names):]:
            ax.axis('off')
        fig.tight_layout()
        fig.savefig(curves / (kind + '_top1.png'), dpi=150)
        plt.close(fig)


def actual_compute(protocol, results, updates, test, terminal):
    result, prior = results['B'], results['A']
    sums = dict(successful_updates=len(updates), selected_rows=sum(row['rows'] for row in updates),
        successful_view_rows=sum(row['successful_view_rows'] for row in updates),
        attempted_view_rows=sum(row['attempted_view_rows'] for row in updates), amp_retries=sum(row['retries'] for row in updates))
    if any(sums[key] != result['presentations' if key == 'selected_rows' else key] for key in sums):
        raise ValueError('Raw formal-update counters must match the published final result')
    if (sums['successful_updates'], sums['selected_rows'], sums['successful_view_rows']) != (9600, 7200000, 14400000):
        raise ValueError('Preserve the exact formal budget, excluding capacity')
    new = dict(sums, mixed_head_rows=0, wall_seconds=result['wall_seconds'], logged_update_seconds=sum(row['seconds'] for row in updates),
        peak_allocated_gib=max(row['peak_memory_gib'] for row in updates), peak_reserved_gib=max(row['peak_reserved_gib'] for row in updates),
        peak_physical_gib=max(row['physical_used_gib'] for row in updates),
        grad_norm=numeric_summary([row['grad_norm'] for row in updates]),
        clip_coefficient=numeric_summary([row['clip_coefficient'] for row in updates]), amp_scale=numeric_summary([row['scale'] for row in updates]),
        joint_NLL=numeric_summary([row['joint_nll'] for row in updates]), NLL_per_dimension=numeric_summary([row['nll_per_dimension'] for row in updates]),
        complete_test_evaluation_seconds=test['models']['B']['complete_evaluation_seconds'])
    historical = dict(successful_updates=prior['successful_updates'], selected_rows=prior['presentations'],
        successful_view_rows=prior['successful_view_rows'], attempted_view_rows=prior['attempted_view_rows'],
        successful_mixed_head_rows=prior['successful_mixed_head_rows'], attempted_mixed_head_rows=prior['attempted_mixed_head_rows'],
        amp_retries=prior['amp_retries'], wall_seconds=prior['wall_seconds'], original_test_already_executed=1, new_R71_tests=0, new_R71_compute=0)
    if (historical['successful_updates'], historical['selected_rows'], historical['amp_retries']) != (9600, 7200000, 5) or abs(historical['wall_seconds'] - 3668.194598888047) > 1e-6:
        raise ValueError('Fixed A expenditure must remain historical R70B, unchanged')
    inherited_path = Path(protocol['inherited_identity']['terminal']['path']).parent / 'stage1_result.json'
    inherited, capacity = read(inherited_path), read(ROOT / 'capacity_receipt.json')
    compute = dict(new_B=new, historical_reused_A=historical,
        inherited_stage1_once=dict(successful_updates=inherited['successful_updates'], selected_rows=inherited['presentations'],
            wall_seconds=inherited['wall_seconds'], counted_once=True, new_compute=0, result_path=str(inherited_path)),
        discarded_capacity=dict(updates=1, selected_rows=768, encoder_view_rows=capacity['values']['successful_view_rows'],
            wall_seconds=capacity['wall_seconds'], update_seconds=capacity['update_seconds'],
            optimizer_moment_gpu_bytes=capacity['optimizer_moment_gpu_bytes'], storage=capacity['optimizer_storage'],
            retries=capacity['values']['retries'], excluded_from_formal_clock_budget_and_runtime=True),
        parameter_counts=read(ROOT / 'parameter_counts.json'), parameter_capacity_matched=False, FLOPs_matched=False,
        retained_parameter_active_update_recipe_matched=True, density_head_and_probability_compute_differ=True,
        target_supervision_different=True, covariance_only_contrast=False, fresh_encoders=False, independent_control_replication=False,
        training_terminal=read(ROOT / 'training_terminal_receipt.json'), test_terminal=terminal,
        new_test_wall_seconds=test['wall_seconds'], solver_executions_during_delivery=0,
        timing_scope='whole full18 probability evaluation; not singleton latency or solver runtime')
    atomic_json(ROOT / 'compute_summary.json', compute)
    counters = read(ROOT / RUN_NAME / 'parameter_step_counts.json')
    write_csv(ROOT / 'parameter_optimizer_steps.csv', [dict(parameter=name, adam_steps=steps) for name, steps in counters['steps'].items()])
    write_csv(ROOT / 'branch_compute.csv', [dict(problem=task, branch_id=index, active_updates=result['updates_by_problem'][task],
        selected_rows=result['rows_by_problem'][task], successful_encoder_view_rows=2 * result['rows_by_problem'][task],
        parameters=compute['parameter_counts']['branch_counts'][task]) for index, task in enumerate(PROBLEMS)])
    return compute


def report(selected, lock, decision, compute, gains):
    lines = ['# R71 Locked Delivery', '',
        f"VAL-frozen official arm {decision['official_arm']}; original native TEST {decision['official_test_correct']}/18000 = {100 * decision['official_test_top1']:.6f}%. Numeric70% threshold:12600/18000. Goal70 remains ACTIVE/unachieved; parent whole-objective audit and reviews remain required.", '',
        'Exactly one new B trained and tested. Fixed A is EXACT complete R70B23/global63/8709 VAL correct. Its original R70 test_results.json/models/B object is reused unchanged; new TEST counts A0/B1, original A already1. No A weights/load/training/inference/recount/reselection/retest. Protocol equivalence is not independent replication.', '',
        f"B selected common epoch{lock['models']['B']['epoch']}/global{lock['models']['B']['global_epoch']}, VAL{lock['models']['B']['native_correct']}/18000: earliest maximum INTEGER original full18 VAL count. Official B requires strictly>8709; otherwise fixed A. All18 branches, normalizers, canonical IDs and fixed Sobol tables were locked before the new TEST. TEST, costs, NLL and task gains do not select.", '',
        '| Model | Split | Top1% | Top2% | Top3% | Winner CE | Mean cost | vsSBS% | Actual regret% | NLL/D |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for model, record in selected.items():
        for split in ('train', 'val', 'test'):
            if record.get(split) is None:
                continue
            value = record[split]['macro']
            nll = value.get('macro_density_nll_per_dimension')
            lines.append(f"| {model} | {split} | {100 * value['macro_top1']:.6f} | {100 * value['macro_top2']:.6f} | {100 * value['macro_top3']:.6f} | {value['macro_ce']:.6f} | {value['macro_mean_cost']:.6f} | {value['macro_vs_sbs_pct']:.6f} | {value['macro_actual_regret_pct']:.6f} | {nll if nll is not None else 'unavailable'} |")
    lines += ['',
        'Comparison CSVs include equal-weight full18 macros, TSP/CVRP/ATSP/MVRP families and every task for TRAIN/VAL/TEST. Histories retain actual best/final/last-five VAL and common actual epochs. Sparse TRAIN every5/final/selected-best stays sparse; no interpolation, five-epoch TRAIN mean, seed uncertainty or averaged checkpoint is asserted.', '',
        'This contrasts the full joint-likelihood/readout/winner-probability package against R70, not covariance alone. Full continuous cost-difference supervision differs from winner CE/mixup. Joint NLL and NLL/D may be negative and are separate from native-winner probability CE. Lower density NLL alone does not establish selection benefit; continuous densities approximate exact cost-tie atoms and may fit irrelevant variation or overfit.', '',
        'B retains the ORIGINAL pre-fine-tuning R67B terminal stage1 averaged-x backbone, replaces/deletes the scalar head, and trains all retained parameters/new density heads. It is not a from-scratch encoder. B has38231040 parameters versus A33173154; capacity and head/inference compute are unmatched. The retained-parameter active-update recipe remains1600 updates/400000 rows per branch.', '',
        'Frozen inference uses one encoder pass/query, eight full-covariance components, the SAME2048 persistent Sobol-normal vectors/task for every component/query, streaming one component and256 vectors. It undoes TRAIN-only normalization, inserts reference zero, counts minima over all legal candidates, weights frequencies by mixture probabilities, and selects max probability/global-ID ties. Only reporting CE uses floor1e-12.16384 predicted vectors/query are not routes or solver executions. Reported density NLL reuses the same encoded parameters.', '',
        '| Expenditure | Updates | Selected rows | Successful/attempted encoder views | Retries | Wall seconds |',
        '|---|---:|---:|---:|---:|---:|']
    for name in ('new_B', 'historical_reused_A'):
        item = compute[name]
        lines.append(f"| {name} | {item['successful_updates']} | {item['selected_rows']} | {item['successful_view_rows']}/{item['attempted_view_rows']} | {item['amp_retries']} | {item['wall_seconds']:.6f} |")
    inherited, capacity, new = compute['inherited_stage1_once'], compute['discarded_capacity'], compute['new_B']
    lines += ['',
        f"Inherited stage1 counted ONCE separately: {inherited['successful_updates']} updates/{inherited['selected_rows']} rows/{inherited['wall_seconds']:.6f}s. The ONE discarded capacity update adds768 TRAIN rows/{capacity['wall_seconds']:.6f}s total ({capacity['update_seconds']:.6f}s update), with{capacity['optimizer_moment_gpu_bytes']} bytes of entire-bank Adam moments materialized. It is outside formal9600; mutated state was discarded and formal initialization recreated. New B has zero mixed-head rows; fixed A's historical mixup work stays historical.", '',
        f"New B peak allocated/reserved/physical memory: {new['peak_allocated_gib']:.6f}/{new['peak_reserved_gib']:.6f}/{new['peak_physical_gib']:.6f}GiB. Whole new TEST evaluation: {new['complete_test_evaluation_seconds']:.6f}s including probability integration; not singleton latency or solver runtime.144 wrapper/active-branch/encoder batches cover all18000 rows including104-row tails. Zero solvers executed; deployment completes one selected solver.", '',
        'Optimization CSVs/curves preserve actual jointNLL/NLL-D/total objective/LR/gradient/clip/AMP/retries/successful-vs-attempted encoder views/time/memory. No R71 cleanCE/mixCE/KL terms are invented. Raw formal counters determine actual expenditure; discarded capacity is extra.', '',
        'Historical TEST exposure is disclosed. Saved locked historical references are descriptive only; no reference inference or NPZ replay occurred. NEW R71 TEST never changes this model/budget/rule and is excluded from successor planning. One seed supplies no seed-uncertainty or accuracy-ceiling claim.', '',
        'Weights, NPZ probabilities/selection scores/predictions/native winners/raw FP64 costs/indices/nodes/global IDs/names, rawupdates, W&B/cache and regenerable schedules stay LOCAL. Git alone is not all inference assets; this worker performs no Git. Parent owns actual Astra reviews, fresh review, whole-objective audit, memory and one completed publication.', '',
        'R58 and eight legacy B feasible-domain semantic inconsistencies remain fully deferred/disclosed/unfixed; they do not veto native70 numerics.', '']
    for entry in gains:
        if entry['right_model'] in (NAMES['A'], 'R60B'):
            lines.append(f"{entry['split']} B versus {entry['right_model']}: {entry['gains']} task gains/{entry['regressions']} regressions/{entry['ties']} ties.")
    text = '\n'.join(lines) + '\n'
    (ROOT / 'RESULTS.md').write_text(text)
    (ROOT / 'REPORT.md').write_text(text)


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
        results = {'A': read(ROOT / 'reused_A_result.json'), 'B': read(ROOT / RUN_NAME / 'result.json')}
        histories = {'A': read(ROOT / 'reused_A_history.json'), 'B': read(ROOT / RUN_NAME / 'history.json')}
        if results['A']['best']['epoch'] != 23 or results['A']['best']['val']['native_correct'] != 8709:
            raise ValueError('Fixed control metadata is the original R70B23 selection')
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
        comparison, gains = comparison_tables(selected)
        official = lock['official_arm']
        correct = test['models'][official]['native_correct']
        decision = dict(official_arm=official, official_model=NAMES[official], official_test_correct=correct,
            official_test_total=18000, official_test_top1=correct / 18000, native_70pct_threshold_correct=12600,
            numeric_threshold_reached=correct >= 12600, validation_count_B_minus_A=lock['models']['B']['native_correct'] - 8709,
            B_vs_fixed_A_test_delta_pp=100 * (test['models']['B']['macro']['macro_top1'] - test['models']['A']['macro']['macro_top1']),
            B_vs_fixed_R60B_test_delta_pp=100 * (test['models']['B']['macro']['macro_top1'] - selected['R60B']['test']['macro']['macro_top1']),
            whole_objective_audit_passed=False, claim_70pct_supported=False,
            goal_status='ACTIVE/unachieved; parent objective audit and actual reviews pending', one_new_arm=True, one_seed=True,
            parameter_capacity_matched=False, FLOPs_matched=False, covariance_only_contrast=False,
            historical_test_exposure=True, new_B_test_used_for_planning=False, test_reselection=False, cost_veto=False)
        provenance[NAMES['A']] = lock['control_reuse']
        provenance[NAMES['B']] = dict(result_path=str(ROOT / RUN_NAME / 'result.json'), test_path=str(ROOT / 'test_results.json'),
            test_json_pointer='/models/B', checkpoint=lock['models']['B'], evaluation='ONE new original full18 frozen-probability TEST')
        atomic_json(ROOT / 'reference_provenance.json', dict(recorded_utc=now(), references=provenance,
            historical_test_exposure=True, reference_inference=False, NPZ_replay=False, solver_executions=0))
        report(selected, lock, decision, compute, gains)
        atomic_json(ROOT / 'analysis.json', dict(recorded_utc=now(), decision=decision, actual_compute=compute,
            selected={model: dict(epoch=record['epoch'], **{split: record[split]['macro'] for split in ('train', 'val', 'test')
                if record.get(split) is not None}) for model, record in selected.items()},
            new_arms=['B'], reused_control=NAMES['A'], legacy_B_semantics='eight mismatches disclosed/unfixed; R58 deferred'))
        receipt = dict(recorded_utc=now(), returncode=0, cpu_only=True, invocations=1, wall_seconds=time.monotonic() - started,
            comparison_rows=len(comparison), png_count=len(list(curves.rglob('*.png'))), inference=False,
            NPZ_replay=False, new_arms=['B'], reused_arms=['A'])
        atomic_json(ROOT / 'analysis_receipt.json', receipt)
        atomic_json(ROOT / 'delivery_receipt.json', dict(recorded_utc=now(), training=compute['training_terminal'],
            new_B_test=terminal, historical_A_test_reused=lock['control_reuse'], analysis=receipt,
            lock_path=str(ROOT / 'final_test_lock.json'), decision=decision, new_tests_executed={'A': 0, 'B': 1},
            original_A_tests_already_executed=1, parent_reviews_goal_audit_memory_and_Git_pending=True,
            worker_Git_actions=False, Git_alone_contains_all_inference_assets=False,
            R58='fully deferred; eight legacy B semantics disclosed/unfixed'))
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
        parser.error('Require the parent-authorized ONCE CPU analysis after actual TEST terminal0')
    summarize()


if __name__ == '__main__':
    main()
