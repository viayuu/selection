"""ONCE CPU reporting of the completed R74 live TEST and saved TRAIN/VAL records."""

import argparse
import json
import os
import time
import traceback

import numpy as np

from .r64_analysis import METRICS, write_csv
from .r67_analysis import read_updates
from .r68_analysis import numeric_summary
from .r70_analysis import metric_rows
from .r74_delivery import (CONTROL, CONTROL_RUN, ROOT, RUN, completed_test, completed_training,
                           exclusive, now, read, write)


NAMES = {'A': 'R74A_reused_R73B23', 'B': 'R74B_bounded_adaptive_acquisition'}


def rows_for(arm, split, epoch, value, measurement='saved original native evaluation'):
    rows = metric_rows(NAMES[arm], split, epoch, value)
    for row in rows:
        row.update(arm=arm, measurement=measurement, CE_definition='native-winner utility CE; not calibrated probability')
    return rows


def learning_rows(histories, results):
    learning, summaries, availability = [], [], []
    for arm in ('A', 'B'):
        history, result = histories[arm], results[arm]
        train_epochs = set()
        for record in history:
            for split in ('train', 'val'):
                if record.get(split) is None:
                    continue
                if split == 'train':
                    train_epochs.add(record['epoch'])
                learning += [dict(row, global_epoch=record['global_epoch'], successful_updates=record['successful_updates'],
                    historical_reuse=arm == 'A') for row in rows_for(arm, split, record['epoch'], record[split],
                        'historical control reuse' if arm == 'A' else 'actual cached hard-policy evaluation')]
        best = result['best']
        if best['epoch'] not in train_epochs:
            learning += [dict(row, global_epoch=best['global_epoch'], successful_updates=best['successful_updates'],
                historical_reuse=arm == 'A') for row in rows_for(arm, 'train', best['epoch'], best['train'],
                    'actual selected full native TRAIN; historical reuse' if arm == 'A' else 'actual selected full native TRAIN')]
            train_epochs.add(best['epoch'])
        for name in ('best', 'final'):
            record = result[name]
            for split in ('train', 'val'):
                if record.get(split) is not None:
                    summaries += [dict(row, summary=name, global_epoch=record['global_epoch'])
                        for row in rows_for(arm, split, record['epoch'], record[split])]
        last = [rows_for(arm, 'val', record['epoch'], record['val']) for record in history[-5:]]
        for index, row in enumerate(last[0]):
            summary = dict(model=NAMES[arm], arm=arm, split='val', kind=row['kind'], scope=row['scope'],
                summary='last5_arithmetic_mean', actual_epochs='36,37,38,39,40', checkpoint_average=False,
                seed_uncertainty=False)
            for key in METRICS + ('native_correct',):
                values = [record[index].get(key) for record in last]
                summary[key] = float(np.mean(values)) if all(value is not None for value in values) else None
            summaries.append(summary)
        availability.append(dict(arm=arm, scope='last-five TRAIN mean',
            available=set(range(36, 41)) <= train_epochs,
            actual_measured_epochs=','.join(map(str, sorted(train_epochs & set(range(36, 41))))),
            reason='Only saved sparse full native measurements; no dense or interpolated TRAIN values'))
    write_csv(ROOT / 'learning_curves.csv', learning)
    write_csv(ROOT / 'stage_summary.csv', summaries)
    write_csv(ROOT / 'metric_availability.csv', availability)
    return learning, summaries, availability


def comparisons(histories, results, test):
    selected = {arm: dict(train=results[arm]['best']['train'], val=results[arm]['best']['val'], test=test['models'][arm])
                for arm in ('A', 'B')}
    rows = [row for arm in ('A', 'B') for split, value in selected[arm].items()
            for row in rows_for(arm, split, results[arm]['best']['epoch'], value)]
    write_csv(ROOT / 'comparison.csv', rows)
    gains = []
    for split in ('train', 'val', 'test'):
        a, b = rows_for('A', split, results['A']['best']['epoch'], selected['A'][split]), rows_for('B', split, results['B']['best']['epoch'], selected['B'][split])
        for left, right in zip(a, b):
            if (left['kind'], left['scope']) != (right['kind'], right['scope']):
                raise ValueError('Saved A/B metrics must retain identical original full18 task/family scopes')
            gains.append(dict(split=split, kind=left['kind'], scope=left['scope'],
                native_correct_delta=right['native_correct']-left['native_correct'],
                top1_delta_pp=100*(right['top1']-left['top1']), top2_delta_pp=100*(right['top2']-left['top2']),
                top3_delta_pp=100*(right['top3']-left['top3']), native_utility_CE_delta=right['ce']-left['ce'],
                mean_cost_delta=right['mean_cost']-left['mean_cost'],
                actual_regret_pct_delta=right['actual_regret_pct']-left['actual_regret_pct'],
                vs_sbs_pct_delta=right['vs_sbs_pct']-left['vs_sbs_pct']))
    write_csv(ROOT / 'gains.csv', gains)
    common = []
    for a, b in zip(histories['A'], histories['B']):
        if a['global_epoch'] != b['global_epoch']:
            raise ValueError('Only actual common data-schedule epochs may be compared')
        for split in ('train', 'val'):
            if a.get(split) is None or b.get(split) is None:
                continue
            left, right = rows_for('A', split, a['epoch'], a[split]), rows_for('B', split, b['epoch'], b[split])
            common += [dict(global_epoch=a['global_epoch'], split=split, kind=x['kind'], scope=x['scope'],
                A_native_correct=x['native_correct'], B_native_correct=y['native_correct'],
                top1_delta_pp=100*(y['top1']-x['top1'])) for x, y in zip(left, right)]
    write_csv(ROOT / 'common_actual_epochs.csv', common)
    lines = ['# R74 Selected Native Comparisons', '', '| Split | Arm | Correct | Top1% | Top2% | Top3% | Utility CE | Actual regret% |',
             '|---|---|---:|---:|---:|---:|---:|---:|']
    for row in rows:
        if row['kind'] == 'aggregate':
            lines.append(f'| {row["split"]} | {row["arm"]} | {row["native_correct"]}/{row["native_total"]} | '
                f'{100*row["top1"]:.6f} | {100*row["top2"]:.6f} | {100*row["top3"]:.6f} | '
                f'{row["ce"]:.6f} | {row["actual_regret_pct"]:.6f} |')
    lines += ['', 'A is the unchanged historical R73B23 object. B is the one locked complete R74 hard policy.',
        'All task and family metrics, mean costs and vsSBS are in comparison.csv. No separate component evaluation.',
        'Utility CE is a reporting diagnostic, not a calibrated probability claim.']
    (ROOT / 'comparison.md').write_text('\n'.join(lines)+'\n')
    return selected, rows, gains


def optimization_rows(updates, history, test):
    fields = ('successful_update', 'epoch', 'global_epoch', 'rows', 'native_presentations',
        'successful_leaf_view_rows', 'loss', 'native_ce', 'kl', 'consistency_weight', 'lr', 'grad_norm',
        'clip_coefficient', 'scale', 'initial_scale', 'retries', 'attempts', 'cumulative_retries',
        'seconds', 'wall_seconds', 'allocated_bytes', 'reserved_bytes', 'peak_allocated_bytes',
        'peak_reserved_bytes', 'physical_used_bytes', 'physical_total_bytes')
    rows, gate_rows = [], []
    for record in updates:
        row = {key: record[key] for key in fields}
        row.update({'cumulative_attempted_'+key: value for key, value in record['cumulative_attempted'].items()})
        rows.append(row)
        for problem, task in zip(record['task_indices'], record['task_losses']):
            gate_rows.append(dict(successful_update=record['successful_update'], problem=problem,
                **task['gate_probabilities']))
    write_csv(ROOT / 'update_curves.csv', rows)
    write_csv(ROOT / 'optimization_gate_probabilities.csv', gate_rows)
    path_rows = []
    for record in history:
        for key, count in record['val']['hard_path_counts'].items():
            problem, path = key.rsplit('/', 1)
            path_rows.append(dict(split='val', epoch=record['epoch'], problem=problem, hard_path=int(path), count=count))
    for key, count in test['models']['B']['hard_path_counts'].items():
        problem, path = key.rsplit('/', 1)
        path_rows.append(dict(split='test', epoch=None, problem=problem, hard_path=int(path), count=count))
    write_csv(ROOT / 'hard_path_usage.csv', path_rows)
    return rows, path_rows


def accounting(protocol, generation, result, history, updates, test, A_result):
    if ([row['successful_update'] for row in updates] != list(range(1, 9601))
            or sum(row['rows'] for row in updates) != 7200000
            or sum(row['successful_leaf_view_rows'] for row in updates) != 38400000
            or sum(row['retries'] for row in updates) != result['amp_retries']):
        raise ValueError('Saved raw update accounting must match the original complete40E budget')
    attempts = {key: sum(row['attempted'][key] for row in updates) for key in result['attempted']}
    if attempts != result['attempted'] or updates[-1]['cumulative_attempted'] != attempts:
        raise ValueError('Preserve exact actually attempted native presentations and adapter leaf views')
    initialization, B = read(ROOT / 'initialization.json'), test['models']['B']
    history_eval = sum(row['val']['seconds']+(row['train']['seconds'] if row['train'] is not None else 0.) for row in history)
    selected_extra = result['best']['train']['seconds'] if history[result['best']['epoch']-1]['train'] is None else 0.
    return dict(new_generation=dict(counters=generation['counters'], seconds=generation['seconds'], memory=generation['memory'],
            TRAIN_distinct_rows=180000, VAL_distinct_rows=18000, frozen_F_query_rows=198000,
            TRAIN_counterfactual_initializations=330000, VAL_counterfactual_initializations=33000,
            both_checkpoints=True, counterfactual_not_selected_path_timing=True, probe_states=generation['probes']),
        new_policy_training=dict(epochs=40, successful_updates=9600, native_presentations=7200000,
            successful_adapter_leaf_view_rows=38400000, attempted=attempts, retries=result['amp_retries'],
            frozen_F_training_forwards=0, probe_forwards_during_optimization=0,
            new_trainable_parameters=initialization['trainable_policy_parameters'],
            whole_model_parameters=initialization['full_model_parameters'], frozen_base_parameters=initialization['frozen_base_parameters'],
            wall_seconds=result['wall_seconds'], logged_update_seconds=sum(row['seconds'] for row in updates),
            cached_native_evaluation_seconds=history_eval+selected_extra, cached_evaluation_included_in_training_wall=True,
            memory=result['memory'], new_parameters_only=True, same_state_task_gradient_mean=True,
            optimizer=protocol['optimizer'], initialization='New seed2 policy plus frozen complete R73B23 averaged-x F',
            exposure_capacity_compute_matched=False, adaptivity_causally_isolated=False,
            gradients=numeric_summary([row['grad_norm'] for row in updates]), update_seconds=numeric_summary([row['seconds'] for row in updates])),
        new_live_TEST=dict(passes=1, counters=B['counters'], times=B['timings'],
            whole_evaluation_seconds=B['seconds'], source_and_model_loading_seconds=dict(
                complete_policy=B['base_model_loading_seconds'], probes=B['probe_loading_seconds']),
            setup_and_evaluation_seconds=test['wall_seconds'], actual_layout=B['actual_test_layout'],
            ordinary_batch_seconds=numeric_summary(B['ordinary_batch_seconds']),
            peak_allocated_bytes=B['peak_allocated_bytes'], peak_reserved_bytes=B['peak_reserved_bytes'],
            physical_used_bytes_at_end=B['physical_used_bytes'],
            partial_solver_runs=True, completed_solver_executions=0, new_final_solver_labels=0,
            singleton_or_end_to_end_latency_measured=False, throughput_optimality_claim=False),
        reused_A=dict(name='R73B23/global63', new_training_updates=0, new_TEST_passes=0,
            original_TEST_passes_already_executed=1, historical_training_wall_seconds=A_result.get('wall_seconds'),
            historical_teacher_wall_seconds=A_result.get('teacher_wall_seconds'),
            historical_record=str(CONTROL_RUN / 'result.json'), unchanged_history=str(CONTROL_RUN / 'history.json'),
            protocol_equivalence_not_replication=True),
        inherited_stage1=protocol['inherited_stage1'], shared_inherited_pretraining_counted_once=True,
        historical_accounting=protocol['historical_accounting'],
        historical_R73_protocol_accounting=read(CONTROL / 'protocol.json').get('historical_accounting'),
        original_CPU_check=read(ROOT / 'cpu_check.json'), delivery_syntax=read(ROOT / 'delivery_preparation.json')['syntax_check'],
        deployment='Loading + F + every actually acquired prefix/gate + final utility + one full restarted selected solver',
        prefix_deduction=False, final_completion_time_unmeasured=True, completed_solver_labels_added=0)


def figures(learning, updates, comparison, path_rows, selected_epoch):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    directory = ROOT / 'curves/delivery'
    directory.mkdir(parents=True, exist_ok=False)
    macro = [row for row in learning if row['kind'] == 'aggregate']
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for ax, metric in zip(axes.flat, ('top1', 'top2', 'top3', 'ce', 'mean_cost', 'actual_regret_pct')):
        for arm, color in (('A', '#666666'), ('B', '#007c78')):
            for split in ('val', 'train'):
                rows = sorted((row for row in macro if row['arm'] == arm and row['split'] == split), key=lambda row: row['successful_updates'])
                x, y = [row['successful_updates'] for row in rows], [row[metric] for row in rows]
                if split == 'val':
                    ax.plot(x, y, color=color, label=arm+' VAL')
                else:
                    ax.scatter(x, y, color=color, s=13, marker='x' if arm == 'A' else 'o', label=arm+' sparse TRAIN')
        ax.set(title=metric, xlabel='Actual successful updates in the saved stage')
        ax.grid(alpha=.2)
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(directory / 'native_learning.png', dpi=140)
    plt.close(fig)
    fig, axes = plt.subplots(3, 4, figsize=(16, 10))
    x = [row['successful_update'] for row in updates]
    fields = ('loss', 'native_ce', 'kl', 'lr', 'grad_norm', 'clip_coefficient', 'scale', 'cumulative_retries',
              'successful_leaf_view_rows', 'cumulative_attempted_leaf_view_rows', 'wall_seconds', 'peak_allocated_bytes')
    for ax, key in zip(axes.flat, fields):
        ax.plot(x, [row[key] for row in updates], color='#007c78', linewidth=.8)
        ax.set(title=key, xlabel='Actual successful R74 updates')
        ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(directory / 'optimization.png', dpi=140)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, kind in zip(axes, ('family', 'task')):
        names = list(dict.fromkeys(row['scope'] for row in comparison if row['kind'] == kind))
        positions = np.arange(len(names))
        for index, (arm, split) in enumerate((('A', 'val'), ('B', 'val'), ('A', 'test'), ('B', 'test'))):
            values = [next(row['top1']*100 for row in comparison if row['arm'] == arm and row['split'] == split
                           and row['kind'] == kind and row['scope'] == name) for name in names]
            ax.bar(positions+(index-1.5)*.2, values, width=.2, label=arm+' '+split)
        ax.set_xticks(positions, names, rotation=90 if kind == 'task' else 0)
        ax.set(ylabel='Native Top1%', title=kind+' selected VAL and original TEST')
        ax.legend(fontsize=7)
        ax.grid(axis='y', alpha=.2)
    fig.tight_layout()
    fig.savefig(directory / 'selected_VAL_TEST.png', dpi=140)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for ax, split in zip(axes, ('val', 'test')):
        values = [row for row in path_rows if row['split'] == split and (split == 'test' or row['epoch'] == selected_epoch)]
        names = list(dict.fromkeys(row['problem'] for row in values if row['problem'] not in ('TSP', 'CVRP', 'ATSP')))
        bottom = np.zeros(len(names))
        for path, label in enumerate(('MTL shallow+deep', 'MOEL shallow+deep', 'both shallow')):
            height = np.asarray([sum(row['count'] for row in values if row['problem'] == name and row['hard_path'] == path)/1000 for name in names])
            ax.bar(np.arange(len(names)), height, bottom=bottom, label=label)
            bottom += height
        ax.set_xticks(np.arange(len(names)), names, rotation=90)
        ax.set(title=split+' hard-path usage', ylabel='Fraction of original task rows')
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(directory / 'hard_paths.png', dpi=140)
    plt.close(fig)


def summarize():
    test = completed_test()
    protocol, generation, history, result, identity, official = completed_training()
    A_result, A_history = read(CONTROL_RUN / 'result.json'), read(CONTROL_RUN / 'history.json')
    if (A_result['best']['epoch'] != 23 or A_result['best']['global_epoch'] != 63
            or A_result['best']['val']['native_correct'] != 8959 or len(A_history) != 40):
        raise ValueError('Reuse exact R73B23/global63 complete40E historical control, without reselection')
    histories, results = {'A': A_history, 'B': history}, {'A': A_result, 'B': result}
    updates = read_updates(RUN / 'updates.jsonl')
    learning, summaries, availability = learning_rows(histories, results)
    selected, comparison, gains = comparisons(histories, results, test)
    update_rows, paths = optimization_rows(updates, history, test)
    compute = accounting(protocol, generation, result, history, updates, test, A_result)
    figures(learning, update_rows, comparison, paths, identity['epoch'])
    official_count = test['models'][official]['native_correct']
    threshold = official_count >= 12600
    objective = dict(official_arm=official, original_native_TEST_correct=official_count, original_native_TEST_total=18000,
        required_correct=12600, numerical_threshold_met=threshold,
        status='Numerical threshold met; parent full objective audit remains required' if threshold else 'ACTIVE/unachieved: original locked native TEST below70%',
        integrity_readiness_is_not_performance_acceptance=True)
    provenance = dict(A=dict(history=str(CONTROL_RUN / 'history.json'), result=str(CONTROL_RUN / 'result.json'),
        TEST=read(ROOT / 'control_test_reuse.json'), object_unchanged=True, no_forward_or_prediction_recount=True),
        B=dict(history=str(RUN / 'history.json'), result=str(RUN / 'result.json'),
            test=str(ROOT / 'test_results.json')+'#/models/B', lock=str(ROOT / 'final_test_lock.json')),
        inherited_stage1_counted_once=True, historical_TEST_exposure_disclosed=True,
        NEW_R74_TEST_is_not_a_successor_planning_source=True,
        inference_assets_local=['locked_inference', 'probe_states', 'source_launch', 'test_predictions',
            'test_queries.jsonl', 'feature_cache', 'B_adaptive_seed2/best_checkpoints', 'wandb'],
        Git_alone_is_not_all_inference_assets=True)
    write(ROOT / 'reference_provenance.json', provenance)
    analysis = dict(complete=True, completed_utc=now(), selected_epoch=identity['epoch'], selected_global_epoch=identity['global_epoch'],
        official_arm=official, official_arm_fixed_before_TEST=True, selected=selected, gains=gains,
        stage_summaries=summaries, availability=availability, compute=compute, provenance=provenance, objective=objective,
        new_TEST_passes={'A': 0, 'B': 1}, original_A_TEST_passes_already_executed=1,
        analysis_invocations=1, model_or_solver_forward_during_analysis=False, NPZ_replay=False,
        R58='deferred/unfixed', eight_legacy_B_semantics='deferred/unfixed; not a native70 numerical veto')
    write(ROOT / 'analysis.json', analysis)
    B_test, A_test = test['models']['B'], test['models']['A']
    lines = ['# R74 Delivery Results', '',
        f'Unique selected B: epoch{identity["epoch"]}/global{identity["global_epoch"]}, '
        f'{identity["native_correct"]}/18000 hard native VAL. Fixed A:8959/18000. Official arm:{official}.', '',
        f'Original full18 TEST: reused A {A_test["native_correct"]}/18000 ({100*A_test["macro"]["macro_top1"]:.6f}%); '
        f'new B {B_test["native_correct"]}/18000 ({100*B_test["macro"]["macro_top1"]:.6f}%).',
        f'New TEST passes A0/B1; original A already1. Official arm stayed fixed before TEST. {objective["status"]}.', '',
        f'New optimization:9600 successful updates/7.2M native presentations/38.4M adapter leaf-view rows; '
        f'{result["amp_retries"]} actual whole-update retries; {result["wall_seconds"]:.6f}s training wall.',
        f'Counterfactual TRAIN/VAL generation:{generation["seconds"]:.6f}s,198000 frozen-F query rows, '
        '330000 TRAIN+33000 VAL probe initializations; accounted separately from optimization and live TEST.',
        f'Live TEST:{B_test["seconds"]:.6f}s whole evaluation;144 base batches and18000 single-query hard policies. '
        f'Actual probe initializations:{B_test["counters"]["probe_initializations"]}; '
        'partial solver trajectories ran, zero completed solver executions and zero new final-solver labels.', '',
        'Distance/time, constraints, physical node roles, directed ATSP and all original legal candidates were retained. '
        'The live provider retained shallow state when deepening and acquired only the hard selected path.',
        'A is unchanged historical R73B23, not a fresh replication. Shared inherited pretraining is counted once. '
        'This is an information-acquisition package with added computation, not an exposure/capacity/compute-matched '
        'causal proof of adaptivity. Expected-loss training and hard routing can differ.',
        'Full deployment includes F, every acquired prefix/gate, final utility and one full restarted chosen solver. '
        'No prefix deduction, completion-time measurement, end-to-end speedup or singleton-latency claim.',
        'Native utility CE is a reporting diagnostic. TRAIN remains sparse. Historical TEST exposure is disclosed; '
        'the new TEST does not change this protocol and is excluded from successor planning.',
        'R58 and eight legacy B semantic inconsistencies remain deferred/unfixed, not a native70 numerical veto.',
        'Weights, caches, probe snapshots, raw logs, NPZ and W&B remain local; Git alone is not all inference assets. '
        'Actual parent primary/fresh reviews, objective audit and completed-version publication remain required.', '',
        'See comparison.md/comparison.csv, gains.csv, stage_summary.csv, learning_curves.csv, update_curves.csv, '
        'hard_path_usage.csv, analysis.json and curves/delivery for full18/task/family evidence.']
    (ROOT / 'RESULTS.md').write_text('\n'.join(lines)+'\n')
    (ROOT / 'REPORT.md').write_text('\n'.join(lines)+'\n')
    return objective


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--authorize-analysis', action='store_true')
    args = parser.parse_args()
    if not args.authorize_analysis or os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('Only the conditionally authorized CPU-only ONCE analysis may run')
    completed_test()
    started = time.monotonic()
    exclusive(ROOT / 'analysis_started.json', dict(utc=now(), pid=os.getpid(), invocation=1,
        command='python -B -u -m code.V4.r74_analysis --authorize-analysis', model_forwards=0,
        solver_forwards=0, GPU_operations=0, NPZ_replay=False, automatic_retry=False))
    try:
        objective = summarize()
        exclusive(ROOT / 'analysis_receipt.json', dict(complete=True, completed_utc=now(), pid=os.getpid(),
            invocation=1, seconds=time.monotonic()-started, model_forwards=0, solver_forwards=0,
            GPU_operations=0, NPZ_replay=False, objective=objective))
        print(json.dumps(dict(analysis_complete=True, invocation=1, objective=objective)), flush=True)
    except BaseException as error:
        exclusive(ROOT / 'analysis_failure.json', dict(utc=now(), error=str(error), traceback=traceback.format_exc(),
            invocation=1, automatic_retry=False))
        raise


if __name__ == '__main__':
    main()
