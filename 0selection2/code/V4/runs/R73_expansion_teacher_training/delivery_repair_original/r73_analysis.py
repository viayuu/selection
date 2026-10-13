"""ONCE CPU-only R73 delivery reporting from saved JSON; no prediction replay."""

import argparse
import json
import os
import sys
import time
import traceback

import numpy as np

from code.unified_selector.registry import PROBLEMS
from .r64_analysis import METRICS, write_csv
from .r67_analysis import read_updates
from .r68_analysis import numeric_summary
from .r70_analysis import metric_rows as saved_metric_rows
from .r73_frozen_evaluation import (CONTROL, ROOT, RUN, completed_test, exclusive, now, read, write)


NAMES = {'A': 'R73A_reused_R72_consensus', 'B': 'R73B_warm_started_shared_student'}


def metric_rows(arm, split, epoch, value, measurement='saved selected evaluation'):
    rows = saved_metric_rows(NAMES[arm], split, epoch, value)
    for row in rows:
        ce = row.pop('ce')
        row.update(arm=arm, measurement=measurement, Borda_score_CE=ce if arm == 'A' else None,
            native_utility_CE=ce if arm == 'B' else None,
            CE_definition='ordinal Borda-score CE, not calibrated probability' if arm == 'A' else 'native-winner direct-utility CE')
    return rows


def curves(history, result, updates):
    learning, summaries = [], []
    for record in history:
        for split in ('train', 'val'):
            if record.get(split) is not None:
                learning += [dict(row, successful_updates=record['successful_updates'], global_epoch=record['global_epoch'])
                    for row in metric_rows('B', split, record['epoch'], record[split], 'actual native epoch evaluation')]
    measured_train = {row['epoch'] for row in history if row.get('train') is not None}
    best = result['best']
    if best['epoch'] not in measured_train:
        learning += [dict(row, successful_updates=best['successful_updates'], global_epoch=best['global_epoch'])
            for row in metric_rows('B', 'train', best['epoch'], best['train'], 'actual selected native TRAIN evaluation')]
    for name in ('best', 'final'):
        record = result[name]
        for split in ('train', 'val'):
            if record.get(split) is not None:
                summaries += [dict(row, summary=name, global_epoch=record['global_epoch'])
                    for row in metric_rows('B', split, record['epoch'], record[split])]
    last = [metric_rows('B', 'val', row['epoch'], row['val']) for row in history[-5:]]
    for index, row in enumerate(last[0]):
        for label, function in (('last5_mean', np.mean), ('last5_population_sd', np.std), ('last5_min', np.min), ('last5_max', np.max)):
            summary = dict(model=NAMES['B'], split='val', kind=row['kind'], scope=row['scope'], summary=label,
                actual_epochs='36,37,38,39,40', not_seed_uncertainty=True)
            for key in tuple(metric for metric in METRICS if metric != 'ce') + ('native_utility_CE', 'native_correct'):
                summary[key] = float(function([entries[index][key] for entries in last]))
            summaries.append(summary)
    saved_correction = result.get('correction_harm')
    correction_path = ROOT / 'correction_harm.json'
    if correction_path.exists():
        saved_correction = read(correction_path)
    availability = [dict(model=NAMES['A'], scope='TRAIN and epoch history', available=False,
        reason='Fixed R72 has zero new training and no native TRAIN/epoch history; no control inference authorized'),
        dict(model=NAMES['B'], scope='last-five TRAIN mean', available=False,
             actual_train_epochs=sorted((measured_train | {best['epoch']}) & set(range(36, 41))),
             reason='Only actual sparse native TRAIN measurements exist; no interpolation'),
        dict(model=NAMES['B'], scope='instance correction/harm versus A', available=saved_correction is not None,
             saved_record=saved_correction,
             reason='Existing saved counts only when present; no native teacher pass or NPZ replay is added')]
    write_csv(ROOT / 'learning_curves.csv', learning)
    write_csv(ROOT / 'stage_summary.csv', summaries)
    write_csv(ROOT / 'metric_availability.csv', availability)
    fields = ('successful_update', 'epoch', 'global_epoch', 'loss', 'native_ce', 'kl', 'pseudo_ce',
        'consistency_weight', 'lr', 'grad_norm', 'clip_coefficient', 'scale', 'initial_scale', 'retries', 'attempts',
        'cumulative_retries', 'seconds', 'wall_seconds', 'native_rows', 'auxiliary_rows', 'native_presentations',
        'auxiliary_presentations', 'total_presentations', 'successful_encoder_view_rows',
        'successful_native_view_rows', 'successful_auxiliary_view_rows', 'peak_allocated_bytes', 'peak_reserved_bytes',
        'physical_used_bytes', 'physical_total_bytes')
    update_rows = []
    for row in updates:
        item = {key: row[key] for key in fields}
        item.update({'attempted_' + key: value for key, value in row['cumulative_attempted_views'].items()})
        update_rows.append(item)
    write_csv(ROOT / 'update_curves.csv', update_rows)
    return learning, availability


def compute_accounting(protocol, result, history, updates, test):
    if (len(updates) != 9600 or [row['successful_update'] for row in updates] != list(range(1, 9601))
            or sum(row['native_rows'] for row in updates) != 7200000
            or sum(row['auxiliary_rows'] for row in updates) != 7200000
            or sum(row['successful_native_view_rows'] for row in updates) != 14400000
            or sum(row['successful_auxiliary_view_rows'] for row in updates) != 7200000
            or sum(row['retries'] for row in updates) != result['amp_retries']):
        raise ValueError('Saved raw updates must account for all successful native/auxiliary presentations and retries')
    attempts = {key: sum(row['attempted'][key] for row in updates) for key in result['attempted_views']}
    if attempts != result['attempted_views'] or updates[-1]['cumulative_attempted_views'] != attempts:
        raise ValueError('Preserve exact actually attempted native versus pseudo-supervised view counts')
    teacher, release = read(ROOT / 'teacher_cache_complete.json'), read(ROOT / 'teacher_released.json')
    epoch_eval = sum(row['val']['seconds'] + (row['train']['seconds'] if row.get('train') is not None else 0.) for row in history)
    selected_extra = result['best']['train']['seconds'] if result['best']['selected_train_source'].startswith('one complete') else 0.
    historical_protocol = read(CONTROL / 'protocol.json')
    B = test['models']['B']
    return dict(new_teacher=dict(query_rows=1620000, member_row_forwards=4860000,
            counters=teacher['counters'], wall_seconds=teacher['wall_seconds'], memory=teacher['memory'],
            parameters=teacher['parameter_counts'], targets='hard predictions, not measured solver winners',
            native_teacher_forwards=0, component_evaluations=0, release=release),
        new_student=dict(epochs=40, successful_updates=9600, native_presentations=7200000,
            auxiliary_presentations=7200000, total_presentations=14400000,
            native_encoder_views=14400000, pseudo_supervised_encoder_views=7200000, total_encoder_views=21600000,
            actual_attempted_views=attempts, retries=result['amp_retries'],
            logged_update_seconds=sum(row['seconds'] for row in updates), wall_seconds=result['wall_seconds'],
            preparation_seconds=result['student_preparation_seconds'], preparation_peak_memory=None,
            native_evaluation_seconds=epoch_eval + selected_extra, native_evaluation_is_subset_of_training_wall=True,
            native_evaluation_peak_memory=None, unavailable_memory='Preparation/evaluation peaks were not separately recorded; no fabricated measurement',
            update_peak_allocated_bytes=result['peak_allocated_bytes'], update_peak_reserved_bytes=result['peak_reserved_bytes'],
            parameter_count=result['parameter_count'], capacity_or_exposure_or_compute_matched=False,
            initialization='complete warm-started R67B36/global76 averaged-x model including trained head',
            global41_80_is_data_schedule_not_entire_parameter_lifetime=True,
            distinct_native_inputs=180000, distinct_expansion_inputs=1620000,
            measured_native_winner_rows=180000, auxiliary_costs_read=False, observation_masks_read=False,
            gradient=numeric_summary([row['grad_norm'] for row in updates]),
            update_seconds=numeric_summary([row['seconds'] for row in updates])),
        historical_accounting=historical_protocol['historical_accounting'],
        historical_source=str(CONTROL / 'protocol.json') + '#/historical_accounting',
        shared_R67_R70_stage1_counted_once=True, new_R73_inherited_training=False,
        reused_A=dict(new_training_updates=0, new_TEST_passes=0, original_TEST_passes_already_executed=1,
                      selectors_per_query=3, stored_parameters=protocol['teacher_bundle']['stored_parameters'],
                      active_parameters_per_query=protocol['teacher_bundle']['active_parameters_per_query'],
                      protocol_equivalence_not_replication=True),
        new_B_test=dict(passes=1, counters=B['counters'], complete_evaluation_seconds=B['seconds'],
            forward_seconds=B['forward_seconds'], setup_and_evaluation_seconds=test['wall_seconds'],
            peak_allocated_bytes=B['peak_allocated_bytes'], peak_reserved_bytes=B['peak_reserved_bytes'],
            peak_observed_physical_used_bytes=B['peak_observed_physical_used_bytes'], selectors_per_query=1,
            whole_evaluation_is_not_singleton_or_solver_latency=True, additional_latency_benchmark=False),
        solver_executions=0, new_measured_solver_labels=0,
        original_cpu_check=read(ROOT / 'cpu_check.json'), formal_launch=read(ROOT / 'trainer_launch.json'))


def figures(plt, learning, updates, selected):
    directory = ROOT / 'curves/delivery'
    directory.mkdir(parents=True, exist_ok=False)
    macro = [row for row in learning if row['kind'] == 'aggregate']
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for ax, metric in zip(axes.flat, ('top1', 'top2', 'top3', 'native_utility_CE', 'mean_cost', 'actual_regret_pct')):
        for split, color in (('val', '#197e6b'), ('train', '#aa5b3b')):
            rows = sorted((row for row in macro if row['split'] == split), key=lambda row: row['successful_updates'])
            x, y = [row['successful_updates'] for row in rows], [row[metric] for row in rows]
            if split == 'val':
                ax.plot(x, y, color=color, label='Native VAL')
            else:
                ax.scatter(x, y, color=color, s=22, label='Sparse native TRAIN')
        if metric == 'top1':
            ax.axhline(8923 / 18000, color='#60676e', linestyle=':', label='Fixed A saved VAL')
        ax.set(title=metric, xlabel='Actual successful R73 updates')
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(directory / 'native_learning.png', dpi=150)
    plt.close(fig)
    x = [row['successful_update'] for row in updates]
    fig, axes = plt.subplots(3, 3, figsize=(15, 11))
    for key in ('native_ce', 'pseudo_ce', 'kl', 'loss'):
        axes[0, 0].plot(x, [row[key] for row in updates], label=key, linewidth=.65)
    axes[0, 0].set_title('Actual optimization components')
    axes[0, 0].legend(fontsize=8)
    axes[0, 1].plot(x, [row['lr'] for row in updates])
    axes[0, 1].set_title('ScheduleFree effective LR')
    for key in ('grad_norm', 'clip_coefficient'):
        axes[0, 2].plot(x, [row[key] for row in updates], label=key, linewidth=.7)
    axes[0, 2].set_title('Combined gradient / clip')
    axes[0, 2].legend(fontsize=8)
    axes[1, 0].plot(x, [row['scale'] for row in updates])
    axes[1, 0].set_title('AMP scale')
    axes[1, 1].plot(x, [row['cumulative_retries'] for row in updates])
    axes[1, 1].set_title('Actual cumulative retries')
    for stream, multiplier in (('native', 2), ('auxiliary', 1)):
        axes[1, 2].plot(x, [multiplier * row[stream + '_presentations'] for row in updates], label=stream + ' successful')
        axes[1, 2].plot(x, [row['cumulative_attempted_views'][stream + '_view_rows'] for row in updates],
                        linestyle=':', label=stream + ' attempted')
    axes[1, 2].set_title('Encoder-view rows by supervision')
    axes[1, 2].legend(fontsize=8)
    axes[2, 0].plot(x, [row['seconds'] for row in updates], linewidth=.6)
    axes[2, 0].set_title('Whole update seconds, including retries')
    axes[2, 1].plot(x, [row['wall_seconds'] for row in updates])
    axes[2, 1].set_title('Actual student wall seconds')
    for key in ('peak_allocated_bytes', 'peak_reserved_bytes', 'physical_used_bytes'):
        axes[2, 2].plot(x, [row[key] / 2**30 for row in updates], label=key.replace('_bytes', ''), linewidth=.7)
    axes[2, 2].set_title('Recorded GPU memory (GiB)')
    axes[2, 2].legend(fontsize=7)
    for ax in axes.flat:
        ax.set_xlabel('Actual successful R73 updates')
        ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(directory / 'optimization.png', dpi=150)
    plt.close(fig)
    for kind, scopes, filename in (('task', list(PROBLEMS), 'task_top1.png'),
                                   ('family', ['TSP', 'CVRP', 'ATSP', 'MVRP'], 'family_top1.png')):
        fig, axes = plt.subplots(2, 1, figsize=(13, 8))
        positions = np.arange(len(scopes))
        for ax, split in zip(axes, ('val', 'test')):
            for offset, arm, color in ((-.18, 'A', '#69747a'), (.18, 'B', '#197e6b')):
                value = selected[arm][split]
                points = [value['per_problem'][scope]['top1'] if kind == 'task' else value['families'][scope]['macro_top1'] for scope in scopes]
                ax.bar(positions + offset, 100 * np.asarray(points), width=.36, color=color,
                       label='A: fixed R72 consensus' if arm == 'A' else 'B: shared warm-started student')
            ax.set_xticks(positions, scopes, rotation=45 if kind == 'task' else 0,
                          ha='right' if kind == 'task' else 'center')
            ax.set(title=split.upper() + ' original full pools', ylabel='Native Top1 (%)')
            ax.legend(fontsize=8)
            ax.grid(axis='y', alpha=.2)
            ax.set_axisbelow(True)
        fig.tight_layout()
        fig.savefig(directory / filename, dpi=150)
        plt.close(fig)
    return [str(path) for path in sorted(directory.glob('*.png'))]


def report(selected, lock, decision, compute):
    lines = ['# R73 Expansion-Input Teacher Training', '',
        'One complete warm-started shared student versus unchanged historical R72 three-policy consensus. '
        'The objective, exposure, capacity and inference compute are not matched; this is not isolated coverage causality.', '',
        '| Arm | VAL correct/18000 | TEST correct/18000 | TEST Top1% | TEST Top2% | TEST Top3% | TEST mean cost | TEST vsSBS% | TEST actual regret% |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for arm in ('A', 'B'):
        val, test = selected[arm]['val'], selected[arm]['test']
        macro = test['macro']
        lines.append(f'| {arm} | {val["native_correct"]} | {test["native_correct"]} | {100*macro["macro_top1"]:.6f} | '
            f'{100*macro["macro_top2"]:.6f} | {100*macro["macro_top3"]:.6f} | {macro["macro_mean_cost"]:.9f} | '
            f'{macro["macro_vs_sbs_pct"]:.6f} | {macro["macro_actual_regret_pct"]:.6f} |')
    student, teacher = compute['new_student'], compute['new_teacher']
    lines += ['', f'Official arm is {lock["official_arm"]}, frozen using native validation only. '
        f'B selected local epoch{lock["models"]["B"]["epoch"]}/data-schedule global{lock["models"]["B"]["global_epoch"]} '
        'by earliest maximum integer correct; official B requires more than8923. TEST never changes selection.', '',
        f'Official original TEST: {decision["official_test_correct"]}/18000; threshold12600/18000. '
        'Goal70 remains ACTIVE/unachieved; parent reviews and whole-objective audit remain pending. '
        'Integrity readiness, imitation and inference compression do not establish 70% or a coverage mechanism.', '',
        f'A reporting-only Borda-score CE is {selected["A"]["test"]["macro"]["macro_ce"]:.9f}; '
        f'B native-winner utility CE is {selected["B"]["test"]["macro"]["macro_ce"]:.9f}. '
        'These are distinct diagnostics, not a shared calibrated probability scale. Density NLL is inapplicable.', '',
        'comparison.csv contains full18 equal-task macros, each task and TSP/CVRP/ATSP/MVRP families. '
        'Native TRAIN history remains sparse every5/final/selected. stage_summary.csv reports best/final and '
        'last-five actual VAL summaries; no dense TRAIN measurements or seed uncertainty are invented. '
        'No new correction/harm replay is performed.', '',
        f'The single teacher pass processed1620000 expansion queries,12672 batches,38016 member forwards and4860000 '
        f'member-row forwards in {teacher["wall_seconds"]:.6f}s. It queried only source IDs10000..99999; '
        'native teacher passes, auxiliary-cost reads, observation-mask reads and solver executions were zero. '
        'Hard pseudo-targets are predictions, not measured winners; teacher errors and discarded ranking uncertainty remain limitations.', '',
        f'New student work:9600 successful updates,7200000 native plus7200000 auxiliary presentations, '
        f'21600000 encoder-view rows (14400000 native,7200000 pseudo-supervised), {student["retries"]} whole-update retries. '
        f'Student preparation {student["preparation_seconds"]:.6f}s; student wall {student["wall_seconds"]:.6f}s, '
        f'including {student["native_evaluation_seconds"]:.6f}s of recorded native evaluation. '
        f'Update peak allocation/reservation {student["update_peak_allocated_bytes"]/2**30:.6f}/'
        f'{student["update_peak_reserved_bytes"]/2**30:.6f}GiB. '
        'Preparation and native evaluation peaks were not separately recorded and remain unavailable.', '',
        'The entire trained R67B36/global76 averaged-x selector, including its trained utility head and original '
        'normalization buffers, was inherited. Global41..80 identifies the reused data schedule, not the model parameter '
        'lifetime. Historical R60 expansion, R67 training and R70 training remain separate saved expenditure; shared '
        'R67/R70 stage1 is counted once. reference_provenance.json and analysis.json retain the original accounting.', '',
        f'A stores{compute["reused_A"]["stored_parameters"]} frozen parameters and activates'
        f'{compute["reused_A"]["active_parameters_per_query"]}/query; B stores{student["parameter_count"]}. '
        'Capacity, exposure and compute are unmatched.', '',
        f'A deploys three selectors/query and B one, both followed by one completed solver. New B full18 TEST wall '
        f'{compute["new_B_test"]["complete_evaluation_seconds"]:.6f}s includes live forwards and ordinary evaluation work; '
        f'{compute["new_B_test"]["forward_seconds"]:.6f}s was recorded in forwards. '
        'These are not singleton/completed-solver latency or a matched speedup benchmark.', '',
        'A is precisely R72 test_results.json#/models/B, reused unchanged. New TEST counts A0/B1; original A already1. '
        'No control/component checkpoint load, inference, prediction recount or retrospective test occurred. '
        'Historical TEST exposure is disclosed. Current TEST is excluded from successor planning and never changes this recipe.', '',
        'Weights, NPZ, teacher cache, raw batch/update logs, W&B and regenerable schedules stay local. Git alone is '
        'not all inference assets. R58 and eight legacy B semantic inconsistencies remain deferred/unfixed, not a '
        'native70 numerical veto. Parent owns actual reviews, objective audit, memory and one completed publication. '
        'This worker performs no Git.', '']
    text = '\n'.join(lines)
    (ROOT / 'RESULTS.md').write_text(text)
    (ROOT / 'REPORT.md').write_text(text)


def summarize():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('ONCE saved-record analysis must remain CPU-only')
    protocol, lock, test, terminal = completed_test()
    exclusive(ROOT / 'analysis_started.json', dict(utc=now(), command=' '.join([sys.executable] + sys.argv),
        invocations=1, CPU_only=True, model_forward=False, NPZ_replay=False, automatic_retry=False))
    started = time.monotonic()
    try:
        history, result = read(RUN / 'history.json'), read(RUN / 'result.json')
        updates = read_updates(RUN / 'updates.jsonl')
        selected = {'A': dict(val=read(CONTROL / 'validation_results.json'), test=test['models']['A']),
                    'B': dict(train=result['best']['train'], val=result['best']['val'], test=test['models']['B'])}
        rows = [row for arm, record in selected.items() for split in ('train', 'val', 'test') if record.get(split) is not None
                for row in metric_rows(arm, split, lock['models']['B']['epoch'] if arm == 'B' else None, record[split])]
        write_csv(ROOT / 'comparison.csv', rows)
        indexed = {(row['arm'], row['split'], row['kind'], row['scope']): row for row in rows}
        deltas = []
        for split in ('val', 'test'):
            for row in rows:
                if row['arm'] != 'B' or row['split'] != split:
                    continue
                control = indexed[('A', split, row['kind'], row['scope'])]
                deltas.append(dict(split=split, kind=row['kind'], scope=row['scope'],
                    native_correct_B_minus_A=row['native_correct'] - control['native_correct'],
                    **{key + ('_delta_pp' if key.startswith('top') else '_delta'):
                        (row[key] - control[key]) * (100 if key.startswith('top') else 1) for key in METRICS if key != 'ce'},
                    CE_difference_not_interpreted=True))
        write_csv(ROOT / 'comparison_deltas.csv', deltas)
        learning, availability = curves(history, result, updates)
        compute = compute_accounting(protocol, result, history, updates, test)
        official = lock['official_arm']
        count = test['models'][official]['native_correct']
        decision = dict(official_arm=official, official_test_correct=count, official_test_total=18000,
            official_test_top1=count / 18000, original_native_70pct_threshold_correct=12600,
            numeric_threshold_reached=count >= 12600, validation_B_minus_A=result['best']['val']['native_correct'] - 8923,
            B_minus_A_test_decisions=test['models']['B']['native_correct'] - test['models']['A']['native_correct'],
            TEST_reselection=False, whole_objective_audit_passed=False, claim_70pct_supported=False,
            goal_status='ACTIVE/unachieved; parent reviews and whole-objective audit pending',
            capacity_exposure_compute_matched=False, isolated_coverage_causality=False,
            hard_teacher_targets_are_measured_truth=False, historical_TEST_exposure=True, new_R73_TEST_used_for_planning=False)
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        plt.rcParams.update({'font.size': 9})
        paths = figures(plt, learning, updates, selected)
        provenance = dict(A=lock['control_reuse'], B=dict(result_path=str(RUN / 'result.json'),
            validation_json_pointer='/best/val', test_path=str(ROOT / 'test_results.json'), test_json_pointer='/models/B',
            locked_checkpoint=lock['models']['B'], new_original_TEST_passes=1),
            historical_accounting_path=compute['historical_source'], historical_TEST_exposure=True,
            reference_inference=False, NPZ_replay=False, new_TEST_excluded_from_planning=True)
        write(ROOT / 'reference_provenance.json', provenance)
        report(selected, lock, decision, compute)
        write(ROOT / 'analysis.json', dict(utc=now(), decision=decision, actual_compute=compute, metric_availability=availability,
            selected={arm: {split: value['macro'] for split, value in record.items()} for arm, record in selected.items()},
            figures=paths, original_A_object_unchanged=True, legacy_B_and_R58='deferred/unfixed; not numeric70 veto'))
        receipt = dict(utc=now(), returncode=0, invocations=1, CPU_only=True, wall_seconds=time.monotonic() - started,
            comparison_rows=len(rows), figures=paths, inference=False, NPZ_replay=False, automatic_retry=False)
        write(ROOT / 'analysis_receipt.json', receipt)
        write(ROOT / 'delivery_receipt.json', dict(utc=now(), training=read(ROOT / 'training_terminal_receipt.json'),
            teacher_cache=str(ROOT / 'teacher_cache_complete.json'), new_B_TEST=terminal, analysis=receipt,
            lock_path=str(ROOT / 'final_test_lock.json'), lock=lock, decision=decision,
            new_tests_executed={'A': 0, 'B': 1}, original_A_tests_already_executed=1,
            historical_A_reuse=lock['control_reuse'], original_A_object_unchanged=True,
            parent_reviews_memory_goal_audit_Git_pending=True, worker_Git_actions=False,
            Git_alone_contains_all_inference_assets=False, R58_and_legacy_B='deferred/unfixed; not native70 veto'))
        print(json.dumps(dict(analysis=receipt, decision=decision)), flush=True)
    except BaseException as error:
        exclusive(ROOT / 'analysis_failure.json', dict(utc=now(), error_type=type(error).__name__, message=str(error),
            traceback=traceback.format_exc(), invocations=1, automatic_retry=False, marker_preserved=True))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--authorize-analysis', action='store_true')
    args = parser.parse_args()
    if not args.authorize_analysis:
        parser.error('Require parent conditional authorization after actual ONE B TEST terminal0')
    summarize()


if __name__ == '__main__':
    main()
