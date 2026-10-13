"""ONCE CPU-only R72 saved-JSON reporting; no NPZ replay or model execution."""

import json
import math
import os
import sys
import time
import traceback

from code.unified_selector.registry import PROBLEMS
from .r64_analysis import METRICS, write_csv
from .r70_analysis import metric_rows as saved_metric_rows
from .r72_experiment import ROOT, WORKSPACE, exclusive, read, write
from .r72_frozen_evaluation import completed_test, now


NAMES = {'A': 'R72A_reused_R60B39', 'B': 'R72B_fixed_consensus'}


def saved_counts(value):
    counts, origins = {}, {}
    for task, entry in value['per_problem'].items():
        count = entry.get('native_correct')
        if type(count) is int:
            origin = 'saved original native_correct field'
        else:
            product = entry['top1'] * entry['n']
            count = round(product)
            if not math.isclose(product, count, abs_tol=1e-8):
                raise ValueError('Saved task Top1*n does not identify an integer native count')
            origin = 'arithmetic from saved task Top1*n; no prediction recount'
        counts[task], origins[task] = count, origin
    return counts, origins


def metric_rows(model, split, value):
    rows = saved_metric_rows(model, split, None, value)
    counts, origins = saved_counts(value)
    for row in rows:
        tasks = list(PROBLEMS) if row['kind'] == 'aggregate' else (
            [task for task in PROBLEMS if task not in ('TSP', 'CVRP', 'ATSP')]
            if row['scope'] == 'MVRP' else [row['scope']])
        row['native_correct'] = sum(counts[task] for task in tasks)
        row['native_total'] = sum(value['per_problem'][task]['n'] for task in tasks)
        row['native_correct_origin'] = '; '.join(sorted({origins[task] for task in tasks}))
        row['CE_kind'] = 'Borda-score CE' if model == NAMES['B'] else 'historical native-winner CE'
        row['Borda_score_CE'] = row['ce'] if model == NAMES['B'] else None
        row['historical_native_CE'] = row['ce'] if model != NAMES['B'] else None
        row['measurement'] = 'one fixed consensus evaluation' if model == NAMES['B'] else 'unchanged historical saved metrics'
        del row['epoch']
    return rows


def historical_components(protocol):
    selected, provenance = {}, {}
    base = WORKSPACE / 'code/V4/runs'
    for name, experiment, run, epoch, global_epoch in (
        ('R67B36_global76', 'R67_supervised_contrastive_full18', 'B_contrastive_pretrain_seed2', 36, 76),
        ('R70B23_global63', 'R70_latent_mixup', 'B_latent_mixup_seed2', 23, 63)):
        root = base / experiment
        result = read(root / run / 'result.json')
        test, lock = read(root / 'test_results.json'), read(root / 'final_test_lock.json')
        record = lock['models']['B']
        if (not test['complete'] or test['lock'] != lock or (record['epoch'], record['global_epoch']) != (epoch, global_epoch)
                or record['sha256'] != protocol['members'][name]['original_record']['sha256']
                or test['models']['B']['evaluation_passes'] != 1):
            raise ValueError('Historical components retain their original locked official JSON only')
        selected[name] = dict(val=result['best']['val'], test=test['models']['B'])
        provenance[name] = dict(result_path=str(root / run / 'result.json'), validation_json_pointer='/best/val',
            test_path=str(root / 'test_results.json'), test_json_pointer='/models/B',
            lock_path=str(root / 'final_test_lock.json'), lock_json_pointer='/models/B',
            epoch=epoch, global_epoch=global_epoch, descriptive_only=True,
            reference_inference=False, prediction_recount=False, component_metrics_recalculated=False)
    return selected, provenance


def comparison_figures(plt, selected):
    import numpy as np
    curves = ROOT / 'curves/delivery'
    curves.mkdir(parents=True)
    for kind, scopes, filename, size in (
        ('task', list(PROBLEMS), 'task_top1.png', (14, 9)),
        ('family', ['TSP', 'CVRP', 'ATSP', 'MVRP'], 'family_top1.png', (10, 7))):
        figure, axes = plt.subplots(2, 1, figsize=size)
        positions = np.arange(len(scopes))
        for ax, split in zip(axes, ('val', 'test')):
            for offset, arm, color in ((-.18, 'A', '#63717a'), (.18, 'B', '#16846c')):
                value = selected[NAMES[arm]][split]
                points = [value['per_problem'][scope]['top1'] if kind == 'task'
                          else value['families'][scope]['macro_top1'] for scope in scopes]
                ax.bar(positions + offset, np.array(points)*100, width=.36, color=color,
                       label='Fixed A: R60B39' if arm == 'A' else 'B: fixed three-member consensus')
            ax.set_xticks(positions, scopes, rotation=45 if kind == 'task' else 0, ha='right' if kind == 'task' else 'center')
            ax.set_ylabel('Native Top1 (%)')
            ax.set_title(split.upper() + ': original full legal pools')
            ax.grid(axis='y', alpha=.2)
            ax.set_axisbelow(True)
            ax.legend(fontsize=8)
        figure.tight_layout()
        figure.savefig(curves / filename, dpi=150)
        plt.close(figure)
    return [str(curves / filename) for filename in ('task_top1.png', 'family_top1.png')]


def report(rows, decision, compute, test):
    lines = ['# R72 Fixed Heterogeneous Consensus', '',
        f"Official arm B was locked before TEST from8923/18000 VAL, versus fixed R60B39 A's8885: +38 decisions (+0.211111pp).",
        f"ONE new original full18 B TEST: **{decision['official_test_correct']}/18000 = {100*decision['official_test_top1']:.6f}%**; "
        f"versus unchanged historical A: {decision['A_original_test_correct']}/18000, {decision['B_minus_A_test_decisions']:+d} decisions "
        f"({decision['B_minus_A_test_delta_pp']:+.6f}pp). TEST never changes membership, checkpoint, aggregation or official arm.", '',
        '| Policy | Split | Native Correct | Top1% | Top2% | Top3% | Mean Cost | vsSBS% | Actual Regret% | Borda-score CE | Historical Native CE |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for row in rows:
        if row['kind'] != 'aggregate':
            continue
        borda = f"{row['Borda_score_CE']:.6f}" if row['Borda_score_CE'] is not None else 'not applicable'
        native = f"{row['historical_native_CE']:.6f}" if row['historical_native_CE'] is not None else 'not applicable'
        lines.append(f"| {row['model']} | {row['split'].upper()} | {row['native_correct']}/{row['native_total']} | "
            f"{100*row['top1']:.6f} | {100*row['top2']:.6f} | {100*row['top3']:.6f} | {row['mean_cost']:.6f} | "
            f"{row['vs_sbs_pct']:.6f} | {row['actual_regret_pct']:.6f} | {borda} | {native} |")
    history = compute['historical_accounting']
    r60, r67, r70 = [history['historical_members'][name] for name in ('R60B39', 'R67B36_global76', 'R70B23_global63')]
    stage1 = history['inherited_R67B_stage1_once']
    expanded = sum(r60['extra_rows_by_problem'].values())
    lines += ['',
        'comparison.csv includes equal-weight full18 macros, every task and TSP/CVRP/ATSP/MVRP families for VAL and TEST. '
        'Historical component rows are saved descriptive records, not separately executed member metrics. '
        'R60 JSON has no native_correct fields; its counts in reporting tables are arithmetic from saved task Top1*n, '
        'not prediction recounts. Its original TEST object remains exactly unchanged.', '',
        'The complete offline composite contains R60B39 original cost-selected score-difference/maximin weights, '
        'R67B36/global76 saved averaged-x weights without projector/optimizer transition, and R70B23/global63 all18 '
        'complete common-epoch branches. No historical checkpoint pointers are followed during inference. '
        'All36859060 stored parameters are frozen;5528859 parameters activate per query across three selectors.', '',
        'Every query uses exactly three original returned FP32 logits, aligned over the complete legal pool by global ID. '
        'Doubled midrank q=2*strictly_higher+equal_including_self+1; integer Q sums all three. Minimum Q wins, with smallest '
        'global ID resolving final ties. Reporting-only z=-Q/6 defines Borda-score CE. Historical native CE is on a different '
        'score scale, so its difference is not a calibration gain. Density NLL is inapplicable. Targets never enter '
        'member inputs or aggregation. TRAIN-fitted normalization, geometry, ATSP direction and physical roles remain intact.', '',
        f"New R72 expenditure: zero TRAIN epochs/updates/TRAIN forward rows, zero optimized parameters, zero solver executions. "
        f"VAL wall {compute['validation_seconds']:.6f}s; ONE full18 TEST wall {compute['test_evaluation_seconds']:.6f}s "
        f"({compute['test_forward_and_aggregation_seconds']:.6f}s in three live forwards plus aggregation). "
        f"TEST setup plus evaluation work was {compute['test_setup_and_evaluation_seconds']:.6f}s. "
        'These are ordinary whole-evaluation times, not singleton or completed-solver latency; no separate latency benchmark ran.', '',
        'Each VAL/TEST pass covers18000 original rows,144 query batches,432 member forwards,54000 member-row forwards and144 '
        'active R70 branch calls, including eighteen104-row tails. No standalone A or component evaluation occurred. '
        'Deployment executes three selectors and completes one selected solver.', '',
        f"TEST peak allocated/reserved/observed physical memory: {compute['peak_allocated_bytes']/2**30:.6f}/"
        f"{compute['peak_reserved_bytes']/2**30:.6f}/{compute['peak_observed_physical_used_bytes']/2**30:.6f}GiB. "
        'One offline composite was restored; three historical constituent checkpoint files were not separately reloaded.', '',
        f"Historical R60 expansion: {r60['successful_updates']} updates/{r60['epochs_completed']} epochs/"
        f"{r60['wall_seconds']:.6f}s, with{expanded} extra historical rows across15 MVRP tasks. "
        f"R67 historical total: {r67['successful_updates']} updates/{r67['presentations']} rows/{r67['wall_seconds']:.6f}s, "
        f"already including its stage1. R70 historical fine-tuning: {r70['successful_updates']} updates/"
        f"{r70['presentations']} rows/{r70['wall_seconds']:.6f}s, five retries. Shared R67B stage1 is counted ONCE: "
        f"{stage1['successful_updates']} updates/{stage1['presentations']} rows/{stage1['wall_seconds']:.6f}s. "
        'Do not add that shared stage1 a second time. These are inherited historical costs, not new R72 training.', '',
        'There is one fixed candidate and no R72 TRAIN curve, loss, epoch, best/final search, calibration, membership search '
        'or taskwise assembly. Three-selector inference and stored capacity are unmatched to A. R66 remains weak ensemble '
        'counterevidence, and shared R67/R70 pretraining does not establish complementary instance errors. A small aggregate '
        'change supports only this particular fixed bundle, not error independence or a70% forecast.', '',
        'A reuses the EXACT original R60B39 TEST JSON traced through R64 official metadata to R63 reference/R60B/test_eval.json; '
        'new TEST counts A0/B1, original A already1. Historical TEST exposure is disclosed. The unique new TEST neither changes '
        'this fixed protocol nor goes to successor planning. Analysis ran ONCE on saved JSON only; no NPZ replay, model/solver '
        'forward, reference inference or additional experiment.', '',
        f"Original goal threshold:12600/18000 native TEST decisions. Observed official B:{decision['official_test_correct']}/18000. "
        'Goal70 remains ACTIVE/unachieved; parent actual reviews and whole-objective audit are pending.', '',
        'R58 and eight legacy B semantic inconsistencies remain deferred/disclosed/unfixed, not a native70 numeric veto. '
        'Weights, NPZ/raw scores, batch logs, W&B and caches remain LOCAL; Git alone is not all inference assets. '
        'Parent owns actual Astra reviews, persistent memory, objective audit and one completed-version Git publication. '
        'This worker performs no Git.', '']
    text = '\n'.join(lines)
    (ROOT / 'RESULTS.md').write_text(text)
    (ROOT / 'REPORT.md').write_text(text)


def summarize():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('The ONCE analysis process must be CPU-only')
    exclusive('analysis_started.json', dict(started_utc=now(), invocations=1, cpu_only=True,
        command=' '.join([sys.executable]+sys.argv), inference=False, NPZ_replay=False, automatic_retry=False))
    started = time.monotonic()
    try:
        protocol, lock, test, terminal = completed_test()
        original_A_val = read(WORKSPACE / 'code/V4/runs/R60_expanded_partial_supervision/B_expanded_seed2/result.json')['best']['val']
        selected = {NAMES['A']: dict(val=original_A_val, test=test['models']['A']),
                    NAMES['B']: dict(val=read(ROOT / 'validation_results.json'), test=test['models']['B'])}
        historical, provenance = historical_components(protocol)
        selected.update(historical)
        provenance[NAMES['A']] = lock['control_reuse']
        provenance[NAMES['B']] = dict(validation_path=str(ROOT / 'validation_results.json'),
            test_path=str(ROOT / 'test_results.json'), test_json_pointer='/models/B',
            lock_path=str(ROOT / 'final_test_lock.json'), bundle=lock['models']['B'], one_new_live_TEST=True)
        rows = [row for name, record in selected.items() for split in ('val', 'test')
                for row in metric_rows(name, split, record[split])]
        write_csv(ROOT / 'comparison.csv', rows)
        indexed = {(row['model'], row['split'], row['kind'], row['scope']): row for row in rows}
        deltas = []
        for split in ('val', 'test'):
            for row in rows:
                if row['model'] != NAMES['B'] or row['split'] != split:
                    continue
                control = indexed[(NAMES['A'], split, row['kind'], row['scope'])]
                deltas.append(dict(split=split, kind=row['kind'], scope=row['scope'],
                    native_correct_B_minus_A=row['native_correct']-control['native_correct'],
                    **{metric+('_delta_pp' if metric.startswith('top') else '_delta'):
                       (row[metric]-control[metric])*(100 if metric.startswith('top') else 1)
                       for metric in METRICS if metric != 'ce'},
                    CE_difference_not_interpreted=True))
        write_csv(ROOT / 'comparison_deltas.csv', deltas)
        A_count = indexed[(NAMES['A'], 'test', 'aggregate', 'ALL')]['native_correct']
        B_count = test['models']['B']['native_correct']
        decision = dict(official_arm='B', official_test_correct=B_count, official_test_total=18000,
            official_test_top1=B_count/18000, A_original_test_correct=A_count,
            A_count_source='saved task Top1*n arithmetic; original object unchanged; no prediction recount',
            B_minus_A_test_decisions=B_count-A_count, B_minus_A_test_delta_pp=100*(B_count-A_count)/18000,
            validation_count_B_minus_A=38, selection_frozen_before_TEST=True, TEST_reselection=False,
            original_native_70pct_threshold_correct=12600, numeric_threshold_reached=B_count >= 12600,
            whole_objective_audit_passed=False, claim_70pct_supported=False, goal_status='ACTIVE/unachieved',
            parent_reviews_pending=True, historical_test_exposure=True, NEW_R72_TEST_used_for_planning=False)
        value = test['models']['B']
        compute = dict(new_TRAIN_epochs=0, new_optimizer_updates=0, new_TRAIN_forward_rows=0, trainable_parameters=0,
            stored_parameters=36859060, active_parameters_per_query=5528859, selectors_per_query=3,
            completed_deployment_solvers_per_query=1, solver_executions=0, parameter_and_compute_matched=False,
            validation_seconds=selected[NAMES['B']]['val']['seconds'], test_evaluation_seconds=value['seconds'],
            test_forward_and_aggregation_seconds=value['forward_and_aggregation_seconds'],
            test_setup_and_evaluation_seconds=test['wall_seconds'], test_counters=value['counters'],
            peak_allocated_bytes=value['peak_allocated_bytes'], peak_reserved_bytes=value['peak_reserved_bytes'],
            peak_observed_physical_used_bytes=value['peak_observed_physical_used_bytes'],
            historical_accounting=protocol['historical_accounting'], original_cpu_check=read(ROOT / 'cpu_check_receipt.json'),
            validation_terminal=read(ROOT / 'validation_terminal_receipt.json'), test_terminal=terminal)
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        plt.rcParams.update({'font.size': 9})
        figures = comparison_figures(plt, selected)
        write('reference_provenance.json', dict(recorded_utc=now(), references=provenance,
            historical_TEST_exposure=True, standalone_reference_inference=False,
            separate_component_evaluation=False, NPZ_replay=False, original_A_object_unchanged=True))
        write('compute_summary.json', compute)
        report(rows, decision, compute, test)
        write('analysis.json', dict(recorded_utc=now(), decision=decision, actual_compute=compute,
            saved_metrics_only=True, selected={name: {split: record[split]['macro'] for split in ('val', 'test')}
                                              for name, record in selected.items()},
            new_tests_executed={'A': 0, 'B': 1}, original_A_tests_already_executed=1,
            new_training=False, R72_TRAIN_measurements=None, figures=figures))
        receipt = dict(recorded_utc=now(), returncode=0, cpu_only=True, invocations=1,
            wall_seconds=time.monotonic()-started, comparison_rows=len(rows), figures=figures, png_count=len(figures),
            JSON_only=True, NPZ_replay=False, model_forwards=0, solver_executions=0, original_A_object_unchanged=True)
        exclusive('analysis_receipt.json', receipt)
        print(json.dumps(dict(analysis=receipt, decision=decision)), flush=True)
    except BaseException as error:
        write('analysis_failure.json', dict(recorded_utc=now(), error_type=type(error).__name__, message=str(error),
            traceback=traceback.format_exc(), original_marker_preserved=True, automatic_retry=False))
        raise


if __name__ == '__main__':
    summarize()
