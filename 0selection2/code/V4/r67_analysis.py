"""One future CPU delivery pass over completed R67 and fixed saved references."""

import argparse
import json
import os
import time
import traceback
from pathlib import Path

import numpy as np

from code.unified_selector.registry import PROBLEMS
from .multitask_probe import file_hash
from .r59_experiment import atomic_json
from .r64_analysis import METRICS, write_csv
from .r65_analysis import group_panels, metrics, now, read
from .r66_analysis import metric_rows, reference_records
from .r67_experiment import ROOT, RUN_NAMES
from .r67_frozen_evaluation import require_terminal, validate_lock


def objective_name(arm, stage):
    return 'scaled supervised contrastive Eq2 (not CE)' if arm == 'B' and stage == 1 else 'native CE plus legal RDrop KL'


def fixed_references(root):
    selected, provenance = reference_records(root)
    base = root.parent/'R66_additive_correction_full18'
    test_path, lock_path = base/'test_results.json', base/'final_test_lock.json'
    test, lock = read(test_path), read(lock_path)
    if not test['complete'] or test['lock'] != lock:
        raise ValueError('R66 references must be their previously completed official locked results')
    for arm, name in {'A': 'A_independent_sum_seed2', 'B': 'B_additive_correction_seed2'}.items():
        path = base/name/'result.json'
        result = read(path)
        model = 'R66'+arm
        selected[model] = dict(epoch=lock['models'][arm]['epoch'], train=result['best']['train'],
                               val=result['best']['val'], test=test['models'][arm])
        provenance[model] = dict(result_path=str(path), result_sha256=file_hash(path),
            train_json_pointer='/best/train', validation_json_pointer='/best/val',
            test_path=str(test_path), test_sha256=file_hash(test_path), test_json_pointer='/models/'+arm,
            lock_path=str(lock_path), lock_sha256=file_hash(lock_path), locked_official_arm=lock['official_arm'],
            checkpoint_policy=result['checkpoint_rule'], parameter_mode='saved averaged x',
            deployment='historical two-selector F+g; preserved, not reproduced by R67', historical_test_exposure=True)
    if selected['R45A']['epoch'] != 38 or selected['R60B']['epoch'] != 39:
        raise ValueError('Keep fixed historical official R45A38/R60B39; no retrospective selection')
    return selected, provenance


def normalize_counts(selected):
    for record in selected.values():
        for split in ('train', 'val', 'test'):
            if record.get(split) is None:
                continue
            value = record[split]
            for entry in value['per_problem'].values():
                if entry.get('native_correct') is None:
                    entry['native_correct'] = int(round(entry['top1']*entry['n']))
            if value.get('native_total') is None:
                value['native_total'] = sum(p['n'] for p in value['per_problem'].values())
            if value.get('native_correct') is None:
                value['native_correct'] = sum(p['native_correct'] for p in value['per_problem'].values())


def read_updates(path):
    with path.open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def actual_compute(root, results, updates, test, terminals):
    compute = dict(planned=dict(per_stage_updates=9600, per_stage_presentations=7200000,
        per_arm_updates=19200, per_arm_presentations=14400000, pair_updates=38400, pair_presentations=28800000),
        arms={}, training_terminal=terminals['train'], test_terminal=terminals['test'],
        inherited_pretrained_selectors=0, pilots=0, sizing_sweeps=0, selector_forwards_per_deployment=1,
        completed_solvers_per_deployment=1, solver_executions_during_delivery=0,
        timing_scope='ordinary complete eval wall time; no singleton latency benchmark',
        test_wall_seconds=test['wall_seconds'], actual_logged_counts_authoritative=True)
    for arm, result in results.items():
        rows = updates[arm]
        stages = {}
        for stage in (1, 2):
            items = [r for r in rows if r['stage'] == stage]
            stage_result = read(root/RUN_NAMES[arm]/f'stage{stage}_result.json')
            stages[str(stage)] = dict(successful_updates=len(items), presentations=sum(r['rows'] for r in items),
                amp_retries=sum(r['retries'] for r in items), logged_update_seconds=sum(r['seconds'] for r in items),
                stage_wall_seconds=stage_result['wall_seconds'], objective=objective_name(arm, stage),
                planned_budget_matches=len(items) == 9600 and sum(r['rows'] for r in items) == 7200000,
                finite_updates=all(r['finite_loss'] and r['finite_gradient'] for r in items),
                last_successful_stage_counter=items[-1]['stage_successful_update'] if items else 0,
                last_stage_presentations=items[-1]['stage_presentations'] if items else 0)
        compute['arms'][arm] = dict(stages=stages, successful_updates=len(rows),
            presentations=sum(r['rows'] for r in rows), amp_retries=sum(r['retries'] for r in rows),
            result_recorded_successful_updates=result['successful_updates'],
            result_recorded_presentations=result['presentations'], epochs_completed=result['epochs_completed'],
            wall_seconds=result['wall_seconds'], logged_update_seconds=sum(r['seconds'] for r in rows),
            peak_allocated_gib=max((r['peak_memory_gib'] for r in rows), default=0.),
            peak_reserved_gib=max((r['peak_reserved_gib'] for r in rows), default=0.),
            whole_device_peak_physical_gib=max((r['physical_used_gib'] for r in rows), default=0.),
            ordinary_test_evaluation_seconds=test['models'][arm]['complete_evaluation_seconds'],
            stage_transition=read(root/RUN_NAMES[arm]/'stage_transition.json'))
    compute['actual_pair_successful_updates'] = sum(v['successful_updates'] for v in compute['arms'].values())
    compute['actual_pair_presentations'] = sum(v['presentations'] for v in compute['arms'].values())
    compute['pair_arm_wall_seconds'] = sum(v['wall_seconds'] for v in compute['arms'].values())
    return compute


def stage2_summaries(root, results, histories):
    summaries, last_rows = [], []
    for arm, history in histories.items():
        for label, record in (('best', results[arm]['best']), ('final', results[arm]['final'])):
            for split in ('train', 'val'):
                if record.get(split) is not None:
                    summaries += [dict(arm=arm, summary=label, **r)
                        for r in metric_rows('R67'+arm, split, record['epoch'], record[split])]
        grouped = {}
        for record in [r for r in history if r['stage'] == 2][-5:]:
            for row in metric_rows('R67'+arm, 'val', record['epoch'], record['val']):
                last_rows.append(dict(arm=arm, **row))
                grouped.setdefault((row['kind'], row['scope']), []).append(row)
        for (kind, scope), rows in grouped.items():
            for label, function in (('last5_mean', np.mean), ('last5_population_sd', np.std),
                                    ('last5_min', np.min), ('last5_max', np.max)):
                summaries.append(dict(arm=arm, model='R67'+arm, split='val', kind=kind, scope=scope, summary=label,
                    **{key: float(function([r[key] for r in rows])) for key in (*METRICS, 'native_correct', 'native_total')}))
    write_csv(root/'stage2_summary.csv', summaries)
    write_csv(root/'last5_validation.csv', last_rows)


def learning_figures(plt, curves, histories, updates):
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    for ax, metric in zip(axes.flat, METRICS):
        for arm, history in histories.items():
            for split, style in (('train', '--'), ('val', '-')):
                rows = [r for r in history if r['stage'] == 2 and r.get(split) is not None]
                ax.plot([r['epoch'] for r in rows], [r[split]['macro']['macro_'+metric] for r in rows],
                        linestyle=style, label=arm+' '+split+' averaged x')
        ax.set(title=metric, xlabel='Stage2 epoch')
        ax.grid(alpha=.2)
        ax.legend(fontsize=7)
    axes.flat[-1].axis('off')
    fig.tight_layout()
    fig.savefig(curves/'stage2_learning_curves.png', dpi=150)
    plt.close(fig)
    for stage in (1, 2):
        for resolution in ('epoch', 'update'):
            source = {arm: ([dict(r['optimization'], stage_epoch=r['epoch']) for r in history if r['stage'] == stage]
                            if resolution == 'epoch' else [r for r in updates[arm] if r['stage'] == stage])
                      for arm, history in histories.items()}
            xkey = 'stage_epoch' if resolution == 'epoch' else 'stage_successful_update'
            fig, axes = plt.subplots(2, 4, figsize=(18, 8))
            for arm, ax in zip(('A', 'B'), axes.flat):
                for metric in (('loss', 'supcon_raw') if stage == 1 and arm == 'B' else ('loss', 'ce', 'kl')):
                    ax.plot([r[xkey] for r in source[arm]], [r[metric] for r in source[arm]], label=metric, linewidth=.8)
                ax.set(title=arm+' '+objective_name(arm, stage), xlabel=xkey)
                ax.legend(fontsize=7)
            common = ('lr', 'grad_norm', 'clip_coefficient', 'amp_retries', 'peak_memory_gib', 'seconds')
            for ax, metric in zip(list(axes.flat)[2:], common):
                for arm, rows in source.items():
                    key = 'retries' if resolution == 'update' and metric == 'amp_retries' else metric
                    ax.plot([r[xkey] for r in rows], [r[key] for r in rows], label=arm, linewidth=.8)
                ax.set(title=metric, xlabel=xkey)
                ax.legend(fontsize=7)
            for ax in axes.flat:
                ax.grid(alpha=.2)
            fig.tight_layout()
            fig.savefig(curves/f'stage{stage}_{resolution}_objectives_compute.png', dpi=150)
            plt.close(fig)


def selected_figures(plt, curves, selected):
    methods, diagnostics = [], []
    for model in ('R67A', 'R67B'):
        for split in ('train', 'val', 'test'):
            for task, value in selected[model][split]['per_problem'].items():
                directory = curves/model/split/task
                directory.mkdir(parents=True, exist_ok=True)
                pool = value['pool']
                diagnostics.append(dict(model=model, split=split, problem=task,
                    **{k: value[k] for k in ('best_top1', 'best_top2', 'best_top3', 'sbs_name', 'sbs_cost', 'oracle_cost')},
                    **{k: json.dumps(value[k]) for k in ('top1_exceeds_methods', 'top1_not_exceeds_methods',
                                                         'cost_exceeds_methods', 'cost_not_exceeds_methods')}))
                for index, method in enumerate(pool):
                    methods.append(dict(model=model, split=split, problem=task, method=method,
                        method_top1=value['method_top1'][method], method_mean_cost=value['method_mean_cost'][method],
                        selector_top1=value['top1'], selector_mean_cost=value['mean_cost'], pick_fraction=value['pick_dist'][index]))
                for filename, values, selector, label in (
                    ('top1_vs_single_methods.png', [value['method_top1'][m] for m in pool], value['top1'], 'Native Top1'),
                    ('mean_cost_vs_single_methods.png', [value['method_mean_cost'][m] for m in pool], value['mean_cost'], 'Mean cost'),
                    ('arm_distribution.png', value['pick_dist'], None, 'Selection fraction')):
                    fig, ax = plt.subplots(figsize=(max(6, .6*len(pool)), 4))
                    ax.bar(np.arange(len(pool)), values, color='#227f79')
                    if selector is not None:
                        ax.axhline(selector, color='#b84332', label='single selector averaged x')
                        ax.legend(fontsize=8)
                    ax.set_xticks(np.arange(len(pool)), pool, rotation=55, ha='right')
                    ax.set(title=model+' '+split+' '+task, ylabel=label)
                    fig.tight_layout()
                    fig.savefig(directory/filename, dpi=150)
                    plt.close(fig)
    return methods, diagnostics


def report_markdown(root, selected, lock, decision, compute):
    official = lock['official_arm']
    lines = ['# R67 Locked Delivery', '',
        f"Official R67{official}, stage2 epoch{lock['models'][official]['epoch']}, globalepoch{lock['models'][official]['global_epoch']}: locked original18 TEST {decision['official_test_correct']}/18000 = {100*decision['official_test_top1']:.6f}% native Top1.", '',
        'The official arm and checkpoint were frozen using maximum integer validation correct count, earliest exacttie and arm tieA. Costs, TEST and parameter mode do not select models. There is no task-wise assembly. TEST loads one complete saved stage2 averaged-x NativeUtilitySelector directly; no optimizer transition, projector or inherited trained selector.', '',
        'Stage1 A uses native CE/RDrop; stage1 B uses only task-local supervised contrastive Eq.2. These objective values are different quantities. There is no stage1 validation curve. Both stage2 objectives are native CE/RDrop and their complete validation curves are comparable. Dropout-only views and fully trainable stage2 are project adaptations. Fixed contrastive multiplier.1 is a project loss scale, not the author-default temperature/base_temperature scale.', '',
        '## Selected Metrics', '',
        '| Model | Split | Top1 % | Top2 % | Top3 % | CE | Mean Cost | vsSBS % | Actual Regret % |',
        '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for model, record in selected.items():
        for split in ('train', 'val', 'test'):
            if record.get(split) is None:
                continue
            m = metrics(record[split]['macro'], 'macro_')
            lines.append(f"| {model} | {split} | {100*m['top1']:.4f} | {100*m['top2']:.4f} | {100*m['top3']:.4f} | {m['ce']:.6f} | {m['mean_cost']:.6f} | {m['vs_sbs_pct']:.6f} | {m['actual_regret_pct']:.6f} |")
    lines += ['',
        'comparison.csv contains every original task, four families and ALL for available TRAIN/VAL/TEST. ALL is the equal18-task macro; MVRP is15 tasks. Original native winners and ascending-global-ID exact utility ties are preserved. Costs retain FP64. vsSBS is100*(selector mean cost/SBS cost-1); actual regret is mean per-query relative excess over the legal-pool oracle. Mean costs across heterogeneous tasks are descriptive across task units.', '',
        'R45A38/R60B39 retain their historical cost-selected official policies. R64A/B, R65A/B and R66A/B retain original validation locks and official-arm policies. Fixed saved JSON reports are reused; no reference inference, prediction replay, retrospective TEST or solver is run. Historical TEST is exposed, so version comparisons are descriptive. Missing saved historical TRAIN, notably R45A, is disclosed in metric_availability.csv and never reconstructed by inference.', '',
        'stage2_summary.csv contains best/final TRAIN/VAL and last5 validation mean/population SD/min/max by task/family/ALL. Last5 are complete validation epochs36-40, not seed uncertainty. TRAIN exists only on the predeclared every5/final/selected epochs; no dense TRAIN or stage1 validation curve is invented.', '',
        '## Actual Compute', '',
        '| Arm | Stage | Successful Updates | Presentations | AMP Retries | Stage Wall s |',
        '| --- | ---: | ---: | ---: | ---: | ---: |']
    for arm, entry in compute['arms'].items():
        for stage, values in entry['stages'].items():
            lines.append(f"| {arm} | {stage} | {values['successful_updates']} | {values['presentations']} | {values['amp_retries']} | {values['stage_wall_seconds']:.3f} |")
    lines += ['',
        f"Actual pair: {compute['actual_pair_successful_updates']} updates/{compute['actual_pair_presentations']} presentations, {compute['pair_arm_wall_seconds']:.3f}s summed arm wall. Planned eachstage9600/7.2M, eacharm19200/14.4M, pair38400/28.8M. Actual logs and terminal receipts are authoritative; planned counts are never substituted for interrupted execution. There are zero inherited pretrained selectors.", '',
        'Real256/task, three distinct tasks/update, effective768; tail16/task/effective48. Both arms run40+40epochs. Official ScheduleFree/internal720-update stage warmup,3e-4 constant thereafter, native legal pools, FP32 mean task gradients/one clip1 and whole-triple AMP retries are unchanged. Stage transition receipts record terminal x, identical whole fresh head and fresh optimizer/scaler/counters. A projector and B score head are intentionally unused during stage1.', '',
        'One selector forward per query and one completed solver at deployment; evaluation itself executes zero solvers. Reported TEST time is ordinary full-evaluation wall time including gather/transfer, metrics and prediction saving, not singleton latency. compute_summary.json retains measured steps, runtime, memory, counters and actual Slurm/tmux terminal receipts.', '',
        f"Numeric locked official TEST threshold reached: {decision['numeric_threshold_reached']}. Whole-objective completion audit passed: {decision['whole_objective_audit_passed']}. A70% claim is permitted only when BOTH are true. Goal remains ACTIVE until parent closure; no severe-overfit, label-noise or accuracy-ceiling claim follows from this one-seed experiment.", '',
        '## Assets and Limits', '',
        'Selected checkpoints, complete prediction NPZ/raw costs/IDs, raw update logs, W&B and regenerable schedules stay local under current cleanup policy. Code, reports, configs, receipts, CSV curves, figures and review records can be curated into one later completed-version commit by parent. Nothing is force-added. Git alone does not contain all assets needed for inference.', '',
        'The full18 native70% metric remains the user objective. R58 is fully deferred. The eight legacy B-task feasible-domain mismatches remain unresolved and disclosed; they are not quietly repaired or used to veto that native metric. Parent owns the final objective audit, reviews, memory and Git.', '']
    (root/'REPORT.md').write_text('\n'.join(lines))


def summarize(root):
    if root.resolve() != ROOT.resolve() or os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('Use the canonical root and CUDA_VISIBLE_DEVICES= for this CPU-only delivery pass')
    protocol, lock = validate_lock(root)
    test, marker = read(root/'test_results.json'), read(root/'final_test_started.json')
    terminals = dict(train=read(root/'training_terminal_receipt.json'), test=read(root/'test_terminal_receipt.json'))
    require_terminal(terminals['test'], 'r67_final_test', marker['slurm_step'])
    process = read(root/'test_process_receipt.json')
    if (not test.get('complete') or test['lock'] != lock or marker['lock'] != lock
            or test['tests_per_frozen_model'] != 1 or marker['allowed_attempts'] != 1
            or process['returncode'] != 0 or process['slurm_step'] != marker['slurm_step']
            or terminals['test'].get('original_full18_fulllegalpool_tests_per_frozen_model') != 1
            or read(root/'test_launch_receipt.json')['slurm_step'] != marker['slurm_step']):
        raise ValueError('Analysis requires actual terminal0 and the one complete locked TEST per selector')
    for arm in ('A', 'B'):
        value = test['models'][arm]
        if (not value['complete'] or value['evaluation_passes'] != 1 or value['native_total'] != 18000
                or set(value['per_problem']) != set(PROBLEMS) or value['selector_forwards_per_query'] != 1
                or value['evaluated_checkpoint_sha256'] != lock['models'][arm]['inference_sha256']
                or value['cached_test_scores'] is not False or value['optimizer_mode_transition'] is not False):
            raise ValueError('Require complete original18 single-forward saved-x TEST identities')
    with (root/'analysis_started.json').open('x') as stream:
        json.dump(dict(started_utc=now(), cpu_only=True, one_pass=True, module_sha256=file_hash(Path(__file__)),
                       test_terminal_receipt_sha256=file_hash(root/'test_terminal_receipt.json')), stream, indent=2)
        stream.write('\n')
    started = time.monotonic()
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        plt.rcParams.update({'font.size': 9})
        curves = root/'curves/delivery'
        curves.mkdir(parents=True, exist_ok=True)
        results = {arm: read(root/name/'result.json') for arm, name in RUN_NAMES.items()}
        histories = {arm: read(root/name/'history.json') for arm, name in RUN_NAMES.items()}
        updates = {arm: read_updates(root/name/'updates.jsonl') for arm, name in RUN_NAMES.items()}
        compute = actual_compute(root, results, updates, test, terminals)
        atomic_json(root/'compute_summary.json', compute)
        selected = {'R67'+arm: dict(epoch=lock['models'][arm]['epoch'], train=result['best']['train'],
                    val=result['best']['val'], test=test['models'][arm]) for arm, result in results.items()}
        evaluations, task_rows, family_rows, optimization, task_optimization, update_rows = [], [], [], [], [], []
        for arm, history in histories.items():
            for row in history:
                identity = dict(arm=arm, stage=row['stage'], stage_epoch=row['epoch'], global_epoch=row['global_epoch'])
                optimization.append(dict(identity, objective=objective_name(arm, row['stage']),
                    stage_successful_updates=row['stage_successful_updates'], global_successful_updates=row['global_successful_updates'],
                    stage_presentations=row['stage_presentations'], global_presentations=row['global_presentations'], **row['optimization']))
                task_optimization += [dict(identity, problem=task, objective=objective_name(arm, row['stage']), **entry)
                                      for task, entry in row['task_optimization'].items()]
                for split in ('train', 'val'):
                    value = row.get(split)
                    if value is None:
                        continue
                    if row['stage'] != 2:
                        raise ValueError('No stage1 validation or utility evaluation may be invented')
                    evaluations += [dict(identity, **entry)
                        for entry in metric_rows('R67'+arm, split, row['epoch'], value)]
                    task_rows += [dict(arm=arm, epoch=row['epoch'], split=split, problem=p, **metrics(m))
                                  for p, m in value['per_problem'].items()]
                    family_rows += [dict(arm=arm, epoch=row['epoch'], split=split, family=p, **metrics(m, 'macro_'))
                                    for p, m in value['families'].items()]
            for row in updates[arm]:
                entry = {key: value for key, value in row.items() if isinstance(value, (int, float, bool, str))}
                update_rows.append(dict(entry, objective=objective_name(arm, row['stage'])))
        for name, rows in (('learning_curves', evaluations), ('task_curves', task_rows), ('family_curves', family_rows),
                           ('optimization_curves', optimization), ('task_optimization', task_optimization), ('update_curves', update_rows)):
            write_csv(root/(name+'.csv'), rows)
        learning_figures(plt, curves, histories, updates)
        group_panels(plt, curves, task_rows, 'problem')
        group_panels(plt, curves, family_rows, 'family')
        methods, diagnostics = selected_figures(plt, curves, selected)
        write_csv(root/'method_metrics.csv', methods)
        write_csv(root/'task_diagnostics.csv', diagnostics)
        historical, provenance = fixed_references(root)
        selected.update(historical)
        normalize_counts(selected)
        comparison = [row for model, record in selected.items() for split in ('train', 'val', 'test')
            if record.get(split) is not None for row in metric_rows(model, split, record['epoch'], record[split])]
        write_csv(root/'comparison.csv', comparison)
        write_csv(root/'selected_metrics.csv', [r for r in comparison if r['model'].startswith('R67')])
        write_csv(root/'metric_availability.csv', [dict(model=model, split=split, available=record.get(split) is not None,
            policy='saved original selected metrics only; no inference/replay to fill missing historical TRAIN')
            for model, record in selected.items() for split in ('train', 'val', 'test')])
        indexed = {(r['model'], r['split'], r['kind'], r['scope']): r for r in comparison}
        pairs = [('R67B', 'R67A')]+[(model, ref) for model in ('R67A', 'R67B') for ref in historical]
        deltas = []
        for left, right in pairs:
            for split in ('train', 'val', 'test'):
                for kind, scope in [('aggregate', 'ALL')]+[('family', p) for p in ('TSP', 'CVRP', 'ATSP', 'MVRP')]+[('task', p) for p in PROBLEMS]:
                    if (left, split, kind, scope) not in indexed or (right, split, kind, scope) not in indexed:
                        continue
                    a, b = indexed[left, split, kind, scope], indexed[right, split, kind, scope]
                    deltas.append(dict(comparison=left+' minus '+right, split=split, kind=kind, scope=scope,
                        **{key+('_delta_pp' if key.startswith('top') else '_delta'):
                           (a[key]-b[key])*(100 if key.startswith('top') else 1) for key in METRICS}))
        write_csv(root/'comparison_deltas.csv', deltas)
        stage2_summaries(root, results, histories)
        official = lock['official_arm']
        correct = test['models'][official]['native_correct']
        audit_path = root/'objective_audit.json'
        audit = read(audit_path) if audit_path.exists() else {}
        audit_passed = (audit.get('passed') is True and audit.get('scope') == 'whole objective'
            and audit.get('final_test_lock_sha256') == file_hash(root/'final_test_lock.json')
            and audit.get('test_results_sha256') == file_hash(root/'test_results.json'))
        budget_consistent = all(stage['planned_budget_matches'] and stage['finite_updates']
            for arm in compute['arms'].values() for stage in arm['stages'].values())
        decision = dict(official_arm=official, decision_source='original integer validation only; earliest exacttie/armtie A',
            validation_count_B_minus_A=lock['models']['B']['native_correct']-lock['models']['A']['native_correct'],
            contrastive_exceeds_matched_control_validation=lock['models']['B']['native_correct'] > lock['models']['A']['native_correct'],
            official_test_correct=correct, official_test_total=18000, official_test_top1=correct/18000,
            numeric_threshold_reached=10*correct >= 7*18000, whole_objective_audit_passed=audit_passed,
            actual_budget_consistent=budget_consistent,
            claim_70pct_supported=10*correct >= 7*18000 and audit_passed and budget_consistent,
            goal_status='ACTIVE; parent closure pending', one_experiment_one_seed=True,
            severe_overfit_noise_or_ceiling_claim=False, cost_veto=False, test_reselection=False)
        atomic_json(root/'reference_provenance.json', dict(recorded_utc=now(), references=provenance,
            historical_test_exposure=True, reference_inference_reruns=False, prediction_replay=False, solver_executions=0,
            extraction='fixed saved reports only; NPZ/raw-cost/ID arrays remain local for later inspection'))
        report_markdown(root, selected, lock, decision, compute)
        atomic_json(root/'analysis.json', dict(recorded_utc=now(), decision=decision, actual_compute=compute,
            selected={model: dict(epoch=record['epoch'], **{split: record[split]['macro'] for split in ('train', 'val', 'test')
                       if record.get(split) is not None}) for model, record in selected.items()},
            stage1_validation_evaluations=0, deployment='single NativeUtilitySelector saved stage2 x',
            inherited_pretrained_selectors=0, legacy_B_semantics='eight task mismatches unresolved/disclosed; R58 fully deferred'))
        receipt = dict(recorded_utc=now(), returncode=0, cpu_only=True, executions=1, wall_seconds=time.monotonic()-started,
            module_sha256=file_hash(Path(__file__)), comparison_rows=len(comparison), png_count=len(list(curves.rglob('*.png'))))
        atomic_json(root/'analysis_receipt.json', receipt)
        atomic_json(root/'delivery_receipt.json', dict(recorded_utc=now(), training=terminals['train'], test=terminals['test'],
            analysis=receipt, freeze_sha256=file_hash(root/'final_test_lock.json'), decision=decision,
            parent_reviews_objective_audit_memory_and_git=True, git_alone_contains_all_inference_assets=False,
            no_worker_git_actions=True, r58='fully deferred; eight legacy B-task mismatch disclosed'))
        print(json.dumps(dict(analysis=receipt, decision=decision)), flush=True)
    except BaseException as error:
        atomic_json(root/'analysis_failure.json', dict(recorded_utc=now(), exception_type=type(error).__name__,
            message=str(error), traceback=traceback.format_exc(), marker_preserved=True, automatic_retry=False))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    args = parser.parse_args()
    summarize(args.root.resolve())


if __name__ == '__main__':
    main()
