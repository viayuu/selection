"""One future CPU delivery pass over completed R68 and fixed saved references."""

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
from .r66_analysis import metric_rows
from .r67_analysis import fixed_references as prior_references, normalize_counts, read_updates
from .r68_experiment import ROOT, RUN_NAMES
from .r68_frozen_evaluation import require_terminal, validate_lock
from .r68_model import PARAMETER_MODE


def fixed_references(root):
    selected, provenance = prior_references(root)
    base = root.parent/'R67_supervised_contrastive_full18'
    test_path, lock_path = base/'test_results.json', base/'final_test_lock.json'
    test, lock = read(test_path), read(lock_path)
    if not test['complete'] or test['lock'] != lock:
        raise ValueError('R67 references must retain their original completed locked TEST results')
    names = {'A': 'A_classification_pretrain_seed2', 'B': 'B_contrastive_pretrain_seed2'}
    for arm, name in names.items():
        path = base/name/'result.json'
        result = read(path)
        selected['R67'+arm] = dict(epoch=lock['models'][arm]['epoch'], train=result['best']['train'],
                                 val=result['best']['val'], test=test['models'][arm])
        provenance['R67'+arm] = dict(result_path=str(path), result_sha256=file_hash(path),
            train_json_pointer='/best/train', validation_json_pointer='/best/val',
            test_path=str(test_path), test_sha256=file_hash(test_path), test_json_pointer='/models/'+arm,
            lock_path=str(lock_path), lock_sha256=file_hash(lock_path), locked_official_arm=lock['official_arm'],
            checkpoint_policy=result['checkpoint_rule'], parameter_mode=lock['models'][arm]['parameter_mode'],
            official_arm_policy='maximum original validation integer count; earliest epoch tie; arm tie A',
            deployment='historical single stage2 averaged-x selector; unchanged', historical_test_exposure=True)
    for model in ('R66A', 'R66B'):
        provenance[model]['deployment'] = 'historical two-selector F+g; unchanged'
    return selected, provenance


def numeric_summary(values):
    values = np.asarray(values, dtype=np.float64)
    return dict(min=float(values.min()), mean=float(values.mean()), max=float(values.max())) if values.size else None


def actual_compute(root, protocol, results, updates, test, terminals):
    prior_path = Path(protocol['inherited_identity']['terminal']['path']).parent/'stage1_result.json'
    prior = read(prior_path)
    inherited = dict(successful_updates=prior['successful_updates'], presentations=prior['presentations'],
        wall_seconds=prior['wall_seconds'], amp_retries=prior['amp_retries'], counted_once=True,
        source='R67B fixed stage1 terminal SupCon trunk only; entire fresh head670003',
        stage1_result_path=str(prior_path), stage1_result_sha256=file_hash(prior_path),
        source_identity=protocol['inherited_identity'], initialization=protocol['initialization'],
        new_stage1_training=False, fresh_encoders=False, inherited_trained_utility_head=False,
        inherited_projector=False)
    compute = dict(planned=dict(new_epochs_per_arm=40, new_updates_per_arm=9600,
        new_selected_rows_per_arm=7200000, new_updates_pair=19200, new_selected_rows_pair=14400000,
        successful_view_rows=dict(A=14400000, B=28800000),
        inherited_updates_once=9600, inherited_selected_rows_once=7200000),
        inherited=inherited, arms={}, training_terminal=terminals['train'], test_terminal=terminals['test'],
        parameter_mode=PARAMETER_MODE, parameter_averaging=False, perturbed_deployment_weights=False,
        flops_matched=False,
        selector_forwards_per_deployment=1, completed_solvers_per_deployment=1,
        solver_executions_during_delivery=0, timing_scope='ordinary complete evaluation walltime; not singleton latency',
        test_wall_seconds=test['wall_seconds'], actual_logged_counts_authoritative=True)
    for arm, result in results.items():
        rows = updates[arm]
        selected_rows = sum(r['rows'] for r in rows)
        steps = sum(r['base_optimizer_steps'] for r in rows)
        successful_views = sum(r['successful_view_rows'] for r in rows)
        attempted_views = sum(r['attempted_view_rows'] for r in rows)
        retries = sum(r['retries'] for r in rows)
        all_finite = all(r['finite_loss'] and r['finite_gradient'] for r in rows)
        one_step = all(r['base_optimizer_steps'] == r['weight_decay_steps'] == 1 for r in rows)
        restored = all(r['perturbation_restored_exactly'] is True for r in rows)
        sam_rng = all(r['rng_replayed_for_second_pass'] is (arm == 'B') for r in rows)
        budget_matches = (len(rows) == steps == result['successful_updates'] == 9600
            and selected_rows == result['presentations'] == 7200000
            and successful_views == result['successful_view_rows'] == protocol['successful_view_rows_per_arm'][arm]
            and attempted_views == result['attempted_view_rows'] and retries == result['amp_retries']
            and [r['successful_update'] for r in rows] == list(range(1, 9601))
            and all(r['successful_view_rows'] == r['rows']*(4 if arm == 'B' else 2) for r in rows)
            and all(r['attempted_view_rows'] >= r['successful_view_rows'] for r in rows)
            and inherited['successful_updates'] == 9600 and inherited['presentations'] == 7200000)
        compute['arms'][arm] = dict(successful_updates=steps, logged_successful_updates=len(rows),
            selected_rows=selected_rows, successful_view_rows=successful_views, attempted_view_rows=attempted_views,
            attempted_view_rows_by_pass={key: sum(r['attempted_view_rows_by_pass'][key] for r in rows)
                                         for key in ('first', 'perturbed')},
            amp_retries=retries, planned_budget_matches=budget_matches, finite_updates=all_finite,
            one_base_optimizer_and_weight_decay_step_per_update=one_step,
            perturbation_restored_exactly=restored, correct_second_pass_rng_replay=sam_rng,
            perturbation_norm=numeric_summary([r['effective_perturbation_norm'] for r in rows]),
            first_gradient_norm=numeric_summary([r['first_grad_norm'] for r in rows]),
            step_gradient_norm=numeric_summary([r['grad_norm'] for r in rows]),
            clip_coefficient=numeric_summary([r['clip_coefficient'] for r in rows]),
            amp_scale=numeric_summary([r['scale'] for r in rows]),
            wall_seconds=result['wall_seconds'], logged_update_seconds=sum(r['seconds'] for r in rows),
            peak_allocated_gib=max((r['peak_memory_gib'] for r in rows), default=0.),
            peak_reserved_gib=max((r['peak_reserved_gib'] for r in rows), default=0.),
            whole_device_peak_physical_gib=max((r['physical_used_gib'] for r in rows), default=0.),
            ordinary_test_evaluation_seconds=test['models'][arm]['complete_evaluation_seconds'],
            epochs_completed=result['epochs_completed'], result_recorded_successful_updates=result['successful_updates'],
            result_recorded_selected_rows=result['presentations'],
            last_successful_update=rows[-1]['successful_update'] if rows else 0,
            last_schedule_successful_update=rows[-1]['schedule_successful_update'] if rows else 0,
            last_selected_rows=rows[-1]['presentations'] if rows else 0,
            first_pass='native CE/RDrop at ordinary w',
            second_pass='native CE/RDrop at w+epsilon, same rows/dropout masks' if arm == 'B' else None)
    compute['actual_new_pair_successful_updates'] = sum(v['successful_updates'] for v in compute['arms'].values())
    compute['actual_new_pair_selected_rows'] = sum(v['selected_rows'] for v in compute['arms'].values())
    compute['actual_new_pair_successful_view_rows'] = sum(v['successful_view_rows'] for v in compute['arms'].values())
    compute['actual_new_pair_attempted_view_rows'] = sum(v['attempted_view_rows'] for v in compute['arms'].values())
    compute['rows_and_successful_updates_matched'] = (
        compute['arms']['A']['selected_rows'] == compute['arms']['B']['selected_rows']
        and compute['arms']['A']['successful_updates'] == compute['arms']['B']['successful_updates'])
    compute['pair_arm_wall_seconds'] = sum(v['wall_seconds'] for v in compute['arms'].values())
    compute['inherited_once_plus_new_pair_updates'] = inherited['successful_updates']+compute['actual_new_pair_successful_updates']
    compute['inherited_once_plus_new_pair_selected_rows'] = inherited['presentations']+compute['actual_new_pair_selected_rows']
    return compute


def validation_summaries(root, results, histories):
    summaries, last_rows = [], []
    for arm, history in histories.items():
        for label, record in (('best', results[arm]['best']), ('final', results[arm]['final'])):
            for split in ('train', 'val'):
                if record.get(split) is not None:
                    summaries += [dict(arm=arm, summary=label, **row)
                        for row in metric_rows('R68'+arm, split, record['epoch'], record[split])]
        grouped = {}
        for record in history[-5:]:
            for row in metric_rows('R68'+arm, 'val', record['epoch'], record['val']):
                last_rows.append(dict(arm=arm, **row))
                grouped.setdefault((row['kind'], row['scope']), []).append(row)
        for (kind, scope), rows in grouped.items():
            for label, function in (('last5_mean', np.mean), ('last5_population_sd', np.std),
                                    ('last5_min', np.min), ('last5_max', np.max)):
                summaries.append(dict(arm=arm, model='R68'+arm, split='val', kind=kind, scope=scope, summary=label,
                    **{key: float(function([r[key] for r in rows])) for key in (*METRICS, 'native_correct', 'native_total')}))
    write_csv(root/'stage2_summary.csv', summaries)
    write_csv(root/'last5_validation.csv', last_rows)
    common = []
    for split in ('train', 'val'):
        available = {arm: {r['epoch']: r[split] for r in history if r.get(split) is not None}
                     for arm, history in histories.items()}
        for arm, result in results.items():
            available[arm].setdefault(result['best']['epoch'], result['best'][split])
        for epoch in sorted(set(available['A']) & set(available['B'])):
            left = metric_rows('R68A', split, epoch, available['A'][epoch])
            right = {(r['kind'], r['scope']): r for r in metric_rows('R68B', split, epoch, available['B'][epoch])}
            for a in left:
                b = right[(a['kind'], a['scope'])]
                common.append(dict(epoch=epoch, global_epoch=40+epoch, split=split, kind=a['kind'], scope=a['scope'],
                    A_native_correct=a['native_correct'], B_native_correct=b['native_correct'], native_total=a['native_total'],
                    **{arm+'_'+key: row[key] for arm, row in (('A', a), ('B', b)) for key in METRICS},
                    **{key+('_B_minus_A_pp' if key.startswith('top') else '_B_minus_A'):
                       (b[key]-a[key])*(100 if key.startswith('top') else 1) for key in METRICS}))
    write_csv(root/'matched_common_epochs.csv', common)


def learning_figures(plt, curves, histories, updates):
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    for ax, metric in zip(axes.flat, METRICS):
        for arm, history in histories.items():
            for split, style in (('train', '--'), ('val', '-')):
                rows = [r for r in history if r.get(split) is not None]
                ax.plot([r['epoch'] for r in rows], [r[split]['macro']['macro_'+metric] for r in rows],
                        linestyle=style, label=arm+' '+split+' ordinary weights')
        ax.set(title=metric, xlabel='R68 fine-tuning epoch (global41-80)')
        ax.grid(alpha=.2)
        ax.legend(fontsize=7)
    axes.flat[-1].axis('off')
    fig.tight_layout()
    fig.savefig(curves/'learning_curves.png', dpi=150)
    plt.close(fig)
    for resolution in ('epoch', 'update'):
        source = {arm: ([dict(r['optimization'], epoch=r['epoch']) for r in history]
                         if resolution == 'epoch' else updates[arm]) for arm, history in histories.items()}
        xkey = 'epoch' if resolution == 'epoch' else 'successful_update'
        fig, axes = plt.subplots(3, 4, figsize=(18, 11))
        for arm, ax in zip(('A', 'B'), axes.flat):
            for key in ('loss', 'ce', 'kl'):
                ax.plot([r[xkey] for r in source[arm]], [r[key] for r in source[arm]],
                        label='first '+key, linewidth=.8)
                if arm == 'B':
                    ax.plot([r[xkey] for r in source[arm]], [r['perturbed_'+key] for r in source[arm]],
                            label='perturbed '+key, linestyle='--', linewidth=.8)
            ax.set(title=arm+' native CE/RDrop passes', xlabel=xkey)
        keys = ('lr', 'first_grad_norm', 'grad_norm', 'clip_coefficient', 'effective_perturbation_norm',
                'amp_retries' if resolution == 'epoch' else 'retries', 'peak_memory_gib', 'seconds',
                'successful_view_rows', 'attempted_view_rows')
        for ax, key in zip(list(axes.flat)[2:], keys):
            for arm, rows in source.items():
                ax.plot([r[xkey] for r in rows], [r[key] for r in rows], label=arm, linewidth=.8)
            ax.set(title=key, xlabel=xkey)
        for ax in axes.flat:
            ax.grid(alpha=.2)
            ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(curves/('optimization_'+resolution+'.png'), dpi=150)
        plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    for ax, key in zip(axes.flat, ('scale', 'perturbation_restored_exactly', 'rng_replayed_for_second_pass', 'base_optimizer_steps')):
        for arm, rows in updates.items():
            ax.plot([r['successful_update'] for r in rows], [r[key] for r in rows], label=arm, linewidth=.8)
        ax.set(title=key, xlabel='Successful update')
        ax.legend(fontsize=8)
        ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(curves/'amp_restoration_and_steps.png', dpi=150)
    plt.close(fig)


def selected_figures(plt, curves, selected):
    methods, diagnostics = [], []
    for model, record in selected.items():
        for split in ('train', 'val', 'test'):
            for task, value in record[split]['per_problem'].items():
                directory = curves/model/split/task
                directory.mkdir(parents=True, exist_ok=True)
                pool = value['pool']
                diagnostics.append(dict(model=model, split=split, problem=task,
                    **{key: value[key] for key in ('best_top1', 'best_top2', 'best_top3', 'sbs_name', 'sbs_cost', 'oracle_cost')},
                    **{key: json.dumps(value[key]) for key in ('top1_exceeds_methods', 'top1_not_exceeds_methods',
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
                        ax.axhline(selector, color='#b84332', label='single selector ordinary weights')
                        ax.legend(fontsize=8)
                    ax.set_xticks(np.arange(len(pool)), pool, rotation=55, ha='right')
                    ax.set(title=model+' '+split+' '+task, ylabel=label)
                    fig.tight_layout()
                    fig.savefig(directory/filename, dpi=150)
                    plt.close(fig)
    return methods, diagnostics


def report_markdown(root, selected, lock, decision, compute):
    official = lock['official_arm']
    lines = ['# R68 Locked Delivery', '',
        f"Official R68{official}, fine-tuning epoch{lock['models'][official]['epoch']}, globalepoch{lock['models'][official]['global_epoch']}: original18 TEST {decision['official_test_correct']}/18000 = {100*decision['official_test_top1']:.6f}% native Top1.", '',
        'Checkpoint and official arm use maximum INTEGER original18 VAL native correct/18000, earliest epoch exacttie, arm tieA. Costs and TEST never select. Each checkpoint is one complete NativeUtilitySelector at ordinary unperturbed AdamW weights, loaded directly without optimizer transition, averaging, projector, head transplant or task-wise assembly.', '',
        'This is one fixed-seed optimization contrast from the same R67B terminal stage1 SupCon trunk plus the entire identical fresh head670003. These are not fresh encoders. Both arms train all selector parameters for40 new epochs with native CE and legal FP32 symmetric RDrop. A uses ordinary AdamW; B uses global SAM rho.05 before clipping, restores the pre-pass RNG for identical dropout masks, copies original weights back exactly, clips the second mean gradient once and performs one AdamW/weight-decay step. SAM adds gradient compute: rows and updates match, FLOPs do not.', '',
        '## Selected Metrics', '',
        '| Model | Split | Top1 % | Top2 % | Top3 % | CE | Mean Cost | vsSBS % | Actual Regret % |',
        '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for model, record in selected.items():
        for split in ('train', 'val', 'test'):
            if record.get(split) is None:
                continue
            value = metrics(record[split]['macro'], 'macro_')
            lines.append(f"| {model} | {split} | {100*value['top1']:.4f} | {100*value['top2']:.4f} | {100*value['top3']:.4f} | {value['ce']:.6f} | {value['mean_cost']:.6f} | {value['vs_sbs_pct']:.6f} | {value['actual_regret_pct']:.6f} |")
    lines += ['',
        'comparison.csv and selected_metrics.csv contain task/family/ALL TRAIN/VAL/TEST. ALL is the original equal18-task macro; MVRP has15 tasks. Original native winners, full legal pools and ascending-global-ID exact utility ties remain. Raw costs are FP64. vsSBS is100*(selector mean cost/SBS cost-1); actual regret is mean per-query relative excess over the legal-pool oracle. Cross-task mean costs mix task units and are descriptive.', '',
        'stage2_summary.csv reports best/final complete TRAIN/VAL and last5 validation mean/population SD/min/max by task/family/ALL. Last5 are epochs36-40, not seed uncertainty. matched_common_epochs.csv compares only actually available common TRAIN/VAL epochs; no interpolation or selection by those comparisons. Every VAL epoch is complete. TRAIN evaluation occurs every5/final and selected-best. R68 has no new stage1 curve; inherited R67 stage1 SupCon is not mislabeled CE or given an invented validation curve.', '',
        'Historical R45A38/R60B39 retain their original cost-selected official policies. R64A/B,R65A/B,R66A/B,R67A/B retain original validation locks, parameter modes and official-arm rules. All references reuse fixed saved official reports, never inference, NPZ replay, retrospective TEST or solver runs. Historical TEST exposure makes version comparisons descriptive. Missing historical TRAIN, notably R45A, is disclosed in metric_availability.csv.', '',
        '## Actual Compute', '',
        '| Arm | New Updates | New Selected Rows | Successful View Rows | Attempted View Rows | AMP Retries | Wall s | Peak GiB |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for arm, entry in compute['arms'].items():
        lines.append(f"| {arm} | {entry['successful_updates']} | {entry['selected_rows']} | {entry['successful_view_rows']} | {entry['attempted_view_rows']} | {entry['amp_retries']} | {entry['wall_seconds']:.3f} | {entry['peak_allocated_gib']:.4f} |")
    prior = compute['inherited']
    lines += ['',
        f"Inherited R67B stage1 counted ONCE: {prior['successful_updates']} updates/{prior['presentations']} rows, {prior['wall_seconds']:.3f}s. Actual new pair: {compute['actual_new_pair_successful_updates']} updates/{compute['actual_new_pair_selected_rows']} selected rows. Inherited-once plus new pair: {compute['inherited_once_plus_new_pair_updates']} updates/{compute['inherited_once_plus_new_pair_selected_rows']} rows. Planned new perarm9600/7.2M, pair19200/14.4M; A14.4M and B28.8M successful view rows. Actual measured logs and receipts are authoritative, including retries and attempted views.", '',
        'The exact shared R67 global41-80 task/row/dropout slice is retained. Real256/task, three distinct tasks/update/effective768; tails16/task/effective48. Whole-triple AMP retries preserve rows/RNG, with counters advancing only after one successful base step. Fresh ordinary AdamW/scaler, external720-successful-update LR warmup to3e-4, native RDrop coefficient.35 warmup, one global norm clip1. No ScheduleFree optimizer/mode/averaging or early stop.', '',
        'optimization_curves.csv, task_optimization.csv and update_curves.csv preserve first and perturbed native loss/CE/KL separately, LR, first/step gradients, clipping, perturbation norm, exact restoration, RNG replay, AMP scale/retries, rows/views, time/memory and one-step counters. These are training diagnostics, not an extra curvature experiment. compute_summary.json includes actual inherited identity, counts and terminal receipts. TEST time is ordinary full-evaluation walltime including transfer, metrics and NPZ saving, not singleton latency.', '',
        f"SAM exceeds matched AdamW on INTEGER validation count: {decision['sam_exceeds_matched_control_validation']}; SAM matches or exceeds it: {decision['sam_matches_or_exceeds_matched_control_validation']}. SAM at least matches A but remains below fixed R60B TEST: {decision['sam_matches_A_validation_but_below_R60B_test_limited_progress']}. Such a result is limited progress; one seed does not establish severe overfit, label noise or an accuracy ceiling.", '',
        f"Numeric official locked TEST threshold reached: {decision['numeric_threshold_reached']}. Whole-objective audit passed: {decision['whole_objective_audit_passed']}. Supported70% claim: {decision['claim_70pct_supported']}. Goal remains ACTIVE pending parent whole-objective audit and closure; no70% promise follows from preparation or validation.", '',
        '## Assets and Limits', '',
        'Deployment is one selector forward and one completed solver per query; delivery evaluation runs zero solvers. Selected weights, NPZ/raw costs/IDs, raw update logs, W&B and regenerable caches remain LOCAL under user cleanup/ignore policy. Code, reports, configs, receipts, CSV curves, figures and reviews may later be curated once by parent. Nothing is force-added. Git alone does not contain all inference assets.', '',
        'The eight legacy B-task feasible-domain mismatches remain unresolved and disclosed, not repaired or used as a native70% metric veto. R58 is fully deferred. Parent owns objective audit, reviewers, persistent state/memory, publication and Git.', '']
    (root/'REPORT.md').write_text('\n'.join(lines))


def summarize(root):
    if root.resolve() != ROOT.resolve() or os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('Use the canonical root and CUDA_VISIBLE_DEVICES= for one CPU-only delivery pass')
    protocol, lock = validate_lock(root)
    test, marker = read(root/'test_results.json'), read(root/'final_test_started.json')
    terminals = dict(train=read(root/'training_terminal_receipt.json'), test=read(root/'test_terminal_receipt.json'))
    require_terminal(terminals['test'], 'r68_final_test', marker['slurm_step'])
    if (type(terminals['test'].get('elapsed_seconds')) not in (int, float)
            or not np.isfinite(terminals['test']['elapsed_seconds']) or terminals['test']['elapsed_seconds'] < 0
            or not terminals['test'].get('scheduler_evidence')):
        raise ValueError('Retain actual TEST elapsed_seconds and original scheduler_evidence')
    process, launch = read(root/'test_process_receipt.json'), read(root/'test_launch_receipt.json')
    if (not test.get('complete') or test['lock'] != lock or marker['lock'] != lock
            or test['tests_per_frozen_model'] != 1 or marker['allowed_attempts'] != 1
            or process['returncode'] != 0 or process['slurm_step'] != marker['slurm_step']
            or process['tmux_session'] != 'r68_final_test' or process['parameter_mode'] != PARAMETER_MODE
            or launch['slurm_step'] != marker['slurm_step'] or launch['tmux_session'] != 'r68_final_test'
            or terminals['test'].get('original_full18_fulllegalpool_tests_per_frozen_model') != 1
            or test.get('parameter_mode') != PARAMETER_MODE or test.get('original_full18') is not True
            or test.get('perturbed_weights') is not False or test.get('parameter_averaging') is not False
            or test.get('separate_latency_benchmark') is not False):
        raise ValueError('Analysis requires actual terminal0 and one complete ordinary-weight locked TEST per model')
    for arm in ('A', 'B'):
        value = test['models'][arm]
        if (not value['complete'] or value['evaluation_passes'] != 1 or value['native_total'] != 18000
                or set(value['per_problem']) != set(PROBLEMS) or any(v['n'] != 1000 for v in value['per_problem'].values())
                or value['selector_forwards_per_query'] != 1 or value['forward_batch_calls'] != 144
                or value['evaluated_checkpoint_sha256'] != lock['models'][arm]['inference_sha256']
                or value['model_state_sha256'] != lock['models'][arm]['model_state_sha256']
                or value['parameter_mode'] != PARAMETER_MODE or value['perturbed_weights'] is not False
                or value['parameter_averaging'] is not False or value['optimizer_mode_transition'] is not False
                or value['cached_test_scores'] is not False or value['batch_size'] != 128 or value['tails_retained'] is not True):
            raise ValueError('Require the exact complete original18 single-forward ordinary-weight TEST identities')
    with (root/'analysis_started.json').open('x') as stream:
        json.dump(dict(started_utc=now(), cpu_only=True, executions=1, module_sha256=file_hash(Path(__file__)),
            final_test_lock_sha256=file_hash(root/'final_test_lock.json'),
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
        compute = actual_compute(root, protocol, results, updates, test, terminals)
        atomic_json(root/'compute_summary.json', compute)
        selected = {'R68'+arm: dict(epoch=lock['models'][arm]['epoch'], train=result['best']['train'],
                    val=result['best']['val'], test=test['models'][arm]) for arm, result in results.items()}
        evaluations, task_rows, family_rows, optimization, task_optimization, update_rows = [], [], [], [], [], []
        for arm, history in histories.items():
            for row in history:
                identity = dict(arm=arm, stage=row['stage'], epoch=row['epoch'], global_epoch=row['global_epoch'])
                optimization.append(dict(identity, objective='native CE plus legal RDrop KL',
                    successful_updates=row['successful_updates'], presentations=row['presentations'],
                    cumulative_successful_view_rows=row['successful_view_rows'],
                    cumulative_attempted_view_rows=row['attempted_view_rows'], **row['optimization']))
                task_optimization += [dict(identity, problem=task, **entry)
                                      for task, entry in row['task_optimization'].items()]
                for split in ('train', 'val'):
                    value = row.get(split)
                    if value is None:
                        continue
                    evaluations += [dict(identity, **entry) for entry in metric_rows('R68'+arm, split, row['epoch'], value)]
                    task_rows += [dict(arm=arm, epoch=row['epoch'], split=split, problem=p, **metrics(m))
                                  for p, m in value['per_problem'].items()]
                    family_rows += [dict(arm=arm, epoch=row['epoch'], split=split, family=p, **metrics(m, 'macro_'))
                                    for p, m in value['families'].items()]
            for row in updates[arm]:
                update_rows.append({key: json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else value
                                    for key, value in row.items()})
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
        write_csv(root/'selected_metrics.csv', [r for r in comparison if r['model'].startswith('R68')])
        write_csv(root/'metric_availability.csv', [dict(model=model, split=split, available=record.get(split) is not None,
            policy='fixed saved selected reports only; no inference/replay to fill missing historical TRAIN')
            for model, record in selected.items() for split in ('train', 'val', 'test')])
        indexed = {(r['model'], r['split'], r['kind'], r['scope']): r for r in comparison}
        pairs = [('R68B', 'R68A')]+[(model, ref) for model in ('R68A', 'R68B') for ref in historical]
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
        validation_summaries(root, results, histories)
        official = lock['official_arm']
        correct = test['models'][official]['native_correct']
        audit_path = root/'objective_audit.json'
        audit = read(audit_path) if audit_path.exists() else {}
        audit_passed = (audit.get('passed') is True and audit.get('scope') == 'whole objective'
            and audit.get('final_test_lock_sha256') == file_hash(root/'final_test_lock.json')
            and audit.get('test_results_sha256') == file_hash(root/'test_results.json'))
        budget_consistent = all(all(arm[key] for key in ('planned_budget_matches', 'finite_updates',
            'one_base_optimizer_and_weight_decay_step_per_update', 'perturbation_restored_exactly',
            'correct_second_pass_rng_replay')) for arm in compute['arms'].values())
        val_delta = lock['models']['B']['native_correct']-lock['models']['A']['native_correct']
        decision = dict(official_arm=official, decision_source='integer VAL only; earliest epoch tie; arm tie A',
            validation_count_B_minus_A=val_delta, sam_exceeds_matched_control_validation=val_delta > 0,
            sam_matches_or_exceeds_matched_control_validation=val_delta >= 0,
            sam_matches_A_validation_but_below_R60B_test_limited_progress=val_delta >= 0 and
                test['models']['B']['macro']['macro_top1'] < selected['R60B']['test']['macro']['macro_top1'],
            official_test_correct=correct, official_test_total=18000, official_test_top1=correct/18000,
            numeric_threshold_reached=10*correct >= 7*18000, whole_objective_audit_passed=audit_passed,
            actual_budget_consistent=budget_consistent,
            claim_70pct_supported=10*correct >= 7*18000 and audit_passed and budget_consistent,
            goal_status='ACTIVE; parent whole-objective closure pending', one_experiment_one_seed=True,
            shared_inherited_contrastive_prior=True, fresh_encoders=False, flops_matched=False,
            severe_overfit_noise_or_ceiling_claim=False, cost_veto=False, test_reselection=False)
        atomic_json(root/'reference_provenance.json', dict(recorded_utc=now(), references=provenance,
            historical_test_exposure=True, reference_inference_reruns=False, prediction_replay=False, solver_executions=0,
            extraction='fixed saved original official JSON reports only; NPZ/raw costs/IDs stay local'))
        report_markdown(root, selected, lock, decision, compute)
        atomic_json(root/'analysis.json', dict(recorded_utc=now(), decision=decision, actual_compute=compute,
            selected={model: dict(epoch=record['epoch'], **{split: record[split]['macro'] for split in ('train', 'val', 'test')
                       if record.get(split) is not None}) for model, record in selected.items()},
            deployment='single complete ordinary unperturbed AdamW NativeUtilitySelector', new_stage1_training=False,
            inherited_trunk='R67B terminal stage1 SupCon; budget counted once separately',
            legacy_B_semantics='eight task mismatches unresolved/disclosed; R58 fully deferred'))
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
