"""One CPU delivery pass after the actual single paired locked R66 TEST succeeds."""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

from code.unified_selector.registry import PROBLEMS
from .multitask_probe import file_hash
from .r59_experiment import atomic_json
from .r64_analysis import METRICS, write_csv
from .r65_analysis import group_panels, historical_records, metrics, now, read
from .r66_experiment import ROOT, RUN_NAMES
from .r66_model import BASE_SHA256, INFERENCE_RULE


OBJECTIVES = {'A': 'standalone g', 'B': 'composite F+g'}
OPT_METRICS = ('loss', 'ce', 'kl', 'lr', 'grad_norm', 'clip_coefficient',
               'amp_retries', 'peak_memory_gib')


def metric_rows(model, split, epoch, value):
    common = dict(model=model, split=split, epoch=epoch)
    per = value['per_problem']
    rows = [dict(common, scope='ALL', kind='aggregate', native_correct=value['native_correct'],
        native_total=value['native_total'], **metrics(value['macro'], 'macro_'))]
    for family, entry in value['families'].items():
        tasks = [family] if family != 'MVRP' else [p for p in per if p not in ('TSP', 'CVRP', 'ATSP')]
        rows.append(dict(common, scope=family, kind='family',
            native_correct=sum(per[p].get('native_correct', round(per[p]['top1']*per[p]['n'])) for p in tasks),
            native_total=sum(per[p]['n'] for p in tasks), **metrics(entry, 'macro_')))
    rows += [dict(common, scope=task, kind='task', native_correct=entry.get('native_correct'),
        native_total=entry['n'], **metrics(entry)) for task, entry in per.items()]
    return rows


def reference_records(root):
    selected, provenance = historical_records(root)
    base = root.parent/'R65_schedule_free_full18'
    test_path, lock_path = base/'test_results.json', base/'final_test_lock.json'
    test, lock = read(test_path), read(lock_path)
    if not test['complete'] or test['lock'] != lock:
        raise ValueError('Reuse only the original completed locked R65 reference results')
    for arm, name in {'A': 'A_constant_adamw_seed2', 'B': 'B_schedule_free_seed2'}.items():
        path = base/name/'result.json'
        result = read(path)
        model = 'R65'+arm
        selected[model] = dict(epoch=lock['models'][arm]['epoch'], train=result['best']['train'],
            val=result['best']['val'], test=test['models'][arm])
        provenance[model] = dict(result_path=str(path), result_sha256=file_hash(path),
            train_json_pointer='/best/train', validation_json_pointer='/best/val',
            test_path=str(test_path), test_sha256=file_hash(test_path), test_json_pointer='/models/'+arm,
            lock_path=str(lock_path), lock_sha256=file_hash(lock_path),
            locked_official_arm=lock['official_arm'], checkpoint_policy=result['checkpoint_rule'],
            official_arm_policy='maximum original validation integer count; tie A',
            parameter_mode=lock['models'][arm]['parameter_mode'], historical_test_exposure=True)
    policies = {'R45A': 'historical official epoch38; original validation cost policy',
        'R60B': 'historical official epoch39; original validation cost policy',
        'R64A': 'original integer VAL count; earliest tie; historical official arm B',
        'R64B': 'original integer VAL count; earliest tie; historical official arm B'}
    for model, policy in policies.items():
        provenance[model]['fixed_policy_disclosure'] = policy
    return selected, provenance


def learning_panels(plt, curves, histories, updates):
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    for ax, metric in zip(axes.flat, METRICS):
        for arm, history in histories.items():
            for split, style in (('train', '--'), ('val', '-')):
                rows = [row for row in history if row.get(split) is not None]
                ax.plot([r['epoch'] for r in rows], [r[split]['macro']['macro_'+metric] for r in rows],
                    linestyle=style, label=arm+' '+split+' F+g_x')
        ax.set(title=metric, xlabel='Epoch')
        ax.grid(alpha=.2)
        ax.legend(fontsize=7)
    axes.flat[-1].axis('off')
    fig.tight_layout()
    fig.savefig(curves/'learning_curves.png', dpi=150)
    plt.close(fig)
    sources = [('optimization.png', {arm: [dict(epoch=r['epoch'], **r['optimization']) for r in history]
        for arm, history in histories.items()}, 'epoch'), ('optimization_updates.png', updates, 'successful_update')]
    for filename, source, xkey in sources:
        fig, axes = plt.subplots(2, 4, figsize=(18, 8))
        for ax, metric in zip(axes.flat, OPT_METRICS):
            for arm, rows in source.items():
                ax.plot([r[xkey] for r in rows], [r[metric] for r in rows],
                    label=arm+' objective: '+OBJECTIVES[arm], linewidth=.8, alpha=.8)
            ax.set(title='effective LR' if metric == 'lr' else metric, xlabel=xkey)
            ax.grid(alpha=.2)
            ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(curves/filename, dpi=150)
        plt.close(fig)


def selected_panels(plt, curves, selected):
    method_rows, diagnostics = [], []
    for model, record in selected.items():
        for split in ('train', 'val', 'test'):
            for problem, value in record[split]['per_problem'].items():
                directory = curves/model/split/problem
                directory.mkdir(parents=True, exist_ok=True)
                methods = value['pool']
                diagnostics.append(dict(model=model, split=split, problem=problem,
                    **{key: value[key] for key in ('best_top1', 'best_top2', 'best_top3',
                        'sbs_name', 'sbs_cost', 'oracle_cost')},
                    **{key: json.dumps(value[key]) for key in ('top1_exceeds_methods',
                        'top1_not_exceeds_methods', 'cost_exceeds_methods', 'cost_not_exceeds_methods')}))
                for i, method in enumerate(methods):
                    method_rows.append(dict(model=model, split=split, problem=problem, method=method,
                        method_top1=value['method_top1'][method], method_mean_cost=value['method_mean_cost'][method],
                        selector_top1=value['top1'], selector_mean_cost=value['mean_cost'],
                        pick_fraction=value['pick_dist'][i]))
                for filename, values, selector, title in (
                    ('top1_vs_single_methods.png', [value['method_top1'][m] for m in methods], value['top1'], 'Native Top1'),
                    ('mean_cost_vs_single_methods.png', [value['method_mean_cost'][m] for m in methods], value['mean_cost'], 'Mean cost'),
                    ('arm_distribution.png', value['pick_dist'], None, 'Selection distribution')):
                    fig, ax = plt.subplots(figsize=(max(6, .6*len(methods)), 4))
                    ax.bar(np.arange(len(methods)), values, color='#227f79')
                    if selector is not None:
                        ax.axhline(selector, color='#b84332', label='composite F+g_x')
                        ax.legend()
                    ax.set_xticks(np.arange(len(methods)), methods, rotation=55, ha='right')
                    ax.set(title=model+' '+split+' '+problem, ylabel=title)
                    fig.tight_layout()
                    fig.savefig(directory/filename, dpi=150)
                    plt.close(fig)
    return method_rows, diagnostics


def validation_summaries(root, results, histories):
    summaries, last5_rows = [], []
    for arm, history in histories.items():
        for summary, record in [('best', results[arm]['best']), ('final', results[arm]['final'])]:
            summaries += [dict(arm=arm, summary=summary, **row)
                for row in metric_rows('R66'+arm, 'val', record['epoch'], record['val'])]
        last5 = {}
        for record in history[-5:]:
            rows = metric_rows('R66'+arm, 'val', record['epoch'], record['val'])
            last5_rows += [dict(arm=arm, **row) for row in rows]
            for row in rows:
                last5.setdefault((row['kind'], row['scope']), []).append(row)
        for (kind, scope), rows in last5.items():
            for summary, function in [('last5_mean', np.mean), ('last5_population_sd', np.std),
                    ('last5_min', np.min), ('last5_max', np.max)]:
                summaries.append(dict(arm=arm, model='R66'+arm, split='val', kind=kind, scope=scope,
                    summary=summary, **{key: float(function([r[key] for r in rows]))
                        for key in (*METRICS, 'native_correct', 'native_total')}))
    write_csv(root/'validation_summary.csv', summaries)
    write_csv(root/'last5_validation.csv', last5_rows)


def markdown(root, selected, results, histories, lock, decision, compute):
    official = decision['official_arm']
    goal = ('Locked official TEST meets the numeric target; parent completion audit is pending.'
        if decision['locked_official_test_reaches70pct'] else 'Full18 native70% remains ACTIVE and unachieved.')
    lines = ['# R66 Locked Delivery', '',
        f"Official **R66{official}, epoch{lock['models'][official]['epoch']}**, frozen by original integer validation count before TEST: **{decision['locked_official_test_correct']}/18000 = {100*decision['locked_official_test_top1']:.6f}% native TEST Top1**. {goal}", '',
        'Both arms inherit the same immutable R65B40 saved averaged-x base F and start one identically copied fresh member g at memberseed660002, experimentseed2. A optimizes standalone-g native CE/KL; B optimizes composite F+g native CE/KL. BOTH evaluate and deploy F+g_x with coefficient1. The ONLY new A/B intervention is whether member-training CE/KL uses the fixed F offset.', '',
        'Official ScheduleFree trains member y and evaluates/saves member x; TEST loads the complete two-model saved-x bundle directly with no optimizer transition. Real256/task, three-task effective768, tail16/task/effective48, fixed40 epochs/9600 updates/7.2M presentations per member. Internal720-update warmup to3e-4 then constant; native CE plus existing RDrop, same plans/streams, one global clip1 and whole-triple retries. No controller or early stop.', '',
        '## Validation Freeze', '',
        '| Arm | Epoch | VAL Correct | Mode | Unique Bundle SHA256 |',
        '| --- | ---: | ---: | --- | --- |']
    for arm, record in lock['models'].items():
        lines.append(f"| {arm} | {record['epoch']} | {record['native_correct']}/18000 | {record['parameter_mode']} | `{record['unique_bundle_sha256']}` |")
    lines += ['',
        'Each unique selected-epoch bundle is mechanically associated with its published best.pt; the lock retains both paths/hashes and their identical saved base/member states. Max integer native VAL count selects each checkpoint with earliest exact tie; official-arm tie chooses A. Cost, TEST and parameter modes do not choose weights. No per-task assembly.', '',
        f"B minus A validation: **{decision['validation_count_B_minus_A']:+d} correct / {decision['validation_top1_B_minus_A_pp']:+.6f}pp**. This is a descriptive one-stage, one-seed comparison. Standalone member optimization CE is not composite evaluation CE; max-validation performance does not establish heldout performance. No general boosting, label-noise or accuracy-ceiling conclusion follows.", '',
        '## Composite Native Metrics', '',
        '| Model | Split | Top1 % | Top2 % | Top3 % | CE | Mean Cost | vsSBS % | Actual Regret % |',
        '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for arm in ('A', 'B'):
        for split in ('train', 'val', 'test'):
            m = metrics(selected['R66'+arm][split]['macro'], 'macro_')
            lines.append(f"| R66{arm} | {split} | {100*m['top1']:.4f} | {100*m['top2']:.4f} | {100*m['top3']:.4f} | {m['ce']:.6f} | {m['mean_cost']:.6f} | {m['vs_sbs_pct']:.6f} | {m['actual_regret_pct']:.6f} |")
    lines += ['',
        'ALL is the equal18-task macro; MVRP has15 tasks. Native Top1/2/3 use original native winners, raw FP64 costs drive cost/regret metrics, and exact utility ties use ascending global solver ID. vsSBS=100*(selector mean cost/SBS cost-1); actual regret is mean per-query relative excess over the legal-pool oracle. Mixed-task mean costs are descriptive across task units. All18 tasks, four families and ALL for TRAIN/VAL/TEST are in comparison.csv.', '',
        '## Fixed Historical References', '',
        '| Model | Epoch | VAL Top1 % | TEST Top1 % | VAL CE | TEST CE |',
        '| --- | ---: | ---: | ---: | ---: | ---: |']
    for model, record in selected.items():
        val, test = record['val']['macro'], record['test']['macro']
        lines.append(f"| {model} | {record['epoch']} | {100*val['macro_top1']:.4f} | {100*test['macro_top1']:.4f} | {val['macro_ce']:.6f} | {test['macro_ce']:.6f} |")
    lines += ['',
        'R45A38 and R60B39 preserve historical official/cost-selected policies; R45A has no saved selected TRAIN evaluation. R64A/B and R65A/B reuse their original locked results, each historical official arm B. Paths, JSON pointers, hashes and fixed policies are in reference_provenance.json. Historical TEST is exposed, not pristine. Cross-version comparisons are descriptive; no reference inference, prediction replay or solver operation occurred.', '',
        '## Validation Trajectory', '',
        '| Arm | Best Top1 % | Final Top1 % | Last5 Mean % | Last5 Min % | Last5 Max % |',
        '| --- | ---: | ---: | ---: | ---: | ---: |']
    for arm, history in histories.items():
        last5 = [row['val']['macro']['macro_top1'] for row in history[-5:]]
        lines.append(f"| {arm} | {100*results[arm]['best']['val']['macro']['macro_top1']:.4f} | {100*results[arm]['final']['val']['macro']['macro_top1']:.4f} | {100*np.mean(last5):.4f} | {100*min(last5):.4f} | {100*max(last5):.4f} |")
    lines += ['',
        'validation_summary.csv reports all seven metrics/counts for best/final and last5 mean/population SD/min/max for ALL, families and every task; last5_validation.csv retains each epoch. These temporal summaries are not seed uncertainty. Curves retain complete saved evaluation histories and correctly labeled standalone-A/composite-B objective loss/CE/KL, LR, gradients, clipping, retries and memory.', '',
        '## Compute and Deployment', '',
        '| Arm | New Updates | Presentations | AMP Retries | Arm Wall s | Update Wall s | Peak GiB | Live TEST s |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for arm, c in compute['arms'].items():
        lines.append(f"| {arm} | {c['successful_updates']} | {c['presentations']} | {c['amp_retries']} | {c['wall_seconds']:.3f} | {c['logged_update_seconds']:.3f} | {c['peak_allocated_gib']:.3f} | {c['ordinary_test_evaluation_seconds']:.3f} |")
    train, test = compute['training_terminal'], compute['test_terminal']
    lines += ['',
        f"New paired training totals19200 successful updates/14.4M presentations and{compute['new_pair_arm_wall_seconds']:.3f}s arm wall. Separately, shared inherited F was trained once earlier for9600 updates/7.2M presentations/{compute['inherited_base']['training_arm_wall_seconds']:.3f}s. These are marginal member costs plus an explicitly inherited base, not fully fresh training or total compute equal to one old model.", '',
        f"Owned training1465.94: COMPLETED0:0, GPU allocation step wall{train['elapsed_seconds']}s. Owned TEST{test['slurm_step']}: COMPLETED0:0, GPU allocation step wall{test['elapsed_seconds']}s; paired TEST body{compute['test_wall_seconds']:.3f}s. These are actual sole-GPU job/stage wall times including I/O/initialization where applicable, not hardware kernel-active time. Both tmux handles terminal. Only gpu03/index1 UUID GPU-cb626bdb-4319-2f5e-b91f-17fcaa44a83d in existing1465 was used.", '',
        f"Deployment rule: `{INFERENCE_RULE}`. Every current raw input goes through BOTH live selectors, FP32 eval batch128 with tails, with caching disabled. Ordinary complete evaluation times above include both forwards, transfers, metrics and all prediction saves. No separate latency/singleton benchmark was run. Two selector forwards choose one completed solver/query; this cost-record evaluation executed zero solvers. TRAIN/VAL base-cache times are not live two-selector deployment timings.", '',
        '## Delivery Artifacts', '',
        'final_test_lock.json, final_test_started.json, test_data_manifest.json, test_results_A/B.json, test_results.json and training/test_terminal_receipt.json retain the one-time frozen delivery evidence. compute_summary.json, analysis.json, analysis_receipt.json and delivery_receipt.json retain budgets, decisions and this one CPU analysis pass.',
        'comparison.csv/comparison_deltas.csv/selected_metrics.csv contain full18/family/split metrics; learning/task/family/optimization/update_curves.csv and task_optimization.csv retain curves. method_metrics.csv/task_diagnostics.csv retain single-method baselines, exceeded/not-exceeded lists and selection distributions. curves/ contains evaluation/optimization panels and per-task Top1/cost/distribution plots.',
        'Goal decisions use only official locked TEST native Top1 >=70%, with no cost or margin veto. Nightmare100/originalfull18/full legal pools/single completed solver remains the scope. R58 is fully deferred; the legacy eight B-task feasible-domain mismatch remains unresolved and disclosed.',
        'Parent actual Astra primary TRAIN/VAL-only next plan must use R66_before_test reviewer memory to exclude new R66 TEST, followed by one fresh delivery review and one whole-R66 commit/push. These remain pending. No R67, new reviewer/agent, Git action or further experiment was performed by this worker.', '']
    (root/'RESULTS.md').write_text('\n'.join(lines))


def summarize(root):
    root = Path(root).resolve()
    if root != ROOT.resolve() or os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('Use the canonical R66 root and CUDA_VISIBLE_DEVICES empty for CPU analysis')
    lock, test, marker = read(root/'final_test_lock.json'), read(root/'test_results.json'), read(root/'final_test_started.json')
    train_terminal, test_terminal = read(root/'training_terminal_receipt.json'), read(root/'test_terminal_receipt.json')
    for terminal in (train_terminal, test_terminal):
        if terminal['state'] != 'COMPLETED' or terminal['exit_code'] != '0:0' or terminal['tmux_terminal'] is not True:
            raise ValueError('Analysis requires actual successful terminal receipts for both owned stages')
    if (not lock['authorized'] or not test['complete'] or test['lock'] != lock or marker['lock'] != lock
            or set(test['models']) != {'A', 'B'} or test['tests_per_frozen_model'] != 1
            or marker['allowed_attempts'] != 1 or test_terminal['original_full18_fulllegalpool_tests_per_frozen_model'] != 1
            or marker['slurm_step'] != test_terminal['slurm_step'] or train_terminal['slurm_step'] != '1465.94'):
        raise ValueError('Report only the authorized one-pass locked R66 TEST after its actual terminal success')
    results = {arm: read(root/name/'result.json') for arm, name in RUN_NAMES.items()}
    histories = {arm: read(root/name/'history.json') for arm, name in RUN_NAMES.items()}
    for arm in ('A', 'B'):
        value = test['models'][arm]
        if (len(histories[arm]) != 40 or value['evaluation_passes'] != 1 or not value['complete']
                or value['native_total'] != 18000 or set(value['per_problem']) != set(PROBLEMS)
                or value['evaluated_bundle_sha256'] != lock['models'][arm]['unique_bundle_sha256']
                or value['cached_base_inference'] is not False):
            raise ValueError('Require complete fixed-horizon histories and one original locked TEST per bundle')
    with (root/'analysis_started.json').open('x') as stream:
        json.dump(dict(started_utc=now(), cpu_only=True, one_pass=True,
            module=str(Path(__file__).resolve()), test_terminal_receipt='test_terminal_receipt.json',
            final_test_started_sha256=file_hash(root/'final_test_started.json')), stream, indent=2)
        stream.write('\n')
    started = time.monotonic()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    plt.rcParams.update({'font.size': 9})
    curves = root/'curves'
    curves.mkdir(exist_ok=True)
    evaluations, tasks, families, optimization, task_optimization, updates = [], [], [], [], [], {}
    selected = {}
    for arm, history in histories.items():
        result = results[arm]
        selected['R66'+arm] = dict(epoch=lock['models'][arm]['epoch'], train=result['best']['train'],
            val=result['best']['val'], test=test['models'][arm])
        for row in history:
            optimization.append(dict(arm=arm, epoch=row['epoch'], **row['optimization']))
            task_optimization += [dict(arm=arm, epoch=row['epoch'], problem=task,
                objective_scope=OBJECTIVES[arm], **value) for task, value in row['task_optimization'].items()]
            for split in ('train', 'val'):
                value = row.get(split)
                if value is None:
                    continue
                evaluations.append(dict(arm=arm, epoch=row['epoch'], split=split, evaluation_scope='F+g_x',
                    native_correct=value['native_correct'], native_total=value['native_total'], **metrics(value['macro'], 'macro_')))
                tasks += [dict(arm=arm, epoch=row['epoch'], split=split, problem=p,
                    evaluation_scope='F+g_x', **metrics(m)) for p, m in value['per_problem'].items()]
                families += [dict(arm=arm, epoch=row['epoch'], split=split, family=p,
                    evaluation_scope='F+g_x', **metrics(m, 'macro_')) for p, m in value['families'].items()]
        updates[arm] = []
        with (root/RUN_NAMES[arm]/'updates.jsonl').open() as stream:
            for line in stream:
                row = json.loads(line)
                entry = {key: row[key] for key in ('epoch', 'successful_update', 'presentations', 'rows',
                    'loss', 'ce', 'kl', 'lr', 'optimizer_base_lr', 'grad_norm', 'clip_coefficient',
                    'seconds', 'peak_memory_gib', 'peak_reserved_gib', 'physical_used_gib',
                    'finite_loss', 'finite_gradient')}
                updates[arm].append(dict(entry, objective_scope=OBJECTIVES[arm], amp_retries=row['retries']))
    for name, rows in [('learning_curves', evaluations), ('task_curves', tasks), ('family_curves', families),
            ('optimization_curves', optimization), ('task_optimization', task_optimization),
            ('update_curves', [dict(arm=arm, **row) for arm, rows in updates.items() for row in rows])]:
        write_csv(root/(name+'.csv'), rows)
    learning_panels(plt, curves, histories, updates)
    group_panels(plt, curves, tasks, 'problem')
    group_panels(plt, curves, families, 'family')
    methods, diagnostics = selected_panels(plt, curves, selected)
    write_csv(root/'method_metrics.csv', methods)
    write_csv(root/'task_diagnostics.csv', diagnostics)
    historical, provenance = reference_records(root)
    selected.update(historical)
    # Older saved evaluations do not all publish integer aggregate counts.
    for record in selected.values():
        for split in ('train', 'val', 'test'):
            if split not in record:
                continue
            value = record[split]
            value.setdefault('native_total', sum(p['n'] for p in value['per_problem'].values()))
            value.setdefault('native_correct', sum(p.get('native_correct', round(p['top1']*p['n']))
                for p in value['per_problem'].values()))
    comparison = [row for model, record in selected.items() for split in ('train', 'val', 'test')
        if split in record for row in metric_rows(model, split, record['epoch'], record[split])]
    write_csv(root/'comparison.csv', comparison)
    write_csv(root/'selected_metrics.csv', [row for row in comparison if row['model'] in ('R66A', 'R66B')])
    indexed = {(row['model'], row['split'], row['kind'], row['scope']): row for row in comparison}
    pairs = [('R66B', 'R66A')]+[(model, ref) for model in ('R66A', 'R66B') for ref in historical]
    deltas = []
    for left, right in pairs:
        for split in ('val', 'test'):
            for kind, scope in [('aggregate', 'ALL')]+[('family', name) for name in ('TSP', 'CVRP', 'ATSP', 'MVRP')]+[('task', p) for p in PROBLEMS]:
                a, b = indexed[left, split, kind, scope], indexed[right, split, kind, scope]
                deltas.append(dict(comparison=left+' minus '+right, split=split, kind=kind, scope=scope,
                    **{key+('_delta_pp' if key.startswith('top') else '_delta'):
                        (a[key]-b[key])*(100 if key.startswith('top') else 1) for key in METRICS}))
    write_csv(root/'comparison_deltas.csv', deltas)
    validation_summaries(root, results, histories)
    official = lock['official_arm']
    a, b = (lock['models'][arm]['native_correct'] for arm in ('A', 'B'))
    official_test = test['models'][official]
    decision = dict(official_arm=official, decision_source='published integer original validation counts only; tie A',
        validation_count_B_minus_A=b-a, validation_top1_B_minus_A_pp=100*(b-a)/18000,
        offset_training_exceeds_independent_validation=b > a,
        locked_official_test_correct=official_test['native_correct'], locked_official_test_total=18000,
        locked_official_test_top1=official_test['native_correct']/18000,
        locked_official_test_reaches70pct=10*official_test['native_correct'] >= 7*18000,
        goal_status='ACTIVE', goal_completion_audit_pending=True, one_stage_one_seed=True,
        cost_regret_or_margin_veto=False, test_reselection=False, posterior_per_task_assembly=False)
    inherited_path = root.parent/'R65_schedule_free_full18/B_schedule_free_seed2/result.json'
    inherited = read(inherited_path)
    compute = dict(arms={}, training_terminal=train_terminal, test_terminal=test_terminal,
        new_pair_updates=19200, new_pair_presentations=14400000,
        new_pair_arm_wall_seconds=sum(result['wall_seconds'] for result in results.values()),
        new_training_gpu_allocation_step_seconds=train_terminal['elapsed_seconds'],
        new_test_gpu_allocation_step_seconds=test_terminal['elapsed_seconds'],
        test_wall_seconds=test['wall_seconds'], inherited_base=dict(checkpoint_sha256=BASE_SHA256,
            successful_updates=9600, presentations=7200000, counted_once_for_pair=True,
            training_arm_wall_seconds=inherited['wall_seconds'], result_path=str(inherited_path),
            result_sha256=file_hash(inherited_path)), new_pilots_or_sizing=0,
        selector_forwards_per_deployment=2, completed_solvers_per_deployment=1,
        inference_scope='ordinary live F+g_x FP32 eval batch128; original full18/fulllegalpool; no timing benchmark',
        solver_executions_during_delivery=0)
    for arm, result in results.items():
        rows = updates[arm]
        compute['arms'][arm] = dict(epochs_completed=result['epochs_completed'], successful_updates=result['successful_updates'],
            presentations=result['presentations'], amp_retries=result['amp_retries'], wall_seconds=result['wall_seconds'],
            objective_scope=OBJECTIVES[arm], evaluation_scope='composite F+g_x', actual_execution=result['actual_execution'],
            logged_update_seconds=sum(row['seconds'] for row in rows), peak_allocated_gib=max(row['peak_memory_gib'] for row in rows),
            peak_reserved_gib=max(row['peak_reserved_gib'] for row in rows),
            logged_whole_device_peak_physical_gib=max(row['physical_used_gib'] for row in rows),
            selected_evaluation_seconds={split: selected['R66'+arm][split].get('seconds') for split in ('train', 'val', 'test')},
            train_val_timing_uses_cached_base=True,
            ordinary_test_evaluation_seconds=test['models'][arm]['complete_evaluation_seconds'],
            selected_validation_count_delta=result['best']['selected_validation_count_delta'])
    atomic_json(root/'reference_provenance.json', dict(recorded_utc=now(), references=provenance,
        historical_test_exposure=True, reference_inference_reruns=False, prediction_or_solver_replays=False,
        transformation='Structured saved-metric extraction only, after locked R66 TEST terminal success'))
    atomic_json(root/'compute_summary.json', compute)
    markdown(root, selected, results, histories, lock, decision, compute)
    report = dict(recorded_utc=now(), decision=decision, selected={model: dict(epoch=record['epoch'],
        **{split: record[split]['macro'] for split in ('train', 'val', 'test') if split in record})
        for model, record in selected.items()}, actual_compute=compute, reference_provenance='reference_provenance.json',
        objective_scopes=OBJECTIVES, evaluation_scope='both composite F+g_x', locked_test_reported=True,
        historical_test_exposure_disclosed=True, legacy_B_semantics='unresolved; R58 fully deferred',
        parent_plan_review_scope='TRAIN/VAL only; use pre-TEST R66 reviewer memory snapshot',
        new_modules_only=True, frozen_sources_or_weights_edited=False)
    atomic_json(root/'analysis.json', report)
    receipt = dict(recorded_utc=now(), returncode=0, cpu_only=True, executions=1,
        wall_seconds=time.monotonic()-started, wall_scope='analysis body after imports and exclusive start',
        module=str(Path(__file__).resolve()), module_sha256=file_hash(Path(__file__)),
        comparison_rows=len(comparison), png_count=len(list(curves.rglob('*.png'))),
        official_arm=official, official_test_top1=decision['locked_official_test_top1'])
    atomic_json(root/'analysis_receipt.json', receipt)
    atomic_json(root/'delivery_receipt.json', dict(recorded_utc=now(), training=train_terminal,
        freeze=dict(path=str(root/'final_test_lock.json'), official_arm=official, validation_only=True),
        test=test_terminal, analysis=receipt, decision=decision,
        parent_primary_train_validation_review_pending=True, parent_fresh_delivery_review_pending=True,
        plan_reviewer_memory_snapshot='review-stage/r58-to70/R66_before_test_reviewer_memory.md',
        parent_whole_version_commit_push_pending=True, worker_git_actions=False, r67_started=False,
        r58='fully deferred; legacy eight B-task feasible-domain mismatch unresolved/disclosed'))
    print(json.dumps(dict(analysis=receipt, decision=decision)), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    args = parser.parse_args()
    summarize(args.root)


if __name__ == '__main__':
    main()
