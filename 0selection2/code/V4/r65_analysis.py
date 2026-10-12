"""One CPU delivery pass over completed, validation-locked R65 saved metrics."""

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .multitask_probe import file_hash
from .r59_experiment import atomic_json
from .r64_analysis import METRICS, write_csv


RUN_NAMES = {'A': 'A_constant_adamw_seed2', 'B': 'B_schedule_free_seed2'}
ROOT = Path(__file__).resolve().parent/'runs/R65_schedule_free_full18'
OPT_METRICS = ('loss', 'ce', 'kl', 'lr', 'grad_norm', 'clip_coefficient',
               'amp_retries', 'peak_memory_gib')


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text())


def metrics(value, prefix=''):
    return {key: value[prefix+key] for key in METRICS}


def metric_rows(model, split, epoch, value):
    common = dict(model=model, split=split, epoch=epoch)
    rows = [dict(common, scope='ALL', kind='aggregate',
                 native_correct=value.get('native_correct'),
                 native_total=value.get('native_total', sum(p['n'] for p in value['per_problem'].values())),
                 **metrics(value['macro'], 'macro_'))]
    rows += [dict(common, scope=name, kind='family', **metrics(entry, 'macro_'))
             for name, entry in value['families'].items()]
    rows += [dict(common, scope=name, kind='task', native_correct=entry.get('native_correct'),
                  native_total=entry['n'], **metrics(entry))
             for name, entry in value['per_problem'].items()]
    return rows


def historical_records(root):
    base = root.parent/'R64_pcgrad_full18'
    selected, provenance = {}, {}
    for name in ('R45A', 'R60B'):
        path = base/'reference'/(name+'_official.json')
        saved = read(path)
        record = dict(epoch=saved['selected_epoch'], val=saved['validation'], test=saved['test'])
        result_path = Path(saved['provenance']['validation']['path'])
        train = read(result_path)['best'].get('train')
        if train is not None:
            record['train'] = train
        selected[name] = record
        provenance[name] = dict(saved_reference_path=str(path), saved_reference_sha256=file_hash(path),
            original_provenance=saved['provenance'], historical_test_exposure=True,
            train_source=dict(path=str(result_path), sha256=file_hash(result_path), json_pointer='/best/train',
                              available=train is not None))
    test_path = base/'test_results.json'
    test, old_lock = read(test_path), read(base/'final_test_lock.json')
    if not test.get('complete') or test['lock'] != old_lock:
        raise ValueError('Historical R64 reference must be its saved completed locked TEST')
    for arm, name in {'A': 'A_mean_gradient_seed2', 'B': 'B_pcgrad_seed2'}.items():
        path = base/name/'result.json'
        result = read(path)
        model = 'R64'+arm
        selected[model] = dict(epoch=result['best']['epoch'], train=result['best']['train'],
                               val=result['best']['val'], test=test['models'][arm])
        provenance[model] = dict(result_path=str(path), result_sha256=file_hash(path),
            train_json_pointer='/best/train', validation_json_pointer='/best/val',
            test_path=str(test_path), test_sha256=file_hash(test_path),
            test_json_pointer='/models/'+arm, locked_official_arm=old_lock['official_arm'],
            historical_test_exposure=True)
    return selected, provenance


def line_panels(plt, curves, histories, updates):
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    for ax, metric in zip(axes.flat, METRICS):
        for arm, history in histories.items():
            for split, style in (('train', '--'), ('val', '-')):
                rows = [row for row in history if row.get(split) is not None]
                ax.plot([r['epoch'] for r in rows], [r[split]['macro']['macro_'+metric] for r in rows],
                        linestyle=style, label=arm+' '+split)
        ax.set(title=metric, xlabel='Epoch')
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    axes.flat[-1].axis('off')
    fig.tight_layout()
    fig.savefig(curves/'learning_curves.png', dpi=150)
    plt.close(fig)
    for filename, source, xkey in [('optimization.png',
            {arm: [dict(epoch=r['epoch'], **r['optimization']) for r in history]
             for arm, history in histories.items()}, 'epoch'),
            ('optimization_updates.png', updates, 'successful_update')]:
        fig, axes = plt.subplots(2, 4, figsize=(18, 8))
        for ax, metric in zip(axes.flat, OPT_METRICS):
            for arm, rows in source.items():
                ax.plot([r[xkey] for r in rows], [r[metric] for r in rows], label=arm,
                        linewidth=.8, alpha=.8)
            ax.set(title='effective LR' if metric == 'lr' else metric, xlabel=xkey)
            ax.grid(alpha=.2)
            ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(curves/filename, dpi=150)
        plt.close(fig)


def group_panels(plt, curves, rows, field):
    names = list(dict.fromkeys(row[field] for row in rows))
    columns = 3 if field == 'problem' else 2
    height = (len(names)+columns-1)//columns
    fig, axes = plt.subplots(height, columns, figsize=(5*columns, 3*height), squeeze=False)
    for ax, name in zip(axes.flat, names):
        for arm in RUN_NAMES:
            for split, style in (('train', '--'), ('val', '-')):
                points = [r for r in rows if r[field] == name and r['arm'] == arm and r['split'] == split]
                ax.plot([r['epoch'] for r in points], [r['top1'] for r in points],
                        linestyle=style, label=arm+' '+split)
        ax.set(title=name, xlabel='Epoch', ylabel='Native Top1')
        ax.grid(alpha=.2)
        ax.legend(fontsize=7)
    for ax in list(axes.flat)[len(names):]:
        ax.axis('off')
    fig.tight_layout()
    fig.savefig(curves/(field+'_top1.png'), dpi=150)
    plt.close(fig)


def selected_panels(plt, curves, selected):
    method_rows, diagnostics = [], []
    for model, record in selected.items():
        if not model.startswith('R65'):
            continue
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
                        ax.axhline(selector, color='#b84332', label='selector')
                        ax.legend()
                    ax.set_xticks(np.arange(len(methods)), methods, rotation=55, ha='right')
                    ax.set(title=model+' '+split+' '+problem, ylabel=title)
                    fig.tight_layout()
                    fig.savefig(directory/filename, dpi=150)
                    plt.close(fig)
    return method_rows, diagnostics


def markdown(root, selected, results, histories, decision, compute, train_terminal, test_terminal):
    aggregate = ['| Model | Split | Top1 % | Top2 % | Top3 % | CE | Mean Cost | vsSBS % | Actual Regret % |',
                 '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    side = ['| Model | Epoch | VAL Top1 % | TEST Top1 % | VAL CE | TEST CE |',
            '| --- | ---: | ---: | ---: | ---: | ---: |']
    for model, record in selected.items():
        val, test = record['val']['macro'], record['test']['macro']
        side.append(f"| {model} | {record['epoch']} | {100*val['macro_top1']:.4f} | {100*test['macro_top1']:.4f} | {val['macro_ce']:.6f} | {test['macro_ce']:.6f} |")
        for split in ('train', 'val', 'test'):
            if split not in record:
                continue
            m = metrics(record[split]['macro'], 'macro_')
            aggregate.append(f"| {model} | {split} | {100*m['top1']:.4f} | {100*m['top2']:.4f} | {100*m['top3']:.4f} | {m['ce']:.6f} | {m['mean_cost']:.6f} | {m['vs_sbs_pct']:.6f} | {m['actual_regret_pct']:.6f} |")
    comparison = '\n'.join(['# R65 Saved Native Metrics', '', *side, '', *aggregate, '',
        'CSV accuracies are fractions; Markdown accuracies and Top1 deltas are percent/percentage points. Cost and CE retain saved units.',
        'All18 task and ALL/TSP/CVRP/ATSP/MVRP metrics are in comparison.csv; deltas are in comparison_deltas.csv.',
        'R45A has no saved selected TRAIN evaluation. Historical TEST is exposed and reused without inference or reselection.', ''])
    (root/'comparison.md').write_text(comparison)
    official = decision['official_arm']
    locked_test = selected['R65'+official]['test']
    goal = 'Locked official TEST meets the numeric target; parent completion audit is still required.' if decision['locked_official_test_reaches70pct'] else 'The full18 native70% goal remains ACTIVE and unachieved.'
    lines = ['# R65 Locked Delivery', '',
        f"Official **R65{official}, epoch {selected['R65'+official]['epoch']}**, was frozen by original validation integer count before TEST. Locked official TEST: **{locked_test['native_correct']}/18000 = {100*locked_test['macro']['macro_top1']:.4f}%**. {goal}", '',
        'Fresh paired seed2; same initialization, original data, task/row and dropout streams. Both use the unchanged native selector/loss and arithmetic-mean task gradients, global clip1, real256/task (effective768; tail16/task), fixed40E/9600updates/7.2M presentations each. Common effective LR3e-4 after720 successful-update warmup; no controller or early stop. A is ordinary AdamW; B is official non-wrapper AdamWScheduleFree and deploys its already-saved averaged x weights.', '',
        '## Validation-Only Freeze', '',
        '| Arm | Selected Epoch | VAL Correct | Parameter Mode | SHA256 |',
        '| --- | ---: | ---: | --- | --- |']
    lock = read(root/'final_test_lock.json')
    for arm, r in lock['models'].items():
        lines.append(f"| {arm} | {r['epoch']} | {r['native_correct']}/18000 | {r['parameter_mode']} | `{r['sha256']}` |")
    lines += ['', 'Within-arm selection uses strict maximum original integer validation count with earliest exact tie; official-arm exact ties choose A. Cost, regret and TEST do not select weights, parameter mode, inference rule or official arm. Zero posterior per-task assembly/reselection.', '',
        f"B minus A selected validation: **{decision['validation_count_B_minus_A']:+d} correct, {decision['validation_top1_B_minus_A_pp']:+.6f}pp**. This is one seed and one fixed LR. Relative to R64, common LR/controller changes are additional differences; only the new A/B comparison identifies this optimizer-package contrast.", '',
        '## Saved Aggregate Metrics', '', *aggregate, '',
        'Native Top1/2/3 are class accuracies. vsSBS is100*(selected mean cost/SBS cost-1), negative is lower cost. Actual regret is the saved mean per-query relative excess over the full legal-pool oracle (%). ALL is the unweighted18-task macro; MVRP has15 tasks. Mixed-task mean cost is descriptive across heterogeneous task units. Detailed18-task/family/split records are in CSV, not duplicated here.', '',
        '## Historical Validation and TEST', '', *side, '',
        'R45A epoch38 and R60B epoch39 retain their historical official checkpoints and policies (R60B was cost-selected). R45A has no saved selected TRAIN evaluation. R64A/B reuse their saved locked tests; R64 official was B. Historical tests are exposed, not pristine independent evidence. Cross-version differences are descriptive. No reference inference, solver or prediction replay was performed; paths, JSON pointers and SHA256 provenance are in reference_provenance.json.', '',
        '## Final and Last-Five Validation', '',
        '| Arm | Best Top1 % | Final Top1 % | Last5 Mean % | Last5 Min % | Last5 Max % |',
        '| --- | ---: | ---: | ---: | ---: | ---: |']
    for arm, history in histories.items():
        last5 = [r['val']['macro']['macro_top1'] for r in history[-5:]]
        lines.append(f"| {arm} | {100*results[arm]['best']['val']['macro']['macro_top1']:.4f} | {100*results[arm]['final']['val']['macro']['macro_top1']:.4f} | {100*np.mean(last5):.4f} | {100*min(last5):.4f} | {100*max(last5):.4f} |")
    lines += ['', 'validation_summary.csv contains all seven metrics for best/final/last5 mean, population SD, minimum and maximum; last5_validation.csv has each epoch. These are temporal summaries, not independent-seed uncertainty.', '',
        '## Actual Execution and Inference', '',
        '| Arm | Updates | Presentations | AMP Retries | Arm Wall s | Update Wall s | Peak Allocated GiB |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for arm, c in compute['arms'].items():
        lines.append(f"| {arm} | {c['successful_updates']} | {c['presentations']} | {c['amp_retries']} | {c['wall_seconds']:.3f} | {c['logged_update_seconds']:.3f} | {c['peak_allocated_gib']:.3f} |")
    lines += ['',
        f"Training: tmux r65_train / Slurm{train_terminal['slurm_step']}, {train_terminal['state']} {train_terminal['exit_code']}; elapsed{train_terminal['elapsed_seconds']}s, tmux terminal confirmed. TEST: tmux r65_final_test / Slurm{test_terminal['slurm_step']}, {test_terminal['state']} {test_terminal['exit_code']}; elapsed{test_terminal['elapsed_seconds']}s, tmux terminal confirmed. Both use only assigned gpu03/index1 UUID GPU-cb626bdb-4319-2f5e-b91f-17fcaa44a83d in existing allocation1465.", '',
        f"Inference: `{lock['inference_rule']}`. One FP32 model, original18 tasks, all1000 TEST rows/task and full original legal pools, batch128 with tails retained, then one completed solver/query. Evaluation scores existing cost records; no solvers were executed. B uses averaged x; no x/y/z choice. TEST evaluation wall is{compute['test_wall_seconds']:.3f}s for both models, including existing stage work. No new singleton latency benchmark was authorized or run; saved per-split evaluation seconds are in compute_summary.json.", '',
        '## Artifacts and Remaining Work', '',
        'final_test_lock.json, final_test_started.json, test_results.json and training_terminal_receipt.json/test_terminal_receipt.json record the lock and actual terminal evidence. analysis.json, analysis_receipt.json, compute_summary.json and delivery_receipt.json record this one CPU analysis pass.',
        'comparison.csv/comparison_deltas.csv contain aggregate/family/full18 metrics; learning_curves.csv, task_curves.csv, family_curves.csv, optimization_curves.csv, update_curves.csv and task_optimization.csv retain curves. method_metrics.csv/task_diagnostics.csv retain saved single-method comparisons and selection distributions. curves/ includes loss/CE/KL/effectiveLR/gradnorm/clip/retry/memory panels, full18/family Top1 curves and per-task method/cost/distribution plots.',
        'R58 remains fully deferred; the legacy eight B-task feasible-domain mismatch remains unresolved and disclosed. Nightmare/100round/full18/single-solver/native70% scope is preserved, with no cost or margin veto on that target. Evidence is limited to this seed, LR and recipe.',
        'Parent primary TRAIN/validation-only Astra review/next plan, one fresh delivery review and one whole-version Git commit/push remain pending. No R66 or Git action was performed by this worker.', '']
    (root/'RESULTS.md').write_text('\n'.join(lines))


def summarize(root):
    root = Path(root).resolve()
    lock, test = read(root/'final_test_lock.json'), read(root/'test_results.json')
    train_terminal, test_terminal = read(root/'training_terminal_receipt.json'), read(root/'test_terminal_receipt.json')
    for terminal in (train_terminal, test_terminal):
        if terminal['state'] != 'COMPLETED' or terminal['exit_code'] != '0:0' or not terminal['tmux_terminal']:
            raise ValueError('Analysis requires actual successful terminal receipts')
    if lock.get('authorized') is not True or not test.get('complete') or test['lock'] != lock or set(test['models']) != {'A', 'B'}:
        raise ValueError('Only the completed authorized R65 frozen TEST may be reported')
    results = {arm: read(root/name/'result.json') for arm, name in RUN_NAMES.items()}
    histories = {arm: read(root/name/'history.json') for arm, name in RUN_NAMES.items()}
    with (root/'analysis_started.json').open('x') as stream:
        json.dump(dict(started_utc=now(), cpu_only=True, module=str(Path(__file__).resolve()),
                       one_pass=True, test_terminal_receipt=str(root/'test_terminal_receipt.json')), stream, indent=2)
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
        selected['R65'+arm] = dict(epoch=lock['models'][arm]['epoch'], train=result['best']['train'],
                                  val=result['best']['val'], test=test['models'][arm])
        for row in history:
            optimization.append(dict(arm=arm, epoch=row['epoch'], **row['optimization']))
            task_optimization += [dict(arm=arm, epoch=row['epoch'], problem=task, **value)
                                  for task, value in row['task_optimization'].items()]
            for split in ('train', 'val'):
                value = row.get(split)
                if value is None:
                    continue
                evaluations.append(dict(arm=arm, epoch=row['epoch'], split=split,
                    native_correct=value['native_correct'], native_total=value['native_total'], **metrics(value['macro'], 'macro_')))
                tasks += [dict(arm=arm, epoch=row['epoch'], split=split, problem=p, **metrics(m))
                          for p, m in value['per_problem'].items()]
                families += [dict(arm=arm, epoch=row['epoch'], split=split, family=p, **metrics(m, 'macro_'))
                             for p, m in value['families'].items()]
        updates[arm] = []
        with (root/RUN_NAMES[arm]/'updates.jsonl').open() as stream:
            for line in stream:
                row = json.loads(line)
                entry = {key: row[key] for key in ('epoch', 'successful_update', 'presentations', 'rows',
                    'loss', 'ce', 'kl', 'lr', 'optimizer_base_lr', 'grad_norm', 'clip_coefficient',
                    'seconds', 'peak_memory_gib', 'peak_reserved_gib', 'physical_used_gib')}
                updates[arm].append(dict(entry, amp_retries=row['retries']))
    for name, rows in [('learning_curves', evaluations), ('task_curves', tasks), ('family_curves', families),
                       ('optimization_curves', optimization), ('task_optimization', task_optimization),
                       ('update_curves', [dict(arm=arm, **row) for arm, rows in updates.items() for row in rows])]:
        write_csv(root/(name+'.csv'), rows)
    line_panels(plt, curves, histories, updates)
    group_panels(plt, curves, tasks, 'problem')
    group_panels(plt, curves, families, 'family')
    methods, diagnostics = selected_panels(plt, curves, selected)
    write_csv(root/'method_metrics.csv', methods)
    write_csv(root/'task_diagnostics.csv', diagnostics)
    historical, provenance = historical_records(root)
    selected.update(historical)
    comparison = [row for model, record in selected.items() for split in ('train', 'val', 'test')
                  if split in record for row in metric_rows(model, split, record['epoch'], record[split])]
    write_csv(root/'comparison.csv', comparison)
    write_csv(root/'selected_metrics.csv', [r for r in comparison if r['model'].startswith('R65') and r['kind'] == 'aggregate'])
    indexed = {(r['model'], r['split'], r['kind'], r['scope']): r for r in comparison}
    deltas = []
    pairs = [('R65B', 'R65A')]+[(a, b) for a in ('R65A', 'R65B') for b in historical]
    for a, b in pairs:
        for split in ('val', 'test'):
            for scope in ('ALL', 'TSP', 'CVRP', 'ATSP', 'MVRP'):
                kind = 'aggregate' if scope == 'ALL' else 'family'
                left, right = indexed[a, split, kind, scope], indexed[b, split, kind, scope]
                deltas.append(dict(comparison=a+' minus '+b, split=split, scope=scope,
                    **{key+('_delta_pp' if key.startswith('top') else '_delta'):
                       (left[key]-right[key])*(100 if key.startswith('top') else 1) for key in METRICS}))
    write_csv(root/'comparison_deltas.csv', deltas)
    validation, last5_rows = [], []
    for arm, history in histories.items():
        for name, value in [('best', results[arm]['best']['val']), ('final', results[arm]['final']['val'])]:
            validation.append(dict(arm=arm, summary=name, native_correct=value['native_correct'],
                                   **metrics(value['macro'], 'macro_')))
        last5 = [r['val'] for r in history[-5:]]
        for row in history[-5:]:
            last5_rows.append(dict(arm=arm, epoch=row['epoch'], native_correct=row['val']['native_correct'],
                                  **metrics(row['val']['macro'], 'macro_')))
        for name, function in [('last5_mean', np.mean), ('last5_population_sd', np.std),
                               ('last5_min', np.min), ('last5_max', np.max)]:
            validation.append(dict(arm=arm, summary=name,
                native_correct=float(function([r['native_correct'] for r in last5])),
                **{key: float(function([r['macro']['macro_'+key] for r in last5])) for key in METRICS}))
    write_csv(root/'validation_summary.csv', validation)
    write_csv(root/'last5_validation.csv', last5_rows)
    official = lock['official_arm']
    a, b = (lock['models'][arm]['native_correct'] for arm in ('A', 'B'))
    official_test = test['models'][official]
    decision = dict(official_arm=official, decision_source='published integer original validation counts only; tie A',
        validation_count_B_minus_A=b-a, validation_top1_B_minus_A_pp=100*(b-a)/18000,
        schedule_free_exceeds_control_validation=b > a, locked_official_test_correct=official_test['native_correct'],
        locked_official_test_total=official_test['native_total'],
        locked_official_test_top1=official_test['native_correct']/official_test['native_total'],
        locked_official_test_reaches70pct=10*official_test['native_correct'] >= 7*official_test['native_total'],
        goal_completion_audit_pending=True, one_seed_one_lr=True, cost_regret_or_margin_veto=False,
        test_reselection=False, posterior_per_task_assembly=False)
    compute = dict(arms={}, training_terminal=train_terminal, test_terminal=test_terminal,
                   test_wall_seconds=test['wall_seconds'], new_pilots_or_sizing=0,
                   inference_scope='FP32 eval batch128, original full18/fulllegalpool; no singleton timing or solver execution')
    for arm, result in results.items():
        rows = updates[arm]
        compute['arms'][arm] = dict(epochs_completed=result['epochs_completed'], successful_updates=result['successful_updates'],
            presentations=result['presentations'], amp_retries=result['amp_retries'], wall_seconds=result['wall_seconds'],
            actual_execution=result['actual_execution'], logged_update_seconds=sum(r['seconds'] for r in rows),
            peak_allocated_gib=max(r['peak_memory_gib'] for r in rows), peak_reserved_gib=max(r['peak_reserved_gib'] for r in rows),
            logged_whole_device_peak_physical_gib=max(r['physical_used_gib'] for r in rows),
            selected_evaluation_seconds={split: selected['R65'+arm][split].get('seconds') for split in ('train', 'val', 'test')},
            selected_validation_count_delta=result['best']['selected_validation_count_delta'])
    atomic_json(root/'reference_provenance.json', dict(recorded_utc=now(), references=provenance,
        historical_test_exposure=True, reference_inference_reruns=False, solver_or_metric_replays=False,
        transformation='Structured saved-metric extraction only, after frozen R65 TEST terminal success'))
    atomic_json(root/'compute_summary.json', compute)
    markdown(root, selected, results, histories, decision, compute, train_terminal, test_terminal)
    report = dict(recorded_utc=now(), decision=decision, selected={model: dict(epoch=r['epoch'],
        **{split: r[split]['macro'] for split in ('train', 'val', 'test') if split in r}) for model, r in selected.items()},
        actual_compute=compute, reference_provenance='reference_provenance.json', locked_test_reported=True,
        historical_test_exposure_disclosed=True, legacy_B_semantics='unresolved; R58 fully deferred',
        plan_review_scope='TRAIN/validation only; later locked TEST cannot reselect',
        new_module_only=True, frozen_sources_or_weights_edited=False)
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
        parent_whole_version_commit_push_pending=True, worker_git_actions=False, r66_started=False,
        r58='fully deferred; legacy eight B-task feasible-domain mismatch unresolved/disclosed'))
    print(json.dumps(dict(analysis=receipt, decision=decision)))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    args = parser.parse_args()
    summarize(args.root)


if __name__ == '__main__':
    main()
