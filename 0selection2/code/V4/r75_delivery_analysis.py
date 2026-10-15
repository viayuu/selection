"""Bounded CPU-only R75 delivery from saved, completed and VAL-locked metrics.

Public API: analyze(root). CLI: python -m code.V4.r75_delivery_analysis --root ROOT.
No trainer imports, model/checkpoint loads, data/cache reads or prediction replay.
"""

import argparse
import csv
import hashlib
import io
import json
import math
import os
import statistics
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent / 'runs/R75_large_complete_labels'
# Local identities avoid the registry's dataset discovery at import time.
PROBLEMS = ('TSP', 'CVRP', 'ATSP', 'OVRP', 'VRPB', 'VRPL', 'VRPTW',
            'OVRPTW', 'OVRPB', 'OVRPL', 'VRPBL', 'VRPBTW', 'VRPLTW',
            'OVRPBL', 'OVRPBTW', 'OVRPLTW', 'VRPBLTW', 'OVRPBLTW')
GROUPS = {'ALL': PROBLEMS, 'TSP': ('TSP',), 'CVRP': ('CVRP',),
          'ATSP': ('ATSP',), 'MVRP': PROBLEMS[3:]}
METRICS = ('top1', 'top2', 'top3', 'ce', 'mean_cost', 'vs_sbs_pct',
           'vs_oracle_pct', 'actual_regret_pct')
OPT_METRICS = ('loss', 'ce', 'kl', 'grad_norm', 'lr', 'seconds',
               'amp_retries', 'peak_memory_gib')
CONTRACT = (
    'Required saved artifacts: protocol.json (seed=2, groups=1, sizes, max_epochs); '
    'single_seed2/history.json (epoch rows with successful_updates, presentations, '
    'optimization, train or null, val); single_seed2/best_eval.json (selected row '
    'with full train and val); training_complete.json (complete=true, epochs, '
    'successful_updates, presentations, selected_epoch, selected_val_correct); '
    'final_test_lock.json (checkpoint, checkpoint_sha256, epoch, val_correct, '
    'val_total, groups=1, selected_without_test=true, no_per_problem_assembly=true); '
    'test_results.json (model=R75, epoch, checkpoint, checkpoint_sha256, '
    'result=full18 aggregate, complete=true). Aggregates require per_problem '
    '(n, native_correct, pool and all8 metrics), macro, families, native_correct, '
    'native_total, seconds. No missing metrics are inferred.'
)


def require(condition, message):
    if not condition:
        raise ValueError(message + '\n' + CONTRACT)


def utc():
    return datetime.now(timezone.utc).isoformat()


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def integer(value, minimum=0):
    return type(value) is int and value >= minimum


def close(left, right):
    return math.isclose(left, right, rel_tol=1e-7, abs_tol=1e-8)


def metrics(value, prefix=''):
    """Shared table API; preserve saved values, including unavailable fields."""
    return {key: value.get(prefix + key) for key in METRICS}


def counts(value, tasks):
    entries = [value['per_problem'][task] for task in tasks]
    total = sum(entry['n'] for entry in entries)
    correct = [entry.get('native_correct') for entry in entries]
    return (sum(correct) if all(integer(n) for n in correct) else None), total


def metric_rows(model, split, epoch, value, source_path='', source_pointer=''):
    """Flatten a saved evaluation into ALL, four families and all18 tasks."""
    common = dict(model=model, split=split, epoch=epoch, source_path=source_path,
                  source_pointer=source_pointer, historical=model != 'R75')
    rows = []
    for scope, tasks in GROUPS.items():
        correct, total = counts(value, tasks)
        if scope == 'ALL':
            correct = value.get('native_correct', correct)
            total = value.get('native_total', total)
        entry = value['macro'] if scope == 'ALL' else value['families'][scope]
        rows.append(dict(common, kind='aggregate' if scope == 'ALL' else 'family',
                         scope=scope, native_correct=correct, native_total=total,
                         native_top1=correct / total if correct is not None else None,
                         **metrics(entry, 'macro_')))
    for task in PROBLEMS:
        entry = value['per_problem'][task]
        correct = entry.get('native_correct')
        rows.append(dict(common, kind='task', scope=task, native_correct=correct,
                         native_total=entry['n'],
                         native_top1=correct / entry['n'] if correct is not None else None,
                         **metrics(entry)))
    return rows


def validate_aggregate(value, label, sizes):
    require(isinstance(value, dict), label + ': missing aggregate')
    per = value.get('per_problem', {})
    require(set(per) == set(PROBLEMS), label + ': requires exactly all18 tasks')
    require(set(value.get('families', {})) == set(GROUPS) - {'ALL'},
            label + ': requires TSP/CVRP/ATSP/MVRP families')
    for task, entry in per.items():
        require(integer(entry.get('n'), 1) and entry['n'] == sizes[task],
                label + '/' + task + ': incomplete saved row coverage')
        require(all(finite(entry.get(key)) for key in METRICS),
                label + '/' + task + ': missing/nonfinite metrics')
        require(0 <= entry['top1'] <= entry['top2'] <= entry['top3'] <= 1,
                label + '/' + task + ': invalid accuracy ordering')
        require(integer(entry.get('native_correct')) and
                entry['native_correct'] <= entry['n'] and
                close(entry['native_correct'] / entry['n'], entry['top1']),
                label + '/' + task + ': native count disagrees with Top1')
        pool = entry.get('pool')
        require(isinstance(pool, list) and len(pool) > 0 and
                all(isinstance(name, str) for name in pool) and len(set(pool)) == len(pool),
                label + '/' + task + ': missing/ambiguous legal pool')
    for scope, tasks in GROUPS.items():
        saved = value.get('macro', {}) if scope == 'ALL' else value['families'][scope]
        for key in METRICS:
            expected = statistics.fmean(per[task][key] for task in tasks)
            require(finite(saved.get('macro_' + key)) and close(saved['macro_' + key], expected),
                    label + '/' + scope + ': saved macro disagrees with task metrics: ' + key)
    correct, total = counts(value, PROBLEMS)
    require(value.get('native_correct') == correct and value.get('native_total') == total,
            label + ': aggregate native counts disagree with task counts')
    require(finite(value.get('seconds')) and value['seconds'] >= 0,
            label + ': missing evaluation seconds')


def write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, sort_keys=True) if isinstance(value, (dict, list))
                             else value for key, value in row.items()})


def historical_rows(root, read_source, warnings):
    """Reuse saved reporting CSVs, with no historical inference or recount."""
    rows = []
    sources = ((root.parent / 'R65_schedule_free_full18/comparison.csv',
                {'R65B': 'R65B', 'R45A': 'R45A', 'R60B': 'R60B'}),
               (root.parent / 'R73_expansion_teacher_training/comparison.csv',
                {'R73B_warm_started_shared_student': 'R73B'}))
    for path, names in sources:
        if not path.is_file():
            warnings.append('Historical saved CSV unavailable: ' + str(path))
            continue
        for line, saved in enumerate(csv.DictReader(io.StringIO(read_source(path).decode('utf-8-sig'))), 2):
            if saved.get('model') not in names:
                continue
            row = dict(model=names[saved['model']], split=saved['split'], epoch=saved.get('epoch'),
                       kind=saved['kind'], scope=saved['scope'], historical=True,
                       source_path=str(path), source_pointer='CSV record ' + str(line - 1))
            for key in (*METRICS, 'native_correct', 'native_total'):
                raw = saved.get(key)
                if key == 'ce' and not raw:
                    raw = saved.get('native_utility_CE')
                row[key] = float(raw) if raw not in (None, '') else None
                require(row[key] is None or finite(row[key]), 'Nonfinite historical CSV metric')
            row['native_top1'] = (row['native_correct'] / row['native_total']
                                  if row['native_correct'] is not None and row['native_total'] else None)
            rows.append(row)
    if not any(row['model'] == 'R65B' and row['split'] == 'test' for row in rows):
        warnings.append('R65B saved TEST comparison unavailable; no reference rerun is required or performed.')
    return rows


def figures(directory, history, learning):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(4, 4, figsize=(20, 15))
    for ax, metric in zip(axes.flat, METRICS):
        for split, color in (('val', '#197e6b'), ('train', '#bb583b')):
            points = sorted((row for row in learning if row['scope'] == 'ALL' and row['split'] == split),
                            key=lambda row: row['epoch'])
            ax.plot([row['epoch'] for row in points], [row[metric] for row in points],
                    color=color, marker='.', linestyle='-' if split == 'val' else 'None', label=split.upper())
        ax.set(title='Task macro ' + metric, xlabel='Epoch')
        ax.legend(fontsize=8)
    for ax, metric in zip(list(axes.flat)[8:], OPT_METRICS):
        ax.plot([row['epoch'] for row in history], [row['optimization'][metric] for row in history],
                color='#355e91')
        ax.set(title='Optimization ' + metric, xlabel='Epoch')
    for ax in axes.flat:
        ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(directory / 'learning_curves.png', dpi=130)
    plt.close(fig)

    fig, axes = plt.subplots(6, 4, figsize=(24, 22))
    scopes = [('task', task) for task in PROBLEMS] + [
        ('aggregate' if name == 'ALL' else 'family', name) for name in GROUPS]
    for ax, (kind, scope) in zip(axes.flat, scopes):
        regret = ax.twinx()
        for split, color in (('val', '#197e6b'), ('train', '#bb583b')):
            points = sorted((row for row in learning if row['kind'] == kind and row['scope'] == scope
                             and row['split'] == split), key=lambda row: row['epoch'])
            x = [row['epoch'] for row in points]
            ax.plot(x, [100 * row['top1'] for row in points], color=color, marker='.',
                    linestyle='-' if split == 'val' else 'None', label=split + ' Top1')
            regret.plot(x, [row['actual_regret_pct'] for row in points], color=color, alpha=.55,
                        marker='x', linestyle=':' if split == 'val' else 'None', label=split + ' regret')
        ax.set(title=scope + ' (' + kind + ')', xlabel='Epoch', ylabel='Task-macro Top1 (%)')
        regret.set_ylabel('Actual regret (%)')
        ax.grid(alpha=.2)
        handles, labels = ax.get_legend_handles_labels()
        other, names = regret.get_legend_handles_labels()
        ax.legend(handles + other, labels + names, fontsize=6, ncol=2)
    axes.flat[-1].axis('off')
    fig.tight_layout()
    fig.savefig(directory / 'per_task_family_curves.png', dpi=130)
    plt.close(fig)


def display(value, percent=False):
    return 'N/A' if value is None else f'{value * (100 if percent else 1):.6f}'


def report(root, comparison, stages, best, lock, completion, warnings):
    lines = ['# R75 Single-Group Full18 Results', '',
        f'Seed2; selected epoch {best["epoch"]}, frozen solely by full18 native validation count '
        f'{lock["val_correct"]}/{lock["val_total"]}. Checkpoint: `{lock["checkpoint"]}`; '
        f'declared SHA256: `{lock["checkpoint_sha256"]}`. Checkpoint bytes were not opened by analysis.', '',
        'One completed locked TEST is reported; analysis does not run inference or select again. '
        'Historical comparisons are descriptive, not matched data-only causal evidence. '
        'Legacy eight B-task semantics remain deferred/unfixed. The source TSP/CVRP snapshot '
        'misses mainly large-size tails. The prior goal remains paused; no 70% threshold claim '
        'or goal resumption is made. R74 remains terminated.', '',
        '## Selected TRAIN / VAL / TEST', '',
        '| Model | Split | Epoch | Native correct/total | Native Top1 % | Task macro Top1 % | Top2 % | Top3 % | CE | Mean cost | vs SBS % | vs oracle % | Actual regret % |',
        '|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for row in comparison:
        if row['kind'] != 'aggregate':
            continue
        correct, total = row.get('native_correct'), row.get('native_total')
        count = f'{correct:g}/{total:g}' if correct is not None and total else 'N/A'
        values = [display(row.get('native_top1'), True)] + [display(row.get(key), key.startswith('top')) for key in METRICS]
        lines.append('| ' + ' | '.join([row['model'], row['split'], str(row['epoch']), count, *values]) + ' |')
    lines += ['', 'Native Top1 uses integer correct/total; task macros weight each of18 tasks equally. '
        'These differ for unequal TRAIN sizes. MVRP is a15-task macro. Accuracies in CSV are fractions; '
        'regret/vs-SBS/vs-oracle retain saved percent units. Mixed-task mean cost has heterogeneous units.', '',
        '## Full18 And Families', '',
        '| Scope | Kind | TRAIN Top1 % | VAL Top1 % | TEST native correct/total | TEST Top1 % | TEST Top2 % | TEST Top3 % | TEST CE | TEST mean cost | TEST vs SBS % | TEST vs oracle % | TEST actual regret % |',
        '|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    index = {(row['split'], row['kind'], row['scope']): row for row in comparison if row['model'] == 'R75'}
    for row in comparison:
        if row['model'] != 'R75' or row['split'] != 'test':
            continue
        train = index[('train', row['kind'], row['scope'])]
        val = index[('val', row['kind'], row['scope'])]
        values = [display(row[key], key.startswith('top')) for key in METRICS]
        lines.append('| ' + ' | '.join([row['scope'], row['kind'], display(train['top1'], True),
            display(val['top1'], True), f'{row["native_correct"]}/{row["native_total"]}', *values]) + ' |')
    lines += ['', '## Best / Final / Last Five', '',
        '| Summary | Split | Actual epochs | Native Top1 % | Macro Top1 % | Actual regret % |',
        '|---|---|---|---:|---:|---:|']
    for row in stages:
        if row['kind'] == 'aggregate':
            lines.append('| ' + ' | '.join([row['summary'], row['split'], str(row['actual_epochs']),
                display(row.get('native_top1'), True), display(row['top1'], True),
                display(row['actual_regret_pct'])]) + ' |')
    lines += ['', 'Last-five statistics describe temporal variation, not independent-seed uncertainty. '
        'Sparse TRAIN points are not interpolated; a last-five TRAIN summary requires five actual measurements.', '',
        '## Recorded Compute', '', '```json', json.dumps(completion, indent=2, sort_keys=True), '```', '',
        'Training compute above is the saved completion summary, not a new benchmark. '
        'Saved evaluation seconds and optimizer metrics are exported separately. '
        'No solver executions, reference reruns, model/data/cache loads or prediction recounts were performed.', '',
        'Instance correction/harm is unavailable from scalar metrics and is not inferred. '
        'Saved per-method comparisons, best-method metrics and arm distributions are retained in '
        'method_metrics.csv/task_diagnostics.csv. Missing historical fields stay N/A.', '',
        'Sources, byte hashes, pointers and CPU analysis timings: analysis_receipt.json. '
        'All18 task and ALL/TSP/CVRP/ATSP/MVRP rows: comparison.csv. '
        'All saved epoch curves: learning_curves.csv/optimization_curves.csv. '
        'Figures: learning_curves.png and per_task_family_curves.png.', '']
    lines += ['Availability: ' + warning for warning in warnings]
    (root / 'RESULTS.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


def analyze(root):
    """Publish two figures and tables after completed locked full18 evaluation."""
    started, started_utc = time.monotonic(), utc()
    root = Path(root).resolve()
    sources, warnings = {}, []
    implementation = Path(__file__).resolve()
    implementation_bytes = implementation.read_bytes()
    implementation_source = dict(path=str(implementation),
        sha256=hashlib.sha256(implementation_bytes).hexdigest(), bytes=len(implementation_bytes))

    def read_source(path):
        path = Path(path).resolve()
        require(path.suffix in ('.json', '.csv'), 'Metrics analysis only reads saved JSON/CSV')
        require(path.is_file(), 'Missing required saved artifact: ' + str(path))
        raw = path.read_bytes()
        sources[str(path)] = dict(path=str(path), sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw))
        return raw

    def read(name):
        return json.loads(read_source(root / name))

    protocol = read('protocol.json')
    history = read('single_seed2/history.json')
    best = read('single_seed2/best_eval.json')
    completion = read('training_complete.json')
    lock = read('final_test_lock.json')
    test = read('test_results.json')
    read_seconds = time.monotonic() - started
    require(completion.get('complete') is True and test.get('complete') is True,
            'Refusing performance publication before completed TRAIN and TEST')
    require(protocol.get('groups') == 1 and protocol.get('seed') == 2 and
            lock.get('groups') == 1 and lock.get('selected_without_test') is True and
            lock.get('no_per_problem_assembly') is True and 'models' not in lock,
            'Require one unique validation-selected complete checkpoint, not a model assembly')
    require(isinstance(history, list) and len(history) > 0, 'Missing saved epoch history')
    require([row['epoch'] for row in history] == list(range(1, protocol['max_epochs'] + 1)),
            'History must cover the completed fixed epoch schedule')
    sizes = protocol['sizes']
    require(set(sizes) == set(PROBLEMS) and all(integer(n, 1) for n in sizes.values()),
            'Protocol sizes must describe actual full TRAIN coverage')
    heldout = dict.fromkeys(PROBLEMS, 1000)
    previous_updates = previous_presentations = 0
    for row in history:
        validate_aggregate(row['val'], 'history VAL epoch ' + str(row['epoch']), heldout)
        if row.get('train') is not None:
            validate_aggregate(row['train'], 'history TRAIN epoch ' + str(row['epoch']), sizes)
        require(integer(row.get('successful_updates'), previous_updates + 1) and
                integer(row.get('presentations'), previous_presentations + 1), 'Nonmonotone history counters')
        require(all(finite(row.get('optimization', {}).get(key)) for key in OPT_METRICS),
                'Missing saved optimizer metrics')
        previous_updates, previous_presentations = row['successful_updates'], row['presentations']
    selected = max(history, key=lambda row: row['val']['native_correct'])
    require(best['epoch'] == selected['epoch'] == lock.get('epoch') == test.get('epoch') and
            best['val'] == selected['val'] and lock.get('val_correct') == best['val']['native_correct'] and
            lock.get('val_total') == best['val']['native_total'], 'Selected epoch/count violates earliest maximum VAL lock')
    require(test.get('model') == 'R75' and test.get('checkpoint') == lock.get('checkpoint') and
            isinstance(lock.get('checkpoint'), str) and bool(lock['checkpoint']) and
            isinstance(lock.get('checkpoint_sha256'), str) and len(lock['checkpoint_sha256']) == 64 and
            all(c in '0123456789abcdef' for c in lock['checkpoint_sha256'].lower()) and
            test.get('checkpoint_sha256') == lock['checkpoint_sha256'], 'TEST identity differs from unique VAL lock')
    require(completion.get('epochs') == history[-1]['epoch'] and
            completion.get('successful_updates') == previous_updates and
            completion.get('presentations') == previous_presentations and
            completion.get('selected_epoch') == best['epoch'] and
            completion.get('selected_val_correct') == lock['val_correct'], 'Completion summary disagrees with history/selection')
    validate_aggregate(best.get('train'), 'selected full TRAIN', sizes)
    validate_aggregate(test.get('result'), 'locked full TEST', heldout)
    for row in history:
        for split in ('train', 'val'):
            if row.get(split) is not None:
                require(all(row[split]['per_problem'][p]['pool'] == test['result']['per_problem'][p]['pool']
                            for p in PROBLEMS), 'Saved history and TEST legal pools disagree')
    require(all(best['train']['per_problem'][p]['pool'] == test['result']['per_problem'][p]['pool']
                for p in PROBLEMS), 'Selected TRAIN and TEST legal pools disagree')
    validated_seconds = time.monotonic() - started - read_seconds

    def rows(split, row, name, pointer):
        return metric_rows('R75', split, row['epoch'], row[split], str(root / name), pointer)

    comparison = rows('train', best, 'single_seed2/best_eval.json', '/train')
    comparison += rows('val', best, 'single_seed2/best_eval.json', '/val')
    comparison += metric_rows('R75', 'test', best['epoch'], test['result'],
                              str(root / 'test_results.json'), '/result')
    comparison += historical_rows(root, read_source, warnings)
    learning, optimization, stages = [], [], []
    for index, row in enumerate(history):
        optimization.append(dict(epoch=row['epoch'], successful_updates=row['successful_updates'],
                                 presentations=row['presentations'], **row['optimization']))
        for split in ('train', 'val'):
            if row.get(split) is not None:
                learning += [dict(entry, successful_updates=row['successful_updates'],
                                  presentations=row['presentations'], measurement='actual epoch')
                             for entry in rows(split, row, 'single_seed2/history.json', f'/{index}/{split}')]
    if selected.get('train') is None:
        learning += [dict(entry, successful_updates=best['successful_updates'],
                          presentations=best['presentations'], measurement='actual selected full TRAIN')
                     for entry in rows('train', best, 'single_seed2/best_eval.json', '/train')]
    for name, row, source, pointer in (('best', best, 'single_seed2/best_eval.json', ''),
                                      ('final', history[-1], 'single_seed2/history.json', '/' + str(len(history)-1))):
        for split in ('train', 'val'):
            if row.get(split) is not None:
                stages += [dict(entry, summary=name, actual_epochs=str(row['epoch']))
                           for entry in rows(split, row, source, pointer + '/' + split)]
    tail = history[-5:]
    for split in ('train', 'val'):
        measured = [row for row in tail if row.get(split) is not None]
        if len(tail) != 5 or len(measured) != 5:
            warnings.append('No last-five ' + split.upper() + ' statistics: fewer than five actual epoch measurements.')
            continue
        entries = [metric_rows('R75', split, row['epoch'], row[split]) for row in measured]
        for index, entry in enumerate(entries[0]):
            for label, function in (('mean', statistics.fmean), ('population_sd', statistics.pstdev),
                                    ('min', min), ('max', max)):
                summary = dict(model='R75', split=split, kind=entry['kind'], scope=entry['scope'],
                               summary='last5_' + label, actual_epochs=','.join(str(r['epoch']) for r in measured))
                summary.update({key: function([values[index][key] for values in entries])
                                for key in (*METRICS, 'native_top1')})
                stages.append(summary)
    method_rows, diagnostics = [], []
    for split, value in (('train', best['train']), ('val', best['val']), ('test', test['result'])):
        for problem, entry in value['per_problem'].items():
            diagnostics.append(dict(split=split, problem=problem, **{key: entry.get(key) for key in
                ('best_top1', 'best_top2', 'best_top3', 'sbs_name', 'sbs_cost', 'oracle_cost',
                 'top1_exceeds_methods', 'top1_not_exceeds_methods', 'cost_exceeds_methods', 'cost_not_exceeds_methods')}))
            picks = entry.get('pick_dist', [])
            for index, method in enumerate(entry['pool']):
                method_rows.append(dict(split=split, problem=problem, method=method,
                    method_top1=entry.get('method_top1', {}).get(method),
                    method_mean_cost=entry.get('method_mean_cost', {}).get(method),
                    pick_fraction=picks[index] if index < len(picks) else None,
                    selector_top1=entry['top1'], selector_mean_cost=entry['mean_cost']))
    for row in comparison + learning + stages:
        if row.get('source_path'):
            row['source_sha256'] = sources[row['source_path']]['sha256']
    compute = dict(saved_training_complete=completion,
                   evaluation_seconds={split: value['seconds'] for split, value in
                       (('selected_train', best['train']), ('selected_val', best['val']), ('locked_test', test['result']))},
                   timings_are_not_single_query_latency=True, no_new_benchmarks=True)
    built_seconds = time.monotonic() - started - read_seconds - validated_seconds
    with tempfile.TemporaryDirectory(prefix='.r75_delivery_analysis-', dir=root) as temporary:
        output = Path(temporary)
        for filename, entries in (('comparison.csv', comparison), ('learning_curves.csv', learning),
                                  ('optimization_curves.csv', optimization), ('stage_summary.csv', stages),
                                  ('method_metrics.csv', method_rows), ('task_diagnostics.csv', diagnostics)):
            write_csv(output / filename, entries)
        (output / 'compute_summary.json').write_text(json.dumps(compute, indent=2, sort_keys=True) + '\n', encoding='utf-8')
        plot_started = time.monotonic()
        figures(output, history, learning)
        plot_seconds = time.monotonic() - plot_started
        report(output, comparison, stages, best, lock, completion, warnings)
        # Prevent publishing a mixture of artifacts changed by another writer.
        for source in (*sources.values(), implementation_source):
            require(hashlib.sha256(Path(source['path']).read_bytes()).hexdigest() == source['sha256'],
                    'Saved source changed during analysis: ' + source['path'])
        artifacts = [path.name for path in sorted(output.iterdir())]
        for filename in artifacts:
            os.replace(output / filename, root / filename)
        receipt = dict(complete=True, cpu_only=True, started_utc=started_utc, completed_utc=utc(),
            root=str(root), selected_epoch=best['epoch'], checkpoint=lock['checkpoint'],
            checkpoint_sha256=lock['checkpoint_sha256'], checkpoint_hash_verified_against_bytes=False,
            val_native_correct=lock['val_correct'], test_native_correct=test['result']['native_correct'],
            test_native_total=test['result']['native_total'], sources=list(sources.values()),
            implementation_source=implementation_source,
            timings=dict(read_seconds=read_seconds, validate_seconds=validated_seconds,
                         build_seconds=built_seconds, figures_seconds=plot_seconds,
                         total_seconds=time.monotonic()-started), artifacts=artifacts,
            figures=2, reference_inference=False, prediction_replay=False, model_or_data_files_opened=False,
            historical_comparisons_matched=False, correction_harm='unavailable; no prediction files read',
            old_goal='paused', legacy_eight_B_semantics='deferred/unfixed', warnings=warnings)
        (output / 'analysis_receipt.json').write_text(json.dumps(receipt, indent=2, sort_keys=True) + '\n', encoding='utf-8')
        os.replace(output / 'analysis_receipt.json', root / 'analysis_receipt.json')
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    args = parser.parse_args()
    analyze(args.root)


if __name__ == '__main__':
    main()
