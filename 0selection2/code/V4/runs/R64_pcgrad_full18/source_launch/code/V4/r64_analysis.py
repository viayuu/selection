"""Explicit R64 reporting from saved TRAIN/validation and later locked TEST artifacts."""

import csv
import json
from pathlib import Path

import numpy as np

from .r59_experiment import atomic_json


RUN_NAMES = {'A': 'A_mean_gradient_seed2', 'B': 'B_pcgrad_seed2'}
METRICS = ('top1', 'top2', 'top3', 'ce', 'mean_cost', 'actual_regret_pct', 'vs_sbs_pct')


def write_csv(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def summarize(root, reference_tests=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    root = Path(root)
    curves = root/'curves'
    curves.mkdir(exist_ok=True)
    results, histories = {}, {}
    for arm, name in RUN_NAMES.items():
        directory = root/name
        if (directory/'result.json').exists():
            results[arm] = json.loads((directory/'result.json').read_text())
        if (directory/'history.json').exists():
            histories[arm] = json.loads((directory/'history.json').read_text())
    evaluation_rows, task_rows, family_rows, optimization_rows = [], [], [], []
    for arm, history in histories.items():
        for row in history:
            optimization_rows.append(dict(arm=arm, epoch=row['epoch'], **row['optimization']))
            for split in ('train', 'val'):
                value = row.get(split)
                if value is None:
                    continue
                evaluation_rows.append(dict(arm=arm, epoch=row['epoch'], split=split,
                    native_correct=value['native_correct'], native_total=value['native_total'], **value['macro']))
                for problem, metrics in value['per_problem'].items():
                    task_rows.append(dict(arm=arm, epoch=row['epoch'], split=split, problem=problem,
                                          **{key: metrics[key] for key in METRICS}))
                for family, metrics in value['families'].items():
                    family_rows.append(dict(arm=arm, epoch=row['epoch'], split=split, family=family,
                                            **{key: metrics['macro_'+key] for key in METRICS}))
    write_csv(root/'learning_curves.csv', evaluation_rows)
    write_csv(root/'task_curves.csv', task_rows)
    write_csv(root/'family_curves.csv', family_rows)
    write_csv(root/'optimization_curves.csv', optimization_rows)

    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    for ax, metric in zip(axes.flat, METRICS):
        for arm, history in histories.items():
            for split, style in (('train', '--'), ('val', '-')):
                rows = [r for r in history if r.get(split) is not None]
                ax.plot([r['epoch'] for r in rows], [r[split]['macro']['macro_'+metric] for r in rows],
                        linestyle=style, label=arm+' '+split)
        ax.set(title=metric, xlabel='Epoch')
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    axes.flat[-1].axis('off')
    fig.tight_layout()
    fig.savefig(curves/'learning_curves.png', dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    for ax, metric in zip(axes.flat, ('loss', 'ce', 'kl', 'lr', 'grad_norm',
                                     'conflict_frequency', 'projection_mean_change_norm', 'peak_memory_gib')):
        for arm, history in histories.items():
            ax.plot([r['epoch'] for r in history], [r['optimization'][metric] for r in history], label=arm)
        ax.set(title=metric, xlabel='Epoch')
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(curves/'optimization.png', dpi=150)
    plt.close(fig)
    for field, rows in (('family', family_rows), ('problem', task_rows)):
        names = list(dict.fromkeys(r[field] for r in rows))
        if not names:
            continue
        columns = 3 if field == 'problem' else 2
        height = (len(names)+columns-1)//columns
        fig, axes = plt.subplots(height, columns, figsize=(5*columns, 3*height), squeeze=False)
        for ax, name in zip(axes.flat, names):
            for arm in histories:
                for split, style in (('train', '--'), ('val', '-')):
                    points = [r for r in rows if r[field] == name and r['arm'] == arm and r['split'] == split]
                    ax.plot([r['epoch'] for r in points], [r['top1'] for r in points], linestyle=style, label=arm+' '+split)
            ax.set(title=name, xlabel='Epoch', ylabel='Native Top1')
            ax.grid(alpha=.2)
            ax.legend(fontsize=7)
        for ax in list(axes.flat)[len(names):]:
            ax.axis('off')
        fig.tight_layout()
        fig.savefig(curves/(field+'_top1.png'), dpi=150)
        plt.close(fig)

    selected = {arm: dict(train=r['best']['train'], val=r['best']['val']) for arm, r in results.items()}
    test_path, lock = root/'test_results.json', None
    if test_path.exists():
        test = json.loads(test_path.read_text())
        lock = json.loads((root/'final_test_lock.json').read_text())
        if not test.get('complete') or test['lock'] != lock or lock.get('authorized') is not True:
            raise ValueError('Only completed parent-locked R64 TEST outputs may be reported')
        for arm, value in test['models'].items():
            selected[arm]['test'] = value
    references = {}
    if reference_tests is not None:
        if lock is None:
            raise ValueError('Historical TEST references are reported only with the later locked TEST report')
        paths = json.loads(Path(reference_tests).read_text())
        if not set(paths) <= {'R60B', 'R45A'}:
            raise ValueError('Reuse only the requested existing R60B/R45A official TEST references')
        references = {name: json.loads(Path(path).read_text()) for name, path in paths.items()}

    selected_rows = []
    for arm, splits in selected.items():
        for split, value in splits.items():
            selected_rows.append(dict(arm=arm, split=split, native_correct=value['native_correct'],
                                     native_total=value['native_total'], **value['macro']))
            for problem, metrics in value['per_problem'].items():
                directory = curves/arm/split/problem
                directory.mkdir(parents=True, exist_ok=True)
                methods = metrics['pool']
                for filename, values, selector, title in (
                    ('top1_vs_single_methods.png', [metrics['method_top1'][m] for m in methods], metrics['top1'], 'Native Top1'),
                    ('mean_cost_vs_single_methods.png', [metrics['method_mean_cost'][m] for m in methods], metrics['mean_cost'], 'Mean cost'),
                    ('arm_distribution.png', metrics['pick_dist'], None, 'Selection distribution')):
                    fig, ax = plt.subplots(figsize=(max(6, .6*len(methods)), 4))
                    ax.bar(np.arange(len(methods)), values, color='#227f79')
                    if selector is not None:
                        ax.axhline(selector, color='#b84332', label='selector')
                        ax.legend()
                    ax.set_xticks(np.arange(len(methods)), methods, rotation=55, ha='right')
                    ax.set(title=arm+' '+split+' '+problem, ylabel=title)
                    fig.tight_layout()
                    fig.savefig(directory/filename, dpi=150)
                    plt.close(fig)
    write_csv(root/'selected_metrics.csv', selected_rows)
    decision = None
    if set(results) == {'A', 'B'}:
        a, b = (results[x]['best']['val']['native_correct'] for x in ('A', 'B'))
        decision = dict(proposed_official_arm='A' if a >= b else 'B',
            pcgrad_demonstrated_selection_benefit=b > a, native_top1_gain_pp=100*(b-a)/18000,
            single_seed_provisional=True, objective70pct_reached=max(a, b)/18000 >= .70,
            cost_regret_or_margin_veto=False, decision_source='integer validation correct counts only')
    report = dict(selected=selected, decision=decision, actual_compute={arm: dict(
        wall_seconds=r.get('wall_seconds'), successful_updates=r['successful_updates'],
        presentations=r['presentations'], amp_retries=r['amp_retries'], actual_execution=r['actual_execution'])
        for arm, r in results.items()}, reused_official_test_references=references,
        locked_test_reported=lock is not None, historical_test_exposure_disclosed=True,
        evidence_scope='R64 TRAIN/validation adopted plan used no new TEST evidence; any later TEST is locked and cannot reselect',
        legacy_B_semantics='unresolved; R58 deferred')
    atomic_json(root/'analysis.json', report)
    print(json.dumps(dict(decision=decision, locked_test_reported=lock is not None), indent=2))
    return report
