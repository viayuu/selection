"""R40 full train/eval curves and classification-policy comparison tables."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from .multitask_probe import dump
from .performance_analysis import result_table, write_csv
from .r40_experiment import METRICS, ROOT, RUN_NAMES


def read(path):
    return json.loads(path.read_text())


def plot_run(directory):
    history = read(directory / 'history.json')
    initial = read(directory / 'initial_eval.json')
    for key in ('macro_ce', 'macro_top1', 'macro_mean_cost', 'macro_vs_sbs_pct', 'macro_actual_regret_pct'):
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        for ax, xkey, label in zip(axes, ('epoch', 'updates'), ('Data epoch', 'Successful optimizer updates')):
            ax.plot([0] + [r[xkey] for r in history], [initial['val']['macro'][key]] + [r['val'][key] for r in history],
                    label='Full val (eval mode)', color='#287e91')
            train = [r for r in history if 'train_eval' in r]
            ax.plot([0] + [r[xkey] for r in train], [initial['train']['macro'][key]] + [r['train_eval'][key] for r in train],
                    label='Full train (eval mode)', color='#c45746', marker='.')
            ax.set(xlabel=label, ylabel=key.removeprefix('macro_'), title=directory.name)
            ax.grid(alpha=.25)
            ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(directory / f'train_val_{key[6:]}.png', dpi=150)
        plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, xkey, label in zip(axes, ('epoch', 'updates'), ('Data epoch', 'Successful optimizer updates')):
        for key in ('loss', 'ce', 'pair', 'risk'):
            ax.plot([r[xkey] for r in history], [r['train'][key] for r in history], label=key)
        ax.set(xlabel=label, ylabel='Training-mode objective components', title=directory.name)
        ax.legend()
        ax.grid(alpha=.25)
    fig.tight_layout()
    fig.savefig(directory / 'loss_curve.png', dpi=150)
    plt.close(fig)


def plot_methods(record, directory):
    for p, r in record['per_problem'].items():
        output = directory / 'method_plots' / p
        output.mkdir(parents=True, exist_ok=True)
        names = r['pool']
        for metric, actual, filename in (('method_top1', r['top1'], 'top1_vs_single_methods'),
                                         ('method_mean_cost', r['mean_cost'], 'mean_cost_vs_single_methods')):
            fig, ax = plt.subplots(figsize=(9, 4))
            ax.bar(names, [r[metric][name] for name in names], color='#599eae')
            ax.axhline(actual, color='#c45746', label=directory.name)
            ax.set(title=f'{p}: selected validation epoch {record["epoch"]}', ylabel=metric)
            ax.tick_params(axis='x', labelrotation=45)
            ax.legend()
            fig.tight_layout()
            fig.savefig(output / f'{filename}.png', dpi=150)
            plt.close(fig)
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.bar(names, [r['arm_distribution'][name] for name in names], color='#599eae')
        ax.set(title=f'{directory.name}: {p}', ylabel='Selected solver fraction')
        ax.tick_params(axis='x', labelrotation=45)
        fig.tight_layout()
        fig.savefig(output / 'arm_distribution.png', dpi=150)
        plt.close(fig)


def summarize(root):
    reference = read(root / 'reference_R39A.json')
    results = {g: read(root / name / 'result.json') for g, name in RUN_NAMES.items()}
    protocol = read(root / 'protocol.json')
    rows = []
    for group, result in [('R39A reference', reference), *results.items()]:
        for point in ('best', 'final', 'last5'):
            values = result[point] if point == 'last5' else result[point]['val']
            rows.append(dict(group=group, point=point, **{'macro_' + k: values['macro_' + k] for k in METRICS}))
    write_csv(root / 'comparison.csv', rows)
    lines = ['# R40: Discriminative Training', '',
             'Seed 2, full 18-task train/validation. No test evaluation.',
             'A: dual-stream classification. B: direct global classifier. Both use the identical winner-cost loss.',
             'Both are trained from scratch; the shared encoder starts with identical weights.',
             'The sample sequence matches R39A; 64-instance real updates increase the update count tenfold without gradient accumulation.',
             'Learning rate is scheduled per successful update, not per historical R39 step.',
             'ALL percentages average per-problem percentages, not ratios of aggregate costs.',
             '', '## Validation Comparison', '',
             '| Model | Point | Top1 | Top2 | Top3 | Mean cost | vs_SBS (%) | Actual regret (%) | CE |',
             '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for row in rows:
        lines.append(f"| {row['group']} | {row['point']} | {row['macro_top1']:.4f} | {row['macro_top2']:.4f} | "
                     f"{row['macro_top3']:.4f} | {row['macro_mean_cost']:.6f} | {row['macro_vs_sbs_pct']:+.4f} | "
                     f"{row['macro_actual_regret_pct']:.4f} | {row['macro_ce']:.5f} |")
    train_reference = reference['train_final']['macro']
    train_rows = []
    lines += ['', '## Full Training-Set Fit', '',
              'All training metrics below use eval mode, natural distribution and complete tail batches.',
              'The +5 percentage-point fit flag is a screening criterion, not a release criterion.', '',
              '| Model | Top1 | Delta vs R39A (pp) | CE | Mean cost | Actual regret (%) | Fit flag |',
              '| --- | ---: | ---: | ---: | ---: | ---: | --- |']
    for group, values in [('R39A reference', train_reference)] + [(g, r['train_final']) for g, r in results.items()]:
        delta = (values['macro_top1'] - train_reference['macro_top1']) * 100
        flag = delta >= 5 and values['macro_ce'] < train_reference['macro_ce']
        train_rows.append(dict(group=group, top1_delta_pp=delta, fit_flag=flag, **values))
        lines.append(f"| {group} | {values['macro_top1']:.4f} | {delta:+.4f} | {values['macro_ce']:.5f} | "
                     f"{values['macro_mean_cost']:.6f} | {values['macro_actual_regret_pct']:.4f} | {flag} |")
    write_csv(root / 'training_fit.csv', train_rows)
    a, b = results['A'], results['B']
    stop_path = root / 'user_stop.json'
    stopped = read(stop_path) if stop_path.exists() else None
    expected_epochs = {'A': stopped['checkpoint_epoch'] if stopped else 60, 'B': 60}
    checks = dict(seed2_only=a['seed'] == b['seed'] == 2,
                  same_encoder_initialization=a['encoder_initial_hash'] == b['encoder_initial_hash'],
                  same_sample_and_task_schedule=a['schedule_hash'] == b['schedule_hash'] == protocol['schedule_hash'],
                  same_source_snapshot=a['source_hashes'] == b['source_hashes'] == protocol['source_hashes'],
                  declared_run_endpoints=all(r['epochs'] == expected_epochs[g] and r['updates'] == expected_epochs[g] * 2700
                                             for g, r in results.items()),
                  balanced_task_updates=all(set(r['updates_by_problem'].values()) == {expected_epochs[g] * 150}
                                            for g, r in results.items()),
                  no_test_read=not a['test_read'] and not b['test_read'])
    if stopped:
        checks['explicit_user_stop'] = a.get('stopped_by_user') is True and a.get('completed_budget') is False
    if not all(checks.values()):
        raise ValueError(f'R40 paired protocol failed: {checks}')
    dump(root / 'paired_checks.json', checks)
    differences = []
    for point in ('best', 'final', 'last5'):
        av = a[point] if point == 'last5' else a[point]['val']
        bv = b[point] if point == 'last5' else b[point]['val']
        differences.append(dict(point=point, **{key: bv[key] - av[key] for key in av}))
    write_csv(root / 'paired_differences.csv', differences)
    lines += ['', '## Paired B Minus A', '',
              'Best checkpoints use the original validation-cost rule. Terminal and last-five rows below use each run\'s own endpoint.',
              '| Point | Top1 (pp) | Mean cost | vs_SBS (pp) | CE |', '| --- | ---: | ---: | ---: | ---: |']
    for row in differences:
        lines.append(f"| {row['point']} | {row['macro_top1'] * 100:+.4f} | {row['macro_mean_cost']:+.6f} | "
                     f"{row['macro_vs_sbs_pct']:+.4f} | {row['macro_ce']:+.5f} |")
    if stopped:
        common_epoch = min(a['epochs'], b['epochs'])
        histories = {g: read(root / name / 'history.json') for g, name in RUN_NAMES.items()}
        aligned = []
        for group, history in histories.items():
            matched = [r for r in history if r['epoch'] <= common_epoch]
            for point, values in (('matched_final', matched[-1]['val']),
                                  ('matched_last5', {k: float(np.mean([r['val'][k] for r in matched[-5:]]))
                                                     for k in matched[-1]['val']})):
                aligned.append(dict(group=group, point=point, epoch=common_epoch, updates=common_epoch * 2700, **values))
        write_csv(root / 'matched_endpoint_comparison.csv', aligned)
        lines += ['', '## Explicit Early Stop and Matched Budget', '',
                  f"A was stopped by the user after checkpoint epoch {a['epochs']} ({a['updates']} saved successful updates). B completed 60 epochs.",
                  'Unsaved updates after the A checkpoint are excluded. This is not a completed 60/60 pair.',
                  f'Architecture comparisons at equal budget use epoch {common_epoch} and its preceding five validation points.', '',
                  '| Group | Point | Epoch | Top1 | Mean cost | vs_SBS (%) | Actual regret (%) | CE |',
                  '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |']
        for row in aligned:
            lines.append(f"| {row['group']} | {row['point']} | {row['epoch']} | {row['macro_top1']:.4f} | "
                         f"{row['macro_mean_cost']:.6f} | {row['macro_vs_sbs_pct']:+.4f} | "
                         f"{row['macro_actual_regret_pct']:.4f} | {row['macro_ce']:.5f} |")
        train_epochs = set.intersection(*[{r['epoch'] for r in h if 'train_eval' in r} for h in histories.values()])
        train_epoch = max(e for e in train_epochs if e <= common_epoch)
        matched_train = [dict(group=g, epoch=train_epoch, **next(r['train_eval'] for r in h if r['epoch'] == train_epoch))
                         for g, h in histories.items()]
        write_csv(root / 'matched_train_comparison.csv', matched_train)
        lines += ['', f'### Full Train at Matched Epoch {train_epoch}', '',
                  '| Group | Top1 | CE | Mean cost | Actual regret (%) |', '| --- | ---: | ---: | ---: | ---: |']
        for row in matched_train:
            lines.append(f"| {row['group']} | {row['macro_top1']:.4f} | {row['macro_ce']:.5f} | "
                         f"{row['macro_mean_cost']:.6f} | {row['macro_actual_regret_pct']:.4f} |")
    family_rows = []
    for group, result in results.items():
        directory = root / RUN_NAMES[group]
        plot_run(directory)
        plot_methods(result['best'], directory)
        lines += ['', f"## {RUN_NAMES[group]}: Best Validation Epoch {result['best']['epoch']}", '']
        lines.extend(result_table(result['best']))
        lines += ['', '| Problem | Selected solver distribution |', '| --- | --- |']
        for p, r in result['best']['per_problem'].items():
            text = '; '.join(f'{name}: {rate * 100:.1f}%' for name, rate in r['arm_distribution'].items())
            lines.append(f'| {p} | {text} |')
        for name, values in result['best']['families'].items():
            family_rows.append(dict(group=group, family=name, **values))
    write_csv(root / 'family_comparison.csv', family_rows)
    (root / 'comparison.md').write_text('\n'.join(lines) + '\n')
    for key in ('macro_top1', 'macro_ce', 'macro_mean_cost', 'macro_vs_sbs_pct', 'macro_actual_regret_pct'):
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        for ax, xkey, label in zip(axes, ('epoch', 'updates'), ('Data epoch', 'Successful optimizer updates')):
            for group, color in (('A', '#c45746'), ('B', '#287e91')):
                history = read(root / RUN_NAMES[group] / 'history.json')
                ax.plot([r[xkey] for r in history], [r['val'][key] for r in history], color=color, label=group)
            ax.axhline(reference['final']['val'][key], color='#6d746e', linestyle='--', label='R39A final')
            ax.set(xlabel=label, ylabel=key[6:])
            ax.grid(alpha=.25)
            ax.legend()
        fig.tight_layout()
        fig.savefig(root / f'comparison_{key[6:]}.png', dpi=150)
        plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    summarize(parser.parse_args().root)
