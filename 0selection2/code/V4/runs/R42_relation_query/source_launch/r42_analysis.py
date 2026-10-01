"""Reproducible R42 curves and tables; never selects checkpoints from test."""

import argparse
import json
from pathlib import Path

import numpy as np

from code.unified_selector.registry import PROBLEMS
from .performance_analysis import write_csv


ROOT = Path('code/V4/runs/R42_relation_query')
RUN_NAMES = {'A': 'dual_stream_rdrop_seed2', 'B': 'relation_query_rdrop_seed2'}
METRICS = ('ce', 'top1', 'top2', 'top3', 'mean_cost', 'vs_sbs_pct', 'vs_oracle_pct', 'actual_regret_pct')


def plotting():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    return plt


def plot_run(directory):
    plt = plotting()
    history = json.loads((directory / 'history.json').read_text())
    initial = json.loads((directory / 'initial_eval.json').read_text())
    xs = [0] + [r['epoch'] for r in history]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for ax, key in zip(axes.flat, ('ce', 'top1', 'mean_cost', 'actual_regret_pct')):
        for split, color in (('train', '#327b88'), ('val', '#aa536d')):
            values = [initial[split]['macro']['macro_' + key]] + [r[split]['macro']['macro_' + key] for r in history]
            ax.plot(xs, values, label=split, color=color)
        ax.set(title='ALL ' + key, xlabel='Epoch')
        ax.legend()
    axes[1, 1].plot(xs[1:], [r['optimization']['kl'] for r in history], color='#5b8b56')
    axes[1, 1].set(title='Training symmetric KL', xlabel='Epoch')
    axes[1, 2].plot(xs[1:], [r['optimization']['loss'] for r in history], label='WC + R-Drop')
    axes[1, 2].set(title='Training objective', xlabel='Epoch')
    fig.tight_layout()
    fig.savefig(directory / 'train_val_curves.png', dpi=140)
    plt.close(fig)
    fig, axes = plt.subplots(2, 4, figsize=(15, 6))
    for j, family in enumerate(('TSP', 'CVRP', 'ATSP', 'MVRP')):
        for i, key in enumerate(('top1', 'actual_regret_pct')):
            for split, color in (('train', '#327b88'), ('val', '#aa536d')):
                values = [initial[split]['families'][family]['macro_' + key]] + [r[split]['families'][family]['macro_' + key] for r in history]
                axes[i, j].plot(xs, values, label=split, color=color)
            axes[i, j].set(title=family + ' ' + key, xlabel='Epoch')
            axes[i, j].legend()
    fig.tight_layout()
    fig.savefig(directory / 'family_curves.png', dpi=140)
    plt.close(fig)


def rows_for(name, value):
    macro, per = value['macro'], value['per_problem']
    yield dict(model=name, problem='ALL', **{k: macro['macro_' + k] for k in METRICS},
               sbs_cost=float(np.mean([r['sbs_cost'] for r in per.values()])),
               oracle_cost=float(np.mean([r['oracle_cost'] for r in per.values()])))
    for p, row in per.items():
        yield dict(model=name, problem=p, **{k: row[k] for k in METRICS},
                   sbs_cost=row['sbs_cost'], oracle_cost=row['oracle_cost'])


def table(lines, rows):
    lines.extend(['| Problem | Top1 | Top2 | Top3 | Mean cost | SBS cost (vs SBS) | Oracle cost (vs Oracle) | Actual regret | CE |',
                  '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |'])
    for r in rows:
        lines.append(f'| {r["problem"]} | {r["top1"]:.4f} | {r["top2"]:.4f} | {r["top3"]:.4f} | '
                     f'{r["mean_cost"]:.6f} | {r["sbs_cost"]:.6f} ({r["vs_sbs_pct"]:+.4f}%) | '
                     f'{r["oracle_cost"]:.6f} ({r["vs_oracle_pct"]:+.4f}%) | {r["actual_regret_pct"]:.4f}% | {r["ce"]:.4f} |')


def summary(root):
    plt = plotting()
    runs = {g: json.loads((root / name / 'result.json').read_text()) for g, name in RUN_NAMES.items()
            if (root / name / 'result.json').exists()}
    reference = json.loads(Path('code/V4/runs/R39_performance_model/winner_cost_seed2/result.json').read_text())
    ref = reference['best']['val']
    lines = ['# R42 关系感知 Solver-Query 对照', '',
             '固定 seed=2，A/B 从头训练；按完整验证集 macro vs SBS 最小选型。未经 test 重新选轮次。',
             'Top1 使用原生 ind；评估成本直接来自 raw_label 的 FP64。ALL 百分比逐问题宏平均，不使用 ALL 平均成本相除。', '',
             '## 验证选型、终点与末五轮', '',
             '| Run | Epochs | Best epoch | Val Top1 | Val vs SBS | Val regret | Final Top1 | Last5 Top1 | Last5 regret |',
             '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    comparison = []
    for g, result in runs.items():
        name = 'R42' + g
        best, final, last = result['best']['val']['macro'], result['final']['val']['macro'], result['last5']['val']
        lines.append(f'| {name} | {result["epochs_completed"]} | {result["best"]["epoch"]} | {best["macro_top1"]:.4f} | '
                     f'{best["macro_vs_sbs_pct"]:+.4f}% | {best["macro_actual_regret_pct"]:.4f}% | '
                     f'{final["macro_top1"]:.4f} | {last["macro_top1"]:.4f} | {last["macro_actual_regret_pct"]:.4f}% |')
        comparison.append(dict(model=name, epoch=result['best']['epoch'], scope='val_best', **best))
        lines.extend(['', f'### {name} 验证集最佳 checkpoint', ''])
        table(lines, rows_for(name, result['best']['val']))
        lines.extend(['', '| Family | Top1 | Mean cost | Actual regret | Top1 vs R39A |',
                      '| --- | ---: | ---: | ---: | ---: |'])
        for family, scores in result['best']['val']['families'].items():
            baseline = reference['best']['families'][family]
            delta = (scores['macro_top1'] - baseline['macro_top1']) * 100
            lines.append(f'| {family} | {scores["macro_top1"]:.4f} | {scores["macro_mean_cost"]:.6f} | '
                         f'{scores["macro_actual_regret_pct"]:.4f}% | {delta:+.3f} pp |')
        gain = (best['macro_top1'] - ref['macro_top1']) * 100
        reduction = (1 - best['macro_actual_regret_pct'] / ref['macro_actual_regret_pct']) * 100
        target = gain >= 2 and reduction >= 10
        lines.extend(['', f'{name} 相对 R39A 验证 Top1 {gain:+.3f} 个百分点；actual regret 相对下降 {reduction:+.2f}%。',
                      f'预设的整体目标（Top1 +2pp 且 regret -10%）：{"达到" if target else "未达到"}。目标不是统计显著性证明。'])
    write_csv(root / 'validation_comparison.csv', comparison)
    histories = {g: json.loads((root / name / 'history.json').read_text()) for g, name in RUN_NAMES.items()
                 if (root / name / 'history.json').exists()}
    for key in ('ce', 'top1', 'mean_cost', 'actual_regret_pct'):
        fig, ax = plt.subplots(figsize=(8, 4))
        for g, history in histories.items():
            for split, style in (('train', '--'), ('val', '-')):
                ax.plot([r['epoch'] for r in history], [r[split]['macro']['macro_' + key] for r in history],
                        linestyle=style, label='R42' + g + ' ' + split)
        ax.axhline(ref['macro_' + key], color='#555555', linestyle=':', label='R39A best val')
        ax.set(xlabel='Epoch', ylabel=key)
        ax.legend()
        fig.tight_layout()
        fig.savefig(root / f'comparison_{key}.png', dpi=140)
        plt.close(fig)
    for g, history in histories.items():
        directory = root / RUN_NAMES[g]
        for key in ('top1', 'ce', 'actual_regret_pct'):
            fig, axes = plt.subplots(6, 3, figsize=(14, 17))
            for ax, p in zip(axes.flat, PROBLEMS):
                for split, color in (('train', '#327b88'), ('val', '#aa536d')):
                    ax.plot([r['epoch'] for r in history], [r[split]['per_problem'][p][key] for r in history], label=split, color=color)
                ax.set(title=p, xlabel='Epoch', ylabel=key)
                ax.legend()
            fig.tight_layout()
            fig.savefig(directory / f'per_problem_{key}.png', dpi=130)
            plt.close(fig)
    test_path = root / 'test_results.json'
    if test_path.exists():
        current = json.loads(test_path.read_text())
        historical = json.loads(Path('code/V4/runs/R41_signal_audit/test_results.json').read_text())
        results = dict(historical, **{'R42' + g: r for g, r in current.items()})
        all_rows = [row for name, value in results.items() for row in rows_for(name, value)]
        write_csv(root / 'test_comparison.csv', all_rows)
        lines.extend(['', '## 冻结权重的完整 test 对比', '',
                      '| Model | Top1 | Top2 | Top3 | Mean cost | vs SBS | vs Oracle | Actual regret |',
                      '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |'])
        for name, value in results.items():
            m = value['macro']
            lines.append(f'| {name} | {m["macro_top1"]:.4f} | {m["macro_top2"]:.4f} | {m["macro_top3"]:.4f} | '
                         f'{m["macro_mean_cost"]:.6f} | {m["macro_vs_sbs_pct"]:+.4f}% | {m["macro_vs_oracle_pct"]:+.4f}% | {m["macro_actual_regret_pct"]:.4f}% |')
        family_rows = [dict(model=name, family=f, **metrics) for name, value in results.items() for f, metrics in value['families'].items()]
        write_csv(root / 'test_family_comparison.csv', family_rows)
        arms = [dict(model='R42'+g, problem=p, solver=s, pick_fraction=r['arm_distribution'][s],
                     winner_fraction=r['method_top1'][s]) for g, v in current.items() for p, r in v['per_problem'].items() for s in r['pool']]
        write_csv(root / 'test_arm_distribution.csv', arms)
        for g, value in current.items():
            name = 'R42' + g
            lines.extend(['', f'### {name} Test 逐问题', ''])
            table(lines, [r for r in all_rows if r['model'] == name])
            lines.extend(['', '| Family | Top1 | Mean cost | Actual regret | Top1 vs R39A |',
                          '| --- | ---: | ---: | ---: | ---: |'])
            for family, scores in value['families'].items():
                delta = (scores['macro_top1'] - historical['R39A']['families'][family]['macro_top1']) * 100
                lines.append(f'| {family} | {scores["macro_top1"]:.4f} | {scores["macro_mean_cost"]:.6f} | '
                             f'{scores["macro_actual_regret_pct"]:.4f}% | {delta:+.3f} pp |')
            lines.extend(['', '#### 方法选择比例', '', '| Problem | Solver | Pick | Native winner |', '| --- | --- | ---: | ---: |'])
            for row in [r for r in arms if r['model'] == name]:
                lines.append(f'| {row["problem"]} | {row["solver"]} | {row["pick_fraction"]:.2%} | {row["winner_fraction"]:.2%} |')
            for p, r in value['per_problem'].items():
                directory = root / 'test_method_plots' / name / p
                directory.mkdir(parents=True, exist_ok=True)
                for metric, method_key in (('top1', 'method_top1'), ('mean_cost', 'method_mean_cost')):
                    fig, ax = plt.subplots(figsize=(9, 4))
                    ax.bar(r['pool'], [r[method_key][s] for s in r['pool']], color='#bdc6c9')
                    ax.axhline(r[metric], color='#327b88', label=name)
                    ax.tick_params(axis='x', labelrotation=35, labelsize=8)
                    ax.set(title=p + ' ' + metric, ylabel=metric)
                    ax.legend()
                    fig.tight_layout()
                    fig.savefig(directory / f'{metric}_vs_single_methods.png', dpi=140)
                    plt.close(fig)
                fig, ax = plt.subplots(figsize=(9, 4))
                positions = np.arange(len(r['pool']))
                ax.bar(positions - .18, [r['arm_distribution'][s] for s in r['pool']], width=.36, label='Pick')
                ax.bar(positions + .18, [r['method_top1'][s] for s in r['pool']], width=.36, label='Native winner')
                ax.set(xticks=positions, xticklabels=r['pool'], title=p + ' solver distribution')
                ax.tick_params(axis='x', labelrotation=35, labelsize=8)
                ax.legend()
                fig.tight_layout()
                fig.savefig(directory / 'arm_distribution.png', dpi=140)
                plt.close(fig)
        lines.extend(['', '## 解释边界', '',
                      '这是单个 seed 的架构/训练对照，不证明多 seed 稳健性。A 与 B 的比较同时改变编码器和读出路径，不能把差异独立归因于 Q/K/V。',
                      '与 R39A 的比较同时改变 batch、训练调度、早停和 R-Drop，不能单独归因于一致性正则。',
                      '没有重新生成 solver 标签，也不把 R41 的两个 RELD 方法的稳定性推广到全部候选池。', ''])
    (root / 'comparison.md').write_text('\n'.join(lines) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=ROOT)
    args = parser.parse_args()
    summary(args.root)
