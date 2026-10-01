"""R43 curves, fixed-reference comparisons, and complete selection outcomes."""

import argparse
import json
from pathlib import Path

import numpy as np

from code.unified_selector.registry import PROBLEMS
from .multitask_probe import dump
from .performance_analysis import write_csv
from .r40_experiment import aggregate, families
from .r43_experiment import ROOT, R42, RUN_NAMES, policy_metrics, strong_comparisons, add_diagnostics


def plotting():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    return plt


def plot_run(directory):
    plt = plotting()
    history = json.loads((directory / 'history.json').read_text())
    initial = json.loads((directory / 'initial_eval.json').read_text())
    fields = ('top1', 'ce', 'mean_cost', 'actual_regret_pct', 'true3_accuracy', 'reference_top2_accuracy')
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, field in zip(axes.flat, fields):
        for split in ('train', 'val'):
            points = [(0, initial[split]['macro']['macro_' + field])]
            points += [(r['epoch'], r[split]['macro']['macro_' + field]) for r in history if r[split] is not None]
            x, y = zip(*points)
            ax.plot(x, y, marker='.', label=split)
        ax.set(title=field, xlabel='Epoch')
        ax.grid(alpha=.2)
        ax.legend()
    fig.tight_layout()
    fig.savefig(directory / 'train_val_curves.png', dpi=150)
    plt.close(fig)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, names in zip(axes.flat, (('loss',), ('ce', 'cmp'), ('true3', 'pred3', 'all_pairs'), ('risk',), ('kl',), ('lr',))):
        for name in names:
            ax.plot([r['epoch'] for r in history], [r['optimization'][name] for r in history], label=name)
        ax.set(xlabel='Epoch', title=' / '.join(names))
        ax.grid(alpha=.2)
        ax.legend()
    fig.tight_layout()
    fig.savefig(directory / 'optimization_curves.png', dpi=150)
    plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, family in zip(axes.flat, ('TSP', 'CVRP', 'ATSP', 'MVRP')):
        for split in ('train', 'val'):
            rows = [r for r in history if r[split] is not None]
            ax.plot([r['epoch'] for r in rows], [r[split]['families'][family]['macro_top1'] for r in rows], label=split)
        ax.set(title=family, xlabel='Epoch', ylabel='Top1')
        ax.grid(alpha=.2)
        ax.legend()
    fig.tight_layout()
    fig.savefig(directory / 'family_top1_curves.png', dpi=150)
    plt.close(fig)


def rows_for(name, value):
    fields = ('top1', 'top2', 'top3', 'ce', 'mean_cost', 'vs_sbs_pct', 'vs_oracle_pct',
              'actual_regret_pct', 'true3_accuracy', 'reference_top2_accuracy')
    rows = [dict(model=name, problem='ALL', n=sum(r['n'] for r in value['per_problem'].values()),
                 **{key: value['macro']['macro_' + key] for key in fields},
                 sbs_cost=float(np.mean([r['sbs_cost'] for r in value['per_problem'].values()])),
                 oracle_cost=float(np.mean([r['oracle_cost'] for r in value['per_problem'].values()])))]
    rows += [dict(model=name, problem=p, n=r['n'], **{key: r[key] for key in fields},
                  sbs_cost=r['sbs_cost'], oracle_cost=r['oracle_cost']) for p, r in value['per_problem'].items()]
    return rows


def plot_methods(name, value, directory):
    plt = plotting()
    directory.mkdir(parents=True, exist_ok=True)
    for p, r in value['per_problem'].items():
        fig, axes = plt.subplots(1, 3, figsize=(17, 4))
        names = r['pool']
        for ax, field, label in zip(axes[:2], ('method_top1', 'method_mean_cost'), ('top1', 'mean_cost')):
            ax.bar(names, [r[field][s] for s in names], color='#599eae')
            ax.axhline(r[label], color='#c45746', label=name)
            ax.set_title(label)
            ax.legend()
        axes[2].bar(names, [r['arm_distribution'][s] for s in names], color='#599eae')
        axes[2].set_title('Solver pick fraction')
        for ax in axes:
            ax.tick_params(axis='x', labelrotation=55)
        fig.suptitle(f'{name}: {p}')
        fig.tight_layout()
        fig.savefig(directory / f'{p}.png', dpi=130)
        plt.close(fig)


def metric_table(rows):
    lines = ['| Model | Problem | Top1 | Top2 | Top3 | mean_cost | vs_SBS | actual regret |',
             '|---|---|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        lines.append(f'| {r["model"]} | {r["problem"]} | {r["top1"]:.4f} | {r["top2"]:.4f} | {r["top3"]:.4f} | '
                     f'{r["mean_cost"]:.6f} | {r["vs_sbs_pct"]:+.4f}% | {r["actual_regret_pct"]:.4f}% |')
    return lines


def reference_results(root):
    per = {}
    for p in PROBLEMS:
        with np.load(root / 'reference_predictions/test' / f'{p}.npz') as saved:
            raw = dict(costs=saved['costs'], winner=saved['winner'], pool_ids=saved['pool_ids'].tolist(),
                       pool=json.loads((root / 'test_data_manifest.json').read_text())[p]['pool'])
            scores = saved['logits']
            per[p], pred = policy_metrics(scores, raw)
            per[p].update(strong_comparisons(scores[:, :, None] - scores[:, None, :], raw, saved['top2']))
    return add_diagnostics(dict(per_problem=per, macro=aggregate(per), families=families(per)))


def transitions(root, group):
    rows = []
    for p in PROBLEMS:
        with np.load(root / 'test_predictions' / group / f'{p}.npz') as current, np.load(root / 'reference_predictions/test' / f'{p}.npz') as reference:
            np.testing.assert_array_equal(current['winner'], reference['winner'])
            np.testing.assert_array_equal(current['costs'], reference['costs'])
            before, after, winner, costs = reference['pred'], current['pred'], current['winner'], current['costs']
            idx = np.arange(len(before))
            difference = costs[idx, after] - costs[idx, before]
            regret_difference = difference / costs.min(-1) * 100
            masks = dict(corrected=(before != winner) & (after == winner),
                         harmed=(before == winner) & (after != winner),
                         both_wrong_changed=(before != winner) & (after != winner) & (before != after),
                         all=np.ones(len(before), dtype=bool))
            for kind, mask in masks.items():
                rows.append(dict(model='R43' + group, problem=p, category=kind, n=int(mask.sum()),
                                 cost_delta_sum=float(difference[mask].sum()),
                                 cost_delta_per_dataset_instance=float(difference[mask].sum() / len(before)),
                                 regret_delta_per_dataset_instance=float(regret_difference[mask].sum() / len(before))))
    for kind in ('corrected', 'harmed', 'both_wrong_changed', 'all'):
        selected = [r for r in rows if r['category'] == kind]
        rows.append(dict(model='R43' + group, problem='ALL', category=kind, n=sum(r['n'] for r in selected),
                         cost_delta_sum=sum(r['cost_delta_sum'] for r in selected),
                         cost_delta_per_dataset_instance=float(np.mean([r['cost_delta_per_dataset_instance'] for r in selected])),
                         regret_delta_per_dataset_instance=float(np.mean([r['regret_delta_per_dataset_instance'] for r in selected]))))
    return rows


def summarize(root=ROOT):
    root = Path(root)
    results = {'R42A': reference_results(root)}
    results.update({'R43' + group: result for group, result in json.loads((root / 'test_results.json').read_text()).items()})
    rows = [row for name, result in results.items() for row in rows_for(name, result)]
    write_csv(root / 'test_comparison.csv', rows)
    changes = transitions(root, 'A') + transitions(root, 'B')
    write_csv(root / 'decision_changes.csv', changes)
    family_rows = [dict(model=name, family=family, **scores) for name, result in results.items() for family, scores in result['families'].items()]
    write_csv(root / 'family_comparison.csv', family_rows)
    diagnostics = [dict(model=r['model'], problem=r['problem'], true3_accuracy=r['true3_accuracy'],
                        reference_top2_accuracy=r['reference_top2_accuracy']) for r in rows]
    write_csv(root / 'strong_comparisons.csv', diagnostics)
    arm_rows = [dict(model=name, problem=p, solver=s, pick_rate=r['arm_distribution'][s],
                     winner_rate=r['method_top1'][s], mean_cost=r['method_mean_cost'][s])
                for name, result in results.items() for p, r in result['per_problem'].items() for s in r['pool']]
    write_csv(root / 'arm_distributions.csv', arm_rows)
    lines = ['# R43: full-pool pairwise selection', '',
             '一个 seed=2；A/B 从头训练。全部模型使用原生 ind；成本从原始标签以 FP64 汇总。',
             'R42A 只复用已冻结的 best；R43 按验证 macro_vs_sbs_pct 严格最小锁定，test 不参与选型。',
             'ALL 的归一化百分比是逐问题宏平均，不是两个 ALL 平均成本直接相除。', '', '## Test 总览', '']
    lines += metric_table([r for r in rows if r['problem'] == 'ALL'])
    lines += ['', '## 训练与验证选型', '', '| Model | Stop epoch | Best epoch | Best val Top1 | Best val vs_SBS | Final train Top1 | Final val Top1 | Last5 val Top1 |',
              '|---|---:|---:|---:|---:|---:|---:|---:|']
    for group, directory in RUN_NAMES.items():
        record = json.loads((root / directory / 'result.json').read_text())
        best, final = record['best'], record['final']
        lines.append(f'| R43{group} | {record["epochs_completed"]} | {best["epoch"]} | {best["val"]["macro"]["macro_top1"]:.4f} | '
                     f'{best["val"]["macro"]["macro_vs_sbs_pct"]:+.4f}% | {final["train"]["macro"]["macro_top1"]:.4f} | '
                     f'{final["val"]["macro"]["macro_top1"]:.4f} | {record["last5"]["val"]["macro_top1"]:.4f} |')
        plot_run(root / directory)
    lines += ['', '## 强候选比较', '',
              '真实 Top3 内部按实例平均，只统计原始成本不相等的方法对；零 margin 不算正确。',
              '第二个诊断固定比较 R42A 提名的两个方法，不是新模型自己选的容易比较。', '',
              '| Model | true Top3 accuracy | R42A nominated Top2 accuracy |', '|---|---:|---:|']
    for name, value in results.items():
        lines.append(f'| {name} | {value["macro"]["macro_true3_accuracy"]:.4f} | {value["macro"]["macro_reference_top2_accuracy"]:.4f} |')
    lines += ['', '## 相对 R42A 的纠正与伤害', '', '成本与 regret 差值：负值表示改善；分类正确性使用 native winner。', '',
              '| Model | Change | Instances | Delta mean cost | Delta actual regret |', '|---|---|---:|---:|---:|']
    for r in changes:
        if r['problem'] == 'ALL':
            lines.append(f'| {r["model"]} | {r["category"]} | {r["n"]} | {r["cost_delta_per_dataset_instance"]:+.6f} | {r["regret_delta_per_dataset_instance"]:+.5f}% |')
    lines += ['', '## 结果判断', '']
    base = results['R42A']['macro']
    for name in ('R43A', 'R43B'):
        m = results[name]['macro']
        lines.append(f'- {name} 相对 R42A：Top1 {(m["macro_top1"]-base["macro_top1"])*100:+.3f} 个百分点；'
                     f'actual regret {m["macro_actual_regret_pct"]-base["macro_actual_regret_pct"]:+.5f} 个百分点；'
                     f'mean_cost {m["macro_mean_cost"]-base["macro_mean_cost"]:+.6f}。')
    a, b = results['R43A']['macro'], results['R43B']['macro']
    lines.append(f'- 显式比较 B 相对匹配对照 A：Top1 {(b["macro_top1"]-a["macro_top1"])*100:+.3f} 个百分点；'
                 f'actual regret {b["macro_actual_regret_pct"]-a["macro_actual_regret_pct"]:+.5f} 个百分点。')
    lines.append('- 本轮只有一个从头训练 seed，不把微小差异解释为稳定优势；最终选择指标优先于 pair loss 或比较正确率。')
    lines += ['', '## 逐问题 Test', '']
    for p in PROBLEMS:
        lines += metric_table([r for r in rows if r['problem'] == p]) + ['']
    lines += ['## 臂分布', '', '| Model | Problem | Solver picks |', '|---|---|---|']
    for name, value in results.items():
        for p, r in value['per_problem'].items():
            text = ' / '.join(f'{s}: {100*v:.1f}%' for s, v in r['arm_distribution'].items())
            lines.append(f'| {name} | {p} | {text} |')
    (root / 'comparison.md').write_text('\n'.join(lines) + '\n')
    dump(root / 'comparison.json', results)
    for name, value in results.items():
        plot_methods(name, value, root / 'method_plots' / name)
    print(f'[report] {root / "comparison.md"}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=ROOT)
    summarize(parser.parse_args().root)
