from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


BATCH_RE = re.compile(
    r"\[eval\] batch=(?P<batch>\d+) items=(?P<items>\d+)/(?:\d+) "
    r"avg_init=(?P<avg_init>[0-9.]+) avg_final=(?P<avg_final>[0-9.]+) "
    r"improve=(?P<improve>[0-9.]+) elapsed=(?P<elapsed>[0-9.]+)s"
)
WARN_RE = re.compile(r"\[warn\] checkpoint was trained at problem_size=(?P<train_size>\d+) but is now evaluated at problem_size=(?P<eval_size>\d+)")


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding='utf-8'))


def _parse_eval_log(path: Path) -> tuple[list[dict], list[str]]:
    batches: list[dict] = []
    warnings: list[str] = []
    if not path.exists():
        return batches, warnings
    seen_warnings: set[str] = set()
    for line in path.read_text(encoding='utf-8').splitlines():
        match = BATCH_RE.search(line)
        if match:
            item = {
                'batch': int(match.group('batch')),
                'items': int(match.group('items')),
                'avg_init': float(match.group('avg_init')),
                'avg_final': float(match.group('avg_final')),
                'improve': float(match.group('improve')),
                'elapsed': float(match.group('elapsed')),
            }
            item['relative_improve'] = item['improve'] / item['avg_init'] if item['avg_init'] > 1e-12 else 0.0
            item['items_per_second'] = item['items'] / item['elapsed'] if item['elapsed'] > 1e-12 else 0.0
            batches.append(item)
        warn = WARN_RE.search(line)
        if warn:
            warning_text = line.strip()
            if warning_text not in seen_warnings:
                warnings.append(warning_text)
                seen_warnings.add(warning_text)
    return batches, warnings


def _format_dist(names: list[str], values: list[float]) -> str:
    return ', '.join(f'{name}={value:.4f}' for name, value in zip(names, values))


def _argmax_name(names: list[str], values: list[float]) -> str:
    if not values:
        return 'N/A'
    best_idx = max(range(len(values)), key=lambda idx: values[idx])
    return str(names[best_idx])


def _collapse_level(max_share: float) -> str:
    if max_share >= 0.999999:
        return '完全坍缩'
    if max_share >= 0.95:
        return '几乎完全坍缩'
    if max_share >= 0.80:
        return '明显坍缩'
    return '未完全坍缩'


def _plot_batch_progress(out_path: Path, batches: list[dict], title_prefix: str) -> None:
    if not batches:
        return
    x = [row['items'] for row in batches]
    y_init = [row['avg_init'] for row in batches]
    y_final = [row['avg_final'] for row in batches]
    y_improve = [row['improve'] for row in batches]
    y_rel = [row['relative_improve'] for row in batches]
    y_speed = [row['items_per_second'] for row in batches]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    axes[0].plot(x, y_init, marker='o', label='avg_init')
    axes[0].plot(x, y_final, marker='o', label='avg_final')
    axes[0].set_title(f'{title_prefix} batch averages')
    axes[0].set_xlabel('Processed items')
    axes[0].set_ylabel('Length')
    axes[0].grid(alpha=0.25)
    axes[0].legend(frameon=False)

    axes[1].plot(x, y_improve, marker='o', label='improvement', color='tab:green')
    axes[1].plot(x, y_rel, marker='o', label='relative improvement', color='tab:orange')
    axes[1].set_title(f'{title_prefix} improvement stability')
    axes[1].set_xlabel('Processed items')
    axes[1].grid(alpha=0.25)
    axes[1].legend(frameon=False)

    axes[2].plot(x, y_speed, marker='o', color='tab:red')
    axes[2].set_title(f'{title_prefix} throughput')
    axes[2].set_xlabel('Processed items')
    axes[2].set_ylabel('Items / second')
    axes[2].grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _plot_collapse_distributions(out_path: Path, init_names: list[str], init_dist: list[float], op_names: list[str], op_dist: list[float], title_prefix: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))

    axes[0].bar(init_names, init_dist, color=['tab:blue', 'tab:orange', 'tab:green'][: len(init_names)])
    axes[0].set_ylim(0.0, 1.05)
    axes[0].set_ylabel('Share')
    axes[0].set_title(f'{title_prefix} initializer distribution')
    axes[0].grid(alpha=0.25, axis='y')
    for idx, value in enumerate(init_dist):
        axes[0].text(idx, value + 0.02, f'{value:.3f}', ha='center', va='bottom', fontsize=9)

    axes[1].bar(op_names, op_dist, color=['tab:blue', 'tab:orange', 'tab:green', 'tab:red'][: len(op_names)])
    axes[1].set_ylim(0.0, 1.05)
    axes[1].set_ylabel('Share')
    axes[1].set_title(f'{title_prefix} operator distribution')
    axes[1].grid(alpha=0.25, axis='y')
    for idx, value in enumerate(op_dist):
        axes[1].text(idx, value + 0.02, f'{value:.3f}', ha='center', va='bottom', fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _plot_eval_scalars(out_path: Path, metrics: dict, rollout_steps: int, title_prefix: str) -> None:
    init_length = float(metrics['init_length'])
    final_length = float(metrics['final_length'])
    improvement = float(metrics['improvement'])
    relative = improvement / init_length if init_length > 1e-12 else 0.0
    per_step = improvement / max(int(rollout_steps), 1)

    labels = ['init_length', 'final_length', 'improvement', 'relative', 'per_step']
    values = [init_length, final_length, improvement, relative, per_step]
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:purple', 'tab:red']

    plt.figure(figsize=(8.8, 4.6))
    plt.bar(labels, values, color=colors)
    plt.title(f'{title_prefix} key eval metrics')
    plt.grid(alpha=0.25, axis='y')
    for idx, value in enumerate(values):
        plt.text(idx, value + max(values) * 0.015 if max(values) > 0 else 0.01, f'{value:.4f}', ha='center', va='bottom', fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def _write_report(out_path: Path, eval_dir: Path, config: dict, metrics: dict, batches: list[dict], warnings: list[str]) -> None:
    init_names = [str(x) for x in config['init_zoo']]
    op_names = [str(x) for x in config['operator_zoo']]
    init_dist = [float(x) for x in metrics['init_distribution']]
    op_dist = [float(x) for x in metrics['operator_distribution']]
    init_max = max(init_dist) if init_dist else 0.0
    op_max = max(op_dist) if op_dist else 0.0
    init_choice = _argmax_name(init_names, init_dist)
    op_choice = _argmax_name(op_names, op_dist)
    init_length = float(metrics['init_length'])
    final_length = float(metrics['final_length'])
    improvement = float(metrics['improvement'])
    relative = improvement / init_length if init_length > 1e-12 else 0.0
    per_step = improvement / max(int(config.get('rollout_steps', 1)), 1)
    elapsed = float(metrics.get('elapsed_seconds', math.nan))
    num_items = int(metrics.get('num_items', 0))
    throughput = num_items / elapsed if num_items > 0 and elapsed > 1e-12 else math.nan

    lines: list[str] = []
    lines.append(f'# Eval 坍缩分析：`{eval_dir.name}`')
    lines.append('')
    lines.append('## 结果概览')
    lines.append('')
    lines.append(f'- 问题规模：`{config.get("problem_size")}`')
    lines.append(f'- 数据集：`{config.get("eval_data_path")}`')
    lines.append(f'- 评测样本数：`{num_items}`')
    lines.append(f'- rollout 步数：`{config.get("rollout_steps")}`')
    lines.append(f'- 初始解均值：`{init_length:.4f}`')
    lines.append(f'- 最终解均值：`{final_length:.4f}`')
    lines.append(f'- 平均改进：`{improvement:.4f}`')
    lines.append(f'- 相对改进：`{relative:.4%}`')
    lines.append(f'- 单步平均改进：`{per_step:.6f}`')
    if not math.isnan(elapsed):
        lines.append(f'- 总耗时：`{elapsed:.2f}s`，吞吐：`{throughput:.4f}` items/s')
    lines.append('')
    lines.append('## 初始化坍缩')
    lines.append('')
    lines.append(f'- 分布：`{_format_dist(init_names, init_dist)}`')
    lines.append(f'- 最大占比：`{init_max:.4f}`，判定：**{_collapse_level(init_max)}**')
    lines.append(f'- 主导初始化器：`{init_choice}`')
    lines.append(f'- 解释：`init_distribution` 是在整个评测集上统计“每个实例最终选了哪个初始化器”的频率；这里若某一项为 1，表示所有实例都选了同一个初始化器。')
    lines.append('')
    lines.append('## 迭代算子坍缩')
    lines.append('')
    lines.append(f'- 分布：`{_format_dist(op_names, op_dist)}`')
    lines.append(f'- 最大占比：`{op_max:.4f}`，判定：**{_collapse_level(op_max)}**')
    lines.append(f'- 主导算子：`{op_choice}`')
    lines.append(f'- 解释：`operator_distribution` 统计的是整个评测过程中“所有实例 × 所有 rollout step”的动作频率；若某一项为 1，表示每个实例在每一步都用了同一个 operator。')
    lines.append('')
    lines.append('## Batch 过程稳定性')
    lines.append('')
    if batches:
        first = batches[0]
        last = batches[-1]
        lines.append(f'- 日志中解析到 `batch` 进度点：`{len(batches)}` 个。')
        lines.append(f'- 首个进度点：items=`{first["items"]}`，avg_init=`{first["avg_init"]:.4f}`，avg_final=`{first["avg_final"]:.4f}`，improve=`{first["improve"]:.4f}`。')
        lines.append(f'- 最后进度点：items=`{last["items"]}`，avg_init=`{last["avg_init"]:.4f}`，avg_final=`{last["avg_final"]:.4f}`，improve=`{last["improve"]:.4f}`。')
        lines.append('- 若首尾数值变化很小，说明评测过程中统计量较稳定，没有明显因为后续 batch 而漂移。')
    else:
        lines.append('- `eval.log` 中没有足够的 batch 进度点，因此只能依据最终汇总指标判断。')
    lines.append('')
    lines.append('## 注意事项')
    lines.append('')
    if warnings:
        for warning in warnings:
            lines.append(f'- {warning}')
        lines.append('- 这说明当前 selector 是在 `tsp100` 上训练的，再外推到更大尺度；因此 200/500 的结果更应理解为“跨尺度泛化下的动作选择行为”。')
    else:
        lines.append('- 当前规模与训练规模一致，因此这里的坍缩更能反映模型在同尺度测试集上的真实偏好。')
    lines.append('- 这三个 eval 目录都没有逐实例 trace，因此这里的坍缩结论是“数据集整体层面”的，不是逐实例动作序列层面的。')

    out_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def analyze_one(eval_dir: Path) -> None:
    config = _load_json(eval_dir / 'eval_config.json')
    metrics = _load_json(eval_dir / 'eval_metrics.json')
    batches, warnings = _parse_eval_log(eval_dir / 'eval.log')

    analysis_dir = eval_dir / 'analysis'
    analysis_dir.mkdir(parents=True, exist_ok=True)

    init_names = [str(x) for x in config['init_zoo']]
    op_names = [str(x) for x in config['operator_zoo']]
    init_dist = [float(x) for x in metrics['init_distribution']]
    op_dist = [float(x) for x in metrics['operator_distribution']]

    _plot_batch_progress(analysis_dir / 'batch_progress.png', batches, eval_dir.name)
    _plot_collapse_distributions(analysis_dir / 'collapse_distribution.png', init_names, init_dist, op_names, op_dist, eval_dir.name)
    _plot_eval_scalars(analysis_dir / 'key_metrics.png', metrics, int(config.get('rollout_steps', 1)), eval_dir.name)

    summary = {
        'eval_dir': str(eval_dir),
        'problem_size': int(config.get('problem_size', 0)),
        'eval_data_path': config.get('eval_data_path'),
        'num_items': int(metrics.get('num_items', 0)),
        'num_batches': int(metrics.get('num_batches', 0)),
        'elapsed_seconds': float(metrics.get('elapsed_seconds', 0.0)),
        'init_length': float(metrics['init_length']),
        'final_length': float(metrics['final_length']),
        'improvement': float(metrics['improvement']),
        'relative_improvement': float(metrics['improvement']) / float(metrics['init_length']) if float(metrics['init_length']) > 1e-12 else 0.0,
        'improvement_per_rollout_step': float(metrics['improvement']) / max(int(config.get('rollout_steps', 1)), 1),
        'init_distribution_named': {name: float(value) for name, value in zip(init_names, init_dist)},
        'operator_distribution_named': {name: float(value) for name, value in zip(op_names, op_dist)},
        'init_collapse_max_share': max(init_dist) if init_dist else 0.0,
        'operator_collapse_max_share': max(op_dist) if op_dist else 0.0,
        'init_collapse_choice': _argmax_name(init_names, init_dist),
        'operator_collapse_choice': _argmax_name(op_names, op_dist),
        'batch_points_logged': len(batches),
        'warnings': warnings,
    }
    (analysis_dir / 'summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
    _write_report(analysis_dir / 'ANALYSIS.md', eval_dir, config, metrics, batches, warnings)


def main() -> None:
    parser = argparse.ArgumentParser(description='Analyze eval-only operator_policy outputs.')
    parser.add_argument('eval_dirs', nargs='+', help='One or more eval output directories.')
    args = parser.parse_args()

    for item in args.eval_dirs:
        analyze_one(Path(item))


if __name__ == '__main__':
    main()
