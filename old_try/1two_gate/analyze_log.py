#!/usr/bin/env python3
"""
双层强化学习TSP求解器选择训练日志分析工具

分析 train_two_gate_tsp 生成的日志文件，提供：
1. 配置信息摘要
2. 训练曲线可视化（loss、长度、优势函数等）
3. Gate选择分布演化
4. 性能指标分析
5. 收敛性诊断
"""

import re
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.gridspec import GridSpec

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


@dataclass
class TrainStep:
    """单步训练记录"""
    step: int
    loss: float
    len0: float  # 初始化后长度
    len1: float  # 迭代后长度
    adv_mean: float  # 优势函数均值
    adv_std: float   # 优势函数标准差
    critic_mse: float
    entropy: float
    gate1_dist: Dict[str, float]  # Gate1选择分布
    gate2_dist: Dict[str, float]  # Gate2选择分布


@dataclass
class EvalStep:
    """单步评估记录"""
    step: int
    mean_len: float
    best_len: float
    gate1_dist: Dict[str, float]
    gate2_dist: Dict[str, float]


@dataclass
class Config:
    """训练配置"""
    seed: int
    device: str
    problem_size: int
    batch_size: int
    train_steps: int
    lr: float
    log_every: int
    eval_every: int
    init_zoo: List[str]
    iter_zoo: List[str]
    baseline: str
    critic_coef: float
    rrc_steps: int
    two_opt_iters: int


@dataclass
class LogData:
    """解析后的日志数据"""
    config: Config
    train_steps: List[TrainStep] = field(default_factory=list)
    eval_steps: List[EvalStep] = field(default_factory=list)


def parse_log_file(log_path: str) -> LogData:
    """解析日志文件"""
    with open(log_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    config = None
    train_steps = []
    eval_steps = []

    # 训练日志正则
    train_pattern = re.compile(
        r'\[train step\s+(\d+)\]\s+'
        r'loss=([\d\.\-]+)\s+'
        r'len0=([\d\.\-]+)\s+'
        r'len1=([\d\.\-]+)\s+'
        r'adv=([\d\.\-]+)±([\d\.\-]+)\s+'
        r'critic_mse=([\d\.\-]+)\s+'
        r'entropy=([\d\.\-]+)\s+'
        r'gate1=\((.*?)\)\s+'
        r'gate2=\((.*?)\)'
    )

    # 评估日志正则
    eval_pattern = re.compile(
        r'\[eval step\s+(\d+)\]\s+'
        r'mean_len=([\d\.\-]+)\s+'
        r'best=([\d\.\-]+)\s+'
        r'gate1=\((.*?)\)\s+'
        r'gate2=\((.*?)\)'
    )

    # 配置日志正则
    config_start = -1
    for i, line in enumerate(lines):
        if line.strip() == '[config] {':
            config_start = i
            break

    # 解析配置
    if config_start >= 0:
        config_lines = []
        brace_count = 0
        for i in range(config_start, len(lines)):
            config_lines.append(lines[i])
            brace_count += lines[i].count('{') - lines[i].count('}')
            if brace_count == 0:
                break
        config_json = ''.join(config_lines).replace('[config] ', '')
        config_dict = json.loads(config_json)
        config = Config(
            seed=config_dict.get('seed', 0),
            device=config_dict.get('device', 'unknown'),
            problem_size=config_dict.get('problem_size', 0),
            batch_size=config_dict.get('batch_size', 0),
            train_steps=config_dict.get('train_steps', 0),
            lr=config_dict.get('lr', 0.0),
            log_every=config_dict.get('log_every', 1),
            eval_every=config_dict.get('eval_every', 1),
            init_zoo=config_dict.get('init_zoo', []),
            iter_zoo=config_dict.get('iter_zoo', []),
            baseline=config_dict.get('baseline', 'unknown'),
            critic_coef=config_dict.get('critic_coef', 0.0),
            rrc_steps=config_dict.get('rrc_steps', 0),
            two_opt_iters=config_dict.get('two_opt_iters', 0),
        )

    # 解析训练步骤
    for line in lines:
        match = train_pattern.search(line)
        if match:
            gate1_dist = parse_distribution(match.group(9))
            gate2_dist = parse_distribution(match.group(10))

            train_steps.append(TrainStep(
                step=int(match.group(1)),
                loss=float(match.group(2)),
                len0=float(match.group(3)),
                len1=float(match.group(4)),
                adv_mean=float(match.group(5)),
                adv_std=float(match.group(6)),
                critic_mse=float(match.group(7)),
                entropy=float(match.group(8)),
                gate1_dist=gate1_dist,
                gate2_dist=gate2_dist,
            ))

    # 解析评估步骤
    for line in lines:
        match = eval_pattern.search(line)
        if match:
            gate1_dist = parse_distribution(match.group(4))
            gate2_dist = parse_distribution(match.group(5))

            eval_steps.append(EvalStep(
                step=int(match.group(1)),
                mean_len=float(match.group(2)),
                best_len=float(match.group(3)),
                gate1_dist=gate1_dist,
                gate2_dist=gate2_dist,
            ))

    return LogData(config=config, train_steps=train_steps, eval_steps=eval_steps)


def parse_distribution(dist_str: str) -> Dict[str, float]:
    """解析分布字符串，如 'lehd:0.36, elg:0.14, difusco:0.50'"""
    result = {}
    for item in dist_str.split(', '):
        if ':' in item:
            key, value = item.split(':')
            result[key] = float(value)
    return result


def print_summary(data: LogData):
    """打印训练摘要"""
    print("\n" + "="*60)
    print("训练配置摘要")
    print("="*60)
    print(f"问题规模: N={data.config.problem_size}")
    print(f"训练步数: {data.config.train_steps}")
    print(f"Batch大小: {data.config.batch_size}")
    print(f"学习率: {data.config.lr}")
    print(f"Baseline模式: {data.config.baseline}")
    print(f"Critic系数: {data.config.critic_coef}")
    print(f"初始化器zoo: {data.config.init_zoo}")
    print(f"迭代器zoo: {data.config.iter_zoo}")
    print(f"RRC步数: {data.config.rrc_steps}")
    print(f"2-opt迭代数: {data.config.two_opt_iters}")

    print("\n" + "="*60)
    print("训练统计")
    print("="*60)
    print(f"训练步数记录: {len(data.train_steps)}")
    print(f"评估步数记录: {len(data.eval_steps)}")

    if data.train_steps:
        first = data.train_steps[0]
        last = data.train_steps[-1]
        print(f"\n初始状态 (step {first.step}):")
        print(f"  Loss: {first.loss:.4f}")
        print(f"  路径长度: len0={first.len0:.4f}, len1={first.len1:.4f}")
        print(f"  Entropy: {first.entropy:.4f}")
        print(f"  Gate1分布: {first.gate1_dist}")
        print(f"  Gate2分布: {first.gate2_dist}")

        print(f"\n最终状态 (step {last.step}):")
        print(f"  Loss: {last.loss:.4f}")
        print(f"  路径长度: len0={last.len0:.4f}, len1={last.len1:.4f}")
        print(f"  Entropy: {last.entropy:.4f}")
        print(f"  Gate1分布: {last.gate1_dist}")
        print(f"  Gate2分布: {last.gate2_dist}")

        print(f"\n训练改善:")
        print(f"  Loss: {first.loss:.4f} → {last.loss:.4f} ({(last.loss-first.loss)/first.loss*100:+.1f}%)")
        print(f"  Len0: {first.len0:.4f} → {last.len0:.4f} ({(last.len0-first.len0)/first.len0*100:+.1f}%)")
        print(f"  Len1: {first.len1:.4f} → {last.len1:.4f} ({(last.len1-first.len1)/first.len1*100:+.1f}%)")
        print(f"  Entropy: {first.entropy:.4f} → {last.entropy:.4f} (探索度{'下降' if last.entropy < first.entropy else '上升'})")

    if data.eval_steps:
        first_eval = data.eval_steps[0]
        last_eval = data.eval_steps[-1]
        best_eval = min(data.eval_steps, key=lambda x: x.mean_len)

        print(f"\n评估集性能:")
        print(f"  初始评估 (step {first_eval.step}): mean_len={first_eval.mean_len:.4f}")
        print(f"  最终评估 (step {last_eval.step}): mean_len={last_eval.mean_len:.4f}")
        print(f"  最佳评估 (step {best_eval.step}): mean_len={best_eval.mean_len:.4f}")
        print(f"  总改善: {(first_eval.mean_len - best_eval.mean_len)/first_eval.mean_len*100:.2f}%")


def plot_training_curves(data: LogData, output_path: Optional[str] = None):
    """绘制训练曲线"""
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)

    # 提取数据
    steps = [s.step for s in data.train_steps]
    losses = [s.loss for s in data.train_steps]
    len0s = [s.len0 for s in data.train_steps]
    len1s = [s.len1 for s in data.train_steps]
    adv_means = [s.adv_mean for s in data.train_steps]
    adv_stds = [s.adv_std for s in data.train_steps]
    critic_mses = [s.critic_mse for s in data.train_steps]
    entropies = [s.entropy for s in data.train_steps]

    # 1. Loss曲线
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(steps, losses, 'b-', linewidth=1, alpha=0.7)
    ax1.set_xlabel('Step')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss')
    ax1.grid(True, alpha=0.3)

    # 2. 路径长度曲线
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(steps, len0s, 'g-', linewidth=1, alpha=0.7, label='Len0 (init)')
    ax2.plot(steps, len1s, 'r-', linewidth=1, alpha=0.7, label='Len1 (iter)')
    ax2.set_xlabel('Step')
    ax2.set_ylabel('Path Length')
    ax2.set_title('Path Length')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. 优势函数
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(steps, adv_means, 'purple', linewidth=1, alpha=0.7, label='Mean')
    ax3.fill_between(steps,
                     np.array(adv_means) - np.array(adv_stds),
                     np.array(adv_means) + np.array(adv_stds),
                     alpha=0.3, label='±1 Std')
    ax3.axhline(y=0, color='k', linestyle='--', linewidth=0.5)
    ax3.set_xlabel('Step')
    ax3.set_ylabel('Advantage')
    ax3.set_title('Advantage Function')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. Critic MSE
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.plot(steps, critic_mses, 'orange', linewidth=1, alpha=0.7)
    ax4.set_xlabel('Step')
    ax4.set_ylabel('MSE')
    ax4.set_title('Critic MSE Loss')
    ax4.grid(True, alpha=0.3)

    # 5. Entropy
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.plot(steps, entropies, 'brown', linewidth=1, alpha=0.7)
    ax5.set_xlabel('Step')
    ax5.set_ylabel('Entropy')
    ax5.set_title('Policy Entropy (Exploration)')
    ax5.grid(True, alpha=0.3)

    # 6. 评估集性能
    eval_steps = [s.step for s in data.eval_steps]
    eval_mean_lens = [s.mean_len for s in data.eval_steps]

    ax6 = fig.add_subplot(gs[1, 2])
    ax6.plot(eval_steps, eval_mean_lens, 'navy', marker='o', linewidth=2, markersize=4)
    ax6.set_xlabel('Step')
    ax6.set_ylabel('Mean Length')
    ax6.set_title('Evaluation Performance')
    ax6.grid(True, alpha=0.3)

    # 7. Gate1选择分布演化
    ax7 = fig.add_subplot(gs[2, 0])
    plot_gate_evolution(ax7, data.train_steps, 'gate1_dist',
                        f'Gate1 Distribution (Initializer Selection)\nFinal: {list(data.train_steps[-1].gate1_dist.keys())}')

    # 8. Gate2选择分布演化
    ax8 = fig.add_subplot(gs[2, 1])
    plot_gate_evolution(ax8, data.train_steps, 'gate2_dist',
                        f'Gate2 Distribution (Iterator Selection)\nFinal: {list(data.train_steps[-1].gate2_dist.keys())}')

    # 9. 长度改善对比
    ax9 = fig.add_subplot(gs[2, 2])
    ax9.plot(eval_steps, eval_mean_lens, 'navy', marker='o', linewidth=2, markersize=4, label='Eval Mean')
    # 绘制训练集的滑动平均
    window = max(1, len(steps) // 50)
    if len(len1s) >= window:
        from scipy.ndimage import uniform_filter1d
        smoothed = uniform_filter1d(len1s, size=window)
        ax9.plot(steps, smoothed, 'r-', linewidth=1, alpha=0.5, label=f'Train (smoothed, window={window})')
    ax9.set_xlabel('Step')
    ax9.set_ylabel('Length')
    ax9.set_title('Train vs Eval Performance')
    ax9.legend()
    ax9.grid(True, alpha=0.3)

    fig.suptitle(f'Training Analysis - N={data.config.problem_size}, Seed={data.config.seed}',
                 fontsize=14, fontweight='bold')

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\n图表已保存到: {output_path}")
    else:
        plt.show()

    plt.close()


def plot_gate_evolution(ax, train_steps: List[TrainStep], dist_attr: str, title: str):
    """绘制Gate选择分布演化"""
    # 收集所有唯一的选项
    all_options = set()
    for step in train_steps:
        all_options.update(getattr(step, dist_attr).keys())
    all_options = sorted(all_options)

    # 为每个选项收集数据
    steps = [s.step for s in train_steps]
    option_data = {opt: [] for opt in all_options}

    for step in train_steps:
        dist = getattr(step, dist_attr)
        for opt in all_options:
            option_data[opt].append(dist.get(opt, 0.0))

    # 绘制堆叠面积图
    ax.stackplot(steps, *[option_data[opt] for opt in all_options],
                  labels=all_options, alpha=0.8)

    ax.set_xlabel('Step')
    ax.set_ylabel('Probability')
    ax.set_title(title, fontsize=10)
    ax.legend(loc='upper left', fontsize=8, ncol=2)
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3, axis='y')


def analyze_convergence(data: LogData):
    """分析收敛性"""
    print("\n" + "="*60)
    print("收敛性分析")
    print("="*60)

    if len(data.train_steps) < 10:
        print("数据不足，无法分析收敛性")
        return

    # 分析最后10%的训练步数
    n_tail = max(10, len(data.train_steps) // 10)
    tail_steps = data.train_steps[-n_tail:]

    # 计算方差
    loss_var = np.var([s.loss for s in tail_steps])
    len1_var = np.var([s.len1 for s in tail_steps])

    print(f"\n最后{n_tail}步的稳定性:")
    print(f"  Loss方差: {loss_var:.6f}")
    print(f"  Len1方差: {len1_var:.6f}")

    # 判断是否收敛
    if loss_var < 0.01 and len1_var < 0.1:
        print("  ✓ 看起来已经收敛")
    elif loss_var < 0.1 and len1_var < 0.5:
        print("  ~ 基本收敛，仍有轻微波动")
    else:
        print("  ✗ 尚未收敛或存在较大波动")

    # Gate分布稳定性
    print(f"\nGate分布稳定性:")
    first_gate1 = data.train_steps[0].gate1_dist
    last_gate1 = data.train_steps[-1].gate1_dist
    first_gate2 = data.train_steps[0].gate2_dist
    last_gate2 = data.train_steps[-1].gate2_dist

    print(f"  Gate1: {first_gate1} → {last_gate1}")
    print(f"  Gate2: {first_gate2} → {last_gate2}")

    # 检查是否坍缩到单个选项
    if len(last_gate1) == 1:
        print(f"  ⚠ Gate1完全坍缩到 '{list(last_gate1.keys())[0]}'")
    if len(last_gate2) == 1:
        print(f"  ⚠ Gate2完全坍缩到 '{list(last_gate2.keys())[0]}'")


def print_gate_selection_analysis(data: LogData):
    """分析Gate选择行为"""
    print("\n" + "="*60)
    print("Gate选择分析")
    print("="*60)

    # 分段分析（初期、中期、后期）
    n_steps = len(data.train_steps)
    segments = [
        (0, max(100, n_steps // 4), "初期"),
        (n_steps // 4, n_steps // 2, "中期"),
        (n_steps // 2, n_steps, "后期"),
    ]

    for start, end, name in segments:
        if start >= n_steps:
            continue
        segment = data.train_steps[start:end]

        # 统计Gate1选择
        gate1_totals = defaultdict(float)
        gate2_totals = defaultdict(float)

        for step in segment:
            for k, v in step.gate1_dist.items():
                gate1_totals[k] += v
            for k, v in step.gate2_dist.items():
                gate2_totals[k] += v

        # 归一化
        total1 = sum(gate1_totals.values())
        total2 = sum(gate2_totals.values())

        print(f"\n{name} (steps {segment[0].step}-{segment[-1].step}):")
        print(f"  Gate1平均分布:")
        for k in sorted(gate1_totals.keys()):
            print(f"    {k}: {gate1_totals[k]/total1:.3f}")

        print(f"  Gate2平均分布:")
        for k in sorted(gate2_totals.keys()):
            print(f"    {k}: {gate2_totals[k]/total2:.3f}")


def main():
    parser = argparse.ArgumentParser(description='分析双层强化学习TSP训练日志')
    parser.add_argument('--log_path', type=str, default='./outputs/two_gate_tsp_seed2024_n100.log.txt', help='日志文件路径')
    parser.add_argument('--output', '-o', type=str, default='./outputs/two_gate_tsp_seed2024_n100.png',
                        help='输出图表路径（默认显示）')
    parser.add_argument('--no-plot', action='store_true',
                        help='不生成图表，仅打印统计')

    args = parser.parse_args()

    # 解析日志
    print(f"正在解析日志文件: {args.log_path}")
    data = parse_log_file(args.log_path)

    # 打印摘要
    print_summary(data)

    # Gate选择分析
    print_gate_selection_analysis(data)

    # 收敛性分析
    analyze_convergence(data)

    # 绘制图表
    if not args.no_plot:
        print("\n正在生成训练曲线图表...")
        try:
            plot_training_curves(data, args.output)
        except ImportError as e:
            print(f"警告: 缺少必要的绘图库 ({e})")
            print("请安装: pip install matplotlib scipy numpy")
        except Exception as e:
            print(f"错误: 绘图失败 ({e})")


if __name__ == '__main__':
    main()
