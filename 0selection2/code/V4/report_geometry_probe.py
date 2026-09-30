"""Verify the paired R37 protocol, replay predictions, and compare endpoints."""

import json
from pathlib import Path

import numpy as np
import torch

from .geometry_probe import METRICS, ROOT, metrics_from_logits
from .multitask_probe import dump
from .tsp_learnability import gap_and_regret, raw_tsp, write_csv


def load_history(root, result):
    return json.loads((root / result["name"] / "history.json").read_text())


def verify(root, results, raw):
    checks = []
    pairs = {}
    for seed in sorted({r["seed"] for r in results}):
        pair = sorted([r for r in results if r["seed"] == seed], key=lambda r: r["mode"], reverse=True)
        assert {r["mode"] for r in pair} == {"zero", "real"}
        configs = [json.loads((root / r["name"] / "args.json").read_text()) for r in pair]
        keys = ("model_initial_hash", "new_branch_initial_hash", "batch_hash", "dropout_rng_initial_hash", "geometry_stats")
        assert all(configs[0][k] == configs[1][k] for k in keys)
        assert all(c["optimizer_initial_state_count"] == 0 and not c["optimizer_restored"] for c in configs)
        traces = [[json.loads(line) for line in (root / r["name"] / "update_trace.jsonl").read_text().splitlines()] for r in pair]
        assert len(traces[0]) == len(traces[1]) == pair[0]["updates"]
        for left, right in zip(*traces):
            assert all(left[k] == right[k] for k in ("updates", "batch_hash", "dropout_rng_before", "dropout_rng_after", "lr_factor"))
        pairs[str(seed)] = dict(initial_weights_equal=True, batch_indices_equal=True,
                                all_successful_update_dropout_rng_equal=True, lr_schedule_equal=True, optimizers_fresh=True)
    for result in results:
        history = load_history(root, result)
        assert [h["updates"] for h in history] == list(range(0, result["updates"] + 1, 50))
        directory = root / result["name"]
        checkpoint = torch.load(directory / "last.pt", map_location="cpu", weights_only=False)
        assert checkpoint["updates"] == result["updates"]
        assert all(int(s["step"]) == result["updates"] for s in checkpoint["optimizer"]["state"].values())
        for split in ("train", "val"):
            for row in history:
                with np.load(directory / f"predictions_{split}_u{row['updates']:04d}.npz") as p:
                    np.testing.assert_array_equal(p["indices"], np.arange(len(raw[split]["winner"])))
                    np.testing.assert_array_equal(p["winner"], raw[split]["winner"])
                    metrics, pred = metrics_from_logits(torch.from_numpy(p["logits"]), raw[split])
                    np.testing.assert_array_equal(pred, p["pred"])
                for key in METRICS:
                    np.testing.assert_allclose(metrics[key], row[split][key], rtol=0, atol=1e-10)
                checks.append(dict(run=result["name"], split=split, updates=row["updates"]))
    dump(root / "prediction_replay.json", dict(paired_protocol=pairs, evaluations_replayed=len(checks),
                                               checkpoint_optimizer_steps_verified=True, test_read=False))


def comparison_rows(results):
    rows = []
    for result in results:
        for point in ("final", "last5"):
            for split in ("train", "val"):
                rows.append(dict(run=result["name"], seed=result["seed"], mode=result["mode"], summary=point,
                                 split=split, **{k: result[point][split][k] for k in METRICS}))
    return rows


def paired_bootstrap(root, results, raw):
    """Resample common validation instances, retaining all seed predictions."""
    winner, costs = raw["winner"], raw["costs"]
    def outcomes(pred):
        return np.stack([(pred == winner).astype(float), costs[np.arange(len(pred)), pred],
                         gap_and_regret(costs, pred)[1]], axis=1)
    values = {}
    for mode in ("zero", "real"):
        predictions = []
        for r in results:
            if r["mode"] == mode:
                with np.load(root / r["name"] / f"predictions_val_u{r['updates']:04d}.npz") as p:
                    predictions.append(outcomes(p["pred"]))
        values[mode] = np.mean(predictions, axis=0)
    with np.load(root / "baseline_val.npz") as p:
        values["R34"] = outcomes(p["pred"])
    rng = np.random.default_rng(37)
    indices = rng.integers(0, len(winner), size=(5000, len(winner)))
    rows = []
    for reference in ("zero", "R34"):
        delta = values["real"] - values[reference]
        intervals = np.quantile(delta[indices].mean(1), [.025, .975], axis=0)
        for i, metric in enumerate(("top1", "mean_cost", "actual_regret_pct")):
            rows.append(dict(comparison=f"B - {'A' if reference == 'zero' else 'R34'}", metric=metric,
                             mean_difference=float(delta[:, i].mean()), ci95_low=float(intervals[0, i]),
                             ci95_high=float(intervals[1, i])))
    write_csv(root / "paired_bootstrap.csv", rows)
    return rows


def plot(root, results, baseline):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for metric in METRICS:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
        for axis, split in zip(axes, ("train", "val")):
            for mode, label, color in (("zero", "A: redundant input", "#be5725"), ("real", "B: local geometry", "#137d9b")):
                histories = [load_history(root, r) for r in results if r["mode"] == mode]
                x = [h["updates"] for h in histories[0]]
                values = np.array([[h[split][metric] for h in history] for history in histories])
                for seed_curve in values:
                    axis.plot(x, seed_curve, color=color, alpha=.2, linewidth=1)
                axis.plot(x, values.mean(0), color=color, linewidth=2, label=label + " (seed mean)")
            axis.axhline(baseline[split][metric], color="black", linestyle="--", linewidth=1, label="R34 frozen reference")
            axis.axvspan(800, 1000, color="gray", alpha=.06)
            axis.set(xlabel="Successful TSP updates", ylabel=metric, title=f"TSP {split}")
            axis.grid(alpha=.2)
            axis.legend(fontsize=8)
        fig.savefig(root / f"comparison_{metric}.png", dpi=160)
        plt.close(fig)


def write_report(root, results, baseline, rows, intervals):
    differences = []
    seeds = sorted({r["seed"] for r in results})
    for seed in seeds:
        a = next(r for r in results if r["mode"] == "zero" and r["seed"] == seed)
        b = next(r for r in results if r["mode"] == "real" and r["seed"] == seed)
        for point in ("final", "last5"):
            for split in ("train", "val"):
                differences.append(dict(seed=seed, summary=point, split=split,
                                        **{k: b[point][split][k] - a[point][split][k] for k in METRICS}))
    write_csv(root / "paired_differences.csv", differences)
    deltas = [r for r in differences if r["summary"] == "final" and r["split"] == "val"]
    final_improved = all(r["top1"] > 0 and r["mean_cost"] < 0 and r["actual_regret_pct"] < 0 for r in deltas)
    late_deltas = [r for r in differences if r["summary"] == "last5" and r["split"] == "val"]
    stable = final_improved and all(r["top1"] > 0 and r["mean_cost"] < 0 and r["actual_regret_pct"] < 0 for r in late_deltas)
    text = ["# R37：TSP 多尺度局部几何对照", "", "## 实验范围", "",
            "R34 仍为18任务主基线。此处只做 TSP train/val 消融，未读取 test，不用 R36 的过拟合权重。",
            "A/B 均从 R34 第50轮 best.pt 出发，加入同一个21→64→128残差 MLP；最后一层零初始化，无额外乘法 gate。",
            "A 使用原8维节点属性、规模、标准化后置零的12维几何；B 使用相同输入中的真实几何。solver 编码、距离 attention bias、双流交互、Decoder 和 winner-cost 目标不变。",
            "每组 seed2/3/4；自然分布采样，batch640，每组1000次成功更新，完整 train/val 每50次评估。两组原模型 dropout 随机流和批次成对一致，AMP跳步重试同一批次。",
            "fresh AdamW，不恢复 moments；原参数 LR=2e-5，新增分支=2e-4，WD=1e-4，原 dropout0.1、新分支无dropout；50步 warmup，之后 cosine 到10%。沿用 R34 AMP/SDPA、梯度裁剪1.0。",
            "这是同一预训练起点的续训重复，不是独立从头训练。主比较为1000步与最后五个评估点（800/850/900/950/1000），不按最高点选择。", "",
            "## 固定 R34 参考", "", "| Split | CE | Top1 | mean_cost | actual regret | gap>0.1% Top1 | gap>0.5% Top1 |",
            "|---|---:|---:|---:|---:|---:|---:|"]
    for split, r in baseline.items():
        text.append(f"| {split} | {r['ce']:.6f} | {r['top1']:.2%} | {r['mean_cost']:.6f} | {r['actual_regret_pct']:.4f}% | "
                    f"{r['gap_gt_0_1_top1']:.2%} | {r['gap_gt_0_5_top1']:.2%} |")
    for point, title in (("final", "第1000次更新"), ("last5", "最后五点评估均值")):
        text.extend(["", f"## {title}", "", "| 模式 | seed | split | CE | Top1 | mean_cost | actual regret | gap>0.1% Top1 | gap>0.5% Top1 |",
                     "|---|---:|---|---:|---:|---:|---:|---:|---:|"])
        for row in rows:
            if row["summary"] == point:
                label = "A 冗余输入" if row["mode"] == "zero" else "B 局部几何"
                text.append(f"| {label} | {row['seed']} | {row['split']} | {row['ce']:.6f} | {row['top1']:.2%} | {row['mean_cost']:.6f} | "
                            f"{row['actual_regret_pct']:.4f}% | {row['gap_gt_0_1_top1']:.2%} | {row['gap_gt_0_5_top1']:.2%} |")
        text.extend(["", "### 三种续训 seed 汇总", "", "| 模式 | split | Top1 mean±std | mean_cost mean±std | actual regret mean±std |",
                     "|---|---|---:|---:|---:|"])
        for mode in ("zero", "real"):
            for split in ("train", "val"):
                values = [r[point][split] for r in results if r["mode"] == mode]
                stats = {k: (np.mean([v[k] for v in values]), np.std([v[k] for v in values], ddof=1)) for k in ("top1", "mean_cost", "actual_regret_pct")}
                text.append(f"| {'A' if mode == 'zero' else 'B'} | {split} | {stats['top1'][0]:.2%}±{stats['top1'][1]:.2%} | "
                            f"{stats['mean_cost'][0]:.6f}±{stats['mean_cost'][1]:.6f} | {stats['actual_regret_pct'][0]:.4f}±{stats['actual_regret_pct'][1]:.4f}% |")
    text.extend(["", "## 成对验证差值：B−A", "", "Top1 单位为百分点；cost/actual regret 越负越好。", "",
                 "| summary | seed | ΔTop1 (pp) | Δmean_cost | Δactual regret (pp) | Δgap>0.1 Top1 (pp) | Δgap>0.5 Top1 (pp) |",
                 "|---|---:|---:|---:|---:|---:|---:|"])
    for r in differences:
        if r["split"] == "val":
            text.append(f"| {r['summary']} | {r['seed']} | {100*r['top1']:+.3f} | {r['mean_cost']:+.6f} | {r['actual_regret_pct']:+.4f} | "
                        f"{100*r['gap_gt_0_1_top1']:+.3f} | {100*r['gap_gt_0_5_top1']:+.3f} |")
    text.extend(["", "## 判断", ""])
    if stable:
        text.append("相对 A，B 在全部续训 seed 的最终点和末五点均值上，验证 Top1 更高且成本、actual regret 的点估计更低。这是成对方向一致性，不等同于已确认稳定降成本；下方 bootstrap 的成本区间仍需单独检查。")
    else:
        text.append("本轮没有同时满足‘各 seed 的最终点及末五点验证 Top1、成本、actual regret 均优于 A’的稳定收益标准。是否局部改善应结合成对差值和高 gap 指标判断；不把某个最高点当成成功，也不能由此否定所有几何建模。")
    for point in ("final", "last5"):
        selected = [r for r in differences if r["summary"] == point and r["split"] == "val"]
        text.append(f"{point}：平均 ΔTop1={np.mean([r['top1'] for r in selected])*100:+.3f} pp，"
                    f"Δmean_cost={np.mean([r['mean_cost'] for r in selected]):+.6f}，"
                    f"Δactual regret={np.mean([r['actual_regret_pct'] for r in selected]):+.4f} pp。")
    text.extend(["", "### 与固定 R34 的区别", ""])
    for point in ("final", "last5"):
        values = [r[point]["val"] for r in results if r["mode"] == "real"]
        delta = {k: float(np.mean([v[k] for v in values]) - baseline["val"][k]) for k in METRICS}
        text.append(f"B−R34 ({point})：Top1={100*delta['top1']:+.3f} pp，mean_cost={delta['mean_cost']:+.6f}，"
                    f"actual regret={delta['actual_regret_pct']:+.4f} pp，CE={delta['ce']:+.6f}，"
                    f"gap>0.5% Top1={100*delta['gap_gt_0_5_top1']:+.3f} pp。")
    text.append("不能把修复 A 的续训退化视为同等幅度地超过 R34。B 相对 R34 的成本与 actual regret 没有改善，验证 CE 也未下降；大 gap 桶的 Top1 增益较小。局部几何对匹配对照的 Top1 有增量信号，但仍不足以替换18任务主基线。下一步若继续此方向，应只带回这组特征做18任务受控复验，重点检验成本和高损失桶，不同时叠加更多字段。")
    text.extend(["", "### 第1000步的成对实例 bootstrap", "",
                 "对1000个共同验证实例重采样5000次，保留同一实例上的所有续训 seed，并先跨 seed 平均。区间只反映本验证集实例抽样不确定性，不覆盖新的训练 seed、分布偏移或 solver 标签稳定性。",
                 "下表 Top1 已转换为百分点；actual regret 也为百分点，mean_cost 保持原始单位。95%区间包含0时不能仅凭单次点估计断言稳定泛化收益。", "",
                 "| 对照 | 指标 | 均值差 | 95%区间 |", "|---|---|---:|---|"])
    for r in intervals:
        scale = 100 if r["metric"] == "top1" else 1
        text.append(f"| {r['comparison']} | {r['metric']} | {scale*r['mean_difference']:+.6f} | "
                    f"[{scale*r['ci95_low']:+.6f}, {scale*r['ci95_high']:+.6f}] |")
    text.extend(["", "## 曲线", "", "![CE](comparison_ce.png)", "", "![Top1](comparison_top1.png)", "",
                 "![Actual regret](comparison_actual_regret_pct.png)", "", "![Mean cost](comparison_mean_cost.png)", "",
                 "![gap>0.1 Top1](comparison_gap_gt_0_1_top1.png)", "", "![gap>0.5 Top1](comparison_gap_gt_0_5_top1.png)", "",
                 "## 精度与实现核验", "",
                 "12维字段及标准化参数见 geometry_stats.json。k=4/8/16，各自为半径、均距、距离CV、包含中心的协方差各向异性。",
                 "几何用 FP64 计算，排除自连接/padding，包含 kth 边界上的全部并列邻居；仅 train 有效节点拟合 mean/std，val复用并保存到 checkpoint buffer。",
                 "模型与监督隔离：几何只由坐标/mask计算；真实 costs/gap 不进入特征或模型，gap只用于损失及事后分桶。",
                 "损失保留 R34 FP32 native-ind 口径；mean_cost、actual regret 和固定 gap 桶直接由 raw_label.pkl 原始 FP64 成本计算。",
                 "prediction_replay.json 记录全部评估重算、初始权重/批次/dropout RNG逐更新一致和 Adam 实际步数核验；verification.json 记录代码/输入文件哈希。",
                 "checkpoint_replay.json 记录六份最终 checkpoint 在 GPU 上 strict 加载、完整 val 预测重放，以及不提供几何缓存时的在线计算一致性核验。",
                 "每个实验目录含 args/history/train.log/last.pt/逐实例预测/update_trace.jsonl。W&B 为 offline 记录，未上传云端。",
                 "source_before.tar.gz 是改动前备份；source.tar.gz 是启动时实验源码；source_final.tar.gz 包含最终分析脚本。", "",
                 "参考方向：[PointNet++](https://arxiv.org/abs/1706.02413) 支持研究多尺度局部结构，但本轮是手工几何残差消融，不是复现 PointNet++，论文结果不保证 selector 提升。", ""])
    (root / "comparison.md").write_text("\n".join(text))
    cost_ci = next(r for r in intervals if r["comparison"] == "B - A" and r["metric"] == "mean_cost")
    dump(root / "conclusion.json", dict(vs_A_directional_consistency_all_seeds=stable,
                                       vs_A_cost_improvement_ci_excludes_zero=cost_ci["ci95_high"] < 0,
                                       main_baseline="R34", test_read=False,
                                       paired_val_differences=[r for r in differences if r["split"] == "val"]))


def main():
    torch.set_num_threads(1)
    results = json.loads((ROOT / "results.json").read_text())
    baseline = json.loads((ROOT / "baseline.json").read_text())
    raw = {s: raw_tsp(s) for s in ("train", "val")}
    verify(ROOT, results, raw)
    rows = comparison_rows(results)
    write_csv(ROOT / "comparison.csv", rows)
    intervals = paired_bootstrap(ROOT, results, raw["val"])
    plot(ROOT, results, baseline)
    write_report(ROOT, results, baseline, rows, intervals)
    print("R37 paired protocol/predictions verified; report and curves:", ROOT)


if __name__ == "__main__":
    main()
