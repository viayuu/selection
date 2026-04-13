from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from arm_config import get_arm_space_config
from baselines import OracleSelector, SingleBestGlobalSelector, SingleBestPerProblemSelector
from build_dataset import build_joint_dataset, build_nss_benchmark_dataset
import run_offline as ro


LOSS_TRIGGER_RE = re.compile(r"\|\s*t=(\d+)\s*\|\s*TRAIN\s*\|\s*module=representation")
LOSS_VALUE_RE = re.compile(
    r"loss:\s*avg=([0-9eE+\-.]+)\s*\|\s*min=([0-9eE+\-.]+)\s*\|\s*max=([0-9eE+\-.]+)\s*\|\s*last=([0-9eE+\-.]+)"
)


class FixedArmSelector:
    """固定选择某一个 arm，用来和 selector 做“单方法”对比。"""

    def __init__(self, arm: int):
        self.arm = int(arm)

    def select(self, sample) -> int:
        if sample.feasible_mask[self.arm] > 0:
            return int(self.arm)
        feasible = np.flatnonzero(sample.feasible_mask > 0)
        return int(feasible[0])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="用训练好的 checkpoint 在 NSS 的 TSPLIB/CVRPLIB 上做 benchmark 测试。")
    parser.add_argument(
        "--run-dir",
        type=str,
        required=True,
        help="训练输出目录，例如 2cmab2/outputs/2026-04-02-11-17-56_nss_neural_linucb_server",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="best.pt",
        help="使用哪个 checkpoint，默认 best.pt",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="测试时使用的设备，默认 cuda。",
    )
    parser.add_argument(
        "--output-dir-name",
        type=str,
        default="benchmark_eval_tsplib_cvrplib",
        help="评测结果输出到 run_dir 下的哪个子目录。",
    )
    return parser.parse_args()


def _load_checkpoint(checkpoint_path: Path) -> dict:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"找不到 checkpoint: {checkpoint_path}")
    return torch.load(checkpoint_path, map_location="cpu")


def _build_args_from_checkpoint(checkpoint: dict, device: str) -> SimpleNamespace:
    """
    从 checkpoint 恢复出重建 selector 所需的关键参数。

    这里和训练时保持同一套结构，避免：
    - 网络维度不一致
    - 是否 problem-specific heads 不一致
    - buffer/训练相关参数不一致
    """
    config = dict(checkpoint.get("config", {}))
    selector_state = dict(checkpoint.get("selector_state", {}))

    defaults = {
        "method": checkpoint.get("method", "neural_linucb"),
        "alpha": 1.0,
        "nu": 1.0,
        "reg": 1.0,
        "hidden_dim": 64,
        "arm_embed_dim": 16,
        "use_arm_onehot": False,
        "use_arm_bias": True,
        "lr": 1e-3,
        "train_every": 100,
        "representation_steps": 10,
        "representation_buffer_size": 5000,
        "representation_batch_size": 32,
        "representation_balance_mode": "problem_arm",
        "representation_balance_power": 0.5,
        "representation_loss_reweight": True,
        "linear_head_buffer_size": 10000,
        "initial_pulls": 1,
        "train_epsilon": 0.0,
        "problem_specific_heads": True,
        "freeze_encoder": False,
        "train_batch_size": 32,
        "train_steps_per_update": 10,
        "use_problem_arm_safety": True,
        "safety_min_pulls": 20,
        "safety_reward_gap": 0.1,
        "safety_penalty_scale": 1.0,
        "safety_penalty_cap": 2.0,
        "results_dir": "EasyNCO/results/test",
        "data_source": "nss",
        "nss_dataset_root": str(Path(__file__).resolve().parents[1] / "9nss论文" / "neural-solver-selection" / "datasets"),
        "tsp_dataset_path": "",
        "cvrp_dataset_path": "",
        "reward_mode": "linear_zero_one",
        "train_ratio": 0.7,
        "val_ratio": 0.15,
        "epochs": 1,
        "seed": 0,
        "max_samples_per_problem": 0,
        "device": device,
    }

    restored = dict(defaults)
    restored.update(config)
    restored.update(
        {
            "hidden_dim": selector_state.get("hidden_dim", restored["hidden_dim"]),
            "alpha": selector_state.get("alpha", restored["alpha"]),
            "reg": selector_state.get("reg", restored["reg"]),
            "train_every": selector_state.get("train_every", restored["train_every"]),
            "representation_steps": selector_state.get("representation_steps", restored["representation_steps"]),
            "representation_buffer_size": selector_state.get("representation_buffer_size", restored["representation_buffer_size"]),
            "representation_batch_size": selector_state.get("representation_batch_size", restored["representation_batch_size"]),
            "representation_balance_mode": selector_state.get("representation_balance_mode", restored["representation_balance_mode"]),
            "representation_balance_power": selector_state.get("representation_balance_power", restored["representation_balance_power"]),
            "representation_loss_reweight": selector_state.get("representation_loss_reweight", restored["representation_loss_reweight"]),
            "linear_head_buffer_size": selector_state.get("linear_head_buffer_size", restored["linear_head_buffer_size"]),
            "initial_pulls": selector_state.get("initial_pulls", restored["initial_pulls"]),
            "train_epsilon": selector_state.get("train_epsilon", restored["train_epsilon"]),
            "problem_specific_heads": selector_state.get("problem_specific_heads", restored["problem_specific_heads"]),
            "device": device,
        }
    )
    return SimpleNamespace(**restored)


def _resolve_split_indices(dataset, args: SimpleNamespace) -> tuple[list[int], list[int], list[int]]:
    if dataset.predefined_splits is not None:
        return (
            list(dataset.predefined_splits.get("train", [])),
            list(dataset.predefined_splits.get("val", [])),
            list(dataset.predefined_splits.get("test", [])),
        )
    return ro.stratified_split_indices(
        dataset.samples,
        train_ratio=float(args.train_ratio),
        val_ratio=float(args.val_ratio),
        seed=int(args.seed),
    )


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _distribution_rows(trace_rows: list[dict], arm_names: list[str]) -> list[dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    grouped["all"] = list(trace_rows)
    for row in trace_rows:
        grouped[str(row["problem"]).lower()].append(row)

    rows: list[dict] = []
    for scope, scope_rows in grouped.items():
        counter = Counter(int(row["selected_arm"]) for row in scope_rows)
        total = len(scope_rows)
        for arm_id, arm_name in enumerate(arm_names):
            count = int(counter.get(arm_id, 0))
            rows.append(
                {
                    "scope": scope,
                    "arm_id": arm_id,
                    "arm_name": arm_name,
                    "count": count,
                    "ratio": (count / total) if total > 0 else 0.0,
                }
            )
    return rows


def _plot_distribution(rows: list[dict], output_path: Path, title: str) -> None:
    scopes = ["all", "tsp", "cvrp"]
    fig, axes = plt.subplots(1, 3, figsize=(18, 4.8), sharey=True)
    for ax, scope in zip(axes, scopes):
        scope_rows = [row for row in rows if row["scope"] == scope]
        labels = [row["arm_name"] for row in scope_rows]
        ratios = [float(row["ratio"]) for row in scope_rows]
        bars = ax.bar(range(len(labels)), ratios, color="#1f77b4", alpha=0.9)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
        ax.set_title(scope.upper())
        ax.set_ylabel("Ratio")
        ax.grid(axis="y", alpha=0.25)
        ymax = max(ratios) if ratios else 0.0
        ax.set_ylim(0.0, max(0.05, ymax * 1.2))
        for bar, ratio in zip(bars, ratios):
            if ratio <= 0:
                continue
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height(),
                f"{ratio * 100:.1f}%",
                ha="center",
                va="bottom",
                fontsize=8,
                rotation=90,
            )
    fig.suptitle(title, fontsize=16)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_top1_vs_baselines(overall_rows: list[dict], by_problem_rows: list[dict], output_path_overall: Path, output_path_by_problem: Path) -> None:
    labels = [row["name"] for row in overall_rows]
    top1 = [float(row["top1_accuracy"]) for row in overall_rows]
    plt.figure(figsize=(8.4, 4.8))
    bars = plt.bar(range(len(labels)), top1, color=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"][: len(labels)], alpha=0.9)
    plt.xticks(range(len(labels)), labels, rotation=15, ha="right")
    plt.ylabel("Top1 Accuracy")
    plt.title("Benchmark Top1: Selector vs Baselines")
    plt.ylim(0.0, 1.08)
    plt.grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, top1):
        plt.text(bar.get_x() + bar.get_width() / 2.0, value, f"{value:.3f}", ha="center", va="bottom", fontsize=10)
    plt.tight_layout()
    plt.savefig(output_path_overall, dpi=220, bbox_inches="tight")
    plt.close()

    scopes = ["tsp", "cvrp"]
    methods = labels
    x = np.arange(len(methods))
    width = 0.36
    plt.figure(figsize=(9.2, 4.8))
    for idx, scope in enumerate(scopes):
        values = [float(next(row["top1_accuracy"] for row in by_problem_rows if row["scope"] == scope and row["name"] == method)) for method in methods]
        bars = plt.bar(x + (idx - 0.5) * width, values, width=width, label=scope.upper(), alpha=0.9)
        for bar, value in zip(bars, values):
            plt.text(bar.get_x() + bar.get_width() / 2.0, value, f"{value:.3f}", ha="center", va="bottom", fontsize=8, rotation=90)
    plt.xticks(x, methods, rotation=15, ha="right")
    plt.ylabel("Top1 Accuracy")
    plt.title("Benchmark Top1 by Problem")
    plt.ylim(0.0, 1.08)
    plt.grid(axis="y", alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path_by_problem, dpi=220, bbox_inches="tight")
    plt.close()


def _plot_single_method_compare(rows: list[dict], title: str, output_path: Path) -> None:
    selector_top1 = float(rows[0]["selector_top1_accuracy"]) if rows else 0.0
    labels = [row["arm_name"] for row in rows]
    method_values = [float(row["top1_accuracy"]) for row in rows]
    selector_values = [selector_top1 for _ in rows]
    x = np.arange(len(rows))
    width = 0.36

    plt.figure(figsize=(8.8, 4.8))
    bars1 = plt.bar(x - width / 2, method_values, width=width, label="single method", color="#ff7f0e", alpha=0.9)
    bars2 = plt.bar(x + width / 2, selector_values, width=width, label="selector", color="#1f77b4", alpha=0.9)
    plt.xticks(x, labels, rotation=30, ha="right")
    plt.ylabel("Top1 Accuracy")
    plt.title(title)
    plt.grid(axis="y", alpha=0.25)
    ymax = max(method_values + selector_values) if rows else 1.0
    plt.ylim(0.0, max(0.1, ymax * 1.22))
    for bar, value in zip(bars1, method_values):
        plt.text(bar.get_x() + bar.get_width() / 2.0, value, f"{value:.3f}", ha="center", va="bottom", fontsize=8, rotation=90)
    for bar, value in zip(bars2, selector_values):
        plt.text(bar.get_x() + bar.get_width() / 2.0, value, f"{value:.3f}", ha="center", va="bottom", fontsize=8, rotation=90)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close()


def _extract_loss_rows(log_path: Path) -> list[dict]:
    rows: list[dict] = []
    current_step: int | None = None
    if not log_path.exists():
        return rows
    for raw_line in log_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        trigger_match = LOSS_TRIGGER_RE.search(raw_line)
        if trigger_match:
            current_step = int(trigger_match.group(1))
            continue
        loss_match = LOSS_VALUE_RE.search(raw_line)
        if loss_match and current_step is not None:
            avg_loss, min_loss, max_loss, last_loss = map(float, loss_match.groups())
            rows.append(
                {
                    "train_step": current_step,
                    "avg_loss": avg_loss,
                    "min_loss": min_loss,
                    "max_loss": max_loss,
                    "last_loss": last_loss,
                }
            )
            current_step = None
    return rows


def _plot_loss_avg_only(loss_rows: list[dict], output_path: Path) -> None:
    xs = [row["train_step"] for row in loss_rows]
    ys = [row["avg_loss"] for row in loss_rows]
    plt.figure(figsize=(8.5, 4.8))
    plt.plot(xs, ys, linewidth=2.2, color="#1f77b4")
    plt.xlabel("Train Step")
    plt.ylabel("Average Loss")
    plt.title("Training Average Loss Curve")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close()


def _evaluate(selector, samples, indices: list[int], arm_names: list[str], stage: str) -> tuple[dict, list[dict]]:
    trace_rows: list[dict] = []
    metrics = ro.evaluate_selector(
        selector,
        samples,
        indices,
        decision_mode="greedy",
        arm_names=arm_names,
        trace_rows=trace_rows,
        stage=stage,
        show_progress=False,
        log_progress=True,
        progress_log_every=100,
    )
    return metrics, trace_rows


def _compare_single_methods(samples, indices: list[int], arm_names: list[str], selector_metrics: dict, problem: str) -> list[dict]:
    scoped_indices = [idx for idx in indices if samples[idx].problem == problem]
    feasible_arm_ids = sorted({int(arm_id) for idx in scoped_indices for arm_id in np.flatnonzero(samples[idx].feasible_mask > 0)})
    selector_problem_metrics = selector_metrics["by_problem"][problem]
    selector_mean_cost = float(selector_problem_metrics["mean_cost"])
    selector_top1 = float(selector_problem_metrics["top1_accuracy"])

    rows: list[dict] = []
    for arm_id in feasible_arm_ids:
        fixed_selector = FixedArmSelector(arm=arm_id)
        metrics = ro.evaluate_selector(fixed_selector, samples, scoped_indices, decision_mode="greedy")
        rows.append(
            {
                "problem": problem,
                "arm_id": int(arm_id),
                "arm_name": arm_names[arm_id],
                "mean_cost": float(metrics["overall"]["mean_cost"]),
                "top1_accuracy": float(metrics["overall"]["top1_accuracy"]),
                "selector_mean_cost": selector_mean_cost,
                "selector_top1_accuracy": selector_top1,
                "selector_beats_cost": bool(selector_mean_cost < float(metrics["overall"]["mean_cost"])),
                "selector_beats_top1": bool(selector_top1 > float(metrics["overall"]["top1_accuracy"])),
            }
        )
    rows.sort(key=lambda item: item["mean_cost"])
    return rows


def _make_top1_rows(selector_metrics: dict, sb_global_metrics: dict, sb_problem_metrics: dict, oracle_metrics: dict) -> tuple[list[dict], list[dict]]:
    overall_rows = [
        {
            "name": "selector_best_checkpoint",
            "top1_accuracy": float(selector_metrics["overall"]["top1_accuracy"]),
            "mean_cost": float(selector_metrics["overall"]["mean_cost"]),
        },
        {
            "name": "single_best_global",
            "top1_accuracy": float(sb_global_metrics["overall"]["top1_accuracy"]),
            "mean_cost": float(sb_global_metrics["overall"]["mean_cost"]),
        },
        {
            "name": "single_best_per_problem",
            "top1_accuracy": float(sb_problem_metrics["overall"]["top1_accuracy"]),
            "mean_cost": float(sb_problem_metrics["overall"]["mean_cost"]),
        },
        {
            "name": "oracle",
            "top1_accuracy": float(oracle_metrics["overall"]["top1_accuracy"]),
            "mean_cost": float(oracle_metrics["overall"]["mean_cost"]),
        },
    ]

    by_problem_rows: list[dict] = []
    for scope in ("tsp", "cvrp"):
        by_problem_rows.extend(
            [
                {
                    "scope": scope,
                    "name": "selector_best_checkpoint",
                    "top1_accuracy": float(selector_metrics["by_problem"][scope]["top1_accuracy"]),
                },
                {
                    "scope": scope,
                    "name": "single_best_global",
                    "top1_accuracy": float(sb_global_metrics["by_problem"][scope]["top1_accuracy"]),
                },
                {
                    "scope": scope,
                    "name": "single_best_per_problem",
                    "top1_accuracy": float(sb_problem_metrics["by_problem"][scope]["top1_accuracy"]),
                },
                {
                    "scope": scope,
                    "name": "oracle",
                    "top1_accuracy": float(oracle_metrics["by_problem"][scope]["top1_accuracy"]),
                },
            ]
        )
    return overall_rows, by_problem_rows


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir).resolve()
    checkpoint_path = run_dir / "checkpoints" / args.checkpoint
    output_dir = run_dir / args.output_dir_name
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = _load_checkpoint(checkpoint_path)
    restored_args = _build_args_from_checkpoint(checkpoint, device=args.device)
    checkpoint_step = int(checkpoint.get("global_step", 0))

    arm_names, problem_to_methods = get_arm_space_config("nss")

    print("[benchmark] 开始构建原始 NSS 训练数据，用于 single-best baseline")
    train_dataset = build_joint_dataset(
        results_dir=Path(restored_args.results_dir),
        tsp_dataset_path=Path(restored_args.tsp_dataset_path) if restored_args.tsp_dataset_path else None,
        cvrp_dataset_path=Path(restored_args.cvrp_dataset_path) if restored_args.cvrp_dataset_path else None,
        data_source="nss",
        nss_dataset_root=Path(restored_args.nss_dataset_root) if restored_args.nss_dataset_root else None,
        arm_names=arm_names,
        problem_to_methods=problem_to_methods,
        reward_mode=restored_args.reward_mode,
        max_samples_per_problem=int(restored_args.max_samples_per_problem),
        seed=int(restored_args.seed),
        show_progress=False,
    )
    train_idx, _val_idx, _test_idx = _resolve_split_indices(train_dataset, restored_args)
    print(f"[benchmark] 原始训练集大小: {len(train_idx)}")

    print("[benchmark] 开始构建 TSPLIB/CVRPLIB benchmark 数据集")
    benchmark_dataset = build_nss_benchmark_dataset(
        nss_dataset_root=Path(restored_args.nss_dataset_root),
        arm_names=arm_names,
        problem_to_methods=problem_to_methods,
        reward_mode=restored_args.reward_mode,
        show_progress=False,
    )
    benchmark_idx = list(benchmark_dataset.predefined_splits.get("test", []))
    print(f"[benchmark] benchmark 样本总数: {len(benchmark_idx)}")

    print("[benchmark] 开始恢复训练好的 selector")
    selector = ro.build_selector(restored_args, n_arms=len(benchmark_dataset.arm_names), arm_names=benchmark_dataset.arm_names)
    ro._restore_selector_from_checkpoint(checkpoint_path, selector)
    print(f"[benchmark] selector 恢复完成: checkpoint step={checkpoint_step}")

    print("[benchmark] 开始评测 selector")
    selector_metrics, selector_trace = _evaluate(
        selector,
        benchmark_dataset.samples,
        benchmark_idx,
        benchmark_dataset.arm_names,
        stage="benchmark_greedy",
    )

    print("[benchmark] 开始评测 baselines")
    single_best_global = SingleBestGlobalSelector.fit(train_dataset.samples, train_idx, train_dataset.arm_names)
    single_best_per_problem = SingleBestPerProblemSelector.fit(train_dataset.samples, train_idx, train_dataset.arm_names)
    oracle = OracleSelector()
    sb_global_metrics = ro.evaluate_selector(single_best_global, benchmark_dataset.samples, benchmark_idx, decision_mode="greedy")
    sb_problem_metrics = ro.evaluate_selector(single_best_per_problem, benchmark_dataset.samples, benchmark_idx, decision_mode="greedy")
    oracle_metrics = ro.evaluate_selector(oracle, benchmark_dataset.samples, benchmark_idx, decision_mode="greedy")

    print("[benchmark] 开始统计 benchmark 单方法对比")
    tsp_single_rows = _compare_single_methods(
        benchmark_dataset.samples,
        benchmark_idx,
        benchmark_dataset.arm_names,
        selector_metrics,
        problem="tsp",
    )
    cvrp_single_rows = _compare_single_methods(
        benchmark_dataset.samples,
        benchmark_idx,
        benchmark_dataset.arm_names,
        selector_metrics,
        problem="cvrp",
    )

    distribution_rows = _distribution_rows(selector_trace, benchmark_dataset.arm_names)
    overall_top1_rows, by_problem_top1_rows = _make_top1_rows(
        selector_metrics,
        sb_global_metrics,
        sb_problem_metrics,
        oracle_metrics,
    )

    print("[benchmark] 开始写出 benchmark 结果文件")
    _write_json(output_dir / "selector_benchmark.metrics.json", selector_metrics)
    _write_json(output_dir / "single_best_global.metrics.json", sb_global_metrics)
    _write_json(output_dir / "single_best_per_problem.metrics.json", sb_problem_metrics)
    _write_json(output_dir / "oracle.metrics.json", oracle_metrics)
    _write_csv(output_dir / "benchmark_arm_distribution.csv", distribution_rows)
    _write_csv(output_dir / "benchmark_top1_vs_baselines_overall.csv", overall_top1_rows)
    _write_csv(output_dir / "benchmark_top1_vs_baselines_by_problem.csv", by_problem_top1_rows)
    _write_csv(output_dir / "tsplib_single_method_compare.csv", tsp_single_rows)
    _write_csv(output_dir / "cvrplib_single_method_compare.csv", cvrp_single_rows)
    with (output_dir / "selector_benchmark.trace.jsonl").open("w", encoding="utf-8") as f:
        for row in selector_trace:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print("[benchmark] 开始画图")
    loss_rows = _extract_loss_rows(run_dir / "neural_linucb.log")
    if loss_rows:
        _write_csv(output_dir / "training_loss_curve.csv", loss_rows)
        _plot_loss_avg_only(loss_rows, output_dir / "training_loss_avg_curve.png")
    _plot_distribution(distribution_rows, output_dir / "benchmark_arm_distribution.png", title="Benchmark Arm Distribution")
    _plot_top1_vs_baselines(
        overall_top1_rows,
        by_problem_top1_rows,
        output_path_overall=output_dir / "benchmark_top1_vs_baselines.png",
        output_path_by_problem=output_dir / "benchmark_top1_by_problem.png",
    )
    _plot_single_method_compare(
        tsp_single_rows,
        title="TSPLIB Top1: Selector vs Single Methods",
        output_path=output_dir / "tsplib_top1_vs_single_methods.png",
    )
    _plot_single_method_compare(
        cvrp_single_rows,
        title="CVRPLIB Top1: Selector vs Single Methods",
        output_path=output_dir / "cvrplib_top1_vs_single_methods.png",
    )

    summary = {
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_step": checkpoint_step,
        "selector_metrics": selector_metrics,
        "single_best_global_metrics": sb_global_metrics,
        "single_best_per_problem_metrics": sb_problem_metrics,
        "oracle_metrics": oracle_metrics,
        "overall_top1_rows": overall_top1_rows,
        "by_problem_top1_rows": by_problem_top1_rows,
        "tsplib_single_method_rows": tsp_single_rows,
        "cvrplib_single_method_rows": cvrp_single_rows,
    }
    _write_json(output_dir / "summary.json", summary)

    summary_lines = [
        f"checkpoint: {checkpoint_path}",
        f"checkpoint_step: {checkpoint_step}",
        "",
        f"benchmark overall top1(selector): {float(selector_metrics['overall']['top1_accuracy']):.6f}",
        f"benchmark overall top1(single_best_global): {float(sb_global_metrics['overall']['top1_accuracy']):.6f}",
        f"benchmark overall top1(single_best_per_problem): {float(sb_problem_metrics['overall']['top1_accuracy']):.6f}",
        f"benchmark overall top1(oracle): {float(oracle_metrics['overall']['top1_accuracy']):.6f}",
        "",
        f"TSPLIB top1(selector): {float(selector_metrics['by_problem']['tsp']['top1_accuracy']):.6f}",
        f"CVRPLIB top1(selector): {float(selector_metrics['by_problem']['cvrp']['top1_accuracy']):.6f}",
    ]
    (output_dir / "summary.txt").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    print(f"[benchmark] 完成，结果目录: {output_dir}")


if __name__ == "__main__":
    main()
