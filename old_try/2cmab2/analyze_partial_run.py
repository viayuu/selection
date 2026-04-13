from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from arm_config import get_arm_space_config
from baselines import OracleSelector, SingleBestGlobalSelector, SingleBestPerProblemSelector
from build_dataset import build_joint_dataset
import run_offline as ro


LOSS_TRIGGER_RE = re.compile(r"\|\s*t=(\d+)\s*\|\s*TRAIN\s*\|\s*module=representation")
LOSS_VALUE_RE = re.compile(
    r"loss:\s*avg=([0-9eE+\-.]+)\s*\|\s*min=([0-9eE+\-.]+)\s*\|\s*max=([0-9eE+\-.]+)\s*\|\s*last=([0-9eE+\-.]+)"
)


@dataclass
class FixedArmSelector:
    """
    固定方法 baseline。

    作用：
    - 用来回答“如果在某个问题上始终固定选一个方法，会得到什么效果”
    - 这样就能直接和当前 selector 做一对一比较
    """

    arm: int

    def select(self, sample) -> int:
        if sample.feasible_mask[self.arm] > 0:
            return int(self.arm)
        feasible = np.flatnonzero(sample.feasible_mask > 0)
        return int(feasible[0])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="分析当前 run 的部分结果，并输出图表与汇总。")
    parser.add_argument(
        "--run-dir",
        type=str,
        required=True,
        help="某次 run 的输出目录，例如 2cmab2/outputs/2026-04-02-11-17-56_nss_neural_linucb_server",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="best.pt",
        help="要分析的 checkpoint 文件名，默认用 best.pt",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="分析时使用的设备；为了不打扰训练，默认使用 cpu。",
    )
    parser.add_argument(
        "--analysis-dir-name",
        type=str,
        default="analysis_partial",
        help="分析结果输出到 run_dir 下的哪个子目录。",
    )
    return parser.parse_args()


def _load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _write_json(path: Path, payload: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict]) -> None:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _safe_float(value) -> float:
    return float(value) if value is not None else float("nan")


def _format_pct(value: float) -> str:
    return f"{100.0 * float(value):.2f}%"


def _read_checkpoint(run_dir: Path, checkpoint_name: str) -> dict:
    checkpoint_path = run_dir / "checkpoints" / checkpoint_name
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"找不到 checkpoint: {checkpoint_path}")
    return torch.load(checkpoint_path, map_location="cpu")


def _build_args_from_checkpoint(checkpoint: dict, device: str) -> SimpleNamespace:
    """
    根据 checkpoint 还原 selector 构造参数。

    这里优先使用 checkpoint 中的 config，
    再用 selector_state 补齐真正和模型结构相关的字段。
    """

    config = dict(checkpoint.get("config", {}))
    selector_state = dict(checkpoint.get("selector_state", {}))

    # 这些默认值只是兜底，正常情况下会被 checkpoint 里的真实配置覆盖。
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
            "representation_buffer_size": selector_state.get(
                "representation_buffer_size",
                restored["representation_buffer_size"],
            ),
            "representation_batch_size": selector_state.get(
                "representation_batch_size",
                restored["representation_batch_size"],
            ),
            "representation_balance_mode": selector_state.get(
                "representation_balance_mode",
                restored["representation_balance_mode"],
            ),
            "representation_balance_power": selector_state.get(
                "representation_balance_power",
                restored["representation_balance_power"],
            ),
            "representation_loss_reweight": selector_state.get(
                "representation_loss_reweight",
                restored["representation_loss_reweight"],
            ),
            "linear_head_buffer_size": selector_state.get(
                "linear_head_buffer_size",
                restored["linear_head_buffer_size"],
            ),
            "initial_pulls": selector_state.get("initial_pulls", restored["initial_pulls"]),
            "train_epsilon": selector_state.get("train_epsilon", restored["train_epsilon"]),
            "problem_specific_heads": selector_state.get(
                "problem_specific_heads",
                restored["problem_specific_heads"],
            ),
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


def _parse_loss_rows(log_path: Path) -> list[dict]:
    """
    从详细训练日志中提取表示层训练的 loss。

    每次出现：
    - `t=xxx | TRAIN | module=representation`
    紧接着就会跟一行：
    - `loss: avg=... | min=... | max=... | last=...`
    """

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
            avg_loss, min_loss, max_loss, last_loss = loss_match.groups()
            rows.append(
                {
                    "train_step": current_step,
                    "avg_loss": float(avg_loss),
                    "min_loss": float(min_loss),
                    "max_loss": float(max_loss),
                    "last_loss": float(last_loss),
                }
            )
            current_step = None
    return rows


def _plot_loss(loss_rows: list[dict], output_path: Path, best_step: int) -> None:
    plt.figure(figsize=(8, 4.5))
    xs = [row["train_step"] for row in loss_rows]
    ys = [row["avg_loss"] for row in loss_rows]
    plt.plot(xs, ys, marker="o", linewidth=1.8, markersize=3.5)
    plt.axvline(best_step, color="tab:red", linestyle="--", linewidth=1.2, label=f"best step = {best_step}")
    plt.xlabel("Train Step")
    plt.ylabel("Average Loss")
    plt.title("Average Loss of Representation Update")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=180)
    plt.close()


def _distribution_rows_from_trace(trace_rows: list[dict], arm_names: list[str]) -> list[dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    grouped["all"] = list(trace_rows)
    for row in trace_rows:
        grouped[str(row["problem"]).lower()].append(row)

    rows: list[dict] = []
    for scope, rows_in_scope in grouped.items():
        counter = Counter(int(row["selected_arm"]) for row in rows_in_scope)
        total = len(rows_in_scope)
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


def _plot_distribution(distribution_rows: list[dict], arm_names: list[str], output_path: Path, title_prefix: str) -> None:
    scopes = ["all", "tsp", "cvrp"]
    fig, axes = plt.subplots(1, 3, figsize=(18, 4.8), sharey=True)

    for ax, scope in zip(axes, scopes):
        rows = [row for row in distribution_rows if row["scope"] == scope]
        ratios = [next((row["ratio"] for row in rows if row["arm_id"] == arm_id), 0.0) for arm_id in range(len(arm_names))]
        labels = [f"{idx}:{name}" for idx, name in enumerate(arm_names)]
        bars = ax.bar(range(len(arm_names)), ratios, color="tab:blue", alpha=0.85)
        ax.set_xticks(range(len(arm_names)))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax.set_ylim(0.0, max(0.05, max(ratios) * 1.2 if ratios else 0.05))
        ax.set_title(f"{title_prefix} - {scope.upper()}")
        ax.set_ylabel("Selection Ratio")
        ax.grid(axis="y", alpha=0.25)
        for bar, ratio in zip(bars, ratios):
            if ratio <= 0:
                continue
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height(),
                f"{100.0 * ratio:.1f}%",
                ha="center",
                va="bottom",
                fontsize=7,
                rotation=90,
            )

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=180)
    plt.close()


def _evaluate_selector(selector, samples, indices: list[int], arm_names: list[str], stage: str) -> tuple[dict, list[dict]]:
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
        progress_log_every=500,
    )
    return metrics, trace_rows


def _single_method_rows_for_problem(samples, val_indices: list[int], arm_names: list[str], selector_metrics: dict, problem: str) -> list[dict]:
    problem = problem.lower()
    scoped_indices = [idx for idx in val_indices if samples[idx].problem == problem]
    feasible_arm_ids = sorted({int(arm_id) for idx in scoped_indices for arm_id in np.flatnonzero(samples[idx].feasible_mask > 0)})

    selector_problem_metrics = selector_metrics["by_problem"][problem]
    selector_mean_cost = float(selector_problem_metrics["mean_cost"])
    selector_top1 = float(selector_problem_metrics["top1_accuracy"])

    rows: list[dict] = []
    for arm_id in feasible_arm_ids:
        selector = FixedArmSelector(arm=arm_id)
        metrics = ro.evaluate_selector(selector, samples, scoped_indices, decision_mode="greedy")
        mean_cost = float(metrics["overall"]["mean_cost"])
        top1 = float(metrics["overall"]["top1_accuracy"])
        rows.append(
            {
                "problem": problem,
                "arm_id": int(arm_id),
                "arm_name": arm_names[arm_id],
                "mean_cost": mean_cost,
                "top1_accuracy": top1,
                "selector_mean_cost": selector_mean_cost,
                "selector_top1_accuracy": selector_top1,
                "selector_beats_cost": bool(selector_mean_cost < mean_cost),
                "selector_beats_top1": bool(selector_top1 > top1),
            }
        )
    rows.sort(key=lambda item: item["mean_cost"])
    return rows


def _periodic_best_rows(periodic_rows: list[dict]) -> list[dict]:
    rows: list[dict] = []
    for row in periodic_rows:
        step = int(row["global_step"])
        for mode in ["val_greedy", "val_ucb"]:
            metrics = row["metrics"][mode]["overall"]
            rows.append(
                {
                    "mode": mode,
                    "global_step": step,
                    "top1_accuracy": float(metrics["top1_accuracy"]),
                    "mean_cost": float(metrics["mean_cost"]),
                    "mean_reward": float(metrics["mean_reward"]),
                    "mean_regret": float(metrics["mean_regret"]),
                }
            )
    return rows


def _best_row(rows: list[dict], key: str, mode: str) -> dict | None:
    if not rows:
        return None
    if mode == "max":
        return max(rows, key=lambda item: float(item[key]))
    return min(rows, key=lambda item: float(item[key]))


def _compare_top1_rows(selector_metrics: dict, sb_global_metrics: dict, sb_problem_metrics: dict, oracle_metrics: dict) -> list[dict]:
    return [
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


def _summary_markdown(
    run_dir: Path,
    checkpoint_step: int,
    selector_metrics: dict,
    best_top1_row: dict | None,
    best_cost_row: dict | None,
    top1_compare_rows: list[dict],
    tsp_rows: list[dict],
    cvrp_rows: list[dict],
) -> str:
    def fmt(x: float) -> str:
        return f"{float(x):.6f}"

    selector_top1 = float(selector_metrics["overall"]["top1_accuracy"])
    selector_mean_cost = float(selector_metrics["overall"]["mean_cost"])

    lines = []
    lines.append("# 当前部分结果汇总")
    lines.append("")
    lines.append(f"- run目录: `{run_dir}`")
    lines.append(f"- 当前用于分析的checkpoint: `best.pt` (step={checkpoint_step})")
    lines.append(f"- 当前best checkpoint在val集上的top1 accuracy: `{fmt(selector_top1)}`")
    lines.append(f"- 当前best checkpoint在val集上的mean cost: `{fmt(selector_mean_cost)}`")
    lines.append("")
    if best_top1_row is not None:
        lines.append("## periodic val中最好的top1")
        lines.append("")
        lines.append(
            f"- mode=`{best_top1_row['mode']}` | step=`{best_top1_row['global_step']}` | "
            f"top1=`{fmt(best_top1_row['top1_accuracy'])}` | mean_cost=`{fmt(best_top1_row['mean_cost'])}`"
        )
        lines.append("")
    if best_cost_row is not None:
        lines.append("## periodic val中最好的mean cost")
        lines.append("")
        lines.append(
            f"- mode=`{best_cost_row['mode']}` | step=`{best_cost_row['global_step']}` | "
            f"mean_cost=`{fmt(best_cost_row['mean_cost'])}` | top1=`{fmt(best_cost_row['top1_accuracy'])}`"
        )
        lines.append("")

    lines.append("## top1 / mean cost 对比")
    lines.append("")
    for row in top1_compare_rows:
        lines.append(
            f"- {row['name']}: top1=`{fmt(row['top1_accuracy'])}` | mean_cost=`{fmt(row['mean_cost'])}`"
        )
    lines.append("")

    def add_problem_summary(problem_name: str, rows: list[dict]) -> None:
        selector_mean_cost = rows[0]["selector_mean_cost"] if rows else float("nan")
        selector_top1 = rows[0]["selector_top1_accuracy"] if rows else float("nan")
        beats_cost = [row["arm_name"] for row in rows if row["selector_beats_cost"]]
        not_beats_cost = [row["arm_name"] for row in rows if not row["selector_beats_cost"]]
        beats_top1 = [row["arm_name"] for row in rows if row["selector_beats_top1"]]
        not_beats_top1 = [row["arm_name"] for row in rows if not row["selector_beats_top1"]]
        lines.append(f"## {problem_name.upper()} 单方法对比")
        lines.append("")
        lines.append(f"- selector mean_cost: `{fmt(selector_mean_cost)}`")
        lines.append(f"- selector top1_accuracy: `{fmt(selector_top1)}`")
        lines.append(f"- 在 mean_cost 上超过的方法: `{', '.join(beats_cost) if beats_cost else '无'}`")
        lines.append(f"- 在 mean_cost 上未超过的方法: `{', '.join(not_beats_cost) if not_beats_cost else '无'}`")
        lines.append(f"- 在 top1_accuracy 上超过的方法: `{', '.join(beats_top1) if beats_top1 else '无'}`")
        lines.append(f"- 在 top1_accuracy 上未超过的方法: `{', '.join(not_beats_top1) if not_beats_top1 else '无'}`")
        lines.append("")

    add_problem_summary("tsp", tsp_rows)
    add_problem_summary("cvrp", cvrp_rows)
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir).resolve()
    analysis_dir = run_dir / args.analysis_dir_name
    analysis_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = _read_checkpoint(run_dir, args.checkpoint)
    checkpoint_step = int(checkpoint.get("global_step", 0))
    restored_args = _build_args_from_checkpoint(checkpoint, device=args.device)
    arm_names, problem_to_methods = get_arm_space_config(restored_args.data_source)

    print(f"[analysis] 开始构建数据集: data_source={restored_args.data_source}")
    dataset = build_joint_dataset(
        results_dir=Path(restored_args.results_dir),
        tsp_dataset_path=Path(restored_args.tsp_dataset_path) if restored_args.tsp_dataset_path else None,
        cvrp_dataset_path=Path(restored_args.cvrp_dataset_path) if restored_args.cvrp_dataset_path else None,
        data_source=restored_args.data_source,
        nss_dataset_root=Path(restored_args.nss_dataset_root) if restored_args.nss_dataset_root else None,
        arm_names=arm_names,
        problem_to_methods=problem_to_methods,
        reward_mode=restored_args.reward_mode,
        max_samples_per_problem=int(restored_args.max_samples_per_problem),
        seed=int(restored_args.seed),
        show_progress=False,
    )
    train_idx, val_idx, test_idx = _resolve_split_indices(dataset, restored_args)
    print(f"[analysis] 数据集完成: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")

    print(f"[analysis] 开始恢复 selector: method={restored_args.method}")
    selector = ro.build_selector(restored_args, n_arms=len(dataset.arm_names), arm_names=dataset.arm_names)
    ro._restore_selector_from_checkpoint(run_dir / "checkpoints" / args.checkpoint, selector)
    print(f"[analysis] selector 恢复完成: checkpoint step={checkpoint_step}")

    print("[analysis] 开始在 val 集上重放 best checkpoint")
    selector_val_metrics, selector_val_trace = _evaluate_selector(
        selector,
        dataset.samples,
        val_idx,
        dataset.arm_names,
        stage="val_greedy_best_checkpoint",
    )
    _write_json(analysis_dir / "selector_val_greedy_best_checkpoint.metrics.json", selector_val_metrics)
    with (analysis_dir / "selector_val_greedy_best_checkpoint.trace.jsonl").open("w", encoding="utf-8") as f:
        for row in selector_val_trace:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print("[analysis] 计算 baseline 对比")
    single_best_global = SingleBestGlobalSelector.fit(dataset.samples, train_idx, dataset.arm_names)
    single_best_per_problem = SingleBestPerProblemSelector.fit(dataset.samples, train_idx, dataset.arm_names)
    oracle = OracleSelector()
    sb_global_metrics = ro.evaluate_selector(single_best_global, dataset.samples, val_idx, decision_mode="greedy")
    sb_problem_metrics = ro.evaluate_selector(single_best_per_problem, dataset.samples, val_idx, decision_mode="greedy")
    oracle_metrics = ro.evaluate_selector(oracle, dataset.samples, val_idx, decision_mode="greedy")

    top1_compare_rows = _compare_top1_rows(
        selector_metrics=selector_val_metrics,
        sb_global_metrics=sb_global_metrics,
        sb_problem_metrics=sb_problem_metrics,
        oracle_metrics=oracle_metrics,
    )
    _write_csv(analysis_dir / "top1_baseline_compare.csv", top1_compare_rows)

    print("[analysis] 统计单方法对比")
    tsp_compare_rows = _single_method_rows_for_problem(
        dataset.samples,
        val_idx,
        dataset.arm_names,
        selector_val_metrics,
        problem="tsp",
    )
    cvrp_compare_rows = _single_method_rows_for_problem(
        dataset.samples,
        val_idx,
        dataset.arm_names,
        selector_val_metrics,
        problem="cvrp",
    )
    _write_csv(analysis_dir / "tsp_single_method_compare.csv", tsp_compare_rows)
    _write_csv(analysis_dir / "cvrp_single_method_compare.csv", cvrp_compare_rows)

    print("[analysis] 解析 periodic val 与 loss")
    periodic_rows = _load_jsonl(run_dir / "periodic_val.jsonl")
    periodic_metric_rows = _periodic_best_rows(periodic_rows)
    _write_csv(analysis_dir / "periodic_val_metrics.csv", periodic_metric_rows)
    best_top1_row = _best_row(periodic_metric_rows, key="top1_accuracy", mode="max")
    best_cost_row = _best_row(periodic_metric_rows, key="mean_cost", mode="min")

    loss_rows = _parse_loss_rows(run_dir / "neural_linucb.log")
    _write_csv(analysis_dir / "loss_curve.csv", loss_rows)
    _plot_loss(loss_rows, analysis_dir / "loss_avg_curve.png", best_step=checkpoint_step)

    print("[analysis] 统计 train / val 臂分布")
    train_trace_rows = _load_jsonl(run_dir / "traces" / "selector_train_ucb.jsonl")
    # 用前 checkpoint_step 条训练记录，对齐 best checkpoint。
    train_trace_rows = train_trace_rows[:checkpoint_step]
    train_distribution_rows = _distribution_rows_from_trace(train_trace_rows, dataset.arm_names)
    val_distribution_rows = _distribution_rows_from_trace(selector_val_trace, dataset.arm_names)
    _write_csv(analysis_dir / "train_arm_distribution_at_best_step.csv", train_distribution_rows)
    _write_csv(analysis_dir / "val_arm_distribution_best_checkpoint.csv", val_distribution_rows)
    _plot_distribution(
        train_distribution_rows,
        dataset.arm_names,
        analysis_dir / "train_arm_distribution_at_best_step.png",
        title_prefix=f"Train Distribution @ step {checkpoint_step}",
    )
    _plot_distribution(
        val_distribution_rows,
        dataset.arm_names,
        analysis_dir / "val_arm_distribution_best_checkpoint.png",
        title_prefix="Val Distribution @ best checkpoint",
    )

    summary_md = _summary_markdown(
        run_dir=run_dir,
        checkpoint_step=checkpoint_step,
        selector_metrics=selector_val_metrics,
        best_top1_row=best_top1_row,
        best_cost_row=best_cost_row,
        top1_compare_rows=top1_compare_rows,
        tsp_rows=tsp_compare_rows,
        cvrp_rows=cvrp_compare_rows,
    )
    (analysis_dir / "summary.md").write_text(summary_md, encoding="utf-8")

    summary_json = {
        "run_dir": str(run_dir),
        "checkpoint_step": checkpoint_step,
        "selector_val_metrics": selector_val_metrics,
        "best_periodic_top1": best_top1_row,
        "best_periodic_mean_cost": best_cost_row,
        "top1_compare_rows": top1_compare_rows,
        "tsp_single_method_compare": tsp_compare_rows,
        "cvrp_single_method_compare": cvrp_compare_rows,
    }
    _write_json(analysis_dir / "summary.json", summary_json)
    print(f"[analysis] 完成，结果已写到: {analysis_dir}")


if __name__ == "__main__":
    main()
