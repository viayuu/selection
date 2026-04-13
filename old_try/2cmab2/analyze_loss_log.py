from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    """
    计算简单滑动平均。

    这里不用 pandas，避免额外依赖。
    """
    values = np.asarray(values, dtype=np.float64)
    out = np.empty_like(values)
    csum = np.cumsum(np.insert(values, 0, 0.0))
    for idx in range(len(values)):
        start = max(0, idx + 1 - window)
        count = idx + 1 - start
        out[idx] = (csum[idx + 1] - csum[start]) / count
    return out


def _extract_fields(pattern: re.Pattern[str], text: str) -> dict[str, float | int] | None:
    """把一行日志按 regex 提取成字段字典。"""
    match = pattern.search(text)
    if match is None:
        return None
    row: dict[str, float | int] = {}
    for key, value in match.groupdict().items():
        if value is None:
            continue
        if re.fullmatch(r"[0-9]+", value):
            row[key] = int(value)
        else:
            row[key] = float(value)
    return row


def parse_loss_log(log_path: Path) -> tuple[str, list[dict[str, float | int]]]:
    """
    解析 selector loss 日志。

    当前支持两类：
    1. neural_ucb_diag 的 `train reward_net`
    2. neural_linucb 的 `train representation`
    """
    diag_pattern = re.compile(
        r"t=(?P<t>\d+) \| train reward_net \| steps=(?P<opt_steps>\d+) "
        r"\| avg_loss=(?P<avg_loss>[0-9.]+) \| min_loss=(?P<min_loss>[0-9.]+) "
        r"\| max_loss=(?P<max_loss>[0-9.]+) \| last_loss=(?P<last_loss>[0-9.]+)"
    )
    lin_pattern = re.compile(
        r"t=(?P<t>\d+) \| train representation \| epochs=(?P<num_epochs>\d+) "
        r"\| opt_steps=(?P<opt_steps>\d+) \| avg_loss=(?P<avg_loss>[0-9.]+) "
        r"\| min_loss=(?P<min_loss>[0-9.]+) \| max_loss=(?P<max_loss>[0-9.]+) "
        r"\| last_loss=(?P<last_loss>[0-9.]+)"
    )

    rows: list[dict[str, float | int]] = []
    method = "unknown"
    with log_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            row = _extract_fields(diag_pattern, line)
            if row is not None:
                method = "neural_ucb_diag"
                rows.append(row)
                continue
            row = _extract_fields(lin_pattern, line)
            if row is not None:
                method = "neural_linucb"
                rows.append(row)
    if not rows:
        raise ValueError(f"在日志中没有找到 loss 记录: {log_path}")
    return method, rows


def save_loss_csv(rows: list[dict[str, float | int]], out_path: Path) -> None:
    """把解析出的 loss 统计写成 csv。"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_loss_summary(method: str, rows: list[dict[str, float | int]], out_path: Path) -> None:
    """写一个简短摘要，方便快速查看趋势。"""
    t = np.array([int(row["t"]) for row in rows], dtype=np.int32)
    avg_loss = np.array([float(row["avg_loss"]) for row in rows], dtype=np.float64)
    min_loss = np.array([float(row["min_loss"]) for row in rows], dtype=np.float64)
    max_loss = np.array([float(row["max_loss"]) for row in rows], dtype=np.float64)
    last_loss = np.array([float(row["last_loss"]) for row in rows], dtype=np.float64)

    summary = {
        "method": method,
        "count": int(len(rows)),
        "first_t": int(t[0]),
        "last_t": int(t[-1]),
        "first_avg_loss": float(avg_loss[0]),
        "last_avg_loss": float(avg_loss[-1]),
        "avg_of_avg_loss": float(np.mean(avg_loss)),
        "avg_of_last_loss": float(np.mean(last_loss)),
        "global_min_loss": float(np.min(min_loss)),
        "global_max_loss": float(np.max(max_loss)),
        "p95_avg_loss": float(np.percentile(avg_loss, 95)),
        "p99_avg_loss": float(np.percentile(avg_loss, 99)),
        "p95_max_loss": float(np.percentile(max_loss, 95)),
        "p99_max_loss": float(np.percentile(max_loss, 99)),
    }
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


def plot_loss_readable(method: str, rows: list[dict[str, float | int]], out_dir: Path, window: int) -> None:
    """
    生成更适合阅读趋势的图。

    设计思路：
    1. 原始曲线保留，但淡化透明度
    2. 叠加 rolling mean 主趋势线
    3. 使用 p95/p99 截尾视角，避免极端尖峰把主体压扁
    4. 再补一张 log-scale 图，兼顾整体范围
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    t = np.array([int(row["t"]) for row in rows], dtype=np.int32)
    avg_loss = np.array([float(row["avg_loss"]) for row in rows], dtype=np.float64)
    min_loss = np.array([float(row["min_loss"]) for row in rows], dtype=np.float64)
    max_loss = np.array([float(row["max_loss"]) for row in rows], dtype=np.float64)
    last_loss = np.array([float(row["last_loss"]) for row in rows], dtype=np.float64)

    avg_roll = _rolling_mean(avg_loss, window)
    min_roll = _rolling_mean(min_loss, window)
    max_roll = _rolling_mean(max_loss, window)
    last_roll = _rolling_mean(last_loss, window)

    p95_avg = float(np.percentile(avg_loss, 95))
    p99_avg = float(np.percentile(avg_loss, 99))
    p95_max = float(np.percentile(max_loss, 95))
    p99_max = float(np.percentile(max_loss, 99))
    p95_last = float(np.percentile(last_loss, 95))
    p99_last = float(np.percentile(last_loss, 99))

    plt.style.use("seaborn-v0_8-whitegrid")

    # 图 1：面向“趋势是否在变好”的主图。
    fig, axes = plt.subplots(2, 2, figsize=(16, 10), dpi=160)

    ax = axes[0, 0]
    ax.plot(t, avg_loss, color="tab:blue", alpha=0.18, linewidth=0.8, label="avg_loss raw")
    ax.plot(t, avg_roll, color="tab:blue", linewidth=2.2, label=f"avg_loss rolling({window})")
    ax.plot(t, last_loss, color="tab:orange", alpha=0.12, linewidth=0.7, label="last_loss raw")
    ax.plot(t, last_roll, color="tab:orange", linewidth=1.8, label=f"last_loss rolling({window})")
    ax.set_title("Avg / Last Loss Trend")
    ax.set_xlabel("bandit step t")
    ax.set_ylabel("loss")
    ax.legend()

    ax = axes[0, 1]
    ax.plot(t, avg_loss, color="tab:blue", alpha=0.18, linewidth=0.8, label="avg_loss raw")
    ax.plot(t, avg_roll, color="tab:blue", linewidth=2.2, label=f"avg_loss rolling({window})")
    ax.plot(t, last_loss, color="tab:orange", alpha=0.12, linewidth=0.7, label="last_loss raw")
    ax.plot(t, last_roll, color="tab:orange", linewidth=1.8, label=f"last_loss rolling({window})")
    ax.set_ylim(0.0, max(p99_avg, p99_last) * 1.1)
    ax.set_title("Avg / Last Loss Trend (y clipped to p99)")
    ax.set_xlabel("bandit step t")
    ax.set_ylabel("loss")
    ax.legend()

    ax = axes[1, 0]
    ax.plot(t, max_loss, color="tab:red", alpha=0.16, linewidth=0.8, label="max_loss raw")
    ax.plot(t, max_roll, color="tab:red", linewidth=2.0, label=f"max_loss rolling({window})")
    ax.plot(t, min_loss, color="tab:green", alpha=0.18, linewidth=0.8, label="min_loss raw")
    ax.plot(t, min_roll, color="tab:green", linewidth=1.8, label=f"min_loss rolling({window})")
    ax.set_ylim(0.0, max(p99_max, np.percentile(min_loss, 99)) * 1.1)
    ax.set_title("Min / Max Loss Trend (y clipped to p99)")
    ax.set_xlabel("bandit step t")
    ax.set_ylabel("loss")
    ax.legend()

    ax = axes[1, 1]
    ax.plot(t, avg_roll, color="tab:blue", linewidth=2.2, label="avg_loss rolling")
    ax.plot(t, last_roll, color="tab:orange", linewidth=1.8, label="last_loss rolling")
    ax.plot(t, max_roll, color="tab:red", linewidth=1.8, label="max_loss rolling")
    ax.plot(t, min_roll, color="tab:green", linewidth=1.5, label="min_loss rolling")
    ax.set_title("Rolling Trends Only")
    ax.set_xlabel("bandit step t")
    ax.set_ylabel("loss")
    ax.legend()

    fig.suptitle(f"{method} readable loss trends", fontsize=14)
    fig.tight_layout(rect=[0, 0.02, 1, 0.98])
    fig.savefig(out_dir / "loss_trend_readable.png")
    plt.close(fig)

    # 图 2：log 视角，专门看“整体下降趋势”。
    fig, axes = plt.subplots(2, 1, figsize=(15, 9), dpi=160)

    ax = axes[0]
    ax.plot(t, np.maximum(avg_loss, 1e-8), color="tab:blue", alpha=0.18, linewidth=0.8, label="avg_loss raw")
    ax.plot(t, np.maximum(avg_roll, 1e-8), color="tab:blue", linewidth=2.2, label=f"avg_loss rolling({window})")
    ax.plot(t, np.maximum(last_loss, 1e-8), color="tab:orange", alpha=0.12, linewidth=0.7, label="last_loss raw")
    ax.plot(t, np.maximum(last_roll, 1e-8), color="tab:orange", linewidth=1.8, label=f"last_loss rolling({window})")
    ax.set_yscale("log")
    ax.set_title("Avg / Last Loss on Log Scale")
    ax.set_xlabel("bandit step t")
    ax.set_ylabel("loss (log)")
    ax.legend()

    ax = axes[1]
    ax.plot(t, np.maximum(max_loss, 1e-8), color="tab:red", alpha=0.16, linewidth=0.8, label="max_loss raw")
    ax.plot(t, np.maximum(max_roll, 1e-8), color="tab:red", linewidth=2.0, label=f"max_loss rolling({window})")
    ax.plot(t, np.maximum(min_loss, 1e-8), color="tab:green", alpha=0.18, linewidth=0.8, label="min_loss raw")
    ax.plot(t, np.maximum(min_roll, 1e-8), color="tab:green", linewidth=1.8, label=f"min_loss rolling({window})")
    ax.set_yscale("log")
    ax.set_title("Min / Max Loss on Log Scale")
    ax.set_xlabel("bandit step t")
    ax.set_ylabel("loss (log)")
    ax.legend()

    fig.tight_layout()
    fig.savefig(out_dir / "loss_trend_logscale.png")
    plt.close(fig)

    # 图 3：按 chunk 聚合，进一步消掉高频抖动。
    chunk = max(20, window)
    chunk_rows = []
    for start in range(0, len(rows), chunk):
        end = min(len(rows), start + chunk)
        part_avg = avg_loss[start:end]
        part_last = last_loss[start:end]
        part_max = max_loss[start:end]
        part_min = min_loss[start:end]
        chunk_rows.append(
            {
                "chunk_id": start // chunk,
                "start_t": int(t[start]),
                "end_t": int(t[end - 1]),
                "mean_avg_loss": float(np.mean(part_avg)),
                "mean_last_loss": float(np.mean(part_last)),
                "mean_max_loss": float(np.mean(part_max)),
                "mean_min_loss": float(np.mean(part_min)),
            }
        )

    with (out_dir / "loss_chunk_summary.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(chunk_rows[0].keys()))
        writer.writeheader()
        writer.writerows(chunk_rows)

    fig, ax = plt.subplots(figsize=(15, 5), dpi=160)
    chunk_x = np.array([row["end_t"] for row in chunk_rows], dtype=np.int32)
    ax.plot(chunk_x, [row["mean_avg_loss"] for row in chunk_rows], marker="o", linewidth=2.0, label="chunk mean avg_loss")
    ax.plot(chunk_x, [row["mean_last_loss"] for row in chunk_rows], marker="o", linewidth=1.7, label="chunk mean last_loss")
    ax.plot(chunk_x, [row["mean_max_loss"] for row in chunk_rows], marker="o", linewidth=1.7, label="chunk mean max_loss")
    ax.set_title(f"Chunk-Averaged Loss Trend (chunk={chunk})")
    ax.set_xlabel("bandit step t")
    ax.set_ylabel("loss")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "loss_chunk_trend.png")
    plt.close(fig)


def detect_default_log(run_dir: Path) -> Path:
    """如果用户只给 run_dir，就自动找日志文件。"""
    candidates = [
        run_dir / "neural_ucb_diag.log",
        run_dir / "neural_linucb.log",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"在 {run_dir} 下未找到 neural_ucb_diag.log 或 neural_linucb.log")


def main() -> None:
    parser = argparse.ArgumentParser(description="把 selector 的 loss 日志解析成更易读的 csv 和图片。")
    parser.add_argument("--run-dir", type=str, required=True, help="输出目录，例如 2cmab2/outputs/2026-..._neural_ucb_diag")
    parser.add_argument("--log-path", type=str, default="", help="可选，直接指定日志路径")
    parser.add_argument("--window", type=int, default=200, help="rolling mean 的窗口大小")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    log_path = Path(args.log_path) if args.log_path else detect_default_log(run_dir)
    method, rows = parse_loss_log(log_path)

    out_dir = run_dir / "analysis_loss"
    save_loss_csv(rows, out_dir / f"{method}_loss_metrics.csv")
    save_loss_summary(method, rows, out_dir / f"{method}_loss_summary.json")
    plot_loss_readable(method, rows, out_dir=out_dir, window=int(args.window))

    print(f"method: {method}")
    print(f"log: {log_path}")
    print(f"out_dir: {out_dir}")


if __name__ == "__main__":
    main()
