from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parent
TRAIN_LOGS_DIR = ROOT / "train_logs"
OUTPUT_DIR = TRAIN_LOGS_DIR / "observation_metrics"
TARGET_SPLITS = ("val", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reorganize prior experiment artifacts around the observation metrics spec."
    )
    parser.add_argument(
        "--train-logs-dir",
        type=Path,
        default=TRAIN_LOGS_DIR,
        help="Directory containing experiment run folders.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Directory where reorganized observation metrics will be written.",
    )
    parser.add_argument(
        "--runs",
        nargs="*",
        default=None,
        help="Optional run directory names to include. Defaults to all retained full runs.",
    )
    return parser.parse_args()


def discover_runs(train_logs_dir: Path, requested_runs: list[str] | None) -> list[Path]:
    if requested_runs:
        run_dirs = [train_logs_dir / run_name for run_name in requested_runs]
    else:
        run_dirs = sorted(
            path
            for path in train_logs_dir.iterdir()
            if path.is_dir()
            and path.name.startswith("round")
            and (path / "analysis_val").exists()
            and (path / "analysis_test").exists()
        )

    missing = [str(path) for path in run_dirs if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Requested run directories do not exist: {missing}")
    return run_dirs


def ensure_clean_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def safe_filename(name: str) -> str:
    return name.replace("/", "_").replace(" ", "_")


def load_run_name(run_dir: Path) -> str:
    config_path = run_dir / "config.json"
    if not config_path.exists():
        return run_dir.name
    with config_path.open("r", encoding="utf-8") as f:
        config = json.load(f)
    return config.get("run_name") or run_dir.name


def _problem_grid_shape(num_problems: int) -> tuple[int, int]:
    cols = 3
    rows = (num_problems + cols - 1) // cols
    return rows, cols


def _annotate_barh(ax, bars, values, fmt: str, min_padding_ratio: float = 0.18) -> None:
    max_value = max(values) if values else 0.0
    right_limit = max_value * (1.0 + min_padding_ratio)
    if right_limit <= 0:
        right_limit = 1.0
    ax.set_xlim(0, right_limit)
    labels = [format(value, fmt) for value in values]
    ax.bar_label(bars, labels=labels, padding=3, fontsize=7)


def plot_metric_grid(
    compare_map: dict[str, pd.DataFrame],
    problems: list[str],
    metric_col: str,
    selector_col: str,
    out_path: Path,
    split: str,
    ylabel: str,
    higher_is_better: bool,
) -> None:
    rows, cols = _problem_grid_shape(len(problems))
    fig, axes = plt.subplots(rows, cols, figsize=(18, rows * 3.8))
    axes = axes.flatten()

    for ax, problem in zip(axes, problems):
        compare_df = compare_map[problem]
        method_df = compare_df[["arm_name", metric_col]].copy()
        method_df = method_df.sort_values(metric_col, ascending=not higher_is_better)
        selector_value = float(compare_df[selector_col].iloc[0])
        plot_df = pd.concat(
            [
                method_df,
                pd.DataFrame([{"arm_name": "SELECTOR", metric_col: selector_value}]),
            ],
            ignore_index=True,
        )
        colors = ["#4C78A8" if name == "SELECTOR" else "#BDBDBD" for name in plot_df["arm_name"]]
        values = plot_df[metric_col].tolist()
        bars = ax.barh(plot_df["arm_name"], values, color=colors)
        ax.set_title(problem.upper(), fontsize=10)
        ax.invert_yaxis()
        ax.grid(axis="x", linestyle="--", alpha=0.25)
        ax.tick_params(axis="y", labelsize=8)
        ax.tick_params(axis="x", labelsize=8)
        ax.set_xlabel(ylabel, fontsize=8)
        _annotate_barh(ax, bars, values, fmt=".3f", min_padding_ratio=0.22)

    for ax in axes[len(problems) :]:
        ax.axis("off")

    fig.suptitle(f"{split.upper()} {metric_col} vs Single Methods", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_arm_distribution_grid(
    arm_df: pd.DataFrame,
    problems: list[str],
    out_path: Path,
    split: str,
) -> None:
    rows, cols = _problem_grid_shape(len(problems))
    fig, axes = plt.subplots(rows, cols, figsize=(18, rows * 3.8))
    axes = axes.flatten()

    for ax, problem in zip(axes, problems):
        problem_df = arm_df[arm_df["problem"] == problem].copy().sort_values("count", ascending=False)
        colors = ["#4C78A8"] + ["#BDBDBD"] * max(len(problem_df) - 1, 0)
        values = problem_df["count"].tolist()
        bars = ax.barh(problem_df["arm_name"], values, color=colors[: len(problem_df)])
        ax.set_title(problem.upper(), fontsize=10)
        ax.invert_yaxis()
        ax.grid(axis="x", linestyle="--", alpha=0.25)
        ax.tick_params(axis="y", labelsize=8)
        ax.tick_params(axis="x", labelsize=8)
        ax.set_xlabel("Selection Count", fontsize=8)
        _annotate_barh(ax, bars, values, fmt=".0f", min_padding_ratio=0.15)

    for ax in axes[len(problems) :]:
        ax.axis("off")

    fig.suptitle(f"{split.upper()} Selector Arm Distribution", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_loss_curve(log_df: pd.DataFrame, out_path: Path, run_label: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.plot(log_df["epoch"], log_df["train_loss"], marker="o", label="train_loss", color="#4C78A8")
    ax.plot(log_df["epoch"], log_df["val_loss"], marker="o", label="val_loss", color="#F58518")
    ax.set_title(f"{run_label} Loss Curve")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(alpha=0.3, linestyle="--")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def summarize_problem_metrics(
    summary_df: pd.DataFrame,
    compare_dir: Path,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for _, row in summary_df.sort_values("problem").iterrows():
        problem = row["problem"]
        compare_path = compare_dir / f"{problem}_single_method_compare.csv"
        compare_df = pd.read_csv(compare_path)

        beats_top1 = compare_df.loc[compare_df["selector_beats_top1"], "arm_name"].tolist()
        not_beats_top1 = compare_df.loc[~compare_df["selector_beats_top1"], "arm_name"].tolist()
        beats_cost = compare_df.loc[compare_df["selector_beats_cost"], "arm_name"].tolist()
        not_beats_cost = compare_df.loc[~compare_df["selector_beats_cost"], "arm_name"].tolist()

        rows.append(
            {
                "problem": problem,
                "count": int(row["count"]),
                "selector_top1_accuracy": float(row["selector_top1_accuracy"]),
                "best_top1_accuracy": float(row["best_single_top1_accuracy"]),
                "selector_mean_cost": float(row["selector_mean_cost"]),
                "best_mean_cost": float(row["best_single_mean_cost"]),
                "best_single_method_by_cost": row["best_single_method_by_cost"],
                "selector_beats_best_single_top1": bool(row["selector_beats_best_single_top1"]),
                "selector_beats_best_single_cost": bool(row["selector_beats_best_single_cost"]),
                "selector_top1_gain_vs_best_single": float(row["selector_top1_gain_vs_best_single"]),
                "selector_gap_vs_best_single": float(row["selector_gap_vs_best_single"]),
                "oracle_mean_cost": float(row["oracle_mean_cost"]),
                "selector_oracle_ratio": float(row["selector_oracle_ratio"]),
                "num_single_methods": int(len(compare_df)),
                "selector_beats_top1_count": int(len(beats_top1)),
                "selector_not_beats_top1_count": int(len(not_beats_top1)),
                "selector_beats_mean_cost_count": int(len(beats_cost)),
                "selector_not_beats_mean_cost_count": int(len(not_beats_cost)),
                "selector_beats_top1_methods": "; ".join(beats_top1),
                "selector_not_beats_top1_methods": "; ".join(not_beats_top1),
                "selector_beats_mean_cost_methods": "; ".join(beats_cost),
                "selector_not_beats_mean_cost_methods": "; ".join(not_beats_cost),
            }
        )

    return pd.DataFrame(rows)


def write_markdown_summary(df: pd.DataFrame, out_path: Path, title: str) -> None:
    selected_cols = [
        "problem",
        "selector_top1_accuracy",
        "best_top1_accuracy",
        "selector_mean_cost",
        "best_mean_cost",
        "selector_beats_top1_count",
        "selector_not_beats_top1_count",
        "selector_beats_mean_cost_count",
        "selector_not_beats_mean_cost_count",
    ]

    lines = [f"# {title}", "", df[selected_cols].to_markdown(index=False), ""]
    out_path.write_text("\n".join(lines), encoding="utf-8")


def write_output_readme(out_dir: Path, runs: list[dict[str, str]]) -> None:
    lines = [
        "# Observation Metrics",
        "",
        "This directory reorganizes prior experiments around the metric checklist in `观测指标/观测指标.md`.",
        "",
        "For each retained full run:",
        "- `loss_curve.png`: train/val loss over epochs.",
        "- `val_problem_metrics.csv` and `test_problem_metrics.csv`: per-problem metric tables.",
        "- `val_problem_metrics.md` and `test_problem_metrics.md`: readable markdown summaries.",
        "- `val_top1_vs_single_methods.png` and `test_top1_vs_single_methods.png`: one consolidated figure per split, containing all problems.",
        "- `val_mean_cost_vs_single_methods.png` and `test_mean_cost_vs_single_methods.png`: one consolidated figure per split, containing all problems.",
        "- `val_arm_distribution.png` and `test_arm_distribution.png`: one consolidated figure per split, containing all problems.",
        "",
        "Cross-run comparison tables:",
        "- `comparison/val_all_runs_problem_metrics.csv`",
        "- `comparison/test_all_runs_problem_metrics.csv`",
        "",
        "Runs included:",
    ]
    for run in runs:
        lines.append(f"- `{run['run_name']}`")
    lines.append("")
    out_dir.joinpath("README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    train_logs_dir = args.train_logs_dir.resolve()
    output_dir = args.output_dir.resolve()
    run_dirs = discover_runs(train_logs_dir, args.runs)

    ensure_clean_dir(output_dir)
    comparison_dir = output_dir / "comparison"
    comparison_dir.mkdir(parents=True, exist_ok=True)

    combined_split_rows: dict[str, list[pd.DataFrame]] = {split: [] for split in TARGET_SPLITS}
    run_index: list[dict[str, str]] = []

    for run_dir in run_dirs:
        run_name = load_run_name(run_dir)
        run_index.append({"run_name": run_name})

        run_out_dir = output_dir / run_dir.name
        run_out_dir.mkdir(parents=True, exist_ok=True)

        log_df = pd.read_csv(run_dir / "log.csv")
        plot_loss_curve(log_df, run_out_dir / "loss_curve.png", run_name)

        for split in TARGET_SPLITS:
            analysis_dir = run_dir / f"analysis_{split}"
            summary_df = pd.read_csv(analysis_dir / "per_problem_summary.csv")
            arm_df = pd.read_csv(analysis_dir / "arm_distribution.csv")
            problems = sorted(summary_df["problem"].tolist())
            compare_map = {
                problem: pd.read_csv(analysis_dir / f"{problem}_single_method_compare.csv") for problem in problems
            }

            plot_metric_grid(
                compare_map=compare_map,
                problems=problems,
                metric_col="top1_accuracy",
                selector_col="selector_top1_accuracy",
                out_path=run_out_dir / f"{split}_top1_vs_single_methods.png",
                split=split,
                ylabel="Top1 Accuracy",
                higher_is_better=True,
            )
            plot_metric_grid(
                compare_map=compare_map,
                problems=problems,
                metric_col="mean_cost",
                selector_col="selector_mean_cost",
                out_path=run_out_dir / f"{split}_mean_cost_vs_single_methods.png",
                split=split,
                ylabel="Mean Cost",
                higher_is_better=False,
            )
            plot_arm_distribution_grid(
                arm_df=arm_df,
                problems=problems,
                out_path=run_out_dir / f"{split}_arm_distribution.png",
                split=split,
            )

            metrics_df = summarize_problem_metrics(
                summary_df=summary_df,
                compare_dir=analysis_dir,
            )
            metrics_df.insert(0, "run_name", run_name)

            metrics_csv_path = run_out_dir / f"{split}_problem_metrics.csv"
            metrics_md_path = run_out_dir / f"{split}_problem_metrics.md"
            metrics_df.to_csv(metrics_csv_path, index=False)
            write_markdown_summary(
                metrics_df,
                metrics_md_path,
                title=f"{run_name} {split.upper()} Observation Metrics",
            )
            combined_split_rows[split].append(metrics_df)

        with (run_out_dir / "run_info.json").open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "run_name": run_name,
                    "source_dir": str(run_dir),
                },
                f,
                indent=2,
            )

    for split, frames in combined_split_rows.items():
        if not frames:
            continue
        combined_df = pd.concat(frames, ignore_index=True)
        combined_df.to_csv(comparison_dir / f"{split}_all_runs_problem_metrics.csv", index=False)

    write_output_readme(output_dir, run_index)


if __name__ == "__main__":
    main()
