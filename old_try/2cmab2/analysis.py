from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

if __package__ in (None, ""):
    import sys

    THIS_DIR = Path(__file__).resolve().parent
    if str(THIS_DIR) not in sys.path:
        sys.path.insert(0, str(THIS_DIR))
    from default_settings import AnalysisDefaults  # type: ignore
else:
    from .default_settings import AnalysisDefaults


@dataclass
class SummaryRun:
    """
    表示一次已经完成的实验运行。

    这里假设每次运行目录里都有一个 `summary.json`，
    而 summary.json 是 `run_offline.py` 写出来的实验摘要。
    """

    run_dir: Path
    summary_path: Path
    method: str
    timestamp: str
    payload: dict


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    """把一组字典行写成 csv。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _fmt(value) -> str:
    """把数字格式化成更适合报告展示的字符串。"""
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def collect_summary_runs(input_root: Path) -> list[SummaryRun]:
    """
    扫描 outputs 目录下的所有 summary.json。

    默认约定：
    2cmab2/outputs/<timestamp>_<method>/summary.json
    """
    runs: list[SummaryRun] = []
    for summary_path in sorted(input_root.glob("*/summary.json")):
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        run_dir = summary_path.parent
        method = payload.get("config", {}).get("method", run_dir.name.split("_")[-1])
        timestamp = run_dir.name[:19]
        runs.append(
            SummaryRun(
                run_dir=run_dir,
                summary_path=summary_path,
                method=method,
                timestamp=timestamp,
                payload=payload,
            )
        )
    return runs


def _flatten_metric_block(run: SummaryRun, source_name: str, block_name: str, block_payload: dict) -> list[dict]:
    """
    把一个指标块展开成表格行。

    一个 block 里通常包含：
    - overall
    - by_problem
    - macro_average

    这里统一摊平成多行 csv，便于后续筛选和排序。
    """
    rows: list[dict] = []

    overall = block_payload.get("overall", {})
    rows.append(
        {
            "timestamp": run.timestamp,
            "method": run.method,
            "source": source_name,
            "split": block_name,
            "scope": "overall",
            "count": overall.get("count", ""),
            "mean_reward": overall.get("mean_reward", ""),
            "top1_accuracy": overall.get("top1_accuracy", ""),
            "mean_regret": overall.get("mean_regret", ""),
            "mean_cost": overall.get("mean_cost", ""),
            "run_dir": str(run.run_dir),
        }
    )

    macro = block_payload.get("macro_average", {})
    if macro:
        rows.append(
            {
                "timestamp": run.timestamp,
                "method": run.method,
                "source": source_name,
                "split": block_name,
                "scope": "macro_average",
                "count": "",
                "mean_reward": macro.get("mean_reward", ""),
                "top1_accuracy": macro.get("top1_accuracy", ""),
                "mean_regret": macro.get("mean_regret", ""),
                "mean_cost": macro.get("mean_cost", ""),
                "run_dir": str(run.run_dir),
            }
        )

    for problem, metrics in block_payload.get("by_problem", {}).items():
        rows.append(
            {
                "timestamp": run.timestamp,
                "method": run.method,
                "source": source_name,
                "split": block_name,
                "scope": problem,
                "count": metrics.get("count", ""),
                "mean_reward": metrics.get("mean_reward", ""),
                "top1_accuracy": metrics.get("top1_accuracy", ""),
                "mean_regret": metrics.get("mean_regret", ""),
                "mean_cost": metrics.get("mean_cost", ""),
                "run_dir": str(run.run_dir),
            }
        )
    return rows


def build_analysis_report(input_root: Path, output_dir: Path) -> None:
    """
    把多个 summary.json 聚合成：
    - run 清单
    - selector 指标表
    - baseline 指标表
    - 简短 markdown 报告
    """
    runs = collect_summary_runs(input_root)
    if not runs:
        raise ValueError(f"在 {input_root} 下没有找到 summary.json")

    manifest_rows = []
    selector_rows: list[dict] = []
    baseline_rows: list[dict] = []

    for run in runs:
        config = run.payload.get("config", {})
        splits = run.payload.get("splits", {})
        # 运行清单：记录这次实验最关键的超参数和目录
        manifest_rows.append(
            {
                "timestamp": run.timestamp,
                "method": run.method,
                "seed": config.get("seed", ""),
                "alpha": config.get("alpha", ""),
                "reward_mode": config.get("reward_mode", ""),
                "epochs": config.get("epochs", ""),
                "train_size": splits.get("train", ""),
                "val_size": splits.get("val", ""),
                "test_size": splits.get("test", ""),
                "run_dir": str(run.run_dir),
            }
        )

        for split_name, block in run.payload.get("selector", {}).items():
            selector_rows.extend(_flatten_metric_block(run, "selector", split_name, block))

        for baseline_name, block in run.payload.get("baselines", {}).items():
            # baseline 默认只有 test_greedy
            baseline_rows.extend(_flatten_metric_block(run, baseline_name, "test_greedy", block))

    _write_csv(
        output_dir / "runs_manifest.csv",
        manifest_rows,
        ["timestamp", "method", "seed", "alpha", "reward_mode", "epochs", "train_size", "val_size", "test_size", "run_dir"],
    )
    _write_csv(
        output_dir / "selector_test_greedy.csv",
        [row for row in selector_rows if row["split"] == "test_greedy"],
        ["timestamp", "method", "source", "split", "scope", "count", "mean_reward", "top1_accuracy", "mean_regret", "mean_cost", "run_dir"],
    )
    _write_csv(
        output_dir / "selector_test_ucb.csv",
        [row for row in selector_rows if row["split"] == "test_ucb"],
        ["timestamp", "method", "source", "split", "scope", "count", "mean_reward", "top1_accuracy", "mean_regret", "mean_cost", "run_dir"],
    )
    _write_csv(
        output_dir / "baseline_test_greedy.csv",
        baseline_rows,
        ["timestamp", "method", "source", "split", "scope", "count", "mean_reward", "top1_accuracy", "mean_regret", "mean_cost", "run_dir"],
    )

    overall_greedy = [row for row in selector_rows if row["split"] == "test_greedy" and row["scope"] == "overall"]
    overall_greedy_sorted = sorted(
        overall_greedy,
        key=lambda row: (-float(row["mean_reward"]), float(row["mean_regret"]), float(row["mean_cost"])),
    )
    _write_csv(
        output_dir / "selector_overall_ranking.csv",
        overall_greedy_sorted,
        ["timestamp", "method", "source", "split", "scope", "count", "mean_reward", "top1_accuracy", "mean_regret", "mean_cost", "run_dir"],
    )

    best_reward = max(overall_greedy, key=lambda row: row["mean_reward"])
    best_regret = min(overall_greedy, key=lambda row: row["mean_regret"])
    baseline_overall = [row for row in baseline_rows if row["scope"] == "overall"]
    baseline_best_reward = max(baseline_overall, key=lambda row: row["mean_reward"]) if baseline_overall else None
    baseline_best_regret = min(baseline_overall, key=lambda row: row["mean_regret"]) if baseline_overall else None

    best_summary = {
        "selector_test_greedy_best_by_reward": best_reward,
        "selector_test_greedy_best_by_regret": best_regret,
        "baseline_test_greedy_best_by_reward": baseline_best_reward,
        "baseline_test_greedy_best_by_regret": baseline_best_regret,
    }
    (output_dir / "best_summary.json").write_text(json.dumps(best_summary, ensure_ascii=False, indent=2), encoding="utf-8")

    # 当前 markdown 报告保持简洁，主要回答两个问题：
    # 1. 一共收集了哪些 run
    # 2. 当前 selector 里谁最好
    report_lines = [
        "# 2cmab2 结果分析",
        "",
        "## 已收集运行",
        "",
        f"- 运行数量: {len(runs)}",
        f"- 方法列表: {', '.join(sorted({run.method for run in runs}))}",
        "",
        "## Selector Test Greedy 最优结果",
        "",
        f"- 按 `mean_reward` 最佳: `{best_reward['method']}` = {_fmt(best_reward['mean_reward'])}",
        f"- 按 `mean_regret` 最佳: `{best_regret['method']}` = {_fmt(best_regret['mean_regret'])}",
        "",
        "## Baseline 对比",
        "",
    ]
    if baseline_best_reward is not None and baseline_best_regret is not None:
        report_lines.extend(
            [
                f"- baseline 按 `mean_reward` 最佳: `{baseline_best_reward['source']}` = {_fmt(baseline_best_reward['mean_reward'])}",
                f"- baseline 按 `mean_regret` 最佳: `{baseline_best_regret['source']}` = {_fmt(baseline_best_regret['mean_regret'])}",
                "",
                "## Selector vs Baseline",
                "",
                f"- selector 最佳 `mean_reward` 与 baseline 最佳的差值: {_fmt(float(best_reward['mean_reward']) - float(baseline_best_reward['mean_reward']))}",
                f"- selector 最佳 `mean_regret` 与 baseline 最佳的差值: {_fmt(float(best_regret['mean_regret']) - float(baseline_best_regret['mean_regret']))}",
                "",
            ]
        )
    report_lines.extend(
        [
        "## 输出文件",
        "",
        "- `runs_manifest.csv`: 运行清单",
        "- `selector_test_greedy.csv`: selector 的 greedy 测试指标",
        "- `selector_test_ucb.csv`: selector 的 UCB 测试指标",
        "- `baseline_test_greedy.csv`: baseline 测试指标",
        "- `selector_overall_ranking.csv`: selector 总体排名表",
        "- `best_summary.json`: 最佳结果摘要",
        ]
    )
    (output_dir / "report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")


def parse_args():
    """命令行参数。"""
    parser = argparse.ArgumentParser(
        description="聚合 2cmab2 的 summary.json 并生成结果分析表",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input-root", default=AnalysisDefaults.INPUT_ROOT)
    parser.add_argument("--output-dir", default=AnalysisDefaults.OUTPUT_DIR)
    return parser.parse_args()


def main():
    """CLI 入口。"""
    args = parse_args()
    input_root = Path(args.input_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    build_analysis_report(input_root=input_root, output_dir=output_dir)
    print(f"分析结果已写入: {output_dir}")


if __name__ == "__main__":
    main()
