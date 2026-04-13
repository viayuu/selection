from __future__ import annotations

import json
from pathlib import Path


def _code_cell(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


def _markdown_cell(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source.splitlines(keepends=True),
    }


def build_notebook() -> dict:
    """
    生成一个可以直接复盘单个 run 输出目录的 notebook。

    设计约束：
    1. 所有分析表默认只保留按问题拆分后的结果。
    2. top1 / top2 / top3 命中率统一放在同一张表中，方便横向比较。
    """
    cells = [
        _markdown_cell(
            """# 2cmab2 运行结果复盘 Notebook

这个 notebook 用来分析 `2cmab2/outputs/...` 下面某一次训练输出。

使用方式：

1. 先把下面配置单元里的 `RUN_DIR` 改成你要分析的输出目录。
2. 顺序执行全部单元。
3. 如果希望把 notebook 里的表格同时导出成 markdown / csv，就把 `EXPORT_ANALYSIS = True`。

这个 notebook 会重点输出：

- selector / baseline 指标
- top1 / top2 / top3 命中率同表展示
- test 集各方法自身的 mean cost，以及 selector 相比它们更好还是更差
- 选臂分布与选择集中度
- 每个 arm 被选中后的真实效果
- greedy 和 ucb 的差异
- oracle 最优 arm 分布与覆盖情况

说明：

- 所有表格默认只展示按问题拆分后的结果，例如 `tsp`、`cvrp`
"""
        ),
        _code_cell(
            """from pathlib import Path
import sys
from IPython.display import Markdown, display

REPO_ROOT = Path.cwd()
MODULE_DIR = REPO_ROOT / "2cmab2"
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

import output_analysis_utils as au

# 改这里：要分析的 run 输出目录
RUN_DIR = REPO_ROOT / "2cmab2/outputs/2026-03-23-23-07-19_neural_linucb"

# 如果你希望 notebook 同时把表格导出到 run_dir/notebook_analysis/ 下面，就设成 True
EXPORT_ANALYSIS = True
EXPORT_DIR = RUN_DIR / "notebook_analysis"
"""
        ),
        _code_cell(
            """analysis = au.build_analysis_bundle(RUN_DIR)
display(Markdown("# 运行结果总览"))
display(Markdown("\\n".join(analysis["overview_lines"])))
display(Markdown("## 阶段耗时"))
display(Markdown(au.rows_to_markdown_table(
    analysis["runtime"]["runtime_rows"],
    columns=["stage", "seconds", "readable"],
)))
"""
        ),
        _code_cell(
            """display(Markdown("## Selector 指标"))
display(Markdown(au.rows_to_markdown_table(
    analysis["selector_rows"],
    columns=["source", "split", "scope", "count", "mean_reward", "top1_accuracy", "mean_regret", "mean_cost"],
)))

display(Markdown("## Baseline 指标"))
display(Markdown(au.rows_to_markdown_table(
    analysis["baseline_rows"],
    columns=["source", "split", "scope", "count", "mean_reward", "top1_accuracy", "mean_regret", "mean_cost"],
)))
"""
        ),
        _code_cell(
            """display(Markdown(
    "## Top-k 命中分析\\n"
    "- 这里的 `top1/top2/top3` 是根据 trace 中的 `selected_reward` 反推出选中 arm 的 tie-aware 平均 rank 后计算的。\\n"
    "- 因此它适合回答“模型选中的方法有没有落在真实前 1 / 前 2 / 前 3 名”。\\n"
    "- 它和 `summary.json` 里的 `top1_accuracy` 定义不同，后者是 `selected_arm == best_arm`。"
))

for trace_name, rows in analysis["topk_by_trace"].items():
    display(Markdown(f"### {trace_name}"))
    display(Markdown(au.rows_to_markdown_table(
        rows,
        columns=["scope", "count", "mean_selected_rank", "top1_hit_rate", "top2_hit_rate", "top3_hit_rate"],
    )))
"""
        ),
        _code_cell(
            """for trace_name, rows in analysis["selection_summary_by_trace"].items():
    display(Markdown(f"## {trace_name} 选择集中度"))
    display(Markdown(au.rows_to_markdown_table(
        rows,
        columns=["scope", "count", "unique_selected_arms", "entropy", "effective_arms"],
    )))

for trace_name, rows in analysis["selection_distribution_by_trace"].items():
    display(Markdown(f"## {trace_name} 各 arm 被选比例"))
    display(Markdown(au.rows_to_markdown_table(
        rows,
        columns=["scope", "arm", "count", "fraction"],
    )))
"""
        ),
        _code_cell(
            """for trace_name, rows in analysis["selected_arm_quality_by_trace"].items():
    display(Markdown(f"## {trace_name} 每个 arm 被选中后的真实效果"))
    display(Markdown(au.rows_to_markdown_table(
        rows,
        columns=["scope", "arm", "count", "mean_reward", "mean_regret", "top1_rate", "mean_cost"],
    )))

if analysis.get("method_mean_cost_error"):
    display(Markdown("## test 集各方法 mean cost 与 selector 对比"))
    display(Markdown(f"- 该部分生成失败：`{analysis['method_mean_cost_error']}`"))
else:
    for trace_name, rows in analysis["selector_vs_methods_summary_by_trace"].items():
        display(Markdown(f"## {trace_name} selector 与各方法 mean cost 对比汇总"))
        display(Markdown(au.rows_to_markdown_table(
            rows,
            columns=[
                "scope",
                "selector_mean_cost",
                "n_methods",
                "methods_better_than_selector",
                "methods_worse_than_selector",
                "methods_equal_to_selector",
                "selector_rank_among_methods",
            ],
        )))

    for trace_name, rows in analysis["test_method_mean_cost_by_trace"].items():
        display(Markdown(f"## {trace_name} 各方法在 test 集上的 mean cost"))
        display(Markdown(au.rows_to_markdown_table(
            rows,
            columns=["scope", "arm", "count", "mean_cost", "selector_mean_cost", "delta_vs_selector", "selector_better"],
        )))

display(Markdown("## oracle 最优 arm 分布"))
display(Markdown(au.rows_to_markdown_table(
    analysis["best_arm_rows"],
    columns=["scope", "arm", "oracle_best_count", "oracle_best_fraction"],
)))

display(Markdown("## oracle 覆盖度"))
display(Markdown(au.rows_to_markdown_table(
    analysis["oracle_coverage_rows"],
    columns=["arm", "oracle_best_count", "selected_count", "selected_minus_oracle", "coverage_over_oracle"],
)))
"""
        ),
        _code_cell(
            """display(Markdown("## greedy 与 ucb 的差异"))
display(Markdown(au.rows_to_markdown_table(
    analysis["greedy_vs_ucb_summary"],
    columns=["scope", "total", "same_selection", "different_selection", "different_ratio", "ucb_better_when_diff", "ucb_worse_when_diff", "equal_when_diff"],
)))

display(Markdown("### 具体样例"))
display(Markdown(au.rows_to_markdown_table(
    analysis["greedy_vs_ucb_examples"],
    columns=["uid", "problem", "greedy_arm", "greedy_score", "greedy_regret", "ucb_arm", "ucb_score", "ucb_regret", "best_arm", "best_cost"],
)))
"""
        ),
        _code_cell(
            """if EXPORT_ANALYSIS:
    export_paths = au.export_analysis_bundle(analysis, EXPORT_DIR)
    display(Markdown("## 导出结果"))
    display(Markdown("\\n".join([f"- `{key}`: `{value}`" for key, value in export_paths.items()])))
else:
    display(Markdown("## 导出结果\\n当前未开启导出。把 `EXPORT_ANALYSIS` 改成 `True` 即可。"))
"""
        ),
    ]

    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.x",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> None:
    notebook = build_notebook()
    output_path = Path(__file__).resolve().parent / "analysis.ipynb"
    output_path.write_text(json.dumps(notebook, ensure_ascii=False, indent=2), encoding="utf-8")
    print(output_path)


if __name__ == "__main__":
    main()
