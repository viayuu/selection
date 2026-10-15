"""Plot one released dataset's train/test winner counts, without pool changes."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import zipfile

from code.V4.winner_distribution_analysis import render_html, write_csv
from code.V4.winner_distribution_figures import style

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import PercentFormatter
import numpy as np


DEFAULT_OUTPUT = Path(__file__).resolve().parent / "runs/Winner_distribution_new_only_B_single_reld"
SPLITS = ("train", "test")


def read_distribution(data_root, manifest):
    with (data_root / "winner_distribution.csv").open(encoding="utf-8", newline="") as stream:
        source = [row for row in csv.DictReader(stream) if row["split"] in SPLITS]
    lookup = {}
    for row in source:
        key = row["problem"], row["split"], row["solver"]
        if key in lookup:
            raise ValueError("Duplicate distribution row: " + str(key))
        row["instances"] = int(row["instances"])
        row["winner_count"] = int(row["winner_count"])
        if row["winner_count"] < 0:
            raise ValueError("Negative winner count")
        lookup[key] = row
    rows = []
    expected_keys = set()
    for problem, info in manifest["problems"].items():
        methods = info["methods"]
        for split in SPLITS:
            n = int(manifest["split_sizes_per_problem"][split])
            selected = []
            for solver in methods:
                key = problem, split, solver
                expected_keys.add(key)
                row = lookup[key]
                if row["instances"] != n:
                    raise ValueError(f"{problem}/{split}: inconsistent denominator")
                selected.append(dict(problem=problem, split=split, solver=solver,
                                     instances=n, winner_count=row["winner_count"],
                                     winner_pct=100 * row["winner_count"] / n))
            if sum(row["winner_count"] for row in selected) != n:
                raise ValueError(f"{problem}/{split}: winner counts do not sum to {n}")
            rows.extend(selected)
    if set(lookup) != expected_keys:
        raise ValueError("Winner distribution methods do not match dataset metadata")
    return rows, {(r["problem"], r["split"], r["solver"]): r for r in rows}


def draw_problem(problem, methods, lookup):
    height = max(5.6, 2.2 + 0.47 * len(methods))
    fig, axes = plt.subplots(1, 2, figsize=(16, height))
    fig.subplots_adjust(left=0.15, right=0.98, bottom=0.14, top=0.76, wspace=0.56)
    fig.suptitle(f"{problem}｜各方法获胜实例占比", x=0.05, y=0.97,
                 ha="left", fontsize=21, fontweight="bold")
    positions = np.arange(len(methods))
    for ax, split, label, color in zip(axes, SPLITS, ("训练集", "测试集"), ("#147D80", "#C44E52")):
        n = lookup[problem, split, methods[0]]["instances"]
        ax.set_title(f"{label}：{n:,}例", pad=12, fontsize=14, fontweight="bold")
        values = [lookup[problem, split, solver]["winner_pct"] for solver in methods]
        ax.barh(positions, values, height=0.60, color=color, zorder=3)
        for y, value in zip(positions, values):
            if value > 90:
                ax.text(value - 1.1, y, f"{value:.2f}%", ha="right", va="center",
                        fontsize=11, color="white", fontweight="bold")
            else:
                ax.text(value + 1.1, y, f"{value:.2f}%", va="center", fontsize=11)
        ax.set_yticks(positions)
        ax.set_yticklabels(methods)
        ax.set_ylim(len(methods) - 0.4, -0.6)
        ax.set_xlim(0, 100)
        ax.set_xticks(np.arange(0, 101, 20))
        ax.xaxis.set_major_formatter(PercentFormatter(100, decimals=0))
        ax.set_xlabel("获胜实例占比")
        ax.grid(axis="x", color="#E2E7E8", linewidth=0.7, zorder=0)
        ax.tick_params(axis="y", length=0, pad=8)
    fig.text(0.05, 0.03, "仅使用当前数据目录的实际候选池；每个方法一根柱，未模拟删除。", fontsize=10, color="#59656A")
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    data_root, output = args.data_root.resolve(), args.output_dir.resolve()
    manifest_path = data_root / "scenario_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows, lookup = read_distribution(data_root, manifest)
    figures, tables = output / "figures", output / "tables"
    figures.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)
    style()
    train_n = int(manifest["split_sizes_per_problem"]["train"])
    test_n = int(manifest["split_sizes_per_problem"]["test"])
    sections = [
        "# 当前数据：18种问题各方法获胜实例占比",
        f"数据目录：`{data_root}`。",
        f"每种问题：训练集{train_n:,}例、测试集{test_n:,}例。只统计现有候选池，不模拟删除，不包含validation。",
        "每个问题独立展示。左图为训练集，右图为测试集；每个方法一根柱，横轴统一为0–100%。表格同时保留获胜数量和比例。",
        "获胜比例使用该目录已发布的winner计数，分母为该问题对应划分的全部实例数，不是模型预测准确率。",
        "`reld_mt`对应原RELD_MOEL；RELD_MTL已移除。候选及顺序取自此目录的元数据，不沿用旧候选池或文件名字母排序。",
        "[18页图表PDF](winner_distributions.pdf) · [完整图表包](distribution_bundle.zip) · [训练与测试CSV](winner_distribution.csv)",
        "**问题目录：** " + " · ".join(f"[{p}](#{p.lower()})" for p in manifest["problems"]),
    ]
    wide, summary, assets = [], [], []
    pdf_path = output / "winner_distributions.pdf"
    with PdfPages(pdf_path) as pdf:
        for problem, info in manifest["problems"].items():
            methods = info["methods"]
            problem_rows = []
            table = ["| 方法 | 训练获胜数量 | 训练比例 | 测试获胜数量 | 测试比例 |",
                     "| --- | ---: | ---: | ---: | ---: |"]
            for solver in methods:
                train, test = lookup[problem, "train", solver], lookup[problem, "test", solver]
                record = dict(problem=problem, solver=solver,
                              train_n=train_n, train_winner_count=train["winner_count"],
                              train_winner_pct=train["winner_pct"],
                              test_n=test_n, test_winner_count=test["winner_count"],
                              test_winner_pct=test["winner_pct"])
                problem_rows.append(record)
                table.append(f"| {solver} | {train['winner_count']:,} | {train['winner_pct']:.2f}% | "
                             f"{test['winner_count']:,} | {test['winner_pct']:.2f}% |")
            wide.extend(problem_rows)
            write_csv(tables / f"{problem}.csv", problem_rows)
            leader = max(methods, key=lambda s: lookup[problem, "train", s]["winner_count"])
            summary.append(dict(problem=problem, candidates=len(methods),
                                train_most_frequent_winner=leader,
                                train_winner_pct=lookup[problem, "train", leader]["winner_pct"],
                                same_solver_test_winner_pct=lookup[problem, "test", leader]["winner_pct"]))
            fig = draw_problem(problem, methods, lookup)
            png_path, single_pdf = figures / f"{problem}.png", figures / f"{problem}.pdf"
            fig.savefig(png_path)
            fig.savefig(single_pdf)
            pdf.savefig(fig)
            plt.close(fig)
            sections.extend((f"## {problem}",
                             f"![{problem}各方法训练与测试获胜比例](figures/{problem}.png)",
                             "\n".join(table), f"[该问题CSV](tables/{problem}.csv) · [本图PDF](figures/{problem}.pdf)"))
            assets.append(dict(problem=problem, train_n=train_n, test_n=test_n,
                               methods=methods, png=f"figures/{problem}.png", pdf=f"figures/{problem}.pdf"))
            print(f"{problem}: train={train_n}, test={test_n}, candidates={len(methods)}", flush=True)
    sections.append("说明：直接使用数据目录已有的获胜计数，没有加载模型、重新求解、改动数据或重新划分。精确并列沿用该数据发布时的winner规则。")
    source = output / "by_problem.md"
    source.write_text("\n\n".join(sections) + "\n", encoding="utf-8")
    render_html(source)
    write_csv(output / "winner_distribution.csv", rows)
    write_csv(output / "comparison.csv", wide)
    write_csv(output / "summary.csv", summary)
    protocol = dict(data_root=str(data_root), scenario_id=manifest["scenario_id"],
                    splits=list(SPLITS), train_n=train_n, test_n=test_n,
                    source_counts=str(data_root / "winner_distribution.csv"),
                    counts_sha256=hashlib.sha256((data_root / "winner_distribution.csv").read_bytes()).hexdigest(),
                    manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
                    candidate_order="scenario_manifest.problems[problem].methods",
                    altered_candidates=False, altered_labels=False,
                    pools={p: info["methods"] for p, info in manifest["problems"].items()})
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf-8")
    (output / "figure_manifest.json").write_text(json.dumps(assets, indent=2) + "\n", encoding="utf-8")
    with zipfile.ZipFile(output / "distribution_bundle.zip", "w", zipfile.ZIP_DEFLATED) as bundle:
        for name in ("by_problem.md", "by_problem.html", "winner_distribution.csv", "comparison.csv",
                     "summary.csv", "winner_distributions.pdf", "protocol.json", "figure_manifest.json"):
            bundle.write(output / name, name)
        for directory in (figures, tables):
            for path in sorted(directory.glob("*")):
                if path.is_file():
                    bundle.write(path, str(path.relative_to(output)))
        for name in ("dataset_winner_distribution.py", "winner_distribution_analysis.py", "winner_distribution_figures.py"):
            bundle.write(Path(__file__).with_name(name), "scripts/" + name)
    print(f"Report: {output / 'by_problem.html'}", flush=True)
    print("Train-defined most frequent winner per problem:", flush=True)
    for record in summary:
        print(f"{record['problem']}: {record['train_most_frequent_winner']} "
              f"{record['train_winner_pct']:.2f}% / {record['same_solver_test_winner_pct']:.2f}%", flush=True)


if __name__ == "__main__":
    main()
