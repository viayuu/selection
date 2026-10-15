"""Plot training winners, with a separate chart for each candidate scenario."""

import argparse
import csv
import json
import os
from pathlib import Path
import zipfile

os.environ["CUDA_VISIBLE_DEVICES"] = ""
for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "2"

from code.V4.winner_distribution_analysis import DEFAULT_OUTPUT, SCENARIOS, render_html, write_csv
from code.unified_selector.registry import PROBLEMS

os.environ.setdefault("MPLCONFIGDIR", str(DEFAULT_OUTPUT / "figures/.mplconfig"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import font_manager
from matplotlib.ticker import PercentFormatter
import numpy as np


COLORS = ("#147D80", "#C44E52", "#B48A32")
LABELS = ("原候选池", "只删除 RELD_MTL", "只删除 RELD_MOEL")


def style():
    family = "Noto Sans CJK SC"
    for path in ("/usr/share/fonts/google-noto-cjk/NotoSansCJK-Regular.ttc",
                 "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"):
        if Path(path).is_file():
            font_manager.fontManager.addfont(path)
            family = font_manager.FontProperties(fname=path).get_name()
            break
    plt.rcParams.update({
        "font.family": [family, "DejaVu Sans"],
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": False,
        "axes.edgecolor": "#A9B1B4",
        "text.color": "#202528",
        "axes.labelcolor": "#202528",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.dpi": 300,
        "axes.unicode_minus": False,
    })


def load_rows(root):
    with (root / "winner_distribution.csv").open(encoding="utf-8", newline="") as stream:
        rows = [row for row in csv.DictReader(stream)
                if row["problem"] in PROBLEMS and row["split"] == "train"]
    lookup = {}
    for row in rows:
        key = row["problem"], row["split"], row["scenario"], row["solver"]
        if key in lookup:
            raise ValueError("Duplicate winner distribution row: " + str(key))
        row["winner_pct"] = float(row["winner_pct"])
        row["winner_count"] = int(row["winner_count"])
        row["instances"] = int(row["instances"])
        row["candidate_active"] = row["candidate_active"] == "True"
        lookup[key] = row
    return rows, lookup


def draw_scenario(problem, solvers, original_solvers, lookup, scenario, removed, label, color):
    height = max(5.2, 2.2 + 0.44 * len(solvers))
    fig, ax = plt.subplots(figsize=(9.4, height))
    fig.subplots_adjust(left=0.25, right=0.96, bottom=0.15, top=0.77)
    fig.suptitle(f"{problem}｜{label}", x=0.04, y=0.97,
                 ha="left", fontsize=19, fontweight="bold")
    n = lookup[problem, "train", scenario, solvers[0]]["instances"]
    ax.set_title(f"训练集：{n:,}例｜合法候选：{len(solvers)}个", loc="left", pad=12, fontsize=12)
    position = np.arange(len(solvers))
    values = [lookup[problem, "train", scenario, solver]["winner_pct"] for solver in solvers]
    ax.barh(position, values, height=0.60, color=color, zorder=3)
    for y, value in zip(position, values):
        if value > 90:
            ax.text(value - 1.1, y, f"{value:.2f}%", ha="right", va="center",
                    fontsize=11, color="white", fontweight="bold")
        else:
            ax.text(value + 1.1, y, f"{value:.2f}%", va="center", fontsize=11)
    ax.set_yticks(position)
    ax.set_yticklabels(solvers)
    ax.set_ylim(len(solvers) - 0.4, -0.6)
    ax.set_xlim(0, 100)
    ax.set_xticks(np.arange(0, 101, 20))
    ax.xaxis.set_major_formatter(PercentFormatter(100, decimals=0))
    ax.set_xlabel("获胜实例占比")
    ax.grid(axis="x", color="#E2E7E8", linewidth=0.7, zorder=0)
    ax.tick_params(axis="y", length=0, pad=8)
    note = f"分母：本问题全部{n:,}个训练实例。"
    if removed is not None:
        note += ("原候选池不含该方法，分布不变。" if removed not in original_solvers
                 else "被删除方法不参与，其余方法的比例合计100%。")
    fig.text(0.04, 0.035, note, fontsize=9.5, color="#59656A")
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    root = args.root
    figures = root / "figures_train"
    tables = root / "tables_train"
    figures.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)
    rows, lookup = load_rows(root)
    style()
    sections = [
        "# 18种问题：训练集各方法获胜实例占比",
        "只统计训练集，每种问题10,000例。各问题独立展示，不混合汇总。",
        "每个问题分为三套独立图表：原候选池、只删除RELD_MTL、只删除RELD_MOEL。每张图中，每个方法只有一根柱，横轴统一为0–100%。",
        "表格分别列出获胜数量和比例，每张表的获胜数量合计10,000。被删除的方法不参与对应图表。",
        "Top1分布是方法成为真实winner的比例，不是选择模型准确率。原池沿用原标签；删除时仅将被删方法的winner按剩余FP64成本重新分配，完全并列按原成本列顺序。原数据未修改。",
        "[下载54页训练集图表PDF](winner_distributions_train.pdf) · [下载训练集图表包](per_problem_bundle.zip) · [全部训练集CSV](train_winner_distribution.csv)",
        "**问题目录：** " + " · ".join(f"[{p}](#{p.lower()})" for p in PROBLEMS),
    ]
    training_rows, manifest = [], []
    pdf_path = root / "winner_distributions_train.pdf"
    with PdfPages(pdf_path) as pdf:
        for problem in PROBLEMS:
            original_solvers = [row["solver"] for row in rows if row["problem"] == problem
                                and row["scenario"] == "original"]
            sections.append(f"## {problem}")
            for (scenario, removed), label, color in zip(SCENARIOS, LABELS, COLORS):
                solvers = [solver for solver in original_solvers if solver != removed]
                scenario_rows = []
                table = ["| 方法 | 获胜数量 | 占训练实例比例 |", "| --- | ---: | ---: |"]
                for solver in solvers:
                    row = lookup[problem, "train", scenario, solver]
                    scenario_rows.append(dict(
                        problem=problem, split="train", scenario=scenario,
                        removed_solver=removed or "", solver=solver,
                        instances=row["instances"], winner_count=row["winner_count"],
                        winner_pct=row["winner_pct"],
                    ))
                    table.append(f"| {solver} | {row['winner_count']:,} | {row['winner_pct']:.2f}% |")
                training_rows.extend(scenario_rows)
                stem = f"{problem}_{scenario}"
                write_csv(tables / f"{stem}.csv", scenario_rows)
                fig = draw_scenario(problem, solvers, original_solvers, lookup,
                                    scenario, removed, label, color)
                fig.savefig(figures / f"{stem}.png")
                fig.savefig(figures / f"{stem}.pdf")
                pdf.savefig(fig)
                plt.close(fig)
                sections.extend((f"### {label}",
                                 f"![{problem}训练集：{label}](figures_train/{stem}.png)",
                                 "\n".join(table),
                                 f"[本情况CSV](tables_train/{stem}.csv) · [本图PDF](figures_train/{stem}.pdf)"))
                manifest.append(dict(problem=problem, split="train", scenario=scenario,
                                     removed_solver=removed, n=10000, solvers=solvers,
                                     png=f"figures_train/{stem}.png",
                                     pdf=f"figures_train/{stem}.pdf",
                                     table=f"tables_train/{stem}.csv"))
            print(f"Rendered {problem}", flush=True)
    sections.append("备注：带B任务仍使用原成本场景，本次不修复其已知语义问题；没有训练模型或复跑solver。")
    source = root / "by_problem.md"
    source.write_text("\n\n".join(sections) + "\n", encoding="utf-8")
    render_html(source)
    write_csv(root / "by_problem_comparison.csv", training_rows)
    write_csv(root / "train_winner_distribution.csv", training_rows)
    protocol = json.loads((root / "protocol.json").read_text(encoding="utf-8"))
    protocol["sources"] = [row for row in protocol["sources"] if row["split"] == "train"]
    protocol["display_split"] = "train"
    (root / "train_protocol.json").write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf-8")
    (root / "figure_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    with zipfile.ZipFile(root / "per_problem_bundle.zip", "w", zipfile.ZIP_DEFLATED) as bundle:
        for name in ("by_problem.md", "by_problem.html", "by_problem_comparison.csv",
                     "train_winner_distribution.csv", "train_protocol.json",
                     "winner_distributions_train.pdf", "figure_manifest.json"):
            bundle.write(root / name, name)
        for directory in (figures, tables):
            for path in sorted(directory.glob("*")):
                if path.is_file():
                    bundle.write(path, str(path.relative_to(root)))
        for name in ("winner_distribution_analysis.py", "winner_distribution_figures.py"):
            bundle.write(Path(__file__).with_name(name), "scripts/" + name)
    print(f"Report: {root / 'by_problem.html'}", flush=True)
    print(f"PDF: {pdf_path}", flush=True)


if __name__ == "__main__":
    main()
