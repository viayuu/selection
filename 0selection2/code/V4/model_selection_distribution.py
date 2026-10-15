"""Report one locked model's saved TEST choices without another forward pass."""

import argparse
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


def load_predictions(run_root, expected):
    paths = sorted((run_root / "evaluation/test/selected_best").glob("rank*.npz"))
    if not paths:
        raise FileNotFoundError("No saved selected_best TEST predictions")
    parts = {}
    for path in paths:
        with np.load(path, allow_pickle=False) as saved:
            for key in saved.files:
                if not key.endswith("__logits"):
                    continue
                problem = key[:-len("__logits")]
                prefix = problem + "__"
                part = {name: saved[prefix + name].copy()
                        for name in ("logits", "indices", "winner", "pool", "pool_ids")}
                parts.setdefault(problem, []).append(part)
    if set(parts) != set(expected):
        raise ValueError("Saved predictions must cover exactly the reported problems")
    result = {}
    for problem, chunks in parts.items():
        pool, ids = chunks[0]["pool"], chunks[0]["pool_ids"]
        for chunk in chunks:
            if not np.array_equal(chunk["pool"], pool) or not np.array_equal(chunk["pool_ids"], ids):
                raise ValueError("Inconsistent solver identity across prediction shards")
        scores = np.concatenate([chunk["logits"] for chunk in chunks])
        indices = np.concatenate([chunk["indices"] for chunk in chunks])
        winners = np.concatenate([chunk["winner"] for chunk in chunks])
        n = int(expected[problem]["n"])
        if not np.array_equal(np.sort(indices), np.arange(n)):
            raise ValueError(f"{problem}: duplicated or missing TEST instances")
        if scores.shape != (n, len(pool)) or np.isnan(scores).any():
            raise ValueError(f"{problem}: invalid saved scores")
        # The original policy resolves exact score ties by the global solver ID.
        order = np.argsort(ids, kind="stable")
        selected = order[scores[:, order].argmax(axis=1)]
        selected_counts = np.bincount(selected, minlength=len(pool))
        winner_counts = np.bincount(winners, minlength=len(pool))
        correct_counts = np.bincount(selected[selected == winners], minlength=len(pool))
        if not np.allclose(selected_counts / n, expected[problem]["pick_dist"], rtol=0, atol=1e-12):
            raise ValueError(f"{problem}: replayed choices differ from the saved TEST metrics")
        result[problem] = dict(n=n, pool=pool.tolist(), ids=ids.tolist(),
                               selected_counts=selected_counts, winner_counts=winner_counts,
                               correct_counts=correct_counts,
                               top1_pct=100 * float((selected == winners).mean()))
    return result, paths


def plot_problem(problem, value):
    methods, n = value["pool"], value["n"]
    height = max(5.6, 2.2 + 0.47 * len(methods))
    fig, axes = plt.subplots(1, 2, figsize=(16, height))
    fig.subplots_adjust(left=0.15, right=0.98, bottom=0.14, top=0.76, wspace=0.56)
    fig.suptitle(f"{problem}｜模型选择与真实winner分布", x=0.05, y=0.97,
                 ha="left", fontsize=21, fontweight="bold")
    positions = np.arange(len(methods))
    for ax, key, label, color in zip(axes, ("selected_counts", "winner_counts"),
                                   ("模型选择", "真实winner"), ("#147D80", "#C44E52")):
        ax.set_title(f"{label}｜测试集：{n:,}例", pad=12, fontsize=14, fontweight="bold")
        values = value[key] / n * 100
        ax.barh(positions, values, height=0.60, color=color, zorder=3)
        for y, fraction in zip(positions, values):
            if fraction > 90:
                ax.text(fraction - 1.1, y, f"{fraction:.2f}%", ha="right", va="center",
                        fontsize=11, color="white", fontweight="bold")
            else:
                ax.text(fraction + 1.1, y, f"{fraction:.2f}%", va="center", fontsize=11)
        ax.set_yticks(positions)
        ax.set_yticklabels(methods)
        ax.set_ylim(len(methods) - 0.4, -0.6)
        ax.set_xlim(0, 100)
        ax.set_xticks(np.arange(0, 101, 20))
        ax.xaxis.set_major_formatter(PercentFormatter(100, decimals=0))
        ax.set_xlabel("占该问题测试实例比例")
        ax.grid(axis="x", color="#E2E7E8", linewidth=0.7, zorder=0)
        ax.tick_params(axis="y", length=0, pad=8)
    fig.text(0.05, 0.03, "左图是模型选谁，右图是实际谁赢；模型选择占比不是准确率。", fontsize=10, color="#59656A")
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    run_root = args.run_root.resolve()
    output = args.output_dir.resolve() if args.output_dir else run_root / "test_selection_distribution"
    outcome = json.loads((run_root / "test_results.json").read_text(encoding="utf-8"))
    lock = json.loads((run_root / "final_test_lock.json").read_text(encoding="utf-8"))
    expected = outcome["result"]["per_problem"]
    predictions, prediction_paths = load_predictions(run_root, expected)
    figures, tables = output / "figures", output / "tables"
    figures.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)
    style()
    sections = [
        "# 模型test结果：各问题的方法选择情况",
        f"结果目录：`{run_root}`。使用已锁定权重的完整test预测，不重新推理、不更换checkpoint。",
        "18种问题分别统计。左图表示模型选择比例，右图表示真实winner比例；每个比例的分母只使用该问题的测试实例数。",
        "表格并列模型选择数量／比例与真实winner数量／比例。差值为选择比例减winner比例，单位为百分点，不代表该方法的选择一定合理或不合理。",
        "`reld_mt`沿用保存预测中的方法名称，对应当前数据中保留的MOEL部署。",
        "[18页PDF](selection_distributions.pdf) · [全部选择分布CSV](selection_distribution.csv) · [完整图表包](selection_distribution_bundle.zip)",
        "**问题目录：** " + " · ".join(f"[{p}](#{p.lower()})" for p in expected),
    ]
    rows, summary = [], []
    with PdfPages(output / "selection_distributions.pdf") as pdf:
        for problem in expected:
            value = predictions[problem]
            n, methods = value["n"], value["pool"]
            problem_rows = []
            table = ["| 方法 | 模型选择数量 | 选择比例 | 真实winner数量 | winner比例 | 比例差 pp |",
                     "| --- | ---: | ---: | ---: | ---: | ---: |"]
            for slot, (solver, solver_id) in enumerate(zip(methods, value["ids"])):
                chosen, won = int(value["selected_counts"][slot]), int(value["winner_counts"][slot])
                chosen_pct, won_pct = 100 * chosen / n, 100 * won / n
                record = dict(problem=problem, split="test", solver=solver, solver_id=solver_id,
                              instances=n, selected_count=chosen, selected_pct=chosen_pct,
                              winner_count=won, winner_pct=won_pct,
                              selection_minus_winner_pp=chosen_pct - won_pct,
                              correct_selected_count=int(value["correct_counts"][slot]))
                problem_rows.append(record)
                table.append(f"| {solver} | {chosen:,} | {chosen_pct:.2f}% | {won:,} | "
                             f"{won_pct:.2f}% | {chosen_pct - won_pct:+.2f} |")
            rows.extend(problem_rows)
            write_csv(tables / f"{problem}.csv", problem_rows)
            most_selected = int(value["selected_counts"].argmax())
            summary.append(dict(problem=problem, instances=n, candidates=len(methods),
                                methods_ever_selected=int((value["selected_counts"] > 0).sum()),
                                most_selected_solver=methods[most_selected],
                                most_selected_count=int(value["selected_counts"][most_selected]),
                                most_selected_pct=100 * int(value["selected_counts"][most_selected]) / n,
                                same_solver_winner_pct=100 * int(value["winner_counts"][most_selected]) / n,
                                model_top1_pct=value["top1_pct"]))
            fig = plot_problem(problem, value)
            fig.savefig(figures / f"{problem}.png")
            fig.savefig(figures / f"{problem}.pdf")
            pdf.savefig(fig)
            plt.close(fig)
            sections.extend((f"## {problem}",
                             f"测试集{n:,}例，模型Top1为{value['top1_pct']:.2f}%。",
                             f"![{problem}模型选择与真实winner](figures/{problem}.png)",
                             "\n".join(table), f"[该问题CSV](tables/{problem}.csv) · [本图PDF](figures/{problem}.pdf)"))
            print(f"{problem}: {methods[most_selected]} selected "
                  f"{100 * int(value['selected_counts'][most_selected]) / n:.2f}%; "
                  f"nonzero methods={int((value['selected_counts'] > 0).sum())}/{len(methods)}", flush=True)
    source = output / "by_problem.md"
    source.write_text("\n\n".join(sections) + "\n", encoding="utf-8")
    render_html(source)
    write_csv(output / "selection_distribution.csv", rows)
    write_csv(output / "summary.csv", summary)
    protocol = dict(run_root=str(run_root), split="test", checkpoint=lock["checkpoint"],
                    checkpoint_sha256=lock["checkpoint_sha256"], selected_epoch=lock["epoch"],
                    source_tag="evaluation/test/selected_best",
                    selection_rule="maximum saved logit; exact ties use ascending global solver ID",
                    new_forward_passes=0, predictions=[dict(path=str(path),
                    sha256=hashlib.sha256(path.read_bytes()).hexdigest()) for path in prediction_paths])
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf-8")
    with zipfile.ZipFile(output / "selection_distribution_bundle.zip", "w", zipfile.ZIP_DEFLATED) as bundle:
        for name in ("by_problem.md", "by_problem.html", "selection_distribution.csv", "summary.csv",
                     "selection_distributions.pdf", "protocol.json"):
            bundle.write(output / name, name)
        for directory in (figures, tables):
            for path in sorted(directory.glob("*")):
                if path.is_file():
                    bundle.write(path, str(path.relative_to(output)))
        for name in ("model_selection_distribution.py", "winner_distribution_analysis.py", "winner_distribution_figures.py"):
            bundle.write(Path(__file__).with_name(name), "scripts/" + name)
    print(f"Report: {output / 'by_problem.html'}", flush=True)


if __name__ == "__main__":
    main()
