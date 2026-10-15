"""CPU-only R79 distribution report from four saved prediction archives."""

import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import PercentFormatter


ROOT = Path(__file__).resolve().parent / "runs" / "R79_NSS_new_data_batch1024"
OUTPUT = ROOT / "selection_distribution"
FONT = Path("/usr/share/fonts/google-noto-cjk/NotoSansCJK-Regular.ttc")
TASKS = {"TSP": ("TSPLIB", 49, 8), "CVRP": ("CVRPLIB", 100, 10)}
MODEL_SELECTION = "\u6a21\u578b\u9009\u62e9"
MODEL_TOP1 = "\u6a21\u578bTop1"
TOP1_LABEL_RULE = "saved pred == saved winner (native/FP64 ind)"
TRUE_WINNER = "\u771f\u5b9e winner"
NEW_TEST = "\u65b0 TEST"
FOOTNOTE = (
    "\u771f\u5b9ewinner\u4f7f\u7528\u539f\u751f\u6807\u7b7e\uff0c"
    "NSS\u5386\u53f2Top1\u53e6\u6709FP32 argmin\u53e3\u5f84\uff1b"
    "\u4ec5\u8bfb\u4fdd\u5b58\u9884\u6d4b\uff0c\u65e0\u8bad\u7ec3/\u63a8\u7406"
)


def read_json(path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def read_predictions(path, expected_n, expected_k):
    with np.load(path, allow_pickle=False) as archive:
        pool = archive["pool"].tolist()
        labels = {field: archive[field] for field in ("pred", "winner")}
    if len(pool) != expected_k or len(set(pool)) != expected_k:
        raise ValueError(f"Invalid candidate pool: {path}")
    for field, values in labels.items():
        if values.shape != (expected_n,) or values.dtype.kind not in "iu":
            raise ValueError(f"Invalid {field} shape/dtype: {path}")
        if np.any(values < 0) or np.any(values >= expected_k):
            raise ValueError(f"Out-of-pool {field} labels: {path}")
    return pool, labels


def make_panel(key, title, split, field, pool, labels, canonical_pool):
    if set(pool) != set(canonical_pool):
        raise ValueError(f"Candidate mismatch for {split}")
    # Label indices belong to the archive's pool; align counts by solver name.
    local_counts = np.bincount(labels[field], minlength=len(pool))
    counts = local_counts[[pool.index(solver) for solver in canonical_pool]]
    n = len(labels[field])
    if int(counts.sum()) != n:
        raise ValueError(f"Count total does not match n for {key}")
    matches = int(np.count_nonzero(labels["pred"] == labels["winner"])) if field == "pred" else None
    return {
        "key": key,
        "title": title,
        "split": split,
        "field": field,
        "n": n,
        "counts": counts,
        "percentages": counts * (100.0 / n),
        "model_top1_native_match_count": matches,
        "model_top1_native_pct": 100.0 * matches / n if matches is not None else None,
    }


def plot_task(task, pool, panels, checkpoint, sources):
    fig, axes = plt.subplots(1, 4, figsize=(24, 8.4))
    fig.subplots_adjust(left=0.075, right=0.985, top=0.70, bottom=0.225, wspace=0.62)
    fig.text(
        0.025, 0.95, f"{task} | NSS retrained on new data | LIB / {NEW_TEST}",
        fontsize=22, fontweight="bold", va="top",
    )
    epoch = checkpoint.get("best_epoch", "unknown")
    digest = checkpoint.get("sha256", "not recorded")[:12]
    fig.text(
        0.025, 0.895,
        f"Model: {ROOT.name} | {task}/best.pt | epoch {epoch} | stored SHA256: {digest}",
        fontsize=11.5,
    )
    test_source = Path(sources["test"]["original_directory"])
    lib_split = panels[0]["split"]
    lib_source = sources[lib_split]["original_directory"]
    fig.text(
        0.025, 0.855,
        f"New TEST data: {test_source.parent.name}/{test_source.name}",
        fontsize=11.5,
    )
    fig.text(0.025, 0.815, f"LIB data: {lib_source}", fontsize=11.5)
    for ax, panel in zip(axes, panels):
        color = "#147f83" if panel["field"] == "pred" else "#c44e52"
        y = np.arange(len(pool))
        ax.barh(y, panel["percentages"], height=0.6, color=color, zorder=3)
        ax.set_yticks(y, labels=pool, fontsize=10.5)
        ax.set_ylim(len(pool) - 0.4, -0.6)
        ax.set_xlim(0, 100)
        ax.set_xticks(np.arange(0, 101, 20))
        ax.xaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
        ax.tick_params(axis="y", length=0)
        ax.tick_params(axis="x", labelsize=10)
        ax.grid(axis="x", color="#e2e6e8", linewidth=0.7, zorder=0)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color("#aab2b7")
        dataset = f"{task}test" if panel["split"] == "test" else panel["split"]
        accuracy_line = (
            f"{MODEL_TOP1}={panel['model_top1_native_pct']:.2f}% (native/FP64)"
            if panel["field"] == "pred" else "native/FP64 winner labels"
        )
        ax.set_title(
            f"{panel['title']}\n{dataset} | n = {panel['n']:,}\n{accuracy_line}",
            fontsize=13, fontweight="bold", pad=14,
        )
        ax.set_xlabel("Share of dataset instances (%)", fontsize=10.5, labelpad=10)
        for row, (count, pct) in enumerate(zip(panel["counts"], panel["percentages"])):
            inside = pct >= 60
            ax.text(
                pct - 1.5 if inside else pct + 1.2, row,
                f"{pct:.2f}%  ({count:,})", va="center",
                ha="right" if inside else "left", fontsize=9.5,
                color="white" if inside else "#24292c",
            )
    fig.text(
        0.025, 0.115,
        "Labels: percentage (count). All candidates retained, including zero counts; "
        "fixed TEST-pool order; identical 0-100% scale.", fontsize=11,
    )
    fig.text(
        0.025, 0.077,
        "Model choices = saved pred; true winners = saved native/FP64 ind winner. "
        "Model Top1 = per-instance mean(pred == winner), not the largest selection share.", fontsize=11,
    )
    fig.text(0.025, 0.039, FOOTNOTE, fontsize=11)
    return fig


def write_csv(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    if not FONT.is_file():
        raise FileNotFoundError(FONT)
    font_manager.fontManager.addfont(str(FONT))
    plt.rcParams.update({
        "font.family": font_manager.FontProperties(fname=str(FONT)).get_name(),
        "pdf.fonttype": 42,
        "axes.unicode_minus": False,
        "text.color": "#24292c",
        "axes.labelcolor": "#24292c",
        "xtick.color": "#24292c",
        "ytick.color": "#24292c",
        "savefig.facecolor": "white",
    })
    lock_path = ROOT / "final_test_lock.json"
    lock = read_json(lock_path) if lock_path.is_file() else {}
    protocol = {
        "model_version": ROOT.name,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "execution": "CPU-only saved-prediction reporting; no training or inference",
        "model_selection_rule": "Use saved pred directly; no score argmax or tie-break changes",
        "true_winner_rule": "Use saved winner directly (native/FP64 ind), not FP32 cost argmin",
        "model_top1_rule": "100 * count(saved pred == saved winner) / n; native/FP64 ind labels",
        "model_top1_csv_scope": "Dataset-level accuracy repeated only on model-selection rows; blank on true-winner rows",
        "historical_nss_top1_rule": "FP32 cost argmin; not recomputed or mixed into this report",
        "panel_order": ["lib_model_selection", "lib_true_winner", "test_model_selection", "test_true_winner"],
        "candidate_order": "Each task's test_predictions.npz pool; LIB counts aligned by name",
        "percentages": "100 * count / panel sample count; include zero-count solvers",
        "checkpoint_hash_policy": "Copy existing metadata only; never open or rehash weights",
        "font_path": str(FONT),
        "final_test_lock_path": str(lock_path) if lock else None,
        "tasks": {},
    }
    long_rows, wide_rows, task_reports = [], [], []
    for task, (lib_split, lib_n, k) in TASKS.items():
        archives, sources, checkpoints = {}, {}, {}
        for split, n in (("test", 3895), (lib_split, lib_n)):
            prediction_path = ROOT / task / f"{split}_predictions.npz"
            metadata_path = ROOT / task / f"{split}_metrics.json"
            metadata = read_json(metadata_path)
            archives[split] = read_predictions(prediction_path, n, k)
            checkpoints[split] = metadata.get("checkpoint") or lock.get("checkpoints", {}).get(task)
            if not checkpoints[split]:
                raise ValueError(f"Missing checkpoint metadata for {task}/{split}")
            sources[split] = {
                "prediction_path": str(prediction_path),
                "metadata_path": str(metadata_path),
                "original_directory": metadata["original_directory"],
                "n": n,
                "archive_pool_order": archives[split][0],
            }
        checkpoint = checkpoints["test"]
        for key in ("path", "sha256", "best_epoch"):
            if checkpoint.get(key) != checkpoints[lib_split].get(key):
                raise ValueError(f"TEST/LIB checkpoint metadata mismatch: {task}, {key}")
        pool = archives["test"][0]
        specifications = (
            ("lib_model_selection", f"LIB {MODEL_SELECTION}", lib_split, "pred"),
            ("lib_true_winner", f"LIB {TRUE_WINNER}", lib_split, "winner"),
            ("test_model_selection", f"{NEW_TEST} {MODEL_SELECTION}", "test", "pred"),
            ("test_true_winner", f"{NEW_TEST} {TRUE_WINNER}", "test", "winner"),
        )
        panels = [
            make_panel(key, title, split, field, *archives[split], pool)
            for key, title, split, field in specifications
        ]
        for panel_order, panel in enumerate(panels, start=1):
            source = sources[panel["split"]]
            for solver_order, solver in enumerate(pool, start=1):
                count = int(panel["counts"][solver_order - 1])
                long_rows.append({
                    "task": task, "solver": solver, "solver_order": solver_order,
                    "panel_order": panel_order, "group": panel["key"],
                    "dataset": f"{task}test" if panel["split"] == "test" else panel["split"],
                    "label_field": panel["field"], "n_instances": panel["n"],
                    "count": count, "proportion": count / panel["n"],
                    "percentage": float(panel["percentages"][solver_order - 1]),
                    "model_top1_native_match_count": panel["model_top1_native_match_count"],
                    "model_top1_native_pct": panel["model_top1_native_pct"],
                    "model_top1_label_rule": TOP1_LABEL_RULE if panel["field"] == "pred" else "",
                    "model_version": ROOT.name, "data_source": source["original_directory"],
                    "prediction_source": source["prediction_path"],
                    "checkpoint_path": checkpoint["path"],
                })
        for index, solver in enumerate(pool):
            row = {
                "task": task, "solver": solver, "solver_order": index + 1,
                "lib_dataset": lib_split, "test_dataset": f"{task}test",
                "lib_n_instances": lib_n, "test_n_instances": 3895,
            }
            for panel in panels:
                row[f"{panel['key']}_n"] = int(panel["counts"][index])
                row[f"{panel['key']}_pct"] = float(panel["percentages"][index])
                if panel["field"] == "pred":
                    split_group = "test" if panel["split"] == "test" else "lib"
                    row[f"{split_group}_model_top1_native_match_count"] = panel["model_top1_native_match_count"]
                    row[f"{split_group}_model_top1_native_pct"] = panel["model_top1_native_pct"]
            row["model_top1_label_rule"] = TOP1_LABEL_RULE
            wide_rows.append(row)
        protocol["tasks"][task] = {
            "checkpoint": checkpoint, "candidate_order": pool, "sources": sources,
            "model_top1_native": {
                panel["split"]: {
                    "matched_instances": panel["model_top1_native_match_count"],
                    "n_instances": panel["n"],
                    "percentage": panel["model_top1_native_pct"],
                    "label_rule": TOP1_LABEL_RULE,
                }
                for panel in panels if panel["field"] == "pred"
            },
            "count_checks": {
                panel["key"]: {"total_count": int(panel["counts"].sum()), "expected_n": panel["n"]}
                for panel in panels
            },
        }
        task_reports.append((task, pool, panels, checkpoint, sources))

    figures = OUTPUT / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    with PdfPages(OUTPUT / "selection_distributions.pdf") as pdf:
        pdf.infodict().update({"Title": "R79 NSS new-data selection distributions", "Subject": FOOTNOTE})
        for task, pool, panels, checkpoint, sources in task_reports:
            fig = plot_task(task, pool, panels, checkpoint, sources)
            pdf.savefig(fig)
            fig.savefig(figures / f"{task}.pdf")
            fig.savefig(figures / f"{task}.png", dpi=160)
            plt.close(fig)
            for panel in panels:
                nonzero = [
                    f"{solver}={count}" for solver, count in zip(pool, panel["counts"]) if count
                ]
                accuracy = (
                    f" | {MODEL_TOP1}={panel['model_top1_native_pct']:.2f}% (native/FP64)"
                    if panel["field"] == "pred" else ""
                )
                print(f"{task} {panel['key']} (n={panel['n']}){accuracy}: " + ", ".join(nonzero))
        protocol["pdf_pages"] = pdf.get_pagecount()
        if protocol["pdf_pages"] != 2:
            raise ValueError("The combined PDF must contain exactly two pages")
    write_csv(OUTPUT / "selection_distribution.csv", long_rows)
    write_csv(OUTPUT / "comparison.csv", wide_rows)
    with (OUTPUT / "protocol.json").open("w", encoding="utf-8") as handle:
        json.dump(protocol, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(f"Wrote {len(long_rows)} long rows, {len(wide_rows)} comparison rows, 2 PDF pages to {OUTPUT}")


if __name__ == "__main__":
    main()
