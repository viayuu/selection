"""CPU-only R77B statistics from locked predictions, without model imports."""

import argparse
import csv
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import PercentFormatter


ROOT = Path(__file__).resolve().parent / "runs/R77_sample_size_batch1024"
SHA256 = "31c0c75d66b1bde6ad533909a9c04b0036e80af4c4a5775b01b957ce310e6d1b"
FONT = Path("/usr/share/fonts/google-noto-cjk/NotoSansCJK-Regular.ttc")
LIBS = {"TSP": ("TSPLIB", 49), "CVRP": ("CVRPLIB", 100)}
MODEL_SELECTION = "\u6a21\u578b\u9009\u62e9"
MODEL_TOP1 = "\u6a21\u578bTop1"
TRUE_WINNER = "\u771f\u5b9e winner"
PERCENT_KEYS = ("top1_pct", "sbs_top1_pct", "vs_sbs_pct", "vs_oracle_pct", "actual_regret_pct")
METRIC_KEYS = ("top1_pct", "sbs_top1_pct", "mean_cost", "sbs_cost", "oracle_cost",
               "vs_sbs_pct", "vs_oracle_pct", "actual_regret_pct", "top2_pct", "top3_pct")


def read_json(path):
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def write_json(path, value):
    with path.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def write_csv(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


@dataclass
class Predictions:
    path: Path
    pool: list
    pred: np.ndarray
    winner: np.ndarray
    costs: np.ndarray
    indices: np.ndarray
    source_indices: object
    pool_ids: object
    score_ties: object

    @property
    def n(self):
        return len(self.winner)

    @property
    def top1_pct(self):
        return float(np.mean(self.pred == self.winner) * 100)

    @property
    def fp32_top1_pct(self):
        return float(np.mean(self.pred == self.costs.astype(np.float32).argmin(1)) * 100)

    @property
    def mean_cost(self):
        return float(self.costs[np.arange(self.n), self.pred].mean(dtype=np.float64))


def load_predictions(archive, path, prefix="", r77=True):
    def get(name):
        return archive[prefix + name]

    pool = get("pool").tolist()
    costs, winner = get("costs"), get("winner")
    n, k = costs.shape
    if costs.dtype != np.float64 or not np.isfinite(costs).all() or np.any(costs <= 0):
        raise ValueError(f"Expected finite positive FP64 costs: {path} {prefix}")
    if len(pool) != k or len(set(pool)) != k or winner.shape != (n,):
        raise ValueError(f"Invalid pool or labels: {path} {prefix}")
    pool_ids, ties = None, None
    if r77:
        pool_ids = get("pool_ids")
        if pool_ids.shape != (k,) or len(np.unique(pool_ids)) != k:
            raise ValueError(f"Invalid global solver IDs: {path} {prefix}")
        score_key = "scores" if prefix + "scores" in archive else "logits"
        scores = get(score_key)
        if scores.shape != costs.shape or not np.isfinite(scores).all():
            raise ValueError(f"Invalid saved decision scores: {path} {prefix}")
        order = np.argsort(pool_ids, kind="stable")
        pred = order[np.argmax(scores[:, order], axis=1)]
        ties = int(np.count_nonzero((scores == scores.max(1, keepdims=True)).sum(1) > 1))
        if prefix + "pred" in archive:
            np.testing.assert_array_equal(pred, get("pred"))
    else:
        # NSS uses its saved selection, including its original score-tie behavior.
        pred = get("pred")
    for name, values in (("pred", pred), ("winner", winner)):
        if values.shape != (n,) or values.dtype.kind not in "iu":
            raise ValueError(f"Invalid {name}: {path} {prefix}")
        if np.any(values < 0) or np.any(values >= k):
            raise ValueError(f"Out-of-pool {name}: {path} {prefix}")
    indices = get("indices")
    source = get("source_indices") if prefix + "source_indices" in archive else None
    if indices.shape != (n,) or len(np.unique(indices)) != n:
        raise ValueError(f"Invalid instance indices: {path} {prefix}")
    return Predictions(path, pool, pred, winner, costs, indices, source, pool_ids, ties)


def metric_row(problem, split, data, saved, baseline, data_version, metric_source):
    if baseline["selection_split"] != "train":
        raise ValueError("Main SBS must be fixed on FULL TRAIN")
    solver = baseline["sbs_solver"]
    slot = data.pool.index(solver)
    if saved["sbs_name"] != solver or saved.get("sbs_selection_split") != "train":
        raise ValueError(f"Non-TRAIN SBS in main metrics: {problem} {split}")
    np.testing.assert_allclose(data.top1_pct, saved["top1"] * 100, rtol=0, atol=1e-10)
    np.testing.assert_allclose(data.mean_cost, saved["mean_cost"], rtol=0, atol=1e-10)
    np.testing.assert_allclose(data.costs[:, slot].mean(), saved["sbs_cost"], rtol=0, atol=1e-10)
    return dict(
        model="R77B", split=split, problem=problem, n=data.n,
        top1_pct=float(saved["top1"] * 100),
        sbs_top1_pct=float(np.mean(data.winner == slot) * 100),
        mean_cost=float(saved["mean_cost"]), sbs_cost=float(saved["sbs_cost"]),
        oracle_cost=float(saved["oracle_cost"]), vs_sbs_pct=float(saved["vs_sbs_pct"]),
        vs_oracle_pct=float(saved["vs_oracle_pct"]),
        actual_regret_pct=float(saved["actual_regret_pct"]),
        top2_pct=float(saved["top2"] * 100), top3_pct=float(saved["top3"] * 100),
        sbs_name=solver, sbs_selection_split="FULL TRAIN (70114 instances/problem)",
        top1_fp32_argmin_pct=data.fp32_top1_pct,
        native_correct=int(np.count_nonzero(data.pred == data.winner)),
        aggregation="single problem", winner_rule="saved native/FP64 ind",
        data_version=data_version, candidate_pool=json.dumps(data.pool),
        checkpoint_sha256=SHA256, metric_source=str(metric_source),
        prediction_path=str(data.path),
    )


def macro_row(split, rows, saved_macro, data_version, metric_source):
    row = dict(rows[0])
    row.update(problem="ALL", n=sum(r["n"] for r in rows), sbs_name="per-problem TRAIN SBS",
               native_correct=sum(r["native_correct"] for r in rows),
               aggregation="equal weight across 18 problems; raw cost means descriptive only",
               candidate_pool="per-problem legal pools", prediction_path="see per-problem rows")
    mapping = {"top1_pct": "top1", "mean_cost": "mean_cost", "vs_sbs_pct": "vs_sbs_pct",
               "vs_oracle_pct": "vs_oracle_pct", "actual_regret_pct": "actual_regret_pct",
               "top2_pct": "top2", "top3_pct": "top3"}
    for key in (*METRIC_KEYS, "top1_fp32_argmin_pct"):
        value = float(np.mean([r[key] for r in rows]))
        if key in mapping:
            reference = float(saved_macro["macro_" + mapping[key]])
            reference *= 100 if key in ("top1_pct", "top2_pct", "top3_pct") else 1
            np.testing.assert_allclose(value, reference, rtol=0, atol=1e-10)
            value = reference
        row[key] = value
    row.update(split=split, data_version=data_version, metric_source=str(metric_source))
    return row


def check_comparison_alignment(current, nss):
    if current.n != nss.n or set(current.pool) != set(nss.pool):
        raise ValueError("NSS and R77B populations or candidate names differ")
    # R77 source_indices are split-local; NSS retains pre-resplit source IDs.
    key = "indices"
    left, right = getattr(current, key), getattr(nss, key)
    if len(np.unique(left)) != current.n or len(np.unique(right)) != nss.n or set(left) != set(right):
        raise ValueError(f"NSS and R77B instance identity mismatch ({key})")
    lookup = {int(value): i for i, value in enumerate(right)}
    sample = np.unique([0, current.n // 2, current.n - 1])
    other = np.array([lookup[int(left[i])] for i in sample])
    columns = [nss.pool.index(name) for name in current.pool]
    np.testing.assert_allclose(current.costs[sample], nss.costs[other][:, columns], rtol=0, atol=1e-12)
    left_winners = [current.pool[int(i)] for i in current.winner[sample]]
    right_winners = [nss.pool[int(i)] for i in nss.winner[other]]
    if left_winners != right_winners:
        raise ValueError("NSS and R77B sampled native winners differ")
    source_equal = (None if current.source_indices is None or nss.source_indices is None
                    else bool(np.array_equal(current.source_indices[sample], nss.source_indices[other])))
    return dict(instance_key=key, matching_population=current.n,
                checked_instance_ids=[int(left[i]) for i in sample],
                checked_rows=len(sample), costs="FP64, columns aligned by name, atol=1e-12",
                native_winners="sampled solver names match", sampled_source_indices_equal=source_equal,
                source_index_note="R77 uses split-local IDs; NSS may retain pre-resplit IDs. Match the same data version by indices.")


def comparison_row(problem, split, current, nss, nss_lock, version, checks):
    dataset = f"{problem} TEST" if split == "TEST" else split
    return dict(
        dataset=dataset, problem=problem, n=current.n, data_version=version,
        nss_model="R79_NSS_new_data_batch1024 (new-data retrained NSS)", r77_model="R77B",
        nss_top1_native_pct=nss.top1_pct, r77_top1_native_pct=current.top1_pct,
        nss_mean_cost=nss.mean_cost, r77_mean_cost=current.mean_cost,
        nss_original_fp32_top1_pct=nss.fp32_top1_pct,
        r77_fp32_top1_pct=current.fp32_top1_pct,
        winner_rule="saved native/FP64 ind; NSS saved pred; R77 stable global-ID argmax",
        nss_checkpoint_sha256=nss_lock[problem]["sha256"], r77_checkpoint_sha256=SHA256,
        candidate_pool=json.dumps(current.pool), alignment_key=checks["instance_key"],
        alignment_checked_rows=checks["checked_rows"], nss_prediction_path=str(nss.path),
        r77_prediction_path=str(current.path),
    )


def make_panels(problem, test, lib, canonical_pool):
    sources = []
    if problem in LIBS:
        lib_name, lib_n = LIBS[problem]
        sources.append((lib_name, lib, lib_n))
    sources.append(("TEST", test, test.n))
    panels = []
    for split, data, n in sources:
        for field in ("pred", "winner"):
            kind = "model_selection" if field == "pred" else "true_winner"
            if data is None:
                panels.append(dict(split=split, kind=kind, n=n, counts=None, top1=None))
                continue
            if set(data.pool) != set(canonical_pool):
                raise ValueError(f"Panel candidate mismatch: {problem} {split}")
            counts = np.bincount(getattr(data, field), minlength=len(data.pool))
            counts = counts[[data.pool.index(name) for name in canonical_pool]]
            if int(counts.sum()) != n:
                raise ValueError("Selection counts must sum to the panel population")
            panels.append(dict(split=split, kind=kind, n=n, counts=counts,
                               top1=data.top1_pct if field == "pred" else None))
    return panels


def distribution_rows(problem, panels, pool, version):
    rows = []
    for panel_order, panel in enumerate(panels, 1):
        for slot, solver in enumerate(pool):
            count = None if panel["counts"] is None else int(panel["counts"][slot])
            rows.append(dict(
                model="R77B", problem=problem, split=panel["split"], panel=panel["kind"],
                panel_order=panel_order, candidate_order=slot, candidate=solver, n=panel["n"],
                count=count, percentage=None if count is None else count * 100 / panel["n"],
                model_top1_native_pct=panel["top1"],
                status="missing" if count is None else "complete",
                winner_rule="saved native/FP64 ind", checkpoint_sha256=SHA256,
                data_version=version if panel["split"] == "TEST" else "original native LIB / historical NSS comparison",
            ))
    return rows


def setup_style():
    if not FONT.is_file():
        raise FileNotFoundError(FONT)
    font_manager.fontManager.addfont(str(FONT))
    plt.rcParams.update({
        "font.family": font_manager.FontProperties(fname=str(FONT)).get_name(),
        "font.size": 11, "pdf.fonttype": 42, "axes.unicode_minus": False,
        "text.color": "#242b30", "axes.labelcolor": "#242b30",
        "xtick.color": "#242b30", "ytick.color": "#242b30",
        "savefig.facecolor": "white", "figure.facecolor": "white",
    })


def plot_distribution(problem, pool, panels, data_version, lock):
    datasets = sorted(set(panel["split"] for panel in panels), key=lambda split: (split != "TEST", split))
    paired = len(datasets) == 2
    fig, axes = plt.subplots(1, len(datasets), figsize=(21 if paired else 12.5, 9))
    fig.subplots_adjust(left=0.105 if paired else 0.19, right=0.985 if paired else 0.96,
                        top=0.71, bottom=0.20, wspace=0.47)
    fig.text(0.035, 0.955, f"R77B | {problem} | \u65b9\u6cd5\u5206\u5e03",
             fontsize=21, fontweight="bold", color="#093f45", va="top")
    fig.text(0.035, 0.895,
             f"\u540c\u4e00\u9501\u5b9a\u6a21\u578b: epoch {lock['epoch']} / update {lock['selected_update']} / SHA256 {SHA256[:16]}",
             fontsize=11.5)
    fig.text(0.035, 0.855, f"\u65b0 TEST \u6570\u636e\u7248\u672c: {data_version}", fontsize=10.5)
    lib_text = (f"LIB: data/{LIBS[problem][0]} | native/FP64 ind"
                if paired else "\u6807\u7b7e: \u539f\u751f native/FP64 ind; \u4ec5\u8bfb\u9501\u5b9a\u9884\u6d4b")
    fig.text(0.035, 0.815, lib_text, fontsize=11)
    for ax, split in zip(np.atleast_1d(axes), datasets):
        series = {panel["kind"]: panel for panel in panels if panel["split"] == split}
        choice, winner = series["model_selection"], series["true_winner"]
        dataset = f"{problem} TEST" if split == "TEST" else split
        if choice["counts"] is None or winner["counts"] is None:
            ax.set_title(f"{dataset} | {choice['n']:,} \u4f8b | {MODEL_TOP1}=MISSING", fontsize=16)
            ax.text(0.5, 0.5, "MISSING: awaiting same-lock predictions", ha="center", va="center",
                    transform=ax.transAxes, wrap=True, color="#b23a38", fontsize=12)
            ax.set_axis_off()
            continue
        if choice["n"] != winner["n"]:
            raise ValueError(f"Model/winner populations differ: {problem} {split}")
        ax.set_title(f"{dataset}: {choice['n']:,} \u4f8b | {MODEL_TOP1}={choice['top1']:.2f}%",
                     fontsize=17, fontweight="bold", color="#093f45", pad=23)
        y = np.arange(len(pool))
        for panel, offset, color, label in (
            (choice, -0.16, "#197d85", MODEL_SELECTION),
            (winner, 0.16, "#c44f58", "\u771f\u5b9e\u6700\u4f18\u65b9\u6cd5"),
        ):
            percentages = panel["counts"] * 100 / panel["n"]
            ax.barh(y + offset, percentages, height=0.27, color=color, label=label, zorder=3)
            for row, (count, pct) in enumerate(zip(panel["counts"], percentages)):
                inside = pct >= 80
                ax.text(pct - 1.2 if inside else pct + 0.9, row + offset,
                        f"{pct:.2f}% ({int(count):,})", ha="right" if inside else "left",
                        va="center", fontsize=12, color="white" if inside else color)
        ax.set_yticks(y, labels=pool, fontsize=14)
        ax.set_ylim(len(pool) - 0.45, -0.65)
        ax.set_xlim(0, 100)
        ax.set_xticks(np.arange(0, 101, 20))
        ax.xaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
        ax.tick_params(axis="both", length=0, labelsize=13.5, pad=9)
        ax.set_axisbelow(True)
        ax.grid(axis="x", color="#dfe6e9", linewidth=0.9)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color("#aab2b7")
        ax.set_xlabel("\u5360\u5b9e\u4f8b\u6570\u7684\u6bd4\u4f8b (%)", fontsize=12, labelpad=10)
    for ax in np.atleast_1d(axes):
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc="upper right", bbox_to_anchor=(0.985, 0.977),
                       ncol=2, frameon=False, fontsize=14.5, handlelength=1.4, columnspacing=2.0)
            break
    fig.text(0.035, 0.105,
             "\u6807\u6ce8: % (\u6b21\u6570); \u5019\u9009\u6309 TEST \u540d\u79f0\u5bf9\u9f50\uff0c\u5305\u542b\u96f6\u6b21\u6570\u5019\u9009\u3002", fontsize=11)
    fig.text(0.035, 0.065,
             f"{MODEL_TOP1} = 100 * mean(pred == saved native winner); \u4e0d\u662f\u6700\u5927\u9009\u62e9\u9891\u7387\u3002", fontsize=11)
    fig.text(0.035, 0.025,
             "R77B: stable global solver-ID score ties. No training, inference, checkpoint reselection or Rxx mixing.",
             fontsize=10.5)
    return fig


def new_page(title, subtitle, page):
    fig = plt.figure(figsize=(16.54, 11.69))
    fig.text(0.045, 0.953, title, fontsize=22, va="top")
    fig.text(0.045, 0.902, subtitle, fontsize=11, va="top")
    fig.text(0.045, 0.026, f"R77B | same checkpoint {SHA256[:16]} | CPU-only saved-prediction statistics", fontsize=9)
    fig.text(0.955, 0.026, str(page), ha="right", fontsize=9)
    return fig


def add_table(fig, headers, rows, bbox, widths=None, font=10.5, bold_first=False):
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()
    table = ax.table(cellText=rows, colLabels=headers, cellLoc="right", colLoc="center",
                     bbox=bbox, colWidths=widths)
    table.auto_set_font_size(False)
    table.set_fontsize(font)
    for (r, c), cell in table.get_celld().items():
        cell.set_linewidth(0.45)
        cell.set_edgecolor("#d4dde1")
        cell.PAD = 0.065
        if r == 0:
            cell.set_facecolor("#e9eff1")
            cell.set_text_props(weight="bold")
        elif r % 2 == 0:
            cell.set_facecolor("#f5f7f8")
        if bold_first and r == 1:
            cell.set_facecolor("#e4f0ee")
            cell.set_text_props(weight="bold")
        if c == 0 and r > 0:
            cell.set_text_props(ha="left")
    return table


def metric_cells(row, label):
    if row is None:
        return [label] + ["MISSING"] * 8
    return [label, f"{row['top1_pct']:.4f}", f"{row['sbs_top1_pct']:.4f}",
            f"{row['mean_cost']:.6f}", f"{row['sbs_cost']:.6f}", f"{row['oracle_cost']:.6f}",
            f"{row['vs_sbs_pct']:+.4f}", f"{row['vs_oracle_pct']:+.4f}", f"{row['actual_regret_pct']:.4f}"]


def overview_page(lock, metrics, comparisons, version, complete):
    fig = new_page("R77B | \u6700\u65b0\u4fdd\u7559\u6700\u4f73\u5b8c\u6574\u6a21\u578b\u7edf\u8ba1",
                   "R78 \u5404\u5019\u9009\u672a\u901a\u8fc7 gate\uff0c\u7ee7\u7eed\u4fdd\u7559 R77B\u3002\u672c\u8f6e\u4ec5\u7edf\u8ba1\u9501\u5b9a\u9884\u6d4b\uff0c\u4e0d\u66f4\u6539\u6a21\u578b\u6216\u89c4\u5219\u3002", 1)
    fig.text(0.045, 0.852, f"Checkpoint: B_full_seed2/best.pt | epoch {lock['epoch']} | update {lock['selected_update']}", fontsize=12)
    fig.text(0.045, 0.818, f"SHA256: {SHA256}", fontsize=10.5)
    fig.text(0.045, 0.785, "\u9009\u578b: \u5b8c\u6574 VAL macro_vs_sbs_pct \u4e25\u683c\u6700\u5c0f\u503c\uff1bSBS \u5728\u5b8c\u6574 TRAIN \u4e0a\u56fa\u5b9a\u3002\u6240\u6709\u4efb\u52a1\u5171\u4eab\u540c\u4e00 checkpoint\u3002", fontsize=11.5)
    fig.text(0.045, 0.735, "VAL / TEST ALL | \u9010\u95ee\u9898\u7b49\u6743 (18 \u9898\uff0c\u5404 3,895 \u4f8b\uff0c\u5408\u8ba1 70,110)", fontsize=14)
    headers = ["Split", "Top1 (%)", "SBS Top1 (%)", "mean_cost", "SBS cost", "Oracle cost",
               "vs_SBS (%)", "vs_Oracle (%)", "actual regret (%)"]
    rows = [metric_cells(next(r for r in metrics if r["split"] == s and r["problem"] == "ALL"), s)
            for s in ("VAL", "TEST")]
    add_table(fig, headers, rows, [0.045, 0.598, 0.91, 0.11], font=10.7)
    fig.text(0.045, 0.565, "ALL \u6bd4\u7387 = \u5404\u9898\u6bd4\u7387\u7684\u5b8f\u5e73\u5747\uff1bALL \u6210\u672c\u4e3a\u5404\u9898\u5747\u503c\u7684\u63cf\u8ff0\u6027\u5e73\u5747\uff0c\u4e0d\u4ee5\u5176\u6574\u4f53\u6bd4\u503c\u66ff\u4ee3\u5b8f\u5e73\u5747\u3002", fontsize=11)
    fig.text(0.045, 0.507, "NSS / R77B | \u56db\u6570\u636e\u96c6\u540c\u53e3\u5f84\u6bd4\u8f83 (native/FP64 ind Top1)", fontsize=14)
    nss_headers = ["Dataset", "n", "NSS Top1 (%)", "R77B Top1 (%)", "NSS mean_cost", "R77B mean_cost",
                   "NSS \u539f\u7248\nFP32 Top1 (%)", "R77B FP32\nTop1 (%)"]
    nss_rows = []
    for row in comparisons:
        value = lambda key: "MISSING" if row[key] is None else f"{row[key]:.4f}"
        nss_rows.append([row["dataset"], str(row["n"]), value("nss_top1_native_pct"), value("r77_top1_native_pct"),
                         f"{row['nss_mean_cost']:.9f}", "MISSING" if row["r77_mean_cost"] is None else f"{row['r77_mean_cost']:.9f}",
                         value("nss_original_fp32_top1_pct"), value("r77_fp32_top1_pct")])
    add_table(fig, nss_headers, nss_rows, [0.045, 0.285, 0.91, 0.195],
              widths=[0.13, 0.05, 0.13, 0.13, 0.16, 0.16, 0.12, 0.12], font=10.5)
    fig.text(0.045, 0.249, "NSS = R79_NSS_new_data_batch1024\uff0c\u590d\u7528\u73b0\u6210\u9884\u6d4b\uff1b\u4e3b\u8868 Top1 \u4f7f\u7528 saved pred == native winner\u3002FP32 \u5217\u4ec5\u4f5c\u539f\u7248\u517c\u5bb9\u5bf9\u7167\u3002", fontsize=10.7)
    fig.text(0.045, 0.216, "TSP TEST / CVRP TEST: \u65b0\u6570\u636e\uff0c\u5404 n=3,895\uff1bTSPLIB / CVRPLIB: \u539f\u751f LIB\uff0c\u4e0d\u662f\u65b0 TEST\uff0c\u4e0d\u5305\u542b\u65e7\u7248 TEST \u6210\u7ee9\u3002", fontsize=10.7)
    fig.text(0.045, 0.183, f"\u65b0\u6570\u636e\u7248\u672c: {version}", fontsize=10)
    fig.text(0.045, 0.150, "TSP pool (8): BQ, DIFUSCO, DIFUSCO500, ELG, LEHD, OMNI, T2T, T2T500", fontsize=10.5)
    fig.text(0.045, 0.117, "CVRP pool (10): BQ, ELG, ICAM, LEHD, MVMOE, MoSES_CaDA, MoSES_RF, OMNI, RELD_CVRP, RouteFinder", fontsize=10.5)
    status = "\u5b8c\u6574\uff1aVAL / TEST 18 \u9898\u3001\u4e24 LIB \u53ca NSS \u56db\u9879\u5bf9\u7167\u5747\u5df2\u5230\u9f50\u3002" if complete else "\u90e8\u5206\u62a5\u544a\uff1a\u4ecd\u7f3a\u540c\u9501 LIB\uff0c\u7f3a\u9879\u5df2\u660e\u786e\u6807\u8bb0 MISSING\u3002"
    fig.text(0.045, 0.077, status, fontsize=11)
    return fig


def split_page(split, rows, version, page):
    fig = new_page(f"R77B | {split} | ALL + 18 \u4e2a\u95ee\u9898",
                   f"\u539f\u751f native/FP64 ind Top1\uff1bSBS \u4e3a\u5b8c\u6574 TRAIN \u56fa\u5b9a\u65b9\u6cd5\u3002\u6bcf\u9898 n=3,895\uff0cALL n=70,110\u3002", page)
    fig.text(0.045, 0.854, f"\u6570\u636e\u7248\u672c: {version}", fontsize=10.5)
    headers = ["Problem", "Top1 (%)", "SBS Top1 (%)", "mean_cost", "SBS cost", "Oracle cost",
               "vs_SBS (%)", "vs_Oracle (%)", "actual regret (%)", "TRAIN SBS"]
    cells = [metric_cells(r, r["problem"]) + ["per-problem" if r["problem"] == "ALL" else r["sbs_name"]]
             for r in rows]
    add_table(fig, headers, cells, [0.045, 0.214, 0.91, 0.605],
              widths=[0.10, 0.085, 0.10, 0.10, 0.10, 0.10, 0.095, 0.10, 0.11, 0.11], font=10.5, bold_first=True)
    fig.text(0.045, 0.171, "\u4e3b\u6307\u6807\u590d\u7528\u5df2\u4fdd\u5b58\u7684\u5b8c\u6574\u8bc4\u4f30\u7ed3\u679c\u3002SBS Top1 = \u56fa\u5b9a TRAIN SBS \u4e0e native winner \u7684\u4e00\u81f4\u7387\u3002", fontsize=11)
    fig.text(0.045, 0.134, "vs_SBS = 100*(mean_cost/SBS cost-1); vs_Oracle = 100*(mean_cost/Oracle cost-1). \u8d1f\u7684 vs_SBS \u8868\u793a\u4f18\u4e8e SBS\u3002", fontsize=11)
    fig.text(0.045, 0.097, "actual regret = mean_i[100*(selected_cost_i-min_j cost_ij)/min_j cost_ij]; Oracle \u4e3a\u9010\u4f8b\u5408\u6cd5\u5019\u9009\u6700\u4f4e\u6210\u672c\u3002", fontsize=11)
    fig.text(0.045, 0.060, "ALL \u4e0a\u8ff0\u5404\u9879\u5747\u4e3a\u9010\u9898\u7b49\u6743\uff1bmean_cost / SBS cost / Oracle cost \u7684\u8de8\u9898\u5e73\u5747\u4ec5\u4f5c\u63cf\u8ff0\u3002Top2/3 \u53ca\u5355\u65b9\u6cd5\u6bd4\u8f83\u8be6\u89c1 CSV\u3002", fontsize=10.5)
    return fig


def library_page(rows, lib_results, lib_protocols, lock):
    fig = new_page("R77B | LIB \u6210\u672c\u6307\u6807\u4e0e\u7edf\u8ba1\u534f\u8bae",
                   "\u540c\u4e00 best.pt\uff0c\u540c\u4e00\u63a8\u7406\u89c4\u5219\uff1b\u4e3b\u53e3\u5f84\u59cb\u7ec8\u4f7f\u7528 FULL TRAIN SBS\uff0c\u4e0d\u6309 LIB \u91cd\u65b0\u9009\u62e9 SBS\u3002", 4)
    headers = ["Dataset", "Top1 (%)", "SBS Top1 (%)", "mean_cost", "SBS cost", "Oracle cost",
               "vs_SBS (%)", "vs_Oracle (%)", "actual regret (%)"]
    cells = [metric_cells(next((r for r in rows if r["split"] == name), None), f"{name} (n={n})")
             for name, n in LIBS.values()]
    add_table(fig, headers, cells, [0.045, 0.735, 0.91, 0.11], font=10.5)
    fig.text(0.045, 0.692, "TRAIN SBS: TSPLIB = T2T; CVRPLIB = RELD_CVRP\u3002\u4e24\u8005\u5747\u6309\u539f\u751f LIB FP64 \u6210\u672c\u8bc4\u4f30\u56fa\u5b9a\u8eab\u4efd\u3002", fontsize=11.5)
    y = 0.649
    for name, n in LIBS.values():
        if name not in lib_results:
            text = f"{name}: MISSING same-lock predictions/results."
        else:
            native = lib_results[name]["native"]
            text = (f"{name} \u4e8b\u540e SBS (\u4ec5\u8865\u5145\uff0c\u4e0d\u4f5c\u4e3b\u53e3\u5f84): {native['sbs_name']} | "
                    f"cost={native['sbs_cost']:.9f} | vs_SBS={native['vs_sbs_pct']:+.6f}%")
        fig.text(0.045, y, text, fontsize=11)
        y -= 0.037
    fig.text(0.045, 0.533, "\u9501\u5b9a\u4e0e\u6807\u7b7e", fontsize=14)
    lines = [
        "R77B \u6240\u6709 VAL / TEST / LIB \u5747\u4e3a epoch 28 / update 34776\uff1b\u4e0d\u62fc\u63a5\u4e0d\u540c\u7248\u672c\u6216 checkpoint\u3002",
        "\u5148\u6309\u5b8c\u6574 VAL \u56fa\u5b9a checkpoint\uff1b\u672c\u62a5\u544a\u4e0d\u8bad\u7ec3\u3001\u4e0d\u63a8\u7406\u3001\u4e0d\u8c03\u6e29\u5ea6/\u9608\u503c\u3001\u4e0d\u66f4\u6539\u89c4\u5219\u3002",
        "R77B \u9884\u6d4b = argmax(scores), \u5e76\u5217\u5206\u6570\u6309 global solver ID \u7a33\u5b9a\u6253\u7834\uff1bLIB \u540c\u65f6\u6838\u5bf9 saved pred\u3002",
        "NSS \u9884\u6d4b\u4ec5\u7528 saved pred\uff0c\u4e0d\u4ece scores \u91cd\u5efa\u51b3\u7b56\u3002Top1 \u4e3b\u8868\u4e0e\u5206\u5e03\u56fe\u7edf\u4e00\u4f7f\u7528 native winner\u3002",
        "FP32 \u517c\u5bb9\u5217\u4e3a cost.astype(float32).argmin\uff0c\u4e0d\u66ff\u6362\u4e3b\u8868 native \u6807\u7b7e\u3002\u6240\u6709 mean_cost \u4f7f\u7528\u4fdd\u5b58\u7684 FP64 \u6210\u672c\u3002",
        "NSS \u6bd4\u8f83\u4ec5\u6309\u540d\u79f0\u3001\u5b9e\u4f8b\u7d22\u5f15\u5bf9\u9f50\uff0c\u6bcf\u6570\u636e\u96c6\u62bd 3 \u884c\u6838\u5bf9\u6210\u672c\u548c winner\uff1b\u4e0d\u8bfb\u53d6 TRAIN \u6216\u5de8\u5927\u539f\u59cb\u6210\u672c\u6587\u4ef6\u3002",
        "\u591a\u8f6e\u7814\u7a76\u7684 TEST \u7ed3\u679c\u5df2\u90e8\u5206\u53c2\u4e0e\u540e\u7eed\u7814\u53d1\uff0c\u4e0d\u5c06\u5f53\u524d TEST \u79f0\u4e3a\u5b8c\u5168\u76f2\u6d4b\uff1b\u672c\u8f6e\u4e0d\u4f7f\u7528\u5b83\u91cd\u65b0\u9009\u578b\u3002",
    ]
    for i, line in enumerate(lines):
        fig.text(0.045, 0.490 - i * 0.036, line, fontsize=11)
    fig.text(0.045, 0.207, "\u8f93\u5165\u4e0e\u4ea4\u4ed8", fontsize=14)
    fig.text(0.045, 0.168, "TEST: B_full_seed2/evaluation/test/selected_best/rank000.npz | VAL: selected_train_val_predictions/B/val.npz", fontsize=10.5)
    fig.text(0.045, 0.134, "LIB: B_full_seed2/lib_evaluation/predictions/{TSPLIB,CVRPLIB}.npz | NSS: R79_NSS_new_data_batch1024 \u56db\u4efd\u73b0\u6210\u9884\u6d4b", fontsize=10.5)
    fig.text(0.045, 0.100, "\u672c PDF \u540e\u9644 18 \u9875\u5206\u5e03\uff1b\u53e6\u4ea4\u4ed8 selection_distributions.pdf\u3001CSV \u53ca protocol.json\uff0c\u5747\u53ef\u79bb\u7ebf\u67e5\u770b\u3002", fontsize=11)
    fig.text(0.045, 0.066, f"\u539f\u59cb\u6a21\u578b\u9501\u5b9a\u65f6\u95f4: {lock['locked_utc']} | \u5404\u6570\u636e\u7248\u672c\u3001\u6e90\u8def\u5f84\u4e0e\u9501\u8be6\u89c1 protocol.json\u3002", fontsize=10.5)
    return fig


def method_rows(problem, split, saved, data, version):
    rows = []
    mean_costs = saved.get("method_mean_cost", {})
    top1 = saved.get("method_top1", {})
    for slot, solver in enumerate(data.pool):
        cost = float(mean_costs.get(solver, data.costs[:, slot].mean(dtype=np.float64)))
        accuracy = float(top1.get(solver, np.mean(data.winner == slot)) * 100)
        rows.append(dict(model="R77B", split=split, problem=problem, candidate=solver,
                         method_top1_native_pct=accuracy, method_mean_cost=cost,
                         selector_top1_native_pct=saved["top1"] * 100,
                         selector_mean_cost=saved["mean_cost"],
                         selector_strictly_better_top1=bool(saved["top1"] * 100 > accuracy),
                         selector_strictly_better_cost=bool(saved["mean_cost"] < cost),
                         is_train_sbs=solver == saved["sbs_name"], data_version=version))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--require-lib", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    run = root / "B_full_seed2"
    output = (args.output or run / "full_model_statistics").resolve()
    nss_root = root.parent / "R79_NSS_new_data_batch1024"
    lock = read_json(run / "final_test_lock.json")
    manifest = read_json(root / "selected_train_val_predictions/B/manifest.json")
    test_results = read_json(run / "test_results.json")
    val_results = read_json(root / "selected_train_val_predictions/B/val_metrics.json")
    for name, identity in (("lock", lock), ("manifest", manifest), ("TEST results", test_results)):
        if identity["checkpoint_sha256"] != SHA256 or identity["selected_update"] != 34776:
            raise ValueError(f"Wrong R77B checkpoint in {name}")
    if lock["epoch"] != 28 or lock["per_problem_assembly"] or not lock["selected_without_test"]:
        raise ValueError("R77B lock does not match the requested protocol")
    test_results = test_results["result"]
    problems = list(test_results["per_problem"])
    if len(problems) != 18 or set(problems) != set(val_results["per_problem"]):
        raise ValueError("Expected the same 18 problems in VAL and TEST")
    data_root = Path(manifest["data_root"])
    version = data_root.name
    test_path = run / "evaluation/test/selected_best/rank000.npz"
    val_path = root / "selected_train_val_predictions/B/val.npz"
    metric_sources = {"TEST": run / "test_results.json",
                      "VAL": root / "selected_train_val_predictions/B/val_metrics.json"}
    predictions, metrics, methods = {}, [], []
    for split, path, result in (("VAL", val_path, val_results), ("TEST", test_path, test_results)):
        per_problem_rows = []
        with np.load(path, allow_pickle=False) as archive:
            for problem in problems:
                data = load_predictions(archive, path, problem + "__")
                if data.n != 3895:
                    raise ValueError(f"Unexpected {split} population for {problem}")
                predictions[split, problem] = data
                saved = result["per_problem"][problem]
                row = metric_row(problem, split, data, saved, manifest["common_train_sbs"][problem],
                                 version, metric_sources[split])
                per_problem_rows.append(row)
                methods.extend(method_rows(problem, split, saved, data, version))
        metrics.append(macro_row(split, per_problem_rows, result["macro"], version, metric_sources[split]))
        metrics.extend(per_problem_rows)
    lib_results, lib_protocols, missing = {}, {}, []
    for problem, (split, n) in LIBS.items():
        lib_root = run / "lib_evaluation"
        path = lib_root / f"predictions/{split}.npz"
        result_path = lib_root / f"{split}_results.json"
        protocol_path = lib_root / f"{split}_protocol.json"
        if not all(p.is_file() for p in (path, result_path, protocol_path)):
            missing.append(split)
            continue
        result, protocol = read_json(result_path), read_json(protocol_path)
        if protocol["checkpoint_sha256"] != SHA256 or protocol["checkpoint_update"] != 34776:
            raise ValueError(f"{split} belongs to a different checkpoint")
        with np.load(path, allow_pickle=False) as archive:
            if str(archive["checkpoint_sha256"].item()) != SHA256:
                raise ValueError(f"Wrong checkpoint in {path}")
            data = load_predictions(archive, path)
        if data.n != n:
            raise ValueError(f"Wrong {split} population")
        predictions[split, problem] = data
        lib_results[split], lib_protocols[split] = result, protocol
        saved = result["train_sbs_reference"]
        metrics.append(metric_row(problem, split, data, saved, manifest["common_train_sbs"][problem],
                                  protocol["data_version"], result_path))
        methods.extend(method_rows(problem, split, saved, data, protocol["data_version"]))
    if args.require_lib and missing:
        raise FileNotFoundError(f"Missing same-lock LIB predictions/results: {missing}")
    nss_protocol = read_json(nss_root / "protocol.json")
    nss_lock = read_json(nss_root / "final_test_lock.json")
    if nss_protocol["data_root"] != str(data_root) or nss_lock["data_root"] != str(data_root):
        raise ValueError("NSS does not use the same new-data version")
    comparisons, alignment = [], {}
    for problem, split in (("TSP", "TEST"), ("CVRP", "TEST"), ("TSP", "TSPLIB"), ("CVRP", "CVRPLIB")):
        path = nss_root / problem / f"{'test' if split == 'TEST' else split}_predictions.npz"
        with np.load(path, allow_pickle=False) as archive:
            nss = load_predictions(archive, path, r77=False)
        current = predictions.get((split, problem))
        if current is None:
            comparisons.append(dict(dataset=split, problem=problem, n=nss.n,
                data_version="original native LIB / historical NSS comparison",
                nss_model="R79_NSS_new_data_batch1024 (new-data retrained NSS)", r77_model="R77B",
                nss_top1_native_pct=nss.top1_pct, r77_top1_native_pct=None,
                nss_mean_cost=nss.mean_cost, r77_mean_cost=None,
                nss_original_fp32_top1_pct=nss.fp32_top1_pct, r77_fp32_top1_pct=None,
                winner_rule="saved native/FP64 ind", nss_checkpoint_sha256=nss_lock["checkpoints"][problem]["sha256"],
                r77_checkpoint_sha256=SHA256, candidate_pool=json.dumps(nss.pool), alignment_key=None,
                alignment_checked_rows=0, nss_prediction_path=str(path), r77_prediction_path=None))
            continue
        check = check_comparison_alignment(current, nss)
        alignment[f"{problem}/{split}"] = check
        comparisons.append(comparison_row(problem, split, current, nss, nss_lock["checkpoints"],
                           version if split == "TEST" else lib_protocols[split]["data_version"], check))
    panel_sets, distributions = {}, []
    for problem in problems:
        test = predictions["TEST", problem]
        lib = predictions.get((LIBS[problem][0], problem)) if problem in LIBS else None
        panel_sets[problem] = make_panels(problem, test, lib, test.pool)
        distributions.extend(distribution_rows(problem, panel_sets[problem], test.pool, version))
    output.mkdir(parents=True, exist_ok=True)
    figures = output / "figures"
    figures.mkdir(exist_ok=True)
    write_csv(output / "metrics_by_problem.csv", metrics)
    write_csv(output / "nss_comparison.csv", comparisons)
    write_csv(output / "selection_distribution.csv", distributions)
    write_csv(output / "single_method_comparison.csv", methods)
    setup_style()
    metadata = {"Title": "R77B locked full-model statistics", "Author": "R77B CPU-only reporting",
                "Subject": "Native winner metrics, fixed TRAIN SBS, no checkpoint reselection"}
    with PdfPages(output / "MODEL_STATISTICS.pdf", metadata=metadata) as report:
        summary_pages = [overview_page(lock, metrics, comparisons, version, not missing),
                         split_page("VAL", [r for r in metrics if r["split"] == "VAL"], version, 2),
                         split_page("TEST", [r for r in metrics if r["split"] == "TEST"], version, 3),
                         library_page(metrics, lib_results, lib_protocols, lock)]
        for page in summary_pages:
            report.savefig(page)
            plt.close(page)
        with PdfPages(output / "selection_distributions.pdf", metadata=metadata) as distribution_pdf:
            for problem in problems:
                fig = plot_distribution(problem, predictions["TEST", problem].pool,
                                        panel_sets[problem], version, lock)
                fig.savefig(figures / f"{problem}_selection_distribution.pdf")
                fig.savefig(figures / f"{problem}_selection_distribution.png", dpi=180)
                distribution_pdf.savefig(fig)
                report.savefig(fig)
                plt.close(fig)
                print(f"[R77B report] {problem}: saved distribution PDF/PNG", flush=True)
    protocol = dict(
        complete=not missing, missing_same_lock_lib=missing,
        created_utc=datetime.now(timezone.utc).isoformat(), model="R77B",
        retained_best_reason="R78 candidates did not pass gate; retain R77B (user-confirmed)",
        execution="CPU-only NumPy/Matplotlib saved-prediction statistics; no model imports",
        training=False, inference=False, checkpoint_reselection=False, new_trial=False, git_commit=False,
        checkpoint_lock=lock, checkpoint_hash_policy="Reuse existing lock and metadata; never open weights",
        prediction_rule="order=np.argsort(pool_ids,kind='stable'); pred=order[np.argmax(scores[:,order],axis=1)]",
        nss_prediction_rule="Use saved pred directly; never recompute NSS argmax",
        winner_rule="Main metrics and plots: saved native/FP64 raw_label ind",
        fp32_compatibility_rule="Secondary only: pred == costs.astype(np.float32).argmin(1), archive column order",
        top1_rule="100 * mean(pred == saved winner); not largest method-selection frequency",
        true_winner_panel_accuracy="not applicable; no model Top1 attached to true-winner rows",
        sbs_rule="Per-problem minimum mean FP64 cost on FULL TRAIN (70114/problem); identity fixed for VAL/TEST/LIB",
        sbs_metadata=manifest["common_train_sbs"],
        metric_formulas=dict(
            top1_pct="100 * mean(pred == native winner)", sbs_top1_pct="100 * mean(TRAIN SBS slot == native winner)",
            mean_cost="mean_i costs_i[pred_i], saved FP64 costs",
            oracle_cost="mean_i min_j legal_cost_ij",
            vs_sbs_pct="100 * (mean_cost / evaluated_split_cost_of_fixed_TRAIN_SBS - 1)",
            vs_oracle_pct="100 * (mean_cost / oracle_cost - 1)",
            actual_regret_pct="mean_i [100 * (selected_cost_i - min_j cost_ij) / min_j cost_ij]",
            ALL="Equal-weight mean of each per-problem metric; never ratio of cross-problem raw-cost means"),
        test_status="TEST has partly informed subsequent multi-round R&D; not a fully untouched blind set. No reselection here.",
        data_root=str(data_root), data_version=version,
        legal_candidate_order={p: predictions["TEST", p].pool for p in problems},
        distribution_layout="Grouped horizontal bars: model selection and native true winner share one axis per dataset",
        panel_order_tsp_cvrp=["TEST model selection + native true winner", "LIB model selection + native true winner"],
        panel_order_others=["TEST model selection + native true winner"],
        distribution_csv_series_order="Unchanged: LIB pred/winner then TEST pred/winner; grouped figure order is TEST then LIB",
        population=dict(val_per_problem=3895, test_per_problem=3895, problems=18, val_total=70110,
                        test_total=70110, TSPLIB=49, CVRPLIB=100),
        inputs=dict(test_predictions=str(test_path), val_predictions=str(val_path),
                    test_metrics=str(metric_sources["TEST"]), val_metrics=str(metric_sources["VAL"]),
                    train_sbs_manifest=str(root / "selected_train_val_predictions/B/manifest.json"),
                    lib_protocols=lib_protocols, nss_protocol_path=str(nss_root / "protocol.json"),
                    nss_lock=nss_lock),
        lightweight_alignment_checks=alignment,
        primary_metric_policy="Reuse saved VAL/TEST and LIB train_sbs_reference metrics; derive only supplemental prediction statistics",
        score_tie_rows={f"{s}/{p}": d.score_ties for (s, p), d in predictions.items()},
        distributions=dict(pages=18, rows=len(distributions), zero_count_candidates_included=True),
        primary_pdf_pages=22, font_path=str(FONT),
        artifacts=["MODEL_STATISTICS.pdf", "selection_distributions.pdf", "metrics_by_problem.csv",
                   "nss_comparison.csv", "selection_distribution.csv", "single_method_comparison.csv", "RESULTS.md"],
    )
    write_json(output / "protocol.json", protocol)
    all_rows = {r["split"]: r for r in metrics if r["problem"] == "ALL"}
    text = ["# R77B locked full-model statistics", "",
            "[Main report, 22 pages](MODEL_STATISTICS.pdf) | [Selection distributions, 18 pages](selection_distributions.pdf)", "",
            f"Status: {'complete' if not missing else 'PARTIAL; missing ' + ', '.join(missing)}. Retain R77B; R78 candidates did not pass gate.",
            f"One checkpoint for every task/dataset: epoch 28, update 34776; SHA256 `{SHA256}`.", "",
            "| Split | ALL native Top1 (%) | ALL actual regret (%) |", "|---|---:|---:|"]
    text.extend(f"| {s} | {all_rows[s]['top1_pct']:.7f} | {all_rows[s]['actual_regret_pct']:.7f} |" for s in ("VAL", "TEST"))
    text.extend(["", "ALL is equal-weight across 18 problems. SBS identities were fixed on FULL TRAIN; LIB posthoc SBS is supplementary only.",
                 "Each dataset's model selections and native true winners share one grouped horizontal-bar plot; TSP/CVRP show TEST then LIB.",
                 "Primary Top1 uses saved native/FP64 winners. NSS original FP32 Top1 is a separate compatibility column.",
                 f"New VAL/TEST data version: `{version}`; LIB uses original native TSPLIB/CVRPLIB, not old TEST.",
                 "TEST has partly informed subsequent research; this is not a claim of a completely untouched blind set.",
                 "This report only reads existing predictions/metrics. No training, inference, checkpoint reselection, new trials, or commits.", "",
                 "CSV: `metrics_by_problem.csv` (including Top2/3), `nss_comparison.csv`, `selection_distribution.csv`, `single_method_comparison.csv`.",
                 "Exact formulas, source paths, locks, candidate order and three-row-per-dataset alignment checks: `protocol.json`.", ""])
    (output / "RESULTS.md").write_text("\n".join(text), encoding="utf-8")
    print(json.dumps(dict(output=str(output), complete=not missing,
                          val_top1_pct=all_rows["VAL"]["top1_pct"], test_top1_pct=all_rows["TEST"]["top1_pct"],
                          val_regret_pct=all_rows["VAL"]["actual_regret_pct"], test_regret_pct=all_rows["TEST"]["actual_regret_pct"],
                          distribution_pages=18), indent=2), flush=True)


if __name__ == "__main__":
    main()
