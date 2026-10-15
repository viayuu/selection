"""CPU-only diagnostics of saved R76 choices; no model loading or inference."""

import argparse
import csv
from contextlib import nullcontext
import hashlib
import json
import os
from pathlib import Path
import zipfile

os.environ["CUDA_VISIBLE_DEVICES"] = ""
for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "2"
os.environ.setdefault("MPLCONFIGDIR", "/tmp/r76_probability_diagnostics_mpl")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import PercentFormatter
import numpy as np

from code.unified_selector.registry import PROBLEMS
from code.V4.winner_distribution_analysis import render_html
from code.V4.winner_distribution_figures import style


RUN_ROOT = Path(__file__).resolve().parent / "runs/R76_R45A_single_reld_mt"
RESEARCH_AGENT = "01a13b22-453f-71f0-87e3-97680c2cb763"
QUANTILES = (("q05", .05), ("q25", .25), ("median", .5), ("q75", .75), ("q95", .95))


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8")


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_rows(path, rows, fields=None):
    fields = fields or list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def masked_softmax_fp32(logits, mask):
    scores = np.asarray(logits, dtype=np.float32)
    mask = np.asarray(mask, dtype=bool)
    if (scores.ndim != 2 or scores.shape != mask.shape or not mask.any(1).all()
            or not np.isfinite(scores[mask]).all()):
        raise ValueError("Softmax requires finite valid scores and at least one candidate per row")
    masked = np.where(mask, scores, np.float32(-np.inf))
    shifted = masked - masked.max(1, keepdims=True)
    exponentials = np.exp(shifted)
    denominator = exponentials.sum(1, keepdims=True, dtype=np.float32)
    probabilities = exponentials / denominator
    log_probabilities = shifted - np.log(denominator)
    return probabilities, log_probabilities


def score_ranks(scores, pool_ids, mask):
    ids = np.asarray(pool_ids)
    if len(np.unique(ids)) != len(ids):
        raise ValueError("Duplicate solver identities")
    canonical = np.argsort(ids, kind="stable")
    masked = np.where(mask, scores, -np.inf)
    order = canonical[np.argsort(-masked[:, canonical], axis=1, kind="stable")]
    ranks = np.empty_like(order)
    np.put_along_axis(ranks, order, np.arange(1, scores.shape[1] + 1)[None, :], axis=1)
    return order[:, 0], ranks


def prepare_prediction(value):
    value = dict(value)
    scores, costs = value["logits"], value["costs"]
    n, k = scores.shape
    pool, ids = value["pool"], value["pool_ids"]
    winner, indices = value["winner"], value["indices"]
    if (costs.shape != (n, k) or len(pool) != k or len(ids) != k
            or len(set(pool.tolist())) != k or winner.shape != (n,)
            or value["nodes"].shape != (n,) or not np.isfinite(costs).all()
            or costs.dtype != np.float64 or scores.dtype != np.float32
            or not np.issubdtype(winner.dtype, np.integer)
            or not np.issubdtype(indices.dtype, np.integer)
            or not np.array_equal(np.sort(indices), np.arange(n))
            or (winner < 0).any() or (winner >= k).any()
            or np.isnan(scores).any() or np.isposinf(scores).any()):
        raise ValueError(f"{value['problem']}/{value['split']}: invalid or incomplete saved predictions")
    sort = np.argsort(indices, kind="stable")
    for key in ("logits", "costs", "winner", "indices", "nodes"):
        value[key] = value[key][sort]
    scores, costs, winner = value["logits"], value["costs"], value["winner"]
    # NPZ columns are the legal pool, not the global vocabulary; -inf is an explicit mask.
    mask = ~np.isneginf(scores)
    rows = np.arange(n)
    if not mask[rows, winner].all():
        raise ValueError("A native winner is masked out")
    probabilities, log_probabilities = masked_softmax_fp32(scores, mask)
    selected, ranks = score_ranks(scores, ids, mask)
    selected_cost, winner_cost = costs[rows, selected], costs[rows, winner]
    oracle_cost = np.where(mask, costs, np.inf).min(1)
    costloss = selected_cost - winner_cost
    relative = np.divide(costloss * 100, winner_cost, out=np.full(n, np.nan),
                         where=winner_cost != 0)
    one_hot = np.zeros_like(probabilities)
    one_hot[rows, winner] = 1
    value.update(n=n, k=k, mask=mask, probabilities=probabilities, ranks=ranks,
                 selected=selected, correct=selected == winner,
                 winner_probability=probabilities[rows, winner], winner_rank=ranks[rows, winner],
                 winner_logit=scores[rows, winner], confidence=probabilities[rows, selected],
                 selected_cost=selected_cost, winner_cost=winner_cost, oracle_cost=oracle_cost,
                 costloss=costloss, relative_costloss_pct=relative,
                 oracle_costloss=selected_cost - oracle_cost,
                 logit_gap=scores[rows, selected] - scores[rows, winner],
                 nll=-log_probabilities[rows, winner],
                 brier=np.square(probabilities.astype(np.float64) - one_hot).sum(1),
                 entropy=-(np.where(mask, log_probabilities, 0) * probabilities).sum(1))
    np.testing.assert_allclose(probabilities.sum(1, dtype=np.float64), 1, rtol=0,
                               atol=5 * np.finfo(np.float32).eps)
    return value


def check_metrics(value, expected):
    n, k = value["n"], value["k"]
    if n != expected["n"] or value["pool"].tolist() != expected["pool"]:
        raise ValueError("Saved pool or instance count differs from original metrics")
    counts = np.bincount(value["selected"], minlength=k)
    np.testing.assert_allclose(counts / n, expected["pick_dist"], rtol=0, atol=1e-12)
    for top in (1, 2, 3):
        np.testing.assert_allclose((value["winner_rank"] <= top).mean(), expected[f"top{top}"],
                                   rtol=0, atol=1e-12)
    np.testing.assert_allclose(value["selected_cost"].mean(), expected["mean_cost"],
                               rtol=0, atol=1e-9)
    if "method_top1" in expected:
        for slot, name in enumerate(value["pool"]):
            np.testing.assert_allclose((value["winner"] == slot).mean(),
                                       expected["method_top1"][name], rtol=0, atol=1e-12)


def load_shards(directory, split, expected):
    paths = sorted(directory.glob("rank*.npz"))
    if not paths:
        raise FileNotFoundError(directory)
    chunks = {}
    for path in paths:
        with np.load(path, allow_pickle=False) as saved:
            for key in saved.files:
                if key.endswith("__logits"):
                    problem = key[:-len("__logits")]
                    part = {name: saved[problem + "__" + name].copy() for name in
                            ("logits", "costs", "winner", "indices", "pool", "pool_ids", "nodes")}
                    chunks.setdefault(problem, []).append(part)
    if set(chunks) != set(expected):
        raise ValueError("Prediction shards do not cover the expected problems exactly")
    result = {}
    for problem in expected:
        parts = chunks[problem]
        for part in parts:
            for name in ("pool", "pool_ids"):
                if not np.array_equal(part[name], parts[0][name]):
                    raise ValueError("Inconsistent pool identities across shards")
        value = {name: np.concatenate([part[name] for part in parts]) for name in
                 ("logits", "costs", "winner", "indices", "nodes")}
        value.update(pool=parts[0]["pool"], pool_ids=parts[0]["pool_ids"], problem=problem, split=split)
        result[problem] = prepare_prediction(value)
        check_metrics(result[problem], expected[problem])
    return result, paths


def statistics(prefix, values):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    keys = ("mean", "min", "max") + tuple(key for key, _ in QUANTILES)
    result = {prefix + "_" + key: None for key in keys}
    if len(values):
        result.update({prefix + "_mean": float(values.mean()), prefix + "_min": float(values.min()),
                       prefix + "_max": float(values.max())})
        result.update({prefix + "_" + key: float(np.quantile(values, q)) for key, q in QUANTILES})
    return result


def method_diagnostics(value, near_zero_pct):
    n, pool, winner, selected = value["n"], value["pool"], value["winner"], value["selected"]
    selected_counts = np.bincount(selected, minlength=len(pool))
    total_loss = float(value["costloss"].sum(dtype=np.float64))
    total_relative = float(np.nansum(value["relative_costloss_pct"], dtype=np.float64))
    summaries, candidate_rows, flows = [], [], []
    for slot, name in enumerate(pool):
        won = winner == slot
        missed = won & ~value["correct"]
        count, missed_count, selected_count = int(won.sum()), int(missed.sum()), int(selected_counts[slot])
        loss = float(value["costloss"][missed].sum(dtype=np.float64))
        relative = float(np.nansum(value["relative_costloss_pct"][missed], dtype=np.float64))
        row = dict(problem=value["problem"], split=value["split"], solver=str(name),
                   solver_id=int(value["pool_ids"][slot]), instances=n,
                   true_winner_count=count, true_winner_pct=100 * count / n,
                   selected_count=selected_count, selected_pct=100 * selected_count / n,
                   correct_selected_count=count - missed_count, missed_winner_count=missed_count,
                   missed_winner_pct_of_own_wins=100 * missed_count / count if count else None,
                   zero_selected=selected_count == 0, zero_selected_with_winner=selected_count == 0 and count > 0,
                   near_zero_selected_with_winner=0 < 100 * selected_count / n <= near_zero_pct and count > 0,
                   costloss_sum=loss, mean_costloss_contribution=loss / n,
                   problem_costloss_share_pct=100 * loss / total_loss if total_loss else 0.,
                   relative_costloss_sum_pct=relative, relative_costloss_contribution_pp=relative / n,
                   problem_relative_costloss_share_pct=100 * relative / total_relative if total_relative else 0.,
                   missed_cost_tie_count=int((value["costloss"][missed] == 0).sum()),
                   missed_negative_costloss_count=int((value["costloss"][missed] < 0).sum()))
        for prefix, field in (("winner_probability", "winner_probability"), ("winner_rank", "winner_rank"),
                              ("winner_logit", "winner_logit"), ("winner_ce", "nll")):
            row.update(statistics(prefix, value[field][won]))
            row.update(statistics("missed_" + prefix, value[field][missed]))
        for prefix, field in (("missed_costloss", "costloss"),
                              ("missed_relative_costloss_pct", "relative_costloss_pct"),
                              ("missed_logit_gap", "logit_gap")):
            row.update(statistics(prefix, value[field][missed]))
        missed_selections = np.bincount(selected[missed], minlength=len(pool))
        row["missed_selected_method_counts"] = json.dumps(
            {str(method): int(c) for method, c in zip(pool, missed_selections) if c}, ensure_ascii=False)
        summaries.append(row)
        for candidate, method in enumerate(pool):
            flow_mask = missed & (selected == candidate)
            flows.append(dict(problem=value["problem"], split=value["split"], true_winner=str(name),
                              selected_solver=str(method), missed_winner_count=missed_count,
                              selected_count=int(flow_mask.sum()),
                              selected_pct_of_missed=100 * int(flow_mask.sum()) / missed_count if missed_count else None,
                              costloss_sum=float(value["costloss"][flow_mask].sum(dtype=np.float64))))
            candidate_row = dict(problem=value["problem"], split=value["split"], true_winner=str(name),
                                 candidate=str(method), candidate_id=int(value["pool_ids"][candidate]),
                                 missed_winner_count=missed_count, selected_count=int(missed_selections[candidate]))
            for prefix, field in (("probability", "probabilities"), ("logit", "logits"), ("rank", "ranks")):
                candidate_row.update(statistics(prefix, value[field][missed, candidate]))
            candidate_rows.append(candidate_row)
    return summaries, candidate_rows, flows


def calibration_bins(confidence, correct, bins=20):
    confidence = np.asarray(confidence, dtype=np.float64)
    correct = np.asarray(correct, dtype=np.float64)
    if (confidence.shape != correct.shape or not len(confidence)
            or not np.isfinite(confidence).all() or (confidence < 0).any() or (confidence > 1).any()):
        raise ValueError("Invalid calibration inputs")
    membership = np.minimum((confidence * bins).astype(np.int64), bins - 1)
    rows, ece = [], 0.
    for slot in range(bins):
        mask = membership == slot
        count = int(mask.sum())
        mean = float(confidence[mask].mean()) if count else None
        accuracy = float(correct[mask].mean()) if count else None
        gap = abs(mean - accuracy) if count else None
        contribution = count / len(confidence) * gap if count else 0.
        ece += contribution
        rows.append(dict(bin=slot, lower=slot / bins, upper=(slot + 1) / bins, count=count,
                         fraction=count / len(confidence), mean_probability=mean, observed_frequency=accuracy,
                         absolute_gap=gap, ece_contribution=contribution))
    return rows, ece


def calibration_reports(predictions, split):
    summaries, top_bins, classes, class_bins = [], [], [], []
    for problem, value in predictions.items():
        bins, ece = calibration_bins(value["confidence"], value["correct"])
        summaries.append(dict(problem=problem, split=split, instances=value["n"],
                              top1_pct=100 * float(value["correct"].mean()), ce=float(value["nll"].mean(dtype=np.float64)),
                              top2_pct=100 * float((value["winner_rank"] <= 2).mean()),
                              top3_pct=100 * float((value["winner_rank"] <= 3).mean()),
                              mean_cost=float(value["selected_cost"].mean()),
                              brier=float(value["brier"].mean()), ece20=ece,
                              mean_confidence=float(value["confidence"].mean(dtype=np.float64)),
                              mean_entropy=float(value["entropy"].mean(dtype=np.float64))))
        top_bins.extend(dict(problem=problem, split=split, **row) for row in bins)
        for slot, method in enumerate(value["pool"]):
            probability, truth = value["probabilities"][:, slot], value["winner"] == slot
            bins, class_ece = calibration_bins(probability, truth)
            classes.append(dict(problem=problem, split=split, solver=str(method), instances=value["n"],
                                true_winner_count=int(truth.sum()), selected_count=int((value["selected"] == slot).sum()),
                                mean_probability=float(probability.mean(dtype=np.float64)),
                                true_winner_fraction=float(truth.mean()),
                                mean_probability_minus_winner_frequency=float(probability.mean(dtype=np.float64) - truth.mean()),
                                binary_brier=float(np.square(probability.astype(np.float64) - truth).mean()),
                                classwise_ece20=class_ece))
            class_bins.extend(dict(problem=problem, split=split, solver=str(method), **row) for row in bins)
    pooled_confidence = np.concatenate([v["confidence"] for v in predictions.values()])
    pooled_correct = np.concatenate([v["correct"] for v in predictions.values()])
    bins, ece = calibration_bins(pooled_confidence, pooled_correct)
    top_bins.extend(dict(problem="ALL18_pooled", split=split, **row) for row in bins)
    aggregate = dict(split=split, instances=len(pooled_correct),
                     top1_pct=100 * float(pooled_correct.mean()),
                     macro_ce=float(np.mean([r["ce"] for r in summaries])),
                     macro_brier=float(np.mean([r["brier"] for r in summaries])),
                     macro_ece20=float(np.mean([r["ece20"] for r in summaries])),
                     pooled_ece20=ece, mean_confidence=float(pooled_confidence.mean(dtype=np.float64)))
    return summaries, top_bins, classes, class_bins, aggregate


SAMPLE_FIELDS = ("problem", "split", "index", "nodes", "true_winner", "selected_solver",
                 "winner_probability", "selected_probability", "winner_logit", "selected_logit", "winner_rank",
                 "logit_gap", "winner_cost", "selected_cost", "oracle_cost", "costloss", "relative_costloss_pct",
                 "oracle_costloss", "zero_selected_winner", "near_zero_selected_winner")
CANDIDATE_FIELDS = ("problem", "split", "index", "true_winner", "selected_solver", "candidate", "candidate_id",
                    "valid", "probability", "logit", "rank", "cost", "is_true_winner", "is_selected")


def write_detailed_problem(output, value, summaries, sample_writer, candidate_writer):
    pool, winner, selected = value["pool"], value["winner"], value["selected"]
    fields = list(SAMPLE_FIELDS) + [f"{method}__{field}" for method in pool
                                  for field in ("probability", "logit", "rank", "cost")]
    with (output / "tables" / f"{value['problem']}_missed_samples.csv").open("w", newline="", encoding="utf-8") as stream:
        wide_writer = csv.DictWriter(stream, fieldnames=fields)
        wide_writer.writeheader()
        for index in np.flatnonzero(~value["correct"]):
            win, pick = int(winner[index]), int(selected[index])
            row = dict(problem=value["problem"], split=value["split"], index=int(value["indices"][index]),
                       nodes=int(value["nodes"][index]), true_winner=str(pool[win]), selected_solver=str(pool[pick]),
                       selected_probability=float(value["confidence"][index]),
                       selected_logit=float(value["logits"][index, pick]),
                       zero_selected_winner=summaries[win]["zero_selected_with_winner"],
                       near_zero_selected_winner=summaries[win]["near_zero_selected_with_winner"])
            for field in SAMPLE_FIELDS:
                if field in value and isinstance(value[field], np.ndarray):
                    row[field] = value[field][index].item()
            sample_writer.writerow(row)
            wide = dict(row)
            for slot, method in enumerate(pool):
                candidate_row = dict(problem=value["problem"], split=value["split"], index=row["index"],
                                     true_winner=row["true_winner"], selected_solver=row["selected_solver"],
                                     candidate=str(method), candidate_id=int(value["pool_ids"][slot]),
                                     valid=bool(value["mask"][index, slot]),
                                     probability=float(value["probabilities"][index, slot]),
                                     logit=float(value["logits"][index, slot]), rank=int(value["ranks"][index, slot]),
                                     cost=float(value["costs"][index, slot]), is_true_winner=slot == win, is_selected=slot == pick)
                candidate_writer.writerow(candidate_row)
                for field in ("probability", "logit", "rank", "cost"):
                    wide[f"{method}__{field}"] = candidate_row[field]
            wide_writer.writerow(wide)


def plot_missed(problem, rows, near_zero_pct=.1):
    rows = [row for row in rows if row["missed_winner_count"]]
    fig, axes = plt.subplots(1, 2, figsize=(12, max(4.2, 1.9 + .43 * len(rows))))
    fig.subplots_adjust(left=.17, right=.97, bottom=.19, top=.79, wspace=.52)
    fig.suptitle(f"{problem}｜TEST 漏选 winner 条件诊断", x=.04, y=.97, ha="left", fontsize=17, fontweight="bold")
    positions = np.arange(len(rows))
    colors = ["#C44E52" if r["zero_selected_with_winner"] else
              "#B48A32" if r["near_zero_selected_with_winner"] else "#147D80" for r in rows]
    for ax in axes:
        ax.set_yticks(positions)
        ax.set_yticklabels([r["solver"] for r in rows])
        ax.set_ylim(len(rows) - .4, -.6)
        ax.grid(axis="x", color="#E2E7E8", zorder=0)
        ax.tick_params(axis="y", length=0)
    means = np.array([r["missed_winner_probability_mean"] for r in rows]) * 100
    low = np.array([r["missed_winner_probability_q05"] for r in rows]) * 100
    high = np.array([r["missed_winner_probability_q95"] for r in rows]) * 100
    axes[0].barh(positions, means, color=colors, height=.55, zorder=3)
    axes[0].hlines(positions, low, high, color="#202528", linewidth=1.4, zorder=4)
    axes[0].scatter([r["missed_winner_probability_median"] * 100 for r in rows], positions,
                    color="#202528", s=15, zorder=5)
    axes[0].set_xlim(0, max(40, float(high.max()) + 5))
    axes[0].xaxis.set_major_formatter(PercentFormatter(100))
    axes[0].set_title("真实 winner 概率：均值、median、5–95%", fontsize=12)
    ranks = [r["missed_winner_rank_mean"] for r in rows]
    axes[1].barh(positions, ranks, color=colors, height=.55, zorder=3)
    for y, rank in zip(positions, ranks):
        axes[1].text(rank + .10, y, f"{rank:.2f}", va="center", fontsize=10)
    axes[1].set_xlim(0, max(ranks) + 1)
    axes[1].set_xlabel("平均分数排名，1 为最终选法")
    axes[1].set_title("只在该方法是 winner 且漏选时统计", fontsize=12)
    fig.text(.04, .04, f"红色：TEST 选 0 次且 winner>0；金色：非零选择≤{near_zero_pct:g}%；其余：全部漏选。标签分析，不作调参依据。", fontsize=9)
    return fig


def plot_zero_table(output, rows, split):
    paths = []
    for page, start in enumerate(range(0, len(rows), 20), 1):
        subset = rows[start:start + 20]
        fig, ax = plt.subplots(figsize=(14, 1.8 + .35 * len(subset)))
        ax.axis("off")
        ax.set_title(f"{split.upper()}｜选择 0 次且真实 winner>0（{page}）", loc="left", fontsize=16, pad=18)
        labels = ("问题", "方法", "winner数 / 比例", "选择数", "条件均值 P", "条件均值 rank", "漏选均值 Δcost", "本题损失贡献")
        cells = [[r["problem"], r["solver"], f"{r['true_winner_count']:,} / {r['true_winner_pct']:.2f}%", "0",
                  f"{100*r['winner_probability_mean']:.2f}%", f"{r['winner_rank_mean']:.2f}",
                  f"{r['missed_costloss_mean']:.5g}", f"{r['problem_costloss_share_pct']:.2f}%"] for r in subset]
        table = ax.table(cellText=cells, colLabels=labels, cellLoc="center", loc="center",
                         colWidths=(.09, .17, .17, .07, .11, .12, .14, .13))
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.65)
        for (row, _), cell in table.get_celld().items():
            cell.set_edgecolor("#D8DDDF")
            cell.set_facecolor("#EDF1F2" if row == 0 else "white" if row % 2 else "#F7F9F9")
        fig.text(.04, .025, "条件：本方法是真实 winner；成本贡献按本题总 selected−native winner 成本差计算，不跨题相加。", fontsize=9)
        fig.subplots_adjust(left=.03, right=.97, top=.86, bottom=.09)
        path = output / "figures" / f"zero_methods_{split}_{page:02d}.png"
        fig.savefig(path, dpi=180)
        fig.savefig(path.with_suffix(".pdf"))
        plt.close(fig)
        paths.append(path)
    return paths


def plot_calibration(output, bins, split):
    rows = [r for r in bins if r["problem"] == "ALL18_pooled" and r["count"]]
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    ax.plot([0, 1], [0, 1], color="#879398", linestyle="--", linewidth=1)
    ax.plot([r["mean_probability"] for r in rows], [r["observed_frequency"] for r in rows],
            marker="o", color="#147D80", linewidth=1.5)
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="模型最大 softmax 概率", ylabel="真实 Top1 正确率",
           title=f"{split.upper()}｜18 问题 pooled，固定20等宽 bins")
    ax.grid(color="#E2E7E8")
    fig.tight_layout()
    fig.savefig(output / "figures" / f"calibration_{split}.png", dpi=180)
    plt.close(fig)


def split_report(predictions, output, split, near_zero_pct, detailed=False):
    (output / "tables").mkdir(parents=True, exist_ok=True)
    (output / "figures").mkdir(parents=True, exist_ok=True)
    all_methods, candidates, flows, problem_rows = [], [], [], []
    sample_stream = candidate_stream = None
    if detailed:
        sample_stream = (output / "missed_samples.csv").open("w", newline="", encoding="utf-8")
        candidate_stream = (output / "missed_candidates.csv").open("w", newline="", encoding="utf-8")
        sample_writer = csv.DictWriter(sample_stream, fieldnames=SAMPLE_FIELDS)
        candidate_writer = csv.DictWriter(candidate_stream, fieldnames=CANDIDATE_FIELDS)
        sample_writer.writeheader()
        candidate_writer.writeheader()
    try:
        with (PdfPages(output / f"missed_probabilities_{split}.pdf") if detailed else nullcontext()) as pdf:
            for problem, value in predictions.items():
                methods, conditional, selected_flows = method_diagnostics(value, near_zero_pct)
                all_methods.extend(methods)
                candidates.extend(conditional)
                flows.extend(selected_flows)
                write_rows(output / "tables" / f"{split}_{problem}_method_summary.csv", methods)
                zero = [r for r in methods if r["zero_selected_with_winner"]]
                near = [r for r in methods if r["near_zero_selected_with_winner"]]
                missed = ~value["correct"]
                problem_rows.append(dict(problem=problem, split=split, instances=value["n"], candidates=value["k"],
                                         top1_pct=100 * float(value["correct"].mean()),
                                         missed_count=int(missed.sum()), zero_method_count=len(zero),
                                         zero_winner_count=sum(r["true_winner_count"] for r in zero),
                                         zero_winner_pct=100 * sum(r["true_winner_count"] for r in zero) / value["n"],
                                         near_zero_method_count=len(near), zero_methods=";".join(r["solver"] for r in zero),
                                         costloss_sum=float(value["costloss"].sum(dtype=np.float64)),
                                         mean_costloss=float(value["costloss"].mean()),
                                         relative_costloss_pct=float(np.nanmean(value["relative_costloss_pct"])),
                                         zero_costloss_share_pct=sum(r["problem_costloss_share_pct"] for r in zero),
                                         native_winner_nonminimum_count=int((value["winner_cost"] != value["oracle_cost"]).sum()),
                                         missed_cost_tie_count=int((missed & (value["costloss"] == 0)).sum()),
                                         negative_costloss_count=int((value["costloss"] < 0).sum())))
                if detailed:
                    write_detailed_problem(output, value, methods, sample_writer, candidate_writer)
                    fig = plot_missed(problem, methods, near_zero_pct)
                    fig.savefig(output / "figures" / f"{problem}.png", dpi=180)
                    pdf.savefig(fig)
                    plt.close(fig)
                print(f"{split}/{problem}: missed={int(missed.sum()):,}; zero-winner methods={len(zero)}", flush=True)
    finally:
        if sample_stream:
            sample_stream.close()
            candidate_stream.close()
    zero = [r for r in all_methods if r["zero_selected_with_winner"]]
    near = [r for r in all_methods if r["near_zero_selected_with_winner"]]
    write_rows(output / f"method_summary_{split}.csv", all_methods)
    write_rows(output / f"zero_methods_{split}.csv", zero, list(all_methods[0]))
    write_rows(output / f"near_zero_methods_{split}.csv", near, list(all_methods[0]))
    write_rows(output / f"missed_candidate_summary_{split}.csv", candidates)
    write_rows(output / f"missed_selection_frequencies_{split}.csv", flows)
    write_rows(output / f"problem_summary_{split}.csv", problem_rows)
    summaries, bins, classes, class_bins, aggregate = calibration_reports(predictions, split)
    write_rows(output / f"calibration_summary_{split}.csv", summaries)
    write_rows(output / f"calibration_bins_{split}.csv", bins)
    write_rows(output / f"class_calibration_{split}.csv", classes)
    write_rows(output / f"class_calibration_bins_{split}.csv", class_bins)
    aggregate.update(missed_count=sum(r["missed_count"] for r in problem_rows), zero_method_count=len(zero),
                     zero_winner_count=sum(r["true_winner_count"] for r in zero), near_zero_method_count=len(near),
                     zero_winner_pct=100 * sum(r["true_winner_count"] for r in zero) / aggregate["instances"])
    write_rows(output / f"split_summary_{split}.csv", [aggregate])
    plot_zero_table(output, zero, split)
    plot_calibration(output, bins, split)
    return dict(methods=all_methods, problems=problem_rows, aggregate=aggregate)


def method_table(rows):
    lines = ["| 问题 | 方法 | truefreq | selectedfreq | 条件 meanprob | meanrank | 漏选 meanΔcost | 本题 costloss贡献 |",
             "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for r in rows:
        lines.append(f"| {r['problem']} | {r['solver']} | {r['true_winner_count']:,} / {r['true_winner_pct']:.2f}% | "
                     f"{r['selected_count']:,} / {r['selected_pct']:.2f}% | {100*r['winner_probability_mean']:.2f}% | "
                     f"{r['winner_rank_mean']:.2f} | {r['missed_costloss_mean']:.6g} | {r['problem_costloss_share_pct']:.2f}% |")
    return "\n".join(lines)


def write_test_report(output, report, near_zero_pct):
    aggregate, methods = report["aggregate"], report["methods"]
    zero = [r for r in methods if r["zero_selected_with_winner"]]
    lehd = next(r for r in methods if r["problem"] == "TSP" and r["solver"] == "LEHD")
    near = next(r for r in methods if r["problem"] == "CVRP" and r["solver"] == "LEHD")
    lines = ["# R76 TEST 漏选 winner：标签分析，非调参依据", "",
             "此目录仅用于已锁定模型的 TEST 标签诊断，不用于研究规则设计、阈值选择或调参。TRAIN/VAL 研究证据隔离于 `research_train_val/`，不把 TEST 统计传作研究依据。",
             f"18 问题共 {aggregate['instances']:,} 例；漏选 {aggregate['missed_count']:,} 例。"
             f"选择0次且 winner>0 的方法-问题对 {len(zero)} 个，覆盖 {aggregate['zero_winner_count']:,} 个获胜样本"
             f"（全体 {aggregate['zero_winner_pct']:.2f}%，全部漏选 {100*aggregate['zero_winner_count']/aggregate['missed_count']:.2f}%）。",
             "概率是已保存最终 maximin logits 在合法候选池内的 masked FP32 softmax（温度1），不是基础 utility，也不是 pairwise 胜率。未校准，不应解释成已经验证的正确率。选法/排名仍由原 logits 决定，完全并列按 global solver ID 升序。",
             f"零选择：selected_count=0；near-zero：0<selected_pct≤{near_zero_pct:g}%。这是描述性分组，不改变模型规则；所有方法的全部漏选仍完整保留。",
             "成本保留 FP64；Δcost=所选成本−native winner成本。costloss贡献分母是本题总Δcost，不将不同问题的原始成本相加。并列漏选和非最小 native 标签单独计数。",
             "## 关键观察", "",
             f"1. TSP / LEHD 真实获胜 {lehd['true_winner_count']} 次但选0次；条件概率均值 {100*lehd['winner_probability_mean']:.2f}%，"
             f"平均分数排名 {lehd['winner_rank_mean']:.2f}，贡献本题 {lehd['problem_costloss_share_pct']:.2f}% 的成本损失。零选择并不等于概率接近0。",
             f"2. CVRP / LEHD 真实获胜 {near['true_winner_count']} 次，模型只选 {near['selected_count']} 次且正确 {near['correct_selected_count']} 次；"
             f"条件概率均值 {100*near['winner_probability_mean']:.2f}%，平均排名 {near['winner_rank_mean']:.2f}。该 near-zero 现象与全部漏选分析分别保留。",
             "3. 这里只描述现有模型的错误类型，不评估、推荐或用 TEST 选择任何修正规则。", "",
             "## 全部零选择且真实获胜的方法", "", method_table(zero), "",
             "## 输出", "",
             "- `missed_samples.csv`：每个漏选样本的 winner、最终选法、条件概率、分数、排名和成本差。",
             "- `missed_candidates.csv`：每个漏选样本全部候选的概率、分数、排名、成本。",
             "- `tables/<问题>_missed_samples.csv`：逐样本宽表，按候选名称展开。",
             "- `method_summary_test.csv` / `missed_candidate_summary_test.csv`：按真实 winner 聚合的均值、中位数、5/25/75/95分位数。",
             "- `missed_selection_frequencies_test.csv`：真实 winner 漏选后最终选法频数及成本差。",
             "- `zero_methods_test.csv` / `near_zero_methods_test.csv` / `problem_summary_test.csv`：完整分组及成本贡献。",
             "- `figures/zero_methods_test_*.png`：小表格图；`missed_probabilities_test.pdf`：18题概率与排名图。",
             "- `calibration_*_test.csv` / `class_calibration*_test.csv`：TEST 描述性校准诊断，禁止作为调参依据。"]
    (output / "RESULTS.md").write_text("\n\n".join(lines) + "\n", encoding="utf-8")


def write_research_report(output, train, val, lock):
    reports = {"train": train, "val": val}
    lookup = {(r["problem"], r["solver"]): r for r in train["methods"]}
    comparisons = []
    for vr in val["methods"]:
        tr = lookup[vr["problem"], vr["solver"]]
        row = dict(problem=vr["problem"], solver=vr["solver"])
        for split, record in (("train", tr), ("val", vr)):
            for key in ("instances", "selected_count", "selected_pct", "true_winner_count", "true_winner_pct",
                        "zero_selected_with_winner", "near_zero_selected_with_winner", "winner_probability_mean",
                        "winner_probability_median", "winner_rank_mean", "missed_winner_count",
                        "missed_winner_probability_mean", "missed_costloss_mean", "problem_costloss_share_pct"):
                row[split + "_" + key] = record[key]
        row["zero_in_both_with_winner"] = tr["zero_selected_with_winner"] and vr["zero_selected_with_winner"]
        comparisons.append(row)
    write_rows(output / "train_val_comparison.csv", comparisons)
    common = sum(r["zero_in_both_with_winner"] for r in comparisons)
    lines = ["# R76 TRAIN/VAL 研究证据：与 TEST 隔离", "",
             f"接收研究代理：`{RESEARCH_AGENT}`，用于研究理解而非代码 review。只读取 TRAIN `selected_best` 与 VAL `epoch{lock['epoch']:03d}` 的现有 NPZ；锁定 epoch {lock['epoch']}，checkpoint SHA256 `{lock['checkpoint_sha256']}`。此报告不含 TEST 数字，不根据 TEST 调整规则。",
             "这里做标签条件分析，不训练、不神经推理、不拟合温度、不修改或验证新决策规则。若后续设计规则，只能用 TRAIN 研究及 VAL 验证，TEST 必须保持隔离。",
             "| 指标 | TRAIN | VAL |", "| --- | ---: | ---: |"]
    for label, key, fmt in (("实例数", "instances", ",d"), ("Top1 %", "top1_pct", ".4f"),
                            ("漏选数", "missed_count", ",d"), ("宏平均 CE", "macro_ce", ".6f"),
                            ("宏平均 Brier（多类平方和）", "macro_brier", ".6f"),
                            ("宏平均 ECE20", "macro_ece20", ".6f"), ("pooled ECE20", "pooled_ece20", ".6f"),
                            ("选0次且 winner>0 的方法-问题对", "zero_method_count", "d"),
                            ("上述方法获胜样本数", "zero_winner_count", ",d")):
        lines.append(f"| {label} | {format(train['aggregate'][key], fmt)} | {format(val['aggregate'][key], fmt)} |")
    lines.extend(("", "## 关键观察", "",
                  f"1. TRAIN/VAL 同时选0次且真实获胜的方法-问题对有 {common} 个；零选择现象不只存在于 TEST。",
                  "2. 条件 meanprob 是 P(本方法) 在本方法真实获胜的样本上平均，不是模型选择频数；遗漏样本有另一套独立条件统计。分数排名和 logit gap 可以区分概率质量低与无法排到第一。",
                  "3. CE 用稳定 FP32 log-softmax 的 native 标签负对数概率；Brier 是所有候选平方误差之和。ECE 使用预先固定的20个等宽 bins，不拟合、不选 bin、不调温度。宏平均先逐题计算再平均；pooled ECE 合并全部18题，不等于宏平均 ECE。",
                  "4. ECE 是分箱诊断而非校准已证明；每个方法另有二元 classwise ECE/Brier、平均概率与真实获胜频率差。不存在新规则的性能证据。"))
    for split, report in reports.items():
        lines.extend(("", f"## {split.upper()}：全部选0次且真实获胜的方法", "",
                      method_table([r for r in report["methods"] if r["zero_selected_with_winner"]])))
    lines.extend(("", "## 输出", "",
                  "`method_summary_train/val.csv`：全部方法的真实获胜/模型选择频数及 winner/漏选条件概率、分数、排名、成本差分位数。",
                  "`zero_methods_train/val.csv`、`near_zero_methods_train/val.csv`、`train_val_comparison.csv`：完整分组与名称对齐比较。",
                  "`missed_candidate_summary_train/val.csv`、`missed_selection_frequencies_train/val.csv`：全部漏选的候选概率与最终选法频数。",
                  "`calibration_summary_train/val.csv`、`calibration_bins_train/val.csv`、`class_calibration_train/val.csv`、`class_calibration_bins_train/val.csv`：CE、Brier、ECE及原始分箱统计。",
                  "`figures/zero_methods_train/val_*.png`、`figures/calibration_train/val.png`：小表格及校准图。"))
    text = "\n".join(lines) + "\n"
    (output / "RESULTS.md").write_text(text, encoding="utf-8")
    critical = ["# R76 研究交接：仅 TRAIN/VAL 关键数字", "",
                f"接收研究代理：`{RESEARCH_AGENT}`。这是研究数据交接，不是代码 review。", "",
                f"锁定 epoch {lock['epoch']}；SHA256 `{lock['checkpoint_sha256']}`。仅读取已有保存 NPZ，无推理、训练或规则改变。",
                "此文件不含 TEST 统计；TEST 标签诊断单独隔离，不能用于研究规则、阈值或调参。以下是标签条件分析，不是任何新规则的验证。", "",
                "| 关键指标 | TRAIN | VAL |", "| --- | ---: | ---: |"]
    for label, key, fmt in (("实例数", "instances", ",d"), ("Top1 %", "top1_pct", ".4f"),
                            ("宏平均 CE", "macro_ce", ".6f"), ("宏平均多类 Brier", "macro_brier", ".6f"),
                            ("pooled ECE20", "pooled_ece20", ".6f"), ("zero-count且winner>0的方法-问题对", "zero_method_count", "d"),
                            ("上述方法的获胜样本数", "zero_winner_count", ",d")):
        critical.append(f"| {label} | {format(train['aggregate'][key], fmt)} | {format(val['aggregate'][key], fmt)} |")
    val_lookup = {(r["problem"], r["solver"]): r for r in val["methods"]}
    for problem in ("TSP", "CVRP"):
        tr, vr = lookup[problem, "LEHD"], val_lookup[problem, "LEHD"]
        for label, key, fmt, factor in (("winner次数", "true_winner_count", ",d", 1),
                                        ("模型选择次数", "selected_count", ",d", 1),
                                        ("P(LEHD) 条件均值 %", "winner_probability_mean", ".4f", 100),
                                        ("条件平均分数排名", "winner_rank_mean", ".4f", 1),
                                        ("条件 CE", "winner_ce_mean", ".6f", 1),
                                        ("本题成本损失贡献 %", "problem_costloss_share_pct", ".4f", 1)):
            critical.append(f"| {problem}/LEHD {label} | {format(tr[key]*factor, fmt)} | {format(vr[key]*factor, fmt)} |")
    critical.extend(("", f"TRAIN/VAL 共有 {common} 个方法-问题对同时选0次且真实获胜。",
                     "观察：零选择与 near-zero 在 TRAIN/VAL 都存在；即使有15–19%的条件概率质量，也可能分数排不到第一。平均概率不等于选择频数，也不是校准已证实的正确率。",
                     "ECE 固定20等宽bins，不拟合温度、不选择分箱；条件 CE 是本方法真实获胜样本上的稳定 FP32 NLL。不同问题的原始成本不跨题相加。", "",
                     "完整证据：[RESULTS.md](RESULTS.md)、[TRAIN方法表](method_summary_train.csv)、[VAL方法表](method_summary_val.csv)、"
                     "[TRAIN/VAL名称对齐](train_val_comparison.csv)、[TRAIN校准](calibration_summary_train.csv)、[VAL校准](calibration_summary_val.csv)。"))
    (output / "RESEARCH_HANDOFF.md").write_text("\n".join(critical) + "\n", encoding="utf-8")


def align_by_name(reference_pool, pool, values):
    reference_pool, pool = list(reference_pool), list(pool)
    if len(set(pool)) != len(pool) or set(reference_pool) != set(pool):
        raise ValueError("Candidate names must match exactly for distribution comparison")
    return np.asarray(values)[[pool.index(name) for name in reference_pool]]


def distribution_rows(value, dataset):
    rows = []
    for slot, name in enumerate(value["pool"]):
        selected = int((value["selected"] == slot).sum())
        winner = int((value["winner"] == slot).sum())
        rows.append(dict(dataset=dataset, problem=value["problem"], split=value["split"], solver=str(name),
                         solver_id=int(value["pool_ids"][slot]), instances=value["n"],
                         selected_count=selected, selected_pct=100 * selected / value["n"],
                         winner_count=winner, winner_pct=100 * winner / value["n"],
                         selection_minus_winner_pp=100 * (selected - winner) / value["n"],
                         correct_selected_count=int(((value["selected"] == slot) & value["correct"]).sum())))
    return rows


def plot_lib(dataset, lib, train, comparisons):
    pool = lib["pool"]
    fig, axes = plt.subplots(1, 3, figsize=(19, max(5.6, 2.2 + .47 * len(pool))))
    fig.subplots_adjust(left=.11, right=.985, bottom=.17, top=.78, wspace=.65)
    fig.suptitle(f"{dataset}｜LIB 选法、真 winner 与 TRAIN 模型选法", x=.04, y=.97,
                 ha="left", fontsize=20, fontweight="bold")
    for ax, key, title, color in zip(axes, ("lib_selected_pct", "lib_winner_pct", "train_selected_pct"),
                                   (f"LIB 模型选择｜{lib['n']:,}例", f"LIB 真实winner｜{lib['n']:,}例",
                                    f"TRAIN 模型选择｜{train['n']:,}例"), ("#147D80", "#C44E52", "#B48A32")):
        positions = np.arange(len(pool))
        values = [r[key] for r in comparisons]
        ax.barh(positions, values, height=.60, color=color, zorder=3)
        for y, fraction in zip(positions, values):
            ax.text(fraction - 1 if fraction > 90 else fraction + 1, y, f"{fraction:.2f}%", va="center",
                    ha="right" if fraction > 90 else "left", color="white" if fraction > 90 else "#202528", fontsize=10)
        ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
        ax.set_yticks(positions)
        ax.set_yticklabels(pool)
        ax.set_ylim(len(pool) - .4, -.6)
        ax.set_xlim(0, 100)
        ax.set_xticks(np.arange(0, 101, 20))
        ax.xaxis.set_major_formatter(PercentFormatter(100, decimals=0))
        ax.set_xlabel("占各自本问题实例的比例")
        ax.grid(axis="x", color="#E2E7E8", linewidth=.7, zorder=0)
        ax.tick_params(axis="y", length=0)
    fig.text(.04, .04, "同一 epoch 18 checkpoint；按方法名称对齐。模型选择与真实 winner 分开统计，右图不是 TRAIN 标签分布。", fontsize=10)
    return fig


def lib_report(run_root, output, train_predictions, lock):
    (output / "figures").mkdir(parents=True, exist_ok=True)
    (output / "tables").mkdir(parents=True, exist_ok=True)
    all_rows, all_comparisons, summaries, paths = [], [], [], []
    sections = ["# R76 LIB 选择/真实 winner 与 TRAIN 模型选择分布", "",
                "复用 TEST 分布报告的中文横向条形图、独立分母、表格、逐问题 PNG/PDF/CSV、HTML 和整包样式；不覆盖 TEST 报告。",
                f"同一锁定 epoch {lock['epoch']}，checkpoint SHA256 `{lock['checkpoint_sha256']}`。无新推理或训练，候选按名称对齐并验证 global solver ID。",
                "每图三栏：LIB 模型选法、LIB native 标签真实 winner、TRAIN 模型选法。TRAIN 标签频数另有明确字段，不替代模型选择；仅比较 TSP/CVRP 各自对应问题，不混合18题。",
                "[PDF](selection_distributions.pdf) · [长表CSV](selection_distribution.csv) · [名称对齐宽表](comparison.csv) · [完整图表包](selection_distribution_bundle.zip)"]
    with PdfPages(output / "selection_distributions.pdf") as pdf:
        for dataset, problem in (("TSPLIB", "TSP"), ("CVRPLIB", "CVRP")):
            path = run_root / "lib_evaluation/predictions" / f"{dataset}.npz"
            protocol_path = run_root / "lib_evaluation" / f"{dataset}_protocol.json"
            result_path = run_root / "lib_evaluation" / f"{dataset}_results.json"
            protocol, result = read_json(protocol_path), read_json(result_path)
            for metadata in (protocol, result["protocol"]):
                if (metadata["checkpoint_sha256"] != lock["checkpoint_sha256"]
                        or metadata["checkpoint_epoch"] != lock["epoch"]):
                    raise ValueError("LIB predictions are not from the locked checkpoint")
            with np.load(path, allow_pickle=False) as saved:
                if not np.array_equal(saved["scores"], saved["logits"]):
                    raise ValueError("R76 LIB final scores differ from maximin logits")
                raw = {key: saved[key].copy() for key in
                       ("logits", "costs", "winner", "indices", "pool", "pool_ids", "nodes")}
                saved_pred = saved["pred"].copy()
            lib = prepare_prediction(dict(raw, problem=problem, split="lib"))
            np.testing.assert_array_equal(lib["selected"], saved_pred[np.argsort(raw["indices"])])
            check_metrics(lib, dict(result["native"], pool=protocol["candidate_pool"]))
            train = train_predictions[problem]
            np.testing.assert_array_equal(align_by_name(lib["pool"], train["pool"], train["pool_ids"]), lib["pool_ids"])
            lib_rows, train_rows = distribution_rows(lib, dataset), distribution_rows(train, "TRAIN_" + problem)
            lookup = {r["solver"]: r for r in train_rows}
            comparisons = []
            for lr in lib_rows:
                tr = lookup[lr["solver"]]
                comparisons.append(dict(dataset=dataset, problem=problem, solver=lr["solver"], solver_id=lr["solver_id"],
                                        lib_instances=lib["n"], train_instances=train["n"],
                                        lib_selected_count=lr["selected_count"], lib_selected_pct=lr["selected_pct"],
                                        lib_winner_count=lr["winner_count"], lib_winner_pct=lr["winner_pct"],
                                        train_selected_count=tr["selected_count"], train_selected_pct=tr["selected_pct"],
                                        train_winner_count=tr["winner_count"], train_winner_pct=tr["winner_pct"],
                                        lib_minus_train_selection_pp=lr["selected_pct"] - tr["selected_pct"]))
            tv = .5 * sum(abs(r["lib_minus_train_selection_pp"]) for r in comparisons)
            summaries.append(dict(dataset=dataset, problem=problem, lib_instances=lib["n"], train_instances=train["n"],
                                  lib_top1_pct=100 * float(lib["correct"].mean()), train_top1_pct=100 * float(train["correct"].mean()),
                                  lib_mean_cost=float(lib["selected_cost"].mean()), selection_total_variation_pct=tv,
                                  lib_methods_ever_selected=sum(r["selected_count"] > 0 for r in lib_rows),
                                  train_methods_ever_selected=sum(r["selected_count"] > 0 for r in train_rows),
                                  lib_zero_selected_with_winner=sum(r["selected_count"] == 0 and r["winner_count"] > 0 for r in lib_rows)))
            all_rows.extend(lib_rows + train_rows)
            all_comparisons.extend(comparisons)
            write_rows(output / "tables" / f"{dataset}.csv", comparisons)
            fig = plot_lib(dataset, lib, train, comparisons)
            fig.savefig(output / "figures" / f"{dataset}.png", dpi=180)
            fig.savefig(output / "figures" / f"{dataset}.pdf")
            pdf.savefig(fig)
            plt.close(fig)
            table = ["| 方法 | LIB 选择数/比例 | LIB winner数/比例 | TRAIN 模型选择数/比例 | LIB−TRAIN 选择 pp |",
                     "| --- | ---: | ---: | ---: | ---: |"]
            for r in comparisons:
                table.append(f"| {r['solver']} | {r['lib_selected_count']} / {r['lib_selected_pct']:.2f}% | "
                             f"{r['lib_winner_count']} / {r['lib_winner_pct']:.2f}% | {r['train_selected_count']:,} / "
                             f"{r['train_selected_pct']:.2f}% | {r['lib_minus_train_selection_pp']:+.2f} |")
            sections.extend((f"## {dataset}", f"LIB Top1 {summaries[-1]['lib_top1_pct']:.2f}%；LIB 与 TRAIN 选法总变差距离 {tv:.2f}%。",
                             f"![{dataset}分布](figures/{dataset}.png)", "\n".join(table),
                             f"[该问题CSV](tables/{dataset}.csv) · [本图PDF](figures/{dataset}.pdf)"))
            paths.extend((path, protocol_path, result_path))
    write_rows(output / "selection_distribution.csv", all_rows)
    write_rows(output / "comparison.csv", all_comparisons)
    write_rows(output / "summary.csv", summaries)
    train_rows = [row for problem, value in train_predictions.items() for row in distribution_rows(value, "TRAIN_" + problem)]
    write_rows(output / "train_selection_distribution.csv", train_rows)
    source = output / "by_problem.md"
    source.write_text("\n\n".join(sections) + "\n", encoding="utf-8")
    render_html(source)
    lines = ["# R76 LIB / TRAIN 分布对比", "",
             "同一 epoch 18 锁定模型；仅复用已有预测，按候选方法名称对齐。LIB 真 winner 使用 native 标签，绝不作为模型选择频数。TRAIN 的模型选择来自 logits argmax，不是 TRAIN 标签。",
             "| 数据集 | LIB数 | TRAIN数 | LIB Top1 | LIB 选过方法数 | TRAIN选过方法数 | 选法 TV 距离 |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in summaries:
        lines.append(f"| {row['dataset']} | {row['lib_instances']} | {row['train_instances']:,} | {row['lib_top1_pct']:.2f}% | "
                     f"{row['lib_methods_ever_selected']} | {row['train_methods_ever_selected']} | {row['selection_total_variation_pct']:.2f}% |")
    reld = next(r for r in all_comparisons if r["dataset"] == "CVRPLIB" and r["solver"] == "RELD_CVRP")
    lines.extend(("", f"1. CVRPLIB 的 RELD_CVRP 真实获胜 {reld['lib_winner_count']}/100 次，但模型选择 {reld['lib_selected_count']} 次；"
                  f"同 checkpoint 的 TRAIN 模型选择比例为 {reld['train_selected_pct']:.2f}%。这描述固定模型的分布，不用于调参。",
                  "2. LIB 样本数较小（TSPLIB 49、CVRPLIB 100）；这是固定模型的分布观察，不是新规则或训练方案的验证。",
                  "3. 选法分布变化与真实 winner 分布变化是不同量，宽表分别保留；TV=半个各方法选择比例绝对差之和，0%完全一致，100%完全分离。",
                  "主要图表见 `by_problem.html`；原始选择/标签频数见 `selection_distribution.csv`，TRAIN 全18题模型选法另存 `train_selection_distribution.csv`。"))
    (output / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return paths, summaries


def verify_sources(run_root):
    lock = read_json(run_root / "final_test_lock.json")
    if file_hash(Path(lock["checkpoint"])) != lock["checkpoint_sha256"]:
        raise ValueError("Locked checkpoint SHA256 mismatch")
    training = read_json(run_root / "training_complete.json")["selected_best"]
    test = read_json(run_root / "test_results.json")
    if (training["checkpoint_sha256"] != lock["checkpoint_sha256"] or training["epoch"] != lock["epoch"]
            or test["checkpoint_sha256"] != lock["checkpoint_sha256"]):
        raise ValueError("TRAIN/TEST metadata do not share the locked checkpoint")
    val_directory = run_root / "evaluation/val" / f"epoch{lock['epoch']:03d}"
    val_metrics = read_json(val_directory / "metrics.json")
    for key in ("macro_ce", "macro_top1", "macro_vs_sbs_pct"):
        np.testing.assert_allclose(val_metrics["macro"][key], lock["val"][key], rtol=0, atol=1e-12)
    return lock, val_directory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=RUN_ROOT)
    parser.add_argument("--near-zero-pct", type=float, default=.1, help="Descriptive only; not a decision threshold")
    args = parser.parse_args()
    if not 0 <= args.near_zero_pct <= 100:
        parser.error("--near-zero-pct must be in [0, 100]")
    root = args.run_root.resolve()
    output = root / "missed_winner_probabilities"
    research = output / "research_train_val"
    lib_output = root / "lib_selection_distribution"
    protected = sorted(p for p in (root / "test_selection_distribution").rglob("*") if p.is_file())
    protected_before = {str(p): file_hash(p) for p in protected}
    lock, val_directory = verify_sources(root)
    test_dir, train_dir = (root / f"evaluation/{split}/selected_best" for split in ("test", "train"))
    test_metrics, train_metrics = read_json(test_dir / "metrics.json"), read_json(train_dir / "metrics.json")
    val_metrics = read_json(val_directory / "metrics.json")
    if any(set(metrics["per_problem"]) != set(PROBLEMS) for metrics in (test_metrics, train_metrics, val_metrics)):
        raise ValueError("All 18 problems are required")
    test, test_paths = load_shards(test_dir, "test", test_metrics["per_problem"])
    train, train_paths = load_shards(train_dir, "train", train_metrics["per_problem"])
    val, val_paths = load_shards(val_directory, "val", val_metrics["per_problem"])
    style()
    test_report = split_report(test, output, "test", args.near_zero_pct, detailed=True)
    write_test_report(output, test_report, args.near_zero_pct)
    train_report = split_report(train, research, "train", args.near_zero_pct)
    val_report = split_report(val, research, "val", args.near_zero_pct)
    write_research_report(research, train_report, val_report, lock)
    lib_paths, lib_summary = lib_report(root, lib_output, train, lock)
    protected_after = {str(p): file_hash(p) for p in
                       sorted(p for p in (root / "test_selection_distribution").rglob("*") if p.is_file())}
    if protected_before != protected_after:
        raise RuntimeError("Protected TEST distribution changed during this analysis")
    source_sets = ((output, test_paths + [test_dir / "metrics.json"], "TEST label analysis only; never for tuning"),
                   (research, train_paths + val_paths + [train_dir / "metrics.json", val_directory / "metrics.json"],
                    "TRAIN/VAL research evidence only; no TEST statistics"),
                   (lib_output, lib_paths + train_paths + [train_dir / "metrics.json"], "LIB/TRAIN model selection comparison"))
    for directory, paths, purpose in source_sets:
        protocol = dict(purpose=purpose, checkpoint=lock["checkpoint"], checkpoint_sha256=lock["checkpoint_sha256"],
                        selected_epoch=lock["epoch"], new_forward_passes=0, training_runs=0, gpu_used=False,
                        selection_rule="maximum final saved maximin logit; exact ties by ascending global solver ID",
                        probability="masked FP32 softmax of final maximin logits; temperature 1; no calibration fitting",
                        winner="native saved winner; never reassign using FP32 argmin",
                        precision="FP32 softmax; original FP64 costs; FP64 summary reductions",
                        near_zero_selected_pct=args.near_zero_pct, calibration_bins=20,
                        quantiles={key: q for key, q in QUANTILES},
                        source_predictions=[dict(path=str(p), sha256=file_hash(p)) for p in paths],
                        protected_test_distribution_unchanged=True,
                        protected_test_distribution_sha256=protected_before,
                        script_sha256=file_hash(Path(__file__)))
        write_json(directory / "protocol.json", protocol)
    with zipfile.ZipFile(lib_output / "selection_distribution_bundle.zip", "w", zipfile.ZIP_DEFLATED) as bundle:
        for path in sorted(lib_output.rglob("*")):
            if path.is_file() and path.suffix != ".zip":
                bundle.write(path, str(path.relative_to(lib_output)))
        bundle.write(Path(__file__), "scripts/r76_probability_diagnostics.py")
    print(json.dumps(dict(test_label_analysis=test_report["aggregate"],
                          train_research=train_report["aggregate"], val_research=val_report["aggregate"],
                          lib_comparison=lib_summary, protected_test_distribution_unchanged=True), indent=2), flush=True)


if __name__ == "__main__":
    main()
