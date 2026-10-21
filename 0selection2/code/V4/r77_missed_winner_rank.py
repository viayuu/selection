"""Report a missed winner's predicted ranks from the locked R77B TEST scores."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import PercentFormatter

from code.V4.r77_best_model_report import ROOT, load_predictions, read_json, setup_style, write_csv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--problem", default="TSP")
    parser.add_argument("--solver", default="LEHD")
    args = parser.parse_args()
    run = ROOT / "B_full_seed2"
    path = run / "evaluation/test/selected_best/rank000.npz"
    lock = read_json(run / "final_test_lock.json")
    prefix = args.problem + "__"
    with np.load(path, allow_pickle=False) as archive:
        data = load_predictions(archive, path, prefix)
        scores = archive[prefix + "logits"].copy()
        nodes = archive[prefix + "nodes"].copy()
    slot = data.pool.index(args.solver)
    # Preserve the original descending-score / ascending-global-ID tie policy.
    global_order = np.argsort(data.pool_ids, kind="stable")
    ranking = global_order[np.argsort(-scores[:, global_order], axis=1, kind="stable")]
    np.testing.assert_array_equal(ranking[:, 0], data.pred)
    ranks = np.argmax(ranking == slot, axis=1) + 1
    missed = np.flatnonzero((data.winner == slot) & (data.pred != slot))
    n = len(missed)
    if not n:
        raise ValueError("No missed winners in the requested subset")
    counts = np.bincount(ranks[missed], minlength=len(data.pool) + 1)[1:]
    rows = [dict(predicted_rank=rank, count=int(count), percentage=float(100 * count / n),
                 denominator_missed_winners=n)
            for rank, count in enumerate(counts, 1)]
    groups = [("2", int(counts[1])), ("3", int(counts[2])), (">3", int(counts[3:].sum()))]
    grouped = [dict(predicted_rank_group=group, count=count, percentage=100 * count / n,
                    denominator_missed_winners=n) for group, count in groups]
    output = run / "full_model_statistics" / "missed_winner_rank" / f"{args.problem}_{args.solver}"
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "rank_distribution.csv", rows)
    write_csv(output / "grouped_rank_distribution.csv", grouped)
    write_csv(output / "missed_instances.csv", [dict(
        instance_index=int(data.indices[i]),
        source_index="" if data.source_indices is None else int(data.source_indices[i]),
        nodes=int(nodes[i]), true_winner=args.solver, selected_solver=data.pool[data.pred[i]],
        winner_predicted_rank=int(ranks[i]), winner_score=float(scores[i, slot]),
        selected_score=float(scores[i, data.pred[i]]),
        full_predicted_order=" > ".join(data.pool[j] for j in ranking[i])) for i in missed])
    manifest = read_json(ROOT / "selected_train_val_predictions/B/manifest.json")
    version = Path(manifest["data_root"]).name
    metadata = dict(model="R77B", problem=args.problem, solver=args.solver,
        dataset="new TEST", data_version=version, total_instances=data.n,
        model_top1_native_pct=data.top1_pct, solver_true_winners=int((data.winner == slot).sum()),
        solver_selected_count=int((data.pred == slot).sum()), missed_winners=n,
        definition="saved native winner == solver AND original selected solver != solver",
        percentage_denominator="missed_winners, not all TEST errors or all TEST instances",
        rank_rule="descending saved logits; exact ties use ascending global solver ID; rank starts at 1",
        missed_solver_score_tie_rows=int(((scores[missed] == scores[missed, slot, None]).sum(1) > 1).sum()),
        checkpoint_sha256=lock["checkpoint_sha256"], checkpoint_epoch=lock["epoch"],
        checkpoint_update=lock["selected_update"], prediction_path=str(path), candidate_order=data.pool,
        inference=False, training=False, checkpoint_reselection=False, ranks=rows, grouped_ranks=grouped)
    with (output / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    setup_style()
    fig = plt.figure(figsize=(15.5, 9))
    fig.text(0.045, 0.935, f"R77B | {args.problem} TEST | {args.solver} \u6f0f\u9009\u540e\u7684\u9884\u6d4b\u540d\u6b21",
             fontsize=24, color="#093f45")
    fig.text(0.045, 0.875,
        f"TEST: {data.n:,} \u4f8b | \u6a21\u578bTop1={data.top1_pct:.2f}% | "
        f"{args.solver} \u771f\u5b9e winner: {metadata['solver_true_winners']:,} \u4f8b | "
        f"\u6f0f\u9009: {n:,} \u4f8b", fontsize=13)
    fig.text(0.045, 0.825,
        f"\u4ec5\u7edf\u8ba1\u771f\u5b9e winner={args.solver} \u4e14\u6a21\u578b\u672a\u9009\u8be5\u65b9\u6cd5\u7684\u6837\u672c\uff1b\u4ee5\u4e0b\u6bd4\u4f8b\u7684\u5206\u6bcd\u5747\u4e3a {n:,}\u3002",
        fontsize=12, color="#566368")
    top3_pct = 100 * float(counts[1:3].sum()) / n
    fig.text(0.045, 0.745,
        f"{top3_pct:.2f}% \u7684\u6f0f\u9009\u6837\u672c\u4e2d\uff0c{args.solver} \u6392\u5728\u7b2c 2 \u6216\u7b2c 3 \u540d",
        fontsize=19, color="#093f45")
    table_ax = fig.add_axes([0.045, 0.265, 0.39, 0.39])
    table_ax.set_axis_off()
    table = table_ax.table(
        cellText=[[label, f"{row['count']:,}", f"{row['percentage']:.2f}%"]
                  for label, row in zip(("\u7b2c 2 \u540d", "\u7b2c 3 \u540d", "\u7b2c 3 \u540d\u4ee5\u540e"), grouped)],
        colLabels=["\u9884\u6d4b\u540d\u6b21", "\u5b9e\u4f8b\u6570", "\u5360\u6f0f\u9009\u6bd4\u4f8b"],
        cellLoc="center", bbox=[0, 0, 1, 1], colWidths=[0.44, 0.24, 0.32])
    table.auto_set_font_size(False)
    table.set_fontsize(16)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#d6e0e5")
        cell.set_linewidth(0.7)
        cell.set_facecolor("#dce9ee" if row == 0 else ("#f3f6f8" if row % 2 == 0 else "white"))
        if row == 0:
            cell.set_text_props(color="#093f45", weight="bold")
    ax = fig.add_axes([0.55, 0.235, 0.41, 0.415])
    y = np.arange(len(data.pool) - 1)
    percentages = counts[1:] * 100 / n
    ax.barh(y, percentages, height=0.58, color="#197d85", zorder=3)
    for row, (count, pct) in enumerate(zip(counts[1:], percentages)):
        ax.text(pct + 0.65, row, f"{pct:.2f}% ({count:,})", va="center", fontsize=12)
    ax.set_yticks(y, labels=[f"\u7b2c {rank} \u540d" for rank in range(2, len(data.pool) + 1)])
    ax.set_ylim(len(y) - 0.4, -0.6)
    ax.set_xlim(0, max(60, float(percentages.max()) + 17))
    ax.xaxis.set_major_formatter(PercentFormatter(100, decimals=0))
    ax.grid(axis="x", color="#dfe6e9", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", length=0, labelsize=12)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#aab2b7")
    ax.set_xlabel("\u5360\u6f0f\u9009\u6837\u672c\u7684\u6bd4\u4f8b (%)", labelpad=12)
    ax.set_title("\u5b8c\u6574\u540d\u6b21\u5206\u5e03", fontsize=15, pad=15)
    fig.text(0.045, 0.13, f"\u6570\u636e\u7248\u672c: {version}", fontsize=10.5, color="#566368")
    fig.text(0.045, 0.086,
        f"\u539f\u751f native/FP64 ind | \u5206\u6570\u964d\u5e8f\uff0c\u5e76\u5217\u6309 global solver ID | "
        f"epoch {lock['epoch']} / update {lock['selected_update']} / SHA256 {lock['checkpoint_sha256'][:16]}",
        fontsize=10.5, color="#566368")
    fig.text(0.045, 0.045,
        "\u4ec5\u8bfb\u5df2\u9501\u5b9a TEST \u9884\u6d4b\u8fdb\u884c\u7edf\u8ba1\uff1b\u4e0d\u91cd\u65b0\u8bad\u7ec3\u3001\u63a8\u7406\u6216\u66f4\u6539\u9009\u62e9\u89c4\u5219\u3002",
        fontsize=10.5, color="#566368")
    fig.savefig(output / "rank_distribution.pdf")
    fig.savefig(output / "rank_distribution.png", dpi=170)
    plt.close(fig)
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    print(f"PDF: {output / 'rank_distribution.pdf'}")


if __name__ == "__main__":
    main()
