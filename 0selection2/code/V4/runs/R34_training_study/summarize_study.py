"""Plot the preregistered recipes and compare their held-out predictions."""
import json
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from code.unified_selector.registry import DATA_ROOT, PROBLEMS

ROOT = Path(__file__).resolve().parent
MODES = ("original", "ce", "winner_cost")


def dump(path, obj):
    path.write_text(json.dumps(obj, indent=2) + "\n")


def main():
    selection = json.loads((ROOT / "validation_selection.json").read_text())
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    problem_fig, problem_axes = plt.subplots(6, 3, figsize=(16, 22))
    histories = {}
    for mode in MODES:
        run = ROOT.parent / f"R34_{mode}_seed2_4090"
        history = json.loads((run / "history.json").read_text())
        histories[mode] = history
        for row, source in enumerate(("val", "train_eval")):
            points = [r for r in history if source in r]
            for col, metric in enumerate(("macro_top1", "macro_ce", "macro_vs_sbs_pct")):
                axes[row, col].plot([r["updates"] for r in points], [r[source][metric] for r in points], label=mode)
                axes[row, col].set_title(f"{source}: {metric}")
        evaluations = [json.loads((run / f"eval_epoch{r['epoch']}.json").read_text())["per_problem"] for r in history]
        for ax, problem in zip(problem_axes.flat, PROBLEMS):
            ax.plot([r["updates"] for r in history], [r[problem]["top1"] for r in evaluations], label=mode)
            ax.set_title(problem)
    for axes_group in (axes, problem_axes):
        for ax in axes_group.flat:
            ax.set_xlabel("Successful optimizer updates")
            ax.grid(alpha=.25)
            ax.legend()
    fig.tight_layout(); fig.savefig(ROOT / "recipe_curves.png", dpi=150); plt.close(fig)
    problem_fig.tight_layout(); problem_fig.savefig(ROOT / "per_problem_val_curves.png", dpi=150); plt.close(problem_fig)

    selected_name = "R34_" + selection["selected"]["mode"]
    with np.load(ROOT / f"{selected_name}_test_logits.npz") as arrays:
        selected = {p: arrays[p].argmax(1) for p in PROBLEMS}
    paired = {}
    for reference in ("R32e", "R33a"):
        rows = {}
        with np.load(ROOT / f"{reference}_test_logits.npz") as arrays:
            for p in PROBLEMS:
                with (DATA_ROOT / f"{p}test/raw_label.pkl").open("rb") as f:
                    labels = pickle.load(f)
                costs = np.asarray([labels[str(i)]["cost"] for i in range(len(labels))], dtype=np.float64)
                truth = np.asarray([labels[str(i)]["ind"] for i in range(len(labels))])
                old, new = arrays[p].argmax(1), selected[p]
                old_ok, new_ok = old == truth, new == truth
                idx = np.arange(len(truth))
                rows[p] = dict(n=len(truth), corrected=int((~old_ok & new_ok).sum()),
                               harmed=int((old_ok & ~new_ok).sum()),
                               top1_delta_pp=float((new_ok.mean() - old_ok.mean()) * 100),
                               mean_cost_delta=float((costs[idx, new] - costs[idx, old]).mean()),
                               changed=int((new != old).sum()))
        paired[reference] = dict(per_problem=rows,
            corrected=sum(r["corrected"] for r in rows.values()), harmed=sum(r["harmed"] for r in rows.values()),
            macro_top1_delta_pp=float(np.mean([r["top1_delta_pp"] for r in rows.values()])),
            macro_mean_cost_delta=float(np.mean([r["mean_cost_delta"] for r in rows.values()])))
    dump(ROOT / "paired_comparison.json", paired)
    dump(ROOT / "training_summary.json", {
        mode: dict(epochs=len(h), updates=h[-1]["updates"],
                   skipped=sum(r["train"]["skipped_updates"] for r in h),
                   final_lr=h[-1]["lr"], final_train_eval=h[-1].get("train_eval"),
                   best_validation_top1=max(r["val"]["macro_top1"] for r in h))
        for mode, h in histories.items()})
    print("SUMMARY COMPLETE", selected_name, paired)


if __name__ == "__main__":
    main()
