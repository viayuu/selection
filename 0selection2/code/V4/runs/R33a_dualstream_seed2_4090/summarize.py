"""Run-local report using saved predictions and the unchanged dataset labels."""
import json
import pickle
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from code.V4.evaluate import table_line
from code.unified_selector.registry import DATA_ROOT, POOLS, PROBLEMS
from code.unified_selector.analyze import fig_top1_vs_methods, fig_mean_cost_vs_methods, fig_arm_distribution


ROOT = Path(__file__).resolve().parent


def enrich(payload, split):
    rows = {}
    for problem, result in payload["per_problem"].items():
        pool = POOLS[problem]
        with (DATA_ROOT / f"{problem}{split}" / "raw_label.pkl").open("rb") as f:
            labels = pickle.load(f)
        costs = torch.tensor(np.asarray([labels[str(i)]["cost"][:len(pool)] for i in range(len(labels))], dtype=np.float32))
        rank = costs.argsort(dim=1)
        freq = torch.bincount(rank[:, 0], minlength=len(pool)).numpy() / len(costs)
        means = costs.mean(0).numpy()
        sbs = int(means.argmin())
        assert np.isclose(means[sbs], result["sbs"], atol=1e-4)
        assert np.isclose(costs.min(1).values.mean(), result["vbs"], atol=1e-4)
        rows[problem] = dict(
            result, problem=problem, pool_names=pool, sbs_name=pool[sbs],
            sbs_cost=result["sbs"], oracle_cost=result["vbs"], vbs_mean=result["vbs"],
            vs_oracle_pct=(result["mean_cost"] / result["vbs"] - 1) * 100,
            method_top1=dict(zip(pool, freq.tolist())), method_mean=dict(zip(pool, means.tolist())),
            arm_distribution=dict(zip(pool, result["pick_dist"])),
            sbs_topk=[float((rank[:, :k] == sbs).any(1).float().mean()) for k in (1, 2, 3)],
            beat_top1=[s for i, s in enumerate(pool) if result["top1"] > freq[i]],
            lose_top1=[s for i, s in enumerate(pool) if result["top1"] <= freq[i]],
            beat_mean_cost=[s for i, s in enumerate(pool) if result["mean_cost"] < means[i]],
            lose_mean_cost=[s for i, s in enumerate(pool) if result["mean_cost"] >= means[i]],
        )
    return rows


def macro(rows):
    keys = ("top1", "top2", "top3", "mean_cost", "sbs_cost", "oracle_cost", "vs_sbs_pct", "vs_oracle_pct")
    return dict(problem="ALL", **{k: float(np.mean([v[k] for v in rows.values()])) for k in keys})


def main():
    payload = json.loads((ROOT / "test_best.json").read_text())
    epoch = payload["best_epoch"]
    comparisons = []
    for name, folder in [("R32d", "R32d_g0main_seqtopk_seed2_gpu0_b448"),
                         ("R32e solver features", "R32e_solver_features_seed2_4090"),
                         ("R33a dual-stream", ROOT.name)]:
        data = json.loads((ROOT.parent / folder / "test_best.json").read_text())
        comparisons.append(dict(name=name, epoch=data["best_epoch"], **macro(enrich(data, "test"))))
    lines = ["# R33a Dual-Stream Selector", "",
             "## Architecture and Protocol", "",
             "Existing 4-layer node self-attention -> 2 synchronous bidirectional cross-attention layers.",
             "Solver initialization: LayerNorm(ID + feature MLP); the fixed 34-feature table is unchanged.",
             "Updated nodes -> masked mean/max pooling -> instance query over updated solvers.",
             "Shared scorer [h, context, solver] -> logits. No mixed-score external D and no label input.",
             "No legacy pre/gap/support scores or auxiliary heads in this mode.",
             "Training: from scratch, seed 2, 30 epochs, batch 640, AdamW lr=2e-4, wd=1e-4, FP16, dropout=0.1.",
             "Loss: 0.35 winner-balanced CE + 0.30 top-focused pairwise + 0.08 sequential top-k CE + 0.02 expected cost risk.",
             "18 problems, each with 10,000 train / 1,000 val / 1,000 test. No LIB or unseen-problem zero-shot evaluation.",
             "Each training epoch uses 9,600 randomly shuffled samples per problem (existing drop_last behavior).",
             f"Primary checkpoint: best.pt, epoch {epoch} (zero-based), selected exclusively on validation top1-safe score.",
             "Test never selects the checkpoint. ALL percentages are macro means of per-problem percentages, not ratios of aggregate costs.",
             "SBS is the best fixed method on the reported split, used as a benchmark reference only.",
             "This compares complete methods, not an isolated architectural ablation: the old auxiliary heads were removed and feature weight changed to 1.0.", "",
             "## Test Comparison", "",
             "| Model | Epoch (0-based) | top1 | top2 | top3 | mean cost | vs SBS | vs Oracle |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for r in comparisons:
        lines.append(f"| {r['name']} | {r['epoch']} | {r['top1']:.4f} | {r['top2']:.4f} | {r['top3']:.4f} | {r['mean_cost']:.4f} | {r['vs_sbs_pct']:+.3f}% | {r['vs_oracle_pct']:+.3f}% |")
    for split, file in [("val", f"eval_epoch{epoch}.json"), ("test", "test_best.json")]:
        rows = enrich(json.loads((ROOT / file).read_text()), split)
        summary = macro(rows)
        (ROOT / f"analysis_{split}.json").write_text(json.dumps(dict(all=summary, per_problem=rows), indent=2))
        lines += ["", f"## {split.upper()} Results", "",
                  "| Problem | top1 | top2 | top3 | mean_cost | Gap | SBS cost (vs_sbs) | Oracle (vs_Oracle) |",
                  "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |", table_line(summary, True)]
        lines += [table_line(rows[p]) for p in PROBLEMS]
        lines += ["", f"### {split.upper()} Arm Distribution", "", "| Problem | Selection frequency |", "| --- | --- |"]
        for p, r in rows.items():
            dist = "; ".join(f"{s}: {v:.1%}" for s, v in r["arm_distribution"].items())
            lines.append(f"| {p} | {dist} |")
        for name, fn in [("top1_vs_single_methods", fig_top1_vs_methods), ("mean_cost_vs_single_methods", fig_mean_cost_vs_methods), ("arm_distribution", fig_arm_distribution)]:
            fn(rows, ROOT / f"{split}_{name}.png")
        lines += ["", f"All single-method comparisons and SBS top1/2/3 are in `analysis_{split}.json`."]
    (ROOT / "results_summary.md").write_text("\n".join(lines) + "\n")
    (ROOT / "comparison.json").write_text(json.dumps(comparisons, indent=2))
    values = re.findall(r"ep\d+ step(\d+) loss=([\d.]+)", (ROOT / "train.log").read_text())
    evals = [json.loads((ROOT / f"eval_epoch{e}.json").read_text())["macro"] for e in range(30)]
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    ax[0].plot([int(s) for s, v in values], [float(v) for s, v in values])
    ax[0].set(xlabel="Step", ylabel="Running epoch loss")
    ax[1].plot(range(1, 31), [m["macro_top1"] for m in evals]); ax[1].set(xlabel="Epoch", ylabel="Validation top1")
    ax[2].plot(range(1, 31), [m["macro_vs_sbs_pct"] for m in evals]); ax[2].set(xlabel="Epoch", ylabel="Validation vs SBS (%)")
    fig.tight_layout(); fig.savefig(ROOT / "training_curves.png", dpi=140)
    print(json.dumps(comparisons, indent=2))


if __name__ == "__main__":
    main()
