"""Generate run-local tables and plots from saved evaluations and true labels."""
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from code.unified_selector.analyze import fig_top1_vs_methods, fig_mean_cost_vs_methods, fig_arm_distribution
from code.unified_selector.data import UnifiedProblemDataset
from code.unified_selector.registry import PROBLEMS, POOLS
from code.V4.evaluate import table_line


ROOT = Path(__file__).resolve().parent
OLD = ROOT.parent / "R32d_g0main_seqtopk_seed2_gpu0_b448"


def enrich(payload, split):
    rows = {}
    for p, r in payload["per_problem"].items():
        ds = UnifiedProblemDataset(p, split)
        costs = torch.tensor([ds.labels[str(i)]["cost"][:ds.K_p] for i in range(ds.base_N)], dtype=torch.float32)
        rank = costs.argsort(dim=1)
        counts = torch.bincount(rank[:, 0], minlength=ds.K_p).numpy() / len(costs)
        means = costs.mean(dim=0).numpy()
        pool = list(POOLS[p])
        picks = np.asarray(r["pick_dist"])
        hidden = (picks == 0) & (counts > 0)
        sbs_idx = int(means.argmin())
        sbs = float(means[sbs_idx])
        vbs = float(costs.min(dim=1).values.mean())
        assert np.isclose(sbs, r["sbs"], atol=1e-4), (p, sbs, r["sbs"])
        assert np.isclose(vbs, r["vbs"], atol=1e-4), (p, vbs, r["vbs"])
        rows[p] = dict(r, problem=p, pool_names=pool, sbs_name=pool[sbs_idx],
                       sbs_cost=r["sbs"], oracle_cost=r["vbs"], vbs_mean=r["vbs"],
                       vs_oracle_pct=(r["mean_cost"] / r["vbs"] - 1) * 100,
                       method_top1=dict(zip(pool, counts.tolist())), method_mean=dict(zip(pool, means.tolist())),
                       arm_distribution=dict(zip(pool, picks.tolist())),
                       sbs_topk=[float((rank[:, :k] == sbs_idx).any(dim=1).float().mean()) for k in (1, 2, 3)],
                       final_arm_coverage=float((picks > 0).mean()), zero_pick_count=int((picks == 0).sum()),
                       hidden_winner_mass=float(counts[hidden].sum()), hidden_winner_count=int(hidden.sum()),
                       hidden_winners=[pool[i] for i in np.where(hidden)[0]],
                       pick_oracle_ratio={pool[i]: float(picks[i] / counts[i]) if counts[i] else None for i in range(len(pool))},
                       beat_top1=[pool[i] for i in range(len(pool)) if r["top1"] > counts[i]],
                       lose_top1=[pool[i] for i in range(len(pool)) if r["top1"] <= counts[i]],
                       beat_mean_cost=[pool[i] for i in range(len(pool)) if r["mean_cost"] < means[i]],
                       lose_mean_cost=[pool[i] for i in range(len(pool)) if r["mean_cost"] >= means[i]])
    return rows


def macro(rows):
    keys = ("top1", "top2", "top3", "mean_cost", "sbs_cost", "oracle_cost", "vs_sbs_pct",
            "vs_oracle_pct", "hidden_winner_mass", "final_arm_coverage")
    return dict(problem="ALL", **{k: float(np.mean([r[k] for r in rows.values()])) for k in keys})


def main():
    test = json.loads((ROOT / "test_best.json").read_text())
    epoch = test["best_epoch"]
    comparisons = []
    for title, path in [("R32d best (val top1-safe)", OLD / "test_best.json"),
                        ("R32d historical best_top3_safe", OLD / "test_best_top3_safe.json"),
                        ("R32e solver features (val top1-safe)", ROOT / "test_best.json")]:
        payload = json.loads(path.read_text())
        rows = enrich(payload, "test")
        comparisons.append(dict(name=title, epoch=payload["best_epoch"], **macro(rows)))
    lines = ["# R32e Solver Features: Full Training and Test", "",
             "30 epochs; seed 2; batch 640; RTX 4090; solver feature weight 0.3.",
             "Primary checkpoint: `best.pt`, selected only by validation top1-safe score.",
             f"Selected epoch: {epoch} (zero-based), completed epoch {epoch + 1}.",
             "18 problems: 10,000 training, 1,000 validation and 1,000 test instances each. No LIB evaluation.",
             "Historical comparisons are descriptive, not a controlled feature ablation: batch/runtime changed.",
             "ALL is the unweighted mean across problems. Percentage columns average per-problem percentages,",
             "not ratios of the ALL mean costs. SBS is the best fixed solver on the reported split (benchmark reference).", "",
             "## Historical Comparison", "",
             "| Model | Epoch (0-based) | top1 | top2 | top3 | mean cost | vs SBS | hidden mass | arm coverage |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in comparisons:
        lines.append(f"| {row['name']} | {row['epoch']} | {row['top1']:.4f} | {row['top2']:.4f} | {row['top3']:.4f} | {row['mean_cost']:.4f} | {row['vs_sbs_pct']:+.3f}% | {row['hidden_winner_mass']:.4f} | {row['final_arm_coverage']:.4f} |")
    for split, filename in [("val", f"eval_epoch{epoch}.json"), ("test", "test_best.json")]:
        rows = enrich(json.loads((ROOT / filename).read_text()), split)
        summary = macro(rows)
        (ROOT / f"analysis_{split}.json").write_text(json.dumps({"all": summary, "per_problem": rows}, indent=2))
        lines += ["", f"## {split.upper()} Results", "",
                  "| Problem | top1 | top2 | top3 | mean_cost | Gap | SBS cost (vs_sbs) | Oracle (vs_Oracle) |",
                  "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |", table_line(summary, True)]
        lines += [table_line(rows[p]) for p in PROBLEMS]
        lines += ["", f"### {split.upper()} Arm Distribution", "",
                  "| Problem | hidden mass | zero picks / pool size | picks (all methods) |",
                  "| --- | ---: | ---: | --- |"]
        for p, r in rows.items():
            picks = "; ".join(f"{s}: {v:.1%}" for s, v in r["arm_distribution"].items())
            lines.append(f"| {p} | {r['hidden_winner_mass']:.4f} | {r['zero_pick_count']}/{len(r['pool_names'])} | {picks} |")
        for name, fn in [("top1_vs_single_methods", fig_top1_vs_methods),
                         ("mean_cost_vs_single_methods", fig_mean_cost_vs_methods),
                         ("arm_distribution", fig_arm_distribution)]:
            fn(rows, ROOT / f"{split}_{name}.png")
        lines += ["", f"Detailed per-method comparisons, SBS top1/2/3, pick/oracle ratios: `analysis_{split}.json`."]
    (ROOT / "results_summary.md").write_text("\n".join(lines) + "\n")
    (ROOT / "comparison.json").write_text(json.dumps(comparisons, indent=2))
    log = (ROOT / "train.log").read_text()
    pairs = re.findall(r"ep\d+ step(\d+) loss=([\d.]+)", log)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot([int(s) for s, v in pairs], [float(v) for s, v in pairs])
    axes[0].set(xlabel="Step", ylabel="Running epoch loss")
    evaluations = [json.loads((ROOT / f"eval_epoch{e}.json").read_text())["macro"] for e in range(30)]
    axes[1].plot(range(1, 31), [r["macro_top1"] for r in evaluations], label="val top1")
    axes[1].plot(range(1, 31), [r["macro_top3"] for r in evaluations], label="val top3")
    axes[1].set(xlabel="Completed epoch", ylabel="Accuracy")
    axes[1].legend(); fig.tight_layout(); fig.savefig(ROOT / "loss_validation_curve.png", dpi=140)
    print(json.dumps(comparisons, indent=2))


if __name__ == "__main__":
    main()
