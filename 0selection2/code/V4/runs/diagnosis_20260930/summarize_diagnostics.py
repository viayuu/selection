"""Summarize measured diagnostics; never select a checkpoint using test results."""
import json
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from code.V4.V4Model import make_selector
from code.V4.diagnose_training import RUNS
from code.unified_selector.registry import PROBLEMS

ROOT = Path(__file__).resolve().parent


def read(name):
    return json.loads((ROOT / name).read_text())


def main():
    metrics = read("checkpoint_metrics.json")
    probes = read("inference_probes.json")
    components = read("gradient_loss_components.json")
    summary = dict(checkpoints={n: {s: r["macro"] for s, r in rows.items()} for n, rows in metrics.items()})
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    names = list(metrics)
    x = np.arange(len(names))
    for ax, key in zip(axes, ("top1", "ce")):
        for i, split in enumerate(("train", "val")):
            ax.bar(x + (i - .5) * .34, [metrics[n][split]["macro"][key] for n in names], .34, label=split)
        ax.set_xticks(x, names); ax.set_title("Full split, eval mode: " + key); ax.legend()
    fig.tight_layout(); fig.savefig(ROOT / "train_val_checkpoints.png", dpi=160); plt.close(fig)
    summary["inference_probes"] = {}
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    for name, rows in probes.items():
        summary["inference_probes"][name] = {key: {m: float(np.mean([v[key][m] for v in rows.values()]))
            for m in ("top1", "mean_cost", "ce")} for key in next(iter(rows.values()))}
        xs = (0., .25, .5, .75, 1.)
        for ax, key in zip(axes, ("top1", "mean_cost")):
            ax.plot(xs, [summary["inference_probes"][name][f"unbalance_{a}"][key] for a in xs], "o-", label=name)
            ax.set(xlabel="alpha: logits - alpha * log(class_weight)", ylabel=key); ax.legend(); ax.grid(alpha=.25)
    fig.suptitle("Validation-only diagnostic, not a retrained ablation")
    fig.tight_layout(); fig.savefig(ROOT / "class_weight_correction.png", dpi=160); plt.close(fig)
    summary["loss_gradients"] = {}
    for name, rows in components.items():
        summary["loss_gradients"][name] = {}
        for term, weight in [("ce", .35), ("pair", .30), ("topk_ce", .08), ("risk", .02)]:
            values = [v[term] for row in rows.values() for v in row]
            summary["loss_gradients"][name][term] = dict(
                weighted_grad_norm=float(np.mean([v["grad_norm"] for v in values])) * weight,
                mean_cosine_ce=float(np.mean([v["cosine_ce"] for v in values])))
    summary["feature_norms"] = {}
    for name in ("R32e", "R33a"):
        ckpt = torch.load(ROOT.parent / RUNS[name] / "best.pt", map_location="cpu", weights_only=False)
        model = make_selector(ckpt["args"]["model_params"]).eval(); model.load_state_dict(ckpt["model"])
        enc = model.solver_encoder
        weight = getattr(enc, "weight", getattr(enc, "solver_feature_weight", 1.))
        with torch.no_grad():
            f = weight * enc.feature_mlp(enc.solver_features)
            ids = enc.solver_emb.weight
            summary["feature_norms"][name] = dict(feature_norm=f.norm(dim=1).mean().item(),
                id_norm=ids.norm(dim=1).mean().item(),
                centered_feature_norm=(f-f.mean(0)).norm(dim=1).mean().item(),
                centered_id_norm=(ids-ids.mean(0)).norm(dim=1).mean().item())
    floor_stats = {}
    for problem in PROBLEMS:
        with open(Path("data") / f"{problem}train/raw_label.pkl", "rb") as f:
            labels = pickle.load(f)
        costs = np.asarray([labels[str(i)]["cost"] for i in range(len(labels))])
        mask = np.zeros_like(costs, dtype=bool)
        np.put_along_axis(mask, costs.argsort(1)[:, :3], True, axis=1)
        difference = costs[:, None, :] - costs[:, :, None]
        valid = mask[:, :, None] & ~mask[:, None, :] & (difference > 0)
        weight = np.abs(difference) / np.abs(costs).mean(1)[:, None, None]
        floor_stats[problem] = float((weight[valid] <= .25).mean())
    summary["top3_vs_rest_pair_floor_fraction"] = floor_stats
    overfit = read("overfit_probe.json")
    summary["overfit"] = {p: r["history"] for p, r in overfit.items()}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for p, r in overfit.items():
        for ax, key in zip(axes, ("train_top1", "loss")):
            ax.plot([h["step"] for h in r["history"]], [h[key] for h in r["history"]], "o-", label=p)
            ax.set(xlabel="Single-problem updates on fixed 64 train samples", ylabel=key); ax.legend()
    fig.suptitle("Capacity check only: no held-out accuracy claim")
    fig.tight_layout(); fig.savefig(ROOT / "overfit64.png", dpi=160); plt.close(fig)
    (ROOT / "summary_numbers.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
