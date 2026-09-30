"""Read-only checkpoint diagnostics; writes reports without updating model weights."""

import argparse
import json
import re
from pathlib import Path
from types import SimpleNamespace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from code.unified_selector.data import UnifiedProblemDataset, collate_single_problem
from code.unified_selector.registry import PROBLEMS, POOLS
from .evaluate import load_model
from .tensor_loader import TensorBatchLoader
from .train import (configure_torch, selector_loss, top_focused_pair_loss,
                    sequential_topk_ce_loss, risk_loss, build_winner_weights)


RUNS = dict(R32d="R32d_g0main_seqtopk_seed2_gpu0_b448",
            R32e="R32e_solver_features_seed2_4090", R33a="R33a_dualstream_seed2_4090")
ROOT = Path(__file__).resolve().parent / "runs"


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def curves():
    summary = {}
    fig_all, ax_all = plt.subplots(1, 2, figsize=(12, 4))
    for name, folder in RUNS.items():
        run = ROOT / folder
        args = json.loads((run / "args.json").read_text())
        logs = {}
        for line in (run / "train.log").read_text().splitlines():
            match = re.match(r"ep(\d+) step(\d+) (.*)", line)
            if match:
                logs[int(match[1])] = dict(epoch=int(match[1]) + 1, step=int(match[2]),
                    **{k: float(v) for k, v in re.findall(r"(\w+)=([+-]?[\d.]+)", match[3])})
        history = []
        for path in sorted(run.glob("eval_epoch*.json"), key=lambda p: int(re.search(r"epoch(\d+)", p.name)[1])):
            e = int(re.search(r"epoch(\d+)", path.name)[1])
            data = json.loads(path.read_text())
            history.append(dict(epoch=e + 1, steps=(e + 1) * (10000 // args["batch_per_problem"]) * 18,
                                train=logs.get(e, {}), **data))
        save_json(run / "diagnostic_history.json", history)
        fig, axes = plt.subplots(2, 3, figsize=(16, 8))
        epochs = [h["epoch"] for h in history]
        for key in ("loss", "ce", "pair", "risk"):
            axes[0, 0].plot(epochs, [h["train"].get(key, np.nan) for h in history], label=key)
        axes[0, 0].set_title("Train: last logged cumulative epoch means")
        axes[0, 0].legend()
        for k in (1, 2, 3):
            axes[0, 1].plot(epochs, [h["macro"][f"macro_top{k}"] for h in history], label=f"top{k}")
        axes[0, 1].set_title("Validation ranking"); axes[0, 1].legend()
        axes[0, 2].plot(epochs, [h["macro"]["macro_vs_sbs_pct"] for h in history])
        axes[0, 2].set_title("Validation mean relative cost vs SBS (%)")
        for family, ps in [("TSP", ["TSP"]), ("CVRP", ["CVRP"]), ("ATSP", ["ATSP"]), ("MVRP", PROBLEMS[3:])]:
            axes[1, 0].plot(epochs, [np.mean([h["per_problem"][p]["top1"] for p in ps]) for h in history], label=family)
        axes[1, 0].set_title("Validation top1 by family"); axes[1, 0].legend()
        axes[1, 1].plot(epochs, [args["lr"]] * len(epochs)); axes[1, 1].set_title("Learning rate (constant; no scheduler)")
        axes[1, 2].plot([h["steps"] for h in history], [h["macro"]["macro_top1"] for h in history])
        axes[1, 2].set_title("Validation top1 vs optimizer updates"); axes[1, 2].set_xlabel("Updates")
        for ax in axes.flat:
            ax.grid(alpha=.25)
            if ax is not axes[1, 2]:
                ax.set_xlabel("Epoch (1-based)")
        fig.suptitle(name + ": no historical validation loss was recorded")
        fig.tight_layout(); fig.savefig(run / "diagnostic_training_curves.png", dpi=160); plt.close(fig)
        fig, axes = plt.subplots(3, 6, figsize=(21, 9), sharex=True)
        for p, ax in zip(PROBLEMS, axes.flat):
            ax.plot(epochs, [h["per_problem"][p]["top1"] for h in history]); ax.set_title(p); ax.grid(alpha=.25)
        fig.supxlabel("Epoch (1-based)"); fig.supylabel("Validation top1")
        fig.tight_layout(); fig.savefig(run / "diagnostic_per_problem_curves.png", dpi=150); plt.close(fig)
        for ax, xs in zip(ax_all, [epochs, [h["steps"] for h in history]]):
            ax.plot(xs, [h["macro"]["macro_top1"] for h in history], label=name)
        summary[name] = dict(batch=args["batch_per_problem"], updates=history[-1]["steps"],
            best_epoch=1 + json.loads((run / "test_best.json").read_text())["best_epoch"],
            last5_top1=np.mean([h["macro"]["macro_top1"] for h in history[-5:]]),
            previous5_top1=np.mean([h["macro"]["macro_top1"] for h in history[-10:-5]]))
    for ax, label in zip(ax_all, ["Epoch", "Optimizer updates"]):
        ax.set(xlabel=label, ylabel="Validation macro top1"); ax.legend(); ax.grid(alpha=.25)
    fig_all.tight_layout()
    return summary, fig_all


def metric(score, costs):
    rank = score.argsort(dim=1, descending=True)
    best = costs.argmin(dim=1)
    chosen = costs.gather(1, rank[:, :1]).squeeze(1)
    oracle = costs.min(1).values
    gap = (chosen - oracle) / oracle.abs().clamp_min(1e-9) * 100
    true_gap = (costs.sort(1).values[:, 1] - oracle) / oracle.abs().clamp_min(1e-9) * 100
    result = dict(n=len(costs), ce=F.cross_entropy(score, best).item(),
                  mean_cost=chosen.mean().item(), mean_regret_pct=gap.mean().item(),
                  top1_tie_aware=chosen.eq(oracle).float().mean().item(),
                  within_0_1pct=(gap <= .1).float().mean().item())
    for k in (1, 2, 3):
        result[f"top{k}"] = rank[:, :k].eq(best[:, None]).any(1).float().mean().item()
    for threshold in (.1, .5, 1.):
        selected = true_gap > threshold
        result[f"decisive_{threshold}_n"] = int(selected.sum())
        result[f"decisive_{threshold}_top1"] = rank[selected, 0].eq(best[selected]).float().mean().item() if selected.any() else None
    return result


def macro(rows):
    keys = ("top1", "top2", "top3", "ce", "mean_cost", "mean_regret_pct", "top1_tie_aware", "within_0_1pct")
    return {k: float(np.mean([r[k] for r in rows.values()])) for k in keys}


@torch.no_grad()
def predict(model, batch, batch_size, feature_off=False):
    enc = model.solver_encoder
    attr = "weight" if hasattr(enc, "weight") else "solver_feature_weight"
    weight = getattr(enc, attr)
    if feature_off:
        setattr(enc, attr, 0.)
    scores = [model(b)["logits"].float().cpu() for b in TensorBatchLoader(batch, batch_size, False, False)]
    setattr(enc, attr, weight)
    return torch.cat(scores)


def gradient_probe(model, batch, config, weights, sbs):
    params = [p for n, p in model.named_parameters() if n.startswith("instance_encoder.layers.")]
    out = model(batch)
    costs = batch["costs"]
    args = SimpleNamespace(**config)
    total, _ = selector_loss(out, costs, weights, args, sbs_idx=sbs)
    winner = costs.argmin(1)
    terms = dict(total=total, ce=F.cross_entropy(out["logits"], winner, weight=weights),
                 pair=top_focused_pair_loss(out["logits"], costs, sbs),
                 topk_ce=sequential_topk_ce_loss(out["logits"], costs, [1., .4, .2]),
                 risk=risk_loss(out["logits"], costs))
    gradients = {}
    for name, term in terms.items():
        g = torch.autograd.grad(term, params, retain_graph=True, allow_unused=True)
        gradients[name] = torch.cat([(torch.zeros_like(p) if v is None else v).flatten() for p, v in zip(params, g)]).detach().cpu()
    ref = gradients["ce"].double()
    stats = {k: dict(value=v.item(), grad_norm=gradients[k].norm().item(),
                    cosine_ce=F.cosine_similarity(gradients[k].double(), ref, dim=0).item()) for k, v in terms.items()}
    return gradients["total"], stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=ROOT / "diagnosis_20260930")
    parser.add_argument("--batch-size", type=int, default=160)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1); torch.manual_seed(20260930); configure_torch()
    history, fig = curves(); fig.savefig(args.out / "comparison_curves.png", dpi=160); plt.close(fig)
    save_json(args.out / "history_summary.json", history)
    models = {}
    for name in ("R32e", "R33a"):
        for tag in ("best", "last"):
            model, ckpt = load_model(ROOT / RUNS[name] / f"{tag}.pt", "cuda:0")
            models[f"{name}/{tag}"] = (model, ckpt["args"])
    results = {name: {s: {} for s in ("train", "val")} for name in models}
    probes = {name: {} for name in ("R32e", "R33a")}
    label_stats, predictions = {}, {}
    grads = {name: [] for name in probes}
    grad_stats = {name: {} for name in probes}
    weights = build_winner_weights(PROBLEMS)
    for problem in PROBLEMS:
        label_stats[problem] = {}
        train_costs = None
        for split in ("train", "val"):
            ds = UnifiedProblemDataset(problem, split)
            samples = [ds[i] for i in range(len(ds))]
            batch = collate_single_problem(samples)
            costs = batch["costs"]
            label_stats[problem][split] = dict(n=len(costs), pool=POOLS[problem],
                winner_count=torch.bincount(costs.argmin(1), minlength=costs.size(1)).tolist(),
                exact_tie_fraction=(costs.eq(costs.min(1).values[:, None]).sum(1) > 1).float().mean().item(),
                argmin_argsort_disagree=costs.argmin(1).ne(costs.argsort(1)[:, 0]).sum().item(),
                margin_percentiles=np.percentile(((costs.sort(1).values[:, 1]-costs.min(1).values)/costs.min(1).values*100).numpy(), [0,10,25,50,75,90,100]).tolist())
            if split == "train":
                train_costs = costs
            for k in (.02, .1, .5, 1.):
                gap = (costs.sort(1).values[:, 1] - costs.min(1).values) / costs.min(1).values * 100
                label_stats[problem][split][f"margin_le_{k}_pct"] = float((gap <= k).float().mean())
            batch = {k: v.cuda() if torch.is_tensor(v) else v for k, v in batch.items()}
            for name, (model, config) in models.items():
                scores = predict(model, batch, args.batch_size)
                results[name][split][problem] = metric(scores, costs)
                if name.endswith("best") and split == "val":
                    short = name.split("/")[0]
                    predictions[f"{short}_{problem}"] = scores.numpy()
                    probes[short][problem] = {}
                    for alpha in (0., .25, .5, .75, 1.):
                        corrected = scores - alpha * weights[problem].log()
                        probes[short][problem][f"unbalance_{alpha}"] = metric(corrected, costs)
                    probes[short][problem]["feature_off"] = metric(predict(model, batch, args.batch_size, True), costs)
                    perm = torch.randperm(len(costs), generator=torch.Generator().manual_seed(19))
                    probes[short][problem]["instance_shuffle"] = metric(scores[perm], costs)
                    freq = torch.bincount(train_costs.argmin(1), minlength=costs.size(1)).float()
                    prior = (freq + 1).log().expand_as(scores)
                    probes[short][problem]["train_prior_only"] = metric(prior, costs)
                    # Three train batches, eval-mode gradients: no optimizer step or dropout noise.
                    group = []
                    for rep in range(3):
                        ix = np.random.default_rng(700 + rep).choice(len(train_samples), 32, replace=False)
                        small = collate_single_problem([train_samples[i] for i in ix])
                        small = {k: v.cuda() if torch.is_tensor(v) else v for k, v in small.items()}
                        vector, stats = gradient_probe(model, small, config, weights[problem].cuda(), int(train_costs.mean(0).argmin()))
                        group.append(vector); grad_stats[short].setdefault(problem, []).append(stats)
                    grads[short].append(torch.stack(group))
                print(name, problem, split, results[name][split][problem]["top1"], flush=True)
            if split == "train":
                train_samples = samples
            del batch, samples, ds
        save_json(args.out / "checkpoint_metrics.partial.json", results)
    save_json(args.out / "checkpoint_metrics.json", {k: {s: dict(macro=macro(rows), per_problem=rows) for s, rows in split.items()} for k, split in results.items()})
    save_json(args.out / "label_statistics.json", label_stats)
    save_json(args.out / "inference_probes.json", probes)
    save_json(args.out / "gradient_loss_components.json", grad_stats)
    np.savez_compressed(args.out / "val_logits.npz", **predictions)
    for name, vectors in grads.items():
        tensor = torch.stack(vectors)
        cosines = []
        for rep in range(tensor.size(1)):
            normalized = F.normalize(tensor[:, rep].double(), dim=1)
            cosines.append(normalized @ normalized.T)
        cos = torch.stack(cosines).mean(0).numpy()
        save_json(args.out / f"{name}_task_gradient_cosine.json", dict(problems=PROBLEMS, cosine=cos.tolist(),
                  negative_fraction=float((cos[np.triu_indices(len(PROBLEMS), 1)] < 0).mean())))
        fig, ax = plt.subplots(figsize=(10, 8)); im = ax.imshow(cos, vmin=-1, vmax=1, cmap="coolwarm")
        ax.set_xticks(range(18), PROBLEMS, rotation=90); ax.set_yticks(range(18), PROBLEMS)
        ax.set_title(name + " shared encoder gradient cosine; mean of 3 batches")
        fig.colorbar(im, ax=ax); fig.tight_layout(); fig.savefig(args.out / f"{name}_gradient_conflict.png", dpi=140); plt.close(fig)
    print("DIAGNOSTICS COMPLETE", flush=True)


if __name__ == "__main__":
    main()
