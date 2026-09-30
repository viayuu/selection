"""R36: raw-precision TSP error analysis and paired small-subset fitting."""

import argparse
import copy
import csv
import json
import pickle
import sys
import time
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F

from code.unified_selector.registry import DATA_ROOT, GLOBAL_SOLVERS, POOLS
from .V4Model import make_selector
from .multitask_probe import Tee, batch_schedule, dump, file_hash, gather_batch, state_hash
from .tensor_loader import TensorBatchLoader
from .train import make_loader, risk_loss, selector_loss, set_seed, winner_pair_loss


ROOT = Path("code/V4/runs/R36_tsp_learnability")
BASE = Path("code/V4/runs/R34_winner_cost_seed2_4090/best.pt")
PRED_ROOT = Path("code/V4/runs/R35_multitask_probe/joint_seed2")
PROTECTED = ("V4Model.py", "dual_stream.py", "solver_features.py")
GAP_LABELS = ("=0", "(0,0.01%]", "(0.01%,0.1%]", "(0.1%,0.5%]", ">0.5%")


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def raw_tsp(split):
    if split not in ("train", "val"):
        raise ValueError("R36 is restricted to train/val")
    directory = DATA_ROOT / f"TSP{split}"
    with (directory / "raw_label.pkl").open("rb") as stream:
        labels = pickle.load(stream)
    with (directory / "dataset.pkl").open("rb") as stream:
        instances = pickle.load(stream)
    k = len(POOLS["TSP"])
    costs = np.array([labels[str(i)]["cost"][:k] for i in range(len(instances))], dtype=np.float64)
    winner = np.array([int(labels[str(i)]["ind"]) for i in range(len(instances))])
    nodes = np.array([len(x[0]) for x in instances])
    if costs.shape != (len(instances), k) or not np.isfinite(costs).all() or (costs <= 0).any():
        raise ValueError("Invalid raw TSP costs")
    precision = dict(source="raw_label.pkl, directly decoded before float32 conversion",
                     stored_container_dtype=str(np.asarray(labels["0"]["cost"]).dtype),
                     calculation_dtype=str(costs.dtype),
                     values_changed_by_float32=int((costs != costs.astype(np.float32).astype(np.float64)).sum()),
                     native_winner_nonminimal=int((costs[np.arange(len(costs)), winner] != costs.min(1)).sum()))
    return dict(costs=costs, winner=winner, nodes=nodes, instances=instances, precision=precision)


def gap_and_regret(costs, pred):
    ordered = np.sort(costs, axis=1)
    best = ordered[:, 0]
    gap = (ordered[:, 1] - best) / np.abs(best) * 100
    regret = (costs[np.arange(len(costs)), pred] - best) / np.abs(best) * 100
    return gap, regret


def size_boundaries(nodes):
    cuts = np.unique(np.quantile(nodes, [.25, .5, .75]))
    cuts = cuts[cuts < nodes.max()]
    positions = np.searchsorted(np.sort(nodes), cuts, side="right")
    # Discrete sizes can give distinct numerical cuts with the same membership.
    return cuts[np.r_[True, positions[1:] != positions[:-1]]] if len(cuts) else cuts


def size_labels(cuts):
    endpoints = [-np.inf, *cuts, np.inf]
    return [f"({left:g},{right:g}]" for left, right in zip(endpoints[:-1], endpoints[1:])]


def bucket_rows(split, groups, labels, correct, regret, regret_share=False):
    rows = []
    total = regret.sum()
    for i, label in enumerate(labels):
        mask = groups == i
        row = dict(split=split, bucket=label, n=int(mask.sum()),
                   strict_top1=float(correct[mask].mean()) if mask.any() else None,
                   mean_actual_regret_pct=float(regret[mask].mean()) if mask.any() else None)
        if regret_share:
            row["share_of_total_regret"] = float(regret[mask].sum() / total) if total > 0 else None
        rows.append(row)
    return rows


def prediction_metrics(costs, winner, pred):
    selected = costs[np.arange(len(costs)), pred]
    mean_cost = float(selected.mean())
    sbs = float(costs.mean(0).min())
    return dict(n=len(winner), top1=float((pred == winner).mean()), mean_cost=mean_cost,
                sbs=sbs, vs_sbs_pct=(mean_cost / sbs - 1) * 100,
                mean_actual_regret_pct=float(gap_and_regret(costs, pred)[1].mean()))


def analyze_errors(root, data, pred_root):
    cuts = size_boundaries(data["train"]["nodes"])
    labels = size_labels(cuts)
    train_groups = np.searchsorted(cuts, data["train"]["nodes"], side="left")
    rules = []
    for group, label in enumerate(labels):
        counts = np.bincount(data["train"]["winner"][train_groups == group], minlength=len(POOLS["TSP"]))
        if not counts.sum():
            raise ValueError("Empty training size bucket")
        solver = int(counts.argmax())
        rules.append(dict(bucket=label, solver_index=solver, solver=POOLS["TSP"][solver],
                          winner_counts=counts.tolist(), n=int(counts.sum())))
    dump(root / "size_prior_rules.json", dict(boundaries=cuts.tolist(), rules=rules, fitted_on="train only"))
    gap_rows, sizes, prior_rows, summary = [], [], [], {}
    for split, raw in data.items():
        with np.load(pred_root / f"predictions_{split}_u000.npz") as cached:
            indices = cached["indices"]
            if not np.array_equal(indices, np.arange(len(raw["winner"]))):
                raise ValueError("Baseline predictions are not in original instance order")
            np.testing.assert_array_equal(cached["winner"], raw["winner"])
            np.testing.assert_array_equal(cached["pool_ids"], [GLOBAL_SOLVERS.index(s) for s in POOLS["TSP"]])
            pred = cached["logits"].argmax(1)
            np.testing.assert_array_equal(pred, cached["pred"])
        gap, regret = gap_and_regret(raw["costs"], pred)
        correct = pred == raw["winner"]
        groups = np.searchsorted(cuts, raw["nodes"], side="left")
        gap_groups = np.searchsorted([0, .01, .1, .5], gap, side="left")
        gap_rows.extend(bucket_rows(split, gap_groups, GAP_LABELS, correct, regret, True))
        sizes.extend(bucket_rows(split, groups, labels, correct, regret))
        prior_pred = np.array([rules[g]["solver_index"] for g in groups])
        model_metrics = prediction_metrics(raw["costs"], raw["winner"], pred)
        prior_metrics = prediction_metrics(raw["costs"], raw["winner"], prior_pred)
        for name, prediction in (("R34", pred), ("size_prior", prior_pred)):
            for g, label in enumerate(labels):
                mask = groups == g
                if mask.any():
                    prior_rows.append(dict(split=split, model=name, bucket=label,
                                           **prediction_metrics(raw["costs"][mask], raw["winner"][mask], prediction[mask])))
            prior_rows.append(dict(split=split, model=name, bucket="ALL",
                                   **prediction_metrics(raw["costs"], raw["winner"], prediction)))
        wrong = ~correct
        low_regret_errors = int((wrong & (regret <= .1)).sum())
        summary[split] = dict(model=model_metrics, size_prior=prior_metrics, precision=raw["precision"],
                              error_count=int(wrong.sum()), low_regret_error_count=low_regret_errors,
                              error_fraction_regret_le_0_1=low_regret_errors / int(wrong.sum()) if wrong.any() else None,
                              wrong_regret_gt_0_5_count=int((wrong & (regret > .5)).sum()),
                              near_tie_gap_le_0_1_high_regret_gt_0_5_count=int(((gap <= .1) & (regret > .5)).sum()))
        rows = [dict(index=i, nodes=int(raw["nodes"][i]), native_winner=int(raw["winner"][i]), pred=int(pred[i]),
                     strict_correct=bool(correct[i]), winner_gap_pct=float(gap[i]), actual_regret_pct=float(regret[i]),
                     gap_bucket=GAP_LABELS[gap_groups[i]], size_bucket=labels[groups[i]], size_prior_pred=int(prior_pred[i]))
                for i in range(len(pred))]
        write_csv(root / f"instances_{split}.csv", rows)
        print(f"[R36A {split}] top1={model_metrics['top1']:.4f} size_prior={prior_metrics['top1']:.4f} "
              f"wrong={int(wrong.sum())} errors_regret<=0.1%={summary[split]['error_fraction_regret_le_0_1']:.2%}", flush=True)
    write_csv(root / "winner_gap_buckets.csv", gap_rows)
    write_csv(root / "size_buckets.csv", sizes)
    write_csv(root / "size_prior_comparison.csv", prior_rows)
    dump(root / "error_summary.json", summary)
    return summary, cuts


def stratified_indices(winner, groups, count, seed, available=None):
    """Balance available winner classes, cycling size strata within each class."""
    rng = np.random.default_rng(seed)
    available = np.arange(len(winner)) if available is None else np.asarray(available)
    if count > len(available):
        raise ValueError("Subset exceeds available distinct instances")
    queues = {}
    for label in np.unique(winner[available]):
        strata = [list(rng.permutation(available[(winner[available] == label) & (groups[available] == g)]))
                  for g in np.unique(groups[available])]
        queue = []
        while any(strata):
            for stratum in strata:
                if stratum:
                    queue.append(stratum.pop())
        queues[int(label)] = queue
    selected = []
    while len(selected) < count:
        for queue in queues.values():
            if queue and len(selected) < count:
                selected.append(queue.pop(0))
    return rng.permutation(selected)


def fit_args(recipe):
    return SimpleNamespace(loss_mode="ce" if recipe == "ce" else "winner_cost", ce_weight=.35,
                           pair_weight=.10, risk_weight=.02, cost_scale=.01)


def fresh_model(checkpoint, device, lr):
    params = copy.deepcopy(checkpoint["args"]["model_params"])
    params["dropout"] = 0.0
    model = make_selector(params).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    if not all(p.requires_grad for p in model.parameters()):
        raise ValueError("R36 must not freeze parameters")
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.)
    assert not optimizer.state
    return model, optimizer, params


def fit_passed(metrics):
    return metrics["top1"] >= .99 and metrics["ce"] <= .05


@torch.no_grad()
def evaluate_subset(model, loader, indices, raw_costs, args, path):
    model.eval()
    logits = torch.cat([model(batch)["logits"].cpu() for batch in TensorBatchLoader(loader.batch, 32, False, False)])
    winner = loader.batch["ind"].cpu()
    if not torch.isfinite(logits).all():
        raise RuntimeError("Nonfinite evaluation logits")
    loss, parts = selector_loss(dict(logits=logits), loader.batch["costs"].cpu(), None, args, winner=winner)
    pred = logits.argmax(1).numpy()
    result = dict(ce=float(F.cross_entropy(logits, winner)), objective=float(loss),
                  **prediction_metrics(raw_costs, winner.numpy(), pred),
                  parts={k: float(v) for k, v in parts.items()})
    np.savez_compressed(path, indices=indices, logits=logits.numpy(), winner=winner.numpy(), pred=pred,
                        raw_costs=raw_costs, pool_ids=loader.batch["pool_ids"].cpu().numpy())
    return result


def gradient_diagnostics(model, batch):
    model.eval()
    logits = model(batch)["logits"]
    terms = dict(ce=.35 * F.cross_entropy(logits, batch["ind"]),
                 pair=.10 * winner_pair_loss(logits, batch["costs"], batch["ind"], .01),
                 risk=.02 * risk_loss(logits, batch["costs"]) / .01)
    named = list(model.named_parameters())
    gradients = {}
    for key, term in terms.items():
        parts = torch.autograd.grad(term, [p for _, p in named], retain_graph=True, allow_unused=True)
        gradients[key] = torch.cat([g.detach().flatten() if g is not None else torch.zeros_like(p).flatten()
                                    for (_, p), g in zip(named, parts)])
    norms = {k: float(v.norm()) for k, v in gradients.items()}
    def cosine(left, right):
        denominator = left.norm() * right.norm()
        return float(left.dot(right) / denominator) if denominator > 0 else None
    return dict(weighted_losses={k: float(v.detach()) for k, v in terms.items()}, weighted_gradient_norms=norms,
                cosine_ce_pair=cosine(gradients["ce"], gradients["pair"]),
                cosine_ce_risk=cosine(gradients["ce"], gradients["risk"]),
                cosine_ce_auxiliary=cosine(gradients["ce"], gradients["pair"] + gradients["risk"]),
                auxiliary_to_ce_norm_ratio=float((gradients["pair"] + gradients["risk"]).norm() /
                                                gradients["ce"].norm().clamp_min(1e-12)))


@torch.no_grad()
def representation_diagnostics(model, batch):
    model.eval()
    nodes, mask = model.instance_encoder(batch)
    ids = batch["pool_ids"][None].expand(len(nodes), -1)
    solvers = model.solver_encoder(ids)
    solver_mask = torch.ones_like(ids, dtype=torch.bool)
    values = {}
    def measure(name, vector):
        values[name] = dict(mean_sample_std=float(vector.flatten(1).std(0).mean()),
                            mean_sample_norm=float(vector.flatten(1).norm(dim=1).mean()))
    measure("initial_instance_mean", (nodes * mask[..., None]).sum(1) / mask.sum(1, keepdim=True))
    for i, layer in enumerate(model.joint_layers):
        nodes, solvers = layer(nodes, solvers, mask, solver_mask)
        measure(f"joint_{i}_instance_mean", (nodes * mask[..., None]).sum(1) / mask.sum(1, keepdim=True))
        measure(f"joint_{i}_solver", solvers)
    measure("logits", model(batch)["logits"])
    return values


def run_fit(recipe, indices, loader, raw, checkpoint, args):
    directory = args.root / f"{len(indices)}_{recipe}_seed{args.seed}"
    directory.mkdir()
    with (directory / "train.log").open("w") as log:
        with redirect_stdout(Tee(sys.stdout, log)), redirect_stderr(Tee(sys.stderr, log)):
            return _run_fit(recipe, indices, loader, raw, checkpoint, args, directory)


def _run_fit(recipe, indices, full_loader, raw, checkpoint, args, directory):
    set_seed(args.seed)
    model, optimizer, params = fresh_model(checkpoint, args.device, args.lr)
    subset = TensorBatchLoader(gather_batch(full_loader, torch.tensor(indices)), 32, False, False)
    native = subset.batch["ind"].cpu().numpy()
    np.testing.assert_array_equal(native, raw["winner"][indices])
    np.testing.assert_array_equal(subset.lengths.numpy(), raw["nodes"][indices])
    np.testing.assert_array_equal(subset.batch["costs"].cpu().numpy(), raw["costs"][indices].astype(np.float32))
    for j, original in enumerate(indices):
        n = int(raw["nodes"][original])
        torch.testing.assert_close(subset.batch["node"][j, :n, :2].cpu(), raw["instances"][original][0].float(), rtol=0, atol=0)
        assert subset.batch["node_mask"][j].sum().item() == n
    schedule = batch_schedule(len(indices), 32, args.updates, args.seed)
    torch.save(torch.tensor(indices)[schedule], directory / "batch_indices.pt")
    loss_args = fit_args(recipe)
    config = dict(recipe=recipe, seed=args.seed, subset_size=len(indices), lr=args.lr, batch_size=32,
                  dropout=0., weight_decay=0., amp=False, tf32=False, coord_augment=0, optimizer_restored=False,
                  optimizer_initial_state_count=len(optimizer.state), max_updates=args.updates, eval_interval=50,
                  gradient_clip=1.0, loss=vars(loss_args), model_params=params, base_checkpoint=str(args.base),
                  model_initial_hash=state_hash(model.state_dict()), batch_schedule_hash=state_hash(schedule),
                  indices=indices.tolist(), all_parameters_trainable=True,
                  precision_policy="R34 FP32 costs for training; raw FP64 costs for reported cost/regret",
                  pass_rule="Top1 >= 0.99 and unweighted CE <= 0.05 for three consecutive evaluations",
                  wandb_mode="offline", wandb_project="selector")
    dump(directory / "args.json", config)
    start = time.monotonic()
    history, traces, streak = [], [], 0
    first_batch = gather_batch(subset, schedule[0])
    diagnostics = dict(initial_gradients=gradient_diagnostics(model, first_batch),
                       initial_representations=representation_diagnostics(model, first_batch), input_alignment_verified=True)
    initial_parameters = {n: p.detach().clone() for n, p in model.named_parameters()}
    wandb_run = None
    if args.wandb:
        import wandb
        wandb_run = wandb.init(project="selector", entity="yjkds-southern-university-of-science-technology",
                               name="R36_" + directory.name, config=config, dir=str(directory), mode="offline")
    print(f"[start] {directory.name} model_hash={config['model_initial_hash']} fresh_optimizer=True "
          f"lr={args.lr:g} dropout=0 AMP=False", flush=True)

    def evaluate(update):
        nonlocal streak
        record = dict(updates=update, elapsed_seconds=time.monotonic() - start,
                      **evaluate_subset(model, subset, indices, raw["costs"][indices], loss_args,
                                        directory / f"predictions_u{update:04d}.npz"))
        streak = streak + 1 if fit_passed(record) else 0
        record["pass_streak"] = streak
        history.append(record)
        dump(directory / "history.json", history)
        torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(), args=config,
                        updates=update, metrics=record), directory / "last.pt")
        print(f"[eval updates {update:04d}] top1={record['top1']:.4f} ce={record['ce']:.6f} "
              f"mean_cost={record['mean_cost']:.6f} vs_sbs={record['vs_sbs_pct']:+.4f}% "
              f"streak={streak}/3 elapsed={record['elapsed_seconds']:.1f}s", flush=True)
        if wandb_run:
            wandb_run.log({f"subset/{k}": record[k] for k in ("ce", "top1", "mean_cost", "vs_sbs_pct", "objective")}, step=update)

    evaluate(0)
    with (directory / "updates.jsonl").open("w") as trace:
        for update, local_indices in enumerate(schedule, 1):
            model.train()
            batch = gather_batch(subset, local_indices)
            optimizer.zero_grad(set_to_none=True)
            loss, parts = selector_loss(model(batch), batch["costs"], None, loss_args, winner=batch["ind"])
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite fitting objective")
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            stats = dict(updates=update, loss=float(loss.detach()), grad_norm=float(norm),
                         **{k: float(v) for k, v in parts.items()})
            trace.write(json.dumps(stats) + "\n")
            traces.append(stats)
            if update % 50 == 0 or update == args.updates:
                trace.flush()
                evaluate(update)
                if streak >= 3:
                    break
    diagnostics.update(final_gradients=gradient_diagnostics(model, first_batch),
                       final_representations=representation_diagnostics(model, first_batch))
    delta_by_module = {}
    for name, parameter in model.named_parameters():
        group = name.split(".")[0]
        change = float((parameter.detach() - initial_parameters[name]).square().sum())
        delta_by_module[group] = delta_by_module.get(group, 0.) + change
    diagnostics["parameter_delta_l2_by_module"] = {k: v ** .5 for k, v in delta_by_module.items()}
    diagnostics["updates_with_clipped_gradient"] = sum(t["grad_norm"] > 1.0 for t in traces)
    diagnostics["mean_gradient_norm"] = float(np.mean([t["grad_norm"] for t in traces]))
    dump(directory / "diagnostics.json", diagnostics)
    result = dict(name=directory.name, recipe=recipe, n=len(indices), passed=streak >= 3,
                  updates=history[-1]["updates"], final=history[-1],
                  best_ce=min(history, key=lambda h: h["ce"])["ce"],
                  best_top1=max(h["top1"] for h in history), elapsed_seconds=time.monotonic() - start)
    dump(directory / "result.json", result)
    if wandb_run:
        wandb_run.summary.update({"passed": result["passed"], "updates": result["updates"], "best_ce": result["best_ce"]})
        wandb_run.finish()
    print(f"[done] {directory.name} passed={result['passed']} updates={result['updates']} "
          f"top1={history[-1]['top1']:.4f} CE={history[-1]['ce']:.6f}", flush=True)
    del model, optimizer, initial_parameters
    torch.cuda.empty_cache()
    return result


def summarize(root, analysis, results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for metric in ("ce", "top1"):
        fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
        for n, axis in zip((128, 32), axes):
            for result in results:
                if result["n"] != n:
                    continue
                history = json.loads((root / result["name"] / "history.json").read_text())
                axis.plot([r["updates"] for r in history], [r[metric] for r in history], label=result["recipe"].upper())
            axis.axhline(.05 if metric == "ce" else .99, color="gray", linestyle="--", label="pass threshold")
            axis.set(xlabel="Successful updates", ylabel="Unweighted CE" if metric == "ce" else "Strict Top1",
                     title=f"{n} fixed training instances")
            axis.grid(alpha=.2)
            axis.legend()
        fig.savefig(root / f"comparison_{metric}.png", dpi=170)
        plt.close(fig)
    rows = [dict(run=r["name"], n=r["n"], passed=r["passed"], updates=r["updates"],
                 ce=r["final"]["ce"], top1=r["final"]["top1"], mean_cost=r["final"]["mean_cost"],
                 vs_sbs_pct=r["final"]["vs_sbs_pct"], best_ce=r["best_ce"], best_top1=r["best_top1"]) for r in results]
    write_csv(root / "fit_comparison.csv", rows)
    text = ["# R36 TSP learnability", "", "## Scope", "",
            "R34 remains the main baseline. Only TSP train/val were read; no test evaluation or architecture changes.",
            "R36A reuses R35 u000 logits and directly reads raw_label.pkl costs in FP64. Strict Top1 uses native ind.",
            "R36B uses fresh AdamW, LR 2e-4, batch 32, dropout/WD/augmentation/AMP/TF32 disabled; gradient clipping remains R34's 1.0.",
            "Training uses the unchanged R34 FP32 winner-cost recipe; reported cost diagnostics use raw FP64.", "",
            "## R36A", "", "| Split | R34 Top1 | Size-only Top1 | R34 mean regret % | Errors with regret <=0.1% |", "|---|---:|---:|---:|---:|"]
    for split, s in analysis.items():
        text.append(f"| {split} | {s['model']['top1']:.4f} | {s['size_prior']['top1']:.4f} | "
                    f"{s['model']['mean_actual_regret_pct']:.4f} | {s['error_fraction_regret_le_0_1']:.2%} |")
    for filename, title in (("winner_gap_buckets.csv", "Winner-gap buckets"), ("size_buckets.csv", "Size buckets")):
        with (root / filename).open() as stream:
            records = list(csv.DictReader(stream))
        fields = list(records[0])
        text.extend(["", f"### {title}", "", "| " + " | ".join(fields) + " |", "|" + "---|" * len(fields)])
        text.extend("| " + " | ".join(str(r[f]) for f in fields) + " |" for r in records)
    text.extend(["", "## R36B", "", "| Run | Passed | Updates | Final Top1 | Final unweighted CE | Best Top1 | Best CE |",
                 "|---|---|---:|---:|---:|---:|---:|"])
    for r in results:
        text.append(f"| {r['name']} | {r['passed']} | {r['updates']} | {r['final']['top1']:.4f} | "
                    f"{r['final']['ce']:.6f} | {r['best_top1']:.4f} | {r['best_ce']:.6f} |")
    text.extend(["", "![CE](comparison_ce.png)", "", "![Top1](comparison_top1.png)", "",
                 "## Interpretation", "", "The 128-example experiments use identical initial weights and batch schedules.",
                 "32-example runs, if present, start again from R34 rather than from the 128-example endpoints.",
                 "A failed fit is not evidence of label noise. A successful fit establishes memorization ability, not generalizable label predictability.",
                 "Final experiment-specific conclusions are recorded after checking all results and gradient diagnostics.", "",
                 "References: [training diagnostics](https://karpathy.github.io/2019/04/25/recipe/), "
                 "[random-label memorization](https://arxiv.org/abs/1611.03530).", ""])
    (root / "comparison.md").write_text("\n".join(text))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--base", type=Path, default=BASE)
    parser.add_argument("--pred-root", type=Path, default=PRED_ROOT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--updates", type=int, default=3000)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--wandb", action="store_true")
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)
    if (args.root / "manifest.json").exists():
        raise FileExistsError("Use a new root rather than overwriting an existing R36")
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    source = Path(__file__).parent
    paths = [args.base, *(source / p for p in PROTECTED), Path(__file__),
             *(DATA_ROOT / f"TSP{s}" / p for s in ("train", "val") for p in ("dataset.pkl", "raw_label.pkl")),
             *(args.pred_root / f"predictions_{s}_u000.npz" for s in ("train", "val"))]
    manifest = dict(file_sha256={str(p): file_hash(p) for p in paths}, seed=args.seed,
                    device=torch.cuda.get_device_name(0), test_read=False)
    dump(args.root / "manifest.json", manifest)
    data = {s: raw_tsp(s) for s in ("train", "val")}
    analysis, cuts = analyze_errors(args.root, data, args.pred_root)
    groups = np.searchsorted(cuts, data["train"]["nodes"], side="left")
    indices128 = stratified_indices(data["train"]["winner"], groups, 128, args.seed)
    indices32 = stratified_indices(data["train"]["winner"], groups, 32, args.seed, available=indices128)
    for indices in (indices128, indices32):
        write_csv(args.root / f"subset_{len(indices)}_indices.csv",
                  [dict(index=int(i), winner=int(data["train"]["winner"][i]), solver=POOLS["TSP"][data["train"]["winner"][i]],
                        nodes=int(data["train"]["nodes"][i]), size_bucket=int(groups[i])) for i in indices])
    checkpoint = torch.load(args.base, map_location="cpu", weights_only=False)
    source_model_hash = state_hash(checkpoint["model"])
    prior_config = json.loads((args.pred_root / "args.json").read_text())
    if prior_config["base_sha256"] != manifest["file_sha256"][str(args.base)]:
        raise ValueError("R35 u000 predictions originate from a different checkpoint")
    loader = make_loader("TSP", "train", 32, 0, shuffle=False, cache_device=args.device)
    results = []
    for recipe in ("ce", "winner_cost"):
        results.append(run_fit(recipe, indices128, loader, data["train"], checkpoint, args))
        summarize(args.root, analysis, results)
    # A paired 32-example fallback keeps the objective comparison interpretable.
    if not all(r["passed"] for r in results):
        for recipe in ("ce", "winner_cost"):
            results.append(run_fit(recipe, indices32, loader, data["train"], checkpoint, args))
            summarize(args.root, analysis, results)
    dump(args.root / "results.json", results)
    unchanged = {str(source / p): file_hash(source / p) == manifest["file_sha256"][str(source / p)] for p in PROTECTED}
    if not all(unchanged.values()):
        raise RuntimeError("Protected model source changed during R36")
    paired = {}
    for n in sorted({r["n"] for r in results}):
        configs = [json.loads((args.root / r["name"] / "args.json").read_text()) for r in results if r["n"] == n]
        paired[str(n)] = dict(same_initial_weights=len({c["model_initial_hash"] for c in configs}) == 1,
                             same_batch_schedule=len({c["batch_schedule_hash"] for c in configs}) == 1,
                             optimizers_fresh=all(c["optimizer_initial_state_count"] == 0 for c in configs))
    file_checks = {str(p): file_hash(p) == manifest["file_sha256"][str(p)] for p in paths}
    verified = dict(protected_source_unchanged=unchanged, all_input_files_unchanged=file_checks,
                    source_model_unchanged=state_hash(checkpoint["model"]) == source_model_hash,
                    paired_protocol=paired, test_read=False)
    dump(args.root / "verification.json", verified)
    if not all(file_checks.values()) or not verified["source_model_unchanged"] or not all(all(p.values()) for p in paired.values()):
        raise RuntimeError("R36 integrity verification failed")
    print("[R36 complete] results and plots:", args.root, flush=True)


if __name__ == "__main__":
    main()
