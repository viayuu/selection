"""Paired TSP geometry probes: R37 full tuning or R38 frozen-base tuning."""

import argparse
import copy
import json
import math
import shutil
import sys
import time
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from code.unified_selector.registry import DATA_ROOT, POOLS
from .V4Model import make_selector
from .local_geometry import FIELDS, SCALES, local_geometry
from .multitask_probe import (TaskRandomStreams, Tee, batch_schedule, dump, file_hash,
                             gather_batch, state_hash, task_seed, update_batch)
from .tensor_loader import TensorBatchLoader
from .train import configure_torch, make_loader, selector_loss, set_seed
from .training_monitor import capture_rng
from .tsp_learnability import BASE, fit_args, gap_and_regret, raw_tsp


ROOT = Path("code/V4/runs/R37_tsp_local_geometry")
METRICS = ("ce", "top1", "mean_cost", "vs_sbs_pct", "actual_regret_pct",
           "gap_gt_0_1_top1", "gap_gt_0_1_actual_regret_pct", "gap_gt_0_5_top1", "gap_gt_0_5_actual_regret_pct")
PROB_METRICS = ("expected_mean_cost", "expected_regret_pct")


def original_state(model):
    return {name: value for name, value in model.state_dict().items() if "geometry_residual." not in name}


def lr_factor(update, budget, warmup=50):
    if update <= warmup:
        return update / warmup
    progress = (update - warmup) / (budget - warmup)
    return .1 + .9 * (1 + math.cos(math.pi * progress)) / 2


def prepare_geometry(loaders, root):
    total = 0
    sums = torch.zeros(12, dtype=torch.float64, device=loaders["train"].batch["node"].device)
    squares = torch.zeros_like(sums)
    for split, loader in loaders.items():
        start_time = time.monotonic()
        features = torch.zeros(loader.size, loader.batch["node"].size(1), 12)
        for start in range(0, loader.size, 16):
            stop = min(start + 16, loader.size)
            batch = gather_batch(loader, torch.arange(start, stop))
            values = local_geometry(batch["node"][..., :2], batch["node_mask"])
            if not torch.isfinite(values).all():
                raise RuntimeError("Nonfinite local geometry")
            features[start:stop, :values.size(1)] = values.float().cpu()
            if split == "train":
                valid = values[batch["node_mask"]]
                sums += valid.sum(0)
                squares += valid.square().sum(0)
                total += len(valid)
            if start % 1600 == 0:
                print(f"[geometry cache] {split} {stop}/{loader.size}", flush=True)
        torch.save(features, root / f"node_geometry_{split}.pt")
        loader.batch["node_geom"] = features.to(loader.batch["node"].device)
        print(f"[geometry cache done] {split} seconds={time.monotonic() - start_time:.1f}", flush=True)
    mean = sums / total
    std = (squares / total - mean.square()).clamp_min(0).sqrt().clamp_min(1e-6)
    stats = dict(fields=list(FIELDS), scales=list(SCALES), mean=mean.cpu().tolist(), std=std.cpu().tolist(),
                 valid_train_nodes=total, fitted_on="train valid nodes only", calculation_dtype="float64",
                 cache_dtype="float32", epsilon=1e-12, std_floor=1e-6,
                 neighbor_policy="exclude self/padding; include all kth-distance ties; use all if fewer than k",
                 distance_normalizer="mean of all valid off-diagonal Euclidean distances within instance",
                 moments="population standard deviation/covariance; covariance includes center",
                 field_definitions=dict(radius="kth normalized distance", mean_distance="mean neighbor distance",
                                        distance_cv="population std / (mean + epsilon)",
                                        anisotropy="eigenvalue difference / (sum + epsilon)"))
    dump(root / "geometry_stats.json", stats)
    return stats


def metrics_from_logits(logits, raw):
    winner = torch.from_numpy(raw["winner"])
    rank = logits.argsort(1, descending=True).numpy()
    pred = rank[:, 0]
    costs = raw["costs"]
    gap, regret = gap_and_regret(costs, pred)
    selected = costs[np.arange(len(pred)), pred]
    sbs = float(costs.mean(0).min())
    mean_cost = float(selected.mean())
    result = dict(n=len(pred), ce=float(F.cross_entropy(logits, winner)), mean_cost=mean_cost,
                  sbs=sbs, vs_sbs_pct=(mean_cost / sbs - 1) * 100, actual_regret_pct=float(regret.mean()),
                  **{f"top{k}": float((rank[:, :k] == raw["winner"][:, None]).any(1).mean()) for k in (1, 2, 3)})
    probability = logits.double().softmax(1).numpy()
    best = costs.min(1)
    result["expected_mean_cost"] = float((probability * costs).sum(1).mean())
    result["expected_regret_pct"] = float((probability * ((costs - best[:, None]) / best[:, None] * 100)).sum(1).mean())
    for threshold, suffix in ((.1, "0_1"), (.5, "0_5")):
        mask = gap > threshold
        result[f"gap_gt_{suffix}_n"] = int(mask.sum())
        result[f"gap_gt_{suffix}_top1"] = float((pred[mask] == raw["winner"][mask]).mean()) if mask.any() else None
        result[f"gap_gt_{suffix}_actual_regret_pct"] = float(regret[mask].mean()) if mask.any() else None
        result[f"gap_gt_{suffix}_mean_cost"] = float(selected[mask].mean()) if mask.any() else None
    result["pick_dist"] = dict(zip(POOLS["TSP"], (np.bincount(pred, minlength=costs.shape[1]) / len(pred)).tolist()))
    return result, pred


@torch.no_grad()
def evaluate(model, loader, raw, path):
    model.eval()
    logits = torch.cat([model(batch)["logits"].float().cpu()
                        for batch in TensorBatchLoader(loader.batch, loader.batch_size, False, False)])
    if not torch.isfinite(logits).all():
        raise RuntimeError("Nonfinite evaluation logits")
    result, pred = metrics_from_logits(logits, raw)
    np.savez_compressed(path, indices=np.arange(len(pred)), logits=logits.numpy(), winner=raw["winner"], pred=pred,
                        pool_ids=loader.batch["pool_ids"].cpu().numpy())
    return result


def initialize(checkpoint, mode, stats, seed, device, freeze_base=False):
    set_seed(seed)
    params = copy.deepcopy(checkpoint["args"]["model_params"])
    params.update(local_geometry=True, geometry_mode=mode)
    model = make_selector(params).to(device)
    missing, unexpected = model.load_state_dict(checkpoint["model"], strict=False)
    expected = {"instance_encoder.geometry_residual." + k for k in model.instance_encoder.geometry_residual.state_dict()}
    if unexpected or set(missing) != expected:
        raise ValueError("Warm start mismatched fields outside the new geometry module")
    module = model.instance_encoder.geometry_residual
    module.mean.copy_(torch.tensor(stats["mean"], device=device))
    module.std.copy_(torch.tensor(stats["std"], device=device))
    new_ids = {id(p) for p in module.parameters()}
    original = [p for p in model.parameters() if id(p) not in new_ids]
    if freeze_base:
        for parameter in original:
            parameter.requires_grad_(False)
    groups = [] if freeze_base else [dict(params=original, lr=2e-5, initial_lr=2e-5, name="base")]
    groups.append(dict(params=list(module.parameters()), lr=2e-4, initial_lr=2e-4, name="geometry"))
    optimizer = torch.optim.AdamW(groups, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=str(device).startswith("cuda"))
    return model, optimizer, scaler, params


def run_one(mode, seed, args, checkpoint, stats, loaders, raw, baseline):
    directory = args.root / f"{'redundant' if mode == 'zero' else 'geometry'}_seed{seed}"
    directory.mkdir()
    with (directory / "train.log").open("w") as log:
        with redirect_stdout(Tee(sys.stdout, log)), redirect_stderr(Tee(sys.stderr, log)):
            return _run_one(mode, seed, args, checkpoint, stats, loaders, raw, baseline, directory)


def _run_one(mode, seed, args, checkpoint, stats, loaders, raw, baseline, directory):
    model, optimizer, scaler, params = initialize(checkpoint, mode, stats, seed, args.device, args.freeze_base)
    base_hash = state_hash(original_state(model))
    schedule = batch_schedule(loaders["train"].size, args.batch_size, args.updates, task_seed(seed, "TSP"))
    torch.save(schedule, directory / "batch_indices.pt")
    streams = TaskRandomStreams(seed, ["TSP"])
    config = dict(mode=mode, seed=seed, base_checkpoint=str(args.base), base_epoch=checkpoint["epoch"] + 1,
                  model_params=params, geometry_stats=stats, model_initial_hash=state_hash(model.state_dict()),
                  new_branch_initial_hash=state_hash(model.instance_encoder.geometry_residual.state_dict()),
                  batch_hash=state_hash(schedule), dropout_rng_initial_hash=state_hash(streams.states["TSP"]),
                  optimizer_restored=False, optimizer_initial_state_count=len(optimizer.state), loss=vars(fit_args("winner_cost")),
                  dropout=params["dropout"], branch_dropout=0., weight_decay=1e-4,
                  freeze_base=args.freeze_base, original_state_initial_hash=base_hash,
                  trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                  base_lr=0. if args.freeze_base else 2e-5, new_lr=2e-4,
                  warmup_updates=50, schedule="cosine after warmup to 10% initial lr", amp_dtype="float16",
                  scaler_initial=scaler.state_dict(), grad_clip=1., batch_size=args.batch_size, updates=args.updates,
                  eval_interval=50, coord_augment=0, native_winner=True, sampling="natural shuffled full passes with R34 drop-tail",
                  test_read=False, wandb_mode="offline", eval_precision="FP32 logits / raw FP64 cost metrics")
    dump(directory / "args.json", config)
    history, skipped, traces = [], 0, []
    start = time.monotonic()
    wandb_run = None
    if args.wandb:
        import wandb
        wandb_run = wandb.init(project="selector", entity="yjkds-southern-university-of-science-technology",
                               name=("R38_" if args.freeze_base else "R37_") + directory.name,
                               config=config, dir=str(directory), mode="offline")
    print(f"[start] {directory.name} updates={args.updates} batch={args.batch_size} optimizer=fresh "
          f"base_frozen={args.freeze_base} base_lr={config['base_lr']:g} geometry_lr=2e-4 "
          f"model_hash={config['model_initial_hash']}", flush=True)

    def evaluate_and_save(update):
        record = dict(updates=update, skipped=skipped, elapsed_seconds=time.monotonic() - start,
                      lr_by_group={g["name"]: g["lr"] for g in optimizer.param_groups})
        if args.freeze_base:
            record["original_state_hash"] = state_hash(original_state(model))
            if record["original_state_hash"] != base_hash:
                raise RuntimeError("Frozen original parameters/buffers changed")
        for split, loader in loaders.items():
            prediction = directory / f"predictions_{split}_u{update:04d}.npz"
            record[split] = evaluate(model, loader, raw[split], prediction)
            if update == 0:
                with np.load(prediction) as p, np.load(args.root / f"baseline_{split}.npz") as b:
                    np.testing.assert_allclose(p["logits"], b["logits"], atol=1e-6, rtol=1e-6)
            r = record[split]
            print(f"[{split} updates {update:04d}] top1={r['top1']:.4f} ce={r['ce']:.6f} mean_cost={r['mean_cost']:.6f} "
                  f"vs_sbs={r['vs_sbs_pct']:+.4f}% regret={r['actual_regret_pct']:.4f}% "
                  f"expected_regret={r['expected_regret_pct']:.4f}% "
                  f"gap>0.1_top1={r['gap_gt_0_1_top1']:.4f} gap>0.5_top1={r['gap_gt_0_5_top1']:.4f}", flush=True)
        record["elapsed_seconds"] = time.monotonic() - start
        history.append(record)
        dump(directory / "history.json", history)
        torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(), scaler=scaler.state_dict(),
                        args=config, updates=update, metrics=record, rng=capture_rng(), task_rng=streams.states), directory / "last.pt")
        if wandb_run:
            wandb_run.log({f"{split}/{key}": record[split][key] for split in ("train", "val")
                           for key in METRICS + PROB_METRICS}, step=update)
        model.train()

    evaluate_and_save(0)
    with (directory / "update_trace.jsonl").open("w") as trace:
        for update, indices in enumerate(schedule, 1):
            factor = lr_factor(update, args.updates)
            for group in optimizer.param_groups:
                group["lr"] = group["initial_lr"] * factor
            before_hash = state_hash(streams.states["TSP"])
            with streams.activate("TSP"):
                values = update_batch(model, optimizer, scaler, gather_batch(loaders["train"], indices), fit_args("winner_cost"))
            skipped += values["skipped"]
            values.update(updates=update, batch_hash=state_hash(indices), dropout_rng_before=before_hash,
                          dropout_rng_after=state_hash(streams.states["TSP"]), lr_factor=factor)
            trace.write(json.dumps(values) + "\n")
            traces.append(values)
            if update % 50 == 0 or update == args.updates:
                trace.flush()
                recent = traces[-50:]
                print(f"[train updates {update:04d}] loss={np.mean([v['loss'] for v in recent]):.5f} "
                      f"grad={np.mean([v['grad_norm'] for v in recent]):.4f} skipped_total={skipped} "
                      f"lr_{optimizer.param_groups[-1]['name']}={optimizer.param_groups[-1]['lr']:.3g} "
                      f"elapsed={time.monotonic() - start:.1f}s", flush=True)
                evaluate_and_save(update)
    result = dict(name=directory.name, mode=mode, seed=seed, updates=args.updates, skipped=skipped,
                  freeze_base=args.freeze_base,
                  final=history[-1], last5={split: {key: float(np.mean([h[split][key] for h in history[-5:]]))
                                                  for key in METRICS + PROB_METRICS}
                                          for split in ("train", "val")}, elapsed_seconds=time.monotonic() - start)
    dump(directory / "result.json", result)
    if wandb_run:
        wandb_run.summary.update({"updates": args.updates, "final_val_top1": history[-1]["val"]["top1"]})
        wandb_run.finish()
    print(f"[done] {directory.name} updates={args.updates} val_top1={history[-1]['val']['top1']:.4f}", flush=True)
    del model, optimizer, scaler
    torch.cuda.empty_cache()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--base", type=Path, default=BASE)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=640)
    parser.add_argument("--updates", type=int, default=1000)
    parser.add_argument("--seeds", default="2,3,4")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--continue-queue", action="store_true")
    parser.add_argument("--freeze-base", action="store_true")
    parser.add_argument("--geometry-cache", type=Path, help="Reuse an identical prior probe's train-only geometry statistics/cache")
    args = parser.parse_args()
    if args.updates <= 50:
        raise ValueError("Cosine continuation requires a budget above warmup")
    args.root.mkdir(parents=True, exist_ok=True)
    configure_torch()
    torch.set_num_threads(1)
    source = Path(__file__).parent
    paths = [args.base, *(source / p for p in ("V4Model.py", "dual_stream.py", "solver_features.py", "local_geometry.py",
                                             "geometry_probe.py", "tensor_loader.py", "multitask_probe.py", "train.py",
                                             "training_monitor.py", "tsp_learnability.py")),
             *(DATA_ROOT / f"TSP{s}" / p for s in ("train", "val") for p in ("dataset.pkl", "raw_label.pkl"))]
    if args.geometry_cache:
        reference = json.loads((args.geometry_cache / "manifest.json").read_text())["file_sha256"]
        cache_inputs = [args.base, *(DATA_ROOT / f"TSP{s}" / p for s in ("train", "val")
                                    for p in ("dataset.pkl", "raw_label.pkl"))]
        if any(reference.get(str(p)) != file_hash(p) for p in cache_inputs):
            raise ValueError("Geometry reference and current checkpoint/data differ")
        paths.extend(args.geometry_cache / p for p in ("geometry_stats.json", "node_geometry_train.pt", "node_geometry_val.pt"))
    hashes = {str(p): file_hash(p) for p in paths}
    manifest_path = args.root / "manifest.json"
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text())
        if not args.continue_queue or old["file_sha256"] != hashes:
            raise FileExistsError("Existing R37: use --continue-queue only with identical inputs/source")
    else:
        dump(manifest_path, dict(file_sha256=hashes, args={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                                 hardware=torch.cuda.get_device_name(0), test_read=False))
    raw = {split: raw_tsp(split) for split in ("train", "val")}
    loaders = {split: make_loader("TSP", split, args.batch_size, 0, shuffle=False, cache_device=args.device) for split in ("train", "val")}
    stats_path = args.root / "geometry_stats.json"
    if args.geometry_cache or (args.continue_queue and stats_path.exists()):
        if args.geometry_cache:
            for filename in ("geometry_stats.json", "node_geometry_train.pt", "node_geometry_val.pt"):
                shutil.copy2(args.geometry_cache / filename, args.root / filename)
        stats = json.loads(stats_path.read_text())
        for split, loader in loaders.items():
            loader.batch["node_geom"] = torch.load(args.root / f"node_geometry_{split}.pt", weights_only=True).to(args.device)
    else:
        stats = prepare_geometry(loaders, args.root)
    checkpoint = torch.load(args.base, map_location="cpu", weights_only=False)
    base = make_selector(checkpoint["args"]["model_params"]).to(args.device)
    base.load_state_dict(checkpoint["model"], strict=True)
    baseline = {s: evaluate(base, l, raw[s], args.root / f"baseline_{s}.npz") for s, l in loaders.items()}
    dump(args.root / "baseline.json", baseline)
    print(f"[R34 reference] val_top1={baseline['val']['top1']:.4f} val_regret={baseline['val']['actual_regret_pct']:.4f}%", flush=True)
    del base
    results = []
    for seed in map(int, args.seeds.split(",")):
        for mode in ("zero", "real"):
            path = args.root / f"{'redundant' if mode == 'zero' else 'geometry'}_seed{seed}" / "result.json"
            if args.continue_queue and path.exists():
                result = json.loads(path.read_text())
                if result["updates"] != args.updates:
                    raise ValueError("Completed run has a different update budget")
                print("[reuse completed]", result["name"], flush=True)
            else:
                result = run_one(mode, seed, args, checkpoint, stats, loaders, raw, baseline)
            results.append(result)
            dump(args.root / "results.json", results)
    unchanged = {str(p): file_hash(p) == hashes[str(p)] for p in paths}
    dump(args.root / "verification.json", dict(inputs_and_source_unchanged=unchanged, test_read=False))
    if not all(unchanged.values()):
        raise RuntimeError("R37 inputs changed while experiments were running")
    print("[geometry probe training complete]", args.root, flush=True)


if __name__ == "__main__":
    main()
