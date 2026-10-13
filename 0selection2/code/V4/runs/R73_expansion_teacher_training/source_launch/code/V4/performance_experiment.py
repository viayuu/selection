"""R39: paired, from-scratch, 18-problem performance-prediction experiments."""

import argparse
import json
import math
import random
import shutil
import sys
import time
from argparse import Namespace
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import numpy as np
import torch

from code.unified_selector.registry import P2I, PROBLEMS
from .V4Model import get_default_model_params, make_selector
from .local_geometry import FIELDS, SCALES, local_geometry
from .multitask_probe import TaskRandomStreams, Tee, batch_schedule, dump, file_hash, gather_batch, state_hash, task_seed, update_batch
from .performance_evaluation import evaluate_performance, family_metrics
from .performance_targets import fit_training_scales, make_targets, read_raw_costs
from .train import configure_torch, make_loader, print_eval, set_seed
from .training_monitor import capture_rng


ROOT = Path("code/V4/runs/R39_performance_model")
SOURCE_FILES = ("V4Model.py", "dual_stream.py", "solver_features.py", "local_geometry.py", "tensor_loader.py",
                "train.py", "evaluate.py", "multitask_probe.py", "training_monitor.py", "performance_targets.py",
                "performance_evaluation.py", "performance_experiment.py", "performance_analysis.py")


def fit_args(group):
    return Namespace(loss_mode="winner_cost" if group == "A" else "performance", ce_weight=.35,
                     pair_weight=.10, risk_weight=.02, cost_scale=.01, performance_weight=.35)


def model_params(group, sdpa=True):
    return dict(get_default_model_params(), architecture="performance", joint_layer_num=2,
                solver_feature_weight=1.0, local_geometry=True, geometry_mode="real", sdpa=sdpa,
                ignore_coord_dist=True, native_winner=True,
                decision_head="classification" if group == "A" else "performance")


def epoch_lr(epoch, epochs, lr, warmup=3, min_lr=2e-6):
    if epoch < warmup:
        return lr * (epoch + 1) / warmup
    progress = (epoch - warmup) / max(1, epochs - warmup - 1)
    return min_lr + (lr - min_lr) * (1 + math.cos(math.pi * progress)) / 2


def snapshot(root):
    directory = root / "source_launch"
    directory.mkdir(exist_ok=True)
    manifest = {}
    for name in SOURCE_FILES:
        path = Path(__file__).parent / name
        if path.exists():
            manifest[name] = file_hash(path)
    for name in ("data.py", "registry.py"):
        path = Path(__file__).parents[1] / "unified_selector" / name
        manifest[name] = file_hash(path)
    hash_path = directory / "hashes.json"
    if hash_path.exists():
        if json.loads(hash_path.read_text()) != manifest:
            raise ValueError("Source changed: preserve this run's launch snapshot and use a new root")
        return manifest
    for name in manifest:
        path = Path(__file__).parents[1] / "unified_selector" / name if name in ("data.py", "registry.py") else Path(__file__).parent / name
        shutil.copy2(path, directory / name)
    dump(directory / "hashes.json", manifest)
    return manifest


def prepare_geometry(loaders, raw, directory):
    directory.mkdir(exist_ok=True)
    sources = {f"{split}/{p}": raw[split][p]["data_hash"] for split in ("train", "val") for p in PROBLEMS if p != "ATSP"}
    definition_hash = file_hash(Path(__file__).parent / "local_geometry.py")
    stats_path = directory / "stats.json"
    existing = json.loads(stats_path.read_text()) if stats_path.exists() else None
    if existing is not None and (existing["sources"] != sources or existing["definition_hash"] != definition_hash):
        raise ValueError("Geometry cache belongs to a different data/feature definition")
    moments = torch.zeros(2, len(FIELDS), dtype=torch.float64, device=loaders["train"]["TSP"].batch["node"].device)
    counts = {}
    for split in ("train", "val"):
        for problem, loader in loaders[split].items():
            if loader.batch["kind"] != "coord":
                continue
            path = directory / f"{problem}_{split}.pt"
            if existing is not None:
                features = torch.load(path, map_location="cpu", weights_only=True)
            else:
                started = time.monotonic()
                features = torch.zeros(loader.size, loader.batch["node"].size(1), len(FIELDS))
                valid_count = 0
                for start in range(0, loader.size, 16):
                    stop = min(start + 16, loader.size)
                    batch = gather_batch(loader, torch.arange(start, stop))
                    values = local_geometry(batch["node"][..., :2], batch["node_mask"])
                    if not torch.isfinite(values).all():
                        raise RuntimeError("Nonfinite local geometry")
                    features[start:stop, :values.size(1)] = values.float().cpu()
                    if split == "train":
                        valid = values[batch["node_mask"]]
                        moments[0] += valid.sum(0)
                        moments[1] += valid.square().sum(0)
                        valid_count += len(valid)
                torch.save(features, path)
                counts[problem] = valid_count if split == "train" else counts[problem]
                print(f"[geometry] {problem}/{split} instances={loader.size} seconds={time.monotonic()-started:.1f}", flush=True)
            if features.shape != (loader.size, loader.batch["node"].size(1), len(FIELDS)):
                raise ValueError("Geometry cache shape changed")
            loader.batch["node_geom"] = features.to(loader.batch["node"].device)
    if existing is not None:
        return existing
    count = sum(counts.values())
    mean = moments[0] / count
    std = (moments[1] / count - mean.square()).clamp_min(0).sqrt().clamp_min(1e-6)
    stats = dict(fields=list(FIELDS), scales=list(SCALES), mean=mean.cpu().tolist(), std=std.cpu().tolist(),
                 fitted_on="all coordinate training valid nodes only", valid_train_nodes=count,
                 valid_nodes_by_problem=counts, sources=sources, definition_hash=definition_hash,
                 calculation_dtype="float64", cache_dtype="float32", std_floor=1e-6,
                 neighbor_policy="exclude self/padding; include all kth-distance ties; use all when fewer than k")
    dump(stats_path, stats)
    return stats


def prepare_data(args):
    raw_train, scales = fit_training_scales(PROBLEMS)
    raw = {"train": raw_train, "val": {p: read_raw_costs(p, "val") for p in PROBLEMS}}
    scale_path = args.root / "performance_scales.json"
    if scale_path.exists() and json.loads(scale_path.read_text()) != scales:
        raise ValueError("Training costs/pools changed since R39 preparation")
    manifest = {split: {p: {key: record[key] for key in ("pool", "pool_ids", "label_hash", "data_hash")}
                       for p, record in entries.items()} for split, entries in raw.items()}
    manifest_path = args.root / "data_manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise ValueError("Training/validation data changed: preserve this root's data provenance")
    dump(scale_path, scales)
    dump(manifest_path, manifest)
    loaders = {"train": {}, "val": {}}
    for split in loaders:
        for problem in PROBLEMS:
            loader = make_loader(problem, split, args.batch_size, 0, shuffle=False, cache_device=args.device)
            record = raw[split][problem]
            if loader.size != len(record["winner"]):
                raise ValueError("Raw labels and instances disagree")
            np.testing.assert_array_equal(loader.batch["ind"].cpu().numpy(), record["winner"])
            np.testing.assert_array_equal(loader.batch["costs"].cpu().numpy(), record["costs"].astype(np.float32))
            np.testing.assert_array_equal(loader.batch["pool_ids"].cpu().numpy(), record["pool_ids"])
            loader.batch["performance_target"] = make_targets(record["costs"], scales[problem]["scale"]).to(args.device)
            loaders[split][problem] = loader
    geometry = prepare_geometry(loaders, raw, args.root / "geometry_cache")
    print(f"[data ready] train=180000 val=18000 geometry_train_nodes={geometry['valid_train_nodes']} test_read=False", flush=True)
    return loaders, raw, scales, geometry


def paired_schedule(seed, args, loaders):
    batches = {p: batch_schedule(loader.size, args.batch_size, args.epochs * (loader.size // args.batch_size), task_seed(seed, p))
               .view(args.epochs, -1, args.batch_size) for p, loader in loaders.items()}
    generator = random.Random(seed + 701000003)
    orders = []
    for epoch in range(args.epochs):
        remaining = {p: batches[p].size(1) for p in PROBLEMS}
        active, order = list(PROBLEMS), []
        while active:
            generator.shuffle(active)
            for p in list(active):
                if remaining[p] == 0:
                    active.remove(p)
                else:
                    order.append(p)
                    remaining[p] -= 1
        orders.append(order)
    return dict(batches=batches, orders=orders)


def initialize(group, seed, args, scales, geometry):
    set_seed(seed)
    params = getattr(args, "model_parameters", model_params(group, sdpa=str(args.device).startswith("cuda")))
    model = make_selector(params).to(args.device)
    with torch.no_grad():
        for p, record in scales.items():
            model.performance_scales[P2I[p]] = record["scale"]
        branch = model.instance_encoder.geometry_residual
        branch.mean.copy_(torch.tensor(geometry["mean"], device=args.device))
        branch.std.copy_(torch.tensor(geometry["std"], device=args.device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scaler = torch.amp.GradScaler("cuda", enabled=str(args.device).startswith("cuda"))
    return model, params, optimizer, scaler


def runtime_check(args, loaders, scales, geometry):
    initial_hashes, reports = {}, []
    for group in ("A", "B"):
        model, _, optimizer, scaler = initialize(group, 2, args, scales, geometry)
        initial_hashes[group] = state_hash(model.state_dict())
        if not all(p.requires_grad for p in model.parameters()):
            raise RuntimeError("R39 must not freeze original parameters")
        model.eval()
        with torch.no_grad():
            for problem in PROBLEMS:
                batch = gather_batch(loaders["train"][problem], torch.arange(2))
                labeled = model(batch)
                unlabeled = model({k: v for k, v in batch.items() if k not in ("costs", "ind", "performance_target")})
                for key in ("logits", "pred_performance"):
                    torch.testing.assert_close(labeled[key], unlabeled[key], atol=0, rtol=0)
        model.train()
        for problem in ("TSP", "CVRP", "ATSP"):
            loader = loaders["train"][problem]
            indices = loader.lengths.argsort(descending=True)[:args.batch_size]
            torch.cuda.reset_peak_memory_stats()
            started = time.monotonic()
            values = update_batch(model, optimizer, scaler, gather_batch(loader, indices), fit_args(group))
            reports.append(dict(group=group, problem=problem, batch=args.batch_size,
                                max_n=int(loader.lengths[indices].max()), peak_gib=torch.cuda.max_memory_allocated()/2**30,
                                seconds=time.monotonic()-started, **values))
            print(f"[runtime witness] {reports[-1]}", flush=True)
        del model, optimizer, scaler
        torch.cuda.empty_cache()
    if initial_hashes["A"] != initial_hashes["B"]:
        raise RuntimeError("Matched R39 initial weights differ")
    dump(args.root / "runtime_witness.json", dict(initial_hashes=initial_hashes, witnesses=reports,
                                                label_free_forward_verified_on=list(PROBLEMS), test_read=False))


def run_one(group, seed, args, loaders, raw, scales, geometry, schedule, source_hashes, directory=None):
    directory = directory or args.root / f"{'winner_cost' if group == 'A' else 'performance'}_seed{seed}"
    directory.mkdir(exist_ok=True)
    if (directory / "result.json").exists():
        result = json.loads((directory / "result.json").read_text())
        config = json.loads((directory / "args.json").read_text())
        expected = dict(group=group, seed=seed, epochs=args.epochs, batch_per_problem=args.batch_size, lr=args.lr,
                        min_lr=args.min_lr, weight_decay=args.wd, performance_scales=scales, geometry_stats=geometry,
                        source_hashes=source_hashes, schedule_hash=state_hash(schedule),
                        model_params=getattr(args, "model_parameters", model_params(group, sdpa=str(args.device).startswith("cuda"))),
                        loss=vars(getattr(args, "objective", fit_args(group))))
        if result["source_hashes"] != source_hashes or any(config[key] != value for key, value in expected.items()):
            raise ValueError("Existing R39 result does not match the requested protocol/source")
        print(f"[already complete] {directory.name}", flush=True)
        return result
    if (directory / "history.json").exists():
        raise ValueError("Partial run exists; preserve it and use a new root rather than silently restarting")
    with (directory / "train.log").open("w") as log, redirect_stdout(Tee(sys.stdout, log)), redirect_stderr(Tee(sys.stderr, log)):
        return _run_one(group, seed, args, loaders, raw, scales, geometry, schedule, source_hashes, directory)


def _run_one(group, seed, args, loaders, raw, scales, geometry, schedule, source_hashes, directory):
    model, params, optimizer, scaler = initialize(group, seed, args, scales, geometry)
    streams = TaskRandomStreams(seed, PROBLEMS)
    objective = getattr(args, "objective", fit_args(group))
    config = dict(group=group, seed=seed, epochs=args.epochs, batch_per_problem=args.batch_size, lr=args.lr,
                  min_lr=args.min_lr, weight_decay=args.wd, warmup_epochs=3, lr_schedule="epoch-wise cosine",
                  model_params=params, loss=vars(objective), decision_head=model.decision_head, test_read=False,
                  coord_augment=0, sampling="natural shuffled per-problem full passes, drop-tail as R34",
                  optimizer_restored=False, weights_restored=False, all_model_parameters_unfrozen=True,
                  performance_scales=scales, geometry_stats=geometry, source_hashes=source_hashes,
                  model_initial_hash=state_hash(model.state_dict()), schedule_hash=state_hash(schedule),
                  dropout_rng_initial_hash={p: state_hash(s) for p, s in streams.states.items()},
                  eval_precision="FP32 network / original raw FP64 costs", amp_dtype="float16", sdpa=params["sdpa"],
                  grad_clip=1.0, checkpoint_rule="minimum val macro_vs_sbs_pct using actual decision_head",
                  sbs_definition="per evaluation split's minimum mean-cost fixed solver, matching existing reports",
                  wandb_mode="offline", problems=list(PROBLEMS))
    dump(directory / "args.json", config)
    history, best, best_record, updates, skipped = [], float("inf"), None, 0, 0
    started = time.monotonic()
    wandb_run = None
    if args.wandb:
        import wandb
        wandb_run = wandb.init(project="selector", entity="yjkds-southern-university-of-science-technology",
                               name="R39_" + directory.name, config=config, dir=str(directory), mode="offline")
    print(f"[start] {directory.name} scratch=True all18=True epochs={args.epochs} batch={args.batch_size} "
          f"decision={model.decision_head} model_hash={config['model_initial_hash']}", flush=True)

    def validation(epoch):
        per, macro = evaluate_performance(model, PROBLEMS, "val", args.batch_size, 0, args.device,
                                         loaders=loaders["val"], raw=raw["val"],
                                         prediction_dir=directory / "predictions" / f"val_e{epoch:03d}")
        payload = dict(epoch=epoch, per_problem=per, macro=macro, families=family_metrics(per))
        dump(directory / f"eval_epoch{epoch:03d}.json", payload)
        print_eval(f"eval epoch {epoch} {model.decision_head}", per, macro)
        return payload

    def checkpoint(epoch, payload, name):
        torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(), scaler=scaler.state_dict(),
                        args=config, epoch=epoch - 1, macro=payload["macro"], successful_updates=updates,
                        skipped_attempts=skipped, rng=capture_rng(), task_rng=streams.states), directory / name)

    validation(0)
    with (directory / "update_trace.jsonl").open("w") as trace:
        for epoch_index, order in enumerate(schedule["orders"]):
            model.train()
            lr = epoch_lr(epoch_index, args.epochs, args.lr, min_lr=args.min_lr)
            for pg in optimizer.param_groups:
                pg["lr"] = lr
            counters = {p: 0 for p in PROBLEMS}
            meters, epoch_started = [], time.monotonic()
            for p in order:
                indices = schedule["batches"][p][epoch_index, counters[p]]
                batch = gather_batch(loaders["train"][p], indices)
                with streams.activate(p):
                    values = update_batch(model, optimizer, scaler, batch, objective)
                counters[p] += 1
                updates += 1
                skipped += values["skipped"]
                meters.append(values)
                trace.write(json.dumps(dict(step=updates, problem=p, epoch=epoch_index + 1, **values)) + "\n")
                if updates % 50 == 0:
                    stats = {key: float(np.mean([v[key] for v in meters])) for key in meters[0] if key != "amp_scale"}
                    speed = len(meters) * args.batch_size / (time.monotonic() - epoch_started)
                    peak = torch.cuda.max_memory_allocated() / 2**30
                    msg = " ".join(f"{k}={stats[k]:.5f}" for k in ("loss", "ce", "pair", "risk", "performance") if k in stats)
                    print(f"ep{epoch_index+1:03d} step{updates:06d} {msg} lr={lr:.3g} "
                          f"samples/s={speed:.0f} peak_GiB={peak:.2f} time={(time.monotonic()-started)/60:.1f}m", flush=True)
                    if wandb_run:
                        wandb_run.log({**{f"train/{key}": value for key, value in stats.items()}, "train/lr": lr,
                                       "speed/samples_per_sec": speed, "gpu/peak_memory_gib": peak}, step=updates)
            trace.flush()
            epoch = epoch_index + 1
            train_stats = {key: float(np.mean([v[key] for v in meters])) for key in meters[0]}
            payload = validation(epoch)
            record = dict(epoch=epoch, updates=updates, lr=lr, train=train_stats, val=payload["macro"],
                          families=payload["families"], per_problem=payload["per_problem"],
                          updates_by_problem=counters, skipped_attempts_total=skipped,
                          task_rng_hashes={p: state_hash(s) for p, s in streams.states.items()},
                          elapsed_seconds=time.monotonic()-started)
            if epoch % 5 == 0 or epoch == args.epochs:
                train_per, train_macro = evaluate_performance(model, PROBLEMS, "train", args.batch_size, 0, args.device,
                                                             loaders=loaders["train"], raw=raw["train"],
                                                             prediction_dir=directory / "predictions" / "train_final" if epoch == args.epochs else None)
                record["train_eval"] = train_macro
                dump(directory / f"train_eval_epoch{epoch:03d}.json", dict(per_problem=train_per, macro=train_macro,
                                                                         families=family_metrics(train_per)))
            if payload["macro"]["macro_vs_sbs_pct"] < best:
                best = payload["macro"]["macro_vs_sbs_pct"]
                best_record = record
                checkpoint(epoch, payload, "best.pt")
                print(f"[save] best.pt epoch={epoch} val_vs_sbs={best:+.5f}%", flush=True)
            checkpoint(epoch, payload, "last.pt")
            history.append(record)
            dump(directory / "history.json", history)
            print(f"[epoch done {epoch}] updates={updates} skipped_attempts_total={skipped} "
                  f"grad={train_stats['grad_norm']:.4f} elapsed={(time.monotonic()-started)/60:.1f}m", flush=True)
            if wandb_run:
                tracked = {f"val/{key}": value for key, value in payload["macro"].items()}
                for p, r in payload["per_problem"].items():
                    tracked.update({f"val/{p}/{key}": r[key] for key in ("ce", "performance_mse", "top1", "top2", "top3", "mean_cost", "vs_sbs_pct", "actual_regret_pct")})
                wandb_run.log(dict(epoch=epoch, **tracked), step=updates)
            if epoch % 5 == 0 or epoch == args.epochs:
                from .performance_analysis import plot_run
                plot_run(history, directory)
    result = dict(group=group, seed=seed, epochs=args.epochs, updates=updates, skipped_attempts=skipped,
                  best=best_record, final=history[-1], elapsed_seconds=time.monotonic()-started, source_hashes=source_hashes,
                  last5={key: float(np.mean([r["val"][key] for r in history[-5:]])) for key in history[-1]["val"]},
                  last5_families={name: {key: float(np.mean([r["families"][name][key] for r in history[-5:]]))
                                          for key in history[-1]["families"][name]} for name in history[-1]["families"]})
    dump(directory / "result.json", result)
    if wandb_run:
        wandb_run.summary.update(dict(best_epoch=best_record["epoch"], best_val_vs_sbs=best,
                                     final_val_top1=history[-1]["val"]["macro_top1"]))
        wandb_run.finish()
    print(f"[done] {directory.name} epochs={args.epochs} updates={updates} best_epoch={best_record['epoch']} "
          f"final_top1={result['final']['val']['macro_top1']:.4f} seconds={result['elapsed_seconds']:.1f}", flush=True)
    del model, optimizer, scaler
    torch.cuda.empty_cache()
    return result


def run_from_train(args):
    if args.epochs != 60 or args.warmup_epochs != 3 or args.eval_every != 1 or args.early_stop_patience:
        raise ValueError("R39 is fixed at 60 epochs, 3 warmup epochs, validation every epoch and no early stopping")
    if args.resume or args.problems or args.coord_augment or not args.local_geometry or not args.native_winner or args.winner_balance or args.no_solver_features:
        raise ValueError("R39 requires scratch/all18/natural sampling/local geometry/native winners")
    if not (args.amp and args.amp_dtype == "fp16" and args.sdpa and args.cache_gpu) or args.lr_schedule != "cosine":
        raise ValueError("R39 CUDA protocol requires --amp --amp-dtype fp16 --sdpa --cache-gpu --lr-schedule cosine")
    if args.wandb and args.wandb_mode != "offline":
        raise ValueError("This R39 queue records W&B offline; use --wandb-mode offline")
    if args.loss_mode not in ("winner_cost", "performance"):
        raise ValueError("R39 needs winner_cost (control) or performance (CE + MSE)")
    group = "A" if args.loss_mode == "winner_cost" else "B"
    if args.decision_head != ("classification" if group == "A" else "performance") or not args.skip_test:
        raise ValueError("R39 requires the matching decision head and --skip-test")
    protocol = Namespace(root=Path(args.save_dir).parent, device=args.device, batch_size=args.batch_per_problem,
                         epochs=args.epochs, lr=args.lr, min_lr=args.min_lr, wd=args.wd, wandb=args.wandb)
    from .train import build_model
    _, protocol.model_parameters = build_model(args)
    protocol.objective = Namespace(loss_mode=args.loss_mode, ce_weight=args.ce_weight,
                                   pair_weight=args.pair_weight, risk_weight=args.risk_weight,
                                   cost_scale=args.cost_scale, performance_weight=args.performance_weight)
    protocol.root.mkdir(parents=True, exist_ok=True)
    configure_torch()
    loaders, raw, scales, geometry = prepare_data(protocol)
    sources = snapshot(protocol.root)
    return run_one(group, args.seed, protocol, loaders, raw, scales, geometry,
                   paired_schedule(args.seed, protocol, loaders["train"]), sources, Path(args.save_dir))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=640)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--min-lr", type=float, default=2e-6)
    parser.add_argument("--wd", type=float, default=1e-4)
    parser.add_argument("--seeds", default="2")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    configure_torch()
    args.root.mkdir(parents=True, exist_ok=True)
    budget_path = args.root / "run_budget.json"
    seeds = [int(x) for x in args.seeds.split(",")]
    if not budget_path.exists():
        dump(budget_path, dict(seeds=seeds, groups=["A", "B"], expected_runs=2*len(seeds), epochs_per_run=args.epochs))
    elif json.loads(budget_path.read_text())["seeds"] != seeds:
        raise ValueError("Requested seeds disagree with this root's declared run budget")
    loaders, raw, scales, geometry = prepare_data(args)
    if args.prepare_only:
        return
    runtime_check(args, loaders, scales, geometry)
    if args.verify_only:
        return
    sources = snapshot(args.root)
    (args.root / "protocol").mkdir(exist_ok=True)
    for seed in seeds:
        schedule = paired_schedule(seed, args, loaders["train"])
        torch.save(schedule, args.root / "protocol" / f"seed{seed}.pt")
        for group in ("A", "B"):
            run_one(group, seed, args, loaders, raw, scales, geometry, schedule, sources)
            from .performance_analysis import summarize
            summarize(args.root)
    print("[all done] R39 completed; test deferred, all results use validation", flush=True)


if __name__ == "__main__":
    main()
