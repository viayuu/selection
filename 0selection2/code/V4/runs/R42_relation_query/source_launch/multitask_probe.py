"""R35: paired joint/TSP-only continuation, measured in successful TSP updates."""

import argparse
import copy
import csv
import hashlib
import json
import sys
import time
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from code.unified_selector.registry import DATA_ROOT, GLOBAL_SOLVERS, POOLS, PROBLEMS
from .V4Model import make_selector
from .tensor_loader import TensorBatchLoader
from .train import configure_torch, make_loader, selector_loss, set_seed
from .training_monitor import capture_rng, restore_rng


DEFAULT_ROOT = Path("code/V4/runs/R35_multitask_probe")
DEFAULT_BASE = Path("code/V4/runs/R34_winner_cost_seed2_4090/best.pt")
PROTECTED = ("V4Model.py", "dual_stream.py", "solver_features.py")
METRICS = ("ce", "top1", "top2", "top3", "mean_cost", "vs_sbs_pct")


def dump(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def state_hash(value):
    digest = hashlib.sha256()

    def visit(item):
        if torch.is_tensor(item) or isinstance(item, np.ndarray):
            array = item.detach().cpu().contiguous().numpy() if torch.is_tensor(item) else item
            digest.update(str((array.dtype, array.shape)).encode())
            digest.update(array.tobytes())
        elif isinstance(item, dict):
            for key in sorted(item, key=str):
                digest.update(str(key).encode())
                visit(item[key])
        elif isinstance(item, (list, tuple)):
            for child in item:
                visit(child)
        else:
            digest.update(repr(item).encode())

    visit(value)
    return digest.hexdigest()


def task_seed(seed, problem):
    return seed * 100003 + PROBLEMS.index(problem) * 1009 + 17


def batch_schedule(size, batch_size, updates, seed):
    """Repeat independently shuffled passes, retaining R34's drop-tail policy."""
    if size < batch_size:
        raise ValueError("Training dataset must contain at least one full batch")
    generator = torch.Generator().manual_seed(seed)
    batches = []
    while len(batches) < updates:
        order = torch.randperm(size, generator=generator)
        for start in range(0, size - batch_size + 1, batch_size):
            batches.append(order[start:start + batch_size])
            if len(batches) == updates:
                break
    return torch.stack(batches)


class TaskRandomStreams:
    def __init__(self, seed, problems):
        outer = capture_rng()
        self.states = {}
        for problem in problems:
            set_seed(task_seed(seed, problem) + 500000003)
            self.states[problem] = capture_rng()
        restore_rng(outer)

    @contextmanager
    def activate(self, problem):
        outer = capture_rng()
        restore_rng(self.states[problem])
        try:
            yield
        finally:
            self.states[problem] = capture_rng()
            restore_rng(outer)


def gather_batch(loader, indices):
    max_n = int(loader.lengths[indices].max())
    device_indices = indices.to(loader.batch["costs"].device)
    batch = {}
    for key, value in loader.batch.items():
        if not torch.is_tensor(value) or key == "pool_ids":
            batch[key] = value
            continue
        if key in ("node", "node_mask", "node_geom"):
            value = value[:, :max_n]
        elif key == "matrix":
            value = value[:, :max_n, :max_n]
        batch[key] = value.index_select(0, device_indices)
    return batch


def restore_continuation(checkpoint, device, lr):
    params = copy.deepcopy(checkpoint["args"]["model_params"])
    model = make_selector(params).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    # Adam's scalar step tensors can alias a CPU checkpoint without this copy.
    optimizer.load_state_dict(copy.deepcopy(checkpoint["optimizer"]))
    for group in optimizer.param_groups:
        group["lr"] = lr
        group["initial_lr"] = lr
        if group["weight_decay"] != 1e-4:
            raise ValueError("Unexpected checkpoint weight decay")
    scaler = torch.amp.GradScaler("cuda", enabled=str(device).startswith("cuda"))
    scaler.load_state_dict(copy.deepcopy(checkpoint["scaler"]))
    return model, optimizer, scaler


def update_batch(model, optimizer, scaler, batch, args):
    """Retry AMP overflow with identical indices/dropout and a reduced scale."""
    before = capture_rng()
    for attempt in range(20):
        restore_rng(before)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=batch["costs"].device.type, dtype=torch.float16,
                            enabled=batch["costs"].is_cuda):
            out = model(batch)
            loss, parts = selector_loss(out, batch["costs"], None, args, winner=batch["ind"],
                                        performance_target=batch.get("performance_target"))
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite training objective")
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        old_scale = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        if scaler.get_scale() >= old_scale:
            if not torch.isfinite(norm):
                raise RuntimeError("A nonfinite gradient was not skipped")
            return dict(loss=float(loss.detach()), grad_norm=float(norm), skipped=attempt,
                        amp_scale=scaler.get_scale(), **{k: float(v) for k, v in parts.items()})
        print(f"[AMP retry] attempt={attempt + 1} scale={old_scale:g}->{scaler.get_scale():g}", flush=True)
    raise RuntimeError("AMP overflow persisted for 20 attempts on the same batch")


@torch.no_grad()
def evaluate_tsp(model, loader, path):
    model.eval()
    batches = TensorBatchLoader(loader.batch, loader.batch_size, shuffle=False, drop_last=False)
    logits = torch.cat([model(batch)["logits"].float().cpu() for batch in batches])
    if not torch.isfinite(logits).all():
        raise RuntimeError("Nonfinite evaluation logits")
    costs = loader.batch["costs"].cpu()
    winner = loader.batch["ind"].cpu()
    rank = logits.argsort(dim=1, descending=True)
    pred = rank[:, 0]
    selected = costs.gather(1, pred[:, None]).squeeze(1)
    n = len(winner)
    sbs = float(costs.double().mean(0).min())
    oracle = float(costs.double().min(1).values.mean())
    mean_cost = float(selected.double().mean())
    result = dict(n=n, ce=float(F.cross_entropy(logits, winner)), mean_cost=mean_cost,
                  sbs=sbs, oracle=oracle, vs_sbs_pct=(mean_cost / sbs - 1) * 100,
                  vs_oracle_pct=(mean_cost / oracle - 1) * 100,
                  top1_tie_aware=float((selected == costs.min(1).values).double().mean()),
                  **{f"top{k}": float((rank[:, :k] == winner[:, None]).any(1).double().mean()) for k in (1, 2, 3)})
    result["pick_dist"] = dict(zip(POOLS["TSP"], (torch.bincount(pred, minlength=costs.size(1)).double() / n).tolist()))
    result["method_top1"] = dict(zip(POOLS["TSP"], (torch.bincount(winner, minlength=costs.size(1)).double() / n).tolist()))
    result["method_mean_cost"] = dict(zip(POOLS["TSP"], costs.double().mean(0).tolist()))
    np.savez_compressed(path, indices=np.arange(n), logits=logits.numpy(), winner=winner.numpy(),
                        costs=costs.numpy(), pred=pred.numpy(), selected_cost=selected.numpy(),
                        pool_ids=loader.batch["pool_ids"].cpu().numpy())
    return result


class Tee:
    def __init__(self, stream, file):
        self.stream, self.file = stream, file

    def write(self, value):
        self.stream.write(value)
        self.file.write(value)
        self.file.flush()

    def flush(self):
        self.stream.flush()
        self.file.flush()

    def __getattr__(self, name):
        return getattr(self.stream, name)


def run_one(mode, seed, args, checkpoint, loaders, val_loader):
    directory = args.root / f"{mode}_seed{seed}"
    directory.mkdir()
    with (directory / "train.log").open("w") as log:
        with redirect_stdout(Tee(sys.stdout, log)), redirect_stderr(Tee(sys.stderr, log)):
            _run_one(mode, seed, args, checkpoint, loaders, val_loader, directory)


def _run_one(mode, seed, args, checkpoint, loaders, val_loader, directory):
    problems = list(PROBLEMS) if mode == "joint" else ["TSP"]
    set_seed(seed)
    model, optimizer, scaler = restore_continuation(checkpoint, args.device, args.lr)
    config = copy.deepcopy(checkpoint["args"])
    config.update(mode=mode, seed=seed, base_checkpoint=str(args.base), base_sha256=file_hash(args.base),
                  base_epoch=checkpoint["epoch"] + 1, lr=args.lr, wd=1e-4, batch_per_problem=args.batch_size,
                  tsp_updates=args.updates, eval_interval=args.eval_interval, train_problems=problems,
                  coord_augment=0, warmup_epochs=0, lr_schedule="constant", early_stop_patience=0,
                  skip_test=True, optimizer_initial_hash=state_hash(optimizer.state_dict()),
                  model_initial_hash=state_hash(model.state_dict()), scaler_initial=scaler.state_dict(),
                  task_order=problems, rng_policy="independent per-task sampling and dropout; retry identical batch/RNG")
    dump(directory / "args.json", config)
    schedules = {p: batch_schedule(loaders[p].size, args.batch_size, args.updates, task_seed(seed, p)) for p in problems}
    schedule_path = args.root / f"tsp_batches_seed{seed}.pt"
    if schedule_path.exists():
        if not torch.equal(torch.load(schedule_path, weights_only=True), schedules["TSP"]):
            raise RuntimeError("Paired TSP batch schedules differ")
    else:
        torch.save(schedules["TSP"], schedule_path)
    torch.save(schedules, directory / "batch_schedules.pt")
    streams = TaskRandomStreams(seed, problems)
    counts, skipped = {p: 0 for p in problems}, {p: 0 for p in problems}
    history = []
    start = time.monotonic()
    wandb_run = None
    if args.wandb:
        import wandb
        wandb_run = wandb.init(project=config["wandb_project"], entity=config["wandb_entity"],
                               name="R35_" + directory.name, config=config,
                               dir=str(directory), mode="offline")
    print(f"[start] {directory.name} checkpoint_epoch=50 lr={args.lr:g} tasks={problems}", flush=True)

    def evaluate_and_save():
        step = counts["TSP"]
        record = dict(tsp_updates=step, total_updates=sum(counts.values()), updates_by_problem=dict(counts),
                      skipped_by_problem=dict(skipped), lr=optimizer.param_groups[0]["lr"],
                      elapsed_seconds=time.monotonic() - start)
        for split, loader in (("train", loaders["TSP"]), ("val", val_loader)):
            record[split] = evaluate_tsp(model, loader, directory / f"predictions_{split}_u{step:03d}.npz")
            r = record[split]
            print(f"[{split} TSP updates {step:03d}] ce={r['ce']:.6f} top1={r['top1']:.4f} "
                  f"top2={r['top2']:.4f} top3={r['top3']:.4f} mean_cost={r['mean_cost']:.6f} "
                  f"vs_sbs={r['vs_sbs_pct']:+.4f}% n={r['n']} total_updates={record['total_updates']}", flush=True)
        record["elapsed_seconds"] = time.monotonic() - start
        history.append(record)
        dump(directory / "history.json", history)
        torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(), scaler=scaler.state_dict(),
                        args=config, base_epoch=checkpoint["epoch"], tsp_updates=step,
                        global_step=checkpoint["global_step"] + sum(counts.values()) + sum(skipped.values()),
                        successful_updates=checkpoint["successful_updates"] + sum(counts.values()),
                        updates_by_problem=dict(counts), skipped_by_problem=dict(skipped),
                        task_rng=streams.states, rng=capture_rng(), next_batch=step,
                        metrics=record, scheduler=None), directory / "last.pt")
        if wandb_run is not None:
            wandb_run.log({f"{split}/{key}": record[split][key] for split in ("train", "val") for key in METRICS}, step=step)
        model.train()

    evaluate_and_save()
    with (directory / "tsp_update_trace.jsonl").open("w") as trace:
        for round_index in range(args.updates):
            for problem in problems:
                indices = schedules[problem][round_index]
                batch = gather_batch(loaders[problem], indices)
                rng_before = state_hash(streams.states[problem]) if problem == "TSP" else None
                with streams.activate(problem):
                    stats = update_batch(model, optimizer, scaler, batch, args)
                counts[problem] += 1
                skipped[problem] += stats["skipped"]
                if problem == "TSP":
                    event = dict(tsp_update=counts[problem], indices_hash=state_hash(indices),
                                 dropout_rng_before=rng_before, dropout_rng_after=state_hash(streams.states[problem]), **stats)
                    trace.write(json.dumps(event) + "\n")
                    trace.flush()
            step = counts["TSP"]
            if step % 10 == 0:
                print(f"[progress] {directory.name} TSP={step}/{args.updates} total={sum(counts.values())} "
                      f"skipped={sum(skipped.values())} lr={optimizer.param_groups[0]['lr']:g} "
                      f"elapsed={(time.monotonic() - start) / 60:.2f}m", flush=True)
            if step % args.eval_interval == 0 or step == args.updates:
                evaluate_and_save()
    if wandb_run is not None:
        wandb_run.finish()
    dump(directory / "finished.json", dict(tsp_updates=counts["TSP"], total_updates=sum(counts.values()),
                                             skipped=skipped, elapsed_seconds=time.monotonic() - start))
    print(f"[done] {directory.name} TSP={counts['TSP']} total={sum(counts.values())}", flush=True)
    del model, optimizer, scaler
    torch.cuda.empty_cache()


def summarize(root, seeds, updates=300, eval_interval=30):
    rows, histories, paired, checks = [], {}, {}, {}
    baseline_logits = {}
    for seed in seeds:
        for mode in ("joint", "tsp_only"):
            directory = root / f"{mode}_seed{seed}"
            finished = json.loads((directory / "finished.json").read_text())
            if finished["tsp_updates"] != updates or finished["total_updates"] != updates * (18 if mode == "joint" else 1):
                raise RuntimeError(f"Incomplete update budget: {directory}")
            history = json.loads((directory / "history.json").read_text())
            if [r["tsp_updates"] for r in history] != list(range(0, updates + 1, eval_interval)):
                raise RuntimeError(f"Evaluation points do not match: {directory}")
            histories[(mode, seed)] = history
            for split in ("train", "val"):
                with np.load(directory / f"predictions_{split}_u000.npz") as prediction:
                    value = prediction["logits"].copy()
                    if split in baseline_logits and not np.array_equal(value, baseline_logits[split]):
                        raise RuntimeError("Baseline predictions must match for all six continuations")
                    baseline_logits[split] = value
                row = dict(mode=mode, seed=seed, split=split, tsp_updates=updates,
                           total_updates=finished["total_updates"], skipped_updates=sum(finished["skipped"].values()))
                for metric in METRICS:
                    row[metric] = history[-1][split][metric]
                    row["baseline_" + metric] = history[0][split][metric]
                    row["last5_" + metric] = float(np.mean([r[split][metric] for r in history[-5:]]))
                rows.append(row)
        paths = [root / f"{mode}_seed{seed}" for mode in ("joint", "tsp_only")]
        configs = [json.loads((p / "args.json").read_text()) for p in paths]
        for key in ("base_sha256", "optimizer_initial_hash", "model_initial_hash", "scaler_initial", "model_params"):
            if configs[0][key] != configs[1][key]:
                raise RuntimeError(f"Initial states differ: seed={seed}, {key}")
        traces = [[json.loads(line) for line in (p / "tsp_update_trace.jsonl").read_text().splitlines()] for p in paths]
        if len(traces[0]) != updates or len(traces[1]) != updates:
            raise RuntimeError("TSP trace length mismatch")
        for left, right in zip(*traces):
            for key in ("tsp_update", "indices_hash", "dropout_rng_before", "dropout_rng_after"):
                if left[key] != right[key]:
                    raise RuntimeError(f"Paired random streams diverged: seed={seed}, update={left['tsp_update']}, key={key}")
        checks[str(seed)] = dict(initial_model_optimizer_scaler_equal=True, tsp_batches_equal=True,
                                tsp_dropout_streams_equal=True, baseline_predictions_equal=True, successful_updates=updates)
        paired[str(seed)] = {}
        for split in ("train", "val"):
            with np.load(paths[0] / f"predictions_{split}_u{updates:03d}.npz") as a, np.load(paths[1] / f"predictions_{split}_u{updates:03d}.npz") as b:
                for key in ("indices", "winner", "costs", "pool_ids"):
                    if not np.array_equal(a[key], b[key]):
                        raise RuntimeError(f"Prediction alignment mismatch: {seed}/{split}/{key}")
                a_ok, b_ok = a["pred"] == a["winner"], b["pred"] == b["winner"]
                paired[str(seed)][split] = dict(n=len(a_ok), corrected=int((~a_ok & b_ok).sum()),
                    harmed=int((a_ok & ~b_ok).sum()), changed=int((a["pred"] != b["pred"]).sum()),
                    top1_delta_pp=float((b_ok.mean() - a_ok.mean()) * 100),
                    mean_cost_delta=float((b["selected_cost"].astype(float) - a["selected_cost"].astype(float)).mean()))
    with (root / "comparison.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    dump(root / "paired_comparison.json", paired)
    dump(root / "verification.json", checks)
    plot_comparison(root, histories, seeds)
    write_comparison(root, histories, rows, paired, seeds, updates)
    print("[summary]", root / "comparison.md", flush=True)


def plot_comparison(root, histories, seeds):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for metric in ("top1", "ce", "mean_cost"):
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
        for ax, split in zip(axes, ("train", "val")):
            for mode, color in (("joint", "tab:blue"), ("tsp_only", "tab:orange")):
                values = np.array([[r[split][metric] for r in histories[(mode, seed)]] for seed in seeds])
                x = [r["tsp_updates"] for r in histories[(mode, seeds[0])]]
                for value in values:
                    ax.plot(x, value, color=color, alpha=.2, linewidth=1)
                mean, std = values.mean(0), values.std(0, ddof=1) if len(seeds) > 1 else np.zeros(values.shape[1])
                ax.plot(x, mean, color=color, label=mode + " mean (3 continuation seeds)")
                ax.fill_between(x, mean - std, mean + std, color=color, alpha=.15)
            ax.axhline(histories[("joint", seeds[0])][0][split][metric], linestyle="--", color="gray", label="R34 start")
            ax.set(title=f"TSP {split}: {metric}", xlabel="Successful TSP updates", ylabel=metric)
            ax.grid(alpha=.25)
            ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(root / f"comparison_{metric}.png", dpi=160)
        plt.close(fig)


def write_comparison(root, histories, rows, paired, seeds, updates):
    lines = ["# R35: TSP joint / single-task continuation", "",
             "All runs start from R34 winner-cost best.pt (epoch 50), restoring the same model, optimizer and AMP scaler.",
             "The model, native labels, pools and winner-cost loss are unchanged. LR is fixed at 2e-5 after restoring AdamW.",
             "No test split is loaded. Seeds 2/3/4 are continuation repeats from ONE pretrained checkpoint, not independent training runs.",
             "A complete joint round updates every problem once; evaluation follows the entire round. TSP sampling and dropout streams are paired.", "",
             "## Endpoint: 300 successful TSP updates", "",
             "| Mode | Seed | Total updates | Train CE | Train top1 | Train cost | Val CE | Val top1 | Val cost | Val vs_SBS |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    start = histories[("joint", seeds[0])][0]
    lines.append(f"| R34 start | - | 0 | {start['train']['ce']:.5f} | {start['train']['top1']:.2%} | {start['train']['mean_cost']:.6f} | {start['val']['ce']:.5f} | {start['val']['top1']:.2%} | {start['val']['mean_cost']:.6f} | {start['val']['vs_sbs_pct']:+.4f}% |")
    for mode in ("joint", "tsp_only"):
        for seed in seeds:
            record = histories[(mode, seed)][-1]
            a, b = record["train"], record["val"]
            lines.append(f"| {mode} | {seed} | {record['total_updates']} | {a['ce']:.5f} | {a['top1']:.2%} | {a['mean_cost']:.6f} | {b['ce']:.5f} | {b['top1']:.2%} | {b['mean_cost']:.6f} | {b['vs_sbs_pct']:+.4f}% |")
    lines += ["", "## Three-seed aggregates", "",
              "Mean +/- sample SD across continuation seeds. last5 is the arithmetic mean at TSP updates 180, 210, 240, 270, 300.", "",
              "| Window | Split | Mode | CE | top1 | mean_cost | vs_SBS |",
              "| --- | --- | --- | ---: | ---: | ---: | ---: |"]
    for window in ("endpoint", "last5"):
        prefix = "" if window == "endpoint" else "last5_"
        for split in ("train", "val"):
            for mode in ("joint", "tsp_only"):
                values = [r for r in rows if r["mode"] == mode and r["split"] == split]
                cells = []
                for metric in ("ce", "top1", "mean_cost", "vs_sbs_pct"):
                    data = np.array([r[prefix + metric] for r in values])
                    factor = 100 if metric == "top1" else 1
                    suffix = "%" if metric in ("top1", "vs_sbs_pct") else ""
                    cells.append(f"{data.mean() * factor:.5f} +/- {data.std(ddof=1) * factor:.5f}{suffix}")
                lines.append(f"| {window} | {split} | {mode} | " + " | ".join(cells) + " |")
    lines += ["", "## Paired differences: TSP-only minus joint", "",
              "Positive top1 and negative cost deltas favor TSP-only. corrected/harmed compare exactly the same instances.", "",
              "| Seed | Split | top1 delta (pp) | cost delta | Corrected | Harmed | Changed predictions |",
              "| --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    for seed in seeds:
        for split, result in paired[str(seed)].items():
            lines.append(f"| {seed} | {split} | {result['top1_delta_pp']:+.4f} | {result['mean_cost_delta']:+.6f} | {result['corrected']} | {result['harmed']} | {result['changed']} |")
    for split in ("train", "val"):
        a = [r for r in rows if r["mode"] == "joint" and r["split"] == split]
        b = [r for r in rows if r["mode"] == "tsp_only" and r["split"] == split]
        delta = np.array([right["last5_top1"] - left["last5_top1"] for left, right in zip(a, b)])
        costs = np.array([right["last5_mean_cost"] - left["last5_mean_cost"] for left, right in zip(a, b)])
        lines += ["", f"{split} last5: TSP-only minus joint top1 = {delta.mean() * 100:+.4f} pp; cost = {costs.mean():+.6f}; top1 improves in {int((delta > 0).sum())}/{len(seeds)} seeds."]
    lines += ["", "## Scope and interpretation", "",
              "This probe estimates the effect of interleaved task updates on TSP over 300 TSP updates, not equal-compute performance.",
              "Restored Adam moments are shared; other-task updates can affect parameters and optimizer moments. This is the intended treatment.",
              "Three continuation seeds alone do not establish statistical significance or a conclusion for other tasks.",
              "Use endpoint AND last5 train/val behavior, with cost, to decide whether a later adapter experiment is justified.",
              "Model/optimizer/scaler identity, TSP index and dropout alignment, and complete baseline predictions are checked in verification.json.",
              "W&B records are offline. No adapters, solver feature changes, attention changes or test-based model selection were added.", "",
              "![top1](comparison_top1.png)", "![CE](comparison_ce.png)", "![Cost](comparison_mean_cost.png)"]
    (root / "comparison.md").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--base", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--seeds", default="2,3,4")
    parser.add_argument("--updates", type=int, default=300)
    parser.add_argument("--eval-interval", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=640)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--continue-queue", action="store_true")
    args = parser.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    if args.summarize_only:
        summarize(args.root, seeds, args.updates, args.eval_interval)
        return
    if args.updates <= 0 or args.eval_interval <= 0 or args.updates % args.eval_interval or args.updates // args.eval_interval < 5:
        parser.error("Need at least five regular evaluation intervals")
    torch.set_num_threads(1)
    configure_torch()
    checkpoint = torch.load(args.base, map_location="cpu", weights_only=False)
    original_optimizer_hash = state_hash(checkpoint["optimizer"])
    config = checkpoint["args"]
    if config["loss_mode"] != "winner_cost" or checkpoint["epoch"] != 49:
        raise ValueError("Expected R34 winner-cost epoch-50 checkpoint")
    if config["model_params"]["solver_feature_spec"]["solver_names"] != GLOBAL_SOLVERS:
        raise ValueError("Global solver IDs differ from R34")
    for name in ("ce_weight", "pair_weight", "risk_weight", "cost_scale", "loss_mode"):
        setattr(args, name, config[name])
    args.root.mkdir(parents=True, exist_ok=True)
    if not args.continue_queue and any((args.root / f"{mode}_seed{seed}").exists() for seed in seeds for mode in ("joint", "tsp_only")):
        raise ValueError("Refusing to overwrite an existing continuation")
    protected = {name: file_hash(Path(__file__).parent / name) for name in PROTECTED}
    manifest = {}
    for problem, split in [(p, "train") for p in PROBLEMS] + [("TSP", "val")]:
        directory = DATA_ROOT / f"{problem}{split}"
        manifest[f"{problem}/{split}"] = dict(pool=POOLS[problem], **{name: file_hash(directory / name) for name in ("dataset.pkl", "raw_label.pkl")})
    manifest_payload = dict(checkpoint=str(args.base), checkpoint_sha256=file_hash(args.base),
         original_optimizer_hash=original_optimizer_hash, protected_model_hashes=protected, data=manifest,
         seeds=seeds, updates=args.updates, eval_interval=args.eval_interval, task_order=PROBLEMS)
    manifest_path = args.root / "manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest_payload:
        raise ValueError("Study checkpoint, data or protocol changed")
    dump(manifest_path, manifest_payload)
    loaders = {p: make_loader(p, "train", args.batch_size, 0, coord_augment=0, shuffle=False, cache_device=args.device) for p in PROBLEMS}
    val_loader = make_loader("TSP", "val", args.batch_size, 0, coord_augment=0, shuffle=False, cache_device=args.device)
    for seed in seeds:
        for mode in ("joint", "tsp_only"):
            directory = args.root / f"{mode}_seed{seed}"
            if args.continue_queue and directory.exists():
                complete = json.loads((directory / "finished.json").read_text())
                if complete["tsp_updates"] != args.updates or complete["total_updates"] != args.updates * (18 if mode == "joint" else 1):
                    raise ValueError("Cannot skip an incomplete continuation")
                print(f"[skip completed] {directory.name}", flush=True)
                continue
            run_one(mode, seed, args, checkpoint, loaders, val_loader)
            if state_hash(checkpoint["optimizer"]) != original_optimizer_hash:
                raise RuntimeError("Source optimizer state was mutated")
    if protected != {name: file_hash(Path(__file__).parent / name) for name in PROTECTED}:
        raise RuntimeError("Protected model source changed during R35")
    summarize(args.root, seeds, args.updates, args.eval_interval)
    print("[R35 COMPLETE] six continuations and train/val comparison saved", flush=True)


if __name__ == "__main__":
    main()
