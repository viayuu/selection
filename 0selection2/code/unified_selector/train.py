"""Train the unified selector across 18 problems.

Usage:
    python -m code.unified_selector.train --epochs 20 --batch-per-problem 8 \
           --lr 3e-4 --tau 0.02 --dropout 0.1 --problem-dropout 0.25 \
           --save-dir code/unified_selector/runs/R1_seed0 --seed 0 --wandb

Training loop: each gradient step = one single-problem microbatch of
BATCH_PER_PROBLEM instances, cycling through all 18 problems in a fixed
shuffle order per epoch (= balanced per-problem exposure, oracle-recommended).
"""
from __future__ import annotations
import argparse, os, json, random, time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .registry import PROBLEMS, P2I
from .data import UnifiedProblemDataset, collate_single_problem
from .model import UnifiedSelector, regret_soft_targets, masked_listwise_ce, predict_cost, combined_cost_loss


# Oversampling weights (fix 3) — more weight on under-performing problems.
PROBLEM_WEIGHT = {
    "CVRP": 4.0, "VRPB": 3.0, "OVRPB": 3.0, "OVRP": 2.0,
    "VRPBL": 1.5, "VRPBTW": 1.5, "OVRPBL": 1.5, "OVRPBTW": 1.5,
    "VRPBLTW": 1.5, "OVRPBLTW": 1.5,
}


def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def build_loaders(split: str, problems, batch_per_problem: int, shuffle: bool, num_workers: int = 2):
    loaders = {}
    for p in problems:
        ds = UnifiedProblemDataset(p, split)
        loaders[p] = DataLoader(
            ds, batch_size=batch_per_problem, shuffle=shuffle, num_workers=num_workers,
            collate_fn=collate_single_problem, drop_last=shuffle, persistent_workers=(num_workers>0),
        )
    return loaders


def to_device(batch, device):
    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device, non_blocking=True)
        else:
            out[k] = v
    return out


def evaluate(model, loaders_val, device, tau: float, audit_vbs_sbs: dict):
    model.eval()
    per_p = {}
    with torch.no_grad():
        for p, loader in loaders_val.items():
            top1 = top2 = top3 = n_tot = 0
            cost_sum = 0.0
            pool_ids_ref = None
            for batch in loader:
                batch = to_device(batch, device)
                logits = model(batch)  # (B, M)
                log_p = torch.log_softmax(logits, dim=1)
                pool_ids = batch["pool_ids"]
                pool_ids_ref = pool_ids
                log_p_pool = log_p[:, pool_ids]  # (B, K_p)
                # Ranking over pool for top-k accuracy vs. true oracle ranking
                costs = batch["costs"]
                true_rank = torch.argsort(costs, dim=1)  # best first
                pred_rank = torch.argsort(-log_p_pool, dim=1)  # best first
                best_true = true_rank[:, 0]
                top1 += (pred_rank[:, 0] == best_true).sum().item()
                top2 += ((pred_rank[:, 0] == best_true) | (pred_rank[:, 1] == best_true)).sum().item() if pred_rank.shape[1] > 1 else top2
                # top3:
                if pred_rank.shape[1] > 2:
                    top3 += ((pred_rank[:, 0] == best_true) | (pred_rank[:, 1] == best_true) | (pred_rank[:, 2] == best_true)).sum().item()
                else:
                    top3 += pred_rank.shape[0]
                # Selected cost
                sel = pred_rank[:, 0]
                sel_cost = costs.gather(1, sel.unsqueeze(1)).squeeze(1)
                cost_sum += sel_cost.sum().item()
                n_tot += costs.shape[0]
            sbs = audit_vbs_sbs[p]["val"]["sbs_mean"]
            vbs = audit_vbs_sbs[p]["val"]["vbs_mean"]
            mean_cost = cost_sum / max(1, n_tot)
            per_p[p] = {
                "top1": top1 / n_tot, "top2": top2 / n_tot, "top3": top3 / n_tot,
                "mean_cost": mean_cost, "sbs": sbs, "vbs": vbs,
                "vs_sbs_pct": (mean_cost - sbs) / (abs(sbs) + 1e-9) * 100,
                "vbs_gap_closed_pct": ((sbs - mean_cost) / (sbs - vbs + 1e-9)) * 100 if sbs != vbs else 0.0,
                "n": n_tot,
            }
    # Macro
    macro_top1 = sum(v["top1"] for v in per_p.values()) / len(per_p)
    macro_vs_sbs = sum(v["vs_sbs_pct"] for v in per_p.values()) / len(per_p)
    macro_vbs_closed = sum(v["vbs_gap_closed_pct"] for v in per_p.values()) / len(per_p)
    model.train()
    return per_p, {"macro_top1": macro_top1, "macro_vs_sbs_pct": macro_vs_sbs, "macro_vbs_gap_closed_pct": macro_vbs_closed}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-per-problem", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--warmup-steps", type=int, default=1500)
    ap.add_argument("--cosine-total", type=int, default=40000)
    ap.add_argument("--tau", type=float, default=0.02)
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--problem-dropout", type=float, default=0.25)
    ap.add_argument("--no-fact", action="store_true")
    ap.add_argument("--save-dir", type=str, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--hold-out", type=str, nargs="*", default=[],
                    help="MVRP bitvector-based problems to hold out (for zero-shot exp).")
    ap.add_argument("--audit", type=str, default="code/unified_selector/runs/audit.json")
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--wandb-project", type=str, default="selector")
    ap.add_argument("--wandb-entity", type=str, default="yjkds-southern-university-of-science-technology")
    ap.add_argument("--wandb-tag", type=str, default="R1")
    ap.add_argument("--eval-every-epoch", type=int, default=1)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--overfit", type=int, default=0, help="If >0, use first N instances of each problem and train for more epochs.")
    ap.add_argument("--meta-only", action="store_true", help="Zero-out instance features (R2 baseline).")
    ap.add_argument("--loss", choices=["soft", "combined"], default="soft")
    ap.add_argument("--no-problem-solver-bias", action="store_true")
    ap.add_argument("--oversample", action="store_true", help="Apply PROBLEM_WEIGHT oversampling.")
    args = ap.parse_args()

    set_seed(args.seed)
    save_dir = Path(args.save_dir); save_dir.mkdir(parents=True, exist_ok=True)
    (save_dir / "args.json").write_text(json.dumps(vars(args), indent=2))

    device = torch.device(args.device)
    problems_train = [p for p in PROBLEMS if p not in args.hold_out]
    problems_val_all = PROBLEMS
    print(f"[train] train problems: {problems_train}")
    print(f"[train] held-out: {args.hold_out}")

    # Load audit for SBS/VBS
    audit = json.loads(Path(args.audit).read_text())

    # Val loaders: load once, keep cached (1k each is manageable).
    val_loaders   = build_loaders("val",   problems_val_all, args.batch_per_problem, shuffle=False, num_workers=args.num_workers)
    # Train datasets: stream per-problem to keep RSS low.
    def _stream_train_loader(problem):
        ds = UnifiedProblemDataset(problem, "train")
        if args.overfit > 0:
            ds.instances = ds.instances[:args.overfit]
            ds.labels = {str(i): ds.labels[str(i)] for i in range(args.overfit)}
            ds.N = args.overfit
        return DataLoader(ds, batch_size=args.batch_per_problem, shuffle=True,
                          num_workers=args.num_workers, collate_fn=collate_single_problem, drop_last=True)

    model = UnifiedSelector(d=args.d, depth=args.depth, dropout=args.dropout,
                            use_mvrp_factorized=not args.no_fact,
                            use_problem_solver_bias=not args.no_problem_solver_bias).to(device)
    model.set_prob_dropout(args.problem_dropout)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd, betas=(0.9, 0.95))

    def lr_at(step):
        if step < args.warmup_steps:
            return args.lr * step / max(1, args.warmup_steps)
        t = (step - args.warmup_steps) / max(1, args.cosine_total - args.warmup_steps)
        t = min(1.0, t)
        import math
        return 3e-5 + (args.lr - 3e-5) * 0.5 * (1 + math.cos(math.pi * t))

    # WandB
    if args.wandb:
        try:
            import wandb
            wandb.init(project=args.wandb_project, entity=args.wandb_entity, name=Path(args.save_dir).name,
                       tags=[args.wandb_tag], config=vars(args))
        except Exception as e:
            print(f"[wandb] disabled ({e})")
            args.wandb = False

    step = 0
    best_macro_vs_sbs = 1e9
    loss_ema = None
    t0 = time.time()
    for epoch in range(args.epochs):
        # Stream each problem sequentially — load, train one epoch's worth, drop.
        epoch_order = problems_train.copy(); random.shuffle(epoch_order)
        for p in epoch_order:
            n_passes = int(PROBLEM_WEIGHT.get(p, 1.0)) if args.oversample else 1
            for _pass in range(n_passes):
                loader = _stream_train_loader(p)
                for batch in loader:
                    batch = to_device(batch, device)
                    if args.meta_only and batch["kind"] == "coord":
                        batch["node"] = torch.zeros_like(batch["node"])
                    elif args.meta_only and batch["kind"] == "matrix":
                        batch["matrix"] = torch.zeros_like(batch["matrix"])

                    logits = model(batch)
                    if args.loss == "combined":
                        sbs_idx = audit[p]["sbs_pool_idx"]
                        loss = combined_cost_loss(logits, batch["pool_ids"], batch["costs"],
                                                   sbs_pool_idx=sbs_idx, tau=0.07)
                    else:
                        soft = regret_soft_targets(batch["costs"], tau=args.tau)
                        loss = masked_listwise_ce(logits, batch["pool_ids"], soft)
                    for g in opt.param_groups:
                        g["lr"] = lr_at(step)
                    opt.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()
                    loss_ema = loss.item() if loss_ema is None else 0.98 * loss_ema + 0.02 * loss.item()
                    step += 1
                    if step % args.log_every == 0:
                        msg = f"ep{epoch} step{step} prob={p} lr={lr_at(step):.2e} loss={loss.item():.4f} ema={loss_ema:.4f}"
                        print(msg, flush=True)
                        if args.wandb:
                            import wandb; wandb.log({"train/loss": loss.item(), "train/loss_ema": loss_ema, "train/lr": lr_at(step), "step": step})
                del loader  # free per-problem train dataset memory

        if (epoch + 1) % args.eval_every_epoch == 0 or epoch == args.epochs - 1:
            per_p, macro = evaluate(model, val_loaders, device, args.tau, audit)
            print(f"[eval epoch {epoch}] macro_top1={macro['macro_top1']:.4f} vs_sbs={macro['macro_vs_sbs_pct']:+.3f}% vbs_closed={macro['macro_vbs_gap_closed_pct']:+.2f}%")
            for p in PROBLEMS:
                r = per_p[p]
                mark = ""
                if p in args.hold_out: mark = " [HELD-OUT]"
                print(f"   {p:>10}: top1={r['top1']:.3f} mean_cost={r['mean_cost']:.4f} (sbs={r['sbs']:.4f}) vs_sbs={r['vs_sbs_pct']:+.2f}%{mark}")
            # Save per-epoch
            (save_dir / f"eval_epoch{epoch}.json").write_text(json.dumps({"per_problem": per_p, "macro": macro}, indent=2))
            if args.wandb:
                import wandb
                wandb.log({"val/macro_top1": macro["macro_top1"], "val/macro_vs_sbs_pct": macro["macro_vs_sbs_pct"], "val/macro_vbs_closed": macro["macro_vbs_gap_closed_pct"], "epoch": epoch})
                for p, r in per_p.items():
                    wandb.log({f"val_{p}/top1": r["top1"], f"val_{p}/mean_cost": r["mean_cost"], f"val_{p}/vs_sbs_pct": r["vs_sbs_pct"], "epoch": epoch})
            if macro["macro_vs_sbs_pct"] < best_macro_vs_sbs:
                best_macro_vs_sbs = macro["macro_vs_sbs_pct"]
                torch.save({"model": model.state_dict(), "args": vars(args), "macro": macro}, save_dir / "best.pt")
                print(f"   -> saved best checkpoint (macro_vs_sbs={best_macro_vs_sbs:+.3f}%)")

    elapsed = time.time() - t0
    print(f"[done] {args.epochs} epochs in {elapsed/60:.1f} min, best macro_vs_sbs={best_macro_vs_sbs:+.3f}%")
    (save_dir / "summary.json").write_text(json.dumps({"elapsed_sec": elapsed, "best_macro_vs_sbs_pct": best_macro_vs_sbs}, indent=2))


if __name__ == "__main__":
    main()
