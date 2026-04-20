"""Specialist-selector baseline: train one selector per problem (restricted to its local pool).

For each of the 18 problems, train an independent selector that only ever sees that problem's
instances and only scores its own pool's solvers. Uses the same backbone/loss/lr/wd as the unified.

Usage:
    python -m code.unified_selector.specialist_train \
        --epochs 4 --batch 32 --out code/unified_selector/runs/specialists \
        --device cuda:0 --wandb

Writes per-problem: specialists/<problem>/best.pt, eval.json (val+test).
At the end, aggregates per-problem results into specialists/aggregate.json and prints macro.
"""
from __future__ import annotations
import argparse, json, math, random, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .registry import PROBLEMS, P2I, M_GLOBAL, K_CBITS, D_COORD
from .data import UnifiedProblemDataset, collate_single_problem
from .model import CoordEncoder, MatrixEncoder


class SpecialistSelector(nn.Module):
    """Per-problem selector: same backbone, but pool-sized head (no global solver table)."""
    def __init__(self, K_p: int, d: int = 128, depth: int = 4, dropout: float = 0.1,
                 head_hidden: int = 256, is_matrix: bool = False):
        super().__init__()
        self.K_p = K_p
        self.is_matrix = is_matrix
        self.coord_enc = CoordEncoder(d_in_max=8, d=d, depth=depth, dropout=dropout)
        self.matrix_enc = MatrixEncoder(d=d)
        self.solver_emb = nn.Embedding(K_p, d)
        self.head = nn.Sequential(
            nn.Linear(4*d, head_hidden), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, 1),
        )

    def forward(self, batch):
        if batch["kind"] == "coord":
            g = self.coord_enc(batch["node"], batch["node_mask"])
        else:
            g = self.matrix_enc(batch["matrix"], batch["node_mask"])
        B = g.shape[0]
        e = self.solver_emb.weight
        h_exp = g.unsqueeze(1).expand(B, self.K_p, g.shape[-1])
        e_exp = e.unsqueeze(0).expand(B, self.K_p, g.shape[-1])
        z = torch.cat([h_exp, e_exp, h_exp * e_exp, (h_exp - e_exp).abs()], dim=-1)
        return self.head(z).squeeze(-1)  # (B, K_p)


def regret_soft(costs, tau=0.02, eps=1e-6, tie_tol=1e-3):
    c_star = costs.min(dim=1, keepdim=True).values
    r = (costs - c_star) / (c_star.abs() + eps)
    r = torch.where(r < tie_tol, torch.zeros_like(r), r)
    return torch.softmax(-r / tau, dim=1)


def train_one_problem(problem: str, args, audit, device, wandb_run=None):
    K_p = len(audit[problem]["pool"])
    is_matrix = (problem == "ATSP")
    ds_tr = UnifiedProblemDataset(problem, "train")
    ds_va = UnifiedProblemDataset(problem, "val")
    ds_te = UnifiedProblemDataset(problem, "test")
    dl_tr = DataLoader(ds_tr, batch_size=args.batch, shuffle=True, num_workers=2,
                        collate_fn=collate_single_problem, drop_last=True)
    dl_va = DataLoader(ds_va, batch_size=args.batch, shuffle=False, num_workers=2,
                        collate_fn=collate_single_problem)
    dl_te = DataLoader(ds_te, batch_size=args.batch, shuffle=False, num_workers=2,
                        collate_fn=collate_single_problem)

    model = SpecialistSelector(K_p=K_p, d=args.d, depth=args.depth, dropout=args.dropout,
                               is_matrix=is_matrix).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd, betas=(0.9, 0.95))
    total_steps = args.epochs * max(1, len(dl_tr))

    def lr_at(step):
        if step < args.warmup:
            return args.lr * step / max(1, args.warmup)
        t = (step - args.warmup) / max(1, total_steps - args.warmup)
        return 3e-5 + (args.lr - 3e-5) * 0.5 * (1 + math.cos(math.pi * min(1.0, t)))

    step = 0
    best_vs_sbs = 1e9
    best_path = Path(args.out) / problem / "best.pt"
    best_path.parent.mkdir(parents=True, exist_ok=True)
    for ep in range(args.epochs):
        model.train()
        for b in dl_tr:
            for k, v in b.items():
                if torch.is_tensor(v): b[k] = v.to(device, non_blocking=True)
            logits = model(b)                   # (B, K_p)
            q = regret_soft(b["costs"], tau=args.tau)
            loss = -(q * F.log_softmax(logits, dim=1)).sum(dim=1).mean()
            for g in opt.param_groups:
                g["lr"] = lr_at(step)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            step += 1
        # eval val
        model.eval()
        with torch.no_grad():
            cost_sum = 0.0; n = 0; top1 = 0
            for b in dl_va:
                for k, v in b.items():
                    if torch.is_tensor(v): b[k] = v.to(device)
                logits = model(b)
                pred = logits.argmax(dim=1)
                sel = b["costs"].gather(1, pred.unsqueeze(1)).squeeze(1)
                best = b["costs"].min(dim=1).values
                best_idx = b["costs"].argmin(dim=1)
                top1 += int((pred == best_idx).sum().item())
                cost_sum += float(sel.sum()); n += sel.shape[0]
        val_mean = cost_sum / max(1, n)
        sbs = audit[problem]["val"]["sbs_mean"]
        vs_sbs = (val_mean - sbs) / (abs(sbs) + 1e-9) * 100
        print(f"  [{problem}] ep{ep} top1={top1/n:.3f} mc={val_mean:.4f} vs_sbs={vs_sbs:+.3f}%", flush=True)
        if vs_sbs < best_vs_sbs:
            best_vs_sbs = vs_sbs
            torch.save({"model": model.state_dict(), "K_p": K_p}, best_path)
        if wandb_run is not None:
            wandb_run.log({f"{problem}/val_top1": top1/n, f"{problem}/val_vs_sbs": vs_sbs, f"{problem}_step": step})

    # Final eval test (using best ckpt)
    ck = torch.load(best_path, map_location=device)
    model.load_state_dict(ck["model"]); model.eval()
    with torch.no_grad():
        cost_sum = 0.0; n = 0; top1 = 0
        for b in dl_te:
            for k, v in b.items():
                if torch.is_tensor(v): b[k] = v.to(device)
            logits = model(b)
            pred = logits.argmax(dim=1)
            sel = b["costs"].gather(1, pred.unsqueeze(1)).squeeze(1)
            best_idx = b["costs"].argmin(dim=1)
            top1 += int((pred == best_idx).sum().item())
            cost_sum += float(sel.sum()); n += sel.shape[0]
    test_mean = cost_sum / max(1, n)
    sbs_test = audit[problem]["test"]["sbs_mean"]
    test_vs_sbs = (test_mean - sbs_test) / (abs(sbs_test) + 1e-9) * 100
    return dict(
        problem=problem, K_p=K_p,
        val_best_vs_sbs_pct=best_vs_sbs,
        test_top1=top1/n, test_mean_cost=test_mean, test_sbs=sbs_test,
        test_vs_sbs_pct=test_vs_sbs,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--tau", type=float, default=0.02)
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--dropout-prob", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--audit", type=str, default="code/unified_selector/runs/audit.json")
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--problems", nargs="*", default=PROBLEMS)
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    Path(args.out).mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    audit = json.loads(Path(args.audit).read_text())

    run = None
    if args.wandb:
        try:
            import wandb
            run = wandb.init(project="selector",
                             entity="yjkds-southern-university-of-science-technology",
                             name=f"specialists_seed{args.seed}",
                             tags=["specialist-baseline"], config=vars(args))
        except Exception as e:
            print(f"[wandb] disabled {e}")

    results = {}
    t0 = time.time()
    for p in args.problems:
        print(f"\n=== Training specialist: {p} ===", flush=True)
        results[p] = train_one_problem(p, args, audit, device, run)

    macro_vs_sbs = float(np.mean([r["test_vs_sbs_pct"] for r in results.values()]))
    macro_top1 = float(np.mean([r["test_top1"] for r in results.values()]))
    out = Path(args.out) / "aggregate.json"
    out.write_text(json.dumps({"per_problem": results, "macro_vs_sbs_test_pct": macro_vs_sbs,
                                "macro_top1_test": macro_top1}, indent=2))
    elapsed = (time.time() - t0) / 60
    print(f"\n[specialist done] {elapsed:.1f} min macro_test_top1={macro_top1:.3f} macro_test_vs_sbs={macro_vs_sbs:+.3f}%")
    if run is not None:
        run.log({"specialists/macro_test_top1": macro_top1, "specialists/macro_test_vs_sbs": macro_vs_sbs})
        run.finish()


if __name__ == "__main__":
    main()
