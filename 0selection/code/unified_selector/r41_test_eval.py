"""Pre-registered test-split evaluation for a UnifiedSelector checkpoint
vs the R18 baseline.

Val-locked: no hyperparameter tuning on test. Reports:
- Per-problem and macro top1
- Δ vs R18 baseline
- 10k bootstrap CI for macro top1 Δ
- Zero-pick mass

Usage:
  python -m code.unified_selector.r41_test_eval \
    --ckpt code/unified_selector/runs/R41_R18_plus_local/best_top1.pt \
    --base-ckpt code/unified_selector/runs/R18_alltail/best_top1.pt \
    --out code/unified_selector/runs/R41_R18_plus_local/test_eval \
    --device cuda:1
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .registry import PROBLEMS, POOLS, S2I, M_GLOBAL
from .data import UnifiedProblemDataset, collate_single_problem
from .model import UnifiedSelector, _migrate_state_dict


def load_ckpt(path, device):
    ck = torch.load(path, map_location=device, weights_only=False)
    sd = _migrate_state_dict(ck["model"])
    args = ck.get("args", {})
    model = UnifiedSelector(
        d=args.get("d", 128),
        depth=args.get("depth", 4),
        dropout=args.get("dropout", 0.1),
        use_film=args.get("use_film", False),
        encoder_type=args.get("encoder_type", "standard"),
        rezero=args.get("rezero", False),
        block_num=args.get("block_num", 2),
        encoder_layer_num=args.get("encoder_layer_num", 2),
        heads=args.get("heads", 4),
        downsample_ratio=args.get("downsample_ratio", 0.8),
        local_head=args.get("local_head", False),
        residual_adapter=args.get("residual_adapter", False),
    ).to(device)
    model.load_state_dict(sd, strict=False)
    model.eval()
    return model


@torch.no_grad()
def infer_picks(model, split, device, batch_size=16):
    """Returns per-problem pick_rank (0=best, 1=2nd, ...) as np array indexed
    flatly across all test instances in fixed problem order.
    Also returns:
      - picks (pool-local arm index)
      - pick_rank (oracle rank of the pick)
      - sel_cost (cost of the pick)
      - oracle (oracle arm, pool-local)
    """
    all_picks = {}
    all_pick_rank = {}
    all_sel_cost = {}
    all_oracle = {}
    for p in PROBLEMS:
        ds = UnifiedProblemDataset(p, split, aug_8fold=False)
        dl = DataLoader(ds, batch_size=batch_size, collate_fn=collate_single_problem,
                        num_workers=0, shuffle=False)
        picks = []
        pick_ranks = []
        sel_costs = []
        oracles = []
        for b in dl:
            for k, v in b.items():
                if torch.is_tensor(v):
                    b[k] = v.to(device)
            logits = model(b)
            pool_ids = b["pool_ids"]
            costs = b["costs"]
            log_p_pool = F.log_softmax(logits, dim=1)[:, pool_ids]
            pred = log_p_pool.argmax(dim=1)
            true_rank = torch.argsort(costs, dim=1)
            inv_rank = torch.argsort(true_rank, dim=1)
            pick_rank = inv_rank.gather(1, pred.unsqueeze(1)).squeeze(1)
            sel_cost = costs.gather(1, pred.unsqueeze(1)).squeeze(1)
            oracle = costs.argmin(dim=1)
            picks.append(pred.cpu().numpy())
            pick_ranks.append(pick_rank.cpu().numpy())
            sel_costs.append(sel_cost.cpu().numpy())
            oracles.append(oracle.cpu().numpy())
        all_picks[p] = np.concatenate(picks)
        all_pick_rank[p] = np.concatenate(pick_ranks)
        all_sel_cost[p] = np.concatenate(sel_costs)
        all_oracle[p] = np.concatenate(oracles)
    return all_picks, all_pick_rank, all_sel_cost, all_oracle


def macro_top1(pick_rank):
    return float(np.mean([np.mean(pick_rank[p] == 0) for p in PROBLEMS]))


def bootstrap_paired(method_rank, base_rank, n_boot=10000, seed=0):
    """Paired bootstrap over instances for macro top1 Δ (method - base).
    Resamples within each problem independently, averages macro.
    """
    rng = np.random.RandomState(seed)
    deltas = np.zeros(n_boot)
    Ns = {p: len(method_rank[p]) for p in PROBLEMS}
    for b in range(n_boot):
        m_macro = 0.0
        b_macro = 0.0
        for p in PROBLEMS:
            idx = rng.randint(0, Ns[p], size=Ns[p])
            m_macro += float(np.mean(method_rank[p][idx] == 0))
            b_macro += float(np.mean(base_rank[p][idx] == 0))
        deltas[b] = (m_macro - b_macro) / len(PROBLEMS)
    lo = float(np.percentile(deltas, 2.5))
    hi = float(np.percentile(deltas, 97.5))
    p_gt_0 = float(np.mean(deltas > 0))
    return {"mean": float(np.mean(deltas)), "lo": lo, "hi": hi, "p_gt_0": p_gt_0,
            "significant": bool(lo > 0)}


def zero_pick_mass(method_picks, base_picks):
    """Support-based zero-pick definition (R29/R39).

    For each problem:
      base_support[p] = set(base_picks[p])  # arms base ever selected on test
      oracle_arm[i]   = argmin(costs[i])
      zp_instance     = (oracle_arm not in base_support)
    Report:
      base_zp_mass  — fraction of test instances in the zero-pick subset
      method_zp_mass — same subset, for method's picks (should == base_zp_mass)
      rescue        — #{zp_instance and method_pick == oracle_arm}
      harm          — #{not zp_instance and base_pick == oracle and method_pick != oracle}
      net           — rescue - harm
    """
    total_n = 0
    base_zp_n = 0
    rescue_n = 0
    harm_n = 0
    # method_picks[p] is a dict with 'picks' and 'oracle' (numpy arrays) OR it is picks+cost
    # so let's be liberal and recover oracle from costs
    per_problem = {}
    for p in PROBLEMS:
        m_picks = method_picks[p]
        b_picks = base_picks[p]
        m_arr = m_picks["picks"] if isinstance(m_picks, dict) else m_picks
        b_arr = b_picks["picks"] if isinstance(b_picks, dict) else b_picks
        oracle = m_picks["oracle"] if isinstance(m_picks, dict) else None
        if oracle is None:
            # Fall back to per-instance argmin of costs
            costs = m_picks["costs"] if isinstance(m_picks, dict) else None
            if costs is None:
                raise ValueError("zero_pick_mass needs oracle or costs")
            oracle = costs.argmin(axis=1)
        base_support = set(np.unique(b_arr).tolist())
        zp_mask = ~np.isin(oracle, list(base_support))
        n = len(oracle)
        zp = int(zp_mask.sum())
        rescue = int(((m_arr == oracle) & zp_mask).sum())
        non_zp = ~zp_mask
        base_correct = (b_arr == oracle)
        method_correct = (m_arr == oracle)
        harm = int((non_zp & base_correct & ~method_correct).sum())
        per_problem[p] = {
            "n": n, "zp": zp,
            "rescue": rescue, "harm": harm, "net": rescue - harm,
            "zp_frac": zp / max(1, n),
        }
        total_n += n
        base_zp_n += zp
        rescue_n += rescue
        harm_n += harm
    return {
        "total_n": total_n,
        "base_zp_n": base_zp_n,
        "base_zp_frac": base_zp_n / max(1, total_n),
        "rescue_total": rescue_n,
        "harm_total": harm_n,
        "net_total": rescue_n - harm_n,
        "per_problem": per_problem,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--base-ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    print(f"[eval] loading method checkpoint: {args.ckpt}")
    m_model = load_ckpt(args.ckpt, device)
    print(f"[eval] loading base checkpoint:   {args.base_ckpt}")
    b_model = load_ckpt(args.base_ckpt, device)

    print(f"[eval] inferring test picks (method)")
    m_picks, m_rank, m_cost, m_oracle = infer_picks(m_model, "test", device)
    print(f"[eval] inferring test picks (base)")
    b_picks, b_rank, b_cost, b_oracle = infer_picks(b_model, "test", device)

    # Per-problem top1
    per_problem = {}
    for p in PROBLEMS:
        per_problem[p] = {
            "base_top1": float(np.mean(b_rank[p] == 0)),
            "method_top1": float(np.mean(m_rank[p] == 0)),
            "delta_top1": float(np.mean(m_rank[p] == 0) - np.mean(b_rank[p] == 0)),
            "base_mean_cost": float(np.mean(b_cost[p])),
            "method_mean_cost": float(np.mean(m_cost[p])),
            "n": int(len(m_rank[p])),
        }
    macro_base = float(np.mean([per_problem[p]["base_top1"] for p in PROBLEMS]))
    macro_method = float(np.mean([per_problem[p]["method_top1"] for p in PROBLEMS]))

    print(f"[eval] bootstrap {args.n_boot} iterations")
    boot = bootstrap_paired(m_rank, b_rank, n_boot=args.n_boot, seed=0)

    print(f"[eval] zero-pick mass (R29/R39 base-support definition)")
    method_dict = {p: {"picks": m_picks[p], "oracle": m_oracle[p]} for p in PROBLEMS}
    base_dict = {p: {"picks": b_picks[p], "oracle": b_oracle[p]} for p in PROBLEMS}
    zp = zero_pick_mass(method_dict, base_dict)

    summary = {
        "ckpt": args.ckpt,
        "base_ckpt": args.base_ckpt,
        "macro_base_top1": macro_base,
        "macro_method_top1": macro_method,
        "macro_delta": macro_method - macro_base,
        "bootstrap": boot,
        "zero_pick": zp,
        "per_problem": per_problem,
    }
    (out / "test_eval.json").write_text(json.dumps(summary, indent=2, default=float))

    # Markdown report
    lines = [
        f"# Test-split evaluation: {Path(args.ckpt).parent.name}",
        f"",
        f"- Base: `{args.base_ckpt}` → macro_top1 = **{macro_base:.4f}**",
        f"- Method: `{args.ckpt}` → macro_top1 = **{macro_method:.4f}**",
        f"- Δ = **{macro_method - macro_base:+.4f}**  (95% CI [{boot['lo']:+.4f}, {boot['hi']:+.4f}], "
        f"p(Δ>0) = {boot['p_gt_0']:.3f}, significant = **{boot['significant']}**)",
        f"",
        f"- Zero-pick mass (base support): {zp['base_zp_frac']*100:.2f}% of test instances "
        f"({zp['base_zp_n']} / {zp['total_n']})",
        f"- Rescue: {zp['rescue_total']} | Harm: {zp['harm_total']} | Net: {zp['net_total']:+d}",
        "",
        "## Per-problem",
        "| Problem | Base top1 | Method top1 | Δ | zp | rescue | harm | net |",
        "|:---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for p in PROBLEMS:
        v = per_problem[p]
        zpp = zp["per_problem"][p]
        lines.append(
            f"| {p} | {v['base_top1']:.3f} | {v['method_top1']:.3f} | {v['delta_top1']:+.3f} "
            f"| {zpp['zp']} | {zpp['rescue']} | {zpp['harm']} | {zpp['net']:+d} |"
        )
    (out / "test_eval.md").write_text("\n".join(lines))
    print("\n" + "\n".join(lines))


if __name__ == "__main__":
    main()
