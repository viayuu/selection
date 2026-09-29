import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from code.unified_selector.data import UnifiedProblemDataset, collate_single_problem
from code.unified_selector.registry import POOLS, PROBLEMS

from .V4Model import ProblemToSolverSelector


def to_device(batch, device):
    return {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}


def make_loader(problem, split, batch_size, num_workers):
    ds = UnifiedProblemDataset(problem, split, coord_augment=0)
    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_single_problem,
        pin_memory=(num_workers > 0),
    )
    return ds, dl


def load_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    model_params = ckpt["args"]["model_params"]
    model = ProblemToSolverSelector(**model_params).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, ckpt


def support_shortlist_mask(support_logits, costs, topk=3):
    if support_logits is None:
        return None
    if support_logits.dim() == 2:
        support_logits = support_logits[:, None, :]
    kk = min(max(1, topk), support_logits.size(-1))
    idx = torch.sigmoid(support_logits).topk(kk, dim=2).indices
    shortlist = torch.zeros(costs.size(0), costs.size(1), dtype=torch.bool, device=costs.device)
    for g in range(support_logits.size(1)):
        shortlist.scatter_(1, idx[:, g], True)
    return shortlist


def support_recall_stats(support_logits, costs, topk=3):
    shortlist = support_shortlist_mask(support_logits, costs, topk=topk)
    if shortlist is None:
        return None
    true_rank = torch.argsort(costs, dim=1)
    best = true_rank[:, 0]
    true_top3 = true_rank[:, : min(3, costs.size(1))]
    return (
        shortlist.gather(1, best[:, None]).float().sum().item(),
        shortlist.gather(1, true_top3).any(dim=1).float().sum().item(),
        shortlist.any(dim=0).detach().cpu(),
        float(shortlist.float().sum(dim=1).mean().item()),
    )


def _init_comp():
    return {"top1": 0, "top2": 0, "top3": 0, "cost_sum": 0.0}


def evaluate_problem(model, problem, split, batch_size, num_workers, device, support_mask_sweep=False):
    ds, dl = make_loader(problem, split, batch_size, num_workers)
    pool_names = list(POOLS[problem])
    comp = {
        "final": _init_comp(),
        "pre": _init_comp(),
        "gap": _init_comp(),
        "utility": _init_comp(),
    }
    n_total = 0
    all_costs = []
    pick_count = None
    support_top1 = 0.0
    support_top3 = 0.0
    support_shortlist_size = 0.0
    support_arm_seen = None
    with torch.no_grad():
        for batch in dl:
            batch = to_device(batch, device)
            out = model(batch)
            scores = {
                "final": out["logits"],
                "pre": out["pre_score"],
                "gap": -out["pred_gap"],
                "utility": out["utility"],
            }
            if out.get("support_logits") is not None:
                support_logits = out["support_logits"]
                if support_logits.dim() == 2:
                    support_logits = support_logits[:, None, :]
                scores["support_g0"] = support_logits[:, 0]
                scores["support_max"] = support_logits.max(dim=1).values
                scores["support_mean"] = support_logits.mean(dim=1)
            if support_mask_sweep and out.get("support_logits") is not None:
                for topk in (1, 2, 3):
                    shortlist = support_shortlist_mask(out["support_logits"], batch["costs"], topk=topk)
                    for base_name in ("final", "pre", "gap", "utility", "support_g0", "support_max", "support_mean"):
                        scores[f"{base_name}_su{topk}"] = scores[base_name].masked_fill(
                            ~shortlist, torch.finfo(scores[base_name].dtype).min
                        )
            costs = batch["costs"]
            true_rank = torch.argsort(costs, dim=1)
            best = true_rank[:, 0]
            for name, score in scores.items():
                if name not in comp:
                    comp[name] = _init_comp()
                pred_rank = torch.argsort(-score, dim=1)
                pred = pred_rank[:, 0]
                comp[name]["top1"] += (pred == best).sum().item()
                comp[name]["top2"] += (pred_rank[:, : min(2, score.size(1))] == best[:, None]).any(dim=1).sum().item()
                comp[name]["top3"] += (pred_rank[:, : min(3, score.size(1))] == best[:, None]).any(dim=1).sum().item()
                comp[name]["cost_sum"] += costs.gather(1, pred[:, None]).sum().item()
            n_total += costs.size(0)
            all_costs.append(costs.detach().cpu().numpy())
            pred = torch.argsort(-scores["final"], dim=1)[:, 0]
            cur = torch.bincount(pred.detach().cpu(), minlength=scores["final"].size(1)).float()
            pick_count = cur if pick_count is None else pick_count + cur
            support_stats = support_recall_stats(out.get("support_logits"), costs, topk=3)
            if support_stats is not None:
                s_top1, s_top3, s_seen, s_size = support_stats
                support_top1 += s_top1
                support_top3 += s_top3
                support_shortlist_size += s_size * costs.size(0)
                support_arm_seen = s_seen if support_arm_seen is None else (support_arm_seen | s_seen)

    costs_np = np.concatenate(all_costs, axis=0)
    method_mean = costs_np.mean(axis=0)
    sbs_idx = int(method_mean.argmin())
    sbs_cost = float(method_mean[sbs_idx])
    oracle_cost = float(costs_np.min(axis=1).mean())
    mean_cost = float(comp["final"]["cost_sum"] / max(1, n_total))
    pick_dist = (pick_count / pick_count.sum().clamp_min(1)).tolist()
    result = {
        "problem": problem,
        "top1": comp["final"]["top1"] / n_total,
        "top2": comp["final"]["top2"] / n_total,
        "top3": comp["final"]["top3"] / n_total,
        "mean_cost": mean_cost,
        "sbs_name": pool_names[sbs_idx],
        "sbs_cost": sbs_cost,
        "oracle_cost": oracle_cost,
        "vs_sbs_pct": (mean_cost - sbs_cost) / (abs(sbs_cost) + 1.0e-9) * 100,
        "vs_oracle_pct": (mean_cost - oracle_cost) / (abs(oracle_cost) + 1.0e-9) * 100,
        "arm_distribution": {pool_names[i]: float(pick_dist[i]) for i in range(len(pool_names))},
        "n": n_total,
    }
    if support_arm_seen is not None:
        result["support_top1_recall"] = support_top1 / max(1, n_total)
        result["support_top3_recall"] = support_top3 / max(1, n_total)
        result["support_avg_shortlist"] = support_shortlist_size / max(1, n_total)
        result["support_arm_coverage"] = float(support_arm_seen.float().mean().item())
    for name in sorted(k for k in comp if k != "final"):
        c = comp[name]
        mean_cost_name = float(c["cost_sum"] / max(1, n_total))
        result[f"top1_{name}"] = c["top1"] / n_total
        result[f"top2_{name}"] = c["top2"] / n_total
        result[f"top3_{name}"] = c["top3"] / n_total
        result[f"mean_cost_{name}"] = mean_cost_name
        result[f"vs_sbs_pct_{name}"] = (mean_cost_name - sbs_cost) / (abs(sbs_cost) + 1.0e-9) * 100
    return result


def macro_row(per_problem):
    vals = list(per_problem.values())
    row = {
        "problem": "ALL",
        "top1": float(np.mean([v["top1"] for v in vals])),
        "top2": float(np.mean([v["top2"] for v in vals])),
        "top3": float(np.mean([v["top3"] for v in vals])),
        "mean_cost": float(np.mean([v["mean_cost"] for v in vals])),
        "sbs_cost": float(np.mean([v["sbs_cost"] for v in vals])),
        "oracle_cost": float(np.mean([v["oracle_cost"] for v in vals])),
        "vs_sbs_pct": float(np.mean([v["vs_sbs_pct"] for v in vals])),
        "vs_oracle_pct": float(np.mean([v["vs_oracle_pct"] for v in vals])),
        "top1_pre": float(np.mean([v["top1_pre"] for v in vals])),
        "top1_gap": float(np.mean([v["top1_gap"] for v in vals])),
        "top1_utility": float(np.mean([v["top1_utility"] for v in vals])),
    }
    if all("support_top1_recall" in v for v in vals):
        row["support_top1_recall"] = float(np.mean([v["support_top1_recall"] for v in vals]))
        row["support_top3_recall"] = float(np.mean([v["support_top3_recall"] for v in vals]))
        row["support_avg_shortlist"] = float(np.mean([v["support_avg_shortlist"] for v in vals]))
        row["support_arm_coverage"] = float(np.mean([v["support_arm_coverage"] for v in vals]))
    for key in sorted({k for v in vals for k in v if k.startswith(("top1_", "top2_", "top3_", "mean_cost_", "vs_sbs_pct_"))}):
        row[key] = float(np.mean([v[key] for v in vals if key in v]))
    return row


def table_line(row, bold=False):
    name = row["problem"]
    top1 = f"{row['top1']:.4f}" if bold else f"{row['top1']:.3f}"
    top2 = f"{row['top2']:.4f}" if bold else f"{row['top2']:.3f}"
    top3 = f"{row['top3']:.4f}" if bold else f"{row['top3']:.3f}"
    mean_cost = f"{row['mean_cost']:.4f}"
    sbs_delta = row.get("vs_sbs_pct")
    if sbs_delta is None:
        sbs_delta = (row["mean_cost"] - row["sbs_cost"]) / (abs(row["sbs_cost"]) + 1.0e-9) * 100
    oracle_delta = row.get("vs_oracle_pct")
    if oracle_delta is None:
        oracle_delta = (row["mean_cost"] - row["oracle_cost"]) / (abs(row["oracle_cost"]) + 1.0e-9) * 100
    sbs = f"{row['sbs_cost']:.4f} ({sbs_delta:+.3f}%)"
    oracle = f"{row['oracle_cost']:.4f} ({oracle_delta:+.3f}%)"
    if bold:
        return f"| **{name}** | **{top1}** | **{top2}** | **{top3}** | **{mean_cost}** | **—** | **{sbs}** | **{oracle}** |"
    return f"| {name} | {top1} | {top2} | {top3} | {mean_cost} | — | {sbs} | {oracle} |"


def write_report(payload, out_dir):
    per_problem = payload["per_problem"]
    lines = [
        "# V4 R31c Test Evaluation",
        "",
        f"- Checkpoint: `{payload['ckpt']}`",
        f"- Split: `{payload['split']}`",
        f"- Checkpoint epoch: `{payload.get('epoch')}`",
        f"- Val macro stored in checkpoint: `{payload.get('val_macro')}`",
        "",
        "| Problem | top1 | top2 | top3 | mean_cost | Gap | SBS cost (vs_sbs) | Oracle (vs_Oracle) |",
        "| -------- | ---------: | ---------: | ---------: | ----------: | -----: | ---------------------: | ---------------------: |",
        table_line(payload["all"], bold=True),
    ]
    for problem in PROBLEMS:
        if problem in per_problem:
            lines.append(table_line(per_problem[problem]))
    lines += ["", "## Arm Distribution Snippet", ""]
    for problem in ["TSP", "CVRP", "ATSP", "VRPTW", "VRPLTW", "VRPBLTW"]:
        if problem not in per_problem:
            continue
        items = sorted(per_problem[problem]["arm_distribution"].items(), key=lambda kv: kv[1], reverse=True)[:5]
        text = " | ".join([f"{name}:{frac * 100:.1f}%" for name, frac in items if frac > 0])
        lines.append(f"- **{problem}**: {text}")
    if "support_top1_recall" in payload["all"]:
        lines += [
            "",
            "## Support Branch Diagnostics",
            "",
            "| Problem | support top1 recall | support top3 recall | arm coverage |",
            "| -------- | ---------: | ---------: | ---------: |",
            (
                f"| **ALL** | **{payload['all']['support_top1_recall']:.4f}** | "
                f"**{payload['all']['support_top3_recall']:.4f}** | "
                f"**{payload['all']['support_arm_coverage']:.4f}** |"
            ),
        ]
        for problem in PROBLEMS:
            if problem in per_problem and "support_top1_recall" in per_problem[problem]:
                r = per_problem[problem]
                lines.append(
                    f"| {problem} | {r['support_top1_recall']:.3f} | "
                    f"{r['support_top3_recall']:.3f} | {r['support_arm_coverage']:.3f} |"
                )
    sweep_names = [
        name
        for name in sorted({k[5:] for k in payload["all"] if k.startswith("top1_")})
        if "_su" in name
    ]
    if sweep_names:
        lines += [
            "",
            "## Support Hard-mask Sweep",
            "",
            "| Score | top1 | top2 | top3 | mean_cost | vs_sbs |",
            "| -------- | ---------: | ---------: | ---------: | ----------: | ---------: |",
        ]
        rows = []
        for name in sweep_names:
            rows.append(
                (
                    payload["all"][f"top1_{name}"],
                    f"| {name} | {payload['all'][f'top1_{name}']:.4f} | "
                    f"{payload['all'][f'top2_{name}']:.4f} | "
                    f"{payload['all'][f'top3_{name}']:.4f} | "
                    f"{payload['all'][f'mean_cost_{name}']:.4f} | "
                    f"{payload['all'][f'vs_sbs_pct_{name}']:+.3f}% |",
                )
            )
        for _, text in sorted(rows, reverse=True):
            lines.append(text)
    lines += [
        "",
        "## Component Diagnostics",
        "",
        "| Problem | final top1 | pre top1 | gap top1 | utility top1 |",
        "| -------- | ---------: | ---------: | ---------: | ---------: |",
        f"| **ALL** | **{payload['all']['top1']:.4f}** | **{payload['all']['top1_pre']:.4f}** | **{payload['all']['top1_gap']:.4f}** | **{payload['all']['top1_utility']:.4f}** |",
    ]
    for problem in PROBLEMS:
        if problem in per_problem:
            r = per_problem[problem]
            lines.append(
                f"| {problem} | {r['top1']:.3f} | {r['top1_pre']:.3f} | "
                f"{r['top1_gap']:.3f} | {r['top1_utility']:.3f} |"
            )
    lines.append("")
    (out_dir / "test_summary.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--support-mask-sweep", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    model, ckpt = load_model(args.ckpt, args.device)
    per_problem = {}
    for problem in PROBLEMS:
        r = evaluate_problem(
            model,
            problem,
            args.split,
            args.batch_size,
            args.num_workers,
            args.device,
            support_mask_sweep=args.support_mask_sweep,
        )
        per_problem[problem] = r
        print(
            f"{problem:>10}: top1={r['top1']:.3f} top2={r['top2']:.3f} top3={r['top3']:.3f} "
            f"mean_cost={r['mean_cost']:.4f} vs_sbs={r['vs_sbs_pct']:+.3f}% "
            f"pre={r['top1_pre']:.3f} gap={r['top1_gap']:.3f} util={r['top1_utility']:.3f}"
        )
    all_row = macro_row(per_problem)
    payload = {
        "ckpt": str(args.ckpt),
        "epoch": ckpt.get("epoch"),
        "val_macro": ckpt.get("macro"),
        "split": args.split,
        "per_problem": per_problem,
        "all": all_row,
    }
    (out_dir / "test_analysis.json").write_text(json.dumps(payload, indent=2))
    write_report(payload, out_dir)
    print("")
    print(table_line(all_row, bold=True))
    print(f"Saved {out_dir / 'test_summary.md'}")


if __name__ == "__main__":
    main()
