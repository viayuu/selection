import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from code.unified_selector.data import UnifiedProblemDataset, collate_single_problem
from code.unified_selector.registry import POOLS, PROBLEMS

from .V2Model import SolverConditionedSelector


def to_device(batch, device):
    return {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}


def make_loader(problem, split, batch_size):
    ds = UnifiedProblemDataset(problem, split, coord_augment=0)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=collate_single_problem)
    return ds, dl


def load_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    model_params = ckpt["args"]["model_params"]
    model = SolverConditionedSelector(**model_params).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, ckpt


def evaluate_problem(model, problem, split, batch_size, device):
    ds, dl = make_loader(problem, split, batch_size)
    pool_names = list(POOLS[problem])
    top1 = top2 = top3 = n_total = 0
    cost_sum = 0.0
    all_costs = []
    pick_count = None
    with torch.no_grad():
        for batch in dl:
            batch = to_device(batch, device)
            score, _ = model(batch)
            costs = batch["costs"]
            true_rank = torch.argsort(costs, dim=1)
            pred_rank = torch.argsort(-score, dim=1)
            best = true_rank[:, 0]
            pred = pred_rank[:, 0]
            top1 += (pred == best).sum().item()
            top2 += (pred_rank[:, : min(2, score.size(1))] == best[:, None]).any(dim=1).sum().item()
            top3 += (pred_rank[:, : min(3, score.size(1))] == best[:, None]).any(dim=1).sum().item()
            cost_sum += costs.gather(1, pred[:, None]).sum().item()
            n_total += costs.size(0)
            all_costs.append(costs.detach().cpu().numpy())
            cur_count = torch.bincount(pred.detach().cpu(), minlength=score.size(1)).float()
            pick_count = cur_count if pick_count is None else pick_count + cur_count

    costs_np = np.concatenate(all_costs, axis=0)
    method_mean = costs_np.mean(axis=0)
    sbs_idx = int(method_mean.argmin())
    sbs_cost = float(method_mean[sbs_idx])
    oracle_cost = float(costs_np.min(axis=1).mean())
    mean_cost = float(cost_sum / max(1, n_total))
    pick_dist = (pick_count / pick_count.sum().clamp_min(1)).tolist()
    arm_distribution = {pool_names[i]: float(pick_dist[i]) for i in range(len(pool_names))}
    return {
        "problem": problem,
        "top1": top1 / n_total,
        "top2": top2 / n_total,
        "top3": top3 / n_total,
        "mean_cost": mean_cost,
        "sbs_name": pool_names[sbs_idx],
        "sbs_cost": sbs_cost,
        "oracle_cost": oracle_cost,
        "vs_sbs_pct": (mean_cost - sbs_cost) / (abs(sbs_cost) + 1.0e-9) * 100,
        "vs_oracle_pct": (mean_cost - oracle_cost) / (abs(oracle_cost) + 1.0e-9) * 100,
        "arm_distribution": arm_distribution,
        "n": n_total,
    }


def macro_row(per_problem):
    vals = list(per_problem.values())
    return {
        "problem": "ALL",
        "top1": float(np.mean([v["top1"] for v in vals])),
        "top2": float(np.mean([v["top2"] for v in vals])),
        "top3": float(np.mean([v["top3"] for v in vals])),
        "mean_cost": float(np.mean([v["mean_cost"] for v in vals])),
        "sbs_cost": float(np.mean([v["sbs_cost"] for v in vals])),
        "oracle_cost": float(np.mean([v["oracle_cost"] for v in vals])),
    }


def fmt_pct(x):
    return f"{x:+.3f}%"


def table_line(row, bold=False):
    name = row["problem"]
    top1 = f"{row['top1']:.4f}" if bold else f"{row['top1']:.3f}"
    top2 = f"{row['top2']:.4f}" if bold else f"{row['top2']:.3f}"
    top3 = f"{row['top3']:.4f}" if bold else f"{row['top3']:.3f}"
    mean_cost = f"{row['mean_cost']:.4f}"
    gap = "—"
    sbs_delta = (row["mean_cost"] - row["sbs_cost"]) / (abs(row["sbs_cost"]) + 1.0e-9) * 100
    oracle_delta = (row["mean_cost"] - row["oracle_cost"]) / (abs(row["oracle_cost"]) + 1.0e-9) * 100
    sbs = f"{row['sbs_cost']:.4f} ({fmt_pct(sbs_delta)})"
    oracle = f"{row['oracle_cost']:.4f} ({fmt_pct(oracle_delta)})"
    if bold:
        return f"| **{name}** | **{top1}** | **{top2}** | **{top3}** | **{mean_cost}** | **{gap}** | **{sbs}** | **{oracle}** |"
    return f"| {name} | {top1} | {top2} | {top3} | {mean_cost} | {gap} | {sbs} | {oracle} |"


def arm_snippets(per_problem, topk=5):
    lines = []
    for problem in ["TSP", "CVRP", "ATSP", "VRPTW", "VRPLTW", "VRPBLTW"]:
        if problem not in per_problem:
            continue
        dist = per_problem[problem]["arm_distribution"]
        items = sorted(dist.items(), key=lambda kv: kv[1], reverse=True)[:topk]
        txt = " | ".join([f"{name}:{frac * 100:.1f}%" for name, frac in items if frac > 0])
        lines.append(f"- **{problem}**: {txt}")
    return lines


def write_report(payload, out_dir):
    per_problem = payload["per_problem"]
    all_row = payload["all"]
    lines = [
        "# V2 Test Evaluation",
        "",
        f"- Checkpoint: `{payload['ckpt']}`",
        f"- Checkpoint epoch: `{payload.get('epoch')}`",
        f"- Val macro stored in checkpoint: `{payload.get('val_macro')}`",
        "",
        "| Problem | top1 | top2 | top3 | mean_cost | Gap | SBS cost (vs_sbs) | Oracle (vs_Oracle) |",
        "| -------- | ---------: | ---------: | ---------: | ----------: | -----: | ---------------------: | ---------------------: |",
        table_line(all_row, bold=True),
    ]
    for p in PROBLEMS:
        lines.append(table_line(per_problem[p]))
    lines += ["", "## Arm Distribution Snippet", ""]
    lines += arm_snippets(per_problem)
    lines.append("")
    (out_dir / "test_summary.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    model, ckpt = load_model(args.ckpt, args.device)
    per_problem = {}
    for problem in PROBLEMS:
        r = evaluate_problem(model, problem, args.split, args.batch_size, args.device)
        per_problem[problem] = r
        print(f"{problem:>10}: top1={r['top1']:.3f} top2={r['top2']:.3f} top3={r['top3']:.3f} mean_cost={r['mean_cost']:.4f} vs_sbs={r['vs_sbs_pct']:+.3f}%")
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
