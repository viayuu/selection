from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import torch
import yaml
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataset import SelectionDataset, collate_fn
from model import Selection_model
from utils import prepare_dataset


RUNS = {
    "TSP": {
        "problem_type": "TSP",
        "split_name": "TSPtest",
        "config": "config_TSP_ours.yml",
        "checkpoint": "train_logs/config_TSP_ours_rank_2024/checkpoint_epoch_best.pt",
    },
    "TSPLIB": {
        "problem_type": "TSP",
        "split_name": "TSPLIB",
        "config": "config_TSP_ours.yml",
        "checkpoint": "train_logs/config_TSP_ours_rank_2024/checkpoint_epoch_best.pt",
    },
    "CVRP": {
        "problem_type": "CVRP",
        "split_name": "CVRPtest",
        "config": "config_CVRP_ours.yml",
        "checkpoint": "train_logs/config_CVRP_ours_rank_2024/checkpoint_epoch_best.pt",
    },
    "CVRPLIB": {
        "problem_type": "CVRP",
        "split_name": "CVRPLIB",
        "config": "config_CVRP_ours.yml",
        "checkpoint": "train_logs/config_CVRP_ours_rank_2024/checkpoint_epoch_best.pt",
    },
}


def load_solver_order(path: Path) -> dict[str, list[str]]:
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def build_model(config_path: Path, checkpoint_path: Path, device: torch.device) -> Selection_model:
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.load(f.read(), Loader=yaml.FullLoader)
    config["model_params"]["problem_type"] = config["problem_type"]
    config["model_params"]["output_dim"] = config["train_params"]["num_classes"]
    config["model_params"]["ns_feature"] = config["train_params"]["ns_feature"]
    model = Selection_model(**config["model_params"]).to(device)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def set_default_tensor_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.set_default_tensor_type("torch.cuda.FloatTensor")
    else:
        torch.set_default_tensor_type("torch.FloatTensor")


def evaluate_one(name: str, spec: dict[str, str], device: torch.device) -> dict[str, Any]:
    # CVRP preprocessing in the original utils expects CPU tensors from pickle.
    torch.set_default_tensor_type("torch.FloatTensor")
    _, _, test_set, test_label = prepare_dataset(spec["problem_type"], name=spec["split_name"])
    set_default_tensor_device(device)
    model = build_model(Path(spec["config"]), Path(spec["checkpoint"]), device)
    dataset = SelectionDataset(test_set, test_label, manual_feature=False)
    # Keep the original NSS test batch size for a clean comparison.
    loader = DataLoader(dataset, batch_size=8, collate_fn=collate_fn, shuffle=False)

    all_scores = []
    all_costs = []
    all_gaps = []
    all_y = []
    with torch.no_grad():
        for batch in loader:
            x = batch[0].to(device)
            y = batch[1].to(device)
            cost = batch[2].to(device)
            scales = batch[3].to(device)
            mask = batch[4].to(device)
            gap = batch[5].to(device)
            logits = model(x, scales, None, mask)
            all_scores.append(logits.detach().cpu())
            all_costs.append(cost.detach().cpu())
            all_gaps.append(gap.detach().cpu())
            all_y.append(y.detach().cpu())

    scores = torch.cat(all_scores, dim=0)
    costs = torch.cat(all_costs, dim=0)
    gaps = torch.cat(all_gaps, dim=0)
    y = torch.cat(all_y, dim=0)
    n = int(costs.shape[0])
    probs = torch.softmax(scores, dim=1)
    k3 = min(3, scores.shape[1])
    top = scores.topk(k3, dim=1).indices
    pred = top[:, 0]
    pred_cost = costs.gather(1, pred[:, None]).squeeze(1)
    oracle_cost, oracle_idx = costs.min(dim=1)
    sbs_cost, sbs_idx = costs.mean(dim=0).min(dim=0)

    top1 = (pred == oracle_idx).float().mean().item()
    top2 = (top[:, : min(2, k3)] == oracle_idx[:, None]).any(dim=1).float().mean().item()
    top3 = (top[:, :k3] == oracle_idx[:, None]).any(dim=1).float().mean().item()
    mean_cost = pred_cost.mean().item()
    sbs = sbs_cost.item()
    oracle = oracle_cost.mean().item()
    vs_sbs = (mean_cost - sbs) / (abs(sbs) + 1e-12) * 100.0
    vs_oracle = (mean_cost - oracle) / (abs(oracle) + 1e-12) * 100.0

    selected_gap = gaps.gather(1, pred[:, None]).squeeze(1)
    gap_value = None
    if gaps.abs().sum().item() > 0:
        gap_value = selected_gap.mean().item()

    strategies = strategy_metrics(probs, costs, gaps, p=0.5)

    return {
        "problem": name,
        "n": n,
        "top1": top1,
        "top2": top2,
        "top3": top3,
        "mean_cost": mean_cost,
        "gap": gap_value,
        "sbs_cost": sbs,
        "sbs_idx": int(sbs_idx.item()),
        "oracle_cost": oracle,
        "vs_sbs_pct": vs_sbs,
        "vs_oracle_pct": vs_oracle,
        "pick_counts": dict(Counter(pred.tolist())),
        "oracle_counts": dict(Counter(oracle_idx.tolist())),
        "strategies": strategies,
    }


def summarize_selection(
    selected: list[list[int]],
    costs: torch.Tensor,
    gaps: torch.Tensor,
    oracle_idx: torch.Tensor,
    sbs_cost: float,
    oracle_cost_mean: float,
) -> dict[str, Any]:
    selected_cost = []
    selected_gap = []
    hit_oracle = []
    selected_size = []
    picks = []
    for i, inds in enumerate(selected):
        idx = torch.tensor(inds, dtype=torch.long, device=costs.device)
        row_cost = costs[i, idx]
        best_pos = int(row_cost.argmin().item())
        selected_cost.append(row_cost[best_pos].item())
        if gaps.abs().sum().item() > 0:
            selected_gap.append(gaps[i, idx][best_pos].item())
        hit_oracle.append(int(oracle_idx[i].item() in inds))
        selected_size.append(len(inds))
        picks.append(int(inds[best_pos]))
    mean_cost = sum(selected_cost) / len(selected_cost)
    return {
        "coverage": sum(hit_oracle) / len(hit_oracle),
        "mean_cost": mean_cost,
        "gap": None if not selected_gap else sum(selected_gap) / len(selected_gap),
        "avg_solvers": sum(selected_size) / len(selected_size),
        "vs_sbs_pct": (mean_cost - sbs_cost) / (abs(sbs_cost) + 1e-12) * 100.0,
        "vs_oracle_pct": (mean_cost - oracle_cost_mean) / (abs(oracle_cost_mean) + 1e-12) * 100.0,
        "pick_counts": dict(Counter(picks)),
    }


def strategy_metrics(probs: torch.Tensor, costs: torch.Tensor, gaps: torch.Tensor, p: float) -> dict[str, dict[str, Any]]:
    n = costs.shape[0]
    oracle_cost, oracle_idx = costs.min(dim=1)
    sbs_cost = costs.mean(dim=0).min().item()
    oracle_cost_mean = oracle_cost.mean().item()

    selected: dict[str, list[list[int]]] = {}
    selected["Greedy"] = [[int(probs[i].argmax().item())] for i in range(n)]
    top2 = probs.topk(min(2, probs.shape[1]), dim=1).indices
    selected["Top-k (k = 2)"] = [[int(v) for v in top2[i].tolist()] for i in range(n)]

    # Original NSS rejection: accept highest-confidence 80% with top-1, reject lowest 20% with top-2.
    confidence = probs.max(dim=1).values
    sort_ind = confidence.sort(descending=True).indices
    threshold = int(n * 0.8)
    accepted = set(int(v) for v in sort_ind[:threshold].tolist())
    rejected_top2 = selected["Top-k (k = 2)"]
    selected["Rejection (20%)"] = [
        selected["Greedy"][i] if i in accepted else rejected_top2[i]
        for i in range(n)
    ]

    top_p_selected = []
    for i in range(n):
        for j in range(1, probs.shape[1] + 1):
            top_j = probs[i].topk(j, largest=True)
            if top_j.values.sum().item() >= p or j == probs.shape[1]:
                top_p_selected.append([int(v) for v in top_j.indices.tolist()])
                break
    selected[f"Top-p (p = {p:.1f})"] = top_p_selected

    return {
        name: summarize_selection(inds, costs, gaps, oracle_idx, sbs_cost, oracle_cost_mean)
        for name, inds in selected.items()
    }


def pct(value: float) -> str:
    return f"{value:+.3f}%"


def row_md(row: dict[str, Any], bold: bool = False) -> str:
    problem = f"**{row['problem']}**" if bold else row["problem"]
    top1 = f"{row['top1']:.4f}"
    top2 = f"{row['top2']:.4f}"
    top3 = f"{row['top3']:.4f}"
    mean_cost = f"{row['mean_cost']:.4f}"
    gap = "—" if row["gap"] is None else f"{row['gap']:.3f}%"
    sbs = f"{row['sbs_cost']:.4f} ({pct(row['vs_sbs_pct'])})"
    oracle = f"{row['oracle_cost']:.4f} ({pct(row['vs_oracle_pct'])})"
    if bold:
        top1 = f"**{top1}**"
        top2 = f"**{top2}**"
        top3 = f"**{top3}**"
        mean_cost = f"**{mean_cost}**"
        gap = f"**{gap}**"
        sbs = f"**{sbs}**"
        oracle = f"**{oracle}**"
    return f"| {problem} | {top1} | {top2} | {top3} | {mean_cost} | {gap} | {sbs} | {oracle} |"


def macro_all(rows: list[dict[str, Any]]) -> dict[str, Any]:
    keys = ["top1", "top2", "top3", "mean_cost", "sbs_cost", "oracle_cost"]
    total_n = sum(row["n"] for row in rows)
    out: dict[str, Any] = {"problem": "ALL", "gap": None, "n": total_n}
    for key in keys:
        out[key] = sum(row[key] * row["n"] for row in rows) / total_n
    out["vs_sbs_pct"] = (out["mean_cost"] - out["sbs_cost"]) / (abs(out["sbs_cost"]) + 1e-12) * 100.0
    out["vs_oracle_pct"] = (out["mean_cost"] - out["oracle_cost"]) / (abs(out["oracle_cost"]) + 1e-12) * 100.0
    return out


def arm_distribution(row: dict[str, Any], order: list[str], topn: int = 5) -> str:
    counts = Counter({int(k): int(v) for k, v in row["pick_counts"].items()})
    total = sum(counts.values()) or 1
    parts = []
    for idx, count in counts.most_common(topn):
        name = order[idx] if idx < len(order) else f"solver{idx}"
        parts.append(f"{name}:{count / total * 100:.1f}%")
    return " | ".join(parts)


def weighted_strategy_all(rows: list[dict[str, Any]], strategy: str) -> dict[str, Any]:
    total_n = sum(row["n"] for row in rows)
    out: dict[str, Any] = {"problem": "ALL", "strategy": strategy, "gap": None, "n": total_n}
    for key in ["coverage", "mean_cost", "avg_solvers"]:
        out[key] = sum(row["strategies"][strategy][key] * row["n"] for row in rows) / total_n
    sbs_cost = sum(row["sbs_cost"] * row["n"] for row in rows) / total_n
    oracle_cost = sum(row["oracle_cost"] * row["n"] for row in rows) / total_n
    out["vs_sbs_pct"] = (out["mean_cost"] - sbs_cost) / (abs(sbs_cost) + 1e-12) * 100.0
    out["vs_oracle_pct"] = (out["mean_cost"] - oracle_cost) / (abs(oracle_cost) + 1e-12) * 100.0
    return out


def strategy_row_md(problem: str, strategy: str, values: dict[str, Any], bold: bool = False) -> str:
    p = f"**{problem}**" if bold else problem
    s = f"**{strategy}**" if bold else strategy
    cov = f"{values['coverage']:.4f}"
    mean_cost = f"{values['mean_cost']:.4f}"
    gap = "—" if values["gap"] is None else f"{values['gap']:.3f}%"
    avg_solvers = f"{values['avg_solvers']:.2f}"
    vs_sbs = pct(values["vs_sbs_pct"])
    vs_oracle = pct(values["vs_oracle_pct"])
    if bold:
        cov = f"**{cov}**"
        mean_cost = f"**{mean_cost}**"
        gap = f"**{gap}**"
        avg_solvers = f"**{avg_solvers}**"
        vs_sbs = f"**{vs_sbs}**"
        vs_oracle = f"**{vs_oracle}**"
    return f"| {p} | {s} | {cov} | {mean_cost} | {gap} | {vs_sbs} | {vs_oracle} | {avg_solvers} |"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", default="results/nss_retrain_rank_test_summary.md")
    parser.add_argument("--json-output", default="results/nss_retrain_rank_test_summary.json")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    order = load_solver_order(Path("solver_order_ours.json"))
    rows = [evaluate_one(name, spec, device) for name, spec in RUNS.items()]
    all_row = macro_all(rows)

    lines = [
        "# NSS Retrain Rank Test Summary",
        "",
        "说明：`ALL` 是下方四个数据集按实例数加权的平均；当前标签中的 `gap` 全为 0，因此 `Gap` 列记为 `—`。",
        "",
        "| Problem | top1 | top2 | top3 | mean_cost | Gap | SBS cost (vs_sbs) | Oracle (vs_Oracle) |",
        "| -------- | ---------: | ---------: | ---------: | ----------: | -----: | ---------------------: | ---------------------: |",
        row_md(all_row, bold=True),
    ]
    for row in rows:
        lines.append(row_md(row))
    lines.extend(["", "## Arm Distribution", ""])
    for row in rows:
        problem_type = RUNS[row["problem"]]["problem_type"]
        lines.append(f"- **{row['problem']}**: {arm_distribution(row, order.get(problem_type, []))}")

    strategy_names = ["Greedy", "Top-k (k = 2)", "Rejection (20%)", "Top-p (p = 0.5)"]
    lines.extend([
        "",
        "## Selection Strategies",
        "",
        "说明：`coverage` 表示该策略选中的 solver 集合是否包含 oracle best solver；Greedy 的 coverage 等价于 exact top1。",
        "",
        "| Problem | Strategy | coverage | mean_cost | Gap | vs_sbs | vs_Oracle | avg_solvers |",
        "| -------- | -------- | ---------: | ----------: | -----: | -----: | -----: | ----------: |",
    ])
    for strategy in strategy_names:
        lines.append(strategy_row_md("ALL", strategy, weighted_strategy_all(rows, strategy), bold=True))
    for row in rows:
        for strategy in strategy_names:
            lines.append(strategy_row_md(row["problem"], strategy, row["strategies"][strategy]))

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    json_output = Path(args.json_output)
    json_output.parent.mkdir(parents=True, exist_ok=True)
    json_output.write_text(json.dumps({"all": all_row, "rows": rows}, indent=2), encoding="utf-8")
    print("\n".join(lines))
    print(f"\n[wrote] {output}")
    print(f"[wrote] {json_output}")


if __name__ == "__main__":
    main()
