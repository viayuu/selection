import argparse
import csv
import json
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import UnifiedSelectionDataset, build_dataset_specs_from_export_root, collate_fn
from model import UnifiedSelectionModel


def load_run_bundle(run_dir):
    run_dir = os.path.abspath(run_dir)
    project_root = os.path.dirname(os.path.dirname(run_dir))

    with open(os.path.join(run_dir, "config.json"), "r", encoding="utf-8") as file_obj:
        config = json.load(file_obj)
    with open(os.path.join(run_dir, "metadata.json"), "r", encoding="utf-8") as file_obj:
        metadata = json.load(file_obj)

    export_root = os.path.abspath(os.path.join(project_root, config["data_params"]["export_root"]))
    train_specs, val_specs, test_specs, benchmark_specs = build_dataset_specs_from_export_root(
        export_root, config["data_params"]
    )
    split_specs = {
        "val": val_specs,
        "test": test_specs,
        "benchmark": benchmark_specs,
    }

    solver_pool = metadata["solver_pool"]
    problem_pool = metadata["problem_pool"]
    solver_features = torch.tensor(metadata["solver_features"], dtype=torch.float32)

    model = UnifiedSelectionModel(solver_features=solver_features, **config["model_params"])
    checkpoint = torch.load(os.path.join(run_dir, "checkpoint_epoch_best.pt"), map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])

    return {
        "run_dir": run_dir,
        "project_root": project_root,
        "config": config,
        "solver_pool": solver_pool,
        "problem_pool": problem_pool,
        "split_specs": split_specs,
        "model": model,
    }


def build_dataset(run_bundle, split_name):
    return UnifiedSelectionDataset(
        run_bundle["split_specs"][split_name],
        solver_pool=run_bundle["solver_pool"],
        problem_pool=run_bundle["problem_pool"],
        node_feature_dim=run_bundle["config"]["data_params"].get("node_feature_dim", 12),
        data_aug=False,
    )


def get_device(gpu_id):
    if gpu_id is None or gpu_id < 0 or not torch.cuda.is_available():
        return torch.device("cpu")
    torch.cuda.set_device(gpu_id)
    return torch.device("cuda", gpu_id)


def evaluate_run_on_split(run_bundle, split_name, device):
    dataset = build_dataset(run_bundle, split_name)
    loader = DataLoader(
        dataset,
        batch_size=run_bundle["config"]["train_params"]["test_batch_size"],
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=False,
    )

    model = run_bundle["model"].to(device)
    model.eval()

    rows = []
    costs_all = []
    feasible_all = []
    problem_ids_all = []
    preds_all = []
    labels_all = []
    solver_pool = run_bundle["solver_pool"]

    with torch.no_grad():
        offset = 0
        for batch in loader:
            tensor_batch = {}
            for key, value in batch.items():
                tensor_batch[key] = value.to(device) if torch.is_tensor(value) else value

            logits = model(
                tensor_batch["nodes"],
                tensor_batch["scales"],
                tensor_batch["node_mask"],
                tensor_batch["problem_ids"],
                tensor_batch["problem_features"],
                tensor_batch["feasible_mask"],
                tensor_batch["instance_stats"],
            )
            masked_logits = logits.masked_fill(~tensor_batch["feasible_mask"].bool(), -1e9)
            preds = torch.argmax(masked_logits, dim=1)
            costs = tensor_batch["costs"]
            feasible = tensor_batch["feasible_mask"]
            labels = tensor_batch["labels"]
            lengths = batch["lengths"]

            costs_all.append(costs.detach().cpu())
            feasible_all.append(feasible.detach().cpu())
            problem_ids_all.append(tensor_batch["problem_ids"].detach().cpu())
            preds_all.append(preds.detach().cpu())
            labels_all.append(labels.detach().cpu())

            for idx in range(costs.size(0)):
                row = {
                    "problem_name": batch["problem_names"][idx].lower(),
                    "dataset_name": batch["dataset_names"][idx],
                    "instance_id": int(batch["instance_ids"][idx].item()),
                    "node_count": int(lengths[idx].item()),
                    "selector_solver": solver_pool[int(preds[idx].item())],
                    "oracle_solver": solver_pool[int(labels[idx].item())],
                    "selector_cost": float(costs[idx, preds[idx]].item()),
                    "oracle_cost": float(costs[idx, labels[idx]].item()),
                }
                rows.append(row)
                offset += 1

    costs = torch.cat(costs_all, dim=0)
    feasible = torch.cat(feasible_all, dim=0)
    problem_ids = torch.cat(problem_ids_all, dim=0)
    preds = torch.cat(preds_all, dim=0)

    # Best single solver per problem, using the same split.
    best_single_solver_idx = {}
    best_single_cost = torch.zeros(costs.size(0), dtype=torch.float32)
    for problem_id in problem_ids.unique(sorted=True):
        mask = problem_ids == problem_id
        problem_costs = costs[mask]
        problem_feasible = feasible[mask]
        mean_costs = problem_costs.masked_fill(~problem_feasible, float("inf")).mean(dim=0)
        solver_idx = int(torch.argmin(mean_costs).item())
        best_single_solver_idx[int(problem_id.item())] = solver_idx
        best_single_cost[mask] = problem_costs[:, solver_idx]

    for idx, row in enumerate(rows):
        pid = int(problem_ids[idx].item())
        solver_idx = best_single_solver_idx[pid]
        row["best_single_solver"] = solver_pool[solver_idx]
        row["best_single_cost"] = float(best_single_cost[idx].item())
        row["gap_vs_best_single"] = row["selector_cost"] - row["best_single_cost"]
        row["gap_vs_oracle"] = row["selector_cost"] - row["oracle_cost"]

    return {
        "rows": rows,
        "costs": costs,
        "feasible": feasible,
        "problem_ids": problem_ids,
        "preds": preds,
        "best_single_cost": best_single_cost,
        "best_single_solver_idx": best_single_solver_idx,
    }


def bootstrap_mean_ci(diff_array, num_bootstrap=5000, seed=2024):
    rng = np.random.default_rng(seed)
    diff_array = np.asarray(diff_array, dtype=np.float64)
    n = diff_array.shape[0]
    samples = rng.integers(0, n, size=(num_bootstrap, n))
    means = diff_array[samples].mean(axis=1)
    lower, upper = np.percentile(means, [2.5, 97.5])
    p_non_positive = float(np.mean(means <= 0.0))
    p_non_negative = float(np.mean(means >= 0.0))
    return {
        "mean_diff": float(diff_array.mean()),
        "ci_low": float(lower),
        "ci_high": float(upper),
        "p_non_positive": p_non_positive,
        "p_non_negative": p_non_negative,
        "win_rate_negative": float(np.mean(diff_array < 0.0)),
        "tie_rate": float(np.mean(diff_array == 0.0)),
    }


def align_rows(rows_a, rows_b):
    keyed_a = {
        (row["problem_name"], row["dataset_name"], row["instance_id"]): row
        for row in rows_a
    }
    keyed_b = {
        (row["problem_name"], row["dataset_name"], row["instance_id"]): row
        for row in rows_b
    }
    keys = sorted(set(keyed_a.keys()) & set(keyed_b.keys()))
    return keys, keyed_a, keyed_b


def write_csv(path, rows):
    if not rows:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def summarize_shift(rows, selector_name):
    cvrp_rows = [row for row in rows if row["problem_name"] == "cvrp"]
    by_size = {}
    by_solver = {}
    for row in cvrp_rows:
        size_key = row["node_count"]
        by_size.setdefault(size_key, []).append(row["gap_vs_best_single"])
        solver_key = row["selector_solver"]
        by_solver.setdefault(solver_key, []).append(row["gap_vs_best_single"])

    size_rows = [
        {
            "node_count": size_key,
            "count": len(values),
            "mean_gap_vs_best_single": float(np.mean(values)),
            "median_gap_vs_best_single": float(np.median(values)),
            "max_gap_vs_best_single": float(np.max(values)),
        }
        for size_key, values in sorted(by_size.items())
    ]
    solver_rows = [
        {
            "selector_solver": solver_key,
            "count": len(values),
            "mean_gap_vs_best_single": float(np.mean(values)),
            "median_gap_vs_best_single": float(np.median(values)),
            "max_gap_vs_best_single": float(np.max(values)),
        }
        for solver_key, values in sorted(by_solver.items())
    ]
    worst_rows = sorted(cvrp_rows, key=lambda row: row["gap_vs_best_single"], reverse=True)[:25]

    return {
        "selector_name": selector_name,
        "cvrp_count": len(cvrp_rows),
        "mean_gap_vs_best_single": float(np.mean([row["gap_vs_best_single"] for row in cvrp_rows])),
        "size_rows": size_rows,
        "solver_rows": solver_rows,
        "worst_rows": worst_rows,
    }


def main():
    parser = argparse.ArgumentParser(description="Bootstrap/significance and CVRP-LIB shift analysis")
    parser.add_argument("--run-a", required=True, help="Main run directory")
    parser.add_argument("--run-b", required=True, help="Comparison run directory")
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--num-bootstrap", type=int, default=5000)
    args = parser.parse_args()

    device = get_device(args.gpu_id)
    run_a = load_run_bundle(args.run_a)
    run_b = load_run_bundle(args.run_b)

    output_dir = os.path.join(run_a["project_root"], "train_logs", "analysis_followup")
    os.makedirs(output_dir, exist_ok=True)

    summary_rows = []
    shift_manifest = {}

    for split_name in ["test", "benchmark"]:
        result_a = evaluate_run_on_split(run_a, split_name, device)
        result_b = evaluate_run_on_split(run_b, split_name, device)

        keys, keyed_a, keyed_b = align_rows(result_a["rows"], result_b["rows"])
        aligned_rows = []
        diff_a_vs_b = []
        diff_a_vs_best_single = []
        diff_b_vs_best_single = []
        for key in keys:
            row_a = keyed_a[key]
            row_b = keyed_b[key]
            aligned_rows.append(
                {
                    "problem_name": row_a["problem_name"],
                    "dataset_name": row_a["dataset_name"],
                    "instance_id": row_a["instance_id"],
                    "node_count": row_a["node_count"],
                    "run_a_selector_solver": row_a["selector_solver"],
                    "run_b_selector_solver": row_b["selector_solver"],
                    "best_single_solver": row_a["best_single_solver"],
                    "run_a_selector_cost": row_a["selector_cost"],
                    "run_b_selector_cost": row_b["selector_cost"],
                    "best_single_cost": row_a["best_single_cost"],
                    "oracle_cost": row_a["oracle_cost"],
                    "run_a_gap_vs_best_single": row_a["gap_vs_best_single"],
                    "run_b_gap_vs_best_single": row_b["gap_vs_best_single"],
                    "run_a_minus_run_b": row_a["selector_cost"] - row_b["selector_cost"],
                }
            )
            diff_a_vs_b.append(row_a["selector_cost"] - row_b["selector_cost"])
            diff_a_vs_best_single.append(row_a["selector_cost"] - row_a["best_single_cost"])
            diff_b_vs_best_single.append(row_b["selector_cost"] - row_b["best_single_cost"])

        write_csv(os.path.join(output_dir, f"{split_name}_paired_rows.csv"), aligned_rows)

        summary_rows.append(
            {
                "split": split_name,
                "comparison": "run_a_vs_run_b",
                **bootstrap_mean_ci(diff_a_vs_b, num_bootstrap=args.num_bootstrap, seed=2024),
            }
        )
        summary_rows.append(
            {
                "split": split_name,
                "comparison": "run_a_vs_best_single",
                **bootstrap_mean_ci(diff_a_vs_best_single, num_bootstrap=args.num_bootstrap, seed=2025),
            }
        )
        summary_rows.append(
            {
                "split": split_name,
                "comparison": "run_b_vs_best_single",
                **bootstrap_mean_ci(diff_b_vs_best_single, num_bootstrap=args.num_bootstrap, seed=2026),
            }
        )

        if split_name == "benchmark":
            shift_a = summarize_shift(result_a["rows"], os.path.basename(run_a["run_dir"]))
            shift_b = summarize_shift(result_b["rows"], os.path.basename(run_b["run_dir"]))
            shift_manifest["run_a"] = {
                "summary": {
                    "selector_name": shift_a["selector_name"],
                    "cvrp_count": shift_a["cvrp_count"],
                    "mean_gap_vs_best_single": shift_a["mean_gap_vs_best_single"],
                }
            }
            shift_manifest["run_b"] = {
                "summary": {
                    "selector_name": shift_b["selector_name"],
                    "cvrp_count": shift_b["cvrp_count"],
                    "mean_gap_vs_best_single": shift_b["mean_gap_vs_best_single"],
                }
            }
            write_csv(os.path.join(output_dir, "benchmark_cvrp_shift_by_size_run_a.csv"), shift_a["size_rows"])
            write_csv(os.path.join(output_dir, "benchmark_cvrp_shift_by_solver_run_a.csv"), shift_a["solver_rows"])
            write_csv(os.path.join(output_dir, "benchmark_cvrp_shift_worst_run_a.csv"), shift_a["worst_rows"])
            write_csv(os.path.join(output_dir, "benchmark_cvrp_shift_by_size_run_b.csv"), shift_b["size_rows"])
            write_csv(os.path.join(output_dir, "benchmark_cvrp_shift_by_solver_run_b.csv"), shift_b["solver_rows"])
            write_csv(os.path.join(output_dir, "benchmark_cvrp_shift_worst_run_b.csv"), shift_b["worst_rows"])

    write_csv(os.path.join(output_dir, "bootstrap_summary.csv"), summary_rows)
    with open(os.path.join(output_dir, "shift_manifest.json"), "w", encoding="utf-8") as file_obj:
        json.dump(shift_manifest, file_obj, indent=2)

    print("Output directory:", output_dir)
    for row in summary_rows:
        print(row)


if __name__ == "__main__":
    main()
