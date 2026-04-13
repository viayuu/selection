import argparse
import json
import os
import pickle
from collections import defaultdict
from typing import Any, Dict, List

import torch


DATASET_SPECS = {
    "TSPtrain": {"problem": "tsp", "distribution": "gaussian"},
    "TSPval": {"problem": "tsp", "distribution": "gaussian"},
    "TSPtest": {"problem": "tsp", "distribution": "gaussian"},
    "TSPLIB": {"problem": "tsp", "distribution": "lib"},
    "CVRPtrain": {"problem": "cvrp", "distribution": "gaussian"},
    "CVRPval": {"problem": "cvrp", "distribution": "gaussian"},
    "CVRPtest": {"problem": "cvrp", "distribution": "gaussian"},
    "CVRPLIB": {"problem": "cvrp", "distribution": "lib"},
}

FIXED_SCALE_GRAPH_DIR = "graph_fixed_by_scale"


def _dataset_file_stem(dataset_name: str) -> str:
    if dataset_name == "TSPLIB":
        return "tsp_lib"
    if dataset_name == "CVRPLIB":
        return "cvrp_lib"
    return dataset_name.lower()


def _load_pickle(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


def _to_list_tensor(data: Any) -> List:
    return torch.as_tensor(data).detach().cpu().tolist()


def _normalize_tsp_item(item: Any) -> torch.Tensor:
    tensor = torch.as_tensor(item, dtype=torch.float32)
    if tensor.ndim == 3 and tensor.size(0) == 1:
        tensor = tensor.squeeze(0)
    if tensor.ndim != 2 or tensor.size(-1) != 2:
        raise ValueError(f"Unexpected TSP item shape: {tuple(tensor.shape)}")
    return tensor


def _normalize_cvrp_item(item: Dict[str, Any]):
    loc = torch.as_tensor(item["loc"], dtype=torch.float32)
    demand = torch.as_tensor(item["demand"], dtype=torch.float32)
    depot = torch.as_tensor(item["depot"], dtype=torch.float32)
    if loc.ndim == 3 and loc.size(0) == 1:
        loc = loc.squeeze(0)
    if demand.ndim == 2 and demand.size(0) == 1:
        demand = demand.squeeze(0)
    if depot.ndim == 3 and depot.size(0) == 1:
        depot = depot.squeeze(0)
    if depot.ndim == 2 and depot.size(0) == 1:
        depot = depot.squeeze(0)
    if loc.ndim != 2 or loc.size(-1) != 2:
        raise ValueError(f"Unexpected CVRP loc shape: {tuple(loc.shape)}")
    if demand.ndim != 1 or demand.size(0) != loc.size(0):
        raise ValueError(
            f"Unexpected CVRP demand shape: {tuple(demand.shape)} for loc shape {tuple(loc.shape)}"
        )
    if depot.ndim != 1 or depot.size(0) != 2:
        raise ValueError(f"Unexpected CVRP depot shape: {tuple(depot.shape)}")
    return loc, demand, depot


def _build_common_meta(raw_label_entry: Dict[str, Any]) -> Dict[str, Any]:
    meta = {}
    if not isinstance(raw_label_entry, dict):
        return meta
    if "ind" in raw_label_entry:
        meta["nss_best_method_index"] = int(raw_label_entry["ind"])
    if "cost" in raw_label_entry:
        meta["nss_costs"] = [float(x) for x in raw_label_entry["cost"]]
        meta["nss_best_cost"] = float(min(raw_label_entry["cost"]))
    if "time" in raw_label_entry:
        meta["nss_times"] = [float(x) for x in raw_label_entry["time"]]
    if "gap" in raw_label_entry:
        meta["nss_gaps"] = [float(x) for x in raw_label_entry["gap"]]
    return meta


def _rel_to_datasets(output_root: str, output_path: str) -> str:
    datasets_root = os.path.dirname(output_root)
    return os.path.relpath(output_path, datasets_root)


def _build_tsp_fixed_scale_files(
    dataset_name: str,
    dataset: List[Any],
    output_root: str,
) -> List[Dict[str, Any]]:
    scale_groups = defaultdict(list)
    for idx, item in enumerate(dataset):
        node_xy = _normalize_tsp_item(item)
        scale_groups[int(node_xy.size(0))].append(
            {
                "global_index": idx,
                "name": f"{dataset_name}_{idx:05d}",
                "node_xy": node_xy,
            }
        )

    fixed_root = os.path.join(output_root, FIXED_SCALE_GRAPH_DIR, dataset_name)
    os.makedirs(fixed_root, exist_ok=True)

    fixed_files = []
    for scale in sorted(scale_groups):
        items = scale_groups[scale]
        tensor = torch.stack([entry["node_xy"] for entry in items], dim=0)
        output_path = os.path.join(fixed_root, f"{dataset_name}_scale_{scale}.pt")
        torch.save({"node_xy": tensor}, output_path)
        fixed_files.append(
            {
                "scale": scale,
                "num_instances": len(items),
                "relative_path": _rel_to_datasets(output_root, output_path),
                "global_indices": [entry["global_index"] for entry in items],
                "names": [entry["name"] for entry in items],
            }
        )
    return fixed_files


def convert_one_dataset(source_root: str, output_root: str, dataset_name: str) -> Dict[str, Any]:
    spec = DATASET_SPECS[dataset_name]
    dataset_path = os.path.join(source_root, dataset_name, "dataset.pkl")
    raw_label_path = os.path.join(source_root, dataset_name, "raw_label.pkl")
    dataset = _load_pickle(dataset_path)
    raw_labels = _load_pickle(raw_label_path)

    converted: Dict[str, Any] = {
        "source_name": dataset_name,
        "problem": spec["problem"],
        "distribution_list": [],
        "scale_list": [],
        "dataset": {},
    }
    if spec["problem"] == "cvrp":
        converted["capacity_list"] = []

    for idx, item in enumerate(dataset):
        raw_label_entry = raw_labels.get(str(idx), {})
        sample_name = f"{dataset_name}_{idx:05d}"
        key = f"num{idx}"

        if spec["problem"] == "tsp":
            node_xy = _normalize_tsp_item(item)
            record = {
                "name": sample_name,
                "node_xy": _to_list_tensor(node_xy),
                **_build_common_meta(raw_label_entry),
            }
            scale = int(node_xy.size(0))
        else:
            node_xy, node_demand, depot_xy = _normalize_cvrp_item(item)
            record = {
                "name": sample_name,
                "depot_xy": _to_list_tensor(depot_xy),
                "node_xy": _to_list_tensor(node_xy),
                "node_demand": _to_list_tensor(node_demand),
                "capacity": 1.0,
                **_build_common_meta(raw_label_entry),
            }
            scale = int(node_xy.size(0))
            converted["capacity_list"].append(1.0)

        converted["dataset"][key] = record
        converted["scale_list"].append(scale)
        converted["distribution_list"].append(spec["distribution"])

    os.makedirs(output_root, exist_ok=True)
    output_path = os.path.join(
        output_root, f"nss_{_dataset_file_stem(dataset_name)}_varying.pt"
    )
    torch.save(converted, output_path)

    scale_list = converted["scale_list"]
    summary = {
        "dataset_name": dataset_name,
        "problem": spec["problem"],
        "output_path": output_path,
        "relative_output_path": _rel_to_datasets(output_root, output_path),
        "num_instances": len(scale_list),
        "min_scale": min(scale_list),
        "max_scale": max(scale_list),
        "unique_scales": len(set(scale_list)),
        "distribution": spec["distribution"],
    }
    if spec["problem"] == "cvrp":
        summary["capacity_values"] = sorted(set(converted["capacity_list"]))
    if spec["problem"] == "tsp":
        summary["fixed_scale_files"] = _build_tsp_fixed_scale_files(
            dataset_name=dataset_name,
            dataset=dataset,
            output_root=output_root,
        )
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-root",
        default=os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "9nss论文",
            "neural-solver-selection",
            "datasets",
        ),
    )
    parser.add_argument(
        "--output-root",
        default=os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "data",
            "datasets",
            "nss_varying",
        ),
    )
    args = parser.parse_args()

    manifest = {}
    for dataset_name in DATASET_SPECS:
        manifest[dataset_name] = convert_one_dataset(
            source_root=args.source_root,
            output_root=args.output_root,
            dataset_name=dataset_name,
        )
        print(
            f"converted {dataset_name}: "
            f"{manifest[dataset_name]['num_instances']} instances, "
            f"scale {manifest[dataset_name]['min_scale']}..{manifest[dataset_name]['max_scale']}"
        )

    manifest_path = os.path.join(args.output_root, "nss_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f"saved manifest to {manifest_path}")


if __name__ == "__main__":
    main()
