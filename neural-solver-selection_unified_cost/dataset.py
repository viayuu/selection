import glob
import os
import pickle

import numpy as np
import torch
from torch.utils.data import Dataset

from utils import augment_xy_by_8_fold, resolve_path


DEFAULT_NODE_FEATURE_DIM = 12
DEFAULT_PROBLEM_FEATURE_DIM = 10
DEFAULT_INSTANCE_STATS_DIM = 10
SPLIT_SUFFIXES = [("train", "train"), ("val", "val"), ("test", "test"), ("LIB", "lib")]
MVRP_BASIC = {"CVRP", "OVRP", "VRPB", "OVRPB"}
MVRP_LENGTH = {"VRPL", "OVRPL", "VRPBL", "OVRPBL"}
MVRP_TW = {"VRPTW", "OVRPTW", "VRPBTW", "OVRPBTW"}
MVRP_LENGTH_TW = {"VRPLTW", "OVRPLTW", "VRPBLTW", "OVRPBLTW"}


def canonicalize_solver_name(solver_name):
    return solver_name.strip().upper()


def canonical_problem_name(dataset_name):
    for suffix, split_name in SPLIT_SUFFIXES:
        if dataset_name.endswith(suffix):
            return dataset_name[: -len(suffix)].upper(), split_name
    return dataset_name.upper(), "unknown"


def build_problem_feature(problem_name):
    name = problem_name.upper()
    features = torch.zeros(DEFAULT_PROBLEM_FEATURE_DIM, dtype=torch.float32)
    features[0] = 0.0 if name == "ATSP" else 1.0  # coordinate geometry available
    features[1] = 1.0 if ("VRP" in name or name == "CVRP") else 0.0  # demand / capacity
    features[2] = 1.0 if name == "PCTSP" else 0.0  # prize
    features[3] = 1.0 if name == "PCTSP" else 0.0  # penalty
    features[4] = 1.0 if "TW" in name else 0.0  # time window
    features[5] = 1.0 if name.startswith("O") else 0.0  # open route
    features[6] = 1.0 if "B" in name and name != "PCTSP" else 0.0  # backhaul
    features[7] = 1.0 if "L" in name else 0.0  # route length limit
    features[8] = 1.0 if name == "ATSP" else 0.0  # asymmetry
    features[9] = 1.0 if name == "ATSP" else 0.0  # matrix-style input
    return features


def normalize_dataset_specs(dataset_specs, base_dir):
    normalized = []
    for spec in dataset_specs:
        spec_norm = dict(spec)
        spec_norm["dataset_dir"] = resolve_path(base_dir, spec_norm["dataset_dir"])
        spec_norm["problem_type"] = spec_norm["problem_type"].upper()
        spec_norm.setdefault("name", os.path.basename(spec_norm["dataset_dir"]))
        spec_norm.setdefault("problem_features", build_problem_feature(spec_norm["problem_type"]))
        normalized.append(spec_norm)
    return normalized


def infer_solver_names(dataset_dir):
    solver_names = list(load_solver_result_tables(dataset_dir).keys())
    if not solver_names:
        raise FileNotFoundError(f"No solver result files found under {dataset_dir}/results")
    return solver_names


def load_solver_result_tables(dataset_dir):
    result_paths = sorted(glob.glob(os.path.join(dataset_dir, "results", "result_*.txt")))
    solver_rows = {}
    for path in result_paths:
        file_name = os.path.basename(path)
        solver_name = file_name[len("result_") : -len(".txt")]
        solver_name = canonicalize_solver_name(solver_name)
        if solver_name == "OPT":
            continue
        solver_rows[solver_name] = read_result_file(path)
    return solver_rows


def discover_export_dataset_groups(export_root, include_problems=None, exclude_problems=None):
    include_set = None if not include_problems else {name.upper() for name in include_problems}
    exclude_set = set() if not exclude_problems else {name.upper() for name in exclude_problems}
    groups = {}

    for entry in sorted(os.listdir(export_root)):
        dataset_dir = os.path.join(export_root, entry)
        if not os.path.isdir(dataset_dir):
            continue
        if not os.path.exists(os.path.join(dataset_dir, "dataset.pkl")):
            continue

        problem_name, split_name = canonical_problem_name(entry)
        if include_set is not None and problem_name not in include_set:
            continue
        if problem_name in exclude_set:
            continue

        groups.setdefault(problem_name, {})[split_name] = dataset_dir
    return groups


def _dataset_size(dataset_dir):
    with open(os.path.join(dataset_dir, "dataset.pkl"), "rb") as file_obj:
        data = pickle.load(file_obj)
    return len(data)


def _make_split_indices(num_instances, seed, val_ratio, test_ratio):
    rng = np.random.default_rng(seed)
    indices = rng.permutation(num_instances)
    val_count = int(num_instances * val_ratio)
    test_count = int(num_instances * test_ratio)
    train_count = max(num_instances - val_count - test_count, 1)
    val_end = train_count + val_count
    train_indices = indices[:train_count].tolist()
    val_indices = indices[train_count:val_end].tolist()
    test_indices = indices[val_end:].tolist()
    if len(val_indices) == 0 and num_instances >= 3:
        val_indices = train_indices[-1:]
        train_indices = train_indices[:-1]
    if len(test_indices) == 0 and num_instances >= 4:
        test_indices = train_indices[-1:]
        train_indices = train_indices[:-1]
    return train_indices, val_indices, test_indices


def _truncate_indices(indices, max_count):
    if max_count is None:
        return indices
    return indices[: min(len(indices), int(max_count))]


def build_dataset_specs_from_export_root(export_root, data_params):
    export_root = os.path.abspath(export_root)
    groups = discover_export_dataset_groups(
        export_root,
        include_problems=data_params.get("include_problems"),
        exclude_problems=data_params.get("exclude_problems"),
    )
    split_cfg = data_params.get("split_missing_from_train", {})
    split_seed = int(split_cfg.get("seed", 2024))
    val_ratio = float(split_cfg.get("val_ratio", 0.1))
    test_ratio = float(split_cfg.get("test_ratio", 0.1))
    max_train_samples = data_params.get("max_train_samples")
    max_eval_samples = data_params.get("max_eval_samples")

    train_specs = []
    val_specs = []
    test_specs = []
    benchmark_specs = []

    for problem_name in sorted(groups):
        split_dirs = groups[problem_name]
        train_dir = split_dirs.get("train")
        val_dir = split_dirs.get("val")
        test_dir = split_dirs.get("test")
        lib_dir = split_dirs.get("lib")

        if train_dir is None:
            continue

        solver_names = infer_solver_names(train_dir)
        base_spec = {
            "problem_type": problem_name,
            "solver_names": solver_names,
            "problem_features": build_problem_feature(problem_name),
        }

        if val_dir is None or test_dir is None:
            num_instances = _dataset_size(train_dir)
            local_seed = split_seed + sum(ord(ch) for ch in problem_name)
            train_idx, derived_val_idx, derived_test_idx = _make_split_indices(
                num_instances, local_seed, val_ratio, test_ratio
            )
            train_specs.append(
                {
                    **base_spec,
                    "dataset_dir": train_dir,
                    "name": f"{problem_name}train_split",
                    "sample_indices": _truncate_indices(train_idx, max_train_samples),
                }
            )
            if val_dir is None and derived_val_idx:
                val_specs.append(
                    {
                        **base_spec,
                        "dataset_dir": train_dir,
                        "name": f"{problem_name}val_split",
                        "sample_indices": _truncate_indices(derived_val_idx, max_eval_samples),
                    }
                )
            if test_dir is None and derived_test_idx:
                test_specs.append(
                    {
                        **base_spec,
                        "dataset_dir": train_dir,
                        "name": f"{problem_name}test_split",
                        "sample_indices": _truncate_indices(derived_test_idx, max_eval_samples),
                    }
                )
        else:
            train_specs.append(
                {
                    **base_spec,
                    "dataset_dir": train_dir,
                    "name": os.path.basename(train_dir),
                    "max_samples": max_train_samples,
                }
            )

        if val_dir is not None:
            val_specs.append(
                {
                    **base_spec,
                    "dataset_dir": val_dir,
                    "name": os.path.basename(val_dir),
                    "max_samples": max_eval_samples,
                    "solver_names": infer_solver_names(val_dir),
                }
            )
        if test_dir is not None:
            test_specs.append(
                {
                    **base_spec,
                    "dataset_dir": test_dir,
                    "name": os.path.basename(test_dir),
                    "max_samples": max_eval_samples,
                    "solver_names": infer_solver_names(test_dir),
                }
            )
        if lib_dir is not None:
            benchmark_specs.append(
                {
                    **base_spec,
                    "dataset_dir": lib_dir,
                    "name": os.path.basename(lib_dir),
                    "max_samples": max_eval_samples,
                    "solver_names": infer_solver_names(lib_dir),
                }
            )

    return train_specs, val_specs, test_specs, benchmark_specs


def build_solver_pool(dataset_specs):
    solver_pool = []
    seen = set()
    for spec in dataset_specs:
        solver_names = [canonicalize_solver_name(name) for name in spec.get("solver_names", infer_solver_names(spec["dataset_dir"]))]
        solver_names = list(dict.fromkeys(solver_names))
        spec["solver_names"] = solver_names
        for solver_name in solver_names:
            if solver_name not in seen:
                seen.add(solver_name)
                solver_pool.append(solver_name)
    return solver_pool


def build_problem_pool(dataset_specs):
    pool = []
    for spec in dataset_specs:
        if spec["problem_type"] not in pool:
            pool.append(spec["problem_type"])
    return pool


def build_solver_features(solver_pool, dataset_specs, problem_pool):
    solver_to_idx = {solver_name: idx for idx, solver_name in enumerate(solver_pool)}
    problem_to_idx = {problem_name: idx for idx, problem_name in enumerate(problem_pool)}
    solver_features = torch.zeros(len(solver_pool), len(problem_pool), dtype=torch.float32)
    for spec in dataset_specs:
        problem_idx = problem_to_idx[spec["problem_type"]]
        for solver_name in spec["solver_names"]:
            solver_features[solver_to_idx[solver_name], problem_idx] = 1.0
    return solver_features


def read_result_file(file_path):
    rows = {}
    with open(file_path, "r", encoding="utf-8") as file_obj:
        for raw_line in file_obj:
            line = raw_line.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) < 2:
                raise ValueError(f"Unexpected label format in {file_path}: {raw_line}")
            instance_id = str(parts[0])
            cost = float(parts[1])
            time_value = float(parts[2]) if len(parts) >= 3 else 0.0
            rows[instance_id] = {"cost": cost, "time": time_value}
    return rows


def encode_tsp_instance(instance, node_feature_dim):
    nodes = instance.squeeze(0).float()
    encoded = torch.zeros(nodes.size(0), node_feature_dim, dtype=torch.float32)
    encoded[:, :2] = nodes[:, :2]
    return encoded, True


def encode_atsp_instance(instance, node_feature_dim):
    matrix = instance.squeeze(0).float()
    row = matrix
    col = matrix.transpose(0, 1)
    encoded = torch.zeros(matrix.size(0), node_feature_dim, dtype=torch.float32)
    encoded[:, 0] = row.mean(dim=1)
    encoded[:, 1] = row.min(dim=1).values
    encoded[:, 2] = row.max(dim=1).values
    encoded[:, 3] = row.std(dim=1, unbiased=False)
    encoded[:, 4] = col.mean(dim=1)
    encoded[:, 5] = col.min(dim=1).values
    encoded[:, 6] = col.max(dim=1).values
    encoded[:, 7] = col.std(dim=1, unbiased=False)
    if matrix.size(0) > 1 and node_feature_dim > 9:
        encoded[:, 9] = torch.arange(matrix.size(0), dtype=torch.float32) / float(matrix.size(0) - 1)
    return encoded, False


def encode_pctsp_instance(instance, node_feature_dim):
    nodes = instance.squeeze(0).float()
    encoded = torch.zeros(nodes.size(0), node_feature_dim, dtype=torch.float32)
    encoded[:, :2] = nodes[:, :2]
    encoded[:, 3] = nodes[:, 2]
    encoded[:, 4] = nodes[:, 3]
    encoded[0, 8] = 1.0
    return encoded, True


def encode_cvrp_dict_instance(instance, node_feature_dim):
    depot = instance["depot"].squeeze(0).float()
    loc = instance["loc"].squeeze(0).float()
    demand = instance["demand"].squeeze(0).float()

    num_nodes = loc.size(0) + 1
    encoded = torch.zeros(num_nodes, node_feature_dim, dtype=torch.float32)
    encoded[0, :2] = depot[0]
    encoded[1:, :2] = loc
    encoded[1:, 2] = demand
    encoded[0, 8] = 1.0
    return encoded, True


def _to_tensor_list(values):
    return torch.tensor(values, dtype=torch.float32)


def parse_mvrp_tuple(problem_name, instance):
    problem_name = problem_name.upper()
    if problem_name in MVRP_BASIC:
        depot_xy, node_xy, node_demand, capacity = instance
        route_limit = 0.0
        service_time = None
        tw_start = None
        tw_end = None
    elif problem_name in MVRP_LENGTH:
        depot_xy, node_xy, node_demand, capacity, route_limit = instance
        service_time = None
        tw_start = None
        tw_end = None
    elif problem_name in MVRP_TW:
        depot_xy, node_xy, node_demand, capacity, service_time, tw_start, tw_end = instance
        route_limit = 0.0
    elif problem_name in MVRP_LENGTH_TW:
        depot_xy, node_xy, node_demand, capacity, route_limit, service_time, tw_start, tw_end = instance
    else:
        raise ValueError(f"Unsupported MVRP tuple problem: {problem_name}")

    demand = _to_tensor_list(node_demand)
    capacity = float(capacity)
    if abs(capacity) > 1e-12:
        demand = demand / capacity

    return {
        "depot_xy": _to_tensor_list(depot_xy),
        "node_xy": _to_tensor_list(node_xy),
        "node_demand": demand,
        "route_limit": float(route_limit),
        "service_time": None if service_time is None else _to_tensor_list(service_time),
        "tw_start": None if tw_start is None else _to_tensor_list(tw_start),
        "tw_end": None if tw_end is None else _to_tensor_list(tw_end),
    }


def encode_mvrp_instance(problem_name, instance, node_feature_dim):
    parsed = parse_mvrp_tuple(problem_name, instance)
    depot = parsed["depot_xy"].reshape(1, 2)
    loc = parsed["node_xy"]
    demand = parsed["node_demand"]

    num_nodes = loc.size(0) + 1
    encoded = torch.zeros(num_nodes, node_feature_dim, dtype=torch.float32)
    encoded[0, :2] = depot[0]
    encoded[1:, :2] = loc
    encoded[1:, 2] = demand
    if parsed["service_time"] is not None:
        encoded[1:, 5] = parsed["service_time"]
    if parsed["tw_start"] is not None:
        encoded[1:, 6] = parsed["tw_start"]
    if parsed["tw_end"] is not None:
        encoded[1:, 7] = parsed["tw_end"]
    encoded[0, 8] = 1.0
    encoded[:, 9] = parsed["route_limit"]
    return encoded, True


def encode_instance(problem_type, instance, node_feature_dim):
    problem_name = problem_type.upper()
    if problem_name == "TSP":
        return encode_tsp_instance(instance, node_feature_dim)
    if problem_name == "ATSP":
        return encode_atsp_instance(instance, node_feature_dim)
    if problem_name == "PCTSP":
        return encode_pctsp_instance(instance, node_feature_dim)
    if isinstance(instance, dict):
        return encode_cvrp_dict_instance(instance, node_feature_dim)
    if isinstance(instance, tuple):
        return encode_mvrp_instance(problem_name, instance, node_feature_dim)
    raise ValueError(f"Unsupported raw instance type for {problem_type}: {type(instance)}")


def compute_instance_stats(problem_type, raw_instance, encoded_nodes):
    problem_name = problem_type.upper()
    stats = torch.zeros(DEFAULT_INSTANCE_STATS_DIM, dtype=torch.float32)

    if problem_name == "ATSP":
        matrix = raw_instance.squeeze(0).float()
        asym = (matrix - matrix.transpose(0, 1)).abs()
        row_mean = matrix.mean(dim=1)
        col_mean = matrix.mean(dim=0)
        stats[0] = row_mean.mean()
        stats[1] = row_mean.std(unbiased=False)
        stats[2] = matrix.min(dim=1).values.mean()
        stats[3] = matrix.max(dim=1).values.mean()
        stats[4] = col_mean.mean()
        stats[5] = col_mean.std(unbiased=False)
        stats[6] = asym.mean()
        stats[7] = asym.std(unbiased=False)
        stats[8] = matrix.mean()
        stats[9] = matrix.std(unbiased=False)
        return stats

    xy = encoded_nodes[:, :2]
    stats[0] = xy[:, 0].mean()
    stats[1] = xy[:, 0].std(unbiased=False)
    stats[2] = xy[:, 1].mean()
    stats[3] = xy[:, 1].std(unbiased=False)

    demand = encoded_nodes[:, 2]
    stats[4] = demand.mean()
    stats[5] = demand.std(unbiased=False)

    if problem_name == "PCTSP":
        stats[6] = encoded_nodes[:, 3].mean()
        stats[7] = encoded_nodes[:, 3].std(unbiased=False)
        stats[8] = encoded_nodes[:, 4].mean()
        stats[9] = np.log1p(encoded_nodes.size(0))
    else:
        tw_span = torch.clamp(encoded_nodes[:, 7] - encoded_nodes[:, 6], min=0.0)
        stats[6] = encoded_nodes[:, 5].mean()
        stats[7] = tw_span.mean()
        stats[8] = encoded_nodes[:, 9].max()
        stats[9] = np.log1p(encoded_nodes.size(0))

    return stats


class UnifiedSelectionDataset(Dataset):
    def __init__(
        self,
        dataset_specs,
        solver_pool,
        problem_pool,
        node_feature_dim=DEFAULT_NODE_FEATURE_DIM,
        data_aug=False,
    ):
        super().__init__()
        self.dataset_specs = dataset_specs
        self.solver_pool = solver_pool
        self.problem_pool = problem_pool
        self.node_feature_dim = node_feature_dim
        self.data_aug = data_aug
        self.solver_to_idx = {solver_name: idx for idx, solver_name in enumerate(self.solver_pool)}
        self.problem_to_idx = {problem_name: idx for idx, problem_name in enumerate(self.problem_pool)}
        self.spec_cache = []
        self.entries = []
        self._load_all_specs()

    def _load_all_specs(self):
        for spec_id, spec in enumerate(self.dataset_specs):
            with open(os.path.join(spec["dataset_dir"], "dataset.pkl"), "rb") as file_obj:
                raw_instances = pickle.load(file_obj)

            solver_rows = load_solver_result_tables(spec["dataset_dir"])
            spec["solver_names"] = [canonicalize_solver_name(name) for name in spec["solver_names"]]
            self.spec_cache.append({"spec": spec, "raw_instances": raw_instances, "solver_rows": solver_rows})

            if spec.get("sample_indices") is not None:
                selected_indices = list(spec["sample_indices"])
            else:
                max_samples = spec.get("max_samples")
                total_instances = len(raw_instances) if max_samples is None else min(len(raw_instances), int(max_samples))
                selected_indices = list(range(total_instances))

            for instance_idx in selected_indices:
                if self.data_aug and spec["problem_type"].upper() != "ATSP":
                    for aug_idx in range(8):
                        self.entries.append({"spec_id": spec_id, "instance_idx": instance_idx, "aug_idx": aug_idx})
                else:
                    self.entries.append({"spec_id": spec_id, "instance_idx": instance_idx, "aug_idx": None})

    def _build_sample(self, spec, raw_instance, instance_idx, solver_rows):
        encoded_nodes, can_aug = encode_instance(spec["problem_type"], raw_instance, self.node_feature_dim)
        instance_stats = compute_instance_stats(spec["problem_type"], raw_instance, encoded_nodes)
        costs = torch.full((len(self.solver_pool),), float("inf"), dtype=torch.float32)
        times = torch.zeros(len(self.solver_pool), dtype=torch.float32)
        feasible_mask = torch.zeros(len(self.solver_pool), dtype=torch.bool)

        instance_key = str(instance_idx)
        for solver_name in spec["solver_names"]:
            result = solver_rows[solver_name].get(instance_key)
            if result is None:
                continue
            solver_idx = self.solver_to_idx[solver_name]
            costs[solver_idx] = result["cost"]
            times[solver_idx] = result["time"]
            feasible_mask[solver_idx] = True

        if not feasible_mask.any():
            raise ValueError(
                f"No feasible solver label found for instance {instance_idx} in {spec['dataset_dir']}"
            )

        label = torch.argmin(costs).long()
        return {
            "nodes": encoded_nodes,
            "label": label,
            "costs": costs,
            "times": times,
            "feasible_mask": feasible_mask,
            "problem_id": torch.tensor(self.problem_to_idx[spec["problem_type"]], dtype=torch.long),
            "problem_features": spec["problem_features"].clone(),
            "problem_name": spec["problem_type"],
            "scale": torch.tensor(float(encoded_nodes.size(0)), dtype=torch.float32),
            "instance_stats": instance_stats,
            "dataset_name": spec["name"],
            "instance_id": torch.tensor(instance_idx, dtype=torch.long),
            "can_aug": can_aug,
        }

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, index):
        entry = self.entries[index]
        cache = self.spec_cache[entry["spec_id"]]
        spec = cache["spec"]
        raw_instance = cache["raw_instances"][entry["instance_idx"]]
        sample = self._build_sample(spec, raw_instance, entry["instance_idx"], cache["solver_rows"])
        if entry["aug_idx"] is not None:
            aug_coords = augment_xy_by_8_fold(sample["nodes"][:, :2].unsqueeze(0))
            sample["nodes"] = sample["nodes"].clone()
            sample["nodes"][:, :2] = aug_coords[entry["aug_idx"]]
        return sample


def collate_fn(batch):
    batch_size = len(batch)
    node_feature_dim = batch[0]["nodes"].size(-1)
    lengths = torch.tensor([sample["nodes"].size(0) for sample in batch], dtype=torch.float32)
    max_length = int(lengths.max().item())

    batch_x = torch.zeros(batch_size, max_length, node_feature_dim, dtype=torch.float32)
    node_mask = torch.zeros(batch_size, max_length, dtype=torch.float32)
    for idx, sample in enumerate(batch):
        current_length = sample["nodes"].size(0)
        batch_x[idx, :current_length] = sample["nodes"]
        if current_length < max_length:
            node_mask[idx, current_length:] = float("-inf")

    return {
        "nodes": batch_x,
        "labels": torch.stack([sample["label"] for sample in batch]),
        "costs": torch.stack([sample["costs"] for sample in batch]),
        "times": torch.stack([sample["times"] for sample in batch]),
        "feasible_mask": torch.stack([sample["feasible_mask"] for sample in batch]),
        "problem_ids": torch.stack([sample["problem_id"] for sample in batch]),
        "problem_features": torch.stack([sample["problem_features"] for sample in batch]),
        "problem_names": [sample["problem_name"] for sample in batch],
        "scales": torch.stack([sample["scale"] for sample in batch]),
        "instance_stats": torch.stack([sample["instance_stats"] for sample in batch]),
        "node_mask": node_mask,
        "instance_ids": torch.stack([sample["instance_id"] for sample in batch]),
        "dataset_names": [sample["dataset_name"] for sample in batch],
        "lengths": lengths,
    }
