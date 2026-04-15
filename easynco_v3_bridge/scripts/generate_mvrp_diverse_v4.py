#!/usr/bin/env python
import argparse
import csv
import importlib.util
import json
import math
import pickle
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
BRIDGE_ROOT = SCRIPT_DIR.parent
if str(BRIDGE_ROOT) not in sys.path:
    sys.path.insert(0, str(BRIDGE_ROOT))
if str(BRIDGE_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(BRIDGE_ROOT.parent))

from config import DATASETS_ROOT, MVRP_SPECS, MVRP_VARIANTS, SEED, default_mvrp_demand_scaler

EASYNCO_ROOT = BRIDGE_ROOT.parent / "EasyNCO"
MVRP_GENERATOR_PATH = EASYNCO_ROOT / "data" / "MVRPGenerator.py"


def load_get_tw_data_1():
    spec = importlib.util.spec_from_file_location("mvrp_generator_module", MVRP_GENERATOR_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.get_tw_data_1


get_tw_data_1 = load_get_tw_data_1()


V4_DATA_ROOT = DATASETS_ROOT / "offline_init_v4"
V4_MVRP_DIVERSE_ROOT = V4_DATA_ROOT / "mvrp_diverse_v1"
DISTRIBUTIONS = ("uniform", "gaussian_mixture", "clustered", "ring")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def stable_text_offset(text: str) -> int:
    value = 0
    for idx, ch in enumerate(text):
        value += (idx + 1) * ord(ch)
    return value


def variant_dir_name(variant: str) -> str:
    return variant.lower()


def normalize_points(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)
    mins = points.min(axis=0, keepdims=True)
    maxs = points.max(axis=0, keepdims=True)
    spans = np.maximum(maxs - mins, 1e-6)
    normalized = (points - mins) / spans
    return np.clip(normalized, 0.0, 1.0)


def sample_uniform_coords(num_nodes: int) -> np.ndarray:
    return np.random.rand(num_nodes, 2).astype(np.float32)


def sample_gaussian_mixture_coords(num_nodes: int, tight: bool) -> np.ndarray:
    if tight:
        mode_low, mode_high = 3, 9
        var_low, var_high = 0.3, 8.0
    else:
        mode_low, mode_high = 2, 7
        var_low, var_high = 1.0, 25.0

    num_modes = np.random.randint(mode_low, mode_high + 1)
    mix = np.random.dirichlet(np.ones(num_modes))
    counts = np.random.multinomial(num_nodes, mix)
    points = []

    for count in counts:
        if count == 0:
            continue
        center = np.random.uniform(0.0, 100.0, size=(2,))
        var_x = np.random.uniform(var_low, var_high)
        var_y = np.random.uniform(var_low, var_high)
        cov_xy = np.random.uniform(-0.35, 0.35) * math.sqrt(var_x * var_y)
        cov = np.array([[var_x, cov_xy], [cov_xy, var_y]], dtype=np.float32)
        coords = np.random.multivariate_normal(center, cov, size=count)
        points.append(coords)

    points = np.concatenate(points, axis=0)
    np.random.shuffle(points)
    return normalize_points(points)


def sample_ring_coords(num_nodes: int) -> np.ndarray:
    angles = np.random.uniform(0.0, 2.0 * math.pi, size=(num_nodes,))
    radii = np.random.uniform(0.22, 0.46, size=(num_nodes,))
    jitter = np.random.normal(loc=0.0, scale=0.03, size=(num_nodes, 2))
    base = np.stack(
        [
            0.5 + radii * np.cos(angles),
            0.5 + radii * np.sin(angles),
        ],
        axis=1,
    )
    points = np.clip(base + jitter, 0.0, 1.0)
    return points.astype(np.float32)


def sample_customer_coords(num_nodes: int, distribution: str) -> np.ndarray:
    if distribution == "uniform":
        return sample_uniform_coords(num_nodes)
    if distribution == "gaussian_mixture":
        return sample_gaussian_mixture_coords(num_nodes, tight=False)
    if distribution == "clustered":
        return sample_gaussian_mixture_coords(num_nodes, tight=True)
    if distribution == "ring":
        return sample_ring_coords(num_nodes)
    raise ValueError(f"Unsupported distribution: {distribution}")


def balanced_distribution_plan(count: int) -> List[str]:
    base = count // len(DISTRIBUTIONS)
    remainder = count % len(DISTRIBUTIONS)
    plan = []
    for idx, name in enumerate(DISTRIBUTIONS):
        plan.extend([name] * (base + (1 if idx < remainder else 0)))
    random.shuffle(plan)
    return plan


def build_scale_distribution_plans(scale_to_count: Dict[int, int], seed: int) -> Dict[int, List[str]]:
    total = sum(scale_to_count.values())
    global_plan = balanced_distribution_plan(total)
    rng = random.Random(seed)
    rng.shuffle(global_plan)

    plans = {}
    offset = 0
    for scale, count in sorted(scale_to_count.items()):
        plans[scale] = global_plan[offset : offset + count]
        offset += count
    return plans


def build_instance_dict(
    variant: str,
    scale: int,
    distribution: str,
    demand_scaler: int,
    backhaul_ratio: float = 0.2,
):
    customers = torch.tensor(sample_customer_coords(scale, distribution), dtype=torch.float32)
    depot = torch.rand((1, 2), dtype=torch.float32)
    depot_node_xy = torch.cat((depot, customers), dim=0)

    node_demand = torch.randint(1, 10, size=(scale,), dtype=torch.int64).float() / float(demand_scaler)
    if "B" in variant:
        backhaul_count = max(1, int(scale * backhaul_ratio))
        backhaul_index = torch.randperm(scale)[:backhaul_count]
        node_demand[backhaul_index] = -node_demand[backhaul_index]
    depot_node_demand = torch.cat((torch.zeros(1), node_demand), dim=0).unsqueeze(-1)

    route_limit = torch.tensor(3.0 if "L" in variant else 0.0, dtype=torch.float32)
    depot_node_service_time = torch.zeros(scale + 1, dtype=torch.float32)
    depot_node_tw_start = torch.zeros(scale + 1, dtype=torch.float32)
    depot_node_tw_end = torch.zeros(scale + 1, dtype=torch.float32)

    if "TW" in variant:
        depot_node_service_time, depot_node_tw_start, depot_node_tw_end = get_tw_data_1(
            scale,
            depot_node_xy,
            speed=1.0,
            depot_start=torch.tensor([0.0], dtype=torch.float32),
            depot_end=torch.tensor([3.0], dtype=torch.float32),
            device="cpu",
        )

    return {
        "depot_node_xy": depot_node_xy,
        "depot_node_demand": depot_node_demand,
        "route_limit": route_limit,
        "depot_node_service_time": depot_node_service_time,
        "depot_node_tw_start": depot_node_tw_start,
        "depot_node_tw_end": depot_node_tw_end,
    }


def pack_variant_sample(sample: dict, variant: str):
    depot_node_xy = sample["depot_node_xy"]
    depot_node_demand = sample["depot_node_demand"].squeeze(-1)
    depot_xy = depot_node_xy[:1].tolist()
    node_xy = depot_node_xy[1:].tolist()
    node_demand = depot_node_demand[1:].tolist()
    capacity = 1.0
    route_limit = float(sample["route_limit"].item())
    service_time = sample["depot_node_service_time"][1:].tolist()
    tw_start = sample["depot_node_tw_start"][1:].tolist()
    tw_end = sample["depot_node_tw_end"][1:].tolist()

    if variant in {"CVRP", "OVRP", "VRPB", "OVRPB"}:
        return (depot_xy, node_xy, node_demand, capacity)
    if variant in {"VRPL", "OVRPL", "VRPBL", "OVRPBL"}:
        return (depot_xy, node_xy, node_demand, capacity, route_limit)
    if variant in {"VRPTW", "OVRPTW", "VRPBTW", "OVRPBTW"}:
        return (depot_xy, node_xy, node_demand, capacity, service_time, tw_start, tw_end)
    if variant in {"VRPLTW", "OVRPLTW", "VRPBLTW", "OVRPBLTW"}:
        return (
            depot_xy,
            node_xy,
            node_demand,
            capacity,
            route_limit,
            service_time,
            tw_start,
            tw_end,
        )
    raise ValueError(f"Unsupported MVRP variant: {variant}")


def generate_variant_dataset(variant: str, force: bool) -> Dict:
    variant_root = V4_MVRP_DIVERSE_ROOT / variant_dir_name(variant)
    variant_root.mkdir(parents=True, exist_ok=True)

    variant_manifest = {
        "problem": variant,
        "source": "bridge_diverse_mvrp_generator_v1",
        "seed": SEED,
        "min_scale": min(MVRP_SPECS),
        "max_scale": max(MVRP_SPECS),
        "distributions": list(DISTRIBUTIONS),
        "shards": [],
    }
    meta_rows = []
    global_index = 0
    scale_distribution_plans = build_scale_distribution_plans(
        MVRP_SPECS,
        seed=SEED + stable_text_offset(variant),
    )

    for scale, count in sorted(MVRP_SPECS.items()):
        demand_scaler = default_mvrp_demand_scaler(scale)
        shard_seed = SEED + scale * 1000 + stable_text_offset(variant)
        seed_everything(shard_seed)
        shard_plan = scale_distribution_plans[scale]
        rows = []
        distribution_counts = {name: 0 for name in DISTRIBUTIONS}

        output_path = variant_root / f"{variant_dir_name(variant)}{scale}_nums{count}.pkl"
        meta_path = variant_root / f"{variant_dir_name(variant)}{scale}_nums{count}.meta.csv"

        if not output_path.exists() or force:
            for local_index, distribution in enumerate(shard_plan):
                sample = build_instance_dict(
                    variant=variant,
                    scale=scale,
                    distribution=distribution,
                    demand_scaler=demand_scaler,
                )
                rows.append(pack_variant_sample(sample, variant))
                distribution_counts[distribution] += 1
                meta_rows.append(
                    {
                        "instance_id": global_index + local_index,
                        "scale": scale,
                        "scale_local_index": local_index,
                        "distribution": distribution,
                        "source_relative_path": str(output_path.relative_to(DATASETS_ROOT)),
                    }
                )

            with output_path.open("wb") as f:
                pickle.dump(rows, f, pickle.HIGHEST_PROTOCOL)
            with meta_path.open("w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=[
                        "instance_id",
                        "scale",
                        "scale_local_index",
                        "distribution",
                        "source_relative_path",
                    ],
                )
                writer.writeheader()
                writer.writerows(meta_rows[-count:])
        else:
            with meta_path.open("r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                shard_meta = list(reader)
            for item in shard_meta:
                distribution_counts[item["distribution"]] += 1
            meta_rows.extend(shard_meta)

        variant_manifest["shards"].append(
            {
                "scale": scale,
                "num_instances": count,
                "demand_scaler": demand_scaler,
                "distribution_counts": distribution_counts,
                "relative_path": str(output_path.relative_to(DATASETS_ROOT)),
                "format": "variant_pickle",
            }
        )
        global_index += count

    variant_manifest["total_instances"] = global_index
    variant_manifest_path = variant_root / "manifest.json"
    variant_manifest_path.write_text(json.dumps(variant_manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    variant_meta_path = variant_root / "instance_meta.csv"
    with variant_meta_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "instance_id",
                "scale",
                "scale_local_index",
                "distribution",
                "source_relative_path",
            ],
        )
        writer.writeheader()
        writer.writerows(meta_rows)

    return variant_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate diverse multi-scale MVRP dataset (v4)")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--variants", nargs="*", default=list(MVRP_VARIANTS))
    args = parser.parse_args()

    V4_MVRP_DIVERSE_ROOT.mkdir(parents=True, exist_ok=True)
    global_manifest = {
        "problem_family": "mvrp_variants",
        "source": "bridge_diverse_mvrp_generator_v1",
        "seed": SEED,
        "scale_range": [min(MVRP_SPECS), max(MVRP_SPECS)],
        "specs": MVRP_SPECS,
        "distributions": list(DISTRIBUTIONS),
        "variants": {},
    }

    for variant in args.variants:
        print(f"[generate] {variant}", flush=True)
        global_manifest["variants"][variant] = generate_variant_dataset(variant, force=args.force)

    manifest_path = V4_MVRP_DIVERSE_ROOT / "manifest.json"
    manifest_path.write_text(json.dumps(global_manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote diverse MVRP dataset to {V4_MVRP_DIVERSE_ROOT}", flush=True)


if __name__ == "__main__":
    main()
