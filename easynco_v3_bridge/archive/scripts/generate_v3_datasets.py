#!/usr/bin/env python
import argparse
import json
import pickle
import random
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
BRIDGE_ROOT = SCRIPT_DIR.parent
if str(BRIDGE_ROOT) not in sys.path:
    sys.path.insert(0, str(BRIDGE_ROOT))
if str(BRIDGE_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(BRIDGE_ROOT.parent))

from config import (
    ATSP_SPECS,
    BRIDGE_ROOT,
    default_mvrp_demand_scaler,
    EASYNCO_ROOT,
    MANIFEST_ROOT,
    MVRP_MODE,
    MVRP_SPECS,
    MVRP_VARIANTS,
    NSS_COPY_DATASET_NAMES,
    NSS_COPY_MANIFEST,
    NSS_DATASETS,
    NSS_SOURCE_MANIFEST,
    NSS_SOURCE_ROOT,
    PCTSP_SPECS,
    SEED,
    V3_ATSP_ROOT,
    V3_MVRP_ROOT,
    V3_NSS_COPY_ROOT,
    V3_PCTSP_ROOT,
    ensure_layout,
    relative_to_datasets,
    variant_dir_name,
)

if str(EASYNCO_ROOT) not in sys.path:
    sys.path.insert(0, str(EASYNCO_ROOT))

from EasyNCO.data.ATSPGenerator import ATSPGenerator
from EasyNCO.data.MVRPGenerator import MVRPGenerator
from EasyNCO.data.PCTSPGenerator import PCTSPGenerator


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def stable_text_offset(text: str) -> int:
    value = 0
    for idx, ch in enumerate(text):
        value += (idx + 1) * ord(ch)
    return value


def copy_file(src: Path, dst: Path, force: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and not force:
        return
    shutil.copy2(src, dst)


def copy_tree(src: Path, dst: Path, force: bool) -> None:
    if dst.exists() and force:
        shutil.rmtree(dst)
    if dst.exists():
        return
    shutil.copytree(src, dst)


def copy_nss_train_sets(force: bool) -> Path:
    ensure_layout()
    with NSS_SOURCE_MANIFEST.open("r", encoding="utf-8") as f:
        src_manifest = json.load(f)

    manifest_out: Dict[str, dict] = {}
    for dataset_name in NSS_DATASETS:
        meta = dict(src_manifest[dataset_name])
        copied_name = NSS_COPY_DATASET_NAMES[dataset_name]
        src_rel = Path(meta["relative_output_path"])
        src_file = NSS_SOURCE_ROOT.parent / src_rel
        dst_file = V3_NSS_COPY_ROOT / src_rel.name
        copy_file(src_file, dst_file, force=force)
        meta["output_path"] = str(dst_file)
        meta["relative_output_path"] = relative_to_datasets(dst_file)

        fixed_scale_files = []
        for entry in meta.get("fixed_scale_files", []):
            new_entry = dict(entry)
            src_fixed = NSS_SOURCE_ROOT.parent / Path(entry["relative_path"])
            dst_fixed = V3_NSS_COPY_ROOT / "graph_fixed_by_scale" / dataset_name / src_fixed.name
            copy_file(src_fixed, dst_fixed, force=force)
            new_entry["relative_path"] = relative_to_datasets(dst_fixed)
            fixed_scale_files.append(new_entry)
        if fixed_scale_files:
            meta["fixed_scale_files"] = fixed_scale_files

        meta["dataset_name"] = copied_name
        manifest_out[copied_name] = meta

    graph_src = NSS_SOURCE_ROOT / "graph_fixed_by_scale" / "TSPtrain"
    graph_dst = V3_NSS_COPY_ROOT / "graph_fixed_by_scale" / "TSPtrain"
    copy_tree(graph_src, graph_dst, force=force)

    with NSS_COPY_MANIFEST.open("w", encoding="utf-8") as f:
        json.dump(manifest_out, f, ensure_ascii=False, indent=2)

    bridge_copy = MANIFEST_ROOT / "nss_copy_manifest.json"
    shutil.copy2(NSS_COPY_MANIFEST, bridge_copy)
    return NSS_COPY_MANIFEST


def generate_atsp(force: bool, spec_override: Optional[Dict[int, int]] = None) -> Path:
    ensure_layout()
    specs = spec_override or ATSP_SPECS
    manifest = {
        "problem": "atsp",
        "source": "EasyNCO.data.ATSPGenerator",
        "seed": SEED,
        "min_scale": min(specs),
        "max_scale": max(specs),
        "shards": [],
    }

    for scale, count in sorted(specs.items()):
        seed_everything(SEED + scale)
        output_path = V3_ATSP_ROOT / f"atsp{scale}_nums{count}.pt"
        if not output_path.exists() or force:
            loader = ATSPGenerator(
                data_size=count,
                problem_size=scale,
                batch_size=min(count, 256),
                device="cpu",
            )
            data = torch.empty((count, scale, scale), dtype=torch.float32)
            for idx in range(count):
                data[idx] = loader.dataset[idx].cpu()
            output_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(data, output_path)

        manifest["shards"].append(
            {
                "scale": scale,
                "num_instances": count,
                "relative_path": relative_to_datasets(output_path),
                "format": "pt_tensor",
            }
        )

    manifest["total_instances"] = sum(item["num_instances"] for item in manifest["shards"])
    manifest_path = V3_ATSP_ROOT / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    shutil.copy2(manifest_path, MANIFEST_ROOT / "atsp_manifest.json")
    return manifest_path


def convert_mvrp_sample(sample: dict, variant: str):
    depot_node_xy = sample["depot_node_xy"].detach().cpu()
    depot_node_demand = sample["depot_node_demand"].detach().cpu().squeeze(-1)
    depot_xy = depot_node_xy[:1].tolist()
    node_xy = depot_node_xy[1:].tolist()
    node_demand = depot_node_demand[1:].tolist()
    capacity = 1.0
    route_limit = float(sample["route_limit"].detach().cpu().item())
    service_time = sample["depot_node_service_time"].detach().cpu()[1:].tolist()
    tw_start = sample["depot_node_tw_start"].detach().cpu()[1:].tolist()
    tw_end = sample["depot_node_tw_end"].detach().cpu()[1:].tolist()

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


def generate_mvrp(
    force: bool,
    mode: int,
    variants: Optional[List[str]] = None,
    spec_override: Optional[Dict[int, int]] = None,
) -> Path:
    ensure_layout()
    specs = spec_override or MVRP_SPECS
    variants = variants or list(MVRP_VARIANTS)
    global_manifest = {
        "problem_family": "mvrp_variants",
        "source": "EasyNCO.data.MVRPGenerator",
        "seed": SEED,
        "mode": mode,
        "min_scale": min(specs),
        "max_scale": max(specs),
        "variants": {},
    }

    for variant in variants:
        variant_root = V3_MVRP_ROOT / variant_dir_name(variant)
        variant_root.mkdir(parents=True, exist_ok=True)
        variant_manifest = {
            "problem": variant,
            "source": "EasyNCO.data.MVRPGenerator",
            "seed": SEED,
            "mode": mode,
            "min_scale": min(specs),
            "max_scale": max(specs),
            "shards": [],
        }
        for scale, count in sorted(specs.items()):
            demand_scaler = default_mvrp_demand_scaler(scale)
            seed_everything(SEED + scale * 1000 + stable_text_offset(variant))
            output_path = variant_root / f"{variant_dir_name(variant)}{scale}_nums{count}.pkl"
            if not output_path.exists() or force:
                loader = MVRPGenerator(
                    data_size=count,
                    problem_size=scale,
                    batch_size=min(count, 128),
                    device="cpu",
                    demand_scaler=demand_scaler,
                    train_problems=[variant],
                    mode=mode,
                )
                rows = []
                for idx in range(count):
                    rows.append(convert_mvrp_sample(loader.dataset[idx], variant))
                with output_path.open("wb") as f:
                    pickle.dump(rows, f, pickle.HIGHEST_PROTOCOL)
            variant_manifest["shards"].append(
                {
                    "scale": scale,
                    "num_instances": count,
                    "demand_scaler": demand_scaler,
                    "relative_path": relative_to_datasets(output_path),
                    "format": "variant_pickle",
                }
            )

        variant_manifest["total_instances"] = sum(
            item["num_instances"] for item in variant_manifest["shards"]
        )
        manifest_path = variant_root / "manifest.json"
        with manifest_path.open("w", encoding="utf-8") as f:
            json.dump(variant_manifest, f, ensure_ascii=False, indent=2)
        global_manifest["variants"][variant] = variant_manifest

    global_path = V3_MVRP_ROOT / "manifest.json"
    with global_path.open("w", encoding="utf-8") as f:
        json.dump(global_manifest, f, ensure_ascii=False, indent=2)
    shutil.copy2(global_path, MANIFEST_ROOT / "mvrp_manifest.json")
    return global_path


def generate_pctsp(force: bool, spec_override: Optional[Dict[int, int]] = None) -> Path:
    ensure_layout()
    specs = spec_override or PCTSP_SPECS
    manifest = {
        "problem": "pctsp",
        "source": "EasyNCO.data.PCTSPGenerator",
        "seed": SEED,
        "shards": [],
    }

    for scale, count in sorted(specs.items()):
        seed_everything(SEED + scale)
        output_path = V3_PCTSP_ROOT / f"pctsp{scale}_nums{count}.pt"
        if not output_path.exists() or force:
            loader = PCTSPGenerator(
                data_size=count,
                problem_size=scale,
                batch_size=min(count, 256),
                device="cpu",
            )
            data = torch.empty((count, scale + 1, 4), dtype=torch.float32)
            for idx in range(count):
                data[idx] = loader.dataset[idx].cpu()
            output_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(data, output_path)
        manifest["shards"].append(
            {
                "scale": scale,
                "num_instances": count,
                "relative_path": relative_to_datasets(output_path),
                "format": "pt_tensor",
            }
        )

    manifest["total_instances"] = sum(item["num_instances"] for item in manifest["shards"])
    manifest_path = V3_PCTSP_ROOT / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    shutil.copy2(manifest_path, MANIFEST_ROOT / "pctsp_manifest.json")
    return manifest_path


def parse_scale_overrides(items: List[str]) -> Dict[int, int]:
    specs = {}
    for item in items:
        scale_text, count_text = item.split(":", 1)
        specs[int(scale_text)] = int(count_text)
    return specs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate offline_init_v3 datasets by calling existing EasyNCO generators."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    copy_parser = subparsers.add_parser("copy-nss")
    copy_parser.add_argument("--force", action="store_true")

    atsp_parser = subparsers.add_parser("generate-atsp")
    atsp_parser.add_argument("--force", action="store_true")
    atsp_parser.add_argument("--spec", action="append", default=[], help="scale:count override")

    mvrp_parser = subparsers.add_parser("generate-mvrp")
    mvrp_parser.add_argument("--force", action="store_true")
    mvrp_parser.add_argument("--mode", type=int, default=MVRP_MODE)
    mvrp_parser.add_argument("--variant", action="append", default=[], help="Repeat to limit variants.")
    mvrp_parser.add_argument("--spec", action="append", default=[], help="scale:count override")

    pctsp_parser = subparsers.add_parser("generate-pctsp")
    pctsp_parser.add_argument("--force", action="store_true")
    pctsp_parser.add_argument("--spec", action="append", default=[], help="scale:count override")

    all_parser = subparsers.add_parser("all")
    all_parser.add_argument("--force", action="store_true")
    all_parser.add_argument("--mode", type=int, default=MVRP_MODE)
    all_parser.add_argument("--with-pctsp", action="store_true")

    args = parser.parse_args()
    ensure_layout()

    if args.command == "copy-nss":
        manifest = copy_nss_train_sets(force=args.force)
        print(f"NSS train datasets copied: {manifest}")
    elif args.command == "generate-atsp":
        manifest = generate_atsp(force=args.force, spec_override=parse_scale_overrides(args.spec))
        print(f"ATSP manifest written: {manifest}")
    elif args.command == "generate-mvrp":
        manifest = generate_mvrp(
            force=args.force,
            mode=args.mode,
            variants=args.variant or None,
            spec_override=parse_scale_overrides(args.spec),
        )
        print(f"MVRP manifest written: {manifest}")
    elif args.command == "generate-pctsp":
        manifest = generate_pctsp(force=args.force, spec_override=parse_scale_overrides(args.spec))
        print(f"PCTSP manifest written: {manifest}")
    elif args.command == "all":
        copy_nss_train_sets(force=args.force)
        generate_atsp(force=args.force)
        generate_mvrp(force=args.force, mode=args.mode)
        if args.with_pctsp:
            generate_pctsp(force=args.force)
        print(f"All requested V3 datasets are ready under: {V3_NSS_COPY_ROOT.parent}")


if __name__ == "__main__":
    main()
