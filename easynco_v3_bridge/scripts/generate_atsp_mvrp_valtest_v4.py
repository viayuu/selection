#!/usr/bin/env python
import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
BRIDGE_ROOT = SCRIPT_DIR.parent
if str(BRIDGE_ROOT) not in sys.path:
    sys.path.insert(0, str(BRIDGE_ROOT))
if str(BRIDGE_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(BRIDGE_ROOT.parent))

from config import (
    DATASETS_ROOT,
    MVRP_MODE,
    MVRP_VARIANTS,
    SEED,
    build_dense_scale_specs,
)


EASYNCO_ROOT = BRIDGE_ROOT.parent / "EasyNCO"
GEN_MVRP_PATH = SCRIPT_DIR / "generate_mvrp_diverse_v4.py"


spec = importlib.util.spec_from_file_location("generate_mvrp_diverse_v4_module", GEN_MVRP_PATH)
mvrp_v4 = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(mvrp_v4)

V4_SPLIT_ROOT = DATASETS_ROOT / "offline_init_v4" / "nss_style_splits_v1"
ATSP_SPLIT_ROOT = V4_SPLIT_ROOT / "atsp"
MVRP_SPLIT_ROOT = V4_SPLIT_ROOT / "mvrp_diverse"

ATSP_VALTEST_TOTAL = 1000
ATSP_VALTEST_SPECS = build_dense_scale_specs(20, 100, ATSP_VALTEST_TOTAL)
MVRP_VALTEST_TOTAL = 1000
MVRP_VALTEST_SPECS = build_dense_scale_specs(50, 100, MVRP_VALTEST_TOTAL)

SPLIT_SEED_OFFSETS = {
    "val": 100_000,
    "test": 200_000,
}


def relative_to_datasets(path: Path) -> str:
    return str(path.relative_to(DATASETS_ROOT))


def seed_everything(seed: int) -> None:
    mvrp_v4.seed_everything(seed)


def sample_atsp_instance(problem_size: int, int_min: int = 0, int_max: int = 1_000_000, scaler: int = 1_000_000):
    problems = torch.randint(low=int_min, high=int_max, size=(problem_size, problem_size))
    problems[torch.arange(problem_size), torch.arange(problem_size)] = 0
    while True:
        old_problems = problems.clone()
        problems, _ = (problems[:, None, :] + problems[None, :, :].transpose(1, 2)).min(dim=2)
        if torch.equal(problems, old_problems):
            break
    return problems.float() / scaler


def generate_atsp_split(split: str, force: bool) -> dict:
    split_root = ATSP_SPLIT_ROOT / f"ATSP{split}"
    split_root.mkdir(parents=True, exist_ok=True)
    seed_offset = SPLIT_SEED_OFFSETS[split]

    manifest = {
        "dataset_name": f"ATSP{split}",
        "problem": "atsp",
        "source": "EasyNCO.data.ATSPGenerator",
        "seed": SEED + seed_offset,
        "min_scale": min(ATSP_VALTEST_SPECS),
        "max_scale": max(ATSP_VALTEST_SPECS),
        "unique_scales": len(ATSP_VALTEST_SPECS),
        "shards": [],
    }

    for scale, count in sorted(ATSP_VALTEST_SPECS.items()):
        seed_everything(SEED + seed_offset + scale)
        output_path = split_root / f"atsp{scale}_nums{count}.pt"
        if force or not output_path.exists():
            data = torch.empty((count, scale, scale), dtype=torch.float32)
            for idx in range(count):
                data[idx] = sample_atsp_instance(scale)
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
    manifest_path = split_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def generate_mvrp_split(split: str, force: bool, variants) -> dict:
    split_root = MVRP_SPLIT_ROOT / split
    split_root.mkdir(parents=True, exist_ok=True)
    seed_offset = SPLIT_SEED_OFFSETS[split]

    global_manifest = {
        "dataset_family": f"mvrp_diverse_{split}",
        "problem_family": "mvrp_variants",
        "source": "bridge_diverse_mvrp_generator_v1",
        "seed": SEED + seed_offset,
        "mode": MVRP_MODE,
        "distributions": list(mvrp_v4.DISTRIBUTIONS),
        "min_scale": min(MVRP_VALTEST_SPECS),
        "max_scale": max(MVRP_VALTEST_SPECS),
        "variants": {},
    }

    original_root = mvrp_v4.V4_MVRP_DIVERSE_ROOT
    original_specs = mvrp_v4.MVRP_SPECS
    original_seed = mvrp_v4.SEED
    try:
        mvrp_v4.V4_MVRP_DIVERSE_ROOT = split_root
        mvrp_v4.MVRP_SPECS = MVRP_VALTEST_SPECS
        mvrp_v4.SEED = SEED + seed_offset
        for variant in variants:
            print(f"[generate:{split}] {variant}", flush=True)
            global_manifest["variants"][variant] = mvrp_v4.generate_variant_dataset(variant, force=force)
    finally:
        mvrp_v4.V4_MVRP_DIVERSE_ROOT = original_root
        mvrp_v4.MVRP_SPECS = original_specs
        mvrp_v4.SEED = original_seed

    manifest_path = split_root / "manifest.json"
    manifest_path.write_text(json.dumps(global_manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return global_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate NSS-style val/test splits for ATSP and MVRP")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--splits", nargs="*", default=["val", "test"], choices=["val", "test"])
    parser.add_argument("--variants", nargs="*", default=list(MVRP_VARIANTS))
    args = parser.parse_args()

    summary = {"atsp": {}, "mvrp": {}}
    for split in args.splits:
        print(f"[ATSP {split}] start", flush=True)
        summary["atsp"][split] = generate_atsp_split(split, force=args.force)
        print(f"[MVRP {split}] start", flush=True)
        summary["mvrp"][split] = generate_mvrp_split(split, force=args.force, variants=args.variants)

    summary_path = V4_SPLIT_ROOT / "manifest.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote split datasets to {V4_SPLIT_ROOT}", flush=True)


if __name__ == "__main__":
    main()
