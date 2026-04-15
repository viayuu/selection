#!/usr/bin/env python
import argparse
import csv
import json
import pickle
import shutil
import sys
from pathlib import Path

import torch


SCRIPT_DIR = Path(__file__).resolve().parent
BRIDGE_ROOT = SCRIPT_DIR.parent
if str(BRIDGE_ROOT) not in sys.path:
    sys.path.insert(0, str(BRIDGE_ROOT))

from config import DATASETS_ROOT, EASYNCO_ROOT, RESULTS_ROOT


NSS_DATASETS_ROOT = EASYNCO_ROOT.parent / "9nss论文" / "neural-solver-selection" / "datasets"
DEFAULT_OUTPUT_ROOT = BRIDGE_ROOT / "exports" / "nss_like_export_v1"

ATSP_MANIFEST = DATASETS_ROOT / "offline_init_v3" / "atsp" / "manifest.json"
ATSP_METHODS = (
    ("glop", "GLOP"),
    ("matnet", "MATNET"),
    ("matpoenet", "MATPOENET"),
)
MVRP_MANIFEST = DATASETS_ROOT / "offline_init_v3" / "mvrp" / "manifest.json"
MVRP_METHODS = (
    ("mtpomo", "MTPOMO"),
    ("mvmoe", "MVMOE"),
)


def load_pickle(path: Path):
    with path.open("rb") as f:
        return pickle.load(f)


def dump_pickle(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def dataset_length(dataset_obj) -> int:
    if hasattr(dataset_obj, "__len__"):
        return len(dataset_obj)
    raise TypeError(f"Dataset object of type {type(dataset_obj)} has no length")


def rewrite_result_file(src: Path, dst: Path) -> int:
    dst.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with src.open("r", encoding="utf-8") as fin, dst.open("w", encoding="utf-8") as fout:
        for raw in fin:
            line = raw.strip()
            if not line:
                continue
            parts = [part.strip() for part in line.split(",")]
            if len(parts) < 2:
                raise ValueError(f"Unexpected result line in {src}: {raw!r}")
            fout.write(f"{parts[0]},{parts[1]}\n")
            count += 1
    return count


def export_nss_like_sources(output_root: Path):
    exported = []
    for src_dir in sorted(NSS_DATASETS_ROOT.iterdir()):
        if not src_dir.is_dir():
            continue
        if not (src_dir.name.startswith("TSP") or src_dir.name.startswith("CVRP")):
            continue
        dataset_path = src_dir / "dataset.pkl"
        results_dir = src_dir / "results"
        if not dataset_path.is_file() or not results_dir.is_dir():
            continue

        dst_dir = output_root / src_dir.name
        dst_results = dst_dir / "results"
        dst_results.mkdir(parents=True, exist_ok=True)

        dataset_obj = load_pickle(dataset_path)
        expected = dataset_length(dataset_obj)
        shutil.copy2(dataset_path, dst_dir / "dataset.pkl")

        raw_label_path = src_dir / "raw_label.pkl"
        if raw_label_path.is_file():
            shutil.copy2(raw_label_path, dst_dir / "raw_label.pkl")

        result_counts = {}
        for src_result in sorted(results_dir.glob("result_*.txt")):
            dst_result = dst_results / src_result.name
            count = rewrite_result_file(src_result, dst_result)
            if count != expected:
                raise ValueError(
                    f"{src_result} has {count} lines but dataset {src_dir.name} has {expected} instances"
                )
            result_counts[src_result.name] = count

        exported.append(
            {
                "dataset": src_dir.name,
                "num_instances": expected,
                "num_result_files": len(result_counts),
                "result_counts": result_counts,
            }
        )
    return exported


def load_atsp_manifest():
    with ATSP_MANIFEST.open("r", encoding="utf-8") as f:
        return json.load(f)


def export_atsp_dataset(output_root: Path):
    payload = load_atsp_manifest()
    shards = payload["shards"]
    dataset_entries = []
    meta_rows = []
    global_offset = 0

    for shard in shards:
        shard_path = DATASETS_ROOT / shard["relative_path"]
        scale = int(shard["scale"])
        num_instances = int(shard["num_instances"])
        tensor = torch.load(shard_path, map_location="cpu")
        if tensor.shape[0] != num_instances:
            raise ValueError(
                f"{shard_path} expected {num_instances} instances, got {tensor.shape[0]}"
            )
        for local_index in range(num_instances):
            dataset_entries.append(tensor[local_index : local_index + 1].clone())
            meta_rows.append(
                {
                    "instance_id": global_offset + local_index,
                    "scale": scale,
                    "scale_local_index": local_index,
                    "source_relative_path": shard["relative_path"],
                }
            )
        global_offset += num_instances

    if global_offset != int(payload["total_instances"]):
        raise ValueError(
            f"ATSP total mismatch: manifest says {payload['total_instances']}, merged {global_offset}"
        )

    atsp_dir = output_root / "ATSPtrain"
    dump_pickle(dataset_entries, atsp_dir / "dataset.pkl")

    meta_path = atsp_dir / "instance_meta.csv"
    with meta_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["instance_id", "scale", "scale_local_index", "source_relative_path"],
        )
        writer.writeheader()
        writer.writerows(meta_rows)

    return atsp_dir, meta_rows


def export_atsp_results(output_root: Path, meta_rows):
    atsp_dir = output_root / "ATSPtrain"
    results_dir = atsp_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    payload = load_atsp_manifest()
    total_instances = int(payload["total_instances"])
    shards = payload["shards"]
    all_scores = {}

    for method_key, method_name in ATSP_METHODS:
        scores = [None] * total_instances
        global_offset = 0
        for shard in shards:
            scale = int(shard["scale"])
            num_instances = int(shard["num_instances"])
            result_path = RESULTS_ROOT / "label_v3" / f"{method_key}_atsp" / f"scale_{scale}" / "instance_results.jsonl"
            if not result_path.is_file():
                raise FileNotFoundError(f"Missing ATSP result file: {result_path}")

            with result_path.open("r", encoding="utf-8") as f:
                rows = [json.loads(line) for line in f if line.strip()]

            if len(rows) != num_instances:
                raise ValueError(
                    f"{result_path} expected {num_instances} rows, got {len(rows)}"
                )

            for local_index, row in enumerate(rows):
                row_index = int(row.get("global_index", local_index))
                if row_index != local_index:
                    raise ValueError(
                        f"{result_path} row {local_index} has global_index={row_index}, expected local order"
                    )
                score = row.get("no_aug_score")
                if score is None:
                    raise ValueError(f"{result_path} row {local_index} missing no_aug_score")
                scores[global_offset + local_index] = float(score)

            global_offset += num_instances

        if any(score is None for score in scores):
            missing = sum(score is None for score in scores)
            raise ValueError(f"{method_key} missing {missing} ATSP scores after merge")

        result_path = results_dir / f"result_{method_name}.txt"
        with result_path.open("w", encoding="utf-8") as f:
            for index, score in enumerate(scores):
                f.write(f"{index},{score}\n")
        all_scores[method_name] = scores

    raw_label = {}
    method_names = [method_name for _, method_name in ATSP_METHODS]
    for index in range(total_instances):
        costs = [all_scores[method_name][index] for method_name in method_names]
        best_ind = min(range(len(costs)), key=lambda idx: costs[idx])
        raw_label[str(index)] = {
            "cost": costs,
            "time": [0.0] * len(costs),
            "ind": best_ind,
            "gap": [0.0] * len(costs),
        }
    dump_pickle(raw_label, atsp_dir / "raw_label.pkl")

    return {
        "dataset": "ATSPtrain",
        "num_instances": total_instances,
        "methods": method_names,
        "result_files": [f"result_{method_name}.txt" for method_name in method_names],
    }


def load_mvrp_manifest():
    with MVRP_MANIFEST.open("r", encoding="utf-8") as f:
        return json.load(f)


def export_mvrp_variants(output_root: Path):
    payload = load_mvrp_manifest()
    variant_exports = []

    for variant_name, variant_meta in sorted(payload["variants"].items()):
        shards = variant_meta["shards"]
        total_instances = int(variant_meta["total_instances"])
        dataset_entries = []
        meta_rows = []
        global_offset = 0

        for shard in shards:
            shard_path = DATASETS_ROOT / shard["relative_path"]
            scale = int(shard["scale"])
            num_instances = int(shard["num_instances"])
            shard_entries = load_pickle(shard_path)
            if len(shard_entries) != num_instances:
                raise ValueError(
                    f"{shard_path} expected {num_instances} instances, got {len(shard_entries)}"
                )
            for local_index, item in enumerate(shard_entries):
                dataset_entries.append(item)
                meta_rows.append(
                    {
                        "instance_id": global_offset + local_index,
                        "scale": scale,
                        "scale_local_index": local_index,
                        "source_relative_path": shard["relative_path"],
                    }
                )
            global_offset += num_instances

        if global_offset != total_instances:
            raise ValueError(
                f"{variant_name} total mismatch: manifest says {total_instances}, merged {global_offset}"
            )

        dataset_dir = output_root / f"{variant_name}train"
        dump_pickle(dataset_entries, dataset_dir / "dataset.pkl")

        meta_path = dataset_dir / "instance_meta.csv"
        with meta_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["instance_id", "scale", "scale_local_index", "source_relative_path"],
            )
            writer.writeheader()
            writer.writerows(meta_rows)

        results_dir = dataset_dir / "results"
        results_dir.mkdir(parents=True, exist_ok=True)
        all_scores = {}
        variant_lower = variant_name.lower()

        for method_key, method_name in MVRP_METHODS:
            scores = [None] * total_instances
            global_offset = 0
            for shard in shards:
                scale = int(shard["scale"])
                num_instances = int(shard["num_instances"])
                result_path = (
                    RESULTS_ROOT
                    / "label_v3"
                    / f"{method_key}_{variant_lower}"
                    / f"scale_{scale}"
                    / "instance_results.jsonl"
                )
                if not result_path.is_file():
                    raise FileNotFoundError(f"Missing MVRP result file: {result_path}")

                with result_path.open("r", encoding="utf-8") as f:
                    rows = [json.loads(line) for line in f if line.strip()]

                if len(rows) != num_instances:
                    raise ValueError(
                        f"{result_path} expected {num_instances} rows, got {len(rows)}"
                    )

                for local_index, row in enumerate(rows):
                    row_index = int(row.get("global_index", local_index))
                    if row_index != local_index:
                        raise ValueError(
                            f"{result_path} row {local_index} has global_index={row_index}, expected local order"
                        )
                    score = row.get("no_aug_score")
                    if score is None:
                        raise ValueError(f"{result_path} row {local_index} missing no_aug_score")
                    scores[global_offset + local_index] = float(score)

                global_offset += num_instances

            if any(score is None for score in scores):
                missing = sum(score is None for score in scores)
                raise ValueError(f"{variant_name}/{method_key} missing {missing} scores after merge")

            result_path = results_dir / f"result_{method_name}.txt"
            with result_path.open("w", encoding="utf-8") as f:
                for index, score in enumerate(scores):
                    f.write(f"{index},{score}\n")
            all_scores[method_name] = scores

        method_names = [method_name for _, method_name in MVRP_METHODS]
        raw_label = {}
        for index in range(total_instances):
            costs = [all_scores[method_name][index] for method_name in method_names]
            best_ind = min(range(len(costs)), key=lambda idx: costs[idx])
            raw_label[str(index)] = {
                "cost": costs,
                "time": [0.0] * len(costs),
                "ind": best_ind,
                "gap": [0.0] * len(costs),
            }
        dump_pickle(raw_label, dataset_dir / "raw_label.pkl")

        variant_exports.append(
            {
                "dataset": f"{variant_name}train",
                "num_instances": total_instances,
                "methods": method_names,
                "result_files": [f"result_{method_name}.txt" for method_name in method_names],
            }
        )

    return variant_exports


def write_export_manifest(output_root: Path, nss_exports, atsp_export, mvrp_exports):
    payload = {
        "output_root": str(output_root),
        "nss_exports": nss_exports,
        "atsp_export": atsp_export,
        "mvrp_exports": mvrp_exports,
    }
    path = output_root / "export_manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Export NSS-like bundles for TSP/CVRP/ATSP")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Directory to write the exported NSS-like bundle",
    )
    args = parser.parse_args()

    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    nss_exports = export_nss_like_sources(output_root)
    _, atsp_meta = export_atsp_dataset(output_root)
    atsp_export = export_atsp_results(output_root, atsp_meta)
    mvrp_exports = export_mvrp_variants(output_root)
    write_export_manifest(output_root, nss_exports, atsp_export, mvrp_exports)

    print(f"Export completed: {output_root}")
    print(f"NSS-derived datasets: {len(nss_exports)}")
    print(f"ATSP instances: {atsp_export['num_instances']}")
    print(f"MVRP variants: {len(mvrp_exports)}")


if __name__ == "__main__":
    main()
