#!/usr/bin/env python
import argparse
import csv
import json
import pickle
import shutil
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import torch


SCRIPT_DIR = Path(__file__).resolve().parent
BRIDGE_ROOT = SCRIPT_DIR.parent
if str(BRIDGE_ROOT) not in sys.path:
    sys.path.insert(0, str(BRIDGE_ROOT))

from config import DATASETS_ROOT, EASYNCO_ROOT, RESULTS_ROOT


NSS_SOURCE_ROOT = EASYNCO_ROOT.parent / "9nss论文" / "neural-solver-selection" / "datasets"
V1_EXPORT_ROOT = BRIDGE_ROOT / "exports" / "nss_like_export_v1"
DEFAULT_OUTPUT_ROOT = BRIDGE_ROOT / "exports" / "nss_like_export_final_full"

TSP_CVRP_DATASETS = (
    "TSPtrain",
    "TSPval",
    "TSPtest",
    "TSPLIB",
    "CVRPtrain",
    "CVRPval",
    "CVRPtest",
    "CVRPLIB",
)

ATSP_TRAIN_MANIFEST = DATASETS_ROOT / "offline_init_v3" / "atsp" / "manifest.json"
ATSP_VAL_MANIFEST = DATASETS_ROOT / "offline_init_v4" / "nss_style_splits_v1" / "atsp" / "ATSPval" / "manifest.json"
ATSP_TEST_MANIFEST = DATASETS_ROOT / "offline_init_v4" / "nss_style_splits_v1" / "atsp" / "ATSPtest" / "manifest.json"

ATSP_METHODS = (
    ("glop", "GLOP"),
    ("matnet", "MATNET"),
    ("matpoenet", "MATPOENET"),
)

ATSP_RESULT_ROOTS = {
    "train": RESULTS_ROOT / "label_v3",
    "val": RESULTS_ROOT / "label_v4" / "atsp" / "val",
    "test": RESULTS_ROOT / "label_v4" / "atsp" / "test",
}

MVRP_TRAIN_MANIFEST = DATASETS_ROOT / "offline_init_v4" / "mvrp_diverse_v1" / "manifest.json"
MVRP_VAL_MANIFEST = (
    DATASETS_ROOT / "offline_init_v4" / "nss_style_splits_v1" / "mvrp_diverse" / "val" / "manifest.json"
)
MVRP_TEST_MANIFEST = (
    DATASETS_ROOT / "offline_init_v4" / "nss_style_splits_v1" / "mvrp_diverse" / "test" / "manifest.json"
)

MVRP_METHODS = (
    ("mtpomo", "MTPOMO"),
    ("mvmoe", "MVMOE"),
)


def load_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


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


def count_nonempty_lines(path: Path) -> int:
    if not path.is_file():
        return 0
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def ensure_clean_dir(path: Path, force: bool) -> None:
    if path.exists():
        if not force:
            raise FileExistsError(f"{path} already exists. Use --force to replace it.")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def copy_tree(src: Path, dst: Path) -> None:
    shutil.copytree(src, dst)


def copy_tsp_cvrp_complete(output_root: Path) -> List[Dict]:
    exports = []
    for dataset_name in TSP_CVRP_DATASETS:
        src_dir = V1_EXPORT_ROOT / dataset_name
        if not src_dir.is_dir():
            raise FileNotFoundError(f"Missing source dataset in v1 export: {src_dir}")
        dst_dir = output_root / dataset_name
        copy_tree(src_dir, dst_dir)

        # Validate against NSS source length.
        nss_dataset = NSS_SOURCE_ROOT / dataset_name / "dataset.pkl"
        if not nss_dataset.is_file():
            raise FileNotFoundError(f"Missing NSS source dataset: {nss_dataset}")
        expected = dataset_length(load_pickle(nss_dataset))
        got = dataset_length(load_pickle(dst_dir / "dataset.pkl"))
        if expected != got:
            raise ValueError(f"{dataset_name}: copied dataset has {got} instances, NSS source has {expected}")

        result_counts = {}
        results_dir = dst_dir / "results"
        if results_dir.is_dir():
            for result_file in sorted(results_dir.glob("result_*.txt")):
                line_count = count_nonempty_lines(result_file)
                if line_count != got:
                    raise ValueError(
                        f"{result_file} has {line_count} lines, expected {got} for {dataset_name}"
                    )
                result_counts[result_file.name] = line_count

        exports.append(
            {
                "dataset": dataset_name,
                "num_instances": got,
                "num_result_files": len(result_counts),
                "source": "copy_from_nss_like_export_v1_validated_against_nss_source",
                "result_counts": result_counts,
            }
        )
    return exports


def build_raw_label(method_names: List[str], all_scores: Dict[str, List[float]]) -> Dict[str, Dict]:
    total_instances = len(all_scores[method_names[0]]) if method_names else 0
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
    return raw_label


def export_atsp_split(
    split: str,
    dataset_name: str,
    manifest_path: Path,
    output_root: Path,
) -> Dict:
    payload = load_json(manifest_path)
    shards = payload["shards"]
    total_instances = int(payload["total_instances"])
    dataset_entries = []
    meta_rows = []
    global_offset = 0

    for shard in shards:
        shard_path = DATASETS_ROOT / shard["relative_path"]
        scale = int(shard["scale"])
        num_instances = int(shard["num_instances"])
        tensor = torch.load(shard_path, map_location="cpu")
        if tensor.shape[0] != num_instances:
            raise ValueError(f"{shard_path} expected {num_instances} instances, got {tensor.shape[0]}")
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

    if global_offset != total_instances:
        raise ValueError(f"{dataset_name}: merged {global_offset}, manifest says {total_instances}")

    dataset_dir = output_root / dataset_name
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

    for method_key, method_name in ATSP_METHODS:
        scores = [None] * total_instances
        global_offset = 0
        for shard in shards:
            scale = int(shard["scale"])
            num_instances = int(shard["num_instances"])
            if split == "train":
                result_path = ATSP_RESULT_ROOTS[split] / f"{method_key}_atsp" / f"scale_{scale}" / "instance_results.jsonl"
            else:
                result_path = ATSP_RESULT_ROOTS[split] / method_key / f"scale_{scale}" / "instance_results.jsonl"
            if not result_path.is_file():
                raise FileNotFoundError(f"Missing ATSP result file: {result_path}")

            with result_path.open("r", encoding="utf-8") as f:
                rows = [json.loads(line) for line in f if line.strip()]
            if len(rows) != num_instances:
                raise ValueError(f"{result_path} expected {num_instances} rows, got {len(rows)}")

            for local_index, row in enumerate(rows):
                row_index = int(row.get("global_index", local_index))
                if row_index != local_index:
                    raise ValueError(
                        f"{result_path} row {local_index} has global_index={row_index}, expected {local_index}"
                    )
                score = row.get("no_aug_score")
                if score is None:
                    raise ValueError(f"{result_path} row {local_index} missing no_aug_score")
                scores[global_offset + local_index] = float(score)
            global_offset += num_instances

        if any(score is None for score in scores):
            missing = sum(score is None for score in scores)
            raise ValueError(f"{dataset_name}/{method_name} missing {missing} scores after merge")

        result_path = results_dir / f"result_{method_name}.txt"
        with result_path.open("w", encoding="utf-8") as f:
            for index, score in enumerate(scores):
                f.write(f"{index},{score}\n")
        all_scores[method_name] = scores

    method_names = [method_name for _, method_name in ATSP_METHODS]
    dump_pickle(build_raw_label(method_names, all_scores), dataset_dir / "raw_label.pkl")

    return {
        "dataset": dataset_name,
        "split": split,
        "num_instances": total_instances,
        "methods": method_names,
        "result_files": [f"result_{name}.txt" for name in method_names],
    }


def write_meta_csv(rows: Iterable[Dict], path: Path, fieldnames: List[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_csv_rows(path: Path) -> List[Dict]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def export_mvrp_split(output_root: Path, split: str, manifest_path: Path) -> List[Dict]:
    payload = load_json(manifest_path)
    variant_exports = []

    for variant_name, variant_meta in sorted(payload["variants"].items()):
        dataset_name = f"{variant_name}{split}"
        variant_lower = variant_name.lower()
        shards = variant_meta["shards"]
        total_instances = int(variant_meta["total_instances"])
        dataset_entries = []
        global_offset = 0

        for shard in shards:
            shard_path = DATASETS_ROOT / shard["relative_path"]
            num_instances = int(shard["num_instances"])
            shard_entries = load_pickle(shard_path)
            if len(shard_entries) != num_instances:
                raise ValueError(f"{shard_path} expected {num_instances} instances, got {len(shard_entries)}")
            dataset_entries.extend(shard_entries)
            global_offset += num_instances

        if global_offset != total_instances:
            raise ValueError(f"{dataset_name}: merged {global_offset}, manifest says {total_instances}")

        dataset_dir = output_root / dataset_name
        dump_pickle(dataset_entries, dataset_dir / "dataset.pkl")

        source_meta = DATASETS_ROOT / shards[0]["relative_path"]
        source_variant_dir = source_meta.parent
        meta_candidates = [
            source_variant_dir / "instance_meta.csv",
            DATASETS_ROOT / Path(shards[0]["relative_path"]).parent / "instance_meta.csv",
        ]
        copied_meta = False
        for candidate in meta_candidates:
            if candidate.is_file():
                shutil.copy2(candidate, dataset_dir / "instance_meta.csv")
                copied_meta = True
                break
        if not copied_meta:
            meta_rows = []
            global_offset = 0
            for shard in shards:
                scale = int(shard["scale"])
                num_instances = int(shard["num_instances"])
                for local_index in range(num_instances):
                    meta_rows.append(
                        {
                            "instance_id": global_offset + local_index,
                            "scale": scale,
                            "scale_local_index": local_index,
                            "source_relative_path": shard["relative_path"],
                        }
                    )
                global_offset += num_instances
            write_meta_csv(
                meta_rows,
                dataset_dir / "instance_meta.csv",
                ["instance_id", "scale", "scale_local_index", "source_relative_path"],
            )

        results_dir = dataset_dir / "results"
        results_dir.mkdir(parents=True, exist_ok=True)
        all_scores = {}

        for method_key, method_name in MVRP_METHODS:
            scores = [None] * total_instances
            global_offset = 0
            for shard in shards:
                scale = int(shard["scale"])
                num_instances = int(shard["num_instances"])
                result_path = (
                    RESULTS_ROOT
                    / "label_v4"
                    / "mvrp"
                    / split
                    / f"{method_key}_{variant_lower}"
                    / f"scale_{scale}"
                    / "instance_results.jsonl"
                )
                if not result_path.is_file():
                    raise FileNotFoundError(f"Missing MVRP result file: {result_path}")

                with result_path.open("r", encoding="utf-8") as f:
                    rows = [json.loads(line) for line in f if line.strip()]
                if len(rows) != num_instances:
                    raise ValueError(f"{result_path} expected {num_instances} rows, got {len(rows)}")

                for local_index, row in enumerate(rows):
                    row_index = int(row.get("global_index", local_index))
                    if row_index != local_index:
                        raise ValueError(
                            f"{result_path} row {local_index} has global_index={row_index}, expected {local_index}"
                        )
                    score = row.get("no_aug_score")
                    if score is None:
                        raise ValueError(f"{result_path} row {local_index} missing no_aug_score")
                    scores[global_offset + local_index] = float(score)
                global_offset += num_instances

            if any(score is None for score in scores):
                missing = sum(score is None for score in scores)
                raise ValueError(f"{dataset_name}/{method_name} missing {missing} scores after merge")

            result_path = results_dir / f"result_{method_name}.txt"
            with result_path.open("w", encoding="utf-8") as f:
                for index, score in enumerate(scores):
                    f.write(f"{index},{score}\n")
            all_scores[method_name] = scores

        method_names = [method_name for _, method_name in MVRP_METHODS]
        dump_pickle(build_raw_label(method_names, all_scores), dataset_dir / "raw_label.pkl")

        variant_exports.append(
            {
                "dataset": dataset_name,
                "variant": variant_name,
                "split": split,
                "num_instances": total_instances,
                "methods": method_names,
                "result_files": [f"result_{name}.txt" for name in method_names],
            }
        )

    return variant_exports


def validate_final_export(output_root: Path, export_manifest: Dict) -> None:
    # Validate every dataset in the manifest by dataset length and result-line counts.
    dataset_specs = []
    dataset_specs.extend(item["dataset"] for item in export_manifest["tsp_cvrp_exports"])
    dataset_specs.extend(item["dataset"] for item in export_manifest["atsp_exports"])
    dataset_specs.extend(item["dataset"] for item in export_manifest["mvrp_exports"])

    for dataset_name in dataset_specs:
        dataset_path = output_root / dataset_name / "dataset.pkl"
        if not dataset_path.is_file():
            raise FileNotFoundError(f"Missing exported dataset.pkl: {dataset_path}")
        num_instances = dataset_length(load_pickle(dataset_path))
        results_dir = output_root / dataset_name / "results"
        if results_dir.is_dir():
            for result_file in results_dir.glob("result_*.txt"):
                line_count = count_nonempty_lines(result_file)
                if line_count != num_instances:
                    raise ValueError(
                        f"{result_file} has {line_count} lines, expected {num_instances} for {dataset_name}"
                    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Export one final NSS-like bundle with TSP/CVRP/ATSP/MVRP full data.")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Directory to write the final unified NSS-like bundle",
    )
    parser.add_argument("--force", action="store_true", help="Replace the output directory if it exists")
    args = parser.parse_args()

    output_root = args.output_root.resolve()
    ensure_clean_dir(output_root, force=args.force)

    tsp_cvrp_exports = copy_tsp_cvrp_complete(output_root)

    atsp_exports = []
    atsp_exports.append(export_atsp_split("train", "ATSPtrain", ATSP_TRAIN_MANIFEST, output_root))
    atsp_exports.append(export_atsp_split("val", "ATSPval", ATSP_VAL_MANIFEST, output_root))
    atsp_exports.append(export_atsp_split("test", "ATSPtest", ATSP_TEST_MANIFEST, output_root))

    mvrp_exports = []
    mvrp_exports.extend(export_mvrp_split(output_root, "train", MVRP_TRAIN_MANIFEST))
    mvrp_exports.extend(export_mvrp_split(output_root, "val", MVRP_VAL_MANIFEST))
    mvrp_exports.extend(export_mvrp_split(output_root, "test", MVRP_TEST_MANIFEST))

    export_manifest = {
        "output_root": str(output_root),
        "tsp_cvrp_exports": tsp_cvrp_exports,
        "atsp_exports": atsp_exports,
        "mvrp_exports": mvrp_exports,
    }
    (output_root / "export_manifest.json").write_text(
        json.dumps(export_manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    validate_final_export(output_root, export_manifest)

    print(f"Final export completed: {output_root}")
    print(f"TSP/CVRP datasets: {len(tsp_cvrp_exports)}")
    print(f"ATSP datasets: {len(atsp_exports)}")
    print(f"MVRP datasets: {len(mvrp_exports)}")


if __name__ == "__main__":
    main()
