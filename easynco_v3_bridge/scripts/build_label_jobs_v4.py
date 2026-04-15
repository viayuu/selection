#!/usr/bin/env python
import json
import shlex
import sys
from pathlib import Path
from typing import Dict, List

SCRIPT_DIR = Path(__file__).resolve().parent
BRIDGE_ROOT = SCRIPT_DIR.parent
if str(BRIDGE_ROOT) not in sys.path:
    sys.path.insert(0, str(BRIDGE_ROOT))

from config import (
    ATSP_METHODS,
    ATSP_MODEL_OVERRIDES,
    COMMAND_ROOT,
    EASYNCO_ROOT,
    LOG_ROOT,
    MANIFEST_ROOT,
    MVRP_METHODS,
    MVRP_MODEL_OVERRIDES,
    MVRP_MODE,
    MVRP_VARIANT_GROUPS,
    MVRP_VARIANTS,
    RESULTS_ROOT,
    V4_ATSP_SPLIT_ROOT,
    V4_MVRP_DIVERSE_ROOT,
    V4_MVRP_SPLIT_ROOT,
    ensure_layout,
    variant_dir_name,
)


def load_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_shell_script(path: Path, commands: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = ["#!/usr/bin/env bash", "set -euo pipefail", ""] + commands + [""]
    path.write_text("\n".join(content), encoding="utf-8")
    path.chmod(0o755)


def compatible_batch_size(preferred: int, episodes: int) -> int:
    upper = max(1, min(preferred, episodes))
    for candidate in range(upper, 0, -1):
        if episodes % candidate == 0:
            return candidate
    return 1


def build_atsp_jobs() -> List[Dict]:
    jobs = []
    for split in ("val", "test"):
        manifest = load_json(V4_ATSP_SPLIT_ROOT / f"ATSP{split}" / "manifest.json")
        for method in ATSP_METHODS:
            spec = ATSP_MODEL_OVERRIDES[method]
            for shard in manifest["shards"]:
                scale = int(shard["scale"])
                count = int(shard["num_instances"])
                rel_path = shard["relative_path"]
                batch_size = compatible_batch_size(spec["batch_size"], count)
                result_dir = RESULTS_ROOT / "label_v4" / "atsp" / split / method / f"scale_{scale}"
                jobs.append(
                    {
                        "name": f"{method}_atsp_{split}_scale_{scale}",
                        "group": f"atsp_{split}",
                        "family": "atsp",
                        "split": split,
                        "cwd": str(EASYNCO_ROOT),
                        "log_path": str(LOG_ROOT / f"{method}_atsp_{split}_scale_{scale}.log"),
                        "expected_output": str(result_dir / "instance_results.jsonl"),
                        "expected_lines": count,
                        "command": [
                            "python",
                            "eval.py",
                            f"settings={spec['settings']}",
                            "mode=test",
                            f"model={method}",
                            "problem=atsp",
                            f"scale={scale}",
                            f"batch_size={batch_size}",
                            f"episodes={count}",
                            "decoder_strategy=greedy",
                            "cuda=[0]",
                            f"test_data_path={rel_path}",
                            f"settings.test_loader.model_dirpath={spec['model_dirpath']}",
                            f"settings.test_loader.model_filename={spec['model_filename']}",
                            f"dir=results/label_v4/atsp/{split}/{method}/scale_{scale}",
                        ]
                        + (
                            [
                                f"settings.env.pomo_size={scale}",
                                f"settings.module.initialization_params.pomo_size={scale}",
                            ]
                            if method == "matnet"
                            else []
                        )
                        + (
                            ["settings.iteration._target_=EasyNCO.neural_solvers.pipeline.NoIteration"]
                            if method == "glop"
                            else []
                        ),
                    }
                )
    return jobs


def build_mvrp_jobs_from_manifest(root_manifest: Path, split: str) -> List[Dict]:
    payload = load_json(root_manifest)
    variant_payload = payload["variants"]
    jobs = []
    for variant in MVRP_VARIANTS:
        variant_lower = variant_dir_name(variant)
        manifest = variant_payload[variant]
        for method in MVRP_METHODS:
            spec = MVRP_MODEL_OVERRIDES[method]
            for shard in manifest["shards"]:
                scale = int(shard["scale"])
                count = int(shard["num_instances"])
                rel_path = shard["relative_path"]
                result_dir = RESULTS_ROOT / "label_v4" / "mvrp" / split / f"{method}_{variant_lower}" / f"scale_{scale}"
                jobs.append(
                    {
                        "name": f"{method}_{variant_lower}_{split}_scale_{scale}",
                        "group": f"mvrp_{split}",
                        "family": "mvrp",
                        "split": split,
                        "variant": variant,
                        "cwd": str(EASYNCO_ROOT),
                        "log_path": str(LOG_ROOT / f"{method}_{variant_lower}_{split}_scale_{scale}.log"),
                        "expected_output": str(result_dir / "instance_results.jsonl"),
                        "expected_lines": count,
                        "command": [
                            "python",
                            "eval.py",
                            f"settings={spec['settings']}",
                            "mode=test",
                            f"model={method}",
                            f"problem={variant}",
                            f"scale={scale}",
                            f"batch_size={spec['batch_size']}",
                            f"episodes={count}",
                            "decoder_strategy=greedy",
                            "cuda=[0]",
                            f"test_data_path={rel_path}",
                            f"settings.test_loader.model_dirpath={spec['model_dirpath']}",
                            f"settings.test_loader.model_filename={spec['model_filename']}",
                            f"settings.env.mode={MVRP_MODE}",
                            f"++settings.module.test_data_params.mode={MVRP_MODE}",
                            f"dir=results/label_v4/mvrp/{split}/{method}_{variant_lower}/scale_{scale}",
                        ],
                    }
                )
    return jobs


def build_mvrp_train_jobs() -> List[Dict]:
    return build_mvrp_jobs_from_manifest(V4_MVRP_DIVERSE_ROOT / "manifest.json", "train")


def build_mvrp_valtest_jobs() -> List[Dict]:
    jobs = []
    for split in ("val", "test"):
        jobs.extend(build_mvrp_jobs_from_manifest(V4_MVRP_SPLIT_ROOT / split / "manifest.json", split))
    return jobs


def write_command_collections(jobs: List[Dict]) -> None:
    grouped: Dict[str, List[str]] = {}
    for job in jobs:
        line = f"(cd {shlex.quote(job['cwd'])} && {shlex.join(job['command'])})"
        grouped.setdefault(job["group"], []).append(line)
    for group, commands in grouped.items():
        write_shell_script(COMMAND_ROOT / f"run_{group}_labels_v4.sh", commands)
    all_commands = [line for commands in grouped.values() for line in commands]
    write_shell_script(COMMAND_ROOT / "run_all_labels_v4.sh", all_commands)


def write_parallel_manifests(jobs: List[Dict]) -> None:
    atsp_jobs = [job for job in jobs if job["family"] == "atsp"]
    for method in ATSP_METHODS:
        subset = [job for job in atsp_jobs if job["name"].startswith(f"{method}_atsp_")]
        path = MANIFEST_ROOT / f"label_jobs_v4_atsp_{method}.json"
        path.write_text(json.dumps(subset, ensure_ascii=False, indent=2), encoding="utf-8")

    mvrp_jobs = [job for job in jobs if job["family"] == "mvrp"]
    for method in MVRP_METHODS:
        for group_name, variants in MVRP_VARIANT_GROUPS:
            tokens = {f"_{variant_dir_name(variant)}_" for variant in variants}
            subset = []
            for job in mvrp_jobs:
                if not job["name"].startswith(f"{method}_"):
                    continue
                if any(token in job["name"] for token in tokens):
                    subset.append(job)
            path = MANIFEST_ROOT / f"label_jobs_v4_mvrp_{method}_{group_name}.json"
            path.write_text(json.dumps(subset, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    ensure_layout()
    jobs = build_mvrp_train_jobs() + build_atsp_jobs() + build_mvrp_valtest_jobs()
    jobs_path = MANIFEST_ROOT / "label_jobs_v4.json"
    jobs_path.write_text(json.dumps(jobs, ensure_ascii=False, indent=2), encoding="utf-8")
    write_command_collections(jobs)
    write_parallel_manifests(jobs)
    print(f"Wrote {len(jobs)} v4 jobs to {jobs_path}")


if __name__ == "__main__":
    main()
