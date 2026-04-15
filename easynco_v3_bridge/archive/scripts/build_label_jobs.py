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
    ATSP_SPECS,
    COMMAND_ROOT,
    EASYNCO_ROOT,
    LOG_ROOT,
    MANIFEST_ROOT,
    MVRP_METHODS,
    MVRP_MODEL_OVERRIDES,
    MVRP_MODE,
    MVRP_SPECS,
    MVRP_VARIANT_GROUPS,
    MVRP_VARIANTS,
    NSS_COPY_MANIFEST,
    NSS_COPY_DATASET_NAMES,
    NSS_CVRP_METHODS,
    NSS_TSP_METHODS,
    RESULTS_ROOT,
    ensure_layout,
    variant_dir_name,
)


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


def build_nss_jobs() -> List[Dict]:
    jobs = []
    jobs.append(
        {
            "name": "nss_tsptrain_labels",
            "group": "nss",
            "cwd": str(EASYNCO_ROOT),
            "log_path": str(LOG_ROOT / "nss_tsptrain_labels.log"),
            "command": [
                "python",
                "run_nss_eval_suite.py",
                "--cuda",
                "[0]",
                "--manifest",
                str(NSS_COPY_MANIFEST),
                "--datasets",
                NSS_COPY_DATASET_NAMES["TSPtrain"],
                "--methods",
                ",".join(NSS_TSP_METHODS),
            ],
        }
    )
    jobs.append(
        {
            "name": "nss_cvrptrain_labels",
            "group": "nss",
            "cwd": str(EASYNCO_ROOT),
            "log_path": str(LOG_ROOT / "nss_cvrptrain_labels.log"),
            "command": [
                "python",
                "run_nss_eval_suite.py",
                "--cuda",
                "[0]",
                "--manifest",
                str(NSS_COPY_MANIFEST),
                "--datasets",
                NSS_COPY_DATASET_NAMES["CVRPtrain"],
                "--methods",
                ",".join(NSS_CVRP_METHODS),
            ],
        }
    )
    return jobs


def build_atsp_jobs() -> List[Dict]:
    jobs = []
    for method in ATSP_METHODS:
        for scale, count in sorted(ATSP_SPECS.items()):
            spec = ATSP_MODEL_OVERRIDES[method]
            batch_size = compatible_batch_size(spec["batch_size"], count)
            result_dir = RESULTS_ROOT / "label_v3" / f"{method}_atsp" / f"scale_{scale}"
            jobs.append(
                {
                    "name": f"{method}_atsp_scale_{scale}",
                    "group": "atsp",
                    "cwd": str(EASYNCO_ROOT),
                    "log_path": str(LOG_ROOT / f"{method}_atsp_scale_{scale}.log"),
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
                        f"test_data_path=offline_init_v3/atsp/atsp{scale}_nums{count}.pt",
                        f"settings.test_loader.model_dirpath={spec['model_dirpath']}",
                        f"settings.test_loader.model_filename={spec['model_filename']}",
                        f"dir=results/label_v3/{method}_atsp/scale_{scale}",
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


def build_mvrp_jobs() -> List[Dict]:
    jobs = []
    for variant in MVRP_VARIANTS:
        variant_lower = variant_dir_name(variant)
        for method in MVRP_METHODS:
            for scale, count in sorted(MVRP_SPECS.items()):
                spec = MVRP_MODEL_OVERRIDES[method]
                result_dir = RESULTS_ROOT / "label_v3" / f"{method}_{variant_lower}" / f"scale_{scale}"
                jobs.append(
                    {
                        "name": f"{method}_{variant_lower}_scale_{scale}",
                        "group": "mvrp",
                        "cwd": str(EASYNCO_ROOT),
                        "log_path": str(LOG_ROOT / f"{method}_{variant_lower}_scale_{scale}.log"),
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
                            f"test_data_path=offline_init_v3/mvrp/{variant_lower}/{variant_lower}{scale}_nums{count}.pkl",
                            f"settings.test_loader.model_dirpath={spec['model_dirpath']}",
                            f"settings.test_loader.model_filename={spec['model_filename']}",
                            f"settings.env.mode={MVRP_MODE}",
                            f"++settings.module.test_data_params.mode={MVRP_MODE}",
                            f"dir=results/label_v3/{method}_{variant_lower}/scale_{scale}",
                        ],
                    }
                )
    return jobs


def write_command_collections(jobs: List[Dict]) -> None:
    nss_lines = []
    atsp_lines = []
    mvrp_lines = []
    for job in jobs:
        line = f"(cd {shlex.quote(job['cwd'])} && {shlex.join(job['command'])})"
        if job["group"] == "nss":
            nss_lines.append(line)
        elif job["group"] == "atsp":
            atsp_lines.append(line)
        elif job["group"] == "mvrp":
            mvrp_lines.append(line)

    write_shell_script(COMMAND_ROOT / "run_nss_labels.sh", nss_lines)
    write_shell_script(COMMAND_ROOT / "run_atsp_labels.sh", atsp_lines)
    write_shell_script(COMMAND_ROOT / "run_mvrp_labels.sh", mvrp_lines)
    write_shell_script(
        COMMAND_ROOT / "run_all_labels.sh",
        nss_lines + atsp_lines + mvrp_lines,
    )


def write_parallel_mvrp_manifests(jobs: List[Dict]) -> None:
    mvrp_jobs = [job for job in jobs if job["group"] == "mvrp"]
    for method in MVRP_METHODS:
        for group_name, variants in MVRP_VARIANT_GROUPS:
            variant_tokens = {f"_{variant_dir_name(variant)}_" for variant in variants}
            subset = []
            for job in mvrp_jobs:
                if not job["name"].startswith(f"{method}_"):
                    continue
                if any(token in job["name"] for token in variant_tokens):
                    subset.append(job)
            subset_path = MANIFEST_ROOT / f"mvrp_{method}_{group_name}.json"
            with subset_path.open("w", encoding="utf-8") as f:
                json.dump(subset, f, ensure_ascii=False, indent=2)


def main() -> None:
    ensure_layout()
    jobs = build_nss_jobs() + build_atsp_jobs() + build_mvrp_jobs()
    jobs_path = MANIFEST_ROOT / "label_jobs.json"
    with jobs_path.open("w", encoding="utf-8") as f:
        json.dump(jobs, f, ensure_ascii=False, indent=2)
    write_command_collections(jobs)
    write_parallel_mvrp_manifests(jobs)
    print(f"Wrote {len(jobs)} jobs to {jobs_path}")


if __name__ == "__main__":
    main()
