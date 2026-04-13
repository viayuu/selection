#!/usr/bin/env python3
"""
Benchmark EasyNCO solvers on TSP datasets by *calling* EasyNCO's CLI (no internal API calls).

Runs:
  python EasyNCO/eval.py settings=... mode=test model=... problem=tsp scale=... test_data_path=...

Outputs:
  - Per-run log files under `--out_dir/logs/...`
  - A tab-separated summary file `--out_dir/summary.tsv`

Notes:
  - This script does NOT modify any existing EasyNCO code.
  - This script enforces **initialization-only** evaluation by overriding:
      settings.iteration._target_=benchmarks.easynco_init_only_iteration.InitOnlyIteration
  - DACT checkpoints are only available for TSP50/TSP100 in this repo; other scales are skipped.
  - HTSP's lower-level group_size is hard-coded to 200 in EasyNCO; scales < 200 are skipped.
"""

from __future__ import annotations

import argparse
import csv
import os
import pickle
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]
EASYNCO_EVAL = REPO_ROOT / "EasyNCO" / "eval.py"
EASYNCO_DATASETS = REPO_ROOT / "EasyNCO" / "data" / "datasets"
EASYNCO_PRETRAINED = REPO_ROOT / "EasyNCO" / "pretrained"


@dataclass(frozen=True)
class DatasetSpec:
    key: str  # path relative to EasyNCO/data/datasets
    scale: int
    episodes: int


@dataclass(frozen=True)
class MethodSpec:
    name: str
    settings: str
    ckpt_filename: str
    # Pre-run validation hook: return (ok, reason_if_not_ok)
    def is_compatible(self, scale: int) -> Tuple[bool, str]:
        if self.name == "dact":
            if scale not in (50, 100):
                return False, "no checkpoint for this scale (only 50/100 here)"
        if self.name == "htsp":
            if scale < 200:
                return False, "HTSP lower group_size=200 (scale<200 will likely fail)"
        return True, ""


METHODS: Dict[str, MethodSpec] = {
    "omni": MethodSpec("omni", "omni_settings", "omni_tsp_maml_fomaml.ckpt"),
    "lehd": MethodSpec("lehd", "lehd_settings", "lehd_tsp100.ckpt"),
    "elg": MethodSpec("elg", "elg_settings", "elg_tsp100.ckpt"),
    "icam": MethodSpec("icam", "icam_settings", "icam_tsp_best.ckpt"),
    "dact": MethodSpec("dact", "dact_settings", "dact_tsp100.ckpt"),
    # Only htsp_tsp1000.ckpt is provided in this repo; use it for all scales >=200.
    "htsp": MethodSpec("htsp", "htsp_settings", "htsp_tsp1000.ckpt"),
}


_RE_TEST_TSP_UNIFORM = re.compile(r"^test_tsp(?P<scale>\d+)_nums(?P<num>\d+)_uniform\.pt$")
_RE_CROSS_DIST_TSP = re.compile(r"^tsp(?P<scale>\d+)_(?P<tag>[a-zA-Z0-9_-]+)\.pkl$")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--datasets",
        default="cross_distribution,test_dataset_tsp_uniform",
        help="Comma-separated dataset subfolders under EasyNCO/data/datasets",
    )
    p.add_argument(
        "--methods",
        default="omni,lehd,elg,icam,dact,htsp",
        help="Comma-separated methods to run",
    )
    p.add_argument("--cuda", default="0", help="CUDA device id(s), e.g. '0' or '0,1'")
    p.add_argument("--decoder_strategy", default="greedy", choices=["greedy", "sampling"])
    p.add_argument(
        "--aug_factor",
        type=int,
        default=1,
        help="Override settings.env.aug_factor for all methods (set to 1 to disable aug).",
    )
    p.add_argument(
        "--batch_size",
        type=int,
        default=0,
        help="If 0, choose a conservative batch_size based on scale.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="EasyNCO seed (Hydra cfg.seed).",
    )
    p.add_argument(
        "--out_dir",
        default=str(REPO_ROOT / "results" / "benchmarks" / "tsp_datasets"),
        help="Output directory for logs and summary.tsv",
    )
    p.add_argument("--dry_run", action="store_true")
    p.add_argument("--resume", action="store_true", help="Skip runs already in summary.tsv")
    p.add_argument("--continue_on_error", action="store_true")
    return p.parse_args()


def _cuda_override(cuda: str) -> str:
    cuda = cuda.strip()
    if not cuda:
        return "cuda=[0]"
    if "," in cuda:
        ids = ",".join(x.strip() for x in cuda.split(",") if x.strip())
        return f"cuda=[{ids}]"
    return f"cuda=[{cuda}]"


def _pick_batch_size(scale: int, episodes: int, user_batch_size: int) -> int:
    if user_batch_size and user_batch_size > 0:
        return min(user_batch_size, episodes)
    # Conservative heuristics (avoid OOM on large N)
    if scale <= 200:
        return min(128, episodes)
    if scale <= 1000:
        return min(32, episodes)
    if scale <= 5000:
        return min(8, episodes)
    return 1


def _discover_tsp_datasets(dataset_folders: Sequence[str]) -> List[DatasetSpec]:
    specs: List[DatasetSpec] = []
    for folder in dataset_folders:
        folder = folder.strip()
        if not folder:
            continue
        abs_dir = EASYNCO_DATASETS / folder
        if not abs_dir.exists():
            raise FileNotFoundError(f"Dataset folder not found: {abs_dir}")

        for path in sorted(abs_dir.rglob("*")):
            if not path.is_file():
                continue
            rel = path.relative_to(EASYNCO_DATASETS).as_posix()

            # test_dataset_tsp_uniform/*.pt
            m = _RE_TEST_TSP_UNIFORM.match(path.name)
            if m:
                scale = int(m.group("scale"))
                episodes = int(m.group("num"))
                specs.append(DatasetSpec(key=rel, scale=scale, episodes=episodes))
                continue

            # cross_distribution/tsp*.pkl
            m = _RE_CROSS_DIST_TSP.match(path.name)
            if m:
                scale = int(m.group("scale"))
                # episodes from pickle length (these are python lists in this repo)
                with path.open("rb") as f:
                    data = pickle.load(f)
                episodes = len(data)
                specs.append(DatasetSpec(key=rel, scale=scale, episodes=episodes))
                continue

    # de-dup by key
    uniq: Dict[str, DatasetSpec] = {}
    for s in specs:
        uniq.setdefault(s.key, s)
    return list(uniq.values())


def _parse_eval_done_metrics(log_text: str) -> Dict[str, Optional[float]]:
    """
    Extract final aggregates from EasyNCO logs.
    """
    metrics: Dict[str, Optional[float]] = {
        "avg_score": None,
        "avg_aug_score": None,
        "total_time_sec": None,
        "avg_time_per_episode_sec": None,
    }

    # Example:
    # Eval Done ==> Problem[tsp], Score Summary: Avg: 123.4567, Avg Augmented: 120.0000
    m = re.search(
        r"Eval Done ==> Problem\\[[^\\]]+\\], Score Summary: Avg: ([0-9.+-eE]+), Avg Augmented: ([0-9.+-eE]+)",
        log_text,
    )
    if m:
        metrics["avg_score"] = float(m.group(1))
        metrics["avg_aug_score"] = float(m.group(2))

    m = re.search(r"Total Testing Time: ([0-9.+-eE]+) seconds", log_text)
    if m:
        metrics["total_time_sec"] = float(m.group(1))

    m = re.search(r"Avg Time per Episode: ([0-9.+-eE]+) seconds", log_text)
    if m:
        metrics["avg_time_per_episode_sec"] = float(m.group(1))

    return metrics


def _read_summary_existing(path: Path) -> Dict[Tuple[str, str], Dict[str, str]]:
    if not path.exists():
        return {}
    existing: Dict[Tuple[str, str], Dict[str, str]] = {}
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            key = (row.get("method", ""), row.get("dataset", ""))
            existing[key] = row
    return existing


def _ensure_ckpt_exists(method: MethodSpec) -> None:
    ckpt = EASYNCO_PRETRAINED / method.ckpt_filename
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found for {method.name}: {ckpt}")


def _run_one(
    *,
    method: MethodSpec,
    dataset: DatasetSpec,
    cuda: str,
    decoder_strategy: str,
    aug_factor: int,
    batch_size: int,
    seed: int,
    out_dir: Path,
    dry_run: bool,
) -> Tuple[str, Path]:
    """
    Returns (status, log_path)
      status in {"ok","skipped","error"}
    """
    ok, reason = method.is_compatible(dataset.scale)
    log_dir = out_dir / "logs" / method.name
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{dataset.key.replace('/', '__')}.log"

    if not ok:
        log_path.write_text(f"[skipped] {reason}\n", encoding="utf-8")
        return "skipped", log_path

    # Some methods have multiple checkpoints by scale.
    ckpt_filename = method.ckpt_filename
    if method.name == "dact":
        ckpt_filename = "dact_tsp50.ckpt" if dataset.scale == 50 else "dact_tsp100.ckpt"

    ckpt_path = EASYNCO_PRETRAINED / ckpt_filename
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found for {method.name}: {ckpt_path}")

    # Build hydra overrides.
    overrides: List[str] = [
        f"settings={method.settings}",
        "mode=test",
        f"model={method.name}",
        "problem=tsp",
        f"scale={dataset.scale}",
        f"decoder_strategy={decoder_strategy}",
        f"test_data_path={dataset.key}",
        _cuda_override(cuda),
        f"seed={seed}",
        f"episodes={dataset.episodes}",
        f"batch_size={batch_size}",
        # Keep all runs under a dedicated benchmark root.
        f"dir=./results/benchmarks/tsp_datasets/{method.name}/{dataset.key.replace('/', '_')}/${{now:%Y-%m-%d-%H-%M-%S}}",
        # Make ckpt explicit for reproducibility.
        "settings.test_loader.model_dirpath=pretrained",
        f"settings.test_loader.model_filename={ckpt_filename}",
        # Optional global aug override.
        f"++settings.env.aug_factor={aug_factor}",
        # Force initialization-only evaluation (do not run method iteration).
        "settings.iteration._target_=benchmarks.easynco_init_only_iteration.InitOnlyIteration",
    ]

    cmd = [sys.executable, str(EASYNCO_EVAL)] + overrides

    if dry_run:
        log_path.write_text("DRY_RUN:\n" + " ".join(cmd) + "\n", encoding="utf-8")
        return "ok", log_path

    env = os.environ.copy()
    # Keep the run reproducible and avoid noisy warnings.
    env.setdefault("PYTHONWARNINGS", "ignore")

    start = time.time()
    proc = subprocess.run(
        cmd,
        cwd=str(REPO_ROOT),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    elapsed = time.time() - start

    log_path.write_text(proc.stdout, encoding="utf-8")
    if proc.returncode != 0:
        log_path.write_text(
            log_path.read_text(encoding="utf-8")
            + f"\n[runner] returncode={proc.returncode} elapsed_sec={elapsed:.3f}\n",
            encoding="utf-8",
        )
        return "error", log_path

    # Add runner wall time (useful if EasyNCO doesn't reach on_test_end on crash)
    log_path.write_text(
        log_path.read_text(encoding="utf-8") + f"\n[runner] elapsed_sec={elapsed:.3f}\n",
        encoding="utf-8",
    )
    return "ok", log_path


def main() -> None:
    args = _parse_args()

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset_folders = [x.strip() for x in args.datasets.split(",") if x.strip()]
    method_names = [x.strip() for x in args.methods.split(",") if x.strip()]

    datasets = _discover_tsp_datasets(dataset_folders)
    if not datasets:
        raise RuntimeError("No TSP datasets discovered under the given folders.")

    methods: List[MethodSpec] = []
    for m in method_names:
        if m not in METHODS:
            raise ValueError(f"Unknown method: {m}. Available: {','.join(METHODS)}")
        methods.append(METHODS[m])

    summary_path = out_dir / "summary.tsv"
    existing = _read_summary_existing(summary_path) if args.resume else {}

    header = [
        "method",
        "dataset",
        "scale",
        "episodes",
        "batch_size",
        "aug_factor",
        "seed",
        "cuda",
        "decoder_strategy",
        "status",
        "avg_score",
        "avg_aug_score",
        "total_time_sec",
        "avg_time_per_episode_sec",
        "log_path",
    ]

    # Open summary for append; create if missing.
    if not summary_path.exists():
        summary_path.write_text("\t".join(header) + "\n", encoding="utf-8")

    with summary_path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header, delimiter="\t")

        for dataset in datasets:
            for method in methods:
                key = (method.name, dataset.key)
                if args.resume and key in existing:
                    continue

                batch_size = _pick_batch_size(dataset.scale, dataset.episodes, args.batch_size)

                status, log_path = _run_one(
                    method=method,
                    dataset=dataset,
                    cuda=args.cuda,
                    decoder_strategy=args.decoder_strategy,
                    aug_factor=args.aug_factor,
                    batch_size=batch_size,
                    seed=args.seed,
                    out_dir=out_dir,
                    dry_run=args.dry_run,
                )

                metrics = {"avg_score": None, "avg_aug_score": None, "total_time_sec": None, "avg_time_per_episode_sec": None}
                if status == "ok" and not args.dry_run:
                    metrics = _parse_eval_done_metrics(log_path.read_text(encoding="utf-8"))

                row = {
                    "method": method.name,
                    "dataset": dataset.key,
                    "scale": dataset.scale,
                    "episodes": dataset.episodes,
                    "batch_size": batch_size,
                    "aug_factor": args.aug_factor,
                    "seed": args.seed,
                    "cuda": args.cuda,
                    "decoder_strategy": args.decoder_strategy,
                    "status": status,
                    "avg_score": "" if metrics["avg_score"] is None else f"{metrics['avg_score']:.6f}",
                    "avg_aug_score": "" if metrics["avg_aug_score"] is None else f"{metrics['avg_aug_score']:.6f}",
                    "total_time_sec": "" if metrics["total_time_sec"] is None else f"{metrics['total_time_sec']:.3f}",
                    "avg_time_per_episode_sec": ""
                    if metrics["avg_time_per_episode_sec"] is None
                    else f"{metrics['avg_time_per_episode_sec']:.6f}",
                    "log_path": str(log_path),
                }
                writer.writerow(row)
                f.flush()

                if status == "error" and not args.continue_on_error:
                    raise RuntimeError(f"Run failed for method={method.name} dataset={dataset.key}. See {log_path}")


if __name__ == "__main__":
    main()
