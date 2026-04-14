#!/usr/bin/env python
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

SCRIPT_DIR = Path(__file__).resolve().parent
BRIDGE_ROOT = SCRIPT_DIR.parent
if str(BRIDGE_ROOT) not in sys.path:
    sys.path.insert(0, str(BRIDGE_ROOT))

from config import MANIFEST_ROOT, PYTHONPATH_ROOT, ensure_layout


def count_jsonl_lines(path: Path) -> int:
    if not path.is_file():
        return 0
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for _ in f if _.strip())


def load_jobs(path: Path) -> List[Dict]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def should_skip(job: dict) -> bool:
    expected_output = job.get("expected_output")
    expected_lines = job.get("expected_lines")
    if not expected_output or expected_lines is None:
        return False
    return count_jsonl_lines(Path(expected_output)) == int(expected_lines)


def run_job(job: dict, dry_run: bool, continue_on_error: bool) -> None:
    cmd = job["command"]
    cwd = job["cwd"]
    log_path = Path(job["log_path"])
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if should_skip(job):
        print(f"SKIP {job['name']} (output already complete)")
        return
    print(f"RUN  {job['name']}")
    print("     ", " ".join(cmd))
    if dry_run:
        return

    env = os.environ.copy()
    env["PYTHONPATH"] = str(PYTHONPATH_ROOT)
    with log_path.open("w", encoding="utf-8") as f:
        f.write("COMMAND:\n")
        f.write(" ".join(cmd))
        f.write("\n\n")
        f.flush()
        completed = subprocess.run(
            cmd,
            cwd=cwd,
            env=env,
            stdout=f,
            stderr=subprocess.STDOUT,
            text=True,
        )
    if completed.returncode != 0 and not continue_on_error:
        raise SystemExit(completed.returncode)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the V3 label queue sequentially.")
    parser.add_argument("--jobs", default=str(MANIFEST_ROOT / "label_jobs.json"))
    parser.add_argument(
        "--groups",
        default="",
        help="Comma-separated subset: nss,atsp,mvrp",
    )
    parser.add_argument("--name", default="", help="Run only jobs whose name contains this substring.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    args = parser.parse_args()

    ensure_layout()
    jobs = load_jobs(Path(args.jobs))
    if args.groups:
        wanted = {item.strip() for item in args.groups.split(",") if item.strip()}
        jobs = [job for job in jobs if job["group"] in wanted]
    if args.name:
        jobs = [job for job in jobs if args.name in job["name"]]

    for job in jobs:
        run_job(job, dry_run=args.dry_run, continue_on_error=args.continue_on_error)


if __name__ == "__main__":
    main()
