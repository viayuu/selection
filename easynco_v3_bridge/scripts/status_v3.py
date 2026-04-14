#!/usr/bin/env python
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
BRIDGE_ROOT = SCRIPT_DIR.parent
if str(BRIDGE_ROOT) not in sys.path:
    sys.path.insert(0, str(BRIDGE_ROOT))

from config import MANIFEST_ROOT


def count_jsonl_lines(path: Path) -> int:
    if not path.is_file():
        return 0
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for _ in f if _.strip())


def main() -> None:
    jobs_path = MANIFEST_ROOT / "label_jobs.json"
    if not jobs_path.is_file():
        raise SystemExit(f"Missing jobs file: {jobs_path}")
    with jobs_path.open("r", encoding="utf-8") as f:
        jobs = json.load(f)

    summary = {}
    for job in jobs:
        group = job["group"]
        summary.setdefault(group, {"done": 0, "total": 0})
        summary[group]["total"] += 1
        expected_output = job.get("expected_output")
        expected_lines = job.get("expected_lines")
        if expected_output and expected_lines is not None:
            if count_jsonl_lines(Path(expected_output)) == int(expected_lines):
                summary[group]["done"] += 1

    for group, meta in sorted(summary.items()):
        print(f"{group}: {meta['done']}/{meta['total']} complete")


if __name__ == "__main__":
    main()
