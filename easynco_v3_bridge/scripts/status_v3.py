#!/usr/bin/env python
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
BRIDGE_ROOT = SCRIPT_DIR.parent
if str(BRIDGE_ROOT) not in sys.path:
    sys.path.insert(0, str(BRIDGE_ROOT))

from config import ACTIVE_EXPERIMENT_GROUPS
from monitor_v3 import build_snapshot


def main() -> None:
    snapshot = build_snapshot()
    if "nss" in ACTIVE_EXPERIMENT_GROUPS:
        for dataset, meta in sorted(snapshot["nss"].items()):
            print(f"nss:{dataset}: {meta['completed']}/{meta['expected']} complete, failed={meta['failed']}")
    for group, meta in sorted(snapshot["other_groups"].items()):
        if group in ACTIVE_EXPERIMENT_GROUPS:
            print(f"{group}: {meta['done']}/{meta['total']} complete")


if __name__ == "__main__":
    main()
