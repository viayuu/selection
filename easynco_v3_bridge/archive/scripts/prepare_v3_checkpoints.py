#!/usr/bin/env python
import argparse
import sys
import zipfile
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
BRIDGE_ROOT = SCRIPT_DIR.parent
if str(BRIDGE_ROOT) not in sys.path:
    sys.path.insert(0, str(BRIDGE_ROOT))

from config import EASYNCO_ROOT


CHECKPOINT_ARCHIVES = {
    "mtpomo": {
        "archive": EASYNCO_ROOT / "pretrained" / "mtpomo" / "mtpomo.zip",
        "target_dir": EASYNCO_ROOT / "pretrained" / "mtpomo",
        "required_files": (
            EASYNCO_ROOT / "pretrained" / "mtpomo" / "mtpomo" / "mtpomo_mvrp_50.ckpt",
            EASYNCO_ROOT / "pretrained" / "mtpomo" / "mtpomo" / "mtpomo_mvrp_100.ckpt",
        ),
    },
    "mvmoe": {
        "archive": EASYNCO_ROOT / "pretrained" / "mvmoe" / "mvmoe.zip",
        "target_dir": EASYNCO_ROOT / "pretrained" / "mvmoe",
        "required_files": (
            EASYNCO_ROOT / "pretrained" / "mvmoe" / "mvmoe" / "mvmoe_mvrp50.ckpt",
            EASYNCO_ROOT / "pretrained" / "mvmoe" / "mvmoe" / "mvmoe_mvrp100.ckpt",
        ),
    },
}


def extract_archive(name: str, force: bool) -> None:
    spec = CHECKPOINT_ARCHIVES[name]
    if not force and all(path.exists() for path in spec["required_files"]):
        print(f"{name}: checkpoints already available")
        return
    if not spec["archive"].is_file():
        raise FileNotFoundError(f"Missing archive for {name}: {spec['archive']}")
    spec["target_dir"].mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(spec["archive"], "r") as zf:
        zf.extractall(spec["target_dir"])
    missing = [str(path) for path in spec["required_files"] if not path.exists()]
    if missing:
        raise FileNotFoundError(f"{name}: extraction finished but files are missing: {missing}")
    print(f"{name}: extracted to {spec['target_dir']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare external V3 checkpoints without editing EasyNCO.")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--only", choices=sorted(CHECKPOINT_ARCHIVES), action="append", default=[])
    args = parser.parse_args()

    targets = args.only or sorted(CHECKPOINT_ARCHIVES)
    for name in targets:
        extract_archive(name, force=args.force)


if __name__ == "__main__":
    main()
