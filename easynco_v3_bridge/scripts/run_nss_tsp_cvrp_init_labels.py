import argparse
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional


WORKSPACE = Path("/public/home/zhoucl/shiys")
EASY_ROOT = WORKSPACE / "EasyNCO"
RESULTS_TEST = EASY_ROOT / "results" / "test"
NSS_STATUS_CSV = EASY_ROOT / "results" / "nss_eval" / "运行清单.csv"

TSP_METHODS = [
    "dact",
    "difusco",
    "elg",
    "glop",
    "icam",
    "invit",
    "lehd",
    "lih",
    "omni",
    "t2t",
    "pointerformer",
    "udc",
]

CVRP_METHODS = [
    "dact",
    "elg",
    "glop",
    "icam",
    "invit",
    "lehd",
    "lih",
    "omni",
    "udc",
]


def ensure_glop_cvrp_base_override() -> Path:
    """Create a minimal base hydra override so run_nss_eval_suite can discover glop_cvrp."""
    existing = sorted(RESULTS_TEST.glob("glop_cvrp/*/.hydra/overrides.yaml"))
    if existing:
        return existing[-1]

    stamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    hydra_dir = RESULTS_TEST / "glop_cvrp" / f"{stamp}_glop_cvrp_100_test" / ".hydra"
    hydra_dir.mkdir(parents=True, exist_ok=True)
    override_path = hydra_dir / "overrides.yaml"
    overrides = [
        "- mode=test",
        "- settings=glop_settings",
        "- model=glop",
        "- problem=cvrp",
        "- scale=100",
        "- batch_size=16",
        "- decoder_strategy=greedy",
        "- cuda=[0]",
        "- test_data_path=test_dataset_vrp_uniform/test_vrp100_capacity50_nums10000_uniform.pt",
        "- settings.iteration._target_=EasyNCO.neural_solvers.pipeline.NoIteration",
        "- settings.test_loader.model_dirpath=pretrained/glop",
        "- settings.test_loader.model_filename=glop_policy_cvrp_1000.pt",
    ]
    override_path.write_text("\n".join(overrides) + "\n", encoding="utf-8")
    return override_path


def archive_status_csv() -> Optional[Path]:
    if not NSS_STATUS_CSV.exists():
        return None
    backup = NSS_STATUS_CSV.with_name(
        f"{NSS_STATUS_CSV.stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}{NSS_STATUS_CSV.suffix}"
    )
    shutil.move(str(NSS_STATUS_CSV), str(backup))
    return backup


def build_command(args: argparse.Namespace) -> List[str]:
    method_list = []
    for method in TSP_METHODS + CVRP_METHODS:
        if method not in method_list:
            method_list.append(method)
    if args.methods:
        wanted = [item.strip() for item in args.methods.split(",") if item.strip()]
        method_list = [method for method in method_list if method in wanted]
    command = [
        sys.executable,
        str(EASY_ROOT / "run_nss_eval_suite.py"),
        "--easy-root",
        str(EASY_ROOT),
        "--cuda",
        args.cuda,
        "--methods",
        ",".join(method_list),
    ]
    if args.datasets:
        command.extend(["--datasets", args.datasets])
    if args.sample_size > 0:
        command.extend(["--sample-size", str(args.sample_size)])
    if args.max_runs > 0:
        command.extend(["--max-runs", str(args.max_runs)])
    if args.stop_on_error:
        command.append("--stop-on-error")
    return command


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", default="[0]")
    parser.add_argument("--methods", default="")
    parser.add_argument("--datasets", default="")
    parser.add_argument("--sample-size", type=int, default=0)
    parser.add_argument("--max-runs", type=int, default=0)
    parser.add_argument("--archive-status", action="store_true")
    parser.add_argument("--stop-on-error", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    override_path = ensure_glop_cvrp_base_override()
    backup = archive_status_csv() if args.archive_status else None

    command = build_command(args)
    print(f"glop_cvrp_base_override: {override_path}")
    if backup is not None:
        print(f"archived_status_csv: {backup}")
    print("command:")
    print(" ".join(command))

    if args.dry_run:
        return 0

    env = os.environ.copy()
    env.setdefault("PYTHONPATH", str(WORKSPACE))
    completed = subprocess.run(command, cwd=str(EASY_ROOT), env=env)
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
