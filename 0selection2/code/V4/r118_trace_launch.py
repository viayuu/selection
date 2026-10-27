"""Finite two-3090 TRAIN capture using the existing allocation transport."""

import argparse
import os
from pathlib import Path
import subprocess
import sys
import time

from . import r117_launch as transport
from . import r118_trace_assets as data


resource = transport.resource
ROOT = data.ROOT
PREP = ROOT / "preparation"
WRAPPER = resource.REPO / "code/V4/run_v4_r118_trace.sh"
MODE = "capture"
SLOTS = dict(A="3090_0", B="3090_1")

# Reuse the proven entry/dispatch functions in this process, not their old jobs.
transport.ROOT, transport.PREP, transport.WRAPPER = ROOT, PREP, WRAPPER
transport.TOKEN, transport.MODE, transport.ARM = "R118_CAPTURE_TOKEN", "R118_CAPTURE_MODE", "R118_CAPTURE_ARM"
transport.SLOTS = SLOTS


def prepare():
    plan = data.load_plan()
    files = ["r118_trace_assets.py", "r118_reld_trace.py", "r118_td_trace.py",
             "r118_trace_launch.py", "run_v4_r118_trace.sh", "r117_launch.py", "r113_3090_queue.py"]
    sources = [resource.REPO / "code/V4" / name for name in files] + [ROOT / "capture_plan.json"]
    value = dict(main_approved=True, one_attempt_only=True, mode=MODE, no_new_allocation=True,
        token=resource.uuid.uuid4().hex, mutex_inodes=resource.mutex_inodes(),
        sources={str(p):resource.digest(p) for p in sources}, arms=list(SLOTS),
        slot_for_arm=SLOTS, max_seconds=2 * 3600, plan_sha256=plan["plan_sha256"],
        selector_training=False, heldout_inference=False, actual_solver_capture_authorized=True)
    resource.save(transport.ready_path(MODE), value, exclusive=True)
    return value


def worker(arm):
    value = transport.authorize(MODE, arm)
    slot = value["slot_for_arm"][arm]
    item = resource.require_slot(slot)
    directory = transport.folder(MODE, arm)
    receipt = dict(state="checking", pid=os.getpid(), arm=arm, gpu_uuid=item["uuid"],
        started_utc=resource.utc(), stages=[], selector_training=False)
    resource.save(directory / "worker.json", receipt, exclusive=True)
    try:
        with resource.original_mutex(slot, value["mutex_inodes"]) as fd:
            receipt["checks"] = resource.worker_checks(value, slot, directory)
            env = resource.environment(item["uuid"])
            for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
                env[key] = "1"
            receipt.update(state="running", initialized=True)
            resource.save(directory / "worker.json", receipt)
            deadline = time.monotonic() + value["max_seconds"]
            for problem in data.SLOTS[arm]:
                argv = [sys.executable, "-B", "-u", "-m", "code.V4.r118_trace_assets", "capture",
                        "--problem", problem, "--allow-native-capture"]
                log = ROOT / "capture" / problem / "capture.log"
                log.parent.mkdir(parents=True, exist_ok=True)
                stage = dict(stage=problem, argv=argv, state="running", started_utc=resource.utc(), log=str(log))
                receipt["stages"].append(stage)
                with log.open("a") as stream:
                    child = subprocess.Popen(argv, cwd=resource.REPO, env=env,
                        stdout=stream, stderr=subprocess.STDOUT, pass_fds=(fd,))
                    stage["child_pid"] = child.pid
                    resource.save(directory / "worker.json", receipt)
                    try:
                        rc = child.wait(timeout=max(1., deadline - time.monotonic()))
                    except subprocess.TimeoutExpired:
                        resource.wait_owned(child)
                        stage.update(state="incomplete_budget", exit_status=124)
                        receipt.update(state="incomplete_budget", exit_status=124)
                        return 124
                    except BaseException:
                        resource.wait_owned(child)
                        raise
                stage.update(state="complete" if rc == 0 else "failed", exit_status=rc, finished_utc=resource.utc())
                resource.save(directory / "worker.json", receipt)
                if rc:
                    receipt.update(state="failed", exit_status=rc)
                    return rc
            receipt.update(state="complete", exit_status=0)
            return 0
    except BaseException as error:
        receipt.update(state="failed", exit_status=1, error_type=type(error).__name__, error=str(error))
        raise
    finally:
        receipt["finished_utc"] = resource.utc()
        resource.save(directory / "worker.json", receipt)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "dispatch", "entry", "worker"))
    parser.add_argument("--arm", choices=tuple(SLOTS), default=os.environ.get(transport.ARM))
    args = parser.parse_args()
    if args.command == "prepare":
        print(resource.json.dumps(prepare()))
        return 0
    if args.arm is None:
        parser.error("Explicit capture arm required")
    if args.command == "worker":
        return worker(args.arm)
    return getattr(transport, args.command)(MODE, args.arm)


if __name__ == "__main__":
    raise SystemExit(main())
