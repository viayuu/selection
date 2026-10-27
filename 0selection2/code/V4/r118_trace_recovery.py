"""Explicit observer-only recovery; original capture sources/receipts stay intact."""

import argparse
import os
from pathlib import Path
import subprocess
import sys
import time

from . import r118_reld_trace as observer
from . import r118_trace_assets as data
from . import r118_trace_launch as launch


resource, transport = launch.resource, launch.transport
ORIGINAL_ROOT, ORIGINAL_ASSETS, ORIGINAL_VERSION = data.ROOT, data.ASSETS, data.VERSION
ROOT = ORIGINAL_ROOT / "recovery_column_indices_v2"
ASSETS = ORIGINAL_ASSETS / "recovery_column_indices_v2"
VERSION = ORIGINAL_VERSION + "_column_indices_v2"
WRAPPER = resource.REPO / "code/V4/run_v4_r118_trace_recovery.sh"
data.ROOT, data.ASSETS, data.VERSION = ROOT, ASSETS, VERSION
transport.ROOT, transport.PREP, transport.WRAPPER = ROOT, ROOT / "preparation", WRAPPER
transport.TOKEN, transport.MODE, transport.ARM = "R118_REC_TOKEN", "R118_REC_MODE", "R118_REC_ARM"


def prepare():
    failed_path = ORIGINAL_ROOT / "preparation/resources/capture/B/worker.json"
    failed = resource.read(failed_path)
    case = resource.read(ORIGINAL_ROOT / "capture/CVRP/status.json")
    if (failed.get("state") != "failed" or case.get("completed") != 0
            or case.get("error_type") != "AttributeError" or "POMO_IDX" not in case.get("error", "")):
        raise RuntimeError("This explicit repair applies only to the preserved CVRP column-index failure")
    plan = resource.read(ORIGINAL_ROOT / "capture_plan.json")
    old_sha = plan.pop("plan_sha256")
    plan.update(version=VERSION, asset_root=str(ASSETS), max_array_bytes=1024 * 2**20,
        recovery=dict(original_plan_sha256=old_sha, original_failed_worker=str(failed_path),
            original_failed_worker_sha256=resource.digest(failed_path),
            observer_fix="derive column-index metadata when native CVRP has no POMO_IDX; no native state writes",
            memory_bound_reason="full POMO arrays through up to2N actions may exceed384MiB;1GiB observer bound fits9GiB CPU allocation",
            input_membership_or_backend_or_labels_changed=False, automatic_retry=False))
    plan["plan_sha256"] = data.common.canonical_hash(plan)
    data.common.write_new_json(ROOT / "capture_plan.json", plan)
    files = [resource.REPO / "code/V4" / name for name in
        ("r118_trace_assets.py", "r118_reld_trace.py", "r118_td_trace.py", "r118_trace_launch.py",
         "r118_trace_recovery.py", "run_v4_r118_trace_recovery.sh", "r117_launch.py", "r113_3090_queue.py")]
    files.append(ROOT / "capture_plan.json")
    ready = dict(main_approved=True, one_attempt_only=True, mode="capture", no_new_allocation=True,
        token=resource.uuid.uuid4().hex, mutex_inodes=resource.mutex_inodes(),
        sources={str(p):resource.digest(p) for p in files}, arms=["B"],
        slot_for_arm=dict(B="3090_1"), max_seconds=2 * 3600, plan_sha256=plan["plan_sha256"],
        selector_training=False, input_membership_or_backend_or_labels_changed=False,
        manual_recovery_of=str(failed_path))
    resource.save(transport.ready_path("capture"), ready, exclusive=True)
    return ready


def capture(problem):
    import torch
    original_before, original_capture = observer._Trace.before, observer.capture_reld

    class IndexView:
        def __init__(self, env):
            self.env = env

        def __getattr__(self, name):
            if name == "POMO_IDX":
                width = self.env.ninf_mask.shape[1]
                return torch.arange(width, device=self.env.ninf_mask.device)[None, :]
            return getattr(self.env, name)

    def before(trace, env, selected):
        view = env if hasattr(env, "POMO_IDX") else IndexView(env)
        return original_before(trace, view, selected)

    def record(*args, **kwargs):
        result, arrays, info = original_capture(*args, **kwargs)
        info["column_index_metadata"] = "native POMO_IDX if present; otherwise tensor column index; actual physical starts separately recorded"
        info["observer_recovery"] = "r118_column_indices_v2; native environment not modified"
        return result, arrays, info

    observer._Trace.before, observer.capture_reld = before, record
    try:
        return data.run_problem(problem)
    finally:
        observer._Trace.before, observer.capture_reld = original_before, original_capture


def worker():
    ready = transport.authorize("capture", "B")
    slot, directory = "3090_1", transport.folder("capture", "B")
    item = resource.require_slot(slot)
    receipt = dict(state="checking", pid=os.getpid(), arm="B", gpu_uuid=item["uuid"],
        started_utc=resource.utc(), stages=[], selector_training=False, explicit_recovery=True)
    resource.save(directory / "worker.json", receipt, exclusive=True)
    try:
        with resource.original_mutex(slot, ready["mutex_inodes"]) as fd:
            receipt["checks"] = resource.worker_checks(ready, slot, directory)
            env = resource.environment(item["uuid"])
            for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
                env[key] = "1"
            receipt.update(state="running", initialized=True)
            resource.save(directory / "worker.json", receipt)
            deadline = time.monotonic() + ready["max_seconds"]
            for problem in data.SLOTS["B"]:
                argv = [sys.executable, "-B", "-u", "-m", "code.V4.r118_trace_recovery", "capture",
                        "--problem", problem, "--allow-native-capture"]
                log = ROOT / "capture" / problem / "capture.log"
                log.parent.mkdir(parents=True, exist_ok=True)
                stage = dict(stage=problem, state="running", argv=argv, log=str(log), started_utc=resource.utc())
                receipt["stages"].append(stage)
                with log.open("a") as stream:
                    child = subprocess.Popen(argv, cwd=resource.REPO, env=env,
                        stdout=stream, stderr=subprocess.STDOUT, pass_fds=(fd,))
                    stage["child_pid"] = child.pid
                    resource.save(directory / "worker.json", receipt)
                    try:
                        rc = child.wait(timeout=max(1., deadline - time.monotonic()))
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
    parser.add_argument("command", choices=("prepare", "dispatch", "entry", "worker", "capture"))
    parser.add_argument("--arm", choices=("B",), default=os.environ.get(transport.ARM, "B"))
    parser.add_argument("--problem", choices=data.SLOTS["B"])
    parser.add_argument("--allow-native-capture", action="store_true")
    args = parser.parse_args()
    if args.command == "prepare":
        print(resource.json.dumps(prepare()))
        return 0
    if args.command == "worker":
        return worker()
    if args.command == "capture":
        if not args.problem or not args.allow_native_capture:
            parser.error("Explicit frozen TRAIN problem and --allow-native-capture required")
        return capture(args.problem)
    return getattr(transport, args.command)("capture", "B")


if __name__ == "__main__":
    raise SystemExit(main())
