"""Finite R116 work inside the two existing RTX3090 allocation/container slots."""

import argparse
import fcntl
import os
from pathlib import Path
import re
import shlex
import socket
import subprocess
import sys
import time

from . import r113_3090_queue as resource


ROOT = resource.REPO / "code/V4/runs/R116_subset_elimination"
PREP = ROOT / "preparation"
WRAPPER = resource.REPO / "code/V4/run_v4_r116.sh"
TOKEN = "R116_LAUNCH_TOKEN"
PHASE = "R116_LAUNCH_MODE"
ARM = "R116_LAUNCH_ARM"
SLOTS = dict(A_static="3090_0", B_subset="3090_1")


def ready_path(mode):
    return PREP / (mode + "_ready.json")


def folder(mode, arm):
    return PREP / "resources" / mode / arm


def prepare(mode):
    if mode == "capacity":
        sources = ("r116_model.py", "r116_subset.py", "r116_capacity.py", "r116_launch.py", "run_v4_r116.sh")
        data = dict(arms=["B_subset"], slot_for_arm={"B_subset": "3090_0"}, max_seconds=1200)
    else:
        from . import r116_experiment as runner
        from .r116_capacity import CAPS
        capacity_path = folder("capacity", "B_subset") / "capacity.json"
        witness = resource.read(capacity_path)
        if witness.get("complete") is not True or witness.get("microbatch_caps") != CAPS:
            raise ValueError("One successful capacity witness is required before formal lock")
        caps = witness["microbatch_caps"]
        lock = dict(runner.required_lock_fields(), data_root=str(runner.DEFAULT_DATA_ROOT),
            prepared_root=str(runner.DEFAULT_PREPARED_ROOT), primary_cache="/tmp/shiys_R84_new90k_cache",
            microbatch_caps=caps, eval_batch_size=witness["eval_batch_size"],
            capacity_witness=str(capacity_path), capacity_witness_sha256=resource.digest(capacity_path),
            authorized_slots=resource.SLOTS, gpu_budget_hours_per_arm=14,
            complete_gpu_budget_hours=28, no_4090=True)
        resource.save(ROOT / "protocol_lock.json", lock, exclusive=True)
        sources = ("r116_model.py", "r116_subset.py", "r116_objective.py", "r116_experiment.py", "r116_launch.py", "run_v4_r116.sh")
        data = dict(arms=list(SLOTS), slot_for_arm=SLOTS, max_seconds=14 * 3600,
                    protocol_lock=str(ROOT / "protocol_lock.json"),
                    protocol_lock_sha256=resource.digest(ROOT / "protocol_lock.json"))
    sources = [resource.REPO / "code/V4" / name for name in sources]
    value = dict(main_approved=True, one_attempt_only=True, mode=mode, no_new_allocation=True,
        token=resource.uuid.uuid4().hex, mutex_inodes=resource.mutex_inodes(),
        sources={str(path): resource.digest(path) for path in sources}, **data)
    resource.save(ready_path(mode), value, exclusive=True)
    return value


def load(mode):
    value = resource.read(ready_path(mode))
    if value.get("main_approved") is not True or value.get("mode") != mode:
        raise ValueError("Require this version's explicit main-ready receipt")
    for name, expected in value["sources"].items():
        if resource.digest(name) != expected:
            raise RuntimeError("Frozen executable source changed: " + name)
    if mode == "training" and resource.digest(value["protocol_lock"]) != value["protocol_lock_sha256"]:
        raise RuntimeError("Formal protocol changed after main-ready")
    return value


def command(command_name, arm):
    return [sys.executable, "-B", "-u", "-m", "code.V4.r116_experiment", command_name,
            "--root", str(ROOT), "--arm", arm, "--gpu-type", "3090", "--device", "cuda:0",
            "--wandb-mode", "offline", "--cache-root", "/tmp/shiys_R84_new90k_cache",
            "--protocol-lock", str(ROOT / "protocol_lock.json"), "--log-every", "18"]


def authorize(mode, arm):
    value = load(mode)
    if arm not in value["arms"] or os.environ.get(TOKEN) != value["token"]:
        raise ValueError("Missing current main-owned worker token or arm")
    return value


def entry(mode, arm):
    value = authorize(mode, arm)
    slot = value["slot_for_arm"][arm]
    item = resource.require_slot(slot)
    directory = folder(mode, arm)
    cgroup = Path("/proc/self/cgroup").read_text()
    if (socket.gethostname().split(".")[0] != "gpu03" or os.environ.get("SLURM_JOB_ID") != "1465"
            or not re.search(r"/job_1465(?:/|$)", cgroup)):
        raise RuntimeError("Use only original allocation1465 entry, never direct SSH training")
    allocation = resource.original.r81_queue.allocation(item)
    container = resource.original.r81_queue.container_inspect(item)
    allowed = sorted(set(allocation["cpus"]) & set(container["allowed_cpus"]))
    cpus = allowed[item["index"] * 4:(item["index"] + 1) * 4]
    if len(cpus) != 4:
        raise RuntimeError("Require four disjoint authorized CPUs per existing slot")
    resource.save(directory / "entry.json", dict(state="container_exec", cpus=cpus,
        allocation=allocation, container=container, cgroup=cgroup), exclusive=True)
    env = resource.environment(item["uuid"])
    env.update({resource.CPU_ENV: ",".join(map(str, cpus)), TOKEN: value["token"], PHASE: mode, ARM: arm})
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[key] = "2"
    argv = ["sudo", "-n", "/usr/bin/ai_docker", "exec", "-u", "2022"]
    for key in ("CUDA_VISIBLE_DEVICES", "PYTHONPATH", "PYTHONDONTWRITEBYTECODE", "MPLBACKEND",
                "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS",
                "PYTORCH_CUDA_ALLOC_CONF", "WANDB_MODE", "SLURM_JOB_ID", "SLURM_STEP_ID",
                resource.CPU_ENV, TOKEN, PHASE, ARM):
        argv.extend(["-e", key + "=" + env[key]])
    argv.extend([resource.CONTAINER, "taskset", "-c", ",".join(map(str, cpus)),
                 "bash", str(WRAPPER), "worker"])
    rc = subprocess.call(argv, cwd=resource.REPO, env=env)
    worker_path = directory / "worker.json"
    if not worker_path.exists():
        raise RuntimeError("Container transport returned without an actual worker receipt")
    worker_record = resource.read(worker_path)
    return worker_record.get("exit_status", rc if rc else 1)


def worker(mode, arm):
    value = authorize(mode, arm)
    directory = folder(mode, arm)
    slot = value["slot_for_arm"][arm]
    item = resource.require_slot(slot)
    receipt = dict(state="checking", initialized=False, mode=mode, arm=arm,
        pid=os.getpid(), started_utc=resource.utc(), gpu_uuid=item["uuid"], stages=[])
    resource.save(directory / "worker.json", receipt, exclusive=True)
    try:
        with resource.original_mutex(slot, value["mutex_inodes"]) as fd:
            receipt["checks"] = resource.worker_checks(value, slot, directory)
            if mode == "capacity":
                stages = [("capacity", [sys.executable, "-B", "-u", "-m", "code.V4.r116_capacity",
                                         "--output", str(directory / "capacity.json")])]
            else:
                resource.original.check_cache({"plan": {"protocol_lock": value["protocol_lock"]}})
                stages = [("train", command("train", arm)), ("finalize", command("finalize", arm))]
            deadline = time.monotonic() + value["max_seconds"]
            env = resource.environment(item["uuid"])
            for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
                env[key] = "2"
            receipt.update(state="running", initialized=True)
            resource.save(directory / "worker.json", receipt)
            for name, argv in stages:
                row = dict(stage=name, argv=argv, state="running", started_utc=resource.utc())
                receipt["stages"].append(row)
                log_path = (ROOT / arm / (name + ".log")) if mode == "training" else directory / (name + ".log")
                log_path.parent.mkdir(parents=True, exist_ok=True)
                with log_path.open("a") as log:
                    child = subprocess.Popen(argv, cwd=resource.REPO, env=env,
                        stdout=log, stderr=subprocess.STDOUT, pass_fds=(fd,))
                    row.update(child_pid=child.pid, log=str(log_path))
                    resource.save(directory / "worker.json", receipt)
                    try:
                        rc = child.wait(timeout=max(1., deadline - time.monotonic()))
                    except subprocess.TimeoutExpired:
                        resource.wait_owned(child)
                        row.update(state="incomplete_budget", exit_status=124, finished_utc=resource.utc())
                        receipt.update(state="incomplete_budget", exit_status=124,
                            reason="Predeclared GPU wall budget; existing saved state retained, no automatic retry")
                        return 124
                    except BaseException:
                        resource.wait_owned(child)
                        raise
                row.update(state="complete" if rc == 0 else "failed", exit_status=rc, finished_utc=resource.utc())
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


def dispatch(mode, arm):
    value = load(mode)
    if arm not in value["arms"]:
        raise ValueError("Arm not in this finite dispatch")
    directory = folder(mode, arm)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "controller.lock").open("a") as mutex:
        fcntl.flock(mutex, fcntl.LOCK_EX | fcntl.LOCK_NB)
        resource.save(directory / "request.json", dict(started_utc=resource.utc(),
            ready_sha256=resource.digest(ready_path(mode)), mode=mode, arm=arm), exclusive=True)
        env = resource.environment()
        env.update({TOKEN: value["token"], PHASE: mode, ARM: arm})
        argv = ["srun", "--jobid=1465", "--overlap", "--exact", "--gres=none", "--immediate=5",
                "-N1", "-n1", "--nodelist=gpu03", "--cpus-per-task=4", "--mem=9G",
                "bash", str(WRAPPER), "entry"]
        child = subprocess.Popen(argv, cwd=resource.REPO, env=env)
        record = dict(state="running", pid=os.getpid(), transport_pid=child.pid, argv=argv,
                      mode=mode, arm=arm, updated_utc=resource.utc())
        resource.save(directory / "controller.json", record)
        try:
            rc = child.wait()
        except BaseException:
            resource.wait_owned(child)
            raise
        record.update(state="complete" if rc == 0 else "failed", exit_status=rc, updated_utc=resource.utc())
        resource.save(directory / "controller.json", record)
        return rc


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "dispatch", "entry", "worker"))
    parser.add_argument("--mode", choices=("capacity", "training"), default=os.environ.get(PHASE))
    parser.add_argument("--arm", choices=tuple(SLOTS), default=os.environ.get(ARM))
    args = parser.parse_args()
    if args.mode is None:
        parser.error("Explicit capacity/training mode required")
    if args.command == "prepare":
        value = prepare(args.mode)
        print(resource.json.dumps(dict(ready=str(ready_path(args.mode)), mode=value["mode"])))
        return 0
    if args.arm is None:
        parser.error("Explicit arm required")
    return globals()[args.command](args.mode, args.arm)


if __name__ == "__main__":
    raise SystemExit(main())
