"""Pause only the two legacy R79 workers at their next complete epoch checkpoint."""

import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import signal
import socket
import subprocess
import time

import torch

ROOT = Path(__file__).resolve().parent
EXPECTED = {"TSP": "GPU-cb24f168-9f18-f0d5-6f4a-6d2406958f50",
            "CVRP": "GPU-cb626bdb-4319-2f5e-b91f-17fcaa44a83d"}
if socket.gethostname().split(".")[0] != "gpu03" or os.environ.get("SLURM_JOB_ID") != "1465":
    raise RuntimeError("Legacy R79 handoff is confined to allocation1465/gpu03")


def read_last(path):
    return json.loads(path.read_text().splitlines()[-1])


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def verify_process(pid, task):
    proc = Path(f"/proc/{pid}")
    if proc.stat().st_uid != os.getuid():
        raise RuntimeError("Refuse to signal another user's worker")
    args = shlex.split(proc.joinpath("cmdline").read_bytes().replace(b"\0", b" ").decode())
    env = dict(part.split("=", 1) for part in proc.joinpath("environ").read_bytes().decode().split("\0") if "=" in part)
    if ("code.V4.r79_nss_experiment" not in args or args[args.index("--task") + 1] != task
            or Path(args[args.index("--root") + 1]).resolve() != ROOT
            or env.get("CUDA_VISIBLE_DEVICES") != EXPECTED[task]
            or env.get("SLURM_JOB_ID") != "1465" or env.get("SLURM_STEP_ID") != "149"):
        raise RuntimeError("Owned R79 process identity changed; refuse handoff")


matches = subprocess.check_output(["pgrep", "-af", "^python -B -u -m code.V4.r79_nss_experiment --stage train --task "], text=True)
pids = {}
for line in matches.splitlines():
    pid, command = line.split(" ", 1)
    args = shlex.split(command)
    if Path(args[args.index("--root") + 1]).resolve() == ROOT:
        task = args[args.index("--task") + 1]
        verify_process(int(pid), task)
        status = Path(f"/proc/{pid}/status").read_text().splitlines()
        parent = next(int(line.split()[1]) for line in status if line.startswith("PPid:"))
        parent_args = Path(f"/proc/{parent}/cmdline").read_bytes().replace(b"\0", b" ").decode()
        if "code.V4.r79_nss_experiment" in parent_args:
            continue
        if task in pids:
            raise RuntimeError("Duplicate root R79 training processes")
        pids[task] = int(pid)
if set(pids) != set(EXPECTED):
    raise RuntimeError("Exactly the two owned legacy R79 workers must exist")
targets = {task: read_last(ROOT / task / "epochs.jsonl")["updates"] + 548 for task in pids}
print("HANDOFF_TARGETS", pids, targets, flush=True)
records = {}
while len(records) != 2:
    for task, pid in pids.items():
        if task in records:
            continue
        epoch = read_last(ROOT / task / "epochs.jsonl")
        if epoch["updates"] < targets[task]:
            continue
        verify_process(pid, task)
        os.kill(pid, signal.SIGSTOP)
        latest = read_last(ROOT / task / "updates.jsonl")
        if latest["update"] != epoch["updates"]:
            os.kill(pid, signal.SIGCONT)
            targets[task] = epoch["updates"] + 548
            print("MISSED_BOUNDARY", task, "wait for", targets[task], flush=True)
            continue
        path = ROOT / task / "last.pt"
        state = torch.load(path, map_location="cpu", weights_only=False)
        if state["updates"] != epoch["updates"] or any(k.startswith("module.") for k in state["model_state_dict"]):
            raise RuntimeError("Unsafe checkpoint or prefixed model state")
        archive = ROOT / task / f"migration_before_dual_{state['updates']}.pt"
        shutil.copy2(path, archive)
        best_archive = ROOT / task / f"migration_best_before_dual_{state['updates']}.pt"
        shutil.copy2(ROOT / task / "best.pt", best_archive)
        records[task] = dict(pid=pid, updates=state["updates"], epoch=state["epoch"],
                             checkpoint=str(archive), sha256=digest(archive), best_checkpoint=str(best_archive),
                             best_sha256=digest(best_archive), best=state["best"], best_epoch=state["best_epoch"],
                             cursor=dict(next_epoch=state["epoch"] + 1, next_batch_index=0),
                             model_optimizer_shuffle_torch_cuda_rng_saved=True, uncheckpointed_updates_lost=0)
        del state
        print("SAFE_CHECKPOINT", task, records[task], flush=True)
    time.sleep(0.05)
manifest = dict(records=records, original_protocol=json.loads((ROOT / "protocol.json").read_text()),
                method="SIGSTOP after epoch checkpoint and before the next completed optimizer update")
tmp = ROOT / "migration_handoff.json.tmp"
tmp.write_text(json.dumps(manifest, indent=2) + "\n")
os.replace(tmp, ROOT / "migration_handoff.json")
for task, pid in pids.items():
    verify_process(pid, task)
    os.kill(pid, signal.SIGTERM)
    os.kill(pid, signal.SIGCONT)
print("OWNED_WORKERS_TERMINATED_SAFE", flush=True)
