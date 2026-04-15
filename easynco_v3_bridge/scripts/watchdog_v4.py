#!/usr/bin/env python
import json
import os
import shlex
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
BRIDGE_ROOT = SCRIPT_DIR.parent
if str(BRIDGE_ROOT) not in sys.path:
    sys.path.insert(0, str(BRIDGE_ROOT))

from config import LOG_ROOT, MANIFEST_ROOT, PYTHONPATH_ROOT
from monitor_v4 import build_snapshot, write_snapshot


WATCHDOG_HISTORY = LOG_ROOT / "watchdog_v4_history.jsonl"
PID_ROOT = LOG_ROOT / "pids_v4"
BUILD_SCRIPT = BRIDGE_ROOT / "scripts" / "build_label_jobs_v4.py"
RUN_QUEUE_SCRIPT = BRIDGE_ROOT / "scripts" / "run_label_queue.py"
ENV_PYTHON = "/public/home/zhoucl/anaconda3/envs/easynco_zhoucl/bin/python"


QUEUE_LAYOUT = (
    ("v4_atsp_matnet", "label_jobs_v4_atsp_matnet.json", "0"),
    ("v4_atsp_matpoenet", "label_jobs_v4_atsp_matpoenet.json", "1"),
    ("v4_atsp_glop", "label_jobs_v4_atsp_glop.json", "1"),
    ("v4_mvrp_mtpomo_part1", "label_jobs_v4_mvrp_mtpomo_part1.json", "0"),
    ("v4_mvrp_mtpomo_part2", "label_jobs_v4_mvrp_mtpomo_part2.json", "1"),
    ("v4_mvrp_mvmoe_part1", "label_jobs_v4_mvrp_mvmoe_part1.json", "0"),
    ("v4_mvrp_mvmoe_part2", "label_jobs_v4_mvrp_mvmoe_part2.json", "1"),
)


def append_watchdog_event(payload):
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    with WATCHDOG_HISTORY.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def count_jsonl_lines(path: Path) -> int:
    if not path.is_file():
        return 0
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def jobs_pending_from_manifest(jobs_path: Path) -> bool:
    if not jobs_path.is_file():
        return False
    try:
        jobs = json.loads(jobs_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    for job in jobs:
        expected_output = job.get("expected_output")
        expected_lines = int(job.get("expected_lines", 0))
        if not expected_output or count_jsonl_lines(Path(expected_output)) != expected_lines:
            return True
    return False


def pid_path(session_name: str) -> Path:
    return PID_ROOT / f"{session_name}.pid"


def process_running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def session_running(session_name: str) -> bool:
    path = pid_path(session_name)
    if not path.is_file():
        return False
    try:
        pid = int(path.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return False
    return process_running(pid)


def start_queue(session_name: str, jobs_path: Path, gpu_id: str):
    PID_ROOT.mkdir(parents=True, exist_ok=True)
    log_path = LOG_ROOT / f"{session_name}.log"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PYTHONPATH_ROOT)
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    cmd = [
        ENV_PYTHON,
        str(RUN_QUEUE_SCRIPT),
        "--jobs",
        str(jobs_path),
        "--continue-on-error",
    ]
    with log_path.open("ab") as f:
        f.write(f"\n[{datetime.now().isoformat(timespec='seconds')}] START {' '.join(map(shlex.quote, cmd))}\n".encode("utf-8"))
        process = subprocess.Popen(
            cmd,
            cwd=str(PYTHONPATH_ROOT),
            env=env,
            stdout=f,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    pid_path(session_name).write_text(str(process.pid), encoding="utf-8")
    return process.pid


def ensure_manifests():
    subprocess.run([ENV_PYTHON, str(BUILD_SCRIPT)], cwd=str(PYTHONPATH_ROOT), check=True)


def maybe_start_queues():
    actions = []
    ensure_manifests()
    for session_name, manifest_name, gpu_id in QUEUE_LAYOUT:
        jobs_path = MANIFEST_ROOT / manifest_name
        if not jobs_pending_from_manifest(jobs_path):
            actions.append({"action": "skip_complete", "session": session_name, "jobs_path": str(jobs_path)})
            continue
        if session_running(session_name):
            actions.append({"action": "already_running", "session": session_name})
            continue
        pid = start_queue(session_name, jobs_path, gpu_id)
        actions.append(
            {
                "action": "start_queue",
                "session": session_name,
                "jobs_path": str(jobs_path),
                "gpu": gpu_id,
                "pid": pid,
            }
        )
    return actions


def loop(interval_seconds: int):
    while True:
        snapshot = build_snapshot()
        write_snapshot(snapshot)
        actions = maybe_start_queues()
        event = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "actions": actions,
            "groups": snapshot.get("groups", {}),
            "gpu": snapshot.get("gpu", {}),
        }
        append_watchdog_event(event)
        print(json.dumps(event, ensure_ascii=False), flush=True)
        time.sleep(interval_seconds)


def main():
    interval_seconds = 60
    if len(sys.argv) >= 2:
        interval_seconds = int(sys.argv[1])
    loop(interval_seconds)


if __name__ == "__main__":
    main()
