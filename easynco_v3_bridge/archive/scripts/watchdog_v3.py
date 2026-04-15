#!/usr/bin/env python
import json
import shlex
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
BRIDGE_ROOT = SCRIPT_DIR.parent
if str(BRIDGE_ROOT) not in sys.path:
    sys.path.insert(0, str(BRIDGE_ROOT))

from config import (
    ACTIVE_EXPERIMENT_GROUPS,
    ATSP_SPECS,
    CONDA_ENV,
    DATASETS_ROOT,
    LOG_ROOT,
    MANIFEST_ROOT,
    MVRP_METHODS,
    MVRP_SPECS,
    MVRP_VARIANT_GROUPS,
    MVRP_VARIANTS,
    NSS_PROGRESS_EXPECTED,
    V3_ATSP_ROOT,
    V3_MVRP_ROOT,
    WORKSPACE_ROOT,
    variant_dir_name,
)
from monitor_v3 import build_snapshot, write_snapshot


CONDA_SH = "/public/home/zhoucl/anaconda3/etc/profile.d/conda.sh"
WATCHDOG_HISTORY = LOG_ROOT / "watchdog_history.jsonl"
AUTOSTART_GROUPS = set(ACTIVE_EXPERIMENT_GROUPS)


def run_command(cmd):
    completed = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    return completed.returncode, completed.stdout.strip()


def session_exists(name):
    rc, _ = run_command(["tmux", "has-session", "-t", name])
    return rc == 0


def gpu_utilization(snapshot):
    if not snapshot.get("gpu", {}).get("available"):
        return None
    raw = snapshot.get("gpu", {}).get("raw", "")
    if not raw:
        return None
    first_line = raw.splitlines()[0]
    parts = [item.strip() for item in first_line.split(",")]
    if len(parts) < 5:
        return None
    try:
        return int(parts[-1])
    except ValueError:
        return None


def nss_pending(snapshot):
    progress = snapshot.get("nss", {})
    for dataset_name, expected in NSS_PROGRESS_EXPECTED.items():
        meta = progress.get(dataset_name, {})
        completed = int(meta.get("completed", 0))
        failed = int(meta.get("failed", 0))
        if completed + failed < expected:
            return True
    return False


def group_pending(snapshot, group_name):
    meta = snapshot.get("other_groups", {}).get(group_name, {})
    total = int(meta.get("total", 0))
    done = int(meta.get("done", 0))
    return total > 0 and done < total


def start_session(session_name, inner_command, tmux_log_name):
    tmux_log = LOG_ROOT / tmux_log_name
    shell_parts = [
        "source",
        shlex.quote(CONDA_SH),
        "&&",
        "conda",
        "activate",
        shlex.quote(CONDA_ENV),
        "&&",
        "export",
        "PYTHONPATH=" + shlex.quote(str(WORKSPACE_ROOT)),
        "&&",
        "cd",
        shlex.quote(str(WORKSPACE_ROOT)),
        "&&",
        inner_command,
        "|&",
        "tee",
        shlex.quote(str(tmux_log)),
    ]
    shell_command = " ".join(shell_parts)
    rc, out = run_command(
        [
            "tmux",
            "new-session",
            "-d",
            "-s",
            session_name,
            "bash -lc " + shlex.quote(shell_command),
        ]
    )
    return rc == 0, out


def append_watchdog_event(payload):
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    with WATCHDOG_HISTORY.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def load_jobs():
    jobs_path = MANIFEST_ROOT / "label_jobs.json"
    if not jobs_path.is_file():
        return []
    try:
        return json.loads(jobs_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []


def count_jsonl_lines(path):
    path = Path(path)
    if not path.is_file():
        return 0
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def jobs_pending_from_manifest(jobs_path):
    jobs_path = Path(jobs_path)
    if not jobs_path.is_file():
        return False
    try:
        jobs = json.loads(jobs_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    for job in jobs:
        expected_output = job.get("expected_output")
        expected_lines = int(job.get("expected_lines", 0))
        if not expected_output or count_jsonl_lines(expected_output) != expected_lines:
            return True
    return False


def mvrp_parallel_sessions():
    sessions = []
    for method in MVRP_METHODS:
        for group_name, _variants in MVRP_VARIANT_GROUPS:
            session_name = f"v3_mvrp_{method}_{group_name}"
            jobs_path = MANIFEST_ROOT / f"mvrp_{method}_{group_name}.json"
            sessions.append((session_name, jobs_path))
    return sessions


def shard_manifest_ready(manifest_path, expected_specs):
    if not manifest_path.is_file():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False

    shards = manifest.get("shards", [])
    shard_specs = {int(item["scale"]): int(item["num_instances"]) for item in shards}
    if shard_specs != expected_specs:
        return False

    for item in shards:
        rel_path = item.get("relative_path")
        if not rel_path:
            return False
        if not (DATASETS_ROOT / rel_path).is_file():
            return False
    return True


def dense_dataset_ready(group_name):
    if group_name == "atsp":
        return shard_manifest_ready(V3_ATSP_ROOT / "manifest.json", ATSP_SPECS)
    if group_name == "mvrp":
        global_manifest = V3_MVRP_ROOT / "manifest.json"
        if not global_manifest.is_file():
            return False
        try:
            payload = json.loads(global_manifest.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return False
        variants = payload.get("variants", {})
        for variant in MVRP_VARIANTS:
            if variant not in variants:
                return False
            shard_specs = {
                int(item["scale"]): int(item["num_instances"])
                for item in variants[variant].get("shards", [])
            }
            if shard_specs != MVRP_SPECS:
                return False
            for item in variants[variant].get("shards", []):
                rel_path = item.get("relative_path")
                if not rel_path or not (DATASETS_ROOT / rel_path).is_file():
                    return False
            variant_manifest = V3_MVRP_ROOT / variant_dir_name(variant) / "manifest.json"
            if not shard_manifest_ready(variant_manifest, MVRP_SPECS):
                return False
        return True
    return False


def maybe_start_queues(snapshot):
    actions = []

    if "nss" in AUTOSTART_GROUPS and nss_pending(snapshot) and not session_exists("v3_nss_resume"):
        ok, out = start_session(
            "v3_nss_resume",
            "python /public/home/zhoucl/shiys/easynco_v3_bridge/scripts/build_label_jobs.py && "
            "python /public/home/zhoucl/shiys/easynco_v3_bridge/scripts/run_label_queue.py --groups nss --continue-on-error",
            "v3_nss_resume.tmux.log",
        )
        actions.append({"action": "start_nss", "ok": ok, "detail": out})

    if "atsp" in AUTOSTART_GROUPS and group_pending(snapshot, "atsp") and not session_exists("v3_atsp_resume"):
        if dense_dataset_ready("atsp"):
            ok, out = start_session(
                "v3_atsp_resume",
                "python /public/home/zhoucl/shiys/easynco_v3_bridge/scripts/build_label_jobs.py && "
                "python /public/home/zhoucl/shiys/easynco_v3_bridge/scripts/run_label_queue.py --groups atsp --continue-on-error",
                "v3_atsp_resume.tmux.log",
            )
            actions.append({"action": "start_atsp", "ok": ok, "detail": out})
        else:
            actions.append({"action": "skip_atsp_restart", "reason": "dense_datasets_not_ready"})

    if "mvrp" in AUTOSTART_GROUPS and group_pending(snapshot, "mvrp"):
        if dense_dataset_ready("mvrp"):
            for session_name, jobs_path in mvrp_parallel_sessions():
                if session_exists(session_name):
                    continue
                if not jobs_pending_from_manifest(jobs_path):
                    continue
                ok, out = start_session(
                    session_name,
                    "python /public/home/zhoucl/shiys/easynco_v3_bridge/scripts/build_label_jobs.py && "
                    f"python /public/home/zhoucl/shiys/easynco_v3_bridge/scripts/run_label_queue.py --jobs {jobs_path} --continue-on-error",
                    f"{session_name}.tmux.log",
                )
                actions.append(
                    {
                        "action": "start_mvrp_subset",
                        "session": session_name,
                        "jobs_path": str(jobs_path),
                        "ok": ok,
                        "detail": out,
                    }
                )
        else:
            actions.append({"action": "skip_mvrp_restart", "reason": "dense_datasets_not_ready"})

    return actions


def loop(interval_seconds):
    while True:
        snapshot = build_snapshot()
        write_snapshot(snapshot)
        actions = maybe_start_queues(snapshot)
        event = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "gpu": snapshot.get("gpu", {}),
            "running_processes": snapshot.get("running_processes", ""),
            "latest_activity": snapshot.get("latest_activity", {}),
            "actions": actions,
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
