#!/usr/bin/env python
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
BRIDGE_ROOT = SCRIPT_DIR.parent
if str(BRIDGE_ROOT) not in sys.path:
    sys.path.insert(0, str(BRIDGE_ROOT))

from config import LOG_ROOT, MANIFEST_ROOT, RESULTS_ROOT


MONITOR_JSON = LOG_ROOT / "monitor_v4_snapshot.json"
MONITOR_MD = LOG_ROOT / "MONITOR_V4_SUMMARY.md"
MONITOR_HISTORY = LOG_ROOT / "monitor_v4_history.jsonl"

ERROR_PATTERNS = (
    "Traceback",
    "CUDA error",
    "AssertionError",
    "RuntimeError",
    "device-side assert",
    "No such file",
    "ModuleNotFoundError",
    "failed",
)

PROCESS_TOKENS = (
    "run_label_queue.py",
    "eval.py",
    "watchdog_v4.py",
    "monitor_v4.py",
)


def load_jobs() -> List[Dict]:
    with (MANIFEST_ROOT / "label_jobs_v4.json").open("r", encoding="utf-8") as f:
        return json.load(f)


def count_jsonl_lines(path: Path) -> int:
    if not path.is_file():
        return 0
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def run_command(cmd: List[str]) -> Tuple[int, str]:
    completed = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    return completed.returncode, completed.stdout


def run_command_retry(cmd: List[str], attempts: int = 3, delay_seconds: float = 0.5) -> Tuple[int, str]:
    last_rc = 1
    last_out = ""
    for attempt in range(attempts):
        rc, out = run_command(cmd)
        last_rc, last_out = rc, out
        if rc == 0:
            return rc, out
        if attempt + 1 < attempts:
            time.sleep(delay_seconds)
    return last_rc, last_out


def gpu_status() -> Dict:
    info = {"available": False, "raw": "", "apps": "", "cached": False}
    rc, out = run_command_retry(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.used,memory.total,utilization.gpu",
            "--format=csv,noheader,nounits",
        ]
    )
    info["available"] = rc == 0
    info["raw"] = out.strip()
    rc_apps, out_apps = run_command_retry(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ]
    )
    info["apps"] = out_apps.strip() if rc_apps == 0 else ""
    if info["available"]:
        return info

    if MONITOR_JSON.is_file():
        try:
            previous = json.loads(MONITOR_JSON.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            previous = {}
        previous_gpu = previous.get("gpu", {})
        if previous_gpu.get("available") and previous_gpu.get("raw"):
            info["available"] = True
            info["raw"] = previous_gpu.get("raw", "")
            info["apps"] = previous_gpu.get("apps", "")
            info["cached"] = True
    return info


def running_processes() -> str:
    rc, out = run_command(["ps", "-eo", "pid,etime,cmd"])
    matches = []
    if rc == 0:
        for raw_line in out.splitlines():
            line = raw_line.strip()
            if not line or "codex-linux-sandbox" in line:
                continue
            if any(token in line for token in PROCESS_TOKENS):
                matches.append(line)
    seen = set()
    deduped = []
    for line in matches:
        if line in seen:
            continue
        seen.add(line)
        deduped.append(line)
    return "\n".join(deduped[:40])


def latest_activity() -> Dict:
    activity = {}
    label_root = RESULTS_ROOT / "label_v4"
    if not label_root.exists():
        return activity

    latest_path = None
    latest_mtime = None
    for pattern in ("instance_results.jsonl", "eval.log"):
        for path in label_root.rglob(pattern):
            try:
                mtime = path.stat().st_mtime
            except OSError:
                continue
            if latest_mtime is None or mtime > latest_mtime:
                latest_mtime = mtime
                latest_path = path
    if latest_path is not None and latest_mtime is not None:
        activity["label_v4"] = {
            "timestamp": datetime.fromtimestamp(latest_mtime).isoformat(timespec="seconds"),
            "path": str(latest_path),
        }
    return activity


def group_progress(jobs: List[Dict]) -> Dict:
    grouped = {}
    for job in jobs:
        meta = grouped.setdefault(job["group"], {"done": 0, "total": 0})
        meta["total"] += 1
        expected_output = job.get("expected_output")
        expected_lines = job.get("expected_lines")
        if expected_output and expected_lines is not None:
            if count_jsonl_lines(Path(expected_output)) == int(expected_lines):
                meta["done"] += 1
    return grouped


def scan_recent_errors() -> List[Dict]:
    findings = []
    candidates = []
    now = time.time()
    freshness_seconds = 15 * 60
    for path in sorted(LOG_ROOT.glob("*.log"), key=lambda p: p.stat().st_mtime, reverse=True):
        if (
            "monitor_v4" in path.name
            or "chat_status_v4" in path.name
            or path.name.startswith("v3_")
        ):
            continue
        try:
            if now - path.stat().st_mtime > freshness_seconds:
                continue
        except OSError:
            continue
        candidates.append(path)
        if len(candidates) >= 40:
            break
    label_root = RESULTS_ROOT / "label_v4"
    if label_root.exists():
        candidates.extend(sorted(label_root.rglob("eval.log"), key=lambda p: p.stat().st_mtime, reverse=True)[:20])

    seen = set()
    for path in candidates:
        if path in seen or not path.is_file():
            continue
        seen.add(path)
        try:
            tail = path.read_text(encoding="utf-8", errors="ignore")[-12000:]
        except OSError:
            continue
        matched = [token for token in ERROR_PATTERNS if token.lower() in tail.lower()]
        if matched:
            findings.append({"path": str(path), "matched": matched[:5]})
    return findings[:12]


def build_snapshot() -> Dict:
    jobs = load_jobs()
    return {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "gpu": gpu_status(),
        "running_processes": running_processes(),
        "groups": group_progress(jobs),
        "latest_activity": latest_activity(),
        "recent_errors": scan_recent_errors(),
    }


def write_snapshot(snapshot: Dict) -> None:
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    MONITOR_JSON.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")
    with MONITOR_HISTORY.open("a", encoding="utf-8") as f:
        f.write(json.dumps(snapshot, ensure_ascii=False) + "\n")

    lines = [
        "# V4 Monitor Summary",
        "",
        f"- Timestamp: `{snapshot['timestamp']}`",
        "",
        "## GPU",
        f"- Available: `{snapshot['gpu']['available']}`",
        f"- Query: `{snapshot['gpu']['raw'] or 'n/a'}`",
        f"- Compute apps: `{snapshot['gpu']['apps'] or 'n/a'}`",
        "",
        "## Progress",
    ]
    for group, meta in sorted(snapshot["groups"].items()):
        lines.append(f"- `{group}`: `{meta['done']}/{meta['total']}` complete")

    lines.extend(["", "## Running", "```text", snapshot["running_processes"] or "(none)", "```", ""])

    lines.append("## Latest Activity")
    if snapshot["latest_activity"]:
        for name, meta in sorted(snapshot["latest_activity"].items()):
            lines.append(f"- `{name}` latest write: `{meta['timestamp']}`")
            lines.append(f"  path: `{meta['path']}`")
    else:
        lines.append("- none")
    lines.append("")

    lines.append("## Recent Errors")
    if snapshot["recent_errors"]:
        for item in snapshot["recent_errors"]:
            lines.append(f"- `{item['path']}`")
            lines.append(f"  matches: {', '.join(item['matched'])}")
    else:
        lines.append("- none")
    lines.append("")

    MONITOR_MD.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    snapshot = build_snapshot()
    write_snapshot(snapshot)
    print(json.dumps(snapshot, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
