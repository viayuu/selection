#!/usr/bin/env python
import csv
import json
import re
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
BRIDGE_ROOT = SCRIPT_DIR.parent
if str(BRIDGE_ROOT) not in sys.path:
    sys.path.insert(0, str(BRIDGE_ROOT))

from config import (
    ACTIVE_EXPERIMENT_GROUPS,
    LOG_ROOT,
    MANIFEST_ROOT,
    NSS_PROGRESS_EXPECTED,
    RESULTS_ROOT,
)


NSS_STATUS_CSV = RESULTS_ROOT / "nss_eval" / "运行清单.csv"
MONITOR_JSON = LOG_ROOT / "monitor_snapshot.json"
MONITOR_MD = LOG_ROOT / "MONITOR_SUMMARY.md"
MONITOR_HISTORY = LOG_ROOT / "monitor_history.jsonl"

ERROR_PATTERNS = (
    "Traceback",
    "CUDA error",
    "AssertionError",
    "RuntimeError",
    "device-side assert",
    "failed",
)

PROCESS_TOKENS = (
    "run_nss_eval_suite.py",
    "run_label_queue.py",
    "eval.py",
)


def load_jobs() -> List[Dict]:
    with (MANIFEST_ROOT / "label_jobs.json").open("r", encoding="utf-8") as f:
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
    info = {
        "available": False,
        "raw": "",
        "apps": "",
        "cached": False,
    }
    rc, out = run_command_retry(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.used,memory.total,utilization.gpu",
            "--format=csv,noheader,nounits",
        ]
    )
    info["raw"] = out.strip()
    info["available"] = rc == 0
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
            if not line or "monitor_v3.py" in line or "codex-linux-sandbox" in line:
                continue
            if any(token in line for token in PROCESS_TOKENS):
                matches.append(line)

    if not matches:
        for pattern in ("run_label_queue.py", "run_nss_eval_suite.py", "eval.py"):
            rc_pgrep, out_pgrep = run_command(["pgrep", "-af", pattern])
            if rc_pgrep != 0:
                continue
            for raw_line in out_pgrep.splitlines():
                line = raw_line.strip()
                if not line or "monitor_v3.py" in line or "codex-linux-sandbox" in line or "pgrep -af" in line:
                    continue
                if any(token in line for token in PROCESS_TOKENS):
                    matches.append(line)

    deduped = []
    seen = set()
    for line in matches:
        if line in seen:
            continue
        seen.add(line)
        deduped.append(line)
    return "\n".join(deduped[:20])


def latest_activity() -> Dict:
    activity = {}
    roots = {
        "nss_eval": RESULTS_ROOT / "nss_eval",
        "label_v3": RESULTS_ROOT / "label_v3",
    }
    for name, root in roots.items():
        if not root.exists():
            continue

        latest_path = None
        latest_mtime = None
        for pattern in ("instance_results.jsonl", "eval.log"):
            for path in root.rglob(pattern):
                try:
                    mtime = path.stat().st_mtime
                except OSError:
                    continue
                if latest_mtime is None or mtime > latest_mtime:
                    latest_mtime = mtime
                    latest_path = path

        if latest_path is not None and latest_mtime is not None:
            activity[name] = {
                "timestamp": datetime.fromtimestamp(latest_mtime).isoformat(timespec="seconds"),
                "path": str(latest_path),
            }
    return activity


def nss_progress() -> Dict:
    progress = {}
    if not NSS_STATUS_CSV.is_file():
        return progress

    latest = {}
    with NSS_STATUS_CSV.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            dataset = row["数据集"]
            if dataset not in NSS_PROGRESS_EXPECTED:
                continue
            key = (row["方法"], row["问题"], dataset)
            latest[key] = row

    grouped = defaultdict(lambda: {"completed": 0, "failed": 0, "running": 0, "expected": 0})
    for (_, _, dataset), row in latest.items():
        grouped[dataset]["expected"] = NSS_PROGRESS_EXPECTED[dataset]
        status = row["状态"]
        if status == "完成":
            grouped[dataset]["completed"] += 1
        elif status == "失败":
            grouped[dataset]["failed"] += 1
        else:
            grouped[dataset]["running"] += 1
    return grouped


def non_nss_progress(jobs: List[Dict]) -> Dict:
    grouped = defaultdict(lambda: {"done": 0, "total": 0})
    for job in jobs:
        if job["group"] == "nss":
            continue
        grouped[job["group"]]["total"] += 1
        expected_output = job.get("expected_output")
        expected_lines = job.get("expected_lines")
        if expected_output and expected_lines is not None:
            if count_jsonl_lines(Path(expected_output)) == int(expected_lines):
                grouped[job["group"]]["done"] += 1
    return grouped


def scan_recent_errors() -> List[Dict]:
    findings = []
    candidates = []
    for path in sorted(LOG_ROOT.glob("*.log"), key=lambda p: p.stat().st_mtime, reverse=True):
        if "monitor" in path.name or "chat_status" in path.name:
            continue
        candidates.append(path)
        if len(candidates) >= 20:
            break
    nss_eval = RESULTS_ROOT / "nss_eval"
    if nss_eval.exists():
        candidates.extend(sorted(nss_eval.rglob("eval.log"), key=lambda p: p.stat().st_mtime, reverse=True)[:20])

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
            findings.append(
                {
                    "path": str(path),
                    "matched": matched[:5],
                }
            )
    return findings[:10]


def build_snapshot() -> Dict:
    jobs = load_jobs()
    snapshot = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "gpu": gpu_status(),
        "running_processes": running_processes(),
        "latest_activity": latest_activity(),
        "nss": nss_progress(),
        "other_groups": non_nss_progress(jobs),
        "recent_errors": scan_recent_errors(),
    }
    return snapshot


def write_snapshot(snapshot: Dict) -> None:
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    with MONITOR_JSON.open("w", encoding="utf-8") as f:
        json.dump(snapshot, f, ensure_ascii=False, indent=2)
    with MONITOR_HISTORY.open("a", encoding="utf-8") as f:
        f.write(json.dumps(snapshot, ensure_ascii=False) + "\n")

    lines = [
        "# V3 Monitor Summary",
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
    nss = snapshot["nss"]
    if "nss" in ACTIVE_EXPERIMENT_GROUPS and nss:
        for dataset, meta in sorted(nss.items()):
            lines.append(
                f"- NSS `{dataset}`: completed `{meta['completed']}/{meta['expected']}`, failed `{meta['failed']}`"
            )
    for group, meta in sorted(snapshot["other_groups"].items()):
        if group in ACTIVE_EXPERIMENT_GROUPS:
            lines.append(f"- `{group}`: done `{meta['done']}/{meta['total']}`")

    lines.extend(["", "## Running", "```text", snapshot["running_processes"] or "(none)", "```", ""])

    lines.append("## Latest Activity")
    if snapshot["latest_activity"]:
        for name, meta in sorted(snapshot["latest_activity"].items()):
            if name == "nss_eval" and "nss" not in ACTIVE_EXPERIMENT_GROUPS:
                continue
            lines.append(f"- `{name}` latest write: `{meta['timestamp']}`")
            lines.append(f"  path: `{meta['path']}`")
    else:
        lines.append("- none")
    lines.append("")

    lines.append("## Recent Errors")
    if snapshot["recent_errors"]:
        for item in snapshot["recent_errors"]:
            lines.append(f"- `{item['path']}`")
            lines.append("  matches: " + ", ".join(item["matched"]))
    else:
        lines.append("- none")

    MONITOR_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    snapshot = build_snapshot()
    write_snapshot(snapshot)
    print(json.dumps(snapshot, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
