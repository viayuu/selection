#!/usr/bin/env python
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
BRIDGE_ROOT = SCRIPT_DIR.parent
if str(BRIDGE_ROOT) not in sys.path:
    sys.path.insert(0, str(BRIDGE_ROOT))

from config import ACTIVE_EXPERIMENT_GROUPS, LOG_ROOT
from monitor_v3 import build_snapshot, write_snapshot


CHAT_STATUS_MD = LOG_ROOT / "CHAT_LIVE_STATUS.md"
CHAT_STATUS_JSON = LOG_ROOT / "chat_live_status.json"
CHAT_STATUS_HISTORY = LOG_ROOT / "chat_status_history.jsonl"


def gpu_line(snapshot):
    gpu = snapshot.get("gpu", {})
    if not gpu.get("available"):
        return "GPU: unavailable"
    raw = gpu.get("raw", "")
    first = raw.splitlines()[0] if raw else ""
    parts = [item.strip() for item in first.split(",")]
    prefix = "GPU (cached)" if gpu.get("cached") else "GPU"
    if len(parts) >= 5:
        return f"{prefix}: util {parts[-1]}%, mem {parts[2]}/{parts[3]} MiB"
    return f"{prefix}: {raw}"


def render(snapshot):
    lines = [
        "# Chat Live Status",
        "",
        f"- Timestamp: `{snapshot.get('timestamp', 'n/a')}`",
        f"- {gpu_line(snapshot)}",
    ]

    if "nss" in ACTIVE_EXPERIMENT_GROUPS:
        for dataset, meta in sorted(snapshot.get("nss", {}).items()):
            lines.append(
                f"- NSS `{dataset}`: `{meta.get('completed', 0)}/{meta.get('expected', 0)}` complete, failed `{meta.get('failed', 0)}`"
            )

    for group, meta in sorted(snapshot.get("other_groups", {}).items()):
        if group in ACTIVE_EXPERIMENT_GROUPS:
            lines.append(f"- `{group}`: `{meta.get('done', 0)}/{meta.get('total', 0)}` complete")

    latest = snapshot.get("latest_activity", {})
    if latest:
        lines.append("")
        lines.append("## Latest Writes")
        for name, meta in sorted(latest.items()):
            if name == "nss_eval" and "nss" not in ACTIVE_EXPERIMENT_GROUPS:
                continue
            lines.append(f"- `{name}`: `{meta.get('timestamp', 'n/a')}`")
            lines.append(f"  path: `{meta.get('path', 'n/a')}`")

    running = snapshot.get("running_processes", "").strip()
    lines.append("")
    lines.append("## Running")
    lines.append("```text")
    lines.append(running or "(none)")
    lines.append("```")

    errors = snapshot.get("recent_errors", [])
    lines.append("")
    lines.append("## Recent Errors")
    if errors:
        for item in errors:
            lines.append(f"- `{item.get('path', 'n/a')}` :: {', '.join(item.get('matched', []))}")
    else:
        lines.append("- none")

    return "\n".join(lines) + "\n"


def main():
    snapshot = build_snapshot()
    write_snapshot(snapshot)

    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    CHAT_STATUS_MD.write_text(render(snapshot), encoding="utf-8")
    CHAT_STATUS_JSON.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")
    with CHAT_STATUS_HISTORY.open("a", encoding="utf-8") as f:
        f.write(json.dumps(snapshot, ensure_ascii=False) + "\n")

    print(render(snapshot))


if __name__ == "__main__":
    main()
