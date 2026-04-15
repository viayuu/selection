#!/usr/bin/env python
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
BRIDGE_ROOT = SCRIPT_DIR.parent
if str(BRIDGE_ROOT) not in sys.path:
    sys.path.insert(0, str(BRIDGE_ROOT))

from config import LOG_ROOT
from monitor_v4 import build_snapshot, write_snapshot


CHAT_STATUS_MD = LOG_ROOT / "CHAT_LIVE_STATUS_V4.md"
CHAT_STATUS_JSON = LOG_ROOT / "chat_live_status_v4.json"
CHAT_STATUS_HISTORY = LOG_ROOT / "chat_status_v4_history.jsonl"


def gpu_lines(snapshot):
    gpu = snapshot.get("gpu", {})
    if not gpu.get("available"):
        return ["GPU: unavailable"]
    raw = gpu.get("raw", "")
    lines = []
    prefix = "GPU (cached)" if gpu.get("cached") else "GPU"
    for row in raw.splitlines():
        parts = [item.strip() for item in row.split(",")]
        if len(parts) >= 5:
            lines.append(f"{prefix} {parts[0]}: util {parts[-1]}%, mem {parts[2]}/{parts[3]} MiB")
        else:
            lines.append(f"{prefix}: {row}")
    return lines or ["GPU: unavailable"]


def render(snapshot):
    lines = [
        "# Chat Live Status V4",
        "",
        f"- Timestamp: `{snapshot.get('timestamp', 'n/a')}`",
    ]
    for line in gpu_lines(snapshot):
        lines.append(f"- {line}")

    for group, meta in sorted(snapshot.get("groups", {}).items()):
        lines.append(f"- `{group}`: `{meta.get('done', 0)}/{meta.get('total', 0)}` complete")

    latest = snapshot.get("latest_activity", {})
    if latest:
        lines.append("")
        lines.append("## Latest Writes")
        for name, meta in sorted(latest.items()):
            lines.append(f"- `{name}`: `{meta.get('timestamp', 'n/a')}`")
            lines.append(f"  path: `{meta.get('path', 'n/a')}`")

    lines.append("")
    lines.append("## Running")
    lines.append("```text")
    lines.append(snapshot.get("running_processes", "").strip() or "(none)")
    lines.append("```")

    lines.append("")
    lines.append("## Recent Errors")
    errors = snapshot.get("recent_errors", [])
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
