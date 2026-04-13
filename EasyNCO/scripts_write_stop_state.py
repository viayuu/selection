import json
import os
import sys
import time


def parse_pid_arg(raw: str):
    if not raw.strip():
        return []
    return [int(item) for item in raw.split()]


def main():
    if len(sys.argv) != 4:
        raise SystemExit("usage: scripts_write_stop_state.py STATE_FILE CONTROLLER_PIDS WORKER_PIDS")

    state_file = sys.argv[1]
    controller_pids = parse_pid_arg(sys.argv[2])
    worker_pids = parse_pid_arg(sys.argv[3])

    os.makedirs(os.path.dirname(state_file), exist_ok=True)
    state = {
        "time": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime()),
        "controller_pids": controller_pids,
        "worker_pids": worker_pids,
        "mode": "stopped_restart_from_saved_progress",
        "resume_command": "/public/home/zhoucl/shiys/EasyNCO/resume_nss_eval.sh",
    }
    with open(state_file, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
