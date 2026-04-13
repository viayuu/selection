import json
import os
import signal
import sys


def try_resume(pid: int):
    try:
        os.kill(pid, signal.SIGCONT)
        return True
    except ProcessLookupError:
        return False


def main():
    if len(sys.argv) != 2:
        raise SystemExit("usage: scripts_resume_from_pause_state.py STATE_FILE")

    state_file = sys.argv[1]
    with open(state_file, "r", encoding="utf-8") as f:
        state = json.load(f)

    resumed = []
    missing = []
    for pid in state.get("worker_pids", []) + state.get("controller_pids", []):
        if try_resume(int(pid)):
            resumed.append(int(pid))
        else:
            missing.append(int(pid))

    print(json.dumps({"resumed": resumed, "missing": missing}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
