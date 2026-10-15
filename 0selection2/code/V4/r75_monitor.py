"""CPU-only R75 monitor; records meaningful changes, never restarts training."""

import argparse
import json
import time
from pathlib import Path

from .r75_capacity import ROOT, append_json, utc, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--interval', type=int, default=60)
    args = parser.parse_args()
    last_phase, last_best, last_report = None, None, 0.
    while True:
        state_path = args.root/'status.json'
        if not state_path.exists():
            time.sleep(args.interval)
            continue
        try:
            state = json.loads(state_path.read_text())
        except (OSError, ValueError):
            time.sleep(args.interval)
            continue
        phase, best = state['phase'], state.get('best_val_correct')
        cache_path = args.root/'cache_progress.json'
        cache = json.loads(cache_path.read_text()) if cache_path.exists() else None
        age = time.time()-state.get('heartbeat_unix', time.time())
        if cache and phase.startswith('preparing'):
            age = min(age, time.time()-cache_path.stat().st_mtime)
        decision = 'STOP' if phase == 'failed' else 'CONTINUE'
        if age > 1800 and phase not in ('complete', 'failed'):
            decision = 'WAIT'
        snapshot = dict(utc=utc(), phase=phase, decision=decision, heartbeat_age_seconds=age,
                        epoch=state.get('epoch'), updates=state.get('successful_updates'),
                        best_val_correct=best, latest=state.get('latest'), cache=cache,
                        automatic_restart=False, protocol_changes=False)
        write_json(args.root/'monitor_status.json', snapshot)
        if phase != last_phase or best != last_best or time.time()-last_report >= 1800:
            append_json(args.root/'monitor_events.jsonl', snapshot)
            print('[R75 MONITOR] ' + json.dumps(snapshot, ensure_ascii=False), flush=True)
            last_phase, last_best, last_report = phase, best, time.time()
        if phase in ('complete', 'failed'):
            return
        time.sleep(args.interval)


if __name__ == '__main__':
    main()
