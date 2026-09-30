"""Record R40 GPU telemetry and keep following both owned experiments."""

import argparse
import csv
import json
import subprocess
import time
from datetime import datetime
from pathlib import Path

from .r40_experiment import ROOT, RUN_NAMES


def load(path):
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--interval', type=float, default=30)
    args = parser.parse_args()
    fields = ('timestamp', 'gpu_util_pct', 'memory_mib', 'power_w', 'completed', 'group', 'epoch', 'updates', 'ce', 'top1', 'vs_sbs_pct')
    path = args.root / 'gpu_telemetry.csv'
    with path.open('a', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        if path.stat().st_size == 0:
            writer.writeheader()
        while True:
            gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,power.draw',
                                           '--format=csv,noheader,nounits'], text=True).strip().split(',')
            completed = sum((args.root / name / 'result.json').exists() for name in RUN_NAMES.values())
            for group, name in RUN_NAMES.items():
                history = load(args.root / name / 'history.json')
                last = history[-1] if history else {}
                macro = last.get('val', {})
                row = dict(timestamp=datetime.now().isoformat(timespec='seconds'), gpu_util_pct=float(gpu[0]), memory_mib=float(gpu[1]),
                           power_w=float(gpu[2]), completed=completed, group=group, epoch=last.get('epoch', 0), updates=last.get('updates', 0),
                           ce=macro.get('macro_ce', ''), top1=macro.get('macro_top1', ''), vs_sbs_pct=macro.get('macro_vs_sbs_pct', ''))
                writer.writerow(row)
                print('[watch] ' + json.dumps(row), flush=True)
            stream.flush()
            exit_file = args.root / 'queue.exit_code'
            if exit_file.exists():
                print(f'[queue exit] code={exit_file.read_text().strip()} completed={completed}/2', flush=True)
                return
            time.sleep(args.interval)


if __name__ == '__main__':
    main()
