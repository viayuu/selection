"""Watch only this R39 queue; telemetry and quality checks remain on disk."""

import argparse
import csv
import json
import subprocess
import time
from datetime import datetime
from pathlib import Path


def load(path):
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('code/V4/runs/R39_performance_model'))
    parser.add_argument('--interval', type=float, default=30)
    args = parser.parse_args()
    fields = ('timestamp', 'gpu_util_pct', 'memory_util_pct', 'memory_mib', 'power_w', 'completed',
              'run', 'epoch', 'updates', 'top1', 'vs_sbs_pct', 'classification_ce', 'performance_mse')
    path = args.root / 'gpu_telemetry.csv'
    with path.open('a', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        if path.stat().st_size == 0:
            writer.writeheader()
        while True:
            rows = sorted(args.root.glob('*_seed*/result.json'))
            active = [p for p in args.root.glob('*_seed*') if (p/'train.log').exists() and not (p/'result.json').exists()]
            current = max(active, key=lambda p: (p/'train.log').stat().st_mtime) if active else None
            history = load(current/'history.json') if current is not None else None
            last = history[-1] if history else {}
            macro = last.get('val', {})
            gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=utilization.gpu,utilization.memory,memory.used,power.draw',
                                           '--format=csv,noheader,nounits'], text=True).strip().split(',')
            record = dict(timestamp=datetime.now().isoformat(timespec='seconds'), gpu_util_pct=float(gpu[0]),
                          memory_util_pct=float(gpu[1]), memory_mib=float(gpu[2]), power_w=float(gpu[3]),
                          completed=len(rows), run=current.name if current is not None else '-', epoch=last.get('epoch', 0),
                          updates=last.get('updates', 0), top1=macro.get('macro_top1', ''),
                          vs_sbs_pct=macro.get('macro_vs_sbs_pct', ''), classification_ce=macro.get('macro_ce', ''),
                          performance_mse=macro.get('macro_performance_mse', ''))
            writer.writerow(record)
            stream.flush()
            print('[watch] ' + json.dumps(record), flush=True)
            exit_file = args.root/'queue.exit_code'
            if exit_file.exists():
                code = int(exit_file.read_text().strip())
                print(f'[queue exit] code={code} completed={len(rows)}/6', flush=True)
                return
            time.sleep(args.interval)


if __name__ == '__main__':
    main()
