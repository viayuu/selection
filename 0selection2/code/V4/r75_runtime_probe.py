"""Bounded read-only process/IO sampling and R75 input RAM sizing."""

import argparse
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


UUID = 'GPU-cb626bdb-4319-2f5e-b91f-17fcaa44a83d'


def pairs(path):
    result = {}
    for line in path.read_text().splitlines():
        fields = line.split(None, 1)
        if len(fields) == 2:
            try:
                result[fields[0].rstrip(':')] = int(fields[1].split()[0])
            except ValueError:
                continue
    return result


def process(pid):
    root = Path('/proc') / str(pid)
    fields = (root / 'stat').read_text().rpartition(')')[2].split()
    return dict(cpu_ticks=int(fields[11]) + int(fields[12]),
                minor_faults=int(fields[7]), major_faults=int(fields[9]),
                io=pairs(root / 'io'), status={key: value for key, value in
                    pairs(root / 'status').items() if key in ('VmRSS', 'RssAnon', 'RssFile', 'VmLck')})


def memory_limits(pid):
    lines = (Path('/proc') / str(pid) / 'cgroup').read_text().splitlines()
    result = {}
    for line in lines:
        _, controllers, relative = line.split(':', 2)
        if 'memory' in controllers.split(','):
            root = Path('/sys/fs/cgroup/memory') / relative.lstrip('/')
            for name in ('memory.limit_in_bytes', 'memory.usage_in_bytes', 'memory.stat'):
                path = root / name
                if path.exists():
                    result[str(path)] = pairs(path) if name.endswith('stat') else int(path.read_text())
    return result


def input_sizes(cache):
    result = []
    for split in ('train', 'val'):
        for directory in sorted((cache / split).iterdir()):
            record = json.loads((directory / 'metadata.json').read_text())
            n = np.load(directory / 'n.npy', mmap_mode='r', allow_pickle=False)
            specs = record['arrays']
            dense = sum(int(np.prod(spec['shape'])) * np.dtype(spec['dtype']).itemsize
                        for spec in specs.values())
            scalar = sum(int(np.prod(spec['shape'])) * np.dtype(spec['dtype']).itemsize
                         for name, spec in specs.items() if name not in ('node', 'node_geom', 'matrix'))
            if record['kind'] == 'coord':
                width = specs['node']['shape'][-1] + specs['node_geom']['shape'][-1]
                packed = int(n.sum(dtype=np.int64)) * width * 4
            else:
                packed = int(np.square(n).sum(dtype=np.int64)) * 4
            result.append(dict(split=split, problem=record['problem'], rows=len(n),
                               dense_bytes=dense, packed_input_bytes=packed + scalar,
                               offset_bytes=(len(n) + 1) * 8))
            n._mmap.close()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pid', type=int, required=True)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--seconds', type=int, default=12)
    args = parser.parse_args()
    protocol = json.loads((args.root / 'protocol.json').read_text())
    started = time.monotonic()
    initial, samples = process(args.pid), []
    for _ in range(args.seconds):
        value = subprocess.run(['nvidia-smi', '-i', UUID,
            '--query-gpu=uuid,utilization.gpu,memory.used,memory.total',
            '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=5)
        samples.append(dict(seconds=time.monotonic() - started, output=value.stdout.strip(),
                            returncode=value.returncode))
        time.sleep(1)
    final = process(args.pid)
    elapsed = time.monotonic() - started
    sizes = input_sizes(Path(protocol['cache_root']))
    report = dict(utc=datetime.now(timezone.utc).isoformat(), pid=args.pid, uuid=UUID,
        elapsed_seconds=elapsed, cpu_core_equivalents=(final['cpu_ticks'] - initial['cpu_ticks']) /
            os.sysconf('SC_CLK_TCK') / elapsed,
        minor_faults=final['minor_faults'] - initial['minor_faults'],
        major_faults=final['major_faults'] - initial['major_faults'],
        io_delta={key: final['io'][key] - initial['io'][key] for key in initial['io']},
        process_status=final['status'], cgroup_memory=memory_limits(args.pid),
        gpu_samples=samples, input_sizes=sizes,
        packed_total_bytes=sum(v['packed_input_bytes'] + v['offset_bytes'] for v in sizes),
        dense_total_bytes=sum(v['dense_bytes'] for v in sizes),
        new_model_forwards=0, new_optimizer_updates=0, data_modified=False)
    path = args.root / 'profile_io_before.json'
    path.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
