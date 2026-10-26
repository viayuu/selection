"""Explicit user-requested resource pause; never signal an unrelated process."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import time


ROOT = Path(__file__).resolve().parent / 'runs/R113_executable_process'
CONTAINER = '76045731f6b453443fe18a9bce671599bdbf17c2713b8edcdd090925eea14bf6'


def read(path):
    return json.loads(Path(path).read_text())


def command(pid):
    try:
        return (Path('/proc') / str(pid) / 'cmdline').read_bytes().split(b'\0')
    except FileNotFoundError:
        return []


def alive(pid):
    try:
        value = (Path('/proc') / str(pid) / 'stat').read_text()
        return value.rsplit(')', 1)[1].strip().split()[0] != 'Z'
    except FileNotFoundError:
        return False


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(part)
    return value.hexdigest()


def stop_4090(root):
    if (socket.gethostname() != CONTAINER[:12] or CONTAINER not in Path('/proc/self/cgroup').read_text()
            or os.getuid() != 2022 or os.environ.get('SLURM_JOB_ID') != '1475'):
        raise RuntimeError('Cleanup must execute inside the original authorized allocation container')
    dispatch = root / 'preparation/wave_dispatch'
    controller = read(dispatch / 'controller.json')
    destination = root / 'preparation/migration_3090_20261026/stop_4090_receipt.json'
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError('Preserve the previous explicit resource-pause receipt')
    receipt = dict(reason='User now authorizes only two RTX3090 GPUs', host=socket.gethostname(),
        execution_cgroup=Path('/proc/self/cgroup').read_text(), source_dispatch=str(dispatch),
        training_restart=False, fresh_solver_calls=0, signals=[], complete=False)
    for arm in ('B_snapshot', 'C_transition'):
        job = controller['jobs'][arm]
        worker = read(Path(job['attempt_dir']) / 'worker.json')
        stage = worker['stages'][-1]
        if stage['name'] != 'train' or worker.get('job') != arm or job.get('allocation') != 1475:
            raise ValueError('Only the two known R113 selector-training children may be interrupted')
        pid = stage['pid']
        args = command(pid)
        record = dict(arm=arm, pid=pid, source_worker=str(Path(job['attempt_dir']) / 'worker.json'),
            observed_status=read(root / arm / 'status.json'), already_exited=not alive(pid))
        if args and alive(pid):
            if (b'code.V4.r113_experiment' not in args or b'train' not in args
                    or b'--arm' not in args or args[args.index(b'--arm') + 1] != arm.encode()):
                raise RuntimeError('PID identity changed; refusing to signal ' + str(pid))
            os.kill(pid, signal.SIGINT)
            record['first_signal'] = 'SIGINT'
        receipt['signals'].append(record)
    deadline = time.monotonic() + 45
    while any(alive(row['pid']) for row in receipt['signals']) and time.monotonic() < deadline:
        time.sleep(.5)
    for row in receipt['signals']:
        if alive(row['pid']):
            args = command(row['pid'])
            if b'code.V4.r113_experiment' not in args or row['arm'].encode() not in args:
                raise RuntimeError('PID identity changed during cleanup')
            os.kill(row['pid'], signal.SIGTERM)
            row['fallback_signal'] = 'SIGTERM'
    deadline = time.monotonic() + 15
    while any(alive(row['pid']) for row in receipt['signals']) and time.monotonic() < deadline:
        time.sleep(.5)
    import torch
    for row in receipt['signals']:
        row['exited'] = not alive(row['pid'])
        path = root / row['arm'] / 'last.pt'
        saved = torch.load(path, map_location='cpu', weights_only=False)
        row.update(checkpoint=str(path), checkpoint_sha256=digest(path),
            checkpoint_update=saved['successful_updates'], state_hash=saved['state_hash'],
            historical_gpu_type=saved['args'].get('gpu_type'),
            optimizer_retained='optimizer' in saved, scaler_retained='scaler' in saved,
            rng_retained='rng' in saved, stream_retained='stream_draws' in saved,
            unpersisted_updates_at_signal=max(0, row['observed_status'].get('successful_updates', 0)
                                             - saved['successful_updates']))
    receipt['complete'] = all(row['exited'] for row in receipt['signals'])
    with destination.open('x') as stream:
        json.dump(receipt, stream, indent=2, allow_nan=False)
    print(json.dumps(receipt, indent=2, allow_nan=False))
    if not receipt['complete']:
        raise RuntimeError('A known child did not exit; no unrelated process was signalled')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('stop-4090',))
    parser.add_argument('--root', type=Path, default=ROOT)
    args = parser.parse_args()
    stop_4090(args.root.resolve())


if __name__ == '__main__':
    main()
