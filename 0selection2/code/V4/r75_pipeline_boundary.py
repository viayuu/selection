"""Capture the running RAM-input trainer at its next durable R75 checkpoint."""

import argparse
import fcntl
import hashlib
import json
import os
import shutil
import signal
import time
from pathlib import Path

from .r75_checkpoint_boundary import complete_tail, interrupted, latest, utc, write


def identity(pid, step):
    root = Path('/proc') / str(pid)
    environment = dict(item.split(b'=', 1) for item in
        (root / 'environ').read_bytes().split(b'\0') if b'=' in item)
    if (environment.get(b'SLURM_JOB_ID') != b'1465'
            or environment.get(b'SLURM_STEP_ID') not in (step.encode(), ('1465.' + step).encode())
            or b'code.V4.r75_fast_runner' not in (root / 'cmdline').read_bytes()):
        raise RuntimeError('Refusing to signal a different trainer or Slurm step')
    fields = (root / 'stat').read_text().rpartition(')')[2].split()
    return fields[0], fields[19]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--pid', type=int, required=True)
    parser.add_argument('--step', required=True)
    parser.add_argument('--timeout', type=int, default=240)
    args = parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        parser.error('The boundary controller must have CUDA explicitly hidden')
    root = args.root.resolve()
    directory = root / 'runtime_pipeline_continuation2'
    directory.mkdir(exist_ok=True)
    with (directory / 'boundary.lock').open('w') as guard:
        fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (directory / 'quiesced.json').exists():
            raise RuntimeError('This continuation already has a boundary')
        _, birth = identity(args.pid, args.step)
        updates = root / 'single_seed2/updates.jsonl'
        first = latest(updates)['successful_update']
        deadline, stopped, accepted = time.monotonic() + args.timeout, False, False
        signal.signal(signal.SIGTERM, interrupted)
        try:
            while time.monotonic() < deadline:
                observed = latest(updates)['successful_update']
                if observed <= first or observed % 60:
                    time.sleep(.02)
                    continue
                if identity(args.pid, args.step)[1] != birth:
                    raise RuntimeError('Trainer PID was replaced')
                os.kill(args.pid, signal.SIGSTOP)
                stopped = True
                for _ in range(2000):
                    state, now_birth = identity(args.pid, args.step)
                    if now_birth != birth:
                        raise RuntimeError('Trainer PID changed while stopping')
                    if state in ('T', 't'):
                        break
                    time.sleep(.01)
                else:
                    raise RuntimeError('Trainer did not quiesce')
                import torch

                path = root / 'single_seed2/last.pt'
                saved = torch.load(path, map_location='cpu', weights_only=False)
                row = latest(updates)
                if (not complete_tail(updates) or row['successful_update'] != saved['successful_updates']
                        or row['presentations'] != saved['presentations']
                        or row['epoch'] != saved['epoch_index'] + 1
                        or row['update_in_epoch'] != saved['next_step']
                        or not 0 < saved['next_step'] < saved['args']['updates_per_epoch']):
                    os.kill(args.pid, signal.SIGCONT)
                    stopped = False
                    first = row['successful_update']
                    continue
                if (saved['args'] != json.loads((root / 'protocol.json').read_text())
                        or saved['parameter_mode'] != 'averaged x'):
                    raise RuntimeError('Checkpoint is not the exact resumable scientific configuration')
                shutil.copy2(path, directory / 'last_at_boundary.pt')
                for relative, name in (('formal_started.json', 'prior_formal_started.json'),
                                       ('status.json', 'prior_status.json'),
                                       ('single_seed2/history.json', 'prior_history.json')):
                    source = root / relative
                    if source.exists():
                        shutil.copy2(source, directory / name)
                record = dict(utc=utc(), target_pid=args.pid, target_step='1465.' + args.step,
                    target_state='SIGSTOP', checkpoint=str(path),
                    checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                    successful_updates=saved['successful_updates'], presentations=saved['presentations'],
                    epoch_index=saved['epoch_index'], next_step=saved['next_step'],
                    current_epoch_saved_step_seconds=saved['totals'].get('seconds', 0.),
                    parameter_mode=saved['parameter_mode'], committed_record=row,
                    existing_logged_updates_replayed=0, possibly_discarded_inflight_groups_upper_bound=1)
                write(directory / 'quiesced.json', record)
                value = json.loads((root / 'status.json').read_text())
                value.update(phase='paused_for_pipeline_optimization', heartbeat_utc=utc(),
                             heartbeat_unix=time.time(), durable_update=saved['successful_updates'])
                write(root / 'status.json', value)
                accepted = True
                print(json.dumps(record, indent=2), flush=True)
                return
            raise TimeoutError('No matching durable boundary captured; original trainer remains running')
        finally:
            if stopped and not accepted:
                os.kill(args.pid, signal.SIGCONT)


if __name__ == '__main__':
    main()
