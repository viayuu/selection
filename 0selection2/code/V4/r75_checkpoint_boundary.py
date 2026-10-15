"""Quiesce only the owned R75 trainer after a logged durable checkpoint."""

import argparse
import fcntl
import hashlib
import json
import os
import shutil
import signal
import time
from datetime import datetime, timezone
from pathlib import Path


PID = 1632292


def utc():
    return datetime.now(timezone.utc).isoformat()


def write(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def latest(path):
    with path.open('rb') as stream:
        offset = max(0, os.fstat(stream.fileno()).st_size - 65536)
        stream.seek(offset)
        data = stream.read()
    if offset:
        data = data.partition(b'\n')[2]
    lines = data.split(b'\n')[:-1]
    return json.loads(lines[-1])


def complete_tail(path):
    with path.open('rb') as stream:
        stream.seek(-1, os.SEEK_END)
        return stream.read(1) == b'\n'


def identity():
    root = Path('/proc') / str(PID)
    environment = dict(item.split(b'=', 1) for item in
        (root / 'environ').read_bytes().split(b'\0') if b'=' in item)
    if (environment.get(b'SLURM_JOB_ID') != b'1465'
            or environment.get(b'SLURM_STEP_ID') not in (b'121', b'1465.121')
            or b'code.V4.r75_experiment' not in (root / 'cmdline').read_bytes()):
        raise RuntimeError('Refusing to signal a different process or Slurm step')
    fields = (root / 'stat').read_text().rpartition(')')[2].split()
    return fields[0], fields[19]


def interrupted(signum, frame):
    raise InterruptedError('Boundary controller interrupted')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--timeout', type=int, default=300)
    args = parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        parser.error('Boundary controller must have CUDA explicitly hidden')
    root = args.root.resolve()
    directory = root / 'runtime_io_continuation1'
    directory.mkdir(exist_ok=True)
    with (directory / 'boundary.lock').open('w') as guard:
        fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (directory / 'quiesced.json').exists():
            raise RuntimeError('A boundary was already captured; no repeated signalling')
        _, birth = identity()
        updates = root / 'single_seed2' / 'updates.jsonl'
        first = latest(updates)['successful_update']
        deadline, stopped, success, attempts = time.monotonic() + args.timeout, False, False, []
        signal.signal(signal.SIGTERM, interrupted)
        try:
            while time.monotonic() < deadline:
                row = latest(updates)
                update = row['successful_update']
                if update <= first or update % 60:
                    time.sleep(.025)
                    continue
                if identity()[1] != birth:
                    raise RuntimeError('Trainer PID was replaced')
                os.kill(PID, signal.SIGSTOP)
                stopped = True
                for _ in range(2000):
                    state, current_birth = identity()
                    if current_birth != birth:
                        raise RuntimeError('Trainer PID was replaced during quiescence')
                    if state in ('T', 't'):
                        break
                    time.sleep(.01)
                else:
                    raise RuntimeError('Trainer did not stop')
                # Import after stopping so import latency cannot miss the boundary.
                import torch

                checkpoint = root / 'single_seed2' / 'last.pt'
                saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
                committed = latest(updates)
                attempt = dict(utc=utc(), observed_update=update,
                    logged_update=committed['successful_update'],
                    checkpoint_update=saved['successful_updates'], complete_log_tail=complete_tail(updates))
                attempts.append(attempt)
                if (not attempt['complete_log_tail']
                        or committed['successful_update'] != saved['successful_updates']
                        or committed['presentations'] != saved['presentations']
                        or committed['epoch'] != saved['epoch_index'] + 1
                        or committed['update_in_epoch'] != saved['next_step']
                        or not 0 < saved['next_step'] < saved['args']['updates_per_epoch']):
                    os.kill(PID, signal.SIGCONT)
                    stopped = False
                    first = committed['successful_update']
                    time.sleep(.05)
                    continue
                config = json.loads((root / 'protocol.json').read_text())
                if (saved['args'] != config or saved['parameter_mode'] != 'averaged x'
                        or any(key not in saved for key in ('model', 'optimizer', 'scaler', 'rng',
                                                           'epoch_index', 'next_step', 'totals', 'per_task'))):
                    raise RuntimeError('Durable state is not the exact resumable R75 checkpoint')
                shutil.copy2(checkpoint, directory / 'last_at_boundary.pt')
                for name in ('formal_started.json', 'status.json', 'monitor_status.json'):
                    source = root / name
                    if source.exists():
                        shutil.copy2(source, directory / ('prior_' + name))
                record = dict(utc=utc(), target_pid=PID, target_step='1465.121',
                    target_state='SIGSTOP', checkpoint=str(checkpoint),
                    checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                    successful_updates=saved['successful_updates'], presentations=saved['presentations'],
                    epoch_index=saved['epoch_index'], next_step=saved['next_step'],
                    current_epoch_saved_step_seconds=saved['totals'].get('seconds', 0.),
                    parameter_mode=saved['parameter_mode'], committed_record=committed, attempts=attempts,
                    existing_logged_updates_replayed=0,
                    possibly_discarded_inflight_groups_upper_bound=1,
                    note='No logged successful update is lost or replayed. A next uncommitted in-flight group may be discarded and retried from saved RNG.',
                    cuda_visible_devices='', new_optimizer_updates=0)
                write(directory / 'quiesced.json', record)
                status = json.loads((root / 'status.json').read_text())
                status.update(phase='paused_for_input_optimization', heartbeat_utc=utc(),
                              heartbeat_unix=time.time(), durable_update=saved['successful_updates'])
                write(root / 'status.json', status)
                success = True
                print(json.dumps(record, indent=2), flush=True)
                return
            raise TimeoutError('No safe durable boundary captured; original trainer remains running')
        finally:
            if stopped and not success:
                os.kill(PID, signal.SIGCONT)


if __name__ == '__main__':
    main()
