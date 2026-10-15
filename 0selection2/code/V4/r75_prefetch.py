"""CPU-only TRAIN input read-ahead alongside the unchanged live R75 trainer."""

import argparse
import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path


TRAINER_PID = 1632292
TRAINER_JOB = '1465'
TRAINER_STEP = '121'
THREADS = 3
COPY_BUDGET = 192_000_000
PER_COPY_LIMIT = COPY_BUDGET // (2 * THREADS)


def utc():
    return datetime.now(timezone.utc).isoformat()


class TrainerGuard:
    def __init__(self, root):
        self.root = root
        self.start_time = None

    def state(self):
        if any((self.root / name).exists() for name in (
                'failure.json', 'training_complete.json', 'pipeline_complete.json')):
            return 'stop', 'trainer_finished_or_failed'
        try:
            status = json.loads((self.root / 'status.json').read_text())
        except (OSError, ValueError):
            return 'pause', 'status_unavailable'
        phase = status.get('phase', '')
        if phase in ('failed', 'complete', 'training_complete'):
            return 'stop', phase
        if status.get('pid') != TRAINER_PID:
            return 'stop', 'trainer_pid_mismatch'
        try:
            proc = Path('/proc') / str(TRAINER_PID)
            fields = (proc / 'stat').read_text().rpartition(')')[2].split()
            if fields[0] in ('Z', 'X', 'x'):
                return 'stop', 'trainer_exited'
            if self.start_time is None:
                environment = dict(entry.split(b'=', 1) for entry in
                                   (proc / 'environ').read_bytes().split(b'\0')
                                   if b'=' in entry)
                job = environment.get(b'SLURM_JOB_ID', b'').decode()
                step = environment.get(b'SLURM_STEP_ID', b'').decode()
                if job != TRAINER_JOB or step not in (TRAINER_STEP, TRAINER_JOB + '.' + TRAINER_STEP):
                    return 'stop', 'formal_step_mismatch'
                self.start_time = fields[19]
            elif self.start_time != fields[19]:
                return 'stop', 'trainer_pid_reused'
        except (OSError, ValueError, IndexError):
            return 'stop', 'local_trainer_unavailable'
        return ('training' if phase == 'training' else 'pause'), phase


def latest_update(path):
    """Only newline-terminated JSON records are committed; ignore a partial tail."""
    try:
        with path.open('rb') as stream:
            offset = max(0, os.fstat(stream.fileno()).st_size - 65536)
            stream.seek(offset)
            data = stream.read(65536)
        if offset:
            data = data.partition(b'\n')[2]
        for line in reversed(data.split(b'\n')[:-1]):
            try:
                row = json.loads(line)
                epoch, update = row['epoch'], row['update_in_epoch']
                if type(epoch) is int and type(update) is int and epoch > 0 and update > 0:
                    return epoch - 1, update
            except (ValueError, KeyError, TypeError):
                continue
    except OSError:
        pass
    return None


class TrainInputs:
    """Open only the four permitted TRAIN input array names, lazily by task."""

    def __init__(self, cache_root, sizes):
        self.root, self.sizes, self.tasks = cache_root, sizes, {}

    def get(self, problem):
        import numpy as np

        if problem not in self.tasks:
            directory = self.root / 'train' / problem
            metadata = json.loads((directory / 'metadata.json').read_text())
            if (metadata['split'] != 'train' or metadata['problem'] != problem
                    or not metadata['complete'] or metadata['rows'] != self.sizes[problem]):
                raise ValueError('Require the completed original TRAIN input cache')
            keys = ('n', 'matrix') if metadata['kind'] == 'matrix' else ('n', 'node', 'node_geom')
            arrays = {}
            self.tasks[problem] = arrays
            for key in keys:
                spec = metadata['arrays'][key]
                if spec['file'] != key + '.npy':
                    raise ValueError('Only named TRAIN input arrays may be opened')
                arrays[key] = np.load(directory / spec['file'], mmap_mode='r', allow_pickle=False)
        return self.tasks[problem]

    def close(self):
        for arrays in self.tasks.values():
            for array in arrays.values():
                array._mmap.close()


def warm_task(problem, indices, inputs, guard):
    import numpy as np

    result = dict(complete=False, input_array_touches=0, input_rows_touched=0,
                  input_copy_bytes=0, input_gather_calls=0)
    if guard.state()[0] != 'training':
        return result
    arrays = inputs.get(problem)
    maximum = int(arrays['n'][indices].max())
    result['input_array_touches'] += 1
    result['input_rows_touched'] += len(indices)
    for key, array in arrays.items():
        if key == 'n':
            continue
        view = array[:, :maximum, :maximum] if key == 'matrix' else array[:, :maximum]
        row_bytes = int(np.prod(view.shape[1:])) * array.dtype.itemsize
        rows_per_copy = PER_COPY_LIMIT // row_bytes
        if rows_per_copy < 1:
            raise ValueError('One input row exceeds the bounded prefetch copy allowance')
        # At most three workers, one copy each; allow two copies for contiguity.
        for start in range(0, len(indices), rows_per_copy):
            if guard.state()[0] != 'training':
                return result
            selected = indices[start:start + rows_per_copy]
            values = np.ascontiguousarray(view[selected])
            size = values.nbytes
            del values
            result['input_copy_bytes'] += size
            result['input_gather_calls'] += 1
            result['input_rows_touched'] += len(selected)
        result['input_array_touches'] += 1
    result['complete'] = True
    return result


class Reporter:
    def __init__(self, root, ahead):
        self.root, self.ahead = root, ahead
        self.started, self.last_write, self.last_log = utc(), 0., 0.
        self.previous = None
        self.logger = logging.getLogger('r75_prefetch')
        self.logger.setLevel(logging.INFO)
        handler = RotatingFileHandler(root / 'prefetch.log', maxBytes=131072, backupCount=1)
        handler.setFormatter(logging.Formatter('%(message)s'))
        self.logger.addHandler(handler)

    def write(self, phase, trainer_phase, counters, epoch=None, cursor=None, warmed=0, error=None):
        now = time.monotonic()
        changed = (phase, trainer_phase) != self.previous
        if not changed and now - self.last_write < 5:
            return
        record = dict(phase=phase, trainer_phase=trainer_phase, heartbeatUTC=utc(),
                      startedUTC=self.started, pid=os.getpid(), trainer_pid=TRAINER_PID,
                      trainer_step=TRAINER_JOB + '.' + TRAINER_STEP,
                      ahead=self.ahead, threads=THREADS, copy_budget_bytes=COPY_BUDGET,
                      epoch_index=epoch, next_group_index=cursor, warmed_groups_in_window=warmed,
                      cpu_only=True, input_pages_only=True, training_presentations=0,
                      counters=dict(counters), error=error)
        temporary = self.root / f'.prefetch_status.{os.getpid()}.tmp'
        temporary.write_text(json.dumps(record, allow_nan=False) + '\n')
        temporary.replace(self.root / 'prefetch_status.json')
        if changed or now - self.last_log >= 60:
            self.logger.info(json.dumps(record, allow_nan=False))
            self.last_log = now
        self.last_write, self.previous = now, (phase, trainer_phase)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--ahead', type=int, default=12)
    args = parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        parser.error('CUDA_VISIBLE_DEVICES must be explicitly empty')
    if not 1 <= args.ahead <= 12:
        parser.error('--ahead must be between 1 and 12')
    root = args.root.expanduser().resolve()
    if not root.is_dir():
        parser.error('--root must be an existing R75 run directory')
    for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
        os.environ[name] = '1'
    reporter, guard = Reporter(root, args.ahead), TrainerGuard(root)
    counters = dict(groups_prefetched=0, input_array_touches=0, input_rows_touched=0,
                    input_copy_bytes=0, input_gather_calls=0)
    inputs = None
    epoch_index = cursor = None
    prefetched, plan = set(), None
    try:
        action, phase = guard.state()
        if action == 'stop':
            reporter.write('stopped', phase, counters)
            return 0
        protocol = json.loads((root / 'protocol.json').read_text())
        sizes, cache_root = protocol['sizes'], Path(protocol['cache_root'])
        inputs = TrainInputs(cache_root, sizes)
        # Import the exact private schedule only after hiding all CUDA devices.
        from .r75_experiment import epoch_plan

        latest = None
        with ThreadPoolExecutor(max_workers=THREADS, thread_name_prefix='r75-input-io') as executor:
            while True:
                action, phase = guard.state()
                if action == 'stop':
                    reporter.write('stopped', phase, counters, epoch_index, cursor, len(prefetched))
                    return 0
                if action != 'training':
                    reporter.write('paused', phase, counters, epoch_index, cursor, len(prefetched))
                    time.sleep(1)
                    continue
                committed = latest_update(root / 'single_seed2' / 'updates.jsonl')
                if committed is not None and (latest is None or committed > latest):
                    latest = committed
                if latest is None:
                    reporter.write('waiting_for_updates', phase, counters)
                    time.sleep(1)
                    continue
                next_epoch, cursor = latest
                if cursor >= protocol['updates_per_epoch']:
                    next_epoch, cursor = next_epoch + 1, 0
                if next_epoch >= protocol['max_epochs']:
                    reporter.write('stopped', 'all_training_updates_committed', counters, next_epoch, cursor)
                    return 0
                if epoch_index != next_epoch:
                    plan = epoch_plan(sizes, next_epoch)
                    epoch_index = next_epoch
                    prefetched.clear()
                prefetched = {key for key in prefetched if key[0] == epoch_index and key[1] >= cursor}
                target = next((step for step in range(cursor, min(cursor + args.ahead, len(plan)))
                               if (epoch_index, step) not in prefetched), None)
                if target is None:
                    reporter.write('waiting', phase, counters, epoch_index, cursor, len(prefetched))
                    time.sleep(.25)
                    continue
                futures = [executor.submit(warm_task, problem, indices, inputs, guard)
                           for problem, indices in plan[target]]
                results = [future.result() for future in futures]
                for result in results:
                    for key in counters:
                        if key != 'groups_prefetched':
                            counters[key] += result[key]
                if all(result['complete'] for result in results):
                    prefetched.add((epoch_index, target))
                    counters['groups_prefetched'] += 1
                reporter.write('warming', phase, counters, epoch_index, cursor, len(prefetched))
    except KeyboardInterrupt:
        reporter.write('stopped', 'helper_interrupted', counters, epoch_index, cursor, len(prefetched))
        return 0
    except Exception as error:
        reporter.write('failed', 'helper_error', counters, epoch_index, cursor, len(prefetched), repr(error))
        return 1
    finally:
        if inputs is not None:
            inputs.close()


if __name__ == '__main__':
    raise SystemExit(main())
