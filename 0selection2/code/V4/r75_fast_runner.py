"""Resume the same R75 run with lossless RAM inputs and bounded IO profiling."""

import gc
import hashlib
import json
import os
import socket
import sys
import time
import weakref
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from code.unified_selector.registry import PROBLEMS
from . import r75_experiment as experiment
from .r75_cache import ProblemCache
from .r75_capacity import ROOT, append_json, utc, write_json
from .r75_fast_input import FastProblemCache, estimate_resident_bytes


INPUT_LIMIT = 16 * 2**30
PROFILE_UPDATES = 24
CHECK_TASKS = ('TSP', 'CVRP', 'ATSP', 'OVRPTW')


def latest(path):
    with path.open('rb') as stream:
        stream.seek(max(0, path.stat().st_size - 65536))
        lines = stream.read().splitlines()
    return json.loads(lines[-1])


class RuntimeInputs:
    def __init__(self, root, boundary):
        self.root, self.boundary = root, boundary
        self.directory = root / 'runtime_io_continuation1'
        self.live, self.pending = weakref.WeakSet(), []
        self.profiled, self.equivalent, self.evaluating = 0, {}, False

    def equivalence(self, loader):
        base = ProblemCache(loader.directory)
        try:
            n = loader.lengths
            if loader.problem == 'CVRP':
                rows = np.arange(loader.size - 17, loader.size)
            else:
                selected = [int(n.argmin()), int(n.argmax())]
                selected.extend(range(min(loader.size, 384)))
                rows = np.asarray(list(dict.fromkeys(selected))[:384], dtype=np.int64)
            old, new = base.batch(rows, device='cpu'), loader.batch_cpu(rows)
            if set(old) != set(new):
                raise ValueError('Compact input changes batch fields')
            for key in old:
                if torch.is_tensor(old[key]):
                    if (old[key].dtype != new[key].dtype or old[key].shape != new[key].shape
                            or not torch.equal(old[key], new[key])):
                        raise ValueError(f'{loader.problem}: compact batch differs at {key}, including padding')
                elif old[key] != new[key]:
                    raise ValueError(f'{loader.problem}: compact batch metadata differs at {key}')
            if loader.raw['costs'].dtype != np.float64 or loader.raw['winner'].dtype != np.int64:
                raise ValueError('Raw supervision precision must not change')
            self.equivalent[loader.problem] = dict(rows=len(rows), minimum_n=int(n[rows].min()),
                maximum_n=int(n[rows].max()), all_fields_identical=True, entire_padding_identical=True,
                fields=sorted(old), raw_cost_dtype=str(loader.raw['costs'].dtype),
                batch_cost_dtype=str(new['costs'].dtype), new_model_forwards=0)
            write_json(self.directory / 'cpu_batch_equivalence.json', self.equivalent)
        finally:
            base.close()

    def batch(self, reference, indices, device):
        loader = reference()
        if loader is None:
            raise RuntimeError('Input cache was released')
        device = torch.device(device)
        if device.type == 'cpu':
            return loader.batch_cpu(indices)
        if (loader.split != 'train' or self.evaluating
                or self.profiled >= PROFILE_UPDATES or device.type != 'cuda'):
            return loader.batch_gpu(loader.batch_cpu(indices), device)
        before = time.monotonic()
        cpu = loader.batch_cpu(indices)
        gathered = time.monotonic()
        pinned = loader.pin_batch(cpu)
        prepared = time.monotonic()
        first, last = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        first.record(torch.cuda.current_stream(device))
        result = loader.transfer_pinned(pinned, device)
        last.record(torch.cuda.current_stream(device))
        self.pending.append(dict(problem=loader.problem, rows=len(cpu['n']),
            gather_seconds=gathered - before, pin_seconds=prepared - gathered,
            enqueue_seconds=time.monotonic() - prepared, first=first, last=last))
        return result

    def loaders(self, cache_root, split):
        estimates = {p: estimate_resident_bytes(cache_root / split / p) for p in PROBLEMS}
        current = sum(loader.resident_bytes for loader in self.live)
        if current + sum(estimates.values()) > INPUT_LIMIT:
            raise MemoryError('Compact resident TRAIN/VAL inputs exceed the predeclared 16GiB budget')
        result, began = {}, time.monotonic()
        for problem in PROBLEMS:
            experiment.status(self.root, 'building_ram_inputs', input_split=split,
                              input_problem=problem, input_tasks_ready=len(result))
            loader = FastProblemCache(cache_root / split / problem,
                max_resident_bytes=estimates[problem], max_batch_size=384)
            self.live.add(loader)
            if split == 'train' and problem in CHECK_TASKS and problem not in self.equivalent:
                self.equivalence(loader)
            reference = weakref.ref(loader)
            loader.batch = lambda indices, device='cuda:0', ref=reference: self.batch(ref, indices, device)
            result[problem] = loader
            print('[R75 RAM INPUT] ' + json.dumps(dict(split=split, problem=problem,
                resident_bytes=loader.resident_bytes, seconds=time.monotonic() - began)), flush=True)
        receipt = dict(utc=utc(), split=split, tasks=len(result), seconds=time.monotonic() - began,
            owned_resident_bytes=sum(loader.resident_bytes for loader in self.live),
            budget_bytes=INPUT_LIMIT, precision_changed=False, full_dataset_pinned=False)
        append_json(self.directory / 'ram_loading.jsonl', receipt)
        return result

    def wrap_update(self, update):
        def measured(*args, **kwargs):
            if self.profiled >= PROFILE_UPDATES:
                return update(*args, **kwargs)
            entries, self.pending = self.pending, []
            if len(entries) != len(args[3]):
                raise ValueError('Runtime stage samples must match this real task group')
            for entry in entries:
                entry['last'].synchronize()
                entry['h2d_seconds'] = entry['first'].elapsed_time(entry['last']) / 1000.
                del entry['first'], entry['last']
            first, last = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            first.record()
            before = time.monotonic()
            result = update(*args, **kwargs)
            last.record()
            last.synchronize()
            timing = dict(utc=utc(), expected_update=self.boundary['successful_updates'] + self.profiled + 1,
                tasks=entries, gather_seconds=sum(row['gather_seconds'] for row in entries),
                pin_seconds=sum(row['pin_seconds'] for row in entries),
                h2d_seconds=sum(row['h2d_seconds'] for row in entries),
                update_wall_seconds=time.monotonic() - before,
                update_stream_span_seconds=first.elapsed_time(last) / 1000.,
                stream_span_is_not_pure_kernel_busy_time=True,
                same_official_optimizer_call=True, retries=result['retries'])
            append_json(self.directory / 'stage_timings.jsonl', timing)
            result['runtime_io_timing'] = timing
            self.profiled += 1
            return result
        return measured

    def close_all(self):
        for loader in list(self.live):
            loader.close()
        gc.collect()


def main():
    if '--resume' not in sys.argv:
        raise ValueError('This runtime-only entry must resume the existing single group')
    root = ROOT
    if '--root' in sys.argv:
        root = Path(sys.argv[sys.argv.index('--root') + 1]).resolve()
    directory = root / 'runtime_io_continuation1'
    boundary = json.loads((directory / 'quiesced.json').read_text())
    if (latest(root / 'single_seed2/updates.jsonl')['successful_update'] != boundary['successful_updates']
            or experiment.digest(root / 'single_seed2/last.pt') != boundary['checkpoint_sha256']):
        raise ValueError('Original logged/durable boundary changed; no silent replay permitted')
    protocol = json.loads((root / 'protocol.json').read_text())
    for name, identity in protocol['source_hashes'].items():
        if experiment.digest(Path(__file__).parent / name) != identity:
            raise ValueError(f'Original scientific source changed: {name}')
    sources = ('r75_fast_runner.py', 'r75_fast_input.py', 'run_v4_r75_fast.sh',
               'r75_checkpoint_boundary.py', 'r75_runtime_probe.py')
    write_json(directory / 'runtime_started.json', dict(utc=utc(), pid=os.getpid(), host=socket.gethostname(),
        slurm_step=os.environ.get('SLURM_STEP_ID'), batch=384, seed=2, groups=1,
        resumed_successful_update=boundary['successful_updates'],
        existing_logged_updates_replayed=0, profile_official_updates=PROFILE_UPDATES,
        input_budget_bytes=INPUT_LIMIT, new_architecture=False, protocol_unchanged=True,
        sources={name: experiment.digest(Path(__file__).parent / name) for name in sources}))
    runtime = RuntimeInputs(root, boundary)
    original_train = experiment.train
    original_evaluate = experiment.evaluate

    def evaluate(*args, **kwargs):
        runtime.evaluating = True
        try:
            return original_evaluate(*args, **kwargs)
        finally:
            runtime.evaluating = False

    def train(*args, **kwargs):
        try:
            value = original_train(*args, **kwargs)
        finally:
            runtime.close_all()
        completion = json.loads((root / 'training_complete.json').read_text())
        initial = json.loads((directory / 'prior_formal_started.json').read_text())
        elapsed = (datetime.fromisoformat(completion['completed_utc']) -
                   datetime.fromisoformat(initial['started_utc'])).total_seconds()
        completion.update(runtime_continuations=1,
            training_wall_seconds_scope='last process segment after runtime IO continuation only',
            total_formal_pipeline_elapsed_seconds=elapsed,
            total_elapsed_scope='includes original input preparation and runtime optimization pause',
            resumed_partial_epoch=boundary['epoch_index'] + 1,
            partial_epoch_optimization_seconds_scope='post-resume epoch wall only; prior successful step times retained separately',
            prior_partial_epoch_saved_step_seconds=boundary['current_epoch_saved_step_seconds'],
            logged_successful_updates_replayed=0,
            discarded_uncommitted_inflight_groups_upper_bound=1)
        write_json(root / 'training_complete.json', completion)
        return value

    experiment.cache_loaders = runtime.loaders
    experiment.update_group = runtime.wrap_update(experiment.update_group)
    experiment.train = train
    experiment.evaluate = evaluate
    try:
        experiment.main()
    finally:
        runtime.close_all()


if __name__ == '__main__':
    main()
