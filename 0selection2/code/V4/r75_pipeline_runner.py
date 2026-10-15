"""Continue R75 with one-ahead CPU batches and no redundant step-wide barriers."""

import ast
import inspect
import json
import os
import socket
import sys
import time
from datetime import datetime
from pathlib import Path

from . import r75_experiment as experiment
from .r75_capacity import ROOT, utc, write_json
from .r75_cpu_prefetch import CpuGroupPrefetch
from .r75_fast_runner import PROFILE_UPDATES, RuntimeInputs, latest


class Pipeline:
    def __init__(self):
        self.group = self.plan = None
        self.groups, self.rows = 0, 0
        self.wait_seconds = self.transfer_enqueue_seconds = 0.

    def take(self, loaders, plan, step):
        if self.plan is not plan:
            self.close()
            self.plan = plan
            self.group = CpuGroupPrefetch(loaders, plan, step)
        started = time.monotonic()
        cpu = self.group.take(step)
        prepared = time.monotonic()
        if len(cpu) != len(plan[step]):
            raise ValueError('Prefetch must retain every real task in the original group')
        # Only the main thread touches CUDA or reusable pinned staging.
        batches = [loaders[p].batch_gpu(batch, 'cuda:0')
                   for (p, _), batch in zip(plan[step], cpu)]
        self.groups += 1
        self.rows += sum(len(indices) for _, indices in plan[step])
        self.wait_seconds += prepared - started
        self.transfer_enqueue_seconds += time.monotonic() - prepared
        return batches

    def close(self):
        if self.group is not None:
            self.group.close()
        self.group = self.plan = None

    def receipt(self):
        return dict(groups=self.groups, rows=self.rows,
                    cpu_prefetch_wait_seconds=self.wait_seconds,
                    main_thread_transfer_enqueue_seconds=self.transfer_enqueue_seconds,
                    cpu_prefetch_depth=1, worker_cuda_calls=0)


class RuntimeTrain(ast.NodeTransformer):
    """Change only input supply and two timing barriers in the original loop."""

    def __init__(self):
        self.barriers = self.batches = 0

    def visit_Expr(self, node):
        if ast.unparse(node) == 'torch.cuda.synchronize(0)':
            self.barriers += 1
            return ast.copy_location(ast.Pass(), node)
        return self.generic_visit(node)

    def visit_Assign(self, node):
        if (len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id == 'batches' and isinstance(node.value, ast.ListComp)):
            expected = ast.parse(
                "[train_loaders[p].batch(indices, device='cuda:0') for p, indices in entries]",
                mode='eval').body
            if ast.dump(node.value) != ast.dump(expected):
                raise ValueError('Original batch expression changed; no broad loop rewrite permitted')
            node.value = ast.copy_location(ast.Call(
                func=ast.Name(id='_r75_prefetched_batches', ctx=ast.Load()),
                args=[ast.Name(id=name, ctx=ast.Load()) for name in ('train_loaders', 'plan', 'step')],
                keywords=[]), node.value)
            self.batches += 1
        return self.generic_visit(node)


def runtime_train(pipeline):
    tree = ast.parse(inspect.getsource(experiment.train))
    rewrite = RuntimeTrain()
    tree = ast.fix_missing_locations(rewrite.visit(tree))
    if (rewrite.barriers, rewrite.batches) != (2, 1):
        raise ValueError('Runtime rewrite must remove exactly two barriers and replace one input expression')
    experiment.__dict__['_r75_prefetched_batches'] = pipeline.take
    namespace = {}
    exec(compile(tree, str(Path(__file__)) + '::<train>', 'exec'), experiment.__dict__, namespace)
    return namespace['train']


def main():
    if '--resume' not in sys.argv:
        raise ValueError('Pipeline optimization must continue the existing single R75 run')
    root = ROOT
    if '--root' in sys.argv:
        root = Path(sys.argv[sys.argv.index('--root') + 1]).resolve()
    directory = root / 'runtime_pipeline_continuation2'
    boundary = json.loads((directory / 'quiesced.json').read_text())
    if (latest(root / 'single_seed2/updates.jsonl')['successful_update'] != boundary['successful_updates']
            or experiment.digest(root / 'single_seed2/last.pt') != boundary['checkpoint_sha256']):
        raise ValueError('Logged and durable checkpoint boundary must agree before continuation')
    protocol = json.loads((root / 'protocol.json').read_text())
    for name, identity in protocol['source_hashes'].items():
        if experiment.digest(Path(__file__).parent / name) != identity:
            raise ValueError('Original scientific source changed: ' + name)
    runtime, pipeline = RuntimeInputs(root, boundary), Pipeline()
    runtime.directory = directory
    runtime.profiled = PROFILE_UPDATES
    runtime.equivalent = json.loads(
        (root / 'runtime_io_continuation1/cpu_batch_equivalence.json').read_text())
    train_loop = runtime_train(pipeline)
    sources = ('r75_pipeline_runner.py', 'r75_cpu_prefetch.py', 'run_v4_r75_pipeline.sh',
               'r75_fast_runner.py', 'r75_fast_input.py', 'r75_pipeline_boundary.py')
    write_json(directory / 'runtime_started.json', dict(
        utc=utc(), pid=os.getpid(), host=socket.gethostname(),
        slurm_step=os.environ.get('SLURM_STEP_ID'), batch=384, seed=2, groups=1,
        resumed_successful_update=boundary['successful_updates'], existing_logged_updates_replayed=0,
        cpu_prefetch_groups=1, cpu_worker_threads=1, gpu_prefetch_groups=0,
        removed_global_timing_barriers_per_update=2, scientific_protocol_unchanged=True,
        optimizer_and_amp_retry_implementation_unchanged=True,
        sources={name: experiment.digest(Path(__file__).parent / name) for name in sources}))
    original_evaluate = experiment.evaluate

    def evaluate(*args, **kwargs):
        runtime.evaluating = True
        try:
            return original_evaluate(*args, **kwargs)
        finally:
            runtime.evaluating = False

    def train(*args, **kwargs):
        try:
            value = train_loop(*args, **kwargs)
        finally:
            pipeline.close()
            write_json(directory / 'prefetch_totals.json', pipeline.receipt())
            runtime.close_all()
        completion = json.loads((root / 'training_complete.json').read_text())
        initial = json.loads((root / 'runtime_io_continuation1/prior_formal_started.json').read_text())
        completion.update(runtime_continuations=2,
            training_wall_seconds_scope='last process segment after pipeline continuation only',
            total_formal_pipeline_elapsed_seconds=(datetime.fromisoformat(completion['completed_utc']) -
                                                   datetime.fromisoformat(initial['started_utc'])).total_seconds(),
            total_elapsed_scope='includes original input preparation and both runtime optimization pauses',
            partial_epoch_timings=[dict(epoch_index=b['epoch_index'], next_step=b['next_step'],
                prior_saved_step_seconds=b['current_epoch_saved_step_seconds'])
                for b in (json.loads((root / 'runtime_io_continuation1/quiesced.json').read_text()), boundary)],
            logged_successful_updates_replayed=0,
            discarded_uncommitted_inflight_groups_upper_bound=2)
        write_json(root / 'training_complete.json', completion)
        return value

    experiment.cache_loaders = runtime.loaders
    experiment.evaluate = evaluate
    experiment.train = train
    try:
        experiment.main()
    finally:
        pipeline.close()
        runtime.close_all()


if __name__ == '__main__':
    main()
