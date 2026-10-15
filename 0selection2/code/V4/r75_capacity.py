"""Bounded real-TRAIN capacity sizing; never launches a formal R75 run."""

import argparse
import csv
import gc
import json
import os
import socket
import subprocess
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from code.unified_selector.registry import GLOBAL_SOLVERS, PROBLEMS
from .r64_model import initialize
from .r65_gradients import GradientLayout, update_triple
from .r65_optimizer import make_optimizer, training_mode
from .train import configure_torch, set_seed


WORKSPACE = Path('/public/home/shiys/0selection2')
ROOT = WORKSPACE / 'code/V4/runs/R75_large_complete_labels'
DATA_ROOT = Path('/public/home/shiys/'
                 '\u8bad\u7ec3\u6570\u636e_\u516c\u5171\u5b8c\u6574\u6807\u7b7e_20261013_1923')
GPU_UUID = 'GPU-cb626bdb-4319-2f5e-b91f-17fcaa44a83d'
STRESS_TASKS = ('TSP', 'CVRP', 'ATSP', 'OVRPBLTW')
HEADROOM_BYTES = 3 * 2**30
MAX_CANDIDATES = 8
MAX_UPDATES = 80
MAX_GPU_SECONDS = 600
MAX_BATCH = 1024


def utc():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    temporary.replace(path)


def append_json(path, value):
    with path.open('a') as stream:
        stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + '\n')


def file_identity(path):
    stat = path.stat()
    return dict(path=str(path), bytes=stat.st_size, mtime_ns=stat.st_mtime_ns)


def protocol(args):
    order = np.random.default_rng(2).permutation(PROBLEMS).tolist()
    return dict(seed=2, stage='batch_capacity_only', formal_training_authorized=False,
        user_batch_approval_received=False, formal_updates=0, validation_read=False,
        test_read=False, model='unchanged r64_model.NativeUtilitySelector',
        optimizer='unchanged official R65B AdamWScheduleFree, scratch/discarded',
        objective='native CE of two dropout forwards + symmetric legal-pool KL',
        consistency_weight=.35, dropout=.1, gradient_clip=1., amp='FP16',
        task_gradients='three distinct tasks, unchanged state, FP32 arithmetic mean, one step',
        effective_batch='3 * per_task_batch; no sample microbatching',
        data_root=str(args.data_root), problems=list(PROBLEMS),
        solver_names=list(GLOBAL_SOLVERS), cache='selected real TRAIN rows on CPU only',
        geometry='input-only geometry cached on CPU; preprocessing reported separately',
        geometry_normalization='historical R65 TRAIN stats ONLY for capacity; refit new TRAIN before formal run',
        representative_triples=[order[i:i+3] for i in range(0, len(order), 3)],
        stress_triples=[['TSP', 'CVRP', 'OVRPBLTW'], ['TSP', 'CVRP', 'ATSP']],
        warmup_updates_per_candidate=2, timed_updates_per_candidate=6,
        timing='CPU cached-row gathering, H2D transfer, both forwards, backward, reduction, clipping, optimizer and AMP retries',
        minimum_physical_headroom_bytes=HEADROOM_BYTES,
        selection='largest safety-passing batch within95% of best measured representative throughput',
        selection_reason='user explicitly requests a large batch; avoid larger batches with material throughput loss',
        candidates='start256,384,512; bounded growth to1024 or32-row refinement at first unsafe size',
        max_candidates=MAX_CANDIDATES, max_successful_scratch_updates=MAX_UPDATES,
        max_gpu_seconds=MAX_GPU_SECONDS, labels='native ind unchanged; no final costs in model input',
        approved_gpu_uuid=GPU_UUID, parent_goal='paused', R74='terminated; disk files retained',
        legacy_B_semantics='R58 deferred/unfixed; this is the legacy deployment scenario')


def prepare(args):
    from .r75_data import build_batch, lengths, load_dataset

    args.root.mkdir(parents=True, exist_ok=True)
    target = args.root / 'capacity_inputs'
    manifest_path = args.root / 'capacity_data.json'
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if not manifest.get('complete') or manifest['data_root'] != str(args.data_root):
            raise ValueError('Do not silently reuse incomplete/different capacity inputs')
        for problem, entry in manifest['tasks'].items():
            if not (target / (problem + '.pt')).is_file():
                raise ValueError('Missing capacity input: ' + problem)
            for key in ('instances', 'labels'):
                old = entry[key]
                if file_identity(Path(old['path'])) != old:
                    raise ValueError('Snapshot changed: ' + old['path'])
        print('[R75] reuse complete CPU capacity fixtures; no model or formal update', flush=True)
        return manifest
    if any(target.glob('*.pt')):
        raise ValueError('An incomplete capacity preparation exists; preserve it for diagnosis')
    target.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    planned = protocol(args)
    write_json(args.root / 'capacity_protocol.json', planned)
    manifest = dict(started_utc=utc(), data_root=str(args.data_root), tasks={},
                    complete=False, validation_read=False, test_read=False)
    for problem in PROBLEMS:
        before = time.monotonic()
        dataset = load_dataset(problem, data_root=args.data_root)
        sizes = np.asarray(lengths(dataset), dtype=np.int64)
        generator = np.random.default_rng(2 + PROBLEMS.index(problem))
        mixed_indices = generator.permutation(len(dataset))[:MAX_BATCH]
        cases = dict(mixed=build_batch(dataset, mixed_indices, device='cpu'))
        indices = dict(mixed=mixed_indices.tolist())
        if problem in STRESS_TASKS:
            largest = np.argsort(-sizes, kind='stable')[:MAX_BATCH]
            cases['stress'] = build_batch(dataset, largest, device='cpu')
            indices['stress'] = largest.tolist()
        counts, frequencies = np.unique(sizes, return_counts=True)
        directory = args.data_root / (problem + 'train')
        entry = dict(rows=len(dataset), min_nodes=int(sizes.min()), max_nodes=int(sizes.max()),
            node_histogram={str(int(n)): int(c) for n, c in zip(counts, frequencies)},
            sampled_indices=indices, sampled_new_rows={k: sum(i >= 10000 for i in v) for k, v in indices.items()},
            instances=file_identity(directory / 'dataset.pkl'), labels=file_identity(directory / 'raw_label.pkl'),
            prepared_seconds=time.monotonic()-before)
        torch.save(cases, target / (problem + '.pt'))
        manifest['tasks'][problem] = entry
        write_json(manifest_path, manifest)
        print('[R75 CPU inputs] ' + json.dumps(dict(problem=problem, rows=entry['rows'],
            nodes=[entry['min_nodes'], entry['max_nodes']], seconds=entry['prepared_seconds'])), flush=True)
        del cases, dataset
        gc.collect()
    manifest.update(complete=True, completed_utc=utc(), seconds=time.monotonic()-started,
                    total_train_rows=sum(e['rows'] for e in manifest['tasks'].values()))
    write_json(manifest_path, manifest)
    return manifest


def availability():
    gpu = subprocess.check_output(['nvidia-smi', '--id=' + GPU_UUID,
        '--query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu',
        '--format=csv,noheader,nounits'], text=True)
    clients = subprocess.check_output(['nvidia-smi', '--id=' + GPU_UUID,
        '--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory',
        '--format=csv,noheader,nounits'], text=True)
    rows = list(csv.reader(gpu.splitlines()))
    if len(rows) != 1:
        raise RuntimeError('The assigned UUID must resolve to exactly one GPU')
    index, actual, name, total, used, utilization = [item.strip() for item in rows[0]]
    if actual != GPU_UUID or '3090' not in name:
        raise RuntimeError('Never switch to a different GPU')
    receipt = dict(observed_utc=utc(), host=socket.gethostname(),
        slurm_step=os.environ.get('SLURM_JOB_ID', '') + '.' + os.environ.get('SLURM_STEP_ID', ''),
        gpu_uuid=actual, physical_index=int(index), logical_index=0, name=name,
        total_mib=int(total), used_mib=int(used), utilization_pct=int(utilization),
        clients=list(csv.reader(clients.splitlines())))
    if int(used) >= 500 or receipt['clients']:
        raise RuntimeError('Assigned GPU occupied; stop without disturbing other work: ' + json.dumps(receipt))
    return receipt


def gather_cpu(batch, batch_size):
    maximum = int(batch['n'][:batch_size].max())
    selected = {}
    for key, value in batch.items():
        if not torch.is_tensor(value) or key == 'pool_ids':
            selected[key] = value
            continue
        value = value[:batch_size]
        if key in ('node', 'node_mask', 'node_geom'):
            value = value[:, :maximum]
        elif key == 'matrix':
            value = value[:, :maximum, :maximum]
        selected[key] = value.contiguous()
    if len(selected['n']) != batch_size:
        raise ValueError('Capacity rows must be real distinct examples, not repeated padding')
    return selected


def candidate(args, batch_size, fixtures, planned, geometry, budget, started):
    model = optimizer = scaler = layout = batches = None
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(0)
    available, total = torch.cuda.mem_get_info(0)
    external = total - available - torch.cuda.memory_reserved(0)
    physical_peak = [total-available]
    stop = threading.Event()

    def sample_memory():
        while not stop.is_set():
            free, size = torch.cuda.mem_get_info(0)
            physical_peak[0] = max(physical_peak[0], size-free)
            stop.wait(.02)

    observer = threading.Thread(target=sample_memory, daemon=True)
    result = dict(batch_size=batch_size, effective_batch=3*batch_size, updates=[],
        complete=False, genuine_cuda_oom=False, formal_updates=0,
        minimum_headroom_bytes=HEADROOM_BYTES, benchmark_state_discarded=True,
        allocated_peak_bytes=0, reserved_peak_bytes=0, device_peak_bound_bytes=0)
    phases = [('stress', group) for group in planned['stress_triples']]
    phases += [('mixed', group) for group in planned['representative_triples']]
    observer.start()
    try:
        model, _ = initialize(geometry, device='cuda:0', seed=2)
        optimizer = make_optimizer(model, 'B')
        training_mode(model, optimizer, 'B')
        scaler = torch.amp.GradScaler('cuda', enabled=True)
        layout, growth = GradientLayout(model), 0
        for step, (phase, tasks) in enumerate(phases):
            if budget['successful_updates'] >= MAX_UPDATES or time.monotonic()-started > MAX_GPU_SECONDS:
                result['budget_exhausted'] = True
                break
            torch.cuda.reset_peak_memory_stats(0)
            set_seed(750002 + step)
            torch.cuda.synchronize(0)
            before = time.monotonic()
            cpu_batches = [gather_cpu(fixtures[p][phase], batch_size) for p in tasks]
            batches = [{k: v.to('cuda:0', non_blocking=True) if torch.is_tensor(v) else v
                        for k, v in batch.items()} for batch in cpu_batches]
            values = update_triple(model, optimizer, scaler, batches, .35, layout, growth)
            torch.cuda.synchronize(0)
            elapsed = time.monotonic()-before
            growth = values['scale_growth_successes']
            budget['successful_updates'] += 1
            budget['presentations'] += 3*batch_size
            budget['amp_retries'] += values['retries']
            allocated = torch.cuda.max_memory_allocated(0)
            reserved = torch.cuda.max_memory_reserved(0)
            peak = max(physical_peak[0], external+reserved)
            row = dict(batch_size=batch_size, phase=phase, tasks=tasks, step=step+1,
                rows=3*batch_size, seconds=elapsed, instances_per_second=3*batch_size/elapsed,
                padded_nodes=[int(b['node_mask'].shape[1]) for b in batches],
                min_nodes=[int(b['n'].min()) for b in batches],
                peak_allocated_gib=allocated/2**30, peak_reserved_gib=reserved/2**30,
                device_peak_bound_gib=peak/2**30, physical_headroom_gib=(total-peak)/2**30,
                amp_retries=values['retries'], finite_gradient=values['finite_gradient'],
                grad_norm=values['grad_norm'], loss=values['loss'], optimizer_state_entries=len(optimizer.state),
                formal=False)
            result['updates'].append(row)
            append_json(args.root/'capacity_updates.jsonl', row)
            result['allocated_peak_bytes'] = max(result['allocated_peak_bytes'], allocated)
            result['reserved_peak_bytes'] = max(result['reserved_peak_bytes'], reserved)
            result['device_peak_bound_bytes'] = max(result['device_peak_bound_bytes'], peak)
            print('[R75 capacity step] ' + json.dumps(row), flush=True)
            batches = cpu_batches = None
            if total-peak < HEADROOM_BYTES:
                result['insufficient_headroom'] = True
                break
        result['complete'] = len(result['updates']) == len(phases)
    except torch.cuda.OutOfMemoryError as error:
        result.update(genuine_cuda_oom=True, error=str(error), failed_phase=phase,
                      failed_tasks=tasks, failed_attempt_rows=3*batch_size)
        budget['cuda_oom_attempts'] += 1
        print('[R75 capacity OOM] batch=' + str(batch_size), flush=True)
    except Exception as error:
        result.update(technical_failure=repr(error))
        raise
    finally:
        stop.set()
        observer.join(timeout=2)
        result['allocated_peak_bytes'] = max(result['allocated_peak_bytes'], torch.cuda.max_memory_allocated(0))
        result['reserved_peak_bytes'] = max(result['reserved_peak_bytes'], torch.cuda.max_memory_reserved(0))
        peak = max(physical_peak[0], external+result['reserved_peak_bytes'])
        result.update(device_peak_bound_bytes=peak, physical_headroom_gib=(total-peak)/2**30,
                      total_device_gib=total/2**30)
        model = optimizer = scaler = layout = batches = None
        gc.collect()
        torch.cuda.empty_cache()
        result['eligible'] = result['complete'] and total-peak >= HEADROOM_BYTES
        timed = [row for row in result['updates'] if row['phase'] == 'mixed']
        result['mixed_instances_per_second'] = (sum(r['rows'] for r in timed)/sum(r['seconds'] for r in timed)
                                                if len(timed) == 6 else None)
        write_json(args.root / ('capacity_batch' + str(batch_size) + '.json'), result)
    return result


def next_batch(results):
    safe = [r['batch_size'] for r in results if r['eligible']]
    unsafe = [r['batch_size'] for r in results if not r['eligible']]
    if not safe:
        return results[-1]['batch_size']//2 if results[-1]['batch_size'] > 64 else None
    lower = max(safe)
    upper = min((b for b in unsafe if b > lower), default=None)
    if upper is not None:
        middle = ((lower + upper)//2)//32*32
        return middle if lower < middle < upper else None
    if lower >= MAX_BATCH:
        return None
    return {256: 384, 384: 512, 512: 768, 768: 1024}.get(lower, min(MAX_BATCH, lower*2))


def report(args, results, budget, elapsed, device):
    eligible = [r for r in results if r['eligible']]
    best_speed = max((r['mixed_instances_per_second'] for r in eligible), default=None)
    selected = max((r for r in eligible if r['mixed_instances_per_second'] >= best_speed*.95),
                   key=lambda r: r['batch_size'], default=None)
    decision = dict(status='awaiting_user_batch_approval' if selected else 'no_safe_batch_measured',
        recommended_per_task_batch=selected['batch_size'] if selected else None,
        effective_batch=selected['effective_batch'] if selected else None,
        criterion='largest safety-passing batch within95% of best representative throughput',
        minimum_headroom_gib=3., device=device, candidates=results, scratch_work=budget,
        allocator_configuration=os.environ.get('PYTORCH_CUDA_ALLOC_CONF', 'default'),
        gpu_benchmark_seconds=elapsed, formal_training_started=False,
        batch_approved=False, goal_resumed=False, validation_read=False, test_read=False,
        R74='terminated; all disk files retained')
    write_json(args.root/'batch_recommendation.json', decision)
    fields = ['per_task_batch', 'effective_batch', 'status', 'peak_allocated_gib',
              'peak_reserved_gib', 'physical_headroom_gib', 'mixed_instances_per_second', 'scratch_updates']
    with (args.root/'batch_comparison.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for r in results:
            writer.writerow(dict(per_task_batch=r['batch_size'], effective_batch=r['effective_batch'],
                status='safe' if r['eligible'] else ('OOM' if r['genuine_cuda_oom'] else 'insufficient_headroom'),
                peak_allocated_gib=r['allocated_peak_bytes']/2**30,
                peak_reserved_gib=r['reserved_peak_bytes']/2**30,
                physical_headroom_gib=r['physical_headroom_gib'],
                mixed_instances_per_second=r['mixed_instances_per_second'], scratch_updates=len(r['updates'])))
    lines = ['# R75 Batch Selection Only', '',
        'Formal training, validation and TEST have NOT started. Await explicit user batch approval.', '',
        'Model: unchanged R65B native-utility dual-stream selector; seed2. All sizing states are discarded.',
        'Full new TRAIN has1,786,887 rows. Only deterministic real TRAIN subsets are used here.',
        'Two maximum-size stress updates precede six representative mixed-size updates covering all18 tasks.',
        'The minimum conservative physical-memory headroom is3GiB. Larger batch is preferred only within95% of best safe throughput.', '',
        '| Per Task | Effective | Allocated GiB | Reserved GiB | Headroom GiB | Mixed Instances/s | Safe |',
        '|---:|---:|---:|---:|---:|---:|---|']
    for r in results:
        speed = r['mixed_instances_per_second']
        lines.append('| {} | {} | {:.3f} | {:.3f} | {:.3f} | {} | {} |'.format(
            r['batch_size'], r['effective_batch'], r['allocated_peak_bytes']/2**30,
            r['reserved_peak_bytes']/2**30, r['physical_headroom_gib'],
            '{:.1f}'.format(speed) if speed is not None else 'not measured', r['eligible']))
    lines += ['', 'Recommendation: per-task batch{} / effective batch{}.'.format(
        decision['recommended_per_task_batch'], decision['effective_batch']), '',
        'Calibration cost: {} successful scratch updates, {} native presentations, {} AMP retries, {} CUDA OOM attempts.'.format(
            budget['successful_updates'], budget['presentations'], budget['amp_retries'], budget['cuda_oom_attempts']),
        'These are capacity measurements, not a loss/learning-rate/model search or performance result.',
        'CPU input/geometry preparation is excluded from steady-state timing and reported separately in capacity_data.json.',
        'Current-batch CPU slicing, H2D, complete updates and retry work are included in measured update time.',
        'Historical geometry normalization is used only for sizing; formal training requires new TRAIN statistics.',
        'Future formal training must use new sample-count-aware schedules, fresh seed2 model/optimizer/scaler and no sizing weights.',
        'The original native70 goal remains paused. R74 was terminated at user request; disk files are preserved.',
        'Legacy eight B-task semantics remain deferred/unfixed. No Git commit is created at this batch-selection stage.', '']
    (args.root/'BATCH_SELECTION.md').write_text('\n'.join(lines))
    print('[R75 waiting for user approval] ' + json.dumps({k: decision[k] for k in
          ('status', 'recommended_per_task_batch', 'effective_batch', 'formal_training_started')}), flush=True)
    return decision


def measure(args):
    if (args.root/'batch_recommendation.json').exists():
        raise ValueError('A batch recommendation already exists; do not silently rerun calibration')
    planned = json.loads((args.root/'capacity_protocol.json').read_text())
    expected_allocator = planned.get('allocator_configuration', 'default')
    if os.environ.get('PYTORCH_CUDA_ALLOC_CONF', 'default') != expected_allocator:
        raise ValueError('The allocator configuration differs from the capacity protocol')
    manifest = json.loads((args.root/'capacity_data.json').read_text())
    if not manifest['complete'] or manifest['data_root'] != str(args.data_root):
        raise ValueError('Complete TRAIN-only capacity fixtures are required')
    if os.environ.get('CUDA_VISIBLE_DEVICES') != GPU_UUID:
        raise RuntimeError('Bind only the assigned UUID before importing CUDA state')
    device = availability()
    write_json(args.root/'capacity_device.json', device)
    configure_torch()
    torch.cuda.set_device(0)
    fixtures = {p: torch.load(args.root/'capacity_inputs'/(p+'.pt'), map_location='cpu', weights_only=False)
                for p in PROBLEMS}
    for groups in fixtures.values():
        for batch in groups.values():
            for key, value in batch.items():
                if torch.is_tensor(value):
                    batch[key] = value.pin_memory()
    historical = WORKSPACE/'code/V4/runs/R65_schedule_free_full18/protocol.json'
    geometry = json.loads(historical.read_text())['geometry_stats']
    write_json(args.root/'capacity_started.json', dict(utc=utc(), stage='capacity_only', formal_updates=0,
        pid=os.getpid(), device=device, fixtures=manifest['total_train_rows']))
    import wandb
    run = wandb.init(project='selector', entity='yjkds-southern-university-of-science-technology',
        mode='offline', name='r75-capacity-only-seed2', config=planned, dir=str(args.root))
    started = time.monotonic()-planned.get('prior_gpu_seconds', 0.)
    budget = dict(successful_updates=0, presentations=0, amp_retries=0, cuda_oom_attempts=0, formal_updates=0)
    budget.update(planned.get('prior_scratch_work', {}))
    results, size = [], planned.get('initial_batch', 256)
    candidate_limit = planned.get('remaining_candidate_limit', MAX_CANDIDATES)
    try:
        while size is not None and len(results) < candidate_limit:
            if budget['successful_updates']+8 > MAX_UPDATES or time.monotonic()-started >= MAX_GPU_SECONDS:
                break
            result = candidate(args, size, fixtures, planned, geometry, budget, started)
            results.append(result)
            write_json(args.root/'capacity_measurements.json', dict(results=results, scratch_work=budget))
            run.log(dict(per_task_batch=size, eligible=int(result['eligible']),
                physical_headroom_gib=result['physical_headroom_gib'],
                peak_allocated_gib=result['allocated_peak_bytes']/2**30,
                mixed_instances_per_second=result['mixed_instances_per_second'] or 0))
            size = next_batch(results)
        return report(args, results, budget, time.monotonic()-started, device)
    except Exception as error:
        write_json(args.root/'capacity_failure.json', dict(utc=utc(), error=repr(error), scratch_work=budget,
            formal_training_started=False, validation_read=False, test_read=False))
        raise
    finally:
        run.finish()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('prepare', 'measure'))
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--data-root', type=Path, default=DATA_ROOT)
    args = parser.parse_args()
    if args.stage == 'prepare':
        prepare(args)
    else:
        measure(args)


if __name__ == '__main__':
    main()
