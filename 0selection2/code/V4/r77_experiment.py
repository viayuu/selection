"""Update-matched R77 arms: independent single GPUs, immutable cosine LR, one TEST."""

import argparse
import csv
import fcntl
import hashlib
import json
import math
import os
import random
import socket
import subprocess
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from code.unified_selector.registry import GLOBAL_SOLVERS, PROBLEMS
from .multitask_probe import task_seed
from .pairwise_objective import inputs_only
from .r40_experiment import aggregate, families
from .r76_experiment import (
    DATA_ROOT, LOSS_RECIPE, PART_KEYS, PinnedPrefetch, RDropObjective, Runtime,
    capture_rank_rng, file_hash, fixed_sbs_metrics, open_cache, read_json,
    restore_rank_rng, save_checkpoint, update_batch, utc, write_json,
)
from .r77_data import (
    DEFAULT_ROOT, DEFAULT_SOURCE, FULL_ROWS, GROUPS, SUBSET_ROWS, IndexedCache, parameter_hash,
)
from .solver_code_encoder import make_r45_model
from .train import configure_torch, set_seed


BATCH_SIZE = 1024
EVAL_EVERY = len(PROBLEMS) * math.ceil(FULL_ROWS / BATCH_SIZE)  # 1242 successful updates
MAX_UPDATES, WARMUP_UPDATES = 40 * EVAL_EVERY, 3 * EVAL_EVERY
PEAK_LR, END_LR = 1e-4, 1e-6


def learning_rate(update, max_updates=MAX_UPDATES, warmup_updates=WARMUP_UPDATES):
    """One-based successful update number; no validation-dependent LR state."""
    if not 1 <= update <= max_updates or not 0 < warmup_updates < max_updates:
        raise ValueError('LR is defined only on the precommitted successful-update budget')
    if update <= warmup_updates:
        return PEAK_LR * update / warmup_updates
    progress = (update - warmup_updates) / (max_updates - warmup_updates)
    return END_LR + .5 * (PEAK_LR - END_LR) * (1. + math.cos(math.pi * progress))


def task_order(updates=MAX_UPDATES, seed=2, problems=PROBLEMS):
    if updates < 1 or len(problems) < 2 or len(set(problems)) != len(problems):
        raise ValueError('Use positive updates and at least two distinct alternating tasks')
    result, previous = [], None
    for cycle in range(math.ceil(updates / len(problems))):
        tasks = list(problems)
        random.Random(seed + 701000003 + cycle * 1000003).shuffle(tasks)
        if tasks[0] == previous:
            tasks[0], tasks[1] = tasks[1], tasks[0]
        result.extend(tasks)
        previous = tasks[-1]
    return result[:updates]


class ShuffleStream:
    """Every row once per shuffle cycle; retain tails when forming exact 1024 batches."""

    def __init__(self, size, seed, draws=0):
        if size <= 0 or draws < 0:
            raise ValueError('Shuffle streams require a positive population and nonnegative cursor')
        self.size, self.seed, self.draws = int(size), int(seed), int(draws)
        self._cycle, self._permutation = None, None

    def take(self, count=BATCH_SIZE):
        if count <= 0:
            raise ValueError('A logical batch must contain real rows')
        parts, remaining = [], int(count)
        while remaining:
            cycle, offset = divmod(self.draws, self.size)
            if cycle != self._cycle:
                self._permutation = torch.randperm(self.size, generator=torch.Generator().manual_seed(
                    self.seed + cycle * 1000003))
                self._cycle = cycle
            take = min(remaining, self.size - offset)
            parts.append(self._permutation[offset:offset + take])
            self.draws += take
            remaining -= take
        return torch.cat(parts)


def micro_caps(gpu_type='4090', large_cap=None):
    limit = {'4090': 352, '3090': 384}[gpu_type]
    cap = limit if large_cap is None else large_cap
    if not 1 <= cap <= limit:
        raise ValueError(f'{gpu_type} TSP/CVRP microbatch cap must be at most {limit}')
    return {problem: cap if problem in ('TSP', 'CVRP') else BATCH_SIZE for problem in PROBLEMS}


def initialize_identical(initial, geometry, device):
    set_seed(2)
    selector = make_r45_model(initial['model_params'])
    actual = dict(selector.named_parameters())
    if set(actual) != set(initial['trainable']):
        raise ValueError('Both arms must have exactly the same named trainable parameters')
    with torch.no_grad():
        for name, tensor in actual.items():
            tensor.copy_(initial['trainable'][name])
        branch = selector.instance_encoder.geometry_residual
        branch.mean.copy_(torch.tensor(geometry['mean']))
        branch.std.copy_(torch.tensor(geometry['std']))
    if parameter_hash(actual) != initial['trainable_sha256']:
        raise ValueError('identicalinit failed before the first optimizer update')
    return selector.to(device)


def prepared_config(args):
    ready = read_json(args.root / 'prepared.json')
    if ready['experiment'] != 'R77' or Path(ready['data_root']).resolve() != args.data_root.resolve():
        raise ValueError('Use the exact TRAIN-only prepared R77 release')
    for path, digest in ready['artifact_sha256'].items():
        if file_hash(args.root / path) != digest:
            raise ValueError(f'Locked preparation changed: {path}')
    return ready, read_json(args.root / 'source_cache_manifest.json')


def open_arm_cache(args, manifest, split, problem):
    source = open_cache(args.cache_root, split, problem, manifest, args.data_root)
    indices = np.load(args.root / 'indices/A' / (problem + '.npy'), allow_pickle=False) if (
        split == 'train' and args.arm == 'A') else None
    return IndexedCache(source, indices)


def metrics_summary(per_problem):
    return dict(per_problem=per_problem, macro=aggregate(per_problem), families=families(per_problem),
        native_total=sum(value['n'] for value in per_problem.values()),
        native_correct=sum(value['native_correct'] for value in per_problem.values()))


@torch.no_grad()
def evaluate(selector, args, manifest, common_sbs, own_sbs, split, tag):
    arm_root = args.root / GROUPS[args.arm]
    directory = arm_root / 'evaluation' / split / tag
    directory.mkdir(parents=True, exist_ok=True)
    write_json(arm_root / 'status.json', dict(phase='evaluation', split=split, tag=tag, updated_utc=utc()))
    was_training, arrays, primary, own = selector.training, {}, {}, {}
    selector.eval()
    device = next(selector.parameters()).device
    started = time.monotonic()
    try:
        for problem in PROBLEMS:
            cache = open_arm_cache(args, manifest, split, problem)
            try:
                requests = ((problem, torch.arange(start, min(start + args.eval_batch_size, cache.size)),
                    min(args.eval_batch_size, cache.size - start))
                    for start in range(0, cache.size, args.eval_batch_size))
                scores = []
                with PinnedPrefetch({problem: cache}, requests, device) as batches:
                    for _, _, _, batch in batches:
                        scores.append(selector(inputs_only(batch))['logits'].float().cpu().numpy())
                scores = np.concatenate(scores)
                if len(scores) != cache.size:
                    raise ValueError('Full evaluation must retain every original tail row')
                primary[problem], _ = fixed_sbs_metrics(scores, cache.raw, common_sbs[problem])
                own[problem], _ = fixed_sbs_metrics(scores, cache.raw, own_sbs[problem])
                source_rows = np.arange(cache.size, dtype=np.int64) if cache.indices is None else cache.indices
                prefix = problem + '__'
                arrays.update({prefix + key: value for key, value in dict(logits=scores,
                    indices=np.arange(cache.size, dtype=np.int64), source_indices=source_rows.copy(),
                    costs=cache.raw['costs'].copy(), winner=cache.raw['winner'].copy(),
                    pool=np.asarray(cache.pool), pool_ids=np.asarray(cache.pool_ids, dtype=np.int64),
                    nodes=np.asarray(cache.lengths).copy()).items()})
            finally:
                cache.close()
    finally:
        selector.train(was_training)
    temporary = directory / f'.rank000.{os.getpid()}.tmp'
    with temporary.open('wb') as stream:
        np.savez(stream, **arrays)
    temporary.replace(directory / 'rank000.npz')
    result = metrics_summary(primary)
    result.update(own_train_sbs=metrics_summary(own), seconds=time.monotonic() - started,
        sbs_reference='common FULL 70114 TRAIN-fitted SBS, identical denominator for both arms',
        own_sbs_usage='reporting only; never LR, budget, checkpoint or TEST selection',
        evaluated_population='own TRAIN population' if split == 'train' else 'unchanged full held-out split')
    write_json(directory / 'metrics.json', result)
    print('[R77 eval] ' + json.dumps(dict(arm=args.arm, split=split, tag=tag,
        n=result['native_total'], seconds=result['seconds'], **result['macro'])), flush=True)
    return result


def protocol(args, ready, geometry, common_sbs, own_sbs, order, caps):
    return dict(experiment='R77', arm=args.arm, seed=2, initialization='scratch_identicalinit',
        common_trainable_sha256=ready['trainable_sha256'], solver_names=GLOBAL_SOLVERS,
        model_params=ready['model_params'], data_root=str(args.data_root.resolve()),
        preparation_sha256=file_hash(args.root / 'prepared.json'),
        train_rows_per_problem=SUBSET_ROWS if args.arm == 'A' else FULL_ROWS,
        global_batch_size=BATCH_SIZE, max_successful_updates=MAX_UPDATES,
        updates_per_problem=MAX_UPDATES // len(PROBLEMS), eval_every_updates=EVAL_EVERY,
        warmup_updates=WARMUP_UPDATES, rdrop_ramp_updates=WARMUP_UPDATES,
        lr=dict(rule='fixed successful-update warmup then cosine; independent of VAL',
            peak=PEAK_LR, end=END_LR, warmup_updates=WARMUP_UPDATES, max_updates=MAX_UPDATES),
        early_stop=False, fixed_budget=True, optimizer='AdamW', weight_decay=1e-4,
        grad_clip=1., dropout=.1, rdrop_weight=.35, loss=LOSS_RECIPE,
        checkpoint_rule='strict minimum complete VAL macro_vs_sbs_pct using common FULL TRAIN SBS',
        baselines=common_sbs, own_train_sbs=own_sbs, own_sbs_selection_usage=False,
        geometry_stats=geometry, task_order_sha256=hashlib.sha256('\n'.join(order).encode('ascii')).hexdigest(),
        stream='continuous per-task shuffle; exact1024 rows; retained tails concatenate across cycles',
        task_schedule='common seeded permutations of all18; no consecutive identical task, including boundaries',
        microbatch_caps=caps, threads=2, prefetch_buffers=2, world_size=1,
        gradients='row-count/global1024 weighted means; one clip and step; overflow replays entire batch and RNG',
        train_eval='own TRAIN every5 VAL points, plus the validation-selected best finally',
        eval_batch_size=args.eval_batch_size, test='one full pass only after immutable best lock',
        r76_difference='same architecture/loss, intentionally changed batch and update-axis LR/budget; not a sample-size control',
        wandb=args.wandb, wandb_mode=args.wandb_mode)


def wandb_run(args, config):
    if not args.wandb:
        return None
    settings = dict(project='selector', entity='yjkds-southern-university-of-science-technology',
        name=f'R77{args.arm}_{GROUPS[args.arm]}',
        id='r77-' + hashlib.sha256((str(args.root) + args.arm).encode()).hexdigest()[:16],
        config=config, dir=str(args.root / GROUPS[args.arm]))
    try:
        import wandb
        mode = args.wandb_mode
        if mode == 'auto':
            mode = 'online' if os.environ.get('WANDB_API_KEY') else 'offline'
        return wandb.init(**settings, mode=mode, resume='allow' if mode == 'online' else None)
    except Exception as error:
        print(f'[R77 W&B] {type(error).__name__}; trying offline fallback', flush=True)
        try:
            import wandb
            return wandb.init(**settings, mode='offline')
        except Exception as fallback:
            print(f'[R77 W&B] {type(fallback).__name__}; local records remain authoritative', flush=True)
            return None


def load_checkpoint(path, device='cpu'):
    state = torch.load(path, map_location='cpu', weights_only=False)
    if state['args']['experiment'] != 'R77' or state['args']['initialization'] != 'scratch_identicalinit':
        raise ValueError('R77 resumes only its own from-scratch paired checkpoints')
    model = make_r45_model(state['args']['model_params'], state_dict=state['model'])
    return model.to(device), state


def train(args, device):
    root = args.root / GROUPS[args.arm]
    if (root / 'training_complete.json').exists():
        raise ValueError('This R77 arm already completed its fixed training budget')
    ready, manifest = prepared_config(args)
    geometry = read_json(root / 'geometry_stats.json')
    common_sbs = read_json(args.root / 'common_full_train_sbs.json')
    own_sbs = read_json(root / 'own_train_sbs.json')
    order, caps = task_order(), micro_caps(args.gpu_type, args.large_micro_cap)
    config = protocol(args, ready, geometry, common_sbs, own_sbs, order, caps)
    if (root / 'protocol.json').exists() and read_json(root / 'protocol.json') != config:
        raise ValueError('Keep the exact prepared protocol, caps and update budget on resume')
    last = root / 'last.pt'
    if last.exists() != args.resume:
        raise ValueError('Use --resume exactly when this arm already has last.pt')
    checkpoint = None
    if args.resume:
        selector, checkpoint = load_checkpoint(last, device)
        if checkpoint['args'] != config:
            raise ValueError('Resume checkpoint differs from the immutable arm protocol')
    else:
        initial = torch.load(args.root / 'identicalinit_seed2.pt', map_location='cpu', weights_only=False)
        selector = initialize_identical(initial, geometry, device)
        write_json(root / 'initialization.json', dict(seed=2, identicalinit=True,
            trainable_sha256=parameter_hash(dict(selector.named_parameters())),
            geometry_sha256=file_hash(root / 'geometry_stats.json'), historical_checkpoint_loaded=False))
        del initial
    write_json(root / 'protocol.json', config)
    optimizer = torch.optim.AdamW(selector.parameters(), lr=PEAK_LR, weight_decay=1e-4)
    scaler = torch.amp.GradScaler('cuda', enabled=device.type == 'cuda')
    objective, runtime = RDropObjective(selector), Runtime(0, 1, device)
    history, best, successful, retries_total, state = [], None, 0, 0, None
    counts, cursors = Counter({p: 0 for p in PROBLEMS}), {p: 0 for p in PROBLEMS}
    if checkpoint is not None:
        optimizer.load_state_dict(checkpoint['optimizer'])
        for state in optimizer.state.values():
            for key, value in state.items():
                if torch.is_tensor(value):
                    state[key] = value.cpu() if key == 'step' else value.to(device)
        scaler.load_state_dict(checkpoint['scaler'])
        history, best = checkpoint['history'], checkpoint['best']
        successful, counts = checkpoint['successful_updates'], Counter(checkpoint['updates_by_problem'])
        retries_total, cursors = checkpoint['amp_retries'], checkpoint['stream_draws']
        if successful % EVAL_EVERY or successful != len(history) * EVAL_EVERY or any(
                cursors[p] != counts[p] * BATCH_SIZE for p in PROBLEMS):
            raise ValueError('Resume requires a complete evaluation boundary and exact successful stream cursors')
        restore_rank_rng(checkpoint['rng'], device)
    else:
        set_seed(2)
    del checkpoint
    caches = {p: open_arm_cache(args, manifest, 'train', p) for p in PROBLEMS}
    if any(cache.size != config['train_rows_per_problem'] for cache in caches.values()):
        raise ValueError('The two TRAIN populations must be exactly10000 vs70114 per problem')
    streams = {p: ShuffleStream(cache.size, task_seed(2, p), draws=cursors[p]) for p, cache in caches.items()}
    tracker = wandb_run(args, config)

    def checkpoint_state():
        return dict(model=selector.state_dict(), optimizer=optimizer.state_dict(), scaler=scaler.state_dict(),
            args=config, epoch=len(history) - 1, completed_evaluations=len(history),
            successful_updates=successful, updates_by_problem=dict(counts), amp_retries=retries_total,
            stream_draws={p: stream.draws for p, stream in streams.items()},
            rng=capture_rank_rng(device), history=history, best=best)

    if not args.resume:
        save_checkpoint(last, checkpoint_state())
    try:
        while successful < MAX_UPDATES:
            end = min(MAX_UPDATES, successful + EVAL_EVERY)
            requests = [(p, streams[p].take(), BATCH_SIZE) for p in order[successful:end]]
            interval_start, interval_first = time.monotonic(), successful
            weighted = torch.zeros(len(PART_KEYS), dtype=torch.float64, device=device)
            selector.train()
            if device.type == 'cuda':
                torch.cuda.reset_peak_memory_stats(device)
            with PinnedPrefetch(caches, requests, device) as batches:
                for problem, n_rank, n_global, batch in batches:
                    if n_rank != BATCH_SIZE or n_global != BATCH_SIZE:
                        raise ValueError('Every successful logical update must use exactly1024 real rows')
                    lr = learning_rate(successful + 1)
                    progress = min(1., (successful + 1) / WARMUP_UPDATES)
                    for group in optimizer.param_groups:
                        group['lr'] = lr
                    parts, retries = update_batch(objective, optimizer, scaler, batch, .35 * progress,
                        n_rank, n_global, runtime, microbatch_cap=caps[problem])
                    successful += 1
                    counts[problem] += 1
                    retries_total += retries
                    weighted.add_(parts.double())
                    if successful % args.log_every == 0 or successful == end:
                        done, elapsed = successful - interval_first, time.monotonic() - interval_start
                        averages = (weighted / done).cpu().tolist()
                        info = dict(arm=args.arm, successful_updates=successful, max_updates=MAX_UPDATES,
                            interval_updates=done, samples=done * BATCH_SIZE, seconds=elapsed,
                            seconds_per_update=elapsed / done, samples_per_second=done * BATCH_SIZE / elapsed,
                            problem=problem, lr=lr, consistency_weight=.35 * progress,
                            amp_retries=retries_total, updates_by_problem=dict(counts),
                            peak_allocated_gib=torch.cuda.max_memory_allocated(device) / 2**30 if device.type == 'cuda' else 0.,
                            **dict(zip(PART_KEYS, averages)))
                        with (root / 'updates.jsonl').open('a') as stream:
                            stream.write(json.dumps(dict(updated_utc=utc(), **info), allow_nan=False) + '\n')
                        write_json(root / 'status.json', dict(phase='training', updated_utc=utc(), **info))
                        print('[R77 train] ' + json.dumps(info), flush=True)
                        if tracker is not None:
                            tracker.log({**{'optimization/' + key: value for key, value in zip(PART_KEYS, averages)},
                                'successful_updates': successful, 'lr': lr, 'samples_per_second': info['samples_per_second'],
                                'gpu/peak_allocated_gib': info['peak_allocated_gib']}, step=successful)
            point = successful // EVAL_EVERY
            optimization = dict(zip(PART_KEYS, (weighted / (successful - interval_first)).cpu().tolist()))
            optimization.update(lr=lr, consistency_weight=.35 * progress, amp_retries=retries_total,
                samples=(successful - interval_first) * BATCH_SIZE, seconds=time.monotonic() - interval_start)
            validation = evaluate(selector, args, manifest, common_sbs, own_sbs, 'val', f'epoch{point:03d}')
            cost = validation['macro']['macro_vs_sbs_pct']
            if not math.isfinite(cost):
                raise ValueError('The common-SBS validation selection metric must be finite')
            training = evaluate(selector, args, manifest, common_sbs, own_sbs, 'train', f'epoch{point:03d}') if point % 5 == 0 else None
            row = dict(epoch=point, evaluation_point=point, successful_updates=successful,
                updates_by_problem=dict(counts), optimization=optimization, val=validation, train=training)
            history.append(row)
            improved = best is None or cost < best['val']['macro']['macro_vs_sbs_pct']
            if improved:
                best = dict(row)
            state = checkpoint_state()
            if improved:
                save_checkpoint(root / 'best.pt', state)
                write_json(root / 'best_eval.json', best)
            save_checkpoint(last, state)
            write_json(root / 'history.json', history)
            write_json(root / 'status.json', dict(phase='evaluation_complete', updated_utc=utc(),
                successful_updates=successful, max_updates=MAX_UPDATES, selected_update=best['successful_updates'],
                val=validation['macro'], own_sbs_val=validation['own_train_sbs']['macro']))
            if tracker is not None:
                values = dict(evaluation_point=point, successful_updates=successful)
                for split, value in (('val', validation), ('train_eval', training)):
                    if value is not None:
                        values.update({split + '/' + key: score for key, score in value['macro'].items()})
                        values.update({split + '/own_sbs/' + key: score
                            for key, score in value['own_train_sbs']['macro'].items()})
                        for problem, metrics in value['per_problem'].items():
                            values.update({split + '/' + problem + '/' + key: metrics[key]
                                for key in ('top1', 'top2', 'top3', 'mean_cost', 'vs_sbs_pct')})
                tracker.log(values, step=successful)
        expected = {p: MAX_UPDATES // len(PROBLEMS) for p in PROBLEMS}
        if dict(counts) != expected or successful != MAX_UPDATES:
            raise ValueError('Both arms must finish the exact same total and per-task successful-update budget')
        del objective, optimizer, scaler, selector, state
        selector, selected = load_checkpoint(root / 'best.pt', device)
        if selected['successful_updates'] != best['successful_updates'] or selected['args'] != config:
            raise ValueError('The unique validation best must match its checkpoint')
        if best['train'] is None:
            best = dict(best, train=evaluate(selector, args, manifest, common_sbs, own_sbs, 'train', 'selected_best'))
        for row in history:
            if row['successful_updates'] == best['successful_updates']:
                row['train'] = best['train']
        write_json(root / 'history.json', history)
        write_json(root / 'best_eval.json', best)
        lock = lock_best(root, best, config)
        result = dict(best=best, final=history[-1], evaluations_completed=len(history),
            successful_updates=successful, updates_by_problem=dict(counts), amp_retries=retries_total,
            stop_reason='fixed identical successful-update budget; no early stopping', test_read=False)
        write_json(root / 'result.json', result)
        write_json(root / 'final_metrics.json', dict(epoch=best['epoch'], selected_update=best['successful_updates'],
            train=best['train'], val=best['val'], checkpoint_sha256=lock['checkpoint_sha256'], test_read=False))
        write_json(root / 'training_complete.json', dict(completed_utc=utc(), successful_updates=successful,
            updates_by_problem=dict(counts), selected_best=lock, next_stage='one full locked TEST', test_read=False))
        write_json(root / 'status.json', dict(phase='training_complete', updated_utc=utc(),
            successful_updates=successful, selected_update=best['successful_updates'], test_read=False))
        if tracker is not None:
            tracker.log({'selected_train/' + key: value for key, value in best['train']['macro'].items()}, step=successful)
    finally:
        for cache in caches.values():
            cache.close()
        if tracker is not None:
            tracker.finish()


def lock_best(root, best, config):
    path = root / 'best.pt'
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    if checkpoint['successful_updates'] != best['successful_updates'] or checkpoint['args'] != config:
        raise PermissionError('Lock only the complete validation-selected checkpoint')
    record = dict(experiment='R77', arm=config['arm'], checkpoint=str(path.resolve()),
        checkpoint_sha256=file_hash(path), epoch=best['epoch'], selected_update=best['successful_updates'],
        rule=config['checkpoint_rule'], selected_without_test=True, per_problem_assembly=False,
        val=best['val']['macro'])
    lock = root / 'final_test_lock.json'
    if lock.exists():
        existing = read_json(lock)
        if any(existing.get(key) != value for key, value in record.items()):
            raise PermissionError('The locked unique best cannot change')
        return existing
    with lock.open('x') as stream:
        json.dump(dict(record, locked_utc=utc()), stream, indent=2, allow_nan=False)
    return record


def authorize_test(args):
    root = args.root / GROUPS[args.arm]
    complete = read_json(root / 'training_complete.json') if (root / 'training_complete.json').exists() else {}
    if complete.get('successful_updates') != MAX_UPDATES:
        raise PermissionError('Complete the fixed budget before opening any TEST input')
    lock, config = read_json(root / 'final_test_lock.json'), read_json(root / 'protocol.json')
    if (lock.get('experiment') != 'R77' or lock.get('arm') != args.arm or not lock.get('selected_without_test')
            or Path(lock['checkpoint']).resolve() != (root / 'best.pt').resolve()
            or lock['checkpoint_sha256'] != file_hash(root / 'best.pt')
            or config['max_successful_updates'] != MAX_UPDATES
            or Path(config['data_root']).resolve() != args.data_root.resolve()):
        raise PermissionError('TEST requires the unchanged unique paired checkpoint')
    if (root / 'test_started.json').exists() or (root / 'test_results.json').exists():
        raise PermissionError('The single full TEST has already been claimed; never automatically repeat it')
    return lock, config


def test(args, device):
    lock, config = authorize_test(args)
    root = args.root / GROUPS[args.arm]
    selector, checkpoint = load_checkpoint(root / 'best.pt', device)
    if checkpoint['args'] != config or checkpoint['successful_updates'] != lock['selected_update']:
        raise PermissionError('TEST checkpoint/protocol differs from the locked selection')
    with (root / 'test_started.json').open('x') as stream:
        json.dump(dict(started_utc=utc(), checkpoint_sha256=lock['checkpoint_sha256'], previous_full_tests=0), stream)
    # Existing R76 TEST arrays are only opened after the R77 lock and one-shot claim.
    manifest = read_json(args.root / 'source_cache_manifest.json')
    result = evaluate(selector, args, manifest, config['baselines'], config['own_train_sbs'], 'test', 'selected_best')
    if file_hash(root / 'best.pt') != lock['checkpoint_sha256']:
        raise PermissionError('Checkpoint changed during the single full TEST')
    write_json(root / 'test_results.json', dict(result=result, epoch=lock['epoch'], selected_update=lock['selected_update'],
        checkpoint_sha256=lock['checkpoint_sha256'], complete=True, completed_utc=utc()))
    write_json(root / 'test_complete.json', dict(full_tests=1, completed_utc=utc(),
        checkpoint_sha256=lock['checkpoint_sha256']))
    write_json(root / 'status.json', dict(phase='test_complete', updated_utc=utc(), successful_updates=MAX_UPDATES,
        selected_update=lock['selected_update'], test=result['macro']))


def analyze_arm(args):
    from .r76_analysis import fixed_baselines, method_observations, metric_rows, plot_history, write_csv
    root = args.root / GROUPS[args.arm]
    if not (root / 'test_complete.json').is_file():
        raise ValueError('Analysis reads saved predictions only, after the one completed TEST')
    outcome, config, test_result = (read_json(root / name) for name in ('result.json', 'protocol.json', 'test_results.json'))
    if not test_result.get('complete'):
        raise ValueError('No analysis from a partial TEST')
    best, history = outcome['best'], read_json(root / 'history.json')
    rows = []
    for split, result in (('train', best['train']), ('val', best['val']), ('test', test_result['result'])):
        for reference, value in (('common_full_train_sbs', result), ('own_train_sbs', result['own_train_sbs'])):
            for row in metric_rows(split, value, best['epoch']):
                rows.append(dict(row, model=f'R77{args.arm}', sbs_reference=reference,
                    selected_update=best['successful_updates'], actual_updates=outcome['successful_updates']))
    write_csv(root / 'comparison.csv', rows)
    write_csv(root / 'fixed_baselines.csv', fixed_baselines(root, config['baselines']))
    method_observations(root, best, test_result, config['baselines'])
    plot_history(root, history)
    write_json(root / 'pipeline_complete.json', dict(experiment='R77', arm=args.arm, completed_utc=utc(),
        successful_updates=outcome['successful_updates'], selected_update=best['successful_updates'], full_tests=1))
    write_json(root / 'status.json', dict(phase='complete', updated_utc=utc(), successful_updates=MAX_UPDATES,
        selected_update=best['successful_updates'], test=test_result['result']['macro']))
    print(f'[R77 complete] {args.arm}: {root}', flush=True)


def pair_summary(args):
    roots = {arm: args.root / name for arm, name in GROUPS.items()}
    if not all((root / 'pipeline_complete.json').exists() for root in roots.values()):
        return
    values = {arm: read_json(root / 'test_results.json') for arm, root in roots.items()}
    configs = {arm: read_json(root / 'protocol.json') for arm, root in roots.items()}
    for key in ('common_trainable_sha256', 'task_order_sha256', 'lr', 'max_successful_updates', 'baselines'):
        if configs['A'][key] != configs['B'][key]:
            raise ValueError(f'Paired invariant changed: {key}')
    rows = []
    scopes = [('ALL', 'ALL')] + [('family', family) for family in ('TSP', 'CVRP', 'ATSP', 'MVRP')]
    scopes += [('problem', problem) for problem in PROBLEMS]
    fields = dict(top1_pct=('top1', 100.), top2_pct=('top2', 100.), top3_pct=('top3', 100.),
        ce=('ce', 1.), mean_cost=('mean_cost', 1.), regret_pct=('actual_regret_pct', 1.),
        vs_common_sbs_pct=('vs_sbs_pct', 1.))
    for scope_type, scope in scopes:
        row = dict(scope_type=scope_type, scope=scope)
        for arm in GROUPS:
            result = values[arm]['result']
            metrics = result['macro'] if scope_type == 'ALL' else result['families'][scope] if (
                scope_type == 'family') else result['per_problem'][scope]
            prefix = '' if scope_type == 'problem' else 'macro_'
            tasks = list(PROBLEMS) if scope_type == 'ALL' else list(PROBLEMS[3:]) if scope == 'MVRP' else [scope]
            row[arm + '_n'] = sum(result['per_problem'][p]['n'] for p in tasks)
            for label, (key, factor) in fields.items():
                row[arm + '_' + label] = factor * metrics[prefix + key]
        for label in fields:
            row['B_minus_A_' + label] = row['B_' + label] - row['A_' + label]
        rows.append(row)
    from .r76_analysis import write_csv
    write_csv(args.root / 'paired_test_comparison.csv', rows)
    write_json(args.root / 'paired_results.json', dict(experiment='R77', actual_updates_per_arm=MAX_UPDATES,
        common_trainable_sha256=configs['A']['common_trainable_sha256'],
        common_lr=configs['A']['lr'], complete=True, full_tests_per_arm=1,
        comparison=rows, selected_updates={arm: value['selected_update'] for arm, value in values.items()},
        interpretation='paired seed2 sample-size comparison; historical R76 is not the matched control'))
    lines = ['# R77 Locked TEST Rows', '', 'One full TEST per arm after fixed49680 updates and unique VAL-best locks.', '',
        '| Type | Scope | N/arm | A Top1 % | B Top1 % | A CE | B CE | A regret % | B regret % | A vs common SBS % | B vs common SBS % |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for row in rows:
        lines.append('| {scope_type} | {scope} | {A_n} | {A_top1_pct:.4f} | {B_top1_pct:.4f} | '
            '{A_ce:.6f} | {B_ce:.6f} | {A_regret_pct:.6f} | {B_regret_pct:.6f} | '
            '{A_vs_common_sbs_pct:.6f} | {B_vs_common_sbs_pct:.6f} |'.format(**row))
    (args.root / 'TEST_ROWS.md').write_text('\n'.join(lines) + '\n')


def diagnostic_handoff(args):
    """Preserve locked TRAIN/VAL predictions without inference or TEST access."""
    destination = args.root / 'selected_train_val_predictions'
    destination.mkdir(parents=True, exist_ok=True)
    arms = {}
    for arm, name in GROUPS.items():
        root = args.root / name
        if not (root / 'training_complete.json').exists():
            arms[arm] = dict(ready=False, reason='waiting for fixed budget and unique VAL-best lock')
            continue
        lock = read_json(root / 'final_test_lock.json')
        best = read_json(root / 'best_eval.json')
        if best['successful_updates'] != lock['selected_update'] or not lock['selected_without_test']:
            raise PermissionError('Export only the final locked validation-selected TRAIN/VAL predictions')
        directory = destination / arm
        directory.mkdir(parents=True, exist_ok=True)
        manifest_path = directory / 'manifest.json'
        if manifest_path.exists():
            record = read_json(manifest_path)
            if (record['checkpoint_sha256'] != lock['checkpoint_sha256']
                    or record['selected_update'] != lock['selected_update']):
                raise PermissionError('A diagnostic handoff cannot switch its locked checkpoint')
            arms[arm] = record
            continue
        files = {}
        for split in ('train', 'val'):
            tag = f"epoch{best['epoch']:03d}" if split == 'val' or best['epoch'] % 5 == 0 else 'selected_best'
            source = root / 'evaluation' / split / tag
            metrics = read_json(source / 'metrics.json')
            for filename, suffix in (('rank000.npz', '.npz'), ('metrics.json', '_metrics.json')):
                target, original = directory / (split + suffix), source / filename
                if target.exists():
                    if not os.path.samefile(target, original):
                        raise PermissionError(f'Never replace an existing prediction handoff: {target}')
                else:
                    os.link(original, target)
            files[split] = dict(path=str((directory / (split + '.npz')).resolve()),
                source_path=str((source / 'rank000.npz').resolve()),
                sha256=file_hash(source / 'rank000.npz'),
                metrics_path=str((directory / (split + '_metrics.json')).resolve()),
                rows=metrics['native_total'],
                rows_per_problem={p: metrics['per_problem'][p]['n'] for p in PROBLEMS})
        config = read_json(root / 'protocol.json')
        record = dict(ready=True, arm=arm, saved_utc=utc(), selected_update=lock['selected_update'],
            evaluation_point=best['epoch'], completed_successful_updates=MAX_UPDATES,
            checkpoint=lock['checkpoint'], checkpoint_sha256=lock['checkpoint_sha256'],
            selected_without_test=True, data_root=config['data_root'], files=files,
            common_train_sbs=config['baselines'], own_train_sbs=config['own_train_sbs'],
            geometry_normalization=str((root / 'geometry_stats.json').resolve()))
        write_json(manifest_path, record)
        arms[arm] = record
    record = dict(experiment='R77', saved_utc=utc(), complete=all(v['ready'] for v in arms.values()),
        input_splits=['train', 'val'], inference_performed=False, train_rerun=False,
        test_inputs_read=False, storage='hard links to already saved evaluation files; no copied dataset',
        array_key_prefix='<problem>__',
        arrays=dict(logits='raw scores, columns in pool slot order; not calibrated probabilities',
            pool='solver names in score/cost column order', pool_ids='global solver IDs, one per pool slot',
            costs='native unnormalized R76 cached solver cost matrix, not geometric normalization',
            winner='original winner in local pool slot space', nodes='valid node counts, not coordinates',
            indices='local evaluation row IDs', source_indices='row IDs in the original R76 split'),
        source_cache_manifest=str((args.root / 'source_cache_manifest.json').resolve()),
        shared_input_cache=str((DEFAULT_SOURCE / 'cache').resolve()), arms=arms)
    write_json(destination / 'manifest.json', record)
    print('[R77 diagnostic handoff] ' + json.dumps(dict(complete=record['complete'],
        ready_arms=[arm for arm, value in arms.items() if value['ready']],
        manifest=str((destination / 'manifest.json').resolve()))), flush=True)
    return record


def diagnostic_watch(args):
    while not diagnostic_handoff(args)['complete']:
        time.sleep(30)


def status(args):
    for arm, name in GROUPS.items():
        path = args.root / name / 'status.json'
        value = read_json(path) if path.exists() else dict(phase='prepared' if (args.root / 'prepared.json').exists() else 'not_prepared')
        print(json.dumps({'arm': arm, **value}, allow_nan=False))


def live_gpu_status():
    uuids = ('GPU-71479e7e-f8bd-e430-66ed-f8db9f0d47be', 'GPU-597c682a-4b7f-7609-80a6-8ced791f5c5d')
    command = ('set -e\nsource /public/home/shiys/miniconda3/etc/profile.d/conda.sh\nconda activate easynco\n'
        'nvidia-smi --id=' + ','.join(uuids) +
        ' --query-gpu=uuid,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits')
    try:
        result = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=5', 'gpu05', command],
            capture_output=True, text=True, timeout=20, check=True)
        rows = {}
        for uuid, util, used, total in csv.reader(result.stdout.splitlines()):
            uuid = uuid.strip()
            if uuid not in uuids:
                raise ValueError('GPU telemetry returned a card outside the approved allocation')
            rows[uuid] = dict(utilization_pct=int(util), memory_used_mib=int(used), memory_total_mib=int(total))
        if set(rows) != set(uuids):
            raise ValueError('GPU telemetry did not return both approved UUIDs')
        return dict(observed_utc=utc(), source='read-only nvidia-smi on approved UUIDs; no GPU workload', by_uuid=rows)
    except (subprocess.SubprocessError, OSError, ValueError) as error:
        return dict(observed_utc=utc(), unavailable=True, error=type(error).__name__, by_uuid={})


def train_curve_assessment(history):
    points = []
    for row in history:
        if row.get('train') is None:
            continue
        train, val = row['train']['macro'], row['val']['macro']
        points.append(dict(successful_updates=row['successful_updates'],
            train_top1_pct=100 * train['macro_top1'], val_top1_pct=100 * val['macro_top1'],
            train_ce=train['macro_ce'], val_ce=val['macro_ce'],
            val_minus_train_ce=val['macro_ce'] - train['macro_ce'],
            train_regret_pct=train['macro_actual_regret_pct'], val_regret_pct=val['macro_actual_regret_pct']))
    changes = [dict(from_update=a['successful_updates'], to_update=b['successful_updates'],
        train_ce_delta=b['train_ce'] - a['train_ce'], val_ce_delta=b['val_ce'] - a['val_ce'],
        gap_delta=b['val_minus_train_ce'] - a['val_minus_train_ce']) for a, b in zip(points, points[1:])]
    divergence = [change['train_ce_delta'] < 0 and change['val_ce_delta'] > 0 for change in changes]
    if len(points) < 2:
        judgment = 'insufficient full TRAIN curve points; no longitudinal overfit judgment'
    elif len(divergence) >= 2 and all(divergence[-2:]):
        judgment = 'two consecutive TRAIN-CE-down / VAL-CE-up intervals suggest overfitting; reporting only'
    elif divergence[-1]:
        judgment = 'latest TRAIN-CE-down / VAL-CE-up interval suggests divergence; not yet persistent'
    elif changes[-1]['gap_delta'] > 0 and changes[-1]['val_ce_delta'] < 0:
        judgment = 'generalization gap widening but VAL CE still improving; no persistent overfit evidence'
    else:
        judgment = 'no persistent TRAIN-CE-down / VAL-CE-up evidence in measured intervals'
    return dict(points=points, changes=changes, judgment=judgment, reporting_only=True,
        caveat='Own TRAIN populations differ; compare TRAIN/VAL trends within each arm, not raw TRAIN metrics across arms. '
            'Sparse, adaptively reused VAL diagnostics do not change LR, budget, checkpoint selection or TEST.')


def monitoring_snapshot(args):
    """Small phase receipt; evaluation status retains the last real update counter."""
    arms, validation_points, curves = {}, {}, {}
    metric_keys = ('macro_top1', 'macro_actual_regret_pct', 'macro_vs_sbs_pct', 'macro_ce')
    for arm, name in GROUPS.items():
        root = args.root / name
        state, launch = read_json(root / 'status.json'), read_json(root / 'launch_receipt.json')
        history = read_json(root / 'history.json') if (root / 'history.json').exists() else []
        curves[arm] = train_curve_assessment(history)
        updates = root / 'updates.jsonl'
        last_train = json.loads(updates.read_text().splitlines()[-1]) if updates.exists() else {}
        val_paths = sorted((root / 'evaluation/val').glob('epoch[0-9][0-9][0-9]/metrics.json'))
        validation_points[arm] = {int(path.parent.name[5:]) * EVAL_EVERY: read_json(path)['macro'] for path in val_paths}
        latest_val = read_json(val_paths[-1]) if val_paths else None
        best = read_json(root / 'best_eval.json') if (root / 'best_eval.json').exists() else None
        arms[arm] = dict(phase=state['phase'], updated_utc=state['updated_utc'],
            successful_updates=state.get('successful_updates', last_train.get('successful_updates', 0)),
            max_successful_updates=MAX_UPDATES, amp_retries=last_train.get('amp_retries'),
            seconds_per_update=last_train.get('seconds_per_update'), gpu_uuid=launch['gpu_uuid'],
            gpu_name=launch['gpu_name'], peak_train_allocated_gib=last_train.get('peak_allocated_gib'),
            gpu_metric_source='last actual training log torch.cuda.max_memory_allocated; not live NVML utilization',
            latest_val_update=int(val_paths[-1].parent.name[5:]) * EVAL_EVERY if val_paths else None,
            latest_val_macro={key: latest_val['macro'][key] for key in metric_keys} if latest_val else None,
            current_best_update=best['successful_updates'] if best else None,
            current_best_vs_sbs_pct=best['val']['macro']['macro_vs_sbs_pct'] if best else None,
            test_started=(root / 'test_started.json').exists(),
            training_complete=(root / 'training_complete.json').exists(),
            best_locked=(root / 'final_test_lock.json').exists(),
            test_complete=(root / 'test_complete.json').exists(),
            complete=(root / 'pipeline_complete.json').exists())
        if (root / 'final_test_lock.json').exists():
            arms[arm]['locked_best'] = read_json(root / 'final_test_lock.json')
        train_seconds = [row['train']['seconds'] for row in history if row.get('train') is not None]
        val_seconds = [row['val']['seconds'] for row in history]
        train_remaining = max(0, MAX_UPDATES - arms[arm]['successful_updates']) * last_train.get('seconds_per_update', 0.)
        remaining_val = max(0, MAX_UPDATES // EVAL_EVERY - len(val_paths))
        periodic_train_done = sum(row.get('train') is not None and row['epoch'] % 5 == 0 for row in history)
        periodic_train_remaining = max(0, MAX_UPDATES // EVAL_EVERY // 5 - periodic_train_done)
        eval_remaining = remaining_val * (float(np.mean(val_seconds)) if val_seconds else 0.)
        eval_remaining += (periodic_train_remaining + 1) * (float(np.mean(train_seconds)) if train_seconds else 0.)
        arms[arm]['estimated_training_remaining_seconds'] = train_remaining
        arms[arm]['estimated_remaining_with_evaluations_seconds'] = train_remaining + eval_remaining + 60.
        arms[arm]['eta_caveat'] = 'Measured update/evaluation times; allowance for one selected-best TRAIN and60s TEST/analysis. Not a guarantee.'
    common = sorted(set(validation_points['A']) & set(validation_points['B']))
    common_rows = []
    for update in common:
        row = dict(successful_updates=update)
        for label, key, factor in (('top1_pct', 'macro_top1', 100.), ('ce', 'macro_ce', 1.),
                ('regret_pct', 'macro_actual_regret_pct', 1.), ('vs_sbs_pct', 'macro_vs_sbs_pct', 1.)):
            for arm in GROUPS:
                row[arm + '_' + label] = factor * validation_points[arm][update][key]
            row['B_minus_A_' + label] = row['B_' + label] - row['A_' + label]
        common_rows.append(row)
    if common_rows:
        with (args.root / 'common_update_comparison.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(common_rows[0]))
            writer.writeheader()
            writer.writerows(common_rows)
    receipt = dict(recorded_utc=utc(), experiment='R77', arms=arms,
        monitor=read_json(args.root / 'monitor_status.json')['phase'],
        common_update_comparison=common_rows[-1] if common_rows else None, train_curves=curves,
        policy='Fixed49680 successful updates per arm; unique common-SBS VAL best; no early TEST or rule changes')
    if getattr(args, 'live_gpu', False):
        receipt['live_gpu'] = live_gpu_status()
        for arm in GROUPS:
            arms[arm]['live_gpu'] = receipt['live_gpu']['by_uuid'].get(arms[arm]['gpu_uuid'])
    write_json(args.root / 'latest_phase_snapshot.json', receipt)
    root_status = dict(receipt, root=str(args.root), phase='complete' if all(a['complete'] for a in arms.values()) else
        'failed' if receipt['monitor'] == 'failed' else 'running',
        continuation=dict(tmux_session='R77_sample_size_batch1024', job_id=1475, node='gpu05',
            command='bash /public/home/shiys/0selection2/code/V4/run_v4_r77.sh snapshot',
            instructions='Read this root STATUS and per-arm status/history/locks. Leave healthy tmux processes running; '
                'no new preflight, cache build, capacity scan or solver execution. Do not restart a healthy run.',
            resume_source='A resumed its own saved R77 update2484 state; not historical weights and not a Git commit.'),
        fixed_protocol=dict(successful_updates_per_arm=MAX_UPDATES, logical_batch=BATCH_SIZE, warmup_updates=WARMUP_UPDATES,
            val_interval=EVAL_EVERY, train_eval_every_val_points=5, early_stop=False,
            lr='fixed warmup then cosine1e-4 to1e-6 on successful updates',
            checkpoint='unique minimum VAL macro_vs_sbs_pct using shared FULL TRAIN SBS; never TEST',
            full_tests_per_arm=1), git_commits_made_by_R77=0)
    write_json(args.root / 'STATUS.json', root_status)
    lines = ['# R77 Root Status', '', 'Updated UTC: ' + receipt['recorded_utc'], '',
        'Phase: ' + root_status['phase'] + '. Leave healthy training/monitor tmux processes running.', '',
        '| Arm | Phase | Successful updates | Latest complete VAL | Top1 % | CE | Regret % | Current best | TEST complete |',
        '|---|---|---:|---:|---:|---:|---:|---:|---|']
    for arm, value in arms.items():
        metrics = value['latest_val_macro'] or {}
        lines.append('| {} | {} | {} | {} | {:.4f} | {:.6f} | {:.6f} | {} | {} |'.format(
            arm, value['phase'], value['successful_updates'], value['latest_val_update'],
            100 * metrics.get('macro_top1', 0.), metrics.get('macro_ce', 0.), metrics.get('macro_actual_regret_pct', 0.),
            value['current_best_update'], value['test_complete']))
    if common_rows:
        row = common_rows[-1]
        lines += ['', '## Latest Common Update', '', 'Update: ' + str(row['successful_updates']), '',
            '| Arm | Top1 % | CE | Regret % | vs common FULL TRAIN SBS % |', '|---|---:|---:|---:|---:|']
        for arm in GROUPS:
            lines.append(f'| {arm} | {row[arm+"_top1_pct"]:.4f} | {row[arm+"_ce"]:.6f} | '
                f'{row[arm+"_regret_pct"]:.6f} | {row[arm+"_vs_sbs_pct"]:.6f} |')
    lines += ['', '## TRAIN Curve', '']
    for arm, curve in curves.items():
        lines.append(arm + ': ' + curve['judgment'] + '.')
        for point in curve['points']:
            lines.append('- Update{successful_updates}: TRAIN/VAL Top1 {train_top1_pct:.4f}/{val_top1_pct:.4f}%; '
                'CE {train_ce:.6f}/{val_ce:.6f}; regret {train_regret_pct:.6f}/{val_regret_pct:.6f}%.'.format(**point))
    lines += ['', '## Continuation', '', root_status['continuation']['instructions'], '',
        root_status['continuation']['resume_source'], '',
        'Both arms must finish49680 successful updates. TEST follows each immutable own VAL-best lock once; '
        'never use TEST for rules/checkpoint selection. No Git commits.', '',
        'Per-arm launch_receipt.json identifies live GPU process PIDs. Controller and monitor remain in '
        'tmux R77_sample_size_batch1024; snapshot receipts and STATUS refresh on a CPU-only reporter.', '',
        'Once both analyses finish, TEST_ROWS.md / paired_test_comparison.csv include ALL, four families and all18 tasks.']
    (args.root / 'STATUS.md').write_text('\n'.join(lines) + '\n')
    with (args.root / 'phase_receipts.jsonl').open('a') as stream:
        stream.write(json.dumps(receipt, allow_nan=False) + '\n')
    print(json.dumps(receipt, allow_nan=False), flush=True)
    return receipt


def reporting_watch(args):
    """Observe only: root handoff every5 minutes, complete summaries after existing TESTs."""
    with (args.root / 'reporting.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        write_json(args.root / 'reporter_receipt.json', dict(pid=os.getpid(), started_utc=utc(), poll_seconds=300,
            actions='read existing logs/metrics and approved GPU telemetry; never control or pause training'))
        while True:
            receipt = monitoring_snapshot(args)
            if all(value['complete'] for value in receipt['arms'].values()):
                pair_summary(args)
                validation_overview(args)
                write_json(args.root / 'delivery_ready.json', dict(completed_utc=utc(), successful_updates_per_arm=MAX_UPDATES,
                    test_and_analysis_complete=True, test_rows='TEST_ROWS.md', status='STATUS.json',
                    full_tests_per_arm=1, git_commits_made_by_R77=0))
                return
            time.sleep(300)


def finalize_delivery(args):
    """Wait for existing summary writers to exit; render complete rows from saved results."""
    with (args.root / 'delivery.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        while True:
            arms_done = all((args.root / name / 'pipeline_complete.json').exists() for name in GROUPS.values())
            exit_path = args.root / 'launcher_exit_code'
            controller_done = exit_path.exists() and exit_path.read_text().strip() == '0'
            writers_alive = False
            for path, key in ((args.root / 'monitor_status.json', 'monitor_pid'),
                              (args.root / 'reporter_receipt.json', 'pid')):
                if path.exists():
                    try:
                        os.kill(int(read_json(path)[key]), 0)
                        writers_alive = True
                    except ProcessLookupError:
                        pass
            if arms_done and controller_done and not writers_alive:
                for name in GROUPS.values():
                    root = args.root / name
                    train_result = read_json(root / 'training_complete.json')
                    test_result = read_json(root / 'test_results.json')
                    lock_record = read_json(root / 'final_test_lock.json')
                    if (train_result['successful_updates'] != MAX_UPDATES or not test_result['complete']
                            or read_json(root / 'test_complete.json')['full_tests'] != 1
                            or test_result['checkpoint_sha256'] != lock_record['checkpoint_sha256']
                            or file_hash(root / 'best.pt') != lock_record['checkpoint_sha256']
                            or set(test_result['result']['per_problem']) != set(PROBLEMS)
                            or any(test_result['result']['per_problem'][p]['n'] != 3895 for p in PROBLEMS)):
                        raise ValueError('Final saved-result counters, unique locks or full TEST coverage disagree')
                pair_summary(args)
                validation_overview(args)
                receipt = monitoring_snapshot(args)
                write_json(args.root / 'FINAL_DELIVERY.json', dict(completed_utc=utc(), phase='complete',
                    actual_updates_per_arm=MAX_UPDATES, full_tests_per_arm=1, test_scope_rows=23,
                    prior_summary_writers_exited=True, repeated_model_inference=False,
                    all_tasks_and_families='TEST_ROWS.md', status='STATUS.json',
                    common_update_comparison=receipt['common_update_comparison'], git_commits_made_by_R77=0))
                return
            time.sleep(30)


def final_comparison(args):
    """Long-form saved TRAIN/VAL/locked TEST, with the actual common update axis."""
    rows = []
    for arm, name in GROUPS.items():
        root = args.root / name
        config, outcome = read_json(root / 'protocol.json'), read_json(root / 'result.json')
        history, lock = read_json(root / 'history.json'), read_json(root / 'final_test_lock.json')
        test_result = read_json(root / 'test_results.json')
        entries = [(split, row['successful_updates'], row[split]) for row in history
            for split in ('train', 'val') if row.get(split) is not None]
        entries.append(('test', lock['selected_update'], test_result['result']))
        for split, update, result in entries:
            for reference, value in (('common_full_train', result), ('own_train_reporting_only', result['own_train_sbs'])):
                scopes = [('ALL', 'ALL', list(PROBLEMS))]
                scopes += [('family', family, list(PROBLEMS[3:]) if family == 'MVRP' else [family])
                    for family in ('TSP', 'CVRP', 'ATSP', 'MVRP')]
                scopes += [('problem', problem, [problem]) for problem in PROBLEMS]
                for scope_type, scope, tasks in scopes:
                    metrics = value['macro'] if scope_type == 'ALL' else value['families'][scope] if (
                        scope_type == 'family') else value['per_problem'][scope]
                    prefix = '' if scope_type == 'problem' else 'macro_'
                    n = sum(value['per_problem'][p]['n'] for p in tasks)
                    correct = sum(value['per_problem'][p]['native_correct'] for p in tasks)
                    problem_scope = scope_type == 'problem'
                    rows.append(dict(experiment='R77', arm=arm, split=split, scope_type=scope_type, scope=scope,
                        common_updates=update, checkpoint_update=update, selected_checkpoint_update=lock['selected_update'],
                        selected_best=update == lock['selected_update'], actual_successful_updates=outcome['successful_updates'],
                        common_update_budget=MAX_UPDATES, logical_batch=BATCH_SIZE, updates_per_problem=MAX_UPDATES // len(PROBLEMS),
                        train_rows_per_problem=config['train_rows_per_problem'],
                        train_unique_rows_total=config['train_rows_per_problem'] * len(PROBLEMS),
                        common_full_train_rows_per_problem=FULL_ROWS,
                        train_sample_exposures_per_problem=MAX_UPDATES // len(PROBLEMS) * BATCH_SIZE,
                        n=n, native_correct=correct, native_top1_pct=100. * correct / n,
                        top1_tie_aware_pct=100. * metrics[prefix + 'top1_tie_aware'],
                        selector_top2_pct=100. * metrics[prefix + 'top2'], selector_top3_pct=100. * metrics[prefix + 'top3'],
                        ce=metrics[prefix + 'ce'], mean_cost=metrics[prefix + 'mean_cost'],
                        actual_regret_pct=metrics[prefix + 'actual_regret_pct'], train_fixed_sbs_reference=reference,
                        vs_train_fixed_sbs_pct=metrics[prefix + 'vs_sbs_pct'],
                        train_fixed_sbs_cost=float(np.mean([value['per_problem'][p]['sbs_cost'] for p in tasks])),
                        common_train_sbs_solver=config['baselines'][scope]['sbs_solver'] if problem_scope else 'per_problem_train_fixed',
                        own_train_sbs_solver=config['own_train_sbs'][scope]['sbs_solver'] if problem_scope else 'per_problem_train_fixed',
                        fixed_sbs_solver=metrics['sbs_name'] if problem_scope else 'per_problem_train_fixed',
                        data_subset='input-metadata stratified10000 subset of FULL70114' if arm == 'A' else 'FULL70114',
                        training_data_root=config['data_root'], checkpoint_sha256=lock['checkpoint_sha256'],
                        evaluation_stage='locked_best_test_once' if split == 'test' else 'saved_train_val',
                        test_used_for_selection=False))
    temporary = args.root / f'.finalcomparison.{os.getpid()}.tmp'
    with temporary.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(args.root / 'finalcomparison.csv')
    from .r76_analysis import plt
    fig, axes = plt.subplots(2, 2, figsize=(11, 9), constrained_layout=True)
    for axis, (key, label) in zip(axes.flat, (('native_top1_pct', 'Native Top1 (%)'), ('ce', 'CE'),
            ('actual_regret_pct', 'Actual regret (%)'), ('vs_train_fixed_sbs_pct', 'vs common TRAIN SBS (%)'))):
        for arm, color in (('A', '#237c68'), ('B', '#b84d48')):
            for split, style in (('val', '-'), ('train', '--')):
                selected = sorted((row for row in rows if row['arm'] == arm and row['split'] == split
                    and row['scope_type'] == 'ALL' and row['train_fixed_sbs_reference'] == 'common_full_train'),
                    key=lambda row: row['common_updates'])
                axis.plot([row[key] for row in selected], [row['common_updates'] for row in selected],
                    style, color=color, label=arm + ' ' + split, marker='.' if split == 'train' else None)
        axis.set(xlabel=label, ylabel='Common successful updates', ylim=(0, MAX_UPDATES))
        axis.grid(alpha=.2)
        axis.legend()
    fig.savefig(args.root / 'COMMON_UPDATE_CURVES.png', dpi=160)
    plt.close(fig)
    write_json(args.root / 'finalcomparison_schema.json', dict(rows=len(rows), scopes_per_split=23,
        includes=['ALL', 'four families', '18 problems', 'all saved TRAIN/VAL points', 'one locked TEST per arm'],
        common_updates='Actual successful update of evaluated checkpoint; TEST uses the unique VAL-best checkpoint update',
        actual_successful_updates='Both completed49680; selected checkpoint may be earlier',
        sbs='common FULL TRAIN fitted identity plus separate own TRAIN reporting-only reference',
        counts='Unique TRAIN rows are10000 vs70114/problem; each arm exposes2826240 samples/problem',
        native_top1='native_correct/n, strict original FP64 winner identity; tie-aware accuracy is separate',
        macro_sbs='equal-task mean of per-problem percent ratios; not ratio of aggregate costs',
        train_comparison='TRAIN populations differ; use within-arm TRAIN/VAL curves, and common-update VAL for paired comparison',
        curve='COMMON_UPDATE_CURVES.png uses common successful updates on the vertical axis'))
    return len(rows)


def approved_gpu_release():
    telemetry = live_gpu_status()
    uuids = ('GPU-71479e7e-f8bd-e430-66ed-f8db9f0d47be', 'GPU-597c682a-4b7f-7609-80a6-8ced791f5c5d')
    command = ('set -e\nsource /public/home/shiys/miniconda3/etc/profile.d/conda.sh\nconda activate easynco\n'
        'nvidia-smi --id=' + ','.join(uuids) +
        ' --query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory --format=csv,noheader,nounits')
    try:
        result = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=5', 'gpu05', command],
            capture_output=True, text=True, timeout=20, check=True)
        processes = []
        for row in csv.reader(result.stdout.splitlines()):
            if not row:
                continue
            uuid, pid, process, memory = row
            if uuid.strip() in uuids:
                processes.append(dict(gpu_uuid=uuid.strip(), pid=int(pid), process=process.strip(), used_memory_mib=memory.strip()))
        idle = not telemetry.get('unavailable') and not processes and all(
            record['utilization_pct'] == 0 for record in telemetry['by_uuid'].values())
        return dict(verified=idle, observed_utc=utc(), gpu_telemetry=telemetry,
            compute_processes=processes, automatic_gpu_reassignment_performed=False)
    except (subprocess.SubprocessError, OSError, ValueError) as error:
        return dict(verified=False, observed_utc=utc(), error=type(error).__name__, gpu_telemetry=telemetry)


def success_watch(args):
    """Publish canonical success only after complete pipelines and idle approved GPUs."""
    with (args.root / 'success.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        while True:
            done = all((args.root / name / 'pipeline_complete.json').exists() for name in GROUPS.values())
            exit_path = args.root / 'launcher_exit_code'
            if done and exit_path.exists() and exit_path.read_text().strip() == '0':
                selected, test_macro = {}, {}
                for arm, name in GROUPS.items():
                    root = args.root / name
                    complete, test_value = read_json(root / 'training_complete.json'), read_json(root / 'test_results.json')
                    locked, history = read_json(root / 'final_test_lock.json'), read_json(root / 'history.json')
                    minimum = min(history, key=lambda row: row['val']['macro']['macro_vs_sbs_pct'])
                    if (complete['successful_updates'] != MAX_UPDATES
                            or any(complete['updates_by_problem'].get(p) != MAX_UPDATES // len(PROBLEMS) for p in PROBLEMS)
                            or len(history) != MAX_UPDATES // EVAL_EVERY
                            or [row['successful_updates'] for row in history] != list(range(EVAL_EVERY, MAX_UPDATES + 1, EVAL_EVERY))
                            or minimum['successful_updates'] != locked['selected_update']
                            or not locked['selected_without_test'] or not test_value['complete']
                            or read_json(root / 'test_complete.json')['full_tests'] != 1
                            or test_value['checkpoint_sha256'] != locked['checkpoint_sha256']
                            or file_hash(root / 'best.pt') != locked['checkpoint_sha256']
                            or set(test_value['result']['per_problem']) != set(PROBLEMS)
                            or any(test_value['result']['per_problem'][p]['n'] != 3895 for p in PROBLEMS)):
                        raise ValueError('Saved full budget, unique validation minimum or one-shot full TEST disagrees')
                    selected[arm], test_macro[arm] = locked, test_value['result']['macro']
                gpu = approved_gpu_release()
                if gpu['verified']:
                    count = final_comparison(args)
                    snapshot = monitoring_snapshot(args)
                    record = dict(experiment='R77', phase='complete', completed_utc=utc(),
                        actual_successful_updates_per_arm=MAX_UPDATES, updates_per_problem=MAX_UPDATES // len(PROBLEMS),
                        train_rows_per_problem=dict(A=SUBSET_ROWS, B=FULL_ROWS), full_tests_per_arm=1,
                        unique_val_best=selected, test_macro=test_macro, gpu_release=gpu,
                        gpu_reassignment='R77 completed and released both approved4090s; NSS/user may now allocate them. No allocation performed here.',
                        finalcomparison='finalcomparison.csv', finalcomparison_rows=count, scopes_per_split=23,
                        common_update_curves='COMMON_UPDATE_CURVES.png', status='STATUS.json',
                        common_update_comparison=snapshot['common_update_comparison'], train_curves=snapshot['train_curves'],
                        selected_without_test=True, repeated_model_inference=False, git_commits_made_by_R77=0)
                    write_json(args.root / 'SUCCESS_RECEIPT.json', record)
                    state = read_json(args.root / 'STATUS.json')
                    state.update(phase='complete', success_receipt='SUCCESS_RECEIPT.json', gpu_release=gpu,
                        finalcomparison='finalcomparison.csv', all_required_artifacts_complete=True)
                    write_json(args.root / 'STATUS.json', state)
                    lines = ['# R77 Success Receipt', '', 'Completed UTC: ' + record['completed_utc'], '',
                        'Both arms completed49680 successful updates, unique VAL-best locks, one full TEST and analysis.',
                        'Both approved4090s have no compute processes and were observed idle. NSS may be assigned them now.', '',
                        '| Arm | TRAIN rows/problem | Selected update | TEST native Top1 % | CE | Actual regret % | vs FULL TRAIN SBS % |',
                        '|---|---:|---:|---:|---:|---:|---:|']
                    for arm in GROUPS:
                        value = test_macro[arm]
                        lines.append(f'| {arm} | {record["train_rows_per_problem"][arm]} | {selected[arm]["selected_update"]} | '
                            f'{100*value["macro_top1"]:.4f} | {value["macro_ce"]:.6f} | '
                            f'{value["macro_actual_regret_pct"]:.6f} | {value["macro_vs_sbs_pct"]:.6f} |')
                    lines += ['', 'finalcomparison.csv includes all saved TRAIN/VAL and locked TEST, all18 tasks, '
                        'four families, subset counts, common/own TRAIN-fixed SBS and common_updates.',
                        'No repeated TEST inference, rule change, GPU reassignment or Git commit was performed.']
                    (args.root / 'SUCCESS_RECEIPT.md').write_text('\n'.join(lines) + '\n')
                    print('[R77 SUCCESS] ' + json.dumps(record, allow_nan=False), flush=True)
                    return
                write_json(args.root / 'SUCCESS_RECEIPT_PENDING.json', dict(updated_utc=utc(),
                    phase='waiting_for_verified_gpu_release', pipelines_complete=True, gpu_release=gpu))
            time.sleep(30)


def validation_overview(args):
    """Reporting-only class coverage from saved VAL; never loads TEST or a model."""
    trajectories, classes, configs = {}, [], {}
    metric_keys = ('macro_top1', 'macro_actual_regret_pct', 'macro_vs_sbs_pct', 'macro_ce')
    for arm, name in GROUPS.items():
        root = args.root / name
        configs[arm] = read_json(root / 'protocol.json')
        history = read_json(root / 'history.json') if (root / 'history.json').exists() else []
        points, initial_zero = [], None
        for row in history:
            tag = f'epoch{row["epoch"]:03d}'
            zero, rare, rare_zero, near_zero, present = set(), 0, 0, 0, 0
            with np.load(root / 'evaluation/val' / tag / 'rank000.npz', allow_pickle=False) as saved:
                for problem in PROBLEMS:
                    prefix = problem + '__'
                    pool, ids = saved[prefix + 'pool'].tolist(), saved[prefix + 'pool_ids']
                    winner, scores = saved[prefix + 'winner'], saved[prefix + 'logits']
                    order = np.argsort(ids, kind='stable')
                    prediction = order[np.argmax(scores[:, order], axis=1)]
                    winner_counts = np.bincount(winner, minlength=len(pool))
                    pick_counts = np.bincount(prediction, minlength=len(pool))
                    correct_counts = np.bincount(winner[winner == prediction], minlength=len(pool))
                    for slot, solver in enumerate(pool):
                        support, picked = int(winner_counts[slot]), int(pick_counts[slot])
                        winner_pct, pick_pct = 100 * support / len(winner), 100 * picked / len(winner)
                        cold = support > 0 and picked == 0
                        is_rare = 0 < winner_pct < 1.
                        is_near_zero = support > 0 and 0 < pick_pct <= .1
                        present += int(support > 0)
                        rare += int(is_rare)
                        rare_zero += int(is_rare and cold)
                        near_zero += int(is_near_zero)
                        if cold:
                            zero.add(problem + '/' + solver)
                        classes.append(dict(arm=arm, successful_updates=row['successful_updates'],
                            problem=problem, solver=solver, global_solver_id=int(ids[slot]), n=len(winner),
                            winner_count=support, selected_count=picked, winner_pct=winner_pct,
                            selected_pct=pick_pct, recall_pct=100 * correct_counts[slot] / support if support else None,
                            zero_selected_with_winner=cold, near_zero_selected_with_winner=is_near_zero,
                            rare_val_support=is_rare))
            if initial_zero is None:
                initial_zero = zero
            points.append(dict(successful_updates=row['successful_updates'],
                metrics={key: row['val']['macro'][key] for key in metric_keys},
                val_present_task_classes=present, zero_selected_present_classes=len(zero),
                near_zero_selected_present_classes=near_zero, rare_val_support_classes=rare,
                rare_support_zero_selected_classes=rare_zero, zero_selected_methods=sorted(zero),
                recovered_from_first_val=sorted(initial_zero - zero), newly_zero_since_first_val=sorted(zero - initial_zero)))
        trajectories[arm] = points
    shared_updates = set(point['successful_updates'] for point in trajectories['A']) & set(
        point['successful_updates'] for point in trajectories['B'])
    matched = max(shared_updates) if shared_updates else None
    invariant_keys = ('model_params', 'optimizer', 'weight_decay', 'grad_clip', 'dropout', 'loss',
        'lr', 'rdrop_weight', 'rdrop_ramp_updates', 'global_batch_size', 'microbatch_caps',
        'task_order_sha256', 'common_trainable_sha256', 'max_successful_updates', 'baselines')
    equal = {key: configs['A'][key] == configs['B'][key] for key in invariant_keys}
    if not all(equal.values()):
        raise ValueError('The saved paired protocols differ in a shared training invariant')
    receipt = dict(experiment='R77', generated_utc=utc(), reporting_only=True, input_splits=['val'],
        test_inputs_read=False, checkpoints_or_training_rules_changed=False,
        definitions=dict(zero='positive native VAL winner support and zero predicted selections',
            near_zero='positive native VAL winner support and 0 < selected_pct <=0.1',
            rare_val_support='0 < native VAL winner_pct <1; descriptive only, not a training prior'),
        latest={arm: points[-1] if points else None for arm, points in trajectories.items()},
        latest_matched_update=matched,
        latest_matched={arm: next((point for point in points if point['successful_updates'] == matched), None)
            for arm, points in trajectories.items()}, trajectories=trajectories, paired_core_config_equal=equal,
        limitations='Early descriptive VAL reuse; differing current update numbers are not a matched A/B comparison. '
            'Zero-selection reductions alone do not imply better cost or rare-class recall.',
        resume='A continued this R77 run\'s own saved update2484 checkpoint, not historical weights or a Git commit; '
            'model/optimizer/scaler/RNG/shuffle restored; no architecture/optimization formula change')
    write_json(args.root / 'validation_only_overview.json', receipt)
    if classes:
        with (args.root / 'validation_class_history.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(classes[0]))
            writer.writeheader()
            writer.writerows(classes)
    lines = ['# R77 Validation-Only Overview', '', 'Generated: ' + receipt['generated_utc'], '',
        'Saved VAL metrics/scores only. No TEST input, model inference, fitting, checkpoint or protocol change.', '',
        '| Arm | Successful update | Top1 % | Actual regret % | vs common FULL TRAIN SBS % | CE | Zero-pick supported classes |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for arm, point in receipt['latest'].items():
        if point is not None:
            m = point['metrics']
            lines.append(f'| {arm} | {point["successful_updates"]} | {100*m["macro_top1"]:.4f} | '
                f'{m["macro_actual_regret_pct"]:.6f} | {m["macro_vs_sbs_pct"]:.6f} | '
                f'{m["macro_ce"]:.6f} | {point["zero_selected_present_classes"]} |')
    lines += ['', 'Latest shared update: ' + str(matched) + '. Use the matched JSON section for A/B comparisons.', '',
        'Zero-pick means positive VAL winner support but no predictions. Near-zero <=0.1% selection and '
        'rare VAL support <1% are descriptive only. Nonzero selection is not necessarily correct or beneficial.', '',
        '## Class Changes From First VAL', '']
    for arm, point in receipt['latest'].items():
        if point is None:
            continue
        lines.extend([f'### {arm}', '',
            f'At update{point["successful_updates"]}: {point["zero_selected_present_classes"]}/'
            f'{point["val_present_task_classes"]} supported classes have zero predictions; '
            f'{point["rare_support_zero_selected_classes"]}/{point["rare_val_support_classes"]} '
            'rare-support classes have zero predictions.', '',
            'Recovered selection (not a correctness claim): ' + ', '.join(point['recovered_from_first_val']), '',
            'Newly zero: ' + ', '.join(point['newly_zero_since_first_val']), ''])
    lines += ['## Continuity', '', receipt['resume'], '',
        'Architecture/objective/optimizer/LR/batch/clipping/R-Drop/init/task/budget/common-SBS configs are equal. '
        'The GRES correction did not edit the shared R76 objective, R43 loss or model factory. '
        'Geometry mean/std and TRAIN populations are the predeclared arm-specific differences.', '',
        'Cold-class diagnostics never feed checkpoint choice, LR, budget or TEST. No Git commits were made.']
    (args.root / 'VALIDATION_ONLY.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps(dict(report=str(args.root / 'VALIDATION_ONLY.md'), latest=receipt['latest'],
        latest_matched_update=matched), allow_nan=False), flush=True)
    return receipt


def watch(args):
    """CPU-only tmux supervisor; preserve actionable failure/completion for the parent."""
    updates, speeds = {arm: 0 for arm in GROUPS}, {}
    while True:
        arms, failures = {}, []
        transition_path = args.root / 'allocation_transition.json'
        transition = read_json(transition_path) if transition_path.exists() else {}
        transitioning = transition.get('in_progress', False)
        for arm, name in GROUPS.items():
            root = args.root / name
            state = read_json(root / 'status.json') if (root / 'status.json').exists() else dict(phase='awaiting_step')
            updates[arm] = state.get('successful_updates', updates[arm])
            if (root / 'training_complete.json').exists():
                updates[arm] = MAX_UPDATES
            if state.get('seconds_per_update'):
                speeds[arm] = state['seconds_per_update']
            complete = (root / 'pipeline_complete.json').exists()
            exit_path = args.root / (arm + '_execution_exit_code')
            exit_code = exit_path.read_text().strip() if exit_path.exists() else None
            pid_path = args.root / (arm + '_controller.pid')
            controller_alive = None
            if pid_path.exists():
                try:
                    os.kill(int(pid_path.read_text().strip()), 0)
                    controller_alive = True
                except ProcessLookupError:
                    controller_alive = False
            if not transitioning and not complete and (state.get('phase') == 'failed'
                    or exit_code not in (None, 'running', '0') or controller_alive is False):
                failures.append(arm)
            arms[arm] = dict(state, successful_updates=updates[arm], complete=complete,
                controller_alive=controller_alive, execution_exit_code=exit_code,
                training_remaining_seconds=max(0, MAX_UPDATES - updates[arm]) * speeds[arm] if arm in speeds else None)
        phase = 'failed' if failures else 'complete' if all(value['complete'] for value in arms.values()) else 'running'
        receipt = dict(phase=phase, updated_utc=utc(), monitor_pid=os.getpid(), arms=arms,
            failed_arms=failures, allocation_transition=transitioning,
            failure_policy='report required human action; never automatic GPU reassignment, restart or solver execution')
        write_json(args.root / 'monitor_status.json', receipt)
        print('[R77 monitor] ' + json.dumps(receipt, allow_nan=False), flush=True)
        if phase != 'running':
            if phase == 'complete':
                pair_summary(args)
                write_json(args.root / 'paired_complete.json', dict(completed_utc=utc(),
                    successful_updates_per_arm=MAX_UPDATES, full_tests_per_arm=1))
            return 1 if failures else 0
        time.sleep(30)


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('--stage', choices=('train', 'test', 'finish', 'analysis', 'status', 'watch', 'val_report', 'snapshot', 'report_watch', 'finalize_delivery', 'success_watch', 'diagnostic_handoff', 'diagnostic_watch'), required=True)
    result.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    result.add_argument('--data-root', type=Path, default=DATA_ROOT)
    result.add_argument('--cache-root', type=Path, default=DEFAULT_SOURCE / 'cache')
    result.add_argument('--arm', choices=tuple(GROUPS))
    result.add_argument('--gpu-type', choices=('4090', '3090'), default='4090')
    result.add_argument('--large-micro-cap', type=int)
    result.add_argument('--eval-batch-size', type=int, default=256)
    result.add_argument('--log-every', type=int, default=72)
    result.add_argument('--resume', action='store_true')
    result.add_argument('--live-gpu', action='store_true', help='Read-only NVML telemetry for the two approved UUIDs')
    result.add_argument('--wandb', action=argparse.BooleanOptionalAction, default=True)
    result.add_argument('--wandb-mode', choices=('auto', 'online', 'offline', 'disabled'), default='auto')
    return result


def main():
    cli = parser()
    args = cli.parse_args()
    args.root, args.data_root, args.cache_root = (path.expanduser().resolve()
        for path in (args.root, args.data_root, args.cache_root))
    if args.stage == 'status':
        status(args)
        return
    if args.stage == 'watch':
        raise SystemExit(watch(args))
    if args.stage == 'val_report':
        validation_overview(args)
        return
    if args.stage == 'snapshot':
        monitoring_snapshot(args)
        return
    if args.stage == 'report_watch':
        reporting_watch(args)
        return
    if args.stage == 'finalize_delivery':
        finalize_delivery(args)
        return
    if args.stage == 'success_watch':
        success_watch(args)
        return
    if args.stage == 'diagnostic_handoff':
        diagnostic_handoff(args)
        return
    if args.stage == 'diagnostic_watch':
        diagnostic_watch(args)
        return
    if args.arm is None or not 1 <= args.eval_batch_size <= 352 or args.log_every < 1:
        cli.error('Specify A/B, a positive eval batch <=352 and a positive log interval')
    micro_caps(args.gpu_type, args.large_micro_cap)
    root = args.root / GROUPS[args.arm]
    root.mkdir(parents=True, exist_ok=True)
    with (root / 'execution.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.stage == 'analysis':
            analyze_arm(args)
            pair_summary(args)
            return
        if os.environ.get('R77_APPROVED') != '1':
            cli.error('GPU launch requires the user-approved allocation: R77_APPROVED=1')
        visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
        allocated = {'GPU-71479e7e-f8bd-e430-66ed-f8db9f0d47be', 'GPU-597c682a-4b7f-7609-80a6-8ced791f5c5d'}
        if visible not in allocated or not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            cli.error('Bind exactly one of the two user-approved job1475/gpu05 GPU UUIDs')
        expected_gpu = 'GPU-71479e7e-f8bd-e430-66ed-f8db9f0d47be' if args.arm == 'A' else 'GPU-597c682a-4b7f-7609-80a6-8ced791f5c5d'
        if visible != expected_gpu:
            cli.error('Keep the approved A-small/B-full GPU ownership')
        if args.gpu_type != '4090' or (args.large_micro_cap is not None and args.large_micro_cap > 352):
            cli.error('The approved4090 allocation uses TSP/CVRP caps <=352')
        configure_torch()
        torch.set_num_threads(2)
        torch.set_num_interop_threads(2)
        device = torch.device('cuda:0')
        write_json(root / 'launch_receipt.json', dict(launched_utc=utc(), arm=args.arm, pid=os.getpid(),
            parent_pid=os.getppid(), host=socket.gethostname(), gpu_uuid=visible,
            gpu_name=torch.cuda.get_device_name(device), job_id=os.environ.get('SLURM_JOB_ID'),
            step_id=os.environ.get('SLURM_STEP_ID'), logical_batch=BATCH_SIZE,
            threads=2, prefetch_buffers=2, microbatch_caps=micro_caps(args.gpu_type, args.large_micro_cap),
            cache_root=str(args.cache_root), resume=args.resume, stage=args.stage))
        if args.stage in ('train', 'finish') and not (root / 'training_complete.json').exists():
            train(args, device)
        elif args.stage == 'train':
            cli.error('This arm has already completed training')
        if args.stage in ('test', 'finish') and not (root / 'test_complete.json').exists():
            test(args, device)
        elif args.stage == 'test':
            cli.error('This arm has already completed its one full TEST')
        if args.stage == 'finish':
            analyze_arm(args)
            pair_summary(args)


if __name__ == '__main__':
    main()
