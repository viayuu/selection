"""R78 round003: matched full-B continuations, prepared without GPU execution."""

import argparse
from collections import Counter
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import socket
import time

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from code.unified_selector.registry import PROBLEMS
from .multitask_probe import task_seed
from .pairwise_objective import comparison_loss, inputs_only, predicted_top3, single_objective
from .r42_experiment import symmetric_kl
from .r76_experiment import (
    DATA_ROOT, LOSS_RECIPE, PART_KEYS, PinnedPrefetch, Runtime, capture_rank_rng,
    file_hash, fixed_sbs_metrics, open_cache, read_json, restore_rank_rng,
    save_checkpoint, update_batch, utc, write_json,
)
from .r77_data import DEFAULT_ROOT as R77_ROOT, FULL_ROWS, GROUPS
from .r77_experiment import (
    BATCH_SIZE, EVAL_EVERY, MAX_UPDATES, WARMUP_UPDATES, ShuffleStream,
    learning_rate, load_checkpoint as load_fullb, metrics_summary, micro_caps, task_order,
)
from .solver_code_encoder import make_r45_model
from .train import configure_torch, set_seed


GOAL_ROOT = Path(__file__).resolve().parent / 'runs/R78_zero_selection_goal'
ROOT = GOAL_ROOT / 'round003'
PLAN = GOAL_ROOT / 'review-stage/FULLB_MONOTONE_COST_PLAN.md'
ARMS = ('control', 'treatment')
SOURCE_UPDATE = 34776
CONTINUATION_UPDATES = 14904
ENDPOINTS = (4968, 9936, 14904)
BASE_SHA = '31c0c75d66b1bde6ad533909a9c04b0036e80af4c4a5775b01b957ce310e6d1b'
GATE_COST = 11.39555834354421
GATE_REGRET = 0.40132674544775454
MONO_LABEL = 'r78_monotone_weights'
APPROVED_GPUS = {
    'GPU-71479e7e-f8bd-e430-66ed-f8db9f0d47be': 'gpu05',
    'GPU-597c682a-4b7f-7609-80a6-8ced791f5c5d': 'gpu05',
    'GPU-cb24f168-9f18-f0d5-6f4a-6d2406958f50': 'gpu03',
    'GPU-cb626bdb-4319-2f5e-b91f-17fcaa44a83d': 'gpu03',
}


def workflow(root, phase, **values):
    record = dict(updated_utc=utc(), phase=phase, **values)
    write_json(root / 'status.json', record)
    path = root.parent / 'STATUS.json'
    if path.exists():
        status = read_json(path)
        status['recorded_at'] = record['updated_utc']
        status['file_based_goal'].update(status='active', round=3, phase=phase, complete=False)
        status['round003'] = dict(status.get('round003', {}), **record)
        if 'next_action' in record:
            status['next_action'] = record['next_action']
        write_json(path, status)


def relative_regrets(costs, mask=None):
    if costs.dtype != np.float64 or costs.ndim != 2 or not costs.size:
        raise ValueError('Use original, nonempty FP64 costs before constructing r')
    mask = np.ones_like(costs, dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    if mask.shape != costs.shape or not mask.any(1).all() or not np.isfinite(costs[mask]).all():
        raise ValueError('Every TRAIN row requires a nonempty finite valid pool')
    minimum = np.where(mask, costs, np.inf).min(1)
    if (minimum <= 0).any():
        raise ValueError('Relative costs require positive C*')
    with np.errstate(over='raise', divide='raise', invalid='raise'):
        valid_costs = np.where(mask, costs, minimum[:, None])
        regret = (valid_costs - minimum[:, None]) / minimum[:, None]
    return regret, mask


def train_weights(costs, mask=None):
    regret, mask = relative_regrets(costs, mask)
    scale = float((regret.sum(1, dtype=np.float64) / mask.sum(1)).mean(dtype=np.float64))
    weights = np.zeros_like(regret) if scale == 0 else regret / scale
    if not np.isfinite(weights).all():
        raise ValueError('Nonfinite TRAIN-normalized cost weights')
    return weights.astype(np.float32), scale


def forward_inputs(batch):
    return inputs_only({key: value for key, value in batch.items() if key != MONO_LABEL})


def monotone_risk(scores, weights, mask):
    mask = mask.bool()
    if scores.shape != weights.shape or scores.shape != mask.shape or not mask.any(1).all():
        raise ValueError('Monotone risk requires the same nonempty legal score/weight pool')
    valid_scores = torch.where(mask, scores.float(), 0.)
    count = mask.sum(1)
    centered = valid_scores - valid_scores.sum(1, keepdim=True) / count[:, None]
    weights = torch.where(mask, weights.detach().float(), 0.)
    return ((weights * F.softplus(centered)).sum(1) / count).mean()


class FullBCostObjective(nn.Module):
    def __init__(self, selector, arm):
        super().__init__()
        if arm not in ARMS:
            raise ValueError('Only the predeclared control and treatment are permitted')
        self.selector, self.arm = selector, arm

    def single(self, output, batch, focus):
        if self.arm == 'control':
            return single_objective(output, batch, focus)
        scores = output['logits'].float()
        ce = F.cross_entropy(scores, batch['ind'])
        compare, parts = comparison_loss(output['pair_margin'], batch, focus)
        risk = monotone_risk(scores, batch[MONO_LABEL], output['solver_mask'])
        return .35 * ce + .35 * compare + .02 * risk, dict(ce=ce, cmp=compare, risk=risk, **parts)

    def forward(self, batch, consistency_weight):
        device = batch['ind'].device
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == 'cuda'):
            first = self.selector(forward_inputs(batch))
            second = self.selector(forward_inputs(batch))
        focus = predicted_top3(first['logits'], second['logits'], batch['pool_ids'], first['solver_mask'])
        loss1, parts1 = self.single(first, batch, focus)
        loss2, parts2 = self.single(second, batch, focus)
        kl = symmetric_kl(first['logits'], second['logits'], first['solver_mask'])
        loss = .5 * (loss1 + loss2) + consistency_weight * kl
        parts = {key: .5 * (parts1[key] + parts2[key]) for key in parts1}
        parts.update(loss=loss, kl=kl)
        return loss, torch.stack([parts[key].detach() for key in PART_KEYS[:-1]])


class MonotoneCache:
    def __init__(self, source, weights_path):
        self.source = source
        self.weights = np.load(weights_path, mmap_mode='r', allow_pickle=False)
        self.size, self.raw, self.lengths = source.size, source.raw, source.lengths
        self.pool, self.pool_ids = source.pool, source.pool_ids
        if self.weights.dtype != np.float32 or self.weights.shape != source.raw['costs'].shape:
            raise ValueError('Prepared TRAIN supervision must retain the exact full-B row/column space')

    def batch(self, rows, device='cpu'):
        if str(device) != 'cpu':
            raise ValueError('Use the inherited CPU gather/pin/async-transfer pipeline')
        indices = rows.cpu().numpy() if torch.is_tensor(rows) else np.asarray(rows)
        batch = self.source.batch(rows, device='cpu')
        batch[MONO_LABEL] = torch.from_numpy(self.weights[indices].copy())
        return batch

    def close(self):
        self.raw, self.lengths = {}, None
        self.source.close()
        self.weights._mmap.close()


def prepare(args):
    root = args.root
    receipt = root / 'prepared.json'
    if receipt.exists():
        print('[R78 prepare] Existing preparation retained; no refit or GPU launch', flush=True)
        return read_json(receipt)
    root.mkdir(parents=True, exist_ok=True)
    r77 = read_json(args.r77_root / 'prepared.json')
    b_root = args.r77_root / GROUPS['B']
    # This lock contains only the VAL-selected checkpoint metadata, not TEST results.
    lock = read_json(b_root / 'final_test_lock.json')
    config = read_json(b_root / 'protocol.json')
    best = read_json(b_root / 'best_eval.json')
    checkpoint = Path(lock['checkpoint'])
    if (lock['arm'] != 'B' or lock['epoch'] != 28 or lock['selected_update'] != SOURCE_UPDATE
            or lock['checkpoint_sha256'] != BASE_SHA or file_hash(checkpoint) != BASE_SHA
            or best['successful_updates'] != SOURCE_UPDATE or best['train'] is None
            or config['loss'] != LOSS_RECIPE or config['optimizer'] != 'AdamW'
            or config['weight_decay'] != 1e-4 or config['grad_clip'] != 1.
            or config['rdrop_ramp_updates'] != WARMUP_UPDATES
            or r77['sizes']['B'] != FULL_ROWS):
        raise ValueError('Use exactly the locked full-B ep28/u34776 model and inherited loss/optimizer recipe')
    if SOURCE_UPDATE + CONTINUATION_UPDATES != MAX_UPDATES or ENDPOINTS[-1] != CONTINUATION_UPDATES:
        raise ValueError('The paired continuation must stop at logical update49680')
    manifest = read_json(args.r77_root / 'source_cache_manifest.json')
    if set(manifest['splits']['train']) != set(PROBLEMS):
        raise ValueError('Keep all18 full TRAIN populations')
    cache_root = args.prepare_cache or Path(r77['cache_source'])
    scales, artifacts = {}, {}
    (root / 'train_weights').mkdir(exist_ok=True)
    for problem in PROBLEMS:
        directory = cache_root / 'train' / problem
        metadata = read_json(directory / 'metadata.json')
        source = manifest['splits']['train'][problem]
        if (metadata['split'] != 'train' or metadata['rows'] != FULL_ROWS
                or metadata['pool_ids'] != source['pool_ids'] or metadata['pool'] != source['pool']):
            raise ValueError(f'{problem}: use the existing complete full-B TRAIN cache')
        costs = np.load(directory / 'costs.npy', mmap_mode='r', allow_pickle=False)
        try:
            weights, scale = train_weights(costs)
            if costs.shape != (FULL_ROWS, len(metadata['pool'])):
                raise ValueError('Original FP64 TRAIN cost columns must match the candidate identities')
            path = root / 'train_weights' / f'{problem}.npy'
            np.save(path, weights, allow_pickle=False)
            scales[problem] = dict(a_t=scale, rows=FULL_ROWS, candidates=weights.shape[1],
                                   pool=metadata['pool'], pool_ids=metadata['pool_ids'],
                                   source_costs=str(directory / 'costs.npy'),
                                   source_costs_dtype='float64', weights_dtype='float32',
                                   weights_mean=float(weights.mean(dtype=np.float64)),
                                   weights_max=float(weights.max()), zero_block=scale == 0)
            artifacts[str(path.relative_to(root))] = file_hash(path)
        finally:
            costs._mmap.close()
        print(f'[R78 prepare] {problem}: N={FULL_ROWS}, K={weights.shape[1]}, frozen a_t={scale:.17g}', flush=True)
    rates = np.asarray([learning_rate(SOURCE_UPDATE + update) for update in
                        range(1, CONTINUATION_UPDATES + 1)], dtype=np.float64)
    np.save(root / 'late_lr.npy', rates, allow_pickle=False)
    artifacts['late_lr.npy'] = file_hash(root / 'late_lr.npy')
    write_json(root / 'task_scales.json', dict(fit_split='train', frozen=True,
        rule='a_t=mean_TRAIN(mean_valid_K(FP64 relative regret)); zero block iff a_t=0', tasks=scales))
    artifacts['task_scales.json'] = file_hash(root / 'task_scales.json')
    baseline = dict(name='locked_R77_fullB', checkpoint=str(checkpoint), checkpoint_sha256=BASE_SHA,
        source_epoch=28, logical_update=SOURCE_UPDATE, continuation_update=0,
        train=best['train'], val=best['val'],
        prediction_directories=dict(train=str(b_root / 'evaluation/train/selected_best'),
                                    val=str(b_root / 'evaluation/val/epoch028')))
    for split in ('train', 'val'):
        if not (Path(baseline['prediction_directories'][split]) / 'metrics.json').is_file():
            raise FileNotFoundError('Existing selected full-B TRAIN/VAL scores must be available')
        complete_evaluation(baseline[split])
    write_json(root / 'baseline_B_train_val.json', baseline)
    artifacts['baseline_B_train_val.json'] = file_hash(root / 'baseline_B_train_val.json')
    order = task_order(CONTINUATION_UPDATES, seed=2)
    prepared = dict(experiment='R78_round003', prepared_utc=utc(), plan=str(PLAN), plan_sha256=file_hash(PLAN),
        source_r77_root=str(args.r77_root), data_root=r77['data_root'], source_cache_manifest=str(
            args.r77_root / 'source_cache_manifest.json'), prepare_cache=str(cache_root),
        source_checkpoint=str(checkpoint), source_checkpoint_sha256=BASE_SHA,
        source_epoch=28, source_logical_update=SOURCE_UPDATE, arms=list(ARMS),
        model_params=config['model_params'], seed=2, batch_size=BATCH_SIZE,
        train_rows_per_task=FULL_ROWS, tasks=list(PROBLEMS), train_candidates=sum(
            item['candidates'] for item in scales.values()), continuation_updates=CONTINUATION_UPDATES,
        final_logical_update=MAX_UPDATES, endpoints=list(ENDPOINTS),
        endpoint_logical_updates=[SOURCE_UPDATE + u for u in ENDPOINTS],
        optimizer=dict(name='AdamW', reset='fresh in both arms; never load B optimizer or RNG',
            initial_lr=float(rates[0]), betas=[.9, .999], eps=1e-8, weight_decay=1e-4),
        gradient_clip=1., amp='inherited FP16 forwards, FP32 loss reductions; fresh GradScaler per arm',
        lr=dict(source='late_lr.npy: complete fixed FP64 shared sequence', first=float(rates[0]),
            last=float(rates[-1]), scheduler_position='logical34777 through49680, no restarted warmup'),
        rdrop=dict(weight=.35, inherited_warmup_position=SOURCE_UPDATE,
            ramp_updates=WARMUP_UPDATES, two_dropout_branches='mean the two objectives, then add .35 symmetric KL'),
        original_loss=LOSS_RECIPE, treatment='replace only .02 old scaled risk with .02 mean_valid_K((r/a_t)*softplus(z-mean_valid_K(z)))',
        task_scale='TRAIN-only FP64, frozen once; no per-instance extra normalization, margin or coefficient scan',
        control='unchanged R77B single_objective', task_order_sha256=hashlib.sha256(
            '\n'.join(order).encode('ascii')).hexdigest(),
        sampler='seed2 fresh task-order cycles and fresh per-task R77 ShuffleStreams; natural frequencies',
        updates_per_task=CONTINUATION_UPDATES // len(PROBLEMS),
        microbatch_caps=micro_caps('4090', 352), eval_batch_size=256, threads=2,
        selection=dict(raw_macro_mean_cost_max=GATE_COST, actual_regret_pct_max=GATE_REGRET,
            native_top1_min_exclusive=baseline['val']['macro']['macro_top1'],
            plan_displayed_native_top1_pct=70.89430894308944, tolerance=0.,
            rule='B fallback; eligible complete18 endpoints: highest native Top1, then lower actual regret, then earlier continuation update',
            replacement_specific_positive='selected treatment must also strictly exceed selected matched control Top1'),
        train_evaluation='one full TRAIN pass for each selected new endpoint; reuse B cached report for fallback',
        inference='unchanged maximin logits, native inference, ascending global solver ID ties; no label forward',
        artifact_sha256=artifacts, preparation_gpu_used=False, optimizer_updates_executed=0,
        new_model_forwards=0, val_labels_used_for_scale=False, test_read=False,
        hardware_approved=False, state='ready_pending_hardware',
        future_smoke='one approved GPU, one disposable TRAIN/TSP treatment update at exact1024, no VAL or saved model; deferred')
    write_json(receipt, prepared)
    for arm in ARMS:
        (root / arm).mkdir(exist_ok=True)
    workflow(root, 'ready_pending_hardware', prepared=True, plan=str(PLAN), source_checkpoint_sha256=BASE_SHA,
        seed=2, logical_batch_size=BATCH_SIZE, continuation_updates_per_arm=CONTINUATION_UPDATES,
        hardware_approved=False, gpu_execution_started=False, optimizer_updates_executed=0,
        native_app_goal_status_not_changed=True,
        next_action='Round003 code/CPU TRAIN-target preparation ready. NSS has priority; await explicit user GPU allocation before one GPU smoke or either matched continuation. No TEST, old-goal operation or reopening of closed R76 routes.')
    return prepared


def complete_evaluation(result):
    if set(result['per_problem']) != set(PROBLEMS):
        raise ValueError('Selection and reporting require complete all18 metrics')
    if any(not math.isfinite(result['macro'][key]) for key in
           ('macro_top1', 'macro_mean_cost', 'macro_actual_regret_pct')):
        raise ValueError('Use unrounded finite native metrics')


def select_checkpoint(baseline, history):
    complete_evaluation(baseline['val'])
    candidates = [baseline]
    for row in history:
        if row['continuation_update'] not in ENDPOINTS:
            raise ValueError('Only continuation endpoints4/8/12 may be candidates')
        complete_evaluation(row['val'])
        m = row['val']['macro']
        if (m['macro_mean_cost'] <= GATE_COST and m['macro_actual_regret_pct'] <= GATE_REGRET
                and m['macro_top1'] > baseline['val']['macro']['macro_top1']):
            candidates.append(row)
    return min(candidates, key=lambda row: (-row['val']['macro']['macro_top1'],
        row['val']['macro']['macro_actual_regret_pct'], row['continuation_update']))


def prepared_config(root):
    ready = read_json(root / 'prepared.json')
    if ready['experiment'] != 'R78_round003' or ready['plan_sha256'] != file_hash(PLAN):
        raise ValueError('Keep the explicitly approved round003 research protocol')
    for relative, digest in ready['artifact_sha256'].items():
        if file_hash(root / relative) != digest:
            raise ValueError(f'Prepared round003 artifact changed: {relative}')
    return ready, read_json(Path(ready['source_cache_manifest']))


def hardware_allocation(args):
    if os.environ.get('R78_COST_APPROVED') != '1' or args.allocation is None:
        raise PermissionError('No GPU run before explicit hardware approval')
    allocation = read_json(args.allocation)
    arm = allocation.get('arms', {}).get(args.arm, {})
    uuid, node = arm.get('gpu_uuid'), arm.get('node')
    if (not allocation.get('approved') or not arm.get('approved') or not arm.get('job_id')
            or uuid not in APPROVED_GPUS or node != APPROVED_GPUS[uuid]
            or socket.gethostname().split('.')[0] != node
            or os.environ.get('SLURM_JOB_ID') != str(arm['job_id'])
            or not os.environ.get('SLURM_STEP_ID')
            or os.environ.get('CUDA_VISIBLE_DEVICES') != uuid):
        raise PermissionError('Use the explicitly assigned UUID through an exact existing-job Slurm step')
    return dict(arm, allocation_file=str(args.allocation), allocation_sha256=file_hash(args.allocation))


def start_model(ready, device):
    if file_hash(Path(ready['source_checkpoint'])) != ready['source_checkpoint_sha256']:
        raise ValueError('Locked B initialization changed')
    selector, source = load_fullb(ready['source_checkpoint'], device)
    if source['successful_updates'] != SOURCE_UPDATE or source['args']['arm'] != 'B':
        raise ValueError('Start each fresh optimizer from full-B ep28/u34776')
    del source
    set_seed(2)
    return selector


def open_train(args, ready, manifest, problem):
    source = open_cache(args.cache_root, 'train', problem, manifest, Path(ready['data_root']))
    return MonotoneCache(source, args.root / 'train_weights' / f'{problem}.npy')


@torch.no_grad()
def evaluate(selector, args, ready, manifest, split, tag):
    if split not in ('train', 'val'):
        raise PermissionError('This experiment has no TEST/LIB stage')
    directory = args.root / args.arm / 'evaluation' / split / tag
    directory.mkdir(parents=True, exist_ok=True)
    baseline = read_json(Path(ready['source_r77_root']) / 'common_full_train_sbs.json')
    was_training, arrays, results = selector.training, {}, {}
    device = next(selector.parameters()).device
    selector.eval()
    try:
        for problem in PROBLEMS:
            cache = open_cache(args.cache_root, split, problem, manifest, Path(ready['data_root']))
            try:
                requests = ((problem, torch.arange(start, min(start + ready['eval_batch_size'], cache.size)),
                    min(ready['eval_batch_size'], cache.size - start))
                    for start in range(0, cache.size, ready['eval_batch_size']))
                scores = []
                with PinnedPrefetch({problem: cache}, requests, device) as batches:
                    for _, _, _, batch in batches:
                        scores.append(selector(forward_inputs(batch))['logits'].float().cpu().numpy())
                scores = np.concatenate(scores)
                results[problem], _ = fixed_sbs_metrics(scores, cache.raw, baseline[problem])
                prefix = problem + '__'
                arrays.update({prefix + key: value for key, value in dict(logits=scores,
                    indices=np.arange(cache.size, dtype=np.int64), costs=cache.raw['costs'].copy(),
                    winner=cache.raw['winner'].copy(), pool=np.asarray(cache.pool),
                    pool_ids=np.asarray(cache.pool_ids, dtype=np.int64), nodes=np.asarray(cache.lengths).copy()).items()})
            finally:
                cache.close()
    finally:
        selector.train(was_training)
    temporary = directory / f'.rank000.{os.getpid()}.tmp'
    with temporary.open('wb') as stream:
        np.savez(stream, **arrays)
    temporary.replace(directory / 'rank000.npz')
    result = metrics_summary(results)
    write_json(directory / 'metrics.json', result)
    print('[R78 eval] ' + json.dumps(dict(arm=args.arm, split=split, tag=tag, **result['macro'])), flush=True)
    return result


def tracker_for(args, config):
    if args.wandb_mode == 'disabled':
        return None
    try:
        import wandb
        mode = args.wandb_mode
        if mode == 'auto':
            mode = 'online' if os.environ.get('WANDB_API_KEY') else 'offline'
        return wandb.init(project='selector', entity='yjkds-southern-university-of-science-technology',
            name=f'R78_round003_{args.arm}_seed2', mode=mode, dir=str(args.root / args.arm), config=config,
            settings=wandb.Settings(init_timeout=15))
    except Exception as error:
        print(f'[R78 W&B] {type(error).__name__}: local logs remain authoritative', flush=True)
        return None


def train(args, device, allocation):
    ready, manifest = prepared_config(args.root)
    root = args.root / args.arm
    if (root / 'training_complete.json').exists():
        raise ValueError('This fixed-budget arm already finished')
    last = root / 'last.pt'
    if last.exists() != args.resume:
        raise ValueError('Use --resume exactly when this arm owns an existing checkpoint')
    baseline = read_json(args.root / 'baseline_B_train_val.json')
    config = dict(experiment='R78_round003', arm=args.arm, prepared_sha256=file_hash(args.root / 'prepared.json'),
                  continuation_updates=CONTINUATION_UPDATES, batch_size=BATCH_SIZE, seed=2)
    checkpoint = None
    if args.resume:
        checkpoint = torch.load(last, map_location='cpu', weights_only=False)
        if checkpoint['args'] != config:
            raise ValueError('Resume only this same arm and immutable protocol')
        selector = make_r45_model(ready['model_params'], state_dict=checkpoint['model']).to(device)
    else:
        selector = start_model(ready, device)
    optimizer = torch.optim.AdamW(selector.parameters(), lr=ready['optimizer']['initial_lr'],
                                  betas=(.9, .999), eps=1e-8, weight_decay=1e-4)
    scaler = torch.amp.GradScaler('cuda', enabled=True)
    objective, runtime = FullBCostObjective(selector, args.arm), Runtime(0, 1, device)
    successful, retries_total, history = 0, 0, []
    counts, cursors = Counter({p: 0 for p in PROBLEMS}), {p: 0 for p in PROBLEMS}
    if checkpoint is not None:
        optimizer.load_state_dict(checkpoint['optimizer'])
        for state in optimizer.state.values():
            for key, value in state.items():
                if torch.is_tensor(value):
                    state[key] = value.cpu() if key == 'step' else value.to(device)
        scaler.load_state_dict(checkpoint['scaler'])
        successful, retries_total = checkpoint['successful_updates'], checkpoint['amp_retries']
        history, counts = checkpoint['history'], Counter(checkpoint['updates_by_problem'])
        cursors = checkpoint['stream_draws']
        if successful != len(history) * ENDPOINTS[0] or any(cursors[p] != counts[p] * BATCH_SIZE for p in PROBLEMS):
            raise ValueError('Resume only a saved whole18 endpoint and its exact stream cursor')
        restore_rank_rng(checkpoint['rng'], device)
    else:
        if optimizer.state:
            raise ValueError('Both continuation arms must start with an empty fresh AdamW state')
        write_json(root / 'initialization.json', dict(source_checkpoint_sha256=BASE_SHA, seed=2,
            optimizer_state_entries=0, optimizer_reset=True, B_optimizer_loaded=False, B_rng_loaded=False,
            lr=ready['optimizer']['initial_lr'], logical_update=SOURCE_UPDATE, allocation=allocation))
    del checkpoint
    caches = {}
    tracker = None
    try:
        for problem in PROBLEMS:
            caches[problem] = open_train(args, ready, manifest, problem)
        streams = {p: ShuffleStream(FULL_ROWS, task_seed(2, p), draws=cursors[p]) for p in PROBLEMS}
        order = task_order(CONTINUATION_UPDATES, seed=2)
        rates = np.load(args.root / 'late_lr.npy', allow_pickle=False)
        tracker = tracker_for(args, config)

        def state():
            return dict(model=selector.state_dict(), optimizer=optimizer.state_dict(), scaler=scaler.state_dict(),
                args=config, successful_updates=successful, logical_update=SOURCE_UPDATE + successful,
                updates_by_problem=dict(counts), amp_retries=retries_total,
                stream_draws={p: stream.draws for p, stream in streams.items()},
                rng=capture_rank_rng(device), history=history)

        if not args.resume:
            save_checkpoint(last, state())
        torch.cuda.reset_peak_memory_stats(device)
        for end in ENDPOINTS:
            if end <= successful:
                continue
            first, started = successful, time.monotonic()
            weighted = torch.zeros(len(PART_KEYS), dtype=torch.float64, device=device)
            requests = [(p, streams[p].take(), BATCH_SIZE) for p in order[successful:end]]
            selector.train()
            with PinnedPrefetch(caches, requests, device) as batches:
                for problem, n_rank, n_global, batch in batches:
                    if n_rank != BATCH_SIZE or n_global != BATCH_SIZE:
                        raise ValueError('A successful update must contain exactly1024 real TRAIN rows')
                    lr = float(rates[successful])
                    for group in optimizer.param_groups:
                        group['lr'] = lr
                    consistency = .35 * min(1., (SOURCE_UPDATE + successful + 1) / WARMUP_UPDATES)
                    parts, retries = update_batch(objective, optimizer, scaler, batch, consistency,
                        n_rank, n_global, runtime, microbatch_cap=ready['microbatch_caps'][problem])
                    successful += 1
                    counts[problem] += 1
                    retries_total += retries
                    weighted.add_(parts.double())
                    if successful in (1, 10) or successful % args.log_every == 0 or successful == end:
                        averages = (weighted / (successful - first)).cpu().tolist()
                        seconds_per_update = (time.monotonic() - started) / (successful - first)
                        row = dict(updated_utc=utc(), arm=args.arm, continuation_update=successful,
                            logical_update=SOURCE_UPDATE + successful, lr=lr, consistency_weight=consistency,
                            pid=os.getpid(), max_successful_updates=CONTINUATION_UPDATES,
                            optimizer_examples=successful * BATCH_SIZE, effective_batch_size=BATCH_SIZE,
                            seconds_per_update=seconds_per_update,
                            training_eta_seconds=(CONTINUATION_UPDATES - successful) * seconds_per_update,
                            eta_excludes_remaining_val_and_selected_train_evaluation=True,
                            peak_allocated_gib=torch.cuda.max_memory_allocated(device) / 2**30,
                            peak_reserved_gib=torch.cuda.max_memory_reserved(device) / 2**30,
                            amp_retries=retries_total, **dict(zip(PART_KEYS, averages)))
                        with (root / 'updates.jsonl').open('a') as stream:
                            stream.write(json.dumps(row, allow_nan=False) + '\n')
                        write_json(root / 'status.json', dict(phase='training', **row))
                        print('[R78 train] ' + json.dumps(row), flush=True)
                        if tracker is not None:
                            tracker.log(row, step=successful)
            tag = f'epoch{successful // EVAL_EVERY:03d}'
            checkpoint_path = root / f'{tag}.pt'
            validation = evaluate(selector, args, ready, manifest, 'val', tag)
            history.append(dict(name=args.arm, continuation_update=successful,
                logical_update=SOURCE_UPDATE + successful, continuation_epoch=successful // EVAL_EVERY,
                checkpoint=str(checkpoint_path), val=validation,
                optimization=dict(zip(PART_KEYS, (weighted / (successful - first)).cpu().tolist())),
                prediction_directories=dict(val=str(root / 'evaluation/val' / tag))))
            save_checkpoint(checkpoint_path, state())
            save_checkpoint(last, state())
            write_json(root / 'history.json', history)
            if tracker is not None:
                tracker.log({'val/' + k: v for k, v in validation['macro'].items()}, step=successful)
        if successful != CONTINUATION_UPDATES or set(counts.values()) != {CONTINUATION_UPDATES // len(PROBLEMS)}:
            raise ValueError('Both arms require exactly14904 successful updates and828 per task')
        selected = dict(select_checkpoint(baseline, history))
        if selected['continuation_update']:
            selected_state = torch.load(selected['checkpoint'], map_location='cpu', weights_only=False)
            selector.load_state_dict(selected_state['model'], strict=True)
            del selected_state
            selected['train'] = evaluate(selector, args, ready, manifest, 'train', 'selected')
            selected['prediction_directories'] = dict(selected['prediction_directories'],
                train=str(root / 'evaluation/train/selected'))
        selected['checkpoint_sha256'] = file_hash(Path(selected['checkpoint']))
        result = dict(arm=args.arm, successful_updates=successful, logical_update=MAX_UPDATES,
            updates_by_problem=dict(counts), selected=selected, history=history, amp_retries=retries_total,
            test_read=False, formal_test_authorized=False, goal_complete=False)
        write_json(root / 'selected_train_val.json', result)
        write_json(root / 'training_complete.json', dict(completed_utc=utc(), successful_updates=successful,
            selected_checkpoint=selected['checkpoint'], selected_continuation_update=selected['continuation_update'],
            test_read=False, next_stage='matched TRAIN/VAL analysis only'))
        write_json(root / 'status.json', dict(phase='training_complete', updated_utc=utc(),
            successful_updates=successful, selected_continuation_update=selected['continuation_update'], test_read=False))
    finally:
        for cache in caches.values():
            cache.close()
        if tracker is not None:
            tracker.finish()


def smoke(args, device, allocation):
    path = args.root / 'future_gpu_smoke.json'
    if path.exists():
        raise ValueError('The single GPU smoke is already recorded; no smoke loop')
    ready, manifest = prepared_config(args.root)
    selector = start_model(ready, device)
    objective = FullBCostObjective(selector, 'treatment')
    optimizer = torch.optim.AdamW(selector.parameters(), lr=ready['optimizer']['initial_lr'], weight_decay=1e-4)
    scaler = torch.amp.GradScaler('cuda')
    cache = open_train(args, ready, manifest, 'TSP')
    try:
        rows = ShuffleStream(FULL_ROWS, task_seed(2, 'TSP')).take(BATCH_SIZE)
        with PinnedPrefetch({'TSP': cache}, [('TSP', rows, BATCH_SIZE)], device) as batches:
            _, count, total, batch = next(batches)
            parts, retries = update_batch(objective, optimizer, scaler, batch, .35, count, total,
                Runtime(0, 1, device), microbatch_cap=ready['microbatch_caps']['TSP'])
        write_json(path, dict(completed_utc=utc(), allocation=allocation, problem='TSP', arm='treatment',
            diagnostic_optimizer_updates=1, scientific_optimizer_updates=0, batch_size=BATCH_SIZE,
            microbatch_cap=ready['microbatch_caps']['TSP'], amp_retries=retries,
            parts=dict(zip(PART_KEYS, parts.cpu().tolist())), model_saved=False, val_read=False, test_read=False))
        print('[R78 smoke] one real TRAIN/TSP batch1024 update completed; model discarded; no VAL/TEST', flush=True)
    finally:
        cache.close()


def csv_rows(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def analyze(args):
    from .r76_probability_diagnostics import load_shards
    root = args.root
    baseline = read_json(root / 'baseline_B_train_val.json')
    selected = {arm: read_json(root / arm / 'selected_train_val.json')['selected'] for arm in ARMS}
    sources = dict(B=baseline, **selected)
    control, treatment = selected['control']['val']['macro'], selected['treatment']['val']['macro']
    positive = (selected['treatment']['continuation_update'] > 0
                and treatment['macro_top1'] > baseline['val']['macro']['macro_top1']
                and treatment['macro_top1'] > control['macro_top1'])
    comparisons, methods, task_rows = [], [], []
    for split in ('train', 'val'):
        predictions, choices, samples = {}, {}, {}
        for name, item in sources.items():
            complete_evaluation(item[split])
            predictions[name], _ = load_shards(Path(item['prediction_directories'][split]), split,
                                              item[split]['per_problem'])
            choices[name], samples[name] = {}, {}
            for problem in PROBLEMS:
                value = predictions[name][problem]
                order = np.argsort(value['pool_ids'], kind='stable')
                choice = order[np.argmax(value['logits'][:, order], axis=1)]
                cost = value['costs'][np.arange(value['n']), choice]
                regret = (cost - value['costs'].min(1)) / value['costs'].min(1) * 100
                choices[name][problem], samples[name][problem] = choice, dict(cost=cost, regret=regret)
                m = item[split]['per_problem'][problem]
                task_rows.append(dict(rule=name, split=split, problem=problem, top1=m['top1'],
                    mean_cost=m['mean_cost'], actual_regret_pct=m['actual_regret_pct'], vs_sbs_pct=m['vs_sbs_pct']))
                support = np.bincount(value['winner'], minlength=value['k'])
                picks = np.bincount(choice, minlength=value['k'])
                for slot, method in enumerate(value['pool']):
                    methods.append(dict(rule=name, split=split, problem=problem, method=str(method),
                        winner_count=int(support[slot]), selected_count=int(picks[slot]),
                        zero_with_winner=bool(picks[slot] == 0 and support[slot] > 0),
                        near_zero_with_winner=bool(0 < 100 * picks[slot] / value['n'] <= .1 and support[slot] > 0)))
        for old_name, new_name in (('B', 'control'), ('B', 'treatment'), ('control', 'treatment')):
            for problem in PROBLEMS:
                old, new = predictions[old_name][problem], predictions[new_name][problem]
                if any(not np.array_equal(old[key], new[key]) for key in ('indices', 'pool_ids', 'winner', 'costs')):
                    raise ValueError('Changed-decision readout requires identical native row/pool/cost space')
                old_pick, new_pick = choices[old_name][problem], choices[new_name][problem]
                changed = old_pick != new_pick
                old_correct, new_correct = old_pick == old['winner'], new_pick == new['winner']
                cost_delta = samples[new_name][problem]['cost'] - samples[old_name][problem]['cost']
                regret_delta = samples[new_name][problem]['regret'] - samples[old_name][problem]['regret']
                comparisons.append(dict(split=split, problem=problem, comparison=f'{new_name}_vs_{old_name}',
                    changed_predictions=int(changed.sum()), corrected_top1=int((~old_correct & new_correct).sum()),
                    broken_top1=int((old_correct & ~new_correct).sum()),
                    wrong_to_wrong=int((changed & ~old_correct & ~new_correct).sum()),
                    cost_decrease_count=int((cost_delta < 0).sum()), cost_increase_count=int((cost_delta > 0).sum()),
                    changed_exact_cost_ties=int((changed & (cost_delta == 0)).sum()),
                    negative_cost_change_sum=float(cost_delta[cost_delta < 0].sum()),
                    positive_cost_change_sum=float(cost_delta[cost_delta > 0].sum()),
                    mean_cost_delta=float(cost_delta.mean()), actual_regret_delta_pp=float(regret_delta.mean())))
    outcome = 'exploratory_replacement_positive' if positive else 'finite_replacement_trial_closed_negative'
    summary = dict(completed_utc=utc(), outcome=outcome, reference='locked full-B ep28/u34776',
        selected={name: dict(continuation_update=item['continuation_update'], checkpoint=item['checkpoint'],
            train_macro=item['train']['macro'], val_macro=item['val']['macro']) for name, item in sources.items()},
        treatment_vs_control={key: treatment[key] - control[key] for key in
            ('macro_top1', 'macro_mean_cost', 'macro_actual_regret_pct')},
        control_only_improved=not positive and selected['control']['continuation_update'] > 0,
        interpretation='single-seed exploratory; no Pareto claim against control or automatic control promotion',
        test_read=False, formal_test_authorized=False, goal_complete=False)
    csv_rows(root / 'selected_all18_train_val.csv', task_rows)
    csv_rows(root / 'changed_decisions_train_val.csv', comparisons)
    csv_rows(root / 'method_selection_train_val.csv', methods)
    write_json(root / 'result_summary.json', summary)
    workflow(root, 'round003_complete_waiting_research', outcome=outcome, goal_complete=False,
        test_read=False, formal_test_authorized=False,
        next_action='Return matched full18 TRAIN/VAL results to Astra; no coefficient retry, TEST-based tuning or automatic promotion. Goal remains active.')
    print('[R78 round003 result] ' + json.dumps(summary), flush=True)
    return summary


def first_val_report(args):
    root, directory = args.root, args.root / 'first_val'
    if (directory / 'summary.json').exists():
        return read_json(directory / 'summary.json')
    paths = {arm: root / arm / 'evaluation/val/epoch004' for arm in ARMS}
    if any(not (path / 'metrics.json').exists() for path in paths.values()):
        return None
    directory.mkdir(exist_ok=True)
    baseline = read_json(root / 'baseline_B_train_val.json')
    paths['B'] = Path(baseline['prediction_directories']['val'])
    reports = dict(B=baseline['val'], **{arm: read_json(paths[arm] / 'metrics.json') for arm in ARMS})
    task_rows, method_rows, cold = [], [], {}
    for rule in ('B', *ARMS):
        report = reports[rule]
        complete_evaluation(report)
        zero_pairs, near_pairs, zero_winners, rows_total, lehd = 0, 0, 0, 0, None
        with np.load(paths[rule] / 'rank000.npz', allow_pickle=False) as saved:
            for problem in PROBLEMS:
                m = report['per_problem'][problem]
                task_rows.append(dict(rule=rule, problem=problem, rows=m['n'],
                    top1_pct=100 * m['top1'], actual_regret_pct=m['actual_regret_pct'],
                    mean_cost=m['mean_cost'], ce=m['ce']))
                prefix = problem + '__'
                scores, winner = saved[prefix + 'logits'], saved[prefix + 'winner']
                pool, ids = saved[prefix + 'pool'], saved[prefix + 'pool_ids']
                order = np.argsort(ids, kind='stable')
                choice = order[np.argmax(scores[:, order], axis=1)]
                support = np.bincount(winner, minlength=len(pool))
                picks = np.bincount(choice, minlength=len(pool))
                rows_total += len(winner)
                for slot, method in enumerate(pool):
                    zero = bool(picks[slot] == 0 and support[slot] > 0)
                    near = bool(0 < 100 * picks[slot] / len(winner) <= .1 and support[slot] > 0)
                    zero_pairs += int(zero)
                    near_pairs += int(near)
                    zero_winners += int(support[slot]) if zero else 0
                    method_rows.append(dict(rule=rule, problem=problem, method=str(method),
                        winner_count=int(support[slot]), selected_count=int(picks[slot]),
                        zero_with_winner=zero, near_zero_with_winner=near))
                    if problem == 'TSP' and method == 'LEHD':
                        won = winner == slot
                        probabilities = torch.softmax(torch.from_numpy(scores).float(), -1).numpy()
                        ranking = order[np.argsort(-scores[:, order], axis=1, kind='stable')]
                        rank = np.argmax(ranking == slot, axis=1) + 1
                        costs = saved[prefix + 'costs']
                        loss = costs[np.arange(len(winner)), choice] - costs[np.arange(len(winner)), winner]
                        total_loss = float(loss.sum(dtype=np.float64))
                        lehd = dict(winner_count=int(support[slot]), selected_count=int(picks[slot]),
                            winner_conditional_probability_mean=float(probabilities[won, slot].mean(dtype=np.float64)),
                            winner_conditional_rank_mean=float(rank[won].mean()),
                            within_TSP_raw_costloss_share_pct=100 * float(loss[won & (choice != slot)].sum(dtype=np.float64))
                            / total_loss if total_loss else 0.)
        cold[rule] = dict(zero_with_winner_pairs=zero_pairs, near_zero_with_winner_pairs=near_pairs,
            zero_winner_count=zero_winners, zero_winner_coverage_pct=100 * zero_winners / rows_total,
            rows=rows_total, TSP_LEHD=lehd)
    gate = {}
    for arm in ARMS:
        selected = select_checkpoint(baseline, [dict(continuation_update=ENDPOINTS[0], val=reports[arm])])
        gate[arm] = dict(endpoint_eligible=selected['continuation_update'] == ENDPOINTS[0],
                         fallback_rule='B' if selected['continuation_update'] == 0 else arm)
    summary = dict(completed_utc=utc(), continuation_update=4968, logical_update=39744,
        tasks=18, rows_per_task=3895, macro={rule: value['macro'] for rule, value in reports.items()},
        cold_methods=cold, endpoint_gate=gate, diagnostic_threshold_near_zero_pct=.1,
        no_early_stop=True, test_read=False, additional_model_forwards=0,
        note='Descriptive saved-VAL readout only; continue both fixed budgets. No cold-method gate or tuning.')
    csv_rows(directory / 'all18_metrics.csv', task_rows)
    csv_rows(directory / 'native_method_counts.csv', method_rows)
    lines = ['# Round003 First Complete18 VAL', '',
             'Continuation+4968 / logical39744. Both branches continue the fixed14904 budget.',
             'All18 tasks,3895 rows/task; equal-task macro. Saved predictions only, no TEST or new forwards.',
             '', '| Rule | Top1 % | Actual regret % | Raw mean cost | CE | Zero/winner pairs | Zero winner coverage % |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for rule in ('B', *ARMS):
        m, c = reports[rule]['macro'], cold[rule]
        lines.append(f"| {rule} | {100*m['macro_top1']:.9f} | {m['macro_actual_regret_pct']:.9f} | "
                     f"{m['macro_mean_cost']:.12f} | {m['macro_ce']:.9f} | {c['zero_with_winner_pairs']} | {c['zero_winner_coverage_pct']:.6f} |")
    lines += ['', '## LEHD/TSP Descriptive Observation', '',
              '| Rule | Native winners | Picks | Winner-conditional p % | Winner-conditional rank | Within-TSP costloss share % |',
              '|---|---:|---:|---:|---:|---:|']
    for rule in ('B', *ARMS):
        c = cold[rule]['TSP_LEHD']
        lines.append(f"| {rule} | {c['winner_count']} | {c['selected_count']} | "
                     f"{100*c['winner_conditional_probability_mean']:.6f} | {c['winner_conditional_rank_mean']:.6f} | {c['within_TSP_raw_costloss_share_pct']:.6f} |")
    lines += ['', '## Complete18 Task Metrics', '',
              '| Task | Rule | Top1 % | Actual regret % | Raw mean cost | CE |',
              '|---|---|---:|---:|---:|---:|']
    for row in task_rows:
        lines.append(f"| {row['problem']} | {row['rule']} | {row['top1_pct']:.9f} | "
                     f"{row['actual_regret_pct']:.9f} | {row['mean_cost']:.12f} | {row['ce']:.9f} |")
    lines += ['', 'Cold counts and conditional probabilities are descriptive, not an acceptance condition.',
              'First endpoint eligibility is not final replacement certification or TEST authorization.']
    (directory / 'REPORT.md').write_text('\n'.join(lines) + '\n')
    write_json(directory / 'summary.json', summary)
    print('[R78 first VAL] ' + json.dumps(summary), flush=True)
    return summary


def wait_first_val(args):
    while first_val_report(args) is None:
        time.sleep(args.poll_seconds)


def final_report(args):
    root = args.root
    while not (root / 'result_summary.json').exists():
        if read_json(root / 'status.json')['phase'] == 'round003_execution_error_not_scientific_negative':
            raise RuntimeError('Execution failure: preserve logs, no negative scientific disposition')
        time.sleep(args.poll_seconds)
    summary = read_json(root / 'result_summary.json')
    baseline = read_json(root / 'baseline_B_train_val.json')
    results = {arm: read_json(root / arm / 'selected_train_val.json') for arm in ARMS}
    endpoints, tasks = [], []
    for arm, value in results.items():
        for row in value['history']:
            m = row['val']['macro']
            accepted = dict(native_top1_improves=m['macro_top1'] > baseline['val']['macro']['macro_top1'],
                raw_mean_cost_not_worse=m['macro_mean_cost'] <= GATE_COST,
                actual_regret_not_worse=m['macro_actual_regret_pct'] <= GATE_REGRET)
            endpoints.append(dict(arm=arm, continuation_update=row['continuation_update'],
                top1_pct=100 * m['macro_top1'], mean_cost=m['macro_mean_cost'],
                actual_regret_pct=m['macro_actual_regret_pct'], ce=m['macro_ce'],
                endpoint_eligible=all(accepted.values()), **accepted))
            for problem in PROBLEMS:
                p = row['val']['per_problem'][problem]
                tasks.append(dict(arm=arm, continuation_update=row['continuation_update'], problem=problem,
                    top1_pct=100 * p['top1'], mean_cost=p['mean_cost'],
                    actual_regret_pct=p['actual_regret_pct'], ce=p['ce']))
    csv_rows(root / 'all_endpoint_val_metrics.csv', tasks)
    csv_rows(root / 'all_endpoint_val_gates.csv', endpoints)
    counts = {arm: value['successful_updates'] for arm, value in results.items()}
    record = dict(summary, successful_updates_by_arm=counts, total_optimizer_updates=sum(counts.values()),
        endpoint_metrics_and_gates=endpoints, CE_acceptance_condition=False,
        candidate_TEST_eligible=summary['outcome'] == 'exploratory_replacement_positive',
        TEST_opened=False, additional_model_forwards_for_report=0, GPU_release_check='pending_worker_exit')
    lines = ['# R78 Round003 Fixed-Gate Final Report', '',
             f"Outcome: {summary['outcome']}. Full18, batch1024, seed2;14904 successful updates per arm.",
             'Original B ep28/u34776 is fallback. CE is descriptive only; no TEST read or coefficient retry.',
             '', '| Rule | Continuation update | Top1 % | Raw mean cost | Actual regret % | CE | Eligible |',
             '|---|---:|---:|---:|---:|---:|---|']
    b = baseline['val']['macro']
    lines.append(f"| B | 0 | {100*b['macro_top1']:.9f} | {b['macro_mean_cost']:.12f} | {b['macro_actual_regret_pct']:.9f} | {b['macro_ce']:.9f} | fallback |")
    for row in endpoints:
        lines.append(f"| {row['arm']} | {row['continuation_update']} | {row['top1_pct']:.9f} | "
                     f"{row['mean_cost']:.12f} | {row['actual_regret_pct']:.9f} | {row['ce']:.9f} | {row['endpoint_eligible']} |")
    lines += ['', '## Selected Whole18 Rules', '',
              '| Branch | Selected continuation update | Checkpoint |', '|---|---:|---|']
    for arm, value in results.items():
        chosen = value['selected']
        lines.append(f"| {arm} | {chosen['continuation_update']} | {chosen['checkpoint']} |")
    lines += ['', 'A selected update0 means B fallback, not identical fitted trajectories. All negative endpoint scores/checkpoints are retained.',
              'Treatment-specific positive requires strictly higher selected Top1 than matched control; cost/regret differences against control remain reported.',
              'No automatic control promotion or TEST release; overall research goal remains active.',
              'GPU release is checked after both workers exit, separately from scientific acceptance.']
    (root / 'FINAL_RESULTS.md').write_text('\n'.join(lines) + '\n')
    write_json(root / 'final_report.json', record)
    workflow(root, 'round003_complete_waiting_research', outcome=summary['outcome'],
        optimizer_updates_executed=sum(counts.values()), successful_updates_by_arm=counts,
        final_report=str(root / 'FINAL_RESULTS.md'), final_gate_record=str(root / 'final_report.json'),
        test_read=False, candidate_TEST_eligible=record['candidate_TEST_eligible'],
        GPU_release_check='pending_worker_exit', goal_complete=False,
        next_action='Finite budgets and fixed B/control gate complete. Preserve all endpoint negatives. Check only the two assigned GPU releases after worker exit; no TEST-based tuning, coefficient retry or goal completion.')
    print('[R78 fixed final report] ' + json.dumps(record), flush=True)
    return record


def watch(args):
    """Observe only this pair's files; do not query GPUs or restart workers."""
    root = args.root
    remembered, last_event = {arm: {} for arm in ARMS}, None
    baseline = read_json(root / 'baseline_B_train_val.json')
    while True:
        if (root / 'result_summary.json').exists():
            print('[R78 watch] Final matched TRAIN/VAL report already saved; no repeated analysis', flush=True)
            return
        arms, checkpoints, errors = {}, {}, {}
        for arm in ARMS:
            directory = root / arm
            current = read_json(directory / 'status.json') if (directory / 'status.json').exists() else {}
            remembered[arm].update(current)
            receipt = read_json(directory / 'launch_receipt.json') if (directory / 'launch_receipt.json').exists() else {}
            complete = (directory / 'training_complete.json').exists()
            value = remembered[arm]
            arms[arm] = dict(pid=receipt.get('pid'), gpu_uuid=receipt.get('observed_device_uuid'),
                successful_updates=value.get('continuation_update', value.get('successful_updates', 0)),
                max_successful_updates=CONTINUATION_UPDATES, complete=complete,
                seconds_per_update=value.get('seconds_per_update'),
                training_eta_seconds=value.get('training_eta_seconds'),
                peak_allocated_gib=value.get('peak_allocated_gib'),
                peak_reserved_gib=value.get('peak_reserved_gib'), amp_retries=value.get('amp_retries'))
            history = read_json(directory / 'history.json') if (directory / 'history.json').exists() else []
            checkpoints[arm] = [dict(continuation_update=row['continuation_update'],
                macro=row['val']['macro']) for row in history]
            if not complete and (directory / 'train.log').exists():
                with (directory / 'train.log').open('rb') as stream:
                    stream.seek(0, 2)
                    stream.seek(max(0, stream.tell() - 8192))
                    tail = stream.read().decode('utf-8', errors='replace')
                if 'Traceback (most recent call last)' in tail:
                    errors[arm] = tail
        if errors:
            workflow(root, 'round003_execution_error_not_scientific_negative',
                monitor_pid=os.getpid(), running_arms=arms, val_checkpoints=checkpoints,
                execution_errors=errors, test_read=False, goal_complete=False,
                next_action='Report the genuine execution failure; do not kill the other worker, claim a negative scientific outcome, extend budget or read TEST.')
            raise RuntimeError('A pair worker reported an execution error; see its existing train.log')
        if all(value['complete'] for value in arms.values()):
            print('[R78 watch] Both finite budgets completed; computing the single matched TRAIN/VAL handoff', flush=True)
            analyze(args)
            return
        event = tuple((arm, arms[arm]['successful_updates'], len(checkpoints[arm]), arms[arm]['complete'])
                      for arm in ARMS)
        workflow(root, 'round003_running_matched_continuations', monitor_pid=os.getpid(),
            hardware_approved=True, gpu_execution_started=True, running_arms=arms,
            optimizer_updates_executed=sum(value['successful_updates'] for value in arms.values()),
            val_checkpoints=checkpoints, fixed_endpoints=list(ENDPOINTS),
            baseline_gate=baseline['val']['macro'], test_read=False, goal_complete=False,
            next_action='Continue only the existing matched branches to14904 successful updates and VAL4/8/12. CPU watcher observes own files; then one matched TRAIN/VAL analysis and gate disposition, no TEST or automatic promotion.')
        if event != last_event:
            print('[R78 watch] ' + json.dumps(dict(updated_utc=utc(), arms=arms, val_checkpoints=checkpoints)), flush=True)
            last_event = event
        time.sleep(args.poll_seconds)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('prepare', 'train', 'smoke', 'analyze', 'status', 'watch', 'first_val', 'final_report'), default='prepare')
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--r77-root', type=Path, default=R77_ROOT)
    parser.add_argument('--prepare-cache', type=Path)
    parser.add_argument('--cache-root', type=Path, default=Path('/tmp/shiys_R76_R45A_single_reld_mt_cache'))
    parser.add_argument('--arm', choices=ARMS, default='treatment')
    parser.add_argument('--allocation', type=Path)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--log-every', type=int, default=100)
    parser.add_argument('--poll-seconds', type=int, default=60)
    parser.add_argument('--wandb-mode', choices=('auto', 'offline', 'online', 'disabled'), default='auto')
    args = parser.parse_args()
    args.root, args.r77_root = args.root.resolve(), args.r77_root.resolve()
    torch.set_num_threads(2)
    if args.stage == 'prepare':
        prepare(args)
        return
    if args.stage == 'status':
        print(json.dumps(read_json(args.root / 'status.json'), indent=2))
        return
    if args.stage == 'analyze':
        analyze(args)
        return
    if args.stage == 'watch':
        watch(args)
        return
    if args.stage == 'first_val':
        wait_first_val(args)
        return
    if args.stage == 'final_report':
        final_report(args)
        return
    allocation = hardware_allocation(args)
    available, count = torch.cuda.is_available(), torch.cuda.device_count()
    if not available or count != 1:
        raise RuntimeError(f'The exact approved UUID must expose one CUDA device: available={available}, '
                           f'count={count}, requested={allocation["gpu_uuid"]}, '
                           f'SLURM_STEP_GPUS={os.environ.get("SLURM_STEP_GPUS")}')
    configure_torch()
    device = torch.device('cuda', 0)
    properties = torch.cuda.get_device_properties(device)
    observed_uuid = getattr(properties, 'uuid', None)
    if observed_uuid is not None and str(observed_uuid).lower().removeprefix('gpu-') != allocation['gpu_uuid'].lower().removeprefix('gpu-'):
        raise PermissionError('The CUDA device UUID differs from the explicit allocation')
    write_json(args.root / args.arm / 'launch_receipt.json', dict(updated_utc=utc(), pid=os.getpid(),
        stage=args.stage, allocation=allocation, cuda_visible_devices=os.environ['CUDA_VISIBLE_DEVICES'],
        observed_device_uuid=str(observed_uuid) if observed_uuid is not None else None,
        device_name=properties.name, no_other_gpu_used=True))
    if args.stage == 'smoke':
        smoke(args, device, allocation)
    else:
        train(args, device, allocation)


if __name__ == '__main__':
    main()
