"""Single-arm R45A on prepared R76 caches, with exact-coverage heterogeneous DDP.

Capacity is a disposable, single-GPU probe. Train/test require one four-rank
torchrun group; cache preparation and Slurm/tmux orchestration belong elsewhere.
W&B defaults to auto (offline without an explicit API key), with offline fallback.
Resume is from an R76 epoch-boundary checkpoint in this same locked run only.
"""

import argparse
import gc
import hashlib
import json
import math
import os
import random
import socket
import time
from collections import Counter, deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP

from code.unified_selector.registry import GLOBAL_SOLVERS, PROBLEMS
from .multitask_probe import task_seed
from .pairwise_objective import inputs_only, predicted_top3, single_objective
from .r40_experiment import aggregate, families
from .r42_experiment import CostController, symmetric_kl
from .r43_experiment import policy_metrics
from .r45_experiment import initialize as initialize_r45
from .solver_code_encoder import make_r45_model
from .train import configure_torch, set_seed


DATA_ROOT = Path('/public/home/shiys/selection_data_new_only_B_filtered_single_reld_mt_seed2_20261014')
DEFAULT_CAPS = (512, 512, 384, 384)
DEFAULT_LOGICAL_BATCH_SIZE = 9856
PART_KEYS = ('loss', 'ce', 'cmp', 'risk', 'kl', 'true3', 'pred3', 'all_pairs', 'grad_norm')
LOSS_RECIPE = dict(ce=.35, comparison=.35, risk=.02, cost_scale=.01,
                   focus_weights=dict(true3=.5, predicted3=.25, all_pairs=.25))


def utc():
    return datetime.now(timezone.utc).isoformat()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def save_checkpoint(path, value):
    path = Path(path)
    temporary = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    torch.save(value, temporary)
    temporary.replace(path)


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _caps(values):
    values = tuple(values)
    if not values or any(isinstance(n, bool) or not isinstance(n, (int, np.integer)) or n <= 0 for n in values):
        raise ValueError('Batch caps must be positive integers')
    return tuple(int(n) for n in values)


def problem_caps(sizes, batch_caps=DEFAULT_CAPS):
    if isinstance(batch_caps, dict):
        if set(batch_caps) != set(sizes):
            raise ValueError('Batch configuration must name every problem exactly once')
        result = {p: _caps(batch_caps[p]) for p in sizes}
    else:
        result = {p: _caps(batch_caps) for p in sizes}
    if len({len(v) for v in result.values()}) != 1:
        raise ValueError('Every problem must use the same rank count')
    return result


def epoch_plan(sizes, batch_caps=DEFAULT_CAPS, seed=2, epoch=0,
               logical_batch_size=DEFAULT_LOGICAL_BATCH_SIZE):
    """Pure seeded plan: active-task cycles, all real rows once, all tails retained."""
    if not sizes or epoch < 0 or any(isinstance(n, bool) or not isinstance(n, (int, np.integer)) or n <= 0
                                   for n in sizes.values()):
        raise ValueError('A plan needs positive task sizes and a nonnegative epoch')
    problem_caps(sizes, batch_caps)
    logical_batch_size, = _caps((logical_batch_size,))
    tasks = sorted(sizes, key=PROBLEMS.index)
    permutations = {p: torch.randperm(int(sizes[p]), generator=torch.Generator().manual_seed(
        task_seed(seed, p) + epoch * 1000003)) for p in tasks}
    remaining = {p: math.ceil(sizes[p] / logical_batch_size) for p in tasks}
    cursor, plan = Counter(), []
    rng = random.Random(seed + 701000003 + epoch * 1000003)
    while any(remaining.values()):
        cycle = [p for p in tasks if remaining[p]]
        rng.shuffle(cycle)
        for p in cycle:
            start = cursor[p] * logical_batch_size
            plan.append((p, permutations[p][start:start + logical_batch_size]))
            cursor[p] += 1
            remaining[p] -= 1
    return plan


def split_batch(indices, caps):
    """Split logical rows proportionally to microcaps, using largest remainders."""
    caps = _caps(caps)
    n, total = len(indices), sum(caps)
    counts = [n * cap // total for cap in caps]
    order = sorted(range(len(caps)), key=lambda rank: (-(n * caps[rank] % total), rank))
    for rank in order[:n - sum(counts)]:
        counts[rank] += 1
    result, offset = [], 0
    for count in counts:
        result.append(indices[offset:offset + count])
        offset += count
    return result


def rank_loss_scale(n_rank, n_global, world_size):
    if n_global <= 0 or not 0 <= n_rank <= n_global or world_size <= 0:
        raise ValueError('Invalid distributed batch counts')
    return world_size * n_rank / n_global


def slice_batch(batch, start, stop):
    """Slice only row-shaped tensors; solver/task descriptors stay shared."""
    if batch is None:
        return None
    count = len(batch['ind'])
    static = {'pool_ids', 'problem_id', 'kind', 'd_in'}
    return {key: value[start:stop] if (key not in static and torch.is_tensor(value)
            and value.ndim > 0 and value.shape[0] == count) else value
            for key, value in batch.items()}


class RDropObjective(nn.Module):
    """Both reused-parameter graphs and their losses live inside ONE DDP forward."""

    def __init__(self, selector):
        super().__init__()
        self.selector = selector

    def forward(self, batch, consistency_weight):
        if batch is None:
            # A sub-world-size tail has idle ranks, not duplicated dummy data.
            zero = next(self.selector.score_head.parameters()).sum() * 0.
            return zero, zero.detach().new_zeros(len(PART_KEYS) - 1)
        device = batch['ind'].device
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == 'cuda'):
            first = self.selector(inputs_only(batch))
            second = self.selector(inputs_only(batch))
        focus = predicted_top3(first['logits'], second['logits'], batch['pool_ids'], first['solver_mask'])
        # R45 does the reductions, BCE, risk and legal symmetric KL in FP32.
        loss1, parts1 = single_objective(first, batch, focus)
        loss2, parts2 = single_objective(second, batch, focus)
        kl = symmetric_kl(first['logits'], second['logits'], first['solver_mask'])
        loss = .5 * (loss1 + loss2) + consistency_weight * kl
        parts = {key: .5 * (parts1[key] + parts2[key]) for key in parts1}
        parts.update(loss=loss, kl=kl)
        return loss, torch.stack([parts[key].detach() for key in PART_KEYS[:-1]])


def capture_rank_rng(device):
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state(device) if device.type == 'cuda' else None)


def restore_rank_rng(state, device):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'].cpu())
    if state['cuda'] is not None:
        torch.cuda.set_rng_state(state['cuda'].cpu(), device)


@dataclass
class Runtime:
    rank: int
    world_size: int
    device: torch.device
    local_rank: int = 0

    def barrier(self):
        if self.world_size > 1:
            dist.barrier()

    def broadcast(self, value):
        if self.world_size == 1:
            return value
        payload = [value if self.rank == 0 else None]
        dist.broadcast_object_list(payload, src=0)
        return payload[0]

    def sum(self, value):
        value = value.clone()
        if self.world_size > 1:
            dist.all_reduce(value)
        return value


def update_batch(objective, optimizer, scaler, batch, consistency_weight, n_rank, n_global, runtime,
                 max_retries=20, microbatch_cap=None):
    rank_loss_scale(n_rank, n_global, runtime.world_size)
    microbatch_cap, = _caps((max(1, n_rank) if microbatch_cap is None else microbatch_cap,))
    rounds = torch.tensor(math.ceil(n_rank / microbatch_cap), dtype=torch.int64, device=runtime.device)
    if runtime.world_size > 1:
        dist.all_reduce(rounds, op=dist.ReduceOp.MAX)
    common_rounds = max(1, int(rounds.item()))
    rng = capture_rank_rng(runtime.device)
    for retry in range(max_retries):
        restore_rank_rng(rng, runtime.device)
        optimizer.zero_grad(set_to_none=True)
        parts_sum = torch.zeros(len(PART_KEYS) - 1, dtype=torch.float64, device=runtime.device)
        bad = torch.zeros((), dtype=torch.int32, device=runtime.device)
        for micro_round in range(common_rounds):
            start = min(micro_round * microbatch_cap, n_rank)
            stop = min(start + microbatch_cap, n_rank)
            count = stop - start
            # Exhausted nonempty ranks retain the task's graph on the final DDP sync.
            microbatch = slice_batch(batch, start, stop) if count else slice_batch(batch, 0, 1)
            sync = objective.no_sync() if runtime.world_size > 1 and micro_round + 1 < common_rounds else nullcontext()
            with sync:
                loss, parts = objective(microbatch, consistency_weight)
                scaler.scale(loss * rank_loss_scale(count, n_global, runtime.world_size)).backward()
            bad |= (~torch.isfinite(loss.detach())).to(torch.int32)
            if count:
                parts_sum.add_(parts.detach().double(), alpha=count)
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(objective.parameters(), 1.)
        bad |= (~torch.isfinite(grad_norm)).to(torch.int32)
        if runtime.world_size > 1:
            dist.all_reduce(bad, op=dist.ReduceOp.MAX)
        if bad.item():
            # Coordinate BEFORE scaler.step: no rank can update alone on overflow.
            if not scaler.is_enabled():
                if retry + 1 == max_retries:
                    break
            else:
                scaler.update(new_scale=scaler.get_scale() * .5)
            continue
        scaler.step(optimizer)
        scaler.update()
        parts = parts_sum / max(1, n_rank)
        return torch.cat((parts, grad_norm.detach().reshape(1))), retry
    raise RuntimeError('Nonfinite gradients persisted; no samples were skipped and no rank stepped')


class PinnedPrefetch:
    """One CPU worker and two bounded pinned batches; cache reads are node-local."""

    def __init__(self, caches, requests, device):
        self.caches, self.requests, self.device = caches, iter(requests), device
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='r76-prefetch')
        self.pending = deque()
        self.copy_stream = torch.cuda.Stream(device=device) if device.type == 'cuda' else None
        for _ in range(2):
            self._submit()

    def _prepare(self, request):
        problem, indices, n_global = request
        batch = self.caches[problem].batch(indices, device='cpu') if len(indices) else None
        if batch is not None and self.device.type == 'cuda':
            batch = {key: value.pin_memory() if torch.is_tensor(value) and not value.is_pinned() else value
                     for key, value in batch.items()}
        return problem, len(indices), n_global, batch

    def _submit(self):
        request = next(self.requests, None)
        if request is not None:
            self.pending.append(self.executor.submit(self._prepare, request))

    def __iter__(self):
        return self

    def __next__(self):
        if not self.pending:
            raise StopIteration
        problem, count, total, batch = self.pending.popleft().result()
        self._submit()
        if batch is not None and self.copy_stream is not None:
            with torch.cuda.stream(self.copy_stream):
                batch = {key: value.to(self.device, non_blocking=True) if torch.is_tensor(value) else value
                         for key, value in batch.items()}
            torch.cuda.current_stream(self.device).wait_stream(self.copy_stream)
            for value in batch.values():
                if torch.is_tensor(value):
                    value.record_stream(torch.cuda.current_stream(self.device))
        return problem, count, total, batch

    def __enter__(self):
        return self

    def __exit__(self, *unused):
        self.executor.shutdown(wait=True, cancel_futures=True)
        self.pending.clear()


def _numpy(value):
    return value.cpu().numpy() if torch.is_tensor(value) else np.asarray(value)


def open_cache(cache_root, split, problem, manifest, data_root, cache_factory=None):
    if cache_factory is None:
        from .r76_cache import ScenarioCache
        cache_factory = ScenarioCache
    metadata = manifest['splits'][split][problem]
    candidates = (Path(cache_root) / split / problem, Path(cache_root) / (problem + split))
    directory = next((path for path in candidates if path.is_dir()), None)
    if directory is None:
        raise FileNotFoundError(f'Prepare/copy the {problem}/{split} cache under {cache_root} first')
    cache = cache_factory(directory)
    rows = metadata.get('rows', metadata.get('size'))
    if (cache.problem != problem or cache.split != split or cache.size != rows
            or list(cache.pool) != list(metadata['pool'])
            or list(cache.pool_ids) != list(metadata['pool_ids'])):
        raise ValueError(f'Local cache disagrees with the run manifest: {problem}/{split}')
    if Path(cache.metadata['data_root']).resolve() != Path(data_root).resolve():
        raise ValueError('Cache belongs to a different data release')
    raw = cache.raw
    if (raw['costs'].dtype != np.float64 or raw['winner'].dtype != np.int64
            or raw['costs'].shape != (cache.size, len(cache.pool))
            or raw['winner'].shape != (cache.size,) or len(cache.lengths) != cache.size):
        raise ValueError('Cache must preserve original FP64 costs and native int64 winners')
    return cache


def fixed_sbs_metrics(scores, raw, baseline):
    result, pred = policy_metrics(scores, raw)
    result['native_correct'] = int(np.count_nonzero(pred == raw['winner']))
    if baseline['selection_split'] != 'train':
        raise ValueError('SBS must be fitted on TRAIN only')
    solver = baseline['sbs_solver']
    slot = list(raw['pool']).index(solver)
    if slot != baseline['sbs_slot']:
        raise ValueError('TRAIN SBS slot differs from the scenario pool order')
    fixed = float(raw['costs'][:, slot].mean(dtype=np.float64))
    result.update(posthoc_split_sbs_cost=result['sbs_cost'], posthoc_split_sbs_name=result['sbs_name'],
                  posthoc_split_vs_sbs_pct=result['vs_sbs_pct'],
                  train_sbs_mean_cost=float(baseline['mean_cost_by_solver'][solver]),
                  sbs=fixed, sbs_cost=fixed, sbs_name=solver, sbs_selection_split='train',
                  vs_sbs_pct=(result['mean_cost'] / fixed - 1.) * 100.,
                  vbs_gap_closed_pct=(fixed - result['mean_cost']) / (fixed - result['vbs']) * 100.
                  if fixed != result['vbs'] else 0.)
    return result, pred


@torch.no_grad()
def evaluate(selector, args, runtime, manifest, baselines, split, tag):
    directory = args.root / 'evaluation' / split / tag
    if runtime.rank == 0:
        directory.mkdir(parents=True, exist_ok=True)
        write_json(args.root / 'status.json', dict(phase='evaluation', split=split, tag=tag, updated_utc=utc()))
    runtime.barrier()
    was_training, arrays = selector.training, {}
    selector.eval()
    batch_size = min(args.eval_batch_size, 128) if runtime.rank >= 2 else args.eval_batch_size
    try:
        for problem in PROBLEMS[runtime.rank::runtime.world_size]:
            cache = open_cache(args.cache_root, split, problem, manifest, args.data_root)
            requests = ((problem, torch.arange(start, min(start + batch_size, cache.size)),
                         min(batch_size, cache.size - start)) for start in range(0, cache.size, batch_size))
            scores = []
            with PinnedPrefetch({problem: cache}, requests, runtime.device) as batches:
                for _, _, _, batch in batches:
                    scores.append(selector(inputs_only(batch))['logits'].float().cpu().numpy())
            scores = np.concatenate(scores)
            if len(scores) != cache.size:
                raise ValueError('Evaluation dropped a real tail row')
            prefix = problem + '__'
            arrays.update({prefix + key: value for key, value in dict(
                logits=scores, indices=np.arange(cache.size, dtype=np.int64),
                costs=cache.raw['costs'], winner=cache.raw['winner'], pool=np.asarray(cache.pool),
                pool_ids=np.asarray(cache.pool_ids, dtype=np.int64), nodes=_numpy(cache.lengths)).items()})
        path = directory / f'rank{runtime.rank:03d}.npz'
        temporary = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
        with temporary.open('wb') as stream:
            np.savez(stream, **arrays)
        temporary.replace(path)
    finally:
        selector.train(was_training)
    runtime.barrier()
    result = None
    if runtime.rank == 0:
        per_problem = {}
        for rank in range(runtime.world_size):
            with np.load(directory / f'rank{rank:03d}.npz', allow_pickle=False) as saved:
                for problem in PROBLEMS[rank::runtime.world_size]:
                    prefix = problem + '__'
                    raw = {key: saved[prefix + key] for key in ('costs', 'winner', 'pool', 'pool_ids')}
                    raw['pool'], raw['pool_ids'] = raw['pool'].tolist(), raw['pool_ids'].tolist()
                    rows = manifest['splits'][split][problem].get('rows',
                        manifest['splits'][split][problem].get('size'))
                    np.testing.assert_array_equal(saved[prefix + 'indices'], np.arange(rows))
                    per_problem[problem], _ = fixed_sbs_metrics(saved[prefix + 'logits'], raw, baselines[problem])
        if set(per_problem) != set(PROBLEMS):
            raise ValueError('Evaluation must aggregate all 18 tasks exactly once')
        per_problem = {p: per_problem[p] for p in PROBLEMS}
        result = dict(per_problem=per_problem, macro=aggregate(per_problem), families=families(per_problem),
                      native_total=sum(v['n'] for v in per_problem.values()),
                      native_correct=sum(v['native_correct'] for v in per_problem.values()),
                      sbs_reference='TRAIN-selected solver evaluated on this split; never post-hoc split SBS')
        write_json(directory / 'metrics.json', result)
    return runtime.broadcast(result)


def _batch_configuration(args):
    values = read_json(args.batch_config) if args.batch_config is not None else args.batch_caps
    return problem_caps(dict.fromkeys(PROBLEMS, 1), values)


def protocol(args, params, manifest, geometry, baselines, sizes, caps):
    return dict(experiment='R76', arm='R45A_handcrafted', seed=2, initialization='scratch',
        data_root=str(args.data_root), model_params=params, solver_names=GLOBAL_SOLVERS,
        batch_config={p: list(caps[p]) for p in PROBLEMS},
        logical_global_batch=args.logical_batch_size,
        microbatch_caps_by_problem={p: list(caps[p]) for p in PROBLEMS},
        global_batch_by_problem={p: args.logical_batch_size for p in PROBLEMS},
        updates_per_epoch={p: math.ceil(sizes[p] / args.logical_batch_size) for p in PROBLEMS},
        total_updates_per_epoch=sum(math.ceil(n / args.logical_batch_size) for n in sizes.values()),
        uniform_updates_per_task=len({math.ceil(n / args.logical_batch_size) for n in sizes.values()}) == 1,
        batch_deviation='Uniform logical batch replaces original R45 batch128; not strict replication; LR unchanged',
        world_size=4, rank_layout=['gpu05/4090', 'gpu05/4090', 'gpu03/3090', 'gpu03/3090'],
        sizes=sizes, max_epochs=args.epochs, eval_batch_size=args.eval_batch_size,
        eval_batch_size_3090=min(args.eval_batch_size, 128),
        learning_rate=1e-4, weight_decay=1e-4, optimizer='AdamW', dropout=.1, clip_norm=1.,
        warmup_epochs=3, rdrop_weight=.35, rdrop_ramp_epochs=3, loss=LOSS_RECIPE,
        lr_patience=3, min_epochs=15, stop_patience=8, cost_min_delta_pct=.001,
        checkpoint_rule='strict minimum complete validation macro_vs_sbs_pct, independent of controller min_delta',
        sbs_reference='TRAIN-selected solver, using its FP64 cost on each evaluated split',
        baselines=baselines, geometry_stats=geometry,
        data_manifest={s: manifest['splits'][s] for s in ('train', 'val')},
        full_coverage=True, drop_last=False, natural_sampling=True, no_augmentation=True,
        scheduling='seed2 shuffled active-task cycles; same problem on every DDP step',
        rank_rng='seed2 + 104729 * rank, restored unchanged on an AMP retry',
        distributed_loss='entire micro-objective including KL * world_size * actual_micro_rows / logical_global_rows',
        accumulation=dict(rank_split='largest-remainder proportional to per-problem microcaps',
            common_rounds='max_rank ceil(rank_logical_rows / rank_microcap)',
            synchronization='no_sync on every round except the last common round; zero-weight valid dummy when exhausted',
            gradients='same weights across all microchunks; one unscale, global clip_norm1 and optimizer step per logical batch',
            metrics='detached parts weighted by actual micro rows; dummy rows excluded',
            retry='restore rank RNG at logical start and replay all microchunks; no successful update advance'),
        find_unused_parameters=True, dropout_passes=2, optimizer_updates_per_batch=1,
        amp='FP16 GradScaler; collective finite-gradient decision before optimizer update',
        resume='same-run R76 epoch-boundary last.pt only; optimizer/controller/scaler/all rank RNG restored',
        prefetch='node-local cache; single async CPU worker, two pinned batch buffers',
        train_eval='every5, stopping epoch, maximum epoch and selected best', test_read=False)


def load_r76_checkpoint(path, device='cpu'):
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    config = checkpoint['args']
    if (config.get('experiment') != 'R76' or config.get('initialization') != 'scratch'
            or config['solver_names'] != GLOBAL_SOLVERS
            or config['model_params'].get('solver_representation') != 'handcrafted'):
        raise ValueError('Only a self-contained scratch R76 handcrafted checkpoint is accepted')
    model = make_r45_model(config['model_params'], state_dict=checkpoint['model'])
    return model.to(device), checkpoint


def _objective(selector, runtime):
    wrapped = RDropObjective(selector)
    if runtime.world_size > 1:
        kwargs = dict(find_unused_parameters=True, broadcast_buffers=False)
        if runtime.device.type == 'cuda':
            kwargs.update(device_ids=[runtime.local_rank], output_device=runtime.local_rank)
        wrapped = DDP(wrapped, **kwargs)
    return wrapped


def _wandb(args, config):
    if not args.wandb:
        return None
    try:
        import wandb
    except ImportError:
        print('[R76 W&B] package unavailable; local histories remain authoritative', flush=True)
        return None
    mode = args.wandb_mode
    if mode == 'auto':
        mode = 'online' if os.environ.get('WANDB_API_KEY') else 'offline'
    settings = dict(project='selector', entity='yjkds-southern-university-of-science-technology',
        name='R76_R45A_seed2', id='r76-' + hashlib.sha256(str(args.root).encode()).hexdigest()[:16],
        config=config, dir=str(args.root))
    try:
        return wandb.init(**settings, mode=mode, resume='allow' if mode == 'online' else None)
    except Exception as error:
        print(f'[R76 W&B] {type(error).__name__}; attempting offline fallback (credentials not logged)', flush=True)
        try:
            return wandb.init(**settings, mode='offline')
        except Exception as fallback:
            print(f'[R76 W&B] {type(fallback).__name__}; continuing with local histories', flush=True)
            return None


def _rng_states(runtime):
    state = capture_rank_rng(runtime.device)
    if runtime.world_size == 1:
        return [state]
    states = [None] * runtime.world_size
    dist.all_gather_object(states, state)
    return states


def _validate_best_epoch(checkpoint, epoch):
    if checkpoint.get('epoch') != epoch - 1 or checkpoint.get('best', {}).get('epoch') != epoch:
        raise ValueError('Selected best checkpoint epoch does not match best metadata')


def lock_best(root, best):
    path = Path(root) / 'best.pt'
    _validate_best_epoch(torch.load(path, map_location='cpu', weights_only=False), best['epoch'])
    record = dict(checkpoint=str(path.resolve()), checkpoint_sha256=file_hash(path), epoch=best['epoch'],
        experiment='R76', seed=2, groups=1, selected_without_test=True, no_per_problem_assembly=True,
        rule='strict minimum validation macro_vs_sbs_pct with TRAIN-fixed SBS', val=best['val']['macro'])
    lock = Path(root) / 'final_test_lock.json'
    if lock.exists():
        existing = read_json(lock)
        if any(existing.get(key) != value for key, value in record.items()):
            raise ValueError('The uniquely locked best checkpoint cannot change')
        return existing
    record['locked_utc'] = utc()
    with lock.open('x') as stream:
        json.dump(record, stream, indent=2, allow_nan=False)
        stream.write('\n')
    return record


def train(args, runtime):
    manifest = read_json(args.root / 'cache_manifest.json')
    geometry = read_json(args.root / 'geometry_stats.json')
    baselines = read_json(args.data_root / 'baselines.json')
    caps = _batch_configuration(args)
    if any(len(value) != runtime.world_size for value in caps.values()):
        raise ValueError('TRAIN batch configuration must contain exactly four rank caps per task')
    caches = {p: open_cache(args.cache_root, 'train', p, manifest, args.data_root) for p in PROBLEMS}
    sizes = {p: cache.size for p, cache in caches.items()}
    last = args.root / 'last.pt'
    if (args.root / 'training_complete.json').exists():
        raise ValueError('This single-arm training already completed')
    if last.exists() != args.resume:
        raise ValueError('Use --resume only for an existing R76 last.pt in this root')
    checkpoint = None
    if args.resume:
        selector, checkpoint = load_r76_checkpoint(last, runtime.device)
        params = checkpoint['args']['model_params']
        _validate_best_epoch(torch.load(args.root / 'best.pt', map_location='cpu', weights_only=False),
                             checkpoint['best']['epoch'])
    else:
        selector, params = initialize_r45('A', geometry, device=runtime.device, seed=2)
    config = protocol(args, params, manifest, geometry, baselines, sizes, caps)
    if runtime.rank == 0:
        path = args.root / 'protocol.json'
        if path.exists() and read_json(path) != config:
            raise ValueError('The locked protocol/data/caps changed; use a fresh root')
        if checkpoint is not None and checkpoint['args'] != config:
            raise ValueError('Resume must preserve this exact scratch R76 protocol')
        write_json(path, config)
        write_json(args.root / 'args.json', config)
    runtime.barrier()
    objective = _objective(selector, runtime)
    optimizer = torch.optim.AdamW(selector.parameters(), lr=1e-4, weight_decay=1e-4)
    scaler = torch.amp.GradScaler('cuda', enabled=runtime.device.type == 'cuda')
    controller = CostController(min_delta=.001, min_epochs=15, patience=8, lr_patience=3)
    history, best, successful, retries_total, peak_lr, start_epoch, stop = [], None, 0, 0, 1e-4, 0, False
    state = None
    counts = Counter({p: 0 for p in PROBLEMS})
    if checkpoint is not None:
        optimizer.load_state_dict(checkpoint['optimizer'])
        for state in optimizer.state.values():
            for key, value in state.items():
                if torch.is_tensor(value):
                    state[key] = value.cpu() if key == 'step' else value.to(runtime.device)
        scaler.load_state_dict(checkpoint['scaler'])
        vars(controller).update(checkpoint['controller'])
        history, best = checkpoint['history'], checkpoint['best']
        successful, counts = checkpoint['successful_updates'], Counter(checkpoint['updates_by_problem'])
        retries_total, peak_lr = checkpoint['amp_retries'], checkpoint['peak_lr']
        start_epoch, stop = checkpoint['completed_epochs'], checkpoint['stop']
        if len(checkpoint['rng_by_rank']) != runtime.world_size:
            raise ValueError('Resume requires the same rank count and rank RNG ownership')
        restore_rank_rng(checkpoint['rng_by_rank'][runtime.rank], runtime.device)
    else:
        set_seed(2 + 104729 * runtime.rank)
    wandb_run = _wandb(args, config) if runtime.rank == 0 else None
    steps = sum(config['updates_per_epoch'].values())
    runtime.barrier()
    try:
        for epoch in range(start_epoch, args.epochs):
            if stop:
                break
            selector.train()
            started, local_samples, covered = time.monotonic(), 0, Counter()
            weighted = torch.zeros(len(PART_KEYS) + 1, dtype=torch.float64, device=runtime.device)
            plan = epoch_plan(sizes, caps, seed=2, epoch=epoch, logical_batch_size=args.logical_batch_size)
            requests = ((p, split_batch(indices, caps[p])[runtime.rank], len(indices)) for p, indices in plan)
            if runtime.device.type == 'cuda':
                torch.cuda.reset_peak_memory_stats(runtime.device)
            with PinnedPrefetch(caches, requests, runtime.device) as batches:
                for step, (problem, n_rank, n_global, batch) in enumerate(batches):
                    progress = min(1., (successful + 1) / (3 * steps))
                    lr = 1e-4 * progress if epoch < 3 else peak_lr
                    for group in optimizer.param_groups:
                        group['lr'] = lr
                    parts, retries = update_batch(objective, optimizer, scaler, batch, .35 * progress,
                                                  n_rank, n_global, runtime, microbatch_cap=caps[problem][runtime.rank])
                    successful += 1
                    retries_total += retries
                    counts[problem] += 1
                    covered[problem] += n_global
                    local_samples += n_rank
                    weighted[:-1].add_(parts.double() * n_rank)
                    weighted[-1] += n_rank
                    if (step + 1) % args.log_every == 0 or step + 1 == steps:
                        totals = runtime.sum(weighted).cpu().tolist()
                        info = dict(epoch=epoch + 1, step=step + 1, steps=steps, problem=problem,
                            successful_updates=successful, samples=int(totals[-1]), lr=lr,
                            consistency_weight=.35 * progress, amp_retries=retries_total,
                            seconds=time.monotonic() - started,
                            **{key: totals[j] / totals[-1] for j, key in enumerate(PART_KEYS)})
                        if runtime.rank == 0:
                            with (args.root / 'updates.jsonl').open('a') as stream:
                                stream.write(json.dumps(dict(updated_utc=utc(), **info), allow_nan=False) + '\n')
                            write_json(args.root / 'status.json', dict(phase='training', updated_utc=utc(), **info))
                            print('[R76 train] ' + json.dumps(info), flush=True)
            totals = runtime.sum(weighted).cpu().tolist()
            if dict(covered) != sizes or int(totals[-1]) != sum(sizes.values()):
                raise ValueError('The epoch must cover every actual training row exactly once')
            optimization = {key: totals[j] / totals[-1] for j, key in enumerate(PART_KEYS)}
            optimization.update(samples=int(totals[-1]), seconds=time.monotonic() - started,
                                lr=lr, consistency_weight=.35 * progress, amp_retries=retries_total)
            validation = evaluate(selector, args, runtime, manifest, baselines, 'val', f'epoch{epoch + 1:03d}')
            cost = validation['macro']['macro_vs_sbs_pct']
            if not math.isfinite(cost):
                raise ValueError('Validation selection cost must be finite')
            halve, stop = controller.observe(cost, epoch + 1)
            training = evaluate(selector, args, runtime, manifest, baselines, 'train', f'epoch{epoch + 1:03d}') if (
                (epoch + 1) % 5 == 0 or stop or epoch + 1 == args.epochs) else None
            row = dict(epoch=epoch + 1, successful_updates=successful, updates_by_problem=dict(counts),
                       optimization=optimization, train=training, val=validation)
            history.append(row)
            improved = best is None or cost < best['val']['macro']['macro_vs_sbs_pct']
            if improved:
                best = dict(row)
            if halve:
                peak_lr *= .5
            state = dict(model=selector.state_dict(), optimizer=optimizer.state_dict(), scaler=scaler.state_dict(),
                args=config, epoch=epoch, completed_epochs=epoch + 1, macro=validation['macro'],
                successful_updates=successful, updates_by_problem=dict(counts), amp_retries=retries_total,
                rng_by_rank=_rng_states(runtime), controller=vars(controller).copy(), peak_lr=peak_lr,
                history=history, best=best, stop=stop)
            if runtime.rank == 0:
                if improved:
                    save_checkpoint(args.root / 'best.pt', state)
                    write_json(args.root / 'best_eval.json', best)
                save_checkpoint(last, state)
                write_json(args.root / 'history.json', history)
                write_json(args.root / 'status.json', dict(phase='epoch_complete', updated_utc=utc(),
                    epoch=epoch + 1, selected_epoch=best['epoch'], successful_updates=successful,
                    val=validation['macro'], peak_lr=peak_lr, stop=stop))
                if wandb_run is not None:
                    values = dict(epoch=epoch + 1, successful_updates=successful,
                                  **{'optimization/' + key: value for key, value in optimization.items()})
                    for split, value in (('train_eval', training), ('val', validation)):
                        if value is not None:
                            values.update({split + '/' + key: score for key, score in value['macro'].items()})
                    wandb_run.log(values)
            runtime.barrier()
        if best is None:
            raise ValueError('Training has no validation-selected checkpoint')
        del objective, optimizer, scaler, checkpoint, state, selector
        selector, selected_checkpoint = load_r76_checkpoint(args.root / 'best.pt', runtime.device)
        _validate_best_epoch(selected_checkpoint, best['epoch'])
        selected_train = best['train']
        if selected_train is None:
            selected_train = evaluate(selector, args, runtime, manifest, baselines, 'train', 'selected_best')
        best = dict(best, train=selected_train)
        if runtime.rank == 0:
            for row in history:
                if row['epoch'] == best['epoch']:
                    row['train'] = selected_train
            write_json(args.root / 'history.json', history)
            write_json(args.root / 'best_eval.json', best)
            write_json(args.root / 'train_eval_curve.json', [dict(epoch=row['epoch'],
                selected_best=row['epoch'] == best['epoch'], train=row['train']) for row in history if row['train'] is not None])
            lock = lock_best(args.root, best)
            final = dict(best=best, final=history[-1], epochs_completed=len(history),
                successful_updates=successful, updates_by_problem=dict(counts),
                stop_reason='validation early stop' if stop else 'maximum epochs', test_read=False)
            write_json(args.root / 'result.json', final)
            write_json(args.root / 'final_metrics.json', dict(epoch=best['epoch'], train=selected_train,
                val=best['val'], checkpoint_sha256=lock['checkpoint_sha256'], test_read=False))
            write_json(args.root / 'training_complete.json', dict(completed_utc=utc(), selected_best=lock,
                epochs_completed=len(history), next_stage='prepare TEST caches, then four-rank --stage test', test_read=False))
            write_json(args.root / 'status.json', dict(phase='training_complete', updated_utc=utc(),
                selected_epoch=best['epoch'], successful_updates=successful, test_read=False))
            if wandb_run is not None:
                wandb_run.log({'selected_train/' + key: value for key, value in selected_train['macro'].items()})
        runtime.barrier()
    finally:
        if wandb_run is not None:
            wandb_run.finish()


def authorize_test(root, data_root):
    root = Path(root)
    if not (root / 'training_complete.json').is_file():
        raise PermissionError('TRAIN must complete and lock its unique best before TEST')
    lock = read_json(root / 'final_test_lock.json')
    config = read_json(root / 'protocol.json')
    path = root / 'best.pt'
    if (lock.get('experiment') != 'R76' or not lock.get('selected_without_test')
            or Path(lock['checkpoint']).resolve() != path.resolve()
            or lock['checkpoint_sha256'] != file_hash(path)
            or Path(config['data_root']).resolve() != Path(data_root).resolve()):
        raise PermissionError('TEST requires the unchanged uniquely selected R76 best')
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    try:
        _validate_best_epoch(checkpoint, lock['epoch'])
    except ValueError as error:
        raise PermissionError(str(error)) from error
    if checkpoint.get('args') != config:
        raise PermissionError('The TEST checkpoint differs from the locked protocol')
    if (root / 'test_started.json').exists() or (root / 'test_results.json').exists():
        raise PermissionError('The single selected-best TEST has already been claimed; no automatic repeat')
    return lock, config


def test(args, runtime):
    lock, config = authorize_test(args.root, args.data_root)
    runtime.barrier()
    if runtime.rank == 0:
        with (args.root / 'test_started.json').open('x') as stream:
            json.dump(dict(started_utc=utc(), checkpoint_sha256=lock['checkpoint_sha256'],
                           selected_epoch=lock['epoch'], previous_full_tests=0), stream, indent=2)
    runtime.barrier()
    # Nothing above this point imports ScenarioCache or opens any TEST inputs.
    manifest = read_json(args.root / 'cache_manifest.json')
    selector, checkpoint = load_r76_checkpoint(args.root / 'best.pt', runtime.device)
    if checkpoint['args'] != config or checkpoint['epoch'] + 1 != lock['epoch']:
        raise PermissionError('The TEST checkpoint differs from the locked protocol/epoch')
    result = evaluate(selector, args, runtime, manifest, config['baselines'], 'test', 'selected_best')
    if runtime.rank == 0:
        if file_hash(args.root / 'best.pt') != lock['checkpoint_sha256']:
            raise PermissionError('The locked checkpoint changed during TEST')
        write_json(args.root / 'test_results.json', dict(result=result, epoch=lock['epoch'],
            checkpoint_sha256=lock['checkpoint_sha256'], complete=True, completed_utc=utc()))
        write_json(args.root / 'test_complete.json', dict(completed_utc=utc(),
            checkpoint_sha256=lock['checkpoint_sha256'], full_tests=1))
        write_json(args.root / 'status.json', dict(phase='test_complete', updated_utc=utc(), selected_epoch=lock['epoch']))
    runtime.barrier()


def capacity_trial(cache, geometry, count, device):
    selector = optimizer = objective = scaler = batch = None
    try:
        torch.cuda.empty_cache()
        selector, _ = initialize_r45('A', geometry, device=device, seed=2)
        selector.train()
        objective = RDropObjective(selector)
        optimizer = torch.optim.AdamW(selector.parameters(), lr=1e-4, weight_decay=1e-4)
        scaler = torch.amp.GradScaler('cuda')
        runtime = Runtime(0, 1, device)
        base = max(torch.cuda.memory_allocated(device), torch.cuda.memory_reserved(device))
        indices = torch.from_numpy(np.argsort(-_numpy(cache.lengths), kind='stable')[:count].copy())
        torch.cuda.reset_peak_memory_stats(device)
        with PinnedPrefetch({cache.problem: cache}, [(cache.problem, indices, count)] * 2, device) as batches:
            times, retries = [], 0
            for _, n_rank, n_global, batch in batches:
                torch.cuda.synchronize(device)
                started = time.monotonic()
                _, skipped = update_batch(objective, optimizer, scaler, batch, .35, n_rank, n_global, runtime)
                torch.cuda.synchronize(device)
                times.append(time.monotonic() - started)
                retries += skipped
        allocated = torch.cuda.max_memory_allocated(device) / 2**30
        reserved = torch.cuda.max_memory_reserved(device) / 2**30
        return dict(cap=count, peak_allocated_gib=allocated, peak_reserved_gib=reserved,
            base_gib=base / 2**30, step_seconds=times[-1], step_seconds_all=times,
            max_nodes=int(_numpy(cache.lengths)[indices.numpy()].max()), amp_retries=retries, oom=False)
    finally:
        selector = optimizer = objective = scaler = batch = None
        gc.collect()
        torch.cuda.empty_cache()


def measured_capacity(cache, geometry, initial, device, target_gib=21.):
    count = min(initial, cache.size)
    trials, best, grew, backoffs = [], None, False, 0
    while True:
        try:
            row = capacity_trial(cache, geometry, count, device)
            trials.append(row)
            peak = max(row['peak_allocated_gib'], row['peak_reserved_gib'])
        except torch.cuda.OutOfMemoryError:
            trials.append(dict(cap=count, oom=True))
            backoffs += 1
            if backoffs > 2:
                if best is None:
                    raise RuntimeError(f'{cache.problem}: no viable cap after two OOM halve backoffs')
                break
            count = max(1, count // 2)
            if best is not None and count <= best['cap']:
                break
            continue
        if peak > target_gib:
            backoffs += 1
            if backoffs > 2:
                if best is None:
                    raise RuntimeError(f'{cache.problem}: no measured cap within the memory target')
                break
            count = max(1, min(count - 1, int(count * target_gib / peak * .92)))
            if best is not None and count <= best['cap']:
                break
            continue
        if best is None or row['cap'] > best['cap']:
            best = row
        if grew:
            break
        estimate = int(count * max(.1, target_gib - row['base_gib']) / max(.1, peak - row['base_gib']) * .92)
        estimate = min(cache.size, max(count, estimate // 32 * 32))
        if estimate <= count:
            break
        grew, count = True, estimate
    return dict(best, trials=trials, initial_cap=initial, target_memory_gib=target_gib)


def capacity(args):
    if any((args.root / name).exists() for name in ('last.pt', 'best.pt', 'training_complete.json')):
        raise ValueError('Capacity is pre-formal only; do not probe an existing training run')
    device = torch.device(args.device)
    if device.type != 'cuda' or not torch.cuda.is_available():
        raise RuntimeError('Capacity needs one real GPU, but no initialized distributed group')
    torch.cuda.set_device(device)
    name = torch.cuda.get_device_name(device)
    kind = args.gpu_type or ('4090' if '4090' in name else '3090' if '3090' in name else None)
    if kind is None or kind not in name:
        raise ValueError('Capacity must run on the declared 4090 or 3090 representative card')
    manifest = read_json(args.root / 'cache_manifest.json')
    geometry = read_json(args.root / 'geometry_stats.json')
    initial_big, initial_small = (512, 2048) if kind == '4090' else (384, 1536)
    result = dict(gpu_type=kind, gpu_name=name, host=socket.gethostname(), device=str(device),
        created_utc=utc(), target_memory_gib=args.capacity_target_gib,
        policy='one estimated upward trial, at most two OOM/memory-budget backoffs; two real steps per trial',
        formal_weights_retained=False, distributed_initialized=False, problems={})
    for problem in ('TSP', 'CVRP', 'OVRPTW', 'ATSP'):
        cache = open_cache(args.cache_root, 'train', problem, manifest, args.data_root)
        initial = initial_big if problem in ('TSP', 'CVRP') else initial_small
        result['problems'][problem] = measured_capacity(cache, geometry, initial, device, args.capacity_target_gib)
        write_json(args.root / f'capacity_{kind}.json', result)
        print('[R76 capacity] ' + json.dumps(dict(problem=problem, **result['problems'][problem])), flush=True)
    result['recommended_caps'] = {p: result['problems'][p if p in ('TSP', 'CVRP', 'ATSP') else 'OVRPTW']['cap']
                                  for p in PROBLEMS}
    result.update(complete=True, completed_utc=utc())
    write_json(args.root / f'capacity_{kind}.json', result)


def batch_config(args):
    """CPU-only receipt merge; this phase neither opens caches nor constructs a model."""
    if any((args.root / name).exists() for name in ('protocol.json', 'last.pt', 'training_complete.json')):
        raise ValueError('Batch caps must be fixed before formal training')
    receipts = {kind: read_json(args.root / f'capacity_{kind}.json') for kind in ('4090', '3090')}
    for kind, receipt in receipts.items():
        if (receipt.get('gpu_type') != kind or not receipt.get('complete')
                or receipt.get('formal_weights_retained') is not False
                or set(receipt['recommended_caps']) != set(PROBLEMS)):
            raise ValueError(f'A complete disposable capacity_{kind}.json is required')
        _caps(receipt['recommended_caps'].values())
    values = {p: [receipts['4090']['recommended_caps'][p]] * 2 +
                  [receipts['3090']['recommended_caps'][p]] * 2 for p in PROBLEMS}
    path = args.root / 'batch_config.json'
    if path.exists() and read_json(path) != values:
        raise ValueError('Measured batch configuration already exists with different caps')
    write_json(path, values)
    manifest = read_json(args.root / 'cache_manifest.json')
    sizes = {p: manifest['splits']['train'][p]['rows'] for p in PROBLEMS}
    write_json(args.root / 'batch_config_receipt.json', dict(created_utc=utc(),
        capacity_receipts={kind: file_hash(args.root / f'capacity_{kind}.json') for kind in receipts},
        global_batch_by_problem={p: sum(values[p]) for p in PROBLEMS},
        updates_per_epoch={p: math.ceil(sizes[p] / sum(values[p])) for p in PROBLEMS},
        learning_rate=1e-4, lr_auto_scaling=False, natural_full_coverage=True, tails_retained=True))
    print('[R76 batch_config] ' + str(path), flush=True)


def distributed_runtime():
    rank, world, local = (int(os.environ.get(key, default)) for key, default in
                          (('RANK', 0), ('WORLD_SIZE', 1), ('LOCAL_RANK', 0)))
    if world != 4 or not torch.cuda.is_available():
        raise RuntimeError('TRAIN/TEST require exactly four CUDA ranks in one torchrun group')
    if local != rank % 2:
        raise ValueError('Expected two local ranks on gpu05, then two on gpu03')
    device = torch.device('cuda', local)
    torch.cuda.set_device(device)
    expected = '4090' if rank < 2 else '3090'
    if expected not in torch.cuda.get_device_name(device):
        raise ValueError('Expected gpu05 4090 ranks0/1, gpu03 3090 ranks2/3')
    dist.init_process_group('nccl', timeout=timedelta(minutes=60))
    return Runtime(rank, world, device, local)


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('--stage', choices=('capacity', 'batch_config', 'train', 'test'), required=True)
    result.add_argument('--root', type=Path, required=True)
    result.add_argument('--data-root', type=Path, default=DATA_ROOT)
    result.add_argument('--cache-root', type=Path, required=True, help='prepared node-local cache, normally /tmp')
    result.add_argument('--batch-caps', type=lambda text: _caps(int(v) for v in text.split(',')), default=DEFAULT_CAPS)
    result.add_argument('--batch-config', type=Path, help='JSON mapping each of all18 problems to four rank caps')
    result.add_argument('--logical-batch-size', type=int, default=DEFAULT_LOGICAL_BATCH_SIZE)
    result.add_argument('--epochs', type=int, default=40)
    result.add_argument('--eval-batch-size', type=int, default=256)
    result.add_argument('--resume', action='store_true')
    result.add_argument('--device', default='cuda:0', help='single-GPU capacity device; torchrun controls train/test')
    result.add_argument('--gpu-type', choices=('4090', '3090'), help='capacity representative type, otherwise detected')
    result.add_argument('--capacity-target-gib', type=float, default=21.)
    result.add_argument('--log-every', type=int, default=180)
    result.add_argument('--wandb', action=argparse.BooleanOptionalAction, default=True)
    result.add_argument('--wandb-mode', choices=('auto', 'online', 'offline'), default=os.environ.get('WANDB_MODE', 'auto'))
    return result


def main():
    cli = parser()
    args = cli.parse_args()
    if not 1 <= args.epochs <= 40 or args.eval_batch_size < 1 or args.log_every < 1 or args.logical_batch_size < 1:
        cli.error('Use at most40 epochs and positive logical/evaluation batches and log intervals')
    if not 0 < args.capacity_target_gib <= 21.:
        cli.error('Capacity target must leave headroom: 0 < target <=21 GiB')
    if args.resume and args.stage != 'train':
        cli.error('--resume is limited to same-run R76 training')
    args.root, args.data_root, args.cache_root = (path.expanduser().resolve() for path in
                                                (args.root, args.data_root, args.cache_root))
    args.root.mkdir(parents=True, exist_ok=True)
    configure_torch()
    torch.set_num_threads(4)
    if args.stage == 'capacity':
        capacity(args)
        return
    if args.stage == 'batch_config':
        batch_config(args)
        return
    runtime = distributed_runtime()
    try:
        if args.stage == 'train':
            train(args, runtime)
        else:
            test(args, runtime)
    except Exception as error:
        if runtime.rank == 0:
            write_json(args.root / 'status.json', dict(phase='failed', stage=args.stage,
                updated_utc=utc(), error_type=type(error).__name__))
        raise
    finally:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
