"""R120: one full-pool selector for 18 routing and six typed nonrouting tasks.

Assumptions (base 048664148; final scientific contract in implement-stage):
uniform task cycles, natural TRAIN shuffles, native FP64 supervision, and fixed
TRAIN-SBS objective units. Formal runs require all 24 complete measured pools.
CPU smoke runs are explicitly separate. DDP splits one logical batch, rather
than training independent GPU-specific models. Validation alone locks one state.
"""

from __future__ import annotations

import argparse
import copy
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import timedelta
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import random
import socket
import time

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from .r120_reporting import (NONROUTING_PROBLEMS, PROBLEMS, ROUTING_PROBLEMS,
    aggregate_metrics, fit_train_reference, native_task_metrics, read_json, write_json)


V4 = Path(__file__).resolve().parent
DEFAULT_ROOT = V4 / 'runs/R120_multidomain_selector'
DEFAULT_PREPARED = V4 / 'runs/R84_new90k_attention/prepared'
DEFAULT_NONROUTING = Path('/public/home/shiys/selection_data_multidomain_seed2_20261029')
ROUTING_INPUT_KEYS = ('problem_id', 'kind', 'n', 'node_mask', 'cbits', 'problem_desc',
    'coord_dist', 'pool_ids', 'mask', 'solver_mask', 'node', 'node_geom', 'matrix')
MAX_UPDATES, WARMUP_UPDATES, EVAL_EVERY = 57600, 4320, 1440
PEAK_LR, END_LR = 1e-4, 1e-6


def utc():
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def save_checkpoint(path, state):
    path = Path(path)
    temporary = path.with_name('.' + path.name + f'.{os.getpid()}.tmp')
    torch.save(state, temporary)
    temporary.replace(path)


@dataclass
class Runtime:
    rank: int
    world_size: int
    device: torch.device

    @property
    def primary(self):
        return self.rank == 0

    def barrier(self):
        if self.world_size > 1:
            dist.barrier()


def distributed_runtime(device='cuda'):
    world = int(os.environ.get('WORLD_SIZE', '1'))
    rank, local = int(os.environ.get('RANK', '0')), int(os.environ.get('LOCAL_RANK', '0'))
    target = torch.device(device)
    if target.type == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError('Requested CUDA is unavailable; no silent CPU training fallback')
        index = local if world > 1 else (target.index or 0)
        torch.cuda.set_device(index)
        target = torch.device('cuda', index)
    if world > 1 and not dist.is_initialized():
        dist.init_process_group('nccl' if target.type == 'cuda' else 'gloo',
                                timeout=timedelta(minutes=90))
    return Runtime(rank, world, target)


def gather_objects(value, runtime):
    if runtime.world_size == 1:
        return [value]
    output = [None] * runtime.world_size if runtime.primary else None
    dist.gather_object(value, output, dst=0)
    return output


def broadcast_object(value, runtime):
    if runtime.world_size > 1:
        box = [value if runtime.primary else None]
        dist.broadcast_object_list(box, src=0)
        return box[0]
    return value


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def rng_state(device):
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state(device) if device.type == 'cuda' else None)


def restore_rng(state, device):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    if state['cuda'] is not None:
        torch.cuda.set_rng_state(state['cuda'], device)


def learning_rate(update, *, max_updates=MAX_UPDATES, warmup_updates=WARMUP_UPDATES,
                  peak_lr=PEAK_LR, end_lr=END_LR):
    if not 1 <= update <= max_updates or not 0 <= warmup_updates <= max_updates:
        raise ValueError('Learning rate uses the successful one-based update axis')
    if update <= warmup_updates:
        return peak_lr * update / warmup_updates
    progress = (update - warmup_updates) / max(1, max_updates - warmup_updates)
    return end_lr + .5 * (peak_lr - end_lr) * (1 + math.cos(math.pi * progress))


def task_order(count, tasks=PROBLEMS, seed=2):
    generator = np.random.default_rng(seed)
    result = []
    while len(result) < count:
        result.extend(generator.permutation(tasks).tolist())
    return result[:count]


def rank_indices(indices, rank, world_size, weights=None):
    """Disjoint real rows, including uneven boundaries without padding duplicates."""
    if weights is None:
        start = len(indices) * rank // world_size
        stop = len(indices) * (rank + 1) // world_size
    else:
        if len(weights) != world_size or any(value <= 0 for value in weights):
            raise ValueError('Rank weights must be positive and match the distributed world size')
        total = sum(weights)
        start = len(indices) * sum(weights[:rank]) // total
        stop = len(indices) * sum(weights[:rank + 1]) // total
    return indices[start:stop]


def move_to(value, device, *, pin=False):
    if torch.is_tensor(value):
        if pin and device.type == 'cuda' and value.device.type == 'cpu' and not value.is_pinned():
            value = value.pin_memory()
        return value.to(device, non_blocking=device.type == 'cuda')
    if isinstance(value, dict):
        return {key: move_to(item, device, pin=pin) for key, item in value.items()}
    return value


def model_inputs(batch):
    """A positive interface allowlist; never pass metadata or label namespaces."""
    return {key: batch[key] for key in ('problem', 'inputs', 'pool_ids', 'solver_mask')}


def slice_batch(batch, start, stop):
    count = batch['labels']['native_objective'].shape[0]

    def sliced(value, key=''):
        if torch.is_tensor(value):
            if key == 'pool_ids' and value.ndim == 1:
                return value
            return value[start:stop] if value.ndim and value.shape[0] == count else value
        if isinstance(value, dict):
            return {name: sliced(item, name) for name, item in value.items()}
        if isinstance(value, list) and len(value) == count:
            return value[start:stop]
        return value

    return sliced(batch)


def _manifest_path(root):
    root = Path(root)
    return root / 'manifest.json' if root.is_dir() else root


def merged_nonrouting_manifest(nonrouting_root, supplement_root=None):
    primary = read_json(_manifest_path(nonrouting_root))
    result = copy.deepcopy(primary)
    result['merged_sources'] = [str(_manifest_path(nonrouting_root).resolve())]
    if supplement_root is not None:
        supplement = read_json(_manifest_path(supplement_root))
        if supplement['schema_version'] != result['schema_version']:
            raise ValueError('Supplement raw schema differs from the ready-three export')
        for task, entry in supplement['nonrouting'].items():
            if task not in ('BPP', 'MKP', 'SMTWTP'):
                if task in result['nonrouting'] and result['nonrouting'][task] != entry:
                    raise ValueError('Supplement must not replace an existing frozen ready-three task')
                continue
            result['nonrouting'][task] = copy.deepcopy(entry)
        result['merged_sources'].append(str(_manifest_path(supplement_root).resolve()))
    result['all_six_training_pools_ready'] = all(
        result['nonrouting'].get(p, {}).get('ready_for_selector_training', False)
        and all(s in result['nonrouting'][p].get('splits', {}) for s in ('train', 'val', 'test'))
        for p in NONROUTING_PROBLEMS)
    return result


class RoutingTask:
    def __init__(self, cache_root, split, problem, manifest, data_root):
        from .r76_experiment import open_cache
        self.cache = open_cache(cache_root, split, problem, manifest, data_root)
        self.problem, self.split = problem, split
        self.candidate_solver_ids = tuple(self.cache.pool)
        self.objective_direction = 'min'
        self.label_arrays = dict(native_objective=self.cache.raw['costs'],
            native_winner_index=self.cache.raw['winner'])

    def __len__(self):
        return self.cache.size

    def batch(self, indices, pool_ids):
        indices = np.asarray(indices, dtype=np.int64)
        original = self.cache.batch(indices, device='cpu')
        inputs = {key: original[key] for key in ROUTING_INPUT_KEYS if key in original}
        costs = np.asarray(self.cache.raw['costs'][indices], dtype=np.float64)
        best = costs == costs.min(1, keepdims=True)
        return dict(problem=self.problem, inputs=inputs,
            pool_ids=torch.tensor(pool_ids, dtype=torch.long),
            solver_mask=torch.ones(costs.shape, dtype=torch.bool),
            labels=dict(native_objective=torch.from_numpy(costs),
                observed_mask=torch.ones(costs.shape, dtype=torch.bool),
                all_best_mask=torch.from_numpy(best), direction='min',
                native_winner_index=original['ind']), row_indices=torch.from_numpy(indices))

    def close(self):
        self.cache.close()


class NonroutingTask:
    def __init__(self, manifest_path, problem, split):
        from .nonrouting_data import NonRoutingDataset
        self.dataset = NonRoutingDataset(manifest_path, problem, split)
        self.problem, self.split = problem, split
        self.candidate_solver_ids = self.dataset.candidate_solver_ids
        self.objective_direction = self.dataset.objective_direction
        self.label_arrays = self.dataset.label_arrays

    def __len__(self):
        return len(self.dataset)

    def batch(self, indices, pool_ids):
        from .nonrouting_data import collate_nonrouting
        indices = np.asarray(indices, dtype=np.int64)
        result = collate_nonrouting([self.dataset[int(index)] for index in indices])
        labels = dict(result['labels'])
        labels['direction'] = labels.pop('objective_direction')
        return dict(problem=self.problem, inputs=result['inputs'], labels=labels,
            pool_ids=torch.tensor(pool_ids, dtype=torch.long),
            solver_mask=torch.ones(labels['observed_mask'].shape, dtype=torch.bool),
            metadata=result['metadata'], row_indices=torch.from_numpy(indices))

    def close(self):
        self.dataset.close()


class DataStore:
    def __init__(self, prepared, *, cache_root=None):
        self.prepared = prepared
        self.cache_root = Path(cache_root or prepared['cache_root'])
        self.tasks = {}

    def open(self, problem, split):
        key = problem, split
        if key not in self.tasks:
            if problem in ROUTING_PROBLEMS:
                self.tasks[key] = RoutingTask(self.cache_root, split, problem,
                    self.prepared['routing_manifest'], self.prepared['routing_data_root'])
            else:
                self.tasks[key] = NonroutingTask(self.prepared['nonrouting_manifest_path'], problem, split)
        return self.tasks[key]

    def batch(self, problem, split, indices, pools):
        return self.open(problem, split).batch(indices, pools[problem])

    def close_split(self, split):
        for key in list(self.tasks):
            if key[1] == split:
                self.tasks.pop(key).close()

    def close(self):
        for task in self.tasks.values():
            task.close()
        self.tasks.clear()


class BatchPrefetch:
    """One CPU reader, two bounded batches, and a separate CUDA transfer stream.

    Requests contain indices already drawn for exactly one validation interval.
    The caller consumes that entire finite interval before saving stream cursors.
    """

    def __init__(self, data, requests, pools, device):
        self.data, self.requests, self.pools, self.device = data, iter(requests), pools, device
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='r120-prefetch')
        self.pending = deque()
        self.copy_stream = torch.cuda.Stream(device=device) if device.type == 'cuda' else None
        for _ in range(2):
            self._submit()

    def _prepare(self, request):
        step, problem, indices = request
        batch = self.data.batch(problem, 'train', indices, self.pools)

        def pin(value):
            if torch.is_tensor(value):
                return value.pin_memory() if not value.is_pinned() else value
            if isinstance(value, dict):
                return {key: pin(item) for key, item in value.items()}
            return value

        if self.device.type == 'cuda':
            torch.cuda.set_device(self.device)
            batch = pin(batch)
        return step, problem, batch

    def _submit(self):
        request = next(self.requests, None)
        if request is not None:
            self.pending.append(self.executor.submit(self._prepare, request))

    def __iter__(self):
        return self

    def __next__(self):
        if not self.pending:
            raise StopIteration
        step, problem, batch = self.pending.popleft().result()
        self._submit()
        if self.copy_stream is not None:
            with torch.cuda.stream(self.copy_stream):
                batch = move_to(batch, self.device)
            current = torch.cuda.current_stream(self.device)
            current.wait_stream(self.copy_stream)

            def record(value):
                if torch.is_tensor(value) and value.is_cuda:
                    value.record_stream(current)
                elif isinstance(value, dict):
                    for item in value.values():
                        record(item)

            record(batch)
        return step, problem, batch

    def __enter__(self):
        return self

    def __exit__(self, *unused):
        self.executor.shutdown(wait=True, cancel_futures=True)
        self.pending.clear()


def prepare(args):
    """Fit only lightweight TRAIN references, using already-existing raw caches."""
    from code.unified_selector.registry import GLOBAL_SOLVERS
    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)
    r84 = Path(args.routing_prepared_root)
    routing_ready = read_json(r84 / 'prepared.json')
    routing_manifest = read_json(r84 / 'source_cache_manifest.json')
    manifest = merged_nonrouting_manifest(args.nonrouting_root, args.supplement_root)
    tasks = list(ROUTING_PROBLEMS)
    tasks.extend(p for p in NONROUTING_PROBLEMS if
                 manifest['nonrouting'].get(p, {}).get('ready_for_selector_training'))
    ready24 = set(tasks) == set(PROBLEMS) and manifest['all_six_training_pools_ready']
    if args.smoke_tasks:
        if not args.smoke_updates:
            raise ValueError('--smoke-tasks may only restrict an explicitly marked smoke run')
        requested = args.smoke_tasks.split(',')
        if not set(requested) <= set(tasks):
            raise ValueError('Smoke task selection includes pending or unknown data')
        tasks = [problem for problem in tasks if problem in requested]
        ready24 = ready24 and set(tasks) == set(PROBLEMS)
    if not ready24 and not args.smoke_updates:
        missing = sorted(set(PROBLEMS) - set(tasks))
        raise ValueError(f'Formal R120 requires all six new measured pools; pending {missing}. '
                         'Use --smoke-updates only for a clearly marked CPU spine.')
    merged_path = root / 'nonrouting_manifest.json'
    if merged_path.exists() and read_json(merged_path) != manifest:
        raise ValueError('Merged data sources changed; create a new version/root')
    write_json(merged_path, manifest)
    candidates = {p: list(routing_manifest['splits']['train'][p]['pool']) for p in ROUTING_PROBLEMS}
    candidates.update({p: manifest['nonrouting'][p]['candidate_solver_ids'] for p in tasks if p in NONROUTING_PROBLEMS})
    old = list(GLOBAL_SOLVERS)
    additions = sorted({f'NR::{problem}::{name}' for problem in tasks if problem in NONROUTING_PROBLEMS
                        for name in candidates[problem]})
    names = old + additions
    aliases = routing_manifest.get('scenario_solver_aliases', {})
    pools = {p: [names.index(aliases.get(name, name)) for name in pool] if p in ROUTING_PROBLEMS
             else [names.index(f'NR::{p}::{name}') for name in pool] for p, pool in candidates.items()}
    value = dict(experiment='R120',
        data_version='routing_new90k_nonrouting_v2_supplement_20261029' +
                     ('_cpu_smoke' if args.smoke_updates else ''),
        complete24=ready24, smoke=bool(args.smoke_updates), tasks=tasks,
        smoke_rows=args.smoke_rows if args.smoke_updates else None,
        routing_data_root=routing_ready['data_root'], routing_prepared_root=str(r84.resolve()),
        routing_manifest=routing_manifest,
        cache_root=str(Path(args.cache_root or routing_ready['cache_source']).resolve()),
        nonrouting_manifest_path=str(merged_path.resolve()),
        nonrouting_source_manifests=manifest['merged_sources'], candidate_names=candidates,
        model=dict(problem_names=tasks, solver_names=names, pools=pools,
            embedding_dim=128, head_num=4, encoder_layer_num=4, joint_layer_num=2, dropout=.1),
        geometry_stats=str((r84 / 'geometry_stats.json').resolve()),
        fresh_initialization=str((r84 / 'identicalinit_seed2.pt').resolve()), references={}, rows={})
    data = DataStore(value)
    try:
        for problem in tasks:
            task = data.open(problem, 'train')
            if len(task.candidate_solver_ids) < 2:
                raise ValueError(f'{problem} is not a method-selection pool with at least two candidates')
            objectives = task.label_arrays['native_objective']
            observed = (task.label_arrays['observed_mask'] if problem in NONROUTING_PROBLEMS
                        else np.ones(objectives.shape, dtype=bool))
            if args.smoke_updates:
                objectives, observed = objectives[:args.smoke_rows], observed[:args.smoke_rows]
            if not observed.all():
                raise ValueError(f'{problem} TRAIN measured pool is incomplete; do not create a query-dependent mask')
            zero_scale = None
            if problem == 'SMTWTP':
                signed_means = objectives.mean(0)
                if signed_means.min() <= 0:
                    total = 0.
                    scale_count = len(objectives)
                    for index in range(scale_count):
                        inputs = task.dataset[index]['inputs']
                        total += float(inputs['p'].sum(dtype=np.float64) * inputs['w'].sum(dtype=np.float64))
                    zero_scale = total / scale_count
            value['references'][problem] = fit_train_reference(objectives, observed,
                task.objective_direction, task.candidate_solver_ids, zero_scale=zero_scale)
            value['rows'][problem] = {'train': len(task)}
            for split in ('val', 'test'):
                if problem in ROUTING_PROBLEMS:
                    value['rows'][problem][split] = int(routing_manifest['splits'][split][problem]['rows'])
                else:
                    descriptor = manifest['nonrouting'][problem]['splits'][split]
                    if not descriptor.get('measured_pool_complete', False):
                        raise ValueError(f'{problem}/{split}: complete predeclared pool labels are required')
                    value['rows'][problem][split] = int(descriptor['count'])
            if args.smoke_updates:
                value['rows'][problem] = {split: min(count, args.smoke_rows)
                                         for split, count in value['rows'][problem].items()}
            data.close_split('train')
    finally:
        data.close()
    path = root / 'prepared.json'
    if path.exists() and read_json(path) != value:
        raise ValueError('Existing preparation differs; preserve it and choose a new root')
    write_json(path, value)
    print(json.dumps(dict(stage='prepared', complete24=ready24, tasks=tasks,
                          root=str(root), rows=value['rows'])), flush=True)
    return value


def make_protocol(args, prepared):
    model = copy.deepcopy(prepared['model'])
    module = importlib.import_module(args.model_module)
    if callable(getattr(module, 'default_config', None)):
        model = module.default_config(model['solver_names'], model['problem_names'], **{
            key: value for key, value in model.items() if key not in ('solver_names', 'problem_names')})
    return dict(experiment='R120', model_name='R120_typed_full_pool', seed=2,
        data_version=prepared['data_version'], complete24=prepared['complete24'],
        smoke=bool(args.smoke_updates), smoke_rows=args.smoke_rows if args.smoke_updates else None,
        model=model,
        prepared=prepared, max_updates=args.smoke_updates or args.max_updates,
        logical_batch_size=args.logical_batch_size, microbatch_size=args.microbatch_size,
        rank_weights=list(args.rank_weights) if args.rank_weights is not None else None,
        microbatch_caps=dict(args.microbatch_caps or {}), eval_batch_size=args.eval_batch_size,
        warmup_updates=min(args.warmup_updates, args.smoke_updates or args.max_updates),
        eval_every=min(args.eval_every, args.smoke_updates or args.max_updates),
        peak_lr=args.peak_lr, end_lr=args.end_lr, weight_decay=1e-4,
        early_stop_patience=args.early_stop_patience,
        objective=dict(select=1., advantage=.25, risk=.10, rdrop=0.,
            selection='routing native CE; new tasks best-set probability mass',
            scale='fixed positive native TRAIN-SBS mean; SMTWTP input scale only if zero'),
        selection_rule='max full-pool VAL task-macro any-best Top1, then min TRAIN-scaled excess, then earliest',
        inference_rule='one forward, original logits argmax, fixed global candidate order for exact score ties',
        winner_rules=dict(routing='saved native/FP64 ind; any-best separately',
                          nonrouting='exact native any-best; canonical first fixed solver ID displayed separately'),
        initialization='scratch; matching R84 random parameters and TRAIN-only geometry where applicable',
        comparison_status='new 24-task baseline; historical R94B is not a matched architectural control',
        model_factory='code.V4.r120_experiment.load_checkpoint',
        references=prepared['references'],
        model_module=args.model_module, objective_module=args.objective_module,
        wandb=True, wandb_mode=args.wandb_mode,
        source_hashes={name: file_hash(V4 / name) for name in
            ('r120_experiment.py', 'r120_reporting.py', 'nonrouting_data.py',
             'r120_model.py', 'r120_objective.py')})


def initialize_model(config, device, *, model_factory=None):
    module = importlib.import_module(config['model_module']) if model_factory is None else None
    seed_all(2)
    selector = (model_factory or module.MultidomainSelector)(config['model'])
    receipt = dict(seed=2, trained_checkpoint_loaded=False, initialization='scratch',
                   reused_r84_parameters=[], new_parameters=[])
    if module is not None and callable(getattr(module, 'initialize_from_r84', None)):
        receipt.update(module.initialize_from_r84(selector, config['prepared']['fresh_initialization'],
                                                config['prepared']['geometry_stats']))
    else:
        # A redesigned frontend may not retain old parameter names. Copy only
        # exact, unambiguous random-initialization matches, never trained weights.
        initial_path = Path(config['prepared']['fresh_initialization'])
        if initial_path.exists():
            initial = torch.load(initial_path, map_location='cpu', weights_only=False)
            if initial.get('initialization') != 'scratch' or initial.get('seed') != 2:
                raise ValueError('R84 source must be its frozen random initialization')
            reference = initial['trainable']
            with torch.no_grad():
                for name, parameter in selector.named_parameters():
                    source_name = name
                    if name.startswith('frontend.route.'):
                        source_name = 'instance_encoder.' + name[len('frontend.route.'):]
                    elif name.startswith('layers.') and '.global_layer.' in name:
                        layer, suffix = name.split('.global_layer.', 1)
                        source_name = 'instance_encoder.layers.' + layer.split('.')[1] + '.' + suffix
                    matches = [old for old in reference if source_name == old or source_name.endswith('.' + old)]
                    if len(matches) == 1 and parameter.shape == reference[matches[0]].shape:
                        parameter.copy_(reference[matches[0]])
                        receipt['reused_r84_parameters'].append(name)
                    else:
                        receipt['new_parameters'].append(name)
        geometry = read_json(config['prepared']['geometry_stats'])
        if hasattr(selector, 'frontend') and hasattr(selector.frontend, 'set_routing_geometry'):
            selector.frontend.set_routing_geometry(geometry)
        with torch.no_grad():
            for name, buffer in selector.named_buffers():
                field = 'mean' if name.endswith('geometry_residual.mean') else (
                    'std' if name.endswith('geometry_residual.std') else None)
                if field is not None:
                    buffer.copy_(torch.as_tensor(geometry[field], dtype=buffer.dtype))
    receipt['trainable_parameters'] = sum(p.numel() for p in selector.parameters() if p.requires_grad)
    return selector.to(device), receipt


def load_checkpoint(path, device='cpu'):
    state = torch.load(path, map_location='cpu', weights_only=False)
    config = state['args']
    if (config.get('experiment') != 'R120' or config.get('seed') != 2
            or state.get('state_hash') != json_hash(config)):
        raise ValueError('Not an R120 seed2 checkpoint')
    module = importlib.import_module(config['model_module'])
    selector = module.MultidomainSelector(config['model'])
    selector.load_state_dict(state['model'], strict=True)
    return selector.to(device), state


def make_objective(selector, config):
    module = importlib.import_module(config['objective_module'])
    scales = {p: ref['objective_scale'] for p, ref in config['references'].items()}
    return module.MultidomainObjective(selector, scales)


def update_batch(objective, optimizer, scaler, batch, runtime, *, microbatch_size, global_count,
                 max_retries=10):
    """One exact logical update, including DDP averaging and same-RNG AMP retries."""
    count = len(batch['labels']['native_objective'])
    if not 0 < count <= global_count or microbatch_size < 1:
        raise ValueError('Each rank requires actual, nonempty rows and a positive microbatch')
    state = rng_state(runtime.device)
    for retry in range(max_retries):
        restore_rng(state, runtime.device)
        optimizer.zero_grad(set_to_none=True)
        totals, bad = {}, torch.zeros((), dtype=torch.int32, device=runtime.device)
        for start in range(0, count, microbatch_size):
            stop = min(start + microbatch_size, count)
            synchronize = stop == count
            context = (objective.no_sync() if isinstance(objective, DDP) and not synchronize else nullcontext())
            with context:
                with torch.autocast(device_type=runtime.device.type,
                                    dtype=torch.float16, enabled=scaler.is_enabled()):
                    loss, parts = objective(slice_batch(batch, start, stop), 0.)
                factor = (stop - start) * runtime.world_size / global_count
                scaler.scale(loss * factor).backward()
            bad |= (~torch.isfinite(loss.detach())).to(torch.int32)
            for key, value in parts.items():
                if key not in totals:
                    totals[key] = torch.zeros((), dtype=torch.float64, device=runtime.device)
                totals[key].add_(torch.as_tensor(value, device=runtime.device).detach().double(),
                                 alpha=(stop - start) / global_count)
        scaler.unscale_(optimizer)
        norm = torch.nn.utils.clip_grad_norm_(objective.parameters(), 1.)
        bad |= (~torch.isfinite(norm)).to(torch.int32)
        if runtime.world_size > 1:
            dist.all_reduce(bad, op=dist.ReduceOp.MAX)
        if int(bad.item()):
            if scaler.is_enabled():
                scaler.update(new_scale=max(1., scaler.get_scale() / 2))
            continue
        scaler.step(optimizer)
        scaler.update()
        if runtime.world_size > 1:
            for value in totals.values():
                dist.all_reduce(value)
        values = {key: float(value) for key, value in totals.items()}
        values.update(grad_norm=float(norm), amp_retries=retry)
        return values
    raise RuntimeError('Nonfinite loss/gradient persists; no instance or update was skipped')


def _prediction_data(task, indices, scores):
    indices = np.asarray(indices, dtype=np.int64)
    values = np.array(task.label_arrays['native_objective'][indices], dtype=np.float64)
    observed = (np.array(task.label_arrays['observed_mask'][indices], dtype=bool)
                if task.problem in NONROUTING_PROBLEMS else np.ones(values.shape, dtype=bool))
    signed = values if task.objective_direction == 'min' else -values
    best = observed & (signed == np.where(observed, signed, np.inf).min(1, keepdims=True))
    data = dict(indices=indices, logits=np.asarray(scores, dtype=np.float32),
        native_objective=values, observed_mask=observed, all_best_mask=best,
        pool=np.asarray(task.candidate_solver_ids), objective_direction=np.asarray(task.objective_direction))
    if task.problem in ROUTING_PROBLEMS:
        data['native_winner_index'] = np.array(task.label_arrays['native_winner_index'][indices], dtype=np.int64)
        data['nodes'] = np.array(task.cache.lengths[indices], dtype=np.int64)
    return data


def evaluate(selector, data, config, runtime, *, split, tag, step, root, checkpoint_sha256=None,
             tasks=None):
    """Shard every task's inference across ranks, then reduce once per task."""
    directory = Path(root) / 'evaluation' / split / tag
    tasks = list(tasks or config['model']['problem_names'])
    previous = selector.training
    selector.eval()
    per_problem = {}
    identity = dict(experiment='R120', split=split, tag=tag, successful_updates=step,
                    checkpoint_sha256=checkpoint_sha256, state_hash=json_hash(config))
    if runtime.primary:
        directory.mkdir(parents=True, exist_ok=True)
        write_json(directory / 'identity.json', identity)
    started = time.monotonic()
    try:
        with torch.inference_mode():
            for problem in tasks:
                task = data.open(problem, split)
                population = min(len(task), config['smoke_rows']) if config.get('smoke') else len(task)
                indices = rank_indices(np.arange(population), runtime.rank, runtime.world_size)
                chunks = []
                cap = min(config['eval_batch_size'], config['microbatch_caps'].get(problem, config['microbatch_size']))
                for start in range(0, len(indices), cap):
                    batch = data.batch(problem, split, indices[start:start + cap], config['model']['pools'])
                    batch = move_to(batch, runtime.device, pin=True)
                    outputs = selector(model_inputs(batch))
                    chunks.append(outputs['logits'].float().cpu().numpy())
                scores = np.concatenate(chunks) if chunks else np.empty((0, len(task.candidate_solver_ids)), np.float32)
                shards = gather_objects(_prediction_data(task, indices, scores), runtime)
                if runtime.primary:
                    constant = ('pool', 'objective_direction')
                    combined = {key: shards[0][key] if key in constant else
                                np.concatenate([part[key] for part in shards]) for key in shards[0]}
                    if not np.array_equal(combined['indices'], np.arange(population)):
                        raise ValueError(f'{problem}/{split}: distributed evaluation lost/repeated instances')
                    metric, pred = native_task_metrics(combined['logits'], combined['native_objective'],
                        combined['observed_mask'], task.objective_direction,
                        all_best_mask=combined['all_best_mask'], native_winner=combined.get('native_winner_index'),
                        reference=config['references'][problem])
                    combined['pred'] = pred
                    (directory / 'by_problem').mkdir(exist_ok=True)
                    np.savez_compressed(directory / 'by_problem' / (problem + '.npz'), **combined)
                    per_problem[problem] = metric
                data.close_split(split)
    finally:
        selector.train(previous)
    result = None
    if runtime.primary:
        result = dict(identity=identity, per_problem=per_problem,
                      aggregates=aggregate_metrics(per_problem), seconds=time.monotonic() - started,
                      complete24=set(per_problem) == set(PROBLEMS) and not config.get('smoke', False),
                      smoke=bool(config.get('smoke', False)))
        write_json(directory / 'metrics.json', result)
        print('[R120 eval] ' + json.dumps(dict(split=split, tag=tag, update=step,
            aggregates=result['aggregates'])), flush=True)
    return broadcast_object(result, runtime)


def validation_key(result):
    group = result['aggregates'].get('ALL24')
    if group is None:
        raise ValueError('Validation has no task-macro selection summary')
    return group['task_macro_any_best_top1_pct'], -group['task_macro_train_scaled_excess']


def status(root, runtime, phase, step, **fields):
    if runtime.primary:
        write_json(Path(root) / 'status.json', dict(experiment='R120', phase=phase,
            successful_updates=step, updated_utc=utc(), pid=os.getpid(), host=socket.gethostname(),
            world_size=runtime.world_size, device=str(runtime.device), **fields))


def _tracker(root, config, args, runtime):
    if not runtime.primary:
        return None
    import wandb
    mode, reason = args.wandb_mode, None
    if mode == 'auto':
        # W&B's settings/netrc lookup avoids printing credentials or requiring a
        # key in an environment variable when the account is already configured.
        try:
            from wandb.sdk.lib import apikey
            available = bool(apikey.api_key())
        except Exception:
            available = bool(os.environ.get('WANDB_API_KEY'))
        mode = 'online' if available else 'offline'
        reason = None if available else 'No configured W&B credential; explicit offline fallback'
    tracker = wandb.init(project='selector', entity='yjkds-southern-university-of-science-technology',
        name='R120_multidomain_selector', dir=str(root), mode=mode, config=config,
        id=read_json(root / 'wandb_receipt.json').get('id') if (root / 'wandb_receipt.json').exists() else None,
        resume='allow')
    write_json(root / 'wandb_receipt.json', dict(id=tracker.id, mode=mode, reason=reason))
    return tracker


def train(args, runtime, *, model_factory=None, objective_factory=None):
    from .r77_experiment import ShuffleStream
    root = Path(args.root)
    prepared = read_json(root / 'prepared.json')
    config = make_protocol(args, prepared)
    if not config['smoke'] and (not prepared['complete24'] or set(config['model']['problem_names']) != set(PROBLEMS)):
        raise ValueError('Formal training requires all 24 complete measured pools')
    protocol_path = root / 'protocol.json'
    if protocol_path.exists() and read_json(protocol_path) != config:
        raise ValueError('Existing R120 protocol changed; preserve this version')
    if (root / 'training_complete.json').exists():
        if config['smoke']:
            return read_json(root / 'training_complete.json')
        return test(args, runtime)
    if (root / 'last.pt').exists() and not args.resume:
        raise ValueError('Existing training state requires --resume')
    if runtime.primary:
        write_json(protocol_path, config)
    runtime.barrier()
    selector, initialization = initialize_model(config, runtime.device, model_factory=model_factory)
    optimizer = torch.optim.AdamW(selector.parameters(), lr=config['peak_lr'], weight_decay=config['weight_decay'])
    scaler = torch.amp.GradScaler('cuda', enabled=runtime.device.type == 'cuda' and not args.no_amp)
    objective = (objective_factory or make_objective)(selector, config)
    objective.to(runtime.device)
    if runtime.world_size > 1:
        objective = DDP(objective, device_ids=[runtime.device.index] if runtime.device.type == 'cuda' else None,
                        find_unused_parameters=True, broadcast_buffers=False)
    start, history, best, bad_points, retries = 0, [], None, 0, 0
    cursors = dict.fromkeys(config['model']['problem_names'], 0)
    last = root / 'last.pt'
    if args.resume and last.exists():
        saved = torch.load(last, map_location='cpu', weights_only=False)
        if saved['args'] != config or len(saved['rng_by_rank']) != runtime.world_size:
            raise ValueError('Resume requires this protocol and the same distributed world size')
        selector.load_state_dict(saved['model'])
        optimizer.load_state_dict(saved['optimizer'])
        scaler.load_state_dict(saved['scaler'])
        start, history, best = saved['successful_updates'], saved['history'], saved['best']
        cursors, bad_points, retries = saved['stream_draws'], saved['bad_validation_points'], saved['amp_retries']
        restore_rng(saved['rng_by_rank'][runtime.rank], runtime.device)
    else:
        seed_all(2 + runtime.rank)
    data = DataStore(prepared, cache_root=args.cache_root)
    streams = {problem: ShuffleStream(prepared['rows'][problem]['train'],
        200003 + int.from_bytes(hashlib.sha256(problem.encode()).digest()[:4], 'little'), cursors[problem])
        for problem in config['model']['problem_names']}
    order = task_order(config['max_updates'], config['model']['problem_names'])
    tracker = None

    def snapshot(step):
        states = gather_objects(rng_state(runtime.device), runtime)
        if not runtime.primary:
            return None
        return dict(args=config, model={key: value.detach().cpu() for key, value in selector.state_dict().items()},
            optimizer=optimizer.state_dict(), scaler=scaler.state_dict(), successful_updates=step,
            stream_draws={p: stream.draws for p, stream in streams.items()}, rng_by_rank=states,
            history=history, best=best, bad_validation_points=bad_points, amp_retries=retries,
            initialization=initialization, state_hash=json_hash(config))

    try:
        if runtime.primary:
            write_json(root / 'initialization.json', initialization)
        tracking_rng = rng_state(runtime.device)
        tracker = _tracker(root, config, args, runtime)
        restore_rng(tracking_rng, runtime.device)
        runtime.barrier()
        step = start
        for first in range(start + 1, config['max_updates'] + 1, config['eval_every']):
            if start > config['warmup_updates'] and bad_points >= config['early_stop_patience']:
                break
            last_step = min(config['max_updates'], first + config['eval_every'] - 1)
            requests = []
            for step in range(first, last_step + 1):
                problem = order[step - 1]
                indices = streams[problem].take(config['logical_batch_size']).numpy()
                indices = rank_indices(indices, runtime.rank, runtime.world_size, config['rank_weights'])
                requests.append((step, problem, indices))
            total_parts, interval_count = {}, 0
            with BatchPrefetch(data, requests, config['model']['pools'], runtime.device) as batches:
                for step, problem, batch in batches:
                    lr = learning_rate(step, max_updates=config['max_updates'], warmup_updates=config['warmup_updates'],
                                       peak_lr=config['peak_lr'], end_lr=config['end_lr'])
                    for group in optimizer.param_groups:
                        group['lr'] = lr
                    selector.train()
                    parts = update_batch(objective, optimizer, scaler, batch, runtime,
                        microbatch_size=config['microbatch_caps'].get(problem, config['microbatch_size']),
                        global_count=config['logical_batch_size'])
                    retries += int(parts['amp_retries'])
                    for key, value in parts.items():
                        total_parts[key] = total_parts.get(key, 0.) + value
                    interval_count += 1
                    if runtime.primary and (step == 1 or step % args.status_every == 0):
                        status(root, runtime, 'training_smoke' if config['smoke'] else 'training', step,
                               max_updates=config['max_updates'], task=problem, learning_rate=lr, parts=parts,
                               sample_exposures=step * config['logical_batch_size'])
                        print('[R120 train] ' + json.dumps(dict(update=step, task=problem, lr=lr, **parts)), flush=True)
            status(root, runtime, 'validation', step)
            validation = evaluate(selector, data, config, runtime, split='val', tag=f'update{step:07d}',
                                  step=step, root=root)
            entry = dict(successful_updates=step, learning_rate=lr,
                train_parts={key: value / interval_count for key, value in total_parts.items()},
                validation=validation)
            history.append(entry)
            improved = best is None or validation_key(validation) > tuple(best['key'])
            if improved:
                best = dict(update=step, key=list(validation_key(validation)),
                            validation_path=str(root / 'evaluation/val' / f'update{step:07d}' / 'metrics.json'))
                bad_points = 0
            elif step > config['warmup_updates']:
                bad_points += 1
            state = snapshot(step)
            if runtime.primary:
                save_checkpoint(last, state)
                if improved:
                    save_checkpoint(root / 'best.pt', state)
                write_json(root / 'history.json', history)
                if tracker is not None:
                    tracker.log(dict(update=step, learning_rate=lr, **entry['train_parts'],
                        val_any_best_top1=validation_key(validation)[0],
                        val_scaled_excess=-validation_key(validation)[1]), step=step)
            runtime.barrier()
            total_parts, interval_count = {}, 0
            if step > config['warmup_updates'] and bad_points >= config['early_stop_patience']:
                break
        if runtime.primary:
            write_json(root / 'training_complete.json', dict(complete=True, smoke=config['smoke'],
                successful_updates=step, max_updates=config['max_updates'], best=best,
                stopped_early=step < config['max_updates'], finished_utc=utc()))
        runtime.barrier()
        if not config['smoke']:
            return test(args, runtime, data=data)
        status(root, runtime, 'smoke_complete', step, formal_training=False)
        return history[-1] if history else None
    finally:
        data.close()
        if tracker is not None:
            tracker.finish()


def test(args, runtime, *, data=None):
    root = Path(args.root)
    config = read_json(root / 'protocol.json')
    if config['smoke']:
        raise ValueError('A smoke checkpoint cannot be delivered as formal TEST')
    if not (root / 'training_complete.json').exists():
        raise ValueError('Finish training and lock its unique VAL choice before TEST')
    selector, state = load_checkpoint(root / 'best.pt', runtime.device)
    lock = dict(experiment='R120', checkpoint=str((root / 'best.pt').resolve()),
        checkpoint_sha256=file_hash(root / 'best.pt'), selected_update=state['successful_updates'],
        selected_without_test=True, per_problem_assembly=False, state_hash=json_hash(config),
        inference_rule=config['inference_rule'])
    lock_path = root / 'final_test_lock.json'
    if lock_path.exists() and read_json(lock_path) != lock:
        raise ValueError('The already-locked checkpoint/inference rule changed')
    if runtime.primary:
        write_json(lock_path, lock)
    runtime.barrier()
    own_data = data is None
    data = data or DataStore(config['prepared'], cache_root=args.cache_root)
    try:
        for split in ('val', 'test'):
            directory = root / 'evaluation' / split / 'selected_best'
            previous = read_json(directory / 'metrics.json') if (directory / 'metrics.json').exists() else None
            reuse = previous is not None and previous['identity']['checkpoint_sha256'] == lock['checkpoint_sha256']
            if broadcast_object(reuse, runtime):
                continue
            status(root, runtime, 'locked_' + split, state['successful_updates'])
            evaluate(selector, data, config, runtime, split=split, tag='selected_best',
                step=state['successful_updates'], root=root, checkpoint_sha256=lock['checkpoint_sha256'])
        if runtime.primary:
            write_json(root / 'test_complete.json', dict(complete=True,
                checkpoint_sha256=lock['checkpoint_sha256'], selected_update=lock['selected_update']))
        status(root, runtime, 'test_complete', state['successful_updates'])
    finally:
        if own_data:
            data.close()
    runtime.barrier()
    return lock


def lib_evaluate(args, runtime):
    """Original LIB inputs, current full-selector checkpoint, native labels unchanged."""
    from .nss_comparison import read_costs
    from .r41_frozen_evaluation import attach_geometry
    from .train import make_loader
    root = Path(args.root)
    lock, config = read_json(root / 'final_test_lock.json'), read_json(root / 'protocol.json')
    if file_hash(root / 'best.pt') != lock['checkpoint_sha256']:
        raise ValueError('LIB requires the unique unchanged checkpoint')
    if not runtime.primary:
        runtime.barrier()
        return
    selector, state = load_checkpoint(root / 'best.pt', runtime.device)
    selector.eval()
    for dataset, problem in (('TSPLIB', 'TSP'), ('CVRPLIB', 'CVRP')):
        directory = root / 'evaluation' / dataset / 'selected_best'
        if (directory / 'metrics.json').exists():
            if read_json(directory / 'metrics.json')['identity']['checkpoint_sha256'] != lock['checkpoint_sha256']:
                raise ValueError('Saved LIB predictions belong to another checkpoint')
            continue
        raw = read_costs(problem, dataset)
        expected = config['prepared']['candidate_names'][problem]
        if raw['pool'] != expected:
            raise ValueError('LIB candidate identities/order differ from this frozen scenario')
        loader = make_loader(problem, dataset, args.lib_batch_size, 0, coord_augment=0,
                             shuffle=False, cache_device=runtime.device)
        attach_geometry(loader, runtime.device)
        scores = []
        with torch.inference_mode():
            from .tensor_loader import TensorBatchLoader
            for old in TensorBatchLoader(loader.batch, args.lib_batch_size, False, False):
                safe = {key: old[key] for key in ROUTING_INPUT_KEYS if key in old}
                batch = dict(problem=problem, inputs=safe,
                    pool_ids=torch.tensor(config['model']['pools'][problem], device=runtime.device),
                    solver_mask=torch.ones((len(old['n']), len(expected)), dtype=torch.bool, device=runtime.device))
                scores.append(selector(batch)['logits'].float().cpu().numpy())
        values = raw['costs']
        best = values == values.min(1, keepdims=True)
        saved = dict(logits=np.concatenate(scores), native_objective=values,
            observed_mask=np.ones(values.shape, dtype=bool), all_best_mask=best,
            native_winner_index=raw['winner'], pool=np.asarray(raw['pool']),
            indices=np.arange(len(values)), objective_direction=np.asarray('min'))
        metric, pred = native_task_metrics(saved['logits'], values, saved['observed_mask'], 'min',
            all_best_mask=best, native_winner=raw['winner'], reference=config['references'][problem])
        saved['pred'] = pred
        (directory / 'by_problem').mkdir(parents=True, exist_ok=True)
        np.savez_compressed(directory / 'by_problem' / (problem + '.npz'), **saved)
        identity = dict(experiment='R120', split=dataset, tag='selected_best',
            successful_updates=state['successful_updates'], checkpoint_sha256=lock['checkpoint_sha256'])
        write_json(directory / 'metrics.json', dict(identity=identity, per_problem={problem: metric},
            aggregates={}, complete24=False))
        del loader
    runtime.barrier()


def _caps(value):
    problem, count = value.split('=', 1)
    if problem not in PROBLEMS or int(count) < 1:
        raise argparse.ArgumentTypeError('Use TASK=positive_microbatch_size')
    return problem, int(count)


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('action', choices=('prepare', 'train', 'test', 'lib', 'report', 'status'))
    result.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    result.add_argument('--nonrouting-root', type=Path, default=DEFAULT_NONROUTING)
    result.add_argument('--supplement-root', type=Path)
    result.add_argument('--routing-prepared-root', type=Path, default=DEFAULT_PREPARED)
    result.add_argument('--cache-root', type=Path)
    result.add_argument('--device', default='cuda')
    result.add_argument('--max-updates', type=int, default=MAX_UPDATES)
    result.add_argument('--logical-batch-size', type=int, default=1024)
    result.add_argument('--microbatch-size', type=int, default=64)
    result.add_argument('--rank-weights', type=lambda value: tuple(int(part) for part in value.split(',')),
                        help='TRAIN-only positive rank weights, e.g. 5,5,3,3; evaluation remains equally sharded')
    result.add_argument('--microbatch-cap', action='append', type=_caps, dest='microbatch_caps')
    result.add_argument('--eval-batch-size', type=int, default=64)
    result.add_argument('--warmup-updates', type=int, default=WARMUP_UPDATES)
    result.add_argument('--eval-every', type=int, default=EVAL_EVERY)
    result.add_argument('--peak-lr', type=float, default=PEAK_LR)
    result.add_argument('--end-lr', type=float, default=END_LR)
    result.add_argument('--early-stop-patience', type=int, default=6)
    result.add_argument('--seed', type=int, choices=(2,), default=2)
    result.add_argument('--smoke-updates', type=int, default=0)
    result.add_argument('--smoke-rows', type=int, default=8)
    result.add_argument('--smoke-tasks', help='Comma-separated ready tasks; only for an explicitly marked smoke run')
    result.add_argument('--resume', action='store_true')
    result.add_argument('--no-amp', action='store_true')
    result.add_argument('--status-every', type=int, default=24)
    result.add_argument('--threads', type=int, default=2)
    result.add_argument('--wandb-mode', choices=('auto', 'online', 'offline'), default='auto')
    result.add_argument('--model-module', default='code.V4.r120_model')
    result.add_argument('--objective-module', default='code.V4.r120_objective')
    result.add_argument('--nss-root', type=Path, default=V4 / 'runs/R84_new90k_attention/nss_reference')
    result.add_argument('--lib-batch-size', type=int, default=1)
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    positive = ('max_updates', 'logical_batch_size', 'microbatch_size', 'eval_batch_size',
                'eval_every', 'early_stop_patience', 'status_every', 'threads', 'lib_batch_size', 'smoke_rows')
    if any(getattr(args, key) < 1 for key in positive) or args.smoke_updates < 0:
        raise ValueError('Budgets, batches, and intervals must be positive')
    torch.set_num_threads(args.threads)
    args.microbatch_caps = dict(args.microbatch_caps or [])
    if args.action == 'prepare':
        return prepare(args)
    if args.action == 'report':
        from .r120_reporting import report
        return report(args.root, nss_root=args.nss_root)
    if args.action == 'status':
        result = read_json(args.root / 'status.json')
        print(json.dumps(result, indent=2))
        return result
    runtime = distributed_runtime(args.device)
    if runtime.world_size > args.logical_batch_size:
        raise ValueError('Every DDP rank must receive real rows')
    if args.rank_weights is not None:
        for rank in range(runtime.world_size):
            if not len(rank_indices(np.arange(args.logical_batch_size), rank,
                                    runtime.world_size, args.rank_weights)):
                raise ValueError('Every weighted DDP rank must receive real rows')
    try:
        return {'train': train, 'test': test, 'lib': lib_evaluate}[args.action](args, runtime)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == '__main__':
    main()
