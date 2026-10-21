"""R80's four fixed-budget arms; scheduling/allocation belong to the launcher.

Only the locked R77 fresh initialization, FULL TRAIN statistics and existing
source caches are reused. `train` finishes its unique VAL selection and TEST;
reporting is a separate, repeatable command owned by r80_reporting.
"""

import argparse
import csv
import fcntl
import hashlib
import json
import math
import os
import shutil
import socket
import sys
import time
import traceback
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from code.unified_selector.registry import GLOBAL_SOLVERS, PROBLEMS
from .multitask_probe import task_seed
from .pairwise_objective import inputs_only
from .r40_experiment import aggregate, families
from .r76_experiment import (
    DATA_ROOT, LOSS_RECIPE, PART_KEYS, PinnedPrefetch, RDropObjective, Runtime,
    capture_rank_rng, file_hash, fixed_sbs_metrics, open_cache, read_json,
    restore_rank_rng, save_checkpoint, update_batch, utc, write_json,
)
from .r77_data import DEFAULT_ROOT as DEFAULT_PREPARED_ROOT, FULL_ROWS, HELDOUT_ROWS, parameter_hash
from .r77_experiment import ShuffleStream, micro_caps as r77_micro_caps, task_order
from .solver_code_embeddings import bundle_provenance, load_bundle, validate_bundle
from .solver_code_encoder import make_r45_model
from .train import configure_torch, set_seed


V4 = Path(__file__).resolve().parent
DEFAULT_ROOT = V4 / 'runs/R80_training_and_code_ablation'
DEFAULT_EMBEDDINGS = V4 / 'runs/R45_solver_code_embeddings/solver_code_vectors_atlas.pt'
ARMS = ('A_ce_only', 'B_batch512', 'C_lr3e4', 'D_code_embedding')
GROUPS = {arm: arm for arm in ARMS}
BASE_BATCH = 1024
BASE_EVAL = 1242
BASE_MAX = 49680
BASE_WARMUP = 3726
WINNER_RULE = 'native int64 ind from locked source cache, verified against original FP64 argmin; never FP32 argmin'
SOURCE_BUFFERS = ('code_vectors', 'role_mask', 'semantic_rows', 'solver_identity', 'chunk_vectors', 'chunk_indices')


def engine_arm(args):
    """A named source-only run may reuse a fixed R80 arm without borrowing its identity."""
    return getattr(args, 'engine_arm', args.arm)


def experiment_name(args):
    return getattr(args, 'experiment', 'R80')


@dataclass(frozen=True)
class ArmSpec:
    arm: str
    batch_size: int = BASE_BATCH
    max_updates: int = BASE_MAX
    warmup_updates: int = BASE_WARMUP
    eval_every: int = BASE_EVAL
    peak_lr: float = 1e-4
    end_lr: float = 1e-6

    @property
    def splits(self):
        return BASE_BATCH // self.batch_size


def arm_spec(arm):
    if arm not in ARMS:
        raise ValueError(f'Unknown R80 arm: {arm}')
    if arm == 'B_batch512':
        return ArmSpec(arm, 512, 99360, 7452, 2484)
    return ArmSpec(arm, peak_lr=3e-4 if arm == 'C_lr3e4' else 1e-4)


def learning_rate(update, arm='A_ce_only'):
    spec = arm if isinstance(arm, ArmSpec) else arm_spec(arm)
    if not 1 <= update <= spec.max_updates:
        raise ValueError('LR is defined on the precommitted one-based successful-update axis')
    if update <= spec.warmup_updates:
        return spec.peak_lr * update / spec.warmup_updates
    progress = (update - spec.warmup_updates) / (spec.max_updates - spec.warmup_updates)
    return spec.end_lr + .5 * (spec.peak_lr - spec.end_lr) * (1. + math.cos(math.pi * progress))


def consistency_weight(update, arm):
    spec = arm if isinstance(arm, ArmSpec) else arm_spec(arm)
    return 0. if spec.arm == 'A_ce_only' else .35 * min(1., update / spec.warmup_updates)


def micro_caps(arm, gpu_type='4090', large_cap=None):
    spec = arm_spec(arm)
    return {p: min(cap, spec.batch_size) for p, cap in r77_micro_caps(gpu_type, large_cap).items()}


def interval_requests(order, streams, start, end, spec):
    """B splits each unchanged baseline source request, not an independently drawn stream."""
    if start % spec.splits or end % spec.splits or not 0 <= start < end <= spec.max_updates:
        raise ValueError('Resume/request boundaries must retain complete baseline source requests')
    requests = []
    for problem in order[start // spec.splits:end // spec.splits]:
        rows = streams[problem].take(BASE_BATCH)
        for offset in range(0, BASE_BATCH, spec.batch_size):
            requests.append((problem, rows[offset:offset + spec.batch_size], spec.batch_size))
    return requests


def json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def frozen_source_hash(model):
    buffers = {name: value for name, value in model.solver_encoder.named_buffers() if name in SOURCE_BUFFERS}
    return parameter_hash(buffers) if buffers else None


class CEOnlyObjective(nn.Module):
    """The original maximin score path, two dropout passes, and only .35 * mean CE."""

    def __init__(self, selector):
        super().__init__()
        self.selector = selector

    def forward(self, batch, unused_consistency_weight=0.):
        device = batch['ind'].device
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == 'cuda'):
            first = self.selector(inputs_only(batch))['logits']
            second = self.selector(inputs_only(batch))['logits']
        ce = .5 * (F.cross_entropy(first.float(), batch['ind']) + F.cross_entropy(second.float(), batch['ind']))
        loss = .35 * ce
        zero = ce.detach().new_zeros(())
        parts = dict.fromkeys(PART_KEYS[:-1], zero)
        parts.update(loss=loss.detach(), ce=ce.detach())
        return loss, torch.stack([parts[key] for key in PART_KEYS[:-1]])


def initialize_shared(initial, geometry, arm, device='cpu', bundle=None):
    """Copy every common same-name/same-shape parameter, including solver ID/LN."""
    arm_spec(arm)
    if initial.get('seed') != 2 or initial.get('initialization') != 'scratch':
        raise ValueError('Only the locked seed2 fresh R77 initialization is admissible')
    reference = initial['trainable']
    if parameter_hash(reference) != initial['trainable_sha256']:
        raise ValueError('Fresh initialization parameter hash mismatch')
    params = dict(initial['model_params'])
    if params.get('dropout') != .1 or params.get('solver_representation', 'handcrafted') != 'handcrafted':
        raise ValueError('The common initialization must be the original R77 dropout.1 handcrafted model')
    if arm == 'D_code_embedding':
        params['solver_representation'] = 'code'
    set_seed(2)
    model = make_r45_model(params, bundle=bundle)
    actual = dict(model.named_parameters())
    common = {name: tensor for name, tensor in actual.items()
              if name in reference and tensor.shape == reference[name].shape}
    if arm != 'D_code_embedding' and set(common) != set(actual):
        raise ValueError('A/B/C must retain every R77 named parameter and shape')
    if arm == 'D_code_embedding' and any(
            name not in common for name in actual if not name.startswith('solver_encoder.')):
        raise ValueError('D may replace only the solver feature branch')
    with torch.no_grad():
        for name, tensor in common.items():
            tensor.copy_(reference[name])
        branch = model.instance_encoder.geometry_residual
        branch.mean.copy_(torch.as_tensor(geometry['mean'], dtype=branch.mean.dtype))
        branch.std.copy_(torch.as_tensor(geometry['std'], dtype=branch.std.dtype))
    expected = parameter_hash({name: reference[name] for name in common})
    if parameter_hash(common) != expected:
        raise ValueError('Common fresh parameters were not copied exactly')
    receipt = dict(seed=2, initialization='scratch_shared_r77', historical_checkpoint_loaded=False,
        r77_fresh_trainable_sha256=initial['trainable_sha256'], common_parameter_names=sorted(common),
        common_parameter_sha256=expected, trainable_sha256=parameter_hash(actual),
        new_parameter_names=sorted(set(actual) - set(common)), frozen_source_sha256=frozen_source_hash(model),
        projected_tokens_cached=False, geometry_fitted_on='FULL TRAIN only')
    return model.to(device), params, receipt


def load_checkpoint(path, device='cpu', *, experiment='R80', arms=ARMS, bundle_validator=None):
    """Unified reporter/runner factory. No API calls or historical-weight fallbacks."""
    state = torch.load(path, map_location='cpu', weights_only=False)
    config = state['args']
    if (config.get('experiment') != experiment or config.get('arm') not in arms
            or config.get('initialization') != 'scratch_shared_r77'
            or config.get('solver_names') != GLOBAL_SOLVERS or state.get('state_hash') != json_hash(config)):
        raise ValueError(f'Checkpoint is not an unchanged {experiment} from-scratch state')
    if config.get('engine_arm', config['arm']) == 'D_code_embedding':
        tensors = {name: state['model']['solver_encoder.' + name] for name in
                   ('chunk_vectors', 'chunk_indices') if 'solver_encoder.' + name in state['model']}
        (bundle_validator or validate_bundle)(dict(config['embedding_provenance'], vectors=state['model']['solver_encoder.code_vectors'],
            role_mask=state['model']['solver_encoder.role_mask'], **tensors))
    model = make_r45_model(config['model_params'], state_dict=state['model'])
    if frozen_source_hash(model) != state['initialization']['frozen_source_sha256']:
        raise ValueError('Frozen external source buffers changed')
    return model.to(device), state


load_model = load_checkpoint
from_checkpoint = load_checkpoint


def prepared_config(args):
    root = args.prepared_root
    ready = read_json(root / 'prepared.json')
    if (ready.get('experiment') != 'R77' or ready.get('seed') != 2
            or Path(ready['data_root']).resolve() != args.data_root.resolve()):
        raise ValueError('Use the locked seed2 R77 preparation for the latest singleRELD release')
    needed = ('source_cache_manifest.json', 'identicalinit_seed2.pt', 'common_full_train_sbs.json',
              'B_full_seed2/geometry_stats.json')
    for name in needed:
        if file_hash(root / name) != ready['artifact_sha256'][name]:
            raise ValueError(f'Locked R77 preparation changed: {name}')
    manifest = read_json(root / 'source_cache_manifest.json')
    for split, size in (('train', FULL_ROWS), ('val', HELDOUT_ROWS), ('test', HELDOUT_ROWS)):
        if set(manifest['splits'][split]) != set(PROBLEMS) or any(
                manifest['splits'][split][p]['rows'] != size for p in PROBLEMS):
            raise ValueError(f'Keep every original {split} row in all18 problems')
    if args.cache_root is None:
        args.cache_root = Path(ready['cache_source']).resolve()
    return ready, manifest, read_json(root / needed[-1]), read_json(root / 'common_full_train_sbs.json')


def protocol(args, ready, manifest, geometry, baselines, params, bundle, order, caps):
    spec = arm_spec(engine_arm(args))
    return dict(experiment=experiment_name(args), schema_version=1, arm=args.arm, seed=2,
        initialization='scratch_shared_r77', model_params=params, solver_names=GLOBAL_SOLVERS,
        data_root=str(args.data_root.resolve()), cache_root=str(args.cache_root.resolve()),
        prepared_root=str(args.prepared_root.resolve()), preparation_sha256=file_hash(args.prepared_root / 'prepared.json'),
        common_trainable_sha256=ready['trainable_sha256'], train_rows_per_problem=FULL_ROWS,
        data_manifest=manifest['splits'], winner_rule=WINNER_RULE, baselines=baselines, own_train_sbs=baselines,
        geometry_stats=geometry, global_batch_size=spec.batch_size, logical_global_batch=spec.batch_size,
        baseline_source_request_rows=BASE_BATCH, baseline_requests=BASE_MAX,
        max_successful_updates=spec.max_updates, updates_per_problem=spec.max_updates // len(PROBLEMS),
        total_sample_exposures=BASE_BATCH * BASE_MAX, sample_exposures_per_problem=BASE_BATCH * BASE_MAX // len(PROBLEMS),
        eval_every_updates=spec.eval_every, warmup_updates=spec.warmup_updates, rdrop_ramp_updates=spec.warmup_updates,
        lr=dict(rule='successful sample-exposure warmup then cosine, independent of VAL', peak=spec.peak_lr,
            end=spec.end_lr, warmup_updates=spec.warmup_updates, max_updates=spec.max_updates),
        loss=dict(ce=.35, comparison=0., risk=0., kl=0., dropout_passes=2) if spec.arm == 'A_ce_only' else LOSS_RECIPE,
        rdrop_weight=0. if spec.arm == 'A_ce_only' else .35, dropout=.1, optimizer='AdamW',
        weight_decay=1e-4, grad_clip=1., fixed_budget=True, early_stop=False,
        checkpoint_rule='strict minimum full VAL macro_vs_sbs_pct with common FULL TRAIN-fitted SBS; first wins ties',
        inference_rule='eval mode, original R77 maximin scores, stable ascending-global-solver-ID tie break',
        task_order_sha256=hashlib.sha256('\n'.join(order).encode('ascii')).hexdigest(),
        source_plan='R77 seed2 task/source requests; B splits each1024 request into two consecutive512 updates',
        microbatch_caps=caps, gpu_type=args.gpu_type, threads=2, prefetch_buffers=2, world_size=1,
        gradients='actual micro rows / effective batch; one clip and step; AMP replay same batch+RNG without advancing success',
        train_eval='full TRAIN every5 VAL points, plus the selected best if absent',
        eval_batch_size=args.eval_batch_size, test='full TEST only after unique immutable VAL-best lock; saved predictions reusable',
        test_rnd_disclosure='Earlier TEST results informed multi-round R&D; not an untouched blind evaluation',
        embedding_provenance=bundle_provenance(bundle) if bundle is not None else None,
        embedding_lock=read_json(args.embeddings.with_suffix('.lock.json')) if bundle is not None else None,
        source_sha256={Path(__file__).name: file_hash(Path(__file__))},
        wandb=True, wandb_mode=args.wandb_mode,
        decisions=dict(checkpoint_resume='completed interval boundary, including pending evaluation',
            exposure_alignment='B LR/ramp evaluated at half-baseline exposure increments; at2k exactly baseline k',
            source_indices='source_indices are locked-cache row positions; release_source_indices retain input metadata when present'))


def wandb_run(args, config):
    import wandb
    root = args.root / args.arm
    experiment = experiment_name(args)
    settings = dict(project='selector', entity='yjkds-southern-university-of-science-technology',
        name=experiment + '_' + args.arm, id=experiment.lower() + '-' + hashlib.sha256((str(args.root) + args.arm).encode()).hexdigest()[:16],
        config=config, dir=str(root), settings=wandb.Settings(init_timeout=30))
    requested = args.wandb_mode
    mode = 'online' if requested == 'auto' else requested
    fallback_reason = None
    try:
        run = wandb.init(**settings, mode=mode, resume='allow' if mode == 'online' else None)
    except Exception as error:
        fallback_reason = type(error).__name__ + ': ' + str(error)
        print(f'[{experiment} W&B] online initialization failed; using offline: ' + type(error).__name__, flush=True)
        wandb.finish(exit_code=1)
        mode = 'offline'
        run = wandb.init(**settings, mode=mode)
    write_json(root / 'wandb_receipt.json', dict(requested=requested, mode=mode, fallback_reason=fallback_reason,
        project=settings['project'], entity=settings['entity'], run_id=run.id, url=run.url if mode == 'online' else None,
        updated_utc=utc()))
    return run


def status_update(args, phase, step=None, **values):
    root = args.root / args.arm
    path = root / 'status.json'
    previous = read_json(path) if path.exists() else {}
    spec = arm_spec(engine_arm(args))
    if step is None:
        step = previous.get('successful_updates', 0)
    previous.update(experiment=experiment_name(args), arm=args.arm, phase=phase, pid=os.getpid(), host=socket.gethostname(),
        gpu=os.environ.get('CUDA_VISIBLE_DEVICES', 'external allocation'), gpu_type=args.gpu_type,
        device=str(args.device), updated_utc=utc(), successful_updates=step, step=step,
        max_successful_updates=spec.max_updates, sample_exposures=step * spec.batch_size,
        exposures=step * spec.batch_size, baseline_requests_completed=step // spec.splits,
        latest_val=previous.get('latest_val'), eta_seconds=previous.get('eta_seconds'))
    previous.update(values)
    write_json(path, previous)


def save_npz(path, arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    with temporary.open('wb') as stream:
        np.savez(stream, **arrays)
    temporary.replace(path)


def summarize(per_problem):
    result = dict(per_problem=per_problem, macro=aggregate(per_problem), families=families(per_problem),
        native_total=sum(v['n'] for v in per_problem.values()),
        native_correct=sum(v['native_correct'] for v in per_problem.values()), winner_rule=WINNER_RULE,
        sbs_reference='common FULL70114 TRAIN-fitted SBS, evaluated on the named split',
        all_aggregation='equal weight per problem; macro raw mean_cost descriptive only')
    result['macro'].update(macro_sbs_cost=float(np.mean([v['sbs_cost'] for v in per_problem.values()])),
        macro_oracle_cost=float(np.mean([v['oracle_cost'] for v in per_problem.values()])))
    return result


@torch.no_grad()
def evaluate(selector, args, manifest, baselines, split, tag, step, state_hash, checkpoint_sha256=None):
    root = args.root / args.arm
    directory = root / 'evaluation' / split / tag
    directory.mkdir(parents=True, exist_ok=True)
    identity = dict(experiment=experiment_name(args), arm=args.arm, split=split, tag=tag, successful_updates=step,
        state_hash=state_hash, model_state_sha256=parameter_hash(selector.state_dict()),
        checkpoint_sha256=checkpoint_sha256, winner_rule=WINNER_RULE)
    receipt_path = directory / 'identity.json'
    if receipt_path.exists() and read_json(receipt_path) != identity:
        raise ValueError(f'Saved evaluation belongs to a different state: {directory}')
    write_json(receipt_path, identity)
    if (directory / 'metrics.json').exists() and (directory / 'rank000.npz').exists():
        return read_json(directory / 'metrics.json')
    status_update(args, 'evaluation', step, split=split, tag=tag, evaluation_problem=None)
    device = next(selector.parameters()).device
    was_training = selector.training
    selector.eval()
    started, arrays, per_problem = time.monotonic(), {}, {}
    try:
        for problem in PROBLEMS:
            status_update(args, 'evaluation', step, split=split, tag=tag, evaluation_problem=problem)
            shard = directory / 'by_problem' / (problem + '.npz')
            if shard.exists():
                with np.load(shard, allow_pickle=False) as saved:
                    data = {key: saved[key] for key in saved.files}
            else:
                cache = open_cache(args.cache_root, split, problem, manifest, args.data_root)
                try:
                    batch_size = min(args.eval_batch_size, micro_caps(engine_arm(args), args.gpu_type, args.large_micro_cap)[problem])
                    requests = ((problem, torch.arange(start, min(start + batch_size, cache.size)),
                        min(batch_size, cache.size - start)) for start in range(0, cache.size, batch_size))
                    scores = []
                    with PinnedPrefetch({problem: cache}, requests, device) as batches:
                        for _, _, _, batch in batches:
                            scores.append(selector(inputs_only(batch))['logits'].float().cpu().numpy())
                    source_indices = np.arange(cache.size, dtype=np.int64)
                    release_indices = source_indices.copy()
                    metadata_path = args.data_root / (problem + split) / 'input_metadata.npz'
                    if metadata_path.exists():
                        with np.load(metadata_path, allow_pickle=False) as metadata:
                            if 'source_index' in metadata:
                                release_indices = metadata['source_index'].copy()
                    data = dict(logits=np.concatenate(scores), indices=source_indices.copy(),
                        source_indices=source_indices, release_source_indices=release_indices,
                        costs=cache.raw['costs'].copy(), winner=cache.raw['winner'].copy(),
                        pool=np.asarray(cache.pool), pool_ids=np.asarray(cache.pool_ids, dtype=np.int64),
                        nodes=np.asarray(cache.lengths).copy())
                    save_npz(shard, data)
                finally:
                    cache.close()
            count = manifest['splits'][split][problem]['rows']
            if (data['costs'].dtype != np.float64 or data['winner'].dtype != np.int64
                    or len(data['logits']) != count or not np.array_equal(data['indices'], np.arange(count))
                    or not np.array_equal(data['winner'], data['costs'].argmin(axis=1))):
                raise ValueError(f'Incomplete/misaligned native prediction archive: {problem}/{split}')
            raw = {key: data[key] for key in ('costs', 'winner', 'pool', 'pool_ids')}
            raw['pool'], raw['pool_ids'] = raw['pool'].tolist(), raw['pool_ids'].tolist()
            per_problem[problem], pred = fixed_sbs_metrics(data['logits'], raw, baselines[problem])
            data['pred'] = pred
            arrays.update({problem + '__' + key: value for key, value in data.items()})
    finally:
        selector.train(was_training)
    save_npz(directory / 'rank000.npz', arrays)
    result = summarize(per_problem)
    result.update(seconds=time.monotonic() - started, tag=tag, successful_updates=step,
        sample_exposures=step * arm_spec(engine_arm(args)).batch_size, identity=identity,
        prediction_archive=str((directory / 'rank000.npz').resolve()),
        own_train_sbs=summarize(per_problem))
    write_json(directory / 'metrics.json', result)
    print(f'[{experiment_name(args)} eval] ' + json.dumps(dict(arm=args.arm, split=split, tag=tag, **result['macro'])), flush=True)
    return result


def write_metrics_csv(root, history):
    path = root / 'metrics.csv'
    temporary = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    rows = []
    for row in history:
        for split in ('val', 'train'):
            value = row.get(split)
            if value is not None:
                rows.append(dict(evaluation_point=row['evaluation_point'], successful_updates=row['successful_updates'],
                    sample_exposures=row['sample_exposures'], split=split, **value['macro']))
    if rows:
        with temporary.open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        temporary.replace(path)


def validate_resume(state, config, spec):
    if state['args'] != config or state['state_hash'] != json_hash(config):
        raise ValueError('Resume must retain the exact arm protocol/state hash')
    step = state['successful_updates']
    pending = state.get('pending_evaluation') is not None
    if (not 0 <= step <= spec.max_updates or step % spec.eval_every
            or step != (len(state['history']) + int(pending)) * spec.eval_every):
        raise ValueError('Resume requires a completed interval, with an explicit pending VAL if needed')
    counts = Counter(state['updates_by_problem'])
    expected = Counter(p for p in task_order()[:step // spec.splits] for _ in range(spec.splits))
    if counts != expected or any(state['stream_draws'][p] != counts[p] * spec.batch_size for p in PROBLEMS):
        raise ValueError('Resume task counts/source cursors disagree with successful exposures')


def train(args, device, *, checkpoint_loader=None, bundle_loader=None, protocol_builder=None):
    loader = checkpoint_loader or load_checkpoint
    root, spec = args.root / args.arm, arm_spec(engine_arm(args))
    if (root / 'training_complete.json').exists():
        print(f'[{experiment_name(args)}] Training complete; reusing fixed checkpoint and finishing/reusing TEST', flush=True)
        return test(args, device, checkpoint_loader=loader)
    ready, manifest, geometry, baselines = prepared_config(args)
    order, caps = task_order(), micro_caps(engine_arm(args), args.gpu_type, args.large_micro_cap)
    bundle = (bundle_loader or load_bundle)(args.embeddings) if spec.arm == 'D_code_embedding' else None
    params = dict(ready['model_params'])
    if bundle is not None:
        params['solver_representation'] = 'code'
    config = (protocol_builder or protocol)(args, ready, manifest, geometry, baselines, params, bundle, order, caps)
    state_hash = json_hash(config)
    if args.state_hash is not None and args.state_hash != state_hash:
        raise ValueError(f'Expected state hash {args.state_hash}, actual {state_hash}')
    protocol_path = root / 'protocol.json'
    if protocol_path.exists() and read_json(protocol_path) != config:
        raise ValueError(f'Existing {experiment_name(args)} protocol differs; preserve it and use a different run root')
    last = root / 'last.pt'
    if last.exists() and not args.resume:
        raise ValueError('Use --resume to continue this arm; no existing state will be overwritten')
    status_update(args, 'initializing', 0, state_hash=state_hash, source_sha256=config['source_sha256'])
    write_json(protocol_path, config)
    checkpoint = None
    if last.exists():
        selector, checkpoint = loader(last, device)
        validate_resume(checkpoint, config, spec)
        initialization = checkpoint['initialization']
    else:
        initial = torch.load(args.prepared_root / 'identicalinit_seed2.pt', map_location='cpu', weights_only=False)
        selector, actual_params, initialization = initialize_shared(initial, geometry, engine_arm(args), device, bundle)
        if actual_params != params or initial['trainable_sha256'] != ready['trainable_sha256']:
            raise ValueError('Fresh parameter source differs from the locked preparation')
        write_json(root / 'initialization.json', initialization)
        del initial
    del bundle
    optimizer = torch.optim.AdamW(selector.parameters(), lr=spec.peak_lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler('cuda', enabled=device.type == 'cuda')
    objective = CEOnlyObjective(selector) if spec.arm == 'A_ce_only' else RDropObjective(selector)
    runtime = Runtime(0, 1, device)
    history, best, successful, retries_total, pending = [], None, 0, 0, None
    counts, cursors = Counter(), dict.fromkeys(PROBLEMS, 0)
    if checkpoint is not None:
        optimizer.load_state_dict(checkpoint['optimizer'])
        for optimizer_state in optimizer.state.values():
            for key, value in optimizer_state.items():
                if torch.is_tensor(value):
                    optimizer_state[key] = value.cpu() if key == 'step' else value.to(device)
        scaler.load_state_dict(checkpoint['scaler'])
        history, best, successful = checkpoint['history'], checkpoint['best'], checkpoint['successful_updates']
        retries_total, counts = checkpoint['amp_retries'], Counter(checkpoint['updates_by_problem'])
        cursors, pending = checkpoint['stream_draws'], checkpoint.get('pending_evaluation')
        restore_rank_rng(checkpoint['rng'], device)
    else:
        set_seed(2)
    del checkpoint
    caches, tracker = {}, None
    streams = {p: ShuffleStream(FULL_ROWS, task_seed(2, p), cursors[p]) for p in PROBLEMS}

    def checkpoint_state():
        if frozen_source_hash(selector) != initialization['frozen_source_sha256']:
            raise ValueError('The frozen source code buffers must never be trained or replaced')
        return dict(model=selector.state_dict(), optimizer=optimizer.state_dict(), scaler=scaler.state_dict(),
            args=config, state_hash=state_hash, initialization=initialization, epoch=len(history),
            successful_updates=successful, updates_by_problem=dict(counts), sample_exposures=successful * spec.batch_size,
            stream_draws={p: stream.draws for p, stream in streams.items()}, amp_retries=retries_total,
            rng=capture_rank_rng(device), history=history, best=best, pending_evaluation=pending)

    if not last.exists():
        save_checkpoint(last, checkpoint_state())
    try:
        tracker = wandb_run(args, config)
        # W&B/setup must not consume the training RNG on either fresh start or resume.
        saved_rng = torch.load(last, map_location='cpu', weights_only=False)['rng']
        restore_rank_rng(saved_rng, device)
        del saved_rng
        while successful < spec.max_updates or pending is not None:
            if pending is None:
                if not caches:
                    for p in PROBLEMS:
                        caches[p] = open_cache(args.cache_root, 'train', p, manifest, args.data_root)
                        if caches[p].size != FULL_ROWS:
                            raise ValueError('Every arm trains on the full70114 rows per task')
                end = min(spec.max_updates, successful + spec.eval_every)
                requests = interval_requests(order, streams, successful, end, spec)
                interval_start, first = time.monotonic(), successful
                weighted = torch.zeros(len(PART_KEYS), dtype=torch.float64, device=device)
                selector.train()
                with PinnedPrefetch(caches, requests, device) as batches:
                    for problem, n_rank, n_global, batch in batches:
                        if n_rank != spec.batch_size or n_global != spec.batch_size:
                            raise ValueError('An optimizer update must contain exactly the effective batch real rows')
                        lr, weight = learning_rate(successful + 1, spec), consistency_weight(successful + 1, spec)
                        for group in optimizer.param_groups:
                            group['lr'] = lr
                        parts, retries = update_batch(objective, optimizer, scaler, batch, weight,
                            n_rank, n_global, runtime, microbatch_cap=caps[problem])
                        successful += 1
                        counts[problem] += 1
                        retries_total += retries
                        weighted.add_(parts.double())
                        if successful == 1 or successful % args.log_every == 0 or successful == end:
                            done, elapsed = successful - first, time.monotonic() - interval_start
                            info = dict(successful_updates=successful, max_updates=spec.max_updates,
                                problem=problem, lr=lr, consistency_weight=weight, sample_exposures=successful * spec.batch_size,
                                seconds_per_update=elapsed / done, samples_per_second=done * spec.batch_size / elapsed,
                                amp_retries=retries_total, updates_by_problem=dict(counts),
                                **dict(zip(PART_KEYS, (weighted / done).cpu().tolist())))
                            val_seconds = np.mean([row['val']['seconds'] for row in history]) if history else 0.
                            eta = (spec.max_updates - successful) * elapsed / done
                            eta += (spec.max_updates // spec.eval_every - len(history)) * float(val_seconds)
                            status_update(args, 'training', successful, eta_seconds=eta, **{
                                key: value for key, value in info.items() if key != 'successful_updates'})
                            with (root / 'updates.jsonl').open('a') as stream:
                                stream.write(json.dumps(dict(updated_utc=utc(), **info), allow_nan=False) + '\n')
                            print(f'[{experiment_name(args)} train] ' + json.dumps(dict(arm=args.arm, **info)), flush=True)
                            if tracker is not None:
                                tracker.log({**{'optimization/' + key: info[key] for key in PART_KEYS},
                                    'successful_updates': successful, 'sample_exposures': info['sample_exposures'],
                                    'lr': lr, 'samples_per_second': info['samples_per_second']}, step=successful)
                pending = dict(zip(PART_KEYS, (weighted / (successful - first)).cpu().tolist()))
                pending.update(lr=lr, consistency_weight=weight, amp_retries=retries_total,
                    samples=(successful - first) * spec.batch_size, seconds=time.monotonic() - interval_start)
                save_checkpoint(last, checkpoint_state())
            point = successful // spec.eval_every
            tag = f'update{successful:07d}'
            validation = evaluate(selector, args, manifest, baselines, 'val', tag, successful, state_hash)
            cost = validation['macro']['macro_vs_sbs_pct']
            if not math.isfinite(cost):
                raise ValueError('Primary full VAL macro_vs_sbs_pct must be finite')
            training = evaluate(selector, args, manifest, baselines, 'train', tag, successful, state_hash) if point % 5 == 0 else None
            row = dict(epoch=point, evaluation_point=point, successful_updates=successful,
                sample_exposures=successful * spec.batch_size, updates_by_problem=dict(counts),
                evaluation_tag=tag, optimization=pending, val=validation, train=training)
            history.append(row)
            improved = best is None or cost < best['val']['macro']['macro_vs_sbs_pct']
            if improved:
                best = dict(row)
            pending = None
            state = checkpoint_state()
            if improved:
                save_checkpoint(root / 'best.pt', state)
                write_json(root / 'best_eval.json', best)
            save_checkpoint(last, state)
            del state
            write_json(root / 'history.json', history)
            write_metrics_csv(root, history)
            status_update(args, 'evaluation_complete', successful, latest_val=validation['macro'],
                latest_val_update=successful, selected_update=best['successful_updates'])
            if tracker is not None:
                values = dict(successful_updates=successful, sample_exposures=successful * spec.batch_size)
                for split, value in (('val', validation), ('train_eval', training)):
                    if value is not None:
                        values.update({split + '/' + key: score for key, score in value['macro'].items()})
                tracker.log(values, step=successful)
        if successful != spec.max_updates or any(counts[p] != spec.max_updates // len(PROBLEMS) for p in PROBLEMS):
            raise ValueError('The fixed total/per-task successful-update budget was not completed')
        for cache in caches.values():
            cache.close()
        caches.clear()
        del objective, optimizer, scaler, selector
        selector, selected = loader(root / 'best.pt', device)
        if selected['args'] != config or selected['successful_updates'] != best['successful_updates']:
            raise ValueError('The unique VAL-best checkpoint disagrees with training history')
        del selected
        lock = lock_best(root, best, config, state_hash)
        selected_val = root / 'evaluation/val/selected_best'
        selected_val.mkdir(parents=True, exist_ok=True)
        source_val = root / 'evaluation/val' / best['evaluation_tag']
        for name in ('rank000.npz', 'metrics.json', 'identity.json'):
            target = selected_val / name
            if not target.exists():
                shutil.copyfile(source_val / name, target)
        if best['train'] is None:
            best = dict(best, train=evaluate(selector, args, manifest, baselines, 'train', 'selected_best',
                best['successful_updates'], state_hash, lock['checkpoint_sha256']))
        for row in history:
            if row['successful_updates'] == best['successful_updates']:
                row['train'] = best['train']
        write_json(root / 'history.json', history)
        write_json(root / 'best_eval.json', best)
        write_metrics_csv(root, history)
        write_json(root / 'result.json', dict(best=best, final=history[-1], successful_updates=successful,
            updates_by_problem=dict(counts), amp_retries=retries_total, state_hash=state_hash,
            stop_reason='fixed successful-update budget, no early stopping', test_read=False))
        write_json(root / 'final_metrics.json', dict(epoch=best['epoch'], selected_update=best['successful_updates'],
            checkpoint_sha256=lock['checkpoint_sha256'], state_hash=state_hash, train=best['train'], val=best['val'], test_read=False))
        write_json(root / 'training_complete.json', dict(completed_utc=utc(), successful_updates=successful,
            sample_exposures=successful * spec.batch_size, updates_by_problem=dict(counts),
            selected_best=lock, state_hash=state_hash, next_stage='locked full TEST', test_read=False))
        status_update(args, 'training_complete', successful, selected_update=best['successful_updates'])
        del selector
    finally:
        for cache in caches.values():
            cache.close()
        if tracker is not None:
            tracker.finish()
    return test(args, device, checkpoint_loader=loader)


def lock_best(root, best, config, state_hash):
    record = dict(experiment=config['experiment'], arm=config['arm'], checkpoint=str((root / 'best.pt').resolve()),
        checkpoint_sha256=file_hash(root / 'best.pt'), state_hash=state_hash,
        epoch=best['epoch'], selected_update=best['successful_updates'], rule=config['checkpoint_rule'],
        selected_without_test=True, per_problem_assembly=False, val=best['val']['macro'],
        inference_rule=config['inference_rule'], winner_rule=WINNER_RULE)
    path = root / 'final_test_lock.json'
    if path.exists():
        previous = read_json(path)
        if any(previous.get(key) != value for key, value in record.items()):
            raise PermissionError('The unique checkpoint/inference lock cannot change')
        return previous
    record['locked_utc'] = utc()
    write_json(path, record)
    return record


def authorize_test(args):
    root, spec = args.root / args.arm, arm_spec(engine_arm(args))
    complete_path = root / 'training_complete.json'
    if not complete_path.exists() or read_json(complete_path).get('successful_updates') != spec.max_updates:
        raise PermissionError('Finish the fixed training budget before opening any TEST input')
    lock, config = read_json(root / 'final_test_lock.json'), read_json(root / 'protocol.json')
    if (lock.get('experiment') != experiment_name(args) or config.get('experiment') != experiment_name(args)
            or lock.get('arm') != args.arm
            or not lock.get('selected_without_test') or lock.get('per_problem_assembly') is not False
            or lock.get('state_hash') != json_hash(config) or config.get('arm') != args.arm
            or config['max_successful_updates'] != spec.max_updates
            or Path(lock['checkpoint']).resolve() != (root / 'best.pt').resolve()
            or lock['checkpoint_sha256'] != file_hash(root / 'best.pt')
            or Path(config['data_root']).resolve() != args.data_root.resolve()):
        raise PermissionError(f'TEST requires the unchanged unique {experiment_name(args)} best/configuration lock')
    if args.state_hash is not None and args.state_hash != lock['state_hash']:
        raise ValueError('Expected state hash differs from the immutable TEST lock')
    return lock, config


def test(args, device, *, checkpoint_loader=None):
    root, spec = args.root / args.arm, arm_spec(engine_arm(args))
    lock, config = authorize_test(args)
    result_path = root / 'test_results.json'
    if result_path.exists():
        saved = read_json(result_path)
        if (not saved.get('complete') or saved.get('checkpoint_sha256') != lock['checkpoint_sha256']
                or saved.get('state_hash') != lock['state_hash']):
            raise PermissionError('Saved TEST result differs from the immutable lock')
        result = saved['result']
        print(f'[{experiment_name(args)}] Reusing saved full TEST; no repeated training/inference', flush=True)
    else:
        args.cache_root = Path(config['cache_root']) if args.cache_root is None else args.cache_root
        args.gpu_type, args.eval_batch_size = config['gpu_type'], config['eval_batch_size']
        args.large_micro_cap = config['microbatch_caps']['TSP']
        selector, state = (checkpoint_loader or load_checkpoint)(root / 'best.pt', device)
        if state['args'] != config or state['successful_updates'] != lock['selected_update']:
            raise PermissionError('TEST model does not match its frozen selection')
        del state
        started = root / 'test_started.json'
        if started.exists() and read_json(started).get('checkpoint_sha256') != lock['checkpoint_sha256']:
            raise PermissionError('Earlier TEST attempt used a different locked checkpoint')
        write_json(started, dict(started_utc=utc(), checkpoint_sha256=lock['checkpoint_sha256'],
            inference_rule=config['inference_rule'], resumable_same_checkpoint=True))
        manifest = read_json(Path(config['prepared_root']) / 'source_cache_manifest.json')
        result = evaluate(selector, args, manifest, config['baselines'], 'test', 'selected_best',
            lock['selected_update'], lock['state_hash'], lock['checkpoint_sha256'])
        if file_hash(root / 'best.pt') != lock['checkpoint_sha256']:
            raise PermissionError('The locked checkpoint changed during TEST')
        write_json(result_path, dict(complete=True, completed_utc=utc(), result=result, epoch=lock['epoch'],
            selected_update=lock['selected_update'], checkpoint_sha256=lock['checkpoint_sha256'], state_hash=lock['state_hash']))
        del selector
    final = read_json(root / 'final_metrics.json')
    final.update(test=result, test_read=True, complete=True, checkpoint_sha256=lock['checkpoint_sha256'])
    write_json(root / 'final_metrics.json', final)
    write_json(root / 'test_complete.json', dict(full_tests=1, completed_utc=utc(),
        checkpoint_sha256=lock['checkpoint_sha256'], saved_predictions_reusable=True))
    status_update(args, 'test_complete', spec.max_updates, selected_update=lock['selected_update'],
        test=result['macro'], eta_seconds=0., state_hash=lock['state_hash'])
    print(f'[{experiment_name(args)} ready for reporting] {root}', flush=True)
    return result


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('command', choices=('train', 'test', 'describe', 'status'))
    result.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    result.add_argument('--arm', choices=ARMS, required=True)
    result.add_argument('--prepared-root', '--r77-root', type=Path, default=DEFAULT_PREPARED_ROOT)
    result.add_argument('--data-root', type=Path, default=DATA_ROOT)
    result.add_argument('--cache-root', type=Path)
    result.add_argument('--embeddings', type=Path, default=DEFAULT_EMBEDDINGS)
    result.add_argument('--device', default='cuda:0')
    result.add_argument('--gpu-type', choices=('4090', '3090'), default='4090')
    result.add_argument('--large-micro-cap', type=int)
    result.add_argument('--eval-batch-size', type=int, default=128)
    result.add_argument('--log-every', type=int, default=18)
    result.add_argument('--wandb-mode', choices=('auto', 'online', 'offline'), default='auto')
    result.add_argument('--resume', action='store_true')
    result.add_argument('--state-hash', '--expected-state-hash')
    return result


def run(args, *, train_fn=None, test_fn=None):
    """Shared CLI dispatch; identity/source hooks do not change the training engine."""
    for name in ('root', 'prepared_root', 'data_root', 'cache_root', 'embeddings'):
        if getattr(args, name) is not None:
            setattr(args, name, getattr(args, name).expanduser().resolve())
    if args.command == 'describe':
        description = dict(arm_spec(engine_arm(args)).__dict__, root=str(args.root / args.arm),
            microbatch_caps=micro_caps(engine_arm(args), args.gpu_type, args.large_micro_cap),
            total_sample_exposures=BASE_BATCH * BASE_MAX)
        if experiment_name(args) != 'R80':
            description.update(experiment=experiment_name(args), arm=args.arm, engine_arm=engine_arm(args),
                embeddings=str(args.embeddings), seed=2, checkpoint_rule='strict minimum full VAL macro_vs_sbs_pct; first wins ties')
        print(json.dumps(description, indent=2))
        return
    if args.command == 'status':
        print(json.dumps(read_json(args.root / args.arm / 'status.json'), indent=2))
        return
    if args.eval_batch_size < 1 or args.log_every < 1:
        raise ValueError('Evaluation/logging intervals must be positive')
    root = args.root / args.arm
    root.mkdir(parents=True, exist_ok=True)
    with (root / 'runner.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit(f'{experiment_name(args)} {args.arm} is already running; do not start a duplicate') from None
        try:
            torch.set_num_threads(2)
            torch.set_num_interop_threads(2)
            configure_torch()
            device = torch.device(args.device)
            if device.type != 'cuda':
                raise ValueError('Formal training/inference requires the externally allocated CUDA device; tests are CPU-only')
            if args.command == 'train':
                (train_fn or train)(args, device)
            else:
                (test_fn or test)(args, device)
        except BaseException as error:
            message = dict(type=type(error).__name__, message=str(error), updated_utc=utc(),
                traceback=traceback.format_exc(), pid=os.getpid(), host=socket.gethostname())
            write_json(root / 'error.json', message)
            status_update(args, 'failed', error=message)
            print(f'[{experiment_name(args)} error] {message["type"]}: {message["message"]}', file=sys.stderr, flush=True)
            raise


def main(argv=None):
    return run(parser().parse_args(argv))


if __name__ == '__main__':
    main()
