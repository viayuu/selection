"""R84 new90k full18, four fresh arms, update-matched VAL selection and TEST.

This independent runner reuses the R76 cache/prefetch/AMP step and R77 natural
shuffle streams. It does not replace globals in any historical experiment.
"""

import argparse
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
from pathlib import Path

import numpy as np
import torch

from code.unified_selector.registry import GLOBAL_SOLVERS, PROBLEMS
from .multitask_probe import task_seed
from .pairwise_objective import inputs_only
from .r40_experiment import aggregate, families
from .r76_experiment import (
    LOSS_RECIPE, PART_KEYS, PinnedPrefetch, RDropObjective, Runtime,
    capture_rank_rng, file_hash, fixed_sbs_metrics, open_cache, read_json,
    restore_rank_rng, save_checkpoint, slice_batch, utc, write_json,
)
from .r77_data import parameter_hash
from .r77_experiment import ShuffleStream, micro_caps as r77_micro_caps, task_order
from .r80_experiment import WINNER_RULE, json_hash, lock_best, save_npz, wandb_run, write_metrics_csv
from .r84_prepare import DEFAULT_DATA_ROOT, DEFAULT_PREPARED_ROOT, DEFAULT_ROOT, SPLIT_ROWS
from .solver_code_encoder import make_r45_model
from .train import configure_torch, set_seed


V4 = Path(__file__).resolve().parent
ARMS = ('A_baseline', 'B_attention_direct', 'C_focal', 'D_attention_focal')
ATTENTION_ARMS = ('B_attention_direct', 'D_attention_focal')
FOCAL_ARMS = ('C_focal', 'D_attention_focal')
BATCH_SIZE, EVAL_EVERY, WARMUP_UPDATES, MAX_UPDATES = 1024, 1440, 4320, 57600
PEAK_LR, END_LR = 1e-4, 1e-6
MODEL_FACTORY = 'code.V4.r84_experiment.load_checkpoint'


def learning_rate(update):
    if not 1 <= update <= MAX_UPDATES:
        raise ValueError('Learning rate requires a one-based successful update')
    if update <= WARMUP_UPDATES:
        return PEAK_LR * update / WARMUP_UPDATES
    progress = (update - WARMUP_UPDATES) / (MAX_UPDATES - WARMUP_UPDATES)
    return END_LR + .5 * (PEAK_LR - END_LR) * (1 + math.cos(math.pi * progress))


def consistency_weight(update):
    return .35 * min(1., update / WARMUP_UPDATES)


def micro_caps(gpu_type='4090', large_cap=None):
    return r77_micro_caps(gpu_type, large_cap)


def make_model(params, arm, state_dict=None):
    if arm not in ARMS:
        raise ValueError('Unknown R84 arm')
    if arm in ATTENTION_ARMS:
        from .r84_attention_selector import make_r84_model
        return make_r84_model(params, state_dict=state_dict)
    return make_r45_model(params, state_dict=state_dict)


def make_objective(selector, arm):
    if arm in FOCAL_ARMS:
        from .r84_objective import FocalRDropObjective
        return FocalRDropObjective(selector)
    return RDropObjective(selector)


def update_batch(objective, optimizer, scaler, batch, weight, runtime, cap, max_retries=20):
    """Original row-weighted AMP update, with detached successful focal metrics."""
    count = len(batch['ind'])
    if runtime.world_size != 1 or count <= 0 or cap <= 0:
        raise ValueError('R84 uses one externally bound GPU and real nonempty rows')
    rng = capture_rank_rng(runtime.device)
    has_focal = hasattr(objective, 'reset_focal_metrics')
    for retry in range(max_retries):
        restore_rank_rng(rng, runtime.device)
        optimizer.zero_grad(set_to_none=True)
        if has_focal:
            objective.reset_focal_metrics()
        parts_sum = torch.zeros(len(PART_KEYS) - 1, dtype=torch.float64, device=runtime.device)
        bad = torch.zeros((), dtype=torch.int32, device=runtime.device)
        for start in range(0, count, cap):
            stop = min(start + cap, count)
            micro = slice_batch(batch, start, stop)
            loss, parts = objective(micro, weight)
            scaler.scale(loss * ((stop - start) / count)).backward()
            bad |= (~torch.isfinite(loss.detach())).to(torch.int32)
            parts_sum.add_(parts.detach().double(), alpha=stop - start)
            if has_focal:
                objective.accumulate_focal_metrics(stop - start)
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(objective.parameters(), 1.)
        bad |= (~torch.isfinite(grad_norm)).to(torch.int32)
        if bad.item():
            if scaler.is_enabled():
                scaler.update(new_scale=scaler.get_scale() * .5)
            continue
        scaler.step(optimizer)
        scaler.update()
        metrics = objective.focal_metrics() if has_focal else dict(
            focal=parts_sum.new_zeros(()), focal_extra=parts_sum.new_zeros(()))
        return torch.cat((parts_sum / count, grad_norm.detach().reshape(1))), retry, metrics
    raise RuntimeError('Nonfinite gradients persisted; no rows skipped or optimizer update accepted')


def initialize_shared(initial, geometry, arm, device='cpu', *, model_factory=None):
    reference = initial['trainable']
    if (initial.get('seed') != 2 or initial.get('initialization') != 'scratch'
            or parameter_hash(reference) != initial.get('trainable_sha256')):
        raise ValueError('Only the locked fresh seed2 initialization is permitted')
    set_seed(2)
    model = (model_factory or make_model)(initial['model_params'], arm)
    actual = dict(model.named_parameters())
    common = {name: tensor for name, tensor in actual.items()
              if name in reference and reference[name].shape == tensor.shape}
    if set(common) != set(actual):
        raise ValueError('Every retained R84 parameter must have the common fresh initialization')
    with torch.no_grad():
        for name, tensor in common.items():
            tensor.copy_(reference[name])
        branch = model.instance_encoder.geometry_residual
        branch.mean.copy_(torch.as_tensor(geometry['mean'], dtype=branch.mean.dtype))
        branch.std.copy_(torch.as_tensor(geometry['std'], dtype=branch.std.dtype))
    expected = parameter_hash({name: reference[name] for name in common})
    if parameter_hash(common) != expected:
        raise ValueError('Shared initialization differs before the first update')
    params = dict(model.params)
    receipt = dict(seed=2, initialization='scratch_shared_r77', historical_checkpoint_loaded=False,
        r77_fresh_trainable_sha256=initial['trainable_sha256'], common_parameter_names=sorted(common),
        common_parameter_sha256=expected, trainable_sha256=parameter_hash(actual),
        removed_parameter_names=sorted(set(reference) - set(actual)), new_parameter_names=[],
        geometry_fitted_on='new81000 TRAIN only', frozen_source_sha256=None)
    return model.to(device), params, receipt


def prepared_config(args):
    ready = read_json(args.prepared_root / 'prepared.json')
    if (ready.get('experiment') != 'R84' or ready.get('seed') != 2
            or Path(ready['data_root']).resolve() != args.data_root.resolve()
            or ready['rows_per_problem'] != SPLIT_ROWS
            or ready['scenario_sha256'] != file_hash(args.data_root / 'scenario_manifest.json')):
        raise ValueError('Use the locked complete new90k preparation')
    for name, digest in ready['artifact_sha256'].items():
        if file_hash(args.prepared_root / name) != digest:
            raise ValueError(f'Locked TRAIN preparation changed: {name}')
    manifest = read_json(args.prepared_root / 'source_cache_manifest.json')
    for split in ('train', 'val'):
        if set(manifest['splits'].get(split, {})) != set(PROBLEMS) or any(
                manifest['splits'][split][p]['rows'] != SPLIT_ROWS[split] for p in PROBLEMS):
            raise ValueError('Preparation must retain every TRAIN/VAL row')
    if args.cache_root is None:
        args.cache_root = Path(ready['cache_source']).resolve()
    return (ready, manifest, read_json(args.prepared_root / 'geometry_stats.json'),
            read_json(args.prepared_root / 'common_full_train_sbs.json'))


def protocol(args, ready, manifest, geometry, baselines, params, order, caps):
    attention, focal = args.arm in ATTENTION_ARMS, args.arm in FOCAL_ARMS
    return dict(experiment='R84', schema_version=1, arm=args.arm, seed=2,
        initialization='scratch_shared_r77', model_params=params, model_factory=MODEL_FACTORY,
        solver_names=list(GLOBAL_SOLVERS), scenario_solver_aliases=manifest.get('scenario_solver_aliases', {}),
        data_root=str(args.data_root.resolve()), cache_root=str(args.cache_root.resolve()),
        prepared_root=str(args.prepared_root.resolve()), preparation_sha256=file_hash(args.prepared_root / 'prepared.json'),
        common_trainable_sha256=ready['trainable_sha256'], rows_per_problem=SPLIT_ROWS,
        train_rows_per_problem=SPLIT_ROWS['train'], heldout_rows_per_problem=SPLIT_ROWS['val'],
        candidate_pools={p: manifest['splits']['train'][p]['pool'] for p in PROBLEMS},
        candidate_pool_ids={p: manifest['splits']['train'][p]['pool_ids'] for p in PROBLEMS},
        winner_rule=WINNER_RULE, baselines=baselines, own_train_sbs=baselines,
        geometry_stats=geometry, global_batch_size=BATCH_SIZE, logical_global_batch=BATCH_SIZE,
        max_successful_updates=MAX_UPDATES, updates_per_problem=MAX_UPDATES // len(PROBLEMS),
        total_sample_exposures=BATCH_SIZE * MAX_UPDATES, eval_every_updates=EVAL_EVERY,
        warmup_updates=WARMUP_UPDATES, rdrop_ramp_updates=WARMUP_UPDATES,
        lr=dict(rule='successful-update warmup then cosine, independent of VAL', peak=PEAK_LR,
            end=END_LR, warmup_updates=WARMUP_UPDATES, max_updates=MAX_UPDATES),
        loss=dict(LOSS_RECIPE, focal_extra=dict(enabled=focal, weight=.35 if focal else 0.,
            gamma=2., alpha=None, warmup=False, two_dropout_mean=True, full_gradient=True)),
        rdrop_weight=.35, dropout=.1, optimizer='AdamW', weight_decay=1e-4, grad_clip=1.,
        fixed_budget=True, early_stop=False,
        checkpoint_rule='strict minimum full VAL macro_vs_sbs_pct with common new81000 TRAIN-fitted SBS; first wins ties',
        inference_rule=('eval mode, direct attention logits, stable ascending-global-solver-ID tie break'
                        if attention else 'eval mode, original R77 maximin logits, stable ascending-global-solver-ID tie break'),
        selection_head='direct_attention' if attention else 'original_score_mlp_maximin',
        task_order_sha256=hashlib.sha256('\n'.join(order).encode('ascii')).hexdigest(),
        source_plan='common seed2 full18 natural stream; exact1024 rows, shuffled tails carried across cycles',
        microbatch_caps=caps, gpu_type=args.gpu_type, threads=2, prefetch_buffers=2, world_size=1,
        gradients='micro rows / global1024; one clip/step; AMP overflow replays same batch and RNG',
        train_eval='full TRAIN every5 VAL points plus the selected best if absent',
        eval_batch_size=args.eval_batch_size, test='one full TEST after unique VAL-best lock; saved predictions reusable',
        test_rnd_disclosure='TEST informed earlier R&D; preserved old holdouts plus previously unused input-stratified extensions',
        nss_reference='locked R79 weights, fresh predictions on the same new4500 TEST; no NSS retraining',
        source_sha256={name: file_hash(V4 / name) for name in
            ('r84_experiment.py', 'r84_attention_selector.py', 'r84_objective.py', 'r84_prepare.py',
             'pairwise_selector.py', 'pairwise_objective.py', 'r76_experiment.py', 'r77_experiment.py')},
        wandb=True, wandb_mode=args.wandb_mode)


def load_checkpoint(path, device='cpu'):
    state = torch.load(path, map_location='cpu', weights_only=False)
    config = state['args']
    if (config.get('experiment') != 'R84' or config.get('arm') not in ARMS
            or config.get('seed') != 2 or config.get('initialization') != 'scratch_shared_r77'
            or config.get('solver_names') != GLOBAL_SOLVERS or state.get('state_hash') != json_hash(config)
            or config.get('max_successful_updates') != MAX_UPDATES):
        raise ValueError('Checkpoint is not an unchanged fresh R84 state')
    model = make_model(config['model_params'], config['arm'], state_dict=state['model'])
    return model.to(device), state


load_model = load_checkpoint
from_checkpoint = load_checkpoint


def status_update(args, phase, step=None, **values):
    path = args.root / args.arm / 'status.json'
    previous = read_json(path) if path.exists() else {}
    step = previous.get('successful_updates', 0) if step is None else step
    previous.update(experiment=getattr(args, 'experiment', 'R84'), arm=args.arm, phase=phase, pid=os.getpid(), host=socket.gethostname(),
        gpu=os.environ.get('CUDA_VISIBLE_DEVICES', 'external allocation'), gpu_type=args.gpu_type,
        device=str(args.device), updated_utc=utc(), successful_updates=step, step=step,
        max_successful_updates=MAX_UPDATES, sample_exposures=step * BATCH_SIZE)
    previous.update(values)
    write_json(path, previous)


def summarize(per_problem):
    result = dict(per_problem=per_problem, macro=aggregate(per_problem), families=families(per_problem),
        native_total=sum(v['n'] for v in per_problem.values()),
        native_correct=sum(v['native_correct'] for v in per_problem.values()), winner_rule=WINNER_RULE,
        sbs_reference='common new81000 TRAIN-fitted SBS, evaluated on the named split',
        all_aggregation='equal weight per problem; macro raw mean_cost descriptive only')
    result['macro'].update(macro_sbs_cost=float(np.mean([v['sbs_cost'] for v in per_problem.values()])),
        macro_oracle_cost=float(np.mean([v['oracle_cost'] for v in per_problem.values()])))
    return result


@torch.no_grad()
def evaluate(selector, args, manifest, baselines, split, tag, step, state_hash, checkpoint_sha256=None):
    root = args.root / args.arm
    directory = root / 'evaluation' / split / tag
    directory.mkdir(parents=True, exist_ok=True)
    identity = dict(experiment=getattr(args, 'experiment', 'R84'), arm=args.arm, split=split, tag=tag, successful_updates=step,
        state_hash=state_hash, model_state_sha256=parameter_hash(selector.state_dict()),
        checkpoint_sha256=checkpoint_sha256, winner_rule=WINNER_RULE)
    identity_path = directory / 'identity.json'
    if identity_path.exists() and read_json(identity_path) != identity:
        raise ValueError('Saved evaluation belongs to a different state')
    write_json(identity_path, identity)
    if (directory / 'metrics.json').exists() and (directory / 'rank000.npz').exists():
        return read_json(directory / 'metrics.json')
    was_training, arrays, per_problem = selector.training, {}, {}
    selector.eval()
    device, started = next(selector.parameters()).device, time.monotonic()
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
                    cap = min(args.eval_batch_size, micro_caps(args.gpu_type, args.large_micro_cap)[problem])
                    requests = ((problem, torch.arange(i, min(i + cap, cache.size)), min(cap, cache.size - i))
                                for i in range(0, cache.size, cap))
                    scores = []
                    with PinnedPrefetch({problem: cache}, requests, device) as batches:
                        for _, _, _, batch in batches:
                            scores.append(selector(inputs_only(batch))['logits'].float().cpu().numpy())
                    with np.load(args.data_root / (problem + split) / 'input_metadata.npz', allow_pickle=False) as metadata:
                        source_indices = metadata['source_index'].copy()
                    data = dict(logits=np.concatenate(scores), indices=np.arange(cache.size, dtype=np.int64),
                        source_indices=np.arange(cache.size, dtype=np.int64), release_source_indices=source_indices,
                        costs=cache.raw['costs'].copy(), winner=cache.raw['winner'].copy(),
                        pool=np.asarray(cache.pool), pool_ids=np.asarray(cache.pool_ids, dtype=np.int64),
                        nodes=np.asarray(cache.lengths).copy())
                    save_npz(shard, data)
                finally:
                    cache.close()
            count = manifest['splits'][split][problem]['rows']
            if (count != SPLIT_ROWS[split] or data['costs'].dtype != np.float64
                    or data['winner'].dtype != np.int64 or len(data['logits']) != count
                    or not np.array_equal(data['indices'], np.arange(count))
                    or not np.array_equal(data['winner'], data['costs'].argmin(1))):
                raise ValueError(f'Incomplete native predictions: {problem}/{split}')
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
        sample_exposures=step * BATCH_SIZE, identity=identity,
        prediction_archive=str((directory / 'rank000.npz').resolve()), own_train_sbs=summarize(per_problem))
    write_json(directory / 'metrics.json', result)
    print('[' + getattr(args, 'experiment', 'R84') + ' eval] ' + json.dumps(dict(arm=args.arm, split=split, tag=tag, **result['macro'])), flush=True)
    return result


def validate_resume(state, config, order):
    step, pending = state['successful_updates'], state.get('pending_evaluation') is not None
    if (state['args'] != config or state['state_hash'] != json_hash(config)
            or not 0 <= step <= MAX_UPDATES or step % EVAL_EVERY
            or step != (len(state['history']) + int(pending)) * EVAL_EVERY):
        raise ValueError('Resume requires the unchanged protocol at a completed interval boundary')
    counts = Counter(order[:step])
    if (Counter(state['updates_by_problem']) != counts
            or any(state['stream_draws'][p] != counts[p] * BATCH_SIZE for p in PROBLEMS)):
        raise ValueError('Resume source cursors disagree with successful updates')


def train(args, device, *, integration=None):
    hooks = integration or sys.modules[__name__]
    root = args.root / args.arm
    if (root / 'training_complete.json').exists():
        return hooks.test(args, device)
    ready, manifest, geometry, baselines = hooks.prepared_config(args)
    order, caps = task_order(MAX_UPDATES), micro_caps(args.gpu_type, args.large_micro_cap)
    initial = torch.load(args.prepared_root / 'identicalinit_seed2.pt', map_location='cpu', weights_only=False)
    selector, params, initialization = hooks.initialize_shared(initial, geometry, args.arm, device)
    del initial
    config = hooks.protocol(args, ready, manifest, geometry, baselines, params, order, caps)
    state_hash = json_hash(config)
    if args.state_hash is not None and args.state_hash != state_hash:
        raise ValueError('Expected state hash differs from prepared protocol')
    protocol_path, last = root / 'protocol.json', root / 'last.pt'
    if protocol_path.exists() and read_json(protocol_path) != config:
        raise ValueError('Existing ' + config['experiment'] + ' protocol differs; preserve this run')
    if last.exists() and not args.resume:
        raise ValueError('Use --resume for this existing run; no state will be overwritten')
    write_json(protocol_path, config)
    status_update(args, 'initializing', 0, state_hash=state_hash)
    checkpoint = None
    if last.exists():
        del selector
        selector, checkpoint = hooks.load_checkpoint(last, device)
        validate_resume(checkpoint, config, order)
        initialization = checkpoint['initialization']
    else:
        write_json(root / 'initialization.json', initialization)
    optimizer = torch.optim.AdamW(selector.parameters(), lr=PEAK_LR, weight_decay=1e-4)
    scaler = torch.amp.GradScaler('cuda', enabled=device.type == 'cuda')
    objective, runtime = hooks.make_objective(selector, args.arm), Runtime(0, 1, device)
    history, best, successful, retries_total, pending = [], None, 0, 0, None
    counts, cursors = Counter(), dict.fromkeys(PROBLEMS, 0)
    if checkpoint is not None:
        if integration is not None:
            hooks.restore_objective(objective, checkpoint)
        optimizer.load_state_dict(checkpoint['optimizer'])
        for state in optimizer.state.values():
            for key, value in state.items():
                if torch.is_tensor(value):
                    state[key] = value.cpu() if key == 'step' else value.to(device)
        scaler.load_state_dict(checkpoint['scaler'])
        history, best, successful = checkpoint['history'], checkpoint['best'], checkpoint['successful_updates']
        retries_total, counts = checkpoint['amp_retries'], Counter(checkpoint['updates_by_problem'])
        cursors, pending = checkpoint['stream_draws'], checkpoint.get('pending_evaluation')
        restore_rank_rng(checkpoint['rng'], device)
    else:
        set_seed(2)
    del checkpoint
    streams = {p: ShuffleStream(SPLIT_ROWS['train'], task_seed(2, p), cursors[p]) for p in PROBLEMS}
    caches, tracker = {}, None

    def snapshot():
        value = dict(model=selector.state_dict(), optimizer=optimizer.state_dict(), scaler=scaler.state_dict(),
            args=config, state_hash=state_hash, initialization=initialization, epoch=len(history),
            successful_updates=successful, updates_by_problem=dict(counts), sample_exposures=successful * BATCH_SIZE,
            stream_draws={p: stream.draws for p, stream in streams.items()}, amp_retries=retries_total,
            rng=capture_rank_rng(device), history=history, best=best, pending_evaluation=pending)
        if integration is not None:
            value.update(hooks.checkpoint_extras(objective))
        return value

    if not last.exists():
        save_checkpoint(last, snapshot())
    try:
        tracker = wandb_run(args, config)
        rng = torch.load(last, map_location='cpu', weights_only=False)['rng']
        restore_rank_rng(rng, device)
        del rng
        while successful < MAX_UPDATES or pending is not None:
            if pending is None:
                if not caches:
                    for problem in PROBLEMS:
                        caches[problem] = open_cache(args.cache_root, 'train', problem, manifest, args.data_root)
                        if caches[problem].size != SPLIT_ROWS['train']:
                            raise ValueError('Keep all81000 TRAIN instances')
                end, first = min(MAX_UPDATES, successful + EVAL_EVERY), successful
                requests = [(p, streams[p].take(BATCH_SIZE), BATCH_SIZE) for p in order[successful:end]]
                started = time.monotonic()
                weighted = torch.zeros(len(PART_KEYS), dtype=torch.float64, device=device)
                extra_weighted = {}
                selector.train()
                with PinnedPrefetch(caches, requests, device) as batches:
                    for problem, n_rank, n_global, batch in batches:
                        if n_rank != BATCH_SIZE or n_global != BATCH_SIZE:
                            raise ValueError('Each update must contain exactly1024 natural rows')
                        lr, weight = learning_rate(successful + 1), consistency_weight(successful + 1)
                        for group in optimizer.param_groups:
                            group['lr'] = lr
                        if integration is None:
                            parts, retries, extra_metrics = update_batch(objective, optimizer, scaler, batch,
                                weight, runtime, caps[problem])
                        else:
                            parts, retries, extra_metrics = hooks.update_batch(objective, optimizer, scaler, batch,
                                weight, runtime, caps[problem], problem=problem)
                        successful += 1
                        counts[problem] += 1
                        retries_total += retries
                        weighted.add_(parts.double())
                        if extra_weighted and set(extra_metrics) != set(extra_weighted):
                            raise ValueError('Objective metric keys changed within an evaluation interval')
                        for key, value in extra_metrics.items():
                            if key not in extra_weighted:
                                extra_weighted[key] = torch.zeros((), dtype=torch.float64, device=device)
                            extra_weighted[key].add_(torch.as_tensor(value, device=device).detach().double())
                        if successful == 1 or successful % args.log_every == 0 or successful == end:
                            done, elapsed = successful - first, time.monotonic() - started
                            info = dict(successful_updates=successful, max_updates=MAX_UPDATES,
                                problem=problem, lr=lr, consistency_weight=weight,
                                sample_exposures=successful * BATCH_SIZE, seconds_per_update=elapsed / done,
                                samples_per_second=done * BATCH_SIZE / elapsed, amp_retries=retries_total,
                                updates_by_problem=dict(counts),
                                **{key: float(value / done) for key, value in extra_weighted.items()},
                                **dict(zip(PART_KEYS, (weighted / done).cpu().tolist())))
                            val_seconds = float(np.mean([r['val']['seconds'] for r in history])) if history else 0.
                            eta = (MAX_UPDATES - successful) * elapsed / done + (40 - len(history)) * val_seconds
                            status_update(args, 'training', successful, eta_seconds=eta,
                                **{k: v for k, v in info.items() if k != 'successful_updates'})
                            with (root / 'updates.jsonl').open('a') as stream:
                                stream.write(json.dumps(dict(updated_utc=utc(), **info), allow_nan=False) + '\n')
                            print('[' + config['experiment'] + ' train] ' + json.dumps(dict(arm=args.arm, **info)), flush=True)
                            if tracker is not None:
                                tracker.log({**{'optimization/' + key: info[key] for key in PART_KEYS},
                                    **{'optimization/' + key: info[key] for key in extra_weighted},
                                    'successful_updates': successful, 'lr': lr,
                                    'samples_per_second': info['samples_per_second']}, step=successful)
                pending = dict(zip(PART_KEYS, (weighted / (successful - first)).cpu().tolist()))
                pending.update(lr=lr, consistency_weight=weight, amp_retries=retries_total,
                    **{key: float(value / (successful - first)) for key, value in extra_weighted.items()},
                    samples=(successful - first) * BATCH_SIZE, seconds=time.monotonic() - started)
                save_checkpoint(last, snapshot())
            point, tag = successful // EVAL_EVERY, f'update{successful:07d}'
            validation = hooks.evaluate(selector, args, manifest, baselines, 'val', tag, successful, state_hash)
            cost = validation['macro']['macro_vs_sbs_pct']
            if not math.isfinite(cost):
                raise ValueError('VAL primary metric must be finite')
            training = hooks.evaluate(selector, args, manifest, baselines, 'train', tag, successful, state_hash) if point % 5 == 0 else None
            row = dict(epoch=point, evaluation_point=point, successful_updates=successful,
                sample_exposures=successful * BATCH_SIZE, updates_by_problem=dict(counts),
                evaluation_tag=tag, optimization=pending, val=validation, train=training)
            history.append(row)
            improved = best is None or cost < best['val']['macro']['macro_vs_sbs_pct']
            if improved:
                best = dict(row)
            pending = None
            state = snapshot()
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
                values = dict(successful_updates=successful)
                for split, result in (('val', validation), ('train_eval', training)):
                    if result is not None:
                        values.update({split + '/' + k: v for k, v in result['macro'].items()})
                tracker.log(values, step=successful)
        if successful != MAX_UPDATES or any(counts[p] != MAX_UPDATES // len(PROBLEMS) for p in PROBLEMS):
            raise ValueError('The fixed full18 successful-update budget was not completed')
        for cache in caches.values():
            cache.close()
        caches.clear()
        del objective, optimizer, scaler, selector
        selector, selected = hooks.load_checkpoint(root / 'best.pt', device)
        if selected['args'] != config or selected['successful_updates'] != best['successful_updates']:
            raise ValueError('Selected checkpoint differs from VAL history')
        del selected
        lock = lock_best(root, best, config, state_hash)
        destination = root / 'evaluation/val/selected_best'
        destination.mkdir(parents=True, exist_ok=True)
        for name in ('rank000.npz', 'metrics.json', 'identity.json'):
            target = destination / name
            if not target.exists():
                shutil.copyfile(root / 'evaluation/val' / best['evaluation_tag'] / name, target)
        if best['train'] is None:
            best = dict(best, train=hooks.evaluate(selector, args, manifest, baselines, 'train', 'selected_best',
                best['successful_updates'], state_hash, lock['checkpoint_sha256']))
        for row in history:
            if row['successful_updates'] == best['successful_updates']:
                row['train'] = best['train']
        write_json(root / 'history.json', history)
        write_json(root / 'best_eval.json', best)
        write_metrics_csv(root, history)
        write_json(root / 'result.json', dict(best=best, final=history[-1], successful_updates=successful,
            updates_by_problem=dict(counts), amp_retries=retries_total, state_hash=state_hash,
            stop_reason='fixed successful-update budget; no early stopping', test_read=False))
        write_json(root / 'final_metrics.json', dict(epoch=best['epoch'], selected_update=best['successful_updates'],
            checkpoint_sha256=lock['checkpoint_sha256'], state_hash=state_hash, train=best['train'], val=best['val'], test_read=False))
        write_json(root / 'training_complete.json', dict(completed_utc=utc(), successful_updates=successful,
            sample_exposures=successful * BATCH_SIZE, updates_by_problem=dict(counts), selected_best=lock,
            state_hash=state_hash, next_stage='locked full TEST', test_read=False))
        status_update(args, 'training_complete', successful, selected_update=best['successful_updates'])
        del selector
    finally:
        for cache in caches.values():
            cache.close()
        if tracker is not None:
            tracker.finish()
    return hooks.test(args, device)


def authorize_test(args):
    root = args.root / args.arm
    complete = read_json(root / 'training_complete.json')
    lock, config = read_json(root / 'final_test_lock.json'), read_json(root / 'protocol.json')
    experiment = getattr(args, 'experiment', 'R84')
    if (complete.get('successful_updates') != MAX_UPDATES or config.get('experiment') != experiment
            or lock.get('experiment') != experiment or config.get('arm') != args.arm or lock.get('arm') != args.arm
            or not lock.get('selected_without_test') or lock.get('per_problem_assembly') is not False
            or lock.get('state_hash') != json_hash(config) or config['max_successful_updates'] != MAX_UPDATES
            or Path(lock['checkpoint']).resolve() != (root / 'best.pt').resolve()
            or lock['checkpoint_sha256'] != file_hash(root / 'best.pt')
            or Path(config['data_root']).resolve() != args.data_root.resolve()):
        raise PermissionError('TEST requires the complete budget and unchanged unique ' + experiment + ' VAL lock')
    if args.state_hash is not None and args.state_hash != lock['state_hash']:
        raise ValueError('Expected state hash differs from immutable lock')
    return lock, config


def ensure_test_cache(args, lock):
    """One shared TEST feature cache, first created after a genuine arm VAL lock."""
    from .r84_prepare import _Scenario, finalize, prepare_problem
    args.prepared_root.mkdir(parents=True, exist_ok=True)
    with (args.prepared_root / 'test_cache_prepare.lock').open('a') as claim:
        fcntl.flock(claim, fcntl.LOCK_EX)
        scenario = _Scenario(args.data_root)
        for problem in PROBLEMS:
            directory = args.cache_root / 'test' / problem
            path = directory / 'metadata.json'
            if not path.exists() or not read_json(path).get('complete'):
                prepare_problem(args.prepared_root, scenario, args.cache_root, problem, 'test',
                    torch.device(args.device), [Path('/tmp/shiys_R76_R45A_single_reld_mt_cache')],
                    Path('/public/home/shiys/selection_data_new_only_B_filtered_single_reld_mt_seed2_20261014'),
                    args.root / args.arm)
        finalize(args.prepared_root, args.data_root, args.cache_root, splits=('test',))
    return read_json(args.prepared_root / 'source_cache_manifest.json')


def test(args, device, *, integration=None):
    hooks = integration or sys.modules[__name__]
    root = args.root / args.arm
    lock, config = hooks.authorize_test(args)
    args.cache_root = Path(config['cache_root']) if args.cache_root is None else args.cache_root
    args.prepared_root = Path(config['prepared_root'])
    args.gpu_type, args.eval_batch_size = config['gpu_type'], config['eval_batch_size']
    args.large_micro_cap = config['microbatch_caps']['TSP']
    result_path = root / 'test_results.json'
    if result_path.exists():
        saved = read_json(result_path)
        if (not saved.get('complete') or saved.get('checkpoint_sha256') != lock['checkpoint_sha256']
                or saved.get('state_hash') != lock['state_hash']):
            raise PermissionError('Saved TEST differs from locked checkpoint')
        result = saved['result']
    else:
        manifest = hooks.ensure_test_cache(args, lock)
        model, state = hooks.load_checkpoint(root / 'best.pt', device)
        if state['args'] != config or state['successful_updates'] != lock['selected_update']:
            raise PermissionError('TEST factory differs from locked selection')
        del state
        write_json(root / 'test_started.json', dict(started_utc=utc(), checkpoint_sha256=lock['checkpoint_sha256']))
        result = hooks.evaluate(model, args, manifest, config['baselines'], 'test', 'selected_best',
            lock['selected_update'], lock['state_hash'], lock['checkpoint_sha256'])
        if file_hash(root / 'best.pt') != lock['checkpoint_sha256']:
            raise PermissionError('Checkpoint changed during TEST')
        write_json(result_path, dict(complete=True, completed_utc=utc(), result=result, epoch=lock['epoch'],
            selected_update=lock['selected_update'], checkpoint_sha256=lock['checkpoint_sha256'], state_hash=lock['state_hash']))
        del model
    final = read_json(root / 'final_metrics.json')
    final.update(test=result, test_read=True, complete=True, checkpoint_sha256=lock['checkpoint_sha256'])
    write_json(root / 'final_metrics.json', final)
    write_json(root / 'test_complete.json', dict(full_tests=1, completed_utc=utc(),
        checkpoint_sha256=lock['checkpoint_sha256'], saved_predictions_reusable=True))
    status_update(args, 'test_complete', MAX_UPDATES, selected_update=lock['selected_update'],
        test=result['macro'], eta_seconds=0., state_hash=lock['state_hash'])
    return result


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('command', choices=('train', 'test', 'describe', 'status'))
    result.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    result.add_argument('--arm', choices=ARMS, required=True)
    result.add_argument('--prepared-root', type=Path, default=DEFAULT_PREPARED_ROOT)
    result.add_argument('--data-root', type=Path, default=DEFAULT_DATA_ROOT)
    result.add_argument('--cache-root', type=Path)
    result.add_argument('--device', default='cuda:0')
    result.add_argument('--gpu-type', choices=('4090', '3090'), default='4090')
    result.add_argument('--large-micro-cap', type=int)
    result.add_argument('--eval-batch-size', type=int, default=256)
    result.add_argument('--log-every', type=int, default=18)
    result.add_argument('--wandb-mode', choices=('auto', 'online', 'offline'), default='auto')
    result.add_argument('--resume', action='store_true')
    result.add_argument('--state-hash', '--expected-state-hash')
    result.set_defaults(experiment='R84')
    return result


def main(argv=None, *, integration=None):
    hooks = integration or sys.modules[__name__]
    args = hooks.parser().parse_args(argv)
    for name in ('root', 'prepared_root', 'data_root', 'cache_root'):
        value = getattr(args, name)
        if value is not None:
            setattr(args, name, value.expanduser().resolve())
    if args.command == 'describe':
        if integration is not None:
            value = hooks.describe(args)
            print(json.dumps(value, indent=2))
            return value
        value = dict(experiment='R84', arm=args.arm, rows=SPLIT_ROWS, seed=2, batch_size=BATCH_SIZE,
            max_updates=MAX_UPDATES, eval_every=EVAL_EVERY, warmup_updates=WARMUP_UPDATES,
            microbatch_caps=micro_caps(args.gpu_type, args.large_micro_cap),
            attention=args.arm in ATTENTION_ARMS, focal=args.arm in FOCAL_ARMS)
        print(json.dumps(value, indent=2))
        return value
    if args.command == 'status':
        path = args.root / args.arm / 'status.json'
        value = read_json(path) if path.exists() else dict(phase='not_started', arm=args.arm)
        print(json.dumps(value, indent=2))
        return value
    root = args.root / args.arm
    root.mkdir(parents=True, exist_ok=True)
    if args.eval_batch_size < 1 or args.log_every < 1:
        raise ValueError('Evaluation and log intervals must be positive')
    with (root / 'runner.lock').open('a') as claim:
        fcntl.flock(claim, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            torch.set_num_threads(2)
            torch.set_num_interop_threads(2)
            configure_torch()
            device = torch.device(args.device)
            if device.type != 'cuda':
                raise ValueError('Formal training/TEST requires a launcher-assigned CUDA device')
            return hooks.train(args, device) if args.command == 'train' else hooks.test(args, device)
        except BaseException as error:
            receipt = dict(type=type(error).__name__, message=str(error), updated_utc=utc(),
                traceback=traceback.format_exc(), pid=os.getpid(), host=socket.gethostname())
            write_json(root / 'error.json', receipt)
            status_update(args, 'failed', error=receipt)
            raise


if __name__ == '__main__':
    main()
