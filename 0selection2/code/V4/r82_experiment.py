"""R82B fixed-budget context-matched continuous-regret ranking runner.

No allocator or queue lives here. Formal GPU commands require an externally
assigned device. Preparation and the CPU witnesses never launch training.
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
from pathlib import Path

import numpy as np
import torch

from code.unified_selector.registry import GLOBAL_SOLVERS, PROBLEMS
from .multitask_probe import task_seed
from .pairwise_objective import inputs_only
from .r76_experiment import (
    DATA_ROOT, LOSS_RECIPE, PinnedPrefetch, capture_rank_rng, file_hash,
    fixed_sbs_metrics, open_cache, read_json, restore_rank_rng, save_checkpoint,
    utc, write_json,
)
from .r77_data import DEFAULT_ROOT as R77_ROOT, FULL_ROWS, HELDOUT_ROWS, parameter_hash
from .r77_experiment import ShuffleStream, initialize_identical, micro_caps, task_order
from .r80_experiment import (
    json_hash, prepared_config as r77_sources, save_npz, summarize,
)
from .solver_code_encoder import make_r45_model
from .train import configure_torch, set_seed


V4 = Path(__file__).resolve().parent
DEFAULT_ROOT = V4 / 'runs/R82_context_matched_ranking'
GROUP = 'B_context_matched_rank_seed2'
ALIAS = 'Requested_R80B_context_matched_residual_ranking'
BATCH, MAX_UPDATES, WARMUP, EVAL_EVERY = 1024, 49680, 3726, 1242
WINNER_RULE = 'native int64 ind verified against original FP64 argmin; never FP32 argmin'
LOCAL_CACHE = Path('/tmp/shiys_R76_R45A_single_reld_mt_cache')
TARGET_RULE = 'same TRAIN-defined context and method; lower exact FP64 relative regret receives higher applicability; equal regrets excluded'


def array_hash(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def native_targets(costs):
    if costs.dtype != np.float64 or costs.ndim != 2 or not costs.shape[1]:
        raise ValueError('Targets require original FP64 costs, never upcast batch FP32')
    if not np.isfinite(costs).all() or np.any(costs <= 0):
        raise ValueError('Legal native costs must be positive and finite')
    minimum = costs.min(1, keepdims=True)
    minima = costs == minimum
    pos = minima & (minima.sum(1, keepdims=True) == 1)
    neg = costs > minimum
    return pos, neg, pos.any(0) & neg.any(0)


def learning_rate(update):
    if not 1 <= update <= MAX_UPDATES:
        raise ValueError('Learning rate uses the one-based successful-update axis')
    if update <= WARMUP:
        return 1e-4 * update / WARMUP
    progress = (update - WARMUP) / (MAX_UPDATES - WARMUP)
    return 1e-6 + .5 * (1e-4 - 1e-6) * (1 + math.cos(math.pi * progress))


def ramp_weight(update):
    return .35 * min(1., update / WARMUP)


def cmr_weight(update):
    return .10 * min(1., update / WARMUP)


def cache_directory(root, split, problem):
    candidates = (root / split / problem, root / (problem + split))
    return next((path for path in candidates if path.is_dir()), candidates[0])


def source_config(args):
    if args.cache_root is None and LOCAL_CACHE.is_dir():
        args.cache_root = LOCAL_CACHE.resolve()
    return r77_sources(args)


def prepared_path(args):
    return args.root / 'prepared/prepared.json'


def load_prepared(args):
    path = prepared_path(args)
    ready = read_json(path)
    if (ready.get('experiment') != 'R82' or ready.get('group') != GROUP
            or ready.get('alias') != ALIAS or not ready.get('train_only')
            or not ready.get('pair_coverage_sufficient')
            or Path(ready['data_root']).resolve() != args.data_root.resolve()
            or Path(ready['r77_root']).resolve() != args.prepared_root.resolve()):
        raise ValueError('Require unchanged R82B TRAIN-only preparation with sufficient matched-pair coverage')
    if args.cache_root is None:
        args.cache_root = Path(ready['cache_root'])
    for name, digest in ready['artifact_sha256'].items():
        if file_hash(args.root / name) != digest:
            raise ValueError(f'Prepared TRAIN support changed: {name}')
    return ready


def freeze_historical_val(args):
    """Independent diagnostic artifact; never included in training config/masks."""
    output = args.root / 'diagnostic/historical_r77b_val.json'
    if output.exists():
        saved = read_json(output)
        if saved.get('r77_root') != str(args.prepared_root):
            raise ValueError('Historical diagnostic baseline belongs to another R77 root')
        if saved.get('status') == 'complete':
            if file_hash(Path(saved['prediction_path'])) != saved['prediction_sha256']:
                raise ValueError('Historical VAL predictions changed after diagnostic freeze')
        return saved
    run = args.prepared_root / 'B_full_seed2'
    lock_path = run / 'final_test_lock.json'
    if not lock_path.exists():
        record = dict(status='missing_historical_VAL_lock', r77_root=str(args.prepared_root),
            diagnostic_only=True, training_influence=False, matched_r82a='not_run')
        write_json(output, record)
        return record
    lock = read_json(lock_path)
    if (lock.get('experiment') != 'R77' or lock.get('arm') != 'B'
            or lock.get('selected_without_test') is not True):
        raise ValueError('Historical VAL diagnostic requires the locked R77B model')
    path = run / f"evaluation/val/epoch{lock['epoch']:03d}/rank000.npz"
    rows, arrays = [], {}
    with np.load(path, allow_pickle=False) as saved:
        for problem in PROBLEMS:
            get = lambda key: saved[problem + '__' + key]
            costs, winner, scores, pool, ids = (get(key) for key in ('costs', 'winner', 'logits', 'pool', 'pool_ids'))
            if costs.dtype != np.float64 or len(winner) != HELDOUT_ROWS:
                raise ValueError('Historical diagnostics require complete native VAL predictions')
            np.testing.assert_array_equal(winner, costs.argmin(1))
            order = np.argsort(ids, kind='stable')
            pred = order[np.argmax(scores[:, order], axis=1)]
            for slot, (name, solver_id) in enumerate(zip(pool.tolist(), ids.tolist())):
                support, selected = int((winner == slot).sum()), int((pred == slot).sum())
                rows.append(dict(problem=problem, method=name, solver_id=solver_id,
                    n=len(winner), native_winner_support=support, selection_count=selected,
                    selection_pct=100 * selected / len(winner),
                    zero=bool(support > 0 and selected == 0),
                    near_zero=bool(support > 0 and 100 * selected / len(winner) <= .1)))
            arrays.update({problem + '__' + key: value for key, value in
                dict(pred=pred, winner=winner, costs=costs, pool=pool, pool_ids=ids).items()})
    archive = args.root / 'diagnostic/historical_r77b_val.npz'
    save_npz(archive, arrays)
    record = dict(experiment='R82', status='complete', alias=ALIAS,
        r77_root=str(args.prepared_root), checkpoint_sha256=lock['checkpoint_sha256'],
        selected_update=lock['selected_update'], prediction_path=str(path.resolve()),
        prediction_sha256=file_hash(path), frozen_archive=str(archive.resolve()),
        frozen_archive_sha256=file_hash(archive), methods=rows,
        zero_support=sum(r['native_winner_support'] for r in rows if r['zero']),
        near_zero_support=sum(r['native_winner_support'] for r in rows if r['near_zero']),
        near_zero_rule='native winner support >0 and selection percentage <=0.1%; includes zero',
        zero_rule='native winner support >0 and selection count ==0',
        diagnostic_only=True, training_influence=False, historical_reference_only=True,
        matched_r82a='not_run; no strict A/B causal conclusion', winner_rule=WINNER_RULE)
    write_json(output, record)
    return record


def prepare(args):
    from .r82_context import prepare_training_context
    record = prepare_training_context(args)
    load_prepared(args)
    diagnostic = freeze_historical_val(args)
    print(json.dumps(dict(preparation_receipt=str(prepared_path(args)),
        sha256=file_hash(prepared_path(args)), native_train_cost_bytes=record.get('native_train_cost_bytes'),
        diagnostic_zero_support=diagnostic.get('zero_support'), training_started=False), indent=2), flush=True)
    return record


def protocol(args):
    prepared = load_prepared(args)
    ready, manifest, geometry, baselines = source_config(args)
    if prepared['r77_prepared_sha256'] != file_hash(args.prepared_root / 'prepared.json'):
        raise ValueError('R77 preparation changed after R82 support freeze')
    caps = micro_caps(args.gpu_type, args.large_micro_cap)
    order = task_order()
    support = read_json(args.root / 'support/manifest.json')['problems']
    sources = ['r82_experiment.py', 'context_matched_objective.py', 'r82_context.py', 'pairwise_selector.py', 'R82_INTERFACE.md']
    config = dict(experiment='R82', alias=ALIAS, arm=GROUP, group=GROUP, schema_version=1,
        seed=2, initialization='scratch_shared_r77', historical_checkpoint_loaded=False,
        model_params=ready['model_params'], solver_names=GLOBAL_SOLVERS,
        data_root=str(args.data_root), cache_root=str(args.cache_root), source_cache_root=prepared['cache_root'],
        runtime_cache_policy='existing source-identical local cache; open_cache metadata/pool/rows and native FP64 cost hashes checked before optimization; no copy/rebuild',
        prepared_root=str(args.prepared_root),
        r82_preparation_sha256=file_hash(prepared_path(args)), r77_preparation_sha256=prepared['r77_prepared_sha256'],
        common_trainable_sha256=ready['trainable_sha256'], train_rows_per_problem=FULL_ROWS,
        data_manifest=manifest['splits'], winner_rule=WINNER_RULE, baselines=baselines, own_train_sbs=baselines,
        geometry_stats=geometry, global_batch_size=BATCH, logical_global_batch=BATCH,
        max_successful_updates=MAX_UPDATES, updates_per_problem=MAX_UPDATES // len(PROBLEMS),
        total_sample_exposures=BATCH * MAX_UPDATES, sample_exposures_per_problem=BATCH * MAX_UPDATES // len(PROBLEMS),
        eval_every_updates=EVAL_EVERY, warmup_updates=WARMUP, rdrop_ramp_updates=WARMUP,
        lr=dict(rule='successful-update warmup then cosine; independent of VAL', peak=1e-4, end=1e-6,
            warmup_updates=WARMUP, max_updates=MAX_UPDATES),
        loss=dict(base=LOSS_RECIPE, cmr_weight=.10, cmr_ramp_updates=WARMUP,
            cmr_score='u_m - logsumexp(other legal u)',
            cmr='weighted pairwise logistic over all unequal-regret unordered pairs within the same TRAIN-defined context',
            cmr_weights='clip(abs(r_i-r_j)/0.01,0,1); pair denominator is count, not sum of weights',
            aggregation='mean pairs within group, mean nonempty groups within method, mean participating methods',
            two_views='base and CMR independently evaluated on both dropout views then averaged',
            context='problem + TRAIN-fitted log-n quantile bin + recorded distribution + capacity_type where applicable',
            centering='TRAIN group means saved; within-group subtraction cancels, loss uses original FP64 regret differences',
            target_rule=TARGET_RULE, feature_queue=False),
        dropout=.1, optimizer='AdamW', weight_decay=1e-4, grad_clip=1., fixed_budget=True, early_stop=False,
        checkpoint_rule='strict minimum full VAL macro_vs_sbs_pct with common FULL TRAIN-fitted SBS; first wins ties',
        inference_rule='eval mode, original R77 maximin scores, stable ascending-global-solver-ID tie break',
        task_order_sha256=hashlib.sha256('\n'.join(order).encode('ascii')).hexdigest(),
        source_plan='unchanged R77 seed2 task order and per-task full-population shuffle streams',
        microbatch_caps=caps, gpu_type=args.gpu_type, threads=2, prefetch_buffers=2, world_size=1,
        gradients='full1024 utility GradCache; sequential two-view micro RNG replay; one unscale/clip/step; same batch on AMP retry',
        train_eval='full TRAIN every5 VAL points, plus the selected best if absent',
        eval_batch_size=args.eval_batch_size, test='full TEST only after unique immutable VAL-best lock',
        test_rnd_disclosure='Earlier TEST results informed multi-round R&D; not an untouched blind evaluation',
        train_support=support, train_support_manifest_sha256=file_hash(args.root / 'support/manifest.json'),
        diagnostic_only='historical R77B VAL zero/near-zero list; not loaded by the optimizer or sampling',
        matched_r82a='not_run; historical R77B cannot establish strict A/B causality',
        source_sha256={name: file_hash(V4 / name) for name in sources},
        wandb=True, wandb_project='selector', wandb_entity='yjkds-southern-university-of-science-technology',
        wandb_mode=args.wandb_mode, checkpoint_resume='completed interval boundary, including pending evaluation')
    return config, manifest


def load_checkpoint(path, device='cpu'):
    state = torch.load(path, map_location='cpu', weights_only=False)
    config = state['args']
    if (config.get('experiment') != 'R82' or config.get('group') != GROUP or config.get('arm') != GROUP
            or config.get('initialization') != 'scratch_shared_r77' or config.get('seed') != 2
            or config.get('solver_names') != GLOBAL_SOLVERS or state.get('state_hash') != json_hash(config)):
        raise ValueError('Require unchanged R82B scratch/shared-initialization checkpoint; no historical model fallback')
    model = make_r45_model(config['model_params'], state_dict=state['model'])
    if parameter_hash(model.state_dict()) != parameter_hash(state['model']):
        raise ValueError('R82 factory did not restore every saved parameter and buffer')
    return model.to(device), state


load_model = load_checkpoint
from_checkpoint = load_checkpoint


def status_update(args, phase, step=None, **fields):
    path = args.root / GROUP / 'status.json'
    status = read_json(path) if path.exists() else {}
    if step is None:
        step = status.get('successful_updates', 0)
    status.update(experiment='R82', alias=ALIAS, group=GROUP, arm=GROUP, phase=phase,
        successful_updates=step, step=step, max_successful_updates=MAX_UPDATES,
        sample_exposures=step * BATCH, exposures=step * BATCH, pid=os.getpid(), host=socket.gethostname(),
        gpu=os.environ.get('CUDA_VISIBLE_DEVICES', 'external allocation'), gpu_type=args.gpu_type,
        device=str(args.device), updated_utc=utc())
    status.update(fields)
    write_json(path, status)


class SupervisedCache:
    """Attach TRAIN context IDs and FP64 regrets outside the model inputs."""

    def __init__(self, source, path, expected):
        from .context_matched_objective import CMR_GROUP_KEY, CMR_REGRET_KEY
        self.source, self.group_key, self.regret_key = source, CMR_GROUP_KEY, CMR_REGRET_KEY
        with np.load(path, allow_pickle=False) as saved:
            self.groups, self.regrets = (saved[key] for key in ('groups', 'regrets'))
            np.testing.assert_array_equal(saved['pool'], source.pool)
            np.testing.assert_array_equal(saved['pool_ids'], source.pool_ids)
            np.testing.assert_array_equal(saved['source_indices'], np.arange(source.size))
        if (self.groups.dtype != np.int64 or self.regrets.dtype != np.float64
                or self.groups.shape != (source.size,) or self.regrets.shape != source.raw['costs'].shape
                or np.any(self.groups < 0) or not np.isfinite(self.regrets).all()
                or array_hash(self.groups) != expected['groups_sha256']
                or array_hash(self.regrets) != expected['regrets_sha256']
                or array_hash(source.raw['costs']) != expected['native_costs_sha256']
                or array_hash(source.raw['winner']) != expected['native_winner_sha256']):
            raise ValueError('Frozen TRAIN contexts/regrets and original FP64 costs are misaligned')
        self.size = source.size

    def batch(self, rows, device='cpu'):
        batch = self.source.batch(rows, device)
        rows = rows.cpu().numpy() if torch.is_tensor(rows) else np.asarray(rows)
        batch[self.group_key] = torch.from_numpy(self.groups[rows]).to(device)
        batch[self.regret_key] = torch.from_numpy(self.regrets[rows]).to(device)
        return batch

    def close(self):
        self.source.close()


def update_participation(records, groups, regrets):
    """Count successful-batch unordered unequal pairs, without double-counting dropout views."""
    groups = groups.detach().cpu().numpy() if torch.is_tensor(groups) else np.asarray(groups)
    regrets = regrets.detach().cpu().numpy() if torch.is_tensor(regrets) else np.asarray(regrets)
    for slot, record in enumerate(records):
        record['batches_seen'] += 1
        participated = False
        for group in np.unique(groups):
            values = regrets[groups == group, slot]
            n = len(values)
            _, counts = np.unique(values, return_counts=True)
            pairs = int((n * (n-1) - np.sum(counts * (counts-1))) // 2)
            if not pairs:
                continue
            participated = True
            key = str(int(group))
            detail = record['groups'].setdefault(key, dict(participating_batches=0, pair_count=0, instances_seen=0))
            detail['participating_batches'] += 1
            detail['pair_count'] += pairs
            detail['instances_seen'] += n
            record['pair_count'] += pairs
        record['participating_batches'] += int(participated)


@torch.no_grad()
def evaluate(selector, args, manifest, baselines, split, tag, step, state_hash, checkpoint_sha256=None):
    directory = args.root / GROUP / 'evaluation' / split / tag
    identity = dict(experiment='R82', alias=ALIAS, group=GROUP, arm=GROUP, split=split, tag=tag,
        successful_updates=step, state_hash=state_hash, model_state_sha256=parameter_hash(selector.state_dict()),
        checkpoint_sha256=checkpoint_sha256, winner_rule=WINNER_RULE, auxiliary='raw utility; maximin inference unchanged')
    identity_path = directory / 'identity.json'
    if identity_path.exists() and read_json(identity_path) != identity:
        raise ValueError('Existing evaluation belongs to another R82 model state')
    write_json(identity_path, identity)
    if (directory / 'metrics.json').exists() and (directory / 'rank000.npz').exists():
        return read_json(directory / 'metrics.json')
    device, was_training = next(selector.parameters()).device, selector.training
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
                    cap = min(args.eval_batch_size, micro_caps(args.gpu_type, args.large_micro_cap)[problem])
                    requests = ((problem, torch.arange(start, min(start + cap, cache.size)), min(cap, cache.size-start))
                        for start in range(0, cache.size, cap))
                    scores, utility = [], []
                    with PinnedPrefetch({problem: cache}, requests, device) as batches:
                        for _, _, _, batch in batches:
                            output = selector(inputs_only(batch))
                            scores.append(output['logits'].float().cpu().numpy())
                            utility.append(output['utility'].float().cpu().numpy())
                    indices = np.arange(cache.size, dtype=np.int64)
                    release = indices.copy()
                    metadata = args.data_root / (problem + split) / 'input_metadata.npz'
                    if metadata.exists():
                        with np.load(metadata, allow_pickle=False) as saved:
                            if 'source_index' in saved:
                                release = saved['source_index'].copy()
                    data = dict(logits=np.concatenate(scores), utility=np.concatenate(utility), indices=indices,
                        source_indices=indices.copy(), release_source_indices=release,
                        costs=cache.raw['costs'].copy(), winner=cache.raw['winner'].copy(),
                        pool=np.asarray(cache.pool), pool_ids=np.asarray(cache.pool_ids, dtype=np.int64),
                        nodes=np.asarray(cache.lengths).copy())
                    save_npz(shard, data)
                finally:
                    cache.close()
            count = manifest['splits'][split][problem]['rows']
            if (data['costs'].dtype != np.float64 or data['winner'].dtype != np.int64
                    or data['utility'].shape != data['costs'].shape or not np.isfinite(data['utility']).all()
                    or len(data['logits']) != count or not np.array_equal(data['indices'], np.arange(count))
                    or not np.array_equal(data['winner'], data['costs'].argmin(1))):
                raise ValueError(f'Incomplete/misaligned native R82 prediction archive: {problem}/{split}')
            raw = {key: data[key] for key in ('costs', 'winner', 'pool', 'pool_ids')}
            raw['pool'], raw['pool_ids'] = raw['pool'].tolist(), raw['pool_ids'].tolist()
            per_problem[problem], data['pred'] = fixed_sbs_metrics(data['logits'], raw, baselines[problem])
            arrays.update({problem + '__' + key: value for key, value in data.items()})
    finally:
        selector.train(was_training)
    save_npz(directory / 'rank000.npz', arrays)
    result = summarize(per_problem)
    result.update(seconds=time.monotonic()-started, tag=tag, successful_updates=step, sample_exposures=step*BATCH,
        identity=identity, prediction_archive=str((directory / 'rank000.npz').resolve()))
    from .r82_reporting import write_cmr_diagnostics
    diagnostic = write_cmr_diagnostics(directory / 'rank000.npz', args.root, split, directory)
    result['cmr_diagnostics'] = diagnostic
    write_json(directory / 'metrics.json', result)
    print('[R82 eval] ' + json.dumps(dict(split=split, tag=tag, **result['macro'])), flush=True)
    return result


def wandb_run(args, config):
    import wandb
    run = args.root / GROUP
    settings = dict(project='selector', entity='yjkds-southern-university-of-science-technology',
        name='R82_' + GROUP, id='r82-' + hashlib.sha256((str(args.root)+GROUP).encode()).hexdigest()[:16],
        config=config, dir=str(run), settings=wandb.Settings(init_timeout=30))
    mode = 'online' if args.wandb_mode == 'auto' else args.wandb_mode
    reason = None
    try:
        tracker = wandb.init(**settings, mode=mode, resume='allow' if mode == 'online' else None)
    except Exception as error:
        reason = f'{type(error).__name__}: {error}'
        wandb.finish(exit_code=1)
        mode = 'offline'
        tracker = wandb.init(**settings, mode=mode)
    write_json(run / 'wandb_receipt.json', dict(requested=args.wandb_mode, mode=mode,
        fallback_reason=reason, project=settings['project'], entity=settings['entity'],
        run_id=tracker.id, url=tracker.url if mode == 'online' else None, updated_utc=utc()))
    return tracker


def validate_resume(state, config):
    if state['args'] != config or state['state_hash'] != json_hash(config):
        raise ValueError('Resume must retain the exact R82B protocol hash')
    step = state['successful_updates']
    pending = state.get('pending_evaluation') is not None
    if (not 0 <= step <= MAX_UPDATES or step % EVAL_EVERY
            or step != (len(state['history']) + int(pending)) * EVAL_EVERY):
        raise ValueError('Resume only from a complete interval and explicit pending VAL')
    counts = Counter(state['updates_by_problem'])
    if counts != Counter(task_order()[:step]) or any(
            state['stream_draws'][p] != counts[p] * BATCH for p in PROBLEMS):
        raise ValueError('Resume counters/source cursors disagree with successful updates')
    if set(state.get('method_participation', {})) != set(PROBLEMS):
        raise ValueError('Resume requires every method participation counter')


def write_metrics_csv(root, history):
    rows, optimization = [], []
    for point in history:
        shared = {key: point[key] for key in ('evaluation_point', 'successful_updates', 'sample_exposures')}
        values = dict(shared, **point['optimization'])
        optimization.append(values)
        for split in ('val', 'train'):
            if point.get(split) is not None:
                rows.append(dict(shared, split=split, **point[split]['macro'],
                    **{'optimization_' + key: value for key, value in point['optimization'].items()}))
    for name, values in (('metrics.csv', rows), ('optimization.csv', optimization)):
        if not values:
            continue
        path = root / name
        temporary = path.with_name(f'.{name}.{os.getpid()}.tmp')
        with temporary.open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(dict.fromkeys(key for row in values for key in row)))
            writer.writeheader()
            writer.writerows(values)
        temporary.replace(path)


def train(args, device):
    from .context_matched_objective import CMR_GROUP_KEY, CMR_REGRET_KEY, CachedCMRObjective, utility_update
    root = args.root / GROUP
    if (root / 'training_complete.json').exists():
        return test(args, device)
    config, manifest = protocol(args)
    state_hash = json_hash(config)
    if args.state_hash is not None and args.state_hash != state_hash:
        raise ValueError(f'Expected state hash {args.state_hash}, actual {state_hash}')
    protocol_path, last = root / 'protocol.json', root / 'last.pt'
    if protocol_path.exists() and read_json(protocol_path) != config:
        raise ValueError('Existing R82 protocol differs; never overwrite it')
    if last.exists() and not args.resume:
        raise ValueError('Use --resume; existing R82 checkpoints will not be overwritten')
    write_json(protocol_path, config)
    status_update(args, 'initializing', 0, state_hash=state_hash)
    state = None
    if last.exists():
        selector, state = load_checkpoint(last, device)
        validate_resume(state, config)
        initialization = state['initialization']
    else:
        initial = torch.load(args.prepared_root / 'identicalinit_seed2.pt', map_location='cpu', weights_only=False)
        if (initial.get('initialization') != 'scratch' or initial.get('seed') != 2
                or initial['trainable_sha256'] != config['common_trainable_sha256']
                or parameter_hash(initial['trainable']) != initial['trainable_sha256']):
            raise ValueError('Only the locked R77 seed2 fresh initialization is admissible')
        selector = initialize_identical(initial, config['geometry_stats'], device)
        initialization = dict(experiment='R82', seed=2, initialization='scratch_shared_r77',
            trainable_sha256=parameter_hash(dict(selector.named_parameters())),
            initial_model_state_sha256=parameter_hash(selector.state_dict()),
            historical_checkpoint_loaded=False, geometry_fitted_on='R77 FULL TRAIN only',
            model_representation='ID + handcrafted34', projected_tokens_cached=False)
        write_json(root / 'initialization.json', initialization)
        del initial
    optimizer = torch.optim.AdamW(selector.parameters(), lr=1e-4, weight_decay=1e-4)
    scaler = torch.amp.GradScaler('cuda', enabled=device.type == 'cuda')
    objective = CachedCMRObjective(selector, cmr_weight=.10, ramp_updates=WARMUP)
    history, best, successful, retries_total, pending = [], None, 0, 0, None
    counts, cursors = Counter(), dict.fromkeys(PROBLEMS, 0)
    participation = {p: [dict(method=m['method'], solver_id=m['solver_id'],
        continuous_supported=m['continuous_supported'], batches_seen=0,
        participating_batches=0, pair_count=0, groups={})
        for m in config['train_support'][p]['methods']] for p in PROBLEMS}
    if state is not None:
        optimizer.load_state_dict(state['optimizer'])
        for values in optimizer.state.values():
            for key, value in values.items():
                if torch.is_tensor(value):
                    values[key] = value.cpu() if key == 'step' else value.to(device)
        scaler.load_state_dict(state['scaler'])
        history, best, successful = state['history'], state['best'], state['successful_updates']
        retries_total, counts = state['amp_retries'], Counter(state['updates_by_problem'])
        cursors, pending, participation = state['stream_draws'], state.get('pending_evaluation'), state['method_participation']
        restore_rank_rng(state['rng'], device)
    else:
        set_seed(2)
    del state
    streams = {p: ShuffleStream(FULL_ROWS, task_seed(2, p), cursors[p]) for p in PROBLEMS}
    caches, tracker = {}, None

    def checkpoint_state():
        return dict(model=selector.state_dict(), optimizer=optimizer.state_dict(), scaler=scaler.state_dict(),
            args=config, state_hash=state_hash, initialization=initialization, epoch=len(history),
            successful_updates=successful, updates_by_problem=dict(counts), sample_exposures=successful*BATCH,
            stream_draws={p: stream.draws for p, stream in streams.items()}, amp_retries=retries_total,
            rng=capture_rank_rng(device), history=history, best=best, pending_evaluation=pending,
            method_participation=participation)

    if not last.exists():
        save_checkpoint(last, checkpoint_state())
    try:
        tracker = wandb_run(args, config)
        saved_rng = torch.load(last, map_location='cpu', weights_only=False)['rng']
        restore_rank_rng(saved_rng, device)
        del saved_rng
        while successful < MAX_UPDATES or pending is not None:
            if pending is None:
                if not caches:
                    for p in PROBLEMS:
                        source = open_cache(args.cache_root, 'train', p, manifest, args.data_root)
                        expected = config['train_support'][p]
                        caches[p] = SupervisedCache(source, args.root / expected['path'], expected)
                end, first = min(MAX_UPDATES, successful + EVAL_EVERY), successful
                order = task_order()[successful:end]
                requests = [(p, streams[p].take(BATCH), BATCH) for p in order]
                interval_start, totals = time.monotonic(), Counter()
                selector.train()
                with PinnedPrefetch(caches, requests, device) as batches:
                    for problem, n_rank, n_global, batch in batches:
                        if n_rank != BATCH or n_global != BATCH:
                            raise ValueError('CMR must see one complete1024 real-row single-problem batch')
                        lr, weight = learning_rate(successful+1), ramp_weight(successful+1)
                        for group in optimizer.param_groups:
                            group['lr'] = lr
                        groups, regrets = batch.pop(CMR_GROUP_KEY), batch.pop(CMR_REGRET_KEY)
                        parts, retries = utility_update(objective, optimizer, scaler, batch,
                            config['microbatch_caps'][problem], consistency_weight=weight,
                            successful_update=successful+1, groups=groups, regrets=regrets)
                        successful += 1
                        counts[problem] += 1
                        retries_total += retries
                        for key, value in parts.items():
                            value = float(value)
                            if not math.isfinite(value):
                                raise ValueError(f'Nonfinite successful optimization metric: {key}')
                            totals[key] += value
                        update_participation(participation[problem], groups, regrets)
                        if successful == 1 or successful % args.log_every == 0 or successful == end:
                            done, elapsed = successful-first, time.monotonic()-interval_start
                            info = dict(successful_updates=successful, max_updates=MAX_UPDATES, problem=problem,
                                sample_exposures=successful*BATCH, lr=lr, consistency_weight=weight,
                                lambda_cmr=cmr_weight(successful), amp_retries=retries_total,
                                updates_by_problem=dict(counts), seconds_per_update=elapsed/done,
                                samples_per_second=done*BATCH/elapsed,
                                **{key: value/done for key, value in totals.items()})
                            val_seconds = float(np.mean([row['val']['seconds'] for row in history])) if history else 0.
                            eta = (MAX_UPDATES-successful)*elapsed/done + (MAX_UPDATES//EVAL_EVERY-len(history))*val_seconds
                            status_update(args, 'training', successful, eta_seconds=eta,
                                **{key: value for key, value in info.items() if key != 'successful_updates'})
                            with (root / 'updates.jsonl').open('a') as stream:
                                stream.write(json.dumps(dict(updated_utc=utc(), **info), allow_nan=False)+'\n')
                            write_json(root / 'method_participation.json', participation)
                            print('[R82 train] ' + json.dumps(info), flush=True)
                            if tracker is not None:
                                tracker.log({**{'optimization/'+key: value for key, value in parts.items()},
                                    'successful_updates': successful, 'sample_exposures': successful*BATCH,
                                    'lr': lr, 'lambda_cmr': cmr_weight(successful)}, step=successful)
                pending = {key: value/(successful-first) for key, value in totals.items()}
                pending.update(lr=lr, consistency_weight=weight, lambda_cmr=cmr_weight(successful),
                    amp_retries=retries_total, samples=(successful-first)*BATCH,
                    seconds=time.monotonic()-interval_start)
                save_checkpoint(last, checkpoint_state())
            point, tag = successful//EVAL_EVERY, f'update{successful:07d}'
            validation = evaluate(selector, args, manifest, config['baselines'], 'val', tag, successful, state_hash)
            cost = validation['macro']['macro_vs_sbs_pct']
            if not math.isfinite(cost):
                raise ValueError('Primary full VAL macro_vs_sbs_pct must be finite')
            training = evaluate(selector, args, manifest, config['baselines'], 'train', tag, successful,
                state_hash) if point % 5 == 0 else None
            row = dict(epoch=point, evaluation_point=point, successful_updates=successful,
                sample_exposures=successful*BATCH, updates_by_problem=dict(counts), evaluation_tag=tag,
                optimization=pending, val=validation, train=training)
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
                tracker.log({**{'val/'+key: value for key, value in validation['macro'].items()},
                    'successful_updates': successful}, step=successful)
        if successful != MAX_UPDATES or any(counts[p] != MAX_UPDATES//len(PROBLEMS) for p in PROBLEMS):
            raise ValueError('The total/per-task successful-update budget was not completed')
        for cache in caches.values():
            cache.close()
        caches.clear()
        del objective, optimizer, scaler, selector
        selector, selected = load_checkpoint(root / 'best.pt', device)
        if selected['args'] != config or selected['successful_updates'] != best['successful_updates']:
            raise ValueError('Unique VAL-best checkpoint disagrees with training history')
        del selected
        lock = lock_best(root, best, config, state_hash)
        source_val = root / 'evaluation/val' / best['evaluation_tag']
        selected_val = root / 'evaluation/val/selected_best'
        selected_val.mkdir(parents=True, exist_ok=True)
        for name in ('rank000.npz', 'metrics.json', 'identity.json'):
            target = selected_val / name
            if target.exists():
                if file_hash(target) != file_hash(source_val / name):
                    raise ValueError('Selected VAL copy changed after unique lock')
            else:
                shutil.copyfile(source_val / name, target)
        if best['train'] is None:
            best = dict(best, train=evaluate(selector, args, manifest, config['baselines'], 'train', 'selected_best',
                best['successful_updates'], state_hash, lock['checkpoint_sha256']))
        for row in history:
            if row['successful_updates'] == best['successful_updates']:
                row['train'] = best['train']
        write_json(root / 'history.json', history)
        write_json(root / 'best_eval.json', best)
        write_metrics_csv(root, history)
        write_json(root / 'method_participation.json', participation)
        write_json(root / 'result.json', dict(best=best, final=history[-1], successful_updates=successful,
            updates_by_problem=dict(counts), amp_retries=retries_total, state_hash=state_hash,
            stop_reason='fixed successful-update budget; no early stopping', test_read=False))
        write_json(root / 'final_metrics.json', dict(epoch=best['epoch'], selected_update=best['successful_updates'],
            checkpoint_sha256=lock['checkpoint_sha256'], state_hash=state_hash, train=best['train'], val=best['val'], test_read=False))
        write_json(root / 'training_complete.json', dict(experiment='R82', group=GROUP, completed_utc=utc(),
            successful_updates=successful, sample_exposures=successful*BATCH, updates_by_problem=dict(counts),
            selected_best=lock, state_hash=state_hash, next_stage='locked full TEST', test_read=False))
        status_update(args, 'training_complete', successful, selected_update=lock['selected_update'])
        del selector
    finally:
        for cache in caches.values():
            cache.close()
        if tracker is not None:
            tracker.finish()
    return test(args, device)


def lock_best(root, best, config, state_hash):
    record = dict(experiment='R82', alias=ALIAS, group=GROUP, arm=GROUP,
        checkpoint=str((root / 'best.pt').resolve()), checkpoint_sha256=file_hash(root / 'best.pt'), state_hash=state_hash,
        epoch=best['epoch'], selected_update=best['successful_updates'], rule=config['checkpoint_rule'],
        selected_without_test=True, per_problem_assembly=False, val=best['val']['macro'],
        inference_rule=config['inference_rule'], winner_rule=WINNER_RULE)
    path = root / 'final_test_lock.json'
    if path.exists():
        saved = read_json(path)
        if any(saved.get(key) != value for key, value in record.items()):
            raise PermissionError('Unique R82 checkpoint/inference lock cannot change')
        return saved
    record['locked_utc'] = utc()
    write_json(path, record)
    return record


def authorize_test(args):
    root = args.root / GROUP
    complete = read_json(root / 'training_complete.json')
    config, lock = read_json(root / 'protocol.json'), read_json(root / 'final_test_lock.json')
    if (complete.get('experiment') != 'R82' or complete.get('successful_updates') != MAX_UPDATES
            or complete.get('state_hash') != lock.get('state_hash')
            or config.get('experiment') != 'R82' or config.get('group') != GROUP
            or lock.get('experiment') != 'R82' or lock.get('group') != GROUP
            or lock.get('selected_without_test') is not True or lock.get('per_problem_assembly') is not False
            or lock.get('state_hash') != json_hash(config) or config.get('max_successful_updates') != MAX_UPDATES
            or not 0 < lock.get('selected_update', 0) <= MAX_UPDATES
            or lock.get('rule') != config.get('checkpoint_rule') or lock.get('inference_rule') != config.get('inference_rule')
            or Path(lock['checkpoint']).resolve() != (root / 'best.pt').resolve()
            or file_hash(root / 'best.pt') != lock.get('checkpoint_sha256')
            or Path(config['data_root']).resolve() != args.data_root.resolve()):
        raise PermissionError('TEST requires completed R82B and its unchanged unique VAL lock')
    if args.state_hash is not None and args.state_hash != lock['state_hash']:
        raise ValueError('Expected state hash differs from immutable TEST lock')
    return lock, config


def test(args, device):
    root = args.root / GROUP
    lock, config = authorize_test(args)
    result_path = root / 'test_results.json'
    if result_path.exists():
        saved = read_json(result_path)
        if (saved.get('complete') is not True or saved.get('checkpoint_sha256') != lock['checkpoint_sha256']
                or saved.get('state_hash') != lock['state_hash'] or saved.get('selected_update') != lock['selected_update']):
            raise PermissionError('Saved TEST result differs from unique R82 lock')
        result = saved['result']
    else:
        if args.cache_root is not None and str(args.cache_root) != config['cache_root']:
            raise ValueError('TEST must use the frozen source cache')
        args.cache_root, args.gpu_type = Path(config['cache_root']), config['gpu_type']
        args.eval_batch_size, args.large_micro_cap = config['eval_batch_size'], config['microbatch_caps']['TSP']
        selector, state = load_checkpoint(root / 'best.pt', device)
        if state['args'] != config or state['successful_updates'] != lock['selected_update']:
            raise PermissionError('TEST model does not match frozen selection')
        del state
        started_path = root / 'test_started.json'
        if started_path.exists() and read_json(started_path).get('checkpoint_sha256') != lock['checkpoint_sha256']:
            raise PermissionError('Earlier TEST attempt used another checkpoint')
        write_json(started_path, dict(experiment='R82', started_utc=utc(), checkpoint_sha256=lock['checkpoint_sha256'],
            inference_rule=config['inference_rule'], resumable_same_checkpoint=True))
        manifest = dict(splits=config['data_manifest'])
        result = evaluate(selector, args, manifest, config['baselines'], 'test', 'selected_best',
            lock['selected_update'], lock['state_hash'], lock['checkpoint_sha256'])
        if file_hash(root / 'best.pt') != lock['checkpoint_sha256']:
            raise PermissionError('Locked checkpoint changed during TEST')
        write_json(result_path, dict(complete=True, completed_utc=utc(), result=result, epoch=lock['epoch'],
            selected_update=lock['selected_update'], checkpoint_sha256=lock['checkpoint_sha256'], state_hash=lock['state_hash']))
        del selector
    final = read_json(root / 'final_metrics.json')
    final.update(test=result, test_read=True, complete=True, checkpoint_sha256=lock['checkpoint_sha256'])
    write_json(root / 'final_metrics.json', final)
    write_json(root / 'test_complete.json', dict(experiment='R82', full_tests=1, completed_utc=utc(),
        checkpoint_sha256=lock['checkpoint_sha256'], state_hash=lock['state_hash'], saved_predictions_reusable=True))
    status_update(args, 'test_complete', MAX_UPDATES, selected_update=lock['selected_update'], test=result['macro'], eta_seconds=0.)
    return result


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('command', choices=('prepare', 'describe', 'train', 'test', 'status'))
    result.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    result.add_argument('--group', '--arm', choices=(GROUP,), default=GROUP)
    result.add_argument('--prepared-root', '--r77-root', type=Path, default=R77_ROOT)
    result.add_argument('--data-root', type=Path, default=DATA_ROOT)
    result.add_argument('--cache-root', type=Path)
    result.add_argument('--device', default='cuda:0')
    result.add_argument('--gpu-type', choices=('4090', '3090'), default='4090')
    result.add_argument('--large-micro-cap', type=int)
    result.add_argument('--eval-batch-size', type=int, default=128)
    result.add_argument('--log-every', type=int, default=18)
    result.add_argument('--wandb-mode', choices=('auto', 'online', 'offline'), default='auto')
    result.add_argument('--resume', action='store_true')
    result.add_argument('--state-hash', '--expected-state-hash')
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    for name in ('root', 'prepared_root', 'data_root', 'cache_root'):
        if getattr(args, name) is not None:
            setattr(args, name, getattr(args, name).expanduser().resolve())
    if args.eval_batch_size < 1 or args.log_every < 1:
        raise ValueError('Evaluation/log intervals must be positive')
    if args.command == 'status':
        print(json.dumps(read_json(args.root / GROUP / 'status.json'), indent=2))
        return
    if args.command == 'describe':
        description = dict(experiment='R82', alias=ALIAS, group=GROUP, root=str(args.root),
            max_successful_updates=MAX_UPDATES, batch_size=BATCH, warmup_updates=WARMUP,
            eval_every_updates=EVAL_EVERY, total_sample_exposures=BATCH * MAX_UPDATES,
            microbatch_caps=micro_caps(args.gpu_type, args.large_micro_cap),
            preparation_receipt=str(prepared_path(args)), matched_r82a='not_run', training_started=False)
        if prepared_path(args).exists() and (V4 / 'context_matched_objective.py').exists():
            config, _ = protocol(args)
            description['state_hash'] = json_hash(config)
        print(json.dumps(description, indent=2))
        return
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    if args.command == 'prepare':
        directory = args.root / 'prepared'
        directory.mkdir(parents=True, exist_ok=True)
        with (directory / 'prepare.lock').open('a') as claim:
            fcntl.flock(claim, fcntl.LOCK_EX | fcntl.LOCK_NB)
            prepare(args)
        return
    run = args.root / GROUP
    run.mkdir(parents=True, exist_ok=True)
    with (run / 'runner.lock').open('a') as claim:
        try:
            fcntl.flock(claim, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit('R82B already running; no duplicate process') from None
        try:
            device = torch.device(args.device)
            if device.type != 'cuda':
                raise ValueError('Formal training/TEST needs an externally assigned CUDA device; CPU is for fixtures only')
            configure_torch()
            if args.command == 'train':
                train(args, device)
            else:
                test(args, device)
        except BaseException as error:
            details = dict(type=type(error).__name__, message=str(error), updated_utc=utc(),
                traceback=traceback.format_exc(), pid=os.getpid(), host=socket.gethostname())
            write_json(run / 'error.json', details)
            status_update(args, 'failed', error=details)
            raise


if __name__ == '__main__':
    main()
