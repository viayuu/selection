"""R122/D2: full18 hard-decision constraints on the unchanged R45A selector.

``prepare`` is CPU-only and reads saved R84A predictions, never a model forward.
``train`` and ``test`` are explicit, separate, single-device commands. Historical
modules are reused read-only; the private R84 engine preserves their metrics.
"""

from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import importlib.util
import json
import math
import shutil
import sys
import time
import traceback
from collections import Counter
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch

from code.unified_selector.registry import GLOBAL_SOLVERS, PROBLEMS
from . import r84_experiment as engine
from .multitask_probe import task_seed
from .pairwise_selector import ScoreDifferenceSelector
from .r76_cache import ScenarioCache, _row_indices
from .r76_experiment import (
    PinnedPrefetch, Runtime, capture_rank_rng, file_hash, open_cache, read_json,
    restore_rank_rng, save_checkpoint, slice_batch, utc, write_json,
)
from .r77_data import parameter_hash
from .r77_experiment import ShuffleStream, task_order
from .r84_prepare import DEFAULT_DATA_ROOT, DEFAULT_PREPARED_ROOT, SPLIT_ROWS
from .r122_objective import HardDecisionController, HardDecisionObjective, QUERY_INPUTS
from .train import configure_torch, set_seed


V4 = Path(__file__).resolve().parent
DEFAULT_ROOT = V4 / 'runs/R122_hard_decision_constraints'
LOCAL_CACHE_ROOT = Path('/tmp/shiys_R84_new90k_cache')
FROZEN_TEST_CACHE_ROOT = V4 / 'runs/R120_multidomain_selector/existing_test_cache'
PREPARATION_VERSION = 'v2'
REFERENCE_ROOT = V4 / 'runs/R84_new90k_attention/A_baseline'
TARGET_LIST = REFERENCE_ROOT.parent / 'comparison_artifacts/target_list.json'
REFERENCE_SHA = '23fb9a5cf621045138fcdcfe5e6613a1bfd061c8d4dfeca108db3548dbf8031f'
FRESH_SHA = 'dad87d8598f7560e6f170ef048ba1bf515edeb03cef7ec42e16839a36af4ef93'
REFERENCE_UPDATE = 37440
ARMS = ('model',)
BATCH_SIZE, MAX_UPDATES, WARMUP_UPDATES, EVAL_EVERY = 1024, 14400, 1080, 720
PEAK_LR, END_LR = 1e-4, 1e-6
MODEL_FACTORY = 'code.V4.r122_experiment.load_checkpoint'
WINNER_RULE = engine.WINNER_RULE
INFERENCE_RULE = ('eval mode, original R77 maximin logits, '
                  'stable ascending-global-solver-ID tie break')
CHECKPOINT_RULE = (
    'full18 VAL only: feasible iff hard macro actual regret and unconditional '
    'fixed36 missed frequency <= separate saved R84A VAL references; among '
    'feasible maximize native Top1, then minimize actual regret, then earliest; '
    'if none feasible minimize sum of positive violations / max(reference,1e-12), '
    'then maximize native Top1, minimize actual regret, earliest')
CONTROLLER = dict(risk_unit=.01, dual_lr=.05, dual_cap=20., ema_decay=.9)
LOSS = dict(classification_proxy='native winner CE / log(2)',
    regret_proxy='sum full-pool FP64 relative regret * softplus(utility - max_other) / log(2)',
    fixed_miss_proxy='fixed36 native-winner indicator * classification_proxy',
    hard_feedback='one post-successful-update commit; equal-task macro per full18 cycle',
    forward_passes=1, rdrop=False, comparison=False)
SOURCE_FILES = (
    'r122_experiment.py', 'r122_objective.py', 'r84_experiment.py', 'r84_prepare.py',
    'r76_cache.py', 'r76_experiment.py', 'r77_experiment.py', 'r77_data.py',
    'r80_experiment.py', 'pairwise_selector.py', 'pairwise_objective.py',
    'solver_code_encoder.py', 'dual_stream.py', 'V4Model.py', 'solver_features.py',
    'r40_experiment.py', 'r43_experiment.py', '../unified_selector/registry.py',
)


def learning_rate(update):
    if not 1 <= update <= MAX_UPDATES:
        raise ValueError('Learning rate uses one-based successful updates')
    if update <= WARMUP_UPDATES:
        return PEAK_LR * update / WARMUP_UPDATES
    phase = (update - WARMUP_UPDATES) / (MAX_UPDATES - WARMUP_UPDATES)
    return END_LR + .5 * (PEAK_LR - END_LR) * (1. + math.cos(math.pi * phase))


def micro_caps(gpu_type='4090', large_cap=None):
    if gpu_type not in ('3090', '4090') or large_cap not in (None, 256):
        raise ValueError('R122 fixes microbatch caps; no automatic OOM shrink')
    return {p: 256 if p in ('TSP', 'CVRP') else BATCH_SIZE for p in PROBLEMS}


def query_inputs(batch):
    return {key: batch[key] for key in QUERY_INPUTS if key in batch}


def immutable_json(path, value):
    path = Path(path)
    if path.exists():
        if read_json(path) != value:
            raise PermissionError(f'Immutable R122 artifact differs; preserve it: {path}')
    else:
        write_json(path, value)
    return value


def receipt(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=file_hash(path))


def verify_receipts(records):
    for item in records:
        if file_hash(Path(item['path'])) != item['sha256']:
            raise PermissionError('Locked R122 source changed: ' + item['path'])


def fixed_targets(record, pools, *, expected_count=36):
    rows = record.get('targets', [])
    pairs = [(row['problem'], row['method']) for row in rows]
    if len(pairs) != expected_count or len(set(pairs)) != expected_count:
        raise ValueError('Use the unique historical fixed36, not a dynamic zero group')
    if any(p not in pools or method not in pools[p] for p, method in pairs):
        raise ValueError('Fixed targets must name legal native deployment names')
    return {p: tuple(method for task, method in pairs if task == p) for p in pools}


def hard_statistics(data, targets):
    """Pure CPU parsing with native labels and stable deployment-ID score ties."""
    scores, costs, winner = (np.asarray(data[k]) for k in ('logits', 'costs', 'winner'))
    pool, ids = np.asarray(data['pool']), np.asarray(data['pool_ids'])
    n = len(winner)
    if (costs.dtype != np.float64 or winner.dtype != np.int64 or n == 0
            or costs.ndim != 2 or scores.shape != costs.shape
            or winner.shape != (costs.shape[0],) or pool.shape != (costs.shape[1],)
            or ids.shape != pool.shape or ids.dtype != np.int64
            or len(set(pool.tolist())) != len(pool) or len(np.unique(ids)) != len(ids)
            or not np.isfinite(costs).all() or not np.isfinite(scores).all()
            or (costs <= 0).any() or not np.array_equal(winner, costs.argmin(1))):
        raise ValueError('Require full native FP64 costs, int64 winner and legal aligned scores')
    order = np.argsort(ids, kind='stable')
    prediction = order[np.argmax(scores[:, order], axis=1)]
    fixed = np.isin(pool[winner], list(targets))
    oracle = costs.min(1)
    regret = (costs[np.arange(n), prediction] - oracle) / oracle
    return dict(n=n, hard_regret=float(regret.mean(dtype=np.float64)),
        hard_fixed_miss=float(np.mean(fixed & (prediction != winner))),
        fixed_population=float(fixed.mean()), native_top1=float(np.mean(prediction == winner)),
        fixed_count=int(fixed.sum()), fixed_missed_count=int((fixed & (prediction != winner)).sum()))


def macro_hard(per_problem):
    if not per_problem:
        raise ValueError('An equal-task macro needs tasks')
    return {key: float(np.mean([row[key] for row in per_problem.values()]))
            for key in ('hard_regret', 'hard_fixed_miss', 'fixed_population', 'native_top1')}


def selection_key(result, val_reference, update):
    point = result['hard_decision']['macro']
    risk, miss = point['hard_regret'], point['hard_fixed_miss']
    top1 = result['macro']['macro_top1']
    thresholds = (val_reference['hard_regret'], val_reference['hard_fixed_miss'])
    if (not all(math.isfinite(x) for x in (risk, miss, top1, *thresholds))
            or min(risk, miss, *thresholds) < 0 or not 0 <= top1 <= 1
            or not math.isclose(point['native_top1'], top1, rel_tol=0., abs_tol=1e-12)):
        raise ValueError('VAL selection requires finite unchanged native macro metrics')
    violation = sum(max(value - budget, 0.) / max(budget, 1e-12)
                    for value, budget in zip((risk, miss), thresholds))
    feasible = risk <= thresholds[0] and miss <= thresholds[1]
    return (0 if feasible else 1, 0. if feasible else violation, -top1, risk, update)


class NativeConstraintCache(ScenarioCache):
    """Keep base input gathering, restore FP64 supervision before any transfer."""

    def __init__(self, directory, *, targets):
        super().__init__(directory)
        self.fixed_methods = tuple(targets[self.problem])

    def batch(self, indices, device='cpu'):
        rows = _row_indices(indices, self.size)
        batch = super().batch(indices, device='cpu')
        batch['costs'] = torch.from_numpy(np.array(self.raw['costs'][rows], copy=True))
        native_names = np.asarray(self.pool)[self.raw['winner'][rows]]
        batch['fixed_target'] = torch.from_numpy(np.isin(native_names, self.fixed_methods))
        return {key: value.to(device) if torch.is_tensor(value) else value
                for key, value in batch.items()}


def validate_cache_copy(directory, expected, *, label_reference):
    """Transport may change location, not metadata, typed arrays or FP64 labels."""
    directory, label_reference = Path(directory), Path(label_reference)
    if read_json(directory / 'metadata.json') != expected:
        raise ValueError('Resource cache must retain original metadata/source identity: ' + str(directory))
    for key, spec in expected['arrays'].items():
        path = directory / spec['file']
        array = np.load(path, mmap_mode='r', allow_pickle=False)
        try:
            if (list(array.shape) != spec['shape'] or str(array.dtype) != spec['dtype']
                    or path.stat().st_size != array.offset + array.nbytes):
                raise ValueError('Resource cache typed array changed: ' + str(path))
        finally:
            array._mmap.close()
        if key in ('costs', 'winner') and path.resolve() != (label_reference / spec['file']).resolve():
            if file_hash(path) != file_hash(label_reference / spec['file']):
                raise ValueError('Resource copy changed frozen native costs/winner: ' + str(path))


def source_context(args):
    ready, manifest, geometry, baselines = engine.prepared_config(args)
    if (ready['trainable_sha256'] != FRESH_SHA or manifest.get('complete') is not True
            or manifest.get('scenario_solver_aliases') != {'reld_mt': 'RELD_MOEL'}):
        raise ValueError('Reuse the complete frozen R84 natural cache and seed2 initialization')
    source_protocol = read_json(REFERENCE_ROOT / 'protocol.json')
    if (source_protocol.get('experiment') != 'R84' or source_protocol.get('arm') != 'A_baseline'
            or source_protocol.get('seed') != 2 or source_protocol.get('rows_per_problem') != SPLIT_ROWS
            or source_protocol.get('selection_head') != 'original_score_mlp_maximin'
            or source_protocol.get('winner_rule') != WINNER_RULE
            or source_protocol.get('common_trainable_sha256') != FRESH_SHA
            or Path(source_protocol['data_root']).resolve() != args.data_root.resolve()
            or source_protocol.get('model_params') != ready['model_params']):
        raise ValueError('R84A source protocol/data/model differs from the frozen reference')
    pools = source_protocol['candidate_pools']
    if set(pools) != set(PROBLEMS):
        raise ValueError('R122 is the complete full18 legal pool')
    for split in ('train', 'val'):
        for p in PROBLEMS:
            record = manifest['splits'][split][p]
            if (record.get('complete') is not True or record['rows'] != SPLIT_ROWS[split]
                    or record['pool'] != pools[p]
                    or record['pool_ids'] != source_protocol['candidate_pool_ids'][p]
                    or 'RELD_MTL' in record['pool'] or 'RELD_MOEL' in record['pool']
                    or ('B' in p and any(m in record['pool'] for m in
                                        ('RouteFinder', 'MoSES_RF', 'MoSES_CaDA')))):
                raise ValueError('Frozen native candidate pools/rows differ: ' + p + '/' + split)
            validate_cache_copy(args.cache_root / split / p, record,
                                label_reference=Path(ready['cache_source']) / split / p)
    return ready, manifest, geometry, baselines, source_protocol


def reference_identity(source_protocol):
    lock = read_json(REFERENCE_ROOT / 'final_test_lock.json')
    checkpoint_path = REFERENCE_ROOT / 'best.pt'
    if (lock.get('checkpoint_sha256') != REFERENCE_SHA or file_hash(checkpoint_path) != REFERENCE_SHA
            or lock.get('selected_update') != REFERENCE_UPDATE
            or lock.get('experiment') != 'R84' or lock.get('arm') != 'A_baseline'
            or lock.get('state_hash') != engine.json_hash(source_protocol)
            or Path(lock['checkpoint']).resolve() != checkpoint_path.resolve()
            or lock.get('selected_without_test') is not True or lock.get('per_problem_assembly') is not False):
        raise ValueError('The immutable selected R84A reference lock differs')
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    if (checkpoint['args'] != source_protocol or checkpoint['state_hash'] != lock['state_hash']
            or checkpoint['successful_updates'] != REFERENCE_UPDATE):
        raise ValueError('R84A reference checkpoint metadata differs from its lock')
    return lock, parameter_hash(checkpoint['model'])


def reference_split(args, manifest, targets, split, lock, model_hash):
    directory = REFERENCE_ROOT / 'evaluation' / split / 'selected_best'
    identity = read_json(directory / 'identity.json')
    summary = read_json(directory / 'metrics.json')
    if (identity != summary.get('identity') or identity.get('experiment') != 'R84'
            or identity.get('arm') != 'A_baseline' or identity.get('split') != split
            or identity.get('successful_updates') != REFERENCE_UPDATE
            or identity.get('state_hash') != lock['state_hash']
            or identity.get('model_state_sha256') != model_hash
            or identity.get('winner_rule') != WINNER_RULE
            or identity.get('checkpoint_sha256') not in (None, REFERENCE_SHA)
            or (split == 'train' and identity.get('checkpoint_sha256') != REFERENCE_SHA)
            or set(summary['per_problem']) != set(PROBLEMS)):
        raise ValueError('Reference capture is not bound to the selected R84A model: ' + split)
    paths = [receipt(directory / name) for name in ('identity.json', 'metrics.json', 'rank000.npz')]
    shards = directory / 'by_problem'
    if not shards.is_dir():
        shards = REFERENCE_ROOT / 'evaluation' / split / identity['tag'] / 'by_problem'
    per = {}
    for p in PROBLEMS:
        path = shards / (p + '.npz')
        with np.load(path, allow_pickle=False) as saved:
            data = {key: saved[key] for key in saved.files}
        record = manifest['splits'][split][p]
        cache_directory = args.cache_root / split / p
        if (data['pool'].tolist() != record['pool'] or data['pool_ids'].tolist() != record['pool_ids']
                or len(data['winner']) != SPLIT_ROWS[split]
                or not np.array_equal(data['indices'], np.arange(SPLIT_ROWS[split]))
                or not np.array_equal(data['source_indices'], data['indices'])):
            raise ValueError('Reference capture pool/order/rows differ: ' + p + '/' + split)
        for key in ('costs', 'winner'):
            label_path = cache_directory / (key + '.npy')
            native = np.load(label_path, mmap_mode='r', allow_pickle=False)
            try:
                if native.dtype != data[key].dtype or not np.array_equal(native, data[key]):
                    raise ValueError('Saved reference does not use original native labels: ' + p)
            finally:
                native._mmap.close()
            paths.append(receipt(label_path))
        metadata_path = args.data_root / (p + split) / 'input_metadata.npz'
        with np.load(metadata_path, allow_pickle=False) as metadata:
            if not np.array_equal(data['release_source_indices'], metadata['source_index']):
                raise ValueError('Reference source rows differ: ' + p + '/' + split)
        paths.extend((receipt(path), receipt(cache_directory / 'metadata.json'), receipt(metadata_path)))
        point = hard_statistics(data, targets[p])
        reported = summary['per_problem'][p]
        if (reported['n'] != point['n']
                or not math.isclose(reported['top1'], point['native_top1'], abs_tol=1e-12, rel_tol=0.)
                or not math.isclose(reported['actual_regret_pct'] / 100., point['hard_regret'],
                                    abs_tol=1e-12, rel_tol=0.)):
            raise ValueError('Reference shard and saved original metrics disagree: ' + p + '/' + split)
        per[p] = point
    return dict(source_split=split, use='dual TRAIN budgets' if split == 'train' else 'VAL selection only; never dual feedback',
                per_problem=per, macro=macro_hard(per), identity=identity), paths


def prepare(args):
    if torch.device(args.device).type != 'cpu':
        raise ValueError('prepare is CPU-only; pass --device cpu')
    args.preparation_root.mkdir(parents=True, exist_ok=True)
    with (args.preparation_root / 'prepare.lock').open('a') as claim:
        fcntl.flock(claim, fcntl.LOCK_EX | fcntl.LOCK_NB)
        ready, manifest, geometry, _, source_protocol = source_context(args)
        lock, model_hash = reference_identity(source_protocol)
        target_record = read_json(TARGET_LIST)
        if (target_record['identity']['baseline_checkpoint_sha256'] != REFERENCE_SHA
                or target_record['identity']['baseline_state_hash'] != lock['state_hash']
                or Path(target_record['identity']['data_root']).resolve() != args.data_root.resolve()
                or target_record['identity']['baseline_val_prediction_sha256'] != file_hash(
                    REFERENCE_ROOT / 'evaluation/val/selected_best/rank000.npz')):
            raise ValueError('The historical fixed36 list belongs to another baseline/data release')
        targets = fixed_targets(target_record, source_protocol['candidate_pools'])
        frozen_args = copy.copy(args)
        frozen_args.cache_root = Path(ready['cache_source']).resolve()
        train_reference, train_paths = reference_split(frozen_args, manifest, targets, 'train', lock, model_hash)
        val_reference, val_paths = reference_split(frozen_args, manifest, targets, 'val', lock, model_hash)
        initial = torch.load(args.prepared_root / 'identicalinit_seed2.pt', map_location='cpu', weights_only=False)
        model, params, initialization = engine.initialize_shared(initial, geometry, 'A_baseline', 'cpu')
        if (type(model) is not ScoreDifferenceSelector or params != ready['model_params']
                or parameter_hash(dict(model.named_parameters())) != FRESH_SHA):
            raise ValueError('R122 requires unchanged original R45A fresh seed2 tensors')
        paths = [receipt(args.prepared_root / name) for name in
                 ('prepared.json', 'source_cache_manifest.json', *ready['artifact_sha256'])]
        paths.extend(receipt(path) for path in (
            args.data_root / 'scenario_manifest.json', REFERENCE_ROOT / 'protocol.json',
            REFERENCE_ROOT / 'final_test_lock.json', REFERENCE_ROOT / 'best.pt', TARGET_LIST))
        paths.extend(receipt(V4 / name) for name in SOURCE_FILES)
        paths.extend(train_paths + val_paths)
        # A path is locked once even when an identity is cited by several roles.
        paths = list({item['path']: item for item in paths}.values())
        reference = dict(schema_version=1, experiment='R122', arm='model', seed=2,
            data_root=str(args.data_root), prepared_root=str(args.prepared_root), cache_root=str(frozen_args.cache_root),
            resource_cache_policy='location-only copy; original metadata/array specs/source identity and exact native label hashes',
            rows_per_problem=dict(SPLIT_ROWS), winner_rule=WINNER_RULE,
            checkpoint_reference=lock, reference_model_state_sha256=model_hash,
            fixed_targets=target_record['targets'], fixed_target_file=receipt(TARGET_LIST),
            candidate_pools=source_protocol['candidate_pools'],
            candidate_pool_ids=source_protocol['candidate_pool_ids'],
            train=train_reference, val_selection=val_reference,
            controller=dict(CONTROLLER), checkpoint_rule=CHECKPOINT_RULE,
            initialization=initialization, model_params=params,
            trainable_parameter_count=sum(p.numel() for p in model.parameters()),
            evidence='CPU-only saved-prediction fitting; no model forward, optimizer, solver, GPU or TEST labels',
            receipts=paths)
        immutable_json(args.preparation_root / 'constraint_reference.json', reference)
        record = dict(experiment='R122', locked=True, schema_version=1,
            constraint_reference=receipt(args.preparation_root / 'constraint_reference.json'),
            cpu_only=True, training_started=False, test_read=False)
        immutable_json(args.preparation_root / 'prepared.json', record)
    result = dict(preparation=str(args.preparation_root), **record,
        train_budgets=train_reference['macro'], val_selection_thresholds=val_reference['macro'])
    print(json.dumps(result, indent=2), flush=True)
    return result


def load_preparation(args):
    record = read_json(args.preparation_root / 'prepared.json')
    expected = args.preparation_root / 'constraint_reference.json'
    if (record.get('experiment') != 'R122' or record.get('locked') is not True
            or Path(record['constraint_reference']['path']).resolve() != expected.resolve()):
        raise ValueError('Run the CPU-only R122 prepare command first')
    verify_receipts([record['constraint_reference']])
    reference = read_json(expected)
    verify_receipts(reference['receipts'])
    ready, manifest, geometry, baselines, _ = source_context(args)
    if (reference['data_root'] != str(args.data_root) or reference['prepared_root'] != str(args.prepared_root)
            or reference['controller'] != CONTROLLER
            or reference['checkpoint_rule'] != CHECKPOINT_RULE):
        raise ValueError('R122 preparation paths/rules differ from the locked CPU reference')
    return reference, record, ready, manifest, geometry, baselines


def make_controller(reference):
    train = reference['train']
    values = train['macro']
    return HardDecisionController(PROBLEMS, train_regret_budget=values['hard_regret'],
        train_fixed_miss_budget=values['hard_fixed_miss'], train_fixed_population=values['fixed_population'],
        budget_source_split=train['source_split'], **CONTROLLER)


def make_model(params, arm='model', state_dict=None):
    if arm != 'model':
        raise ValueError('R122 has exactly one group: model')
    model = engine.make_model(params, 'A_baseline', state_dict=state_dict)
    if type(model) is not ScoreDifferenceSelector:
        raise ValueError('R122 must not change the original R45A ScoreDifferenceSelector')
    return model


def protocol(args, reference, prepared, ready, geometry, baselines, params):
    order = task_order(MAX_UPDATES)
    config = {}
    config.update(experiment='R122', schema_version=1, arm='model', seed=2,
        initialization='scratch_shared_r77', model_params=params, model_factory=MODEL_FACTORY,
        solver_names=list(GLOBAL_SOLVERS), scenario_solver_aliases={'reld_mt': 'RELD_MOEL'},
        data_root=str(args.data_root), cache_root=reference['cache_root'], prepared_root=str(args.prepared_root),
        resource_cache_policy=reference['resource_cache_policy'],
        preparation_root=str(args.preparation_root), preparation_lock=prepared,
        common_trainable_sha256=FRESH_SHA, rows_per_problem=dict(SPLIT_ROWS),
        train_rows_per_problem=81000, heldout_rows_per_problem=4500,
        candidate_pools=reference['candidate_pools'], candidate_pool_ids=reference['candidate_pool_ids'],
        winner_rule=WINNER_RULE, baselines=baselines, own_train_sbs=baselines, geometry_stats=geometry,
        constraint_reference=reference, controller=dict(CONTROLLER),
        global_batch_size=BATCH_SIZE, logical_global_batch=BATCH_SIZE, world_size=1,
        max_successful_updates=MAX_UPDATES, updates_per_problem=MAX_UPDATES // len(PROBLEMS),
        total_sample_exposures=BATCH_SIZE * MAX_UPDATES, eval_every_updates=EVAL_EVERY,
        warmup_updates=WARMUP_UPDATES, fixed_budget=True, early_stop=False,
        lr=dict(rule='successful-update warmup then cosine, independent of VAL', peak=PEAK_LR,
                end=END_LR, warmup_updates=WARMUP_UPDATES, max_updates=MAX_UPDATES),
        loss=dict(LOSS), rdrop_weight=0., rdrop_ramp_updates=0, dropout=.1,
        optimizer='AdamW', weight_decay=1e-4, grad_clip=1.,
        checkpoint_rule=CHECKPOINT_RULE, inference_rule=INFERENCE_RULE,
        selection_head='original_score_mlp_maximin',
        task_order_sha256=hashlib.sha256('\n'.join(order).encode('ascii')).hexdigest(),
        source_plan='full81000 per task natural seed2 shuffle; tails carried, no replacement/quota sampling',
        microbatch_caps=micro_caps(), gpu_type=args.gpu_type, eval_batch_size=args.eval_batch_size,
        gradients='row-weighted complete1024; one clip/step; AMP retries same rows/RNG; commit once after success',
        train_eval='selected checkpoint only; TRAIN hard feedback uses training batches only',
        test='separate fullTEST command after fixed14400 and unique VAL lock; existing R84 cache only',
        test_rnd_disclosure='Old heldouts and TEST informed earlier R&D; not a fresh blind confirmation',
        current_query_labels_in_forward=False, historical_TRAIN_label_context_used=False,
        metric_semantics=dict(optimization_ce='classification upper bound: native utility CE in bits',
            evaluation_ce='unchanged R84 maximin-logit CE in nats',
            constraints='unscaled per-instance hard relative regret and unconditional fixed36 missed frequency',
            all='equal-task macro; raw macro cost descriptive only'),
        source_sha256={name: file_hash(V4 / name) for name in SOURCE_FILES},
        wandb=True, wandb_mode=args.wandb_mode)
    validate_config(config)
    return config


def validate_config(config):
    expected = dict(experiment='R122', schema_version=1, arm='model', seed=2,
        initialization='scratch_shared_r77', model_factory=MODEL_FACTORY,
        solver_names=list(GLOBAL_SOLVERS), common_trainable_sha256=FRESH_SHA,
        rows_per_problem=dict(SPLIT_ROWS), global_batch_size=BATCH_SIZE, logical_global_batch=BATCH_SIZE,
        max_successful_updates=MAX_UPDATES, eval_every_updates=EVAL_EVERY, warmup_updates=WARMUP_UPDATES,
        updates_per_problem=MAX_UPDATES // len(PROBLEMS), microbatch_caps=micro_caps(), eval_batch_size=256,
        loss=LOSS, controller=CONTROLLER, rdrop_weight=0., rdrop_ramp_updates=0,
        optimizer='AdamW', weight_decay=1e-4, grad_clip=1., dropout=.1,
        checkpoint_rule=CHECKPOINT_RULE, inference_rule=INFERENCE_RULE,
        selection_head='original_score_mlp_maximin', world_size=1, fixed_budget=True, early_stop=False,
        current_query_labels_in_forward=False, historical_TRAIN_label_context_used=False)
    if any(config.get(key) != value for key, value in expected.items()):
        raise ValueError('R122 fixed full18 architecture/objective/execution protocol differs')
    reference = config['constraint_reference']
    if (reference['train']['source_split'] != 'train' or reference['val_selection']['source_split'] != 'val'
            or reference['checkpoint_reference']['checkpoint_sha256'] != REFERENCE_SHA
            or reference['checkpoint_reference']['selected_update'] != REFERENCE_UPDATE
            or config['model_params'] != reference['model_params']
            or config['model_params'].get('dropout') != .1
            or config['candidate_pools'] != reference['candidate_pools']
            or config['candidate_pool_ids'] != reference['candidate_pool_ids']
            or config['lr'] != dict(rule='successful-update warmup then cosine, independent of VAL',
                peak=PEAK_LR, end=END_LR, warmup_updates=WARMUP_UPDATES, max_updates=MAX_UPDATES)):
        raise ValueError('R122 TRAIN/VAL budgets, original model identity or LR changed')
    fixed_targets(dict(targets=reference['fixed_targets']), config['candidate_pools'])


def load_checkpoint(path, device='cpu'):
    """LIB-compatible factory: return unchanged inference selector and full state."""
    state = torch.load(path, map_location='cpu', weights_only=False)
    config = state['args']
    validate_config(config)
    if state.get('state_hash') != engine.json_hash(config):
        raise ValueError('Checkpoint protocol hash differs')
    model = make_model(config['model_params'], state_dict=state['model'])
    if parameter_hash(model.state_dict()) != parameter_hash(state['model']):
        raise ValueError('R122 factory failed to restore complete parameters and buffers')
    controller = make_controller(config['constraint_reference'])
    budget, unit = controller.budget.clone(), controller.unit.clone()
    controller.load_state_dict(state['controller'], strict=True)
    if (not torch.equal(controller.budget, budget) or not torch.equal(controller.unit, unit)
            or not all(bool(torch.isfinite(value).all()) for value in controller.state_dict().values())
            or bool((controller.multipliers < 0).any()) or bool((controller.multipliers > CONTROLLER['dual_cap']).any())):
        raise ValueError('Checkpoint controller must retain its locked TRAIN budgets and finite bounded state')
    validate_resume(state, config, task_order(MAX_UPDATES))
    return model.to(device), state


load_model = from_checkpoint = load_checkpoint


def update_batch(objective, optimizer, scaler, batch, runtime, cap, *, problem,
                 successful_update, max_retries=20):
    """Row-weighted microbatches, transactional AMP retry, exactly one commit."""
    count = len(batch['ind'])
    controller = objective.controller
    if (runtime.world_size != 1 or count < 1 or cap < 1 or max_retries < 1
            or problem not in controller.problems or not objective.training
            or not controller.training or successful_update != int(controller.committed_updates) + 1):
        raise ValueError('Need the next complete single-device logical TRAIN update')
    rng = capture_rank_rng(runtime.device)
    keys = ('hard_error', 'hard_regret', 'hard_fixed_miss', 'fixed_population')
    for retry in range(max_retries):
        restore_rank_rng(rng, runtime.device)
        optimizer.zero_grad(set_to_none=True)
        parts_sum = torch.zeros(8, dtype=torch.float64, device=runtime.device)
        hard_sum = torch.zeros(4, dtype=torch.float64, device=runtime.device)
        bad = False
        for start in range(0, count, cap):
            stop = min(start + cap, count)
            loss, parts = objective(slice_batch(batch, start, stop))
            scaler.scale(loss * ((stop - start) / count)).backward()
            bad = bad or not bool(torch.isfinite(loss.detach()))
            parts_sum.add_(parts.detach().double(), alpha=stop - start)
            hard_sum.add_(torch.stack([objective.last_hard[k] for k in keys]).double(), alpha=stop - start)
        scaler.unscale_(optimizer)
        norm = torch.nn.utils.clip_grad_norm_(objective.selector.parameters(), 1.)
        bad = bad or not bool(torch.isfinite(norm)) or not bool(torch.isfinite(hard_sum).all())
        if bad:
            if scaler.is_enabled():
                scaler.update(new_scale=scaler.get_scale() * .5)
            continue
        scale_before = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        if scaler.is_enabled() and scaler.get_scale() < scale_before:
            continue
        hard = hard_sum / count
        cycle = controller.commit(problem, hard_regret=hard[1].item(), hard_fixed_miss=hard[2].item(),
                                  successful_update=successful_update)
        metrics = dict(zip(keys, hard.cpu().tolist()))
        metrics.update(dual_regret=float(controller.multipliers[0]),
            dual_fixed_miss=float(controller.multipliers[1]), controller_cycle_completed=bool(cycle))
        return torch.cat((parts_sum / count, norm.detach().double().reshape(1))), retry, metrics
    optimizer.zero_grad(set_to_none=True)
    raise RuntimeError('AMP retry budget exhausted; no optimizer/controller update accepted')


@lru_cache(maxsize=1)
def private_engine():
    name = 'code.V4._r122_private_r84'
    spec = importlib.util.spec_from_file_location(name, engine.__file__)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    module.MAX_UPDATES, module.EVAL_EVERY = MAX_UPDATES, EVAL_EVERY
    module.micro_caps, module.inputs_only = micro_caps, query_inputs
    return module


def status_update(args, phase, step=None, **values):
    return private_engine().status_update(args, phase, step, **values)


def evaluation_hard(result, targets, split):
    archive = Path(result['prediction_archive'])
    directory = archive.parent / 'by_problem'
    per = {}
    with np.load(archive, allow_pickle=False) as combined:
        for p in PROBLEMS:
            shard = directory / (p + '.npz')
            if shard.exists():
                with np.load(shard, allow_pickle=False) as saved:
                    data = {k: saved[k] for k in ('logits', 'costs', 'winner', 'pool', 'pool_ids')}
            else:
                data = {k: combined[p + '__' + k] for k in ('logits', 'costs', 'winner', 'pool', 'pool_ids')}
            per[p] = hard_statistics(data, targets[p])
            reported = result['per_problem'][p]
            if (per[p]['n'] != SPLIT_ROWS[split] or per[p]['n'] != reported['n']
                    or not math.isclose(per[p]['native_top1'], reported['top1'], abs_tol=1e-12, rel_tol=0.)
                    or not math.isclose(per[p]['hard_regret'] * 100., reported['actual_regret_pct'],
                                        abs_tol=1e-10, rel_tol=0.)):
                raise ValueError('Hard-decision statistics disagree with original R84 evaluation metrics')
    return dict(per_problem=per, macro=macro_hard(per),
                aggregation='equal-task natural population; fixed misses unconditional, not conditional recall')


def evaluate(selector, args, manifest, baselines, split, tag, step, state_hash, checkpoint_sha256=None):
    if split == 'test':
        lock, _ = authorize_test(args)
        if (step != lock['selected_update'] or state_hash != lock['state_hash']
                or checkpoint_sha256 != lock['checkpoint_sha256']
                or parameter_hash(selector.state_dict()) != lock['model_state_sha256']):
            raise PermissionError('FullTEST may evaluate only the exact locked checkpoint and inference state')
    result = private_engine().evaluate(selector, args, manifest, baselines, split, tag, step,
                                       state_hash, checkpoint_sha256)
    reference = read_json(args.preparation_root / 'constraint_reference.json')
    targets = fixed_targets(dict(targets=reference['fixed_targets']), reference['candidate_pools'])
    hard = evaluation_hard(result, targets, split)
    if 'hard_decision' in result and result['hard_decision'] != hard:
        raise PermissionError('Saved R122 hard-decision evaluation changed')
    result['hard_decision'] = hard
    write_json(args.root / args.arm / 'evaluation' / split / tag / 'metrics.json', result)
    return result


def validate_resume(state, config, order):
    step = state['successful_updates']
    pending = state.get('pending_evaluation') is not None
    if (state['args'] != config or state['state_hash'] != engine.json_hash(config)
            or not 0 <= step <= MAX_UPDATES or step % EVAL_EVERY
            or step != (len(state['history']) + int(pending)) * EVAL_EVERY):
        raise ValueError('Resume requires the identical protocol at a saved interval boundary')
    counts = Counter(order[:step])
    if (Counter(state['updates_by_problem']) != counts
            or any(state['stream_draws'][p] != counts[p] * BATCH_SIZE for p in PROBLEMS)
            or int(state['controller']['committed_updates']) != step
            or int(state['controller']['completed_cycles']) != step // len(PROBLEMS)
            or bool(state['controller']['task_count'].any())
            or bool(state['controller']['task_sum'].any())):
        raise ValueError('Resume controller/task buffers/source cursors disagree with successful updates')


def bind_selected_val(root, best, lock, config):
    source = root / 'evaluation/val' / best['evaluation_tag']
    destination = root / 'evaluation/val/selected_best'
    destination.mkdir(parents=True, exist_ok=True)
    capture = read_json(source / 'identity.json')
    if (capture['successful_updates'] != lock['selected_update']
            or capture['state_hash'] != lock['state_hash'] or capture['model_state_sha256'] != lock['model_state_sha256']):
        raise ValueError('Selected VAL does not belong to the locked checkpoint')
    target = destination / 'rank000.npz'
    if target.exists():
        if file_hash(target) != file_hash(source / 'rank000.npz'):
            raise PermissionError('Unique selected VAL predictions changed')
    else:
        shutil.copyfile(source / 'rank000.npz', target)
    identity = dict(capture, tag='selected_best', checkpoint_sha256=lock['checkpoint_sha256'])
    immutable_json(destination / 'identity.json', identity)
    result = copy.deepcopy(best['val'])
    result.update(identity=identity, tag='selected_best', prediction_archive=str(target.resolve()),
        archive_capture_identity=capture,
        checkpoint_binding=dict(checkpoint_sha256=lock['checkpoint_sha256'],
            model_state_sha256=lock['model_state_sha256'], state_hash=engine.json_hash(config),
            selected_update=lock['selected_update'], original_metrics_unchanged=True))
    immutable_json(destination / 'metrics.json', result)
    return result


def train(args, device):
    root = args.root / args.arm
    if (root / 'training_complete.json').exists():
        authorize_test(args)
        return read_json(root / 'final_metrics.json')
    reference, prepared, ready, manifest, geometry, baselines = load_preparation(args)
    initial = torch.load(args.prepared_root / 'identicalinit_seed2.pt', map_location='cpu', weights_only=False)
    selector, params, initialization = engine.initialize_shared(initial, geometry, 'A_baseline', device)
    del initial
    config = protocol(args, reference, prepared, ready, geometry, baselines, params)
    state_hash = engine.json_hash(config)
    if args.state_hash is not None and args.state_hash != state_hash:
        raise ValueError('Expected protocol state hash differs')
    last = root / 'last.pt'
    if last.exists() and not args.resume:
        raise PermissionError('Existing R122 run requires --resume; preserve it')
    immutable_json(root / 'protocol.json', config)
    write_json(root / 'cache_transport.json', dict(runtime_cache_root=str(args.cache_root),
        frozen_cache_root=reference['cache_root'], scientific_protocol_unchanged=True,
        validation=reference['resource_cache_policy']))
    order, caps = task_order(MAX_UPDATES), micro_caps()
    checkpoint = None
    if last.exists():
        del selector
        selector, checkpoint = load_checkpoint(last, device)
        validate_resume(checkpoint, config, order)
        initialization = checkpoint['initialization']
    objective = HardDecisionObjective(selector, make_controller(reference)).to(device)
    optimizer = torch.optim.AdamW(selector.parameters(), lr=PEAK_LR, weight_decay=1e-4)
    scaler = torch.amp.GradScaler('cuda', enabled=device.type == 'cuda')
    history, best, successful, retries_total, pending = [], None, 0, 0, None
    counts, cursors = Counter(), dict.fromkeys(PROBLEMS, 0)
    if checkpoint is not None:
        objective.controller.load_state_dict(checkpoint['controller'], strict=True)
        optimizer.load_state_dict(checkpoint['optimizer'])
        for state in optimizer.state.values():
            for key, value in state.items():
                if torch.is_tensor(value):
                    state[key] = value.cpu() if key == 'step' else value.to(device)
        scaler.load_state_dict(checkpoint['scaler'])
        history, best, successful = checkpoint['history'], checkpoint['best'], checkpoint['successful_updates']
        retries_total, counts = checkpoint['amp_retries'], Counter(checkpoint['updates_by_problem'])
        cursors, pending = checkpoint['stream_draws'], checkpoint['pending_evaluation']
        restore_rank_rng(checkpoint['rng'], device)
    else:
        set_seed(2)
    del checkpoint
    streams = {p: ShuffleStream(81000, task_seed(2, p), cursors[p]) for p in PROBLEMS}
    targets = fixed_targets(dict(targets=reference['fixed_targets']), reference['candidate_pools'])
    caches, tracker = {}, None
    runtime = Runtime(rank=0, world_size=1, device=device)

    def snapshot():
        return dict(model=selector.state_dict(), controller=objective.controller.state_dict(),
            optimizer=optimizer.state_dict(), scaler=scaler.state_dict(), args=config, state_hash=state_hash,
            initialization=initialization, successful_updates=successful, updates_by_problem=dict(counts),
            sample_exposures=successful * BATCH_SIZE, stream_draws={p: s.draws for p, s in streams.items()},
            amp_retries=retries_total, rng=capture_rank_rng(device), history=history, best=best,
            pending_evaluation=pending)

    if not last.exists():
        save_checkpoint(last, snapshot())
    try:
        tracker = engine.wandb_run(args, config)
        restore_rank_rng(torch.load(last, map_location='cpu', weights_only=False)['rng'], device)
        while successful < MAX_UPDATES or pending is not None:
            if pending is None:
                if not caches:
                    factory = lambda directory: NativeConstraintCache(directory, targets=targets)
                    for p in PROBLEMS:
                        caches[p] = open_cache(args.cache_root, 'train', p, manifest, args.data_root,
                                               cache_factory=factory)
                first, end = successful, min(MAX_UPDATES, successful + EVAL_EVERY)
                requests = [(p, streams[p].take(BATCH_SIZE), BATCH_SIZE) for p in order[first:end]]
                started = time.monotonic()
                parts_sum = torch.zeros(9, dtype=torch.float64, device=device)
                hard_sum = Counter()
                objective.train()
                with PinnedPrefetch(caches, requests, device) as batches:
                    for problem, local, total, batch in batches:
                        if local != BATCH_SIZE or total != BATCH_SIZE:
                            raise ValueError('Each update must use all1024 natural TRAIN rows')
                        lr = learning_rate(successful + 1)
                        for group in optimizer.param_groups:
                            group['lr'] = lr
                        parts, retries, hard = update_batch(objective, optimizer, scaler, batch, runtime,
                            caps[problem], problem=problem, successful_update=successful + 1)
                        successful += 1
                        counts[problem] += 1
                        retries_total += retries
                        parts_sum.add_(parts)
                        hard_sum.update({k: hard[k] for k in ('hard_error', 'hard_regret', 'hard_fixed_miss')})
                        if successful == 1 or successful % args.log_every == 0 or successful == end:
                            done, elapsed = successful - first, time.monotonic() - started
                            info = dict(successful_updates=successful, problem=problem, lr=lr,
                                amp_retries=retries_total, seconds_per_update=elapsed / done,
                                samples_per_second=done * BATCH_SIZE / elapsed, updates_by_problem=dict(counts),
                                controller_commits=int(objective.controller.committed_updates),
                                controller_cycles=int(objective.controller.completed_cycles),
                                **dict(zip(engine.PART_KEYS, (parts_sum / done).cpu().tolist())), **hard)
                            status_update(args, 'training', successful, **{k: v for k, v in info.items()
                                                                          if k != 'successful_updates'})
                            with (root / 'updates.jsonl').open('a') as stream:
                                stream.write(json.dumps(dict(updated_utc=utc(), **info), allow_nan=False) + '\n')
                            print('[R122 train] ' + json.dumps(info), flush=True)
                            tracker.log({**{'optimization/' + k: info[k] for k in engine.PART_KEYS},
                                **{'constraints/' + k: v for k, v in hard.items()}, 'lr': lr,
                                'successful_updates': successful}, step=successful)
                pending = dict(zip(engine.PART_KEYS, (parts_sum / (successful - first)).cpu().tolist()))
                pending.update({k: v / (successful - first) for k, v in hard_sum.items()})
                pending.update(seconds=time.monotonic() - started, lr=lr, amp_retries=retries_total)
                save_checkpoint(last, snapshot())
            tag = f'update{successful:07d}'
            validation = evaluate(selector, args, manifest, baselines, 'val', tag, successful, state_hash)
            key = selection_key(validation, reference['val_selection']['macro'], successful)
            row = dict(epoch=successful // EVAL_EVERY, evaluation_point=successful // EVAL_EVERY,
                evaluation_tag=tag, successful_updates=successful,
                sample_exposures=successful * BATCH_SIZE, updates_by_problem=dict(counts), optimization=pending,
                selection_key=list(key), feasible=key[0] == 0, val=validation, train=None)
            history.append(row)
            improved = best is None or key < tuple(best['selection_key'])
            if improved:
                best = copy.deepcopy(row)
            pending = None
            state = snapshot()
            if improved:
                save_checkpoint(root / 'best.pt', state)
                write_json(root / 'best_eval.json', best)
            save_checkpoint(last, state)
            del state
            write_json(root / 'history.json', history)
            engine.write_metrics_csv(root, history)
            status_update(args, 'evaluation_complete', successful, latest_val=validation['macro'],
                latest_constraints=validation['hard_decision']['macro'], selected_update=best['successful_updates'],
                selected_feasible=best['feasible'])
            tracker.log({**{'val/' + k: v for k, v in validation['macro'].items()},
                **{'val_constraints/' + k: v for k, v in validation['hard_decision']['macro'].items()},
                'val_constraints/feasible': key[0] == 0, 'successful_updates': successful}, step=successful)
        if (successful != MAX_UPDATES or any(counts[p] != MAX_UPDATES // len(PROBLEMS) for p in PROBLEMS)
                or int(objective.controller.committed_updates) != successful
                or int(objective.controller.completed_cycles) != MAX_UPDATES // len(PROBLEMS)):
            raise ValueError('Complete full18 optimizer/controller update budget not reached')
        for cache in caches.values():
            cache.close()
        caches.clear()
        del objective, optimizer, scaler, selector
        selector, selected = load_checkpoint(root / 'best.pt', device)
        if selected['args'] != config or selected['successful_updates'] != best['successful_updates']:
            raise ValueError('Selected model differs from complete VAL-only history')
        model_hash = parameter_hash(selector.state_dict())
        lock = dict(experiment='R122', arm='model', checkpoint=str((root / 'best.pt').resolve()),
            checkpoint_sha256=file_hash(root / 'best.pt'), state_hash=state_hash,
            model_state_sha256=model_hash, selected_update=best['successful_updates'], epoch=best['epoch'],
            rule=CHECKPOINT_RULE, inference_rule=INFERENCE_RULE, winner_rule=WINNER_RULE,
            selected_without_test=True, per_problem_assembly=False, full18=True,
            selection_key=best['selection_key'], feasible=best['feasible'], val=best['val']['macro'],
            hard_val=best['val']['hard_decision']['macro'], preparation_lock=prepared)
        immutable_json(root / 'final_test_lock.json', lock)
        selected_val = bind_selected_val(root, best, lock, config)
        training = evaluate(selector, args, manifest, baselines, 'train', 'selected_best',
                            lock['selected_update'], state_hash, lock['checkpoint_sha256'])
        best.update(train=training, val=selected_val)
        for row in history:
            if row['successful_updates'] == lock['selected_update']:
                row.update(train=training, val=selected_val)
        write_json(root / 'history.json', history)
        engine.write_metrics_csv(root, history)
        write_json(root / 'best_eval.json', best)
        write_json(root / 'result.json', dict(best=best, final=history[-1], successful_updates=successful,
            updates_by_problem=dict(counts), amp_retries=retries_total, state_hash=state_hash,
            stop_reason='fixed14400 successful updates; no TEST selection or early stopping', test_read=False))
        final = dict(selected_update=lock['selected_update'], checkpoint_sha256=lock['checkpoint_sha256'],
            state_hash=state_hash, train=training, val=selected_val, test_read=False)
        write_json(root / 'final_metrics.json', final)
        write_json(root / 'training_complete.json', dict(successful_updates=successful,
            updates_by_problem=dict(counts), selected_best=lock, state_hash=state_hash,
            sample_exposures=successful * BATCH_SIZE, next_stage='explicit locked fullTEST', test_read=False))
        status_update(args, 'training_complete', successful, selected_update=lock['selected_update'])
        return final
    finally:
        for cache in caches.values():
            cache.close()
        if tracker is not None:
            tracker.finish()


def authorize_test(args):
    lock, config = private_engine().authorize_test(args)
    validate_config(config)
    complete = read_json(args.root / args.arm / 'training_complete.json')
    if (complete.get('selected_best') != lock or complete.get('state_hash') != lock['state_hash']
            or complete.get('updates_by_problem') != dict.fromkeys(PROBLEMS, MAX_UPDATES // len(PROBLEMS))
            or lock.get('model_state_sha256') is None or lock.get('rule') != CHECKPOINT_RULE
            or lock.get('inference_rule') != INFERENCE_RULE
            or Path(config['preparation_root']).resolve() != args.preparation_root.resolve()
            or config['preparation_lock'] != read_json(args.preparation_root / 'prepared.json')):
        raise PermissionError('TEST requires the unchanged full18 unique checkpoint and CPU preparation lock')
    verify_receipts([config['preparation_lock']['constraint_reference']])
    for name, digest in config['source_sha256'].items():
        if file_hash(V4 / name) != digest:
            raise PermissionError('R122 executable source changed after protocol lock: ' + name)
    return lock, config


def ensure_test_cache(args, lock):
    verified, config = authorize_test(args)
    if verified != lock:
        raise PermissionError('TEST cache request differs from the unique lock')
    manifest = read_json(Path(config['prepared_root']) / 'source_cache_manifest.json')
    tasks = manifest['splits'].get('test', {})
    if (set(tasks) != set(PROBLEMS) or any(tasks[p].get('complete') is not True
            or tasks[p]['rows'] != 4500 or tasks[p]['pool'] != config['candidate_pools'][p]
            or tasks[p]['pool_ids'] != config['candidate_pool_ids'][p] for p in PROBLEMS)):
        raise PermissionError('Require existing full18 R84 TEST cache; never build/copy/solve')
    root = args.cache_root
    if not all((root / 'test' / p / 'metadata.json').is_file() for p in PROBLEMS):
        root = FROZEN_TEST_CACHE_ROOT
    for p in PROBLEMS:
        validate_cache_copy(root / 'test' / p, tasks[p],
                            label_reference=FROZEN_TEST_CACHE_ROOT / 'test' / p)
    args.cache_root = root
    return manifest


def test(args, device):
    lock, _ = authorize_test(args)
    result = private_engine().test(args, device, integration=sys.modules[__name__])
    if (result['identity']['checkpoint_sha256'] != lock['checkpoint_sha256']
            or result['identity']['state_hash'] != lock['state_hash']
            or result['identity']['model_state_sha256'] != lock['model_state_sha256']):
        raise PermissionError('TEST capture must name the unique locked checkpoint')
    return result


def describe(args):
    return dict(experiment='R122', arm='model', root=str(args.root / 'model'), model_factory=MODEL_FACTORY,
        preparation_root=str(args.preparation_root), reference_root=str(REFERENCE_ROOT),
        preferred_resource_cache_root=str(LOCAL_CACHE_ROOT),
        frozen_test_cache_root=str(FROZEN_TEST_CACHE_ROOT),
        reference_checkpoint_sha256=REFERENCE_SHA, reference_selected_update=REFERENCE_UPDATE,
        fixed_target_list=str(TARGET_LIST), fixed_target_count=36, seed=2, rows_per_problem=dict(SPLIT_ROWS),
        global_batch_size=BATCH_SIZE, logical_global_batch=BATCH_SIZE, max_successful_updates=MAX_UPDATES,
        warmup_updates=WARMUP_UPDATES, eval_every_updates=EVAL_EVERY, microbatch_caps=micro_caps(),
        optimizer='AdamW', peak_lr=PEAK_LR, end_lr=END_LR, weight_decay=1e-4, dropout=.1, grad_clip=1.,
        architecture='unchanged R45A / R84A ScoreDifferenceSelector full18',
        initialization='frozen R84 scratch seed2; no trained weights',
        controller=CONTROLLER, loss=LOSS, checkpoint_rule=CHECKPOINT_RULE, inference_rule=INFERENCE_RULE,
        winner_rule=WINNER_RULE, test='separate explicit command after full budget and unique lock')


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('command', choices=('prepare', 'train', 'test', 'describe', 'status'))
    result.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    result.add_argument('--arm', choices=ARMS, default='model')
    result.add_argument('--preparation-root', type=Path)
    result.add_argument('--prepared-root', type=Path, default=DEFAULT_PREPARED_ROOT)
    result.add_argument('--data-root', type=Path, default=DEFAULT_DATA_ROOT)
    result.add_argument('--cache-root', type=Path)
    result.add_argument('--device', default='cpu')
    result.add_argument('--gpu-type', choices=('3090', '4090'), default='4090')
    result.add_argument('--large-micro-cap', type=int, choices=(256,), default=256)
    result.add_argument('--eval-batch-size', type=int, choices=(256,), default=256)
    result.add_argument('--log-every', type=int, default=18)
    result.add_argument('--wandb-mode', choices=('auto', 'online', 'offline'), default='auto')
    result.add_argument('--resume', action='store_true')
    result.add_argument('--state-hash', '--expected-state-hash')
    result.set_defaults(experiment='R122')
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    for name in ('root', 'prepared_root', 'data_root', 'cache_root', 'preparation_root'):
        value = getattr(args, name)
        if value is not None:
            setattr(args, name, value.expanduser().resolve())
    args.preparation_root = args.preparation_root or args.root / 'preparation' / PREPARATION_VERSION
    if args.command == 'describe':
        value = describe(args)
        print(json.dumps(value, indent=2), flush=True)
        return value
    if args.command == 'status':
        path = args.root / args.arm / 'status.json'
        value = read_json(path) if path.exists() else dict(experiment='R122', phase='not_started')
        print(json.dumps(value, indent=2), flush=True)
        return value
    if args.cache_root is None:
        args.cache_root = (LOCAL_CACHE_ROOT.resolve()
            if args.command in ('train', 'test') and LOCAL_CACHE_ROOT.is_dir()
            else Path(read_json(args.prepared_root / 'prepared.json')['cache_source']).resolve())
    torch.set_num_threads(2)
    if args.command == 'prepare':
        return prepare(args)
    device = torch.device(args.device)
    if device.type != 'cuda':
        raise ValueError('Formal train/TEST requires an explicitly assigned CUDA device; prepare is CPU-only')
    if args.log_every < 1:
        raise ValueError('Log interval must be positive')
    root = args.root / args.arm
    root.mkdir(parents=True, exist_ok=True)
    with (root / 'runner.lock').open('a') as claim:
        fcntl.flock(claim, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            configure_torch()
            return train(args, device) if args.command == 'train' else test(args, device)
        except BaseException as error:
            record = dict(type=type(error).__name__, message=str(error), traceback=traceback.format_exc())
            write_json(root / 'error.json', record)
            status_update(args, 'failed', error=record)
            raise


if __name__ == '__main__':
    main()
