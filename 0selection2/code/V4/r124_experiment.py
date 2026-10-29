"""One R124 full18 Top2 experiment: cache, train, VAL lock, TEST, LIB factory.

No input cache is rebuilt by the TRAIN/VAL score-cache command. Direct deployed
inference includes the complete frozen R77B and the independently trained
reranker. TEST features are opened only after the R124-specific lock; no input
features or frozen expert caches are rebuilt by this module.
"""

from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
import fcntl
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
import torch

from code.unified_selector.registry import GLOBAL_SOLVERS, PROBLEMS
from .multitask_probe import task_seed
from .r76_experiment import (capture_rank_rng, restore_rank_rng, file_hash, fixed_sbs_metrics,
    open_cache, read_json, save_checkpoint, slice_batch, utc, write_json)
from .r77_data import parameter_hash
from .r77_experiment import ShuffleStream, task_order
from .r80_experiment import WINNER_RULE, json_hash, lock_best, save_npz, wandb_run
from .r84_experiment import prepared_config as r84_prepared_config, summarize
from .r84_prepare import DEFAULT_DATA_ROOT, DEFAULT_PREPARED_ROOT, SPLIT_ROWS
from .r123_meta_data import (PREPARATION as META_PREPARATION, EXPERTS,
                            load_expert, load_meta_indices)
from .r124_model import (Top2Reranker, FrozenTop2Selector, candidate_ids, top2_indices,
                         pair_objective, query_inputs)
from .train import set_seed


V4 = Path(__file__).resolve().parent
DEFAULT_ROOT = V4 / 'runs/R124_top2_reranking'
PREPARATION = DEFAULT_ROOT / 'preparation'
SCORE_ROOT = PREPARATION / 'stage1_cache'
MODEL_FACTORY = 'code.V4.r124_experiment.load_checkpoint'
EXPERT_ID = 'R77B'
EXPERT_SHA = '31c0c75d66b1bde6ad533909a9c04b0036e80af4c4a5775b01b957ce310e6d1b'
FRESH_SHA = 'dad87d8598f7560e6f170ef048ba1bf515edeb03cef7ec42e16839a36af4ef93'
META_ROWS, BATCH_SIZE, MAX_UPDATES, WARMUP_UPDATES, EVAL_EVERY = 10886, 1024, 7200, 540, 720
CHECKPOINT_RULE = 'strict minimum full18 VAL macro_actual_regret_pct; higher native macro_top1; earliest update'
INFERENCE_RULE = ('frozen complete R77B actual legal Top2, pair-native sigmoid(d); replace only Top2 ordering '
                  'and keep both above rank3; ascending-global-method-ID exact ties; quality head is not an action')
LOSS = dict(native='indicator(Y in S2) * BCE(native pair label, sigmoid(d)); mean over ALL logical rows',
    quality='separate antisymmetric prediction of (C_b-C_a)/C_*; Huber(pred/.01, target/.01)',
    quality_scale=.01, quality_coefficient=1., huber_beta=1., expected_risk=False,
    label_precision='original native int64 winner; unchanged FP64 costs; exact ties retained')
SOURCES = ('r124_model.py', 'r124_experiment.py', 'r123_meta_data.py', 'V4Model.py',
           'dual_stream.py', 'solver_features.py', 'local_geometry.py', 'r76_cache.py',
           'r77_experiment.py', 'r76_experiment.py', 'pairwise_selector.py', 'solver_code_encoder.py')
SCIENCE = ('docs/research/post_r119_global_reassessment_20261028/HANDOFF.md',
    'docs/research/return18_major_routes_20261030/HANDOFF.md',
    'docs/research/return18_major_routes_20261030/astra_followup.md')


def learning_rate(update):
    if not 1 <= update <= MAX_UPDATES:
        raise ValueError('Learning rate uses a one-based successful update')
    if update <= WARMUP_UPDATES:
        return 1e-4 * update / WARMUP_UPDATES
    progress = (update - WARMUP_UPDATES) / (MAX_UPDATES - WARMUP_UPDATES)
    return 1e-6 + .5 * (1e-4 - 1e-6) * (1. + math.cos(math.pi * progress))


def configure_precision():
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')


def micro_caps(args, *, evaluation=False):
    other = args.eval_batch_size if evaluation else args.microbatch_size
    large = min(args.large_microbatch_size, other)
    return {p: large if p in ('TSP', 'CVRP') else other for p in PROBLEMS}


def immutable_json(path, value):
    path = Path(path)
    if path.exists():
        if read_json(path) != value:
            raise ValueError(f'Preserve existing immutable R124 artifact: {path}')
    else:
        write_json(path, value)
    return value


@contextmanager
def exclusive(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def first_stage_binding(meta):
    info = meta['experts'][EXPERT_ID]
    path = V4 / 'runs' / EXPERTS[EXPERT_ID][0] / 'best.pt'
    lock_path = path.parent / 'final_test_lock.json'
    lock = read_json(lock_path)
    if (info['loader_status'] != 'loaded_strictly_on_cpu'
            or not info['full18_training_budget_complete']
            or info['declared_train_rows_per_problem'] != 70114
            or info.get('membership_evidence_gap')
            or info['checkpoint_sha256'] != EXPERT_SHA or file_hash(path) != EXPERT_SHA
            or lock['checkpoint_sha256'] != EXPERT_SHA or lock['selected_update'] != 34776
            or not lock['selected_without_test'] or lock['per_problem_assembly'] is not False):
        raise ValueError('Only the frozen COMPLETE full18 R77B deployment is permitted')
    return dict(expert_id=EXPERT_ID, checkpoint_path=str(path), checkpoint_sha256=EXPERT_SHA,
        selected_update=34776, complete_updates=49680, historical_train_rows_per_problem=70114,
        external_pretraining=True, updated_by_r124=False, meta_membership_scope='source-record only')


def source_context(args):
    ready, manifest, geometry, baselines = r84_prepared_config(args)
    meta = read_json(args.meta_preparation / 'manifest.json')
    if (not meta['meta_training_ready_at_source_record_scope']
            or Path(meta['current_train_data_root']).resolve() != args.data_root.resolve()
            or ready['trainable_sha256'] != FRESH_SHA):
        raise ValueError('Use the frozen R84 fresh preparation and certified R123 membership')
    indices = {p: load_meta_indices(p, args.meta_preparation) for p in PROBLEMS}
    for p, rows in indices.items():
        info = meta['problems'][p]
        if (len(rows) != META_ROWS or rows.dtype != np.int64 or len(np.unique(rows)) != META_ROWS
                or rows.min() < 0 or rows.max() >= 81000
                or info['current_train_count'] != 81000 or info['historical_train_count'] != 70114
                or manifest['splits']['train'][p]['pool'] != info['candidates']
                or manifest['splits']['train'][p]['pool_ids'] != info['global_solver_ids']):
            raise ValueError(f'Frozen source membership/pool changed: {p}')
    params = dict(ready['model_params'], joint_layer_num=2, dropout=.1)
    if (params['embedding_dim'] != 128 or params['encoder_layer_num'] != 4
            or params['solver_feature_spec']['feature_dim'] != 34):
        raise ValueError('R124 requires original4layer128 plus34 method features')
    return dict(ready=ready, manifest=manifest, geometry=geometry, baselines=baselines,
                meta=meta, indices=indices, params=params, first_stage=first_stage_binding(meta))


def contract(args, context):
    method_path = args.root / 'preparation/method_config.json'
    if method_path.exists():
        frozen, actual = read_json(method_path), describe(args)
        if any(frozen.get(key) != value for key, value in actual.items()):
            raise ValueError('Method, loss, caps or inference rule differ from the prelaunch R124 freeze')
        for name, digest in frozen['software_source_sha256'].items():
            if file_hash(V4 / name) != digest:
                raise ValueError('Frozen R124 software source changed: ' + name)
    source = context['manifest']
    repo = V4.parents[1]
    return dict(experiment='R124', arm='model', schema_version=1, seed=2,
        initialization='scratch_r84_compatible_new_encoder', model_params=context['params'],
        checkpoint_scope='new reranker only; frozen external R77B independently hash verified by factory',
        model_factory=MODEL_FACTORY, solver_names=list(GLOBAL_SOLVERS),
        data_root=str(args.data_root), prepared_root=str(args.prepared_root),
        cache_root=str(Path(context['ready']['cache_source']).resolve()),
        meta_preparation=str(args.meta_preparation), score_root=str(args.score_root),
        preparation_sha256=file_hash(args.prepared_root / 'prepared.json'),
        meta_preparation_sha256=file_hash(args.meta_preparation / 'manifest.json'),
        meta_indices_sha256=context['meta']['artifacts']['meta_train_indices.npz']['sha256'],
        train_val_inputs_sha256=json_hash({s: source['splits'][s] for s in ('train', 'val')}),
        first_stage=context['first_stage'], rows_per_problem=SPLIT_ROWS,
        train_rows_per_problem=META_ROWS, current_train_rows_per_problem=81000,
        heldout_rows_per_problem=4500, meta_train_total=META_ROWS * len(PROBLEMS),
        membership_boundary='source-record novelty only; not exhaustive content/base-dedup; cohort representativeness unproved',
        candidate_pools={p: source['splits']['train'][p]['pool'] for p in PROBLEMS},
        candidate_pool_ids={p: source['splits']['train'][p]['pool_ids'] for p in PROBLEMS},
        scenario_solver_aliases=source['scenario_solver_aliases'], winner_rule=WINNER_RULE,
        baselines=context['baselines'], own_train_sbs=context['baselines'], geometry_stats=context['geometry'],
        common_trainable_sha256=FRESH_SHA, global_batch_size=BATCH_SIZE, logical_global_batch=BATCH_SIZE,
        max_successful_updates=MAX_UPDATES, eval_every_updates=EVAL_EVERY, warmup_updates=WARMUP_UPDATES,
        updates_per_problem=MAX_UPDATES // len(PROBLEMS), loss=LOSS,
        optimizer='AdamW', weight_decay=1e-4, grad_clip=1., dropout=.1,
        lr=dict(peak=1e-4, end=1e-6, warmup_updates=WARMUP_UPDATES, rule='successful-update cosine after warmup'),
        microbatch_size=args.microbatch_size, eval_batch_size=args.eval_batch_size,
        large_microbatch_size=args.large_microbatch_size, microbatch_caps=micro_caps(args),
        eval_batch_caps=micro_caps(args, evaluation=True),
        cache_policy='read-only R77B slice from independent R123 bank; no R124 expert-cache inference or input builds',
        checkpoint_rule=CHECKPOINT_RULE, inference_rule=INFERENCE_RULE, fixed_budget=True, early_stop=False,
        test='one full TEST after complete budget and unique VAL lock; fixed predictions reusable',
        test_rnd_disclosure='Existing TEST participated in R&D; not a fresh blind test',
        science={name: file_hash(repo / name) for name in SCIENCE},
        science_identity='Explicit Astra followup request recorded; actual model/provider identity unverified',
        source_sha256={name: file_hash(V4 / name) for name in SOURCES},
        first_stage_precision='FP32; TF32 disabled; eval; no autocast',
        wandb=True, wandb_mode=args.wandb_mode)


def initialize_fresh(initial, geometry, params):
    if (initial.get('seed') != 2 or initial.get('initialization') != 'scratch'
            or parameter_hash(initial['trainable']) != initial.get('trainable_sha256')):
        raise ValueError('A trained selector checkpoint is not a fresh initialization')
    set_seed(2)
    model = Top2Reranker(**params)
    reference, copied, new = initial['trainable'], [], []
    with torch.no_grad():
        for name, value in model.named_parameters():
            if name in reference and value.shape == reference[name].shape:
                value.copy_(reference[name])
                copied.append(name)
            else:
                new.append(name)
        branch = model.instance_encoder.geometry_residual
        branch.mean.copy_(branch.mean.new_tensor(geometry['mean']))
        branch.std.copy_(branch.std.new_tensor(geometry['std']))
    if any(name.startswith('instance_encoder.') for name in new):
        raise ValueError('Original independent encoder must be fully fresh-compatible')
    return model, dict(initialization='scratch_r84_compatible_new_encoder', seed=2,
        copied_fresh_parameters=copied, newly_initialized_parameters=new,
        common_fresh_sha256=initial['trainable_sha256'], trainable_sha256=parameter_hash(dict(model.named_parameters())),
        trained_selector_initialization=False, first_stage_shared_parameters=False,
        geometry_fitted_on='existing R84 current TRAIN81000 only')


def split_indices(context, problem, split):
    return context['indices'][problem] if split == 'train' else np.arange(SPLIT_ROWS[split], dtype=np.int64)


def feature_root(args, split):
    return args.test_cache_root if split == 'test' else args.cache_root


def validate_score_arrays(data, *, indices, pool, pool_ids, release_source_indices):
    if (data['scores'].shape != (len(indices), len(pool)) or not np.isfinite(data['scores']).all()
            or data['scores'].dtype not in (np.float32, np.float64)):
        raise ValueError('Invalid first-stage score cache')
    for name, expected in (('indices', indices), ('pool', pool), ('pool_ids', pool_ids),
                           ('release_source_indices', release_source_indices)):
        if not np.array_equal(data[name], expected):
            raise ValueError(f'Stage1 cache identity/order mismatch: {name}')


def score_identity(args, context, problem, split, rows):
    if split == 'test':
        authorize_test(args)
    record = context['manifest']['splits'][split][problem]
    root = feature_root(args, split)
    directory = next((p for p in (root / split / problem, root / (problem + split)) if p.is_dir()),
                     root / split / problem)
    fields = ('problem', 'split', 'problem_id', 'rows', 'kind', 'pool', 'pool_ids',
              'scenario_id', 'arrays', 'source_files', 'feature_sources', 'geometry_fields', 'geometry_scales')
    actual = read_json(directory / 'metadata.json')
    semantic = {key: actual[key] for key in fields}
    if semantic != {key: record[key] for key in fields}:
        raise ValueError('Transported input-cache semantics differ from frozen R84 source')
    with np.load(args.data_root / (problem + split) / 'input_metadata.npz', allow_pickle=False) as stream:
        source = stream['source_index'][rows].copy()
    return dict(experiment='R124', schema_version='r124_r77b_scores_v1', expert_id=EXPERT_ID,
        expert_checkpoint_sha256=EXPERT_SHA, normalization='raw', input_only=True, split=split, problem=problem,
        data_root=str(args.data_root), rows=len(rows), pool=record['pool'], pool_ids=record['pool_ids'],
        local_indices_sha256=hashlib.sha256(rows.tobytes()).hexdigest(),
        source_indices_sha256=hashlib.sha256(source.tobytes()).hexdigest(),
        meta_indices_sha256=context['meta']['artifacts']['meta_train_indices.npz']['sha256'] if split == 'train' else None,
        source_input_metadata_semantic_sha256=json_hash(semantic),
        precision='FP32; TF32 disabled; eval; no autocast'), source


def read_score_file(path, identity, rows, source):
    info = read_json(path.with_suffix('.json'))
    if info['identity'] != identity or info['sha256'] != file_hash(path):
        raise ValueError('Frozen first-stage predictions or provenance changed')
    with np.load(path, allow_pickle=False) as stream:
        data = {name: stream[name].copy() for name in stream.files}
    validate_score_arrays(data, indices=rows, pool=identity['pool'], pool_ids=identity['pool_ids'],
                          release_source_indices=source)
    return data


def shared_bank_scores(args, context, problem, split, rows, source):
    """Read only R77B's slice of the R123-owned cache; never write that bank."""
    from . import r123_prediction_cache as bank_api
    if '_shared_bank_context' not in context:
        context['_shared_bank_context'] = bank_api.context(args.prepared_root, input_cache_root=args.cache_root)
    bank_context = context['_shared_bank_context']
    data = bank_api.read_expert(args.shared_bank_root, split, problem, EXPERT_ID, bank_context)
    info = data['metadata']
    record = context['manifest']['splits'][split][problem]
    if (info['checkpoint_sha256'] != EXPERT_SHA or not info['complete']
            or info['meta_preparation_identity'] != context['meta']['preparation_identity']
            or not np.array_equal(data['row_indices'], rows) or not np.array_equal(data['source_indices'], source)
            or data['pool'].tolist() != record['pool'] or data['pool_ids'].tolist() != record['pool_ids']):
        raise ValueError('Independent R123 R77B archive does not match R124 source/model/pool identities')
    return data['logits']


def build_score_cache(args, context, splits=('train', 'val')):
    """Extract existing bank predictions only; never run experts or build inputs."""
    if set(splits) - {'train', 'val'}:
        raise ValueError('Cache extraction is TRAIN/VAL only; TEST uses one full live composite forward')
    if args.shared_bank_root is None:
        raise ValueError('R124 cache requires --shared-bank-root from the independent R123 cache owner')
    root = args.score_root
    root.mkdir(parents=True, exist_ok=True)
    records = {}
    for split in splits:
        records[split] = {}
        for problem in PROBLEMS:
            rows = split_indices(context, problem, split)
            identity, source = score_identity(args, context, problem, split, rows)
            path = root / split / (problem + '.npz')
            if path.exists():
                read_score_file(path, identity, rows, source)
            else:
                scores = shared_bank_scores(args, context, problem, split, rows, source)
                archive = args.shared_bank_root / 'prediction_cache' / split / problem / 'R77B.npz'
                origin = dict(kind='read_only_independent_R123_R77B_archive', archive=str(archive),
                              archive_sha256=file_hash(archive))
                data = dict(scores=scores, indices=rows, release_source_indices=source,
                            pool=np.asarray(identity['pool']), pool_ids=np.asarray(identity['pool_ids'], dtype=np.int64))
                validate_score_arrays(data, indices=rows, pool=identity['pool'], pool_ids=identity['pool_ids'],
                                      release_source_indices=source)
                save_npz(path, data)
                immutable_json(path.with_suffix('.json'), dict(identity=identity, sha256=file_hash(path), origin=origin))
            records[split][problem] = dict(path=str(path), sha256=file_hash(path), identity=identity,
                metadata_sha256=file_hash(path.with_suffix('.json')),
                origin=read_json(path.with_suffix('.json'))['origin'])
            print(f'[R124 cache] {split}/{problem}: {len(rows)} shared-bank {EXPERT_ID} predictions', flush=True)
    return records


def cache(args):
    context = source_context(args)
    immutable_json(args.root / 'preparation/frozen_contract.json', contract(args, context))
    with exclusive(args.score_root / 'writer.lock'):
        records = build_score_cache(args, context)
        return immutable_json(args.score_root / 'manifest.json', dict(schema_version='r124_stage1_cache_v1',
            expert_id=EXPERT_ID, expert_checkpoint_sha256=EXPERT_SHA, splits=records,
            data_root=str(args.data_root), input_features_rebuilt=False, test_read=False))


class MetaTask:
    def __init__(self, args, context, problem, split, *, live_first_stage=False):
        self.rows = split_indices(context, problem, split)
        self.cache = None
        if live_first_stage:
            if split != 'test':
                raise ValueError('Live first-stage evaluation is the post-lock TEST path only')
            authorize_test(args)
            with np.load(args.data_root / (problem + split) / 'input_metadata.npz', allow_pickle=False) as stream:
                self.release_source_indices = stream['source_index'][self.rows].copy()
            self.scores = None
            self.cache = open_cache(feature_root(args, split), split, problem, context['manifest'], args.data_root)
            self.size = len(self.rows)
            return
        identity, source = score_identity(args, context, problem, split, self.rows)
        manifest_path = args.score_root / ('test_manifest.json' if split == 'test' else 'manifest.json')
        if split != 'test' and context.get('score_cache_manifest_sha256') != file_hash(manifest_path):
            raise ValueError('TRAIN/VAL score manifest differs from the frozen training protocol')
        record = read_json(manifest_path)['splits'][split][problem]
        path = args.score_root / split / (problem + '.npz')
        if record['sha256'] != file_hash(path) or record['metadata_sha256'] != file_hash(path.with_suffix('.json')):
            raise ValueError('Stage1 score artifact differs from its pinned manifest')
        data = read_score_file(path, identity, self.rows, source)
        self.scores, self.release_source_indices = data['scores'], source
        self.cache = open_cache(feature_root(args, split), split, problem, context['manifest'], args.data_root)
        self.size = len(self.rows)

    def batch(self, indices, device='cpu'):
        indices = np.asarray(indices, dtype=np.int64)
        rows = self.rows[indices]
        batch = self.cache.batch(rows, device=device)
        # ScenarioCache's FP32 convenience costs must never supervise R124.
        batch['costs'] = torch.from_numpy(self.cache.raw['costs'][rows].copy()).to(device)
        if self.scores is not None:
            batch['stage1_scores'] = torch.from_numpy(self.scores[indices].copy()).to(device)
        return batch

    def close(self):
        self.cache.close()


def describe(args=None):
    return dict(experiment='R124', first_stage='frozen COMPLETE full18 R77B update34776',
        train_rows_per_problem=META_ROWS, train_total=META_ROWS * 18, rows=SPLIT_ROWS,
        seed=2, batch_size=BATCH_SIZE, max_updates=MAX_UPDATES, warmup_updates=WARMUP_UPDATES,
        eval_every=EVAL_EVERY, architecture='independent4layer128; two method ID+34 tokens; 2 bidirectional JointLayers',
        optimizer='AdamW', weight_decay=1e-4, dropout=.1, grad_clip=1.,
        lr=dict(peak=1e-4, end=1e-6, warmup_updates=WARMUP_UPDATES, max_updates=MAX_UPDATES,
                rule='successful-update warmup then cosine'),
        initialization='R84 certified scratch-compatible independent weights; new heads seed2; never trained-selector initialization',
        native_head='shared role-equivariant scoring; difference is antisymmetric in eval',
        quality_head='separate shared scoring; antisymmetric cost-quality difference',
        loss=LOSS, checkpoint_rule=CHECKPOINT_RULE, inference_rule=INFERENCE_RULE,
        model_factory=MODEL_FACTORY, membership='source-record certificate only, not content/base deduplication',
        microbatch_caps=micro_caps(args) if args is not None else None,
        deployment_cost='one complete frozen R77B plus one independent reranker; not readout-only',
        eval_batch_caps=micro_caps(args, evaluation=True) if args is not None else None,
        science_identity='explicit Astra request; returned identity unverified', gpu_or_solver_launched=False)


def update_batch(model, optimizer, scaler, batch, device, cap, max_retries=20):
    """One logical batch, one clip/step; overflow replays identical rows and RNG."""
    count = len(batch['ind'])
    if count <= 0 or cap <= 0:
        raise ValueError('Positive actual rows and microbatch cap required')
    rng = capture_rank_rng(device)
    for retry in range(max_retries):
        restore_rank_rng(rng, device)
        optimizer.zero_grad(set_to_none=True)
        totals = dict.fromkeys(('native_ce', 'pair_quality_huber', 'covered', 'loss'), 0.)
        try:
            for start in range(0, count, cap):
                micro = slice_batch(batch, start, min(start + cap, count))
                micro = {k: v.to(device) if torch.is_tensor(v) else v for k, v in micro.items()}
                with torch.autocast(device_type=device.type, enabled=device.type == 'cuda'):
                    output = model(query_inputs(micro), micro['stage1_scores'])
                loss, parts = pair_objective(output, micro['costs'], micro['ind'], row_denominator=count)
                scaler.scale(loss).backward()
                for name in totals:
                    totals[name] += float(parts[name])
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            finite = math.isfinite(totals['loss']) and bool(torch.isfinite(grad_norm))
            if not finite:
                if scaler.is_enabled():
                    scaler.update(new_scale=scaler.get_scale() * .5)
                continue
            before = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if scaler.is_enabled() and scaler.get_scale() < before:
                continue
            totals.update(grad_norm=float(grad_norm), covered_fraction=totals.pop('covered') / count)
            return totals, retry, cap
        except torch.cuda.OutOfMemoryError:
            optimizer.zero_grad(set_to_none=True)
            if device.type != 'cuda' or cap == 1:
                raise
            if scaler.is_enabled():
                scaler.update(new_scale=scaler.get_scale())
            cap = max(1, cap // 2)
            torch.cuda.empty_cache()
    raise RuntimeError('No successful finite optimizer update; no rows may be skipped')


def selection_key(result, update):
    risk, accuracy = result['macro']['macro_actual_regret_pct'], result['macro']['macro_top1']
    if not math.isfinite(risk) or not math.isfinite(accuracy):
        raise ValueError('Complete VAL selection metrics must be finite')
    return float(risk), -float(accuracy), int(update)


def top2_statistics(data):
    scores = torch.from_numpy(data['stage1_scores'])
    ids = torch.from_numpy(data['pool_ids'])[None].expand_as(scores)
    legal = torch.ones_like(scores, dtype=torch.bool)
    pair = top2_indices(scores, ids, legal).numpy()
    winner, costs = data['winner'], data['costs']
    final = torch.from_numpy(data['logits'])
    pred = top2_indices(final, ids, legal)[:, 0].numpy()
    covered = (pair == winner[:, None]).any(1)
    base_pred = pair[:, 0]
    row = np.arange(len(winner))
    optimum = costs.min(1)
    pair_cost = np.take_along_axis(costs, pair, axis=1).min(1)
    corrected = (base_pred != winner) & (pred == winner)
    damaged = (base_pred == winner) & (pred != winner)
    return dict(n=len(winner), native_covered=int(covered.sum()),
        native_pair_correct=int((pred == winner).sum()), native_top2_coverage=float(covered.mean()),
        conditional_native_top1=float((pred == winner).sum() / covered.sum()) if covered.any() else None,
        first_stage_top1=float((base_pred == winner).mean()), final_top1=float((pred == winner).mean()),
        corrected=int(corrected.sum()), damaged=int(damaged.sum()),
        correction_rate=float(corrected.mean()), damage_rate=float(damaged.mean()),
        net_top1_change=float(corrected.mean() - damaged.mean()),
        recall_floor_regret_pct=float(np.mean((pair_cost - optimum) / optimum) * 100.),
        reranking_excess_regret_pct=float(np.mean((costs[row, pred] - pair_cost) / optimum) * 100.))


@torch.no_grad()
def evaluate(model, args, context, split, tag, update, state_hash, checkpoint_sha256=None):
    if split == 'test':
        authorize_test(args)
    root = args.root / 'model'
    directory = root / 'evaluation' / split / tag
    directory.mkdir(parents=True, exist_ok=True)
    identity = dict(experiment='R124', arm='model', split=split, tag=tag, successful_updates=update,
        state_hash=state_hash, model_state_sha256=parameter_hash(model.state_dict()),
        checkpoint_sha256=checkpoint_sha256, first_stage_checkpoint_sha256=EXPERT_SHA, winner_rule=WINNER_RULE)
    immutable_json(directory / 'identity.json', identity)
    if (directory / 'metrics.json').exists() and (directory / 'rank000.npz').exists():
        return read_json(directory / 'metrics.json')
    was_training, device = model.training, next(model.parameters()).device
    live_first_stage = isinstance(model, FrozenTop2Selector)
    if split == 'test' and not live_first_stage:
        raise ValueError('Formal TEST must time the full frozen-stage0 plus reranker composite')
    model.eval()
    arrays, per_problem, pair_results = {}, {}, {}
    started = time.monotonic()
    try:
        for problem in PROBLEMS:
            shard = directory / 'by_problem' / (problem + '.npz')
            if shard.exists():
                with np.load(shard, allow_pickle=False) as stream:
                    data = {k: stream[k].copy() for k in stream.files}
            else:
                task = MetaTask(args, context, problem, split, live_first_stage=live_first_stage)
                try:
                    values = {k: [] for k in ('logits', 'pair_indices', 'difference', 'pair_quality')}
                    stage1, forward_seconds = [], 0.
                    cap = micro_caps(args, evaluation=True)[problem]
                    for start in range(0, task.size, cap):
                        batch = task.batch(np.arange(start, min(start + cap, task.size)), device)
                        if device.type == 'cuda':
                            torch.cuda.synchronize(device)
                        before = time.perf_counter()
                        output = (model(query_inputs(batch)) if live_first_stage
                                  else model(query_inputs(batch), batch['stage1_scores']))
                        if device.type == 'cuda':
                            torch.cuda.synchronize(device)
                        forward_seconds += time.perf_counter() - before
                        for key in values:
                            values[key].append(output[key].cpu().numpy())
                        if live_first_stage:
                            stage1.append(output['stage1_scores'].cpu().numpy())
                    data = {k: np.concatenate(v) for k, v in values.items()}
                    data.update(indices=np.arange(task.size, dtype=np.int64), source_indices=task.rows,
                        release_source_indices=task.release_source_indices,
                        stage1_scores=np.concatenate(stage1) if live_first_stage else task.scores,
                        costs=task.cache.raw['costs'][task.rows].copy(), winner=task.cache.raw['winner'][task.rows].copy(),
                        pool=np.asarray(task.cache.pool), pool_ids=np.asarray(task.cache.pool_ids, dtype=np.int64),
                        nodes=task.cache.lengths[task.rows].copy(), forward_seconds=np.asarray(forward_seconds),
                        forward_batch_cap=np.asarray(cap), full_pipeline_timed=np.asarray(live_first_stage))
                    save_npz(shard, data)
                finally:
                    task.close()
            count = META_ROWS if split == 'train' else SPLIT_ROWS[split]
            if (len(data['logits']) != count or data['costs'].dtype != np.float64
                    or data['winner'].dtype != np.int64
                    or not np.array_equal(data['indices'], np.arange(count))):
                raise ValueError(f'Incomplete native {split}/{problem} evaluation')
            raw = {k: data[k] for k in ('costs', 'winner', 'pool', 'pool_ids')}
            raw['pool'], raw['pool_ids'] = raw['pool'].tolist(), raw['pool_ids'].tolist()
            per_problem[problem], data['pred'] = fixed_sbs_metrics(data['logits'], raw, context['baselines'][problem])
            per_problem[problem]['inference_cost'] = dict(forward_seconds=float(data['forward_seconds']),
                rows=count, milliseconds_per_instance=1000. * float(data['forward_seconds']) / count,
                batch_cap=int(data['forward_batch_cap']), full_stage0_plus_reranker=bool(data['full_pipeline_timed']),
                scope='complete frozen R77B + independent reranker' if live_first_stage else 'reranker only; stage0 bank precomputed',
                excludes='data loading, host-to-device transfer and solvers', precision='FP32; TF32 off; no autocast',
                device=str(device), measurement='actual first-pass batched forward; no synthetic warmup; CUDA synchronized')
            pair_results[problem] = top2_statistics(data)
            arrays.update({problem + '__' + k: value for k, value in data.items()})
            print(f'[R124 {split}] {tag} {problem} Top1={per_problem[problem]["top1"]:.6f}', flush=True)
    finally:
        model.train(was_training)
    save_npz(directory / 'rank000.npz', arrays)
    result = summarize(per_problem)
    pair_macro = {key: float(np.mean([row[key] for row in pair_results.values()])) for key in
        ('native_top2_coverage', 'first_stage_top1', 'final_top1', 'correction_rate', 'damage_rate',
         'net_top1_change', 'recall_floor_regret_pct', 'reranking_excess_regret_pct')}
    coverage = pair_macro['native_top2_coverage']
    pair_macro['conditional_native_top1'] = pair_macro['final_top1'] / coverage if coverage else None
    pair_macro['conditional_rule'] = 'macro joint correct / macro coverage, NOT mean task conditional rates'
    result.update(seconds=time.monotonic() - started, tag=tag, successful_updates=update,
        sample_exposures=update * BATCH_SIZE, identity=identity,
        prediction_archive=str(directory / 'rank000.npz'),
        top2_reranking=dict(per_problem=pair_results, macro=pair_macro),
        inference_cost=dict(full_stage0_plus_reranker=live_first_stage,
            forward_seconds=sum(v['inference_cost']['forward_seconds'] for v in per_problem.values()),
            rows=sum(v['n'] for v in per_problem.values()),
            macro_milliseconds_per_instance=float(np.mean([
                v['inference_cost']['milliseconds_per_instance'] for v in per_problem.values()])),
            scope='full composite deployment inference' if live_first_stage else 'cached-stage0 reranker evaluation, NOT deployment cost',
            bank_precomputation_not_free=True),
        training_population='source-record-certified metaTRAIN10886 per task' if split == 'train' else None)
    write_json(directory / 'metrics.json', result)
    return result


def validate_config(config):
    expected = dict(experiment='R124', arm='model', seed=2, initialization='scratch_r84_compatible_new_encoder',
        solver_names=list(GLOBAL_SOLVERS), max_successful_updates=MAX_UPDATES, global_batch_size=BATCH_SIZE,
        logical_global_batch=BATCH_SIZE, warmup_updates=WARMUP_UPDATES, eval_every_updates=EVAL_EVERY,
        checkpoint_rule=CHECKPOINT_RULE, inference_rule=INFERENCE_RULE, loss=LOSS)
    if any(config.get(k) != v for k, v in expected.items()):
        raise ValueError('Checkpoint is not the fixed R124 protocol')
    if (config['first_stage']['checkpoint_sha256'] != EXPERT_SHA
            or config['first_stage']['updated_by_r124'] is not False
            or set(config['candidate_pools']) != set(PROBLEMS)):
        raise ValueError('Frozen first stage/full18 candidate support changed')


def load_reranker(path, device='cpu'):
    state = torch.load(path, map_location='cpu', weights_only=False)
    config = state['args']
    validate_config(config)
    if state['state_hash'] != json_hash(config):
        raise ValueError('Checkpoint protocol hash changed')
    model = Top2Reranker(**config['model_params'])
    model.load_state_dict(state['model'], strict=True)
    return model.to(device), state


def load_checkpoint(path, device='cpu'):
    """LIB-compatible factory, with frozen first-stage inference included."""
    configure_precision()
    reranker, state = load_reranker(path, device)
    if file_hash(Path(state['args']['first_stage']['checkpoint_path'])) != EXPERT_SHA:
        raise ValueError('External first-stage weights changed')
    first_stage, _ = load_expert(EXPERT_ID, device)
    return FrozenTop2Selector(reranker, first_stage).to(device).eval(), state


load_model = load_checkpoint
from_checkpoint = load_checkpoint


def validate_resume(state, config, order):
    step = state['successful_updates']
    pending = state.get('pending_evaluation') is not None
    if (state['args'] != config or state['state_hash'] != json_hash(config)
            or not 0 <= step <= MAX_UPDATES or step % EVAL_EVERY
            or step != (len(state['history']) + int(pending)) * EVAL_EVERY):
        raise ValueError('Resume only the unchanged R124 interval-boundary state')
    counts = Counter(order[:step])
    if (Counter(state['updates_by_problem']) != counts
            or any(state['stream_draws'][p] != counts[p] * BATCH_SIZE for p in PROBLEMS)):
        raise ValueError('Resume stream cursors disagree with successful updates')


def status(args, phase, step, **values):
    write_json(args.root / 'model/status.json', dict(experiment='R124', arm='model', phase=phase,
        successful_updates=step, max_successful_updates=MAX_UPDATES, updated_utc=utc(), **values))


def train(args):
    root = args.root / 'model'
    if (root / 'training_complete.json').exists():
        authorize_test(args)
        return read_json(root / 'training_complete.json')
    context = source_context(args)
    config = contract(args, context)
    immutable_json(args.root / 'preparation/frozen_contract.json', config)
    score_manifest = args.score_root / 'manifest.json'
    if not score_manifest.exists():
        raise FileNotFoundError('Run the R124 cache CLI before training')
    config = dict(config, score_cache_manifest_sha256=file_hash(score_manifest))
    context['score_cache_manifest_sha256'] = config['score_cache_manifest_sha256']
    validate_config(config)
    immutable_json(root / 'protocol.json', config)
    order, state_hash = task_order(MAX_UPDATES, problems=PROBLEMS), json_hash(config)
    last = root / 'last.pt'
    if last.exists() and not args.resume:
        raise FileExistsError('Preserve existing run; use train --resume')
    device = torch.device(args.device)
    initial = torch.load(args.prepared_root / 'identicalinit_seed2.pt', map_location='cpu', weights_only=False)
    model, initialization = initialize_fresh(initial, context['geometry'], context['params'])
    del initial
    model = model.to(device)
    checkpoint = None
    if last.exists():
        model, checkpoint = load_reranker(last, device)
        validate_resume(checkpoint, config, order)
        initialization = checkpoint['initialization']
    else:
        immutable_json(root / 'initialization.json', initialization)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scaler = torch.amp.GradScaler('cuda', enabled=device.type == 'cuda')
    history, best, step, retries, pending, counts = [], None, 0, 0, None, Counter()
    caps = micro_caps(args)
    cursors = dict.fromkeys(PROBLEMS, 0)
    if checkpoint is not None:
        optimizer.load_state_dict(checkpoint['optimizer'])
        scaler.load_state_dict(checkpoint['scaler'])
        history, best, step = checkpoint['history'], checkpoint['best'], checkpoint['successful_updates']
        retries, pending = checkpoint['amp_retries'], checkpoint.get('pending_evaluation')
        counts, cursors = Counter(checkpoint['updates_by_problem']), checkpoint['stream_draws']
        caps = checkpoint.get('effective_microbatch_caps', caps)
        restore_rank_rng(checkpoint['rng'], device)
    else:
        set_seed(2)
    del checkpoint
    streams = {p: ShuffleStream(META_ROWS, task_seed(2, p), cursors[p]) for p in PROBLEMS}
    tasks, tracker = {}, None

    def snapshot():
        return dict(model=model.state_dict(), optimizer=optimizer.state_dict(), scaler=scaler.state_dict(),
            args=config, state_hash=state_hash, initialization=initialization, successful_updates=step,
            epoch=step // EVAL_EVERY, sample_exposures=step * BATCH_SIZE, updates_by_problem=dict(counts),
            stream_draws={p: stream.draws for p, stream in streams.items()}, rng=capture_rank_rng(device),
            history=history, best=best, amp_retries=retries, pending_evaluation=pending,
            effective_microbatch_caps=dict(caps))

    if not last.exists():
        save_checkpoint(last, snapshot())
    try:
        tracker = wandb_run(args, config)
        restore_rank_rng(torch.load(last, map_location='cpu', weights_only=False)['rng'], device)
        while step < MAX_UPDATES or pending is not None:
            if pending is None:
                first, end, started = step, min(MAX_UPDATES, step + EVAL_EVERY), time.monotonic()
                sums = Counter()
                model.train()
                while step < end:
                    problem = order[step]
                    if problem not in tasks:
                        tasks[problem] = MetaTask(args, context, problem, 'train')
                    batch = tasks[problem].batch(streams[problem].take(BATCH_SIZE).numpy())
                    lr = learning_rate(step + 1)
                    for group in optimizer.param_groups:
                        group['lr'] = lr
                    parts, retried, cap = update_batch(model, optimizer, scaler, batch, device, caps[problem])
                    caps[problem] = cap
                    step += 1
                    counts[problem] += 1
                    retries += retried
                    sums.update(parts)
                    if step == 1 or step % args.log_every == 0 or step == end:
                        info = dict(problem=problem, lr=lr, amp_retries=retries, effective_microbatch=cap,
                            seconds_per_update=(time.monotonic() - started) / (step - first),
                            **{k: v / (step - first) for k, v in sums.items()})
                        status(args, 'training', step, **info)
                        with (root / 'updates.jsonl').open('a') as stream:
                            stream.write(json.dumps(dict(successful_updates=step, **info), allow_nan=False) + '\n')
                        tracker.log(dict(successful_updates=step, **info), step=step)
                pending = dict(seconds=time.monotonic() - started, **{k: v / (step - first) for k, v in sums.items()})
                save_checkpoint(last, snapshot())
            tag = f'update{step:07d}'
            validation = evaluate(model, args, context, 'val', tag, step, state_hash)
            row = dict(epoch=step // EVAL_EVERY, evaluation_tag=tag, successful_updates=step,
                sample_exposures=step * BATCH_SIZE, optimization=pending, val=validation, train=None)
            improved = best is None or selection_key(validation, step) < selection_key(best['val'], best['successful_updates'])
            history.append(row)
            if improved:
                best = row
            pending = None
            if improved:
                save_checkpoint(root / 'best.pt', snapshot())
                write_json(root / 'best_eval.json', best)
            save_checkpoint(last, snapshot())
            write_json(root / 'history.json', history)
            status(args, 'validation_complete', step, selected_update=best['successful_updates'], val=validation['macro'])
            tracker.log(dict(successful_updates=step, **{'val/' + k: v for k, v in validation['macro'].items()}), step=step)
        if step != MAX_UPDATES or Counter(counts) != Counter(dict.fromkeys(PROBLEMS, MAX_UPDATES // len(PROBLEMS))):
            raise ValueError('Complete the single full18 successful-update budget before locking')
        selected, saved = load_reranker(root / 'best.pt', device)
        if saved['successful_updates'] != best['successful_updates'] or saved['args'] != config:
            raise ValueError('Selected model does not match complete VAL history')
        lock = lock_best(root, best, config, state_hash)
        selected_val = evaluate(selected, args, context, 'val', 'selected_best', lock['selected_update'], state_hash,
                                lock['checkpoint_sha256'])
        if selected_val['macro'] != best['val']['macro']:
            raise ValueError('Locked selected VAL metrics differ from checkpoint selection')
        selected_train = evaluate(selected, args, context, 'train', 'selected_best', lock['selected_update'], state_hash,
                                  lock['checkpoint_sha256'])
        final = dict(epoch=lock['epoch'], selected_update=lock['selected_update'], checkpoint_sha256=lock['checkpoint_sha256'],
            state_hash=state_hash, train=selected_train, val=selected_val, test_read=False)
        write_json(root / 'final_metrics.json', final)
        write_json(root / 'training_complete.json', dict(completed_utc=utc(), successful_updates=step,
            updates_by_problem=dict(counts), selected_best=lock, state_hash=state_hash, test_read=False))
        write_json(root / 'result.json', dict(best=best, final=history[-1], successful_updates=step,
            updates_by_problem=dict(counts), state_hash=state_hash, test_read=False))
        status(args, 'training_complete', step, selected_update=lock['selected_update'], next_stage='explicit test CLI')
        return final
    finally:
        for task in tasks.values():
            task.close()
        if tracker is not None:
            tracker.finish()


def authorize_test(args):
    """Validate the full global VAL choice BEFORE any TEST metadata/features."""
    root = args.root / 'model'
    try:
        complete = read_json(root / 'training_complete.json')
        lock = read_json(root / 'final_test_lock.json')
        config = read_json(root / 'protocol.json')
        history = read_json(root / 'history.json')
    except FileNotFoundError as error:
        raise PermissionError('TEST requires completed R124 training and unique full18 VAL lock') from error
    validate_config(config)
    if (complete.get('successful_updates') != MAX_UPDATES
            or complete.get('updates_by_problem') != dict.fromkeys(PROBLEMS, MAX_UPDATES // len(PROBLEMS))
            or complete.get('state_hash') != json_hash(config)
            or lock.get('experiment') != 'R124' or lock.get('arm') != 'model'
            or lock.get('selected_without_test') is not True or lock.get('per_problem_assembly') is not False
            or lock.get('state_hash') != json_hash(config) or lock.get('inference_rule') != INFERENCE_RULE
            or Path(lock['checkpoint']).resolve() != (root / 'best.pt').resolve()
            or lock.get('checkpoint_sha256') != file_hash(root / 'best.pt')
            or Path(config['data_root']).resolve() != args.data_root.resolve()
            or len(history) != MAX_UPDATES // EVAL_EVERY
            or [r['successful_updates'] for r in history] != list(range(EVAL_EVERY, MAX_UPDATES + 1, EVAL_EVERY))):
        raise PermissionError('Changed/incomplete R124 global checkpoint or inference lock')
    chosen = min(history, key=lambda r: selection_key(r['val'], r['successful_updates']))
    if chosen['successful_updates'] != lock['selected_update'] or chosen['val']['macro'] != lock['val']:
        raise PermissionError('Lock is not the fixed strict-minimum full18 VAL selection')
    if args.state_hash is not None and args.state_hash != lock['state_hash']:
        raise PermissionError('Expected state hash differs from R124 lock')
    return lock, config


def ensure_test_features(args, context):
    """Use only the parent's already-built/transported post-lock TEST inputs."""
    authorize_test(args)
    shared = args.cache_root
    if args.test_cache_root is None:
        args.test_cache_root = shared
    test_records = {}
    for problem in PROBLEMS:
        path = args.test_cache_root / 'test' / problem / 'metadata.json'
        if not path.exists():
            raise FileNotFoundError('Parent must transport the existing TEST input cache; R124 never rebuilds inputs: ' + str(path))
        record = read_json(path)
        if (not record.get('complete') or record['rows'] != 4500
                or record['pool'] != context['manifest']['splits']['train'][problem]['pool']
                or record['pool_ids'] != context['manifest']['splits']['train'][problem]['pool_ids']):
            raise ValueError('Post-lock TEST features must retain frozen4500 and exact legal pools')
        test_records[problem] = record
    context['manifest']['splits']['test'] = test_records
    # Validate local arrays using the unchanged native cache reader.
    for problem in PROBLEMS:
        task = open_cache(args.test_cache_root, 'test', problem, context['manifest'], args.data_root)
        task.close()
    immutable_json(args.root / 'preparation/test_input_receipt.json', dict(
        selected_checkpoint_sha256=read_json(args.root / 'model/final_test_lock.json')['checkpoint_sha256'],
        test_cache_root=str(args.test_cache_root), splits=dict(test=test_records)))
    return context


def test(args):
    lock, config = authorize_test(args)
    root = args.root / 'model'
    result_path = root / 'test_results.json'
    if result_path.exists():
        saved = read_json(result_path)
        if (not saved.get('complete') or saved.get('checkpoint_sha256') != lock['checkpoint_sha256']
                or saved.get('state_hash') != lock['state_hash']):
            raise PermissionError('Saved TEST belongs to another checkpoint/configuration')
        finalize_test(args, saved['result'], lock)
        return saved['result']
    for name in ('prepared_root', 'score_root', 'meta_preparation'):
        setattr(args, name, Path(config[name]))
    if args.cache_root is None:
        args.cache_root = Path(config['cache_root'])
    args.eval_batch_size = config['eval_batch_size']
    args.large_microbatch_size = config['large_microbatch_size']
    context = ensure_test_features(args, source_context(args))
    context['score_cache_manifest_sha256'] = config['score_cache_manifest_sha256']
    with exclusive(root / 'test.lock'):
        model, state = load_checkpoint(root / 'best.pt', args.device)
        if state['args'] != config or state['successful_updates'] != lock['selected_update']:
            raise PermissionError('TEST restored a different comparator')
        immutable_json(root / 'test_started.json', dict(checkpoint_sha256=lock['checkpoint_sha256'],
            state_hash=lock['state_hash'], selected_update=lock['selected_update']))
        result = evaluate(model, args, context, 'test', 'selected_best', lock['selected_update'],
                          lock['state_hash'], lock['checkpoint_sha256'])
        authorize_test(args)
        write_json(result_path, dict(complete=True, completed_utc=utc(), result=result,
            epoch=lock['epoch'], selected_update=lock['selected_update'],
            checkpoint_sha256=lock['checkpoint_sha256'], state_hash=lock['state_hash']))
        finalize_test(args, result, lock)
        return result


def finalize_test(args, result, lock):
    root = args.root / 'model'
    final = read_json(root / 'final_metrics.json')
    final.update(test=result, test_read=True, complete=True)
    write_json(root / 'final_metrics.json', final)
    write_json(root / 'test_complete.json', dict(full_tests=1, completed_utc=utc(),
        checkpoint_sha256=lock['checkpoint_sha256'], saved_predictions_reusable=True))
    status(args, 'test_complete', MAX_UPDATES, selected_update=lock['selected_update'], test=result['macro'])


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('command', choices=('cache', 'train', 'test', 'describe'))
    result.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    result.add_argument('--prepared-root', type=Path, default=DEFAULT_PREPARED_ROOT)
    result.add_argument('--data-root', type=Path, default=DEFAULT_DATA_ROOT)
    result.add_argument('--cache-root', type=Path)
    result.add_argument('--meta-preparation', type=Path, default=META_PREPARATION)
    result.add_argument('--score-root', type=Path, default=SCORE_ROOT)
    result.add_argument('--shared-bank-root', type=Path, default=META_PREPARATION.parent,
                        help='Independent R123 root containing prediction_cache/SPLIT/PROBLEM/R77B.npz')
    result.add_argument('--test-cache-root', type=Path)
    result.add_argument('--device', default='cuda:0')
    result.add_argument('--microbatch-size', type=int, default=1024, help='Nonlarge TRAIN cap; logical batch stays1024')
    result.add_argument('--large-microbatch-size', type=int, default=128, help='TSP/CVRP cap; choose128 or192 before freeze')
    result.add_argument('--eval-batch-size', type=int, default=512, help='Nonlarge eval cap; TSP/CVRP capped by large-microbatch-size')
    result.add_argument('--log-every', type=int, default=18)
    result.add_argument('--wandb-mode', choices=('auto', 'online', 'offline'), default='auto')
    result.add_argument('--resume', action='store_true')
    result.add_argument('--state-hash', '--expected-state-hash')
    result.set_defaults(experiment='R124', arm='model')
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    for name in ('root', 'prepared_root', 'data_root', 'cache_root', 'meta_preparation', 'score_root', 'test_cache_root',
                 'shared_bank_root'):
        value = getattr(args, name)
        if value is not None:
            setattr(args, name, value.expanduser().resolve())
    if (min(args.microbatch_size, args.large_microbatch_size, args.eval_batch_size, args.log_every) <= 0
            or max(args.microbatch_size, args.large_microbatch_size) > BATCH_SIZE):
        raise ValueError('Positive batch/log sizes required; microbatch <= logical1024')
    torch.set_num_threads(2)
    if args.command == 'describe':
        value = describe(args)
        print(json.dumps(value, indent=2))
        return value
    configure_precision()
    if args.command == 'cache':
        return cache(args)
    with exclusive(args.root / 'model/experiment.lock'):
        return train(args) if args.command == 'train' else test(args)


if __name__ == '__main__':
    main()
