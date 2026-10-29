"""R123: full18 input-only Brier reliability, then one frozen complete expert.

Explicit commands: describe, cache-bank, train, test. Cache generation and
training are not performed at import. TEST requires this run's final VAL lock.
"""

from __future__ import annotations

import argparse
import fcntl
import gc
import math
from pathlib import Path

import numpy as np
import torch

from code.unified_selector.registry import GLOBAL_SOLVERS, PROBLEMS
from .multitask_probe import task_seed
from .r76_cache import ScenarioCache
from .r76_experiment import (capture_rank_rng, fixed_sbs_metrics, restore_rank_rng,
                             save_checkpoint, utc, write_json)
from .r77_data import parameter_hash
from .r77_experiment import ShuffleStream, task_order
from .r84_experiment import summarize
from .r84_prepare import DEFAULT_PREPARED_ROOT, SPLIT_ROWS
from .r123_meta_data import (CURRENT_ROOT, DEFAULT_EXPERTS, PREPARATION,
    json_hash, load_expert, load_meta_indices, model_inputs, read_json, sha256, source_record_ids)
from .r123_model import HistoricalModelRouter, ReliabilityGate, brier_loss, correctness_targets, gate_inputs
from .train import set_seed


DEFAULT_ROOT = PREPARATION / 'learner_run'
DEFAULT_BANK_ROOT = PREPARATION.parent
MAX_UPDATES, BATCH_SIZE, WARMUP, EVAL_EVERY = 7200, 1024, 540, 720
FACTORY = 'code.V4.r123_experiment.load_checkpoint'
WINNER_RULE = 'saved native int64 winner, original positive FP64 costs; never FP32 argmin'
CACHE_SCHEMA = 'historical_complete_bank_v1'


def source_context(prepared_root=DEFAULT_PREPARED_ROOT, meta_preparation=PREPARATION):
    prepared_root, meta_preparation = Path(prepared_root).resolve(), Path(meta_preparation).resolve()
    meta = read_json(meta_preparation / 'manifest.json')
    ready = read_json(prepared_root / 'prepared.json')
    cm = read_json(prepared_root / 'source_cache_manifest.json')
    if (meta['selected_experts'] != list(DEFAULT_EXPERTS)
            or not meta['meta_training_ready_at_source_record_scope']
            or set(meta['problems']) != set(PROBLEMS)
            or Path(ready['data_root']).resolve() != CURRENT_ROOT.resolve()
            or ready['trainable_sha256'] != 'dad87d8598f7560e6f170ef048ba1bf515edeb03cef7ec42e16839a36af4ef93'):
        raise ValueError('Require the fixed complete historical bank and frozen current R84 scratch preparation')
    for name, digest in ready['artifact_sha256'].items():
        if sha256(prepared_root / name) != digest:
            raise ValueError('Frozen R84 TRAIN-only artifact changed: ' + name)
    for split in ('train', 'val'):
        if set(cm['splits'][split]) != set(PROBLEMS):
            raise ValueError('Complete full18 TRAIN/VAL input caches required')
        for p in PROBLEMS:
            r, certified = cm['splits'][split][p], meta['problems'][p]
            if (r['rows'] != SPLIT_ROWS[split] or r['pool'] != certified['candidates']
                    or r['pool_ids'] != certified['global_solver_ids']
                    or certified['certified_source_unseen_count'] != 10886):
                raise ValueError('Frozen pool or source-record membership changed: ' + p)
    bank = []
    for key in DEFAULT_EXPERTS:
        r = meta['experts'][key]
        if sha256(r['checkpoint_path']) != r['checkpoint_sha256']:
            raise ValueError('Historical frozen expert changed: ' + key)
        bank.append(dict(expert_id=key, checkpoint_path=r['checkpoint_path'],
                         checkpoint_sha256=r['checkpoint_sha256'], factory=r['factory']))
    frozen_sources = {path: record['sha256'] for path, record in meta['evidence_files'].items()
                      if Path(path).suffix == '.py'}
    for path, digest in frozen_sources.items():
        if sha256(path) != digest:
            raise ValueError('Historical model/input source changed since certification: ' + path)
    train_val = {s: {p: {k: cm['splits'][s][p][k] for k in
                         ('rows', 'pool', 'pool_ids', 'source_files')} for p in PROBLEMS}
                 for s in ('train', 'val')}
    identity = json_hash(dict(meta_manifest_sha256=sha256(meta_preparation / 'manifest.json'),
        artifacts=ready['artifact_sha256'], bank=bank, train_val=train_val))
    return dict(prepared_root=prepared_root, meta_preparation=meta_preparation,
        meta=meta, ready=ready, cache_manifest=cm, bank=bank, identity=identity,
        frozen_sources=frozen_sources,
        geometry=read_json(prepared_root / 'geometry_stats.json'),
        baselines=read_json(prepared_root / 'common_full_train_sbs.json'))


def expected_rows(context, split, problem):
    if split == 'train':
        return load_meta_indices(problem, context['meta_preparation'])
    if split not in ('val', 'test'):
        raise ValueError('Bank split must be train, val or test')
    return np.arange(SPLIT_ROWS[split], dtype=np.int64)


def test_permission(run_root):
    if run_root is None:
        raise PermissionError('An explicit completed R123 run lock is required for TEST')
    run_root = Path(run_root)
    lock_path = run_root / 'final_test_lock.json'
    if not lock_path.is_file():
        raise PermissionError('R123 TEST requires a unique completed-run full18 VAL lock')
    lock = read_json(lock_path)
    state = torch.load(run_root / 'best.pt', map_location='cpu', weights_only=False)
    if (lock.get('experiment') != 'R123' or lock.get('completed_updates') != MAX_UPDATES
            or lock.get('checkpoint_sha256') != sha256(run_root / 'best.pt')
            or lock.get('protocol_sha256') != json_hash(read_json(run_root / 'protocol.json'))
            or state['state_hash'] != lock['protocol_sha256']
            or state['successful_updates'] != lock['selected_update']
            or lock.get('selected_without_test') is not True):
        raise PermissionError('R123 final lock/checkpoint/protocol mismatch')
    return lock


def open_source(context, split, problem, *, run_root=None, input_cache_root=None):
    if split == 'test':
        test_permission(run_root)
        cm = read_json(context['prepared_root'] / 'source_cache_manifest.json')
    else:
        cm = context['cache_manifest']
    try:
        record = cm['splits'][split][problem]
    except KeyError as error:
        raise FileNotFoundError('Parent must prepare the existing R84 TEST cache; R123 never regenerates inputs') from error
    directory = Path(record['directory'])
    if input_cache_root is not None:
        directory = Path(input_cache_root) / directory.relative_to(context['ready']['cache_source'])
    cache = ScenarioCache(directory)
    expected = context['meta']['problems'][problem]
    if (cache.split != split or cache.problem != problem or cache.size != SPLIT_ROWS[split]
            or cache.pool != expected['candidates'] or cache.pool_ids != expected['global_solver_ids']
            or cache.metadata['source_files'] != record['source_files']):
        cache.close()
        raise ValueError('Input cache/pool differs from certified current data')
    return cache


def bank_spec(context, split, problem, rows):
    path = CURRENT_ROOT / (problem + split) / 'input_metadata.npz'
    metadata_digest = sha256(path)
    if split == 'train' and metadata_digest != context['meta']['evidence_files'][str(path.resolve())]['sha256']:
        raise ValueError('Current TRAIN identity metadata differs from the immutable source certificate')
    with np.load(path, allow_pickle=False) as stream:
        source = stream['source_index'][rows].copy()
    catalogue = read_json(context['meta_preparation'] / 'source_catalogues.json')[problem]
    generation_path = Path(catalogue['generation_metadata'])
    evidence = context['meta']['evidence_files'][str(generation_path.resolve())]
    if sha256(generation_path) != evidence['sha256']:
        raise ValueError('Frozen generator identity changed')
    generation = read_json(generation_path)
    qualified = source_record_ids(problem, source, generation)
    record = context['meta']['problems'][problem]
    spec = dict(schema_version=CACHE_SCHEMA, split=split, problem=problem,
        context_identity=context['identity'], bank=context['bank'],
        expert_ids=list(DEFAULT_EXPERTS), pool=record['candidates'], pool_ids=record['global_solver_ids'],
        input_metadata_sha256=metadata_digest, row_indices_sha256=sha256_array(rows),
        source_indices_sha256=sha256_array(source), source_identity_sha256=sha256_array(qualified),
        rows=len(rows), shape=[len(rows), len(DEFAULT_EXPERTS), len(record['candidates'])],
        precision='FP32; TF32 disabled; eval; no autocast',
        score_ties='ascending global solver ID', labels_stored=False,
        source_certificate_scope='source record only; no exhaustive base/content-copy guarantee')
    return spec, source, qualified


def sha256_array(array):
    import hashlib
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


class BankPredictionCache:
    """Read-only portable bank scores, explicitly keyed by current row identities."""

    def __init__(self, directory, expected, rows, *, allow_incomplete=False):
        self.directory = Path(directory)
        self.metadata = read_json(self.directory / 'metadata.json')
        if (self.metadata['spec'] != expected
                or (not self.metadata['complete'] and not allow_incomplete)
                or self.metadata['written'] != len(rows)
                or set(self.metadata['artifacts']) != {'logits.npy', 'row_indices.npy',
                                                      'source_indices.npy', 'source_identity.npy'}):
            raise ValueError('Incomplete or identity-mismatched bank prediction cache')
        for name, digest in self.metadata['artifacts'].items():
            if sha256(self.directory / name) != digest:
                raise ValueError('Bank prediction artifact changed: ' + name)
        self.logits = np.load(self.directory / 'logits.npy', mmap_mode='r', allow_pickle=False)
        self.row_indices = np.load(self.directory / 'row_indices.npy', mmap_mode='r', allow_pickle=False)
        self.source_indices = np.load(self.directory / 'source_indices.npy', mmap_mode='r', allow_pickle=False)
        self.source_identity = np.load(self.directory / 'source_identity.npy', mmap_mode='r', allow_pickle=False)
        if (self.logits.dtype != np.float32 or list(self.logits.shape) != expected['shape']
                or self.row_indices.dtype != np.int64 or not np.array_equal(self.row_indices, rows)
                or sha256_array(self.source_indices) != expected['source_indices_sha256']
                or sha256_array(self.source_identity) != expected['source_identity_sha256']):
            self.close()
            raise ValueError('Wrong bank score dtype/shape/current-row ordering')

    def take(self, positions):
        return torch.from_numpy(np.array(self.logits[np.asarray(positions)], copy=True))

    def close(self):
        for name in ('logits', 'row_indices', 'source_indices', 'source_identity'):
            array = getattr(self, name, None)
            if array is not None and getattr(array, '_mmap', None) is not None:
                array._mmap.close()
            setattr(self, name, None)


def open_bank_cache(bank_root, split, problem, context, *, run_root=None):
    if split == 'test':
        test_permission(run_root)
    bank_root = Path(bank_root)
    if not (bank_root / split / problem / 'metadata.json').is_file():
        from . import r123_prediction_cache as shared
        root = bank_root.parent if bank_root.name == 'prediction_cache' else bank_root
        if '_shared_prediction_context' not in context:
            context['_shared_prediction_context'] = shared.context(context['prepared_root'])
        return SharedBankPredictionCache(shared.load_scores(root, split, problem,
            context['_shared_prediction_context'], test_lock_root=run_root))
    rows = expected_rows(context, split, problem)
    spec, _, _ = bank_spec(context, split, problem, rows)
    return BankPredictionCache(Path(bank_root) / split / problem, spec, rows)


class SharedBankPredictionCache:
    def __init__(self, data):
        self.logits, self.row_indices = data['logits'], data['row_indices']
        self.source_indices, self.source_identity = data['source_indices'], data['source_identity']
        self.metadata = data['metadata']

    def take(self, positions):
        return torch.from_numpy(np.array(self.logits[np.asarray(positions)], copy=True))

    def close(self):
        self.logits = self.row_indices = self.source_indices = self.source_identity = None


def load_bank(context, device='cpu'):
    for path, digest in context.get('frozen_sources', {}).items():
        if sha256(path) != digest:
            raise ValueError('Frozen bank model/input source changed: ' + path)
    models = []
    for record in context['bank']:
        if sha256(record['checkpoint_path']) != record['checkpoint_sha256']:
            raise ValueError('Frozen expert checkpoint changed')
        model, _ = load_expert(record['expert_id'], device=device)
        models.append(model.eval().requires_grad_(False))
    return models


def configure_precision():
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')


@torch.no_grad()
def cache_bank(args, context):
    from .r123_prediction_cache import build
    return build(argparse.Namespace(root=args.bank_root, prepared_root=args.prepared_root,
        input_cache_root=args.input_cache_root, test_lock_root=args.root,
        split=[args.split], problem=args.problems, experts=list(DEFAULT_EXPERTS),
        shard_index=0, shards=1, batch_size=args.predict_batch, device=args.device,
        limit=args.limit, describe=False))


def micro_caps(gpu_type='3090', large=None, other=None):
    default_large, default_other = (128, 512) if gpu_type == '3090' else (256, 1024)
    large = default_large if large is None else large
    other = default_other if other is None else other
    if not 1 <= large <= BATCH_SIZE or not 1 <= other <= BATCH_SIZE:
        raise ValueError('Microcaps must be real row counts within logical batch1024')
    return {p: large if p in ('TSP', 'CVRP') else other for p in PROBLEMS}


def backward_logical_batch(gate, optimizer, source, bank, positions, device, cap):
    optimizer.zero_grad(set_to_none=True)
    loss_sum = 0.
    for start in range(0, len(positions), cap):
        pos = positions[start:start + cap]
        batch = source.batch(bank.row_indices[pos], device=device)
        scores = bank.take(pos).to(device)
        targets, _ = correctness_targets(scores, batch['ind'], batch['pool_ids'])
        loss = brier_loss(gate(gate_inputs(batch))['reliability_logits'], targets)
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite Brier loss; optimizer not advanced')
        (loss * (len(pos) / len(positions))).backward()
        loss_sum += float(loss.detach()) * len(pos) / len(positions)
    return loss_sum


def initialize_gate(context, device='cpu'):
    ready = context['ready']
    initial = torch.load(context['prepared_root'] / 'identicalinit_seed2.pt', map_location='cpu', weights_only=False)
    if (initial['initialization'] != 'scratch' or initial['seed'] != 2
            or parameter_hash(initial['trainable']) != ready['trainable_sha256']):
        raise ValueError('Only verified seed2 scratch weights may initialize the gate')
    set_seed(2)
    gate = ReliabilityGate(ready['model_params'])
    common = {}
    with torch.no_grad():
        for name, tensor in gate.instance_encoder.named_parameters():
            source = initial['trainable']['instance_encoder.' + name]
            if tensor.shape != source.shape:
                raise ValueError('Fresh encoder parameter shape mismatch')
            tensor.copy_(source)
            common[name] = source
        branch = gate.instance_encoder.geometry_residual
        branch.mean.copy_(torch.as_tensor(context['geometry']['mean']))
        branch.std.copy_(torch.as_tensor(context['geometry']['std']))
    return gate.to(device), dict(initialization='scratch_R84_compatible_encoder_and_new_seed2_query_head',
        historical_trained_gate_initialization=False, fresh_source_sha256=ready['artifact_sha256']['identicalinit_seed2.pt'],
        encoder_parameter_sha256=parameter_hash(common), geometry_fit_population='current TRAIN81k only')


def learning_rate(update):
    if not 1 <= update <= MAX_UPDATES:
        raise ValueError('LR is keyed to successful updates 1..7200')
    if update <= WARMUP:
        return 1e-4 * update / WARMUP
    return 1e-6 + .5 * (1e-4 - 1e-6) * (1 + math.cos(math.pi * (update - WARMUP) / (MAX_UPDATES - WARMUP)))


def selection_key(result, update):
    if set(result['per_problem']) != set(PROBLEMS):
        raise ValueError('Checkpoint selection must use the complete 18-task VAL')
    risk, top1 = result['macro']['macro_actual_regret_pct'], result['macro']['macro_top1']
    if not math.isfinite(risk) or not math.isfinite(top1) or risk < 0 or not 0 <= top1 <= 1:
        raise ValueError('Full18 native VAL risk and Top1 must be finite')
    return risk, -top1, update


def protocol(context):
    return dict(experiment='R123', schema_version=1, model_factory=FACTORY,
        context_identity=context['identity'], meta_preparation=str(context['meta_preparation']),
        prepared_root=str(context['prepared_root']), data_root=str(CURRENT_ROOT),
        expert_bank=context['bank'], expert_order=list(DEFAULT_EXPERTS),
        frozen_bank_source_sha256=context['frozen_sources'],
        model_params=context['ready']['model_params'], solver_names=list(GLOBAL_SOLVERS),
        gate_architecture='independent raw InstanceEncoder4x128 plus expert-query attention and sigmoid Brier',
        gate_initialization='verified R84 seed2 SCRATCH encoder subset; new independent query/head; no trained weights',
        candidate_pools={p: r['candidates'] for p, r in context['meta']['problems'].items()},
        candidate_pool_ids={p: r['global_solver_ids'] for p, r in context['meta']['problems'].items()},
        train_rows_per_problem=10886, source_certificate_scope='source-record only; base/content copies unverified',
        heldout_exposure='historical heldouts development-exposed; not blind', seed=2,
        global_batch_size=BATCH_SIZE, logical_global_batch=BATCH_SIZE, max_successful_updates=MAX_UPDATES,
        warmup_updates=WARMUP, eval_every=EVAL_EVERY, optimizer='AdamW',
        peak_lr=1e-4, end_lr=1e-6, weight_decay=1e-4, dropout=.1, clip_grad_norm=1.,
        primary_loss='mean independent expert sigmoid Brier on saved native correctness; no cost weighting',
        gate_inputs='raw coordinate/directed-matrix/constraint inputs only; no bank predictions',
        inference='pre-sigmoid reliability argmax (monotone sigmoid); fixed bank-order ties; selected FP32 expert; solver-ID ties',
        checkpoint_selection='full18 equal-task VAL actual regret, then native Top1, then earliest update',
        winner_rule=WINNER_RULE, score_precision='original FP32 expert logits; no cross-expert mixing',
        shared_prediction_cache_module='code.V4.r123_prediction_cache',
        shared_prediction_cache_sha256=sha256(Path(__file__).with_name('r123_prediction_cache.py')),
        microcap_policy={'3090': {'TSP_CVRP': 128, 'others': 512},
                         '4090': {'TSP_CVRP': 256, 'others': 1024}},
        oom_policy='one halving only after actual CUDA OOM; replay same batch/RNG before optimizer advance',
        source_sha256={name: sha256(Path(__file__).with_name(name)) for name in
                       ('r123_model.py', 'r123_experiment.py')},
        scientific_followup_sha256=sha256(Path(__file__).parents[2] /
            'docs/research/return18_major_routes_20261030/astra_followup.md'),
        wandb_project='selector', wandb_entity='yjkds-southern-university-of-science-technology')


def start_wandb(args, config):
    import os
    import wandb
    mode = os.environ.get('WANDB_MODE', 'online' if os.environ.get('WANDB_API_KEY') else 'offline')
    return wandb.init(project=config['wandb_project'], entity=config['wandb_entity'],
        name='R123_full18_Brier_hard_router_seed2',
        id=json_hash(dict(root=str(args.root.resolve()), protocol=config))[:20],
        resume='allow', mode=mode, dir=str(args.root), config=config)


@torch.no_grad()
def evaluate(gate, args, context, split, update, *, save=False):
    if split == 'test':
        test_permission(args.root)
    was_training = gate.training
    gate.eval()
    per, archive = {}, {}
    try:
        for problem in PROBLEMS:
            source = open_source(context, split, problem, run_root=args.root,
                                 input_cache_root=args.input_cache_root)
            bank = None
            try:
                bank = open_bank_cache(args.bank_root, split, problem, context, run_root=args.root)
                logits, routed, reliability = [], [], []
                for start in range(0, len(bank.row_indices), args.predict_batch):
                    pos = np.arange(start, min(start + args.predict_batch, len(bank.row_indices)))
                    batch = source.batch(bank.row_indices[pos], device=args.device)
                    output = gate(gate_inputs(batch))
                    chosen = output['reliability_logits'].argmax(-1).cpu().numpy()
                    logits.append(np.array(bank.logits[pos, chosen], copy=True))
                    routed.append(chosen)
                    reliability.append(output['reliability'].cpu().numpy())
                scores, chosen = np.concatenate(logits), np.concatenate(routed)
                raw = dict(costs=np.array(source.raw['costs'][bank.row_indices], copy=True),
                    winner=np.array(source.raw['winner'][bank.row_indices], copy=True),
                    pool=source.pool, pool_ids=source.pool_ids)
                metrics, prediction = fixed_sbs_metrics(scores, raw, context['baselines'][problem])
                metrics['expert_pick_counts'] = np.bincount(chosen, minlength=3).tolist()
                per[problem] = metrics
                if save:
                    data = dict(logits=scores, pred=prediction, costs=raw['costs'], winner=raw['winner'],
                        indices=np.array(bank.row_indices), source_indices=np.array(bank.row_indices),
                        release_source_indices=np.array(bank.source_indices),
                        source_identity=np.array(bank.source_identity), pool=np.asarray(source.pool),
                        pool_ids=np.asarray(source.pool_ids, dtype=np.int64), chosen_expert=chosen,
                        reliability=np.concatenate(reliability))
                    directory = args.root / 'evaluation' / split / 'selected_best' / 'by_problem'
                    directory.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(directory / (problem + '.npz'), **data)
                    archive.update({problem + '__' + k: v for k, v in data.items()})
            finally:
                if bank is not None:
                    bank.close()
                source.close()
    finally:
        gate.train(was_training)
    result = summarize(per)
    result.update(experiment='R123', split=split, successful_updates=update,
        evaluation_semantics='cached chosen-expert scores, identical FP32 rule to live sparse inference',
        full_bank_precomputation_cost_not_free=True)
    if save:
        directory = args.root / 'evaluation' / split / 'selected_best'
        np.savez_compressed(directory / 'rank000.npz', **archive)
        write_json(directory / 'metrics.json', result)
    return result


def train(args, context):
    args.root.mkdir(parents=True, exist_ok=True)
    with (args.root / '.training.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        config = protocol(context)
        path = args.root / 'protocol.json'
        if path.is_file() and read_json(path) != config:
            raise ValueError('One main R123 only: existing run protocol differs')
        write_json(path, config)
        state_hash = json_hash(config)
        from . import r123_prediction_cache as shared
        shared_context = shared.context(args.prepared_root, args.input_cache_root)
        preflight_args = argparse.Namespace(root=args.bank_root, split=['train', 'val'],
            test_lock_root=args.root, limit=None, experts=list(DEFAULT_EXPERTS))
        shared.preflight(preflight_args, shared_context, list(PROBLEMS))
        for p in PROBLEMS:
            for split in ('train', 'val'):
                checked = open_bank_cache(args.bank_root, split, p, context)
                checked.close()
        gate, initialization = initialize_gate(context, args.device)
        optimizer = torch.optim.AdamW(gate.parameters(), lr=1e-4, weight_decay=1e-4)
        step, best, history = 0, None, []
        streams = {p: ShuffleStream(10886, task_seed(2, p)) for p in PROBLEMS}
        caps = micro_caps(args.gpu_type, args.large_micro_cap, args.other_micro_cap)
        if args.microbatch is not None:
            caps = {p: args.microbatch for p in PROBLEMS}
        last_path = args.root / 'last.pt'
        if last_path.is_file():
            if not args.resume:
                raise FileExistsError('Existing run requires explicit --resume')
            saved = torch.load(last_path, map_location=args.device, weights_only=False)
            if saved['state_hash'] != state_hash or saved['args'] != config:
                raise ValueError('Resume cannot change bank/data/model/protocol')
            gate.load_state_dict(saved['model'], strict=True)
            optimizer.load_state_dict(saved['optimizer'])
            step, best, history = saved['successful_updates'], saved['best'], saved['history']
            caps = saved['runtime_microcaps']
            streams = {p: ShuffleStream(10886, task_seed(2, p), saved['stream_draws'][p]) for p in PROBLEMS}
            restore_rank_rng(saved['rng'], torch.device(args.device))
        sources, banks = {}, {}
        wandb_run = None
        def checkpoint():
            return dict(model=gate.state_dict(), model_scope='gate_only; external immutable bank pinned in args',
                optimizer=optimizer.state_dict(), args=config, state_hash=state_hash,
                successful_updates=step, best=best, history=history,
                stream_draws={p: stream.draws for p, stream in streams.items()},
                runtime_microcaps=caps.copy(),
                rng=capture_rank_rng(torch.device(args.device)), initialization=initialization)
        try:
            for p in PROBLEMS:
                banks[p] = open_bank_cache(args.bank_root, 'train', p, context)
                checked = open_bank_cache(args.bank_root, 'val', p, context)
                checked.close()
                sources[p] = open_source(context, 'train', p, input_cache_root=args.input_cache_root)
            wandb_run = start_wandb(args, config)
            order = task_order(MAX_UPDATES, seed=2)
            gate.train()
            for problem in order[step:]:
                positions = streams[problem].take(BATCH_SIZE).numpy()
                rng = capture_rank_rng(torch.device(args.device))
                retry_oom = False
                try:
                    loss_sum = backward_logical_batch(gate, optimizer, sources[problem], banks[problem],
                                                     positions, args.device, caps[problem])
                except torch.cuda.OutOfMemoryError:
                    if torch.device(args.device).type != 'cuda' or caps[problem] < 2:
                        raise
                    retry_oom = True
                if retry_oom:
                    # Leave the exception scope before releasing activations;
                    # its traceback otherwise retains the failed autograd graph.
                    optimizer.zero_grad(set_to_none=True)
                    gc.collect()
                    torch.cuda.empty_cache()
                    restore_rank_rng(rng, torch.device(args.device))
                    caps[problem] //= 2
                    print('actual CUDA OOM: one retry', problem, 'cap', caps[problem], flush=True)
                    loss_sum = backward_logical_batch(gate, optimizer, sources[problem], banks[problem],
                                                     positions, args.device, caps[problem])
                norm = torch.nn.utils.clip_grad_norm_(gate.parameters(), 1., error_if_nonfinite=True)
                lr = learning_rate(step + 1)
                for group in optimizer.param_groups:
                    group['lr'] = lr
                optimizer.step()
                step += 1
                write_json(args.root / 'status.json', dict(phase='training', successful_updates=step,
                    problem=problem, brier=loss_sum, lr=lr, grad_norm=float(norm),
                    microcap=caps[problem], utc=utc()))
                wandb_run.log({'train/brier': loss_sum, 'train/lr': lr,
                               'train/grad_norm': float(norm), 'successful_updates': step}, step=step)
                if step % EVAL_EVERY == 0:
                    result = evaluate(gate, args, context, 'val', step)
                    key = selection_key(result, step)
                    history.append(dict(successful_updates=step, macro=result['macro']))
                    if best is None or key < tuple(best['key']):
                        best = dict(key=list(key), successful_updates=step, macro=result['macro'])
                        save_checkpoint(args.root / 'best.pt', checkpoint())
                    write_json(args.root / 'val_history.json', history)
                    wandb_run.log({'val/' + k: v for k, v in result['macro'].items()}, step=step)
                    print('R123 VAL', step, result['macro'], flush=True)
                if step % 180 == 0:
                    save_checkpoint(last_path, checkpoint())
            if step != MAX_UPDATES or best is None:
                raise RuntimeError('Unique final TEST lock requires complete successful-update budget')
            lock = dict(experiment='R123', checkpoint=str((args.root / 'best.pt').resolve()),
                checkpoint_sha256=sha256(args.root / 'best.pt'), protocol_sha256=state_hash,
                selected_update=best['successful_updates'], completed_updates=step,
                selected_without_test=True, per_problem_assembly=False, inference_rule=config['inference'])
            write_json(args.root / 'final_test_lock.json', lock)
            selected = torch.load(args.root / 'best.pt', map_location=args.device, weights_only=False)
            gate.load_state_dict(selected['model'], strict=True)
            evaluate(gate, args, context, 'val', best['successful_updates'], save=True)
            write_json(args.root / 'status.json', dict(phase='training_complete_test_not_run',
                successful_updates=step, selected_update=best['successful_updates'], utc=utc()))
            return lock
        finally:
            for cache in list(banks.values()) + list(sources.values()):
                cache.close()
            if wandb_run is not None:
                wandb_run.finish()


def load_checkpoint(path, device='cpu'):
    configure_precision()
    state = torch.load(path, map_location='cpu', weights_only=False)
    config = state['args']
    if (config.get('experiment') != 'R123' or config.get('model_factory') != FACTORY
            or config.get('expert_order') != list(DEFAULT_EXPERTS) or json_hash(config) != state['state_hash']):
        raise ValueError('Not a locked-shape R123 checkpoint')
    for name, digest in config['source_sha256'].items():
        if sha256(Path(__file__).with_name(name)) != digest:
            raise ValueError('R123 inference source changed: ' + name)
    gate = ReliabilityGate(config['model_params'])
    gate.load_state_dict(state['model'], strict=True)
    bank = load_bank(dict(bank=config['expert_bank'], frozen_sources=config['frozen_bank_source_sha256']), device='cpu')
    return HistoricalModelRouter(gate, bank, config['candidate_pool_ids']).to(device).eval(), state


def test(args, context):
    lock = test_permission(args.root)
    state = torch.load(args.root / 'best.pt', map_location='cpu', weights_only=False)
    if state['args'] != protocol(context):
        raise ValueError('TEST must use the unchanged trained protocol')
    gate = ReliabilityGate(state['args']['model_params']).to(args.device)
    gate.load_state_dict(state['model'], strict=True)
    return evaluate(gate, args, context, 'test', lock['selected_update'], save=True)


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('command', choices=('describe', 'cache-bank', 'train', 'test'))
    result.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    result.add_argument('--bank-root', type=Path, default=DEFAULT_BANK_ROOT)
    result.add_argument('--prepared-root', type=Path, default=DEFAULT_PREPARED_ROOT)
    result.add_argument('--meta-preparation', type=Path, default=PREPARATION)
    result.add_argument('--input-cache-root', type=Path)
    result.add_argument('--device', default='cpu')
    result.add_argument('--split', choices=('train', 'val', 'test'), default='train')
    result.add_argument('--problems', nargs='+', choices=PROBLEMS, default=list(PROBLEMS))
    result.add_argument('--gpu-type', choices=('3090', '4090'), default='3090')
    result.add_argument('--large-micro-cap', type=int)
    result.add_argument('--other-micro-cap', type=int)
    result.add_argument('--microbatch', type=int, default=None)
    result.add_argument('--predict-batch', type=int, default=128)
    result.add_argument('--limit', type=int)
    result.add_argument('--resume', action='store_true')
    result.add_argument('--write-protocol', action='store_true')
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    caps = (args.microbatch, args.large_micro_cap, args.other_micro_cap)
    if (any(value is not None and not 1 <= value <= BATCH_SIZE for value in caps)
            or args.predict_batch <= 0 or (args.limit is not None and args.limit <= 0)):
        raise ValueError('Positive real batch sizes/CPU-smoke limit required')
    if args.limit is not None and args.command != 'cache-bank':
        raise ValueError('Only incomplete cache-bank CPU smoke accepts --limit')
    torch.set_num_threads(2)
    configure_precision()
    context = source_context(args.prepared_root, args.meta_preparation)
    if args.command == 'describe':
        config = protocol(context)
        if args.write_protocol:
            write_json(PREPARATION / 'learner_protocol.json', config)
        print(__import__('json').dumps(config, indent=2), flush=True)
        return config
    if args.command == 'cache-bank':
        return cache_bank(args, context)
    if args.command == 'train':
        return train(args, context)
    return test(args, context)


if __name__ == '__main__':
    main()
