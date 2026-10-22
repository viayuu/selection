"""R95 full18 adaptive TRAIN exposure, with correction of the complete row loss."""

import argparse
import copy
import importlib.util
import sys
import time
from pathlib import Path

import numpy as np
import torch

from code.unified_selector.registry import GLOBAL_SOLVERS, PROBLEMS
from . import r84_experiment as engine
from .multitask_probe import task_seed
from .pairwise_objective import inputs_only
from .r76_experiment import PART_KEYS, PinnedPrefetch, file_hash, open_cache, read_json, utc, write_json
from .r77_data import parameter_hash
from .r80_experiment import save_npz
from .r84_prepare import DEFAULT_DATA_ROOT, DEFAULT_PREPARED_ROOT, SPLIT_ROWS
from .r85_experiment import ensure_test_cache, evaluate as evaluate_original
from .r85_experiment import selection_counts, zero_summary
from .r95_model import ARMS, DROPOUT, MODEL_ROLE_ARM, ROLE_SLOT, SELECTION_HEAD, SPEC, make_model
from .r95_objective import WeightedRDropObjective, final_affine_proxy
from .r95_sampling import AdaptiveStream, WeightedCache, make_proposal


V4 = Path(__file__).resolve().parent
DEFAULT_ROOT = V4 / 'runs/R95_adaptive_example_sampling'
NEURAL_ARMS = ARMS
BATCH_SIZE, MAX_UPDATES = engine.BATCH_SIZE, engine.MAX_UPDATES
WARMUP_UPDATES, EVAL_EVERY = engine.WARMUP_UPDATES, engine.EVAL_EVERY
PEAK_LR, END_LR = engine.PEAK_LR, engine.END_LR
BUDGETS = dict.fromkeys(ARMS, MAX_UPDATES)
MODEL_FACTORY, WINNER_RULE = 'code.V4.r95_experiment.load_checkpoint', engine.WINNER_RULE
prepared_config, micro_caps = engine.prepared_config, engine.micro_caps
json_hash, task_order, learning_rate = engine.json_hash, engine.task_order, engine.learning_rate
status_update, consistency_weight = engine.status_update, engine.consistency_weight
PROPOSAL_EVERY = 7200
SAMPLING = dict(warm_natural_updates=7200, refresh_every_updates=7200,
    proposal_uniform_mixture=.5, proxy='deterministic_full_loss_final_affine_gradient_norm',
    correction='1/(N*q)', normalization='arithmetic_batch_mean_no_self_normalization')
INFERENCE_RULE = ('eval mode; fixed physical-node0 TSP slot3 role; original score MLP, '
                  'pair differences and maximin; stable ascending-global-solver-ID tie break')


def initialize_shared(initial, geometry, arm, device='cpu'):
    model, params, receipt = engine.initialize_shared(initial, geometry, arm, device,
                                                     model_factory=make_model)
    actual = dict(model.named_parameters())
    if (set(actual) != set(initial['trainable'])
            or parameter_hash(actual) != initial['trainable_sha256']
            or params.get('role_arm') != MODEL_ROLE_ARM):
        raise ValueError('R95 retains all and only the identical fresh R84 parameters')
    receipt.update(new_parameter_count=0, trainable_parameter_count=sum(v.numel() for v in actual.values()),
        model_role_arm=MODEL_ROLE_ARM, role_slot=ROLE_SLOT, sampling_arm=arm)
    return model, params, receipt


def protocol_lock(args):
    path = args.protocol_lock or args.root / 'protocol_lock.json'
    record = read_json(path)
    expected = dict(experiment='R95', locked=True, arms=list(ARMS), seed=2,
        task_scope='full18', rows_per_problem=SPLIT_ROWS, batch_size=BATCH_SIZE,
        logical_global_batch=BATCH_SIZE, max_updates=MAX_UPDATES,
        warmup_updates=WARMUP_UPDATES, eval_every=EVAL_EVERY, dropout=DROPOUT,
        large_micro_cap=352, eval_batch_size=256, sampling=SAMPLING,
        arm_budgets=BUDGETS, optimizer='AdamW', peak_lr=PEAK_LR, end_lr=END_LR,
        weight_decay=1e-4, grad_clip=1., role_arm=MODEL_ROLE_ARM, role_slot=ROLE_SLOT,
        selection_head=SELECTION_HEAD, wandb_mode='offline')
    if any(record.get(name) != value for name, value in expected.items()):
        raise ValueError('R95 requires the fixed full18 matched sampling protocol')
    for name in ('data_root', 'prepared_root'):
        if Path(record.get(name, '')).resolve() != getattr(args, name).resolve():
            raise ValueError('R95 lock names a different ' + name)
    return dict(path=str(path.resolve()), sha256=file_hash(path), record=record)


def objective_spec(arm):
    if arm not in ARMS:
        raise ValueError('Unknown R95 arm')
    return dict(main='original maximin CE, same-instance comparison, relative risk and RDrop',
        coefficients=dict(ce=.35, comparison=.35, risk=.02, rdrop_final=.35),
        forward_passes=2, weighting='entire per-row objective / (TRAIN_N * draw_probability)',
        weight_normalization='actual arithmetic batch mean; no clipping or self-normalization',
        extra_loss=False, class_rebalancing=False, mutable_objective_state=False,
        gradient_claim='unbiased sampling estimator BEFORE global gradient clipping/AdamW; not an unbiased optimizer update')


def protocol(args, ready, manifest, geometry, baselines, params, order, caps):
    if (args.arm not in ARMS or args.large_micro_cap != 352 or args.eval_batch_size != 256
            or caps.get('TSP') != 352 or caps.get('CVRP') != 352
            or params.get('r95_arm') != args.arm or params.get('r95_spec') != SPEC):
        raise ValueError('R95 retains the locked original model and micro352/eval256')
    config = engine.protocol(args, ready, manifest, geometry, baselines, params, order, caps)
    config['loss'].pop('focal_extra', None)
    config.update(experiment='R95', r95_schema_version=1, model_factory=MODEL_FACTORY,
        main_protocol_lock=protocol_lock(args), objective=objective_spec(args.arm),
        model_spec=copy.deepcopy(SPEC), sampling=copy.deepcopy(SAMPLING),
        objective_state='none; immutable proposal files bound to their own TRAIN evaluation state',
        training_unit='successful_neural_updates', selection_head=SELECTION_HEAD,
        input_roles=dict(arm=MODEL_ROLE_ARM, problem='TSP', node_column=ROLE_SLOT,
            assignment='physical input node0', label_dependent=False, new_parameter_count=0),
        inference_rule=INFERENCE_RULE,
        source_plan='same natural rows for first7200 updates; then coupled uniform random numbers with replacement; A uniform and B corrected adaptive proposal',
        gradients='micro rows / logical1024; entire loss weighted 1/(Nq); one clip/step; replay same sampled rows, weights and RNG on AMP retry',
        proxy_boundary='exact deterministic eval final-affine gradient norm of CE+comparison+risk, including cancellation; eval RDrop is zero; not full-network or stochastic-training gradient norm',
        historical_reference=dict(primary='R89B', secondary='R84A', retrained=False,
            matched_original_loss_baseline=False,
            primary_run=str((V4 / 'runs/R89_optimizer_geometry/B_muon_hybrid').resolve()),
            secondary_run=str((engine.DEFAULT_ROOT / 'A_baseline').resolve())),
        evidence_boundary='Matched exposure experiment, not guaranteed variance reduction, class balancing or extra expected error penalty. Large batch1024 and stale/partial gradients may leave little benefit. Hardware differs across arms.',
        source_sha256={name: file_hash(V4 / name) for name in
            ('r95_experiment.py', 'r95_objective.py', 'r95_sampling.py', 'r95_model.py',
             'r84_experiment.py', 'r84_prepare.py', 'r85_experiment.py', 'r88_model.py',
             'pairwise_selector.py', 'pairwise_objective.py', 'r76_experiment.py', 'r77_experiment.py')})
    return config


def make_objective(selector, arm):
    objective_spec(arm)
    return WeightedRDropObjective(selector)


def checkpoint_extras(objective):
    if type(objective) is not WeightedRDropObjective:
        raise TypeError('R95 requires the complete stateless weighted RDrop objective')
    return {}


def restore_objective(objective, checkpoint):
    checkpoint_extras(objective)
    if 'objective_state' in checkpoint:
        raise ValueError('R95 has no mutable objective state')


def update_batch(objective, optimizer, scaler, batch, weight, runtime, cap,
                 max_retries=20, *, problem):
    if (problem not in PROBLEMS or len(batch['ind']) != BATCH_SIZE
            or cap != micro_caps('3090', 352)[problem] or 'sampling_weight' not in batch):
        raise ValueError('R95 requires correctly weighted logical1024 TRAIN rows')
    parts, retries, _ = engine.update_batch(objective, optimizer, scaler, batch, weight,
                                           runtime, cap, max_retries=max_retries)
    weights = batch['sampling_weight'].detach().float()
    return parts, retries, dict(sampling_mean_weight=weights.mean(),
        sampling_min_weight=weights.min(), sampling_max_weight=weights.max(),
        sampling_unique_fraction=batch['sampling_row'].unique().numel() / BATCH_SIZE)


def validate_checkpoint_config(config):
    arm = config.get('arm')
    expected = dict(experiment='R95', r95_schema_version=1, seed=2,
        initialization='scratch_shared_r77', model_factory=MODEL_FACTORY,
        solver_names=list(GLOBAL_SOLVERS), rows_per_problem=SPLIT_ROWS,
        global_batch_size=BATCH_SIZE, logical_global_batch=BATCH_SIZE,
        max_successful_updates=MAX_UPDATES, warmup_updates=WARMUP_UPDATES,
        eval_every_updates=EVAL_EVERY, eval_batch_size=256, optimizer='AdamW',
        weight_decay=1e-4, grad_clip=1., dropout=.1, rdrop_weight=.35,
        rdrop_ramp_updates=WARMUP_UPDATES, sampling=SAMPLING, model_spec=SPEC,
        loss=engine.LOSS_RECIPE, selection_head=SELECTION_HEAD,
        inference_rule=INFERENCE_RULE, winner_rule=WINNER_RULE)
    if arm not in ARMS or any(config.get(name) != value for name, value in expected.items()):
        raise ValueError('R95 checkpoint model, loss or fixed training controls changed')
    params = config.get('model_params', {})
    if (params.get('r95_arm') != arm or params.get('r95_spec') != SPEC
            or params.get('dropout') != .1 or config.get('objective') != objective_spec(arm)):
        raise ValueError('R95 checkpoint arm or original objective changed')


def load_checkpoint(path, device='cpu'):
    state = torch.load(path, map_location='cpu', weights_only=False)
    config = state['args']
    validate_checkpoint_config(config)
    if state.get('state_hash') != json_hash(config) or 'objective_state' in state:
        raise ValueError('R95 checkpoint protocol hash or objective state changed')
    return make_model(config['model_params'], config['arm'], state_dict=state['model']).to(device), state


load_model = load_checkpoint
from_checkpoint = load_checkpoint


def proposal_paths(args, step):
    base = args.root / args.arm / 'proposals' / f'update{step:07d}'
    return base.with_suffix('.npz'), base.with_suffix('.json')


def proposal_provider(args, state_hash):
    loaded = {}

    def read(problem, source_step):
        if source_step not in loaded:
            path, record_path = proposal_paths(args, source_step)
            record = read_json(record_path)
            if (record.get('complete') is not True or record.get('split') != 'train'
                    or record.get('state_hash') != state_hash or record.get('arm') != args.arm
                    or record.get('successful_updates') != source_step
                    or record.get('rows_per_problem') != SPLIT_ROWS['train']
                    or record.get('proposal_sha256') != file_hash(path)):
                raise ValueError('Sampling requires its own exact TRAIN proposal, never held-out labels')
            with np.load(path, allow_pickle=False) as saved:
                loaded.clear()
                loaded[source_step] = {p: saved[p + '__q'].copy() for p in PROBLEMS}
        return loaded[source_step][problem]

    return read


def write_proposal(args, directory, identity, arrays):
    step = identity['successful_updates']
    path, record_path = proposal_paths(args, step)
    values, summaries = {}, {}
    for problem in PROBLEMS:
        scores = np.asarray(arrays[problem + '__sampling_gradient_norm'], dtype=np.float64)
        if scores.shape != (SPLIT_ROWS['train'],) or not np.isfinite(scores).all() or (scores < 0).any():
            raise ValueError('Every TRAIN row needs a finite final-affine gradient proxy')
        q = make_proposal(scores)
        values[problem + '__q'], values[problem + '__gradient_norm'] = q, scores
        uniform_second = float(np.mean(scores * scores))
        sampled_second = float(np.mean(scores * scores / (len(q) * q)))
        summaries[problem] = dict(n=len(q), q_min=float(q.min()), q_max=float(q.max()),
            max_inverse_weight=float(1. / (len(q) * q.min())),
            proposal_effective_population=float(1. / np.sum(q * q)),
            gradient_norm_mean=float(scores.mean()), gradient_norm_max=float(scores.max()),
            deterministic_last_layer_second_moment_ratio=(sampled_second / uniform_second
                if uniform_second > 0. else 1.))
    record = dict(experiment='R95', arm=args.arm, complete=True, split='train',
        successful_updates=step, rows_per_problem=SPLIT_ROWS['train'], state_hash=identity['state_hash'],
        model_state_sha256=identity['model_state_sha256'], sampling=SAMPLING,
        source_identity_sha256=file_hash(directory / 'identity.json'),
        source_prediction_sha256=file_hash(directory / 'rank000.npz'),
        per_problem=summaries, proxy_boundary='deterministic final-affine proxy, not full-network variance',
        gradient_boundary='before global clip and nonlinear AdamW; no unbiased optimizer-update claim')
    if record_path.exists():
        previous = read_json(record_path)
        if previous != dict(record, proposal_sha256=file_hash(path)):
            raise ValueError('An immutable TRAIN proposal already belongs to a different source')
        return
    save_npz(path, values)
    write_json(record_path, dict(record, proposal_sha256=file_hash(path)))


@torch.no_grad()
def evaluate_sampling_train(selector, args, manifest, baselines, tag, step, state_hash,
                            checkpoint_sha256=None):
    """Reuse the scheduled full TRAIN forward; retain only one extra scalar per row."""
    directory = args.root / args.arm / 'evaluation/train' / tag
    directory.mkdir(parents=True, exist_ok=True)
    identity = dict(experiment='R95', arm=args.arm, split='train', tag=tag,
        successful_updates=step, state_hash=state_hash,
        model_state_sha256=parameter_hash(selector.state_dict()),
        checkpoint_sha256=checkpoint_sha256, winner_rule=WINNER_RULE)
    identity_path = directory / 'identity.json'
    if identity_path.exists() and read_json(identity_path) != identity:
        raise ValueError('TRAIN proposal evaluation belongs to another model state')
    write_json(identity_path, identity)
    if (directory / 'metrics.json').exists() and (directory / 'rank000.npz').exists():
        with np.load(directory / 'rank000.npz', allow_pickle=False) as saved:
            arrays = {key: saved[key] for key in saved.files}
        write_proposal(args, directory, identity, arrays)
        return read_json(directory / 'metrics.json')
    was_training, arrays, per_problem = selector.training, {}, {}
    selector.eval()
    device, started = next(selector.parameters()).device, time.monotonic()
    try:
        for problem in PROBLEMS:
            status_update(args, 'evaluation', step, split='train', tag=tag, evaluation_problem=problem)
            shard = directory / 'by_problem' / (problem + '.npz')
            if shard.exists():
                with np.load(shard, allow_pickle=False) as saved:
                    data = {key: saved[key] for key in saved.files}
            else:
                cache = open_cache(args.cache_root, 'train', problem, manifest, args.data_root)
                hidden = []
                handle = selector.score_head[-1].register_forward_pre_hook(
                    lambda _module, values: hidden.append(values[0].detach()))
                try:
                    cap = min(args.eval_batch_size, micro_caps(args.gpu_type, args.large_micro_cap)[problem])
                    requests = ((problem, torch.arange(i, min(i + cap, cache.size)),
                                 min(cap, cache.size - i)) for i in range(0, cache.size, cap))
                    scores, norms = [], []
                    with PinnedPrefetch({problem: cache}, requests, device) as batches:
                        for _, _, _, batch in batches:
                            hidden.clear()
                            output = selector(inputs_only(batch))
                            if len(hidden) != 1:
                                raise ValueError('R95 proxy requires exactly the original final affine head')
                            norms.append(final_affine_proxy(output, batch, hidden.pop()).cpu().numpy())
                            scores.append(output['logits'].float().cpu().numpy())
                    with np.load(args.data_root / (problem + 'train') / 'input_metadata.npz', allow_pickle=False) as source:
                        source_indices = source['source_index'].copy()
                    data = dict(logits=np.concatenate(scores), sampling_gradient_norm=np.concatenate(norms),
                        indices=np.arange(cache.size, dtype=np.int64),
                        source_indices=np.arange(cache.size, dtype=np.int64), release_source_indices=source_indices,
                        costs=cache.raw['costs'].copy(), winner=cache.raw['winner'].copy(),
                        pool=np.asarray(cache.pool), pool_ids=np.asarray(cache.pool_ids, dtype=np.int64),
                        nodes=np.asarray(cache.lengths).copy())
                    save_npz(shard, data)
                finally:
                    handle.remove()
                    cache.close()
            if (len(data['winner']) != SPLIT_ROWS['train'] or data['costs'].dtype != np.float64
                    or not np.array_equal(data['winner'], data['costs'].argmin(1))
                    or not np.array_equal(data['indices'], np.arange(SPLIT_ROWS['train']))):
                raise ValueError('Sampling must retain all original ordered TRAIN supervision')
            raw = {name: data[name] for name in ('costs', 'winner', 'pool', 'pool_ids')}
            raw['pool'], raw['pool_ids'] = raw['pool'].tolist(), raw['pool_ids'].tolist()
            per_problem[problem], data['pred'] = engine.fixed_sbs_metrics(data['logits'], raw, baselines[problem])
            arrays.update({problem + '__' + key: value for key, value in data.items()})
    finally:
        selector.train(was_training)
    save_npz(directory / 'rank000.npz', arrays)
    result = engine.summarize(per_problem)
    result.update(seconds=time.monotonic() - started, tag=tag, successful_updates=step,
        sample_exposures=step * BATCH_SIZE, identity=identity,
        prediction_archive=str((directory / 'rank000.npz').resolve()),
        own_train_sbs=engine.summarize(per_problem))
    write_json(directory / 'metrics.json', result)
    write_proposal(args, directory, identity, arrays)
    print('[R95 eval] ' + str(dict(arm=args.arm, split='train', tag=tag, **result['macro'])), flush=True)
    return result


def evaluate(selector, args, manifest, baselines, split, tag, step, state_hash,
             checkpoint_sha256=None):
    if split == 'train' and tag.startswith('update') and step > 0 and step % PROPOSAL_EVERY == 0:
        return evaluate_sampling_train(selector, args, manifest, baselines, tag, step,
                                       state_hash, checkpoint_sha256)
    return evaluate_original(selector, args, manifest, baselines, split, tag, step,
                             state_hash, checkpoint_sha256)


def authorize_test(args):
    lock, config = engine.authorize_test(args)
    validate_checkpoint_config(config)
    if lock.get('inference_rule') != INFERENCE_RULE or lock.get('winner_rule') != WINNER_RULE:
        raise PermissionError('R95 TEST requires unchanged original inference and native labels')
    return lock, config


def training_engine(args):
    """Patch only a private runner copy; other active experiments keep their streams."""
    name = 'code.V4._r95_training_engine_' + args.arm
    spec = importlib.util.spec_from_file_location(name, engine.__file__)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    # Requests are materialized at interval boundaries; proposals refresh on the
    # existing 7200-update full-TRAIN evaluation, never mid-prefetch interval.
    seed_to_problem = {task_seed(2, p): p for p in PROBLEMS}

    proposal = None

    def provide(problem, source_step):
        nonlocal proposal
        if proposal is None:
            config = read_json(args.root / args.arm / 'protocol.json')
            proposal = proposal_provider(args, json_hash(config))
        return proposal(problem, source_step)

    def stream(size, seed, draws=0):
        return AdaptiveStream(size, seed, draws, arm=args.arm,
            problem=seed_to_problem[seed], proposal_provider=provide)

    def weighted_cache(cache_root, split, problem, manifest, data_root, cache_factory=None):
        cache = open_cache(cache_root, split, problem, manifest, data_root, cache_factory)
        return WeightedCache(cache) if split == 'train' else cache

    module.ShuffleStream, module.open_cache = stream, weighted_cache
    return module


def train(args, device):
    protocol_lock(args)
    return training_engine(args).train(args, device, integration=sys.modules[__name__])


def test(args, device):
    return engine.test(args, device, integration=sys.modules[__name__])


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('command', choices=('train', 'test', 'describe', 'status'))
    result.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    result.add_argument('--arm', choices=ARMS, required=True)
    result.add_argument('--prepared-root', type=Path, default=DEFAULT_PREPARED_ROOT)
    result.add_argument('--data-root', type=Path, default=DEFAULT_DATA_ROOT)
    result.add_argument('--cache-root', type=Path)
    result.add_argument('--protocol-lock', type=Path)
    result.add_argument('--gpu-type', choices=('3090', '4090'), default='3090')
    result.add_argument('--large-micro-cap', type=int, default=352)
    result.add_argument('--eval-batch-size', type=int, default=256)
    result.add_argument('--device', default='cuda:0')
    result.add_argument('--wandb-mode', choices=('auto', 'online', 'offline'), default='offline')
    result.add_argument('--log-every', type=int, default=18)
    result.add_argument('--resume', action='store_true')
    result.add_argument('--state-hash', '--expected-state-hash')
    result.set_defaults(experiment='R95')
    return result


def describe(args):
    return dict(experiment='R95', arm=args.arm, seed=2, rows=SPLIT_ROWS,
        max_updates=MAX_UPDATES, logical_batch=BATCH_SIZE, warmup_updates=WARMUP_UPDATES,
        eval_every=EVAL_EVERY, sampling=SAMPLING, objective=objective_spec(args.arm),
        model_factory=MODEL_FACTORY, data_root=str(args.data_root), prepared_root=str(args.prepared_root))


def main(argv=None):
    return engine.main(argv, integration=sys.modules[__name__])


if __name__ == '__main__':
    main()
