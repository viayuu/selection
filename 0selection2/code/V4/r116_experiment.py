"""Bounded matched R116 whole18 runner. No allocation, capacity search or automatic launches.

The private R84/R112 engine retains natural streams/prefetch/checkpoint plumbing,
not their model or objectives. Main owns execution caps and the 14h/arm wall timeout.
"""

import argparse
import copy
import fcntl
import importlib
import inspect
import json
import math
from functools import lru_cache
from pathlib import Path
import sys
import time
import types

import numpy as np
import torch

from code.unified_selector.registry import GLOBAL_SOLVERS, PROBLEMS
from . import r84_experiment as engine
from .r76_experiment import (capture_rank_rng, restore_rank_rng, slice_batch,
                            file_hash, read_json, utc, write_json)
from .r77_data import parameter_hash
from .r84_prepare import DEFAULT_DATA_ROOT, DEFAULT_PREPARED_ROOT, SPLIT_ROWS
from .r116_objective import (EliminationObjective, FP64Cache, LOSS_RECIPE, PART_KEYS,
                            inputs_only, per_instance_losses)


V4 = Path(__file__).resolve().parent
DEFAULT_ROOT = V4 / 'runs/R116_subset_elimination'
DEFAULT_CACHE_ROOT = Path('/tmp/shiys_R84_new90k_cache')
ARMS = ('A_static', 'B_subset')
BATCH_SIZE, MAX_UPDATES, WARMUP_UPDATES = 1024, 28440, 4266
EVAL_EVERY, TRAIN_EVERY = 1422, 7110
PEAK_LR, END_LR = 1e-4, 1e-6
BUDGETS = dict.fromkeys(ARMS, MAX_UPDATES)
MODEL_FACTORY = 'code.V4.r116_experiment.load_checkpoint'
MODEL_ROLE_ARM = 'B_execution_anchor'
INITIALIZATION = 'fresh_seed2_identical_complete_R116_parameters_R84_subset_reuse'
CHECKPOINT_RULE = ('strict maximum full18 VAL macro_top1; exact Top1 ties: minimum '
                   'macro_actual_regret_pct; exact double ties: first checkpoint')
INFERENCE_RULE = ('eval mode; exact all-subset elimination log-champion DP; argmax log q; '
                  'stable ascending-global-solver-ID ties; no maximin')
WINNER_RULE = engine.WINNER_RULE
json_hash, task_order = engine.json_hash, engine.task_order
SOURCE_FILES = ('r116_experiment.py', 'r116_objective.py', 'r116_model.py', 'r116_subset.py',
    'r116_reporting.py', 'r116_finish.py', 'r112_experiment.py', 'r112_reporting.py', 'r112_finish.py',
    'r84_experiment.py', 'r84_reporting.py', 'r84_prepare.py', 'r80_experiment.py', 'r80_reporting.py',
    'r76_experiment.py', 'r76_cache.py', 'r75_cache.py', 'r77_experiment.py', 'r77_data.py',
    'r88_model.py', 'pairwise_selector.py', 'pairwise_objective.py')


def private_module(source, name):
    """Source-name adaptation is isolated; historical modules/globals are never patched."""
    path = Path(source.__file__ if hasattr(source, '__file__') else source)
    module = types.ModuleType('code.V4._r116_' + name)
    module.__file__, module.__package__ = str(path), 'code.V4'
    sys.modules[module.__name__] = module
    source_text = path.read_text(encoding='utf-8').replace('r112', 'r116').replace('R112', 'R116')
    exec(compile(source_text, str(path) + ':R116_private', 'exec'), module.__dict__)
    return module


def adapt_function(module, original, replacements):
    source = inspect.getsource(original)
    for old, new in replacements:
        if source.count(old) != 1:
            raise RuntimeError('Historical template drift: ' + old)
        source = source.replace(old, new)
    exec(compile(source, original.__code__.co_filename + ':R116_private', 'exec'), module.__dict__)


_base = private_module(V4 / 'r112_experiment.py', 'runner_callbacks')
selection_key = _base.selection_key


def _models():
    module = importlib.import_module('code.V4.r116_model')
    if tuple(module.ARMS) != ARMS or module.SPEC['role_arm'] != MODEL_ROLE_ARM:
        raise ValueError('R116 model/runner arms or execution role differ')
    return module


def make_model(params, arm, state_dict=None):
    return _models().make_model(params, arm, state_dict=state_dict)


def initialize_shared(initial, geometry, arm, device='cpu'):
    return _base.initialize_shared(initial, geometry, arm, device)


def required_lock_fields():
    return dict(experiment='R116', locked=True, arms=list(ARMS), seed=2, task_scope='full18',
        rows_per_problem=dict(SPLIT_ROWS), logical_global_batch=BATCH_SIZE,
        max_successful_updates=MAX_UPDATES, warmup_updates=WARMUP_UPDATES,
        eval_every_updates=EVAL_EVERY, train_eval_every_updates=TRAIN_EVERY,
        optimizer='AdamW', peak_lr=PEAK_LR, end_lr=END_LR, weight_decay=1e-4,
        grad_clip=1., dropout=.1, checkpoint_rule=CHECKPOINT_RULE, inference_rule=INFERENCE_RULE,
        loss=copy.deepcopy(LOSS_RECIPE), max_train_epochs=20, combined_gpu_hours_cap=28,
        per_arm_gpu_hours_cap=14, gpu_type='3090', wandb_mode='offline')


def locked_caps(record):
    caps, evaluation = record.get('microbatch_caps', {}), record.get('eval_batch_size')
    if (set(caps) != set(PROBLEMS) or any(type(v) is not int or not 1 <= v <= BATCH_SIZE for v in caps.values())
            or type(evaluation) is not int or not 1 <= evaluation <= SPLIT_ROWS['val']):
        raise ValueError('Main lock must set positive all18 microbatch_caps and eval_batch_size')
    return dict(caps), evaluation


def protocol_lock(args):
    path = Path(args.protocol_lock or args.root / 'protocol_lock.json').expanduser().resolve()
    record = read_json(path)
    mismatch = [k for k, v in required_lock_fields().items() if record.get(k) != v]
    if mismatch:
        raise ValueError('R116 main lock differs: ' + ', '.join(mismatch))
    caps, evaluation = locked_caps(record)
    for field, name in (('data_root', 'data_root'), ('prepared_root', 'prepared_root'), ('primary_cache', 'cache_root')):
        if Path(record[field]).resolve() != Path(getattr(args, name)).resolve():
            raise ValueError('R116 main lock names another ' + field)
    if (args.gpu_type != '3090' or args.wandb_mode != 'offline'
            or args.large_micro_cap is not None and args.large_micro_cap != caps['TSP']
            or args.eval_batch_size is not None and args.eval_batch_size != evaluation):
        raise ValueError('No override of locked 3090/caps/eval/offline W&B')
    lib_batch = record.get('lib_batch_size', min(evaluation, caps['TSP'], caps['CVRP']))
    if type(lib_batch) is not int or not 1 <= lib_batch <= min(evaluation, caps['TSP'], caps['CVRP']):
        raise ValueError('LIB batch must fit the fixed capacity caps')
    if args.lib_batch_size is not None and args.lib_batch_size != lib_batch:
        raise ValueError('CLI cannot change the fixed LIB batch')
    args.large_micro_cap, args.eval_batch_size, args.lib_batch_size = caps['TSP'], evaluation, lib_batch
    return dict(path=str(path), sha256=file_hash(path), record=record)


def prepared_config(args):
    protocol_lock(args)
    ready, manifest, geometry, baselines = engine.prepared_config(args)
    for split in ('train', 'val'):
        for problem in PROBLEMS:
            row, reference = manifest['splits'][split][problem], manifest['splits']['train'][problem]
            if (row.get('complete') is not True or row['pool'] != reference['pool']
                    or row['pool_ids'] != reference['pool_ids']
                    or row['pool_ids'] != sorted(set(row['pool_ids'])) or len(row['pool_ids']) > 10
                    or baselines[problem].get('selection_split') != 'train'):
                raise ValueError('Retain complete frozen all18 pools, native order and TRAIN-only SBS')
    return ready, manifest, geometry, baselines


def open_cache(cache_root, split, problem, manifest, data_root):
    return engine.open_cache(cache_root, split, problem, manifest, data_root, cache_factory=FP64Cache)


@lru_cache(maxsize=8)
def _private_engine(binding):
    record = json.loads(binding)
    caps, _ = locked_caps(record)
    local = private_module(engine, 'engine_' + json_hash(record)[:12])
    for name in ('BATCH_SIZE', 'MAX_UPDATES', 'WARMUP_UPDATES', 'EVAL_EVERY', 'PEAK_LR', 'END_LR', 'PART_KEYS'):
        setattr(local, name, globals()[name])
    local.SPLIT_ROWS = dict(SPLIT_ROWS)
    local.LOSS_RECIPE, local.MODEL_FACTORY = copy.deepcopy(LOSS_RECIPE), MODEL_FACTORY
    local.consistency_weight = lambda update: 0.
    local.open_cache, local.inputs_only = open_cache, inputs_only
    local.selection_key, local.write_json = selection_key, _base._engine_write_json
    local.wandb_run = wandb_run
    def micro_caps(gpu_type='3090', large_cap=None):
        if gpu_type != '3090' or large_cap is not None and large_cap != caps['TSP']:
            raise ValueError('No post-lock GPU/cap change or automatic shrinking')
        return dict(caps)

    local.micro_caps = micro_caps
    adapt_function(local, engine.train, (
        ('params, order, caps)', 'params, order, caps, model=selector, initialization=initialization)'),
        ("cost = validation['macro']['macro_vs_sbs_pct']", 'cost = selection_key(validation)'),
        ('if not math.isfinite(cost):', 'if not all(math.isfinite(v) for v in cost):'),
        ("cost < best['val']['macro']['macro_vs_sbs_pct']", "cost < selection_key(best['val'])"),
        ('(40 - len(history))', '(MAX_UPDATES // EVAL_EVERY - len(history))'),
        ('problem=problem, lr=lr, consistency_weight=weight,', 'problem=problem, lr=lr,'),
        ('pending.update(lr=lr, consistency_weight=weight,', 'pending.update(lr=lr,')))
    return local


def private_engine(args=None, *, config=None):
    record = config['main_protocol_lock']['record'] if config is not None else protocol_lock(args)['record']
    return _private_engine(json.dumps(record, sort_keys=True))


def protocol(args, ready, manifest, geometry, baselines, params, order, caps, *, model, initialization):
    config = private_engine(args).protocol(args, ready, manifest, geometry, baselines, params, order, caps)
    for name in ('common_trainable_sha256', 'rdrop_weight', 'rdrop_ramp_updates'):
        config.pop(name, None)
    config.update(experiment='R116', r116_schema_version=1, model_factory=MODEL_FACTORY,
        initialization=INITIALIZATION, main_protocol_lock=protocol_lock(args),
        loss=copy.deepcopy(LOSS_RECIPE), objective=copy.deepcopy(LOSS_RECIPE),
        model_spec=copy.deepcopy(params['r116_spec']), role_arm=MODEL_ROLE_ARM,
        trainable_parameter_count=sum(p.numel() for p in model.parameters() if p.requires_grad),
        initial_trainable_sha256=initialization['trainable_sha256'],
        initial_parameter_layout_sha256=initialization['parameter_layout_sha256'],
        original_fresh_reference_sha256=ready['trainable_sha256'],
        common_parameter_sha256=initialization.get('common_parameter_sha256'),
        cross_arm_initialization='all named parameters and fresh buffers identical; R84 reuse is a subset',
        train_eval_every_updates=TRAIN_EVERY, checkpoint_rule=CHECKPOINT_RULE, inference_rule=INFERENCE_RULE,
        selection_head='exact_elimination_champion_marginal', max_train_epochs=20,
        train_exposures_per_problem=MAX_UPDATES // len(PROBLEMS) * BATCH_SIZE / SPLIT_ROWS['train'],
        selected_val_binding_rule='exact_model_protocol_and_update',
        source_plan='shared seed2 whole18 natural ShuffleStream; exact1024 rows; tails carried across cycles',
        gradients='micro rows / logical1024; one clip/step; same-batch/RNG AMP replay; no cap shrinking',
        train_eval=f'full TRAIN every{TRAIN_EVERY} successful updates plus selected checkpoint',
        current_query_labels_in_forward=False, historical_TRAIN_label_context_used=False,
        cache_policy='read-only R84 raw caches; FP64 row gather only; no full-cache rank/pair targets',
        ranking_label_policy=LOSS_RECIPE['ranking_target'],
        historical_comparison='R77 is historical, not an isolated loss ablation; both R116 arms use new objective',
        combined_gpu_hours_cap=28, per_arm_gpu_hours_cap=14,
        timeout_policy='main hard wall timeout includes train/eval/timing/LIB; partial is incomplete, never negative',
        oom_policy='fail; no automatic cap shrink',
        source_sha256={name: file_hash(V4 / name) for name in SOURCE_FILES})
    config = json.loads(json.dumps(config, allow_nan=False))
    validate_checkpoint_config(config)
    return config


def validate_checkpoint_config(config):
    record = config.get('main_protocol_lock', {}).get('record', {})
    caps, evaluation = locked_caps(record)
    params = config.get('model_params', {})
    expected = dict(experiment='R116', r116_schema_version=1, model_factory=MODEL_FACTORY,
        initialization=INITIALIZATION, seed=2, solver_names=list(GLOBAL_SOLVERS),
        rows_per_problem=SPLIT_ROWS, global_batch_size=BATCH_SIZE, logical_global_batch=BATCH_SIZE,
        max_successful_updates=MAX_UPDATES, warmup_updates=WARMUP_UPDATES,
        eval_every_updates=EVAL_EVERY, train_eval_every_updates=TRAIN_EVERY,
        eval_batch_size=evaluation, microbatch_caps=caps, optimizer='AdamW', weight_decay=1e-4,
        grad_clip=1., dropout=.1, role_arm=MODEL_ROLE_ARM, loss=LOSS_RECIPE, objective=LOSS_RECIPE,
        checkpoint_rule=CHECKPOINT_RULE, inference_rule=INFERENCE_RULE,
        selection_head='exact_elimination_champion_marginal', current_query_labels_in_forward=False,
        gpu_type='3090', wandb=True, wandb_mode='offline')
    if (config.get('arm') not in ARMS or any(config.get(k) != v for k, v in expected.items())
            or any(record.get(k) != v for k, v in required_lock_fields().items())
            or config.get('lr') != dict(rule='successful-update warmup then cosine, independent of VAL',
                peak=PEAK_LR, end=END_LR, warmup_updates=WARMUP_UPDATES, max_updates=MAX_UPDATES)
            or params.get('r116_arm') != config.get('arm') or params.get('dropout') != .1
            or params.get('role_arm') != MODEL_ROLE_ARM or config.get('model_spec') != params.get('r116_spec')
            or params.get('r116_spec') != json.loads(json.dumps(_models().SPEC))
            or any(k in config for k in ('rdrop_weight', 'rdrop_ramp_updates'))
            or set(config.get('source_sha256', {})) != set(SOURCE_FILES)):
        raise ValueError('R116 checkpoint differs from frozen model/objective/training/VAL protocol')
    for field, locked in (('data_root', 'data_root'), ('prepared_root', 'prepared_root'), ('cache_root', 'primary_cache')):
        if Path(config[field]).resolve() != Path(record[locked]).resolve():
            raise ValueError('R116 embedded path differs: ' + field)


def load_checkpoint(path, device='cpu'):
    return _base.load_checkpoint(path, device)


load_model = from_checkpoint = load_checkpoint
selected_capture_proof = _base.selected_capture_proof


def make_objective(selector, arm):
    if arm not in ARMS:
        raise ValueError('Unknown R116 arm')
    return EliminationObjective(selector)


def checkpoint_extras(objective):
    if type(objective) is not EliminationObjective:
        raise TypeError('R116 requires its stateless single-forward elimination objective')
    return {}


def restore_objective(objective, checkpoint):
    checkpoint_extras(objective)
    if 'objective_state' in checkpoint:
        raise ValueError('Unexpected mutable objective state')


def update_batch(objective, optimizer, scaler, batch, weight, runtime, cap, *, problem, max_retries=20):
    count = len(batch['ind'])
    if (problem not in PROBLEMS or count != BATCH_SIZE or runtime.world_size != 1
            or type(cap) is not int or not 1 <= cap <= count):
        raise ValueError('R116 requires a whole natural logical1024 batch and fixed positive cap')
    rng = capture_rank_rng(runtime.device)
    for retry in range(max_retries):
        restore_rank_rng(rng, runtime.device)
        optimizer.zero_grad(set_to_none=True)
        total = torch.zeros(3, dtype=torch.float64, device=runtime.device)
        bad = torch.zeros((), dtype=torch.bool, device=runtime.device)
        for start in range(0, count, cap):
            stop = min(start + cap, count)
            loss, parts = objective(slice_batch(batch, start, stop))
            scaler.scale(loss * ((stop - start) / count)).backward()
            total.add_(parts.double(), alpha=stop - start)
            bad |= ~torch.isfinite(loss.detach())
        scaler.unscale_(optimizer)
        norm = torch.nn.utils.clip_grad_norm_(objective.parameters(), 1.)
        bad |= ~torch.isfinite(norm)
        if bool(bad):
            if not scaler.is_enabled():
                raise RuntimeError('Nonfinite R116 loss/gradient without AMP; no rows skipped')
            scaler.update(new_scale=scaler.get_scale() * .5)
            continue
        scaler.step(optimizer)
        scaler.update()
        return torch.cat((total / count, norm.detach().reshape(1))), retry, {}
    raise RuntimeError('AMP replay exhausted on identical batch/RNG; no successful update accepted')


def wandb_run(args, config):
    if args.wandb_mode != 'offline':
        raise ValueError('R116 enables offline W&B only; no credential probe')
    return engine.wandb_run(args, config)


@torch.no_grad()
def evaluate(selector, args, manifest, baselines, split, tag, step, state_hash, checkpoint_sha256=None):
    """Native selector metrics plus post-query FP64 ranking diagnostics, in one forward."""
    local = private_engine(args)
    directory = args.root / args.arm / 'evaluation' / split / tag
    directory.mkdir(parents=True, exist_ok=True)
    identity = dict(experiment='R116', arm=args.arm, split=split, tag=tag, successful_updates=step,
        state_hash=state_hash, model_state_sha256=parameter_hash(selector.state_dict()),
        checkpoint_sha256=checkpoint_sha256, winner_rule=WINNER_RULE)
    if (directory / 'identity.json').exists() and read_json(directory / 'identity.json') != identity:
        raise ValueError('Evaluation already belongs to another R116 state')
    write_json(directory / 'identity.json', identity)
    if (directory / 'metrics.json').exists() and (directory / 'rank000.npz').exists():
        return read_json(directory / 'metrics.json')
    was_training, arrays, per_problem = selector.training, {}, {}
    device, started = next(selector.parameters()).device, time.monotonic()
    selector.eval()
    try:
        caps, _ = locked_caps(protocol_lock(args)['record'])
        for problem in PROBLEMS:
            local.status_update(args, 'evaluation', step, split=split, tag=tag, evaluation_problem=problem)
            shard = directory / 'by_problem' / (problem + '.npz')
            if shard.exists():
                with np.load(shard, allow_pickle=False) as saved:
                    data = {k: saved[k] for k in saved.files}
            else:
                cache = open_cache(args.cache_root, split, problem, manifest, args.data_root)
                try:
                    cap = min(args.eval_batch_size, caps[problem])
                    requests = ((problem, torch.arange(i, min(i + cap, cache.size)), min(cap, cache.size - i))
                                for i in range(0, cache.size, cap))
                    scores, totals = [], np.zeros(3, dtype=np.float64)
                    with local.PinnedPrefetch({problem: cache}, requests, device) as batches:
                        for _, count, _, batch in batches:
                            output = selector(inputs_only(batch))
                            diagnostics = per_instance_losses(output, batch)
                            values = torch.stack([diagnostics[k].double().sum() for k in PART_KEYS[:-1]])
                            if not bool(torch.isfinite(values).all()):
                                raise ValueError('Nonfinite R116 evaluation likelihood')
                            totals += values.cpu().numpy()
                            scores.append(output['logits'].float().cpu().numpy())
                    with np.load(args.data_root / (problem + split) / 'input_metadata.npz', allow_pickle=False) as metadata:
                        source = metadata['source_index'].copy()
                    data = dict(logits=np.concatenate(scores), indices=np.arange(cache.size, dtype=np.int64),
                        source_indices=np.arange(cache.size, dtype=np.int64), release_source_indices=source,
                        costs=cache.raw['costs'].copy(), winner=cache.raw['winner'].copy(),
                        pool=np.asarray(cache.pool), pool_ids=np.asarray(cache.pool_ids, dtype=np.int64),
                        nodes=np.asarray(cache.lengths).copy(), objective_means=totals / cache.size)
                    local.save_npz(shard, data)
                finally:
                    cache.close()
            count = manifest['splits'][split][problem]['rows']
            if (count != SPLIT_ROWS[split] or data['costs'].dtype != np.float64
                    or data['winner'].dtype != np.int64 or len(data['logits']) != count
                    or not np.array_equal(data['indices'], np.arange(count))
                    or not np.array_equal(data['winner'], data['costs'].argmin(1))
                    or not np.isfinite(data['objective_means']).all()):
                raise ValueError('Incomplete native/FP64 R116 evaluation: ' + problem)
            raw = {k: data[k] for k in ('costs', 'winner', 'pool', 'pool_ids')}
            raw['pool'], raw['pool_ids'] = raw['pool'].tolist(), raw['pool_ids'].tolist()
            metrics, data['pred'] = local.fixed_sbs_metrics(data['logits'], raw, baselines[problem])
            metrics.update(dict(zip(PART_KEYS[:-1], data['objective_means'].tolist())))
            per_problem[problem] = metrics
            arrays.update({problem + '__' + k: v for k, v in data.items()})
    finally:
        selector.train(was_training)
    local.save_npz(directory / 'rank000.npz', arrays)
    result = local.summarize(per_problem)
    result['macro'].update({'macro_' + key: float(np.mean([v[key] for v in per_problem.values()]))
                            for key in PART_KEYS[:-1]})
    result.update(seconds=time.monotonic() - started, tag=tag, successful_updates=step,
        sample_exposures=step * BATCH_SIZE, identity=identity,
        prediction_archive=str((directory / 'rank000.npz').resolve()), own_train_sbs=local.summarize(per_problem))
    write_json(directory / 'metrics.json', result)
    return result


def authorize_test(args):
    return _base.authorize_test(args)


def ensure_test_cache(args, lock):
    return _base.ensure_test_cache(args, lock)


def train(args, device):
    return private_engine(args).train(args, device, integration=sys.modules[__name__])


def test(args, device):
    return private_engine(args).test(args, device, integration=sys.modules[__name__])


cached_input_batch = _base.cached_input_batch


def describe(args):
    path = args.protocol_lock or args.root / 'protocol_lock.json'
    return dict(experiment='R116', arm=args.arm, required_protocol_lock=required_lock_fields(),
        required_execution_fields=['data_root', 'prepared_root', 'primary_cache', 'microbatch_caps', 'eval_batch_size'],
        main_protocol_lock=protocol_lock(args) if path.exists() else None, model_factory=MODEL_FACTORY,
        source_files=list(SOURCE_FILES), wandb_mode='offline', gpu_type='3090',
        finisher='python -m code.V4.r116_finish wait --root PATH --publish-after')


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('command', choices=('train', 'test', 'finalize', 'timing', 'describe', 'status'))
    result.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    result.add_argument('--arm', choices=ARMS, required=True)
    result.add_argument('--data-root', type=Path, default=DEFAULT_DATA_ROOT)
    result.add_argument('--prepared-root', type=Path, default=DEFAULT_PREPARED_ROOT)
    result.add_argument('--cache-root', type=Path, default=DEFAULT_CACHE_ROOT)
    result.add_argument('--protocol-lock', type=Path)
    result.add_argument('--device', default='cuda:0')
    result.add_argument('--gpu-type', choices=('3090',), default='3090')
    result.add_argument('--large-micro-cap', type=int)
    result.add_argument('--eval-batch-size', type=int)
    result.add_argument('--log-every', type=int, default=18)
    result.add_argument('--wandb-mode', choices=('offline',), default='offline')
    result.add_argument('--resume', action='store_true')
    result.add_argument('--state-hash', '--expected-state-hash')
    result.add_argument('--lib-batch-size', type=int)
    result.add_argument('--nss-root', type=Path, default=V4 / 'runs/R84_new90k_attention/nss_reference')
    result.set_defaults(experiment='R116')
    return result


def timing(args, device):
    from .r116_reporting import measure_timing
    return measure_timing(args, device)


def finalize(args, device):
    result = test(args, device)
    timing(args, device)
    from . import r116_reporting
    r116_reporting.lib_stage(args)
    return dict(test=result, report=r116_reporting.report(args))


def main(argv=None):
    args = parser().parse_args(argv)
    for name in ('root', 'data_root', 'prepared_root', 'cache_root', 'protocol_lock', 'nss_root'):
        value = getattr(args, name)
        if value is not None:
            setattr(args, name, value.expanduser().resolve())
    if args.command in ('describe', 'status'):
        path = args.root / args.arm / 'status.json'
        value = describe(args) if args.command == 'describe' else (
            read_json(path) if path.exists() else dict(experiment='R116', arm=args.arm, phase='not_started'))
        print(json.dumps(value, indent=2))
        return value
    protocol_lock(args)
    if args.log_every < 1:
        raise ValueError('Log interval must be positive')
    run = args.root / args.arm
    run.mkdir(parents=True, exist_ok=True)
    with (run / 'runner.lock').open('a') as claim:
        fcntl.flock(claim, fcntl.LOCK_EX | fcntl.LOCK_NB)
        torch.set_num_threads(2)
        engine.configure_torch()
        device = torch.device(args.device)
        if device.type != 'cuda':
            raise ValueError('Formal execution requires the main-assigned 3090 CUDA device')
        try:
            return globals()[args.command](args, device)
        except BaseException as error:
            write_json(run / 'error.json', dict(experiment='R116', arm=args.arm, updated_utc=utc(),
                command=args.command, error_type=type(error).__name__, message=str(error),
                complete=False, interpretation='incomplete; not negative evidence'))
            raise


# Bind the adapted callbacks to R116 only, including factories and stateless objective.
for _name in ('ARMS', 'BUDGETS', 'BATCH_SIZE', 'MAX_UPDATES', 'WARMUP_UPDATES', 'EVAL_EVERY',
              'TRAIN_EVERY', 'PEAK_LR', 'END_LR', 'MODEL_FACTORY', 'MODEL_ROLE_ARM', 'INITIALIZATION',
              'CHECKPOINT_RULE', 'INFERENCE_RULE', 'DEFAULT_ROOT', '_models', 'make_model',
              'protocol_lock', 'private_engine', 'validate_checkpoint_config',
              'checkpoint_extras', 'required_lock_fields'):
    setattr(_base, _name, globals()[_name])


if __name__ == '__main__':
    main()
