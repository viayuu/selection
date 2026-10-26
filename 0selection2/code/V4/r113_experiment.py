"""R113 three-arm full18 selector runner; no allocation or historical rewrites.

Pretrain B/C, train each identical final architecture, then finalize the unique
VAL-selected checkpoint. Execution sizes/device are bound by the main thread.
"""

import argparse
import copy
import fcntl
import json
import math
import shutil
import sys
import time
from collections import Counter
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch

from code.unified_selector.registry import GLOBAL_SOLVERS, PROBLEMS
from . import r84_experiment as engine
from . import r113_model as models
from . import r113_pretrain as process
from .multitask_probe import task_seed
from .r76_experiment import (PART_KEYS, PinnedPrefetch, RDropObjective, Runtime,
    capture_rank_rng, file_hash, open_cache, read_json, restore_rank_rng, save_checkpoint, utc, write_json)
from .r77_data import parameter_hash
from .r77_experiment import ShuffleStream
from .r112_experiment import cached_input_batch, measure_inputs, private_module, selection_key
from .r84_prepare import DEFAULT_DATA_ROOT, DEFAULT_PREPARED_ROOT, SPLIT_ROWS


V4 = Path(__file__).resolve().parent
DEFAULT_ROOT = V4 / 'runs/R113_executable_process'
DEFAULT_CACHE_ROOT = Path('/tmp/shiys_R84_new90k_cache')
DEFAULT_PROCESS_MANIFEST = DEFAULT_ROOT / 'process_train_v1/manifest.json'
ARMS = ('A_scratch', 'B_snapshot', 'C_transition')
BATCH_SIZE, MAX_UPDATES, WARMUP_UPDATES = 512, 57600, 4320
EVAL_EVERY, TRAIN_EVERY = 2880, 14400
PEAK_LR, END_LR = 1e-4, 1e-6
BUDGETS = dict.fromkeys(ARMS, MAX_UPDATES)
MODEL_FACTORY = 'code.V4.r113_experiment.load_checkpoint'
MODEL_ROLE_ARM = 'B_execution_anchor'
INITIALIZATION = 'identical_complete_fresh_R113; only_BC_load_own_new_TRAIN_processor'
CHECKPOINT_RULE = ('strict maximum full18 VAL macro_top1; exact Top1 ties: minimum '
                   'macro_actual_regret_pct; exact double ties: first checkpoint')
INFERENCE_RULE = ('eval mode; self-initialized predicted-state transition feedback; original R77 '
                  'score-difference maximin; stable ascending-global-solver-ID ties')
WINNER_RULE = engine.WINNER_RULE
json_hash, task_order = engine.json_hash, engine.task_order
SOURCE_FILES = ('r113_experiment.py', 'r113_pretrain.py', 'r113_model.py', 'r113_processor.py',
    'r113_torch_inputs.py', 'r113_process_data.py', 'r84_experiment.py', 'r84_prepare.py',
    'r76_experiment.py', 'r77_experiment.py', 'r77_data.py', 'pairwise_objective.py',
    'pairwise_selector.py', 'r88_model.py', 'r112_experiment.py')


def make_model(params, arm, state_dict=None):
    if tuple(models.ARMS) != ARMS:
        raise ValueError('R113 model/runner arms differ')
    return models.make_model(params, arm, state_dict=state_dict)


def initialize_shared(initial, geometry, arm, device='cpu'):
    """Always fresh; applying a process export is a separate, audited operation."""
    if (arm not in ARMS or initial.get('seed') != 2 or initial.get('initialization') != 'scratch'
            or parameter_hash(initial['trainable']) != initial.get('trainable_sha256')):
        raise ValueError('Require prepared fresh seed2 tensors, never a trained selector')
    model, params, receipt = models.initialize_shared(initial, geometry, arm, device)
    if params.get('dropout') != .1 or params.get('r113_arm') != arm:
        raise ValueError('R113 requires the common complete architecture and dropout.1')
    weights = dict(model.named_parameters())
    schema = process.tensor_schema(model.state_dict())
    receipt = dict(receipt, fresh_reference_sha256=initial['trainable_sha256'],
        trainable_parameter_count=sum(p.numel() for p in weights.values() if p.requires_grad),
        parameter_layout_sha256=json_hash(schema),
        fresh_nonprocessor_sha256=nonprocessor_hash(model),
        initial_processor_sha256=parameter_hash(model.instance_encoder.processor.state_dict()),
        geometry_fitted_on='frozen R84 81000 TRAIN only')
    return model, params, receipt


def nonprocessor_hash(model):
    return parameter_hash({k: v for k, v in model.state_dict().items()
                           if not k.startswith('instance_encoder.processor.')})


def required_lock_fields():
    return dict(experiment='R113', locked=True, arms=list(ARMS), seed=2, task_scope='full18',
        rows_per_problem=dict(SPLIT_ROWS), logical_global_batch=BATCH_SIZE,
        max_successful_updates=MAX_UPDATES, warmup_updates=WARMUP_UPDATES,
        eval_every_updates=EVAL_EVERY, train_eval_every_updates=TRAIN_EVERY,
        optimizer='AdamW', peak_lr=PEAK_LR, end_lr=END_LR, weight_decay=1e-4,
        grad_clip=1., dropout=.1, checkpoint_rule=CHECKPOINT_RULE, inference_rule=INFERENCE_RULE,
        pretraining=process.required_fields())


def locked_caps(record):
    result = []
    for field in ('microbatch_caps', 'pretrain_microbatch_caps'):
        caps = record.get(field, {})
        if set(caps) != set(PROBLEMS) or any(type(v) is not int or not 1 <= v <= BATCH_SIZE for v in caps.values()):
            raise ValueError('Main lock must bind positive all18 ' + field)
        result.append(dict(caps))
    evaluation = record.get('eval_batch_size')
    if type(evaluation) is not int or not 1 <= evaluation <= SPLIT_ROWS['val']:
        raise ValueError('Main lock must bind eval_batch_size')
    return result[0], result[1], evaluation


def protocol_lock(args):
    path = Path(args.protocol_lock or args.root / 'protocol_lock.json').expanduser().resolve()
    record = read_json(path)
    mismatches = [k for k, v in required_lock_fields().items() if record.get(k) != v]
    if mismatches:
        raise ValueError('R113 main lock differs: ' + ', '.join(mismatches))
    caps, _, evaluation = locked_caps(record)
    for field, name in (('data_root', 'data_root'), ('prepared_root', 'prepared_root'),
                        ('primary_cache', 'cache_root'), ('pretrain_root', 'pretrain_root')):
        if Path(record[field]).resolve() != Path(getattr(args, name)).resolve():
            raise ValueError('R113 main lock names another ' + field)
    digest = record.get('process_manifest_sha256', '')
    if not isinstance(digest, str) or len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
        raise ValueError('Main lock must bind the final complete process manifest SHA256')
    if not record.get('process_manifest'):
        raise ValueError('Main lock must name the fixed process manifest')
    if (args.large_micro_cap is not None and args.large_micro_cap != caps['TSP']
            or args.eval_batch_size is not None and args.eval_batch_size != evaluation):
        raise ValueError('CLI cannot override main-locked execution sizes')
    args.large_micro_cap, args.eval_batch_size = caps['TSP'], evaluation
    return dict(path=str(path), sha256=file_hash(path), record=record)


def prepared_config(args):
    protocol_lock(args)
    ready, manifest, geometry, baselines = engine.prepared_config(args)
    for split in ('train', 'val'):
        for p in PROBLEMS:
            record, reference = manifest['splits'][split][p], manifest['splits']['train'][p]
            if (record.get('complete') is not True or record['pool'] != reference['pool']
                    or record['pool_ids'] != reference['pool_ids']
                    or baselines[p].get('selection_split') != 'train'):
                raise ValueError('R113 must retain complete frozen TRAIN/VAL pools and TRAIN SBS')
    return ready, manifest, geometry, baselines


@lru_cache(maxsize=8)
def _private_engine(binding):
    record = json.loads(binding)
    caps, _, _ = locked_caps(record)
    local = private_module(engine, 'r113_engine_' + json_hash(record)[:12])
    local.SPLIT_ROWS = copy.deepcopy(SPLIT_ROWS)
    for name in ('BATCH_SIZE', 'MAX_UPDATES', 'WARMUP_UPDATES', 'EVAL_EVERY', 'PEAK_LR', 'END_LR'):
        setattr(local, name, globals()[name])

    def micro_caps(gpu_type=None, large_cap=None):
        if large_cap is not None and large_cap != caps['TSP']:
            raise ValueError('No post-lock automatic cap shrink')
        return dict(caps)

    local.micro_caps, local.write_json = micro_caps, _engine_write_json
    return local


def private_engine(args=None, *, config=None):
    record = config['main_protocol_lock']['record'] if config is not None else protocol_lock(args)['record']
    return _private_engine(json.dumps(record, sort_keys=True))


def apply_pretraining(model, receipt, args):
    if args.arm == 'A_scratch':
        return dict(receipt, processor_pretraining_applied=False, processor_pretraining=None)
    lock = protocol_lock(args)
    path = args.pretrain_root / args.arm / 'processor.pt'
    value = torch.load(path, map_location='cpu', weights_only=False)
    config = process.validate_export(value, model, receipt, args.arm,
        manifest_sha256=lock['record']['process_manifest_sha256'], main_lock=lock)
    model.instance_encoder.processor.load_state_dict(value['processor'], strict=True)
    if nonprocessor_hash(model) != receipt['fresh_nonprocessor_sha256']:
        raise ValueError('Pretraining may replace the processor only')
    return dict(receipt, processor_pretraining_applied=True,
        trainable_sha256=parameter_hash(dict(model.named_parameters())),
        processor_pretraining=dict(path=str(path), file_sha256=file_hash(path), state_hash=value['state_hash'],
            processor_sha256=value['processor_sha256'], successful_updates=value['successful_updates'],
            mode=config['mode'], dataset=config['dataset'], source_sha256=config['source_sha256'],
            last_update_losses=value.get('last_update_losses'), pretraining_seconds=value.get('pretraining_seconds')))


def protocol(args, ready, manifest, geometry, baselines, params, order, caps, *, model, initialization):
    local = private_engine(args)
    config = local.protocol(args, ready, manifest, geometry, baselines, params, order, caps)
    config['loss'].pop('focal_extra', None)
    config.pop('common_trainable_sha256', None)
    config.update(experiment='R113', r113_schema_version=1, model_factory=MODEL_FACTORY,
        initialization=INITIALIZATION, main_protocol_lock=protocol_lock(args),
        model_spec=copy.deepcopy(params['r113_spec']), role_arm=MODEL_ROLE_ARM,
        trainable_parameter_count=sum(p.numel() for p in model.parameters() if p.requires_grad),
        initial_trainable_sha256=initialization['trainable_sha256'],
        initial_parameter_layout_sha256=initialization['parameter_layout_sha256'],
        fresh_state_sha256=initialization['fresh_state_sha256'],
        fresh_trainable_sha256=initialization['fresh_trainable_sha256'],
        fresh_nonprocessor_sha256=initialization['fresh_nonprocessor_sha256'],
        original_fresh_reference_sha256=ready['trainable_sha256'],
        common_parameter_sha256=initialization.get('common_parameter_sha256'),
        processor_pretraining=initialization.get('processor_pretraining'),
        cross_arm_initialization='identical complete fresh state; only own pretrained processor replaces B/C subset',
        train_eval_every_updates=TRAIN_EVERY, checkpoint_rule=CHECKPOINT_RULE, inference_rule=INFERENCE_RULE,
        selection_head='original_score_mlp_maximin_on_predicted_state_trajectory_only',
        selected_val_binding_rule='exact_model_protocol_and_update',
        source_plan=f'common seed2 full18 natural stream; exact{BATCH_SIZE} rows; shuffle tails carried',
        gradients=f'micro rows / logical{BATCH_SIZE}; one clip/step; same-batch/RNG AMP replay',
        train_eval=f'full TRAIN every{TRAIN_EVERY} updates plus selected checkpoint',
        objective=dict(classification='original R77 native CE', forward_passes=2, comparison=True,
            relative_risk=True, symmetric_rdrop=True, mutable_objective_state=False),
        current_query_labels_in_forward=False, historical_TRAIN_label_context_used=False,
        deployment_mode='transition', cache_policy='read-only frozen R84 caches; no implicit generation',
        oom_policy='fail; main-thread allocation/caps only',
        source_sha256={name: file_hash(V4 / name) for name in SOURCE_FILES})
    config = json.loads(json.dumps(config, allow_nan=False))
    validate_checkpoint_config(config)
    return config


def validate_checkpoint_config(config):
    record = config.get('main_protocol_lock', {}).get('record', {})
    caps, _, evaluation = locked_caps(record)
    params = config.get('model_params', {})
    expected = dict(experiment='R113', r113_schema_version=1, model_factory=MODEL_FACTORY,
        initialization=INITIALIZATION, seed=2, solver_names=list(GLOBAL_SOLVERS), rows_per_problem=SPLIT_ROWS,
        global_batch_size=BATCH_SIZE, logical_global_batch=BATCH_SIZE, max_successful_updates=MAX_UPDATES,
        warmup_updates=WARMUP_UPDATES, eval_every_updates=EVAL_EVERY, train_eval_every_updates=TRAIN_EVERY,
        eval_batch_size=evaluation, microbatch_caps=caps, optimizer='AdamW', weight_decay=1e-4,
        grad_clip=1., dropout=.1, rdrop_weight=.35, rdrop_ramp_updates=WARMUP_UPDATES,
        loss=engine.LOSS_RECIPE, checkpoint_rule=CHECKPOINT_RULE, inference_rule=INFERENCE_RULE,
        deployment_mode='transition', role_arm=MODEL_ROLE_ARM, current_query_labels_in_forward=False)
    if (config.get('arm') not in ARMS or any(config.get(k) != v for k, v in expected.items())
            or any(record.get(k) != v for k, v in required_lock_fields().items())
            or config.get('lr') != dict(rule='successful-update warmup then cosine, independent of VAL',
                peak=PEAK_LR, end=END_LR, warmup_updates=WARMUP_UPDATES, max_updates=MAX_UPDATES)
            or params.get('r113_arm') != config.get('arm') or params.get('dropout') != .1
            or config.get('model_spec') != params.get('r113_spec')
            or config.get('model_spec') != json.loads(json.dumps(models.SPEC))):
        raise ValueError('R113 checkpoint differs from fixed data/model/training/selection protocol')
    pretraining = config.get('processor_pretraining')
    if config['arm'] == 'A_scratch':
        if pretraining is not None:
            raise ValueError('Scratch cannot load a pretrained processor')
    elif (not isinstance(pretraining, dict) or pretraining.get('successful_updates') != process.MAX_UPDATES
            or pretraining.get('mode') != process.MODE[config['arm']]
            or pretraining.get('dataset', {}).get('manifest_sha256') != record.get('process_manifest_sha256')):
        raise ValueError('B/C requires its own complete processor export provenance')
    for field, locked in (('data_root', 'data_root'), ('prepared_root', 'prepared_root'), ('cache_root', 'primary_cache')):
        if Path(config[field]).resolve() != Path(record[locked]).resolve():
            raise ValueError('R113 embedded path differs: ' + field)


def load_checkpoint(path, device='cpu'):
    """Restore complete weights; never open a pretraining export or process NPZ."""
    state = torch.load(path, map_location='cpu', weights_only=False)
    config = state['args']
    validate_checkpoint_config(config)
    if state.get('state_hash') != json_hash(config) or 'objective_state' in state:
        raise ValueError('R113 checkpoint protocol/stateless objective differs')
    if config.get('source_sha256') != {name: file_hash(V4 / name) for name in SOURCE_FILES}:
        raise ValueError('Frozen R113 inference source changed')
    model = make_model(config['model_params'], config['arm'], state_dict=state['model'])
    if (sum(p.numel() for p in model.parameters() if p.requires_grad) != config['trainable_parameter_count']
            or parameter_hash(model.state_dict()) != parameter_hash(state['model'])
            or json_hash(process.tensor_schema(model.state_dict())) != config['initial_parameter_layout_sha256']):
        raise ValueError('R113 factory must restore all parameter/buffer tensors exactly')
    return model.to(device), state


load_model = from_checkpoint = load_checkpoint


def make_objective(selector, arm):
    if arm not in ARMS:
        raise ValueError('Unknown R113 arm')
    return RDropObjective(selector)


def update_batch(objective, optimizer, scaler, batch, weight, runtime, cap, *, problem, max_retries=20):
    if problem not in PROBLEMS or len(batch['ind']) != BATCH_SIZE:
        raise ValueError('R113 requires the complete natural logical batch')
    return engine.update_batch(objective, optimizer, scaler, batch, weight, runtime, cap, max_retries)


def selected_capture_proof(run, result, lock, split, state=None):
    run = Path(run)
    if file_hash(run / 'best.pt') != lock['checkpoint_sha256']:
        raise ValueError('Selected capture requires the immutable checkpoint')
    if state is None:
        state = torch.load(run / 'best.pt', map_location='cpu', weights_only=False)
    capture = result.get('archive_capture_identity', result['identity'])
    path = Path(result['prediction_archive']).resolve()
    expected = dict(experiment='R113', arm=lock['arm'], split=split,
        model_state_sha256=parameter_hash(state['model']), state_hash=lock['state_hash'],
        successful_updates=lock['selected_update'])
    if (not path.is_relative_to((run / 'evaluation' / split).resolve())
            or not path.is_file() or read_json(path.parent / 'identity.json') != capture
            or any(capture.get(k) != v for k, v in expected.items())
            or capture.get('checkpoint_sha256') not in (None, lock['checkpoint_sha256'])
            or state['args'].get('experiment') != 'R113' or state['args'].get('arm') != lock['arm']
            or state.get('state_hash') != lock['state_hash'] or json_hash(state['args']) != lock['state_hash']
            or state.get('successful_updates') != lock['selected_update']):
        raise ValueError('Selected prediction capture differs from checkpoint/model/protocol/update')
    return copy.deepcopy(capture)


def _engine_write_json(path, value):
    path = Path(path)
    if path.name == 'initialization.json':
        for arm in ARMS:
            other = path.parent.parent / arm / path.name
            if arm != path.parent.name and other.exists():
                previous = read_json(other)
                for key in ('fresh_reference_sha256', 'fresh_state_sha256', 'fresh_trainable_sha256',
                            'fresh_nonprocessor_sha256', 'initial_processor_sha256', 'parameter_layout_sha256'):
                    if previous.get(key) != value.get(key):
                        raise ValueError('R113 three-arm commonfresh initialization differs: ' + key)
    if path.name == 'final_metrics.json':
        value = copy.deepcopy(value)
        lock = read_json(path.parent / 'final_test_lock.json')
        if any(value.get(k) != lock[k] for k in ('state_hash', 'selected_update', 'checkpoint_sha256')):
            raise ValueError('Final metrics differ from unique VAL checkpoint')
        capture = selected_capture_proof(path.parent, value['val'], lock, 'val')
        value['val'].update(archive_capture_identity=capture,
            identity=dict(capture, checkpoint_sha256=lock['checkpoint_sha256']))
    return write_json(path, value)


def evaluate(selector, args, *positional, **keywords):
    return private_engine(args).evaluate(selector, args, *positional, **keywords)


def authorize_test(args):
    lock, config = private_engine(args).authorize_test(args)
    validate_checkpoint_config(config)
    if (protocol_lock(args) != config['main_protocol_lock']
            or lock.get('rule') != CHECKPOINT_RULE or lock.get('inference_rule') != INFERENCE_RULE):
        raise PermissionError('R113 TEST requires unchanged unique VAL/main/inference locks')
    return lock, config


def ensure_test_cache(args, lock):
    verified, config = authorize_test(args)
    if verified != lock:
        raise PermissionError('TEST request names another checkpoint')
    manifest = read_json(Path(config['prepared_root']) / 'source_cache_manifest.json')
    tasks = manifest.get('splits', {}).get('test', {})
    if set(tasks) != set(PROBLEMS) or any(tasks[p].get('rows') != SPLIT_ROWS['test']
                                        or tasks[p].get('complete') is not True for p in PROBLEMS):
        raise ValueError('Preexisting complete full18 R84 TEST cache required; no implicit generation')
    return manifest


def test(args, device):
    return private_engine(args).test(args, device, integration=sys.modules[__name__])


def train(args, device):
    """Explicit R84-style loop, not exec/source-replacement of historical code."""
    local, root = private_engine(args), args.root / args.arm
    if (root / 'training_complete.json').exists():
        return test(args, device)
    ready, manifest, geometry, baselines = prepared_config(args)
    order, caps = task_order(MAX_UPDATES), local.micro_caps()
    last, checkpoint = root / 'last.pt', None
    if last.exists():
        if not args.resume:
            raise ValueError('Use --resume; existing R113 state is never overwritten')
        selector, checkpoint = load_checkpoint(last, device)
        initialization = checkpoint['initialization']
        params = checkpoint['args']['model_params']
    else:
        initial = torch.load(args.prepared_root / 'identicalinit_seed2.pt', map_location='cpu', weights_only=False)
        selector, params, initialization = initialize_shared(initial, geometry, args.arm, device)
        initialization = apply_pretraining(selector, initialization, args)
        del initial
    config = protocol(args, ready, manifest, geometry, baselines, params, order, caps,
        model=selector, initialization=initialization)
    state_hash = json_hash(config)
    if args.state_hash is not None and args.state_hash != state_hash:
        raise ValueError('Expected R113 protocol state hash differs')
    if (root / 'protocol.json').exists() and read_json(root / 'protocol.json') != config:
        raise ValueError('Existing R113 protocol differs; preserve this version')
    write_json(root / 'protocol.json', config)
    if checkpoint is not None:
        local.validate_resume(checkpoint, config, order)
    else:
        _engine_write_json(root / 'initialization.json', initialization)
    optimizer = torch.optim.AdamW(selector.parameters(), lr=PEAK_LR, weight_decay=1e-4)
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == 'cuda')
    objective, runtime = make_objective(selector, args.arm), Runtime(0, 1, device)
    history, best, successful, retries_total, pending = [], None, 0, 0, None
    counts, cursors = Counter(), dict.fromkeys(PROBLEMS, 0)
    if checkpoint is not None:
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
        engine.set_seed(2)
    del checkpoint
    streams = {p: ShuffleStream(SPLIT_ROWS['train'], task_seed(2, p), cursors[p]) for p in PROBLEMS}
    caches, tracker = {}, None

    def snapshot():
        return dict(model=selector.state_dict(), optimizer=optimizer.state_dict(), scaler=scaler.state_dict(),
            args=config, state_hash=state_hash, initialization=initialization, epoch=len(history),
            successful_updates=successful, updates_by_problem=dict(counts), sample_exposures=successful * BATCH_SIZE,
            stream_draws={p: s.draws for p, s in streams.items()}, amp_retries=retries_total,
            rng=capture_rank_rng(device), history=history, best=best, pending_evaluation=pending)

    if not last.exists():
        save_checkpoint(last, snapshot())
    try:
        rng = capture_rank_rng(device)
        tracker = local.wandb_run(args, config)
        restore_rank_rng(rng, device)
        while successful < MAX_UPDATES or pending is not None:
            if pending is None:
                if not caches:
                    for p in PROBLEMS:
                        caches[p] = open_cache(args.cache_root, 'train', p, manifest, args.data_root)
                        if caches[p].size != SPLIT_ROWS['train']:
                            raise ValueError('Keep every frozen TRAIN row')
                end, first = min(MAX_UPDATES, successful + EVAL_EVERY), successful
                requests = [(p, streams[p].take(BATCH_SIZE), BATCH_SIZE) for p in order[successful:end]]
                started = time.monotonic()
                weighted = torch.zeros(len(PART_KEYS), dtype=torch.float64, device=device)
                selector.train()
                with PinnedPrefetch(caches, requests, device) as batches:
                    for problem, n_rank, n_global, batch in batches:
                        if n_rank != BATCH_SIZE or n_global != BATCH_SIZE:
                            raise ValueError('Each R113 update requires exactly512 natural rows')
                        lr, weight = local.learning_rate(successful + 1), local.consistency_weight(successful + 1)
                        for group in optimizer.param_groups:
                            group['lr'] = lr
                        parts, retries, _ = update_batch(objective, optimizer, scaler, batch,
                            weight, runtime, caps[problem], problem=problem)
                        successful += 1
                        counts[problem] += 1
                        retries_total += retries
                        weighted.add_(parts.double())
                        if successful == 1 or successful % args.log_every == 0 or successful == end:
                            done, elapsed = successful - first, time.monotonic() - started
                            info = dict(successful_updates=successful, max_updates=MAX_UPDATES, problem=problem,
                                lr=lr, consistency_weight=weight, sample_exposures=successful * BATCH_SIZE,
                                seconds_per_update=elapsed / done, samples_per_second=done * BATCH_SIZE / elapsed,
                                amp_retries=retries_total, updates_by_problem=dict(counts),
                                **dict(zip(PART_KEYS, (weighted / done).cpu().tolist())))
                            local.status_update(args, 'training', successful,
                                **{k: v for k, v in info.items() if k != 'successful_updates'})
                            with (root / 'updates.jsonl').open('a') as stream:
                                stream.write(json.dumps(dict(updated_utc=utc(), **info), allow_nan=False) + '\n')
                            print('[R113 train] ' + json.dumps(dict(arm=args.arm, **info)), flush=True)
                            tracker.log({**{'optimization/' + k: info[k] for k in PART_KEYS},
                                'successful_updates': successful, 'lr': lr}, step=successful)
                pending = dict(zip(PART_KEYS, (weighted / (successful - first)).cpu().tolist()))
                pending.update(lr=lr, consistency_weight=weight, amp_retries=retries_total,
                    samples=(successful - first) * BATCH_SIZE, seconds=time.monotonic() - started)
                save_checkpoint(last, snapshot())
            point, tag = successful // EVAL_EVERY, f'update{successful:07d}'
            validation = evaluate(selector, args, manifest, baselines, 'val', tag, successful, state_hash)
            key = selection_key(validation)
            training = evaluate(selector, args, manifest, baselines, 'train', tag, successful, state_hash) if successful % TRAIN_EVERY == 0 else None
            row = dict(epoch=point, evaluation_point=point, successful_updates=successful,
                sample_exposures=successful * BATCH_SIZE, updates_by_problem=dict(counts),
                evaluation_tag=tag, optimization=pending, val=validation, train=training)
            history.append(row)
            improved = best is None or key < selection_key(best['val'])
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
            local.write_metrics_csv(root, history)
            local.status_update(args, 'evaluation_complete', successful, latest_val=validation['macro'],
                latest_val_update=successful, selected_update=best['successful_updates'])
            tracker.log({**{'val/' + k: v for k, v in validation['macro'].items()},
                'successful_updates': successful}, step=successful)
        if successful != MAX_UPDATES or any(counts[p] != MAX_UPDATES // len(PROBLEMS) for p in PROBLEMS):
            raise ValueError('Fixed full18 successful-update budget incomplete')
        for cache in caches.values():
            cache.close()
        caches.clear()
        del objective, optimizer, scaler, selector
        selector, selected = load_checkpoint(root / 'best.pt', device)
        if selected['args'] != config or selected['successful_updates'] != best['successful_updates']:
            raise ValueError('Selected checkpoint differs from VAL history')
        del selected
        lock = local.lock_best(root, best, config, state_hash)
        destination = root / 'evaluation/val/selected_best'
        destination.mkdir(parents=True, exist_ok=True)
        for name in ('rank000.npz', 'metrics.json', 'identity.json'):
            source, target = root / 'evaluation/val' / best['evaluation_tag'] / name, destination / name
            if target.exists():
                if file_hash(target) != file_hash(source):
                    raise ValueError('Existing selected VAL copy differs')
            else:
                shutil.copyfile(source, target)
        if best['train'] is None:
            best = dict(best, train=evaluate(selector, args, manifest, baselines, 'train', 'selected_best',
                best['successful_updates'], state_hash, lock['checkpoint_sha256']))
        for row in history:
            if row['successful_updates'] == best['successful_updates']:
                row['train'] = best['train']
        write_json(root / 'history.json', history)
        write_json(root / 'best_eval.json', best)
        local.write_metrics_csv(root, history)
        write_json(root / 'result.json', dict(best=best, final=history[-1], successful_updates=successful,
            updates_by_problem=dict(counts), amp_retries=retries_total, state_hash=state_hash,
            stop_reason='fixed successful-update budget; no early stopping', test_read=False))
        _engine_write_json(root / 'final_metrics.json', dict(epoch=best['epoch'], selected_update=best['successful_updates'],
            checkpoint_sha256=lock['checkpoint_sha256'], state_hash=state_hash, train=best['train'], val=best['val'], test_read=False))
        write_json(root / 'training_complete.json', dict(completed_utc=utc(), successful_updates=successful,
            sample_exposures=successful * BATCH_SIZE, updates_by_problem=dict(counts), selected_best=lock,
            state_hash=state_hash, next_stage='locked full TEST', test_read=False))
        local.status_update(args, 'training_complete', successful, selected_update=best['successful_updates'])
        del selector
    finally:
        for cache in caches.values():
            cache.close()
        if tracker is not None:
            tracker.finish()
    return test(args, device)


@torch.no_grad()
def timing(args, device):
    if device.type != 'cuda':
        raise ValueError('Formal timing requires the main-thread assigned CUDA device')
    from .r113_reporting import inference_timing
    lock, config = authorize_test(args)
    run = args.root / args.arm
    saved = read_json(run / 'test_results.json')
    if (saved.get('complete') is not True or saved.get('checkpoint_sha256') != lock['checkpoint_sha256']
            or saved.get('state_hash') != lock['state_hash']):
        raise PermissionError('Timing follows complete same-checkpoint TEST')
    model_hash = saved['result']['identity']['model_state_sha256']
    if (run / 'inference_timing.json').exists():
        return inference_timing(run, config, lock, model_hash)
    model, state = load_checkpoint(run / 'best.pt', device)
    if parameter_hash(state['model']) != model_hash:
        raise ValueError('Timing and TEST must use the same complete model')
    del state
    model.eval()
    manifest, count, rows = ensure_test_cache(args, lock), config['eval_batch_size'], []
    for problem in PROBLEMS:
        batch = cached_input_batch(config, manifest, 'test', problem, device, count=count)
        rows.append(dict(problem=problem, instances=count, standalone_model=measure_inputs(model, batch, device)))
        del batch
    if file_hash(run / 'best.pt') != lock['checkpoint_sha256'] or parameter_hash(model.state_dict()) != model_hash:
        raise ValueError('Timing changed the locked checkpoint/model')
    write_json(run / 'inference_timing.json', dict(experiment='R113', arm=args.arm, complete=True,
        completed_utc=utc(), checkpoint_sha256=lock['checkpoint_sha256'], state_hash=lock['state_hash'],
        selected_update=lock['selected_update'], model_state_sha256=model_hash, model_factory=MODEL_FACTORY,
        inference_rule=INFERENCE_RULE, device=str(device), device_name=torch.cuda.get_device_name(device),
        repetitions=3, warmup_batches=1, eval_batch_size=count, parameter_count=config['trainable_parameter_count'],
        scope='complete_standalone_model_original_GPU_resident_inputs', sample_policy=f'first{count} frozen TEST inputs per problem',
        all_current_model_included=True, end_to_end_measured=False, input_labels_read=False,
        excludes='dataset I/O, CPU-to-GPU transfer, training and solver execution',
        per_problem=rows, model_reselection=False))
    return inference_timing(run, config, lock, model_hash)


def finalize(args, device):
    result = test(args, device)
    timing(args, device)
    from . import r113_reporting
    from .r113_diagnostics import finalize_diagnostics
    r113_reporting.lib_stage(args)
    diagnostics = finalize_diagnostics(args)
    return dict(test=result, process_diagnostics=diagnostics, report=r113_reporting.report(args))


def describe(args):
    path = args.protocol_lock or args.root / 'protocol_lock.json'
    lock = protocol_lock(args) if path.exists() else None
    return dict(experiment='R113', arm=args.arm, required_protocol_lock=required_lock_fields(),
        required_execution_fields=['data_root', 'prepared_root', 'primary_cache', 'pretrain_root',
            'process_manifest', 'process_manifest_sha256 (complete full18)',
            'microbatch_caps (all18)', 'pretrain_microbatch_caps (all18)', 'eval_batch_size'],
        main_protocol_lock=lock, model_factory=MODEL_FACTORY,
        model_spec=copy.deepcopy(models.SPEC),
        inference_source_sha256={name: file_hash(V4 / name) for name in SOURCE_FILES},
        pretrain_source_sha256={name: file_hash(V4 / name) for name in process.SOURCE_FILES},
        reporting_source_sha256={name: file_hash(V4 / name) for name in
            ('r113_reporting.py', 'r113_diagnostics.py', 'r84_reporting.py', 'r80_reporting.py', 'r77_best_model_report.py')},
        model_api='ARMS; make_model(params,arm,state_dict); initialize_shared(initial,geometry,arm,device)',
        process_manifest=str(DEFAULT_PROCESS_MANIFEST), pretrain_root=str(args.pretrain_root),
        caps_status='locked' if lock else 'awaiting_main_bindings', training_started=False,
        wandb_mode=args.wandb_mode)


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('command', choices=('pretrain', 'train', 'test', 'timing', 'finalize', 'describe', 'status'))
    result.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    result.add_argument('--arm', choices=ARMS, required=True)
    result.add_argument('--data-root', type=Path, default=DEFAULT_DATA_ROOT)
    result.add_argument('--prepared-root', type=Path, default=DEFAULT_PREPARED_ROOT)
    result.add_argument('--cache-root', type=Path, default=DEFAULT_CACHE_ROOT)
    result.add_argument('--pretrain-root', type=Path)
    result.add_argument('--protocol-lock', type=Path)
    result.add_argument('--device', default='cuda:0')
    result.add_argument('--gpu-type', default='external')
    result.add_argument('--large-micro-cap', type=int)
    result.add_argument('--eval-batch-size', type=int)
    result.add_argument('--log-every', type=int, default=18)
    result.add_argument('--wandb-mode', choices=('auto', 'online', 'offline'), default='auto')
    result.add_argument('--resume', action='store_true')
    result.add_argument('--state-hash', '--expected-state-hash')
    result.add_argument('--lib-batch-size', type=int)
    result.add_argument('--nss-root', type=Path, default=V4 / 'runs/R84_new90k_attention/nss_reference')
    result.set_defaults(experiment='R113')
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    args.pretrain_root = args.pretrain_root or args.root / 'pretrain'
    for name in ('root', 'data_root', 'prepared_root', 'cache_root', 'pretrain_root', 'protocol_lock', 'nss_root'):
        value = getattr(args, name)
        if value is not None:
            setattr(args, name, value.expanduser().resolve())
    if args.command in ('describe', 'status'):
        base = args.pretrain_root if args.command == 'status' and args.arm in process.ARMS else args.root
        path = args.root / args.arm / 'status.json'
        if not path.exists():
            path = base / args.arm / 'status.json'
        value = describe(args) if args.command == 'describe' else (read_json(path) if path.exists()
            else dict(experiment='R113', arm=args.arm, phase='not_started'))
        print(json.dumps(value, indent=2))
        return value
    protocol_lock(args)
    if args.log_every < 1 or args.lib_batch_size is not None and args.lib_batch_size < 1:
        raise ValueError('Log interval and LIB batch must be positive')
    device = torch.device(args.device)
    if device.type != 'cuda':
        raise ValueError('Formal execution requires a main-thread assigned CUDA device; CPU fixtures use functions')
    torch.set_num_threads(2)
    engine.configure_torch()
    if args.command == 'pretrain':
        try:
            return process.run(args, device)
        except BaseException as error:
            write_json(args.pretrain_root / args.arm / 'error.json', dict(experiment='R113_pretrain',
                arm=args.arm, updated_utc=utc(), error_type=type(error).__name__, message=str(error)))
            raise
    root = args.root / args.arm
    root.mkdir(parents=True, exist_ok=True)
    with (root / 'runner.lock').open('a') as claim:
        fcntl.flock(claim, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            return globals()[args.command](args, device)
        except BaseException as error:
            write_json(root / 'error.json', dict(experiment='R113', arm=args.arm, updated_utc=utc(),
                command=args.command, error_type=type(error).__name__, message=str(error)))
            raise


if __name__ == '__main__':
    main()
