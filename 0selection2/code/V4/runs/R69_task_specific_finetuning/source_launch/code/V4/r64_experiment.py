"""R64 full18 paired mean-gradient/PCGrad training; TEST requires a later parent lock."""

import argparse
import gc
import json
import math
import os
import random
import shutil
import socket
import sys
import threading
import time
from collections import Counter
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from code.unified_selector.registry import GLOBAL_SOLVERS, PROBLEMS
from .multitask_probe import Tee, file_hash, gather_batch, state_hash
from .pairwise_objective import inputs_only
from .performance_experiment import SOURCE_FILES
from .r40_experiment import aggregate, families
from .r42_experiment import CostController, paired_schedule, prepare_data
from .r43_experiment import policy_metrics
from .r59_experiment import (GPU_UUID, REFERENCE, R43_ARGS, WORKSPACE, atomic_json,
                             atomic_save, check_original_scope, publish_best)
from .r64_gradients import GradientLayout, SHARED_PREFIXES, update_triple
from .r64_model import INFERENCE_RULE, initialize
from .train import configure_torch, print_eval, set_seed
from .training_monitor import capture_rng, restore_rng


ROOT = WORKSPACE / 'code/V4/runs/R64_pcgrad_full18'
RUN_NAMES = {'A': 'A_mean_gradient_seed2', 'B': 'B_pcgrad_seed2'}
PRESENTATIONS_PER_EPOCH = 180000
ADOPTED_PLAN = WORKSPACE / 'code/V4/runs/R63_recurrent_workspaces/reviews/R63_closure_R64_plan.response.md'
BATCH_AMENDMENT = WORKSPACE / '.aris/traces/R64_large_batch_plan.response.md'
ABORTED_ROOT = WORKSPACE / 'code/V4/runs/R64_pcgrad_full18_aborted_batch42_20261012T140928'
THROUGHPUT_TIE_FRACTION, REQUIRED_HEADROOM = .03, .15
NEW_SOURCES = ('r64_model.py', 'r64_gradients.py', 'r64_experiment.py', 'r64_analysis.py',
               'test_r64.py', 'run_v4_r64.sh', 'R64_README.md', 'r45_experiment.py',
               'r59_experiment.py', 'r43_experiment.py', 'r42_experiment.py', 'r40_experiment.py',
               'solver_code_encoder.py', 'pairwise_selector.py', 'pairwise_objective.py')


def append_jsonl(path, value):
    with path.open('a') as stream:
        stream.write(json.dumps(value, allow_nan=False) + '\n')
        stream.flush()


def batch_protocol(batch_size):
    if not 1 <= batch_size <= 10000:
        raise ValueError('Per-task batch must retain real original TRAIN rows')
    cycles = math.ceil(10000 / batch_size)
    tail = 10000 - batch_size * (cycles-1)
    return dict(batch_size=batch_size, effective_batch_size=3*batch_size,
                final_effective_batch_size=3*tail, tail_per_task=tail,
                cycles_per_epoch=cycles, updates_per_epoch=6*cycles,
                warmup_updates=3*6*cycles, maximum_updates_per_arm=40*6*cycles)


def paired_triples(sizes, epochs=40, seed=2, batch_size=256):
    if len(sizes) != 18 or set(sizes.values()) != {10000}:
        raise ValueError('R64 requires exactly 18 original 10000-row TRAIN tasks')
    batch = batch_protocol(batch_size)
    schedule = paired_schedule(sizes, epochs, batch_size=batch_size, seed=seed)
    cycles, triples = [], []
    for order in schedule['orders']:
        epoch_cycles = [order[i:i+18] for i in range(0, len(order), 18)]
        if len(epoch_cycles) != batch['cycles_per_epoch'] or any(len(c) != 18 or set(c) != set(sizes) for c in epoch_cycles):
            raise ValueError('The existing paired schedule must really shuffle all18 once per cycle')
        cycles.append(epoch_cycles)
        triples.append([cycle[i:i+3] for cycle in epoch_cycles for i in range(0, 18, 3)])
        if len(triples[-1]) != batch['updates_per_epoch']:
            raise ValueError('R64 requires exactly six triples in each complete all18-task cycle')
    return dict(permutations=schedule['permutations'], cycles=cycles, triples=triples,
                **batch)


def triple_indices(schedule, epoch, step):
    cycle, batch_size = step // 6, schedule['batch_size']
    return [(task, schedule['permutations'][epoch][task][cycle*batch_size:cycle*batch_size+batch_size])
            for task in schedule['triples'][epoch][step]]


def prefer_checkpoint(candidate, previous):
    count = candidate['native_correct']
    if type(count) is not int or not 0 <= count <= candidate['native_total']:
        raise ValueError('Validation checkpoint selection requires integer native correct counts')
    return previous is None or count > previous['native_correct']


def dropout_seed(successful):
    return 500000003 + 2 * 100003 + successful


def planned_config(args):
    batch = batch_protocol(args.batch_size)
    return dict(seed=2, arms=RUN_NAMES, problems=list(PROBLEMS), solver_names=list(GLOBAL_SOLVERS),
        max_epochs=40, min_epochs=15, **batch, triples_per_cycle=6,
        presentations_per_epoch=PRESENTATIONS_PER_EPOCH,
        maximum_presentations_per_arm=7200000, eval_batch_size=args.eval_batch_size,
        learning_rate=1e-4, weight_decay=1e-4, adamw_betas=[.9, .999], adamw_epsilon=1e-8,
        clip_norm=1., rdrop_weight=.35, rdrop_ramp_updates=batch['warmup_updates'],
        dropout=.1, full_task_batch_forwards=True, dropout_passes=2,
        loss='nativeCE mean of two independent dropout forwards + existing legal-pool symmetric KL',
        shared_prefixes=list(SHARED_PREFIXES), projection_rng_seed=6402,
        gradients='same-state task gradients; unscale once into FP32; selective PCGrad or arithmetic mean; one global clip and AdamW',
        retries='whole triple; restore Python/NumPy/CPU/CUDA and independent projection RNG; reduce scale; no cursor/update advancement',
        amp='FP16 autocast and GradScaler.scale; explicit FP32 task unscale; direct optimizer.step; explicit successful scale-growth counter',
        dropout_stream='torch/Python/NumPy seeded by 500000003 + 2*100003 + zero-based successful update; retry restores all RNG',
        scheduler='A controls actual LR-by-successful-update trace and epoch horizon; B replays exactly',
        top1_min_delta_pp=.001, lr_patience=3, lr_factor=.5, stop_patience=8,
        checkpoint_rule='strict maximum integer total native validation correct /18000; exact ties retain earliest epoch; never cost',
        official_arm_rule='highest validation-selected integer correct count; exact tie A; parent freezes after both arms',
        inference_rule=INFERENCE_RULE, train_eval='complete eval-mode original TRAIN every5 epochs and termination/selected checkpoint',
        full_coverage=True, drop_last=False, no_augmentation=True, geometry_fitted_on='original TRAIN only',
        train_prediction_saves='every5, termination, selected checkpoint', val_prediction_saves='every epoch and selected checkpoint',
        checkpoint_every_successful_updates=60, inference_timing='once during final selected validation; first16/task singleton, remainder batched',
        wandb=dict(project='selector', entity='yjkds-southern-university-of-science-technology', mode='offline'),
        adopted_plan=str(ADOPTED_PLAN.relative_to(WORKSPACE)), adopted_plan_sha256=file_hash(ADOPTED_PLAN),
        adopted_batch_amendment=str(BATCH_AMENDMENT.relative_to(WORKSPACE)),
        adopted_batch_amendment_sha256=file_hash(BATCH_AMENDMENT),
        batch_decision_sha256=file_hash(args.root/'large_batch_decision.json'),
        planner_boundary='Astra/max text/context planning; no direct local-code audit or GPU execution',
        aborted_batch42_archive=str(ABORTED_ROOT), fresh_restart_both_arms=True,
        authorized_train_only_batch_sizing=True, learning_rate_scaled_with_batch=False,
        optimizer_update_budget_reduced_from_aborted_57360=True,
        legacy_B_semantics='unresolved and preserved; R58 deferred', target_native_macro_top1=.70,
        test_read=False, gpu_preflight=False, automatic_analysis=False)


def source_hashes():
    paths = [Path(__file__).parent / n for n in sorted(set(SOURCE_FILES + NEW_SOURCES))]
    paths += [WORKSPACE / 'code/unified_selector' / n for n in ('registry.py', 'data.py')]
    return {str(p.relative_to(WORKSPACE)): file_hash(p) for p in paths}


def load_inputs(args):
    loaders, raw, geometry = prepare_data(args)
    check_original_scope(loaders, raw)
    if len(GLOBAL_SOLVERS) != 22 or sum(len(r['pool_ids']) for r in raw['train'].values()) != 128:
        raise ValueError('R64 must retain all22 identities and all128 original legal deployments')
    return loaders, raw, geometry


def prepare(args, inputs=None):
    protocol = planned_config(args)
    protocol.update(source_hashes=source_hashes(), r43_args_sha256=file_hash(R43_ARGS))
    for relative, digest in protocol['source_hashes'].items():
        destination = args.root / 'source_launch' / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists() and file_hash(destination) != digest:
            raise ValueError('Preserve the existing R64 launch snapshot: ' + str(destination))
        if not destination.exists():
            shutil.copy2(WORKSPACE / relative, destination)
    atomic_json(args.root / 'source_launch/hashes.json', protocol['source_hashes'])
    loaders, raw, geometry = load_inputs(args) if inputs is None else inputs
    sizes = {p: l.size for p, l in loaders['train'].items()}
    schedule = paired_triples(sizes, batch_size=args.batch_size)
    atomic_save(args.root / 'schedule_seed2.pt', schedule)
    model, params = initialize(geometry, args.device)
    initial = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    layout = GradientLayout(model)
    atomic_save(args.root / 'selector_initial.pt', initial)
    protocol.update(initial_model_hash=state_hash(initial), model_params=params, sizes=sizes,
        trainable_parameters=sum(p.numel() for p in model.parameters()),
        gradient_layout=layout.receipt, geometry_stats=geometry, schedule_hash=state_hash(schedule),
        data_manifest=json.loads((args.root / 'data_manifest.json').read_text()))
    atomic_json(args.root / 'initialization.json', dict(seed=2, fresh=True, shared_state_sha256=protocol['initial_model_hash'],
        no_historical_weights=True, projected_parameters=layout.receipt))
    atomic_json(args.root / 'protocol.json', protocol)
    del model, layout
    batch = batch_protocol(args.batch_size)
    plan = ('# R64 adopted larger real-batch restart\n\n'
            f'Both arms restart fresh at seed2. Old run is aborted and immutable: `{ABORTED_ROOT}`.\n'
            f'Adopted amendment: `{BATCH_AMENDMENT}`; verbatim response stored alongside this plan.\n'
            'Astra/max supplied text/context planning, not a direct local-code audit.\n\n'
            f'Locked per-task batch {args.batch_size}, effective batch {batch["effective_batch_size"]}; '
            f'{batch["cycles_per_epoch"]} all18-task cycles, six triples each, '
            f'{batch["updates_per_epoch"]} updates/epoch. Final task tail {batch["tail_per_task"]}, '
            f'final effective batch {batch["final_effective_batch_size"]}. All180000 rows/epoch retained.\n'
            f'Warmup/R-Drop ramp {batch["warmup_updates"]} successful updates (three epochs). '
            f'Max40 epochs/{batch["maximum_updates_per_arm"]} updates/7.2M presentations per arm. '
            'The reduced optimizer-update budget is part of this amendment.\n\n'
            'Unchanged LR1e-4, AdamW WD1e-4/betas(.9,.999)/eps1e-8, global clip1, nativeCE + '
            '.35 symmetric R-Drop; no microbatching/accumulation or automatic LR scaling. '
            'Same R45A architecture, original full18/fullpools/native winners and directed ATSP inputs. '
            'A averages; B alone projects the adopted shared dense prefixes. A controls actual LR/horizon; '
            'B replays exactly. Integer validation correct counts choose best; exact ties earlier, official arm tieA.\n\n'
            'Selection used only the predeclared bounded TRAIN capacity/throughput pass, '
            'two warmup plus three timed complete updates per arm/size, at most three sizes. '
            'Eligibility requires at least15% physical memory headroom; highest PCGrad throughput '
            'with3% timing ties toward smaller batch. All calibration states/updates are discarded and costed separately.\n\n'
            'Full validation every epoch; eval-mode full TRAIN every5 and termination/selected checkpoint. '
            'No automatic TEST/analysis/git/review gate. Parent freezes both completed validation-selected models, '
            'inference and official arm before authorizing one full TEST/model; existing R45A/R60B tests reused. '
            'R58 and unresolved legacy B semantics remain deferred; single-solver70% goal active.\n')
    (args.root/'PLAN.md').write_text(plan)
    atomic_json(args.root/'planned_configs.json', protocol)
    print('[R64 prepared] locked batch ' + json.dumps(batch) + '; next update is fresh FORMAL', flush=True)
    return protocol, loaders, raw, geometry, schedule


def load_training_data(args):
    protocol = json.loads((args.root / 'protocol.json').read_text())
    if (protocol['source_hashes'] != source_hashes() or protocol['r43_args_sha256'] != file_hash(R43_ARGS)
            or protocol['eval_batch_size'] != args.eval_batch_size
            or protocol['batch_size'] != args.batch_size
            or protocol['adopted_plan_sha256'] != file_hash(ADOPTED_PLAN)
            or protocol['adopted_batch_amendment_sha256'] != file_hash(BATCH_AMENDMENT)
            or protocol['batch_decision_sha256'] != file_hash(args.root/'large_batch_decision.json')):
        raise ValueError('R64 frozen source/configuration changed after preparation')
    loaders, raw, geometry = load_inputs(args)
    if (geometry != protocol['geometry_stats']
            or json.loads((args.root / 'data_manifest.json').read_text()) != protocol['data_manifest']):
        raise ValueError('R64 original data or TRAIN-fitted geometry changed')
    schedule = torch.load(args.root / 'schedule_seed2.pt', map_location='cpu', weights_only=False)
    if state_hash(schedule) != protocol['schedule_hash']:
        raise ValueError('R64 shared paired batches changed')
    return protocol, loaders, raw, geometry, schedule


def synchronize(device):
    if str(device).startswith('cuda'):
        torch.cuda.synchronize(device)


@torch.no_grad()
def evaluate(model, loaders, raw, batch_size=128, prediction_dir=None, timing_path=None):
    was_training, results, timings = model.training, {}, {}
    model.eval()
    started = time.monotonic()
    if prediction_dir is not None:
        prediction_dir.mkdir(parents=True, exist_ok=True)
    try:
        for problem, loader in loaders.items():
            device, scores, measured = loader.batch['ind'].device, [], []
            singleton = min(16, loader.size) if timing_path is not None else 0
            groups = [torch.tensor([i]) for i in range(singleton)]
            groups += [torch.arange(i, min(i+batch_size, loader.size))
                       for i in range(singleton, loader.size, batch_size)]
            for indices in groups:
                synchronize(device)
                before = time.monotonic()
                batch = inputs_only(gather_batch(loader, indices))
                output = model(batch)['logits'].float().cpu()
                scores.append(output)
                if timing_path is not None:
                    synchronize(device)
                    measured.append(dict(queries=len(indices), seconds=time.monotonic()-before,
                                         indices=indices.tolist(), true_sizes=loader.lengths[indices].tolist()))
            scores = torch.cat(scores).numpy()
            if len(scores) != loader.size:
                raise ValueError('R64 evaluation must retain every original row and tail')
            result, pred = policy_metrics(scores, raw[problem])
            result['native_correct'] = int(np.count_nonzero(pred == raw[problem]['winner']))
            results[problem] = result
            if prediction_dir is not None:
                np.savez_compressed(prediction_dir / f'{problem}.npz', indices=np.arange(loader.size),
                    logits=scores, pred=pred, winner=raw[problem]['winner'], costs=raw[problem]['costs'],
                    pool_ids=np.array(raw[problem]['pool_ids']), pool=np.array(raw[problem]['pool']),
                    nodes=loader.lengths.numpy())
            if timing_path is not None:
                timings[problem] = dict(queries=loader.size, singleton=measured[:singleton],
                                       batched=measured[singleton:])
        value = dict(per_problem=results, macro=aggregate(results), families=families(results),
                     native_correct=sum(r['native_correct'] for r in results.values()),
                     native_total=sum(r['n'] for r in results.values()), seconds=time.monotonic()-started)
        value['macro']['macro_top1'] = value['native_correct'] / value['native_total']
        if timing_path is not None:
            atomic_json(timing_path, dict(inference_rule=INFERENCE_RULE, per_problem=timings,
                scope='one ordinary selected validation evaluation; each query exactly once; cached input gather, forward, CPU output transfer',
                excludes=['disk decode', 'geometry cache preparation', 'checkpoint load', 'solver execution'],
                seconds=value['seconds']))
        return value
    finally:
        model.train(was_training)


def shared_replay_schedule(root):
    directory = root / RUN_NAMES['A']
    result = json.loads((directory / 'result.json').read_text())
    checkpoint = torch.load(directory / 'last.pt', map_location='cpu', weights_only=False)
    trace = checkpoint['lr_trace']
    value = dict(controller_arm='A', epochs=result['epochs_completed'],
        successful_updates=result['successful_updates'], presentations=result['presentations'],
        batch_size=checkpoint['args']['batch_size'],
        updates_per_epoch=checkpoint['args']['updates_per_epoch'], warmup_updates=checkpoint['args']['warmup_updates'],
        lr_by_successful_update=trace,
        controller_trace=[dict(epoch=r['epoch'], native_correct=r['val']['native_correct'],
                               lr_first=r['optimization']['lr_first'], lr_last=r['optimization']['lr'],
                               halve=r['controller']['halve'], stop=r['controller']['stop']) for r in checkpoint['history']],
        paired_schedule_hash=checkpoint['args']['schedule_hash'],
        dropout_seed_rule=checkpoint['args']['dropout_stream'])
    if len(trace) != value['successful_updates'] or len(trace) != value['epochs'] * value['updates_per_epoch']:
        raise ValueError('Incomplete A-controlled actual update LR trace')
    path = root / 'replay_schedule.json'
    if path.exists() and json.loads(path.read_text()) != value:
        raise ValueError('R64 actual shared LR trace/horizon is immutable')
    atomic_json(path, value)
    return value


def executed_updates(directory):
    path = directory / 'updates.jsonl'
    count = retries = samples = 0
    if path.exists():
        with path.open() as stream:
            for line in stream:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                count += 1
                retries += row['retries']
                samples += row['rows']
    return dict(recorded_successful_optimizer_steps=count, recorded_query_presentations=samples,
                recorded_amp_retries=retries)


def train(arm, args, directory, data, launch):
    protocol, loaders, raw, geometry, schedule = data
    updates_per_epoch, warmup_updates = protocol['updates_per_epoch'], protocol['warmup_updates']
    shared = shared_replay_schedule(args.root) if arm == 'B' else None
    horizon = shared['epochs'] if shared is not None else 40
    model, params = initialize(geometry, args.device)
    layout = GradientLayout(model)
    if (params != protocol['model_params'] or layout.receipt != protocol['gradient_layout']
            or state_hash(model.state_dict()) != protocol['initial_model_hash']):
        raise ValueError('R64 arm did not start from the identical fresh seed2 complete state')
    initial = torch.load(args.root / 'selector_initial.pt', map_location='cpu', weights_only=False)
    if state_hash(initial) != protocol['initial_model_hash']:
        raise ValueError('R64 shared initialization changed')
    model.load_state_dict(initial, strict=True)
    atomic_json(directory / 'initialization.json', dict(seed=2, fresh_state_hash=state_hash(initial),
        shared_initial_model_hash=protocol['initial_model_hash'], matches=True))
    config = dict(protocol, group=arm, model_params=params, replay_trace_sha256=(
                  file_hash(args.root / 'replay_schedule.json') if shared is not None else None))
    config_path = directory / 'args.json'
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError('R64 arm configuration changed')
    atomic_json(config_path, config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4,
                                  betas=(.9, .999), eps=1e-8)
    scaler = torch.amp.GradScaler('cuda', enabled=args.device.startswith('cuda'))
    controller, projection_rng = CostController(), random.Random(6402)
    history, best, best_model, lr_trace = [], None, None, []
    successful = presentations = retries = growth_successes = start_epoch = next_step = 0
    counts, task_stats, totals = Counter(), {}, Counter()
    peak_lr, stop, checkpoint = 1e-4, False, None
    epoch_prior_seconds, epoch_peak_gib = 0., 0.
    prior = executed_updates(directory)
    if (directory / 'last.pt').exists():
        if not args.resume:
            raise ValueError('R64 continuation requires --resume; preserve the existing arm')
        checkpoint = torch.load(directory / 'last.pt', map_location=args.device, weights_only=False)
        if checkpoint['args'] != config:
            raise ValueError('R64 resume configuration mismatch')
        model.load_state_dict(checkpoint['model'], strict=True)
        optimizer.load_state_dict(checkpoint['optimizer'])
        scaler.load_state_dict(checkpoint['scaler'])
        controller.__dict__.update(checkpoint['controller'])
        projection_rng.setstate(checkpoint['projection_rng'])
        history, best, best_model = checkpoint['history'], checkpoint['best'], checkpoint['best_model']
        lr_trace = checkpoint['lr_trace']
        successful, presentations, retries, growth_successes = (checkpoint[k] for k in
            ('successful_updates', 'presentations', 'amp_retries', 'scale_growth_successes'))
        start_epoch, next_step = checkpoint['epoch_index'], checkpoint['next_step']
        counts, task_stats, totals = Counter(checkpoint['updates_by_problem']), checkpoint['task_stats'], Counter(checkpoint['totals'])
        peak_lr, stop = checkpoint['peak_lr'], checkpoint['stop']
        epoch_prior_seconds, epoch_peak_gib = checkpoint['epoch_seconds'], checkpoint['epoch_peak_memory_gib']
        if best is not None:
            publish_best(directory, checkpoint)
    elif prior['recorded_successful_optimizer_steps']:
        raise ValueError('R64 has executed updates but no resume checkpoint; preserve artifacts and recover last.pt')
    launch.update(resumed_from_successful_update=successful,
        previously_recorded_execution=prior,
        recorded_steps_after_checkpoint=max(0, prior['recorded_successful_optimizer_steps'] - successful))
    atomic_json(directory / 'launches' / (launch['id'] + '.json'), launch)
    wandb_run = None
    started = time.monotonic()
    epoch_started = started

    def save_state(epoch_index, step, epoch_seconds, peak_memory):
        nonlocal checkpoint
        checkpoint = dict(model=model.state_dict(), optimizer=optimizer.state_dict(), scaler=scaler.state_dict(),
            args=config, epoch_index=epoch_index, next_step=step, successful_updates=successful,
            presentations=presentations, amp_retries=retries, scale_growth_successes=growth_successes,
            updates_by_problem=dict(counts), controller=dict(vars(controller)), projection_rng=projection_rng.getstate(),
            rng=capture_rng(), history=history, best=best, best_model=best_model, peak_lr=peak_lr,
            stop=stop, lr_trace=lr_trace, task_stats=task_stats, totals=dict(totals),
            epoch_seconds=epoch_seconds, epoch_peak_memory_gib=peak_memory,
            initial_model_hash=protocol['initial_model_hash'])
        atomic_save(directory / 'last.pt', checkpoint)
        if best is not None:
            publish_best(directory, checkpoint)
        launch.update(status='running', wall_seconds=time.monotonic()-started,
                      last_committed_update=successful, heartbeat_unix=time.time())
        atomic_json(directory / 'launches' / (launch['id'] + '.json'), launch)

    try:
        if args.wandb:
            import wandb
            wandb_run = wandb.init(**protocol['wandb'], name='R64_' + directory.name,
                config=config, dir=str(directory), id='r64-' + arm.lower() + '-' + launch['id'])
        restore_rng(checkpoint['rng']) if checkpoint is not None else set_seed(2)
        for epoch in range(start_epoch, horizon):
            if stop:
                break
            model.train()
            torch.cuda.reset_peak_memory_stats(args.device)
            epoch_started = time.monotonic()
            first_step = next_step if epoch == start_epoch else 0
            if first_step == 0:
                totals, task_stats, epoch_prior_seconds, epoch_peak_gib = Counter(), {}, 0., 0.
            for step in range(first_step, updates_per_epoch):
                entries = triple_indices(schedule, epoch, step)
                batch_list = [gather_batch(loaders['train'][task], indices) for task, indices in entries]
                progress = min(1., (successful + 1) / warmup_updates)
                lr = (shared['lr_by_successful_update'][successful] if shared is not None
                      else 1e-4 * progress if successful < warmup_updates else peak_lr)
                for group in optimizer.param_groups:
                    group['lr'] = lr
                set_seed(dropout_seed(successful))
                synchronize(args.device)
                before = time.monotonic()
                values = update_triple(model, optimizer, scaler, batch_list, .35 * progress,
                                       layout, projection_rng, arm == 'B', growth_successes)
                synchronize(args.device)
                seconds = time.monotonic()-before
                growth_successes = values['scale_growth_successes']
                rows = sum(len(indices) for _, indices in entries)
                successful += 1
                presentations += rows
                retries += values['retries']
                lr_trace.append(lr)
                metrics = dict(launch_id=launch['id'], arm=arm, epoch=epoch+1, cycle=step//6+1,
                    triple=step%6+1, update_in_epoch=step+1, successful_update=successful,
                    presentations=presentations, tasks=[task for task, _ in entries],
                    task_rows=[len(indices) for _, indices in entries], rows=rows,
                    dropout_seed=dropout_seed(successful-1), lr=lr, consistency_weight=.35*progress,
                    seconds=seconds, queries_per_second=rows/seconds,
                    peak_memory_gib=torch.cuda.max_memory_allocated(args.device)/2**30, **values)
                append_jsonl(directory / 'updates.jsonl', metrics)
                for key in ('loss', 'ce', 'kl'):
                    totals[key] += values[key] * rows
                for key in ('grad_norm', 'clip_coefficient', 'projection_mean_change_norm', 'projection_relative_mean_change'):
                    totals[key] += values[key]
                for key in ('conflict_pairs', 'comparable_pairs', 'retries'):
                    totals[key] += values[key]
                totals['samples'] += rows
                totals['updates'] += 1
                for i, (task, indices) in enumerate(entries):
                    counts[task] += 1
                    item = task_stats.setdefault(task, dict(rows=0, batches=0, loss=0., ce=0., kl=0.,
                        full_grad_norm=0., shared_grad_norm=0., projection_change_norm=0.))
                    item['rows'] += len(indices)
                    item['batches'] += 1
                    for key in ('loss', 'ce', 'kl'):
                        item[key] += values['task_losses'][i][key] * len(indices)
                    item['full_grad_norm'] += values['full_task_grad_norms'][i]
                    item['shared_grad_norm'] += values['shared_task_grad_norms'][i]
                    item['projection_change_norm'] += values['projection_task_change_norms'][i]
                if successful == 1:
                    atomic_json(directory / 'first_formal_update.json', metrics)
                    print('[R64 first FORMAL update] ' + json.dumps(metrics), flush=True)
                if successful == 1 or (step+1) % 60 == 0:
                    epoch_peak_gib = max(epoch_peak_gib, metrics['peak_memory_gib'])
                    save_state(epoch, step+1, epoch_prior_seconds+time.monotonic()-epoch_started, epoch_peak_gib)
                if (step+1) % 180 == 0 or step+1 == updates_per_epoch:
                    print(f'[R64{arm}] epoch={epoch+1} update={step+1}/{updates_per_epoch} total={successful} '
                          f'loss={totals["loss"]/totals["samples"]:.5f} lr={lr:.3g} retries={retries}', flush=True)
                if wandb_run is not None and (successful == 1 or successful % 60 == 0):
                    wandb_run.log({k: metrics[k] for k in ('successful_update', 'epoch', 'loss', 'ce', 'kl', 'lr',
                        'grad_norm', 'clip_coefficient', 'conflict_pairs', 'projection_mean_change_norm',
                        'retries', 'scale', 'peak_memory_gib', 'queries_per_second')})
            if (totals['samples'] != PRESENTATIONS_PER_EPOCH or totals['updates'] != updates_per_epoch
                    or any(item['rows'] != 10000 or item['batches'] != protocol['cycles_per_epoch'] for item in task_stats.values())
                    or len(task_stats) != 18 or successful != (epoch+1)*updates_per_epoch):
                raise ValueError('R64 incomplete cycle/triple/query coverage')
            optimization = {k: totals[k]/totals['samples'] for k in ('loss', 'ce', 'kl')}
            optimization.update({k: totals[k]/totals['updates'] for k in
                ('grad_norm', 'clip_coefficient', 'projection_mean_change_norm', 'projection_relative_mean_change')})
            optimization.update(lr=lr, lr_first=lr_trace[epoch*updates_per_epoch], consistency_weight=.35*progress,
                samples=totals['samples'], updates=totals['updates'], amp_retries=int(totals['retries']),
                conflict_pairs=int(totals['conflict_pairs']), comparable_pairs=int(totals['comparable_pairs']),
                conflict_frequency=totals['conflict_pairs']/max(1, totals['comparable_pairs']),
                seconds=epoch_prior_seconds+time.monotonic()-epoch_started,
                peak_memory_gib=max(epoch_peak_gib, torch.cuda.max_memory_allocated(args.device)/2**30))
            per_task = {}
            for task, item in task_stats.items():
                per_task[task] = dict(item)
                for key in ('loss', 'ce', 'kl'):
                    per_task[task][key] /= item['rows']
                for key in ('full_grad_norm', 'shared_grad_norm', 'projection_change_norm'):
                    per_task[task][key] /= item['batches']
            validation = evaluate(model, loaders['val'], raw['val'], args.eval_batch_size,
                                  directory / f'val_predictions/epoch{epoch+1:03d}')
            if validation['native_total'] != 18000 or any(r['n'] != 1000 for r in validation['per_problem'].values()):
                raise ValueError('Validation count selection requires all18 original1000-row tasks')
            halve, stop = controller.observe(-100*validation['native_correct']/18000, epoch+1) if arm == 'A' else (False, False)
            training = (evaluate(model, loaders['train'], raw['train'], args.eval_batch_size,
                                 directory / f'train_predictions/epoch{epoch+1:03d}')
                        if (epoch+1) % 5 == 0 or stop or epoch+1 == horizon else None)
            row = dict(epoch=epoch+1, successful_updates=successful, presentations=presentations,
                updates_by_problem=dict(counts), optimization=optimization, task_optimization=per_task,
                train=training, val=validation, controller=dict(halve=halve, stop=stop,
                    effective_best=controller.effective_best if arm == 'A' else None))
            history.append(row)
            if prefer_checkpoint(validation, None if best is None else best['val']):
                best, best_model = row, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            if halve:
                peak_lr *= .5
            totals, task_stats = Counter(), {}
            save_state(epoch+1, 0, 0., 0.)
            atomic_json(directory / 'controller_trace.json', [r['controller'] | dict(epoch=r['epoch']) for r in history])
            append_jsonl(args.root / 'curves' / (arm + '_metrics.jsonl'), row)
            print_eval(f'R64{arm} val epoch{epoch+1}', validation['per_problem'], validation['macro'])
            if wandb_run is not None:
                wb = dict(epoch=epoch+1, successful_update=successful,
                          **{'optimization/' + k: v for k, v in optimization.items()})
                for split, value in (('train_eval', training), ('val', validation)):
                    if value is not None:
                        wb.update({split+'/'+k: v for k, v in value['macro'].items()})
                        for task, result in value['per_problem'].items():
                            wb.update({f'{split}/{task}/{k}': result[k] for k in
                                       ('top1', 'top2', 'top3', 'ce', 'mean_cost', 'actual_regret_pct')})
                wandb_run.log(wb)
        if best is None or successful != len(history)*updates_per_epoch or presentations != len(history)*PRESENTATIONS_PER_EPOCH:
            raise ValueError('R64 training must terminate at a complete epoch with a validation-selected best')
        if shared is not None and (lr_trace != shared['lr_by_successful_update']
                                  or successful != shared['successful_updates'] or len(history) != shared['epochs']):
            raise ValueError('R64 B must exactly replay A LR trace and horizon')
        model.load_state_dict(best_model, strict=True)
        selected_train = evaluate(model, loaders['train'], raw['train'], args.eval_batch_size,
                                  directory / 'train_predictions/best')
        selected_val = evaluate(model, loaders['val'], raw['val'], args.eval_batch_size,
                                directory / 'val_predictions/best', directory / 'inference_timing.json')
        best_eval = dict(best, train=selected_train, selected_validation=selected_val,
                        selected_validation_count_delta=selected_val['native_correct']-best['val']['native_correct'])
        atomic_json(directory / 'best_eval.json', best_eval)
        actual = executed_updates(directory)
        atomic_json(directory / 'result.json', dict(best=best_eval, final=history[-1],
            epochs_completed=len(history), successful_updates=successful, presentations=presentations,
            updates_by_problem=dict(counts), amp_retries=retries, actual_execution=actual,
            best_checkpoint_sha256=file_hash(directory / 'best.pt'), checkpoint_rule=protocol['checkpoint_rule'],
            inference_rule=INFERENCE_RULE, paired_horizon=True, replay_lr_trace_sha256=state_hash(lr_trace),
            selection_uses_original_epoch_validation=True, selected_timing_eval_does_not_reselect=True,
            stop_reason='A-controlled common horizon' if arm == 'B' else 'validation nativeTop1 early stop' if stop else 'maximum40 epochs',
            test_read=False, test_authorized=False))
    finally:
        if wandb_run is not None:
            wandb_run.finish()


def run(arm, args, data):
    directory = args.root / RUN_NAMES[arm]
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / 'result.json').exists():
        if not args.resume:
            raise ValueError('Preserve completed R64 arm; use --resume')
        return
    (directory / 'launches').mkdir(exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    launch = dict(id=stamp, arm=arm, status='started', started_unix=time.time(), host=socket.gethostname(),
        pid=os.getpid(), gpu_uuid=os.environ.get('CUDA_VISIBLE_DEVICES'),
        slurm_step=os.environ.get('SLURM_JOB_ID', '')+'.'+os.environ.get('SLURM_STEP_ID', ''),
        tmux_session='r64_train', formal=True)
    started, status = time.monotonic(), 'failed'
    atomic_json(directory / 'launches' / (stamp+'.json'), launch)
    with (directory / 'train.log').open('a') as stream, redirect_stdout(Tee(sys.stdout, stream)), redirect_stderr(Tee(sys.stderr, stream)):
        try:
            train(arm, args, directory, data, launch)
            status = 'complete'
        finally:
            launch.update(status=status, wall_seconds=time.monotonic()-started, finished_unix=time.time())
            atomic_json(directory / 'launches' / (stamp+'.json'), launch)
            attempts = [json.loads(p.read_text()) for p in sorted((directory / 'launches').glob('*.json'))]
            atomic_json(directory / 'wall_time.json', attempts)
            if status == 'complete':
                result = json.loads((directory / 'result.json').read_text())
                result['wall_seconds'] = sum(row.get('wall_seconds', 0.) for row in attempts)
                atomic_json(directory / 'result.json', result)


def publish_validation_candidates(root):
    records = {}
    for arm, name in RUN_NAMES.items():
        directory = root / name
        result = json.loads((directory / 'result.json').read_text())
        records[arm] = dict(path=str(directory/'best.pt'), sha256=result['best_checkpoint_sha256'],
                           epoch=result['best']['epoch'], native_correct=result['best']['val']['native_correct'],
                           native_total=18000, val=result['best']['val']['macro'])
    official = 'A' if records['A']['native_correct'] >= records['B']['native_correct'] else 'B'
    atomic_json(root / 'validation_candidates.json', dict(models=records, proposed_official_arm=official,
        inference_rule=INFERENCE_RULE, parent_freeze_pending=True, test_authorized=False))


def final_test(args):
    from .performance_targets import read_raw_costs
    from .train import make_loader

    protocol = json.loads((args.root/'protocol.json').read_text())
    lock = json.loads((args.root/'final_test_lock.json').read_text())
    candidates = json.loads((args.root/'validation_candidates.json').read_text())
    if (lock.get('authorized') is not True or lock.get('inference_rule') != INFERENCE_RULE
            or lock.get('official_arm') != candidates['proposed_official_arm'] or set(lock.get('models', {})) != {'A', 'B'}):
        raise ValueError('TEST requires parent authorization, both frozen R64 bests, inference rule, and validation-selected official arm')
    if protocol['source_hashes'] != source_hashes():
        raise ValueError('R64 frozen sources changed before TEST')
    for arm, record in lock['models'].items():
        expected = candidates['models'][arm]
        if any(record[k] != expected[k] for k in ('path', 'sha256', 'epoch', 'native_correct')):
            raise ValueError('TEST cannot change a validation-selected R64 model')
        if file_hash(Path(record['path'])) != record['sha256']:
            raise ValueError('Frozen R64 best checkpoint changed')
    with (args.root/'final_test_started.json').open('x') as stream:
        json.dump(dict(lock=lock, started_unix=time.time(), one_complete_original_test_per_frozen_model=True), stream, indent=2)
    started, results = time.monotonic(), {}
    raw = {p: read_raw_costs(p, 'test') for p in PROBLEMS}
    loaders = {}
    for p, record in raw.items():
        manifest = protocol['data_manifest']['train'][p]
        if len(record['winner']) != 1000 or any(record[k] != manifest[k] for k in ('pool', 'pool_ids')):
            raise ValueError('Locked TEST must retain the original1000 rows and full legal pool')
        loader = make_loader(p, 'test', args.eval_batch_size, 0, shuffle=False, cache_device=args.device)
        loader.drop_last = False
        np.testing.assert_array_equal(loader.batch['ind'].cpu(), record['winner'])
        np.testing.assert_array_equal(loader.batch['pool_ids'].cpu(), record['pool_ids'])
        np.testing.assert_array_equal(loader.batch['costs'].cpu(), record['costs'].astype(np.float32))
        loaders[p] = loader
    for arm, record in lock['models'].items():
        model, _ = initialize(protocol['geometry_stats'], args.device)
        checkpoint = torch.load(record['path'], map_location='cpu', weights_only=False)
        model.load_state_dict(checkpoint['model'], strict=True)
        if checkpoint['epoch']+1 != record['epoch']:
            raise ValueError('Frozen TEST checkpoint epoch mismatch')
        results[arm] = evaluate(model, loaders, raw, args.eval_batch_size,
                                args.root/RUN_NAMES[arm]/'test_predictions')
        del model, checkpoint
    atomic_json(args.root/'test_results.json', dict(models=results, lock=lock, complete=True,
        wall_seconds=time.monotonic()-started, references_reuse_existing_official_tests=True))


def capacity_arm(args, inputs, entries, batch_size, arm):
    loaders, _, geometry = inputs
    model = layout = optimizer = scaler = batches = None
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(args.device)
    free, total = torch.cuda.mem_get_info(args.device)
    external_bytes = max(0, total-free-torch.cuda.memory_reserved(args.device))
    sampled_peak = [total-free]
    stop_sampling = threading.Event()

    def sample_physical():
        while not stop_sampling.is_set():
            available, physical_total = torch.cuda.mem_get_info(args.device)
            sampled_peak[0] = max(sampled_peak[0], physical_total-available)
            stop_sampling.wait(.02)

    sampler = threading.Thread(target=sample_physical, daemon=True)
    receipt = dict(arm=arm, batch_size=batch_size, effective_batch=3*batch_size,
                   tasks=[task for task, _ in entries], updates=[], genuine_cuda_oom=False,
                   warmup_updates=2, timed_updates=3, validation_used=False, test_used=False,
                   accumulated_or_microbatched=False, optimizer_state_initialized_before_timing=False)
    started, phase = time.monotonic(), 'initialization'
    sampler.start()
    try:
        model, _ = initialize(geometry, args.device)
        layout = GradientLayout(model)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4,
                                      betas=(.9, .999), eps=1e-8)
        scaler = torch.amp.GradScaler('cuda', enabled=True)
        projection_rng, growth = random.Random(6402), 0
        batches = [gather_batch(loaders['train'][task], indices) for task, indices in entries]
        model.train()
        for step in range(5):
            phase = 'warmup' if step < 2 else 'timed'
            if step == 2:
                receipt['optimizer_state_initialized_before_timing'] = bool(optimizer.state)
                if not receipt['optimizer_state_initialized_before_timing']:
                    raise RuntimeError('Capacity timing requires initialized AdamW states')
            set_seed(dropout_seed(step))
            synchronize(args.device)
            before = time.monotonic()
            values = update_triple(model, optimizer, scaler, batches, .35, layout,
                                   projection_rng, arm == 'B', growth)
            synchronize(args.device)
            seconds = time.monotonic()-before
            growth = values['scale_growth_successes']
            row = dict(arm=arm, batch_size=batch_size, update=step+1, phase=phase,
                       rows=3*batch_size, seconds=seconds, queries_per_second=3*batch_size/seconds,
                       finite_gradient=values['finite_gradient'], retries=values['retries'],
                       grad_norm=values['grad_norm'], loss=values['loss'], formal=False)
            receipt['updates'].append(row)
            append_jsonl(args.root/'capacity_updates.jsonl', row)
        receipt['complete'] = True
    except torch.cuda.OutOfMemoryError as error:
        receipt.update(complete=False, genuine_cuda_oom=True, oom_phase=phase,
                       error=str(error), failed_attempt_rows=3*batch_size)
        print(f'[R64 capacity CUDA OOM] batch={batch_size} arm={arm} phase={phase}', flush=True)
    finally:
        synchronize(args.device)
        stop_sampling.set()
        sampler.join()
        free_after, physical_total = torch.cuda.mem_get_info(args.device)
        sampled_peak[0] = max(sampled_peak[0], physical_total-free_after)
        allocated_peak = torch.cuda.max_memory_allocated(args.device)
        reserved_peak = torch.cuda.max_memory_reserved(args.device)
        # The reservation bound covers allocator peaks between physical-memory samples.
        physical_peak = max(sampled_peak[0], external_bytes+reserved_peak)
        receipt.update(wall_seconds=time.monotonic()-started, total_physical_bytes=total,
            peak_allocated_bytes=allocated_peak, peak_reserved_bytes=reserved_peak,
            sampled_physical_peak_bytes=sampled_peak[0], physical_peak_bound_bytes=physical_peak,
            external_and_context_baseline_bytes=external_bytes,
            physical_headroom_fraction=(total-physical_peak)/total,
            physical_measurement='assigned CUDA device mem_get_info sampled20ms; conservative external baseline + peak reserved bound',
            calibration_successful_updates=len(receipt['updates']),
            discarded_presentations=sum(row['rows'] for row in receipt['updates']))
        model = layout = optimizer = scaler = batches = None
        gc.collect()
        torch.cuda.empty_cache()
    timed = [row for row in receipt['updates'] if row['phase'] == 'timed']
    receipt['timed_queries_per_second'] = (sum(row['rows'] for row in timed)/sum(row['seconds'] for row in timed)
                                           if len(timed) == 3 else None)
    receipt['eligible'] = bool(receipt['complete'] and receipt['physical_headroom_fraction'] >= REQUIRED_HEADROOM)
    return receipt


def bounded_batch_sizing(args, inputs):
    with (args.root/'capacity_started.json').open('x') as stream:
        json.dump(dict(started_unix=time.time(), at_most_three_distinct_sizes=True,
                       authorized_train_only=True, no_formal_updates=True), stream, indent=2)
    policy = dict(start=256, maximum_candidates=3, feasible256_branch=[512, '1024 if512 fits else384'],
        infeasible256_branch=[128, '192 only if128 fits'], warmup_updates_per_arm=2,
        timed_updates_per_arm=3, tie_fraction=THROUGHPUT_TIE_FRACTION,
        minimum_physical_headroom_fraction=REQUIRED_HEADROOM,
        criterion='highest eligible PCGrad presentations/second; within3% of fastest prefer smaller batch',
        timing_scope='complete real update_triple including AMP retries, clipping, AdamW; both dropout forwards; no accumulation',
        representative_triple='three distinct TRAIN tasks with largest maximum padded node counts; largest rows within each task',
        resident_inputs='normal original TRAIN and validation caches/geometry; validation never forwarded or used to choose batch',
        formal_initialization='discard every calibration model/optimizer/scaler/projection RNG; both formal arms fresh seed2')
    atomic_json(args.root/'capacity_protocol.json', policy)
    loaders = inputs[0]['train']
    tasks = sorted(PROBLEMS, key=lambda p: -int(loaders[p].lengths.max()))[:3]
    started, candidates, next_size = time.monotonic(), [], 256
    for index in range(3):
        batch_size = next_size
        entries = [(task, loaders[task].lengths.argsort(descending=True, stable=True)[:batch_size]) for task in tasks]
        row = dict(batch_size=batch_size, tasks=tasks,
            representative_rows={task: dict(indices=indices.tolist(),
                minimum_nodes=int(loaders[task].lengths[indices].min()),
                padded_nodes=int(loaders[task].lengths[indices].max())) for task, indices in entries}, arms={})
        for arm in RUN_NAMES:
            receipt = capacity_arm(args, inputs, entries, batch_size, arm)
            row['arms'][arm] = receipt
            atomic_json(args.root/f'capacity_batch{batch_size}_{arm}.json', receipt)
            print('[R64 capacity] ' + json.dumps(dict(batch=batch_size, arm=arm,
                eligible=receipt['eligible'], genuine_cuda_oom=receipt['genuine_cuda_oom'],
                queries_per_second=receipt['timed_queries_per_second'],
                allocated_gib=receipt['peak_allocated_bytes']/2**30,
                reserved_gib=receipt['peak_reserved_bytes']/2**30,
                physical_peak_bound_gib=receipt['physical_peak_bound_bytes']/2**30,
                physical_headroom_fraction=receipt['physical_headroom_fraction'])), flush=True)
        row['eligible'] = all(r['eligible'] for r in row['arms'].values())
        candidates.append(row)
        atomic_json(args.root/'capacity_measurement.json', dict(policy=policy, candidates=candidates))
        if index == 0:
            next_size = 512 if row['eligible'] else 128
        elif index == 1:
            if candidates[0]['eligible']:
                next_size = 1024 if row['eligible'] else 384
            elif row['eligible']:
                next_size = 192
            else:
                break
    eligible = [row for row in candidates if row['eligible']]
    costs = dict(wall_seconds=time.monotonic()-started, distinct_sizes=len(candidates),
        actual_successful_discarded_updates=sum(len(r['updates']) for c in candidates for r in c['arms'].values()),
        actual_discarded_query_presentations=sum(r['discarded_presentations'] for c in candidates for r in c['arms'].values()),
        actual_amp_retries=sum(u['retries'] for c in candidates for r in c['arms'].values() for u in r['updates']),
        genuine_cuda_oom_attempts=sum(r['genuine_cuda_oom'] for c in candidates for r in c['arms'].values()),
        formal_updates=0, failed_attempt_work_in_individual_receipts=True)
    atomic_json(args.root/'calibration_cost.json', costs)
    if not eligible:
        raise RuntimeError('R64 capacity blocker: neither256 nor128 meets full real-batch updates and15% physical headroom; see capacity_measurement.json')
    fastest = max(row['arms']['B']['timed_queries_per_second'] for row in eligible)
    selected = min((row for row in eligible if row['arms']['B']['timed_queries_per_second'] >= fastest*(1-THROUGHPUT_TIE_FRACTION)),
                   key=lambda row: row['batch_size'])
    decision = dict(**batch_protocol(selected['batch_size']), locked=True, policy=policy,
        selected_pcgrad_queries_per_second=selected['arms']['B']['timed_queries_per_second'],
        largest_eligible_pcgrad_queries_per_second=fastest, measured_candidates=[row['batch_size'] for row in candidates],
        eligible_candidates=[row['batch_size'] for row in eligible], calibration_cost=costs,
        chosen_using_train_only=True, validation_used=False, test_used=False,
        original_lr=1e-4, scaled_lr=False, original_initialization_seed=2,
        aborted_batch42_archive=str(ABORTED_ROOT), old_trained_weights_reused=False)
    atomic_json(args.root/'large_batch_decision.json', decision)
    shutil.copy2(BATCH_AMENDMENT, args.root/'adopted_large_batch_plan.response.md')
    atomic_json(args.root/'large_batch_amendment.json', dict(status='adopted_and_locked',
        adopted_response=str(BATCH_AMENDMENT), adopted_response_sha256=file_hash(BATCH_AMENDMENT),
        planner='gpt-6-astra/max', planner_boundary='text/context planning; no direct local-code audit',
        executor='gpt-6.1-sol/max', explicit_user_restart_both=True, aborted_archive=str(ABORTED_ROOT),
        selected_protocol=batch_protocol(selected['batch_size']), learning_rate_scaled=False,
        maximum_presentations_per_arm=7200000, optimizer_update_budget_reduced=True,
        calibration_state_discarded=True, test_authorized=False))
    set_seed(2)
    print('[R64 locked larger batch] ' + json.dumps(decision), flush=True)
    return selected['batch_size']


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('plan', 'prepare', 'train', 'restart-large-batch', 'test', 'analyze'), default='plan')
    parser.add_argument('--arm', choices=tuple(RUN_NAMES))
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--batch-size', type=int, choices=(128, 192, 256, 384, 512, 1024))
    parser.add_argument('--epochs', type=int, choices=(40,), default=40)
    parser.add_argument('--seed', type=int, choices=(2,), default=2)
    parser.add_argument('--eval-batch-size', type=int, default=128)
    parser.add_argument('--device', choices=('cuda:0', 'cpu'), default='cuda:0')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--wandb', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--reference-tests', type=Path)
    args = parser.parse_args(argv)
    args.root, args.reference = args.root.resolve(), REFERENCE
    if args.eval_batch_size < 1:
        parser.error('A positive evaluation batch size is required')
    if args.stage in ('prepare', 'train', 'restart-large-batch', 'test'):
        baton = 'locked-test' if args.stage == 'test' else 'pcgrad-full18'
        if (args.device != 'cuda:0' or os.environ.get('R64_PARENT_GPU_BATON') != baton
                or os.environ.get('CUDA_VISIBLE_DEVICES') != GPU_UUID
                or os.environ.get('SLURM_JOB_ID') != '1465' or socket.gethostname().split('.')[0] != 'gpu03'):
            parser.error('R64 requires its parent baton and the assigned gpu03/Slurm1465/UUID launcher')
        if args.stage == 'test' and args.arm is not None:
            parser.error('Both frozen R64 models receive one full TEST together')
        if args.stage == 'restart-large-batch' and (args.arm is not None or args.batch_size is not None):
            parser.error('The bounded larger-batch restart chooses one shared batch and restarts both arms')
        configure_torch()
    torch.set_num_threads(1)
    args.root.mkdir(parents=True, exist_ok=True)
    (args.root/'curves').mkdir(exist_ok=True)
    if args.stage == 'restart-large-batch':
        if (args.root/'protocol.json').exists():
            parser.error('The explicit larger-batch restart requires the fresh canonical root; preserve existing protocol')
        before = time.monotonic()
        args.batch_size = 256
        inputs = load_inputs(args)
        preparation_seconds = time.monotonic()-before
        args.batch_size = bounded_batch_sizing(args, inputs)
        atomic_json(args.root/'restart_preparation.json', dict(seconds=preparation_seconds,
            normal_original_train_val_input_cache_only=True, old_trained_states_loaded=False))
        data = prepare(args, inputs)
        for arm in RUN_NAMES:
            run(arm, args, data)
        shared_replay_schedule(args.root)
        publish_validation_candidates(args.root)
        return
    decision_path = args.root/'large_batch_decision.json'
    if decision_path.exists():
        selected = json.loads(decision_path.read_text())['batch_size']
        if args.batch_size is not None and args.batch_size != selected:
            parser.error('R64 uses the immutable measured larger-batch decision')
        args.batch_size = selected
    elif args.stage in ('plan', 'prepare', 'train', 'test'):
        parser.error('Run the authorized restart-large-batch stage first to lock the measured batch')
    if args.stage == 'plan':
        atomic_json(args.root/'planned_configs.json', planned_config(args))
    elif args.stage in ('prepare', 'train'):
        data = (load_training_data(args) if (args.root/'protocol.json').exists() else prepare(args))
        if args.stage == 'train':
            for arm in (args.arm,) if args.arm else RUN_NAMES:
                run(arm, args, data)
            if all((args.root/name/'result.json').exists() for name in RUN_NAMES.values()):
                shared_replay_schedule(args.root)
                publish_validation_candidates(args.root)
    elif args.stage == 'test':
        final_test(args)
    elif args.stage == 'analyze':
        from .r64_analysis import summarize
        summarize(args.root, args.reference_tests)


if __name__ == '__main__':
    main()
