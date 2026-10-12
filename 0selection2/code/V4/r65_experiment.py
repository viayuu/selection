"""R65: fixed paired constant-LR AdamW / official Schedule-Free AdamW, TRAIN+VAL only."""

import argparse
import gc
import json
import os
import shutil
import socket
import sys
import time
from collections import Counter
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from code.unified_selector.registry import GLOBAL_SOLVERS, PROBLEMS
from .multitask_probe import Tee, file_hash, gather_batch, state_hash
from .performance_experiment import SOURCE_FILES
from .r42_experiment import prepare_data
from .r59_experiment import GPU_UUID, REFERENCE, WORKSPACE, atomic_json, atomic_save, check_original_scope
from .r64_experiment import (batch_protocol, dropout_seed, evaluate, executed_updates,
                             paired_triples, prefer_checkpoint, synchronize, triple_indices)
from .r64_model import INFERENCE_RULE, initialize
from .r65_gradients import GradientLayout, update_triple
from .r65_optimizer import (ARM_CONFIGS, BASE_LR, WARMUP_UPDATES, configure_update_lr,
                            evaluation_mode, make_optimizer, observed_lr, training_mode)
from .train import configure_torch, print_eval, set_seed
from .training_monitor import capture_rng, restore_rng


ROOT = WORKSPACE/'code/V4/runs/R65_schedule_free_full18'
ADOPTED_PLAN = WORKSPACE/'code/V4/runs/R64_pcgrad_full18/reviews/R64_closure_R65_plan.response.md'
RUN_NAMES = {'A': 'A_constant_adamw_seed2', 'B': 'B_schedule_free_seed2'}
PRESENTATIONS_PER_EPOCH = 180000
NEW_SOURCES = ('r65_experiment.py', 'r65_optimizer.py', 'r65_gradients.py', 'r65_cpu_check.py',
               'run_v4_r65.sh', 'R65_README.md', 'r64_experiment.py', 'r64_gradients.py',
               'r64_model.py', 'r45_experiment.py', 'r59_experiment.py', 'r43_experiment.py',
               'r42_experiment.py', 'r40_experiment.py', 'solver_code_encoder.py',
               'pairwise_selector.py', 'pairwise_objective.py',
               'r65_official_schedulefree/__init__.py', 'r65_official_schedulefree/adamw_schedulefree.py',
               'r65_official_schedulefree/LICENSE', 'r65_official_schedulefree/provenance.json')


def append_jsonl(path, value):
    with path.open('a') as stream:
        stream.write(json.dumps(value, allow_nan=False)+'\n')
        stream.flush()


def planned_config():
    return dict(seed=2, arms=RUN_NAMES, arm_optimizers=ARM_CONFIGS,
        problems=list(PROBLEMS), solver_names=list(GLOBAL_SOLVERS),
        max_epochs=40, **batch_protocol(256), presentations_per_epoch=PRESENTATIONS_PER_EPOCH,
        maximum_presentations_per_arm=7200000, forward_rows_per_arm=14400000,
        eval_batch_size=128, learning_rate=BASE_LR, warmup_successful_updates=WARMUP_UPDATES,
        learning_rate_rule='3e-4 * min(successful_update/720,1); then constant; B internal warmup only',
        scheduler=None, controller=None, early_stopping=False, accumulation=False, microbatching=False,
        objective='exact r64_model.classification_objective: two-dropout native CE + legal-pool symmetric KL',
        rdrop_weight=.35, rdrop_ramp_updates=720, clip_norm=1.,
        gradients='three unchanged-state task gradients; FP32 arithmetic mean; global clip1; one step',
        pcgrad=False, per_task_normalization=False, fp32_parameters=True, autocast='FP16',
        retries='whole selected triple; restore Python/NumPy/CPU/CUDA RNG; reduce scale; no cursor/ramp/update advancement',
        dropout_stream='exact R64 dropout_seed(successful); shared seed2 init and task/row schedule',
        schedule_generation='exact R64 paired_triples/triple_indices at256; fresh generated schedule, no trained state',
        model='exact r64_model.NativeUtilitySelector, handcrafted34+ID',
        schedule_free_modes='optimizer.train() -> y gradients; optimizer.eval() -> averaged x evaluation and checkpoint',
        last_checkpoint_mode='eval; B model x and optimizer train_mode=False; train() once on continuation',
        checkpoint_every_successful_updates=60, full_validation='every epoch; FP32 batch128 all18000',
        train_evaluation='full TRAIN every5/end/selected; same deployment weights; FP32 batch128',
        checkpoint_rule='strict max integer native_correct/18000; exact ties earliest epoch; never cost',
        official_arm_rule='validation-selected native integer count; exact tie A; parent freeze pending',
        inference_rule=INFERENCE_RULE, full_coverage=True, drop_last=False, original_data_only=True,
        geometry_fitted_on='original TRAIN only; reuse original reference cache',
        selected_evaluation='fixed batch128, no singleton timing variation and no reselection',
        val_prediction_saves='every epoch and selected', train_prediction_saves='every5/end/selected',
        wandb=dict(project='selector', entity='yjkds-southern-university-of-science-technology', mode='offline'),
        adopted_plan=str(ADOPTED_PLAN.relative_to(WORKSPACE)), adopted_plan_sha256=file_hash(ADOPTED_PLAN),
        optimizer_provenance=json.loads((Path(__file__).parent/'r65_official_schedulefree/provenance.json').read_text()),
        review_independence='same-family', acceptance_status='provisional', plan_score=8,
        plan_score_scope='plan only; performance unverified', target_native_macro_top1=.70,
        goal_status='ACTIVE; unachieved', difficulty='nightmare', max_rounds=100,
        legacy_B_semantics='feasible-domain mismatch unresolved and disclosed; R58 completely deferred',
        test_read=False, test_authorized=False, automatic_test=False, automatic_analysis=False,
        reference_inference_reruns=False, gpu_preflight=False, batch_sizing=False)


def source_hashes():
    paths = [Path(__file__).parent/name for name in sorted(set(SOURCE_FILES+NEW_SOURCES))]
    paths += [WORKSPACE/'code/unified_selector'/name for name in ('registry.py', 'data.py')]
    return {str(path.relative_to(WORKSPACE)): file_hash(path) for path in paths}


def select_triple_batches(loaders, schedule, epoch, step):
    entries = triple_indices(schedule, epoch, step)
    batches = [gather_batch(loaders[task], indices) for task, indices in entries]
    if any(len(batch['ind']) != len(indices) for batch, (_, indices) in zip(batches, entries)):
        raise ValueError('The actual forward must use every selected row')
    return entries, batches


def load_inputs(args):
    loaders, raw, geometry = prepare_data(args)
    check_original_scope(loaders, raw)
    if len(GLOBAL_SOLVERS) != 22 or sum(len(r['pool_ids']) for r in raw['train'].values()) != 128:
        raise ValueError('R65 retains all22 identities and all128 original legal deployments')
    return loaders, raw, geometry


def arm_config(protocol, arm):
    # JSON normalization also makes checkpoint and on-disk configuration comparisons exact.
    return json.loads(json.dumps(dict(protocol, group=arm, optimizer=ARM_CONFIGS[arm])))


def prepare(args):
    protocol = planned_config()
    protocol['source_hashes'] = source_hashes()
    for relative, digest in protocol['source_hashes'].items():
        destination = args.root/'source_launch'/relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists() and file_hash(destination) != digest:
            raise ValueError('Preserve the existing R65 launch snapshot')
        if not destination.exists():
            shutil.copy2(WORKSPACE/relative, destination)
    atomic_json(args.root/'source_launch/hashes.json', protocol['source_hashes'])
    shutil.copy2(ADOPTED_PLAN, args.root/'adopted_R65_plan.response.md')
    loaders, raw, geometry = load_inputs(args)
    sizes = {task: loader.size for task, loader in loaders['train'].items()}
    schedule = paired_triples(sizes, epochs=40, seed=2, batch_size=256)
    atomic_save(args.root/'schedule_seed2.pt', schedule)
    model, params = initialize(geometry, args.device, seed=2)
    initial = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    atomic_save(args.root/'selector_initial.pt', initial)
    protocol.update(initial_model_hash=state_hash(initial), model_params=params, sizes=sizes,
        trainable_parameters=sum(parameter.numel() for parameter in model.parameters()),
        gradient_layout=GradientLayout(model).receipt, geometry_stats=geometry,
        schedule_hash=state_hash(schedule), data_manifest=json.loads((args.root/'data_manifest.json').read_text()))
    protocol = json.loads(json.dumps(protocol))
    atomic_json(args.root/'protocol.json', protocol)
    atomic_json(args.root/'initialization.json', dict(seed=2, fresh=True,
        shared_state_sha256=protocol['initial_model_hash'], no_historical_weights=True,
        optimizer_and_cursor_reused=False))
    for arm, name in RUN_NAMES.items():
        directory = args.root/name
        directory.mkdir(exist_ok=True)
        atomic_json(directory/'args.json', arm_config(protocol, arm))
    atomic_json(args.root/'planned_configs.json', dict(protocol=protocol, arm_optimizers=ARM_CONFIGS))
    del model
    print('[R65 prepared] '+json.dumps(dict(batch_size=256, effective_batch=768, tail_effective=48,
        updates_per_epoch=240, epochs=40, updates_per_arm=9600, presentations_per_arm=7200000,
        fresh=True, next_update='FORMAL', test_authorized=False)), flush=True)
    return protocol, loaders, raw, geometry, schedule


def load_training_data(args):
    protocol = json.loads((args.root/'protocol.json').read_text())
    if protocol['source_hashes'] != source_hashes() or protocol['adopted_plan_sha256'] != file_hash(ADOPTED_PLAN):
        raise ValueError('Frozen R65 source/configuration changed')
    loaders, raw, geometry = load_inputs(args)
    schedule = torch.load(args.root/'schedule_seed2.pt', map_location='cpu', weights_only=False)
    if geometry != protocol['geometry_stats'] or state_hash(schedule) != protocol['schedule_hash']:
        raise ValueError('R65 geometry or paired schedule changed')
    if json.loads((args.root/'data_manifest.json').read_text()) != protocol['data_manifest']:
        raise ValueError('R65 original data changed')
    return protocol, loaders, raw, geometry, schedule


def publish_best(directory, checkpoint):
    atomic_json(directory/'history.json', checkpoint['history'])
    if checkpoint['best'] is None:
        return
    best = checkpoint['best']
    atomic_save(directory/'best.pt', dict(model=checkpoint['best_model'], args=checkpoint['args'],
        epoch=best['epoch']-1, macro=best['val']['macro'], successful_updates=best['successful_updates'],
        native_correct=best['val']['native_correct'], native_total=18000,
        parameter_mode=checkpoint['parameter_mode']))
    atomic_json(directory/'best_eval.json', best)


def train(arm, args, directory, data, launch):
    protocol, loaders, raw, geometry, schedule = data
    model, params = initialize(geometry, args.device, seed=2)
    if params != protocol['model_params'] or state_hash(model.state_dict()) != protocol['initial_model_hash']:
        raise ValueError('R65 arms must start from the same fresh seed2 state')
    initial = torch.load(args.root/'selector_initial.pt', map_location='cpu', weights_only=False)
    if state_hash(initial) != protocol['initial_model_hash']:
        raise ValueError('R65 fresh initialization changed')
    model.load_state_dict(initial, strict=True)
    del initial
    atomic_json(directory/'initialization.json', dict(seed=2, fresh_state_hash=protocol['initial_model_hash'],
        shared_initial_model_hash=protocol['initial_model_hash'], matches=True, no_trained_state_loaded=True))
    config = arm_config(protocol, arm)
    if json.loads((directory/'args.json').read_text()) != config:
        raise ValueError('R65 arm configuration changed')
    optimizer = make_optimizer(model, arm)
    layout = GradientLayout(model)
    scaler = torch.amp.GradScaler('cuda', enabled=True)
    history, best, best_model, lr_trace = [], None, None, []
    successful = presentations = retries = growth_successes = start_epoch = next_step = 0
    counts, task_stats, totals = Counter(), {}, Counter()
    epoch_prior_seconds, epoch_peak_gib, checkpoint = 0., 0., None
    prior = executed_updates(directory)
    if (directory/'last.pt').exists():
        if not args.resume:
            raise ValueError('R65 continuation requires --resume')
        checkpoint = torch.load(directory/'last.pt', map_location=args.device, weights_only=False)
        if checkpoint['args'] != config or checkpoint['checkpoint_mode'] != 'eval':
            raise ValueError('R65 resume configuration or parameter-mode mismatch')
        model.load_state_dict(checkpoint['model'], strict=True)
        optimizer.load_state_dict(checkpoint['optimizer'])
        if arm == 'B' and any(group['train_mode'] for group in optimizer.param_groups):
            raise ValueError('Resumable schedule-free checkpoints must contain x in eval mode')
        scaler.load_state_dict(checkpoint['scaler'])
        history, best, best_model = checkpoint['history'], checkpoint['best'], checkpoint['best_model']
        lr_trace = checkpoint['lr_trace']
        successful, presentations, retries, growth_successes = (checkpoint[key] for key in
            ('successful_updates', 'presentations', 'amp_retries', 'scale_growth_successes'))
        start_epoch, next_step = checkpoint['epoch_index'], checkpoint['next_step']
        counts, task_stats, totals = Counter(checkpoint['updates_by_problem']), checkpoint['task_stats'], Counter(checkpoint['totals'])
        epoch_prior_seconds, epoch_peak_gib = checkpoint['epoch_seconds'], checkpoint['epoch_peak_memory_gib']
        publish_best(directory, checkpoint)
    elif prior['recorded_successful_optimizer_steps']:
        raise ValueError('R65 updates exist without a resumable checkpoint; preserve artifacts')
    launch.update(resumed_from_successful_update=successful, previously_recorded_execution=prior,
        recorded_steps_after_checkpoint=max(0, prior['recorded_successful_optimizer_steps']-successful))
    atomic_json(directory/'launches'/(launch['id']+'.json'), launch)
    wandb_run, started = None, time.monotonic()

    def save_state(epoch_index, step, epoch_seconds, peak_memory):
        nonlocal checkpoint
        was_training = model.training
        evaluation_mode(model, optimizer, arm)
        checkpoint = dict(model=model.state_dict(), optimizer=optimizer.state_dict(), scaler=scaler.state_dict(),
            args=config, checkpoint_mode='eval', parameter_mode='averaged x' if arm == 'B' else 'ordinary',
            epoch_index=epoch_index, next_step=step, successful_updates=successful,
            presentations=presentations, amp_retries=retries, scale_growth_successes=growth_successes,
            updates_by_problem=dict(counts), rng=capture_rng(), history=history, best=best,
            best_model=best_model, lr_trace=lr_trace, task_stats=task_stats, totals=dict(totals),
            epoch_seconds=epoch_seconds, epoch_peak_memory_gib=peak_memory,
            initial_model_hash=protocol['initial_model_hash'])
        atomic_save(directory/'last.pt', checkpoint)
        publish_best(directory, checkpoint)
        launch.update(status='running', wall_seconds=time.monotonic()-started,
                      last_committed_update=successful, heartbeat_unix=time.time())
        atomic_json(directory/'launches'/(launch['id']+'.json'), launch)
        if was_training:
            training_mode(model, optimizer, arm)

    try:
        if args.wandb:
            import wandb
            wandb_run = wandb.init(**protocol['wandb'], name='R65_'+directory.name, config=config,
                                   dir=str(directory), id='r65-'+arm.lower()+'-'+launch['id'])
        restore_rng(checkpoint['rng']) if checkpoint is not None else set_seed(2)
        for epoch in range(start_epoch, 40):
            # Loaded optimizer state is eval; this is the single x -> y transition before continuation.
            training_mode(model, optimizer, arm)
            torch.cuda.reset_peak_memory_stats(args.device)
            epoch_started = time.monotonic()
            first_step = next_step if epoch == start_epoch else 0
            if first_step == 0:
                totals, task_stats, epoch_prior_seconds, epoch_peak_gib = Counter(), {}, 0., 0.
            for step in range(first_step, 240):
                entries, batches = select_triple_batches(loaders['train'], schedule, epoch, step)
                progress = min(1., (successful+1)/720)
                expected_lr = configure_update_lr(optimizer, arm, successful)
                set_seed(dropout_seed(successful))
                synchronize(args.device)
                before = time.monotonic()
                values = update_triple(model, optimizer, scaler, batches, .35*progress, layout, growth_successes)
                synchronize(args.device)
                seconds = time.monotonic()-before
                lr = observed_lr(optimizer, arm, expected_lr)
                growth_successes = values['scale_growth_successes']
                rows = sum(len(indices) for _, indices in entries)
                successful += 1
                presentations += rows
                retries += values['retries']
                lr_trace.append(lr)
                free_bytes, total_bytes = torch.cuda.mem_get_info(args.device)
                metrics = dict(launch_id=launch['id'], arm=arm, epoch=epoch+1, cycle=step//6+1,
                    triple=step%6+1, update_in_epoch=step+1, successful_update=successful,
                    presentations=presentations, tasks=[task for task, _ in entries],
                    task_rows=[len(indices) for _, indices in entries], rows=rows,
                    batch_indices_sha256=[state_hash(indices) for _, indices in entries],
                    actual_forward_rows=[len(batch['ind']) for batch in batches],
                    dropout_seed=dropout_seed(successful-1), lr=lr,
                    optimizer_base_lr=optimizer.param_groups[0]['lr'], consistency_weight=.35*progress,
                    gradient_parameter_mode='y' if arm == 'B' else 'ordinary',
                    seconds=seconds, queries_per_second=rows/seconds,
                    peak_memory_gib=torch.cuda.max_memory_allocated(args.device)/2**30,
                    peak_reserved_gib=torch.cuda.max_memory_reserved(args.device)/2**30,
                    physical_used_gib=(total_bytes-free_bytes)/2**30, physical_total_gib=total_bytes/2**30,
                    **values)
                append_jsonl(directory/'updates.jsonl', metrics)
                for key in ('loss', 'ce', 'kl'):
                    totals[key] += values[key]*rows
                for key in ('grad_norm', 'clip_coefficient', 'retries'):
                    totals[key] += values[key]
                totals['samples'] += rows
                totals['updates'] += 1
                for index, (task, indices) in enumerate(entries):
                    counts[task] += 1
                    item = task_stats.setdefault(task, dict(rows=0, batches=0, loss=0., ce=0., kl=0., full_grad_norm=0.))
                    item['rows'] += len(indices)
                    item['batches'] += 1
                    for key in ('loss', 'ce', 'kl'):
                        item[key] += values['task_losses'][index][key]*len(indices)
                    item['full_grad_norm'] += values['full_task_grad_norms'][index]
                if successful == 1:
                    atomic_json(directory/'first_formal_update.json', dict(metrics,
                        selected_indices={task: indices.tolist() for task, indices in entries}, launch=launch))
                    print('[R65 first FORMAL update] '+json.dumps({key: metrics[key] for key in
                        ('arm', 'successful_update', 'tasks', 'task_rows', 'rows', 'lr', 'optimizer_base_lr',
                         'loss', 'ce', 'kl', 'grad_norm', 'seconds', 'peak_memory_gib', 'physical_used_gib',
                         'retries', 'finite_gradient')}), flush=True)
                if successful == 1 or (step+1) % 60 == 0:
                    epoch_peak_gib = max(epoch_peak_gib, metrics['peak_memory_gib'])
                    save_state(epoch, step+1, epoch_prior_seconds+time.monotonic()-epoch_started, epoch_peak_gib)
                if (step+1) % 180 == 0 or step+1 == 240:
                    print(f'[R65{arm}] epoch={epoch+1} update={step+1}/240 total={successful} '
                          f'loss={totals["loss"]/totals["samples"]:.5f} lr={lr:.9g} retries={retries}', flush=True)
                if wandb_run is not None and (successful == 1 or successful % 60 == 0):
                    wandb_run.log({key: metrics[key] for key in ('successful_update', 'epoch', 'loss', 'ce',
                        'kl', 'lr', 'optimizer_base_lr', 'grad_norm', 'clip_coefficient', 'retries', 'scale',
                        'peak_memory_gib', 'queries_per_second')})
            if (totals['samples'] != PRESENTATIONS_PER_EPOCH or totals['updates'] != 240
                    or len(task_stats) != 18 or successful != (epoch+1)*240
                    or any(item['rows'] != 10000 or item['batches'] != 40 for item in task_stats.values())):
                raise ValueError('R65 must retain every original TRAIN row once per epoch')
            optimization = {key: totals[key]/totals['samples'] for key in ('loss', 'ce', 'kl')}
            optimization.update({key: totals[key]/totals['updates'] for key in ('grad_norm', 'clip_coefficient')})
            optimization.update(lr=lr, lr_first=lr_trace[epoch*240], consistency_weight=.35*progress,
                samples=totals['samples'], updates=totals['updates'], amp_retries=int(totals['retries']),
                seconds=epoch_prior_seconds+time.monotonic()-epoch_started,
                peak_memory_gib=max(epoch_peak_gib, torch.cuda.max_memory_allocated(args.device)/2**30))
            per_task = {}
            for task, item in task_stats.items():
                per_task[task] = dict(item)
                for key in ('loss', 'ce', 'kl'):
                    per_task[task][key] /= item['rows']
                per_task[task]['full_grad_norm'] /= item['batches']
            evaluation_mode(model, optimizer, arm)
            validation = evaluate(model, loaders['val'], raw['val'], 128,
                                  directory/f'val_predictions/epoch{epoch+1:03d}')
            if validation['native_total'] != 18000 or any(r['n'] != 1000 for r in validation['per_problem'].values()):
                raise ValueError('Validation must contain all18000 original queries')
            training = (evaluate(model, loaders['train'], raw['train'], 128,
                                 directory/f'train_predictions/epoch{epoch+1:03d}')
                        if (epoch+1) % 5 == 0 or epoch+1 == 40 else None)
            row = dict(epoch=epoch+1, successful_updates=successful, presentations=presentations,
                updates_by_problem=dict(counts), optimization=optimization, task_optimization=per_task,
                train=training, val=validation, parameter_mode='averaged x' if arm == 'B' else 'ordinary')
            history.append(row)
            if prefer_checkpoint(validation, None if best is None else best['val']):
                best, best_model = row, {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            totals, task_stats = Counter(), {}
            save_state(epoch+1, 0, 0., 0.)
            append_jsonl(args.root/'curves'/(arm+'_metrics.jsonl'), row)
            print_eval(f'R65{arm} val epoch{epoch+1}', validation['per_problem'], validation['macro'])
            if wandb_run is not None:
                wb = dict(epoch=epoch+1, successful_update=successful,
                          **{'optimization/'+key: value for key, value in optimization.items()})
                for split, value in (('train_eval', training), ('val', validation)):
                    if value is not None:
                        wb.update({split+'/'+key: entry for key, entry in value['macro'].items()})
                        wb[split+'/native_correct'] = value['native_correct']
                        for task, result in value['per_problem'].items():
                            wb.update({f'{split}/{task}/{key}': result[key] for key in
                                       ('top1', 'top2', 'top3', 'ce', 'mean_cost', 'actual_regret_pct')})
                wandb_run.log(wb)
        if best is None or len(history) != 40 or successful != 9600 or presentations != 7200000:
            raise ValueError('R65 requires the complete fixed40E/9600update/7.2M horizon')
        # The last model/optimizer remains saved consistently at final x. Best x is evaluation only.
        model.load_state_dict(best_model, strict=True)
        model.eval()
        selected_train = evaluate(model, loaders['train'], raw['train'], 128, directory/'train_predictions/best')
        selected_val = evaluate(model, loaders['val'], raw['val'], 128, directory/'val_predictions/best')
        best_eval = dict(best, train=selected_train, selected_validation=selected_val,
            selected_validation_count_delta=selected_val['native_correct']-best['val']['native_correct'])
        atomic_json(directory/'best_eval.json', best_eval)
        atomic_json(directory/'result.json', dict(best=best_eval, final=history[-1], epochs_completed=40,
            successful_updates=successful, presentations=presentations, updates_by_problem=dict(counts),
            amp_retries=retries, actual_execution=executed_updates(directory),
            best_checkpoint_sha256=file_hash(directory/'best.pt'), checkpoint_rule=protocol['checkpoint_rule'],
            inference_rule=INFERENCE_RULE, lr_trace_sha256=state_hash(lr_trace), fixed_matched_horizon=True,
            selection_uses_original_epoch_validation=True, selected_eval_does_not_reselect=True,
            deployment_parameter_mode='averaged x' if arm == 'B' else 'ordinary',
            stop_reason='fixed maximum40 epochs', test_read=False, test_authorized=False))
    finally:
        if wandb_run is not None:
            wandb_run.finish()


def run(arm, args, data):
    directory = args.root/RUN_NAMES[arm]
    if (directory/'result.json').exists():
        if not args.resume:
            raise ValueError('Preserve completed R65 arms; --resume skips them')
        return
    (directory/'launches').mkdir(exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    launch = dict(id=stamp, arm=arm, status='started', started_unix=time.time(), host=socket.gethostname(),
        pid=os.getpid(), gpu_uuid=os.environ.get('CUDA_VISIBLE_DEVICES'),
        slurm_step=os.environ.get('SLURM_JOB_ID', '')+'.'+os.environ.get('SLURM_STEP_ID', ''),
        tmux_session='r65_train', formal=True)
    started, status = time.monotonic(), 'failed'
    atomic_json(directory/'launches'/(stamp+'.json'), launch)
    with (directory/'train.log').open('a') as stream, redirect_stdout(Tee(sys.stdout, stream)), redirect_stderr(Tee(sys.stderr, stream)):
        try:
            train(arm, args, directory, data, launch)
            status = 'complete'
        finally:
            launch.update(status=status, wall_seconds=time.monotonic()-started, finished_unix=time.time())
            atomic_json(directory/'launches'/(stamp+'.json'), launch)
            attempts = [json.loads(path.read_text()) for path in sorted((directory/'launches').glob('*.json'))]
            atomic_json(directory/'wall_time.json', attempts)
            if status == 'complete':
                result = json.loads((directory/'result.json').read_text())
                result['wall_seconds'] = sum(row.get('wall_seconds', 0.) for row in attempts)
                atomic_json(directory/'result.json', result)


def publish_validation_candidates(root):
    records, lr_hashes = {}, []
    for arm, name in RUN_NAMES.items():
        directory = root/name
        result = json.loads((directory/'result.json').read_text())
        records[arm] = dict(path=str(directory/'best.pt'), sha256=result['best_checkpoint_sha256'],
            epoch=result['best']['epoch'], native_correct=result['best']['val']['native_correct'],
            native_total=18000, val=result['best']['val']['macro'], parameter_mode=result['deployment_parameter_mode'])
        lr_hashes.append(result['lr_trace_sha256'])
    if len(set(lr_hashes)) != 1:
        raise ValueError('R65 arms must use the same effective LR sequence')
    official = 'A' if records['A']['native_correct'] >= records['B']['native_correct'] else 'B'
    atomic_json(root/'validation_candidates.json', dict(models=records, proposed_official_arm=official,
        inference_rule=INFERENCE_RULE, parent_freeze_pending=True, test_authorized=False,
        same_effective_lr_trace_sha256=lr_hashes[0]))


def final_test(args):
    from .performance_targets import read_raw_costs
    from .train import make_loader

    protocol = json.loads((args.root/'protocol.json').read_text())
    lock = json.loads((args.root/'final_test_lock.json').read_text())
    candidates = json.loads((args.root/'validation_candidates.json').read_text())
    if (lock.get('authorized') is not True or lock.get('inference_rule') != INFERENCE_RULE
            or lock.get('official_arm') != candidates['proposed_official_arm']
            or set(lock.get('models', {})) != {'A', 'B'}):
        raise ValueError('TEST requires parent authorization of both frozen weights, rule and official arm')
    if protocol['source_hashes'] != source_hashes():
        raise ValueError('R65 frozen sources changed before authorized TEST')
    for arm, record in lock['models'].items():
        expected = candidates['models'][arm]
        if any(record[key] != expected[key] for key in ('path', 'sha256', 'epoch', 'native_correct')):
            raise ValueError('TEST cannot change a validation-selected R65 model')
        if file_hash(Path(record['path'])) != record['sha256']:
            raise ValueError('Frozen R65 checkpoint changed')
    with (args.root/'final_test_started.json').open('x') as stream:
        json.dump(dict(lock=lock, started_unix=time.time(), one_complete_original_test_per_frozen_model=True), stream, indent=2)
    started, results = time.monotonic(), {}
    raw = {task: read_raw_costs(task, 'test') for task in PROBLEMS}
    loaders = {}
    for task, record in raw.items():
        manifest = protocol['data_manifest']['train'][task]
        if len(record['winner']) != 1000 or any(record[key] != manifest[key] for key in ('pool', 'pool_ids')):
            raise ValueError('Locked TEST retains all original rows and full legal pools')
        loader = make_loader(task, 'test', 128, 0, shuffle=False, cache_device=args.device)
        loader.drop_last = False
        np.testing.assert_array_equal(loader.batch['ind'].cpu(), record['winner'])
        np.testing.assert_array_equal(loader.batch['pool_ids'].cpu(), record['pool_ids'])
        np.testing.assert_array_equal(loader.batch['costs'].cpu(), record['costs'].astype(np.float32))
        loaders[task] = loader
    for arm, record in lock['models'].items():
        model, _ = initialize(protocol['geometry_stats'], args.device, seed=2)
        checkpoint = torch.load(record['path'], map_location='cpu', weights_only=False)
        if checkpoint['epoch']+1 != record['epoch'] or checkpoint['parameter_mode'] != candidates['models'][arm]['parameter_mode']:
            raise ValueError('Frozen R65 epoch or deployment parameter mode mismatch')
        model.load_state_dict(checkpoint['model'], strict=True)
        model.eval()
        results[arm] = evaluate(model, loaders, raw, 128, args.root/RUN_NAMES[arm]/'test_predictions')
        del model, checkpoint
    atomic_json(args.root/'test_results.json', dict(models=results, lock=lock, complete=True,
        wall_seconds=time.monotonic()-started, references_reuse_existing_official_tests=True))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('cpu-check', 'train', 'test'), default='train')
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--wandb', action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args(argv)
    args.root, args.reference = args.root.resolve(), REFERENCE
    args.device, args.batch_size, args.eval_batch_size = 'cuda:0', 256, 128
    torch.set_num_threads(1)
    args.root.mkdir(parents=True, exist_ok=True)
    if args.stage == 'cpu-check':
        from .r65_cpu_check import focused_check
        result = focused_check()
        atomic_json(args.root/'cpu_check.json', result)
        print(json.dumps(result), flush=True)
        return
    baton = 'locked-test' if args.stage == 'test' else 'schedule-free-full18'
    if (os.environ.get('R65_PARENT_GPU_BATON') != baton or os.environ.get('CUDA_VISIBLE_DEVICES') != GPU_UUID
            or os.environ.get('SLURM_JOB_ID') != '1465' or socket.gethostname().split('.')[0] != 'gpu03'):
        parser.error('R65 requires its explicit baton and the assigned gpu03/Slurm1465/UUID launcher')
    configure_torch()
    if args.stage == 'test':
        final_test(args)
        return
    check = json.loads((args.root/'cpu_check.json').read_text())
    if not check.get('passed'):
        parser.error('The one focused CPU mode/resume check must pass before formal training')
    (args.root/'curves').mkdir(exist_ok=True)
    data = load_training_data(args) if (args.root/'protocol.json').exists() else prepare(args)
    for arm in RUN_NAMES:
        run(arm, args, data)
        gc.collect()
        torch.cuda.empty_cache()
    publish_validation_candidates(args.root)
    atomic_json(args.root/'paired_training_complete.json', dict(completed_utc=datetime.now(timezone.utc).isoformat(),
        arms=list(RUN_NAMES), epochs_per_arm=40, updates_per_arm=9600, presentations_per_arm=7200000,
        validation_candidates_ready=True, parent_freeze_pending=True, test_authorized=False, analysis_started=False))
    print('[R65 paired training complete] parent validation freeze / TEST authorization pending', flush=True)


if __name__ == '__main__':
    main()
