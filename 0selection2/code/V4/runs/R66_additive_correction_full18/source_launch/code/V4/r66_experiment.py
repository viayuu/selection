"""R66 fixed full18 paired additive learning, TRAIN/VAL only; no TEST entrypoint."""

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

import torch

from code.unified_selector.registry import GLOBAL_SOLVERS, PROBLEMS
from .multitask_probe import Tee, file_hash, state_hash
from .performance_experiment import SOURCE_FILES
from .r59_experiment import GPU_UUID, REFERENCE, WORKSPACE, atomic_json, atomic_save
from .r64_experiment import (batch_protocol, dropout_seed, evaluate, executed_updates,
                             paired_triples, prefer_checkpoint, synchronize)
from .r65_experiment import load_inputs, select_triple_batches
from .r65_gradients import GradientLayout, update_triple
from .r65_optimizer import (ARM_CONFIGS, BASE_LR, WARMUP_UPDATES, configure_update_lr,
                            evaluation_mode, make_optimizer, observed_lr, training_mode)
from .r66_model import (BASE_PATH, BASE_SHA256, CACHE_INPUT, INFERENCE_RULE, MEMBER_SEED,
                        build_selector, load_base_caches, load_base_checkpoint)
from .train import configure_torch, print_eval, set_seed
from .training_monitor import capture_rng, restore_rng


ROOT = WORKSPACE/'code/V4/runs/R66_additive_correction_full18'
ADOPTED_PLAN = WORKSPACE/'.aris/traces/R65_closure_R66_plan.response.md'
RUN_NAMES = {'A': 'A_independent_sum_seed2', 'B': 'B_additive_correction_seed2'}
NEW_SOURCES = ('r66_model.py', 'r66_objective.py', 'r66_experiment.py', 'r66_cpu_check.py',
    'run_v4_r66.sh', 'R66_README.md', 'r64_model.py', 'r64_experiment.py', 'r64_gradients.py',
    'r65_experiment.py', 'r65_optimizer.py', 'r65_gradients.py', 'r45_experiment.py',
    'r59_experiment.py', 'r43_experiment.py', 'r42_experiment.py', 'r40_experiment.py',
    'pairwise_selector.py', 'pairwise_objective.py', 'solver_code_encoder.py',
    'r65_official_schedulefree/__init__.py', 'r65_official_schedulefree/adamw_schedulefree.py',
    'r65_official_schedulefree/LICENSE', 'r65_official_schedulefree/provenance.json')


def append_jsonl(path, value):
    with path.open('a') as stream:
        stream.write(json.dumps(value, allow_nan=False)+'\n')
        stream.flush()


def source_hashes():
    paths = [Path(__file__).parent/name for name in sorted(set(SOURCE_FILES+NEW_SOURCES))]
    paths += [WORKSPACE/'code/unified_selector'/name for name in ('registry.py', 'data.py')]
    return {str(path.relative_to(WORKSPACE)): file_hash(path) for path in paths}


def planned_config():
    return dict(seed=2, member_seed=MEMBER_SEED, arms=RUN_NAMES, max_epochs=40,
        problems=list(PROBLEMS), solver_names=list(GLOBAL_SOLVERS), **batch_protocol(256),
        presentations_per_epoch=180000, maximum_presentations_per_arm=7200000,
        new_pair_updates=19200, new_pair_presentations=14400000, dropout_forwards_per_update=2,
        inherited_base_updates=9600, inherited_base_presentations=7200000,
        base_checkpoint=str(BASE_PATH.relative_to(WORKSPACE)), base_checkpoint_sha256=BASE_SHA256,
        base_parameter_mode='immutable saved R65B40 averaged x', base_gradients=False,
        model='actual R64/R65 NativeUtilitySelector; handcrafted34+ID; original TRAIN geometry',
        member_initialization='one fresh memberseed660002, scalar score weight+bias zero; explicitly copy and compare',
        arm_objectives={'A': 'standalone g two-forward native CE + legal-pool symmetric KL',
                        'B': 'detached F + g two-forward native CE + legal-pool symmetric KL'},
        evaluation_logits='F + g_x in BOTH arms; cached F on TRAIN/VAL only',
        optimizer=ARM_CONFIGS['B'], learning_rate=BASE_LR, warmup_successful_updates=WARMUP_UPDATES,
        learning_rate_rule='official optimizer internal720-update warmup to3e-4; constant after',
        optimizer_parameters='only g', external_warmup=False, schedule_free_modes='train y; eval/save x',
        rdrop_weight=.35, rdrop_ramp_updates=720, clip_norm=1.,
        gradients='unchanged r65_gradients: three same-state FP32 task gradients, mean, global clip1, one update',
        retries='unchanged whole-triple AMP retry; same batches; restore RNG; counters and ramp advance only on success',
        fp32_parameters=True, autocast='FP16', accumulation=False, microbatching=False,
        scheduler=None, controller=None, early_stopping=False, drop_last=False, original_data_only=True,
        dropout_stream='R64 dropout_seed(successful); shared task/row/dropout plans',
        eval_batch_size=128, full_validation='every epoch; all18000; FP32 with tails; composite',
        train_evaluation='every5/final/selected; all180000; FP32 with tails; composite',
        optimization_metrics='raw training objective loss/CE/KL; A standalone, B composite; never relabel A as composite CE',
        checkpoint_rule='max integer native_correct /18000; earliest exact tie',
        official_arm_rule='max selected integer validation count; exact tie A; parent later locks',
        checkpoint_every_successful_updates=60, checkpoint_mode='eval x; optimizer train_mode=False',
        best_bundle='full immutable F and selected g_x, coefficient1 and exact inference rule',
        inference_rule=INFERENCE_RULE, selector_forwards_per_deployment=2, selected_solver_executions=1,
        deployment_timing='future ordinary final TEST computes BOTH live selectors on current raw inputs; no cached base TEST and no separate benchmark',
        cache_boundary='extract only logits, original row indices, pool names and pool IDs; no archive supervision fields',
        wandb=dict(project='selector', entity='yjkds-southern-university-of-science-technology', mode='offline'),
        adopted_plan=str(ADOPTED_PLAN.relative_to(WORKSPACE)), adopted_plan_sha256=file_hash(ADOPTED_PLAN),
        target_native_macro_top1=.70, goal_status='ACTIVE; unachieved', difficulty='nightmare', max_rounds=100,
        legacy_B_semantics='eight B-task feasible-domain mismatches unresolved and disclosed; R58 fully deferred',
        r58_new_labels_repairs_scenarios_routes_or_solver_runs=False,
        test_read=False, test_authorized=False, automatic_test=False, reference_inference=False,
        gpu_sizing_sweep=False, training_pilot=False, review_independence='same-family',
        acceptance_status='provisional', optimizer_provenance=json.loads(
            (Path(__file__).parent/'r65_official_schedulefree/provenance.json').read_text()))


def prepare_cpu(args):
    if (args.root/'protocol.json').exists() or (args.root/'cpu_check.json').exists():
        raise ValueError('Preserve R66 preparation; do not repeat the focused CPU check')
    from .r66_cpu_check import focused_check
    check, base, initial, initialization, caches = focused_check()
    schedule = paired_triples({task: 10000 for task in PROBLEMS}, epochs=40, seed=2, batch_size=256)
    protocol = planned_config()
    protocol.update(source_hashes=source_hashes(), model_params=base['args']['model_params'],
        geometry_stats=base['args']['geometry_stats'], data_manifest=base['args']['data_manifest'],
        base_state_sha256=state_hash(base['model']), initial_model_hash=state_hash(initial),
        trainable_parameters=sum(value.numel() for key, value in initial.items()
            if key not in ('solver_encoder.solver_features', 'instance_encoder.geometry_residual.mean',
                           'instance_encoder.geometry_residual.std')),
        schedule_hash=state_hash(schedule), cache_manifest=caches)
    for relative, digest in protocol['source_hashes'].items():
        destination = args.root/'source_launch'/relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists() and file_hash(destination) != digest:
            raise ValueError('Existing R66 source snapshot changed')
        if not destination.exists():
            shutil.copy2(WORKSPACE/relative, destination)
    atomic_json(args.root/'source_launch/hashes.json', protocol['source_hashes'])
    shutil.copy2(ADOPTED_PLAN, args.root/'adopted_R66_plan.response.md')
    atomic_save(args.root/'member_initial.pt', initial)
    atomic_save(args.root/'schedule_seed2.pt', schedule)
    atomic_json(args.root/'initialization.json', initialization)
    protocol = json.loads(json.dumps(protocol))
    atomic_json(args.root/'protocol.json', protocol)
    atomic_json(args.root/'planned_configs.json', {arm: arm_config(protocol, arm) for arm in RUN_NAMES})
    for arm, name in RUN_NAMES.items():
        directory = args.root/name
        directory.mkdir(exist_ok=True)
        atomic_json(directory/'args.json', arm_config(protocol, arm))
    atomic_json(args.root/'cpu_check.json', check)
    print('[R66 focused CPU check] '+json.dumps(check), flush=True)


def arm_config(protocol, arm):
    return dict(protocol, group=arm, objective_scope='standalone g' if arm == 'A' else 'composite F+g')


def load_training_data(args):
    protocol = json.loads((args.root/'protocol.json').read_text())
    if protocol['source_hashes'] != source_hashes() or protocol['adopted_plan_sha256'] != file_hash(ADOPTED_PLAN):
        raise ValueError('Frozen R66 launch sources or plan changed')
    base = load_base_checkpoint()
    if state_hash(base['model']) != protocol['base_state_sha256']:
        raise ValueError('Frozen saved base x changed')
    loaders, raw, geometry = load_inputs(args)
    if (geometry != protocol['geometry_stats'] or json.loads(
            (args.root/'data_manifest.json').read_text()) != protocol['data_manifest']):
        raise ValueError('Original TRAIN/VAL or TRAIN geometry changed')
    caches, _ = load_base_caches(protocol['cache_manifest'], args.device)
    for split, tasks in loaders.items():
        for task, loader in tasks.items():
            loader.batch[CACHE_INPUT] = caches[split][task]
    schedule = torch.load(args.root/'schedule_seed2.pt', map_location='cpu', weights_only=False)
    initial = torch.load(args.root/'member_initial.pt', map_location='cpu', weights_only=False)
    if state_hash(schedule) != protocol['schedule_hash'] or state_hash(initial) != protocol['initial_model_hash']:
        raise ValueError('Paired schedule or shared copied initialization changed')
    return protocol, loaders, raw, schedule, base['model'], initial


def publish_best(directory, protocol, arm, best, member, base):
    bundle = dict(base=base, member=member, args=arm_config(protocol, arm), epoch=best['epoch']-1,
        macro=best['val']['macro'], native_correct=best['val']['native_correct'], native_total=18000,
        successful_updates=best['successful_updates'], parameter_mode='averaged x',
        base_checkpoint_sha256=BASE_SHA256, base_state_sha256=protocol['base_state_sha256'],
        inference_rule=INFERENCE_RULE, coefficient=1., selector_forwards=2, solver_executions=1,
        cache_allowed_at_deployment=False)
    unique = directory/'best_checkpoints'/f'epoch{best["epoch"]:03d}.pt'
    unique.parent.mkdir(exist_ok=True)
    atomic_save(unique, bundle)
    atomic_save(directory/'best.pt', bundle)
    atomic_json(directory/'best_eval.json', best)


def train(arm, args, directory, data, launch):
    protocol, loaders, raw, schedule, base, initial = data
    config = arm_config(protocol, arm)
    if json.loads((directory/'args.json').read_text()) != config:
        raise ValueError('R66 arm configuration changed')
    model = build_selector(protocol['model_params'], base, initial, arm, args.device, cached=True)
    atomic_json(directory/'initialization.json', dict(member_seed=MEMBER_SEED,
        shared_initial_model_hash=protocol['initial_model_hash'],
        copied_from=str(args.root/'member_initial.pt'), explicit_shared_copy_verified_in_cpu_check=True))
    optimizer = make_optimizer(model.member, 'B')
    layout, scaler = GradientLayout(model), torch.amp.GradScaler('cuda', enabled=True)
    history, best, best_member, lr_trace = [], None, None, []
    successful = presentations = retries = growth_successes = start_epoch = next_step = 0
    counts, task_stats, totals = Counter(), {}, Counter()
    epoch_prior_seconds, epoch_peak_gib, checkpoint = 0., 0., None
    prior = executed_updates(directory)
    if (directory/'last.pt').exists():
        if not args.resume:
            raise ValueError('R66 continuation requires explicit --resume')
        checkpoint = torch.load(directory/'last.pt', map_location=args.device, weights_only=False)
        if checkpoint['args'] != config or checkpoint['checkpoint_mode'] != 'eval':
            raise ValueError('R66 resume configuration/mode changed')
        model.load_state_dict(checkpoint['model'], strict=True)
        if state_hash(model.base.state_dict()) != protocol['base_state_sha256']:
            raise ValueError('Resume must retain immutable saved base x')
        optimizer.load_state_dict(checkpoint['optimizer'])
        if any(group['train_mode'] for group in optimizer.param_groups):
            raise ValueError('Resumable optimizer must be at eval x')
        scaler.load_state_dict(checkpoint['scaler'])
        history, best, best_member = checkpoint['history'], checkpoint['best'], checkpoint['best_member']
        lr_trace = checkpoint['lr_trace']
        successful, presentations, retries, growth_successes = (checkpoint[key] for key in
            ('successful_updates', 'presentations', 'amp_retries', 'scale_growth_successes'))
        start_epoch, next_step = checkpoint['epoch_index'], checkpoint['next_step']
        counts, task_stats, totals = Counter(checkpoint['updates_by_problem']), checkpoint['task_stats'], Counter(checkpoint['totals'])
        epoch_prior_seconds, epoch_peak_gib = checkpoint['epoch_seconds'], checkpoint['epoch_peak_memory_gib']
    elif prior['recorded_successful_optimizer_steps']:
        raise ValueError('Formal updates exist without resumable state; preserve artifacts')
    launch.update(resumed_from_successful_update=successful, previously_recorded_execution=prior)
    atomic_json(directory/'launches'/(launch['id']+'.json'), launch)
    wandb_run, started = None, time.monotonic()

    def save_state(epoch_index, step, epoch_seconds, peak_memory):
        evaluation_mode(model, optimizer, 'B')
        atomic_save(directory/'last.pt', dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
            scaler=scaler.state_dict(), args=config, checkpoint_mode='eval', parameter_mode='averaged x',
            epoch_index=epoch_index, next_step=step, successful_updates=successful,
            presentations=presentations, amp_retries=retries, scale_growth_successes=growth_successes,
            updates_by_problem=dict(counts), rng=capture_rng(), history=history, best=best,
            best_member=best_member, lr_trace=lr_trace, task_stats=task_stats, totals=dict(totals),
            epoch_seconds=epoch_seconds, epoch_peak_memory_gib=peak_memory))
        launch.update(status='running', wall_seconds=time.monotonic()-started,
                      last_committed_update=successful, heartbeat_unix=time.time())
        atomic_json(directory/'launches'/(launch['id']+'.json'), launch)
        atomic_json(directory/'history.json', history)

    try:
        if args.wandb:
            import wandb
            wandb_run = wandb.init(**protocol['wandb'], name='R66_'+directory.name,
                config=config, dir=str(directory), id='r66-'+arm.lower()+'-'+launch['id'])
        restore_rng(checkpoint['rng']) if checkpoint is not None else set_seed(2)
        for epoch in range(start_epoch, 40):
            training_mode(model, optimizer, 'B')
            torch.cuda.reset_peak_memory_stats(args.device)
            epoch_started = time.monotonic()
            first_step = next_step if epoch == start_epoch else 0
            if first_step == 0:
                totals, task_stats, epoch_prior_seconds, epoch_peak_gib = Counter(), {}, 0., 0.
            for step in range(first_step, 240):
                entries, batches = select_triple_batches(loaders['train'], schedule, epoch, step)
                progress = min(1., (successful+1)/720)
                expected_lr = configure_update_lr(optimizer, 'B', successful)
                set_seed(dropout_seed(successful))
                synchronize(args.device)
                before = time.monotonic()
                values = update_triple(model, optimizer, scaler, batches, .35*progress, layout, growth_successes)
                synchronize(args.device)
                seconds = time.monotonic()-before
                lr = observed_lr(optimizer, 'B', expected_lr)
                growth_successes = values['scale_growth_successes']
                rows = sum(len(indices) for _, indices in entries)
                successful += 1
                presentations += rows
                retries += values['retries']
                lr_trace.append(lr)
                free_bytes, total_bytes = torch.cuda.mem_get_info(args.device)
                metrics = dict(launch_id=launch['id'], arm=arm, epoch=epoch+1, update_in_epoch=step+1,
                    successful_update=successful, presentations=presentations,
                    tasks=[task for task, _ in entries], task_rows=[len(indices) for _, indices in entries],
                    rows=rows, batch_indices_sha256=[state_hash(indices) for _, indices in entries],
                    actual_forward_rows=[len(batch['ind']) for batch in batches],
                    dropout_seed=dropout_seed(successful-1), lr=lr, optimizer_base_lr=optimizer.param_groups[0]['lr'],
                    consistency_weight=.35*progress, gradient_parameter_mode='g y; F saved x frozen',
                    objective_scope=config['objective_scope'], seconds=seconds, queries_per_second=rows/seconds,
                    peak_memory_gib=torch.cuda.max_memory_allocated(args.device)/2**30,
                    peak_reserved_gib=torch.cuda.max_memory_reserved(args.device)/2**30,
                    physical_used_gib=(total_bytes-free_bytes)/2**30, physical_total_gib=total_bytes/2**30, **values)
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
                    print('[R66 first FORMAL update] '+json.dumps({key: metrics[key] for key in
                        ('arm', 'successful_update', 'task_rows', 'rows', 'lr', 'loss', 'ce', 'kl',
                         'grad_norm', 'seconds', 'peak_memory_gib', 'retries', 'finite_loss', 'finite_gradient')}), flush=True)
                if successful == 1 or (step+1) % 60 == 0:
                    epoch_peak_gib = max(epoch_peak_gib, metrics['peak_memory_gib'])
                    save_state(epoch, step+1, epoch_prior_seconds+time.monotonic()-epoch_started, epoch_peak_gib)
                    training_mode(model, optimizer, 'B')
                if (step+1) % 60 == 0:
                    print(f'[R66{arm}] epoch={epoch+1} update={step+1}/240 total={successful} '
                          f'objective={config["objective_scope"]} loss={totals["loss"]/totals["samples"]:.5f} retries={retries}', flush=True)
                if wandb_run is not None and (successful == 1 or successful % 60 == 0):
                    wandb_run.log({key: metrics[key] for key in ('successful_update', 'epoch', 'loss', 'ce',
                        'kl', 'lr', 'grad_norm', 'clip_coefficient', 'retries', 'scale', 'peak_memory_gib')})
            if (totals['samples'] != 180000 or totals['updates'] != 240 or len(task_stats) != 18
                    or successful != (epoch+1)*240
                    or any(item['rows'] != 10000 or item['batches'] != 40 for item in task_stats.values())):
                raise ValueError('Every original TRAIN row must be presented exactly once per epoch')
            optimization = {key: totals[key]/totals['samples'] for key in ('loss', 'ce', 'kl')}
            optimization.update({key: totals[key]/totals['updates'] for key in ('grad_norm', 'clip_coefficient')})
            optimization.update(lr=lr, lr_first=lr_trace[epoch*240], consistency_weight=.35*progress,
                objective_scope=config['objective_scope'], samples=totals['samples'], updates=totals['updates'],
                amp_retries=int(totals['retries']), seconds=epoch_prior_seconds+time.monotonic()-epoch_started,
                peak_memory_gib=max(epoch_peak_gib, torch.cuda.max_memory_allocated(args.device)/2**30))
            per_task = {}
            for task, item in task_stats.items():
                per_task[task] = dict(item)
                for key in ('loss', 'ce', 'kl'):
                    per_task[task][key] /= item['rows']
                per_task[task]['full_grad_norm'] /= item['batches']
            evaluation_mode(model, optimizer, 'B')
            validation = evaluate(model, loaders['val'], raw['val'], 128,
                                  directory/f'val_predictions/epoch{epoch+1:03d}')
            if validation['native_total'] != 18000 or any(r['n'] != 1000 for r in validation['per_problem'].values()):
                raise ValueError('Validation must contain all original18000 rows')
            training = (evaluate(model, loaders['train'], raw['train'], 128,
                                 directory/f'train_predictions/epoch{epoch+1:03d}')
                        if (epoch+1) % 5 == 0 else None)
            row = dict(epoch=epoch+1, successful_updates=successful, presentations=presentations,
                updates_by_problem=dict(counts), optimization=optimization, task_optimization=per_task,
                train=training, val=validation, parameter_mode='averaged x', evaluation_scope='composite F+g_x')
            history.append(row)
            if prefer_checkpoint(validation, None if best is None else best['val']):
                best = row
                best_member = {key: value.detach().cpu().clone() for key, value in model.member.state_dict().items()}
                publish_best(directory, protocol, arm, best, best_member, base)
            totals, task_stats = Counter(), {}
            save_state(epoch+1, 0, 0., 0.)
            append_jsonl(args.root/'curves'/(arm+'_metrics.jsonl'), row)
            print_eval(f'R66{arm} composite val epoch{epoch+1}', validation['per_problem'], validation['macro'])
            if wandb_run is not None:
                wb = dict(epoch=epoch+1, successful_update=successful,
                    **{'optimization/'+key: value for key, value in optimization.items() if isinstance(value, (int, float))})
                for split, value in (('train_composite', training), ('val_composite', validation)):
                    if value is not None:
                        wb.update({split+'/'+key: entry for key, entry in value['macro'].items()})
                        wb[split+'/native_correct'] = value['native_correct']
                        for task, result in value['per_problem'].items():
                            wb.update({f'{split}/{task}/{key}': result[key] for key in
                                ('top1', 'top2', 'top3', 'ce', 'mean_cost', 'actual_regret_pct')})
                wandb_run.log(wb)
        if best is None or len(history) != 40 or successful != 9600 or presentations != 7200000:
            raise ValueError('R66 requires the complete fixed40-epoch/9600-update/7.2M horizon')
        model.member.load_state_dict(best_member, strict=True)
        model.eval()
        selected_train = evaluate(model, loaders['train'], raw['train'], 128, directory/'train_predictions/best')
        selected_val = evaluate(model, loaders['val'], raw['val'], 128, directory/'val_predictions/best')
        if selected_val['native_correct'] != best['val']['native_correct']:
            raise ValueError('Selected composite validation recount must match original integer selection')
        if state_hash(model.base.state_dict()) != protocol['base_state_sha256']:
            raise ValueError('Frozen saved base x changed during formal training')
        best_eval = dict(best, train=selected_train, selected_validation=selected_val,
                         selected_validation_count_delta=0)
        atomic_json(directory/'best_eval.json', best_eval)
        atomic_json(directory/'result.json', dict(best=best_eval, final=history[-1], epochs_completed=40,
            successful_updates=successful, presentations=presentations, updates_by_problem=dict(counts),
            amp_retries=retries, actual_execution=executed_updates(directory),
            best_checkpoint_sha256=file_hash(directory/'best.pt'), checkpoint_rule=protocol['checkpoint_rule'],
            inference_rule=INFERENCE_RULE, lr_trace_sha256=state_hash(lr_trace), fixed_matched_horizon=True,
            objective_scope=config['objective_scope'], deployment_parameter_mode='averaged x',
            inherited_base_updates=9600, inherited_base_presentations=7200000,
            selector_forwards_per_deployment=2, selected_solver_executions=1,
            selection_uses_original_epoch_validation=True, selected_eval_does_not_reselect=True,
            stop_reason='fixed maximum40 epochs', test_read=False, test_authorized=False))
    finally:
        if wandb_run is not None:
            wandb_run.finish()


def run(arm, args, data):
    directory = args.root/RUN_NAMES[arm]
    if (directory/'result.json').exists():
        if not args.resume:
            raise ValueError('Preserve completed R66 arm')
        return
    (directory/'launches').mkdir(exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    launch = dict(id=stamp, arm=arm, status='started', started_unix=time.time(), host=socket.gethostname(),
        pid=os.getpid(), gpu_uuid=os.environ.get('CUDA_VISIBLE_DEVICES'),
        slurm_step=os.environ.get('SLURM_JOB_ID', '')+'.'+os.environ.get('SLURM_STEP_ID', ''),
        tmux_session='r66_train', formal=True)
    started, status = time.monotonic(), 'failed'
    atomic_json(directory/'launches'/(stamp+'.json'), launch)
    with (directory/'train.log').open('a') as stream, redirect_stdout(Tee(sys.stdout, stream)), redirect_stderr(Tee(sys.stderr, stream)):
        try:
            train(arm, args, directory, data, launch)
            status = 'complete'
        finally:
            launch.update(status=status, wall_seconds=time.monotonic()-started, finished_unix=time.time())
            atomic_json(directory/'launches'/(stamp+'.json'), launch)
            atomic_json(directory/'wall_time.json', [json.loads(path.read_text())
                for path in sorted((directory/'launches').glob('*.json'))])
            if status == 'complete':
                result = json.loads((directory/'result.json').read_text())
                result['wall_seconds'] = launch['wall_seconds']
                atomic_json(directory/'result.json', result)


def publish_validation_candidates(root):
    records, hashes = {}, []
    for arm, name in RUN_NAMES.items():
        result = json.loads((root/name/'result.json').read_text())
        if (result['epochs_completed'], result['successful_updates'], result['presentations']) != (40, 9600, 7200000):
            raise ValueError('Both arms must genuinely complete before publication')
        records[arm] = dict(path=str(root/name/'best.pt'), sha256=result['best_checkpoint_sha256'],
            epoch=result['best']['epoch'], native_correct=result['best']['val']['native_correct'],
            native_total=18000, val=result['best']['val']['macro'], parameter_mode='averaged x')
        hashes.append(result['lr_trace_sha256'])
    if len(set(hashes)) != 1:
        raise ValueError('Both members must have identical successful-update LR schedules')
    official = 'A' if records['A']['native_correct'] >= records['B']['native_correct'] else 'B'
    atomic_json(root/'validation_candidates.json', dict(models=records, proposed_official_arm=official,
        inference_rule=INFERENCE_RULE, coefficient=1., parent_freeze_pending=True, test_authorized=False,
        same_effective_lr_trace_sha256=hashes[0]))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('cpu-check', 'train'), default='train')
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args(argv)
    args.root, args.reference = args.root.resolve(), REFERENCE
    args.device, args.batch_size, args.eval_batch_size, args.wandb = 'cuda:0', 256, 128, True
    torch.set_num_threads(1)
    args.root.mkdir(parents=True, exist_ok=True)
    if args.stage == 'cpu-check':
        prepare_cpu(args)
        return
    if (os.environ.get('R66_PARENT_GPU_BATON') != 'additive-correction-full18'
            or os.environ.get('CUDA_VISIBLE_DEVICES') != GPU_UUID or os.environ.get('SLURM_JOB_ID') != '1465'
            or socket.gethostname().split('.')[0] != 'gpu03'):
        parser.error('Use the authorized existing Slurm1465/gpu03/assigned-UUID launcher')
    if not json.loads((args.root/'cpu_check.json').read_text()).get('passed'):
        parser.error('The one focused necessary CPU check must pass before formal training')
    configure_torch()
    (args.root/'curves').mkdir(exist_ok=True)
    data = load_training_data(args)
    for arm in RUN_NAMES:
        run(arm, args, data)
        gc.collect()
        torch.cuda.empty_cache()
    publish_validation_candidates(args.root)
    atomic_json(args.root/'paired_training_complete.json', dict(completed_utc=datetime.now(timezone.utc).isoformat(),
        arms=list(RUN_NAMES), epochs_per_arm=40, updates_per_arm=9600, presentations_per_arm=7200000,
        inherited_base_updates=9600, inherited_base_presentations=7200000,
        validation_candidates_ready=True, parent_freeze_pending=True, test_authorized=False))
    print('[R66 paired TRAIN/VAL complete] parent freeze pending; TEST has not run', flush=True)


if __name__ == '__main__':
    main()
