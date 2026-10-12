"""One new R69 task-specific arm versus the fixed, reused R68A shared control."""

import argparse
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
from .multitask_probe import Tee, file_hash, gather_batch, state_hash
from .performance_experiment import SOURCE_FILES
from .r42_experiment import prepare_data
from .r59_experiment import GPU_UUID, REFERENCE, WORKSPACE, atomic_json, atomic_save, check_original_scope
from .r64_experiment import dropout_seed, evaluate, executed_updates, paired_triples, prefer_checkpoint, synchronize, triple_indices
from .r64_model import INFERENCE_RULE
from .r69_gradients import GradientLayout, update_triple
from .r69_model import OPTIMIZER_CONFIG, PARAMETER_MODE, cpu_state, make_bank, make_optimizer, update_lr
from .train import configure_torch, print_eval, set_seed
from .training_monitor import capture_rng


ROOT = WORKSPACE/'code/V4/runs/R69_task_specific_finetuning'
CONTROL_ROOT = WORKSPACE/'code/V4/runs/R68_sharpness_aware_finetuning'
RUN_NAME = 'B_task_specific_seed2'
ADOPTED_PLAN = WORKSPACE/'.aris/traces/R68_closure_next_plan.response.md'
REUSE_AMENDMENT = WORKSPACE/'.aris/traces/R69_reuse_shared_control.response.md'
OWNED_SOURCES = ('r69_model.py', 'r69_gradients.py', 'r69_experiment.py', 'r69_cpu_check.py',
                 'run_v4_r69.sh', 'R69_README.md')
REUSED_SOURCES = ('r64_model.py', 'r64_experiment.py', 'r64_gradients.py', 'r65_gradients.py',
    'r45_experiment.py', 'r59_experiment.py', 'r43_experiment.py', 'r42_experiment.py',
    'r40_experiment.py', 'pairwise_objective.py', 'pairwise_selector.py', 'solver_code_encoder.py',
    'r67_model.py', 'r67_experiment.py', 'r68_model.py', 'r68_gradient.py')
UPDATES, PRESENTATIONS, VIEW_ROWS = 9600, 7200000, 14400000


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def append_jsonl(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as stream:
        stream.write(json.dumps(value, allow_nan=False)+'\n')
        stream.flush()


def source_hashes():
    paths = [Path(__file__).parent/name for name in sorted(set(SOURCE_FILES+OWNED_SOURCES+REUSED_SOURCES))]
    paths += [WORKSPACE/'code/unified_selector'/name for name in ('registry.py', 'data.py')]
    return {str(path.relative_to(WORKSPACE)): file_hash(path) for path in paths}


def planned_config():
    return dict(seed=2, new_arms={'B': RUN_NAME}, reused_control='fixed R68A28/global68',
        experiment='one new arm compared with a reused shared control; not an independent control replication',
        epochs=40, updates_per_epoch=240, successful_updates=UPDATES, presentations=PRESENTATIONS,
        successful_view_rows=VIEW_ROWS, active_updates_per_branch=1600, selected_rows_per_branch=400000,
        examples_matched=True, parameter_capacity_matched=False, per_parameter_update_counts_matched=False,
        batch_size=256, effective_batch_size=768, tasks_per_update=3,
        tail_per_task=16, final_effective_batch_size=48, eval_batch_size=128,
        problems=list(PROBLEMS), solver_names=list(GLOBAL_SOLVERS),
        model='18 complete NativeUtilitySelector copies; d128/instance4/joint2/heads4/FFN512/readout256/dropout.1/handcrafted34+ID',
        routing='one original known problem_id selects one complete branch; costs/winners never route',
        expected_shared_parameters=1842953, expected_bank_parameters=33173154,
        initialization='EXACT R68 shared_initialization.pt pre-fine-tuning selector: R67B terminal stage1 averaged-x TRUNK + entire fresh head670003',
        optimizer=OPTIMIZER_CONFIG, parameter_mode=PARAMETER_MODE,
        learning_rate='3e-4*min(successful_update/720,1); successful_update=1..9600',
        warmup_successful_updates=720, rdrop='.35*min(successful_update/720,1)',
        objective='same task mean: two-forward native CE + FP32 legal-pool symmetric RDrop',
        gradient_reduction='FP32 gradient of each active task divided by3 in disjoint branch coordinates',
        clip_norm=1., clip_calls_per_update=1, optimizer_steps_per_update=1,
        inactive='grad=None; no weight decay, parameter, Adam moment or Adam counter advancement',
        adam_clock='global successful-update LR; per-parameter Adam step only when that parameter is active',
        retry='whole same triple; restore pre-update RNG/clear gradients; lower scaler; no LR/cursor/state/counter advance;20 attempts',
        amp_growth='2000 successful updates; reset successful-growth counter on overflow',
        schedule='EXACT R67/R68 global41-80 task/row/dropout schedule; real256; tails16',
        schedule_regenerate='paired_triples(sizes, epochs=80, seed=2, batch_size=256); slice permutations/cycles/triples[40:80]',
        dropout_successful_offset=9600, saved_row_schedule_cache=False,
        full_coverage=True, drop_last=False, microbatching=False, accumulation=False,
        full_val_eval='every epoch; original18*1000 FP32batch128 and tails; eval mode',
        full_train_eval='every5/final/selected best; original18*10000 FP32batch128 and tails; eval mode',
        checkpoint_rule='max INTEGER full18 validation native_correct/18000; earliest exacttie; one common-epoch complete18 bundle',
        official_arm_rule='B only when best validation native_correct>8644; otherwise fixed A28',
        inference_rule=INFERENCE_RULE, selector_forwards_per_query=1, final_completed_solvers_per_query=1,
        no_per_task_best_stitching=True, cost_based_selection=False,
        sam=False, projector_included=False, parameter_averaging=False, new_labels=False,
        augmentation=False, candidate_pruning=False, task_grouping_search=False,
        scheduler=None, early_stopping=False, plateau_search=False, all_parameters_trainable=True,
        directed_ATSP_preserved=True, original_pools_global_ids_native_winners=True,
        cost_dtype='original FP64 targets/metrics; labels and costs excluded by input whitelist',
        adopted_plan=str(ADOPTED_PLAN), reuse_amendment=str(REUSE_AMENDMENT),
        review_independence='same-family', acceptance_status='provisional',
        wandb=dict(project='selector', entity='yjkds-southern-university-of-science-technology', mode='offline'),
        inherited_stage1_updates_once=9600, inherited_stage1_presentations_once=7200000,
        fresh_encoders=False, inherited_pretraining_is_separate=True,
        historical_test_exposure=True, new_R68_test_used_for_design_or_selection=False,
        legacy_B_semantics='eight feasible-domain mismatches unresolved/disclosed; R58 deferred; not a native70 veto',
        target_native_macro_top1=.70, goal_status='active; unachieved',
        test_read=False, test_authorized=False, automatic_test=False, automatic_analysis=False,
        control_retraining=False, control_recount=False, control_reselection=False, control_retest=False,
        reference_inference=False, pilot=False, gpu_sizing=False, git_actions=False,
        assets='weights/NPZ/rawupdates/W&B/cache/schedules local; parent publishes one completed version')


def prepare(args):
    if (args.root/'formal_started.json').exists() or (args.root/'protocol.json').exists():
        raise ValueError('R69 formal launch already prepared/started; preserve artifacts, never replay')
    protocol = planned_config()
    protocol['source_hashes'] = source_hashes()
    for relative in protocol['source_hashes']:
        destination = args.root/'source_launch'/relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(WORKSPACE/relative, destination)
    atomic_json(args.root/'source_launch/hashes.json', protocol['source_hashes'])
    shutil.copy2(ADOPTED_PLAN, args.root/'adopted_R69_plan.response.md')
    shutil.copy2(REUSE_AMENDMENT, args.root/'adopted_control_reuse.response.md')
    shared_protocol = json.loads((CONTROL_ROOT/'protocol.json').read_text())
    initialization = json.loads((CONTROL_ROOT/'initialization.json').read_text())
    schedule_reference = json.loads((CONTROL_ROOT/'schedule_seed2.json').read_text())
    source = CONTROL_ROOT/'shared_initialization.pt'
    initial = torch.load(source, map_location='cpu', weights_only=False)['selector']
    if state_hash(initial) != initialization['starting_selector_hash']:
        raise ValueError('R69 requires the exact saved R68 PRE-fine-tuning initialization')
    initialization = dict(initialization, explicit_identical_copy_all18=True,
        initialization_container=str(source), initialized_from_trained_R68=False,
        initialized_from_R67_stage2=False, deployment='eighteen ordinary fully trainable complete selectors')
    inherited = json.loads((CONTROL_ROOT/'inherited_source_identity.json').read_text())
    inherited['reused_prefinetuning_container'] = dict(path=str(source),
        state_sha256=initialization['starting_selector_hash'], identity_source=str(CONTROL_ROOT/'initialization.json'))
    selected_control = json.loads((CONTROL_ROOT/'validation_candidates.json').read_text())['models']['A']
    if (selected_control['epoch'] != 28 or selected_control['global_epoch'] != 68
            or selected_control['native_correct'] != 8644 or selected_control['native_total'] != 18000):
        raise ValueError('R69 reuses only fixed R68A28/global68/8644 control identity')
    control = dict(model=selected_control, run=str(CONTROL_ROOT/'A_plain_adamw_seed2'),
        history=str(CONTROL_ROOT/'A_plain_adamw_seed2/history.json'),
        result=str(CONTROL_ROOT/'A_plain_adamw_seed2/result.json'),
        protocol=str(CONTROL_ROOT/'protocol.json'), initialization=str(CONTROL_ROOT/'initialization.json'),
        schedule=str(CONTROL_ROOT/'schedule_seed2.json'),
        identity='existing saved R68 TRAIN/VAL identity; checkpoint bytes not reloaded/recounted/reselected',
        reuse_is_protocol_equivalence_not_independent_replication=True,
        historical_updates=9600, historical_presentations=7200000,
        historical_successful_view_rows=14400000, historical_attempted_view_rows=14405632,
        historical_wall_seconds=3002.024237, historical_amp_retries=5,
        new_compute=0, control_test='parent reuses existing A TEST unchanged; no read or inference in this phase',
        test_read=False, trained_A_weights_used_for_B_initialization=False)
    atomic_json(args.root/'reused_control.json', control)
    atomic_json(args.root/'inheritance_budget.json', dict(
        new_B=dict(updates=UPDATES, selected_rows=PRESENTATIONS, successful_view_rows=VIEW_ROWS,
                   active_updates_per_branch=1600, selected_rows_per_branch=400000, runtime='measured new B only'),
        reused_A=dict(updates=9600, selected_rows=7200000, successful_view_rows=14400000,
                      wall_seconds=3002.024237, amp_retries=5, accounting='historical; no new compute'),
        inherited_stage1_once=dict(updates=9600, selected_rows=7200000,
            source='R67B terminal stage1 averaged-x trunk', accounting='historical pretraining counted once separately'),
        parameter_capacity_matched=False, per_parameter_update_counts_matched=False))
    if not (args.reference/'geometry_cache/stats.json').is_file():
        raise ValueError('Existing original TRAIN-normalized geometry cache required')
    loaders, raw, geometry = prepare_data(args)
    check_original_scope(loaders, raw)
    if geometry != shared_protocol['geometry_stats']:
        raise ValueError('R69 retains the exact original TRAIN-only R67/R68 geometry normalization')
    if len(GLOBAL_SOLVERS) != 22 or sum(len(record['pool_ids']) for record in raw['train'].values()) != 128:
        raise ValueError('All22 solver identities and128 original legal deployments required')
    sizes = {task: loader.size for task, loader in loaders['train'].items()}
    full = paired_triples(sizes, epochs=80, seed=2, batch_size=256)
    if state_hash(full) != schedule_reference['original80epoch_sha256']:
        raise ValueError('R69 schedule must reproduce the original R67/R68 80-epoch identity')
    schedule = {key: value[40:80] if key in ('permutations', 'cycles', 'triples') else value
                for key, value in full.items()}
    if state_hash(schedule) != schedule_reference['selected_stage2_sha256']:
        raise ValueError('R69 must reuse exact global41-80 task/row/dropout order')
    del full
    shutil.copy2(source, args.root/'shared_initialization.pt')
    atomic_json(args.root/'initialization.json', initialization)
    atomic_json(args.root/'inherited_source_identity.json', inherited)
    atomic_json(args.root/'schedule_seed2.json', dict(schedule_reference,
        reference=str(CONTROL_ROOT/'schedule_seed2.json'), shared_control_new_execution=False))
    protocol.update(model_params=shared_protocol['model_params'], initialization=initialization,
        inherited_identity=inherited, schedule_identity=schedule_reference,
        reused_control=control, geometry_stats=geometry, sizes=sizes,
        data_manifest=json.loads((args.root/'data_manifest.json').read_text()),
        adopted_plan_sha256=file_hash(ADOPTED_PLAN), reuse_amendment_sha256=file_hash(REUSE_AMENDMENT))
    protocol = json.loads(json.dumps(protocol))
    atomic_json(args.root/'protocol.json', protocol)
    directory = args.root/RUN_NAME
    directory.mkdir(exist_ok=True)
    atomic_json(directory/'args.json', dict(protocol, arm='B', model_type='complete_task_specific_bank'))
    atomic_json(args.root/'formal_started.json', dict(started_utc=utc_now(), pid=os.getpid(),
        new_arms=['B'], control_reused=True, test_authorized=False, no_automatic_restart=True))
    print('[R69 prepared] one new B; exact PRE-fine-tuning state and global41-80 schedule; next update FORMAL', flush=True)
    return protocol, loaders, raw, schedule, initial


def first_update_handoff(args, launch, metrics):
    atomic_json(args.root/'handoff.json', dict(status='formal B training running',
        command='bash code/V4/run_v4_r69.sh train',
        training_command='python -B -u -m code.V4.r69_experiment --stage train',
        tmux='r69_train', tmux_pane=launch['tmux_pane'], tmux_pane_pid=launch['tmux_pane_pid'],
        tmux_server_host=launch['tmux_server_host'], slurm_job='1465', slurm_step=launch['slurm_step'],
        host=launch['host'], pid=launch['pid'], parent_pid=os.getppid(), gpu_uuid=GPU_UUID, gpu_index=1,
        arm='B', new_arms=['B'], reused_control='fixed R68A28/global68/8644 correct; no new A compute',
        real_batch_per_task=256, task_rows=metrics['task_rows'], effective_batch=metrics['rows'],
        tail_batch_per_task=16, tail_effective_batch=48, first_finite_formal_update=metrics,
        parameter_counts=json.loads((args.root/'parameter_counts.json').read_text()),
        cpu_check=json.loads((args.root/'cpu_check.json').read_text()),
        memory={key: metrics[key] for key in ('peak_memory_gib', 'peak_reserved_gib', 'physical_used_gib', 'physical_total_gib')},
        changed_files=['code/V4/'+name for name in OWNED_SOURCES]+[str(args.root.relative_to(WORKSPACE))+'/'],
        next_normal_monitor_instruction='Every180-300s read current_status.json and small pipeline/launch/history summaries for r69_train/exact Slurm step. Let fixed40E B finish. Never duplicate/restart an observation-timedout job. After terminal0/training_complete/validation_candidates, parent freezes one common-epoch B bundle and official arm by VAL, then authorizes ONE original full18 TEST for B only. Existing A TEST is reused without inference/recount. No Git actions by executor.',
        inherited_pretraining='R67B stage1 terminal trunk9600updates/7.2Mrows counted once separately; entire fresh head670003',
        test_authorized=False, no_git_actions=True))


def train_arm(args, data, launch):
    protocol, loaders, raw, schedule, initial = data
    directory = args.root/RUN_NAME
    config = dict(protocol, arm='B', model_type='complete_task_specific_bank')
    model = make_bank(protocol['model_params'], initial, args.device)
    branch_hashes = [state_hash(branch.state_dict()) for branch in model.branches]
    if set(branch_hashes) != {protocol['initialization']['starting_selector_hash']}:
        raise ValueError('All18 branches must start from the identical complete pre-fine-tuning state')
    parameter_counts = dict(shared=sum(p.numel() for p in model.branches[0].parameters()),
        B=sum(p.numel() for p in model.parameters()), branches=18,
        trainable_B=sum(p.numel() for p in model.parameters() if p.requires_grad),
        capacity_matched=False, per_parameter_update_counts_matched=False,
        branch_active_updates=1600, shared_historical_updates=9600)
    if parameter_counts['shared'] != 1842953 or parameter_counts['B'] != 33173154:
        raise ValueError('The complete adopted NativeUtilitySelector capacity must be retained')
    atomic_json(args.root/'parameter_counts.json', parameter_counts)
    optimizer = make_optimizer(model)
    scaler = torch.amp.GradScaler('cuda', enabled=True)
    layout = GradientLayout(model)
    atomic_json(directory/'initialization.json', dict(branch_hashes=branch_hashes,
        explicit_identical_full18=True, inherited_trunk=True, entire_fresh_head=True, projector_included=False,
        all_parameters_trainable=all(p.requires_grad for p in model.parameters()),
        optimizer_fresh=not optimizer.state, scaler_fresh=scaler.state_dict(),
        parameter_counts=parameter_counts, gradient_layout=layout.receipt,
        new_successful_updates=0, new_presentations=0))
    history, lr_trace, best, best_model = [], [], None, None
    successful = presentations = view_presentations = attempted_views = growth_successes = total_retries = 0
    counts, task_rows = Counter(), Counter()
    wandb_run = None
    if args.wandb:
        import wandb
        wandb_run = wandb.init(**protocol['wandb'], name='R69_'+RUN_NAME,
            config=config, dir=str(directory), id='r69-b-'+launch['id'])

    def save_state(epoch, step):
        atomic_save(directory/'last.pt', dict(model=cpu_state(model), optimizer=optimizer.state_dict(),
            scaler=scaler.state_dict(), args=config, epoch=epoch, global_epoch=40+epoch, next_step=step,
            successful_updates=successful, presentations=presentations, successful_view_rows=view_presentations,
            attempted_view_rows=attempted_views, scale_growth_successes=growth_successes,
            amp_retries=total_retries, updates_by_problem=dict(counts), rows_by_problem=dict(task_rows),
            rng=capture_rng(), history=history, best=best, lr_trace=lr_trace, parameter_mode=PARAMETER_MODE,
            all18_common_epoch=True, checkpoint_mode='eval', projector_included=False))
        atomic_json(directory/'history.json', history)

    def save_bundle(path, state, epoch, validation):
        atomic_save(path, dict(model=state, args=config, epoch=epoch-1, global_epoch=40+epoch,
            common_epoch=epoch, branch_epochs=[epoch]*18, problem_names=list(PROBLEMS),
            native_correct=validation['native_correct'], native_total=18000, macro=validation['macro'],
            parameter_mode=PARAMETER_MODE, checkpoint_mode='eval', projector_included=False,
            inference_rule=INFERENCE_RULE, successful_updates=epoch*240,
            presentations=epoch*180000, model_state_sha256=state_hash(state), all18_complete_branches=True))

    try:
        for epoch in range(1, 41):
            model.train()
            torch.cuda.reset_peak_memory_stats(args.device)
            started = time.monotonic()
            totals, task_stats = Counter(), {}
            for step in range(240):
                entries = triple_indices(schedule, epoch-1, step)
                batches = [gather_batch(loaders['train'][task], indices) for task, indices in entries]
                expected_rows = 256 if step < 234 else 16
                if [len(batch['ind']) for batch in batches] != [expected_rows]*3:
                    raise ValueError('R69 retains real256/task and16/task tails')
                weight, lr = .35*min((successful+1)/720, 1.), update_lr(successful)
                view_seed = dropout_seed(9600+successful)
                set_seed(view_seed)
                synchronize(args.device)
                before = time.monotonic()
                values = update_triple(model, optimizer, scaler, batches, weight, layout, lr, growth_successes)
                synchronize(args.device)
                seconds = time.monotonic()-before
                growth_successes = values['scale_growth_successes']
                rows = sum(len(indices) for _, indices in entries)
                successful += 1
                presentations += rows
                view_presentations += values['successful_view_rows']
                attempted_views += values['attempted_view_rows']
                total_retries += values['retries']
                lr_trace.append(lr)
                free_bytes, total_bytes = torch.cuda.mem_get_info(args.device)
                metrics = dict(launch_id=launch['id'], arm='B', epoch=epoch, global_epoch=40+epoch,
                    stage=2, update_in_epoch=step+1, successful_update=successful,
                    schedule_successful_update=9600+successful, presentations=presentations,
                    view_presentations=view_presentations, attempted_view_presentations=attempted_views,
                    tasks=[task for task, _ in entries], task_rows=[len(indices) for _, indices in entries], rows=rows,
                    actual_forward_rows=[len(batch['ind']) for batch in batches],
                    batch_indices_sha256=[state_hash(indices) for _, indices in entries],
                    dropout_seed=view_seed, lr=lr, consistency_weight=weight, gradient_parameter_mode=PARAMETER_MODE,
                    seconds=seconds, queries_per_second=rows/seconds,
                    peak_memory_gib=torch.cuda.max_memory_allocated(args.device)/2**30,
                    peak_reserved_gib=torch.cuda.max_memory_reserved(args.device)/2**30,
                    physical_used_gib=(total_bytes-free_bytes)/2**30,
                    physical_total_gib=total_bytes/2**30, **values)
                append_jsonl(directory/'updates.jsonl', metrics)
                for key in ('loss', 'ce', 'kl'):
                    totals[key] += values[key]*rows
                for key in ('grad_norm', 'clip_coefficient', 'retries'):
                    totals[key] += values[key]
                totals['samples'] += rows
                totals['updates'] += 1
                totals['successful_view_rows'] += values['successful_view_rows']
                totals['attempted_view_rows'] += values['attempted_view_rows']
                for index, (task, indices) in enumerate(entries):
                    counts[task] += 1
                    task_rows[task] += len(indices)
                    item = task_stats.setdefault(task, dict(rows=0, batches=0, full_grad_norm=0.,
                                                           loss=0., ce=0., kl=0.))
                    item['rows'] += len(indices)
                    item['batches'] += 1
                    for key in ('loss', 'ce', 'kl'):
                        item[key] += values['task_losses'][index][key]*len(indices)
                    item['full_grad_norm'] += values['full_task_grad_norms'][index]
                if successful == 1:
                    atomic_json(directory/'first_formal_update.json', dict(metrics,
                        selected_indices={task: indices.tolist() for task, indices in entries}, launch=launch))
                    print('[R69 first FORMAL update] '+json.dumps({key: metrics[key] for key in
                        ('arm', 'successful_update', 'tasks', 'task_rows', 'rows', 'lr', 'loss',
                         'grad_norm', 'seconds', 'peak_memory_gib', 'retries', 'finite_gradient')}), flush=True)
                if successful == 1 or (step+1) % 60 == 0:
                    save_state(epoch, step+1)
                    launch.update(status='running', epoch=epoch, global_epoch=40+epoch,
                        successful_updates=successful, presentations=presentations, heartbeat_unix=time.time())
                    atomic_json(directory/'launches'/(launch['id']+'.json'), launch)
                    atomic_json(args.root/'current_status.json', dict(launch, last_metrics=metrics))
                    if successful == 1:
                        first_update_handoff(args, launch, metrics)
                    print(f'[R69B] epoch={epoch}/40 update={step+1}/240 total={successful} '
                          f'loss={values["loss"]:.6f} lr={lr:.9g} retries={total_retries}', flush=True)
                    if wandb_run is not None:
                        wandb_run.log({key: metrics[key] for key in ('epoch', 'global_epoch', 'successful_update',
                            'loss', 'ce', 'kl', 'lr', 'grad_norm', 'scale', 'retries', 'peak_memory_gib', 'queries_per_second')})
            if (totals['samples'] != 180000 or totals['updates'] != 240 or len(task_stats) != 18
                    or successful != epoch*240
                    or any(item['rows'] != 10000 or item['batches'] != 40 for item in task_stats.values())):
                raise ValueError('Each epoch must cover all180000 original TRAIN rows once')
            optimization = {key: totals[key]/180000 for key in ('loss', 'ce', 'kl')}
            optimization.update({key: totals[key]/240 for key in ('grad_norm', 'clip_coefficient')})
            optimization.update(lr=lr, lr_first=lr_trace[(epoch-1)*240], consistency_weight=weight,
                samples=180000, updates=240, amp_retries=int(totals['retries']),
                successful_view_rows=int(totals['successful_view_rows']),
                attempted_view_rows=int(totals['attempted_view_rows']), seconds=time.monotonic()-started,
                peak_memory_gib=torch.cuda.max_memory_allocated(args.device)/2**30)
            for item in task_stats.values():
                for key in ('loss', 'ce', 'kl'):
                    item[key] /= item['rows']
                item['full_grad_norm'] /= item['batches']
            model.eval()
            validation = evaluate(model, loaders['val'], raw['val'], 128, directory/f'val_predictions/epoch{epoch:03d}')
            if validation['native_total'] != 18000 or any(value['n'] != 1000 for value in validation['per_problem'].values()):
                raise ValueError('Every validation requires all18000 original rows')
            training = None
            if epoch % 5 == 0:
                training = evaluate(model, loaders['train'], raw['train'], 128, directory/f'train_predictions/epoch{epoch:03d}')
                if training['native_total'] != 180000:
                    raise ValueError('Full eval-mode TRAIN coverage required')
            row = dict(epoch=epoch, global_epoch=40+epoch, stage=2, successful_updates=successful,
                presentations=presentations, successful_view_rows=view_presentations, attempted_view_rows=attempted_views,
                updates_by_problem=dict(counts), rows_by_problem=dict(task_rows), optimization=optimization,
                task_optimization=task_stats, train=training, val=validation, parameter_mode=PARAMETER_MODE,
                all18_common_epoch=True)
            history.append(row)
            if prefer_checkpoint(validation, None if best is None else best['val']):
                best, best_model = row, cpu_state(model)
                unique = directory/f'best_checkpoints/epoch{epoch:03d}.pt'
                unique.parent.mkdir(exist_ok=True)
                save_bundle(unique, best_model, epoch, validation)
                shutil.copy2(unique, directory/'best.pt')
                atomic_json(directory/'best_eval.json', best)
            if epoch == 40:
                save_bundle(directory/'final.pt', cpu_state(model), epoch, validation)
            save_state(epoch, 240)
            append_jsonl(args.root/'curves/B_metrics.jsonl', row)
            atomic_json(args.root/'current_status.json', dict(launch, status='running', completed_epochs=epoch,
                successful_updates=successful, presentations=presentations,
                selected_B_epoch=best['epoch'], selected_B_validation_correct=best['val']['native_correct'],
                last_validation_correct=validation['native_correct'], heartbeat_unix=time.time()))
            print_eval(f'R69B val epoch{epoch}', validation['per_problem'], validation['macro'])
            if wandb_run is not None:
                wb = dict(epoch=epoch, global_epoch=40+epoch,
                          **{'optimization/'+key: value for key, value in optimization.items()})
                for split, value in (('train_eval', training), ('val', validation)):
                    if value is not None:
                        wb.update({split+'/'+key: entry for key, entry in value['macro'].items()})
                        wb[split+'/native_correct'] = value['native_correct']
                        for task, result in value['per_problem'].items():
                            wb.update({f'{split}/{task}/{key}': result[key] for key in
                                       ('top1', 'top2', 'top3', 'ce', 'mean_cost', 'actual_regret_pct')})
                wandb_run.log(wb)
        if (successful != UPDATES or presentations != PRESENTATIONS or view_presentations != VIEW_ROWS
                or len(history) != 40 or set(counts.values()) != {1600} or set(task_rows.values()) != {400000}):
            raise ValueError('R69 requires fixed40E/9600updates/7.2Mrows/14.4Mviews and1600updates/400000rows per branch')
        counters = {name: int(optimizer.state[parameter]['step']) if parameter in optimizer.state else 0
                    for name, parameter in model.named_parameters()}
        atomic_json(directory/'parameter_step_counts.json', dict(steps=counters, branch_activations=dict(counts),
            explanation='Per-parameter counters advance only with a gradient; unused input paths can remain inactive'))
        model.load_state_dict(best_model, strict=True)
        model.eval()
        selected_train = evaluate(model, loaders['train'], raw['train'], 128, directory/'train_predictions/best')
        if selected_train['native_total'] != 180000:
            raise ValueError('Selected bundle TRAIN evaluation requires all180000 original rows')
        best_eval = dict(best, train=selected_train)
        atomic_json(directory/'best_eval.json', best_eval)
        atomic_json(directory/'result.json', dict(best=best_eval, final=history[-1], epochs_completed=40,
            successful_updates=successful, presentations=presentations, successful_view_rows=view_presentations,
            attempted_view_rows=attempted_views, amp_retries=total_retries, actual_execution=executed_updates(directory),
            updates_by_problem=dict(counts), rows_by_problem=dict(task_rows),
            best_checkpoint_sha256=file_hash(directory/'best.pt'), final_checkpoint_sha256=file_hash(directory/'final.pt'),
            selected_epoch_checkpoint=str(directory/f'best_checkpoints/epoch{best["epoch"]:03d}.pt'),
            lr_trace_sha256=state_hash(lr_trace), parameter_counts=parameter_counts,
            deployment_parameter_mode=PARAMETER_MODE, checkpoint_rule=protocol['checkpoint_rule'],
            inference_rule=INFERENCE_RULE, initialization=protocol['initialization'],
            fixed_horizon=True, stop_reason='fixed40epoch task-specific fine-tuning',
            all18_common_epoch=True, new_arms=['B'], reused_control=str(args.root/'reused_control.json'),
            test_read=False, test_authorized=False))
    finally:
        if wandb_run is not None:
            wandb_run.finish()


def run(args, data):
    directory = args.root/RUN_NAME
    (directory/'launches').mkdir(exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    launch = dict(id=stamp, arm='B', status='started', started_unix=time.time(), host=socket.gethostname(),
        pid=os.getpid(), gpu_uuid=GPU_UUID, formal=True, tmux_session='r69_train',
        tmux_pane=os.environ.get('R69_TMUX_PANE_ID', ''), tmux_pane_pid=os.environ.get('R69_TMUX_PANE_PID', ''),
        tmux_server_host=os.environ.get('R69_TMUX_SERVER_HOST', ''),
        slurm_step=os.environ.get('SLURM_JOB_ID', '')+'.'+os.environ.get('SLURM_STEP_ID', ''))
    started, status = time.monotonic(), 'failed'
    atomic_json(directory/'launches'/(stamp+'.json'), launch)
    with (directory/'train.log').open('a') as stream, redirect_stdout(Tee(sys.stdout, stream)), redirect_stderr(Tee(sys.stderr, stream)):
        try:
            train_arm(args, data, launch)
            status = 'complete'
        finally:
            launch.update(status=status, wall_seconds=time.monotonic()-started, finished_unix=time.time())
            atomic_json(directory/'launches'/(stamp+'.json'), launch)
            atomic_json(directory/'wall_time.json', [launch])
            atomic_json(args.root/'current_status.json', launch)
            if status == 'complete':
                result = json.loads((directory/'result.json').read_text())
                result['wall_seconds'] = launch['wall_seconds']
                atomic_json(directory/'result.json', result)


def publish_candidate(root):
    result = json.loads((root/RUN_NAME/'result.json').read_text())
    control = json.loads((root/'reused_control.json').read_text())['model']
    candidate = dict(path=str(root/RUN_NAME/'best.pt'), sha256=result['best_checkpoint_sha256'],
        selected_epoch_checkpoint=result['selected_epoch_checkpoint'], epoch=result['best']['epoch'],
        global_epoch=result['best']['global_epoch'], native_correct=result['best']['val']['native_correct'],
        native_total=18000, val=result['best']['val']['macro'], parameter_mode=PARAMETER_MODE,
        model_type='complete_task_specific_bank', all18_common_epoch=True)
    official = 'B' if candidate['native_correct'] > 8644 else 'A'
    atomic_json(root/'validation_candidates.json', dict(models={'A': control, 'B': candidate},
        proposed_official_arm=official, official_rule='B strictly>8644; tie/lower fixed A',
        new_arms=['B'], reused_arms=['A'], parent_freeze_pending=True, inference_rule=INFERENCE_RULE,
        test_authorized=False, later_test='parent freezes B bundle and official arm, then ONE original full18 TEST for new B; reuse existing A TEST unchanged'))


def cpu_check(args):
    from .r69_cpu_check import focused_check
    path = args.root/'cpu_check_invocations.jsonl'
    if path.exists():
        previous = [json.loads(line) for line in path.read_text().splitlines()]
        if any(item['status'] == 'passed' for item in previous):
            raise ValueError('The one focused R69 CPU check already passed; no repeat campaign')
        invocation = 1+sum(item['status'] == 'started' for item in previous)
    else:
        invocation = 1
    started = time.monotonic()
    receipt = dict(invocation=invocation, type='focused synthetic CPU check', started_utc=utc_now(),
                   pid=os.getpid(), status='started', gpu_used=False, source_hashes=source_hashes())
    append_jsonl(path, receipt)
    with (args.root/'cpu_check.log').open('a') as stream, redirect_stdout(Tee(sys.stdout, stream)), redirect_stderr(Tee(sys.stderr, stream)):
        try:
            result = dict(focused_check(), invocation=invocation, started_utc=receipt['started_utc'],
                          wall_seconds=time.monotonic()-started, source_hashes=receipt['source_hashes'])
            atomic_json(args.root/'cpu_check.json', result)
            append_jsonl(path, dict(receipt, status='passed', wall_seconds=result['wall_seconds']))
            print('[R69 focused CPU check] '+json.dumps(result), flush=True)
        except BaseException as error:
            append_jsonl(path, dict(receipt, status='failed', wall_seconds=time.monotonic()-started,
                                   error_type=type(error).__name__, error=str(error)))
            raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('cpu-check', 'train'), default='train')
    parser.add_argument('--wandb', action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args(argv)
    args.root, args.reference, args.device, args.batch_size = ROOT, REFERENCE, 'cuda:0', 256
    torch.set_num_threads(1)
    args.root.mkdir(parents=True, exist_ok=True)
    if args.stage == 'cpu-check':
        cpu_check(args)
        return
    if (os.environ.get('CUDA_VISIBLE_DEVICES') != GPU_UUID or os.environ.get('SLURM_JOB_ID') != '1465'
            or socket.gethostname().split('.')[0] != 'gpu03'):
        parser.error('R69 requires assigned existing Slurm1465/gpu03/index1 UUID launcher')
    check = json.loads((args.root/'cpu_check.json').read_text())
    if not check.get('passed') or check['source_hashes'] != source_hashes():
        parser.error('The one focused CPU check must pass on these exact launch sources')
    configure_torch()
    try:
        data = prepare(args)
        run(args, data)
        publish_candidate(args.root)
        atomic_json(args.root/'training_complete.json', dict(completed_utc=utc_now(), new_arms=['B'],
            reused_control='fixed R68A28', epochs=40, new_updates=UPDATES, new_presentations=PRESENTATIONS,
            successful_view_rows=VIEW_ROWS, inherited_stage1_updates_once=9600,
            inherited_stage1_presentations_once=7200000, parent_freeze_pending=True, test_authorized=False))
        print('[R69 training complete] one new B; parent freeze pending; no TEST/analysis executed', flush=True)
    except BaseException as error:
        atomic_json(args.root/'terminal_failure.json', dict(recorded_utc=utc_now(),
            error_type=type(error).__name__, error=str(error), pid=os.getpid(),
            slurm_step=os.environ.get('SLURM_JOB_ID', '')+'.'+os.environ.get('SLURM_STEP_ID', ''),
            automatic_restart=False, samples_skipped=False, duplicate_attempt_authorized=False))
        raise


if __name__ == '__main__':
    main()
