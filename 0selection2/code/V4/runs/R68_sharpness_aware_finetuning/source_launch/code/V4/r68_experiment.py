"""One R68 AdamW/SAM contrast from R67B's terminal contrastive trunk."""

import argparse
import gc
import json
import os
import shutil
import socket
import subprocess
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
from .r68_gradient import GradientLayout, update_triple
from .r68_model import OPTIMIZER_CONFIG, PARAMETER_MODE, cpu_state, make_optimizer, make_selector, starting_state, update_lr
from .train import configure_torch, print_eval, set_seed
from .training_monitor import capture_rng


ROOT = WORKSPACE/'code/V4/runs/R68_sharpness_aware_finetuning'
PRIOR = WORKSPACE/'code/V4/runs/R67_supervised_contrastive_full18'
ADOPTED_PLAN = WORKSPACE/'.aris/traces/R67_closure_next_plan.response.md'
RUN_NAMES = {'A': 'A_plain_adamw_seed2', 'B': 'B_sam_adamw_seed2'}
OWNED_SOURCES = ('r68_model.py', 'r68_gradient.py', 'r68_experiment.py', 'r68_cpu_check.py',
                 'run_v4_r68.sh', 'R68_README.md')
REUSED_SOURCES = ('r64_model.py', 'r64_experiment.py', 'r64_gradients.py', 'r65_gradients.py',
    'r45_experiment.py', 'r59_experiment.py', 'r43_experiment.py', 'r42_experiment.py',
    'r40_experiment.py', 'pairwise_objective.py', 'pairwise_selector.py', 'solver_code_encoder.py',
    'r67_model.py', 'r67_experiment.py')
UPDATES, PRESENTATIONS = 9600, 7200000


def append_jsonl(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as stream:
        stream.write(json.dumps(value, allow_nan=False)+'\n')
        stream.flush()


def source_hashes():
    paths = [Path(__file__).parent/name for name in sorted(set(SOURCE_FILES+OWNED_SOURCES+REUSED_SOURCES))]
    paths += [WORKSPACE/'code/unified_selector'/name for name in ('registry.py', 'data.py')]
    return {str(path.relative_to(WORKSPACE)): file_hash(path) for path in paths}


def worktree_identity():
    repository = next((path for path in (WORKSPACE, *WORKSPACE.parents)
                       if (path/'.git').exists()), None)
    if repository is None:
        raise ValueError('Cannot record actual repository identity')
    directory = repository/'.git'
    if directory.is_file():
        directory = (repository/directory.read_text().strip().removeprefix('gitdir: ')).resolve()
    common = directory
    if (directory/'commondir').is_file():
        common = (directory/(directory/'commondir').read_text().strip()).resolve()
    head = (directory/'HEAD').read_text().strip()
    reference = head.removeprefix('ref: ') if head.startswith('ref: ') else None
    commit = head if reference is None else None
    if reference:
        ref = common/reference
        if ref.exists():
            commit = ref.read_text().strip()
        elif (common/'packed-refs').exists():
            for line in (common/'packed-refs').read_text().splitlines():
                if line and not line.startswith(('#', '^')):
                    value, name = line.split(' ', 1)
                    if name == reference:
                        commit = value
    if commit is None:
        raise ValueError('Actual repository HEAD could not be resolved')
    return dict(head_commit=commit, head_reference=reference, repository_root=str(repository),
        workspace_relative_path=str(WORKSPACE.relative_to(repository)), recorded_utc=datetime.now(timezone.utc).isoformat(),
        read_only_filesystem_identity=True, git_commands_executed=False,
        worktree_version='source_launch/hashes.json is authoritative for actual launch sources',
        unrelated_staged_changes_enumerated=False)


def planned_config():
    return dict(seed=2, arms=RUN_NAMES, epochs_per_arm=40, updates_per_epoch=240,
        successful_updates_per_arm=UPDATES, successful_updates_pair=2*UPDATES,
        presentations_per_epoch=180000, presentations_per_arm=PRESENTATIONS,
        presentations_pair=2*PRESENTATIONS, inherited_stage1_updates_once=9600,
        inherited_stage1_presentations_once=7200000, fresh_encoders=False,
        successful_view_rows_per_arm={'A': 14400000, 'B': 28800000}, flops_matched=False,
        batch_size=256, effective_batch_size=768, tail_per_task=16, final_effective_batch_size=48,
        tasks_per_update=3, eval_batch_size=128, problems=list(PROBLEMS), solver_names=list(GLOBAL_SOLVERS),
        model='R64 NativeUtilitySelector d128, instance4+joint2, heads4, FFN512, score256, dropout.1, handcrafted34+ID',
        initialization='R67B stage1 terminal averaged-x trunk + entire R67 shared seed670003 fresh head; explicit full-state copy A/B',
        optimizer=OPTIMIZER_CONFIG, deployment_parameter_mode=PARAMETER_MODE,
        learning_rate='3e-4*min(successful_update/720,1); u=1..9600',
        warmup_successful_updates=720, rdrop='.35*min(successful_update/720,1)',
        objective='R64 two-forward native CE + FP32 symmetric legal-pool RDrop KL',
        sam_radius=.05, sam_rule='one global L2 perturbation of whole3-task unscaled FP32 mean BEFORE clipping; replay pre-pass RNG; exact copy restoration; clip SECOND mean once1; one AdamW step',
        sam_first_gradient_clipped=False, plain_gradient_clipped_once=True,
        optimizer_steps_per_update=1, weight_decay_steps_per_update=1,
        gradient_reduction='unscaled FP32 arithmetic mean of3 same-state task gradients; unused grad=None',
        retry='whole same3-task update; exact restore weights and pre-update RNG; lower AMP scale; no LR/state/row/counter advancement until success;20 attempts',
        amp_growth='ordinary2000 successful-update growth interval; backoff/reset on failure; explicit growth_successes counter checkpointed',
        schedule='EXACT R67 stage2 slice from80epoch R64 paired_triples seed2; global epochs41-80; dropout_seed(9600+new_successful_before)',
        schedule_regenerate='paired_triples(sizes, epochs=80, seed=2, batch_size=256); verify R67 full hash; slice permutations/cycles/triples[40:80]',
        saved_row_schedule_cache=False, full_coverage=True, drop_last=False, accumulation=False,
        microbatching=False, full_train_eval='every5/final/selected40epoch fine-tune',
        full_val_eval='everyepoch original18*1000, FP32batch128 includingtails',
        checkpoint_rule='max INTEGER full18 validation native correct/18000; earliest exacttie',
        official_arm_rule='maximum validation native correct; arm exacttie A', inference_rule=INFERENCE_RULE,
        selector_forwards_per_query=1, final_completed_solvers_per_query=1,
        parameter_averaging=False, schedulefree_optimizer=False, optimizer_mode_transition=False,
        projector_included=False, scheduler=None, controller=None, early_stopping=False, pcgrad=False,
        cost_dtype='original FP64 metrics/native winners; costs/labels excluded from forward',
        adopted_plan=str(ADOPTED_PLAN.relative_to(WORKSPACE)), adopted_plan_sha256=file_hash(ADOPTED_PLAN),
        plan_basis='actual Astra/max completed R67 TRAIN/VAL review, score8 ready PLAN ONLY; new TEST excluded from planning',
        wandb=dict(project='selector', entity='yjkds-southern-university-of-science-technology', mode='offline'),
        legacy_B_semantics='eight feasible-domain mismatches unresolved/disclosed; R58 fully deferred; not a numerical70 veto',
        assets='weights/NPZ/rawupdates/W&B/regenerable schedules LOCAL; Git alone is not complete inference assets',
        test_read=False, test_authorized=False, automatic_test=False, automatic_analysis=False,
        reference_inference_reruns=False, pilot=False, gpu_capacity_sweep=False, learning_rate_sweep=False)


def prepare(args):
    if (args.root/'formal_started.json').exists() or (args.root/'protocol.json').exists():
        raise ValueError('R68 runs the formal pair once; preserve existing artifacts')
    protocol = planned_config()
    protocol.update(source_hashes=source_hashes(), worktree_identity=worktree_identity())
    for relative in protocol['source_hashes']:
        destination = args.root/'source_launch'/relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(WORKSPACE/relative, destination)
    atomic_json(args.root/'source_launch/hashes.json', protocol['source_hashes'])
    shutil.copy2(ADOPTED_PLAN, args.root/'adopted_R68_plan.response.md')
    inherited_paths = dict(terminal=PRIOR/'B_contrastive_pretrain_seed2/stage1_terminal_x.pt',
        fresh_head_container=PRIOR/'shared_initialization.pt', protocol=PRIOR/'protocol.json',
        transition=PRIOR/'B_contrastive_pretrain_seed2/stage_transition.json')
    inherited_identity = {key: dict(path=str(path), sha256=file_hash(path))
                          for key, path in inherited_paths.items()}
    prior_protocol = json.loads(inherited_paths['protocol'].read_text())
    terminal = torch.load(inherited_paths['terminal'], map_location='cpu', weights_only=False)
    shared = torch.load(inherited_paths['fresh_head_container'], map_location='cpu', weights_only=False)
    params, initial, initialization = starting_state(terminal, shared['stage2_head'],
                                                   prior_protocol['initialization']['hashes']['stage2_head'])
    transition = json.loads(inherited_paths['transition'].read_text())
    if initialization['inherited_trunk_hash'] != transition['terminal_trunk_hash']:
        raise ValueError('R68 trunk must equal actual R67B terminal-x transfer identity')
    del terminal, shared
    if not (args.reference/'geometry_cache/stats.json').is_file():
        raise ValueError('Original existing TRAIN-normalized geometry cache required; no reference writes')
    loaders, raw, geometry = prepare_data(args)
    check_original_scope(loaders, raw)
    if geometry != prior_protocol['geometry_stats']:
        raise ValueError('R68 must retain original R67 TRAIN-only geometry normalization')
    if len(GLOBAL_SOLVERS) != 22 or sum(len(r['pool_ids']) for r in raw['train'].values()) != 128:
        raise ValueError('All22 identities and128 original legal deployments required')
    sizes = {task: loader.size for task, loader in loaders['train'].items()}
    full = paired_triples(sizes, epochs=80, seed=2, batch_size=256)
    full_hash = state_hash(full)
    if full_hash != prior_protocol['schedule_hash']:
        raise ValueError('Regenerated80epoch schedule must match actual R67 exactly')
    schedule = {key: value[40:80] if key in ('permutations', 'cycles', 'triples') else value
                for key, value in full.items()}
    schedule_identity = dict(original80epoch_sha256=full_hash, selected_stage2_sha256=state_hash(schedule),
        global_epochs=[41, 80], zero_based_slice=[40, 80], seed=2, batch_size=256,
        dropout_successful_offset=9600, same_task_row_dropout_order_across_arms=True,
        regeneration=protocol['schedule_regenerate'], saved_row_cache=False)
    del full
    atomic_save(args.root/'shared_initialization.pt', dict(selector=initial))
    atomic_json(args.root/'initialization.json', initialization)
    atomic_json(args.root/'inherited_source_identity.json', inherited_identity)
    atomic_json(args.root/'schedule_seed2.json', schedule_identity)
    protocol.update(model_params=params, initialization=initialization, inherited_identity=inherited_identity,
        geometry_stats=geometry, sizes=sizes, schedule_identity=schedule_identity,
        data_manifest=json.loads((args.root/'data_manifest.json').read_text()))
    protocol = json.loads(json.dumps(protocol))
    atomic_json(args.root/'protocol.json', protocol)
    for arm, name in RUN_NAMES.items():
        directory = args.root/name
        directory.mkdir(exist_ok=True)
        atomic_json(directory/'args.json', dict(protocol, arm=arm))
    atomic_json(args.root/'formal_started.json', dict(started_utc=datetime.now(timezone.utc).isoformat(),
        pid=os.getpid(), arms_in_order=['A', 'B'], test_authorized=False, no_automatic_restart=True))
    print('[R68 prepared] identical inherited trunk/fresh head; exact R67 stage2 plan; next update FORMAL', flush=True)
    return protocol, loaders, raw, schedule, initial


def first_update_handoff(args, launch, metrics):
    pane = subprocess.run(['tmux', 'list-panes', '-t', '=r68_train', '-F', '#{pane_pid}'],
                          capture_output=True, text=True, timeout=20)
    atomic_json(args.root/'handoff.json', dict(status='formal training running',
        command='bash code/V4/run_v4_r68.sh train',
        tmux_start_command='tmux new-session -d -s r68_train "bash code/V4/run_v4_r68.sh train >> code/V4/runs/R68_sharpness_aware_finetuning/pipeline.log 2>&1"',
        tmux='r68_train', tmux_pane_pid=pane.stdout.strip(), slurm_job='1465', slurm_step=launch['slurm_step'],
        host=socket.gethostname(), pid=os.getpid(), parent_pid=os.getppid(), gpu_uuid=GPU_UUID, gpu_index=1,
        arm=metrics['arm'], current_stage='R68 matched full fine-tuning',
        real_batch_per_task=256, task_rows=metrics['task_rows'], effective_batch=metrics['rows'],
        tail_batch_per_task=16, tail_effective_batch=48, first_finite_formal_update=metrics,
        cpu_check=json.loads((args.root/'cpu_check.json').read_text()),
        memory={key: metrics[key] for key in ('peak_memory_gib', 'peak_reserved_gib', 'physical_used_gib', 'physical_total_gib')},
        changed_files=['code/V4/'+name for name in OWNED_SOURCES]+[str(args.root.relative_to(WORKSPACE))+'/'],
        next_normal_monitor_instruction='Every180-300s read current_status.json and small pipeline/launch/history summaries for existing tmux r68_train/exact Slurm step. Let A then B finish. No duplicate launch/pilot/TEST/analysis. After both actual terminal0 and paired_training_complete/validation_candidates, parent handles freeze/one TEST.',
        inherited_pretraining='shared R67B stage1 trunk9600updates/7.2Mrows counted once; no trained head/projector',
        test_authorized=False, no_git_actions=True))


def train_arm(arm, args, data, launch):
    protocol, loaders, raw, schedule, initial = data
    directory = args.root/RUN_NAMES[arm]
    config = dict(protocol, arm=arm)
    model = make_selector(protocol['model_params'], initial, args.device)
    initial_hash = state_hash(model.state_dict())
    if initial_hash != protocol['initialization']['starting_selector_hash']:
        raise ValueError('Both arms require explicit identical complete starting-state copy')
    optimizer = make_optimizer(model)
    scaler = torch.amp.GradScaler('cuda', enabled=True)
    layout = GradientLayout(model)
    atomic_json(directory/'initialization.json', dict(selector_hash=initial_hash, explicit_shared_copy=True,
        inherited_trunk=True, entire_fresh_head=True, projector_included=False,
        all_parameters_trainable=all(p.requires_grad for p in model.parameters()),
        optimizer_fresh=len(optimizer.state) == 0, scaler_fresh=scaler.state_dict(),
        new_successful_updates=0, new_presentations=0, gradient_layout=layout.receipt))
    history, lr_trace, best, best_model = [], [], None, None
    successful = presentations = view_presentations = attempted_views = growth_successes = total_retries = 0
    counts = Counter()
    sam = arm == 'B'
    wandb_run = None
    if args.wandb:
        import wandb
        wandb_run = wandb.init(**protocol['wandb'], name='R68_'+directory.name,
            config=config, dir=str(directory), id='r68-'+arm.lower()+'-'+launch['id'])

    def save_state(epoch, step):
        atomic_save(directory/'last.pt', dict(model=cpu_state(model), optimizer=optimizer.state_dict(),
            scaler=scaler.state_dict(), args=config, epoch=epoch, global_epoch=40+epoch, next_step=step,
            successful_updates=successful, presentations=presentations, successful_view_rows=view_presentations,
            attempted_view_rows=attempted_views, scale_growth_successes=growth_successes,
            amp_retries=total_retries, updates_by_problem=dict(counts), rng=capture_rng(),
            history=history, best=best, lr_trace=lr_trace, parameter_mode=PARAMETER_MODE,
            checkpoint_mode='eval', projector_included=False, perturbed_weights=False))
        atomic_json(directory/'history.json', history)

    try:
        for epoch in range(1, 41):
            model.train()
            torch.cuda.reset_peak_memory_stats(args.device)
            started = time.monotonic()
            totals, task_stats = Counter(), {}
            objective_fields = ['loss', 'ce', 'kl']+(['perturbed_loss', 'perturbed_ce', 'perturbed_kl'] if sam else [])
            for step in range(240):
                entries = triple_indices(schedule, epoch-1, step)
                batches = [gather_batch(loaders['train'][task], indices) for task, indices in entries]
                expected_rows = 256 if step < 234 else 16
                if [len(b['ind']) for b in batches] != [expected_rows]*3:
                    raise ValueError('R68 retains exact real256/task and16/task tails')
                weight = .35*min((successful+1)/720, 1.)
                lr = update_lr(successful)
                view_seed = dropout_seed(9600+successful)
                set_seed(view_seed)
                synchronize(args.device)
                before = time.monotonic()
                values = update_triple(model, optimizer, scaler, batches, weight, layout, sam, lr, growth_successes)
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
                metrics = dict(launch_id=launch['id'], arm=arm, epoch=epoch, global_epoch=40+epoch,
                    stage=2, update_in_epoch=step+1, successful_update=successful,
                    stage_successful_update=successful, schedule_successful_update=9600+successful,
                    presentations=presentations, view_presentations=view_presentations,
                    attempted_view_presentations=attempted_views, tasks=[task for task, _ in entries],
                    task_rows=[len(indices) for _, indices in entries], rows=rows,
                    actual_forward_rows=[len(b['ind']) for b in batches],
                    batch_indices_sha256=[state_hash(indices) for _, indices in entries],
                    dropout_seed=view_seed, lr=lr, consistency_weight=weight,
                    gradient_parameter_mode=PARAMETER_MODE, seconds=seconds, queries_per_second=rows/seconds,
                    peak_memory_gib=torch.cuda.max_memory_allocated(args.device)/2**30,
                    peak_reserved_gib=torch.cuda.max_memory_reserved(args.device)/2**30,
                    physical_used_gib=(total_bytes-free_bytes)/2**30, physical_total_gib=total_bytes/2**30, **values)
                append_jsonl(directory/'updates.jsonl', metrics)
                for key in objective_fields:
                    totals[key] += values[key]*rows
                for key in ('grad_norm', 'first_grad_norm', 'clip_coefficient', 'effective_perturbation_norm', 'retries'):
                    totals[key] += values[key]
                totals['samples'] += rows
                totals['updates'] += 1
                totals['successful_view_rows'] += values['successful_view_rows']
                totals['attempted_view_rows'] += values['attempted_view_rows']
                for index, (task, indices) in enumerate(entries):
                    counts[task] += 1
                    item = task_stats.setdefault(task, dict(rows=0, batches=0, full_grad_norm=0.,
                                                           **{key: 0. for key in objective_fields}))
                    item['rows'] += len(indices)
                    item['batches'] += 1
                    for key in ('loss', 'ce', 'kl'):
                        item[key] += values['task_losses'][index][key]*len(indices)
                        if sam:
                            item['perturbed_'+key] += values['perturbed_task_losses'][index][key]*len(indices)
                    item['full_grad_norm'] += values['full_task_grad_norms'][index]
                if successful == 1:
                    atomic_json(directory/'first_formal_update.json', dict(metrics,
                        selected_indices={task: indices.tolist() for task, indices in entries}, launch=launch))
                    print('[R68 first FORMAL update] '+json.dumps({key: metrics[key] for key in
                        ('arm', 'successful_update', 'tasks', 'task_rows', 'rows', 'lr', 'loss', 'perturbed_loss',
                         'first_grad_norm', 'grad_norm', 'effective_perturbation_norm', 'seconds', 'peak_memory_gib', 'retries', 'finite_gradient')}), flush=True)
                if successful == 1 or (step+1) % 60 == 0:
                    save_state(epoch, step+1)
                    launch.update(status='running', epoch=epoch, global_epoch=40+epoch,
                        successful_updates=successful, presentations=presentations, heartbeat_unix=time.time())
                    atomic_json(directory/'launches'/(launch['id']+'.json'), launch)
                    atomic_json(args.root/'current_status.json', dict(launch, last_metrics=metrics))
                    if arm == 'A' and successful == 1:
                        first_update_handoff(args, launch, metrics)
                    print(f'[R68{arm}] epoch={epoch}/40 update={step+1}/240 total={successful} '
                          f'loss={values["loss"]:.6f} perturbed={values["perturbed_loss"]} '
                          f'lr={lr:.9g} retries={total_retries}', flush=True)
                    if wandb_run is not None:
                        wandb_run.log({key: metrics[key] for key in ('epoch', 'global_epoch', 'successful_update',
                            'loss', 'perturbed_loss', 'lr', 'grad_norm', 'first_grad_norm',
                            'effective_perturbation_norm', 'scale', 'peak_memory_gib', 'queries_per_second')})
            if (totals['samples'] != 180000 or totals['updates'] != 240 or len(task_stats) != 18
                    or successful != epoch*240
                    or any(item['rows'] != 10000 or item['batches'] != 40 for item in task_stats.values())):
                raise ValueError('Every epoch must cover all180000 original TRAIN rows once')
            optimization = {key: totals[key]/totals['samples'] for key in objective_fields}
            optimization.update({key: totals[key]/240 for key in
                                 ('grad_norm', 'first_grad_norm', 'clip_coefficient', 'effective_perturbation_norm')})
            optimization.update(lr=lr, lr_first=lr_trace[(epoch-1)*240], consistency_weight=weight,
                samples=180000, updates=240, amp_retries=int(totals['retries']),
                successful_view_rows=int(totals['successful_view_rows']), attempted_view_rows=int(totals['attempted_view_rows']),
                seconds=time.monotonic()-started, peak_memory_gib=torch.cuda.max_memory_allocated(args.device)/2**30,
                first_pass='native CE/RDrop at unperturbed w', second_pass='native CE/RDrop at w+epsilon' if sam else None)
            for item in task_stats.values():
                for key in objective_fields:
                    item[key] /= item['rows']
                item['full_grad_norm'] /= item['batches']
            model.eval()
            validation = evaluate(model, loaders['val'], raw['val'], 128, directory/f'val_predictions/epoch{epoch:03d}')
            if validation['native_total'] != 18000 or any(r['n'] != 1000 for r in validation['per_problem'].values()):
                raise ValueError('Every validation must include all18000 original rows')
            training = None
            if epoch % 5 == 0:
                training = evaluate(model, loaders['train'], raw['train'], 128, directory/f'train_predictions/epoch{epoch:03d}')
                if training['native_total'] != 180000:
                    raise ValueError('Full eval-mode TRAIN coverage required')
            row = dict(epoch=epoch, global_epoch=40+epoch, stage=2, successful_updates=successful,
                presentations=presentations, successful_view_rows=view_presentations, attempted_view_rows=attempted_views,
                updates_by_problem=dict(counts), optimization=optimization, task_optimization=task_stats,
                train=training, val=validation, parameter_mode=PARAMETER_MODE)
            history.append(row)
            if prefer_checkpoint(validation, None if best is None else best['val']):
                best, best_model = row, cpu_state(model)
                unique = directory/f'best_checkpoints/epoch{epoch:03d}.pt'
                unique.parent.mkdir(exist_ok=True)
                atomic_save(unique, dict(model=best_model, args=config, epoch=epoch-1, global_epoch=40+epoch,
                    native_correct=validation['native_correct'], native_total=18000, macro=validation['macro'],
                    parameter_mode=PARAMETER_MODE, checkpoint_mode='eval', projector_included=False,
                    perturbed_weights=False, inference_rule=INFERENCE_RULE,
                    successful_updates=successful, presentations=presentations, model_state_sha256=state_hash(best_model)))
                shutil.copy2(unique, directory/'best.pt')
                atomic_json(directory/'best_eval.json', best)
            save_state(epoch, 240)
            append_jsonl(args.root/'curves'/(arm+'_metrics.jsonl'), row)
            print_eval(f'R68{arm} val epoch{epoch}', validation['per_problem'], validation['macro'])
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
        if successful != UPDATES or presentations != PRESENTATIONS or len(history) != 40:
            raise ValueError('Each fixed arm must complete9600 updates/7.2M selected rows')
        if view_presentations != protocol['successful_view_rows_per_arm'][arm]:
            raise ValueError('Successful A/B dropout-view budgets must be recorded separately')
        model.load_state_dict(best_model, strict=True)
        model.eval()
        selected_train = evaluate(model, loaders['train'], raw['train'], 128, directory/'train_predictions/best')
        selected_val = evaluate(model, loaders['val'], raw['val'], 128, directory/'val_predictions/best')
        if selected_val['native_correct'] != best['val']['native_correct'] or selected_train['native_total'] != 180000:
            raise ValueError('Selected complete recount must match integer validation selection')
        best_eval = dict(best, train=selected_train, selected_validation=selected_val, selected_validation_count_delta=0)
        atomic_json(directory/'best_eval.json', best_eval)
        atomic_json(directory/'result.json', dict(best=best_eval, final=history[-1], epochs_completed=40,
            successful_updates=successful, presentations=presentations, successful_view_rows=view_presentations,
            attempted_view_rows=attempted_views, amp_retries=total_retries, actual_execution=executed_updates(directory),
            best_checkpoint_sha256=file_hash(directory/'best.pt'),
            selected_epoch_checkpoint=str(directory/f'best_checkpoints/epoch{best["epoch"]:03d}.pt'),
            lr_trace_sha256=state_hash(lr_trace), deployment_parameter_mode=PARAMETER_MODE,
            checkpoint_rule=protocol['checkpoint_rule'], inference_rule=INFERENCE_RULE,
            inherited_pretraining=protocol['initialization'], projector_included=False, perturbed_weights=False,
            fixed_matched_horizon=True, stop_reason='fixed40epoch fine-tuning', test_read=False, test_authorized=False))
    finally:
        if wandb_run is not None:
            wandb_run.finish()


def run(arm, args, data):
    directory = args.root/RUN_NAMES[arm]
    (directory/'launches').mkdir(exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    launch = dict(id=stamp, arm=arm, status='started', started_unix=time.time(), host=socket.gethostname(),
        pid=os.getpid(), gpu_uuid=GPU_UUID, formal=True, tmux_session='r68_train',
        slurm_step=os.environ.get('SLURM_JOB_ID', '')+'.'+os.environ.get('SLURM_STEP_ID', ''))
    started, status = time.monotonic(), 'failed'
    atomic_json(directory/'launches'/(stamp+'.json'), launch)
    with (directory/'train.log').open('a') as stream, redirect_stdout(Tee(sys.stdout, stream)), redirect_stderr(Tee(sys.stderr, stream)):
        try:
            train_arm(arm, args, data, launch)
            status = 'complete'
        finally:
            launch.update(status=status, wall_seconds=time.monotonic()-started, finished_unix=time.time())
            atomic_json(directory/'launches'/(stamp+'.json'), launch)
            atomic_json(directory/'wall_time.json', [launch])
            if status == 'complete':
                result = json.loads((directory/'result.json').read_text())
                result['wall_seconds'] = launch['wall_seconds']
                atomic_json(directory/'result.json', result)


def publish_candidates(root):
    records, lr_hashes = {}, []
    for arm, name in RUN_NAMES.items():
        directory = root/name
        result = json.loads((directory/'result.json').read_text())
        records[arm] = dict(path=str(directory/'best.pt'), sha256=result['best_checkpoint_sha256'],
            selected_epoch_checkpoint=result['selected_epoch_checkpoint'], epoch=result['best']['epoch'],
            global_epoch=result['best']['global_epoch'], native_correct=result['best']['val']['native_correct'],
            native_total=18000, val=result['best']['val']['macro'], parameter_mode=PARAMETER_MODE,
            projector_included=False, perturbed_weights=False)
        lr_hashes.append(result['lr_trace_sha256'])
    if len(set(lr_hashes)) != 1:
        raise ValueError('Both arms require identical successful-update LR traces')
    official = 'A' if records['A']['native_correct'] >= records['B']['native_correct'] else 'B'
    atomic_json(root/'validation_candidates.json', dict(models=records, proposed_official_arm=official,
        inference_rule=INFERENCE_RULE, same_lr_trace_sha256=lr_hashes[0], parent_freeze_pending=True, test_authorized=False))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('cpu-check', 'train'), default='train')
    parser.add_argument('--wandb', action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args(argv)
    args.root, args.reference, args.device, args.batch_size = ROOT, REFERENCE, 'cuda:0', 256
    torch.set_num_threads(1)
    args.root.mkdir(parents=True, exist_ok=True)
    if args.stage == 'cpu-check':
        from .r68_cpu_check import focused_check
        result = focused_check()
        result['source_hashes'] = source_hashes()
        atomic_json(args.root/'cpu_check.json', result)
        print(json.dumps(result), flush=True)
        return
    if (os.environ.get('CUDA_VISIBLE_DEVICES') != GPU_UUID or os.environ.get('SLURM_JOB_ID') != '1465'
            or socket.gethostname().split('.')[0] != 'gpu03'):
        parser.error('R68 requires the assigned existing Slurm1465/gpu03/index1 UUID launcher')
    check = json.loads((args.root/'cpu_check.json').read_text())
    if not check.get('passed') or check['source_hashes'] != source_hashes():
        parser.error('One focused CPU check must pass on these exact launch sources')
    configure_torch()
    try:
        data = prepare(args)
        for arm in ('A', 'B'):
            run(arm, args, data)
            gc.collect()
            torch.cuda.empty_cache()
        publish_candidates(args.root)
        atomic_json(args.root/'paired_training_complete.json', dict(completed_utc=datetime.now(timezone.utc).isoformat(),
            arms=list(RUN_NAMES), epochs_per_arm=40, updates_per_arm=UPDATES, presentations_per_arm=PRESENTATIONS,
            updates_pair=2*UPDATES, presentations_pair=2*PRESENTATIONS,
            successful_view_rows={'A': 14400000, 'B': 28800000},
            inherited_stage1_updates_once=9600, inherited_stage1_presentations_once=7200000,
            parent_freeze_pending=True, test_authorized=False))
        print('[R68 paired training complete] parent freeze pending; no TEST/analysis executed', flush=True)
    except BaseException as error:
        atomic_json(args.root/'terminal_failure.json', dict(recorded_utc=datetime.now(timezone.utc).isoformat(),
            error_type=type(error).__name__, error=str(error), pid=os.getpid(),
            slurm_step=os.environ.get('SLURM_JOB_ID', '')+'.'+os.environ.get('SLURM_STEP_ID', ''),
            automatic_restart=False, samples_skipped=False, duplicate_attempt_authorized=False))
        raise


if __name__ == '__main__':
    main()
