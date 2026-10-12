"""Exactly one R70 latent-mixup arm; TRAIN/VAL only, fixed R69B20 control."""

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
from .multitask_probe import Tee, file_hash, gather_batch, state_hash
from .performance_experiment import SOURCE_FILES
from .r42_experiment import prepare_data
from .r59_experiment import GPU_UUID, REFERENCE, WORKSPACE, atomic_json, atomic_save, check_original_scope
from .r64_experiment import dropout_seed, evaluate, executed_updates, paired_triples, prefer_checkpoint, synchronize, triple_indices
from .r64_model import INFERENCE_RULE
from .r69_experiment import append_jsonl, planned_config as r69_config, utc_now
from .r69_model import OPTIMIZER_CONFIG, PARAMETER_MODE, cpu_state, make_bank, make_optimizer, update_lr
from .r70_gradients import GradientLayout, update_triple
from .train import configure_torch, print_eval, set_seed
from .training_monitor import capture_rng, restore_rng


ROOT = WORKSPACE / 'code/V4/runs/R70_latent_mixup'
CONTROL_ROOT = WORKSPACE / 'code/V4/runs/R69_task_specific_finetuning'
RUN_NAME = 'B_latent_mixup_seed2'
OWNED_SOURCES = ('r70_gradients.py', 'r70_experiment.py', 'r70_cpu_check.py', 'run_v4_r70.sh', 'R70_README.md')
REUSED_SOURCES = ('r64_model.py', 'r64_experiment.py', 'r64_gradients.py', 'r65_gradients.py',
    'r45_experiment.py', 'r59_experiment.py', 'r43_experiment.py', 'r42_experiment.py',
    'r40_experiment.py', 'pairwise_objective.py', 'pairwise_selector.py', 'solver_code_encoder.py',
    'r67_model.py', 'r67_experiment.py', 'r68_model.py', 'r68_gradient.py',
    'r69_model.py', 'r69_gradients.py', 'r69_experiment.py')
UPDATES, PRESENTATIONS, VIEW_ROWS = 9600, 7200000, 14400000
LOSS_FIELDS = ('loss', 'clean_ce', 'mix_ce', 'kl')


def source_hashes():
    paths = [Path(__file__).parent / name for name in sorted(set(SOURCE_FILES + OWNED_SOURCES + REUSED_SOURCES))]
    paths += [WORKSPACE / 'code/unified_selector' / name for name in ('registry.py', 'data.py')]
    return {str(path.relative_to(WORKSPACE)): file_hash(path) for path in paths}


def planned_config():
    config = r69_config()
    for key in ('adopted_plan', 'reuse_amendment', 'new_R68_test_used_for_design_or_selection', 'expected_shared_parameters'):
        config.pop(key, None)
    config.update(new_arms={'B': RUN_NAME}, reused_control='fixed R69B20/global60/8604 correct',
        experiment='one new clean-plus-latent-mixup arm; protocol-equivalent control reuse, not replication',
        parameter_capacity_matched=True, per_parameter_update_counts_matched=True,
        expected_A_parameters=33173154, expected_bank_parameters=33173154,
        initialization='ORIGINAL PRE-fine-tuning R69 shared_initialization.pt: R67B terminal stage1 averaged-x TRUNK + entire fresh head670003; never B20',
        objective='0.5*meanCEclean + 0.5*meanCEmix + .35*min(u/720,1)*KLclean',
        clean_ce_weight=.5, mixed_ce_weight=.5, mixed_kl=False,
        latent_layer='candidate-aligned encode_pairs output immediately before the entire existing score_head',
        mixup='one row permutation and scalar Uniform(0,1) per task/update; same across both views; same legal mask and solver IDs',
        mixed_targets='lambda*CE(mixed,y)+(1-lambda)*CE(mixed,y[pi]); gather two legal FP32 log probabilities',
        mixed_feature_gradients='through both partners; no detach; targets only enter loss',
        auxiliary_rng='700002+18*(u-1)+problem_id; CPU pi/lambda and auxiliary-head dropout; fork CPU and sole visible CUDA RNG after both clean forwards',
        successful_mixed_head_rows=VIEW_ROWS, extra_head_compute=True,
        successful_clean_head_rows=VIEW_ROWS, encoder_budget_matches_control=True,
        official_arm_rule='B only when best INTEGER full18 VAL correct>8604; otherwise fixed A20',
        augmentation='only this fixed latent mixup; no input augmentation',
        capacity_update='ONE discarded TRAIN-only scheduled worst-padded real256 triple; all bank Adam moments materialized; recreate formal model/optimizer/RNG afterward',
        gpu_sizing='one approved capacity update only; no sweep',
        new_R69_test_used_for_design_or_selection=False,
        inherited_stage1_once=dict(updates=9600, selected_rows=7200000),
        reused_A_training=dict(updates=9600, selected_rows=7200000, wall_seconds=3554.574466, amp_retries=5),
        no_automatic_restart=True, git_actions=False, automatic_freeze=False,
        later_test='parent freezes the common-epoch new B bank and official arm before ONE new B full18 TEST regardless of improvement; A TEST reused unchanged',
        approved_contract='user-provided actual Astra/max TRAIN/VAL-only plan; R70_README.md and approved_plan.md',
        deferred=['R58', 'legacy eight B-domain feasible-semantics mismatches'],
        target_native_macro_top1=.70, goal_status='active; unachieved')
    return config


def pipeline(args, stage, **details):
    receipt = dict(recorded_utc=utc_now(), stage=stage, pid=os.getpid(),
                   slurm_step=os.environ.get('SLURM_JOB_ID', '') + '.' + os.environ.get('SLURM_STEP_ID', ''), **details)
    atomic_json(args.root / 'pipeline_status.json', receipt)
    append_jsonl(args.root / 'pipeline.jsonl', receipt)


def prepare(args):
    if (args.root / 'protocol.json').exists() or (args.root / 'formal_started.json').exists():
        raise ValueError('R70 already prepared/started; preserve artifacts and never replay')
    protocol = planned_config()
    protocol['source_hashes'] = source_hashes()
    for relative in protocol['source_hashes']:
        destination = args.root / 'source_launch' / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(WORKSPACE / relative, destination)
    atomic_json(args.root / 'source_launch/hashes.json', protocol['source_hashes'])
    prior_protocol = json.loads((CONTROL_ROOT / 'protocol.json').read_text())
    initialization = json.loads((CONTROL_ROOT / 'initialization.json').read_text())
    schedule_reference = json.loads((CONTROL_ROOT / 'schedule_seed2.json').read_text())
    source = CONTROL_ROOT / 'shared_initialization.pt'
    initial = torch.load(source, map_location='cpu', weights_only=False)['selector']
    if state_hash(initial) != initialization['starting_selector_hash']:
        raise ValueError('R70 requires the original saved R69 PRE-fine-tuning initialization')
    initialization = dict(initialization, initialization_container=str(source),
        initialized_from_trained_R69_B20=False, explicit_identical_copy_all18=True)
    inherited = json.loads((CONTROL_ROOT / 'inherited_source_identity.json').read_text())
    selected = json.loads((CONTROL_ROOT / 'validation_candidates.json').read_text())['models']['B']
    if (selected['epoch'], selected['global_epoch'], selected['native_correct'], selected['native_total']) != (20, 60, 8604, 18000):
        raise ValueError('Only the fixed complete R69B20 bank is R70 A')
    history_path = CONTROL_ROOT / 'B_task_specific_seed2/history.json'
    history = json.loads(history_path.read_text())
    if len(history) != 40 or history[19]['val']['native_correct'] != 8604:
        raise ValueError('Fixed control identity needs its existing complete40E TRAIN/VAL history')
    shutil.copy2(history_path, args.root / 'reused_A_history.json')
    control = dict(model=selected, history=str(history_path), reused_history=str(args.root / 'reused_A_history.json'),
        protocol=str(CONTROL_ROOT / 'protocol.json'), initialization=str(CONTROL_ROOT / 'initialization.json'),
        schedule=str(CONTROL_ROOT / 'schedule_seed2.json'),
        identity='fixed existing R69 TRAIN/VAL selection; no checkpoint loading, recount, reselection or inference',
        reuse_is_protocol_equivalence_not_independent_replication=True,
        historical_updates=9600, historical_presentations=7200000, historical_successful_view_rows=14400000,
        historical_attempted_view_rows=14405152, historical_wall_seconds=3554.574466, historical_amp_retries=5,
        parameters=33173154, new_compute=0,
        control_test='reuse already-authorized R69B TEST unchanged; never read or execute it in this phase',
        test_read=False, trained_A_weights_used_for_B_initialization=False)
    atomic_json(args.root / 'reused_control.json', control)
    atomic_json(args.root / 'inheritance_budget.json', dict(
        new_B=dict(updates=UPDATES, selected_rows=PRESENTATIONS, encoder_view_rows=VIEW_ROWS,
                   mixed_head_rows=VIEW_ROWS, runtime='measured new formal B only; capacity separate'),
        reused_A=dict(updates=9600, selected_rows=7200000, encoder_view_rows=14400000,
                      wall_seconds=3554.574466, amp_retries=5, accounting='historical; zero new compute'),
        inherited_stage1_once=dict(updates=9600, selected_rows=7200000,
            source='R67B terminal stage1 averaged-x trunk', accounting='historical pretraining counted once separately'),
        parameter_capacity_matched=True, A_parameters=33173154, B_parameters=33173154,
        per_parameter_update_counts_matched=True, additional_head_compute=True, independent_control_replication=False))
    if not (args.reference / 'geometry_cache/stats.json').is_file():
        raise ValueError('Existing original TRAIN-only geometry cache is required; no cache rebuild')
    pipeline(args, 'loading_original_TRAIN_VAL')
    loaders, raw, geometry = prepare_data(args)
    check_original_scope(loaders, raw)
    if geometry != prior_protocol['geometry_stats']:
        raise ValueError('R70 preserves the original R69 TRAIN-only geometry normalization')
    if len(GLOBAL_SOLVERS) != 22 or sum(len(record['pool_ids']) for record in raw['train'].values()) != 128:
        raise ValueError('Original22 global solver IDs and128 legal deployments required')
    sizes = {task: loader.size for task, loader in loaders['train'].items()}
    full = paired_triples(sizes, epochs=80, seed=2, batch_size=256)
    if state_hash(full) != schedule_reference['original80epoch_sha256']:
        raise ValueError('Original R67/R69 80-epoch schedule identity mismatch')
    schedule = {key: value[40:80] if key in ('permutations', 'cycles', 'triples') else value for key, value in full.items()}
    if state_hash(schedule) != schedule_reference['selected_stage2_sha256']:
        raise ValueError('Exact global41-80 task/row schedule identity mismatch')
    del full
    shutil.copy2(source, args.root / 'shared_initialization.pt')
    atomic_json(args.root / 'initialization.json', initialization)
    atomic_json(args.root / 'inherited_source_identity.json', inherited)
    atomic_json(args.root / 'schedule_seed2.json', dict(schedule_reference,
        reference=str(CONTROL_ROOT / 'schedule_seed2.json'), control_new_execution=False))
    protocol.update(model_params=prior_protocol['model_params'], initialization=initialization,
        inherited_identity=inherited, schedule_identity=schedule_reference, reused_control=control,
        geometry_stats=geometry, sizes=sizes,
        data_manifest=json.loads((args.root / 'data_manifest.json').read_text()))
    protocol = json.loads(json.dumps(protocol))
    atomic_json(args.root / 'protocol.json', protocol)
    directory = args.root / RUN_NAME
    directory.mkdir(exist_ok=True)
    atomic_json(directory / 'args.json', dict(protocol, arm='B', model_type='complete_task_specific_bank'))
    return protocol, loaders, raw, schedule, initial


def worst_padded_triple(loaders, schedule):
    chosen, maximum = None, None
    for epoch in range(40):
        for step in range(234):
            entries = triple_indices(schedule, epoch, step)
            nodes = [int(loaders[task].lengths[indices].max()) for task, indices in entries]
            pools = [len(loaders[task].batch['pool_ids']) for task, _ in entries]
            padded = [256 * (n * n + 2 * n * k + k * k) for n, k in zip(nodes, pools)]
            # Sequential tasks retain one task's graphs: peak geometry first, sum breaks ties.
            key = (max(padded), sum(padded), sum(nodes))
            if maximum is None or key > maximum:
                maximum, chosen = key, (epoch, step, entries, nodes, pools, padded)
    epoch, step, entries, nodes, pools, padded = chosen
    return entries, dict(epoch=epoch + 1, global_epoch=epoch + 41, update_in_epoch=step + 1,
        successful_update=epoch * 240 + step + 1, tasks=[task for task, _ in entries], task_rows=[256] * 3,
        padded_nodes=nodes, legal_pool_sizes=pools, padded_attention_slots=padded,
        selection='maximum per-task256*(n^2+2*n*k+k^2), then sum, then total padded nodes; earliest schedule tie',
        inspected_scheduled_full_triples=40 * 234, GPU_forward_size_sweep=False)


def capacity_update(args, data):
    protocol, loaders, _, schedule, initial = data
    entries, selection = worst_padded_triple(loaders['train'], schedule)
    receipt = dict(status='started', started_utc=utc_now(), selection=selection,
        discarded=True, TRAIN_only=True, real_batch_per_task=256, effective_batch=768,
        automatic_restart=False, formal_model_or_optimizer_transfer=False)
    atomic_json(args.root / 'capacity_started.json', receipt)
    pipeline(args, 'one_discarded_capacity_update', selection=selection)
    rng = capture_rng()
    model = optimizer = layout = scaler = batches = None
    started = time.monotonic()
    try:
        model = make_bank(protocol['model_params'], initial, args.device)
        optimizer, layout = make_optimizer(model), GradientLayout(model)
        # Materialize BOTH moments for ALL18 branches, including normally unused paths.
        for parameter in model.parameters():
            optimizer.state[parameter].update(step=torch.zeros((), dtype=torch.float32),
                exp_avg=torch.zeros_like(parameter), exp_avg_sq=torch.zeros_like(parameter))
        storage = sum(value.numel() * value.element_size() for state in optimizer.state.values()
                      for value in state.values() if torch.is_tensor(value) and value.is_cuda)
        expected = 2 * sum(parameter.numel() * parameter.element_size() for parameter in model.parameters())
        if storage != expected or sum(p.numel() for p in model.parameters()) != 33173154:
            raise ValueError('Capacity must include the complete bank and all Adam moments')
        receipt.update(full_bank_parameters=33173154, optimizer_storage='materialized, not merely estimated/reserved',
            optimizer_moment_gpu_bytes=storage, optimizer_moment_gpu_gib=storage / 2**30,
            optimizer_step_scalars='one CPU FP32 zero scalar per parameter',
            all_bank_moments_including_inactive_and_unused=True)
        atomic_json(args.root / 'capacity_started.json', receipt)
        scaler = torch.amp.GradScaler('cuda', enabled=True)
        batches = [gather_batch(loaders['train'][task], indices) for task, indices in entries]
        u = selection['successful_update']
        model.train()
        set_seed(dropout_seed(9600 + u - 1))
        torch.cuda.reset_peak_memory_stats(args.device)
        synchronize(args.device)
        update_started = time.monotonic()
        values = update_triple(model, optimizer, scaler, batches, .35 * min(u / 720, 1.), layout,
                               update_lr(u - 1), successful_update=u)
        synchronize(args.device)
        free, total = torch.cuda.mem_get_info(args.device)
        receipt.update(status='passed', finished_utc=utc_now(), wall_seconds=time.monotonic() - started,
            update_seconds=time.monotonic() - update_started, values=values,
            peak_memory_gib=torch.cuda.max_memory_allocated(args.device) / 2**30,
            peak_reserved_gib=torch.cuda.max_memory_reserved(args.device) / 2**30,
            physical_used_gib=(total - free) / 2**30, physical_total_gib=total / 2**30,
            indices={task: indices.tolist() for task, indices in entries})
        atomic_json(args.root / 'capacity_receipt.json', receipt)
        print('[R70 capacity passed] ' + json.dumps({key: receipt[key] for key in
            ('status', 'selection', 'optimizer_storage', 'optimizer_moment_gpu_bytes', 'peak_memory_gib', 'peak_reserved_gib')}), flush=True)
    except BaseException as error:
        receipt.update(status='failed', finished_utc=utc_now(), wall_seconds=time.monotonic() - started,
                       error_type=type(error).__name__, error=str(error))
        atomic_json(args.root / 'capacity_failure.json', receipt)
        raise
    finally:
        model = optimizer = layout = scaler = batches = None
        gc.collect()
        torch.cuda.empty_cache()
        restore_rng(rng)
    atomic_json(args.root / 'capacity_discarded.json', dict(recorded_utc=utc_now(),
        mutated_model_optimizer_scaler_and_graphs_discarded=True, pre_capacity_rng_restored=True,
        formal_model_recreation_required=True, no_weight_optimizer_or_stream_transfer=True))


def first_update_handoff(args, launch, metrics):
    atomic_json(args.root / 'handoff.json', dict(status='formal B training running',
        command='bash code/V4/run_v4_r70.sh train',
        training_command='python -B -u -m code.V4.r70_experiment --stage train',
        tmux='r70_train', tmux_pane=launch['tmux_pane'], tmux_pane_pid=launch['tmux_pane_pid'],
        tmux_server_host=launch['tmux_server_host'], slurm_job='1465', slurm_step=launch['slurm_step'],
        host=launch['host'], trainer_pid=launch['pid'], parent_pid=os.getppid(), gpu_uuid=GPU_UUID, gpu_index=1,
        arm='B', new_arms=['B'], reused_control='fixed R69B20/global60/8604; no new A compute',
        real_batch_per_task=256, task_rows=metrics['task_rows'], effective_batch=metrics['rows'],
        first_finite_formal_update=metrics,
        parameter_counts=json.loads((args.root / 'parameter_counts.json').read_text()),
        cpu_check=json.loads((args.root / 'cpu_check.json').read_text()),
        capacity=json.loads((args.root / 'capacity_receipt.json').read_text()),
        memory={key: metrics[key] for key in ('peak_memory_gib', 'peak_reserved_gib', 'physical_used_gib', 'physical_total_gib')},
        changed_files=['code/V4/' + name for name in OWNED_SOURCES] + [str(args.root.relative_to(WORKSPACE)) + '/'],
        next_normal_monitor_instruction='Read current_status.json and small pipeline/launch/history summaries every180-300s. Leave fixed40E formal B running; never restart because observation times out. After terminal0/training_complete/validation_candidates, parent freezes one common-epoch B bundle and official arm by VAL, then ONE original full18 new B TEST regardless of improvement. Reuse existing A TEST unchanged. Parent alone publishes ONE completed-version commit.',
        inherited_pretraining='R67B terminal stage1 averaged-x trunk9600updates/7.2Mrows once; entire fresh head670003',
        test_authorized=False, no_git_actions=True, goal_status='active; unachieved'))


def train_arm(args, data, launch):
    protocol, loaders, raw, schedule, initial = data
    directory = args.root / RUN_NAME
    config = dict(protocol, arm='B', model_type='complete_task_specific_bank')
    model = make_bank(protocol['model_params'], initial, args.device)
    branch_hashes = [state_hash(branch.state_dict()) for branch in model.branches]
    if set(branch_hashes) != {protocol['initialization']['starting_selector_hash']}:
        raise ValueError('Formal bank must recreate all18 original PRE-fine-tuning states after capacity disposal')
    parameter_counts = dict(A=33173154, B=sum(p.numel() for p in model.parameters()),
        branch=sum(p.numel() for p in model.branches[0].parameters()), branches=18,
        trainable_B=sum(p.numel() for p in model.parameters() if p.requires_grad), capacity_matched=True,
        per_parameter_update_counts_matched=True, branch_active_updates=1600, rows_per_branch=400000)
    if parameter_counts['branch'] != 1842953 or parameter_counts['B'] != 33173154:
        raise ValueError('R70 keeps the complete native18-branch bank')
    atomic_json(args.root / 'parameter_counts.json', parameter_counts)
    optimizer, layout = make_optimizer(model), GradientLayout(model)
    scaler = torch.amp.GradScaler('cuda', enabled=True)
    atomic_json(directory / 'initialization.json', dict(branch_hashes=branch_hashes,
        explicit_identical_full18=True, inherited_trunk=True, entire_fresh_head=True, projector_included=False,
        all_parameters_trainable=all(p.requires_grad for p in model.parameters()),
        optimizer_fresh=not optimizer.state, scaler_fresh=scaler.state_dict(),
        recreated_after_capacity_discard=True, no_capacity_model_optimizer_transfer=True,
        optimizer_storage='fresh lazy AdamW state; capacity separately materialized all bank moments',
        parameter_counts=parameter_counts, gradient_layout=layout.receipt,
        new_successful_updates=0, new_presentations=0))
    atomic_json(args.root / 'formal_started.json', dict(started_utc=utc_now(), pid=os.getpid(),
        new_arms=['B'], control_reused=True, capacity_discarded=True,
        fresh_optimizer_state_entries=len(optimizer.state), test_authorized=False, no_automatic_restart=True))
    pipeline(args, 'formal_fixed40E', launch=launch)
    history, lr_trace, best, best_model = [], [], None, None
    successful = presentations = view_presentations = attempted_views = mixed_rows = attempted_mixed = growth_successes = total_retries = 0
    counts, task_rows = Counter(), Counter()
    wandb_run = None
    if args.wandb:
        import wandb
        wandb_run = wandb.init(**protocol['wandb'], name='R70_' + RUN_NAME,
            config=config, dir=str(directory), id='r70-b-' + launch['id'])

    def save_state(epoch, step):
        atomic_save(directory / 'last.pt', dict(model=cpu_state(model), optimizer=optimizer.state_dict(),
            scaler=scaler.state_dict(), args=config, epoch=epoch, global_epoch=40 + epoch, next_step=step,
            successful_updates=successful, presentations=presentations, successful_view_rows=view_presentations,
            attempted_view_rows=attempted_views, successful_mixed_head_rows=mixed_rows, attempted_mixed_head_rows=attempted_mixed,
            scale_growth_successes=growth_successes, amp_retries=total_retries,
            updates_by_problem=dict(counts), rows_by_problem=dict(task_rows), rng=capture_rng(), history=history,
            best=best, lr_trace=lr_trace, parameter_mode=PARAMETER_MODE, all18_common_epoch=True,
            checkpoint_mode='eval', projector_included=False))
        atomic_json(directory / 'history.json', history)

    def save_bundle(path, state, epoch, validation):
        atomic_save(path, dict(model=state, args=config, epoch=epoch - 1, global_epoch=40 + epoch,
            common_epoch=epoch, branch_epochs=[epoch] * 18, problem_names=list(PROBLEMS),
            native_correct=validation['native_correct'], native_total=18000, macro=validation['macro'],
            parameter_mode=PARAMETER_MODE, checkpoint_mode='eval', projector_included=False,
            inference_rule=INFERENCE_RULE, successful_updates=epoch * 240,
            presentations=epoch * 180000, model_state_sha256=state_hash(state), all18_complete_branches=True))

    try:
        for epoch in range(1, 41):
            model.train()
            torch.cuda.reset_peak_memory_stats(args.device)
            started = time.monotonic()
            totals, task_stats = Counter(), {}
            for step in range(240):
                entries = triple_indices(schedule, epoch - 1, step)
                batches = [gather_batch(loaders['train'][task], indices) for task, indices in entries]
                expected_rows = 256 if step < 234 else 16
                if [len(batch['ind']) for batch in batches] != [expected_rows] * 3:
                    raise ValueError('R70 preserves real256/task and16/task tails')
                weight, lr = .35 * min((successful + 1) / 720, 1.), update_lr(successful)
                view_seed = dropout_seed(9600 + successful)
                set_seed(view_seed)
                synchronize(args.device)
                before = time.monotonic()
                values = update_triple(model, optimizer, scaler, batches, weight, layout, lr,
                                       successful_update=successful + 1, growth_successes=growth_successes)
                synchronize(args.device)
                seconds = time.monotonic() - before
                growth_successes = values['scale_growth_successes']
                rows = sum(len(indices) for _, indices in entries)
                successful += 1
                presentations += rows
                view_presentations += values['successful_view_rows']
                attempted_views += values['attempted_view_rows']
                mixed_rows += values['successful_mixed_head_rows']
                attempted_mixed += values['attempted_mixed_head_rows']
                total_retries += values['retries']
                lr_trace.append(lr)
                free, total = torch.cuda.mem_get_info(args.device)
                metrics = dict(launch_id=launch['id'], arm='B', epoch=epoch, global_epoch=40 + epoch,
                    stage=2, update_in_epoch=step + 1, successful_update=successful,
                    schedule_successful_update=9600 + successful, presentations=presentations,
                    view_presentations=view_presentations, attempted_view_presentations=attempted_views,
                    mixed_head_presentations=mixed_rows, attempted_mixed_head_presentations=attempted_mixed,
                    tasks=[task for task, _ in entries], task_rows=[len(indices) for _, indices in entries], rows=rows,
                    actual_forward_rows=[len(batch['ind']) for batch in batches],
                    batch_indices_sha256=[state_hash(indices) for _, indices in entries],
                    dropout_seed=view_seed, lr=lr, consistency_weight=weight, gradient_parameter_mode=PARAMETER_MODE,
                    seconds=seconds, queries_per_second=rows / seconds,
                    peak_memory_gib=torch.cuda.max_memory_allocated(args.device) / 2**30,
                    peak_reserved_gib=torch.cuda.max_memory_reserved(args.device) / 2**30,
                    physical_used_gib=(total - free) / 2**30, physical_total_gib=total / 2**30, **values)
                append_jsonl(directory / 'updates.jsonl', metrics)
                for key in LOSS_FIELDS:
                    totals[key] += values[key] * rows
                for key in ('grad_norm', 'clip_coefficient', 'retries'):
                    totals[key] += values[key]
                totals['samples'] += rows
                totals['updates'] += 1
                for key in ('successful_view_rows', 'attempted_view_rows', 'successful_mixed_head_rows', 'attempted_mixed_head_rows'):
                    totals[key] += values[key]
                for index, (task, indices) in enumerate(entries):
                    counts[task] += 1
                    task_rows[task] += len(indices)
                    item = task_stats.setdefault(task, dict(rows=0, batches=0, full_grad_norm=0.,
                                                           **{key: 0. for key in LOSS_FIELDS}))
                    item['rows'] += len(indices)
                    item['batches'] += 1
                    for key in LOSS_FIELDS:
                        item[key] += values['task_losses'][index][key] * len(indices)
                    item['full_grad_norm'] += values['full_task_grad_norms'][index]
                if successful == 1:
                    atomic_json(directory / 'first_formal_update.json', dict(metrics,
                        selected_indices={task: indices.tolist() for task, indices in entries}, launch=launch))
                    print('[R70 first FORMAL update] ' + json.dumps({key: metrics[key] for key in
                        ('successful_update', 'tasks', 'task_rows', 'rows', 'lr', 'clean_ce', 'mix_ce', 'kl',
                         'loss', 'grad_norm', 'seconds', 'peak_memory_gib', 'retries', 'finite_gradient')}), flush=True)
                if successful == 1 or (step + 1) % 60 == 0:
                    save_state(epoch, step + 1)
                    launch.update(status='running', epoch=epoch, global_epoch=40 + epoch,
                        successful_updates=successful, presentations=presentations, heartbeat_unix=time.time())
                    atomic_json(directory / 'launches' / (launch['id'] + '.json'), launch)
                    atomic_json(args.root / 'current_status.json', dict(launch, last_metrics=metrics))
                    if successful == 1:
                        first_update_handoff(args, launch, metrics)
                    print(f'[R70B] epoch={epoch}/40 update={step + 1}/240 total={successful} '
                          f'loss={values["loss"]:.6f} lr={lr:.9g} retries={total_retries}', flush=True)
                if wandb_run is not None:
                    wandb_run.log({key: metrics[key] for key in ('epoch', 'global_epoch', 'successful_update',
                        'loss', 'clean_ce', 'mix_ce', 'kl', 'lr', 'grad_norm', 'scale', 'retries',
                        'peak_memory_gib', 'peak_reserved_gib', 'queries_per_second', 'presentations',
                        'view_presentations', 'mixed_head_presentations')})
            if (totals['samples'] != 180000 or totals['updates'] != 240 or len(task_stats) != 18
                    or successful != epoch * 240
                    or any(item['rows'] != 10000 or item['batches'] != 40 for item in task_stats.values())):
                raise ValueError('Each epoch must cover all180000 original TRAIN rows once')
            optimization = {key: totals[key] / 180000 for key in LOSS_FIELDS}
            optimization.update({key: totals[key] / 240 for key in ('grad_norm', 'clip_coefficient')})
            optimization.update(lr=lr, lr_first=lr_trace[(epoch - 1) * 240], consistency_weight=weight,
                samples=180000, updates=240, amp_retries=int(totals['retries']),
                **{key: int(totals[key]) for key in ('successful_view_rows', 'attempted_view_rows',
                    'successful_mixed_head_rows', 'attempted_mixed_head_rows')},
                seconds=time.monotonic() - started, peak_memory_gib=torch.cuda.max_memory_allocated(args.device) / 2**30)
            for item in task_stats.values():
                for key in LOSS_FIELDS:
                    item[key] /= item['rows']
                item['full_grad_norm'] /= item['batches']
            model.eval()
            validation = evaluate(model, loaders['val'], raw['val'], 128, directory / f'val_predictions/epoch{epoch:03d}')
            if validation['native_total'] != 18000 or any(value['n'] != 1000 for value in validation['per_problem'].values()):
                raise ValueError('Full18 VAL requires all18000 original rows')
            training = None
            if epoch % 5 == 0:
                training = evaluate(model, loaders['train'], raw['train'], 128, directory / f'train_predictions/epoch{epoch:03d}')
                if training['native_total'] != 180000:
                    raise ValueError('Full eval-mode TRAIN coverage required')
            row = dict(epoch=epoch, global_epoch=40 + epoch, stage=2, successful_updates=successful,
                presentations=presentations, successful_view_rows=view_presentations, attempted_view_rows=attempted_views,
                successful_mixed_head_rows=mixed_rows, attempted_mixed_head_rows=attempted_mixed,
                updates_by_problem=dict(counts), rows_by_problem=dict(task_rows), optimization=optimization,
                task_optimization=task_stats, train=training, val=validation, parameter_mode=PARAMETER_MODE,
                all18_common_epoch=True)
            history.append(row)
            if prefer_checkpoint(validation, None if best is None else best['val']):
                best, best_model = row, cpu_state(model)
                unique = directory / f'best_checkpoints/epoch{epoch:03d}.pt'
                unique.parent.mkdir(exist_ok=True)
                save_bundle(unique, best_model, epoch, validation)
                shutil.copy2(unique, directory / 'best.pt')
                atomic_json(directory / 'best_eval.json', best)
            if epoch == 40:
                save_bundle(directory / 'final.pt', cpu_state(model), epoch, validation)
            save_state(epoch, 240)
            append_jsonl(args.root / 'curves/B_metrics.jsonl', row)
            atomic_json(args.root / 'current_status.json', dict(launch, status='running', completed_epochs=epoch,
                successful_updates=successful, presentations=presentations,
                selected_B_epoch=best['epoch'], selected_B_validation_correct=best['val']['native_correct'],
                last_validation_correct=validation['native_correct'], heartbeat_unix=time.time()))
            print_eval(f'R70B val epoch{epoch}', validation['per_problem'], validation['macro'])
            if wandb_run is not None:
                wb = dict(epoch=epoch, global_epoch=40 + epoch,
                          **{'optimization/' + key: value for key, value in optimization.items()})
                for split, value in (('train_eval', training), ('val', validation)):
                    if value is not None:
                        wb.update({split + '/' + key: entry for key, entry in value['macro'].items()})
                        wb[split + '/native_correct'] = value['native_correct']
                        for task, result in value['per_problem'].items():
                            wb.update({f'{split}/{task}/{key}': result[key] for key in
                                       ('top1', 'top2', 'top3', 'ce', 'mean_cost', 'actual_regret_pct')})
                wandb_run.log(wb)
        if (successful != UPDATES or presentations != PRESENTATIONS or view_presentations != VIEW_ROWS
                or mixed_rows != VIEW_ROWS or len(history) != 40 or set(counts.values()) != {1600}
                or set(task_rows.values()) != {400000}):
            raise ValueError('R70 fixed40E budget/coverage mismatch')
        counters = {name: int(optimizer.state[p]['step']) if p in optimizer.state else 0 for name, p in model.named_parameters()}
        atomic_json(directory / 'parameter_step_counts.json', dict(steps=counters, branch_activations=dict(counts),
            explanation='Active-gradient parameters only; unused input paths may remain at zero'))
        model.load_state_dict(best_model, strict=True)
        model.eval()
        selected_train = evaluate(model, loaders['train'], raw['train'], 128, directory / 'train_predictions/best')
        if selected_train['native_total'] != 180000:
            raise ValueError('Selected complete bank TRAIN evaluation requires all180000 rows')
        best_eval = dict(best, train=selected_train)
        atomic_json(directory / 'best_eval.json', best_eval)
        atomic_json(directory / 'result.json', dict(best=best_eval, final=history[-1], epochs_completed=40,
            successful_updates=successful, presentations=presentations, successful_view_rows=view_presentations,
            attempted_view_rows=attempted_views, successful_mixed_head_rows=mixed_rows, attempted_mixed_head_rows=attempted_mixed,
            amp_retries=total_retries, actual_execution=executed_updates(directory),
            updates_by_problem=dict(counts), rows_by_problem=dict(task_rows),
            best_checkpoint_sha256=file_hash(directory / 'best.pt'), final_checkpoint_sha256=file_hash(directory / 'final.pt'),
            selected_epoch_checkpoint=str(directory / f'best_checkpoints/epoch{best["epoch"]:03d}.pt'),
            lr_trace_sha256=state_hash(lr_trace), parameter_counts=parameter_counts,
            deployment_parameter_mode=PARAMETER_MODE, checkpoint_rule=protocol['checkpoint_rule'], inference_rule=INFERENCE_RULE,
            initialization=protocol['initialization'], fixed_horizon=True, stop_reason='fixed40epoch latent mixup fine-tuning',
            all18_common_epoch=True, new_arms=['B'], reused_control=str(args.root / 'reused_control.json'),
            capacity_compute_accounting='one discarded update separately recorded; excluded from formal clock/budget/runtime',
            test_read=False, test_authorized=False))
    finally:
        if wandb_run is not None:
            wandb_run.finish()


def run(args, data):
    directory = args.root / RUN_NAME
    (directory / 'launches').mkdir(exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    launch = dict(id=stamp, arm='B', status='started', started_unix=time.time(), host=socket.gethostname(),
        pid=os.getpid(), gpu_uuid=GPU_UUID, formal=True, tmux_session='r70_train',
        tmux_pane=os.environ.get('R70_TMUX_PANE_ID', ''), tmux_pane_pid=os.environ.get('R70_TMUX_PANE_PID', ''),
        tmux_server_host=os.environ.get('R70_TMUX_SERVER_HOST', ''),
        slurm_step=os.environ.get('SLURM_JOB_ID', '') + '.' + os.environ.get('SLURM_STEP_ID', ''))
    started, status = time.monotonic(), 'failed'
    atomic_json(directory / 'launches' / (stamp + '.json'), launch)
    try:
        train_arm(args, data, launch)
        status = 'complete'
    finally:
        launch.update(status=status, wall_seconds=time.monotonic() - started, finished_unix=time.time())
        atomic_json(directory / 'launches' / (stamp + '.json'), launch)
        atomic_json(directory / 'wall_time.json', [launch])
        atomic_json(args.root / 'current_status.json', launch)
        if status == 'complete':
            result = json.loads((directory / 'result.json').read_text())
            result['wall_seconds'] = launch['wall_seconds']
            atomic_json(directory / 'result.json', result)


def publish_candidate(root):
    result = json.loads((root / RUN_NAME / 'result.json').read_text())
    control = json.loads((root / 'reused_control.json').read_text())['model']
    candidate = dict(path=str(root / RUN_NAME / 'best.pt'), sha256=result['best_checkpoint_sha256'],
        selected_epoch_checkpoint=result['selected_epoch_checkpoint'], epoch=result['best']['epoch'],
        global_epoch=result['best']['global_epoch'], native_correct=result['best']['val']['native_correct'],
        native_total=18000, val=result['best']['val']['macro'], parameter_mode=PARAMETER_MODE,
        model_type='complete_task_specific_bank', all18_common_epoch=True)
    official = 'B' if candidate['native_correct'] > 8604 else 'A'
    atomic_json(root / 'validation_candidates.json', dict(models={'A': control, 'B': candidate},
        proposed_official_arm=official, official_rule='B strictly>8604; tie/lower fixed A20',
        new_arms=['B'], reused_arms=['A'], parent_freeze_pending=True, inference_rule=INFERENCE_RULE,
        test_authorized=False, later_test='parent freezes common-epoch new B bank and official arm before ONE full18 new B TEST regardless of improvement; A TEST reused unchanged'))


def cpu_check(args):
    from .r70_cpu_check import focused_check
    path = args.root / 'cpu_check_invocations.jsonl'
    if path.exists():
        previous = [json.loads(line) for line in path.read_text().splitlines()]
        if any(item['status'] == 'passed' for item in previous):
            raise ValueError('The one focused R70 CPU check already passed; no repeat campaign')
        invocation = 1 + sum(item['status'] == 'started' for item in previous)
    else:
        invocation = 1
    started = time.monotonic()
    receipt = dict(invocation=invocation, type='focused synthetic mixed-gradient/RNG/whole-retry CPU check',
        started_utc=utc_now(), pid=os.getpid(), status='started', gpu_used=False, source_hashes=source_hashes())
    append_jsonl(path, receipt)
    with (args.root / 'cpu_check.log').open('a') as stream, redirect_stdout(Tee(sys.stdout, stream)), redirect_stderr(Tee(sys.stderr, stream)):
        try:
            result = dict(focused_check(), invocation=invocation, started_utc=receipt['started_utc'],
                          wall_seconds=time.monotonic() - started, source_hashes=receipt['source_hashes'])
            atomic_json(args.root / 'cpu_check.json', result)
            append_jsonl(path, dict(receipt, status='passed', wall_seconds=result['wall_seconds']))
            print('[R70 focused CPU check] ' + json.dumps(result), flush=True)
        except BaseException as error:
            append_jsonl(path, dict(receipt, status='failed', wall_seconds=time.monotonic() - started,
                                   error_type=type(error).__name__, error=str(error)))
            raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('cpu-check', 'train'), default='train')
    args = parser.parse_args(argv)
    args.root, args.reference, args.device, args.batch_size, args.wandb = ROOT, REFERENCE, 'cuda:0', 256, True
    torch.set_num_threads(1)
    args.root.mkdir(parents=True, exist_ok=True)
    if args.stage == 'cpu-check':
        cpu_check(args)
        return
    if (os.environ.get('CUDA_VISIBLE_DEVICES') != GPU_UUID or os.environ.get('SLURM_JOB_ID') != '1465'
            or socket.gethostname().split('.')[0] != 'gpu03'):
        parser.error('R70 requires existing Slurm1465/gpu03/index1 assigned-UUID launcher')
    check = json.loads((args.root / 'cpu_check.json').read_text())
    if not check.get('passed') or check['source_hashes'] != source_hashes():
        parser.error('The focused CPU check must pass on these exact launch sources')
    configure_torch()
    set_seed(2)
    with (args.root / 'pipeline.log').open('a') as stream, redirect_stdout(Tee(sys.stdout, stream)), redirect_stderr(Tee(sys.stderr, stream)):
        try:
            data = prepare(args)
            capacity_update(args, data)
            run(args, data)
            publish_candidate(args.root)
            atomic_json(args.root / 'training_complete.json', dict(completed_utc=utc_now(), new_arms=['B'],
                reused_control='fixed R69B20', epochs=40, new_updates=UPDATES, new_presentations=PRESENTATIONS,
                successful_view_rows=VIEW_ROWS, successful_mixed_head_rows=VIEW_ROWS,
                inherited_stage1_updates_once=9600, inherited_stage1_presentations_once=7200000,
                parent_freeze_pending=True, test_authorized=False))
            pipeline(args, 'training_complete_parent_freeze_pending')
            print('[R70 training complete] one new B; parent freeze pending; no TEST/analysis executed', flush=True)
        except BaseException as error:
            failure = dict(recorded_utc=utc_now(), error_type=type(error).__name__, error=str(error), pid=os.getpid(),
                slurm_step=os.environ.get('SLURM_JOB_ID', '') + '.' + os.environ.get('SLURM_STEP_ID', ''),
                automatic_restart=False, samples_skipped=False, duplicate_attempt_authorized=False)
            atomic_json(args.root / 'terminal_failure.json', failure)
            pipeline(args, 'failed', failure=failure)
            raise


if __name__ == '__main__':
    main()
