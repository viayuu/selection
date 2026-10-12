"""R67: one paired native-winner pretraining experiment; original TRAIN/VAL only."""

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
from .r65_optimizer import ARM_CONFIGS, configure_update_lr, evaluation_mode, make_optimizer, observed_lr, training_mode
from .r67_model import PretrainingSelector, cpu_state, shared_initialization, transition_to_stage2
from .r67_objective import GradientLayout, update_triple
from .train import configure_torch, print_eval, set_seed
from .training_monitor import capture_rng


ROOT = WORKSPACE/'code/V4/runs/R67_supervised_contrastive_full18'
ADOPTED_PLAN = WORKSPACE/'.aris/traces/R66_closure_R67_plan.response.md'
RUN_NAMES = {'A': 'A_classification_pretrain_seed2', 'B': 'B_contrastive_pretrain_seed2'}
OWNED_SOURCES = ('r67_model.py', 'r67_objective.py', 'r67_experiment.py', 'r67_cpu_check.py',
                 'run_v4_r67.sh', 'R67_README.md')
REUSED_SOURCES = ('r64_model.py', 'r64_experiment.py', 'r64_gradients.py', 'r65_optimizer.py',
    'r65_gradients.py', 'r45_experiment.py', 'r59_experiment.py', 'r43_experiment.py',
    'r42_experiment.py', 'r40_experiment.py', 'pairwise_objective.py', 'pairwise_selector.py',
    'solver_code_encoder.py', 'r65_official_schedulefree/__init__.py',
    'r65_official_schedulefree/adamw_schedulefree.py', 'r65_official_schedulefree/LICENSE',
    'r65_official_schedulefree/provenance.json')
STAGE_UPDATES, STAGE_PRESENTATIONS = 9600, 7200000


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
    return dict(seed=2, arms=RUN_NAMES, stage_epochs=40, total_epochs_per_arm=80,
        stages_per_arm=2, updates_per_epoch=240, successful_updates_per_stage=STAGE_UPDATES,
        successful_updates_per_arm=19200, successful_updates_pair=38400,
        presentations_per_epoch=180000, presentations_per_stage=STAGE_PRESENTATIONS,
        presentations_per_arm=14400000, presentations_pair=28800000,
        batch_size=256, effective_batch_size=768, tail_per_task=16, final_effective_batch_size=48,
        tasks_per_update=3, contrastive_views_per_task=512, contrastive_tail_views_per_task=32,
        problems=list(PROBLEMS), solver_names=list(GLOBAL_SOLVERS),
        model='fresh R64 NativeUtilitySelector d128, instance4+joint2, heads4, FFN512, score256, dropout.1, handcrafted34+ID',
        optimizer=ARM_CONFIGS['B'], optimizer_rule='official R65B each arm/stage; train y, eval/save averaged x',
        learning_rate=3e-4, warmup_successful_updates=720, external_warmup=False,
        scheduler=None, controller=None, early_stopping=False, pcgrad=False,
        stage1_A='two-forward native CE + legal FP32 symmetric RDrop KL; .35*min(stage_update/720,1)',
        stage1_B='only task-local supervised contrastive Eq2, both dropout views anchors, all same-native-winner positives, self excluded',
        contrastive_temperature=.1, contrastive_fixed_multiplier=.1,
        contrastive_scale_attribution='PROJECT choice equivalent base_temperature1; author scales temperature/base_temperature, default base_temperature.07',
        contrastive_denominator='all nonself views in this actual task batch only',
        contrastive_positive_aggregation='mean positive log probabilities outside log; stable FP32 logsumexp; no detached branch',
        representation='masked legal-candidate mean of384 pair features -> L2 -> projector384/256/ReLU/128 -> L2',
        encode_pairs_calls='once per stage1 dropout view; reuse for native utility and representation',
        projector_seed=670002, stage2_head_seed=670003, explicit_copied_shared_initialization=True,
        stage1_selection='fixed terminal epoch40 averaged-x trunk; no validation evaluation or checkpoint selection',
        stage2='discard projector; identical entire ordinary fresh scalar score_head MLP; reset optimizer/scaler/counters; train ALL selector parameters40E native CE/RDrop',
        paper_adaptations=['dropout-only views of unchanged input', 'fully trainable downstream fine-tuning, not frozen linear classification'],
        stage2_global_epochs=[41, 80], gradient_reduction='FP32 mean of3 same-state task gradients, clip once1, one step',
        retry='whole same triple; restore RNG and rows; reduce AMP scale; advance counters only after success',
        unused_stage1_parameters='A projector/B utility head have grad=None and no optimizer state or decay',
        accumulation=False, microbatching=False, drop_last=False, full_coverage=True,
        full_validation='stage2 every epoch, complete original18000, FP32batch128 and tails',
        train_evaluation='stage2 every5/final/selected, complete original180000 eval x FP32batch128',
        checkpoint_rule='max INTEGER native_correct/18000; earliest exacttie',
        official_arm_rule='selected validation integer count; exacttie A; parent freezes after both complete',
        inference_rule=INFERENCE_RULE, selector_forwards_per_query=1, final_solver_calls_per_query=1,
        deployment='complete NativeUtilitySelector averaged x; projector excluded',
        schedule='one R64 paired_triples80E seed2 real256 task/row plan plus global dropout_seed; shared A/B',
        schedule_cache_saved=False, schedule_regenerate='paired_triples({task:10000 for task in PROBLEMS}, epochs=80, seed=2, batch_size=256)',
        original_data_only=True, original_train_normalized_geometry=True, directed_ATSP_preserved=True,
        no_new_labels=True, no_augmentation=True, no_expert=True, no_source_embedding=True, no_retrieval=True,
        wandb=dict(project='selector', entity='yjkds-southern-university-of-science-technology', mode='offline'),
        adopted_plan=str(ADOPTED_PLAN.relative_to(WORKSPACE)), adopted_plan_sha256=file_hash(ADOPTED_PLAN),
        primary_sources=['https://papers.neurips.cc/paper_files/paper/2020/file/d89a66c7c80a29b1bdbab0f2a1a94af8-Paper.pdf',
                         'https://github.com/HobbitLong/SupContrast/blob/master/losses.py'],
        operational_clarification='R67_README.md and source are the locked numeric operational specification',
        target_native_macro_top1=.70, goal_status='ACTIVE; unachieved', difficulty='nightmare', max_rounds=100,
        legacy_B_semantics='eight legacy B-task feasible-domain semantics unresolved and disclosed; R58 fully deferred',
        test_read=False, test_authorized=False, automatic_test=False, reference_inference_reruns=False,
        automatic_analysis=False, gpu_capacity_sweep=False, learning_rate_sweep=False, pilot=False)


def prepare(args):
    if (args.root/'formal_started.json').exists() or (args.root/'protocol.json').exists():
        raise ValueError('Preserve existing R67 formal artifacts; this launcher runs the pair once')
    protocol = planned_config()
    protocol['source_hashes'] = source_hashes()
    for relative in protocol['source_hashes']:
        destination = args.root/'source_launch'/relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(WORKSPACE/relative, destination)
    atomic_json(args.root/'source_launch/hashes.json', protocol['source_hashes'])
    shutil.copy2(ADOPTED_PLAN, args.root/'adopted_R67_plan.response.md')
    loaders, raw, geometry = prepare_data(args)
    check_original_scope(loaders, raw)
    if len(GLOBAL_SOLVERS) != 22 or sum(len(r['pool_ids']) for r in raw['train'].values()) != 128:
        raise ValueError('R67 requires all22 identities and all128 original legal deployments')
    sizes = {task: loader.size for task, loader in loaders['train'].items()}
    schedule = paired_triples(sizes, epochs=80, seed=2, batch_size=256)
    params, states, initialization = shared_initialization(geometry, args.device)
    atomic_save(args.root/'shared_initialization.pt', states)
    atomic_json(args.root/'initialization.json', initialization)
    protocol.update(model_params=params, geometry_stats=geometry, sizes=sizes,
        initialization=initialization, schedule_hash=state_hash(schedule),
        data_manifest=json.loads((args.root/'data_manifest.json').read_text()))
    protocol = json.loads(json.dumps(protocol))
    atomic_json(args.root/'protocol.json', protocol)
    atomic_json(args.root/'schedule_seed2.json', dict(seed=2, epochs=80, sizes=sizes, batch_size=256,
        sha256=protocol['schedule_hash'], regeneration=protocol['schedule_regenerate'], saved_row_cache=False))
    for arm, name in RUN_NAMES.items():
        directory = args.root/name
        directory.mkdir(exist_ok=True)
        atomic_json(directory/'args.json', dict(protocol, arm=arm))
    atomic_json(args.root/'formal_started.json', dict(started_utc=datetime.now(timezone.utc).isoformat(),
        pid=os.getpid(), arms_in_order=['A_stage1', 'A_stage2', 'B_stage1', 'B_stage2'], test_authorized=False))
    print('[R67 prepared] fresh shared init; real256/task;80E shared schedule; next update FORMAL', flush=True)
    return protocol, loaders, raw, geometry, schedule, states


def first_update_handoff(args, launch, metrics):
    changed = ['code/V4/'+name for name in OWNED_SOURCES]
    changed.append(str(args.root.relative_to(WORKSPACE))+'/')
    atomic_json(args.root/'handoff.json', dict(status='formal training running',
        command='bash code/V4/run_v4_r67.sh train',
        tmux_start_command='tmux new-session -d -s r67_train "bash code/V4/run_v4_r67.sh train >> code/V4/runs/R67_supervised_contrastive_full18/pipeline.log 2>&1"',
        pipeline_log=str(args.root/'pipeline.log'), tmux='r67_train', slurm_job='1465',
        slurm_step=launch['slurm_step'], host=socket.gethostname(), pid=os.getpid(),
        gpu_uuid=GPU_UUID, gpu_index=1, arm=metrics['arm'], current_stage=metrics['stage'],
        real_batch_per_task=256, real_task_rows=metrics['task_rows'], effective_batch=metrics['rows'],
        tail_batch_per_task=16, tail_effective_batch=48, first_finite_formal_update=metrics,
        memory={key: metrics[key] for key in ('peak_memory_gib', 'peak_reserved_gib', 'physical_used_gib', 'physical_total_gib')},
        cpu_check=json.loads((args.root/'cpu_check.json').read_text()), changed_files=changed,
        next_normal_monitor_instruction='Read pipeline.log, handoff.json, current_status.json and latest updates/history/launch files; let A1->A2->B1->B2 finish. No pilot or TEST. After paired_training_complete.json parent freezes validation_candidates.json before TEST.',
        test_authorized=False, no_commit=True))


def train_arm(arm, args, data, launch):
    protocol, loaders, raw, geometry, schedule, states = data
    directory = args.root/RUN_NAMES[arm]
    config = dict(protocol, arm=arm)
    model = PretrainingSelector(protocol['model_params'], states, args.device)
    hashes = dict(selector=state_hash(model.selector.state_dict()), projector=state_hash(model.projector.state_dict()))
    if any(hashes[key] != protocol['initialization']['hashes'][key] for key in hashes):
        raise ValueError('Both R67 arms require explicitly copied identical fresh selector/projector states')
    atomic_json(directory/'initialization.json', dict(hashes=hashes, explicit_shared_copy=True, fresh=True))
    optimizer = make_optimizer(model, 'B')
    scaler = torch.amp.GradScaler('cuda', enabled=True)
    history, best, best_model, stage_records, lr_trace = [], None, None, [], []
    total_retries = 0
    wandb_run = None
    if args.wandb:
        import wandb
        wandb_run = wandb.init(**protocol['wandb'], name='R67_'+directory.name,
                               config=config, dir=str(directory), id='r67-'+arm.lower()+'-'+launch['id'])

    def save_state(stage, epoch, step, successful, presentations, growth, counts):
        was_training = model.training
        evaluation_mode(model, optimizer, 'B')
        atomic_save(directory/'last.pt', dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
            scaler=scaler.state_dict(), args=config, stage=stage, stage_epoch=epoch, next_step=step,
            stage_successful_updates=successful, global_successful_updates=(stage-1)*STAGE_UPDATES+successful,
            stage_presentations=presentations, global_presentations=(stage-1)*STAGE_PRESENTATIONS+presentations,
            checkpoint_mode='eval', parameter_mode='averaged x', projector_included=stage == 1,
            scale_growth_successes=growth, amp_retries=total_retries, updates_by_problem=dict(counts),
            rng=capture_rng(), history=history, best=best, best_model=best_model, lr_trace=lr_trace))
        atomic_json(directory/'history.json', history)
        if was_training:
            training_mode(model, optimizer, 'B')

    try:
        for stage in (1, 2):
            stage_started = time.monotonic()
            successful = presentations = growth_successes = stage_retries = 0
            counts = Counter()
            layout = GradientLayout(model)
            objective_fields = ('loss', 'supcon_raw') if stage == 1 and arm == 'B' else ('loss', 'ce', 'kl')
            for stage_epoch in range(1, 41):
                global_epoch = (stage-1)*40+stage_epoch
                training_mode(model, optimizer, 'B')
                torch.cuda.reset_peak_memory_stats(args.device)
                epoch_started = time.monotonic()
                totals, task_stats = Counter(), {}
                for step in range(240):
                    entries = triple_indices(schedule, global_epoch-1, step)
                    batches = [gather_batch(loaders['train'][task], indices) for task, indices in entries]
                    expected_rows = 256 if step < 234 else 16
                    if [len(b['ind']) for b in batches] != [expected_rows]*3:
                        raise ValueError('Never trim, split, accumulate or drop calibrated task batches')
                    global_before = (stage-1)*STAGE_UPDATES+successful
                    consistency_weight = .35*min((successful+1)/720, 1.)
                    if stage == 1 and arm == 'B':
                        consistency_weight = 0.
                    expected_lr = configure_update_lr(optimizer, 'B', successful)
                    set_seed(dropout_seed(global_before))
                    synchronize(args.device)
                    before = time.monotonic()
                    values = update_triple(model, optimizer, scaler, batches, consistency_weight,
                                           layout, arm, stage, growth_successes)
                    synchronize(args.device)
                    seconds = time.monotonic()-before
                    lr = observed_lr(optimizer, 'B', expected_lr)
                    growth_successes = values['scale_growth_successes']
                    rows = sum(len(indices) for _, indices in entries)
                    successful += 1
                    presentations += rows
                    stage_retries += values['retries']
                    total_retries += values['retries']
                    lr_trace.append(lr)
                    free_bytes, total_bytes = torch.cuda.mem_get_info(args.device)
                    metrics = dict(launch_id=launch['id'], arm=arm, stage=stage, stage_epoch=stage_epoch,
                        global_epoch=global_epoch, update_in_epoch=step+1, stage_successful_update=successful,
                        global_successful_update=(stage-1)*STAGE_UPDATES+successful,
                        stage_presentations=presentations, global_presentations=(stage-1)*STAGE_PRESENTATIONS+presentations,
                        tasks=[task for task, _ in entries], task_rows=[len(indices) for _, indices in entries],
                        actual_forward_rows=[len(batch['ind']) for batch in batches], rows=rows,
                        batch_indices_sha256=[state_hash(indices) for _, indices in entries],
                        dropout_seed=dropout_seed(global_before), lr=lr, optimizer_base_lr=optimizer.param_groups[0]['lr'],
                        consistency_weight=consistency_weight, gradient_parameter_mode='y',
                        seconds=seconds, queries_per_second=rows/seconds,
                        peak_memory_gib=torch.cuda.max_memory_allocated(args.device)/2**30,
                        peak_reserved_gib=torch.cuda.max_memory_reserved(args.device)/2**30,
                        physical_used_gib=(total_bytes-free_bytes)/2**30, physical_total_gib=total_bytes/2**30, **values)
                    append_jsonl(directory/'updates.jsonl', metrics)
                    for key in objective_fields:
                        totals[key] += values[key]*rows
                    for key in ('grad_norm', 'clip_coefficient', 'retries'):
                        totals[key] += values[key]
                    totals['samples'] += rows
                    totals['updates'] += 1
                    for index, (task, indices) in enumerate(entries):
                        counts[task] += 1
                        item = task_stats.setdefault(task, dict(rows=0, batches=0, full_grad_norm=0.,
                                                               **{key: 0. for key in objective_fields}))
                        item['rows'] += len(indices)
                        item['batches'] += 1
                        for key in objective_fields:
                            item[key] += values['task_losses'][index][key]*len(indices)
                        item['full_grad_norm'] += values['full_task_grad_norms'][index]
                    if successful == 1:
                        atomic_json(directory/f'stage{stage}_first_formal_update.json', dict(metrics,
                            selected_indices={task: indices.tolist() for task, indices in entries}, launch=launch))
                        print('[R67 first FORMAL update] '+json.dumps({key: metrics[key] for key in
                            ('arm', 'stage', 'stage_successful_update', 'tasks', 'task_rows', 'rows', 'lr',
                             'loss', 'grad_norm', 'seconds', 'peak_memory_gib', 'retries', 'finite_gradient')}), flush=True)
                    if successful == 1 or (step+1) % 60 == 0:
                        save_state(stage, stage_epoch, step+1, successful, presentations, growth_successes, counts)
                        launch.update(status='running', arm=arm, current_stage=stage,
                                      global_successful_updates=metrics['global_successful_update'], heartbeat_unix=time.time())
                        atomic_json(directory/'launches'/(launch['id']+'.json'), launch)
                        atomic_json(args.root/'current_status.json', dict(launch, last_metrics=metrics))
                    if arm == 'A' and stage == 1 and successful == 1:
                        first_update_handoff(args, launch, metrics)
                    if successful == 1 or (step+1) % 60 == 0:
                        print(f'[R67{arm}] stage={stage} epoch={stage_epoch}/40 global_epoch={global_epoch} '
                              f'update={step+1}/240 stage_total={successful} loss={values["loss"]:.6f} '
                              f'lr={lr:.9g} retries={total_retries}', flush=True)
                        if wandb_run is not None:
                            wandb_run.log({key: metrics[key] for key in ('stage', 'stage_epoch', 'global_epoch',
                                'stage_successful_update', 'global_successful_update', 'loss', 'lr', 'grad_norm',
                                'scale', 'peak_memory_gib', 'queries_per_second')})
                if (totals['samples'] != 180000 or totals['updates'] != 240 or len(task_stats) != 18
                        or successful != stage_epoch*240
                        or any(item['rows'] != 10000 or item['batches'] != 40 for item in task_stats.values())):
                    raise ValueError('Every epoch must present all180000 original rows exactly once')
                optimization = {key: totals[key]/totals['samples'] for key in objective_fields}
                optimization.update({key: totals[key]/240 for key in ('grad_norm', 'clip_coefficient')})
                optimization.update(lr=lr, lr_first=lr_trace[(global_epoch-1)*240], consistency_weight=consistency_weight,
                    samples=180000, updates=240, amp_retries=int(totals['retries']),
                    seconds=time.monotonic()-epoch_started, peak_memory_gib=torch.cuda.max_memory_allocated(args.device)/2**30)
                for item in task_stats.values():
                    for key in objective_fields:
                        item[key] /= item['rows']
                    item['full_grad_norm'] /= item['batches']
                evaluation_mode(model, optimizer, 'B')
                validation = training = None
                if stage == 2:
                    validation = evaluate(model, loaders['val'], raw['val'], 128,
                                          directory/f'val_predictions/epoch{global_epoch:03d}')
                    if validation['native_total'] != 18000 or any(r['n'] != 1000 for r in validation['per_problem'].values()):
                        raise ValueError('Stage2 validation requires all18000 original rows')
                    if stage_epoch % 5 == 0 or stage_epoch == 40:
                        training = evaluate(model, loaders['train'], raw['train'], 128,
                                            directory/f'train_predictions/epoch{global_epoch:03d}')
                        if training['native_total'] != 180000:
                            raise ValueError('Complete eval-mode TRAIN coverage is required')
                row = dict(stage=stage, epoch=stage_epoch, global_epoch=global_epoch,
                    stage_successful_updates=successful, global_successful_updates=(stage-1)*STAGE_UPDATES+successful,
                    stage_presentations=presentations, global_presentations=(stage-1)*STAGE_PRESENTATIONS+presentations,
                    updates_by_problem=dict(counts), optimization=optimization, task_optimization=task_stats,
                    train=training, val=validation, parameter_mode='averaged x')
                history.append(row)
                if stage == 2 and prefer_checkpoint(validation, None if best is None else best['val']):
                    best, best_model = row, cpu_state(model)
                    atomic_save(directory/'best.pt', dict(model=best_model, args=config, epoch=stage_epoch-1,
                        global_epoch=global_epoch, native_correct=validation['native_correct'], native_total=18000,
                        macro=validation['macro'], parameter_mode='averaged x', checkpoint_mode='eval',
                        stage=2, projector_included=False, inference_rule=INFERENCE_RULE,
                        stage_successful_updates=successful, global_successful_updates=9600+successful))
                    atomic_json(directory/'best_eval.json', best)
                save_state(stage, stage_epoch, 240, successful, presentations, growth_successes, counts)
                append_jsonl(args.root/'curves'/(arm+'_metrics.jsonl'), row)
                if validation is not None:
                    print_eval(f'R67{arm} stage2 val epoch{stage_epoch}', validation['per_problem'], validation['macro'])
                if wandb_run is not None:
                    wb = dict(stage=stage, stage_epoch=stage_epoch, global_epoch=global_epoch,
                              **{'optimization/'+key: value for key, value in optimization.items()})
                    for split, value in (('train_eval', training), ('val', validation)):
                        if value is not None:
                            wb.update({split+'/'+key: entry for key, entry in value['macro'].items()})
                            wb[split+'/native_correct'] = value['native_correct']
                            for task, result in value['per_problem'].items():
                                wb.update({f'{split}/{task}/{key}': result[key] for key in
                                           ('top1', 'top2', 'top3', 'ce', 'mean_cost', 'actual_regret_pct')})
                    wandb_run.log(wb)
            if successful != STAGE_UPDATES or presentations != STAGE_PRESENTATIONS:
                raise ValueError('Each fixed stage requires9600 successful updates and7.2M presentations')
            stage_record = dict(stage=stage, epochs_completed=40, successful_updates=successful,
                presentations=presentations, amp_retries=stage_retries, wall_seconds=time.monotonic()-stage_started,
                lr_trace_sha256=state_hash(lr_trace[(stage-1)*9600:stage*9600]), final=history[-1])
            stage_records.append(stage_record)
            atomic_json(directory/f'stage{stage}_result.json', stage_record)
            if stage == 1:
                atomic_save(directory/'stage1_terminal_x.pt', dict(model=cpu_state(model.selector),
                    args=config, stage=1, epoch=40, parameter_mode='averaged x', selection='fixed terminal',
                    successful_updates=successful, presentations=presentations))
                old_model, old_optimizer = model, optimizer
                model, optimizer, scaler, transition = transition_to_stage2(old_model, old_optimizer,
                                                                           states['stage2_head'], amp=True)
                if transition['stage2_head_hash'] != protocol['initialization']['hashes']['stage2_head']:
                    raise ValueError('Stage2 must copy the identical whole fresh head across arms')
                atomic_json(directory/'stage_transition.json', dict(transition,
                    stage1_successful_updates=successful, global_successful_updates=9600,
                    old_optimizer_k=[g['k'] for g in old_optimizer.param_groups]))
                print('[R67 stage transition] '+json.dumps(transition), flush=True)
                del old_model, old_optimizer, layout
                gc.collect()
                torch.cuda.empty_cache()
        if best is None or len(history) != 80 or len(lr_trace) != 19200:
            raise ValueError('Both fixed40epoch stages must finish before selected evaluation')
        model.load_state_dict(best_model, strict=True)
        model.eval()
        selected_train = evaluate(model, loaders['train'], raw['train'], 128, directory/'train_predictions/best')
        selected_val = evaluate(model, loaders['val'], raw['val'], 128, directory/'val_predictions/best')
        if selected_val['native_correct'] != best['val']['native_correct'] or selected_train['native_total'] != 180000:
            raise ValueError('Selected fixed-batch full-split recount must match selection and coverage')
        best_eval = dict(best, train=selected_train, selected_validation=selected_val, selected_validation_count_delta=0)
        atomic_json(directory/'best_eval.json', best_eval)
        atomic_json(directory/'result.json', dict(best=best_eval, final=history[-1], stages=stage_records,
            epochs_completed=80, successful_updates=19200, presentations=14400000,
            amp_retries=total_retries, actual_execution=executed_updates(directory),
            best_checkpoint_sha256=file_hash(directory/'best.pt'), lr_trace_sha256=state_hash(lr_trace),
            checkpoint_rule=protocol['checkpoint_rule'], inference_rule=INFERENCE_RULE,
            deployment_parameter_mode='averaged x', projector_included=False,
            selected_eval_does_not_reselect=True, fixed_matched_horizon=True,
            stop_reason='fixed40epoch pretraining +40epoch full fine-tuning', test_read=False, test_authorized=False))
    finally:
        if wandb_run is not None:
            wandb_run.finish()


def run(arm, args, data):
    directory = args.root/RUN_NAMES[arm]
    (directory/'launches').mkdir(exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    launch = dict(id=stamp, arm=arm, status='started', started_unix=time.time(), host=socket.gethostname(),
        pid=os.getpid(), gpu_uuid=GPU_UUID, formal=True, tmux_session='r67_train',
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
            epoch=result['best']['epoch'], global_epoch=result['best']['global_epoch'],
            native_correct=result['best']['val']['native_correct'], native_total=18000,
            val=result['best']['val']['macro'], parameter_mode='averaged x', projector_included=False)
        lr_hashes.append(result['lr_trace_sha256'])
    if len(set(lr_hashes)) != 1:
        raise ValueError('Both arms must use the identical per-stage warmup/LR trace')
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
        from .r67_cpu_check import focused_check
        result = focused_check()
        result['source_hashes'] = source_hashes()
        atomic_json(args.root/'cpu_check.json', result)
        print(json.dumps(result), flush=True)
        return
    if (os.environ.get('CUDA_VISIBLE_DEVICES') != GPU_UUID or os.environ.get('SLURM_JOB_ID') != '1465'
            or socket.gethostname().split('.')[0] != 'gpu03'):
        parser.error('R67 requires the assigned existing Slurm1465/gpu03/index1 UUID launcher')
    check = json.loads((args.root/'cpu_check.json').read_text())
    if not check.get('passed') or check['source_hashes'] != source_hashes():
        parser.error('The one focused CPU check must pass on these exact sources')
    configure_torch()
    data = prepare(args)
    for arm in ('A', 'B'):
        run(arm, args, data)
        gc.collect()
        torch.cuda.empty_cache()
    publish_candidates(args.root)
    atomic_json(args.root/'paired_training_complete.json', dict(completed_utc=datetime.now(timezone.utc).isoformat(),
        arms=list(RUN_NAMES), epochs_per_arm=80, updates_per_arm=19200, presentations_per_arm=14400000,
        updates_pair=38400, presentations_pair=28800000, parent_freeze_pending=True, test_authorized=False))
    print('[R67 paired training complete] parent validation freeze pending; no TEST executed', flush=True)


if __name__ == '__main__':
    main()
