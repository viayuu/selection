"""R73: one formal expansion teacher pass followed by one warm-started student."""

import argparse
import gc
import json
import os
import shutil
import socket
import time
import traceback
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from code.unified_selector.registry import GLOBAL_SOLVERS, P2I, PROBLEMS
from .multitask_probe import file_hash, gather_batch, state_hash
from .performance_experiment import SOURCE_FILES
from .r59_experiment import GPU_UUID, REFERENCE, WORKSPACE, atomic_json, atomic_save
from .r64_experiment import (dropout_seed, evaluate, load_inputs, paired_triples,
                             prefer_checkpoint, synchronize, triple_indices)
from .r64_model import INFERENCE_RULE, NativeUtilitySelector
from .r65_optimizer import (ARM_CONFIGS, configure_update_lr, evaluation_mode,
                            make_optimizer, observed_lr, training_mode)
from .r67_model import cpu_state
from .r72_consensus import INFERENCE_RULE as TEACHER_RULE, MEMBER_NAMES, restore_composite
from .r73_data import ExpansionInputs, expansion_manifest, expansion_order, global_to_local
from .r73_gradients import GradientLayout, update_triple
from .train import configure_torch, print_eval, set_seed
from .training_monitor import capture_rng


ROOT = WORKSPACE / 'code/V4/runs/R73_expansion_teacher_training'
DIRECTORY = ROOT / 'B_teacher_training_seed2'
R67 = WORKSPACE / 'code/V4/runs/R67_supervised_contrastive_full18'
STUDENT = R67 / 'B_contrastive_pretrain_seed2/best.pt'
R72 = WORKSPACE / 'code/V4/runs/R72_heterogeneous_consensus'
TEACHER = R72 / 'consensus_complete.pt'
OWNED_SOURCES = ('r73_data.py', 'r73_gradients.py', 'r73_experiment.py', 'r73_cpu_check.py',
                 'run_v4_r73.sh', 'R73_README.md')
REUSED_SOURCES = ('r72_consensus.py', 'r69_model.py', 'r68_model.py', 'r67_model.py',
    'r65_gradients.py', 'r65_optimizer.py', 'r64_model.py', 'r64_experiment.py',
    'r59_experiment.py', 'r45_experiment.py', 'r43_experiment.py', 'r42_experiment.py',
    'r40_experiment.py', 'pairwise_objective.py', 'pairwise_selector.py', 'solver_code_encoder.py',
    'r65_official_schedulefree/__init__.py', 'r65_official_schedulefree/adamw_schedulefree.py',
    'r65_official_schedulefree/provenance.json', 'r65_official_schedulefree/LICENSE')


def now():
    return datetime.now(timezone.utc).isoformat()


def append(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as stream:
        stream.write(json.dumps(value, allow_nan=False) + '\n')
        stream.flush()


def memory(device):
    free, total = torch.cuda.mem_get_info(device)
    return dict(allocated_bytes=torch.cuda.memory_allocated(device),
        peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
        reserved_bytes=torch.cuda.memory_reserved(device), peak_reserved_bytes=torch.cuda.max_memory_reserved(device),
        physical_used_bytes=total - free, physical_total_bytes=total)


def owned_hashes():
    return {name: file_hash(Path(__file__).parent / name) for name in OWNED_SOURCES}


def status(stage, launch, **fields):
    value = dict(stage=stage, updated_utc=now(), launch=launch, **fields)
    atomic_json(ROOT / 'current_status.json', value)
    atomic_json(ROOT / 'pipeline_status.json', dict(stage=stage, updated_utc=value['updated_utc'],
                                                 pid=os.getpid(), slurm_step=launch['slurm_step']))


def prepare(launch):
    for name in ('protocol.json', 'formal_started.json', 'teacher_cache_complete.json', 'training_complete.json'):
        if (ROOT / name).exists():
            raise ValueError('Preserve original R73 artifacts; no automatic restart: ' + name)
    check = json.loads((ROOT / 'cpu_check.json').read_text())
    if not check['passed'] or check['source_hashes'] != owned_hashes():
        raise ValueError('The one focused CPU invocation must pass on these sources')
    student_args = json.loads((STUDENT.parent / 'args.json').read_text())
    teacher_identity = json.loads((R72 / 'composite_metadata.json').read_text())
    native_schedule = json.loads((R67 / 'schedule_seed2.json').read_text())
    schedule = paired_triples({problem: 10000 for problem in PROBLEMS}, epochs=80, seed=2, batch_size=256)
    if state_hash(schedule) != native_schedule['sha256']:
        raise ValueError('R73 must reuse the exact original global41-80 task/row schedule')
    protocol = dict(experiment='R73_expansion_teacher_training', seed=2,
        approved_plan=str(ROOT / 'approved_plan.md'), approved_plan_sha256=file_hash(ROOT / 'approved_plan.md'),
        scientific_scope='one warm-started hard-teacher-training package; not isolated input-coverage causality',
        new_arms=['B'], reused_control='A: exact complete R72 consensus; saved VAL8923/18000',
        control_native_correct=8923, control_native_total=18000, control_reuse='protocol equivalence, not replication',
        teacher_bundle=teacher_identity, teacher_rule=TEACHER_RULE, teacher_members=list(MEMBER_NAMES),
        teacher_hard_target='pred_global_id; no softmax, confidence filter, balancing or refresh',
        teacher_authorized_inputs='expansion source IDs10000..99999 only; no native TRAIN/VAL/TEST pass',
        teacher_rows=1620000, teacher_batch_size=128, teacher_tail_per_task=16,
        teacher_query_batches=12672, teacher_member_forwards=38016, teacher_member_rows=4860000,
        expansion_inputs=expansion_manifest(), auxiliary_costs_read=False, observation_masks_read=False,
        expansion_rows_per_task=90000, expansion_distinct_rows=1620000,
        expansion_order='one default_rng(730002+problem_id), concatenate five permutations90000, take400000/task',
        expansion_occurrences='all90000 four times;40000 of them a fifth time per task',
        student_checkpoint=str(STUDENT), student_checkpoint_sha256=teacher_identity['member_source_sha256'][MEMBER_NAMES[1]],
        student_initialization='strict complete R67B36/global76 saved averaged-x selector, trained head and all buffers',
        student_shared=True, task_specific_bank=False, new_head=False, projector=False,
        model_params=student_args['model_params'], geometry_stats=student_args['geometry_stats'],
        normalization='unchanged original TRAIN-fitted buffers; no expansion fit', all_selector_parameters_trainable=True,
        optimizer=ARM_CONFIGS['B'], external_warmup=False, optimizer_and_scaler_fresh=True,
        objective='native two-view mean CE + .35*min(u/720,1)*legal symmetric KL + one-view hard-pseudo CE',
        objective_coefficients=dict(native_ce=1., rdrop_max=.35, rdrop_ramp=720, pseudo_ce=1.),
        gradient_rule='backprop/release native graphs, then auxiliary; same-state FP32 task gradient mean/3; clip1 once; step once',
        retry='entire selected triple, same rows and restored ordinary/private RNG; reduce AMP scale;20 attempts',
        native_dropout_seed='r64.dropout_seed(9600+successful_before); original R67 seed2 stage2 schedule',
        auxiliary_dropout_seed='730003+18*(u-1)+problem_id; ordinary CPU/CUDA RNG restored afterward',
        bitwise_historical_mask_equivalence_claim=False,
        schedule_hash=native_schedule['sha256'], schedule_global_epochs=[41, 80],
        schedule_regenerate=native_schedule['regeneration'], batch_size=256, auxiliary_batch_size=256,
        effective_native_batch=768, effective_auxiliary_batch=768, tail_per_task=16,
        effective_native_tail=48, effective_auxiliary_tail=48, tasks_per_update=3,
        epochs=40, updates_per_epoch=240, successful_updates=9600, native_presentations=7200000,
        auxiliary_presentations=7200000, total_presentations=14400000, successful_encoder_view_rows=21600000,
        native_encoder_view_rows=14400000, auxiliary_encoder_view_rows=7200000, native_distinct_train_rows=180000,
        native_validation='full18x1000 every epoch; averaged x, eval FP32, batch128/tails104',
        native_training_evaluation='full18x10000 every5/final/selected; sparse actual measurements only',
        checkpoint_rule='maximum integer native VAL correct/18000; earliest exact tie; one complete shared selector',
        official_arm_rule='B iff selected native_correct strictly>8923, otherwise fixed A',
        inference_rule=INFERENCE_RULE, selector_forwards_per_query=1, completed_solvers_in_deployment=1,
        executed_solvers=0, new_measured_solver_labels=0, capacity_or_compute_matched=False,
        historical_accounting='fixed historical member training separate; shared R67/R70 inherited stage1 counted once',
        inherited_stage1=dict(successful_updates=9600, rows=7200000, wall_seconds=2764.217798, counted_once=True),
        no_native_teacher_reference_or_component_evaluation=True, test_read=False, test_authorized=False,
        test_freeze_authorized=False, automatic_analysis=False, pilot=False, capacity_update=False,
        wandb=dict(project='selector', entity='yjkds-southern-university-of-science-technology', mode='offline'),
        target_native_macro_top1=.70, native_test_threshold=12600, native_test_total=18000,
        goal_status='ACTIVE; unachieved', legacy_B_semantics='eight legacy B inconsistencies deferred/unfixed; R58 deferred',
        new_R72_TEST_excluded_from_design=True, historical_TEST_exposure_disclosed=True)
    paths = [Path(__file__).parent / name for name in sorted(set(SOURCE_FILES + OWNED_SOURCES + REUSED_SOURCES))]
    paths += [WORKSPACE / 'code/unified_selector' / name for name in ('data.py', 'registry.py')]
    protocol['source_hashes'] = {}
    for path in paths:
        relative = path.relative_to(WORKSPACE)
        destination = ROOT / 'source_launch' / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
        protocol['source_hashes'][str(relative)] = file_hash(path)
    shutil.copy2(ROOT / 'approved_plan.md', ROOT / 'source_launch/approved_plan.md')
    atomic_json(ROOT / 'source_launch/hashes.json', protocol['source_hashes'])
    DIRECTORY.mkdir(exist_ok=True)
    atomic_json(ROOT / 'protocol.json', protocol)
    atomic_json(ROOT / 'control_reuse.json', dict(control='R72 consensus', saved_VAL_correct=8923,
        original_total=18000, bundle=teacher_identity, standalone_control_forwards=0,
        component_evaluations=0, historical_test_read_for_this_design=False, fresh_replication=False))
    atomic_json(ROOT / 'schedule_seed2.json', dict(native=native_schedule, used_global_epochs=[41, 80],
        auxiliary=protocol['expansion_order'], auxiliary_rows_per_task=400000, auxiliary_schedule_local=True))
    atomic_json(DIRECTORY / 'args.json', dict(protocol, arm='B'))
    atomic_json(ROOT / 'formal_started.json', dict(launch, formal_recorded_utc=now(),
        phases=['formal_teacher_cache', 'warm_started_student_training'], test_authorized=False,
        automatic_restart=False))
    return protocol, schedule


def teacher_cache(args, protocol, launch, wb):
    started = time.monotonic()
    status('teacher_loading', launch, student_constructed=False, student_successful_updates=0)
    if file_hash(TEACHER) != protocol['teacher_bundle']['sha256']:
        raise ValueError('The complete fixed R72 teacher bundle changed')
    bundle = torch.load(TEACHER, map_location='cpu', weights_only=False)
    teacher = restore_composite(bundle, args.device).float().eval()
    counts = {name: sum(parameter.numel() for parameter in model.parameters())
              for name, model in zip(MEMBER_NAMES, teacher.members)}
    if counts != protocol['teacher_bundle']['parameter_counts'] or any(p.requires_grad for p in teacher.parameters()):
        raise ValueError('All three complete teacher members must be frozen and unchanged')
    atomic_json(ROOT / 'teacher_initialization.json', dict(loaded_utc=now(), checkpoint=str(TEACHER),
        sha256=protocol['teacher_bundle']['sha256'], strict_complete_restore=True, parameter_counts=counts,
        stored_parameters=sum(counts.values()), trainable_parameters=0, optimizer_constructed=False,
        student_parameters_from_required_R67_member=counts[MEMBER_NAMES[1]], student_constructed=False,
        fp32=True, tf32=False, members_eval=True))
    datasets, records = {}, {}
    totals = Counter()
    torch.cuda.reset_peak_memory_stats(args.device)
    for problem in PROBLEMS:
        dataset = ExpansionInputs(problem)
        datasets[problem] = dataset
        directory = ROOT / 'teacher_cache' / problem
        directory.mkdir(parents=True, exist_ok=False)
        targets = np.lib.format.open_memmap(directory / 'teacher_global_ids.npy', mode='w+', dtype=np.int64, shape=(90000,))
        np.save(directory / 'source_indices.npy', np.asarray(dataset.sources), allow_pickle=False)
        task_started, task_batches = time.monotonic(), 0
        for start in range(0, 90000, 128):
            stop = min(start + 128, 90000)
            indices = np.arange(start, stop, dtype=np.int64)
            synchronize(args.device)
            batch_started = time.monotonic()
            inputs = dataset.gather(indices, args.device)
            with torch.no_grad():
                output = teacher(inputs)
                prediction = output['pred_global_id']
                global_to_local(prediction, inputs['pool_ids'], output['solver_mask'])
                targets[start:stop] = prediction.cpu().numpy()
            synchronize(args.device)
            seconds = time.monotonic() - batch_started
            rows = stop - start
            task_batches += 1
            totals.update(query_rows=rows, query_batches=1, member_forwards=3, member_rows=3 * rows,
                          r70_branch_calls=1, tails=int(rows == 16))
            metrics = dict(utc=now(), problem=problem, problem_id=P2I[problem], source_first=10000 + start,
                source_last=10000 + stop - 1, rows=rows, task_batch=task_batches, cumulative=dict(totals),
                seconds=seconds, wall_seconds=time.monotonic() - started, **memory(args.device))
            append(ROOT / 'teacher_batches.jsonl', metrics)
            del inputs, output, prediction
            if totals['query_batches'] == 1:
                targets.flush()
                first = dict(metrics, launch=launch, parameter_counts=counts, teacher_trainable_parameters=0,
                    student_constructed=False, student_successful_updates=0, formal=True,
                    teacher_rule=TEACHER_RULE, native_teacher_forwards=0, executed_solvers=0)
                atomic_json(ROOT / 'first_formal_teacher_batch.json', first)
                atomic_json(ROOT / 'handoff.json', dict(status='formal teacher cache running; student not started',
                    command='bash code/V4/run_v4_r73.sh train', tmux='r73_train', launch=launch,
                    first_formal_teacher_batch=first, cpu_check=json.loads((ROOT / 'cpu_check.json').read_text()),
                    next_phases='finish the one complete expansion cache, release teacher, then fixed40E shared student',
                    test_authorized=False, goal_status='ACTIVE; unachieved'))
                print('[R73 first FORMAL teacher batch] ' + json.dumps(first), flush=True)
            if totals['query_batches'] == 1 or totals['query_batches'] % 32 == 0 or stop == 90000:
                targets.flush()
                status('formal_teacher_cache', launch, last_teacher_batch=metrics,
                       student_constructed=False, student_successful_updates=0)
                print(f'[R73 teacher] {problem} {stop}/90000 total={totals["query_rows"]}/1620000 '
                      f'batches={totals["query_batches"]}/12672', flush=True)
                if wb is not None:
                    wb.log({'phase': 'teacher_cache', **{'teacher/' + key: value for key, value in totals.items()},
                            'teacher/wall_seconds': metrics['wall_seconds'],
                            'teacher/peak_allocated_bytes': metrics['peak_allocated_bytes']})
        targets.flush()
        del targets
        record = dict(dataset.metadata, written_rows=90000, query_batches=task_batches, member_forwards=3 * task_batches,
            member_rows=270000, tail_rows=16, targets='teacher_global_ids.npy', source_ids='source_indices.npy',
            target_kind='hard teacher predictions, not measured solver winners', complete=True,
            wall_seconds=time.monotonic() - task_started)
        if task_batches != 704:
            raise ValueError('Every expansion task must retain all90000 rows and its16-row tail')
        records[problem] = record
        atomic_json(directory / 'metadata.json', record)
    expected = dict(query_rows=1620000, query_batches=12672, member_forwards=38016,
                    member_rows=4860000, r70_branch_calls=12672, tails=18)
    if dict(totals) != expected:
        raise ValueError('The one formal teacher pass must cover the full approved expansion range')
    receipt = dict(completed_utc=now(), complete=True, counters=dict(totals), per_problem=records,
        wall_seconds=time.monotonic() - started, memory=memory(args.device), parameter_counts=counts,
        teacher_bundle=protocol['teacher_bundle'], native_teacher_forwards=0, component_evaluations=0,
        measured_labels_added=0, auxiliary_costs_read=False, observation_masks_read=False, executed_solvers=0)
    atomic_json(ROOT / 'teacher_cache_complete.json', receipt)
    del teacher, bundle
    gc.collect()
    torch.cuda.empty_cache()
    atomic_json(ROOT / 'teacher_released.json', dict(released_utc=now(), before_student_construction=True,
        graphs_retained=False, memory_after_release=memory(args.device)))
    return datasets, receipt


def student_train(args, protocol, schedule, datasets, teacher_receipt, launch, wb):
    preparation_started = time.monotonic()
    status('student_preparation', launch, teacher_cache_complete=True, student_successful_updates=0)
    if not (REFERENCE / 'geometry_cache/stats.json').exists():
        raise ValueError('Original geometry cache missing; R73 never refits or writes historical normalization')
    loaders, raw, geometry = load_inputs(args)
    if geometry != protocol['geometry_stats']:
        raise ValueError('Original TRAIN-fitted geometry statistics changed')
    if file_hash(STUDENT) != protocol['student_checkpoint_sha256']:
        raise ValueError('Required complete R67B36/global76 warm start changed')
    checkpoint = torch.load(STUDENT, map_location='cpu', weights_only=False)
    if (checkpoint['epoch'] != 35 or checkpoint['global_epoch'] != 76
            or checkpoint['parameter_mode'] != 'averaged x' or checkpoint['projector_included']
            or checkpoint['args']['model_params'] != protocol['model_params']):
        raise ValueError('R73 must warm-start the exact complete saved R67B36 averaged-x selector')
    set_seed(2)
    model = NativeUtilitySelector(**protocol['model_params'])
    model.load_state_dict(checkpoint['model'], strict=True)
    model.to(args.device).requires_grad_(True)
    optimizer, scaler = make_optimizer(model, 'B'), torch.amp.GradScaler('cuda', enabled=True)
    initialization = dict(utc=now(), checkpoint=str(STUDENT), sha256=protocol['student_checkpoint_sha256'],
        epoch=36, global_epoch=76, strict_complete_restore=True, trained_head_retained=True,
        all_original_buffers_retained=True, parameter_count=sum(p.numel() for p in model.parameters()),
        trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
        optimizer_fresh=len(optimizer.state) == 0, optimizer_k=[group['k'] for group in optimizer.param_groups],
        scaler_fresh=scaler.state_dict(), teacher_released_before_construction=True,
        initialization_model_state_sha256=state_hash(model.state_dict()), normalization_refitted=False)
    atomic_json(DIRECTORY / 'initialization.json', initialization)
    del checkpoint
    orders, targets = {}, {}
    for problem in PROBLEMS:
        orders[problem] = expansion_order(P2I[problem])
        occurrence = np.bincount(orders[problem], minlength=90000)
        if not (len(orders[problem]) == 400000 and (occurrence == 4).sum() == 50000 and (occurrence == 5).sum() == 40000):
            raise ValueError('Expansion schedule must retain exact four/five exposure counts')
        path = ROOT / 'auxiliary_schedule'
        path.mkdir(exist_ok=True)
        np.save(path / (problem + '.npy'), orders[problem], allow_pickle=False)
        targets[problem] = np.load(ROOT / 'teacher_cache' / problem / 'teacher_global_ids.npy', mmap_mode='r', allow_pickle=False)
        if targets[problem].shape != (90000,) or targets[problem].dtype != np.int64:
            raise ValueError('Only the completed immutable hard-target cache may train the student')
    atomic_json(ROOT / 'auxiliary_schedule.json', dict(seed_rule='730002+problem_id', rows_per_task=400000,
        all_rows_four_times=90000, extra_fifth_occurrences=40000, sha256=state_hash(orders),
        source_id_rule='10000+row_index', local_regenerable_arrays=True))
    layout = GradientLayout(model)
    preparation_seconds = time.monotonic() - preparation_started
    training_started = time.monotonic()
    history, best, best_model, best_path, lr_trace = [], None, None, None, []
    successful = native_rows = auxiliary_rows = growth = retries = 0
    attempted_views, task_counts, cursors, active_counts = Counter(), Counter(), Counter(), Counter()
    gradient_names = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    peak_allocated = peak_reserved = 0

    def save_state(epoch, step):
        was_training = model.training
        evaluation_mode(model, optimizer, 'B')
        atomic_save(DIRECTORY / 'last.pt', dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
            scaler=scaler.state_dict(), args=protocol, epoch=epoch, next_step=step, parameter_mode='averaged x',
            checkpoint_mode='eval', successful_updates=successful, native_presentations=native_rows,
            auxiliary_presentations=auxiliary_rows, successful_encoder_view_rows=2 * native_rows + auxiliary_rows,
            attempted_views=dict(attempted_views), amp_retries=retries, scale_growth_successes=growth,
            task_updates=dict(task_counts), auxiliary_cursors=dict(cursors), active_gradient_counts=dict(active_counts),
            rng=capture_rng(), history=history, best=best, best_model=best_model,
            best_checkpoint=str(best_path) if best_path else None, lr_trace=lr_trace))
        atomic_json(DIRECTORY / 'history.json', history)
        if was_training:
            training_mode(model, optimizer, 'B')

    for epoch in range(1, 41):
        global_epoch = 40 + epoch
        training_mode(model, optimizer, 'B')
        torch.cuda.reset_peak_memory_stats(args.device)
        epoch_started = time.monotonic()
        totals, per_task = Counter(), {}
        for step in range(240):
            entries = triple_indices(schedule, global_epoch - 1, step)
            batches = [gather_batch(loaders['train'][problem], indices) for problem, indices in entries]
            expected = 256 if step < 234 else 16
            if [len(batch['ind']) for batch in batches] != [expected] * 3:
                raise ValueError('Real native256/16 task batches and tails cannot change')
            auxiliary_indices = [orders[problem][cursors[problem]:cursors[problem] + len(indices)]
                                 for problem, indices in entries]

            def auxiliary_batch(task):
                problem = entries[task][0]
                indices = auxiliary_indices[task]
                inputs = datasets[problem].gather(indices, args.device)
                target = torch.from_numpy(np.array(targets[problem][indices], copy=True)).to(args.device)
                return inputs, target

            consistency = .35 * min((successful + 1) / 720, 1.)
            expected_lr = configure_update_lr(optimizer, 'B', successful)
            native_seed = dropout_seed(9600 + successful)
            set_seed(native_seed)
            synchronize(args.device)
            update_started = time.monotonic()
            values = update_triple(model, optimizer, scaler, batches, auxiliary_batch, consistency,
                                   layout, successful + 1, growth)
            synchronize(args.device)
            seconds = time.monotonic() - update_started
            successful += 1
            rows = sum(len(indices) for _, indices in entries)
            native_rows += rows
            auxiliary_rows += rows
            growth, retries = values['scale_growth_successes'], retries + values['retries']
            attempted_views.update(values['attempted'])
            active_counts.update(values.pop('active_gradient_names'))
            lr = observed_lr(optimizer, 'B', expected_lr)
            lr_trace.append(lr)
            for problem, indices in entries:
                task_counts[problem] += 1
                cursors[problem] += len(indices)
            measured_memory = memory(args.device)
            peak_allocated = max(peak_allocated, measured_memory['peak_allocated_bytes'])
            peak_reserved = max(peak_reserved, measured_memory['peak_reserved_bytes'])
            metrics = dict(utc=now(), epoch=epoch, global_epoch=global_epoch, update_in_epoch=step + 1,
                successful_update=successful, tasks=[problem for problem, _ in entries], task_rows=[expected] * 3,
                rows=rows, native_rows=rows, auxiliary_rows=rows, native_presentations=native_rows,
                auxiliary_presentations=auxiliary_rows, total_presentations=native_rows + auxiliary_rows,
                successful_encoder_view_rows=2 * native_rows + auxiliary_rows, cumulative_attempted_views=dict(attempted_views),
                native_indices={problem: indices.tolist() for problem, indices in entries},
                auxiliary_source_ids={problem: (10000 + indices).tolist()
                                      for (problem, _), indices in zip(entries, auxiliary_indices)},
                dropout_seed=native_seed, auxiliary_dropout_seeds=[730003 + 18 * (successful - 1) + P2I[problem]
                                                                  for problem, _ in entries],
                consistency_weight=consistency, lr=lr, optimizer_base_lr=3e-4, parameter_mode='y',
                seconds=seconds, wall_seconds=time.monotonic() - training_started,
                cumulative_retries=retries, **measured_memory, **values)
            append(DIRECTORY / 'updates.jsonl', metrics)
            for key in ('loss', 'native_ce', 'kl', 'pseudo_ce'):
                totals[key] += values[key] * rows
            for key in ('grad_norm', 'clip_coefficient', 'retries'):
                totals[key] += values[key]
            totals['rows'] += rows
            for task, (problem, indices) in enumerate(entries):
                item = per_task.setdefault(problem, dict(rows=0, auxiliary_rows=0, batches=0,
                    loss=0., native_ce=0., kl=0., pseudo_ce=0., grad_norm=0.))
                item['rows'] += len(indices)
                item['auxiliary_rows'] += len(indices)
                item['batches'] += 1
                for key in ('loss', 'native_ce', 'kl', 'pseudo_ce'):
                    item[key] += values['task_losses'][task][key] * len(indices)
                item['grad_norm'] += values['full_task_grad_norms'][task]
            if successful == 1:
                atomic_json(ROOT / 'first_formal_student_update.json', dict(metrics, launch=launch))
                print('[R73 first FORMAL student update] ' + json.dumps(metrics), flush=True)
            if successful == 1 or (step + 1) % 60 == 0:
                save_state(epoch, step + 1)
                status('student_training', launch, last_metrics=metrics, teacher_cache_complete=True,
                       student_constructed=True, student_successful_updates=successful)
                print(f'[R73 B] epoch={epoch}/40 update={step + 1}/240 u={successful} '
                      f'C={values["native_ce"]:.6f} R={values["kl"]:.6f} P={values["pseudo_ce"]:.6f} '
                      f'L={values["loss"]:.6f} retries={retries}', flush=True)
                if wb is not None:
                    wb.log({'phase': 'student_training', **{'update/' + key: metrics[key] for key in
                        ('successful_update', 'loss', 'native_ce', 'kl', 'pseudo_ce', 'grad_norm',
                         'clip_coefficient', 'lr', 'scale', 'cumulative_retries', 'wall_seconds', 'peak_allocated_bytes')}})
            del batches
        if (totals['rows'] != 180000 or successful != epoch * 240 or len(per_task) != 18
                or any(item['rows'] != 10000 or item['auxiliary_rows'] != 10000 or item['batches'] != 40
                       for item in per_task.values())):
            raise ValueError('Each epoch must retain all native rows plus equal-sized expansion presentations')
        optimization = {key: totals[key] / 180000 for key in ('loss', 'native_ce', 'kl', 'pseudo_ce')}
        optimization.update(grad_norm=totals['grad_norm'] / 240, clip_coefficient=totals['clip_coefficient'] / 240,
            retries=int(totals['retries']), lr=lr, lr_first=lr_trace[(epoch - 1) * 240],
            native_rows=180000, auxiliary_rows=180000, successful_encoder_view_rows=540000,
            seconds=time.monotonic() - epoch_started, **memory(args.device))
        for item in per_task.values():
            for key in ('loss', 'native_ce', 'kl', 'pseudo_ce'):
                item[key] /= item['rows']
            item['grad_norm'] /= item['batches']
        evaluation_mode(model, optimizer, 'B')
        with torch.no_grad():
            validation = evaluate(model, loaders['val'], raw['val'], 128, DIRECTORY / f'val_predictions/epoch{epoch:03d}')
            training = evaluate(model, loaders['train'], raw['train'], 128,
                DIRECTORY / f'train_predictions/epoch{epoch:03d}') if epoch % 5 == 0 else None
        if (validation['native_total'] != 18000 or len(validation['per_problem']) != 18
                or any(item['n'] != 1000 for item in validation['per_problem'].values())
                or (training is not None and training['native_total'] != 180000)):
            raise ValueError('Only complete native full18 evaluations can be selected or reported')
        row = dict(epoch=epoch, global_epoch=global_epoch, successful_updates=successful,
            native_presentations=native_rows, auxiliary_presentations=auxiliary_rows,
            successful_encoder_view_rows=2 * native_rows + auxiliary_rows, attempted_views=dict(attempted_views),
            optimization=optimization, task_optimization=per_task, train=training, val=validation,
            updates_by_problem=dict(task_counts), parameter_mode='averaged x')
        history.append(row)
        if prefer_checkpoint(validation, None if best is None else best['val']):
            best, best_model = row, cpu_state(model)
            path = DIRECTORY / 'best_checkpoints'
            path.mkdir(exist_ok=True)
            best_path = path / f'epoch{epoch:03d}.pt'
            atomic_save(best_path, dict(model=best_model, args=protocol, epoch=epoch, global_epoch=global_epoch,
                native_correct=validation['native_correct'], native_total=18000, macro=validation['macro'],
                parameter_mode='averaged x', checkpoint_mode='eval', inference_rule=INFERENCE_RULE,
                successful_updates=successful, native_presentations=native_rows, auxiliary_presentations=auxiliary_rows,
                complete_shared_selector=True, trained_head_retained=True))
            atomic_json(DIRECTORY / 'best_checkpoint.json', dict(path=str(best_path), epoch=epoch,
                global_epoch=global_epoch, native_correct=validation['native_correct'], native_total=18000,
                selection_rule=protocol['checkpoint_rule'], parent_freeze_pending=True))
            atomic_json(DIRECTORY / 'best_eval.json', best)
        save_state(epoch, 240)
        append(ROOT / 'curves/B_metrics.jsonl', row)
        print_eval(f'R73 B native VAL epoch{epoch}', validation['per_problem'], validation['macro'])
        if wb is not None:
            values = {'phase': 'native_evaluation', 'epoch': epoch,
                      **{'optimization/' + key: value for key, value in optimization.items()}}
            for split, record in (('train_eval', training), ('val', validation)):
                if record is not None:
                    values.update({split + '/' + key: value for key, value in record['macro'].items()})
                    values[split + '/native_correct'] = record['native_correct']
                    for problem, item in record['per_problem'].items():
                        values.update({f'{split}/{problem}/{key}': item[key] for key in
                            ('top1', 'top2', 'top3', 'ce', 'mean_cost', 'actual_regret_pct')})
            wb.log(values)
    if (successful != 9600 or native_rows != 7200000 or auxiliary_rows != 7200000 or len(history) != 40
            or set(cursors.values()) != {400000} or set(task_counts.values()) != {1600}):
        raise ValueError('The fixed full40E student budget and both streams must complete exactly')
    evaluation_mode(model, optimizer, 'B')
    atomic_save(DIRECTORY / 'final.pt', dict(model=cpu_state(model), args=protocol, epoch=40, global_epoch=80,
        parameter_mode='averaged x', checkpoint_mode='eval', successful_updates=successful,
        native_presentations=native_rows, auxiliary_presentations=auxiliary_rows, inference_rule=INFERENCE_RULE))
    selected_train = best['train']
    selected_train_source = 'existing complete native TRAIN evaluation at selected epoch'
    if selected_train is None:
        model.load_state_dict(best_model, strict=True)
        model.eval()
        with torch.no_grad():
            selected_train = evaluate(model, loaders['train'], raw['train'], 128, DIRECTORY / 'train_predictions/best')
        selected_train_source = 'one complete native TRAIN evaluation of selected averaged-x state'
    if selected_train['native_total'] != 180000:
        raise ValueError('Selected native TRAIN evaluation must remain complete')
    best_eval = dict(best, train=selected_train, selected_train_source=selected_train_source,
                     selected_validation=best['val'], validation_recount=False)
    atomic_json(DIRECTORY / 'best_eval.json', best_eval)
    atomic_json(DIRECTORY / 'parameter_update_counts.json', dict(
        gradient_present_before_reduction=dict(active_counts),
        optimizer_steps_by_parameter={name: successful for name in gradient_names},
        rule='original R65 shared-selector GradientLayout and official ScheduleFree; all selector parameters trainable'))
    best_identity = dict(path=str(best_path), sha256=file_hash(best_path), epoch=best['epoch'],
        global_epoch=best['global_epoch'], native_correct=best['val']['native_correct'], native_total=18000,
        val=best['val']['macro'], parameter_mode='averaged x', complete_shared_selector=True)
    result = dict(best=best_eval, final=history[-1], best_checkpoint=best_identity,
        epochs_completed=40, global_epochs=[41, 80], successful_updates=successful,
        native_presentations=native_rows, auxiliary_presentations=auxiliary_rows,
        presentations=native_rows + auxiliary_rows, successful_native_encoder_view_rows=2 * native_rows,
        successful_auxiliary_encoder_view_rows=auxiliary_rows, successful_encoder_view_rows=2 * native_rows + auxiliary_rows,
        attempted_views=dict(attempted_views), amp_retries=retries, lr_trace_sha256=state_hash(lr_trace),
        parameter_count=initialization['parameter_count'], trainable_parameters=initialization['trainable_parameters'],
        wall_seconds=time.monotonic() - training_started, student_preparation_seconds=preparation_seconds,
        peak_allocated_bytes=peak_allocated, peak_reserved_bytes=peak_reserved,
        teacher_cache_receipt=str(ROOT / 'teacher_cache_complete.json'), teacher_wall_seconds=teacher_receipt['wall_seconds'],
        checkpoint_rule=protocol['checkpoint_rule'], inference_rule=INFERENCE_RULE,
        deployment_parameter_mode='averaged x', selected_eval_does_not_reselect=True,
        official_arm_rule=protocol['official_arm_rule'], proposed_official_arm='B' if best_identity['native_correct'] > 8923 else 'A',
        stop_reason='fixed40E completed', test_read=False, test_authorized=False, parent_freeze_pending=True)
    atomic_json(DIRECTORY / 'result.json', result)
    atomic_json(ROOT / 'validation_candidates.json', dict(models=dict(B=best_identity),
        fixed_control=dict(A='exact unchanged R72 consensus', native_correct=8923, native_total=18000,
                           bundle=protocol['teacher_bundle']), proposed_official_arm=result['proposed_official_arm'],
        official_arm_rule=protocol['official_arm_rule'], inference_rule=INFERENCE_RULE,
        parent_freeze_pending=True, test_authorized=False))
    atomic_json(ROOT / 'training_complete.json', dict(completed_utc=now(), epochs_completed=40,
        successful_updates=successful, native_presentations=native_rows, auxiliary_presentations=auxiliary_rows,
        successful_encoder_view_rows=2 * native_rows + auxiliary_rows, amp_retries=retries,
        best_checkpoint=best_identity, proposed_official_arm=result['proposed_official_arm'],
        parent_freeze_pending=True, test_authorized=False))
    status('training_complete', launch, student_successful_updates=successful, proposed_official_arm=result['proposed_official_arm'])


def cpu_check():
    for name in ('cpu_check_started.json', 'cpu_check.json', 'formal_started.json'):
        if (ROOT / name).exists():
            raise ValueError('Preserve the original R73 CPU invocation; do not rerun: ' + name)
    command = 'bash code/V4/run_v4_r73.sh cpu-check'
    atomic_json(ROOT / 'cpu_check_started.json', dict(utc=now(), pid=os.getpid(), command=command, invocation=1))
    from .r73_cpu_check import focused_check
    receipt = focused_check()
    receipt.update(completed_utc=now(), command=command, pid=os.getpid(), source_hashes=owned_hashes())
    atomic_json(ROOT / 'cpu_check.json', receipt)
    print('[R73 focused synthetic CPU] ' + json.dumps(receipt), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('cpu-check', 'train'), required=True)
    args = parser.parse_args()
    args.root, args.reference, args.device, args.batch_size = ROOT, REFERENCE, 'cuda:0', 256
    torch.set_num_threads(1)
    wb, launch = None, None
    started = time.monotonic()
    try:
        if args.stage == 'cpu-check':
            cpu_check()
            return
        if (os.environ.get('CUDA_VISIBLE_DEVICES') != GPU_UUID or os.environ.get('SLURM_JOB_ID') != '1465'
                or socket.gethostname().split('.')[0] != 'gpu03'):
            raise ValueError('Use only the assigned existing1465/gpu03/index1 UUID launcher')
        launch = json.loads((ROOT / 'launch_receipt.json').read_text())
        launch.update(trainer_pid=os.getpid(), started_utc=now(),
            slurm_step=os.environ['SLURM_JOB_ID'] + '.' + os.environ['SLURM_STEP_ID'],
            command='bash code/V4/run_v4_r73.sh train', status='running')
        atomic_json(ROOT / 'trainer_launch.json', launch)
        protocol, schedule = prepare(launch)
        import wandb
        wb = wandb.init(**protocol['wandb'], name='R73_B_expansion_teacher_seed2',
                        config=protocol, dir=str(ROOT), id='r73-b-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S'))
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision('highest')
        datasets, teacher_receipt = teacher_cache(args, protocol, launch, wb)
        configure_torch()
        student_train(args, protocol, schedule, datasets, teacher_receipt, launch, wb)
        launch.update(status='complete', completed_utc=now(), total_formal_wall_seconds=time.monotonic() - started)
        atomic_json(ROOT / 'trainer_launch.json', launch)
        print('[R73 complete] fixed teacher cache and40E student complete; parent freeze pending; no TEST', flush=True)
    except BaseException as error:
        failure = dict(recorded_utc=now(), stage=args.stage, pid=os.getpid(), error_type=type(error).__name__,
            error=str(error), traceback=traceback.format_exc(), automatic_restart=False,
            update_failure_receipt=getattr(error, 'r73_update_receipt', None),
            slurm_step=os.environ.get('SLURM_JOB_ID', '') + '.' + os.environ.get('SLURM_STEP_ID', ''))
        append(ROOT / ('cpu_check_failures.jsonl' if args.stage == 'cpu-check' else 'failures.jsonl'), failure)
        if args.stage == 'train':
            atomic_json(ROOT / 'pipeline_status.json', dict(failure, stage='failed_preserved_no_retry', requested_stage=args.stage))
        raise
    finally:
        if wb is not None:
            wb.finish()


if __name__ == '__main__':
    main()
