"""Conditionally authorized R73 shared-selector lock and ONE original native TEST."""

import argparse
import csv
import json
import os
import shlex
import shutil
import socket
import subprocess
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path


WORKSPACE = Path(__file__).resolve().parents[2]
ROOT = WORKSPACE / 'code/V4/runs/R73_expansion_teacher_training'
RUN = ROOT / 'B_teacher_training_seed2'
CONTROL = WORKSPACE / 'code/V4/runs/R72_heterogeneous_consensus'
GPU_UUID = 'GPU-cb626bdb-4319-2f5e-b91f-17fcaa44a83d'
DELIVERY_SOURCES = ('r73_frozen_evaluation.py', 'r73_delivery.py', 'r73_analysis.py',
                    'run_v4_r73_test.sh', 'run_v4_r73_delivery.sh')


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    os.replace(temporary, path)


def exclusive(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def append(path, value):
    with Path(path).open('a') as stream:
        stream.write(json.dumps(value, allow_nan=False) + '\n')
        stream.flush()


def capture(command):
    started, stamp = time.monotonic(), now()
    result = subprocess.run(command, text=True, capture_output=True)
    return dict(command=shlex.join(command), recorded_utc=stamp, finished_utc=now(),
        seconds=time.monotonic() - started, returncode=result.returncode, stdout=result.stdout, stderr=result.stderr)


def original_identity():
    handoff, launch = read(ROOT / 'execution_handoff.json'), read(ROOT / 'trainer_launch.json')
    if (handoff['slurm_step'] != launch['slurm_step'] or handoff['trainer_pid'] != launch['trainer_pid']
            or handoff['tmux_pane'] != launch['tmux_pane'] or str(handoff['tmux_pane_pid']) != launch['tmux_pane_pid']
            or launch['gpu_uuid'] != GPU_UUID or launch['gpu_index'] != 1 or launch['host'].split('.')[0] != 'gpu03'
            or launch['tmux_session'] != 'r73_train' or not launch['slurm_step'].startswith('1465.')):
        raise ValueError('Use only the original recorded R73 cache-then-training process')
    return launch


def delivery_hashes():
    from .multitask_probe import file_hash
    paths = [WORKSPACE / 'code/V4' / name for name in DELIVERY_SOURCES] + [ROOT / 'DELIVERY_PROTOCOL.md']
    return {str(path.relative_to(WORKSPACE)): file_hash(path) for path in paths}


def require_training_terminal():
    from .r69_frozen_evaluation import require_scheduler_terminal
    launch = original_identity()
    receipt = read(ROOT / 'training_terminal_receipt.json')
    require_scheduler_terminal(receipt, 'r73_train', launch['slurm_step'])
    if launch['status'] != 'complete':
        raise ValueError('Require original process published completion as well as actual scheduler terminal0')
    return receipt


def completed_training():
    from code.unified_selector.registry import PROBLEMS
    from .r64_model import INFERENCE_RULE
    from .r69_frozen_evaluation import require_evaluation

    terminal = require_training_terminal()
    protocol, complete = read(ROOT / 'protocol.json'), read(ROOT / 'training_complete.json')
    teacher, released = read(ROOT / 'teacher_cache_complete.json'), read(ROOT / 'teacher_released.json')
    initial = read(RUN / 'initialization.json')
    history, result, candidates = read(RUN / 'history.json'), read(RUN / 'result.json'), read(ROOT / 'validation_candidates.json')
    expected_teacher = dict(query_rows=1620000, query_batches=12672, member_forwards=38016,
                            member_rows=4860000, r70_branch_calls=12672, tails=18)
    if (not teacher['complete'] or teacher['counters'] != expected_teacher
            or set(teacher['per_problem']) != set(PROBLEMS) or teacher['teacher_bundle'] != protocol['teacher_bundle']
            or teacher['native_teacher_forwards'] != 0 or teacher['component_evaluations'] != 0
            or teacher['measured_labels_added'] != 0 or teacher['executed_solvers'] != 0
            or teacher['auxiliary_costs_read'] or teacher['observation_masks_read']
            or not released['before_student_construction'] or released['graphs_retained']
            or datetime.fromisoformat(released['released_utc']) >= datetime.fromisoformat(initial['utc'])
            or not initial['teacher_released_before_construction'] or not initial['strict_complete_restore']
            or not initial['trained_head_retained'] or not initial['all_original_buffers_retained']
            or initial['normalization_refitted'] or not initial['optimizer_fresh']
            or initial['optimizer_k'] != [0] or (initial['epoch'], initial['global_epoch']) != (36, 76)
            or initial['sha256'] != protocol['student_checkpoint_sha256']
            or (initial['parameter_count'], initial['trainable_parameters']) != (1842953, 1842953)):
        raise ValueError('Require the completed single input-only teacher cache and released teacher before strict warm start')
    for problem, record in teacher['per_problem'].items():
        if (not record['complete'] or (record['count'], record['written_rows'], record['source_first'], record['source_last'])
                != (90000, 90000, 10000, 99999)
                or (record['query_batches'], record['member_forwards'], record['member_rows'], record['tail_rows'])
                != (704, 2112, 270000, 16)
                or record['pool'] != protocol['expansion_inputs'][problem]['pool']
                or record['auxiliary_costs_opened'] or record['observation_masks_opened']
                or not (ROOT / 'teacher_cache' / problem / record['targets']).is_file()
                or not (ROOT / 'teacher_cache' / problem / record['source_ids']).is_file()):
            raise ValueError('Every task must publish all90000 aligned expansion hard targets exactly once')
    if (protocol['new_arms'] != ['B'] or protocol['control_native_correct'] != 8923
            or protocol['inference_rule'] != INFERENCE_RULE or protocol['task_specific_bank']
            or protocol['new_head'] or protocol['projector'] or not protocol['student_shared']
            or read(ROOT / 'source_launch/hashes.json') != protocol['source_hashes']
            or len(history) != 40 or [row['epoch'] for row in history] != list(range(1, 41))
            or [row['global_epoch'] for row in history] != list(range(41, 81))
            or (complete['epochs_completed'], complete['successful_updates'], complete['native_presentations'],
                complete['auxiliary_presentations'], complete['successful_encoder_view_rows']) != (40, 9600, 7200000, 7200000, 21600000)
            or (result['epochs_completed'], result['successful_updates'], result['native_presentations'],
                result['auxiliary_presentations'], result['presentations'], result['successful_encoder_view_rows'])
                != (40, 9600, 7200000, 7200000, 14400000, 21600000)
            or result['successful_native_encoder_view_rows'] != 14400000
            or result['successful_auxiliary_encoder_view_rows'] != 7200000
            or result['amp_retries'] != complete['amp_retries']
            or result['deployment_parameter_mode'] != 'averaged x' or result['final'] != history[-1]
            or result['parameter_count'] != 1842953 or result['trainable_parameters'] != 1842953
            or result['attempted_views']['native_view_rows'] < 14400000
            or result['attempted_views']['auxiliary_view_rows'] < 7200000
            or not (RUN / 'last.pt').is_file()):
        raise ValueError('Require original completed40E/9600updates and both exact supervised stream budgets')
    for row in history:
        require_evaluation(row['val'], 1000)
        if (row['successful_updates'] != 240 * row['epoch'] or row['native_presentations'] != 180000 * row['epoch']
                or row['auxiliary_presentations'] != row['native_presentations']
                or row['successful_encoder_view_rows'] != 3 * row['native_presentations']
                or row['parameter_mode'] != 'averaged x' or set(row['updates_by_problem']) != set(PROBLEMS)
                or set(row['updates_by_problem'].values()) != {40 * row['epoch']}):
            raise ValueError('Every recorded native common epoch must preserve both streams and averaged-x evaluation')
        if row['epoch'] % 5 == 0:
            require_evaluation(row['train'], 10000)
        elif row.get('train') is not None:
            raise ValueError('Native TRAIN history must retain its actual sparse measurements')
    selected = max(history, key=lambda row: row['val']['native_correct'])
    candidate = candidates['models']['B']
    expected_path = RUN / f'best_checkpoints/epoch{selected["epoch"]:03d}.pt'
    official = 'B' if selected['val']['native_correct'] > 8923 else 'A'
    if (set(candidates['models']) != {'B'} or candidate != result['best_checkpoint'] or candidate != complete['best_checkpoint']
            or candidate['path'] != str(expected_path) or candidate['epoch'] != selected['epoch']
            or candidate['global_epoch'] != selected['global_epoch'] or candidate['native_total'] != 18000
            or candidate['native_correct'] != selected['val']['native_correct'] or candidate['val'] != selected['val']['macro']
            or candidate['parameter_mode'] != 'averaged x' or not candidate['complete_shared_selector']
            or result['best']['epoch'] != selected['epoch'] or result['best']['val'] != selected['val']
            or result['best']['selected_validation'] != selected['val']
            or any(value != official for value in (candidates['proposed_official_arm'], result['proposed_official_arm'],
                                                    complete['proposed_official_arm']))):
        raise ValueError('Select only earliest max INTEGER native full18 VAL and the unique complete shared B state')
    require_evaluation(result['best']['train'], 10000)
    return protocol, candidate, official, terminal


def reused_control():
    from .r69_frozen_evaluation import require_evaluation
    saved, original_lock = read(CONTROL / 'test_results.json'), read(CONTROL / 'final_test_lock.json')
    record, value = original_lock['models']['B'], saved['models']['B']
    if (not saved['complete'] or saved['lock'] != original_lock or original_lock['official_arm'] != 'B'
            or record['native_correct'] != 8923 or record['native_total'] != 18000
            or record['model_class'] != 'FrozenConsensus' or not record['all_three_complete_states_and_buffers']
            or not value['complete'] or value['evaluation_passes'] != 1
            or value['evaluated_checkpoint_sha256'] != record['inference_sha256']):
        raise ValueError('Reuse exactly R72 test_results.json#/models/B unchanged; no control computation')
    require_evaluation(value, 1000)
    provenance = dict(label='R73 A = unchanged R72 fixed full3 consensus', test_path=str(CONTROL / 'test_results.json'),
        test_json_pointer='/models/B', lock_path=str(CONTROL / 'final_test_lock.json'), lock_json_pointer='/models/B',
        validation_path=str(CONTROL / 'validation_results.json'), original_tests_already_executed=1, new_R73_tests=0,
        new_checkpoint_load=False, inference=False, prediction_recount=False, reselection=False,
        original_object_unchanged=True, historical_test_exposure=True, protocol_equivalence_not_replication=True)
    return record, value, provenance


def read_checkpoint(record, protocol):
    import torch
    path = Path(record['path'])
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    state = checkpoint['model']
    # Checkpoint tuples (optimizer betas) become lists in the JSON protocol.
    checkpoint_args = json.loads(json.dumps(checkpoint['args'], allow_nan=False))
    if (checkpoint_args != protocol or checkpoint['epoch'] != record['epoch']
            or checkpoint['global_epoch'] != record['global_epoch'] or checkpoint['global_epoch'] != 40 + checkpoint['epoch']
            or checkpoint['parameter_mode'] != 'averaged x' or checkpoint['checkpoint_mode'] != 'eval'
            or not checkpoint['complete_shared_selector'] or not checkpoint['trained_head_retained']
            or checkpoint['native_correct'] != record['native_correct'] or checkpoint['native_total'] != 18000
            or checkpoint['macro'] != record['val'] or checkpoint['inference_rule'] != protocol['inference_rule']
            or checkpoint['successful_updates'] != 240 * record['epoch']
            or checkpoint['native_presentations'] != 180000 * record['epoch']
            or checkpoint['auxiliary_presentations'] != checkpoint['native_presentations']
            or any(name.startswith('branches.') or 'projector' in name for name in state)
            or state['score_head.0.weight'].shape != (256, 384) or state['score_head.3.weight'].shape != (1, 256)
            or not torch.equal(state['instance_encoder.geometry_residual.mean'], torch.tensor(protocol['geometry_stats']['mean'], dtype=torch.float32))
            or not torch.equal(state['instance_encoder.geometry_residual.std'], torch.tensor(protocol['geometry_stats']['std'], dtype=torch.float32))):
        raise ValueError('The unique selected state must be the complete shared averaged-x NativeUtilitySelector with original buffers')
    return checkpoint


def freeze():
    from code.unified_selector.registry import PROBLEMS
    from .multitask_probe import file_hash, state_hash
    if any((ROOT / name).exists() for name in ('final_test_lock.json', 'final_test_started.json')):
        raise ValueError('Preserve the original exclusive lock/TEST marker; no reselection or retry')
    protocol, candidate, official, terminal = completed_training()
    preparation = read(ROOT / 'delivery_preparation.json')
    if not preparation['syntax_passed'] or preparation['delivery_source_hashes'] != delivery_hashes():
        raise ValueError('Require the single passed syntax invocation on these delivery sources')
    path = Path(candidate['path']).resolve()
    if not path.is_relative_to(RUN) or file_hash(path) != candidate['sha256']:
        raise ValueError('Lock only the unique published native-VAL-selected shared B checkpoint')
    checkpoint = read_checkpoint(candidate, protocol)
    state = checkpoint['model']
    record = dict(candidate, inference_path=str(path), inference_sha256=candidate['sha256'],
        model_state_sha256=state_hash(state), model_class='NativeUtilitySelector',
        factory='code.V4.r64_model.NativeUtilitySelector(**model_params); strict complete load_state_dict',
        model_params=protocol['model_params'], state_schema={name: dict(shape=list(value.shape), dtype=str(value.dtype))
            for name, value in state.items()}, checkpoint_mode='eval', complete_shared_selector=True,
        original_normalization_buffers_retained=True, projector=False, teacher_at_inference=False,
        warm_started_from_R67B36_global76=True, parameter_lifetime_is_not_schedule_global41_80=True)
    control, _, provenance = reused_control()
    manifest = read(ROOT / 'data_manifest.json')
    lock = dict(authorized=True, parent_conditional_authorization=True, frozen_utc=now(), official_arm=official,
        models={'A': control, 'B': record}, control_reuse=provenance,
        selection_rule=protocol['checkpoint_rule'], official_arm_rule=protocol['official_arm_rule'],
        A_original_VAL_correct=8923, B_original_VAL_correct=candidate['native_correct'],
        inference_rule=protocol['inference_rule'], problems=list(PROBLEMS), native_total=18000, rows_per_task=1000,
        full_legal_pools={problem: manifest['train'][problem]['pool_ids'] for problem in PROBLEMS},
        test_batch_size=128, tail_rows=104, query_batches=144, selector_forwards_per_query=1,
        completed_solvers_per_deployment_query=1, evaluation_solver_executions=0,
        new_tests_required={'A': False, 'B': True}, new_test_attempts_allowed={'A': 0, 'B': 1},
        historical_original_tests_reused={'A': 1}, test_reselection=False, cost_veto=False,
        parameter_mode='averaged x', task_wise_assembly=False, ensemble_at_B_inference=False,
        capacity_exposure_and_compute_matched=False, warm_started_package=True,
        training_terminal=terminal, delivery_source_hashes=delivery_hashes())
    exclusive(ROOT / 'final_test_lock.json', lock)
    snapshot = ROOT / 'source_delivery'
    snapshot.mkdir()
    for name in DELIVERY_SOURCES:
        shutil.copy2(WORKSPACE / 'code/V4' / name, snapshot / name)
    shutil.copy2(ROOT / 'DELIVERY_PROTOCOL.md', snapshot / 'DELIVERY_PROTOCOL.md')
    print('[R73 mechanical native VAL lock] ' + json.dumps(dict(official_arm=official,
        B_epoch=record['epoch'], B_VAL_correct=record['native_correct'], A_VAL_correct=8923, TEST_started=False)), flush=True)
    return lock


def validate_lock():
    protocol, candidate, official, _ = completed_training()
    lock = read(ROOT / 'final_test_lock.json')
    if (not lock['authorized'] or lock['official_arm'] != official
            or any(lock['models']['B'].get(key) != value for key, value in candidate.items())
            or lock['models']['B']['inference_path'] != candidate['path']
            or lock['inference_rule'] != protocol['inference_rule'] or lock['parameter_mode'] != 'averaged x'
            or lock['new_test_attempts_allowed'] != {'A': 0, 'B': 1}
            or (lock['test_batch_size'], lock['tail_rows'], lock['query_batches']) != (128, 104, 144)
            or lock['delivery_source_hashes'] != delivery_hashes()):
        raise ValueError('Keep the unique complete shared B state, native utility rule and exclusive one-TEST budget')
    return protocol, lock


def assigned_idle():
    if (os.environ.get('CUDA_VISIBLE_DEVICES') != GPU_UUID or os.environ.get('SLURM_JOB_ID') != '1465'
            or socket.gethostname().split('.')[0] != 'gpu03'):
        raise ValueError('Use only the assigned existing1465/gpu03/index1 UUID launcher')
    while True:
        gpu = capture(['nvidia-smi', '--id=' + GPU_UUID,
            '--query-gpu=index,uuid,name,memory.used,memory.total', '--format=csv,noheader,nounits'])
        apps = capture(['nvidia-smi', '--id=' + GPU_UUID,
            '--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory', '--format=csv,noheader,nounits'])
        rows = list(csv.reader(gpu['stdout'].splitlines()))
        if gpu['returncode'] or apps['returncode'] or len(rows) != 1:
            raise RuntimeError('Assigned-UUID idle query failed; preserve error and never switch devices')
        index, uuid, name, used, total = [field.strip() for field in rows[0]]
        if index != '1' or uuid != GPU_UUID:
            raise ValueError('Assigned UUID/index mismatch')
        clients = [row for row in csv.reader(apps['stdout'].splitlines()) if row and row[0].strip() == GPU_UUID]
        idle = int(used) < 500 and not clients
        observation = dict(utc=now(), host=socket.gethostname(), gpu_uuid=uuid, gpu_index=1, gpu_name=name,
            memory_used_mib=int(used), memory_total_mib=int(total), compute_apps=clients, idle=idle,
            only_assigned_uuid_queried=True, gpu_query=gpu, assigned_app_query=apps)
        append(ROOT / 'test_availability.jsonl', observation)
        if idle:
            launch = dict(observation, slurm_step=os.environ['SLURM_JOB_ID'] + '.' + os.environ['SLURM_STEP_ID'],
                trainer_pid=os.getpid(), command='bash code/V4/run_v4_r73_test.sh', tmux_session='r73_final_test',
                tmux_pane=os.environ['R73_TEST_TMUX_PANE_ID'], tmux_pane_pid=os.environ['R73_TEST_TMUX_PANE_PID'])
            exclusive(ROOT / 'test_launch_receipt.json', launch)
            return launch
        print('[R73 TEST wait] Assigned UUID occupied; waiting30s on the same device', flush=True)
        time.sleep(30)


def final_test():
    protocol, lock = validate_lock()
    if (ROOT / 'final_test_started.json').exists():
        raise ValueError('The original one-new-B TEST marker exists; never repeat')
    launch = assigned_idle()
    exclusive(ROOT / 'final_test_started.json', dict(started_utc=now(), lock=lock, launch=launch,
        slurm_step=launch['slurm_step'], allowed_attempts=1, models_to_evaluate=['B'], automatic_retry=False))
    import numpy as np
    import torch
    from code.unified_selector.registry import PROBLEMS
    from .multitask_probe import file_hash, gather_batch, state_hash
    from .performance_targets import read_raw_costs
    from .r40_experiment import aggregate, families
    from .r43_experiment import policy_metrics
    from .r64_model import NativeUtilitySelector
    from .r67_model import INPUT_KEYS, model_inputs
    from .r69_frozen_evaluation import require_evaluation
    from .train import configure_torch, make_loader

    torch.set_num_threads(1)
    configure_torch()
    whole_started = time.monotonic()
    record = lock['models']['B']
    if file_hash(Path(record['inference_path'])) != record['inference_sha256']:
        raise ValueError('The uniquely locked averaged-x checkpoint changed')
    checkpoint = read_checkpoint(record, protocol)
    if state_hash(checkpoint['model']) != record['model_state_sha256']:
        raise ValueError('The complete shared state and buffers must match the pre-TEST lock')
    model = NativeUtilitySelector(**record['model_params'])
    model.load_state_dict(checkpoint['model'], strict=True)
    model.to('cuda:0').float().requires_grad_(False).eval()
    del checkpoint
    if sum(parameter.numel() for parameter in model.parameters()) != 1842953:
        raise ValueError('Restore only the original complete shared student, not a bank or teacher')
    raw, loaders, manifest = {}, {}, {}
    for problem in PROBLEMS:
        value = read_raw_costs(problem, 'test')
        if len(value['winner']) != 1000 or value['costs'].dtype != np.float64 or value['pool_ids'] != lock['full_legal_pools'][problem]:
            raise ValueError('Require original18x1000 TEST and every original legal solver')
        loader = make_loader(problem, 'test', 128, 0, shuffle=False, cache_device='cuda:0')
        loader.drop_last = False
        np.testing.assert_array_equal(loader.batch['pool_ids'].cpu(), value['pool_ids'])
        np.testing.assert_array_equal(loader.batch['ind'].cpu(), value['winner'])
        if loader.size != 1000:
            raise ValueError('Original native TEST row coverage changed')
        raw[problem], loaders[problem] = value, loader
        manifest[problem] = dict(rows=1000, pool=value['pool'], pool_ids=value['pool_ids'],
            data_hash=value['data_hash'], label_hash=value['label_hash'], raw_cost_dtype='float64',
            original_nodes_order_roles_and_ATSP_direction=True, normalization='saved original TRAIN buffers')
    write(ROOT / 'test_data_manifest.json', manifest)
    counters = dict(query_batches=0, query_rows=0, selector_forward_calls=0, selector_forward_rows=0,
                    encoder_forward_calls=0, encoder_forward_rows=0, tails104=0)

    def observe(module, inputs):
        batch = inputs[0]
        problem = PROBLEMS[int(batch['problem_id'])]
        if not set(batch) <= INPUT_KEYS or batch['pool_ids'].cpu().tolist() != lock['full_legal_pools'][problem]:
            raise ValueError('Only original full-pool input information may enter the shared selector')
        counters['selector_forward_calls'] += 1
        counters['selector_forward_rows'] += len(batch['n'])

    def observe_encoder(module, inputs):
        counters['encoder_forward_calls'] += 1
        counters['encoder_forward_rows'] += len(inputs[0]['n'])

    hooks = [model.register_forward_pre_hook(observe), model.instance_encoder.register_forward_pre_hook(observe_encoder)]
    import wandb
    wb = wandb.init(**protocol['wandb'], name='R73_ONE_locked_shared_B_TEST', dir=str(ROOT),
        id='r73-test-' + launch['slurm_step'].replace('.', '-'), config=dict(lock=lock, new_tests={'A': 0, 'B': 1}))
    predictions = ROOT / 'test_predictions'
    predictions.mkdir()
    results, timings = {}, []
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started = time.monotonic()
    try:
        with torch.no_grad():
            for problem in PROBLEMS:
                loader, outputs = loaders[problem], []
                ids = np.asarray(raw[problem]['pool_ids'], dtype=np.int64)
                canonical = np.argsort(ids, kind='stable')
                for start in range(0, 1000, 128):
                    indices = torch.arange(start, min(start + 128, 1000))
                    batch = model_inputs(gather_batch(loader, indices))
                    torch.cuda.synchronize()
                    before = time.monotonic()
                    output = model(batch)
                    torch.cuda.synchronize()
                    seconds = time.monotonic() - before
                    if (output['logits'].dtype != torch.float32 or output['logits'].shape != (len(indices), len(ids))
                            or not output['solver_mask'].all() or not torch.isfinite(output['logits']).all()):
                        raise ValueError('Every original legal utility must remain finite; no candidate pruning')
                    outputs.append(output['logits'].cpu())
                    counters['query_batches'] += 1
                    counters['query_rows'] += len(indices)
                    counters['tails104'] += int(len(indices) == 104)
                    free, total = torch.cuda.mem_get_info()
                    timing = dict(utc=now(), problem=problem, start_index=start, rows=len(indices),
                        true_nodes=loader.lengths[indices].tolist(), forward_seconds=seconds,
                        elapsed_seconds=time.monotonic() - started, counters=dict(counters),
                        peak_allocated_bytes=torch.cuda.max_memory_allocated(), peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                        physical_used_bytes=total - free, physical_total_bytes=total,
                        target_fields_forwarded=[], solver_executions=0)
                    timings.append(timing)
                    append(ROOT / 'test_batches.jsonl', timing)
                    write(ROOT / 'delivery_current_status.json', dict(stage='ONE_B_full18_TEST', **timing))
                    if counters['query_batches'] == 1:
                        exclusive(ROOT / 'test_first_formal_batch.json', dict(timing, launch=launch,
                            parameter_count=1842953, trainable_parameters=0, teacher_at_inference=False))
                    del output, batch
                scores = torch.cat(outputs).numpy()
                metrics, predicted = policy_metrics(scores, raw[problem])
                metrics['native_correct'] = int(np.count_nonzero(predicted == raw[problem]['winner']))
                results[problem] = metrics
                np.savez_compressed(predictions / (problem + '.npz'), indices=np.arange(1000), logits=scores,
                    utility_scores=scores, canonical_utility_scores=scores[:, canonical], canonical_pool_ids=ids[canonical],
                    pred=predicted, pred_global_id=ids[predicted], winner=raw[problem]['winner'],
                    winner_global_id=ids[raw[problem]['winner']], costs=raw[problem]['costs'],
                    nodes=loader.lengths.numpy(), pool_ids=ids, pool=np.asarray(raw[problem]['pool']))
                wb.log({'test/' + problem + '/top1': metrics['top1'], 'test/' + problem + '/native_utility_CE': metrics['ce'],
                        'test/rows_completed': counters['query_rows']})
        torch.cuda.synchronize()
        elapsed = time.monotonic() - started
        expected = dict(query_batches=144, query_rows=18000, selector_forward_calls=144, selector_forward_rows=18000,
                        encoder_forward_calls=144, encoder_forward_rows=18000, tails104=18)
        if counters != expected:
            raise ValueError('Require exactly144 query/encoder/selector calls and18000 rows with all18 tails')
        value = dict(per_problem=results, macro=aggregate(results), families=families(results),
            native_correct=sum(item['native_correct'] for item in results.values()), native_total=18000,
            complete=True, evaluation_passes=1, split='test', seconds=elapsed, complete_evaluation_seconds=elapsed,
            forward_seconds=sum(row['forward_seconds'] for row in timings), counters=counters,
            query_batch_calls=144, batch_size=128, tail_rows=104, inference_rule=lock['inference_rule'],
            CE_definition='native-winner CE of original direct utility logits; not A Borda-score CE',
            evaluated_checkpoint_path=record['inference_path'], evaluated_checkpoint_sha256=record['inference_sha256'],
            model_state_sha256=record['model_state_sha256'], epoch=record['epoch'], global_epoch=record['global_epoch'],
            parameter_mode='averaged x', complete_shared_selector=True, selector_forwards_per_query=1,
            teacher_at_inference=False, solver_executions=0, deployment_completed_solvers_per_query=1,
            new_optimizer_updates=0, trainable_parameters=0, parameter_count=1842953,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(), peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            peak_observed_physical_used_bytes=max(row['physical_used_bytes'] for row in timings),
            timing_scope='whole full18 TEST including one live selector forward/query, input-only geometry, transfers, metrics and NPZ; '
                         'excludes initial checkpoint/data loading; not singleton or completed-solver latency',
            separate_latency_benchmark=False, reference_inference=False, prediction_replay=False)
        value['macro']['macro_top1'] = value['native_correct'] / 18000
        require_evaluation(value, 1000)
        _, original_A, provenance = reused_control()
        write(ROOT / 'test_results_B.json', value)
        write(ROOT / 'test_results.json', dict(models={'A': original_A, 'B': value}, lock=lock, complete=True,
            wall_seconds=time.monotonic() - whole_started, new_tests_executed={'A': 0, 'B': 1},
            historical_original_tests_reused={'A': 1}, original_A_object_unchanged=True,
            control_reuse=provenance, test_reselection=False, solver_executions=0))
        wb.log({'test/native_correct': value['native_correct'], 'test/top1': value['macro']['macro_top1'],
                'test/complete_evaluation_seconds': elapsed})
    finally:
        for hook in hooks:
            hook.remove()
        wb.finish()
    exclusive(ROOT / 'test_process_receipt.json', dict(utc=now(), returncode=0, launch=launch,
        slurm_step=launch['slurm_step'], counters=counters, evaluation_passes=1,
        whole_work_seconds=time.monotonic() - whole_started, scheduler_terminal_pending=True))
    print('[R73 ONE B TEST process complete] ' + json.dumps(dict(native_correct=value['native_correct'],
        native_total=18000, complete_evaluation_seconds=elapsed)), flush=True)


def completed_test():
    from .r69_frozen_evaluation import require_evaluation, require_scheduler_terminal
    protocol, lock = validate_lock()
    test, marker = read(ROOT / 'test_results.json'), read(ROOT / 'final_test_started.json')
    terminal, process = read(ROOT / 'test_terminal_receipt.json'), read(ROOT / 'test_process_receipt.json')
    require_scheduler_terminal(terminal, 'r73_final_test', marker['slurm_step'])
    _, original_A, _ = reused_control()
    value = test['models']['B']
    require_evaluation(value, 1000)
    if (not test['complete'] or test['lock'] != lock or marker['lock'] != lock
            or test['models']['A'] != original_A or test['new_tests_executed'] != {'A': 0, 'B': 1}
            or process['returncode'] != 0 or process['slurm_step'] != marker['slurm_step']
            or marker['slurm_step'] == original_identity()['slurm_step'] or marker['allowed_attempts'] != 1
            or value['evaluation_passes'] != 1 or value['counters'] != process['counters']
            or value['counters'] != dict(query_batches=144, query_rows=18000, selector_forward_calls=144,
                selector_forward_rows=18000, encoder_forward_calls=144, encoder_forward_rows=18000, tails104=18)
            or value['solver_executions'] != 0 or value['teacher_at_inference']
            or value['evaluated_checkpoint_sha256'] != lock['models']['B']['inference_sha256']):
        raise ValueError('Require independently completed one-new-B original TEST and exactly unchanged historical A')
    return protocol, lock, test, terminal


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('freeze', 'test'), required=True)
    args = parser.parse_args()
    try:
        if args.stage == 'freeze':
            freeze()
        else:
            final_test()
    except BaseException as error:
        exclusive(ROOT / ('freeze_failure.json' if args.stage == 'freeze' else 'test_bootstrap_failure.json'),
            dict(utc=now(), phase=args.stage, error_type=type(error).__name__, message=str(error),
                 traceback=traceback.format_exc(), automatic_retry=False,
                 slurm_step=os.environ.get('SLURM_JOB_ID', '') + '.' + os.environ.get('SLURM_STEP_ID', '')))
        raise


if __name__ == '__main__':
    main()
