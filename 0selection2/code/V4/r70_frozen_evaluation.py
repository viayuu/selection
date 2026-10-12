"""R70 conditional delivery: original terminal gates, VAL freeze, ONE new B TEST."""

import argparse
import csv
import json
import os
import shlex
import socket
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

from code.unified_selector.registry import PROBLEMS
from .multitask_probe import file_hash
from .r59_experiment import GPU_UUID, WORKSPACE, atomic_json
from .r64_model import INFERENCE_RULE
from .r69_frozen_evaluation import (load_bank, read_bundle, require_evaluation,
                                    require_scheduler_terminal)
from .r69_model import INPUT_KEYS, PARAMETER_MODE
from .r70_experiment import CONTROL_ROOT, ROOT, RUN_NAME, append_jsonl


TRAIN_STEP = '1465.103'
CONTROL_RUN = CONTROL_ROOT / 'B_task_specific_seed2'
NEW_TESTS_REQUIRED = {'A': False, 'B': True}
DELIVERY_FILES = ('r70_frozen_evaluation.py', 'run_v4_r70_test.sh', 'r70_analysis.py')


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text())


def exclusive_json(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def delivery_hashes():
    paths = [WORKSPACE / 'code/V4' / name for name in DELIVERY_FILES] + [ROOT / 'DELIVERY_PROTOCOL.md']
    return {str(path.relative_to(WORKSPACE)): file_hash(path) for path in paths}


def reused_control():
    lock_path, test_path = CONTROL_ROOT / 'final_test_lock.json', CONTROL_ROOT / 'test_results.json'
    old_lock, old_test = read(lock_path), read(test_path)
    record, value = old_lock['models']['B'], old_test['models']['B']
    candidate = read(ROOT / 'reused_control.json')['model']
    if (any(record.get(key) != val for key, val in candidate.items())
            or (record['epoch'], record['global_epoch'], record['native_correct'], record['native_total']) != (20, 60, 8604, 18000)
            or record['model_class'] != 'TaskSpecificSelector' or record['all18_common_epoch'] is not True
            or old_test['complete'] is not True or old_test['lock'] != old_lock
            or value['complete'] is not True or value['evaluation_passes'] != 1 or value['native_total'] != 18000
            or value['evaluated_checkpoint_sha256'] != record['inference_sha256']
            or value['model_state_sha256'] != record['model_state_sha256']):
        raise ValueError('Reuse only unchanged locked R69B20/global60 complete-bank metadata and its original TEST object')
    provenance = dict(label='fixed R70 A = historical R69B20', lock_path=str(lock_path),
        lock_json_pointer='/models/B', test_path=str(test_path), test_json_pointer='/models/B',
        history_path=str(CONTROL_RUN / 'history.json'), result_path=str(CONTROL_RUN / 'result.json'),
        original_tests_already_executed=1, new_R70_tests=0, new_R70_training=False,
        weights_loaded=False, inference=False, recount=False, reselection=False,
        historical_test_exposure=True, reuse_is_protocol_equivalence_not_independent_replication=True)
    return record, value, provenance


def completed_training():
    terminal = read(ROOT / 'training_terminal_receipt.json')
    require_scheduler_terminal(terminal, 'r70_train', TRAIN_STEP)
    launch = read(ROOT / 'launch_receipt.json')
    protocol, complete = read(ROOT / 'protocol.json'), read(ROOT / 'training_complete.json')
    candidates = read(ROOT / 'validation_candidates.json')
    result, history = read(ROOT / RUN_NAME / 'result.json'), read(ROOT / RUN_NAME / 'history.json')
    if (launch['slurm_step'] != TRAIN_STEP or launch['gpu_uuid'] != GPU_UUID or launch['gpu_index'] != 1
            or protocol['new_arms'] != {'B': RUN_NAME} or protocol['inference_rule'] != INFERENCE_RULE
            or complete['new_arms'] != ['B'] or complete['epochs'] != 40
            or (complete['new_updates'], complete['new_presentations'], complete['successful_view_rows'],
                complete['successful_mixed_head_rows']) != (9600, 7200000, 14400000, 14400000)
            or len(history) != 40 or [row['epoch'] for row in history] != list(range(1, 41))
            or [row['global_epoch'] for row in history] != list(range(41, 81))
            or (result['epochs_completed'], result['successful_updates'], result['presentations'],
                result['successful_view_rows'], result['successful_mixed_head_rows']) != (40, 9600, 7200000, 14400000, 14400000)
            or result['actual_execution']['recorded_successful_optimizer_steps'] != 9600
            or result['actual_execution']['recorded_query_presentations'] != 7200000
            or result['actual_execution']['recorded_amp_retries'] != result['amp_retries']
            or result['attempted_view_rows'] < 14400000 or result['attempted_mixed_head_rows'] < 14400000
            or result['all18_common_epoch'] is not True or result['parameter_counts']['B'] != 33173154
            or result['parameter_counts']['A'] != 33173154 or result['parameter_counts']['branch'] != 1842953
            or set(result['updates_by_problem'].values()) != {1600}
            or set(result['rows_by_problem'].values()) != {400000}):
        raise ValueError('Require original completed fixed40E/9600updates/7.2Mrows/14.4Mencoder+14.4Mmixed-head rows')
    for row in history:
        require_evaluation(row['val'], 1000)
        if (row['all18_common_epoch'] is not True or row['successful_updates'] != 240 * row['epoch']
                or row['presentations'] != 180000 * row['epoch']
                or row['successful_view_rows'] != 2 * row['presentations']
                or row['successful_mixed_head_rows'] != 2 * row['presentations']):
            raise ValueError('Every saved epoch must retain full18 VAL and the original cumulative budget')
        if row['epoch'] % 5 == 0:
            require_evaluation(row['train'], 10000)
        elif row.get('train') is not None:
            raise ValueError('Do not invent dense TRAIN evaluations')
    selected = max(history, key=lambda row: row['val']['native_correct'])
    candidate = candidates['models']['B']
    unique = ROOT / RUN_NAME / f'best_checkpoints/epoch{selected["epoch"]:03d}.pt'
    expected = dict(epoch=selected['epoch'], global_epoch=selected['global_epoch'],
        native_correct=selected['val']['native_correct'], native_total=18000,
        val=selected['val']['macro'], selected_epoch_checkpoint=str(unique),
        parameter_mode=PARAMETER_MODE, all18_common_epoch=True, model_type='complete_task_specific_bank')
    official = 'B' if candidate['native_correct'] > 8604 else 'A'
    if (any(candidate.get(key) != value for key, value in expected.items())
            or result['best']['epoch'] != selected['epoch'] or result['best']['val'] != selected['val']
            or result['final'] != history[-1] or result['selected_epoch_checkpoint'] != str(unique)
            or result['best_checkpoint_sha256'] != candidate['sha256']
            or candidates['proposed_official_arm'] != official
            or candidates['models']['A'] != read(ROOT / 'reused_control.json')['model']):
        raise ValueError('Published B must equal the earliest maximum INTEGER full18 VAL epoch; official B strictly>8604')
    require_evaluation(result['best']['train'], 10000)
    return protocol, candidate, official


def freeze():
    if (ROOT / 'final_test_lock.json').exists() or (ROOT / 'final_test_started.json').exists():
        raise ValueError('Preserve the exclusive existing VAL lock/TEST marker; no reselection or retry')
    protocol, candidate, official = completed_training()
    control, _, provenance = reused_control()
    args = read(ROOT / RUN_NAME / 'args.json')
    path = Path(candidate['selected_epoch_checkpoint']).resolve()
    if not path.is_relative_to(ROOT / RUN_NAME) or file_hash(path) != candidate['sha256']:
        raise ValueError('Selected unique complete B bank must match its published file identity')
    checkpoint = read_bundle(path, candidate, args)
    record = dict(candidate, inference_path=str(path), inference_sha256=candidate['sha256'],
        model_state_sha256=checkpoint['model_state_sha256'], model_class='TaskSpecificSelector',
        factory='code.V4.r69_model.make_bank', common_epoch=candidate['epoch'],
        branch_epochs=[candidate['epoch']] * 18, problem_names=list(PROBLEMS), checkpoint_mode='eval',
        parameter_averaging=False, perturbed_weights=False, optimizer_mode_transition=False, projector_included=False)
    lock = dict(authorized=True, conditional_parent_authorization=True, frozen_utc=now(), official_arm=official,
        models={'A': control, 'B': record}, control_reuse=provenance,
        new_tests_required=NEW_TESTS_REQUIRED, new_test_attempts_allowed={'A': 0, 'B': 1},
        historical_original_tests_reused={'A': 1}, new_arms=['B'], reused_arms=['A'],
        selection='B earliest maximum INTEGER full18 VAL correct; official B strictly>8604, tie/lower fixed A20',
        original_full18=True, problems=list(PROBLEMS), rows_per_task=1000, native_total=18000,
        full_legal_pools={task: protocol['data_manifest']['train'][task]['pool_ids'] for task in PROBLEMS},
        inference_rule=INFERENCE_RULE, test_batch_size=128, tail_rows=104,
        selector_forwards_per_query=1, deployment_completed_solvers_per_query=1, evaluation_solver_executions=0,
        parameter_mode=PARAMETER_MODE, task_wise_assembly=False, test_reselection=False, costs_select_checkpoint=False,
        parameter_capacity_matched=True, A_parameters=33173154, B_parameters=33173154,
        extra_training_head_compute=True, training_terminal=read(ROOT / 'training_terminal_receipt.json'),
        delivery_source_hashes=delivery_hashes())
    exclusive_json(ROOT / 'final_test_lock.json', lock)
    print('[R70 VAL freeze] ' + json.dumps(dict(official_arm=official, B_epoch=record['epoch'],
        B_VAL_correct=record['native_correct'], A_VAL_correct=8604, TEST_executed=False)), flush=True)


def validate_lock():
    protocol, candidate, official = completed_training()
    lock = read(ROOT / 'final_test_lock.json')
    if (lock['authorized'] is not True or lock['official_arm'] != official
            or any(lock['models']['B'].get(key) != value for key, value in candidate.items())
            or lock['models']['B']['inference_path'] != candidate['selected_epoch_checkpoint']
            or lock['new_test_attempts_allowed'] != {'A': 0, 'B': 1}
            or lock['new_tests_required'] != NEW_TESTS_REQUIRED or lock['test_batch_size'] != 128
            or lock['tail_rows'] != 104 or lock['inference_rule'] != INFERENCE_RULE
            or lock['delivery_source_hashes'] != delivery_hashes()):
        raise ValueError('Retain the parent-authorized common-epoch VAL lock, fixed TEST budget and delivery sources')
    return protocol, lock


def gpu_bootstrap():
    validate_lock()
    if (ROOT / 'final_test_started.json').exists():
        raise ValueError('The one new B TEST marker already exists; no automatic retry')
    while True:
        gpu = subprocess.check_output(['nvidia-smi', '--id=' + GPU_UUID,
            '--query-gpu=index,uuid,name,memory.used,memory.total', '--format=csv,noheader,nounits'], text=True)
        apps = subprocess.check_output(['nvidia-smi', '--id=' + GPU_UUID,
            '--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory', '--format=csv,noheader,nounits'], text=True)
        rows = list(csv.reader(gpu.splitlines()))
        if len(rows) != 1:
            raise ValueError('Only the assigned UUID may be queried')
        index, actual, name, memory, total = [value.strip() for value in rows[0]]
        if index != '1' or actual != GPU_UUID:
            raise ValueError('Assigned UUID/index identity mismatch; no device switch')
        clients = [row for row in csv.reader(apps.splitlines()) if row and row[0].strip() == GPU_UUID]
        receipt = dict(checked_utc=now(), host=socket.gethostname(),
            slurm_step=os.environ['SLURM_JOB_ID'] + '.' + os.environ['SLURM_STEP_ID'],
            gpu_uuid=actual, gpu_index=1, gpu_name=name, memory_used_mib=int(memory), memory_total_mib=int(total),
            compute_apps=clients, idle=int(memory) < 500 and not clients, tmux_session='r70_final_test',
            tmux_pane=os.environ['R70_TEST_TMUX_PANE_ID'], tmux_pane_pid=os.environ['R70_TEST_TMUX_PANE_PID'],
            tmux_server_host=os.environ['R70_TEST_TMUX_SERVER_HOST'], only_assigned_uuid_queried=True)
        append_jsonl(ROOT / 'test_availability.jsonl', receipt)
        if receipt['idle']:
            atomic_json(ROOT / 'test_launch_receipt.json', receipt)
            print('[R70 assigned-UUID idle] ' + json.dumps(receipt), flush=True)
            return
        print('[R70 TEST wait] assigned UUID occupied; wait30s on this same device', flush=True)
        time.sleep(30)


def final_test():
    import numpy as np
    import torch
    from .performance_targets import read_raw_costs
    from .r64_experiment import evaluate, synchronize
    from .train import configure_torch, make_loader

    protocol, lock = validate_lock()
    _, control_metrics, provenance = reused_control()
    gpu_bootstrap()
    marker = dict(started_utc=now(), lock=lock, allowed_attempts=1, models_to_evaluate=['B'],
        new_tests_required=NEW_TESTS_REQUIRED, tmux_session='r70_final_test',
        tmux_pane=os.environ['R70_TEST_TMUX_PANE_ID'], tmux_pane_pid=os.environ['R70_TEST_TMUX_PANE_PID'],
        host=socket.gethostname(), pid=os.getpid(), gpu_uuid=GPU_UUID,
        slurm_step=os.environ['SLURM_JOB_ID'] + '.' + os.environ['SLURM_STEP_ID'], automatic_retry=False)
    exclusive_json(ROOT / 'final_test_started.json', marker)
    configure_torch()
    torch.set_num_threads(1)
    started = time.monotonic()
    raw, loaders, manifest = {}, {}, {}
    for task in PROBLEMS:
        value = read_raw_costs(task, 'test')
        original = protocol['data_manifest']['train'][task]
        if (len(value['winner']) != 1000 or value['costs'].dtype != np.float64
                or any(value[key] != original[key] for key in ('pool', 'pool_ids'))):
            raise ValueError('Retain original full18x1000 TEST, raw FP64 costs and full legal pools')
        loader = make_loader(task, 'test', 128, 0, shuffle=False, cache_device='cuda:0')
        loader.drop_last = False
        if loader.size != 1000:
            raise ValueError('Original TEST input coverage changed')
        np.testing.assert_array_equal(loader.batch['ind'].cpu(), value['winner'])
        np.testing.assert_array_equal(loader.batch['pool_ids'].cpu(), value['pool_ids'])
        raw[task], loaders[task] = value, loader
        manifest[task] = dict(rows=1000, pool=value['pool'], pool_ids=value['pool_ids'],
            data_hash=value['data_hash'], label_hash=value['label_hash'], costs_dtype='raw FP64', ordered_original_rows=True)
    atomic_json(ROOT / 'test_data_manifest.json', manifest)
    model = load_bank(ROOT, lock['models']['B'], read(ROOT / RUN_NAME / 'args.json'), 'cuda:0')
    calls, branch_calls = [], []

    def input_boundary(module, inputs):
        batch = inputs[0]
        branch_id = model.branch_index(batch['problem_id'])
        if (not set(batch) <= INPUT_KEYS
                or batch['pool_ids'].cpu().tolist() != lock['full_legal_pools'][PROBLEMS[branch_id]]):
            raise ValueError('One ordinary clean full-pool forward; labels stay outside model inputs')
        calls.append((branch_id, len(batch['n'])))

    def branch_boundary(index):
        def observe(module, inputs):
            batch = inputs[0]
            if model.branch_index(batch['problem_id']) != index or not set(batch) <= INPUT_KEYS:
                raise ValueError('Exactly one known-problem complete branch must activate')
            branch_calls.append((index, len(batch['n'])))
        return observe

    hooks = [model.register_forward_pre_hook(input_boundary)]
    hooks += [branch.register_forward_pre_hook(branch_boundary(index)) for index, branch in enumerate(model.branches)]
    synchronize('cuda:0')
    before = time.monotonic()
    value = evaluate(model, loaders, raw, batch_size=128, prediction_dir=ROOT / RUN_NAME / 'test_predictions', timing_path=None)
    synchronize('cuda:0')
    elapsed = time.monotonic() - before
    for hook in hooks:
        hook.remove()
    require_evaluation(value, 1000)
    expected = [(index, rows) for index in range(18) for rows in [128] * 7 + [104]]
    if calls != expected or branch_calls != expected:
        raise ValueError('Require exactly144 wrapper/active-branch batches and all104-row tails')
    record = lock['models']['B']
    value.update(complete=True, evaluation_passes=1, evaluation_origin='new R70 B original full18 TEST',
        complete_evaluation_seconds=elapsed, evaluated_checkpoint_path=record['inference_path'],
        evaluated_checkpoint_sha256=record['inference_sha256'], model_state_sha256=record['model_state_sha256'],
        common_epoch=record['common_epoch'], all18_common_epoch=True, parameter_mode=PARAMETER_MODE,
        parameter_averaging=False, perturbed_weights=False, optimizer_mode_transition=False, projector_included=False,
        selector_forwards_per_query=1, active_branch_forward_calls=len(branch_calls), forward_batch_calls=len(calls),
        batch_size=128, tail_rows=104, tails_retained=True, cached_test_scores=False,
        solver_executions=0, deployment_completed_solvers_per_query=1, inference_rule=INFERENCE_RULE,
        separate_latency_benchmark=False,
        timing_scope='ordinary complete evaluation walltime including transfer/metrics/NPZ saving; not singleton latency or solver runtime')
    atomic_json(ROOT / 'test_results_B.json', value)
    atomic_json(ROOT / 'test_results.json', dict(models={'A': control_metrics, 'B': value}, lock=lock,
        complete=True, wall_seconds=time.monotonic() - started, original_full18=True,
        new_tests_executed={'A': 0, 'B': 1}, historical_original_tests_reused={'A': 1},
        tests_per_new_model=1, control_reuse=provenance, solver_executions=0, prediction_replay=False,
        model_execution={'A': 'unchanged R69 test_results.json/models/B object; no new inference', 'B': 'ONE new ordinary-clean full18 TEST'}))
    print('[R70 one NEW B TEST] ' + json.dumps(dict(native_correct=value['native_correct'], native_total=18000,
        complete_evaluation_seconds=elapsed, new_A_tests=0, new_B_tests=1)), flush=True)


def completed_test():
    protocol, lock = validate_lock()
    test, marker = read(ROOT / 'test_results.json'), read(ROOT / 'final_test_started.json')
    terminal, process = read(ROOT / 'test_terminal_receipt.json'), read(ROOT / 'test_process_receipt.json')
    require_scheduler_terminal(terminal, 'r70_final_test', marker['slurm_step'])
    _, original_A, _ = reused_control()
    value = test['models']['B']
    if (test['complete'] is not True or test['lock'] != lock or marker['lock'] != lock
            or test['models']['A'] != original_A or test['new_tests_executed'] != {'A': 0, 'B': 1}
            or process['returncode'] != 0 or process['slurm_step'] != marker['slurm_step']
            or marker['slurm_step'] == TRAIN_STEP or marker['allowed_attempts'] != 1
            or value['evaluation_passes'] != 1 or value['forward_batch_calls'] != 144
            or value['active_branch_forward_calls'] != 144 or value['common_epoch'] != lock['models']['B']['epoch']):
        raise ValueError('Require independently completed ONE new B TEST and unchanged historical A object')
    require_evaluation(value, 1000)
    return protocol, lock, test, terminal


def capture_command(command):
    started = time.monotonic()
    recorded = now()
    result = subprocess.run(command, text=True, capture_output=True)
    return dict(command=shlex.join(command), recorded_utc=recorded, finished_utc=now(),
        wall_seconds=time.monotonic() - started, returncode=result.returncode, stdout=result.stdout, stderr=result.stderr)


def terminal_observation(step, session):
    scheduler = capture_command(['sacct', '-j', step, '--noheader', '--parsable2',
        '--format=JobIDRaw,State,ExitCode,ElapsedRaw,Start,End,NodeList'])
    tmux = capture_command(['tmux', 'has-session', '-t', '=' + session])
    records = [line.strip().split('|') for line in scheduler['stdout'].splitlines() if line.strip()]
    matching = [row for row in records if len(row) >= 7 and row[0] == step]
    row = matching[0] if len(matching) == 1 else None
    absent = tmux['returncode'] == 1 and any(text in tmux['stderr'] for text in ("can't find session", 'no server running', 'no sessions'))
    observation = dict(recorded_utc=now(), slurm_step=step, tmux_session=session,
        state=row[1] if row else 'UNAVAILABLE', exit_code=row[2] if row else None,
        elapsed_seconds=float(row[3]) if row and row[3].isdigit() else None,
        node=row[6] if row else None, tmux_terminal=absent, scheduler_evidence=scheduler, tmux_evidence=tmux)
    return observation


def wait_terminal(step, session, filename, interval):
    failed_states = ('FAILED', 'CANCELLED', 'TIMEOUT', 'OUT_OF_MEMORY', 'NODE_FAIL', 'PREEMPTED', 'BOOT_FAIL', 'DEADLINE', 'REVOKED')
    while True:
        observation = terminal_observation(step, session)
        if session == 'r70_train':
            status, pipeline = read(ROOT / 'current_status.json'), read(ROOT / 'pipeline_status.json')
            observation['training_status'] = {key: status.get(key) for key in
                ('status', 'epoch', 'completed_epochs', 'successful_updates', 'presentations', 'selected_B_epoch', 'selected_B_validation_correct')}
            observation['pipeline_stage'] = pipeline['stage']
        append_jsonl(ROOT / 'delivery_monitor.jsonl', observation)
        atomic_json(ROOT / 'delivery_current_status.json', dict(stage='waiting_' + session, **observation))
        print('[R70 delivery monitor] ' + json.dumps(dict(session=session, step=step,
            state=observation['state'], exit_code=observation['exit_code'], tmux_gone=observation['tmux_terminal'],
            training=observation.get('training_status'))), flush=True)
        if observation['state'].startswith(failed_states) or (observation['state'] == 'COMPLETED' and observation['exit_code'] != '0:0'):
            atomic_json(ROOT / filename, observation)
            raise RuntimeError('Original owned step failed; no automatic retry/restart: ' + step)
        if observation['state'] == 'COMPLETED' and observation['tmux_terminal']:
            require_scheduler_terminal(observation, session, step)
            atomic_json(ROOT / filename, observation)
            return observation
        if session == 'r70_final_test' and observation['tmux_terminal'] and (ROOT / 'test_bootstrap_failure.json').exists():
            raise RuntimeError('Owned TEST bootstrap failed; preserve failure, no retry')
        time.sleep(interval)


def deliver():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or os.environ.get('R70_CONDITIONAL_DELIVERY') != 'authorized':
        raise ValueError('Conditional delivery must remain CPU-only with the explicit parent authorization')
    preparation = read(ROOT / 'delivery_preparation.json')
    if preparation['syntax_check']['returncode'] != 0:
        raise ValueError('New delivery syntax must pass before starting the CPU controller')
    exclusive_json(ROOT / 'delivery_started.json', dict(started_utc=now(), pid=os.getpid(),
        tmux='r70_delivery', pane=os.environ.get('TMUX_PANE'), parent_conditional_authorization=True,
        original_training_step=TRAIN_STEP, monitor_interval_seconds=240,
        no_training_stop_or_restart=True, no_test_retry=True, no_Git=True))
    try:
        wait_terminal(TRAIN_STEP, 'r70_train', 'training_terminal_receipt.json', 240)
        completed_training()
        freeze()
        environment = dict(os.environ, R70_PARENT_GPU_BATON='locked-test')
        command = 'bash code/V4/run_v4_r70_test.sh >> code/V4/runs/R70_latent_mixup/test_bootstrap.log 2>&1'
        launch = subprocess.run(['tmux', 'new-session', '-d', '-s', 'r70_final_test', '-c', str(WORKSPACE), command],
                                text=True, capture_output=True, env=environment)
        atomic_json(ROOT / 'test_tmux_dispatch.json', dict(recorded_utc=now(), returncode=launch.returncode,
            stdout=launch.stdout, stderr=launch.stderr, command=command, session='r70_final_test', attempts=1))
        if launch.returncode != 0:
            raise RuntimeError('ONE owned TEST tmux dispatch failed; no automatic retry')
        while not (ROOT / 'test_launch_receipt.json').exists():
            if (ROOT / 'test_bootstrap_failure.json').exists():
                raise RuntimeError('ONE TEST bootstrap failed; no automatic retry')
            owned_tmux = capture_command(['tmux', 'has-session', '-t', '=r70_final_test'])
            if owned_tmux['returncode'] == 1:
                path = ROOT / 'test_bootstrap.log'
                atomic_json(ROOT / 'test_bootstrap_failure.json', dict(recorded_utc=now(),
                    message='Owned TEST tmux exited before an assigned-UUID launch receipt', tmux_evidence=owned_tmux,
                    original_log_tail=path.read_text()[-8000:] if path.exists() else None, automatic_retry=False))
                raise RuntimeError('ONE TEST dispatch/bootstrap exited before launch; no automatic retry')
            time.sleep(30)
        step = read(ROOT / 'test_launch_receipt.json')['slurm_step']
        if not step.startswith('1465.') or step == TRAIN_STEP:
            raise ValueError('TEST must use its one new step within existing allocation1465')
        wait_terminal(step, 'r70_final_test', 'test_terminal_receipt.json', 30)
        completed_test()
        analysis_command = [sys.executable, '-B', '-u', '-m', 'code.V4.r70_analysis', '--authorize-analysis']
        with (ROOT / 'analysis.log').open('x') as stream:
            analysis = subprocess.run(analysis_command, cwd=WORKSPACE, stdout=stream, stderr=subprocess.STDOUT)
        atomic_json(ROOT / 'analysis_process_receipt.json', dict(recorded_utc=now(), command=shlex.join(analysis_command),
            returncode=analysis.returncode, invocations=1, cpu_only=True, automatic_retry=False))
        if analysis.returncode != 0:
            raise RuntimeError('The ONCE CPU analysis failed; preserve original failure, no automatic retry')
        receipt = read(ROOT / 'delivery_receipt.json')
        receipt['owned_training_and_test_tmux_gone'] = dict(
            train=capture_command(['tmux', 'has-session', '-t', '=r70_train']),
            test=capture_command(['tmux', 'has-session', '-t', '=r70_final_test']))
        atomic_json(ROOT / 'delivery_receipt.json', receipt)
        atomic_json(ROOT / 'delivery_current_status.json', dict(stage='complete', recorded_utc=now(),
            analysis_invocations=1, parent_reviews_audit_memory_Git_pending=True))
        print('[R70 conditional delivery complete] ' + json.dumps(receipt['decision']), flush=True)
    except BaseException as error:
        atomic_json(ROOT / 'delivery_failure.json', dict(recorded_utc=now(), error_type=type(error).__name__,
            message=str(error), traceback=traceback.format_exc(), automatic_retry=False, original_failures_preserved=True))
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('deliver', 'test'), required=True)
    args = parser.parse_args(argv)
    if args.stage == 'deliver':
        deliver()
        return
    if (os.environ.get('R70_PARENT_GPU_BATON') != 'locked-test'
            or os.environ.get('CUDA_VISIBLE_DEVICES') != GPU_UUID or os.environ.get('SLURM_JOB_ID') != '1465'
            or socket.gethostname().split('.')[0] != 'gpu03' or os.environ.get('R70_TMUX_SESSION') != 'r70_final_test'):
        parser.error('Use only owned r70_final_test and existing1465/gpu03/assigned-UUID locked TEST launcher')
    status = 1
    try:
        final_test()
        status = 0
    except BaseException as error:
        filename = 'final_test_failure.json' if (ROOT / 'final_test_started.json').exists() else 'test_bootstrap_failure.json'
        atomic_json(ROOT / filename, dict(recorded_utc=now(), error_type=type(error).__name__, message=str(error),
            traceback=traceback.format_exc(), automatic_retry=False, marker_preserved=True))
        raise
    finally:
        exclusive_json(ROOT / 'test_process_receipt.json', dict(recorded_utc=now(), returncode=status,
            slurm_step=os.environ['SLURM_JOB_ID'] + '.' + os.environ['SLURM_STEP_ID'], tmux_session='r70_final_test',
            new_evaluated_arms=['B'], new_test_attempts_started={'A': 0, 'B': int((ROOT / 'final_test_started.json').exists())},
            new_tests_completed={'A': 0, 'B': int(status == 0)}, historical_original_A_tests_reused=1,
            process_receipt_is_scheduler_evidence=False, automatic_retry=False))


if __name__ == '__main__':
    main()
