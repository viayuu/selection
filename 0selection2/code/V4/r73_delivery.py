"""Independent CPU-only R73 conditional delivery controller; no training restart."""

import os
import shlex
import socket
import subprocess
import sys
import time
import traceback

from .r73_frozen_evaluation import (ROOT, WORKSPACE, append, capture, completed_test,
    completed_training, exclusive, freeze, now, original_identity, read, write)


FAILED = ('FAILED', 'CANCELLED', 'TIMEOUT', 'OUT_OF_MEMORY', 'NODE_FAIL', 'PREEMPTED', 'BOOT_FAIL', 'DEADLINE', 'REVOKED')


def absent(observation):
    return observation['returncode'] == 1 and any(word in observation['stderr'] for word in
        ("can't find session", 'no server running', 'no sessions'))


def terminal(step, session):
    from .r71_frozen_evaluation import terminal_observation
    return terminal_observation(step, session)


def wait_training(launch):
    while True:
        current, pipeline = read(ROOT / 'current_status.json'), read(ROOT / 'pipeline_status.json')
        owned = capture(['tmux', 'has-session', '-t', '=r73_train'])
        formal_failures = [name for name in ('failures.jsonl', 'bootstrap_failures.jsonl') if (ROOT / name).exists()]
        last = current.get('last_teacher_batch', current.get('last_metrics', {}))
        observation = dict(utc=now(), phase=current['stage'], status_utc=current['updated_utc'],
            student_successful_updates=current.get('student_successful_updates', 0),
            teacher_counters=last.get('cumulative') if current['stage'] == 'formal_teacher_cache' else None,
            native_presentations=last.get('native_presentations'), auxiliary_presentations=last.get('auxiliary_presentations'),
            owned_training_tmux=owned, pipeline_stage=pipeline['stage'], scheduler_queried=False,
            formal_failure_files=formal_failures)
        append(ROOT / 'delivery_monitor.jsonl', observation)
        write(ROOT / 'delivery_current_status.json', dict(observation, stage='waiting_original_formal_process'))
        print('[R73 delivery wait] ' + str({key: observation[key] for key in
            ('phase', 'student_successful_updates', 'teacher_counters', 'formal_failure_files')}), flush=True)
        if formal_failures or pipeline['stage'] == 'failed_preserved_no_retry':
            actual = terminal(launch['slurm_step'], 'r73_train')
            write(ROOT / 'delivery_formal_failure_observation.json', dict(actual, original_failures=formal_failures,
                no_process_stop_or_restart=True))
            raise RuntimeError('Original formal failure recorded; preserve it without regeneration or training restart')
        if owned['returncode'] not in (0, 1):
            raise RuntimeError('Owned training tmux observation failed; no inference about process completion')
        if absent(owned):
            actual = terminal(launch['slurm_step'], 'r73_train')
            append(ROOT / 'delivery_terminal_observations.jsonl', actual)
            if actual['state'].startswith(FAILED) or (actual['state'] == 'COMPLETED' and actual['exit_code'] != '0:0'):
                if not (ROOT / 'training_terminal_receipt.json').exists():
                    exclusive(ROOT / 'training_terminal_receipt.json', actual)
                raise RuntimeError('Original assigned formal step failed; preserve original failure without retry')
            if actual['state'] == 'COMPLETED' and actual['tmux_terminal']:
                from .r69_frozen_evaluation import require_scheduler_terminal
                require_scheduler_terminal(actual, 'r73_train', launch['slurm_step'])
                actual.update(original_trainer_pid=launch['trainer_pid'], original_pane_pid=launch['tmux_pane_pid'],
                    formal_process_gone=True,
                    formal_process_evidence='Original exec Python tied to this Slurm step; scheduler terminal0 confirms the step process exited; owned launcher tmux is independently absent')
                if (ROOT / 'training_terminal_receipt.json').exists():
                    require_scheduler_terminal(read(ROOT / 'training_terminal_receipt.json'), 'r73_train', launch['slurm_step'])
                    write(ROOT / 'delivery_training_terminal_observation.json', actual)
                else:
                    exclusive(ROOT / 'training_terminal_receipt.json', actual)
                completed_training()
                return actual
        time.sleep(240)


def run_test():
    command = 'bash code/V4/run_v4_r73_test.sh >> code/V4/runs/R73_expansion_teacher_training/test_bootstrap.log 2>&1'
    dispatch = capture(['tmux', 'new-session', '-d', '-s', 'r73_final_test', '-c', str(WORKSPACE), command])
    exclusive(ROOT / 'test_tmux_dispatch.json', dict(dispatch, session='r73_final_test', attempts=1))
    if dispatch['returncode'] != 0:
        raise RuntimeError('ONE TEST tmux dispatch failed; preserve any existing session; no retry')
    while not (ROOT / 'test_launch_receipt.json').exists():
        owned = capture(['tmux', 'has-session', '-t', '=r73_final_test'])
        if (ROOT / 'test_bootstrap_failure.json').exists():
            if absent(owned):
                raise RuntimeError('ONE TEST bootstrap failed and owned session exited; no retry')
        if absent(owned):
            path = ROOT / 'test_bootstrap.log'
            exclusive(ROOT / 'test_dispatch_failure.json', dict(utc=now(), owned_tmux=owned,
                original_log_tail=path.read_text()[-8000:] if path.exists() else None, automatic_retry=False))
            raise RuntimeError('ONE TEST tmux exited before actual assigned-UUID launch; no retry')
        time.sleep(30)
    step = read(ROOT / 'test_launch_receipt.json')['slurm_step']
    if not step.startswith('1465.') or step == original_identity()['slurm_step']:
        raise ValueError('Use one actual new TEST step in the original allocation only')
    while True:
        observation = terminal(step, 'r73_final_test')
        append(ROOT / 'delivery_test_monitor.jsonl', observation)
        if observation['tmux_terminal']:
            if observation['state'].startswith(FAILED) or (observation['state'] == 'COMPLETED' and observation['exit_code'] != '0:0'):
                exclusive(ROOT / 'test_terminal_receipt.json', observation)
                raise RuntimeError('The original ONE TEST failed; preserve it without retry or reselection')
            if observation['state'] == 'COMPLETED':
                from .r69_frozen_evaluation import require_scheduler_terminal
                require_scheduler_terminal(observation, 'r73_final_test', step)
                exclusive(ROOT / 'test_terminal_receipt.json', observation)
                completed_test()
                return observation
        time.sleep(30)


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or os.environ.get('R73_CONDITIONAL_DELIVERY') != 'authorized':
        raise ValueError('Parent-authorized independent controller must remain CPU-only')
    preparation = read(ROOT / 'delivery_preparation.json')
    if not preparation['syntax_passed'] or preparation['syntax_check']['returncode'] != 0:
        raise ValueError('Require the one passed combined syntax invocation')
    pane = os.environ.get('TMUX_PANE', '')
    identity = capture(['tmux', 'display-message', '-p', '-t', pane, '#{session_name}|#{pane_id}|#{pane_pid}'])
    if identity['returncode'] != 0 or not identity['stdout'].startswith('r73_delivery|'):
        raise ValueError('Independent controller belongs only in owned r73_delivery')
    launch = original_identity()
    exclusive(ROOT / 'delivery_started.json', dict(utc=now(), pid=os.getpid(), host=socket.gethostname(),
        tmux='r73_delivery', pane=pane, pane_pid=int(identity['stdout'].strip().split('|')[2]), identity=identity,
        command='bash code/V4/run_v4_r73_delivery.sh', controller='python -B -u -m code.V4.r73_delivery',
        original_formal_process=launch, monitor_interval_seconds=240, parent_conditional_authorization=True,
        CPU_only=True, future_execution='UNVERIFIED', automatic_retry=False, no_training_stop_or_restart=True))
    try:
        train_terminal = wait_training(launch)
        freeze()
        test_terminal = run_test()
        command = [sys.executable, '-B', '-u', '-m', 'code.V4.r73_analysis', '--authorize-analysis']
        with (ROOT / 'analysis.log').open('x') as stream:
            result = subprocess.run(command, cwd=WORKSPACE, stdout=stream, stderr=subprocess.STDOUT)
        exclusive(ROOT / 'analysis_process_receipt.json', dict(utc=now(), command=shlex.join(command),
            returncode=result.returncode, invocations=1, CPU_only=True, automatic_retry=False))
        if result.returncode:
            raise RuntimeError('ONCE CPU analysis failed; preserve the original invocation without retry')
        receipt = read(ROOT / 'delivery_receipt.json')
        receipt.update(controller_pid=os.getpid(), original_train_terminal=train_terminal, test_terminal=test_terminal,
            owned_training_and_test_tmux_gone=dict(train=capture(['tmux', 'has-session', '-t', '=r73_train']),
                                                  test=capture(['tmux', 'has-session', '-t', '=r73_final_test'])),
            controller_test_and_analysis_foreground_settled=True)
        write(ROOT / 'delivery_receipt.json', receipt)
        write(ROOT / 'delivery_current_status.json', dict(stage='complete', utc=now(), analysis_invocations=1,
            new_tests_executed={'A': 0, 'B': 1}, parent_reviews_audit_memory_Git_pending=True))
        print('[R73 conditional delivery complete] ' + str(receipt['decision']), flush=True)
    except BaseException as error:
        exclusive(ROOT / 'delivery_failure.json', dict(utc=now(), error_type=type(error).__name__, message=str(error),
            traceback=traceback.format_exc(), automatic_retry=False, original_failures_preserved=True))
        write(ROOT / 'delivery_current_status.json', dict(stage='failed_preserved_no_retry', utc=now(), error=str(error)))
        raise


if __name__ == '__main__':
    main()
