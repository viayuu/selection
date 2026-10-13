"""One explicitly authorized continuation after the preserved R73 pre-lock failure."""

import os
import shlex
import socket
import subprocess
import sys
import traceback

from code.V4.r73_delivery import absent, run_test
from code.V4.r73_frozen_evaluation import (ROOT, WORKSPACE, capture, exclusive, freeze,
    now, read, require_training_terminal, write)


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('Repair continuation must remain CPU-only')
    failure = read(ROOT / 'delivery_failure.json')
    if failure['error_type'] != 'ValueError' or failure['message'] != (
            'The unique selected state must be the complete shared averaged-x '
            'NativeUtilitySelector with original buffers'):
        raise ValueError('Only the preserved, explicitly authorized pre-lock failure may continue')
    for name in ('final_test_lock.json', 'final_test_started.json', 'test_tmux_dispatch.json',
                 'test_launch_receipt.json', 'analysis_started.json', 'analysis_process_receipt.json'):
        if (ROOT / name).exists():
            raise ValueError('Preserve the existing lock or invocation; no automatic retry')
    old_controller = capture(['tmux', 'has-session', '-t', '=r73_delivery'])
    if not absent(old_controller):
        raise ValueError('The original controller must be gone; never replace a live process')
    pane = os.environ['TMUX_PANE']
    identity = capture(['tmux', 'display-message', '-p', '-t', pane,
                        '#{session_name}|#{pane_id}|#{pane_pid}'])
    if identity['returncode'] or not identity['stdout'].startswith('r73_delivery_continuation|'):
        raise ValueError('Use the isolated owned continuation session')
    training_terminal = require_training_terminal()
    exclusive(ROOT / 'delivery_continuation_started.json', dict(utc=now(), pid=os.getpid(),
        host=socket.gethostname(), tmux='r73_delivery_continuation', pane=pane,
        pane_pid=int(identity['stdout'].strip().split('|')[2]), identity=identity,
        command=shlex.join([sys.executable, '-B', '-u', str(ROOT / 'delivery_continue.py')]),
        original_controller_gone=old_controller, original_training_terminal=training_terminal,
        original_failure=failure, repair_receipt=str(ROOT / 'delivery_repair.json'),
        authorization='explicit parent repair and continuation from original pre-lock failure',
        original_controller_main_rerun=False, new_TEST_attempts_allowed=1, analysis_invocations_allowed=1,
        CPU_only=True, automatic_retry=False))
    try:
        freeze()
        test_terminal = run_test()
        if (ROOT / 'analysis_started.json').exists():
            raise ValueError('Preserve the existing analysis invocation; never rerun it')
        command = [sys.executable, '-B', '-u', '-m', 'code.V4.r73_analysis', '--authorize-analysis']
        with (ROOT / 'analysis.log').open('x') as stream:
            result = subprocess.run(command, cwd=WORKSPACE, stdout=stream, stderr=subprocess.STDOUT)
        exclusive(ROOT / 'analysis_process_receipt.json', dict(utc=now(), command=shlex.join(command),
            returncode=result.returncode, invocations=1, CPU_only=True, automatic_retry=False,
            explicit_repair_continuation=True))
        if result.returncode:
            raise RuntimeError('ONCE analysis failed; preserve the invocation without retry')
        receipt = read(ROOT / 'delivery_receipt.json')
        receipt.update(explicit_repair_continuation=True, continuation_pid=os.getpid(),
            original_delivery_failure_preserved=True, original_train_terminal=training_terminal,
            test_terminal=test_terminal, original_controller_main_rerun=False,
            owned_training_and_test_tmux_gone={
                'train': capture(['tmux', 'has-session', '-t', '=r73_train']),
                'test': capture(['tmux', 'has-session', '-t', '=r73_final_test'])},
            controller_test_and_analysis_foreground_settled=True)
        write(ROOT / 'delivery_receipt.json', receipt)
        write(ROOT / 'delivery_current_status.json', dict(stage='complete_after_explicit_prelock_repair',
            utc=now(), new_tests_executed={'A': 0, 'B': 1}, analysis_invocations=1,
            original_failure_preserved=True, parent_reviews_audit_memory_Git_pending=True))
        exclusive(ROOT / 'delivery_continuation_process_receipt.json', dict(utc=now(), returncode=0,
            invocations=1, pid=os.getpid(), test_terminal=test_terminal, analysis_returncode=result.returncode,
            original_failure_preserved=True, foreground_work_settled=True))
        print('[R73 explicit continuation complete] ' + str(receipt['decision']), flush=True)
    except BaseException as error:
        exclusive(ROOT / 'delivery_continuation_failure.json', dict(utc=now(), error_type=type(error).__name__,
            message=str(error), traceback=traceback.format_exc(), automatic_retry=False,
            original_delivery_failure_preserved=True))
        raise


if __name__ == '__main__':
    main()
