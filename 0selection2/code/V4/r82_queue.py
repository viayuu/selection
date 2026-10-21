"""One-shot R82B resource queue, behind the already authorized R81B queue."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shlex
import signal
import socket
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock
import uuid

from . import r81_queue as transport


REPO = transport.REPO
ROOT = REPO / 'code/V4/runs/R82_context_matched_ranking'
PREDECESSOR = transport.ROOT
GROUP = 'B_context_matched_rank_seed2'
ALIAS = 'Requested_R80B_context_matched_residual_ranking'
SESSION = 'r82b_once'
WRAPPER = REPO / 'code/V4/run_v4_r82.sh'
CANDIDATES = transport.CANDIDATES
CONTAINERS = transport.CONTAINERS
CACHE = transport.CACHE
PREPARED = transport.PREPARED
DATA = transport.DATA
INITIAL_STATE_HASH = transport.INITIAL_STATE_HASH
POLL_SECONDS = 60
LOADED_QUEUE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def utc():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text())


def owned(path):
    path = Path(path)
    if path.parent != ROOT or path.suffix != '.json' or not path.name.startswith(('queue', 'launch')):
        raise ValueError(f'Output outside R82 queue ownership: {path}')
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def write(path, value):
    path = owned(path)
    temporary = path.with_name(path.stem + f'_tmp_{os.getpid()}.json')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


exclusive = transport.exclusive
lock_file = transport.lock_file
capture = transport.capture


def source_hashes():
    names = ('r82_queue.py', 'run_v4_r82.sh', 'r82_experiment.py', 'r82_reporting.py',
        'r82_context.py', 'context_matched_objective.py', 'R82_INTERFACE.md',
        'r81_queue.py', 'r81_experiment.py', 'method_auc_objective.py',
        'pairwise_selector.py', 'pairwise_objective.py', 'r77_experiment.py',
        'r80_experiment.py', 'r80_reporting.py')
    paths = [REPO / 'code/V4' / name for name in names]
    paths.append(PREPARED / 'identicalinit_seed2.pt')
    return {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in paths if path.is_file()}


def status(state, **fields):
    record = dict(experiment='R82B', proposal_alias=ALIAS, group=GROUP, state=state,
        phase=state, updated_utc=utc(), tmux=SESSION, controller_pid=os.getpid(),
        controller_host=socket.gethostname(), loaded_queue_sha256=LOADED_QUEUE_SHA256,
        poll_seconds=POLL_SECONDS, continuous_model_monitor=False, **fields)
    write(ROOT / 'queue_status.json', record)
    print(json.dumps({key: record[key] for key in
        ('updated_utc', 'state', 'controller_pid', 'tmux')}, sort_keys=True), flush=True)
    return record


def fresh_run():
    for name in ('last.pt', 'best.pt', 'status.json', 'protocol.json',
            'training_complete.json', 'final_test_lock.json', 'test_started.json'):
        if (ROOT / GROUP / name).exists():
            raise RuntimeError(f'Fresh-only queue refuses existing {name}; no resume')
    if (ROOT / 'queue_claim.json').exists() or (ROOT / f'launch_{GROUP}.json').exists():
        raise RuntimeError('R82B already claimed or launched; never retry implicitly')


def predecessor_priority():
    claim_path = PREDECESSOR / 'queue_claim.json'
    if not claim_path.exists():
        state_path = PREDECESSOR / 'queue_status.json'
        state = read(state_path).get('state') if state_path.exists() else 'not_queued'
        return False, dict(state=state, reason='R81B must claim its first GPU before R82B may select a GPU')
    claim = read(claim_path)
    candidate = CANDIDATES.get(claim.get('candidate'))
    if (claim.get('group') != transport.GROUP or not candidate or not claim.get('token')
            or claim.get('gpu_uuid') != candidate['uuid'] or not claim.get('claimed_utc')):
        return False, dict(state='invalid_predecessor_claim', reason='R81 priority receipt is invalid; fail closed')
    return True, dict(state='claimed', gpu_uuid=claim['gpu_uuid'], candidate=claim['candidate'],
        claim_path=str(claim_path), reason='Earlier R81B already claimed; its UUID stays excluded from this one-shot queue')


def assert_ready():
    ready = read(ROOT / 'queue_ready.json')
    for field in ('main_thread_ready', 'objective_cpu_tests_passed',
            'runner_cpu_tests_passed', 'train_pair_coverage_checked'):
        if ready.get(field) is not True:
            raise RuntimeError(f'Explicit main-thread readiness missing: {field}')
    if not ready.get('evidence') or not ready.get('confirmed_utc'):
        raise RuntimeError('A dated main-thread readiness message is required')
    if ready.get('source_sha256') != source_hashes():
        raise RuntimeError('Code changed after ready; R82B will not launch')
    if ready['source_sha256'].get(str(Path(__file__).resolve())) != LOADED_QUEUE_SHA256:
        raise RuntimeError('Standby loaded an older R82 queue; explicit refresh required')
    for prefix in ('preparation', 'coverage'):
        path = Path(ready[prefix + '_receipt'])
        if hashlib.sha256(path.read_bytes()).hexdigest() != ready[prefix + '_sha256']:
            raise RuntimeError(f'{prefix} receipt changed after ready; no launch')
    return ready


def srun_command(arm, stage, token=None, probe=False):
    candidate = CANDIDATES[arm]
    command = ['srun', f'--jobid={candidate["job"]}', '--overlap', '--exact', '--gres=none',
        '--immediate=5', '-N1', '-n1', f'--nodelist={candidate["node"]}',
        f'--cpus-per-task={1 if probe else 4}', f'--mem={"256M" if probe else "9G"}',
        'bash', str(WRAPPER), stage, '--candidate', arm]
    if token:
        command += ['--token', token]
    return command


def remote_probe(arm):
    return json.loads(capture(srun_command(arm, 'probe-node', probe=True), timeout=55))


def runner_commands(candidate, mode):
    common = ['--root', str(ROOT), '--group', GROUP, '--data-root', str(DATA),
        '--cache-root', str(CACHE), '--gpu-type', candidate['gpu_type'],
        '--large-micro-cap', str(candidate['cap']), '--device', 'cuda:0',
        '--eval-batch-size', '128', '--log-every', '18', '--wandb-mode', mode,
        '--prepared-root', str(PREPARED)]
    return [dict(stage='train_uniqueVALlock_fullTEST', log='train.log', command=[
        sys.executable, '-B', '-u', '-m', 'code.V4.r82_experiment', 'train', *common]),
        dict(stage='twoLIB_and_PDF', log='reporting.log', command=[
            sys.executable, '-B', '-u', '-m', 'code.V4.r82_reporting', '--root', str(ROOT),
            '--group', GROUP, '--device', 'cuda:0'])]


def cli_check():
    from . import r82_experiment as runner
    from . import r82_reporting as reporter
    if (runner.GROUP, runner.BATCH, runner.MAX_UPDATES, runner.WARMUP) != (GROUP, 1024, 49680, 3726):
        raise RuntimeError('Actual runner identity or fixed budget differs from the queue contract')
    rows = []
    for arm, candidate in CANDIDATES.items():
        for step in runner_commands(candidate, 'offline'):
            command = step['command']
            module = command[command.index('-m') + 1]
            parser = runner.parser() if module == 'code.V4.r82_experiment' else reporter.parser()
            parsed = parser.parse_args(command[command.index('-m') + 2:])
            if parsed.group != GROUP or parsed.root != ROOT or parsed.device != 'cuda:0':
                raise RuntimeError('Actual runner/reporter argv does not match queued R82B')
            rows.append(dict(candidate=arm, stage=step['stage'], argv_parsed=True, command=command))
    receipt = dict(checked_utc=utc(), passed=True, rows=rows, CPU_only=True,
        GPU_contexts_created=0, training_started=False, source_sha256=source_hashes())
    write(ROOT / 'queue_cli_check.json', receipt)
    print('Actual R82 train/report CLI: all 8 candidate/stage argv parsed; no training started', flush=True)
    return receipt


def plan():
    return dict(experiment='R82B', proposal_alias=ALIAS, group=GROUP, root=str(ROOT),
        budget=dict(seed=2, batch=1024, successful_updates=49680, warmup_updates=3726,
            cmr_max_weight=.10, eval_batch_size=128, initialization='fresh R77 identicalinit_seed2.pt',
            initialization_trainable_sha256=INITIAL_STATE_HASH,
            peak_lr=1e-4, minimum_lr=1e-6, schedule='successful-update cosine'),
        candidates=CANDIDATES, containers=CONTAINERS, poll_seconds=POLL_SECONDS,
        priority='R81B claims first; R82B excludes that UUID and may claim another idle authorized GPU',
        global_uuid_lock_root=str(transport.GPU_LOCK_ROOT),
        reservation='exclusive UUID flock spans train -> unique VAL lock + full TEST -> both LIBs and PDF',
        existing_jobs=[1465, 1475], new_allocation=False, stop_existing_jobs=False,
        launch_A=False, extra_seeds=False, automatic_retry=False, git_commit=False,
        cpu_per_pipeline=4, requested_memory_per_pipeline='9G',
        cache=str(CACHE), cache_reused=True, copied_files=0,
        idle_rule='old R80 entire pipeline terminal, no live stage, verified allocation/container, no compute processes, <=512MiB',
        readiness='explicit main-thread ready after focused objective/runner tests and TRAIN pair coverage',
        continuous_model_monitor=False, source_sha256=source_hashes(),
        srun_launch={arm: srun_command(arm, 'container-entry', '<ONE_SHOT_TOKEN>') for arm in CANDIDATES},
        argv={arm: runner_commands(candidate, '<online-or-explicit-offline>') for arm, candidate in CANDIDATES.items()})


def standby():
    ROOT.mkdir(parents=True, exist_ok=True)
    with lock_file(owned(ROOT / 'queue_control_lock.json')):
        path = ROOT / 'queue_tmux.json'
        if path.exists():
            previous = read(path)
            if subprocess.run(['tmux', 'has-session', '-t', SESSION], stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL).returncode == 0:
                if capture(['tmux', 'display-message', '-p', '-t', SESSION, '#{pane_dead}']).strip() == '0':
                    print(json.dumps(previous, indent=2))
                    return
            raise RuntimeError('Standby receipt exists without a live pane; no implicit restart')
        fresh_run()
        command = shlex.join(['bash', str(WRAPPER), 'loop'])
        record = dict(created_utc=utc(), state='preparing', tmux=SESSION,
            queue_source_sha256=LOADED_QUEUE_SHA256, command=command,
            main_thread_ready=False, creator_pid=os.getpid(), host=socket.gethostname())
        exclusive(owned(path), record)
        result = subprocess.run(['tmux', 'new-session', '-d', '-s', SESSION, command])
        record['tmux_exit_status'] = result.returncode
        if result.returncode:
            write(path, record)
            raise RuntimeError('tmux creation failed; no automatic retry')
        subprocess.run(['tmux', 'set-option', '-t', SESSION, 'remain-on-exit', 'on'], check=True)
        pane = capture(['tmux', 'display-message', '-p', '-t', SESSION,
            '#{pane_pid}|#{pane_id}']).strip().split('|')
        record.update(pane_pid=int(pane[0]), pane_id=pane[1])
        write(path, record)
        print(json.dumps(record, indent=2))


def arm_queue(args):
    if not (args.ready_evidence and args.objective_tests_passed and args.runner_tests_passed
            and args.pair_coverage_checked):
        raise RuntimeError('Arm requires explicit main-thread evidence, both CPU test confirmations and TRAIN pair coverage')
    preparation = args.preparation_receipt.resolve()
    coverage = args.coverage_receipt.resolve()
    if preparation != ROOT / 'prepared/prepared.json' or coverage != ROOT / 'prepared/pair_coverage.json':
        raise RuntimeError('Use runner-owned preparation and TRAIN-only pair-coverage receipts')
    ready, pair_coverage = read(preparation), read(coverage)
    if (ready.get('experiment') != 'R82' or ready.get('group') != GROUP
            or ready.get('train_only') is not True or ready.get('trainable_sha256') != INITIAL_STATE_HASH):
        raise RuntimeError('R82 preparation identity/TRAIN-only/fresh-initialization mismatch')
    if not isinstance(pair_coverage, dict) or not pair_coverage:
        raise RuntimeError('TRAIN pair coverage receipt must be a nonempty record')
    required = ('r82_experiment.py', 'r82_reporting.py', 'r82_context.py',
        'context_matched_objective.py', 'R82_INTERFACE.md')
    if any(not (REPO / 'code/V4' / name).is_file() for name in required):
        raise RuntimeError('R82 implementation/interface is not complete')
    tests = read(ROOT / 'queue_selftest.json')
    if tests.get('passed') is not True or tests.get('source_sha256') != source_hashes():
        raise RuntimeError('Run focused queue selftests against current source before arm')
    cli = read(ROOT / 'queue_cli_check.json')
    if cli.get('passed') is not True or cli.get('source_sha256') != source_hashes():
        raise RuntimeError('Check actual runner/report CLI against current source before arm')
    fresh_run()
    standby()
    with lock_file(owned(ROOT / 'queue_control_lock.json')):
        exclusive(owned(ROOT / 'queue_ready.json'), dict(confirmed_utc=utc(),
            main_thread_ready=True, objective_cpu_tests_passed=True, runner_cpu_tests_passed=True,
            train_pair_coverage_checked=True, evidence=args.ready_evidence, group=GROUP,
            proposal_alias=ALIAS, preparation_receipt=str(preparation),
            preparation_sha256=hashlib.sha256(preparation.read_bytes()).hexdigest(),
            coverage_receipt=str(coverage), coverage_sha256=hashlib.sha256(coverage.read_bytes()).hexdigest(),
            source_sha256=source_hashes()))
    print(f'ARMED R82B only: tmux={SESSION}; R81 priority; no GPU occupied until resource gates pass', flush=True)


def claim_record(arm, probe, predecessor):
    return dict(claimed_utc=utc(), group=GROUP, proposal_alias=ALIAS, token=uuid.uuid4().hex,
        candidate=arm, gpu_uuid=CANDIDATES[arm]['uuid'], selected_cpus=probe['selected_cpus'],
        resource_proof=probe, predecessor=predecessor, controller_pid=os.getpid(),
        controller_host=socket.gethostname(), tmux=SESSION,
        uuid_lock=str(transport.gpu_lock_path(CANDIDATES[arm]['uuid'])),
        reservation='entire train + full TEST + both LIBs and PDF; one fresh attempt')


def claim_for(arm, token):
    claim = read(ROOT / 'queue_claim.json')
    if claim.get('group') != GROUP or claim.get('candidate') != arm or not token or claim.get('token') != token:
        raise RuntimeError('Missing or mismatched one-shot R82 claim')
    return claim


def run_queue():
    ROOT.mkdir(parents=True, exist_ok=True)
    with lock_file(owned(ROOT / 'queue_lock.json')):
        fresh_run()
        previous = None
        while True:
            if not (ROOT / 'queue_ready.json').exists():
                if previous != 'preparing':
                    status('preparing', main_thread_ready=False, GPU_training_started=False,
                        reason='Waiting for main-thread objective/runner tests and TRAIN pair coverage; no remote probe')
                    previous = 'preparing'
                time.sleep(POLL_SECONDS)
                continue
            try:
                assert_ready()
            except (OSError, ValueError, KeyError, RuntimeError) as error:
                reason = str(error)
                if previous != reason:
                    status('preparing', GPU_training_started=False, reason=reason)
                    previous = reason
                time.sleep(POLL_SECONDS)
                continue
            priority, predecessor = predecessor_priority()
            if not priority:
                status('pending_R81_priority', main_thread_ready=True, GPU_training_started=False,
                    predecessor=predecessor, reason=predecessor['reason'])
                time.sleep(POLL_SECONDS)
                continue
            candidates = {}
            for arm, candidate in CANDIDATES.items():
                if candidate['uuid'] == predecessor['gpu_uuid']:
                    candidates[arm] = dict(eligible=False, reason='UUID belongs to earlier R81B claim')
                    continue
                selected_probe = None
                try:
                    receipt = transport.old_receipt(arm)
                    terminal, reason = transport.old_terminal(receipt, candidate)
                    item = dict(eligible=False, reason=reason, old_state=receipt.get('state'))
                    if terminal:
                        probe = remote_probe(arm)
                        eligible, reason = transport.usable(arm, transport.old_receipt(arm), probe)
                        item.update(eligible=eligible, reason=reason, resource_proof=probe)
                        if eligible:
                            selected_probe = probe
                    candidates[arm] = item
                except (OSError, ValueError, KeyError, RuntimeError, subprocess.TimeoutExpired) as error:
                    candidates[arm] = dict(eligible=False, reason=f'{type(error).__name__}: {error}')
                if selected_probe is not None:
                    try:
                        with lock_file(transport.gpu_lock_path(candidate['uuid'])):
                            assert_ready()
                            fresh_run()
                            okay, current_predecessor = predecessor_priority()
                            if not okay or current_predecessor['gpu_uuid'] == candidate['uuid']:
                                raise RuntimeError('Earlier R81 priority changed before claim; no launch')
                            claim = claim_record(arm, selected_probe, current_predecessor)
                            exclusive(owned(ROOT / 'queue_claim.json'), claim)
                            return launch_claim(claim)
                    except BlockingIOError:
                        candidates[arm] = dict(eligible=False, reason='Another queue owns this global UUID lock')
            status('pending', main_thread_ready=True, GPU_training_started=False,
                predecessor=predecessor, candidates=candidates,
                reason='Waiting for another fully finished R80 pipeline and genuinely idle authorized GPU')
            previous = 'pending'
            time.sleep(POLL_SECONDS)


def launch_claim(claim):
    arm, token = claim['candidate'], claim['token']
    command = srun_command(arm, 'container-entry', token)
    receipt = dict(created_utc=utc(), state='launching', tmux=SESSION, group=GROUP,
        candidate=arm, gpu_uuid=claim['gpu_uuid'], controller_pid=os.getpid(), command=command,
        uuid_lock=claim['uuid_lock'])
    path = ROOT / 'launch_controller.json'
    exclusive(owned(path), receipt)
    status('launching', claim=claim, command=command, GPU_training_started=False)
    process = None
    try:
        process = subprocess.Popen(command)
        receipt.update(srun_pid=process.pid, state='running')
        write(path, receipt)
        code = process.wait()
        worker_path = ROOT / f'launch_{GROUP}.json'
        worker = read(worker_path) if worker_path.exists() else None
        if code == 0 and (not worker or worker.get('state') != 'complete'):
            raise RuntimeError('Transport exited zero without a completed worker receipt')
        receipt.update(state='complete' if code == 0 else 'failed', exit_status=code, finished_utc=utc())
    except BaseException as error:
        if process is not None and process.poll() is None:
            process.wait()
        receipt.update(state='failed', exit_status=1, error=f'{type(error).__name__}: {error}', finished_utc=utc())
        write(path, receipt)
        status('failed', launch=receipt, automatic_retry=False)
        raise
    write(path, receipt)
    status(receipt['state'], launch=receipt, worker=worker, automatic_retry=False,
        reason='One-shot pipeline exited; no extra group, seed, retry or monitoring task')
    return code


def container_entry(arm, token):
    claim = claim_for(arm, token)
    assert_ready()
    okay, priority = predecessor_priority()
    if not okay or priority['gpu_uuid'] == CANDIDATES[arm]['uuid']:
        raise RuntimeError('R81 priority does not permit this R82 launch')
    probe = transport.node_probe(arm)
    okay, reason = transport.usable(arm, transport.old_receipt(arm), probe)
    if not okay:
        raise RuntimeError(f'Resource recheck failed: {reason}')
    partner = {cpu for other, rows in probe['processes'].items() if other != arm
        for row in rows for cpu in row['cpus']}
    selected = set(claim['selected_cpus'])
    if not selected <= set(probe['allowed_cpus']) or selected & partner:
        raise RuntimeError('Claimed CPU affinity now overlaps partner or exceeds allocation')
    candidate = CANDIDATES[arm]
    name, identity = CONTAINERS[candidate['node']]
    command = ['sudo', '-n', '/usr/bin/ai_docker', 'exec', '-u', str(os.getuid()),
        '-e', f'CUDA_VISIBLE_DEVICES={candidate["uuid"]}',
        '-e', f'SLURM_JOB_ID={candidate["job"]}', '-e', 'SLURM_STEP_ID=existing_allocation_container',
        '-e', f'R82_QUEUE_TOKEN={token}',
        '-e', 'R82_ASSIGNED_CPUS=' + ','.join(map(str, sorted(selected))),
        name, 'bash', str(WRAPPER), 'container-worker', '--candidate', arm, '--token', token]
    exclusive(owned(ROOT / 'launch_container_entry.json'), dict(started_utc=utc(), pid=os.getpid(),
        host=socket.gethostname(), job=candidate['job'], step=os.environ.get('SLURM_STEP_ID'),
        container_name=name, container_id=identity, resource_recheck=probe, command=command))
    return subprocess.run(command, check=False).returncode


def container_worker(arm, token):
    claim = claim_for(arm, token)
    candidate = CANDIDATES[arm]
    identity = CONTAINERS[candidate['node']][1]
    path = ROOT / f'launch_{GROUP}.json'
    receipt = dict(state='preflight', started_utc=utc(), worker_pid=os.getpid(),
        worker_host=socket.gethostname(), tmux=SESSION, group=GROUP,
        proposal_alias=ALIAS, candidate=arm, gpu_uuid=candidate['uuid'], stages=[])
    with lock_file(owned(ROOT / 'queue_worker_lock.json')):
        exclusive(owned(path), receipt)
        process = None
        try:
            assert_ready()
            if identity not in Path('/proc/self/cgroup').read_text() or socket.gethostname() != identity[:12]:
                raise RuntimeError('Worker is outside the identity-locked existing allocation container')
            if os.environ.get('SLURM_JOB_ID') != str(candidate['job']) or os.environ.get('R82_QUEUE_TOKEN') != token:
                raise RuntimeError('Worker allocation/token mismatch')
            if os.environ.get('CUDA_VISIBLE_DEVICES') != candidate['uuid']:
                raise RuntimeError('Exactly one authorized UUID must be bound to CUDA_VISIBLE_DEVICES')
            selected = set(claim['selected_cpus'])
            if not selected <= set(os.sched_getaffinity(0)):
                raise RuntimeError('Claimed CPU affinity is outside the existing container cpuset')
            os.sched_setaffinity(0, selected)
            receipt['container'] = dict(id=identity, cgroup=Path('/proc/self/cgroup').read_text(),
                assigned_cpus=sorted(os.sched_getaffinity(0)), job=candidate['job'])
            okay, reason = transport.old_terminal(transport.old_receipt(arm), candidate)
            if not okay or transport.process_inventory(candidate['node']).get(arm):
                raise RuntimeError(f'Previous pipeline changed or still has a live process: {reason}')
            gpu = transport.gpu_inventory(candidate)
            if gpu['compute_processes'] or gpu['memory_used_mib'] > transport.MEMORY_MIB:
                raise RuntimeError('GPU occupied at final container check; no launch')
            receipt['gpu_preflight'] = gpu
            receipt['cache'] = transport.cache_inventory()
            mode, receipt['wandb'] = transport.wandb_mode()
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=candidate['uuid'], R82_APPROVED='1',
                WANDB_MODE=mode, SLURM_JOB_ID=str(candidate['job']))
            directory = ROOT / GROUP
            directory.mkdir(parents=True, exist_ok=True)
            for name in ('last.pt', 'best.pt', 'status.json', 'protocol.json'):
                if (directory / name).exists():
                    raise RuntimeError(f'Fresh-only worker refuses existing {name}')

            def forward(signum, _frame):
                if process is not None and process.poll() is None:
                    os.killpg(process.pid, signum)

            signal.signal(signal.SIGTERM, forward)
            signal.signal(signal.SIGINT, forward)
            for step in runner_commands(candidate, mode):
                assert_ready()
                if not receipt['stages']:
                    gpu = transport.gpu_inventory(candidate)
                    if gpu['compute_processes'] or gpu['memory_used_mib'] > transport.MEMORY_MIB:
                        raise RuntimeError('GPU became occupied before the first training child')
                record = dict(stage=step['stage'], command=step['command'], state='starting',
                    log=str(directory / step['log']), started_utc=utc())
                receipt['stages'].append(record)
                receipt.update(state='running', phase=step['stage'])
                write(path, receipt)
                with (directory / step['log']).open('a', buffering=1) as output:
                    output.write(f'\n[R82B queue] {utc()} {shlex.join(step["command"])}\n')
                    process = subprocess.Popen(step['command'], cwd=REPO, env=env,
                        stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
                    record.update(pid=process.pid, state='running')
                    write(path, receipt)
                    status('running', pipeline_stage=step['stage'], worker_pid=os.getpid(),
                        child_pid=process.pid, gpu_uuid=candidate['uuid'], candidate=arm,
                        assigned_cpus=sorted(selected), wandb=receipt['wandb'], log=record['log'],
                        GPU_training_started=True, reservation_released=False)
                    code = process.wait()
                    output.write(f'\n[R82B queue] {utc()} stage={step["stage"]} exit_status={code}\n')
                record.update(state='complete' if code == 0 else 'failed', exit_status=code, finished_utc=utc())
                write(path, receipt)
                if code:
                    receipt.update(state='failed', failed_stage=step['stage'], exit_status=code,
                        finished_utc=utc(), reservation=transport.final_reservation(candidate))
                    write(path, receipt)
                    return code
            receipt.update(state='complete', phase='complete', exit_status=0, finished_utc=utc(),
                reservation=transport.final_reservation(candidate))
            write(path, receipt)
            return 0
        except BaseException as error:
            if process is not None and process.poll() is None:
                process.wait()
            receipt.update(state='failed', exit_status=1, finished_utc=utc(),
                error=f'{type(error).__name__}: {error}', automatic_retry=False)
            write(path, receipt)
            raise


def selftests():
    class QueueTests(unittest.TestCase):
        def test_only_fixed_fresh_B(self):
            value = plan()
            self.assertEqual(value['group'], GROUP)
            self.assertEqual((value['budget']['batch'], value['budget']['successful_updates'],
                value['budget']['warmup_updates'], value['budget']['cmr_max_weight']), (1024, 49680, 3726, .10))
            self.assertFalse(value['launch_A'])
            self.assertFalse(value['automatic_retry'])
            self.assertFalse(value['new_allocation'])

        def test_existing_container_transport_and_runner_contract(self):
            for arm, candidate in CANDIDATES.items():
                command = srun_command(arm, 'container-entry', 'token')
                self.assertIn(f'--jobid={candidate["job"]}', command)
                self.assertIn('--gres=none', command)
                self.assertIn('--cpus-per-task=4', command)
                self.assertIn('--mem=9G', command)
                self.assertNotIn('ssh', command)
                commands = runner_commands(candidate, 'offline')
                self.assertEqual(len(commands), 2)
                self.assertNotIn('--resume', commands[0]['command'])
                self.assertIn('train', commands[0]['command'])
                self.assertIn(GROUP, commands[0]['command'])
                self.assertEqual(commands[0]['command'][-1], str(PREPARED))
                self.assertIn('cuda:0', commands[1]['command'])

        def test_R81_priority_requires_a_valid_first_claim(self):
            with tempfile.TemporaryDirectory(prefix='r82_queue_cpu_') as directory:
                with mock.patch.object(sys.modules[__name__], 'PREDECESSOR', Path(directory)):
                    self.assertFalse(predecessor_priority()[0])
                    path = Path(directory) / 'queue_claim.json'
                    exclusive(path, dict(group=transport.GROUP, candidate='B_batch512',
                        gpu_uuid=CANDIDATES['B_batch512']['uuid'], token='prior', claimed_utc='now'))
                    self.assertTrue(predecessor_priority()[0])
                    self.assertEqual(predecessor_priority()[1]['gpu_uuid'], CANDIDATES['B_batch512']['uuid'])
                    path.unlink()
                    exclusive(path, dict(group=GROUP, candidate='B_batch512', token='wrong'))
                    self.assertFalse(predecessor_priority()[0])

        def test_R81_and_R82_share_the_same_uuid_lock(self):
            with tempfile.TemporaryDirectory(prefix='r82_queue_cpu_') as directory:
                with mock.patch.object(transport, 'GPU_LOCK_ROOT', Path(directory)):
                    path = transport.gpu_lock_path(CANDIDATES['D_code_embedding']['uuid'])
                    with lock_file(path):
                        with self.assertRaises(BlockingIOError):
                            with transport.lock_file(transport.gpu_lock_path(CANDIDATES['D_code_embedding']['uuid'])):
                                self.fail('Other queue acquired the reserved UUID')

        def test_resource_banner_and_fail_closed_parser(self):
            banner = 'Distribute Task Container.\ntask_id:1475\n'
            row = f'0, {CANDIDATES["B_batch512"]["uuid"]}, NVIDIA GeForce RTX 4090, 2, 24564, 0\n'
            self.assertEqual(len(transport.nvml_csv(banner + row, 6)), 1)
            self.assertEqual(transport.nvml_csv(banner, 4), [])
            with self.assertRaises(RuntimeError):
                transport.nvml_csv('unrecognized error\n' + row, 6)

        def test_fresh_only_and_duplicate_claim(self):
            with tempfile.TemporaryDirectory(prefix='r82_queue_cpu_') as directory:
                with mock.patch.object(sys.modules[__name__], 'ROOT', Path(directory)):
                    fresh_run()
                    exclusive(Path(directory) / 'queue_claim.json', dict(token='first'))
                    with self.assertRaises(RuntimeError):
                        fresh_run()
                    with self.assertRaises(FileExistsError):
                        exclusive(Path(directory) / 'queue_claim.json', dict(token='second'))

        def test_readiness_and_priority_precede_resource_queries(self):
            class StopLoop(Exception):
                pass
            module = sys.modules[__name__]
            with tempfile.TemporaryDirectory(prefix='r82_queue_cpu_') as directory:
                root = Path(directory)
                with mock.patch.object(module, 'ROOT', root), mock.patch.object(module, 'status'), \
                        mock.patch.object(module.time, 'sleep', side_effect=StopLoop), \
                        mock.patch.object(module, 'remote_probe') as probe:
                    with self.assertRaises(StopLoop):
                        run_queue()
                    self.assertEqual(probe.call_count, 0)
                    exclusive(root / 'queue_ready.json', {})
                    with mock.patch.object(module, 'assert_ready'), \
                            mock.patch.object(module, 'predecessor_priority', return_value=(False, dict(reason='R81 first'))):
                        with self.assertRaises(StopLoop):
                            run_queue()
                    self.assertEqual(probe.call_count, 0)

    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(QueueTests))
    write(ROOT / 'queue_selftest.json', dict(checked_utc=utc(), passed=result.wasSuccessful(),
        tests_run=result.testsRun, errors=len(result.errors), failures=len(result.failures),
        CPU_only=True, training_subprocesses_started=0, GPU_contexts_created=0, source_sha256=source_hashes()))
    return 0 if result.wasSuccessful() else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('dry-run', 'selftest', 'cli-check', 'standby', 'arm', 'status',
        'loop', 'probe-node', 'container-entry', 'container-worker'))
    parser.add_argument('--candidate', choices=tuple(CANDIDATES))
    parser.add_argument('--token')
    parser.add_argument('--ready-evidence')
    parser.add_argument('--objective-tests-passed', action='store_true')
    parser.add_argument('--runner-tests-passed', action='store_true')
    parser.add_argument('--pair-coverage-checked', action='store_true')
    parser.add_argument('--preparation-receipt', type=Path, default=ROOT / 'prepared/prepared.json')
    parser.add_argument('--coverage-receipt', type=Path, default=ROOT / 'prepared/pair_coverage.json')
    args = parser.parse_args(argv)
    if args.command in ('probe-node', 'container-entry', 'container-worker') and not args.candidate:
        parser.error('Internal allocation commands need --candidate')
    if args.command == 'dry-run':
        value = plan()
        write(ROOT / 'queue_plan.json', value)
        print(json.dumps(value, indent=2))
    elif args.command == 'selftest':
        return selftests()
    elif args.command == 'cli-check':
        cli_check()
    elif args.command == 'standby':
        standby()
    elif args.command == 'arm':
        arm_queue(args)
    elif args.command == 'loop':
        try:
            return run_queue()
        except BlockingIOError:
            print('Another R82 one-shot queue already owns the controller lock; no duplicate launch', file=sys.stderr)
            return 2
        except BaseException as error:
            status('failed', error=f'{type(error).__name__}: {error}', automatic_retry=False)
            raise
    elif args.command == 'probe-node':
        print(json.dumps(transport.node_probe(args.candidate)))
    elif args.command == 'container-entry':
        return container_entry(args.candidate, args.token)
    elif args.command == 'container-worker':
        return container_worker(args.candidate, args.token)
    else:
        print(json.dumps(read(ROOT / 'queue_status.json') if (ROOT / 'queue_status.json').exists()
            else dict(state='preparing', tmux=SESSION, group=GROUP, main_thread_ready=False), indent=2))
    return 0


if __name__ == '__main__':
    sys.exit(main())
