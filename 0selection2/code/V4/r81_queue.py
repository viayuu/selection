"""One-shot R81B queue in the four already authorized R80 GPU slots."""

import argparse
from contextlib import contextmanager
import copy
import csv
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
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


REPO = Path('/public/home/shiys/0selection2')
ROOT = REPO / 'code/V4/runs/R81_method_conditioned_auc'
OLD_ROOT = REPO / 'code/V4/runs/R80_training_and_code_ablation'
PREPARED = REPO / 'code/V4/runs/R77_sample_size_batch1024'
SOURCE = REPO / 'code/V4/runs/R76_R45A_single_reld_mt/cache'
CACHE = Path('/tmp/shiys_R76_R45A_single_reld_mt_cache')
DATA = Path('/public/home/shiys/selection_data_new_only_B_filtered_single_reld_mt_seed2_20261014')
WRAPPER = REPO / 'code/V4/run_v4_r81.sh'
GROUP = 'B_cross_instance_rank_seed2'
SESSION = 'r81b_once'
POLL_SECONDS = 60
MEMORY_MIB = 512
INITIAL_STATE_HASH = 'dad87d8598f7560e6f170ef048ba1bf515edeb03cef7ec42e16839a36af4ef93'
LOADED_QUEUE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
CANDIDATES = {
    'A_ce_only': dict(job=1465, node='gpu03', index=0, gpu_type='3090', cap=384,
        uuid='GPU-cb24f168-9f18-f0d5-6f4a-6d2406958f50'),
    'C_lr3e4': dict(job=1465, node='gpu03', index=1, gpu_type='3090', cap=384,
        uuid='GPU-cb626bdb-4319-2f5e-b91f-17fcaa44a83d'),
    'B_batch512': dict(job=1475, node='gpu05', index=0, gpu_type='4090', cap=352,
        uuid='GPU-71479e7e-f8bd-e430-66ed-f8db9f0d47be'),
    'D_code_embedding': dict(job=1475, node='gpu05', index=1, gpu_type='4090', cap=352,
        uuid='GPU-597c682a-4b7f-7609-80a6-8ced791f5c5d'),
}
CONTAINERS = {
    'gpu03': ('1465_gpu03', 'b8ced5d8a8cdafdc5006f827a0d36c319c61056a65316af4ff9937453fa0635c'),
    'gpu05': ('1475_gpu05', '76045731f6b453443fe18a9bce671599bdbf17c2713b8edcdd090925eea14bf6'),
}
GPU_LOCK_ROOT = ROOT


def utc():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text())


def owned(path):
    path = Path(path)
    if path.parent != ROOT or not (path.suffix == '.json'
            and path.name.startswith(('queue', 'launch'))):
        raise ValueError(f'Queue output outside its owned scope: {path}')
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def write(path, value):
    path = owned(path)
    temporary = path.with_name(path.stem + f'_tmp_{os.getpid()}.json')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def exclusive(path, value):
    with Path(path).open('x') as output:
        json.dump(value, output, indent=2, allow_nan=False)
        output.write('\n')
        output.flush()
        os.fsync(output.fileno())


@contextmanager
def lock_file(path):
    with Path(path).open('a+') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if handle.tell() == 0:
            handle.write('{}\n')
            handle.flush()
        yield handle


def gpu_lock_path(gpu_uuid):
    if gpu_uuid not in {item['uuid'] for item in CANDIDATES.values()}:
        raise ValueError('GPU reservation requires an authorized UUID')
    GPU_LOCK_ROOT.mkdir(parents=True, exist_ok=True)
    return GPU_LOCK_ROOT / f'queue_gpu_{gpu_uuid}.json'


def capture(command, timeout=25):
    result = subprocess.run(command, text=True, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, timeout=timeout, check=False)
    if result.returncode:
        raise RuntimeError(f'{shlex.join(command)}: exit={result.returncode}: {result.stderr[-2000:]}')
    return result.stdout


def status(state, **values):
    record = dict(experiment='R81B', proposal_alias='requestedR80B', group=GROUP,
        state=state, phase=state, updated_utc=utc(), tmux=SESSION,
        controller_pid=os.getpid(), controller_host=socket.gethostname(),
        loaded_queue_sha256=LOADED_QUEUE_SHA256,
        poll_seconds=POLL_SECONDS, continuous_model_monitor=False, **values)
    write(ROOT / 'queue_status.json', record)
    print(json.dumps({key: record[key] for key in (
        'updated_utc', 'state', 'phase', 'controller_pid', 'tmux')}, sort_keys=True), flush=True)
    return record


def source_hashes():
    names = ('r81_queue.py', 'run_v4_r81.sh', 'r81_experiment.py', 'r81_reporting.py',
        'method_auc_objective.py', 'pairwise_selector.py', 'pairwise_objective.py',
        'r80_experiment.py', 'r80_reporting.py', 'r77_experiment.py', 'R81_INTERFACE.md')
    paths = [REPO / 'code/V4' / name for name in names]
    paths += [PREPARED / 'identicalinit_seed2.pt']
    return {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in paths if path.is_file()}


def assert_ready():
    gate = read(ROOT / 'queue_ready.json')
    for key in ('main_thread_ready', 'objective_cpu_tests_passed', 'runner_cpu_tests_passed'):
        if gate.get(key) is not True:
            raise RuntimeError(f'Missing explicit main-thread gate: {key}')
    if not gate.get('evidence') or not gate.get('confirmed_utc'):
        raise RuntimeError('Ready gate needs dated main-thread evidence')
    if gate.get('source_sha256') != source_hashes():
        raise RuntimeError('Sources changed since the main-thread ready gate; no launch')
    if gate['source_sha256'].get(str(Path(__file__).resolve())) != LOADED_QUEUE_SHA256:
        raise RuntimeError('Standby process loaded an older queue version; no launch')
    preparation = Path(gate['preparation_receipt'])
    if hashlib.sha256(preparation.read_bytes()).hexdigest() != gate['preparation_sha256']:
        raise RuntimeError('Preparation receipt changed after ready; no launch')
    return gate


def fresh_run():
    directory = ROOT / GROUP
    for name in ('last.pt', 'best.pt', 'status.json', 'training_complete.json',
            'protocol.json', 'final_test_lock.json', 'test_started.json'):
        if (directory / name).exists():
            raise RuntimeError(f'{name} already exists; no implicit restart or resume')
    if (ROOT / 'queue_claim.json').exists() or (ROOT / f'launch_{GROUP}.json').exists():
        raise RuntimeError('Existing claim or worker receipt: this queue is strictly one-shot')


def old_terminal(receipt, candidate):
    if receipt.get('group', {}).get('uuid') != candidate['uuid']:
        return False, 'R80 receipt GPU identity missing or mismatched'
    if receipt.get('state') not in ('complete', 'failed') or not receipt.get('finished_utc'):
        return False, 'R80 full pipeline is not terminal (train_complete alone never qualifies)'
    if receipt['state'] == 'complete':
        stages = receipt.get('stages', [])
        expected = ['train', 'test', 'LIB_and_PDF_report']
        if [stage.get('stage') for stage in stages] != expected:
            return False, 'R80 complete receipt lacks full TEST + LIB/PDF stages'
        if any(stage.get('state') != 'complete' or stage.get('exit_status') != 0 for stage in stages):
            return False, 'R80 complete receipt has unfinished or failed stages'
    elif receipt.get('exit_status') in (None, 0):
        return False, 'R80 failed receipt lacks a nonzero exit status'
    return True, f'R80 pipeline {receipt["state"]}; never restarted by this queue'


def old_receipt(arm):
    return read(OLD_ROOT / f'launch_{arm}.json')


def cpulist(value):
    result = set()
    for part in value.split(','):
        if not part:
            continue
        numbers = part.split('-')
        start, end = int(numbers[0]), int(numbers[-1])
        if len(numbers) > 2 or start < 0 or start > end:
            raise ValueError('Invalid CPU list')
        result.update(range(start, end + 1))
    return result


def allocation(candidate):
    text = capture(['scontrol', 'show', 'job', '--details', '--oneliner', str(candidate['job'])])
    fields = dict(re.findall(r'(?:^|\s)([A-Za-z][A-Za-z0-9_/]*?)=([^\s]+)', text))
    if fields.get('JobState') != 'RUNNING' or fields.get('NodeList') != candidate['node']:
        raise RuntimeError('Authorized allocation is no longer RUNNING on its original node')
    if fields.get('UserId') != f'{os.environ.get("USER", "shiys")}({os.getuid()})':
        raise RuntimeError('Allocation owner differs from current user')
    if fields.get('NumCPUs') != '10' or fields.get('MinMemoryNode') != '20G':
        raise RuntimeError('Existing allocation CPU/memory contract changed')
    if fields.get('JOB_GRES') != 'gpu:2':
        raise RuntimeError('Existing two-GPU allocation contract changed')
    cpus = sorted(cpulist(fields['CPU_IDs']))
    if len(cpus) != 10:
        raise RuntimeError('Cannot prove all ten allowed allocation CPUs')
    return dict(job=candidate['job'], state='RUNNING', node=candidate['node'],
        cpus=cpus, cpu_limit=10, memory_bytes=20 * 2**30, gpu_count=2)


def process_inventory(node):
    result = {arm: [] for arm, candidate in CANDIDATES.items() if candidate['node'] == node}
    modules = {'code.V4.r80_experiment', 'code.V4.r80_reporting', 'code.V4.r80_launch'}
    queued_modules = {'code.V4.r81_experiment', 'code.V4.r81_reporting',
        'code.V4.r82_experiment', 'code.V4.r82_reporting'}
    for path in Path('/proc').glob('[0-9]*'):
        try:
            if path.stat().st_uid != os.getuid():
                continue
            command = path.joinpath('cmdline').read_bytes().decode(errors='replace').split('\0')
            queued = bool(queued_modules.intersection(command))
            if not queued and not modules.intersection(command) and str(REPO / 'code/V4/run_v4_r80.sh') not in command:
                continue
            details = dict(line.split(':', 1) for line in path.joinpath('status').read_text().splitlines() if ':' in line)
            if details.get('State', '').strip().startswith(('Z', 'X')):
                continue
            environment = {}
            if queued:
                environment = dict(entry.split('=', 1) for entry in
                    path.joinpath('environ').read_bytes().decode(errors='replace').split('\0') if '=' in entry)
            for arm in result:
                if ((not queued and (arm in command or f'--arm={arm}' in command))
                        or (queued and environment.get('CUDA_VISIBLE_DEVICES') == CANDIDATES[arm]['uuid'])):
                    result[arm].append(dict(pid=int(path.name), command=command,
                        namespace_pids=[int(value) for value in details.get('NSpid', path.name).split()],
                        cpus=sorted(os.sched_getaffinity(int(path.name)))))
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
    return result


def container_inspect(candidate):
    name, identity = CONTAINERS[candidate['node']]
    rows = json.loads(capture(['sudo', '-n', '/usr/bin/ai_docker', 'inspect', name]))
    if len(rows) != 1 or rows[0].get('Id') != identity or rows[0].get('State', {}).get('Running') is not True:
        raise RuntimeError('Allocation-owned container ID changed or container is no longer running')
    info = rows[0]
    limit = int(info.get('HostConfig', {}).get('Memory', 0))
    if not 0 < limit <= 20 * 2**30:
        raise RuntimeError('Cannot prove existing container memory cap <=20 GiB')
    return dict(name=name, id=identity, memory_limit_bytes=limit,
        allowed_cpus=sorted(os.sched_getaffinity(int(info['State']['Pid']))))


def nvml_csv(raw, width):
    # The platform wrapper prefixes exec output, but not native nvidia-smi output.
    clean = re.sub(r'\ADistribute Task Container\.\r?\ntask_id:\d+\r?\n', '', raw)
    rows = list(csv.reader(clean.splitlines(), skipinitialspace=True))
    rows = [row for row in rows if row]
    if any(len(row) != width for row in rows):
        raise RuntimeError('NVML returned malformed output after the known container banner')
    return rows


def gpu_inventory(candidate, container=None):
    prefix = []
    if container is not None:
        name, identity = CONTAINERS[candidate['node']]
        if (container.get('name'), container.get('id')) != (name, identity):
            raise RuntimeError('NVML query requires the verified allocation-owned container')
        prefix = ['sudo', '-n', '/usr/bin/ai_docker', 'exec', '-u', str(os.getuid()), identity]
    devices = capture(prefix + ['nvidia-smi', f'--id={candidate["uuid"]}',
        '--query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu',
        '--format=csv,noheader,nounits'])
    rows = nvml_csv(devices, 6)
    if len(rows) != 1 or len(rows[0]) != 6:
        raise RuntimeError('NVML GPU identity query failed')
    row = rows[0]
    if row[1] != candidate['uuid'] or candidate['gpu_type'] not in row[2]:
        raise RuntimeError('Assigned GPU UUID/type mismatch')
    apps = capture(prefix + ['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory',
        '--format=csv,noheader,nounits'])
    app_rows = nvml_csv(apps, 4)
    if any(not item[0].startswith('GPU-') or not item[1].isdigit() for item in app_rows):
        raise RuntimeError('NVML returned an invalid compute-process record')
    processes = [item for item in app_rows if item[0] == candidate['uuid']]
    return dict(uuid=row[1], index=int(row[0]), nvml_index=int(row[0]),
        authorized_logical_index=candidate['index'], identity_rule='exact UUID + hardware type; ordinal is informational',
        name=row[2], memory_used_mib=int(row[3]),
        memory_total_mib=int(row[4]), utilization_pct=int(row[5]), compute_processes=processes,
        query_transport='verified_allocation_container_exec' if prefix else 'inside_allocation_container',
        api='NVML via nvidia-smi; no CUDA context or training probe')


def choose_cpus(arm, allowed, processes, baseline=None):
    occupied = {cpu for other, rows in processes.items() if other != arm
        for row in rows for cpu in row['cpus']}
    available = set(allowed) - occupied
    preferred = set((baseline or {}).get('cpus_by_arm', {}).get(arm, [])) & available
    chosen = sorted(preferred)[:4]
    for cpu in sorted(available - set(chosen)):
        if len(chosen) == 4:
            break
        chosen.append(cpu)
    if len(chosen) != 4:
        raise RuntimeError('Fewer than four allocation/container CPUs disjoint from live R80 partner')
    return chosen


def node_probe(arm, baseline_only=False):
    candidate = CANDIDATES[arm]
    if socket.gethostname().split('.')[0] != candidate['node'] or os.environ.get('SLURM_JOB_ID') != str(candidate['job']):
        raise RuntimeError('Node probe/entry must run in the existing allocation CPU step')
    job = allocation(candidate)
    container = container_inspect(candidate)
    processes = process_inventory(candidate['node'])
    allowed = sorted(set(job['cpus']) & set(container['allowed_cpus']))
    result = dict(captured_utc=utc(), candidate=arm, allocation=job, container=container,
        processes=processes, allowed_cpus=allowed, cpu_only=baseline_only)
    if not baseline_only:
        baseline_path = ROOT / 'queue_cpu_baseline.json'
        baseline = read(baseline_path) if baseline_path.exists() else {}
        result.update(gpu=gpu_inventory(candidate, container),
            selected_cpus=choose_cpus(arm, allowed, processes, baseline))
    return result


def usable(arm, receipt, probe):
    candidate = CANDIDATES[arm]
    okay, reason = old_terminal(receipt, candidate)
    if not okay:
        return False, reason
    if probe['allocation']['state'] != 'RUNNING':
        return False, 'Allocation no longer RUNNING'
    if probe['container']['id'] != CONTAINERS[candidate['node']][1]:
        return False, 'Container identity changed'
    if probe['processes'].get(arm):
        return False, 'Old R80 worker/stage still alive despite terminal receipt'
    if probe['gpu']['compute_processes'] or probe['gpu']['memory_used_mib'] > MEMORY_MIB:
        return False, 'NVML compute process present or GPU memory above 512 MiB'
    selected = set(probe['selected_cpus'])
    occupied = {cpu for other, rows in probe['processes'].items() if other != arm
        for row in rows for cpu in row['cpus']}
    if len(selected) != 4 or not selected <= set(probe['allowed_cpus']) or selected & occupied:
        return False, 'CPU affinity overlaps live partner or exceeds allocation/container cpuset'
    return True, reason


def srun_command(arm, stage, token=None, probe=False):
    candidate = CANDIDATES[arm]
    command = ['srun', f'--jobid={candidate["job"]}', '--overlap', '--exact', '--gres=none',
        '--immediate=5', '-N1', '-n1', f'--nodelist={candidate["node"]}',
        f'--cpus-per-task={1 if probe else 4}', f'--mem={"256M" if probe else "9G"}',
        'bash', str(WRAPPER), stage, '--candidate', arm]
    if token:
        command += ['--token', token]
    return command


def remote_probe(arm, baseline_only=False):
    command = srun_command(arm, 'probe-node', probe=True)
    if baseline_only:
        command.append('--baseline-only')
    output = capture(command, timeout=55)
    return json.loads(output)


def runner_commands(candidate, mode):
    common = ['--root', str(ROOT), '--group', GROUP, '--data-root', str(DATA),
        '--cache-root', str(CACHE), '--gpu-type', candidate['gpu_type'],
        '--large-micro-cap', str(candidate['cap']), '--device', 'cuda:0',
        '--eval-batch-size', '256', '--log-every', '18', '--wandb-mode', mode,
        '--prepared-root', str(PREPARED)]
    return [dict(stage='train_uniqueVALlock_fullTEST', log='train.log',
        command=[sys.executable, '-B', '-u', '-m', 'code.V4.r81_experiment', 'train', *common]),
        dict(stage='twoLIB_and_PDF', log='reporting.log', command=[sys.executable, '-B', '-u',
            '-m', 'code.V4.r81_reporting', '--root', str(ROOT), '--group', GROUP,
            '--device', 'cuda:0'])]


def plan():
    result = dict(experiment='R81B', proposal_alias='requestedR80B', group=GROUP, root=str(ROOT),
        budget=dict(seed=2, initialization='R77 identicalinit_seed2.pt; fresh untrained parameters',
            initialization_trainable_sha256=INITIAL_STATE_HASH, batch=1024, successful_updates=49680,
            warmup_updates=3726, lr=1e-4, schedule='cosine', minimum_lr=1e-6),
        candidates=CANDIDATES, containers=CONTAINERS, poll_seconds=POLL_SECONDS,
        cpu_per_pipeline=4, requested_memory_per_pipeline='9G', allocation_cpu=10,
        allocation_memory='20G', cache=str(CACHE), cache_reused=True,
        gpu_free_rule='R80 full pipeline terminal + no live R80 stage/worker + no NVML compute process + memory<=512MiB',
        cpu_rule='reuse old per-GPU CPUs where possible; exclude live partner; remain in allocation/container cpuset',
        reservation='one UUID, held across train -> unique VAL lock + full TEST -> both LIBs + PDF',
        readiness='explicit main-thread ready + objective CPU tests + runner/preparation ready; immutable source hashes',
        only_group=GROUP, extra_seeds=False, launch_A=False, stop_R80=False,
        new_allocation=False, direct_ssh_training=False, git_commit=False, automatic_retry=False,
        source_sha256=source_hashes(), continuous_model_monitor=False,
        srun_launch={arm: srun_command(arm, 'container-entry', '<ONE_SHOT_TOKEN>') for arm in CANDIDATES},
        argv={arm: runner_commands(candidate, '<online-or-explicit-offline>') for arm, candidate in CANDIDATES.items()})
    return result


def standby():
    ROOT.mkdir(parents=True, exist_ok=True)
    with lock_file(owned(ROOT / 'queue_control_lock.json')):
        path = ROOT / 'queue_tmux.json'
        if path.exists():
            previous = read(path)
            if subprocess.run(['tmux', 'has-session', '-t', SESSION],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0:
                if capture(['tmux', 'display-message', '-p', '-t', SESSION, '#{pane_dead}']).strip() != '0':
                    raise RuntimeError('Standby tmux pane has exited; refusing implicit restart')
                print(json.dumps(previous, indent=2))
                return
            raise RuntimeError('Standby receipt exists but tmux is absent; refusing implicit restart')
        fresh_run()
        command = shlex.join(['bash', str(WRAPPER), 'loop'])
        record = dict(created_utc=utc(), tmux=SESSION, state='preparing', command=command,
            queue_source_sha256=LOADED_QUEUE_SHA256,
            main_thread_ready=False, creator_pid=os.getpid(), host=socket.gethostname())
        exclusive(owned(path), record)
        result = subprocess.run(['tmux', 'new-session', '-d', '-s', SESSION, command])
        record['tmux_exit_status'] = result.returncode
        if result.returncode:
            record['state'] = 'failed'
            write(path, record)
            raise RuntimeError('tmux creation failed; no retry or training was attempted')
        subprocess.run(['tmux', 'set-option', '-t', SESSION, 'remain-on-exit', 'on'], check=True)
        pane = capture(['tmux', 'display-message', '-p', '-t', SESSION,
            '#{pane_pid}|#{session_name}|#{pane_id}']).strip().split('|')
        record.update(pane_pid=int(pane[0]), pane_id=pane[2])
        write(path, record)
        print(json.dumps(record, indent=2))


def arm_queue(args):
    if not args.ready_evidence or not args.objective_tests_passed or not args.runner_tests_passed:
        raise RuntimeError('arm requires explicit main-thread evidence and both test confirmations')
    preparation = args.preparation_receipt.resolve()
    if preparation != ROOT / 'prepared/prepared.json' or not preparation.is_file():
        raise RuntimeError('The runner-owned prepared/prepared.json receipt is required')
    ready = read(preparation)
    if (ready.get('experiment') != 'R81' or ready.get('group') != GROUP
            or ready.get('train_only') is not True or ready.get('trainable_sha256') != INITIAL_STATE_HASH):
        raise RuntimeError('Runner preparation identity/TRAIN-only/fresh-initialization mismatch')
    for name in ('method_auc_objective.py', 'r81_experiment.py', 'r81_reporting.py'):
        if not (REPO / 'code/V4' / name).is_file():
            raise RuntimeError(f'Not ready: {name} missing')
    if read(PREPARED / 'prepared.json')['trainable_sha256'] != INITIAL_STATE_HASH:
        raise RuntimeError('R77 fresh initialization hash differs from the agreed state')
    selftest = read(ROOT / 'queue_selftest.json')
    if not selftest.get('passed') or selftest.get('source_sha256') != source_hashes():
        raise RuntimeError('Queue CPU selftests must pass against the current sources before arm')
    fresh_run()
    standby()
    with lock_file(owned(ROOT / 'queue_control_lock.json')):
        gate = dict(confirmed_utc=utc(), main_thread_ready=True, objective_cpu_tests_passed=True,
            runner_cpu_tests_passed=True, evidence=args.ready_evidence,
            preparation_receipt=str(preparation),
            preparation_sha256=hashlib.sha256(preparation.read_bytes()).hexdigest(),
            source_sha256=source_hashes(), group=GROUP, proposal_alias='requestedR80B')
        exclusive(owned(ROOT / 'queue_ready.json'), gate)
    print(f'ARMED: tmux={SESSION}; group={GROUP}; poll=60s; GPU training starts only after full resource gates', flush=True)


def cpu_baseline():
    if (ROOT / 'queue_claim.json').exists():
        raise RuntimeError('Do not collect a new baseline after claiming a GPU')
    result = dict(captured_utc=utc(), cpu_only=True, gpu_query=False, cpus_by_arm={}, nodes={})
    for arm in ('A_ce_only', 'B_batch512'):
        probe = remote_probe(arm, baseline_only=True)
        result['nodes'][CANDIDATES[arm]['node']] = probe
        for old_arm, rows in probe['processes'].items():
            result['cpus_by_arm'][old_arm] = sorted({cpu for row in rows for cpu in row['cpus']})
    write(ROOT / 'queue_cpu_baseline.json', result)
    print(json.dumps(result, indent=2))


def claim_record(arm, probe):
    return dict(claimed_utc=utc(), token=uuid.uuid4().hex, group=GROUP,
        proposal_alias='requestedR80B', candidate=arm, gpu_uuid=CANDIDATES[arm]['uuid'],
        selected_cpus=probe['selected_cpus'], resource_proof=probe,
        controller_pid=os.getpid(), controller_host=socket.gethostname(), tmux=SESSION,
        reservation='train through full TEST + both LIBs + PDF; never recycled by this queue')


def claim_for(arm, token):
    claim = read(ROOT / 'queue_claim.json')
    if claim.get('candidate') != arm or not token or claim.get('token') != token or claim.get('group') != GROUP:
        raise RuntimeError('Missing or mismatched exclusive one-shot claim')
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
                        reason='Waiting for explicit main-thread ready; local gate only, no resource probes')
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
            candidates = {}
            for old_arm, candidate in CANDIDATES.items():
                selected_probe = None
                try:
                    receipt = old_receipt(old_arm)
                    terminal, reason = old_terminal(receipt, candidate)
                    item = dict(eligible=False, reason=reason, old_state=receipt.get('state'))
                    if terminal:
                        probe = remote_probe(old_arm)
                        eligible, reason = usable(old_arm, old_receipt(old_arm), probe)
                        item.update(eligible=eligible, reason=reason, resource_proof=probe)
                        if eligible:
                            selected_probe = probe
                    candidates[old_arm] = item
                except (OSError, ValueError, KeyError, RuntimeError, subprocess.TimeoutExpired) as error:
                    candidates[old_arm] = dict(eligible=False, reason=f'{type(error).__name__}: {error}')
                if selected_probe is not None:
                    try:
                        with lock_file(gpu_lock_path(candidate['uuid'])):
                            assert_ready()
                            fresh_run()
                            claim = claim_record(old_arm, selected_probe)
                            exclusive(owned(ROOT / 'queue_claim.json'), claim)
                            return launch_claim(claim)
                    except BlockingIOError:
                        candidates[old_arm] = dict(eligible=False,
                            reason='Authorized UUID is reserved by another one-shot queue')
            status('pending', main_thread_ready=True, GPU_training_started=False, candidates=candidates,
                reason='No authorized GPU has passed every full-pipeline/allocation/container/CPU/NVML gate')
            previous = 'pending'
            time.sleep(POLL_SECONDS)


def launch_claim(claim):
    arm, token = claim['candidate'], claim['token']
    command = srun_command(arm, 'container-entry', token)
    receipt = dict(created_utc=utc(), state='launching', tmux=SESSION, group=GROUP,
        candidate=arm, gpu_uuid=claim['gpu_uuid'], controller_pid=os.getpid(), command=command)
    path = ROOT / 'launch_controller.json'
    exclusive(owned(path), receipt)
    status('launching', claim=claim, command=command, GPU_training_started=False)
    process = None
    try:
        process = subprocess.Popen(command)
        receipt.update(srun_pid=process.pid, state='running')
        write(path, receipt)
        code = process.wait()
        receipt.update(state='complete' if code == 0 else 'failed', exit_status=code, finished_utc=utc())
    except BaseException as error:
        if process is not None and process.poll() is None:
            process.wait()
        receipt.update(state='failed', exit_status=1, error=f'{type(error).__name__}: {error}', finished_utc=utc())
        write(path, receipt)
        status('failed', launch=receipt, automatic_retry=False)
        raise
    write(path, receipt)
    worker_path = ROOT / f'launch_{GROUP}.json'
    worker = read(worker_path) if worker_path.exists() else None
    status(receipt['state'], launch=receipt, worker=worker, automatic_retry=False,
        reason='One-shot queue exited; no subsequent model or seed will be launched')
    return code


def container_entry(arm, token):
    claim = claim_for(arm, token)
    assert_ready()
    probe = node_probe(arm)
    okay, reason = usable(arm, old_receipt(arm), probe)
    if not okay:
        raise RuntimeError(f'Child-side resource recheck failed: {reason}')
    if not set(claim['selected_cpus']) <= set(probe['allowed_cpus']):
        raise RuntimeError('Claimed CPU affinity no longer available')
    partner = {cpu for other, rows in probe['processes'].items() if other != arm
        for row in rows for cpu in row['cpus']}
    if set(claim['selected_cpus']) & partner:
        raise RuntimeError('Claimed CPUs now overlap the live partner; no automatic reassignment')
    candidate = CANDIDATES[arm]
    name, identity = CONTAINERS[candidate['node']]
    command = ['sudo', '-n', '/usr/bin/ai_docker', 'exec', '-u', str(os.getuid()),
        '-e', f'CUDA_VISIBLE_DEVICES={candidate["uuid"]}',
        '-e', f'SLURM_JOB_ID={candidate["job"]}', '-e', 'SLURM_STEP_ID=existing_allocation_container',
        '-e', f'R81_QUEUE_TOKEN={token}',
        '-e', 'R81_ASSIGNED_CPUS=' + ','.join(map(str, claim['selected_cpus'])),
        name, 'bash', str(WRAPPER), 'container-worker', '--candidate', arm, '--token', token]
    exclusive(owned(ROOT / 'launch_container_entry.json'), dict(started_utc=utc(), pid=os.getpid(),
        host=socket.gethostname(), job=candidate['job'], step=os.environ.get('SLURM_STEP_ID'),
        container_name=name, container_id=identity, resource_recheck=probe, command=command))
    return subprocess.run(command, check=False).returncode


def cache_inventory():
    count, size = 0, 0
    for split in ('train', 'val', 'test'):
        metadata_paths = sorted((SOURCE / split).glob('*/metadata.json'))
        if len(metadata_paths) != 18:
            raise RuntimeError('Frozen source cache must have all 18 task metadata records')
        for source in metadata_paths:
            target = CACHE / source.relative_to(SOURCE)
            metadata = read(target)
            if target.read_bytes() != source.read_bytes() or not metadata.get('complete'):
                raise RuntimeError(f'Frozen cache metadata identity failed: {target}')
            if metadata.get('cache_format') != 'r76_scenario_v1':
                raise RuntimeError('Unexpected existing cache format')
            paths = [(source, target)] + [(source.parent / spec['file'], target.parent / spec['file'])
                for spec in metadata['arrays'].values()]
            for original, cached in paths:
                length = cached.stat().st_size
                if length != original.stat().st_size:
                    raise RuntimeError(f'Frozen cache file-size mismatch: {cached}')
                count += 1
                size += length
    if (count, size) != (375, 17026558189):
        raise RuntimeError(f'Existing authorized cache inventory differs: files={count}, bytes={size}')
    return dict(root=str(CACHE), files=count, bytes=size, reused=True, copied_files=0,
        validation='54 metadata identities + all 321 array sizes; no heldout cost values inspected')


def wandb_mode():
    record = dict(enabled=True, project='selector',
        entity='yjkds-southern-university-of-science-technology',
        lookup='standard W&B environment/settings/netrc; no prompt, no key recorded')
    try:
        from wandb.sdk.lib import apikey
        credential = apikey.api_key()
        mode = 'online' if credential else 'offline'
        record.update(selected_mode=mode, credential_present=bool(credential))
        if not credential:
            record['offline_reason'] = 'No existing W&B credential found; explicit offline mode'
    except Exception as error:
        mode = 'offline'
        record.update(selected_mode=mode, credential_present=False,
            offline_reason=f'Standard credential lookup failed: {type(error).__name__}; explicit offline mode')
    return mode, record


def final_reservation(candidate):
    snapshot = gpu_inventory(candidate)
    return dict(checked_utc=utc(), gpu=snapshot,
        released=not snapshot['compute_processes'] and snapshot['memory_used_mib'] <= MEMORY_MIB,
        claim_reusable=False)


def container_worker(arm, token):
    claim = claim_for(arm, token)
    candidate = CANDIDATES[arm]
    identity = CONTAINERS[candidate['node']][1]
    path = ROOT / f'launch_{GROUP}.json'
    receipt = dict(state='preflight', started_utc=utc(), worker_pid=os.getpid(),
        worker_host=socket.gethostname(), tmux=SESSION, group=GROUP,
        proposal_alias='requestedR80B', candidate=arm, gpu_uuid=candidate['uuid'], stages=[])
    with lock_file(owned(ROOT / 'queue_worker_lock.json')):
        exclusive(owned(path), receipt)
        process = None
        try:
            assert_ready()
            if identity not in Path('/proc/self/cgroup').read_text() or socket.gethostname() != identity[:12]:
                raise RuntimeError('Worker is not in the immutable allocation-owned container cgroup')
            if os.environ.get('SLURM_JOB_ID') != str(candidate['job']) or os.environ.get('R81_QUEUE_TOKEN') != token:
                raise RuntimeError('Container worker job/token mismatch')
            if os.environ.get('CUDA_VISIBLE_DEVICES') != candidate['uuid']:
                raise RuntimeError('CUDA_VISIBLE_DEVICES must contain exactly one approved GPU UUID')
            selected = set(claim['selected_cpus'])
            if not selected <= set(os.sched_getaffinity(0)):
                raise RuntimeError('Claimed CPUs are not available inside the authorized container')
            os.sched_setaffinity(0, selected)
            receipt['container'] = dict(id=identity, cgroup=Path('/proc/self/cgroup').read_text(),
                assigned_cpus=sorted(os.sched_getaffinity(0)), job=candidate['job'])
            okay, reason = old_terminal(old_receipt(arm), candidate)
            if not okay or process_inventory(candidate['node']).get(arm):
                raise RuntimeError(f'Old R80 pipeline changed or still has a live process: {reason}')
            gpu = gpu_inventory(candidate)
            if gpu['compute_processes'] or gpu['memory_used_mib'] > MEMORY_MIB:
                raise RuntimeError('GPU occupied at the final in-container check; no launch')
            receipt['gpu_preflight'] = gpu
            receipt['cache'] = cache_inventory()
            mode, receipt['wandb'] = wandb_mode()
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=candidate['uuid'], R81_APPROVED='1',
                WANDB_MODE=mode, SLURM_JOB_ID=str(candidate['job']))
            directory = ROOT / GROUP
            directory.mkdir(parents=True, exist_ok=True)
            for name in ('last.pt', 'best.pt', 'status.json', 'protocol.json'):
                if (directory / name).exists():
                    raise RuntimeError(f'Fresh-only worker refuses existing {name}')
            commands = runner_commands(candidate, mode)

            def forward(signum, _frame):
                if process is not None and process.poll() is None:
                    os.killpg(process.pid, signum)

            signal.signal(signal.SIGTERM, forward)
            signal.signal(signal.SIGINT, forward)
            for step in commands:
                assert_ready()
                if not receipt['stages']:
                    gpu = gpu_inventory(candidate)
                    if gpu['compute_processes'] or gpu['memory_used_mib'] > MEMORY_MIB:
                        raise RuntimeError('GPU occupied immediately before the first training child')
                record = dict(stage=step['stage'], command=step['command'], state='starting',
                    log=str(directory / step['log']), started_utc=utc())
                receipt['stages'].append(record)
                receipt.update(state='running', phase=step['stage'])
                write(path, receipt)
                with (directory / step['log']).open('a', buffering=1) as output:
                    output.write(f'\n[R81B queue] {utc()} {shlex.join(step["command"])}\n')
                    process = subprocess.Popen(step['command'], cwd=REPO, env=env,
                        stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
                    record.update(pid=process.pid, state='running')
                    write(path, receipt)
                    status('running', pipeline_stage=step['stage'], worker_pid=os.getpid(),
                        child_pid=process.pid, gpu_uuid=candidate['uuid'], candidate=arm,
                        assigned_cpus=sorted(selected), wandb=receipt['wandb'], log=record['log'],
                        GPU_training_started=True, reservation_released=False)
                    code = process.wait()
                    output.write(f'\n[R81B queue] {utc()} stage={step["stage"]} exit_status={code}\n')
                record.update(state='complete' if code == 0 else 'failed', exit_status=code, finished_utc=utc())
                write(path, receipt)
                if code:
                    receipt.update(state='failed', phase=step['stage'], failed_stage=step['stage'],
                        exit_status=code, finished_utc=utc(), reservation=final_reservation(candidate))
                    write(path, receipt)
                    return code
            receipt.update(state='complete', phase='complete', exit_status=0, finished_utc=utc(),
                reservation=final_reservation(candidate))
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
        def fixture(self):
            candidate = CANDIDATES['A_ce_only']
            receipt = dict(group=candidate, state='complete', finished_utc='now', exit_status=0,
                stages=[dict(stage=stage, state='complete', exit_status=0)
                    for stage in ('train', 'test', 'LIB_and_PDF_report')])
            probe = dict(allocation=dict(state='RUNNING'), container=dict(id=CONTAINERS['gpu03'][1]),
                processes={'A_ce_only': [], 'C_lr3e4': [dict(cpus=[4, 5, 6, 7])]},
                gpu=dict(memory_used_mib=2, compute_processes=[]), selected_cpus=[0, 1, 2, 3],
                allowed_cpus=list(range(10)))
            return receipt, probe

        def test_full_pipeline_not_training_marker(self):
            receipt, probe = self.fixture()
            self.assertTrue(usable('A_ce_only', receipt, probe)[0])
            receipt['state'] = 'running'
            self.assertFalse(usable('A_ce_only', receipt, probe)[0])
            receipt['state'] = 'complete'
            receipt['stages'].pop()
            self.assertFalse(usable('A_ce_only', receipt, probe)[0])

        def test_failed_pipeline_allowed_only_without_live_stage(self):
            receipt, probe = self.fixture()
            receipt.update(state='failed', exit_status=1)
            self.assertTrue(usable('A_ce_only', receipt, probe)[0])
            probe['processes']['A_ce_only'] = [dict(pid=7, cpus=[0, 1, 2, 3])]
            self.assertFalse(usable('A_ce_only', receipt, probe)[0])

        def test_every_idle_gate_is_required(self):
            receipt, original = self.fixture()
            for section, key, value in (
                    ('gpu', 'compute_processes', [['uuid', '7']]),
                    ('gpu', 'memory_used_mib', 513),
                    ('allocation', 'state', 'COMPLETED'), ('container', 'id', 'changed')):
                probe = copy.deepcopy(original)
                probe[section][key] = value
                self.assertFalse(usable('A_ce_only', receipt, probe)[0], (section, key))
            original['selected_cpus'] = [3, 4, 5, 6]
            self.assertFalse(usable('A_ce_only', receipt, original)[0])

        def test_nvml_ordinal_is_informational_uuid_and_type_are_required(self):
            candidate = CANDIDATES['A_ce_only']
            module = sys.modules[__name__]
            for actual_index in (0, 3, 4):
                devices = f'{actual_index}, {candidate["uuid"]}, NVIDIA GeForce RTX 3090, 20000, 24576, 100\n'
                with mock.patch.object(module, 'capture', side_effect=[devices, '']):
                    result = gpu_inventory(candidate)
                self.assertEqual(result['nvml_index'], actual_index)
                self.assertEqual(result['authorized_logical_index'], 0)
            for bad_uuid, hardware in (('GPU-wrong', '3090'), (candidate['uuid'], '4090')):
                devices = f'3, {bad_uuid}, NVIDIA GeForce RTX {hardware}, 2, 24576, 0\n'
                with mock.patch.object(module, 'capture', return_value=devices):
                    with self.assertRaisesRegex(RuntimeError, 'UUID/type mismatch'):
                        gpu_inventory(candidate)

        def test_node_nvml_queries_use_the_identity_locked_container(self):
            candidate = CANDIDATES['A_ce_only']
            name, identity = CONTAINERS[candidate['node']]
            devices = f'3, {candidate["uuid"]}, NVIDIA GeForce RTX 3090, 20000, 24576, 100\n'
            with mock.patch.object(sys.modules[__name__], 'capture', side_effect=[devices, '']) as query:
                result = gpu_inventory(candidate, dict(name=name, id=identity))
            expected = ['sudo', '-n', '/usr/bin/ai_docker', 'exec', '-u', str(os.getuid()), identity, 'nvidia-smi']
            self.assertTrue(all(call.args[0][:len(expected)] == expected for call in query.call_args_list))
            self.assertEqual(result['query_transport'], 'verified_allocation_container_exec')
            with self.assertRaisesRegex(RuntimeError, 'verified allocation-owned container'):
                gpu_inventory(candidate, dict(name=name, id='changed'))

        def test_platform_exec_banner_is_not_a_gpu_record(self):
            candidate = CANDIDATES['B_batch512']
            banner = 'Distribute Task Container.\ntask_id:1475\n'
            row = f'0, {candidate["uuid"]}, NVIDIA GeForce RTX 4090, 2, 24564, 0\n'
            with mock.patch.object(sys.modules[__name__], 'capture', side_effect=[banner + row, banner]):
                self.assertEqual(gpu_inventory(candidate)['compute_processes'], [])
            with self.assertRaises(RuntimeError):
                nvml_csv('unexpected wrapper text\n' + row, 6)

        def test_global_uuid_lock_is_shared_and_exclusive(self):
            with tempfile.TemporaryDirectory(prefix='r81_resource_') as directory:
                with mock.patch.object(sys.modules[__name__], 'GPU_LOCK_ROOT', Path(directory)):
                    path = gpu_lock_path(CANDIDATES['A_ce_only']['uuid'])
                    with lock_file(path):
                        with self.assertRaises(BlockingIOError):
                            with lock_file(path):
                                self.fail('A second queue acquired the same UUID')
                    self.assertNotEqual(path, gpu_lock_path(CANDIDATES['C_lr3e4']['uuid']))

        def test_cpu_affinity_excludes_live_partner(self):
            rows = {'A_ce_only': [], 'C_lr3e4': [dict(cpus=[0, 1, 2, 3])]}
            baseline = dict(cpus_by_arm={'A_ce_only': [4, 5, 8, 9]})
            self.assertEqual(choose_cpus('A_ce_only', range(10), rows, baseline), [4, 5, 8, 9])
            with self.assertRaises(RuntimeError):
                choose_cpus('A_ce_only', range(6), rows)
            self.assertEqual(cpulist('12-17,64-67'), {12, 13, 14, 15, 16, 17, 64, 65, 66, 67})

        def test_existing_allocation_only_argv(self):
            for arm, candidate in CANDIDATES.items():
                command = srun_command(arm, 'container-entry', 'token')
                for part in (f'--jobid={candidate["job"]}', '--overlap', '--exact',
                        '--gres=none', '--cpus-per-task=4', '--mem=9G'):
                    self.assertIn(part, command)
                self.assertNotIn('ssh', command)
                self.assertNotIn('sbatch', command)
                commands = runner_commands(candidate, 'offline')
                self.assertEqual(len(commands), 2)
                self.assertIn(GROUP, commands[0]['command'])
                self.assertNotIn('--resume', commands[0]['command'])
                self.assertNotIn('--state-hash', commands[0]['command'])
                self.assertEqual(commands[0]['command'].count('train'), 1)
                self.assertNotIn('test', commands[1]['command'])
                self.assertIn('cuda:0', commands[1]['command'])

        def test_duplicate_claim_and_flock_refused(self):
            with tempfile.TemporaryDirectory(prefix='r81_cpu_') as directory:
                path = Path(directory) / 'claim.json'
                exclusive(path, dict(token='first'))
                with self.assertRaises(FileExistsError):
                    exclusive(path, dict(token='second'))
                self.assertEqual(read(path)['token'], 'first')
                with lock_file(Path(directory) / 'lock.json'):
                    with self.assertRaises(BlockingIOError):
                        with lock_file(Path(directory) / 'lock.json'):
                            self.fail('Duplicate lock acquired')

        def test_launch_failure_is_terminal_not_requeued(self):
            receipt, probe = self.fixture()
            module = sys.modules[__name__]
            with tempfile.TemporaryDirectory(prefix='r81_cpu_') as directory:
                root = Path(directory)
                exclusive(root / 'queue_ready.json', {})
                with mock.patch.object(module, 'ROOT', root), \
                        mock.patch.object(module, 'fresh_run'), \
                        mock.patch.object(module, 'assert_ready'), \
                        mock.patch.object(module, 'old_receipt', return_value=receipt), \
                        mock.patch.object(module, 'remote_probe', return_value=probe) as remote, \
                        mock.patch.object(module, 'launch_claim', side_effect=RuntimeError('child failed')) as launch:
                    with self.assertRaisesRegex(RuntimeError, 'child failed'):
                        run_queue()
                    self.assertTrue((root / 'queue_claim.json').exists())
                    self.assertEqual(remote.call_count, 1)
                    self.assertEqual(launch.call_count, 1)

        def test_fixed_budget_and_only_B(self):
            value = plan()
            self.assertEqual(value['only_group'], GROUP)
            self.assertEqual(value['budget']['batch'], 1024)
            self.assertEqual(value['budget']['successful_updates'], 49680)
            self.assertEqual(value['budget']['warmup_updates'], 3726)
            self.assertFalse(value['launch_A'])
            self.assertFalse(value['automatic_retry'])
            self.assertEqual(len({item['uuid'] for item in CANDIDATES.values()}), 4)

    suite = unittest.defaultTestLoader.loadTestsFromTestCase(QueueTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    receipt = dict(checked_utc=utc(), passed=result.wasSuccessful(), tests_run=result.testsRun,
        errors=len(result.errors), failures=len(result.failures), CPU_only=True,
        subprocesses_started=0, GPU_contexts_created=0, source_sha256=source_hashes())
    write(ROOT / 'queue_selftest.json', receipt)
    return 0 if result.wasSuccessful() else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('dry-run', 'selftest', 'standby', 'arm', 'status',
        'loop', 'cpu-baseline', 'probe-node', 'container-entry', 'container-worker'))
    parser.add_argument('--candidate', choices=tuple(CANDIDATES))
    parser.add_argument('--token')
    parser.add_argument('--baseline-only', action='store_true')
    parser.add_argument('--ready-evidence')
    parser.add_argument('--objective-tests-passed', action='store_true')
    parser.add_argument('--runner-tests-passed', action='store_true')
    parser.add_argument('--preparation-receipt', type=Path, default=ROOT / 'prepared/prepared.json')
    args = parser.parse_args(argv)
    if args.command in ('probe-node', 'container-entry', 'container-worker') and not args.candidate:
        parser.error('Internal allocation commands require --candidate')
    if args.command == 'dry-run':
        value = plan()
        write(ROOT / 'queue_plan.json', value)
        print(json.dumps(value, indent=2))
    elif args.command == 'selftest':
        return selftests()
    elif args.command == 'standby':
        standby()
    elif args.command == 'arm':
        arm_queue(args)
    elif args.command == 'loop':
        try:
            return run_queue()
        except BlockingIOError:
            print('Another R81 one-shot queue owns the global flock; no launch', file=sys.stderr)
            return 2
        except BaseException as error:
            status('failed', error=f'{type(error).__name__}: {error}', automatic_retry=False)
            raise
    elif args.command == 'cpu-baseline':
        cpu_baseline()
    elif args.command == 'probe-node':
        print(json.dumps(node_probe(args.candidate, args.baseline_only)))
    elif args.command == 'container-entry':
        return container_entry(args.candidate, args.token)
    elif args.command == 'container-worker':
        return container_worker(args.candidate, args.token)
    elif args.command == 'status':
        print(json.dumps(read(ROOT / 'queue_status.json') if (ROOT / 'queue_status.json').exists()
            else dict(state='preparing', tmux=SESSION, group=GROUP, main_thread_ready=False), indent=2))
    return 0


if __name__ == '__main__':
    sys.exit(main())
