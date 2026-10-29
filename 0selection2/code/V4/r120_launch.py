"""Use the existing four authorized GPUs for one distributed R120 model."""

from __future__ import annotations

import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import shlex
import socket
import subprocess
import sys
import time


REPO = Path('/public/home/shiys/0selection2')
ROOT = REPO / 'code/V4/runs/R120_multidomain_selector'
WRAPPER = REPO / 'code/V4/run_v4_r120.sh'
GPU_MUTEX = REPO / 'code/V4/runs/R81_method_conditioned_auc'
CACHE = Path('/tmp/shiys_R84_new90k_cache')
NODES = {
    'gpu05': dict(job=1475, rank=0, container='1475_gpu05',
        identity='76045731f6b453443fe18a9bce671599bdbf17c2713b8edcdd090925eea14bf6',
        uuids=['GPU-71479e7e-f8bd-e430-66ed-f8db9f0d47be', 'GPU-597c682a-4b7f-7609-80a6-8ced791f5c5d']),
    'gpu03': dict(job=1465, rank=1, container='1465_gpu03',
        identity='b8ced5d8a8cdafdc5006f827a0d36c319c61056a65316af4ff9937453fa0635c',
        uuids=['GPU-cb24f168-9f18-f0d5-6f4a-6d2406958f50', 'GPU-cb626bdb-4319-2f5e-b91f-17fcaa44a83d']),
}


def now():
    return datetime.now(timezone.utc).isoformat()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name('.' + path.name + f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    temporary.replace(path)


def capture(argv):
    return subprocess.check_output(argv, text=True, stderr=subprocess.STDOUT, timeout=25).strip()


def gpu_rows():
    raw = capture(['nvidia-smi', '--query-gpu=uuid,name,memory.used,utilization.gpu,temperature.gpu',
                   '--format=csv,noheader,nounits'])
    result = {}
    for line in raw.splitlines():
        gpu, name, memory, utilization, temperature = (part.strip() for part in line.split(','))
        result[gpu] = dict(name=name, memory_mib=int(memory), utilization=int(utilization),
                           temperature_c=int(temperature))
    return result


def srun_command(node, stage, root, remainder=()):
    item = NODES[node]
    probe = stage == 'probe'
    model_stage = stage in ('train', 'test', 'lib')
    return ['srun', f'--jobid={item["job"]}', '--overlap', '--exact',
        '--gres=gpu:2' if model_stage else '--gres=none',
        '--immediate=10', '-N1', '-n1', f'--nodelist={node}',
        '--cpus-per-task=' + ('1' if probe else '8'), '--mem=' + ('512M' if probe else '18G'),
        'bash', str(WRAPPER), 'host-node' if model_stage else 'node-entry',
        '--node', node, '--stage', stage,
        '--root', str(root), *remainder]


def host_node(args):
    """Use host networking inside a GPU-owning Slurm step, never ordinary SSH."""
    item = NODES[args.node]
    cgroup = Path('/proc/self/cgroup').read_text()
    if (socket.gethostname().split('.')[0] != args.node
            or os.environ.get('SLURM_JOB_ID') != str(item['job'])
            or f'job_{item["job"]}' not in cgroup):
        raise RuntimeError('Model workers require their existing Slurm allocation GPU step')
    cpus = sorted(os.sched_getaffinity(0))
    if len(cpus) < 8:
        raise RuntimeError('The allocated GPU step requires eight CPUs')
    cpus = cpus[:8]
    os.sched_setaffinity(0, cpus)
    rows = gpu_rows()
    selected = {uuid: rows[uuid] for uuid in item['uuids']}
    proof = dict(utc=now(), node=args.node, job=item['job'], cgroup=cgroup,
                 step=os.environ.get('SLURM_STEP_ID'), cpus=cpus, gpus=selected,
                 transport='allocation-host-network', cache=str(CACHE))
    write(args.root / 'resources' / f'{args.node}_{args.stage}_entry.json', proof)
    with ExitStack() as stack:
        _gpu_mutexes(stack, item['uuids'])
        if any(row['memory_mib'] > 512 or row['utilization'] > 10 for row in selected.values()):
            raise RuntimeError('An assigned GPU is busy; existing work is preserved')
        CACHE.mkdir(parents=True, exist_ok=True)
        with (CACHE / '.r120_copy.lock').open('a+') as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            marker = CACHE / '.r120_frozen_copy_complete.json'
            if not marker.exists():
                source = REPO / 'code/V4/runs/R84_new90k_attention/input_cache'
                started = time.monotonic()
                print(f'[R120 cache] Copy existing frozen arrays on {args.node}: {source} -> {CACHE}', flush=True)
                subprocess.run(['cp', '-a', '--reflink=auto', str(source) + '/.', str(CACHE)], check=True)
                write(marker, dict(source=str(source), completed_utc=now(),
                                   seconds=time.monotonic() - started, preprocessing=False))
            if not (CACHE / 'train/TSP/metadata.json').is_file():
                raise FileNotFoundError('The frozen host-local input cache is incomplete')
        env = dict(os.environ, R120_ASSIGNED_CPUS=','.join(map(str, cpus)),
                   CUDA_VISIBLE_DEVICES=','.join(item['uuids']),
                   MASTER_ADDR='12.12.12.26', MASTER_PORT=str(args.master_port),
                   NCCL_IB_DISABLE='1', NCCL_P2P_DISABLE='1', NCCL_SOCKET_FAMILY='AF_INET',
                   NCCL_SOCKET_IFNAME='=bond0', TORCH_NCCL_ASYNC_ERROR_HANDLING='1')
        argv = ['torchrun', '--nnodes=2', '--nproc-per-node=2', f'--node-rank={item["rank"]}',
                '--master-addr=12.12.12.26', f'--master-port={args.master_port}',
                '-m', 'code.V4.r120_launch', 'rank', '--stage', args.stage,
                '--root', str(args.root), *args.runner_args]
        print(f'[R120 train] Starting allocation-host DDP on {args.node}', flush=True)
        return subprocess.call(argv, env=env)


def node_entry(args):
    item = NODES[args.node]
    if socket.gethostname().split('.')[0] != args.node or os.environ.get('SLURM_JOB_ID') != str(item['job']):
        raise RuntimeError('The entry point must be inside its existing allocation CPU step')
    identity = capture(['sudo', '-n', '/usr/bin/ai_docker', 'inspect', '--format={{.Id}}', item['container']])
    if identity != item['identity']:
        raise RuntimeError('Allocation-owned container identity changed')
    from .r81_queue import allocation, container_inspect
    candidate = dict(item, node=args.node)
    allocation_cpus = allocation(candidate)['cpus']
    container_cpus = container_inspect(candidate)['allowed_cpus']
    allowed = sorted(set(allocation_cpus) & set(container_cpus))
    needed = 1 if args.stage == 'probe' else 8
    if len(allowed) < needed:
        raise RuntimeError('The existing allocation/container lacks the required CPU set')
    cpus = ','.join(map(str, allowed[:needed]))
    argv = ['sudo', '-n', '/usr/bin/ai_docker', 'exec', '-u', str(os.getuid()),
            '-e', f'R120_ASSIGNED_CPUS={cpus}', '-e', f'SLURM_JOB_ID={item["job"]}',
            '-e', 'CUDA_VISIBLE_DEVICES=' + ','.join(item['uuids']), item['container'],
            'bash', str(WRAPPER), 'container-node', '--node', args.node,
            '--stage', args.stage, '--root', str(args.root),
            '--master-port', str(args.master_port), *args.runner_args]
    if args.label_plan:
        argv += ['--label-plan', str(args.label_plan)]
    return subprocess.call(argv)


def node_proof(args):
    item = NODES[args.node]
    cgroup = Path('/proc/self/cgroup').read_text()
    if item['identity'] not in cgroup or socket.gethostname() != item['identity'][:12]:
        raise RuntimeError('The worker must remain in its allocation-owned container')
    assigned = {int(value) for value in os.environ['R120_ASSIGNED_CPUS'].split(',')}
    if not assigned or not assigned <= os.sched_getaffinity(0):
        raise RuntimeError('CPU step and container CPU sets do not match')
    os.sched_setaffinity(0, assigned)
    rows = gpu_rows()
    selected = {uuid: rows[uuid] for uuid in item['uuids']}
    return dict(utc=now(), node=args.node, job=item['job'], container=item['identity'],
                cgroup=cgroup, cpus=sorted(assigned), gpus=selected,
                local_routing_cache=(CACHE / 'train/TSP/metadata.json').is_file())


def _gpu_mutexes(stack, uuids):
    for gpu in uuids:
        handle = stack.enter_context((GPU_MUTEX / f'queue_gpu_{gpu}.json').open('a+'))
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)


def container_node(args):
    proof = node_proof(args)
    write(args.root / 'resources' / f'{args.node}_{args.stage}_entry.json', proof)
    if args.stage == 'probe':
        print(json.dumps(proof), flush=True)
        return 0
    with ExitStack() as stack:
        item = NODES[args.node]
        _gpu_mutexes(stack, item['uuids'])
        if any(value['memory_mib'] > 512 or value['utilization'] > 10 for value in proof['gpus'].values()):
            raise RuntimeError('An assigned GPU is busy; existing work is preserved')
        if args.stage in ('capacity', 'labels'):
            processes = []
            for lane, gpu in enumerate(item['uuids']):
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu)
                cpus = proof['cpus'][lane * 4:(lane + 1) * 4]
                prefix = ['taskset', '-c', ','.join(map(str, cpus)), sys.executable, '-B', '-u', '-m']
                if args.stage == 'capacity':
                    argv = prefix + ['code.V4.r120_capacity', '--root', str(args.root),
                            '--slot', f'{args.node}_{lane}', '--cache-root', str(CACHE), *args.runner_args]
                else:
                    if args.label_plan is None:
                        raise ValueError('A concrete nonoverlapping annotation plan is required')
                    argv = prefix + ['code.V4.r120_launch', 'label-lane', '--node', args.node,
                            '--lane', str(lane), '--root', str(args.root), '--label-plan', str(args.label_plan)]
                log = stack.enter_context((args.root / 'resources' / f'{args.stage}_{args.node}_{lane}.log').open('a'))
                processes.append(subprocess.Popen(argv, env=env, stdout=log, stderr=subprocess.STDOUT))
            codes = [process.wait() for process in processes]
            return 1 if any(codes) else 0
        if not proof['local_routing_cache']:
            raise FileNotFoundError('Existing R84 node-local input cache is unavailable')
        env = dict(os.environ, MASTER_ADDR='12.12.12.26', MASTER_PORT=str(args.master_port),
                   NCCL_IB_DISABLE='1', NCCL_P2P_DISABLE='1', NCCL_SOCKET_FAMILY='AF_INET',
                   NCCL_SOCKET_IFNAME='=bond0', TORCH_NCCL_ASYNC_ERROR_HANDLING='1')
        argv = ['torchrun', '--nnodes=2', '--nproc-per-node=2', f'--node-rank={item["rank"]}',
                '--master-addr=12.12.12.26', f'--master-port={args.master_port}',
                '-m', 'code.V4.r120_launch', 'rank', '--stage', args.stage,
                '--root', str(args.root), *args.runner_args]
        return subprocess.call(argv, env=env)


def rank(args):
    local_rank = int(os.environ['LOCAL_RANK'])
    cpus = sorted(int(x) for x in os.environ['R120_ASSIGNED_CPUS'].split(','))
    os.sched_setaffinity(0, cpus[local_rank * 4:(local_rank + 1) * 4])
    from .r120_experiment import main
    main([args.stage, '--root', str(args.root), '--cache-root', str(CACHE), *args.runner_args])
    return 0


def label_lane(args):
    plan = json.loads(args.label_plan.read_text())
    slot = args.node + '_' + str(args.lane)
    jobs = plan['slots'][slot]
    parallel = int(plan.get('parallel_per_slot', 1))
    if parallel > 1 and args.worker is None:
        cpus = sorted(os.sched_getaffinity(0))
        if parallel > len(cpus):
            raise RuntimeError('Annotation concurrency exceeds the assigned CPU cores')
        processes = []
        for worker in range(parallel):
            argv = ['taskset', '-c', str(cpus[worker]), sys.executable, '-B', '-u',
                    '-m', 'code.V4.r120_launch', 'label-lane', '--node', args.node,
                    '--lane', str(args.lane), '--root', str(args.root),
                    '--label-plan', str(args.label_plan), '--worker', str(worker)]
            env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
                       OPENBLAS_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1')
            processes.append(subprocess.Popen(argv, env=env))
        codes = [process.wait() for process in processes]
        return 1 if any(codes) else 0
    if args.worker is not None:
        jobs = [job for job in jobs if job['worker'] == args.worker]
        slot += '_w' + str(args.worker)
    receipt = dict(slot=slot, started_utc=now(), plan=str(args.label_plan), stages=[], state='running')
    path = args.root / 'resources' / (plan['name'] + '_' + slot + '.json')
    for job in jobs:
        argv = [sys.executable, '-B', '-u', plan['script'], 'label', '--root', plan['data_root'],
                '--split', job['split'], '--start', str(job['start']), '--stop', str(job['stop']),
                '--device', 'cuda:0', '--main-authorized-gpu']
        row = dict(job, started_utc=now(), command=argv)
        receipt['stages'].append(row)
        write(path, receipt)
        code = subprocess.call(argv)
        row.update(exit_status=code, ended_utc=now())
        if code:
            receipt.update(state='failed', ended_utc=now())
            write(path, receipt)
            return code
    receipt.update(state='complete', ended_utc=now())
    write(path, receipt)
    return 0


def launch(args):
    args.root.mkdir(parents=True, exist_ok=True)
    stage = args.stage
    if stage == 'train':
        ready = json.loads((args.root / 'prepared.json').read_text())
        if not ready['complete24']:
            raise ValueError('Formal training requires the complete declared 24-task pool')
        if (args.root / 'last.pt').exists():
            raise FileExistsError('Existing model retained; explicit resume is a separate action')
    with (args.root / 'launch.lock').open('a+') as claim:
        fcntl.flock(claim, fcntl.LOCK_EX | fcntl.LOCK_NB)
        processes = []
        with ExitStack() as stack:
            for node in NODES:
                extra = ['--master-port', str(args.master_port), *args.runner_args]
                if args.label_plan:
                    extra += ['--label-plan', str(args.label_plan)]
                argv = srun_command(node, stage, args.root, extra)
                log_path = args.root / f'{stage}_{node}.log'
                log = stack.enter_context(log_path.open('a'))
                process = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT)
                processes.append(process)
                write(args.root / 'resources' / f'{node}_{stage}_submission.json',
                      dict(utc=now(), pid=process.pid, command=argv, log=str(log_path)))
            codes = [process.wait() for process in processes]
        write(args.root / 'resources' / f'{stage}_exit.json', dict(utc=now(), exit_codes=codes))
        if any(codes):
            raise RuntimeError(f'{stage} exited with {codes}; logs and checkpoints retained')
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('launch', 'node-entry', 'container-node', 'host-node', 'rank', 'label-lane'))
    parser.add_argument('--stage', choices=('probe', 'capacity', 'train', 'test', 'lib', 'labels'), default='probe')
    parser.add_argument('--node', choices=tuple(NODES))
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--master-port', type=int, default=29820)
    parser.add_argument('--label-plan', type=Path)
    parser.add_argument('--lane', type=int, choices=(0, 1))
    parser.add_argument('--worker', type=int)
    args, args.runner_args = parser.parse_known_args(argv)
    if args.runner_args[:1] == ['--']:
        args.runner_args = args.runner_args[1:]
    if args.action in ('node-entry', 'container-node', 'host-node') and args.node is None:
        parser.error('--node is required for allocation entry')
    return {'launch': launch, 'node-entry': node_entry, 'container-node': container_node, 'host-node': host_node,
            'rank': rank, 'label-lane': label_lane}[args.action](args)


if __name__ == '__main__':
    raise SystemExit(main())
