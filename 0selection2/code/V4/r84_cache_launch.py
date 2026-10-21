"""One-shot four-GPU TRAIN/VAL cache preparation; never starts training."""

from contextlib import ExitStack
import argparse
import os
from pathlib import Path
import shlex
import socket
import subprocess
import sys
import time
import uuid

from . import r84_launch as base


ROOT = base.ROOT / 'cache_prepare'
PREPARED = base.REPO / 'code/V4/runs/R84_new90k_attention/prepared'
CACHE = base.REPO / 'code/V4/runs/R84_new90k_attention/input_cache'
DATA = Path('/public/home/shiys/selection_data_new90k_B_filtered_single_reld_mt_seed2_20261018')
WRAPPER = base.REPO / 'code/V4/run_v4_r84_cache_prepare.sh'
SLOTS = ('3090_0', '4090_0', '3090_1', '4090_1')
CPUS = ((12, 13, 14, 15), (0, 1, 2, 3), (16, 17, 64, 65), (4, 5, 72, 73))
SESSION = 'r84_cache_prepare_once'
SCRIPT = base.REPO / 'code/V4/r84_prepare.py'


def command(stage, rank=None):
    result = [sys.executable, '-B', '-u', '-m', 'code.V4.r84_cache_launch', stage]
    if rank is not None:
        result += ['--rank', str(rank)]
    return result


def runner(rank=None):
    result = [sys.executable, '-B', '-u', '-m', 'code.V4.r84_prepare',
        'finalize' if rank is None else 'prepare', '--root', str(PREPARED),
        '--data-root', str(DATA), '--cache-root', str(CACHE),
        '--device', 'cpu' if rank is None else 'cuda:0', '--splits', 'train', 'val']
    if rank is not None:
        result += ['--reuse-cache-root', str(base.transport.CACHE),
            '--worker-rank', str(rank), '--workers', '4']
    return result


def check_source():
    ready = base.read(ROOT / 'ready.json')
    if not ready.get('cache_preparation_only') or not ready.get('main_authorized'):
        raise RuntimeError('Missing preparation-only authorization')
    if base.sha(SCRIPT) != ready['prepare_sha256']:
        raise RuntimeError('r84_prepare changed after its CPU tests and launch authorization')
    return ready


def container_command(rank):
    slot = SLOTS[rank]
    spec = base.SLOTS[slot]
    identity = base.transport.CONTAINERS[spec['node']][1]
    ready = check_source()
    result = ['sudo', '-n', '/usr/bin/ai_docker', 'exec', '-u', str(os.getuid())]
    env = base.environment(slot, ready['token'])
    names = ('CUDA_VISIBLE_DEVICES', 'PYTHONPATH', 'PYTHONDONTWRITEBYTECODE',
        'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
        'NUMEXPR_NUM_THREADS', 'PYTHONUNBUFFERED', 'MPLBACKEND',
        'PYTORCH_CUDA_ALLOC_CONF', 'SLURM_JOB_ID', base.TOKEN_ENV)
    for key in names:
        result += ['-e', key + '=' + env[key]]
    return result + [identity, 'bash', str(WRAPPER), 'worker', '--rank', str(rank)]


def remote_command(rank):
    spec = base.SLOTS[SLOTS[rank]]
    return ['srun', f'--jobid={spec["job"]}', '--overlap', '--exact', '--gres=none',
        '--immediate=5', '-N1', '-n1', f'--nodelist={spec["node"]}',
        '--cpus-per-task=4', '--mem=9G', 'bash', '-lc', base.activated(command('entry', rank))]


def entry(rank):
    check_source()
    spec = base.SLOTS[SLOTS[rank]]
    if (socket.gethostname().split('.')[0] != spec['node']
            or os.environ.get('SLURM_JOB_ID') != str(spec['job'])
            or f'/job_{spec["job"]}/' not in Path('/proc/self/cgroup').read_text()):
        raise RuntimeError('Cache entry must use the original allocation CPU-step cgroup')
    container = base.transport.container_inspect(spec)
    if not set(CPUS[rank]) <= set(container['allowed_cpus']):
        raise RuntimeError('Assigned preparation CPU set exceeds allocation container')
    base.save(ROOT / f'entry_{rank}.json', dict(checked_utc=base.utc(), rank=rank,
        container=container, slurm_cgroup=Path('/proc/self/cgroup').read_text()), exclusive=True)
    return subprocess.run(container_command(rank), check=False).returncode


def worker(rank):
    ready = check_source()
    slot, spec = SLOTS[rank], base.SLOTS[SLOTS[rank]]
    identity = base.assert_container(slot)
    if (os.environ.get('CUDA_VISIBLE_DEVICES') != spec['uuid']
            or os.environ.get(base.TOKEN_ENV) != ready['token']
            or not set(CPUS[rank]) <= set(os.sched_getaffinity(0))):
        raise RuntimeError('Preparation UUID, token or CPU binding mismatch')
    os.sched_setaffinity(0, CPUS[rank])
    gpu = base.transport.gpu_inventory(spec)
    if gpu['compute_processes'] or gpu['memory_used_mib'] > 512:
        raise RuntimeError('Preparation GPU is occupied; no existing process is stopped')
    record = dict(started_utc=base.utc(), rank=rank, slot=slot, gpu_uuid=spec['uuid'],
        container=identity, cpus=list(CPUS[rank]), argv=runner(rank), state='running',
        cache_preparation_only=True, training_started=False, splits=['train', 'val'],
        reuse_cache_root=str(base.transport.CACHE), prepare_sha256=ready['prepare_sha256'])
    receipt = ROOT / f'worker_{rank}.json'
    base.save(receipt, record, exclusive=True)
    child = subprocess.Popen(runner(rank), cwd=base.REPO)
    record['pid'] = child.pid
    base.save(receipt, record)
    rc = child.wait()
    record.update(state='complete' if rc == 0 else 'failed', exit_status=rc, finished_utc=base.utc())
    base.save(receipt, record)
    return rc


def controller():
    check_source()
    with base.lock(ROOT / 'controller.lock'), ExitStack() as locks:
        for slot in SLOTS:
            spec = base.SLOTS[slot]
            path = base.transport.ROOT / f'queue_gpu_{spec["uuid"]}.json'
            locks.enter_context(base.lock(path, existing=path.exists()))
        if (ROOT / 'controller.json').exists():
            raise RuntimeError('Preparation already attempted; no automatic duplicate or retry')
        record = dict(started_utc=base.utc(), state='preparing', cache_preparation_only=True,
            training_started=False, tmux=SESSION, workers=[], finalization=None)
        base.save(ROOT / 'controller.json', record, exclusive=True)
        children, streams = [], []
        try:
            for rank in range(4):
                log = ROOT / f'worker_{rank}.log'
                stream = log.open('a', buffering=1)
                streams.append(stream)
                argv = remote_command(rank)
                stream.write(f'[R84 cache] {base.utc()} {shlex.join(argv)}\n')
                child = subprocess.Popen(argv, cwd=base.REPO, stdout=stream, stderr=subprocess.STDOUT)
                children.append(child)
                record['workers'].append(dict(rank=rank, slot=SLOTS[rank], pid=child.pid,
                    log=str(log), argv=argv, state='running'))
                base.save(ROOT / 'controller.json', record)
            while any(child.poll() is None for child in children):
                for index, child in enumerate(children):
                    rc = child.poll()
                    if rc is not None:
                        record['workers'][index].update(state='complete' if rc == 0 else 'failed', exit_status=rc)
                record['updated_utc'] = base.utc()
                base.save(ROOT / 'controller.json', record)
                time.sleep(3)
            for index, child in enumerate(children):
                rc = child.returncode
                record['workers'][index].update(state='complete' if rc == 0 else 'failed', exit_status=rc)
            if any(child.returncode for child in children):
                raise RuntimeError('A preparation shard failed; retain other shard outputs; no finalize or training')
            check_source()
            record['state'] = 'finalizing_cpu'
            base.save(ROOT / 'controller.json', record)
            if socket.gethostname().split('.')[0] != 'admin01':
                raise RuntimeError('Finalize belongs on admin01 CPU')
            log = ROOT / 'finalize.log'
            with log.open('a', buffering=1) as stream:
                argv = runner()
                stream.write(f'[R84 cache] {base.utc()} {shlex.join(argv)}\n')
                final = subprocess.Popen(argv, cwd=base.REPO, env=base.environment(),
                    stdout=stream, stderr=subprocess.STDOUT)
                record['finalization'] = dict(pid=final.pid, argv=argv, log=str(log), state='running')
                base.save(ROOT / 'controller.json', record)
                rc = final.wait()
            record['finalization'].update(state='complete' if rc == 0 else 'failed', exit_status=rc)
            if rc or not (PREPARED / 'prepared.json').is_file():
                raise RuntimeError('CPU finalize failed or prepared.json is missing')
            record.update(state='complete', finished_utc=base.utc(), exit_status=0,
                prepared_json=str(PREPARED / 'prepared.json'))
            base.save(ROOT / 'controller.json', record)
            return 0
        except BaseException as error:
            # Existing workers are never stopped. Keep reservations until every
            # started shard exits, and preserve each original return code.
            for index, child in enumerate(children):
                rc = child.wait()
                record['workers'][index].update(state='complete' if rc == 0 else 'failed', exit_status=rc)
            record.update(state='failed', finished_utc=base.utc(), exit_status=1, error=str(error))
            base.save(ROOT / 'controller.json', record)
            raise
        finally:
            for stream in streams:
                stream.close()


def start():
    if socket.gethostname().split('.')[0] != 'admin01':
        raise RuntimeError('Preparation controller must run on admin01')
    ROOT.mkdir(parents=True, exist_ok=True)
    with base.lock(ROOT / 'start.lock'):
        if (ROOT / 'ready.json').exists():
            raise RuntimeError('Preparation already armed; no duplicate or retry')
        if not (DATA / 'scenario_manifest.json').is_file():
            raise RuntimeError('Published release scenario_manifest.json missing')
        base.save(ROOT / 'ready.json', dict(confirmed_utc=base.utc(), main_authorized=True,
            cache_preparation_only=True, training_authorized=False, training_started=False,
            CPU_tests='code.V4.test_r84_prepare: 4 passed before launch',
            prepare_sha256=base.sha(SCRIPT), token=uuid.uuid4().hex,
            source_release=str(DATA), prepared_root=str(PREPARED), shared_cache=str(CACHE),
            reuse_cache_root=str(base.transport.CACHE), slots=list(SLOTS), cpus=list(CPUS),
            workers=4, splits=['train', 'val'], CPU_finalize_once_after_all_rc0=True), exclusive=True)
        argv = ['bash', '-lc', base.activated(command('controller'))]
        subprocess.run(['tmux', 'new-session', '-d', '-s', SESSION, shlex.join(argv)], check=True)
        subprocess.run(['tmux', 'set-option', '-t', SESSION, 'remain-on-exit', 'on'], check=True)
        base.save(ROOT / 'tmux.json', dict(created_utc=base.utc(), session=SESSION, argv=argv), exclusive=True)
    print(str(ROOT / 'controller.json'), flush=True)
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('start', 'controller', 'entry', 'worker'))
    parser.add_argument('--rank', type=int, choices=range(4))
    args = parser.parse_args()
    if args.stage in ('entry', 'worker') and args.rank is None:
        parser.error('--rank is required')
    return dict(start=start, controller=controller,
        entry=lambda: entry(args.rank), worker=lambda: worker(args.rank))[args.stage]()


if __name__ == '__main__':
    raise SystemExit(main())
