"""One-shot R83 launch through existing Slurm-owned containers; never allocates."""

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import socket
import subprocess
import sys
import tempfile
import time
from unittest import mock
import uuid

from . import r81_queue as transport


REPO = transport.REPO
ROOT = REPO / 'code/V4/runs/R83_model_only_source_embedding'
GROUP = 'model_only_seed2'
SESSION = 'r83_model_only_once'
CANDIDATES = transport.CANDIDATES
CONTAINERS = transport.CONTAINERS
CACHE = transport.CACHE
PREPARED = transport.PREPARED
DATA = transport.DATA
EMBEDDINGS = ROOT / 'model_only_vectors.pt'
FONT = Path('/usr/share/fonts/google-noto-cjk/NotoSansCJK-Regular.ttc')
CONDA_SH = Path('/public/home/shiys/miniconda3/etc/profile.d/conda.sh')
COMPAT = '/public/home/shiys/easynco_v3_bridge/exports/train_100k_20261002/python_compat'
DEFAULT_CANDIDATE = 'A_ce_only'
POLL_SECONDS = 60
LOADED_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
capture = transport.capture
read = transport.read


def utc():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def output_path(name):
    path = ROOT / name
    if path.parent != ROOT or not name.startswith('launch_'):
        raise ValueError('Only R83 launch receipts/logs may be written')
    ROOT.mkdir(parents=True, exist_ok=True)
    return path


def write(name, record, exclusive=False):
    path = output_path(name)
    temporary = output_path(f'launch_tmp_{os.getpid()}_{uuid.uuid4().hex}.json')
    try:
        with temporary.open('x') as output:
            json.dump(record, output, indent=2, allow_nan=False)
            output.write('\n')
            output.flush()
            os.fsync(output.fileno())
        if exclusive:
            # Publish the ready/claim atomically without ever replacing an
            # existing one-shot receipt or exposing partly written JSON.
            os.link(temporary, path)
        else:
            temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


@contextmanager
def locked(path):
    # Same flock files as R81/R82, without changing their receipt contents.
    with Path(path).open('a+') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield handle


def status(state, **fields):
    record = dict(experiment='R83', group=GROUP, state=state, updated_utc=utc(),
        controller_pid=os.getpid(), host=socket.gethostname(), tmux=SESSION,
        training_started=False, automatic_retry=False)
    record.update(fields)
    write('launch_status.json', record)
    print(json.dumps(record, sort_keys=True), flush=True)


def activated(argv):
    return '\n'.join(('set -euo pipefail', f'source {shlex.quote(str(CONDA_SH))}',
        'conda activate easynco', f'cd {shlex.quote(str(REPO))}',
        'export PYTHONDONTWRITEBYTECODE=1', 'exec ' + shlex.join(argv)))


def module_command(stage, candidate=None, token=None):
    result = [sys.executable, '-B', '-u', '-m', 'code.V4.r83_launch', stage]
    if candidate:
        result += ['--candidate', candidate]
    if token:
        result += ['--token', token]
    return result


def srun_command(candidate, stage, token=None):
    spec = CANDIDATES[candidate]
    probe = stage == 'node-probe'
    return ['srun', f'--jobid={spec["job"]}', '--overlap', '--exact', '--gres=none',
        '--immediate=5', '-N1', '-n1', f'--nodelist={spec["node"]}',
        f'--cpus-per-task={1 if probe else 4}', f'--mem={"256M" if probe else "9G"}',
        'bash', '-lc', activated(module_command(stage, candidate, token))]


def clean_exec_json(raw):
    raw = re.sub(r'\ADistribute Task Container\.\r?\ntask_id:\d+\r?\n', '', raw)
    return json.loads(raw)


def container_command(candidate, stage, token=None):
    spec = CANDIDATES[candidate]
    identity = CONTAINERS[spec['node']][1]
    # The enclosing srun shell has already activated easynco. Use that exact
    # interpreter in the immutable container; never inspect/pass credentials.
    environment = dict(CUDA_VISIBLE_DEVICES=spec['uuid'] if token else '',
        SLURM_JOB_ID=str(spec['job']), SLURM_STEP_ID='existing_allocation_container',
        PYTHONPATH=COMPAT + ':' + str(REPO), PYTHONDONTWRITEBYTECODE='1',
        PYTHONUNBUFFERED='1', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2',
        OPENBLAS_NUM_THREADS='2', NUMEXPR_NUM_THREADS='2', MPLBACKEND='Agg',
        PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True',
        WANDB_PROJECT='selector', WANDB_ENTITY='yjkds-southern-university-of-science-technology',
        WANDB_INIT_TIMEOUT='30', WANDB_HTTP_TIMEOUT='15')
    if token:
        environment['R83_LAUNCH_TOKEN'] = token
    result = ['sudo', '-n', '/usr/bin/ai_docker', 'exec', '-u', str(os.getuid())]
    for key, value in environment.items():
        result += ['-e', key + '=' + value]
    return result + [identity, *module_command(stage, candidate, token)]


def assert_container(candidate):
    spec = CANDIDATES[candidate]
    identity = CONTAINERS[spec['node']][1]
    cgroup = Path('/proc/self/cgroup').read_text()
    if identity not in cgroup or socket.gethostname() != identity[:12]:
        raise RuntimeError('Not in the immutable allocation-owned container cgroup')
    if os.environ.get('SLURM_JOB_ID') != str(spec['job']):
        raise RuntimeError('Allocation/container job identity mismatch')
    if Path(sys.prefix).name != 'easynco':
        raise RuntimeError('The activated easynco interpreter is required')
    return dict(id=identity, cgroup=cgroup, job=spec['job'],
        cpus=sorted(os.sched_getaffinity(0)), conda_prefix=sys.prefix)


def container_probe(candidate):
    container = assert_container(candidate)
    return dict(container=container, cache=transport.cache_inventory(),
        cjk_font_exists=FONT.is_file(),
        gpus={arm: transport.gpu_inventory(spec) for arm, spec in CANDIDATES.items()
            if spec['node'] == CANDIDATES[candidate]['node']})


def process_inventory(gpus):
    gpu_pids = {int(row[1]) for gpu in gpus.values() for row in gpu['compute_processes']}
    found_gpu_pids, rows = set(), []
    for path in Path('/proc').glob('[0-9]*'):
        try:
            details = dict(line.split(':', 1) for line in path.joinpath('status').read_text().splitlines() if ':' in line)
            namespace_pids = {int(value) for value in details.get('NSpid', path.name).split()}
            matched = gpu_pids & namespace_pids
            if not matched and path.stat().st_uid != os.getuid():
                continue
            command = path.joinpath('cmdline').read_bytes().decode(errors='replace').split('\0')
            model_stage = any(part.startswith('code.V4.') and part.endswith(('_experiment', '_reporting'))
                for part in command)
            if not matched and (not model_stage or path.stat().st_uid != os.getuid()):
                continue
            if details.get('State', '').strip().startswith(('Z', 'X')):
                continue
            found_gpu_pids.update(matched)
            rows.append(dict(pid=int(path.name), namespace_pids=sorted(namespace_pids),
                command=[part for part in command if part],
                cpus=sorted(os.sched_getaffinity(int(path.name))),
                cgroup=path.joinpath('cgroup').read_text(), matched_gpu_pids=sorted(matched)))
        except (FileNotFoundError, ProcessLookupError):
            continue
        except PermissionError:
            if int(path.name) in gpu_pids:
                raise RuntimeError('Cannot prove CPU affinity of an existing GPU process') from None
    if gpu_pids - found_gpu_pids:
        raise RuntimeError('An existing GPU process is not attributable through /proc; fail closed')
    return rows


def node_probe(candidate):
    spec = CANDIDATES[candidate]
    if socket.gethostname().split('.')[0] != spec['node'] or os.environ.get('SLURM_JOB_ID') != str(spec['job']):
        raise RuntimeError('Probe/entry requires the existing allocation CPU step')
    step_cgroup = Path('/proc/self/cgroup').read_text()
    if f'/job_{spec["job"]}/' not in step_cgroup:
        raise RuntimeError('CPU entry is outside the existing Slurm allocation cgroup')
    allocation = transport.allocation(spec)
    name, identity = CONTAINERS[spec['node']]
    fmt = '{{.Id}},{{.State.Running}},{{.State.Pid}},{{.HostConfig.Memory}}'
    values = capture(['sudo', '-n', '/usr/bin/ai_docker', 'inspect', '--format=' + fmt, name]).strip().split(',')
    if len(values) != 4 or values[:2] != [identity, 'true']:
        raise RuntimeError('Allocation-owned container ID/running-state mismatch')
    pid, memory = int(values[2]), int(values[3])
    if not 0 < memory <= allocation['memory_bytes']:
        raise RuntimeError('Container memory cap exceeds owned allocation')
    if identity not in Path(f'/proc/{pid}/cgroup').read_text():
        raise RuntimeError('Container host PID/cgroup identity mismatch')
    cpus = sorted(set(os.sched_getaffinity(pid)) & set(allocation['cpus']))
    nested = clean_exec_json(capture(container_command(candidate, 'container-probe'), timeout=90))
    if not set(nested['container']['cpus']) <= set(allocation['cpus']):
        raise RuntimeError('Container CPU affinity exceeds owned allocation')
    processes = process_inventory(nested['gpus'])
    occupied = {cpu for row in processes for cpu in row['cpus']}
    selected = sorted(set(cpus) - occupied)[:4]
    return dict(captured_utc=utc(), candidate=candidate, allocation=allocation,
        allocation_step=dict(step_id=os.environ.get('SLURM_STEP_ID'), cgroup=step_cgroup),
        container=dict(nested['container'], name=name, memory_limit_bytes=memory, allowed_cpus=cpus),
        selected_cpus=selected, processes=processes,
        gpus=nested['gpus'], cache=nested['cache'], cjk_font_exists=nested['cjk_font_exists'])


def remote_probe(candidate):
    return json.loads(capture(srun_command(candidate, 'node-probe'), timeout=150))


def predecessor_terminal(candidate):
    spec = CANDIDATES[candidate]
    old = read(transport.OLD_ROOT / f'launch_{candidate}.json')
    okay, reason = transport.old_terminal(old, spec)
    if not okay:
        return False, reason
    for root, group in ((transport.ROOT, transport.GROUP),
            (REPO / 'code/V4/runs/R82_context_matched_ranking', 'B_context_matched_rank_seed2')):
        path = root / f'launch_{group}.json'
        if not path.exists():
            claim = root / 'queue_claim.json'
            if claim.exists() and read(claim).get('candidate') == candidate:
                return False, f'{root.name} has a claim without a terminal worker receipt'
            continue
        record = read(path)
        if record.get('candidate') != candidate:
            continue
        if record.get('state') not in ('complete', 'failed') or not record.get('finished_utc'):
            return False, f'{root.name} pipeline is not terminal'
        rc = record.get('exit_status')
        if not isinstance(rc, int) or (record['state'] == 'failed') != (rc != 0):
            return False, f'{root.name} terminal exit status is invalid'
    return True, 'Previous full pipelines are terminal; NVML/process/flock gates also required'


def usable(candidate, probe):
    okay, reason = predecessor_terminal(candidate)
    if not okay:
        return False, reason
    spec = CANDIDATES[candidate]
    gpu = probe['gpus'][candidate]
    if gpu['uuid'] != spec['uuid'] or gpu['compute_processes'] or gpu['memory_used_mib'] > 512 or gpu['utilization_pct']:
        return False, 'Selected GPU is not idle: exact UUID, zero compute processes/utilization and <=512 MiB required'
    selected = set(probe['selected_cpus'])
    occupied = {cpu for process in probe['processes'] for cpu in process['cpus']}
    if len(selected) != 4 or not selected <= set(probe['container']['allowed_cpus']) or selected & occupied:
        return False, 'Four allocation/container CPUs disjoint from existing model stages are unavailable'
    return True, reason


def source_hashes():
    names = ('r83_launch.py', 'r83_experiment.py', 'r83_reporting.py', 'run_v4_r83.sh',
        'r81_queue.py', 'r80_experiment.py', 'r80_reporting.py', 'r77_experiment.py',
        'r76_experiment.py', 'r77_data.py', 'r77_lib.py', 'r77_best_model_report.py',
        'solver_code_encoder.py', 'solver_code_embeddings.py', 'pairwise_selector.py',
        'pairwise_objective.py', 'train.py', 'nss_comparison.py')
    paths = [REPO / 'code/V4' / name for name in names]
    paths += [PREPARED / 'prepared.json', PREPARED / 'identicalinit_seed2.pt',
        EMBEDDINGS, EMBEDDINGS.with_suffix('.lock.json')]
    return {str(path): digest(path) for path in paths}


def fresh():
    if (ROOT / GROUP).exists() and any((ROOT / GROUP).iterdir()):
        raise RuntimeError('R83 model output already exists; no implicit resume/restart')
    for name in ('launch_claim.json', 'launch_controller.json', 'launch_worker.json'):
        if (ROOT / name).exists():
            raise RuntimeError(f'{name} exists; exactly one attempt, never a duplicate/retry')


def assert_ready(candidate):
    ready = read(ROOT / 'launch_ready.json')
    if (ready.get('experiment') != 'R83' or ready.get('group') != GROUP
            or ready.get('candidate') != candidate or ready.get('main_thread_ready') is not True
            or ready.get('runner_cpu_tests_passed') is not True
            or not ready.get('main_ready_evidence') or not ready.get('confirmed_utc')):
        raise RuntimeError('Explicit main readiness, runner tests and frozen single-GPU identity required')
    current = source_hashes()
    if current != ready.get('source_sha256') or current.get(str(Path(__file__).resolve())) != LOADED_SHA256:
        raise RuntimeError('Source/preparation/embedding changed after ready; explicit review required')
    return ready


def runner_commands(candidate, token=None):
    spec = CANDIDATES[candidate]
    train = [sys.executable, '-B', '-u', '-m', 'code.V4.r83_experiment', 'train',
        '--root', str(ROOT), '--group', GROUP, '--prepared-root', str(PREPARED),
        '--data-root', str(DATA), '--cache-root', str(CACHE), '--embeddings', str(EMBEDDINGS),
        '--device', 'cuda:0', '--gpu-type', spec['gpu_type'],
        '--large-micro-cap', str(spec['cap']), '--eval-batch-size', '256',
        '--log-every', '18', '--wandb-mode', 'auto']
    lib = module_command('lib-worker', candidate, token)
    cpu = [sys.executable, '-B', '-u', '-m', 'code.V4.r83_reporting', '--root', str(ROOT),
        '--group', GROUP, '--device', 'cpu', '--lib-batch-size', '2']
    return train, lib, cpu


def cli_check(candidate):
    from . import r83_experiment as runner, r83_reporting as reporter
    train, _, report = runner_commands(candidate)
    parsed = runner.parser().parse_args(train[train.index('-m') + 2:])
    parsed_report = reporter.parser().parse_args(report[report.index('-m') + 2:])
    spec = runner.engine.arm_spec(runner.ENGINE_ARM)
    if (runner.GROUP, spec.batch_size, spec.max_updates) != (GROUP, 1024, 49680):
        raise RuntimeError('Runner identity/logical batch/fixed update budget mismatch')
    if (parsed.arm != GROUP or parsed.root != ROOT or parsed.device != 'cuda:0'
            or parsed.engine_arm != 'D_code_embedding' or parsed.experiment != 'R83'
            or parsed.gpu_type != CANDIDATES[candidate]['gpu_type']
            or parsed.large_micro_cap != CANDIDATES[candidate]['cap']
            or parsed.embeddings != EMBEDDINGS or parsed_report.root != ROOT
            or parsed_report.arm != GROUP or parsed.eval_batch_size != 256
            or parsed.resume or parsed_report.device != 'cpu'):
        raise RuntimeError('Fresh train/evaluation/CPU reporting CLI mismatch')
    return dict(checked_utc=utc(), passed=True, candidate=candidate, train_argv=train,
        cpu_report_argv=report, CPU_only=True, training_started=False,
        GPU_contexts_created=0, logical_batch=1024, successful_updates=49680, seed=2)


def standby(candidate):
    if socket.gethostname().split('.')[0] != 'admin01':
        raise RuntimeError('Persistent controller and font-capable CPU reporter belong on admin01')
    with locked(output_path('launch_control.lock')):
        path = ROOT / 'launch_tmux.json'
        if path.exists():
            previous = read(path)
            alive = subprocess.run(['tmux', 'has-session', '-t', SESSION],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0
            if (alive and previous.get('candidate') == candidate
                    and previous.get('launch_sha256') == LOADED_SHA256 and capture(
                        ['tmux', 'display-message', '-p', '-t', SESSION, '#{pane_dead}']).strip() == '0'):
                return previous
            raise RuntimeError('Standby already attempted; no implicit tmux restart')
        fresh()
        command = shlex.join(['bash', '-lc', activated(module_command('loop', candidate))])
        record = dict(created_utc=utc(), candidate=candidate, tmux=SESSION,
            main_thread_ready=False, command=command, launch_sha256=LOADED_SHA256)
        write('launch_tmux.json', record, exclusive=True)
        subprocess.run(['tmux', 'new-session', '-d', '-s', SESSION, command], check=True)
        subprocess.run(['tmux', 'set-option', '-t', SESSION, 'remain-on-exit', 'on'], check=True)
        return record


def arm(args):
    if not args.main_ready_evidence or not args.runner_tests_passed:
        raise RuntimeError('Only main may arm, with explicit ready evidence and focused runner test confirmation')
    if socket.gethostname().split('.')[0] != 'admin01' or not FONT.is_file():
        raise RuntimeError('admin01 existing CJK font is required for the automatic CPU PDF stage')
    fresh()
    from . import r83_experiment as runner
    bundle = runner.load_bundle(EMBEDDINGS)
    del bundle
    cli = cli_check(args.candidate)
    hashes = source_hashes()
    tests = read(ROOT / 'launch_selftest.json')
    if tests.get('passed') is not True or tests.get('launch_sha256') != LOADED_SHA256:
        raise RuntimeError('Run the focused CPU launch selftest before arm')
    write('launch_cli_check.json', cli)
    with locked(output_path('launch_control.lock')):
        fresh()
        write('launch_ready.json', dict(experiment='R83', group=GROUP,
            candidate=args.candidate, confirmed_utc=utc(), main_thread_ready=True,
            runner_cpu_tests_passed=True, main_ready_evidence=args.main_ready_evidence,
            source_sha256=hashes, W_and_B='runner standard lookup; explicit recorded offline fallback',
            seed=2, batch=1024, updates=49680, eval_batch_size=256), exclusive=True)
    return standby(args.candidate)


def claim_for(candidate, token):
    claim = read(ROOT / 'launch_claim.json')
    if (not token or claim.get('token') != token or claim.get('candidate') != candidate
            or claim.get('group') != GROUP):
        raise RuntimeError('Missing or mismatched one-shot R83 claim')
    return claim


def assert_worker(candidate, token):
    claim = claim_for(candidate, token)
    assert_ready(candidate)
    proof = assert_container(candidate)
    if os.environ.get('R83_LAUNCH_TOKEN') != token or os.environ.get('CUDA_VISIBLE_DEVICES') != CANDIDATES[candidate]['uuid']:
        raise RuntimeError('Worker token or exactly-one-UUID binding mismatch')
    cpus = set(claim['selected_cpus'])
    if len(cpus) != 4 or not cpus <= set(os.sched_getaffinity(0)):
        raise RuntimeError('Claimed CPUs unavailable inside allocation container')
    os.sched_setaffinity(0, cpus)
    proof['assigned_cpus'] = sorted(cpus)
    return claim, proof


def run_stage(name, command, record, receipt_name, env=None):
    stage = dict(stage=name, argv=command, started_utc=utc(), state='running',
        log=str(output_path(f'launch_{name}.log')))
    record.setdefault('stages', []).append(stage)
    record.update(state='running', phase=name)
    write(receipt_name, record)
    with Path(stage['log']).open('a', buffering=1) as log:
        log.write(f'\n[R83] {utc()} {shlex.join(command)}\n')
        child = subprocess.Popen(command, cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT)
        stage['pid'] = child.pid
        write(receipt_name, record)
        try:
            rc = child.wait()
        except BaseException:
            # Retain the UUID reservation until our own child exits; never
            # orphan a live stage and make its GPU appear available to a queue.
            child.wait()
            raise
        log.write(f'\n[R83] {utc()} exit_status={rc}\n')
    stage.update(state='complete' if rc == 0 else 'failed', exit_status=rc, finished_utc=utc())
    write(receipt_name, record)
    if rc:
        raise RuntimeError(f'{name} exit={rc}; no later stage and no automatic retry')


def container_entry(candidate, token):
    claim = claim_for(candidate, token)
    assert_ready(candidate)
    probe = node_probe(candidate)
    okay, reason = usable(candidate, probe)
    occupied = {cpu for row in probe['processes'] for cpu in row['cpus']}
    if not okay or set(claim['selected_cpus']) & occupied:
        raise RuntimeError(f'Child-side idle/CPU recheck failed: {reason}')
    if not set(claim['selected_cpus']) <= set(probe['container']['allowed_cpus']):
        raise RuntimeError('Claimed CPUs changed; no reassignment or retry')
    write('launch_container_entry.json', dict(checked_utc=utc(), resource_proof=probe), exclusive=True)
    return subprocess.run(container_command(candidate, 'container-worker', token), check=False).returncode


def container_worker(candidate, token):
    claim, proof = assert_worker(candidate, token)
    record = dict(started_utc=utc(), candidate=candidate, group=GROUP,
        container=proof, state='preflight', stages=[], automatic_retry=False)
    with locked(output_path('launch_worker.lock')):
        write('launch_worker.json', record, exclusive=True)
        try:
            fresh_outputs = ROOT / GROUP
            if fresh_outputs.exists() and any(fresh_outputs.iterdir()):
                raise RuntimeError('Fresh-only worker refuses existing model outputs')
            record['cache'] = transport.cache_inventory()
            gpu = transport.gpu_inventory(CANDIDATES[candidate])
            if gpu['compute_processes'] or gpu['memory_used_mib'] > 512 or gpu['utilization_pct']:
                raise RuntimeError('GPU occupied at final in-container check')
            train, lib, _ = runner_commands(candidate, token)
            for name, command in (('train_VALlock_fullTEST', train), ('twoLIB', lib)):
                assert_ready(candidate)
                run_stage(name, command, record, 'launch_worker.json')
            record.update(state='complete', exit_status=0, finished_utc=utc())
            write('launch_worker.json', record)
            return 0
        except BaseException as error:
            record.update(state='failed', exit_status=1, error=f'{type(error).__name__}: {error}', finished_utc=utc())
            write('launch_worker.json', record)
            raise


def library_worker(candidate, token):
    assert_worker(candidate, token)
    from . import r83_reporting as reporter
    run, config, lock, test, _ = reporter.locked_inputs(ROOT, GROUP)
    with locked(run / 'postprocess.lock'):
        def phase(name, **fields):
            write('launch_lib_status.json', dict(updated_utc=utc(), phase=name, **fields))
        results = reporter.engine.r80_lib(ROOT, run, config, lock, 'cuda:0', 2, phase,
            test['identity']['model_state_sha256'], factory_module=reporter.FACTORY_MODULE)
        if set(results) != {'TSPLIB', 'CVRPLIB'}:
            raise RuntimeError('Both same-lock LIB datasets must complete before CPU reporting')
        paths = [run / 'lib_evaluation/predictions' / (dataset + '.npz') for dataset in results]
        write('launch_lib_complete.json', dict(completed_utc=utc(), exit_status=0,
            checkpoint_sha256=lock['checkpoint_sha256'], predictions_sha256={str(p): digest(p) for p in paths},
            training=False, TEST_inference=False, PDF_rendered=False), exclusive=True)
    return 0


def assert_saved_libraries():
    receipt = read(ROOT / 'launch_lib_complete.json')
    lock = read(ROOT / GROUP / 'final_test_lock.json')
    expected = {str(ROOT / GROUP / 'lib_evaluation/predictions' / (dataset + '.npz'))
        for dataset in ('TSPLIB', 'CVRPLIB')}
    if (receipt.get('exit_status') != 0 or receipt.get('checkpoint_sha256') != lock['checkpoint_sha256']
            or set(receipt.get('predictions_sha256', {})) != expected):
        raise RuntimeError('CPU PDF requires both checked same-lock saved LIB predictions')
    if any(digest(path) != value for path, value in receipt['predictions_sha256'].items()):
        raise RuntimeError('Saved LIB predictions changed; never rerun training or infer on admin01')


def pipeline(candidate, claim):
    record = dict(started_utc=utc(), group=GROUP, candidate=candidate,
        gpu_uuid=CANDIDATES[candidate]['uuid'], stages=[], automatic_retry=False)
    write('launch_controller.json', record, exclusive=True)
    try:
        run_stage('allocation_train_TEST_LIB', srun_command(candidate, 'container-entry', claim['token']),
            record, 'launch_controller.json')
        worker = read(ROOT / 'launch_worker.json')
        if worker.get('state') != 'complete' or worker.get('exit_status') != 0:
            raise RuntimeError('No CPU report until container pipeline rc0')
        assert_ready(candidate)
        assert_saved_libraries()
        if socket.gethostname().split('.')[0] != 'admin01' or not FONT.is_file():
            raise RuntimeError('Automatic PDF stage needs admin01 existing CJK font')
        _, _, report = runner_commands(candidate)
        environment = dict(os.environ, CUDA_VISIBLE_DEVICES='', MPLBACKEND='Agg',
            PYTHONPATH=COMPAT + ':' + str(REPO), OMP_NUM_THREADS='2', MKL_NUM_THREADS='2',
            OPENBLAS_NUM_THREADS='2', NUMEXPR_NUM_THREADS='2', PYTHONDONTWRITEBYTECODE='1')
        run_stage('CPU_saved_predictions_PDF', report, record, 'launch_controller.json', environment)
        for name in ('MODEL_STATISTICS.pdf', 'selection_distributions.pdf', 'training_and_method_diagnostics.pdf'):
            path = ROOT / GROUP / 'full_model_statistics' / name
            if not path.is_file() or path.stat().st_size == 0:
                raise RuntimeError(f'Final required PDF missing: {name}')
        record.update(state='complete', exit_status=0, finished_utc=utc())
        write('launch_controller.json', record)
        status('complete', training_started=True, final_report_complete=True)
        return 0
    except BaseException as error:
        record.update(state='failed', exit_status=1, finished_utc=utc(), error=f'{type(error).__name__}: {error}')
        write('launch_controller.json', record)
        worker_path = ROOT / 'launch_worker.json'
        worker = read(worker_path) if worker_path.is_file() else {}
        status('failed', training_started=bool(worker.get('stages')),
            final_report_complete=False, error=record['error'])
        raise


def loop(candidate):
    if socket.gethostname().split('.')[0] != 'admin01':
        raise RuntimeError('Controller must stay on admin01; training belongs inside allocation container')
    with locked(output_path('launch_loop.lock')):
        fresh()
        previous = None
        while True:
            if not (ROOT / 'launch_ready.json').exists():
                message = 'Waiting for explicit main ready, runner tests and frozen embedding; no remote probes'
            else:
                try:
                    assert_ready(candidate)
                except (OSError, ValueError, RuntimeError) as error:
                    status('ready_rejected', candidate=candidate, reason=str(error))
                    raise
                try:
                    proof = remote_probe(candidate)
                    okay, message = usable(candidate, proof)
                except (RuntimeError, subprocess.TimeoutExpired, OSError) as error:
                    okay, message = False, f'Resource not ready: {type(error).__name__}: {error}'
                if okay:
                    gpu_lock = transport.gpu_lock_path(CANDIDATES[candidate]['uuid'])
                    claimed = False
                    try:
                        with locked(gpu_lock):
                            assert_ready(candidate)
                            fresh()
                            claim = dict(group=GROUP, candidate=candidate, token=uuid.uuid4().hex,
                                claimed_utc=utc(), gpu_uuid=CANDIDATES[candidate]['uuid'],
                                selected_cpus=proof['selected_cpus'], resource_proof=proof,
                                shared_uuid_lock=str(gpu_lock), no_retry=True)
                            write('launch_claim.json', claim, exclusive=True)
                            claimed = True
                            return pipeline(candidate, claim)
                    except BlockingIOError:
                        if claimed:
                            raise
                        message = 'Selected UUID is reserved by an existing R81/R82/R83 pipeline'
            if previous != message:
                status('waiting', reason=message, candidate=candidate)
                previous = message
            time.sleep(POLL_SECONDS)


def inspect_resources():
    nodes = {}
    for candidate in ('B_batch512', 'A_ce_only'):
        try:
            nodes[CANDIDATES[candidate]['node']] = remote_probe(candidate)
        except (RuntimeError, subprocess.TimeoutExpired, OSError) as error:
            nodes[CANDIDATES[candidate]['node']] = dict(error=f'{type(error).__name__}: {error}')
    eligibility = {}
    for candidate, spec in CANDIDATES.items():
        proof = nodes[spec['node']]
        if 'error' in proof:
            eligibility[candidate] = dict(eligible=False, reason=proof['error'])
        else:
            okay, reason = usable(candidate, proof)
            eligibility[candidate] = dict(eligible=okay, reason=reason)
    record = dict(captured_utc=utc(), host=socket.gethostname(), nodes=nodes,
        eligibility=eligibility,
        preferred_candidate=DEFAULT_CANDIDATE, admin01_cjk_font_exists=FONT.is_file(),
        credentials_read=False, process_environment_read=False, CUDA_contexts_created=0,
        new_allocations=0, training_started=False, TEST_started=False, report_started=False)
    write('launch_resources.json', record)
    return record


def plan(candidate):
    train, lib, cpu = runner_commands(candidate, '<ONE_SHOT_CLAIM_TOKEN>')
    record = dict(experiment='R83', root=str(ROOT), group=GROUP, candidate=candidate,
        resource=CANDIDATES[candidate], container=CONTAINERS[CANDIDATES[candidate]['node']],
        batch=1024, seed=2, successful_updates=49680, eval_batch_size=256,
        microcap=CANDIDATES[candidate]['cap'], cache=str(CACHE), cache_reuse_only=True,
        embeddings=str(EMBEDDINGS), explicit_main_ready_required=True, training_started=False,
        new_allocation=False, direct_ssh_training=False, credential_inspection=False,
        new_seeds=False, retries=False, commits=False, tmux=SESSION,
        GPU_stages=[train, lib], admin01_CPU_report=cpu,
        test_policy='train invokes exactly the locked full TEST; no additional test command',
        report_policy='twoLIB rc0 -> saved-prediction CPU PDF on admin01; never train again')
    snapshot = ROOT / 'launch_resources.json'
    if snapshot.is_file():
        resources = read(snapshot)
        eligibility = {}
        for arm, spec in CANDIDATES.items():
            proof = resources['nodes'][spec['node']]
            if 'error' in proof:
                eligibility[arm] = dict(eligible=False, reason=proof['error'])
            else:
                okay, reason = usable(arm, proof)
                eligibility[arm] = dict(eligible=okay, reason=reason)
        record.update(resource_snapshot=str(snapshot), resource_snapshot_utc=resources['captured_utc'],
            snapshot_eligibility=eligibility, recheck_before_claim_and_in_container=True)
    write('launch_plan.json', record)
    return record


def selftest():
    module = sys.modules[__name__]
    checks = []
    with tempfile.TemporaryDirectory(prefix='r83_launch_') as directory:
        with mock.patch.object(module, 'ROOT', Path(directory)):
            fresh()
            write('launch_claim.json', dict(token='test'), exclusive=True)
            try:
                write('launch_claim.json', dict(token='duplicate'), exclusive=True)
            except FileExistsError:
                assert read(output_path('launch_claim.json'))['token'] == 'test'
                checks.append('atomic_exclusive_receipt_never_overwrites_existing_claim')
            else:
                raise AssertionError('Existing one-shot receipt overwritten')
            try:
                fresh()
            except RuntimeError:
                checks.append('irreversible_claim_blocks_duplicate_and_resume')
            else:
                raise AssertionError('Duplicate claim accepted')
            with locked(Path(directory) / 'uuid.lock'):
                try:
                    with locked(Path(directory) / 'uuid.lock'):
                        raise AssertionError('Duplicate UUID flock accepted')
                except BlockingIOError:
                    checks.append('shared_UUID_flock_exclusive')
            with mock.patch.object(module, 'capture', side_effect=AssertionError('Unexpected remote/credential call')):
                try:
                    assert_ready(DEFAULT_CANDIDATE)
                except FileNotFoundError:
                    checks.append('missing_ready_blocks_before_remote_probe')
            output_path('launch_claim.json').unlink()
            with mock.patch.object(module, 'remote_probe', side_effect=AssertionError('Pre-ready remote probe')):
                with mock.patch.object(time, 'sleep', side_effect=StopIteration), mock.patch.object(module, 'status'):
                    try:
                        loop(DEFAULT_CANDIDATE)
                    except StopIteration:
                        checks.append('actual_loop_waits_without_remote_probe_or_training_before_ready')
            record = dict(stages=[])
            child = mock.Mock(pid=123, wait=mock.Mock(return_value=7))
            with mock.patch.object(subprocess, 'Popen', return_value=child):
                try:
                    run_stage('synthetic_failure', ['not-executed'], record, 'launch_synthetic.json')
                except RuntimeError:
                    checks.append('nonzero_stage_stops_pipeline_without_retry')
                else:
                    raise AssertionError('Nonzero child exit accepted')
            try:
                assert_saved_libraries()
            except FileNotFoundError:
                checks.append('CPU_report_requires_saved_LIBs')
            with mock.patch.object(module, 'run_stage', side_effect=RuntimeError('synthetic nonzero stage')) as stage:
                with mock.patch.object(module, 'assert_saved_libraries', side_effect=AssertionError('Premature CPU stage')):
                    with mock.patch.object(module, 'status'):
                        try:
                            pipeline(DEFAULT_CANDIDATE, dict(token='test'))
                        except RuntimeError:
                            assert stage.call_count == 1
                            checks.append('actual_pipeline_never_runs_CPU_report_after_container_failure')
            proof = dict(gpus={DEFAULT_CANDIDATE: dict(uuid=CANDIDATES[DEFAULT_CANDIDATE]['uuid'],
                compute_processes=[], memory_used_mib=2, utilization_pct=0)}, selected_cpus=[1, 2, 3, 4],
                processes=[], container=dict(allowed_cpus=[1, 2, 3, 4]))
            with mock.patch.object(module, 'predecessor_terminal', return_value=(True, 'terminal')):
                assert usable(DEFAULT_CANDIDATE, proof)[0]
                proof['processes'] = [dict(cpus=[1, 2])]
                assert not usable(DEFAULT_CANDIDATE, proof)[0]
                checks.append('idle_GPU_is_insufficient_when_existing_process_CPUs_overlap')
    cli = cli_check(DEFAULT_CANDIDATE)
    checks += ['actual_runner_and_reporter_argv_parsed', 'seed2_batch1024_updates49680_eval256',
        'train_only_one_full_TEST_via_existing_runner', 'CPU_PDF_outside_allocation_container']
    receipt = dict(checked_utc=utc(), passed=True, checks=checks, launch_sha256=LOADED_SHA256,
        CPU_only=True, credentials_read=False, real_training_TEST_report_started=False, cli=cli)
    write('launch_selftest.json', receipt)
    return receipt


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('stage', choices=('plan', 'inspect', 'selftest', 'cli-check', 'standby', 'arm',
        'loop', 'node-probe', 'container-probe', 'container-entry', 'container-worker', 'lib-worker'))
    result.add_argument('--candidate', choices=tuple(CANDIDATES), default=DEFAULT_CANDIDATE)
    result.add_argument('--token')
    result.add_argument('--main-ready-evidence')
    result.add_argument('--runner-tests-passed', action='store_true')
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    if args.stage == 'plan':
        value = plan(args.candidate)
    elif args.stage == 'inspect':
        value = inspect_resources()
    elif args.stage == 'selftest':
        value = selftest()
    elif args.stage == 'cli-check':
        value = cli_check(args.candidate)
        write('launch_cli_check.json', value)
    elif args.stage == 'standby':
        value = standby(args.candidate)
    elif args.stage == 'arm':
        value = arm(args)
    elif args.stage == 'node-probe':
        value = node_probe(args.candidate)
    elif args.stage == 'container-probe':
        value = container_probe(args.candidate)
    elif args.stage == 'loop':
        return loop(args.candidate)
    elif args.stage == 'container-entry':
        return container_entry(args.candidate, args.token)
    elif args.stage == 'container-worker':
        return container_worker(args.candidate, args.token)
    elif args.stage == 'lib-worker':
        return library_worker(args.candidate, args.token)
    print(json.dumps(value, indent=2), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
