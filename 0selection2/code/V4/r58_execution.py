"""Persistent R58 execution shared by timing refinement and production.

RF mode is a single explicit global choice, never inferred from pilot results.
Each worker keeps its backend across groups and passes, writes only its own
atomic shard, and resets seed=2 before each unchanged inference batch. Timing
includes independent checking, durable worker writes, and parent merge writes.
MPS is opt-in, private, legacy/default-only, and restricted to the assigned GPU.
"""

import copy
import json
import math
import multiprocessing as mp
import os
import pickle
import re
import shutil
import socket
import subprocess
import tempfile
import time
import traceback
from multiprocessing.connection import wait
from pathlib import Path

from ..unified_selector.registry import POOLS
from .r58_backends import REVISED_RECIPE, RF_METHODS
from .r58_scenario import canonical_hash, check_route, fields


GPU_UUID = 'GPU-cb626bdb-4319-2f5e-b91f-17fcaa44a83d'
RF_MODES = ('serial1', 'normal4', 'mps4')
WORKER_THREAD_ENV = dict(OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')


def make_execution_plan(rf_mode):
    return validate_execution_plan(dict(revision=1, recipe=REVISED_RECIPE,
        seed=2, rf_mode=rf_mode, device_uuid=GPU_UUID))


def validate_execution_plan(plan):
    plan = copy.deepcopy(plan)
    if (plan.get('revision') != 1 or plan.get('recipe') != REVISED_RECIPE or
            type(plan.get('seed')) is not int or plan['seed'] != 2 or
            plan.get('rf_mode') not in RF_MODES or plan.get('device_uuid') != GPU_UUID):
        raise ValueError('Explicit revised R58 execution plan required; no inferred/default mode')
    return plan


def read_execution_plan(root):
    return validate_execution_plan(json.loads((Path(root) / 'execution_plan.json').read_text()))


def require_budget_approval(root, plan, hashes):
    approval = json.loads((Path(root) / 'execution_approval.json').read_text())
    hours = approval.get('whole_projected_gpu_hours')
    if (approval.get('approved') is not True or
            approval.get('execution_plan_sha256') != canonical_hash(plan) or
            approval.get('implementation_sha256') != hashes or
            approval.get('deployments') != sum(map(len, POOLS.values())) or
            isinstance(hours, bool) or not isinstance(hours, (int, float)) or
            not math.isfinite(hours) or not 0 < hours <= 100):
        raise ValueError('Whole-R58 <=100h budget approval does not match execution plan/frozen sources')
    return approval


def backend_batch_size(problem, method):
    return 1 if problem in ('TSP', 'CVRP', 'ATSP') or method in RF_METHODS else 16


def execution_profile(problem, method, plan):
    plan = validate_execution_plan(plan)
    mode = plan['rf_mode'] if method in RF_METHODS else 'serial1'
    return dict(mode=mode, workers=1 if mode == 'serial1' else 4,
        inference_batch_size=backend_batch_size(problem, method), seed=2,
        device_uuid=plan['device_uuid'], scheduler='persistent_workers_v1',
        thread_environment=WORKER_THREAD_ENV.copy(), torch_threads=1,
        assignment='fixed size-homogeneous batches, round-robin within each group',
        mps='private legacy service, existing binaries and default settings' if mode == 'mps4' else None)


def gpu_guard():
    if (socket.gethostname().split('.')[0] != 'gpu03' or
            os.environ.get('SLURM_JOB_ID') != '1465' or
            os.environ.get('CUDA_VISIBLE_DEVICES') != GPU_UUID or
            os.environ.get('CONDA_DEFAULT_ENV') != 'easynco'):
        raise RuntimeError('Execution restricted to easynco/job1465/gpu03 and the designated idle GPU UUID')


def _write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w') as stream:
        json.dump(value, stream, allow_nan=False, sort_keys=True)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def _token(value):
    token = str(value)
    if not re.fullmatch(r'[A-Za-z0-9_-]+', token):
        raise ValueError('Group/pass IDs must contain only letters, digits, underscore or hyphen')
    return token


def group_batches(problem, instances, batches, max_instances=100):
    """Group existing batches without splitting them or mixing true sizes."""
    groups, current, scale = [], [], None
    for batch in batches:
        n = fields(problem, instances[batch[0]])['n']
        if current and (n != scale or len(current) + len(batch) > max_instances):
            groups.append(dict(id=str(len(groups)), indices=current))
            current = []
        current.extend(batch)
        scale = n
    if current:
        groups.append(dict(id=str(len(groups)), indices=current))
    return groups


def checked_rows(problem, instances, rows, expected):
    expected = set(expected)
    result = {}
    for row in rows:
        index = row.get('index')
        if type(index) is not int or index not in expected or index in result:
            raise ValueError('Duplicate, unexpected or invalid shard index')
        checked = check_route(problem, instances[index], row['route'], row['reported_cost'])
        if (not checked['feasible'] or not checked['cost_matches'] or
                not math.isfinite(row['cost']) or row['cost'] != checked['cost']):
            raise ValueError(f'Independent shard check failed for instance {index}')
        result[index] = row
    return result


def merge_shards(problem, instances, paths, expected, metadata, output_path):
    merged = {}
    width = metadata['execution_profile']['inference_batch_size']
    batches = [expected[i:i + width] for i in range(0, len(expected), width)]
    for worker, path in enumerate(paths):
        payload = json.loads(Path(path).read_text())
        owned_batches = batches[worker::len(paths)]
        owned = [index for batch in owned_batches for index in batch]
        if (payload.get('metadata') != metadata or payload.get('worker') != worker or
                payload.get('batches') != owned_batches):
            raise ValueError('Shard metadata or worker ownership differs')
        rows = checked_rows(problem, instances, payload['rows'], owned)
        if any(row.get('worker') != worker or row.get('inference_batch') not in owned_batches
               for row in rows.values()):
            raise ValueError('Row batch/worker differs from the fixed execution assignment')
        if merged.keys() & rows.keys():
            raise ValueError('Duplicate index across worker shards')
        merged.update(rows)
    if set(merged) != set(expected):
        raise ValueError('Incomplete worker shards; no merged group can be published')
    rows = [merged[i] for i in sorted(merged)]
    _write_json(output_path, dict(metadata=metadata, rows=rows, complete=True))
    return rows


class PrivateMPS:
    """Never reuse, configure, or terminate a global/another user's MPS service."""

    def __init__(self):
        self.directory, self.env, self.daemon_pid = None, None, None
        self.guard_evidence = None

    def check_machine(self):
        processes = subprocess.run(['ps', '-eo', 'uid=,pid=,comm=,args='],
            capture_output=True, text=True, timeout=15, check=True).stdout
        mps_processes = []
        for line in processes.splitlines():
            pieces = line.split(None, 3)
            if len(pieces) >= 3 and (pieces[2].startswith('nvidia-cuda-mps') or
                    (len(pieces) == 4 and pieces[3].split()[0].split('/')[-1].startswith('nvidia-cuda-mps'))):
                mps_processes.append(dict(uid=int(pieces[0]), pid=int(pieces[1]),
                                         command=pieces[3] if len(pieces) == 4 else pieces[2]))
        mode = subprocess.run(['nvidia-smi', '-i', GPU_UUID, '--query-gpu=uuid,compute_mode',
            '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=15, check=True).stdout.strip()
        apps = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,gpu_uuid',
            '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=15, check=True).stdout
        target_clients = []
        for line in apps.splitlines():
            pieces = [value.strip() for value in line.split(',')]
            if len(pieces) == 2 and pieces[1] == GPU_UUID:
                if not pieces[0].isdigit():
                    raise RuntimeError('Unable to identify an existing target GPU context')
                target_clients.append(int(pieces[0]))
        self.guard_evidence = dict(checked_at=time.time(), process_scope='all users, ps -eo uid,pid,comm,args',
            process_count=len(processes.splitlines()), existing_mps_processes=mps_processes,
            target_gpu_mode=mode, existing_target_context_pids=target_clients,
            unrelated_gpu_contexts_ignored=True)
        default_mode = [value.strip() for value in mode.split(',')] == [GPU_UUID, 'Default']
        if mps_processes or target_clients or not default_mode:
            raise RuntimeError(f'MPS unavailable: existing daemon/context or non-default GPU mode: {self.guard_evidence}')
        return self.guard_evidence

    def control(self, command):
        result = subprocess.run(['/usr/bin/nvidia-cuda-mps-control'], input=command + '\n',
            text=True, capture_output=True, env=self.env, timeout=15, check=True)
        return result.stdout.strip()

    def __enter__(self):
        gpu_guard()
        for name in ('/usr/bin/nvidia-cuda-mps-control', '/usr/bin/nvidia-cuda-mps-server'):
            if not os.access(name, os.X_OK):
                raise RuntimeError('Existing official legacy MPS binaries are required')
        if os.getuid() == 0 or any(k.startswith('CUDA_MPS_') for k in os.environ):
            raise RuntimeError('Require unprivileged user and unchanged default MPS settings')
        self.check_machine()
        self.directory = Path(tempfile.mkdtemp(prefix=f'r58-mps-{os.getuid()}-', dir='/tmp'))
        for name in ('pipe', 'log'):
            (self.directory / name).mkdir(mode=0o700)
        self.env = dict(os.environ, CUDA_VISIBLE_DEVICES=GPU_UUID,
            CUDA_MPS_PIPE_DIRECTORY=str(self.directory / 'pipe'),
            CUDA_MPS_LOG_DIRECTORY=str(self.directory / 'log'))
        try:
            subprocess.run(['/usr/bin/nvidia-cuda-mps-control', '-d'], env=self.env,
                           capture_output=True, text=True, timeout=15, check=True)
            pid_path = self.directory / 'pipe/nvidia-cuda-mps-control.pid'
            deadline = time.monotonic() + 15
            while not pid_path.exists() and time.monotonic() < deadline:
                time.sleep(.05)
            self.daemon_pid = int(pid_path.read_text().strip())
            if Path(f'/proc/{self.daemon_pid}').stat().st_uid != os.getuid():
                raise RuntimeError('MPS daemon ownership differs from current user')
            return self
        except BaseException:
            self.close()
            raise

    def proof(self, worker_pids):
        raw_servers = self.control('get_server_list')
        servers = [int(line.strip()) for line in raw_servers.splitlines() if line.strip()]
        clients, raw_clients = set(), {}
        for server in servers:
            raw_clients[str(server)] = self.control(f'get_client_list {server}')
            clients.update(int(line.strip()) for line in raw_clients[str(server)].splitlines() if line.strip())
        if len(worker_pids) != 4 or clients != set(worker_pids):
            raise RuntimeError('MPS4 requires actual attachment of exactly the four owned CUDA workers')
        observed = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,gpu_uuid',
            '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=15, check=True).stdout
        mapping = {}
        for line in observed.splitlines():
            pieces = [value.strip() for value in line.split(',')]
            if len(pieces) == 2 and pieces[0].isdigit():
                mapping[int(pieces[0])] = pieces[1]
        if not servers or any(mapping.get(server) != GPU_UUID for server in servers):
            raise RuntimeError('MPS server GPU UUID is not independently confirmed')
        return dict(verified=True, hostname=socket.gethostname(), uid=os.getuid(),
            daemon_pid=self.daemon_pid, server_pids=servers, worker_pids=sorted(worker_pids),
            raw_servers=raw_servers, raw_clients=raw_clients,
            server_gpu_uuids={str(pid): mapping[pid] for pid in servers},
            pipe_directory=self.env['CUDA_MPS_PIPE_DIRECTORY'],
            log_directory=self.env['CUDA_MPS_LOG_DIRECTORY'], default_settings=True,
            startup_guard=self.guard_evidence,
            binaries=['/usr/bin/nvidia-cuda-mps-control', '/usr/bin/nvidia-cuda-mps-server'])

    def terminate_owned_clients(self, owned_pids):
        ownership = set(owned_pids)
        terminated = []
        for server in [int(line) for line in self.control('get_server_list').splitlines() if line.strip()]:
            clients = [int(line) for line in self.control(f'get_client_list {server}').splitlines() if line.strip()]
            if not set(clients) <= ownership:
                raise RuntimeError('Private MPS server contains an unowned client; no termination is allowed')
            for client in clients:
                result = self.control(f'terminate_client {server} {client}')
                if result.strip() != '0':
                    raise RuntimeError(f'MPS context termination was not CUDA_SUCCESS: {result}')
                terminated.append(dict(server_pid=server, client_pid=client, result=result))
        return terminated

    def close(self):
        if self.env is not None and (self.directory / 'pipe/nvidia-cuda-mps-control.pid').exists():
            for server in self.control('get_server_list').splitlines():
                if server.strip() and self.control(f'get_client_list {int(server)}').strip():
                    raise RuntimeError('Private MPS still has active clients; daemon/evidence retained')
            self.control('quit')
            deadline = time.monotonic() + 15
            while self.daemon_pid and Path(f'/proc/{self.daemon_pid}').exists() and time.monotonic() < deadline:
                time.sleep(.05)
            if self.daemon_pid and Path(f'/proc/{self.daemon_pid}').exists():
                raise RuntimeError('Owned MPS daemon did not exit; private evidence directory retained')
        if self.directory is not None:
            shutil.rmtree(self.directory)
            self.directory = None

    def __exit__(self, *unused):
        self.close()


def _worker(connection, config, instances, factory=None):
    """One process owns one backend and its disjoint group checkpoint files."""
    import torch
    from .r58_backends import make_backend
    from .r58_labels import seed_solver
    backend, command = None, {}
    try:
        if any(os.environ.get(key) != value for key, value in WORKER_THREAD_ENV.items()):
            raise RuntimeError('Worker native thread environment must be fixed BEFORE process/module startup')
        if factory is None:
            gpu_guard()
        elif config['device'] != 'cpu':
            raise RuntimeError('Synthetic backend injection is restricted to CPU-only tests')
        torch.set_num_threads(1)
        if config.get('mps_env'):
            os.environ.update(config['mps_env'])
        if instances is None:
            with Path(config['input_path']).open('rb') as stream:
                instances = pickle.load(stream)
        seed_solver(2)
        backend = (factory or make_backend)(config['problem'], config['method'], config['device'])
        profile = backend.profile()
        if config['expected_profile'] is not None and profile != config['expected_profile']:
            raise ValueError('Actual worker deployment differs from expected/locked profile')

        def synchronize():
            if config['device'] != 'cpu':
                torch.cuda.synchronize()

        def solve(batch):
            seed_solver(2)
            synchronize()
            start = time.perf_counter()
            output = backend.run([instances[i] for i in batch])
            synchronize()
            seconds = time.perf_counter() - start
            if len(output.costs) != len(batch) or len(output.tours) != len(batch):
                raise ValueError('Solver output count changed')
            rows = []
            for offset, index in enumerate(batch):
                checked = check_route(config['problem'], instances[index], output.tours[offset], output.costs[offset])
                if not checked['feasible'] or not checked['cost_matches']:
                    raise ValueError(f'Independent route check failed: {checked}')
                rows.append(dict(index=index, true_size=fields(config['problem'], instances[index])['n'],
                    cost=checked['cost'], reported_cost=float(output.costs[offset]), route=output.tours[offset],
                    feasible=True, inference_batch=batch, seconds=seconds / len(batch),
                    worker=config['worker'], seed=2, execution_witness=output.audit.get('execution_witness')))
            return rows

        synchronize()
        connection.send(dict(kind='loaded', pid=os.getpid(), profile=profile))
        while True:
            command = connection.recv()
            if command['kind'] == 'close':
                synchronize()
                break
            if command['kind'] == 'prepare':
                for batch in command['batches']:
                    solve(batch)
                synchronize()
                connection.send(dict(kind='ready', pid=os.getpid()))
            elif command['kind'] == 'barrier':
                synchronize()
                connection.send(dict(kind='ready', pid=os.getpid()))
            elif command['kind'] == 'run':
                path, batches = Path(command['path']), command['batches']
                expected = [index for batch in batches for index in batch]
                payload = dict(metadata=command['metadata'], worker=config['worker'], batches=batches, rows=[])
                if path.exists():
                    previous = json.loads(path.read_text())
                    if any(previous.get(k) != payload[k] for k in ('metadata', 'worker', 'batches')):
                        raise ValueError('Resumed worker shard belongs to another execution/group')
                    payload['rows'] = previous['rows']
                completed = checked_rows(config['problem'], instances, payload['rows'], expected)
                resumed = len(completed)
                for batch in batches:
                    if all(index in completed for index in batch):
                        continue
                    # Replay the original full batch on a partially completed checkpoint.
                    for row in solve(batch):
                        if row['index'] not in completed:
                            completed[row['index']] = row
                    payload['rows'] = [completed[index] for index in sorted(completed)]
                    _write_json(path, payload)
                _write_json(path, payload)
                synchronize()
                connection.send(dict(kind='complete', pid=os.getpid(), resumed=resumed,
                                     solved=len(completed) - resumed, path=str(path)))
            else:
                raise ValueError('Unknown execution command')
    except EOFError:
        pass
    except BaseException as error:
        synchronized = True
        synchronization_error = None
        try:
            if config['device'] != 'cpu' and torch.cuda.is_initialized():
                torch.cuda.synchronize()
        except BaseException as sync_error:
            synchronized, synchronization_error = False, repr(sync_error)
        failure = dict(kind='error', pid=os.getpid(), error=repr(error), traceback=traceback.format_exc(),
                       cuda_synchronized=synchronized, synchronization_error=synchronization_error)
        try:
            _write_json(Path(config['output_dir']) / f'worker_{config["worker"]}_failure.json', failure)
            connection.send(failure)
        except (OSError, EOFError, BrokenPipeError):
            pass
        if config.get('mps_env') and not synchronized:
            # Do not exit a live MPS CUDA context after a failed synchronization.
            # The parent first obtains terminate_client=CUDA_SUCCESS, then signals
            # this owned process. A queued ordinary close is not sufficient.
            while True:
                try:
                    connection.recv()
                except EOFError:
                    time.sleep(1)
        raise
    finally:
        connection.close()


class ExecutionSession:
    def __init__(self, problem, method, execution_plan, output_dir, metadata, *,
                 instances=None, input_path=None, expected_profile=None,
                 timeout_seconds=3600, _test_backend_factory=None):
        self.problem, self.method = problem, method
        self.plan = validate_execution_plan(execution_plan)
        self.execution = execution_profile(problem, method, self.plan)
        self.output_dir = Path(output_dir)
        self.metadata = json.loads(json.dumps(metadata, allow_nan=False))
        self.expected_profile = copy.deepcopy(expected_profile)
        self.instances, self.input_path = instances, input_path
        if (instances is None) == (input_path is None):
            raise ValueError('Provide exactly one of instances or input_path')
        self.timeout = timeout_seconds
        self.factory = _test_backend_factory
        self.processes, self.connections, self.mps = [], [], None
        self.profile, self.prepared, self.used = None, False, False
        self.startup_seconds, self.prepare_seconds, self.mps_proof = 0., 0., None

    @property
    def worker_pids(self):
        return [process.pid for process in self.processes]

    def _receive(self, kind):
        pending, responses = set(range(len(self.connections))), {}
        deadline = time.monotonic() + self.timeout
        while pending:
            ready = wait([self.connections[i] for i in pending], timeout=.1)
            for index in list(pending):
                if self.connections[index] in ready:
                    try:
                        response = self.connections[index].recv()
                    except EOFError:
                        raise RuntimeError(f'Execution worker {index} exited before {kind}') from None
                    if response.get('kind') != kind:
                        raise RuntimeError(f'Execution worker {index} failed: {response}')
                    responses[index] = response
                    pending.remove(index)
                elif not self.processes[index].is_alive():
                    raise RuntimeError(f'Execution worker {index} exited before {kind}')
            if time.monotonic() >= deadline:
                raise TimeoutError(f'Execution workers did not reach {kind} barrier')
        return [responses[i] for i in range(len(self.connections))]

    def _send(self, commands, kind):
        for connection, command in zip(self.connections, commands):
            connection.send(command)
        return self._receive(kind)

    def __enter__(self):
        if self.used:
            raise RuntimeError('ExecutionSession cannot be reopened')
        self.used = True
        before = time.perf_counter()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        try:
            if self.factory is None:
                gpu_guard()
            if self.input_path is not None:
                with Path(self.input_path).open('rb') as stream:
                    self.instances = pickle.load(stream)
            if self.execution['mode'] == 'mps4':
                if self.factory is not None:
                    raise ValueError('CPU fixtures cannot start MPS')
                self.mps = PrivateMPS().__enter__()
            context = mp.get_context('spawn')
            for worker in range(self.execution['workers']):
                parent, child = context.Pipe()
                config = dict(problem=self.problem, method=self.method, worker=worker,
                    output_dir=str(self.output_dir), input_path=str(self.input_path) if self.input_path else None,
                    device='cpu' if self.factory is not None else 'cuda',
                    expected_profile=self.expected_profile,
                    mps_env={key: self.mps.env[key] for key in ('CUDA_MPS_PIPE_DIRECTORY', 'CUDA_MPS_LOG_DIRECTORY')}
                        if self.mps else None)
                process = context.Process(target=_worker, args=(child, config,
                    self.instances if self.input_path is None else None, self.factory))
                # Spawn imports this module before _worker is called. Set inherited
                # native-library limits here, not after numpy/torch initialization.
                previous = {key: os.environ.get(key) for key in WORKER_THREAD_ENV}
                os.environ.update(WORKER_THREAD_ENV)
                try:
                    process.start()
                finally:
                    for key, value in previous.items():
                        if value is None:
                            os.environ.pop(key, None)
                        else:
                            os.environ[key] = value
                child.close()
                self.processes.append(process)
                self.connections.append(parent)
            loaded = self._receive('loaded')
            self.profile = loaded[0]['profile']
            if any(row['profile'] != self.profile for row in loaded):
                raise ValueError('Workers loaded different deployment profiles')
            self.startup_seconds = time.perf_counter() - before
            return self
        except BaseException:
            self.close()
            raise

    def _groups(self, groups):
        output, seen = [], set()
        for position, value in enumerate(groups):
            group = value if isinstance(value, dict) else dict(id=str(position), indices=value)
            group_id, indices = _token(group['id']), list(group['indices'])
            if group_id in seen or not indices or len(set(indices)) != len(indices):
                raise ValueError('Duplicate group ID/index or empty group')
            if any(type(i) is not int or not 0 <= i < len(self.instances) for i in indices):
                raise ValueError('Group index outside input split')
            if len({fields(self.problem, self.instances[i])['n'] for i in indices}) != 1:
                raise ValueError('Execution groups must be true-size homogeneous')
            seen.add(group_id)
            width = self.execution['inference_batch_size']
            output.append(dict(id=group_id, indices=indices,
                batches=[indices[i:i + width] for i in range(0, len(indices), width)]))
        return output

    def prepare(self, warmup_groups):
        if not self.connections:
            raise RuntimeError('Session must be entered before prepare')
        groups = self._groups(warmup_groups)
        if not groups:
            raise ValueError('Explicit warmup groups required')
        before = time.perf_counter()
        batches = [group['batches'][0] for group in groups]
        self._send([dict(kind='prepare', batches=batches) for _ in self.connections], 'ready')
        if self.mps:
            self.mps_proof = self.mps.proof(self.worker_pids)
        self.prepare_seconds += time.perf_counter() - before
        self.prepared = True
        return dict(startup_seconds=self.startup_seconds, prepare_seconds=self.prepare_seconds,
            worker_pids=self.worker_pids, mps_proof=self.mps_proof,
            deployment_profile=self.profile, execution_profile=self.execution)

    def run_groups(self, groups, pass_id):
        if not self.prepared or not self.connections:
            raise RuntimeError('All workers must be warmed and ready before timing/production')
        pass_id = _token(pass_id)
        groups = self._groups(groups)
        if not groups:
            raise ValueError('At least one execution group required')
        seen = [index for group in groups for index in group['indices']]
        if len(set(seen)) != len(seen):
            raise ValueError('Duplicate instance across groups within a pass')
        reports, all_rows, global_start = [], [], None
        try:
            for group in groups:
                self._send([dict(kind='barrier') for _ in self.connections], 'ready')
                if self.mps:
                    self.mps_proof = self.mps.proof(self.worker_pids)
                before = time.perf_counter()
                if global_start is None:
                    global_start = before
                directory = self.output_dir / f'pass_{pass_id}' / f'group_{group["id"]}'
                metadata = dict(source=self.metadata, execution_plan=self.plan,
                    execution_profile=self.execution, deployment_profile_sha256=canonical_hash(self.profile),
                    pass_id=pass_id, group_id=group['id'], indices=group['indices'])
                paths = [directory / f'worker_{worker}.json' for worker in range(len(self.connections))]
                commands = [dict(kind='run', path=str(path), metadata=metadata,
                    batches=group['batches'][worker::len(self.connections)]) for worker, path in enumerate(paths)]
                responses = self._send(commands, 'complete')
                rows = merge_shards(self.problem, self.instances, paths, group['indices'], metadata,
                                    directory / 'merged.json')
                seconds = time.perf_counter() - before
                reports.append(dict(id=group['id'], indices=group['indices'], seconds=seconds,
                    resumed=sum(row['resumed'] for row in responses), solved=sum(row['solved'] for row in responses),
                    worker_paths=list(map(str, paths))))
                all_rows.extend(rows)
            result = dict(rows=all_rows, groups=reports, completion_seconds=time.perf_counter() - global_start,
                startup_seconds=self.startup_seconds, prepare_seconds=self.prepare_seconds,
                worker_pids=self.worker_pids, deployment_profile=self.profile,
                execution_profile=self.execution, mps_proof=self.mps_proof,
                timing='Ready barrier before each group; CUDA sync, independent checks, worker/merged writes '
                       'included. Global interval also includes inter-group barriers; startup/warmup separate.')
            _write_json(self.output_dir / f'pass_{pass_id}_summary.json', result)
            return result
        except BaseException:
            self.close()
            raise

    def close(self):
        for connection in self.connections:
            try:
                connection.send(dict(kind='close'))
            except (OSError, EOFError, BrokenPipeError):
                pass
        for process in self.processes:
            process.join(timeout=5)
        alive = [process for process in self.processes if process.is_alive()]
        safe_to_signal = {process.pid for process in alive} if not self.mps else set()
        if self.mps:
            if self.mps.directory is not None:
                shutil.copytree(self.mps.directory / 'log', self.output_dir / 'mps_logs', dirs_exist_ok=True)
            if alive:
                try:
                    terminated = self.mps.terminate_owned_clients(self.worker_pids)
                    safe_to_signal.update(row['client_pid'] for row in terminated)
                    _write_json(self.output_dir / 'mps_client_termination.json', dict(terminated=terminated))
                except BaseException as error:
                    _write_json(self.output_dir / 'mps_cleanup_blocked.json',
                        dict(error=repr(error), retained_worker_pids=[process.pid for process in alive],
                             private_pipe_directory=self.mps.env['CUDA_MPS_PIPE_DIRECTORY'],
                             no_unsynchronized_clients_signalled=True))
                    raise
        for process in alive:
            if process.pid not in safe_to_signal:
                process.join(timeout=5)
                if process.is_alive():
                    _write_json(self.output_dir / 'mps_cleanup_blocked.json',
                        dict(error='Alive worker has no confirmed terminated MPS context',
                             retained_worker_pids=[process.pid], no_unsynchronized_clients_signalled=True))
                    raise RuntimeError('MPS cleanup blocked: cannot safely signal an unconfirmed client')
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
            if process.is_alive():
                process.kill()
                process.join(timeout=5)
        for connection in self.connections:
            connection.close()
        self.connections = []
        self.prepared = False
        if self.mps:
            try:
                if self.mps.directory is not None:
                    shutil.copytree(self.mps.directory / 'log', self.output_dir / 'mps_logs', dirs_exist_ok=True)
            finally:
                self.mps.close()
                self.mps = None

    def __exit__(self, *unused):
        self.close()


def run_deployment(problem, method, execution_plan, output_dir, metadata, groups, *,
                   instances=None, input_path=None, expected_profile=None, warmup_groups=None,
                   pass_id='production'):
    with ExecutionSession(problem, method, execution_plan, output_dir, metadata,
            instances=instances, input_path=input_path, expected_profile=expected_profile) as session:
        if not groups:
            raise ValueError('At least one execution group required')
        session.prepare(warmup_groups or ([groups[0]] if len(groups) == 1 else [groups[0], groups[-1]]))
        return session.run_groups(groups, pass_id)
