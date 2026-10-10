"""Bounded TRAIN-only singleton pilot, isolated from the frozen R58 pipeline.

Only trailing consecutive depot-zero padding is ignored in VRP tour comparison.
Paid FP64 route costs must be exactly equal; checker tolerances do not establish
equivalence. Diffusion candidate timings are not an algorithm-equivalence claim.
"""

import argparse
import contextlib
import copy
import json
import os
import pickle
import random
import signal
import socket
import statistics
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE / 'runs/R58_unified_scenario'
OUTPUT = ROOT / 'throughput_pilot'
GPU = 'GPU-cb626bdb-4319-2f5e-b91f-17fcaa44a83d'
OVERLAY = '/public/home/shiys/easynco_v3_bridge/exports/train_100k_20261002/python_compat'
CASES = [('VRPBLTW', 'RouteFinder'), ('VRPBLTW', 'MoSES_CaDA'),
         ('CVRP', 'BQ'), ('CVRP', 'LEHD')]
DIFFUSION = ['DIFUSCO', 'DIFUSCO500', 'T2T', 'T2T500']


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def environment():
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=GPU, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
               OPENBLAS_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1',
               PYTHONDONTWRITEBYTECODE='1', WANDB_MODE='offline',
               WANDB_DIR=str(OUTPUT), MPLCONFIGDIR=str(OUTPUT / 'mplconfig'),
               PYTHONPATH=OVERLAY + ':' + str(HERE.parents[1]))
    return env


def load_train(problem):
    from .r58_scenario import INPUT_ROOT
    path = INPUT_ROOT / (problem + 'train') / 'dataset.pkl'
    with path.open('rb') as stream:
        return pickle.load(stream), path


def stratified_indices(sizes, per_scale):
    """Preselect from physical sizes alone, reserving one disjoint warmup/scale."""
    ordered = sorted(sizes)
    targets = sorted(set([ordered[0], ordered[len(ordered) // 2], ordered[-1]]))
    rng = random.Random(2)
    selected, warmups = [], []
    for n in targets:
        group = [i for i, value in enumerate(sizes) if value == n]
        if len(group) < per_scale + 1:
            raise ValueError(f'True size {n}: need {per_scale + 1} distinct TRAIN instances')
        chosen = rng.sample(group, per_scale + 1)
        selected.extend(chosen[:-1])
        warmups.append(chosen[-1])
    return dict(scales=targets, indices=sorted(selected), warmup_indices=warmups,
                true_sizes={str(i): int(sizes[i]) for i in selected + warmups})


def prepare():
    from .r58_scenario import CONTRACT, canonical_hash, fields, file_hash
    OUTPUT.mkdir(parents=True, exist_ok=True)
    manifest = dict(seed=2, split='train', inference_batch_size=1,
                    cases=[list(case) for case in CASES], diffusion=DIFFUSION, datasets={},
                    contract_sha256=canonical_hash(CONTRACT),
                    frozen_source_sha256={name: file_hash(HERE / name) for name in
                        ['r58_scenario.py', 'r58_environments.py', 'r58_backends.py', 'r58_labels.py']},
                    pilot_source_sha256=file_hash(__file__),
                    cpu_threads_per_worker=1,
                    selection='seed2 uniform without replacement within exact min/upper-median/max size',
                    padding='Ignore only consecutive trailing depot zeros, retaining the final depot',
                    quality_claim='none for changed diffusion budget; concurrency requires exact FP64 paid cost',
                    excluded='No val/test inputs, archive costs, labels, full generation or training',
                    hardware=dict(host='gpu03', uuid=GPU, slurm_job_id='1465'))
    for problem in ['VRPBLTW', 'CVRP', 'TSP']:
        items, path = load_train(problem)
        sizes = [fields(problem, item)['n'] for item in items]
        manifest['datasets'][problem] = dict(path=str(path), input_sha256=file_hash(path),
            count=len(items), **stratified_indices(sizes, 4 if problem == 'TSP' else 8))
    path = OUTPUT / 'selection.json'
    if path.exists() and read_json(path) != manifest:
        previous, current = read_json(path), copy.deepcopy(manifest)
        previous.pop('pilot_source_sha256', None)
        current.pop('pilot_source_sha256', None)
        if previous != current or (OUTPUT / 'launch.json').exists():
            raise ValueError('Existing selection/source differs; refusing to overwrite pilot provenance')
    write_json(path, manifest)
    print(json.dumps(manifest['datasets'], indent=2), flush=True)
    return manifest


def preflight_gate(remote=False):
    state = read_json(ROOT / 'pipeline_state.json')
    if state.get('status') != 'preflight_complete' or not state.get('finished'):
        evidence_path = OUTPUT / 'gate_evidence.json'
        evidence = read_json(evidence_path) if evidence_path.exists() else {}
        coverage = state.get('budget_coverage', {})
        if not (state.get('stage') == 'labels' and state.get('status') == 'skipped_over_budget'
                and state.get('finished') and coverage.get('current_passing_deployments') == 128
                and coverage.get('expected_deployments') == 128 and coverage.get('all_current_passing')
                and coverage.get('complete_budget') and evidence.get('preflight_status') == 'preflight_complete'
                and evidence.get('preflight_finished_unix') and evidence.get('preflight_pid_absent_on_gpu03')
                and evidence.get('preflight_tmux_handle_absent')):
            raise RuntimeError('No completed preflight evidence; no GPU launch allowed')
        from .r58_scenario import file_hash
        manifest = read_json(OUTPUT / 'selection.json')
        if any(file_hash(HERE / name) != digest for name, digest in manifest['frozen_source_sha256'].items()):
            raise RuntimeError('Frozen sources changed since preflight')
        state = dict(status='preflight_complete', pid=evidence['preflight_pid'],
                     finished=evidence['preflight_finished_unix'],
                     proof='Original pilot launch evidence plus current128/128 completed budget gate',
                     current_main_state=state)
    if remote:
        pid = state['pid']
        if Path(f'/proc/{pid}').exists():
            raise RuntimeError(f'Preflight process {pid} has not exited')
    else:
        result = subprocess.run(['tmux', 'has-session', '-t', '=r58_preflight'],
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if result.returncode == 0:
            raise RuntimeError('Original r58_preflight handle still exists')
    return state


def gpu_snapshot():
    result = subprocess.run(['nvidia-smi', '-i', GPU,
        '--query-gpu=uuid,memory.used,memory.total,utilization.gpu', '--format=csv,noheader,nounits'],
        capture_output=True, text=True, timeout=10, check=True)
    values = [v.strip() for v in result.stdout.strip().split(',')]
    if len(values) != 4 or values[0] != GPU:
        raise RuntimeError('Designated GPU UUID not available')
    apps = subprocess.run(['nvidia-smi', '-i', GPU,
        '--query-compute-apps=pid,used_memory', '--format=csv,noheader,nounits'],
        capture_output=True, text=True, timeout=10, check=True)
    return dict(time=time.time(), uuid=values[0], used_mib=int(values[1]),
                total_mib=int(values[2]), utilization=int(values[3]), contexts=apps.stdout.strip())


def gpu_guard():
    if socket.gethostname().split('.')[0] != 'gpu03':
        raise RuntimeError('Pilot may execute only on gpu03')
    if os.environ.get('SLURM_JOB_ID') != '1465' or os.environ.get('CUDA_VISIBLE_DEVICES') != GPU:
        raise RuntimeError('Require existing job1465 and the exact designated UUID')
    if os.environ.get('CONDA_DEFAULT_ENV') != 'easynco':
        raise RuntimeError('Require conda easynco')
    if os.environ.get('PYTHONPATH') != OVERLAY + ':' + str(HERE.parents[1]):
        raise RuntimeError('Require unchanged read-only run_v4_r58.sh compatibility overlay')
    return preflight_gate(remote=True)


def override_diffusion(backend):
    """Both copies matter: original test_step reads model.args, not parameters."""
    overrides = dict(inference_diffusion_steps=20, two_opt_iterations=100,
                     rewrite=backend.is_t2t, rewrite_steps=1 if backend.is_t2t else 0,
                     inference_steps=10, parallel_sampling=1, sequential_sampling=1)
    ratio = backend.parameters['rewrite_ratio']
    for key, value in overrides.items():
        backend.parameters[key] = value
        setattr(backend.model.args, key, value)
    actual = {key: getattr(backend.model.args, key) for key in overrides}
    if actual != overrides or backend.model.args.rewrite_ratio != ratio:
        raise ValueError('Diffusion override was not applied to actual model arguments')
    return dict(parameters=copy.deepcopy(backend.parameters), model_args=actual,
                rewrite_ratio=ratio, candidate_only=True, algorithm_equivalence=False)


@contextlib.contextmanager
def diffusion_witness(backend):
    witness = dict(denoising_calls=0, guided_calls=0, two_opt_limits=[])
    originals = {}
    for name, counter in [('categorical_denoise_step', 'denoising_calls'),
                          ('guided_categorical_denoise_step', 'guided_calls')]:
        if not hasattr(backend.model, name):
            continue
        original = getattr(backend.model, name)
        originals[name] = original
        def observe(*args, _original=original, _counter=counter, **kwargs):
            witness[_counter] += 1
            return _original(*args, **kwargs)
        setattr(backend.model, name, observe)
    refine = backend.module.batched_two_opt_torch
    def observe_refine(*args, **kwargs):
        witness['two_opt_limits'].append(kwargs.get('max_iterations'))
        return refine(*args, **kwargs)
    backend.module.batched_two_opt_torch = observe_refine
    try:
        yield witness
        expected = dict(denoising_calls=20, guided_calls=10 if backend.is_t2t else 0,
                        two_opt_limits=[100, 100] if backend.is_t2t else [100])
        if witness != expected:
            raise ValueError(f'Actual solver budget differs: {witness}; expected {expected}')
    finally:
        for name, original in originals.items():
            setattr(backend.model, name, original)
        backend.module.batched_two_opt_torch = refine


def normalized_tour(route):
    route = list(route)
    while len(route) > 1 and route[-1] == route[-2] == 0:
        route.pop()
    return route


def merge_outputs(expected, paths):
    expected = list(expected)
    rows, errors, invalid = {}, [], set()
    if len(set(expected)) != len(expected):
        raise ValueError('Duplicate expected original indices')
    for path in paths:
        if not Path(path).exists():
            errors.append(dict(file=str(path), error='missing_worker_file'))
            continue
        with Path(path).open() as stream:
            for line_number, line in enumerate(stream, 1):
                try:
                    row = json.loads(line)
                    index = row['index']
                    if type(index) is not int or index not in expected:
                        raise ValueError('unexpected original index')
                    if index in rows:
                        invalid.add(index)
                        raise ValueError('duplicate original index')
                    rows[index] = row
                    if row.get('status') != 'ok':
                        errors.append(dict(index=index, error=row.get('error', 'solver_failure'),
                                           error_type=row.get('error_type')))
                except (ValueError, KeyError, TypeError) as error:
                    errors.append(dict(file=str(path), line=line_number, error=str(error)))
    missing = sorted(set(expected) - set(rows))
    successful = [rows[i] for i in sorted(rows) if rows[i].get('status') == 'ok' and i not in invalid]
    complete = not errors and not missing and len(successful) == len(expected)
    return dict(complete=complete, rows=[rows[i] for i in sorted(rows)],
                successful=len(successful), expected=len(expected), missing=missing, errors=errors)


def compare_equivalence(serial, concurrent):
    a = {row['index']: row for row in serial['rows'] if row.get('status') == 'ok'}
    b = {row['index']: row for row in concurrent['rows'] if row.get('status') == 'ok'}
    differences = []
    for index in sorted(a.keys() & b.keys()):
        left, right = a[index], b[index]
        delta = right['cost'] - left['cost']
        raw_equal = left['route'] == right['route']
        physical_equal = normalized_tour(left['route']) == normalized_tour(right['route'])
        differences.append(dict(index=index, cost_delta=delta, exact_paid_cost=delta == 0.,
                                same_raw_tour=raw_equal, same_physical_tour=physical_equal))
    complete = serial['complete'] and concurrent['complete'] and a.keys() == b.keys()
    return dict(complete=complete, compared=len(differences),
                exact_cost_differences=sum(not r['exact_paid_cost'] for r in differences),
                physical_tour_differences=sum(not r['same_physical_tour'] for r in differences),
                raw_tour_differences=sum(not r['same_raw_tour'] for r in differences),
                max_abs_paid_cost_delta=max([abs(r['cost_delta']) for r in differences], default=0.),
                equivalent=complete and all(r['exact_paid_cost'] and r['same_physical_tour'] for r in differences),
                details=differences)


def premature_worker_exit(states, returncodes, target):
    return any(code is not None and (code != 0 or target != 'complete' or state.get('status') != 'complete')
               for state, code in zip(states, returncodes))


def routefinder_aug1_type():
    import torch
    from .r58_backends import AdaptedRouteFinder, independent_audit
    from .r58_environments import routefinder_depot_clock, routefinder_mask
    from .r58_scenario import fields, file_hash

    class Aug1RouteFinder(AdaptedRouteFinder):
        """New timing protocol: identity orientation, original weights/all starts."""

        def profile(self):
            profile = super().profile()
            profile['budget'].update(augmentation=1, orientation='original, untransformed',
                selection='minimum over ALL N starts in original orientation; stable first tie')
            profile.update(pilot_protocol='RF_aug1_candidate_only', algorithm_equivalence_to_aug8=False,
                           pilot_adapter_sha256=file_hash(__file__))
            return profile

        @torch.inference_mode()
        def run(self, items):
            if len(items) != 1:
                raise ValueError('Aug1 pilot requires singleton inference')
            spec = fields(self.problem, items[0])
            scale, bucket = spec['n'], 50 if spec['n'] <= 75 else 100
            runner = self.original.pool.get(self.original.key, bucket)
            if bucket not in self.original.validated_buckets:
                checkpoint = torch.load(self.original.checkpoints[str(bucket)], map_location='cpu', weights_only=False)
                weights = {k[len('policy.'):]: v for k, v in checkpoint['state_dict'].items() if k.startswith('policy.')}
                if not weights:
                    raise ValueError('Missing original policy weights')
                from train_100k_alias_weights import load_alias_complete_state
                load_alias_complete_state(runner.policy, weights)
                runner.policy.eval().requires_grad_(False)
                runner.env.get_action_mask = routefinder_mask
                original_step = runner.env._step
                runner.env._step = lambda td: routefinder_depot_clock(original_step(td))
                self.original.validated_buckets.add(bucket)
            torch.set_float32_matmul_precision('highest')
            batch = routefinder_depot_clock(runner.env.reset(self.tensor_dict(items)).to(self.device))
            expected = torch.as_tensor(spec['xy'], dtype=torch.float32, device=self.device)[None]
            if not torch.equal(batch['locs'], expected):
                raise ValueError('Aug1 policy coordinates differ from original orientation')
            if torch.is_autocast_enabled() or any(p.dtype != torch.float32 for p in runner.policy.parameters()):
                raise ValueError('Aug1 pilot requires unchanged FP32 weights and inference')
            # No StateAugmentation call: the reset singleton is passed unchanged.
            output = runner.policy(batch, runner.env, phase='test', num_starts=scale,
                                   decode_type='multistart_greedy', return_actions=True)
            rewards = output['reward'].reshape(1, scale)
            actions = output['actions'].reshape(1, scale, -1)
            best = rewards.argmax(-1)
            tours = actions[torch.arange(1, device=self.device), best].cpu().tolist()
            tours = [[0, *tour] if not tour or tour[0] != 0 else tour for tour in tours]
            costs = (-rewards[torch.arange(1, device=self.device), best]).cpu().tolist()
            audit = independent_audit(self.problem, items, tours, costs)
            self.last_pilot_witness = dict(augmentation=1, original_orientation=True,
                policy_input_shape=list(batch['locs'].shape), starts=scale, checkpoint_bucket=bucket,
                precision='FP32', inference_batch_size=1, decoding='multistart_greedy')
            return self.base.InferenceResult(audit['costs_fp64'], tours, audit)

    return Aug1RouteFinder


def prepare_aug1(methods, supplement=False):
    from .r58_scenario import file_hash
    original = read_json(OUTPUT / 'selection.json')
    selection = dict(protocol='RF_aug1_new_candidate_not_aug8_equivalent', seed=2, repeats=2,
        workers=[1, 2, 4], instances_per_problem=12, methods=methods, problems=['VRPBLTW', 'CVRP'],
        frozen_source_sha256=original['frozen_source_sha256'], pilot_source_sha256=file_hash(__file__),
        selection_rule='First4 original preselected TRAIN indices at each scale, selected without costs', datasets={})
    for problem in selection['problems']:
        data = copy.deepcopy(original['datasets'][problem])
        data['indices'] = sorted(i for scale in data['scales'] for i in
            [i for i in data['indices'] if data['true_sizes'][str(i)] == scale][:4])
        selection['datasets'][problem] = data
    if supplement:
        parent_path = OUTPUT / 'aug1_selection.json'
        parent = read_json(parent_path)
        if methods != ['MoSES_RF'] or selection['datasets'] != parent['datasets']:
            raise RuntimeError('Supplement must measure only MoSES_RF on identical preselected TRAIN items')
        if read_json(OUTPUT / 'aug1_state.json')['status'] != 'complete':
            raise RuntimeError('Finish the original aug1 pilot before supplement preparation')
        selection.update(parent_manifest=str(parent_path), parent_manifest_sha256=file_hash(parent_path))
    path = OUTPUT / ('aug1_moses_rf_selection.json' if supplement else 'aug1_selection.json')
    if path.exists() and read_json(path) != selection:
        raise RuntimeError('Existing aug1 preselection differs; do not overwrite')
    write_json(path, selection)
    print(json.dumps(selection, indent=2), flush=True)


def worker(task_path, worker_id):
    gpu_guard()
    import torch
    from .r58_backends import make_backend
    from .r58_labels import seed_solver
    from .r58_scenario import check_route, fields
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    if torch.cuda.device_count() != 1:
        raise RuntimeError('Exactly one visible GPU required')
    task = read_json(task_path)
    directory = Path(task['directory'])
    prefix = directory / f'worker_{worker_id}'
    summary = dict(worker=worker_id, pid=os.getpid(), status='initializing', started=time.time())
    write_json(prefix.with_suffix('.state.json'), summary)
    try:
        items, _ = load_train(task['problem'])
        seed_solver(2)
        before = time.perf_counter()
        backend = (routefinder_aug1_type()(task['problem'], task['method'], 'cuda')
                   if task.get('rf_aug1') else make_backend(task['problem'], task['method']))
        if task['candidate']:
            actual = override_diffusion(backend)
        else:
            actual = None
        profile = copy.deepcopy(backend.profile())
        if actual:
            profile.update(pilot_actual_arguments=actual,
                inherited_budget_warning='Inherited budget text describes frozen50-step recipe, not this candidate')
            profile['budget'].update(denoising_steps=20, rewrite_steps=1 if backend.is_t2t else 0)
        write_json(prefix.with_suffix('.deployment.json'), profile)
        summary.update(initialization_seconds=time.perf_counter() - before,
                       gpu_name=torch.cuda.get_device_name(0), warmups=[])

        def solve(index):
            seed_solver(2)
            torch.cuda.synchronize()
            start = time.perf_counter()
            with diffusion_witness(backend) if task['candidate'] else contextlib.nullcontext(None) as witnessed:
                output = backend.run([items[index]])
                torch.cuda.synchronize()
            seconds = time.perf_counter() - start
            if len(output.tours) != 1 or len(output.costs) != 1:
                raise ValueError('Singleton output count changed')
            checked = check_route(task['problem'], items[index], output.tours[0], output.costs[0])
            if not checked['feasible'] or not checked['cost_matches']:
                raise ValueError(f'Independent physical checker failed: {checked}')
            route = [int(node) for node in output.tours[0]]
            return dict(index=index, true_size=fields(task['problem'], items[index])['n'],
                        status='ok', route=route, cost=checked['cost'], reported_cost=float(output.costs[0]),
                        independent=checked, synchronized_solve_seconds=seconds, budget_witness=witnessed,
                        rf_budget_witness=getattr(backend, 'last_pilot_witness', None),
                        worker=worker_id, seed=2, inference_batch_size=1)

        for index in task['warmup_indices']:
            summary['warmups'].append(solve(index))
        torch.cuda.synchronize()
        summary.update(cold_startup_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                       cold_startup_peak_reserved_bytes=torch.cuda.max_memory_reserved())
        torch.cuda.reset_peak_memory_stats()
        summary.update(status='ready', ready=time.time())
        write_json(prefix.with_suffix('.state.json'), summary)
        while not (directory / 'go.json').exists():
            time.sleep(.01)
        start = time.perf_counter()
        with prefix.with_suffix('.jsonl').open('x', buffering=1) as stream:
            for index in task['assignments'][worker_id]:
                try:
                    row = solve(index)
                except Exception as error:
                    row = dict(index=index, worker=worker_id, status='error', error_type=type(error).__name__,
                               error=str(error), traceback=traceback.format_exc())
                    stream.write(json.dumps(row, allow_nan=False) + '\n')
                    raise
                stream.write(json.dumps(row, allow_nan=False) + '\n')
            stream.flush()
            os.fsync(stream.fileno())
        torch.cuda.synchronize()
        summary.update(status='complete', steady_worker_seconds=time.perf_counter() - start,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(), finished=time.time())
        write_json(prefix.with_suffix('.state.json'), summary)
    except Exception as error:
        summary.update(status='failed', error_type=type(error).__name__, error=str(error),
                       traceback=traceback.format_exc(), finished=time.time())
        write_json(prefix.with_suffix('.state.json'), summary)
        raise


def stop_children(processes):
    for process in processes:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
    for process in processes:
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()


def run_phase(problem, method, width, selection, deadline, candidate=False, suffix='', rf_aug1=False, repeat=None):
    directory = OUTPUT / (('aug1__' if rf_aug1 else '') +
        (f'diffusion20__{method}' if candidate else f'{problem}__{method}__w{width}') + suffix)
    directory.mkdir(exist_ok=False)
    task = dict(problem=problem, method=method, width=width, directory=str(directory),
                candidate=candidate, rf_aug1=rf_aug1, repeat=repeat, warmup_indices=selection['warmup_indices'],
                assignments=[selection['indices'][i::width] for i in range(width)])
    write_json(directory / 'task.json', task)
    phase = dict(problem=problem, method=method, workers=width, candidate=candidate, rf_aug1=rf_aug1, repeat=repeat,
                 status='running', started=time.time(), directory=str(directory))
    processes, logs, snapshots, sampling_errors = [], [], [], []
    stop = threading.Event()
    def sample():
        while not stop.is_set():
            try:
                snapshots.append(gpu_snapshot())
            except Exception as error:
                sampling_errors.append(str(error))
            stop.wait(.5)
    monitor = threading.Thread(target=sample, daemon=True)
    monitor.start()
    cold_start = time.perf_counter()
    phase_deadline = min(deadline, time.monotonic() + 1200)
    def states():
        return [read_json(directory / f'worker_{i}.state.json')
                if (directory / f'worker_{i}.state.json').exists() else {} for i in range(width)]
    def barrier(target):
        while True:
            current = states()
            if all(state.get('status') == target for state in current):
                return current
            if any(state.get('status') == 'failed' for state in current):
                raise RuntimeError('Worker failure; preserve partial outputs and stop this phase')
            if premature_worker_exit(current, [p.poll() for p in processes], target):
                raise RuntimeError('Worker exited before completion barrier')
            if time.monotonic() >= phase_deadline:
                raise TimeoutError('Bounded phase/global wall-time budget exhausted')
            time.sleep(.01)
    try:
        for i in range(width):
            log = (directory / f'worker_{i}.log').open('x')
            logs.append(log)
            command = [sys.executable, '-u', '-m', 'code.V4.r58_throughput', '--stage', 'worker',
                       '--task', str(directory / 'task.json'), '--worker-id', str(i)]
            processes.append(subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                               env=environment(), start_new_session=True))
        barrier('ready')
        phase['cold_ready_seconds'] = time.perf_counter() - cold_start
        warm_start = time.perf_counter()
        write_json(directory / 'go.json', dict(time=time.time()))
        barrier('complete')
        phase['warm_completion_barrier_seconds'] = time.perf_counter() - warm_start
        for process in processes:
            process.wait(timeout=max(1., phase_deadline - time.monotonic()))
        if any(p.returncode != 0 for p in processes):
            raise RuntimeError('Worker exited nonzero after completion barrier')
        phase['status'] = 'complete'
    except Exception as error:
        phase.update(status='failed', error_type=type(error).__name__, error=str(error))
    finally:
        stop_children(processes)
        phase.update(cold_wall_seconds=time.perf_counter() - cold_start,
                     worker_states=states(), returncodes=[p.returncode for p in processes])
        stop.set()
        monitor.join(timeout=25)
        for log in logs:
            log.close()
    merged = merge_outputs(selection['indices'], [directory / f'worker_{i}.jsonl' for i in range(width)])
    # Parent rechecks the same original physical TRAIN objects, independently of worker status.
    from .r58_scenario import check_route
    items, _ = load_train(problem)
    for row in merged['rows']:
        if row.get('status') == 'ok':
            checked = check_route(problem, items[row['index']], row['route'], row['reported_cost'])
            if not checked['feasible'] or not checked['cost_matches'] or checked['cost'] != row['cost']:
                merged['complete'] = False
                merged['errors'].append(dict(index=row['index'], error='parent_physical_check_failed'))
    if not merged['complete']:
        phase['status'] = 'failed'
    phase.update(merged=merged, sampled_global_peak_mib=max([s['used_mib'] for s in snapshots], default=None),
                 memory_note='UUID-only whole-device samples at~0.5s; includes contexts, not an exact hardware peak',
                 sampling_errors=sampling_errors)
    if phase['status'] == 'complete':
        phase.update(warm_instances_per_second=len(selection['indices']) / phase['warm_completion_barrier_seconds'],
                     cold_instances_per_second=len(selection['indices']) / phase['cold_wall_seconds'])
    write_json(directory / 'gpu_samples.json', snapshots)
    write_json(directory / 'result.json', phase)
    print(json.dumps({key: phase.get(key) for key in ['problem', 'method', 'workers', 'status',
        'warm_completion_barrier_seconds', 'cold_wall_seconds', 'sampled_global_peak_mib']}), flush=True)
    return phase


def summarize(phases, manifest):
    results = dict(concurrency=[], diffusion=[],
        timing='Actual same fixed workload, global start and completion barriers. Warmups excluded only from warm time; '
               'cold includes subprocess startup/imports/checkpoints, all worker warmups, solving and teardown.',
        caveats=['Small TRAIN-only pilot, no release, budget lock, test, training or quality claim.',
                 'Speedups are only for fully completed identical workloads; no multiplied singleton estimates.',
                 'Both paid FP64 costs and physical tour sequence must match to recommend concurrency.'])
    for problem, method in CASES:
        group = [p for p in phases if (p['problem'], p['method']) == (problem, method) and not p['candidate']]
        serial = next((p for p in group if p['workers'] == 1), None)
        for phase in group:
            row = {key: phase.get(key) for key in ['problem', 'method', 'workers', 'status', 'directory',
                'warm_completion_barrier_seconds', 'cold_wall_seconds', 'warm_instances_per_second',
                'sampled_global_peak_mib']}
            if serial:
                row['equivalence'] = compare_equivalence(serial['merged'], phase['merged'])
                if serial['status'] == phase['status'] == 'complete':
                    row.update(warm_speedup=serial['warm_completion_barrier_seconds'] / phase['warm_completion_barrier_seconds'],
                               cold_speedup=serial['cold_wall_seconds'] / phase['cold_wall_seconds'])
            results['concurrency'].append(row)
        eligible = [r for r in results['concurrency'] if (r['problem'], r['method']) == (problem, method)
                    and r.get('equivalence', {}).get('equivalent') and 'warm_speedup' in r]
        if eligible:
            best = max(eligible, key=lambda r: r['warm_speedup'])
            results.setdefault('measured_recommendations', []).append(dict(problem=problem, method=method,
                workers=best['workers'], warm_speedup=best['warm_speedup'],
                note='Measured valid width only; failed4 never receives an inferred speedup'))
    for phase in [p for p in phases if p['candidate']]:
        row = {key: phase.get(key) for key in ['method', 'status', 'directory',
            'warm_completion_barrier_seconds', 'cold_wall_seconds', 'sampled_global_peak_mib']}
        baseline_path = ROOT / 'preflight' / f'TSP__{phase["method"]}.json'
        baseline = read_json(baseline_path)
        reference = {v['true_size']: v['seconds'] for v in baseline['results']}
        row['per_scale'] = []
        for n in manifest['datasets']['TSP']['scales']:
            values = [v['synchronized_solve_seconds'] for v in phase['merged']['rows']
                      if v.get('status') == 'ok' and v['true_size'] == n]
            row['per_scale'].append(dict(true_size=n, count=len(values),
                mean_seconds=statistics.mean(values) if values else None,
                median_seconds=statistics.median(values) if values else None,
                frozen50_preflight_reference_seconds=reference.get(n)))
        row.update(candidate_only=True, quality_equivalence=False,
                   reference_note='TRAIN preflight timings only; different indices/cold conditions, not a speedup claim')
        results['diffusion'].append(row)
    return results


def run(include_diffusion, wall_seconds):
    os.environ.update(environment())
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f'Pilot received signal{signum}; clean up only owned workers')
    signal.signal(signal.SIGTERM, interrupted)
    state = gpu_guard()
    before = gpu_snapshot()
    if before['used_mib'] > 100 or before['utilization'] > 5 or before['contexts']:
        raise RuntimeError(f'Designated GPU is not idle; no other process will be touched: {before}')
    from .r58_scenario import file_hash
    manifest = read_json(OUTPUT / 'selection.json')
    if file_hash(__file__) != manifest['pilot_source_sha256']:
        raise RuntimeError('Pilot source changed after CPU preparation')
    for name, digest in manifest['frozen_source_sha256'].items():
        if file_hash(HERE / name) != digest:
            raise RuntimeError('Frozen source differs from CPU preparation')
    for problem, selected in manifest['datasets'].items():
        if file_hash(selected['path']) != selected['input_sha256']:
            raise RuntimeError(f'Original TRAIN changed: {problem}')
    start = time.monotonic()
    deadline = start + min(wall_seconds, 10800)
    write_json(OUTPUT / 'run_state.json', dict(status='running', pid=os.getpid(), host=socket.gethostname(),
        started=time.time(), preflight=state, gpu_before=before, max_single_gpu_wall_seconds=min(wall_seconds, 10800)))
    phases = []
    try:
        for problem, method in CASES:
            for width in [1, 2, 4]:
                if time.monotonic() >= deadline:
                    raise TimeoutError('Global single-GPU wall budget exhausted')
                phases.append(run_phase(problem, method, width, manifest['datasets'][problem], deadline))
                write_json(OUTPUT / 'summary.json', summarize(phases, manifest))
        if include_diffusion:
            for method in DIFFUSION:
                if time.monotonic() >= deadline:
                    raise TimeoutError('Global single-GPU wall budget exhausted')
                phases.append(run_phase('TSP', method, 1, manifest['datasets']['TSP'], deadline, candidate=True))
                write_json(OUTPUT / 'summary.json', summarize(phases, manifest))
        changed = [name for name, digest in manifest['frozen_source_sha256'].items()
                   if file_hash(HERE / name) != digest]
        write_json(OUTPUT / 'run_state.json', dict(status='complete' if not changed else 'source_changed',
            finished=time.time(), wall_seconds=time.monotonic() - start, frozen_sources_changed=changed,
            failed_phases=[p['directory'] for p in phases if p['status'] != 'complete'], gpu_after=gpu_snapshot()))
        try:
            import wandb
            with wandb.init(project='selector', entity='yjkds-southern-university-of-science-technology',
                            name='R58_train_only_throughput_pilot', mode='offline', dir=str(OUTPUT),
                            config=dict(seed=2, train_only=True, candidate_only=True, gpu_uuid=GPU)) as tracking:
                for phase in phases:
                    if phase['status'] == 'complete':
                        tracking.log({phase['directory'].split('/')[-1] + '/warm_seconds':
                                      phase['warm_completion_barrier_seconds']})
        except Exception as error:
            write_json(OUTPUT / 'wandb_error.json', dict(error=str(error)))
    except BaseException as error:
        write_json(OUTPUT / 'run_state.json', dict(status='failed', finished=time.time(),
            wall_seconds=time.monotonic() - start, error=str(error), traceback=traceback.format_exc()))
        raise


def retry_diffusion(wall_seconds):
    os.environ.update(environment())
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f'Diffusion retry received signal{signum}')
    signal.signal(signal.SIGTERM, interrupted)
    proof = gpu_guard()
    before = gpu_snapshot()
    if before['used_mib'] > 100 or before['utilization'] > 5 or before['contexts']:
        raise RuntimeError(f'Designated GPU not idle: {before}')
    from .r58_scenario import file_hash
    manifest = read_json(OUTPUT / 'selection.json')
    original = read_json(OUTPUT / 'attempt1/run_state.json')
    remaining = 10800 - original['wall_seconds']
    if remaining <= 0:
        raise RuntimeError('Original3h GPU wall cap exhausted')
    for name, digest in manifest['frozen_source_sha256'].items():
        if file_hash(HERE / name) != digest:
            raise RuntimeError('Frozen source changed')
    if file_hash(manifest['datasets']['TSP']['path']) != manifest['datasets']['TSP']['input_sha256']:
        raise RuntimeError('Original TSP TRAIN changed')
    phases = [read_json(OUTPUT / f'{problem}__{method}__w{width}' / 'result.json')
              for problem, method in CASES for width in [1, 2, 4]]
    failed_attempts = [str(OUTPUT / f'diffusion20__{method}' / 'result.json') for method in DIFFUSION]
    start = time.monotonic()
    state = dict(status='running', started=time.time(), pid=os.getpid(), host=socket.gethostname(),
        preflight_proof=proof, gpu_before=before, original_wall_seconds=original['wall_seconds'],
        remaining_wall_seconds=remaining, pilot_source_sha256=file_hash(__file__),
        failed_attempts_preserved=failed_attempts, rerun_scope='Only four diffusion methods, no RF/CVRP reruns')
    write_json(OUTPUT / 'diffusion_retry_state.json', state)
    try:
        deadline = start + min(wall_seconds, remaining)
        for method in DIFFUSION:
            if time.monotonic() >= deadline:
                raise TimeoutError('Remaining original wall cap exhausted')
            phases.append(run_phase('TSP', method, 1, manifest['datasets']['TSP'], deadline,
                                    candidate=True, suffix='__retry1'))
            report = summarize(phases, manifest)
            report['failed_diffusion_attempts_preserved'] = failed_attempts
            write_json(OUTPUT / 'summary.json', report)
        failed = [p['directory'] for p in phases if p['candidate'] and p['status'] != 'complete']
        state.update(status='failed' if failed else 'complete', failed_diffusion_phases=failed,
            wall_seconds=time.monotonic() - start, finished=time.time(), gpu_after=gpu_snapshot(),
            frozen_sources_changed=[name for name, digest in manifest['frozen_source_sha256'].items()
                                    if file_hash(HERE / name) != digest])
        state['combined_single_gpu_wall_seconds'] = original['wall_seconds'] + state['wall_seconds']
        write_json(OUTPUT / 'diffusion_retry_state.json', state)
        return 2 if failed else 0
    except BaseException as error:
        state.update(status='failed', finished=time.time(), wall_seconds=time.monotonic() - start,
                     error=str(error), traceback=traceback.format_exc())
        write_json(OUTPUT / 'diffusion_retry_state.json', state)
        raise


def aug1_summary(phases, manifest):
    report = dict(protocol=manifest['protocol'], repeat_aggregation='Both complete repetitions: ratio of summed wall times, never best/min',
                  candidate_only=True, quality_equivalence_to_aug8=False, phases=[], aggregates=[])
    for phase in phases:
        row = {key: phase.get(key) for key in ['problem', 'method', 'workers', 'repeat', 'status', 'directory',
            'warm_completion_barrier_seconds', 'cold_wall_seconds', 'sampled_global_peak_mib']}
        serial = next((p for p in phases if p['problem'] == phase['problem'] and p['method'] == phase['method']
                       and p['repeat'] == phase['repeat'] and p['workers'] == 1), None)
        if serial:
            row['equivalence_to_aug1_serial'] = compare_equivalence(serial['merged'], phase['merged'])
        report['phases'].append(row)
    for problem in manifest['problems']:
        for method in manifest['methods']:
            for width in manifest['workers']:
                group = [r for r in report['phases'] if (r['problem'], r['method'], r['workers']) == (problem, method, width)]
                serial = [r for r in report['phases'] if (r['problem'], r['method'], r['workers']) == (problem, method, 1)]
                complete = (len(group) == len(serial) == 2 and all(r['status'] == 'complete' and
                    r.get('equivalence_to_aug1_serial', {}).get('equivalent') for r in group + serial))
                row = dict(problem=problem, method=method, workers=width, repetitions=len(group), complete=complete)
                if complete:
                    warm = [r['warm_completion_barrier_seconds'] for r in group]
                    cold = [r['cold_wall_seconds'] for r in group]
                    row.update(warm_seconds_by_repeat=warm, cold_seconds_by_repeat=cold,
                        mean_warm_seconds=statistics.mean(warm), mean_cold_seconds=statistics.mean(cold),
                        warm_speedup=sum(r['warm_completion_barrier_seconds'] for r in serial) / sum(warm),
                        cold_speedup=sum(r['cold_wall_seconds'] for r in serial) / sum(cold),
                        warm_instances_per_second=24 / sum(warm),
                        sampled_global_peak_mib=max(r['sampled_global_peak_mib'] for r in group))
                    row['per_scale'] = []
                    for scale in manifest['datasets'][problem]['scales']:
                        values = [v['synchronized_solve_seconds'] for p in phases
                                  if (p['problem'], p['method'], p['workers']) == (problem, method, width)
                                  for v in p['merged']['rows'] if v.get('status') == 'ok' and v['true_size'] == scale]
                        row['per_scale'].append(dict(true_size=scale, count=len(values), mean_seconds=statistics.mean(values),
                            max_seconds=max(values)))
                report['aggregates'].append(row)
    return report


def run_aug1(wall_seconds, supplement=False):
    os.environ.update(environment())
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f'Aug1 pilot received signal{signum}')
    signal.signal(signal.SIGTERM, interrupted)
    proof = gpu_guard()
    before = gpu_snapshot()
    if before['used_mib'] > 100 or before['utilization'] > 5 or before['contexts']:
        raise RuntimeError(f'Designated GPU not idle: {before}')
    from .r58_scenario import file_hash
    prefix = 'aug1_moses_rf' if supplement else 'aug1'
    manifest = read_json(OUTPUT / f'{prefix}_selection.json')
    if manifest['pilot_source_sha256'] != file_hash(__file__):
        raise RuntimeError('Aug1 source changed after preparation')
    for name, digest in manifest['frozen_source_sha256'].items():
        if file_hash(HERE / name) != digest:
            raise RuntimeError('Frozen source changed')
    for data in manifest['datasets'].values():
        if file_hash(data['path']) != data['input_sha256']:
            raise RuntimeError('Original TRAIN changed')
    previous = read_json(OUTPUT / ('aug1_state.json' if supplement else 'diffusion_retry_state.json'))
    if previous['status'] != 'complete':
        raise RuntimeError('Previous pilot must finish before sequential continuation')
    if supplement:
        parent_path = OUTPUT / 'aug1_selection.json'
        parent = read_json(parent_path)
        if (manifest['methods'] != ['MoSES_RF'] or manifest['datasets'] != parent['datasets']
                or manifest['parent_manifest_sha256'] != file_hash(parent_path)):
            raise RuntimeError('Supplement provenance differs from the completed aug1 pilot')
    used = previous['combined_single_gpu_wall_seconds']
    remaining = 10800 - used
    if remaining <= 0:
        raise RuntimeError('Original total3h GPU wall cap exhausted')
    start = time.monotonic()
    state = dict(status='running', started=time.time(), pid=os.getpid(), host=socket.gethostname(),
        preflight_proof=proof, gpu_before=before, previous_single_gpu_wall_seconds=used,
        remaining_wall_seconds=remaining, current_wall_cap_seconds=min(wall_seconds, remaining),
        pilot_source_sha256=file_hash(__file__), protocol=manifest['protocol'])
    write_json(OUTPUT / f'{prefix}_state.json', state)
    phases = []
    try:
        deadline = start + min(wall_seconds, remaining)
        for problem in manifest['problems']:
            for method in manifest['methods']:
                for repeat in [1, 2]:
                    for width in [1, 2, 4]:
                        if time.monotonic() >= deadline:
                            raise TimeoutError('Original total GPU wall cap exhausted')
                        phases.append(run_phase(problem, method, width, manifest['datasets'][problem], deadline,
                            suffix=f'__repeat{repeat}', rf_aug1=True, repeat=repeat))
                        write_json(OUTPUT / f'{prefix}_summary.json', aug1_summary(phases, manifest))
        state.update(status='complete' if all(p['status'] == 'complete' for p in phases) else 'complete_with_failures',
            wall_seconds=time.monotonic() - start, finished=time.time(), gpu_after=gpu_snapshot(),
            failed_phases=[p['directory'] for p in phases if p['status'] != 'complete'],
            frozen_sources_changed=[name for name, digest in manifest['frozen_source_sha256'].items()
                                    if file_hash(HERE / name) != digest])
        state['combined_single_gpu_wall_seconds'] = used + state['wall_seconds']
        write_json(OUTPUT / f'{prefix}_state.json', state)
        return 0 if not state['failed_phases'] else 2
    except BaseException as error:
        state.update(status='failed', wall_seconds=time.monotonic() - start, finished=time.time(),
                     error=str(error), traceback=traceback.format_exc())
        state['combined_single_gpu_wall_seconds'] = used + state['wall_seconds']
        write_json(OUTPUT / f'{prefix}_state.json', state)
        raise


def launch(include_diffusion, wall_seconds, retry=False, aug1=False, supplement=False):
    preflight_gate()
    if not (OUTPUT / 'selection.json').exists():
        raise RuntimeError('Run CPU preparation and tests first')
    if supplement:
        stage, launch_path, state_path, handle = ('run-aug1-supplement', 'aug1_moses_rf_launch.json',
            'aug1_moses_rf_state.json', 'r58_aug1_moses_rf')
    elif aug1:
        stage, launch_path, state_path, handle = 'run-aug1', 'aug1_launch.json', 'aug1_state.json', 'r58_aug1_pilot'
    elif retry:
        stage, launch_path, state_path, handle = 'retry-diffusion', 'diffusion_retry_launch.json', 'diffusion_retry_state.json', 'r58_diffusion_retry'
    else:
        stage, launch_path, state_path, handle = 'run', 'launch.json', 'run_state.json', 'r58_throughput_pilot'
    command = ['srun', '--jobid=1465', '--overlap', '--exact', '--nodes=1', '--ntasks=1',
               '--cpus-per-task=4', '--mem=16G', 'env',
               *[f'{key}={value}' for key, value in environment().items() if key in
                 ['CUDA_VISIBLE_DEVICES', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                  'NUMEXPR_NUM_THREADS', 'PYTHONDONTWRITEBYTECODE', 'WANDB_MODE', 'WANDB_DIR', 'MPLCONFIGDIR', 'PYTHONPATH']],
               'timeout', '--signal=TERM', '--kill-after=30s', str(min(wall_seconds, 10800)) + 's',
               sys.executable, '-u', '-m', 'code.V4.r58_throughput', '--stage', stage,
               '--wall-seconds', str(min(wall_seconds, 10800))]
    if include_diffusion:
        command.append('--diffusion')
    write_json(OUTPUT / launch_path,
        dict(command=command, started=time.time(), pid=os.getpid(), queue_wait_limit_seconds=600,
             tmux_handle=handle))
    print(' '.join(command), flush=True)
    process = subprocess.Popen(command, start_new_session=True, env=environment())
    wait_start = time.monotonic()
    try:
        while not (OUTPUT / state_path).exists():
            if process.poll() is not None:
                raise RuntimeError(f'srun exited before GPU pilot startup: {process.returncode}')
            if time.monotonic() - wait_start >= 600:
                raise TimeoutError('Existing allocation blocked for600s; no new allocation requested')
            time.sleep(1)
        return process.wait(timeout=min(wall_seconds, 10800) + 90)
    finally:
        stop_children([process])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', required=True,
                        choices=['prepare', 'launch', 'run', 'worker', 'launch-diffusion-retry', 'retry-diffusion',
                                 'prepare-aug1', 'launch-aug1', 'run-aug1', 'prepare-aug1-supplement',
                                 'launch-aug1-supplement', 'run-aug1-supplement'])
    parser.add_argument('--diffusion', action='store_true')
    parser.add_argument('--wall-seconds', type=int, default=10800)
    parser.add_argument('--task', type=Path)
    parser.add_argument('--worker-id', type=int)
    parser.add_argument('--aug1-methods', nargs='+', choices=['RouteFinder', 'MoSES_CaDA', 'MoSES_RF'],
                        default=['RouteFinder', 'MoSES_CaDA'])
    args = parser.parse_args()
    if not 0 < args.wall_seconds <= 10800:
        parser.error('Pilot wall budget must be positive and at most3 single-GPU hours')
    if args.stage == 'prepare':
        prepare()
    elif args.stage == 'launch':
        raise SystemExit(launch(args.diffusion, args.wall_seconds))
    elif args.stage == 'run':
        run(args.diffusion, args.wall_seconds)
    elif args.stage == 'launch-diffusion-retry':
        raise SystemExit(launch(False, args.wall_seconds, retry=True))
    elif args.stage == 'retry-diffusion':
        raise SystemExit(retry_diffusion(args.wall_seconds))
    elif args.stage == 'prepare-aug1':
        prepare_aug1(args.aug1_methods)
    elif args.stage == 'launch-aug1':
        raise SystemExit(launch(False, args.wall_seconds, aug1=True))
    elif args.stage == 'run-aug1':
        raise SystemExit(run_aug1(args.wall_seconds))
    elif args.stage == 'prepare-aug1-supplement':
        prepare_aug1(['MoSES_RF'], supplement=True)
    elif args.stage == 'launch-aug1-supplement':
        raise SystemExit(launch(False, args.wall_seconds, supplement=True))
    elif args.stage == 'run-aug1-supplement':
        raise SystemExit(run_aug1(args.wall_seconds, supplement=True))
    else:
        worker(args.task, args.worker_id)


if __name__ == '__main__':
    main()
