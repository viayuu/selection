"""Authorized R74 continuation7: reuse complete caches, restore initial state, train40E."""

import csv
import json
import os
import socket
import subprocess
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path


WORKSPACE = Path(__file__).resolve().parents[2]
ROOT = WORKSPACE / 'code/V4/runs/R74_adaptive_incomplete_acquisition'
RECORD = ROOT / 'continuation7'
RUN = ROOT / 'B_adaptive_seed2'
GPU_UUID = 'GPU-cb626bdb-4319-2f5e-b91f-17fcaa44a83d'
SOURCES = ('r74_experiment.py', 'r74_train_only_continuation.py', 'run_v4_r74_continuation7.sh')


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(path.read_text())


def exclusive(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def assigned_launch():
    if (os.environ.get('SLURM_JOB_ID') != '1465' or os.environ.get('CUDA_VISIBLE_DEVICES') != GPU_UUID
            or socket.gethostname().split('.')[0] != 'gpu03'):
        raise ValueError('Only existing1465/gpu03/assigned UUID is authorized')
    if any((RECORD / name).exists() for name in ('launch_receipt.json', 'formal_started.json', 'failures.jsonl')):
        raise ValueError('Original continuation7 already attempted; no automatic retry')
    while True:
        gpu_command = ['nvidia-smi', '--id='+GPU_UUID, '--query-gpu=index,uuid,name,memory.used,memory.total', '--format=csv,noheader,nounits']
        apps_command = ['nvidia-smi', '--id='+GPU_UUID, '--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory', '--format=csv,noheader,nounits']
        gpu = subprocess.run(gpu_command, text=True, capture_output=True)
        apps = subprocess.run(apps_command, text=True, capture_output=True)
        rows = list(csv.reader(gpu.stdout.splitlines()))
        if gpu.returncode or apps.returncode or len(rows) != 1:
            raise RuntimeError('Assigned-only idle observation failed; preserve without switching devices')
        index, uuid, name, used, total = [item.strip() for item in rows[0]]
        if index != '1' or uuid != GPU_UUID:
            raise ValueError('Assigned UUID/index mismatch')
        clients = [row for row in csv.reader(apps.stdout.splitlines()) if row and row[0].strip() == GPU_UUID]
        receipt = dict(utc=now(), host=socket.gethostname(), gpu_uuid=uuid, gpu_index=1, gpu_name=name,
            memory_used_mib=int(used), memory_total_mib=int(total), compute_apps=clients,
            idle=int(used)<500 and not clients, only_assigned_uuid_queried=True,
            gpu_observation=dict(command=gpu_command, returncode=gpu.returncode, stdout=gpu.stdout, stderr=gpu.stderr),
            assigned_client_observation=dict(command=apps_command, returncode=apps.returncode, stdout=apps.stdout, stderr=apps.stderr))
        with (RECORD / 'availability.jsonl').open('a') as stream:
            stream.write(json.dumps(receipt)+'\n')
        if receipt['idle']:
            launch = dict(receipt, trainer_pid=os.getpid(), trainer_ppid=os.getppid(), started_utc=now(),
                slurm_step=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],
                tmux_session='r74_continuation7', tmux_pane=os.environ['R74_TMUX_PANE_ID'],
                tmux_pane_pid=os.environ['R74_TMUX_PANE_PID'], tmux_server_host=os.environ['R74_TMUX_SERVER_HOST'],
                command='bash code/V4/run_v4_r74_continuation7.sh', execution_continuation=True,
                continuation_record='continuation7', train_only_continuation=True, status='running', automatic_restart=False)
            exclusive(RECORD / 'launch_receipt.json', launch)
            return launch
        print('[R74 continuation7 wait] assigned UUID occupied; waiting30s on the same device', flush=True)
        time.sleep(30)


def prepare(experiment, launch):
    authorization = read(ROOT / 'runtime_repair7_authorization.json')
    failure = read(ROOT / 'continuation6/pipeline_status.json')
    prior = read(ROOT / 'continuation6/trainer_launch.json')
    terminal = read(ROOT / 'delivery_refresh/formal_failure_observation.json')
    if (authorization['failed_step'] != '1465.117' or authorization['charged_prior_slurm_seconds'] != 3839
            or failure['error'] != "dict() got multiple values for keyword argument 'successful_leaf_view_rows'"
            or failure['slurm_step'] != '1465.117' or terminal['state'] != 'FAILED'
            or terminal['exit_code'] != '1:0' or terminal['elapsed_seconds'] != 2292
            or not terminal['tmux_terminal'] or prior['status'] != 'failed_preserved_no_retry'):
        raise ValueError('Only the demonstrated unsaved-update logging failure may continue')
    for name in ('last.pt', 'updates.jsonl', 'history.json', 'final.pt', 'best_checkpoint.json'):
        if (RUN / name).exists():
            raise ValueError('Unexpected persisted training work; preserve without restart: '+name)
    if (ROOT / 'first_formal_optimization_update.json').exists():
        raise ValueError('An actual persisted update already exists; do not duplicate formal training')
    protocol, generation = read(ROOT / 'protocol.json'), read(ROOT / 'feature_cache_complete.json')
    released = read(ROOT / 'probes_released.json')
    if (not generation['complete'] or not released['no_live_probe_states'] or not released['before_optimizer_construction']
            or generation['counters']['base_query_rows'] != 198000 or generation['counters']['query_batches'] != 14745
            or generation['counters']['trajectory_initializations'] != 363000
            or generation['probe_execution']['actual_cumulative_initialization_rows'] != 363032
            or generation['counters']['completed_trajectories'] != 0
            or generation['execution_amendment'] != str(ROOT / 'continuation6/execution_amendment.json')):
        raise ValueError('Reuse the actual complete original cache and released probes only')
    syntax = read(RECORD / 'syntax_check.json')
    hashes = {name: experiment.file_hash(Path(__file__).parent / name) for name in SOURCES}
    if syntax['returncode'] != 0 or syntax['invocation'] != 1 or any(syntax['source_hashes'][name] != value for name, value in hashes.items()):
        raise ValueError('Require the one combined syntax invocation on these actual continuation sources')
    prior_steps = prior['prior_steps'] + [dict(step='1465.117', state='FAILED', exit_code='1:0', elapsed_seconds=2292)]
    if sum(row['elapsed_seconds'] for row in prior_steps) != 3839:
        raise ValueError('All seven failed Slurm attempts remain separately charged')
    launch.update(prior_steps=prior_steps, charged_prior_Slurm_wall_seconds=3839)
    previous = read(ROOT / 'continuation6/execution_amendment.json')
    old = RECORD / 'source_before/r74_experiment.py'
    amendment = dict(utc=now(), authorization=str(ROOT / 'runtime_repair7_authorization.json'),
        reason='Disambiguate per-update leaf_view_rows from cumulative successful_leaf_view_rows in logging; '
               'preserve old optimizer receipt and train only from persistent unchanged initial state and complete cache',
        previous_amendment=str(ROOT / 'continuation6/execution_amendment.json'),
        generation_execution_amendment=generation['execution_amendment'], prior_steps=prior_steps,
        charged_prior_Slurm_wall_seconds=3839, original_native_sources=previous['original_native_sources'],
        original_protocol_sha256=previous['original_protocol_sha256'], approved_plan_sha256=previous['approved_plan_sha256'],
        exact_source_changes={name: dict(old_sha256=experiment.file_hash(old) if name == 'r74_experiment.py' else None,
            new_sha256=hashes[name], old_source=str(old) if name == 'r74_experiment.py' else None,
            new_source=str(RECORD / 'source_execution' / name)) for name in SOURCES},
        unchanged_scientific_protocol=True, seed2_initial_state=str(RUN / 'initial.pt'),
        complete_generation_receipt=str(ROOT / 'feature_cache_complete.json'), generation_regenerated=False,
        base_or_probe_forwards=0, no_restore_from_unsaved_update=True, fresh_optimizer_scaler_and_clock=True,
        accepted_updates=9600, accepted_native_presentations=7200000, accepted_leaf_view_rows=38400000,
        abandoned_successful_optimizer_steps_source_derived=1,
        old_optimizer_receipt=str(RUN / 'optimizer_initialization.json'),
        archived_old_optimizer_receipt=str(RECORD / 'abandoned_optimizer_initialization.json'),
        original_CPU_check_repeated=False, syntax_check=str(RECORD / 'syntax_check.json'), automatic_retry=False)
    snapshot = RECORD / 'source_execution'
    snapshot.mkdir(exist_ok=False)
    import shutil
    for name in SOURCES:
        shutil.copy2(Path(__file__).parent / name, snapshot / name)
    exclusive(RECORD / 'execution_amendment.json', amendment)
    exclusive(RECORD / 'formal_started.json', dict(utc=now(), launch=launch,
        phases=['complete_cache_and_initial_state_restore', 'fixed40E_policy_training'],
        generation_authoritative=str(ROOT / 'feature_cache_complete.json'), generation_regenerated=False,
        base_or_probe_forwards=0, automatic_retry=False, test_authorized=False))
    experiment.atomic_json(RECORD / 'trainer_launch.json', launch)
    experiment.status('complete_cache_training_setup', launch, successful_updates=0,
        generation_complete=True, generation_counters=generation['counters'], new_base_or_probe_forwards=0)
    return protocol, generation


def setup_training(experiment, protocol, generation, launch):
    import torch
    from .r74_model import R74Model
    schedule = experiment.paired_triples({task: 10000 for task in experiment.PROBLEMS}, epochs=80, seed=2, batch_size=256)
    if experiment.state_hash(schedule) != protocol['schedule']['sha256']:
        raise ValueError('Exact original seed2 global41-80 task/row schedule required')
    entries = experiment.triple_indices(schedule, 40, 0)
    native = sum(len(indices) for _, indices in entries)
    leaf_views = sum(2*len(indices)*(1 if problem in ('TSP', 'CVRP', 'ATSP') else 3) for problem, indices in entries)
    abandoned = dict(utc=now(), evidence_type='Source/control-flow-derived; unsaved optimizer state/log values unavailable',
        failed_step='1465.117', successful_optimizer_steps=1, native_presentations=native, leaf_view_rows=leaf_views,
        scheduled_first_triple={problem: indices.tolist() for problem, indices in entries}, task_rows=[len(indices) for _, indices in entries],
        global_epoch=41, update_in_epoch=1, dropout_seed=experiment.dropout_seed(9600),
        optimizer_state_restorable=False, old_loss_gradient_memory_and_AMP_retries=None,
        attempted_rows_including_old_AMP_retries=None, counted_in_new_accepted_budget=False,
        derivation='update_triple returns only after one optimizer.step; train advanced counters then duplicate-key metrics construction raised before log/save/first-update receipt')
    exclusive(RECORD / 'abandoned_update.json', abandoned)
    if experiment.file_hash(RECORD / 'abandoned_optimizer_initialization.json') != experiment.file_hash(RUN / 'optimizer_initialization.json'):
        raise ValueError('Original failed-attempt optimizer initialization receipt must remain archived and untouched')
    raw, caches = {}, {}
    manifest = read(ROOT / 'data_manifest.json')
    for split, count in (('train', 10000), ('val', 1000)):
        raw[split], caches[split] = {}, {}
        for problem in experiment.PROBLEMS:
            value = experiment.read_raw_costs(problem, split)
            metadata = generation['per_task'][split+'/'+problem]
            if (not metadata['complete'] or metadata['rows'] != count or len(value['winner']) != count
                    or {key: value[key] for key in ('pool', 'pool_ids', 'data_hash', 'label_hash')} != manifest[split][problem]
                    or metadata['source_identity'] != manifest[split][problem]):
                raise ValueError('Only complete aligned original TRAIN/VAL caches and unchanged supervision are allowed')
            raw[split][problem] = value
            caches[split][problem] = experiment.CachedTask(metadata, value)
    initial = torch.load(RUN / 'initial.pt', map_location='cpu', weights_only=False)
    if json.loads(json.dumps(initial['args'])) != protocol:
        raise ValueError('Persistent original initial model recipe differs from the scientific protocol')
    model = R74Model(protocol['model_params'])
    model.load_state_dict(initial['model'], strict=True)
    model.to('cuda:0').float().eval()
    counts = initial['parameter_counts']
    if (sum(p.numel() for p in model.parameters()) != counts['full_model_parameters']
            or sum(p.numel() for p in model.policy.parameters()) != counts['trainable_policy_parameters']
            or any(p.requires_grad for p in model.base.parameters())):
        raise ValueError('Strict original initial policy and frozen complete F required')
    for field in ('mean', 'std'):
        if not torch.equal(getattr(model.base.instance_encoder.geometry_residual, field),
                           torch.tensor(protocol['geometry_stats'][field], device='cuda:0', dtype=torch.float32)):
            raise ValueError('Original saved normalization buffers changed')
    experiment.atomic_json(RECORD / 'initialization.json', dict(utc=now(), persistent_initial_state=str(RUN / 'initial.pt'),
        strict_complete_restore=True, seed=2, fresh_optimizer_scaler_clock=True,
        unsaved_update_transferred=False, generation_regenerated=False, base_or_probe_forwards=0, **counts))
    del initial
    return model, caches, raw, schedule


def main():
    started, launch, wb = time.monotonic(), None, None
    try:
        launch = assigned_launch()
        from . import r74_experiment as experiment
        import torch
        torch.set_num_threads(1)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision('highest')
        protocol, generation = prepare(experiment, launch)
        import wandb
        wb = wandb.init(**protocol['wandb'], name='R74_B_cache_only_continuation7_seed2', dir=str(ROOT),
            id='r74-b-cont7-'+launch['slurm_step'].replace('.', '-'), config=protocol)
        model, caches, raw, schedule = setup_training(experiment, protocol, generation, launch)
        experiment.train(protocol, schedule, model, caches, raw, generation, launch, wb, 'cuda:0')
        elapsed = time.monotonic()-started
        launch.update(status='continuation_complete', completed_utc=now(), continuation_formal_wall_seconds=elapsed,
            cumulative_formal_wall_seconds=3839+elapsed, charged_prior_Slurm_wall_seconds=3839,
            generation_regenerated=False, new_base_or_probe_forwards=0)
        experiment.atomic_json(RECORD / 'trainer_launch.json', launch)
    except BaseException as error:
        failure = dict(utc=now(), pid=os.getpid(), error_type=type(error).__name__, error=str(error),
            traceback=traceback.format_exc(), automatic_restart=False, charged_prior_Slurm_wall_seconds=3839,
            slurm_step=os.environ.get('SLURM_JOB_ID', '')+'.'+os.environ.get('SLURM_STEP_ID', ''),
            train_only_continuation=True, generation_regenerated=False, new_base_or_probe_forwards=0,
            continuation_formal_wall_seconds=time.monotonic()-started)
        with (RECORD / 'failures.jsonl').open('a') as stream:
            stream.write(json.dumps(failure)+'\n')
        from .r73_frozen_evaluation import write
        write(RECORD / 'pipeline_status.json', dict(failure, stage='failed_preserved_no_retry'))
        if launch is not None:
            launch.update(status='failed_preserved_no_retry', failed_utc=now())
            write(RECORD / 'trainer_launch.json', launch)
        raise
    finally:
        if wb is not None:
            wb.finish()


if __name__ == '__main__':
    main()
