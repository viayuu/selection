"""Explicit second authorized continuation after the two preserved R74 failures."""

import difflib
import json
import os
import shutil
import socket
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

from . import r74_experiment as experiment


ROOT = experiment.ROOT
RECORD = ROOT / 'continuation2'
PRIOR_RECORD = ROOT / 'continuation'
CHANGED = ('r74_acquisition.py', 'r74_experiment.py', 'r74_continuation.py', 'run_v4_r74_continuation.sh')
ADDED = ()
PRIOR_STEPS = [dict(step='1465.111', state='FAILED', exit_code='1:0', elapsed_seconds=1427),
               dict(step='1465.112', state='FAILED', exit_code='1:0', elapsed_seconds=20)]
PRIOR_SECONDS = 1447
PREFIX = dict(base_query_rows=20000, query_batches=1728, trajectory_initializations=20000,
    start_trajectories=982780, model_action_calls=117370, decoder_calls=116512,
    completed_trajectories=0, train_rows=20000, train_initializations=20000)


def prepare(process_started):
    authorization = json.loads((ROOT / 'runtime_repair2_authorization.json').read_text())
    first_authorization = json.loads((ROOT / 'runtime_repair_authorization.json').read_text())
    protocol = json.loads((ROOT / 'protocol.json').read_text())
    failure = json.loads((ROOT / 'pipeline_status.json').read_text())
    prior_failure = json.loads((PRIOR_RECORD / 'pipeline_status.json').read_text())
    prior_amendment = json.loads((PRIOR_RECORD / 'execution_amendment.json').read_text())
    prior = json.loads((ROOT / 'current_status.json').read_text())
    if (first_authorization['original_step'] != '1465.111' or first_authorization['original_state'] != 'FAILED'
            or first_authorization['original_exit_code'] != '1:0' or first_authorization['original_elapsed_seconds'] != 1427
            or failure['slurm_step'] != '1465.111' or failure['stage'] != 'failed_preserved_no_retry'
            or failure['error_type'] != 'ModuleNotFoundError' or failure['error'] != "No module named 'neural_solvers'"):
        raise ValueError('Only the specifically authorized original import failure may continue')
    if (authorization['prior_steps'] != PRIOR_STEPS or authorization['charged_prior_Slurm_wall_seconds'] != PRIOR_SECONDS
            or prior_failure['slurm_step'] != '1465.112' or prior_failure['stage'] != 'failed_preserved_no_retry'
            or prior_failure['error_type'] != 'OSError' or 'GLIBC_2.29' not in prior_failure['error']
            or prior_failure['last_saved_progress']['counters'] != PREFIX):
        raise ValueError('Only the specifically authorized second namespace repair may continue')
    for path, digest in prior_amendment['protected_original_text_sha256'].items():
        if experiment.file_hash(path) != digest:
            raise ValueError('Protected original record changed since first repair: '+path)
    if (prior.get('split') != 'train' or prior.get('problem') != 'ATSP' or prior['counters'] != PREFIX
            or prior['successful_updates'] != 0):
        raise ValueError('Original saved ATSP-start counters differ; no replay or inferred resume')
    expected_authorization = {key: value for key, value in PREFIX.items()
                              if key not in ('train_rows', 'train_initializations')}
    expected_authorization.update(tasks=['train/TSP', 'train/CVRP'], successful_updates=0)
    if first_authorization['completed_prefix'] != expected_authorization:
        raise ValueError('Authorized prefix differs from actual saved prefix')
    last = None
    with (ROOT / 'feature_batches.jsonl').open() as stream:
        for line in stream:
            if line.strip():
                last = json.loads(line)
    if last is None or last['counters'] != PREFIX or (last['split'], last['problem']) != ('train', 'CVRP'):
        raise ValueError('Last original successful batch does not match the saved prefix')
    for name in ('feature_cache_complete.json', 'training_complete.json', 'first_formal_optimization_update.json'):
        if (ROOT / name).exists():
            raise ValueError('Unexpected later formal work; preserve it without restarting: '+name)
    for name in ('updates.jsonl', 'optimizer_initialization.json', 'last.pt', 'final.pt', 'best_checkpoint.json'):
        if (experiment.DIRECTORY / name).exists():
            raise ValueError('Policy training already exists; fresh optimization is not authorized: '+name)
    if (RECORD / 'formal_started.json').exists() or (RECORD / 'execution_amendment.json').exists():
        raise ValueError('This single continuation was already attempted; no automatic retry')
    syntax = json.loads((RECORD / 'syntax_check.json').read_text())
    sources = Path(__file__).parent
    identities = {name: experiment.file_hash(sources / name) for name in CHANGED+ADDED}
    if not syntax['passed'] or syntax['source_hashes'] != identities or syntax['invocations'] != 1:
        raise ValueError('Exactly one syntax-only check on these execution sources is required')
    check = json.loads((ROOT / 'cpu_check.json').read_text())
    original_owned = {name: protocol['source_identities'][str(sources / name)] for name in experiment.OWNED}
    if not check['passed'] or check['source_hashes'] != original_owned:
        raise ValueError('Preserve the passed original combined CPU check and its source identities')
    for path, digest in protocol['source_identities'].items():
        if path not in {str(sources / name) for name in CHANGED} and experiment.file_hash(path) != digest:
            raise ValueError('Unapproved original source change: '+path)
    if experiment.file_hash(ROOT / 'approved_plan.md') != experiment.file_hash(ROOT / 'source_launch/approved_plan.md'):
        raise ValueError('Approved scientific plan changed')
    completed, protected, sums = {}, {}, experiment.Counter()
    original_manifest = json.loads((ROOT / 'data_manifest.json').read_text())
    for split in ('train', 'val'):
        for problem in experiment.PROBLEMS:
            task_root = ROOT / 'feature_cache' / split / problem
            key = split+'/'+problem
            if key not in ('train/TSP', 'train/CVRP'):
                if task_root.exists():
                    raise ValueError('Unexpected partial or completed task; no reconstruction: '+key)
                continue
            metadata_path = task_root / 'metadata.json'
            metadata = json.loads(metadata_path.read_text())
            if (not metadata['complete'] or metadata['rows'] != 10000 or metadata['problem'] != problem
                    or metadata['split'] != split or metadata['source_identity'] != original_manifest[split][problem]
                    or metadata['labels_used_in_features']):
                raise ValueError('Original complete task metadata differs: '+key)
            rows = batches = starts = 0
            for group in metadata['groups']:
                group_path = Path(group['directory']) / 'metadata.json'
                if (group_path.parent.parent != task_root or not group['complete']
                        or json.loads(group_path.read_text()) != group):
                    raise ValueError('Original complete group metadata differs: '+str(group_path))
                protected[str(group_path)] = experiment.file_hash(group_path)
                rows += group['rows']
                batches += (group['rows']+15)//16
                starts += group['rows']*group['starts']
            counters = metadata['counters']
            if rows != 10000 or counters['rows'] != rows or counters['query_batches'] != batches:
                raise ValueError('Original complete group counts differ: '+key)
            sums.update(base_query_rows=rows, train_rows=rows, query_batches=batches,
                trajectory_initializations=counters['trajectory_initializations'],
                train_initializations=counters['trajectory_initializations'], start_trajectories=starts,
                model_action_calls=counters['model_action_calls'], decoder_calls=counters['decoder_calls'],
                completed_trajectories=0)
            protected[str(metadata_path)] = experiment.file_hash(metadata_path)
            completed[key] = metadata
    if dict(sums) != PREFIX:
        raise ValueError('Complete task text metadata does not account for the exact saved counters')
    schedule = experiment.paired_triples({task: 10000 for task in experiment.PROBLEMS},
        epochs=80, seed=2, batch_size=256)
    if experiment.state_hash(schedule) != protocol['schedule']['sha256']:
        raise ValueError('Original seed2 global41-80 task/row schedule changed')
    r48 = json.loads((experiment.WORKSPACE / 'code/V4/runs/R48_core_diagnosis/audit_plan.json').read_text())['solvers']
    if {key: value['runtime_params_contract'] for key, value in r48.items()} != protocol['reld_runtime_contracts']:
        raise ValueError('Original native ReLD runtime contracts changed')
    diffs = {}
    for name in CHANGED+ADDED:
        destination = RECORD / 'source_execution' / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        current = sources / name
        old = PRIOR_RECORD / 'source_execution' / name
        before = old.read_text()
        if experiment.file_hash(old) != prior_amendment['exact_source_changes'][name]['new_sha256']:
            raise ValueError('First continuation execution snapshot identity changed: '+name)
        diffs[name] = dict(old_sha256=experiment.file_hash(old),
            new_sha256=identities[name], old_source=str(old),
            new_source=str(destination), unified_diff=''.join(difflib.unified_diff(
                before.splitlines(keepends=True), current.read_text().splitlines(keepends=True),
                fromfile=str(old), tofile=str(current))))
        shutil.copy2(current, destination)
    for name in ('protocol.json', 'approved_plan.md', 'cpu_check.json', 'pipeline_status.json',
                 'current_status.json', 'failures.jsonl', 'train.log', 'feature_batches.jsonl',
                 'first_formal_extraction_batch.json', 'formal_started.json', 'trainer_launch.json',
                 'initialization.json', 'probe_initialization.json', 'data_manifest.json'):
        protected[str(ROOT / name)] = experiment.file_hash(ROOT / name)
    for path in PRIOR_RECORD.rglob('*'):
        if path.is_file() and path.suffix in ('.json', '.jsonl', '.log', '.py', '.sh'):
            protected[str(path)] = experiment.file_hash(path)
    native = {name: dict(path=str(path), sha256=experiment.file_hash(path)) for name, path in (
        ('MatNetEncoder', experiment.EASY_ROOT / 'neural_solvers/methods/matnet/matnet_encoder.py'),
        ('ATSPDecoder_and_FFSPDecoder', experiment.EASY_ROOT / 'neural_solvers/methods/matnet/matnet_decoder.py'),
        ('ATSPPolicy', experiment.EASY_ROOT / 'neural_solvers/methods/matnet/policy.py'),
        ('ATSPEnv', experiment.EASY_ROOT / 'neural_solvers/envs/ATSPEnv.py'))}
    amendment = dict(utc=experiment.now(), authorization=str(ROOT / 'runtime_repair2_authorization.json'),
        reason='Scoped native package paths and real MATNET exports bypass unrelated solver/env registry initializers; no DIFUSCO or GNN substitution',
        original_step='1465.111', original_state='FAILED', original_exit_code='1:0',
        original_step_elapsed_seconds=1427, original_failures_retained=True, original_job_reclassified=False,
        prior_steps=PRIOR_STEPS, charged_prior_Slurm_wall_seconds=PRIOR_SECONDS,
        previous_amendment=str(PRIOR_RECORD / 'execution_amendment.json'), original_native_sources=native,
        exact_source_changes=diffs, original_protocol_sha256=protected[str(ROOT / 'protocol.json')],
        approved_plan_sha256=protected[str(ROOT / 'approved_plan.md')],
        unchanged=['scientific configuration', 'complete averaged-x R73B23/global63 base and buffers',
            'probe classes/configuration/actual frozen weights', 'directed matrices and original inputs',
            'private seed2 MATNET permutation per true n, independent of batches', 'native labels and legal candidates',
            'objective', 'seed2 global41-80 task/row order', 'seed2 fresh policy initialization',
            'fresh official optimizer and scaler', 'fixed 40 epochs/9600 updates', 'acquisition budgets'],
        initial_counters=PREFIX, successful_updates_before_continuation=0,
        completed_tasks=list(completed), next_task='train/ATSP', completed_tasks_skipped_before_native_inputs_or_probes=True,
        cache_resume_validation='JSON metadata only; no old feature arrays or model replay',
        base_total_unique_inputs=198000, base_inputs_already_processed=20000, base_inputs_remaining=178000,
        protected_original_text_sha256=protected, original_neural_cpu_check_repeated=False,
        syntax_check=syntax, automatic_retry=False, test_authorized=False, goal_status='ACTIVE; unachieved')
    experiment.atomic_json(RECORD / 'execution_amendment.json', amendment)
    experiment.atomic_json(RECORD / 'original_prefix.json', dict(counters=PREFIX, completed_tasks=completed,
        original_step_elapsed_seconds=1427, prior_steps=PRIOR_STEPS, charged_prior_Slurm_wall_seconds=PRIOR_SECONDS,
        original_status=failure, first_continuation_status=prior_failure, zero_optimizer_updates=True))
    loaded = json.loads((ROOT / 'probe_initialization.json').read_text())
    if set(loaded) != {'BQ', 'RELD_CVRP'}:
        raise ValueError('Actual original frozen probe identities differ')
    return protocol, schedule, r48, dict(initial_counters=PREFIX, completed_tasks=completed, loaded_probes=loaded,
        original_elapsed_seconds=PRIOR_SECONDS, original_step_elapsed_seconds=1427, process_started=process_started)


def main():
    started, launch, wb, continuation = time.monotonic(), None, None, None
    try:
        if (os.environ.get('CUDA_VISIBLE_DEVICES') != experiment.GPU_UUID
                or os.environ.get('SLURM_JOB_ID') != '1465' or socket.gethostname().split('.')[0] != 'gpu03'):
            raise ValueError('Only existing1465/gpu03/sole assigned UUID is authorized')
        launch = json.loads((RECORD / 'launch_receipt.json').read_text())
        started = time.monotonic()-max(0., (datetime.now(timezone.utc)-
            datetime.fromisoformat(launch['bootstrap_started_utc'])).total_seconds())
        launch.update(trainer_pid=os.getpid(), trainer_ppid=os.getppid(), started_utc=experiment.now(),
            command='bash code/V4/run_v4_r74_continuation.sh', execution_continuation=True,
            continuation_record='continuation2', prior_steps=PRIOR_STEPS, charged_prior_Slurm_wall_seconds=PRIOR_SECONDS,
            slurm_step=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'], status='running')
        experiment.atomic_json(RECORD / 'trainer_launch.json', launch)
        import torch
        torch.set_num_threads(1)
        protocol, schedule, r48, continuation = prepare(started)
        experiment.atomic_json(RECORD / 'formal_started.json', dict(launch,
            original_step='1465.111', original_status='FAILED 1:0', automatic_retry=False,
            test_authorized=False, phases=['remaining_feature_generation', 'unchanged_fixed40E_policy']))
        import wandb
        wb = wandb.init(**protocol['wandb'], name='R74_B_bounded_acquisition_seed2_continuation2',
            config=protocol, dir=str(ROOT), id='r74-b-cont2-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S'))
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision('highest')
        experiment.set_seed(2)
        device = 'cuda:0'
        model, counts = experiment.restore_base(protocol, device, record_root=RECORD)
        caches, raw, generation = experiment.generate(protocol, r48, model, counts, launch, wb, device,
            continuation=continuation)
        experiment.train(protocol, schedule, model, caches, raw, generation, launch, wb, device)
        elapsed = time.monotonic()-started
        launch.update(status='continuation_complete', completed_utc=experiment.now(),
            continuation_formal_wall_seconds=elapsed, original_failed_step_seconds=1427,
            charged_prior_Slurm_wall_seconds=PRIOR_SECONDS,
            cumulative_formal_wall_seconds=PRIOR_SECONDS+elapsed, original_job_status='FAILED 1:0; never reclassified')
        experiment.atomic_json(RECORD / 'trainer_launch.json', launch)
    except BaseException as error:
        elapsed = time.monotonic()-started
        progress = json.loads((RECORD / 'current_status.json').read_text()) if (RECORD / 'current_status.json').exists() else None
        failure = dict(utc=experiment.now(), pid=os.getpid(), requested_stage='explicit_continuation',
            error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc(),
            automatic_restart=False, original_job_status='FAILED 1:0; preserved', original_failed_step_seconds=1427,
            prior_steps=PRIOR_STEPS, charged_prior_Slurm_wall_seconds=PRIOR_SECONDS,
            continuation_formal_seconds=elapsed, cumulative_formal_wall_seconds=PRIOR_SECONDS+elapsed,
            slurm_step=os.environ.get('SLURM_JOB_ID', '')+'.'+os.environ.get('SLURM_STEP_ID', ''),
            last_saved_progress=progress, update_failure_receipt=getattr(error, 'r74_update_receipt', None))
        experiment.append(RECORD / 'failures.jsonl', failure)
        experiment.atomic_json(RECORD / 'pipeline_status.json', dict(failure, stage='failed_preserved_no_retry'))
        if launch is not None:
            launch.update(status='failed_preserved_no_retry', failed_utc=experiment.now(),
                continuation_formal_wall_seconds=elapsed, original_failed_step_seconds=1427,
                charged_prior_Slurm_wall_seconds=PRIOR_SECONDS, cumulative_formal_wall_seconds=PRIOR_SECONDS+elapsed)
            experiment.atomic_json(RECORD / 'trainer_launch.json', launch)
        raise
    finally:
        if wb is not None:
            wb.finish()


if __name__ == '__main__':
    main()
