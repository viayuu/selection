"""Sixth authorized execution repair, preserving failed partial native work."""

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
RECORD = ROOT / 'continuation6'
PRIOR_RECORD = ROOT / 'continuation5'
CHANGED = ('r74_acquisition.py', 'r74_experiment.py', 'r74_continuation.py', 'run_v4_r74_continuation.sh')
ADDED = ()
PRIOR_STEPS = [dict(step='1465.111', state='FAILED', exit_code='1:0', elapsed_seconds=1427),
               dict(step='1465.112', state='FAILED', exit_code='1:0', elapsed_seconds=20),
               dict(step='1465.113', state='FAILED', exit_code='1:0', elapsed_seconds=14),
               dict(step='1465.114', state='FAILED', exit_code='1:0', elapsed_seconds=13),
               dict(step='1465.115', state='FAILED', exit_code='1:0', elapsed_seconds=13),
               dict(step='1465.116', state='FAILED', exit_code='1:0', elapsed_seconds=60)]
PRIOR_SECONDS = 1547
PREFIX = dict(base_query_rows=30000, query_batches=2376, trajectory_initializations=30000,
    start_trajectories=1581966, model_action_calls=126530, decoder_calls=125024,
    completed_trajectories=0, train_rows=30000, train_initializations=30000)


def prepare(process_started):
    authorization = json.loads((ROOT / 'runtime_repair6_authorization.json').read_text())
    protocol = json.loads((ROOT / 'protocol.json').read_text())
    failure = json.loads((ROOT / 'pipeline_status.json').read_text())
    prior_failure = json.loads((PRIOR_RECORD / 'pipeline_status.json').read_text())
    prior_handoff = json.loads((PRIOR_RECORD / 'terminal_handoff.json').read_text())
    prior_amendment = json.loads((PRIOR_RECORD / 'execution_amendment.json').read_text())
    prefix_receipt = json.loads((PRIOR_RECORD / 'original_prefix.json').read_text())
    prior_base = json.loads((PRIOR_RECORD / 'base_execution_progress.json').read_text())
    if (authorization['prior_steps'] != PRIOR_STEPS or authorization['charged_prior_Slurm_wall_seconds'] != PRIOR_SECONDS
            or prior_failure['slurm_step'] != '1465.116' or prior_failure['stage'] != 'failed_preserved_no_retry'
            or prior_failure['error_type'] != 'AttributeError'
            or prior_failure['error'] != "'tuple' object has no attribute 'detach'"
            or prior_failure['last_saved_progress']['counters'] != PREFIX):
        raise ValueError('Only the authorized native MOEL tuple probability capture repair may continue')
    if (prior_handoff['actual_successful_prefix_counters'] != PREFIX
            or prior_handoff['base_execution']['new_failed_OVRP_forward_rows'] != 16
            or prior_base['actual_cumulative_forward_rows'] != 30016
            or prior_base['actual_cumulative_forward_batches'] != 2377):
        raise ValueError('Actual failed partial work differs from the preserved handoff')
    completed = prefix_receipt['completed_tasks']
    completed['train/ATSP'] = json.loads((PRIOR_RECORD / 'feature_cache/train/ATSP/metadata.json').read_text())
    if (not prefix_receipt['zero_optimizer_updates'] or prior_handoff['successful_optimizer_updates'] != 0
            or set(completed) != {'train/TSP', 'train/CVRP', 'train/ATSP'}
            or any(not item['complete'] or item['rows'] != 10000 for item in completed.values())):
        raise ValueError('Preserved completed-prefix receipt differs')
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
    protected = prior_amendment['protected_original_text_sha256']
    partial_directory = ROOT / 'feature_cache/train/OVRP/n50_s50_p0'
    partial_metadata = json.loads((partial_directory / 'metadata.json').read_text())
    if (partial_metadata['complete'] or partial_metadata['rows'] != 197
            or partial_metadata['true_n'] != 50 or partial_metadata['starts'] != 50):
        raise ValueError('Preserved partial first OVRP group differs')
    partial_files = ['source_indices.npy', 'nodes.npy', 'features.npy', 'h.npy', 'base_logits.npy']
    partial_base = dict(directory=str(partial_directory), problem='OVRP', group=[50, 50, 0], rows=16,
        reuse_available=all((partial_directory / name).is_file() for name in partial_files),
        failed_base_forward_rows=16, failed_base_forward_batches=1)
    prior_probe_work = dict(evidence_type='Source/control-flow-derived; not instrumented prior-call measurements',
        failed_step='1465.116', problem='OVRP', rows=16, true_n=50, starts_per_row=50,
        probe_initialization_rows=32, start_trajectories=1600, encoder_forward_batches=2,
        returned_model_action_calls=14, attempted_model_action_calls=15,
        failed_model_action_calls=1, decoder_computation_batches=11, environment_step_batches=14,
        MTL=dict(initialization_rows=16, initialization_action_calls=2, acquisition_action_calls=10,
            model_action_calls=12, decoder_computation_batches=10, checkpoints_completed=2),
        MOEL=dict(initialization_rows=16, initialization_action_calls=2, returned_model_action_calls=2,
            failed_acquisition_action_attempts=1, decoder_computation_batches=1, checkpoints_completed=0),
        derivation='n50 gives increment5; MTL returned after two initialization and ten acquisition calls. MOEL returned two initialization calls; its first decoder returned a tuple before the outer model hook raised.',
        trace=str(PRIOR_RECORD / 'pipeline_status.json'),
        native_state_recoverable=False, authorized_replay_rows=16, original_partial_outputs_preserved=True)
    experiment.atomic_json(RECORD / 'prior_partial_probe_work.json', prior_probe_work)
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
            raise ValueError('Previous continuation execution snapshot identity changed: '+name)
        diffs[name] = dict(old_sha256=experiment.file_hash(old),
            new_sha256=identities[name], old_source=str(old),
            new_source=str(destination), unified_diff=''.join(difflib.unified_diff(
                before.splitlines(keepends=True), current.read_text().splitlines(keepends=True),
                fromfile=str(old), tofile=str(current))))
        shutil.copy2(current, destination)
    native = prior_amendment['original_native_sources']
    amendment = dict(utc=experiment.now(), authorization=str(ROOT / 'runtime_repair6_authorization.json'),
        reason='Capture native MOEL tuple element0 probabilities or the direct Tensor; preserve values, exclude moe_loss, and resume only remaining tasks with separately charged uncommitted-batch replay',
        omitted_framework_bookkeeping=['EnvBase.batch_size input/output CompositeSpec resizing',
            'EnvBase.reset modern spec-processing wrapper; original native _reset(None) is called directly'],
        original_step='1465.111', original_state='FAILED', original_exit_code='1:0',
        original_step_elapsed_seconds=1427, original_failures_retained=True, original_job_reclassified=False,
        prior_steps=PRIOR_STEPS, charged_prior_Slurm_wall_seconds=PRIOR_SECONDS,
        previous_amendment=str(PRIOR_RECORD / 'execution_amendment.json'), original_native_sources=native,
        exact_source_changes=diffs, original_protocol_sha256=protected[str(ROOT / 'protocol.json')],
        approved_plan_sha256=protected[str(ROOT / 'approved_plan.md')],
        unchanged=['scientific configuration', 'complete averaged-x R73B23/global63 base and buffers',
            'native policies/encoder/decoder/state transitions/configuration/actual frozen weights', 'directed matrices and original inputs',
            'private seed2 MATNET permutation per true n, independent of batches', 'native labels and legal candidates',
            'objective', 'seed2 global41-80 task/row order', 'seed2 fresh policy initialization',
            'fresh official optimizer and scaler', 'fixed 40 epochs/9600 updates', 'acquisition budgets'],
        initial_counters=PREFIX, successful_updates_before_continuation=0,
        completed_tasks=list(completed), next_task='train/OVRP', completed_tasks_skipped_before_native_inputs_or_probes=True,
        cache_resume_validation='Established TSP/CVRP metadata and exact complete continuation5 TRAIN/ATSP JSON reused; no repeated prefix audit, complete-cache array read or model replay',
        base_total_unique_inputs=198000, committed_base_inputs=30000,
        unique_base_inputs_already_forwarded=30016, unique_base_inputs_requiring_forward=167984,
        feature_rows_remaining_to_commit=168000,
        actual_prior_base_forward_rows=30016, actual_prior_base_forward_batches=2377,
        failed_partial_base_forward_rows=16, partial_base=partial_base,
        partial_output_use='Read only failed first16 OVRP base outputs during formal generation; validate alignment/shape/finite values; only an unavailable or invalid uncommitted base slice may be replayed and explicitly charged',
        prior_partial_probe_work=prior_probe_work,
        partial_probe_output_use='Preserve old MTL checkpoint arrays; replay only the uncommitted16-row OVRP native batch because live native state is unrecoverable. Prior derived work is added separately to actual execution totals.',
        new_OVRP_cache_directory=str(RECORD / 'feature_cache/train/OVRP'),
        protected_original_text_sha256=protected, protection_hashes_inherited_not_rechecked=True,
        established_protection_receipt=str(PRIOR_RECORD / 'execution_amendment.json'),
        established_partial_base_work_receipt=str(PRIOR_RECORD / 'base_execution_progress.json'),
        original_neural_cpu_check_repeated=False,
        syntax_check=syntax, automatic_retry=False, test_authorized=False, goal_status='ACTIVE; unachieved')
    experiment.atomic_json(RECORD / 'execution_amendment.json', amendment)
    experiment.atomic_json(RECORD / 'original_prefix.json', dict(counters=PREFIX, completed_tasks=completed,
        original_step_elapsed_seconds=1427, prior_steps=PRIOR_STEPS, charged_prior_Slurm_wall_seconds=PRIOR_SECONDS,
        original_status=failure, previous_continuation_status=prior_failure, zero_optimizer_updates=True))
    loaded = json.loads((PRIOR_RECORD / 'probe_initialization.json').read_text())
    if set(loaded) != {'BQ', 'RELD_CVRP', 'MATNET', 'RELD_MTL', 'RELD_MOEL'}:
        raise ValueError('Actual original frozen probe identities differ')
    base_execution = dict(planned_unique_inputs=198000, prior_completed_forward_rows=30000,
        prior_failed_forward_rows=16, prior_total_forward_rows=30016, prior_total_forward_batches=2377,
        new_forward_rows=0, new_forward_batches=0, reused_failed_output_rows=0, replayed_uncommitted_rows=0,
        actual_cumulative_forward_rows=30016, actual_cumulative_forward_batches=2377)
    experiment.atomic_json(RECORD / 'base_execution_progress.json', dict(utc=experiment.now(), **base_execution))
    return protocol, schedule, r48, dict(initial_counters=PREFIX, completed_tasks=completed, loaded_probes=loaded,
        original_elapsed_seconds=PRIOR_SECONDS, original_step_elapsed_seconds=1427, process_started=process_started,
        partial_base=partial_base, base_execution=base_execution, prior_partial_probe_work=prior_probe_work)


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
            continuation_record='continuation6', prior_steps=PRIOR_STEPS, charged_prior_Slurm_wall_seconds=PRIOR_SECONDS,
            slurm_step=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'], status='running')
        experiment.atomic_json(RECORD / 'trainer_launch.json', launch)
        import torch
        torch.set_num_threads(1)
        protocol, schedule, r48, continuation = prepare(started)
        experiment.atomic_json(RECORD / 'formal_started.json', dict(launch,
            original_step='1465.111', original_status='FAILED 1:0', automatic_retry=False,
            test_authorized=False, phases=['remaining_feature_generation', 'unchanged_fixed40E_policy']))
        import wandb
        wb = wandb.init(**protocol['wandb'], name='R74_B_bounded_acquisition_seed2_continuation6',
            config=protocol, dir=str(ROOT), id='r74-b-cont6-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S'))
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
        if continuation is not None:
            failure['base_execution'] = continuation['base_execution']
            failure['prior_partial_probe_work'] = continuation['prior_partial_probe_work']
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
