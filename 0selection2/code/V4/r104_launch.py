"""Explicit-ready, one-shot R104 dispatch using private historical adapters.

Importing this module starts nothing. Main owns readiness and actual queue/start.
Original GPU mutexes, allocation1465, gpu03 container and node mutex are reused.
"""

import argparse
import copy
import importlib.util
import json
from pathlib import Path
import sys
import time


REPO = Path(__file__).resolve().parents[2]
ROOT = REPO / 'code/V4/runs/R98_parallel_dispatch'
MODEL_ROOT = REPO / 'code/V4/runs/R104_capacity_scaling'
WRAPPER = REPO / 'code/V4/run_v4_r104_launch.sh'
TOKEN_ENV = 'R104_DISPATCH_TOKEN'
ARMS = ('A_width256', 'B_depth8')
ARM_SLOTS = dict(zip(ARMS, ('3090_0', '3090_1')))
PREDECESSORS = dict(zip(ARMS, ('A', 'B')))
SLOT_CPUS = {'3090_0': [12, 13, 14, 15], '3090_1': [16, 17, 64, 65]}
DATA_ROOT = Path('/public/home/shiys/selection_data_new90k_B_filtered_single_reld_mt_seed2_20261018')
PREPARED_ROOT = REPO / 'code/V4/runs/R84_new90k_attention/prepared'
CACHE_SOURCE = REPO / 'code/V4/runs/R84_new90k_attention/input_cache'
CACHE_ROOT = Path('/tmp/shiys_R84_new90k_cache')
STAGES = ('train', 'TEST', 'timing', 'LIB', 'PDF')
SOURCE_NAMES = ('r104_launch.py', 'r104_wave.py', 'r104_goal.py',
    'r104_model.py', 'r104_experiment.py', 'r104_reporting.py',
    'test_r104_launch.py', 'test_r104_goal.py', 'test_r104_model.py',
    'test_r104_experiment.py', 'test_r104_reporting.py',
    'r98_launch.py', 'r98_goal.py', 'r84_launch.py', 'r81_queue.py',
    'run_v4_r84_launch.sh', 'run_v4_r81.sh')
PUBLICATION_EXTRAS = ('docs/research/capacity_scaling_for_solver_discrimination.md',
                      'code/V4/bind_cached_val_checkpoint.py')


def private(name, filename):
    spec = importlib.util.spec_from_file_location('code.V4.' + name, REPO / 'code/V4' / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


adapter = private('_r104_dispatch', 'r98_launch.py')
predecessor_goal = private('_r104_parent_admin', 'r98_goal.py')
base = adapter.base
SLOTS = {slot: dict(adapter.SLOTS[slot], cap=128) for slot in ARM_SLOTS.values()}
original_start, original_node_resources = adapter.start, base.node_resources
original_check_ready = base.check_ready


def run_id(arm):
    if arm not in ARMS:
        raise ValueError('Choose A_width256 or B_depth8')
    return 'R104_' + arm


def arm_of(spec):
    for arm in ARMS:
        if spec.get('run_id') == run_id(arm):
            return arm
    raise ValueError('R104 owns only its two exact capacity arms')


def self_command(stage, spec=None, slot=None, token=None):
    argv = [sys.executable, '-B', '-u', '-m', 'code.V4.r104_launch', stage]
    for key, value in (('--spec', spec), ('--slot', slot), ('--token', token)):
        if value is not None:
            argv.extend((key, str(value)))
    return argv


def locked_protocol(path, spec):
    path = Path(path)
    if path != MODEL_ROOT / 'protocol_lock.json' or not path.is_file():
        raise ValueError('Wait for main R104 protocol_lock.json; no guessed readiness')
    science = base.read(path)
    required = dict(experiment='R104', locked=True, arms=list(ARMS), seed=2,
        task_scope='full18', data_root=spec['data_root'], prepared_root=spec['prepared_root'],
        rows_per_problem=dict(train=81000, val=4500, test=4500), logical_global_batch=1024,
        max_updates=57600, warmup_updates=4320, eval_every=1440, train_eval_every=7200,
        large_micro_cap=128, other_micro_cap=512, eval_batch_size=128,
        current_query_labels_in_forward=False, external_pretraining=False)
    if not isinstance(science, dict) or any(science.get(k) != v for k, v in required.items()):
        raise ValueError('Require main-locked R104 full18 NEW90k seed2 logical1024/57600 and fixed128/512/eval128')
    return science


def expected_stages(arm):
    run_id(arm)
    result = []
    for name, command in zip(STAGES, ('train', 'test', 'timing', 'lib', 'report')):
        reporter = name in ('LIB', 'PDF')
        options = ['--root', str(MODEL_ROOT), '--arm', arm]
        if not reporter:
            options += ['--prepared-root', str(PREPARED_ROOT), '--data-root', str(DATA_ROOT),
                '--cache-root', '{cache_root}', '--protocol-lock', str(MODEL_ROOT / 'protocol_lock.json'),
                '--gpu-type', '3090', '--large-micro-cap', '128', '--eval-batch-size', '128',
                '--wandb-mode', 'offline']
        options += ['--device', '{device}']
        if name == 'LIB':
            options += ['--lib-batch-size', '2']
        result.append(dict(name=name, device='cpu' if name == 'PDF' else 'gpu',
            argv=['{python}', '-B', '-u', '-m',
                'code.V4.r104_reporting' if reporter else 'code.V4.r104_experiment', command, *options]))
    return result


def load_spec(path):
    spec = adapter.original_load_spec(path)
    arm = arm_of(spec)
    expected = dict(run_dir=str(MODEL_ROOT / arm), common_root=str(MODEL_ROOT),
        science_protocol=str(MODEL_ROOT / 'protocol_lock.json'), data_root=str(DATA_ROOT),
        prepared_root=str(PREPARED_ROOT), cache_source=str(CACHE_SOURCE), cache_root=str(CACHE_ROOT),
        cache_kind='r84_inputs', wandb_mode='offline', slots=[ARM_SLOTS[arm]],
        cpu_count=4, memory_gib=9, cpus=SLOT_CPUS[ARM_SLOTS[arm]], stages=expected_stages(arm))
    forbidden = ('feature_root', 'routing_feature_root', 'routing_stage_argv', 'routing_stage_json',
        'teacher_root', 'context_root', 'preprocessing', 'feature_stage')
    if any(spec.get(k) != v for k, v in expected.items()) or any(k in spec for k in forbidden):
        raise ValueError('Use exact R104 train/TEST/timing/LIB2/CPU-PDF, fixed gpu03 CPU quartets and direct R84 reuse')
    locked_protocol(spec['science_protocol'], spec)
    return spec


def declared_sources():
    return [REPO / 'code/V4' / name for name in SOURCE_NAMES] + [WRAPPER,
        MODEL_ROOT / 'protocol_lock.json', MODEL_ROOT / 'PLAN.md',
        MODEL_ROOT / 'preparation/LAUNCH_PLAN.md'] + [PREPARED_ROOT / name for name in
        ('prepared.json', 'geometry_stats.json', 'source_cache_manifest.json', 'identicalinit_seed2.pt')]


def source_files(spec):
    # Only named entrypoints and their real static imports, never r104*.py/test globs.
    sources = [*declared_sources(), *spec.get('source_files', [])]
    if any(not Path(p).is_file() for p in sources):
        raise RuntimeError('Wait for final main/model/runner/report/launcher sources before readiness')
    tests = {str(Path(p).resolve()) for p in sources if Path(p).name.startswith('test_')}
    runtime = [p for p in sources if str(Path(p).resolve()) not in tests]
    return sorted(set(adapter.source_closure(runtime)) | tests)


def explicit_ready(evidence, tested):
    if not isinstance(evidence, str) or not evidence.strip() or tested is not True:
        raise ValueError('Explicit main-ready evidence and passed CPU tests are required')


def check_ready(path):
    spec, plan = original_check_ready(path)
    ready = base.read(base.receipts(spec) / 'ready.json')
    if (plan.get('main_thread_ready') is not True or plan.get('evidence') != ready.get('evidence')
            or ready.get('source_freeze_ready') is not True
            or plan.get('primary_cache_reuse_only') is not True
            or plan.get('cache_copy') is not False or plan.get('cache_preprocessing') is not False
            or plan.get('automatic_retry') is not False
            or not plan.get('source_sha256')
            or set(plan['source_sha256']) != set(source_files(spec))):
        raise RuntimeError('Explicit main-ready exact source set required; no automatic refreeze')
    return spec, plan


def prepare(path, evidence=None, tested=False):
    explicit_ready(evidence, tested)
    spec = load_spec(path)
    sources = source_files(spec)
    result = adapter.prepare(path, evidence, tested)
    output = base.receipts(spec)
    plan = base.read(output / 'plan.json')
    plan['source_sha256'] = {name: base.sha(name) for name in sources}
    plan.update(primary_cache_reuse_only=True, cache_preprocessing=False, cache_copy=False,
        automatic_retry=False, predecessor='R99_' + PREDECESSORS[arm_of(spec)])
    base.save(output / 'plan.json', plan)
    ready = base.read(output / 'ready.json')
    ready.update(plan_sha256=base.sha(output / 'plan.json'), source_freeze_ready=True)
    base.save(output / 'ready.json', ready)
    check_ready(path)
    return dict(result, source_freeze_ready=True, cache_copied=False)


def stage_cache(spec, manifest):
    root = Path(spec['cache_root'])
    stamp = root / '.r84_cache_identity.json'
    with base.lock(root.with_name(root.name + '.r84_stage.lock'), blocking=True):
        if not stamp.is_file() or base.read(stamp).get('identity_sha256') != manifest['identity_sha256']:
            raise RuntimeError('Original R84 local cache required; no preprocessing, rebuild or copy')
        for row in manifest['files']:
            path = root / row['path']
            if (not path.is_file() or path.is_symlink() or path.stat().st_size != row['bytes']
                    or ('sha256' in row and base.sha(path) != row['sha256'])):
                raise RuntimeError('Original R84 cache incomplete; no repair or copy')
    return dict(root=str(root), reused=True, identity_sha256=manifest['identity_sha256'], copied=False)


def recovery_file(item, *, expected=None, within=None, pdf=False):
    if not isinstance(item, dict) or not isinstance(item.get('path'), str) or not item.get('sha256'):
        raise ValueError('Recovery artifact requires explicit path and SHA256')
    path = Path(item['path'])
    if (not path.is_absolute() or not path.is_file()
            or (expected is not None and path != expected)
            or (within is not None and not path.resolve().is_relative_to(within.resolve()))
            or (within is not None and path.suffix not in ('.json', '.pdf'))
            or base.sha(path) != item['sha256']):
        raise ValueError('Missing, foreign or changed recovery artifact: ' + str(path))
    if pdf and not predecessor_goal.pdf_ready(path):
        raise ValueError('Invalid recovery PDF: ' + str(path))
    return path


def recovered_predecessor(model, prior_arm, output, checks):
    """Read main's per-arm recovery v1; never execute recovery or bind metadata."""
    run = model / prior_arm
    root = model / 'preparation/postprocess_recovery'
    path = root / prior_arm / 'recovery_complete.json'
    if not path.exists():
        return None
    record = base.read(path)
    lock = base.read(run / 'final_test_lock.json')
    required = dict(schema='r99_postprocess_recovery_v1', experiment='R99', arm=prior_arm,
        complete=True, exit_status=0, state='complete', checkpoint_lock=lock,
        checkpoint_sha256=lock.get('checkpoint_sha256'), state_hash=lock.get('state_hash'),
        original_failure_receipts_preserved=True, original_worker_state='failed',
        original_controller_state='failed', training_rerun=False, full_test_rerun=False,
        feature_copy=False, feature_rebuild=False, decision_rule_changed=False,
        efficacy_reused_as_exact_online=False,
        stages_completed=['original_locked_full_TEST', 'recovery_timing', 'original_reporter_LIB',
            'private_CPU_full_report', 'metadata_only_VAL_checkpoint_binding'])
    if (any(record.get(k) != v for k, v in required.items()) or record.get('error') or record.get('errors')
            or not all(checks.get(k) for k in ('protocol', 'training_complete', 'unique_VAL_lock',
                                             'full_VAL', 'full_TEST', 'assigned_predecessor', 'exact_pipeline'))):
        raise ValueError('Recovery needs the exact main v1 completion and original locked TRAIN/VAL/TEST')
    worker, controller_record = base.read(output / 'worker.json'), base.read(output / 'controller.json')
    stages = worker.get('stages', [])
    if (worker.get('state') != 'failed' or worker.get('exit_status') != 1
            or [s.get('name') for s in stages] != ['input_features', 'train', 'TEST', 'timing']
            or not all(predecessor_goal.rc0(s) for s in stages[:3])
            or stages[-1].get('state') != 'failed' or stages[-1].get('exit_status') != 1
            or controller_record.get('state') != 'failed' or controller_record.get('exit_status') != 1
            or [s.get('name') for s in controller_record.get('stages', [])] != ['allocation_GPU_pipeline']
            or controller_record['stages'][0].get('state') != 'failed'
            or controller_record['stages'][0].get('exit_status') != 1):
        raise ValueError('Only original rc0 training/TEST followed by timing-only failure can be recovered')
    failures = record.get('original_failures', {})
    names = {'worker.json', 'controller.json', 'timing.log', 'allocation_GPU_pipeline.log'}
    if set(failures) != names or predecessor_goal.json_hash(failures) != record.get('original_failure_sha256'):
        raise ValueError('Original failure evidence set/digest changed')
    for name in names:
        recovery_file(failures[name], expected=output / name)
    report_path = recovery_file(record.get('report_receipt'),
        expected=root / prior_arm / 'report_stage.json', within=root)
    binding_path = recovery_file(record.get('val_binding'),
        expected=root / prior_arm / 'bind_val_stage.json', within=root)
    report, binding = base.read(report_path), base.read(binding_path)
    for stage, value in (('report', report), ('bind_val', binding)):
        if (value.get('schema') != required['schema'] or value.get('experiment') != 'R99'
                or value.get('arm') != prior_arm or value.get('stage') != stage
                or value.get('complete') is not True or value.get('exit_status') != 0
                or value.get('original_failure_sha256') != record['original_failure_sha256']
                or value.get('error') or value.get('errors')):
            raise ValueError('Recovery report/VAL-binding is incomplete or foreign')
    if (report.get('lock') != lock or report.get('pdfs') != record.get('pdfs')
            or binding.get('checkpoint_sha256') != lock.get('checkpoint_sha256')
            or binding.get('state_hash') != lock.get('state_hash')
            or binding.get('selected_update') != lock.get('selected_update')
            or binding.get('metadata_only') is not True
            or binding.get('after_sha256') != base.sha(run / 'final_metrics.json')
            or binding.get('changed_field') != 'val.identity.checkpoint_sha256'
            or binding.get('old_value', 'missing') is not None
            or binding.get('new_value') != lock.get('checkpoint_sha256')
            or any(binding.get(k) is not False for k in ('numeric_values_changed', 'prediction_arrays_changed',
                                                       'weights_changed', 'original_failure_receipts_changed'))):
        raise ValueError('Explicit same-lock metadata-only VAL binding required; no null-SHA bypass')
    backup = base.read(recovery_file(binding.get('backup'), within=root))
    archive_path = recovery_file(binding.get('original_archived_val_identity'),
        expected=run / 'evaluation/val/selected_best/identity.json')
    archive = base.read(archive_path)
    final = base.read(run / 'final_metrics.json')
    unbound = copy.deepcopy(final)
    unbound['val']['identity']['checkpoint_sha256'] = None
    if (unbound != backup or backup['val']['identity'] != archive
            or archive.get('checkpoint_sha256', 'missing') is not None
            or binding.get('full_report_receipt') != record.get('report_receipt')
            or not binding.get('model_state_sha256')
            or binding['model_state_sha256'] != archive.get('model_state_sha256')
            or archive.get('state_hash') != lock.get('state_hash')
            or archive.get('successful_updates') != lock.get('selected_update')):
        raise ValueError('VAL binding changed numeric/other JSON fields or original selected archive identity')
    protocol_path = recovery_file(report.get('report_protocol'), within=root)
    protocol = base.read(protocol_path)
    if protocol.get('complete') is not True or protocol.get('checkpoint_lock') != lock:
        raise ValueError('Recovered CPU PDF protocol belongs to another checkpoint')
    pdfs = record.get('pdfs', {})
    if set(pdfs) != set(predecessor_goal.R99_PDFS):
        raise ValueError('Recovery must supply all five full-arm PDFs')
    for item in pdfs.values():
        recovery_file(item, within=root, pdf=True)
    timing = record.get('timing', {})
    timing_path = recovery_file(timing, within=root)
    timing_record = base.read(timing_path)
    if (timing.get('status') != 'complete_private_numerical_revision'
            or timing.get('recovery_schema') != required['schema']
            or timing.get('efficacy_reused_as_exact_online') is not False
            or timing.get('original_inference_equivalence') != record.get('original_inference_equivalence')
            or record.get('original_inference_equivalence') is not False
            or not timing.get('numerical_disclosure')
            or timing.get('timing_path') != 'mixed' or timing.get('requested_tf32_policy') != 'default'
            or timing_record.get('complete') is not True or timing_record.get('exit_status') != 0
            or timing_record.get('checkpoint_sha256') != lock.get('checkpoint_sha256')
            or timing_record.get('state_hash') != lock.get('state_hash')
            or timing_record.get('inference_rule') != lock.get('inference_rule')
            or timing_record.get('original_inference_equivalence') != record['original_inference_equivalence']
            or timing_record.get('efficacy_reused_as_exact_online') is not False
            or timing_record.get('timing_path') != 'mixed' or timing_record.get('requested_tf32_policy') != 'default'
            or timing_record.get('recovery_schema') != required['schema']
            or timing_record.get('repetitions') != 3 or timing_record.get('warmup_batches') != 1
            or timing.get('transform_plus_selector_measured') is not True or timing.get('pdf_panel') is not True):
        raise ValueError('Same-lock disclosed recovery timing is required, not claimed exact cached latency')
    if protocol.get('inference_timing', {}).get('sha256') != timing['sha256']:
        raise ValueError('Recovered PDF protocol does not bind the official timing file')
    for rows in (timing_record.get('per_problem', []), timing.get('measurements', [])):
        if (len(rows) != 18 or {r.get('problem') for r in rows} != set(predecessor_goal.PROBLEMS)
                or any(r.get('instances') != 128 for r in rows)):
            raise ValueError('Recovery timing must cover all18 first128-input populations')
    lib = record.get('lib', {})
    manual = base.read(recovery_file(lib.get('manual_receipt'),
        expected=model / 'preparation/manual_stages' / (prior_arm + '_lib.json')))
    complete = base.read(recovery_file(lib.get('completion'), expected=run / 'lib_stage_complete.json'))
    if (manual.get('stage') != 'lib' or manual.get('arm') != prior_arm
            or not predecessor_goal.rc0(manual) or manual.get('checkpoint_lock') != lock
            or manual.get('original_worker_sha256') != failures['worker.json']['sha256']
            or manual.get('original_controller_sha256') != failures['controller.json']['sha256']
            or complete.get('complete') is not True or complete.get('exit_status') != 0
            or complete.get('checkpoint_sha256') != lock.get('checkpoint_sha256')
            or complete.get('state_hash') != lock.get('state_hash')
            or set(complete.get('datasets', [])) != {'TSPLIB', 'CVRPLIB'}):
        raise ValueError('Main rc0 original-reporter LIB and original failure linkage required')
    return dict(path=str(path), sha256=base.sha(path), schema=required['schema'], complete=True,
        original_failure_receipts_preserved=True, original_inference_equivalence=record['original_inference_equivalence'],
        efficacy_reused_as_exact_online=False)


def predecessor_gate(arm):
    run_id(arm)
    prior_arm = PREDECESSORS[arm]
    model = REPO / 'code/V4/runs/R99_global_competition_input'
    run, parent_id = model / prior_arm, 'R99_' + prior_arm
    output = ROOT / parent_id
    current = predecessor_goal.status('R99', model, ROOT, persist=False)
    row = next(r for r in current['arms'] if r['arm'] == prior_arm)
    checks = dict(row['checks'])
    spec = predecessor_goal.optional(ROOT / 'specs' / (parent_id + '.json'))
    checks['assigned_predecessor'] = (spec.get('run_id') == parent_id
        and spec.get('run_dir') == str(run) and spec.get('slots') == [ARM_SLOTS[arm]])
    checks['exact_pipeline'] = [(s.get('name'), s.get('device')) for s in spec.get('stages', [])] == [
        (name, 'cpu' if name == 'PDF' else 'gpu') for name in
        ('input_features', 'train', 'TEST', 'timing', 'LIB', 'PDF')]
    errors = []
    raw = current.get('errors', [])
    for error in raw if isinstance(raw, list) else [raw]:
        path = error.get('path') if isinstance(error, dict) else None
        if path is None or Path(path).is_relative_to(run) or Path(path).is_relative_to(output):
            errors.append(error)
    # Never late-bind null summary hashes here. Main alone may do so after rc0.
    recovered = None
    if errors or not all(checks.values()):
        try:
            recovered = recovered_predecessor(model, prior_arm, output, checks)
        except (OSError, AttributeError, KeyError, TypeError, ValueError) as error:
            errors.append(dict(path=str(model / 'preparation/postprocess_recovery' / prior_arm / 'recovery_complete.json'),
                               error=str(error)))
    original_errors = list(errors)
    if recovered is not None:
        allowed = {str(output / name) for name in ('worker.json', 'controller.json')}
        errors = [e for e in errors if not isinstance(e, dict) or e.get('path') not in allowed]
        checks = {k: checks[k] for k in ('protocol', 'training_complete', 'unique_VAL_lock', 'full_VAL',
                                        'full_TEST', 'assigned_predecessor', 'exact_pipeline')}
        checks['explicit_main_postprocess_recovery'] = True
    return dict(arm=arm, predecessor=parent_id, complete=not errors and all(checks.values()),
        checks=checks, errors=errors, preserved_original_errors=original_errors if recovered else [],
        recovery=recovered, healthy_runs_stopped=False, automatic_retry=False)


def start(path, evidence=None, tested=False):
    explicit_ready(evidence, tested)
    spec, plan = check_ready(path)
    if plan.get('evidence') != evidence:
        raise ValueError('Start evidence must match main-ready plan')
    gate = predecessor_gate(arm_of(spec))
    if gate['errors'] or not gate['complete']:
        raise RuntimeError('Wait for assigned R99 full train/TEST/timing/LIB/PDF rc0 and full-VAL metadata; no bypass')
    return original_start(path, evidence, tested)


def queue(path, evidence=None, tested=False, interval=60):
    explicit_ready(evidence, tested)
    if interval <= 0:
        raise ValueError('Queue interval must be positive')
    spec, plan = check_ready(path)
    if plan.get('evidence') != evidence:
        raise ValueError('Queue evidence must match main-ready plan')
    output = base.receipts(spec)
    with base.lock(output / 'queue.lock'):
        receipt_path = output / 'queue_receipt.json'
        if receipt_path.exists():
            raise RuntimeError('Queue already attempted; no replacement or retry')
        receipt = dict(state='waiting_predecessor', cpu_only=True, automatic_retry=False,
            started_utc=base.utc(), main_ready_evidence=evidence)
        base.save(receipt_path, receipt, exclusive=True)
        try:
            while True:
                check_ready(path)
                gate = predecessor_gate(arm_of(spec))
                if gate['errors']:
                    raise RuntimeError('Assigned R99 predecessor failed: ' + json.dumps(gate['errors']))
                if gate['complete']:
                    result = start(path, evidence, tested)
                    receipt.update(state='complete', exit_status=0, completed_utc=base.utc(), start=result)
                    base.save(receipt_path, receipt)
                    return result
                time.sleep(interval)
        except BaseException as error:
            receipt.update(state='failed', exit_status=1, error=str(error), completed_utc=base.utc())
            base.save(receipt_path, receipt)
            raise


def controller(path):
    spec, _ = check_ready(path)
    gate = predecessor_gate(arm_of(spec))
    if gate['errors'] or not gate['complete']:
        raise RuntimeError('Controller cannot bypass its R99 completion gate')
    return base.controller(path)


def node_resources(slot):
    probe = original_node_resources(slot)
    cpus = SLOT_CPUS[slot]
    if not set(cpus) <= set(probe['allowed_cpus']):
        raise RuntimeError('Original gpu03 CPU quartet is outside allocation/container affinity')
    return dict(probe, allowed_cpus=list(cpus))


# Rebind private instances only; original GPU and node lock roots stay intact.
adapter.SLOTS = base.SLOTS = SLOTS
adapter.ROOT = base.ROOT = ROOT
adapter.TOKEN_ENV = base.TOKEN_ENV = TOKEN_ENV
adapter.WRAPPER = WRAPPER
adapter.self_command = base.self_command = self_command
adapter.load_spec = base.load_spec = load_spec
base.check_ready = check_ready
base.container_command, base.environment = adapter.container_command, adapter.environment
base.reserved_cpus, base.stage_cache = adapter.reserved_cpus, stage_cache
base.node_resources = node_resources
base.session_name = lambda spec: 'goal_' + spec['run_id'].lower()
environment, container_command = adapter.environment, adapter.container_command


def parser():
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument('stage', choices=('prepare', 'queue', 'start', 'predecessor', 'controller',
        'node-resources', 'container-resources', 'entry', 'worker'))
    value.add_argument('--spec', type=Path)
    value.add_argument('--arm', choices=ARMS)
    value.add_argument('--slot', choices=tuple(SLOTS))
    value.add_argument('--token')
    value.add_argument('--evidence')
    value.add_argument('--runner-tests-passed', action='store_true')
    value.add_argument('--interval', type=int, default=60)
    return value


def main(argv=None):
    args = parser().parse_args(argv)
    if args.stage in ('prepare', 'start'):
        result = globals()[args.stage](args.spec, args.evidence, args.runner_tests_passed)
    elif args.stage == 'queue':
        result = queue(args.spec, args.evidence, args.runner_tests_passed, args.interval)
    elif args.stage == 'predecessor':
        result = predecessor_gate(args.arm or arm_of(load_spec(args.spec)))
    elif args.stage == 'controller':
        return controller(args.spec)
    elif args.stage in ('entry', 'worker'):
        return getattr(base, args.stage)(args.spec, args.slot, args.token)
    else:
        result = getattr(base, args.stage.replace('-', '_'))(args.slot)
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
