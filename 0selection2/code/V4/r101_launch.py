"""Explicit-ready R101 A/B dispatch in the existing shared allocations.

Only private module copies are rebound. ROOT/node CPU locks and the old R81
GPU lock inodes remain shared with R99/R100. No GPU operation runs on import.
Main supplies the scientific protocol and all six exact stage argument lists.
"""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time

from .r98_goal import PROBLEMS, optional, pdf_ready, receipt_errors, stages_ready


REPO = Path(__file__).resolve().parents[2]
ROOT = REPO / 'code/V4/runs/R98_parallel_dispatch'
MODEL_ROOT = REPO / 'code/V4/runs/R101_hint_pretraining'
PROTOCOL_LOCK = MODEL_ROOT / 'protocol_lock.json'
WRAPPER = REPO / 'code/V4/run_v4_r101_launch.sh'
TOKEN_ENV = 'R101_DISPATCH_TOKEN'
CACHE_ROOT = Path('/tmp/shiys_R84_new90k_cache')
ARM_SLOTS = dict(A='4090_1', B='4090_0')
TEACHER_ROOTS = dict(A=Path('/tmp/shiys_R101_A_teacher'), B=Path('/tmp/shiys_R94_B_node_features'))
TEACHER_TASKS = tuple(p for p in PROBLEMS if p not in ('TSP', 'ATSP'))
STAGES = ('pretrain', 'train', 'TEST', 'timing', 'LIB', 'PDF')


def private(name, filename):
    spec = importlib.util.spec_from_file_location('code.V4.' + name, REPO / 'code/V4' / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


adapter = private('_r101_dispatch', 'r98_launch.py')
base = adapter.base
SLOTS = {slot: dict(adapter.SLOTS[slot]) for slot in ARM_SLOTS.values()}
original_load_spec = adapter.original_load_spec
original_start = adapter.start


def self_command(stage, spec=None, slot=None, token=None):
    argv = [sys.executable, '-B', '-u', '-m', 'code.V4.r101_launch', stage]
    for key, value in (('--spec', spec), ('--slot', slot), ('--token', token)):
        if value is not None:
            argv.extend((key, str(value)))
    return argv


def arm_of(spec):
    if spec.get('run_id') not in ('R101_A', 'R101_B'):
        raise ValueError('R101 has exactly R101_A and R101_B')
    return spec['run_id'][-1]


def locked_protocol(path, spec):
    path = Path(path)
    if not path.is_absolute() or not path.is_file():
        raise ValueError('Main must supply the locked R101 scientific protocol before preparation')
    science = base.read(path)
    if (science.get('experiment') != 'R101' or science.get('locked') is not True
            or science.get('seed') != 2 or science.get('task_scope') != 'full18'
            or science.get('arms') != ['A', 'B'] or science.get('data_root') != spec['data_root']
            or science.get('current_query_labels_in_forward', False) is not False
            or science.get('rows_per_problem') != dict(train=81000, val=4500, test=4500)):
        raise ValueError('Require the explicit R101 seed2 full18 frozen-data A/B protocol')
    logical = science.get('logical_global_batch', science.get('global_batch_size', science.get('batch_size')))
    stage1 = science.get('pretrain_updates', science.get('hint_pretraining', {}).get('updates',
        science.get('pretraining', {}).get('max_successful_updates')))
    stage2 = science.get('max_updates', science.get('max_successful_updates'))
    if (logical != 1024 or stage1 != 12800 or stage2 != 57600):
        raise ValueError('Require separate Stage-I12800 / Stage-II57600 budgets and logical batch1024')
    return science


def _module(argv):
    if '-m' not in argv:
        raise ValueError('Every R101 stage must expose an exact Python module CLI')
    index = argv.index('-m')
    module = argv[index + 1]
    if not module.startswith('code.V4.r101_'):
        raise ValueError('Only the main-supplied R101 runner/pretrainer/reporter may execute')
    return module, argv[index + 2:]


def load_spec(path):
    # Bypass r98.load_spec's deliberately frozen R98/R99/R100 prefix gate.
    spec = original_load_spec(path)
    arm = arm_of(spec)
    if (spec['slots'] != [ARM_SLOTS[arm]] or spec.get('wandb_mode') != 'offline'
            or spec.get('cache_kind') != 'r84_inputs' or Path(spec['cache_root']) != CACHE_ROOT
            or Path(spec['run_dir']) != MODEL_ROOT / arm
            or Path(spec.get('common_root', str(MODEL_ROOT))) != MODEL_ROOT
            or Path(spec.get('teacher_root', '')) != TEACHER_ROOTS[arm]):
        raise ValueError('Use fixed R101 arm/4090 bindings and the existing primary/teacher cache roots')
    locked_protocol(spec.get('science_protocol', ''), spec)
    stages = spec['stages']
    if tuple(stage['name'] for stage in stages) != STAGES:
        raise ValueError('Exact stage order: pretrain, train, TEST, timing, LIB, PDF')
    for stage in stages:
        module, arguments = _module(stage['argv'])
        expected_device = 'cpu' if stage['name'] == 'PDF' else 'gpu'
        if stage['device'] != expected_device:
            raise ValueError('Only final PDF runs on CPU; the five preceding stages use the assigned GPU')
        if stage['name'] not in ('pretrain', 'train') and any(arg.startswith('--teacher') for arg in arguments):
            raise ValueError('Teacher features are Stage-I-only, never a final selector forward dependency')
        if stage['name'] in ('LIB', 'PDF'):
            command = 'lib' if stage['name'] == 'LIB' else 'report'
            if module != 'code.V4.r101_reporting' or not arguments or arguments[0] != command:
                raise ValueError('Use the R101 native LIB / CPU PDF reporter')
            if stage['name'] == 'LIB':
                if '--lib-batch-size' not in arguments or arguments[arguments.index('--lib-batch-size')+1] != '2':
                    raise ValueError('Native LIB must use --lib-batch-size 2')
    if arm == 'A' and Path(spec.get('teacher_source', '')) != MODEL_ROOT / 'teacher_cache/A':
        raise ValueError('A stages only the shared, already exported R94_A TRAIN features')
    if arm == 'A':
        spec.setdefault('teacher_receipt', str(MODEL_ROOT / 'preparation/A_teacher_export_receipt.json'))
    return spec


def teacher_manifest(root, arm, expected=None):
    """32 file stats and 16 small receipts; no NPY reads or teacher extraction."""
    root, files, hashes = Path(root), [], {}
    for problem in TEACHER_TASKS:
        path = root / 'train' / problem / 'metadata.json'
        record = base.read(path)
        shape = record.get('shape', [])
        if (record.get('complete') is not True or record.get('input_only') is not True
                or record.get('problem') != problem or record.get('split') != 'train'
                or record.get('rows') != 81000 or record.get('rows_written') != 81000
                or record.get('dtype') != 'float16' or record.get('file') != 'nco_nodes.npy'
                or len(shape) != 3 or shape[0] != 81000 or shape[2] != 128
                or record.get('donor', {}).get('variant') != {'A': 'random', 'B': 'pretrained'}[arm]):
            raise ValueError('Require complete input-only R94 native16 TRAIN teacher metadata: ' + problem)
        hashes[problem] = base.sha(path)
        for name in ('metadata.json', 'nco_nodes.npy'):
            file = path.with_name(name)
            if file.is_symlink() or any(parent.is_symlink() for parent in file.parents) or not file.is_file():
                raise ValueError('Teacher cache must contain ordinary complete files, never symlinks')
            row = dict(path=str(file.relative_to(root)), bytes=file.stat().st_size)
            if name == 'metadata.json':
                row['sha256'] = hashes[problem]
            files.append(row)
    if expected is not None and hashes != expected:
        raise ValueError('Teacher TRAIN metadata differs from the main/bridge frozen receipt')
    raw = json.dumps(files, sort_keys=True, separators=(',', ':')).encode()
    return dict(files=files, bytes=sum(row['bytes'] for row in files), metadata_sha256=hashes,
                identity_sha256=hashlib.sha256(raw).hexdigest())


def prepare_teacher(spec):
    arm = arm_of(spec)
    if arm == 'A':
        return dict(teacher_manifest(spec['teacher_source'], arm), mode='stage_exported_TRAIN_only',
                    source=spec['teacher_source'], root=spec['teacher_root'])
    if not spec.get('teacher_receipt'):
        return dict(mode='reuse_existing_TRAIN_only', root=spec['teacher_root'],
                    validate_existing_native16_on_worker=True)
    path = Path(spec['teacher_receipt'])
    receipt = base.read(path)
    hashes = receipt.get('metadata_sha256', {}) or {p: row.get('teacher_metadata_sha256')
        for p, row in receipt.get('tasks', {}).items()}
    if (receipt.get('complete') is not True or receipt.get('arm') != 'B'
            or set(hashes) != set(TEACHER_TASKS) or any(not isinstance(v, str) or len(v) != 64 for v in hashes.values())):
        raise ValueError('B teacher_receipt needs complete=true, arm=B and native16 metadata_sha256')
    return dict(mode='reuse_existing_TRAIN_only', root=spec['teacher_root'], metadata_sha256=hashes,
                receipt=str(path.resolve()), receipt_sha256=base.sha(path))


def prepare(path, evidence=None, tested=False):
    if not isinstance(evidence, str) or not evidence.strip() or tested is not True:
        raise ValueError('Explicit main-ready evidence and passed CPU parser/runner tests are required')
    spec = load_spec(path)
    teacher = prepare_teacher(spec)
    result = adapter.prepare(path, evidence, tested)
    output = base.receipts(spec)
    plan = base.read(output / 'plan.json')
    sources = [Path(__file__), WRAPPER, Path(adapter.__file__), Path(base.__file__),
               Path(base.transport.__file__), Path(spec['science_protocol'])]
    sources += [REPO.joinpath(*_module(stage['argv'])[0].split('.')).with_suffix('.py') for stage in spec['stages']]
    if spec.get('teacher_receipt'):
        sources.append(Path(spec['teacher_receipt']))
    plan['source_sha256'].update({name: base.sha(name) for name in adapter.source_closure(sources)})
    plan['teacher_cache'] = teacher
    base.save(output / 'plan.json', plan)
    ready = base.read(output / 'ready.json')
    ready['plan_sha256'] = base.sha(output / 'plan.json')
    base.save(output / 'ready.json', ready)
    base.check_ready(path)
    return dict(result, teacher_mode=teacher['mode'])


def stage_teacher(spec, plan):
    from .r101_features import stage_teacher as implementation
    arm, root = arm_of(spec), Path(spec['teacher_root'])
    receipt_path = base.receipts(spec) / 'teacher_stage.json'
    if receipt_path.exists():
        raise RuntimeError('Teacher staging already attempted; preserve its receipt, no automatic retry')
    if arm == 'B':
        if plan.get('receipt') and base.sha(plan['receipt']) != plan['receipt_sha256']:
            raise ValueError('Frozen B teacher receipt changed')
        # B discovery is metadata-only, not a bridge source-copy receipt.
        result = implementation(root, variant='B', cache_root=spec['cache_root'], source_receipt=None)
        if plan.get('metadata_sha256') and {p: row.get('teacher_metadata_sha256')
                for p, row in result.get('tasks', {}).items()} != plan['metadata_sha256']:
            raise ValueError('Existing B teacher differs from its metadata-only discovery receipt')
        base.save(receipt_path, result, exclusive=True)
        return result
    current = teacher_manifest(plan['source'], arm, plan['metadata_sha256'])
    if current['identity_sha256'] != plan['identity_sha256']:
        raise ValueError('Shared A teacher export changed after ready')
    return implementation(plan['source'], root, variant='A', cache_root=spec['cache_root'],
        source_receipt=spec.get('teacher_receipt'), receipt_path=receipt_path)


def stage_cache(spec, manifest):
    if not (Path(spec['cache_root']) / '.r84_cache_identity.json').is_file():
        raise RuntimeError('R101 reuses the existing R84 primary cache; no fresh copy or regeneration')
    primary = adapter.original_stage_cache(spec, manifest)
    plan = base.read(base.receipts(spec) / 'plan.json')
    return dict(primary, teacher=stage_teacher(spec, plan['teacher_cache']))


def predecessor_gate(arm):
    version = {'A': 'R100', 'B': 'R98'}[arm]
    run_id = version + '_A'
    root = REPO / 'code/V4/runs' / {'A': 'R100_pairwise_context', 'B': 'R98_performance_context'}[arm]
    launch = ROOT / run_id
    controller, worker = optional(launch / 'controller.json'), optional(launch / 'worker.json')
    spec = optional(ROOT / 'specs' / (run_id + '.json'))
    gpu = [s['name'] for s in spec.get('stages', []) if s.get('device') == 'gpu']
    cpu = [s['name'] for s in spec.get('stages', []) if s.get('device') == 'cpu']
    run, post = optional(root / 'run_complete.json'), optional(root / 'postprocess_status.json')
    records = ((launch / 'controller.json', controller), (launch / 'worker.json', worker),
               (root / 'run_complete.json', run), (root / 'postprocess_status.json', post))
    errors = [error for path, value in records for error in receipt_errors(path, value)]
    checks = dict(controller_rc0=stages_ready(controller, ['allocation_GPU_pipeline', *cpu]),
        all_GPU_stages_rc0=stages_ready(worker, gpu), CPU_stage_declared=bool(cpu),
        model_complete=run.get('complete') is True, report_complete=post.get('complete') is True
        and post.get('phase') == 'complete' and post.get('exit_status', 0) == 0,
        PDF=pdf_ready(root / 'RESULTS.pdf'))
    return dict(arm=arm, predecessor=run_id, complete=not errors and all(checks.values()),
                checks=checks, errors=errors, healthy_runs_stopped=False)


def start(path, evidence=None, tested=False):
    if not evidence or tested is not True:
        raise ValueError('Explicit main-ready evidence and CPU tests are required before start')
    spec, _ = base.check_ready(path)
    predecessor = predecessor_gate(arm_of(spec))
    if predecessor['errors'] or not predecessor['complete']:
        raise RuntimeError('Wait for the predecessor full controller/GPU/CPU/PDF rc0; no start or interruption')
    return original_start(path, evidence, tested)


def queue(path, evidence=None, tested=False, interval=60):
    """Explicitly authorized CPU wait; start once only after the predecessor."""
    if not evidence or tested is not True or interval <= 0:
        raise ValueError('Queue requires main-ready CPU-test evidence and a positive interval')
    spec, plan = base.check_ready(path)
    if plan.get('evidence') != evidence:
        raise ValueError('Queue evidence must match the armed launch plan')
    output = base.receipts(spec)
    with base.lock(output / 'queue.lock'):
        receipt_path = output / 'queue_receipt.json'
        if receipt_path.exists():
            raise RuntimeError('Queue already attempted; preserve original receipt, no automatic retry')
        receipt = dict(state='waiting_predecessor', cpu_only=True, automatic_retry=False,
            started_utc=base.utc(), main_ready_evidence=evidence)
        base.save(receipt_path, receipt, exclusive=True)
        try:
            while True:
                predecessor = predecessor_gate(arm_of(spec))
                if predecessor['errors']:
                    raise RuntimeError('Predecessor failed; no start, stop or automatic retry')
                if predecessor['complete']:
                    result = start(path, evidence, tested)
                    receipt.update(state='complete', exit_status=0, completed_utc=base.utc(), start=result)
                    base.save(receipt_path, receipt)
                    return result
                time.sleep(interval)
        except BaseException as error:
            receipt.update(state='failed', exit_status=1, error=str(error), completed_utc=base.utc())
            base.save(receipt_path, receipt)
            raise


# These are private module instances, not imported r98/r84/r81 namespaces.
adapter.SLOTS = base.SLOTS = SLOTS
adapter.ROOT = base.ROOT = ROOT
adapter.TOKEN_ENV = base.TOKEN_ENV = TOKEN_ENV
adapter.WRAPPER = WRAPPER
adapter.self_command = base.self_command = self_command
adapter.load_spec = base.load_spec = load_spec
base.container_command = adapter.container_command
base.environment = adapter.environment
base.reserved_cpus = adapter.reserved_cpus
base.stage_cache = stage_cache
base.session_name = lambda spec: 'goal_' + spec['run_id'].lower()
environment, container_command = adapter.environment, adapter.container_command


def parser():
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument('stage', choices=('prepare', 'start', 'queue', 'predecessor', 'controller',
        'node-resources', 'container-resources', 'entry', 'worker'))
    value.add_argument('--spec', type=Path)
    value.add_argument('--slot', choices=tuple(SLOTS))
    value.add_argument('--arm', choices=tuple(ARM_SLOTS))
    value.add_argument('--token')
    value.add_argument('--evidence')
    value.add_argument('--runner-tests-passed', action='store_true')
    value.add_argument('--interval', type=int, default=60)
    return value


def main(argv=None):
    args = parser().parse_args(argv)
    if args.stage == 'prepare':
        result = prepare(args.spec, args.evidence, args.runner_tests_passed)
    elif args.stage == 'start':
        result = start(args.spec, args.evidence, args.runner_tests_passed)
    elif args.stage == 'queue':
        result = queue(args.spec, args.evidence, args.runner_tests_passed, args.interval)
    elif args.stage == 'predecessor':
        result = predecessor_gate(args.arm or arm_of(load_spec(args.spec)))
    elif args.stage == 'controller':
        return base.controller(args.spec)
    elif args.stage == 'node-resources':
        result = base.node_resources(args.slot)
    elif args.stage == 'container-resources':
        result = base.container_resources(args.slot)
    elif args.stage == 'entry':
        return base.entry(args.spec, args.slot, args.token)
    else:
        return base.worker(args.spec, args.slot, args.token)
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
