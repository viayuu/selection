"""Future, explicit-ready R102 dispatch. Importing this module launches nothing.

Only private R98/R84/R81 adapters are rebound. Existing allocation/container,
GPU mutex inodes and the shared dispatch CPU mutex remain unchanged.
"""

import argparse
import importlib.util
import json
from pathlib import Path
import sys
import time

from . import r98_goal as predecessor_goal


REPO = Path(__file__).resolve().parents[2]
ROOT = REPO / 'code/V4/runs/R98_parallel_dispatch'
MODEL_ROOT = REPO / 'code/V4/runs/R102_full_type_pretrained_fusion'
WRAPPER = REPO / 'code/V4/run_v4_r102_launch.sh'
TOKEN_ENV = 'R102_DISPATCH_TOKEN'
DATA_ROOT = Path('/public/home/shiys/selection_data_new90k_B_filtered_single_reld_mt_seed2_20261018')
PREPARED_ROOT = REPO / 'code/V4/runs/R84_new90k_attention/prepared'
CACHE_SOURCE = REPO / 'code/V4/runs/R84_new90k_attention/input_cache'
CACHE_ROOT = Path('/tmp/shiys_R84_new90k_cache')
ROUTING_ROOT = Path('/tmp/shiys_R94_B_node_features')
FEATURE_ROOTS = {arm: Path('/tmp/shiys_R102_' + arm + '_extra_nodes') for arm in ('A', 'B')}
ARM_SLOTS = dict(A='4090_0', B='4090_1')
SLOT_CPUS = {'4090_0': [0, 1, 2, 3], '4090_1': [4, 5, 72, 73]}
PREDECESSORS = dict(A='B', B='A')
RELD_SHA256 = '2ce5fb49950c4c9f51d2015a125744be9960aa0c60c609cfe235e74d44ad4be3'
ROUTING_PROBLEMS = tuple(p for p in predecessor_goal.PROBLEMS if p not in ('TSP', 'ATSP'))
STAGES = ('input_features', 'train', 'TEST', 'timing', 'LIB', 'PDF')


def private(name, filename):
    spec = importlib.util.spec_from_file_location('code.V4.' + name, REPO / 'code/V4' / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


adapter = private('_r102_dispatch', 'r98_launch.py')
base = adapter.base
SLOTS = {slot: dict(adapter.SLOTS[slot], cap=352) for slot in ARM_SLOTS.values()}
original_load_spec, original_start = adapter.original_load_spec, adapter.start
original_node_resources = base.node_resources


def self_command(stage, spec=None, slot=None, token=None):
    argv = [sys.executable, '-B', '-u', '-m', 'code.V4.r102_launch', stage]
    for key, value in (('--spec', spec), ('--slot', slot), ('--token', token)):
        if value is not None:
            argv.extend((key, str(value)))
    return argv


def arm_of(spec):
    if spec.get('run_id') not in ('R102_A', 'R102_B'):
        raise ValueError('R102 owns only R102_A and R102_B')
    return spec['run_id'][-1]


def locked_protocol(path, spec):
    path = Path(path)
    if path != MODEL_ROOT / 'protocol_lock.json' or not path.is_file():
        raise ValueError('Wait for main to supply the final R102 protocol_lock.json')
    science = base.read(path)
    logical = science.get('logical_global_batch', science.get('global_batch_size', science.get('batch_size')))
    if (science.get('experiment') != 'R102' or science.get('locked') is not True
            or science.get('seed') != 2 or science.get('task_scope') != 'full18'
            or science.get('arms') != ['A', 'B'] or science.get('data_root') != spec['data_root']
            or science.get('prepared_root') != spec['prepared_root']
            or science.get('rows_per_problem') != dict(train=81000, val=4500, test=4500)
            or logical != 1024 or science.get('max_updates', science.get('max_successful_updates')) != 57600
            or science.get('warmup_updates') != 4320 or science.get('eval_every') != 1440
            or science.get('current_query_labels_in_forward', False) is not False):
        raise ValueError('Require main-locked full18 NEW90k/R84 seed2 logical1024/57600/warm4320/VAL1440')
    return science


def stage_options(arm, command, reporter=False):
    options = ['--root', str(MODEL_ROOT), '--arm', arm, '--device', '{device}']
    if reporter:
        return options + (['--lib-batch-size', '2'] if command == 'lib' else [])
    return ['--root', str(MODEL_ROOT), '--arm', arm, '--prepared-root', str(PREPARED_ROOT),
        '--data-root', str(DATA_ROOT), '--cache-root', '{cache_root}',
        '--feature-root', str(FEATURE_ROOTS[arm]), '--routing-feature-root', str(ROUTING_ROOT),
        '--protocol-lock', str(MODEL_ROOT / 'protocol_lock.json'), '--gpu-type', '4090',
        '--large-micro-cap', '352', '--eval-batch-size', '256',
        '--device', '{device}', '--wandb-mode', 'offline']


def expected_stages(arm):
    result = []
    for name, command in zip(STAGES, ('prepare', 'train', 'test', 'timing', 'lib', 'report')):
        reporter = name in ('LIB', 'PDF')
        module = 'code.V4.r102_reporting' if reporter else 'code.V4.r102_experiment'
        result.append(dict(name=name, device='cpu' if name == 'PDF' else 'gpu',
            argv=['{python}', '-B', '-u', '-m', module, command] + stage_options(arm, command, reporter)))
    return result


def load_spec(path):
    # Bypass the historical R98/R99/R100 prefix gate without changing its source.
    spec = original_load_spec(path)
    arm = arm_of(spec)
    expected = dict(run_dir=str(MODEL_ROOT / arm), common_root=str(MODEL_ROOT),
        data_root=str(DATA_ROOT), prepared_root=str(PREPARED_ROOT), cache_source=str(CACHE_SOURCE),
        cache_root=str(CACHE_ROOT), cache_kind='r84_inputs', wandb_mode='offline',
        feature_root=str(FEATURE_ROOTS[arm]), routing_feature_root=str(ROUTING_ROOT),
        science_protocol=str(MODEL_ROOT / 'protocol_lock.json'), cpu_count=4, memory_gib=9,
        routing_mode='reuse_original_R94B', cpus=SLOT_CPUS[ARM_SLOTS[arm]])
    if (any(spec.get(key) != value for key, value in expected.items()) or spec['slots'] != [ARM_SLOTS[arm]]
            or any(key in spec for key in ('routing_stage_json', 'routing_stage_argv', 'routing_source'))):
        raise ValueError('Use fixed R102 gpu05/4090 slots and DIRECT original R84/pretrained routing B reuse; no staging')
    if spec['stages'] != expected_stages(arm):
        raise ValueError('Use the exact R102 prepare/train/test/timing/LIB2/CPU-report CLI in order')
    locked_protocol(spec['science_protocol'], spec)
    return spec


def source_files(spec):
    from .r102_donor import ELG_SOURCE, MATNET_BACKBONE, MATNET_SOURCE, reld

    code = REPO / 'code/V4'
    # These native modules are loaded by absolute path, outside the AST import closure.
    native = [ELG_SOURCE, MATNET_SOURCE, MATNET_BACKBONE,
              reld.RELD_ROOT / 'models/MOEModel_Light.py', reld.RELD_ROOT / 'models/MOELayer.py']
    if any(not path.is_absolute() or not path.is_file() or path.stat().st_size > 1024**2 for path in native):
        raise RuntimeError('All five small native donor Python sources are required for readiness fingerprints')
    sources = [__file__, WRAPPER, adapter.__file__, base.__file__, base.transport.__file__,
        code / 'r98_goal.py', code / 'r101_goal.py', spec['science_protocol'],
        MODEL_ROOT / 'PLAN.md', MODEL_ROOT / 'preparation/LAUNCH_PLAN.md',
        *native, *spec.get('source_files', [])]
    sources += list(code.glob('r102_*.py')) + list(code.glob('test_r102*.py'))
    sources += [code / name for name in ('r102_model.py', 'r102_objective.py', 'r102_donor.py', 'r102_features.py',
        'r102_experiment.py', 'r102_reporting.py', 'r102_goal.py', 'r102_wave.py',
        'r80_reporting.py', 'r84_reporting.py', 'r85_reporting.py', 'r87_reporting.py')]
    sources += [REPO / 'docs/research' / name for name in
        ('pretrained_representation_coverage_tsp_atsp.md', 'post_r100_broad_direction.md')]
    if any(not Path(path).is_file() for path in sources):
        raise RuntimeError('Wait for the whole R102 team source set before readiness')
    return adapter.source_closure(sources)


def prepare(path, evidence=None, tested=False):
    if not isinstance(evidence, str) or not evidence.strip() or tested is not True:
        raise ValueError('Explicit main team-ready evidence and passed CPU tests are required; no fixed count')
    spec = load_spec(path)
    sources = source_files(spec)
    result = adapter.prepare(path, evidence, tested)
    output = base.receipts(spec)
    plan = base.read(output / 'plan.json')
    plan['source_sha256'].update({name: base.sha(name) for name in sources})
    plan.update(routing_cache=dict(mode='reuse_original_R94B', root=str(ROUTING_ROOT),
        splits=['train', 'val'], pretrained_in_both_arms=True, validate_metadata_on_worker=True,
        copy=False, export=False, extract=False, TEST_after_new_VAL_lock_only=True),
        primary_cache_reuse_only=True, automatic_retry=False)
    base.save(output / 'plan.json', plan)
    ready = base.read(output / 'ready.json')
    ready['plan_sha256'] = base.sha(output / 'plan.json')
    base.save(output / 'ready.json', ready)
    base.check_ready(path)
    return dict(result, source_freeze_ready=True, routing_cache_copied=False)


def routing_manifest(root, cache_root):
    """Read/stat TRAIN/VAL only, with no array/TEST opens and no cache writes."""
    root, cache_root = Path(root), Path(cache_root)
    tasks, donor = {}, None
    for split, rows in (('train', 81000), ('val', 4500)):
        tasks[split] = {}
        for problem in ROUTING_PROBLEMS:
            directory = root / split / problem
            path = directory / 'metadata.json'
            metadata = base.read(path)
            original = base.read(cache_root / split / problem / 'metadata.json')
            shape, current = metadata.get('shape', []), metadata.get('donor', {})
            maximum = original.get('true_length_max')
            if (metadata.get('format') != 'r94_frozen_nodes_v1' or metadata.get('complete') is not True
                    or metadata.get('input_only') is not True or metadata.get('split') != split
                    or metadata.get('problem') != problem or metadata.get('rows') != rows
                    or metadata.get('rows_written') != rows or metadata.get('dtype') != 'float16'
                    or metadata.get('file') != 'nco_nodes.npy' or shape != [rows, maximum, 128]
                    or not isinstance(maximum, int) or maximum < 2
                    or current.get('variant') != 'pretrained' or current.get('checkpoint_sha256') != RELD_SHA256
                    or current.get('output_dim') != 128 or current.get('supported_problems') != list(ROUTING_PROBLEMS)
                    or original.get('complete') is not True or original.get('rows') != rows
                    or original.get('split') != split or original.get('problem') != problem
                    or original.get('data_root') != str(DATA_ROOT)
                    or metadata.get('source_cache', {}).get('metadata_sha256') != predecessor_goal.json_hash(original)):
                raise ValueError('Require original pretrained R94B/R84 TRAIN/VAL metadata: ' + split + '/' + problem)
            if donor is not None and donor != current:
                raise ValueError('All routing tasks/splits must use the identical pretrained R94B donor')
            donor = current
            payload = directory / 'nco_nodes.npy'
            if (path.is_symlink() or payload.is_symlink() or not payload.is_file()
                    or any(parent.is_symlink() for parent in path.parents)
                    or payload.stat().st_size <= rows * maximum * 128 * 2):
                raise ValueError('Original routing FP16 payload missing/incomplete; no regeneration/copy')
            tasks[split][problem] = dict(metadata_sha256=base.sha(path), bytes=payload.stat().st_size)
    return dict(root=str(root), mode='reuse_original_R94B', complete=True, metadata_only=True,
        payload_opened=False, splits=['train', 'val'], tasks=tasks, donor=donor,
        copy=False, export=False, extraction=False, TEST_opened=False)


def stage_cache(spec, manifest):
    # Deliberately do not call the historical copy-capable primary-cache helper.
    root = Path(spec['cache_root'])
    stamp = root / '.r84_cache_identity.json'
    with base.lock(root.with_name(root.name + '.r84_stage.lock'), blocking=True):
        if not stamp.is_file() or base.read(stamp).get('identity_sha256') != manifest['identity_sha256']:
            raise RuntimeError('Original local R84 cache required; no rebuild or fresh copy')
        for row in manifest['files']:
            path = root / row['path']
            if (not path.is_file() or path.is_symlink() or path.stat().st_size != row['bytes']
                    or ('sha256' in row and base.sha(path) != row['sha256'])):
                raise RuntimeError('Original local R84 cache incomplete; no repair/copy')
    return dict(root=str(root), reused=True, identity_sha256=manifest['identity_sha256'],
                routing=routing_manifest(spec['routing_feature_root'], root))


def predecessor_gate(arm):
    if arm not in ARM_SLOTS:
        raise ValueError('Choose R102 arm A or B')
    from . import r101_goal
    prior_arm = PREDECESSORS[arm]
    run_id = 'R101_' + prior_arm
    model = REPO / 'code/V4/runs/R101_hint_pretraining'
    run, launch = model / prior_arm, ROOT / run_id
    current = r101_goal.status(root=model, launch_root=ROOT, persist=False)
    row = next(row for row in current['arms'] if row['arm'] == prior_arm)
    checks = dict(row['checks'])
    spec = predecessor_goal.optional(ROOT / 'specs' / (run_id + '.json'))
    checks.update(assigned_predecessor=spec.get('run_id') == run_id and spec.get('run_dir') == str(run)
        and spec.get('slots') == [ARM_SLOTS[arm]],
        exact_pipeline=[(s.get('name'), s.get('device')) for s in spec.get('stages', [])]
        == [(name, 'cpu' if name == 'PDF' else 'gpu')
            for name in ('pretrain', 'train', 'TEST', 'timing', 'LIB', 'PDF')])
    # A different predecessor's failure or missing comparison never blocks this healthy arm.
    errors = [error for error in current['errors']
              if Path(error['path']).is_relative_to(run) or Path(error['path']).is_relative_to(launch)]
    return dict(arm=arm, predecessor=run_id, complete=not errors and all(checks.values()),
        checks=checks, errors=errors, healthy_runs_stopped=False, timeout_takeover=False)


def start(path, evidence=None, tested=False):
    if not isinstance(evidence, str) or not evidence.strip() or tested is not True:
        raise ValueError('Explicit main-ready evidence and CPU tests are required before start')
    spec, _ = base.check_ready(path)
    predecessor = predecessor_gate(arm_of(spec))
    if predecessor['errors'] or not predecessor['complete']:
        raise RuntimeError('Wait for this R101 arm full train/TEST/timing/LIB/PDF rc0; no timeout takeover')
    return original_start(path, evidence, tested)


def queue(path, evidence=None, tested=False, interval=60):
    if not isinstance(evidence, str) or not evidence.strip() or tested is not True or interval <= 0:
        raise ValueError('Queue requires main-ready CPU evidence and a positive interval')
    spec, plan = base.check_ready(path)
    if plan.get('evidence') != evidence:
        raise ValueError('Queue evidence must match the source-frozen ready plan')
    output = base.receipts(spec)
    with base.lock(output / 'queue.lock'):
        path_receipt = output / 'queue_receipt.json'
        if path_receipt.exists():
            raise RuntimeError('Queue already attempted; preserve receipt, no automatic retry')
        receipt = dict(state='waiting_predecessor', cpu_only=True, automatic_retry=False,
            started_utc=base.utc(), main_ready_evidence=evidence)
        base.save(path_receipt, receipt, exclusive=True)
        try:
            while True:
                base.check_ready(path)
                predecessor = predecessor_gate(arm_of(spec))
                if predecessor['errors']:
                    raise RuntimeError('Assigned predecessor failed; no launch or automatic retry')
                if predecessor['complete']:
                    result = start(path, evidence, tested)
                    receipt.update(state='complete', exit_status=0, completed_utc=base.utc(), start=result)
                    base.save(path_receipt, receipt)
                    return result
                time.sleep(interval)
        except BaseException as error:
            receipt.update(state='failed', exit_status=1, error=str(error), completed_utc=base.utc())
            base.save(path_receipt, receipt)
            raise


def controller(path):
    spec, _ = base.check_ready(path)
    gate = predecessor_gate(arm_of(spec))
    if gate['errors'] or not gate['complete']:
        raise RuntimeError('Controller cannot bypass the assigned predecessor completion gate')
    return base.controller(path)


def node_resources(slot):
    probe = original_node_resources(slot)
    cpus = SLOT_CPUS[slot]
    if not set(cpus) <= set(probe['allowed_cpus']):
        raise RuntimeError('Original gpu05 CPU quartet is outside the allocation/container affinity')
    return dict(probe, allowed_cpus=list(cpus))


# Historical module globals stay untouched; lock roots intentionally stay shared.
adapter.SLOTS = base.SLOTS = SLOTS
adapter.ROOT = base.ROOT = ROOT
adapter.TOKEN_ENV = base.TOKEN_ENV = TOKEN_ENV
adapter.WRAPPER = WRAPPER
adapter.self_command = base.self_command = self_command
adapter.load_spec = base.load_spec = load_spec
base.container_command, base.environment = adapter.container_command, adapter.environment
base.reserved_cpus, base.stage_cache = adapter.reserved_cpus, stage_cache
base.node_resources = node_resources
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
        return controller(args.spec)
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
