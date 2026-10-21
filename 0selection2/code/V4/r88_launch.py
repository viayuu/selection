"""R88 launch preparation on the two existing job1465 RTX 3090 UUIDs."""

import argparse
import importlib
import importlib.util
import json
import os
from pathlib import Path
import sys


REPO = Path('/public/home/shiys/0selection2')
ROOT = REPO / 'code/V4/runs/R88_launch'
MODEL_ROOT = REPO / 'code/V4/runs/R88_execution_role_inputs'
WRAPPER = REPO / 'code/V4/run_v4_r88_launch.sh'
TOKEN_ENV = 'R88_LAUNCH_TOKEN'
ARM_SLOTS = dict(A_canonical_anchor='3090_0', B_execution_anchor='3090_1')
PREDECESSORS = dict(A_canonical_anchor='A_ovr_control', B_execution_anchor='B_gradient_balance')

# Private helper namespaces do not alter the running R84/R85/R87 launchers.
_spec = importlib.util.spec_from_file_location('code.V4._r88_resource_helpers', REPO / 'code/V4/r87_launch.py')
engine = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(engine)
base = engine.base
DATA, PREPARED, CACHE_SOURCE, CACHE_ROOT = engine.DATA, engine.PREPARED, engine.CACHE_SOURCE, engine.CACHE_ROOT
SLOTS = {key: dict(engine.saved.SLOTS[key], cap=352) for key in ARM_SLOTS.values()}
_template = engine.template


def self_command(stage, spec=None, slot=None, token=None):
    command = [sys.executable, '-B', '-u', '-m', 'code.V4.r88_launch', stage]
    for flag, value in (('--spec', spec), ('--slot', slot), ('--token', token)):
        if value is not None:
            command += [flag, str(value)]
    return command


def commands(spec, slot):
    return engine._commands(dict(spec, wandb_mode=os.environ.get('R88_RESOLVED_WANDB_MODE', 'auto')), slot)


def template(arm):
    value = _template(arm)
    value['source_files'] = [path.replace('/r87_', '/r88_') for path in value['source_files']]
    value['protocol'].update(role_slot=3, extra_model_parameters=0,
        task_scope='full18', A_canonical_exact_ties='inclusive')
    for stage in value['stages']:
        stage['argv'] = [arg.replace('code.V4.r87_', 'code.V4.r88_') for arg in stage['argv']]
        if '--gpu-type' in stage['argv']:
            stage['argv'][stage['argv'].index('--gpu-type') + 1] = '3090'
    return value


def prepare(path):
    value = engine.prepare(path)
    for source in (Path(__file__).resolve(), REPO / 'code/V4/r87_launch.py', WRAPPER):
        value['source_sha256'][str(source)] = base.sha(source)
    value.update(experiment='R88', main_thread_ready=False)
    base.save(base.receipts(value['spec']) / 'plan.json', value)
    return value


def worker(path, slot, token):
    engine.require_predecessor(path)
    base.assert_container(slot)
    spec, _ = base.check_ready(path)
    claim = base.read(base.receipts(spec) / 'claim.json')
    if claim['slot'] != slot or claim['token'] != token or os.environ.get(TOKEN_ENV) != token:
        raise RuntimeError('R88 worker requires its unique claimed 3090 slot/token')
    os.sched_setaffinity(0, claim['cpus'])
    lookup = engine.wandb_lookup()
    base.save(base.receipts(spec) / 'wandb_lookup.json', lookup, exclusive=True)
    os.environ['R88_RESOLVED_WANDB_MODE'] = lookup['selected_mode']
    os.environ['WANDB_MODE'] = lookup['selected_mode']
    return engine._worker(path, slot, token)


def write_specs():
    runner = importlib.import_module('code.V4.r88_experiment')
    reporter = importlib.import_module('code.V4.r88_reporting')
    if tuple(runner.ARMS) != tuple(ARM_SLOTS) or tuple(reporter.ARMS) != tuple(ARM_SLOTS):
        raise RuntimeError('Use the locked R88 canonical/execution-anchor arms')
    rows, lock = [], None
    for arm, slot in ARM_SLOTS.items():
        value = template(arm)
        path = ROOT / 'specs' / (arm + '.json')
        if path.is_file() and base.read(path) != value:
            raise RuntimeError('Existing R88 spec differs; explicit main update required')
        stages = commands(value, slot)
        for mode in ('train', 'test'):
            argv = [mode, *stages[0]['argv'][6:]]
            parsed = runner.parser().parse_args(argv)
            if (parsed.arm != arm or parsed.root != MODEL_ROOT or parsed.cache_root != CACHE_ROOT
                    or parsed.gpu_type != '3090' or parsed.large_micro_cap != 352
                    or parsed.eval_batch_size != 256 or parsed.resume):
                raise RuntimeError('R88 runner command differs from the locked controls')
            if lock is None:
                lock = runner.protocol_lock(parsed)
            rows.append(dict(arm=arm, command=mode, argv=argv, parsed=True, slot=slot))
        for stage in stages[1:]:
            argv = stage['argv'][5:]
            parsed = reporter.parser().parse_args(argv)
            if (parsed.arm != arm or parsed.root != MODEL_ROOT
                    or parsed.device != ('cuda:0' if stage['device'] == 'gpu' else 'cpu')):
                raise RuntimeError('R88 reporting command differs from the locked controls')
            rows.append(dict(arm=arm, command=parsed.command, argv=argv, parsed=True))
        if not path.is_file():
            base.save(path, value, exclusive=True)
        engine.load_spec(path)
    result = dict(created_utc=base.utc(), experiment='R88', state='CLI_parsed', commands=rows,
        parsed_count=len(rows), protocol_lock=lock, slots=ARM_SLOTS,
        source_hashes_frozen=False, main_thread_ready=False, training_started=False,
        cache_audit_repeated=False,
        predecessor_gates={arm: engine.predecessor_gate(arm) for arm in ARM_SLOTS})
    base.save(ROOT / 'spec_parse.json', result)
    return result


engine.ROOT, engine.MODEL_ROOT, engine.WRAPPER, engine.TOKEN_ENV = ROOT, MODEL_ROOT, WRAPPER, TOKEN_ENV
engine.ARM_SLOTS, engine.PREDECESSORS, engine.SLOTS = ARM_SLOTS, PREDECESSORS, SLOTS
base.ROOT, base.SLOTS, base.TOKEN_ENV = ROOT, SLOTS, TOKEN_ENV
base.self_command = self_command
base.container_command = engine.container_command
base.commands = commands
base.session_name = lambda value: 'r88_' + value['run_id']


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('template', 'specs', 'prepare', 'start', 'controller',
        'node-resources', 'container-resources', 'entry', 'worker', 'gates'))
    parser.add_argument('--spec', type=Path)
    parser.add_argument('--arm', choices=tuple(ARM_SLOTS), default='A_canonical_anchor')
    parser.add_argument('--slot', choices=tuple(SLOTS))
    parser.add_argument('--token')
    parser.add_argument('--main-ready-evidence')
    parser.add_argument('--runner-tests-passed', action='store_true')
    args = parser.parse_args(argv)
    if args.stage in ('prepare', 'start', 'controller', 'entry', 'worker') and not args.spec:
        parser.error('--spec required')
    if args.stage in ('node-resources', 'container-resources', 'entry', 'worker') and not args.slot:
        parser.error('--slot required')
    if args.stage == 'template':
        value = template(args.arm)
    elif args.stage == 'specs':
        value = write_specs()
    elif args.stage == 'gates':
        value = {arm: engine.predecessor_gate(arm) for arm in ARM_SLOTS}
    elif args.stage == 'prepare':
        value = prepare(args.spec)
    elif args.stage == 'start':
        value = base.start(args.spec, args.main_ready_evidence, args.runner_tests_passed)
    elif args.stage == 'controller':
        return engine.controller(args.spec)
    elif args.stage == 'node-resources':
        value = engine.saved._node_resources(args.slot)
    elif args.stage == 'container-resources':
        value = base.container_resources(args.slot)
    elif args.stage == 'entry':
        return engine.entry(args.spec, args.slot, args.token)
    else:
        return worker(args.spec, args.slot, args.token)
    print(json.dumps(value, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
