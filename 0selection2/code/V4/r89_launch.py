"""Isolated R89 optimizer-comparison launch preparation on job1475's 4090s."""

import argparse
import importlib
import importlib.util
import os
from pathlib import Path
import sys
import json


REPO = Path('/public/home/shiys/0selection2')
ROOT = REPO / 'code/V4/runs/R89_launch'
MODEL_ROOT = REPO / 'code/V4/runs/R89_optimizer_geometry'
WRAPPER = REPO / 'code/V4/run_v4_r89_launch.sh'
TOKEN_ENV = 'R89_LAUNCH_TOKEN'
ARM_SLOTS = dict(A_adamw='4090_0', B_muon_hybrid='4090_1')
PREDECESSORS = dict(A_adamw='A_shared_residual', B_muon_hybrid='B_private_residual')
PREVIOUS_LAUNCH = REPO / 'code/V4/runs/R87_launch'
PREVIOUS_MODELS = REPO / 'code/V4/runs/R87_private_method_residual'

# All routing mutations belong to private namespaces, not the frozen launchers.
_spec = importlib.util.spec_from_file_location(
    'code.V4._r89_resource_helpers', REPO / 'code/V4/r87_launch.py')
engine = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(engine)
base = engine.base
DATA, PREPARED = engine.DATA, engine.PREPARED
CACHE_SOURCE, CACHE_ROOT = engine.CACHE_SOURCE, engine.CACHE_ROOT
CACHE_EVIDENCE = engine.saved.ROOT / 'resources.json'
SLOTS = {key: dict(engine.saved.SLOTS[key], cap=352) for key in ARM_SLOTS.values()}
_template, _predecessor_gate = engine.template, engine.predecessor_gate


def self_command(stage, spec=None, slot=None, token=None):
    command = [sys.executable, '-B', '-u', '-m', 'code.V4.r89_launch', stage]
    for flag, value in (('--spec', spec), ('--slot', slot), ('--token', token)):
        if value is not None:
            command += [flag, str(value)]
    return command


def predecessor_gate(arm):
    result = _predecessor_gate(arm)
    if not result['complete']:
        return result
    path = engine.R85_MODELS / 'goal_compare_receipt.json'
    result['complete'] = False
    if not path.is_file():
        result['reason'] = 'waiting for R87 completed round comparison receipt'
        return result
    comparison = base.read(path)
    result['round_comparison'] = dict(path=str(path), state=comparison.get('state'),
        exit_status=comparison.get('exit_status'))
    if (comparison.get('state') == 'failed'
            or comparison.get('exit_status') not in (None, 0)):
        result.update(failed=True, reason='R87 comparison failed; no automatic repair or retry')
    elif comparison.get('state') == 'complete' and comparison.get('exit_status') == 0:
        result.update(complete=True, reason='R87 train/TEST/LIB/PDF and comparison complete rc0')
    else:
        result['reason'] = 'waiting for R87 comparison complete rc0'
    return result


def commands(spec, slot):
    return engine._commands(dict(spec,
        wandb_mode=os.environ.get('R89_RESOLVED_WANDB_MODE', 'auto')), slot)


def template(arm):
    value = _template(arm)
    source_names = (
        'r89_experiment.py', 'r89_reporting.py', 'r88_model.py', 'r88_experiment.py',
        'r88_reporting.py', 'r87_experiment.py', 'r87_reporting.py',
        'r85_experiment.py', 'r85_reporting.py', 'r84_experiment.py',
        'r84_reporting.py', 'r84_prepare.py', 'r76_experiment.py', 'r77_experiment.py',
        'r80_experiment.py', 'r80_reporting.py', 'r77_best_model_report.py',
        'pairwise_selector.py', 'pairwise_objective.py', 'solver_code_encoder.py')
    sources = [str(REPO / 'code/V4' / name) for name in source_names]
    # Freeze every final R89 runtime module, including the optimizer's actual name.
    sources += [str(path) for path in sorted((REPO / 'code/V4').glob('r89_*.py'))]
    sources += [str(PREPARED / name) for name in (
        'prepared.json', 'geometry_stats.json', 'common_full_train_sbs.json',
        'identicalinit_seed2.pt')]
    sources += [str(MODEL_ROOT / 'protocol_lock.json')]
    value['source_files'] = list(dict.fromkeys(sources))
    value['protocol'].update(role_slot=3, model_role_arm='B_execution_anchor',
        extra_model_parameters=0, task_scope='full18', optimizer_arm=arm,
        forward_precision='FP16', muon_internal_precision='BF16',
        muon_newton_schulz_steps=5)
    value['startup_evidence_required'] = dict(
        real_PID_UUID_cgroup=True, full18_finite_PART_KEYS=True,
        actual_optimizer_type_and_groups=True,
        actual_muon_step_applied_positive=(arm == 'B_muon_hybrid'),
        actual_update_RMS_finite=(arm == 'B_muon_hybrid'),
        validation_wait_required=False)
    for stage in value['stages']:
        stage['argv'] = [arg.replace('code.V4.r87_', 'code.V4.r89_')
                         for arg in stage['argv']]
    return value


def prepare(path):
    # R87 is the predecessor, but the reusable cache evidence belongs to R85.
    value = engine._prepare(path)
    for source in (Path(__file__).resolve(), REPO / 'code/V4/r87_launch.py',
                   REPO / 'code/V4/r85_launch.py', WRAPPER, CACHE_EVIDENCE):
        value['source_sha256'][str(source)] = base.sha(source)
    value.update(experiment='R89', cache_mode='existing_read_only',
        main_thread_ready=False, predecessor=predecessor_gate(value['spec']['run_id']))
    base.save(base.receipts(value['spec']) / 'plan.json', value)
    return value


def worker(path, slot, token):
    engine.require_predecessor(path)
    base.assert_container(slot)
    spec, _ = base.check_ready(path)
    claim = base.read(base.receipts(spec) / 'claim.json')
    if claim['slot'] != slot or claim['token'] != token or os.environ.get(TOKEN_ENV) != token:
        raise RuntimeError('R89 worker requires its unique claimed 4090 slot/token')
    os.sched_setaffinity(0, claim['cpus'])
    lookup = engine.wandb_lookup()
    base.save(base.receipts(spec) / 'wandb_lookup.json', lookup, exclusive=True)
    os.environ['R89_RESOLVED_WANDB_MODE'] = lookup['selected_mode']
    os.environ['WANDB_MODE'] = lookup['selected_mode']
    return engine._worker(path, slot, token)


def write_specs():
    runner = importlib.import_module('code.V4.r89_experiment')
    reporter = importlib.import_module('code.V4.r89_reporting')
    if tuple(runner.ARMS) != tuple(ARM_SLOTS) or tuple(reporter.ARMS) != tuple(ARM_SLOTS):
        raise RuntimeError('Use the main-locked R89 AdamW/Muon-hybrid arms')
    rows, lock = [], None
    for arm, slot in ARM_SLOTS.items():
        value = template(arm)
        path = ROOT / 'specs' / (arm + '.json')
        if path.is_file() and base.read(path) != value:
            raise RuntimeError('Existing R89 spec differs; explicit main update required')
        stages = commands(value, slot)
        for mode in ('train', 'test'):
            argv = [mode, *stages[0]['argv'][6:]]
            parsed = runner.parser().parse_args(argv)
            if (parsed.arm != arm or parsed.root != MODEL_ROOT or parsed.cache_root != CACHE_ROOT
                    or parsed.gpu_type != '4090' or parsed.large_micro_cap != 352
                    or parsed.eval_batch_size != 256 or parsed.resume):
                raise RuntimeError('R89 runner argv differs from the matched controls')
            if lock is None:
                lock = runner.protocol_lock(parsed)
            rows.append(dict(arm=arm, command=mode, argv=argv, parsed=True, slot=slot))
        for stage in stages[1:]:
            argv = stage['argv'][5:]
            parsed = reporter.parser().parse_args(argv)
            if (parsed.arm != arm or parsed.root != MODEL_ROOT
                    or parsed.device != ('cuda:0' if stage['device'] == 'gpu' else 'cpu')):
                raise RuntimeError('R89 reporter argv differs from the locked controls')
            rows.append(dict(arm=arm, command=parsed.command, argv=argv, parsed=True))
        if not path.is_file():
            base.save(path, value, exclusive=True)
        engine.load_spec(path)
    result = dict(created_utc=base.utc(), experiment='R89', state='CLI_parsed',
        commands=rows, parsed_count=len(rows), protocol_lock=lock, slots=ARM_SLOTS,
        source_hashes_frozen=False, main_thread_ready=False, training_started=False,
        cache_audit_repeated=False,
        predecessor_gates={arm: predecessor_gate(arm) for arm in ARM_SLOTS})
    base.save(ROOT / 'spec_parse.json', result)
    return result


engine.ROOT, engine.MODEL_ROOT = ROOT, MODEL_ROOT
engine.WRAPPER, engine.TOKEN_ENV = WRAPPER, TOKEN_ENV
engine.ARM_SLOTS, engine.PREDECESSORS, engine.SLOTS = ARM_SLOTS, PREDECESSORS, SLOTS
engine.R85_LAUNCH, engine.R85_MODELS = PREVIOUS_LAUNCH, PREVIOUS_MODELS
engine.predecessor_gate = predecessor_gate
base.ROOT, base.SLOTS, base.TOKEN_ENV = ROOT, SLOTS, TOKEN_ENV
base.self_command = self_command
base.container_command = engine.container_command
base.commands = commands
base.session_name = lambda value: 'r89_' + value['run_id']


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('template', 'specs', 'prepare', 'start',
        'controller', 'node-resources', 'container-resources', 'entry', 'worker', 'gates'))
    parser.add_argument('--spec', type=Path)
    parser.add_argument('--arm', choices=tuple(ARM_SLOTS), default='A_adamw')
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
        value = {arm: predecessor_gate(arm) for arm in ARM_SLOTS}
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
