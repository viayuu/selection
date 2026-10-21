"""R91 sidecar: explicit-ready, one-shot launches on existing job1465 3090s."""

import argparse
import importlib
import importlib.util
import json
import os
from pathlib import Path
import sys


REPO = Path('/public/home/shiys/0selection2')
ROOT = REPO / 'code/V4/runs/R91_launch'
MODEL_ROOT = REPO / 'code/V4/runs/R91_probability_link_control'
WRAPPER = REPO / 'code/V4/run_v4_r91_launch.sh'
TOKEN_ENV = 'R91_LAUNCH_TOKEN'
ARM_SLOTS = dict(A_utility_ce='3090_0', B_utility_probabilities='3090_1')
PREDECESSORS = dict(A_utility_ce='A_canonical_anchor',
                    B_utility_probabilities='B_execution_anchor')
PREVIOUS_LAUNCH = REPO / 'code/V4/runs/R88_launch'
PREVIOUS_MODELS = REPO / 'code/V4/runs/R88_execution_role_inputs'
CONTAINER = 'b8ced5d8a8cdafdc5006f827a0d36c319c61056a65316af4ff9937453fa0635c'
ALLOWED_CPUS = (12, 13, 14, 15, 16, 17, 64, 65, 66, 67)

# As in R88/R89, mutate only privately loaded helpers, not live launcher modules.
_spec = importlib.util.spec_from_file_location(
    'code.V4._r91_resource_helpers', REPO / 'code/V4/r87_launch.py')
engine = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(engine)
base = engine.base
DATA, PREPARED = engine.DATA, engine.PREPARED
CACHE_SOURCE, CACHE_ROOT = engine.CACHE_SOURCE, engine.CACHE_ROOT
CACHE_EVIDENCE = engine.saved.ROOT / 'resources.json'
SLOTS = {key: dict(engine.saved.SLOTS[key], cap=352, physical_index=index)
         for key, index in zip(ARM_SLOTS.values(), (3, 4))}
_template, _environment = engine.template, base.environment
_predecessor_gate, _node_resources = engine.predecessor_gate, engine.saved._node_resources


def self_command(stage, spec=None, slot=None, token=None):
    command = [sys.executable, '-B', '-u', '-m', 'code.V4.r91_launch', stage]
    for flag, value in (('--spec', spec), ('--slot', slot), ('--token', token)):
        if value is not None:
            command += [flag, str(value)]
    return command


def environment(slot=None, token=None):
    return dict(_environment(slot, token), WANDB_MODE='offline')


def predecessor_gate(arm):
    result = _predecessor_gate(arm)
    for record in result['records'].values():
        if (record.get('exit_status') not in (None, 0)
                or any(stage.get('state') == 'failed'
                       or stage.get('exit_status') not in (None, 0)
                       for stage in record['stages'])):
            result.update(complete=False, failed=True,
                          reason='R88 predecessor failed; no automatic repair or retry')
    report = engine.R85_MODELS / PREDECESSORS[arm] / 'postprocess_status.json'
    if report.is_file():
        status = base.read(report)
        if 'failed' in status.get('phase', '') or status.get('exit_status') not in (None, 0):
            result.update(complete=False, failed=True,
                          reason='R88 report failed; no automatic repair or retry')
        elif result['complete'] and status.get('complete') is not True:
            result.update(complete=False, reason='waiting for R88 completed report delivery receipt')
    return result


def commands(spec, slot):
    if ARM_SLOTS.get(spec['run_id']) != slot or spec['slots'] != [slot]:
        raise ValueError('Each R91 utility arm has one fixed job1465 GPU UUID')
    return engine._commands(dict(spec, wandb_mode='offline'), slot)


def template(arm):
    value = _template(arm)
    source_names = (
        'r91_experiment.py', 'r91_objective.py', 'r91_reporting.py',
        'r88_model.py', 'r88_experiment.py', 'r88_reporting.py',
        'r87_model.py', 'r87_experiment.py', 'r87_reporting.py',
        'r85_candidate_selector.py', 'r85_objectives.py',
        'r85_experiment.py', 'r85_reporting.py', 'r84_experiment.py',
        'r84_reporting.py', 'r84_prepare.py', 'r84_nss_reference.py',
        'r76_experiment.py', 'r77_experiment.py', 'r80_experiment.py',
        'r80_reporting.py', 'r77_best_model_report.py', 'r42_experiment.py',
        'pairwise_selector.py', 'pairwise_objective.py', 'solver_code_encoder.py')
    sources = [str(REPO / 'code/V4' / name) for name in source_names]
    sources += [str(path) for path in sorted((REPO / 'code/V4').glob('r91_*.py'))]
    sources += [str(PREPARED / name) for name in (
        'prepared.json', 'geometry_stats.json', 'common_full_train_sbs.json',
        'identicalinit_seed2.pt')]
    sources += [str(MODEL_ROOT / 'protocol_lock.json')]
    value.update(source_files=list(dict.fromkeys(sources)), wandb_mode='offline')
    train = dict(name='train', device='gpu', argv=[
        '{python}', '-B', '-u', '-m', 'code.V4.r91_experiment', 'train',
        '--root', '{common_root}', '--arm', arm, '--prepared-root', '{prepared_root}',
        '--data-root', '{data_root}', '--cache-root', '{cache_root}',
        '--protocol-lock', '{common_root}/protocol_lock.json', '--gpu-type', '3090',
        '--large-micro-cap', '352', '--eval-batch-size', '256', '--device', '{device}',
        '--wandb-mode', 'offline'])
    test = dict(name='TEST', device='gpu', argv=[*train['argv']])
    test['argv'][5] = 'test'
    value['stages'] = [train, test,
        dict(name='LIB', device='gpu', argv=['{python}', '-B', '-u', '-m',
            'code.V4.r91_reporting', 'lib', '--root', '{common_root}', '--arm', arm,
            '--device', '{device}', '--lib-batch-size', '2']),
        dict(name='PDF', device='gpu', argv=['{python}', '-B', '-u', '-m',
            'code.V4.r91_reporting', 'report', '--root', '{common_root}', '--arm', arm,
            '--device', '{device}'])]
    return value


def load_spec(path):
    value = engine.saved._load_spec(path)
    if value != template(value['run_id']):
        raise ValueError('R91 spec must match exact utility arms, paths, controls and same-GPU stages')
    return value


def parse_commands(require_reporter=False):
    runner = importlib.import_module('code.V4.r91_experiment')
    reporter_path = REPO / 'code/V4/r91_reporting.py'
    reporter = importlib.import_module('code.V4.r91_reporting') if reporter_path.is_file() else None
    if tuple(runner.ARMS) != tuple(ARM_SLOTS):
        raise RuntimeError('Runner arms must be A_utility_ce,B_utility_probabilities in order')
    if require_reporter and reporter is None:
        raise RuntimeError('Final R91 reporting parser required before prepare')
    if reporter is not None and tuple(reporter.ARMS) != tuple(ARM_SLOTS):
        raise RuntimeError('Reporter arms differ from the matched R91 utility arms')
    rows, reports = [], []
    for arm, slot in ARM_SLOTS.items():
        for stage in commands(template(arm), slot):
            argv = stage['argv'][5:]
            if stage['name'] in ('train', 'TEST'):
                parsed = runner.parser().parse_args(argv)
                expected = dict(command=argv[0], arm=arm, root=MODEL_ROOT,
                    prepared_root=PREPARED, data_root=DATA, cache_root=CACHE_ROOT,
                    protocol_lock=MODEL_ROOT / 'protocol_lock.json', gpu_type='3090',
                    large_micro_cap=352, eval_batch_size=256, wandb_mode='offline',
                    device='cuda:0', resume=False)
                if any(getattr(parsed, key, None) != value for key, value in expected.items()):
                    raise RuntimeError('R91 runner argv differs from the fixed matched controls')
                rows.append(dict(arm=arm, slot=slot, command=parsed.command,
                                 argv=argv, parsed=True))
            elif reporter is not None:
                parsed = reporter.parser().parse_args(argv)
                if (parsed.arm != arm or parsed.root != MODEL_ROOT or parsed.device != 'cuda:0'
                        or parsed.command != argv[0] or parsed.lib_batch_size != 2):
                    raise RuntimeError('R91 reporting must retain the same claimed GPU and arm')
                reports.append(dict(arm=arm, slot=slot, command=parsed.command,
                                    argv=argv, parsed=True))
    return dict(train_test_argv=rows, train_test_parsed_count=len(rows),
        LIB_PDF_argv=reports, LIB_PDF_parsed_count=len(reports),
        LIB_PDF_parser_confirmed=reporter is not None)


def write_specs():
    parsed = parse_commands()
    for arm in ARM_SLOTS:
        path, value = ROOT / 'specs' / (arm + '.json'), template(arm)
        if path.is_file() and base.read(path) != value:
            raise RuntimeError('Existing R91 spec differs; explicit main review/update required')
        if not path.is_file():
            base.save(path, value, exclusive=True)
        load_spec(path)
    result = dict(created_utc=base.utc(), experiment='R91', state='CLI_parsed', **parsed,
        slots=ARM_SLOTS, source_hashes_frozen=False, main_thread_ready=False,
        training_started=False, cache_audit_repeated=False,
        protocol_lock_path=str(MODEL_ROOT / 'protocol_lock.json'), protocol_written=False,
        predecessor_gates={arm: predecessor_gate(arm) for arm in ARM_SLOTS})
    base.save(ROOT / 'spec_parse.json', result)
    return result


def prepare(path):
    parse_commands(require_reporter=True)
    # Predecessor roots are R88; reusable cache evidence remains the real R85 file.
    value = engine._prepare(path)
    for source in (Path(__file__).resolve(), REPO / 'code/V4/r87_launch.py',
                   REPO / 'code/V4/r85_launch.py', WRAPPER, CACHE_EVIDENCE):
        value['source_sha256'][str(source)] = base.sha(source)
    value.update(experiment='R91', cache_mode='existing_read_only',
        main_thread_ready=False, automatic_retry=False,
        predecessor=predecessor_gate(value['spec']['run_id']))
    base.save(base.receipts(value['spec']) / 'plan.json', value)
    return value


def node_resources(slot):
    value = _node_resources(slot)
    if (value['container']['id'] != CONTAINER
            or value['allowed_cpus'] != list(ALLOWED_CPUS)):
        raise RuntimeError('R91 requires the existing job1465 container and CPU allocation')
    return value


def worker(path, slot, token):
    engine.require_predecessor(path)
    base.assert_container(slot)
    spec, _ = base.check_ready(path)
    claim = base.read(base.receipts(spec) / 'claim.json')
    if (claim['slot'] != slot or claim['token'] != token or os.environ.get(TOKEN_ENV) != token
            or not set(claim['cpus']) <= set(ALLOWED_CPUS)):
        raise RuntimeError('R91 worker requires its unique claimed job1465 GPU/CPU/token')
    base.save(base.receipts(spec) / 'wandb_lookup.json', dict(enabled=True,
        project='selector', entity='yjkds-southern-university-of-science-technology',
        selected_mode='offline', credential_lookup_performed=False,
        reason='Main-locked offline mode; no known credentials or login prompt'), exclusive=True)
    os.environ['WANDB_MODE'] = 'offline'
    return engine._worker(path, slot, token)


engine.ROOT, engine.MODEL_ROOT = ROOT, MODEL_ROOT
engine.WRAPPER, engine.TOKEN_ENV = WRAPPER, TOKEN_ENV
engine.ARM_SLOTS, engine.PREDECESSORS, engine.SLOTS = ARM_SLOTS, PREDECESSORS, SLOTS
engine.R85_LAUNCH, engine.R85_MODELS = PREVIOUS_LAUNCH, PREVIOUS_MODELS
engine.load_spec, engine.predecessor_gate = load_spec, predecessor_gate
base.ROOT, base.SLOTS, base.TOKEN_ENV = ROOT, SLOTS, TOKEN_ENV
base.self_command, base.load_spec = self_command, load_spec
base.container_command, base.commands = engine.container_command, commands
base.environment, base.node_resources = environment, node_resources
base.session_name = lambda value: 'r91_' + value['run_id']


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('template', 'specs', 'prepare', 'start',
        'controller', 'node-resources', 'container-resources', 'entry', 'worker', 'gates'))
    parser.add_argument('--spec', type=Path)
    parser.add_argument('--arm', choices=tuple(ARM_SLOTS), default='A_utility_ce')
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
        value = node_resources(args.slot)
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
