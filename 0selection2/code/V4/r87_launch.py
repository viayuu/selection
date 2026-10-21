"""R87 resource adapter for two existing allocation-owned RTX 4090 slots."""

import argparse
import importlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time


REPO = Path('/public/home/shiys/0selection2')
ROOT = REPO / 'code/V4/runs/R87_launch'
MODEL_ROOT = REPO / 'code/V4/runs/R87_private_method_residual'
WRAPPER = REPO / 'code/V4/run_v4_r87_launch.sh'
TOKEN_ENV = 'R87_LAUNCH_TOKEN'
ARM_SLOTS = dict(A_shared_residual='4090_0', B_private_residual='4090_1')
PREDECESSORS = dict(A_shared_residual='C_cost_profile', B_private_residual='D_candidate_subset')
R85_LAUNCH = REPO / 'code/V4/runs/R85_launch'
R85_MODELS = REPO / 'code/V4/runs/R85_zero_selection_research'

# Isolate all routing changes from the frozen R84/R85 module namespaces.
_spec = importlib.util.spec_from_file_location('code.V4._r87_saved_resources', REPO / 'code/V4/r85_launch.py')
saved = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(saved)
base = saved.base
DATA, PREPARED, CACHE_SOURCE, CACHE_ROOT = saved.DATA, saved.PREPARED, saved.CACHE_SOURCE, saved.CACHE_ROOT
SLOTS = {key: dict(saved.SLOTS[key]) for key in ARM_SLOTS.values()}
_prepare, _controller, _entry, _worker, _commands = base.prepare, base.controller, base.entry, base.worker, base.commands


def self_command(stage, spec=None, slot=None, token=None):
    command = [sys.executable, '-B', '-u', '-m', 'code.V4.r87_launch', stage]
    for flag, value in (('--spec', spec), ('--slot', slot), ('--token', token)):
        if value is not None:
            command += [flag, str(value)]
    return command


def container_command(stage, slot, spec=None, token=None):
    command = saved._container_command(stage, slot, spec, token)
    command[command.index(str(REPO / 'code/V4/run_v4_r84_launch.sh'))] = str(WRAPPER)
    return command


def load_spec(path):
    value = saved._load_spec(path)
    arm = value['run_id']
    if arm not in ARM_SLOTS or value['slots'] != [ARM_SLOTS[arm]]:
        raise ValueError('Each R87 arm has one fixed existing 4090 UUID; no slot fallback')
    expected = dict(data_root=DATA, prepared_root=PREPARED, cache_source=CACHE_SOURCE,
                    cache_root=CACHE_ROOT, run_dir=MODEL_ROOT / arm)
    if any(Path(value[key]) != path for key, path in expected.items()):
        raise ValueError('Use the frozen NEW90k cache/preparation and isolated R87 output')
    controls = dict(seed=2, logical_batch=1024, max_successful_updates=57600,
                    warmup_updates=4320, evaluation_every_updates=1440,
                    evaluation_batch=256, large_instance_micro_cap=352,
                    initialization='scratch', training_splits=['train', 'val'])
    if any(value.get('protocol', {}).get(key) != expected for key, expected in controls.items()):
        raise ValueError('R87 launch spec differs from the locked matched controls')
    return value


def predecessor_gate(arm):
    previous = PREDECESSORS[arm]
    directory = R85_LAUNCH / previous
    required = {
        'controller': (directory / 'controller.json', ('allocation_GPU_pipeline', 'PDF')),
        'worker': (directory / 'worker.json', ('train_VALlock_TEST', 'LIB')),
    }
    result = dict(predecessor=previous, complete=False, failed=False, records={})
    for name, (path, names) in required.items():
        if not path.is_file():
            result['reason'] = 'waiting for predecessor receipt: ' + str(path)
            return result
        record = base.read(path)
        result['records'][name] = dict(path=str(path), state=record.get('state'),
            exit_status=record.get('exit_status'), stages=record.get('stages', []))
        if record.get('state') == 'failed':
            result.update(failed=True, reason='predecessor failed; no automatic repair or retry')
            return result
        stages = {stage['name']: stage for stage in record.get('stages', [])}
        if (record.get('state') != 'complete' or record.get('exit_status') != 0
                or any(name not in stages or stages[name].get('state') != 'complete'
                       or stages[name].get('exit_status') != 0 for name in names)):
            result['reason'] = 'waiting for predecessor complete train/TEST/LIB/PDF rc0'
            return result
    path = R85_MODELS / previous / 'postprocess_status.json'
    if not path.is_file() or base.read(path).get('phase') != 'complete':
        result['reason'] = 'waiting for predecessor completed report receipt'
        return result
    result.update(complete=True, reason='predecessor full delivery complete rc0', report_receipt=str(path))
    return result


def prepare(path):
    result = _prepare(path)
    for source in (Path(__file__).resolve(), WRAPPER, REPO / 'code/V4/r85_launch.py', R85_LAUNCH / 'resources.json'):
        result['source_sha256'][str(source)] = base.sha(source)
    result.update(cache_mode='existing_read_only', main_thread_ready=False,
                  predecessor=predecessor_gate(result['spec']['run_id']))
    base.save(base.receipts(result['spec']) / 'plan.json', result)
    return result


def controller(path):
    spec, _ = base.check_ready(path)
    output = base.receipts(spec)
    with base.lock(output / 'predecessor_wait.lock'):
        while True:
            gate = predecessor_gate(spec['run_id'])
            if gate['failed']:
                base.save(output / 'predecessor_failure.json', gate, exclusive=True)
                raise RuntimeError(gate['reason'])
            if gate['complete']:
                base.save(output / 'predecessor_complete.json', gate, exclusive=True)
                break
            base.save(output / 'status.json', dict(state='waiting_predecessor_delivery',
                updated_utc=base.utc(), training_started=False, predecessor=gate,
                tmux=base.session_name(spec)))
            time.sleep(30)
        return _controller(path)


def require_predecessor(path):
    spec = load_spec(path)
    gate = predecessor_gate(spec['run_id'])
    if not gate['complete']:
        raise RuntimeError(gate['reason'])


def entry(path, slot, token):
    require_predecessor(path)
    return _entry(path, slot, token)


def wandb_lookup():
    record = dict(enabled=True, project='selector', entity='yjkds-southern-university-of-science-technology',
                  checked_utc=base.utc(), lookup='standard W&B API: environment/settings/netrc')
    try:
        import wandb
        from wandb.sdk.lib import apikey
        credential = apikey.api_key(settings=wandb.Settings())
        record.update(credential_present=bool(credential), selected_mode='online' if credential else 'offline')
        if not credential:
            record['offline_reason'] = 'No existing W&B credential found; no login prompt was started'
    except Exception as error:
        record.update(credential_present=False, selected_mode='offline',
                      offline_reason='Standard credential lookup failed: ' + type(error).__name__)
    return record


def commands(spec, slot):
    resolved = dict(spec, wandb_mode=os.environ.get('R87_RESOLVED_WANDB_MODE', 'auto'))
    return _commands(resolved, slot)


def worker(path, slot, token):
    require_predecessor(path)
    base.assert_container(slot)
    spec, _ = base.check_ready(path)
    claim = base.read(base.receipts(spec) / 'claim.json')
    if claim['slot'] != slot or claim['token'] != token or os.environ.get(TOKEN_ENV) != token:
        raise RuntimeError('R87 worker requires its unique claimed slot/token')
    os.sched_setaffinity(0, claim['cpus'])
    lookup = wandb_lookup()
    base.save(base.receipts(spec) / 'wandb_lookup.json', lookup, exclusive=True)
    os.environ['R87_RESOLVED_WANDB_MODE'] = lookup['selected_mode']
    os.environ['WANDB_MODE'] = lookup['selected_mode']
    return _worker(path, slot, token)


def template(arm):
    if arm not in ARM_SLOTS:
        raise ValueError('Choose one of the two main-locked R87 arms')
    source_names = ('r87_experiment.py', 'r87_model.py', 'r87_reporting.py',
        'r85_experiment.py', 'r85_reporting.py', 'r84_experiment.py', 'r84_reporting.py',
        'r84_prepare.py', 'r76_experiment.py', 'r77_experiment.py', 'r80_experiment.py',
        'r80_reporting.py', 'r77_best_model_report.py', 'pairwise_selector.py',
        'pairwise_objective.py', 'solver_code_encoder.py')
    sources = [str(REPO / 'code/V4' / name) for name in source_names]
    sources += [str(PREPARED / name) for name in ('prepared.json', 'geometry_stats.json',
        'common_full_train_sbs.json', 'identicalinit_seed2.pt')]
    sources += [str(MODEL_ROOT / 'protocol_lock.json')]
    return dict(schema_version=1, run_id=arm, run_dir=str(MODEL_ROOT / arm), common_root=str(MODEL_ROOT),
        data_root=str(DATA), prepared_root=str(PREPARED), cache_source=str(CACHE_SOURCE),
        cache_root=str(CACHE_ROOT), slots=[ARM_SLOTS[arm]], cpu_count=4, memory_gib=9,
        wandb_mode='auto', source_files=sources,
        protocol=dict(initialization='scratch', training_splits=['train', 'val'], seed=2,
            logical_batch=1024, max_successful_updates=57600, warmup_updates=4320,
            evaluation_every_updates=1440, evaluation_batch=256, large_instance_micro_cap=352),
        stages=[dict(name='train_VALlock_TEST', device='gpu', argv=[
            '{python}', '-B', '-u', '-m', 'code.V4.r87_experiment', 'train',
            '--root', '{common_root}', '--arm', arm, '--prepared-root', '{prepared_root}',
            '--data-root', '{data_root}', '--cache-root', '{cache_root}',
            '--protocol-lock', '{common_root}/protocol_lock.json', '--gpu-type', '4090',
            '--large-micro-cap', '352', '--eval-batch-size', '256', '--device', '{device}',
            '--wandb-mode', '{wandb_mode}']),
            dict(name='LIB', device='gpu', argv=['{python}', '-B', '-u', '-m',
                'code.V4.r87_reporting', 'lib', '--root', '{common_root}', '--arm', arm,
                '--device', '{device}', '--lib-batch-size', '2']),
            dict(name='PDF', device='cpu', argv=['{python}', '-B', '-u', '-m',
                'code.V4.r87_reporting', 'report', '--root', '{common_root}', '--arm', arm,
                '--device', '{device}'])])


def write_specs():
    runner_path = REPO / 'code/V4/r87_experiment.py'
    reporter_path = REPO / 'code/V4/r87_reporting.py'
    runner = importlib.import_module('code.V4.r87_experiment') if runner_path.is_file() else None
    reporter = importlib.import_module('code.V4.r87_reporting') if reporter_path.is_file() else None
    records = []
    for arm, slot in ARM_SLOTS.items():
        value = template(arm)
        path = ROOT / 'specs' / (arm + '.json')
        if path.is_file() and base.read(path) != value:
            raise RuntimeError('Existing R87 spec differs; main must explicitly review the change')
        if not path.is_file():
            base.save(path, value, exclusive=True)
        load_spec(path)
        stages = commands(value, slot)
        if runner is not None:
            for mode in ('train', 'test'):
                argv = [mode, *stages[0]['argv'][6:]]
                parsed = runner.parser().parse_args(argv)
                if (parsed.arm != arm or parsed.gpu_type != '4090' or parsed.large_micro_cap != 352
                        or parsed.eval_batch_size != 256 or parsed.cache_root != CACHE_ROOT
                        or getattr(parsed, 'resume', False)):
                    raise RuntimeError('R87 runner parsed controls differ from the locked protocol')
                records.append(dict(arm=arm, command=mode, argv=argv, parsed=True))
        if reporter is not None:
            for stage in stages[1:]:
                argv = stage['argv'][5:]
                parsed = reporter.parser().parse_args(argv)
                if parsed.arm != arm or parsed.device != ('cuda:0' if stage['device'] == 'gpu' else 'cpu'):
                    raise RuntimeError('R87 reporter parsed controls differ')
                records.append(dict(arm=arm, command=parsed.command, argv=argv, parsed=True))
    record = dict(created_utc=base.utc(), state='CLI_parsed' if runner and reporter else 'waiting_CLI_owner',
        runner_parser_present=runner is not None, reporter_parser_present=reporter is not None,
        commands=records, parsed_count=len(records), source_hashes_frozen=False,
        main_thread_ready=False, training_started=False,
        predecessor_gates={arm: predecessor_gate(arm) for arm in ARM_SLOTS})
    base.save(ROOT / 'spec_parse.json', record)
    return record


base.ROOT = ROOT
base.SLOTS = SLOTS
base.TOKEN_ENV = TOKEN_ENV
base.self_command = self_command
base.container_command = container_command
base.load_spec = load_spec
base.cache_manifest = saved.cache_manifest
base.stage_cache = saved.stage_cache
base.commands = commands
base.session_name = lambda value: 'r87_' + value['run_id']


def self_test():
    import tempfile
    import unittest
    from unittest import mock

    class Tests(unittest.TestCase):
        def test_two_fixed_4090_slots_only(self):
            self.assertEqual(set(SLOTS), {'4090_0', '4090_1'})
            self.assertEqual(len({value['uuid'] for value in SLOTS.values()}), 2)
            for arm, slot in ARM_SLOTS.items():
                value = template(arm)
                command = commands(value, slot)[0]['argv']
                self.assertEqual(value['slots'], [slot])
                self.assertEqual(command[command.index('--large-micro-cap') + 1], '352')
                self.assertEqual(command[command.index('--eval-batch-size') + 1], '256')

        def test_no_historical_namespace_mutation(self):
            from . import r85_launch as historical
            self.assertEqual(historical.base.ROOT, R85_LAUNCH)
            self.assertEqual(len(historical.base.SLOTS), 4)
            self.assertEqual(historical.base.TOKEN_ENV, 'R85_LAUNCH_TOKEN')

        def test_container_cgroup_route_and_CPU_only_step(self):
            for slot in SLOTS:
                argv = container_command('worker', slot, '/shared/spec.json', 'token')
                self.assertIn(str(WRAPPER), argv)
                self.assertIn('CUDA_VISIBLE_DEVICES=' + SLOTS[slot]['uuid'], argv)
                self.assertIn('R87_LAUNCH_TOKEN=token', argv)
                self.assertIn(base.transport.CONTAINERS['gpu05'][1], argv)
                step = base.srun_command('entry', slot, cpus=4, memory='9G')
                self.assertIn('--jobid=1475', step)
                self.assertIn('--gres=none', step)
                self.assertNotIn('ssh', step)

        def test_main_ready_required_before_tmux(self):
            with mock.patch.object(base.socket, 'gethostname', return_value='admin01'), \
                    mock.patch.object(Path, 'is_file', return_value=True), \
                    mock.patch.object(base.subprocess, 'run') as run:
                with self.assertRaisesRegex(RuntimeError, 'explicit ready'):
                    base.start(Path('/missing/spec.json'), '', False)
                run.assert_not_called()

        def test_predecessor_requires_report_rc0_not_just_train_exit(self):
            with tempfile.TemporaryDirectory() as directory:
                folder = Path(directory)
                with mock.patch.dict(globals(), R85_LAUNCH=folder / 'launch', R85_MODELS=folder / 'models'):
                    arm = 'D_candidate_subset'
                    base.save(R85_LAUNCH / arm / 'worker.json', dict(state='complete', exit_status=0,
                        stages=[dict(name=name, state='complete', exit_status=0) for name in ('train_VALlock_TEST', 'LIB')]))
                    path = R85_LAUNCH / arm / 'controller.json'
                    base.save(path, dict(state='running', stages=[]))
                    self.assertFalse(predecessor_gate('B_private_residual')['complete'])
                    base.save(path, dict(state='complete', exit_status=0,
                        stages=[dict(name=name, state='complete', exit_status=0) for name in ('allocation_GPU_pipeline', 'PDF')]))
                    self.assertFalse(predecessor_gate('B_private_residual')['complete'])
                    base.save(R85_MODELS / arm / 'postprocess_status.json', dict(phase='complete'))
                    self.assertTrue(predecessor_gate('B_private_residual')['complete'])
                    base.save(path, dict(state='failed', exit_status=1))
                    self.assertTrue(predecessor_gate('B_private_residual')['failed'])

        def test_cache_reuse_no_copy_or_array_read(self):
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                for split in ('train', 'val'):
                    (root / split).mkdir()
                base.save(root / '.r84_cache_identity.json', dict(data_root=str(DATA), source=str(CACHE_SOURCE)))
                manifest = dict(files=[dict(path='train/TSP/node.npy', bytes=99)], bytes=99, identity_sha256='existing')
                with mock.patch.object(base.shutil, 'copyfile', side_effect=AssertionError('no cache copy')):
                    result = saved.stage_cache(dict(cache_root=str(root), data_root=str(DATA), cache_source=str(CACHE_SOURCE)), manifest)
                self.assertEqual(result['copied_bytes'], 0)
                self.assertFalse(result['cache_audit_repeated'])

        def test_standard_wandb_missing_credential_is_explicit_offline(self):
            import types
            apikey = types.SimpleNamespace(api_key=lambda **kw: None)
            fake = types.SimpleNamespace(Settings=lambda: object())
            with mock.patch.dict(sys.modules, {'wandb': fake, 'wandb.sdk.lib': types.SimpleNamespace(apikey=apikey)}):
                value = wandb_lookup()
            self.assertEqual(value['selected_mode'], 'offline')
            self.assertIn('offline_reason', value)

        def test_CPU_report_hides_CUDA_and_two_sessions_are_unique(self):
            self.assertEqual(base.environment()['CUDA_VISIBLE_DEVICES'], '')
            self.assertEqual(len({base.session_name(template(arm)) for arm in ARM_SLOTS}), 2)
            for arm, slot in ARM_SLOTS.items():
                self.assertEqual(commands(template(arm), slot)[-1]['argv'][-1], 'cpu')

    result = unittest.TextTestRunner(verbosity=1).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests))
    record = dict(completed_utc=base.utc(), passed=result.testsRun - len(result.failures) - len(result.errors),
        failures=len(result.failures), errors=len(result.errors), training_started=False, GPU_context_created=False)
    base.save(ROOT / 'launcher_CPU_tests.json', record)
    return result.wasSuccessful()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('template', 'specs', 'self-test', 'prepare', 'start', 'controller',
        'node-resources', 'container-resources', 'entry', 'worker', 'gates'))
    parser.add_argument('--spec', type=Path)
    parser.add_argument('--arm', choices=tuple(ARM_SLOTS), default='A_shared_residual')
    parser.add_argument('--slot', choices=tuple(SLOTS))
    parser.add_argument('--token')
    parser.add_argument('--main-ready-evidence')
    parser.add_argument('--runner-tests-passed', action='store_true')
    args = parser.parse_args(argv)
    if args.stage in ('prepare', 'start', 'controller', 'entry', 'worker') and not args.spec:
        parser.error('--spec required')
    if args.stage in ('node-resources', 'container-resources', 'entry', 'worker') and not args.slot:
        parser.error('--slot required')
    if args.stage == 'self-test':
        return 0 if self_test() else 1
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
        return controller(args.spec)
    elif args.stage == 'node-resources':
        value = saved._node_resources(args.slot)
    elif args.stage == 'container-resources':
        value = base.container_resources(args.slot)
    elif args.stage == 'entry':
        return entry(args.spec, args.slot, args.token)
    else:
        return worker(args.spec, args.slot, args.token)
    print(json.dumps(value, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
