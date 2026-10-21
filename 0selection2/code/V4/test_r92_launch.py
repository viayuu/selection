"""CPU-only R92 plumbing tests. No real runtime, resource probe, or launch."""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest import mock

from . import r92_launch as launch


def protocol_fixture():
    return dict(experiment='R92', locked=True, arms=list(launch.ARMS), seed=2,
        logical_global_batch=1024, task_scope='full18', data_root=str(launch.DATA),
        prepared_root=str(launch.PREPARED), rows_per_problem=dict(train=81000, val=4500, test=4500),
        warmup_updates=4320, eval_every=1440,
        arm_budgets=dict(A=57600, B=57600, C=3000, D=3000),
        large_micro_cap=352, eval_batch_size=256)


def parser_stub(runner):
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=('train', 'test') if runner else ('lib', 'report'))
    paths = ('root', 'prepared-root', 'data-root', 'cache-root', 'protocol-lock') if runner else ('root',)
    for name in paths:
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--arm', choices=launch.ARMS, required=True)
    parser.add_argument('--device', required=True)
    if runner:
        parser.add_argument('--gpu-type', choices=('3090', '4090'), required=True)
        parser.add_argument('--large-micro-cap', type=int, required=True)
        parser.add_argument('--eval-batch-size', type=int, required=True)
        parser.add_argument('--wandb-mode', choices=('offline',), required=True)
        parser.add_argument('--resume', action='store_true')
    else:
        parser.add_argument('--lib-batch-size', type=int, default=2)
    return parser


def runtime_stubs():
    def forbidden(*_):
        raise AssertionError('Do not generate a protocol, train, or report from parser checks')
    return [SimpleNamespace(ARMS=launch.ARMS, parser=lambda: parser_stub(True),
                            protocol_lock=forbidden, train=forbidden, test=forbidden),
            SimpleNamespace(ARMS=launch.ARMS, parser=lambda: parser_stub(False),
                            lib=forbidden, report=forbidden)]


class R92LaunchTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        # Every test fails closed if an unmocked external action slips through.
        for target in ('run', 'Popen'):
            patcher = mock.patch.object(launch.base.subprocess, target,
                side_effect=AssertionError('No external command in focused CPU tests'))
            patcher.start()
            self.addCleanup(patcher.stop)

    def final_lock(self, value=None):
        path = self.root / 'protocol_lock.json'
        launch.base.save(path, protocol_fixture() if value is None else value)
        patcher = mock.patch.object(launch, 'PROTOCOL_LOCK', path)
        patcher.start()
        self.addCleanup(patcher.stop)
        return path

    def predecessor_fixture(self):
        models = dict(R91=self.root / 'R91_models', R89=self.root / 'R89_models')
        launches = dict(R91=self.root / 'R91_launch', R89=self.root / 'R89_launch')
        recovery = models['R91'] / 'report_only_recovery_complete.json'
        for name, value in (('PREVIOUS_MODELS', models), ('PREVIOUS_LAUNCH', launches),
                            ('R91_RECOVERY_EVIDENCE', recovery)):
            patcher = mock.patch.object(launch, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        protected, reports, failures = {}, {}, {}
        for previous in (launch.PREDECESSORS['A'], launch.PREDECESSORS['C']):
            directory, model = launches['R91'] / previous, models['R91'] / previous
            worker = dict(state='failed', exit_status=1, stages=[
                dict(name=name, state='complete', exit_status=0) for name in ('train', 'TEST', 'LIB')]
                + [dict(name='PDF', state='failed', exit_status=1)])
            launch.base.save(directory / 'worker.json', worker)
            launch.base.save(directory / 'controller.json', dict(state='failed', exit_status=1,
                stages=[dict(name='allocation_GPU_pipeline', state='failed', exit_status=1)]))
            (directory / 'PDF.log').write_text('FileNotFoundError: NotoSansCJK-Regular.ttc\n')
            model.mkdir(parents=True)
            (model / 'best.pt').write_bytes(b'synthetic unchanged checkpoint')
            lock = dict(checkpoint=str(model / 'best.pt'), checkpoint_sha256=launch.base.sha(model / 'best.pt'))
            launch.base.save(model / 'final_test_lock.json', lock)
            failure = dict(phase='postprocess_failed', error='missing NotoSansCJK-Regular.ttc')
            launch.base.save(model / 'postprocess_status.json', failure)
            failures[previous] = dict(original_postprocess=failure)
            pdf = model / 'full_model_statistics/MODEL_STATISTICS.pdf'
            pdf.parent.mkdir()
            pdf.write_bytes(b'%PDF-synthetic-test')
            reports[previous] = dict(checkpoint_lock=lock, MODEL_STATISTICS_pdf=str(pdf))
            for path in [directory / name for name in ('worker.json', 'controller.json', 'PDF.log')] + [
                    model / name for name in ('best.pt', 'final_test_lock.json', 'postprocess_status.json')]:
                protected[str(path)] = launch.base.sha(path)
        commands = [dict(exit_status=0, argv=['stub', command]) for command in ('report', 'report', 'compare')]
        flags = dict.fromkeys(('training', 'inference', 'test_rerun', 'lib_rerun',
            'checkpoint_reselection', 'automatic_retry', 'frozen_reporter_modified'), False)
        detail_path = models['R91'] / 'report_only_recovery/manual/recovery_receipt.json'
        detail = dict(experiment='R91', complete=True, state='recovered_report_only', host='admin01',
            recovery_kind='report_only', protected_unchanged=True, original_failures=failures,
            protected_sha256_before=protected, protected_sha256_after=protected, commands=commands, **flags)
        launch.base.save(detail_path, detail)
        receipt = dict(experiment='R91', complete=True, state='recovered_report_only', host='admin01',
            old_failure_unchanged=True, evaluation_unchanged=True, recovery_receipt=str(detail_path),
            both_arm_report_paths=reports, protected_sha256_before=protected,
            protected_sha256_after=protected, report_commands=commands, **flags)
        launch.base.save(recovery, receipt)
        for previous in (launch.PREDECESSORS['B'], launch.PREDECESSORS['D']):
            directory, model = launches['R89'] / previous, models['R89'] / previous
            for name, stages in (('worker', ('train_VALlock_TEST', 'LIB')),
                                  ('controller', ('allocation_GPU_pipeline', 'PDF'))):
                launch.base.save(directory / (name + '.json'), dict(state='complete', exit_status=0,
                    stages=[dict(name=stage, state='complete', exit_status=0) for stage in stages]))
            launch.base.save(model / 'postprocess_status.json', dict(phase='complete', complete=True))
        return models, launches, recovery, detail_path

    def test_four_generic_fixed_slots_original_jobs_and_containers(self):
        self.assertEqual(launch.ARM_SLOTS, dict(A='3090_0', B='4090_0', C='3090_1', D='4090_1'))
        self.assertEqual(len({slot['uuid'] for slot in launch.SLOTS.values()}), 4)
        for arm, slot in launch.ARM_SLOTS.items():
            spec, item = launch.template(arm), launch.SLOTS[slot]
            self.assertEqual(spec['slots'], [slot])
            self.assertEqual((spec['cpu_count'], spec['memory_gib']), (4, 9))
            self.assertEqual((item['job'], item['node']),
                             (1465, 'gpu03') if arm in ('A', 'C') else (1475, 'gpu05'))
            self.assertEqual(launch.base.session_name(spec), 'r92_' + arm)
            self.assertNotIn('cap', item)
        self.assertEqual(launch.MODEL_ROOT.name, 'R92_broad_learning')

    def test_template_before_science_or_runtime_has_no_side_effects(self):
        with mock.patch.object(launch, 'runtime_modules', side_effect=AssertionError('not final')), \
                mock.patch.object(launch.base, 'save', side_effect=AssertionError('no write')):
            for arm in launch.ARMS:
                spec = launch.template(arm)
                self.assertEqual(spec['protocol']['arm_budget']['count'], 'MAIN_SUPPLIES_ARM_BUDGET')
                argv = launch.commands(spec, launch.ARM_SLOTS[arm])[0]['argv']
                self.assertEqual(argv[argv.index('--large-micro-cap') + 1], '352')
                self.assertEqual(argv[argv.index('--eval-batch-size') + 1], '256')

    def test_NN_and_tree_budget_units_are_not_equalized(self):
        value = protocol_fixture()
        self.assertEqual([launch.arm_budget(value, arm)['count'] for arm in launch.ARMS],
                         [57600, 57600, 3000, 3000])
        self.assertEqual([launch.arm_budget(value, arm)['unit'] for arm in launch.ARMS],
                         ['successful_updates', 'successful_updates', 'trees_per_task', 'trees_per_task'])
        self.assertNotIn('warmup_updates', launch.arm_budget(value, 'C'))
        value['max_updates'] = 57600
        del value['arm_budgets']['C']
        with self.assertRaisesRegex(ValueError, 'tree budget'):
            launch.arm_budget(value, 'C')

    def test_budget_descriptions_and_aliases(self):
        value = protocol_fixture()
        value['arm_budgets'] = dict(A=dict(max_updates=500, warmup_updates=50, eval_every=25),
            B=dict(max_successful_updates=700, warmup_updates=70, evaluation_every_updates=35),
            C=dict(iterations=30), D=dict(trees_per_task=40))
        self.assertEqual([launch.arm_budget(value, arm)['count'] for arm in launch.ARMS], [500, 700, 30, 40])

    def test_final_lock_requires_full18_frozen_data_seed_and_batch(self):
        self.final_lock()
        self.assertEqual(launch.locked_protocol()['arm_budgets']['C'], 3000)
        for key, bad in (('locked', False), ('seed', 1), ('logical_global_batch', 512),
                         ('task_scope', 'subset'), ('data_root', '/old/data'),
                         ('arms', ['A', 'B']), ('rows_per_problem', dict(train=8100))):
            value = protocol_fixture()
            value[key] = bad
            launch.base.save(launch.PROTOCOL_LOCK, value)
            with self.subTest(key=key), self.assertRaises(ValueError):
                launch.locked_protocol()

    def test_conflicting_batch_aliases_and_invalid_budget_fail_closed(self):
        value = protocol_fixture()
        value['batch_size'] = 512
        self.final_lock(value)
        with self.assertRaisesRegex(ValueError, 'Conflicting'):
            launch.locked_protocol()
        for count in (0, -1, True, '3000'):
            value = protocol_fixture()
            value['arm_budgets']['D'] = dict(iterations=count)
            launch.base.save(launch.PROTOCOL_LOCK, value)
            with self.subTest(count=count), self.assertRaises(ValueError):
                launch.locked_protocol()

    def test_per_arm_protocol_caps_override_defaults_not_static_slots(self):
        value = protocol_fixture()
        value['arm_descriptions'] = dict(B=dict(large_micro_cap=176, eval_batch_size=128))
        self.final_lock(value)
        locked = launch.locked_protocol()
        for arm in launch.ARMS:
            spec = launch.template(arm, locked)
            argv = launch.commands(spec, launch.ARM_SLOTS[arm])[0]['argv']
            self.assertEqual(argv[argv.index('--large-micro-cap') + 1], '176' if arm == 'B' else '352')
            self.assertEqual(argv[argv.index('--eval-batch-size') + 1], '128' if arm == 'B' else '256')

    def test_all_16_positional_command_contracts_parse_with_stubs(self):
        self.final_lock()
        with mock.patch.object(launch, 'runtime_modules', return_value=runtime_stubs()):
            result = launch.parse_commands()
        self.assertEqual(result['parsed_count'], 16)
        self.assertFalse(result['protocol_written'])
        self.assertFalse(result['source_hashes_frozen'])
        self.assertEqual([row['command'] for row in result['commands']], ['train', 'test', 'lib', 'report'] * 4)
        self.assertFalse(any('--resume' in row['argv'] for row in result['commands']))

    def test_parser_mismatch_is_not_silently_accepted(self):
        stubs = runtime_stubs()
        parser = parser_stub(True)
        parser.set_defaults(resume=True)
        stubs[0].parser = lambda: parser
        with mock.patch.object(launch, 'runtime_modules', return_value=stubs), self.assertRaisesRegex(RuntimeError, 'resume'):
            launch.parse_commands(protocol_fixture())

    def test_missing_runtime_never_freezes_or_writes_plan(self):
        with mock.patch.object(launch, 'locked_protocol', return_value=protocol_fixture()), \
                mock.patch.object(launch, 'runtime_modules', side_effect=RuntimeError('Waiting for final runtime')), \
                mock.patch.object(launch.engine, '_prepare') as prepare:
            with self.assertRaisesRegex(RuntimeError, 'final runtime'):
                launch.prepare(self.root / 'spec.json', 'main ready', True)
            prepare.assert_not_called()

    def test_main_ready_and_CPU_test_confirmation_precede_any_spec_or_tmux(self):
        with mock.patch.object(launch, '_start') as start, \
                mock.patch.object(launch, 'parse_commands') as parse, \
                mock.patch.object(launch.base, 'save') as save:
            for evidence, tested in (('', False), ('', True), ('main ready', False), (' ', True)):
                for action in (lambda: launch.start(self.root / 'spec.json', evidence, tested),
                               lambda: launch.write_specs(evidence, tested),
                               lambda: launch.prepare(self.root / 'spec.json', evidence, tested)):
                    with self.assertRaisesRegex(RuntimeError, 'explicit ready'):
                        action()
            start.assert_not_called()
            parse.assert_not_called()
            save.assert_not_called()

    def test_spec_write_is_explicit_and_refuses_existing_changes(self):
        self.final_lock()
        with mock.patch.object(launch, 'ROOT', self.root / 'launch'), \
                mock.patch.object(launch, 'runtime_modules', return_value=runtime_stubs()):
            result = launch.write_specs('main reviewed final runtime', True)
            self.assertEqual(len(result['specs']), 4)
            path = Path(result['specs'][0])
            value = launch.base.read(path)
            value['cpu_count'] = 8
            launch.base.save(path, value)
            with self.assertRaisesRegex(RuntimeError, 'overwrite'):
                launch.write_specs('main reviewed final runtime', True)

    def test_exact_spec_rejects_GPU_PDF_and_wrong_slot(self):
        self.final_lock()
        spec = launch.template('A', launch.locked_protocol())
        path = self.root / 'spec.json'
        launch.base.save(path, spec)
        self.assertEqual(launch.load_spec(path), spec)
        spec['stages'][-1]['device'] = 'gpu'
        launch.base.save(path, spec)
        with self.assertRaisesRegex(ValueError, 'CPU PDF'):
            launch.load_spec(path)
        with self.assertRaisesRegex(ValueError, 'fixed slots'):
            launch.commands(launch.template('A'), '3090_1')

    def test_allocation_CPU_step_then_exact_container_UUID_no_SSH(self):
        for slot, item in launch.SLOTS.items():
            step = launch.base.srun_command('entry', slot, '/shared/spec.json', 'token', cpus=4, memory='9G')
            for flag in (f'--jobid={item["job"]}', f'--nodelist={item["node"]}', '--overlap',
                         '--exact', '--gres=none', '--cpus-per-task=4', '--mem=9G'):
                self.assertIn(flag, step)
            self.assertIn('conda activate easynco', step[-1])
            self.assertIn('code.V4.r92_launch entry', step[-1])
            command = launch.base.container_command('worker', slot, '/shared/spec.json', 'token')
            self.assertEqual(command[:4], ['sudo', '-n', '/usr/bin/ai_docker', 'exec'])
            self.assertIn(launch.transport.CONTAINERS[item['node']][1], command)
            self.assertIn('CUDA_VISIBLE_DEVICES=' + item['uuid'], command)
            self.assertIn('R92_LAUNCH_TOKEN=token', command)
            self.assertIn(str(launch.WRAPPER), command)
            self.assertFalse(any(word in step or word in command for word in ('ssh', 'sbatch', 'salloc')))

    def test_private_helpers_never_mutate_frozen_modules(self):
        from . import r84_launch, r85_launch, r87_launch, r89_launch, r91_launch, r81_queue
        for name, module in (('R84', r84_launch), ('R85', r85_launch.base),
                             ('R87', r87_launch.base), ('R89', r89_launch.base), ('R91', r91_launch.base)):
            self.assertIsNot(module, launch.base)
            self.assertEqual(module.TOKEN_ENV, name + '_LAUNCH_TOKEN')
        self.assertIsNot(launch.transport, r81_queue)
        self.assertIsNot(launch.transport.capture, r81_queue.capture)
        self.assertEqual(launch.engine.saved.ROOT, launch.REPO / 'code/V4/runs/R85_launch')

    def test_ai_docker_preamble_and_stderr_do_not_corrupt_JSON(self):
        for newline in ('\n', '\r\n'):
            stdout = newline.join(('Distribute Task Container.', 'task_id:1465', '{"ok":true}'))
            result = SimpleNamespace(returncode=0, stdout=stdout, stderr='warning: {"not":"stdout"}')
            with mock.patch.object(launch.transport.subprocess, 'run', return_value=result) as run:
                self.assertEqual(json.loads(launch.capture(['mock-only'])), dict(ok=True))
                self.assertIs(run.call_args.kwargs['stderr'], launch.base.subprocess.PIPE)
        with mock.patch.object(launch.transport.subprocess, 'run', return_value=SimpleNamespace(
                returncode=1, stdout='{}', stderr='container error')):
            with self.assertRaisesRegex(RuntimeError, 'container error'):
                launch.capture(['mock-only'])

    def test_offline_environment_and_CPU_PDF_are_forced(self):
        with mock.patch.dict(launch.os.environ, {'WANDB_MODE': 'online', 'R87_RESOLVED_WANDB_MODE': 'online'}):
            for arm, slot in launch.ARM_SLOTS.items():
                stages = launch.commands(launch.template(arm), slot)
                self.assertEqual([stage['device'] for stage in stages], ['gpu', 'gpu', 'gpu', 'cpu'])
                self.assertEqual([stage['argv'][stage['argv'].index('--device') + 1] for stage in stages],
                                 ['cuda:0', 'cuda:0', 'cuda:0', 'cpu'])
                self.assertEqual(launch.environment(slot, 'token')['WANDB_MODE'], 'offline')
        self.assertEqual(launch.environment()['CUDA_VISIBLE_DEVICES'], '')

    def test_actual_R85_cache_evidence_and_final_tree_sources(self):
        sources = launch.template('C')['source_files']
        for name in ('r92_tree.py', 'r92_tree_features.py'):
            self.assertIn(str(launch.REPO / 'code/V4' / name), sources)
        self.assertEqual(launch.CACHE_EVIDENCE, launch.REPO / 'code/V4/runs/R85_launch/resources.json')
        value = dict(spec=dict(run_id='A'), source_sha256={})
        with mock.patch.object(launch, 'parse_commands'), \
                mock.patch.object(launch.engine, '_prepare', return_value=value), \
                mock.patch.object(launch.base, 'sha', return_value='source-hash'), \
                mock.patch.object(launch.base, 'save'), \
                mock.patch.object(launch, 'predecessor_gate', return_value=dict(complete=False)):
            result = launch.prepare(self.root / 'spec.json', 'final runtime ready', True)
        self.assertIn(str(launch.CACHE_EVIDENCE), result['source_sha256'])
        self.assertNotIn(str(launch.PREVIOUS_LAUNCH['R91'] / 'resources.json'), result['source_sha256'])
        self.assertFalse(result['automatic_retry'])

    def test_cache_reuse_never_copies_rebuilds_or_hashes_arrays(self):
        for split in ('train', 'val', 'test'):
            (self.root / split).mkdir()
        launch.base.save(self.root / '.r84_cache_identity.json',
                         dict(data_root=str(launch.DATA), source=str(launch.CACHE_SOURCE)))
        manifest = dict(files=[dict(path='train/TSP/nodes.npy', bytes=10)], bytes=10, identity_sha256='prior')
        with mock.patch.object(launch.base.shutil, 'copyfile', side_effect=AssertionError('no copy')), \
                mock.patch.object(launch.base, 'sha', side_effect=AssertionError('no hashing')):
            result = launch.base.stage_cache(dict(cache_root=str(self.root), data_root=str(launch.DATA),
                cache_source=str(launch.CACHE_SOURCE)), manifest)
        self.assertEqual(result['copied_bytes'], 0)
        self.assertFalse(result['preprocessing'])
        self.assertFalse(result['array_contents_read'])
        self.assertFalse(result['cache_audit_repeated'])

    def test_R91_train_complete_alone_is_not_pipeline_complete(self):
        _, _, receipt, _ = self.predecessor_fixture()
        receipt.unlink()
        for arm in ('A', 'C'):
            gate = launch.predecessor_gate(arm)
            self.assertFalse(gate['complete'])
            self.assertFalse(gate['failed'])
            self.assertIn('report-only recovery', gate['reason'])

    def test_explicit_R91_recovery_accepts_preserved_failed_receipts(self):
        _, launches, _, _ = self.predecessor_fixture()
        for arm in ('A', 'C'):
            gate = launch.predecessor_gate(arm)
            self.assertTrue(gate['complete'], gate['reason'])
            self.assertTrue(gate['recovery']['original_failures_preserved'])
            self.assertEqual(launch.base.read(launches['R91'] / launch.PREDECESSORS[arm] / 'worker.json')['state'], 'failed')

    def test_R91_training_failure_cannot_be_bypassed_by_recovery(self):
        _, launches, _, _ = self.predecessor_fixture()
        path = launches['R91'] / launch.PREDECESSORS['A'] / 'worker.json'
        value = launch.base.read(path)
        value['stages'][0].update(state='failed', exit_status=1)
        launch.base.save(path, value)
        gate = launch.predecessor_gate('A')
        self.assertTrue(gate['failed'])
        self.assertFalse(gate['complete'])

    def test_R91_changed_checkpoint_or_original_receipt_is_rejected(self):
        models, launches, _, _ = self.predecessor_fixture()
        for path in (models['R91'] / launch.PREDECESSORS['A'] / 'best.pt',
                     launches['R91'] / launch.PREDECESSORS['C'] / 'PDF.log'):
            path.write_bytes(b'changed')
        for arm in ('A', 'C'):
            gate = launch.predecessor_gate(arm)
            self.assertFalse(gate['complete'])
            self.assertTrue(gate['failed'])

    def test_R91_recovery_requires_completed_unchanged_report_only_detail(self):
        _, _, _, path = self.predecessor_fixture()
        original = launch.base.read(path)
        for key, bad in (('protected_unchanged', False), ('complete', False),
                         ('training', True), ('inference', True), ('checkpoint_reselection', True)):
            value = dict(original, **{key: bad})
            launch.base.save(path, value)
            gate = launch.predecessor_gate('A')
            self.assertFalse(gate['complete'], key)
            self.assertTrue(gate['failed'], key)

    def test_R89_full_report_rc0_required_for_both_4090_arms(self):
        models, launches, _, _ = self.predecessor_fixture()
        for arm in ('B', 'D'):
            self.assertTrue(launch.predecessor_gate(arm)['complete'])
            path = launches['R89'] / launch.PREDECESSORS[arm] / 'controller.json'
            value = launch.base.read(path)
            value.update(state='running', exit_status=None)
            value['stages'] = value['stages'][:1]
            launch.base.save(path, value)
            self.assertFalse(launch.predecessor_gate(arm)['complete'])
            value.update(state='failed', exit_status=1)
            launch.base.save(path, value)
            self.assertTrue(launch.predecessor_gate(arm)['failed'])

    def test_predecessor_gate_precedes_tmux_even_with_main_ready(self):
        with mock.patch.object(launch, 'load_spec', return_value=dict(run_id='A')), \
                mock.patch.object(launch, 'predecessor_gate', return_value=dict(complete=False, reason='waiting report')), \
                mock.patch.object(launch, '_start') as start:
            with self.assertRaisesRegex(RuntimeError, 'waiting report'):
                launch.start(self.root / 'spec.json', 'main ready', True)
            start.assert_not_called()

    def test_PDF_runs_on_ADMIN01_only_after_GPU_lock_release(self):
        spec, slot = launch.template('A'), '3090_0'
        lock_active, seen = [], []
        output = self.root / 'A'

        def gpu_controller(_):
            lock_active.append(True)
            launch.base.save(output / 'controller.json', dict(state='gpu_complete', stages=[], automatic_retry=False))
            lock_active.pop()
            seen.append('GPU_exit_and_UUID_unlock')
            return 0

        def report(stage, _, record, __, env):
            self.assertFalse(lock_active)
            self.assertEqual(env['CUDA_VISIBLE_DEVICES'], '')
            self.assertEqual(stage['device'], 'cpu')
            self.assertEqual(stage['argv'][-1], 'cpu')
            self.assertNotIn('ai_docker', stage['argv'])
            seen.append(stage['name'])

        with mock.patch.object(launch.base, 'ROOT', self.root), \
                mock.patch.object(launch, 'require_predecessor'), \
                mock.patch.object(launch, '_controller', side_effect=gpu_controller), \
                mock.patch.object(launch, 'check_ready', return_value=(spec, {})), \
                mock.patch.object(launch.base.socket, 'gethostname', return_value='admin01'), \
                mock.patch.object(Path, 'is_file', return_value=True), \
                mock.patch.object(launch.base, 'run_stage', side_effect=report):
            self.assertEqual(launch.controller(self.root / 'spec.json'), 0)
        self.assertEqual(seen, ['GPU_exit_and_UUID_unlock', 'PDF'])

    def test_failed_GPU_stage_blocks_later_stages_without_retry(self):
        spec, slot, token = launch.template('C'), '3090_1', 'test-token'
        for failing in ('train', 'TEST', 'LIB'):
            folder = self.root / failing
            cpus = [12, 13, 14, 15]
            launch.base.save(folder / 'C/claim.json', dict(slot=slot, token=token, cpus=cpus))
            seen = []

            def run_stage(stage, *_):
                seen.append(stage['name'])
                if stage['name'] == failing:
                    raise RuntimeError('simulated stage failure')

            with mock.patch.object(launch.base, 'ROOT', folder), \
                    mock.patch.object(launch.base, 'check_ready', return_value=(spec, dict(cache={}))), \
                    mock.patch.object(launch.base, 'assert_container', return_value='owned'), \
                    mock.patch.object(launch.base, 'stage_cache', return_value=dict(copied_bytes=0)), \
                    mock.patch.object(launch.base, 'run_stage', side_effect=run_stage), \
                    mock.patch.object(launch.base.os, 'sched_getaffinity', return_value=set(cpus)), \
                    mock.patch.object(launch.base.os, 'sched_setaffinity'), \
                    mock.patch.dict(launch.os.environ, {launch.TOKEN_ENV: token,
                        'CUDA_VISIBLE_DEVICES': launch.SLOTS[slot]['uuid']}):
                with self.assertRaisesRegex(RuntimeError, 'simulated stage failure'):
                    launch._worker(self.root / 'spec.json', slot, token)
            expected = ['train', 'TEST', 'LIB']
            self.assertEqual(seen, expected[:expected.index(failing) + 1])
            record = launch.base.read(folder / 'C/worker.json')
            self.assertEqual(record['state'], 'failed')
            self.assertFalse(record['automatic_retry'])

    def test_failed_GPU_pipeline_never_enters_PDF(self):
        with mock.patch.object(launch, 'require_predecessor'), \
                mock.patch.object(launch, '_controller', side_effect=RuntimeError('GPU failed')), \
                mock.patch.object(launch.base, 'run_stage') as stage:
            with self.assertRaisesRegex(RuntimeError, 'GPU failed'):
                launch.controller(self.root / 'spec.json')
            stage.assert_not_called()

    def test_real_inherited_UUID_flock_is_released_before_CPU_report(self):
        spec, slot = launch.template('A'), '3090_0'
        output = self.root / 'launch/A'
        mutex = self.root / 'mutex' / ('queue_gpu_' + launch.SLOTS[slot]['uuid'] + '.json')
        probe = dict(gpus={slot: dict(compute_processes=[], memory_used_mib=0, utilization_pct=0)},
            processes=[], allowed_cpus=[12, 13, 14, 15],
            memory=dict(headroom_excluding_reclaimable_cache_bytes=20 * 2**30))
        seen = []

        def stage(command, _, record, __, env):
            seen.append(command['name'])
            if command['name'] == 'allocation_GPU_pipeline':
                with self.assertRaises(BlockingIOError), launch.base.lock(mutex, existing=True):
                    pass
                launch.base.save(output / 'worker.json', dict(state='complete', exit_status=0, stages=[
                    dict(name=name, state='complete', exit_status=0) for name in ('train', 'TEST', 'LIB')]))
            else:
                with launch.base.lock(mutex, existing=True):
                    self.assertEqual(command['name'], 'PDF')
                self.assertEqual(env['CUDA_VISIBLE_DEVICES'], '')

        with mock.patch.object(launch.base, 'ROOT', self.root / 'launch'), \
                mock.patch.object(launch, 'ROOT', self.root / 'launch'), \
                mock.patch.object(launch.transport, 'ROOT', self.root / 'mutex'), \
                mock.patch.object(launch.base.socket, 'gethostname', return_value='admin01'), \
                mock.patch.object(launch.base, 'FONT', launch.WRAPPER), \
                mock.patch.object(launch, 'require_predecessor'), \
                mock.patch.object(launch.base, 'check_ready', return_value=(spec, {})), \
                mock.patch.object(launch, 'check_ready', return_value=(spec, {})), \
                mock.patch.object(launch.transport, 'capture', return_value=json.dumps(probe)), \
                mock.patch.object(launch.base, 'run_stage', side_effect=stage):
            self.assertEqual(launch.controller(self.root / 'spec.json'), 0)
        self.assertEqual(seen, ['allocation_GPU_pipeline', 'PDF'])

    def test_CPU_reporting_does_not_reserve_allocation_CPUs(self):
        output = self.root / 'A'
        launch.base.save(output / 'claim.json', dict(node='gpu03', cpus=[12, 13, 14, 15]))
        launch.base.save(output / 'controller.json', dict(state='running', phase='PDF', gpu_finished_utc='done'))
        with mock.patch.object(launch, 'ROOT', self.root):
            self.assertEqual(launch.reserved_cpus('gpu03', self.root / 'B'), set())

    def test_wrapper_activates_environment_without_hidden_launch(self):
        wrapper = launch.WRAPPER.read_text()
        self.assertIn('conda activate easynco', wrapper)
        self.assertIn('code.V4.r92_launch "$@"', wrapper)
        self.assertNotIn('tmux', wrapper)
        self.assertNotIn('nvidia-smi', wrapper)


if __name__ == '__main__':
    unittest.main()
