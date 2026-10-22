"""Tiny CPU dispatch fixtures only. Resource probes, copy and starts are mocked."""

import copy
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from . import r101_launch as launch


def spec_fixture(root, arm='B'):
    model = root / 'R101_hint_pretraining'
    science = dict(experiment='R101', locked=True, seed=2, arms=['A', 'B'], task_scope='full18',
        current_query_labels_in_forward=False, data_root=str(root / 'data'),
        rows_per_problem=dict(train=81000, val=4500, test=4500), logical_global_batch=1024,
        hint_pretraining=dict(updates=12800), max_updates=57600)
    protocol = model / 'protocol_lock.json'
    launch.base.save(protocol, science)
    stages = []
    for name, command in (('pretrain', 'pretrain'), ('train', 'train'), ('TEST', 'test'),
                          ('timing', 'timing'), ('LIB', 'lib'), ('PDF', 'report')):
        module = 'code.V4.r101_reporting' if name in ('LIB', 'PDF') else 'code.V4.r101_experiment'
        argv = ['{python}', '-B', '-u', '-m', module, command,
                '--root', '{common_root}', '--arm', arm, '--device', '{device}']
        if name == 'pretrain':
            argv += ['--teacher-root', '{teacher_root}']
        if name == 'LIB':
            argv += ['--lib-batch-size', '2']
        stages.append(dict(name=name, device='cpu' if name == 'PDF' else 'gpu', argv=argv))
    spec = dict(schema_version=1, run_id='R101_'+arm, run_dir=str(model / arm), common_root=str(model),
        data_root=science['data_root'], prepared_root=str(root / 'prepared'),
        cache_source=str(root / 'shared_cache'), cache_root=str(launch.CACHE_ROOT),
        cache_kind='r84_inputs', science_protocol=str(protocol), slots=[launch.ARM_SLOTS[arm]],
        cpu_count=4, memory_gib=9, wandb_mode='offline', teacher_root=str(launch.TEACHER_ROOTS[arm]),
        source_files=[], stages=stages)
    if arm == 'A':
        spec['teacher_source'] = str(model / 'teacher_cache/A')
    path = root / ('R101_'+arm+'.json')
    launch.base.save(path, spec)
    return model, path, spec, science


class LaunchTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='r101-launch-cpu-')
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def test_private_rebinding_keeps_old_modules_and_shared_mutex_roots(self):
        from . import r98_launch, r84_launch, r81_queue
        self.assertIsNot(launch.adapter, r98_launch)
        self.assertIsNot(launch.base, r84_launch)
        self.assertIsNot(launch.base.transport, r81_queue)
        self.assertEqual(launch.ROOT, r98_launch.ROOT)
        self.assertEqual(launch.base.transport.ROOT, r81_queue.ROOT)
        self.assertEqual(launch.ARM_SLOTS, dict(A='4090_1', B='4090_0'))
        self.assertEqual(r98_launch.TOKEN_ENV, 'R98_DISPATCH_TOKEN')
        self.assertEqual(r84_launch.TOKEN_ENV, 'R84_LAUNCH_TOKEN')
        self.assertIs(launch.base.controller.__globals__['self_command'], launch.self_command)
        self.assertIs(launch.base.worker.__globals__['load_spec'], launch.load_spec)

    def test_all_entry_and_worker_children_use_r101_wrapper_and_allocation(self):
        for slot in launch.SLOTS:
            argv = launch.container_command('worker', slot, '/tmp/spec.json', 'token')
            self.assertIn(str(launch.WRAPPER), argv)
            self.assertNotIn('run_v4_r98_launch.sh', ' '.join(argv))
            self.assertIn(launch.base.transport.CONTAINERS[launch.SLOTS[slot]['node']][1], argv)
            command = launch.base.srun_command('entry', slot, '/tmp/spec.json', 'token', cpus=4, memory='9G')
            self.assertIn('code.V4.r101_launch', ' '.join(command))
            self.assertIn('--cpus-per-task=4', command)
            self.assertIn('--mem=9G', command)
            self.assertIn('--gres=none', command)
            self.assertEqual(launch.environment(slot, 'token')[launch.TOKEN_ENV], 'token')

    def test_strict_r101_spec_and_protocol_with_independent_b(self):
        model, path, original, science = spec_fixture(self.root)
        with patch.object(launch, 'MODEL_ROOT', model):
            spec = launch.load_spec(path)
            self.assertEqual(spec['slots'], ['4090_0'])
            with patch.object(launch, 'teacher_manifest', side_effect=AssertionError('B must not inspect A')):
                teacher = launch.prepare_teacher(spec)
            self.assertTrue(teacher['validate_existing_native16_on_worker'])
            self.assertEqual(len(launch.base.commands(spec, '4090_0')), 6)
            for change in (dict(slots=['4090_1']), dict(run_id='R100_A'), dict(cache_root='/tmp/new_cache')):
                launch.base.save(path, dict(original, **change))
                with self.assertRaises(ValueError):
                    launch.load_spec(path)
            launch.base.save(path, original)
            launch.base.save(model / 'protocol_lock.json', dict(science, hint_pretraining=dict(updates=12801)))
            with self.assertRaisesRegex(ValueError, 'Stage-I12800'):
                launch.load_spec(path)

    def test_main_ready_and_exact_stage_order_are_required(self):
        for evidence, tested in ((None, True), ('passed', False), ('', True)):
            with self.assertRaises(ValueError):
                launch.prepare('/nonexistent', evidence, tested)
            with self.assertRaises(ValueError):
                launch.start('/nonexistent', evidence, tested)
        model, path, spec, _ = spec_fixture(self.root)
        bad = copy.deepcopy(spec)
        bad['stages'][2]['argv'] += ['--teacher-root', '/wrong']
        launch.base.save(path, bad)
        with patch.object(launch, 'MODEL_ROOT', model), self.assertRaisesRegex(ValueError, 'Stage-I-only'):
            launch.load_spec(path)
        parsed = launch.parser().parse_args(['queue', '--spec', str(path), '--evidence', 'CPU tests', '--runner-tests-passed'])
        self.assertEqual(parsed.stage, 'queue')

    def test_b_teacher_reuses_native_bridge_without_copy_or_extract(self):
        _, _, spec, _ = spec_fixture(self.root)
        output = self.root / 'receipts'
        output.mkdir()
        calls = []

        def stage(source, target_root=None, **kwargs):
            calls.append((source, target_root, kwargs))
            return dict(complete=True, copied=False, mode='reuse_existing_local')

        module = SimpleNamespace(stage_teacher=stage)
        with patch.dict(sys.modules, {'code.V4.r101_features': module}), \
                patch.object(launch.base, 'receipts', return_value=output):
            result = launch.stage_teacher(spec, launch.prepare_teacher(spec))
        self.assertFalse(result['copied'])
        self.assertIsNone(calls[0][1])
        self.assertEqual(calls[0][0], launch.TEACHER_ROOTS['B'])
        self.assertEqual(calls[0][2]['variant'], 'B')

    def test_predecessors_are_per_arm_and_never_interrupt_healthy_runs(self):
        _, _, spec, _ = spec_fixture(self.root)
        dispatch = self.root / 'dispatch'
        predecessor = self.root / 'code/V4/runs/R98_performance_context'
        launch.base.save(dispatch / 'specs/R98_A.json', dict(stages=[
            dict(name='run', device='gpu'), dict(name='PDF', device='cpu')]))
        launch.base.save(dispatch / 'R98_A/worker.json', dict(state='complete', exit_status=0,
            stages=[dict(name='run', state='complete', exit_status=0)]))
        launch.base.save(dispatch / 'R98_A/controller.json', dict(state='complete', exit_status=0,
            stages=[dict(name=name, state='complete', exit_status=0) for name in ('allocation_GPU_pipeline', 'PDF')]))
        launch.base.save(predecessor / 'run_complete.json', dict(complete=True))
        launch.base.save(predecessor / 'postprocess_status.json', dict(complete=True, phase='complete'))
        (predecessor / 'RESULTS.pdf').write_bytes(b'%PDF-synthetic')
        with patch.object(launch, 'REPO', self.root), patch.object(launch, 'ROOT', dispatch):
            self.assertTrue(launch.predecessor_gate('B')['complete'])
            self.assertFalse(launch.predecessor_gate('A')['complete'])
            with patch.object(launch.base, 'check_ready', return_value=(spec, {})), \
                    patch.object(launch, 'original_start', return_value=dict(started=True)) as start:
                self.assertTrue(launch.start('/spec', 'CPU tests', True)['started'])
                self.assertEqual(start.call_count, 1)

    def test_cpu_queue_waits_then_starts_once_without_automatic_retry(self):
        _, _, spec, _ = spec_fixture(self.root, 'A')
        output = self.root / 'queue'
        output.mkdir()
        with patch.object(launch.base, 'check_ready', return_value=(spec, dict(evidence='ready'))), \
                patch.object(launch.base, 'receipts', return_value=output), \
                patch.object(launch, 'predecessor_gate', side_effect=[dict(complete=False, errors=[]), dict(complete=True, errors=[])]), \
                patch.object(launch.time, 'sleep') as sleep, \
                patch.object(launch, 'start', return_value=dict(session='synthetic')) as start:
            self.assertEqual(launch.queue('/spec', 'ready', True, interval=1)['session'], 'synthetic')
            with self.assertRaisesRegex(RuntimeError, 'already attempted'):
                launch.queue('/spec', 'ready', True)
        self.assertEqual(start.call_count, 1)
        self.assertEqual(sleep.call_count, 1)

    def test_primary_cache_is_reuse_only(self):
        with patch.object(launch.adapter, 'original_stage_cache') as stage:
            with self.assertRaisesRegex(RuntimeError, 'no fresh copy'):
                launch.stage_cache(dict(cache_root=str(self.root / 'missing')), {})
        stage.assert_not_called()


if __name__ == '__main__':
    unittest.main()
