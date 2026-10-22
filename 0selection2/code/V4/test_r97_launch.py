"""Small CPU-only launcher checks; every transport/runner is synthetic."""

import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from . import r84_launch
from . import r97_launch as launch


def runner_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=('run', 'report'))
    for name in ('root', 'data-root', 'prepared-root', 'feature-root', 'context-root', 'protocol-lock'):
        parser.add_argument('--' + name, type=Path)
    parser.add_argument('--device')
    parser.add_argument('--wandb-mode')
    return parser


class LauncherTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.patch(launch, 'ROOT', self.root / 'launch')
        self.patch(launch.base, 'ROOT', launch.ROOT)
        self.patch(launch, 'MODEL_ROOT', self.root / 'model')
        self.patch(launch, 'PROTOCOL_LOCK', launch.MODEL_ROOT / 'protocol_lock.json')
        self.patch(launch, 'CACHE_SOURCE', launch.MODEL_ROOT / 'context')
        self.patch(launch, 'CACHE_ROOT', self.root / 'local')
        self.patch(launch, 'PREPARED', self.root / 'prepared')
        self.patch(launch, 'DATA', self.root / 'data')
        self.patch(launch, 'FEATURE_ROOT', self.root / 'features')
        self.patch(launch, 'PREDECESSOR_LAUNCH', self.root / 'previous_launch')
        self.patch(launch, 'PREDECESSOR_MODEL', self.root / 'previous_model')
        self.protocol = dict(experiment='R97', locked=True, arms=['A'], task_scope='full18',
            data_root=str(launch.DATA), prepared_root=str(launch.PREPARED),
            launch=dict(pythonpath=['/locked/tabicl/src']))
        launch.base.save(launch.PROTOCOL_LOCK, self.protocol)
        for path in (launch.DATA, launch.PREPARED, launch.FEATURE_ROOT, launch.CACHE_SOURCE):
            path.mkdir(parents=True, exist_ok=True)
        (launch.CACHE_SOURCE / 'TSP.npy').write_bytes(b'context-only')
        launch.base.save(launch.CACHE_SOURCE / 'manifest.json', dict(split='train'))
        self.sources = [str(launch.PROTOCOL_LOCK), str(launch.CACHE_SOURCE / 'manifest.json')]
        self.patch(launch, '_sources', mock.Mock(side_effect=lambda protocol: list(self.sources)))
        self.real_parse_runtime = launch._parse_runtime
        self.patch(launch, '_parse_runtime', mock.Mock(side_effect=lambda argv, protocol:
            {k: str(v) if isinstance(v, Path) else v for k, v in vars(runner_parser().parse_args(argv)).items()}))
        self.capture = self.patch(launch.transport, 'capture', mock.Mock(side_effect=AssertionError('No real transport')))
        self.patch(launch.transport, 'gpu_inventory', mock.Mock(side_effect=AssertionError('No GPU probes')))

    def patch(self, obj, name, value):
        patcher = mock.patch.object(obj, name, value)
        self.addCleanup(patcher.stop)
        return patcher.start()

    def spec(self):
        path = self.root / 'spec.json'
        launch.base.save(path, launch.template(protocol=self.protocol))
        return path

    def prepare(self):
        path = self.spec()
        launch.prepare(path, 'main final CPU tests', True)
        return path

    def predecessor(self):
        for name, stages in (('worker', ('input_features', 'train', 'TEST', 'timing', 'LIB')),
                ('controller', ('allocation_GPU_pipeline', 'PDF'))):
            launch.base.save(launch.PREDECESSOR_LAUNCH / (name + '.json'), dict(state='complete',
                exit_status=0, stages=[dict(name=s, state='complete', exit_status=0) for s in stages]))
        launch.base.save(launch.PREDECESSOR_MODEL / 'final_test_lock.json', dict(experiment='R94', arm='B',
            checkpoint_sha256='same-checkpoint', selected_without_test=True, per_problem_assembly=False))
        for name in ('test_results.json', 'lib_stage_complete.json'):
            launch.base.save(launch.PREDECESSOR_MODEL / name, dict(complete=True, checkpoint_sha256='same-checkpoint'))
        post = dict(complete=True, phase='complete', exit_status=0)
        for key in ('model_statistics_pdf', 'selection_distributions_pdf', 'target_recovery_pdf'):
            path = launch.PREDECESSOR_MODEL / (key + '.pdf')
            path.write_bytes(b'%PDF-synthetic')
            post[key] = str(path)
        launch.base.save(launch.PREDECESSOR_MODEL / 'postprocess_status.json', post)

    def test_private_engine_fixed_allocation_and_runtime_contract(self):
        self.assertIsNot(launch.base, r84_launch)
        self.assertIsNot(launch.transport, r84_launch.transport)
        self.assertEqual(r84_launch.TOKEN_ENV, 'R84_LAUNCH_TOKEN')
        self.assertEqual(launch.SLOTS[launch.SLOT]['job'], 1475)
        self.assertEqual(launch.SLOTS[launch.SLOT]['uuid'], 'GPU-597c682a-4b7f-7609-80a6-8ced791f5c5d')
        spec = launch.template(protocol=self.protocol)
        self.assertEqual((spec['cpu_count'], spec['memory_gib']), (4, 9))
        self.assertEqual(spec['protocol']['sgd_updates'], 0)
        self.assertNotIn('scratch', json.dumps(spec))
        self.assertNotIn('57600', json.dumps(spec))
        result = launch.parse_commands(self.protocol, spec)
        self.assertEqual(result['parsed_count'], 2)
        self.assertEqual([r['argv'][0] for r in result['commands']], ['run', 'report'])
        self.assertFalse(result['resource_probe_performed'])
        self.assertEqual(launch.environment(protocol=self.protocol)['CUDA_VISIBLE_DEVICES'], '')
        self.assertTrue(launch.environment(protocol=self.protocol)['PYTHONPATH'].startswith('/locked/tabicl/src:'))
        with self.assertRaises(ValueError):
            launch.commands(spec, '4090_0')
        argv = launch.base.srun_command('entry', launch.SLOT, self.root / 'spec', 'token', cpus=4, memory='9G')
        for flag in ('--jobid=1475', '--gres=none', '--cpus-per-task=4', '--mem=9G'):
            self.assertIn(flag, argv)
        container = launch.container_command('worker', launch.SLOT, self.root / 'spec', 'token')
        for value in (str(launch.WRAPPER), 'WANDB_MODE=offline', 'R97_LAUNCH_TOKEN=token',
                'CUDA_VISIBLE_DEVICES=' + launch.SLOTS[launch.SLOT]['uuid'],
                '76045731f6b453443fe18a9bce671599bdbf17c2713b8edcdd090925eea14bf6'):
            self.assertIn(value, container)
        self.capture.assert_not_called()

    def test_confirmation_and_missing_runtime_block_before_transport(self):
        for action in (launch.generate_spec, launch.prepare, launch.start, launch.queue):
            for evidence, tested in (('', True), (' ', True), ('tests', False)):
                with self.subTest(action=action.__name__), self.assertRaises(RuntimeError):
                    action(evidence, tested) if action == launch.generate_spec else action('/missing', evidence, tested)
        with self.assertRaises(RuntimeError):
            launch.authorize_resources(None, launch.SLOT)
        self.patch(launch, '_parse_runtime', mock.Mock(side_effect=RuntimeError('Waiting for main parser')))
        with self.assertRaisesRegex(RuntimeError, 'Waiting'):
            launch.generate_spec('final tests', True)
        self.assertFalse((launch.ROOT / 'specs/A.json').exists())
        self.capture.assert_not_called()

    def test_main_singular_arm_protocol_uses_isolated_library_root(self):
        protocol = dict(self.protocol, arm='A', library_root=str(launch.MODEL_ROOT / 'runtime/python'))
        del protocol['arms']
        del protocol['launch']
        launch.base.save(launch.PROTOCOL_LOCK, protocol)
        self.assertEqual(launch.locked_protocol(), protocol)
        env = launch.environment(protocol=protocol)
        self.assertTrue(env['PYTHONPATH'].startswith(protocol['library_root'] + ':'))
        self.assertEqual(env['CUDA_VISIBLE_DEVICES'], '')
        container = launch.container_command('worker', launch.SLOT, self.root / 'spec', 'token')
        self.assertIn('PYTHONPATH=' + launch.environment(launch.SLOT, 'token')['PYTHONPATH'], container)
        for changed in (dict(protocol, arm='B'), dict(protocol, arms=['A', 'B'])):
            launch.base.save(launch.PROTOCOL_LOCK, changed)
            with self.assertRaises(ValueError):
                launch.locked_protocol()
        self.capture.assert_not_called()

    def test_readiness_waits_for_final_reporter_source(self):
        path = self.spec()
        reporter = self.root / 'r97_reporting.py'
        self.sources.append(str(reporter))
        with self.assertRaisesRegex(RuntimeError, 'Waiting for final R97 source'):
            launch.prepare(path, 'synthetic final tests', True)
        self.assertFalse((launch.ROOT / 'A/ready.json').exists())
        self.assertFalse((launch.ROOT / 'A/plan.json').exists())
        self.capture.assert_not_called()

    def test_freeze_ready_before_queue_and_ignore_future_sources(self):
        path = self.spec()
        added_before_ready = self.root / 'ready_test.py'
        added_before_ready.write_text('# Included before readiness.\n')
        self.sources.append(str(added_before_ready))
        plan = launch.prepare(path, 'main final CPU tests', True)
        self.assertIn(str(added_before_ready), plan['source_sha256'])
        self.assertTrue((launch.ROOT / 'A/ready.json').is_file())
        launch.queue_prepared(path, 'main final CPU tests', True)
        self.sources.append(str(self.root / 'future_test.py'))
        launch.queue_prepared(path, 'main final CPU tests', True)
        with self.assertRaisesRegex(RuntimeError, 'same explicit'):
            launch.queue_prepared(path, 'different evidence', True)
        launch.base.save(launch.CACHE_SOURCE / 'manifest.json', dict(split='test'))
        with self.assertRaisesRegex(RuntimeError, 'source changed|identity changed'):
            launch.queue_prepared(path, 'main final CPU tests', True)
        self.capture.assert_not_called()

    def test_parser_subprocess_is_cpu_only_and_missing_cli_waits(self):
        self.patch(launch, 'REPO', self.root)
        with self.assertRaisesRegex(RuntimeError, 'Waiting'):
            self.real_parse_runtime(['run'], self.protocol)
        runtime = self.root / 'code/V4/r97_experiment.py'
        runtime.parent.mkdir(parents=True)
        runtime.write_text('# Synthetic parser; never executed.\n')
        run = self.patch(launch.base.subprocess, 'run', mock.Mock(return_value=
            argparse.Namespace(returncode=0, stdout='{"command":"run"}', stderr='')))
        self.assertEqual(self.real_parse_runtime(['run'], self.protocol), dict(command='run'))
        self.assertEqual(run.call_args.kwargs['env']['CUDA_VISIBLE_DEVICES'], '')
        self.assertTrue(run.call_args.kwargs['env']['PYTHONPATH'].startswith('/locked/tabicl/src:'))
        run.return_value = argparse.Namespace(returncode=2, stdout='', stderr='run CLI not ready')
        with self.assertRaisesRegex(RuntimeError, 'Waiting for compatible'):
            self.real_parse_runtime(['run'], self.protocol)
        self.capture.assert_not_called()

    def test_context_stages_once_checks_identity_size_and_symlinks(self):
        spec = launch.template(protocol=self.protocol)
        manifest = launch.context_manifest(launch.CACHE_SOURCE)
        first = launch.stage_cache(spec, manifest)
        with mock.patch.object(launch.base.shutil, 'copyfile', side_effect=AssertionError('copy twice')):
            self.assertEqual(launch.stage_cache(spec, manifest), first)
        (launch.CACHE_ROOT / 'TSP.npy').write_bytes(b'wrong size')
        with self.assertRaisesRegex(RuntimeError, 'size'):
            launch.stage_cache(spec, manifest)
        (launch.CACHE_ROOT / 'TSP.npy').unlink()
        (launch.CACHE_ROOT / 'TSP.npy').symlink_to(launch.CACHE_SOURCE / 'TSP.npy')
        with self.assertRaisesRegex(ValueError, 'symlinks'):
            launch.stage_cache(spec, manifest)
        alias = self.root / 'source_alias'
        alias.symlink_to(launch.CACHE_SOURCE, target_is_directory=True)
        with self.assertRaises(ValueError):
            launch.context_manifest(alias)

    def test_nested_worker_context_layout_stages_all_18_tasks(self):
        source = self.root / 'worker_context'
        for number in range(18):
            directory = source / ('PROBLEM_' + str(number))
            directory.mkdir(parents=True)
            for name in ('X.npy', 'y.npy', 'source_ids.npy'):
                (directory / name).write_bytes(b'synthetic TRAIN array')
            launch.base.save(directory / 'identity.json', dict(label_splits=['train'], rows=81000))
        spec = dict(cache_source=str(source), cache_root=str(launch.CACHE_ROOT))
        manifest = launch.context_manifest(source)
        result = launch.stage_cache(spec, manifest)
        self.assertEqual(result['files'], 18 * 4)
        self.assertEqual(len(list(launch.CACHE_ROOT.glob('*/identity.json'))), 18)
        self.assertEqual(result['identity_sha256'], manifest['identity_sha256'])

    def test_predecessor_requires_full_worker_controller_same_lock_pdf(self):
        self.assertFalse(launch.predecessor_gate()['complete'])
        self.predecessor()
        self.assertTrue(launch.predecessor_gate()['complete'])
        launch.base.save(launch.PREDECESSOR_LAUNCH / 'controller.json', dict(state='reporting'))
        self.assertFalse(launch.predecessor_gate()['complete'])
        self.assertFalse(launch.predecessor_gate()['failed'])
        launch.base.save(launch.PREDECESSOR_LAUNCH / 'controller.json', dict(state='failed', exit_status=4))
        self.assertTrue(launch.predecessor_gate()['failed'])
        self.predecessor()
        launch.base.save(launch.PREDECESSOR_MODEL / 'test_results.json', dict(complete=True, checkpoint_sha256='wrong'))
        self.assertTrue(launch.predecessor_gate()['failed'])

    def test_queue_waits_for_predecessor_starts_once_and_stops_on_failure(self):
        path = self.prepare()
        self.patch(launch, 'predecessor_gate', mock.Mock(side_effect=[dict(complete=False, failed=False),
            dict(complete=True, failed=False)]))
        start = self.patch(launch, 'start', mock.Mock(return_value='one launch'))
        sleep = self.patch(launch.time, 'sleep', mock.Mock())
        self.assertEqual(launch.queue(path, 'main final CPU tests', True), 'one launch')
        start.assert_called_once_with(path, 'main final CPU tests', True)
        sleep.assert_called_once_with(60)
        self.patch(launch, 'predecessor_gate', mock.Mock(return_value=dict(complete=False, failed=True, reason='failed')))
        with self.assertRaisesRegex(RuntimeError, 'failed'):
            launch.queue(path, 'main final CPU tests', True)
        self.assertEqual(start.call_count, 1)
        launch.base.save(launch.ROOT / 'A/attempt.json', dict(automatic_retry=False))
        with self.assertRaisesRegex(RuntimeError, 'already attempted'):
            launch.queue(path, 'main final CPU tests', True)

    def test_r94_r95_cpu_reservations_release_only_finished_gpu_stages(self):
        self.patch(launch, 'REPO', self.root)
        for number, cpus in (('94', [0, 1, 2, 3]), ('95', [4, 5, 6, 7])):
            directory = self.root / ('code/V4/runs/R' + number + '_launch/B')
            launch.base.save(directory / 'claim.json', dict(node='gpu05', cpus=cpus))
            launch.base.save(directory / 'controller.json', dict(state='failed'))
        self.assertEqual(launch.reserved_cpus('gpu05', self.root), set(range(8)))
        launch.base.save(self.root / 'code/V4/runs/R94_launch/B/controller.json', dict(state='reporting',
            stages=[dict(name='allocation_GPU_pipeline', state='complete', exit_status=0)]))
        self.assertEqual(launch.reserved_cpus('gpu05', self.root), {4, 5, 6, 7})

    def test_fake_controller_report_requires_rc0_and_releases_gpu_mutex(self):
        spec = launch.template(protocol=self.protocol)
        output = launch.base.receipts(spec)
        self.patch(launch, 'authorize_resources', mock.Mock(return_value=(spec, {})))
        self.patch(launch, 'check_ready', mock.Mock(return_value=(spec, {})))
        self.patch(launch.base.socket, 'gethostname', mock.Mock(return_value='admin01'))
        self.patch(launch.base, 'FONT', launch.PROTOCOL_LOCK)
        self.patch(launch, 'reserved_cpus', mock.Mock(return_value={0, 1, 2, 3}))
        probe = dict(allowed_cpus=list(range(8)), processes=[],
            memory=dict(headroom_excluding_reclaimable_cache_bytes=10 * 2**30),
            gpus={launch.SLOT: dict(compute_processes=[], memory_used_mib=0, utilization_pct=0)})
        self.capture.side_effect = None
        self.capture.return_value = json.dumps(probe)
        gpu_locked = []
        original_lock = launch.base.lock

        @contextmanager
        def tracked_lock(path, **kwargs):
            gpu = Path(path).name.startswith('queue_gpu_')
            if gpu:
                gpu_locked.append(True)
            with original_lock(path, **kwargs):
                yield
            if gpu:
                gpu_locked.pop()

        self.patch(launch.base, 'lock', tracked_lock)
        self.patch(launch.transport, 'ROOT', self.root / 'mutex')
        calls = []

        def run_stage(stage, spec, record, receipt, env):
            calls.append(stage['name'])
            if stage['device'] == 'gpu':
                self.assertTrue(gpu_locked)
                launch.base.save(output / 'worker.json', dict(state='complete', exit_status=0,
                    stages=[dict(name='run', state='complete', exit_status=0)]))
            else:
                self.assertFalse(gpu_locked)
                self.assertEqual(env['CUDA_VISIBLE_DEVICES'], '')
            record['stages'].append(dict(name=stage['name'], state='complete', exit_status=0))

        self.patch(launch.base, 'run_stage', mock.Mock(side_effect=run_stage))
        self.assertEqual(launch.controller(self.root / 'spec'), 0)
        self.assertEqual(calls, ['allocation_GPU_pipeline', 'PDF'])
        claim = launch.base.read(output / 'claim.json')
        self.assertEqual(claim['cpus'], [4, 5, 6, 7])
        with self.assertRaisesRegex(RuntimeError, 'duplicate'):
            launch.controller(self.root / 'spec')
        self.patch(launch.base, 'ROOT', self.root / 'failed_launch')
        output = launch.base.receipts(spec)
        calls.clear()

        def failed_worker(stage, spec, record, receipt, env):
            calls.append(stage['name'])
            launch.base.save(output / 'worker.json', dict(state='failed', exit_status=1, stages=[]))

        self.patch(launch.base, 'run_stage', mock.Mock(side_effect=failed_worker))
        with self.assertRaisesRegex(RuntimeError, 'CPU PDF requires rc0'):
            launch.controller(self.root / 'spec')
        self.assertEqual(calls, ['allocation_GPU_pipeline'])
        self.patch(launch.base, 'ROOT', self.root / 'busy_launch')
        probe['gpus'][launch.SLOT]['compute_processes'] = [['gpu', '123', 'healthy predecessor']]
        self.capture.return_value = json.dumps(probe)
        calls.clear()
        with self.assertRaisesRegex(RuntimeError, 'leave healthy runs alone'):
            launch.controller(self.root / 'spec')
        self.assertEqual(calls, [])
        with self.assertRaisesRegex(RuntimeError, 'duplicate'):
            launch.controller(self.root / 'spec')


if __name__ == '__main__':
    unittest.main()
