"""Focused CPU-only R91 launcher checks; no prepare/start or resource probes."""

from pathlib import Path
import tempfile
import unittest
from unittest import mock

from . import r91_launch as launch


class R91LaunchTests(unittest.TestCase):
    def test_only_two_exact_job1465_3090_UUIDs_and_matched_controls(self):
        self.assertEqual(launch.ARM_SLOTS,
            dict(A_utility_ce='3090_0', B_utility_probabilities='3090_1'))
        self.assertEqual({slot: row['uuid'] for slot, row in launch.SLOTS.items()}, {
            '3090_0': 'GPU-cb24f168-9f18-f0d5-6f4a-6d2406958f50',
            '3090_1': 'GPU-cb626bdb-4319-2f5e-b91f-17fcaa44a83d'})
        self.assertEqual([row['physical_index'] for row in launch.SLOTS.values()], [3, 4])
        for arm, slot in launch.ARM_SLOTS.items():
            spec, item = launch.template(arm), launch.SLOTS[slot]
            self.assertEqual((item['job'], item['node'], item['gpu_type']), (1465, 'gpu03', '3090'))
            self.assertEqual(spec['slots'], [slot])
            self.assertEqual((spec['cpu_count'], spec['memory_gib']), (4, 9))
            protocol = spec['protocol']
            self.assertEqual((protocol['seed'], protocol['logical_batch'],
                protocol['max_successful_updates'], protocol['warmup_updates'],
                protocol['evaluation_every_updates']), (2, 1024, 57600, 4320, 1440))
            self.assertEqual(spec['wandb_mode'], 'offline')

    def test_four_real_runner_argv_parse_without_protocol_generation(self):
        from . import r91_experiment as runner
        with mock.patch.object(runner, 'protocol_lock',
                side_effect=AssertionError('launcher must not generate the main protocol')):
            result = launch.parse_commands()
        self.assertEqual(result['train_test_parsed_count'], 4)
        self.assertEqual([(row['arm'], row['command']) for row in result['train_test_argv']],
            [(arm, command) for arm in launch.ARM_SLOTS for command in ('train', 'test')])
        for row in result['train_test_argv']:
            self.assertTrue(row['parsed'])
            self.assertNotIn('--resume', row['argv'])
            for flag, value in (('--gpu-type', '3090'), ('--large-micro-cap', '352'),
                    ('--eval-batch-size', '256'), ('--wandb-mode', 'offline')):
                self.assertEqual(row['argv'][row['argv'].index(flag) + 1], value)

    def test_owned_srun_CPU_route_then_container_not_SSH_or_new_allocation(self):
        for slot in launch.SLOTS:
            command = launch.base.container_command('worker', slot, '/shared/spec.json', 'token')
            self.assertEqual(command[:4], ['sudo', '-n', '/usr/bin/ai_docker', 'exec'])
            self.assertIn(launch.CONTAINER, command)
            self.assertIn(str(launch.WRAPPER), command)
            self.assertIn('CUDA_VISIBLE_DEVICES=' + launch.SLOTS[slot]['uuid'], command)
            self.assertIn('R91_LAUNCH_TOKEN=token', command)
            step = launch.base.srun_command('entry', slot, cpus=4, memory='9G')
            for value in ('--jobid=1465', '--overlap', '--exact', '--gres=none',
                    '--immediate=5', '--nodelist=gpu03', '--cpus-per-task=4', '--mem=9G'):
                self.assertIn(value, step)
            for forbidden in ('ssh', 'sbatch', 'salloc', 'gpu05'):
                self.assertNotIn(forbidden, step)
            self.assertIn('conda activate easynco', step[-1])
            self.assertIn('code.V4.r91_launch entry', step[-1])

    def test_private_adapter_does_not_mutate_frozen_launchers(self):
        from . import r84_launch, r85_launch, r87_launch, r88_launch, r89_launch
        for name, module in (('R84', r84_launch), ('R85', r85_launch.base),
                             ('R87', r87_launch.base), ('R88', r88_launch.base),
                             ('R89', r89_launch.base)):
            self.assertEqual(module.TOKEN_ENV, name + '_LAUNCH_TOKEN')
            self.assertIsNot(module, launch.base)
        self.assertEqual(r89_launch.base.ROOT, launch.REPO / 'code/V4/runs/R89_launch')
        self.assertEqual(launch.engine.saved.ROOT, launch.REPO / 'code/V4/runs/R85_launch')
        self.assertEqual(launch.engine.R85_LAUNCH, launch.PREVIOUS_LAUNCH)

    def test_missing_main_ready_or_test_confirmation_never_invokes_tmux(self):
        with mock.patch.object(launch.base.socket, 'gethostname', return_value='admin01'), \
                mock.patch.object(Path, 'is_file', return_value=True), \
                mock.patch.object(launch.base.subprocess, 'run') as run:
            for evidence, tested in (('', False), ('', True), ('main evidence', False)):
                with self.assertRaisesRegex(RuntimeError, 'explicit ready'):
                    launch.base.start(Path('/unused/spec.json'), evidence, tested)
            run.assert_not_called()

    def test_R88_full_delivery_gate_not_only_training_or_idle_GPU(self):
        self.assertEqual(launch.PREDECESSORS,
            dict(A_utility_ce='A_canonical_anchor', B_utility_probabilities='B_execution_anchor'))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with mock.patch.object(launch.engine, 'R85_LAUNCH', root / 'launch'), \
                    mock.patch.object(launch.engine, 'R85_MODELS', root / 'models'):
                for arm, previous in launch.PREDECESSORS.items():
                    worker = root / 'launch' / previous / 'worker.json'
                    controller = worker.with_name('controller.json')
                    report = root / 'models' / previous / 'postprocess_status.json'
                    self.assertFalse(launch.predecessor_gate(arm)['complete'])
                    launch.base.save(worker, dict(state='complete', exit_status=0, stages=[
                        dict(name=name, state='complete', exit_status=0)
                        for name in ('train_VALlock_TEST', 'LIB')]))
                    launch.base.save(controller, dict(state='running', stages=[]))
                    self.assertFalse(launch.predecessor_gate(arm)['complete'])
                    launch.base.save(controller, dict(state='complete', exit_status=0, stages=[
                        dict(name=name, state='complete', exit_status=0)
                        for name in ('allocation_GPU_pipeline', 'PDF')]))
                    self.assertFalse(launch.predecessor_gate(arm)['complete'])
                    launch.base.save(report, dict(phase='complete', complete=True))
                    self.assertTrue(launch.predecessor_gate(arm)['complete'])
                    launch.base.save(controller, dict(state='complete', exit_status=1, stages=[]))
                    result = launch.predecessor_gate(arm)
                    self.assertFalse(result['complete'])
                    self.assertTrue(result['failed'])

    def test_same_GPU_stages_and_offline_mode_are_not_environment_overridden(self):
        with mock.patch.dict(launch.os.environ, {'WANDB_MODE': 'online',
                'R87_RESOLVED_WANDB_MODE': 'online', 'R88_RESOLVED_WANDB_MODE': 'online'}):
            for arm, slot in launch.ARM_SLOTS.items():
                stages = launch.commands(launch.template(arm), slot)
                self.assertEqual([stage['name'] for stage in stages], ['train', 'TEST', 'LIB', 'PDF'])
                self.assertTrue(all(stage['device'] == 'gpu' for stage in stages))
                self.assertTrue(all(stage['argv'][stage['argv'].index('--device') + 1]
                                    == 'cuda:0' for stage in stages))
                env = launch.base.environment(slot, 'token')
                self.assertEqual(env['CUDA_VISIBLE_DEVICES'], launch.SLOTS[slot]['uuid'])
                self.assertEqual(env['WANDB_MODE'], 'offline')
        self.assertEqual(launch.base.environment()['CUDA_VISIBLE_DEVICES'], '')
        self.assertEqual({launch.base.session_name(launch.template(arm)) for arm in launch.ARM_SLOTS},
            {'r91_A_utility_ce', 'r91_B_utility_probabilities'})

    def test_failed_stage_blocks_TEST_or_reports_and_has_no_retry(self):
        arm, slot, token = 'A_utility_ce', '3090_0', 'test-token'
        spec = launch.template(arm)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for failing, expected in (('train', ['train']), ('TEST', ['train', 'TEST']),
                                      ('LIB', ['train', 'TEST', 'LIB'])):
                folder = root / failing
                claim = dict(slot=slot, token=token, cpus=list(launch.ALLOWED_CPUS[:4]))
                launch.base.save(folder / arm / 'claim.json', claim)
                seen = []

                def run_stage(stage, *_):
                    seen.append(stage['name'])
                    if stage['name'] == failing:
                        raise RuntimeError('simulated nonzero exit; no later stages')

                with mock.patch.object(launch.base, 'ROOT', folder), \
                        mock.patch.object(launch.base, 'check_ready', return_value=(spec, dict(cache={}))), \
                        mock.patch.object(launch.base, 'assert_container', return_value=launch.CONTAINER), \
                        mock.patch.object(launch.base, 'stage_cache', return_value=dict(copied_bytes=0)), \
                        mock.patch.object(launch.base, 'run_stage', side_effect=run_stage), \
                        mock.patch.object(launch.base.os, 'sched_getaffinity', return_value=set(claim['cpus'])), \
                        mock.patch.object(launch.base.os, 'sched_setaffinity'), \
                        mock.patch.dict(launch.os.environ, {launch.TOKEN_ENV: token,
                            'CUDA_VISIBLE_DEVICES': launch.SLOTS[slot]['uuid']}):
                    with self.assertRaisesRegex(RuntimeError, 'simulated nonzero'):
                        launch.engine._worker(Path('/unused/spec.json'), slot, token)
                self.assertEqual(seen, expected)
                result = launch.base.read(folder / arm / 'worker.json')
                self.assertEqual((result['state'], result['exit_status']), ('failed', 1))
                self.assertFalse(result['automatic_retry'])

    def test_cache_reuse_has_no_copy_rebuild_or_array_audit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for split in ('train', 'val', 'test'):
                (root / split).mkdir()
            launch.base.save(root / '.r84_cache_identity.json',
                dict(data_root=str(launch.DATA), source=str(launch.CACHE_SOURCE)))
            manifest = dict(files=[dict(path='train/TSP/node.npy', bytes=1)],
                            bytes=1, identity_sha256='existing')
            with mock.patch.object(launch.base.shutil, 'copyfile',
                    side_effect=AssertionError('no cache copy')), \
                    mock.patch.object(launch.base, 'sha', side_effect=AssertionError('no array hash')):
                result = launch.base.stage_cache(dict(cache_root=str(root),
                    data_root=str(launch.DATA), cache_source=str(launch.CACHE_SOURCE)), manifest)
            self.assertEqual(result['copied_bytes'], 0)
            self.assertFalse(result['preprocessing'])
            self.assertFalse(result['array_contents_read'])
            self.assertFalse(result['cache_audit_repeated'])
            self.assertTrue((root / 'test').is_dir())

    def test_prepare_freezes_real_R85_evidence_not_fake_R88_resources(self):
        value = dict(spec=dict(run_id='A_utility_ce'), source_sha256={})
        with mock.patch.object(launch, 'parse_commands'), \
                mock.patch.object(launch.engine, '_prepare', return_value=value) as original, \
                mock.patch.object(launch.engine, 'prepare',
                    side_effect=AssertionError('R87 late-bound prepare mixes predecessor/cache roots')), \
                mock.patch.object(launch.base, 'sha', return_value='source-hash'), \
                mock.patch.object(launch.base, 'save'), \
                mock.patch.object(launch, 'predecessor_gate', return_value=dict(complete=True)):
            result = launch.prepare(Path('/unused/spec.json'))
        original.assert_called_once_with(Path('/unused/spec.json'))
        self.assertIn(str(launch.REPO / 'code/V4/runs/R85_launch/resources.json'),
                      result['source_sha256'])
        self.assertNotIn(str(launch.PREVIOUS_LAUNCH / 'resources.json'), result['source_sha256'])
        self.assertFalse(result['main_thread_ready'])
        self.assertFalse(result['automatic_retry'])

    def test_exact_spec_rejects_wrong_arms_flags_and_report_location(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'spec.json'
            spec = launch.template('B_utility_probabilities')
            launch.base.save(path, spec)
            self.assertEqual(launch.load_spec(path), spec)
            spec['stages'][-1]['device'] = 'cpu'
            launch.base.save(path, spec)
            with self.assertRaisesRegex(ValueError, 'same-GPU'):
                launch.load_spec(path)
            with self.assertRaises(ValueError):
                launch.commands(launch.template('A_utility_ce'), '3090_1')


if __name__ == '__main__':
    unittest.main()
