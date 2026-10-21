"""Small CPU-only R89 launcher checks; never submit training or probe GPUs."""

from pathlib import Path
import tempfile
import unittest
from unittest import mock

from . import r89_launch as launch


class R89LaunchTests(unittest.TestCase):
    def test_two_matched_4090_slots_and_controls(self):
        self.assertEqual(set(launch.SLOTS), {'4090_0', '4090_1'})
        self.assertEqual(len({row['uuid'] for row in launch.SLOTS.values()}), 2)
        for arm, slot in launch.ARM_SLOTS.items():
            spec = launch.template(arm)
            self.assertEqual(spec['slots'], [slot])
            self.assertEqual((spec['cpu_count'], spec['memory_gib']), (4, 9))
            controls = spec['protocol']
            self.assertEqual((controls['logical_batch'], controls['seed']), (1024, 2))
            self.assertEqual((controls['max_successful_updates'], controls['warmup_updates'],
                controls['evaluation_every_updates']), (57600, 4320, 1440))
            self.assertEqual((controls['model_role_arm'], controls['role_slot'],
                controls['extra_model_parameters']), ('B_execution_anchor', 3, 0))
            command = launch.commands(spec, slot)[0]['argv']
            for flag, value in (('--gpu-type', '4090'), ('--large-micro-cap', '352'),
                    ('--eval-batch-size', '256')):
                self.assertEqual(command[command.index(flag) + 1], value)
            self.assertIn('code.V4.r89_experiment', command)
            self.assertNotIn('--resume', command)

    def test_frozen_namespaces_are_unchanged(self):
        from . import r85_launch, r87_launch, r88_launch
        self.assertEqual(r85_launch.base.TOKEN_ENV, 'R85_LAUNCH_TOKEN')
        self.assertEqual(r87_launch.base.TOKEN_ENV, 'R87_LAUNCH_TOKEN')
        self.assertEqual(r88_launch.base.TOKEN_ENV, 'R88_LAUNCH_TOKEN')
        self.assertEqual(r87_launch.base.ROOT, launch.PREVIOUS_LAUNCH)
        self.assertEqual(set(r88_launch.SLOTS), {'3090_0', '3090_1'})

    def test_owned_container_and_CPU_allocation_entry(self):
        for slot in launch.SLOTS:
            command = launch.base.container_command('worker', slot, '/shared/spec.json', 'token')
            self.assertIn(str(launch.WRAPPER), command)
            self.assertIn('CUDA_VISIBLE_DEVICES=' + launch.SLOTS[slot]['uuid'], command)
            self.assertIn('R89_LAUNCH_TOKEN=token', command)
            self.assertIn(launch.base.transport.CONTAINERS['gpu05'][1], command)
            step = launch.base.srun_command('entry', slot, cpus=4, memory='9G')
            for value in ('--jobid=1475', '--nodelist=gpu05', '--gres=none',
                    '--cpus-per-task=4', '--mem=9G'):
                self.assertIn(value, step)
            self.assertNotIn('ssh', step)

    def test_no_start_without_explicit_main_ready(self):
        with mock.patch.object(launch.base.socket, 'gethostname', return_value='admin01'), \
                mock.patch.object(Path, 'is_file', return_value=True), \
                mock.patch.object(launch.base.subprocess, 'run') as run:
            with self.assertRaisesRegex(RuntimeError, 'explicit ready'):
                launch.base.start(Path('/missing/spec.json'), '', False)
            run.assert_not_called()

    def test_R87_full_delivery_and_round_comparison_gate(self):
        self.assertEqual(launch.PREDECESSORS,
            dict(A_adamw='A_shared_residual', B_muon_hybrid='B_private_residual'))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with mock.patch.object(launch.engine, 'R85_LAUNCH', root / 'launch'), \
                    mock.patch.object(launch.engine, 'R85_MODELS', root / 'models'):
                for arm, previous in launch.PREDECESSORS.items():
                    launch.base.save(root / 'launch' / previous / 'worker.json',
                        dict(state='complete', exit_status=0, stages=[
                            dict(name=name, state='complete', exit_status=0)
                            for name in ('train_VALlock_TEST', 'LIB')]))
                    controller = root / 'launch' / previous / 'controller.json'
                    launch.base.save(controller, dict(state='running', stages=[]))
                    self.assertFalse(launch.predecessor_gate(arm)['complete'])
                    launch.base.save(controller, dict(state='complete', exit_status=0,
                        stages=[dict(name=name, state='complete', exit_status=0)
                                for name in ('allocation_GPU_pipeline', 'PDF')]))
                    self.assertFalse(launch.predecessor_gate(arm)['complete'])
                    launch.base.save(root / 'models' / previous / 'postprocess_status.json',
                        dict(phase='complete'))
                    comparison = root / 'models' / 'goal_compare_receipt.json'
                    launch.base.save(comparison, dict(state='running'))
                    self.assertFalse(launch.predecessor_gate(arm)['complete'])
                    launch.base.save(comparison, dict(state='complete', exit_status=0))
                    self.assertTrue(launch.predecessor_gate(arm)['complete'])
                    launch.base.save(comparison, dict(state='failed', exit_status=1))
                    gate = launch.predecessor_gate(arm)
                    self.assertFalse(gate['complete'])
                    self.assertTrue(gate['failed'])

    def test_independent_sessions_CPU_report_and_WandB_resolution(self):
        self.assertEqual({launch.base.session_name(launch.template(arm))
            for arm in launch.ARM_SLOTS}, {'r89_A_adamw', 'r89_B_muon_hybrid'})
        self.assertEqual(launch.base.environment()['CUDA_VISIBLE_DEVICES'], '')
        with mock.patch.dict(launch.os.environ,
                {'R89_RESOLVED_WANDB_MODE': 'offline', 'R87_RESOLVED_WANDB_MODE': 'online'}):
            for arm, slot in launch.ARM_SLOTS.items():
                stages = launch.commands(launch.template(arm), slot)
                command = stages[0]['argv']
                self.assertEqual(command[command.index('--wandb-mode') + 1], 'offline')
                self.assertEqual(stages[-1]['argv'][-1], 'cpu')

    def test_cache_reuse_without_copy_or_array_scan(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for split in ('train', 'val'):
                (root / split).mkdir()
            launch.base.save(root / '.r84_cache_identity.json',
                dict(data_root=str(launch.DATA), source=str(launch.CACHE_SOURCE)))
            manifest = dict(files=[dict(path='train/TSP/node.npy', bytes=1)],
                bytes=1, identity_sha256='existing')
            with mock.patch.object(launch.base.shutil, 'copyfile',
                    side_effect=AssertionError('no copy')):
                result = launch.engine.saved.stage_cache(dict(cache_root=str(root),
                    data_root=str(launch.DATA), cache_source=str(launch.CACHE_SOURCE)), manifest)
            self.assertEqual(result['copied_bytes'], 0)
            self.assertFalse(result['cache_audit_repeated'])

    def test_actual_Muon_witness_required_not_only_configuration(self):
        value = launch.template('B_muon_hybrid')
        witness = value['startup_evidence_required']
        self.assertTrue(witness['actual_optimizer_type_and_groups'])
        self.assertTrue(witness['actual_muon_step_applied_positive'])
        self.assertTrue(witness['actual_update_RMS_finite'])
        self.assertFalse(witness['validation_wait_required'])
        self.assertFalse(launch.template('A_adamw')['startup_evidence_required'][
            'actual_muon_step_applied_positive'])

    def test_prepare_keeps_R85_cache_evidence_separate_from_R87_predecessor(self):
        value = dict(spec=dict(run_id='A_adamw'), source_sha256={})
        with mock.patch.object(launch.engine, '_prepare', return_value=value) as original, \
                mock.patch.object(launch.engine, 'prepare',
                    side_effect=AssertionError('R87 prepare mixes predecessor/cache paths')), \
                mock.patch.object(launch.base, 'sha', return_value='recorded-hash'), \
                mock.patch.object(launch.base, 'save'), \
                mock.patch.object(launch, 'predecessor_gate', return_value=dict(complete=True)):
            result = launch.prepare(Path('/unused/spec.json'))
        original.assert_called_once_with(Path('/unused/spec.json'))
        self.assertIn(str(launch.REPO / 'code/V4/runs/R85_launch/resources.json'),
            result['source_sha256'])
        self.assertNotIn(str(launch.PREVIOUS_LAUNCH / 'resources.json'),
            result['source_sha256'])
        self.assertTrue(result['predecessor']['complete'])
        self.assertFalse(result['main_thread_ready'])


if __name__ == '__main__':
    unittest.main()
