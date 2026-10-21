"""CPU-only R88 routing, gate, and matched-CLI checks; no jobs are started."""

from pathlib import Path
import tempfile
import unittest
from unittest import mock

from . import r88_launch as launch


class R88LaunchTests(unittest.TestCase):
    def test_only_matched_3090_slots(self):
        self.assertEqual(set(launch.SLOTS), {'3090_0', '3090_1'})
        self.assertEqual(len({row['uuid'] for row in launch.SLOTS.values()}), 2)
        for arm, slot in launch.ARM_SLOTS.items():
            spec = launch.template(arm)
            command = launch.commands(spec, slot)[0]['argv']
            self.assertEqual(spec['slots'], [slot])
            self.assertEqual(command[command.index('--gpu-type') + 1], '3090')
            self.assertEqual(command[command.index('--large-micro-cap') + 1], '352')
            self.assertEqual(command[command.index('--eval-batch-size') + 1], '256')
            self.assertIn('code.V4.r88_experiment', command)
            self.assertEqual(spec['protocol']['role_slot'], 3)
            self.assertEqual(spec['protocol']['extra_model_parameters'], 0)

    def test_frozen_R85_R87_namespaces_unchanged(self):
        from . import r85_launch, r87_launch
        self.assertEqual(r85_launch.base.TOKEN_ENV, 'R85_LAUNCH_TOKEN')
        self.assertEqual(r87_launch.base.TOKEN_ENV, 'R87_LAUNCH_TOKEN')
        self.assertEqual(set(r87_launch.SLOTS), {'4090_0', '4090_1'})
        self.assertEqual(r87_launch.base.ROOT, launch.REPO / 'code/V4/runs/R87_launch')

    def test_owned_container_and_CPU_entry(self):
        for slot in launch.SLOTS:
            command = launch.base.container_command('worker', slot, '/shared/spec.json', 'token')
            self.assertIn(str(launch.WRAPPER), command)
            self.assertIn('CUDA_VISIBLE_DEVICES=' + launch.SLOTS[slot]['uuid'], command)
            self.assertIn('R88_LAUNCH_TOKEN=token', command)
            self.assertIn(launch.base.transport.CONTAINERS['gpu03'][1], command)
            step = launch.base.srun_command('entry', slot, cpus=4, memory='9G')
            self.assertIn('--jobid=1465', step)
            self.assertIn('--nodelist=gpu03', step)
            self.assertIn('--gres=none', step)
            self.assertNotIn('ssh', step)

    def test_no_start_without_main_ready(self):
        with mock.patch.object(launch.base.socket, 'gethostname', return_value='admin01'), \
                mock.patch.object(Path, 'is_file', return_value=True), \
                mock.patch.object(launch.base.subprocess, 'run') as run:
            with self.assertRaisesRegex(RuntimeError, 'explicit ready'):
                launch.base.start(Path('/missing/spec.json'), '', False)
            run.assert_not_called()

    def test_R85_pair_full_report_gate(self):
        self.assertEqual(launch.PREDECESSORS, dict(A_canonical_anchor='A_ovr_control', B_execution_anchor='B_gradient_balance'))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with mock.patch.object(launch.engine, 'R85_LAUNCH', root / 'launch'), \
                    mock.patch.object(launch.engine, 'R85_MODELS', root / 'models'):
                for arm, previous in launch.PREDECESSORS.items():
                    launch.base.save(root / 'launch' / previous / 'worker.json', dict(state='complete', exit_status=0,
                        stages=[dict(name=name, state='complete', exit_status=0) for name in ('train_VALlock_TEST', 'LIB')]))
                    path = root / 'launch' / previous / 'controller.json'
                    launch.base.save(path, dict(state='running', stages=[]))
                    self.assertFalse(launch.engine.predecessor_gate(arm)['complete'])
                    launch.base.save(path, dict(state='complete', exit_status=0,
                        stages=[dict(name=name, state='complete', exit_status=0) for name in ('allocation_GPU_pipeline', 'PDF')]))
                    self.assertFalse(launch.engine.predecessor_gate(arm)['complete'])
                    launch.base.save(root / 'models' / previous / 'postprocess_status.json', dict(phase='complete'))
                    self.assertTrue(launch.engine.predecessor_gate(arm)['complete'])

    def test_sessions_CPU_report_and_independent_WandB_resolution(self):
        self.assertEqual(len({launch.base.session_name(launch.template(arm)) for arm in launch.ARM_SLOTS}), 2)
        self.assertEqual(launch.base.environment()['CUDA_VISIBLE_DEVICES'], '')
        with mock.patch.dict(launch.os.environ, {'R88_RESOLVED_WANDB_MODE': 'offline', 'R87_RESOLVED_WANDB_MODE': 'online'}):
            for arm, slot in launch.ARM_SLOTS.items():
                stages = launch.commands(launch.template(arm), slot)
                command = stages[0]['argv']
                self.assertEqual(command[command.index('--wandb-mode') + 1], 'offline')
                self.assertEqual(stages[-1]['argv'][-1], 'cpu')

    def test_cache_reuse_does_not_copy_or_scan_arrays(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for split in ('train', 'val'):
                (root / split).mkdir()
            launch.base.save(root / '.r84_cache_identity.json', dict(data_root=str(launch.DATA), source=str(launch.CACHE_SOURCE)))
            manifest = dict(files=[dict(path='train/TSP/node.npy', bytes=1)], bytes=1, identity_sha256='existing')
            with mock.patch.object(launch.base.shutil, 'copyfile', side_effect=AssertionError('no copy')):
                result = launch.engine.saved.stage_cache(dict(cache_root=str(root), data_root=str(launch.DATA),
                    cache_source=str(launch.CACHE_SOURCE)), manifest)
            self.assertEqual(result['copied_bytes'], 0)
            self.assertFalse(result['cache_audit_repeated'])


if __name__ == '__main__':
    unittest.main()
