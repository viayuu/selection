"""CPU-only launch isolation and argv checks; no GPU or training dispatch."""

from pathlib import Path
import tempfile
import unittest
from unittest import mock

from . import r125_launch as r125
from . import r126_launch as r126
from . import r127_launch as launch


class LaunchTests(unittest.TestCase):
    def test_private_transport_leaves_both_running_launchers_unchanged(self):
        self.assertIsNot(launch.transport, r125)
        self.assertIsNot(launch.transport, r126.transport)
        self.assertEqual((r125.SLOT, r126.SLOT), ('4090_0', '4090_1'))
        self.assertEqual((r125.ITEM['node'], r126.ITEM['node']), ('gpu05', 'gpu05'))
        self.assertEqual((launch.SLOT, launch.ITEM['node'], launch.ITEM['job']), ('3090_0', 'gpu03', 1465))
        self.assertEqual(launch.CPUS, [12, 13, 14, 15])
        self.assertIs(launch.transport.command, launch.command)
        self.assertNotIn(launch.MUTEX, (r125.MUTEX, r126.MUTEX))

    def test_exact_container_uuid_and_disk_script(self):
        argv = launch.transport.container_command('worker', '--attempt', '/shared/r127', gpu=True, token='test')
        self.assertIn('CUDA_VISIBLE_DEVICES=GPU-cb24f168-9f18-f0d5-6f4a-6d2406958f50', argv)
        self.assertIn('SLURM_JOB_ID=1465', argv)
        self.assertIn('b8ced5d8a8cdafdc5006f827a0d36c319c61056a65316af4ff9937453fa0635c', argv)
        self.assertIn(str(launch.WRAPPER), argv)
        self.assertIn('R127_LAUNCH_TOKEN=test', argv)
        self.assertNotIn('-c', argv)
        self.assertNotIn('-lc', argv)
        ssh = launch.transport.ssh_command('worker', '--attempt', '/shared/r127')
        self.assertEqual(ssh[-2], 'gpu03')

    def test_only_r127_argv_and_safe_3090_recipe(self):
        for stage in ('prepare', 'train', 'test', 'timing', 'lib', 'report', 'publish'):
            argv = launch.command(stage, launch.CACHE)
            self.assertIn('code.V4.r127_', ' '.join(argv))
            self.assertNotIn('r125', ' '.join(argv).lower())
            self.assertNotIn('r126', ' '.join(argv).lower())
            self.assertNotIn('--resume', argv)
        train = launch.command('train', launch.CACHE)
        self.assertEqual(train[train.index('--gpu-type') + 1], '3090')
        self.assertEqual(train[train.index('--large-micro-cap') + 1], '256')
        self.assertEqual(train[train.index('--eval-batch-size') + 1], '128')
        self.assertEqual(train[train.index('--arm') + 1], 'model_only_seed2')
        self.assertIn('--push', launch.command('publish', launch.CACHE))

    def test_requires_main_confirmation_and_never_retrains(self):
        with mock.patch.object(launch.subprocess, 'run') as run:
            with self.assertRaisesRegex(ValueError, 'main-ready'):
                launch.start('train')
            run.assert_not_called()
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(launch.transport, 'ROOT', Path(directory)):
            launch.save(Path(directory) / launch.ARM / 'status.json', dict(phase='failed'))
            with self.assertRaisesRegex(RuntimeError, 'retraining'):
                launch.transport.fresh_train()


if __name__ == '__main__':
    unittest.main()
