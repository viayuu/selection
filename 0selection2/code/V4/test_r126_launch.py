"""CPU-only R126 launch isolation checks; never dispatch a real training run."""

import importlib
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from . import r125_launch as r125
from . import r126_launch as launch


class LaunchTests(unittest.TestCase):
    def test_private_transport_does_not_change_r125(self):
        self.assertIsNot(launch.transport, r125)
        self.assertEqual(r125.SLOT, '4090_0')
        self.assertEqual(r125.ARM, 'code_seed2')
        self.assertIn('R125_no_bq_lehd', str(r125.ROOT))
        self.assertEqual(launch.transport.SLOT, '4090_1')
        self.assertEqual(launch.CPUS, [4, 5, 72, 73])
        self.assertIs(launch.transport.command, launch.command)
        self.assertNotEqual(launch.MUTEX, r125.MUTEX)
        self.assertEqual(importlib.import_module('code.V4.r84_launch').SLOTS['4090_0']['uuid'], r125.ITEM['uuid'])

    def test_exact_uuid_and_disk_wrapper(self):
        argv = launch.transport.container_command('worker', '--attempt', '/shared/r126', gpu=True, token='test')
        self.assertIn('CUDA_VISIBLE_DEVICES=GPU-597c682a-4b7f-7609-80a6-8ced791f5c5d', argv)
        self.assertIn('SLURM_JOB_ID=1475', argv)
        self.assertIn(str(launch.WRAPPER), argv)
        self.assertNotIn(str(r125.WRAPPER), argv)
        self.assertNotIn('-c', argv)
        self.assertNotIn('-lc', argv)
        self.assertIn('R126_LAUNCH_TOKEN=test', argv)

    def test_all_stage_argv_are_r126_and_recipe_is_fixed(self):
        for stage in ('prepare', 'train', 'test', 'timing', 'lib', 'report', 'publish'):
            argv = launch.command(stage, launch.CACHE)
            self.assertIn('code.V4.r126_', ' '.join(argv))
            self.assertNotIn('r125', ' '.join(argv).lower())
            self.assertNotIn('--resume', argv)
        train = launch.command('train', launch.CACHE)
        self.assertEqual(train[train.index('--arm') + 1], 'model_only_seed2')
        self.assertEqual(train[train.index('--large-micro-cap') + 1], '256')
        self.assertEqual(train[train.index('--eval-batch-size') + 1], '128')
        self.assertIn('--push', launch.command('publish', launch.CACHE))

    def test_no_start_without_main_and_no_retraining(self):
        with mock.patch.object(launch.subprocess, 'run') as run:
            with self.assertRaisesRegex(ValueError, 'main-ready'):
                launch.start('train')
            run.assert_not_called()
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(launch.transport, 'ROOT', Path(directory)):
            run = Path(directory) / launch.ARM
            launch.save(run / 'status.json', dict(phase='failed'))
            with self.assertRaisesRegex(RuntimeError, 'retraining'):
                launch.transport.fresh_train()


if __name__ == '__main__':
    unittest.main()
