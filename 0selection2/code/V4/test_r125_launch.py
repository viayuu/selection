"""Small CPU-only safety checks; no scheduler, GPU, training or publication."""

from pathlib import Path
import tempfile
import unittest
from unittest import mock

from . import r125_launch as launch


class LaunchTests(unittest.TestCase):
    def test_transport_is_exact_container_uuid_and_disk_script(self):
        argv = launch.container_command('worker', '--attempt', '/shared/attempt', gpu=True, token='test')
        self.assertIn(launch.CONTAINER, argv)
        self.assertIn('CUDA_VISIBLE_DEVICES=' + launch.ITEM['uuid'], argv)
        self.assertIn(str(launch.WRAPPER), argv)
        self.assertNotIn('-c', argv)
        self.assertNotIn('-lc', argv)
        self.assertIn('SLURM_JOB_ID=1475', argv)

    def test_stage_recipe_and_publish_order(self):
        cache = str(launch.HOST_CACHE)
        prepare = launch.command('prepare', cache)
        self.assertIn('code.V4.r125_data', prepare)
        self.assertEqual(prepare[prepare.index('--cache-root') + 1], cache)
        train = launch.command('train', cache)
        self.assertEqual(train[train.index('--large-micro-cap') + 1], '256')
        self.assertEqual(train[train.index('--eval-batch-size') + 1], '128')
        self.assertNotIn('--resume', train)
        self.assertIn('--push', launch.command('publish', cache))

    def test_start_requires_main_confirmation_and_never_retrains(self):
        with mock.patch.object(launch.subprocess, 'run') as run:
            with self.assertRaisesRegex(ValueError, 'main-ready'):
                launch.start('train')
            run.assert_not_called()
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(launch, 'ROOT', Path(directory)):
            run = Path(directory) / launch.ARM
            run.mkdir()
            launch.save(run / 'status.json', dict(phase='failed'))
            with self.assertRaisesRegex(RuntimeError, 'retraining'):
                launch.fresh_train()

    def test_first_cycle_checks_all_tasks_and_finite_metrics(self):
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(launch, 'ROOT', Path(directory)):
            root = Path(directory)
            status = dict(successful_updates=18, updates_by_problem={str(i): 1 for i in range(18)},
                loss=1., ce=1., cmp=.5, risk=.1, kl=.02, grad_norm=1.)
            launch.save(root / launch.ARM / 'status.json', status)
            self.assertTrue(launch.first_cycle(root / 'attempt'))
            status['loss'] = 'nan'
            launch.save(root / launch.ARM / 'status.json', status)
            with self.assertRaisesRegex(RuntimeError, 'nonfinite'):
                launch.first_cycle(root / 'invalid')


if __name__ == '__main__':
    unittest.main()
