import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from . import r98_launch as launcher


class LaunchTests(unittest.TestCase):
    def test_unique_bindings(self):
        self.assertEqual(len(launcher.SLOTS), 4)
        self.assertEqual(len({row['uuid'] for row in launcher.SLOTS.values()}), 4)
        for slot, row in launcher.SLOTS.items():
            with patch.dict(launcher.os.environ, {}, clear=True):
                env = launcher.environment(slot, 'one-token')
            self.assertEqual(env['CUDA_VISIBLE_DEVICES'], row['uuid'])
            self.assertEqual(env['WANDB_MODE'], 'offline')
            self.assertEqual(env[launcher.TOKEN_ENV], 'one-token')

    def test_only_authorized_container_and_cpu_step(self):
        for slot in launcher.SLOTS:
            argv = launcher.container_command('worker', slot, '/tmp/spec.json', 'token')
            self.assertEqual(argv[:4], ['sudo', '-n', '/usr/bin/ai_docker', 'exec'])
            item = launcher.SLOTS[slot]
            self.assertIn(launcher.base.transport.CONTAINERS[item['node']][1], argv)
            self.assertIn(str(launcher.WRAPPER), argv)
            srun = launcher.base.srun_command('entry', slot, '/tmp/spec.json', 'token', cpus=4, memory='9G')
            self.assertIn('--gres=none', srun)
            self.assertIn('--cpus-per-task=4', srun)
            self.assertIn('--mem=9G', srun)
            self.assertIn('code.V4.r98_launch', ' '.join(srun))

    def test_no_old_module_global_mutation(self):
        from . import r84_launch, r81_queue
        self.assertNotEqual(launcher.base.ROOT, r84_launch.ROOT)
        self.assertIsNot(launcher.base, r84_launch)
        self.assertIsNot(launcher.base.transport, r81_queue)
        self.assertNotEqual(r84_launch.TOKEN_ENV, launcher.TOKEN_ENV)

    def test_explicit_readiness(self):
        for evidence, tested in ((None, True), ('', True), ('passed', False)):
            with self.assertRaises(ValueError):
                launcher.prepare('/not/read.json', evidence, tested)

    def test_source_closure_is_snapshot_and_includes_dependencies(self):
        roots = launcher.source_closure([launcher.REPO/'code/V4/r99_model.py'])
        self.assertIn(str(launcher.REPO/'code/V4/r99_relations.py'), roots)
        self.assertIn(str(launcher.REPO/'code/V4/V4Model.py'), roots)
        self.assertEqual(roots, sorted(set(roots)))

    def test_parser_does_not_launch(self):
        parsed = launcher.parser().parse_args(['start', '--spec', '/tmp/spec.json',
                                  '--evidence', 'focused tests', '--runner-tests-passed'])
        self.assertEqual(parsed.stage, 'start')
        self.assertTrue(parsed.runner_tests_passed)


if __name__ == '__main__':
    unittest.main()
