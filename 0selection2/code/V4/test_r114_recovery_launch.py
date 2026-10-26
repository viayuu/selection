"""Small CPU fixtures; never start an allocation, native solver or CUDA."""

import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

from . import r114_recovery_launch as launch


class RecoveryLaunchTests(unittest.TestCase):
    def test_exact_existing_3090_and_full_acquisition_arguments(self):
        plan = dict(manifest_path='/frozen/manifest.json', deployment_lock_path='/frozen/lock.json')
        with patch.object(launch.resource, 'read', return_value=plan):
            argv = launch.command('/frozen/plan.json', [16, 17, 64, 65])
        self.assertEqual(argv[argv.index('--gpu-uuid') + 1],
                         'GPU-cb626bdb-4319-2f5e-b91f-17fcaa44a83d')
        self.assertEqual(argv[argv.index('--cpu-ids') + 1], '16,17')
        self.assertEqual(argv[argv.index('--host') + 1], launch.resource.CONTAINER[:12])
        self.assertIn('--execute', argv)
        self.assertIn('--finalize', argv)
        self.assertNotIn('--max-jobs', argv)

    def test_source_change_and_wrong_slot_refuse_before_any_process(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'source.py'
            source.write_text('frozen\n')
            config = root / 'ready.json'
            value = dict(main_approved=True, slot=launch.SLOT,
                         sources={str(source): launch.resource.digest(source)})
            launch.resource.save(config, value)
            with patch.object(launch, 'CONFIG', config), patch.object(launch.subprocess, 'Popen') as popen:
                self.assertEqual(launch.load(), value)
                source.write_text('changed\n')
                with self.assertRaisesRegex(RuntimeError, 'Frozen recovery source changed'):
                    launch.load()
                value['slot'] = '4090_0'
                launch.resource.save(config, value)
                with self.assertRaisesRegex(RuntimeError, 'existing two-3090 followup'):
                    launch.load()
                popen.assert_not_called()


if __name__ == '__main__':
    unittest.main()
