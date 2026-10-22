"""Bounded CPU tests for fixed R94 queues, without Slurm or GPU probing."""

from contextlib import nullcontext
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from . import r94_launch as launch
from .r76_experiment import write_json


class R94LaunchTests(unittest.TestCase):
    def test_fixed_allocated_slot_mapping(self):
        self.assertEqual(launch.ARM_SLOTS, dict(A='3090_1', B='4090_1'))
        self.assertEqual(launch.SLOTS['3090_1']['job'], 1465)
        self.assertEqual(launch.SLOTS['4090_1']['job'], 1475)
        self.assertNotEqual(launch.SLOTS['3090_1']['uuid'], launch.SLOTS['4090_1']['uuid'])

    def test_stage_order_and_one_fresh_train(self):
        for arm in launch.ARMS:
            spec = launch.template(arm)
            stages = launch.commands(spec, launch.ARM_SLOTS[arm])
            self.assertEqual([row['name'] for row in stages],
                ['input_features', 'train', 'TEST', 'timing', 'LIB', 'PDF'])
            self.assertEqual([row['device'] for row in stages], ['gpu'] * 5 + ['cpu'])
            self.assertEqual(sum(row['argv'][5] == 'train' for row in stages), 1)
            self.assertFalse(any('--resume' in row['argv'] for row in stages))
            self.assertEqual(spec['protocol']['logical_batch'], 1024)
            self.assertEqual(spec['protocol']['arm_budget']['count'], 57600)

    def test_parser_contract_no_launch(self):
        result = launch.parse_commands()
        self.assertEqual(result['parsed_count'], 13)
        self.assertFalse(result['resource_probe_performed'])
        self.assertFalse(result['training_started'])

    def test_wrong_slot_rejected(self):
        with self.assertRaises(ValueError):
            launch.commands(launch.template('A'), '4090_1')

    def test_missing_predecessor_waits_and_failure_does_not_restart(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.object(launch, 'PREDECESSOR_ROOT', root / 'models'), \
                    patch.object(launch, 'PREDECESSOR_LAUNCH', root / 'launch'):
                self.assertFalse(launch.predecessor_gate('A')['complete'])
                self.assertFalse(launch.predecessor_gate('A')['failed'])
                write_json(root / 'launch/A/worker.json', dict(state='failed', exit_status=1))
                self.assertTrue(launch.predecessor_gate('A')['failed'])

    def test_queue_starts_exactly_once_only_after_completion(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            gates = [dict(complete=False, failed=False), dict(complete=True, failed=False)]
            with patch.object(launch, 'queue_prepared', return_value=(dict(run_id='A'), {})), \
                    patch.object(launch.base, 'receipts', return_value=root), \
                    patch.object(launch.base, 'lock', return_value=nullcontext()), \
                    patch.object(launch, 'predecessor_gate', side_effect=gates), \
                    patch.object(launch.time, 'sleep') as sleep, \
                    patch.object(launch.adapter, 'start', return_value=0) as start:
                self.assertEqual(launch.queue(root / 'spec.json', 'ready', True), 0)
                start.assert_called_once()
                sleep.assert_called_once()

    def test_attempted_arm_never_auto_retried(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            write_json(root / 'attempt.json', dict(exit_status=1))
            with patch.object(launch, 'queue_prepared', return_value=(dict(run_id='A'), {})), \
                    patch.object(launch.base, 'receipts', return_value=root), \
                    patch.object(launch.base, 'lock', return_value=nullcontext()), \
                    patch.object(launch.adapter, 'start') as start:
                with self.assertRaises(RuntimeError):
                    launch.queue(root / 'spec.json', 'ready', True)
                start.assert_not_called()


if __name__ == '__main__':
    unittest.main()
