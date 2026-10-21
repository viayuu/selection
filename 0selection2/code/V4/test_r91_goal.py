"""Three bounded CPU receipt checks, no training or inference."""

from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from . import r91_goal as goal


class GoalTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name) / 'model'
        self.launch = Path(temporary.name) / 'launch'

    def test_waiting_queue_and_active_run_are_not_failures(self):
        goal.receipts.save(self.launch / goal.ARMS[0] / 'status.json', dict(state='waiting_resources'))
        value = goal.status(self.root, self.launch)
        self.assertFalse(value['errors'])
        self.assertFalse(value['goal_complete'])
        self.assertEqual(value['arms'][0]['phase'], 'waiting_resources')
        goal.receipts.save(self.root / goal.ARMS[0] / 'status.json', dict(phase='training', successful_updates=18))
        value = goal.status(self.root, self.launch)
        self.assertEqual(value['arms'][0]['phase'], 'training')

    def test_explicit_worker_failure_stops_without_retry(self):
        path = self.launch / goal.ARMS[0] / 'worker.json'
        record = dict(state='failed', error='fixture rc1')
        goal.receipts.save(path, record)
        with patch.object(goal.subprocess, 'run') as run, patch.object(goal.time, 'sleep') as sleep:
            self.assertEqual(goal.watch(self.root, self.launch), 1)
        run.assert_not_called()
        sleep.assert_not_called()
        self.assertEqual(goal.receipts.read(path), record)
        self.assertFalse(goal.receipts.read(self.root / 'goal_progress.json')['automatic_retry'])

    def test_comparison_is_cpu_only_and_cannot_automatically_repeat(self):
        def result(command, **kwargs):
            self.assertEqual(command[4:6], ['code.V4.r91_reporting', 'compare'])
            self.assertEqual(command[-2:], ['--device', 'cpu'])
            self.assertEqual(kwargs['env']['CUDA_VISIBLE_DEVICES'], '')
            return SimpleNamespace(returncode=0)
        self.root.mkdir(parents=True)
        with patch.object(goal.subprocess, 'run', side_effect=result), \
                patch.object(goal, 'status', return_value=dict(comparison_complete=True)):
            goal.compare(self.root, self.launch)
            with self.assertRaises(RuntimeError):
                goal.compare(self.root, self.launch)
        receipt = goal.receipts.read(self.root / 'goal_compare_receipt.json')
        self.assertEqual(receipt['state'], 'complete')
        self.assertFalse(receipt['automatic_retry'])


if __name__ == '__main__':
    unittest.main()
