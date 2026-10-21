"""Three small CPU receipt tests; no weights, inference or actual reporter runs."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from . import r88_goal as goal


class R88GoalTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root, self.launch = Path(temporary.name) / 'R88', Path(temporary.name) / 'launch'

    def ready(self, arm):
        run = self.root / arm
        for filename, record in (
                ('status.json', dict(phase='training_complete', successful_updates=57600)),
                ('training_complete.json', dict(successful_updates=57600)),
                ('final_test_lock.json', dict(selected_without_test=True, per_problem_assembly=False)),
                ('test_results.json', dict(complete=True)),
                ('lib_stage_complete.json', dict(complete=True)),
                ('postprocess_status.json', dict(complete=True, phase='complete')),
                ('full_model_statistics/protocol.json', dict(complete=True))):
            goal.receipts.save(run / filename, record)
        for filename in ('MODEL_STATISTICS.pdf', 'selection_distributions.pdf'):
            (run / 'full_model_statistics' / filename).write_bytes(b'%PDF-fixture')

    def test_wait_does_not_compare_or_treat_queue_as_failure(self):
        goal.receipts.save(self.launch / goal.ARMS[1] / 'status.json', dict(state='waiting_predecessor_delivery'))
        with patch.object(goal.subprocess, 'run') as command:
            self.assertEqual(goal.watch(self.root, self.launch, max_polls=1), 0)
        command.assert_not_called()
        progress = goal.receipts.read(self.root / 'goal_progress.json')
        self.assertEqual(progress['phase'], 'waiting_for_arms')
        self.assertEqual(progress['arms'][1]['phase'], 'waiting_predecessor_delivery')
        self.assertFalse(progress['errors'])
        self.assertFalse(progress['goal_complete'])

    def test_ready_compares_once_on_cpu_and_exits_with_goal_active(self):
        for arm in goal.ARMS:
            self.ready(arm)

        def compare(command, **kwargs):
            self.assertEqual(command[3:5], ['code.V4.r88_reporting', 'compare'])
            self.assertEqual(command[-2:], ['--device', 'cpu'])
            self.assertEqual(kwargs['env']['CUDA_VISIBLE_DEVICES'], '')
            self.assertEqual(kwargs['env']['OMP_NUM_THREADS'], '2')
            destination = self.root / 'comparison_artifacts'
            goal.receipts.save(destination / 'protocol.json', dict(experiment='R88', complete=True))
            (destination / 'COMPARISON.pdf').write_bytes(b'%PDF-fixture')
            return SimpleNamespace(returncode=0)

        with patch.object(goal.subprocess, 'run', side_effect=compare) as command:
            self.assertEqual(goal.watch(self.root, self.launch, max_polls=1), 0)
        self.assertEqual(command.call_count, 1)
        progress = goal.receipts.read(self.root / 'goal_progress.json')
        self.assertTrue(progress['comparison_complete'])
        self.assertFalse(progress['goal_complete'])
        self.assertEqual(progress['goal_status'], 'active_unresolved')

    def test_errors_stop_without_retries_and_preserve_receipts(self):
        path = self.launch / goal.ARMS[0] / 'worker.json'
        original = dict(state='failed', error='fixture stage rc1')
        goal.receipts.save(path, original)
        with patch.object(goal.subprocess, 'run') as command, patch.object(goal.time, 'sleep') as sleep:
            self.assertEqual(goal.watch(self.root, self.launch), 1)
        command.assert_not_called()
        sleep.assert_not_called()
        self.assertEqual(goal.receipts.read(path), original)
        self.assertEqual(goal.receipts.read(self.root / 'goal_watch_receipt.json')['state'], 'stopped_on_error')
        self.assertFalse(goal.receipts.read(self.root / 'goal_progress.json')['automatic_retry'])


if __name__ == '__main__':
    unittest.main()
