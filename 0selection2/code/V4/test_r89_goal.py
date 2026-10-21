"""Four focused receipt mocks, all under temporary roots; no actual monitor launch."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from . import r89_goal as goal


class R89GoalTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        base = Path(temporary.name)
        self.root, self.launch, self.reporter = base / 'R89', base / 'launch', base / 'r89_reporting.py'
        reporter_patch = patch.object(goal, 'REPORTER_PATH', self.reporter)
        reporter_patch.start()
        self.addCleanup(reporter_patch.stop)

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

    def test_optimizer_setup_and_missing_data_are_waiting_not_failure(self):
        goal.receipts.save(self.root / goal.ARMS[0] / 'status.json',
                           dict(phase='initializing_optimizer', successful_updates=0))
        goal.receipts.save(self.launch / goal.ARMS[1] / 'status.json', dict(state='waiting_data'))
        with patch.object(goal.subprocess, 'run') as command:
            self.assertEqual(goal.watch(self.root, self.launch, max_polls=1), 0)
        command.assert_not_called()
        result = goal.receipts.read(self.root / 'goal_progress.json')
        self.assertEqual(result['phase'], 'waiting_for_arms')
        self.assertEqual(result['arms'][0]['phase'], 'initializing_optimizer')
        self.assertEqual(result['arms'][1]['phase'], 'waiting_data')
        self.assertFalse(result['errors'])
        self.assertFalse(result['goal_complete'])

    def test_ready_arms_wait_for_reporter_interface(self):
        for arm in goal.ARMS:
            self.ready(arm)
        with patch.object(goal.subprocess, 'run') as command:
            self.assertEqual(goal.watch(self.root, self.launch, max_polls=1), 0)
        command.assert_not_called()
        result = goal.receipts.read(self.root / 'goal_progress.json')
        self.assertEqual(result['phase'], 'waiting_for_reporter')
        self.assertFalse(result['error_flags'])

    def test_ready_runs_exact_cpu_compare_once_then_exits(self):
        for arm in goal.ARMS:
            self.ready(arm)
        self.reporter.write_text('# fixture interface\n')

        def compare(command, **kwargs):
            self.assertEqual(command[3:5], ['code.V4.r89_reporting', 'compare'])
            self.assertEqual(command[-4:], ['--root', str(self.root), '--device', 'cpu'])
            self.assertEqual(kwargs['env']['CUDA_VISIBLE_DEVICES'], '')
            self.assertEqual(kwargs['env']['OMP_NUM_THREADS'], '2')
            destination = self.root / 'comparison_artifacts'
            goal.receipts.save(destination / 'protocol.json', dict(experiment='R89', complete=True))
            (destination / 'COMPARISON.pdf').write_bytes(b'%PDF-fixture')
            return SimpleNamespace(returncode=0)

        with patch.object(goal.subprocess, 'run', side_effect=compare) as command:
            self.assertEqual(goal.watch(self.root, self.launch, max_polls=1), 0)
        self.assertEqual(command.call_count, 1)
        result = goal.receipts.read(self.root / 'goal_progress.json')
        self.assertTrue(result['comparison_complete'])
        self.assertFalse(result['goal_complete'])
        self.assertEqual(result['goal_status'], 'active_unresolved')

    def test_explicit_error_stops_without_retries_or_receipt_changes(self):
        path = self.launch / goal.ARMS[1] / 'worker.json'
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
