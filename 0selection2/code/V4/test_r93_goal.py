"""Fast receipt/publication regressions; every external command is mocked."""

import contextlib
import fcntl
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from . import r93_goal as goal


class GoalTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.base = Path(temporary.name)
        self.root, self.launch = self.base / 'model', self.base / 'launch'

    def completed_arms(self):
        locks = {}
        for arm in goal.ARMS:
            run = self.root / arm
            lock = dict(experiment='R93', arm=arm, selected_without_test=True,
                per_problem_assembly=False, selected_update=1440,
                checkpoint_sha256=arm * 64, state_hash='protocol-' + arm)
            locks[arm] = lock
            records = {
                'status.json': dict(phase='test_complete', successful_updates=57600),
                'training_complete.json': dict(successful_updates=57600),
                'final_test_lock.json': lock,
                'test_results.json': dict(complete=True, checkpoint_sha256=lock['checkpoint_sha256'],
                    state_hash=lock['state_hash'], result=dict(per_problem=dict.fromkeys(goal.PROBLEMS, {}),
                        identity=dict(checkpoint_sha256=lock['checkpoint_sha256'], state_hash=lock['state_hash']))),
                'lib_stage_complete.json': dict(complete=True, exit_status=0,
                    checkpoint_sha256=lock['checkpoint_sha256'], datasets=['TSPLIB', 'CVRPLIB']),
                'postprocess_status.json': dict(complete=True, phase='complete', exit_status=0),
                'full_model_statistics/protocol.json': dict(complete=True, checkpoint_lock=lock),
            }
            for name, value in records.items():
                goal.save(run / name, value)
            for name in ('MODEL_STATISTICS.pdf', 'selection_distributions.pdf', 'target_recovery_and_train.pdf'):
                (run / 'full_model_statistics' / name).write_bytes(b'%PDF-fixture')
            for name in ('controller.json', 'worker.json'):
                goal.save(self.launch / arm / name, dict(state='complete', exit_status=0,
                    stages=[dict(name='fixture', state='complete', exit_status=0)]))
        return locks

    def completed_comparison(self, locks):
        goal.save(self.root / 'comparison_artifacts/protocol.json',
            dict(complete=True, exit_status=0, checkpoints=locks))
        (self.root / 'comparison_artifacts/COMPARISON.pdf').write_bytes(b'%PDF-fixture')

    def test_status_requires_full_budgets_full18_lib_pdf_and_both_rc0(self):
        locks = self.completed_arms()
        self.assertTrue(goal.status(self.root, self.launch)['all_arms_ready'])
        self.completed_comparison(locks)
        self.assertTrue(goal.status(self.root, self.launch)['comparison_complete'])
        goal.save(self.root / 'B/training_complete.json', dict(successful_updates=57599))
        self.assertFalse(goal.status(self.root, self.launch)['comparison_complete'])
        self.completed_arms()
        path = self.root / 'B/test_results.json'
        record = goal.optional(path)
        record['result']['per_problem'].pop('ATSP')
        goal.save(path, record)
        self.assertFalse(goal.status(self.root, self.launch)['all_arms_ready'])

    def test_failed_stage_updates_terminal_receipt_no_cuda_retry(self):
        path = self.launch / 'A/worker.json'
        original = dict(state='failed', exit_status=1, error='fixture CUDA failure')
        goal.save(path, original)
        with patch.object(goal.subprocess, 'run') as run, patch.object(goal.time, 'sleep') as sleep:
            self.assertEqual(goal.watch(self.root, self.launch), 1)
        run.assert_not_called()
        sleep.assert_not_called()
        self.assertEqual(goal.optional(path), original)
        receipt = goal.optional(self.root / 'goal_watch_receipt.json')
        self.assertEqual(receipt['state'], 'stopped_on_error')
        self.assertEqual(receipt['exit_status'], 1)
        self.assertFalse(receipt['automatic_retry'])

    def test_nested_nonzero_rc_is_failure_even_if_parent_claims_complete(self):
        self.completed_arms()
        goal.save(self.launch / 'A/controller.json', dict(state='complete', exit_status=0,
            stages=[dict(name='PDF', state='complete', exit_status=7)]))
        self.assertTrue(goal.status(self.root, self.launch)['errors'])

    def test_comparison_never_bypasses_incomplete_arm(self):
        with patch.object(goal.subprocess, 'run') as run:
            with self.assertRaises(RuntimeError):
                goal.compare(self.root, self.launch)
        run.assert_not_called()

    def test_successful_compare_is_cpu_once_and_forwards_reference_roots(self):
        locks = self.completed_arms()

        def command(argv, **kwargs):
            self.assertEqual(argv[4:6], ['code.V4.r93_reporting', 'compare'])
            self.assertIn('--r89-root', argv)
            self.assertEqual(kwargs['env']['CUDA_VISIBLE_DEVICES'], '')
            self.assertEqual(kwargs['env']['OMP_NUM_THREADS'], '2')
            self.completed_comparison(locks)
            return SimpleNamespace(returncode=0)

        with patch.object(goal.subprocess, 'run', side_effect=command) as run:
            goal.compare(self.root, self.launch, r89_root=self.base / 'r89')
            with self.assertRaises(RuntimeError):
                goal.compare(self.root, self.launch)
        self.assertEqual(run.call_count, 1)
        self.assertEqual(goal.optional(self.root / 'goal_compare_receipt.json')['state'], 'complete')

    def test_rc0_without_delivery_is_failed_and_cannot_repeat(self):
        self.completed_arms()
        with patch.object(goal.subprocess, 'run', return_value=SimpleNamespace(returncode=0)) as run:
            with self.assertRaises(RuntimeError):
                goal.compare(self.root, self.launch)
            with self.assertRaises(RuntimeError):
                goal.compare(self.root, self.launch)
        self.assertEqual(run.call_count, 1)
        self.assertEqual(goal.optional(self.root / 'goal_compare_receipt.json')['state'], 'failed')

    def test_success_updates_completed_receipt_without_publishing_by_default(self):
        locks = self.completed_arms()
        self.completed_comparison(locks)
        with patch.object(goal, 'publish') as publish, patch.object(goal.time, 'sleep') as sleep:
            self.assertEqual(goal.watch(self.root, self.launch), 0)
        publish.assert_not_called()
        sleep.assert_not_called()
        receipt = goal.optional(self.root / 'goal_watch_receipt.json')
        self.assertEqual(receipt['state'], 'completed')
        self.assertTrue(receipt['pipeline_complete'])
        self.assertFalse(receipt['goal_complete'])

    def test_compare_exception_updates_terminal_watcher_receipt(self):
        self.completed_arms()
        with patch.object(goal, 'compare', side_effect=RuntimeError('fixture compare failed')):
            self.assertEqual(goal.watch(self.root, self.launch), 1)
        self.assertEqual(goal.optional(self.root / 'goal_watch_receipt.json')['state'], 'stopped_on_error')

    def test_pub_failure_updates_terminal_watcher_receipt(self):
        locks = self.completed_arms()
        self.completed_comparison(locks)
        with patch.object(goal, 'publish', side_effect=RuntimeError('fixture push failed')):
            self.assertEqual(goal.watch(self.root, self.launch, publish_github=True), 1)
        self.assertEqual(goal.optional(self.root / 'goal_watch_receipt.json')['state'], 'stopped_on_error')

    def publication_context(self):
        repo = self.base / 'repo'
        root = repo / 'code/V4/runs/R93_plain_classification'
        launch = repo / 'code/V4/runs/R93_launch'
        git_dir = repo / '.git'
        git_dir.mkdir(parents=True)
        root.mkdir(parents=True)
        stack = contextlib.ExitStack()
        self.addCleanup(stack.close)
        stack.enter_context(patch.object(goal, 'REPO', repo))
        stack.enter_context(patch.object(goal, 'ROOT', root))
        stack.enter_context(patch.object(goal, 'LAUNCH', launch))
        return repo, root, launch, git_dir

    def test_curated_assets_exclude_other_versions_checkpoints_caches_labels_and_secrets(self):
        repo, root, launch, _ = self.publication_context()
        files = {
            'code/V4/r93_reporting.py': b'# fixture',
            'code/V4/test_r93_reporting.py': b'# fixture',
            'code/V4/r92_reporting.py': b'# frozen',
            'code/V4/runs/R77/plot.png': b'other dirty file',
            'docs/research/post_r92_broad_research.md': b'allowed research note',
            'code/V4/runs/R93_plain_classification/A/best.pt': b'weights',
            'code/V4/runs/R93_plain_classification/A/labels.json': b'{}',
            'code/V4/runs/R93_plain_classification/A/cache/item.json': b'{}',
            'code/V4/runs/R93_plain_classification/A/wandb/run.json': b'{}',
            'code/V4/runs/R93_plain_classification/A/evaluation/predictions.npz': b'raw',
            'code/V4/runs/R93_plain_classification/A/full_model_statistics/metrics_by_problem.csv': b'cost',
            'code/V4/runs/R93_plain_classification/goal_watch_receipt.json': b'{}',
        }
        for name, value in files.items():
            path = repo / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(value)
        curated = goal.publication_paths(root, launch, repo)
        self.assertEqual(set(curated), {
            'code/V4/r93_reporting.py', 'code/V4/test_r93_reporting.py',
            'docs/research/post_r92_broad_research.md',
            'code/V4/runs/R93_plain_classification/A/full_model_statistics/metrics_by_problem.csv'})
        goal.save(root / 'A/unsafe.json', dict(access_token='must never stage'))
        with self.assertRaises(RuntimeError):
            goal.publication_paths(root, launch, repo)

    def test_publication_global_mutex_covers_checks_staging_commit_and_push(self):
        repo, root, launch, git_dir = self.publication_context()
        path = repo / 'code/V4/r93_reporting.py'
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('# fixture')
        staged = []
        commands = []

        def assert_locked():
            with (git_dir / 'codex-publication.lock').open('a') as probe:
                with self.assertRaises(BlockingIOError):
                    fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)

        def git(argv, cwd):
            if argv == ['rev-parse', '--show-toplevel']:
                return str(repo)
            if argv == ['rev-parse', '--absolute-git-dir']:
                return str(git_dir)
            assert_locked()
            if argv == ['branch', '--show-current']:
                return 'main'
            if argv[:3] == ['diff', '--cached', '--name-only']:
                return '\0'.join(staged)
            if argv == ['rev-parse', 'HEAD']:
                return 'one-commit'
            self.fail(argv)

        def command(argv, **kwargs):
            assert_locked()
            commands.append(argv)
            if argv[1] == 'add':
                staged.extend(argv[4:])
            return SimpleNamespace(returncode=0)

        with patch.object(goal, 'status', return_value=dict(errors=[], comparison_complete=True)), \
                patch.object(goal, 'git_output', side_effect=git), \
                patch.object(goal.subprocess, 'run', side_effect=command):
            receipt = goal.publish(root, launch)
            self.assertTrue(receipt['complete'])
            goal.publish(root, launch)
        self.assertEqual([argv[1] for argv in commands], ['add', 'commit', 'push'])
        self.assertEqual(commands[-1][-1], 'HEAD:refs/heads/main')
        self.assertTrue((git_dir / 'codex-publication.lock').is_file())

    def test_preexisting_unrelated_staged_change_is_not_touched(self):
        repo, root, launch, git_dir = self.publication_context()

        def git(argv, cwd):
            if argv == ['rev-parse', '--show-toplevel']:
                return str(repo)
            if argv == ['rev-parse', '--absolute-git-dir']:
                return str(git_dir)
            if argv == ['branch', '--show-current']:
                return 'main'
            return '0selection2/code/V4/runs/R77/dirty_plot.png'

        with patch.object(goal, 'status', return_value=dict(errors=[], comparison_complete=True)), \
                patch.object(goal, 'git_output', side_effect=git), patch.object(goal.subprocess, 'run') as run:
            with self.assertRaises(RuntimeError):
                goal.publish(root, launch)
        run.assert_not_called()
        self.assertFalse((root / 'github_publication.json').exists())


if __name__ == '__main__':
    unittest.main()
