"""Focused CPU completion/publication tests; all external commands are mocked."""

from contextlib import ExitStack
import copy
import fcntl
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from . import r95_goal as goal
from .test_r95_launch import protocol_fixture
from .test_r95_reporting import config, proposal_fixture, split_result


class GoalTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.root, self.launch = self.base / 'models', self.base / 'launch'
        self.root.mkdir()

    def completed_arms(self):
        scientific = protocol_fixture()
        goal.save(self.root / 'protocol_lock.json', scientific)
        locks = {}
        for arm in goal.ARMS:
            run, launch = self.root / arm, self.launch / arm
            cfg = config(arm)
            cfg['main_protocol_lock'] = dict(record=scientific,
                sha256=goal.file_hash(self.root / 'protocol_lock.json'))
            goal.save(run / 'protocol.json', cfg)
            lock = dict(experiment='R95', arm=arm, selected_update=1440,
                checkpoint_sha256=arm + '-checkpoint', state_hash=goal.json_hash(cfg),
                selected_without_test=True, per_problem_assembly=False,
                rule=cfg['checkpoint_rule'], inference_rule=cfg['inference_rule'])
            locks[arm] = lock
            goal.save(run / 'final_test_lock.json', lock)
            train, val, test = [split_result(arm, name, lock) for name in ('train', 'val', 'test')]
            goal.save(run / 'best_eval.json', dict(successful_updates=1440, train=train, val=val))
            goal.save(run / 'training_complete.json', dict(successful_updates=57600,
                selected_best=lock, state_hash=lock['state_hash'],
                updates_by_problem={p: 3200 for p in goal.PROBLEMS}))
            goal.save(run / 'final_metrics.json', dict(complete=True, test_read=True,
                selected_update=1440, checkpoint_sha256=lock['checkpoint_sha256'],
                state_hash=lock['state_hash'], train=train, val=val, test=test))
            goal.save(run / 'test_results.json', dict(complete=True, result=test,
                checkpoint_sha256=lock['checkpoint_sha256'], state_hash=lock['state_hash']))
            goal.save(run / 'test_complete.json', dict(full_tests=1,
                checkpoint_sha256=lock['checkpoint_sha256'], saved_predictions_reusable=True))
            goal.save(run / 'lib_stage_complete.json', dict(complete=True, exit_status=0,
                checkpoint_sha256=lock['checkpoint_sha256'], state_hash=lock['state_hash'],
                model_state_sha256='same-model', datasets=['TSPLIB', 'CVRPLIB']))
            goal.save(run / 'postprocess_status.json', dict(complete=True, phase='complete', exit_status=0))
            sources = {}
            for step in range(7200, 57601, 7200):
                path = run / 'proposals' / f'update{step:07d}.json'
                goal.save(path, dict(summary=True, arm=arm, step=step))
                sources[str(path)] = goal.file_hash(path)
            goal.save(run / 'full_model_statistics/protocol.json', dict(complete=True,
                checkpoint_lock=lock, rdrop=True, one_forward=False,
                sampling_diagnostics=dict(complete=True, sampling=goal.SAMPLING,
                    raw_arrays_read=False, source_sha256=sources)))
            for name in goal.REQUIRED_PDFS:
                (run / 'full_model_statistics' / name).write_bytes(b'%PDF-synthetic')
            for name, stages in (('controller', ('allocation_GPU_pipeline', 'PDF')),
                    ('worker', ('train', 'TEST', 'LIB'))):
                goal.save(launch / (name + '.json'), dict(state='complete', exit_status=0,
                    stages=[dict(name=s, state='complete', exit_status=0) for s in stages]))
            goal.save(launch / 'plan.json', dict(main_ready_evidence='all final CPU tests passed'))
            goal.save(launch / 'ready.json', dict(main_thread_ready=True, runner_tests_passed=True,
                evidence='all final CPU tests passed', plan_sha256=goal.file_hash(launch / 'plan.json')))
        return locks

    def completed_comparison(self, locks):
        sources = {arm: goal.optional(self.root / arm / 'full_model_statistics/protocol.json')['sampling_diagnostics']
            for arm in goal.ARMS}
        goal.save(self.root / 'comparison_artifacts/protocol.json', dict(experiment='R95', complete=True,
            exit_status=0, checkpoints=locks, sampling=goal.SAMPLING, sampling_diagnostics=sources, causal_claim=False))
        (self.root / 'comparison_artifacts/COMPARISON.pdf').write_bytes(b'%PDF-comparison')
        (self.root / 'RESULTS.pdf').write_bytes(b'%PDF-result')

    def test_waits_with_no_gpu_model_or_external_commands(self):
        with patch.object(goal.subprocess, 'run') as run, patch.object(goal.subprocess, 'check_output') as git:
            result = goal.status(self.root, self.launch)
        run.assert_not_called()
        git.assert_not_called()
        self.assertFalse(result['all_arms_ready'])
        self.assertFalse(result['goal_complete'])
        self.assertEqual(result['phase'], 'waiting_for_arms')

    def test_all_arms_and_comparison_must_match_complete_locks(self):
        locks = self.completed_arms()
        result = goal.status(self.root, self.launch)
        self.assertTrue(result['all_arms_ready'], result['arms'])
        self.assertFalse(result['comparison_complete'])
        self.completed_comparison(locks)
        self.assertTrue(goal.status(self.root, self.launch)['comparison_complete'])
        compare = goal.optional(self.root / 'comparison_artifacts/protocol.json')
        compare['checkpoints']['B']['checkpoint_sha256'] = 'different'
        goal.save(self.root / 'comparison_artifacts/protocol.json', compare)
        self.assertFalse(goal.status(self.root, self.launch)['comparison_complete'])

    def test_test_results_alone_and_nonfinalized_metrics_are_insufficient(self):
        self.completed_arms()
        final = goal.optional(self.root / 'A/final_metrics.json')
        final['test_read'] = False
        goal.save(self.root / 'A/final_metrics.json', final)
        result = goal.status(self.root, self.launch)
        self.assertIn('full_TEST', result['arms'][0]['missing'])
        with patch.object(goal.subprocess, 'run') as run:
            with self.assertRaises(RuntimeError):
                goal.compare(self.root, self.launch)
        run.assert_not_called()

    def test_partial_population_wrong_role_state_and_missing_sampling_are_not_complete(self):
        self.completed_arms()
        final = goal.optional(self.root / 'A/final_metrics.json')
        final['train']['per_problem']['TSP']['n'] = 8100
        goal.save(self.root / 'A/final_metrics.json', final)
        (self.root / 'B/proposals/update0007200.json').unlink()
        result = goal.status(self.root, self.launch)
        self.assertIn('full_TRAIN', result['arms'][0]['missing'])
        self.assertIn('sampling_diagnostics', result['arms'][1]['missing'])

    def test_main_ready_and_scientific_lock_are_required_before_any_git_access(self):
        locks = self.completed_arms()
        self.completed_comparison(locks)
        ready = goal.optional(self.launch / 'A/ready.json')
        ready['runner_tests_passed'] = False
        goal.save(self.launch / 'A/ready.json', ready)
        self.assertIn('main_ready', goal.status(self.root, self.launch)['arms'][0]['missing'])
        with patch.object(goal, 'git_output') as git:
            with self.assertRaises(RuntimeError):
                goal.publish(self.root, self.launch)
        git.assert_not_called()

    def test_nested_failure_stops_watcher_without_restart(self):
        self.completed_arms()
        path = self.launch / 'A/controller.json'
        original = goal.optional(path)
        original['stages'][0]['stages'] = [dict(name='nested', state='failed', exit_status=3)]
        goal.save(path, original)
        with patch.object(goal.subprocess, 'run') as run, patch.object(goal.time, 'sleep') as sleep:
            self.assertEqual(goal.watch(self.root, self.launch), 1)
        run.assert_not_called()
        sleep.assert_not_called()
        self.assertEqual(goal.optional(path), original)
        self.assertEqual(goal.optional(self.root / 'goal_watch_receipt.json')['state'], 'stopped_on_error')

    def test_compare_is_cpu_once_and_passes_reference_roots(self):
        locks = self.completed_arms()

        def command(argv, **kwargs):
            self.assertEqual(argv[4:6], ['code.V4.r95_reporting', 'compare'])
            self.assertIn('--r89-root', argv)
            self.assertEqual(kwargs['env']['CUDA_VISIBLE_DEVICES'], '')
            self.completed_comparison(locks)
            return SimpleNamespace(returncode=0)

        with patch.object(goal.subprocess, 'run', side_effect=command) as run:
            goal.compare(self.root, self.launch, r89_root=self.base / 'r89')
            with self.assertRaises(RuntimeError):
                goal.compare(self.root, self.launch)
        self.assertEqual(run.call_count, 1)

    def test_rc0_without_same_lock_pdf_is_failed_no_automatic_retry(self):
        self.completed_arms()
        with patch.object(goal.subprocess, 'run', return_value=SimpleNamespace(returncode=0)) as run:
            for _ in range(2):
                with self.assertRaises(RuntimeError):
                    goal.compare(self.root, self.launch)
        self.assertEqual(run.call_count, 1)
        self.assertEqual(goal.optional(self.root / 'goal_compare_receipt.json')['state'], 'failed')

    def test_watch_success_does_not_publish_by_default(self):
        locks = self.completed_arms()
        self.completed_comparison(locks)
        with patch.object(goal, 'publish') as publish, patch.object(goal.time, 'sleep') as sleep:
            self.assertEqual(goal.watch(self.root, self.launch), 0)
        publish.assert_not_called()
        sleep.assert_not_called()
        receipt = goal.optional(self.root / 'goal_watch_receipt.json')
        self.assertTrue(receipt['pipeline_complete'])
        self.assertFalse(receipt['goal_complete'])

    def publication_context(self):
        repo = self.base / 'repo'
        root = repo / 'code/V4/runs/R95_adaptive_example_sampling'
        launch = repo / 'code/V4/runs/R95_launch'
        git_dir = repo / '.git'
        git_dir.mkdir(parents=True)
        root.mkdir(parents=True)
        stack = ExitStack()
        self.addCleanup(stack.close)
        for name, value in (('REPO', repo), ('ROOT', root), ('LAUNCH', launch)):
            stack.enter_context(patch.object(goal, name, value))
        source = repo / 'code/V4/r95_reporting.py'
        source.write_text('# synthetic frozen source\n')
        goal.save(launch / 'A/plan.json', dict(source_sha256={str(source): goal.file_hash(source)}))
        return repo, root, launch, git_dir

    def test_curated_publication_only_frozen_r95_sources_and_reports_not_arrays(self):
        repo, root, launch, _ = self.publication_context()
        (repo / 'code/V4/test_r95_added_after_launch.py').write_text('# not frozen')
        for name in ('PLAN.md', 'LAUNCH.md'):
            (root / name).write_text('# frozen campaign documentation')
        goal.save(root / 'protocol_lock.json', dict(experiment='R95', locked=True))
        note = repo / 'docs/research/unbiased_importance_sampling_screen.md'
        note.parent.mkdir(parents=True)
        note.write_text('# research boundary')
        for name in ('A/best.pt', 'A/output.pkl', 'A/weights.bin', 'A/array.npy',
                'A/cache/input.json', 'A/evaluation/predictions.npz', 'A/proposals/update.json'):
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b'{}')
        goal.save(root / 'A/full_model_statistics/summary.json', dict(top1=.7))
        proposal_fixture(root / 'A')
        curated = goal.publication_paths(root, launch, repo)
        self.assertIn('code/V4/r95_reporting.py', curated)
        self.assertIn('docs/research/unbiased_importance_sampling_screen.md', curated)
        for name in ('PLAN.md', 'LAUNCH.md', 'protocol_lock.json'):
            self.assertIn('code/V4/runs/R95_adaptive_example_sampling/' + name, curated)
        self.assertFalse(any('added_after' in name or 'best.pt' in name or name.endswith('.npz') for name in curated))
        self.assertEqual(sum('/proposals/' in name for name in curated), 8)
        self.assertFalse(any(name.endswith('/proposals/update.json') for name in curated))
        self.assertTrue(any('summary.json' in name for name in curated))
        goal.save(root / 'A/unsafe.json', dict(access_token='do-not-publish'))
        with self.assertRaises(RuntimeError):
            goal.publication_paths(root, launch, repo)

    def test_shared_publication_lock_encloses_one_commit_and_push(self):
        repo, root, launch, git_dir = self.publication_context()
        staged, commands = [], []

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
                return 'single-commit'
            self.fail(argv)

        def command(argv, **kwargs):
            assert_locked()
            commands.append(argv)
            if argv[1] == 'add':
                staged.extend(argv[4:])
            return SimpleNamespace(returncode=0)

        with patch.object(goal, 'status', return_value=dict(errors=[], comparison_complete=True)), \
                patch.object(goal, 'git_output', side_effect=git), patch.object(goal.subprocess, 'run', side_effect=command):
            self.assertTrue(goal.publish(root, launch)['complete'])
            goal.publish(root, launch)
        self.assertEqual([row[1] for row in commands], ['add', 'commit', 'push'])
        self.assertEqual(commands[-1][-1], 'HEAD:refs/heads/main')


if __name__ == '__main__':
    unittest.main()
