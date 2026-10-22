"""Synthetic CPU receipts; external commands/publication are always mocked."""

import contextlib
import copy
import fcntl
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from . import r94_goal as goal


def timing_fixture(lock, arm):
    measurement = dict(batch_seconds=[.1, .2, .3], mean_seconds_per_instance=.6 / (128 * 3))
    return dict(experiment='R94', arm=arm, complete=True,
        checkpoint_sha256=lock['checkpoint_sha256'], state_hash=lock['state_hash'],
        inference_rule=lock['inference_rule'], includes_native_encoder=True,
        repetitions=3, scope='synchronized first128 GPU-resident inputs', excludes='I/O, one-time preparation',
        device='cuda:0', per_problem=[dict(problem=p, instances=128,
            donor_plus_selector=copy.deepcopy(measurement), cache_hit_selector=copy.deepcopy(measurement))
            for p in goal.PROBLEMS])


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
            config = dict(experiment='R94', arm=arm, r94_schema_version=1,
                initialization='scratch_shared_r77_plus_node_fusion',
                frozen_donor=dict(variant={'A': 'random', 'B': 'pretrained'}[arm]),
                max_successful_updates=57600, checkpoint_rule='strict full VAL macro_vs_sbs_pct',
                inference_rule='same maximin')
            lock = dict(experiment='R94', arm=arm, selected_without_test=True,
                per_problem_assembly=False, selected_update=1440, epoch=1,
                checkpoint_sha256=arm * 64, state_hash=goal.json_hash(config),
                rule=config['checkpoint_rule'], inference_rule=config['inference_rule'])
            locks[arm] = lock

            def result(split, rows):
                return dict(per_problem={p: dict(n=rows) for p in goal.PROBLEMS},
                    identity=dict(experiment='R94', arm=arm, split=split,
                        checkpoint_sha256=lock['checkpoint_sha256'], state_hash=lock['state_hash'],
                        successful_updates=lock['selected_update'], model_state_sha256='model-' + arm))

            train, val, test = result('train', 81000), result('val', 4500), result('test', 4500)
            timing = timing_fixture(lock, arm)
            records = {
                'protocol.json': config,
                'status.json': dict(phase='test_complete', successful_updates=57600),
                'training_complete.json': dict(successful_updates=57600, state_hash=lock['state_hash'],
                    updates_by_problem=dict.fromkeys(goal.PROBLEMS, 3200)),
                'final_test_lock.json': lock,
                'best_eval.json': dict(successful_updates=1440, train=train),
                'final_metrics.json': dict(selected_update=1440, train=train, val=val,
                    checkpoint_sha256=lock['checkpoint_sha256'], state_hash=lock['state_hash']),
                'test_results.json': dict(complete=True, checkpoint_sha256=lock['checkpoint_sha256'],
                    state_hash=lock['state_hash'], result=test),
                'inference_timing.json': timing,
                'lib_stage_complete.json': dict(complete=True, exit_status=0,
                    checkpoint_sha256=lock['checkpoint_sha256'], state_hash=lock['state_hash'],
                    model_state_sha256='model-' + arm, datasets=['TSPLIB', 'CVRPLIB']),
                'postprocess_status.json': dict(complete=True, phase='complete', exit_status=0),
            }
            for name, value in records.items():
                goal.save(run / name, value)
            goal.save(run / 'full_model_statistics/protocol.json', dict(complete=True,
                checkpoint_lock=lock, inference_timing=dict(status='complete', pdf_panel=True,
                    sha256=goal.file_hash(run / 'inference_timing.json'))))
            for name in goal.REQUIRED_PDFS:
                (run / 'full_model_statistics' / name).write_bytes(b'%PDF-synthetic')
            for name in ('controller.json', 'worker.json'):
                goal.save(self.launch / arm / name, dict(state='complete', exit_status=0,
                    stages=[dict(name='synthetic', state='complete', exit_status=0)]))
        return locks

    def completed_comparison(self, locks):
        goal.save(self.root / 'comparison_artifacts/protocol.json',
            dict(experiment='R94', complete=True, exit_status=0, checkpoints=locks,
                inference_timing={arm: dict(sha256=goal.file_hash(self.root / arm / 'inference_timing.json'))
                    for arm in goal.ARMS}))
        (self.root / 'comparison_artifacts/COMPARISON.pdf').write_bytes(b'%PDF-synthetic')
        (self.root / 'RESULTS.pdf').write_bytes(b'%PDF-synthetic')

    def test_parser_and_roots(self):
        self.assertEqual(goal.ROOT.name, 'R94_pretrained_node_fusion')
        self.assertEqual(goal.LAUNCH.name, 'R94_launch')
        self.assertEqual(goal.BUDGETS, dict(A=57600, B=57600))
        args = goal.parser().parse_args(['watch', '--root', '/model', '--launch-root', '/launch',
            '--nss-root', '/nss', '--historical-root', '/r84', '--r89-root', '/r89'])
        self.assertFalse(args.publish_github)
        self.assertEqual(args.interval, 120)
        self.assertEqual(args.r89_root, Path('/r89'))

    def test_readiness_requires_full_budget_all_same_checkpoint_splits_timing_lib_pdf_and_rc0(self):
        locks = self.completed_arms()
        self.assertTrue(goal.status(self.root, self.launch)['all_arms_ready'])
        self.completed_comparison(locks)
        self.assertTrue(goal.status(self.root, self.launch)['comparison_complete'])
        path = self.root / 'B/training_complete.json'
        old = goal.optional(path)
        goal.save(path, dict(old, successful_updates=57599))
        self.assertFalse(goal.status(self.root, self.launch)['comparison_complete'])
        goal.save(path, old)
        for name, field in (('best_eval.json', 'train'), ('final_metrics.json', 'val'), ('test_results.json', 'result')):
            path = self.root / 'B' / name
            record = goal.optional(path)
            bad = copy.deepcopy(record)
            bad[field]['per_problem'].pop('ATSP')
            goal.save(path, bad)
            with self.subTest(split=name):
                self.assertFalse(goal.status(self.root, self.launch)['all_arms_ready'])
            goal.save(path, record)

    def test_incomplete_training_receipts_with_null_train_are_waiting_not_crashes(self):
        goal.save(self.root / 'A/best_eval.json', dict(successful_updates=1440, train=None))
        current = goal.status(self.root, self.launch)
        self.assertFalse(current['all_arms_ready'])
        self.assertFalse(current['errors'])
        self.assertEqual(current['phase'], 'waiting_for_arms')

    def test_different_selected_train_state_blocks_ready(self):
        self.completed_arms()
        path = self.root / 'A/best_eval.json'
        record = goal.optional(path)
        record['train']['identity']['model_state_sha256'] = 'different'
        goal.save(path, record)
        self.assertIn('full_TRAIN', goal.status(self.root, self.launch)['arms'][0]['missing'])

    def test_timing_missing_incomplete_wrong_sha_or_stale_pdf_blocks_ready(self):
        self.completed_arms()
        path = self.root / 'A/inference_timing.json'
        original = goal.optional(path)
        path.unlink()
        self.assertFalse(goal.status(self.root, self.launch)['all_arms_ready'])
        for key, value in [('complete', False), ('checkpoint_sha256', 'other'), ('state_hash', 'other'),
                           ('per_problem', original['per_problem'][:-1])]:
            goal.save(path, dict(original, **{key: value}))
            with self.subTest(key=key):
                self.assertFalse(goal.status(self.root, self.launch)['all_arms_ready'])
        goal.save(path, dict(original, completed_utc='changed after PDF'))
        current = goal.status(self.root, self.launch)
        self.assertNotIn('inference_timing', current['arms'][0]['missing'])
        self.assertIn('PDF', current['arms'][0]['missing'])

    def test_invalid_timing_samples_rejected_without_model_or_arrays(self):
        locks = self.completed_arms()
        original = timing_fixture(locks['A'], 'A')
        for value in (None, 'bad', 0., -1.):
            bad = copy.deepcopy(original)
            bad['per_problem'][0]['donor_plus_selector']['mean_seconds_per_instance'] = value
            self.assertFalse(goal.timing_ready(bad, locks['A']))
        self.assertTrue(goal.timing_ready(original, locks['A']))

    def test_protocol_and_comparison_checkpoint_drift_cannot_publish(self):
        locks = self.completed_arms()
        self.completed_comparison(locks)
        path = self.root / 'A/protocol.json'
        config = goal.optional(path)
        config['initialization'] = 'trained'
        goal.save(path, config)
        self.assertFalse(goal.status(self.root, self.launch)['comparison_complete'])
        self.completed_arms()
        record = goal.optional(self.root / 'comparison_artifacts/protocol.json')
        record['checkpoints']['B']['state_hash'] = 'changed'
        goal.save(self.root / 'comparison_artifacts/protocol.json', record)
        self.assertFalse(goal.status(self.root, self.launch)['comparison_complete'])

    def test_failures_are_preserved_no_cuda_retry_or_sleep(self):
        path = self.launch / 'A/worker.json'
        original = dict(state='failed', exit_status=1, error='synthetic GPU failure')
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

    def test_recursive_stage_failure_even_if_parent_complete(self):
        self.completed_arms()
        goal.save(self.launch / 'A/controller.json', dict(state='complete', exit_status=0,
            stages=[dict(name='pipeline', state='complete', exit_status=0,
                stages=[dict(name='timing', state='complete', exit_status=7)])]))
        self.assertTrue(goal.status(self.root, self.launch)['errors'])

    def test_compare_cannot_bypass_incomplete_arms(self):
        with patch.object(goal.subprocess, 'run') as run:
            with self.assertRaises(RuntimeError):
                goal.compare(self.root, self.launch)
        run.assert_not_called()

    def test_compare_cpu_once_and_forwards_all_reference_flags(self):
        locks = self.completed_arms()

        def command(argv, **kwargs):
            self.assertEqual(argv[4:6], ['code.V4.r94_reporting', 'compare'])
            for flag in ('--nss-root', '--historical-root', '--r89-root'):
                self.assertIn(flag, argv)
            self.assertEqual(kwargs['env']['CUDA_VISIBLE_DEVICES'], '')
            self.assertEqual(kwargs['env']['OMP_NUM_THREADS'], '2')
            self.completed_comparison(locks)
            return SimpleNamespace(returncode=0)

        with patch.object(goal.subprocess, 'run', side_effect=command) as run:
            goal.compare(self.root, self.launch, nss_root=self.base / 'nss',
                historical_root=self.base / 'r84', r89_root=self.base / 'r89')
            with self.assertRaises(RuntimeError):
                goal.compare(self.root, self.launch)
        self.assertEqual(run.call_count, 1)
        self.assertEqual(goal.optional(self.root / 'goal_compare_receipt.json')['state'], 'complete')

    def test_compare_rc0_without_outputs_is_failed_and_never_retried(self):
        self.completed_arms()
        with patch.object(goal.subprocess, 'run', return_value=SimpleNamespace(returncode=0)) as run:
            with self.assertRaises(RuntimeError):
                goal.compare(self.root, self.launch)
            with self.assertRaises(RuntimeError):
                goal.compare(self.root, self.launch)
        self.assertEqual(run.call_count, 1)
        self.assertEqual(goal.optional(self.root / 'goal_compare_receipt.json')['state'], 'failed')

    def test_stubbed_watcher_is_quiet_and_never_publishes_by_default(self):
        locks = self.completed_arms()
        self.completed_comparison(locks)
        with patch.object(goal, 'publish') as publish, patch.object(goal.time, 'sleep') as sleep, \
                patch('builtins.print') as output:
            self.assertEqual(goal.watch(self.root, self.launch), 0)
        publish.assert_not_called()
        sleep.assert_not_called()
        output.assert_not_called()
        receipt = goal.optional(self.root / 'goal_watch_receipt.json')
        self.assertTrue(receipt['pipeline_complete'])
        self.assertFalse(receipt['goal_complete'])
        self.assertFalse(receipt['inference_started'])

    def test_stubbed_watcher_failure_always_updates_terminal_receipt(self):
        self.completed_arms()
        with patch.object(goal, 'compare', side_effect=RuntimeError('synthetic compare failed')):
            self.assertEqual(goal.watch(self.root, self.launch), 1)
        self.assertEqual(goal.optional(self.root / 'goal_watch_receipt.json')['state'], 'stopped_on_error')

    def publication_context(self):
        repo = self.base / 'repo'
        root, launch = repo / 'code/V4/runs/R94_pretrained_node_fusion', repo / 'code/V4/runs/R94_launch'
        git_dir = repo / '.git'
        git_dir.mkdir(parents=True)
        root.mkdir(parents=True)
        stack = contextlib.ExitStack()
        self.addCleanup(stack.close)
        stack.enter_context(patch.object(goal, 'REPO', repo))
        stack.enter_context(patch.object(goal, 'ROOT', root))
        stack.enter_context(patch.object(goal, 'LAUNCH', launch))
        return repo, root, launch, git_dir

    def test_curated_paths_are_allowlisted_complete_version_assets_only(self):
        repo, root, launch, _ = self.publication_context()
        files = {
            'code/V4/r94_reporting.py': b'# synthetic source',
            'code/V4/r94_experiment.py': b'# synthetic source',
            'code/V4/test_r94_goal.py': b'# synthetic source',
            'code/V4/run_v4_r94_launch.sh': b'# synthetic source',
            'docs/research/pretrained_nco_representation_beyond_r50.md': b'research',
            'docs/research/nco_node_fusion_source_feasibility.md': b'research',
            'code/V4/r93_reporting.py': b'# preserve other version',
            'code/V4/runs/R94_pretrained_node_fusion/A/best.pt': b'weights',
            'code/V4/runs/R94_pretrained_node_fusion/A/cache/raw.json': b'{}',
            'code/V4/runs/R94_pretrained_node_fusion/A/wandb/state.json': b'{}',
            'code/V4/runs/R94_pretrained_node_fusion/A/evaluation/per_row.csv': b'raw',
            'code/V4/runs/R94_pretrained_node_fusion/A/lib_evaluation/predictions/TSPLIB.npz': b'raw',
            'code/V4/runs/R94_pretrained_node_fusion/A/full_model_statistics/evaluation_per_row.csv': b'raw',
            'code/V4/runs/R94_pretrained_node_fusion/A/full_model_statistics/metrics_by_problem.csv': b'aggregate',
            'code/V4/runs/R94_pretrained_node_fusion/A/inference_timing.json': b'{}',
            'code/V4/runs/R94_pretrained_node_fusion/A/full_model_statistics/figures/TSP_selection_distribution.pdf': b'%PDF',
            'code/V4/runs/R94_pretrained_node_fusion/goal_watch_receipt.json': b'{}',
        }
        for name, value in files.items():
            path = repo / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(value)
        curated = goal.publication_paths(root, launch, repo)
        expected = {name for name in files if name.startswith(('code/V4/r94_', 'code/V4/test_r94',
            'code/V4/run_v4_r94', 'docs/research/')) or name.endswith(('metrics_by_problem.csv',
                'inference_timing.json', 'TSP_selection_distribution.pdf'))}
        self.assertEqual(set(curated), expected)
        goal.save(root / 'A/protocol.json', dict(access_token='never stage'))
        with self.assertRaises(RuntimeError):
            goal.publication_paths(root, launch, repo)

    def test_symlinked_report_directory_and_source_are_excluded(self):
        repo, root, launch, _ = self.publication_context()
        source = repo / 'code/V4/r94_reporting.py'
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_text('# source', encoding='utf-8')
        outside = self.base / 'outside'
        outside.mkdir()
        (outside / 'MODEL_STATISTICS.pdf').write_bytes(b'%PDF-private')
        (root / 'A').mkdir()
        (root / 'A/full_model_statistics').symlink_to(outside, target_is_directory=True)
        (repo / 'code/V4/r94_private.py').symlink_to(outside / 'MODEL_STATISTICS.pdf')
        self.assertEqual(goal.publication_paths(root, launch, repo), ['code/V4/r94_reporting.py'])

    def test_global_mutex_covers_readiness_index_commit_and_push_once(self):
        repo, root, launch, git_dir = self.publication_context()
        path = repo / 'code/V4/r94_reporting.py'
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('# synthetic', encoding='utf-8')
        staged, commands = [], []

        def assert_locked():
            with (git_dir / 'codex-publication.lock').open('a') as probe:
                with self.assertRaises(BlockingIOError):
                    fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)

        def status(*args):
            assert_locked()
            return dict(errors=[], comparison_complete=True)

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
                return 'single-r94-commit'
            self.fail(argv)

        def command(argv, **kwargs):
            assert_locked()
            commands.append(argv)
            if argv[1] == 'add':
                staged.extend(argv[4:])
            return SimpleNamespace(returncode=0)

        with patch.object(goal, 'status', side_effect=status), patch.object(goal, 'git_output', side_effect=git), \
                patch.object(goal.subprocess, 'run', side_effect=command):
            receipt = goal.publish(root, launch)
            self.assertTrue(receipt['complete'])
            goal.publish(root, launch)
        self.assertEqual(sum(cmd[1] == 'commit' for cmd in commands), 1)
        self.assertEqual(sum(cmd[1] == 'push' for cmd in commands), 1)
        self.assertEqual(commands[-1], ['git', 'push', 'origin', 'HEAD:refs/heads/main'])

    def test_failed_push_is_preserved_and_never_automatically_retried(self):
        repo, root, launch, git_dir = self.publication_context()
        path = repo / 'code/V4/r94_reporting.py'
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('# synthetic', encoding='utf-8')
        staged, commands = [], []

        def git(argv, cwd):
            if argv == ['rev-parse', '--show-toplevel']:
                return str(repo)
            if argv == ['rev-parse', '--absolute-git-dir']:
                return str(git_dir)
            if argv == ['branch', '--show-current']:
                return 'main'
            if argv[:3] == ['diff', '--cached', '--name-only']:
                return '\0'.join(staged)
            if argv == ['rev-parse', 'HEAD']:
                return 'local-r94-commit'
            self.fail(argv)

        def command(argv, **kwargs):
            commands.append(argv)
            if argv[1] == 'add':
                staged.extend(argv[4:])
            if argv[1] == 'push':
                raise RuntimeError('synthetic push failure')
            return SimpleNamespace(returncode=0)

        with patch.object(goal, 'status', return_value=dict(errors=[], comparison_complete=True)), \
                patch.object(goal, 'git_output', side_effect=git), patch.object(goal.subprocess, 'run', side_effect=command):
            with self.assertRaises(RuntimeError):
                goal.publish(root, launch)
            with self.assertRaises(RuntimeError):
                goal.publish(root, launch)
        receipt = goal.optional(root / 'github_publication.json')
        self.assertEqual(receipt['commit'], 'local-r94-commit')
        self.assertEqual(receipt['state'], 'failed')
        self.assertFalse(receipt['automatic_retry'])
        self.assertEqual(sum(cmd[1] == 'commit' for cmd in commands), 1)

    def test_preexisting_staging_or_incomplete_version_is_untouched(self):
        repo, root, launch, git_dir = self.publication_context()

        def git(argv, cwd):
            if argv == ['rev-parse', '--show-toplevel']:
                return str(repo)
            if argv == ['rev-parse', '--absolute-git-dir']:
                return str(git_dir)
            if argv == ['branch', '--show-current']:
                return 'main'
            if argv[:3] == ['diff', '--cached', '--name-only']:
                return 'another-agent-file'
            self.fail(argv)

        with patch.object(goal, 'status', return_value=dict(errors=[], comparison_complete=True)), \
                patch.object(goal, 'git_output', side_effect=git), patch.object(goal.subprocess, 'run') as run:
            with self.assertRaisesRegex(RuntimeError, 'Pre-existing staged'):
                goal.publish(root, launch)
        run.assert_not_called()
        self.assertFalse((root / 'github_publication.json').exists())
        with patch.object(goal, 'status', return_value=dict(errors=[], comparison_complete=False)), \
                patch.object(goal, 'git_output', side_effect=git), patch.object(goal.subprocess, 'run') as run:
            with self.assertRaises(RuntimeError):
                goal.publish(root, launch)
        run.assert_not_called()

    def test_shared_r93_globals_are_unchanged(self):
        original = {name: value for name, value in vars(goal.r93).items() if not name.startswith('__')}
        goal.status(self.root, self.launch)
        for name, value in original.items():
            self.assertIs(getattr(goal.r93, name), value, name)


if __name__ == '__main__':
    unittest.main()
