"""Focused synthetic CPU recovery tests. Every Git command is mocked."""

from contextlib import redirect_stdout
import copy
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from . import r100_goal_recovery as admin
from . import r98_goal as shared
from .test_r98_goal import fixture


class RecoveryTests(unittest.TestCase):
    def make_module(self):
        module = admin.private_goal()
        module.REPO, module.LAUNCH = self.repo, self.launch
        return module

    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='r100-goal-admin-cpu-')
        self.addCleanup(temporary.cleanup)
        self.repo = Path(temporary.name) / 'repo'
        self.root, self.launch, locks = fixture(self.repo, 'R100')
        self.recovery = self.root / admin.RECOVERY_RELATIVE
        self.original = self.recovery / 'original'
        self.original.mkdir(parents=True)
        self.lock = locks['A']
        self.controller_path = self.launch / 'R100_A/controller.json'
        self.watcher_path = self.root / 'goal_watch_receipt.json'
        self.source = self.repo / 'code/V4/r100_reporting.py'
        self.source.write_text('fixed report fixture\n')
        (self.original / 'r100_reporting.py').write_text('original report fixture\n')
        (self.original / 'PDF.log').write_text('TypeError: expected str instance, list found\n')
        shared.save(self.original / 'postprocess_status.json', dict(complete=False, phase='failed'))
        watcher = dict(experiment='R100', state='stopped_on_error', exit_status=1,
                       error=admin.WATCH_ERROR, pid=494390, automatic_retry=False)
        shared.save(self.watcher_path, watcher)
        shared.save(self.original / 'goal_watch_receipt.json', watcher)
        self.old = shared.optional(self.controller_path)
        self.old.update(state='failed', exit_status=1, error='PDF exit=1; no retry')
        self.old['stages'][1].update(state='failed', exit_status=1, pid=659050,
                                   finished_utc='2026-10-22T09:49:21+00:00')
        shared.save(self.original / 'controller.json', self.old)
        shared.save(self.launch / 'R100_A/plan.json', dict(source_sha256={str(self.source):
                    shared.file_hash(self.original / 'r100_reporting.py')}))
        shared.save(self.launch / 'R100_A/ready.json', dict(ready=True))
        provenance_path = self.root / 'full_model_statistics/provenance.json'
        shared.save(provenance_path, dict(shared.optional(provenance_path), inference_performed=False))
        names = ('protocol.json', 'checkpoint.json', 'final_test_lock.json', 'test_input_lock.json',
                 'val_results.json', 'test_results.json', 'lib_results.json', 'run_complete.json',
                 'goal_watch_receipt.json')
        paths = [self.root / name for name in names]
        paths += [self.launch / 'R100_A' / name for name in ('worker.json', 'plan.json', 'ready.json')]
        for name in ('evaluation/val/selected_best/rank000.npz', 'evaluation/test/selected_best/rank000.npz',
                     'lib_evaluation/predictions/TSPLIB.npz', 'lib_evaluation/predictions/CVRPLIB.npz'):
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b'synthetic saved archive; never loaded for inference')
            paths.append(path)
        for path in self.original.iterdir():
            path.chmod(0o444)
        (self.root / 'full_model_statistics/metrics.csv').write_text('problem,n\nTSP,4500\n')
        self.before = dict(protected={str(p): admin.identity(p) for p in paths},
            snapshots={str(p): admin.identity(p) for p in self.original.iterdir()}, lock=self.lock,
            original_reporting_sha256=shared.file_hash(self.original / 'r100_reporting.py'),
            csv_rows=admin.csv_counts(self.root / 'full_model_statistics'))
        shared.save(self.recovery / 'before.json', self.before)
        shared.save(self.recovery / 'after.json', dict(protected=self.before['protected'], unchanged=True,
            snapshots_unchanged=True, csv_rows_unchanged=True, csv_rows=self.before['csv_rows']))
        argv = [admin.sys.executable, '-B', '-u', '-m', 'code.V4.r100_experiment', 'report',
                '--root', str(self.root), '--device', 'cpu']
        self.cpu = dict(name='PDF', device='cpu', state='complete', exit_status=0, pid=736810,
            started_utc='2026-10-22T10:03:52+00:00', finished_utc='2026-10-22T10:04:01+00:00',
            argv=argv, automatic_retry=False)
        shared.save(self.recovery / 'cpu_stage.json', self.cpu)
        shared.save(self.recovery / 'attempt.json', dict(argv=argv, automatic_retry=False,
            started_utc=self.cpu['started_utc'], reporting_sha256=shared.file_hash(self.source)))
        self.receipt = dict(experiment='R100', complete=True, phase='complete', exit_status=0,
            manual_explicit_recovery=True, original_attempt_succeeded=False, inference_performed=False,
            reselection_performed=False, automatic_retry=False, protected_metadata_and_sha256_unchanged=True,
            immutable_snapshots_unchanged=True, stopped_goalwatch_preserved=True, csv_rows_unchanged=True,
            cpu_stage=self.cpu, completed_utc='2026-10-22T10:04:02+00:00',
            checkpoint_sha256=self.lock['checkpoint_sha256'], state_hash=self.lock['state_hash'],
            original_reporting_sha256=self.before['original_reporting_sha256'],
            recovery_reporting_sha256=shared.file_hash(self.source), csv_rows=self.before['csv_rows'],
            PDFs={str(p): admin.identity(p) for p in [self.root / 'RESULTS.pdf',
                  *[self.root / 'full_model_statistics' / n for n in shared.FROZEN_PDFS]]})
        shared.save(self.recovery / 'recovery_complete.json', self.receipt)
        self.controller = copy.deepcopy(self.old)
        self.controller['original_error'] = self.controller.pop('error')
        self.controller.update(state='complete', exit_status=0,
            recovery_receipt=str(self.recovery / 'recovery_complete.json'))
        self.controller['stages'][1] = dict(self.cpu, original_attempt=self.old['stages'][1],
            original_attempt_snapshot=str(self.original / 'controller.json'))
        shared.save(self.controller_path, self.controller)
        self.shared_status = shared.status
        self.module = self.make_module()
        self.original_publish = self.module.publish
        self.goal = admin.RecoveryGoal(self.module)

    def test_exact_superseded_error_only_and_shared_core_unchanged(self):
        original = self.goal.original_status('R100', self.root, self.launch, persist=False)
        self.assertTrue(original['all_arms_ready'])
        self.assertFalse(original['complete'])
        watched = admin.identity(self.watcher_path)
        result = self.goal.status(persist=False)
        self.assertTrue(result['complete'])
        self.assertFalse(result['goal_complete'])
        self.assertEqual(result['errors'], [])
        self.assertEqual(result['superseded_errors'][0]['error'], original['errors'][0])
        self.assertEqual(result['superseded_errors'][0]['original_PDF_exit_status'], 1)
        self.assertEqual(admin.identity(self.watcher_path), watched)
        self.assertIs(shared.status, self.shared_status)
        self.assertIs(self.module.publish, self.original_publish)
        self.assertFalse((self.root / 'goal_progress.json').exists())

    def test_persisted_status_keeps_failure_visible_without_editing_watcher(self):
        watched = admin.identity(self.watcher_path)
        self.goal.status(persist=True)
        saved = shared.optional(self.root / 'goal_progress.json')
        self.assertEqual(saved['original_phase'], 'failed')
        self.assertEqual(saved['superseded_errors'][0]['original_PDF_exit_status'], 1)
        self.assertEqual(admin.identity(self.watcher_path), watched)

    def test_unknown_duplicate_missing_and_same_message_other_path_errors_reject(self):
        original = self.goal.original_status('R100', self.root, self.launch, persist=False)
        error = original['errors'][0]
        for errors in ([], [error, error], [dict(error, path=str(self.root / 'other.json'))],
                       [dict(error, error='new failure')], [error, dict(error, path='elsewhere')]):
            with self.subTest(errors=errors), patch.object(self.goal, 'original_status',
                    return_value=dict(original, errors=errors)):
                with self.assertRaisesRegex(RuntimeError, 'another, missing or duplicate'):
                    self.goal.status(persist=False)

    def test_other_real_receipt_failure_is_not_suppressed(self):
        shared.save(self.root / 'goal_compare_receipt.json',
                    dict(state='failed', exit_status=1, error=admin.WATCH_ERROR))
        with self.assertRaises(RuntimeError):
            self.goal.status(persist=False)

    def test_invalid_recovery_flags_and_bank_reject(self):
        bad_fields = dict(complete=False, exit_status=1, inference_performed=True, reselection_performed=True,
            manual_explicit_recovery=False, original_attempt_succeeded=True, automatic_retry=True,
            stopped_goalwatch_preserved=False, immutable_snapshots_unchanged=False,
            protected_metadata_and_sha256_unchanged=False, checkpoint_sha256='another-bank')
        for key, value in bad_fields.items():
            with self.subTest(field=key):
                shared.save(self.recovery / 'recovery_complete.json', dict(self.receipt, **{key: value}))
                with self.assertRaises(RuntimeError):
                    self.goal.status(persist=False)
        shared.save(self.recovery / 'recovery_complete.json', dict(self.receipt, exit_status=False))
        with self.assertRaises(RuntimeError):
            self.goal.status(persist=False)

    def test_changed_canonical_watcher_rejects(self):
        shared.save(self.watcher_path, dict(shared.optional(self.watcher_path), error='new error'))
        with self.assertRaisesRegex(RuntimeError, 'identity changed'):
            self.goal.status(persist=False)

    def test_changed_archive_metadata_rejects_even_same_bytes(self):
        path = self.original / 'goal_watch_receipt.json'
        stat = path.stat()
        os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1000))
        with self.assertRaisesRegex(RuntimeError, 'identity changed'):
            self.goal.status(persist=False)

    def test_writable_or_changed_original_archive_rejects(self):
        (self.original / 'controller.json').chmod(0o644)
        with self.assertRaisesRegex(RuntimeError, 'identity changed'):
            self.goal.status(persist=False)

    def test_changed_saved_prediction_rejects(self):
        (self.root / 'evaluation/test/selected_best/rank000.npz').write_bytes(b'changed')
        with self.assertRaisesRegex(RuntimeError, 'identity changed'):
            self.goal.status(persist=False)

    def test_bad_gpu_current_cpu_or_erased_first_attempt_rejects(self):
        for name in ('GPU', 'CPU', 'history', 'original_error'):
            with self.subTest(name=name):
                value = copy.deepcopy(self.controller)
                if name == 'GPU':
                    value['stages'][0]['exit_status'] = 1
                elif name == 'CPU':
                    value['stages'][1]['pid'] += 1
                elif name == 'history':
                    value['stages'][1].pop('original_attempt')
                else:
                    value.pop('original_error')
                shared.save(self.controller_path, value)
                with self.assertRaises(RuntimeError):
                    self.goal.status(persist=False)

    def test_missing_core_evidence_rejects(self):
        for field in ('protected', 'snapshots'):
            with self.subTest(field=field):
                shared.save(self.recovery / 'before.json', dict(self.before, **{field: {}}))
                with self.assertRaises(RuntimeError):
                    self.goal.status(persist=False)

    def test_incomplete_after_proof_rejects(self):
        path = self.recovery / 'after.json'
        shared.save(path, dict(shared.optional(path), unchanged=False))
        with self.assertRaises(RuntimeError):
            self.goal.status(persist=False)

    def test_changed_pdf_rejects(self):
        (self.root / 'RESULTS.pdf').write_bytes(b'%PDF-changed fixture')
        with self.assertRaisesRegex(RuntimeError, 'PDF identity changed'):
            self.goal.status(persist=False)

    def test_changed_csv_count_rejects(self):
        (self.root / 'full_model_statistics/metrics.csv').write_text('problem,n\nTSP,4500\nCVRP,4500\n')
        with self.assertRaisesRegex(RuntimeError, 'dataframe counts changed'):
            self.goal.status(persist=False)

    def test_changed_render_source_rejects(self):
        self.source.write_text('changed report fixture\n')
        with self.assertRaisesRegex(RuntimeError, 'report source hashes differ'):
            self.goal.status(persist=False)

    def test_incomplete_original_full_pipeline_checks_reject(self):
        original = self.goal.original_status('R100', self.root, self.launch, persist=False)
        for field in ('all_arms_ready', 'comparison_complete'):
            with self.subTest(field=field), patch.object(self.goal, 'original_status',
                    return_value=dict(original, **{field: False})):
                with self.assertRaisesRegex(RuntimeError, 'full-pipeline checks'):
                    self.goal.status(persist=False)

    def test_failed_publication_receipt_prevents_any_new_git_attempt(self):
        shared.save(self.root / 'github_publication.json', dict(state='failed', exit_status=1, error='push failed'))
        module = self.make_module()
        with patch.object(admin, 'private_goal', return_value=module), \
                patch.object(module, 'git_output', side_effect=AssertionError('No retry')), \
                patch.object(module.subprocess, 'run', side_effect=AssertionError('No subprocess')):
            with self.assertRaises(RuntimeError):
                admin.main(['--version', 'R100', '--publish-github'])

    def test_only_r100_canonical_roots_are_authorized(self):
        for kwargs in (dict(version='R99'), dict(root=self.repo), dict(launch_root=self.repo)):
            with self.subTest(kwargs=kwargs), self.assertRaises(RuntimeError):
                self.goal.status(persist=False, **kwargs)

    def test_exact_status_cli_has_no_git_launch_or_canonical_status_write(self):
        module = self.make_module()
        with patch.object(admin, 'private_goal', return_value=module), redirect_stdout(io.StringIO()) as out, \
                patch.object(module, 'git_output', side_effect=AssertionError('Git is forbidden')), \
                patch.object(module.subprocess, 'run', side_effect=AssertionError('No subprocess')):
            self.assertEqual(admin.main(['--version', 'R100']), 0)
        self.assertTrue(json.loads(out.getvalue())['complete'])
        self.assertFalse((self.root / 'goal_progress.json').exists())
        self.assertFalse((self.root / 'github_publication.json').exists())
        evidence = shared.optional(self.recovery / 'goal_recovery_status.json')
        self.assertEqual(evidence['mode'], 'status_only')
        self.assertFalse(evidence['publication_complete'])

    def test_exact_publication_cli_uses_original_publisher_curated_paths_and_mutex(self):
        module = self.make_module()
        git_dir = self.repo / '.git'
        git_dir.mkdir()
        staged, commands, head = set(), [], ['original-head']

        def git_output(argv, cwd):
            if argv == ['rev-parse', '--show-toplevel']:
                return str(self.repo)
            if argv == ['rev-parse', '--absolute-git-dir']:
                return str(git_dir)
            if argv == ['branch', '--show-current']:
                return 'main'
            if argv == ['rev-parse', 'HEAD']:
                return head[0]
            if argv == ['diff', '--cached', '--name-only', '-z']:
                return '\0'.join(sorted(staged))
            raise AssertionError(argv)

        def run(argv, **kwargs):
            commands.append(argv)
            self.assertEqual(argv[0], 'git')
            if argv[1] == 'add':
                self.assertEqual(argv[2:4], ['-f', '--'])
                staged.update(argv[4:])
            elif argv[1] == 'commit':
                head[0] = 'mock-commit'
            else:
                self.assertEqual(argv, ['git', 'push', 'origin', 'HEAD:refs/heads/main'])

        publisher = module.publish
        with patch.object(admin, 'private_goal', return_value=module), redirect_stdout(io.StringIO()), \
                patch.object(module, 'git_output', side_effect=git_output), \
                patch.object(module.subprocess, 'run', side_effect=run), \
                patch.object(module.fcntl, 'flock', wraps=module.fcntl.flock) as mutex:
            self.assertEqual(admin.main(['--version', 'R100', '--publish-github']), 0)
        self.assertIs(module.publish, publisher)
        mutex.assert_called_once()
        self.assertTrue((git_dir / 'codex-publication.lock').is_file())
        published = shared.optional(self.root / 'github_publication.json')
        self.assertEqual(set(published['files']), staged)
        self.assertTrue(published['complete'] and published['curated_only'])
        self.assertEqual(published['global_mutex'], str(git_dir / 'codex-publication.lock'))
        self.assertEqual(sum(argv[1] == 'commit' for argv in commands), 1)
        self.assertEqual(sum(argv[1] == 'push' for argv in commands), 1)
        self.assertTrue(shared.optional(self.recovery / 'goal_recovery_publication_result.json')['publication']['complete'])
        self.assertEqual(shared.optional(self.root / 'goal_watch_receipt.json')['state'], 'stopped_on_error')


if __name__ == '__main__':
    unittest.main()
