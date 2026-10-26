"""Small stdlib-only fixtures; all mutating Git commands use temporary repos."""

import copy
import json
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from . import r109_publication_recovery as recovery


class SavedEvidenceTest(unittest.TestCase):
    def evidence(self):
        root = Path('/fixture/R109')
        saved = dict(experiment='R109', all_arms_ready=True, comparison_complete=True,
                     matched_controls=True, common_original_initialization=True,
                     complete=True, errors=[], phase='complete',
                     arms=[dict(arm=arm, ready=True, checks=dict(training=True, TEST=True))
                           for arm in recovery.ARMS])
        current = copy.deepcopy(saved)
        current.update(complete=False, phase='failed', errors=[dict(
            path=str(root / 'goal_watch_receipt.json'), error=recovery.WATCH_ERROR)])
        original = dict(experiment='R109', state='stopped_on_error', exit_status=1,
                        error=recovery.WATCH_ERROR, completed_utc=recovery.WATCH_ENDED,
                        main_ready_evidence=recovery.EVIDENCE, automatic_retry=False)
        return current, saved, original, root

    def test_only_exact_administrative_failure_is_accepted_without_mutation(self):
        evidence = self.evidence()
        original = copy.deepcopy(evidence)
        recovery.saved_gate_ready(*evidence)
        self.assertEqual(evidence, original)
        for change in ('extra_error', 'bad_arm', 'wrong_failure', 'forged_success'):
            current, saved, receipt, root = self.evidence()
            if change == 'extra_error':
                current['errors'].append(dict(path='TEST', error='failed'))
            elif change == 'bad_arm':
                current['arms'][1]['checks']['TEST'] = False
            elif change == 'wrong_failure':
                receipt['error'] = 'commit result unknown'
            else:
                current['complete'] = True
            with self.subTest(change=change), self.assertRaises(RuntimeError):
                recovery.saved_gate_ready(current, saved, receipt, root)


class TemporaryGitTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='r109-publication-fixture-')
        self.addCleanup(self.temporary.cleanup)
        top = Path(self.temporary.name)
        self.repo, self.remote = top / 'checkout', top / 'remote.git'
        self.repo.mkdir()
        self.command(['git', 'init', '--bare', str(self.remote)], cwd=top)
        self.command(['git', 'init', '-b', 'main'], cwd=self.repo)
        for key, value in (('user.name', 'R109 CPU Fixture'),
                           ('user.email', 'r109-fixture@example.invalid'),
                           ('core.autocrlf', 'input'), ('commit.gpgsign', 'false')):
            self.command(['git', 'config', key, value], cwd=self.repo)
        (self.repo / 'BASE.md').write_text('isolated fixture\n')
        self.command(['git', 'add', 'BASE.md'], cwd=self.repo)
        self.command(['git', 'commit', '-m', 'fixture base'], cwd=self.repo)
        self.command(['git', 'remote', 'add', 'origin', str(self.remote)], cwd=self.repo)
        self.command(['git', 'push', 'origin', 'main'], cwd=self.repo)
        base = self.command(['git', 'rev-parse', 'HEAD'], cwd=self.repo).stdout.decode().strip()
        self.fixture = recovery.Recovery(self.repo / '0selection2', str(self.remote), base)
        self.fixture.here.mkdir(parents=True)
        self.asset = self.fixture.root / 'RESULTS.md'
        self.asset.write_text('completed saved R109 fixture\n')
        self.fixture.save(self.fixture.here / 'fixture_receipt.json',
                          dict(complete=True, exit_status=0))
        self.fixture.save(self.fixture.here / 'source_closure.json', dict(sources=[]))
        self.closure = dict(sources=[])

    def command(self, argv, *, cwd):
        return subprocess.run(argv, cwd=cwd, capture_output=True, check=True,
                              env=recovery.Recovery().environment())

    def prepare_manifest(self):
        row = self.fixture.asset(self.asset, lambda record: True)
        manifest = dict(parent_commit=self.fixture.base, remote_before=self.fixture.base,
                        records=[row], expected_changed_paths=sorted([
                            row['path'], self.fixture.relative(self.fixture.manifest)]),
                        explicit_stage_paths=sorted([
                            row['path'], self.fixture.relative(self.fixture.manifest)]))
        self.fixture.save(self.fixture.manifest, manifest, exclusive=True)
        return recovery.sha(self.fixture.manifest)

    def scientific_mocks(self):
        replacements = dict(completion_gate={}, source_closure=(self.closure, set()),
                            goal=SimpleNamespace(adapter=SimpleNamespace(_safe_json=lambda record: True)))
        for name, value in replacements.items():
            mocked = patch.object(self.fixture, name, return_value=value)
            mocked.start()
            self.addCleanup(mocked.stop)

    def test_csv_crlf_hash_matches_existing_git_normalization(self):
        path = self.fixture.root / 'metrics.csv'
        path.write_bytes(b'split,value\r\nTEST,71.38\r\n')
        raw_sha = recovery.sha(path)
        expected = self.fixture.blob_oid(path)
        self.fixture.git('add', '--', self.fixture.relative(path))
        staged = self.fixture.git('rev-parse', ':' + self.fixture.relative(path))
        self.assertEqual(staged, expected)
        self.assertEqual(path.read_bytes(), b'split,value\r\nTEST,71.38\r\n')
        self.assertEqual(recovery.sha(path), raw_sha)

    def test_own_frozen_run_document_is_not_an_unrelated_shared_dependency(self):
        source = self.fixture.root / 'PLAN.md'
        source.write_text('original frozen R109 plan\n')
        for arm in recovery.ARMS:
            output = self.fixture.dispatch / ('R109_' + arm)
            output.mkdir(parents=True)
            spec = dict(run_dir=str(self.fixture.root / arm))
            spec_path = self.fixture.dispatch / 'specs' / ('R109_' + arm + '.json')
            spec_path.parent.mkdir(exist_ok=True)
            spec_path.write_text(json.dumps(spec))
            plan_path = output / 'plan.json'
            plan_path.write_text(json.dumps(dict(spec=spec, evidence=recovery.EVIDENCE,
                source_sha256={str(source): recovery.sha(source)})))
            (output / 'ready.json').write_text(json.dumps(dict(evidence=recovery.EVIDENCE,
                plan_sha256=recovery.sha(plan_path), source_freeze_ready=True, main_thread_ready=True)))
        closure, own = self.fixture.source_closure(self.fixture.parent_tree())
        self.assertEqual(own, {source})
        self.assertTrue(closure['sources'][0]['selected_for_publication'])

    def test_dirty_index_and_remote_mismatch_are_rejected_without_cleanup(self):
        self.fixture.git('add', '--', self.fixture.relative(self.asset))
        before = self.fixture.git('diff', '--cached', '--name-only', '-z')
        with self.fixture.locked(), self.assertRaisesRegex(RuntimeError, 'staged index'):
            self.fixture.preflight()
        self.assertEqual(self.fixture.git('diff', '--cached', '--name-only', '-z'), before)

    def test_remote_mismatch_does_not_stage(self):
        with self.fixture.locked(), patch.object(self.fixture, 'remote_head', return_value='0' * 40):
            with self.assertRaisesRegex(RuntimeError, 'HEAD/remote/base mismatch'):
                self.fixture.preflight()
        self.assertEqual(self.fixture.git('diff', '--cached', '--name-only'), '')

    def test_success_is_one_commit_one_normal_push_and_reentry_is_refused(self):
        digest = self.prepare_manifest()
        self.scientific_mocks()
        original = self.asset.read_bytes()
        result = self.fixture.publish(digest, 'isolated fixture GO, not production approval')
        self.assertTrue(result['complete'])
        self.assertEqual(result['commit_attempts'], 1)
        self.assertEqual(result['push_attempts'], 1)
        self.assertEqual(result['commit'], self.fixture.remote_head())
        self.assertEqual(self.fixture.git('rev-list', '--count', self.fixture.base + '..HEAD'), '1')
        self.assertEqual(self.asset.read_bytes(), original)
        self.assertEqual(self.fixture.git('diff', '--cached', '--name-only'), '')
        with self.assertRaisesRegex(RuntimeError, 'already attempted'):
            self.fixture.publish(digest, 'do not repeat')
        self.assertEqual(read_receipt(self.fixture)['commit'], result['commit'])

    def test_changed_asset_and_wrong_approval_never_stage(self):
        digest = self.prepare_manifest()
        self.scientific_mocks()
        with self.assertRaisesRegex(RuntimeError, 'GO digest'):
            self.fixture.publish('0' * 64, 'fixture GO')
        self.asset.write_text('changed after approval\n')
        with self.assertRaisesRegex(RuntimeError, 'Manifest asset changed'):
            self.fixture.publish(digest, 'fixture GO')
        self.assertFalse(self.fixture.receipt.exists())
        self.assertEqual(self.fixture.git('diff', '--cached', '--name-only'), '')

    def test_uncertain_commit_is_preserved_and_not_retried(self):
        digest = self.prepare_manifest()
        self.scientific_mocks()
        original_git = self.fixture.git
        attempts = []

        def interrupted(*args, **kwargs):
            if args and args[0] == 'commit':
                attempts.append(args)
                raise subprocess.TimeoutExpired(['git', *args], 1)
            return original_git(*args, **kwargs)

        with patch.object(self.fixture, 'git', side_effect=interrupted):
            with self.assertRaises(subprocess.TimeoutExpired):
                self.fixture.publish(digest, 'fixture GO')
            with self.assertRaisesRegex(RuntimeError, 'already attempted'):
                self.fixture.publish(digest, 'no retry')
        self.assertEqual(len(attempts), 1)
        receipt = read_receipt(self.fixture)
        self.assertEqual(receipt['state'], 'stopped_on_error')
        self.assertEqual(receipt['commit_attempts'], 1)
        self.assertEqual(receipt['push_attempts'], 0)
        self.assertTrue(self.fixture.git('diff', '--cached', '--name-only'))

    def test_uncertain_push_is_not_retried_or_recommitted(self):
        digest = self.prepare_manifest()
        self.scientific_mocks()
        original_run = self.fixture.run
        pushes = []

        def interrupted(argv, **kwargs):
            if 'push' in argv:
                pushes.append(argv)
                raise subprocess.TimeoutExpired(argv, 1)
            return original_run(argv, **kwargs)

        with patch.object(self.fixture, 'run', side_effect=interrupted):
            with self.assertRaises(subprocess.TimeoutExpired):
                self.fixture.publish(digest, 'fixture GO')
            with self.assertRaisesRegex(RuntimeError, 'already attempted'):
                self.fixture.publish(digest, 'no retry')
        self.assertEqual(len(pushes), 1)
        receipt = read_receipt(self.fixture)
        self.assertEqual(receipt['push_attempts'], 1)
        self.assertEqual(self.fixture.git('rev-list', '--count', self.fixture.base + '..HEAD'), '1')
        self.assertEqual(self.fixture.remote_head(), self.fixture.base)


def read_receipt(fixture):
    return json.loads(fixture.receipt.read_text())


if __name__ == '__main__':
    unittest.main()
