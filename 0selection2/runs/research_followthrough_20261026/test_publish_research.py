"""Bounded local Git fixtures; no production commit/push, model, or GPU calls."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import publish_research as publication


class PublicationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        publication.git(self.root, 'init', '-b', 'main')
        (self.root / 'base.txt').write_text('base\n')
        publication.git(self.root, 'add', '--', 'base.txt')
        self.commit('base')
        self.parent = publication.output(self.root, 'rev-parse', 'HEAD')

    def tearDown(self):
        self.temp.cleanup()

    def commit(self, message):
        # Fixture-local identity overrides; no repository/global config writes.
        publication.git(self.root, '-c', 'user.name=Fixture', '-c', 'user.email=fixture@example.invalid',
                        'commit', '-m', message)
        return publication.output(self.root, 'rev-parse', 'HEAD')

    def test_text_normalization_preserves_csv_cells_and_binary(self):
        text = self.root / 'source.csv'
        text.write_bytes(b'name,value\r\nx,2\r\n')
        self.assertEqual(publication.expected_bytes(text), b'name,value\nx,2\n')
        binary = self.root / 'REPORT.pdf'
        binary.write_bytes(b'%PDF-1.4\r\n\x00')
        self.assertEqual(publication.expected_bytes(binary), binary.read_bytes())

    def test_unrelated_staged_changes_are_preserved(self):
        path = self.root / 'other.txt'
        path.write_text('other writer\n')
        publication.git(self.root, 'add', '--', path.name)
        before = publication.git(self.root, 'diff', '--cached', '--binary').stdout
        with self.assertRaisesRegex(ValueError, 'Unrelated staged changes'):
            publication.assert_clean_index(self.root)
        self.assertEqual(before, publication.git(self.root, 'diff', '--cached', '--binary').stdout)

    def fixture_attempt(self, extra=False):
        path = self.root / 'REPORT.md'
        path.write_text('immutable scientific report\n')
        publication.git(self.root, 'add', '--', path.name)
        if extra:
            (self.root / 'other.txt').write_text('unrelated\n')
            publication.git(self.root, 'add', '--', 'other.txt')
        commit = self.commit('research')
        attempt = dict(parent=self.parent, expected_hashes={str(path): publication.sha(path.read_bytes())},
                       changed_paths=['REPORT.md'])
        return attempt, commit

    def test_staged_blob_binding(self):
        path = self.root / 'REPORT.md'
        path.write_text('test\n')
        publication.git(self.root, 'add', '--', path.name)
        self.assertEqual(publication.committed_hashes(self.root, '', [path]),
                         {str(path): publication.sha(path.read_bytes())})

    def test_uncertain_push_reconciles_without_duplicate_mutations(self):
        attempt, commit = self.fixture_attempt()
        before = publication.output(self.root, 'rev-list', '--count', 'HEAD')
        with patch.object(publication, 'remote_head', return_value=commit):
            self.assertEqual(publication.reconcile_attempt(self.root, attempt, 'origin', 'refs/heads/main'), commit)
        self.assertEqual(before, publication.output(self.root, 'rev-list', '--count', 'HEAD'))
        publication.assert_clean_index(self.root)

    def test_missing_remote_confirmation_never_repushes(self):
        attempt, commit = self.fixture_attempt()
        with patch.object(publication, 'remote_head', return_value=self.parent):
            with self.assertRaisesRegex(ValueError, 'remote confirmation is absent'):
                publication.reconcile_attempt(self.root, attempt, 'origin', 'refs/heads/main')
        self.assertEqual(publication.output(self.root, 'rev-parse', 'HEAD'), commit)

    def test_reconciliation_rejects_unrelated_commit_content(self):
        attempt, commit = self.fixture_attempt(extra=True)
        attempt['commit'] = commit
        with patch.object(publication, 'remote_head', return_value=commit):
            with self.assertRaisesRegex(ValueError, 'no duplicate commit/push'):
                publication.reconcile_attempt(self.root, attempt, 'origin', 'refs/heads/main')

    def test_numeric_contradiction_blocks(self):
        with self.assertRaisesRegex(ValueError, 'numeric contradiction'):
            publication.close_enough('71.3', '71.4')


if __name__ == '__main__':
    unittest.main()
