"""Tiny CPU fixtures only; never execute TEST, CUDA, or Git."""

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from . import r122_test_recovery as recovery


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        root = Path(self.temporary.name)
        self.local, self.source = root / 'local/test/TSP', root / 'source/test/TSP'
        lock = root / 'historic/final_test_lock.json'
        lock.parent.mkdir()
        lock.write_text('{}')
        arrays = dict(costs=np.arange(15, dtype=np.float64).reshape(5, 3) + 1,
                      winner=np.zeros(5, dtype=np.int64),
                      nodes=np.arange(20, dtype=np.float32).reshape(5, 4))
        self.expected = dict(split='test', problem='TSP', rows=5, complete=True,
            data_root='/frozen/data', source_file_sha256='native-source',
            pool=['a', 'b', 'c'], pool_ids=[1, 2, 3], observed_mask=[True] * 3,
            test_lock_root=str(lock.parent), created_utc='original', arrays={})
        for name, value in arrays.items():
            self.expected['arrays'][name] = dict(file=name + '.npy', shape=list(value.shape), dtype=str(value.dtype))
        for directory in (self.local, self.source):
            directory.mkdir(parents=True)
            for name, value in arrays.items():
                np.save(directory / (name + '.npy'), value, allow_pickle=False)
            (directory / 'metadata.json').write_text(json.dumps(self.expected))

    def test_only_six_cosmetic_fields_and_original_non_test_delegate(self):
        value = dict(self.expected, directory='node-local', cache_root='/tmp/copied',
                     created_utc='copied', completed_utc='later', seconds=4.)
        (self.local / 'metadata.json').write_text(json.dumps(value))
        proof = recovery.test_cache_copy(self.local, self.expected, label_reference=self.source)
        self.assertEqual(set(proof['label_sha256']), {'costs', 'winner'})
        self.assertEqual(proof['input_sample_rows'], [0, 1, 2, 3, 4])
        original = recovery.runner.validate_cache_copy
        proofs = []
        with recovery.resource_adapter(proofs):
            recovery.runner.validate_cache_copy(self.local, self.expected, label_reference=self.source)
        self.assertIs(recovery.runner.validate_cache_copy, original)
        self.assertEqual(len(proofs), 1)
        with patch.object(recovery, 'ORIGINAL_VALIDATE') as strict:
            recovery.test_cache_copy(self.local, dict(self.expected, split='train'), label_reference=self.source)
            strict.assert_called_once()

    def test_semantic_cost_winner_input_and_typed_changes_fail(self):
        path = self.local / 'metadata.json'
        for key, value in (('data_root', '/other'), ('source_file_sha256', 'other'),
                           ('pool', ['x', 'b', 'c']), ('observed_mask', [False] * 3)):
            with self.subTest(field=key):
                path.write_text(json.dumps(dict(self.expected, **{key: value})))
                with self.assertRaisesRegex(ValueError, 'semantic'):
                    recovery.test_cache_copy(self.local, self.expected, label_reference=self.source)
        path.write_text(json.dumps(self.expected))
        for name in ('costs', 'winner', 'nodes'):
            with self.subTest(array=name):
                file = self.local / (name + '.npy')
                original = file.read_bytes()
                value = np.load(file, allow_pickle=False)
                value.flat[0] += 1
                np.save(file, value, allow_pickle=False)
                with self.assertRaisesRegex(ValueError, 'SHA256|fixed-row'):
                    recovery.test_cache_copy(self.local, self.expected, label_reference=self.source)
                file.write_bytes(original)
        file = self.local / 'costs.npy'
        value = np.load(file).astype(np.float32)
        for directory in (self.local, self.source):
            np.save(directory / 'costs.npy', value, allow_pickle=False)
        with self.assertRaisesRegex(ValueError, 'typed array'):
            recovery.test_cache_copy(self.local, copy.deepcopy(self.expected), label_reference=self.source)


if __name__ == '__main__':
    unittest.main()
