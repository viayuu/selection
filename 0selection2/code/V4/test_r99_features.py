import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from .r76_experiment import write_json
from .r99_features import _prepare_task, CompetitionFeatureCache, authorize_test


class FeatureTests(unittest.TestCase):
    def test_input_only_ragged_roundtrip(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root/'inputs/train/TSP'
            source.mkdir(parents=True)
            n = np.array([3, 2], dtype=np.int64)
            node = np.array([[[0., 0.], [.1, .2], [.8, .7]],
                             [[0., 0.], [.5, .7], [0., 0.]]], dtype=np.float32)
            record = dict(problem='TSP', problem_id=0, split='train', kind='coord',
                rows=2, rows_written=2, complete=True, scenario_id='tiny', true_length_max=3,
                arrays=dict(n=dict(file='n.npy', shape=[2], dtype='int64'),
                            node=dict(file='node.npy', shape=[2, 3, 2], dtype='float32')))
            write_json(source/'metadata.json', record)
            np.save(source/'n.npy', n)
            np.save(source/'node.npy', node)
            original_load = np.load
            opened = []
            def input_load(path, *args, **kwargs):
                opened.append(Path(path).name)
                return original_load(path, *args, **kwargs)
            with patch('code.V4.r99_features.np.load', side_effect=input_load):
                result = _prepare_task(root/'inputs', root/'features', 'train', 'TSP',
                    'balanced', torch.device('cpu'), 2, None, None, None)
            self.assertTrue(result['complete'])
            self.assertEqual(set(opened), {'n.npy', 'node.npy'})
            self.assertEqual(result['shape'], [5, 32])
            base = SimpleNamespace(problem='TSP', split='train', size=2, lengths=n, metadata=record)
            def batch(rows, device='cpu'):
                maximum = int(n[rows].max())
                return dict(node_mask=torch.arange(maximum)[None] < torch.from_numpy(n[rows])[:, None])
            base.batch, base.close = batch, lambda: None
            wrapped = CompetitionFeatureCache(base, root/'features/train/TSP', 'balanced')
            try:
                got = wrapped.batch(np.array([1, 0]))
                self.assertEqual(got['relation_nodes'].shape, (2, 3, 32))
                self.assertTrue((got['relation_nodes'][0, 2] == 0).all())
            finally:
                wrapped.close()

    def test_test_lock_is_required(self):
        with self.assertRaises(PermissionError):
            authorize_test(None)
        with self.assertRaises(PermissionError):
            authorize_test(dict(experiment='R94', arm='B', selected_without_test=True,
                                per_problem_assembly=False, checkpoint_sha256='old'))

    def test_frozen_source_provenance_mismatch_rejected_before_arrays(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root/'inputs/train/TSP'
            source.mkdir(parents=True)
            record = dict(problem='TSP', split='train', kind='coord', rows=2,
                          source_files=[dict(path='original.pkl', sha256='locked')])
            write_json(source/'metadata.json', record)
            changed = dict(record, source_files=[dict(path='original.pkl', sha256='changed')])
            manifest = dict(splits=dict(train=dict(TSP=changed)))
            with self.assertRaisesRegex(ValueError, 'source_files'):
                _prepare_task(root/'inputs', root/'features', 'train', 'TSP',
                    'row', torch.device('cpu'), 2, manifest, None, None)
            self.assertFalse((root/'features/train/TSP/metadata.json').exists())


if __name__ == '__main__':
    unittest.main()
