"""Tiny CPU cache and row-alignment checks, not solver certification."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from . import r117_features as features


class FakeNative(nn.Module):
    def __init__(self, arm):
        super().__init__()
        self.arm, self.calls = arm, 0

    def provenance(self):
        return dict(fake=True, arm=self.arm)

    def forward_extra(self, batch, base_nodes=None):
        self.calls += 1
        value = batch['node'][..., :1]
        return value.expand(-1, -1, 128) + (1. if self.arm == 'A' else 2.)

    def forward(self, batch):
        return dict(base_nodes=batch['node'][..., :1].expand(-1, -1, 128),
            native_nodes=self.forward_extra(batch), node_mask=batch['node_mask'])


def source(root):
    target = root / 'raw/train/OVRPTW'
    target.mkdir(parents=True)
    node = np.arange(3 * 5 * 8, dtype=np.float32).reshape(3, 5, 8) / 100
    n = np.array([3, 5, 4], dtype=np.int64)
    record = dict(problem='OVRPTW', problem_id=7, split='train', rows=3, rows_written=3,
        true_length_min=3, true_length_max=5, complete=True, kind='coord', d_in=8,
        data_root=str(root / 'data'), scenario_id='fake_only', pool=['fake'], pool_ids=[1],
        arrays=dict(n=dict(file='n.npy', shape=[3], dtype='int64'),
            node=dict(file='node.npy', shape=[3, 5, 8], dtype='float32')))
    np.save(target / 'n.npy', n)
    np.save(target / 'node.npy', node)
    (target / 'metadata.json').write_text(json.dumps(record))
    return dict(splits=dict(train=dict(OVRPTW=record))), node, n


class FeatureTests(unittest.TestCase):
    def test_shared_base_and_original_row_identity(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(features, 'PROBLEMS', ('OVRPTW',)):
            root = Path(tmp)
            manifest, node, n = source(root)
            donors = {arm: FakeNative(arm) for arm in ('A', 'B')}
            for arm, donor in donors.items():
                result = features.prepare_features(root / 'raw', root / arm, 'train', donor,
                    torch.device('cpu'), common_root=root / 'common', manifest=manifest,
                    data_root=root / 'data', batch_size=2)
                self.assertTrue(result['complete'])
                array = np.load(root / arm / 'train/OVRPTW/native_nodes.npy')
                for row, count in enumerate(n):
                    np.testing.assert_allclose(array[row, :count, 0],
                        (node[row, :count, 0] + (1 if arm == 'A' else 2)).astype(np.float16))
                    self.assertFalse(array[row, count:].any())
                calls = donor.calls
                features.prepare_features(root / 'raw', root / arm, 'train', donor,
                    torch.device('cpu'), common_root=root / 'common', manifest=manifest,
                    data_root=root / 'data', batch_size=2)
                self.assertEqual(donor.calls, calls)
            base = np.load(root / 'common/train/OVRPTW/nco_nodes.npy')
            np.testing.assert_allclose(base[1, :, 0], node[1, :, 0].astype(np.float16))

    def test_test_requires_unique_val_lock_before_any_input_open(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaises(PermissionError):
                features.prepare_features(root, root / 'features', 'test', FakeNative('A'),
                    torch.device('cpu'), common_root=root / 'base', manifest={}, data_root=root)


if __name__ == '__main__':
    unittest.main()
