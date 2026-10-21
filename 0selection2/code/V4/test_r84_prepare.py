"""Small CPU cache witnesses; no real dataset scans or solver execution."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from . import r84_prepare as prepare
from .local_geometry import FIELDS, SCALES
from .r76_experiment import write_json


class R84PrepareTests(unittest.TestCase):
    def test_all_old_splits_reused_and_changed_input_not_reused(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cache, data = root / 'cache', root / 'data'
            for split, source_id, value in (('train', 10, 1.), ('val', 20, 2.), ('test', 30, 3.)):
                directory = cache / split / 'TSP'
                directory.mkdir(parents=True)
                inputs = data / ('TSP' + split)
                inputs.mkdir(parents=True)
                np.savez(inputs / 'input_metadata.npz', source_index=np.array([source_id], np.int64))
                values = dict(node=np.full((1, 3, 2), value, np.float32),
                    node_geom=np.full((1, 3, len(FIELDS)), value + 10, np.float32),
                    n=np.array([3], np.int64), coord_dist=np.array([0], np.int64))
                arrays = {}
                for key, array in values.items():
                    np.save(directory / (key + '.npy'), array)
                    arrays[key] = dict(file=key + '.npy', shape=list(array.shape), dtype=str(array.dtype))
                write_json(directory / 'metadata.json', dict(complete=True, rows_written=1, rows=1,
                    kind='coord', geometry_fields=list(FIELDS), geometry_scales=list(SCALES), arrays=arrays))
            reuse = prepare.ReusableInputs('TSP', [cache], data)
            try:
                self.assertEqual(len(reuse.sources), 3)
                batch = dict(kind='coord', node=torch.tensor([[[v, v]] * 3 for v in (1., 2., 3., 4.)]),
                    n=torch.tensor([3, 3, 3, 3]), coord_dist=torch.tensor([0, 0, 0, 0]))
                target = np.zeros((4, 3, len(FIELDS)), np.float32)
                reused = reuse.copy_geometry(batch, np.array([10, 20, 30, 10]), target)
                np.testing.assert_array_equal(reused, [True, True, True, False])
                np.testing.assert_array_equal(target[:, 0, 0], [11., 12., 13., 0.])
            finally:
                reuse.close()

    def test_only_valid_train_nodes_supply_moments(self):
        values = np.array([[[1., 2.], [999., 999.]], [[3., 4.], [5., 6.]]], np.float32)
        moments = dict(count=0, sum=np.zeros(2), squared_sum=np.zeros(2))
        prepare.add_moments(values, np.array([1, 2]), moments)
        self.assertEqual(moments['count'], 3)
        np.testing.assert_array_equal(moments['sum'], [9, 12])
        np.testing.assert_array_equal(moments['squared_sum'], [35, 56])

    def test_test_preparation_denied_before_real_lock(self):
        with self.assertRaises(PermissionError):
            prepare.test_permission(None)

    def test_four_worker_prepare_cli(self):
        for rank in range(4):
            args = prepare.parser().parse_args(['prepare', '--cache-root', '/tmp/synthetic',
                '--workers', '4', '--worker-rank', str(rank)])
            self.assertEqual(args.splits, ['train', 'val'])
            self.assertEqual((args.workers, args.worker_rank), (4, rank))


if __name__ == '__main__':
    unittest.main()
