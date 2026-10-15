import unittest

import numpy as np
import torch

from .r76_old_test import align_labels, aligned_inputs


class OldTestAlignmentTests(unittest.TestCase):
    def test_filter_and_alias_use_named_deployment_not_best_of_reld(self):
        old = dict(pool=np.array(['RELD_MTL', 'MTPOMO', 'RELD_MOEL', 'RouteFinder']),
                   pool_ids=np.array([17, 10, 16, 18]),
                   costs=np.array([[1., 5., 3., 2.], [4., 2., 3., 1.]], dtype=np.float64),
                   winner=np.array([0, 3]), indices=np.arange(2), nodes=np.array([51, 51]))
        result, changes = align_labels(old, ['MTPOMO', 'reld_mt'], [10, 16], {'reld_mt': 'RELD_MOEL'})
        np.testing.assert_array_equal(result['costs'], [[5., 3.], [2., 3.]])
        np.testing.assert_array_equal(result['winner'], [1, 0])
        np.testing.assert_array_equal(result['oracle'], [3., 2.])
        np.testing.assert_array_equal(result['cost_order'], [[1, 0], [0, 1]])
        self.assertEqual(changes, 2)

    def test_fp64_comparison_and_global_id_tie_break(self):
        old = dict(pool=np.array(['MTPOMO', 'RELD_MOEL']), pool_ids=np.array([10, 16]),
                   costs=np.array([[1. + 1e-10, 1.], [2., 2.]], dtype=np.float64),
                   winner=np.array([1, 0]), indices=np.arange(2), nodes=np.array([51, 51]))
        result, changes = align_labels(old, ['MTPOMO', 'reld_mt'], [10, 16], {'reld_mt': 'RELD_MOEL'})
        np.testing.assert_array_equal(result['winner'], [1, 0])
        self.assertEqual(result['costs'].dtype, np.float64)
        self.assertEqual(changes, 0)

    def test_inputs_exclude_labels_and_replace_all_candidate_masks(self):
        node = torch.zeros(2, 3, 8)
        batch = dict(node=node, node_mask=torch.ones(2, 3, dtype=torch.bool),
                     pool_ids=torch.tensor([10, 11, 12, 13, 16, 17, 18]),
                     mask=torch.ones(2, 22), solver_mask=torch.ones(2, 7, dtype=torch.bool),
                     costs=torch.ones(2, 7), ind=torch.ones(2, dtype=torch.long),
                     cmp_valid=torch.ones(2, 21, dtype=torch.bool), relative_gap=torch.ones(2, 7))
        result = aligned_inputs(batch, [10, 11, 16])
        for field in ('costs', 'ind', 'cmp_valid', 'relative_gap'):
            self.assertNotIn(field, result)
        torch.testing.assert_close(result['pool_ids'], torch.tensor([10, 11, 16]))
        self.assertEqual(result['solver_mask'].shape, (2, 3))
        self.assertEqual(result['mask'].sum().item(), 6)
        self.assertIs(result['node'], node)
        self.assertEqual(batch['pool_ids'].numel(), 7)


if __name__ == '__main__':
    unittest.main()
