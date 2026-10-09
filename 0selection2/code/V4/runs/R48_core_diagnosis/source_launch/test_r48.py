"""Focused tests for the diagnostic, without reading any test split."""

import unittest

import numpy as np

from .r48_common import (binary_labels, binary_metrics, check_open_tw_route,
                         original_route, permutation, tree_features)


def example():
    return dict(depot_xy=[[0., 0.]], node_xy=[[.3, 0.], [.6, 0.]],
                node_demand=[.4, .6], capacity=1., service_time=[.05, .1],
                tw_start=[0., 0.], tw_end=[2., 2.])


class CoreTests(unittest.TestCase):
    def test_open_distance_and_route_reset(self):
        row = check_open_tw_route(example(), [0, 1, 2, 0, 0])
        self.assertTrue(row['feasible'])
        self.assertAlmostEqual(row['recomputed_cost'], .6)
        split = check_open_tw_route(example(), [0, 1, 0, 2, 0])
        self.assertTrue(split['feasible'])
        self.assertAlmostEqual(split['recomputed_cost'], .9)
        self.assertEqual(split['routes'], 2)

    def test_capacity_time_and_coverage(self):
        item = example()
        item['node_demand'] = [.6, .6]
        self.assertIn('capacity', check_open_tw_route(item, [0, 1, 2, 0])['violations'])
        self.assertTrue(check_open_tw_route(item, [0, 1, 0, 2, 0])['feasible'])
        item = example()
        item['tw_end'][1] = .64
        self.assertIn('time_window', check_open_tw_route(item, [0, 1, 2, 0])['violations'])
        self.assertIn('missing_customers', check_open_tw_route(example(), [0, 1, 0])['violations'])
        self.assertIn('duplicate_customers', check_open_tw_route(example(), [0, 1, 1, 2, 0])['violations'])

    def test_node_permutation_synchronizes_attributes(self):
        item = example()
        changed = permutation(item, [1, 0])
        self.assertEqual(changed['depot_xy'], item['depot_xy'])
        self.assertEqual(changed['node_demand'], [.6, .4])
        self.assertEqual(item['node_demand'], [.4, .6])
        route = original_route([0, 2, 1, 0], [1, 0])
        self.assertEqual(route, [0, 1, 2, 0])
        self.assertTrue(check_open_tw_route(item, route)['feasible'])

    def test_fp64_comparison_and_exact_ties(self):
        costs = np.array([[1., np.nextafter(1., 2.)], [2., 1.], [1., 1.]], dtype=np.float64)
        self.assertEqual(binary_labels(costs).tolist(), [0, 1, -1])
        with self.assertRaises(ValueError):
            binary_labels(costs.astype(np.float32))

    def test_binary_metrics_exclude_ties(self):
        costs = np.array([[10., 11.], [12., 10.], [10., 10.]], dtype=np.float64)
        result = binary_metrics(np.array([[2., 0.], [2., 0.], [0., 2.]]), costs)
        self.assertEqual(result['n'], 2)
        self.assertEqual(result['n_ties'], 1)
        self.assertEqual(result['accuracy'], .5)
        self.assertAlmostEqual(result['pair_regret_pct'], 10.)

    def test_original_csv_index_order(self):
        import tempfile
        from pathlib import Path
        from .r48_common import read_indexed_costs
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'costs.txt'
            path.write_text('0,1.25\n1,2.5\n')
            np.testing.assert_array_equal(read_indexed_costs(path),[1.25,2.5])
            path.write_text('1,1.25\n0,2.5\n')
            with self.assertRaises(ValueError):
                read_indexed_costs(path)

    def test_tree_inputs_permutation_invariant(self):
        first = tree_features(example())
        second = tree_features(permutation(example(), [1, 0]))
        self.assertEqual(first.keys(), second.keys())
        np.testing.assert_allclose(list(first.values()), list(second.values()), atol=1e-12, rtol=1e-12)

    def test_clean_neural_label_isolation_and_gradient(self):
        import torch
        from .r48_binary import BinaryInstanceClassifier
        params = dict(embedding_dim=16,head_num=4,qkv_dim=4,ff_hidden_dim=64,dropout=0.,
                      encoder_layer_num=1,stats_dim=12,node_only=True,ignore_coord_dist=True)
        torch.manual_seed(2)
        model = BinaryInstanceClassifier(params).eval()
        batch = dict(kind='coord',problem_id=7,node=torch.randn(2,3,8),
                     node_mask=torch.ones(2,3,dtype=torch.bool),n=torch.tensor([3,3]),
                     cbits=torch.zeros(2,5),problem_desc=torch.zeros(2,16),coord_dist=torch.zeros(2,dtype=torch.long))
        expected = model(batch)
        batch.update(costs=torch.randn(2,7),ind=torch.tensor([6,0]),binary_label=torch.tensor([0,1]))
        torch.testing.assert_close(model(batch),expected,atol=0,rtol=0)
        torch.nn.functional.cross_entropy(model(batch),batch['binary_label']).backward()
        self.assertGreater(float(model.instance_encoder.coord_proj.weight.grad.abs().sum()),0.)
        self.assertGreater(float(model.head[-1].weight.grad.abs().sum()),0.)
        self.assertFalse(any('solver' in name or 'retrieval' in name for name,_ in model.named_parameters()))


if __name__ == '__main__':
    unittest.main()
