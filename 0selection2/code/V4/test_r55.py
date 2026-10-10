"""Essential count conversion, bottleneck, symmetry, masking and gradient tests."""

import unittest

import numpy as np
import torch
from torch import nn

from code.unified_selector.registry import K_CBITS, K_PROBLEM_DESC
from .pair_specialist import apply_specialist
from .r53_experiment import model_params
from .r55_model import (CostDifferenceSpecialist, SymmetricEdgeDecoder,
                        analytic_cost_difference, difference_objective)
from .r55_targets import billed_edges, distances


class NodeStub(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(8, 128)

    def forward(self, batch):
        return self.projection(batch['node']), batch['node_mask']


class R55Tests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(2)
        torch.set_num_threads(2)

    def sample(self, batch=2, n=5):
        return dict(node=torch.randn(batch, n, 8), node_mask=torch.ones(batch, n, dtype=torch.bool),
            n=torch.full((batch,), n), cbits=torch.zeros(batch, K_CBITS),
            problem_desc=torch.zeros(batch, K_PROBLEM_DESC))

    def model(self):
        model = CostDifferenceSpecialist(model_params()).eval()
        model.instance_encoder = NodeStub()
        return model

    def test_closed_single_customer_is_charged_twice(self):
        counts = billed_edges([0, 1, 0], 1, False)
        self.assertEqual(counts[0, 1], 2)
        self.assertEqual(counts.sum(), 2)

    def test_open_returns_removed_before_direction_merge(self):
        self.assertEqual(billed_edges([0, 1, 0], 1, True)[0, 1], 1)
        np.testing.assert_array_equal(billed_edges([0, 1, 2, 0], 2, True),
                                      np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]]))

    def test_exact_difference_cost_and_direction(self):
        instance = dict(depot_xy=[[0., 0.]], node_xy=[[1., 0.], [2., 0.]])
        for opened in (True, False):
            a = billed_edges([0, 1, 2, 0], 2, opened)
            b = billed_edges([0, 1, 0, 2, 0], 2, opened)
            delta = ((b - a) * distances(instance)).sum()
            self.assertEqual(delta, 1. if opened else 2.)
            self.assertGreater(delta, 0.)

    def test_invalid_route_rejected(self):
        for route in ([0, 1, 1, 0], [0, 1, 0], [1, 2, 0], [0, 1, 3, 0]):
            with self.assertRaises(ValueError):
                billed_edges(route, 2, False)

    def test_analytic_readout_has_no_bypass(self):
        model, batch = self.model(), self.sample()
        out = model(batch, return_edges=True)
        expected = analytic_cost_difference(out['edge_difference'], out['edge_distance'], out['edge_mask'])
        torch.testing.assert_close(model(batch), expected, rtol=0, atol=0)
        self.assertFalse(hasattr(model, 'head'))
        self.assertFalse(any('solver' in name for name, _ in model.named_parameters()))
        zero = analytic_cost_difference(torch.zeros_like(out['edge_difference']), out['edge_distance'], out['edge_mask'])
        torch.testing.assert_close(zero, torch.zeros_like(zero), rtol=0, atol=0)

    def test_swap_endpoints_and_permute_customers(self):
        model, batch = self.model(), self.sample()
        out = model(batch, return_edges=True)
        n = batch['node'].shape[1]
        dense = torch.zeros(2, n, n)
        dense[:, out['row'], out['col']] = out['edge_difference']
        dense = dense + dense.transpose(1, 2)
        perm = torch.tensor([0, 3, 1, 4, 2])
        second = model(dict(batch, node=batch['node'][:, perm], node_mask=batch['node_mask'][:, perm]), return_edges=True)
        expected = dense[:, perm][:, :, perm][:, second['row'], second['col']]
        torch.testing.assert_close(second['edge_difference'], expected)
        torch.testing.assert_close(second['cost_difference'], out['cost_difference'])

    def test_padding_does_not_change_prediction_or_loss(self):
        model, batch = self.model(), self.sample()
        first = model(batch, return_edges=True)
        padded = dict(batch, node=torch.cat((batch['node'], torch.full((2, 2, 8), 1000.)), 1),
            node_mask=torch.cat((batch['node_mask'], torch.zeros(2, 2, dtype=torch.bool)), 1))
        second = model(padded, return_edges=True)
        torch.testing.assert_close(first['cost_difference'], second['cost_difference'])
        self.assertTrue((second['edge_difference'][~second['edge_mask']] == 0).all())
        target = torch.zeros(2, 7, 7, dtype=torch.int8)
        target[:, 0, 1], target[:, 1, 2] = 1, -1
        a, _ = difference_objective(first, target, torch.ones(2), torch.ones(2))
        target[:, 5:, :] = 100
        target[:, :, 5:] = -100
        b, _ = difference_objective(second, target, torch.ones(2), torch.ones(2))
        torch.testing.assert_close(a, b)

    def test_label_isolation(self):
        model, batch = self.model(), self.sample()
        expected = model(batch)
        altered = dict(batch, costs=torch.randn(2, 7), binary_label=torch.ones(2), ind=torch.ones(2),
            oracle_cost_target=torch.ones(2), cost_difference_target=torch.full((2,), 1000.),
            edge_difference_target=torch.ones(2, 5, 5), route_successor=torch.ones(2, 2, 4))
        torch.testing.assert_close(model(altered), expected, rtol=0, atol=0)

    def test_group_loss_is_instance_normalized_not_zero_dominated(self):
        out = dict(edge_mask=torch.ones(1, 3, dtype=torch.bool), edge_difference=torch.zeros(1, 3),
            row=torch.tensor([0, 0, 1]), col=torch.tensor([1, 2, 2]), cost_difference=torch.zeros(1))
        target = torch.tensor([[[0, 1, -1], [0, 0, 0], [0, 0, 0]]])
        loss, stats = difference_objective(out, target, torch.zeros(1), torch.ones(1))
        torch.testing.assert_close(loss, torch.tensor(2. / 3.))
        torch.testing.assert_close(stats['zero_prediction_edge_loss'], loss)
        out['edge_mask'][:, -1] = False
        loss, _ = difference_objective(out, target, torch.zeros(1), torch.ones(1))
        torch.testing.assert_close(loss, torch.tensor(1.))

    def test_all_zero_targets_and_relative_scale(self):
        out = dict(edge_mask=torch.ones(1, 1, dtype=torch.bool), edge_difference=torch.zeros(1, 1),
            row=torch.tensor([0]), col=torch.tensor([1]), cost_difference=torch.tensor([.1]))
        loss, stats = difference_objective(out, torch.zeros(1, 2, 2), torch.zeros(1), torch.tensor([10.]))
        torch.testing.assert_close(stats['edge_loss'], torch.tensor(0.))
        torch.testing.assert_close(loss, torch.tensor(1.))

    def test_cost_only_gradient_reaches_encoder_and_edge_decoder(self):
        model, batch = self.model(), self.sample()
        out = model(batch, return_edges=True)
        out['cost_difference'].square().mean().backward()
        for module in (model.instance_encoder, model.edge_decoder):
            grads = [p.grad for p in module.parameters() if p.grad is not None]
            self.assertTrue(grads)
            self.assertTrue(all(torch.isfinite(g).all() for g in grads))
            self.assertGreater(sum(float(g.abs().sum()) for g in grads), 0.)

    def test_complete_valid_edges_not_truth_filtered(self):
        out = self.model()(self.sample(), return_edges=True)
        self.assertTrue((out['edge_mask'].sum(-1) == 10).all())

    def test_positive_prediction_selects_moel_only_at_top2_gate(self):
        order = np.array([[0, 1, 2], [1, 0, 2], [2, 0, 1]])
        selected, gate = apply_specialist(order, ['RELD_MOEL', 'RELD_MTL', 'other'], np.array([1., -1., 100.]))
        np.testing.assert_array_equal(selected[:, 0], [0, 1, 2])
        np.testing.assert_array_equal(gate, [True, True, False])


if __name__ == '__main__':
    unittest.main()
