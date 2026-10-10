"""Focused route-target, masking, gradient, and inference-boundary checks."""

import unittest

import numpy as np
import torch
from torch import nn

from .pair_specialist import INPUT_FIELDS
from .r53_experiment import model_params
from .r54_model import RouteSupervisedSpecialist, SuccessorHead, successor_objective
from .r54_routes import route_cost, route_successors


class NodeStub(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(8, 128)

    def forward(self, batch):
        return self.projection(batch['node']), batch['node_mask']


class R54Tests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(2)
        torch.set_num_threads(2)

    def test_directed_successor_and_end(self):
        np.testing.assert_array_equal(route_successors([0, 2, 1, 0, 3, 4, 0, 0], 4), [0, 1, 4, 0])

    def test_vehicle_route_order_is_irrelevant(self):
        a = route_successors([0, 1, 2, 0, 3, 4, 0], 4)
        b = route_successors([0, 3, 4, 0, 1, 2, 0, 0], 4)
        np.testing.assert_array_equal(a, b)

    def test_invalid_routes_are_rejected(self):
        for route in ([0, 1, 1, 0], [0, 1, 0], [1, 2, 0], [0, 3, 1, 2]):
            with self.assertRaises(ValueError):
                route_successors(route, 2)

    def test_open_and_closed_cost(self):
        instance = dict(depot_xy=[[0., 0.]], node_xy=[[1., 0.], [2., 0.]])
        self.assertEqual(route_cost(instance, [0, 1, 2, 0], True), 2.)
        self.assertEqual(route_cost(instance, [0, 1, 2, 0], False), 4.)

    def test_uniform_loss_is_log_candidate_normalized(self):
        nodes = torch.randn(2, 5, 128)
        mask = torch.ones(2, 5, dtype=torch.bool)
        head = SuccessorHead(128)
        logits = head(nodes, mask, torch.randn(2, 5, 2))
        logits = torch.where(torch.isfinite(logits), torch.zeros_like(logits), logits)
        target = torch.tensor([[[2, 3, 4, 0]] * 2] * 2)
        loss, _ = successor_objective(logits[:, None].expand(-1, 2, -1, -1), target, mask[:, 1:])
        torch.testing.assert_close(loss, torch.tensor(1.))

    def test_padding_is_not_a_candidate_and_does_not_change_loss(self):
        head = SuccessorHead(128).eval()
        nodes, xy = torch.randn(1, 4, 128), torch.randn(1, 4, 2)
        mask = torch.ones(1, 4, dtype=torch.bool)
        first = head(nodes, mask, xy)
        padded = head(torch.cat((nodes, torch.full((1, 2, 128), 999.)), 1),
            torch.cat((mask, torch.zeros(1, 2, dtype=torch.bool)), 1),
            torch.cat((xy, torch.full((1, 2, 2), 999.)), 1))
        torch.testing.assert_close(first, padded[:, :3, [0, 1, 2, 5]])
        self.assertTrue(torch.isneginf(padded[:, :3, 3:5]).all())
        target = torch.tensor([[[2, 3, 0]] * 2])
        a, _ = successor_objective(first[:, None].expand(-1, 2, -1, -1), target, mask[:, 1:])
        padded_target = torch.cat((target, torch.full((1, 2, 2), -100)), -1)
        b, _ = successor_objective(padded[:, None].expand(-1, 2, -1, -1), padded_target,
                                  torch.tensor([[True, True, True, False, False]]))
        torch.testing.assert_close(a, b)

    def test_customer_permutation(self):
        head = SuccessorHead(128).eval()
        nodes, xy = torch.randn(1, 5, 128), torch.randn(1, 5, 2)
        mask = torch.ones(1, 5, dtype=torch.bool)
        permutation = torch.tensor([3, 1, 0, 2])
        all_nodes = torch.cat((torch.tensor([0]), permutation + 1))
        expected = head(nodes, mask, xy)[:, permutation][:, :, torch.cat((permutation, torch.tensor([4])))]
        actual = head(nodes[:, all_nodes], mask[:, all_nodes], xy[:, all_nodes])
        torch.testing.assert_close(actual, expected)

    def test_targets_are_loss_only_and_inference_skips_heads(self):
        self.assertNotIn('route_successor', INPUT_FIELDS)
        model = RouteSupervisedSpecialist(model_params()).eval()
        model.instance_encoder = NodeStub()
        batch = dict(node=torch.randn(2, 5, 8), node_mask=torch.ones(2, 5, dtype=torch.bool), n=torch.tensor([5, 5]))
        calls = []
        hooks = [head.register_forward_hook(lambda *args: calls.append(1)) for head in model.successor_heads]
        first = model(batch)
        self.assertEqual(calls, [])
        second = model(dict(batch, route_successor=torch.zeros(2, 2, 4), costs=torch.randn(2, 7), ind=torch.zeros(2)))
        torch.testing.assert_close(first, second, atol=0, rtol=0)
        out = model(batch, return_aux=True)
        torch.testing.assert_close(first, out['logit'], atol=0, rtol=0)
        self.assertEqual(len(calls), 2)
        for hook in hooks:
            hook.remove()

    def test_auxiliary_only_gradient_reaches_encoder_and_both_heads(self):
        model = RouteSupervisedSpecialist(model_params())
        model.instance_encoder = NodeStub()
        batch = dict(node=torch.randn(2, 5, 8), node_mask=torch.ones(2, 5, dtype=torch.bool), n=torch.tensor([5, 5]))
        out = model(batch, return_aux=True)
        target = torch.tensor([[[2, 3, 4, 0], [0, 1, 2, 3]]] * 2)
        loss, _ = successor_objective(out['successor_logits'], target, out['customer_mask'])
        loss.backward()
        for module in [model.instance_encoder, *model.successor_heads]:
            gradients = [p.grad for p in module.parameters() if p.grad is not None]
            self.assertTrue(gradients)
            self.assertTrue(all(torch.isfinite(g).all() for g in gradients))
            self.assertGreater(sum(float(g.abs().sum()) for g in gradients), 0.)
        self.assertTrue(all(p.grad is None for p in model.head.parameters()))

    def test_missing_or_self_successor_is_rejected(self):
        logits = torch.zeros(1, 2, 3, 4)
        for target in (torch.tensor([[[-100, 0, 0]] * 2]), torch.tensor([[[1, 0, 0]] * 2])):
            with self.assertRaises(ValueError):
                successor_objective(logits, target, torch.ones(1, 3, dtype=torch.bool))


if __name__ == '__main__':
    unittest.main()
