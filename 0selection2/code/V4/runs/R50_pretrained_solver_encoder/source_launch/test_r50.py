"""Focused R50 input, frozen-state, node alignment and readout checks."""

import copy
import unittest

import numpy as np
import torch
from torch import nn

from .r48_common import binary_labels, binary_metrics
from .r49_experiment import development_improved
from .r50_readout import SolverEncoderReadout
from .r50_solver_encoder import FrozenSolverPair, encoder_arguments, raw_nodes


class R50Tests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(2)
        self.model = SolverEncoderReadout().eval()
        self.encoded = torch.randn(3, 7, 2, 128)
        self.raw = torch.randn(3, 7, 7)
        self.mask = torch.arange(7)[None] < torch.tensor([4, 6, 7])[:, None]

    def test_padding_does_not_change_scores(self):
        first = self.model(self.encoded, self.raw, self.mask)
        extended = torch.cat((self.encoded, torch.randn(3, 4, 2, 128) * 1000), 1)
        raw = torch.cat((self.raw, torch.randn(3, 4, 7) * 1000), 1)
        mask = torch.cat((self.mask, torch.zeros(3, 4, dtype=torch.bool)), 1)
        torch.testing.assert_close(first, self.model(extended, raw, mask), rtol=1e-6, atol=1e-7)

    def test_customer_permutation_keeps_depot_and_prediction(self):
        order = torch.tensor([0, 4, 2, 6, 1, 3, 5])
        torch.testing.assert_close(self.model(self.encoded, self.raw, self.mask),
            self.model(self.encoded[:, order], self.raw[:, order], self.mask[:, order]), rtol=1e-6, atol=1e-7)

    def test_readout_trainable_but_cache_frozen(self):
        cache = self.encoded.clone()
        self.model.train()
        loss = nn.functional.cross_entropy(self.model(cache, self.raw, self.mask), torch.tensor([0, 1, 0]))
        loss.backward()
        self.assertIsNone(cache.grad)
        for block in (self.model.moel_norm, self.model.mtl_norm, self.model.node_mlp, self.model.head):
            gradients = [p.grad for p in block.parameters()]
            self.assertTrue(all(g is not None and torch.isfinite(g).all() for g in gradients))
            self.assertGreater(sum(float(g.abs().sum()) for g in gradients), 0)

    def test_native_encoder_fields_and_depot(self):
        batch = (torch.randn(2, 1, 2), torch.randn(2, 4, 2), torch.rand(2, 4),
                 torch.rand(2, 4), torch.rand(2, 4), torch.rand(2, 4))
        depot, inputs = encoder_arguments(batch)
        self.assertEqual(inputs.shape, (2, 4, 5))
        torch.testing.assert_close(inputs[..., 2], batch[2])
        torch.testing.assert_close(inputs[..., 3], batch[4])
        nodes = raw_nodes(batch)
        torch.testing.assert_close(nodes[:, 0, :2], depot[:, 0])
        torch.testing.assert_close(nodes[:, 1:, 3], batch[3])
        self.assertTrue((nodes[:, 0, -1] == 1).all())
        self.assertTrue((nodes[:, 1:, -1] == 0).all())

    def test_frozen_pair_cannot_enable_training(self):
        class Encoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(5, 128)

            def forward(self, depot, nodes):
                padded = torch.cat((depot, torch.zeros(len(depot), 1, 3)), -1)
                return self.linear(torch.cat((padded, nodes), 1))

        pair = FrozenSolverPair([Encoder(), Encoder()])
        pair.train()
        self.assertFalse(pair.training)
        self.assertTrue(all(not p.requires_grad for p in pair.parameters()))
        batch = (torch.randn(2, 1, 2), torch.randn(2, 4, 2), *[torch.rand(2, 4) for _ in range(4)])
        self.assertFalse(pair(batch).requires_grad)

    def test_identical_readout_initialization(self):
        other = SolverEncoderReadout()
        other.load_state_dict(copy.deepcopy(self.model.state_dict()))
        other.eval()
        torch.testing.assert_close(other(self.encoded, self.raw, self.mask),
                                  self.model(self.encoded, self.raw, self.mask), rtol=0, atol=0)

    def test_fp64_ties_and_development_only_selection(self):
        costs = np.array([[1., 1. + 1e-10], [2., 2.], [3.1, 3.]], dtype=np.float64)
        np.testing.assert_array_equal(binary_labels(costs), [0, -1, 1])
        metrics = binary_metrics(np.array([[1., 0.], [0., 1.], [0., 1.]]), costs)
        self.assertEqual(metrics['n'], 2)
        self.assertEqual(metrics['accuracy'], 1.)
        self.assertFalse(development_improved({'development': {'ce': .7}, 'internal': {'ce': .1}}, .6))
        self.assertTrue(development_improved({'development': {'ce': .5}, 'original_val': {'ce': 50}}, .6))


if __name__ == '__main__':
    unittest.main()
