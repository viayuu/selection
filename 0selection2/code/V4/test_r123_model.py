"""Small CPU checks; no formal training or prediction-pool audit."""

import unittest

import torch
from torch import nn

from code.unified_selector.registry import P2I, constraint_bits, problem_descriptor
from .r123_model import (HistoricalModelRouter, ReliabilityGate, brier_loss,
                         correctness_targets, gate_inputs)


def params():
    return dict(embedding_dim=16, head_num=2, qkv_dim=8, ff_hidden_dim=32,
        encoder_layer_num=1, dropout=0., stats_dim=12, sdpa=False, local_geometry=False,
        ignore_coord_dist=True)


def batch(problem='TSP'):
    torch.manual_seed(12)
    n = torch.tensor([3, 4])
    data = dict(problem_id=P2I[problem], kind='matrix' if problem == 'ATSP' else 'coord',
        n=n, node_mask=torch.arange(4)[None] < n[:, None],
        cbits=torch.tensor(constraint_bits(problem), dtype=torch.float32).repeat(2, 1),
        problem_desc=torch.tensor(problem_descriptor(problem), dtype=torch.float32).repeat(2, 1),
        coord_dist=torch.zeros(2, dtype=torch.long), pool_ids=torch.tensor([7, 2, 9]),
        solver_mask=torch.ones(2, 3, dtype=torch.bool))
    data['matrix' if problem == 'ATSP' else 'node'] = torch.rand(2, 4, 4 if problem == 'ATSP' else 8)
    return data


class ModelTests(unittest.TestCase):
    def test_multilabel_correctness_and_native_solver_id_ties(self):
        logits = torch.tensor([[[1., 1., 0.], [0., 3., 2.], [3., 2., 1.]],
                               [[5., 0., 0.], [5., 0., 0.], [0., 0., 5.]],
                               [[0., 5., 0.], [5., 0., 0.], [0., 5., 0.]]])
        targets, actions = correctness_targets(logits, torch.tensor([1, 0, 2]), torch.tensor([7, 2, 9]))
        torch.testing.assert_close(targets, torch.tensor([[1., 1., 0.], [1., 1., 0.], [0., 0., 0.]]))
        self.assertEqual(actions[0, 0], 1)
        values = torch.zeros_like(targets, requires_grad=True)
        brier_loss(values, targets).backward()
        self.assertTrue((values.grad[targets == 1] < 0).all())
        self.assertTrue((values.grad[targets == 0] > 0).all())

    def test_gate_is_label_and_prediction_isolated_padding_invariant(self):
        gate = ReliabilityGate(params()).eval()
        data = batch()
        expected = gate(data)['reliability_logits']
        dirty = dict(data, costs=torch.full((2, 3), -999.), ind=torch.tensor([0, 1]),
            expert_logits=torch.full((2, 3, 3), 999.), source_identity=['answer', 'private'])
        dirty['node'] = data['node'].clone()
        dirty['node'][0, 3] = 1000.
        torch.testing.assert_close(gate(dirty)['reliability_logits'], expected)
        self.assertFalse(set(gate_inputs(dirty)) & {'costs', 'ind', 'expert_logits', 'source_identity'})

    def test_raw_coordinate_and_directed_matrix_paths_receive_gradients(self):
        for problem, field in (('VRPTW', 'coord_proj'), ('ATSP', 'matrix_proj')):
            gate = ReliabilityGate(params()).train()
            data = batch(problem)
            brier_loss(gate(data)['reliability_logits'], torch.tensor([[1., 0., 1.], [0., 0., 0.]])).backward()
            self.assertGreater(float(getattr(gate.instance_encoder, field).weight.grad.abs().sum()), 0.)
            self.assertGreater(float(gate.read.Wq.weight.grad.abs().sum()), 0.)

    def test_sparse_live_execution_equals_cache_selects_no_unselected_rows(self):
        class FixedGate(nn.Module):
            expert_ids = ('R77B', 'R80A', 'R83')

            def forward(self, inputs):
                out = torch.tensor([[2., 1., 0.], [0., 0., 2.]])
                return dict(reliability_logits=out, reliability=out.sigmoid())

        class Expert(nn.Module):
            def __init__(self, index):
                super().__init__()
                self.weight = nn.Parameter(torch.tensor(float(index)))
                self.rows = 0

            def forward(self, inputs):
                self.rows += len(inputs['n'])
                self.assert_no_labels = not (set(inputs) & {'costs', 'ind', 'expert_logits'})
                return dict(logits=torch.tensor([[1., 2., 3.]]) + self.weight)

        experts = [Expert(i) for i in range(3)]
        router = HistoricalModelRouter(FixedGate(), experts, {'TSP': [7, 2, 9]}).train()
        data = dict(batch(), costs=torch.zeros(2, 3), ind=torch.zeros(2, dtype=torch.long))
        del data['solver_mask']  # ordinary LIB batches may omit this optional mask
        live = router(data)
        self.assertEqual([e.rows for e in experts], [1, 0, 1])
        self.assertTrue(all(not e.training and not e.weight.requires_grad for e in experts))
        self.assertTrue(experts[0].assert_no_labels and experts[2].assert_no_labels)
        bank = torch.stack([torch.tensor([[1., 2., 3.], [1., 2., 3.]]) + i for i in range(3)], 1)
        torch.testing.assert_close(router(data, expert_logits=bank)['logits'], live['logits'])
        with self.assertRaisesRegex(ValueError, 'legal candidate pool'):
            router(dict(data, pool_ids=torch.tensor([7, 2, 10])))


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main()
