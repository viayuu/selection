"""Focused R56 directed relation, data-boundary and gradient checks."""

import unittest

import torch

from .r56_model import JointRelationLayer, RelationalPairSpecialist


def params():
    return dict(embedding_dim=128, head_num=4, qkv_dim=32, encoder_layer_num=4,
                ff_hidden_dim=512, dropout=.1, stats_dim=12, local_geometry=False,
                ignore_coord_dist=True)


def sample():
    torch.manual_seed(2)
    node = torch.rand(2, 5, 8)
    node[:, :, 3] = 0.
    node[:, 0, 3] = 1.
    node[:, :, 7] += 1.
    return dict(kind='coord', problem_id=3, node=node, node_mask=torch.ones(2, 5, dtype=torch.bool),
                n=torch.tensor([4, 4]), cbits=torch.zeros(2, 5), problem_desc=torch.zeros(2, 16))


class R56Tests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(2)
        self.model = RelationalPairSpecialist(params()).eval()
        self.batch = sample()

    def test_original_encoder_layers_are_replaced(self):
        self.assertEqual(len(self.model.instance_encoder.layers), 4)
        self.assertTrue(all(isinstance(x, JointRelationLayer) for x in self.model.instance_encoder.layers))
        self.assertFalse(any('solver' in n or 'edge_decoder' in n or 'coord_bias' in n
                             for n, _ in self.model.named_parameters()))

    def test_ordered_relations_and_dense_coverage(self):
        h, e, mask = self.model.instance_encoder.initial_states(self.batch)
        self.assertEqual(e.shape, (2, 8, 8, 32))
        self.assertEqual(int(mask.sum()), 16)
        self.assertGreater(float((e[:, 3, 4]-e[:, 4, 3]).abs().max()), 1e-3)

    def test_label_isolation(self):
        before = self.model(self.batch)
        changed = dict(self.batch, costs=torch.rand(2, 7), ind=torch.ones(2),
                       binary_label=torch.zeros(2), cost_weight=torch.ones(2),
                       edge_difference=torch.ones(2, 5, 5), successor=torch.zeros(2, 5))
        torch.testing.assert_close(before, self.model(changed), rtol=0, atol=0)

    def test_padding_invariance(self):
        original = self.model(self.batch)
        padded = dict(self.batch)
        padded['node'] = torch.cat((padded['node'], torch.full((2, 3, 8), 12345.)), dim=1)
        padded['node_mask'] = torch.cat((padded['node_mask'], torch.zeros(2, 3, dtype=torch.bool)), dim=1)
        torch.testing.assert_close(original, self.model(padded), rtol=2e-5, atol=2e-5)

    def test_node_permutation_and_relation_permutation(self):
        z, state = self.model(self.batch, return_states=True)
        perm = torch.tensor([0, 4, 2, 1, 3])
        changed = dict(self.batch, node=self.batch['node'][:, perm], node_mask=self.batch['node_mask'][:, perm])
        other, new_state = self.model(changed, return_states=True)
        torch.testing.assert_close(z, other, rtol=2e-5, atol=2e-5)
        all_perm = torch.cat((torch.arange(3), perm+3))
        torch.testing.assert_close(state['final_relations'][:, all_perm][:, :, all_perm],
                                   new_state['final_relations'], rtol=2e-5, atol=2e-5)

    def test_both_directions_evolve_over_all_layers(self):
        _, initial, _ = self.model.instance_encoder.initial_states(self.batch)
        _, state = self.model(self.batch, return_states=True)
        previous = initial
        for _, edges in state['states']:
            self.assertGreater(float((edges-previous).abs().max()), 1e-4)
            previous = edges

    def test_edge_state_changes_later_node_attention(self):
        enc = self.model.instance_encoder
        nodes, edges, mask = enc.initial_states(self.batch)
        unchanged, _ = enc.layers[0](nodes, edges, mask)
        perturbed, _ = enc.layers[0](nodes, edges+torch.randn_like(edges), mask)
        self.assertGreater(float((unchanged-perturbed).abs().max()), 1e-4)

    def test_bce_reaches_encoder_relation_bias_gate_and_previous_updates(self):
        z = self.model(self.batch)
        torch.nn.functional.binary_cross_entropy_with_logits(z, torch.tensor([1., 0.])).backward()
        def norm(module):
            grads = [p.grad for p in module.parameters() if p.grad is not None]
            self.assertTrue(grads)
            self.assertTrue(all(torch.isfinite(g).all() for g in grads))
            return sum(float(g.abs().sum()) for g in grads)
        self.assertGreater(norm(self.model.head), 0.)
        self.assertGreater(norm(self.model.instance_encoder.source_projection), 0.)
        self.assertGreater(norm(self.model.instance_encoder.target_projection), 0.)
        for i, layer in enumerate(self.model.instance_encoder.layers):
            self.assertGreater(norm(layer.edge_bias), 0.)
            self.assertGreater(norm(layer.edge_gate), 0.)
            if i < 3:
                self.assertGreater(norm(layer.edge_out), 0.)
                self.assertGreater(norm(layer.edge_ff), 0.)
        # A node-only readout does not supervise the terminal edge-only output.
        self.assertTrue(all(p.grad is None for p in self.model.instance_encoder.layers[-1].edge_out.parameters()))

    def test_dropout_and_repeatable_eval(self):
        torch.testing.assert_close(self.model(self.batch), self.model(self.batch), rtol=0, atol=0)
        self.model.train()
        self.assertFalse(torch.equal(self.model(self.batch), self.model(self.batch)))

    def test_matrix_input_rejected(self):
        with self.assertRaises(ValueError):
            self.model(dict(self.batch, kind='matrix'))


if __name__ == '__main__':
    unittest.main()
