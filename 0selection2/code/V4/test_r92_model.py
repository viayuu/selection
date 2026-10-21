"""Offline two-thread CPU checks of R92 A/B, without datasets or GPU access."""

import copy
import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as F

from code.unified_selector.registry import P2I, PROBLEMS
from .pairwise_objective import LABEL_KEYS
from .r92_model import (ARCHITECTURE, ARMS, coarsen, hard_assignment,
                        input_relations, make_model)
from .test_dual_stream import params as original_params
from .test_solver_features import sample_batch


def model_params():
    return dict(original_params(), embedding_dim=128, ff_hidden_dim=512,
                dropout=.1, encoder_layer_num=4, local_geometry=True)


def graph_batch(problem='CVRP', rows=3, nodes=43):
    """Native input widths and constraint slots, with heterogeneous valid lengths."""
    torch.manual_seed(321 + P2I[problem])
    batch = sample_batch(problem)
    for key, value in list(batch.items()):
        if torch.is_tensor(value) and key != 'pool_ids':
            batch[key] = value[:1].expand(rows, *value.shape[1:]).clone()
    mask = torch.ones(rows, nodes, dtype=torch.bool)
    for row in range(rows):
        mask[row, max(1, nodes - row * 3):] = False
    attributes = torch.rand(rows, nodes, 8)
    attributes[..., 3:] = 0.
    attributes[:, 0, 2] = 0.
    attributes[:, 0, 3] = 1.
    if 'B' in problem:
        attributes[:, 1::2, 2] *= -1.
    if 'L' in problem:
        attributes[:, 1:, 4] = 3.
    if 'TW' in problem:
        attributes[:, 1:, 5] = .2
        attributes[:, 1:, 6] = .3
        attributes[:, 1:, 7] = .8
    matrix = torch.rand(rows, nodes, nodes)
    matrix.diagonal(dim1=1, dim2=2).zero_()
    batch.update(node=attributes[..., :2] if problem == 'TSP' else
                 attributes[..., :4] if problem == 'CVRP' else attributes,
                 matrix=matrix, node_mask=mask, n=mask.sum(-1),
                 node_geom=torch.zeros(rows, nodes, 12))
    if problem == 'ATSP':
        batch['node'] = None
        batch.pop('node_geom')
    else:
        batch['matrix'] = None
    return batch


def subset(batch, index):
    return {key: value[index] if torch.is_tensor(value) and key != 'pool_ids' else value
            for key, value in batch.items()}


def gradients(module):
    values = [parameter.grad for parameter in module.parameters() if parameter.grad is not None]
    return bool(values) and all(torch.isfinite(value).all() for value in values) and sum(
        value.abs().sum().item() for value in values) > 0


class R92ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def make(self, arm):
        torch.manual_seed(2)
        return make_model(model_params(), arm)

    def test_all18_native_fields_finite_outputs_and_backward(self):
        for arm in ARMS:
            model = self.make(arm).train()
            for problem in PROBLEMS:
                with self.subTest(arm=arm, problem=problem):
                    model.zero_grad(set_to_none=True)
                    batch = graph_batch(problem, rows=2, nodes=9)
                    output = model(batch)
                    for key in ('utility', 'pair_margin', 'logits', 'auxiliary_loss'):
                        self.assertTrue(torch.isfinite(output[key]).all())
                    self.assertEqual(set(output['auxiliary_parts']), {'link', 'entropy'})
                    self.assertEqual(output['auxiliary_loss'].ndim, 0)
                    for value in output['auxiliary_parts'].values():
                        self.assertEqual(value.ndim, 0)
                    torch.testing.assert_close(output['auxiliary_loss'],
                                               .1 * output['auxiliary_parts']['link']
                                               + .01 * output['auxiliary_parts']['entropy'])
                    expected_margin = output['utility'][:, :, None] - output['utility'][:, None, :]
                    torch.testing.assert_close(output['pair_margin'], expected_margin)
                    loss = F.cross_entropy(output['logits'], torch.arange(2)) + output['auxiliary_loss']
                    loss.backward()
                    for module in (*model.instance_encoder.blocks, *model.joint_layers,
                                   model.solver_encoder, model.decoder, model.score_head):
                        self.assertTrue(gradients(module))
                    if arm == 'B':
                        for network in model.instance_encoder.assignment_nets:
                            self.assertTrue(gradients(network))
                    else:
                        self.assertEqual(output['auxiliary_loss'].item(), 0.)

    def test_hard_assignments_singleton_mass_dummy_masks_and_ties(self):
        for problem in ('TSP', 'CVRP', 'ATSP'):
            model = self.make('A').eval()
            for nodes in (1, 4, 43):
                with self.subTest(problem=problem, nodes=nodes):
                    batch = graph_batch(problem, rows=2, nodes=nodes)
                    if problem != 'ATSP':
                        batch['node'][..., :2] = .5  # exact coordinate ties
                    output = model(batch, return_hierarchy=True)
                    for clusters, stage in zip((32, 8), output['hierarchy_debug']):
                        assignment, valid = stage['assignment'], stage['input_mask']
                        self.assertEqual(assignment.size(-1), clusters)
                        torch.testing.assert_close(assignment.sum(-1), valid.float())
                        self.assertTrue(((assignment == 0) | (assignment == 1)).all())
                        self.assertEqual(assignment[~valid].count_nonzero().item(), 0)
                        self.assertEqual(stage['mass'][~stage['mask']].count_nonzero().item(), 0)
                        torch.testing.assert_close(stage['mass'].sum(-1), batch['n'].float())
                        self.assertEqual(stage['mask'].sum(-1).tolist(),
                                         valid.sum(-1).clamp_max(clusters).tolist())
                        self.assertFalse(any(value.requires_grad for value in stage.values()))
                        if problem == 'ATSP':
                            self.assertEqual(stage['marker'].count_nonzero().item(), 0)
                        else:
                            torch.testing.assert_close(stage['mass'][:, 0], torch.ones(2))
                            torch.testing.assert_close(assignment[:, :, 0], stage['input_protected'].float())
                            torch.testing.assert_close(stage['marker'][:, 0, 0], torch.ones(2))

    def test_learned_soft_assignment_and_main_loss_reaches_actual_coarse_relations(self):
        model = self.make('B').train()
        relations = []

        def capture(module, arguments):
            relation = arguments[1]
            relation.retain_grad()
            relations.append(relation)

        handles = [block.register_forward_pre_hook(capture) for block in model.instance_encoder.blocks[1:]]
        output = model(graph_batch('ATSP'), return_hierarchy=True)
        F.cross_entropy(output['logits'], torch.tensor([0, 1, 2])).backward()
        for handle in handles:
            handle.remove()
        self.assertEqual(len(relations), 3)
        for relation in relations:
            self.assertIsNotNone(relation.grad)
            self.assertTrue(torch.isfinite(relation.grad).all())
            self.assertGreater(relation.grad.abs().sum().item(), 0.)
        for network in model.instance_encoder.assignment_nets:
            self.assertTrue(gradients(network))
        for stage in output['hierarchy_debug']:
            assignment = stage['assignment']
            torch.testing.assert_close(assignment.sum(-1), stage['input_mask'].float())
            self.assertTrue(((assignment > 0) & (assignment < 1)).any())
            self.assertEqual(stage['relation'].size(-1), 2)
            self.assertFalse(torch.allclose(stage['relation'][..., 0], stage['relation'][..., 0].transpose(1, 2)))

    def test_b_coordinate_singleton_and_small_graph_masks(self):
        model = self.make('B').eval()
        for nodes in (1, 4, 43):
            batch = graph_batch('TSP', nodes=nodes)
            for stage in model(batch, return_hierarchy=True)['hierarchy_debug']:
                assignment = stage['assignment']
                torch.testing.assert_close(assignment[:, :, 0], stage['input_protected'].float())
                torch.testing.assert_close(stage['mass'][:, 0], torch.ones(3))
                torch.testing.assert_close(stage['mass'].sum(-1), batch['n'].float())
                torch.testing.assert_close(assignment.sum(-1), stage['input_mask'].float())
                self.assertEqual(assignment[~stage['input_mask']].count_nonzero().item(), 0)
                self.assertTrue(torch.isfinite(stage['relation']).all())
                self.assertEqual(stage['mask'].sum(-1).tolist(),
                                 stage['input_mask'].sum(-1).clamp_max(assignment.size(-1)).tolist())

    def test_padding_exact_invalid_nodes_and_single_solver(self):
        for arm in ARMS:
            model = self.make(arm).eval()
            for problem in ('TSP', 'VRPBLTW', 'ATSP'):
                with self.subTest(arm=arm, problem=problem):
                    batch = graph_batch(problem, rows=2, nodes=9)
                    before = model(batch)['logits']
                    padded = copy.deepcopy(batch)
                    padded['node_mask'] = F.pad(batch['node_mask'], (0, 7), value=False)
                    if problem == 'ATSP':
                        padded['matrix'] = F.pad(batch['matrix'], (0, 7, 0, 7), value=float('nan'))
                    else:
                        padded['node'] = F.pad(batch['node'], (0, 0, 0, 7), value=float('nan'))
                        padded['node_geom'] = F.pad(batch['node_geom'], (0, 0, 0, 7), value=float('nan'))
                    torch.testing.assert_close(model(padded)['logits'], before, rtol=3e-5, atol=3e-6)
                    ids = torch.cat([batch['pool_ids'], torch.tensor([-1])])
                    mask = torch.ones(2, len(ids), dtype=torch.bool)
                    mask[:, -1] = False
                    output = model(dict(batch, pool_ids=ids, solver_mask=mask))
                    torch.testing.assert_close(output['logits'][:, :-1], before, rtol=3e-5, atol=3e-6)
                    self.assertTrue(torch.isneginf(output['logits'][:, -1]).all())
                    single = model(dict(batch, pool_ids=batch['pool_ids'][:1]))
                    torch.testing.assert_close(single['logits'], torch.zeros(2, 1))

    def test_labels_isolated_role_shared_and_eval_skips_auxiliary(self):
        a, b = self.make('A').eval(), self.make('B').eval()
        batch = graph_batch('TSP', nodes=9)
        prepared = a.prepare_role_inputs(batch)
        torch.testing.assert_close(prepared['node'], b.prepare_role_inputs(batch)['node'])
        torch.testing.assert_close(prepared['node'][:, 0, 3], torch.ones(3))
        self.assertFalse(set(LABEL_KEYS) & prepared.keys())
        self.assertEqual(batch['node'].size(-1), 2)
        for model in (a, b):
            reference = model(batch)
            changed = copy.deepcopy(batch)
            for name in LABEL_KEYS:
                changed[name] = torch.full((3, 4), float('nan'))
            with patch('code.V4.r92_model.structural_auxiliary', side_effect=AssertionError('eval aux')):
                output = model(changed, return_auxiliary=True)
            for key in ('utility', 'pair_margin', 'logits'):
                torch.testing.assert_close(output[key], reference[key], rtol=0, atol=0)
            self.assertEqual(output['auxiliary_loss'].item(), 0.)
            model.train()
            with patch('code.V4.r92_model.structural_auxiliary', side_effect=AssertionError('disabled aux')):
                self.assertEqual(model(batch, return_auxiliary=False)['auxiliary_loss'].item(), 0.)

    def test_raw_relation_coarsening_exact_mass_formula_and_raw_attributes(self):
        torch.manual_seed(6)
        nodes = torch.randn(1, 5, 128)
        attributes = torch.randn(1, 5, 8)
        marker = torch.tensor([[[1.], [0.], [0.], [0.], [0.]]])
        mass = torch.tensor([[1., 2., 3., 4., 0.]])
        mask = mass > 0
        relation = input_relations(torch.rand(1, 5, 5), mask)
        assignment = F.one_hot(torch.tensor([[0, 1, 1, 2, 0]]), 3).float() * mask[..., None]
        result = coarsen(nodes, attributes, marker, mass, relation, assignment)
        z, raw, carried_marker, next_mass, edge, valid = result
        weights = assignment * mass[..., None]
        torch.testing.assert_close(next_mass, torch.tensor([[1., 5., 4.]]))
        torch.testing.assert_close(z, weights.transpose(1, 2) @ nodes / next_mass[..., None])
        torch.testing.assert_close(raw, weights.transpose(1, 2) @ attributes / next_mass[..., None])
        torch.testing.assert_close(carried_marker[:, 0], torch.ones(1, 1))
        for channel in range(2):
            expected = weights.transpose(1, 2) @ relation[..., channel] @ weights
            expected = expected / (next_mass[:, :, None] * next_mass[:, None, :])
            torch.testing.assert_close(edge[..., channel], expected)
        self.assertTrue(valid.all())
        changed_relation = relation.clone()
        changed_relation[:, 0, 1, 0] += .5
        changed = coarsen(nodes, attributes, marker, mass, changed_relation, assignment)[4]
        self.assertFalse(torch.equal(edge, changed))

    def test_auxiliary_row_weighted_microbatch_exact_values_and_gradients(self):
        model = self.make('B').train()
        for module in model.modules():
            if isinstance(module, torch.nn.Dropout):
                module.p = 0.
        batch = graph_batch('ATSP', rows=3, nodes=9)
        full = model(batch)
        parts = {key: value.detach().clone() for key, value in full['auxiliary_parts'].items()}
        full['auxiliary_loss'].backward()
        parameter = model.instance_encoder.assignment_nets[0].network[-1].weight
        gradient = parameter.grad.clone()
        model.zero_grad(set_to_none=True)
        aggregate = {key: 0. for key in parts}
        for indices in (slice(0, 2), slice(2, 3)):
            micro = subset(batch, indices)
            fraction = len(micro['n']) / len(batch['n'])
            output = model(micro)
            (output['auxiliary_loss'] * fraction).backward()
            for key in aggregate:
                aggregate[key] += fraction * output['auxiliary_parts'][key].detach()
        for key in parts:
            torch.testing.assert_close(aggregate[key], parts[key], rtol=2e-5, atol=2e-6)
        torch.testing.assert_close(parameter.grad, gradient, rtol=3e-4, atol=2e-6)

    def test_architecture_common_parameters_geometry_buffers_and_strict_reconstruction(self):
        a, b = self.make('A'), self.make('B')
        self.assertEqual(a.params['r92_architecture'], ARCHITECTURE)
        self.assertEqual(len(a.instance_encoder.blocks), 4)
        self.assertEqual(len(a.joint_layers), 2)
        self.assertEqual(a.solver_encoder.solver_features.size(-1), 34)
        self.assertFalse(hasattr(a.instance_encoder, 'layers'))
        for key in a.state_dict():
            self.assertNotIn('cls_token', key)
            self.assertNotIn('bias_scale', key)
            torch.testing.assert_close(a.state_dict()[key], b.state_dict()[key], rtol=0, atol=0)
        for model in (a, b):
            model.instance_encoder.geometry_residual.mean.fill_(.2)
            model.instance_encoder.geometry_residual.std.fill_(1.3)
            restored = make_model(model.params, model.arm, state_dict=model.state_dict()).eval()
            batch = graph_batch('CVRP', nodes=9)
            torch.testing.assert_close(model.eval()(batch)['logits'], restored(batch)['logits'], rtol=0, atol=0)
        with self.assertRaises(RuntimeError):
            make_model(model_params(), 'B', state_dict=a.state_dict())
        with self.assertRaises(ValueError):
            make_model(model_params(), 'C')
        with self.assertRaises(ValueError):
            make_model(dict(model_params(), embedding_dim=64), 'A')
        with self.assertRaises(ValueError):
            make_model(dict(model_params(), r92_architecture={}), 'A')


if __name__ == '__main__':
    unittest.main()
