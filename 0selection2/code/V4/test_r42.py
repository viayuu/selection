import copy
import unittest
from unittest.mock import patch

import numpy as np
import torch
import torch.nn.functional as F

from code.unified_selector.registry import PROBLEMS
from .performance_experiment import fit_args
from .r42_experiment import CostController, inputs_only, paired_schedule, symmetric_kl, update_batch
from .relation_graph import EDGE_FIELDS, build_relation_graph, nearest
from .solver_query import make_r42_model
from .test_dual_stream import params
from .test_solver_features import sample_batch
from .train import selector_loss


class R42Tests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(2)
        torch.set_num_threads(2)
        self.params = dict(params(), architecture='relation_query', encoder_layer_num=4,
                           local_geometry=True, ignore_coord_dist=True, edge_dim=8)

    def model(self):
        return make_r42_model(self.params).eval()

    def fixture(self, problem='OVRPTW'):
        batch = sample_batch(problem)
        batch['node'][..., 3:] = 0
        batch['node'][:, 0, 3] = 1 if problem != 'TSP' else 0
        batch['node'][:, 1:, 6] = torch.tensor([.1, .2, .3, .4])
        batch['node'][:, 1:, 7] = torch.tensor([.5, .8, .7, .9])
        batch['node'][:, 1:, 5] = .1
        batch['node'][:, 1:, 4] = 3
        batch['ind'] = torch.tensor([0, 1])
        batch['costs'] = torch.rand_like(batch['costs']) + 1
        return batch

    def test_all18_label_free_and_candidate_order(self):
        model = self.model()
        for p in PROBLEMS:
            batch = self.fixture(p)
            reference = model(batch)['logits']
            torch.testing.assert_close(reference, model(inputs_only(batch))['logits'], atol=0, rtol=0)
            changed = dict(batch, costs=torch.full_like(batch['costs'], float('nan')), ind=torch.zeros(2, dtype=torch.long))
            torch.testing.assert_close(model(changed)['logits'], reference, atol=0, rtol=0)
            order = torch.randperm(len(batch['pool_ids']))
            torch.testing.assert_close(model(dict(batch, pool_ids=batch['pool_ids'][order]))['logits'], reference[:, order], atol=3e-6, rtol=3e-5)
        self.assertFalse(hasattr(model, 'joint_layers'))
        self.assertFalse(hasattr(model, 'decoder'))

    def test_padding_candidate_mask_and_batch_isolation(self):
        model = self.model()
        for p in ('CVRP', 'ATSP', 'OVRPTW'):
            batch = self.fixture(p)
            reference = model(batch)['logits']
            padded = copy.deepcopy(batch)
            padded['node'] = F.pad(batch['node'], (0, 0, 0, 3), value=999)
            padded['matrix'] = F.pad(batch['matrix'], (0, 3, 0, 3), value=999)
            padded['node_mask'] = F.pad(batch['node_mask'], (0, 3), value=False)
            torch.testing.assert_close(model(padded)['logits'], reference, atol=3e-6, rtol=3e-5)
            changed = copy.deepcopy(batch)
            changed['node'][1] *= 10
            changed['matrix'][1] *= 10
            torch.testing.assert_close(model(changed)['logits'][0], reference[0], atol=3e-6, rtol=3e-5)
            ids = torch.cat((batch['pool_ids'], torch.tensor([-1])))
            mask = torch.ones(2, len(ids), dtype=torch.bool)
            mask[:, -1] = False
            logits = model(dict(batch, pool_ids=ids, solver_mask=mask))['logits']
            torch.testing.assert_close(logits[:, :-1], reference, atol=3e-6, rtol=3e-5)
            self.assertTrue(torch.isneginf(logits[:, -1]).all())

    def test_node_permutation(self):
        model = self.model()
        for p in ('OVRPTW', 'ATSP'):
            batch = self.fixture(p)
            reference = model(batch)['logits']
            order = torch.tensor([3, 0, 4, 2, 1])
            permuted = dict(batch, node=batch['node'][:, order], node_mask=batch['node_mask'][:, order],
                            matrix=batch['matrix'][:, order][:, :, order])
            torch.testing.assert_close(model(permuted)['logits'], reference, atol=4e-6, rtol=4e-5)

    def test_edges_unique_directional_and_not_cross_batch(self):
        batch = self.fixture('ATSP')
        batch['matrix'][0, 1, 2], batch['matrix'][0, 2, 1] = .12, .91
        graph = build_relation_graph(batch)
        a, b = graph['index']
        self.assertEqual(len(a), len(torch.unique(graph['index'], dim=1)[0]))
        self.assertTrue((a // 5 == b // 5).all())
        self.assertFalse((a == b).any())
        edge = (a == 1) & (b == 2)
        torch.testing.assert_close(graph['features'][edge, :2], torch.tensor([[.12, .91]]))
        self.assertTrue((graph['features'][:, 4:19] == 0).all())
        self.assertTrue((graph['features'][:, 28] == 1).all())

    def test_sparse_incoming_outgoing_selection(self):
        batch = self.fixture('ATSP')
        batch['matrix'][:] = .9
        batch['matrix'][:, 1, 2] = .1
        batch['matrix'][:, 3, 1] = .2
        batch['node_mask'][:, 4] = False
        graph = build_relation_graph(batch, geometric_k=1)
        source, target = graph['index']
        outgoing = graph['features'][(source == 1) & (target == 2)][0]
        incoming = graph['features'][(source == 1) & (target == 3)][0]
        self.assertEqual(outgoing[19].item(), 1)
        self.assertEqual(incoming[20].item(), 1)
        self.assertFalse(((source % 5 == 4) | (target % 5 == 4)).any())

    def test_segment_reduction_uneven_degrees_and_backward(self):
        for device in ('cpu', 'cuda') if torch.cuda.is_available() else ('cpu',):
            degree = torch.tensor([2, 0, 3, 1, 0], device=device)
            values = torch.randn(6, 8, device=device, requires_grad=True)
            result = torch.segment_reduce(values, 'sum', lengths=degree)
            zero = values.new_zeros(8)
            reference = torch.stack((values[:2].sum(0), zero, values[2:5].sum(0), values[5], zero))
            torch.testing.assert_close(result, reference)
            grad, = torch.autograd.grad(result.square().sum(), values, retain_graph=True)
            expected, = torch.autograd.grad(reference.square().sum(), values)
            torch.testing.assert_close(grad, expected)
            torch.testing.assert_close(result, torch.segment_reduce(values, 'sum', lengths=degree), atol=0, rtol=0)

    def test_constraint_flags_tw_depot_and_open_return(self):
        closed, opened = self.fixture('VRPLTW'), self.fixture('OVRPLTW')
        opened['node'] = closed['node'].clone()
        g, o = build_relation_graph(closed), build_relation_graph(opened)
        torch.testing.assert_close(g['index'], o['index'])
        self.assertTrue((g['features'][:, 16] >= o['features'][:, 16]).all())
        depot_edges = (g['features'][:, 29:31] > .5).any(1)
        self.assertTrue((g['features'][depot_edges, 13:16] == 0).all())
        self.assertTrue((g['features'][depot_edges, 25] == 0).all())
        absent = build_relation_graph(self.fixture('TSP'))['features']
        self.assertTrue((absent[:, 8:17] == 0).all())
        self.assertTrue((absent[:, 23:28] == 0).all())
        self.assertTrue(torch.isfinite(g['features']).all())
        self.assertEqual(len(EDGE_FIELDS), g['features'].size(1))

    def test_ties_and_small_graphs(self):
        distances = torch.tensor([[[0., 1., 1., 2.]]])
        valid = torch.tensor([[[False, True, True, True]]])
        self.assertEqual(nearest(distances, valid, 1).sum().item(), 2)
        batch = self.fixture('TSP')
        batch['node_mask'][:, 1:] = False
        batch['n'][:] = 1
        graph = build_relation_graph(batch)
        self.assertEqual(graph['features'].shape, (0, len(EDGE_FIELDS)))
        self.assertTrue(torch.isfinite(self.model()(batch)['logits']).all())

    def test_edge_content_and_gradient_connection(self):
        model = self.model()
        batch = self.fixture()
        graph = build_relation_graph(batch)
        original = model(dict(batch, relation_graph=graph))['logits']
        altered = dict(graph, features=torch.zeros_like(graph['features']))
        self.assertGreater((model(dict(batch, relation_graph=altered))['logits'] - original).abs().max().item(), 1e-5)
        original.square().sum().backward()
        for module in (model.instance_encoder.coord_proj, model.instance_encoder.edge_proj,
                       model.instance_encoder.layers[0].edge_value, model.solver_encoder.feature_mlp,
                       model.query_layers[0], model.score_head):
            self.assertGreater(sum(float(p.grad.abs().sum()) for p in module.parameters() if p.grad is not None), 0)

    def test_two_dropout_paths_one_optimizer_step(self):
        for architecture in ('dual_stream', 'relation_query'):
            model = make_r42_model(dict(self.params, architecture=architecture, dropout=.1)).train()
            batch = self.fixture('TSP')
            first, second = model(batch)['logits'], model(batch)['logits']
            self.assertFalse(torch.equal(first, second))
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
            scaler = torch.amp.GradScaler('cuda', enabled=False)
            with patch.object(optimizer, 'step', wraps=optimizer.step) as step:
                values = update_batch(model, optimizer, scaler, batch, .35)
                self.assertEqual(step.call_count, 1)
                self.assertGreater(values['kl'], 0)
            mask = torch.ones_like(batch['costs'], dtype=torch.bool)
            mask[:, -1] = False
            with self.assertRaises(ValueError):
                update_batch(model, optimizer, scaler, dict(batch, solver_mask=mask), .35)
            out = model.eval()(batch)
            loss, parts = selector_loss(out, batch['costs'], None, fit_args('A'), winner=batch['ind'])
            torch.testing.assert_close(loss, .35 * parts['ce'] + .1 * parts['pair'] + .02 * parts['risk'])
            self.assertEqual(symmetric_kl(out['logits'], out['logits']).item(), 0)

    def test_kl_both_branches_and_mask(self):
        first = torch.tensor([[.2, -.3, float('-inf')]], requires_grad=True)
        second = torch.tensor([[-.1, .5, float('-inf')]], requires_grad=True)
        mask = torch.tensor([[True, True, False]])
        loss = symmetric_kl(first, second, mask)
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(loss.item(), 0)
        loss.backward()
        self.assertGreater(first.grad.abs().sum().item(), 0)
        self.assertGreater(second.grad.abs().sum().item(), 0)
        self.assertEqual(first.grad[0, -1].item(), 0)

    def test_schedule_full_coverage_and_interleaving(self):
        sizes = {p: 1000 for p in PROBLEMS}
        a = paired_schedule(sizes, epochs=2)
        b = paired_schedule(sizes, epochs=2)
        for epoch in range(2):
            for p in PROBLEMS:
                torch.testing.assert_close(a['permutations'][epoch][p], b['permutations'][epoch][p])
                np.testing.assert_array_equal(a['permutations'][epoch][p].sort().values, np.arange(1000))
            self.assertEqual(a['orders'][epoch], b['orders'][epoch])
            for start in range(0, len(a['orders'][epoch]), len(PROBLEMS)):
                self.assertEqual(set(a['orders'][epoch][start:start+len(PROBLEMS)]), set(PROBLEMS))
            self.assertEqual(len(a['orders'][epoch]), 8 * len(PROBLEMS))

    def test_cost_controller_prespecified_early_stop(self):
        control = CostController()
        self.assertEqual(control.observe(-1, 1), (False, False))
        halved = []
        for epoch in range(2, 16):
            halve, stop = control.observe(-1, epoch)
            if halve:
                halved.append(epoch)
            self.assertEqual(stop, epoch >= 15)
        self.assertTrue(halved)
        control.observe(-1.002, 16)
        self.assertEqual(control.bad_epochs, 0)


if __name__ == '__main__':
    unittest.main()
