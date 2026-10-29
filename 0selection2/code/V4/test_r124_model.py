"""Small CPU contracts for R124, without real held-out inputs or solver calls."""

import copy
import unittest

import torch
from torch import nn

from code.unified_selector.registry import PROBLEMS
from .r124_model import (Top2Reranker, FrozenTop2Selector, query_inputs, candidate_ids,
                         top2_indices, replace_top2, pair_objective)
from .test_r76 import fixture_batch, fixture_model


def fixture(problem='TSP', size=3):
    _, params = fixture_model(dropout=0.)
    model = Top2Reranker(**params)
    batch, costs = fixture_batch(problem, size)
    batch['costs'] = torch.from_numpy(costs)
    batch['stage1_scores'] = torch.arange(len(batch['pool_ids']), dtype=torch.float32)[None].repeat(size, 1)
    return model, batch


class PairModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_all18_actual_pairs_finite_without_labels(self):
        for problem in PROBLEMS:
            model, batch = fixture(problem)
            model.eval()
            result = model(query_inputs(batch), batch['stage1_scores'])
            self.assertTrue(torch.isfinite(result['logits']).all())
            self.assertEqual(result['pair_indices'].shape, (3, 2))
            self.assertEqual(result['logits'].shape, batch['costs'].shape)

    def test_both_heads_antisymmetry_and_pair_role_consistency(self):
        for problem in ('TSP', 'ATSP', 'VRPB'):
            model, batch = fixture(problem)
            model.eval()
            first = model.compare(batch, batch['stage1_scores'])
            second = model.compare(batch, batch['stage1_scores'], first['pair_indices'].flip(1))
            for name in ('difference', 'pair_quality'):
                torch.testing.assert_close(first[name], -second[name], atol=0, rtol=0)
            torch.testing.assert_close(first['pair_logits'].softmax(1),
                                      second['pair_logits'].softmax(1).flip(1), atol=2e-6, rtol=2e-5)
            ids, mask = candidate_ids(batch, batch['stage1_scores'])
            a = replace_top2(batch['stage1_scores'], first['pair_indices'], first['difference'], ids, mask)
            b = replace_top2(batch['stage1_scores'], second['pair_indices'], second['difference'], ids, mask)
            torch.testing.assert_close(a, b, atol=2e-6, rtol=2e-5)
            with self.assertRaisesRegex(ValueError, 'actual first-stage Top2'):
                model.compare(batch, batch['stage1_scores'], torch.zeros_like(first['pair_indices']))

    def test_score_column_permutation_and_consistent_method_id_relabel(self):
        model, batch = fixture()
        model.eval()
        scores = batch['stage1_scores']
        first = model(batch, scores)
        order = torch.arange(scores.shape[1] - 1, -1, -1)
        permuted = dict(batch, pool_ids=batch['pool_ids'][order])
        second = model(permuted, scores[:, order])
        torch.testing.assert_close(second['logits'], first['logits'][:, order], atol=2e-6, rtol=2e-5)
        # Numeric method IDs have no meaning beyond their attached embedding/features.
        renamed = copy.deepcopy(model)
        mapping = torch.arange(len(renamed.solver_encoder.solver_emb.weight) - 1, -1, -1)
        with torch.no_grad():
            renamed.solver_encoder.solver_emb.weight[mapping] = model.solver_encoder.solver_emb.weight
            renamed.solver_encoder.solver_features[mapping] = model.solver_encoder.solver_features
        third = renamed(dict(batch, pool_ids=mapping[batch['pool_ids']]), scores)
        torch.testing.assert_close(third['logits'], first['logits'], atol=2e-6, rtol=2e-5)

    def test_strict_top2_lift_remaining_order_and_ties(self):
        scores = torch.tensor([[4., 3., 2., 1.], [1e20, 1e20, 1e20, 0.], [-3., -3., -4., -5.]])
        ids = torch.tensor([[7, 2, 5, 9]]).expand(3, -1)
        mask = torch.ones_like(scores, dtype=torch.bool)
        pair = top2_indices(scores, ids, mask)
        for d in (torch.tensor([-1000., 1000., 0.]), torch.tensor([1e-20, -1e-20, 0.])):
            result = replace_top2(scores, pair, d, ids, mask)
            other = mask.clone().scatter(1, pair, False)
            torch.testing.assert_close(result[other], scores[other].to(result.dtype), rtol=0, atol=0)
            self.assertTrue((result.gather(1, pair).amin(1) > result.masked_fill(~other, -torch.inf).amax(1)).all())
            top = top2_indices(result, ids, mask)
            self.assertTrue(torch.equal(top.sort(1).values, pair.sort(1).values))
            self.assertEqual(ids[2, top[2, 0]].item(), 2)
        with self.assertRaises(ValueError):
            candidate_ids(dict(pool_ids=ids), scores[:, :1])

    def test_labels_isolated_and_first_stage_never_updates(self):
        model, batch = fixture()
        model.eval()
        scores = batch['stage1_scores']
        before = model(batch, scores)['logits']
        poisoned = dict(batch, costs=torch.full_like(batch['costs'], float('nan')),
            ind=torch.full_like(batch['ind'], -1), winner=torch.zeros(3), native_label=torch.zeros(3),
            source_id=torch.zeros(3), unknown_query_result=torch.zeros(3))
        torch.testing.assert_close(before, model(poisoned, scores)['logits'], rtol=0, atol=0)

        class Frozen(nn.Module):
            def __init__(self):
                super().__init__()
                self.scale = nn.Parameter(torch.tensor(1.))
                self.keys = None

            def forward(self, inputs):
                self.keys = set(inputs)
                return dict(logits=scores * self.scale)

        frozen = Frozen()
        wrapper = FrozenTop2Selector(model, frozen).train()
        output = wrapper(poisoned)
        output['pair_logits'].sum().backward()
        self.assertFalse(frozen.training)
        self.assertIsNone(frozen.scale.grad)
        self.assertFalse(frozen.scale.requires_grad)
        self.assertEqual(frozen.keys, set(query_inputs(batch)))

    def test_supervision_outside_top2_quality_only_and_native_ties(self):
        logits = torch.zeros(3, 2, requires_grad=True)
        quality = torch.zeros(3, requires_grad=True)
        output = dict(pair_indices=torch.tensor([[0, 1], [0, 1], [0, 1]]), pair_logits=logits,
            pair_quality=quality, solver_mask=torch.ones(3, 3, dtype=torch.bool))
        costs = torch.tensor([[1., 1., 2.], [1.2, 1.1, 1.], [1. + 1e-10, 1., 2.]], dtype=torch.float64)
        loss, parts = pair_objective(output, costs, torch.tensor([1, 2, 1]))
        loss.backward()
        self.assertEqual(parts['covered'].item(), 2)
        self.assertGreater(logits.grad[0, 0].item(), 0.)
        torch.testing.assert_close(logits.grad[1], torch.zeros(2), rtol=0, atol=0)
        self.assertEqual(quality.grad[0].item(), 0.)
        self.assertNotEqual(quality.grad[1].item(), 0.)
        self.assertGreater(logits.grad[2, 0].item(), 0.)
        with self.assertRaisesRegex(ValueError, 'FP64'):
            pair_objective(output, costs.float(), torch.tensor([1, 2, 1]))

    def test_native_and_quality_gradients_reach_new_structural_path(self):
        for problem in ('TSP', 'ATSP'):
            model, batch = fixture(problem)
            batch['stage1_scores'] = -batch['costs'].float()
            output = model(batch, batch['stage1_scores'])
            loss, _ = pair_objective(output, batch['costs'], batch['ind'])
            loss.backward()
            projection = model.instance_encoder.matrix_proj if problem == 'ATSP' else model.instance_encoder.coord_proj
            for module in (projection, model.solver_encoder.feature_mlp, model.stage_score,
                           *model.joint_layers, model.pair_readout, model.score_head, model.quality_head):
                grads = [p.grad for p in module.parameters() if p.grad is not None]
                self.assertTrue(grads)
                self.assertTrue(all(torch.isfinite(g).all() for g in grads))
                self.assertGreater(sum(g.abs().sum().item() for g in grads), 0.)


if __name__ == '__main__':
    unittest.main()
