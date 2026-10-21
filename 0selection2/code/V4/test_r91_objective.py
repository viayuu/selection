"""Small CPU checks for the two probability links, not a GPU preflight."""

import copy
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from .pairwise_objective import LABEL_KEYS, comparison_targets, predicted_top3
from .pairwise_selector import maximin_logits
from .r42_experiment import symmetric_kl
from .r76_experiment import PART_KEYS, RDropObjective, slice_batch
from .r91_objective import ARMS, ProbabilityLinkObjective, single_objective, utility_logits


class ToySelector(nn.Module):
    def __init__(self, dropout=.3):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(3, 6), nn.GELU(), nn.Dropout(dropout))
        self.score_head = nn.Linear(6, 4)
        self.outputs = []

    def forward(self, batch):
        if any(key in batch for key in LABEL_KEYS):
            raise AssertionError('Supervision reached model inputs')
        u = self.score_head(self.encoder(batch['x'])).float()
        mask = batch['solver_mask']
        margins = (u[:, :, None] - u[:, None, :]).masked_fill(
            ~(mask[:, :, None] & mask[:, None, :]), 0.)
        output = dict(utility=u, pair_margin=margins, solver_mask=mask,
                      logits=maximin_logits(margins, mask))
        self.outputs.append(output)
        return output


def fixture():
    costs = np.asarray([[1., 1. + 1e-10, 1.6, 1.7], [1.3, 1., 1.2, 1.1],
                        [1.4, 1.3, 1., 1.2], [1., 1.2, 1.5, 1.6],
                        [1.2, 1., 1.3, 1.4]], dtype=np.float64)
    mask = np.ones_like(costs, dtype=bool)
    mask[-1, -1] = False
    return dict(x=torch.randn(5, 3), solver_mask=torch.from_numpy(mask),
                pool_ids=torch.tensor([4, 1, 9, 6]), ind=torch.tensor([1, 1, 2, 0, 1]),
                costs=torch.from_numpy(costs), **comparison_targets(costs, mask))


class ProbabilityLinkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(2)

    def test_a_changes_only_ce_and_preserves_rng_comparison_risk_kl(self):
        batch = fixture()
        old = RDropObjective(ToySelector()).train()
        new = ProbabilityLinkObjective(copy.deepcopy(old.selector), ARMS[0]).train()
        rng = torch.get_rng_state()
        old_loss, old_parts = old(batch, .175)
        after_old = torch.get_rng_state()
        torch.set_rng_state(rng)
        loss, parts = new(batch, .175)
        self.assertTrue(torch.equal(torch.get_rng_state(), after_old))
        expected_ce = .5 * sum(F.cross_entropy(utility_logits(o), batch['ind'])
                                for o in new.selector.outputs)
        torch.testing.assert_close(loss, old_loss + .35 * (expected_ce - old_parts[1]))
        torch.testing.assert_close(parts[1], expected_ce.detach())
        torch.testing.assert_close(parts[2:], old_parts[2:], rtol=0, atol=0)
        for left, right in zip(old.selector.outputs, new.selector.outputs):
            for key in ('logits', 'utility', 'pair_margin', 'solver_mask'):
                torch.testing.assert_close(left[key], right[key], rtol=0, atol=0)
        self.assertNotEqual(parts[1].item(), old_parts[1].item())
        self.assertEqual(batch['ind'][0].item(), 1)  # Native label, not regenerated argmin.

    def test_b_formula_and_original_focus_for_both_forwards(self):
        batch = fixture()
        objective = ProbabilityLinkObjective(ToySelector(), ARMS[1]).train()
        with patch('code.V4.r91_objective.predicted_top3', wraps=predicted_top3) as focus_call:
            loss, parts = objective(batch, .35)
        first, second = objective.selector.outputs
        self.assertEqual(len(objective.selector.outputs), 2)
        torch.testing.assert_close(focus_call.call_args.args[0], first['logits'])
        torch.testing.assert_close(focus_call.call_args.args[1], second['logits'])
        focus = predicted_top3(first['logits'], second['logits'], batch['pool_ids'], first['solver_mask'])
        kl = symmetric_kl(utility_logits(first), utility_logits(second), first['solver_mask'])
        values = [single_objective(out, batch, focus, ARMS[1]) for out in (first, second)]
        expected = .5 * (values[0][0] + values[1][0]) + .35 * kl
        torch.testing.assert_close(loss, expected, rtol=0, atol=0)
        torch.testing.assert_close(parts[PART_KEYS.index('kl')], kl.detach())
        expected_risk = .5 * sum((utility_logits(out).softmax(-1) * batch['cost_regret'])
                                .sum(-1).clamp_max(1.).mean() / .01 for out in (first, second))
        torch.testing.assert_close(parts[PART_KEYS.index('risk')], expected_risk.detach())
        self.assertTrue(torch.isfinite(parts).all())
        loss.backward()
        for block in (objective.selector.encoder, objective.selector.score_head):
            grads = [p.grad for p in block.parameters()]
            self.assertTrue(all(g is not None and torch.isfinite(g).all() for g in grads))
            self.assertGreater(sum(g.abs().sum().item() for g in grads), 0.)

    def test_full_batch_and_sample_weighted_microbatch_gradients_agree(self):
        batch = fixture()
        for arm in ARMS:
            full = ProbabilityLinkObjective(ToySelector(dropout=0), arm).train()
            micro = copy.deepcopy(full)
            loss, parts = full(batch, .35)
            loss.backward()
            accumulated = torch.zeros_like(parts)
            for start, stop in ((0, 2), (2, 5)):
                value, metrics = micro(slice_batch(batch, start, stop), .35)
                weight = (stop - start) / 5
                (value * weight).backward()
                accumulated += metrics * weight
            torch.testing.assert_close(accumulated, parts, rtol=2e-6, atol=2e-7)
            for left, right in zip(full.parameters(), micro.parameters()):
                torch.testing.assert_close(left.grad, right.grad, rtol=3e-5, atol=2e-7)

    def test_legal_mask_singleton_and_idle_forward(self):
        scores = torch.tensor([[2., -1., float('nan')], [3., float('inf'), float('nan')]],
                              requires_grad=True)
        mask = torch.tensor([[True, True, False], [True, False, False]])
        output = dict(utility=scores, solver_mask=mask)
        logits = utility_logits(output)
        loss = F.cross_entropy(logits, torch.tensor([0, 0]))
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(torch.isfinite(scores.grad).all())
        self.assertTrue((scores.grad[~mask] == 0).all())
        objective = ProbabilityLinkObjective(ToySelector(), ARMS[0])
        loss, parts = objective(None, .35)
        self.assertEqual(parts.shape, (len(PART_KEYS) - 1,))
        self.assertEqual(loss.item(), 0.)
        self.assertFalse(parts.requires_grad)
        loss.backward()

    def test_unknown_arm_rejected(self):
        with self.assertRaises(ValueError):
            ProbabilityLinkObjective(ToySelector(), 'original')


if __name__ == '__main__':
    unittest.main()
