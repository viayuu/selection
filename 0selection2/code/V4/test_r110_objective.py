"""Bounded CPU algebra/gradient checks for whole-legal-pool raw-utility CE."""

import unittest
from unittest.mock import patch

import torch
from torch import nn
import torch.nn.functional as F

from .pairwise_objective import LABEL_KEYS
from .pairwise_selector import maximin_logits
from .r76_experiment import PART_KEYS
from .r110_objective import ARMS, MODEL_ARM_MAP, MethodConditionedCEObjective, make_objective


class ToySelector(nn.Module):
    def __init__(self, arm=ARMS[0], dropout=.1):
        super().__init__()
        self.r109_arm = MODEL_ARM_MAP[arm]
        self.scores = nn.Parameter(torch.tensor([[.8, -.7, 1.4, 2.6], [-.2, 1.1, .4, -.5]]))
        self.dropout = nn.Dropout(dropout)
        self.outputs = []

    def forward(self, batch):
        if set(batch).intersection(LABEL_KEYS):
            raise AssertionError('Supervision reached the selector')
        utility = self.dropout(self.scores)
        mask = batch['solver_mask']
        margins = utility[:, :, None] - utility[:, None, :]
        output = dict(utility=utility, solver_mask=mask, pair_margin=margins,
                      logits=maximin_logits(margins, mask.bool()))
        self.outputs.append(output)
        return output


def fixture(mask=None):
    if mask is None:
        mask = torch.ones(2, 4, dtype=torch.bool)
    return dict(ind=torch.tensor([1, 2]), solver_mask=mask,
                **{key: object() for key in LABEL_KEYS if key != 'ind'})


class R110ObjectiveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def setUp(self):
        guard = patch('torch.cuda._lazy_init', side_effect=AssertionError('CPU tests must not initialize CUDA'))
        guard.start()
        self.addCleanup(guard.stop)

    def test_single_dropout_forward_raw_ce_gradient_and_zero_other_parts(self):
        for arm in ARMS:
            objective = make_objective(ToySelector(arm), arm).train()
            batch = fixture()
            loss, parts = objective(batch, consistency_weight=float('nan'))
            self.assertEqual(len(objective.selector.outputs), 1)
            output = objective.selector.outputs[0]
            expected = F.cross_entropy(output['utility'], batch['ind'])
            torch.testing.assert_close(loss, expected, rtol=0, atol=0)
            self.assertNotEqual(loss.item(), F.cross_entropy(output['logits'], batch['ind']).item())
            self.assertEqual(parts.shape, (len(PART_KEYS) - 1,))
            torch.testing.assert_close(parts[:2], expected.detach().expand(2), rtol=0, atol=0)
            self.assertTrue((parts[2:] == 0).all())
            self.assertFalse(parts.requires_grad)
            gradient = torch.autograd.grad(loss, output['utility'], retain_graph=True)[0]
            expected_gradient = (output['utility'].softmax(-1) - F.one_hot(batch['ind'], 4)) / 2
            torch.testing.assert_close(gradient, expected_gradient, rtol=2e-6, atol=2e-8)
            self.assertTrue((gradient[:, 3] != 0).all(), 'Whole-pool CE must differentiate nonwinner candidates')
            loss.backward()
            self.assertTrue(torch.isfinite(objective.selector.scores.grad).all())

    def test_heterogeneous_masks_ignore_illegal_nan_inf_with_zero_gradients(self):
        masks = (
            torch.tensor([[True, True, False, True], [False, True, True, False]]),
            torch.tensor([[False, True, False, False], [True, False, True, True]]),
        )
        for arm in ARMS:
            for mask in masks:
                with self.subTest(arm=arm, mask=mask.tolist()):
                    selector = ToySelector(arm, dropout=0.)
                    with torch.no_grad():
                        selector.scores[~mask] = float('nan')
                        selector.scores[0, 0] = float('inf') if not mask[0, 0] else .8
                    loss, parts = make_objective(selector, arm)(fixture(mask))
                    self.assertTrue(torch.isfinite(loss))
                    masked = selector.scores.detach().masked_fill(~mask, -torch.inf)
                    expected = (masked.softmax(-1) - F.one_hot(torch.tensor([1, 2]), 4)) / 2
                    loss.backward()
                    torch.testing.assert_close(selector.scores.grad, expected, rtol=2e-6, atol=2e-8)
                    self.assertTrue((selector.scores.grad[~mask] == 0).all())
                    self.assertTrue((parts[2:] == 0).all())

    def test_unweighted_mean_and_common_shift_invariance(self):
        for arm in ARMS:
            selector = ToySelector(arm, dropout=0.)
            objective = make_objective(selector, arm)
            batch = fixture()
            first, _ = objective(batch)
            per_row = F.cross_entropy(selector.scores, batch['ind'], reduction='none')
            torch.testing.assert_close(first, per_row.mean(), rtol=0, atol=0)
            first_grad = torch.autograd.grad(first, selector.scores)[0]
            with torch.no_grad():
                selector.scores.add_(torch.tensor([[3.], [-2.]]))
            second, _ = objective(batch)
            second_grad = torch.autograd.grad(second, selector.scores)[0]
            torch.testing.assert_close(second, first, rtol=1e-6, atol=2e-7)
            torch.testing.assert_close(second_grad, first_grad, rtol=2e-6, atol=2e-8)

    def test_rejects_invalid_winner_pool_and_arm(self):
        with self.assertRaises(ValueError):
            make_objective(ToySelector(), 'not_an_arm')
        with self.assertRaises(ValueError):
            make_objective(ToySelector(ARMS[0]), ARMS[1])
        objective = make_objective(ToySelector(dropout=0.), ARMS[0])
        for bad in (dict(ind=torch.tensor([-1, 2])), dict(ind=torch.tensor([1, 4])),
                    dict(ind=torch.tensor([1., 2.])), dict(ind=torch.tensor([[1], [2]])),
                    dict(solver_mask=torch.zeros(2, 4, dtype=torch.bool)),
                    dict(solver_mask=torch.ones(2, 4)),
                    dict(solver_mask=torch.tensor([[True, False, True, True], [True] * 4]))):
            with self.subTest(bad=list(bad)), self.assertRaises(ValueError):
                objective(dict(fixture(), **bad))

    def test_stateless_none_batch_does_not_forward(self):
        objective = MethodConditionedCEObjective(ToySelector(), ARMS[0])
        loss, parts = objective(None, consistency_weight=100.)
        self.assertEqual(loss.item(), 0.)
        self.assertTrue(loss.requires_grad)
        self.assertTrue((parts == 0).all())
        self.assertEqual(objective.selector.outputs, [])
        self.assertEqual(list(objective.named_buffers()), [])
        self.assertTrue(all(key.startswith('selector.') for key in objective.state_dict()))
        loss.backward()
        self.assertTrue((objective.selector.scores.grad == 0).all())


if __name__ == '__main__':
    unittest.main()
