"""Exact original-objective parity and shared-H checks on tiny CPU fixtures."""

import unittest

import torch

from .pairwise_objective import predicted_top3, single_objective
from .r42_experiment import symmetric_kl
from .r76_experiment import PART_KEYS, RDropObjective as OriginalRDropObjective
from .r94_model import DONOR_FIELDS
from .r94_objective import NodeFusionObjective, RDropObjective
from .test_r94_model import cached_batch, model_pair


class R94ObjectiveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_zero_residual_matches_original_losses_parts_gradients_and_rng_exactly(self):
        for arm in ('A', 'B'):
            model, original = model_pair(arm)
            with torch.no_grad():
                model.instance_encoder.nco_projection.weight.zero_()
            current, reference = RDropObjective(model).train(), OriginalRDropObjective(original).train()
            for problem in ('CVRP', 'VRPB', 'OVRPBLTW', 'TSP', 'ATSP'):
                with self.subTest(arm=arm, problem=problem):
                    batch = cached_batch(problem)
                    current.zero_grad(set_to_none=True)
                    reference.zero_grad(set_to_none=True)
                    torch.manual_seed(226)
                    expected_loss, expected_parts = reference(batch, .35)
                    expected_rng = torch.get_rng_state().clone()
                    expected_loss.backward()
                    torch.manual_seed(226)
                    loss, parts = current(batch, .35)
                    torch.testing.assert_close(loss, expected_loss, rtol=0, atol=0)
                    torch.testing.assert_close(parts, expected_parts, rtol=0, atol=0)
                    torch.testing.assert_close(torch.get_rng_state(), expected_rng, rtol=0, atol=0)
                    loss.backward()
                    actual = dict(model.named_parameters())
                    for name, parameter in original.named_parameters():
                        if parameter.grad is None:
                            self.assertIsNone(actual[name].grad)
                        else:
                            torch.testing.assert_close(actual[name].grad, parameter.grad, rtol=0, atol=0)
                    self.assertEqual(model.donor.calls, [])

    def test_nonzero_fusion_preserves_original_full_loss_recipe_and_dual_dropout_same_H(self):
        model, _ = model_pair()
        objective = RDropObjective(model).train()
        batch = cached_batch('OVRPTW', 3)
        batch['nco_nodes'].requires_grad_()
        inputs, outputs = [], []
        before = model.register_forward_pre_hook(lambda module, args: inputs.append(args[0]))
        after = model.register_forward_hook(lambda module, args, output: outputs.append(output))
        loss, parts = objective(batch, .27)
        before.remove()
        after.remove()
        self.assertEqual(len(inputs), 2)
        self.assertIs(inputs[0], inputs[1])
        self.assertIs(inputs[0]['nco_nodes'], batch['nco_nodes'])
        self.assertIs(inputs[1]['nco_nodes'], batch['nco_nodes'])
        first, second = outputs
        self.assertFalse(torch.equal(first['logits'], second['logits']))
        focus = predicted_top3(first['logits'], second['logits'], batch['pool_ids'], first['solver_mask'])
        loss1, parts1 = single_objective(first, batch, focus)
        loss2, parts2 = single_objective(second, batch, focus)
        kl = symmetric_kl(first['logits'], second['logits'], first['solver_mask'])
        expected_loss = .5 * (loss1 + loss2) + .27 * kl
        expected_parts = {key: .5 * (parts1[key] + parts2[key]) for key in parts1}
        expected_parts.update(loss=expected_loss, kl=kl)
        torch.testing.assert_close(loss, expected_loss, rtol=0, atol=0)
        torch.testing.assert_close(parts, torch.stack([expected_parts[key].detach() for key in PART_KEYS[:-1]]),
                                   rtol=0, atol=0)
        loss.backward()
        self.assertIsNone(batch['nco_nodes'].grad)
        self.assertGreater(model.instance_encoder.nco_projection.weight.grad.abs().sum().item(), 0.)
        self.assertTrue(all(p.grad is None for p in model.donor.parameters()))
        self.assertEqual(model.donor.calls, [])
        self.assertEqual(set(objective.state_dict()), {'selector.' + key for key in model.state_dict()})

    def test_uncached_raw_eval_extracts_once_before_both_passes_but_train_val_reject(self):
        model, _ = model_pair()
        objective = RDropObjective(model).eval()
        batch = {key: value for key, value in cached_batch().items() if key not in DONOR_FIELDS}
        seen = []
        handle = model.register_forward_pre_hook(lambda module, args: seen.append(args[0]['nco_nodes']))
        loss, parts = objective(batch, .35)
        handle.remove()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(parts).all())
        self.assertEqual(len(model.donor.calls), 1)
        self.assertEqual(len(seen), 2)
        self.assertIs(seen[0], seen[1])
        self.assertFalse(seen[0].requires_grad)
        for split in ('train', 'val', 'validation'):
            with self.subTest(split=split), self.assertRaises(ValueError):
                objective(dict(batch, split=split), .35)
        objective.train()
        self.assertFalse(model.donor.training)
        with self.assertRaises(ValueError):
            objective(batch, .35)
        self.assertEqual(len(model.donor.calls), 1)

    def test_empty_rank_is_original_stateless_zero_without_donor_forward(self):
        model, original = model_pair()
        objective = RDropObjective(model)
        loss, parts = objective(None, .35)
        expected_loss, expected_parts = OriginalRDropObjective(original)(None, .35)
        torch.testing.assert_close(loss, expected_loss, rtol=0, atol=0)
        torch.testing.assert_close(parts, expected_parts, rtol=0, atol=0)
        self.assertEqual(parts.shape, (len(PART_KEYS) - 1,))
        loss.backward()
        self.assertIs(NodeFusionObjective, RDropObjective)
        first, *rest = model.score_head.parameters()
        torch.testing.assert_close(first.grad, torch.zeros_like(first), rtol=0, atol=0)
        self.assertTrue(all(p.grad is None for p in rest))
        self.assertIsNone(model.instance_encoder.nco_projection.weight.grad)
        self.assertEqual(model.donor.calls, [])


if __name__ == '__main__':
    unittest.main()
