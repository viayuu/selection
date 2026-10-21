"""Focused CPU tests; no real data preparation, solver execution or training jobs."""

import copy
import unittest
from unittest.mock import patch

import numpy as np
import torch
import torch.nn.functional as F

from code.unified_selector.registry import PROBLEMS
from .pairwise_objective import comparison_targets, inputs_only
from .pairwise_selector import maximin_logits
from .r76_experiment import RDropObjective
from .r85_candidate_selector import CANDIDATE_AUX, make_candidate_model, subset_targets
from .solver_code_encoder import make_r45_model
from .test_r76 import fixture_batch, fixture_model


class CandidateSelectorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        self.base, self.params = fixture_model(dropout=.1)
        torch.manual_seed(2)
        self.model = make_candidate_model(self.params)

    def test_exact_parameters_state_inference_and_strict_reconstruction(self):
        self.assertEqual(set(dict(self.base.named_parameters())), set(dict(self.model.named_parameters())))
        self.assertEqual(set(self.base.state_dict()), set(self.model.state_dict()))
        for name, tensor in self.base.state_dict().items():
            torch.testing.assert_close(tensor, self.model.state_dict()[name], rtol=0, atol=0)
        self.assertEqual(self.model.params['selection_head'], 'original_score_mlp_maximin')
        self.assertEqual(self.model.params['candidate_aux'], CANDIDATE_AUX)
        restored = make_candidate_model(self.model.params, self.base.state_dict()).eval()
        original = make_r45_model(self.model.params, state_dict=restored.state_dict()).eval()
        for problem in PROBLEMS:
            with self.subTest(problem=problem), torch.no_grad():
                batch, _ = fixture_batch(problem, 2)
                rng = torch.get_rng_state().clone()
                expected, actual = original(batch), restored(batch, aux_solver_mask=torch.tensor(False))
                self.assertEqual(set(actual), set(expected))
                for key in expected:
                    torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
                torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)
        bad_state = dict(self.base.state_dict())
        bad_state.pop('score_head.3.weight')
        with self.assertRaises(RuntimeError):
            make_candidate_model(self.params, bad_state)

    def test_noaux_original_rdrop_loss_gradients_and_rng_are_exact(self):
        batch, _ = fixture_batch('TSP', 4)
        records = []
        for model in (self.base.train(), self.model.train()):
            model.zero_grad(set_to_none=True)
            torch.manual_seed(923)
            loss, parts = RDropObjective(model)(batch, .35)
            loss.backward()
            records.append((loss.detach(), parts, torch.get_rng_state(),
                            {name: value.grad.clone() for name, value in model.named_parameters()
                             if value.grad is not None}))
        for first, second in zip(records[0][:3], records[1][:3]):
            torch.testing.assert_close(first, second, rtol=0, atol=0)
        self.assertEqual(set(records[0][3]), set(records[1][3]))
        for name in records[0][3]:
            torch.testing.assert_close(records[0][3][name], records[1][3][name], rtol=0, atol=0)

    def test_aux_reuses_initial_encodings_and_preserves_full_dropout_prediction(self):
        batch, _ = fixture_batch('TSP', 3)
        subset = torch.arange(len(batch['pool_ids'])) < 3
        self.base.train()
        self.model.train()
        torch.manual_seed(726)
        expected = self.base(inputs_only(batch))
        initial = []
        hook = self.model.joint_layers[0].register_forward_pre_hook(
            lambda module, arguments: initial.append(arguments))
        with patch.object(self.model.instance_encoder, 'forward', wraps=self.model.instance_encoder.forward) as enc, \
                patch.object(self.model.solver_encoder, 'forward', wraps=self.model.solver_encoder.forward) as sol:
            torch.manual_seed(726)
            actual = self.model(batch, aux_solver_mask=subset)
            self.assertEqual((enc.call_count, sol.call_count), (1, 1))
        hook.remove()
        self.assertEqual(len(initial), 2)
        self.assertIs(initial[0][0], initial[1][0])
        torch.testing.assert_close(initial[0][1][:, subset], initial[1][1][:, subset], rtol=0, atol=0)
        self.assertEqual(initial[1][1][:, ~subset].count_nonzero().item(), 0)
        for key in expected:
            torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
        self.assertEqual(set(actual) - set(expected), {'aux_subset_logits', 'aux_subset_pair_margin',
                                                     'aux_subset_solver_mask', 'aux_subset_utility'})
        changed = copy.deepcopy(batch)
        for key in ('costs', 'cost_regret', 'cmp_target'):
            changed[key].fill_(float('nan'))
        changed['ind'].fill_(-100)
        changed['cmp_valid'].fill_(False)
        changed['cmp_true3'].fill_(False)
        torch.manual_seed(726)
        poisoned = self.model(changed, aux_solver_mask=subset)
        for key in actual:
            torch.testing.assert_close(actual[key], poisoned[key], rtol=0, atol=0)

    def test_subset_reencodes_shared_joint_stream_and_backpropagates(self):
        params = dict(self.params, dropout=0.)
        model = make_candidate_model(params).train()
        original = make_r45_model(params, state_dict=model.state_dict()).eval()
        for problem in ('TSP', 'ATSP'):
            with self.subTest(problem=problem):
                batch, _ = fixture_batch(problem, 3)
                k = len(batch['pool_ids'])
                mask = torch.zeros(3, k, dtype=torch.bool)
                mask[0, :2] = True
                mask[1, 1:3] = True
                mask[2, -2:] = True
                out = model(batch, aux_solver_mask=mask)
                expected = original(dict(batch, solver_mask=mask))
                for key, value in expected.items():
                    torch.testing.assert_close(out['aux_subset_' + key], value, rtol=0, atol=0)
                torch.testing.assert_close(out['aux_subset_logits'],
                                           maximin_logits(out['aux_subset_pair_margin'], mask))
                self.assertTrue(torch.isneginf(out['aux_subset_logits'][~mask]).all())
                self.assertTrue((out['aux_subset_logits'].softmax(-1)[~mask] == 0).all())
                targets = subset_targets(batch, mask)
                model.zero_grad(set_to_none=True)
                F.cross_entropy(out['aux_subset_logits'], targets['ind']).backward()
                branch = model.instance_encoder.matrix_proj if problem == 'ATSP' else model.instance_encoder.coord_proj
                for module in (branch, model.solver_encoder, *model.joint_layers, model.pool,
                               model.decoder, model.score_head):
                    gradients = [p.grad for p in module.parameters() if p.grad is not None]
                    self.assertTrue(gradients)
                    self.assertTrue(all(torch.isfinite(g).all() for g in gradients))
                    self.assertGreater(sum(g.abs().sum().item() for g in gradients), 0.)

    def test_fp64_subsets_fullwinner_removed_ties_and_original_slots(self):
        ids = torch.tensor([9, 3, 7, 1, 5])
        costs = np.array([[1., 1. + 1e-10, 1. + 2e-10, 1. + 3e-10, 1. + 4e-10],
                          [2., 2., 2., 2., 2.], [1., 2., 2., 3., 3.]], dtype=np.float64)
        masks = torch.tensor([[False, True, True, True, False], [True, True, True, False, True],
                              [False, True, True, True, True]])
        batch = dict(pool_ids=ids, **comparison_targets(costs))
        # Every near-tie cost is identical after FP32 conversion, and the full winner is absent.
        batch['costs'] = torch.from_numpy(costs.astype(np.float32))
        self.assertEqual(batch['costs'][0].unique().numel(), 1)
        batch['ind'] = torch.zeros(3, dtype=torch.long)
        result = subset_targets(batch, masks)
        for row in range(3):
            expected = sorted(range(5), key=lambda slot: (costs[row, slot], int(ids[slot])))
            retained = [slot for slot in expected if masks[row, slot]]
            self.assertEqual(result['full_ranking'][row].tolist(), expected)
            self.assertEqual(result['subset_ranking'][row, :len(retained)].tolist(), retained)
            self.assertEqual(result['ind'][row].item(), retained[0])
        self.assertEqual(result['ind'].tolist(), [1, 1, 1])
        expected = comparison_targets(costs, masks.numpy())
        for key in ('cmp_target', 'cmp_valid', 'cmp_true3'):
            torch.testing.assert_close(result[key], expected[key], rtol=0, atol=0)
        batch.pop('costs')
        batch.pop('ind')
        torch.testing.assert_close(subset_targets(batch, masks)['ind'], result['ind'], rtol=0, atol=0)

    def test_global_id_permutation_and_masked_padding(self):
        ids = torch.tensor([9, 3, 7, 1, -1, 100000])
        costs = np.array([[2., 2., 1., 3., np.nan, np.inf],
                          [4., 4., 4., 4., np.nan, np.inf]], dtype=np.float64)
        legal = torch.tensor([[True, True, True, True, False, False]]).expand(2, -1)
        subset = torch.tensor([True, True, False, True, False, False])
        batch = dict(pool_ids=ids, solver_mask=legal, **comparison_targets(costs, legal.numpy()))
        result = subset_targets(batch, subset)
        order = torch.tensor([5, 2, 3, 0, 4, 1])
        permuted = dict(pool_ids=ids[order][None].expand(2, -1), solver_mask=legal[:, order],
                        **comparison_targets(costs[:, order], legal[:, order].numpy()))
        actual = subset_targets(permuted, subset[order][None])
        torch.testing.assert_close(ids[order][actual['ind']], ids[result['ind']], rtol=0, atol=0)
        torch.testing.assert_close(ids[order][actual['full_ranking'][:, :4]],
                                   ids[result['full_ranking'][:, :4]], rtol=0, atol=0)
        self.assertFalse(result['solver_mask'][:, -2:].any())
        torch.testing.assert_close(result['ind'], torch.tensor([1, 3]), rtol=0, atol=0)
        plain = dict(pool_ids=ids[:4], **comparison_targets(costs[:, :4]))
        torch.testing.assert_close(subset_targets(plain, subset[:4])['ind'], result['ind'], rtol=0, atol=0)

    def test_bad_training_masks_are_rejected_and_input_filter_excludes_labels(self):
        batch, _ = fixture_batch('TSP', 2)
        k = len(batch['pool_ids'])
        self.model.train()
        for bad in (torch.ones(k, dtype=torch.bool), torch.zeros(k, dtype=torch.bool),
                    torch.arange(k) == 0, torch.ones(k), torch.ones(2, k + 1, dtype=torch.bool)):
            with self.subTest(shape=tuple(bad.shape)), self.assertRaises(ValueError):
                self.model(batch, aux_solver_mask=bad)
        self.assertFalse({'costs', 'ind', 'cmp_target', 'cmp_valid', 'cmp_true3', 'cost_regret'} &
                         set(inputs_only(batch)))
        with self.assertRaises(ValueError):
            subset_targets(batch, torch.zeros(k, dtype=torch.bool))
        with self.assertRaises(ValueError):
            make_candidate_model(dict(self.params, selection_head='direct_attention'))


if __name__ == '__main__':
    unittest.main()
