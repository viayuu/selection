"""Small offline CPU checks for R81B; no datasets, GPU probes or training launch."""

import copy
import math
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from .method_auc_objective import (
    AUC_LABEL_KEYS, AUC_NEG_KEY, AUC_POS_KEY, CachedAUCObjective, auc_inputs_only,
    method_auc_loss, method_auc_targets, method_competitive_scores,
    train_eligible_methods, utility_update,
)
from .pairwise_objective import LABEL_KEYS, comparison_targets
from .pairwise_selector import ScoreDifferenceSelector
from .r76_experiment import (
    PART_KEYS, RDropObjective, capture_rank_rng, restore_rank_rng, slice_batch,
)


class ToySelector(ScoreDifferenceSelector):
    def __init__(self):
        nn.Module.__init__(self)
        self.instance_encoder = nn.Sequential(nn.Linear(3, 4), nn.Tanh(), nn.Dropout(.25))
        self.context = nn.Linear(4, 4)
        self.method_embedding = nn.Embedding(12, 4)
        self.score_head = nn.Sequential(nn.Linear(12, 7), nn.GELU(), nn.Dropout(.4), nn.Linear(7, 1))

    def encode_pairs(self, batch):
        if any(key in batch for key in LABEL_KEYS + AUC_LABEL_KEYS):
            raise AssertionError('Supervision reached the model, including in the no_grad pass')
        h = self.instance_encoder(batch['x'])
        ids = batch['pool_ids']
        if ids.ndim == 1:
            ids = ids[None].expand(len(h), -1)
        mask = batch['solver_mask']
        solvers = self.method_embedding(ids.masked_fill(~mask, 0))
        return torch.cat((h[:, None].expand_as(solvers), self.context(h)[:, None].expand_as(solvers),
                          solvers), -1), mask


class CountingSGD(torch.optim.SGD):
    def __init__(self, parameters):
        super().__init__(parameters, lr=1e-4)
        self.steps = 0

    def step(self, *args, **kwargs):
        self.steps += 1
        return super().step(*args, **kwargs)


def fixture_batch(full_logical=False):
    if full_logical:
        costs = np.full((1024, 3), 3., dtype=np.float64)
        for method in range(3):
            costs[256 * method:256 * (method + 1), method] = 1.
        costs[768:, :2] = 2.  # Tied minima; method 2 remains a strict loser.
        native = torch.from_numpy(costs.argmin(1))
    else:
        costs = np.asarray([[1., 1. + 1e-10, 1.4], [2.1, 2., 2.4], [3.2, 3.6, 3.],
                            [1.1, 1., 1.3], [4., 4.4, 4.8], [2.2, 2.3, 2.], [3., 2.8, 2.9]],
                           dtype=np.float64)
        # Deliberately retain a native winner different from the FP64 minimum on row 0.
        native = torch.tensor([1, 1, 2, 1, 0, 2, 1])
    targets = method_auc_targets(costs)
    batch = dict(x=torch.randn(len(costs), 3), solver_mask=torch.ones(len(costs), 3, dtype=torch.bool),
                 pool_ids=torch.tensor([4, 1, 9]), costs=torch.from_numpy(costs.astype(np.float32)),
                 ind=native, **comparison_targets(costs), **targets)
    return batch, train_eligible_methods(targets[AUC_POS_KEY], targets[AUC_NEG_KEY])


class R81AUCTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(2)

    def test_fp64_unique_ties_strict_losers_and_direction(self):
        costs = np.asarray([[1., 1. + 1e-10, 3., np.nan], [1., 1., 2., np.nan],
                            [2., 1., 1., np.nan], [3., 2., 1., np.nan],
                            [1., 1., 1., np.nan], [2., 1., 3., np.nan]], dtype=np.float64)
        mask = torch.tensor([[True, True, True, False]]).expand(6, -1)
        target = method_auc_targets(costs, mask)
        torch.testing.assert_close(target[AUC_POS_KEY], torch.tensor(
            [[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0],
             [0, 0, 1, 0], [0, 0, 0, 0], [0, 1, 0, 0]], dtype=torch.bool))
        torch.testing.assert_close(target[AUC_NEG_KEY], torch.tensor(
            [[0, 1, 1, 0], [0, 0, 1, 0], [1, 0, 0, 0],
             [1, 1, 0, 0], [0, 0, 0, 0], [1, 0, 1, 0]], dtype=torch.bool))
        with self.assertRaises(ValueError):
            method_auc_targets(costs.astype(np.float32), mask)
        with self.assertRaises(ValueError):
            method_auc_targets(costs)  # Padding NaN cannot be treated as a legal cost.
        labels = method_auc_targets(torch.tensor([[1., 2.], [2., 1.]], dtype=torch.float64))
        good = torch.tensor([[2., -1.], [-1., 2.]], requires_grad=True)
        args = (torch.ones_like(good, dtype=torch.bool), labels[AUC_POS_KEY], labels[AUC_NEG_KEY],
                torch.ones(2, dtype=torch.bool))
        loss = method_auc_loss(good, *args)
        self.assertLess(loss.item(), method_auc_loss(-good, *args).item())
        loss.backward()
        self.assertLess(good.grad[0, 0].item(), 0.)
        self.assertGreater(good.grad[1, 0].item(), 0.)

    def test_literal_formula_instance_shifts_permutation_padding_and_singletons(self):
        batch, eligible = fixture_batch()
        mask, pos, neg = batch['solver_mask'], batch[AUC_POS_KEY], batch[AUC_NEG_KEY]
        u = torch.randn(7, 3, dtype=torch.float64, requires_grad=True)
        a = torch.stack([torch.stack([u[row, method] - torch.logsumexp(
            torch.stack([u[row, other] for other in range(3) if other != method]), 0)
            for method in range(3)]) for row in range(7)])
        literal = torch.stack([F.softplus(a[neg[:, m], m, None] - a[pos[:, m], m][None]).mean()
                               for m in range(3)]).mean()
        loss = method_auc_loss(u, mask, pos, neg, eligible)
        torch.testing.assert_close(loss, literal, rtol=1e-12, atol=1e-12)
        grad = torch.autograd.grad(loss, u, retain_graph=True)[0]
        torch.testing.assert_close(grad, torch.autograd.grad(literal, u)[0], rtol=1e-12, atol=1e-12)
        shifted = u.detach().clone().requires_grad_()
        offsets = torch.tensor([100., -80., 4., 30., -1., 3., 7.], dtype=torch.float64)[:, None]
        shifted_loss = method_auc_loss(shifted + offsets, mask, pos, neg, eligible)
        torch.testing.assert_close(shifted_loss, loss, rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(torch.autograd.grad(shifted_loss, shifted)[0], grad, rtol=1e-12, atol=1e-12)
        order = torch.tensor([2, 0, 1])
        torch.testing.assert_close(method_auc_loss(u[:, order], mask[:, order], pos[:, order], neg[:, order],
                                                  eligible[order]), loss, rtol=1e-12, atol=1e-12)
        padded_u = torch.cat((u.detach(), torch.full((7, 2), float('nan'), dtype=torch.float64)), 1)
        padded_u.requires_grad_()
        padded_loss = method_auc_loss(padded_u, F.pad(mask, (0, 2)), F.pad(pos, (0, 2)),
                                      F.pad(neg, (0, 2)), F.pad(eligible, (0, 2)))
        torch.testing.assert_close(padded_loss, loss, rtol=1e-12, atol=1e-12)
        padded_grad = torch.autograd.grad(padded_loss, padded_u)[0]
        torch.testing.assert_close(padded_grad[:, :3], grad, rtol=1e-12, atol=1e-12)
        self.assertTrue((padded_grad[:, 3:] == 0).all())
        for k in (1, 3):
            singleton = torch.tensor([[2.] + [float('nan')] * (k - 1),
                                      [3.] + [float('inf')] * (k - 1)], requires_grad=True)
            one_mask = torch.zeros(2, k, dtype=torch.bool)
            one_mask[:, 0] = True
            labels = method_auc_targets(torch.ones(2, k, dtype=torch.float64), one_mask)
            one_loss = method_auc_loss(singleton, one_mask, labels[AUC_POS_KEY], labels[AUC_NEG_KEY],
                                       torch.zeros(k, dtype=torch.bool))
            self.assertEqual(one_loss.item(), 0.)
            self.assertTrue((method_competitive_scores(singleton, one_mask) == 0).all())
            one_loss.backward()
            self.assertTrue((singleton.grad == 0).all())

    def test_fixed_train_denominator_and_pairs_only_across_micros(self):
        labels = method_auc_targets(torch.tensor([[1., 2., 3.], [2., 1., 3.], [2., 3., 1.]],
                                                 dtype=torch.float64))
        eligible = train_eligible_methods(labels[AUC_POS_KEY], labels[AUC_NEG_KEY])
        self.assertTrue(eligible.all())
        u = torch.zeros(2, 3, requires_grad=True)
        mask = torch.ones_like(u, dtype=torch.bool)
        pos, neg = labels[AUC_POS_KEY][:2], labels[AUC_NEG_KEY][:2]
        full = method_auc_loss(u, mask, pos, neg, eligible)
        self.assertAlmostEqual(full.item(), 2. * math.log(2.) / 3., places=7)
        for row in range(2):
            self.assertEqual(method_auc_loss(u[row:row+1], mask[row:row+1],
                pos[row:row+1], neg[row:row+1], eligible).item(), 0.)
        full.backward()
        self.assertGreater(u.grad.abs().sum().item(), 0.)

    def _check_cached_reference(self, weight):
        batch, eligible = fixture_batch(full_logical=weight > 0)
        n, cap = len(batch['ind']), 256 if weight > 0 else 3
        chunks = [(start, min(start + cap, n)) for start in range(0, n, cap)]
        reference = RDropObjective(ToySelector()).train()
        cached = CachedAUCObjective(copy.deepcopy(reference.selector), auc_weight=weight).train()
        opt_ref, opt_cached = CountingSGD(reference.parameters()), CountingSGD(cached.parameters())
        seen, replay = [], []
        ref_hook = reference.selector.register_forward_hook(lambda module, args, out: seen.append(out['utility']))
        cache_hook = cached.selector.register_forward_hook(
            lambda module, args, out: replay.append((torch.is_grad_enabled(), out['utility'].detach().clone())))
        rng = capture_rank_rng(torch.device('cpu'))
        base, expected_parts = 0., torch.zeros(len(PART_KEYS) - 1, dtype=torch.float64)
        for start, stop in chunks:
            # Original RDropObjective must receive no new private supervision fields.
            micro = {k: v for k, v in slice_batch(batch, start, stop).items() if k not in AUC_LABEL_KEYS}
            loss, parts = reference(micro, .175)
            base = base + loss * ((stop - start) / n)
            expected_parts += parts.double() * ((stop - start) / n)
        final_rng = capture_rank_rng(torch.device('cpu'))
        ref_hook.remove()
        u1, u2 = torch.cat(seen[::2]), torch.cat(seen[1::2])
        self.assertFalse(torch.equal(u1, u2))
        auc_args = (batch['solver_mask'], batch[AUC_POS_KEY], batch[AUC_NEG_KEY], eligible)
        auc = .5 * (method_auc_loss(u1, *auc_args) + method_auc_loss(u2, *auc_args))
        if weight > 0:
            self.assertEqual(n, 1024)
            self.assertGreater(auc.item(), 0.)
            for start, stop in chunks:
                self.assertEqual(method_auc_loss(u1[start:stop], batch['solver_mask'][start:stop],
                    batch[AUC_POS_KEY][start:stop], batch[AUC_NEG_KEY][start:stop], eligible).item(), 0.)
        self.assertGreater(abs(auc.item() - method_auc_loss(.5 * (u1 + u2), *auc_args).item()), 1e-7)
        direct = base + weight * .5 * auc  # Target update 1863 / 3726.
        direct.backward()
        direct_norm = torch.nn.utils.clip_grad_norm_(reference.parameters(), 1.)
        expected_grads = {name: p.grad.clone() for name, p in reference.selector.named_parameters()}
        opt_ref.step()
        restore_rank_rng(rng, torch.device('cpu'))
        scaler = torch.amp.GradScaler('cpu', init_scale=8., growth_interval=100)
        with patch('torch.nn.utils.clip_grad_norm_', wraps=torch.nn.utils.clip_grad_norm_) as clip:
            with patch.object(scaler, 'unscale_', wraps=scaler.unscale_) as unscale:
                metrics, retries = utility_update(cached, opt_cached, scaler, batch, cap,
                    consistency_weight=.175, successful_update=1863, train_eligible=eligible, check_replay=True)
                self.assertEqual(unscale.call_count, 1)
            self.assertEqual(clip.call_count, 1)
        cache_hook.remove()
        self.assertEqual((retries, opt_cached.steps, scaler.get_scale()), (0, 1, 8.))
        self.assertTrue(torch.equal(torch.get_rng_state(), final_rng['torch']))
        view_count = 2 * len(chunks)
        self.assertEqual(len(replay), 2 * view_count)
        for index in range(view_count):
            self.assertFalse(replay[index][0])
            self.assertTrue(replay[index + view_count][0])
            torch.testing.assert_close(replay[index][1], seen[index], rtol=0., atol=0.)
            torch.testing.assert_close(replay[index + view_count][1], seen[index], rtol=0., atol=0.)
        self.assertAlmostEqual(metrics['loss'], direct.item(), places=6)
        self.assertAlmostEqual(metrics['base_loss'], base.item(), places=6)
        self.assertAlmostEqual(metrics['auc_loss'], auc.item(), places=6)
        self.assertAlmostEqual(metrics['auc_weight'], weight * .5, places=7)
        self.assertAlmostEqual(metrics['grad_norm'], direct_norm.item(), places=6)
        for key, value in zip(PART_KEYS[1:-1], expected_parts[1:]):
            torch.testing.assert_close(torch.tensor(metrics[key], dtype=torch.float64), value,
                                       rtol=2e-7, atol=3e-7, msg=key)
        for name, param in cached.selector.named_parameters():
            self.assertIsNotNone(param.grad, name)
            torch.testing.assert_close(param.grad, expected_grads[name], rtol=3e-5, atol=3e-7, msg=name)
            torch.testing.assert_close(param, dict(reference.selector.named_parameters())[name],
                                       rtol=3e-5, atol=3e-7, msg=name)

    def test_lambda_zero_matches_original_micro_rdrop_all_parameter_gradients(self):
        self._check_cached_reference(0.)

    def test_positive_lambda_matches_retained_micrographs_and_exact_two_view_replay(self):
        self._check_cached_reference(.35)

    def test_auc_alone_reaches_encoder_and_score_path_and_private_labels_stay_out(self):
        batch, eligible = fixture_batch()
        objective = CachedAUCObjective(ToySelector()).train()
        self.assertTrue(all(key not in auc_inputs_only(batch) for key in AUC_LABEL_KEYS + LABEL_KEYS))
        first, second = objective.views(batch)
        args = (batch['solver_mask'], batch[AUC_POS_KEY], batch[AUC_NEG_KEY], eligible)
        loss = .5 * (method_auc_loss(first['utility'], *args) + method_auc_loss(second['utility'], *args))
        loss.backward()
        for name in ('instance_encoder.0.weight', 'context.weight', 'method_embedding.weight',
                     'score_head.0.weight', 'score_head.3.weight'):
            grad = dict(objective.selector.named_parameters())[name].grad
            self.assertIsNotNone(grad, name)
            self.assertTrue(torch.isfinite(grad).all(), name)
            self.assertGreater(grad.abs().sum().item(), 0., name)
        self.assertEqual(objective.coefficient(0), 0.)
        self.assertEqual(objective.coefficient(3726), .35)
        self.assertEqual(objective.coefficient(5000), .35)
        # Independently supplied masks and the identical-row [B,K] candidate axis.
        clean_batch = {k: v for k, v in batch.items() if k not in AUC_LABEL_KEYS}
        clean_batch['pool_ids'] = batch['pool_ids'][None].expand(7, -1)
        optimizer = CountingSGD(objective.parameters())
        metrics, retries = utility_update(objective, optimizer, torch.amp.GradScaler('cpu', enabled=False),
            clean_batch, 3, consistency_weight=.01, successful_update=1, train_eligible=eligible,
            pos=batch[AUC_POS_KEY], neg=batch[AUC_NEG_KEY], check_replay=True)
        self.assertEqual(retries, 0)
        self.assertTrue(math.isfinite(metrics['loss']))
        mixed = dict(clean_batch, pool_ids=clean_batch['pool_ids'].clone())
        mixed['pool_ids'][1] = mixed['pool_ids'][1].flip(0)
        with self.assertRaisesRegex(ValueError, 'shared method column axis'):
            utility_update(objective, optimizer, torch.amp.GradScaler('cpu', enabled=False), mixed, 3,
                consistency_weight=.01, successful_update=2, train_eligible=eligible,
                pos=batch[AUC_POS_KEY], neg=batch[AUC_NEG_KEY])

    def test_overflow_retries_whole_batch_same_rng_one_step(self):
        batch, eligible = fixture_batch()
        objective = CachedAUCObjective(ToySelector()).train()
        baseline = copy.deepcopy(objective)
        opt, ref_opt = CountingSGD(objective.parameters()), CountingSGD(baseline.parameters())
        rng = capture_rank_rng(torch.device('cpu'))
        expected, _ = utility_update(baseline, ref_opt, torch.amp.GradScaler('cpu', init_scale=8.),
            batch, 3, consistency_weight=.1, successful_update=100, train_eligible=eligible, check_replay=True)
        expected_rng = torch.get_rng_state().clone()
        restore_rank_rng(rng, torch.device('cpu'))
        attempts, injected = [], [False]
        forward_hook = objective.selector.register_forward_hook(
            lambda module, args, out: attempts.append(out['utility'].detach().clone()))

        def overflow(gradient):
            if not injected[0]:
                injected[0] = True
                return torch.full_like(gradient, float('inf'))
            return gradient

        gradient_hook = next(objective.parameters()).register_hook(overflow)
        scaler = torch.amp.GradScaler('cpu', init_scale=8., growth_interval=100)
        actual, retries = utility_update(objective, opt, scaler, batch, 3, consistency_weight=.1,
            successful_update=100, train_eligible=eligible, check_replay=True, max_retries=3)
        forward_hook.remove()
        gradient_hook.remove()
        self.assertEqual((retries, opt.steps, scaler.get_scale()), (1, 1, 4.))
        self.assertEqual(len(attempts), 24)
        self.assertTrue(all(torch.equal(attempts[i], attempts[i + 12]) for i in range(12)))
        self.assertTrue(torch.equal(expected_rng, torch.get_rng_state()))
        self.assertAlmostEqual(actual['loss'], expected['loss'], places=7)
        for param, expected_param in zip(objective.parameters(), baseline.parameters()):
            torch.testing.assert_close(param, expected_param, rtol=0., atol=0.)


if __name__ == '__main__':
    unittest.main()
