"""Only the two authorized tiny CPU checks for R78 round006."""

import copy
import json
import os
from pathlib import Path
import time
import unittest

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from .pairwise_objective import comparison_targets
from .r76_experiment import capture_rank_rng, restore_rank_rng, slice_batch, utc, write_json
from .r78_profile_loss import (PROFILE_KEY, PROFILE_WEIGHT, ProfileObjective, cached_update,
                               cost_profiles, profile_loss, profile_order)


class ToySelector(nn.Module):
    def __init__(self):
        super().__init__()
        self.params = dict(embedding_dim=4)
        self.encoder = nn.Sequential(nn.Linear(3, 4), nn.Tanh(), nn.Dropout(.25))
        self.context = nn.Linear(4, 4)
        self.solvers = nn.Parameter(torch.randn(3, 4))
        self.score_head = nn.Sequential(nn.Linear(12, 7), nn.GELU(), nn.Dropout(.4), nn.Linear(7, 1))

    def encode_pairs(self, batch):
        if PROFILE_KEY in batch or any(k in batch for k in ('costs', 'ind', 'cost_regret')):
            raise AssertionError('Labels/targets reached the model')
        h = self.encoder(batch['x'])
        return torch.cat((h[:, None].expand(-1, 3, -1), self.context(h)[:, None].expand(-1, 3, -1),
                          self.solvers[None].expand(len(h), -1, -1)), -1), batch['solver_mask']


class RecordingScaler:
    """Exercise actual scaling semantics on CPU, without initializing CUDA."""
    def __init__(self):
        self.scales, self.unscales, self.steps, self.updates = 0, 0, 0, 0

    def scale(self, loss):
        self.scales += 1
        return loss * 8.

    def unscale_(self, optimizer):
        self.unscales += 1
        for group in optimizer.param_groups:
            for param in group['params']:
                if param.grad is not None:
                    param.grad.div_(8.)

    def step(self, optimizer):
        self.steps += 1
        optimizer.step()

    def update(self):
        self.updates += 1

    def is_enabled(self):
        return True


class ProfileCPUChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_profile_mask_exact_ties_and_literal_formula(self):
        costs = torch.tensor([[1., 2., float('nan')], [2., 4., float('nan')],
                              [2., 1., float('nan')], [3., 3., float('nan')]], dtype=torch.float64)
        mask = torch.tensor([[True, True, False]]).expand_as(costs)
        q = cost_profiles(costs, mask)
        self.assertEqual(q.dtype, torch.float64)
        torch.testing.assert_close(q[0], q[1], rtol=0., atol=0.)
        features = torch.tensor([[1., .2, -.1], [.8, .4, .1], [-.3, .7, 1.], [0., 0., 0.]], requires_grad=True)
        actual = profile_loss(features, profile_order(q))
        e = F.normalize(features, dim=1, eps=1e-12)
        s = e @ e.T
        d = (q[:, None] - q[None]).abs().mean(-1)
        literal = torch.stack([torch.logsumexp(torch.stack([s[i, k] for k in range(4)
            if k != i and d[i, k] >= d[i, j]]), 0) - s[i, j]
            for i in range(4) for j in range(4) if i != j]).mean()
        torch.testing.assert_close(actual, literal)
        torch.testing.assert_close(torch.autograd.grad(actual, features, retain_graph=True)[0],
                                   torch.autograd.grad(literal, features)[0])
        with self.assertRaises(ValueError):
            cost_profiles(costs.float(), mask)
        bad_mask = mask.clone()
        bad_mask[-1, 1] = False
        with self.assertRaises(ValueError):
            cost_profiles(costs, bad_mask)

    def test_two_full_forwards_cached_gradient_rng_and_one_scaled_update(self):
        torch.manual_seed(2)
        costs = np.asarray([[1., 1.2, 1.4], [2.1, 2., 2.4], [3.2, 3.6, 3.],
                            [1.1, 1., 1.3], [4., 4.4, 4.8], [2.2, 2.3, 2.], [3., 2.8, 2.9]], dtype=np.float64)
        batch = dict(x=torch.randn(7, 3), solver_mask=torch.ones(7, 3, dtype=torch.bool),
                     ind=torch.from_numpy(costs.argmin(1)), pool_ids=torch.tensor([4, 1, 9]),
                     **comparison_targets(costs))
        batch[PROFILE_KEY] = cost_profiles(torch.from_numpy(costs))
        reference = ProfileObjective(ToySelector()).train()
        cached = copy.deepcopy(reference)
        opt_ref = torch.optim.AdamW(reference.parameters(), lr=2e-5, weight_decay=1e-4)
        opt_cached = torch.optim.AdamW(cached.parameters(), lr=2e-5, weight_decay=1e-4)
        rng = capture_rank_rng(torch.device('cpu'))
        first_h, second_h, base = [], [], 0.
        for start in range(0, 7, 3):
            stop = min(start + 3, 7)
            micro = slice_batch(batch, start, stop)
            first, second = reference.views(micro)
            first_h.append(first['profile_h'])
            second_h.append(second['profile_h'])
            loss, _ = reference.base(first, second, micro)
            base = base + loss * ((stop - start) / 7)
        expected_rng = torch.get_rng_state().clone()
        order = profile_order(batch[PROFILE_KEY])
        aux = PROFILE_WEIGHT * .5 * (profile_loss(torch.cat(first_h), order)
                                     + profile_loss(torch.cat(second_h), order))
        direct = base + aux
        direct.backward()
        direct_norm = torch.nn.utils.clip_grad_norm_(reference.parameters(), 1.)
        expected_grads = {key: param.grad.clone() for key, param in reference.named_parameters()}
        opt_ref.step()
        restore_rank_rng(rng, torch.device('cpu'))
        scaler = RecordingScaler()
        actual, retries = cached_update(cached, opt_cached, scaler, batch, 3, check_replay=True)
        self.assertEqual(retries, 0)
        self.assertTrue(torch.equal(expected_rng, torch.get_rng_state()))
        self.assertEqual((scaler.scales, scaler.unscales, scaler.steps, scaler.updates), (3, 1, 1, 1))
        self.assertAlmostEqual(actual['loss'], direct.item(), places=6)
        self.assertAlmostEqual(actual['grad_norm'], direct_norm.item(), places=6)
        for key, param in cached.named_parameters():
            torch.testing.assert_close(param.grad, expected_grads[key], rtol=2e-5, atol=2e-7)
            torch.testing.assert_close(param, dict(reference.named_parameters())[key], rtol=2e-5, atol=2e-7)


if __name__ == '__main__':
    started = time.monotonic()
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(ProfileCPUChecks))
    if os.environ.get('R78_PROFILE_CHECK_RECEIPT'):
        write_json(Path(os.environ['R78_PROFILE_CHECK_RECEIPT']), dict(utc=utc(),
            planned=2, tests_run=result.testsRun, passed=result.testsRun - len(result.failures) - len(result.errors),
            successful=result.wasSuccessful(), elapsed_seconds=time.monotonic() - started,
            device='cpu', gpu_initialized=torch.cuda.is_initialized(),
            tests=['profile_mask_exact_ties_literal_formula', 'two_full_forwards_gradient_rng_scaled_update']))
    raise SystemExit(not result.wasSuccessful())
