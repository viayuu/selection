"""Focused offline CPU tests; synthetic fixtures are not experiment results."""

import copy
import itertools
import math
import unittest

import numpy as np
import torch
import torch.nn.functional as F

from .context_matched_objective import (
    CMR_GROUP_KEY, CMR_LABEL_KEYS, CMR_REGRET_KEY, CachedCMRObjective,
    context_matched_loss, utility_update,
)
from .method_auc_objective import method_competitive_scores
from .pairwise_objective import comparison_targets
from .r76_experiment import RDropObjective, capture_rank_rng, restore_rank_rng, slice_batch
from .r82_context import apply_context, batch_coverage, fit_context, relative_regret
from .test_r81_auc import CountingSGD, ToySelector


class LeakGuardSelector(ToySelector):
    def encode_pairs(self, batch):
        if any(key in batch for key in CMR_LABEL_KEYS):
            raise AssertionError('Context/regret supervision entered the selector')
        return super().encode_pairs(batch)


class R82CMRTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(2)

    def test_literal_hierarchy_small_gaps_ties_and_direction(self):
        u = torch.randn(4, 3, dtype=torch.float64, requires_grad=True)
        mask = torch.ones_like(u, dtype=torch.bool)
        r = torch.tensor([[0., .001, .02], [.01, 0., .02], [0., .003, .02], [0., .007, .02]], dtype=torch.float64)
        groups = torch.tensor([0, 0, 1, 1])
        a = method_competitive_scores(u, mask)
        methods = []
        for m in range(3):
            grouped = []
            for g in (0, 1):
                pairs = []
                for i, j in itertools.combinations(torch.where(groups == g)[0].tolist(), 2):
                    dr = r[j, m] - r[i, m]
                    if dr != 0:
                        pairs.append((dr.abs()/.01).clamp_max(1.) * F.softplus(-dr.sign()*(a[i, m]-a[j, m])))
                if pairs:
                    grouped.append(torch.stack(pairs).mean())
            if grouped:
                methods.append(torch.stack(grouped).mean())
        literal = torch.stack(methods).mean()
        actual, stats = context_matched_loss(u, mask, r, groups)
        torch.testing.assert_close(actual, literal, rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(torch.autograd.grad(actual, u, retain_graph=True)[0],
            torch.autograd.grad(literal, u)[0], rtol=1e-12, atol=1e-12)
        self.assertEqual(stats['cmr_effective_methods'].item(), 2)
        self.assertEqual(stats['cmr_effective_groups'].item(), 3)
        zero, _ = context_matched_loss(torch.zeros_like(u), mask, r, groups)
        self.assertAlmostEqual(zero.item(), .625*math.log(2.), places=12)
        good = torch.tensor([[2., 0.], [0., 2.]], requires_grad=True)
        labels = torch.tensor([[0., .01], [.01, 0.]], dtype=torch.float64)
        same = torch.zeros(2, dtype=torch.int64)
        self.assertLess(context_matched_loss(good, mask[:2,:2], labels, same)[0].item(),
                        context_matched_loss(-good, mask[:2,:2], labels, same)[0].item())
        # Different contexts cannot create a comparison, even with different regrets.
        loss, stats = context_matched_loss(good, mask[:2,:2], labels, torch.tensor([0, 1]))
        self.assertEqual((loss.item(), stats['cmr_pair_count'].item()), (0., 0))

    def test_permutation_padding_translation_and_fp64_near_ties(self):
        costs = np.asarray([[1., 1.+1e-10], [1.+2e-10, 1.], [1., 1.]], dtype=np.float64)
        regrets = torch.from_numpy(relative_regret(costs))
        self.assertGreater(regrets[0,1].item(), 0.)
        with self.assertRaises(ValueError):
            relative_regret(costs.astype(np.float32))
        groups = torch.zeros(3, dtype=torch.int64)
        u = torch.randn(3, 2, dtype=torch.float64, requires_grad=True)
        mask = torch.ones_like(u, dtype=torch.bool)
        loss, _ = context_matched_loss(u, mask, regrets, groups)
        self.assertGreater(loss.item(), 0.)
        order = torch.tensor([2,0,1])
        shifted = u + torch.tensor([100., -70., 4.], dtype=u.dtype)[:,None]
        torch.testing.assert_close(context_matched_loss(shifted[order].flip(1), mask[order], regrets[order].flip(1),
            groups[order])[0], loss, rtol=1e-10, atol=1e-20)
        padded = torch.cat((u.detach(), torch.full((3,1), float('nan'), dtype=u.dtype)), 1).requires_grad_()
        padded_r = torch.cat((regrets, torch.full((3,1), float('nan'), dtype=regrets.dtype)), 1)
        padded_mask = F.pad(mask, (0,1))
        actual, _ = context_matched_loss(padded, padded_mask, padded_r, groups)
        torch.testing.assert_close(actual, loss, rtol=1e-12, atol=1e-20)
        actual.backward()
        self.assertTrue(torch.isfinite(padded.grad).all())
        self.assertTrue((padded.grad[:,2] == 0).all())
        single = torch.randn(3,1, requires_grad=True)
        zero, _ = context_matched_loss(single, torch.ones_like(single, dtype=torch.bool),
            torch.zeros(3,1,dtype=torch.float64), groups)
        zero.backward()
        self.assertEqual(zero.item(), 0.)
        self.assertTrue((single.grad == 0).all())

    def test_train_fitted_context_equal_sizes_and_unknown_groups(self):
        metadata = dict(scale=np.asarray([50]*5+[500]*5), distribution=np.asarray(['uniform']*10),
            capacity_type=np.asarray(['none']*10))
        groups, spec = fit_context(metadata, 'TSP')
        self.assertEqual(spec['actual_size_bins'], 2)
        self.assertEqual(len(np.unique(groups[:5])), 1)
        self.assertEqual(len(np.unique(groups[5:])), 1)
        self.assertNotEqual(groups[0], groups[-1])
        np.testing.assert_array_equal(apply_context(metadata, spec), groups)
        unknown = dict(metadata, distribution=np.asarray(['not_in_train']*10))
        self.assertTrue((apply_context(unknown, spec) == -1).all())
        constant = dict(metadata, scale=np.full(10,100))
        g, fitted = fit_context(constant, 'TSP')
        self.assertEqual((len(np.unique(g)), fitted['actual_size_bins']), (1,1))
        costs = np.asarray([[1.,1.1],[1.1,1.],[1.,1.1],[1.1,1.]], dtype=np.float64)
        c = batch_coverage(np.array([0,0,1,1]), relative_regret(costs))
        self.assertEqual((c['row_coverage'],c['unequal_method_pairs']), (1.,4))

    def _check_update(self, weight, n, cap):
        if n == 1024:
            costs = np.repeat(np.asarray([[1.,1.003,1.02], [1.01,1.,1.02],
                [1.015,1.02,1.], [1.02,1.007,1.]], dtype=np.float64), 256, axis=0)
        else:
            costs = np.asarray([[1.,1.+1e-10,1.4],[2.1,2.,2.4],[3.2,3.6,3.],
                [1.1,1.,1.3],[4.,4.4,4.8],[2.2,2.3,2.],[3.,2.8,2.9]], dtype=np.float64)
        batch = dict(x=torch.randn(n,3), pool_ids=torch.tensor([4,1,9]),
            solver_mask=torch.ones(n,3,dtype=torch.bool), ind=torch.from_numpy(costs.argmin(1)),
            costs=torch.from_numpy(costs.astype(np.float32)), **comparison_targets(costs))
        regrets, groups = torch.from_numpy(relative_regret(costs)), torch.arange(n)%4
        batch.update({CMR_GROUP_KEY:groups, CMR_REGRET_KEY:regrets})
        reference = RDropObjective(LeakGuardSelector()).train()
        cached = CachedCMRObjective(copy.deepcopy(reference.selector), cmr_weight=weight).train()
        optimizer = CountingSGD(cached.parameters())
        rng = capture_rank_rng(torch.device('cpu'))
        seen = []
        hook = reference.selector.register_forward_hook(lambda module,args,out:seen.append(out['utility']))
        base = 0.
        for start in range(0,n,cap):
            stop = min(start+cap,n)
            micro = {k:v for k,v in slice_batch(batch,start,stop).items() if k not in CMR_LABEL_KEYS}
            base = base + reference(micro,.175)[0]*((stop-start)/n)
        hook.remove()
        final_rng = capture_rank_rng(torch.device('cpu'))
        u1,u2 = torch.cat(seen[::2]),torch.cat(seen[1::2])
        first,_ = context_matched_loss(u1,batch['solver_mask'],regrets,groups)
        second,_ = context_matched_loss(u2,batch['solver_mask'],regrets,groups)
        if n == 1024:
            self.assertGreater(first.item(),0.)
            for start in range(0,n,cap):
                self.assertEqual(context_matched_loss(u1[start:start+cap],batch['solver_mask'][start:start+cap],
                    regrets[start:start+cap],groups[start:start+cap])[0].item(),0.)
        expected = base + weight*.5*.5*(first+second)
        expected.backward()
        norm = torch.nn.utils.clip_grad_norm_(reference.parameters(),1.)
        grads = {k:p.grad.clone() for k,p in reference.selector.named_parameters()}
        restore_rank_rng(rng,torch.device('cpu'))
        scaler = torch.amp.GradScaler('cpu',init_scale=8.)
        parts,retries = utility_update(cached,optimizer,scaler,batch,cap,consistency_weight=.175,
            successful_update=1863,check_replay=True)
        self.assertEqual((optimizer.steps,retries),(1,0))
        self.assertAlmostEqual(parts['loss'],expected.item(),places=6)
        self.assertAlmostEqual(parts['base_loss'],base.item(),places=6)
        self.assertAlmostEqual(parts['grad_norm'],norm.item(),places=6)
        self.assertTrue(torch.equal(torch.get_rng_state(),final_rng['torch']))
        for name,p in cached.selector.named_parameters():
            torch.testing.assert_close(p.grad,grads[name],rtol=3e-5,atol=3e-7,msg=name)

    def test_lambda_zero_original_loss_gradients_and_rng(self):
        self._check_update(0.,7,3)

    def test_full1024_cross_micro_pairs_gradients_and_rng(self):
        self._check_update(.10,1024,256)


if __name__ == '__main__':
    unittest.main()
