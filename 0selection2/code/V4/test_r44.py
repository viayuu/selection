import copy
import unittest

import numpy as np
import torch
import torch.nn.functional as F

from .conditional_node_adapter import ConditionalNodeSelector, NodeResidual, adapter_dimensions, load_baseline
from .pairwise_selector import ScoreDifferenceSelector, maximin_logits, rank_by_solver_ids
from .pairwise_objective import comparison_targets, predicted_top3, single_objective
from .r43_experiment import policy_metrics, update_batch
from .test_dual_stream import params
from .test_solver_features import sample_batch


class R44Tests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(2)
        self.params = dict(params(), pair_mode='score_difference')
        self.baseline = ScoreDifferenceSelector(**self.params).eval()

    def model(self, mode):
        model = ConditionalNodeSelector(**dict(self.params, node_adapter_mode=mode)).eval()
        load_baseline(model, self.baseline.state_dict())
        return model

    def test_identity_and_default_compatibility(self):
        for mode in ('none', 'shared', 'conditional'):
            model = self.model(mode)
            for p in ('TSP', 'CVRP', 'ATSP', 'OVRPTW'):
                batch = sample_batch(p)
                first, second = self.baseline(batch), model(batch, return_utility=True)
                for key in first:
                    torch.testing.assert_close(second[key], first[key], atol=0, rtol=0)
                torch.testing.assert_close(second['logits'], maximin_logits(second['pair_margin'], second['solver_mask']), atol=0, rtol=0)
        no_adapter = ConditionalNodeSelector(**self.params)
        no_adapter.load_state_dict(self.baseline.state_dict(), strict=True)

    def test_parameters_initialization_and_shared_no_condition(self):
        rank, shared_rank, conditional, shared = adapter_dimensions(128)
        self.assertEqual((rank, shared_rank, conditional, shared), (32, 39, 9824, 9984))
        for mode, expected in (('conditional', conditional), ('shared', shared)):
            adapter = NodeResidual(128, mode)
            self.assertEqual(sum(p.numel() for p in adapter.parameters()), expected)
            self.assertEqual(int(adapter.up.weight.count_nonzero()), 0)
        model = self.model('shared')
        batch = sample_batch('TSP')
        changed = dict(batch)
        del changed['problem_desc']
        nodes, mask = model.instance_encoder(batch)
        torch.testing.assert_close(model.adapt_nodes(nodes, mask, batch), model.adapt_nodes(nodes, mask, changed), atol=0, rtol=0)

    def test_metadata_whitelist_padding_and_nonconstant_condition(self):
        adapter = NodeResidual(128, 'conditional')
        batch = sample_batch('CVRP')
        q = adapter.condition(batch, batch['node_mask'])
        changed = dict(batch, costs=torch.full_like(batch['costs'], float('nan')), pool_ids=batch['pool_ids'].flip(0),
                       ind=torch.tensor([1, 0]), problem_id=999, n=torch.tensor([999, 999]))
        torch.testing.assert_close(q, adapter.condition(changed, batch['node_mask']), atol=0, rtol=0)
        mask = F.pad(batch['node_mask'], (0, 3), value=False)
        torch.testing.assert_close(q, adapter.condition(batch, mask), atol=0, rtol=0)
        self.assertFalse(torch.equal(q, adapter.condition(sample_batch('TSP'), batch['node_mask'])))
        torch.testing.assert_close(adapter.gate(batch, batch['node_mask']), torch.ones(2, 32), atol=0, rtol=0)
        nnodes = mask.size(1)
        nodes = torch.randn(2, nnodes, 128)
        torch.nn.init.normal_(adapter.up.weight, std=.1)
        out = adapter(nodes, mask, batch)
        torch.testing.assert_close(out[~mask], nodes[~mask], atol=0, rtol=0)

    def test_masks_and_gradients_after_zero_initialization(self):
        for mode in ('shared', 'conditional'):
            model = self.model(mode).train()
            batch = sample_batch('CVRP')
            k = len(batch['pool_ids'])
            costs = np.stack([np.arange(k)+1., np.arange(k)[::-1]+1.]).astype(np.float64)
            batch.update(comparison_targets(costs))
            batch['ind'] = torch.tensor([0, k-1])
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
            scaler = torch.amp.GradScaler('cuda', enabled=False)
            for _ in range(2):
                update_batch(model, optimizer, scaler, batch, .35)
            optimizer.zero_grad(set_to_none=True)
            output = model(batch)
            focus = predicted_top3(output['logits'], output['logits'], batch['pool_ids'], output['solver_mask'])
            loss, _ = single_objective(output, batch, focus)
            loss.backward()
            for name, parameter in model.node_adapter.named_parameters():
                self.assertIsNotNone(parameter.grad, name)
                self.assertTrue(torch.isfinite(parameter.grad).all(), name)
                self.assertGreater(float(parameter.grad.abs().sum()), 0., name)
            self.assertGreater(float(model.instance_encoder.coord_proj.weight.grad.abs().sum()), 0.)
            model.eval()
            padded = copy.deepcopy(batch)
            padded['pool_ids'] = torch.cat([batch['pool_ids'], torch.tensor([-1])])
            padded['solver_mask'] = torch.ones(2, k+1, dtype=torch.bool)
            padded['solver_mask'][:, -1] = False
            output = model(padded)
            self.assertTrue(torch.isneginf(output['logits'][:, -1]).all())
            self.assertTrue(torch.isfinite(output['logits'][:, :-1]).all())

    def test_saved_score_replay_not_utility(self):
        output = self.model('conditional')(sample_batch('TSP'), return_utility=True)
        k = output['logits'].size(-1)
        raw = dict(costs=np.stack([np.arange(k)+1., np.arange(k)[::-1]+1.]).astype(np.float64),
                   winner=np.array([0, k-1]), pool=[str(i) for i in range(k)], pool_ids=list(range(k)))
        margin = output['utility'][:, :, None]-output['utility'][:, None, :]
        torch.testing.assert_close(margin, output['pair_margin'], atol=0, rtol=0)
        replay = maximin_logits(margin, output['solver_mask'])
        first, pred1 = policy_metrics(output['logits'].detach().numpy(), raw)
        second, pred2 = policy_metrics(replay.detach().numpy(), raw)
        np.testing.assert_array_equal(pred1, pred2)
        for key in ('ce', 'top1', 'mean_cost', 'vs_sbs_pct', 'actual_regret_pct'):
            self.assertEqual(first[key], second[key])
        tied = rank_by_solver_ids(torch.zeros(1, 3), torch.tensor([4, 1, 3]))
        self.assertEqual(int(tied[0, 0]), 1)


if __name__ == '__main__':
    unittest.main()
