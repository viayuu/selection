"""Focused R47 integrity checks, synthetic data only; no test-set access."""

import copy
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from code.unified_selector.registry import POOLS, PROBLEMS
from .behavior_memory import BehaviorBank, base_instance_ids, restore_behavior, shuffled_behavior
from .pairwise_objective import comparison_targets, inputs_only
from .pairwise_selector import ScoreDifferenceSelector
from .r43_experiment import update_batch
from .r47_model import BehaviorMemorySelector, load_r47_checkpoint
from .test_dual_stream import params
from .test_solver_features import sample_batch


class R47Tests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)

    def model(self, problem='CVRP', dropout=0.):
        config = dict(params(), pair_mode='score_difference', neighbors=4, dropout=dropout)
        torch.manual_seed(2)
        model = BehaviorMemorySelector(**config)
        batch = sample_batch(problem)
        batch['base_id'] = torch.tensor([0, 1])
        ids = batch['pool_ids']
        count, d = 48, 2*config['embedding_dim']+1
        gaps = torch.rand(count, len(ids), dtype=torch.float64)
        gaps[:, 0] = 0
        model.memory[problem] = BehaviorBank(torch.randn(count, d), gaps,
            ids[torch.arange(count) % len(ids)], ids, torch.ones(count, dtype=torch.long)*10,
            torch.arange(count))
        return model.eval(), batch, config

    def test_common_scratch_initialization(self):
        config = dict(params(), pair_mode='score_difference', neighbors=4)
        torch.manual_seed(2)
        old = ScoreDifferenceSelector(**config)
        torch.manual_seed(2)
        new = BehaviorMemorySelector(**config)
        for name, value in old.state_dict().items():
            actual = new.state_dict()[name]
            if name == 'score_head.0.weight':
                actual = actual[:, :value.size(1)]
            torch.testing.assert_close(actual, value, atol=0, rtol=0)

    def test_label_isolation_and_retrieval_not_using_behavior(self):
        model, batch, _ = self.model()
        original = model(batch)
        changed = dict(batch, costs=torch.full_like(batch['costs'], float('nan')), ind=torch.tensor([99, 99]))
        torch.testing.assert_close(model(changed)['logits'], original['logits'], atol=0, rtol=0)
        bank = model.memory['CVRP']
        with torch.no_grad():
            bank.gap_input.add_(7.)
            bank.winner_ids[:] = batch['pool_ids'][-1]
        output = model(inputs_only(batch))
        torch.testing.assert_close(output['neighbor_indices'], original['neighbor_indices'])
        torch.testing.assert_close(output['neighbor_weights'], original['neighbor_weights'])
        self.assertFalse(torch.allclose(output['behavior'], original['behavior']))

    def test_shared_squared_l2_and_related_exclusion(self):
        model, batch, _ = self.model()
        _, summary, _, _, _ = model.encode_for_retrieval(batch)
        bank = model.memory['CVRP']
        bank.summary[:2].copy_(summary.detach())
        bank.summary[2].copy_(summary[0].detach())
        bank.base_ids[2] = 0
        query, selected, indices, weights = model.retrieval.retrieve(summary, batch['base_id'], bank)
        self.assertFalse((bank.base_ids[indices] == batch['base_id'][:, None]).any())
        torch.testing.assert_close(weights, (-(query[:, None]-selected).square().sum(-1)).softmax(-1))
        query.retain_grad()
        selected.retain_grad()
        (weights*torch.arange(weights.size(-1))).sum().backward()
        self.assertGreater(float(query.grad.abs().sum()), 0)
        self.assertGreater(float(selected.grad.abs().sum()), 0)
        self.assertIsNone(bank.summary.grad)

    def test_input_identity_groups_copies_but_not_labels(self):
        batch = sample_batch('CVRP')
        original = base_instance_ids(batch)
        changed = copy.deepcopy(batch)
        changed['costs'].fill_(999)
        changed['node'][:, :, 0] = 1-changed['node'][:, :, 0]
        changed['node'] = changed['node'].flip(1)
        changed['node_mask'] = changed['node_mask'].flip(1)
        # Synthetic samples are fixed-size; the flipped node order retains all valid rows.
        torch.testing.assert_close(original, base_instance_ids(changed))

    def test_candidate_permutation_padding_and_single_candidate(self):
        model, batch, _ = self.model()
        expected = model(batch)['logits']
        order = torch.randperm(len(batch['pool_ids']))
        actual = model(dict(batch, pool_ids=batch['pool_ids'][order]))['logits']
        torch.testing.assert_close(actual, expected[:, order], atol=3e-6, rtol=3e-5)
        mask = F.pad(torch.ones_like(expected, dtype=torch.bool), (0, 2), value=False)
        padded = model(dict(batch, pool_ids=F.pad(batch['pool_ids'], (0, 2), value=-1), solver_mask=mask))['logits']
        torch.testing.assert_close(padded[:, :-2], expected, atol=3e-6, rtol=3e-5)
        self.assertTrue(torch.isneginf(padded[:, -2:]).all())
        one = model(dict(batch, pool_ids=batch['pool_ids'][:1]))['logits']
        torch.testing.assert_close(one, torch.zeros_like(one))
        more_nodes = dict(batch, node=F.pad(batch['node'], (0, 0, 0, 3), value=999.),
                          node_mask=F.pad(batch['node_mask'], (0, 3), value=False))
        torch.testing.assert_close(model(more_nodes)['logits'], expected, atol=3e-6, rtol=3e-5)

    def test_memory_rows_are_solver_specific_and_shared_gradients(self):
        model, batch, _ = self.model()
        output = model(batch)
        self.assertFalse(torch.allclose(output['behavior'][:, 0], output['behavior'][:, 1]))
        F.cross_entropy(output['logits'], torch.tensor([0, 1])).backward()
        for module in (model.retrieval.key, model.retrieval.value, model.retrieval.correction,
                       model.instance_encoder.coord_proj, model.joint_layers, model.score_head):
            grads = [v.grad for v in module.parameters() if v.grad is not None]
            self.assertTrue(grads)
            self.assertTrue(all(torch.isfinite(g).all() for g in grads))
            self.assertGreater(float(sum(g.abs().sum() for g in grads)), 0)
        self.assertTrue(all(not v.requires_grad for v in model.memory.buffers()))

    def test_whole_row_shuffle_preserves_marginals_keys_and_size(self):
        model, _, _ = self.model()
        before = copy.deepcopy(model.memory.state_dict())
        original, report = shuffled_behavior(model.memory)
        bank = model.memory['CVRP']
        permutation = torch.tensor(report['CVRP']['permutation'])
        for key in ('gap', 'gap_input', 'winner_ids'):
            torch.testing.assert_close(getattr(bank, key), before['CVRP.'+key][permutation])
        for key in ('summary', 'pool_ids', 'nodes', 'base_ids'):
            torch.testing.assert_close(getattr(bank, key), before['CVRP.'+key])
        torch.testing.assert_close(bank.nodes, bank.nodes[permutation])
        restore_behavior(model.memory, original)
        for key, value in before.items():
            torch.testing.assert_close(model.memory.state_dict()[key], value, atol=0, rtol=0)

    def test_dropout_two_forwards_one_update(self):
        model, batch, _ = self.model(dropout=.1)
        model.train()
        batch['ind'] = torch.tensor([0, 1])
        batch.update(comparison_targets(batch['costs'].numpy().astype(np.float64)))
        first, second = model(batch), model(batch)
        self.assertFalse(torch.equal(first['logits'], second['logits']))
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        values = update_batch(model, optimizer, torch.amp.GradScaler('cuda', enabled=False), batch, .35)
        self.assertTrue(np.isfinite(values['loss']))
        self.assertTrue(all(float(v['step']) == 1 for v in optimizer.state.values()))

    def test_self_contained_checkpoint_and_matrix_path(self):
        model, batch, config = self.model('ATSP')
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'best.pt'
            torch.save(dict(model=model.state_dict(), args=dict(model_params=config, sizes={'ATSP': 48})), path)
            loaded, _ = load_r47_checkpoint(path)
            torch.testing.assert_close(loaded(inputs_only(batch))['logits'], model(batch)['logits'], atol=0, rtol=0)
        with self.assertRaises(ValueError):
            model(dict(batch, problem_id=PROBLEMS.index('TSP')))


if __name__ == '__main__':
    unittest.main()
