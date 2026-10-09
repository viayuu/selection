"""Focused R46 tests; real locked embeddings, no API calls or training-data reads."""

import copy
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
import torch.nn.functional as F

from code.unified_selector.registry import GLOBAL_SOLVERS, POOLS, S2I
from .pairwise_objective import comparison_targets, inputs_only
from .r43_experiment import update_batch
from .r46_model import load_r46_checkpoint, make_r46_model
from .solver_code_embeddings import bundle_provenance, load_bundle
from .solver_code_tokens import DynamicSourcePool, SolverCodeTokens, source_package_audit
from .test_dual_stream import params
from .test_solver_features import sample_batch


class R46Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.bundle = load_bundle(Path(__file__).parent / 'runs/R45_solver_code_embeddings/solver_code_vectors_atlas.pt')

    def model(self, dropout=0.):
        torch.manual_seed(2)
        config = dict(params(), pair_mode='score_difference', solver_representation='source_tokens', dropout=dropout)
        return make_r46_model(config, self.bundle), config

    def test_no_learnable_solver_identity_or_old_feature_path(self):
        model, _ = self.model()
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Embedding):
                self.assertTrue(name.startswith('instance_encoder.condition_encoder.') or
                                name == 'solver_encoder.role_embedding', name)
        for name, _ in model.named_parameters():
            self.assertFalse(any(word in name for word in ('solver_emb', 'arm_emb', 'feature_mlp', 'semantic_rows')))
        self.assertEqual(model.solver_encoder.role_embedding.num_embeddings, 4)
        for value in model.solver_encoder.buffers():
            self.assertFalse(value.requires_grad)

    def test_full_source_views_and_variants_stay_separate(self):
        model, _ = self.model()
        ids = torch.arange(len(GLOBAL_SOLVERS))[None]
        tokens, mask = model.solver_encoder(ids, torch.ones_like(ids, dtype=torch.bool))
        indices = self.bundle['chunk_indices'].flatten(1)
        torch.testing.assert_close(mask[0], indices >= 0)
        self.assertEqual(tokens.shape[2], 8)
        self.assertEqual(int(mask[0, S2I['OMNI']].sum()), 8)
        self.assertEqual(int(mask[0, S2I['BQ']].sum()), 4)
        original = tokens.detach().clone()
        with torch.no_grad():
            model.solver_encoder.code_vectors.normal_()
        torch.testing.assert_close(model.solver_encoder(ids, mask.any(-1))[0], original, atol=0, rtol=0)

    def test_identical_packages_are_reported_and_not_given_hidden_ids(self):
        audit = source_package_audit(self.bundle, POOLS)
        self.assertIn(['DIFUSCO', 'DIFUSCO500'], audit['identical_packages'])
        self.assertIn(['T2T', 'T2T500'], audit['identical_in_pool']['TSP'])
        model, _ = self.model()
        model.eval()
        batch = sample_batch('TSP')
        scores = model(batch)['logits']
        names = POOLS['TSP']
        for a, b in audit['identical_in_pool']['TSP']:
            torch.testing.assert_close(scores[:, names.index(a)], scores[:, names.index(b)], atol=1e-6, rtol=1e-5)

    def test_labels_do_not_enter_forward(self):
        model, _ = self.model()
        model.eval()
        for p in ('TSP', 'CVRP', 'ATSP', 'OVRPTW'):
            batch = sample_batch(p)
            expected = model(inputs_only(batch))
            changed = dict(batch, costs=torch.full_like(batch['costs'], float('nan')), ind=torch.tensor([99, 99]))
            torch.testing.assert_close(model(changed)['logits'], expected['logits'], atol=0, rtol=0)

    def test_permutation_padding_and_single_candidate(self):
        model, _ = self.model()
        model.eval()
        for p in ('CVRP', 'ATSP'):
            batch = sample_batch(p)
            scores = model(batch)['logits']
            order = torch.randperm(len(batch['pool_ids']))
            torch.testing.assert_close(model(dict(batch, pool_ids=batch['pool_ids'][order]))['logits'],
                                       scores[:, order], atol=4e-6, rtol=4e-5)
            padded = dict(batch, node=F.pad(batch['node'], (0, 0, 0, 2), value=999.),
                          matrix=F.pad(batch['matrix'], (0, 2, 0, 2), value=999.),
                          node_mask=F.pad(batch['node_mask'], (0, 2), value=False))
            torch.testing.assert_close(model(padded)['logits'], scores, atol=4e-6, rtol=4e-5)
            ids = F.pad(batch['pool_ids'], (0, 2), value=-1)
            mask = torch.ones(2, len(ids), dtype=torch.bool)
            mask[:, -2:] = False
            actual = model(dict(batch, pool_ids=ids, solver_mask=mask))['logits']
            torch.testing.assert_close(actual[:, :-2], scores, atol=4e-6, rtol=4e-5)
            self.assertTrue(torch.isneginf(actual[:, -2:]).all())
            actual = model(dict(batch, pool_ids=batch['pool_ids'][:1]))['logits']
            torch.testing.assert_close(actual, torch.zeros_like(actual))

    def test_dynamic_states_and_pool_depend_on_instance(self):
        model, _ = self.model()
        model.eval()
        batch = sample_batch('CVRP')
        _, codes, h, _, solver_mask, token_mask = model.encode_source_state(batch)
        solvers, weights = model.source_pool(h, codes, token_mask)
        self.assertFalse(torch.allclose(codes[0], codes[1]))
        self.assertFalse(torch.allclose(solvers[0], solvers[1]))
        torch.testing.assert_close(weights.sum(-1), solver_mask.float())
        self.assertEqual(torch.count_nonzero(weights[~token_mask]), 0)

    def test_missing_roles_and_padding_never_enter_pool(self):
        source = dict(vectors=self.bundle['vectors'].clone(), role_mask=self.bundle['role_mask'].clone(),
                      chunk_vectors=self.bundle['chunk_vectors'], chunk_indices=self.bundle['chunk_indices'].clone())
        source['role_mask'][0, 2:] = False
        source['vectors'][0, 2:] = 0
        source['chunk_indices'][0, 2:] = -1
        encoder = SolverCodeTokens(**source, embedding_dim=16)
        ids = torch.tensor([[0, 1]])
        tokens, mask = encoder(ids, torch.tensor([[True, False]]))
        self.assertEqual(int(mask.sum()), 2)
        self.assertEqual(torch.count_nonzero(tokens[~mask]), 0)
        pool = DynamicSourcePool(16)
        h = torch.randn(1, 16)
        before, weights = pool(h, tokens, mask)
        changed = tokens.masked_fill(~mask[..., None], 999.)
        torch.testing.assert_close(pool(h, changed, mask)[0], before, atol=0, rtol=0)
        self.assertTrue(torch.isfinite(before).all())
        torch.testing.assert_close(weights.sum(-1), torch.tensor([[1., 0.]]))

    def test_two_dropout_passes_gradients_and_one_optimizer_update(self):
        model, _ = self.model(.1)
        batch = sample_batch('CVRP')
        k = len(batch['pool_ids'])
        costs = np.stack((np.arange(k)+1., np.arange(k)[::-1]+1.)).astype(np.float64)
        batch.update(comparison_targets(costs), ind=torch.tensor([0, k-1]))
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        scaler = torch.amp.GradScaler('cuda', enabled=False)
        logits = []
        hook = model.register_forward_hook(lambda module, args, output: logits.append(output['logits'].detach().clone()))
        update_batch(model, optimizer, scaler, batch, .35)
        hook.remove()
        self.assertEqual(len(logits), 2)
        self.assertFalse(torch.equal(logits[0], logits[1]))
        for name in ('solver_encoder.projection', 'source_pool', 'joint_layers', 'instance_encoder', 'decoder', 'score_head'):
            grads = [p.grad for p in dict(model.named_modules())[name].parameters() if p.grad is not None]
            self.assertTrue(grads and all(torch.isfinite(g).all() for g in grads), name)
            self.assertGreater(sum(float(g.abs().sum()) for g in grads), 0., name)
        self.assertEqual({int(s['step']) for s in optimizer.state.values()}, {1})

    def test_checkpoint_runs_offline_and_preserves_registry_identity(self):
        model, config = self.model()
        model.eval()
        batch = sample_batch('TSP')
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'fixture.pt'
            torch.save(dict(model=model.state_dict(), args=dict(group='A', solver_names=GLOBAL_SOLVERS,
                       model_params=config, embedding_provenance=bundle_provenance(self.bundle))), path)
            with patch.dict(os.environ, {}, clear=True), patch('urllib.request.urlopen', side_effect=AssertionError('No API')):
                replay, _ = load_r46_checkpoint(path)
            torch.testing.assert_close(replay(batch)['logits'], model(batch)['logits'], atol=0, rtol=0)
        state = copy.deepcopy(model.state_dict())
        state['solver_encoder.solver_identity'][0] ^= 1
        with self.assertRaises(ValueError):
            make_r46_model(config, state_dict=state)


if __name__ == '__main__':
    unittest.main()
