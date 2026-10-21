"""Focused CPU tests of direct attention scoring; no datasets or GPU jobs."""

import copy
import io
import math
import unittest

import torch
import torch.nn.functional as F

from code.unified_selector.registry import PROBLEMS
from .V4Model import MultiHeadAttention
from .pairwise_objective import inputs_only
from .r76_experiment import RDropObjective
from .r84_attention_selector import AttentionLogitDecoder, make_r84_model
from .test_r76 import fixture_batch, fixture_model


class R84AttentionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        self.baseline, self.params = fixture_model(dropout=0.)
        torch.manual_seed(2)
        self.model = make_r84_model(self.params).eval()

    def test_exact_head_mean_logits_and_not_probability_as_logits(self):
        decoder = AttentionLogitDecoder(MultiHeadAttention(4, 2, 2, 0.))
        with torch.no_grad():
            decoder.Wq.weight.copy_(torch.eye(4))
            decoder.Wk.weight.copy_(torch.eye(4))
        query = torch.tensor([[[2., 1., -1., 3.]]])
        methods = torch.tensor([[[1., 0., 2., 0.], [0., 3., -1., 1.], [2., -1., 0., 2.]]])
        heads = torch.stack([
            (query[:, 0, :2, None] * methods[:, :, :2].transpose(1, 2)).sum(1),
            (query[:, 0, 2:, None] * methods[:, :, 2:].transpose(1, 2)).sum(1),
        ], 1) / math.sqrt(2.)
        logits = decoder(query, methods)
        torch.testing.assert_close(logits, heads.mean(1), rtol=0., atol=0.)
        self.assertFalse(torch.allclose(logits, heads.softmax(-1).mean(1)))
        self.assertFalse(torch.allclose(logits.softmax(-1), heads.softmax(-1).mean(1)))

        batch, _ = fixture_batch('TSP', 3)
        out = self.model(batch)
        _, solvers, h, _, _ = self.model.encode_state(batch)
        expected = self.model.decoder(h[:, None], solvers)
        torch.testing.assert_close(out['logits'], expected)
        torch.testing.assert_close(out['probabilities'], expected.softmax(-1))
        labels = batch['ind']
        ce = F.cross_entropy(out['logits'], labels)
        nll = -out['probabilities'][torch.arange(3), labels].log().mean()
        torch.testing.assert_close(ce, nll)
        self.assertFalse(torch.allclose(ce, F.cross_entropy(out['probabilities'], labels)))

    def test_backbone_and_qk_keep_fresh_baseline_initialization(self):
        old = dict(self.baseline.named_parameters())
        new = dict(self.model.named_parameters())
        removed = {name for name in old if name.startswith(('score_head.', 'decoder.Wv.', 'decoder.out.'))}
        self.assertEqual(set(new), set(old) - removed)
        for name, value in new.items():
            with self.subTest(parameter=name):
                torch.testing.assert_close(value, old[name], rtol=0., atol=0.)
        self.assertFalse(hasattr(self.model, 'score_head'))
        self.assertEqual(set(dict(self.model.decoder.named_parameters())), {'Wq.weight', 'Wk.weight'})

    def test_all18_tasks_use_direct_probabilities_and_score_difference_margins(self):
        for problem in PROBLEMS:
            with self.subTest(problem=problem), torch.no_grad():
                batch, _ = fixture_batch(problem, 3)
                out = self.model(inputs_only(batch))
                self.assertTrue(torch.isfinite(out['logits']).all())
                torch.testing.assert_close(out['probabilities'].sum(-1), torch.ones(3))
                torch.testing.assert_close(out['utility'], out['logits'])
                difference = out['logits'][:, :, None] - out['logits'][:, None, :]
                torch.testing.assert_close(out['pair_margin'], difference)

    def test_candidate_permutation_padding_and_single_candidate(self):
        batch, _ = fixture_batch('TSP', 3)
        base = self.model(batch)
        count = len(batch['pool_ids'])
        ids = torch.cat([batch['pool_ids'], torch.tensor([-1, 100000])])
        mask = torch.ones(3, len(ids), dtype=torch.bool)
        mask[:, -2:] = False
        padded = self.model(dict(batch, pool_ids=ids, solver_mask=mask))
        torch.testing.assert_close(padded['logits'][:, :count], base['logits'], atol=3e-6, rtol=2e-5)
        self.assertTrue(torch.isneginf(padded['logits'][:, -2:]).all())
        self.assertEqual(torch.count_nonzero(padded['probabilities'][:, -2:]).item(), 0)
        self.assertEqual(torch.count_nonzero(padded['pair_margin'][:, -2:]).item(), 0)
        self.assertEqual(torch.count_nonzero(padded['pair_margin'][:, :, -2:]).item(), 0)
        order = torch.randperm(len(ids))
        permuted = self.model(dict(batch, pool_ids=ids[order], solver_mask=mask[:, order]))
        torch.testing.assert_close(permuted['logits'], padded['logits'][:, order], atol=3e-6, rtol=2e-5)
        torch.testing.assert_close(permuted['probabilities'], padded['probabilities'][:, order], atol=3e-6, rtol=2e-5)
        one_mask = torch.zeros_like(mask)
        one_mask[0, 0] = one_mask[1, 2] = one_mask[2, 4] = True
        singleton = self.model(dict(batch, pool_ids=ids, solver_mask=one_mask))
        torch.testing.assert_close(singleton['probabilities'], one_mask.float(), rtol=0., atol=0.)
        with self.assertRaises(ValueError):
            self.model(dict(batch, solver_mask=torch.zeros(3, count, dtype=torch.bool)))

    def test_labels_do_not_enter_forward_and_padding_does_not_change_scores(self):
        for problem in ('TSP', 'ATSP'):
            with self.subTest(problem=problem):
                batch, _ = fixture_batch(problem, 3)
                original = self.model(batch)['logits']
                changed = copy.deepcopy(batch)
                changed['costs'].fill_(float('nan'))
                changed['ind'].fill_(10000)
                changed['cost_regret'].fill_(float('nan'))
                changed['cmp_target'].fill_(float('nan'))
                torch.testing.assert_close(self.model(changed)['logits'], original, rtol=0., atol=0.)
                changed = dict(batch, node=F.pad(batch['node'], (0, 0, 0, 3), value=123.),
                    matrix=F.pad(batch['matrix'], (0, 3, 0, 3), value=123.),
                    node_geom=F.pad(batch['node_geom'], (0, 0, 0, 3), value=123.),
                    node_mask=F.pad(batch['node_mask'], (0, 3), value=False))
                torch.testing.assert_close(self.model(changed)['logits'], original, atol=3e-6, rtol=2e-5)

    def test_original_training_objective_reaches_encoders_joint_layers_and_qk(self):
        _, params = fixture_model(dropout=.1)
        model = make_r84_model(params).train()
        for problem in ('TSP', 'ATSP'):
            model.zero_grad(set_to_none=True)
            batch, _ = fixture_batch(problem, 4)
            loss, parts = RDropObjective(model)(batch, .35)
            self.assertTrue(torch.isfinite(loss))
            self.assertTrue(torch.isfinite(parts).all())
            loss.backward()
            projection = model.instance_encoder.coord_proj if problem == 'TSP' else model.instance_encoder.matrix_proj
            modules = [projection, model.solver_encoder, model.pool, model.decoder, *model.joint_layers]
            for module in modules:
                grads = [value.grad for value in module.parameters() if value.grad is not None]
                self.assertTrue(grads)
                self.assertTrue(all(torch.isfinite(value).all() for value in grads))
                self.assertGreater(sum(value.abs().sum().item() for value in grads), 0.)
            for projection in (model.decoder.Wq, model.decoder.Wk):
                self.assertGreater(projection.weight.grad.abs().sum().item(), 0.)

    def test_strict_offline_state_roundtrip_and_explicit_architecture_guard(self):
        memory = io.BytesIO()
        torch.save(dict(model=self.model.state_dict(), params=self.model.params), memory)
        memory.seek(0)
        state = torch.load(memory, weights_only=False)
        restored = make_r84_model(state['params'], state['model']).eval()
        batch, _ = fixture_batch('CVRP', 3)
        torch.testing.assert_close(restored(batch)['logits'], self.model(batch)['logits'], rtol=0., atol=0.)
        self.assertEqual(restored.params['selection_head'], 'direct_attention')
        with self.assertRaises(RuntimeError):
            make_r84_model(self.params, self.baseline.state_dict())
        with self.assertRaises(ValueError):
            make_r84_model(dict(self.params, solver_representation='code'))
        with self.assertRaises(ValueError):
            make_r84_model(dict(self.params, selection_head='maximin'))

    def test_amp_dot_products_reduce_in_fp32(self):
        batch, _ = fixture_batch('TSP', 3)
        with torch.autocast(device_type='cpu', dtype=torch.bfloat16):
            out = self.model(batch)
        self.assertEqual(out['logits'].dtype, torch.float32)
        self.assertEqual(out['probabilities'].dtype, torch.float32)
        self.assertTrue(torch.isfinite(out['logits']).all())


if __name__ == '__main__':
    unittest.main()
