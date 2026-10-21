"""Five focused CPU tests of R87; no GPU jobs or real data reads."""

import copy
import unittest

import torch
import torch.nn.functional as F

from code.unified_selector.registry import M_GLOBAL, PROBLEMS
from .r77_data import parameter_hash
from .r87_model import ARMS, HIDDEN_DIM, initialize_from_fresh_common, make_model
from .test_r76 import fixture_batch, fixture_model


class R87ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        self.base, self.params = fixture_model(dropout=.1)
        reference = {name: value.detach().clone() for name, value in self.base.named_parameters()}
        self.initial = dict(seed=2, initialization='scratch', trainable=reference,
                            trainable_sha256=parameter_hash(reference), model_params=self.params)
        self.models = [initialize_from_fresh_common(self.initial, self.params, arm) for arm in ARMS]

    def test_zero_init_exact_original_outputs_and_dropout_rng_all18(self):
        self.base.train()
        for problem in PROBLEMS:
            batch, _ = fixture_batch(problem, 2)
            torch.manual_seed(317)
            expected = self.base(batch)
            rng = torch.get_rng_state().clone()
            for model in self.models:
                with self.subTest(problem=problem, arm=model.arm):
                    model.train()
                    torch.manual_seed(317)
                    actual = model(batch)
                    self.assertEqual(set(actual), set(expected))
                    for key in expected:
                        torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
                    torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)

    def test_parameter_matched_heads_identical_W_blocks_and_reserved_id17(self):
        shared, private = self.models
        expected = M_GLOBAL * HIDDEN_DIM * (3 * self.params['embedding_dim'] + 1)
        base_params = dict(self.base.named_parameters())
        counts = []
        for model in self.models:
            parameters = dict(model.named_parameters())
            self.assertEqual(set(parameters) - set(base_params), {'residual.W', 'residual.v'})
            for name, tensor in base_params.items():
                torch.testing.assert_close(parameters[name], tensor, rtol=0, atol=0)
            self.assertEqual(model.residual.v.count_nonzero().item(), 0)
            report = model.parameter_report()
            self.assertEqual(report['residual_parameters'], expected)
            self.assertEqual(report['inactive_candidate_ids'], [17])
            self.assertFalse(report['candidate_pool_changed'])
            counts.append(report['total_parameters'])
        self.assertEqual(counts[0], counts[1])
        bank = shared.residual.W.reshape(-1, M_GLOBAL, HIDDEN_DIM).permute(1, 0, 2)
        torch.testing.assert_close(bank, private.residual.W, rtol=0, atol=0)
        self.assertEqual(private.parameter_report()['inactive_private_parameters'], expected // M_GLOBAL)
        self.assertEqual(shared.parameter_report()['inactive_private_parameters'], 0)
        with torch.no_grad():
            private.residual.v.fill_(.1)
        features = torch.randn(2, 3, 3 * self.params['embedding_dim'])
        mask = torch.ones(2, 3, dtype=torch.bool)
        ids = torch.tensor([0, 3, 7])
        before = private.residual(features, ids, mask)
        with torch.no_grad():
            private.residual.W[3].mul_(2.)
        after = private.residual(features, ids, mask)
        torch.testing.assert_close(after[:, [0, 2]], before[:, [0, 2]], rtol=0, atol=0)
        self.assertFalse(torch.equal(after[:, 1], before[:, 1]))

    def test_candidate_swap_padding_labels_and_2d_ids_after_nonzero_residual(self):
        batch, _ = fixture_batch('TSP', 3)
        k = len(batch['pool_ids'])
        for model in self.models:
            with self.subTest(arm=model.arm):
                model.eval()
                with torch.no_grad():
                    model.residual.v.copy_(torch.linspace(-.2, .3, model.residual.v.numel()).reshape_as(model.residual.v))
                original = model(batch)
                ids = torch.cat((batch['pool_ids'], torch.tensor([-1, 100000])))
                mask = torch.ones(3, k + 2, dtype=torch.bool)
                mask[:, -2:] = False
                padded = dict(batch, pool_ids=ids, solver_mask=mask)
                result = model(padded)
                torch.testing.assert_close(result['logits'][:, :k], original['logits'], rtol=3e-5, atol=3e-6)
                self.assertTrue(torch.isneginf(result['logits'][:, -2:]).all())
                self.assertEqual(result['pair_margin'][:, -2:].count_nonzero().item(), 0)
                self.assertEqual(result['pair_margin'][:, :, -2:].count_nonzero().item(), 0)
                torch.testing.assert_close(model(dict(padded, pool_ids=ids[None].expand(3, -1)))['logits'],
                                           result['logits'], rtol=3e-5, atol=3e-6)
                order = torch.arange(k + 1, -1, -1)
                permuted = model(dict(padded, pool_ids=ids[order], solver_mask=mask[:, order]))
                torch.testing.assert_close(permuted['logits'], result['logits'][:, order], rtol=3e-5, atol=3e-6)
                changed = copy.deepcopy(padded)
                for key in ('costs', 'cost_regret', 'cmp_target'):
                    changed[key].fill_(float('nan'))
                changed['ind'].fill_(-123)
                changed['cmp_valid'].fill_(False)
                for key, value in model(changed).items():
                    torch.testing.assert_close(value, result[key], rtol=0, atol=0)

    def test_private_and_shared_residual_gradients_and_original_encoder_remain_trainable(self):
        batch, _ = fixture_batch('TSP', 4)
        for model in self.models:
            with self.subTest(arm=model.arm):
                model.train()
                optimizer = torch.optim.SGD(model.parameters(), lr=.05)
                first = model(batch)
                F.cross_entropy(first['logits'], batch['ind']).backward()
                self.assertGreater(model.residual.v.grad.abs().sum().item(), 0.)
                # With zero v, W begins learning only after the first v update.
                self.assertEqual(model.residual.W.grad.count_nonzero().item(), 0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                F.cross_entropy(model(batch)['logits'], batch['ind']).backward()
                self.assertGreater(model.residual.W.grad.abs().sum().item(), 0.)
                for module in (model.residual, model.instance_encoder.coord_proj, model.solver_encoder,
                               *model.joint_layers, model.pool, model.decoder, model.score_head):
                    gradients = [p.grad for p in module.parameters() if p.grad is not None]
                    self.assertTrue(gradients)
                    self.assertTrue(all(torch.isfinite(g).all() for g in gradients))
                    self.assertGreater(sum(g.abs().sum().item() for g in gradients), 0.)
                if model.arm == 'B_private_residual':
                    self.assertEqual(model.residual.W.grad[17].count_nonzero().item(), 0)
                    self.assertEqual(model.residual.v.grad[17].count_nonzero().item(), 0)

    def test_strict_checkpoint_roundtrip_and_fresh_contract_only(self):
        batch, _ = fixture_batch('CVRP', 3)
        for model in self.models:
            model.eval()
            with torch.no_grad():
                model.residual.v.fill_(.2)
            restored = make_model(model.params, model.arm, state_dict=model.state_dict()).eval()
            for key, value in model(batch).items():
                torch.testing.assert_close(restored(batch)[key], value, rtol=0, atol=0)
            with self.assertRaises(RuntimeError):
                make_model(model.params, model.arm, state_dict=self.base.state_dict())
        bad = copy.deepcopy(self.initial)
        bad['trainable']['score_head.3.weight'].add_(.1)
        cases = [self.base.state_dict(), dict(model=self.base.state_dict(), update=100), bad,
                 dict(self.initial, seed=3), dict(self.initial, initialization='resume')]
        for value in cases:
            with self.assertRaises(ValueError):
                initialize_from_fresh_common(value, self.params, ARMS[0])
        missing = copy.deepcopy(self.initial)
        missing['trainable'].pop('score_head.3.weight')
        missing['trainable_sha256'] = parameter_hash(missing['trainable'])
        with self.assertRaises(ValueError):
            initialize_from_fresh_common(missing, self.params, ARMS[1])
        with self.assertRaises(ValueError):
            make_model(self.models[0].params, ARMS[1])


if __name__ == '__main__':
    unittest.main()
