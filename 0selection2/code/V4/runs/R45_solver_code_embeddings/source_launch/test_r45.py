"""Offline checks only; synthetic vectors are fixtures, never production artifacts."""

import copy
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from urllib.error import HTTPError
from unittest.mock import patch

import numpy as np
import torch
import torch.nn.functional as F

from code.unified_selector.registry import GLOBAL_SOLVERS, M_GLOBAL
from .pairwise_objective import comparison_targets, inputs_only, single_objective, predicted_top3
from .pairwise_selector import make_r43_model
from .r43_experiment import policy_metrics
from .solver_code_corpus import MANIFEST, MAX_TOKENS, ast_units, bounded_parts, build_corpus, check_sensitive, digest, semantic_config, source_path, write_json
from .solver_code_embeddings import SETTINGS, api_key, cache_key, generate, l2mean, load_bundle, request_embeddings, validate_bundle, validate_corpus
from .solver_code_encoder import CODE_DIM, ROLES, SolverCodeEncoder, derangement, make_r45_model, load_r45_checkpoint
from .test_dual_stream import params
from .test_solver_features import sample_batch


def fixture_bundle():
    generator = torch.Generator().manual_seed(4500)
    vectors = F.normalize(torch.randn(M_GLOBAL, 4, CODE_DIM, generator=generator), dim=-1)
    mask = torch.ones(M_GLOBAL, 4, dtype=torch.bool)
    mask[0, 2:] = False
    vectors[~mask] = 0
    return dict(kind='unit_test_only', vectors=vectors, role_mask=mask)


def fixture_model(mode='code', dropout=0.):
    torch.manual_seed(2)
    config = dict(params(), pair_mode='score_difference', solver_representation=mode,
                  code_init_seed=450002, permutation_seed=4502, dropout=dropout)
    # Only this test helper bypasses API provenance. No production CLI supports it.
    with patch('code.V4.solver_code_embeddings.validate_bundle'):
        model = make_r45_model(config, fixture_bundle())
    return model, config


class CharacterTokenizer:
    """Deliberately fake tokenizer used only to exercise the chunking algorithm."""
    fingerprint = 'unit-test-not-a-production-tokenizer'

    def count(self, text):
        return len(text) + 2


class R45Tests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)

    def test_original_handcrafted_path_is_unchanged(self):
        torch.manual_seed(2)
        config = dict(params(), pair_mode='score_difference')
        original = make_r43_model(config).eval()
        torch.manual_seed(2)
        actual = make_r45_model(config).eval()
        actual.load_state_dict(original.state_dict(), strict=True)
        for p in ('TSP', 'CVRP', 'ATSP', 'OVRPTW'):
            batch = sample_batch(p)
            torch.testing.assert_close(actual(batch)['logits'], original(batch)['logits'], atol=0, rtol=0)

    def test_common_initial_values_and_shuffled_ownership(self):
        a, _ = fixture_model('handcrafted')
        b, _ = fixture_model('code')
        c, _ = fixture_model('shuffled_code')
        bp, cp = dict(b.named_parameters()), dict(c.named_parameters())
        self.assertEqual(set(bp), set(cp))
        for key in bp:
            torch.testing.assert_close(bp[key], cp[key], atol=0, rtol=0)
        for key, value in a.named_parameters():
            if key in bp:
                torch.testing.assert_close(value, bp[key], atol=0, rtol=0)
        self.assertFalse(hasattr(b.solver_encoder, 'feature_mlp'))
        expected = derangement()
        self.assertTrue((expected != torch.arange(M_GLOBAL)).all())
        torch.testing.assert_close(c.solver_encoder.semantic_rows, expected, atol=0, rtol=0)
        torch.testing.assert_close(b.solver_encoder.code_vectors, c.solver_encoder.code_vectors, atol=0, rtol=0)
        torch.testing.assert_close(b.solver_encoder.role_mask, c.solver_encoder.role_mask, atol=0, rtol=0)

    def test_no_labels_and_full_registered_pools(self):
        model, _ = fixture_model()
        model.eval()
        for p in ('TSP', 'CVRP', 'ATSP', 'OVRPTW'):
            batch = sample_batch(p)
            clean = inputs_only(batch)
            first = model(clean)
            changed = dict(batch, costs=torch.full_like(batch['costs'], float('nan')), ind=torch.tensor([99, 99]))
            torch.testing.assert_close(model(changed)['logits'], first['logits'], atol=0, rtol=0)
            torch.testing.assert_close(first['pair_margin'], -first['pair_margin'].transpose(-1, -2), atol=0, rtol=0)
            self.assertTrue(torch.isfinite(first['logits']).all())

    def test_mask_permutation_padding_and_single_solver(self):
        model, _ = fixture_model()
        model.eval()
        for p in ('CVRP', 'ATSP'):
            batch = sample_batch(p)
            original = model(batch)['logits']
            order = torch.randperm(len(batch['pool_ids']))
            actual = model(dict(batch, pool_ids=batch['pool_ids'][order]))['logits']
            torch.testing.assert_close(actual, original[:, order], atol=4e-6, rtol=4e-5)
            padded = copy.deepcopy(batch)
            padded['node'] = F.pad(batch['node'], (0, 0, 0, 2), value=999.)
            padded['matrix'] = F.pad(batch['matrix'], (0, 2, 0, 2), value=999.)
            padded['node_mask'] = F.pad(batch['node_mask'], (0, 2), value=False)
            torch.testing.assert_close(model(padded)['logits'], original, atol=4e-6, rtol=4e-5)
            ids = torch.cat((batch['pool_ids'], torch.tensor([-1, -1])))
            mask = torch.ones(2, len(ids), dtype=torch.bool)
            mask[:, -2:] = False
            scores = model(dict(batch, pool_ids=ids, solver_mask=mask))['logits']
            torch.testing.assert_close(scores[:, :-2], original, atol=4e-6, rtol=4e-5)
            self.assertTrue(torch.isneginf(scores[:, -2:]).all())
            one = model(dict(batch, pool_ids=batch['pool_ids'][:1]))['logits']
            torch.testing.assert_close(one, torch.zeros_like(one))

    def test_gradients_live_and_buffers_fixed(self):
        model, _ = fixture_model(dropout=.1)
        batch = sample_batch('CVRP')
        k = len(batch['pool_ids'])
        costs = np.stack((np.arange(k) + 1., np.arange(k)[::-1] + 1.)).astype(np.float64)
        batch.update(comparison_targets(costs))
        batch['ind'] = torch.tensor([0, k - 1])
        first, second = model(inputs_only(batch)), model(inputs_only(batch))
        self.assertFalse(torch.equal(first['logits'], second['logits']))
        focus = predicted_top3(first['logits'], second['logits'], batch['pool_ids'], first['solver_mask'])
        loss, parts = single_objective(first, batch, focus)
        torch.testing.assert_close(loss, .35 * parts['ce'] + .35 * parts['cmp'] + .02 * parts['risk'])
        loss.backward()
        modules = [*model.solver_encoder.role_proj, model.solver_encoder.fusion, model.solver_encoder.solver_emb,
                   model.instance_encoder.coord_proj, *model.joint_layers, model.decoder, model.score_head]
        for module in modules:
            self.assertGreater(sum(float(p.grad.abs().sum()) for p in module.parameters() if p.grad is not None), 0)
        buffers = dict(model.solver_encoder.named_buffers())
        self.assertTrue({'code_vectors', 'role_mask', 'semantic_rows', 'solver_identity'} <= set(buffers))
        for buffer in buffers.values():
            self.assertFalse(buffer.requires_grad)
            self.assertIsNone(buffer.grad)

    def test_absent_role_is_masked_after_projection(self):
        bundle = fixture_bundle()
        config = params()
        encoder = SolverCodeEncoder(bundle['vectors'], bundle['role_mask'], torch.arange(M_GLOBAL), **config).eval()
        before = encoder(torch.tensor([0]))
        with torch.no_grad():
            encoder.role_proj[2][1].bias.fill_(100.)
            encoder.role_proj[3][1].bias.fill_(-100.)
        torch.testing.assert_close(encoder(torch.tensor([0])), before, atol=0, rtol=0)

    def test_checkpoint_is_self_contained_without_api_or_bundle(self):
        model, config = fixture_model('shuffled_code')
        model.eval()
        batch = sample_batch('TSP')
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'test_fixture.pt'
            torch.save(dict(model=model.state_dict(), args=dict(model_params=config, solver_names=GLOBAL_SOLVERS)), path)
            with patch.dict(os.environ, {}, clear=True), patch('urllib.request.urlopen', side_effect=AssertionError('No network allowed')):
                with self.assertRaises(ValueError):
                    load_r45_checkpoint(path)  # A synthetic fixture is not a production checkpoint.
                with patch('code.V4.solver_code_embeddings.validate_bundle'):
                    replay, _ = load_r45_checkpoint(path)
            torch.testing.assert_close(replay(batch)['logits'], model(batch)['logits'], atol=0, rtol=0)
            state = copy.deepcopy(model.state_dict())
            state['solver_encoder.solver_identity'][0] ^= 1
            with self.assertRaises(ValueError):
                make_r45_model(config, state_dict=state)

    def test_production_refuses_fake_or_misaligned_embeddings(self):
        with self.assertRaises(ValueError):
            validate_bundle(fixture_bundle())
        with self.assertRaises(ValueError):
            make_r45_model(dict(params(), pair_mode='score_difference', solver_representation='code'), fixture_bundle())
        source = fixture_bundle()
        with self.assertRaises(ValueError):
            SolverCodeEncoder(source['vectors'][:-1], source['role_mask'], torch.arange(M_GLOBAL), **params())
        source['role_mask'][1, 0] = False
        with self.assertRaises(ValueError):
            SolverCodeEncoder(source['vectors'], source['role_mask'], torch.arange(M_GLOBAL), **params())
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'fake.pt'
            torch.save(fixture_bundle(), path)
            write_json(path.with_suffix('.lock.json'), dict(sha256='0' * 64))
            with self.assertRaises(ValueError):
                load_bundle(path)

    def test_ast_extraction_preserves_complete_function_and_scrubs_paths(self):
        source = '''def solve(x):
    """Not semantic input."""
    print("logging")
    if x:
        return 1
    path = "/private/checkpoint.pt"
    return 2
'''
        unit = ast_units(source, ['solve'])[0]
        self.assertIn('return 2', unit['text'])
        self.assertNotIn('print(', unit['text'])
        self.assertNotIn('/private', unit['text'])
        self.assertNotIn('Not semantic', unit['text'])
        self.assertEqual(unit['start'], 1)
        self.assertEqual(unit['end'], 7)
        with self.assertRaises(ValueError):
            ast_units(source, ['not_a_symbol'])
        with self.assertRaises(ValueError):
            check_sensitive('api_key = "unit-test-secret-long-value"')
        with self.assertRaises(ValueError):
            semantic_config({'api_key': 'unit-test-secret-long-value'})
        with self.assertRaises(ValueError):
            semantic_config({'mean_cost': 1.234})

    def test_chunk_boundaries_retain_long_tail(self):
        tokenizer = CharacterTokenizer()
        source = 'def solve(x):\n' + '\n'.join(f'    x = x + {i}' for i in range(700)) + '\n    return x'
        parts = bounded_parts(source, tokenizer, limit=512)
        self.assertGreater(len(parts), 1)
        self.assertTrue(all(tokenizer.count(p) <= 512 for p in parts))
        joined = '\n'.join(parts)
        for i in range(700):
            self.assertIn(f'x = x + {i}\n', joined)
        self.assertIn('return x', joined)
        long = bounded_parts('x = "' + 'a' * 10000 + '"\nreturn x', tokenizer, limit=512)
        self.assertTrue(all(tokenizer.count(p) <= 512 for p in long))
        self.assertIn('return x', long[-1])

    def test_manifest_identity_and_real_source_symbols(self):
        manifest = json.loads(MANIFEST.read_text())
        self.assertEqual([r['solver_name'] for r in manifest['solvers']], GLOBAL_SOLVERS)
        for impl in manifest['implementations'].values():
            self.assertEqual(set(impl['views']), set(ROLES))
            for spec in impl['views'].values():
                for selected in spec.get('sources', []):
                    path = source_path(manifest, selected['file'])
                    self.assertTrue(path.exists(), str(path))
                    self.assertTrue(ast_units(path.read_text(encoding='utf-8-sig'), selected['symbols']))
        with self.assertRaises(ValueError):
            source_path(manifest, 'easynco:../.ssh/id_rsa')

    def test_cache_invalidates_changed_source_and_preprocessing(self):
        chunk = dict(id='a', text_sha256='b')
        corpus = dict(preprocessing='v1', tokenizer_sha256='c', max_tokens=MAX_TOKENS, manifest_sha256='d')
        key = cache_key(chunk, corpus)
        self.assertNotEqual(key, cache_key(dict(chunk, id='changed_source'), corpus))
        self.assertNotEqual(key, cache_key(chunk, dict(corpus, preprocessing='v2')))
        a = np.array([[1., 0.], [0., 2.]])
        np.testing.assert_allclose(l2mean(a), [2 ** -.5, 2 ** -.5])

    def test_draft_corpus_and_budget_refuse_upload_before_network(self):
        manifest = json.loads(MANIFEST.read_text())
        corpus = build_corpus(manifest, CharacterTokenizer(), draft=True)
        self.assertFalse(corpus['ready'])
        self.assertEqual(len(corpus['solvers']), M_GLOBAL)
        with tempfile.TemporaryDirectory() as temp, patch('urllib.request.urlopen', side_effect=AssertionError('No request allowed')):
            cache = Path(temp) / 'cache'
            plan = generate(corpus, cache)
            self.assertGreater(plan['uncached_tokens'], 0)
            with self.assertRaises(ValueError):
                generate(corpus, cache, allow_upload=True, max_tokens=10000000)
            self.assertFalse(cache.exists())
        altered = copy.deepcopy(corpus)
        altered['chunks'][0]['text'] += 'changed'
        with self.assertRaises(ValueError):
            validate_corpus(altered)

    def test_api_key_is_external_private_and_missing_is_not_silent(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(ValueError):
                api_key()
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'private_key'
            path.write_text('unit-test-secret-never-used')
            path.chmod(0o644)
            with self.assertRaises(ValueError):
                api_key(path)
            path.chmod(0o600)
            self.assertEqual(api_key(path), 'unit-test-secret-never-used')

    def test_http_payload_response_order_and_error_redaction(self):
        class Response(io.StringIO):
            headers = {'x-request-id': 'unit-test-request'}
        result = dict(model='voyage-code-4', usage=dict(total_tokens=7),
                      data=[dict(index=1, embedding=[2.] * CODE_DIM), dict(index=0, embedding=[1.] * CODE_DIM)])
        with patch('urllib.request.urlopen', return_value=Response(json.dumps(result))) as request:
            vectors, metadata = request_embeddings(['def a(): pass', 'def b(): pass'], 'unit-test-only')
        payload = json.loads(request.call_args.args[0].data)
        self.assertEqual({key: payload[key] for key in SETTINGS}, SETTINGS)
        self.assertEqual(payload['input'], ['def a(): pass', 'def b(): pass'])
        self.assertEqual(vectors.shape, (2, CODE_DIM))
        self.assertEqual(vectors[0, 0], 1.)
        self.assertEqual(metadata['request_id'], 'unit-test-request')
        error = HTTPError('https://api.voyageai.com', 401, 'unit-test-secret', {}, None)
        with patch('urllib.request.urlopen', side_effect=error):
            with self.assertRaisesRegex(RuntimeError, 'Voyage HTTP 401') as raised:
                request_embeddings(['def a(): pass'], 'unit-test-only')
        self.assertNotIn('unit-test-secret', str(raised.exception))

    def test_budget_is_checked_before_any_paid_request(self):
        corpus = build_corpus(json.loads(MANIFEST.read_text()), CharacterTokenizer(), draft=True)
        # Exercise only the pre-request budget guard, never export this fixture.
        corpus['ready'], corpus['issues'] = True, []
        corpus['sha256'] = digest({k: v for k, v in corpus.items() if k != 'sha256'})
        with tempfile.TemporaryDirectory() as temp, patch('urllib.request.urlopen', side_effect=AssertionError('No request allowed')):
            with self.assertRaisesRegex(ValueError, 'budget'):
                generate(corpus, Path(temp), allow_upload=True, max_tokens=1)

    def test_saved_prediction_metrics_replay(self):
        model, _ = fixture_model()
        scores = model.eval()(sample_batch('TSP'))['logits'].detach().numpy()
        k = scores.shape[-1]
        raw = dict(costs=np.tile(np.arange(k) + 1., (2, 1)).astype(np.float64), winner=np.zeros(2, dtype=int),
                   pool=[GLOBAL_SOLVERS[i] for i in sample_batch('TSP')['pool_ids']],
                   pool_ids=sample_batch('TSP')['pool_ids'].numpy())
        expected, pred = policy_metrics(scores, raw)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'unit_predictions.npz'
            np.savez_compressed(path, logits=scores, **raw)
            with np.load(path) as saved:
                replay, replay_pred = policy_metrics(saved['logits'], {key: saved[key] for key in raw})
        np.testing.assert_array_equal(pred, replay_pred)
        for key in ('top1', 'mean_cost', 'actual_regret_pct'):
            self.assertEqual(expected[key], replay[key])


if __name__ == '__main__':
    unittest.main()
