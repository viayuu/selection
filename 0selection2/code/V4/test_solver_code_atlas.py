"""Offline Atlas and whole-view regressions; no paid requests or result fixtures."""

import contextlib
import copy
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
import torch.nn.functional as F

from code.unified_selector.registry import GLOBAL_SOLVERS, M_GLOBAL
from .solver_code_corpus import ROOT, digest, write_json
from .solver_code_embeddings import (ENDPOINT, SETTINGS, assemble, bundle_provenance, cache_key,
                                     generate, load_bundle, request_embeddings, validate_corpus)
from .solver_code_encoder import CODE_DIM, ROLES, SolverCodeEncoder, load_r45_checkpoint, make_r45_model
from .solver_code_views import extract_components, pack_view
from .test_dual_stream import params
from .test_r45 import CharacterTokenizer, fixture_bundle
from .test_solver_features import sample_batch


def components(source, symbols):
    return extract_components(source, symbols, 'unit_test:source.py', digest(source.encode()))


def bank_fixture(overflow=True):
    bundle = fixture_bundle()
    vectors = bundle['vectors'].clone()
    bank = vectors.reshape(-1, CODE_DIM)
    indices = torch.arange(M_GLOBAL * 4).reshape(M_GLOBAL, 4, 1, 1)
    indices[~bundle['role_mask']] = -1
    bank = F.normalize(bank + 1e-4, dim=-1)
    if overflow:
        indices = F.pad(indices, (0, 1), value=-1)
        indices[1, :, 0, 1] = indices[2, :, 0, 0]
    bundle.update(chunk_vectors=bank, chunk_indices=indices)
    return bundle


class AtlasTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)

    def test_complete_view_larger_than_4096_is_not_split(self):
        source = 'class Encoder:\n    def forward(self, x):\n' + ''.join(f'        x = x + {i}\n' for i in range(500)) + '        return x\n'
        selected = components(source, ['Encoder', 'Encoder.forward'])
        self.assertEqual(len(selected), 1)
        self.assertIn('class Encoder:', selected[0]['text'])
        parts, tokens = pack_view(selected, CharacterTokenizer(), 'encoder')
        self.assertGreater(tokens, 4096)
        self.assertLess(tokens, 32000)
        self.assertEqual(len(parts), 1)
        self.assertTrue(parts[0]['complete_view'])
        self.assertIn('return x', parts[0]['text'])

    def test_overflow_preserves_functions_classes_calls_and_tail(self):
        source = 'class Encoder:\n    def forward(self, x):\n' + ''.join(f'        x = x + {i}\n' for i in range(400)) + '        return helper(x)\n'
        source += '\ndef helper(x):\n    return x + 9999\n'
        parts, tokens = pack_view(components(source, ['Encoder', 'helper']), CharacterTokenizer(), 'encoder', limit=1800)
        self.assertGreater(tokens, 1800)
        self.assertGreater(len(parts), 1)
        self.assertTrue(all(p['tokens'] <= 1800 and not p['complete_view'] for p in parts))
        text = '\n'.join(p['text'] for p in parts)
        for index in range(400):
            self.assertIn(f'x = x + {index}\n', text)
        self.assertIn('return helper(x)', text)
        self.assertIn('return x + 9999', text)
        self.assertIn('Selected', text.title())
        self.assertTrue(all(p['provenance'] for p in parts))

    def test_atomic_oversize_is_reported_not_character_truncated(self):
        source = 'def solve():\n    return "' + 'a' * 4000 + '"\n'
        with self.assertRaisesRegex(ValueError, 'Atomic statement'):
            pack_view(components(source, ['solve']), CharacterTokenizer(), 'decision', limit=2000)

    def test_oversized_function_keeps_callable_identity_and_control_blocks(self):
        source = 'def caller(x):\n    return helper(x)\n\ndef helper(x):\n    if x > 0:\n'
        source += ''.join(f'        x = x + {i}\n' for i in range(160))
        source += '    else:\n        x = x - 9999\n    return x\n'
        parts, _ = pack_view(components(source, ['caller', 'helper']), CharacterTokenizer(), 'decision', limit=1400)
        self.assertGreater(len(parts), 1)
        self.assertTrue(all(p['tokens'] <= 1400 for p in parts))
        text = '\n'.join(p['text'] for p in parts)
        self.assertIn('caller -> helper', text)
        self.assertIn('if x > 0:', text)
        self.assertIn('x = x - 9999', text)
        self.assertIn('return x', text)
        for index in range(160):
            self.assertIn(f'x = x + {index}\n', text)
        self.assertTrue(any(c['origin_symbol'] == 'helper' for p in parts for c in p['components']))

    def test_class_initialization_is_retained_and_included_in_token_budget(self):
        source = 'class Encoder:\n    def __init__(self):\n        self.width = 128\n    def forward(self, x):\n'
        source += ''.join(f'        x = x + {i}\n' for i in range(200)) + '        return x\n'
        parts, _ = pack_view(components(source, ['Encoder']), CharacterTokenizer(), 'encoder', limit=1800)
        self.assertGreater(len(parts), 1)
        self.assertTrue(all(p['tokens'] <= 1800 and 'self.width = 128' in p['text'] for p in parts))

    def test_atlas_endpoint_model_and_bearer_are_used(self):
        class Response(io.StringIO):
            headers = {'x-request-id': 'offline-test'}
        response = Response(json.dumps(dict(model='voyage-code-4', data=[dict(index=0, embedding=[1.] * CODE_DIM)])))
        with patch('urllib.request.urlopen', return_value=response) as network:
            _, call = request_embeddings(['def solve(): pass'], 'offline-test-key', token_count=8)
        request = network.call_args.args[0]
        self.assertEqual(request.full_url, ENDPOINT)
        self.assertEqual(request.get_header('Authorization'), 'Bearer offline-test-key')
        self.assertEqual(call['endpoint'], ENDPOINT)
        self.assertEqual(call['source'], 'mongodb-atlas-http')
        payload = json.loads(request.data)
        self.assertEqual({k: payload[k] for k in SETTINGS}, SETTINGS)
        self.assertFalse(payload['truncation'])

    def test_provider_cache_isolation_and_320k_request_cap(self):
        chunk = dict(id='id', text_sha256='text')
        legacy = dict(preprocessing='v1', tokenizer_sha256='tokenizer', max_tokens=4096, manifest_sha256='manifest')
        atlas = dict(legacy, api=dict(provider='mongodb_atlas', endpoint=ENDPOINT))
        self.assertNotEqual(cache_key(chunk, legacy), cache_key(chunk, atlas))
        atlas.update(chunks=[dict(id=str(i), text=f'component {i}', text_sha256=str(i), tokens=32000) for i in range(12)], ready=True)
        def response(texts, key, **kwargs):
            return np.ones((len(texts), CODE_DIM), dtype=np.float32), dict(source='mongodb-atlas-http', endpoint=ENDPOINT, timestamp_utc='offline-test')
        with tempfile.TemporaryDirectory() as temp, patch('code.V4.solver_code_embeddings.validate_corpus'):
            with patch('code.V4.solver_code_embeddings.api_key', return_value='offline-test-key'):
                with patch('code.V4.solver_code_embeddings.request_embeddings', side_effect=response) as request:
                    with contextlib.redirect_stdout(io.StringIO()):
                        plan = generate(atlas, Path(temp), True, 400000)
            self.assertEqual([c.kwargs['token_count'] for c in request.call_args_list], [320000, 64000])
            self.assertEqual(plan['requests_per_minute'], 2000)
            self.assertEqual(plan['tokens_per_minute'], 8000000)
            with patch('urllib.request.urlopen', side_effect=AssertionError('No network expected')):
                self.assertEqual(generate(atlas, Path(temp), True, 0)['uncached_chunks'], 0)

    def test_chunk_aggregation_is_learnable_and_not_a_fixed_mean(self):
        bundle = bank_fixture()
        encoder = SolverCodeEncoder(bundle['vectors'], bundle['role_mask'], torch.arange(M_GLOBAL),
                                    components=bundle, **params()).eval()
        output = encoder(torch.tensor([1]))
        output.square().sum().backward()
        self.assertGreater(float(encoder.component_query.grad.abs().sum()), 0)
        for module in encoder.component_fusion:
            self.assertGreater(sum(float(p.grad.abs().sum()) for p in module.parameters()), 0)
        for module in encoder.role_proj:
            self.assertGreater(sum(float(p.grad.abs().sum()) for p in module.parameters()), 0)
        self.assertFalse(encoder.chunk_vectors.requires_grad)
        with torch.no_grad():
            encoder.component_query.mul_(100)
        self.assertFalse(torch.equal(output, encoder(torch.tensor([1]))))

    def test_single_full_view_bypasses_chunk_aggregation(self):
        bundle = bank_fixture(False)
        bundle['chunk_vectors'] = bundle['vectors'].reshape(-1, CODE_DIM).clone()
        # Absent role rows are never addressed by indices.
        bundle['chunk_vectors'][bundle['chunk_vectors'].norm(dim=-1) == 0, 0] = 1
        torch.manual_seed(2)
        original = SolverCodeEncoder(bundle['vectors'], bundle['role_mask'], torch.arange(M_GLOBAL), **params()).eval()
        torch.manual_seed(2)
        complete = SolverCodeEncoder(bundle['vectors'], bundle['role_mask'], torch.arange(M_GLOBAL), components=bundle, **params()).eval()
        self.assertFalse(hasattr(complete, 'component_query'))
        torch.testing.assert_close(complete(torch.arange(M_GLOBAL)), original(torch.arange(M_GLOBAL)), atol=2e-6, rtol=2e-6)

    def test_chunk_buffers_shuffle_and_offline_checkpoint_replay(self):
        bundle = bank_fixture()
        config = dict(params(), pair_mode='score_difference', solver_representation='shuffled_code', code_init_seed=450002)
        torch.manual_seed(2)
        with patch('code.V4.solver_code_embeddings.validate_bundle'):
            model = make_r45_model(config, bundle=bundle).eval()
        batch = sample_batch('TSP')
        original = model(batch)['logits']
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'unit_test_only.pt'
            torch.save(dict(model=model.state_dict(), args=dict(model_params=config, solver_names=GLOBAL_SOLVERS,
                                                               embedding_provenance=bundle_provenance(bundle))), path)
            with patch.dict(os.environ, {}, clear=True), patch('urllib.request.urlopen', side_effect=AssertionError('Offline only')):
                with patch('code.V4.solver_code_embeddings.validate_bundle'):
                    replay, _ = load_r45_checkpoint(path)
            torch.testing.assert_close(replay(batch)['logits'], original, atol=0, rtol=0)
            torch.testing.assert_close(replay.solver_encoder.chunk_indices, model.solver_encoder.chunk_indices)
        self.assertNotIn('chunk_vectors', bundle_provenance(bundle))
        self.assertNotIn('chunk_indices', bundle_provenance(bundle))

    def test_real_source_corpus_uses_full_views_and_official_limit(self):
        path = ROOT / 'corpus_atlas/corpus.json'
        if not path.exists():
            self.skipTest('Real corpus has not been prepared')
        corpus = json.loads(path.read_text())
        validate_corpus(corpus, json.loads(path.with_name('source_snapshot.json').read_text()))
        self.assertEqual(corpus['max_tokens'], 32000)
        self.assertEqual(len(corpus['solvers']), M_GLOBAL)
        self.assertTrue(any(c['tokens'] > 4096 for c in corpus['chunks']))
        for view in corpus['semantic_views']:
            if view['full_view_tokens'] <= 32000:
                self.assertTrue(view['complete'])
                self.assertEqual(len(view['chunk_ids']), 1)

    def test_whole_view_pipeline_assemble_and_production_loader_offline(self):
        # All simulated responses/artifacts are temporary and explicitly test-only.
        path = ROOT / 'corpus_atlas/corpus.json'
        if not path.exists():
            self.skipTest('Real corpus has not been prepared')
        corpus = json.loads(path.read_text())
        rng = np.random.default_rng(2)
        def response(texts, key, **kwargs):
            vectors = rng.normal(size=(len(texts), CODE_DIM)).astype(np.float32)
            return vectors, dict(source='mongodb-atlas-http', endpoint=ENDPOINT, timestamp_utc='unit-test-only',
                                 request_id='simulated-no-api-call', usage=dict(total_tokens=kwargs['token_count']))
        with tempfile.TemporaryDirectory(prefix='r45-unit-test-only-') as temp:
            root = Path(temp)
            with patch('code.V4.solver_code_embeddings.api_key', return_value='unit-test-only'):
                with patch('code.V4.solver_code_embeddings.request_embeddings', side_effect=response):
                    with contextlib.redirect_stdout(io.StringIO()):
                        generate(corpus, root / 'cache', True, 200000)
            bundle = assemble(corpus, root / 'cache', root / 'unit-test-only.pt')
            locked = load_bundle(root / 'unit-test-only.pt')
            self.assertEqual(len(bundle['chunk_vectors']), len(corpus['chunks']))
            self.assertEqual(len(bundle['chunk_provenance']), len(corpus['chunks']))
            torch.testing.assert_close(locked['chunk_vectors'], bundle['chunk_vectors'])
            config = dict(params(), pair_mode='score_difference', solver_representation='code')
            model = make_r45_model(config, bundle=bundle).eval()
            checkpoint = root / 'unit-test-only-checkpoint.pt'
            torch.save(dict(model=model.state_dict(), args=dict(model_params=config, solver_names=GLOBAL_SOLVERS,
                                                               embedding_provenance=bundle_provenance(bundle))), checkpoint)
            with patch.dict(os.environ, {}, clear=True), patch('urllib.request.urlopen', side_effect=AssertionError('No network')):
                replay, _ = load_r45_checkpoint(checkpoint)
            for problem in ('TSP', 'CVRP', 'ATSP', 'OVRPTW'):
                batch = sample_batch(problem)
                torch.testing.assert_close(replay(batch)['logits'], model(batch)['logits'], atol=0, rtol=0)


if __name__ == '__main__':
    unittest.main()
