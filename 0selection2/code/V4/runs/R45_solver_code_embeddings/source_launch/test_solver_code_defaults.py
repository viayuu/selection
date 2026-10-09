"""Offline tests for explicit assumptions; no solver, dataset or API calls."""

import copy
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from code.unified_selector.registry import GLOBAL_SOLVERS
from .solver_code_corpus import MANIFEST, build_corpus, digest, write_json
from .solver_code_defaults import fill_missing, prepare_defaults
from .solver_code_embeddings import (SETTINGS, assemble, cache_key, generate, load_bundle,
                                     main as embedding_main, validate_bundle, validate_corpus)
from .solver_code_encoder import load_r45_checkpoint, make_r45_model
from .test_dual_stream import params
from .test_r45 import CharacterTokenizer
from .test_solver_features import sample_batch


class DefaultConfigurationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.original_bytes = MANIFEST.read_bytes()
        cls.original = json.loads(cls.original_bytes)
        cls.derived, cls.rows = prepare_defaults(cls.original)
        cls.corpus = build_corpus(cls.derived, CharacterTokenizer(), allow_assumptions=True)

    def test_verified_values_and_aliases_win(self):
        known = dict(embedding_dim=128, num_heads=8, options=dict(a=1), normalization=None)
        actual, extra = fill_missing(known, dict(embed_dim=256, head_num=16,
                                               options=dict(a=99, b=2), normalization='batch'))
        self.assertEqual(actual, dict(known, options=dict(a=1, b=2)))
        self.assertEqual(extra, dict(options=dict(b=2)))
        self.assertEqual(known['options'], dict(a=1))

    def test_original_manifest_and_corrected_identities_are_preserved(self):
        self.assertEqual(MANIFEST.read_bytes(), self.original_bytes)
        self.assertEqual([row['solver_name'] for row in self.derived['solvers']], GLOBAL_SOLVERS)
        for original, derived in zip(self.original['solvers'], self.derived['solvers']):
            self.assertEqual(original['solver_id'], derived['solver_id'])
            self.assertEqual(len(original['deployments']), len(derived['deployments']))
            for before, after in zip(original['deployments'], derived['deployments']):
                for key, value in before.items():
                    self.assertEqual(after[key], value)
        glop = next(row for row in self.derived['solvers'] if row['solver_name'] == 'GLOP')['deployments'][0]
        self.assertEqual(glop['implementation'], 'glop_atsp_init_only')
        self.assertFalse(glop['resolved_config']['learned_policy_used'])
        self.assertEqual(glop['assumed_config'], {})

    def test_defaults_are_source_backed_and_do_not_infer_suffixes(self):
        rows = {row['solver']: row for row in self.rows}
        self.assertEqual(rows['BQ']['assumed_config']['dim_emb'], 192)
        self.assertEqual(rows['DIFUSCO']['assumed_config']['diffusion_type'], 'categorical')
        self.assertFalse(rows['DIFUSCO']['assumed_config']['node_feature_only'])
        self.assertEqual(rows['DIFUSCO']['assumed_config'], rows['DIFUSCO500']['assumed_config'])
        self.assertEqual(rows['T2T']['assumed_config'], rows['T2T500']['assumed_config'])
        self.assertEqual(rows['MTPOMO']['assumed_config']['decoder_strategy'], 'sampling')
        self.assertTrue(all(row['sources'] for row in self.rows))

    def test_explicit_approval_does_not_hide_historical_gaps(self):
        strict = build_corpus(self.derived, CharacterTokenizer(), draft=True)
        self.assertFalse(strict['ready'])
        self.assertTrue(self.corpus['ready'])
        self.assertTrue(self.corpus['uses_assumptions'])
        self.assertFalse(self.corpus['historical_bindings_verified'])
        self.assertEqual(len(self.corpus['provenance_warnings']), len(self.rows))
        self.assertTrue(all(variant['roles']['config'] for solver in self.corpus['solvers'] for variant in solver['variants']))
        with tempfile.TemporaryDirectory() as temp, patch('urllib.request.urlopen', side_effect=AssertionError('No API')):
            plan = generate(self.corpus, Path(temp))
        self.assertTrue(plan['ready'])
        self.assertFalse(plan['historical_bindings_verified'])
        altered = copy.deepcopy(self.corpus)
        altered['historical_bindings_verified'] = True
        altered['sha256'] = digest({k: v for k, v in altered.items() if k != 'sha256'})
        with self.assertRaisesRegex(ValueError, 'Assumptions'):
            validate_corpus(altered)

    def test_missing_core_source_is_still_blocking(self):
        altered = copy.deepcopy(self.derived)
        altered['implementations']['bq']['views']['encoder']['status'] = 'missing'
        corpus = build_corpus(altered, CharacterTokenizer(), draft=True, allow_assumptions=True)
        self.assertFalse(corpus['ready'])

    def test_changed_yaml_defaults_invalidate_corpus(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'defaults.yaml'
            path.write_text('embedding_dim: 128\n')
            manifest, corpus = copy.deepcopy(self.derived), copy.deepcopy(self.corpus)
            manifest['roots']['fixture'] = temp
            manifest['default_source_hashes']['fixture:defaults.yaml'] = digest(path.read_bytes())
            corpus['manifest_sha256'] = digest(manifest)
            corpus['config_source_hashes'] = manifest['default_source_hashes']
            corpus['sha256'] = digest({k: v for k, v in corpus.items() if k != 'sha256'})
            validate_corpus(corpus, manifest)
            path.write_text('embedding_dim: 256\n')
            with self.assertRaisesRegex(ValueError, 'Default configuration source changed'):
                validate_corpus(corpus, manifest)

    def test_bundle_retains_assumptions_for_offline_checkpoints(self):
        # Emulated HTTP cache entries exist only inside this temporary test directory.
        corpus = copy.deepcopy(self.corpus)
        chunk = corpus['chunks'][0]
        corpus['chunks'] = [chunk]
        for solver in corpus['solvers']:
            for variant in solver['variants']:
                variant['roles'] = {role: [chunk['id']] for role in corpus['roles']}
        corpus['sha256'] = digest({k: v for k, v in corpus.items() if k != 'sha256'})
        vector = np.ones(1024, dtype=np.float32)
        key = cache_key(chunk, corpus)
        with tempfile.TemporaryDirectory() as temp, patch('urllib.request.urlopen', side_effect=AssertionError('No API')):
            root = Path(temp)
            write_json(root / (key + '.json'), dict(key=key, provider='voyage', settings=SETTINGS,
                       embedding=vector.tolist(), embedding_sha256=digest(vector.tobytes()),
                       call=dict(source='voyage-http', timestamp_utc='unit-test-only', request_id='emulated')))
            bundle = assemble(corpus, root, root / 'unit-test-only.pt')
            replay = load_bundle(root / 'unit-test-only.pt')
            config = dict(params(), pair_mode='score_difference', solver_representation='code')
            model = make_r45_model(config, bundle).eval()
            batch = sample_batch('TSP')
            expected = model(batch)['logits'].detach()
            provenance = {key: value for key, value in bundle.items() if key not in ('vectors', 'role_mask')}
            checkpoint = root / 'unit-test-only-checkpoint.pt'
            torch.save(dict(model=model.state_dict(), args=dict(model_params=config,
                       solver_names=GLOBAL_SOLVERS, embedding_provenance=provenance)), checkpoint)
            (root / 'unit-test-only.pt').unlink()
            (root / 'unit-test-only.lock.json').unlink()
            with patch.dict('os.environ', {}, clear=True):
                restored, saved = load_r45_checkpoint(checkpoint)
                torch.testing.assert_close(restored(batch)['logits'], expected, atol=0, rtol=0)
            self.assertEqual(saved['args']['embedding_provenance'], provenance)
        for field in ('uses_assumptions', 'historical_bindings_verified', 'provenance_warnings', 'assumed_configurations'):
            self.assertEqual(bundle[field], corpus[field])
            self.assertEqual(replay[field], corpus[field])
        broken = dict(bundle, provenance_warnings=[])
        with self.assertRaisesRegex(ValueError, 'Assumed-default'):
            validate_bundle(broken)

    def test_generation_and_assembly_entrypoint_requires_upload_opt_in(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            write_json(root / 'corpus.json', self.corpus)
            write_json(root / 'source_snapshot.json', self.derived)
            argv = ['solver_code_embeddings', '--stage', 'all', '--corpus', str(root / 'corpus.json'),
                    '--provider', 'voyage',
                    '--cache-dir', str(root / 'cache'), '--output', str(root / 'vectors.pt'),
                    '--max-tokens', '200000']
            with patch('sys.argv', argv), patch('code.V4.solver_code_embeddings.generate') as generation:
                with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as failure:
                    embedding_main()
                self.assertEqual(failure.exception.code, 2)
                generation.assert_not_called()
            with patch('sys.argv', argv + ['--allow-upload']):
                with patch('code.V4.solver_code_embeddings.generate', return_value={'ready': True}) as generation:
                    with patch('code.V4.solver_code_embeddings.assemble', return_value={'vectors': torch.empty(22, 4, 1024)}) as assembly:
                        with contextlib.redirect_stdout(io.StringIO()):
                            embedding_main()
                generation.assert_called_once_with(self.corpus, root / 'cache', True, 200000, None,
                                                   batch_size=16, requests_per_minute=2000, tokens_per_minute=8000000,
                                                   provider='voyage')
                assembly.assert_called_once_with(self.corpus, root / 'cache', root / 'vectors.pt')


if __name__ == '__main__':
    unittest.main()
