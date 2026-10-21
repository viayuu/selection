"""Small synthetic CPU witnesses; no real data, training, GPU, or API calls."""

import copy
import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import torch
import torch.nn.functional as F

from code.unified_selector.registry import GLOBAL_SOLVERS, M_GLOBAL, PROBLEMS
from . import r80_experiment as r80, r80_reporting as report80
from . import r83_experiment as r83, r83_reporting as report83
from .r77_data import parameter_hash
from .solver_code_embeddings import ENDPOINT, SETTINGS, bundle_provenance
from .solver_code_encoder import CODE_DIM, ROLES, SolverCodeEncoder
from .test_r76 import fixture_batch
from .test_r80 import fresh_fixture


def model_only_fixture():
    """Production-shaped provenance for tests only, not a real embedding artifact."""
    generator = torch.Generator().manual_seed(83)
    neural = [index for index, name in enumerate(GLOBAL_SOLVERS) if name != 'GLOP']
    chunks = F.normalize(torch.randn(len(neural) * 4, CODE_DIM, generator=generator), dim=-1)
    indices = torch.full((M_GLOBAL, len(ROLES), 1, 2), -1, dtype=torch.long)
    indices[neural, :2, 0] = torch.arange(len(chunks)).reshape(len(neural), 2, 2)
    mask = torch.zeros(M_GLOBAL, len(ROLES), dtype=torch.bool)
    mask[neural, :2] = True
    vectors = torch.zeros(M_GLOBAL, len(ROLES), CODE_DIM)
    vectors[neural, :2] = F.normalize(chunks.reshape(len(neural), 2, 2, CODE_DIM).mean(2), dim=-1)
    return dict(schema='r45-atlas-v1', kind='production', provider='mongodb_atlas',
        settings=SETTINGS, roles=list(ROLES), solver_names=GLOBAL_SOLVERS,
        vectors=vectors, role_mask=mask, chunk_vectors=chunks, chunk_indices=indices,
        corpus_ready=True, corpus_sha256='c' * 64, cache_sha256='d' * 64, endpoint=ENDPOINT,
        chunk_provenance=[dict(id=f'synthetic-only-{index}', text_sha256='e' * 64,
                              provenance=[dict(file='synthetic_model.py', symbols=['SyntheticModel'], sha256='a' * 64)])
                          for index in range(len(chunks))],
        calls=[dict(source='mongodb-atlas-http', timestamp_utc='synthetic-test-only', endpoint=ENDPOINT)],
        model_only_scope=dict(kind='model_only', active_roles=['encoder', 'decision'],
            excluded_roles=['inference', 'config'], no_neural_model_solvers=['GLOP'],
            excluded_content=['training', 'solver inference', 'environment', 'external configuration'],
            glop_reason='Synthetic fixture for glop_atsp_init_only: random insertion, no neural model'))


def args_fixture(root, command='train'):
    args = r83.parser().parse_args([command, '--root', str(root), '--device', 'cpu'])
    args.embeddings = root / 'model_only_vectors.pt'
    args.prepared_root = root / 'prepared'
    args.data_root = root / 'synthetic-data-never-opened'
    args.cache_root = root / 'synthetic-cache-never-opened'
    return args


def protocol_fixture(args, bundle):
    args.prepared_root.mkdir()
    r80.write_json(args.prepared_root / 'prepared.json', dict(synthetic_only=True))
    r80.write_json(args.embeddings.with_suffix('.lock.json'), dict(sha256='f' * 64))
    ready = dict(model_params=dict(solver_representation='code'), trainable_sha256='a' * 64)
    manifest = dict(splits={split: {} for split in ('train', 'val', 'test')})
    baselines = {p: dict(selection_split='train', sbs_solver=GLOBAL_SOLVERS[0]) for p in PROBLEMS}
    return r83.protocol(args, ready, manifest, {}, baselines, ready['model_params'], bundle,
                        r80.task_order(), r80.micro_caps(r83.ENGINE_ARM, '3090'))


def checkpoint_fixture(bundle):
    initial, geometry = fresh_fixture()
    model, params, receipt = r83.initialize_shared(initial, geometry, bundle=bundle)
    config = dict(experiment='R83', arm=r83.GROUP, engine_arm=r83.ENGINE_ARM, seed=2,
        initialization='scratch_shared_r77', solver_names=GLOBAL_SOLVERS, model_params=params,
        source_scope=r83.SOURCE_SCOPE, model_factory=r83.MODEL_FACTORY, model_only_scope=bundle['model_only_scope'],
        embedding_provenance=bundle_provenance(bundle))
    state = dict(args=config, state_hash=r80.json_hash(config), model=model.state_dict(),
                 initialization=receipt, successful_updates=1242, epoch=1)
    return model, state


class R83Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        self.cuda_guard = patch('torch.cuda._lazy_init', side_effect=AssertionError('CPU only'))
        self.cuda_guard.start()
        self.addCleanup(self.cuda_guard.stop)

    def test_cli_single_group_and_custom_root_embedding_default(self):
        args = r83.parser().parse_args(['train'])
        self.assertEqual((args.arm, args.engine_arm, args.wandb_mode), (r83.GROUP, 'D_code_embedding', 'auto'))
        self.assertEqual((args.gpu_type, args.eval_batch_size), ('3090', 128))
        with patch.object(r80, 'run') as run:
            r83.main(['describe', '--root', '/tmp/synthetic-r83'])
        self.assertEqual(run.call_args.args[0].embeddings, Path('/tmp/synthetic-r83/model_only_vectors.pt'))
        for module in (r83, report83):
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                module.parser().parse_args(['--arm', 'D_code_embedding'])

    def test_describe_and_status_do_not_open_model_data_or_create_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / 'not-created'
            with patch.object(r80, 'train', side_effect=AssertionError('No training')), \
                    patch.object(r80, 'open_cache', side_effect=AssertionError('No data')), \
                    patch.object(r83, 'load_bundle', side_effect=AssertionError('No embedding load')), \
                    patch.object(torch, 'load', side_effect=AssertionError('No checkpoint load')):
                output = io.StringIO()
                with redirect_stdout(output):
                    r83.main(['describe', '--root', str(root)])
                description = json.loads(output.getvalue())
                self.assertEqual(description['experiment'], 'R83')
                self.assertEqual((description['batch_size'], description['max_updates'],
                                  description['warmup_updates'], description['eval_every']), (1024, 49680, 3726, 1242))
                with redirect_stdout(io.StringIO()):
                    status = r83.main(['status', '--root', str(root)])
                self.assertEqual(status['status'], {})
                self.assertFalse(root.exists())

    def test_model_only_bundle_uses_standard_provenance_and_absent_role_rules(self):
        bundle = model_only_fixture()
        r83.validate_model_only_bundle(bundle)
        legacy = dict(bundle, schema='r45-embeddings-v2')
        r83.validate_model_only_bundle(legacy)
        for role in (2, 3):
            invalid = copy.deepcopy(bundle)
            invalid['role_mask'][:, role] = True
            invalid['vectors'][:, role] = invalid['vectors'][:, 0]
            invalid['chunk_indices'][:, role] = invalid['chunk_indices'][:, 0]
            with self.assertRaises(ValueError):
                r83.validate_model_only_bundle(invalid)
        for change in (dict(calls=[]), dict(chunk_provenance=[]), dict(corpus_sha256='missing')):
            with self.assertRaises(ValueError):
                r83.validate_model_only_bundle(dict(bundle, **change))
        invalid = copy.deepcopy(bundle)
        invalid['vectors'][:, 2] = invalid['vectors'][:, 0]
        with self.assertRaises(ValueError):
            r83.validate_model_only_bundle(invalid)
        invalid = copy.deepcopy(bundle)
        glop = GLOBAL_SOLVERS.index('GLOP')
        for key in ('role_mask', 'vectors', 'chunk_indices'):
            invalid[key][glop] = invalid[key][0]
        with self.assertRaises(ValueError):
            r83.validate_model_only_bundle(invalid)

    def test_bundle_file_lock_load_without_api_and_tamper_refusal(self):
        bundle = model_only_fixture()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'synthetic_only.pt'
            torch.save(bundle, path)
            r80.write_json(path.with_suffix('.lock.json'), dict(sha256=r80.file_hash(path)))
            with patch('code.V4.solver_code_embeddings.request_embeddings', side_effect=AssertionError('No API')):
                loaded = r83.load_bundle(path)
            torch.testing.assert_close(loaded['vectors'], bundle['vectors'])
            self.assertEqual(bundle_provenance(loaded), bundle_provenance(bundle))
            r80.write_json(path.with_suffix('.lock.json'), dict(sha256='wrong'))
            with self.assertRaises(ValueError):
                r83.load_bundle(path)

    def test_exact_r80d_encoder_common_id_parameters_geometry_and_budget(self):
        bundle = model_only_fixture()
        initial, geometry = fresh_fixture()
        model, params, receipt = r83.initialize_shared(initial, geometry, bundle=bundle)
        reference, reference_params, _ = r80.initialize_shared(initial, geometry, r83.ENGINE_ARM, bundle=bundle)
        self.assertIs(type(model.solver_encoder), SolverCodeEncoder)
        self.assertEqual(params, reference_params)
        self.assertEqual(parameter_hash(model.state_dict()), parameter_hash(reference.state_dict()))
        for name in receipt['common_parameter_names']:
            torch.testing.assert_close(dict(model.named_parameters())[name], initial['trainable'][name], rtol=0, atol=0)
        self.assertIn('solver_encoder.solver_emb.weight', receipt['common_parameter_names'])
        self.assertFalse(receipt['historical_checkpoint_loaded'])
        self.assertEqual(GLOBAL_SOLVERS[4], 'GLOP')
        encoder = model.solver_encoder.eval()
        self.assertTrue(encoder.has_model_free_rows)
        ids = torch.tensor([GLOBAL_SOLVERS.index('GLOP')])
        with torch.no_grad():
            torch.testing.assert_close(encoder(ids), encoder.norm(encoder.solver_emb(ids)), rtol=0, atol=0)
        for _, tensor in model.solver_encoder.named_buffers():
            self.assertFalse(tensor.requires_grad)
        torch.testing.assert_close(model.solver_encoder.chunk_indices[:, 2:],
                                   torch.full_like(model.solver_encoder.chunk_indices[:, 2:], -1))
        spec = r80.arm_spec(r83.ENGINE_ARM)
        self.assertEqual(spec.max_updates * spec.batch_size, 50872320)
        self.assertEqual(r80.consistency_weight(spec.warmup_updates, spec), .35)
        self.assertAlmostEqual(r80.learning_rate(spec.max_updates, spec), 1e-6)

    def test_checkpoint_offline_roundtrip_and_wrong_identity_refusal(self):
        bundle = model_only_fixture()
        model, state = checkpoint_fixture(bundle)
        batch, _ = fixture_batch('TSP', 3)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'synthetic_checkpoint.pt'
            torch.save(state, path)
            with patch.object(r83, 'load_bundle', side_effect=AssertionError('Offline checkpoint only')):
                loaded, saved = r83.load_checkpoint(path)
            self.assertEqual(saved['args']['experiment'], 'R83')
            self.assertEqual(saved['args']['model_only_scope'], bundle['model_only_scope'])
            self.assertEqual(saved['args']['embedding_provenance'], bundle_provenance(bundle))
            self.assertEqual(parameter_hash(loaded.state_dict()), parameter_hash(model.state_dict()))
            model.eval()
            loaded.eval()
            self.assertTrue(loaded.solver_encoder.has_model_free_rows)
            with torch.no_grad():
                torch.testing.assert_close(model(r80.inputs_only(batch))['logits'],
                                           loaded(r80.inputs_only(batch))['logits'], rtol=0, atol=0)
                encoder = loaded.solver_encoder
                ids = torch.tensor([GLOBAL_SOLVERS.index('GLOP')])
                torch.testing.assert_close(encoder(ids), encoder.norm(encoder.solver_emb(ids)), rtol=0, atol=0)
            with self.assertRaises(ValueError):
                r80.load_checkpoint(path)
            for change in (dict(experiment='R80'), dict(engine_arm='A_ce_only'), dict(source_scope='all source')):
                bad = dict(state, args=dict(state['args'], **change))
                bad['state_hash'] = r80.json_hash(bad['args'])
                torch.save(bad, path)
                with self.assertRaises(ValueError):
                    r83.load_checkpoint(path)
            bad = copy.deepcopy(state)
            bad['model']['solver_encoder.chunk_vectors'] = bad['model']['solver_encoder.chunk_vectors'].roll(1, 0)
            torch.save(bad, path)
            with self.assertRaisesRegex(ValueError, 'Frozen external source buffers changed'):
                r83.load_checkpoint(path)

    def test_protocol_retains_full_objective_and_r83_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            args = args_fixture(Path(directory))
            bundle = model_only_fixture()
            config = protocol_fixture(args, bundle)
            path = args.root / 'synthetic_protocol.json'
            r80.write_json(path, config)
            serialized = r80.read_json(path)
            self.assertEqual(serialized['model_only_scope'], bundle['model_only_scope'])
            self.assertEqual(serialized['embedding_provenance'], bundle_provenance(bundle))
            self.assertEqual(serialized['embedding_provenance']['model_only_scope'], serialized['model_only_scope'])
            self.assertEqual(r80.json_hash(serialized), r80.json_hash(config))
        self.assertEqual((config['experiment'], config['arm'], config['engine_arm']), ('R83', r83.GROUP, r83.ENGINE_ARM))
        self.assertEqual(config['loss'], r80.LOSS_RECIPE)
        self.assertEqual(config['model_factory'], r83.MODEL_FACTORY)
        self.assertEqual(config['role_policy'], dict(encoder=True, decision=True, inference=False, config=False))
        self.assertEqual(config['model_only_scope']['no_neural_model_solvers'], ['GLOP'])
        self.assertEqual(config['embedding_provenance'], bundle_provenance(bundle))
        self.assertEqual(config['task_order_sha256'], r80.hashlib.sha256('\n'.join(r80.task_order()).encode('ascii')).hexdigest())
        self.assertFalse(config['early_stop'])
        self.assertTrue(config['wandb'])
        self.assertIn('r83_experiment.py', config['source_sha256'])
        self.assertIn('r80_experiment.py', config['source_sha256'])

    def test_training_test_and_report_are_engine_delegations_not_copied_trainers(self):
        args = r83.parser().parse_args(['train'])
        device = torch.device('cpu')
        with patch.object(r80, 'train', return_value='not-run') as train:
            self.assertEqual(r83.train(args, device), 'not-run')
        self.assertEqual(train.call_args.kwargs, dict(checkpoint_loader=r83.load_checkpoint,
            bundle_loader=r83.load_bundle, protocol_builder=r83.protocol))
        with patch.object(r80, 'test', return_value='not-run') as test:
            r83.test(args, device)
        self.assertIs(test.call_args.kwargs['checkpoint_loader'], r83.load_checkpoint)
        report_args = report83.parser().parse_args([])
        with patch.object(report80, 'report', return_value='not-run') as report:
            self.assertEqual(report83.report(report_args), 'not-run')
        self.assertEqual(report.call_args.kwargs, dict(experiment='R83', arms=report83.BUDGETS,
                                                      factory_module=report83.FACTORY_MODULE))

    def test_val_lock_and_reporting_reject_cross_version_without_model_or_data_reads(self):
        with tempfile.TemporaryDirectory() as directory:
            args = args_fixture(Path(directory))
            config = protocol_fixture(args, model_only_fixture())
            run = args.root / args.arm
            run.mkdir()
            (run / 'best.pt').write_bytes(b'synthetic CPU lock fixture, not a trained checkpoint')
            r80.write_json(run / 'protocol.json', config)
            best = dict(epoch=1, successful_updates=1242, val=dict(macro={}))
            state_hash = r80.json_hash(config)
            lock = r80.lock_best(run, best, config, state_hash)
            self.assertEqual((lock['experiment'], lock['arm']), ('R83', r83.GROUP))
            with self.assertRaises(PermissionError):
                r80.authorize_test(args)
            complete = dict(successful_updates=49680, selected_best=lock)
            r80.write_json(run / 'training_complete.json', complete)
            self.assertEqual(r80.authorize_test(args)[0], lock)
            metrics = dict(per_problem=dict.fromkeys(PROBLEMS, {}))
            identity = dict(checkpoint_sha256=lock['checkpoint_sha256'], state_hash=state_hash,
                            epoch=1, selected_update=1242)
            r80.write_json(run / 'test_results.json', dict(identity, complete=True, result=metrics))
            r80.write_json(run / 'final_metrics.json', dict(identity, val=metrics))
            self.assertEqual(report83.locked_inputs(args.root)[2], lock)
            with self.assertRaises(ValueError):
                report80.locked_inputs(args.root, args.arm, arms=report83.BUDGETS)
            with self.assertRaises(PermissionError):
                r80.lock_best(run, dict(best, successful_updates=2484), config, state_hash)
            r80.write_json(run / 'final_test_lock.json', dict(lock, experiment='R80'))
            with self.assertRaises(PermissionError):
                r80.authorize_test(args)
            with self.assertRaises(ValueError):
                report83.locked_inputs(args.root)

    def test_wandb_required_with_explicit_offline_fallback_and_r83_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            args = args_fixture(Path(directory))
            (args.root / args.arm).mkdir()
            offline = SimpleNamespace(id='offline-synthetic', url=None)
            wandb = SimpleNamespace(Settings=Mock(return_value={}), finish=Mock(),
                                    init=Mock(side_effect=[RuntimeError('synthetic missing credentials'), offline]))
            with patch.dict(sys.modules, wandb=wandb), redirect_stdout(io.StringIO()):
                self.assertIs(r80.wandb_run(args, dict(experiment='R83')), offline)
            first, second = wandb.init.call_args_list
            self.assertEqual((first.kwargs['mode'], second.kwargs['mode']), ('online', 'offline'))
            self.assertEqual(second.kwargs['name'], 'R83_' + r83.GROUP)
            self.assertTrue(second.kwargs['id'].startswith('r83-'))
            receipt = r80.read_json(args.root / args.arm / 'wandb_receipt.json')
            self.assertEqual(receipt['mode'], 'offline')
            self.assertIn('synthetic missing credentials', receipt['fallback_reason'])

    def test_grouped_distribution_real_pdf_smoke_and_labels(self):
        import matplotlib.pyplot as plt
        helper = report80.reporting_helpers()
        pool = ['solver-a', 'solver-b']
        data = SimpleNamespace(pool=pool, n=5, pred=np.array([0, 0, 0, 1, 1]),
                               winner=np.array([0, 1, 1, 1, 0]), top1_pct=40.)
        lib_n = helper.LIBS['TSP'][1]
        lib = SimpleNamespace(pool=pool, n=lib_n, pred=np.resize(data.pred, lib_n), winner=np.resize(data.winner, lib_n))
        lib.top1_pct = float(np.mean(lib.pred == lib.winner) * 100)
        panels = helper.make_panels('TSP', data, lib, pool)
        lock = dict(epoch=1, selected_update=1242, checkpoint_sha256='f' * 64)
        helper.setup_style()
        fig = report80._distribution(helper, 'TSP', pool, panels, 'synthetic-only-data', 'R83/' + r83.GROUP, lock)
        try:
            self.assertEqual(len(fig.axes), 2)
            self.assertIn('TSP TEST', fig.axes[0].get_title())
            self.assertIn('TSPLIB', fig.axes[1].get_title())
            for ax, source in zip(fig.axes, (data, lib)):
                self.assertIn(f'\u6a21\u578bTop1={source.top1_pct:.2f}%', ax.get_title())
                self.assertEqual(len(ax.patches), 2 * len(pool))
                counts = np.r_[np.bincount(source.pred, minlength=2), np.bincount(source.winner, minlength=2)]
                np.testing.assert_allclose([bar.get_width() for bar in ax.patches], counts * 100 / source.n)
                self.assertEqual(ax.get_yticklabels()[0].get_text(), pool[0])
                self.assertNotEqual(ax.patches[0].get_facecolor(), ax.patches[2].get_facecolor())
            captions = '\n'.join(text.get_text() for text in fig.texts)
            self.assertIn('R83/' + r83.GROUP, captions)
            self.assertIn('native/FP64 ind', captions)
            self.assertNotIn('R80', captions)
            self.assertNotIn('R77B', captions)
            rows = helper.distribution_rows('TSP', panels, pool, 'synthetic-only-data')
            self.assertTrue(all(row['model_top1_native_pct'] is None for row in rows if row['panel'] == 'true_winner'))
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / 'synthetic-only-distribution.pdf'
                fig.savefig(path)
                self.assertTrue(path.read_bytes().startswith(b'%PDF-'))
                self.assertGreater(path.stat().st_size, 1000)
        finally:
            plt.close(fig)
        panels = helper.make_panels('ATSP', data, None, pool)
        fig = report80._distribution(helper, 'ATSP', pool, panels, 'synthetic-only-data',
                                     'R83/' + r83.GROUP, lock, no_model_solvers=['GLOP'])
        try:
            self.assertEqual(len(fig.axes), 1)
            captions = '\n'.join(text.get_text() for text in fig.texts)
            self.assertIn('GLOP: no neural model', captions)
            self.assertIn('original trainable ID only', captions)
            self.assertIn('Candidates/IDs unchanged', captions)
        finally:
            plt.close(fig)


if __name__ == '__main__':
    unittest.main()
