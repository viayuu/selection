"""CPU-only R99 runner witnesses; synthetic inputs, never GPU/jobs/real labels."""

import copy
import io
import inspect
import tempfile
import unittest
from argparse import Namespace
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from code.unified_selector.registry import GLOBAL_SOLVERS, PROBLEMS
from . import r99_experiment as runner
from . import r99_features, r99_model
from .r76_experiment import Runtime, read_json, write_json
from .r77_data import parameter_hash
from .r99_relations import node_messages
from .test_r76 import fixture_batch
from .test_r80 import fresh_fixture


def arguments(directory, arm='A'):
    root = Path(directory)
    return runner.parser().parse_args(['train', '--arm', arm, '--device', 'cpu',
        '--root', str(root / 'run'), '--prepared-root', str(root / 'prepared'),
        '--data-root', str(root / 'data'), '--cache-root', str(root / 'cache')])


def install_lock(args):
    value = dict(runner.required_lock_fields(), data_root=str(args.data_root),
                 prepared_root=str(args.prepared_root))
    write_json(args.root / 'protocol_lock.json', value)
    return value


def synthetic_config(directory, arm='A'):
    args = arguments(directory, arm)
    install_lock(args)
    write_json(args.prepared_root / 'prepared.json', dict(synthetic=True))
    initial, geometry = fresh_fixture()
    model, params, receipt = runner.initialize_shared(initial, geometry, arm)
    ready = dict(trainable_sha256=initial['trainable_sha256'])
    manifest = dict(splits=dict(train={p: dict(pool=['m0', 'm1'], pool_ids=[0, 1]) for p in PROBLEMS}))
    baselines = {p: dict(selection_split='train', sbs_solver='m0') for p in PROBLEMS}
    config = runner.protocol(args, ready, manifest, geometry, baselines, params,
        runner.task_order(runner.MAX_UPDATES), runner.micro_caps('3090', 352))
    return args, config, model, receipt


def input_batch(problem='TSP', rows=3, variant='row'):
    batch, _ = fixture_batch(problem, rows)
    if problem == 'TSP':
        batch['node'] = batch['node'][..., :2].clone()
    batch['relation_nodes'] = node_messages(runner.raw_inputs(batch), variant)
    batch['relation_node_mask'] = batch['node_mask']
    return batch


def feature_receipts(split, arm='A'):
    return {p: dict(complete=True, rows_written=runner.SPLIT_ROWS[split], dtype='float16', width=32,
        identity=dict(problem=p, split=split, variant=runner.VARIANTS[arm],
            rows=runner.SPLIT_ROWS[split], input_only=True, labels_opened=False),
        max_marginal_residual=.02, seconds=1.) for p in PROBLEMS}


class R99RunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        guard = patch('torch.cuda._lazy_init', side_effect=AssertionError('CPU tests must not initialize CUDA'))
        guard.start()
        self.addCleanup(guard.stop)
        runner.private_engine.cache_clear()

    def test_cli_describe_status_never_open_weights_or_features(self):
        with patch.object(torch, 'load', side_effect=AssertionError('no checkpoint reads')):
            for arm in runner.ARMS:
                with redirect_stdout(io.StringIO()):
                    value = runner.main(['describe', '--arm', arm, '--device', 'cpu'])
                self.assertEqual(value['variant'], runner.VARIANTS[arm])
                self.assertEqual(value['required_protocol_lock'], runner.required_lock_fields())
                self.assertEqual(value['max_updates'], 57600)
                for stage in ('prepare', 'train', 'test', 'timing', 'status'):
                    args = runner.parser().parse_args([stage, '--arm', arm])
                    self.assertEqual((args.large_micro_cap, args.eval_batch_size, args.wandb_mode), (352, 256, 'offline'))
            self.assertNotEqual(runner.feature_root(Namespace(arm='A')), runner.feature_root(Namespace(arm='B')))

    def test_main_lock_exact_schema_and_extra_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = arguments(tmp)
            original = install_lock(args)
            original['target_list'] = '/synthetic/fixed36.json'
            write_json(args.root / 'protocol_lock.json', original)
            self.assertEqual(runner.protocol_lock(args)['record'], original)
            for key, value in (('max_updates', 10), ('projection_seed', 2), ('large_micro_cap', 384),
                    ('relation_variants', dict(A='row', B='row')), ('data_root', '/different'),
                    ('train_eval_every', 1440), ('eval_batch_size', 128)):
                write_json(args.root / 'protocol_lock.json', dict(original, **{key: value}))
                with self.subTest(key=key), self.assertRaises(ValueError):
                    runner.protocol_lock(args)

    def test_common_and_added_initialization_hashes_counts_geometry_and_rng(self):
        initial, geometry = fresh_fixture()
        saved_rng = torch.get_rng_state().clone()
        receipts = []
        for arm in runner.ARMS:
            model, params, receipt = runner.initialize_shared(initial, geometry, arm)
            receipts.append(receipt)
            self.assertEqual(torch.get_rng_state().tolist(), saved_rng.tolist())
            self.assertEqual(receipt['common_parameter_sha256'], initial['trainable_sha256'])
            self.assertEqual(receipt['new_parameter_names'], [runner.NEW_PARAMETER])
            self.assertEqual(receipt['new_common_seed'], 990003)
            self.assertFalse(receipt['historical_checkpoint_loaded'])
            self.assertFalse(receipt['external_pretraining'])
            self.assertEqual(receipt['new_parameter_count'], 32 * initial['model_params']['embedding_dim'])
            self.assertEqual(receipt['trainable_parameter_count'], receipt['common_parameter_count'] + receipt['new_parameter_count'])
            self.assertEqual(params['r99_fresh_trainable_sha256'], receipt['trainable_sha256'])
            for name, value in model.named_parameters():
                if name != runner.NEW_PARAMETER:
                    torch.testing.assert_close(value, initial['trainable'][name], rtol=0, atol=0)
            torch.testing.assert_close(model.instance_encoder.geometry_residual.mean, torch.tensor(geometry['mean']))
            torch.testing.assert_close(model.instance_encoder.geometry_residual.std, torch.tensor(geometry['std']))
        self.assertEqual(receipts[0], receipts[1])

    def test_reject_trained_corrupt_removed_or_extra_parameters(self):
        initial, geometry = fresh_fixture()
        for key, value in (('seed', 3), ('initialization', 'trained'), ('trainable_sha256', 'corrupt')):
            bad = dict(initial, **{key: value})
            with self.subTest(key=key), self.assertRaises(ValueError):
                runner.initialize_shared(bad, geometry, 'A')
        model = r99_model.make_model(initial['model_params'], 'A')
        model.register_parameter('unexpected_trainable', torch.nn.Parameter(torch.ones(1)))
        with patch.object(r99_model, 'make_model', return_value=model), self.assertRaises(ValueError):
            runner.initialize_shared(initial, geometry, 'A')
        model = r99_model.make_model(initial['model_params'], 'A')
        with torch.no_grad():
            model.instance_encoder.relation_projection.weight.zero_()
        with patch.object(r99_model, 'make_model', return_value=model), self.assertRaises(ValueError):
            runner.initialize_shared(initial, geometry, 'A')

    def test_private_engine_captures_arm_cache_and_never_patches_historical_globals(self):
        original_open, original_inputs = runner.engine.open_cache, runner.engine.inputs_only
        bases = []
        def base(*unused, **kwargs):
            value = SimpleNamespace(close=Mock())
            bases.append(value)
            return value
        with patch.object(r99_features, 'CompetitionFeatureCache', side_effect=lambda *a, **k: (a, k)) as wrap:
            a = runner.private_engine('A', '/synthetic/A')
            b = runner.private_engine('B', '/synthetic/B')
            self.assertIsNot(a, b)
            self.assertIs(a.train.__globals__, vars(a))
            self.assertIs(inspect.unwrap(a.evaluate).__globals__, vars(a))
            self.assertIs(runner.engine.inputs_only, original_inputs)
            with patch.object(runner.engine, 'open_cache', side_effect=base):
                aa = a.open_cache('/cache', 'train', 'ATSP', {}, '/data')
                bb = b.open_cache('/cache', 'val', 'TSP', {}, '/data')
            self.assertEqual(aa[0][1], Path('/synthetic/A/train/ATSP'))
            self.assertEqual(aa[1]['expected_variant'], 'row')
            self.assertEqual(bb[1]['expected_variant'], 'balanced')
            self.assertIs(aa[0][0], bases[0])
            self.assertEqual(wrap.call_count, 2)
            self.assertIs(runner.engine.open_cache, original_open)
            with patch.object(runner.engine, 'open_cache', side_effect=base), \
                    patch.object(r99_features, 'CompetitionFeatureCache', side_effect=ValueError('bad')):
                # The closure retains the class from factory creation.
                failed = runner.private_engine('A', '/different')
                with self.assertRaises(ValueError):
                    failed.open_cache('/cache', 'train', 'CVRP', {}, '/data')
                bases[-1].close.assert_called_once()

    def test_feature_api_only_train_val_before_lock_all18_after_lock(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = arguments(tmp)
            with patch.object(r99_features, 'prepare_features', side_effect=lambda **kw: feature_receipts(kw['split'])) as prepare:
                result = runner.prepare_input_features(args, torch.device('cpu'))
                self.assertEqual(set(result), {'train', 'val'})
                self.assertEqual([c.kwargs['variant'] for c in prepare.call_args_list], ['row', 'row'])
                self.assertTrue(all(c.kwargs['batch_size'] == 128 for c in prepare.call_args_list))
                with self.assertRaises(PermissionError):
                    runner.prepare_input_features(args, torch.device('cpu'), split='test')
                lock = dict(experiment='R99', arm='A')
                runner.prepare_input_features(args, torch.device('cpu'), split='test', test_lock=lock)
                self.assertIs(prepare.call_args.kwargs['test_lock'], lock)
            bad = feature_receipts('val')
            bad.pop('ATSP')
            with patch.object(r99_features, 'prepare_features', return_value=bad), self.assertRaises(ValueError):
                runner.prepare_input_features(args, torch.device('cpu'), split='val')

    def test_protocol_exact_original_loss_fresh_roundtrip_and_corruption_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            args, config, model, _ = synthetic_config(tmp)
            self.assertEqual(config['loss'], runner.engine.LOSS_RECIPE)
            self.assertEqual(config['objective'], runner.objective_spec('B'))
            self.assertEqual(runner.checkpoint_extras(None), {})
            with self.assertRaises(ValueError):
                runner.restore_objective(None, dict(objective_state={}))
            runner.restore_objective(None, {})
            path = Path(tmp) / 'synthetic.pt'
            state = dict(args=config, model=model.state_dict(), state_hash=runner.json_hash(config))
            torch.save(state, path)
            restored, _ = runner.load_checkpoint(path)
            self.assertEqual(parameter_hash(model.state_dict()), parameter_hash(restored.state_dict()))
            for field, value in (('dropout', 0), ('relation_variant', 'balanced'), ('projection_seed', 2),
                    ('inference_rule', 'TEST best'), ('train_eval_every_updates', 1440),
                    ('loss', dict(ce=1.)), ('new_parameter_sha256', 'different')):
                bad = dict(config, **{field: value})
                with self.subTest(field=field), self.assertRaises(ValueError):
                    runner.validate_checkpoint_config(bad)
            state['objective_state'] = {}
            torch.save(state, path)
            with self.assertRaises(ValueError):
                runner.load_checkpoint(path)

    def test_original_micro_cap_guard_and_real_two_pass_cpu_update(self):
        batch = dict(ind=torch.zeros(1024, dtype=torch.int64))
        with patch.object(runner.engine, 'update_batch', return_value='ok') as update:
            for p, cap in (('TSP', 352), ('CVRP', 352), ('ATSP', 1024), ('VRPL', 1024)):
                self.assertEqual(runner.update_batch(None, None, None, batch, .2, None, cap, problem=p), 'ok')
            with self.assertRaises(ValueError):
                runner.update_batch(None, None, None, batch, .2, None, 352, problem='VRPL')
            self.assertEqual(update.call_count, 4)
        initial, geometry = fresh_fixture()
        for arm in runner.ARMS:
            model, _, _ = runner.initialize_shared(initial, geometry, arm)
            batch = input_batch(rows=1024, variant=runner.VARIANTS[arm])
            sizes = []
            handle = model.register_forward_pre_hook(lambda module, inputs: sizes.append(len(inputs[0]['node_mask'])))
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
            scaler = torch.amp.GradScaler('cpu', enabled=False)
            before = parameter_hash(dict(model.named_parameters()))
            values, retries, _ = runner.update_batch(runner.make_objective(model, arm), optimizer,
                scaler, batch, .1, Runtime(0, 1, torch.device('cpu')), 352, problem='TSP')
            handle.remove()
            self.assertEqual(sizes, [352, 352, 352, 352, 320, 320])
            self.assertEqual(retries, 0)
            self.assertTrue(torch.isfinite(values).all())
            self.assertNotEqual(before, parameter_hash(dict(model.named_parameters())))

    def test_online_timing_really_strips_relation_fields_and_matches_fp16_cache(self):
        initial, geometry = fresh_fixture()
        for arm in runner.ARMS:
            model, _, _ = runner.initialize_shared(initial, geometry, arm)
            model.eval()
            batch = input_batch(rows=3, variant=runner.VARIANTS[arm])
            clean = runner.raw_inputs(batch)
            self.assertFalse({'relation_nodes', 'relation_node_mask', 'costs', 'ind'} & set(clean))
            observed = []
            handle = model.register_forward_pre_hook(lambda module, inputs: observed.append('relation_nodes' in inputs[0]))
            with patch.object(torch.cuda, 'synchronize', side_effect=AssertionError('no CUDA')):
                result = runner.measure_inputs(model, batch, batch, torch.device('cpu'))
            handle.remove()
            self.assertEqual(observed, [False] * 4 + [True] * 4)
            for scope in runner.TIMING_SCOPES:
                self.assertEqual(len(result[scope]['batch_seconds']), 3)
                self.assertGreater(result[scope]['mean_seconds_per_instance'], 0)
            wrong = dict(batch, relation_nodes=torch.zeros_like(batch['relation_nodes']))
            with self.assertRaises(ValueError):
                runner.measure_inputs(model, batch, wrong, torch.device('cpu'))

    def test_formal_cpu_prepare_rejected_before_device_work(self):
        with tempfile.TemporaryDirectory() as tmp, self.assertRaises(ValueError):
            runner.main(['prepare', '--arm', 'A', '--device', 'cpu', '--root', tmp])


if __name__ == '__main__':
    unittest.main()
