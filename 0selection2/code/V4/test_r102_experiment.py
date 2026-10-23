"""Small synthetic runner contracts; no native weights, GPU or real inference."""

import copy
import hashlib
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torch import nn

from . import r102_experiment as runner
from .r76_experiment import read_json, write_json


def provenance(arm='A'):
    extra = 'random' if arm == 'A' else 'pretrained'
    components = {name: dict(variant=extra, output_dim=width, native_source_sha256='c' * 64,
        checkpoint_pin=dict(sha256=name + '-pin'), checkpoint_sha256=None if arm == 'A' else name + '-pin',
        checkpoint_path=None if arm == 'A' else '/synthetic/' + name,
        random_seed=7 if arm == 'A' else None, known_unknown_pretraining_source=dict(
            applies_to_loaded_weights=arm == 'B', historical_cost_recipe_certified=False))
        for name, width in (('TSP', 128), ('ATSP', 512))}
    components['VRP'] = dict(variant='pretrained', checkpoint_sha256=runner.R94B_CHECKPOINT_SHA256,
        output_dim=128, native_source_sha256='d' * 64)
    return dict(format='synthetic_contract_not_native_evidence', arm=arm, variant=runner.ARM_VARIANTS[arm],
        supported_problems=list(runner.PROBLEMS), widths=runner.WIDTH_BY_PROBLEM.copy(),
        components=components, spec=dict(no_finetuning=True))


def feature_tasks(args, split, arm='A'):
    result = {}
    for problem in runner.PROBLEMS:
        result[problem] = dict(complete=True, rows_written=runner.SPLIT_ROWS[split],
            problem=problem, split=split, rows=runner.SPLIT_ROWS[split],
            shape=[runner.SPLIT_ROWS[split], 2, runner.WIDTH_BY_PROBLEM[problem]],
            dtype='float16', input_only=True, donor=provenance(arm),
            variant=runner.ARM_VARIANTS[arm], directory=str(runner.feature_root(args) / split / problem))
        if problem in runner.ROUTING:
            result[problem].update(source_kind='r94b_reference',
                routing_directory=str(runner.routing_feature_root(args) / split / problem))
    return result


def synthetic_config(directory, arm='A'):
    root = Path(directory)
    args = runner.parser().parse_args(['train', '--arm', arm, '--root', str(root),
        '--data-root', str(root / 'data'), '--prepared-root', str(root / 'prepared'),
        '--cache-root', str(root / 'cache'), '--feature-root', str(root / (arm + '_extra')),
        '--routing-feature-root', str(root / 'common_routing'), '--device', 'cpu'])
    lock = dict(runner.required_lock_fields(), data_root=str(args.data_root), prepared_root=str(args.prepared_root))
    write_json(root / 'protocol_lock.json', lock)
    write_json(args.prepared_root / 'prepared.json', dict(synthetic=True))
    write_json(root / arm / 'input_features_ready.json', dict(donor_provenance=provenance(arm)))
    ready = dict(trainable_sha256='a' * 64)
    pools = {p: list(runner.GLOBAL_SOLVERS[:2]) for p in runner.PROBLEMS}
    manifest = dict(splits=dict(train={p: dict(pool=pool, pool_ids=[0, 1]) for p, pool in pools.items()}))
    baselines = {p: dict(selection_split='train', sbs_solver=pool[0]) for p, pool in pools.items()}
    params = dict(frozen_donor=dict(variant=runner.ARM_VARIANTS[arm], provenance=provenance(arm)),
        r102_fresh_trainable_sha256='b' * 64, r102_common_parameter_sha256='a' * 64,
        r102_new_parameter_sha256='e' * 64)
    with patch.object(runner, 'file_hash', return_value='f' * 64):
        config = runner.protocol(args, ready, manifest, dict(mean=[0.], std=[1.]), baselines,
            params, runner.task_order(runner.MAX_UPDATES), runner.micro_caps('4090', 352))
    return args, config


class FakeDonor(nn.Module):
    """Synthetic state container, not an ELG/MatNet/ReLD encoder."""

    def __init__(self, arm='A', *, initialize_from_checkpoint=True, expected_provenance=None):
        super().__init__()
        self.variant = runner.ARM_VARIANTS[arm]
        self.identity = copy.deepcopy(expected_provenance or provenance(arm))
        self.register_buffer('synthetic', torch.tensor(1.))

    def provenance(self):
        return copy.deepcopy(self.identity)

    def forward(self, batch):
        if any(k in batch for k in ('ind', 'costs', 'nco_nodes', 'nco_node_mask')):
            raise AssertionError('Synthetic online donor received labels or cached fields')
        return torch.zeros(*batch['node_mask'].shape, 128)


class TinySelector(nn.Module):
    def __init__(self, bypass=False, arm='A'):
        super().__init__()
        self.donor = FakeDonor(arm)
        self.bypass = bypass
        self.seen = []

    def forward(self, batch):
        self.seen.append(set(batch))
        if 'nco_nodes' not in batch and not self.bypass:
            self.donor(batch)
        return dict(logits=torch.zeros(128, 2), solver_mask=torch.ones(128, 2, dtype=torch.bool))


class R102RunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        guard = patch('torch.cuda._lazy_init', side_effect=AssertionError('CPU fixtures cannot initialize CUDA'))
        guard.start()
        self.addCleanup(guard.stop)

    def test_cli_describe_is_source_and_weight_free(self):
        for command in ('prepare', 'train', 'test', 'timing', 'describe'):
            args = runner.parser().parse_args([command, '--arm', 'A'])
            self.assertEqual((args.gpu_type, args.large_micro_cap, args.eval_batch_size), ('4090', 352, 256))
            self.assertEqual(args.wandb_mode, 'offline')
            self.assertFalse(args.resume)
        with patch.object(torch, 'load', side_effect=AssertionError('describe cannot open weights')), \
                patch.object(runner, 'make_model', side_effect=AssertionError('describe cannot make model')), \
                patch('builtins.print'):
            value = runner.main(['describe', '--arm', 'B'])
        self.assertFalse(value['inference'])
        self.assertEqual(value['trainable_parameter_count'], 1941257)
        self.assertEqual(runner.NEW_PARAMETERS + runner.ORIGINAL_PARAMETERS, 1941257)
        self.assertEqual(runner.routing_feature_root(args), Path('/tmp/shiys_R94_B_node_features'))
        self.assertNotEqual(runner.feature_root(args), runner.feature_root(runner.parser().parse_args(['describe', '--arm', 'B'])))

    def test_protocol_and_main_lock_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            args, config = synthetic_config(directory)
            runner.validate_checkpoint_config(config)
            self.assertEqual(config['loss'], runner.engine.LOSS_RECIPE)
            self.assertEqual(config['objective'], runner.objective_spec('B'))
            self.assertTrue(config['external_solver_pretraining'])
            self.assertFalse(config['extra_type_pretraining'])
            for key, value in (('max_successful_updates', 5), ('new_parameter_count', 98305),
                    ('auxiliary_loss', True), ('early_stop', True), ('inference_rule', 'test tuned'),
                    ('extra_type_pretraining', True), ('rdrop_weight', 0.), ('loss', dict(ce=1.)),
                    ('checkpoint_donor_restore', 'external pretrained weights')):
                bad = copy.deepcopy(config)
                bad[key] = value
                with self.subTest(key=key), self.assertRaises(ValueError):
                    runner.validate_checkpoint_config(bad)
            write_json(args.root / 'protocol_lock.json', dict(runner.required_lock_fields(),
                max_updates=100, data_root=str(args.data_root), prepared_root=str(args.prepared_root)))
            with self.assertRaises(ValueError):
                runner.protocol_lock(args)

    def test_both_arms_always_pretrained_routing(self):
        for arm in runner.ARMS:
            value = provenance(arm)
            runner.validate_donor_provenance(value, arm)
            for component in ('TSP', 'ATSP', 'VRP'):
                bad = copy.deepcopy(value)
                bad['components'][component]['variant'] = 'pretrained' if value['components'][component]['variant'] == 'random' else 'random'
                with self.subTest(arm=arm, component=component), self.assertRaises(ValueError):
                    runner.validate_donor_provenance(bad, arm)

    def test_initialization_retains_backbone_and_exact_three_seeded_projections(self):
        def factory(params, arm):
            model = nn.Module()
            model.register_parameter('backbone', nn.Parameter(torch.zeros(3)))
            model.instance_encoder = nn.Module()
            model.instance_encoder.geometry_residual = nn.Module()
            model.instance_encoder.geometry_residual.register_buffer('mean', torch.zeros(2))
            model.instance_encoder.geometry_residual.register_buffer('std', torch.ones(2))
            model.instance_encoder.nco_projections = nn.ModuleDict()
            for name, shape in runner.PROJECTION_SHAPES.items():
                group = name.split('.')[-2]
                with torch.random.fork_rng(devices=[]):
                    torch.random.default_generator.manual_seed(runner.model_spec()['projection_seeds'][group])
                    model.instance_encoder.nco_projections[group] = nn.Linear(shape[1], shape[0], bias=False)
            model.donor = FakeDonor(arm)
            model.params = dict(params, frozen_donor=dict(variant=runner.ARM_VARIANTS[arm], provenance=provenance(arm)))
            return model

        reference = dict(backbone=torch.tensor([1., 2., 3.]))
        initial = dict(seed=2, initialization='scratch', trainable=reference, model_params={},
            trainable_sha256=runner.parameter_hash(reference))
        geometry = dict(mean=[2., 3.], std=[4., 5.])
        receipts = []
        with patch.object(runner, 'make_model', side_effect=factory), \
                patch.object(runner, 'ORIGINAL_PARAMETERS', 3), patch.object(runner, 'PARAMETER_COUNT', 98307):
            for arm in runner.ARMS:
                rng = torch.get_rng_state().clone()
                model, params, receipt = runner.initialize_shared(initial, geometry, arm)
                self.assertTrue(torch.equal(rng, torch.get_rng_state()))
                self.assertTrue(torch.equal(model.backbone.detach(), reference['backbone']))
                self.assertEqual(model.instance_encoder.geometry_residual.mean.tolist(), geometry['mean'])
                self.assertEqual(set(receipt['new_parameter_names']), set(runner.PROJECTION_NAMES))
                self.assertEqual(receipt['new_parameter_count'], 98304)
                self.assertTrue(receipt['external_solver_pretraining'])
                self.assertEqual(receipt['extra_type_pretraining'], arm == 'B')
                self.assertEqual(params['r102_common_parameter_sha256'], initial['trainable_sha256'])
                receipts.append(receipt)
            self.assertEqual(receipts[0]['trainable_sha256'], receipts[1]['trainable_sha256'])
            with self.assertRaises(ValueError):
                runner.initialize_shared(dict(initial, initialization='trained'), geometry, 'A')

    def test_unique_lock_is_explicit_val_and_refuses_reselection(self):
        with tempfile.TemporaryDirectory() as directory:
            args, config = synthetic_config(directory)
            best = dict(epoch=1, successful_updates=1440, val=dict(macro=dict(macro_vs_sbs_pct=-2.)))
            with patch.object(runner, 'file_hash', return_value='c' * 64):
                value = runner.lock_best(args.root / args.arm, best, config, 'synthetic-state')
                self.assertEqual(value['selection_split'], 'val')
                self.assertTrue(value['selected_without_test'])
                self.assertFalse(value['per_problem_assembly'])
                self.assertEqual(runner.lock_best(args.root / args.arm, best, config, 'synthetic-state'), value)
                with self.assertRaises(PermissionError):
                    runner.lock_best(args.root / args.arm, dict(best, successful_updates=2880), config, 'synthetic-state')

    def test_all18_feature_receipts_and_routing_readonly_reference(self):
        with tempfile.TemporaryDirectory() as directory:
            args, _ = synthetic_config(directory)
            good = feature_tasks(args, 'train')
            self.assertEqual(len(runner.feature_receipts(dict(splits=dict(train=good)), 'train', args, provenance())), 18)
            for problem, key, value in (('ATSP', 'shape', [81000, 2, 256]), ('TSP', 'rows_written', 2),
                    ('CVRP', 'source_kind', 'new_extraction'), ('VRPB', 'routing_directory', '/wrong'),
                    ('TSP', 'input_only', False), ('TSP', 'directory', '/wrong')):
                bad = copy.deepcopy(good)
                bad[problem][key] = value
                with self.subTest(problem=problem, key=key), self.assertRaises(ValueError):
                    runner.feature_receipts(bad, 'train', args, provenance())
            with self.assertRaises(ValueError):
                runner.feature_receipts({k: v for k, v in good.items() if k != 'ATSP'}, 'train', args, provenance())

    def test_feature_api_calls_only_train_val_and_test_lock_is_required(self):
        with tempfile.TemporaryDirectory() as directory:
            args, _ = synthetic_config(directory)
            calls = []

            def prepare(cache_root, feature_root, split, donor, device, **kwargs):
                calls.append((split, kwargs))
                return dict(complete=True, splits={split: feature_tasks(args, split)})

            stub = types.SimpleNamespace(prepare_features=prepare)
            with patch.dict(sys.modules, {'code.V4.r102_features': stub}):
                result = runner.prepare_input_features(args, FakeDonor(), torch.device('cpu'))
                self.assertEqual(list(result), ['train', 'val'])
                self.assertTrue(all(kwargs['batch_size'] == 128 for _, kwargs in calls))
                self.assertTrue(all(kwargs['routing_feature_root'] == runner.routing_feature_root(args) for _, kwargs in calls))
                with self.assertRaises(PermissionError):
                    runner.prepare_input_features(args, FakeDonor(), torch.device('cpu'), split='test')
            self.assertEqual(len(calls), 2)

    def test_private_engine_does_not_rebind_old_engine(self):
        original = runner.engine.open_cache, runner.engine.inputs_only, runner.engine.wandb_run
        closed, observed = [], []

        class FeatureCache:
            def __init__(self, base, feature_directory, **kwargs):
                observed.append((feature_directory, kwargs))

        base = types.SimpleNamespace(close=lambda: closed.append(True))
        stub = types.SimpleNamespace(FrozenFeatureCache=FeatureCache)
        with patch.dict(sys.modules, {'code.V4.r102_features': stub}):
            module = runner.private_engine('A', '/synthetic/extra', '/synthetic/routing', json.dumps(provenance(), sort_keys=True))
        self.assertIsNot(module, runner.engine)
        self.assertEqual(original, (runner.engine.open_cache, runner.engine.inputs_only, runner.engine.wandb_run))
        with patch.object(runner.engine, 'open_cache', return_value=base):
            module.open_cache('/cache', 'val', 'CVRP', {}, '/data')
        self.assertEqual(observed[0][0], Path('/synthetic/extra/val/CVRP'))
        self.assertEqual(observed[0][1]['routing_feature_directory'], Path('/synthetic/routing/val/CVRP'))
        self.assertEqual(observed[0][1]['expected_variant'], 'extra_random')
        self.assertEqual(closed, [])

    def test_checkpoint_restore_explicit_no_external_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            _, config = synthetic_config(directory, 'B')
            model = TinySelector(arm='B')
            state = dict(args=config, state_hash=runner.json_hash(config), model=model.state_dict())
            with patch.object(torch, 'load', return_value=state) as load, \
                    patch('code.V4.r102_donor.FullTypeDonor', wraps=FakeDonor) as donor, \
                    patch.object(runner, 'make_model', return_value=model) as factory:
                restored, saved = runner.load_checkpoint(Path(directory) / 'best.pt')
            self.assertIs(restored, model)
            self.assertIs(saved, state)
            load.assert_called_once()
            donor.assert_called_once_with('B', initialize_from_checkpoint=False, expected_provenance=provenance('B'))
            self.assertIsInstance(factory.call_args.kwargs['donor'], FakeDonor)

    def test_original_micro_weighting_and_no_mutable_objective_state(self):
        batch = dict(ind=torch.zeros(1024, dtype=torch.long), nco_nodes='synthetic', nco_node_mask='synthetic')
        with patch.object(runner.engine, 'update_batch', return_value=('parts', 2, {})) as update:
            self.assertEqual(runner.update_batch(None, None, None, batch, .2, None, 1024, problem='VRPL'), ('parts', 2, {}))
            self.assertEqual(runner.update_batch(None, None, None, batch, .2, None, 352, problem='TSP'), ('parts', 2, {}))
            with self.assertRaises(ValueError):
                runner.update_batch(None, None, None, batch, .2, None, 352, problem='VRPL')
            with self.assertRaises(ValueError):
                runner.update_batch(None, None, None, dict(ind=batch['ind']), .2, None, 1024, problem='ATSP')
            self.assertEqual(update.call_count, 2)
        self.assertEqual(runner.checkpoint_extras(None), {})
        runner.restore_objective(None, {})
        with self.assertRaises(ValueError):
            runner.restore_objective(None, dict(objective_state={}))

    def test_online_timing_strips_both_fields_every_call_and_proves_execution(self):
        mask = torch.ones(128, 2, dtype=torch.bool)
        raw = dict(problem_id=0, kind='coord', node=torch.zeros(128, 2, 2), n=torch.full((128,), 2),
            node_mask=mask, ind=torch.zeros(128, dtype=torch.long), costs=torch.ones(128, 2),
            nco_nodes=torch.ones(128, 2, 128), nco_node_mask=mask)
        original = dict(raw)
        model = TinySelector()
        result = runner.measure_inputs(model, raw, raw, torch.device('cpu'))
        self.assertEqual(result['donor_plus_selector']['donor_forward_calls'], 4)
        self.assertEqual(result['cache_hit_selector']['donor_forward_calls'], 0)
        self.assertTrue(all(not ({'nco_nodes', 'nco_node_mask', 'ind', 'costs'} & keys) for keys in model.seen[:4]))
        self.assertTrue(all({'nco_nodes', 'nco_node_mask'} <= keys for keys in model.seen[4:]))
        self.assertEqual(set(raw), set(original))
        with self.assertRaises(ValueError):
            runner.measure_inputs(TinySelector(bypass=True), raw, raw, torch.device('cpu'))
        self.assertFalse(model.donor._forward_hooks)


if __name__ == '__main__':
    unittest.main()
