"""Focused two-thread CPU witnesses for R93 runner/protocol/checkpoint hooks."""

import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from code.unified_selector.registry import POOLS, PROBLEMS, S2I
from . import r93_experiment as runner
from .pairwise_selector import rank_by_solver_ids
from .r76_experiment import save_checkpoint
from .r77_data import parameter_hash
from .test_dual_stream import params as original_params
from .test_solver_features import sample_batch


def fresh_fixture():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(2)
        params = dict(original_params(), dropout=.1, local_geometry=True,
                      pair_mode='score_difference', solver_representation='handcrafted')
        model = runner.engine.make_model(params, 'A_baseline')
    values = {name: value.detach().clone() for name, value in model.named_parameters()}
    width = model.instance_encoder.geometry_residual.mean.numel()
    return dict(seed=2, initialization='scratch', model_params=copy.deepcopy(model.params),
                trainable=values, trainable_sha256=parameter_hash(values)), dict(mean=[.2] * width, std=[1.3] * width)


def config_fixture(directory, initial, geometry, params, arm):
    args = runner.parser().parse_args(['describe', '--arm', arm, '--root', str(directory),
        '--prepared-root', str(directory / 'prepared'), '--data-root', str(directory / 'data'),
        '--cache-root', str(directory / 'cache'), '--device', 'cpu'])
    lock = dict(experiment='R93', locked=True, arms=list(runner.ARMS), seed=2,
        task_scope='full18', rows_per_problem=runner.SPLIT_ROWS, batch_size=1024,
        logical_global_batch=1024, max_updates=57600, warmup_updates=4320, eval_every=1440,
        dropout=runner.DROPOUT, large_micro_cap=352, eval_batch_size=256,
        data_root=str(args.data_root), prepared_root=str(args.prepared_root))
    runner.write_json(directory / 'protocol_lock.json', lock)
    runner.write_json(args.prepared_root / 'prepared.json', {'fixture': True})
    ready = dict(trainable_sha256=initial['trainable_sha256'])
    pools = {p: dict(pool=POOLS[p], pool_ids=[S2I[s] for s in POOLS[p]]) for p in PROBLEMS}
    manifest = dict(splits=dict(train=pools))
    config = runner.protocol(args, ready, manifest, geometry, {}, params,
                             runner.task_order(runner.MAX_UPDATES), runner.micro_caps('3090', 352))
    return args, config


class R93ExperimentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        guard = patch('torch.cuda._lazy_init', side_effect=AssertionError('CPU tests must not initialize CUDA'))
        guard.start()
        self.addCleanup(guard.stop)

    def test_both_arms_retain_every_fresh_r84_parameter_and_train_only_geometry(self):
        initial, geometry = fresh_fixture()
        original = parameter_hash(initial['trainable'])
        models = []
        for arm in runner.ARMS:
            model, params, receipt = runner.initialize_shared(initial, geometry, arm)
            models.append(model.eval())
            actual = dict(model.named_parameters())
            self.assertEqual(set(actual), set(initial['trainable']))
            self.assertEqual(parameter_hash(actual), original)
            self.assertEqual(receipt['trainable_sha256'], original)
            self.assertEqual(receipt['common_parameter_sha256'], original)
            self.assertFalse(receipt['historical_checkpoint_loaded'])
            self.assertEqual(receipt['new_parameter_names'], [])
            self.assertEqual(receipt['removed_parameter_names'], [])
            self.assertEqual(params['dropout'], runner.DROPOUT[arm])
            self.assertEqual(params['role_arm'], runner.MODEL_ROLE_ARM)
            dropouts = [m for m in model.modules() if isinstance(m, torch.nn.Dropout)]
            self.assertTrue(dropouts)
            self.assertTrue(all(m.p == runner.DROPOUT[arm] for m in dropouts))
            branch = model.instance_encoder.geometry_residual
            torch.testing.assert_close(branch.mean, torch.tensor(geometry['mean']), rtol=0, atol=0)
            torch.testing.assert_close(branch.std, torch.tensor(geometry['std']), rtol=0, atol=0)
        for problem in PROBLEMS:
            batch = sample_batch(problem)
            if problem == 'TSP':
                batch['node'] = batch['node'][..., :2]
            with torch.no_grad():
                first, second = [model(batch) for model in models]
            for name in ('utility', 'pair_margin', 'logits', 'solver_mask'):
                torch.testing.assert_close(first[name], second[name], rtol=0, atol=0)
            torch.testing.assert_close(rank_by_solver_ids(first['utility'], batch['pool_ids']),
                                       rank_by_solver_ids(first['logits'], batch['pool_ids']), rtol=0, atol=0)
        self.assertEqual(parameter_hash(initial['trainable']), original)

    def test_nonfresh_corrupt_and_unknown_initialization_rejected(self):
        initial, geometry = fresh_fixture()
        bad = [dict(initial, seed=1), dict(initial, initialization='trained_checkpoint'),
               dict(initial, trainable_sha256='0' * 64)]
        with patch.object(runner, 'make_model', side_effect=AssertionError('bad source reached factory')):
            for source in bad:
                with self.assertRaises(ValueError):
                    runner.initialize_shared(source, geometry, 'A')
            with self.assertRaises(ValueError):
                runner.initialize_shared(initial, geometry, 'C')

    def test_protocol_and_checkpoint_roundtrip_both_arms(self):
        initial, geometry = fresh_fixture()
        with tempfile.TemporaryDirectory() as temporary:
            for arm in runner.ARMS:
                model, params, receipt = runner.initialize_shared(initial, geometry, arm)
                args, config = config_fixture(Path(temporary), initial, geometry, params, arm)
                runner.validate_checkpoint_config(config)
                self.assertEqual(config['loss']['ce'], 1.)
                self.assertEqual(config['rdrop_weight'], 0.)
                self.assertEqual(config['rdrop_ramp_updates'], 0)
                self.assertEqual(config['objective']['forward_passes'], 1)
                self.assertEqual(config['model_factory'], 'code.V4.r93_experiment.load_checkpoint')
                self.assertEqual(set(config['candidate_pools']), set(PROBLEMS))
                self.assertFalse(config['historical_reference']['matched_original_loss_baseline'])
                with torch.no_grad():
                    model.score_head[-1].bias.add_(.125)
                state = dict(args=config, model=model.state_dict(), initialization=receipt,
                             state_hash=runner.json_hash(config), successful_updates=1440)
                path = args.root / (arm + '.pt')
                save_checkpoint(path, state)
                loaded, restored = runner.load_checkpoint(path, device='cpu')
                self.assertEqual(loaded.params, model.params)
                self.assertEqual(restored['initialization'], receipt)
                for name, value in model.state_dict().items():
                    torch.testing.assert_close(value, loaded.state_dict()[name], rtol=0, atol=0)
                self.assertEqual(runner.checkpoint_extras(runner.make_objective(loaded, arm)), {})

    def test_checkpoint_rejects_changed_hash_loss_dropout_state_and_missing_parameter(self):
        initial, geometry = fresh_fixture()
        model, params, receipt = runner.initialize_shared(initial, geometry, 'B')
        with tempfile.TemporaryDirectory() as temporary:
            _, config = config_fixture(Path(temporary), initial, geometry, params, 'B')
            state = dict(args=config, model=model.state_dict(), initialization=receipt,
                         state_hash=runner.json_hash(config))
            path = Path(temporary) / 'synthetic.pt'
            for kind in ('hash', 'loss', 'dropout', 'state', 'missing'):
                changed = copy.deepcopy(state)
                if kind == 'hash':
                    changed['state_hash'] = '0' * 64
                elif kind == 'loss':
                    changed['args']['objective']['coefficients']['comparison'] = .35
                elif kind == 'dropout':
                    changed['args']['model_params']['dropout'] = .1
                elif kind == 'state':
                    changed['objective_state'] = {}
                else:
                    del changed['model']['score_head.0.weight']
                if kind != 'hash':
                    changed['state_hash'] = runner.json_hash(changed['args'])
                save_checkpoint(path, changed)
                with self.subTest(kind=kind), self.assertRaises((ValueError, RuntimeError)):
                    runner.load_checkpoint(path)

    def test_lock_rejects_budget_dropout_and_data_drift(self):
        initial, geometry = fresh_fixture()
        _, params, _ = runner.initialize_shared(initial, geometry, 'A')
        with tempfile.TemporaryDirectory() as temporary:
            args, _ = config_fixture(Path(temporary), initial, geometry, params, 'A')
            path = args.root / 'protocol_lock.json'
            original = runner.read_json(path)
            for name, value in (('max_updates', 1440), ('dropout', dict(A=.1, B=.1)),
                                ('data_root', str(args.root / 'other'))):
                runner.write_json(path, dict(original, **{name: value}))
                with self.subTest(name=name), self.assertRaises(ValueError):
                    runner.protocol_lock(args)

    def test_test_gate_requires_complete_budget_one_val_lock_and_unchanged_inference(self):
        initial, geometry = fresh_fixture()
        model, params, _ = runner.initialize_shared(initial, geometry, 'A')
        with tempfile.TemporaryDirectory() as temporary:
            args, config = config_fixture(Path(temporary), initial, geometry, params, 'A')
            root = args.root / args.arm
            runner.write_json(root / 'protocol.json', config)
            save_checkpoint(root / 'best.pt', dict(model=model.state_dict()))
            lock = dict(experiment='R93', arm='A', selected_without_test=True,
                per_problem_assembly=False, state_hash=runner.json_hash(config),
                checkpoint=str(root / 'best.pt'), checkpoint_sha256=runner.file_hash(root / 'best.pt'),
                inference_rule=config['inference_rule'], winner_rule=runner.WINNER_RULE)
            runner.write_json(root / 'final_test_lock.json', lock)
            runner.write_json(root / 'training_complete.json', dict(successful_updates=57600))
            self.assertEqual(runner.authorize_test(args)[0], lock)
            runner.write_json(root / 'training_complete.json', dict(successful_updates=1440))
            with self.assertRaises(PermissionError):
                runner.authorize_test(args)
            runner.write_json(root / 'training_complete.json', dict(successful_updates=57600))
            for name, value in (('per_problem_assembly', True), ('inference_rule', 'utility plus bias')):
                runner.write_json(root / 'final_test_lock.json', dict(lock, **{name: value}))
                with self.assertRaises(PermissionError):
                    runner.authorize_test(args)

    def test_private_engine_has_no_fake_rdrop_ramp_and_shared_engine_is_unchanged(self):
        original = runner.engine.consistency_weight
        private = runner.training_engine()
        self.assertIsNot(private, runner.engine)
        self.assertIs(runner.engine.consistency_weight, original)
        self.assertEqual(original(4320), .35)
        for step in (1, 4320, 57600):
            self.assertEqual(private.consistency_weight(step), 0.)
            self.assertEqual(private.learning_rate(step), runner.learning_rate(step))

    def test_parser_and_description_defaults(self):
        for gpu in ('3090', '4090'):
            args = runner.parser().parse_args(['describe', '--arm', 'B', '--gpu-type', gpu])
            self.assertEqual(args.wandb_mode, 'offline')
            self.assertEqual(args.large_micro_cap, 352)
            self.assertEqual(args.eval_batch_size, 256)
            self.assertEqual(runner.describe(args)['dropout'], 0.)
            self.assertEqual(runner.describe(args)['max_updates'], 57600)
        with self.assertRaises(ValueError):
            runner.objective_spec('C')

    def test_first_real_task_accepts_its_existing1024_microcap(self):
        batch = dict(ind=torch.zeros(1024, dtype=torch.long))
        parts = torch.ones(len(runner.PART_KEYS))
        for problem, cap in (('VRPL', 1024), ('ATSP', 1024), ('TSP', 352), ('CVRP', 352)):
            with patch.object(runner.engine, 'update_batch', return_value=(parts, 0, {})) as update:
                actual, retries, _ = runner.update_batch(None, None, None, batch, .35,
                    None, cap, problem=problem)
                self.assertEqual(retries, 0)
                self.assertEqual(actual.tolist(), parts.tolist())
                self.assertEqual(update.call_args.args[4], 0.)
                self.assertEqual(update.call_args.args[6], cap)

    def test_large_instances_still_reject_oversized_microcaps(self):
        batch = dict(ind=torch.zeros(1024, dtype=torch.long))
        with self.assertRaises(ValueError):
            runner.update_batch(None, None, None, batch, 0., None, 1024, problem='TSP')


if __name__ == '__main__':
    unittest.main()
