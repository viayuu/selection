"""Bounded synthetic CPU runner contracts; no real arrays, checkpoints or GPU."""

import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from . import r108_experiment as runner
from .pairwise_objective import LABEL_KEYS
from .pairwise_selector import maximin_logits
from .r76_experiment import Runtime, save_checkpoint, write_json
from .r77_data import parameter_hash
from .r108_model import PRIOR_BUFFERS
from .r108_priors import build_priors
from .test_r108_model import full18_params, input_batch, synthetic_cache


class TinySelector(torch.nn.Module):
    def __init__(self, arm):
        super().__init__()
        self.r108_arm = arm
        self.weight = torch.nn.Parameter(torch.tensor([.1, -.2, .3]))
        self.calls = 0

    def forward(self, batch):
        self.calls += 1
        if set(batch).intersection(LABEL_KEYS):
            raise AssertionError('Supervision reached the selector')
        t = batch['x'][:, None] * self.weight[None]
        pi = t.new_tensor([.2, .3, .5]).expand_as(t)
        s, mask = t + pi.log(), torch.ones_like(t, dtype=torch.bool)
        margins = s[:, :, None] - s[:, None, :]
        return dict(utility=s, residual_utility=t, conditional_prior=pi,
                    solver_mask=mask, logits=maximin_logits(margins, mask))


class R108RunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(2)
        cls.temporary = tempfile.TemporaryDirectory()
        root = Path(cls.temporary.name)
        synthetic_cache(root / 'cache')
        cls.priors = build_priors(root / 'cache', expected_rows=8)
        cls.prior_path = root / 'synthetic_priors.json'
        write_json(cls.prior_path, cls.priors)
        with torch.device('cpu'), torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(2)
            original = runner.engine.make_model(full18_params(), 'A_baseline')
        trainable = {k: v.detach().clone() for k, v in original.named_parameters()}
        cls.initial = dict(seed=2, initialization='scratch', model_params=original.params,
            trainable=trainable, trainable_sha256=parameter_hash(trainable))
        cls.geometry = dict(mean=[.21] * 12, std=[1.31] * 12, split='train')

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()
        torch.set_num_threads(cls.threads)

    def setUp(self):
        for guard in (
            patch.object(runner, 'DEFAULT_PRIORS', self.prior_path),
            patch('torch.cuda._lazy_init', side_effect=AssertionError('CPU only')),
        ):
            guard.start()
            self.addCleanup(guard.stop)

    def test_cli_schedule_private_callbacks_and_describe(self):
        local = runner.private_engine()
        self.assertIsNot(local, runner.engine)
        self.assertIs(local.train.__globals__['micro_caps'], runner.micro_caps)
        self.assertIs(local.evaluate.__wrapped__.__globals__['micro_caps'], runner.micro_caps)
        self.assertIs(local.train.__globals__['write_json'], runner._engine_write_json)
        with self.subTest(contract='zero_consistency_callback'):
            callback = local.train.__globals__['consistency_weight']
            self.assertEqual([callback(u) for u in (1, 4320, 57600)], [0., 0., 0.])
            self.assertIs(callback, runner.consistency_weight)
        self.assertEqual(runner.engine.consistency_weight(4320), .35)
        for arm in runner.ARMS:
            for command in ('train', 'test', 'timing', 'describe', 'status'):
                args = runner.parser().parse_args([command, '--arm', arm])
                self.assertEqual((args.large_micro_cap, args.eval_batch_size), (512, 256))
                self.assertEqual(args.wandb_mode, 'offline')
            with self.subTest(contract='describe_eval_batch', arm=arm):
                self.assertEqual(runner.describe(args)['eval_batch_size'], args.eval_batch_size)
        order = runner.task_order(57600)
        self.assertEqual({p: order.count(p) for p in runner.PROBLEMS},
                         dict.fromkeys(runner.PROBLEMS, 3200))
        self.assertEqual((runner.BATCH_SIZE, runner.MAX_UPDATES), (1024, 57600))
        self.assertEqual((runner.learning_rate(4320), runner.learning_rate(57600)), (1e-4, 1e-6))

    def test_common_fresh_complete_state_including_frozen_priors(self):
        states = []
        for arm in runner.ARMS:
            model, params, receipt = runner.initialize_shared(self.initial, self.geometry, arm)
            self.assertEqual(sum(p.numel() for p in model.parameters()), 1842953)
            self.assertTrue(receipt['cross_arm_complete_state_verified'])
            self.assertFalse(receipt['active_functional_freedom_matched'])
            self.assertEqual(params['dropout'], .1)
            buffers = dict(model.named_buffers())
            for name in PRIOR_BUFFERS:
                self.assertIn(name, model.state_dict())
                self.assertFalse(buffers[name].requires_grad)
            for name, tensor in model.named_parameters():
                torch.testing.assert_close(tensor, self.initial['trainable'][name], rtol=0, atol=0)
            states.append(parameter_hash(model.state_dict()))
        self.assertEqual(*states)

    def test_protocol_and_trained_checkpoint_restore_without_prior_refit(self):
        for arm in runner.ARMS:
            with self.subTest(arm=arm), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                args = runner.parser().parse_args(['train', '--arm', arm, '--root', str(root),
                    '--prepared-root', str(root / 'prepared'), '--data-root', str(root / 'data'),
                    '--cache-root', str(root / 'cache')])
                write_json(args.prepared_root / 'prepared.json', {})
                write_json(root / 'protocol_lock.json', dict(runner.required_lock_fields(),
                    data_root=str(args.data_root), prepared_root=str(args.prepared_root),
                    primary_cache=str(args.cache_root)))
                model, params, _ = runner.initialize_shared(self.initial, self.geometry, arm)
                manifest = dict(splits=dict(train={p: dict(pool=t['pool'], pool_ids=t['pool_ids'])
                                                   for p, t in self.priors['tasks'].items()}))
                config = runner.protocol(args, self.initial, manifest, self.geometry, {}, params,
                    runner.task_order(57600), runner.micro_caps())
                self.assertFalse(config['active_parameter_matched'])
                self.assertTrue(config['model_spec']['same_forward_both_arms'])
                self.assertEqual(config['rdrop_weight'], 0.)
                self.assertEqual(config['loss']['dropout_passes'], 1)
                self.assertEqual([config['loss'][k] for k in ('comparison', 'risk', 'kl')], [0., 0., 0.])
                self.assertFalse(config['objective']['rdrop'])
                self.assertFalse(set(config).intersection(('path_spec', 'path_init_seed')))
                with torch.no_grad():
                    next(model.parameters()).add_(.01)
                self.assertNotEqual(parameter_hash(dict(model.named_parameters())),
                                    runner.ORIGINAL_FRESH_SHA256)
                state = dict(model=model.state_dict(), args=config, state_hash=runner.json_hash(config),
                             successful_updates=1440, epoch=1)
                path = root / 'synthetic_trained.pt'
                save_checkpoint(path, state)
                with patch.object(runner, 'read_json', side_effect=AssertionError('restore must not read priors')), \
                        patch.object(runner.models.ConditionalPriorSelector, 'install_priors',
                                     side_effect=AssertionError('restore must not refit/install priors')):
                    restored, loaded = runner.load_checkpoint(path)
                self.assertEqual(parameter_hash(restored.state_dict()), parameter_hash(state['model']))
                self.assertEqual(loaded['successful_updates'], 1440)
                for key in (next(iter(dict(model.named_parameters()))), PRIOR_BUFFERS[0]):
                    broken = dict(state, model=dict(state['model']))
                    del broken['model'][key]
                    save_checkpoint(path, broken)
                    with self.assertRaises(RuntimeError, msg=key):
                        runner.load_checkpoint(path)
                save_checkpoint(path, dict(state, objective_state={}))
                with self.assertRaises(ValueError):
                    runner.load_checkpoint(path)
                for field, value in (('rdrop_weight', .35), ('loss', runner.engine.LOSS_RECIPE)):
                    broken = copy.deepcopy(config)
                    broken[field] = value
                    with self.assertRaises(ValueError, msg=field):
                        runner.validate_checkpoint_config(broken)

    def test_stateless_single_forward_and_row_weighted_logical_batch(self):
        batch = dict(ind=torch.arange(1024) % 3, x=torch.linspace(.1, 1., 1024),
                     costs=torch.zeros(1024, 3), cost_regret=torch.ones(1024, 3))
        for arm in runner.ARMS:
            outcomes = []
            for problem, cap in (('TSP', 512), ('ATSP', 1024)):
                selector = TinySelector(arm)
                objective = runner.make_objective(selector, arm)
                self.assertEqual(runner.checkpoint_extras(objective), {})
                runner.restore_objective(objective, {})
                with self.assertRaises(ValueError):
                    runner.restore_objective(objective, {'objective_state': {}})
                self.assertTrue(all(k.startswith('selector.') for k in objective.state_dict()))
                optimizer = torch.optim.SGD(selector.parameters(), lr=.01)
                with patch.object(optimizer, 'step', wraps=optimizer.step) as step:
                    parts, retries, _ = runner.update_batch(objective, optimizer,
                        torch.amp.GradScaler('cpu', enabled=False), batch, 0.,
                        Runtime(0, 1, torch.device('cpu')), cap, problem=problem)
                self.assertEqual((step.call_count, retries, selector.calls), (1, 0, 1024 // cap))
                self.assertTrue(torch.isfinite(parts).all())
                outcomes.append((selector.weight.detach().clone(), parts))
            for first, second in zip(*outcomes):
                torch.testing.assert_close(first, second, rtol=1e-6, atol=1e-7)

    @torch.no_grad()
    def test_label_isolation_true_n_includes_depot_and_unseen_n_fallback(self):
        model, _, _ = runner.initialize_shared(self.initial, self.geometry, runner.ARMS[0])
        model.eval()
        for problem in ('TSP', 'CVRP'):
            batch = input_batch(problem)
            batch['n'] = torch.tensor([4, 5])
            batch['node_mask'][1, 5] = False
            original = model(batch)
            task = self.priors['tasks'][problem]
            expected = torch.tensor([task['conditional_prior'][0], task['task_prior']])
            torch.testing.assert_close(original['conditional_prior'], expected, rtol=0, atol=0)
            if problem == 'CVRP':
                self.assertEqual(batch['node'][0, 0, 3], 1.)
                self.assertEqual(batch['n'][0], batch['node_mask'][0].sum())
            polluted = dict(batch, **{key: object() for key in LABEL_KEYS},
                            winner=object(), solver_outputs=object())
            raw = runner.raw_inputs(polluted)
            self.assertFalse(set(raw).intersection(LABEL_KEYS))
            actual = model(raw)
            for name in original:
                torch.testing.assert_close(original[name], actual[name], rtol=0, atol=0)
            self.assertTrue(torch.equal(original['logits'].argmax(-1), original['utility'].argmax(-1)))


if __name__ == '__main__':
    unittest.main()
