"""Focused R91 CPU integration checks; no GPU, data scans or held-out inference."""

import copy
import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import torch

from code.unified_selector.registry import GLOBAL_SOLVERS, PROBLEMS
from . import r91_experiment as runner
from . import r91_reporting as reporter
from .r76_experiment import PART_KEYS, Runtime, save_checkpoint, write_json
from .r77_data import parameter_hash
from .r88_model import make_model as role_model
from .r91_objective import ProbabilityLinkObjective, probability_links
from .test_r76 import CountingSGD, fixture_batch
from .test_r80 import fresh_fixture


def lock_record():
    return dict(experiment='R91', locked=True, arms=list(runner.ARMS), seed=2,
        batch_size=1024, max_updates=57600, warmup_updates=4320, eval_every=1440,
        role_arm='B_execution_anchor', role_slot=3, new_parameter_count=0,
        A_utility_ce=dict(ce='legal-masked raw utility', risk='original maximin probabilities',
            rdrop='original maximin symmetric KL'),
        B_utility_probabilities=dict(ce='legal-masked raw utility',
            risk='legal-masked raw utility probabilities', rdrop='legal-masked raw utility symmetric KL'))


def config_fixture(params, arm):
    return dict(experiment='R91', r91_schema_version=1, arm=arm, seed=2,
        initialization='scratch_shared_r77', model_params=params,
        solver_names=GLOBAL_SOLVERS, max_successful_updates=57600,
        model_factory=runner.MODEL_FACTORY, role_arm=runner.MODEL_ROLE_ARM,
        selection_head=runner.SELECTION_HEAD, objective=runner.objective_spec(arm))


def native_batch(problem='TSP', size=3):
    batch, _ = fixture_batch(problem, size)
    if problem == 'TSP':
        node = batch['node'].clone()
        node[..., 2:] = 0.
        batch = dict(batch, node=node)
    return batch


class FailOnce(ProbabilityLinkObjective):
    def __init__(self, selector, arm):
        super().__init__(selector, arm)
        self.failed = False

    def forward(self, batch, weight):
        loss, parts = super().forward(batch, weight)
        if not self.failed:
            self.failed = True
            loss = loss * loss.new_tensor(float('nan'))
        return loss, parts


class R91RunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        guard = patch('torch.cuda._lazy_init', side_effect=AssertionError('CPU tests cannot initialize CUDA'))
        guard.start()
        self.addCleanup(guard.stop)

    def test_cli_matches_launcher_and_describe_does_not_load_data(self):
        with patch.object(torch, 'load', side_effect=AssertionError('CLI cannot open data')):
            for arm in runner.ARMS:
                with redirect_stdout(io.StringIO()):
                    described = runner.main(['describe', '--arm', arm])
                self.assertEqual((described['experiment'], described['role_arm'], described['new_parameter_count']),
                    ('R91', 'B_execution_anchor', 0))
                self.assertEqual((described['batch_size'], described['max_updates'],
                    described['warmup_updates'], described['eval_every']), (1024, 57600, 4320, 1440))
                self.assertEqual(described['objective']['probability_links'], probability_links(arm))
                for command in ('train', 'test'):
                    args = runner.parser().parse_args([command, '--root', '/tmp/r91', '--arm', arm,
                        '--gpu-type', '3090', '--large-micro-cap', '352', '--eval-batch-size', '256',
                        '--wandb-mode', 'offline'])
                    self.assertEqual((args.experiment, args.gpu_type, args.large_micro_cap, args.eval_batch_size),
                        ('R91', '3090', 352, 256))
                for command in ('lib', 'report', 'status'):
                    self.assertEqual(reporter.parser().parse_args([command, '--arm', arm]).command, command)
        self.assertEqual(set(runner.task_order(57600)), set(PROBLEMS))
        self.assertEqual((runner.learning_rate(4320), runner.learning_rate(57600)), (1e-4, 1e-6))
        self.assertEqual(reporter.FACTORY, 'code.V4.r91_experiment')
        self.assertEqual(reporter.parser().parse_args(['compare']).command, 'compare')

    def test_both_arms_have_identical_r88b_fresh_state_and_no_new_parameters(self):
        initial, geometry = fresh_fixture()
        hashes, params_seen = [], []
        for arm in runner.ARMS:
            model, params, receipt = runner.initialize_shared(initial, geometry, arm)
            self.assertEqual(parameter_hash(dict(model.named_parameters())), initial['trainable_sha256'])
            self.assertEqual((receipt['new_parameter_count'], receipt['new_parameter_names'],
                receipt['removed_parameter_names']), (0, [], []))
            self.assertFalse(receipt['historical_checkpoint_loaded'])
            self.assertEqual(params['role_arm'], 'B_execution_anchor')
            torch.testing.assert_close(model.instance_encoder.geometry_residual.mean,
                torch.as_tensor(geometry['mean']), rtol=0, atol=0)
            reference = role_model(params, 'B_execution_anchor', state_dict=model.state_dict())
            self.assertIs(type(model), type(reference))
            hashes.append(parameter_hash(model.state_dict()))
            params_seen.append(params)
        self.assertEqual(hashes[0], hashes[1])
        self.assertEqual(params_seen[0], params_seen[1])

    def test_all18_forwards_and_dropout_rng_are_exact_r88b_with_labels_isolated(self):
        initial, geometry = fresh_fixture()
        first, params, _ = runner.initialize_shared(initial, geometry, runner.ARMS[0])
        second, _, _ = runner.initialize_shared(initial, geometry, runner.ARMS[1])
        reference = role_model(params, 'B_execution_anchor', state_dict=first.state_dict())
        for problem in PROBLEMS:
            batch = native_batch(problem, 2)
            torch.manual_seed(91)
            expected = reference(runner.engine.inputs_only(batch))
            rng = torch.get_rng_state()
            for model in (first, second):
                torch.manual_seed(91)
                actual = model(dict(batch, ind=torch.full_like(batch['ind'], -999)))
                for key in expected:
                    torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
                torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)

    def test_objective_hook_is_stateless_and_uses_main_api(self):
        initial, geometry = fresh_fixture()
        for arm in runner.ARMS:
            model = runner.initialize_shared(initial, geometry, arm)[0]
            objective = runner.make_objective(model, arm)
            self.assertIs(type(objective), ProbabilityLinkObjective)
            self.assertEqual(objective.arm, arm)
            self.assertEqual(set(objective.state_dict()), {'selector.' + name for name in model.state_dict()})
            self.assertEqual(runner.checkpoint_extras(objective), {})
            runner.restore_objective(objective, dict(args=dict(arm=arm)))
            with self.assertRaises(ValueError):
                runner.restore_objective(objective, dict(objective_state={}))
            with self.assertRaises(ValueError):
                runner.restore_objective(objective, dict(args=dict(arm='A_baseline')))

    def test_real_micro_updates_and_amp_replay_apply_one_step(self):
        initial, geometry = fresh_fixture()
        batch = native_batch(size=5)
        for arm in runner.ARMS:
            outcomes = []
            for fail in (False, True):
                model = runner.initialize_shared(initial, geometry, arm)[0]
                objective = FailOnce(model, arm) if fail else runner.make_objective(model, arm)
                optimizer = CountingSGD(model.parameters())
                torch.manual_seed(91)
                before = parameter_hash(model.state_dict())
                parts, retries, extras = runner.update_batch(objective, optimizer,
                    torch.amp.GradScaler('cpu', enabled=False), batch, .1,
                    Runtime(0, 1, torch.device('cpu')), cap=2, problem='TSP')
                self.assertEqual((optimizer.steps, retries, extras), (1, int(fail), {}))
                self.assertEqual(tuple(parts.shape), (len(PART_KEYS),))
                self.assertTrue(torch.isfinite(parts).all())
                self.assertNotEqual(before, parameter_hash(model.state_dict()))
                outcomes.append((parameter_hash(model.state_dict()), parts, torch.get_rng_state()))
            self.assertEqual(outcomes[0][0], outcomes[1][0])
            torch.testing.assert_close(outcomes[0][1], outcomes[1][1], rtol=0, atol=0)
            torch.testing.assert_close(outcomes[0][2], outcomes[1][2], rtol=0, atol=0)

    def test_checkpoint_strict_factory_state_hash_and_no_training_dependencies(self):
        initial, geometry = fresh_fixture()
        for arm in runner.ARMS:
            model, params, _ = runner.initialize_shared(initial, geometry, arm)
            config = config_fixture(params, arm)
            state = dict(args=config, state_hash=runner.json_hash(config), model=model.state_dict())
            with tempfile.TemporaryDirectory() as temporary:
                path = Path(temporary) / 'best.pt'
                save_checkpoint(path, state)
                with patch.object(runner, 'make_objective', side_effect=AssertionError('no training objective')):
                    restored, _ = runner.load_checkpoint(path)
                self.assertEqual(parameter_hash(model.state_dict()), parameter_hash(restored.state_dict()))
                for key, value in (('experiment', 'R88'), ('r91_schema_version', 2),
                        ('model_factory', 'code.V4.r88_experiment.load_checkpoint'),
                        ('role_arm', 'A_canonical_anchor'), ('objective', {})):
                    invalid = dict(config, **{key: value})
                    save_checkpoint(path, dict(state, args=invalid, state_hash=runner.json_hash(invalid)))
                    with self.assertRaises(ValueError):
                        runner.load_checkpoint(path)
                for mutation in (dict(state_hash='changed'), dict(objective_state={})):
                    save_checkpoint(path, dict(state, **mutation))
                    with self.assertRaises(ValueError):
                        runner.load_checkpoint(path)
                bad_model = dict(state['model'], unexpected_parameter=torch.zeros(1))
                save_checkpoint(path, dict(state, model=bad_model))
                with self.assertRaises(RuntimeError):
                    runner.load_checkpoint(path)

    def test_protocol_requires_main_lock_before_engine_and_rejects_changed_roles_or_links(self):
        args = runner.parser().parse_args(['train', '--arm', runner.ARMS[0]])
        with tempfile.TemporaryDirectory() as temporary:
            args.root = Path(temporary)
            with patch.object(runner.engine, 'train') as train:
                with self.assertRaises(FileNotFoundError):
                    runner.train(args, torch.device('cpu'))
                train.assert_not_called()
                record = lock_record()
                write_json(args.root / 'protocol_lock.json', record)
                runner.train(args, torch.device('cpu'))
                self.assertIs(train.call_args.kwargs['integration'], runner)
            for mutation in (dict(role_arm='A_canonical_anchor'), dict(role_slot=7),
                    dict(gpu_type='4090'), dict(large_micro_cap=384), dict(experiment='R88'),
                    dict(A_utility_ce=dict(record['A_utility_ce'], risk='utility'))):
                write_json(args.root / 'protocol_lock.json', dict(record, **mutation))
                with self.assertRaises(ValueError):
                    runner.protocol_lock(args)

    def test_protocol_and_test_integration_keep_original_evaluation_ce(self):
        initial, geometry = fresh_fixture()
        with tempfile.TemporaryDirectory() as temporary:
            args = runner.parser().parse_args(['train', '--arm', runner.ARMS[0], '--root', temporary])
            args.prepared_root, args.cache_root = Path(temporary), Path(temporary) / 'cache'
            write_json(args.root / 'protocol_lock.json', lock_record())
            write_json(args.prepared_root / 'prepared.json', {})
            ready = dict(trainable_sha256=initial['trainable_sha256'])
            manifest = dict(splits=dict(train={p: dict(pool=['x'], pool_ids=[0]) for p in PROBLEMS}))
            configs = []
            for arm in runner.ARMS:
                args.arm = arm
                _, params, _ = runner.initialize_shared(initial, geometry, arm)
                config = runner.protocol(args, ready, manifest, geometry, {}, params,
                    runner.task_order(runner.MAX_UPDATES), runner.micro_caps('3090', 352))
                runner.validate_checkpoint_config(config)
                self.assertEqual(config['objective']['probability_links'], probability_links(arm))
                self.assertEqual((config['train_ce_probability_link'], config['evaluation_ce_probability_link']),
                    ('utility', 'maximin'))
                self.assertEqual((config['role_slot'], config['new_parameter_count']), (3, 0))
                self.assertFalse(config['same_role_reference']['retrained'])
                configs.append(config)
            reporter.same_data(configs[0], configs[1], matched=True)
            changed = copy.deepcopy(configs[1])
            changed['model_params']['role_arm'] = 'A_canonical_anchor'
            with self.assertRaises(ValueError):
                reporter.same_data(configs[0], changed, matched=True)
            with patch.object(runner.engine, 'test', return_value='delegated') as test:
                self.assertEqual(runner.test(args, torch.device('cpu')), 'delegated')
                self.assertIs(test.call_args.kwargs['integration'], runner)

    def test_joint_val_screen_requires_strict_top1_and_nonincreasing_cost_regret_against_both(self):
        config, lock = dict(arm=runner.ARMS[0]), dict(checkpoint_sha256='current')
        baseline = dict(macro_top1=.5, macro_mean_cost=2., macro_actual_regret_pct=1.)
        references = {name: dict(val=dict(macro=baseline), lock=dict(checkpoint_sha256=name))
                      for name in ('R84A', 'R88B')}
        rows = reporter.val_screen(config, lock, dict(macro=dict(baseline, macro_top1=.51)), references)
        self.assertTrue(all(row['joint_val_pass'] for row in rows))
        for candidate in (baseline, dict(baseline, macro_top1=.51, macro_mean_cost=2.01),
                dict(baseline, macro_top1=.51, macro_actual_regret_pct=1.01)):
            self.assertFalse(any(row['joint_val_pass'] for row in
                reporter.val_screen(config, lock, dict(macro=candidate), references)))
        different = copy.deepcopy(references)
        different['R88B']['val']['macro'] = dict(baseline, macro_top1=.52)
        self.assertFalse(any(row['joint_val_pass'] for row in reporter.val_screen(
            config, lock, dict(macro=dict(baseline, macro_top1=.51)), different)))

    def test_frozen_targets_must_be_all36_and_remain_bound_to_file_hash(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'targets.json'
            write_json(path, {})
            identity = dict(path=str(path), identity={}, categories={})
            targets = [('TSP', str(i)) for i in range(36)]
            with patch.object(reporter.helpers, 'load_targets', return_value=(targets, identity)):
                selected, receipt = reporter.load_targets({}, {}, {})
                self.assertEqual(selected, targets)
                self.assertEqual(receipt['target_count'], 36)
                self.assertEqual(len(receipt['sha256']), 64)
            with patch.object(reporter.helpers, 'load_targets', return_value=(targets[:-1], identity)):
                with self.assertRaises(ValueError):
                    reporter.load_targets({}, {}, {})


if __name__ == '__main__':
    unittest.main()
