"""Focused unittest CPU contracts; no pytest, CUDA, solver or historical writes."""

import copy
import io
import tempfile
import unittest
from collections import Counter
from contextlib import ExitStack, redirect_stdout
from pathlib import Path
from unittest.mock import patch
from unittest.mock import Mock

import numpy as np
import torch
from torch import nn

from . import r122_experiment as runner
from .r122_objective import HardDecisionController, HardDecisionObjective, QUERY_INPUTS


def controller(problems=('x', 'y')):
    return HardDecisionController(problems, train_regret_budget=.02,
        train_fixed_miss_budget=.1, train_fixed_population=.25)


class TinySelector(nn.Module):
    def __init__(self, dropout=0.):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(3, 2)
        self.seen, self.keys = [], []

    def forward(self, batch):
        self.keys.append(set(batch))
        output = self.head(self.dropout(batch['node']))
        self.seen.append(output.detach().clone())
        return dict(utility=output, solver_mask=batch['solver_mask'])


class CountingSGD(torch.optim.SGD):
    def __init__(self, parameters):
        super().__init__(parameters, lr=.01)
        self.steps = 0

    def step(self, closure=None):
        self.steps += 1
        return super().step(closure)


def batch():
    costs = torch.tensor([[1., 1.2], [1.3, 1.], [1., 1.1], [1.4, 1.], [1., 1.5]], dtype=torch.float64)
    return dict(node=torch.arange(15, dtype=torch.float32).reshape(5, 3) / 10.,
        costs=costs, ind=costs.argmin(1), pool_ids=torch.tensor([8, 2]),
        solver_mask=torch.ones(5, 2, dtype=torch.bool),
        fixed_target=torch.tensor([True, False, True, False, True]),
        winner=torch.ones(5), route_teacher=torch.ones(5), unknown_label=torch.ones(5),
        cmp_valid=torch.ones(5, 1, dtype=torch.bool))


def cpu_runtime():
    return runner.Runtime(rank=0, world_size=1, device=torch.device('cpu'))


class RetryScaler:
    """CPU AMP-step simulator: overflow before step, or skip the first step."""

    def __init__(self, objective, *, skip_step=False, always_bad=False):
        self.objective, self.skip_step, self.always_bad = objective, skip_step, always_bad
        self.scale_value, self.attempts, self.skipped = 8., 0, False

    def is_enabled(self):
        return True

    def get_scale(self):
        return self.scale_value

    def scale(self, loss):
        return loss

    def unscale_(self, optimizer):
        if int(self.objective.controller.committed_updates) != 0:
            raise AssertionError('Controller advanced during a retry')
        self.attempts += 1
        if self.always_bad or (not self.skip_step and self.attempts == 1):
            next(self.objective.selector.parameters()).grad.fill_(torch.inf)

    def step(self, optimizer):
        if self.skip_step and self.attempts == 1:
            self.skipped = True
        else:
            optimizer.step()

    def update(self, new_scale=None):
        if new_scale is not None:
            self.scale_value = new_scale
        elif self.skipped:
            self.scale_value *= .5
            self.skipped = False


def val_result(risk, miss, top1):
    return dict(macro=dict(macro_top1=top1), hard_decision=dict(macro=dict(
        hard_regret=risk, hard_fixed_miss=miss, native_top1=top1)))


class RunnerCPUContracts(unittest.TestCase):
    def setUp(self):
        self.previous_threads = torch.get_num_threads()
        torch.set_num_threads(2)

    def tearDown(self):
        torch.set_num_threads(self.previous_threads)

    def test_budget_parser_fp64_native_and_stable_ties(self):
        data = dict(costs=np.array([[1. + 1e-10, 1.], [1., 1. + 2e-10]], dtype=np.float64),
            winner=np.array([1, 0], dtype=np.int64), logits=np.zeros((2, 2), dtype=np.float32),
            pool=np.array(['first', 'second']), pool_ids=np.array([8, 2], dtype=np.int64))
        result = runner.hard_statistics(data, ('first',))
        self.assertEqual(result['native_top1'], .5)
        self.assertGreater(result['hard_regret'], 0.)
        self.assertEqual(result['hard_fixed_miss'], .5)
        self.assertEqual(result['fixed_population'], .5)
        with self.assertRaisesRegex(ValueError, 'FP64'):
            runner.hard_statistics(dict(data, costs=data['costs'].astype(np.float32)), ('first',))
        with self.assertRaises(ValueError):
            runner.hard_statistics(dict(data, winner=np.array([0, 0], dtype=np.int64)), ('first',))

    def test_fixed_target_uses_native_deployment_not_alias_or_dynamic_group(self):
        pools = dict(x=['reld_mt', 'MVMOE'], y=['a'])
        record = dict(targets=[dict(problem='x', method='reld_mt'), dict(problem='y', method='a')])
        self.assertEqual(runner.fixed_targets(record, pools, expected_count=2), dict(x=('reld_mt',), y=('a',)))
        with self.assertRaisesRegex(ValueError, 'native deployment'):
            runner.fixed_targets(dict(targets=[dict(problem='x', method='RELD_MOEL')]), pools, expected_count=1)
        with self.assertRaisesRegex(ValueError, 'unique historical'):
            runner.fixed_targets(record, pools)

    def test_macro_is_equal_task_and_fixed_frequency_is_unconditional(self):
        small = dict(n=2, hard_regret=.04, hard_fixed_miss=.5, fixed_population=.5, native_top1=.5)
        large = dict(n=2000, hard_regret=0., hard_fixed_miss=0., fixed_population=0., native_top1=1.)
        macro = runner.macro_hard(dict(x=small, y=large))
        self.assertEqual(macro['hard_regret'], .02)
        self.assertEqual(macro['hard_fixed_miss'], .25)
        self.assertEqual(macro['fixed_population'], .25)

    def test_train_budgets_never_read_val_selection_thresholds(self):
        reference = dict(train=dict(source_split='train', macro=dict(
            hard_regret=.03, hard_fixed_miss=.02, fixed_population=.1)),
            val_selection=dict(source_split='val', macro=dict(hard_regret=float('nan'))))
        item = runner.make_controller(reference)
        torch.testing.assert_close(item.budget, torch.tensor([.03, .02], dtype=torch.float64))
        with self.assertRaisesRegex(ValueError, 'TRAIN'):
            runner.make_controller(dict(reference, train=dict(reference['train'], source_split='val')))

    def test_positive_query_allowlist_one_forward_and_pure_primal_loss(self):
        model = TinySelector()
        objective = HardDecisionObjective(model, controller())
        before = copy.deepcopy(objective.controller.state_dict())
        loss, _ = objective(batch())
        loss.backward()
        self.assertEqual(len(model.seen), 1)
        self.assertEqual(model.keys, [set(batch()).intersection(QUERY_INPUTS)])
        self.assertFalse(set(('costs', 'ind', 'fixed_target', 'winner', 'route_teacher',
                              'unknown_label', 'cmp_valid')) & model.keys[0])
        for key, value in objective.controller.state_dict().items():
            torch.testing.assert_close(value, before[key], rtol=0., atol=0.)

    def test_cache_restores_fp64_and_fixed_target_from_native_winner_names(self):
        item = object.__new__(runner.NativeConstraintCache)
        item.size, item.pool, item.fixed_methods = 2, ['MVMOE', 'reld_mt'], ('reld_mt',)
        item.raw = dict(costs=np.array([[1. + 1e-10, 1.], [1., 1. + 2e-10]]),
                        winner=np.array([1, 0], dtype=np.int64))

        def base_batch(self, indices, device):
            rows = np.asarray(indices)
            return dict(costs=torch.tensor(self.raw['costs'][rows], dtype=torch.float32),
                        ind=torch.from_numpy(self.raw['winner'][rows]))

        with patch.object(runner.ScenarioCache, 'batch', base_batch):
            gathered = item.batch(torch.tensor([1, 0, 1]))
        self.assertEqual(gathered['costs'].dtype, torch.float64)
        self.assertEqual(gathered['fixed_target'].tolist(), [False, True, False])
        torch.testing.assert_close(gathered['costs'], torch.from_numpy(item.raw['costs'][[1, 0, 1]]), rtol=0., atol=0.)

    def test_microbatch_row_weighting_one_commit_after_complete_optimizer_step(self):
        torch.manual_seed(9)
        first = TinySelector()
        second = copy.deepcopy(first)
        outputs = []
        for model, cap in ((first, 2), (second, 5)):
            objective = HardDecisionObjective(model, controller())
            optimizer = CountingSGD(model.parameters())
            original = objective.controller.commit
            calls = []

            def commit(problem, **values):
                self.assertEqual(optimizer.steps, 1)
                self.assertEqual(len(model.seen), (5 + cap - 1) // cap)
                calls.append(values)
                return original(problem, **values)

            with patch.object(objective.controller, 'commit', commit):
                result, retries, hard = runner.update_batch(objective, optimizer,
                    torch.amp.GradScaler('cpu', enabled=False), batch(), cpu_runtime(), cap,
                    problem='x', successful_update=1)
            self.assertEqual((retries, optimizer.steps, len(calls)), (0, 1, 1))
            self.assertEqual(int(objective.controller.committed_updates), 1)
            self.assertEqual(int(objective.controller.completed_cycles), 0)
            self.assertEqual(objective.controller.task_count.tolist(), [1, 0])
            outputs.append((result, hard))
        torch.testing.assert_close(outputs[0][0], outputs[1][0], rtol=1e-6, atol=1e-6)
        for a, b in zip(first.parameters(), second.parameters()):
            torch.testing.assert_close(a, b, rtol=1e-6, atol=1e-6)
        self.assertAlmostEqual(outputs[0][1]['hard_regret'], outputs[1][1]['hard_regret'])

    def test_amp_overflow_and_skipped_step_retry_same_rng_batch_no_commit(self):
        for skip_step in (False, True):
            with self.subTest(skip_step=skip_step):
                torch.manual_seed(31)
                model = TinySelector(dropout=.4)
                objective = HardDecisionObjective(model, controller())
                optimizer = CountingSGD(model.parameters())
                scaler = RetryScaler(objective, skip_step=skip_step)
                _, retries, _ = runner.update_batch(objective, optimizer, scaler, batch(), cpu_runtime(), 2,
                                                   problem='x', successful_update=1)
                self.assertEqual((retries, scaler.attempts, optimizer.steps), (1, 2, 1))
                self.assertEqual(int(objective.controller.committed_updates), 1)
                self.assertEqual(len(model.seen), 6)
                for initial, repeated in zip(model.seen[:3], model.seen[3:]):
                    torch.testing.assert_close(initial, repeated, atol=0., rtol=0.)

    def test_failed_replayed_or_evaluation_update_cannot_commit(self):
        model = TinySelector()
        objective = HardDecisionObjective(model, controller())
        optimizer = CountingSGD(model.parameters())
        original = copy.deepcopy(objective.state_dict())
        with self.assertRaisesRegex(RuntimeError, 'no optimizer/controller'):
            runner.update_batch(objective, optimizer, RetryScaler(objective, always_bad=True),
                batch(), cpu_runtime(), 2, problem='x', successful_update=1, max_retries=2)
        self.assertEqual(optimizer.steps, 0)
        for key, value in objective.state_dict().items():
            torch.testing.assert_close(value, original[key], atol=0., rtol=0.)
        with self.assertRaisesRegex(ValueError, 'next complete'):
            runner.update_batch(objective, optimizer, torch.amp.GradScaler('cpu', enabled=False),
                batch(), cpu_runtime(), 2, problem='x', successful_update=2)
        objective.eval()
        with self.assertRaises(ValueError):
            runner.update_batch(objective, optimizer, torch.amp.GradScaler('cpu', enabled=False),
                batch(), cpu_runtime(), 2, problem='x', successful_update=1)

    def test_checkpoint_roundtrip_retains_partial_equal_task_feedback_cycle(self):
        first = controller()
        first.commit('x', hard_regret=.05, hard_fixed_miss=.2, successful_update=1)
        stream = io.BytesIO()
        torch.save(first.state_dict(), stream)
        stream.seek(0)
        restored = controller()
        restored.load_state_dict(torch.load(stream, weights_only=True))
        for item in (first, restored):
            item.commit('y', hard_regret=.01, hard_fixed_miss=0., successful_update=2)
        self.assertEqual(int(restored.completed_cycles), 1)
        self.assertEqual(int(restored.committed_updates), 2)
        for key, value in first.state_dict().items():
            torch.testing.assert_close(value, restored.state_dict()[key], rtol=0., atol=0.)

    def test_val_feasibility_top1_regret_earliest_and_normalized_fallback(self):
        reference = dict(hard_regret=.02, hard_fixed_miss=.1)
        key = lambda risk, miss, top1, step: runner.selection_key(val_result(risk, miss, top1), reference, step)
        self.assertLess(key(.02, .1, .7, 720), key(.021, .1, .99, 720))
        self.assertLess(key(.02, .1, .8, 1440), key(.01, .01, .7, 720))
        self.assertLess(key(.01, .1, .8, 1440), key(.02, .01, .8, 720))
        self.assertLess(key(.01, .1, .8, 720), key(.01, .1, .8, 1440))
        self.assertLess(key(.03, .1, .7, 720), key(.02, .2, .99, 720))
        self.assertAlmostEqual(key(.03, .1, .7, 720)[1], .5)
        with self.assertRaises(ValueError):
            key(float('nan'), .1, .8, 720)

    def test_resume_controller_count_and_source_cursors(self):
        config = dict(marker='cpu test')
        order = runner.task_order(runner.MAX_UPDATES)
        counts = Counter(order[:720])
        state = dict(args=config, state_hash=runner.engine.json_hash(config), successful_updates=720,
            pending_evaluation=None, history=[{}], updates_by_problem=dict(counts),
            stream_draws={p: counts[p] * 1024 for p in runner.PROBLEMS},
            controller=dict(committed_updates=torch.tensor(720), completed_cycles=torch.tensor(40),
                task_count=torch.zeros(18, dtype=torch.long), task_sum=torch.zeros(18, 2)))
        runner.validate_resume(state, config, order)
        broken = copy.deepcopy(state)
        broken['controller']['committed_updates'] -= 1
        with self.assertRaisesRegex(ValueError, 'controller'):
            runner.validate_resume(broken, config, order)
        broken = copy.deepcopy(state)
        broken['stream_draws'][runner.PROBLEMS[0]] += 1024
        with self.assertRaisesRegex(ValueError, 'source cursors'):
            runner.validate_resume(broken, config, order)

    def test_unique_locks_refuse_overwrite(self):
        directory = runner.DEFAULT_ROOT / 'preparation'
        directory.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='cpu_test_', dir=directory) as temporary:
            path = Path(temporary) / 'lock.json'
            runner.immutable_json(path, dict(checkpoint='a'))
            runner.immutable_json(path, dict(checkpoint='a'))
            with self.assertRaisesRegex(PermissionError, 'Immutable'):
                runner.immutable_json(path, dict(checkpoint='b'))
            self.assertEqual(runner.read_json(path), dict(checkpoint='a'))

    def test_resource_copy_accepts_new_path_but_rejects_changed_native_cost_or_source(self):
        directory = runner.DEFAULT_ROOT / 'preparation'
        with tempfile.TemporaryDirectory(prefix='cpu_resource_', dir=directory) as temporary:
            original, local = Path(temporary) / 'original', Path(temporary) / 'node_local'
            costs = np.array([[1., 1.1]], dtype=np.float64)
            values = dict(costs=costs, winner=np.array([0], dtype=np.int64), node=np.zeros((1, 2), dtype=np.float32))
            expected = dict(source_files=[dict(path='frozen_source', size_bytes=17)],
                arrays={k: dict(file=k + '.npy', shape=list(v.shape), dtype=str(v.dtype)) for k, v in values.items()})
            for root in (original, local):
                runner.write_json(root / 'metadata.json', expected)
                for key, value in values.items():
                    np.save(root / (key + '.npy'), value, allow_pickle=False)
            runner.validate_cache_copy(local, expected, label_reference=original)
            np.save(local / 'costs.npy', costs + .01, allow_pickle=False)
            with self.assertRaisesRegex(ValueError, 'frozen native costs/winner'):
                runner.validate_cache_copy(local, expected, label_reference=original)
            np.save(local / 'costs.npy', costs, allow_pickle=False)
            runner.write_json(local / 'metadata.json', dict(expected, source_files=[]))
            with self.assertRaisesRegex(ValueError, 'source identity'):
                runner.validate_cache_copy(local, expected, label_reference=original)

    def test_existing_test_cache_prefers_node_local_and_falls_back_to_shared_copy(self):
        directory = runner.DEFAULT_ROOT / 'preparation'
        with tempfile.TemporaryDirectory(prefix='cpu_resource_test_', dir=directory) as temporary:
            root = Path(temporary)
            local, shared = root / 'node_local', root / 'shared_original'
            task = dict(complete=True, rows=4500, pool=['a'], pool_ids=[0])
            manifest = dict(splits=dict(test=dict(x=task)))
            runner.write_json(root / 'prepared/source_cache_manifest.json', manifest)
            runner.write_json(shared / 'test/x/metadata.json', task)
            config = dict(prepared_root=str(root / 'prepared'), candidate_pools=dict(x=['a']), candidate_pool_ids=dict(x=[0]))
            args = runner.argparse.Namespace(cache_root=local)
            with patch.object(runner, 'PROBLEMS', ('x',)), patch.object(runner, 'FROZEN_TEST_CACHE_ROOT', shared):
                with patch.object(runner, 'authorize_test', return_value=({'locked': True}, config)):
                    with patch.object(runner, 'validate_cache_copy') as validate:
                        self.assertEqual(runner.ensure_test_cache(args, {'locked': True}), manifest)
                        self.assertEqual(args.cache_root, shared)
                        validate.assert_called_once_with(shared / 'test/x', task, label_reference=shared / 'test/x')
                    runner.write_json(local / 'test/x/metadata.json', task)
                    args.cache_root = local
                    with patch.object(runner, 'validate_cache_copy') as validate:
                        runner.ensure_test_cache(args, {'locked': True})
                        self.assertEqual(args.cache_root, local)
                        validate.assert_called_once_with(local / 'test/x', task, label_reference=shared / 'test/x')

    def test_fulltest_rejects_another_model_before_reading_any_test_data(self):
        lock = dict(selected_update=720, state_hash='protocol', checkpoint_sha256='checkpoint',
                    model_state_sha256='not-the-query-model')
        with patch.object(runner, 'authorize_test', return_value=(lock, {})):
            with patch.object(runner, 'private_engine', side_effect=AssertionError('Must not read TEST')):
                with self.assertRaisesRegex(PermissionError, 'exact locked checkpoint'):
                    runner.evaluate(TinySelector(), None, {}, {}, 'test', 'selected_best',
                                    720, 'protocol', 'checkpoint')

    def test_fixed_schedule_full18_and_private_engine_isolation(self):
        self.assertAlmostEqual(runner.learning_rate(1080), 1e-4)
        self.assertAlmostEqual(runner.learning_rate(14400), 1e-6)
        self.assertAlmostEqual(runner.learning_rate(1), 1e-4 / 1080)
        self.assertEqual(set(Counter(runner.task_order(14400)).values()), {800})
        original = runner.engine.MAX_UPDATES
        private = runner.private_engine()
        self.assertIsNot(private, runner.engine)
        self.assertEqual(private.MAX_UPDATES, 14400)
        self.assertEqual(runner.engine.MAX_UPDATES, original)
        self.assertEqual(original, 57600)
        self.assertIs(private.inputs_only, runner.query_inputs)

    def test_formal_cpu_train_rejected_without_model_artifacts(self):
        directory = runner.DEFAULT_ROOT / 'preparation'
        directory.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='cpu_test_', dir=directory) as temporary:
            root = Path(temporary) / 'run'
            with self.assertRaisesRegex(ValueError, 'explicitly assigned CUDA'):
                runner.main(['train', '--root', str(root), '--cache-root', temporary, '--device', 'cpu'])
            self.assertFalse(root.exists())

    def test_tiny_cpu_train_pending_val_resume_csv_and_lock_without_test(self):
        """Exercise the real train loop with four synthetic updates, not a result."""
        directory = runner.DEFAULT_ROOT / 'preparation'
        directory.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='cpu_spine_', dir=directory) as temporary:
            root = Path(temporary)
            args = runner.parser().parse_args(['train', '--root', temporary, '--device', 'cpu'])
            args.prepared_root = root / 'source'
            args.prepared_root.mkdir()
            args.preparation_root = root / 'preparation'
            args.cache_root = root / 'cache'
            (root / 'model').mkdir()
            torch.save({}, args.prepared_root / 'identicalinit_seed2.pt')
            reference = dict(train=dict(source_split='train', macro=dict(
                hard_regret=.02, hard_fixed_miss=.1, fixed_population=.25)),
                val_selection=dict(source_split='val', macro=dict(hard_regret=.02, hard_fixed_miss=.1)),
                fixed_targets=[dict(problem='x', method='b')], candidate_pools=dict(x=['a', 'b'], y=['a', 'b']),
                cache_root=str(args.cache_root), resource_cache_policy='synthetic resource fixture')
            config = dict(experiment='R122', synthetic_cpu_test=True)
            pending_failure = [True]

            class Cache:
                def batch(self, indices, device='cpu'):
                    sample = batch()
                    rows = torch.arange(len(indices)) % len(sample['ind'])
                    return {k: value[rows] if torch.is_tensor(value) and k != 'pool_ids' else value
                            for k, value in sample.items()}

                def close(self):
                    pass

            def initialize(*unused):
                torch.manual_seed(2)
                return TinySelector(), {}, dict(historical_checkpoint_loaded=False)

            def load(path, device):
                state = torch.load(path, map_location='cpu', weights_only=False)
                model = TinySelector()
                model.load_state_dict(state['model'])
                return model, state

            def evaluate(model, args, manifest, baselines, split, tag, step, state_hash, checkpoint_sha256=None):
                if split == 'val' and pending_failure[0]:
                    pending_failure[0] = False
                    raise RuntimeError('synthetic interrupted pending VAL')
                path = root / 'model/evaluation' / split / tag
                identity = dict(successful_updates=step, state_hash=state_hash,
                    model_state_sha256=runner.parameter_hash(model.state_dict()),
                    checkpoint_sha256=checkpoint_sha256, split=split, tag=tag)
                result = val_result(.01, .05, .7 + step / 100.)
                result.update(identity=identity, tag=tag, prediction_archive=str(path / 'rank000.npz'))
                runner.write_json(path / 'identity.json', identity)
                runner.engine.save_npz(path / 'rank000.npz', dict(synthetic=np.array([step])))
                runner.write_json(path / 'metrics.json', result)
                return result

            with ExitStack() as stack:
                stack.enter_context(patch.object(runner, 'PROBLEMS', ('x', 'y')))
                stack.enter_context(patch.object(runner, 'MAX_UPDATES', 4))
                stack.enter_context(patch.object(runner, 'EVAL_EVERY', 2))
                stack.enter_context(patch.object(runner, 'task_order', side_effect=lambda updates: ['x', 'y'] * (updates // 2)))
                stack.enter_context(patch.object(runner, 'task_seed', side_effect=lambda seed, problem: seed + ('x', 'y').index(problem)))
                stack.enter_context(patch.object(runner, 'fixed_targets', return_value=dict(x=('b',), y=())))
                stack.enter_context(patch.object(runner, 'load_preparation', return_value=(reference, {}, {}, {}, {}, {})))
                stack.enter_context(patch.object(runner, 'protocol', return_value=config))
                stack.enter_context(patch.object(runner.engine, 'initialize_shared', initialize))
                stack.enter_context(patch.object(runner.engine, 'wandb_run', return_value=Mock()))
                stack.enter_context(patch.object(runner, 'open_cache', side_effect=lambda *a, **k: Cache()))
                stack.enter_context(patch.object(runner, 'load_checkpoint', load))
                stack.enter_context(patch.object(runner, 'evaluate', evaluate))
                stack.enter_context(patch.object(runner, 'status_update'))
                no_test = stack.enter_context(patch.object(runner, 'test', side_effect=AssertionError('No auto TEST')))
                stack.enter_context(redirect_stdout(io.StringIO()))
                with self.assertRaisesRegex(RuntimeError, 'interrupted pending VAL'):
                    runner.train(args, torch.device('cpu'))
                saved = torch.load(root / 'model/last.pt', map_location='cpu', weights_only=False)
                self.assertEqual(saved['successful_updates'], 2)
                self.assertEqual(int(saved['controller']['committed_updates']), 2)
                self.assertIsNotNone(saved['pending_evaluation'])
                args.resume = True
                result = runner.train(args, torch.device('cpu'))
                no_test.assert_not_called()
            self.assertFalse(result['test_read'])
            self.assertEqual(result['selected_update'], 4)
            complete = runner.read_json(root / 'model/training_complete.json')
            self.assertEqual(complete['updates_by_problem'], dict(x=2, y=2))
            saved = torch.load(root / 'model/last.pt', map_location='cpu', weights_only=False)
            self.assertEqual(int(saved['controller']['committed_updates']), 4)
            self.assertEqual(int(saved['controller']['completed_cycles']), 2)
            self.assertTrue((root / 'model/metrics.csv').is_file())
            self.assertIn('train', (root / 'model/metrics.csv').read_text())
            lock = runner.read_json(root / 'model/final_test_lock.json')
            self.assertTrue(lock['selected_without_test'])
            self.assertEqual(result['val']['identity']['checkpoint_sha256'], lock['checkpoint_sha256'])


if __name__ == '__main__':
    unittest.main()
