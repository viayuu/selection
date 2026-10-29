"""Focused CPU witnesses for the R124 runner and immutable selection boundary."""

import copy
from collections import Counter
import io
import json
from contextlib import redirect_stdout
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from code.unified_selector.registry import GLOBAL_SOLVERS, PROBLEMS
from . import r124_experiment as runner
from .r124_model import FrozenTop2Selector, pair_objective
from .test_r124_model import fixture
from .test_r76 import FakeCache
from .test_r80 import fresh_fixture


def config(params):
    return dict(experiment='R124', arm='model', seed=2, initialization='scratch_r84_compatible_new_encoder',
        solver_names=list(GLOBAL_SOLVERS), max_successful_updates=7200, global_batch_size=1024,
        logical_global_batch=1024, warmup_updates=540, eval_every_updates=720,
        checkpoint_rule=runner.CHECKPOINT_RULE, inference_rule=runner.INFERENCE_RULE, loss=runner.LOSS,
        first_stage=dict(checkpoint_sha256=runner.EXPERT_SHA, updated_by_r124=False,
                         checkpoint_path='/fixture/external/best.pt'),
        candidate_pools=dict.fromkeys(PROBLEMS, ['a', 'b']), model_params=params)


class TinyPair(nn.Module):
    def __init__(self):
        super().__init__()
        self.head = nn.Linear(3, 2)

    def forward(self, batch, scores):
        values = self.head(batch['node'])
        return dict(pair_logits=torch.stack((values[:, 0] / 2, -values[:, 0] / 2), 1),
            pair_quality=values[:, 1] * .01,
            pair_indices=torch.tensor([[0, 1]], device=values.device).expand(len(values), -1),
            solver_mask=batch['solver_mask'])


def tiny_batch():
    costs = torch.tensor([[1., 1., 2.], [1.2, 1.1, 1.], [1.01, 1., 2.],
                          [1.2, 1.3, 1.], [1., 1.02, 2.]], dtype=torch.float64)
    return dict(node=torch.arange(15, dtype=torch.float32).reshape(5, 3) / 10,
        costs=costs, ind=torch.tensor([1, 2, 1, 2, 0]),
        stage1_scores=torch.tensor([[3., 2., 1.]]).expand(5, -1),
        solver_mask=torch.ones(5, 3, dtype=torch.bool), pool_ids=torch.tensor([4, 2, 8]))


class RunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_description_parser_schedule_without_data_or_weights(self):
        with patch.object(torch, 'load', side_effect=AssertionError('No weights for describe')):
            with redirect_stdout(io.StringIO()):
                result = runner.main(['describe'])
        self.assertEqual((result['batch_size'], result['max_updates'], result['eval_every'], result['warmup_updates']),
                         (1024, 7200, 720, 540))
        self.assertEqual(result['train_total'], 195948)
        for command in ('cache', 'train', 'test'):
            self.assertEqual(runner.parser().parse_args([command]).arm, 'model')
        self.assertEqual(Counter(runner.task_order(7200)), Counter(dict.fromkeys(PROBLEMS, 400)))
        self.assertEqual(runner.learning_rate(540), 1e-4)
        self.assertEqual(runner.learning_rate(7200), 1e-6)
        self.assertFalse(runner.LOSS['expected_risk'])
        self.assertEqual(runner.LOSS['quality_coefficient'], 1.)
        self.assertEqual(result['microbatch_caps']['TSP'], 128)
        self.assertEqual(result['microbatch_caps']['CVRP'], 128)
        self.assertEqual(result['microbatch_caps']['ATSP'], 1024)

    def test_both_loss_terms_have_full_logical_denominator(self):
        torch.manual_seed(2)
        first = TinyPair()
        second = copy.deepcopy(first)
        batch = tiny_batch()
        results = []
        for model, cap in ((first, 2), (second, 5)):
            optimizer = torch.optim.SGD(model.parameters(), lr=.01)
            result, retries, effective = runner.update_batch(model, optimizer,
                torch.amp.GradScaler('cpu', enabled=False), batch, torch.device('cpu'), cap)
            self.assertEqual((retries, effective), (0, cap))
            self.assertAlmostEqual(result['covered_fraction'], .6)
            results.append(result)
        for key in results[0]:
            self.assertAlmostEqual(results[0][key], results[1][key], places=5)
        for a, b in zip(first.parameters(), second.parameters()):
            torch.testing.assert_close(a, b, rtol=1e-6, atol=1e-7)

    def test_quality_is_separate_from_native_action_and_not_expected_risk(self):
        model, batch = fixture()
        model.eval()
        before = model(batch, batch['stage1_scores'])
        with torch.no_grad():
            model.quality_head[-1].weight.add_(100.)
        after = model(batch, batch['stage1_scores'])
        torch.testing.assert_close(before['difference'], after['difference'], rtol=0, atol=0)
        torch.testing.assert_close(before['logits'], after['logits'], rtol=0, atol=0)
        self.assertFalse(torch.equal(before['pair_quality'], after['pair_quality']))

    def test_fresh_compatible_parameters_and_strict_factory_roundtrip(self):
        initial, geometry = fresh_fixture()
        model, receipt = runner.initialize_fresh(initial, geometry, initial['model_params'])
        self.assertFalse(receipt['trained_selector_initialization'])
        self.assertFalse(receipt['first_stage_shared_parameters'])
        for name in receipt['copied_fresh_parameters']:
            torch.testing.assert_close(dict(model.named_parameters())[name], initial['trainable'][name], rtol=0, atol=0)
        self.assertTrue(any(x.startswith('stage_score.') for x in receipt['newly_initialized_parameters']))
        params = model.params
        cfg = config(params)
        state = dict(args=cfg, state_hash=runner.json_hash(cfg), model=model.state_dict(), successful_updates=720)
        _, batch = fixture()
        class FrozenStage(nn.Module):
            def __init__(self):
                super().__init__()
                self.scale = nn.Parameter(torch.ones(()))

            def forward(self, inputs):
                return dict(logits=batch['stage1_scores'] * self.scale)

        frozen = FrozenStage()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'synthetic.pt'
            torch.save(state, path)
            restored, _ = runner.load_reranker(path)
            with patch.object(runner, 'file_hash', return_value=runner.EXPERT_SHA), \
                    patch.object(runner, 'load_expert', return_value=(frozen, {})) as loader:
                deployed, _ = runner.load_checkpoint(path)
            self.assertIsInstance(deployed, FrozenTop2Selector)
            loader.assert_called_once_with('R77B', 'cpu')
        model.eval()
        restored.eval()
        torch.testing.assert_close(model(batch, batch['stage1_scores'])['logits'],
                                   restored(batch, batch['stage1_scores'])['logits'], rtol=0, atol=0)
        torch.testing.assert_close(deployed(batch)['logits'], restored(batch, batch['stage1_scores'])['logits'], rtol=0, atol=0)
        self.assertEqual(set(deployed.state_dict()), set(state['model']))
        self.assertEqual(runner.parameter_hash(deployed.state_dict()), runner.parameter_hash(state['model']))
        invalid = dict(initial, initialization='trained')
        with self.assertRaisesRegex(ValueError, 'trained selector'):
            runner.initialize_fresh(invalid, geometry, params)

    def test_score_cache_rejects_pool_rows_and_source_misalignment(self):
        arrays = dict(scores=np.ones((2, 3), dtype=np.float32), indices=np.array([3, 7]),
            release_source_indices=np.array([1003, 2007]), pool=np.array(['a', 'b', 'c']), pool_ids=np.array([1, 4, 8]))
        expected = dict(indices=arrays['indices'], pool=arrays['pool'], pool_ids=arrays['pool_ids'],
                        release_source_indices=arrays['release_source_indices'])
        runner.validate_score_arrays(arrays, **expected)
        for key in ('indices', 'pool', 'pool_ids', 'release_source_indices'):
            bad = dict(arrays, **{key: arrays[key][::-1]})
            with self.assertRaisesRegex(ValueError, 'identity/order'):
                runner.validate_score_arrays(bad, **expected)
        with self.assertRaises(ValueError):
            runner.validate_score_arrays(dict(arrays, scores=np.full((2, 3), np.nan)), **expected)

    def test_cache_real_writer_roundtrip_and_reuse_without_input_rebuild(self):
        source = FakeCache('TSP')
        pools = dict(TSP=dict(pool=source.pool, pool_ids=source.pool_ids))
        context = dict(indices=dict(TSP=np.array([1, 3], dtype=np.int64)),
            manifest=dict(splits=dict(train=pools, val=pools)), meta=dict(candidate_aliases={'reld_mt': 'RELD_MOEL'}))

        def identity(args, ctx, problem, split, rows):
            return dict(pool=source.pool, pool_ids=source.pool_ids, split=split, rows=len(rows)), rows + 1000

        with tempfile.TemporaryDirectory() as directory:
            args = runner.parser().parse_args(['cache', '--root', directory, '--score-root', directory + '/scores',
                '--shared-bank-root', directory + '/bank', '--device', 'cpu'])
            for split in ('train', 'val'):
                runner.write_json(args.shared_bank_root / 'prediction_cache' / split / 'TSP/R77B.npz', dict(synthetic=True))
            with patch.object(runner, 'PROBLEMS', ('TSP',)), \
                    patch.object(runner, 'SPLIT_ROWS', dict(train=5, val=2, test=2)), \
                    patch.object(runner, 'source_context', return_value=context), \
                    patch.object(runner, 'contract', return_value=dict(synthetic=True)), \
                    patch.object(runner, 'score_identity', side_effect=identity), \
                    patch.object(runner, 'open_cache', side_effect=AssertionError('Cache may not rebuild/open inputs')), \
                    patch.object(runner, 'load_expert', side_effect=AssertionError('Cache may not repeat frozen inference')), \
                    patch.object(runner, 'shared_bank_scores', side_effect=lambda a, c, p, s, rows, ref:
                        np.tile(np.arange(len(source.pool), dtype=np.float32), (len(rows), 1))):
                first = runner.cache(args)
                with patch.object(runner, 'load_expert', side_effect=AssertionError('Already cached')), \
                        patch.object(runner, 'open_cache', side_effect=AssertionError('No input rebuild/reopen')):
                    second = runner.cache(args)
            self.assertEqual(first, second)
            self.assertFalse(first['input_features_rebuilt'])
            self.assertFalse(first['test_read'])
            self.assertEqual(set(first['splits']), {'train', 'val'})

    def test_full_test_evaluation_is_one_live_composite_and_times_both_stages(self):
        source = FakeCache('TSP', split='test', size=3)
        reranker, _ = fixture(size=3)

        class Stage(nn.Module):
            def __init__(self):
                super().__init__()
                self.bias = nn.Parameter(torch.zeros(()))
                self.rows = 0

            def forward(self, inputs):
                if 'costs' in inputs or 'ind' in inputs:
                    raise AssertionError('No query labels in complete inference')
                self.rows += len(inputs['n'])
                return dict(logits=torch.arange(len(source.pool), dtype=torch.float32)[None].repeat(len(inputs['n']), 1))

        class Task:
            def __init__(self, *args, **kwargs):
                if kwargs.get('live_first_stage') is not True:
                    raise AssertionError('TEST must compute the full pipeline')
                self.cache, self.size = source, source.size
                self.rows = np.arange(source.size)
                self.release_source_indices = self.rows + 1000
                self.scores = None

            def batch(self, indices, device):
                return source.batch(indices, device)

            def close(self):
                pass

        baseline = dict(selection_split='train', sbs_solver=source.pool[0], sbs_slot=0,
            mean_cost_by_solver={name: float(source.raw['costs'][:, i].mean()) for i, name in enumerate(source.pool)})
        first_stage = Stage()
        composite = FrozenTop2Selector(reranker, first_stage)
        with tempfile.TemporaryDirectory() as directory:
            args = runner.parser().parse_args(['test', '--root', directory, '--device', 'cpu',
                                               '--eval-batch-size', '3', '--large-microbatch-size', '2'])
            context = dict(baselines=dict(TSP=baseline))
            def summarize(per):
                return dict(per_problem=per, macro=dict(macro_top1=per['TSP']['top1']))
            with patch.object(runner, 'PROBLEMS', ('TSP',)), \
                    patch.object(runner, 'SPLIT_ROWS', dict(test=3)), \
                    patch.object(runner, 'authorize_test'), patch.object(runner, 'MetaTask', Task), \
                    patch.object(runner, 'summarize', side_effect=summarize):
                result = runner.evaluate(composite, args, context, 'test', 'selected_best', 720, 'synthetic')
            self.assertEqual(first_stage.rows, 3)
            self.assertTrue(result['inference_cost']['full_stage0_plus_reranker'])
            self.assertGreater(result['inference_cost']['forward_seconds'], 0.)
            self.assertEqual(result['per_problem']['TSP']['inference_cost']['batch_cap'], 2)

    def test_independent_per_expert_archive_api_roundtrip_without_inference(self):
        from . import r123_prediction_cache as bank
        rows, source = np.array([3, 7], dtype=np.int64), np.array([1003, 2007], dtype=np.int64)
        qualified = np.array(['TSP:fixture:3', 'TSP:fixture:7'])
        scores = np.array([[.1, .9], [.8, .2]], dtype=np.float32)
        pool, ids = np.array(['a', 'b']), np.array([2, 4], dtype=np.int64)
        info = dict(complete=True, checkpoint_sha256=runner.EXPERT_SHA, meta_preparation_identity='synthetic',
            pool=pool.tolist(), pool_ids=ids.tolist(), source_indices_sha256=bank.array_hash(source),
            source_identity_sha256=bank.array_hash(qualified))
        context = dict(meta=dict(preparation_identity='synthetic'),
            manifest=dict(splits=dict(train=dict(TSP=dict(pool=pool.tolist(), pool_ids=ids.tolist())))))
        with tempfile.TemporaryDirectory() as directory:
            args = runner.parser().parse_args(['cache', '--shared-bank-root', directory, '--device', 'cpu'])
            path = bank.archive_path(args.shared_bank_root, 'train', 'TSP', 'R77B')
            path.parent.mkdir(parents=True)
            bank.publish(path, dict(logits=scores, row_indices=rows, source_indices=source, source_identity=qualified,
                pool=pool, pool_ids=ids, metadata_json=np.asarray(json.dumps(dict(info, logits_sha256=bank.array_hash(scores))))))
            with patch.object(bank, 'context', return_value={}), patch.object(bank, 'rows_for', return_value=rows), \
                    patch.object(bank, 'identity', return_value=(info, source, qualified)), \
                    patch.object(runner, 'load_expert', side_effect=AssertionError('No duplicated expert inference')):
                actual = runner.shared_bank_scores(args, context, 'TSP', 'train', rows, source)
            np.testing.assert_array_equal(actual, scores)

    def test_full18_val_selection_exact_ties_and_test_gate_before_features(self):
        def result(risk, top):
            return dict(macro=dict(macro_actual_regret_pct=risk, macro_top1=top))
        self.assertLess(runner.selection_key(result(.2, .1), 1440), runner.selection_key(result(.3, .9), 720))
        self.assertLess(runner.selection_key(result(.2, .9), 1440), runner.selection_key(result(.2, .8), 720))
        self.assertLess(runner.selection_key(result(.2, .9), 720), runner.selection_key(result(.2, .9), 1440))
        with tempfile.TemporaryDirectory() as directory:
            args = runner.parser().parse_args(['test', '--root', directory])
            with patch.object(runner, 'source_context', side_effect=AssertionError('TEST gate must precede inputs')):
                with self.assertRaises(PermissionError):
                    runner.test(args)

    def test_resume_uses_meta_stream_cursors_not_whole_train(self):
        order = runner.task_order(7200)
        counts = Counter(order[:720])
        cfg = dict(synthetic=True)
        state = dict(args=cfg, state_hash=runner.json_hash(cfg), successful_updates=720,
            history=[{}], pending_evaluation=None, updates_by_problem=dict(counts),
            stream_draws={p: counts[p] * 1024 for p in PROBLEMS})
        runner.validate_resume(state, cfg, order)
        state['stream_draws']['TSP'] += 1
        with self.assertRaises(ValueError):
            runner.validate_resume(state, cfg, order)

    def test_synthetic_complete_train_lock_and_resume_boundary(self):
        class Tracker:
            def log(self, *args, **kwargs):
                pass

            def finish(self):
                pass

        class Task:
            def __init__(self, *args):
                pass

            def batch(self, rows):
                return tiny_batch()

            def close(self):
                pass

        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            args = runner.parser().parse_args(['train', '--root', directory, '--prepared-root', directory + '/prepared',
                '--score-root', directory + '/scores', '--device', 'cpu', '--wandb-mode', 'offline'])
            args.prepared_root.mkdir()
            initial, geometry = fresh_fixture()
            torch.save(initial, args.prepared_root / 'identicalinit_seed2.pt')
            runner.write_json(args.score_root / 'manifest.json', dict(synthetic=True))
            cfg = dict(config(initial['model_params']), max_successful_updates=4, global_batch_size=5,
                logical_global_batch=5, warmup_updates=2, eval_every_updates=2,
                candidate_pools=dict(TSP=['a', 'b'], ATSP=['a', 'b']), data_root=str(args.data_root))
            context = dict(geometry=geometry, params=initial['model_params'])

            def load(path, device):
                saved = torch.load(path, map_location=device, weights_only=False)
                model = TinyPair()
                model.load_state_dict(saved['model'])
                return model, saved

            def evaluation(model, args, ctx, split, tag, step, *rest):
                return dict(macro=dict(macro_actual_regret_pct=.1 if step == 2 else .2, macro_top1=.75))

            with patch.object(runner, 'PROBLEMS', ('TSP', 'ATSP')), patch.object(runner, 'MAX_UPDATES', 4), \
                    patch.object(runner, 'BATCH_SIZE', 5), patch.object(runner, 'EVAL_EVERY', 2), \
                    patch.object(runner, 'WARMUP_UPDATES', 2), \
                    patch.object(runner, 'source_context', return_value=context), \
                    patch.object(runner, 'contract', return_value=cfg), \
                    patch.object(runner, 'initialize_fresh', return_value=(TinyPair(), dict(synthetic=True))), \
                    patch.object(runner, 'MetaTask', Task), patch.object(runner, 'evaluate', side_effect=evaluation), \
                    patch.object(runner, 'load_reranker', side_effect=load), \
                    patch.object(runner, 'wandb_run', return_value=Tracker()):
                result = runner.train(args)
                self.assertEqual(result['selected_update'], 2)
                lock, _ = runner.authorize_test(args)
                self.assertEqual(lock['selected_update'], 2)
                self.assertTrue(lock['selected_without_test'])
                self.assertFalse(lock['per_problem_assembly'])
                final = runner.read_json(base / 'model/training_complete.json')
                self.assertEqual(final['updates_by_problem'], dict(TSP=2, ATSP=2))
                self.assertEqual(final['successful_updates'], 4)
                self.assertFalse(final['test_read'])
                args.resume = True
                runner.train(args)

    def test_correction_damage_and_exact_regret_decomposition(self):
        data = dict(stage1_scores=np.array([[3., 2., 1.], [3., 2., 1.], [3., 2., 1.]], dtype=np.float32),
            logits=np.array([[4., 5., 1.], [4., 5., 1.], [5., 4., 1.]], dtype=np.float32),
            pool_ids=np.array([1, 4, 8]), winner=np.array([1, 0, 2]),
            costs=np.array([[1.1, 1., 1.3], [1., 1.2, 1.4], [1.3, 1.2, 1.]], dtype=np.float64))
        stats = runner.top2_statistics(data)
        self.assertEqual((stats['corrected'], stats['damaged']), (1, 1))
        self.assertEqual(stats['net_top1_change'], 0.)
        regret = np.mean([0., .2, .3]) * 100
        self.assertAlmostEqual(stats['recall_floor_regret_pct'] + stats['reranking_excess_regret_pct'], regret)


if __name__ == '__main__':
    unittest.main()
