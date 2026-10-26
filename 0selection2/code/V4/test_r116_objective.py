"""Small synthetic CPU contracts only; no data preparation, GPU, or experiment launches."""

import copy
from contextlib import contextmanager, redirect_stdout
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from . import r116_experiment as runner
from . import r116_objective as objective
from .r76_experiment import Runtime, read_json, save_checkpoint, write_json
from .r77_data import parameter_hash
from .r116_subset import champion_log_probs, full_ranking_nll, stable_cost_order


class TinySelector(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([[.2, -.4, .1], [.7, .5, -.3],
                                                       [.3, -.2, .9], [.4, .6, -.1]]))
        self.calls = 0

    def forward(self, batch):
        if set(batch) - objective.INPUT_KEYS:
            raise AssertionError('Query supervision reached the selector')
        self.calls += 1
        logits = self.weight[None] * batch['node'][:, :1, :1]
        return dict(logits=champion_log_probs(logits), subset_logits=logits,
                    solver_mask=torch.ones(len(logits), 3, dtype=torch.bool))


def batch(count=7):
    x = torch.linspace(.2, 1.1, count).reshape(count, 1, 1)
    costs = torch.tensor([[1. + 1e-10, 1., 1.2]], dtype=torch.float64).expand(count, -1).clone()
    return dict(node=x, node_mask=torch.ones(count, 1, dtype=torch.bool), pool_ids=torch.tensor([2, 5, 9]),
                costs_fp64=costs, costs=costs.float(), ind=torch.ones(count, dtype=torch.long),
                winner='must not enter query', target_order='must not enter query', cmp_target=object())


def fixture(root):
    args = runner.parser().parse_args(['train', '--root', str(root), '--arm', runner.ARMS[0]])
    args.data_root, args.prepared_root, args.cache_root = root / 'data', root / 'prepared', root / 'cache'
    args.prepared_root.mkdir()
    write_json(args.prepared_root / 'prepared.json', {})
    record = dict(runner.required_lock_fields(), data_root=str(args.data_root), prepared_root=str(args.prepared_root),
                  primary_cache=str(args.cache_root), microbatch_caps=dict.fromkeys(runner.PROBLEMS, min(256, runner.BATCH_SIZE)),
                  eval_batch_size=min(256, runner.SPLIT_ROWS['val']))
    write_json(root / 'protocol_lock.json', record)
    manifest = dict(splits={s: {p: dict(pool=runner.GLOBAL_SOLVERS[:3], pool_ids=[0, 1, 2],
        rows=n, complete=True) for p in runner.PROBLEMS} for s, n in runner.SPLIT_ROWS.items()})
    params = dict(dropout=.1, role_arm=runner.MODEL_ROLE_ARM, r116_arm=args.arm,
                  r116_spec=copy.deepcopy(runner._models().SPEC))
    model = TinySelector()
    model.params = params
    receipt = dict(trainable_sha256='synthetic', parameter_layout_sha256='synthetic')
    baselines = {p: dict(selection_split='train', sbs_solver=runner.GLOBAL_SOLVERS[0]) for p in runner.PROBLEMS}
    config = runner.protocol(args, dict(trainable_sha256='fresh'), manifest, {}, baselines, params,
        runner.task_order(runner.MAX_UPDATES), record['microbatch_caps'], model=model, initialization=receipt)
    return args, record, config


class R116Contracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_single_forward_actual_parts_and_fp64_ranking(self):
        model, data = TinySelector(), batch()
        loss, parts = objective.EliminationObjective(model)(data)
        self.assertEqual(model.calls, 1)
        self.assertEqual(objective.PART_KEYS, ('loss', 'champion_nll', 'ranking_nll', 'grad_norm'))
        order = stable_cost_order(data['costs_fp64'], data['ind'], data['pool_ids'])
        self.assertEqual(order[0].tolist(), [1, 0, 2])
        self.assertEqual(data['costs'].argmin(1)[0].item(), 0)
        output = model(objective.inputs_only(data))
        ce = torch.nn.functional.cross_entropy(output['logits'], data['ind'])
        ranking = full_ranking_nll(output['subset_logits'], order).mean()
        torch.testing.assert_close(parts, torch.stack([.5 * (ce + ranking), ce, ranking]))
        loss.backward()
        self.assertTrue(torch.isfinite(model.weight.grad).all())
        self.assertGreater(model.weight.grad.abs().sum().item(), 0)
        with self.assertRaises(ValueError):
            objective.per_instance_losses(output, dict(data, costs_fp64=data['costs']))

    def test_all_targets_removed_before_query(self):
        clean = objective.inputs_only(dict(batch(), arbitrary_future_label=object(), costs_fp64=object()))
        self.assertEqual(set(clean), {'node', 'node_mask', 'pool_ids'})

    def test_row_weighted_microbatch_matches_whole_logical_batch(self):
        first, second, data = TinySelector(), TinySelector(), batch(1024)
        opt1, opt2 = torch.optim.SGD(first.parameters(), lr=.03), torch.optim.SGD(second.parameters(), lr=.03)
        value, expected = objective.EliminationObjective(first)(data)
        value.backward()
        norm = torch.nn.utils.clip_grad_norm_(first.parameters(), 1.)
        opt1.step()
        scaler = torch.amp.GradScaler('cuda', enabled=False)
        parts, retries, extra = runner.update_batch(objective.EliminationObjective(second), opt2, scaler,
            data, 0., Runtime(0, 1, torch.device('cpu')), 333, problem='TSP')
        torch.testing.assert_close(second.weight, first.weight, atol=1e-7, rtol=1e-6)
        torch.testing.assert_close(parts, torch.cat([expected.double(), norm.reshape(1).double()]), atol=1e-6, rtol=1e-6)
        self.assertEqual((retries, extra), (0, {}))
        self.assertEqual(second.calls, 4)

    def test_cache_gathers_original_fp64_without_touching_raw_or_reordering(self):
        cache = objective.FP64Cache.__new__(objective.FP64Cache)
        raw = np.array([[1. + 1e-10, 1.], [4., 3.], [5., 6.]], dtype=np.float64)
        raw.flags.writeable = False
        cache.raw, cache.size = dict(costs=raw), 3
        with patch.object(objective.ScenarioCache, 'batch', return_value={'ind': torch.tensor([1, 0])}):
            gathered = cache.batch(torch.tensor([1, 0]))
        self.assertEqual(gathered['costs_fp64'].dtype, torch.float64)
        np.testing.assert_array_equal(gathered['costs_fp64'].numpy(), raw[[1, 0]])
        self.assertFalse(raw.flags.writeable)

    def test_protocol_private_engine_and_unique_val_key(self):
        with tempfile.TemporaryDirectory() as directory:
            args, record, config = fixture(Path(directory))
            local = runner.private_engine(args)
            self.assertIsNot(local, runner.engine)
            self.assertEqual(local.PART_KEYS, objective.PART_KEYS)
            self.assertEqual((local.BATCH_SIZE, local.MAX_UPDATES, local.EVAL_EVERY), (1024, 28440, 1422))
            self.assertEqual(runner.engine.EVAL_EVERY, 1440)
            self.assertEqual(local.learning_rate(4266), 1e-4)
            self.assertEqual(local.learning_rate(28440), 1e-6)
            self.assertEqual(config, json.loads(json.dumps(config)))
            self.assertNotIn('rdrop_weight', config)
            self.assertNotIn('rdrop_ramp_updates', config)
            self.assertFalse(config['objective']['rdrop'])
            self.assertEqual(config['model_factory'], runner.MODEL_FACTORY)
            self.assertIs(local.train.__globals__['selection_key'], runner.selection_key)
            self.assertNotIn('consistency_weight=weight', local.train.__code__.co_names)
            with self.assertRaises(ValueError):
                local.micro_caps(large_cap=128)
            with self.assertRaises(ValueError):
                runner.validate_checkpoint_config(dict(config, loss=runner.engine.LOSS_RECIPE))
            args.eval_batch_size = 128
            with self.assertRaises(ValueError):
                runner.protocol_lock(args)
        def val(top1, regret):
            return dict(macro=dict(macro_top1=top1, macro_actual_regret_pct=regret))
        self.assertLess(runner.selection_key(val(.51, 20)), runner.selection_key(val(.50, 0)))
        self.assertLess(runner.selection_key(val(.51, 1)), runner.selection_key(val(.51, 2)))
        self.assertEqual(runner.selection_key(val(.51, 1)), runner.selection_key(val(.51, 1)))

    def test_reporting_and_finish_imports_have_correct_factories(self):
        from . import r116_reporting as reporting
        from . import r116_finish as finish
        self.assertEqual(reporting._callbacks.FACTORY, 'code.V4.r116_experiment')
        self.assertGreaterEqual(reporting.REPETITIONS, 10)
        self.assertIs(reporting.shared.learning_curves, reporting.learning_curves)
        self.assertEqual(finish.ARMS, runner.ARMS)
        self.assertIs(finish._callbacks.completed, finish.completed)

    def test_budget_truncation_never_publishes_or_claims_negative_evidence(self):
        from . import r116_finish as finish
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / 'preparation/resources/training/A_static/worker.json'
            write_json(path, dict(state='incomplete_budget', exit_status=124))
            with patch.object(finish, 'publish', side_effect=AssertionError('No partial publication')):
                value = finish.wait(root, publish_after=True)
            self.assertFalse(value['complete'])
            self.assertEqual(value['state'], 'incomplete_budget')
            self.assertIn('not negative', value['interpretation'])

    def test_private_train_resume_val_selection_and_lock_with_tiny_cpu_fixtures(self):
        rows = dict(train=8, val=2, test=2)
        real_spec = copy.deepcopy(runner._models().SPEC)

        def factory(params, arm, state_dict=None):
            model = TinySelector()
            model.params = params
            if state_dict is not None:
                model.load_state_dict(state_dict, strict=True)
            return model

        with tempfile.TemporaryDirectory() as directory, patch.multiple(runner,
                BATCH_SIZE=4, MAX_UPDATES=54, WARMUP_UPDATES=4, EVAL_EVERY=18, TRAIN_EVERY=90, SPLIT_ROWS=rows), \
                patch.object(runner, '_models', return_value=SimpleNamespace(SPEC=real_spec, make_model=factory)):
            args, record, config = fixture(Path(directory))
            local, model = runner.private_engine(args), factory(config['model_params'], args.arm)
            run = args.root / args.arm
            run.mkdir()
            receipt = dict(trainable_sha256='synthetic', parameter_layout_sha256='synthetic')
            manifest = dict(splits={s: {p: dict(pool=runner.GLOBAL_SOLVERS[:3], pool_ids=[0, 1, 2], rows=n,
                complete=True) for p in runner.PROBLEMS} for s, n in rows.items()})
            baselines = {p: dict(selection_split='train', sbs_solver=runner.GLOBAL_SOLVERS[0]) for p in runner.PROBLEMS}
            save_checkpoint(args.prepared_root / 'identicalinit_seed2.pt', {})
            evaluated, interrupted = [], []

            @contextmanager
            def prefetch(caches, requests, device):
                yield ((p, len(indices), count, batch(len(indices))) for p, indices, count in requests)

            def evaluate(selector, arguments, cache_manifest, baseline, split, tag, step, state_hash, sha=None):
                if not interrupted:
                    interrupted.append(True)
                    raise RuntimeError('synthetic pending VAL interruption')
                evaluated.append(split)
                path = run / 'evaluation' / split / tag
                path.mkdir(parents=True, exist_ok=True)
                identity = dict(experiment='R116', arm=args.arm, split=split, tag=tag, successful_updates=step,
                    state_hash=state_hash, model_state_sha256=parameter_hash(selector.state_dict()), checkpoint_sha256=sha,
                    winner_rule=runner.WINNER_RULE)
                write_json(path / 'identity.json', identity)
                np.savez(path / 'rank000.npz', synthetic=np.zeros(1))
                result = dict(identity=identity, prediction_archive=str(path / 'rank000.npz'), seconds=.001,
                    per_problem={p: dict(n=rows[split]) for p in runner.PROBLEMS},
                    macro=dict(macro_top1=.4 if step == 18 else .6,
                               macro_actual_regret_pct=1. if step == 36 else 2., macro_vs_sbs_pct=-step))
                write_json(path / 'metrics.json', result)
                return result

            with patch.object(runner, 'prepared_config', return_value=(dict(trainable_sha256='fresh'), manifest, {}, baselines)), \
                    patch.object(runner, 'initialize_shared', return_value=(model, model.params, receipt)), \
                    patch.object(runner, 'evaluate', side_effect=evaluate), \
                    patch.object(local, 'open_cache', return_value=SimpleNamespace(size=8, close=lambda: None)), \
                    patch.object(local, 'PinnedPrefetch', prefetch), patch.object(local, 'wandb_run', return_value=None), \
                    patch.object(runner, 'test', return_value={}) as test, redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(RuntimeError, 'pending VAL'):
                    runner.train(args, torch.device('cpu'))
                args.resume = True
                runner.train(args, torch.device('cpu'))
                test.assert_called_once()
            lock, saved = runner.authorize_test(args)
            self.assertEqual(lock['selected_update'], 36)
            self.assertNotIn('test', evaluated)
            self.assertEqual(saved['model_factory'], runner.MODEL_FACTORY)
            self.assertEqual(read_json(run / 'final_metrics.json')['val']['identity']['checkpoint_sha256'], lock['checkpoint_sha256'])


if __name__ == '__main__':
    unittest.main()
