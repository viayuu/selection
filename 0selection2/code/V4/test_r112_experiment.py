"""Bounded CPU contracts; synthetic artifacts stay in temporary directories."""

import io
import json
import copy
import csv
import subprocess
import sys
import tempfile
import unittest
from collections import Counter
from contextlib import contextmanager, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from . import r112_experiment as runner
from . import r112_reporting as reporting
from .pairwise_objective import comparison_targets
from .pairwise_selector import maximin_logits
from .r76_experiment import Runtime, save_checkpoint, write_json
from .r77_data import parameter_hash


class TinySelector(torch.nn.Module):
    def __init__(self, params, arm):
        super().__init__()
        self.params = dict(params, r112_arm=arm)
        self.weight = torch.nn.Parameter(torch.tensor([.2, -.1]))
        self.register_buffer('geometry', torch.tensor([.3, 1.2]))

    def forward(self, batch):
        utility = batch['node'][:, 0, :1] * self.weight[None]
        mask = torch.ones_like(utility, dtype=torch.bool)
        margins = utility[:, :, None] - utility[:, None, :]
        return dict(logits=maximin_logits(margins, mask), utility=utility,
                    pair_margin=margins, solver_mask=mask)


def tiny_factory(params, arm, state_dict=None):
    model = TinySelector(params, arm)
    if state_dict is not None:
        model.load_state_dict(state_dict, strict=True)
    return model


def fixture(root, arm=None):
    args = runner.parser().parse_args(['train', '--arm', arm or runner.ARMS[0], '--root', str(root)])
    args.data_root, args.prepared_root, args.cache_root = root / 'data', root / 'prepared', root / 'cache'
    write_json(args.prepared_root / 'prepared.json', {})
    record = dict(runner.required_lock_fields(), data_root=str(args.data_root),
        prepared_root=str(args.prepared_root), primary_cache=str(args.cache_root),
        microbatch_caps=dict.fromkeys(runner.PROBLEMS, min(2, runner.BATCH_SIZE)), eval_batch_size=2)
    write_json(root / 'protocol_lock.json', record)
    runner.protocol_lock(args)
    params = dict(dropout=.1, r112_arm=args.arm,
        r112_spec=dict(role_arm=runner.MODEL_ROLE_ARM, hierarchy=(4, 2, 1)))
    model = tiny_factory(params, args.arm)
    receipt = dict(trainable_sha256=parameter_hash(dict(model.named_parameters())),
        parameter_layout_sha256=runner.json_hash({'weight': [2]}), fresh_reference_sha256='synthetic_fresh')
    ready = dict(trainable_sha256='synthetic_fresh')
    manifest = dict(splits={split: {p: dict(pool=[runner.GLOBAL_SOLVERS[0], runner.GLOBAL_SOLVERS[1]],
        pool_ids=[0, 1], rows=count, complete=True) for p in runner.PROBLEMS}
        for split, count in runner.SPLIT_ROWS.items()})
    baselines = {p: dict(selection_split='train', sbs_solver=runner.GLOBAL_SOLVERS[0]) for p in runner.PROBLEMS}
    config = runner.protocol(args, ready, manifest, {}, baselines, params,
        runner.task_order(runner.MAX_UPDATES), record['microbatch_caps'], model=model, initialization=receipt)
    return args, config, model, receipt, ready, manifest, baselines


def archive(run, model, config, split, tag, step, checkpoint=None, *, top1=.5, regret=2.):
    directory = run / 'evaluation' / split / tag
    directory.mkdir(parents=True, exist_ok=True)
    identity = dict(experiment='R112', arm=config['arm'], split=split, tag=tag,
        successful_updates=step, state_hash=runner.json_hash(config),
        model_state_sha256=parameter_hash(model.state_dict()), checkpoint_sha256=checkpoint)
    write_json(directory / 'identity.json', identity)
    np.savez(directory / 'rank000.npz', synthetic=np.zeros(1))
    result = dict(identity=identity, prediction_archive=str(directory / 'rank000.npz'),
        per_problem={p: dict(n=runner.SPLIT_ROWS[split]) for p in runner.PROBLEMS},
        macro=dict(macro_top1=top1, macro_actual_regret_pct=regret, macro_vs_sbs_pct=100 - step), seconds=.001)
    write_json(directory / 'metrics.json', result)
    return result


class TinyObjective(torch.nn.Module):
    def __init__(self, fail=False):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(.2))
        self.fail, self.draws = fail, []

    def forward(self, batch, weight):
        self.draws.append(torch.rand_like(batch['x']))
        loss = (self.weight * batch['x']).square().mean()
        if self.fail:
            loss, self.fail = loss * float('nan'), False
        return loss, loss.detach().expand(8)


class R112RunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_describe_is_cpu_only_and_does_not_require_model_or_lock(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(torch, 'load', side_effect=AssertionError):
            for arm in runner.ARMS:
                with redirect_stdout(io.StringIO()):
                    result = runner.main(['describe', '--arm', arm, '--root', directory])
                self.assertEqual(result['caps_status'], 'awaiting_main_GPU_witness')
                self.assertEqual(result['required_protocol_lock']['logical_global_batch'], 512)

    def test_private_engine_schedule_caps_and_old_globals_are_isolated(self):
        with tempfile.TemporaryDirectory() as directory:
            args, config, *_ = fixture(Path(directory))
            local = runner.private_engine(args)
            self.assertIsNot(local, runner.engine)
            self.assertEqual((local.BATCH_SIZE, local.EVAL_EVERY, local.MAX_UPDATES), (512, 2880, 57600))
            self.assertEqual((runner.engine.BATCH_SIZE, runner.engine.EVAL_EVERY), (1024, 1440))
            self.assertIs(local.train.__globals__['selection_key'], runner.selection_key)
            self.assertEqual(local.learning_rate(4320), 1e-4)
            self.assertEqual(local.learning_rate(57600), 1e-6)
            self.assertEqual(Counter(runner.task_order(57600)), Counter(dict.fromkeys(runner.PROBLEMS, 3200)))
            self.assertEqual(config, json.loads(json.dumps(config)))
            with self.assertRaises(ValueError):
                local.micro_caps(large_cap=1)
            args.eval_batch_size = 3
            with self.assertRaises(ValueError):
                runner.protocol_lock(args)

    def test_exact_top1_then_actual_regret_not_vs_sbs_selection(self):
        def metric(top1, regret, sbs):
            return dict(macro=dict(macro_top1=top1, macro_actual_regret_pct=regret, macro_vs_sbs_pct=sbs))
        self.assertLess(runner.selection_key(metric(.6, 100, 100)), runner.selection_key(metric(.5, 0, -50)))
        self.assertLess(runner.selection_key(metric(.6, 1, 100)), runner.selection_key(metric(.6, 2, -50)))
        self.assertEqual(runner.selection_key(metric(.6, 1, 100)), runner.selection_key(metric(.6, 1, -50)))
        for bad in (float('nan'), float('inf'), 1.1):
            with self.assertRaises(ValueError):
                runner.selection_key(metric(bad, 1, 0))

    def test_microbatch_row_weights_and_nonfinite_replay(self):
        batch = dict(ind=torch.zeros(512, dtype=torch.long), x=torch.linspace(.1, 1, 512))
        outcomes, rngs = [], []
        for cap, fail in ((512, False), (73, False), (73, True)):
            objective = TinyObjective(fail)
            optimizer = torch.optim.SGD(objective.parameters(), lr=.1)
            torch.manual_seed(2)
            parts, retries, _ = runner.update_batch(objective, optimizer,
                torch.amp.GradScaler('cpu', enabled=False), batch, .2,
                Runtime(0, 1, torch.device('cpu')), cap, problem='TSP')
            self.assertTrue(torch.isfinite(parts).all())
            self.assertEqual(retries, int(fail))
            outcomes.append(objective.weight.detach())
            if cap == 73:
                rngs.append(torch.get_rng_state())
            if fail:
                for first, replay in zip(objective.draws[:8], objective.draws[8:]):
                    torch.testing.assert_close(first, replay, rtol=0, atol=0)
        for value in outcomes[1:]:
            torch.testing.assert_close(value, outcomes[0], rtol=0, atol=1e-7)
        torch.testing.assert_close(*rngs, rtol=0, atol=0)

    def test_checkpoint_factory_restores_buffers_and_rejects_protocol_changes(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(runner, 'make_model', tiny_factory):
            args, config, model, *_ = fixture(Path(directory))
            path = args.root / 'synthetic.pt'
            state = dict(args=config, model=model.state_dict(), state_hash=runner.json_hash(config),
                         successful_updates=2880)
            save_checkpoint(path, state)
            restored, saved = runner.load_checkpoint(path)
            self.assertEqual(parameter_hash(model.state_dict()), parameter_hash(restored.state_dict()))
            self.assertEqual(saved['args'], config)
            for field, value in (('checkpoint_rule', 'minimum TEST cost'), ('logical_global_batch', 1024),
                                 ('eval_batch_size', 4), ('loss', {}), ('model_factory', 'old.load_checkpoint')):
                with self.assertRaises(ValueError):
                    runner.validate_checkpoint_config(dict(config, **{field: value}))
            save_checkpoint(path, dict(state, objective_state={}))
            with self.assertRaises(ValueError):
                runner.load_checkpoint(path)
            broken = copy.deepcopy(config)
            broken['source_sha256']['r112_model.py'] = 'changed_source'
            save_checkpoint(path, dict(state, args=broken, state_hash=runner.json_hash(broken)))
            with self.assertRaisesRegex(ValueError, 'inference source changed'):
                runner.load_checkpoint(path)

    def test_real_model_initialization_receipt_and_identical_arm_parameters(self):
        from .test_r80 import fresh_fixture
        initial, geometry = fresh_fixture()
        receipts = []
        for arm in runner.ARMS:
            model, params, receipt = runner.initialize_shared(initial, geometry, arm)
            self.assertEqual(params['r112_arm'], arm)
            self.assertEqual(receipt['common_source_trainable_sha256'], initial['trainable_sha256'])
            self.assertNotEqual(receipt['common_parameter_sha256'], initial['trainable_sha256'])
            self.assertEqual(receipt['fresh_trainable_sha256'], receipt['trainable_sha256'])
            with tempfile.TemporaryDirectory() as directory:
                args, _, _, _, ready, manifest, baselines = fixture(Path(directory), arm)
                ready['trainable_sha256'] = initial['trainable_sha256']
                config = runner.protocol(args, ready, manifest, geometry, baselines, params,
                    runner.task_order(runner.MAX_UPDATES), runner.private_engine(args).micro_caps(),
                    model=model, initialization=receipt)
                self.assertNotIn('common_trainable_sha256', config)
                self.assertEqual(config['common_parameter_sha256'], receipt['common_parameter_sha256'])
                path = args.root / 'synthetic_real_model.pt'
                save_checkpoint(path, dict(args=config, model=model.state_dict(), state_hash=runner.json_hash(config)))
                restored, _ = runner.load_checkpoint(path)
                self.assertEqual(parameter_hash(model.state_dict()), parameter_hash(restored.state_dict()))
                del restored
            receipts.append(receipt)
            del model
        self.assertEqual(receipts[0]['trainable_sha256'], receipts[1]['trainable_sha256'])
        self.assertEqual(receipts[0]['trainable_parameter_count'], receipts[1]['trainable_parameter_count'])

    def test_full18_cpu_training_selection_resume_and_immutable_test_lock(self):
        rows = dict(train=8, val=2, test=2)
        with tempfile.TemporaryDirectory() as directory, patch.multiple(runner,
                BATCH_SIZE=4, MAX_UPDATES=54, WARMUP_UPDATES=4, EVAL_EVERY=18,
                TRAIN_EVERY=90, SPLIT_ROWS=rows), patch.object(runner, 'make_model', tiny_factory):
            args, config, model, receipt, ready, manifest, baselines = fixture(Path(directory))
            local = runner.private_engine(args)
            local.SPLIT_ROWS = rows
            run = args.root / args.arm
            run.mkdir()
            save_checkpoint(args.prepared_root / 'identicalinit_seed2.pt', {'synthetic': True})
            seen, evaluations, interrupted = [], [], []

            @contextmanager
            def prefetch(caches, requests, device):
                def batches():
                    for problem, indices, count in requests:
                        seen.append((problem, indices.tolist()))
                        costs = np.tile(np.array([[1., 2.]], dtype=np.float64), (count, 1))
                        batch = dict(ind=torch.zeros(count, dtype=torch.long),
                            node=(indices.float() + 1)[:, None, None] / 8,
                            pool_ids=torch.tensor([0, 1]), **comparison_targets(costs))
                        yield problem, count, count, batch
                yield batches()

            def evaluate(selector, arguments, cache_manifest, baseline, split, tag, step, state_hash, sha=None):
                if not interrupted:
                    interrupted.append(step)
                    raise RuntimeError('synthetic interruption before pending VAL')
                evaluations.append((split, step))
                # Top1 tie at36/54; regret chooses36 even though cost-ratio chooses54.
                return archive(run, selector, config, split, tag, step, sha,
                    top1=.4 if step == 18 else .6, regret=1. if step == 36 else 2.)

            cache = lambda *a: SimpleNamespace(size=8, close=lambda: None)
            with patch.object(runner, 'prepared_config', return_value=(ready, manifest, {}, baselines)), \
                    patch.object(runner, 'initialize_shared', return_value=(model, model.params, receipt)), \
                    patch.object(runner, 'evaluate', side_effect=evaluate), \
                    patch.object(local, 'open_cache', side_effect=cache), \
                    patch.object(local, 'PinnedPrefetch', prefetch), \
                    patch.object(local, 'wandb_run', return_value=None), \
                    patch.object(runner, 'test', return_value={'synthetic_test_delegated': True}) as test, \
                    redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(RuntimeError, 'synthetic interruption'):
                    runner.train(args, torch.device('cpu'))
                args.resume = True
                runner.train(args, torch.device('cpu'))
                test.assert_called_once()
            lock, saved_config = runner.authorize_test(args)
            self.assertEqual(lock['selected_update'], 36)
            self.assertEqual(len(seen), 54)
            self.assertEqual(Counter(p for p, _ in seen), Counter(dict.fromkeys(runner.PROBLEMS, 3)))
            streams = {p: local.ShuffleStream(8, local.task_seed(2, p)) for p in runner.PROBLEMS}
            self.assertEqual(seen, [(p, streams[p].take(4).tolist()) for p in runner.task_order(54)])
            self.assertNotIn('test', [s for s, _ in evaluations])
            final = runner.read_json(run / 'final_metrics.json')
            self.assertEqual(final['val']['identity']['checkpoint_sha256'], lock['checkpoint_sha256'])
            self.assertIsNone(final['val']['archive_capture_identity']['checkpoint_sha256'])
            capture_path = Path(final['val']['prediction_archive']).parent / 'identity.json'
            self.assertIsNone(runner.read_json(capture_path)['checkpoint_sha256'])
            saved = torch.load(run / 'last.pt', map_location='cpu', weights_only=False)
            local.validate_resume(saved, saved_config, runner.task_order(54))
            saved['stream_draws']['TSP'] += 1
            with self.assertRaises(ValueError):
                local.validate_resume(saved, saved_config, runner.task_order(54))
            write_json(run / 'training_complete.json', dict(successful_updates=53))
            with self.assertRaises(PermissionError):
                runner.authorize_test(args)

    def test_pair_initialization_mismatch_cannot_be_written(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            value = dict(fresh_reference_sha256='fresh', trainable_sha256='same', parameter_layout_sha256='shape')
            runner._engine_write_json(root / runner.ARMS[0] / 'initialization.json', value)
            with self.assertRaises(ValueError):
                runner._engine_write_json(root / runner.ARMS[1] / 'initialization.json', dict(value, trainable_sha256='different'))

    def test_no_implicit_test_cache_rebuild_or_unlocked_caps(self):
        with tempfile.TemporaryDirectory() as directory:
            args, config, _, _, _, manifest, _ = fixture(Path(directory))
            path = args.prepared_root / 'source_cache_manifest.json'
            write_json(path, manifest)
            lock = dict(checkpoint_sha256='synthetic')
            with patch.object(runner, 'authorize_test', return_value=(lock, config)):
                self.assertEqual(runner.ensure_test_cache(args, lock), manifest)
                del manifest['splits']['test']['TSP']
                write_json(path, manifest)
                before = runner.file_hash(path)
                with self.assertRaises(ValueError):
                    runner.ensure_test_cache(args, lock)
                self.assertEqual(runner.file_hash(path), before)
            record = runner.read_json(args.root / 'protocol_lock.json')
            record['microbatch_caps']['TSP'] = True
            with self.assertRaises(ValueError):
                runner.locked_caps(record)

    def test_nss_missing_is_four_explicit_rows_without_inference(self):
        with tempfile.TemporaryDirectory() as directory:
            _, config, *_ = fixture(Path(directory))
            data = SimpleNamespace(n=2, top1_pct=50., mean_cost=1.5, pool=['synthetic'], path=Path('synthetic.npz'))
            predictions = {(split, p): data for _, p, split in reporting.NSS_DATASETS}
            with patch.object(reporting.common, 'reporting_helpers', return_value=object()):
                rows, _ = reporting.nss_comparisons(config, dict(checkpoint_sha256='sha', state_hash='state',
                    selected_update=2880), predictions, Path(directory) / 'absent_nss')
            self.assertEqual(len(rows), 4)
            self.assertTrue(all(row['status'] == 'missing' and row['nss_mean_cost'] is None for row in rows))
            self.assertEqual(reporting._nss_display(None, 6), 'MISSING')
            self.assertEqual(reporting._nss_display(50., 2), '50.00')

    def test_runtime_imports_without_unpublished_r109_or_r111(self):
        program = '''
import importlib.abc
import sys

def unpublished(name):
    return name.rsplit('.', 1)[-1].startswith(('r109', 'r111', '_r109', '_r111', 'test_r109', 'test_r111'))

class BlockUnpublished(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if unpublished(fullname):
            raise ImportError('Unpublished dependency: ' + fullname)

sys.meta_path.insert(0, BlockUnpublished())
from code.V4 import r112_experiment as e, r112_reporting as r, test_r112_model
e._models()
r.common.reporting_helpers()
assert r.common.write_csv is r.write_csv
assert not any(unpublished(name) for name in sys.modules)
print('R112 runtime/report/model imports independent of R109/R111')
'''
        result = subprocess.run([sys.executable, '-B', '-c', program], cwd=runner.V4.parents[1],
            capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_local_timing_cache_opens_only_input_arrays(self):
        with tempfile.TemporaryDirectory() as directory:
            root, count = Path(directory), 2
            for problem, kind in (('TSP', 'coord'), ('ATSP', 'matrix')):
                config = dict(cache_root=str(root / 'cache'), data_root=str(root / 'data'), eval_batch_size=count,
                    candidate_pools={problem: ['synthetic0', 'synthetic1']}, candidate_pool_ids={problem: [0, 1]})
                record = dict(problem=problem, split='test', rows=runner.SPLIT_ROWS['test'],
                    pool=config['candidate_pools'][problem], pool_ids=[0, 1], data_root=config['data_root'],
                    kind=kind, problem_id=runner.PROBLEMS.index(problem), cbits=[0.] * 5,
                    problem_desc=[0.] * 12, mask=[1., 1.], d_in=8)
                write_json(root / 'cache' / 'test' / problem / 'metadata.json', record)
                arrays = dict(n=np.full(count, 4, dtype=np.int64), coord_dist=np.ones(count, dtype=np.float32),
                    node=np.zeros((count, 4, 8), dtype=np.float32), node_geom=np.zeros((count, 4, 12), dtype=np.float32),
                    matrix=np.zeros((count, 4, 4), dtype=np.float32))
                specs = dict.fromkeys((*arrays, 'costs', 'winner', 'cmp_target'), 'synthetic')
                with patch('code.V4.r76_cache._validate_record', return_value=specs), \
                        patch('code.V4.r75_cache._open_arrays', return_value=arrays) as opened, \
                        patch('code.V4.r75_cache._close') as closed:
                    batch = runner.cached_input_batch(config, dict(splits=dict(test={problem: record})),
                        'test', problem, torch.device('cpu'))
                    self.assertEqual(len(batch['node_mask']), count)
                    expected = {'n', 'coord_dist', 'node', 'node_geom'} if kind == 'coord' else {'n', 'coord_dist', 'matrix'}
                    self.assertEqual(set(opened.call_args.args[1]), expected)
                    closed.assert_called_once_with(arrays)
                    self.assertNotIn('costs', batch)
                    self.assertNotIn('ind', batch)

    def test_local_timing_strips_labels_and_checks_finite_predictions(self):
        seen = []

        def model(batch):
            seen.append(set(batch))
            return dict(logits=torch.zeros(2, 2), solver_mask=torch.ones(2, 2, dtype=torch.bool))

        batch = dict(node=torch.ones(2, 1, 1), node_mask=torch.ones(2, 1, dtype=torch.bool),
            ind=object(), costs=object(), winner=object(), cmp_target=object())
        result = runner.measure_inputs(model, batch, torch.device('cpu'))
        self.assertEqual(seen, [{'node', 'node_mask'}] * 4)
        self.assertEqual(len(result['batch_seconds']), 3)
        self.assertAlmostEqual(result['mean_seconds_per_instance'], sum(result['batch_seconds']) / 6)
        with self.assertRaisesRegex(ValueError, 'Nonfinite R112 timing prediction'):
            runner.measure_inputs(lambda b: dict(logits=torch.full((2, 2), float('nan')),
                solver_mask=torch.ones(2, 2, dtype=torch.bool)), batch, torch.device('cpu'))

    def test_local_report_csv_is_atomic_bom_free_and_keeps_missing_status(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / 'nss.csv'
            rows = [dict(status='missing', nss_mean_cost=None), dict(status='complete', nss_mean_cost=1.5)]
            reporting.write_csv(path, rows)
            self.assertFalse(path.read_bytes().startswith(b'\xef\xbb\xbf'))
            saved = list(csv.DictReader(io.StringIO(path.read_text(encoding='utf-8'))))
            self.assertEqual([r['status'] for r in saved], ['missing', 'complete'])
            self.assertEqual(saved[0]['nss_mean_cost'], '')
            self.assertEqual(list(root.glob('.*.tmp')), [])

    def test_finalizer_calls_only_frozen_stages_in_order(self):
        calls = []
        with patch.object(runner, 'test', side_effect=lambda *a: calls.append('test')), \
                patch.object(runner, 'timing', side_effect=lambda *a: calls.append('timing')), \
                patch.object(reporting, 'lib_stage', side_effect=lambda *a: calls.append('lib')), \
                patch.object(reporting, 'report', side_effect=lambda *a: calls.append('report')):
            runner.finalize(None, torch.device('cpu'))
        self.assertEqual(calls, ['test', 'timing', 'lib', 'report'])


if __name__ == '__main__':
    unittest.main()
