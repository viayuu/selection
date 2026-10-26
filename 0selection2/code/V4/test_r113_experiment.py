"""Bounded CPU contracts; synthetic data/updates stay in temporary directories."""

import copy
import io
import json
import shutil
import subprocess
import tempfile
import unittest
from collections import Counter
from contextlib import contextmanager, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from . import r113_experiment as runner
from . import r113_pretrain as process
from . import r113_process_data as data
from . import r113_reporting as reporting
from .pairwise_objective import comparison_targets, inputs_only
from .pairwise_selector import maximin_logits
from .r76_experiment import Runtime, save_checkpoint, write_json
from .r77_data import parameter_hash
from .test_r80 import fresh_fixture
from .test_r113_process import raw_routing


class TinySelector(torch.nn.Module):
    def __init__(self, params, arm):
        super().__init__()
        self.params = dict(params, r113_arm=arm)
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


def fixture(root, arm='A_scratch'):
    args = runner.parser().parse_args(['train', '--arm', arm, '--root', str(root)])
    args.data_root, args.prepared_root, args.cache_root = root / 'data', root / 'prepared', root / 'cache'
    args.pretrain_root = root / 'pretrain'
    args.nss_root = root / 'missing_synthetic_nss'
    write_json(args.prepared_root / 'prepared.json', {})
    record = dict(runner.required_lock_fields(), data_root=str(args.data_root),
        prepared_root=str(args.prepared_root), primary_cache=str(args.cache_root),
        pretrain_root=str(args.pretrain_root), process_manifest=str(root / 'process/manifest.json'),
        process_manifest_sha256='a' * 64, microbatch_caps=dict.fromkeys(runner.PROBLEMS, 2),
        pretrain_microbatch_caps=dict.fromkeys(runner.PROBLEMS, 2), eval_batch_size=2)
    write_json(root / 'protocol_lock.json', record)
    runner.protocol_lock(args)
    params = dict(dropout=.1, r113_arm=arm, r113_spec=copy.deepcopy(runner.models.SPEC))
    model = tiny_factory(params, arm)
    h = parameter_hash(dict(model.named_parameters()))
    receipt = dict(trainable_sha256=h, fresh_trainable_sha256=h,
        fresh_state_sha256=parameter_hash(model.state_dict()), fresh_nonprocessor_sha256='same',
        initial_processor_sha256='same_processor', fresh_reference_sha256='synthetic_fresh',
        parameter_layout_sha256=runner.json_hash(process.tensor_schema(model.state_dict())))
    if arm in process.ARMS:
        receipt['processor_pretraining'] = dict(successful_updates=process.MAX_UPDATES,
            mode=process.MODE[arm], dataset=dict(manifest_sha256='a' * 64))
    ready = dict(trainable_sha256='synthetic_fresh')
    manifest = dict(splits={split: {p: dict(pool=list(runner.GLOBAL_SOLVERS[:2]),
        pool_ids=[0, 1], rows=count, complete=True) for p in runner.PROBLEMS}
        for split, count in runner.SPLIT_ROWS.items()})
    baselines = {p: dict(selection_split='train', sbs_solver=runner.GLOBAL_SOLVERS[0]) for p in runner.PROBLEMS}
    config = runner.protocol(args, ready, manifest, {}, baselines, params,
        runner.task_order(runner.MAX_UPDATES), record['microbatch_caps'], model=model, initialization=receipt)
    return args, config, model, receipt, ready, manifest, baselines


def process_fixture(root, count=8):
    """A tiny complete18 fixture, not a formal process dataset."""
    source_root, target = root / 'raw', root / 'process'
    shards, identities = [], []
    for problem in data.PROBLEMS:
        metadata = source_root / (problem + 'train') / 'input_metadata.npz'
        metadata.parent.mkdir(parents=True)
        np.savez(metadata, source_index=np.arange(count, dtype=np.int64) + 1000)
        identities.append(dict(problem=problem, rows=count, split='train', scenario_id=data.SCENARIO,
            data_root=str(source_root), source_ids_sha256=runner.file_hash(metadata)))
    write_json(target / 'schema.json', data.schema())
    write_json(target / 'source_manifest.json', dict(sources=identities, seed=2, split='train',
        scenario_id=data.SCENARIO, source_policy='synthetic TRAIN raw inputs only'))
    source_hash = runner.file_hash(target / 'source_manifest.json')
    for problem in data.PROBLEMS:
        if problem == 'ATSP':
            raw = dict(kind='matrix', matrix=torch.ones(1, 5, 5) - torch.eye(5)[None], n=torch.tensor([5]))
        elif problem == 'TSP':
            raw = dict(kind='coord', node=torch.arange(10).float().reshape(1, 5, 2) / 10., n=torch.tensor([5]))
        else:
            raw = raw_routing(problem, [.1] * 4)
        compiled = data.compile_inputs(raw, problem)
        targets = data.execute_program(compiled)
        for start in range(0, count, 4):
            stop = min(start + 4, count)
            rows = np.arange(start, stop, dtype=np.int64)
            arrays = {**{'input__' + k: np.repeat(v.numpy(), len(rows), 0) for k, v in compiled.items()},
                **{'target__' + k: np.repeat(v.numpy(), len(rows), 0) for k, v in targets.items()},
                'row_index': rows, 'source_index': rows + 1000,
                'input_sha256': np.full(len(rows), b'b' * 64, dtype='S64'), 'version': np.array(data.VERSION),
                'split': np.array('train'), 'seed': np.array(2), 'source_manifest_sha256': np.array(source_hash)}
            name = f'shards/{problem}_{start}_{stop}.npz'
            path = target / name
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(path, **arrays)
            shards.append(dict(path=name, problem=problem, rows=stop-start, start_row=start, stop_row=stop,
                split='train', seed=2, source_manifest_sha256=source_hash,
                sha256=runner.file_hash(path), bytes=path.stat().st_size))
    manifest = dict(status='complete', split='train', seed=2, version=data.VERSION, schema=data.schema(),
        spec=dict(max_steps=12, neighbors=2), sources=identities, shards=shards,
        rows=18 * count, total_expected_rows=18 * count, native_solver_calls=0,
        source_manifest_sha256=source_hash, code_sha256={'r113_process_data.py': runner.file_hash(runner.V4 / 'r113_process_data.py')})
    write_json(target / 'manifest.json', manifest)
    write_json(target / 'completion_receipt.json', dict(status='complete', manifest_sha256=runner.file_hash(target / 'manifest.json'),
        source_manifest_sha256=source_hash, schema_sha256=runner.file_hash(target / 'schema.json'),
        per_problem_rows=dict.fromkeys(data.PROBLEMS, count), shards=len(shards)))
    return target / 'manifest.json', source_root, manifest


def export_fixture(model, receipt, args, *, mutate=True):
    state = {k: v.detach().cpu().clone() for k, v in model.instance_encoder.processor.state_dict().items()}
    if mutate:
        state[next(iter(state))].add_(.01)
    config = dict(experiment='R113_pretrain', arm=args.arm, seed=2, mode=process.MODE[args.arm],
        **process.required_fields(), initial_full_state_sha256=receipt['fresh_state_sha256'],
        initial_processor_sha256=receipt['initial_processor_sha256'],
        fresh_reference_sha256=receipt['fresh_reference_sha256'],
        processor_parameter_schema=process.tensor_schema(state), r113_spec=receipt['r113_spec'],
        dataset=dict(manifest_sha256='a' * 64), main_protocol_lock=runner.protocol_lock(args),
        source_sha256={name: runner.file_hash(runner.V4 / name) for name in process.SOURCE_FILES})
    return dict(format='r113_processor_only_v1', complete=True, args=config,
        state_hash=runner.json_hash(config), processor=state, processor_sha256=parameter_hash(state),
        successful_updates=process.MAX_UPDATES,
        updates_by_problem=dict.fromkeys(data.PROBLEMS, process.MAX_UPDATES // 18),
        stream_draws=dict.fromkeys(data.PROBLEMS, process.MAX_UPDATES // 18 * process.BATCH_SIZE))


def archive(run, model, config, split, tag, step, checkpoint=None, *, top1=.5, regret=2.):
    directory = run / 'evaluation' / split / tag
    directory.mkdir(parents=True, exist_ok=True)
    identity = dict(experiment='R113', arm=config['arm'], split=split, tag=tag,
        successful_updates=step, state_hash=runner.json_hash(config),
        model_state_sha256=parameter_hash(model.state_dict()), checkpoint_sha256=checkpoint)
    write_json(directory / 'identity.json', identity)
    np.savez(directory / 'rank000.npz', synthetic=np.zeros(1))
    value = dict(identity=identity, prediction_archive=str(directory / 'rank000.npz'),
        per_problem={p: dict(n=runner.SPLIT_ROWS[split]) for p in runner.PROBLEMS},
        macro=dict(macro_top1=top1, macro_actual_regret_pct=regret, macro_vs_sbs_pct=100-step), seconds=.001)
    write_json(directory / 'metrics.json', value)
    return value


class R113Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def setUp(self):
        self.guard = patch('torch.cuda._lazy_init', side_effect=AssertionError('CPU fixtures only'))
        self.guard.start()
        self.addCleanup(self.guard.stop)

    def test_describe_no_lock_no_weights_no_cuda(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(torch, 'load', side_effect=AssertionError):
            for arm in runner.ARMS:
                with redirect_stdout(io.StringIO()):
                    result = runner.main(['describe', '--arm', arm, '--root', directory])
                self.assertEqual(result['caps_status'], 'awaiting_main_bindings')
                self.assertEqual(result['required_protocol_lock']['pretraining']['max_successful_updates'], 8640)
                self.assertFalse(result['training_started'])

    def test_fixed_schedule_caps_and_historical_globals_unchanged(self):
        from . import r112_experiment
        with tempfile.TemporaryDirectory() as directory:
            args, config, *_ = fixture(Path(directory))
            local = runner.private_engine(args)
            self.assertIsNot(local, runner.engine)
            self.assertEqual((local.BATCH_SIZE, local.MAX_UPDATES, local.EVAL_EVERY), (512, 57600, 2880))
            self.assertEqual((runner.engine.BATCH_SIZE, runner.engine.EVAL_EVERY), (1024, 1440))
            self.assertEqual((r112_experiment.ARMS, r112_experiment.MODEL_FACTORY),
                (('A_collapsed', 'B_interfaces'), 'code.V4.r112_experiment.load_checkpoint'))
            self.assertEqual(local.learning_rate(4320), 1e-4)
            self.assertEqual(local.learning_rate(57600), 1e-6)
            self.assertEqual(process.learning_rate(432), 3e-4)
            self.assertEqual(process.learning_rate(8640), 1e-5)
            self.assertEqual(Counter(runner.task_order(8640)), Counter(dict.fromkeys(data.PROBLEMS, 480)))
            self.assertEqual(config, json.loads(json.dumps(config)))
            record = runner.read_json(args.root / 'protocol_lock.json')
            record['pretrain_microbatch_caps']['TSP'] = True
            with self.assertRaises(ValueError):
                runner.locked_caps(record)
            with self.assertRaises(ValueError):
                local.micro_caps(large_cap=1)

    def test_top1_then_actual_regret_not_mean_cost_ratio(self):
        def metric(top1, regret, sbs):
            return dict(macro=dict(macro_top1=top1, macro_actual_regret_pct=regret, macro_vs_sbs_pct=sbs))
        self.assertLess(runner.selection_key(metric(.6, 10, 10)), runner.selection_key(metric(.5, 0, -100)))
        self.assertLess(runner.selection_key(metric(.6, 1, 10)), runner.selection_key(metric(.6, 2, -100)))
        self.assertEqual(runner.selection_key(metric(.6, 1, 10)), runner.selection_key(metric(.6, 1, -100)))
        with self.assertRaises(ValueError):
            runner.selection_key(metric(float('nan'), 0, 0))

    def test_manifest_complete18_ids_sequences_and_identical_resumed_stream(self):
        with tempfile.TemporaryDirectory() as directory:
            path, source_root, manifest = process_fixture(Path(directory))
            dataset = process.ProcessDataset(path, source_root, rows_per_problem=8, cache_shards=2)
            for problem in data.PROBLEMS:
                first, second = process.ProcessStream(dataset, problem), process.ProcessStream(dataset, problem)
                rows = first.take(19)
                np.testing.assert_array_equal(rows, second.take(19))
                np.testing.assert_array_equal(np.sort(rows[:8]), np.arange(8))
                np.testing.assert_array_equal(first.take(11), process.ProcessStream(dataset, problem, 19).take(11))
                p, y, ids = dataset.batch(problem, rows)
                self.assertEqual(set(p), set(data.INPUT_KEYS))
                self.assertEqual(set(y), set(data.TARGET_KEYS))
                np.testing.assert_array_equal(ids['row_index'], rows)
                np.testing.assert_array_equal(ids['source_index'], rows + 1000)
                self.assertEqual(p['sequence'].shape, (19, 12))
                self.assertLessEqual(len(dataset.cache), 2)
            with self.assertRaisesRegex(ValueError, 'complete'):
                process.ProcessDataset(path, source_root)  # tiny cannot masquerade as81000
            manifest['status'] = 'building'
            write_json(path, manifest)
            with self.assertRaisesRegex(ValueError, 'complete'):
                process.ProcessDataset(path, source_root, rows_per_problem=8)

    def test_manifest_omitted_task_and_changed_shard_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path, source_root, manifest = process_fixture(Path(directory))
            dataset = process.ProcessDataset(path, source_root, rows_per_problem=8)
            shard = path.parent / manifest['shards'][0]['path']
            original = shard.read_bytes()
            # Equal-size corruption must still fail before loading any target.
            with shard.open('r+b') as stream:
                stream.seek(len(original) - 1)
                stream.write(bytes([original[-1] ^ 1]))
            with self.assertRaisesRegex(ValueError, 'hash'):
                dataset.batch('TSP', [0])
            manifest['shards'] = [s for s in manifest['shards'] if s['problem'] != 'ATSP']
            write_json(path, manifest)
            complete = runner.read_json(path.parent / 'completion_receipt.json')
            complete.update(manifest_sha256=runner.file_hash(path), shards=len(manifest['shards']))
            write_json(path.parent / 'completion_receipt.json', complete)
            with self.assertRaisesRegex(ValueError, 'Missing'):
                process.ProcessDataset(path, source_root, rows_per_problem=8)

    def test_all_three_complete_initializations_match_and_processor_is_extracted(self):
        initial, geometry = fresh_fixture()
        receipts = []
        for arm in runner.ARMS:
            model, params, receipt = runner.initialize_shared(initial, geometry, arm)
            self.assertEqual(params['r113_arm'], arm)
            self.assertTrue(hasattr(model.instance_encoder, 'node_projection'))
            self.assertTrue(hasattr(model.instance_encoder, 'edge_projection'))
            self.assertTrue(hasattr(model.instance_encoder, 'state_types'))
            if arm in process.ARMS:
                extracted, extraction = process.extract_processor(initial, geometry, arm)
                self.assertEqual(parameter_hash(extracted.state_dict()),
                    parameter_hash(model.instance_encoder.processor.state_dict()))
                self.assertEqual(extraction['fresh_state_sha256'], receipt['fresh_state_sha256'])
            receipts.append(receipt)
        for field in ('fresh_state_sha256', 'fresh_trainable_sha256', 'initial_processor_sha256',
                      'fresh_nonprocessor_sha256', 'parameter_layout_sha256', 'trainable_parameter_count'):
            self.assertEqual(len({r[field] for r in receipts}), 1, field)

    def test_processor_only_import_and_complete_checkpoint_without_external_export(self):
        initial, geometry = fresh_fixture()
        for arm in runner.ARMS:
            with self.subTest(arm=arm), tempfile.TemporaryDirectory() as directory:
                args, _, _, _, ready, manifest, baselines = fixture(Path(directory), arm)
                model, params, receipt = runner.initialize_shared(initial, geometry, arm)
                before = runner.nonprocessor_hash(model)
                if arm in process.ARMS:
                    path = args.pretrain_root / arm / 'processor.pt'
                    path.parent.mkdir(parents=True)
                    save_checkpoint(path, export_fixture(model, receipt, args))
                receipt = runner.apply_pretraining(model, receipt, args)
                self.assertEqual(before, runner.nonprocessor_hash(model))
                ready['trainable_sha256'] = initial['trainable_sha256']
                config = runner.protocol(args, ready, manifest, geometry, baselines, params,
                    runner.task_order(runner.MAX_UPDATES), runner.private_engine(args).micro_caps(),
                    model=model, initialization=receipt)
                checkpoint = args.root / 'synthetic_full.pt'
                save_checkpoint(checkpoint, dict(args=config, state_hash=runner.json_hash(config), model=model.state_dict()))
                if arm in process.ARMS:
                    path.unlink()
                restored, _ = runner.load_checkpoint(checkpoint)
                self.assertEqual(parameter_hash(restored.state_dict()), parameter_hash(model.state_dict()))
                restored.eval()
                p = data.compile_inputs(raw_routing('CVRP', [.1] * 4), 'CVRP')
                with self.assertRaises(PermissionError):
                    restored.instance_encoder.processor.run(p, mode='snapshot')

    def test_export_rejects_wrong_mode_initial_state_and_incomplete_budget(self):
        initial, geometry = fresh_fixture()
        with tempfile.TemporaryDirectory() as directory:
            args, *_ = fixture(Path(directory), 'B_snapshot')
            model, _, receipt = runner.initialize_shared(initial, geometry, args.arm)
            value = export_fixture(model, receipt, args)
            process.validate_export(value, model, receipt, args.arm)
            for field, bad in (('successful_updates', 1), ('complete', False)):
                with self.assertRaises(ValueError):
                    process.validate_export(dict(value, **{field: bad}), model, receipt, args.arm)
            altered = copy.deepcopy(value)
            altered['args']['mode'] = 'transition'
            altered['state_hash'] = runner.json_hash(altered['args'])
            with self.assertRaises(ValueError):
                process.validate_export(altered, model, receipt, args.arm)
            with self.assertRaises(ValueError):
                process.validate_export(value, model, dict(receipt, fresh_state_sha256='other'), args.arm)

    def test_process_modes_have_same_targets_no_teacher_forward_and_row_weighting(self):
        from .r113_model import FeedbackProcessor
        p = data.compile_inputs(raw_routing('VRPB', [.1] * 4), 'VRPB', data.ProgramSpec(3, 2))
        y = data.execute_program(p)
        p = {k: v.repeat((5,) + (1,) * (v.ndim - 1)) for k, v in p.items()}
        y = {k: v.repeat((5,) + (1,) * (v.ndim - 1)) for k, v in y.items()}
        for mode in ('snapshot', 'transition'):
            outcomes = []
            torch.manual_seed(2)
            original = FeedbackProcessor(8).state_dict()
            for cap in (5, 2):
                model = FeedbackProcessor(8).train()
                model.load_state_dict(original)
                optimizer = torch.optim.SGD(model.parameters(), lr=.01)
                with patch.object(model, 'run', wraps=model.run) as called:
                    losses, retries, _ = process.update_batch(model, optimizer,
                        torch.amp.GradScaler('cpu', enabled=False), p, y, mode, cap)
                    for call in called.call_args_list:
                        self.assertEqual(set(call.args[0]), set(data.INPUT_KEYS))
                        self.assertIsNone(call.kwargs['context'])
                        self.assertEqual(call.kwargs['mode'], mode)
                self.assertEqual(retries, 0)
                self.assertEqual(set(losses), {*process.LOSS_KEYS, 'total'})
                self.assertAlmostEqual(losses['total'], sum(losses[k] for k in process.LOSS_KEYS), places=5)
                outcomes.append(model.state_dict())
            for key in outcomes[0]:
                torch.testing.assert_close(outcomes[0][key], outcomes[1][key], rtol=1e-5, atol=1e-7)

    def test_pretrain_entrypoint_both_modes_resume_same_rows_and_atomic_export(self):
        initial, geometry = fresh_fixture()
        real_dataset, real_update = process.ProcessDataset, process.update_batch
        with tempfile.TemporaryDirectory() as directory, patch.multiple(process,
                BATCH_SIZE=4, MAX_UPDATES=36, WARMUP_UPDATES=4, SAVE_EVERY=18):
            root = Path(directory)
            manifest_path, source_root, _ = process_fixture(root / 'assets')
            results, logs = [], []
            for arm in process.ARMS:
                args, *_ = fixture(root / 'run', arm)
                args.data_root = source_root
                write_json(source_root / 'scenario_manifest.json', dict(synthetic_only=True))
                args.prepared_root.mkdir(exist_ok=True)
                save_checkpoint(args.prepared_root / 'identicalinit_seed2.pt', initial)
                write_json(args.prepared_root / 'geometry_stats.json', geometry)
                write_json(args.prepared_root / 'prepared.json', dict(experiment='R84', seed=2,
                    data_root=str(source_root), rows_per_problem=dict(train=81000, val=4500, test=4500),
                    scenario_sha256=runner.file_hash(source_root / 'scenario_manifest.json'),
                    trainable_sha256=initial['trainable_sha256'],
                    artifact_sha256={name: runner.file_hash(args.prepared_root / name)
                        for name in ('identicalinit_seed2.pt', 'geometry_stats.json')}))
                record = runner.read_json(args.root / 'protocol_lock.json')
                record.update(data_root=str(source_root), process_manifest=str(manifest_path),
                    process_manifest_sha256=runner.file_hash(manifest_path))
                write_json(args.root / 'protocol_lock.json', record)
                calls = []

                def update(*a, **kw):
                    calls.append(1)
                    if arm == process.ARMS[0] and len(calls) == 19:
                        raise RuntimeError('synthetic interrupted process update')
                    return real_update(*a, **kw)

                tracker = SimpleNamespace(log=lambda *a, **k: None, finish=lambda: None)
                with patch.object(process, 'ProcessDataset', side_effect=lambda p, d:
                        real_dataset(p, d, rows_per_problem=8)), \
                        patch.object(process, 'wandb_run', return_value=tracker), \
                        patch.object(process, 'update_batch', side_effect=update), redirect_stdout(io.StringIO()):
                    if arm == process.ARMS[0]:
                        with self.assertRaisesRegex(RuntimeError, 'interrupted process'):
                            process.run(args, torch.device('cpu'))
                        args.resume = True
                    result = process.run(args, torch.device('cpu'))
                    # A completed export is reusable even after completion-receipt interruption.
                    exported = args.pretrain_root / arm / 'processor.pt'
                    (exported.parent / 'completion.json').unlink()
                    process.run(args, torch.device('cpu'))
                value = torch.load(exported, map_location='cpu', weights_only=False)
                self.assertEqual(value['successful_updates'], 36)
                self.assertEqual(value['updates_by_problem'], dict.fromkeys(data.PROBLEMS, 2))
                self.assertEqual(value['stream_draws'], dict.fromkeys(data.PROBLEMS, 8))
                self.assertNotIn('model', value)
                self.assertNotIn('optimizer', value)
                results.append(result)
                logs.append([json.loads(line)['sample_ids_sha256']
                    for line in (exported.parent / 'updates.jsonl').read_text().splitlines()])
            self.assertEqual(logs[0], logs[1])
            self.assertEqual(results[0]['initial_full_state_sha256'], results[1]['initial_full_state_sha256'])
            self.assertEqual(results[0]['initial_processor_sha256'], results[1]['initial_processor_sha256'])

    def test_full_cpu_runner_resumes_pending_val_and_selects_top1_regret(self):
        rows = dict(train=8, val=2, test=2)
        with tempfile.TemporaryDirectory() as directory, patch.multiple(runner,
                BATCH_SIZE=4, MAX_UPDATES=54, WARMUP_UPDATES=4, EVAL_EVERY=18,
                TRAIN_EVERY=90, SPLIT_ROWS=rows), patch.object(runner, 'make_model', tiny_factory):
            args, config, model, receipt, ready, manifest, baselines = fixture(Path(directory))
            local, run = runner.private_engine(args), args.root / args.arm
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
                return archive(run, selector, config, split, tag, step, sha,
                    top1=.4 if step == 18 else .6, regret=1. if step == 36 else 2.)

            tracker = SimpleNamespace(log=lambda *a, **k: None, finish=lambda: None)
            with patch.object(runner, 'prepared_config', return_value=(ready, manifest, {}, baselines)), \
                    patch.object(runner, 'initialize_shared', return_value=(model, model.params, receipt)), \
                    patch.object(runner, 'evaluate', side_effect=evaluate), \
                    patch.object(runner, 'open_cache', return_value=SimpleNamespace(size=8, close=lambda: None)), \
                    patch.object(runner, 'PinnedPrefetch', prefetch), \
                    patch.object(local, 'wandb_run', return_value=tracker), \
                    patch.object(runner, 'test', return_value={'synthetic_test_delegated': True}) as tested, \
                    redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(RuntimeError, 'synthetic interruption'):
                    runner.train(args, torch.device('cpu'))
                args.resume = True
                runner.train(args, torch.device('cpu'))
                tested.assert_called_once()
            lock, saved_config = runner.authorize_test(args)
            self.assertEqual(lock['selected_update'], 36)
            self.assertEqual(len(seen), 54)
            self.assertEqual(Counter(p for p, _ in seen), Counter(dict.fromkeys(runner.PROBLEMS, 3)))
            streams = {p: runner.ShuffleStream(8, runner.task_seed(2, p)) for p in runner.PROBLEMS}
            self.assertEqual(seen, [(p, streams[p].take(4).tolist()) for p in runner.task_order(54)])
            self.assertNotIn('test', [s for s, _ in evaluations])
            final = runner.read_json(run / 'final_metrics.json')
            self.assertEqual(final['val']['identity']['checkpoint_sha256'], lock['checkpoint_sha256'])
            self.assertIsNone(final['val']['archive_capture_identity']['checkpoint_sha256'])
            self.assertEqual(saved_config['processor_pretraining'], None)
            write_json(run / 'training_complete.json', dict(successful_updates=53))
            with self.assertRaises(PermissionError):
                runner.authorize_test(args)

    def test_three_arm_receipts_only_fresh_hashes_must_match(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            value = dict(fresh_reference_sha256='same', fresh_state_sha256='same',
                fresh_trainable_sha256='same', fresh_nonprocessor_sha256='same',
                initial_processor_sha256='same', parameter_layout_sha256='same', trainable_sha256='scratch')
            runner._engine_write_json(root / runner.ARMS[0] / 'initialization.json', value)
            runner._engine_write_json(root / runner.ARMS[1] / 'initialization.json', dict(value, trainable_sha256='pretrained'))
            with self.assertRaises(ValueError):
                runner._engine_write_json(root / runner.ARMS[2] / 'initialization.json', dict(value, fresh_state_sha256='other'))

    def test_nss_missing_is_explicit_four_rows_and_private_reporting(self):
        with tempfile.TemporaryDirectory() as directory:
            _, config, *_ = fixture(Path(directory))
            value = SimpleNamespace(n=2, top1_pct=50., mean_cost=1.5, pool=['synthetic'], path=Path('synthetic.npz'))
            predictions = {(s, p): value for _, p, s in reporting.NSS_DATASETS}
            with patch.object(reporting.common, 'reporting_helpers', return_value=object()):
                rows, _ = reporting.nss_comparisons(config, dict(checkpoint_sha256='sha', state_hash='state',
                    selected_update=2880), predictions, Path(directory) / 'absent_nss')
            self.assertEqual(len(rows), 4)
            self.assertTrue(all(r['status'] == 'missing' and r['nss_mean_cost'] is None for r in rows))
            self.assertEqual(reporting.display(None, 6), 'MISSING')
            self.assertIs(reporting.common.write_csv, reporting.write_csv)

    def test_finalizer_keeps_frozen_stage_order(self):
        calls = []
        with patch.object(runner, 'test', side_effect=lambda *a: calls.append('test')), \
                patch.object(runner, 'timing', side_effect=lambda *a: calls.append('timing')), \
                patch.object(reporting, 'lib_stage', side_effect=lambda *a: calls.append('lib')), \
                patch('code.V4.r113_diagnostics.finalize_diagnostics',
                      side_effect=lambda *a: calls.append('process_diagnostics')), \
                patch.object(reporting, 'report', side_effect=lambda *a: calls.append('report')):
            runner.finalize(None, torch.device('cpu'))
        self.assertEqual(calls, ['test', 'timing', 'lib', 'process_diagnostics', 'report'])

    def test_saved_only_pdf_smoke_and_paired_test_lib_axes(self):
        helper = reporting.common.reporting_helpers()
        with tempfile.TemporaryDirectory() as directory:
            args, config, *_ = fixture(Path(directory))
            run = args.root / args.arm
            run.mkdir()
            lock = dict(epoch=1, selected_update=2880, checkpoint_sha256='e' * 64,
                state_hash=runner.json_hash(config), rule=runner.CHECKPOINT_RULE)
            predictions, metrics, methods, per_problem = {}, [], [], {}
            for split in ('VAL', 'TEST', 'TSPLIB', 'CVRPLIB'):
                tasks = runner.PROBLEMS if split in ('VAL', 'TEST') else (reporting.LIBS[split],)
                rows = []
                for problem in tasks:
                    count = 49 if split == 'TSPLIB' else 100 if split == 'CVRPLIB' else 2
                    costs = np.stack([np.array([1., 2.]) if i % 2 == 0 else np.array([2., 1.])
                                      for i in range(count)]).astype(np.float64)
                    pool = list(runner.GLOBAL_SOLVERS[:2])
                    baseline = dict(selection_split='train', sbs_solver=pool[0], sbs_slot=0,
                        mean_cost_by_solver={pool[0]: 1.5, pool[1]: 1.5})
                    raw = dict(costs=costs, winner=costs.argmin(1), pool=pool, pool_ids=[0, 1])
                    saved, pred = runner.engine.fixed_sbs_metrics(np.tile([[2., 1.]], (count, 1)), raw, baseline)
                    value = helper.Predictions(path=run / 'synthetic_only.npz', pool=pool, pred=pred,
                        winner=raw['winner'], costs=costs, indices=np.arange(count), source_indices=np.arange(count),
                        pool_ids=np.array([0, 1]), score_ties=0)
                    predictions[split, problem] = value
                    rows.append(reporting.common.metric_row(problem, split, value, saved, baseline,
                        config, lock, value.path))
                    methods.extend(helper.method_rows(problem, split, saved, value, 'synthetic-only'))
                    if split == 'VAL':
                        per_problem[problem] = saved
                metrics.extend(rows)
                if split in ('VAL', 'TEST'):
                    metrics.append(reporting.common.macro_row(split, rows, runner.engine.summarize(dict.fromkeys(runner.PROBLEMS, saved))))
            summary = runner.engine.summarize(per_problem)
            write_json(run / 'history.json', [dict(successful_updates=2880,
                optimization=dict(loss=1., ce=.5, cmp=.3, risk=.2, kl=.1), val=summary, train=summary)])
            comparisons, alignment = reporting.nss_comparisons(config, lock, predictions, args.nss_root)
            seen = []
            actual_distribution = reporting.common._distribution

            def distribution(*a, **kw):
                fig = actual_distribution(*a, **kw)
                seen.append([(ax.get_title(), len(ax.containers)) for ax in fig.axes])
                return fig

            # One paired TSP panel exercises this runner's renderer; the mature
            # all18 diagnostic renderer is reused unchanged, not re-smoked here.
            with patch.object(reporting, 'PROBLEMS', ('TSP',)), \
                    patch.object(reporting.common, '_distribution', side_effect=distribution), \
                    patch.object(reporting.common, 'diagnostic_pdf', return_value=dict(synthetic_only=True)), \
                    patch.object(runner, 'load_checkpoint', side_effect=AssertionError('No inference in report')):
                result = reporting.build_reports(run, config, lock, summary, summary, predictions,
                    metrics, methods, comparisons, alignment, dict(per_problem=[]))
            self.assertTrue(Path(result['model_statistics_pdf']).read_bytes().startswith(b'%PDF'))
            self.assertEqual(len(seen), 1)
            self.assertEqual(len(seen[0]), 2)
            self.assertTrue(all(containers == 2 and 'Top1' in title for title, containers in seen[0]))
            if shutil.which('pdftotext'):
                text = subprocess.run(['pdftotext', result['model_statistics_pdf'], '-'],
                    capture_output=True, text=True, check=True).stdout
                self.assertIn('MISSING', text)
                self.assertIn('R113/A_scratch', text)
                self.assertIn('FP64', text)


if __name__ == '__main__':
    unittest.main()
