"""Executable learner interfaces, using explicit CPU-only synthetic fixtures."""

from pathlib import Path
from types import SimpleNamespace
from contextlib import ExitStack
import io
import json
import random
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from . import r121_experiment as exp
from . import r121_model as model_api
from .r76_experiment import file_hash, write_json
from .r80_experiment import json_hash
from .test_r121_model import batch, params
from .test_r121_structural_data import fixture


def rows():
    result = []
    for family, partition in (('a', 'family_train'), ('b', 'family_val')):
        for cell in ('original', 'z00', 'z01', 'z10', 'z11'):
            result.append(dict(endpoint=dict(family_id=family, group_id=family, cell=cell,
                family_partition=partition, endpoint_id=family + '_' + cell)))
    return result


class LearnerTests(unittest.TestCase):
    def assert_rng_unchanged(self, expected):
        current = exp.capture_rank_rng(torch.device('cpu'))
        self.assertEqual(expected['python'], current['python'])
        self.assertEqual(expected['numpy'][0], current['numpy'][0])
        np.testing.assert_array_equal(expected['numpy'][1], current['numpy'][1])
        self.assertEqual(expected['numpy'][2:], current['numpy'][2:])
        torch.testing.assert_close(expected['torch'], current['torch'], rtol=0, atol=0)

    def test_fixed_train_diagnostic_sampling_is_eligible_only_stable_and_rng_private(self):
        eligible = {p: np.asarray([3, 9] if i % 2 else [1, 4, 7, 11, 18, 24, 42], dtype=np.int64)
            for i, p in enumerate(exp.PROBLEMS)}
        state = exp.capture_rank_rng(torch.device('cpu'))
        with patch.object(exp, 'TRAIN_DIAGNOSTIC_ROWS', 4):
            selected, definition = exp.fixed_train_diagnostic_subset(eligible)
            repeated, again = exp.fixed_train_diagnostic_subset(eligible)
        self.assert_rng_unchanged(state)
        self.assertEqual(definition, again)
        for p in exp.PROBLEMS:
            np.testing.assert_array_equal(selected[p], repeated[p])
            self.assertTrue(np.isin(selected[p], eligible[p]).all())
            self.assertEqual(len(selected[p]), min(4, len(eligible[p])))
            self.assertEqual(definition['per_problem'][p]['indices_sha256'], exp.index_hash(selected[p]))
            self.assertEqual(definition['per_problem'][p]['eligible_indices_sha256'], exp.index_hash(eligible[p]))
        self.assertFalse(definition['used_for_checkpoint'])
        self.assertIn('NOT complete TRAIN', definition['population'])

    def test_train_diagnostic_eval_no_grad_mode_rng_and_history_schema(self):
        model = model_api.make_model(params())
        pool = [exp.GLOBAL_SOLVERS[i] for i in (0, 3, 5)]
        observations = []

        def witness(module, inputs):
            observations.append((module.training, torch.is_grad_enabled()))
            random.random()
            np.random.random()
            torch.rand(3)

        class Cache:
            pool_ids = [0, 3, 5]

            def __init__(self, problem):
                self.problem, self.pool = problem, pool
                costs = np.tile(np.asarray([[1.02, 1.01, 1.03]], dtype=np.float64), (6, 1))
                self.raw = dict(costs=costs, winner=costs.argmin(1))

            def batch(self, indices, device):
                random.random()
                np.random.random()
                torch.rand(2)
                data = batch(len(indices))
                data['problem_id'] = exp.P2I[self.problem]
                return data

        eligible = {p: np.asarray([1, 3, 5], dtype=np.int64) for p in exp.PROBLEMS}
        selected, definition = exp.fixed_train_diagnostic_subset(eligible)
        caches = {p: Cache(p) for p in exp.PROBLEMS}
        baselines = {p: dict(selection_split='train', sbs_solver=pool[1], sbs_slot=1,
            mean_cost_by_solver={name: cost for name, cost in zip(pool, (1.02, 1.01, 1.03))}) for p in exp.PROBLEMS}
        parameter = next(model.parameters())
        parameter.grad = torch.ones_like(parameter)
        before = exp.parameter_hash(model.state_dict())
        handle = model.register_forward_pre_hook(witness)
        try:
            for training in (True, False):
                model.train(training)
                state = exp.capture_rank_rng(torch.device('cpu'))
                result = exp.train_subset_diagnostic(model, caches, baselines, selected, definition, cap=2)
                self.assert_rng_unchanged(state)
                self.assertEqual(model.training, training)
                self.assertTrue(torch.is_grad_enabled())
                self.assertEqual(result['native_total'], 3 * len(exp.PROBLEMS))
                self.assertFalse(result['used_for_checkpoint'])
                self.assertEqual(result['subset'], definition)
                for key in ('macro_top1', 'macro_actual_regret_pct', 'macro_vs_sbs_pct', 'macro_ce'):
                    self.assertTrue(np.isfinite(result['macro'][key]))
                for p in exp.PROBLEMS:
                    self.assertEqual(result['per_problem'][p]['n'], 3)
                    self.assertEqual(result['per_problem'][p]['sbs_selection_split'], 'train')
                    self.assertTrue(np.isfinite(result['per_problem'][p]['ce']))
            model.train()
            state = exp.capture_rank_rng(torch.device('cpu'))
            with patch.object(model, 'forward', side_effect=RuntimeError('fixture eval failure')):
                with self.assertRaisesRegex(RuntimeError, 'fixture eval failure'):
                    exp.train_subset_diagnostic(model, caches, baselines, selected, definition, cap=2)
            self.assert_rng_unchanged(state)
            self.assertTrue(model.training)
            self.assertTrue(torch.is_grad_enabled())
        finally:
            handle.remove()
        self.assertTrue(observations)
        self.assertEqual(set(observations), {(False, False)})
        self.assertEqual(exp.parameter_hash(model.state_dict()), before)
        torch.testing.assert_close(parameter.grad, torch.ones_like(parameter), rtol=0, atol=0)

    def test_default_family_export_is_universal_single_root(self):
        expected = exp.structural.ROOT / 'collection_explicit_cpu_v2/family_labels.jsonl'
        with patch.object(exp, 'describe', return_value=dict(solver_execution=False)) as describe, \
                patch('sys.stdout', new_callable=io.StringIO):
            self.assertEqual(exp.main(['describe', '--root', str(exp.ROOT)]), 0)
        args = describe.call_args.args[0]
        self.assertEqual(args.family_labels, expected)
        self.assertEqual(args.natural_isolation, exp.DEFAULT_ISOLATION)
        self.assertEqual(args.device, 'cpu')

    def test_fixed_full18_schedule_and_population_exposure(self):
        order = exp.task_order()
        self.assertEqual(len(order), 14400)
        for start in range(0, 14400, 18):
            self.assertEqual(set(order[start:start + 18]), set(exp.PROBLEMS))
        self.assertEqual(exp.NATURAL_ROWS + 2 * exp.FAMILY_PAIRS, 1024)
        self.assertEqual(exp.exposures(14400)['natural_rows'], 11808000)
        self.assertAlmostEqual(exp.lr_at(1080), 1e-4)
        self.assertAlmostEqual(exp.lr_at(14400), 1e-6)

    def test_whole_family_partitions_and_original_to_four_real_corners(self):
        data = rows()
        self.assertEqual(exp.family_edges(data, 'family_train').tolist(), [[0, 1], [0, 2], [0, 3], [0, 4]])
        data[-1]['endpoint']['family_partition'] = 'family_train'
        with self.assertRaises(ValueError):
            exp.family_edges(data, 'family_val')

    def test_official_val_only_selects_checkpoint(self):
        value = lambda regret, top1: dict(macro=dict(macro_actual_regret_pct=regret, macro_top1=top1))
        self.assertLess(exp.choose_key(value(1., .1), 720), exp.choose_key(value(2., .99), 1440))
        self.assertLess(exp.choose_key(value(1., .9), 1440), exp.choose_key(value(1., .8), 720))
        self.assertLess(exp.choose_key(value(1., .9), 720), exp.choose_key(value(1., .9), 1440))

    def test_holdout_uses_actual_source_mapping_and_all_known_cross_task_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sources = dict(TSP=[90, 17, 81, 17, 44, 52], CVRP=[501, 401, 201, 301, 601, 701])
            endpoints = []
            for p, source in (('TSP', 17), ('CVRP', 401)):
                target = root / (p + 'train')
                target.mkdir()
                np.savez(target / 'input_metadata.npz', source_index=np.asarray(sources[p], dtype=np.int32))
                endpoints.append(dict(problem=p, cell='original', source_index=source,
                    local_index=0, family_id=p + '_val', family_partition='family_val',
                    endpoint_id=p + '_original', group_id='known_shared_geometry',
                    base_id=p + '_base', source_base_id=p + '_source'))
            manifest = exp.structural.seal(dict(endpoints=endpoints,
                tasks=[dict(task_id=e['endpoint_id'] + '__candidate', endpoint_id=e['endpoint_id'])
                       for e in endpoints]), 'manifest_sha256')
            with patch.object(exp, 'PROBLEMS', tuple(sources)), patch.object(exp, 'SPLIT_ROWS', dict(train=6)):
                eligible, identity = exp.natural_eligibility(manifest, root)
                repeated, again = exp.natural_eligibility(manifest, root)
            self.assertEqual(eligible['TSP'].tolist(), [0, 2, 4, 5])
            self.assertEqual(eligible['CVRP'].tolist(), [0, 2, 3, 4, 5])
            self.assertEqual(identity['per_problem']['TSP']['excluded_local_rows'], [1, 3])
            self.assertEqual(identity['per_problem']['CVRP']['excluded_local_rows'], [1])
            self.assertEqual(identity['associated_native_task_count'], 2)
            self.assertEqual(identity['known_geometry_groups'], ['known_shared_geometry'])
            self.assertEqual(identity['isolation_sha256'], again['isolation_sha256'])
            self.assertEqual(identity['per_problem']['TSP']['eligible_local_rows_sha256'],
                exp.index_hash(repeated['TSP']))
            draws = exp.ShuffleStream(len(eligible['TSP']), 20000).take(820).numpy()
            self.assertFalse(np.isin(eligible['TSP'][draws], [1, 3]).any())
            self.assertTrue(identity['full81000_normalization_retained'])
            self.assertTrue(identity['full81000_train_sbs_retained'])
            self.assertFalse(identity['unknown_content_copies_exhaustively_excluded'])

    def test_train_release_gate_precedes_cache_or_model_loading(self):
        args = SimpleNamespace()
        with patch.object(exp, 'authorized', side_effect=PermissionError('fixture: not released')), \
                patch.object(exp.natural, 'prepared_config') as opening:
            with self.assertRaises(PermissionError):
                exp.train(args)
            opening.assert_not_called()

    def test_native_cache_float64_not_rounded_training_cost(self):
        class Cache:
            raw = dict(costs=np.asarray([[1.0000000003, 1.]], dtype=np.float64))

            def batch(self, rows, device):
                return dict(costs=torch.tensor([[1., 1.]]), ind=torch.tensor([1]))

        data = exp.native_batch(Cache(), torch.tensor([0]), 'cpu')
        self.assertEqual(data['costs'].dtype, torch.float64)
        self.assertGreater(float(data['costs'][0, 0]), float(data['costs'][0, 1]))

    def test_real_microbatch_gradients_equal_logical_three_term_objective(self):
        torch.manual_seed(2)
        model = model_api.make_model(params()).eval()
        whole = model_api.make_model(params(), model.state_dict()).eval()
        natural_data, family_data = batch(6), batch(4)
        scaler = torch.amp.GradScaler('cuda', enabled=False)
        with patch.object(exp, 'NATURAL_ROWS', 6), patch.object(exp, 'FAMILY_PAIRS', 2):
            parts = exp.backward_parts(model, natural_data, family_data, 3, scaler)
        output = whole(natural_data)
        terms = model_api.family_losses(whole(family_data), family_data)
        loss = model_api.natural_loss(output, natural_data).mean() + sum(x.mean() for x in terms.values())
        loss.backward()
        self.assertAlmostEqual(parts['loss'], float(loss), places=5)
        for left, right in zip(model.parameters(), whole.parameters()):
            if left.grad is not None or right.grad is not None:
                torch.testing.assert_close(left.grad, right.grad, atol=2e-6, rtol=2e-5)

    def test_checkpoint_reconstruction_uses_current_factory_and_full_buffers(self):
        model = model_api.make_model(params()).eval()
        config = dict(experiment='R121', arm=exp.ARM, version=exp.VERSION, seed=2,
            max_successful_updates=14400, solver_names=list(exp.GLOBAL_SOLVERS),
            loss=dict(weights=model_api.LOSS_WEIGHTS), checkpoint_rule=exp.SELECT_RULE,
            model_params=params(), source_sha256={'r121_model.py': file_hash(Path(exp.__file__).parent / 'r121_model.py')})
        state = dict(args=config, state_hash=json_hash(config), model=model.state_dict())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'fixture.pt'
            torch.save(state, path)
            loaded, restored = exp.load_checkpoint(path)
            self.assertEqual(restored['state_hash'], state['state_hash'])
            loaded.eval()
            data = batch()
            torch.testing.assert_close(loaded(data)['logits'], model(data)['logits'], atol=0, rtol=0)

    def test_incomplete_family_export_cannot_be_training_population(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'fixture.jsonl'
            write_json(path, {})
            write_json(path.with_suffix('.receipt.json'), dict(complete=False))
            with self.assertRaises(ValueError):
                exp.family_rows(path)

    def test_cross_task_shared_base_cannot_cross_family_holdout(self):
        endpoints, exported = [], []
        pools = {p: [dict(global_id=i) for i in (0, 3, 5)] for p in exp.PROBLEMS}
        costs = np.asarray([1.02, 1.01, 1.03], dtype=np.float64)
        for p in exp.PROBLEMS:
            for row in rows():
                e = row['endpoint']
                e.update(problem=p, endpoint_id=p + '_' + e['endpoint_id'],
                    family_id=p + '_' + e['family_id'], group_id=p + '_' + e['group_id'])
                if (p == 'CVRP' and e['cell'] in ('original', 'z00', 'z01', 'z10', 'z11')
                        and e['family_partition'] == 'family_train') or (p == 'VRPB' and e['family_partition'] == 'family_val'):
                    e['group_id'] = 'explicit_fixture_shared_geometry'
                endpoints.append(e)
                exported.append(dict(endpoint=e, collection_sha256='fixture', targets=dict(
                    global_ids=[0, 3, 5], native_cost_fp64=costs.tolist(), observed_mask=[True] * 3,
                    native_winner_global_id=3, centered_log_advantage=(np.log(costs) - np.log(costs).mean()).tolist())))
        manifest = exp.structural.seal(dict(endpoints=endpoints, pools=pools,
            full18_prepared=True, full_planned_population_prepared=True), 'manifest_sha256')
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / 'fixture.jsonl'
            path.write_text(''.join(json.dumps(row, sort_keys=True) + '\n' for row in exported))
            write_json(root / 'manifest.json', manifest)
            write_json(path.with_suffix('.receipt.json'), dict(complete=True, full18=True,
                candidate_pool_unchanged=True, historical_labels_replaced=False, sha256=file_hash(path),
                path=str(path), family_input_manifest=str(root / 'manifest.json'),
                manifest_sha256=manifest['manifest_sha256'], collection_sha256='fixture', endpoints=len(endpoints)))
            with self.assertRaisesRegex(ValueError, 'including across tasks'):
                exp.family_rows(path)

    def test_real_coordinate_matrix_and_signed_feature_cache_roundtrip(self):
        problems = ('TSP', 'ATSP', 'VRPBLTW')
        population = {}
        pools = {p: [dict(method=exp.GLOBAL_SOLVERS[i], global_id=i) for i in (0, 3)] for p in problems}
        for p in problems:
            population[p] = rows()
            for row in population[p]:
                row['endpoint'].update(problem=p, scale=20)
                row['targets'] = dict(native_cost_fp64=[1.0000000003, 1.])
        receipt = dict(sha256='explicit_unit_fixture_not_production')
        with tempfile.TemporaryDirectory() as directory:
            args = SimpleNamespace(device='cpu', family_labels=Path(directory) / 'fixture.jsonl', family_cache=Path(directory))
            native = lambda data, index: np.asarray(fixture(data.problem, 1), dtype=np.float32) if data.problem == 'ATSP' else fixture(data.problem, 1)
            with patch.object(exp, 'PROBLEMS', problems), \
                    patch.object(exp, 'family_rows', return_value=(population, receipt, dict(pools=pools))), \
                    patch.object(exp.structural, 'owned', side_effect=lambda path: Path(path)), \
                    patch.object(exp.NativeInputs, '__getitem__', native):
                exp.prepare_family(args)
            for p in problems:
                cache = exp.FamilyCache(directory, p, population[p], receipt['sha256'])
                try:
                    data = cache.batch(cache.edges['family_train'][:2].reshape(-1))
                    self.assertEqual(data['costs'].dtype, torch.float64)
                    self.assertGreater(float(data['costs'][0, 0]), float(data['costs'][0, 1]))
                    self.assertEqual(len(data['n']), 4)
                    if p == 'VRPBLTW':
                        self.assertTrue((data['node'][..., 2] < 0).any())
                    if p == 'ATSP':
                        self.assertEqual(data['kind'], 'matrix')
                    else:
                        self.assertIn('node_geom', data)
                    self.assertFalse({'costs', 'ind'} & set(model_api.query_inputs(data)))
                finally:
                    cache.close()

    def test_locked_checkpoint_identity_is_accepted_by_existing_report_gate(self):
        from . import r80_reporting as reporting
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory) / exp.ARM
            run.mkdir()
            torch.save({'fixture': True}, run / 'best.pt')
            config = dict(experiment='R121', arm=exp.ARM, max_successful_updates=14400,
                checkpoint_rule=exp.SELECT_RULE, inference_rule=exp.INFERENCE_RULE,
                baselines={p: dict(selection_split='train') for p in exp.PROBLEMS})
            best = dict(epoch=1, successful_updates=720, val=dict(macro=dict(macro_actual_regret_pct=1.)))
            lock = exp.lock_best(run, best, config, json_hash(config))
            result = dict(per_problem={p: dict(n=4500) for p in exp.PROBLEMS})
            write_json(run / 'protocol.json', config)
            write_json(run / 'training_complete.json', dict(successful_updates=14400, selected_best=lock))
            write_json(run / 'test_results.json', dict(lock, complete=True, result=result))
            write_json(run / 'final_metrics.json', dict(lock, val=result))
            loaded = reporting.locked_inputs(Path(directory), exp.ARM, experiment='R121', arms={exp.ARM: 14400})
            self.assertEqual(loaded[2], lock)

    def test_cpu_fixture_full18_train_resume_lock_and_actual_test_spine(self):
        """Reduced fixture budget, real model/optimizer/evaluation, no native jobs."""
        pool_ids = [0, 3, 5]
        pool = [exp.GLOBAL_SOLVERS[i] for i in pool_ids]

        class NaturalFixture:
            def __init__(self, problem, split):
                self.problem, self.pool, self.pool_ids = problem, pool, pool_ids
                self.split = split
                self.size = 81000 if split == 'train' else 4
                costs = np.tile(np.array([[1.02, 1.01, 1.03]], dtype=np.float64), (self.size, 1))
                self.raw = dict(costs=costs, winner=costs.argmin(1).astype(np.int64))
                self.lengths = np.full(self.size, 5, dtype=np.int64)

            def batch(self, indices, device='cpu'):
                if self.split == 'train' and np.isin(np.asarray(indices), [80998]).any():
                    raise AssertionError('Known family-val source1 received natural CE gradients')
                data = batch(len(indices))
                data['problem_id'] = exp.P2I[self.problem]
                data['costs'] = data['costs'].float()
                return {k: v.to(device) if torch.is_tensor(v) else v for k, v in data.items()}

            def close(self):
                pass

        class FamilyFixture:
            def __init__(self, root, problem, ignored, digest):
                self.problem, self.pool_ids = problem, pool_ids
                self.edges = dict(family_train=np.asarray([[0, 1], [0, 2]], dtype=np.int64),
                    family_val=np.asarray([[2, 3]], dtype=np.int64))

            def batch(self, indices, device='cpu'):
                data = batch(len(indices))
                data['problem_id'] = exp.P2I[self.problem]
                for i, index in enumerate(indices):
                    if int(index) % 2:
                        data['costs'][i] = torch.tensor([1.01, 1.03, 1.02], dtype=torch.float64)
                data['ind'] = data['costs'].argmin(1)
                return {k: v.to(device) if torch.is_tensor(v) else v for k, v in data.items()}

            def close(self):
                pass

        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            root = Path(directory)
            prepared = root / 'prepared'
            prepared.mkdir()
            torch.save({'explicit_untrained_unit_fixture': True}, prepared / 'identicalinit_seed2.pt')
            write_json(prepared / 'prepared.json', dict(fixture=True))
            population = {p: rows() for p in exp.PROBLEMS}
            for p, records in population.items():
                for row in records:
                    e = row['endpoint']
                    source = 1 if e['family_partition'] == 'family_val' else 0
                    e.update(problem=p, source_index=source, local_index=47,
                        endpoint_id=p + '_' + e['endpoint_id'], family_id=p + '_' + e['family_id'],
                        group_id=p + '_' + e['group_id'], base_id=p + '_base_' + str(source),
                        source_base_id=p + '_source_' + str(source))
            family_manifest = exp.structural.seal(dict(plan_sha256='explicit_cpu_fixture',
                endpoints=[row['endpoint'] for records in population.values() for row in records],
                pools={p: [dict(global_id=i) for i in pool_ids] for p in exp.PROBLEMS}), 'manifest_sha256')
            write_json(root / 'family_manifest.json', family_manifest)
            for p in exp.PROBLEMS:
                for split in ('train', 'val', 'test'):
                    destination = root / 'data' / (p + split)
                    destination.mkdir(parents=True)
                    source = np.arange(81000, dtype=np.int64)[::-1] if split == 'train' else np.arange(4, dtype=np.int64)
                    np.savez(destination / 'input_metadata.npz', source_index=source)
            baselines = {p: dict(selection_split='train', sbs_solver=pool[1], sbs_slot=1,
                mean_cost_by_solver={name: cost for name, cost in zip(pool, (1.02, 1.01, 1.03))}) for p in exp.PROBLEMS}
            manifest = dict(splits={s: {p: dict(pool=pool, pool_ids=pool_ids, rows=81000 if s == 'train' else 4)
                for p in exp.PROBLEMS} for s in ('train', 'val', 'test')})
            args = SimpleNamespace(root=root / 'learner_fixture', arm=exp.ARM, device='cpu', experiment='R121',
                data_root=root / 'data', prepared_root=prepared, cache_root=root / 'cache',
                test_cache_root=root / 'test_cache', family_cache=root / 'family_cache',
                family_labels=root / 'fixture.jsonl', family_input_manifest=root / 'family_manifest.json',
                natural_isolation=root / 'learner_isolation.json',
                gpu_type='4090', micro_cap=2, eval_batch_size=2, wandb_mode='disabled', resume=False)
            for name, value in dict(MAX_UPDATES=36, WARMUP=3, EVAL_EVERY=18,
                    BATCH_SIZE=6, NATURAL_ROWS=4, FAMILY_PAIRS=1, TRAIN_DIAGNOSTIC_ROWS=4,
                    SPLIT_ROWS=dict(train=81000, val=4, test=4)).items():
                stack.enter_context(patch.object(exp, name, value))
            stack.enter_context(patch.object(exp.structural, 'owned', side_effect=lambda p: Path(p)))
            stack.enter_context(patch.object(exp, 'authorized', return_value=dict(explicit_cpu_fixture=True)))
            stack.enter_context(patch.object(exp.natural, 'prepared_config',
                return_value=(dict(trainable_sha256='untrained_fixture'), manifest, {}, baselines)))
            stack.enter_context(patch.object(exp, 'family_rows', return_value=(
                population, dict(sha256='fixture_labels'), family_manifest)))
            stack.enter_context(patch.object(exp.model_api, 'initialize_shared',
                side_effect=lambda initial, geometry: (model_api.make_model(params()), params(), dict(fixture=True))))
            stack.enter_context(patch.object(exp, 'FamilyCache', FamilyFixture))
            stack.enter_context(patch.object(exp, 'open_cache',
                side_effect=lambda root, split, p, manifest, data_root: NaturalFixture(p, split)))
            stack.enter_context(patch.object(exp, 'wandb_run', return_value=None))
            with patch.object(exp, 'evaluate', side_effect=RuntimeError('fixture interruption at pending VAL')):
                with self.assertRaisesRegex(RuntimeError, 'fixture interruption'):
                    exp.train(args)
            saved = torch.load(args.root / exp.ARM / 'last.pt', map_location='cpu', weights_only=False)
            self.assertEqual(saved['successful_updates'], 18)
            self.assertIsNotNone(saved['pending_evaluation'])
            self.assertEqual(set(saved['args']['natural_ce_eligible_rows_per_problem'].values()), {80999})
            self.assertFalse(saved['args']['family_population']['heldout_base_still_in_natural_CE'])
            self.assertEqual(saved['args']['train_diagnostic']['total_rows_per_interval'], 72)
            args.resume = True
            outcome = exp.train(args)
            self.assertTrue(outcome['complete'])
            final_state = torch.load(args.root / exp.ARM / 'last.pt', map_location='cpu', weights_only=False)
            self.assertEqual(final_state['successful_updates'], 36)
            self.assertEqual(set(final_state['updates_by_problem'].values()), {2})
            for entry in exp.read_json(args.root / exp.ARM / 'history.json'):
                self.assertEqual(entry['train']['native_total'], 72)
                self.assertEqual(entry['train']['subset'], saved['args']['train_diagnostic'])
                self.assertFalse(entry['train']['used_for_checkpoint'])
                self.assertIn('family_val', entry)
            result = exp.test(args)
            self.assertEqual(set(result['per_problem']), set(exp.PROBLEMS))
            self.assertEqual(result['native_total'], 72)
            # The second TEST call must reuse the exact saved predictions/metrics.
            with patch.object(exp, 'open_cache', side_effect=AssertionError('saved result should not read data')):
                repeated = exp.test(args)
            self.assertEqual(result['identity'], repeated['identity'])
            from . import r80_reporting as reporting
            loaded = reporting.locked_inputs(args.root, exp.ARM, experiment='R121', arms={exp.ARM: 36})
            self.assertEqual(loaded[2]['selected_update'], outcome['selected_best']['successful_updates'])

    def test_learning_curves_show_fixed_train_scope_and_preserve_saved_points(self):
        import matplotlib.pyplot as plt
        history = [dict(successful_updates=720, optimization=dict(loss=2.),
            train=dict(macro=dict(macro_top1=.25, macro_actual_regret_pct=3., macro_vs_sbs_pct=-1.)),
            val=dict(macro=dict(macro_top1=.2, macro_actual_regret_pct=4., macro_vs_sbs_pct=-.5))),
            dict(successful_updates=1440, optimization=dict(loss=1.),
            train=dict(macro=dict(macro_top1=.5, macro_actual_regret_pct=2., macro_vs_sbs_pct=-2.)),
            val=dict(macro=dict(macro_top1=.4, macro_actual_regret_pct=3., macro_vs_sbs_pct=-1.5)))]
        config = dict(experiment='R121', arm=exp.ARM, train_diagnostic=dict(seed=2,
            per_problem={p: dict(rows=1024) for p in exp.PROBLEMS}))
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            write_json(run / 'history.json', history)
            with patch.object(plt, 'close', wraps=plt.close) as close, \
                    patch.object(exp, 'load_checkpoint', side_effect=AssertionError('saved-only chart')):
                receipt = exp.scoped_learning_curves(run, run, config, dict(selected_update=720))
            fig = close.call_args.args[0]
            footer = ' '.join(text.get_text() for text in fig.texts)
            self.assertIn('fixed eligible TRAIN subset (1024/task; seed 2)', footer)
            self.assertIn('NOT full 81,000 TRAIN rows/task', footer)
            self.assertIn('training batches', fig.axes[0].get_title())
            for ax, values in zip(fig.axes[1:], ([25., 50.], [3., 2.], [-1., -2.])):
                self.assertIn('TRAIN: fixed eligible subset (1024/task)',
                    [text.get_text() for text in ax.get_legend().get_texts()])
                np.testing.assert_array_equal(ax.lines[0].get_xdata(), [720, 1440])
                np.testing.assert_array_equal(ax.lines[0].get_ydata(), values)
                np.testing.assert_array_equal(ax.lines[-1].get_xdata(), [720, 720])
            self.assertTrue((run / 'learning_curves.pdf').read_bytes().startswith(b'%PDF-'))
            self.assertTrue((run / 'learning_curves.png').read_bytes().startswith(b'\x89PNG'))
            self.assertFalse(receipt['full_train'])
            self.assertFalse(receipt['model_inference'])
            self.assertEqual(exp.read_json(run / 'history.json'), history)

    def test_finish_requires_both_libs_and_only_uses_saved_results(self):
        from . import r80_reporting as reporting
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory) / exp.ARM
            run.mkdir()
            output = run / 'full_model_statistics'
            output.mkdir()
            args = SimpleNamespace(root=Path(directory))
            config = dict(model_factory='code.V4.r121_experiment.load_checkpoint', family_labels={'fixture': True},
                family_population=dict(explicit_unit_fixture=True),
                train_diagnostic=dict(explicit_unit_fixture=True),
                loss={'weights': model_api.LOSS_WEIGHTS}, exposures=exp.exposures(14400), historical_control='not matched')
            lock = dict(checkpoint_sha256='fixture', selected_update=720, state_hash='fixture')
            result = dict(macro={})
            write_json(run / 'lib_stage_complete.json', dict(complete=True, checkpoint_sha256='fixture', datasets=['TSPLIB']))
            with patch.object(reporting, 'locked_inputs', return_value=(run, config, lock, result, result)), \
                    patch.object(exp, 'report', return_value=dict(fixture=True)) as render, \
                    patch.object(exp, 'load_checkpoint', side_effect=AssertionError('finish must not infer')), \
                    patch.object(exp, 'authorized', side_effect=AssertionError('finish must be CPU/saved-only')):
                with self.assertRaisesRegex(ValueError, 'both native LIBs'):
                    exp.finish(args)
                render.assert_not_called()
                write_json(run / 'lib_stage_complete.json', dict(complete=True, checkpoint_sha256='fixture', datasets=['TSPLIB', 'CVRPLIB']))
                for name in ('MODEL_STATISTICS.pdf', 'selection_distributions.pdf', 'learning_curves.pdf',
                        'training_and_method_diagnostics.pdf', 'structural_objectives.pdf', 'metrics_by_problem.csv',
                        'selection_distribution.csv', 'single_method_comparison.csv', 'nss_comparison.csv', 'protocol.json'):
                    (output / name).write_bytes(b'%PDF-1.4\nEXPLICIT_UNIT_STUB' if name.endswith('.pdf') else b'EXPLICIT_UNIT_STUB')
                summary = exp.finish(args)
                self.assertTrue(summary['complete'])
                self.assertFalse(summary['model_inference'])
                self.assertFalse(summary['research_goal_complete'])
                self.assertEqual(len(summary['artifacts']), 10)


if __name__ == '__main__':
    unittest.main()
