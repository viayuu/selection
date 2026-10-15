"""Focused offline CPU tests; no real training, dataset reads, solver or API calls."""

import copy
import math
import tempfile
import unittest
from argparse import Namespace
from collections import Counter
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from code.unified_selector.registry import GLOBAL_SOLVERS, PROBLEMS
from . import r76_experiment as r76
from .pairwise_objective import comparison_targets
from .solver_code_encoder import make_r45_model
from .test_dual_stream import params
from .test_solver_features import sample_batch


def fixture_model(dropout=.1):
    torch.manual_seed(2)
    config = dict(params(), dropout=dropout, local_geometry=True, geometry_mode='real',
                  pair_mode='score_difference', solver_representation='handcrafted')
    return make_r45_model(config), config


def fixture_batch(problem, size=5):
    torch.manual_seed(700 + PROBLEMS.index(problem))
    batch = sample_batch(problem)
    if problem not in ('TSP', 'CVRP', 'ATSP'):
        batch['pool_ids'] = torch.tensor([10, 11, 16] if 'B' in problem else [10, 11, 12, 13, 16, 18])
    for key, value in list(batch.items()):
        if torch.is_tensor(value) and key != 'pool_ids':
            batch[key] = value[:1].repeat((size,) + (1,) * (value.ndim - 1))
    batch['node_geom'] = torch.zeros(size, 5, 12)
    k = len(batch['pool_ids'])
    costs = np.stack([1. + np.roll(np.arange(k), row % k) for row in range(size)]).astype(np.float64)
    costs[0, :2] = [2. + 5e-10, 2.]
    batch.update(comparison_targets(costs))
    batch['costs'] = torch.from_numpy(costs.astype(np.float32))
    batch['ind'] = torch.from_numpy(costs.argmin(-1).astype(np.int64))
    return batch, costs


def subset(batch, indices):
    return {key: value[indices] if torch.is_tensor(value) and key != 'pool_ids' else value
            for key, value in batch.items()}


class FakeCache:
    def __init__(self, problem, split='train', size=5):
        self.problem, self.split, self.size = problem, split, size
        self.data, costs = fixture_batch(problem, size)
        self.kind, self.pool_ids = self.data['kind'], self.data['pool_ids'].tolist()
        self.pool = ['reld_mt' if index == 16 else GLOBAL_SOLVERS[index] for index in self.pool_ids]
        self.raw = dict(costs=costs, winner=self.data['ind'].numpy().copy(), pool=self.pool, pool_ids=self.pool_ids)
        self.lengths = self.data['n'].numpy()
        self.metadata = dict(data_root='/fixture/data', rows=size, pool=self.pool, pool_ids=self.pool_ids)
        self.seen = []

    def batch(self, indices, device='cpu'):
        self.seen.extend(indices.tolist())
        return subset(self.data, indices)


class CountingSGD(torch.optim.SGD):
    def __init__(self, parameters):
        super().__init__(parameters, lr=1e-4)
        self.steps = 0

    def step(self, *args, **kwargs):
        self.steps += 1
        return super().step(*args, **kwargs)


def ddp_worker(rank, init_file, output_directory):
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method='file://' + init_file, rank=rank, world_size=2)
    try:
        model, _ = fixture_model()
        runtime = r76.Runtime(rank, 2, torch.device('cpu'))
        objective = r76._objective(model, runtime)
        optimizer = CountingSGD(model.parameters())
        scaler = torch.amp.GradScaler('cpu', init_scale=8., growth_interval=100)
        batch, _ = fixture_batch('TSP', 3)
        rows = r76.split_batch(torch.arange(3), [2, 1])[rank]
        calls, injected = [], [False]

        def observe(module, arguments, output):
            calls.append(output['logits'].detach().clone())

        def inject_overflow(gradient):
            if rank == 1 and not injected[0]:
                injected[0] = True
                return torch.full_like(gradient, float('nan'))
            return gradient

        forward_hook = model.register_forward_hook(observe)
        gradient_hook = next(model.score_head.parameters()).register_hook(inject_overflow)
        r76.set_seed(2 + 104729 * rank)
        _, retries = r76.update_batch(objective, optimizer, scaler, subset(batch, rows), .35,
                                      len(rows), 3, runtime, max_retries=3, microbatch_cap=1)
        forward_hook.remove()
        gradient_hook.remove()
        replayed = len(calls) == 8 and all(torch.equal(calls[i], calls[i + 4]) for i in range(4))
        # Change input branches with a genuine zero-row rank, then activate MVRP.
        batch, _ = fixture_batch('ATSP', 1)
        _, second_retries = r76.update_batch(objective, optimizer, scaler, batch if rank == 0 else None,
                                             .35, int(rank == 0), 1, runtime)
        batch, _ = fixture_batch('OVRPTW', 2)
        _, third_retries = r76.update_batch(objective, optimizer, scaler, subset(batch, torch.tensor([rank])),
                                            .35, 1, 2, runtime)
        r76.save_checkpoint(Path(output_directory) / f'rank{rank}.pt', dict(
            model=model.state_dict(), steps=optimizer.steps, retries=retries,
            other_retries=[second_retries, third_retries], replayed=replayed, scale=scaler.get_scale()))
    finally:
        dist.destroy_process_group()


class R76Tests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)

    def test_epoch_exact_coverage_problem_caps_and_active_cycles(self):
        sizes = dict(TSP=3587, CVRP=1793, ATSP=23, OVRPTW=17)
        caps = dict(TSP=[512, 512, 384, 384], CVRP=[384, 384, 256, 256],
                    ATSP=[8, 8, 6, 6], OVRPTW=[3, 3, 2, 2])
        logical = 1000
        plan = r76.epoch_plan(sizes, caps, logical_batch_size=logical)
        expected = {p: math.ceil(size / logical) for p, size in sizes.items()}
        self.assertEqual(Counter(p for p, _ in plan), expected)
        remaining, offset = expected.copy(), 0
        while any(remaining.values()):
            active = {p for p, n in remaining.items() if n}
            cycle = plan[offset:offset + len(active)]
            self.assertEqual({p for p, _ in cycle}, active)
            for p, indices in cycle:
                self.assertLessEqual(len(indices), logical)
                remaining[p] -= 1
            offset += len(active)
        for p, size in sizes.items():
            rows = torch.cat([indices for problem, indices in plan if problem == p])
            torch.testing.assert_close(rows.sort().values, torch.arange(size), rtol=0, atol=0)
        repeated = r76.epoch_plan(dict(reversed(list(sizes.items()))), caps, logical_batch_size=logical)
        for (p, rows), (q, other) in zip(plan, repeated):
            self.assertEqual(p, q)
            torch.testing.assert_close(rows, other, rtol=0, atol=0)
        changed = r76.epoch_plan(sizes, caps, epoch=1, logical_batch_size=logical)
        self.assertFalse(torch.equal(torch.cat([x for p, x in plan if p == 'TSP']),
                                     torch.cat([x for p, x in changed if p == 'TSP'])))

    def test_split_all_tails_no_duplicate_padding_or_cap_overrun(self):
        caps = [512, 512, 384, 384]
        for count in range(sum(caps) + 1):
            rows = torch.arange(count)
            chunks = r76.split_batch(rows, caps)
            torch.testing.assert_close(torch.cat(chunks), rows, rtol=0, atol=0)
            for rank, chunk in enumerate(chunks):
                self.assertLessEqual(len(chunk), caps[rank])
                self.assertLess(abs(len(chunk) - count * caps[rank] / sum(caps)), 1.000001)
        self.assertEqual([len(v) for v in r76.split_batch(torch.arange(1), caps)], [1, 0, 0, 0])
        self.assertEqual([len(v) for v in r76.split_batch(torch.arange(1792), caps)], caps)
        torch.testing.assert_close(torch.cat(r76.split_batch(torch.arange(9856), caps)), torch.arange(9856))
        with self.assertRaises(ValueError):
            r76.epoch_plan({'TSP': 1}, {'CVRP': [1, 1, 1, 1]})
        with self.assertRaises(ValueError):
            r76.epoch_plan({'TSP': 1}, logical_batch_size=0)

    def test_uniform_logical_batch_all18_rows_once_and_protocol(self):
        caps = {p: [2464] * 4 for p in PROBLEMS}
        caps.update(TSP=[443, 443, 416, 416], CVRP=[352, 352, 384, 384],
                    ATSP=[2496, 2496, 2464, 2464])
        sizes = dict.fromkeys(PROBLEMS, 70114)
        plan = r76.epoch_plan(sizes, caps)
        self.assertEqual(len(plan), 144)
        self.assertEqual(Counter(p for p, _ in plan), dict.fromkeys(PROBLEMS, 8))
        for p in PROBLEMS:
            batches = [rows for problem, rows in plan if problem == p]
            self.assertEqual([len(rows) for rows in batches], [9856] * 7 + [1122])
            rows = torch.cat([rank for batch in batches for rank in r76.split_batch(batch, caps[p])])
            torch.testing.assert_close(rows.sort().values, torch.arange(70114), rtol=0, atol=0)
            rounds = max(math.ceil(len(rank) / cap)
                         for rank, cap in zip(r76.split_batch(batches[0], caps[p]), caps[p]))
            self.assertEqual(rounds, 6 if p == 'TSP' else 7 if p == 'CVRP' else 1)
        args = r76.parser().parse_args(['--stage', 'train', '--root', '/fixture/run',
                                       '--cache-root', '/fixture/cache'])
        self.assertEqual(args.logical_batch_size, 9856)
        config = r76.protocol(args, {}, dict(splits={'train': {}, 'val': {}}), {}, {}, sizes, caps)
        self.assertEqual(config['logical_global_batch'], 9856)
        self.assertEqual(config['microbatch_caps_by_problem'], caps)
        self.assertEqual(config['global_batch_by_problem'], dict.fromkeys(PROBLEMS, 9856))
        self.assertEqual(config['updates_per_epoch'], dict.fromkeys(PROBLEMS, 8))
        self.assertEqual(config['total_updates_per_epoch'], 144)
        self.assertTrue(config['uniform_updates_per_task'])

    def test_microbatch_slice_preserves_static_fields_even_with_matching_lengths(self):
        batch = dict(ind=torch.arange(4), samples=torch.randn(4, 3), pool_ids=torch.arange(4),
                     kind='coord', problem_id=torch.arange(4), d_in=torch.arange(4), scalar=torch.tensor(4))
        chunk = r76.slice_batch(batch, 1, 3)
        torch.testing.assert_close(chunk['ind'], torch.tensor([1, 2]))
        torch.testing.assert_close(chunk['samples'], batch['samples'][1:3])
        for key in ('pool_ids', 'kind', 'problem_id', 'd_in', 'scalar'):
            self.assertIs(chunk[key], batch[key])
        self.assertIsNone(r76.slice_batch(None, 0, 1))

    def test_entire_weighted_objective_gradient_is_instance_mean(self):
        torch.manual_seed(2)
        caps = [3, 3, 2, 2]
        for count in (1, 3, 7, 10, 13, 21):
            samples = torch.randn(count, 4, dtype=torch.float64)
            reference = torch.tensor([.2, -.3, .1, .4], dtype=torch.float64, requires_grad=True)
            distributed = reference.detach().clone().requires_grad_(True)

            def objective(weight, rows):
                terms = (samples[rows] * weight).square()
                return (.35 * terms[:, 0] + .35 * terms[:, 1] + .02 * terms[:, 2] + .35 * terms[:, 3]).mean()

            full = objective(reference, torch.arange(count))
            full.backward()
            combined = distributed.sum() * 0
            for rows, cap in zip(r76.split_batch(torch.arange(count), caps), caps):
                for micro_rows in rows.split(cap):
                    if len(micro_rows):
                        combined = combined + objective(distributed, micro_rows) * r76.rank_loss_scale(
                            len(micro_rows), count, 4) / 4
            combined.backward()
            torch.testing.assert_close(combined, full, rtol=1e-12, atol=1e-12)
            torch.testing.assert_close(distributed.grad, reference.grad, rtol=1e-12, atol=1e-12)

    def test_update_microchunks_match_full_gradient_metrics_and_single_step(self):
        class Objective(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor([.2, -.3, .1, .4], dtype=torch.float64))
                self.calls = []

            def forward(self, batch, consistency_weight):
                self.calls.append((len(batch['ind']), self.weight.detach().clone(), consistency_weight))
                ce, cmp, risk, kl = (batch['samples'] * self.weight).square().mean(0).unbind()
                loss = .35 * ce + .35 * cmp + .02 * risk + consistency_weight * kl
                return loss, torch.stack((loss, ce, cmp, risk, kl, ce, cmp, risk))

        torch.manual_seed(2)
        batch = dict(ind=torch.arange(13), samples=torch.randn(13, 4, dtype=torch.float64))
        reference, accumulated = Objective(), Objective()
        runtime = r76.Runtime(0, 1, torch.device('cpu'))
        reference_optimizer = CountingSGD(reference.parameters())
        optimizer = CountingSGD(accumulated.parameters())
        full_parts, _ = r76.update_batch(reference, reference_optimizer, torch.amp.GradScaler('cpu', enabled=False),
                                         batch, .35, 13, 13, runtime)
        scaler = torch.amp.GradScaler('cpu', enabled=False)
        with (patch.object(scaler, 'unscale_', wraps=scaler.unscale_) as unscale,
              patch.object(torch.nn.utils, 'clip_grad_norm_', wraps=torch.nn.utils.clip_grad_norm_) as clip):
            parts, retries = r76.update_batch(accumulated, optimizer, scaler, batch, .35,
                                               13, 13, runtime, microbatch_cap=3)
        self.assertEqual(retries, 0)
        self.assertEqual(optimizer.steps, 1)
        unscale.assert_called_once()
        clip.assert_called_once()
        self.assertEqual([count for count, _, _ in accumulated.calls], [3, 3, 3, 3, 1])
        for _, weights, consistency_weight in accumulated.calls:
            torch.testing.assert_close(weights, accumulated.calls[0][1], rtol=0, atol=0)
            self.assertEqual(consistency_weight, .35)
        torch.testing.assert_close(parts, full_parts, rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(accumulated.weight.grad, reference.weight.grad, rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(accumulated.weight, reference.weight, rtol=1e-12, atol=1e-12)

    def test_cpu_r45_objective_both_passes_receive_gradients(self):
        for problem in ('TSP', 'CVRP', 'ATSP', 'OVRPTW', 'VRPB'):
            with self.subTest(problem=problem):
                model, _ = fixture_model()
                batch, costs = fixture_batch(problem, 3)
                self.assertEqual(costs.dtype, np.float64)
                self.assertFalse(bool(batch['cmp_target'][0, 0]))
                self.assertTrue(bool(batch['cmp_valid'][0, 0]))
                outputs, gradients = [], []

                def observe(module, arguments, output):
                    outputs.append(output['logits'].detach().clone())
                    output['logits'].register_hook(lambda grad: gradients.append(grad.detach().clone()))

                hook = model.register_forward_hook(observe)
                before = next(model.score_head.parameters()).detach().clone()
                optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
                scaler = torch.amp.GradScaler('cuda', enabled=False)
                parts, retries = r76.update_batch(r76.RDropObjective(model), optimizer, scaler, batch, .35,
                                                  3, 3, r76.Runtime(0, 1, torch.device('cpu')))
                hook.remove()
                self.assertEqual(retries, 0)
                self.assertEqual(len(outputs), 2)
                self.assertEqual(len(gradients), 2)
                self.assertFalse(torch.equal(outputs[0], outputs[1]))
                self.assertTrue(all(torch.isfinite(g).all() and g.abs().sum() > 0 for g in gradients))
                torch.testing.assert_close(parts[0], .35 * parts[1] + .35 * parts[2] + .02 * parts[3] + .35 * parts[4])
                self.assertFalse(torch.equal(before, next(model.score_head.parameters())))
                self.assertGreater(sum(float(p.grad.abs().sum()) for p in model.solver_encoder.feature_mlp.parameters()), 0.)

    def test_prefetch_is_ordered_and_never_gathers_empty_rank(self):
        cache = FakeCache('TSP')
        requests = [('TSP', torch.tensor([4, 1]), 3), ('TSP', torch.empty(0, dtype=torch.long), 1),
                    ('TSP', torch.tensor([0]), 1)]
        with r76.PinnedPrefetch({'TSP': cache}, requests, torch.device('cpu')) as stream:
            batches = list(stream)
        self.assertEqual([count for _, count, _, _ in batches], [2, 0, 1])
        self.assertIsNone(batches[1][3])
        self.assertEqual(cache.seen, [4, 1, 0])

    def test_fixed_train_sbs_fp64_and_exact_native_correct(self):
        raw = dict(costs=np.array([[1. + 5e-10, 1., 4.], [10., 12., 13.]], dtype=np.float64),
                   winner=np.array([1, 0], dtype=np.int64), pool=['a', 'b', 'reld_mt'], pool_ids=[11, 10, 16])
        scores = np.array([[0., 0., 0.], [0., 1., -.1]], dtype=np.float32)
        baseline = dict(selection_split='train', sbs_solver='reld_mt', sbs_slot=2,
                        mean_cost_by_solver={'a': 10., 'b': 9., 'reld_mt': 8.})
        result, pred = r76.fixed_sbs_metrics(scores, raw, baseline)
        np.testing.assert_array_equal(pred, [1, 1])
        self.assertEqual(result['native_correct'], 1)
        self.assertEqual(result['actual_regret_pct'], 10.)
        self.assertEqual(result['train_sbs_mean_cost'], 8.)
        self.assertEqual(result['sbs_cost'], 8.5)
        self.assertEqual(result['posthoc_split_sbs_name'], 'a')
        self.assertAlmostEqual(result['vs_sbs_pct'], (6.5 / 8.5 - 1.) * 100.)
        self.assertNotEqual(result['vs_sbs_pct'], result['posthoc_split_vs_sbs_pct'])

    def test_full18_evaluation_keeps_tails_and_rank_npz_precision(self):
        model, _ = fixture_model()
        caches = {p: FakeCache(p, 'val', 3 + i % 3) for i, p in enumerate(PROBLEMS)}
        manifest = dict(splits={'val': {p: dict(rows=c.size) for p, c in caches.items()}})
        baselines = {p: dict(selection_split='train', sbs_solver=c.pool[0], sbs_slot=0,
                            mean_cost_by_solver=dict(zip(c.pool, c.raw['costs'].mean(0).tolist())))
                     for p, c in caches.items()}
        with tempfile.TemporaryDirectory() as directory:
            args = Namespace(root=Path(directory), cache_root=Path(directory), data_root=Path('/fixture/data'),
                             eval_batch_size=2)
            with patch.object(r76, 'open_cache', side_effect=lambda root, split, p, *rest: caches[p]):
                result = r76.evaluate(model, args, r76.Runtime(0, 1, torch.device('cpu')),
                                      manifest, baselines, 'val', 'fixture')
            self.assertEqual(result['native_total'], sum(c.size for c in caches.values()))
            self.assertEqual(result['native_correct'], sum(v['native_correct'] for v in result['per_problem'].values()))
            self.assertEqual(set(result['families']), {'TSP', 'CVRP', 'ATSP', 'MVRP'})
            self.assertTrue(model.training)
            with np.load(Path(directory) / 'evaluation/val/fixture/rank000.npz') as saved:
                for p, cache in caches.items():
                    self.assertEqual(cache.seen, list(range(cache.size)))
                    self.assertEqual(saved[p + '__costs'].dtype, np.float64)
                    self.assertEqual(saved[p + '__winner'].dtype, np.int64)

    def test_checkpoint_reconstruction_is_self_contained(self):
        model, config = fixture_model(dropout=0.)
        model.eval()
        batch, _ = fixture_batch('VRPB', 2)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'checkpoint.pt'
            r76.save_checkpoint(path, dict(model=model.state_dict(), args=dict(
                experiment='R76', initialization='scratch', solver_names=GLOBAL_SOLVERS, model_params=config)))
            with (patch.object(r76, 'initialize_r45', side_effect=AssertionError('No external architecture file')),
                  patch('urllib.request.urlopen', side_effect=AssertionError('No API'))):
                loaded, _ = r76.load_r76_checkpoint(path)
            loaded.eval()
            torch.testing.assert_close(loaded(batch)['logits'], model(batch)['logits'], rtol=0, atol=0)

    def test_unique_test_lock_and_one_shot_guard(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaises(PermissionError):
                r76.authorize_test(root, '/fixture/data')
            config = {'data_root': '/fixture/data'}
            r76.save_checkpoint(root / 'best.pt', dict(epoch=2, best={'epoch': 3}, args=config))
            best = dict(epoch=3, val=dict(macro={'macro_vs_sbs_pct': -.1}))
            lock = r76.lock_best(root, best)
            self.assertEqual(lock, r76.lock_best(root, best))
            r76.write_json(root / 'training_complete.json', {'complete': True})
            r76.write_json(root / 'protocol.json', config)
            self.assertEqual(r76.authorize_test(root, '/fixture/data')[0], lock)
            with self.assertRaises(ValueError):
                r76.lock_best(root, dict(best, epoch=4))
            r76.write_json(root / 'test_started.json', {'claimed': True})
            with self.assertRaises(PermissionError):
                r76.authorize_test(root, '/fixture/data')

    def test_best_epoch_mismatch_rejected_before_test_marker(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = {'data_root': '/fixture/data'}
            selected = dict(epoch=3, val=dict(macro={'macro_vs_sbs_pct': -.1}))
            checkpoint = dict(epoch=2, best={'epoch': 3}, args=config)
            r76.save_checkpoint(root / 'best.pt', checkpoint)
            r76.lock_best(root, selected)
            r76.write_json(root / 'training_complete.json', {'complete': True})
            r76.write_json(root / 'protocol.json', config)
            checkpoint['epoch'] = 1
            with self.assertRaisesRegex(ValueError, 'epoch'):
                r76._validate_best_epoch(checkpoint, selected['epoch'])
            r76.save_checkpoint(root / 'best.pt', checkpoint)
            with self.assertRaisesRegex(ValueError, 'epoch'):
                r76.lock_best(root, selected)
            lock = r76.read_json(root / 'final_test_lock.json')
            lock['checkpoint_sha256'] = r76.file_hash(root / 'best.pt')
            r76.write_json(root / 'final_test_lock.json', lock)
            args = Namespace(root=root, data_root=Path('/fixture/data'))
            with self.assertRaisesRegex(PermissionError, 'epoch'):
                r76.test(args, r76.Runtime(0, 1, torch.device('cpu')))
            self.assertFalse((root / 'test_started.json').exists())

    def test_test_authorization_barrier_precedes_marker(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = Namespace(root=root, data_root=Path('/fixture/data'))
            events = []

            class BarrierRuntime:
                rank = 0

                def barrier(self):
                    events.append(('barrier', (root / 'test_started.json').exists()))

            with (patch.object(r76, 'authorize_test', return_value=(dict(checkpoint_sha256='fixture', epoch=1), {})),
                  patch.object(r76, 'read_json', side_effect=RuntimeError('Stop before opening TEST'))):
                with self.assertRaisesRegex(RuntimeError, 'Stop before opening TEST'):
                    r76.test(args, BarrierRuntime())
            self.assertEqual(events, [('barrier', False), ('barrier', True)])

    def test_admin_batch_config_combines_receipts_without_gpu_or_model(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for kind, big, small in (('4090', 1024, 4096), ('3090', 768, 3072)):
                r76.write_json(root / f'capacity_{kind}.json', dict(gpu_type=kind, complete=True,
                    formal_weights_retained=False, recommended_caps={
                        p: big if p in ('TSP', 'CVRP') else small for p in PROBLEMS}))
            r76.write_json(root / 'cache_manifest.json', dict(splits={'train': {p: dict(rows=101) for p in PROBLEMS}}))
            with (patch.object(r76, 'initialize_r45', side_effect=AssertionError('No model')),
                  patch.object(r76, 'open_cache', side_effect=AssertionError('No cache read'))):
                r76.batch_config(Namespace(root=root))
            config = r76.read_json(root / 'batch_config.json')
            self.assertEqual(config['TSP'], [1024, 1024, 768, 768])
            self.assertEqual(config['ATSP'], [4096, 4096, 3072, 3072])
            self.assertFalse(r76.read_json(root / 'batch_config_receipt.json')['lr_auto_scaling'])

    def test_capacity_one_growth_and_bounded_oom_backoff(self):
        cache = Namespace(problem='TSP', size=100000)
        row = dict(cap=512, peak_allocated_gib=8., peak_reserved_gib=8., base_gib=1., oom=False,
                   step_seconds=.1)
        grown = dict(row, cap=672, peak_allocated_gib=10., peak_reserved_gib=10.)
        with patch.object(r76, 'capacity_trial', side_effect=[row, torch.cuda.OutOfMemoryError(), grown]) as trial:
            result = r76.measured_capacity(cache, {}, 512, torch.device('cuda:0'))
        self.assertEqual([call.args[2] for call in trial.call_args_list], [512, 1344, 672])
        self.assertEqual(result['cap'], 672)
        with patch.object(r76, 'capacity_trial', side_effect=torch.cuda.OutOfMemoryError()) as trial:
            with self.assertRaises(RuntimeError):
                r76.measured_capacity(cache, {}, 384, torch.device('cuda:0'))
        self.assertEqual([call.args[2] for call in trial.call_args_list], [384, 192, 96])

    def test_ddp_reused_graph_unused_branches_idle_tail_and_collective_retry(self):
        with tempfile.TemporaryDirectory() as directory:
            mp.spawn(ddp_worker, args=(str(Path(directory) / 'init'), directory), nprocs=2, join=True)
            states = [torch.load(Path(directory) / f'rank{rank}.pt', weights_only=False) for rank in range(2)]
        for state in states:
            self.assertEqual(state['steps'], 3)
            self.assertEqual(state['retries'], 1)
            self.assertEqual(state['other_retries'], [0, 0])
            self.assertTrue(state['replayed'])
            self.assertEqual(state['scale'], 4.)
        for key in states[0]['model']:
            torch.testing.assert_close(states[0]['model'][key], states[1]['model'][key], rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
