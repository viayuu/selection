"""Focused CPU witnesses for R95; no datasets, runner, GPU, or training jobs."""

import random
import unittest

import numpy as np
import torch

from code.unified_selector.registry import PROBLEMS
from .multitask_probe import task_seed
from .r76_experiment import PinnedPrefetch
from .r77_data import IndexedCache
from .r77_experiment import ShuffleStream
from .r95_sampling import (
    BATCH_SIZE, REFRESH_UPDATES, RNG_OFFSET, WARM_DRAWS,
    AdaptiveStream, WeightedCache, WeightedRows, make_proposal,
)


def inverse_rows(q, uniforms):
    cdf = np.cumsum(q, dtype=np.float64)
    cdf[-1] = 1.
    return np.searchsorted(cdf, uniforms, side='right')


class ProposalTests(unittest.TestCase):
    def test_formula_normalization_full_support_and_no_mutation(self):
        scores = np.array([0., 1., 3., 12.], dtype=np.float64)
        original = scores.copy()
        scores.setflags(write=False)
        q = make_proposal(scores)
        self.assertEqual(q.dtype, np.dtype('float64'))
        np.testing.assert_allclose(q, .5 / len(scores) + .5 * scores / scores.sum(),
            rtol=1e-15, atol=0.)
        self.assertAlmostEqual(float(q.sum()), 1., places=15)
        self.assertTrue(np.all(q >= .5 / len(scores)))
        self.assertFalse(np.shares_memory(q, scores))
        np.testing.assert_array_equal(scores, original)

    def test_all_zero_and_zero_score_fraction_are_uniform(self):
        for scores in (np.zeros(17), np.arange(17, dtype=np.float64)):
            q = make_proposal(scores, alpha=0 if scores.any() else .5)
            np.testing.assert_array_equal(q, np.full(17, 1. / 17))
        np.testing.assert_array_equal(make_proposal([0.]), [1.])
        np.testing.assert_allclose(make_proposal([0., 1., 3.], alpha=.25),
            .75 / 3 + .25 * np.array([0., 1., 3.]) / 4)

    def test_finite_extreme_scores_do_not_overflow(self):
        maximum = np.finfo(np.float64).max
        tiny = np.nextafter(0., 1.)
        with np.errstate(over='raise', invalid='raise', divide='raise'):
            for scores in ([maximum, maximum, 0.], [tiny, tiny, 0.]):
                np.testing.assert_allclose(make_proposal(scores), [5. / 12, 5. / 12, 1. / 6])

    def test_inverse_weight_expectation_is_original_empirical_objective(self):
        gradients = np.array([[1., -2.], [4., 3.], [-7., 8.], [5., -9.]])
        losses = np.array([.2, 9., 2., 15.])
        q = make_proposal(np.linalg.norm(gradients, axis=1))
        weights = 1. / (len(q) * q)
        self.assertAlmostEqual(float(np.sum(q * weights * losses)), float(losses.mean()), places=14)
        np.testing.assert_allclose(np.sum(q[:, None] * weights[:, None] * gradients, axis=0),
            gradients.mean(axis=0), rtol=1e-14, atol=1e-14)

    def test_invalid_scores_and_fractions_fail(self):
        for scores in ([], 1., [[1., 2.]], [-1., 2.], [np.nan], [np.inf]):
            with self.subTest(scores=scores), self.assertRaises(ValueError):
                make_proposal(scores)
        for alpha in (-.01, 1., 1.01, np.nan, np.inf):
            with self.subTest(alpha=alpha), self.assertRaises(ValueError):
                make_proposal([1., 2.], alpha=alpha)


class AdaptiveStreamTests(unittest.TestCase):
    def assert_rows_equal(self, first, second):
        self.assertEqual(first.source_step, second.source_step)
        torch.testing.assert_close(first.indices, second.indices, rtol=0, atol=0)
        torch.testing.assert_close(first.weights, second.weights, rtol=0, atol=0)

    @staticmethod
    def forbidden_provider(problem, source_step):
        raise AssertionError('Uniform replacement and warmup must not request proposals')

    def test_fixed_full18_global_cadence(self):
        self.assertEqual(len(PROBLEMS), 18)
        self.assertEqual(BATCH_SIZE, 1024)
        self.assertEqual(WARM_DRAWS, 409600)
        self.assertEqual(REFRESH_UPDATES, 18 * 400)

    def test_warm_parity_all18_and_resumed_tails(self):
        for problem in PROBLEMS:
            seed = task_seed(2, problem)
            for draws in (0, 77 * BATCH_SIZE, WARM_DRAWS - 2 * BATCH_SIZE):
                original = ShuffleStream(2053, seed, draws=draws)
                streams = [AdaptiveStream(2053, seed, draws=draws, arm=arm,
                    problem=problem, proposal_provider=self.forbidden_provider) for arm in ('A', 'B')]
                for _ in range(2):
                    expected = original.take(BATCH_SIZE)
                    for stream in streams:
                        batch = stream.take(BATCH_SIZE)
                        self.assertEqual((len(batch), batch.source_step), (BATCH_SIZE, 0))
                        self.assertEqual(batch.indices.dtype, torch.long)
                        self.assertEqual(batch.weights.dtype, torch.float32)
                        self.assertEqual(batch.indices.device.type, 'cpu')
                        torch.testing.assert_close(batch.indices, expected, rtol=0, atol=0)
                        torch.testing.assert_close(batch.weights, torch.ones(BATCH_SIZE))
                        self.assertEqual(stream.draws, original.draws)

    def test_entire_400_batch_warmup_is_original_stream(self):
        original = ShuffleStream(81000, 123)
        stream = AdaptiveStream(81000, 123, arm='B', problem='TSP',
            proposal_provider=self.forbidden_provider)
        for _ in range(400):
            batch = stream.take()
            torch.testing.assert_close(batch.indices, original.take(), rtol=0, atol=0)
            self.assertEqual(batch.source_step, 0)
        self.assertEqual(stream.draws, WARM_DRAWS)

    def test_replacement_retains_duplicates_and_coupled_uniforms(self):
        size, seed = 7, 29
        proposal = make_proposal(np.array([0., 0., 0., 1., 2., 5., 80.]))
        uniforms = np.random.Generator(np.random.PCG64(seed + RNG_OFFSET)).random(BATCH_SIZE)
        calls = []

        def provider(problem, source_step):
            calls.append((problem, source_step))
            return proposal

        for arm in ('A', 'B'):
            stream = AdaptiveStream(size, seed, WARM_DRAWS, arm=arm, problem='TSP',
                proposal_provider=provider if arm == 'B' else self.forbidden_provider)
            batch = stream.take()
            q = proposal if arm == 'B' else np.full(size, 1. / size)
            expected = inverse_rows(q, uniforms)
            np.testing.assert_array_equal(batch.indices.numpy(), expected)
            np.testing.assert_array_equal(batch.weights.numpy(), (1. / (size * q[expected])).astype(np.float32))
            self.assertEqual((len(batch), batch.source_step, stream.draws),
                (BATCH_SIZE, 7200, WARM_DRAWS + BATCH_SIZE))
            self.assertLess(len(torch.unique(batch.indices)), len(batch))
            if arm == 'B':
                self.assertNotAlmostEqual(float(batch.weights.mean()), 1., places=6)
                self.assertGreater(float(batch.weights.max()), 1.)
        self.assertEqual(calls, [('TSP', 7200)])

    def test_all_zero_importance_matches_uniform_replacement(self):
        streams = [AdaptiveStream(31, 7, WARM_DRAWS, arm=arm, problem='ATSP',
            proposal_provider=lambda problem, step: make_proposal(np.zeros(31))) for arm in ('A', 'B')]
        for _ in range(4):
            self.assert_rows_equal(streams[0].take(), streams[1].take())

    def test_random_access_uses_exact_pcg64_advance(self):
        size, seed = 53, 101
        q = make_proposal(np.arange(size, dtype=np.float64))
        for offset in (0, 1, 79 * BATCH_SIZE, WARM_DRAWS, 3 * WARM_DRAWS + 97):
            draws = WARM_DRAWS + offset
            calls = []

            def provider(problem, source_step):
                calls.append((problem, source_step))
                return q

            stream = AdaptiveStream(size, seed, draws, arm='B', problem='CVRP', proposal_provider=provider)
            rng = np.random.Generator(np.random.PCG64(seed + RNG_OFFSET))
            rng.bit_generator.advance(offset)
            expected = inverse_rows(q, rng.random(BATCH_SIZE))
            batch = stream.take()
            np.testing.assert_array_equal(batch.indices.numpy(), expected)
            expected_step = (draws // WARM_DRAWS) * REFRESH_UPDATES
            self.assertEqual(batch.source_step, expected_step)
            self.assertEqual(calls, [('CVRP', expected_step)])

    def test_resume_prefixes_every_1440_global_updates_all18(self):
        size = 1031

        def provider(problem, source_step):
            return make_proposal(np.roll(np.arange(size, dtype=np.float64), source_step // 7200))

        # 1440 full18 updates give exactly 80 logical batches per task.
        interval = 1440 // len(PROBLEMS)
        for arm in ('A', 'B'):
            for problem in PROBLEMS:
                seed = task_seed(2, problem)
                stream = AdaptiveStream(size, seed, arm=arm, problem=problem, proposal_provider=provider)
                reference = [stream.take() for _ in range(1203)]
                for start in range(0, 1201, interval):
                    resumed = AdaptiveStream(size, seed, start * BATCH_SIZE, arm=arm,
                        problem=problem, proposal_provider=provider)
                    for offset in range(3):
                        self.assert_rows_equal(reference[start + offset], resumed.take())
                    self.assertEqual(resumed.draws, (start + 3) * BATCH_SIZE)

    def test_source_steps_transition_for_both_arms(self):
        q = make_proposal([1., 3., 9.])
        for arm in ('A', 'B'):
            calls = []

            def provider(problem, source_step):
                calls.append((problem, source_step))
                return q

            stream = AdaptiveStream(3, 2, WARM_DRAWS - BATCH_SIZE, arm=arm,
                problem='VRPTW', proposal_provider=provider)
            self.assertEqual(stream.take().source_step, 0)
            for _ in range(400):
                self.assertEqual(stream.take().source_step, 7200)
            self.assertEqual(stream.take().source_step, 14400)
            self.assertEqual(calls, [('VRPTW', 7200), ('VRPTW', 14400)] if arm == 'B' else [])

    def test_proposal_and_cdf_are_frozen_and_provider_array_is_untouched(self):
        table = make_proposal([1., 2., 20.])
        original = table.copy()
        calls = []

        def provider(problem, source_step):
            calls.append((problem, source_step))
            return table

        stream = AdaptiveStream(3, 17, WARM_DRAWS, arm='B', problem='TSP', proposal_provider=provider)
        stream.take()
        np.testing.assert_array_equal(table, original)
        table[:] = table[::-1]
        rng = np.random.Generator(np.random.PCG64(17 + RNG_OFFSET))
        rng.bit_generator.advance(BATCH_SIZE)
        expected = inverse_rows(original, rng.random(BATCH_SIZE))
        batch = stream.take()
        np.testing.assert_array_equal(batch.indices.numpy(), expected)
        np.testing.assert_array_equal(batch.weights.numpy(), (1. / (3 * original[expected])).astype(np.float32))
        self.assertEqual(calls, [('TSP', 7200)])
        table.setflags(write=False)
        for _ in range(398):
            self.assertEqual(stream.take().source_step, 7200)
        rng.bit_generator.advance(WARM_DRAWS - 2 * BATCH_SIZE)
        expected = inverse_rows(table, rng.random(BATCH_SIZE))
        refreshed = stream.take()
        self.assertEqual(refreshed.source_step, 14400)
        np.testing.assert_array_equal(refreshed.indices.numpy(), expected)
        np.testing.assert_array_equal(refreshed.weights.numpy(), (1. / (3 * table[expected])).astype(np.float32))
        self.assertEqual(calls, [('TSP', 7200), ('TSP', 14400)])
        np.testing.assert_array_equal(table, original[::-1])

    def test_only_1024_and_no_crossing_rejection_is_atomic(self):
        for draws in (0, WARM_DRAWS, 2 * WARM_DRAWS):
            stream = AdaptiveStream(5, 2, draws, arm='A', problem='TSP', proposal_provider=None)
            for count in (0, 1, 1023, 1025, 2048, 1024.):
                with self.subTest(draws=draws, count=count), self.assertRaises((ValueError, TypeError)):
                    stream.take(count)
                self.assertEqual(stream.draws, draws)
            fresh = AdaptiveStream(5, 2, draws, arm='A', problem='TSP', proposal_provider=None)
            self.assert_rows_equal(stream.take(), fresh.take())
        for draws in (WARM_DRAWS - 1, WARM_DRAWS - 512, 2 * WARM_DRAWS - 512):
            stream = AdaptiveStream(5, 2, draws, arm='B', problem='TSP',
                proposal_provider=self.forbidden_provider)
            with self.assertRaises(ValueError):
                stream.take()
            self.assertEqual(stream.draws, draws)

    def test_invalid_proposals_do_not_consume_draws_or_mutate_arrays(self):
        for values in ([.5], [[.5, .5]], [0., 1.], [-1., 2.], [1., 1.], [np.nan, .5], [np.inf, .5]):
            table = np.array(values, dtype=np.float64)
            original = table.copy()
            stream = AdaptiveStream(2, 2, WARM_DRAWS, arm='B', problem='TSP',
                proposal_provider=lambda problem, step: table)
            with self.subTest(values=values), self.assertRaises(ValueError):
                stream.take()
            self.assertEqual(stream.draws, WARM_DRAWS)
            np.testing.assert_array_equal(table, original)
            stream.proposal_provider = lambda problem, step: np.array([.25, .75])
            fresh = AdaptiveStream(2, 2, WARM_DRAWS, arm='B', problem='TSP',
                proposal_provider=stream.proposal_provider)
            self.assert_rows_equal(stream.take(), fresh.take())

    def test_bad_stream_arguments_fail(self):
        for size, seed, draws in ((0, 2, 0), (2, -1, 0), (2, 2, -1), (2.5, 2, 0), (2, 2, 1.5)):
            with self.subTest(size=size, seed=seed, draws=draws), self.assertRaises((ValueError, TypeError)):
                AdaptiveStream(size, seed, draws, arm='A', problem='TSP', proposal_provider=None)
        with self.assertRaises(ValueError):
            AdaptiveStream(2, 2, arm='C', problem='TSP', proposal_provider=None)
        with self.assertRaises(TypeError):
            AdaptiveStream(2, 2, arm='B', problem='TSP', proposal_provider=None)

    def test_no_global_random_state_changes(self):
        python_state, numpy_state = random.getstate(), np.random.get_state()
        torch_state = torch.random.get_rng_state().clone()
        for arm in ('A', 'B'):
            for draws in (0, WARM_DRAWS, 3 * WARM_DRAWS + BATCH_SIZE):
                stream = AdaptiveStream(31, 91, draws, arm=arm, problem='TSP',
                    proposal_provider=lambda problem, step: make_proposal(np.arange(31)))
                stream.take()
        self.assertEqual(python_state, random.getstate())
        current = np.random.get_state()
        self.assertEqual(numpy_state[0], current[0])
        np.testing.assert_array_equal(numpy_state[1], current[1])
        self.assertEqual(numpy_state[2:], current[2:])
        torch.testing.assert_close(torch_state, torch.random.get_rng_state(), rtol=0, atol=0)


class FakeCache:
    def __init__(self):
        self.size, self.problem, self.split, self.kind = 7, 'TSP', 'train', 'coord'
        self.pool, self.pool_ids, self.metadata = ['a', 'b'], [0, 1], {'sentinel': 91}
        self.lengths = np.arange(7, dtype=np.int64) + 2
        self.raw = dict(costs=np.arange(14, dtype=np.float64).reshape(7, 2) + 1,
            winner=np.zeros(7, dtype=np.int64))
        self.x = torch.arange(21, dtype=torch.float32).reshape(7, 3)
        self.closed = False

    def batch(self, rows, device='cpu'):
        self.seen, self.device = rows, device
        self.last_batch = dict(x=self.x[rows].to(device),
            ind=torch.tensor(self.raw['winner'][rows], dtype=torch.long, device=device))
        return self.last_batch

    def close(self):
        self.closed = True
        return 'closed'


class WeightedCacheTests(unittest.TestCase):
    def test_weighted_original_gather_keeps_duplicates_and_local_row_identity(self):
        source = FakeCache()
        original_costs, original_x = source.raw['costs'].copy(), source.x.clone()
        base = IndexedCache(source, np.array([1, 3, 6]))
        cache = WeightedCache(base)
        rows = WeightedRows(torch.tensor([2, 0, 2], dtype=torch.long),
            torch.tensor([.25, 2., .25], dtype=torch.float32), 7200)
        batch = cache.batch(rows, device=torch.device('cpu'))
        np.testing.assert_array_equal(source.seen, [6, 1, 6])
        torch.testing.assert_close(batch['x'], source.x[[6, 1, 6]])
        torch.testing.assert_close(batch['sampling_row'], rows.indices, rtol=0, atol=0)
        torch.testing.assert_close(batch['sampling_weight'], rows.weights, rtol=0, atol=0)
        self.assertEqual((batch['sampling_row'].dtype, batch['sampling_weight'].dtype),
            (torch.long, torch.float32))
        self.assertEqual(batch['sampling_weight'].device, torch.device('cpu'))
        self.assertEqual(batch['sampling_row'].device, torch.device('cpu'))
        self.assertNotIn('sampling_weight', source.last_batch)
        self.assertNotIn('sampling_row', source.last_batch)
        np.testing.assert_array_equal(source.raw['costs'], original_costs)
        torch.testing.assert_close(source.x, original_x)
        torch.testing.assert_close(rows.indices, torch.tensor([2, 0, 2]))
        torch.testing.assert_close(rows.weights, torch.tensor([.25, 2., .25]))

    def test_ordinary_tensor_is_exact_passthrough_without_extras(self):
        base = FakeCache()
        cache = WeightedCache(base)
        rows = torch.tensor([4, 0, 4])
        result = cache.batch(rows, device='cpu')
        self.assertIs(result, base.last_batch)
        self.assertIs(base.seen, rows)
        self.assertEqual(set(result), {'x', 'ind'})
        self.assertEqual(base.device, 'cpu')

    def test_existing_cpu_prefetch_accepts_weighted_and_ordinary_rows(self):
        base = FakeCache()
        cache = WeightedCache(base)
        weighted = WeightedRows(torch.tensor([5, 1, 5]), torch.tensor([.5, 1.5, .5]), 7200)
        ordinary = torch.tensor([4, 2])
        requests = [('TSP', weighted, len(weighted)), ('TSP', ordinary, len(ordinary))]
        with PinnedPrefetch({'TSP': cache}, requests, torch.device('cpu')) as batches:
            results = list(batches)
        self.assertEqual([result[:3] for result in results], [('TSP', 3, 3), ('TSP', 2, 2)])
        first, second = results[0][3], results[1][3]
        torch.testing.assert_close(first['x'], base.x[[5, 1, 5]])
        torch.testing.assert_close(first['sampling_row'], weighted.indices)
        torch.testing.assert_close(first['sampling_weight'], weighted.weights)
        torch.testing.assert_close(second['x'], base.x[[4, 2]])
        self.assertEqual(set(second), {'x', 'ind'})

    def test_attributes_methods_and_close_are_proxied(self):
        base = FakeCache()
        cache = WeightedCache(base)
        for name in ('size', 'problem', 'split', 'kind', 'pool', 'pool_ids', 'metadata', 'raw', 'lengths'):
            self.assertIs(getattr(cache, name), getattr(base, name))
        base.extra_attribute = object()
        self.assertIs(cache.extra_attribute, base.extra_attribute)
        self.assertFalse(cache.closed)
        self.assertEqual(cache.close(), 'closed')
        self.assertTrue(cache.closed)


if __name__ == '__main__':
    unittest.main()
