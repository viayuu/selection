"""Focused R78 mathematical, split, identity, mask and one-shot TEST checks."""

from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from .r76_probability_diagnostics import prepare_prediction
from .r78_zero_selection import (FIT, align_bias, batches, fit_intercepts, metrics,
                                 objective, test_once_guard, val_gate, write_json,
                                 SQUARED_FIT, PROBLEMS, fit_cost_intercepts, progress,
                                 run_round002, squared_inputs, squared_snapshot, svrg_direction)


def example(split='train'):
    return prepare_prediction(dict(problem='TSP', split=split,
                                   logits=np.array([[.5, -.5], [-.5, .5], [.1, -.1]], dtype=np.float32),
                                   costs=np.array([[1., 2.], [2., 1.], [1., 3.]], dtype=np.float64),
                                   winner=np.array([0, 1, 0]), indices=np.arange(3), nodes=np.full(3, 50),
                                   pool=np.array(['A', 'B']), pool_ids=np.array([4, 1])))


class InterceptTests(unittest.TestCase):
    def test_full_objective_gradient_and_unit_prior(self):
        value = example()
        b = np.array([.2, -.2])
        checked = objective(value['logits'], value['winner'], b)
        self.assertAlmostEqual(checked['penalty'], .04 / 3)
        direction = np.array([1., -1.])
        eps = 1e-6
        numeric = (objective(value['logits'], value['winner'], b + eps * direction)['objective']
                   - objective(value['logits'], value['winner'], b - eps * direction)['objective']) / (2 * eps)
        self.assertAlmostEqual(numeric, float(checked['gradient'] @ direction), places=7)

    def test_exact_1024_and_no_dropped_tail(self):
        stream = batches(1031)
        first, second = next(stream), next(stream)
        self.assertEqual(len(first), 1024)
        self.assertEqual(len(second), 1024)
        np.testing.assert_array_equal(np.sort(np.concatenate([first, second[:7]])), np.arange(1031))
        np.testing.assert_array_equal(first, next(batches(1031)))

    def test_train_only_optimizer_and_fixed_batch(self):
        with self.assertRaisesRegex(ValueError, 'Only TRAIN'):
            fit_intercepts(example('val'))
        with self.assertRaisesRegex(ValueError, 'fixed'):
            fit_intercepts(example(), dict(FIT, batch_size=512))

    def test_real_fit_reduces_natural_ce_and_keeps_sum_zero(self):
        value = example()
        initial = objective(value['logits'], value['winner'], np.zeros(2))['objective']
        fitted = fit_intercepts(value, dict(FIT, max_updates=100, check_every=20))
        self.assertLess(fitted['final']['objective'], initial)
        self.assertAlmostEqual(sum(fitted['bias']), 0., places=12)
        self.assertEqual(fitted['optimizer_examples'], fitted['successful_updates'] * 1024)
        self.assertFalse(fitted['converged'])

    def test_identity_exact_and_named_alignment(self):
        value = example()
        m, chosen, _ = metrics(value, np.zeros(2), 'A')
        np.testing.assert_array_equal(chosen, value['selected'])
        self.assertEqual(m['top1'], value['correct'].mean())
        self.assertEqual(m['mean_cost'], value['selected_cost'].mean())
        fitted = dict(pool=['B', 'A'], pool_ids=[1, 4], bias=[-.2, .2])
        np.testing.assert_array_equal(align_bias(value, fitted), [.2, -.2])
        with self.assertRaises(ValueError):
            align_bias(value, dict(fitted, pool_ids=[7, 4]))

    def test_mask_and_global_id_tie(self):
        value = example()
        value['logits'][0] = 0.
        value['logits'][2, 1] = -np.inf
        value = prepare_prediction(value)
        _, chosen, _ = metrics(value, np.zeros(2), 'A')
        self.assertEqual(chosen[0], 1)
        self.assertEqual(chosen[2], 0)

    def test_paired_regret_respects_masked_oracle(self):
        value = example()
        value['logits'][2, 1] = -np.inf
        value['costs'][2, 1] = .5
        value = prepare_prediction(value)
        result, _, samples = metrics(value, np.zeros(2), 'A')
        self.assertEqual(samples['actual_regret_pct'][2], 0.)
        self.assertAlmostEqual(samples['actual_regret_pct'].mean(), result['actual_regret_pct'])

    def test_strict_gate_no_macro_vs_substitution(self):
        old = dict(macro_top1=.6, macro_mean_cost=2., macro_actual_regret_pct=.2, macro_vs_sbs_pct=-1.)
        new = dict(old, macro_top1=.61, macro_vs_sbs_pct=1.)
        self.assertTrue(val_gate(old, new, True)['accepted'])
        self.assertFalse(val_gate(old, old, True)['accepted'])
        self.assertFalse(val_gate(old, dict(new, macro_mean_cost=2.0000000001), True)['accepted'])
        self.assertFalse(val_gate(old, dict(new, macro_actual_regret_pct=.2000000001), True)['accepted'])
        self.assertEqual(val_gate(old, new, False)['status'], 'inconclusive')

    def test_rejected_candidate_never_opens_test_and_acceptance_is_once(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            self.assertFalse(test_once_guard(root, dict(accepted=False, status='negative'), {}))
            self.assertFalse((root / 'test_started.json').exists())
            self.assertFalse((root / 'formal_candidate_lock.json').exists())
            write_json(root / 'train_locked_candidate.json', dict(bias=[.1, -.1]))
            self.assertTrue(test_once_guard(root, dict(accepted=True), dict(bias=[.1, -.1])))
            with self.assertRaises(FileExistsError):
                test_once_guard(root, dict(accepted=True), dict(bias=[.9, -.9]))
            from .r76_probability_diagnostics import read_json
            self.assertEqual(read_json(root / 'formal_candidate_lock.json')['bias'], [.1, -.1])


class SquaredInterceptTests(unittest.TestCase):
    def test_centered_constraint_gradient_and_quadratic_reference(self):
        data = squared_inputs(example())
        b = np.array([.2, -.2])
        np.testing.assert_allclose((data['h'] + b).sum(1), 0., atol=1e-15)
        self.assertFalse(np.allclose(data['h'] + b, -1.))
        checked = squared_snapshot(data, b)
        direct = np.mean(data['w'] * (1 + data['h'] + b) ** 2) + b @ b / (2 * data['n'])
        self.assertAlmostEqual(checked['objective'], direct)
        direction = np.array([1., -1.])
        eps = 1e-6
        derivative = (squared_snapshot(data, b + eps * direction)['objective']
                      - squared_snapshot(data, b - eps * direction)['objective']) / (2 * eps)
        self.assertAlmostEqual(derivative, float(checked['gradient'] @ direction), places=7)
        reference = squared_snapshot(data, data['reference'])
        self.assertLess(reference['gradient_inf'], 1e-12)
        self.assertAlmostEqual(checked['objective'] - reference['objective'], checked['exact_objective_gap'])

    def test_svrg_is_exact_batch_gradient_difference_plus_full_snapshot(self):
        data = squared_inputs(example())
        b, anchor = np.array([.2, -.2]), np.array([-.1, .1])
        indices = np.random.default_rng(2).integers(data['n'], size=1024)
        full = squared_snapshot(data, anchor)['gradient']
        def batch_gradient(offsets):
            return (2 / data['k'] * (data['w'][indices] *
                    (1 + data['h'][indices] + offsets)).mean(0) + offsets / data['n'])
        np.testing.assert_allclose(svrg_direction(data, b, anchor, full, indices),
                                   batch_gradient(b) - batch_gradient(anchor) + full, atol=1e-14)

    def test_first_update_is_zero_initialized_not_closed_form_substitution(self):
        value = example()
        data = squared_inputs(value)
        initial = squared_snapshot(data, np.zeros(2))
        expected = -data['eta'] * initial['gradient']
        expected -= expected.mean()
        fitted = fit_cost_intercepts(value, dict(SQUARED_FIT, max_updates=1, check_every=1))
        np.testing.assert_allclose(fitted['bias'], expected)
        self.assertFalse(np.allclose(fitted['bias'], data['reference']))
        self.assertEqual(fitted['optimizer_examples'], 1024)
        self.assertEqual(fitted['full_gradient_snapshot_passes'], 2)
        self.assertEqual(fitted['snapshot_rows_read'], 6)
        self.assertFalse(fitted['history'][0]['qualifies'])

    def test_minibatch_training_reaches_declared_reference_checks(self):
        fitted = fit_cost_intercepts(example(), dict(SQUARED_FIT, max_updates=1000, check_every=100))
        self.assertTrue(fitted['converged'])
        self.assertLessEqual(fitted['final']['reference_error_inf'], 1e-6)
        self.assertTrue(all(row['qualifies'] for row in fitted['history'][-2:]))
        self.assertAlmostEqual(sum(fitted['bias']), 0., places=14)
        self.assertEqual(fitted['optimizer_examples'], fitted['successful_updates'] * 1024)

    def test_true_gap_constructed_in_fp64_before_normalization(self):
        value = example()
        value['costs'] = np.ones((3, 2), dtype=np.float64)
        value['costs'][0, 1] = np.nextafter(1., 2.)
        data = squared_inputs(value)
        self.assertGreater(data['c'], 0.)
        self.assertAlmostEqual(data['w'][0, 1], 6.)
        self.assertEqual(data['w'].dtype, np.float64)

    def test_zero_regret_has_identity_unique_prior_solution(self):
        value = example()
        value['costs'] = np.ones((3, 2), dtype=np.float64)
        data = squared_inputs(value)
        self.assertEqual(data['c'], 1.)
        fitted = fit_cost_intercepts(value, dict(SQUARED_FIT, max_updates=100, check_every=10))
        np.testing.assert_array_equal(fitted['bias'], [0., 0.])
        self.assertTrue(fitted['converged'])
        self.assertEqual(fitted['successful_updates'], 20)

    def test_no_val_fitting_nonfinite_repair_or_nonpositive_cost_drop(self):
        with self.assertRaisesRegex(ValueError, 'Only TRAIN'):
            squared_inputs(example('val'))
        value = example()
        value['costs'][0, 0] = 0.
        with self.assertRaisesRegex(ValueError, 'positive'):
            squared_inputs(value)
        value = example()
        value['logits'][0, 1] = -np.inf
        with self.assertRaisesRegex(ValueError, 'finite complete'):
            squared_inputs(value)

    def test_round002_failed_training_never_reads_val_or_test(self):
        train = {p: dict(example(), problem=p) for p in PROBLEMS}
        sha = '5ba5777e437235e22682c4d54cfd8cdefca6fb53c27204a8e1e3f753ea9043c9'
        def metadata(path):
            self.assertFalse('val' in Path(path).parts or 'test' in Path(path).parts)
            if Path(path).name == 'training_complete.json':
                return {'selected_best': dict(epoch=18, checkpoint_sha256=sha, checkpoint='fixture')}
            return {'per_problem': {p: {'sbs_name': 'A'} for p in PROBLEMS}}
        def fit(value, **kwargs):
            return dict(problem=value['problem'], pool=['A', 'B'], pool_ids=[4, 1],
                        bias=[0., 0.], converged=False, status='inconclusive_budget',
                        successful_updates=0, history=[], final=None)
        with tempfile.TemporaryDirectory() as folder, \
                mock.patch('code.V4.r78_zero_selection.read_json', side_effect=metadata), \
                mock.patch('code.V4.r78_zero_selection.file_hash', return_value=sha), \
                mock.patch('code.V4.r78_zero_selection.load_shards', return_value=(train, [])) as loader, \
                mock.patch('code.V4.r78_zero_selection.fit_cost_intercepts', side_effect=fit), \
                mock.patch('code.V4.r78_zero_selection.figures'):
            root = Path(folder) / 'round002'
            decision = run_round002(Path(folder) / 'source', root, 'disabled')
            self.assertFalse(decision['val_evaluated'])
            self.assertFalse(decision['test_opened'])
            self.assertEqual(loader.call_count, 1)
            self.assertFalse((root / 'train_locked_candidate.json').exists())
            self.assertFalse((root / 'val_comparison.json').exists())

    def test_round002_progress_preserves_round001_and_old_goal(self):
        from .r76_probability_diagnostics import read_json
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder) / 'round002'
            status = dict(file_based_goal=dict(status='active', round=1),
                          app_goal=dict(status='paused'), round001=dict(outcome='inconclusive'))
            write_json(root.parent / 'STATUS.json', status)
            progress(root, 'fit_round2', round_number=2, pid=123)
            updated = read_json(root.parent / 'STATUS.json')
            self.assertEqual(updated['round001'], status['round001'])
            self.assertEqual(updated['app_goal'], status['app_goal'])
            self.assertEqual(updated['file_based_goal']['round'], 2)


if __name__ == '__main__':
    unittest.main()
