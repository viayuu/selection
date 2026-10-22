"""Dummy CPU classifiers only: isolation, pair lifetimes and fixed Borda."""

import gc
import inspect
import unittest
from unittest.mock import Mock
import weakref

import numpy as np

from . import r100_model as model
from .test_r100_data import tables


class PairwiseTests(unittest.TestCase):
    def test_all_strict_rows_even_third_winners_one_model_live_and_class_mapping(self):
        context, relations = tables()
        refs, fits, queries = [], [], []
        test = self
        class Classifier:
            classes_ = np.array([1, 0], dtype=np.int64)
            def fit(self, x, y, *, kv_cache):
                test.assertIs(kv_cache, False)
                fits.append((x.copy(), y.copy()))
                x[:] = -999
                y[:] = 9
                return self
            def predict_proba(self, x):
                queries.append(x.copy())
                x[:] = -88
                return np.tile([.2, .8], (len(x), 1))
        def factory():
            gc.collect()
            self.assertTrue(all(ref() is None for ref in refs))
            classifier = Classifier()
            refs.append(weakref.ref(classifier))
            return classifier
        instance = model.FrozenPairwiseContext(context, relations, factory)
        query = np.array([[5., 6.], [7., 8.]], dtype=np.float32)
        p = instance.predict_features(query, [900, 901])
        self.assertEqual(len(fits), 3)
        for pair, (x, y) in enumerate(fits):
            valid = relations.labels[:, pair] >= 0
            np.testing.assert_array_equal(x, context.x[valid])
            np.testing.assert_array_equal(y, relations.labels[valid, pair])
            self.assertEqual(y.dtype, np.int64)
        for x in queries:
            np.testing.assert_array_equal(x, query)
        for i, j in instance.pairs:
            np.testing.assert_array_equal(p[:, i, j], [.8, .8])
            np.testing.assert_array_equal(p[:, j, i], [1.-.8, 1.-.8])
        self.assertTrue(all(ref() is None for ref in refs))
        self.assertEqual([r['stage'] for r in instance.call_records], ['fit_pair', 'predict_pair']*3)

    def test_reject_full_train_baseid_before_filter_even_all_ties(self):
        context, relations = tables(np.ones((6, 3), dtype=np.float64))
        factory = Mock(side_effect=AssertionError('No classifier before source isolation'))
        instance = model.FrozenPairwiseContext(context, relations, factory)
        with self.assertRaisesRegex(ValueError, 'full TRAIN'):
            instance.predict_features(np.ones((1, 2)), [100])
        p = instance.predict_features(np.ones((1, 2)), [900])
        np.testing.assert_array_equal(p, np.full((1, 3, 3), .5))
        np.testing.assert_array_equal(instance.predict(np.ones((1, 2)), [900]), [1])
        factory.assert_not_called()

    def test_one_class_only_shortcut_and_near_oneclass_not_skipped(self):
        context, relations = tables(np.tile([1., 2., 3.], (6, 1)))
        factory = Mock(side_effect=AssertionError('Only exact constants skip'))
        p = model.FrozenPairwiseContext(context, relations, factory).predict_features(np.ones((1, 2)), [900])
        self.assertEqual(p[0, 0, 1], 1.)
        self.assertEqual(p[0, 1, 0], 0.)
        costs = np.tile([1., 2., 3.], (81000, 1))
        costs[-1, 1] = 1.-2.**-40
        context, relations = tables(costs)
        classifier = Mock(classes_=np.array([0, 1]))
        classifier.predict_proba.return_value = np.array([[.4, .6]])
        instance = model.FrozenPairwiseContext(context, relations, lambda: classifier)
        instance.predict_features(np.ones((1, 2)), [999999])
        classifier.fit.assert_called_once()
        self.assertEqual(classifier.fit.call_args.args[0].shape, (81000, 2))

    def test_no_current_query_outcome_api_and_invalid_inputs_fail_before_fit(self):
        context, relations = tables()
        factory = Mock(side_effect=AssertionError('No classifier'))
        instance = model.FrozenPairwiseContext(context, relations, factory)
        self.assertEqual(list(inspect.signature(instance.predict_features).parameters), ['query_features', 'query_source_ids'])
        for key in ('costs', 'winner', 'labels'):
            with self.assertRaises(TypeError):
                instance.predict_features(np.ones((1, 2)), [900], **{key: [0]})
        for x, ids in ((np.ones((1, 3)), [900]), (np.ones((1, 2)), [900.]),
                       (np.ones((1, 2)), [-1]), (np.full((1, 2), np.nan), [900])):
            with self.assertRaises(ValueError):
                instance.predict_features(x, ids)
        factory.assert_not_called()

    def test_fixed_borda_not_coupling_and_exact_globalid_ties(self):
        p = np.array([[[.5, .9, .2], [.1, .5, .8], [.8, .2, .5]]], dtype=np.float64)
        # Set exact complements, not rounded decimal approximations.
        p[:, 1, 0], p[:, 2, 0], p[:, 2, 1] = 1.-p[:, 0, 1], 1.-p[:, 0, 2], 1.-p[:, 1, 2]
        score = model.mean_opponent_win(p)
        np.testing.assert_array_equal(score, [[.55, .45, .5]])
        np.testing.assert_array_equal(model.stable_mean_win_rank(score, [7, 0, 2]), [[0, 2, 1]])
        score = np.array([[.5, .5, .5], [.5, np.nextafter(.5, 1.), .5]])
        np.testing.assert_array_equal(model.stable_mean_win_rank(score, [7, 0, 2]), [[1, 2, 0], [1, 2, 0]])
        logits, formal = model.formal_score_view(score)
        np.testing.assert_allclose(np.exp(logits), formal)
        np.testing.assert_allclose(formal.sum(1), 1.)
        self.assertFalse(np.array_equal(formal, score))


if __name__ == '__main__':
    unittest.main()
