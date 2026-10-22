"""Focused context-isolation and candidate-alignment tests; no public download."""

from types import SimpleNamespace
import unittest

import numpy as np

from .r97_model import FrozenContextClassifier, expand_probabilities


def context(labels=(0, 2)):
    return SimpleNamespace(x=np.ones((len(labels), 2), dtype=np.float32),
        y=np.asarray(labels), source_ids=np.arange(len(labels)) + 100,
        kept_columns=np.asarray([0, 4]), pool=('a', 'b', 'c'), pool_ids=(0, 7, 19))


class FakeClassifier:
    def fit(self, x, y):
        self.x, self.y = x.copy(), y.copy()
        self.classes_ = np.asarray([2, 0])
        return self

    def predict_proba(self, x):
        return np.tile([.7, .3], (len(x), 1))


class ContextModelTests(unittest.TestCase):
    def test_historical_labels_only_and_class_alignment(self):
        model = FrozenContextClassifier(context(), FakeClassifier).fit()
        scores = model.predict_features(np.asarray([[2., 3.]]), np.asarray([900]))
        np.testing.assert_array_equal(model.classifier.y, [0, 2])
        np.testing.assert_allclose(scores, [[.3, 0., .7]])
        self.assertEqual(model.predict(np.asarray([[2., 3.]]), [900]).item(), 2)

    def test_self_or_related_base_query_rejected(self):
        model = FrozenContextClassifier(context(), FakeClassifier).fit()
        with self.assertRaisesRegex(ValueError, 'own context'):
            model.predict_features(np.zeros((1, 2)), [100])

    def test_no_query_supervision_parameter(self):
        model = FrozenContextClassifier(context(), FakeClassifier).fit()
        with self.assertRaises(TypeError):
            model.predict_features(np.zeros((1, 2)), [900], winner=[0])

    def test_degenerate_historical_class_and_lowest_global_tie(self):
        model = FrozenContextClassifier(context((2, 2)), FakeClassifier).fit()
        self.assertIsNone(model.classifier)
        self.assertEqual(model.predict(np.zeros((1, 2)), [900]).item(), 2)
        scores = expand_probabilities([[.5, .5]], [2, 0], 3)
        self.assertEqual(scores.argmax(1).item(), 0)

    def test_invalid_local_class_cannot_select_removed_candidate(self):
        with self.assertRaises(ValueError):
            expand_probabilities([[.5, .5]], [0, 7], 3)
        with self.assertRaises(ValueError):
            expand_probabilities([[1., -1.]], [0, 2], 3)


if __name__ == '__main__':
    unittest.main()
