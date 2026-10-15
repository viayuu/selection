"""Only the two necessary synthetic fixed-mixture checks."""

import unittest
import numpy as np

from .r78_ab_equal_weight_screen import mixture_nll, probability_average


class EqualProbabilityScreenChecks(unittest.TestCase):
    def test_FP32_mask_and_global_ID_exact_tie(self):
        mask = np.array([[True, True, False], [True, False, False]])
        a = np.array([[2., 0., -np.inf], [0., -np.inf, -np.inf]], dtype=np.float32)
        b = np.array([[0., 2., -np.inf], [0., -np.inf, -np.inf]], dtype=np.float32)
        p = probability_average(a, b, mask)
        self.assertEqual(p.dtype, np.float32)
        np.testing.assert_array_equal(p[~mask], np.zeros((~mask).sum(), dtype=np.float32))
        ids = np.array([17, 3, 9])
        canonical = np.argsort(ids, kind='stable')
        chosen = canonical[np.argmax(np.where(mask, p, -np.inf)[:, canonical], axis=1)]
        np.testing.assert_array_equal(chosen, [1, 0])

    def test_normalized_mixture_descriptive_CE(self):
        p = np.array([[.2, .3], [.1, .9]], dtype=np.float32)
        winner = np.array([1, 0])
        expected = -np.log(p.astype(np.float64)[np.arange(2), winner] / p.sum(1, dtype=np.float64))
        np.testing.assert_array_equal(mixture_nll(p, winner), expected)


if __name__ == '__main__':
    unittest.main()
