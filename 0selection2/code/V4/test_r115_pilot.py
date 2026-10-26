import unittest

import numpy as np

from .r115_pilot import centered_response, make_folds, ridge_fit_predict


class ResponsePilotTests(unittest.TestCase):
    def test_base_folds_share_all_copies_and_balance(self):
        rows = [dict(problem="CVRP", base_id=f"b{i}", input_check=dict(n=50 + i // 2))
                for i in range(32) for _ in range(4)]
        folds = make_folds(rows)
        self.assertEqual(len(folds), 32)
        self.assertEqual(sum(folds.values()), 16)
        self.assertEqual(folds, make_folds(rows[::-1]))

    def test_response_cancels_common_cost_change(self):
        before, after = np.array([10., 12., 13.]), np.array([11., 10., 15.])
        np.testing.assert_allclose(centered_response(before, after, 10),
                                   centered_response(before, after + 100, 10), atol=1e-14)
        self.assertAlmostEqual(float(centered_response(before, after, 10).sum()), 0.)

    def test_ridge_queries_do_not_change_fit(self):
        x = np.arange(12.).reshape(4, 3)
        y = np.stack([np.arange(4.), -np.arange(4.)], 1)
        a, _ = ridge_fit_predict(x, y, x[:1])
        b, _ = ridge_fit_predict(x, y, np.concatenate([x[:1], x * 1e6]))
        np.testing.assert_allclose(a, b[:1], atol=1e-10)


if __name__ == "__main__":
    unittest.main()
