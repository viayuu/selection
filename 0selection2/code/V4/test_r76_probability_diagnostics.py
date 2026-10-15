"""Small CPU-only regression tests for saved-prediction diagnostics."""

import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from . import r76_probability_diagnostics as diagnostics


def fixture():
    return dict(problem="TSP", split="test", pool=np.array(["B", "A", "C"]),
                pool_ids=np.array([20, 10, 30]), indices=np.arange(3), nodes=np.array([50, 60, 70]),
                logits=np.array([[1, 1, -2], [4, 3, -2], [2, 1, 0]], dtype=np.float32),
                costs=np.array([[2, 1, 3], [4, 3, 5], [1, 2, 3]], dtype=np.float64),
                winner=np.array([1, 1, 0]))


class ProbabilityDiagnosticsTests(unittest.TestCase):
    def test_fp32_masked_softmax_matches_cpu_torch(self):
        scores = np.array([[1000, 999, -1000], [-1000, 0, 10]], dtype=np.float64)
        mask = np.array([[True, True, False], [True, True, True]])
        p, logp = diagnostics.masked_softmax_fp32(scores, mask)
        tensor = torch.tensor(scores, dtype=torch.float32).masked_fill(~torch.from_numpy(mask), -torch.inf)
        np.testing.assert_allclose(p, torch.softmax(tensor, -1).numpy(), rtol=2e-7, atol=2e-8)
        np.testing.assert_allclose(logp, torch.log_softmax(tensor, -1).numpy(), rtol=2e-7, atol=2e-8)
        self.assertEqual(p.dtype, np.float32)
        self.assertEqual(p[0, 2], 0)
        self.assertTrue(np.isneginf(logp[0, 2]))

    def test_invalid_softmax_inputs_fail(self):
        for scores, mask in (([[0, 1]], [[False, False]]), ([[np.nan, 1]], [[True, True]])):
            with self.assertRaises(ValueError):
                diagnostics.masked_softmax_fp32(scores, mask)

    def test_score_ties_use_global_id_and_ignore_softmax_rounding(self):
        value = diagnostics.prepare_prediction(fixture())
        np.testing.assert_array_equal(value["selected"], [1, 0, 0])
        np.testing.assert_array_equal(value["ranks"][0], [2, 1, 3])
        scores = np.array([[0, 1e-8]], dtype=np.float32)
        p, _ = diagnostics.masked_softmax_fp32(scores, np.ones_like(scores, dtype=bool))
        np.testing.assert_array_equal(p, [[.5, .5]])
        selected, _ = diagnostics.score_ranks(scores, [0, 1], np.ones_like(scores, dtype=bool))
        self.assertEqual(selected[0], 1)

    def test_native_label_and_fp64_costs_are_preserved(self):
        raw = fixture()
        raw["winner"][1] = 0
        value = diagnostics.prepare_prediction(raw)
        self.assertEqual(value["winner"][1], 0)
        self.assertEqual(value["costs"].dtype, np.float64)
        self.assertEqual(value["winner_cost"][1], 4)
        self.assertEqual(value["oracle_cost"][1], 3)
        self.assertEqual(value["costloss"][1], 0)
        self.assertEqual(value["oracle_costloss"][1], 1)

    def test_missed_conditioning_not_global_selection_frequency(self):
        value = diagnostics.prepare_prediction(fixture())
        summaries, candidates, flows = diagnostics.method_diagnostics(value, near_zero_pct=40)
        a = summaries[1]
        self.assertEqual(a["true_winner_count"], 2)
        self.assertEqual(a["selected_count"], 1)
        self.assertEqual(a["missed_winner_count"], 1)
        self.assertTrue(a["near_zero_selected_with_winner"])
        self.assertAlmostEqual(a["missed_winner_probability_mean"], value["probabilities"][1, 1])
        self.assertAlmostEqual(a["winner_probability_mean"], value["probabilities"][:2, 1].mean(), places=7)
        self.assertEqual(a["costloss_sum"], 1)
        self.assertEqual(a["problem_costloss_share_pct"], 100)
        flow = next(r for r in flows if r["true_winner"] == "A" and r["selected_solver"] == "B")
        self.assertEqual(flow["selected_count"], 1)
        self.assertEqual(flow["selected_pct_of_missed"], 100)
        candidate = next(r for r in candidates if r["true_winner"] == "A" and r["candidate"] == "C")
        self.assertEqual(candidate["rank_mean"], 3)
        self.assertIsNone(summaries[2]["winner_probability_mean"])

    def test_fixed_calibration_bins_and_boundaries(self):
        bins, ece = diagnostics.calibration_bins(np.array([.8, .6]), np.array([1, 0]))
        self.assertEqual(sum(r["count"] for r in bins), 2)
        self.assertAlmostEqual(ece, .4)
        bins, ece = diagnostics.calibration_bins(np.array([0., 1., .05]), np.array([0, 1, 0]))
        self.assertEqual(bins[0]["count"], 1)
        self.assertEqual(bins[1]["count"], 1)
        self.assertEqual(bins[-1]["count"], 1)
        self.assertAlmostEqual(ece, .05 / 3)

    def test_name_alignment_not_column_position(self):
        aligned = diagnostics.align_by_name(["C", "A", "B"], ["A", "B", "C"], [1, 2, 3])
        np.testing.assert_array_equal(aligned, [3, 1, 2])
        with self.assertRaises(ValueError):
            diagnostics.align_by_name(["A", "B"], ["A", "C"], [1, 2])

    def test_duplicate_or_missing_indices_rejected(self):
        raw = fixture()
        raw["indices"] = np.array([0, 0, 2])
        with self.assertRaises(ValueError):
            diagnostics.prepare_prediction(raw)

    def test_detailed_csv_contains_all_candidates_for_each_miss(self):
        value = diagnostics.prepare_prediction(fixture())
        summaries, _, _ = diagnostics.method_diagnostics(value, .1)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "tables").mkdir()
            sample_path, candidate_path = root / "samples.csv", root / "candidates.csv"
            with sample_path.open("w", newline="") as sample, candidate_path.open("w", newline="") as candidate:
                sw = csv.DictWriter(sample, fieldnames=diagnostics.SAMPLE_FIELDS)
                cw = csv.DictWriter(candidate, fieldnames=diagnostics.CANDIDATE_FIELDS)
                sw.writeheader()
                cw.writeheader()
                diagnostics.write_detailed_problem(root, value, summaries, sw, cw)
            with sample_path.open() as stream:
                sample_rows = list(csv.DictReader(stream))
            with candidate_path.open() as stream:
                candidate_rows = list(csv.DictReader(stream))
            self.assertEqual(len(sample_rows), 1)
            self.assertEqual(sample_rows[0]["true_winner"], "A")
            self.assertEqual(sample_rows[0]["selected_solver"], "B")
            self.assertEqual(len(candidate_rows), 3)
            self.assertEqual(sum(r["is_selected"] == "True" for r in candidate_rows), 1)
            with (root / "tables/TSP_missed_samples.csv").open() as stream:
                row = next(csv.DictReader(stream))
            self.assertIn("C__probability", row)
            self.assertEqual(row["A__rank"], "2")


if __name__ == "__main__":
    unittest.main()
