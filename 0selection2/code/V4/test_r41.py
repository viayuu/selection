import copy
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from .multitask_probe import state_hash
from .r41_label_stability import OVRPTW_FIELDS, proportional_indices, validate_saved_run
from .r40_experiment import aggregate, classification_metrics
from .r41_analysis import aligned_repeats, complete_pool_metrics, head_pair_tables


class R41Tests(unittest.TestCase):
    def test_sampling_proportional_distinct_reproducible(self):
        nodes = np.repeat([50, 100, 200, 400], [10, 20, 30, 40])
        cuts = np.array([75, 150, 300])
        indices, groups, quotas = proportional_indices(nodes, cuts, 64, 2)
        self.assertEqual(len(np.unique(indices)), 64)
        np.testing.assert_array_equal(quotas, [6, 13, 19, 26])
        np.testing.assert_array_equal(np.bincount(groups[indices]), quotas)
        np.testing.assert_array_equal(indices, proportional_indices(nodes, cuts, 64, 2)[0])

    def test_native_winner_and_raw_precision(self):
        costs = np.array([[1.00000001, 1.00000002], [2., 2.]])
        raw = dict(costs=costs, winner=np.array([0, 1]), pool=['a', 'b'])
        result, pred = classification_metrics(np.array([[0., 1.], [0., 1.]], np.float32), raw)
        self.assertEqual(result['top1'], .5)
        self.assertGreater(result['actual_regret_pct'], 0)
        self.assertEqual(result['top1_tie_aware'], .5)
        self.assertEqual(result['mean_cost'], (1.00000002 + 2.) / 2)

    def test_macro_percentages_are_not_ratio_of_mean_costs(self):
        per = {}
        for name, costs in [('a', [1., 2.]), ('b', [100., 90.])]:
            raw = dict(costs=np.array([costs]), winner=np.array([np.argmin(costs)]), pool=['x', 'y'])
            per[name], _ = classification_metrics(np.array([[1., 0.]], np.float32), raw)
        macro = aggregate(per)
        self.assertAlmostEqual(macro['macro_vs_sbs_pct'], (0 + (100 / 90 - 1) * 100) / 2)

    def test_missing_candidate_never_enters_full_pool_metrics(self):
        costs = np.array([[[1., 2.], [1., np.nan]], [[1., 3.], [1., np.nan]],
                          [[np.nan, 4.], [np.nan, np.nan]]])
        old = np.array([[1., 2.], [1., 2.]])
        rows, complete, reused = complete_pool_metrics(costs, old, np.array([0, 0]), np.array([1, 0]), [True, False])
        np.testing.assert_array_equal(complete, [True, False])
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]['observed_full_vectors'], 2)
        self.assertEqual(rows[0]['aligned_vectors'], 3)
        self.assertEqual(rows[0]['reused_deterministic_costs'], 1)
        self.assertTrue(np.isnan(costs[2, 0, 0]))
        self.assertAlmostEqual(rows[0]['r39a_mean_regret_pct'], 200)

    def test_nonconstant_deterministic_cost_is_not_fabricated(self):
        costs = np.array([[[1., 2.]], [[1.000001, 2.]], [[np.nan, 2.]]])
        aligned, reused = aligned_repeats(costs, [True, False])
        self.assertTrue(np.isnan(aligned[2, 0, 0]))
        self.assertFalse(reused.any())

    def test_numerical_ties_are_separate_from_strict_winner(self):
        costs = np.array([[[1., 1.00000001]], [[1.00000001, 1.]], [[1., 1.00000001]]])
        rows, _, _ = complete_pool_metrics(costs, costs[0], np.array([0]), np.array([0]), [False, False])
        self.assertFalse(rows[0]['strict_winner_consistent'])
        self.assertTrue(rows[0]['numerical_tie_consistent'])

    def run_fixture(self):
        instances = [{field: np.zeros((2, 2) if field == 'node_xy' else (2,)) for field in OVRPTW_FIELDS}
                     for _ in range(2)]
        tensors = tuple(torch.stack([torch.as_tensor(item[field], dtype=torch.float32) for item in instances])
                        for field in OVRPTW_FIELDS)
        params = dict(eval_type='argmax', encoder_layer_num=6)
        entry = dict(problem='OVRPTW', solver='RELD_MTL', randomness='deterministic',
             checkpoint=dict(path='original.pt', sha256='weight_hash'), source_sha256='source_hash',
             runtime_params_contract=params, decode_budget=dict(aug_factor=1, sample_size=1, original_batch_size=128))
        sample = dict(indices=[5, 9], precheck_indices=[5, 9])
        witness = dict(problem='OVRPTW', solver='RELD_MTL', split='train', seed=2, precheck=False,
             model_eval=True, independent_process=True, pid=123, augmentation=1, sample_size=1,
             checkpoint=copy.deepcopy(entry['checkpoint']), source_sha256='source_hash', params=copy.deepcopy(params),
             input_witness=[dict(scale=2, original_indices=[5, 9], fields=list(OVRPTW_FIELDS),
                                 input_hash=state_hash(tensors), input_matches_original=True)])
        return entry, sample, witness, instances

    def test_repeat_rejects_wrong_contract_or_empty_input_witness(self):
        entry, sample, witness, instances = self.run_fixture()
        validate_saved_run(entry, sample, 'train', witness, np.ones(2), np.array([5, 9]), instances, set(), 0)
        mutations = [('augmentation', 8), ('sample_size', 99), ('input_witness', []),
                     ('model_eval', False), ('independent_process', False), ('seed', 3),
                     ('params', dict(eval_type='sampling', encoder_layer_num=6))]
        for key, value in mutations:
            with self.subTest(key=key), self.assertRaises(ValueError):
                changed = copy.deepcopy(witness)
                changed[key] = value
                validate_saved_run(entry, sample, 'train', changed, np.ones(2), np.array([5, 9]), instances, set(), 0)
        with self.assertRaises(ValueError):
            validate_saved_run(entry, sample, 'train', witness, np.ones(2), np.array([5, 9]), instances, {123}, 1)
        for costs in (np.array([1., np.nan]), np.array([1., 0.])):
            with self.assertRaises(ValueError):
                validate_saved_run(entry, sample, 'train', witness, costs, np.array([5, 9]), instances, set(), 0)

    def test_repeat_rejects_changed_original_input(self):
        entry, sample, witness, instances = self.run_fixture()
        instances[0]['node_xy'][0, 0] = 1.
        with self.assertRaises(ValueError):
            validate_saved_run(entry, sample, 'train', witness, np.ones(2), np.array([5, 9]), instances, set(), 0)

    def test_pair_report_with_no_in_pair_predictions(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sample = dict(indices=[5, 9], nodes=[2, 2], pool=['RELD_MOEL', 'RELD_MTL', 'other'])
            (root / 'audit_indices.json').write_text(json.dumps(dict(problems=dict(OVRPTW=dict(splits=dict(train=sample))))))
            costs = np.array([[[1., 2., np.nan], [2., 1., np.nan]],
                              [[1., 2., np.nan], [2., 1., np.nan]],
                              [[np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan]]])
            np.savez_compressed(root / 'repeated_costs.npz', OVRPTW_train_costs=costs,
                 OVRPTW_train_historical=np.array([[1., 2., 3.], [2., 1., 3.]]),
                 OVRPTW_train_pred=np.array([2, 2]), OVRPTW_train_winner=np.array([0, 1]))
            head_pair_tables(root)
            report = (root / 'head_pair_stability.md').read_text()
            self.assertIn('not measured', report)
            self.assertNotIn('This supports representation/generalization', report)
            self.assertNotIn('many instances', report)
            summary = json.loads((root / 'head_pair_stability_summary.json').read_text())[0]
            self.assertEqual(summary['r39a_picks_in_pair'], 0)
            self.assertIsNone(summary['r39a_pair_top1'])


if __name__ == '__main__':
    unittest.main()
