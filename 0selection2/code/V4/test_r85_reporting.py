"""Small saved-prediction checks for R85 identity, zero mass and NSS reuse."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from code.unified_selector.registry import PROBLEMS
from . import r85_reporting as reporter
from .r76_experiment import file_hash, write_json
from .r85_experiment import selection_counts, zero_summary


class R85ReportingTests(unittest.TestCase):
    def test_exact_four_arm_cli_reuses_the_existing_r84_nss_reference(self):
        for arm in reporter.ARMS:
            for mode, device in (('lib', 'cuda:0'), ('report', 'cpu')):
                args = reporter.parser().parse_args([mode, '--arm', arm, '--device', device])
                self.assertEqual(args.nss_root, reporter.DEFAULT_R84 / 'nss_reference')
                self.assertEqual((args.command, args.device), (mode, device))
        self.assertEqual(reporter.FACTORY, 'code.V4.r85_experiment')

    def test_zero_winner_mass_is_not_all_missed_winners_or_a_recall_claim(self):
        data = dict(TSP__winner=np.array([0, 0, 1, 2]), TSP__pred=np.array([1, 1, 1, 1]),
            TSP__pool=np.array(['A', 'B', 'C']))
        rows = selection_counts(data, 'TSP__', 'TSP', 'val')
        summary = zero_summary(rows)['all']
        self.assertEqual((summary['zero_with_winner_groups'], summary['zero_winner_mass'],
            summary['total_missed_winner_count']), (2, 3, 3))
        data['TSP__pred'][0] = 2  # An incorrect new choice removes C from the strict-zero list.
        summary = zero_summary(selection_counts(data, 'TSP__', 'TSP', 'val'))['all']
        self.assertEqual((summary['zero_with_winner_groups'], summary['zero_winner_mass'],
            summary['total_missed_winner_count']), (1, 2, 3))

    def test_pairing_checks_cost_labels_and_uses_equal_problem_deltas(self):
        old, new = {}, {}
        for i, problem in enumerate(PROBLEMS):
            costs = np.array([[1., 2.], [3., 1.], [1., 1.5]], np.float64) * (i + 1)
            common = dict(pool=['A', 'B'], indices=np.arange(3), source_indices=np.arange(3),
                pool_ids=np.array([0, 1]), winner=np.array([0, 1, 0]), costs=costs, n=3)
            old[problem] = SimpleNamespace(**common, pred=np.array([1, 1, 0]))
            new[problem] = SimpleNamespace(**common, pred=np.array([0, 0, 0]))
        rows = reporter.paired_comparison(old, new, 'R85/test', 'R84A', 'VAL')
        self.assertEqual(rows[-1]['problem'], 'ALL')
        self.assertEqual(rows[-1]['corrected'], 18)
        self.assertEqual(rows[-1]['harmed'], 18)
        self.assertEqual(rows[-1]['top1_delta_pp'], 0.)
        self.assertAlmostEqual(rows[-1]['actual_regret_delta_pp'], 100. / 3.)
        new['TSP'].costs = new['TSP'].costs.copy()
        new['TSP'].costs[0, 0] += 1e-10
        with self.assertRaises(AssertionError):
            reporter.paired_comparison(old, new, 'R85/test', 'R84A', 'VAL')

    def test_nss_reference_hash_population_and_pool_are_required(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data_root = root / 'release'
            plan = dict(data_root=str(data_root), expected_test_rows_per_task=4500,
                retrained_on_new90k=False, checkpoints={p: dict(sha256=p + '-checkpoint') for p in ('TSP', 'CVRP')})
            write_json(root / 'protocol.json', plan)
            for problem in ('TSP', 'CVRP'):
                directory = root / problem
                directory.mkdir()
                path = directory / 'test_predictions.npz'
                np.savez(path, pool=np.array(['A', 'B']), winner=np.zeros(4500, np.int64),
                    costs=np.ones((4500, 2), np.float64))
                write_json(directory / 'test_once.json', dict(state='complete', predictions_sha256=file_hash(path)))
                write_json(directory / 'test_metrics.json', dict(n=4500, checkpoint=plan['checkpoints'][problem]))
            config = dict(data_root=str(data_root), candidate_pools={p: ['A', 'B'] for p in ('TSP', 'CVRP')})
            receipt = reporter.validate_nss_reference(config, root)
            self.assertFalse(receipt['new_inference'])
            self.assertEqual(receipt['exact_test_rows'], 4500)
            write_json(root / 'TSP/test_once.json', dict(state='complete', predictions_sha256='wrong'))
            with self.assertRaises(ValueError):
                reporter.validate_nss_reference(config, root)


if __name__ == '__main__':
    unittest.main()
