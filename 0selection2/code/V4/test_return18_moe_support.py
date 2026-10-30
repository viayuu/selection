"""Small one-CPU geometry/identity fixtures, never model inference."""

import json
import unittest
from unittest.mock import patch

from . import return18_moe_support as m

import numpy as np


def setUpModule():
    m.pin_cpu()


class GeometryTests(unittest.TestCase):
    def test_native_missed_candidate_can_become_unique_top1(self):
        scores = np.array([[.5, .1, .4], [.1, .5, .4], [.5, .1, .4]])
        self.assertFalse(np.any(m.saved.hard_actions(scores, np.arange(3)) == 2))
        result = m.solve_support(scores, 2, np.arange(3))
        self.assertEqual(result['state'], m.REACHABLE)
        self.assertEqual(result['reason'], 'positive_margin')
        self.assertAlmostEqual(sum(result['weights']), 1.)

    def test_convex_obstruction_without_single_common_dominator(self):
        scores = np.array([[.3, .6, .1], [.3, .1, .6], [.3, .6, .1]])
        self.assertFalse(np.any(np.all(scores[:, 1:] > scores[:, :1], axis=0)))
        result = m.solve_support(scores, 0, np.arange(3))
        self.assertEqual(result['state'], m.UNREACHABLE)
        self.assertEqual(result['reason'], 'convex_rival_certificate')
        self.assertLess(result['margin_upper'], -m.TOL)

    def test_tied_candidate_with_lower_ID_rival_is_not_rescued(self):
        scores = np.ones((3, 3))
        result = m.solve_support(scores, 2, np.array([5, 10, 20]))
        self.assertEqual(result['state'], m.UNREACHABLE)
        self.assertEqual(result['reason'], 'exact_lower_ID_block')

    def test_target_lowest_global_ID_can_win_an_exact_tied_face(self):
        scores = np.array([[0., 1., -1.], [0., -1., 1.], [0., 1., -1.]])
        result = m.solve_support(scores, 0, np.array([0, 2, 3]))
        self.assertEqual(result['state'], m.REACHABLE)
        self.assertEqual(result['reason'], 'verified_ID_tie_face')

    def test_tiny_positive_margin_is_boundary_not_confident_rescue(self):
        scores = np.array([[0., 1., -1. - 1e-11], [0., -1. - 1e-11, 1.],
                           [0., 1., -1. - 1e-11]])
        result = m.solve_support(scores, 0, np.array([3, 0, 2]))
        self.assertEqual(result['state'], m.BOUNDARY)

    def test_strict_dominance_skips_LP_and_preserves_hard_partition(self):
        scores = np.array([[[1., 0.]] * 3, [[0., 1.]] * 3])
        with patch.object(m, 'solve_support', side_effect=AssertionError('no LP needed')):
            result = m.support(scores, np.array([1, 1]), np.array([10, 2]), np.array([False, True]))
        np.testing.assert_array_equal(result['state'], [m.UNREACHABLE, m.HARD])
        self.assertEqual(result['lp_calls'], 0)

    def test_failure_is_boundary_without_retry(self):
        failed = m.SimpleNamespace(success=False, status=1, x=None)
        with patch.object(m, 'margin_lp', return_value=failed) as lp:
            result = m.solve_support(np.ones((3, 2)), 0, np.arange(2))
        self.assertEqual(lp.call_count, 1)
        self.assertEqual(result['state'], m.BOUNDARY)

    def test_softmax_is_fixed_temperature_and_numerically_stable(self):
        scores = np.array([[[1000., 999., -1000.]] * 3])
        p = m.softmax64(scores)
        self.assertTrue(np.isfinite(p).all())
        np.testing.assert_allclose(p.sum(-1), 1.)
        self.assertAlmostEqual(p[0, 0, 0] / p[0, 0, 1], np.e)


class IdentityTests(unittest.TestCase):
    def fixture(self):
        n = 4500
        bank = dict(logits=np.zeros((n, 2), dtype=np.float32),
            row_indices=np.arange(n, dtype=np.int64), source_indices=np.arange(n),
            source_identity=np.asarray([str(i).encode() for i in range(n)]),
            pool=np.array(['A', 'B']), pool_ids=np.array([10, 20]))
        protocol = dict(expert_bank=[dict(expert_id='R77B', checkpoint_path='fixed.pt',
            checkpoint_sha256='abc', factory='frozen.factory')],
            candidate_pools={'TSP': ['A', 'B']}, candidate_pool_ids={'TSP': [10, 20]},
            shared_prediction_cache_sha256='helper')
        meta = dict(schema='r123_shared_expert_npz_v1', complete=True, rows=n,
            split='test', problem='TSP', expert_id='R77B', bank_order=list(m.saved.EXPERTS),
            checkpoint_path='fixed.pt', checkpoint_sha256='abc', factory='frozen.factory',
            meta_preparation_identity='manifest', pool=['A', 'B'], pool_ids=[10, 20],
            helper_sha256='helper', score_ties='ascending global solver ID', query_labels_stored=False)
        for name in ('logits', 'row_indices', 'source_indices', 'source_identity'):
            meta[name + '_sha256'] = m.array_hash(bank[name])
        bank['metadata_json'] = np.asarray(json.dumps(meta))
        capture = dict(winner=np.zeros(n, dtype=np.int64), pool=bank['pool'], pool_ids=bank['pool_ids'],
            indices=bank['row_indices'], release_source_indices=bank['source_indices'],
            source_identity=bank['source_identity'])
        args = dict(protocol=protocol, manifest={'preparation_identity': 'manifest'},
                    split='test', problem='TSP', expert='R77B')
        return bank, capture, args

    def test_saved_schema_and_source_identity_tamper_fail_closed(self):
        bank, capture, args = self.fixture()
        m.validate_archive(bank, capture, **args)
        bank['source_indices'] = bank['source_indices'][::-1]
        with self.assertRaisesRegex(ValueError, 'hash'):
            m.validate_archive(bank, capture, **args)

    def test_checkpoint_or_legal_pool_cannot_be_substituted(self):
        bank, capture, args = self.fixture()
        args['protocol']['expert_bank'][0]['checkpoint_sha256'] = 'other'
        with self.assertRaisesRegex(ValueError, 'version'):
            m.validate_archive(bank, capture, **args)


if __name__ == '__main__':
    unittest.main()
