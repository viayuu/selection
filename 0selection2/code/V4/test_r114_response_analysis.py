import copy
import unittest

from .r114_response_analysis import build_tables


def fixture():
    pairs, endpoints = [], []
    for arm, costs in [('directed', [11., 12., 13.]), ('random', [10., 12., 14.])]:
        original = dict(endpoint_id='base', base_id='source1', methods=['a', 'b', 'c'],
            costs_fp64=[10., 11., 12.], valid_mask=[True]*3, winner='a')
        changed = dict(original, endpoint_id=arm, costs_fp64=costs)
        endpoints.append(changed)
        change = [a-b for a, b in zip(costs, original['costs_fp64'])]
        pairs.append(dict(pair_id=arm, base_id='source1', problem='CVRP', family='demand',
            arm=arm, original_id='base', changed_id=arm, l1=.2, moved_customers=4,
            methods=['a', 'b', 'c'], winner_pair_mask=True,
            Delta_mn=[[a-b for b in change] for a in change]))
    endpoints.append(original)
    manifest = dict(version='fixture', manifest_sha256='fixturehash', pairs=copy.deepcopy(pairs))
    labels = dict(version='fixture', manifest_sha256='fixturehash', pairs=pairs, endpoints=endpoints)
    return manifest, labels


class ResponseAnalysisTests(unittest.TestCase):
    def test_common_shift_not_mistaken_for_relative_response(self):
        tables = build_tables(*fixture())
        directed = next(r for r in tables['pairs'] if r['arm'] == 'directed')
        self.assertEqual(directed['rms_delta_pct'], 0.)
        self.assertEqual(directed['common_cost_change_pct'], 10.)
        self.assertEqual(len(tables['matched']), 1)
        self.assertEqual(tables['matched_summary'][0]['matched_full_pool_bases'], 1)
        self.assertLess(tables['matched_summary'][0]['mean_rms_delta_difference_pp'], 0.)

    def test_partial_costs_never_supply_full_pool_statistics(self):
        manifest, labels = fixture()
        labels['endpoints'][0]['valid_mask'][1] = False
        labels['endpoints'][0]['costs_fp64'][1] = None
        labels['pairs'][0]['winner_pair_mask'] = False
        tables = build_tables(manifest, labels)
        self.assertIsNone(tables['pairs'][0]['winner_flip'])
        self.assertIsNone(tables['pairs'][0]['rms_delta_pct'])
        self.assertEqual(tables['matched_summary'][0]['matched_full_pool_bases'], 0)

    def test_wrong_native_delta_and_version_rejected(self):
        manifest, labels = fixture()
        labels['pairs'][0]['Delta_mn'][0][1] = 3
        with self.assertRaises(ValueError):
            build_tables(manifest, labels)
        manifest, labels = fixture()
        labels['version'] = 'different'
        with self.assertRaises(ValueError):
            build_tables(manifest, labels)


if __name__ == '__main__':
    unittest.main()
