"""Bounded CPU checks for qualified identities, candidate alignment and isolation."""

import copy
import unittest

import numpy as np
import torch

from .r123_meta_data import (ExpertAdapter, align_candidate_scores, candidate_permutation,
    model_inputs, normalize_scores, source_record_ids, unseen_intersection, validate_lineage)


def generation(start=10, digest='a' * 64):
    return dict(original_instances=start, shards=[dict(id='shard0', start_index=start,
        count=10, sha256=digest, scale=50, distribution='uniform', capacity_type='none', seed=2)])


class MetaDataTests(unittest.TestCase):
    def test_source_identity_not_bare_index_or_path(self):
        first = source_record_ids('TSP', np.asarray([17]), generation())
        relocated = source_record_ids('TSP', np.asarray([107]), generation(start=100))
        other_content = source_record_ids('TSP', np.asarray([17]), generation(digest='b' * 64))
        np.testing.assert_array_equal(first, relocated)
        self.assertNotEqual(first[0], other_content[0])
        self.assertNotEqual(first[0], source_record_ids('CVRP', np.asarray([17]), generation())[0])
        with self.assertRaises(ValueError):
            source_record_ids('TSP', np.asarray([9]), generation())

    def test_source_lineage_requires_explicit_shard_binding(self):
        source = generation()
        old = dict(index_mapping='local_index == original_global_index', original_instances=10,
            global_index_start=0, global_index_stop=20, instances=20,
            included_new_shards=copy.deepcopy(source['shards']))
        validate_lineage(old, source)
        changed = copy.deepcopy(old)
        changed['included_new_shards'][0]['sha256'] = 'b' * 64
        with self.assertRaisesRegex(ValueError, 'immutable shard'):
            validate_lineage(changed, source)

    def test_unseen_union_and_missing_evidence_are_conservative(self):
        current = np.asarray(['a', 'b', 'c', 'd'], dtype='S2')
        local, gaps = unseen_intersection(current, {'first': current[[0]], 'second': current[[2]]})
        np.testing.assert_array_equal(local, [1, 3])
        self.assertEqual(gaps, [])
        local, gaps = unseen_intersection(current, {'unknown': None})
        self.assertEqual(len(local), 0)
        self.assertTrue(gaps)
        local, _ = unseen_intersection(current, {'full81k': current})
        self.assertEqual(len(local), 0)

    def test_candidate_alignment_never_merges_reld(self):
        aliases = {'reld_mt': 'RELD_MOEL'}
        values = np.asarray([[1., 2., 3.]])
        aligned = align_candidate_scores(values, ['ELG', 'BQ', 'RELD_MOEL'],
                                         ['BQ', 'reld_mt', 'ELG'], aliases)
        np.testing.assert_array_equal(aligned, [[2., 3., 1.]])
        with self.assertRaisesRegex(ValueError, 'collide'):
            candidate_permutation(['reld_mt', 'RELD_MOEL'], ['reld_mt'], aliases)
        with self.assertRaisesRegex(ValueError, 'legal candidate'):
            candidate_permutation(['BQ', 'ELG'], ['BQ', 'RouteFinder'])

    def test_explicit_label_free_normalization(self):
        scores = np.asarray([[1., 2., 3.], [4., 4., 4.]])
        np.testing.assert_array_equal(normalize_scores(scores), scores)
        np.testing.assert_allclose(normalize_scores(scores, 'center').mean(-1), 0.)
        scaled = normalize_scores(scores, 'zscore')
        self.assertAlmostEqual(float(scaled[0].std()), 1.)
        np.testing.assert_array_equal(scaled[1], [0., 0., 0.])
        with self.assertRaises(ValueError):
            normalize_scores(scores, 'fit_on_validation')

    def test_expert_adapter_excludes_supervision_and_freezes_weights(self):
        class TinyExpert(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor(1.))
                self.seen = None

            def forward(self, batch):
                self.seen = set(batch)
                return dict(logits=self.weight * batch['node'][:, 0, :2])

        model = TinyExpert()
        adapter = ExpertAdapter(model, ['BQ', 'RELD_MOEL'], {'VRPB': ['BQ', 'reld_mt']},
                                {'reld_mt': 'RELD_MOEL'})
        batch = dict(n=torch.tensor([1]), node=torch.tensor([[[2., 3.]]]),
            costs=torch.tensor([[0., 9.]]), ind=torch.tensor([0]),
            native_objective=torch.tensor([[0., 9.]]), metadata={'instance_id': 'private'},
            labels={'winner': 0}, source_identity='private')
        values = adapter.predict('VRPB', batch, ['reld_mt', 'BQ'])
        np.testing.assert_array_equal(values, [[3., 2.]])
        self.assertFalse(model.training)
        self.assertFalse(model.weight.requires_grad)
        self.assertTrue(model.seen <= set(model_inputs(batch)) | {'pool_ids', 'solver_mask'})
        self.assertFalse(model.seen & {'costs', 'ind', 'labels', 'metadata', 'native_objective', 'source_identity'})


if __name__ == '__main__':
    unittest.main()
