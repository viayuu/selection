import unittest

import torch

from .r99_relations import input_geometry, transport
from .r88_model import prepare_role_inputs


class RelationTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)

    def batch(self):
        return dict(kind='coord', problem_id=0, node=torch.tensor(
            [[[0., 0.], [.1, .2], [.8, .7], [0., 0.]]]),
            node_mask=torch.tensor([[True, True, True, False]]), n=torch.tensor([3]))

    def test_finite_padding_and_labels_unused(self):
        batch = self.batch()
        before = batch['node'].clone()
        for variant in ('row', 'balanced'):
            messages, residual = transport(batch, variant)
            self.assertEqual(messages.shape, (1, 4, 32))
            self.assertEqual(messages.dtype, torch.float16)
            self.assertTrue(torch.isfinite(messages).all())
            self.assertTrue((messages[:, -1] == 0).all())
            changed = dict(batch, costs=torch.full((1, 8), torch.nan), ind=torch.tensor([99]),
                           winner=torch.tensor([-1]), routes='must_not_be_read')
            torch.testing.assert_close(messages, transport(changed, variant)[0], rtol=0, atol=0)
            self.assertTrue(torch.isfinite(residual))
        torch.testing.assert_close(batch['node'], before)

    def test_two_node_exact_swap(self):
        batch = self.batch()
        batch['node'], batch['node_mask'], batch['n'] = batch['node'][:, :2], batch['node_mask'][:, :2], torch.tensor([2])
        attributes, _, _ = input_geometry(batch)
        for variant in ('row', 'balanced'):
            messages, residual = transport(batch, variant)
            torch.testing.assert_close(messages[..., :8].float(), attributes.flip(1), atol=1e-3, rtol=1e-3)
            torch.testing.assert_close(messages[..., 16:24].float(), attributes, atol=1e-3, rtol=1e-3)
            self.assertEqual(float(residual), 0.)

    def test_physical_role_preserved_under_relabeling(self):
        batch = prepare_role_inputs(self.batch(), 'B_execution_anchor')
        permutation = torch.tensor([2, 0, 1, 3])
        relabeled = dict(batch, node=batch['node'][:, permutation], node_mask=batch['node_mask'][:, permutation])
        for variant in ('row', 'balanced'):
            expected = transport(batch, variant)[0][:, permutation]
            torch.testing.assert_close(transport(relabeled, variant)[0], expected, atol=1e-3, rtol=1e-3)

    def test_directed_matrix_and_no_self_arc(self):
        batch = dict(kind='matrix', problem_id=2, matrix=torch.tensor([
            [[0., 1., 4.], [3., 0., 2.], [.5, 2., 0.]]]),
            node_mask=torch.ones(1, 3, dtype=torch.bool), n=torch.tensor([3]))
        first = transport(batch, 'row')[0]
        reverse = transport(dict(batch, matrix=batch['matrix'].transpose(1, 2)), 'row')[0]
        self.assertFalse(torch.allclose(first, reverse))
        self.assertTrue(torch.isfinite(transport(batch, 'balanced')[0]).all())

    def test_global_competition_is_not_row_softmax(self):
        batch = self.batch()
        self.assertFalse(torch.equal(transport(batch, 'row')[0], transport(batch, 'balanced')[0]))

    def test_bad_inputs_rejected(self):
        with self.assertRaises(ValueError):
            transport(self.batch(), 'unknown')
        bad = self.batch()
        bad['node'][0, 0, 0] = torch.nan
        with self.assertRaises(ValueError):
            transport(bad, 'row')


if __name__ == '__main__':
    unittest.main()
