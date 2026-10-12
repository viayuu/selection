"""Focused CPU checks for directed neighbors, padding and true hub identities."""

import unittest

import torch

from code.unified_selector.registry import P2I
from .r63_graph import build_workspace_graph


def edge_set(graph):
    return set(zip(graph['flat_source'].tolist(), graph['flat_receiver'].tolist()))


class R63GraphTests(unittest.TestCase):
    def test_directed_outgoing_incoming_union_and_four_distance_features(self):
        n = 21
        matrix = torch.arange(1, n * n + 1, dtype=torch.float32).reshape(n, n)
        matrix.fill_diagonal_(0.)
        matrix.requires_grad_(True)
        graph = build_workspace_graph(dict(kind='matrix', problem_id=P2I['ATSP'],
            matrix=matrix[None], node_mask=torch.ones(1, n, dtype=torch.bool)))
        expected = set()
        for i in range(n):
            outgoing = sorted((j for j in range(n) if j != i), key=lambda j: float(matrix[i, j]))[:16]
            incoming = sorted((j for j in range(n) if j != i), key=lambda j: float(matrix[j, i]))[:16]
            expected.update((i + 3, j + 3) for j in outgoing)
            expected.update((j + 3, i + 3) for j in incoming)
        edges = edge_set(graph)
        self.assertEqual({(i, j) for i, j in edges if i >= 3 and j >= 3}, expected)
        self.assertEqual(len(edges), graph['flat_source'].numel())
        self.assertTrue(all(i != j for i, j in edges))
        scale = matrix.detach()[~torch.eye(n, dtype=torch.bool)].abs().mean()
        for index, (i, j) in enumerate(zip(graph['flat_source'].tolist(), graph['flat_receiver'].tolist())):
            features = graph['distance_features'][index]
            if i >= 3 and j >= 3:
                f, r = matrix.detach()[i - 3, j - 3], matrix.detach()[j - 3, i - 3]
                torch.testing.assert_close(features, torch.stack((f, f / scale, r, r / scale)))
            else:
                self.assertTrue((features == 0).all())
        self.assertFalse(graph['distance_features'].requires_grad)
        self.assertEqual(graph['node_roles'][0, 3:].tolist(), [0] * n)

    def test_padding_flat_batch_offsets_roles_and_label_independence(self):
        node = torch.arange(40, dtype=torch.float32).reshape(2, 5, 4) / 20
        node[..., 3] = 0.
        node[:, 0, 3] = 1.
        mask = torch.tensor([[True, True, True, False, False], [True, True, True, True, True]])
        batch = dict(kind='coord', problem_id=P2I['CVRP'], node=node, node_mask=mask)
        graph = build_workspace_graph(batch)
        self.assertEqual(graph['n_nodes'], 8)
        torch.testing.assert_close(graph['node_mask'], torch.cat((torch.ones(2, 3, dtype=torch.bool), mask), 1))
        self.assertEqual(graph['node_roles'][0].tolist(), [2, 3, 4, 1, 0, 0, -1, -1])
        valid = graph['node_mask'].flatten()
        for source, receiver in edge_set(graph):
            self.assertTrue(valid[source] and valid[receiver])
            self.assertEqual(source // 8, receiver // 8)
        changed_node = node.clone()
        changed_node[~mask] = 100000.
        other = build_workspace_graph(dict(batch, node=changed_node, costs=torch.zeros(2, 10),
                                          ind=torch.ones(2), winner='unused', route='unused'))
        self.assertEqual(edge_set(graph), edge_set(other))
        torch.testing.assert_close(graph['distance_features'], other['distance_features'])
        torch.testing.assert_close(graph['distance_scale'], other['distance_scale'])

    def test_tsp_zero_is_customer_true_depot_and_specials_are_hubs(self):
        n = 25
        xy = torch.stack((torch.arange(n, dtype=torch.float32), torch.zeros(n)), -1)
        xy[0, 0] = 1000.
        mask = torch.ones(1, n, dtype=torch.bool)
        tsp = build_workspace_graph(dict(kind='coord', problem_id=P2I['TSP'], node=xy[None], node_mask=mask))
        edges = edge_set(tsp)
        self.assertEqual(tsp['node_roles'][0, 3], 0)
        self.assertLess(sum(source == 3 and target >= 3 for source, target in edges), n - 1)
        for special in range(3):
            for other in range(n + 3):
                if special != other:
                    self.assertIn((special, other), edges)
                    self.assertIn((other, special), edges)
        node = torch.cat((xy, torch.zeros(n, 2)), -1)
        node[0, 3] = 1.
        vrp = build_workspace_graph(dict(kind='coord', problem_id=P2I['VRPB'], node=node[None], node_mask=mask))
        vrp_edges = edge_set(vrp)
        self.assertEqual(vrp['node_roles'][0, 3], 1)
        for other in range(n + 3):
            if other != 3:
                self.assertIn((3, other), vrp_edges)
                self.assertIn((other, 3), vrp_edges)


if __name__ == '__main__':
    unittest.main()
