"""A few CPU contracts for the complete executing selector, no experiment."""

import unittest

import torch

from . import r113_model as models
from .r77_data import parameter_hash
from .test_r107_model import full18_params, input_batch as graph_fixture


def input_batch(problem):
    batch = graph_fixture(problem)
    if problem == 'ATSP':
        return batch
    node = batch['node'].clone()
    if problem == 'TSP':
        batch['node'] = node[..., :2]
        return batch
    # The old graph-only fixture deliberately used negative demands everywhere.
    # Process tests instead require legal typed inputs, not a weakened compiler.
    if 'B' not in problem:
        node[..., 2] = node[..., 2].abs()
    node[..., 3] = 0
    node[:, 0, 3] = 1
    node[:, 0, 2] = 0
    node[..., 4] = 3 if 'L' in problem else 0
    node[..., 5:7] = 0
    node[..., 7] = 3 if 'TW' in problem else 0
    node[:, 0, 4:] = 0
    batch['node'] = node[..., :4] if problem == 'CVRP' else node
    return batch


class ExecutingSelectorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.models = [models.make_model(full18_params(), arm).eval() for arm in models.ARMS]

    @torch.no_grad()
    def test_fresh_arms_same_model_and_only_predicted_feedback(self):
        hashes = [parameter_hash(model.state_dict()) for model in self.models]
        self.assertEqual(len(set(hashes)), 1)
        for name in ('TSP', 'CVRP', 'ATSP', 'OVRPBLTW'):
            batch = input_batch(name)
            outputs = [model(batch)['logits'] for model in self.models]
            self.assertTrue(torch.isfinite(outputs[0]).all())
            for output in outputs[1:]:
                torch.testing.assert_close(output, outputs[0], rtol=0, atol=0)

    @torch.no_grad()
    def test_labels_isolated_and_candidates_equivariant(self):
        model, batch = self.models[0], input_batch('TSP')
        expected = model(batch)['logits']
        actual = model(dict(batch, ind=object(), costs=object(), process_targets=object()))['logits']
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        order = torch.arange(batch['pool_ids'].numel() - 1, -1, -1)
        changed = model(dict(batch, pool_ids=batch['pool_ids'][order]))['logits']
        torch.testing.assert_close(changed, expected[:, order], atol=3e-6, rtol=1e-5)

    def test_prediction_feedback_and_raw_stem_receive_gradients(self):
        model = self.models[0]
        model.zero_grad(set_to_none=True)
        logits = model(input_batch('CVRP'))['logits']
        torch.nn.functional.cross_entropy(logits, torch.tensor([0, 1])).backward()
        for name in ('instance_encoder.stem.coord_proj.weight',
                     'instance_encoder.processor.resource_operator.0.weight',
                     'instance_encoder.processor.graph_state.0.weight',
                     'instance_encoder.edge_projection.0.weight'):
            gradient = dict(model.named_parameters())[name].grad
            self.assertIsNotNone(gradient, name)
            self.assertTrue(torch.isfinite(gradient).all(), name)
            self.assertGreater(gradient.abs().max().item(), 0, name)


if __name__ == '__main__':
    unittest.main()
