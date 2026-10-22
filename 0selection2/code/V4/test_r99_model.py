import unittest

import torch

from .r88_model import make_model as original_model
from .r99_model import make_model, inputs_with_relations
from .r99_relations import node_messages
from .test_r94_model import fixture_params, cached_batch


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_same_original_parameters_and_matched_new_projection(self):
        torch.manual_seed(2)
        reference = original_model(fixture_params(), 'B_execution_anchor')
        models = []
        for arm in ('A', 'B'):
            torch.manual_seed(2)
            model = make_model(fixture_params(), arm)
            parameters = dict(model.named_parameters())
            for key, value in reference.named_parameters():
                torch.testing.assert_close(parameters[key], value, atol=0, rtol=0)
            self.assertEqual(set(parameters)-set(dict(reference.named_parameters())),
                             {'instance_encoder.relation_projection.weight'})
            models.append(model)
        for (key, first), (_, second) in zip(models[0].named_parameters(), models[1].named_parameters()):
            torch.testing.assert_close(first, second, atol=0, rtol=0, msg=key)

    def test_all_input_kinds_have_finite_outputs_and_label_isolation(self):
        for problem in ('TSP', 'CVRP', 'ATSP', 'VRPTW'):
            model = make_model(fixture_params(), 'B').eval()
            batch = cached_batch(problem)
            baseline = model(batch)['logits']
            changed = dict(batch, costs=batch['costs']*100, ind=1-batch['ind'],
                           winner=torch.tensor([-100, 100]), route='not_an_input')
            torch.testing.assert_close(model(changed)['logits'], baseline, rtol=0, atol=0)
            self.assertTrue(torch.isfinite(baseline).all())
            self.assertNotIn('winner', inputs_with_relations(changed))

    def test_projection_and_original_backbone_receive_gradients(self):
        model = make_model(fixture_params(), 'B')
        batch = cached_batch('CVRP')
        batch['relation_nodes'] = node_messages(batch, 'balanced')
        batch['relation_node_mask'] = batch['node_mask']
        logits = model(batch)['logits']
        logits.square().mean().backward()
        for parameter in (model.instance_encoder.relation_projection.weight,
                          model.instance_encoder.coord_proj.weight):
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(float(parameter.grad.abs().sum()), 0.)

    def test_uncached_training_rejected(self):
        model = make_model(fixture_params(), 'A').train()
        with self.assertRaises(ValueError):
            model(cached_batch('CVRP'))


if __name__ == '__main__':
    unittest.main()
