"""CPU witnesses for a dropout-only control with the unchanged complete model."""

import copy
import unittest

import torch
from torch import nn

from .r77_data import parameter_hash
from .r93_model import DROPOUT, make_model
from .test_r92_model import graph_batch, model_params


class R93ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_same_parameters_and_eval_with_only_stochastic_dropout_changed(self):
        torch.manual_seed(2)
        a = make_model(model_params(), 'A')
        torch.manual_seed(2)
        b = make_model(model_params(), 'B')
        self.assertEqual(parameter_hash(a.state_dict()), parameter_hash(b.state_dict()))
        self.assertEqual(set(dict(a.named_parameters())), set(dict(b.named_parameters())))
        for arm, model in (('A', a), ('B', b)):
            drops = [part.p for part in model.modules() if isinstance(part, nn.Dropout)]
            self.assertTrue(drops)
            self.assertTrue(all(probability == DROPOUT[arm] for probability in drops))
            self.assertEqual(model.params['dropout'], DROPOUT[arm])
        a.eval()
        b.eval()
        for problem in ('TSP', 'CVRP', 'ATSP', 'OVRPBLTW'):
            batch = graph_batch(problem, rows=2, nodes=11)
            with torch.no_grad():
                first, second = a(batch), b(batch)
            for name in ('logits', 'utility', 'pair_margin', 'solver_mask'):
                torch.testing.assert_close(first[name], second[name], rtol=0, atol=0)
            self.assertTrue(torch.equal(first['logits'].argmax(-1), first['utility'].argmax(-1)))

    def test_strict_arm_dropout_restore_and_label_isolation(self):
        for arm in ('A', 'B'):
            model = make_model(model_params(), arm).eval()
            restored = make_model(model.params, arm, model.state_dict()).eval()
            self.assertEqual(parameter_hash(model.state_dict()), parameter_hash(restored.state_dict()))
            with self.assertRaises(ValueError):
                make_model(model.params, 'B' if arm == 'A' else 'A')
            invalid = copy.deepcopy(model.params)
            invalid['dropout'] = .4
            with self.assertRaises(ValueError):
                make_model(invalid, arm)
            batch = graph_batch('TSP', rows=2, nodes=11)
            alternate = dict(batch, costs=torch.full_like(batch['costs'], 123.),
                             ind=torch.zeros(len(batch['n']), dtype=torch.long))
            with torch.no_grad():
                torch.testing.assert_close(model(batch)['logits'], model(alternate)['logits'], rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
