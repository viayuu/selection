import copy
import unittest

import torch
from torch import nn

from .r101_pretrain import hint_loss, learning_rate, raw_inputs, step


class SmallEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(8, 128)

    def forward(self, batch):
        if any(key in batch for key in ('teacher_nodes', 'costs', 'ind', 'winner')):
            raise AssertionError('Supervision entered student forward')
        return self.projection(batch['node']), batch['node_mask']


class HintTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def batch(self):
        generator = torch.Generator().manual_seed(2)
        mask = torch.tensor([[True, True, False], [True, True, True],
                             [True, True, False], [True, True, True], [True, True, True]])
        return dict(kind='coord', problem_id=1, node=torch.randn(5, 3, 8, generator=generator),
            node_mask=mask, n=mask.sum(1), teacher_node_mask=mask,
            teacher_nodes=torch.randn(5, 3, 128, generator=generator))

    def test_no_current_outcomes_or_teacher_in_forward_inputs(self):
        batch = dict(self.batch(), costs=torch.full((5, 10), torch.nan), ind=torch.arange(5),
                     winner='not_an_input', route='not_an_input')
        prepared = raw_inputs(batch)
        for name in ('teacher_nodes', 'costs', 'ind', 'winner', 'route'):
            self.assertNotIn(name, prepared)

    def test_loss_is_per_instance_not_node_count_weighted(self):
        torch.manual_seed(2)
        student = torch.randn(2, 4, 128, requires_grad=True)
        teacher = torch.randn_like(student)
        mask = torch.tensor([[True, False, False, False], [True, True, True, True]])
        got = hint_loss(student, teacher, mask)
        first = hint_loss(student[:1, :1], teacher[:1, :1], mask[:1, :1])
        second = hint_loss(student[1:], teacher[1:], mask[1:])
        torch.testing.assert_close(got, (first+second)/2)
        got.backward()
        self.assertTrue(torch.isfinite(student.grad).all())
        self.assertTrue((student.grad[0, 1:] == 0).all())

    def test_microbatch_and_remainder_preserve_full_batch(self):
        torch.manual_seed(3)
        whole = SmallEncoder()
        micro = copy.deepcopy(whole)
        batch = self.batch()
        optimizers = [torch.optim.SGD(model.parameters(), lr=.01) for model in (whole, micro)]
        records = []
        for model, optimizer, cap in ((whole, optimizers[0], 5), (micro, optimizers[1], 2)):
            scaler = torch.amp.GradScaler('cuda', enabled=False)
            records.append(step(model, optimizer, scaler, batch, cap))
        self.assertAlmostEqual(records[0]['loss'], records[1]['loss'], places=6)
        for first, second in zip(whole.parameters(), micro.parameters()):
            torch.testing.assert_close(first, second, rtol=1e-5, atol=1e-7)
        self.assertEqual(records[0]['retries'], 0)

    def test_invalid_masks_rejected(self):
        encoder = SmallEncoder()
        batch = self.batch()
        batch['teacher_node_mask'] = ~batch['node_mask']
        with self.assertRaises(ValueError):
            step(encoder, torch.optim.SGD(encoder.parameters(), lr=.01),
                 torch.amp.GradScaler('cuda', enabled=False), batch, 2)

    def test_fixed_learning_rate_axis(self):
        self.assertAlmostEqual(learning_rate(960), 1e-4)
        self.assertAlmostEqual(learning_rate(12800), 1e-6)
        self.assertLess(learning_rate(1), learning_rate(960))
        with self.assertRaises(ValueError):
            learning_rate(0)


if __name__ == '__main__':
    unittest.main()
