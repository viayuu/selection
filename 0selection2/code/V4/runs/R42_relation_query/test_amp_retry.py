"""Post-run CUDA verification of the actual R42 skipped-step retry path."""

import unittest
from unittest.mock import patch

import torch
from torch import nn

from code.V4.r42_experiment import update_batch


class DropoutModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.params = {'architecture': 'dual_stream'}
        self.projection = nn.Linear(8, 4)
        self.dropout = nn.Dropout(.5)
        self.outputs, self.indices, self.weights = [], [], []

    def forward(self, batch):
        assert 'costs' not in batch and 'ind' not in batch
        logits = self.projection(self.dropout(batch['node'].mean(1)))
        self.outputs.append(logits.detach().clone())
        self.indices.append(batch['sample_ids'].clone())
        self.weights.append(self.projection.weight.detach().clone())
        return {'logits': logits}


@unittest.skipUnless(torch.cuda.is_available(), 'requires CUDA GradScaler')
class RetryTests(unittest.TestCase):
    def test_overflow_preserves_batch_and_dropout_streams(self):
        torch.manual_seed(2)
        model = DropoutModel().cuda().train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        scaler = torch.amp.GradScaler('cuda', init_scale=64.)
        costs = 1 + torch.rand(16, 4, device='cuda')
        batch = dict(node=torch.randn(16, 5, 8, device='cuda'), costs=costs,
                     ind=costs.argmin(1), sample_ids=torch.arange(16, device='cuda'))
        backward_calls = []

        def first_overflow(gradient):
            backward_calls.append(True)
            return torch.full_like(gradient, float('inf')) if len(backward_calls) == 1 else gradient

        handle = model.projection.weight.register_hook(first_overflow)
        with patch.object(optimizer, 'step', wraps=optimizer.step) as step:
            result = update_batch(model, optimizer, scaler, batch, .35)
            self.assertEqual(step.call_count, 1)
        handle.remove()
        self.assertEqual(result['skipped'], 1)
        self.assertEqual(len(model.outputs), 4)
        self.assertEqual(scaler.get_scale(), 32.)
        for indices in model.indices:
            torch.testing.assert_close(indices, batch['sample_ids'], rtol=0, atol=0)
        torch.testing.assert_close(model.outputs[0], model.outputs[2], rtol=0, atol=0)
        torch.testing.assert_close(model.outputs[1], model.outputs[3], rtol=0, atol=0)
        self.assertFalse(torch.equal(model.outputs[0], model.outputs[1]))
        torch.testing.assert_close(model.weights[0], model.weights[2], rtol=0, atol=0)
        self.assertFalse(torch.equal(model.weights[0], model.projection.weight))
        self.assertTrue(all(float(v['step']) == 1 for v in optimizer.state.values()))


if __name__ == '__main__':
    unittest.main(verbosity=2)
