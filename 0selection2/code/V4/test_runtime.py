import copy
import unittest

import torch

from code.unified_selector.data import UnifiedProblemDataset, collate_single_problem

from .V4Model import MultiHeadAttention
from .tensor_loader import TensorBatchLoader


class RuntimeTests(unittest.TestCase):
    def test_sdpa_preserves_attention_and_bias_gradients(self):
        torch.manual_seed(2)
        regular = MultiHeadAttention(16, 2, 8, 0.0)
        fused = copy.deepcopy(regular)
        fused.sdpa = True
        x = torch.randn(2, 7, 16, requires_grad=True)
        bias = torch.randn(2, 2, 7, 7, requires_grad=True)
        mask = torch.ones(2, 7, dtype=torch.bool)
        mask[0, -2:] = False
        x2 = x.detach().clone().requires_grad_()
        bias2 = bias.detach().clone().requires_grad_()
        a = regular(x, x, x, key_mask=mask, attn_bias=bias)
        b = fused(x2, x2, x2, key_mask=mask, attn_bias=bias2)
        torch.testing.assert_close(a, b, rtol=2e-5, atol=1e-6)
        a.square().sum().backward()
        b.square().sum().backward()
        torch.testing.assert_close(x.grad, x2.grad, rtol=2e-5, atol=1e-6)
        torch.testing.assert_close(bias.grad, bias2.grad, rtol=2e-5, atol=1e-6)
        for left, right in zip(regular.parameters(), fused.parameters()):
            torch.testing.assert_close(left.grad, right.grad, rtol=2e-5, atol=2e-6)

    def test_cache_matches_original_samples_and_padding(self):
        for problem in ("TSP", "CVRP", "ATSP", "OVRPBLTW"):
            with self.subTest(problem=problem):
                dataset = UnifiedProblemDataset(problem, "val")
                samples = [dataset[i] for i in (0, 17, 99, 300, 999)]
                cached = collate_single_problem(samples)
                loader = TensorBatchLoader(cached, 2, shuffle=False, drop_last=False)
                self.assertEqual(len(loader), 3)
                for i, actual in enumerate(loader):
                    expected = collate_single_problem(samples[2 * i : 2 * i + 2])
                    for key in expected:
                        if torch.is_tensor(expected[key]):
                            torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
                        else:
                            self.assertEqual(actual[key], expected[key])
                dropped = TensorBatchLoader(cached, 2, shuffle=False, drop_last=True)
                self.assertEqual(sum(len(b["costs"]) for b in dropped), 4)
                shuffled = TensorBatchLoader(cached, 2, shuffle=True, drop_last=False)
                indices = torch.cat([b["ind"] for b in shuffled]).sort().values
                torch.testing.assert_close(indices, cached["ind"].sort().values)


if __name__ == "__main__":
    unittest.main()
