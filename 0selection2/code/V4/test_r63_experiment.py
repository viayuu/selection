"""Small CPU fixtures for R63 invariance, differentiable memory and accumulation."""

import copy
import unittest

import torch
from torch import nn

from code.unified_selector.registry import K_CBITS, K_PROBLEM_DESC, P2I
from .pairwise_objective import inputs_only
from .r45_experiment import parameters
from .r63_model import RecurrentWorkspaceSelector, classification_objective, update_batch


def fixture(rows=2, candidates=4):
    generator = torch.Generator().manual_seed(2)
    lengths = torch.tensor([5 if i % 2 == 0 else 3 for i in range(rows)])
    return dict(kind='coord', problem_id=P2I['TSP'], node=torch.rand(rows, 5, 2, generator=generator),
                node_mask=torch.arange(5)[None] < lengths[:, None], n=lengths,
                cbits=torch.zeros(rows, K_CBITS), problem_desc=torch.zeros(rows, K_PROBLEM_DESC),
                coord_dist=torch.zeros(rows, dtype=torch.long), pool_ids=torch.arange(candidates),
                costs=torch.rand(rows, candidates, generator=generator, dtype=torch.float64),
                ind=torch.arange(rows) % candidates)


def small_model(communicating):
    torch.manual_seed(2)
    params = dict(parameters('A', sdpa=False), embedding_dim=16, head_num=2, qkv_dim=8,
                  ff_hidden_dim=32, head_hidden_dim=16, solver_feature_hidden=16,
                  local_geometry=False, dropout=0., workspace_rounds=8, message_dim=32)
    return RecurrentWorkspaceSelector(params, communicating)


class TinySelector(nn.Module):
    def __init__(self, dropout=0.):
        super().__init__()
        self.linear = nn.Linear(2, 3)
        self.dropout = nn.Dropout(dropout)
        self.calls = []

    def forward(self, batch):
        logits = self.linear(self.dropout(batch['node'].mean(1)))
        self.calls.append((set(batch), logits.detach().clone()))
        return dict(logits=logits, solver_mask=torch.ones_like(logits, dtype=torch.bool))


class CountingSGD(torch.optim.SGD):
    def __init__(self, model):
        super().__init__(model.parameters(), lr=.02)
        self.steps = 0

    def step(self, closure=None):
        self.steps += 1
        return super().step(closure)


class RetryScaler:
    def __init__(self):
        self.value, self.attempt = 128., 0

    def is_enabled(self):
        return False

    def scale(self, loss):
        return loss

    def unscale_(self, optimizer):
        pass

    def get_scale(self):
        return self.value

    def step(self, optimizer):
        if self.attempt:
            optimizer.step()

    def update(self):
        if not self.attempt:
            self.value /= 2
        self.attempt += 1


class R63Tests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_candidate_permutation_mask_padding_label_isolation_and_direct_logits(self):
        batch, model = fixture(), small_model(True).eval()
        captured = []
        hook = model.score_head.register_forward_hook(lambda module, args, output: captured.append(output))
        with torch.no_grad():
            original = model(batch)
            torch.testing.assert_close(original['logits'], captured[-1].squeeze(-1))
            permutation = torch.tensor([2, 0, 3, 1])
            permuted = model(dict(batch, pool_ids=batch['pool_ids'][permutation]))
            torch.testing.assert_close(permuted['logits'], original['logits'][:, permutation])
            torch.testing.assert_close(permuted['pair_margin'], original['pair_margin'][:, permutation][:, :, permutation])
            changed = dict(batch, costs=torch.full_like(batch['costs'], 1e9), ind=1-batch['ind'],
                           performance_target=torch.ones(2, 4), cmp_target=torch.zeros(2, 6),
                           route='unused', source_indices=torch.tensor([999, 123]))
            changed['node'] = batch['node'].clone()
            changed['node'][~batch['node_mask']] = 10000.
            torch.testing.assert_close(model(changed)['logits'], original['logits'])
            mask = torch.tensor([[True, True, False, True], [True, False, True, True]])
            masked = model(dict(batch, solver_mask=mask))
            ids = batch['pool_ids'][None].expand(2, -1).clone().masked_fill(~mask, 999)
            torch.testing.assert_close(model(dict(batch, pool_ids=ids, solver_mask=mask))['logits'], masked['logits'])
            self.assertTrue(torch.isneginf(masked['logits'][~mask]).all())
            torch.testing.assert_close(masked['pair_margin'][~(mask[:, :, None] & mask[:, None, :])],
                                       torch.zeros_like(masked['pair_margin'][~(mask[:, :, None] & mask[:, None, :])]))
        hook.remove()

    def test_eight_tied_rounds_static_memory_keeps_gradients_and_shared_initialization(self):
        batch = fixture()
        static, dynamic = small_model(False), small_model(True)
        for key, value in static.state_dict().items():
            torch.testing.assert_close(value, dynamic.state_dict()[key], rtol=0, atol=0)
        memories = []
        hook = static.cell.register_forward_pre_hook(lambda module, args: memories.append((args[0], args[1])))
        static(batch)['logits'].sum().backward()
        hook.remove()
        self.assertEqual(len(memories), 8)
        self.assertTrue(all(memory is memories[0][1] and memory.requires_grad for _, memory in memories))
        self.assertTrue(all(current is not memory for current, memory in memories[1:]))
        for parameter in (static.cell.source.weight, static.instance_encoder.coord_proj.weight,
                          static.solver_encoder.solver_emb.weight):
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(float(parameter.grad.abs().sum()), 0.)
        dynamic_memories = []
        hook = dynamic.cell.register_forward_pre_hook(lambda module, args: dynamic_memories.append(args[0] is args[1]))
        dynamic(batch)
        hook.remove()
        self.assertEqual(dynamic_memories, [True] * 8)

    def test_tail_weighted_one_step_and_entire_batch_rng_retry(self):
        batch = fixture(rows=7, candidates=3)
        torch.manual_seed(2)
        model = TinySelector()
        reference = copy.deepcopy(model)
        optimizer, reference_optimizer = CountingSGD(model), CountingSGD(reference)
        scaler = torch.amp.GradScaler('cuda', enabled=False)
        values = update_batch(model, optimizer, scaler, batch, .35, microbatch_size=3)
        first, second = reference(inputs_only(batch)), reference(inputs_only(batch))
        loss, _ = classification_objective(first, second, batch['ind'], .35)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(reference.parameters(), 1.)
        reference_optimizer.step()
        self.assertEqual(optimizer.steps, 1)
        self.assertEqual(values['microbatches'], 3)
        self.assertAlmostEqual(values['loss'], float(loss), places=6)
        for actual, expected in zip(model.parameters(), reference.parameters()):
            torch.testing.assert_close(actual, expected)
        self.assertTrue(all('costs' not in keys and 'ind' not in keys for keys, _ in model.calls))
        torch.manual_seed(9)
        ordinary = TinySelector(.2)
        retried = copy.deepcopy(ordinary)
        optimizer, retry_optimizer = CountingSGD(ordinary), CountingSGD(retried)
        torch.manual_seed(77)
        update_batch(ordinary, optimizer, scaler, batch, .35, microbatch_size=3)
        torch.manual_seed(77)
        values = update_batch(retried, retry_optimizer, RetryScaler(), batch, .35, microbatch_size=3)
        self.assertEqual(values['skipped'], 1)
        self.assertEqual(retry_optimizer.steps, 1)
        self.assertEqual(len(retried.calls), 12)
        for (_, initial), (_, replayed) in zip(retried.calls[:6], retried.calls[6:]):
            torch.testing.assert_close(initial, replayed, rtol=0, atol=0)
        for actual, expected in zip(retried.parameters(), ordinary.parameters()):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
