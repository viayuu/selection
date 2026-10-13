"""The single focused synthetic R72 check; no data or real checkpoint access."""

import time

import torch
from torch import nn

from .r72_consensus import FrozenConsensus, INPUT_KEYS, doubled_midranks


class FixtureMember(nn.Module):
    def __init__(self, values, permutation):
        super().__init__()
        self.register_buffer('values', torch.tensor(values, dtype=torch.float32))
        self.register_buffer('permutation', torch.tensor(permutation))
        self.calls = 0

    def forward(self, batch):
        assert set(batch) <= INPUT_KEYS
        self.calls += 1
        ids = batch['pool_ids']
        if ids.ndim == 1:
            ids = ids[None].expand(len(batch['n']), -1)
        mask = batch['solver_mask']
        # ID-keyed fixture scores make input and returned-column permutations independent.
        scores = self.values[:, ids.clamp_min(0)]
        if scores.ndim == 3:
            scores = scores.diagonal(dim1=0, dim2=1).T
        p = self.permutation
        return dict(logits=scores[:, p], pool_ids=ids[:, p], solver_mask=mask[:, p])


def run_check():
    started = time.monotonic()
    torch.set_num_threads(1)
    members = [FixtureMember([[0, 4, 0, 4], [0, 1, 0, 0]], p)
               for p in ([0, 1, 2, 3], [2, 0, 3, 1], [3, 2, 1, 0])]
    model = FrozenConsensus(members)
    batch = dict(pool_ids=torch.tensor([3, -1, 1, 0]),
                 solver_mask=torch.tensor([[True, False, True, True], [True, False, True, True]]),
                 n=torch.tensor([5, 7]), problem_id=0, costs=object(), ind=object(),
                 performance_target=object(), route=object(), unknown_target=object())
    first = model(batch)
    assert first['canonical_pool_ids'].tolist() == [[0, 1, 3], [0, 1, 3]]
    assert first['doubled_midranks'][:, 0].tolist() == [[6, 3, 3]] * 3
    assert first['Q'].tolist() == [[18, 9, 9], [15, 6, 15]]
    assert first['pred_global_id'].tolist() == [1, 1]
    assert first['pred'].tolist() == [2, 2]
    assert torch.isneginf(first['logits'][:, 1]).all()
    assert torch.equal(first['canonical_logits'], -first['Q'].float() / 6)
    reordered = dict(batch, pool_ids=batch['pool_ids'][torch.tensor([2, 3, 0, 1])],
                     solver_mask=batch['solver_mask'][:, [2, 3, 0, 1]], costs='changed', ind='changed')
    second = model(reordered)
    assert torch.equal(first['Q'], second['Q'])
    assert torch.equal(first['pred_global_id'], second['pred_global_id'])
    assert doubled_midranks(torch.tensor([[5., 5., 5., float('nan')]]),
                           torch.tensor([[True, True, True, False]])).tolist() == [[4, 4, 4, 0]]
    assert all(member.calls == 2 for member in members)
    assert not any(p.requires_grad for p in model.parameters())
    return dict(passed=True, seconds=time.monotonic()-started, synthetic_only=True,
                checks=['input/returned column permutations', 'padding/legal masks', 'exact-score midranks',
                        'integer sum', 'smallest global-ID final ties', 'target exclusion'],
                real_checkpoint_loads=0, real_data_forwards=0, GPU_operations=0)
