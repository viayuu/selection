"""Training-only performance records; input-derived identities exclude related queries."""

import hashlib

import numpy as np
import torch
from torch import nn

from .V4Model import _masked_max, _masked_mean
from .multitask_probe import gather_batch, state_hash
from .pairwise_objective import inputs_only


def instance_summary(nodes, mask, sizes):
    return torch.cat([_masked_mean(nodes, mask), _masked_max(nodes, mask),
                      sizes.float().log1p()[:, None] / 6.], -1).float()


def base_instance_ids(batch):
    """Conservatively group reordered/D4-related coordinate inputs, never labels."""
    kind = batch['kind']
    values = batch['node' if kind == 'coord' else 'matrix'].detach().cpu().numpy()
    lengths = batch['n'].detach().cpu().tolist()
    identities = []
    for value, n in zip(values, lengths):
        if kind == 'matrix':
            contents = np.ascontiguousarray(value[:n, :n], dtype='<f4').tobytes()
        else:
            node = np.zeros((n, 8), dtype=np.float64)
            node[:, :value.shape[-1]] = value[:n, :8]
            x, y = node[:, 0].copy(), node[:, 1].copy()
            variants = []
            for xx, yy in ((x, y), (1-x, y), (x, 1-y), (1-x, 1-y),
                           (y, x), (1-y, x), (y, 1-x), (1-y, 1-x)):
                node[:, 0], node[:, 1] = xx, yy
                rounded = np.rint(node * 1e6).astype('<i8')
                order = np.lexsort(rounded.T[::-1])
                variants.append(rounded[order].tobytes())
            contents = min(variants)
        prefix = str((int(batch['problem_id']), kind, n)).encode()
        digest = hashlib.sha256(prefix + contents).digest()
        identities.append(int.from_bytes(digest[:8], 'little') & ((1 << 63)-1))
    return torch.tensor(identities, dtype=torch.long, device=batch['n'].device)


class BehaviorBank(nn.Module):
    def __init__(self, summary, gap, winner_ids, pool_ids, nodes, base_ids, solver_mask=None):
        super().__init__()
        if gap.dtype != torch.float64:
            raise ValueError('Behavior gaps must be constructed from original FP64 costs')
        if solver_mask is None:
            solver_mask = torch.ones_like(gap, dtype=torch.bool)
        for key, value in dict(summary=summary.detach().float(), gap=gap.detach(),
              gap_input=torch.log1p(gap / .01).float(), winner_ids=winner_ids.long(),
              pool_ids=pool_ids.long(), nodes=nodes.long(), base_ids=base_ids.long(),
              solver_mask=solver_mask.bool()).items():
            self.register_buffer(key, value)

    @classmethod
    def from_training(cls, loader, raw, dim):
        costs = raw['costs']
        if costs.dtype != np.float64 or not np.isfinite(costs).all() or (costs.min(-1) <= 0).any():
            raise ValueError('Memory requires finite original FP64 training costs and positive oracle')
        gap = (costs - costs.min(-1, keepdims=True)) / costs.min(-1, keepdims=True)
        device = loader.batch['n'].device
        ids = torch.as_tensor(raw['pool_ids'], device=device)
        return cls(torch.zeros(loader.size, dim, device=device), torch.from_numpy(gap).to(device),
                   ids[torch.as_tensor(raw['winner'], device=device)], ids, loader.batch['n'], loader.batch['base_id'])


def attach_query_ids(loaders):
    report = {}
    for problem, loader in loaders.items():
        loader.batch['base_id'] = base_instance_ids(loader.batch)
        loader.batch['instance_index'] = torch.arange(loader.size, device=loader.batch['n'].device)
        unique, counts = loader.batch['base_id'].unique(return_counts=True)
        report[problem] = dict(instances=loader.size, unique_base_ids=len(unique),
                               related_rows=int(counts[counts > 1].sum()),
                               base_id_sha256=state_hash(loader.batch['base_id']))
    return report


@torch.no_grad()
def refresh_memory(model, train_loaders, batch_size=128):
    """Only detached historical graph summaries are cached; keys stay trainable."""
    was_training = model.training
    model.eval()
    try:
        for problem, loader in train_loaders.items():
            order = loader.lengths.argsort(stable=True)
            for indices in order.split(batch_size):
                batch = inputs_only(gather_batch(loader, indices))
                nodes, mask = model.instance_encoder(batch)
                summary = instance_summary(nodes, mask, batch['n'])
                model.memory[problem].summary[indices.to(summary.device)] = summary
    finally:
        model.train(was_training)
    return dict(encoder_sha256=state_hash(model.instance_encoder.state_dict()),
                summary_sha256=state_hash({p: bank.summary for p, bank in model.memory.items()}),
                instances={p: len(bank.summary) for p, bank in model.memory.items()},
                mode='eval/no_grad; train references only; original row order retained')


def shuffled_behavior(memory, seed=4702):
    """Permute whole behavior rows within exact problem/size; keep keys unchanged."""
    generator = torch.Generator().manual_seed(seed)
    original, report = {}, {}
    for problem, bank in memory.items():
        original[problem] = {k: getattr(bank, k).clone() for k in ('gap', 'gap_input', 'winner_ids')}
        permutation = torch.arange(len(bank.nodes))
        for n in bank.nodes.cpu().unique(sorted=True):
            rows = (bank.nodes.cpu() == n).nonzero().flatten()
            if len(rows) > 1:
                order = rows[torch.randperm(len(rows), generator=generator)]
                permutation[order] = order.roll(1)
        for key in original[problem]:
            getattr(bank, key).copy_(original[problem][key][permutation.to(bank.nodes.device)])
        report[problem] = dict(rows=len(permutation), moved=int((permutation != torch.arange(len(permutation))).sum()),
                               permutation=permutation.tolist())
    return original, report


def restore_behavior(memory, original):
    for problem, values in original.items():
        for key, value in values.items():
            getattr(memory[problem], key).copy_(value)
