"""Training-only directed route supervision for the R53A pair specialist."""

import math

import torch
from torch import nn
from torch.nn import functional as F

from .pair_specialist import INPUT_FIELDS, PairSpecialist
from .V4Model import _masked_max, _masked_mean


class SuccessorHead(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.distance = nn.Sequential(nn.Linear(1, 32), nn.GELU(), nn.Linear(32, 1))
        self.end = nn.Sequential(nn.Linear(2 * dim, dim), nn.GELU(), nn.Linear(dim, 1))
        self.dim = dim

    def forward(self, nodes, mask, xy):
        customers, valid = nodes[:, 1:], mask[:, 1:]
        count = customers.shape[1]
        logits = torch.matmul(self.query(customers), self.key(customers).transpose(1, 2)) / math.sqrt(self.dim)
        # Only observable coordinates set this scale; never route length or oracle cost.
        distance = torch.cdist(xy.float(), xy.float())
        pairs = mask[:, :, None] & mask[:, None, :]
        scale = (distance * pairs).sum((1, 2)) / (pairs.sum((1, 2)) - mask.sum(1)).clamp_min(1)
        relative = distance[:, 1:, 1:] / scale.clamp_min(1e-8)[:, None, None]
        logits = logits + self.distance(relative[..., None].to(nodes.dtype)).squeeze(-1)
        forbidden = ~valid[:, None, :] | torch.eye(count, dtype=torch.bool, device=nodes.device)[None]
        logits = logits.masked_fill(forbidden, float('-inf'))
        depot = nodes[:, :1].expand(-1, count, -1)
        end = self.end(torch.cat((customers, depot), -1))
        return torch.cat((logits, end), -1)


class RouteSupervisedSpecialist(PairSpecialist):
    def __init__(self, params):
        super().__init__(params)
        dim = int(params.get('embedding_dim', 128))
        self.successor_heads = nn.ModuleList([SuccessorHead(dim), SuccessorHead(dim)])

    def forward(self, batch, return_aux=False):
        clean = {key: batch[key] for key in INPUT_FIELDS if key in batch}
        nodes, mask = self.instance_encoder(clean)
        mean = _masked_mean(nodes, mask)
        maximum = _masked_max(nodes, mask)
        size = torch.log1p(batch['n'].float()).unsqueeze(-1) / 6
        z = self.head(torch.cat((mean, maximum, size), -1)).squeeze(-1)
        if not return_aux:
            return z
        aux = torch.stack([head(nodes, mask, clean['node'][..., :2]) for head in self.successor_heads], 1)
        return dict(logit=z, successor_logits=aux, customer_mask=mask[:, 1:])


def successor_objective(logits, successor, customer_mask):
    """Stored target: original customer ID 1..N, zero=END, -100=padding."""
    count = logits.shape[-2]
    target = successor[..., :count].long()
    if target.shape != logits.shape[:-1]:
        raise ValueError('Successor target/solver/node axes do not align')
    valid = customer_mask[:, None, :].expand_as(target)
    if ((target < 0) & valid).any() or ((target > count) & valid).any():
        raise ValueError('Missing or out-of-range route supervision')
    self_ids = torch.arange(1, count + 1, device=target.device)
    if ((target == self_ids) & valid).any():
        raise ValueError('A customer cannot succeed itself')
    classes = torch.where(target == 0, count, target - 1).masked_fill(~valid, -100)
    loss = F.cross_entropy(logits.float().flatten(0, 2), classes.flatten(), reduction='none', ignore_index=-100)
    loss = loss.reshape_as(target)
    clients = customer_mask.sum(1).clamp_min(1)
    # N-1 other customers plus END = N valid classes per customer.
    normalizer = clients.float().log().clamp_min(1.)
    per_method = loss.sum(-1) / clients[:, None] / normalizer[:, None]
    prediction = logits.argmax(-1)
    accuracy = ((prediction == classes) & valid).sum((0, 2)) / valid.sum((0, 2)).clamp_min(1)
    return per_method.mean(), dict(moel_loss=per_method[:, 0].mean(), mtl_loss=per_method[:, 1].mean(),
        moel_accuracy=accuracy[0], mtl_accuracy=accuracy[1], customers=valid[:, 0].sum())
