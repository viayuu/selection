"""Exact log-space elimination marginals for a homogeneous legal candidate pool.

Queries consume only per-subset elimination logits. Costs and native labels are
used solely by the separate ranking-supervision helpers below.
"""

from functools import lru_cache

import torch
from torch import Tensor


__all__ = ['subset_structure', 'champion_log_probs', 'full_ranking_nll',
           'stable_cost_order']


def _check_k(k: int) -> None:
    if isinstance(k, bool) or not isinstance(k, int) or not 1 <= k <= 10:
        raise ValueError('Candidate count K must be an integer in [1, 10]')


@lru_cache(maxsize=64)
def _cached_structure(k: int, device: torch.device) -> dict:
    ids = torch.arange(1 << k, dtype=torch.long, device=device)
    bits = 1 << torch.arange(k, dtype=torch.long, device=device)
    membership = (ids[:, None] & bits) != 0
    sizes = membership.sum(-1)
    state_ids = ids[sizes >= 2]
    state_rows = torch.full((1 << k,), -1, dtype=torch.long, device=device)
    state_rows[state_ids] = torch.arange(len(state_ids), device=device)
    layers = []
    for size in range(k, 1, -1):
        parents, children = ids[sizes == size], ids[sizes == size - 1]
        removed = (~membership[children]).nonzero(as_tuple=True)[1].reshape(
            len(children), k - size + 1)
        incoming_ids = children[:, None] | bits[removed]
        positions = torch.zeros_like(ids)
        positions[parents] = torch.arange(len(parents), device=device)
        layers.append(dict(size=size, parent_positions=positions[incoming_ids],
                           transition_indices=state_rows[incoming_ids] * k + removed))
    return dict(masks=membership[state_ids], state_ids=state_ids,
                state_rows=state_rows, layers=tuple(layers))


def subset_structure(k: int, device=None) -> dict:
    """Return cached, read-only subset tensors; bit j denotes local candidate j.

    ``masks`` is bool [S,K], ``state_ids`` is long [S], and states of size at
    least two are in ascending integer-bitset order, S = 2**K - K - 1.
    ``layers`` groups incoming transitions by descending parent subset size.
    Callers must not mutate the cached tensors or dictionaries.
    """
    _check_k(k)
    device = torch.device('cpu' if device is None else device)
    if device.type == 'cuda' and device.index is None:
        device = torch.device('cuda', torch.cuda.current_device())
    return _cached_structure(k, device)


def _logit_structure(elimination_logits: Tensor) -> dict:
    if elimination_logits.ndim != 3 or not elimination_logits.is_floating_point():
        raise ValueError('Elimination logits must be floating point [B,S,K]')
    k = elimination_logits.shape[-1]
    _check_k(k)
    if elimination_logits.shape[1] != (1 << k) - k - 1:
        raise ValueError('Expected S = 2**K - K - 1 subset-logit rows')
    return subset_structure(k, elimination_logits.device)


def _work_logits(elimination_logits: Tensor) -> Tensor:
    if elimination_logits.dtype in (torch.float16, torch.bfloat16):
        return elimination_logits.float()
    return elimination_logits


def _elimination_log_probs(logits: Tensor, structure: dict) -> Tensor:
    return logits.masked_fill(~structure['masks'], -torch.inf).log_softmax(-1)


def champion_log_probs(elimination_logits: Tensor) -> Tensor:
    """Sum ALL elimination paths, returning normalized log champion mass [B,K].

    Legal logits must be finite. Entries outside each subset are ignored, even
    if NaN/Inf. FP16/BF16 logits accumulate in FP32; FP64 stays FP64. There are
    only K-1 layer reductions, never a Python loop over individual subsets.
    """
    structure = _logit_structure(elimination_logits)
    logits = _work_logits(elimination_logits)
    if logits.shape[-1] == 1:
        return logits.sum(1)
    transitions = _elimination_log_probs(logits, structure).flatten(1)
    log_mass = logits.new_zeros((logits.shape[0], 1))
    for layer in structure['layers']:
        # Every gathered parent is reachable: no all-minus-infinity reductions.
        incoming = (log_mass[:, layer['parent_positions']]
                    + transitions[:, layer['transition_indices']])
        log_mass = torch.logsumexp(incoming, dim=-1)
    return log_mass


def full_ranking_nll(elimination_logits: Tensor, order_best_first: Tensor) -> Tensor:
    """Return [B] mean elimination NLL over K-1 teacher-forced ranking steps.

    The order contains local candidate indices, best first. Eliminate the worst
    remaining candidate until only the winner remains. K=1 has zero loss, with
    an autograd connection to the empty [B,0,1] logits.
    """
    structure = _logit_structure(elimination_logits)
    batch, _, k = elimination_logits.shape
    if order_best_first.dtype != torch.long or order_best_first.shape != (batch, k):
        raise ValueError('Ranking must be long [B,K] local candidate indices')
    if order_best_first.device != elimination_logits.device:
        raise ValueError('Ranking and logits must be on the same device')
    expected = torch.arange(k, device=elimination_logits.device).expand(batch, -1)
    if not torch.equal(order_best_first.sort(-1).values, expected):
        raise ValueError('Each ranking must be a permutation of [0, K)')
    logits = _work_logits(elimination_logits)
    if k == 1:
        return logits.sum((1, 2))
    remaining_ids = (1 << order_best_first).cumsum(-1)[:, 1:].flip(-1)
    removed = order_best_first[:, 1:].flip(-1)
    indices = structure['state_rows'][remaining_ids] * k + removed
    transitions = _elimination_log_probs(logits, structure).flatten(1)
    return -transitions.gather(1, indices).mean(-1)


@torch.no_grad()
def stable_cost_order(costs_fp64: Tensor, native_winner: Tensor, pool_ids: Tensor) -> Tensor:
    """Return best-first local indices from exact FP64 supervision [B,K].

    Native winners are local indices [B]; pool_ids are unique global IDs [K].
    Sort by exact cost, then global ID. Promote the native winner to first only
    after verifying it belongs to the EXACT minimum-cost tie block.
    """
    if costs_fp64.dtype != torch.float64 or costs_fp64.ndim != 2:
        raise ValueError('Costs must be FP64 [B,K]; never rank FP32 costs')
    batch, k = costs_fp64.shape
    _check_k(k)
    if native_winner.dtype != torch.long or native_winner.shape != (batch,):
        raise ValueError('Native winner must be long [B] local indices')
    if pool_ids.dtype != torch.long or pool_ids.shape != (k,):
        raise ValueError('Pool IDs must be long [K] unique global IDs')
    if native_winner.device != costs_fp64.device or pool_ids.device != costs_fp64.device:
        raise ValueError('Costs, native winners and pool IDs must share a device')
    if pool_ids.unique().numel() != k:
        raise ValueError('Pool global IDs must be unique')
    if not torch.isfinite(costs_fp64).all():
        raise ValueError('All legal candidate costs must be finite')
    if ((native_winner < 0) | (native_winner >= k)).any():
        raise ValueError('Native winner index is outside the candidate pool')
    native_cost = costs_fp64.gather(1, native_winner[:, None]).squeeze(1)
    if not torch.equal(native_cost, costs_fp64.amin(-1)):
        raise ValueError('Native winner is strictly above the exact FP64 minimum')
    by_id = pool_ids.argsort(stable=True).expand(batch, -1)
    by_cost = costs_fp64.gather(1, by_id).argsort(dim=-1, stable=True)
    order = by_id.gather(1, by_cost)
    native_first = (order != native_winner[:, None]).long().argsort(dim=-1, stable=True)
    return order.gather(1, native_first)
