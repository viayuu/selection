"""R85D prototype: train-only random candidate subsets with the R77 backbone.

Runner API::

    model = make_candidate_model(params, state_dict=None)
    output = model(inputs_only(batch), aux_solver_mask=subset_mask)
    targets = subset_targets(batch, subset_mask)

The objective draws ONE label-independent legal subset per logical batch and
reuses it for both dropout forwards and every microbatch. This module never
samples candidates. A mask may be [K], [1, K], or [B, K]; it always refers to
the original pool slots, not global solver IDs. Training subsets retain 2..K-1
legal candidates. Evaluation ignores the auxiliary mask altogether.

Full outputs retain the original logits, utility, pair_margin and solver_mask.
The training-only additions are aux_subset_logits, aux_subset_utility,
aux_subset_pair_margin and aux_subset_solver_mask. Both branches use the SAME
initial instance and solver encodings; each runs the original shared joint
layers, pooling, decoder and score MLP. No parameters or buffers are added.

subset_targets returns ind (an original-pool slot), solver_mask, cmp_target,
cmp_valid, cmp_true3, full_ranking and subset_ranking. Ranking tensors contain
original-pool slots, best first, with excluded slots at the end. Only legal or
retained leading entries are meaningful. Cost ties use ascending global IDs.
It uses the existing original-FP64-derived upper-triangle comparison labels,
never cached FP32 costs or the saved full-pool winner. It does not reconstruct
numeric costs or risk targets. Labels must not be passed as encoder features.
"""

import torch

from .V4Model import _masked_max, _masked_mean
from .pairwise_objective import inputs_only
from .pairwise_selector import ScoreDifferenceSelector, maximin_logits, rank_by_solver_ids


SELECTION_HEAD = 'original_score_mlp_maximin'
CANDIDATE_AUX = dict(train_only=True, min_size=2, max_size='K-1',
                     mask_source='objective_random_legal', shared_initial_encoding=True)


def _candidate_layout(batch, rows, device):
    ids = batch['pool_ids'].to(device=device)
    if ids.ndim == 1:
        ids = ids[None].expand(rows, -1)
    if ids.ndim != 2 or ids.shape[0] != rows or not ids.shape[1]:
        raise ValueError('pool_ids must be [K] or [B, K] with a nonempty pool')
    mask = batch.get('solver_mask')
    mask = (torch.ones_like(ids, dtype=torch.bool) if mask is None else
            mask.to(device=device, dtype=torch.bool))
    if mask.shape != ids.shape or not mask.any(-1).all():
        raise ValueError('solver_mask must match pool_ids and retain a legal candidate per row')
    return ids, mask


def _subset_mask(value, legal, proper=False):
    if not torch.is_tensor(value) or value.dtype != torch.bool:
        raise ValueError('aux_solver_mask must be a boolean tensor independent of labels')
    mask = value.to(device=legal.device)
    if mask.ndim == 1:
        mask = mask[None]
    if mask.ndim != 2 or mask.shape[1] != legal.shape[1] or mask.shape[0] not in (1, legal.shape[0]):
        raise ValueError('aux_solver_mask must be [K], [1, K] or [B, K] in original pool slots')
    mask = mask.expand_as(legal)
    count = mask.sum(-1)
    valid = ~((mask & ~legal).any(-1)) & (count >= (2 if proper else 1))
    if proper:
        valid &= count < legal.sum(-1)
    if not valid.all():
        raise ValueError('A training subset needs 2..K-1 legal candidates; target subsets must be nonempty')
    return mask


@torch.no_grad()
def subset_targets(batch, subsetmask):
    """Recover strict stable cost order from FP64-derived comparisons only."""
    target, valid = batch['cmp_target'].detach(), batch['cmp_valid'].detach()
    if target.ndim != 2 or valid.shape != target.shape or valid.dtype != torch.bool:
        raise ValueError('Expected [B, K*(K-1)/2] comparison labels and a boolean cmp_valid')
    ids, legal = _candidate_layout(batch, target.shape[0], target.device)
    subset = _subset_mask(subsetmask, legal)
    k = ids.shape[-1]
    i, j = torch.triu_indices(k, k, 1, device=target.device)
    if target.shape[-1] != i.numel() or valid.device != target.device:
        raise ValueError('Comparison labels must cover the original pool upper triangle on one device')
    pair_legal = legal[:, i] & legal[:, j]
    valid = valid & pair_legal
    if ((target[valid] != 0) & (target[valid] != 1)).any():
        raise ValueError('Strict comparison targets must be binary')
    tied = pair_legal & ~valid
    precedes = torch.zeros((len(ids), k, k), dtype=torch.bool, device=target.device)
    strict = torch.zeros_like(precedes)
    strict[:, i, j] = valid & (target == 1)
    strict[:, j, i] = valid & (target == 0)
    precedes[:, i, j] = strict[:, i, j] | (tied & (ids[:, i] < ids[:, j]))
    precedes[:, j, i] = strict[:, j, i] | (tied & (ids[:, j] < ids[:, i]))
    rank = precedes.sum(1).float()
    full_ranking = rank_by_solver_ids((-rank).masked_fill(~legal, float('-inf')), ids)
    subset_ranking = rank_by_solver_ids((-rank).masked_fill(~subset, float('-inf')), ids)
    # Count strictly cheaper retained methods, preserving all ties at Top3's boundary.
    cheaper = (strict & subset[:, :, None]).sum(1)
    true3 = subset & (cheaper < subset.sum(-1, keepdim=True).clamp_max(3))
    return dict(ind=subset_ranking[:, 0], solver_mask=subset, cmp_target=target,
                cmp_valid=valid & subset[:, i] & subset[:, j],
                cmp_true3=true3[:, i] & true3[:, j],
                full_ranking=full_ranking, subset_ranking=subset_ranking)


class CandidateSubsetSelector(ScoreDifferenceSelector):
    """The exact original full-pool model, with a shared training-only branch."""

    def __init__(self, **params):
        params = dict(params)
        if (params.get('solver_representation', 'handcrafted') != 'handcrafted'
                or params.get('pair_mode', 'score_difference') != 'score_difference'
                or params.get('selection_head', SELECTION_HEAD) != SELECTION_HEAD):
            raise ValueError('R85D preserves the original handcrafted score-MLP/maximin backbone')
        if params.get('candidate_aux', CANDIDATE_AUX) != CANDIDATE_AUX:
            raise ValueError('candidate_aux must describe shared, train-only random legal subsets')
        params.update(selection_head=SELECTION_HEAD, candidate_aux=dict(CANDIDATE_AUX))
        super().__init__(**params)

    def _score_branch(self, nodes, solvers, node_mask, mask):
        for layer in self.joint_layers:
            nodes, solvers = layer(nodes, solvers, node_mask, mask)
        h = self.pool(torch.cat([_masked_mean(nodes, node_mask), _masked_max(nodes, node_mask)], dim=-1))
        context = self.decoder(h[:, None, :], solvers, solvers, key_mask=mask)
        features = torch.cat([h[:, None, :].expand_as(solvers), context.expand_as(solvers), solvers], dim=-1)
        utility = self.score_head(features).squeeze(-1).float()
        margins = utility[:, :, None] - utility[:, None, :]
        margins = margins.masked_fill(~(mask[:, :, None] & mask[:, None, :]), 0.)
        return dict(logits=maximin_logits(margins, mask), pair_margin=margins,
                    solver_mask=mask, utility=utility)

    def forward(self, batch, aux_solver_mask=None):
        batch = inputs_only(batch)
        if not self.training or aux_solver_mask is None:
            return super().forward(batch)
        nodes, node_mask = self.instance_encoder(batch)
        nodes = self.adapt_nodes(nodes, node_mask, batch)
        ids, mask = _candidate_layout(batch, nodes.size(0), nodes.device)
        ids = ids.masked_fill(~mask, 0)
        solvers = self.solver_encoder(ids).masked_fill(~mask.unsqueeze(-1), 0.)
        # Finish the original prediction before any auxiliary branch dropout.
        output = self._score_branch(nodes, solvers, node_mask, mask)
        subset = _subset_mask(aux_solver_mask, mask, proper=True)
        auxiliary = self._score_branch(nodes, solvers.masked_fill(~subset.unsqueeze(-1), 0.),
                                       node_mask, subset)
        output.update({'aux_subset_' + key: value for key, value in auxiliary.items()})
        return output


def make_candidate_model(params, state_dict=None):
    """Fresh original initialization or strict reconstruction without new weights."""
    model = CandidateSubsetSelector(**params)
    if state_dict is not None:
        model.load_state_dict(state_dict, strict=True)
    return model
