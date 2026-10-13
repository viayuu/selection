"""Frozen R72 three-policy consensus; supervision never enters this interface."""

import torch
from torch import nn


INPUT_KEYS = frozenset(('problem_id', 'kind', 'node', 'node_mask', 'd_in', 'matrix',
                        'cbits', 'problem_desc', 'coord_dist', 'pool_ids', 'mask', 'n',
                        'node_geom', 'solver_mask'))
MEMBER_NAMES = ('R60B39', 'R67B36_global76', 'R70B23_global63')
INFERENCE_RULE = ('Exactly three frozen FP32 policies; ascending global-ID full legal pools; '
                  'q=2*strictly_higher+equal_including_self+1; integer Q=sum(q); '
                  'minimum Q, ties smallest global solver ID; reporting z=-Q/6')


def model_inputs(batch):
    return {key: value for key, value in batch.items() if key in INPUT_KEYS}


def align_legal(scores, ids, mask):
    if scores.ndim != 2 or ids.dtype != torch.long:
        raise ValueError('Returned scores must be matrices with integer global IDs')
    ids = ids.to(scores.device)
    if ids.ndim == 1:
        ids = ids[None].expand_as(scores)
    mask = mask.to(device=scores.device, dtype=torch.bool)
    if ids.shape != scores.shape or mask.shape != scores.shape or not mask.any(-1).all():
        raise ValueError('Every query must retain its nonempty complete legal pool')
    if not torch.isfinite(scores[mask]).all() or (ids[mask] < 0).any():
        raise ValueError('Nonfinite legal returned scores or invalid global IDs; stop')
    sentinel = torch.iinfo(torch.long).max
    order = ids.masked_fill(~mask, sentinel).argsort(dim=-1, stable=True)
    width = int(mask.sum(-1).max())
    order = order[:, :width]
    legal = mask.gather(1, order)
    canonical_ids = ids.gather(1, order).masked_fill(~legal, sentinel)
    if ((canonical_ids[:, 1:] == canonical_ids[:, :-1]) & legal[:, 1:]).any():
        raise ValueError('Duplicate legal global solver ID')
    aligned = scores.gather(1, order).masked_fill(~legal, float('-inf'))
    return aligned, canonical_ids, legal, order


def doubled_midranks(scores, mask):
    legal_opponent = mask[:, None, :]
    higher = ((scores[:, None, :] > scores[:, :, None]) & legal_opponent).sum(-1)
    equal = ((scores[:, None, :] == scores[:, :, None]) & legal_opponent).sum(-1)
    return (2 * higher + equal + 1).masked_fill(~mask, 0)


class FrozenConsensus(nn.Module):
    def __init__(self, members):
        super().__init__()
        if len(members) != 3:
            raise ValueError('R72 requires exactly the three fixed complete members')
        self.members = nn.ModuleList(members)
        self.requires_grad_(False)
        self.eval()

    @torch.no_grad()
    def forward(self, batch):
        if self.training or any(member.training for member in self.members):
            raise ValueError('All R72 members must remain frozen in evaluation mode')
        clean = model_inputs(batch)
        ids = clean['pool_ids']
        rows = len(clean['n'])
        width = ids.shape[-1]
        base_mask = clean.get('solver_mask', torch.ones((rows, width), dtype=torch.bool, device=ids.device))
        zeros = torch.zeros((rows, width), dtype=torch.float32, device=ids.device)
        _, canonical_ids, legal, order = align_legal(zeros, ids, base_mask)
        aligned_members, ranks = [], []
        for member in self.members:
            output = member(clean)
            scores = output['logits']
            if scores.dtype != torch.float32:
                raise ValueError('Every member must return its original FP32 logits')
            aligned, member_ids, member_mask, _ = align_legal(
                scores, output.get('pool_ids', ids), output.get('solver_mask', base_mask))
            if not torch.equal(member_ids, canonical_ids) or not torch.equal(member_mask, legal):
                raise ValueError('Member candidate mismatch; no member/candidate may be dropped')
            aligned_members.append(aligned)
            ranks.append(doubled_midranks(aligned, legal))
        total = torch.stack(ranks).sum(0)
        canonical_logits = (-total.float() / 6).masked_fill(~legal, float('-inf'))
        logits = zeros.new_full(zeros.shape, float('-inf')).scatter(1, order, canonical_logits)
        minimum = total.masked_fill(~legal, torch.iinfo(torch.long).max).argmin(-1)
        return dict(logits=logits, solver_mask=base_mask, Q=total, doubled_midranks=torch.stack(ranks),
                    canonical_logits=canonical_logits, canonical_pool_ids=canonical_ids,
                    canonical_mask=legal, member_returned_scores=torch.stack(aligned_members),
                    pred=order.gather(1, minimum[:, None]).squeeze(1),
                    pred_global_id=canonical_ids.gather(1, minimum[:, None]).squeeze(1))


def restore_member(name, params, state):
    if name == MEMBER_NAMES[0]:
        from .solver_code_encoder import make_r45_model
        model = make_r45_model(params, state_dict=state)
    elif name == MEMBER_NAMES[1]:
        from .r64_model import NativeUtilitySelector
        model = NativeUtilitySelector(**params)
    elif name == MEMBER_NAMES[2]:
        from .r69_model import make_bank
        initial = {key[len('branches.0.'):]: value for key, value in state.items()
                   if key.startswith('branches.0.')}
        model = make_bank(params, initial)
    else:
        raise ValueError('Unknown R72 member')
    model.load_state_dict(state, strict=True)
    return model.float().requires_grad_(False).eval()


def restore_composite(bundle, device='cpu'):
    if bundle['member_order'] != list(MEMBER_NAMES) or bundle['inference_rule'] != INFERENCE_RULE:
        raise ValueError('The fixed composite membership and rule cannot change')
    members = [restore_member(name, bundle['members'][name]['args']['model_params'],
                              bundle['members'][name]['model']) for name in MEMBER_NAMES]
    return FrozenConsensus(members).to(device).eval()
