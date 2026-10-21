"""R84: the R77 backbone with a direct attention distribution over methods.

The existing instance encoder, ID/34-feature solver encoder, synchronous joint
layers and mean/max pooling stay intact. The pooled instance is the query and
the final solver tokens are the keys. Scaled dot products are averaged across
heads *before* softmax. There is no value/context output, final score MLP,
maximin transform, learned head aggregation or probability calibration.

``make_r84_model(params, state_dict=None)`` is the runner/report factory. Pass
the usual R77 model parameters; the factory records ``selection_head`` as
``direct_attention``. A state_dict must belong to this architecture and loads
strictly. The removed baseline decoder/score-head weights are not accepted.

Forward returns raw ``logits`` for CE/risk/R-Drop, ``probabilities`` for callers
that need them, finite ``utility`` (zero on padding), antisymmetric
``pair_margin`` and ``solver_mask``. Comparison supervision uses differences of
the very same attention scores; it cannot replace the inference ranking.
Batch size 1024 and seed 2 are runner settings, not properties of this module.

The original RDropObjective works for nonempty batches. A distributed runner
with idle ranks must anchor its empty-rank zero loss to existing parameters,
not to the baseline's deleted ``score_head``.
"""

import math

import torch
from torch import nn

from .dual_stream import DualStreamSelector


class AttentionLogitDecoder(nn.Module):
    """One instance query attends directly to legal method keys."""

    def __init__(self, attention):
        super().__init__()
        self.head_num = attention.head_num
        self.qkv_dim = attention.qkv_dim
        # Preserve the baseline's query/key initialization and parameter names.
        self.Wq = attention.Wq
        self.Wk = attention.Wk

    def forward(self, query, methods):
        b, k, _ = methods.shape
        q = self.Wq(query).reshape(b, self.head_num, self.qkv_dim)
        keys = self.Wk(methods).reshape(b, k, self.head_num, self.qkv_dim)
        keys = keys.transpose(1, 2)
        # Keep dot-product reduction in FP32 under the runner's AMP context.
        with torch.autocast(device_type=query.device.type, enabled=False):
            scores = (q.float().unsqueeze(2) * keys.float()).sum(-1)
            return (scores / math.sqrt(self.qkv_dim)).mean(1)


class AttentionProbabilitySelector(DualStreamSelector):
    """Shared R77 encoding, followed only by attention logits and softmax."""

    def __init__(self, **params):
        params = dict(params)
        if params.get('solver_representation', 'handcrafted') != 'handcrafted':
            raise ValueError('R84 keeps the R77 handcrafted-feature solver branch')
        if params.get('pair_mode', 'score_difference') != 'score_difference':
            raise ValueError('R84 comparison margins are direct attention-score differences')
        if params.get('selection_head', 'direct_attention') != 'direct_attention':
            raise ValueError('R84 requires selection_head=direct_attention')
        params['selection_head'] = 'direct_attention'
        super().__init__(**params)
        self.decoder = AttentionLogitDecoder(self.decoder)
        del self.score_head

    def forward(self, batch):
        _, solvers, h, _, mask = self.encode_state(batch)
        utility = self.decoder(h[:, None, :], solvers).masked_fill(~mask, 0.)
        logits = utility.masked_fill(~mask, float('-inf'))
        margins = utility[:, :, None] - utility[:, None, :]
        margins = margins.masked_fill(~(mask[:, :, None] & mask[:, None, :]), 0.)
        return dict(logits=logits, probabilities=logits.softmax(-1), utility=utility,
                    pair_margin=margins, solver_mask=mask)


def make_r84_model(params, state_dict=None):
    """Construct a fresh model, or strictly reconstruct an R84 model offline."""
    model = AttentionProbabilitySelector(**params)
    if state_dict is not None:
        model.load_state_dict(state_dict, strict=True)
    return model
