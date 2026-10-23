"""Differentiable, masked execution of the three pinned R102-B encoders.

Only native builders/weights and input validators are reused, never the old
frozen forwards. Adapters keep the original Parameter objects and state keys.
"""

import copy
import sys

import torch
from torch import nn
from torch.nn import functional as F

from . import r102_donor as native
from . import r94_donor as reld


ARMS = ('A_frozen', 'B_finetune')
PROBLEM_NAMES = native.PROBLEM_NAMES
SUPPORTED_PROBLEMS = native.SUPPORTED_PROBLEMS
WIDTH_BY_PROBLEM = copy.deepcopy(native.WIDTH_BY_PROBLEM)
INPUT_ALLOWLIST = native.INPUT_ALLOWLIST
component_for_problem = native.component_for_problem
SPEC = dict(
    format='r106_differentiable_native_v1',
    weights='all three original pinned R102-B pretrained encoders in both arms',
    native_precision=copy.deepcopy(reld.SPEC['native_precision']),
    execution='one masked padded execution per homogeneous microbatch, no exact-N loop',
    TSP='ELG128: masked attention and valid-node affine InstanceNorm',
    ATSP='MatNet FROM256 concat TO256; original distance mixed-score, masked InstanceNorm',
    VRP='ReLD128 norm=none; original eval node-level input-choice topk2 MoE',
    atsp_roles=copy.deepcopy(native.ROLE_SPEC),
    dropout=False, decoder=False, native_auxiliary_loss=False,
    mode='native always eval; eval does not disable B autograd',
    padding='preserve physical node order; invalid inputs sanitized and output zero',
    comparison='numerical agreement only, not bitwise equivalence',
)


def input_only(batch):
    if any(key.startswith('nco_') for key in batch):
        raise ValueError('R106 rejects external nco caches; native execution is online')
    return {key: batch[key] for key in INPUT_ALLOWLIST if key in batch}


def _heads(value, count):
    return value.reshape(value.shape[0], value.shape[1], count, -1).transpose(1, 2)


def _attention(layer, value, mask):
    heads = layer.model_params['head_num']
    q, k, v = (_heads(projection(value), heads)
               for projection in (layer.Wq, layer.Wk, layer.Wv))
    attended = F.scaled_dot_product_attention(
        q, k, v, attn_mask=mask[:, None, None, :], dropout_p=0.)
    return attended.transpose(1, 2).reshape(*value.shape[:2], -1)


def masked_instance_norm(value, mask, norm):
    """Native biased spatial variance, eps and affine, excluding padded nodes."""
    if not isinstance(norm, nn.InstanceNorm1d) or norm.track_running_stats:
        raise ValueError('R106 adapter requires the pinned stateless InstanceNorm')
    value = value.masked_fill(~mask[..., None], 0.)
    count = mask.sum(1, keepdim=True).unsqueeze(-1).to(value.dtype)
    mean = value.sum(1, keepdim=True) / count
    centered = (value - mean).masked_fill(~mask[..., None], 0.)
    variance = centered.square().sum(1, keepdim=True) / count
    result = centered * torch.rsqrt(variance + norm.eps)
    if norm.affine:
        result = result * norm.weight + norm.bias
    return result.masked_fill(~mask[..., None], 0.)


def masked_elg(encoder, xy, mask):
    hidden = encoder.embedding(xy).masked_fill(~mask[..., None], 0.)
    for layer in encoder.layers:
        attention = layer.multi_head_combine(_attention(layer, hidden, mask))
        first = masked_instance_norm(hidden + attention, mask, layer.addAndNormalization1.norm)
        hidden = masked_instance_norm(first + layer.feedForward(first), mask,
                                      layer.addAndNormalization2.norm)
    return hidden


def _eval_moe(moe, hidden, mask):
    if (moe.training or moe.routing_level != 'node' or moe.routing_method != 'input_choice'
            or moe.k != 2):
        raise ValueError('Require the unchanged native eval node-level topk2 MoE')
    valid = hidden[mask]
    # Reuse native top-k, dispatcher, expert FFNs and combine, but do not even
    # construct a load-balancing objective. eval never reads w_noise or RNG.
    gates, _ = moe.noisy_top_k_gating(valid, train=False)
    dispatcher = sys.modules[type(moe).__module__].SparseDispatcher(moe.num_experts, gates,
                                                                  routing_level='node')
    parts = dispatcher.dispatch(valid)
    combined = dispatcher.combine([expert(part) for expert, part in zip(moe.experts, parts)])
    result = hidden.new_zeros(hidden.shape)
    result[mask] = combined
    return result


def masked_reld(encoder, clean):
    problem, raw, mask, _ = reld._validated(clean)
    raw = raw.float().masked_fill(~mask[..., None], 0.)
    customers = raw[:, 1:]
    tw = (customers[..., 6:8] if problem != 'CVRP'
          else customers.new_zeros((*customers.shape[:2], 2)))
    hidden = torch.cat((encoder.embedding_depot(raw[:, :1, :2]),
                        encoder.embedding_node(torch.cat((customers[..., :3], tw), -1))), 1)
    hidden = hidden.masked_fill(~mask[..., None], 0.)
    for layer in encoder.layers:
        if (layer.model_params['norm'] != 'none' or layer.model_params['norm_loc'] != 'norm_last'
                or layer.addAndNormalization1.norm is not None
                or layer.addAndNormalization2.norm is not None):
            raise ValueError('The pinned routing adapter requires norm=none/norm_last')
        attention = layer.multi_head_combine(_attention(layer, hidden, mask))
        first = layer.addAndNormalization1(hidden, attention).masked_fill(~mask[..., None], 0.)
        hidden = layer.addAndNormalization2(first, _eval_moe(layer.feedForward, first, mask))
        hidden = hidden.masked_fill(~mask[..., None], 0.)
    return hidden


def _matnet_block(block, row, col, distance, mask):
    q, k, v = (_heads(projection(value), block.head_num) for projection, value
               in ((block.Wq, row), (block.Wk, col), (block.Wv, col)))
    mixed = block.mixed_score_MHA
    dot = torch.matmul(q, k.transpose(2, 3)) / (mixed.qkv_dim ** .5)
    costs = distance[:, None].expand(-1, block.head_num, -1, -1)
    scores = torch.stack((dot, costs), dim=-1).transpose(1, 2)
    scores = F.relu(torch.matmul(scores, mixed.mix1_weight)
                    + mixed.mix1_bias[None, None, :, None, :])
    scores = (torch.matmul(scores, mixed.mix2_weight)
              + mixed.mix2_bias[None, None, :, None, :]).transpose(1, 2).squeeze(-1)
    weights = scores.masked_fill(~mask[:, None, None, :], float('-inf')).softmax(-1)
    attention = torch.matmul(weights, v).transpose(1, 2).reshape(*row.shape[:2], -1)
    first = masked_instance_norm(row + block.multi_head_combine(attention), mask,
                                 block.normalization_1.normalizer)
    return masked_instance_norm(first + block.feed_forward(first), mask,
                                block.normalization_2.normalizer)


def matnet_roles(mask, dtype=torch.float32):
    """Identity channels follow valid-node order, not padded slot numbers."""
    rank = (mask.long().cumsum(1) - 1).clamp_min(0)
    to_role = F.one_hot(rank, 256).to(dtype).masked_fill(~mask[..., None], 0.)
    return torch.zeros_like(to_role), to_role


def masked_matnet(encoder, matrix, mask):
    row, col = matnet_roles(mask, matrix.dtype)
    for layer in encoder.layers:
        # Both directions must read the PREVIOUS pair of role tensors.
        next_row = _matnet_block(layer.row_encoding_block, row, col, matrix, mask)
        next_col = _matnet_block(layer.col_encoding_block, col, row, matrix.transpose(1, 2), mask)
        row, col = next_row, next_col
    return torch.cat((row, col), -1)


class NativeDonor(nn.Module):
    SUPPORTED_PROBLEMS = SUPPORTED_PROBLEMS
    WIDTH_BY_PROBLEM = WIDTH_BY_PROBLEM
    SPEC = SPEC
    component_for_problem = staticmethod(component_for_problem)

    def __init__(self, arm, *, initialize_from_checkpoint=True, expected_provenance=None):
        super().__init__()
        if arm not in ARMS:
            raise ValueError('Unknown R106 arm')
        if not isinstance(initialize_from_checkpoint, bool):
            raise TypeError('initialize_from_checkpoint must be boolean')
        if not initialize_from_checkpoint and not isinstance(expected_provenance, dict):
            raise ValueError('State-only restore requires saved R106 donor provenance')
        self.arm = arm
        with native._seeded_cpu(native.RANDOM_SEEDS['TSP']):
            self.tsp = native._build_tsp().float()
            if initialize_from_checkpoint:
                native._load_encoder_weights(self.tsp, 'TSP')
        with native._seeded_cpu(native.RANDOM_SEEDS['ATSP']):
            self.atsp = native._build_atsp().float()
            if initialize_from_checkpoint:
                native._load_encoder_weights(self.atsp, 'ATSP')
        with native._seeded_cpu(reld.RANDOM_SEED):
            self.routing = (reld.StaticReLDDonor('pretrained') if initialize_from_checkpoint
                            else native._routing_restore_placeholder()).float()
        components = {name: native._component_provenance(name, 'B') for name in ('TSP', 'ATSP')}
        components['VRP'] = self.routing.provenance()
        components['VRP'].pop('spec')
        components['VRP']['model_params'] = copy.deepcopy(reld.SPEC['model_params'])
        self._provenance = dict(format=SPEC['format'], source_identity='R102-B originals',
                                components=components, spec=copy.deepcopy(SPEC))
        if expected_provenance is not None and self._provenance != expected_provenance:
            raise ValueError('R106 saved donor source/weight/role provenance differs')
        self.requires_grad_(arm == 'B_finetune')
        self._mark_native()
        self.eval()

    def _mark_native(self):
        for parameter in self.parameters():
            parameter._r106_native = True

    def _apply(self, fn, recurse=True):
        result = super()._apply(fn, recurse=recurse)
        self._mark_native()
        return result

    def provenance(self):
        return copy.deepcopy(self._provenance)

    def train(self, mode=True):
        return super().train(False)

    def forward(self, batch):
        clean = input_only(batch)
        problem = native._problem(clean)
        self.eval()
        key = 'matrix' if problem == 'ATSP' else 'node'
        values = clean[key]
        component = component_for_problem(problem)
        encoder = {'TSP': self.tsp, 'ATSP': self.atsp, 'VRP': self.routing.encoder}[component]
        if any(p.device != values.device or p.dtype != torch.float32 for p in encoder.parameters()):
            raise ValueError('Native encoder must remain FP32 on the input device')
        with torch.set_grad_enabled(torch.is_grad_enabled() and self.arm == 'B_finetune'), \
                reld._native_precision(values.device.type):
            if component == 'VRP':
                hidden = masked_reld(encoder, clean)
            else:
                if problem == 'TSP':
                    if values.ndim != 3 or not 2 <= values.shape[-1] <= 8:
                        raise ValueError('ELG requires original xy, with optional selector role slots')
                    clean = dict(clean, node=values[..., :2], d_in=2)
                values, mask, _ = native._native_inputs(clean, problem)
                if problem == 'ATSP':
                    pair_mask = mask[:, :, None] & mask[:, None, :]
                    if not torch.isfinite(values[pair_mask]).all():
                        raise ValueError('Nonfinite valid native matrix')
                    values = values.float().masked_fill(~pair_mask, 0.)
                    hidden = masked_matnet(encoder, values, mask)
                else:
                    if not torch.isfinite(values[mask]).all():
                        raise ValueError('Nonfinite valid native xy')
                    values = values.float().masked_fill(~mask[..., None], 0.)
                    hidden = masked_elg(encoder, values, mask)
            mask = clean['node_mask']
            if (hidden.shape != (*mask.shape, WIDTH_BY_PROBLEM[problem])
                    or hidden.dtype != torch.float32 or not torch.isfinite(hidden[mask]).all()):
                raise ValueError('R106 native output must be finite, node-aligned FP32')
            return hidden.masked_fill(~mask[..., None], 0.)
