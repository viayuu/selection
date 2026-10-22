"""Original full18 selector with input-only competition messages at each node."""

from copy import deepcopy

import torch
from torch import nn

from .V4Model import InstanceEncoder
from .pairwise_selector import ScoreDifferenceSelector
from .r88_model import RoleInputSelector, prepare_role_inputs
from .r99_relations import WIDTH, SPEC as TRANSFORM_SPEC, node_messages


ARMS = ('A', 'B')
VARIANTS = dict(A='row', B='balanced')
INPUT_KEYS = ('problem_id', 'kind', 'n', 'node_mask', 'node', 'matrix', 'node_geom',
              'cbits', 'problem_desc', 'coord_dist', 'pool_ids', 'mask', 'solver_mask', 'd_in')
SPEC = dict(backbone='R88B_original_R77_full18', transform=TRANSFORM_SPEC,
    fusion='original_node_token + Linear32to128(LN_no_affine(messages))',
    fusion_location='before_original_four_instance_layers', projection_seed=990003,
    selection_head='original_score_mlp_maximin', objective='original_R77',
    method_specific_new_parameters=False, external_pretraining=False)


def inputs_with_relations(batch):
    prepared = {key: batch[key] for key in INPUT_KEYS if key in batch}
    for key in ('relation_nodes', 'relation_node_mask'):
        if key in batch:
            prepared[key] = batch[key]
    return prepared


class CompetitionInstanceEncoder(InstanceEncoder):
    def __init__(self, **params):
        super().__init__(**params)
        if not self.node_only:
            raise ValueError('R99 retains the node-only original dual-stream encoder')
        self.relation_norm = nn.LayerNorm(WIDTH, elementwise_affine=False)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(990003)
            self.relation_projection = nn.Linear(WIDTH, params['embedding_dim'], bias=False)

    def forward(self, batch):
        node_mask, special_count = batch['node_mask'], 3
        if batch['kind'] == 'coord':
            node = self._pad_node(batch['node'])
            stats = self._coord_stats(node, node_mask)
            cond, _ = self.condition_encoder(batch, stats, kind_id=0)
            node_token = self.coord_proj(node)
            node_token = node_token + self.geometry_residual(
                node, node_mask, batch['n'], batch.get('node_geom'))
            bias = self._coord_bias(node, node_mask, special_count)
        else:
            stats = self._matrix_stats(batch['matrix'], node_mask)
            cond, _ = self.condition_encoder(batch, stats, kind_id=1)
            features, bias = self._matrix_features_and_bias(batch['matrix'], node_mask, special_count)
            node_token = self.matrix_proj(features)
        messages = batch['relation_nodes'].detach().to(self.relation_projection.weight.dtype)
        residual = self.relation_projection(self.relation_norm(messages))
        node_token = node_token + residual.masked_fill(~node_mask[..., None], 0.)
        count = len(node_mask)
        token = torch.cat((self.cls_token.expand(count, 1, -1), cond[:, None],
                           self.stats_token_proj(stats)[:, None], node_token), 1)
        mask = torch.cat((torch.ones(count, special_count, device=node_mask.device,
                                    dtype=torch.bool), node_mask), 1)
        for layer in self.layers:
            token = layer(token, mask=mask, attn_bias=bias)
        return token[:, special_count:], node_mask


class CompetitionSelector(RoleInputSelector):
    def __init__(self, arm, **params):
        if arm not in ARMS:
            raise ValueError('Unknown R99 arm')
        if params.get('r99_arm', arm) != arm or params.get('r99_spec', SPEC) != SPEC:
            raise ValueError('R99 arm or fixed input transform changed')
        params = dict(params, r99_arm=arm, r99_spec=deepcopy(SPEC))
        super().__init__('B_execution_anchor', **params)
        old = self.instance_encoder
        with torch.random.fork_rng(devices=[]):
            self.instance_encoder = CompetitionInstanceEncoder(**dict(self.params, node_only=True))
        missing, unexpected = self.instance_encoder.load_state_dict(old.state_dict(), strict=False)
        if missing != ['relation_projection.weight'] or unexpected:
            raise RuntimeError('Every original instance-encoder state must be retained')
        self.r99_arm = arm

    def prepare_inputs(self, batch):
        prepared = prepare_role_inputs(inputs_with_relations(batch), 'B_execution_anchor')
        mask = prepared['node_mask']
        if 'relation_nodes' not in prepared:
            if self.training:
                raise ValueError('Formal R99 TRAIN requires its input-only message cache')
            prepared['relation_nodes'] = node_messages(prepared, VARIANTS[self.r99_arm])
            prepared['relation_node_mask'] = mask
        nodes = prepared['relation_nodes']
        feature_mask = prepared.get('relation_node_mask')
        if (not torch.is_tensor(nodes) or nodes.shape != (*mask.shape, WIDTH) or nodes.device != mask.device
                or not torch.is_tensor(feature_mask) or not torch.equal(feature_mask, mask)
                or not torch.isfinite(nodes[mask]).all()):
            raise ValueError('Input-only per-node competition messages are not aligned')
        prepared['relation_nodes'] = nodes.detach().masked_fill(~mask[..., None], 0.)
        return prepared

    def forward(self, batch):
        return ScoreDifferenceSelector.forward(self, self.prepare_inputs(batch))


def make_model(params, arm, state_dict=None):
    model = CompetitionSelector(arm, **params)
    if state_dict is not None:
        model.load_state_dict(state_dict, strict=True)
    return model
