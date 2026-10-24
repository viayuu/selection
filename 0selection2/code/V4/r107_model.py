"""R107: bounded connected path/resource content inside the original encoder.

No data/checkpoint files, solvers, labels, extra losses, or GPU jobs are used.
The control has identical tensors but restricted functional freedom and less
compute: its recurrent nonlinear states belong to edges, never joint paths.
"""

import copy
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from code.unified_selector.registry import P2I
from .r77_data import parameter_hash
from .r88_model import ROLE_SPEC, RoleInputSelector
from .V4Model import InstanceEncoder


ARMS = ('A_edge_factorized', 'B_connected_paths')
ROLE_ARM = 'B_execution_anchor'
FRESH_SEED = 2
PATH_INIT_SEED = 107002
FRESH_REFERENCE_SHA256 = 'dad87d8598f7560e6f170ef048ba1bf515edeb03cef7ec42e16839a36af4ef93'
ORIGINAL_PARAMETER_COUNT = 1842953
NEW_PARAMETER_COUNT = 49280
PARAMETER_COUNT = ORIGINAL_PARAMETER_COUNT + NEW_PARAMETER_COUNT
PARAMETER_COUNTS = dict(original=ORIGINAL_PARAMETER_COUNT, **{arm: PARAMETER_COUNT for arm in ARMS})
ORIGINAL = dict(embedding_dim=128, encoder_layer_num=4, joint_layer_num=2,
                head_num=4, qkv_dim=32, ff_hidden_dim=512, head_hidden_dim=256,
                solver_feature_hidden=128, stats_dim=12, dropout=.1, rezero=False)
EDGE_FIELDS = (
    *(f'source_raw_{i}' for i in range(8)), *(f'target_raw_{i}' for i in range(8)),
    'travel', 'reverse_travel', 'scaled_travel', 'scaled_reverse_travel',
    'arrival', 'service_start', 'departure', 'signed_target_demand', 'window_slack',
    'C', 'O', 'B', 'L', 'TW',
)
COMPOSED_FIELDS = (
    'travel_sum', 'scaled_travel_sum', 'shortcut_ik', 'closure_ki',
    'triangle_travel', 'detour', 'reverse_travel_sum', 'depot_walk_billed_travel',
    'route_limit_slack', 'arrival_j', 'service_start_j', 'departure_j',
    'arrival_k', 'service_start_k', 'departure_k', 'window_slack_j',
    'window_slack_k', 'waiting_sum_jk', 'signed_prefix_i', 'signed_prefix_j',
    'signed_prefix_k', 'signed_prefix_min', 'signed_prefix_max', 'signed_prefix_span',
    'positive_demand_sum', 'negative_demand_magnitude_sum', 'demand_product_ij',
    'demand_product_jk', 'is_depot_i', 'is_depot_j', 'is_depot_k', 'service_sum',
)
EDGE_DIM = 64 + len(EDGE_FIELDS)
COMPOSED_DIM = len(COMPOSED_FIELDS)
PATH_SPEC = dict(
    neighbors=4, max_paths_per_source=16, relation_dim=32, hidden_dim=64,
    after_encoder_layers=[1, 3], edge_fields=list(EDGE_FIELDS),
    composed_fields=list(COMPOSED_FIELDS), edge_input_dim=EDGE_DIM,
    phi_input_dim=2 * EDGE_DIM + COMPOSED_DIM + 32,
    routing='reserve_one_actual_depot_neighbor_for_each_customer',
    neighbor_rule='directed_top4_drop_only_overflowing_boundary_tie_class',
    path_rule='valid_distinct_i_j_k_no_feasibility_filter_no_depot_reset',
    time_rule='time0_depot_start_wait_then_service_no_depot_zero_deadline',
    demand_rule='stored_signed_prefix_extrema_not_onboard_load_or_classical_B',
    length_rule='raw_distance_only_open_omits_final_depot_return',
    injection='masked_path_mean_Linear32to128_times_0.1_after_layers1and3',
    control='phi(edge,edge,zero_composition,previous_edge)_then_average',
    connected='phi(edge_ij,edge_jk,composed_resources,previous_path)',
    initial_relation='mean_projected_endpoints_A_or_three_connected_nodes_B',
    inactive_control_composed_weights=4096,
    comparison='equal_complete_tensors_not_equal_active_freedom_or_FLOPs',
)
SPEC = dict(
    experiment='R107', scope='model_only', parent='full18_R88B_R77_score_path',
    role_arm=ROLE_ARM, role_spec=copy.deepcopy(ROLE_SPEC),
    selection_head='original_score_mlp_maximin', fresh_seed=FRESH_SEED,
    path_init_seed=PATH_INIT_SEED, path_spec=copy.deepcopy(PATH_SPEC),
    original=copy.deepcopy(ORIGINAL), parameter_counts=copy.deepcopy(PARAMETER_COUNTS),
    new_parameter_count=NEW_PARAMETER_COUNT, label_or_route_inputs=False,
    arms={ARMS[0]: 'independent_edge_recurrence_additive_fragment_content',
          ARMS[1]: 'connected_path_recurrence_with_explicit_composed_resources'},
    equal_total_parameters_not_equal_active_freedom=True, equal_flops_claim=False,
)


def _gather_nodes(values, indices):
    batch = torch.arange(values.size(0), device=values.device)
    return values[batch.reshape(-1, *([1] * (indices.ndim - 1))), indices]


def _gather_pairs(matrix, source, target):
    batch = torch.arange(matrix.size(0), device=matrix.device)
    return matrix[batch.reshape(-1, *([1] * (source.ndim - 1))), source, target]


def _neighbors(distances, mask, depot, routing):
    """Bounded, relabeling-equivariant selection, including exact tie cases."""
    b, n, _ = distances.shape
    valid = mask[:, :, None] & mask[:, None, :]
    valid = valid & ~torch.eye(n, dtype=torch.bool, device=mask.device)[None]
    if routing:
        valid = valid & ~depot[:, None, :]
    scores = distances.masked_fill(~valid, float('inf'))
    values, indices = scores.topk(min(4, n), largest=False, sorted=True)
    if n < 4:
        values = F.pad(values, (0, 4 - n), value=float('inf'))
        indices = F.pad(indices, (0, 4 - n))
    budget = torch.where(depot, 4, 3) if routing else torch.full_like(mask, 4, dtype=torch.long)
    cutoff = values.gather(-1, (budget - 1)[..., None])
    overflow = ((scores <= cutoff) & valid).sum(-1) > budget
    selected = torch.isfinite(values) & (values <= cutoff)
    selected &= ~(overflow[..., None] & (values == cutoff))
    if routing:
        depot_index = depot.long().argmax(-1)[:, None].expand(b, n)
        customers = mask & ~depot
        indices = indices.clone()
        indices[..., 3] = torch.where(customers, depot_index, indices[..., 3])
        selected[..., 3] = torch.where(customers, True, selected[..., 3])
    return indices, selected & mask[..., None]


def _step(departure, travel, raw_target, target_depot, tw):
    arrival = departure + travel
    start = torch.where(target_depot, arrival, torch.maximum(arrival, raw_target[..., 6]))
    end = start + raw_target[..., 5]
    slack = torch.where(target_depot, 0., raw_target[..., 7] - start)
    return arrival * tw, start * tw, end * tw, slack * tw, (start - arrival) * tw


@dataclass
class PathInputs:
    neighbors: torch.Tensor
    edge_mask: torch.Tensor
    path_mask: torch.Tensor
    second_nodes: torch.Tensor
    edge_primitives: torch.Tensor
    composed: torch.Tensor


def build_path_inputs(batch):
    """Current raw inputs only; deterministic local walk arithmetic, not a solver.

    Valid-node demand units are unchanged. Prefix demand is a served-demand
    diagnostic, NOT certified physical onboard load. Depot intermediate visits
    do not silently reset demand, length or time. No resource values filter paths.
    """
    mask = batch['node_mask']
    b, n = mask.shape
    routing = int(batch['problem_id']) not in (P2I['TSP'], P2I['ATSP'])
    valid = mask[:, :, None] & mask[:, None, :]
    with torch.autocast(device_type=mask.device.type, enabled=False):
        if batch['kind'] == 'coord':
            raw = batch['node'].float()
            raw = F.pad(raw, (0, 8 - raw.size(-1))).masked_fill(~mask[..., None], 0.)
            distances = torch.cdist(raw[..., :2], raw[..., :2],
                                    compute_mode='donot_use_mm_for_euclid_dist')
        else:
            raw = torch.zeros(b, n, 8, device=mask.device)
            distances = batch['matrix'].float().masked_fill(~valid, 0.)
        depot = mask & (raw[..., 3] == 1) if routing else torch.zeros_like(mask)
        if routing and not (depot.sum(-1) == 1).all():
            raise ValueError('R107 routing inputs require exactly one valid physical depot')
        with torch.no_grad():
            neighbors, edge_mask = _neighbors(distances, mask, depot, routing)
        source = torch.arange(n, device=mask.device)[None, :, None].expand(b, n, 4)
        first = neighbors
        second = _gather_nodes(neighbors, first)
        path_mask = (edge_mask[..., None] & _gather_nodes(edge_mask, first)
                     & (second != source[..., None]))
        ri = raw[:, :, None, :].expand(b, n, 4, 8)
        rj = _gather_nodes(raw, first)
        dij = _gather_pairs(distances, source, first)
        dji = _gather_pairs(distances, first, source)
        off = valid & ~torch.eye(n, dtype=torch.bool, device=mask.device)[None]
        scale = (distances.abs().masked_fill(~off, 0.).sum((1, 2))
                 / off.sum((1, 2)).clamp_min(1)).clamp_min(1e-6)
        bits = batch['cbits'].float()
        tw_edge = bits[:, 4, None, None]
        dep_i = depot[:, :, None].expand_as(first)
        dep_j = _gather_nodes(depot, first)
        if routing:
            dep_index = depot.long().argmax(-1)
            dep_to = distances[torch.arange(b, device=mask.device), dep_index]
            to_dep = distances.gather(2, dep_index[:, None, None].expand(b, n, 1)).squeeze(-1)
        else:
            dep_to = torch.zeros_like(raw[..., 0])
            to_dep = torch.zeros_like(dep_to)
        base_start = torch.where(depot, dep_to, torch.maximum(dep_to, raw[..., 6]))
        departure_i = (base_start + raw[..., 5]) * bits[:, 4, None]
        aj, sj, ej, slackj, waitj = _step(departure_i[..., None], dij, rj, dep_j, tw_edge)
        edge = torch.cat((ri, rj, torch.stack((dij, dji, dij / scale[:, None, None],
            dji / scale[:, None, None], aj, sj, ej, rj[..., 2], slackj), -1),
            bits[:, None, None, :].expand(b, n, 4, 5)), -1)
        edge = edge.masked_fill(~edge_mask[..., None], 0.)

        dijp = dij[..., None].expand(b, n, 4, 4)
        djk = _gather_nodes(dij, first)
        dkj = _gather_nodes(dji, first)
        ip = source[..., None].expand_as(second)
        dik = _gather_pairs(distances, ip, second)
        dki = _gather_pairs(distances, second, ip)
        rk = _gather_nodes(raw, second)
        dep_k = _gather_nodes(depot, second)
        ak, sk, ek, slackk, waitk = _step(ej[..., None], djk, rk, dep_k, tw_edge[..., None])
        travel = dijp + djk
        billed = (dep_to[:, :, None, None] + travel
                  + (1. - bits[:, 1, None, None, None]) * _gather_nodes(to_dep, second))
        if not routing:
            billed = torch.zeros_like(billed)
        limit = raw[..., 4].masked_fill(~mask, 0.).amax(-1)[:, None, None, None]
        pi = ri[..., 2, None].expand_as(travel)
        dj = rj[..., 2, None].expand_as(travel)
        dk = rk[..., 2]
        pj, pk = pi + dj, pi + dj + dk
        pmin = torch.minimum(torch.minimum(pi, pj), pk).clamp_max(0.)
        pmax = torch.maximum(torch.maximum(pi, pj), pk).clamp_min(0.)
        expand = lambda x: x[..., None].expand_as(travel)
        fields = (travel, travel / scale[:, None, None, None], dik, dki,
            travel + dki, travel - dik, expand(dji) + dkj, billed,
            (limit - billed) * bits[:, 3, None, None, None], expand(aj), expand(sj),
            expand(ej), ak, sk, ek, expand(slackj), slackk, expand(waitj) + waitk,
            pi, pj, pk, pmin, pmax, pmax - pmin,
            pi.clamp_min(0.) + dj.clamp_min(0.) + dk.clamp_min(0.),
            (-pi).clamp_min(0.) + (-dj).clamp_min(0.) + (-dk).clamp_min(0.),
            pi * dj, dj * dk, expand(dep_i.float()), expand(dep_j.float()), dep_k.float(),
            (expand(ri[..., 5]) + expand(rj[..., 5]) + rk[..., 5]) * tw_edge[..., None])
        composed = torch.stack(fields, -1).masked_fill(~path_mask[..., None], 0.)
    return PathInputs(first, edge_mask, path_mask, second, edge, composed)


class RelationRound(nn.Module):
    def __init__(self):
        super().__init__()
        self.phi = nn.Sequential(nn.Linear(252, 64), nn.GELU(), nn.Linear(64, 32), nn.LayerNorm(32))
        self.to_nodes = nn.Linear(32, 128)

    def update(self, edge, graph, previous, arm):
        affine = self.phi[0]
        # Split the shared affine to avoid a 252d B*N*4*4 concatenation.
        if arm == ARMS[0]:
            hidden = F.linear(edge, affine.weight[:, :EDGE_DIM] + affine.weight[:, EDGE_DIM:2 * EDGE_DIM], affine.bias)
            zero = edge.new_zeros((*edge.shape[:-1], COMPOSED_DIM))
            hidden = hidden + F.linear(zero, affine.weight[:, 2 * EDGE_DIM:2 * EDGE_DIM + COMPOSED_DIM])
        else:
            left = F.linear(edge, affine.weight[:, :EDGE_DIM], affine.bias)
            right = F.linear(edge, affine.weight[:, EDGE_DIM:2 * EDGE_DIM])
            hidden = left[..., None, :] + _gather_nodes(right, graph.neighbors)
            hidden = hidden + F.linear(graph.composed.to(edge.dtype),
                affine.weight[:, 2 * EDGE_DIM:2 * EDGE_DIM + COMPOSED_DIM])
        if previous is None:
            endpoints = edge[..., :32] + edge[..., 32:64]
            previous = (.5 * endpoints if arm == ARMS[0] else
                (endpoints[..., None, :] + _gather_nodes(edge[..., 32:64], graph.neighbors)) / 3.)
        hidden = hidden + F.linear(previous, affine.weight[:, -32:])
        state = self.phi[3](previous + self.phi[2](self.phi[1](hidden)))
        if arm == ARMS[0]:
            state = state.masked_fill(~graph.edge_mask[..., None], 0.)
            paths = .5 * (state[..., None, :] + _gather_nodes(state, graph.neighbors))
        else:
            state = state.masked_fill(~graph.path_mask[..., None], 0.)
            paths = state
        count = graph.path_mask.sum((-1, -2))
        mean = paths.masked_fill(~graph.path_mask[..., None], 0.).sum((-2, -3)) / count.clamp_min(1)[..., None]
        message = .1 * self.to_nodes(mean)
        return state, message.masked_fill((count == 0)[..., None], 0.)


class ConnectedPathEncoder(InstanceEncoder):
    def __init__(self, arm, **params):
        super().__init__(**params)
        self.path_arm = arm
        self.path_node_projection = nn.Sequential(nn.Linear(128, 32, bias=False), nn.LayerNorm(32))
        self.path_rounds = nn.ModuleList([RelationRound(), RelationRound()])

    def forward(self, batch):
        # Keep original raw projections, conditions, bias, layers and token order.
        node_mask = batch['node_mask']
        safe = dict(batch)
        if batch['kind'] == 'coord':
            node = self._pad_node(batch['node']).masked_fill(~node_mask[..., None], 0.)
            safe['node'] = node
            stats = self._coord_stats(node, node_mask)
            cond, _ = self.condition_encoder(batch, stats, kind_id=0)
            node_token = self.coord_proj(node)
            node_token = node_token + self.geometry_residual(node, node_mask, batch['n'], batch.get('node_geom'))
            bias = self._coord_bias(node, node_mask, 3)
        else:
            valid = node_mask[:, :, None] & node_mask[:, None, :]
            matrix = batch['matrix'].masked_fill(~valid, 0.)
            safe['matrix'] = matrix
            stats = self._matrix_stats(matrix, node_mask)
            cond, _ = self.condition_encoder(batch, stats, kind_id=1)
            feat, bias = self._matrix_features_and_bias(matrix, node_mask, 3)
            node_token = self.matrix_proj(feat)
        graph = build_path_inputs(safe)
        b = node_mask.size(0)
        token = torch.cat((self.cls_token.expand(b, 1, -1), cond[:, None],
                           self.stats_token_proj(stats)[:, None], node_token), 1)
        mask = torch.cat((torch.ones(b, 3, dtype=torch.bool, device=node_mask.device), node_mask), 1)
        previous, round_index = None, 0
        for index, layer in enumerate(self.layers):
            token = layer(token, mask=mask, attn_bias=bias)
            if index in (0, 2):
                physical = token[:, 3:]
                projected = self.path_node_projection(physical)
                source = projected[:, :, None].expand(-1, -1, 4, -1)
                target = _gather_nodes(projected, graph.neighbors)
                edge = torch.cat((source, target, graph.edge_primitives.to(projected.dtype)), -1)
                previous, message = self.path_rounds[round_index].update(edge, graph, previous, self.path_arm)
                token = torch.cat((token[:, :3], physical + message), 1)
                round_index += 1
        return token[:, 3:].masked_fill(~node_mask[..., None], 0.), node_mask


class ExplicitPathSelector(RoleInputSelector):
    def __init__(self, arm, **params):
        super().__init__(ROLE_ARM, **params)
        self.path_arm = arm
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(PATH_INIT_SEED)
            self.instance_encoder = ConnectedPathEncoder(arm, **dict(params, node_only=True))


def _resolve_params(params, arm):
    if arm not in ARMS:
        raise ValueError('Unknown R107 path arm')
    params = copy.deepcopy(params)
    if params.get('path_arm', arm) != arm or params.get('path_spec', PATH_SPEC) != PATH_SPEC:
        raise ValueError('R107 path arm/specification changed during reconstruction')
    if params.get('seed', FRESH_SEED) != FRESH_SEED:
        raise ValueError('R107 requires scratch seed2')
    if any(params.get(name, value) != value for name, value in ORIGINAL.items()):
        raise ValueError('R107 keeps the original R88B four-layer width128 architecture')
    if (params.get('local_geometry') is not True or params.get('geometry_mode', 'real') != 'real'
            or params.get('solver_feature_weight') != 1. or params.get('ignore_coord_dist', True) is not True
            or params.get('decision_head', 'classification') != 'classification'
            or params.get('role_arm', ROLE_ARM) != ROLE_ARM
            or any(name in params for name in ('capacity_arm', 'capacity_spec', 'expert_arm', 'expert_spec'))):
        raise ValueError('R107 retains R88B role, real geometry, ID+34 and original scores')
    params.update(ORIGINAL, path_arm=arm, path_spec=copy.deepcopy(PATH_SPEC))
    return params


def _copy_fresh(model, reference):
    actual = dict(model.named_parameters())
    if any(name not in actual or actual[name].shape != tensor.shape for name, tensor in reference.items()):
        raise ValueError('R107 must preserve every original named trainable tensor')
    added = sorted(actual.keys() - reference.keys())
    if any(not name.startswith(('instance_encoder.path_node_projection.', 'instance_encoder.path_rounds.')) for name in added):
        raise ValueError('Unexpected parameter outside the bounded R107 relation module')
    with torch.no_grad():
        for name, tensor in reference.items():
            actual[name].copy_(tensor)
    common_hash = parameter_hash({name: actual[name] for name in reference})
    if common_hash != parameter_hash(reference):
        raise ValueError('R107 original fresh tensor copy differs')
    return dict(common_parameter_names=sorted(reference), common_parameter_sha256=common_hash,
        common_parameter_count=sum(t.numel() for t in reference.values()),
        new_parameter_names=added, removed_parameter_names=[],
        independently_initialized_parameter_names=added,
        independently_initialized_parameter_count=sum(actual[name].numel() for name in added),
        original_parameter_count=sum(t.numel() for t in reference.values()),
        parameter_count=sum(t.numel() for t in actual.values()))


def make_model(params, arm, state=None, *, state_dict=None):
    """Fresh CPU factory or strict in-memory restore; preserves caller CPU RNG."""
    if state is not None and state_dict is not None:
        raise ValueError('Supply state or state_dict, not both')
    state = state_dict if state_dict is not None else state
    resolved = _resolve_params(params, arm)
    raw = {name: value for name, value in resolved.items() if name not in ('path_arm', 'path_spec')}
    with torch.device('cpu'), torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(FRESH_SEED)
        original = RoleInputSelector(ROLE_ARM, **raw)
        reference = dict(original.named_parameters())
        reference_hash = parameter_hash(reference)
        if reference_hash != FRESH_REFERENCE_SHA256:
            raise ValueError('R107 original factory differs from exact R84 scratch seed2')
        torch.random.default_generator.manual_seed(FRESH_SEED)
        model = ExplicitPathSelector(arm, **resolved)
        receipt = _copy_fresh(model, reference)
    if (receipt['original_parameter_count'] != ORIGINAL_PARAMETER_COUNT
            or receipt['independently_initialized_parameter_count'] != NEW_PARAMETER_COUNT
            or receipt['parameter_count'] != PARAMETER_COUNT):
        raise ValueError('R107 measured parameter count differs from the fixed contract')
    fresh = parameter_hash(dict(model.named_parameters()))
    fresh_state = parameter_hash(model.state_dict())
    receipt.update(seed=FRESH_SEED, path_init_seed=PATH_INIT_SEED,
        initialization='scratch_seed2_common_plus_isolated_path_relations',
        fresh_source='regenerated_original_R84_seed2_trainables',
        r84_fresh_trainable_sha256=reference_hash, historical_checkpoint_loaded=False,
        fresh_initialization=True, state_dict_restored=False,
        fresh_trainable_sha256=fresh, trainable_sha256=fresh,
        fresh_state_sha256=fresh_state, state_sha256=fresh_state,
        intentionally_inactive_composed_weight_count=4096 if arm == ARMS[0] else 0,
        restricted_control='duplicated_edge_inputs_collapse_first_two_affine_blocks_to_their_sum',
        equal_total_parameters_not_equal_active_freedom=True, equal_flops_claim=False,
        parameter_count_arithmetic='1842953 + 4160 + 2*(16192+2080+64+4224) = 1892233')
    if state is not None:
        model.load_state_dict(state, strict=True)
        restored = parameter_hash(dict(model.named_parameters()))
        restored_state = parameter_hash(model.state_dict())
        receipt.update(initialization='strict_caller_supplied_in_memory_restore',
            fresh_source='fresh_hashes_reference_only_not_restore_provenance',
            historical_checkpoint_loaded=None, fresh_initialization=False, state_dict_restored=True,
            restored_trainables_match_fresh=restored == fresh, restored_state_matches_fresh=restored_state == fresh_state,
            trainable_sha256=restored, state_sha256=restored_state)
    model.initialization_receipt = receipt
    return model


def initialize_shared(initial, geometry, arm, device='cpu'):
    """Validate exact scratch record and install caller's frozen TRAIN moments."""
    if (initial.get('seed') != FRESH_SEED or initial.get('initialization') != 'scratch'
            or parameter_hash(initial['trainable']) != initial.get('trainable_sha256')):
        raise ValueError('Only hashed R84 scratch seed2 initialization is permitted')
    model = make_model(initial['model_params'], arm)
    if initial['trainable_sha256'] != model.initialization_receipt['r84_fresh_trainable_sha256']:
        raise ValueError('R84 record is not exact fresh seed2; trained donors are forbidden')
    transfer = _copy_fresh(model, initial['trainable'])
    branch = model.instance_encoder.geometry_residual
    mean = torch.as_tensor(geometry['mean'], dtype=branch.mean.dtype, device='cpu')
    std = torch.as_tensor(geometry['std'], dtype=branch.std.dtype, device='cpu')
    if (mean.shape != branch.mean.shape or std.shape != branch.std.shape
            or not torch.isfinite(mean).all() or not torch.isfinite(std).all() or not (std > 0).all()):
        raise ValueError('Expected twelve finite frozen TRAIN means and positive standard deviations')
    if geometry.get('split', 'train') != 'train':
        raise ValueError('R107 geometry must be fitted on frozen TRAIN only')
    with torch.no_grad():
        branch.mean.copy_(mean)
        branch.std.copy_(std)
    receipt = model.initialization_receipt
    receipt.update(transfer, fresh_source='validated_in_memory_R84_scratch_seed2',
                   geometry_fitted_on=geometry.get('fitted_on', 'caller must bind frozen TRAIN provenance'))
    receipt['fresh_state_sha256'] = receipt['state_sha256'] = parameter_hash(model.state_dict())
    return model.to(device), copy.deepcopy(model.params), copy.deepcopy(receipt)
