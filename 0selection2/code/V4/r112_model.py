"""Boundary-conditioned region composition; no route or outcome enters inference.

The two arms have identical parameters and arithmetic shapes. A averages the
boundary axis before every message; B preserves it through bottom-up joins and
top-down conditioning. Port projection is approximate, not a routing solver.
"""

import copy

import torch
from torch import nn
import torch.nn.functional as F

from .V4Model import _masked_mean
from .pairwise_selector import ScoreDifferenceSelector
from .r77_data import parameter_hash
from .r88_model import prepare_role_inputs


ARMS = ('A_collapsed', 'B_interfaces')
BRANCH_INIT_SEED = 112002
SPEC = dict(experiment='R112', schema_version=1, regions=4, ports=3,
    resource_coordinates=[[.25, .25], [.25, .75], [.75, .25], [.75, .75]],
    node_stem_layers=2, composition_cycles=2, hierarchy=[4, 2, 1],
    interface='ordered_entry_exit_port_x_signed_load_budget_and_time_coordinates',
    resource_semantics='latent boundary coordinates, not certified physical capacity or feasibility',
    grouping='input-only farthest-first on symmetric distances; directed costs retained in messages',
    port_projection='nearest child boundary; approximate and allows repeated projected ports',
    upward='joint two-child states and directed connector costs; both traversal orders',
    downward='parent boundary lookup and sibling context at the same resource coordinate',
    readout='all levels enter original bidirectional candidate interaction; no old-score bypass',
    role_arm='B_execution_anchor', branch_init_seed=BRANCH_INIT_SEED,
    current_query_labels_in_forward=False, original_node_layers=4)


def _mlp(n, d, dropout):
    return nn.Sequential(nn.Linear(n, d), nn.GELU(), nn.Dropout(dropout), nn.Linear(d, d))


def _safe_softmax(score, mask, dim=-1):
    score = score.float().masked_fill(~mask, -1e30)
    weight = torch.softmax(score, dim=dim) * mask
    return weight / weight.sum(dim=dim, keepdim=True).clamp_min(1e-12)


@torch.no_grad()
def region_layout(batch, ports=3, regions=4):
    """Input-only groups and physical ports; no sample indices or target fields."""
    valid = batch['node_mask'].bool()
    if not valid.any(-1).all():
        raise ValueError('R112 requires a nonempty graph')
    if batch['kind'] == 'coord':
        xy = batch['node'][..., :2].float()
        dist = torch.cdist(xy, xy)
        first = (xy[..., 0] + xy[..., 1] * 1e-6).masked_fill(~valid, float('inf')).argmin(-1)
    else:
        dist = batch['matrix'].float()
        first = ((dist + dist.transpose(1, 2)).masked_fill(
            ~(valid[:, :, None] & valid[:, None, :]), 0).sum(-1)
            / valid.sum(-1, keepdim=True).clamp_min(1)).masked_fill(~valid, float('inf')).argmin(-1)
    pair_valid = valid[:, :, None] & valid[:, None, :]
    scale = dist.abs().masked_fill(~pair_valid, 0).sum((1, 2)) / pair_valid.sum((1, 2)).clamp_min(1)
    directed = dist / scale[:, None, None].clamp_min(1e-6)
    symmetric = (directed + directed.transpose(1, 2)) / 2
    b, n = valid.shape
    rows = torch.arange(b, device=valid.device)
    centers, nearest = [], torch.full_like(symmetric[:, 0], float('inf'))
    selected = torch.zeros_like(valid)
    center = first
    for _ in range(regions):
        centers.append(center)
        selected[rows, center] = True
        nearest = torch.minimum(nearest, symmetric[rows, center])
        score = nearest.masked_fill(~valid | selected, -1.)
        center = score.argmax(-1)
    centers = torch.stack(centers, 1)
    assignment = symmetric.gather(2, centers[:, None].expand(b, n, regions)).argmin(-1)
    leaf = (assignment[:, None] == torch.arange(regions, device=valid.device)[None, :, None]) & valid[:, None]
    memberships = [leaf, leaf.reshape(b, 2, 2, n).any(2), valid[:, None]]
    layouts = []
    for members in memberships:
        g = members.size(1)
        chosen, legal, minimum = [], [], torch.full((b, g, n), float('inf'), device=valid.device)
        used = torch.zeros_like(members)
        index = members.long().argmax(-1)
        for _ in range(ports):
            available = (members & ~used).any(-1)
            chosen.append(index)
            legal.append(available)
            used.scatter_(2, index[..., None], True)
            distances = symmetric[rows[:, None], index]
            minimum = torch.minimum(minimum, distances)
            index = minimum.masked_fill(~members | used, -1.).argmax(-1)
        layouts.append(dict(members=members, ports=torch.stack(chosen, -1),
                            port_mask=torch.stack(legal, -1)))
    return directed, layouts


class RegionInterfaceEncoder(nn.Module):
    def __init__(self, stem, arm, **params):
        super().__init__()
        self.stem, self.arm = stem, arm
        self.stem.layers = nn.ModuleList(list(stem.layers[:SPEC['node_stem_layers']]))
        self.d = d = params['embedding_dim']
        self.p, self.r = SPEC['ports'], len(SPEC['resource_coordinates'])
        drop = params['dropout']
        self.register_buffer('resources', torch.tensor(SPEC['resource_coordinates'], dtype=torch.float32))
        self.seed = _mlp(2 * d + 8, d, drop)
        self.q = nn.Linear(d, d, bias=False)
        self.k = nn.Linear(d, d, bias=False)
        self.v = nn.Linear(d, d, bias=False)
        self.leaf_norm = nn.LayerNorm(d)
        self.up = nn.ModuleList([_mlp(4 * d + 2, d, drop) for _ in range(2)])
        self.up_gate = nn.ModuleList([nn.Linear(d, 1) for _ in range(2)])
        self.up_norm = nn.ModuleList([nn.LayerNorm(d) for _ in range(2)])
        self.down = nn.ModuleList([_mlp(3 * d, d, drop) for _ in range(2)])
        self.down_norm = nn.ModuleList([nn.LayerNorm(d) for _ in range(2)])
        self.ff = _mlp(d, d, drop)
        self.out_norm = nn.LayerNorm(d)

    def condition(self, x, mask):
        x = x.masked_fill(~mask[..., None], 0.)
        if self.arm == 'A_collapsed':
            average = x.sum((2, 3, 4), keepdim=True) / mask.sum((2, 3, 4), keepdim=True)[..., None].clamp_min(1)
            x = average.expand_as(x).masked_fill(~mask[..., None], 0.)
        return x

    def seeds(self, nodes, distance, layout):
        b, g, p = layout['ports'].shape
        rows = torch.arange(b, device=nodes.device)[:, None, None]
        endpoint = nodes[rows, layout['ports']]
        entry = endpoint[:, :, :, None, None].expand(b, g, p, p, self.r, self.d)
        exit = endpoint[:, :, None, :, None].expand_as(entry)
        i, j = layout['ports'][:, :, :, None], layout['ports'][:, :, None, :]
        forward = distance[torch.arange(b, device=nodes.device)[:, None, None, None], i, j]
        backward = distance[torch.arange(b, device=nodes.device)[:, None, None, None], j, i]
        size = layout['members'].float().sum(-1) / layout['members'].size(-1)
        base = torch.stack((forward, backward, (i == j).float(), size[:, :, None, None].expand_as(forward)), -1)
        base = base[:, :, :, :, None].expand(b, g, p, p, self.r, 4)
        resources = self.resources.to(nodes.dtype)[None, None, None, None].expand(b, g, p, p, self.r, 2)
        # Input-only sizes and boundary coordinates; neither is a target cost.
        features = torch.cat((base.to(nodes.dtype), resources, resources.square()), -1)
        mask = (layout['port_mask'][:, :, :, None, None] & layout['port_mask'][:, :, None, :, None]).expand(b, g, p, p, self.r)
        return self.condition(self.seed(torch.cat((entry, exit, features), -1)), mask), mask

    def leaf_states(self, nodes, distance, layout, seed, mask):
        b, g = layout['members'].shape[:2]
        q = self.q(seed.flatten(2, 4))
        score = torch.einsum('bgqd,bnd->bgqn', q, self.k(nodes)) / self.d ** .5
        weights = _safe_softmax(score, layout['members'][:, :, None])
        value = torch.einsum('bgqn,bnd->bgqd', weights.to(nodes.dtype), self.v(nodes)).reshape_as(seed)
        return self.condition(self.leaf_norm(seed + value), mask)

    def nearest_ports(self, distance, source, target, target_mask):
        b = distance.size(0)
        cost = distance[torch.arange(b, device=distance.device)[:, None, None, None],
                        source[:, :, :, None], target[:, :, None, :]]
        return cost.masked_fill(~target_mask[:, :, None], float('inf')).argmin(-1)

    def join(self, child, child_layout, parent_layout, parent_seed, parent_mask, distance, level):
        b, groups = parent_seed.shape[:2]
        p, r, d = self.p, self.r, self.d
        left, right = child[:, 0::2], child[:, 1::2]
        lp, rp = child_layout['ports'][:, 0::2], child_layout['ports'][:, 1::2]
        lm, rm = child_layout['port_mask'][:, 0::2], child_layout['port_mask'][:, 1::2]
        parent_ports = parent_layout['ports']
        li = self.nearest_ports(distance, parent_ports, lp, lm)
        ri = self.nearest_ports(distance, parent_ports, rp, rm)
        batch = torch.arange(b, device=child.device)[:, None, None]
        group = torch.arange(groups, device=child.device)[None, :, None]
        messages, masks = [], []
        for first, second, fi, si, fp, sp, fm, sm in (
                (left, right, li, ri, lp, rp, lm, rm),
                (right, left, ri, li, rp, lp, rm, lm)):
            # Preserve outer endpoints; enumerate both internal connector ports.
            a = first[batch, group, fi].permute(0, 1, 2, 4, 3, 5)
            z = second.transpose(2, 3)[batch, group, si].permute(0, 1, 2, 4, 3, 5)
            a = a[:, :, :, None, :, :, None].expand(b, groups, p, p, r, p, p, d)
            z = z[:, :, None, :, :, None].expand_as(a)
            bridge = distance[torch.arange(b, device=child.device)[:, None, None, None],
                              fp[:, :, :, None], sp[:, :, None, :]]
            reverse = distance[torch.arange(b, device=child.device)[:, None, None, None],
                               sp[:, :, None, :], fp[:, :, :, None]]
            edge = torch.stack((bridge, reverse), -1)[:, :, None, None, None].expand(b, groups, p, p, r, p, p, 2)
            query = parent_seed[..., None, None, :].expand_as(a)
            message = self.up[level](torch.cat((a, z, a * z, query, edge.to(a.dtype)), -1))
            messages.append(message.flatten(5, 6))
            legal = (fm[:, :, :, None] & sm[:, :, None, :])[:, :, None, None, None]
            masks.append(legal.expand(b, groups, p, p, r, p, p).flatten(5, 6))
        message, legal = torch.cat(messages, 5), torch.cat(masks, 5)
        weight = _safe_softmax(self.up_gate[level](message).squeeze(-1), legal)
        joined = (message * weight[..., None].to(message.dtype)).sum(5)
        # An empty sibling must not suppress the surviving child's information.
        has_pair = legal.any(-1)
        survivor = child.reshape(b, groups, 2, p, p, r, d).sum(2)
        joined = torch.where(has_pair[..., None], joined, survivor)
        return self.condition(self.up_norm[level](parent_seed + joined), parent_mask)

    def propagate(self, child, child_layout, parent, parent_layout, distance, level, mask):
        b, groups, p = child_layout['ports'].shape
        parent_ports = parent_layout['ports'].repeat_interleave(2, 1)
        parent_legal = parent_layout['port_mask'].repeat_interleave(2, 1)
        match = self.nearest_ports(distance, child_layout['ports'], parent_ports, parent_legal)
        repeated = parent.repeat_interleave(2, 1)
        rows = torch.arange(b, device=child.device)[:, None, None, None]
        group = torch.arange(groups, device=child.device)[None, :, None, None]
        external = repeated[rows, group, match[:, :, :, None], match[:, :, None, :]]
        sibling = child.reshape(b, groups // 2, 2, p, p, self.r, self.d).flip(2).reshape_as(child)
        sibling = sibling.mean((2, 3), keepdim=True).expand_as(child)
        value = self.down[level](torch.cat((child, external, sibling), -1))
        return self.condition(self.down_norm[level](child + value), mask)

    def forward(self, batch):
        nodes, _ = self.stem(batch)
        distance, layouts = region_layout(batch, self.p, SPEC['regions'])
        seeds, masks = zip(*(self.seeds(nodes, distance, layout) for layout in layouts))
        states = [self.leaf_states(nodes, distance, layouts[0], seeds[0], masks[0]), seeds[1], seeds[2]]
        for _ in range(SPEC['composition_cycles']):
            for level in range(2):
                states[level + 1] = self.join(states[level], layouts[level], layouts[level + 1],
                    seeds[level + 1] + states[level + 1], masks[level + 1], distance, level)
            for level in (1, 0):
                states[level] = self.propagate(states[level], layouts[level], states[level + 1],
                    layouts[level + 1], distance, level, masks[level])
        tokens = torch.cat([self.condition(self.out_norm(x + self.ff(x)), m).flatten(1, 4)
                            for x, m in zip(states, masks)], 1)
        return tokens, torch.cat([m.flatten(1, 4) for m in masks], 1)


class RegionInterfaceSelector(ScoreDifferenceSelector):
    def __init__(self, arm, **params):
        if arm not in ARMS:
            raise ValueError('Unknown R112 arm')
        params = dict(params, r112_arm=arm, r112_spec=copy.deepcopy(SPEC))
        super().__init__(**params)
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(BRANCH_INIT_SEED)
            self.instance_encoder = RegionInterfaceEncoder(self.instance_encoder, arm, **params)

    def forward(self, batch):
        return super().forward(prepare_role_inputs(batch, SPEC['role_arm']))


def make_model(params, arm, state_dict=None):
    with torch.device('cpu'), torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(2)
        model = RegionInterfaceSelector(arm, **copy.deepcopy(params))
    if state_dict is not None:
        model.load_state_dict(state_dict, strict=True)
    return model


def initialize_shared(initial, geometry, arm, device='cpu'):
    if initial.get('seed') != 2 or initial.get('initialization') != 'scratch':
        raise ValueError('R112 starts from seed2 random weights, never a trained checkpoint')
    model = make_model(initial['model_params'], arm)
    original = initial['trainable']
    actual = dict(model.named_parameters())
    copied = {}
    with torch.no_grad():
        for name, value in actual.items():
            source = name.replace('instance_encoder.stem.', 'instance_encoder.')
            if source in original:
                value.copy_(original[source])
                copied[name] = value
        branch = model.instance_encoder.stem.geometry_residual
        for key in ('mean', 'std'):
            getattr(branch, key).copy_(torch.as_tensor(geometry[key]))
    receipt = dict(seed=2, initialization='scratch_seed2_shared_stem_and_seed112002_interfaces',
        fresh_initialization=True, historical_checkpoint_loaded=False,
        common_source_trainable_sha256=initial.get('trainable_sha256'),
        common_parameter_sha256=parameter_hash(copied),
        trainable_sha256=parameter_hash(actual), fresh_trainable_sha256=parameter_hash(actual),
        fresh_state_sha256=parameter_hash(model.state_dict()),
        parameter_count=sum(x.numel() for x in actual.values()), common_parameter_names=sorted(copied),
        geometry_fitted_on='R84 frozen TRAIN only', r112_spec=copy.deepcopy(SPEC))
    model.initialization_receipt = receipt
    return model.to(device), copy.deepcopy(model.params), receipt
