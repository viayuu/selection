"""Typed relational full-pool selector for routing and six nonrouting domains.

Raw inputs and deployment IDs are the entire forward contract. Native objectives,
certificates and split/instance IDs belong to the runner's separate namespaces.
The four relation/global layers and subsequent candidate interaction are shared.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F

from code.unified_selector.registry import GLOBAL_SOLVERS, PROBLEMS
from .V4Model import (
    EncoderLayer, InstanceEncoder, MultiHeadAttention, _masked_max, _masked_mean,
    get_default_model_params,
)
from .dual_stream import JointEncoderLayer
from .pairwise_selector import maximin_logits
from .solver_features import build_solver_features


NEW_PROBLEMS = ('MIS', 'JSSP', 'FJSP', 'BPP', 'MKP', 'SMTWTP')
ALL_PROBLEMS = tuple(PROBLEMS) + NEW_PROBLEMS
INPUT_KEYS = {
    'MIS': {'num_nodes', 'edge_index', 'node_mask', 'edge_mask'},
    'JSSP': {'num_jobs', 'num_machines', 'job_ptr', 'operation_job',
             'operation_index', 'precedence_edge_index', 'duration', 'machine',
             'job_mask', 'machine_mask', 'operation_mask', 'precedence_mask', 'job_ptr_mask'},
    'FJSP': {'num_jobs', 'num_machines', 'job_ptr', 'operation_job',
             'operation_index', 'precedence_edge_index', 'processing_time',
             'compatibility', 'job_mask', 'machine_mask', 'operation_mask',
             'precedence_mask', 'job_ptr_mask'},
    'BPP': {'sizes', 'capacity', 'item_mask'},
    'MKP': {'profits', 'resources', 'capacities', 'item_mask', 'resource_mask'},
    'SMTWTP': {'p', 'w', 'd', 'job_mask'},
}
ROUTING_INPUT_KEYS = {
    'problem_id', 'kind', 'n', 'node', 'matrix', 'node_mask', 'cbits',
    'problem_desc', 'coord_dist', 'node_geom', 'pool_ids', 'solver_mask', 'mask',
    'd_in',
}


def method_family(name):
    leaf = str(name).split('::')[-1]
    fields = leaf.split('/')
    if len(fields) >= 2 and fields[0] in NEW_PROBLEMS:
        leaf = fields[1]
    for prefix in ('DIFUSCO', 'L2D', 'DAN', 'T2T', 'DeepACO', 'AntSystem'):
        if leaf.upper().startswith(prefix.upper()):
            return prefix
    return 'ReLD' if leaf in ('RELD_MOEL', 'RELD_MTL', 'RELD_CVRP', 'reld_mt') else leaf


def default_config(solver_names, problem_names=ALL_PROBLEMS, **overrides):
    params = get_default_model_params()
    params.update(local_geometry=True, geometry_mode='real', ignore_coord_dist=True,
                  sdpa=True, joint_layer_num=2, solver_feature_hidden=128)
    params.update(solver_names=list(solver_names), problem_names=list(problem_names),
                  relation_types=10, relation_value_dim=3, node_feature_dim=8)
    params.update(overrides)
    return params


@dataclass
class EncodedGraph:
    nodes: torch.Tensor
    mask: torch.Tensor
    source: torch.Tensor
    target: torch.Tensor
    relation: torch.Tensor
    edge_value: torch.Tensor
    attention_bias: torch.Tensor | None = None
    dense_value: torch.Tensor | None = None
    dense_mask: torch.Tensor | None = None
    dense_offset: int = 0


class TypedFrontend(nn.Module):
    """Keep full raw relations, with units determined from the query inputs."""

    def __init__(self, params):
        super().__init__()
        self.params = params
        d = params['embedding_dim']
        self.route = InstanceEncoder(**dict(params, node_only=True, encoder_layer_num=0))
        self.project = nn.Sequential(nn.Linear(8, d), nn.GELU(), nn.Linear(d, d))
        self.node_type = nn.Embedding(10, d)
        self.task = nn.Embedding(len(params['problem_names']), d)
        self.global_project = nn.Linear(8, d)
        self.norm = nn.LayerNorm(d)
        self.problem_index = {p: i for i, p in enumerate(params['problem_names'])}

    def set_routing_geometry(self, statistics):
        branch = self.route.geometry_residual
        with torch.no_grad():
            branch.mean.copy_(torch.as_tensor(statistics['mean'], dtype=branch.mean.dtype))
            branch.std.copy_(torch.as_tensor(statistics['std'], dtype=branch.std.dtype))

    @staticmethod
    def _blank(mask):
        return torch.zeros((*mask.shape, 8), device=mask.device, dtype=torch.float32)

    @staticmethod
    def _priority(inputs, mask, key='execution_rank'):
        # Native dispatch and greedy ties use input order. Preserve that known
        # execution role, with an explicit field for role-preserving relabeling.
        rank = inputs.get(key)
        if rank is None:
            rank = torch.arange(mask.size(1), device=mask.device)[None].expand_as(mask)
        if rank.shape != mask.shape:
            raise ValueError(f'{key} must align with its native entities')
        return rank.float() / (mask.sum(-1, keepdim=True) - 1).clamp_min(1)

    @staticmethod
    def _relation(rows, source, target, kind, value=None):
        if value is None:
            value = torch.ones((rows.numel(), 3), device=rows.device)
        return rows, source, target, torch.full_like(rows, kind), value.float()

    def _finish(self, problem, features, types, mask, edges, global_features):
        b, n = mask.shape
        token = self.project(features) + self.node_type(types)
        condition = self.task.weight[self.problem_index[problem]][None] + self.global_project(global_features)
        token = self.norm(torch.cat((condition[:, None], token), dim=1))
        mask = torch.cat((torch.ones((b, 1), dtype=torch.bool, device=mask.device), mask), dim=1)
        token = token.masked_fill(~mask[..., None], 0.)
        if edges:
            rows, source, target, relation, value = (torch.cat(parts, dim=0) for parts in zip(*edges))
            source = rows * (n + 1) + source + 1
            target = rows * (n + 1) + target + 1
        else:
            source = target = relation = torch.empty(0, dtype=torch.long, device=mask.device)
            value = token.new_empty((0, 3))
        return EncodedGraph(token, mask, source, target, relation, value)

    def _routing(self, problem, inputs):
        if not set(inputs) <= ROUTING_INPUT_KEYS:
            raise ValueError('Routing forward accepts raw input fields only')
        batch = dict(inputs)
        batch['problem_id'] = PROBLEMS.index(problem)
        mask = batch['node_mask']
        if batch['kind'] == 'coord':
            node = self.route._pad_node(batch['node'])
            if problem == 'TSP':
                node = node.clone()
                role = node[..., 3]
                first = mask.long().argmax(-1)
                derived = torch.zeros_like(role).scatter(1, first[:, None], 1.)
                node[..., 3] = torch.where(role.sum(-1, keepdim=True) > 0, role, derived)
            stats = self.route._coord_stats(node, mask)
            cond, _ = self.route.condition_encoder(batch, stats, 0)
            tokens = self.route.coord_proj(node) + self.route.geometry_residual(
                node, mask, batch['n'], batch.get('node_geom'))
            bias = self.route._coord_bias(node, mask, 3)
            distance = torch.cdist(node[..., :2].float(), node[..., :2].float())
        else:
            stats = self.route._matrix_stats(batch['matrix'], mask)
            cond, _ = self.route.condition_encoder(batch, stats, 1)
            features, bias = self.route._matrix_features_and_bias(batch['matrix'], mask, 3)
            tokens = self.route.matrix_proj(features)
            distance = batch['matrix'].float()
        pair_mask = mask[:, :, None] & mask[:, None, :]
        pair_mask = pair_mask & ~torch.eye(mask.size(1), dtype=torch.bool, device=mask.device)[None]
        unit = distance.abs().masked_fill(~pair_mask, 0.).sum((1, 2)) / pair_mask.sum((1, 2)).clamp_min(1)
        scaled = distance / unit[:, None, None].clamp_min(1e-6)
        dense_value = torch.stack((scaled, scaled.square(), distance.abs().log1p() / 10.), -1)
        dense_value = dense_value.masked_fill(~pair_mask[..., None], 0.)
        tokens = torch.cat((self.route.cls_token.expand(len(mask), 1, -1),
                            cond[:, None], self.route.stats_token_proj(stats)[:, None], tokens), 1)
        mask = torch.cat((torch.ones((len(mask), 3), dtype=torch.bool, device=mask.device), mask), 1)
        empty = torch.empty(0, dtype=torch.long, device=mask.device)
        return EncodedGraph(tokens, mask, empty, empty, empty, tokens.new_empty((0, 3)),
                            bias, dense_value, pair_mask, 3)

    def forward(self, problem, inputs):
        if problem in PROBLEMS:
            return self._routing(problem, inputs)
        optional = ({'job_execution_rank', 'machine_execution_rank'} if problem in ('JSSP', 'FJSP')
                    else {'execution_rank'})
        if problem not in INPUT_KEYS or set(inputs) - optional != INPUT_KEYS[problem]:
            raise ValueError(f'{problem}: raw typed input fields differ from the declared schema')
        edges = []
        if problem == 'MIS':
            mask = inputs['node_mask']
            f = self._blank(mask)
            b, n = mask.shape
            rows, cols = inputs['edge_mask'].nonzero(as_tuple=True)
            source = inputs['edge_index'][rows, 0, cols]
            target = inputs['edge_index'][rows, 1, cols]
            edges = [self._relation(rows, source, target, 0), self._relation(rows, target, source, 0)]
            degree = torch.zeros(b * n, device=mask.device)
            degree.index_add_(0, rows * n + source, torch.ones_like(source, dtype=torch.float32))
            degree.index_add_(0, rows * n + target, torch.ones_like(target, dtype=torch.float32))
            f[..., 0] = 1.
            f[..., 1] = degree.reshape(b, n) / inputs['num_nodes'].float()[:, None].clamp_min(1.)
            f[..., 2] = self._priority(inputs, mask)
            types = torch.full_like(mask, 2, dtype=torch.long)
            global_features = f.new_zeros((b, 8))
            global_features[:, 0] = inputs['num_nodes'].float().log1p() / 8.
        elif problem in ('JSSP', 'FJSP'):
            op_mask, machine_mask = inputs['operation_mask'], inputs['machine_mask']
            b, o = op_mask.shape
            m = machine_mask.size(1)
            if problem == 'JSSP':
                duration = inputs['duration'].float().masked_fill(~op_mask, 0.)
                compatibility = F.one_hot(inputs['machine'].clamp_min(0), m).bool() & op_mask[..., None]
                times = duration[..., None] * compatibility
            else:
                compatibility = inputs['compatibility'] & op_mask[..., None] & machine_mask[:, None]
                times = inputs['processing_time'].float().masked_fill(~compatibility, 0.)
            count = compatibility.sum((1, 2)).clamp_min(1)
            time_unit = (times.sum((1, 2)) / count).clamp_min(1.)
            op_features, machine_features = self._blank(op_mask), self._blank(machine_mask)
            op_mean = times.sum(-1) / compatibility.sum(-1).clamp_min(1)
            lengths = (inputs['job_ptr'][:, 1:] - inputs['job_ptr'][:, :-1]).clamp_min(1)
            job_length = lengths.gather(1, inputs['operation_job'].clamp_min(0))
            op_features[..., 0] = op_mean / time_unit[:, None]
            op_features[..., 1] = inputs['operation_index'].clamp_min(0).float() / job_length
            op_features[..., 2] = compatibility.sum(-1) / inputs['num_machines'][:, None].clamp_min(1)
            op_features[..., 3] = op_mean.log1p() / 10.
            op_features[..., 4] = job_length.float().log1p() / 8.
            job_priority = self._priority(inputs, inputs['job_mask'], 'job_execution_rank')
            op_features[..., 5] = job_priority.gather(1, inputs['operation_job'].clamp_min(0))
            machine_features[..., 0] = times.sum(1) / (time_unit[:, None] * op_mask.sum(-1)[:, None].clamp_min(1))
            machine_features[..., 1] = compatibility.sum(1) / op_mask.sum(-1)[:, None].clamp_min(1)
            machine_features[..., 2] = self._priority(inputs, machine_mask, 'machine_execution_rank')
            mask = torch.cat((op_mask, machine_mask), 1)
            f = torch.cat((op_features, machine_features), 1)
            types = torch.cat((torch.full_like(op_mask, 3, dtype=torch.long),
                               torch.full_like(machine_mask, 4, dtype=torch.long)), 1)
            rows, cols = inputs['precedence_mask'].nonzero(as_tuple=True)
            first = inputs['precedence_edge_index'][rows, 0, cols]
            second = inputs['precedence_edge_index'][rows, 1, cols]
            edges += [self._relation(rows, first, second, 1), self._relation(rows, second, first, 2)]
            rows, op, machine = compatibility.nonzero(as_tuple=True)
            value = torch.stack((times[rows, op, machine] / time_unit[rows],
                                 times[rows, op, machine].log1p() / 10.,
                                 torch.ones_like(rows, dtype=torch.float32)), -1)
            edges += [self._relation(rows, op, o + machine, 3, value),
                      self._relation(rows, o + machine, op, 4, value)]
            global_features = f.new_zeros((b, 8))
            global_features[:, 0] = op_mask.sum(-1).float().log1p() / 8.
            global_features[:, 1] = inputs['num_jobs'].float().log1p() / 8.
            global_features[:, 2] = inputs['num_machines'].float().log1p() / 8.
            global_features[:, 3] = time_unit.log1p() / 10.
        elif problem == 'BPP':
            item_mask = inputs['item_mask']
            b, n = item_mask.shape
            capacity = inputs['capacity'].float().clamp_min(1)
            item = self._blank(item_mask)
            item[..., 0] = inputs['sizes'].clamp_min(0).float() / capacity[:, None]
            item[..., 1] = inputs['sizes'].clamp_min(0).float().log1p() / 10.
            item[..., 2] = self._priority(inputs, item_mask)
            cap = item.new_zeros((b, 1, 8))
            cap[:, 0, 0] = capacity.log1p() / 12.
            f = torch.cat((item, cap), 1)
            mask = torch.cat((item_mask, torch.ones((b, 1), device=item.device, dtype=torch.bool)), 1)
            types = torch.cat((torch.full_like(item_mask, 5, dtype=torch.long),
                               torch.full((b, 1), 8, dtype=torch.long, device=item.device)), 1)
            rows, ix = item_mask.nonzero(as_tuple=True)
            target = torch.full_like(ix, n)
            value = torch.stack((item[rows, ix, 0], item[rows, ix, 1], torch.ones_like(ix).float()), -1)
            edges += [self._relation(rows, ix, target, 7, value), self._relation(rows, target, ix, 8, value)]
            global_features = item.new_zeros((b, 8))
            global_features[:, 0] = item_mask.sum(-1).float().log1p() / 8.
            global_features[:, 3] = capacity.log1p() / 12.
        elif problem == 'MKP':
            items, resources = inputs['item_mask'], inputs['resource_mask']
            b, n = items.shape
            r = resources.size(1)
            profits = inputs['profits'].float().masked_fill(~items, 0.)
            mean_profit = (profits.sum(-1) / items.sum(-1).clamp_min(1)).clamp_min(1.)
            consumption = inputs['resources'].float().masked_fill(~(items[..., None] & resources[:, None]), 0.)
            capacity = inputs['capacities'].float().clamp_min(1.)
            item, resource = self._blank(items), self._blank(resources)
            item[..., 0] = profits / mean_profit[:, None]
            item[..., 1] = profits.log1p() / 12.
            item[..., 2] = self._priority(inputs, items)
            resource[..., 0] = capacity.log1p() / 16.
            resource[..., 1] = capacity / consumption.sum(1).clamp_min(1.)
            f, mask = torch.cat((item, resource), 1), torch.cat((items, resources), 1)
            types = torch.cat((torch.full_like(items, 5, dtype=torch.long),
                               torch.full_like(resources, 6, dtype=torch.long)), 1)
            rows, ix, rx = (items[..., None] & resources[:, None]).nonzero(as_tuple=True)
            value = torch.stack((consumption[rows, ix, rx] / capacity[rows, rx],
                                 consumption[rows, ix, rx].log1p() / 10.,
                                 profits[rows, ix] / mean_profit[rows]), -1)
            edges += [self._relation(rows, ix, n + rx, 5, value), self._relation(rows, n + rx, ix, 6, value)]
            global_features = item.new_zeros((b, 8))
            global_features[:, 0] = items.sum(-1).float().log1p() / 8.
            global_features[:, 2] = resources.sum(-1).float().log1p() / 8.
            global_features[:, 3] = mean_profit.log1p() / 12.
        else:
            jobs = inputs['job_mask']
            b, n = jobs.shape
            p = inputs['p'].float().masked_fill(~jobs, 0.)
            w = inputs['w'].float().masked_fill(~jobs, 0.)
            due = inputs['d'].float().masked_fill(~jobs, 0.)
            total_p = p.sum(-1).clamp_min(1.)
            mean_w = (w.sum(-1) / jobs.sum(-1).clamp_min(1)).clamp_min(1.)
            job = self._blank(jobs)
            job[..., 0], job[..., 1] = p / total_p[:, None], w / mean_w[:, None]
            job[..., 2], job[..., 3] = due / total_p[:, None], p.log1p() / 10.
            job[..., 4] = self._priority(inputs, jobs)
            machine = job.new_zeros((b, 1, 8))
            machine[:, 0, 0] = total_p.log1p() / 12.
            machine[:, 0, 1] = mean_w.log1p() / 8.
            f = torch.cat((job, machine), 1)
            mask = torch.cat((jobs, torch.ones((b, 1), dtype=torch.bool, device=jobs.device)), 1)
            types = torch.cat((torch.full_like(jobs, 7, dtype=torch.long),
                               torch.full((b, 1), 4, dtype=torch.long, device=jobs.device)), 1)
            rows, ix = jobs.nonzero(as_tuple=True)
            target = torch.full_like(ix, n)
            value = job[rows, ix, :3]
            edges += [self._relation(rows, ix, target, 7, value), self._relation(rows, target, ix, 8, value)]
            global_features = job.new_zeros((b, 8))
            global_features[:, 0] = jobs.sum(-1).float().log1p() / 8.
            global_features[:, 3] = total_p.log1p() / 12.
            global_features[:, 4] = mean_w.log1p() / 8.
        f = f.masked_fill(~mask[..., None], 0.)
        return self._finish(problem, f, types, mask, edges, global_features)


class RelationalLayer(nn.Module):
    def __init__(self, params):
        super().__init__()
        d = params['embedding_dim']
        self.source = nn.Linear(d, d, bias=False)
        self.edge_value = nn.Linear(3, d, bias=False)
        self.relation = nn.Embedding(params['relation_types'], d)
        nn.init.normal_(self.relation.weight, std=.02)
        self.gate = nn.Linear(3, 1)
        self.local_output = nn.Linear(d, d, bias=False)
        self.local_norm = nn.LayerNorm(d)
        self.dropout = nn.Dropout(params['dropout'])
        self.global_layer = EncoderLayer(**params)

    def forward(self, nodes, graph):
        if graph.source.numel():
            values = self.source(nodes).reshape(-1, nodes.size(-1))
            message = values[graph.source] + self.edge_value(graph.edge_value) + self.relation(graph.relation)
            message = message * self.gate(graph.edge_value).sigmoid()
            # The sparse native graph is complete; padding never creates messages.
            pooled = values.new_zeros(values.shape).index_add(0, graph.target, message.to(values.dtype))
            count = torch.bincount(graph.target, minlength=len(values)).clamp_min(1)
            pooled = (pooled / count[:, None]).reshape_as(nodes)
            nodes = self.local_norm(nodes + self.dropout(self.local_output(pooled)))
        elif graph.dense_value is not None:
            offset = graph.dense_offset
            values = self.source(nodes[:, offset:])
            weights = self.gate(graph.dense_value).squeeze(-1).sigmoid()
            weights = weights.masked_fill(~graph.dense_mask, 0.)
            count = graph.dense_mask.sum(-1, keepdim=True).clamp_min(1)
            pooled = torch.bmm(weights.to(values.dtype), values) / count
            edge_mean = (weights[..., None] * graph.dense_value).sum(2) / count
            pooled = pooled + self.edge_value(edge_mean) + (
                weights.sum(-1, keepdim=True) / count) * self.relation.weight[9]
            local = torch.cat((torch.zeros_like(nodes[:, :offset]), self.local_output(pooled)), 1)
            nodes = self.local_norm(nodes + self.dropout(local))
        nodes = self.global_layer(nodes, mask=graph.mask, attn_bias=graph.attention_bias)
        return nodes.masked_fill(~graph.mask[..., None], 0.)


class DeploymentEncoder(nn.Module):
    def __init__(self, params):
        super().__init__()
        names, d = params['solver_names'], params['embedding_dim']
        families = params.get('solver_families') or [method_family(name) for name in names]
        if len(families) != len(names):
            raise ValueError('Every deployment needs one explicit family')
        vocab = {name: i for i, name in enumerate(sorted(set(families)))}
        self.register_buffer('family_ids', torch.tensor([vocab[name] for name in families]))
        self.deployment = nn.Embedding(len(names), d)
        self.family = nn.Embedding(len(vocab), d)
        self.known_project = nn.Sequential(nn.Linear(35, d), nn.GELU(), nn.Linear(d, d))
        known = torch.zeros(len(names), 35)
        old_features = build_solver_features(params['solver_feature_spec'])
        mapping = {name: i for i, name in enumerate(GLOBAL_SOLVERS)}
        for i, name in enumerate(names):
            old_name = 'RELD_MOEL' if name == 'reld_mt' else name
            if old_name in mapping:
                known[i, :34], known[i, 34] = old_features[mapping[old_name]], 1.
        self.register_buffer('known_features', known)
        self.norm = nn.LayerNorm(d)

    def forward(self, ids):
        return self.norm(self.deployment(ids) + self.family(self.family_ids[ids])
                         + self.known_project(self.known_features[ids]))


class MultidomainSelector(nn.Module):
    """One shared graph/candidate path and one decision over the legal pool."""

    def __init__(self, config):
        super().__init__()
        self.config = default_config(config['solver_names'], config.get('problem_names', ALL_PROBLEMS), **{
            key: value for key, value in config.items() if key not in ('solver_names', 'problem_names')})
        p, d = self.config, self.config['embedding_dim']
        self.frontend = TypedFrontend(p)
        self.layers = nn.ModuleList([RelationalLayer(p) for _ in range(p['encoder_layer_num'])])
        self.solver_encoder = DeploymentEncoder(p)
        self.candidate_task = nn.Embedding(len(p['problem_names']), d)
        self.joint_layers = nn.ModuleList([JointEncoderLayer(**p) for _ in range(p['joint_layer_num'])])
        self.candidate_layer = EncoderLayer(**p)
        self.pool = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Linear(d, d))
        self.decoder = MultiHeadAttention(d, p['head_num'], p['qkv_dim'], p['dropout'], p['sdpa'])
        self.score_head = nn.Sequential(nn.Linear(3 * d, p['head_hidden_dim']), nn.GELU(),
                                        nn.Dropout(p['dropout']), nn.Linear(p['head_hidden_dim'], 1))
        self.advantage_head = nn.Sequential(nn.Linear(3 * d, d), nn.GELU(), nn.Linear(d, 1))

    def forward(self, batch):
        if set(batch) - {'problem', 'inputs', 'pool_ids', 'solver_mask'}:
            raise ValueError('Labels, solutions and metadata must stay outside the model forward')
        problem = batch['problem']
        graph = self.frontend(problem, batch['inputs'])
        nodes = graph.nodes
        for layer in self.layers:
            nodes = layer(nodes, graph)
        ids = batch['pool_ids'].to(nodes.device)
        if ids.dim() == 1:
            ids = ids[None].expand(nodes.size(0), -1)
        mask = batch.get('solver_mask', torch.ones_like(ids, dtype=torch.bool)).to(nodes.device)
        ids = ids.masked_fill(~mask, 0)
        task_id = self.config['problem_names'].index(problem)
        solvers = self.solver_encoder(ids) + self.candidate_task.weight[task_id]
        for layer in self.joint_layers:
            nodes, solvers = layer(nodes, solvers, graph.mask, mask)
        solvers = self.candidate_layer(solvers, mask=mask).masked_fill(~mask[..., None], 0.)
        h = self.pool(torch.cat((_masked_mean(nodes, graph.mask), _masked_max(nodes, graph.mask)), -1))
        context = self.decoder(h[:, None], solvers, solvers, key_mask=mask)
        features = torch.cat((h[:, None].expand_as(solvers), context.expand_as(solvers), solvers), -1)
        utility = self.score_head(features).squeeze(-1).float()
        margins = utility[:, :, None] - utility[:, None, :]
        margins = margins.masked_fill(~(mask[:, :, None] & mask[:, None, :]), 0.)
        advantage = self.advantage_head(features).squeeze(-1).float()
        average = advantage.masked_fill(~mask, 0.).sum(-1, keepdim=True) / mask.sum(-1, keepdim=True).clamp_min(1)
        advantage = (advantage - average).masked_fill(~mask, 0.)
        return dict(utility=utility.masked_fill(~mask, float('-inf')),
                    logits=maximin_logits(margins, mask), pair_margin=margins,
                    solver_mask=mask, advantage=advantage)


def make_model(config, state_dict=None):
    model = MultidomainSelector(config)
    if state_dict is not None:
        model.load_state_dict(state_dict, strict=True)
    return model
