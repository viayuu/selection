"""Frozen R73 features and the one R74 finite acquisition-tree policy."""

import torch
from torch import nn

from .r64_model import NativeUtilitySelector
from .r67_model import model_inputs


PROBE_NAMES = ('BQ', 'RELD_CVRP', 'MATNET', 'RELD_MTL', 'RELD_MOEL')
TOKEN_DIM = 403
INFERENCE_RULE = ('hard root MTL/MOEL then hard deepen/switch; ties MTL/deepen; '
                  'retain shallow observations; F utility plus correction; full legal pool; global-ID ties')


def choose_global(logits, ids, mask):
    ids = ids[None].expand_as(logits) if ids.ndim == 1 else ids
    if not mask.any(-1).all() or not torch.isfinite(logits[mask]).all():
        raise ValueError('Every legal utility must be finite and every pool nonempty')
    scores = logits.float().masked_fill(~mask, -torch.inf)
    maximum = scores.max(-1, keepdim=True).values
    selected = ids.masked_fill(~mask | (scores != maximum), torch.iinfo(torch.long).max).min(-1).values
    local = (ids == selected[:, None]).long().argmax(-1)
    return local, selected


class CandidateBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.attention = nn.MultiheadAttention(128, 4, dropout=.1, batch_first=True)
        self.ff = nn.Sequential(nn.Linear(128, 256), nn.GELU(), nn.Dropout(.1), nn.Linear(256, 128))
        self.norm1, self.norm2 = nn.LayerNorm(128), nn.LayerNorm(128)
        self.dropout = nn.Dropout(.1)

    def forward(self, query, observed, observed_mask, candidate_mask):
        attended = self.attention(query, observed, observed, key_padding_mask=~observed_mask,
                                  need_weights=False)[0]
        query = self.norm1(query + self.dropout(attended))
        query = self.norm2(query + self.dropout(self.ff(query)))
        return query.masked_fill(~candidate_mask[..., None], 0.)


class AcquisitionPolicy(nn.Module):
    """Only this module's parameters enter the fresh official optimizer."""
    def __init__(self):
        super().__init__()
        self.token_adapter = nn.Sequential(nn.Linear(TOKEN_DIM, 128), nn.GELU(), nn.LayerNorm(128))
        self.candidate_projection = nn.Linear(384, 128)
        self.blocks = nn.ModuleList([CandidateBlock(), CandidateBlock()])
        self.correction = nn.Linear(128, 1)
        self.root_gate = nn.Sequential(nn.Linear(128, 128), nn.GELU(), nn.Linear(128, 2))
        self.second_gate = nn.Sequential(nn.Linear(258, 128), nn.GELU(), nn.Linear(128, 2))
        for final in (self.correction, self.root_gate[-1], self.second_gate[-1]):
            nn.init.zeros_(final.weight)
            nn.init.zeros_(final.bias)

    def root(self, instance_summary):
        with torch.autocast(device_type=instance_summary.device.type, enabled=False):
            return self.root_gate(instance_summary.float()).softmax(-1)

    def second(self, instance_summary, first_shallow, first_mask, first_probe):
        # The signature deliberately has no provider, deep state or other-probe observation.
        with torch.autocast(device_type=instance_summary.device.type, enabled=False):
            adapted = self.token_adapter(first_shallow.float())
            mean = adapted.masked_fill(~first_mask[..., None], 0.).sum(1) / first_mask.sum(1, keepdim=True)
            identity = torch.zeros(len(mean), 2, device=mean.device)
            identity[:, first_probe] = 1.
            return self.second_gate(torch.cat((instance_summary.float(), mean, identity), -1)).softmax(-1)

    def weights(self, batch):
        if not batch['mvrp']:
            return batch['h'].new_ones(len(batch['h']), 1), {}
        root = self.root(batch['h'])
        mtl, mtl_mask = batch['observations'](3, 0)
        moel, moel_mask = batch['observations'](4, 0)
        after_mtl = self.second(batch['h'], mtl, mtl_mask, 0)
        after_moel = self.second(batch['h'], moel, moel_mask, 1)
        a, b, c = root[:, 0], after_mtl[:, 0], after_moel[:, 0]
        weights = torch.stack((a * b, (1-a) * c, a * (1-b) + (1-a) * (1-c)), -1)
        return weights, dict(root_mtl=root[:, 0], deepen_after_mtl=b, deepen_after_moel=c)

    @staticmethod
    def leaf_keys(batch, leaf):
        if not batch['mvrp']:
            return ((batch['probe_id'], 0), (batch['probe_id'], 1))
        return (((3, 0), (3, 1)), ((4, 0), (4, 1)), ((3, 0), (4, 0)))[leaf]

    def leaf_logits(self, batch, leaf, positions=None):
        keys = self.leaf_keys(batch, leaf)
        observations = [batch['observations'](probe, checkpoint, positions) for probe, checkpoint in keys]
        tokens = torch.cat([value[0] for value in observations], 1)
        token_mask = torch.cat([value[1] for value in observations], 1)
        features, base, mask = batch['features'], batch['base_logits'], batch['solver_mask']
        if positions is not None:
            features, base, mask = features[positions], base[positions], mask[positions]
        memory = self.token_adapter(tokens)
        query = self.candidate_projection(features)
        for block in self.blocks:
            query = block(query, memory, token_mask, mask)
        return (base.float() + self.correction(query).squeeze(-1).float()).masked_fill(~mask, -torch.inf)

    def forward(self, batch):
        rows = len(batch['h'])
        if not batch['mvrp']:
            logits = self.leaf_logits(batch, 0)
            paths = torch.zeros(rows, device=logits.device, dtype=torch.long)
            probabilities = {}
        else:
            root = self.root(batch['h'])
            first = root.argmax(-1)
            logits = torch.empty_like(batch['base_logits'])
            paths = torch.empty(rows, device=logits.device, dtype=torch.long)
            deepen = torch.empty(rows, device=logits.device)
            for branch, probe in ((0, 3), (1, 4)):
                positions = (first == branch).nonzero(as_tuple=True)[0]
                if not len(positions):
                    continue
                shallow, shallow_mask = batch['observations'](probe, 0, positions)
                second = self.second(batch['h'][positions], shallow, shallow_mask, branch)
                action = second.argmax(-1)
                deepen[positions] = second[:, 0]
                for decision in (0, 1):
                    selected = positions[action == decision]
                    if not len(selected):
                        continue
                    leaf = branch if decision == 0 else 2
                    logits[selected] = self.leaf_logits(batch, leaf, selected)
                    paths[selected] = leaf
            probabilities = dict(root_mtl=root[:, 0], selected_deepen=deepen)
        local, global_id = choose_global(logits, batch['pool_ids'], batch['solver_mask'])
        return dict(logits=logits, solver_mask=batch['solver_mask'], pred=local,
                    pred_global_id=global_id, hard_paths=paths, gate_probabilities=probabilities)


class R74Model(nn.Module):
    def __init__(self, base_params):
        super().__init__()
        self.base = NativeUtilitySelector(**base_params).requires_grad_(False).float().eval()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(2)
            self.policy = AcquisitionPolicy()

    def train(self, mode=True):
        super().train(mode)
        self.base.eval()
        return self

    @torch.no_grad()
    def extract_base(self, inputs):
        self.base.eval()
        with torch.autocast(device_type=next(self.base.parameters()).device.type, enabled=False):
            nodes, solvers, summary, node_mask, solver_mask = self.base.encode_state(model_inputs(inputs))
            context = self.base.decoder(summary[:, None], solvers, solvers, key_mask=solver_mask)
            features = torch.cat((summary[:, None].expand_as(solvers), context.expand_as(solvers), solvers), -1)
            utility = self.base.score_head(features).squeeze(-1).float().masked_fill(~solver_mask, -torch.inf)
        if not all(torch.isfinite(value).all() for value in (nodes, features, summary, utility[solver_mask])):
            raise ValueError('Nonfinite formal frozen-base features')
        return dict(nodes=nodes, features=features, h=summary, base_logits=utility,
                    solver_mask=solver_mask, node_mask=node_mask)

    def forward(self, batch):
        return self.policy(batch)
