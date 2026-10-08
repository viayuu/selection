"""Instance/solver co-encoding with synchronous bidirectional cross-attention."""

import torch
from torch import nn

from code.unified_selector.registry import M_GLOBAL
from .solver_features import SOLVER_FEATURE_DIM, build_solver_features
from .V4Model import InstanceEncoder, MultiHeadAttention, FeedForward, _masked_mean, _masked_max


class SolverFeatureEncoder(nn.Module):
    def __init__(self, **params):
        super().__init__()
        d = params["embedding_dim"]
        self.register_buffer("solver_features", build_solver_features(params["solver_feature_spec"]))
        self.solver_emb = nn.Embedding(M_GLOBAL, d)
        self.feature_mlp = nn.Sequential(
            nn.Linear(SOLVER_FEATURE_DIM, params["solver_feature_hidden"]),
            nn.GELU(),
            nn.Dropout(params["dropout"]),
            nn.Linear(params["solver_feature_hidden"], d),
        )
        self.weight = params.get("solver_feature_weight", 1.0)
        self.norm = nn.LayerNorm(d)

    def forward(self, solver_ids):
        features = self.feature_mlp(self.solver_features[solver_ids])
        return self.norm(self.solver_emb(solver_ids) + self.weight * features)


class CrossUpdate(nn.Module):
    def __init__(self, **params):
        super().__init__()
        d = params["embedding_dim"]
        self.attn = MultiHeadAttention(d, params["head_num"], params["qkv_dim"], params["dropout"], params.get("sdpa", False))
        self.ffn = FeedForward(d, params["ff_hidden_dim"], params["dropout"])
        self.norm1 = nn.LayerNorm(d)
        self.norm2 = nn.LayerNorm(d)
        self.dropout = nn.Dropout(params["dropout"])

    def forward(self, query, memory, query_mask, memory_mask):
        x = self.norm1(query + self.dropout(self.attn(query, memory, memory, key_mask=memory_mask)))
        x = self.norm2(x + self.dropout(self.ffn(x)))
        return x.masked_fill(~query_mask.unsqueeze(-1), 0.0)


class JointEncoderLayer(nn.Module):
    def __init__(self, **params):
        super().__init__()
        self.node_update = CrossUpdate(**params)
        self.solver_update = CrossUpdate(**params)

    def forward(self, nodes, solvers, node_mask, solver_mask):
        # Both directions read the previous layer, not the other direction's update.
        next_nodes = self.node_update(nodes, solvers, node_mask, solver_mask)
        next_solvers = self.solver_update(solvers, nodes, solver_mask, node_mask)
        return next_nodes, next_solvers


class DualStreamSelector(nn.Module):
    def __init__(self, **params):
        super().__init__()
        self.params = params
        d = params["embedding_dim"]
        self.instance_encoder = InstanceEncoder(**dict(params, node_only=True))
        self.solver_encoder = SolverFeatureEncoder(**params)
        self.joint_layers = nn.ModuleList([JointEncoderLayer(**params) for _ in range(params.get("joint_layer_num", 2))])
        self.pool = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Linear(d, d))
        self.decoder = MultiHeadAttention(d, params["head_num"], params["qkv_dim"], params["dropout"], params.get("sdpa", False))
        self.score_head = nn.Sequential(
            nn.Linear(3 * d, params["head_hidden_dim"]), nn.GELU(),
            nn.Dropout(params["dropout"]), nn.Linear(params["head_hidden_dim"], 1),
        )

    def adapt_nodes(self, nodes, node_mask, batch):
        return nodes

    def encode_state(self, batch):
        nodes, node_mask = self.instance_encoder(batch)
        nodes = self.adapt_nodes(nodes, node_mask, batch)
        ids = batch["pool_ids"].to(nodes.device)
        if ids.dim() == 1:
            ids = ids.unsqueeze(0).expand(nodes.size(0), -1)
        solver_mask = batch.get("solver_mask")
        if solver_mask is None:
            solver_mask = torch.ones_like(ids, dtype=torch.bool)
        else:
            solver_mask = solver_mask.to(device=nodes.device, dtype=torch.bool)
            if solver_mask.shape != ids.shape or not solver_mask.any(dim=1).all():
                raise ValueError("solver_mask must match pool_ids and retain at least one solver per instance")
        ids = ids.masked_fill(~solver_mask, 0)
        solvers = self.solver_encoder(ids).masked_fill(~solver_mask.unsqueeze(-1), 0.0)
        for layer in self.joint_layers:
            nodes, solvers = layer(nodes, solvers, node_mask, solver_mask)
        h = self.pool(torch.cat([_masked_mean(nodes, node_mask), _masked_max(nodes, node_mask)], dim=-1))
        return nodes, solvers, h, node_mask, solver_mask

    def encode_pairs(self, batch):
        nodes, solvers, h, node_mask, solver_mask = self.encode_state(batch)
        context = self.decoder(h[:, None, :], solvers, solvers, key_mask=solver_mask)
        features = torch.cat([h[:, None, :].expand_as(solvers), context.expand_as(solvers), solvers], dim=-1)
        return features, solver_mask

    def forward(self, batch):
        features, solver_mask = self.encode_pairs(batch)
        logits = self.score_head(features).squeeze(-1).masked_fill(~solver_mask, float("-inf"))
        return {"logits": logits}


class PerformanceSelector(DualStreamSelector):
    """R39: independent winner classification and centered performance estimates."""

    def __init__(self, **params):
        super().__init__(**params)
        from code.unified_selector.registry import PROBLEMS
        self.register_buffer("performance_scales", torch.ones(len(PROBLEMS), dtype=torch.float64))
        d, hidden = params["embedding_dim"], params["head_hidden_dim"]
        self.performance_head = nn.Sequential(
            nn.Linear(3 * d, hidden), nn.GELU(),
            nn.Dropout(params["dropout"]), nn.Linear(hidden, 1),
        )
        self.decision_head = params.get("decision_head", "performance")
        if self.decision_head not in ("classification", "performance"):
            raise ValueError("decision_head must be classification or performance")

    def forward(self, batch):
        from .performance_targets import center_valid
        features, mask = self.encode_pairs(batch)
        logits = self.score_head(features).squeeze(-1).masked_fill(~mask, float("-inf"))
        prediction = center_valid(self.performance_head(features).squeeze(-1).float(), mask)
        prediction = prediction.masked_fill(~mask, float("inf"))
        selection = logits if self.decision_head == "classification" else -prediction
        return dict(logits=logits, pred_performance=prediction, solver_mask=mask, selection_scores=selection)
