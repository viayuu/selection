"""Frozen complete source views; shared projection, no learnable solver identity."""

import hashlib
from collections import defaultdict

import torch
from torch import nn

from code.unified_selector.registry import GLOBAL_SOLVERS
from .solver_code_encoder import CODE_DIM, ROLES, check_components, check_vectors, identity_digest


class SolverCodeTokens(nn.Module):
    def __init__(self, vectors, role_mask, chunk_vectors, chunk_indices, embedding_dim, **params):
        super().__init__()
        check_vectors(vectors, role_mask)
        check_components(chunk_vectors, chunk_indices, role_mask)
        for name, value in dict(code_vectors=vectors, role_mask=role_mask,
                                chunk_vectors=chunk_vectors, chunk_indices=chunk_indices).items():
            self.register_buffer(name, value.detach().clone())
        self.register_buffer('solver_identity', identity_digest())
        roles = torch.arange(len(ROLES)).repeat_interleave(chunk_indices.shape[2] * chunk_indices.shape[3])
        self.register_buffer('token_roles', roles)
        self.projection = nn.Sequential(nn.LayerNorm(CODE_DIM, elementwise_affine=False),
                                        nn.Linear(CODE_DIM, embedding_dim), nn.GELU())
        self.role_embedding = nn.Embedding(len(ROLES), embedding_dim)
        self.norm = nn.LayerNorm(embedding_dim)

    def forward(self, solver_ids, solver_mask):
        indices = self.chunk_indices.flatten(1)[solver_ids]
        mask = (indices >= 0) & solver_mask[..., None]
        # Project the unique source bank once, then retain every deployment/role slot.
        # code_vectors contains legacy role summaries for provenance validation ONLY.
        projected = self.projection(self.chunk_vectors)
        tokens = self.norm(projected[indices.clamp_min(0)] + self.role_embedding(self.token_roles))
        return tokens.masked_fill(~mask[..., None], 0.), mask


class DynamicSourcePool(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.instance = nn.Linear(embedding_dim, embedding_dim)
        self.code = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.score = nn.Linear(embedding_dim, 1, bias=False)
        self.norm = nn.LayerNorm(embedding_dim)

    def forward(self, instance, codes, mask):
        score = self.score(torch.nn.functional.gelu(self.instance(instance)[:, None, None] + self.code(codes)))
        score = score.squeeze(-1).float().masked_fill(~mask, -1e9)
        weights = score.softmax(-1) * mask
        weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-12)
        solver = self.norm((codes * weights[..., None]).sum(-2))
        return solver.masked_fill(~mask.any(-1)[..., None], 0.), weights


def source_package_audit(bundle, pools):
    """Order-independent equality of full role/vector multisets, not just mean vectors."""
    bank, indices = bundle['chunk_vectors'], bundle['chunk_indices']
    groups, counts = defaultdict(list), {}
    for name, row in zip(GLOBAL_SOLVERS, indices):
        pieces = []
        for role, entries in enumerate(row):
            for index in entries.flatten().tolist():
                if index >= 0:
                    pieces.append((role, hashlib.sha256(bank[index].numpy().tobytes()).hexdigest()))
        groups[tuple(sorted(pieces))].append(name)
        counts[name] = len(pieces)
    duplicates = [names for names in groups.values() if len(names) > 1]
    in_pool = {p: [names for names in duplicates if len(set(names) & set(pool)) > 1]
               for p, pool in pools.items()}
    return dict(token_counts=counts, global_slots=int(indices[0].numel()),
                identical_packages=duplicates, identical_in_pool={p: v for p, v in in_pool.items() if v},
                limitation='Identical role/vector packages cannot be distinguished without solver-specific inputs; '
                           'exact score ties use ascending global solver ID. No identity or position is added.')
