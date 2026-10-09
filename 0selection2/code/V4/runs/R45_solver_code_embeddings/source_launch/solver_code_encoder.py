"""R45: fixed source embeddings, trainable role projections, and solver identity."""

import hashlib
import json

import torch
from torch import nn

from code.unified_selector.registry import GLOBAL_SOLVERS, M_GLOBAL
from .pairwise_selector import ScoreDifferenceSelector


ROLES = ('encoder', 'decision', 'inference', 'config')
CODE_DIM = 1024


def identity_digest():
    value = json.dumps(GLOBAL_SOLVERS, ensure_ascii=True).encode()
    return torch.tensor(list(hashlib.sha256(value).digest()), dtype=torch.uint8)


def derangement(size=M_GLOBAL, seed=4502):
    if size < 2:
        raise ValueError('A shuffled-code control requires at least two solvers')
    generator = torch.Generator().manual_seed(seed)
    original = torch.arange(size)
    for _ in range(10000):
        permutation = torch.randperm(size, generator=generator)
        if (permutation != original).all():
            return permutation
    raise RuntimeError('Failed to construct the fixed derangement')


def check_vectors(vectors, mask):
    if vectors.shape != (M_GLOBAL, len(ROLES), CODE_DIM) or mask.shape != (M_GLOBAL, len(ROLES)):
        raise ValueError('Code embeddings must align with GLOBAL_SOLVERS and the four fixed roles')
    if vectors.dtype != torch.float32 or mask.dtype != torch.bool or not torch.isfinite(vectors).all():
        raise ValueError('Expected finite FP32 code vectors and a boolean role mask')
    if not mask[:, :2].all():
        raise ValueError('Missing encoder/decision source is not a valid absent role')
    if not torch.allclose(vectors.norm(dim=-1)[mask], torch.ones_like(vectors.norm(dim=-1)[mask]), atol=1e-5):
        raise ValueError('Present code vectors must be L2 normalized')
    if torch.count_nonzero(vectors[~mask]):
        raise ValueError('Absent roles must contain zeros and have a false mask')


def check_components(vectors, indices, role_mask):
    if vectors.ndim != 2 or vectors.shape[1] != CODE_DIM or not len(vectors):
        raise ValueError('Component bank must contain all 1024-dimensional chunk vectors')
    if vectors.dtype != torch.float32 or not torch.isfinite(vectors).all():
        raise ValueError('Component embeddings must be finite FP32 vectors')
    if not torch.allclose(vectors.norm(dim=-1), torch.ones(len(vectors), device=vectors.device), atol=1e-5):
        raise ValueError('Component embeddings must be L2 normalized')
    if indices.ndim != 4 or indices.shape[:2] != (M_GLOBAL, len(ROLES)) or indices.dtype != torch.long:
        raise ValueError('Component indices must be [solver, role, variant, chunk]')
    if not indices.shape[2] or not indices.shape[3] or (indices < -1).any() or (indices >= len(vectors)).any():
        raise ValueError('Invalid component bank indices')
    if not torch.equal((indices >= 0).any(dim=-1).any(dim=-1), role_mask):
        raise ValueError('Component availability must match the solver role mask')


class SolverCodeEncoder(nn.Module):
    def __init__(self, vectors, mask, rows, components=None, **params):
        super().__init__()
        check_vectors(vectors, mask)
        d = params['embedding_dim']
        if d % len(ROLES):
            raise ValueError('embedding_dim must be divisible by four')
        if not torch.equal(rows.sort().values.cpu(), torch.arange(M_GLOBAL)):
            raise ValueError('semantic_rows must be a permutation, never a change of solver IDs')
        self.register_buffer('code_vectors', vectors.detach().clone())
        self.register_buffer('role_mask', mask.detach().clone())
        self.register_buffer('semantic_rows', rows.detach().clone().long())
        self.register_buffer('solver_identity', identity_digest())
        self.solver_emb = nn.Embedding(M_GLOBAL, d)
        self.role_proj = nn.ModuleList([
            nn.Sequential(nn.LayerNorm(CODE_DIM, elementwise_affine=False),
                          nn.Linear(CODE_DIM, d // 4), nn.GELU()) for _ in ROLES
        ])
        if components is not None:
            check_components(components['chunk_vectors'], components['chunk_indices'], mask)
            for name in ('chunk_vectors', 'chunk_indices'):
                self.register_buffer(name, components[name].detach().clone())
            if self.chunk_indices.shape[-1] > 1:
                r = d // 4
                self.component_query = nn.Parameter(torch.randn(len(ROLES), r) / r ** .5)
                self.component_fusion = nn.ModuleList([
                    nn.Sequential(nn.Linear(2 * r, r), nn.GELU()) for _ in ROLES
                ])
        self.fusion = nn.Sequential(nn.Linear(d + 4, d), nn.GELU(),
                                    nn.Dropout(params['dropout']), nn.Linear(d, d))
        self.norm = nn.LayerNorm(d)

    def forward(self, solver_ids):
        rows = self.semantic_rows[solver_ids]
        mask = self.role_mask[rows]
        projected = [self.project_role(rows, role) * mask[..., role, None]
                     for role in range(len(ROLES))]
        code = self.fusion(torch.cat(projected + [mask.to(projected[0].dtype)], dim=-1))
        return self.norm(self.solver_emb(solver_ids) + code)

    def project_role(self, rows, role):
        module = self.role_proj[role]
        if not hasattr(self, 'chunk_vectors'):
            return module(self.code_vectors[rows, role])
        indices = self.chunk_indices[rows, role]
        valid = indices >= 0
        parts = module(self.chunk_vectors[indices.clamp_min(0)])
        count = valid.sum(dim=-1, keepdim=True)
        single = (parts * valid[..., None]).sum(dim=-2)
        if hasattr(self, 'component_query'):
            logits = (parts * self.component_query[role]).sum(dim=-1) / parts.shape[-1] ** .5
            logits = logits.masked_fill(~valid, -1e4)
            weight = logits.softmax(dim=-1) * valid
            attended = (parts * weight[..., None]).sum(dim=-2)
            maximum = parts.masked_fill(~valid[..., None], -1e4).amax(dim=-2)
            merged = self.component_fusion[role](torch.cat((attended, maximum), dim=-1))
            # A whole semantic view stays intact; only genuine overflow uses aggregation.
            single = torch.where(count > 1, merged, single)
        variants = count.squeeze(-1) > 0
        return (single * variants[..., None]).sum(dim=-2) / variants.sum(dim=-1, keepdim=True).clamp_min(1)


def make_r45_model(params, bundle=None, state_dict=None):
    """The old path is unchanged; checkpoint reconstruction never needs the API."""
    mode = params.get('solver_representation', 'handcrafted')
    if mode not in ('handcrafted', 'code', 'shuffled_code') or params.get('pair_mode') != 'score_difference':
        raise ValueError('R45 uses R43A score_difference with handcrafted/code/shuffled_code')
    model = ScoreDifferenceSelector(**params)
    if mode != 'handcrafted':
        components = None
        if state_dict is not None:
            vectors = state_dict['solver_encoder.code_vectors'].cpu()
            mask = state_dict['solver_encoder.role_mask'].cpu()
            rows = state_dict['solver_encoder.semantic_rows'].cpu()
            if 'solver_encoder.chunk_vectors' in state_dict:
                components = {name: state_dict['solver_encoder.' + name].cpu()
                              for name in ('chunk_vectors', 'chunk_indices')}
            if not torch.equal(state_dict['solver_encoder.solver_identity'].cpu(), identity_digest()):
                raise ValueError('Checkpoint solver identities do not match the current registry')
        else:
            from .solver_code_embeddings import validate_bundle
            validate_bundle(bundle)
            vectors, mask = bundle['vectors'], bundle['role_mask']
            if 'chunk_vectors' in bundle:
                components = {name: bundle[name] for name in ('chunk_vectors', 'chunk_indices')}
            rows = derangement(seed=params.get('permutation_seed', 4502)) if mode == 'shuffled_code' else torch.arange(M_GLOBAL)
        expected = derangement(seed=params.get('permutation_seed', 4502)) if mode == 'shuffled_code' else torch.arange(M_GLOBAL)
        if not torch.equal(rows, expected):
            raise ValueError('Code mapping differs from the declared experimental arm')
        # The common R43A modules are constructed first, with exactly the old RNG consumption.
        # Initialize the replacement branch independently, then explicitly retain common ID/LN.
        previous = model.solver_encoder
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(params.get('code_init_seed', 450002))
            model.solver_encoder = SolverCodeEncoder(vectors, mask, rows, components=components, **params)
        model.solver_encoder.solver_emb.load_state_dict(previous.solver_emb.state_dict())
        model.solver_encoder.norm.load_state_dict(previous.norm.state_dict())
    if state_dict is not None:
        model.load_state_dict(state_dict, strict=True)
    return model


def load_r45_checkpoint(path, device='cpu'):
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    config = checkpoint['args']
    if config.get('solver_names') != GLOBAL_SOLVERS:
        raise ValueError('Checkpoint must carry the exact global solver names/order')
    if config['model_params'].get('solver_representation', 'handcrafted') != 'handcrafted':
        from .solver_code_embeddings import validate_bundle
        provenance = config.get('embedding_provenance', {})
        components = {name: checkpoint['model']['solver_encoder.' + name]
                      for name in ('chunk_vectors', 'chunk_indices') if 'solver_encoder.' + name in checkpoint['model']}
        validate_bundle(dict(provenance, vectors=checkpoint['model']['solver_encoder.code_vectors'],
                             role_mask=checkpoint['model']['solver_encoder.role_mask'], **components))
    model = make_r45_model(config['model_params'], state_dict=checkpoint['model'])
    return model.to(device).eval(), checkpoint
