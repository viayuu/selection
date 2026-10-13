"""R71 joint cost-difference mixture and fixed native winner-probability rule."""

import math

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Categorical, MixtureSameFamily, MultivariateNormal

from code.unified_selector.registry import PROBLEMS
from .r64_model import NativeUtilitySelector
from .r69_model import TaskSpecificSelector, model_inputs


COMPONENTS, SOBOL_VECTORS, VECTOR_CHUNK = 8, 2048, 256
DIAGONAL_FLOOR, CE_FLOOR = 1e-3, 1e-12
INFERENCE_RULE = ('FP32 one encoder forward; eight full-covariance mixture components; '
    'fixed scrambled Sobol2048 normals/component; undo TRAIN normalization; full-pool '
    'minimum frequencies weighted by mixture probabilities; maximum win probability; '
    'exact ties by ascending global solver ID; CE-only floor1e-12')


def fit_normalizer(costs, pool_ids):
    values = np.asarray(costs)
    if values.dtype != np.float64 or values.ndim != 2 or len(values) == 0:
        raise ValueError('Normalizer requires original FP64 TRAIN cost rows')
    order = np.argsort(np.asarray(pool_ids), kind='stable')
    ids = np.asarray(pool_ids)[order]
    if len(ids) < 2 or len(np.unique(ids)) != len(ids) or values.shape[1] != len(ids):
        raise ValueError('Every original legal solver must occur exactly once')
    differences = values[:, order[1:]] - values[:, order[:1]]
    mean, std = differences.mean(0), differences.std(0, ddof=0)
    scale = np.maximum(std, 1e-6)
    if not np.isfinite(differences).all() or not np.isfinite(scale).all():
        raise ValueError('Original TRAIN differences and population scales must be finite')
    return dict(pool_ids=ids.tolist(), mean=mean.tolist(), population_std=std.tolist(),
                scale=scale.tolist(), fitted_rows=len(values), fit_split='train',
                reference_global_id=int(ids[0]), dtype='FP64', ddof=0, scale_floor=1e-6)


def standardized_targets(costs, pool_ids, normalizer):
    values = torch.as_tensor(costs)
    if values.dtype != torch.float64:
        raise ValueError('Target differences must start from original FP64 costs')
    ids = torch.as_tensor(pool_ids, device=values.device)
    wanted = torch.as_tensor(normalizer['pool_ids'], device=values.device)
    if ids.ndim != 1 or not torch.equal(ids.sort().values, wanted):
        raise ValueError('Target cost columns must retain exactly the original legal pool')
    order = ids.argsort(stable=True)
    differences = values[:, order[1:]] - values[:, order[:1]]
    mean = torch.as_tensor(normalizer['mean'], dtype=torch.float64, device=values.device)
    scale = torch.as_tensor(normalizer['scale'], dtype=torch.float64, device=values.device)
    return ((differences - mean) / scale).float()


def fixed_normal_table(problem_id, dimension):
    with torch.random.fork_rng(devices=[]):
        engine = torch.quasirandom.SobolEngine(dimension, scramble=True, seed=710003 + problem_id)
        uniform = engine.draw(SOBOL_VECTORS, dtype=torch.float64).clamp(2. ** -24, 1. - 2. ** -24)
        return (math.sqrt(2.) * torch.erfinv(2. * uniform - 1.)).float()


class JointMixtureHead(nn.Module):
    def __init__(self, problem_id, normalizer):
        super().__init__()
        self.dimension = len(normalizer['pool_ids']) - 1
        self.candidates = self.dimension + 1
        self.register_buffer('canonical_pool_ids', torch.tensor(normalizer['pool_ids'], dtype=torch.long))
        self.register_buffer('target_mean', torch.tensor(normalizer['mean'], dtype=torch.float64))
        self.register_buffer('target_scale', torch.tensor(normalizer['scale'], dtype=torch.float64))
        self.register_buffer('normal_vectors', fixed_normal_table(problem_id, self.dimension))
        indices = torch.tril_indices(self.dimension, self.dimension)
        self.register_buffer('tril_rows', indices[0])
        self.register_buffer('tril_columns', indices[1])
        self.register_buffer('diagonal_positions', (indices[0] == indices[1]).nonzero().flatten())
        self.triangle = indices.shape[1]
        width = 1 + self.dimension + self.triangle
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(710002 + problem_id)
            self.layers = nn.Sequential(nn.Linear(384 * self.candidates, 128), nn.GELU(),
                                        nn.Linear(128, COMPONENTS * width))
            with torch.no_grad():
                self.layers[2].weight.zero_()
                bias = self.layers[2].bias.view(COMPONENTS, width)
                bias.zero_()
                bias[:, 1:1 + self.dimension].normal_(0., .1)
                bias[:, 1 + self.dimension + self.diagonal_positions] = math.log(math.expm1(1. - DIAGONAL_FLOOR))

    def canonical_inputs(self, batch):
        clean = model_inputs(batch)
        rows = len(clean['node_mask'])
        ids = clean['pool_ids']
        if ids.ndim == 1:
            ids = ids[None].expand(rows, -1)
        if ids.ndim != 2 or len(ids) != rows:
            raise ValueError('Pool IDs must be one complete legal pool per query')
        mask = clean.get('solver_mask', torch.ones_like(ids, dtype=torch.bool)).bool()
        matches = (ids[:, :, None] == self.canonical_pool_ids[None, None]) & mask[:, :, None]
        if (mask.shape != ids.shape or not bool((mask.sum(1) == self.candidates).all())
                or not bool((matches.sum(1) == 1).all())):
            raise ValueError('Canonicalization must retain every legal solver exactly once; invalid padding is excluded')
        positions = matches.long().argmax(1)
        clean['pool_ids'] = self.canonical_pool_ids
        clean['solver_mask'] = torch.ones((rows, self.candidates), device=ids.device, dtype=torch.bool)
        return clean, positions, mask

    def forward(self, features):
        if features.shape[1:] != (self.candidates, 384):
            raise ValueError('Flatten only the complete canonical legal Kx384 features')
        with torch.autocast(device_type=features.device.type, enabled=False):
            raw = self.layers(features.float().flatten(1)).view(len(features), COMPONENTS, -1)
            logits, location = raw[:, :, 0], raw[:, :, 1:1 + self.dimension]
            triangle = raw[:, :, 1 + self.dimension:]
            factor = raw.new_zeros((len(features), COMPONENTS, self.dimension, self.dimension))
            factor[:, :, self.tril_rows, self.tril_columns] = triangle
            diagonal = F.softplus(triangle[:, :, self.diagonal_positions]) + DIAGONAL_FLOOR
            factor = factor - torch.diag_embed(factor.diagonal(dim1=-2, dim2=-1)) + torch.diag_embed(diagonal)
            return dict(mixture_logits=logits, location=location, scale_tril=factor)

    def nll(self, parameters, target):
        with torch.autocast(device_type=target.device.type, enabled=False):
            distribution = MixtureSameFamily(
                Categorical(logits=parameters['mixture_logits'].float(), validate_args=False),
                MultivariateNormal(parameters['location'].float(),
                    scale_tril=parameters['scale_tril'].float(), validate_args=False),
                validate_args=False)
            return -distribution.log_prob(target.float()).mean() / self.dimension

    @torch.no_grad()
    def winner_probabilities(self, parameters):
        location, factors = parameters['location'].float(), parameters['scale_tril'].float()
        weights = parameters['mixture_logits'].float().softmax(-1)
        if not all(bool(torch.isfinite(value).all()) for value in parameters.values()):
            raise RuntimeError('Nonfinite mixture parameters in frozen probability inference')
        result = location.new_zeros((len(location), self.candidates))
        for component in range(COMPONENTS):
            counts = torch.zeros_like(result, dtype=torch.long)
            for start in range(0, SOBOL_VECTORS, VECTOR_CHUNK):
                normals = self.normal_vectors[start:start + VECTOR_CHUNK]
                standardized = torch.matmul(normals[None], factors[:, component].transpose(-1, -2))
                standardized = standardized + location[:, component, None]
                differences = standardized.double() * self.target_scale + self.target_mean
                full = torch.cat((differences.new_zeros((*differences.shape[:2], 1)), differences), -1)
                winners = full.argmin(-1)
                counts.scatter_add_(1, winners, torch.ones_like(winners))
            result.add_(weights[:, component, None] * (counts.float() / SOBOL_VECTORS))
        return result

    def restore_probabilities(self, canonical, positions, mask):
        probabilities = canonical.new_zeros(mask.shape).scatter(1, positions, canonical)
        chosen = positions.gather(1, canonical.argmax(-1, keepdim=True)).squeeze(1)
        return dict(probabilities=probabilities, pred=chosen, solver_mask=mask)


class JointMixtureSelector(NativeUtilitySelector):
    def __init__(self, params, problem_id, normalizer):
        super().__init__(**params)
        del self.score_head
        self.density_head = JointMixtureHead(problem_id, normalizer)

    def density_parameters(self, batch, amp=False):
        clean, positions, mask = self.density_head.canonical_inputs(batch)
        with torch.autocast(device_type=clean['node_mask'].device.type, dtype=torch.float16, enabled=amp):
            features, legal = self.encode_pairs(clean)
        if not bool(legal.all()):
            raise ValueError('Only the complete canonical legal candidates may enter the density head')
        return self.density_head(features), positions, mask

    def forward(self, batch):
        parameters, positions, mask = self.density_parameters(batch, amp=False)
        probabilities = self.density_head.winner_probabilities(parameters)
        return dict(self.density_head.restore_probabilities(probabilities, positions, mask), density=parameters)


def make_bank(params, initial, normalizers, device='cpu'):
    trunk = {key: value for key, value in initial.items() if not key.startswith('score_head.')}
    branches = []
    with torch.random.fork_rng(devices=[]):
        for problem_id, task in enumerate(PROBLEMS):
            branch = JointMixtureSelector(params, problem_id, normalizers[task])
            incompatible = branch.load_state_dict(trunk, strict=False)
            expected = {key for key in branch.state_dict() if key.startswith('density_head.')}
            if set(incompatible.missing_keys) != expected or incompatible.unexpected_keys:
                raise ValueError('Exactly the inherited trunk must load; the scalar utility head is replaced')
            branches.append(branch)
    bank = TaskSpecificSelector(branches).requires_grad_(True)
    return bank.to(device)
