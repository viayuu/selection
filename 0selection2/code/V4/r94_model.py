"""R94: frozen, aligned ReLD nodes fused before R88B's four instance layers.

Only ``instance_encoder.nco_projection.weight`` is newly trainable. The donor
is registered as ``donor`` (including its frozen checkpoint state), imported
only on construction, and never used for TSP/ATSP. The runner enforces cached
``nco_nodes`` and ``nco_node_mask`` for formal TRAIN/VAL. This model rejects
uncached training-mode or explicitly TRAIN/VAL-tagged inputs. Untagged raw
eval inputs remain compatible with existing LIB callers. Native FP32 output
is quantized to FP16 just like the cache; extraction runs once in
``prepare_inputs``, which the objective shares between dropout passes.

The runner owns copying *all* original fresh R77 parameters and TRAIN-only
geometry statistics. Construction preserves their original names, shapes and
RNG initialization; the extra projection has an isolated seed of 940003.
"""

from copy import deepcopy

import torch
from torch import nn

from code.unified_selector.registry import PROBLEMS
from .V4Model import InstanceEncoder
from .pairwise_objective import inputs_only
from .pairwise_selector import ScoreDifferenceSelector
from .r88_model import (
    ROLE_SLOT, ROLE_SPEC, SELECTION_HEAD, RoleInputSelector, prepare_role_inputs,
)


ARMS = ('A', 'B')
ARM_VARIANTS = dict(A='random', B='pretrained')
MODEL_ROLE_ARM = 'B_execution_anchor'
R94_SCHEMA_VERSION = 1
INITIALIZATION = 'scratch_shared_r77_plus_node_fusion'
NCO_WIDTH = 128
FUSION_SEED = 940003
PROJECTION_NAME = 'instance_encoder.nco_projection.weight'
DONOR_FIELDS = ('nco_nodes', 'nco_node_mask')
COVERED_PROBLEMS = tuple(p for p in PROBLEMS if p not in ('TSP', 'ATSP'))
REQUIRE_CACHE_SPLITS = ('train', 'val', 'validation')
SPEC = dict(
    r94_schema_version=R94_SCHEMA_VERSION,
    backbone='original_full18_r88b_r77_dualstream',
    model_role_arm=MODEL_ROLE_ARM, role_slot=ROLE_SLOT,
    donor='frozen_static_reld_moel_encoder', donor_width=NCO_WIDTH,
    covered_problems=list(COVERED_PROBLEMS), unsupported_problems=['TSP', 'ATSP'],
    fusion='coord_proj(raw8)+geometry_residual+linear(layernorm(stopgrad(H)))',
    fusion_location='before_first_original_instance_encoder_layer',
    layer_norm_affine=False, projection_bias=False, projection_seed=FUSION_SEED,
    raw_fallback_quantization='float16_like_frozen_feature_cache',
    new_trainable_parameter_names=[PROJECTION_NAME],
    encoder_layer_num=4, joint_layer_num=2, solver_features=34, dropout=.1,
    selection_head=SELECTION_HEAD, objective='original_r77_ce_comparison_risk_rdrop',
)


def inputs_with_donor(batch):
    """Original label isolation plus only the two safe ``nco_*`` cache fields."""
    prepared = inputs_only({key: value for key, value in batch.items()
                            if not key.startswith('nco_')})
    prepared.update({key: batch[key] for key in DONOR_FIELDS if key in batch})
    return prepared


def _problem(batch):
    pid = int(batch['problem_id'])
    if not 0 <= pid < len(PROBLEMS):
        raise ValueError('R94 requires a registered full18 problem_id')
    return PROBLEMS[pid]


def _cached_nodes(batch):
    nodes, mask = batch['nco_nodes'], batch['nco_node_mask']
    raw_mask = batch['node_mask']
    if (not torch.is_tensor(nodes) or not nodes.is_floating_point()
            or nodes.shape != (*raw_mask.shape, NCO_WIDTH)
            or nodes.device != batch['node'].device
            or not torch.is_tensor(mask) or mask.dtype != torch.bool
            or mask.shape != raw_mask.shape or mask.device != raw_mask.device
            or raw_mask.dtype != torch.bool or not torch.equal(mask, raw_mask)):
        raise ValueError('nco_nodes [B,Npad,128] and boolean mask must align exactly with raw nodes')
    if not torch.isfinite(nodes[mask]).all():
        raise ValueError('Valid cached donor nodes must be finite')
    return nodes.detach().masked_fill(~mask[..., None], 0.)


def _validate_params(params, arm):
    if arm not in ARMS:
        raise ValueError('R94 arm must be A (random) or B (pretrained)')
    fixed = dict(r94_schema_version=R94_SCHEMA_VERSION, model_spec=SPEC,
                 initialization=INITIALIZATION, role_arm=MODEL_ROLE_ARM,
                 role_spec=ROLE_SPEC, selection_head=SELECTION_HEAD,
                 solver_representation='handcrafted', pair_mode='score_difference',
                 architecture='pairwise', encoder_layer_num=4, joint_layer_num=2,
                 dropout=.1, local_geometry=True, geometry_mode='real',
                 ignore_coord_dist=True, rezero=False)
    for key, value in fixed.items():
        if params.get(key, value) != value:
            raise ValueError(f'R94 requires unchanged {key}={value!r}')
    metadata = params.get('frozen_donor')
    if metadata is not None and (not isinstance(metadata, dict)
                                 or metadata.get('variant') != ARM_VARIANTS[arm]):
        raise ValueError('R94 frozen_donor variant disagrees with its arm')
    result = dict(params)
    result.update(deepcopy(fixed))
    return result


class EarlyFusionInstanceEncoder(InstanceEncoder):
    """Original node-only encoder with one pre-interaction donor residual."""

    def __init__(self, **params):
        super().__init__(**params)
        if not self.node_only:
            raise ValueError('R94 requires the original dual-stream node-only encoder')
        self.nco_norm = nn.LayerNorm(NCO_WIDTH, elementwise_affine=False)
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(FUSION_SEED)
            self.nco_projection = nn.Linear(NCO_WIDTH, params['embedding_dim'], bias=False)

    def forward(self, batch):
        if _problem(batch) not in COVERED_PROBLEMS:
            return super().forward(batch)
        if batch['kind'] != 'coord':
            raise ValueError('Covered ReLD tasks require their original coordinate inputs')
        special_count = 3
        node = self._pad_node(batch['node'])
        node_mask = batch['node_mask']
        stats = self._coord_stats(node, node_mask)
        cond, _ = self.condition_encoder(batch, stats, kind_id=0)
        node_token = self.coord_proj(node)
        node_token = node_token + self.geometry_residual(node, node_mask, batch['n'], batch.get('node_geom'))
        frozen = _cached_nodes(batch).to(dtype=self.nco_projection.weight.dtype)
        residual = self.nco_projection(self.nco_norm(frozen))
        node_token = node_token + residual.masked_fill(~node_mask[..., None], 0.)
        bias = self._coord_bias(node, node_mask, special_count)

        batch_size = node_token.size(0)
        cls = self.cls_token.expand(batch_size, 1, -1)
        token = torch.cat([cls, cond[:, None, :], self.stats_token_proj(stats)[:, None, :], node_token], dim=1)
        special_mask = torch.ones(batch_size, special_count, dtype=torch.bool, device=node_token.device)
        mask = torch.cat([special_mask, node_mask], dim=1)
        for layer in self.layers:
            token = layer(token, mask=mask, attn_bias=bias)
        return token[:, special_count:, :], node_mask


class EarlyNodeFusionSelector(RoleInputSelector):
    """R88B score-difference/maximin selector; no new head or pooled teacher."""

    def __init__(self, arm, *, donor=None, **params):
        params = _validate_params(params, arm)
        super().__init__(MODEL_ROLE_ARM, **params)
        original = self.instance_encoder
        # Rebuild only this small subclass without advancing the backbone RNG.
        with torch.random.fork_rng(devices=[]):
            self.instance_encoder = EarlyFusionInstanceEncoder(**dict(self.params, node_only=True))
        missing, unexpected = self.instance_encoder.load_state_dict(original.state_dict(), strict=False)
        if missing != ['nco_projection.weight'] or unexpected:
            raise RuntimeError('R94 must retain every original instance-encoder state entry')
        if donor is None:
            from .r94_donor import StaticReLDDonor
            with torch.random.fork_rng(devices=[]):
                donor = StaticReLDDonor(variant=ARM_VARIANTS[arm],
                                        checkpoint_path=params.get('donor_checkpoint_path'))
        if not isinstance(donor, nn.Module):
            raise TypeError('The frozen R94 donor must be an nn.Module')
        if getattr(donor, 'variant', None) != ARM_VARIANTS[arm]:
            raise ValueError('Injected donor variant must match the R94 arm')
        coverage = getattr(donor, 'SUPPORTED_PROBLEMS', getattr(donor, 'covered_problems', ()))
        if set(coverage) != set(COVERED_PROBLEMS):
            raise ValueError('The donor must cover exactly CVRP plus the 15 MVRP tasks')
        self.donor = donor.requires_grad_(False).eval()
        provenance = getattr(donor, 'provenance', {})
        provenance = provenance() if callable(provenance) else provenance
        if not isinstance(provenance, dict):
            raise ValueError('Donor provenance must be a dictionary')
        self.arm = arm
        self.model_spec = deepcopy(SPEC)
        self.frozen_donor = dict(variant=ARM_VARIANTS[arm], provenance=deepcopy(provenance))
        if ('frozen_donor' in params and params['frozen_donor'] != self.frozen_donor):
            raise ValueError('Frozen donor provenance differs from the saved model configuration')
        self.params.update(model_spec=deepcopy(self.model_spec), frozen_donor=deepcopy(self.frozen_donor))

    def train(self, mode=True):
        super().train(mode)
        self.donor.requires_grad_(False).eval()
        return self

    def prepare_inputs(self, batch):
        """Preserve aligned caches across role preprocessing; extract at most once."""
        prepared = inputs_with_donor(batch)
        cached = {key: prepared[key] for key in DONOR_FIELDS if key in prepared}
        prepared = prepare_role_inputs(prepared, MODEL_ROLE_ARM)
        prepared.update(cached)
        if _problem(prepared) not in COVERED_PROBLEMS:
            for key in DONOR_FIELDS:
                prepared.pop(key, None)
            return prepared
        if prepared['kind'] != 'coord':
            raise ValueError('R94 must not reinterpret matrix inputs as artificial CVRP')
        if len(cached) == 1:
            raise ValueError('Both nco_nodes and nco_node_mask are required for a cache')
        if not cached:
            split = str(prepared.get('split', '')).lower()
            if self.training or split in REQUIRE_CACHE_SPLITS:
                raise ValueError('TRAIN/VAL require cached donor nodes; raw fallback is eval-only')
            self.donor.eval()
            with torch.no_grad(), torch.autocast(device_type=prepared['node'].device.type, enabled=False):
                prepared['nco_nodes'] = self.donor(prepared).detach().to(dtype=torch.float16)
            prepared['nco_node_mask'] = prepared['node_mask']
        _cached_nodes(prepared)
        return prepared

    def forward(self, batch):
        # Bypass RoleInputSelector.forward's second inputs_only call.
        return ScoreDifferenceSelector.forward(self, self.prepare_inputs(batch))


def make_model(params, arm, state_dict=None, *, donor=None):
    """Build a fresh selector or strictly restore all selector AND donor state."""
    model = EarlyNodeFusionSelector(arm, donor=donor, **params)
    if state_dict is not None:
        model.load_state_dict(state_dict, strict=True)
    return model
