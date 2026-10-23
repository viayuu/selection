"""R94-family coverage extension; original full18 selector and loss paths.

Both arms use the same pretrained routing donor. Only the previously absent
TSP/ATSP donors are random in A versus pretrained in B. Separate projections
avoid assuming that different native encoders share a hidden-channel basis.
"""

from copy import deepcopy

import torch
from torch import nn

from code.unified_selector.registry import PROBLEMS
from .V4Model import InstanceEncoder
from .pairwise_objective import inputs_only
from .pairwise_selector import ScoreDifferenceSelector
from .r88_model import ROLE_SPEC, SELECTION_HEAD, RoleInputSelector, prepare_role_inputs


ARMS = ('A', 'B')
ARM_VARIANTS = dict(A='extra_random', B='full_pretrained')
MODEL_ROLE_ARM = 'B_execution_anchor'
INITIALIZATION = 'fresh_r84_plus_full_type_fusion'
DONOR_FIELDS = ('nco_nodes', 'nco_node_mask')
WIDTH_BY_PROBLEM = {problem: 512 if problem == 'ATSP' else 128 for problem in PROBLEMS}
PROJECTION_SEEDS = dict(TSP=1020003, ATSP=1020004, VRP=940003)
PROJECTION_NAMES = tuple('instance_encoder.nco_projections.' + group + '.weight'
                         for group in PROJECTION_SEEDS)
SPEC = dict(
    r102_schema_version=1, backbone='original_full18_r88b_r77_dualstream',
    model_role_arm=MODEL_ROLE_ARM, coverage=list(PROBLEMS),
    arm_variants=ARM_VARIANTS, routing_donor_in_both_arms='R94B_pretrained_ReLD',
    TSP_source='native_ELG_TSP_encoder128',
    ATSP_source='native_MatNet_FROM256_concat_TO256',
    ATSP_roles='fixed_identity_onehot_first_N_channels_max256',
    widths=WIDTH_BY_PROBLEM, fusion_location='before_first_original_instance_layer',
    fusion='original_raw_token+linear(nonaffine_normalized_frozen_native_nodes)',
    ATSP_normalization='normalize_FROM_and_TO_separately_then_concatenate',
    projection_bias=False, projection_seeds=PROJECTION_SEEDS,
    new_trainable_parameter_names=list(PROJECTION_NAMES), new_parameters=98304,
    formal_trainable_parameters=1941257, encoder_layer_num=4, joint_layer_num=2,
    solver_features=34, dropout=.1, selection_head=SELECTION_HEAD,
    objective='unchanged_original_r77_ce_comparison_risk_rdrop',
    native_feature_precision='FP32_TF32off_then_FP16_like_cache',
)


def problem_name(batch):
    index = int(batch['problem_id'])
    if not 0 <= index < len(PROBLEMS):
        raise ValueError('R102 requires a registered scalar problem_id')
    return PROBLEMS[index]


def inputs_with_donor(batch):
    result = inputs_only({key: value for key, value in batch.items()
                          if not key.startswith('nco_')})
    result.update({key: batch[key] for key in DONOR_FIELDS if key in batch})
    return result


def cached_nodes(batch):
    node, mask = batch['nco_nodes'], batch['nco_node_mask']
    valid = batch['node_mask']
    device = batch['matrix' if batch['kind'] == 'matrix' else 'node'].device
    width = WIDTH_BY_PROBLEM[problem_name(batch)]
    if (not torch.is_tensor(node) or not node.is_floating_point()
            or node.shape != (*valid.shape, width) or node.device != device
            or not torch.is_tensor(mask) or mask.dtype != torch.bool
            or mask.shape != valid.shape or mask.device != valid.device
            or valid.dtype != torch.bool or not torch.equal(mask, valid)):
        raise ValueError('R102 donor features/mask must match original physical nodes and width')
    if not torch.isfinite(node[mask]).all():
        raise ValueError('Nonfinite valid native donor feature')
    return node.detach().masked_fill(~mask[..., None], 0.)


def validate_params(params, arm):
    if arm not in ARMS:
        raise ValueError('R102 has only the matched A/B coverage comparison')
    fixed = dict(r102_schema_version=1, model_spec=SPEC, initialization=INITIALIZATION,
        role_arm=MODEL_ROLE_ARM, role_spec=ROLE_SPEC, selection_head=SELECTION_HEAD,
        solver_representation='handcrafted', pair_mode='score_difference',
        architecture='pairwise', encoder_layer_num=4, joint_layer_num=2,
        dropout=.1, local_geometry=True, geometry_mode='real',
        ignore_coord_dist=True, rezero=False, embedding_dim=128)
    for name, value in fixed.items():
        if params.get(name, value) != value:
            raise ValueError('R102 requires unchanged ' + name)
    metadata = params.get('frozen_donor')
    if metadata is not None and metadata.get('variant') != ARM_VARIANTS[arm]:
        raise ValueError('Saved R102 donor variant disagrees with arm')
    result = dict(params)
    result.update(deepcopy(fixed))
    return result


class FullTypeFusionInstanceEncoder(InstanceEncoder):
    def __init__(self, **params):
        super().__init__(**params)
        if not self.node_only:
            raise ValueError('R102 retains the original dual-stream node-only encoder')
        self.nco_norms = nn.ModuleDict({
            group: nn.LayerNorm(width, elementwise_affine=False)
            for group, width in (('TSP', 128), ('VRP', 128), ('ATSP_from', 256), ('ATSP_to', 256))})
        self.nco_projections = nn.ModuleDict()
        for group, width in (('TSP', 128), ('ATSP', 512), ('VRP', 128)):
            with torch.random.fork_rng(devices=[]):
                torch.random.default_generator.manual_seed(PROJECTION_SEEDS[group])
                self.nco_projections[group] = nn.Linear(width, params['embedding_dim'], bias=False)

    def forward(self, batch):
        special_count, name = 3, problem_name(batch)
        mask = batch['node_mask']
        if batch['kind'] == 'coord':
            if name == 'ATSP':
                raise ValueError('Do not replace directed ATSP with artificial coordinates')
            node = self._pad_node(batch['node'])
            stats = self._coord_stats(node, mask)
            cond, _ = self.condition_encoder(batch, stats, kind_id=0)
            token = self.coord_proj(node)
            token = token + self.geometry_residual(node, mask, batch['n'], batch.get('node_geom'))
            bias = self._coord_bias(node, mask, special_count)
        elif batch['kind'] == 'matrix' and name == 'ATSP':
            stats = self._matrix_stats(batch['matrix'], mask)
            cond, _ = self.condition_encoder(batch, stats, kind_id=1)
            attributes, bias = self._matrix_features_and_bias(batch['matrix'], mask, special_count)
            token = self.matrix_proj(attributes)
        else:
            raise ValueError('Use each problem\'s original coordinate/matrix input kind')
        group = name if name in ('TSP', 'ATSP') else 'VRP'
        frozen = cached_nodes(batch).to(dtype=self.nco_projections[group].weight.dtype)
        if group == 'ATSP':
            frozen = torch.cat([self.nco_norms['ATSP_from'](frozen[..., :256]),
                                self.nco_norms['ATSP_to'](frozen[..., 256:])], dim=-1)
        else:
            frozen = self.nco_norms[group](frozen)
        residual = self.nco_projections[group](frozen)
        token = token + residual.masked_fill(~mask[..., None], 0.)
        size = token.shape[0]
        token = torch.cat([self.cls_token.expand(size, 1, -1), cond[:, None, :],
                           self.stats_token_proj(stats)[:, None, :], token], dim=1)
        full_mask = torch.cat([torch.ones(size, special_count, dtype=torch.bool, device=mask.device), mask], dim=1)
        for layer in self.layers:
            token = layer(token, mask=full_mask, attn_bias=bias)
        return token[:, special_count:, :], mask


class FullTypeFusionSelector(RoleInputSelector):
    def __init__(self, arm, *, donor=None, restore=False, **params):
        params = validate_params(params, arm)
        super().__init__(MODEL_ROLE_ARM, **params)
        original = self.instance_encoder
        with torch.random.fork_rng(devices=[]):
            self.instance_encoder = FullTypeFusionInstanceEncoder(**dict(self.params, node_only=True))
        missing, unexpected = self.instance_encoder.load_state_dict(original.state_dict(), strict=False)
        if set(missing) != {name.removeprefix('instance_encoder.') for name in PROJECTION_NAMES} or unexpected:
            raise RuntimeError('Retain every original instance-encoder parameter and buffer')
        if donor is None:
            from .r102_donor import FullTypeDonor
            provenance = params.get('frozen_donor', {}).get('provenance')
            if restore and provenance is None:
                raise ValueError('Checkpoint restore needs its pinned donor provenance')
            with torch.random.fork_rng(devices=[]):
                donor = FullTypeDonor(arm, initialize_from_checkpoint=not restore,
                                     expected_provenance=provenance)
        if (not isinstance(donor, nn.Module) or getattr(donor, 'variant', None) != ARM_VARIANTS[arm]
                or set(getattr(donor, 'SUPPORTED_PROBLEMS', ())) != set(PROBLEMS)):
            raise ValueError('R102 requires its frozen full-type donor with the matched source variant')
        self.donor = donor.requires_grad_(False).eval()
        provenance = donor.provenance() if callable(donor.provenance) else donor.provenance
        if not isinstance(provenance, dict):
            raise ValueError('Donor provenance must be an explicit dictionary')
        self.arm = arm
        self.model_spec = deepcopy(SPEC)
        self.frozen_donor = dict(variant=ARM_VARIANTS[arm], provenance=deepcopy(provenance))
        if params.get('frozen_donor', self.frozen_donor) != self.frozen_donor:
            raise ValueError('Restored donor identity differs from its saved model configuration')
        self.params.update(model_spec=deepcopy(SPEC), frozen_donor=deepcopy(self.frozen_donor))

    def train(self, mode=True):
        super().train(mode)
        self.donor.requires_grad_(False).eval()
        return self

    def prepare_inputs(self, batch):
        prepared = inputs_with_donor(batch)
        cached = {name: prepared[name] for name in DONOR_FIELDS if name in prepared}
        if len(cached) == 1:
            raise ValueError('Both donor feature and mask are required')
        if not cached:
            if self.training or str(batch.get('split', '')).lower() in ('train', 'val', 'validation'):
                raise ValueError('R102 TRAIN/VAL must use fixed input-only cached donor features')
            with torch.no_grad(), torch.autocast(device_type=prepared['node_mask'].device.type, enabled=False):
                prepared['nco_nodes'] = self.donor(prepared).detach().to(dtype=torch.float16)
            prepared['nco_node_mask'] = prepared['node_mask']
            cached = {name: prepared[name] for name in DONOR_FIELDS}
        prepared = prepare_role_inputs(prepared, MODEL_ROLE_ARM)
        prepared.update(cached)
        cached_nodes(prepared)
        return prepared

    def forward(self, batch):
        return ScoreDifferenceSelector.forward(self, self.prepare_inputs(batch))


def make_model(params, arm, state_dict=None, *, donor=None):
    model = FullTypeFusionSelector(arm, donor=donor, restore=state_dict is not None, **params)
    if state_dict is not None:
        model.load_state_dict(state_dict, strict=True)
    return model
