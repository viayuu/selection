"""Input-only typed native nodes before the original whole18 selector layers."""

from copy import deepcopy

import torch
from torch import nn

from code.unified_selector.registry import PROBLEMS
from .V4Model import InstanceEncoder
from .pairwise_selector import ScoreDifferenceSelector
from .r88_model import ROLE_SPEC, SELECTION_HEAD, RoleInputSelector, prepare_role_inputs


ARMS = ('A', 'B')
ARM_VARIANTS = dict(A='added_random', B='added_pretrained')
MODEL_ROLE_ARM = 'B_execution_anchor'
INITIALIZATION = 'fresh_r84_plus_frozen_native_computation'
DONOR_FIELDS = ('nco_nodes', 'nco_node_mask', 'native_nodes', 'native_node_mask')
INPUT_KEYS = frozenset(('node', 'matrix', 'node_mask', 'n', 'node_geom', 'kind',
    'problem_id', 'problem_name', 'coord_dist', 'cbits', 'problem_desc', 'd_in',
    'pool_ids', 'solver_mask', 'mask', *DONOR_FIELDS))
WIDTH_BY_PROBLEM = {name: 256 if name == 'ATSP' else 128 for name in PROBLEMS}
PROJECTION_SEEDS = dict(TSP=1170003, CVRP=1170004, VRP=1170005, ATSP=1170006)
PROJECTION_SHAPES = {'instance_encoder.nco_projection.weight': (128, 128), **{
    'instance_encoder.native_projections.' + name + '.weight': (128, 256 if name == 'ATSP' else 128)
    for name in PROJECTION_SEEDS}}
SPEC = dict(format='r117_typed_native_computation_v1', arms=ARM_VARIANTS,
    role_arm=MODEL_ROLE_ARM, full18=True, original_four_instance_layers=True,
    original_two_joint_layers=True, original_hand34_id_score_mlp_maximin=True,
    common='identical frozen pretrained ReLD static nodes for CVRP+15MVRP',
    added=dict(TSP='LEHD heavy decoder initial-boundary nodes',
        CVRP='LEHD heavy decoder empty-history full-capacity nodes',
        VRP='ReLD independent reference-current decoder hidden nodes',
        ATSP='MatNet FROM-current TO-key decoder glimpses'),
    fusion='separate affine-free node norms and bias-free projections before first instance layer',
    widths=WIDTH_BY_PROBLEM, frozen=True, no_actions=True, no_new_targets=True,
    no_diffusion_coverage=True, native_precision='FP32 TF32off, FP16 cache and online quantization',
    projection_seeds=PROJECTION_SEEDS, common_projection_seed=940003)


def inputs_with_donor(batch):
    return {key: value for key, value in batch.items() if key in INPUT_KEYS}


def problem_name(batch):
    return PROBLEMS[int(batch['problem_id'])]


def component(problem):
    return problem if problem in ('TSP', 'CVRP', 'ATSP') else 'VRP'


def cached_nodes(batch, field, width):
    value, mask = batch[field], batch[field[:-5] + 'node_mask']
    valid = batch['node_mask']
    if (value.shape != (*valid.shape, width) or not value.is_floating_point()
            or value.device != valid.device or mask.dtype != torch.bool
            or mask.shape != valid.shape or not torch.equal(mask, valid)):
        raise ValueError('Native feature/mask must align with original physical nodes: ' + field)
    # Feature extraction checks finite values once, outside this hot path.
    return value.detach().masked_fill(~valid[..., None], 0.)


class NativeFusionInstanceEncoder(InstanceEncoder):
    def __init__(self, **params):
        super().__init__(**params)
        if not self.node_only:
            raise ValueError('R117 retains the node-only half of the original dual stream')
        self.nco_norm = nn.LayerNorm(128, elementwise_affine=False)
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(940003)
            self.nco_projection = nn.Linear(128, params['embedding_dim'], bias=False)
        self.native_norms = nn.ModuleDict({name: nn.LayerNorm(
            256 if name == 'ATSP' else 128, elementwise_affine=False) for name in PROJECTION_SEEDS})
        self.native_projections = nn.ModuleDict()
        for name, seed in PROJECTION_SEEDS.items():
            with torch.random.fork_rng(devices=[]):
                torch.random.default_generator.manual_seed(seed)
                self.native_projections[name] = nn.Linear(
                    256 if name == 'ATSP' else 128, params['embedding_dim'], bias=False)

    def forward(self, batch):
        name, special = problem_name(batch), 3
        mask = batch['node_mask']
        if batch['kind'] == 'coord' and name != 'ATSP':
            node = self._pad_node(batch['node'])
            stats = self._coord_stats(node, mask)
            cond, _ = self.condition_encoder(batch, stats, kind_id=0)
            token = self.coord_proj(node)
            token = token + self.geometry_residual(node, mask, batch['n'], batch.get('node_geom'))
            bias = self._coord_bias(node, mask, special)
        elif batch['kind'] == 'matrix' and name == 'ATSP':
            stats = self._matrix_stats(batch['matrix'], mask)
            cond, _ = self.condition_encoder(batch, stats, kind_id=1)
            attributes, bias = self._matrix_features_and_bias(batch['matrix'], mask, special)
            token = self.matrix_proj(attributes)
        else:
            raise ValueError('Keep each problem original input kind')
        group = component(name)
        if name not in ('TSP', 'ATSP'):
            base = cached_nodes(batch, 'nco_nodes', 128).to(self.nco_projection.weight.dtype)
            token = token + self.nco_projection(self.nco_norm(base))
        native = cached_nodes(batch, 'native_nodes', WIDTH_BY_PROBLEM[name])
        native = native.to(self.native_projections[group].weight.dtype)
        token = token + self.native_projections[group](self.native_norms[group](native))
        token = token.masked_fill(~mask[..., None], 0.)
        count = len(token)
        token = torch.cat([self.cls_token.expand(count, 1, -1), cond[:, None],
            self.stats_token_proj(stats)[:, None], token], 1)
        full_mask = torch.cat([torch.ones(count, special, dtype=torch.bool, device=mask.device), mask], 1)
        for layer in self.layers:
            token = layer(token, mask=full_mask, attn_bias=bias)
        return token[:, special:], mask


class NativeComputationSelector(RoleInputSelector):
    def __init__(self, arm, *, donor=None, restore=False, **params):
        if arm not in ARMS:
            raise ValueError('R117 has only matched arms A/B')
        params = dict(params)
        fixed = dict(role_arm=MODEL_ROLE_ARM, role_spec=ROLE_SPEC, selection_head=SELECTION_HEAD,
            solver_representation='handcrafted', pair_mode='score_difference', architecture='pairwise',
            encoder_layer_num=4, joint_layer_num=2, local_geometry=True, geometry_mode='real',
            ignore_coord_dist=True, rezero=False, dropout=.1)
        for key, value in fixed.items():
            if params.get(key, value) != value:
                raise ValueError('R117 requires original ' + key)
        params.update(fixed)
        super().__init__(MODEL_ROLE_ARM, **params)
        original = self.instance_encoder
        with torch.random.fork_rng(devices=[]):
            self.instance_encoder = NativeFusionInstanceEncoder(**dict(self.params, node_only=True))
        missing, unexpected = self.instance_encoder.load_state_dict(original.state_dict(), strict=False)
        expected = {key.removeprefix('instance_encoder.') for key in PROJECTION_SHAPES}
        if set(missing) != expected or unexpected:
            raise RuntimeError('Original instance encoder state must be retained exactly')
        if donor is None:
            from .r117_native import NativeComputation
            expected_provenance = params.get('frozen_donor', {}).get('provenance')
            donor = NativeComputation(arm, restore=restore, expected_provenance=expected_provenance)
        if not isinstance(donor, nn.Module):
            raise TypeError('Native computation must be a frozen module')
        self.donor = donor.requires_grad_(False).eval()
        provenance = donor.provenance() if callable(donor.provenance) else deepcopy(donor.provenance)
        self.frozen_donor = dict(variant=ARM_VARIANTS[arm], provenance=provenance)
        if params.get('frozen_donor', self.frozen_donor) != self.frozen_donor:
            raise ValueError('Restored native weight identity changed')
        self.arm, self.model_spec = arm, deepcopy(SPEC)
        self.params.update(initialization=INITIALIZATION, model_spec=deepcopy(SPEC),
            r117_schema_version=1, frozen_donor=deepcopy(self.frozen_donor))

    def train(self, mode=True):
        super().train(mode)
        self.donor.requires_grad_(False).eval()
        return self

    def prepare_inputs(self, batch):
        clean = inputs_with_donor(batch)
        fields = {key: clean[key] for key in DONOR_FIELDS if key in clean}
        if fields and len(fields) != len(DONOR_FIELDS):
            raise ValueError('Provide all four aligned native cache fields')
        if not fields:
            if self.training:
                raise ValueError('Frozen TRAIN nodes must be prepared once, not inside each dropout forward')
            with torch.no_grad():
                features = self.donor(clean)
            fields = dict(nco_nodes=features['base_nodes'].to(torch.float16),
                native_nodes=features['native_nodes'].to(torch.float16),
                nco_node_mask=clean['node_mask'], native_node_mask=clean['node_mask'])
        clean = prepare_role_inputs(clean, MODEL_ROLE_ARM)
        clean.update(fields)
        return clean

    def forward(self, batch):
        return ScoreDifferenceSelector.forward(self, self.prepare_inputs(batch))


def make_model(params, arm, state_dict=None, *, donor=None):
    model = NativeComputationSelector(arm, donor=donor, restore=state_dict is not None, **params)
    if state_dict is not None:
        model.load_state_dict(state_dict, strict=True)
    return model
