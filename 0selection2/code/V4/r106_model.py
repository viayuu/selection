"""R106 full18 raw/native fusion, frozen versus differentiable native encoders."""

from contextlib import contextmanager
from copy import deepcopy

import torch

from code.unified_selector.registry import PROBLEMS
from .pairwise_selector import ScoreDifferenceSelector
from .r77_data import parameter_hash
from .r88_model import ROLE_SPEC, SELECTION_HEAD, RoleInputSelector, prepare_role_inputs
from .r102_model import (FullTypeFusionInstanceEncoder, PROJECTION_NAMES, PROJECTION_SEEDS,
                         WIDTH_BY_PROBLEM)
from .r106_donor import ARMS, NativeDonor, SPEC as DONOR_SPEC


MODEL_ROLE_ARM = 'B_execution_anchor'
INITIALIZATION = 'fresh_r84_seed2_plus_r102_projections_and_pinned_r102b_native'
R84_SEED2_SHA256 = 'dad87d8598f7560e6f170ef048ba1bf515edeb03cef7ec42e16839a36af4ef93'
INPUT_ALLOWLIST = (
    'problem_id', 'problem_name', 'kind', 'd_in', 'node', 'matrix', 'n', 'node_mask',
    'pool_ids', 'solver_mask', 'coord_dist', 'cbits', 'problem_desc', 'node_geom',
)
NATIVE_FIELD = '_r106_online_nodes'
SPEC = dict(
    r106_schema_version=1, backbone='original_full18_r88b_r77_dualstream',
    initialization=INITIALIZATION, arms=list(ARMS), coverage=list(PROBLEMS),
    encoder_layer_num=4, joint_layer_num=2, embedding_dim=128, solver_features=34,
    dropout=.1, selection_head=SELECTION_HEAD, role_arm=MODEL_ROLE_ARM,
    fusion='raw_token+linear(nonaffine_normalized_online_native_nodes)',
    fusion_location='before_first_original_instance_layer',
    ATSP_normalization='separate_FROM256_and_TO256_then_concatenate',
    projection_bias=False, projection_seeds=deepcopy(PROJECTION_SEEDS),
    native=deepcopy(DONOR_SPEC), native_output='FP32 until selector fusion precision path',
    native_lr_multiplier=dict(A_frozen=None, B_finetune=.1),
    native_lr_owner='runner, from first successful update, no unfreezing phase',
    objective='unchanged_original_r77_ce_comparison_risk_rdrop',
    native_pair_context='one current-microbatch graph; at most two preparations; finally clear',
)


def input_only(batch):
    """The sole public boundary: no old cache or arbitrary supervision field."""
    if any(key.startswith('nco_') or key == NATIVE_FIELD for key in batch):
        raise ValueError('R106 rejects external native features/caches')
    return {key: batch[key] for key in INPUT_ALLOWLIST if key in batch}


def _sanitize(clean):
    result = dict(clean)
    mask = result['node_mask']
    if not torch.is_tensor(mask) or mask.dtype != torch.bool or mask.ndim != 2:
        raise ValueError('Require boolean node_mask[B,N]')
    if result['kind'] == 'coord':
        result['node'] = result['node'].masked_fill(~mask[..., None], 0.)
        if 'node_geom' in result:
            result['node_geom'] = result['node_geom'].masked_fill(~mask[..., None], 0.)
    elif result['kind'] == 'matrix':
        pairs = mask[:, :, None] & mask[:, None, :]
        result['matrix'] = result['matrix'].masked_fill(~pairs, 0.)
    else:
        raise ValueError('Require original coordinate or matrix inputs')
    return result


def _tensor_signature(value):
    try:
        version = value._version
    except RuntimeError as error:
        raise ValueError('Pair context requires versioned tensors, not inference-mode tensors') from error
    return (id(value), value.data_ptr(), tuple(value.shape), tuple(value.stride()),
            value.dtype, value.device, version)


def _input_signature(clean):
    return tuple((key, _tensor_signature(value) if torch.is_tensor(value) else value)
                 for key, value in sorted(clean.items()))


def validate_params(params, arm):
    if arm not in ARMS:
        raise ValueError('R106 requires A_frozen or B_finetune')
    if any(key.startswith(('r102_', 'r94_')) or key == 'frozen_donor' for key in params):
        raise ValueError('R106 has its own schema; frozen R102 metadata is not a B configuration')
    fixed = dict(
        r106_schema_version=1, r106_arm=arm, model_spec=SPEC, initialization=INITIALIZATION,
        role_arm=MODEL_ROLE_ARM, role_spec=ROLE_SPEC, selection_head=SELECTION_HEAD,
        solver_representation='handcrafted', pair_mode='score_difference', architecture='pairwise',
        encoder_layer_num=4, joint_layer_num=2, dropout=.1, local_geometry=True,
        geometry_mode='real', ignore_coord_dist=True, rezero=False, embedding_dim=128,
    )
    for name, value in fixed.items():
        if params.get(name, value) != value:
            raise ValueError('R106 requires unchanged ' + name)
    result = deepcopy(params)
    result.update(deepcopy(fixed))
    metadata = result.get('native_donor')
    if metadata is not None and metadata.get('trainable') != (arm == 'B_finetune'):
        raise ValueError('R106 saved native trainability disagrees with arm')
    return result


class NativeFusionInstanceEncoder(FullTypeFusionInstanceEncoder):
    """Reuse only R102's seeded layer construction, NOT its detached forward."""

    def forward(self, batch):
        special_count = 3
        name = PROBLEMS[int(batch['problem_id'])]
        mask = batch['node_mask']
        if batch['kind'] == 'coord' and name != 'ATSP':
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
            raise ValueError('Do not replace original problem input kinds')
        native = batch[NATIVE_FIELD]
        if (native.shape != (*mask.shape, WIDTH_BY_PROBLEM[name])
                or native.dtype != torch.float32 or native.device != mask.device):
            raise ValueError('R106 requires its internally derived aligned FP32 native nodes')
        group = name if name in ('TSP', 'ATSP') else 'VRP'
        native = native.to(dtype=self.nco_projections[group].weight.dtype)
        if group == 'ATSP':
            native = torch.cat((self.nco_norms['ATSP_from'](native[..., :256]),
                                self.nco_norms['ATSP_to'](native[..., 256:])), -1)
        else:
            native = self.nco_norms[group](native)
        residual = self.nco_projections[group](native)
        token = token + residual.masked_fill(~mask[..., None], 0.)
        rows = token.shape[0]
        token = torch.cat((self.cls_token.expand(rows, 1, -1), cond[:, None],
                           self.stats_token_proj(stats)[:, None], token), 1)
        full_mask = torch.cat((torch.ones(rows, special_count, dtype=torch.bool, device=mask.device), mask), 1)
        for layer in self.layers:
            token = layer(token, mask=full_mask, attn_bias=bias)
        return token[:, special_count:], mask


class NativeAdaptationSelector(RoleInputSelector):
    def __init__(self, arm, *, restore=False, **params):
        params = validate_params(params, arm)
        super().__init__(MODEL_ROLE_ARM, **params)
        original = self.instance_encoder
        with torch.random.fork_rng(devices=[]):
            self.instance_encoder = NativeFusionInstanceEncoder(**dict(self.params, node_only=True))
        missing, unexpected = self.instance_encoder.load_state_dict(original.state_dict(), strict=False)
        if (set(missing) != {name.removeprefix('instance_encoder.') for name in PROJECTION_NAMES}
                or unexpected):
            raise RuntimeError('R106 must retain every original selector parameter and buffer')
        metadata = params.get('native_donor')
        if restore and not isinstance(metadata, dict):
            raise ValueError('State-only restore requires saved R106 native_donor metadata')
        self.donor = NativeDonor(
            arm, initialize_from_checkpoint=not restore,
            expected_provenance=None if metadata is None else metadata['provenance'])
        self.arm = arm
        self.model_spec = deepcopy(SPEC)
        self.native_donor = dict(trainable=arm == 'B_finetune', mode='eval',
                                 provenance=self.donor.provenance())
        if metadata is not None and metadata != self.native_donor:
            raise ValueError('R106 donor configuration differs from saved identity')
        self.params.update(native_donor=deepcopy(self.native_donor), model_spec=deepcopy(SPEC))
        self._native_pair = None

    def train(self, mode=True):
        super().train(mode)
        self.donor.eval()
        return self

    def _state_versions(self):
        # Include selector tensors: even an A-only optimizer step ends this scope.
        return tuple((name, id(value), value._version)
                     for name, value in (*self.named_parameters(), *self.named_buffers()))

    @contextmanager
    def native_pair_context(self, clean_inputs):
        """Share a single deterministic native graph for ONE two-view R-Drop loss.

        Pass original inputs to the original objective's two forwards, not the
        private result of prepare_inputs. Do one combined backward, normally
        after leaving this scope. Never retain this scope across an update.
        """
        if self._native_pair is not None:
            raise RuntimeError('Nested R106 native pair contexts are forbidden')
        clean = input_only(clean_inputs)
        signature = _input_signature(clean)
        versions = self._state_versions()
        try:
            features = self.donor(_sanitize(clean))
            self._native_pair = dict(
                signature=signature, versions=versions, features=features, uses=0,
                grad_enabled=torch.is_grad_enabled(),
                mask=clean['node_mask'].detach().clone(), n=clean['n'].detach().clone())
            yield self
        finally:
            self._native_pair = None

    def prepare_inputs(self, batch):
        clean = input_only(batch)
        pair = self._native_pair
        if pair is None:
            features = self.donor(_sanitize(clean))
        else:
            if (pair['signature'] != _input_signature(clean)
                    or pair['versions'] != self._state_versions()
                    or pair['grad_enabled'] != torch.is_grad_enabled()
                    or not torch.equal(pair['mask'], clean['node_mask'])
                    or not torch.equal(pair['n'], clean['n'])):
                raise RuntimeError('R106 native pair context input/state changed; cannot reuse a graph')
            if pair['uses'] >= 2:
                raise RuntimeError('R106 native pair context is limited to two preparations')
            features = pair['features']
            pair['uses'] += 1
        prepared = prepare_role_inputs(_sanitize(clean), MODEL_ROLE_ARM)
        prepared[NATIVE_FIELD] = features
        return prepared

    def forward(self, batch):
        return ScoreDifferenceSelector.forward(self, self.prepare_inputs(batch))


def parameter_counts(model):
    named = dict(model.named_parameters())
    native = {name: value for name, value in named.items() if getattr(value, '_r106_native', False)}
    if set(native) != {name for name in named if name.startswith('donor.')}:
        raise ValueError('Native Parameter tags disagree with registered donor membership')
    selector = {name: value for name, value in named.items() if name not in native}
    projections = {name: selector[name] for name in PROJECTION_NAMES}
    return dict(
        total=sum(p.numel() for p in named.values()),
        trainable=sum(p.numel() for p in named.values() if p.requires_grad),
        selector=sum(p.numel() for p in selector.values()),
        selector_trainable=sum(p.numel() for p in selector.values() if p.requires_grad),
        projections=sum(p.numel() for p in projections.values()),
        native=sum(p.numel() for p in native.values()),
        native_trainable=sum(p.numel() for p in native.values() if p.requires_grad),
        native_by_component={component: sum(p.numel() for name, p in native.items()
                                            if name.startswith('donor.' + component + '.'))
                             for component in ('tsp', 'atsp', 'routing')},
    )


def make_model(params, arm, state_dict=None):
    with torch.random.fork_rng(devices=[]):
        model = NativeAdaptationSelector(arm, restore=state_dict is not None, **params)
    if state_dict is not None:
        expected = model.state_dict()
        if set(expected) != set(state_dict):
            raise ValueError('R106 restore requires the COMPLETE selector and native state')
        for name, target in expected.items():
            value = state_dict[name]
            if not torch.is_tensor(value) or value.shape != target.shape or value.dtype != target.dtype:
                raise ValueError('R106 strict state shape/dtype mismatch: ' + name)
        model.load_state_dict(state_dict, strict=True)
    return model


def initialize_shared(initial, geometry, arm, device='cpu'):
    reference = initial['trainable']
    if (arm not in ARMS or initial.get('seed') != 2 or initial.get('initialization') != 'scratch'
            or initial.get('trainable_sha256') != R84_SEED2_SHA256
            or parameter_hash(reference) != R84_SEED2_SHA256
            or sum(t.numel() for t in reference.values()) != 1842953):
        raise ValueError('R106 requires the original fresh R84 seed2 state, never a trained selector')
    with torch.random.fork_rng(devices=[]), torch.device('cpu'):
        torch.random.default_generator.manual_seed(2)
        model = make_model(initial['model_params'], arm)
    selector = {name: value for name, value in model.named_parameters() if not name.startswith('donor.')}
    retained = {name: value for name, value in selector.items() if name in reference}
    added = {name: value for name, value in selector.items() if name not in reference}
    if (set(retained) != set(reference) or set(added) != set(PROJECTION_NAMES)
            or any(value.shape != reference[name].shape for name, value in retained.items())):
        raise ValueError('R106 must retain the entire R84 selector and only add three projections')
    with torch.no_grad():
        for name, parameter in retained.items():
            parameter.copy_(reference[name])
        branch = model.instance_encoder.geometry_residual
        for name in ('mean', 'std'):
            value = torch.as_tensor(geometry[name], dtype=getattr(branch, name).dtype)
            if (value.shape != getattr(branch, name).shape or not torch.isfinite(value).all()
                    or (name == 'std' and (value <= 0).any())):
                raise ValueError('Invalid frozen R84 geometry ' + name)
            getattr(branch, name).copy_(value)
    counts = parameter_counts(model)
    expected = dict(total=11965097, selector=1941257, projections=98304, native=10023840)
    if any(counts[name] != value for name, value in expected.items()):
        raise ValueError('R106 actual parameter counts differ from the plan: ' + str(counts))
    common_hash, projection_hash, selector_hash, state_hash = map(
        parameter_hash, (retained, added, selector, model.state_dict()))
    if common_hash != R84_SEED2_SHA256:
        raise ValueError('R106 original selector initialization changed')
    receipt = dict(
        seed=2, initialization=INITIALIZATION, historical_selector_checkpoint_loaded=False,
        common_parameter_count=sum(p.numel() for p in retained.values()),
        common_parameter_sha256=common_hash, new_parameter_sha256=projection_hash,
        selector_parameter_sha256=selector_hash, initial_state_sha256=state_hash,
        native_state_sha256=parameter_hash(model.donor.state_dict()),
        parameter_counts=counts, trainable_parameter_count=counts['trainable'],
        total_parameter_count=counts['total'], selector_trainable_parameter_count=counts['selector_trainable'],
        native_trainable_parameter_count=counts['native_trainable'],
        new_parameter_names=sorted(added),
        new_parameter_shapes={name: list(value.shape) for name, value in added.items()},
        native_donor=deepcopy(model.native_donor),
        geometry_fitted_on='frozen R84 new81000 TRAIN only',
        requires_grad_excluded_from_state_hash=True,
    )
    model.params['r106_initial_state_sha256'] = state_hash
    return model.to(device), deepcopy(model.params), receipt


def load_checkpoint(path, device='cpu'):
    """Portable model restore; runner remains responsible for protocol validation.

    Envelope: {args: {arm: ..., model_params: ...}, model: full_state_dict}.
    No external pretrained weight file is opened by this path.
    """
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    config = checkpoint['args']
    model = make_model(config['model_params'], config['arm'], state_dict=checkpoint['model'])
    return model.to(device), checkpoint
