"""Bounded R88B capacity axes; CPU fresh initialization, no experiment runner.

``make_model(raw_r84_params, arm)`` overrides only CAPACITY_FIELDS (plus
descriptive metadata). All forward operations remain the original R88B path.
``initialize_shared(initial, geometry, arm)`` accepts the in-memory, hashed
R84 scratch-seed2 record, not a trained checkpoint. R84's own all-parameters-
must-match initializer cannot initialize these larger architectures.
"""

import copy

import torch

from .r77_data import parameter_hash
from .r88_model import ROLE_SPEC, RoleInputSelector
from .V4Model import MultiHeadAttention


ARMS = ('A_width256', 'B_depth8')
ROLE_ARM = 'B_execution_anchor'
FRESH_SEED = 2
CAPACITY_INIT_SEED = 104002
CAPACITY_FIELDS = (
    'embedding_dim', 'encoder_layer_num', 'joint_layer_num', 'head_num',
    'qkv_dim', 'ff_hidden_dim', 'head_hidden_dim', 'solver_feature_hidden',
)
ORIGINAL = dict(embedding_dim=128, encoder_layer_num=4, joint_layer_num=2,
                head_num=4, qkv_dim=32, ff_hidden_dim=512,
                head_hidden_dim=256, solver_feature_hidden=128)
CAPACITIES = {
    ARMS[0]: dict(embedding_dim=256, encoder_layer_num=4, joint_layer_num=2,
                  head_num=4, qkv_dim=64, ff_hidden_dim=1024,
                  head_hidden_dim=512, solver_feature_hidden=256),
    ARMS[1]: dict(ORIGINAL, encoder_layer_num=8, joint_layer_num=4),
}
SPEC = dict(
    experiment='R104', scope='model_only', parent='full18_R88B_R77_score_path',
    role_arm=ROLE_ARM, role_spec=copy.deepcopy(ROLE_SPEC),
    selection_head='original_score_mlp_maximin', fresh_seed=FRESH_SEED,
    capacity_init_seed=CAPACITY_INIT_SEED, capacity_fields=list(CAPACITY_FIELDS),
    original=copy.deepcopy(ORIGINAL), arms=copy.deepcopy(CAPACITIES),
    head_policy='same_four_heads_qkv_equals_width_divided_by_four',
    scaling='FF=4d_score_hidden=2d_method_feature_hidden=d_geometry_hidden=64',
    comparison='two_capacity_axes_not_parameter_matched_AB',
)


def _resolve_params(params, arm):
    if arm not in ARMS:
        raise ValueError('Unknown R104 capacity arm')
    params = copy.deepcopy(params)
    if params.get('capacity_arm', arm) != arm:
        raise ValueError('R104 capacity arm changed during reconstruction')
    if params.get('capacity_spec', CAPACITIES[arm]) != CAPACITIES[arm]:
        raise ValueError('R104 bounded capacity specification changed')
    if params.get('seed', FRESH_SEED) != FRESH_SEED:
        raise ValueError('R104 fresh initialization requires seed2')
    if (params.get('local_geometry') is not True
            or params.get('geometry_mode', 'real') != 'real'
            or params.get('solver_feature_weight') != 1.0):
        raise ValueError('R104 retains full18 real geometry and methodID+34 at weight1')
    if params.get('role_arm', ROLE_ARM) != ROLE_ARM:
        raise ValueError('R104 retains R88B execution-anchor preprocessing')
    params.update(CAPACITIES[arm], capacity_arm=arm,
                  capacity_spec=copy.deepcopy(CAPACITIES[arm]))
    return params


def model_dimensions(model):
    """Measure instantiated modules, not the incoming nominal R84 settings."""
    encoder, methods = model.instance_encoder, model.solver_encoder
    attention = [module for module in model.modules()
                 if isinstance(module, MultiHeadAttention)]
    signatures = {(a.Wq.in_features, a.head_num, a.qkv_dim, a.Wq.out_features)
                  for a in attention}
    if len(signatures) != 1:
        raise ValueError('R104 attention dimensions must agree throughout the original path')
    width, heads, qkv, projected = signatures.pop()
    ff = {layer.ffn.net[0].out_features for layer in encoder.layers}
    ff.update(update.ffn.net[0].out_features for layer in model.joint_layers
              for update in (layer.node_update, layer.solver_update))
    if len(ff) != 1:
        raise ValueError('R104 instance and two-way joint FF widths must agree')
    geometry = encoder.geometry_residual.projection
    return dict(embedding_dim=width, encoder_layer_num=len(encoder.layers),
                joint_layer_num=len(model.joint_layers), head_num=heads, qkv_dim=qkv,
                attention_projection_dim=projected, attention_modules=len(attention),
                ff_hidden_dim=ff.pop(), head_hidden_dim=model.score_head[0].out_features,
                score_head_input_dim=model.score_head[0].in_features,
                solver_feature_hidden=methods.feature_mlp[0].out_features,
                solver_feature_dim=methods.feature_mlp[0].in_features,
                method_id_rows=methods.solver_emb.num_embeddings,
                geometry_input_dim=geometry[0].in_features,
                geometry_hidden_dim=geometry[0].out_features,
                geometry_output_dim=geometry[2].out_features)


def _copy_fresh(model, reference):
    actual = dict(model.named_parameters())
    common = sorted(name for name, value in actual.items()
                    if name in reference and value.shape == reference[name].shape)
    resized = sorted((actual.keys() & reference.keys()) - set(common))
    added = sorted(actual.keys() - reference.keys())
    with torch.no_grad():
        for name in common:
            actual[name].copy_(reference[name])
    copied_hash = parameter_hash({name: actual[name] for name in common})
    if copied_hash != parameter_hash({name: reference[name] for name in common}):
        raise ValueError('Same-name/same-shape fresh parameter copy differs')
    return dict(
        common_parameter_names=common, common_parameter_sha256=copied_hash,
        common_parameter_count=sum(actual[name].numel() for name in common),
        resized_parameter_shapes={name: dict(original=list(reference[name].shape),
                                             actual=list(actual[name].shape)) for name in resized},
        new_parameter_names=added,
        removed_parameter_names=sorted(reference.keys() - actual.keys()),
        independently_initialized_parameter_names=sorted(resized + added),
        independently_initialized_parameter_count=sum(actual[name].numel() for name in resized + added),
        original_parameter_count=sum(value.numel() for value in reference.values()),
        parameter_count=sum(value.numel() for value in actual.values()),
        trainable_sha256=parameter_hash(actual),
    )


def make_model(params, arm, state_dict=None):
    """Build either bounded arm, optionally strictly restore an in-memory state.

    No file is opened, no checkpoint is fetched, and caller CPU RNG is retained.
    Width-changed tensors are newly initialized, never padded/cropped or called
    identical to the original. A supplied state's training provenance cannot be
    inferred from this interface; only equality with fresh trainables is checked.
    """
    resolved = _resolve_params(params, arm)
    original_params = {name: value for name, value in resolved.items()
                       if name not in ('capacity_arm', 'capacity_spec')}
    original_params.update(ORIGINAL)
    with torch.device('cpu'), torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(FRESH_SEED)
        original = RoleInputSelector(ROLE_ARM, **original_params)
        reference = dict(original.named_parameters())
        torch.random.default_generator.manual_seed(CAPACITY_INIT_SEED)
        model = RoleInputSelector(ROLE_ARM, **resolved)
        receipt = _copy_fresh(model, reference)
    dimensions = model_dimensions(model)
    if any(dimensions[name] != value for name, value in CAPACITIES[arm].items()):
        raise ValueError('Instantiated R104 dimensions differ from the bounded capacity')
    model.capacity_arm = arm
    model.capacity_report = dict(
        arm=arm, original_dimensions=model_dimensions(original), actual_dimensions=dimensions,
        capacity_overrides={name: dict(incoming=params.get(name), actual=resolved[name])
                            for name in CAPACITY_FIELDS},
        parameter_count=receipt['parameter_count'],
        original_parameter_count=receipt['original_parameter_count'],
    )
    receipt.update(seed=FRESH_SEED, capacity_init_seed=CAPACITY_INIT_SEED,
                   initialization='scratch_seed2_common_plus_isolated_capacity',
                   fresh_source='regenerated_original_R84_seed2_trainables',
                   r84_fresh_trainable_sha256=parameter_hash(reference),
                   actual_dimensions=dimensions, historical_checkpoint_loaded=False,
                   state_dict_restored=state_dict is not None)
    if state_dict is not None:
        model.load_state_dict(state_dict, strict=True)
        restored_hash = parameter_hash(dict(model.named_parameters()))
        receipt.update(initialization='strict_caller_supplied_in_memory_restore',
                       restored_trainables_match_fresh=restored_hash == receipt['trainable_sha256'],
                       trainable_sha256=restored_hash, historical_checkpoint_loaded=None)
    model.initialization_receipt = receipt
    return model


def initialize_shared(initial, geometry, arm, device='cpu'):
    """R84-style tuple API, with explicit partial-shape fresh reuse for capacity.

    Caller supplies the scratch record and frozen TRAIN geometry in memory.
    Hash equality with the regenerated seed2 original also rejects a trained
    donor merely relabeled 'scratch'. Geometry split provenance remains the
    caller's responsibility; no statistics are fitted or data read here.
    """
    if (initial.get('seed') != FRESH_SEED or initial.get('initialization') != 'scratch'
            or parameter_hash(initial['trainable']) != initial.get('trainable_sha256')):
        raise ValueError('Only the hashed R84 scratch seed2 record is permitted')
    model = make_model(initial['model_params'], arm)
    if initial['trainable_sha256'] != model.initialization_receipt['r84_fresh_trainable_sha256']:
        raise ValueError('R84 record is not the original factory fresh seed2 state; no donor allowed')
    transfer = _copy_fresh(model, initial['trainable'])
    branch = model.instance_encoder.geometry_residual
    mean = torch.as_tensor(geometry['mean'], dtype=branch.mean.dtype, device='cpu')
    std = torch.as_tensor(geometry['std'], dtype=branch.std.dtype, device='cpu')
    if (mean.shape != branch.mean.shape or std.shape != branch.std.shape
            or not torch.isfinite(mean).all() or not torch.isfinite(std).all()
            or not (std > 0).all()):
        raise ValueError('Expected twelve finite geometry moments with positive std')
    with torch.no_grad():
        branch.mean.copy_(mean)
        branch.std.copy_(std)
    receipt = model.initialization_receipt
    receipt.update(transfer, fresh_source='validated_in_memory_R84_scratch_seed2',
                   geometry_fitted_on=geometry.get('fitted_on', 'caller must bind TRAIN-only provenance'))
    return model.to(device), copy.deepcopy(model.params), copy.deepcopy(receipt)
