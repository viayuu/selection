"""R105 dense instance-FF experts on the otherwise original full18 R88B path.

Factories accept in-memory records only, preserve caller CPU RNG, and never
load a module/checkpoint file. Routing capture is opt-in and input-only; see
``ConditionalRepresentationSelector.capture_routing`` for the reporter API.
"""

import copy

import torch
from torch import nn
from torch.nn import functional as F

from .r77_data import parameter_hash
from .r88_model import ROLE_SPEC, RoleInputSelector
from .V4Model import FeedForward


ARMS = ('A_static_gate', 'B_input_gate')
ROLE_ARM = 'B_execution_anchor'
FRESH_SEED = 2
EXPERT_INIT_SEED = 105002
FRESH_REFERENCE_SHA256 = 'dad87d8598f7560e6f170ef048ba1bf515edeb03cef7ec42e16839a36af4ef93'
ORIGINAL_PARAMETER_COUNT = 1842953
PARAMETER_COUNT = 3425561
EXPERT_SPEC = dict(instance_layers=4, experts_per_layer=4, embedding_dim=128,
                   ff_hidden_dim=512, dropout=0.1, router='Linear(128,4)',
                   router_initialization='zero_weight_and_bias',
                   mixing='all_four_experts_fp32_softmax_and_accumulation',
                   outer_residual_norm='unchanged_original_EncoderLayer')
ORIGINAL = dict(embedding_dim=128, encoder_layer_num=4, joint_layer_num=2,
                head_num=4, qkv_dim=32, ff_hidden_dim=512, head_hidden_dim=256,
                solver_feature_hidden=128, stats_dim=12, dropout=0.1, rezero=False)
SPEC = dict(
    experiment='R105', scope='model_only', parent='full18_R88B_R77_score_path',
    role_arm=ROLE_ARM, role_spec=copy.deepcopy(ROLE_SPEC),
    selection_head='original_score_mlp_maximin', fresh_seed=FRESH_SEED,
    expert_init_seed=EXPERT_INIT_SEED, expert_spec=copy.deepcopy(EXPERT_SPEC),
    original=copy.deepcopy(ORIGINAL), parameter_count=PARAMETER_COUNT,
    arms={ARMS[0]: dict(router_input='zeros_like_post_attention_state',
                        intentionally_inactive_router_feature_parameters=2048),
          ARMS[1]: dict(router_input='post_attention_state',
                        intentionally_inactive_router_feature_parameters=0)},
    comparison='equal_total_tensors_not_equal_active_functional_freedom',
    routing_capture='detached_fp32_valid_physical_node_mean_B_by_4_by_4',
)


def _resolve_params(params, arm):
    if arm not in ARMS:
        raise ValueError('Unknown R105 expert arm')
    params = copy.deepcopy(params)
    if params.get('expert_arm', arm) != arm:
        raise ValueError('R105 expert arm changed during reconstruction')
    if params.get('expert_spec', EXPERT_SPEC) != EXPERT_SPEC:
        raise ValueError('R105 fixed expert specification changed')
    if params.get('seed', FRESH_SEED) != FRESH_SEED:
        raise ValueError('R105 fresh initialization requires seed2')
    if any(params.get(name, value) != value for name, value in ORIGINAL.items()):
        raise ValueError('R105 retains original width128 instance4/joint2 dimensions and dropout0.1')
    if (params.get('local_geometry') is not True
            or params.get('geometry_mode', 'real') != 'real'
            or params.get('solver_feature_weight') != 1.0
            or params.get('ignore_coord_dist', True) is not True
            or params.get('decision_head', 'classification') != 'classification'):
        raise ValueError('R105 retains full18 real geometry, methodID+34 at weight1 and original scores')
    if params.get('role_arm', ROLE_ARM) != ROLE_ARM:
        raise ValueError('R105 retains the R88B execution-anchor role')
    if any(name in params for name in ('capacity_arm', 'capacity_spec')):
        raise ValueError('R105 requires the original R84 architecture, not an R104 capacity arm')
    params.update(ORIGINAL, expert_arm=arm, expert_spec=copy.deepcopy(EXPERT_SPEC))
    return params


class DenseExpertFeedForward(nn.Module):
    """Four distinct original FFNs; only the router input differs between arms."""

    def __init__(self, arm, original):
        super().__init__()
        self.arm = arm
        self.experts = nn.ModuleList([original] + [FeedForward(128, 512, .1) for _ in range(3)])
        self.router = nn.Linear(128, 4)
        nn.init.zeros_(self.router.weight)
        nn.init.zeros_(self.router.bias)
        self._capture_node_mask = None
        self.last_routing = None

    def forward(self, x):
        gate_input = torch.zeros_like(x) if self.arm == ARMS[0] else x
        # Keep both logits and the convex accumulation out of low-precision AMP.
        with torch.autocast(device_type=x.device.type, enabled=False):
            weights = F.linear(gate_input.float(), self.router.weight.float(),
                               self.router.bias.float()).softmax(-1)
        mixed = None
        for index, expert in enumerate(self.experts):
            output = expert(x)
            term = output.float() * weights[..., index:index + 1]
            mixed = term if mixed is None else mixed + term
        if self._capture_node_mask is not None:
            mask = self._capture_node_mask
            # InstanceEncoder prepends CLS, condition and statistics tokens.
            physical = weights[:, 3:].masked_fill(~mask[..., None], 0.)
            self.last_routing = (physical.sum(1) / mask.sum(1, keepdim=True)).detach()
        return mixed.to(output.dtype)


class ConditionalRepresentationSelector(RoleInputSelector):
    def __init__(self, arm, **params):
        super().__init__(ROLE_ARM, **params)
        self.expert_arm = arm
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(EXPERT_INIT_SEED)
            for layer in self.instance_encoder.layers:
                layer.ffn = DenseExpertFeedForward(arm, layer.ffn)

    @torch.no_grad()
    def capture_routing(self, batch):
        """Return ``(routing, outputs)`` using exactly one ordinary forward.

        ``routing`` is detached FP32 [B, 4 layers, 4 experts], in layer/expert
        index order, on the model device. Average only valid physical nodes,
        never the three special tokens or padding. ``outputs`` has the unchanged
        original selector keys. No labels or costs are needed or inspected.
        Mode and RNG behavior are unchanged: callers should use eval() for the
        locked-model diagnostic, or reset RNG to compare training forwards.

        Each instance layer's ``ffn.last_routing`` retains its detached [B,4]
        snapshot until the next capture. Normal forward never updates it and
        adds no hooks, reductions, device synchronization or routing losses.
        """
        mask = batch['node_mask']
        if (mask.ndim != 2 or mask.dtype != torch.bool or not mask.any(-1).all()):
            raise ValueError('Routing capture requires nonempty graphs and boolean node_mask [B,N]')
        branches = [layer.ffn for layer in self.instance_encoder.layers]
        if any(branch._capture_node_mask is not None for branch in branches):
            raise RuntimeError('Routing capture is not reentrant')
        for branch in branches:
            branch._capture_node_mask = mask
            branch.last_routing = None
        try:
            outputs = self(batch)
            routing = torch.stack([branch.last_routing for branch in branches], dim=1)
        except Exception:
            for branch in branches:
                branch.last_routing = None
            raise
        finally:
            for branch in branches:
                branch._capture_node_mask = None
        return routing, outputs


def _copy_fresh(model, reference):
    actual = dict(model.named_parameters())
    common, mapping = [], {}
    for source, tensor in reference.items():
        if source.startswith('instance_encoder.layers.') and '.ffn.' in source:
            target = source.replace('.ffn.', '.ffn.experts.0.', 1)
            mapping[target] = source
        else:
            target = source
            common.append(source)
        if target not in actual or actual[target].shape != tensor.shape:
            raise ValueError('R105 must preserve every original tensor, mapping original FFNs to expert0')
    copied = {name: reference[name] for name in common}
    copied.update({target: reference[source] for target, source in mapping.items()})
    with torch.no_grad():
        for name, tensor in copied.items():
            actual[name].copy_(tensor)
    if parameter_hash({name: actual[name] for name in copied}) != parameter_hash(copied):
        raise ValueError('R105 original/expert0 fresh parameter copy differs')
    independent = sorted(name for name in actual if '.ffn.experts.' in name and name not in mapping)
    routers = sorted(name for name in actual if '.ffn.router.' in name)
    if set(actual) != set(copied) | set(independent) | set(routers):
        raise ValueError('Unexpected R105 parameter outside original, experts or routers')
    return dict(
        common_parameter_names=sorted(common),
        common_parameter_count=sum(actual[name].numel() for name in common),
        fresh_common_parameter_sha256=parameter_hash({name: actual[name] for name in common}),
        expert0_parameter_mapping=dict(sorted(mapping.items())),
        expert0_copied_parameter_count=sum(actual[name].numel() for name in mapping),
        independently_initialized_parameter_names=independent,
        independently_initialized_parameter_count=sum(actual[name].numel() for name in independent),
        zero_initialized_parameter_names=routers,
        zero_initialized_parameter_count=sum(actual[name].numel() for name in routers),
        new_parameter_names=sorted(actual.keys() - reference.keys()),
        removed_parameter_names=sorted(reference.keys() - actual.keys()),
        original_parameter_count=sum(tensor.numel() for tensor in reference.values()),
        parameter_count=sum(tensor.numel() for tensor in actual.values()),
    )


def make_model(params, arm, state_dict=None):
    """Construct a fresh CPU model or strictly restore a caller's in-memory state.

    Trained restoration is supported without a fresh-state equality requirement;
    its receipt explicitly reports restore provenance as unknown. The immutable
    fresh hashes describe construction BEFORE restoration, never a trained state.
    """
    resolved = _resolve_params(params, arm)
    original_params = {key: value for key, value in resolved.items()
                       if key not in ('expert_arm', 'expert_spec')}
    with torch.device('cpu'), torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(FRESH_SEED)
        original = RoleInputSelector(ROLE_ARM, **original_params)
        reference = dict(original.named_parameters())
        reference_hash = parameter_hash(reference)
        if reference_hash != FRESH_REFERENCE_SHA256:
            raise ValueError('Original factory differs from the exact R84 fresh seed2 source')
        torch.random.default_generator.manual_seed(FRESH_SEED)
        model = ConditionalRepresentationSelector(arm, **resolved)
        receipt = _copy_fresh(model, reference)
    if (receipt['original_parameter_count'] != ORIGINAL_PARAMETER_COUNT
            or receipt['parameter_count'] != PARAMETER_COUNT):
        raise ValueError('R105 instantiated parameter arithmetic differs from the fixed full18 contract')
    fresh_hash = parameter_hash(dict(model.named_parameters()))
    fresh_state_hash = parameter_hash(model.state_dict())
    inactive = [f'instance_encoder.layers.{i}.ffn.router.weight' for i in range(4)] if arm == ARMS[0] else []
    receipt.update(
        seed=FRESH_SEED, expert_init_seed=EXPERT_INIT_SEED,
        initialization='scratch_seed2_common_plus_isolated_experts',
        fresh_source='regenerated_original_R84_seed2_trainables',
        r84_fresh_trainable_sha256=reference_hash,
        fresh_trainable_sha256=fresh_hash, trainable_sha256=fresh_hash,
        fresh_state_sha256=fresh_state_hash, state_sha256=fresh_state_hash,
        historical_checkpoint_loaded=False, state_dict_restored=False,
        fresh_initialization=True,
        transfer_receipt_scope='fresh_construction_before_optional_state_dict_restore',
        intentionally_inactive_parameter_names=inactive,
        intentionally_inactive_parameter_count=sum(dict(model.named_parameters())[name].numel() for name in inactive),
        inactive_reason='A feeds zeros_like(x); router feature W has identically zero gradients; bias is learned',
        equal_total_parameters_not_equal_active_freedom=True,
        parameter_count_arithmetic='1842953 + 12 * 131712 + 4 * 516 = 3425561',
    )
    if state_dict is not None:
        model.load_state_dict(state_dict, strict=True)
        restored_hash = parameter_hash(dict(model.named_parameters()))
        restored_state_hash = parameter_hash(model.state_dict())
        receipt.update(
            initialization='strict_caller_supplied_in_memory_restore',
            fresh_source='fresh_hashes_are_reference_only_not_restoration_provenance',
            fresh_initialization=False, state_dict_restored=True,
            historical_checkpoint_loaded=None,
            restored_trainables_match_fresh=restored_hash == fresh_hash,
            restored_state_matches_fresh=restored_state_hash == fresh_state_hash,
            trainable_sha256=restored_hash, state_sha256=restored_state_hash,
        )
    model.initialization_receipt = receipt
    return model


def initialize_shared(initial, geometry, arm, device='cpu'):
    """Validate/copy the exact R84 scratch record and caller's TRAIN moments.

    No fitting, data reads, cost/winner access or checkpoint-file loading occurs.
    Geometry split/source binding remains the caller's responsibility.
    """
    if (initial.get('seed') != FRESH_SEED or initial.get('initialization') != 'scratch'
            or parameter_hash(initial['trainable']) != initial.get('trainable_sha256')):
        raise ValueError('Only the hashed R84 scratch seed2 record is permitted')
    model = make_model(initial['model_params'], arm)
    if initial['trainable_sha256'] != model.initialization_receipt['r84_fresh_trainable_sha256']:
        raise ValueError('R84 record is not the exact original factory fresh seed2 state; no donor allowed')
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
    receipt['fresh_state_sha256'] = receipt['state_sha256'] = parameter_hash(model.state_dict())
    return model.to(device), copy.deepcopy(model.params), copy.deepcopy(receipt)
