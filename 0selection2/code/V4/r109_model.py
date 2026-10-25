"""Method-conditioned node computation before pooling, tied/private comparison."""

import copy

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from code.unified_selector.registry import GLOBAL_SOLVERS
from .pairwise_selector import maximin_logits
from .r77_data import parameter_hash
from .r88_model import ROLE_SPEC, RoleInputSelector
from .V4Model import _masked_max, _masked_mean


ARMS = ('A_tied_late', 'B_private_late')
ROLE_ARM = 'B_execution_anchor'
FRESH_SEED, BRANCH_INIT_SEED = 2, 109002
FRESH_REFERENCE_SHA256 = 'dad87d8598f7560e6f170ef048ba1bf515edeb03cef7ec42e16839a36af4ef93'
ORIGINAL_PARAMETER_COUNT = 1842953
ACTIVE_IDS = tuple(i for i, name in enumerate(GLOBAL_SOLVERS) if name != 'RELD_MTL')
BLOCK_PARAMETER_COUNT = 395776
COMMON_NEW_PARAMETER_COUNT = 66304
PARAMETER_COUNTS = dict(original=ORIGINAL_PARAMETER_COUNT,
    A_tied_late=ORIGINAL_PARAMETER_COUNT + COMMON_NEW_PARAMETER_COUNT + BLOCK_PARAMETER_COUNT,
    B_private_late=ORIGINAL_PARAMETER_COUNT + COMMON_NEW_PARAMETER_COUNT + len(ACTIVE_IDS) * BLOCK_PARAMETER_COUNT)
ORIGINAL = dict(embedding_dim=128, encoder_layer_num=4, joint_layer_num=2,
    head_num=4, qkv_dim=32, ff_hidden_dim=512, head_hidden_dim=256,
    solver_feature_hidden=128, stats_dim=12, dropout=.1, rezero=False)
SPEC = dict(experiment='R109', parent='R88B/R77 full18 original score selector',
    role_arm=ROLE_ARM, role_spec=copy.deepcopy(ROLE_SPEC), original=copy.deepcopy(ORIGINAL),
    shared_encoder_layers=4, late_encoder_layers=2, late_attention='original directed distance bias',
    candidate_condition='LayerNorm(H4 + Linear(solver_ID_plus_34_features))',
    branch_summary='candidate-specific masked mean/max, shared pool and LayerNorm',
    final_features='own graph summary, original common decoder context, original joint solver token',
    coupling='own graph summary also added to solver token before original two joint layers',
    selection_head='original_score_mlp_maximin', active_global_solver_ids=list(ACTIVE_IDS),
    private_bank_key='global solver identity, never candidate column or winner',
    branch_initialization='all graph blocks clone original fresh layers2/3; pool clones original pool',
    fresh_seed=FRESH_SEED, branch_init_seed=BRANCH_INIT_SEED,
    parameter_counts=copy.deepcopy(PARAMETER_COUNTS), equal_total_parameters=False,
    equal_flops_claim=False, current_query_outcomes_in_forward=False,
    training_memory='non-reentrant activation checkpoint with preserved dropout RNG',
    arms={ARMS[0]: 'one tied graph transformation for all candidate branches',
          ARMS[1]: 'independent graph transformation per active global solver'})


def _resolve_params(params, arm):
    params = copy.deepcopy(params)
    if arm not in ARMS or params.get('r109_arm', arm) != arm or params.get('r109_spec', SPEC) != SPEC:
        raise ValueError('R109 arm or model specification differs')
    if any(params.get(key, value) != value for key, value in ORIGINAL.items()):
        raise ValueError('R109 retains the original four shared layers and score architecture')
    if (params.get('local_geometry') is not True or params.get('geometry_mode', 'real') != 'real'
            or params.get('solver_feature_weight') != 1. or params.get('ignore_coord_dist', True) is not True
            or params.get('decision_head', 'classification') != 'classification'
            or any(key.startswith(('path_', 'expert_', 'capacity_', 'r108_')) for key in params)):
        raise ValueError('R109 requires original role, geometry and ID+34 inputs only')
    params.update(ORIGINAL, r109_arm=arm, r109_spec=copy.deepcopy(SPEC))
    return params


class MethodNodeSelector(RoleInputSelector):
    def __init__(self, **params):
        super().__init__(ROLE_ARM, **params)
        self.r109_arm = params['r109_arm']
        d = params['embedding_dim']
        prototype = nn.ModuleList(copy.deepcopy(list(self.instance_encoder.layers)[-2:]))
        self.late_banks = nn.ModuleDict({
            str(i): copy.deepcopy(prototype) for i in (('shared',) if self.r109_arm == ARMS[0] else ACTIVE_IDS)})
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(BRANCH_INIT_SEED)
            self.method_condition = nn.Linear(d, d, bias=False)
        self.method_node_norm = nn.LayerNorm(d)
        self.method_pool = copy.deepcopy(self.pool)
        self.method_summary_norm = nn.LayerNorm(d)

    def branch(self, nodes, node_mask, bias, solver, global_id):
        x = self.method_node_norm(nodes + self.method_condition(solver)[:, None])
        x = x.masked_fill(~node_mask[..., None], 0.)
        layers = self.late_banks['shared' if self.r109_arm == ARMS[0] else str(global_id)]
        for layer in layers:
            if self.training and torch.is_grad_enabled():
                x = checkpoint(layer, x, mask=node_mask, attn_bias=bias,
                               use_reentrant=False, preserve_rng_state=True)
            else:
                x = layer(x, mask=node_mask, attn_bias=bias)
            x = x.masked_fill(~node_mask[..., None], 0.)
        return self.method_summary_norm(self.method_pool(
            torch.cat((_masked_mean(x, node_mask), _masked_max(x, node_mask)), -1)))

    def encode_method_state(self, batch):
        nodes, node_mask = self.instance_encoder(batch)
        ids = batch['pool_ids'].to(device=nodes.device, dtype=torch.long)
        if ids.ndim == 1:
            ids = ids[None].expand(nodes.size(0), -1)
        mask = batch.get('solver_mask', torch.ones_like(ids, dtype=torch.bool)).to(nodes.device).bool()
        if mask.shape != ids.shape or not mask.any(-1).all():
            raise ValueError('R109 requires at least one legal candidate per instance')
        valid_ids = ids[mask].unique().tolist()
        if any(i not in ACTIVE_IDS for i in valid_ids):
            raise ValueError('Removed or unknown global method cannot enter an R109 branch')
        clean_ids = ids.masked_fill(~mask, 0)
        solvers = self.solver_encoder(clean_ids).masked_fill(~mask[..., None], 0.)
        if batch['kind'] == 'coord':
            raw = self.instance_encoder._pad_node(batch['node'])
            bias = self.instance_encoder._coord_bias(raw, node_mask, 0)
        else:
            _, bias = self.instance_encoder._matrix_features_and_bias(batch['matrix'], node_mask, 0)
        b, k, d = solvers.shape
        summary = nodes.new_zeros((b * k, d))
        for global_id in valid_ids:
            rows, columns = ((ids == global_id) & mask).nonzero(as_tuple=True)
            if rows.unique().numel() != rows.numel():
                raise ValueError('Duplicate legal candidate identity in one instance')
            value = self.branch(nodes[rows], node_mask[rows], bias[rows],
                                solvers[rows, columns], global_id)
            summary = summary.index_copy(0, rows * k + columns, value.to(summary.dtype))
        summary = summary.reshape(b, k, d)
        solvers = (solvers + summary).masked_fill(~mask[..., None], 0.)
        for layer in self.joint_layers:
            nodes, solvers = layer(nodes, solvers, node_mask, mask)
        h = self.pool(torch.cat((_masked_mean(nodes, node_mask), _masked_max(nodes, node_mask)), -1))
        return summary, solvers, h, mask

    def forward(self, batch):
        prepared = self.prepare_role_inputs(batch)
        summary, solvers, h, mask = self.encode_method_state(prepared)
        context = self.decoder(h[:, None], solvers, solvers, key_mask=mask)
        features = torch.cat((summary, context.expand_as(solvers), solvers), -1)
        utility = self.score_head(features).squeeze(-1).float()
        margins = utility[:, :, None] - utility[:, None, :]
        margins = margins.masked_fill(~(mask[:, :, None] & mask[:, None, :]), 0.)
        return dict(utility=utility, pair_margin=margins,
                    logits=maximin_logits(margins, mask), solver_mask=mask)


def _copy_fresh(model, reference):
    actual = dict(model.named_parameters())
    if any(name not in actual or actual[name].shape != value.shape for name, value in reference.items()):
        raise ValueError('Every original R84 trainable must remain present')
    with torch.no_grad():
        for name, value in reference.items():
            actual[name].copy_(value)
        for layers in model.late_banks.values():
            for layer, source in zip(layers, list(model.instance_encoder.layers)[-2:]):
                layer.load_state_dict(source.state_dict(), strict=True)
        model.method_pool.load_state_dict(model.pool.state_dict(), strict=True)
    return dict(common_parameter_sha256=parameter_hash({k: actual[k] for k in reference}),
        common_parameter_count=sum(t.numel() for t in reference.values()),
        original_parameter_count=ORIGINAL_PARAMETER_COUNT,
        new_parameter_count=sum(p.numel() for k, p in actual.items() if k not in reference),
        parameter_count=sum(p.numel() for p in actual.values()),
        common_parameter_names=sorted(reference), new_parameter_names=sorted(actual.keys() - reference.keys()),
        removed_parameter_names=[])


def make_model(params, arm, state=None, *, state_dict=None):
    if state is not None and state_dict is not None:
        raise ValueError('Pass only one in-memory state')
    state = state if state_dict is None else state_dict
    params = _resolve_params(params, arm)
    raw = {k: v for k, v in params.items() if k not in ('r109_arm', 'r109_spec')}
    with torch.device('cpu'), torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(FRESH_SEED)
        original = RoleInputSelector(ROLE_ARM, **raw)
        reference = dict(original.named_parameters())
        if parameter_hash(reference) != FRESH_REFERENCE_SHA256:
            raise ValueError('Original factory no longer matches the frozen R84 scratch state')
        torch.random.default_generator.manual_seed(FRESH_SEED)
        model = MethodNodeSelector(**params)
        receipt = _copy_fresh(model, reference)
    if receipt['parameter_count'] != PARAMETER_COUNTS[arm]:
        raise ValueError('R109 parameter count differs from its declared architecture')
    receipt.update(seed=FRESH_SEED, branch_init_seed=BRANCH_INIT_SEED,
        initialization='scratch_seed2_original_plus_identical_prototype_graph_branches',
        r84_fresh_trainable_sha256=parameter_hash(reference), historical_checkpoint_loaded=False,
        fresh_initialization=True, state_dict_restored=False, equal_total_parameters=False,
        equal_flops_claim=False, fresh_trainable_sha256=parameter_hash(dict(model.named_parameters())),
        fresh_state_sha256=parameter_hash(model.state_dict()))
    if state is not None:
        model.load_state_dict(state, strict=True)
        receipt.update(fresh_initialization=False, state_dict_restored=True,
                       initialization='strict_caller_supplied_in_memory_restore')
    receipt.update(trainable_sha256=parameter_hash(dict(model.named_parameters())),
                   state_sha256=parameter_hash(model.state_dict()))
    model.initialization_receipt = receipt
    return model


def initialize_shared(initial, geometry, arm, device='cpu'):
    if (initial.get('seed') != 2 or initial.get('initialization') != 'scratch'
            or initial.get('trainable_sha256') != FRESH_REFERENCE_SHA256
            or parameter_hash(initial['trainable']) != FRESH_REFERENCE_SHA256):
        raise ValueError('R109 only accepts the original hashed fresh seed2 tensors')
    model = make_model(initial['model_params'], arm)
    transfer = _copy_fresh(model, initial['trainable'])
    branch = model.instance_encoder.geometry_residual
    mean, std = (torch.as_tensor(geometry[key], dtype=branch.mean.dtype) for key in ('mean', 'std'))
    if (mean.shape != branch.mean.shape or std.shape != branch.std.shape
            or not torch.isfinite(mean).all() or not torch.isfinite(std).all() or not (std > 0).all()
            or geometry.get('split', 'train') != 'train'):
        raise ValueError('Expected the frozen finite TRAIN geometry moments')
    with torch.no_grad():
        branch.mean.copy_(mean)
        branch.std.copy_(std)
    model.initialization_receipt.update(transfer,
        fresh_source='validated_in_memory_R84_scratch_seed2',
        geometry_fitted_on='frozen R84 new81000 TRAIN only',
        fresh_state_sha256=parameter_hash(model.state_dict()), state_sha256=parameter_hash(model.state_dict()))
    return model.to(device), copy.deepcopy(model.params), copy.deepcopy(model.initialization_receipt)
