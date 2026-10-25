"""R111: fresh R88B stem, private pre-pool family nodes, normalized joint NLL.

Runner API: initialize_shared(initial, geometry, arm, device='cpu') returns
(model, model.params, receipt); make_model(params, arm, state_dict=None)
reconstructs an in-memory state strictly. The only arm is A_family_conditional.
Use F.nll_loss(output['logits'], local_winner_column) and legal-column argmax.
Each non-singleton family transforms post-joint nodes with two private copies
of the original fresh last two EncoderLayers and the original raw-input bias,
then masked mean/max and the shared pool. Root never reads private outputs.
Singleton conditional log-probability is exactly zero, with no private compute.
Training checkpoints each private layer non-reentrantly with preserved RNG;
the actual transformed node outputs still feed masked pooling.
"""

import copy

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from code.unified_selector.registry import GLOBAL_SOLVERS
from .dual_stream import DualStreamSelector
from .r77_data import parameter_hash
from .r88_model import ROLE_SPEC, prepare_role_inputs
from .solver_code_encoder import make_r45_model
from .V4Model import _masked_max, _masked_mean


ARMS = ('A_family_conditional',)
ROLE_ARM = 'B_execution_anchor'
FRESH_SEED, BRANCH_INIT_SEED = 2, 111002
FRESH_REFERENCE_SHA256 = 'dad87d8598f7560e6f170ef048ba1bf515edeb03cef7ec42e16839a36af4ef93'
ORIGINAL = dict(embedding_dim=128, encoder_layer_num=4, joint_layer_num=2,
    head_num=4, qkv_dim=32, ff_hidden_dim=512, head_hidden_dim=256,
    solver_feature_hidden=128, stats_dim=12, dropout=.1, rezero=False)
FAMILIES = dict(diffusion=('DIFUSCO', 'DIFUSCO500', 'T2T', 'T2T500'),
                reld=('RELD_CVRP', 'RELD_MOEL'), decomposition=('GLOP',))
FAMILIES['other_constructive'] = tuple(name for name in GLOBAL_SOLVERS
    if name != 'RELD_MTL' and not any(name in names for names in FAMILIES.values()))
FAMILY_NAMES = tuple(FAMILIES)
FAMILY_IDS = tuple(next((i for i, name in enumerate(FAMILY_NAMES)
    if solver in FAMILIES[name]), -1) for solver in GLOBAL_SOLVERS)
ACTIVE_IDS = tuple(i for i, family in enumerate(FAMILY_IDS) if family >= 0)
SPEC = dict(experiment='R111', schema_version=2, parent='R88B original fresh4node+2joint',
    role_arm=ROLE_ARM, role_spec=copy.deepcopy(ROLE_SPEC), original=copy.deepcopy(ORIGINAL),
    solver_representation='global_ID_plus_34', global_solver_names=list(GLOBAL_SOLVERS),
    families={name: list(names) for name, names in FAMILIES.items()},
    family_names=list(FAMILY_NAMES), solver_family_ids=list(FAMILY_IDS),
    active_global_solver_ids=list(ACTIVE_IDS), excluded_solver_names=['RELD_MTL'],
    root='shared_MLP(h,common_context,mean_joint_family_solver,family_embedding)',
    private_features='postjoint_nodes+family_embedding; two private EncoderLayer node-self-attention/FFN before pooling',
    private_attention_bias='original raw coordinate/directed matrix distance bias, special_count=0',
    private_initialization='independent copies of original fresh seed2 encoder layers2/3',
    family_pool='masked_mean/max followed by original shared pool',
    singleton_conditional='exact logprob zero; skip private node/score computation',
    training_memory='per-private-layer non-reentrant checkpoint, preserve_rng_state=True; node outputs pooled',
    conditional='original shared score_MLP(h_family,common_context,joint_solver)',
    selection_head='family_conditional_logprob', decision_head='logprob',
    probability='P(m|x)=P(f|x)*P(m|f,x)', loss='F.nll_loss(logits,local_winner_column)',
    decision='argmax_all_legal_joint_logprob', fresh_seed=FRESH_SEED,
    branch_init_seed=BRANCH_INIT_SEED, current_query_outcomes_in_forward=False)


def _resolve_params(params, arm):
    params = copy.deepcopy(params)
    if (arm not in ARMS or params.get('r111_arm', arm) != arm
            or params.get('r111_spec', SPEC) != SPEC):
        raise ValueError('R111 arm or family schema differs')
    if any(params.get(key, value) != value for key, value in ORIGINAL.items()):
        raise ValueError('R111 requires the original fresh4node+2joint architecture')
    if (params.get('local_geometry') is not True or params.get('geometry_mode', 'real') != 'real'
            or params.get('solver_feature_weight') != 1. or params.get('ignore_coord_dist', True) is not True
            or params.get('solver_representation', 'handcrafted') != 'handcrafted'
            or params.get('pair_mode', 'score_difference') != 'score_difference'
            or params.get('decision_head', 'classification') not in ('classification', 'logprob')):
        raise ValueError('R111 requires original geometry and global ID+34 inputs')
    params.update(ORIGINAL, r111_arm=arm, r111_spec=copy.deepcopy(SPEC), role_arm=ROLE_ARM,
        role_spec=copy.deepcopy(ROLE_SPEC), selection_head=SPEC['selection_head'], decision_head='logprob',
        pair_mode='score_difference', solver_representation='handcrafted')
    return params


class FamilyConditionalSelector(DualStreamSelector):
    def __init__(self, **params):
        super().__init__(**params)
        d = params['embedding_dim']
        self.register_buffer('solver_family_ids', torch.tensor(FAMILY_IDS, dtype=torch.long))
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(BRANCH_INIT_SEED)
            self.family_embedding = nn.Embedding(len(FAMILY_NAMES), d)
            self.root_head = nn.Sequential(nn.Linear(4 * d, params['head_hidden_dim']), nn.GELU(),
                nn.Dropout(params['dropout']), nn.Linear(params['head_hidden_dim'], 1))
        prototype = self.instance_encoder.layers[-2:]
        self.family_node_layers = nn.ModuleDict({name: copy.deepcopy(prototype) for name in FAMILY_NAMES})

    def prepare_role_inputs(self, batch):
        return prepare_role_inputs(batch, ROLE_ARM)

    def forward(self, batch):
        prepared = self.prepare_role_inputs(batch)
        ids = prepared['pool_ids'].to(device=prepared['node_mask'].device, dtype=torch.long)
        if ids.ndim == 1:
            ids = ids[None].expand(prepared['node_mask'].size(0), -1)
        mask = prepared.get('solver_mask', torch.ones_like(ids, dtype=torch.bool)).to(ids.device).bool()
        if mask.shape != ids.shape or not mask.any(-1).all():
            raise ValueError('R111 requires at least one legal candidate in each row')
        if ((ids[mask] < 0) | (ids[mask] >= len(GLOBAL_SOLVERS))).any():
            raise ValueError('Unknown legal global solver ID')
        ids = ids.masked_fill(~mask, 0)
        families = self.solver_family_ids[ids].masked_fill(~mask, -1)
        if (families[mask] < 0).any():
            raise ValueError('RELD_MTL is excluded from R111')
        nodes, solvers, h, node_mask, mask = self.encode_state(dict(prepared, pool_ids=ids, solver_mask=mask))
        context = self.decoder(h[:, None], solvers, solvers, key_mask=mask)
        if prepared['kind'] == 'coord':
            raw = self.instance_encoder._pad_node(prepared['node'])
            bias = self.instance_encoder._coord_bias(raw, node_mask, 0)
        else:
            _, bias = self.instance_encoder._matrix_features_and_bias(prepared['matrix'], node_mask, 0)
        family_logits = h.new_full((h.size(0), len(FAMILY_NAMES)), float('-inf'), dtype=torch.float32)
        within = h.new_full(mask.shape, float('-inf'), dtype=torch.float32)
        for family, name in enumerate(FAMILY_NAMES):
            members = mask & (families == family)
            counts = members.sum(-1)
            rows = (counts > 0).nonzero(as_tuple=True)[0]
            if not rows.numel():
                continue
            local_mask, tokens = members[rows], solvers[rows]
            embedding = self.family_embedding.weight[family]
            mean = (tokens * local_mask[..., None]).sum(1) / local_mask.sum(1, keepdim=True)
            # Root reads only shared state, never a private conditional feature.
            root_features = torch.cat((h[rows], context[rows, 0], mean, embedding.expand(rows.numel(), -1)), -1)
            family_logits[rows, family] = self.root_head(root_features).squeeze(-1).float()
            r, c = (members & (counts[:, None] == 1)).nonzero(as_tuple=True)
            within[r, c] = 0.
            rows = (counts > 1).nonzero(as_tuple=True)[0]
            if not rows.numel():
                continue
            local_mask, tokens, branch_mask = members[rows], solvers[rows], node_mask[rows]
            x = (nodes[rows] + embedding).masked_fill(~branch_mask[..., None], 0.)
            for layer in self.family_node_layers[name]:
                if self.training and torch.is_grad_enabled():
                    x = checkpoint(layer, x, mask=branch_mask, attn_bias=bias[rows],
                                   use_reentrant=False, preserve_rng_state=True)
                else:
                    x = layer(x, mask=branch_mask, attn_bias=bias[rows])
                x = x.masked_fill(~branch_mask[..., None], 0.)
            family_h = self.pool(torch.cat((_masked_mean(x, branch_mask), _masked_max(x, branch_mask)), -1))
            features = torch.cat((family_h[:, None].expand_as(tokens), context[rows].expand_as(tokens), tokens), -1)
            score = self.score_head(features).squeeze(-1).float().masked_fill(~local_mask, float('-inf'))
            # Compute softmax only on nonempty family rows; absent groups stay -inf.
            logwithin = F.log_softmax(score, dim=-1)
            r, c = local_mask.nonzero(as_tuple=True)
            within[rows[r], c] = logwithin[r, c]
        root = F.log_softmax(family_logits, dim=-1)
        logjoint = (within + root.gather(1, families.clamp_min(0))).masked_fill(~mask, float('-inf'))
        return dict(logits=logjoint, utility=logjoint, solver_mask=mask,
                    family_logits=family_logits, family_ids=families)


def make_model(params, arm, state_dict=None):
    params = _resolve_params(params, arm)
    with torch.device('cpu'), torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(FRESH_SEED)
        original = make_r45_model(params)
        reference = dict(original.named_parameters())
        if parameter_hash(reference) != FRESH_REFERENCE_SHA256:
            raise ValueError('Original R84 factory no longer matches frozen fresh seed2')
        torch.random.default_generator.manual_seed(FRESH_SEED)
        model = FamilyConditionalSelector(**params)
        actual = dict(model.named_parameters())
        with torch.no_grad():
            for name, value in reference.items():
                actual[name].copy_(value)
            for layers in model.family_node_layers.values():
                layers.load_state_dict(model.instance_encoder.layers[-2:].state_dict(), strict=True)
    receipt = dict(seed=FRESH_SEED, branch_init_seed=BRANCH_INIT_SEED,
        initialization='scratch_seed2_original_and_private_node_clones_plus_seed111002_root',
        private_layer_initialization=SPEC['private_initialization'],
        historical_checkpoint_loaded=False, fresh_initialization=True, state_dict_restored=False,
        r84_fresh_trainable_sha256=parameter_hash(reference),
        common_parameter_sha256=parameter_hash({name: actual[name] for name in reference}),
        common_parameter_names=sorted(reference), removed_parameter_names=[],
        new_parameter_names=sorted(actual.keys() - reference.keys()),
        common_parameter_count=sum(p.numel() for p in reference.values()),
        new_parameter_count=sum(p.numel() for name, p in actual.items() if name not in reference),
        parameter_count=sum(p.numel() for p in actual.values()),
        fresh_trainable_sha256=parameter_hash(actual), fresh_state_sha256=parameter_hash(model.state_dict()))
    if state_dict is not None:
        model.load_state_dict(state_dict, strict=True)
        receipt.update(initialization='strict_caller_supplied_in_memory_restore',
                       fresh_initialization=False, state_dict_restored=True)
    receipt.update(trainable_sha256=parameter_hash(actual), state_sha256=parameter_hash(model.state_dict()))
    model.initialization_receipt = receipt
    return model


def initialize_shared(initial, geometry, arm, device='cpu'):
    if (initial.get('seed') != 2 or initial.get('initialization') != 'scratch'
            or initial.get('trainable_sha256') != FRESH_REFERENCE_SHA256
            or parameter_hash(initial['trainable']) != FRESH_REFERENCE_SHA256):
        raise ValueError('R111 only accepts original hashed fresh R84 seed2 tensors')
    model = make_model(initial['model_params'], arm)
    actual, branch = dict(model.named_parameters()), model.instance_encoder.geometry_residual
    mean, std = (torch.as_tensor(geometry[key], dtype=branch.mean.dtype) for key in ('mean', 'std'))
    if (mean.shape != branch.mean.shape or std.shape != branch.std.shape
            or not torch.isfinite(mean).all() or not torch.isfinite(std).all() or not (std > 0).all()
            or geometry.get('split', 'train') != 'train'):
        raise ValueError('Expected frozen finite TRAIN geometry moments')
    with torch.no_grad():
        for name, value in initial['trainable'].items():
            actual[name].copy_(value)
        branch.mean.copy_(mean)
        branch.std.copy_(std)
    model.initialization_receipt.update(fresh_source='validated_in_memory_R84_scratch_seed2',
        geometry_fitted_on='frozen R84 new81000 TRAIN only',
        fresh_state_sha256=parameter_hash(model.state_dict()), state_sha256=parameter_hash(model.state_dict()))
    return model.to(device), copy.deepcopy(model.params), copy.deepcopy(model.initialization_receipt)
