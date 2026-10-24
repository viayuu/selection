"""R108: original R88B utility t plus fixed TRAIN log prior, identical A/B.

Factories use only caller-supplied in-memory data. Compact params allocate the
full persistent prior buffers for strict checkpoint restoration without files.
"""

import copy

import torch

from code.unified_selector.registry import GLOBAL_SOLVERS, PROBLEMS
from .pairwise_selector import maximin_logits
from .r77_data import parameter_hash
from .r88_model import ROLE_SPEC, RoleInputSelector
from .r108_priors import FORMAT, PRIOR_SPEC, json_hash, validate_priors


ARMS = ('A_prior_ce', 'B_density_ratio')
ROLE_ARM = 'B_execution_anchor'
FRESH_SEED = 2
FRESH_REFERENCE_SHA256 = 'dad87d8598f7560e6f170ef048ba1bf515edeb03cef7ec42e16839a36af4ef93'
PARAMETER_COUNTS = dict(original=1842953, **{arm: 1842953 for arm in ARMS})
ORIGINAL = dict(embedding_dim=128, encoder_layer_num=4, joint_layer_num=2,
                head_num=4, qkv_dim=32, ff_hidden_dim=512, head_hidden_dim=256,
                solver_feature_hidden=128, stats_dim=12, dropout=.1, rezero=False)
SPEC = dict(
    experiment='R108', parent='original R88B/R77 full18 score-difference selector',
    original=copy.deepcopy(ORIGINAL), role_arm=ROLE_ARM, role_spec=copy.deepcopy(ROLE_SPEC),
    selection_head='original_score_mlp_maximin', prior_spec=copy.deepcopy(PRIOR_SPEC),
    fresh_seed=FRESH_SEED, parameter_counts=copy.deepcopy(PARAMETER_COUNTS),
    new_trainable_parameters=0, same_forward_both_arms=True,
    utility='s=log(pi(problem,exact_true_n))+original_utility_t',
    inference='original utility differences and maximin; final argmax s',
    forward_inputs='original inputs plus fixed classifier buffers; no labels, costs or instance IDs',
    arms={ARMS[0]: 'ordinary legal CE(s)', ARMS[1]: 'canonical half-positive half-product-noise density ratio'},
    efficacy='low-priority precise falsification; no certified efficacy',
)
PRIOR_BUFFERS = ('prior_n_values', 'prior_task', 'prior_conditional', 'prior_legal', 'prior_identity')


def _prior_descriptor(record):
    return dict(format=FORMAT, identity_sha256=record['identity_sha256'],
        source_identity_sha256=json_hash(record['source_identity']),
        scenario_id=record['source_identity']['scenario_id'],
        data_root=record['source_identity']['data_root'], rows_per_problem=record['rows_per_problem'],
        task_count=len(PROBLEMS), global_solver_count=len(GLOBAL_SOLVERS),
        max_n_bins=max(len(t['n_values']) for t in record['tasks'].values()))


def _resolve_params(params, arm, priors):
    if arm not in ARMS:
        raise ValueError('Unknown R108 conditional-prior arm')
    params = copy.deepcopy(params)
    if params.get('r108_arm', arm) != arm or params.get('r108_spec', SPEC) != SPEC:
        raise ValueError('R108 arm or specification changed during reconstruction')
    if params.get('seed', FRESH_SEED) != FRESH_SEED:
        raise ValueError('R108 requires scratch seed2')
    if any(params.get(name, value) != value for name, value in ORIGINAL.items()):
        raise ValueError('R108 retains original R88B architecture and dropout .1')
    if (params.get('local_geometry') is not True or params.get('geometry_mode', 'real') != 'real'
            or params.get('solver_feature_weight') != 1. or params.get('ignore_coord_dist', True) is not True
            or params.get('decision_head', 'classification') != 'classification'
            or params.get('role_arm', ROLE_ARM) != ROLE_ARM
            or any(name.startswith(('path_', 'expert_', 'capacity_', 'r92_', 'r93_', 'r106_', 'r107_'))
                   for name in params)):
        raise ValueError('R108 retains original R88B role, real geometry, ID+34 and score path')
    descriptor = params.get('r108_prior')
    if priors is not None:
        validate_priors(priors)
        supplied = _prior_descriptor(priors)
        if descriptor is not None and {k: v for k, v in descriptor.items() if k != 'buffers_sha256'} != supplied:
            raise ValueError('R108 checkpoint and supplied prior identities differ')
        descriptor = dict(supplied, **({} if descriptor is None else
                                     {'buffers_sha256': descriptor['buffers_sha256']}))
    if (not isinstance(descriptor, dict) or descriptor.get('format') != FORMAT
            or descriptor.get('task_count') != len(PROBLEMS)
            or descriptor.get('global_solver_count') != len(GLOBAL_SOLVERS)
            or not isinstance(descriptor.get('max_n_bins'), int) or not 0 < descriptor['max_n_bins'] <= 10000
            or not isinstance(descriptor.get('identity_sha256'), str)
            or len(descriptor['identity_sha256']) != 64):
        raise ValueError('Supply TRAIN priors, or checkpoint params with compact prior dimensions/identity')
    params.update(ORIGINAL, r108_arm=arm, r108_spec=copy.deepcopy(SPEC), r108_prior=copy.deepcopy(descriptor))
    return params


class ConditionalPriorSelector(RoleInputSelector):
    """One unchanged learned score path; fixed priors are classifier parameters."""

    def __init__(self, **params):
        super().__init__(ROLE_ARM, **params)
        self.r108_arm = params['r108_arm']
        shape = params['r108_prior']
        p, bins, m = shape['task_count'], shape['max_n_bins'], shape['global_solver_count']
        self.register_buffer('prior_n_values', torch.full((p, bins), torch.iinfo(torch.int64).max, dtype=torch.int64))
        self.register_buffer('prior_task', torch.zeros(p, m))
        self.register_buffer('prior_conditional', torch.zeros(p, bins, m))
        self.register_buffer('prior_legal', torch.zeros(p, m, dtype=torch.bool))
        self.register_buffer('prior_identity', torch.tensor(list(bytes.fromhex(shape['identity_sha256'])), dtype=torch.uint8))

    @torch.no_grad()
    def install_priors(self, record):
        for problem in PROBLEMS:
            task = record['tasks'][problem]
            p, ids = task['problem_id'], torch.tensor(task['pool_ids'], dtype=torch.int64)
            count = len(task['n_values'])
            self.prior_n_values[p, :count] = torch.tensor(task['n_values'], dtype=torch.int64)
            self.prior_task[p, ids] = torch.tensor(task['task_prior'], dtype=torch.float32)
            self.prior_conditional[p, :count, ids] = torch.tensor(task['conditional_prior'], dtype=torch.float32)
            self.prior_legal[p, ids] = True

    def prior_for(self, batch, mask):
        p = int(batch['problem_id'])
        n, ids = batch['n'], batch['pool_ids']
        if (not 0 <= p < len(PROBLEMS) or n.dtype != torch.int64
                or n.shape != (mask.size(0),) or not (n > 0).all()):
            raise ValueError('R108 prior requires a known task and exact positive true N')
        if ids.ndim == 1:
            ids = ids[None].expand(mask.size(0), -1)
        if ids.dtype != torch.int64 or ids.shape != mask.shape:
            raise ValueError('R108 pool IDs must match the original local solver slots')
        ids = ids.to(mask.device).masked_fill(~mask, 0)
        if not ((ids >= 0) & (ids < self.prior_task.size(-1))).all():
            raise ValueError('R108 candidate ID outside original vocabulary')
        # Require the frozen legal set, allowing arbitrary local column order
        # and masked padding. No per-query renormalization or candidate quotas.
        present = torch.zeros_like(self.prior_task[p], dtype=torch.int64)[None].expand(mask.size(0), -1).clone()
        present.scatter_add_(1, ids, mask.long())
        if not (present == self.prior_legal[p].long()[None]).all():
            raise ValueError('R108 forward must retain the complete frozen task candidate pool')
        n = n.to(mask.device)
        positions = torch.searchsorted(self.prior_n_values[p], n).clamp_max(self.prior_n_values.size(1) - 1)
        seen = self.prior_n_values[p, positions] == n
        probabilities = torch.where(seen[:, None], self.prior_conditional[p, positions], self.prior_task[p])
        return probabilities.gather(1, ids).masked_fill(~mask, 0.)

    def forward(self, batch):
        output = super().forward(batch)
        residual, mask = output['utility'], output['solver_mask']
        prior = self.prior_for(batch, mask)
        utility = (residual + prior.masked_fill(~mask, 1.).log()).masked_fill(~mask, 0.)
        margins = utility[:, :, None] - utility[:, None, :]
        margins = margins.masked_fill(~(mask[:, :, None] & mask[:, None, :]), 0.)
        output.update(utility=utility, residual_utility=residual, conditional_prior=prior,
                      pair_margin=margins, logits=maximin_logits(margins, mask))
        return output


def _buffer_hash(model):
    return parameter_hash({name: getattr(model, name) for name in PRIOR_BUFFERS})


def make_model(params, arm, state=None, *, priors=None):
    """Construct exact scratch seed2, or restore every buffer strictly, no I/O."""
    if priors is None and state is None:
        raise ValueError('Fresh R108 construction requires caller-supplied TRAIN priors')
    resolved = _resolve_params(params, arm, priors)
    with torch.device('cpu'), torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(FRESH_SEED)
        model = ConditionalPriorSelector(**resolved)
    fresh = parameter_hash(dict(model.named_parameters()))
    count = sum(p.numel() for p in model.parameters())
    if fresh != FRESH_REFERENCE_SHA256 or count != PARAMETER_COUNTS[arm]:
        raise ValueError('R108 must preserve every exact original R84 scratch seed2 trainable')
    if priors is not None:
        model.install_priors(priors)
        digest = _buffer_hash(model)
        if resolved['r108_prior'].get('buffers_sha256', digest) != digest:
            raise ValueError('R108 fixed prior buffers changed')
        model.params['r108_prior']['buffers_sha256'] = digest
    fresh_state = parameter_hash(model.state_dict()) if priors is not None else None
    if state is not None:
        if 'buffers_sha256' not in model.params['r108_prior']:
            raise ValueError('R108 checkpoint params must identify complete frozen prior buffers')
        model.load_state_dict(state, strict=True)
        expected_identity = torch.tensor(list(bytes.fromhex(resolved['r108_prior']['identity_sha256'])), dtype=torch.uint8)
        if (not torch.equal(model.prior_identity.cpu(), expected_identity)
                or _buffer_hash(model) != model.params['r108_prior']['buffers_sha256']):
            raise ValueError('R108 restored prior buffers differ from checkpoint identity')
    model.initialization_receipt = dict(seed=FRESH_SEED,
        initialization='strict_in_memory_restore' if state is not None else 'scratch_seed2_with_fixed_TRAIN_priors',
        historical_checkpoint_loaded=None if state is not None else False,
        fresh_initialization=state is None, state_dict_restored=state is not None,
        r84_fresh_trainable_sha256=fresh, fresh_trainable_sha256=fresh,
        trainable_sha256=parameter_hash(dict(model.named_parameters())),
        fresh_state_sha256=fresh_state, state_sha256=parameter_hash(model.state_dict()),
        parameter_count=count, original_parameter_count=PARAMETER_COUNTS['original'],
        common_parameter_count=count, common_parameter_names=sorted(dict(model.named_parameters())),
        common_parameter_sha256=fresh, new_parameter_names=[], removed_parameter_names=[],
        independently_initialized_parameter_names=[], independently_initialized_parameter_count=0,
        new_trainable_parameter_count=0, prior_identity_sha256=resolved['r108_prior']['identity_sha256'],
        prior_buffers_sha256=model.params['r108_prior']['buffers_sha256'], prior_fit_split='train',
        fixed_aggregate_prior_not_query_context=True)
    return model


def initialize_shared(initial, geometry, arm, device='cpu', *, priors):
    """Validate the original scratch record and caller-supplied TRAIN moments."""
    reference = initial['trainable']
    if (initial.get('seed') != FRESH_SEED or initial.get('initialization') != 'scratch'
            or initial.get('trainable_sha256') != FRESH_REFERENCE_SHA256
            or parameter_hash(reference) != FRESH_REFERENCE_SHA256):
        raise ValueError('R108 requires exact hashed original R84 scratch seed2, never a trained donor')
    model = make_model(initial['model_params'], arm, priors=priors)
    actual = dict(model.named_parameters())
    if set(actual) != set(reference) or any(actual[k].shape != reference[k].shape for k in reference):
        raise ValueError('R108 must retain every original named trainable and no extra trainables')
    branch = model.instance_encoder.geometry_residual
    mean = torch.as_tensor(geometry['mean'], dtype=branch.mean.dtype, device='cpu')
    std = torch.as_tensor(geometry['std'], dtype=branch.std.dtype, device='cpu')
    if (geometry.get('split', 'train') != 'train' or mean.shape != branch.mean.shape
            or std.shape != branch.std.shape or not torch.isfinite(mean).all()
            or not torch.isfinite(std).all() or not (std > 0).all()):
        raise ValueError('R108 requires twelve finite frozen TRAIN means and positive standard deviations')
    with torch.no_grad():
        for name in reference:
            actual[name].copy_(reference[name])
        branch.mean.copy_(mean)
        branch.std.copy_(std)
    receipt = model.initialization_receipt
    receipt.update(fresh_source='validated_in_memory_R84_scratch_seed2',
                   geometry_fitted_on=geometry.get('fitted_on', 'frozen R84 new81000 TRAIN only'))
    receipt['fresh_state_sha256'] = receipt['state_sha256'] = parameter_hash(model.state_dict())
    return model.to(device), copy.deepcopy(model.params), copy.deepcopy(receipt)
