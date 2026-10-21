"""R87: parameter-matched shared/private nonlinear score residuals.

Runner API::

    model = initialize_from_fresh_common(prepared_initial, params, arm)
    model = make_model(params, arm, state_dict=checkpoint['model'])

Arms are A_shared_residual and B_private_residual. Both retain the complete
ScoreDifferenceSelector, including its shared score_head, and add exactly
M_GLOBAL * 32 * (3 * embedding_dim + 1) parameters. The residual has no bias,
dropout, calibration, separate objective or extra inference rule.

B stores W[m, feature, hidden] and v[m, hidden]. A stores the same initial W
blocks as W[feature, m * hidden + hidden_index] and one shared wide v. A uses
all hidden units for every candidate; B selects a private nonlinear path by
global solver ID. All v start at zero, preserving the exact initial utility,
pair margins and maximin logits of the original model.

The initializer accepts the prepared fresh dict (seed, initialization,
trainable, trainable_sha256, model_params), not a trained checkpoint or a bare
state_dict. It verifies that contract and loads every common parameter
strictly. TRAIN-only geometry buffers remain the runner's responsibility.
make_model instead strictly reconstructs the full R87 checkpoint at inference.

Global ID 17 (RELD_MTL) remains represented in the parameter budget although
it is absent from the frozen scenario's candidate pools. It is not restored
as a candidate. parameter_report() makes the reserved private capacity clear.
Private nonlinear parameters are distinct from the existing ID embedding;
this does not prevent learning method constants or imply improved accuracy.
"""

import math
from collections.abc import Mapping

import torch
import torch.nn.functional as F
from torch import nn

from code.unified_selector.registry import GLOBAL_SOLVERS, M_GLOBAL
from .pairwise_objective import inputs_only
from .pairwise_selector import ScoreDifferenceSelector, maximin_logits


ARMS = ('A_shared_residual', 'B_private_residual')
HIDDEN_DIM = 32
INIT_SEED = 2
SELECTION_HEAD = 'original_score_mlp_maximin'
INACTIVE_CANDIDATE_IDS = tuple(i for i, name in enumerate(GLOBAL_SOLVERS) if name == 'RELD_MTL')


class NonlinearScoreResidual(nn.Module):
    def __init__(self, feature_dim, arm):
        super().__init__()
        self.arm = arm
        self.feature_dim = feature_dim
        generator = torch.Generator(device='cpu').manual_seed(INIT_SEED)
        bank = torch.empty(M_GLOBAL, feature_dim, HIDDEN_DIM).uniform_(
            -1. / math.sqrt(feature_dim), 1. / math.sqrt(feature_dim), generator=generator)
        if arm == 'A_shared_residual':
            self.W = nn.Parameter(bank.permute(1, 0, 2).reshape(feature_dim, -1).contiguous())
            self.v = nn.Parameter(torch.zeros(M_GLOBAL * HIDDEN_DIM))
        else:
            self.W = nn.Parameter(bank)
            self.v = nn.Parameter(torch.zeros(M_GLOBAL, HIDDEN_DIM))

    def forward(self, features, pool_ids, mask):
        if self.arm == 'A_shared_residual':
            hidden = F.gelu(torch.matmul(features, self.W))
            residual = torch.matmul(hidden, self.v)
        else:
            ids = pool_ids.to(device=features.device)
            if ids.ndim == 1:
                # Real task-homogeneous batches gather weights once per method.
                ids = ids.masked_fill(~mask.any(0), 0)
                hidden = F.gelu(torch.einsum('bkf,kfh->bkh', features, self.W[ids]))
                residual = torch.einsum('bkh,kh->bk', hidden, self.v[ids])
            else:
                ids = ids.masked_fill(~mask, 0)
                hidden = F.gelu(torch.einsum('bkf,bkfh->bkh', features, self.W[ids]))
                residual = torch.einsum('bkh,bkh->bk', hidden, self.v[ids])
        return residual.float().masked_fill(~mask, 0.)


class ResidualScoreSelector(ScoreDifferenceSelector):
    def __init__(self, arm, **params):
        params = dict(params)
        if arm not in ARMS:
            raise ValueError('Unknown R87 residual arm')
        if (params.get('solver_representation', 'handcrafted') != 'handcrafted'
                or params.get('pair_mode', 'score_difference') != 'score_difference'
                or params.get('selection_head', SELECTION_HEAD) != SELECTION_HEAD):
            raise ValueError('R87 requires the original handcrafted score-head/maximin model')
        if (params.get('residual_arm', arm) != arm
                or params.get('residual_hidden_dim', HIDDEN_DIM) != HIDDEN_DIM
                or params.get('residual_init_seed', INIT_SEED) != INIT_SEED
                or params.get('residual_global_solvers', GLOBAL_SOLVERS) != GLOBAL_SOLVERS):
            raise ValueError('R87 residual arm, width, initialization or global IDs changed')
        params.update(selection_head=SELECTION_HEAD, residual_arm=arm,
                      residual_hidden_dim=HIDDEN_DIM, residual_init_seed=INIT_SEED,
                      residual_global_solvers=list(GLOBAL_SOLVERS))
        super().__init__(**params)
        self.arm = arm
        self.residual = NonlinearScoreResidual(3 * params['embedding_dim'], arm)

    def forward(self, batch):
        batch = inputs_only(batch)
        features, mask = self.encode_pairs(batch)
        original = self.score_head(features).squeeze(-1).float()
        utility = original + self.residual(features, batch['pool_ids'], mask)
        margins = utility[:, :, None] - utility[:, None, :]
        margins = margins.masked_fill(~(mask[:, :, None] & mask[:, None, :]), 0.)
        return dict(logits=maximin_logits(margins, mask), pair_margin=margins,
                    solver_mask=mask, utility=utility)

    def parameter_report(self):
        per_id = HIDDEN_DIM * (self.residual.feature_dim + 1)
        residual = sum(value.numel() for value in self.residual.parameters())
        total = sum(value.numel() for value in self.parameters())
        return dict(arm=self.arm, common_parameters=total - residual,
                    residual_parameters=residual, total_parameters=total,
                    global_solver_count=M_GLOBAL, residual_hidden_dim=HIDDEN_DIM,
                    inactive_candidate_ids=list(INACTIVE_CANDIDATE_IDS),
                    inactive_candidate_names=[GLOBAL_SOLVERS[i] for i in INACTIVE_CANDIDATE_IDS],
                    inactive_private_parameters=(len(INACTIVE_CANDIDATE_IDS) * per_id
                                                 if self.arm == 'B_private_residual' else 0),
                    candidate_pool_changed=False)


def make_model(params, arm, state_dict=None):
    """Build one R87 arm; checkpoint weights load strictly with no missing keys."""
    model = ResidualScoreSelector(arm, **params)
    if state_dict is not None:
        model.load_state_dict(state_dict, strict=True)
    return model


def initialize_from_fresh_common(initial_state, params, arm):
    """Load the prepared common random parameters; geometry is set by the runner."""
    from .r77_data import parameter_hash

    if (not isinstance(initial_state, Mapping) or initial_state.get('seed') != INIT_SEED
            or initial_state.get('initialization') != 'scratch'
            or not isinstance(initial_state.get('trainable'), Mapping)
            or not isinstance(initial_state.get('model_params'), Mapping)):
        raise ValueError('Expected the prepared fresh seed2 initialization dict, not a checkpoint')
    reference = initial_state['trainable']
    if parameter_hash(reference) != initial_state.get('trainable_sha256'):
        raise ValueError('Prepared fresh trainable hash changed')
    if any(params.get(name) != value for name, value in initial_state['model_params'].items()):
        raise ValueError('Common architecture parameters differ from prepared fresh initialization')
    with torch.random.fork_rng(devices=[]):
        torch.set_rng_state(torch.Generator(device='cpu').manual_seed(INIT_SEED).get_state())
        model = make_model(params, arm)
    common = {name: value for name, value in model.named_parameters()
              if not name.startswith('residual.')}
    if set(reference) != set(common) or any(reference[name].shape != value.shape
                                          or reference[name].dtype != value.dtype
                                          for name, value in common.items()):
        raise ValueError('Prepared fresh parameters must match every common parameter exactly')
    state = model.state_dict()
    state.update(reference)
    model.load_state_dict(state, strict=True)
    if parameter_hash(common) != initial_state['trainable_sha256']:
        raise ValueError('Common parameters did not retain the exact prepared fresh values')
    return model
