"""R67's fresh native selector, disposable projector and averaged-x transfer."""

import copy

import torch
import torch.nn.functional as F
from torch import nn

from .multitask_probe import state_hash
from .r64_model import NativeUtilitySelector, initialize
from .r65_optimizer import evaluation_mode, make_optimizer


PROJECTOR_SEED, HEAD_SEED = 670002, 670003
INPUT_KEYS = frozenset(('problem_id', 'kind', 'node', 'node_mask', 'd_in', 'matrix',
                        'cbits', 'problem_desc', 'coord_dist', 'pool_ids', 'mask', 'n',
                        'node_geom', 'solver_mask'))


def model_inputs(batch):
    return {key: value for key, value in batch.items() if key in INPUT_KEYS}


def cpu_state(module):
    return {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}


def projector():
    return nn.Sequential(nn.Linear(384, 256), nn.ReLU(), nn.Linear(256, 128))


def score_head(params):
    return nn.Sequential(nn.Linear(3*params['embedding_dim'], params['head_hidden_dim']),
                         nn.GELU(), nn.Dropout(params['dropout']),
                         nn.Linear(params['head_hidden_dim'], 1))


def shared_initialization(geometry, device='cpu'):
    selector, params = initialize(geometry, device, seed=2)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(PROJECTOR_SEED)
        projection = projector()
        torch.manual_seed(HEAD_SEED)
        fresh_head = score_head(params)
    states = dict(selector=cpu_state(selector), projector=cpu_state(projection),
                  stage2_head=cpu_state(fresh_head))
    receipt = dict(seed=2, projector_seed=PROJECTOR_SEED, stage2_head_seed=HEAD_SEED,
        fresh=True, no_trained_checkpoint_reused=True, explicitly_copied_across_arms=True,
        hashes={key: state_hash(value) for key, value in states.items()})
    return params, states, receipt


class PretrainingSelector(nn.Module):
    def __init__(self, params, states, device='cpu'):
        super().__init__()
        self.selector = NativeUtilitySelector(**params)
        self.projector = projector()
        self.selector.load_state_dict(states['selector'], strict=True)
        self.projector.load_state_dict(states['projector'], strict=True)
        self.to(device)

    def forward(self, batch):
        features, mask = self.selector.encode_pairs(model_inputs(batch))
        logits = self.selector.score_head(features).squeeze(-1).float()
        mean = features.float().masked_fill(~mask[..., None], 0).sum(1) / mask.sum(1, keepdim=True)
        representation = F.normalize(mean, dim=-1)
        projection = F.normalize(self.projector(representation).float(), dim=-1)
        return dict(logits=logits.masked_fill(~mask, float('-inf')), solver_mask=mask,
                    representation=representation, projection=projection)


def transition_to_stage2(pretraining, old_optimizer, head_state, amp=True):
    evaluation_mode(pretraining, old_optimizer, 'B')
    terminal = cpu_state(pretraining.selector)
    trunk = {key: value for key, value in terminal.items() if not key.startswith('score_head.')}
    selector = NativeUtilitySelector(**pretraining.selector.params)
    selector.load_state_dict(terminal, strict=True)
    selector.score_head = score_head(selector.params)
    selector.score_head.load_state_dict(head_state, strict=True)
    selector.to(next(pretraining.parameters()).device)
    transferred = {key: value for key, value in cpu_state(selector).items()
                   if not key.startswith('score_head.')}
    if state_hash(trunk) != state_hash(transferred):
        raise ValueError('Stage2 must receive the terminal averaged-x trunk exactly')
    optimizer = make_optimizer(selector, 'B')
    scaler = torch.amp.GradScaler('cuda', enabled=amp)
    receipt = dict(source_parameter_mode='terminal averaged x',
        stage1_selector_hash=state_hash(terminal), terminal_trunk_hash=state_hash(trunk),
        transferred_trunk_hash=state_hash(transferred), stage2_head_hash=state_hash(cpu_state(selector.score_head)),
        entire_head_replaced=True, projector_discarded=True, all_selector_parameters_trainable=True,
        optimizer_fresh=len(optimizer.state) == 0, optimizer_k=[g['k'] for g in optimizer.param_groups],
        scaler_fresh=copy.deepcopy(scaler.state_dict()), stage_successful_updates=0,
        stage_presentations=0, stage_rdrop_updates=0, scale_growth_successes=0)
    if not all(p.requires_grad for p in selector.parameters()) or any(g['k'] for g in optimizer.param_groups):
        raise ValueError('Stage2 must start fully trainable with a fresh optimizer')
    return selector, optimizer, scaler, receipt
