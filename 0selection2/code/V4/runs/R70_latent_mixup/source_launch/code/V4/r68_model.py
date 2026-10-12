"""R68's shared contrastive trunk and identical ordinary-weight selectors."""

import torch

from .multitask_probe import state_hash
from .r64_model import NativeUtilitySelector


INPUT_KEYS = frozenset(('problem_id', 'kind', 'node', 'node_mask', 'd_in', 'matrix',
                        'cbits', 'problem_desc', 'coord_dist', 'pool_ids', 'mask', 'n',
                        'node_geom', 'solver_mask'))
OPTIMIZER_CONFIG = dict(lr=3e-4, betas=(.9, .999), eps=1e-8,
                        weight_decay=1e-4, foreach=False)
PARAMETER_MODE = 'ordinary AdamW weights'


def model_inputs(batch):
    return {key: value for key, value in batch.items() if key in INPUT_KEYS}


def cpu_state(module):
    return {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}


def starting_state(terminal, fresh_head, expected_head_hash):
    if (terminal['stage'] != 1 or terminal['epoch'] != 40
            or terminal['parameter_mode'] != 'averaged x' or terminal['selection'] != 'fixed terminal'
            or terminal['successful_updates'] != 9600 or terminal['presentations'] != 7200000
            or terminal['args']['arm'] != 'B'):
        raise ValueError('R68 inherits only R67B fixed stage1 terminal averaged-x trunk')
    if state_hash(fresh_head) != expected_head_hash:
        raise ValueError('The entire fresh R67 seed670003 head must be copied exactly')
    params = terminal['args']['model_params']
    trunk = {key: value.detach().cpu().clone() for key, value in terminal['model'].items()
             if not key.startswith('score_head.')}
    if any('projector' in key for key in trunk):
        raise ValueError('Projector parameters cannot enter a deployment selector')
    combined = dict(trunk, **{'score_head.'+key: value.detach().cpu().clone()
                             for key, value in fresh_head.items()})
    selector = make_selector(params, combined)
    state = cpu_state(selector)
    receipt = dict(inherited_stage1_successful_updates_once=9600,
        inherited_stage1_presentations_once=7200000, inherited_from='R67B stage1 terminal',
        source_parameter_mode='terminal averaged x', deployment_parameter_mode=PARAMETER_MODE,
        inherited_trunk_hash=state_hash(trunk), starting_selector_hash=state_hash(state),
        stage2_head_seed=670003, stage2_head_hash=state_hash(cpu_state(selector.score_head)),
        entire_fresh_head_copied=True, inherited_utility_head_loaded=False,
        projector_loaded=False, explicit_identical_copy_across_arms=True,
        all_selector_parameters_trainable=all(p.requires_grad for p in selector.parameters()),
        fresh_encoders=False, schedulefree_optimizer=False, optimizer_mode_transition=False)
    return params, state, receipt


def make_selector(params, state, device='cpu'):
    selector = NativeUtilitySelector(**params)
    selector.load_state_dict(state, strict=True)
    selector.requires_grad_(True)
    return selector.to(device)


def make_optimizer(model):
    return torch.optim.AdamW(model.parameters(), **OPTIMIZER_CONFIG)


def update_lr(successful_before):
    return 3e-4*min((successful_before+1)/720, 1.)
