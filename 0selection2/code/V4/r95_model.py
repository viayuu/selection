"""R95 keeps the original R88B/R77 model; only the TRAIN sampler changes."""

import copy

from .r88_model import make_model as original_model


ARMS = ('A', 'B')
DROPOUT = dict(A=.1, B=.1)
MODEL_ROLE_ARM, ROLE_SLOT = 'B_execution_anchor', 3
SELECTION_HEAD = 'original_score_mlp_maximin'
SPEC = dict(version=1, base='unchanged original R88B/R77 full18 selector',
    new_parameters=False, training='original CE/compare/risk/two-pass RDrop',
    inference='unchanged original score MLP and maximin', dropout=DROPOUT,
    difference='TRAIN row proposal only; full-row inverse-probability correction')


def make_model(params, arm, state_dict=None):
    if arm not in ARMS:
        raise ValueError('Unknown R95 arm')
    params = copy.deepcopy(params)
    if 'r95_arm' in params and (params['r95_arm'] != arm or params.get('r95_spec') != SPEC):
        raise ValueError('R95 model arm or specification changed')
    if (params.get('role_arm', MODEL_ROLE_ARM) != MODEL_ROLE_ARM
            or params.get('selection_head', SELECTION_HEAD) != SELECTION_HEAD
            or params.get('solver_representation', 'handcrafted') != 'handcrafted'
            or params.get('dropout', .1) != .1
            or any(name.startswith(('r92_', 'r93_', 'r94_', 'residual_')) for name in params)):
        raise ValueError('R95 requires the unchanged full-pool original selector')
    params.update(r95_arm=arm, r95_spec=copy.deepcopy(SPEC))
    return original_model(params, MODEL_ROLE_ARM, state_dict=state_dict)
