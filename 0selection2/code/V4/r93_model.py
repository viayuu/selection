"""R93 original full18 selector, with a prelocked dropout probability only."""

import copy

from .r88_model import make_model as original_model


ARMS = ('A', 'B')
DROPOUT = dict(A=.1, B=0.)
MODEL_ROLE_ARM, ROLE_SLOT = 'B_execution_anchor', 3
SELECTION_HEAD = 'original_score_mlp_maximin'
SPEC = dict(version=1, base='unchanged original R88B/R77 full18 selector',
    new_parameters=False, training='one forward of raw-utility native CE only',
    inference='unchanged maximin; same argmax order as utility',
    dropout=dict(DROPOUT), weight_decay_retained=True)


def make_model(params, arm, state_dict=None):
    if arm not in ARMS:
        raise ValueError('Unknown R93 plain-classification arm')
    params = copy.deepcopy(params)
    if 'r93_arm' in params and (params['r93_arm'] != arm
            or params.get('r93_spec') != SPEC or params.get('dropout') != DROPOUT[arm]):
        raise ValueError('R93 checkpoint arm, dropout or model specification changed')
    if (params.get('role_arm', MODEL_ROLE_ARM) != MODEL_ROLE_ARM
            or params.get('selection_head', SELECTION_HEAD) != SELECTION_HEAD
            or params.get('solver_representation', 'handcrafted') != 'handcrafted'
            or any(name.startswith(('r92_', 'residual_')) for name in params)):
        raise ValueError('R93 preserves the original full-pool model, not a hierarchy or private residual head')
    params.update(dropout=DROPOUT[arm], r93_arm=arm, r93_spec=copy.deepcopy(SPEC))
    return original_model(params, MODEL_ROLE_ARM, state_dict=state_dict)
