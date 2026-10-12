"""The sole R66 contrast: native R-Drop on g or on detached F + g."""

import torch

from .r64_model import classification_objective


def composite_output(member, base_logits):
    mask = member['solver_mask']
    if base_logits.shape != member['logits'].shape:
        raise ValueError('Frozen and trainable utilities must share the complete legal pool')
    scores = base_logits.detach().float() + member['logits'].float()
    return dict(logits=scores.masked_fill(~mask, float('-inf')), solver_mask=mask)


def training_output(member, base_logits, arm):
    if arm == 'A':
        return member
    if arm == 'B':
        return composite_output(member, base_logits)
    raise ValueError('Unknown R66 arm')


def objective(first, second, base_logits, winner, consistency_weight, arm):
    return classification_objective(training_output(first, base_logits, arm),
                                    training_output(second, base_logits, arm),
                                    winner, consistency_weight)
