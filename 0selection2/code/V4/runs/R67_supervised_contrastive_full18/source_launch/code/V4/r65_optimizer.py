"""R65's fixed optimizer settings and official parameter-mode transitions."""

import math

import torch

from .r65_official_schedulefree import AdamWScheduleFree


BASE_LR, WARMUP_UPDATES = 3e-4, 720
COMMON = dict(lr=BASE_LR, betas=(.9, .999), eps=1e-8, weight_decay=1e-4, foreach=False)
ARM_CONFIGS = {
    'A': dict(implementation='torch.optim.AdamW', **COMMON,
              external_warmup_updates=WARMUP_UPDATES, deployment_parameters='ordinary'),
    'B': dict(implementation='official AdamWScheduleFree', **COMMON,
              warmup_steps=WARMUP_UPDATES, r=0., weight_lr_power=2., inner_momentum=0.,
              external_warmup_updates=0, gradient_parameters='y', deployment_parameters='averaged x'),
}


def make_optimizer(model, arm):
    if arm == 'A':
        return torch.optim.AdamW(model.parameters(), **COMMON)
    if arm == 'B':
        return AdamWScheduleFree(model.parameters(), **COMMON, warmup_steps=WARMUP_UPDATES,
                                 r=0., weight_lr_power=2., inner_momentum=0.)
    raise ValueError('Unknown R65 arm')


def training_mode(model, optimizer, arm):
    if arm == 'B':
        optimizer.train()
    model.train()


def evaluation_mode(model, optimizer, arm):
    if arm == 'B':
        optimizer.eval()
    model.eval()


def configure_update_lr(optimizer, arm, successful):
    effective = BASE_LR * min((successful+1)/WARMUP_UPDATES, 1.)
    if arm == 'A':
        for group in optimizer.param_groups:
            group['lr'] = effective
    elif any(group['k'] != successful or group['lr'] != BASE_LR or not group['train_mode']
             for group in optimizer.param_groups):
        raise ValueError('Schedule-free must train at y with its internal warmup and constant base LR')
    return effective


def observed_lr(optimizer, arm, expected):
    values = [group['scheduled_lr'] if arm == 'B' else group['lr'] for group in optimizer.param_groups]
    if any(not math.isclose(value, expected, rel_tol=1e-12, abs_tol=0.) for value in values):
        raise ValueError('The optimizers must use the same effective successful-update LR')
    return float(values[0])
