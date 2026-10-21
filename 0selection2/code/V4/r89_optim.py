"""R89 single-device AdamW / Muon-hybrid optimizer factory.

API: build_optimizer(selector, arm, lr=1e-4, weight_decay=1e-4).
optimizer_manifest(opt) exports actual parameter names/shapes/counts and source
provenance. optimizer_metrics(opt) returns last-step Tensor metrics, including
muon_step_applied (0 or 1, NOT a cumulative counter). Hybrid state, param_groups,
zero_grad, state_dict and load_state_dict implement torch.optim.Optimizer.

A_adamw is the real torch.optim.AdamW over selector.parameters(), in its
original order and with original defaults. B_muon_hybrid uses the same real
AdamW for all auxiliary tensors. Only hidden nn.Linear weights under the
explicit whitelist, with min(shape)>1, receive Muon. All groups share base LR
and WD. Momentum and model parameters are FP32; only the NS iteration is BF16.

The momentum and Newton-Schulz core below are adapted from KellerJordan/Muon
commit f98f1cacc0263b04290753e32be8d498c1efc806, muon.py, MIT license.
NS coefficients, normalization, transpose rule and five iterations are kept.
Execution adaptations: eager single-device execution, autocast disabled inside
NS, out-of-place Nesterov gradient mixing, skip absent grads, staged finite
validation before committing state. The author's aspect-ratio scale is
REPLACED by Moonshot Eq.(4), 0.2*sqrt(max(rows, cols)); there is no extra LR
multiplier. Finite validation performs one collective scalar check per step,
not one CPU synchronization per parameter. Diagnostics remain Tensors.

Call step ONCE after logical-batch accumulation, GradScaler.unscale_ and clip.
GradScaler skips step on overflow, so no momentum/state changes occur on a
failed AMP attempt. Direct nonfinite steps raise, retain last_failure evidence
and never switch optimizer algorithms.

MIT License

Copyright (c) 2024 Keller Jordan

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import copy
import math

import torch
from torch import nn


ARMS = ('A_adamw', 'B_muon_hybrid')
MUON_PREFIXES = ('instance_encoder.layers', 'joint_layers', 'decoder', 'pool', 'score_head.0')
MOMENTUM, NS_STEPS = .95, 5
SOURCE_COMMIT = 'f98f1cacc0263b04290753e32be8d498c1efc806'
PROVENANCE = dict(
    repository='https://github.com/KellerJordan/Muon', commit=SOURCE_COMMIT,
    source_url=f'https://github.com/KellerJordan/Muon/blob/{SOURCE_COMMIT}/muon.py',
    source_sha256='2479665a90124f62e4df557816665851ca317e42fcfda2af1da02c1f44ab5f3d',
    license='MIT', copyright='Copyright (c) 2024 Keller Jordan',
    license_url=f'https://github.com/KellerJordan/Muon/blob/{SOURCE_COMMIT}/LICENSE',
    license_sha256='8c8d17398ecce3dd3fcb1bb77b5d268a427f8250f364ac655c9b38df5b693f6b',
    update_scaling='0.2*sqrt(max(rows,cols))', scaling_reference='https://arxiv.org/abs/2502.16982',
    scaling_equation=4, momentum=MOMENTUM, nesterov=True, ns_steps=NS_STEPS,
    ns_dtype='bfloat16', parameter_and_momentum_dtype='float32', execution='eager_single_device',
    missing_gradient_rule='skip_without_state_or_decay', aspect_ratio_scale_applied=False,
    extra_lr_multiplier=1., auxiliary_optimizer='torch.optim.AdamW')


@torch.no_grad()
def zeropower_via_newtonschulz5(G, steps=NS_STEPS):
    """Author's quintic NS core; finite NS5 is not an exact polar factor."""
    if G.ndim < 2 or steps != NS_STEPS:
        raise ValueError('R89 requires matrix updates and exactly five NS iterations')
    with torch.autocast(device_type=G.device.type, enabled=False):
        a, b, c = 3.4445, -4.7750, 2.0315
        X = G.bfloat16()
        if G.size(-2) > G.size(-1):
            X = X.mT
        X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
        for _ in range(steps):
            A = X @ X.mT
            B = b * A + c * A @ A
            X = a * X + B @ X
        if G.size(-2) > G.size(-1):
            X = X.mT
        return X


@torch.no_grad()
def muon_update(grad, momentum, beta=MOMENTUM, ns_steps=NS_STEPS, nesterov=True):
    """Update a staged FP32 momentum buffer; do not mutate the input gradient."""
    if (grad.ndim != 2 or min(grad.shape) <= 1 or grad.dtype != torch.float32
            or momentum.dtype != torch.float32 or beta != MOMENTUM
            or ns_steps != NS_STEPS or nesterov is not True):
        raise ValueError('R89 uses FP32 hidden matrices, momentum=.95, Nesterov and NS5')
    momentum.lerp_(grad, 1 - beta)
    direction = grad.lerp(momentum, beta)
    direction = zeropower_via_newtonschulz5(direction, steps=ns_steps).float()
    return direction * (.2 * math.sqrt(max(grad.shape)))


def _parameter_groups(selector):
    eligible, protected = set(), set()
    for name, module in selector.named_modules():
        inside = any(name == prefix or name.startswith(prefix + '.') for prefix in MUON_PREFIXES)
        for leaf, parameter in module.named_parameters(recurse=False):
            use_muon = (inside and isinstance(module, nn.Linear) and leaf == 'weight'
                        and parameter.ndim == 2 and min(parameter.shape) > 1)
            (eligible if use_muon else protected).add(id(parameter))
    eligible -= protected
    named = list(selector.named_parameters())
    groups = []
    for algorithm in ('muon', 'adamw'):
        entries = [(name, parameter) for name, parameter in named
                   if (id(parameter) in eligible) == (algorithm == 'muon')]
        if not entries:
            raise ValueError('R89 hybrid requires both hidden Muon matrices and auxiliary AdamW tensors')
        groups.append(dict(algorithm=algorithm, params=[p for _, p in entries],
                           param_names=[name for name, _ in entries],
                           param_shapes=[list(p.shape) for _, p in entries]))
    actual = [id(p) for group in groups for p in group['params']]
    if len(actual) != len(set(actual)) or set(actual) != {id(p) for p in selector.parameters()}:
        raise ValueError('R89 parameter grouping must be exhaustive without duplicates')
    return groups


class MuonHybrid(torch.optim.Optimizer):
    """One GradScaler-compatible optimizer with an actual AdamW auxiliary path."""

    def __init__(self, selector, lr=1e-4, weight_decay=1e-4):
        groups = _parameter_groups(selector)
        if any(p.dtype != torch.float32 for group in groups for p in group['params']):
            raise ValueError('R89 hybrid parameters must remain FP32; AMP applies only to forward')
        if len({p.device for group in groups for p in group['params']}) != 1:
            raise ValueError('R89 is a single-device optimizer')
        for group in groups:
            group.update(lr=lr, weight_decay=weight_decay)
        groups[0].update(momentum=MOMENTUM, nesterov=True, ns_steps=NS_STEPS,
                         scaling=PROVENANCE['update_scaling'])
        super().__init__(groups, dict(lr=lr, weight_decay=weight_decay))
        # AdamW uses these exact group/state objects, including after resume.
        self._adamw = torch.optim.AdamW([self.param_groups[1]], lr=lr, weight_decay=weight_decay)
        self._adamw.state = self.state
        self.r89_arm = ARMS[1]
        self.completed_steps = 0
        self.last_failure = None
        prototype = self.param_groups[0]['params'][0]
        self._last_metrics = {name: prototype.new_zeros(()) for name in (
            'muon_step_applied', 'muon_direction_rms', 'muon_momentum_rms', 'muon_matrices_updated')}

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        if any(group['lr'] != self.param_groups[0]['lr']
               or group['weight_decay'] != self.param_groups[0]['weight_decay']
               for group in self.param_groups):
            raise ValueError('All R89 groups must use the same base LR and weight decay')
        pending, checks, check_names = [], [], []
        for group in self.param_groups:
            for name, parameter in zip(group['param_names'], group['params']):
                grad = parameter.grad
                if grad is None:
                    continue
                if grad.is_sparse:
                    raise RuntimeError('R89 does not support sparse gradients')
                checks.append(torch.isfinite(grad).all() & torch.isfinite(parameter).all())
                check_names.append(name)
                if group['algorithm'] == 'muon':
                    old = self.state.get(parameter, {}).get('momentum_buffer')
                    momentum = torch.zeros_like(parameter) if old is None else old.clone()
                    direction = muon_update(grad.detach(), momentum, group['momentum'],
                                            group['ns_steps'], group['nesterov'])
                    checks.append(torch.isfinite(direction).all() & torch.isfinite(momentum).all())
                    check_names.append(name + ':muon_update')
                    pending.append((parameter, group, momentum, direction))
        if checks:
            finite = torch.stack(checks)
            if not finite.all():
                self.last_failure = dict(reason='nonfinite_gradient_parameter_or_muon_update',
                                         names=check_names, finite=finite.detach().clone())
                raise FloatingPointError('R89 nonfinite step rejected before parameter/state changes; see last_failure')
        direction_rms, momentum_rms = [], []
        for parameter, group, momentum, direction in pending:
            state = self.state[parameter]
            state['momentum_buffer'] = momentum
            state['step'] = state.get('step', 0) + 1
            state['ns_calls'] = state.get('ns_calls', 0) + 1
            state['last_direction_rms'] = direction.square().mean().sqrt()
            state['last_momentum_rms'] = momentum.square().mean().sqrt()
            direction_rms.append(state['last_direction_rms'])
            momentum_rms.append(state['last_momentum_rms'])
            parameter.mul_(1 - group['lr'] * group['weight_decay'])
            parameter.add_(direction, alpha=-group['lr'])
        self._adamw.step()
        prototype = self.param_groups[0]['params'][0]
        self._last_metrics = dict(
            muon_step_applied=prototype.new_tensor(float(bool(pending))),
            muon_direction_rms=torch.stack(direction_rms).mean() if pending else prototype.new_zeros(()),
            muon_momentum_rms=torch.stack(momentum_rms).mean() if pending else prototype.new_zeros(()),
            muon_matrices_updated=prototype.new_tensor(float(len(pending))))
        self.completed_steps += 1
        return loss

    def extra_metrics(self):
        return dict(self._last_metrics)

    def state_dict(self):
        result = super().state_dict()
        result['r89'] = dict(schema=1, source_commit=SOURCE_COMMIT,
                             scaling=PROVENANCE['update_scaling'],
                             completed_steps=self.completed_steps, last_metrics=self.extra_metrics())
        return result

    def load_state_dict(self, state_dict):
        metadata = state_dict.get('r89', {})
        if (metadata.get('schema') != 1 or metadata.get('source_commit') != SOURCE_COMMIT
                or metadata.get('scaling') != PROVENANCE['update_scaling']):
            raise ValueError('Expected an R89 Muon hybrid state with the exact locked algorithm')
        saved_groups = state_dict.get('param_groups', [])
        if len(saved_groups) != len(self.param_groups):
            raise ValueError('R89 optimizer parameter groups changed')
        for current, saved in zip(self.param_groups, saved_groups):
            for key in ('algorithm', 'param_names', 'param_shapes'):
                if saved.get(key) != current[key]:
                    raise ValueError('R89 optimizer parameter identities, order or shapes changed')
        for key, value in (('momentum', MOMENTUM), ('nesterov', True), ('ns_steps', NS_STEPS),
                           ('scaling', PROVENANCE['update_scaling'])):
            if saved_groups[0].get(key) != value:
                raise ValueError('R89 Muon algorithm changed in checkpoint')
        payload = {key: value for key, value in state_dict.items() if key != 'r89'}
        super().load_state_dict(payload)
        self._adamw.param_groups = [self.param_groups[1]]
        self._adamw.state = self.state
        self.completed_steps = int(metadata['completed_steps'])
        device = self.param_groups[0]['params'][0].device
        self._last_metrics = {key: value.detach().clone().to(device=device, dtype=torch.float32)
                              for key, value in metadata['last_metrics'].items()}
        for group in self.param_groups:
            for parameter in group['params']:
                state = self.state.get(parameter, {})
                keys = ('momentum_buffer',) if group['algorithm'] == 'muon' else ('exp_avg', 'exp_avg_sq')
                for key in keys:
                    if key in state and (state[key].shape != parameter.shape or state[key].dtype != torch.float32):
                        raise ValueError('R89 optimizer matrix state is incompatible with its parameter')


def build_optimizer(selector, arm, lr=1e-4, weight_decay=1e-4):
    """Private-engine factory: no global torch monkeypatch or model changes."""
    if arm not in ARMS or not math.isfinite(lr) or lr <= 0 or not math.isfinite(weight_decay) or weight_decay < 0:
        raise ValueError('Unknown R89 arm or invalid LR/weight decay')
    if arm == 'A_adamw':
        optimizer = torch.optim.AdamW(selector.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        optimizer = MuonHybrid(selector, lr=lr, weight_decay=weight_decay)
    optimizer.r89_arm = arm
    optimizer._r89_parameter_names = {id(parameter): name for name, parameter in selector.named_parameters()}
    return optimizer


def optimizer_manifest(optimizer):
    """JSON-compatible actual group membership, shapes, counts and provenance."""
    groups = []
    for group in optimizer.param_groups:
        entries = [dict(name=optimizer._r89_parameter_names[id(p)], shape=list(p.shape),
                        numel=p.numel(), requires_grad=p.requires_grad) for p in group['params']]
        groups.append(dict(algorithm=group.get('algorithm', 'adamw'), parameters=entries,
                           tensor_count=len(entries), parameter_count=sum(entry['numel'] for entry in entries),
                           lr=group['lr'], weight_decay=group['weight_decay']))
    return dict(arm=optimizer.r89_arm, optimizer_class=type(optimizer).__module__ + '.' + type(optimizer).__name__,
                torch_version=torch.__version__, groups=groups,
                total_parameters=sum(group['parameter_count'] for group in groups),
                total_tensors=sum(group['tensor_count'] for group in groups),
                provenance=copy.deepcopy(PROVENANCE) if isinstance(optimizer, MuonHybrid) else None,
                auxiliary_optimizer_class=(type(optimizer._adamw).__module__ + '.' + type(optimizer._adamw).__name__
                                            if isinstance(optimizer, MuonHybrid) else None))


def optimizer_metrics(optimizer):
    """Per-successful-step metrics for the engine's existing 18-step averaging."""
    if isinstance(optimizer, MuonHybrid):
        return optimizer.extra_metrics()
    prototype = optimizer.param_groups[0]['params'][0]
    return {name: prototype.new_zeros(()) for name in (
        'muon_step_applied', 'muon_direction_rms', 'muon_momentum_rms', 'muon_matrices_updated')}
