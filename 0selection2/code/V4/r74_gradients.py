"""Exact expected leaf CE, detached-weight legal KL and whole-triple recovery."""

import math

import torch
import torch.nn.functional as F

from .r65_gradients import GradientLayout
from .training_monitor import capture_rng, restore_rng


def leaf_rows(first, second, winner, mask):
    if not mask.gather(1, winner[:, None]).all():
        raise ValueError('Native target outside the original legal pool')
    log_p = F.log_softmax(first.float().masked_fill(~mask, -1e9), -1)
    log_q = F.log_softmax(second.float().masked_fill(~mask, -1e9), -1)
    ce = -.5 * (log_p.gather(1, winner[:, None]) + log_q.gather(1, winner[:, None])).squeeze(1)
    kl = (.5 * (log_p.exp()-log_q.exp()) * (log_p-log_q)).masked_fill(~mask, 0.).sum(-1)
    return ce, kl


def backward_task(model, batch, winner, consistency, scaler, attempted):
    weights, gates = model.weights(batch)
    if not torch.isfinite(weights).all() or not torch.allclose(weights.sum(-1), weights.new_ones(len(weights)), atol=1e-6):
        return None
    result = dict(loss=0., native_ce=0., kl=0., leaf_ce=[], leaf_kl=[], gate_probabilities={
        key: float(value.detach().mean()) for key, value in gates.items()})
    for leaf in range(weights.shape[1]):
        with torch.autocast(device_type=winner.device.type, dtype=torch.float16, enabled=scaler.is_enabled()):
            first = model.leaf_logits(batch, leaf)
            attempted['leaf_view_rows'] += len(winner)
            attempted['leaf_forwards'] += 1
            second = model.leaf_logits(batch, leaf)
            attempted['leaf_view_rows'] += len(winner)
            attempted['leaf_forwards'] += 1
        if not torch.isfinite(first[batch['solver_mask']]).all() or not torch.isfinite(second[batch['solver_mask']]).all():
            return None
        ce, kl = leaf_rows(first, second, winner, batch['solver_mask'])
        weighted_ce = (weights[:, leaf] * ce).mean()
        weighted_kl = (weights[:, leaf].detach() * kl).mean()
        loss = weighted_ce + consistency * weighted_kl
        if not all(torch.isfinite(value).all() for value in (ce, kl, loss)):
            return None
        scaler.scale(loss).backward(retain_graph=leaf+1 < weights.shape[1])
        result['native_ce'] += float(weighted_ce.detach())
        result['kl'] += float(weighted_kl.detach())
        result['loss'] += float(loss.detach())
        result['leaf_ce'].append(float(ce.detach().mean()))
        result['leaf_kl'].append(float(kl.detach().mean()))
        del first, second, ce, kl, weighted_ce, weighted_kl, loss
    return result


def update_triple(model, optimizer, scaler, batches, consistency, layout, growth_successes=0):
    if len(batches) != 3 or len({batch['problem_id'] for batch in batches}) != 3:
        raise ValueError('R74 uses three distinct same-state task gradients')
    rng = capture_rng()
    if scaler.is_enabled():
        scaler.scale(torch.zeros((), device=batches[0]['winner'].device))
    failures, initial_scale = [], float(scaler.get_scale())
    attempted = dict(native_rows=0, leaf_view_rows=0, leaf_forwards=0)
    names = list(model.named_parameters())
    for retry in range(20):
        restore_rng(rng)
        scale, gradients, values, active = float(scaler.get_scale()), [], [], set()
        failed = None
        for task, batch in enumerate(batches):
            optimizer.zero_grad(set_to_none=True)
            attempted['native_rows'] += len(batch['winner'])
            item = backward_task(model, batch, batch['winner'], consistency, scaler, attempted)
            if item is None:
                failed = dict(task=task, phase='output_or_loss')
                break
            gradient = layout.unscaled_buffer(scale)
            if not torch.isfinite(gradient).all():
                failed = dict(task=task, phase='gradient')
                break
            active.update(name for name, parameter in names if parameter.grad is not None)
            gradients.append(gradient)
            values.append(item)
        if failed is None:
            original = torch.stack(gradients)
            reduced = original.mean(0)
            if not torch.isfinite(reduced).all() or not math.isfinite(float(reduced.norm())):
                failed = dict(phase='reduction')
        if failed is not None:
            optimizer.zero_grad(set_to_none=True)
            failures.append(dict(attempt=retry+1, scale=scale, **failed))
            scaler.update(new_scale=scale * scaler.get_backoff_factor())
            growth_successes = 0
            print('[R74 retry] same complete triple ' + str(failures[-1]), flush=True)
            continue
        layout.assign(reduced)
        for name, parameter in names:
            if name not in active:
                parameter.grad = None
        norm = float(torch.nn.utils.clip_grad_norm_(layout.parameters, 1., error_if_nonfinite=True))
        optimizer.step()
        if scaler.is_enabled():
            growth_successes += 1
            if growth_successes == scaler.get_growth_interval():
                scale *= scaler.get_growth_factor()
                growth_successes = 0
            scaler.update(new_scale=scale)
        return dict(**{key: sum(value[key] for value in values)/3 for key in ('loss', 'native_ce', 'kl')},
            task_losses=values, full_task_grad_norms=original.norm(dim=1).tolist(), grad_norm=norm,
            clip_coefficient=min(1., 1./(norm+1e-6)), retries=retry, retry_attempts=failures,
            attempts=retry+1, initial_scale=initial_scale, scale=float(scaler.get_scale()),
            scale_growth_successes=growth_successes, attempted=attempted,
            successful_leaf_view_rows=sum(2*len(b['winner'])*(3 if b['mvrp'] else 1) for b in batches),
            active_gradient_names=sorted(active), optimizer_steps=1, finite_loss=True, finite_gradient=True)
    restore_rng(rng)
    optimizer.zero_grad(set_to_none=True)
    error = RuntimeError('Twenty failed R74 whole-triple attempts; original rows and clock unchanged')
    error.r74_update_receipt = dict(attempted=attempted, failures=failures)
    raise error
