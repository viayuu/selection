"""R65 updates, with actual-task means for the new dataset's partial groups."""

import math

import torch

from .pairwise_objective import inputs_only
from .r64_model import classification_objective
from .r65_gradients import GradientLayout, update_triple
from .training_monitor import capture_rng, restore_rng


def update_group(model, optimizer, scaler, batches, consistency_weight, layout, growth_successes=0):
    if len(batches) == 3:
        return update_triple(model, optimizer, scaler, batches, consistency_weight, layout, growth_successes)
    count = len(batches)
    if count not in (1, 2) or len({b['problem_id'] for b in batches}) != count:
        raise ValueError('A partial group must contain one or two distinct real tasks')
    rng = capture_rng()
    initial_scale, failures = float(scaler.get_scale()), []
    for retry in range(20):
        restore_rng(rng)
        gradients, task_values = [], []
        scale, failed_task = float(scaler.get_scale()), None
        for task, batch in enumerate(batches):
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=torch.float16, enabled=scaler.is_enabled()):
                first, second = model(inputs_only(batch)), model(inputs_only(batch))
            loss, parts = classification_objective(first, second, batch['ind'], consistency_weight)
            scaler.scale(loss).backward()
            gradient = layout.unscaled_buffer(scale)
            values = {k: float(v.detach()) for k, v in dict(loss=loss, **parts).items()}
            if not torch.isfinite(gradient).all() or not all(math.isfinite(v) for v in values.values()):
                failed_task = task
                break
            gradients.append(gradient)
            task_values.append(values)
        if failed_task is None:
            original = torch.stack(gradients)
            reduced = original.mean(0)
            if not torch.isfinite(reduced).all() or not torch.isfinite(reduced.norm()):
                failed_task = 'reduction'
        if failed_task is not None:
            optimizer.zero_grad(set_to_none=True)
            if not scaler.is_enabled():
                raise RuntimeError('Nonfinite FP32 partial-group update')
            failures.append(dict(attempt=retry+1, task=failed_task, scale=scale))
            scaler.update(new_scale=scale*scaler.get_backoff_factor())
            growth_successes = 0
            print('[R75 AMP retry] partial group, same rows/RNG, scale=' + str(scaler.get_scale()), flush=True)
            continue
        layout.assign(reduced)
        norm = float(torch.nn.utils.clip_grad_norm_(layout.parameters, 1., error_if_nonfinite=True))
        optimizer.step()
        if scaler.is_enabled():
            growth_successes += 1
            if growth_successes == scaler.get_growth_interval():
                scale *= scaler.get_growth_factor()
                growth_successes = 0
            scaler.update(new_scale=scale)
        return dict(loss=sum(v['loss'] for v in task_values)/count,
            ce=sum(v['ce'] for v in task_values)/count, kl=sum(v['kl'] for v in task_values)/count,
            task_losses=task_values, full_task_grad_norms=original.norm(dim=1).tolist(),
            grad_norm=norm, clip_coefficient=min(1., 1./(norm+1e-6)), retries=retry,
            retry_attempts=failures, initial_scale=initial_scale, scale=float(scaler.get_scale()),
            scale_growth_successes=growth_successes, finite_loss=True, finite_gradient=True,
            reduction='FP32 arithmetic mean of actual tasks', projected=False, per_task_normalization=False)
    restore_rng(rng)
    optimizer.zero_grad(set_to_none=True)
    raise RuntimeError('Twenty overflows on the same partial group; no rows skipped')
