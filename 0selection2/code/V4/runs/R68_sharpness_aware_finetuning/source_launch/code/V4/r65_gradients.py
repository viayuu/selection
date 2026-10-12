"""FP32 arithmetic mean of three same-state task gradients and AMP retries."""

import math

import torch

from .pairwise_objective import inputs_only
from .r64_model import classification_objective
from .training_monitor import capture_rng, restore_rng


class GradientLayout:
    def __init__(self, model):
        self.parameters = [p for p in model.parameters() if p.requires_grad]
        self.slices, offset = [], 0
        for parameter in self.parameters:
            self.slices.append(slice(offset, offset + parameter.numel()))
            offset += parameter.numel()
        self.receipt = dict(total_elements=offset, reduction='FP32 arithmetic mean',
                            projected=False, per_task_normalization=False)

    def unscaled_buffer(self, scale):
        return torch.cat([torch.zeros_like(p, dtype=torch.float32).reshape(-1) if p.grad is None
                          else p.grad.detach().float().reshape(-1) / scale for p in self.parameters])

    def assign(self, gradient):
        for parameter, section in zip(self.parameters, self.slices):
            parameter.grad = gradient[section].view_as(parameter).clone()


def update_triple(model, optimizer, scaler, batches, consistency_weight, layout,
                  growth_successes=0):
    if len(batches) != 3 or len({batch['problem_id'] for batch in batches}) != 3:
        raise ValueError('R65 requires three distinct tasks at unchanged parameters')
    rng = capture_rng()
    initial_scale, failures = float(scaler.get_scale()), []
    for retry in range(20):
        restore_rng(rng)
        gradients, task_values = [], []
        scale, failed_task = float(scaler.get_scale()), None
        # The selected batch objects are reused for every whole-triple retry.
        for task, batch in enumerate(batches):
            optimizer.zero_grad(set_to_none=True)
            clean = inputs_only(batch)
            with torch.autocast(device_type=batch['ind'].device.type, dtype=torch.float16,
                                enabled=scaler.is_enabled()):
                first, second = model(clean), model(clean)
            loss, parts = classification_objective(first, second, batch['ind'], consistency_weight)
            scaler.scale(loss).backward()
            gradient = layout.unscaled_buffer(scale)
            values = {key: float(value.detach()) for key, value in dict(loss=loss, **parts).items()}
            if not torch.isfinite(gradient).all() or not all(math.isfinite(v) for v in values.values()):
                failed_task = task
                break
            gradients.append(gradient)
            task_values.append(values)
        if failed_task is None:
            original = torch.stack(gradients)
            reduced = original.mean(dim=0)
            if not torch.isfinite(reduced).all() or not torch.isfinite(reduced.norm()):
                failed_task = 'reduction'
        if failed_task is not None:
            optimizer.zero_grad(set_to_none=True)
            if not scaler.is_enabled():
                raise RuntimeError('Nonfinite FP32 gradient; no sample or optimizer advancement')
            failures.append(dict(attempt=retry+1, task=failed_task, scale=scale))
            scaler.update(new_scale=scale * scaler.get_backoff_factor())
            growth_successes = 0
            print(f'[R65 AMP retry] whole selected triple; task={failed_task}; scale={scaler.get_scale()}', flush=True)
            continue
        layout.assign(reduced)
        norm = float(torch.nn.utils.clip_grad_norm_(layout.parameters, 1., error_if_nonfinite=True))
        # Task buffers are already unscaled exactly once. Step only after all three are finite.
        optimizer.step()
        if scaler.is_enabled():
            growth_successes += 1
            if growth_successes == scaler.get_growth_interval():
                scale *= scaler.get_growth_factor()
                growth_successes = 0
            scaler.update(new_scale=scale)
        return dict(loss=sum(v['loss'] for v in task_values)/3,
                    ce=sum(v['ce'] for v in task_values)/3,
                    kl=sum(v['kl'] for v in task_values)/3,
                    task_losses=task_values, full_task_grad_norms=original.norm(dim=1).tolist(),
                    grad_norm=norm, clip_coefficient=min(1., 1./(norm+1e-6)),
                    retries=retry, retry_attempts=failures, initial_scale=initial_scale,
                    scale=float(scaler.get_scale()), scale_growth_successes=growth_successes,
                    finite_loss=True, finite_gradient=True, reduction='FP32 arithmetic mean',
                    projected=False, per_task_normalization=False)
    restore_rng(rng)
    optimizer.zero_grad(set_to_none=True)
    raise RuntimeError('Twenty AMP overflows on the same R65 triple; no samples skipped')
