"""Task-local Eq.2 with the locked project scale and R65 whole-triple retries."""

import math

import torch

from .r64_model import classification_objective
from .r65_gradients import GradientLayout as NativeGradientLayout
from .r67_model import model_inputs
from .training_monitor import capture_rng, restore_rng


TEMPERATURE, LOSS_MULTIPLIER = .1, .1


def contrastive_masks(winner):
    labels = torch.cat((winner, winner))
    nonself = ~torch.eye(len(labels), dtype=torch.bool, device=labels.device)
    positive = labels[:, None].eq(labels[None, :]) & nonself
    return positive, nonself


def supervised_contrastive(first, second, winner):
    if first.shape != second.shape or first.ndim != 2 or len(first) != len(winner):
        raise ValueError('Eq.2 requires two complete views of the same actual task batch')
    features = torch.cat((first.float(), second.float()))
    positive, nonself = contrastive_masks(winner)
    similarity = features @ features.T / TEMPERATURE
    denominator = torch.logsumexp(similarity.masked_fill(~nonself, float('-inf')), dim=1)
    log_probability = similarity-denominator[:, None]
    positive_count = positive.sum(1)
    if not positive_count.min() >= 1:
        raise ValueError('Every anchor must retain the other view of its own input')
    raw = -(log_probability.masked_fill(~positive, 0).sum(1)/positive_count).mean()
    return LOSS_MULTIPLIER*raw, dict(supcon_raw=raw,
        positives_min=positive_count.min().float(), positives_max=positive_count.max().float(),
        contrastive_views=features.new_tensor(len(features)))


def objective(first, second, winner, consistency_weight, arm, stage):
    if stage == 1 and arm == 'B':
        if (not torch.equal(first['solver_mask'], second['solver_mask'])
                or not first['solver_mask'].gather(1, winner[:, None]).all()):
            raise ValueError('Contrastive labels must remain native legal-pool winners')
        return supervised_contrastive(first['projection'], second['projection'], winner)
    if arm not in ('A', 'B') or stage not in (1, 2):
        raise ValueError('Unknown R67 arm/stage')
    return classification_objective(first, second, winner, consistency_weight)


class GradientLayout(NativeGradientLayout):
    def __init__(self, model):
        super().__init__(model)
        self.names = [name for name, p in model.named_parameters() if p.requires_grad]
        self.receipt['unused_parameters'] = 'retain grad=None when absent from all three task objectives'

    def assign_active(self, gradient, active):
        for parameter, section, used in zip(self.parameters, self.slices, active):
            parameter.grad = gradient[section].view_as(parameter).clone() if used else None


def update_triple(model, optimizer, scaler, batches, consistency_weight, layout,
                  arm, stage, growth_successes=0):
    if len(batches) != 3 or len({batch['problem_id'] for batch in batches}) != 3:
        raise ValueError('R67 requires three distinct actual task batches at unchanged parameters')
    rng = capture_rng()
    initial_scale, failures = float(scaler.get_scale()), []
    for retry in range(20):
        restore_rng(rng)
        gradients, task_values = [], []
        active = [False]*len(layout.parameters)
        scale, failed_task = float(scaler.get_scale()), None
        for task, batch in enumerate(batches):
            optimizer.zero_grad(set_to_none=True)
            clean = model_inputs(batch)
            with torch.autocast(device_type=batch['ind'].device.type, dtype=torch.float16,
                                enabled=scaler.is_enabled()):
                first, second = model(clean), model(clean)
            loss, parts = objective(first, second, batch['ind'], consistency_weight, arm, stage)
            scaler.scale(loss).backward()
            gradient = layout.unscaled_buffer(scale)
            active = [used or p.grad is not None for used, p in zip(active, layout.parameters)]
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
                raise RuntimeError('Nonfinite FP32 triple; no rows or counters advanced')
            failures.append(dict(attempt=retry+1, task=failed_task, scale=scale))
            scaler.update(new_scale=scale*scaler.get_backoff_factor())
            growth_successes = 0
            print(f'[R67 AMP retry] whole same triple; task={failed_task}; scale={scaler.get_scale()}', flush=True)
            continue
        layout.assign_active(reduced, active)
        norm = float(torch.nn.utils.clip_grad_norm_(layout.parameters, 1., error_if_nonfinite=True))
        optimizer.step()
        if scaler.is_enabled():
            growth_successes += 1
            if growth_successes == scaler.get_growth_interval():
                scale *= scaler.get_growth_factor()
                growth_successes = 0
            scaler.update(new_scale=scale)
        return dict(**{key: sum(v[key] for v in task_values)/3 for key in task_values[0]},
            task_losses=task_values, full_task_grad_norms=original.norm(dim=1).tolist(),
            grad_norm=norm, clip_coefficient=min(1., 1./(norm+1e-6)),
            inactive_parameter_names=[name for name, used in zip(layout.names, active) if not used],
            retries=retry, retry_attempts=failures, initial_scale=initial_scale,
            scale=float(scaler.get_scale()), scale_growth_successes=growth_successes,
            finite_loss=True, finite_gradient=True, reduction='FP32 arithmetic mean',
            projected=False, per_task_normalization=False)
    restore_rng(rng)
    optimizer.zero_grad(set_to_none=True)
    raise RuntimeError('Twenty AMP overflows on the same R67 triple; no rows skipped')
