"""Whole-triple FP32 means, exact global SAM and whole-update AMP recovery."""

import math

import torch

from .r64_model import classification_objective
from .r65_gradients import GradientLayout as MeanGradientLayout
from .r68_model import model_inputs
from .training_monitor import capture_rng, restore_rng


class GradientLayout(MeanGradientLayout):
    def __init__(self, model):
        super().__init__(model)
        self.names = [name for name, p in model.named_parameters() if p.requires_grad]
        self.receipt['unused_parameters'] = 'grad=None if absent from all three objectives'

    def assign_active(self, gradient, active):
        for parameter, section, used in zip(self.parameters, self.slices, active):
            parameter.grad = gradient[section].view_as(parameter).clone() if used else None


class NonfinitePass(RuntimeError):
    def __init__(self, pass_name, task):
        super().__init__(f'Nonfinite R68 {pass_name} pass at {task}')
        self.pass_name, self.task = pass_name, task


class SavedWeights:
    def __init__(self, layout):
        self.layout = layout
        self.originals = [p.detach().clone() for p in layout.parameters]

    @torch.no_grad()
    def restore(self):
        for parameter, original in zip(self.layout.parameters, self.originals):
            parameter.copy_(original)

    @torch.no_grad()
    def perturb(self, gradient, active, radius=.05):
        norm = gradient.norm()
        if not torch.isfinite(norm):
            raise NonfinitePass('first', 'global_norm')
        epsilon = gradient*(radius/norm.clamp_min(1e-12))
        squared = gradient.new_zeros(())
        for parameter, section, used, original in zip(
                self.layout.parameters, self.layout.slices, active, self.originals):
            if used:
                parameter.add_(epsilon[section].view_as(parameter))
                squared.add_((parameter-original).float().square().sum())
        effective = squared.sqrt()
        if not torch.isfinite(effective):
            raise NonfinitePass('first', 'perturbation_norm')
        return float(effective), float(epsilon.norm())


def mean_pass(model, optimizer, scaler, batches, consistency_weight, layout, pass_name, view_rows):
    gradients, values = [], []
    active = [False]*len(layout.parameters)
    scale = float(scaler.get_scale())
    for task, batch in enumerate(batches):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=batch['ind'].device.type, dtype=torch.float16,
                            enabled=scaler.is_enabled()):
            view_rows[pass_name] += len(batch['ind'])
            first = model(model_inputs(batch))
            view_rows[pass_name] += len(batch['ind'])
            second = model(model_inputs(batch))
        loss, parts = classification_objective(first, second, batch['ind'], consistency_weight)
        scaled_loss = scaler.scale(loss)
        item = {key: float(value.detach()) for key, value in dict(loss=loss, **parts).items()}
        finite_outputs = all(bool(torch.isfinite(out['logits'][out['solver_mask']]).all())
                             for out in (first, second))
        if not finite_outputs or not all(math.isfinite(v) for v in item.values()):
            raise NonfinitePass(pass_name, task)
        scaled_loss.backward()
        gradient = layout.unscaled_buffer(scale)
        if not torch.isfinite(gradient).all() or not torch.isfinite(gradient.norm()):
            raise NonfinitePass(pass_name, task)
        active = [used or p.grad is not None for used, p in zip(active, layout.parameters)]
        gradients.append(gradient)
        values.append(item)
    stacked = torch.stack(gradients)
    reduced = stacked.mean(dim=0)
    if not torch.isfinite(reduced).all() or not torch.isfinite(reduced.norm()):
        raise NonfinitePass(pass_name, 'reduction')
    return reduced, active, values, stacked.norm(dim=1).tolist()


def update_triple(model, optimizer, scaler, batches, consistency_weight, layout,
                  sam, lr, growth_successes=0):
    if len(batches) != 3 or len({b['problem_id'] for b in batches}) != 3:
        raise ValueError('R68 requires three distinct complete actual task batches')
    rng = capture_rng()
    saved = SavedWeights(layout) if sam else None
    initial_scale, failures = float(scaler.get_scale()), []
    view_rows = dict(first=0, perturbed=0)
    for retry in range(20):
        restore_rng(rng)
        scale = float(scaler.get_scale())
        try:
            first, first_active, first_values, first_norms = mean_pass(
                model, optimizer, scaler, batches, consistency_weight, layout, 'first', view_rows)
            effective = requested = 0.
            second_values = None
            gradient, active, task_norms = first, first_active, first_norms
            if sam:
                try:
                    effective, requested = saved.perturb(first, first_active)
                    # Replaying the pre-pass RNG gives both passes the same two views per task.
                    restore_rng(rng)
                    gradient, active, second_values, task_norms = mean_pass(
                        model, optimizer, scaler, batches, consistency_weight, layout, 'perturbed', view_rows)
                finally:
                    saved.restore()
            layout.assign_active(gradient, active)
            norm = float(torch.nn.utils.clip_grad_norm_(layout.parameters, 1., error_if_nonfinite=False))
            if not math.isfinite(norm):
                raise NonfinitePass('perturbed' if sam else 'first', 'clip_norm')
        except NonfinitePass as error:
            if saved is not None:
                saved.restore()
            restore_rng(rng)
            optimizer.zero_grad(set_to_none=True)
            failures.append(dict(attempt=retry+1, pass_name=error.pass_name, task=error.task, scale=scale))
            if not scaler.is_enabled():
                raise RuntimeError('Nonfinite R68 FP32 update; restored weights/RNG; no advancement') from error
            scaler.update(new_scale=scale*scaler.get_backoff_factor())
            growth_successes = 0
            print(f'[R68 AMP retry] whole same triple; pass={error.pass_name}; '
                  f'task={error.task}; scale={scaler.get_scale()}', flush=True)
            continue
        except BaseException:
            if saved is not None:
                saved.restore()
            restore_rng(rng)
            optimizer.zero_grad(set_to_none=True)
            raise
        # The three-task buffers are already unscaled. There is exactly one base step.
        for group in optimizer.param_groups:
            group['lr'] = lr
        optimizer.step()
        if scaler.is_enabled():
            growth_successes += 1
            if growth_successes == scaler.get_growth_interval():
                scale *= scaler.get_growth_factor()
                growth_successes = 0
            scaler.update(new_scale=scale)
        rows = sum(len(b['ind']) for b in batches)
        return dict(**{key: sum(v[key] for v in first_values)/3 for key in first_values[0]},
            first_pass_loss=sum(v['loss'] for v in first_values)/3,
            perturbed_loss=None if second_values is None else sum(v['loss'] for v in second_values)/3,
            perturbed_ce=None if second_values is None else sum(v['ce'] for v in second_values)/3,
            perturbed_kl=None if second_values is None else sum(v['kl'] for v in second_values)/3,
            task_losses=first_values, perturbed_task_losses=second_values,
            first_task_grad_norms=first_norms, full_task_grad_norms=task_norms,
            first_grad_norm=float(first.norm()), grad_norm=norm,
            clip_coefficient=min(1., 1./(norm+1e-6)),
            effective_perturbation_norm=effective, requested_perturbation_norm=requested,
            sam_radius=.05 if sam else 0., first_gradient_clipped=not sam,
            perturbation_restored_exactly=True, rng_replayed_for_second_pass=bool(sam),
            base_optimizer_steps=1, weight_decay_steps=1,
            successful_view_rows=rows*(4 if sam else 2), attempted_view_rows=sum(view_rows.values()),
            attempted_view_rows_by_pass=view_rows,
            inactive_parameter_names=[n for n, used in zip(layout.names, active) if not used],
            retries=retry, retry_attempts=failures, initial_scale=initial_scale,
            scale=float(scaler.get_scale()), scale_growth_successes=growth_successes,
            finite_loss=True, finite_gradient=True, reduction='FP32 arithmetic mean',
            projected=False, per_task_normalization=False)
    if saved is not None:
        saved.restore()
    restore_rng(rng)
    optimizer.zero_grad(set_to_none=True)
    raise RuntimeError('Twenty R68 AMP failures on the same triple; no rows skipped: '+repr(failures))
