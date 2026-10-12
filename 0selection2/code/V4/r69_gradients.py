"""Disjoint FP32 task gradients divided by three, one clip/step, whole retry."""

import math

import torch

from .r64_model import classification_objective
from .r68_gradient import GradientLayout as BranchLayout
from .r69_model import model_inputs
from .training_monitor import capture_rng, restore_rng


class GradientLayout:
    def __init__(self, model):
        self.branches = [BranchLayout(branch) for branch in model.branches]
        self.parameters = [p for p in model.parameters() if p.requires_grad]
        self.receipt = dict(total_elements=sum(p.numel() for p in self.parameters),
            branch_elements=[sum(p.numel() for p in branch.parameters) for branch in self.branches],
            reduction='each active branch FP32 task gradient divided by3 in disjoint coordinates',
            clip='one global active gradient clip at1', inactive='grad=None; no Adam state/decay advancement',
            projected=False, per_task_normalization=False)


class NonfiniteTask(RuntimeError):
    def __init__(self, task):
        super().__init__(f'Nonfinite R69 task gradient/output: {task}')
        self.task = task


def disjoint_pass(model, optimizer, scaler, batches, consistency_weight, layout, view_rows):
    gradients, values, norms = [], [], []
    scale = float(scaler.get_scale())
    for position, batch in enumerate(batches):
        branch_id = model.branch_index(batch['problem_id'])
        branch = layout.branches[branch_id]
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=batch['ind'].device.type, dtype=torch.float16,
                            enabled=scaler.is_enabled()):
            view_rows[0] += len(batch['ind'])
            first = model(model_inputs(batch))
            view_rows[0] += len(batch['ind'])
            second = model(model_inputs(batch))
        loss, parts = classification_objective(first, second, batch['ind'], consistency_weight)
        scaled_loss = scaler.scale(loss)
        item = {key: float(value.detach()) for key, value in dict(loss=loss, **parts).items()}
        finite_outputs = all(bool(torch.isfinite(out['logits'][out['solver_mask']]).all())
                             for out in (first, second))
        if not finite_outputs or not all(math.isfinite(value) for value in item.values()):
            raise NonfiniteTask(position)
        scaled_loss.backward()
        gradient = branch.unscaled_buffer(scale)
        norm = gradient.norm()
        if not bool(torch.isfinite(gradient).all()) or not bool(torch.isfinite(norm)):
            raise NonfiniteTask(position)
        active = [p.grad is not None for p in branch.parameters]
        gradients.append((branch_id, gradient, active))
        values.append(item)
        norms.append(float(norm))
    optimizer.zero_grad(set_to_none=True)
    for branch_id, gradient, active in gradients:
        layout.branches[branch_id].assign_active(gradient / 3., active)
    return values, norms, gradients


def update_triple(model, optimizer, scaler, batches, consistency_weight, layout, lr,
                  growth_successes=0):
    branch_ids = [model.branch_index(batch['problem_id']) for batch in batches]
    if len(batches) != 3 or len(set(branch_ids)) != 3:
        raise ValueError('R69 requires three distinct actual task batches at unchanged weights')
    rng = capture_rng()
    initial_scale, failures, view_rows = float(scaler.get_scale()), [], [0]
    for retry in range(20):
        restore_rng(rng)
        scale = float(scaler.get_scale())
        try:
            task_values, task_norms, gradients = disjoint_pass(
                model, optimizer, scaler, batches, consistency_weight, layout, view_rows)
            norm = float(torch.nn.utils.clip_grad_norm_(layout.parameters, 1., error_if_nonfinite=False))
            if not math.isfinite(norm):
                raise NonfiniteTask('combined_clip_norm')
        except NonfiniteTask as error:
            restore_rng(rng)
            optimizer.zero_grad(set_to_none=True)
            failures.append(dict(attempt=retry+1, task=error.task, scale=scale,
                                 branch_ids=branch_ids, pass_name='first'))
            if not scaler.is_enabled():
                raise RuntimeError('Nonfinite R69 FP32 update; RNG/gradients restored; no advancement') from error
            scaler.update(new_scale=scale*scaler.get_backoff_factor())
            growth_successes = 0
            print(f'[R69 AMP retry] whole same triple; task={error.task}; '
                  f'scale={scaler.get_scale()}', flush=True)
            continue
        except BaseException:
            restore_rng(rng)
            optimizer.zero_grad(set_to_none=True)
            raise
        # Only these three disjoint branches have gradients; AdamW skips every other branch.
        for group in optimizer.param_groups:
            group['lr'] = lr
        optimizer.step()
        if scaler.is_enabled():
            growth_successes += 1
            if growth_successes == scaler.get_growth_interval():
                scale *= scaler.get_growth_factor()
                growth_successes = 0
            scaler.update(new_scale=scale)
        rows = sum(len(batch['ind']) for batch in batches)
        return dict(**{key: sum(value[key] for value in task_values)/3 for key in task_values[0]},
            task_losses=task_values, full_task_grad_norms=task_norms,
            divided_task_grad_norms=[value/3 for value in task_norms],
            grad_norm=norm, clip_coefficient=min(1., 1./(norm+1e-6)),
            active_branch_ids=branch_ids,
            inactive_branch_ids=[index for index in range(18) if index not in branch_ids],
            active_gradient_elements=sum(sum(p.numel() for p, used in
                zip(layout.branches[index].parameters, active) if used)
                for index, _, active in gradients),
            unused_parameters_active_branches={str(index): [name for name, used in
                zip(layout.branches[index].names, active) if not used] for index, _, active in gradients},
            base_optimizer_steps=1, active_parameter_weight_decay_steps=1,
            inactive_parameter_steps=0, inactive_grad_none=True,
            successful_view_rows=2*rows, attempted_view_rows=view_rows[0],
            retries=retry, retry_attempts=failures, initial_scale=initial_scale,
            scale=float(scaler.get_scale()), scale_growth_successes=growth_successes,
            finite_loss=True, finite_gradient=True,
            reduction='FP32 task gradient /3 in each active disjoint branch',
            global_clip_calls=1, projected=False, per_task_normalization=False)
    restore_rng(rng)
    optimizer.zero_grad(set_to_none=True)
    raise RuntimeError('Twenty R69 AMP failures on the same triple; no rows skipped: '+repr(failures))
