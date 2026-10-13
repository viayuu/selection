"""R71 two-view joint NLL, disjoint task gradients and whole-update recovery."""

import math

import torch

from .r69_gradients import GradientLayout, NonfiniteTask
from .r69_model import model_inputs
from .training_monitor import capture_rng, restore_rng


def disjoint_pass(model, optimizer, scaler, batches, layout, view_rows):
    gradients, values, norms = [], [], []
    scale = float(scaler.get_scale())
    for position, batch in enumerate(batches):
        branch_id = model.branch_index(batch['problem_id'])
        branch, branch_layout = model.branches[branch_id], layout.branches[branch_id]
        optimizer.zero_grad(set_to_none=True)
        amp = scaler.is_enabled() and batch['density_target'].device.type == 'cuda'
        outputs = []
        for _ in range(2):
            view_rows[0] += len(batch['density_target'])
            output, _, _ = branch.density_parameters(model_inputs(batch), amp=amp)
            outputs.append(output)
        finite_outputs = all(bool(torch.isfinite(value).all()) for output in outputs for value in output.values())
        if finite_outputs:
            losses = [branch.density_head.nll(output, batch['density_target']) for output in outputs]
            loss = torch.stack(losses).mean()
        else:
            loss = outputs[0]['mixture_logits'].sum() * float('nan')
        scaled_loss = scaler.scale(loss)
        if not finite_outputs or not bool(torch.isfinite(loss)):
            raise NonfiniteTask(position)
        scaled_loss.backward()
        gradient = branch_layout.unscaled_buffer(scale)
        norm = gradient.norm()
        if not bool(torch.isfinite(gradient).all()) or not bool(torch.isfinite(norm)):
            raise NonfiniteTask(position)
        active = [p.grad is not None for p in branch_layout.parameters]
        gradients.append((branch_id, gradient, active))
        values.append(dict(loss=float(loss.detach()), nll_per_dimension=float(loss.detach()),
            joint_nll=float(loss.detach()) * branch.density_head.dimension,
            view_nll_per_dimension=[float(value.detach()) for value in losses]))
        norms.append(float(norm))
    optimizer.zero_grad(set_to_none=True)
    for branch_id, gradient, active in gradients:
        layout.branches[branch_id].assign_active(gradient / 3., active)
    return values, norms, gradients


def update_triple(model, optimizer, scaler, batches, layout, lr, successful_update, growth_successes=0):
    branch_ids = [model.branch_index(batch['problem_id']) for batch in batches]
    if len(batches) != 3 or len(set(branch_ids)) != 3 or successful_update < 1:
        raise ValueError('R71 requires three distinct tasks and the successful-update clock')
    rng, initial_scale = capture_rng(), float(scaler.get_scale())
    failures, view_rows = [], [0]
    for retry in range(20):
        restore_rng(rng)
        scale = float(scaler.get_scale())
        try:
            values, norms, gradients = disjoint_pass(model, optimizer, scaler, batches, layout, view_rows)
            norm = float(torch.nn.utils.clip_grad_norm_(layout.parameters, 1., error_if_nonfinite=False))
            if not math.isfinite(norm):
                raise NonfiniteTask('combined_clip_norm')
        except NonfiniteTask as error:
            restore_rng(rng)
            optimizer.zero_grad(set_to_none=True)
            failures.append(dict(attempt=retry + 1, task=error.task, scale=scale, branch_ids=branch_ids,
                                 pass_name='two_view_joint_density'))
            if not scaler.is_enabled():
                raise RuntimeError('Nonfinite R71 FP32 update; RNG restored; no advancement') from error
            scaler.update(new_scale=scale * scaler.get_backoff_factor())
            growth_successes = 0
            print(f'[R71 AMP retry] whole same triple; task={error.task}; scale={scaler.get_scale()}', flush=True)
            continue
        except BaseException:
            restore_rng(rng)
            optimizer.zero_grad(set_to_none=True)
            raise
        for group in optimizer.param_groups:
            group['lr'] = lr
        optimizer.step()
        if scaler.is_enabled():
            growth_successes += 1
            if growth_successes == scaler.get_growth_interval():
                scale *= scaler.get_growth_factor()
                growth_successes = 0
            scaler.update(new_scale=scale)
        rows = sum(len(batch['density_target']) for batch in batches)
        return dict(loss=sum(value['loss'] for value in values) / 3,
            nll_per_dimension=sum(value['nll_per_dimension'] for value in values) / 3,
            joint_nll=sum(value['joint_nll'] for value in values) / 3,
            objective='mean of two joint mixture NLLs divided by task dimension; no auxiliary term',
            task_losses=values, full_task_grad_norms=norms, divided_task_grad_norms=[n / 3 for n in norms],
            grad_norm=norm, clip_coefficient=min(1., 1. / (norm + 1e-6)),
            active_branch_ids=branch_ids, inactive_branch_ids=[i for i in range(18) if i not in branch_ids],
            active_gradient_elements=sum(sum(p.numel() for p, used in zip(layout.branches[i].parameters, active) if used)
                                         for i, _, active in gradients),
            unused_parameters_active_branches={str(i): [name for name, used in zip(layout.branches[i].names, active)
                                                       if not used] for i, _, active in gradients},
            base_optimizer_steps=1, active_parameter_weight_decay_steps=1, inactive_parameter_steps=0,
            inactive_grad_none=True, successful_view_rows=2 * rows, attempted_view_rows=view_rows[0],
            retries=retry, retry_attempts=failures, initial_scale=initial_scale, scale=float(scaler.get_scale()),
            scale_growth_successes=growth_successes, finite_loss=True, finite_gradient=True,
            reduction='FP32 task gradient /3 in each active disjoint branch', global_clip_calls=1,
            projected=False, per_task_normalization=False)
    restore_rng(rng)
    optimizer.zero_grad(set_to_none=True)
    raise RuntimeError('Twenty R71 AMP failures on the same triple; no rows skipped: ' + repr(failures))
