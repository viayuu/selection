"""R73 same-state native/RDrop plus hard-teacher gradients and whole-update recovery."""

import math
from contextlib import contextmanager

import torch
import torch.nn.functional as F

from .r64_model import classification_objective
from .r65_gradients import GradientLayout
from .r67_model import model_inputs
from .r73_data import global_to_local
from .training_monitor import capture_rng, restore_rng


@contextmanager
def auxiliary_rng(successful_update, problem_id):
    outer = capture_rng()
    seed = 730003 + 18 * (successful_update - 1) + problem_id
    try:
        torch.manual_seed(seed)
        if outer['cuda']:
            torch.cuda.manual_seed_all(seed)
        yield seed
    finally:
        restore_rng(outer)


def finite_output(output):
    scores, mask = output['logits'], output['solver_mask']
    return bool(mask.any(-1).all() and torch.isfinite(scores[mask]).all())


def update_triple(model, optimizer, scaler, batches, auxiliary_batch, consistency_weight,
                  layout, successful_update, growth_successes=0):
    if len(batches) != 3 or len({batch['problem_id'] for batch in batches}) != 3:
        raise ValueError('R73 requires three distinct tasks at unchanged parameters')
    rng = capture_rng()
    if scaler.is_enabled():
        scaler.scale(torch.zeros((), device=batches[0]['ind'].device))
    initial_scale = float(scaler.get_scale())
    failures, attempted = [], dict(native_view_rows=0, auxiliary_view_rows=0,
                                  native_forward_calls=0, auxiliary_forward_calls=0)
    for retry in range(20):
        restore_rng(rng)
        gradients, task_values, active_names = [], [], set()
        scale, failed_task = float(scaler.get_scale()), None
        for task, batch in enumerate(batches):
            optimizer.zero_grad(set_to_none=True)
            rows, problem_id = len(batch['ind']), batch['problem_id']
            clean = model_inputs(batch)
            with torch.autocast(device_type=batch['ind'].device.type, dtype=torch.float16,
                                enabled=scaler.is_enabled()):
                first = model(clean)
                attempted['native_view_rows'] += rows
                attempted['native_forward_calls'] += 1
                second = model(clean)
                attempted['native_view_rows'] += rows
                attempted['native_forward_calls'] += 1
            if not finite_output(first) or not finite_output(second):
                failed_task = dict(task=task, problem_id=problem_id, phase='native_output')
                del first, second, clean
                break
            native, parts = classification_objective(first, second, batch['ind'], consistency_weight)
            native_values = {key: float(value.detach()) for key, value in parts.items()}
            native_finite = math.isfinite(float(native.detach())) and all(
                math.isfinite(value) for value in native_values.values())
            if not native_finite:
                failed_task = dict(task=task, problem_id=problem_id, phase='native_loss')
                del first, second, clean, native, parts
                break
            scaler.scale(native).backward()
            # Free both native graphs before constructing the current expansion batch.
            del first, second, clean, native, parts
            auxiliary, targets_global = auxiliary_batch(task)
            if auxiliary['problem_id'] != problem_id or len(targets_global) != rows or len(auxiliary['n']) != rows:
                raise ValueError('Native and expansion batches must have the same task and row count')
            with auxiliary_rng(successful_update, problem_id):
                with torch.autocast(device_type=batch['ind'].device.type, dtype=torch.float16,
                                    enabled=scaler.is_enabled()):
                    output = model(model_inputs(auxiliary))
                    attempted['auxiliary_view_rows'] += rows
                    attempted['auxiliary_forward_calls'] += 1
                if finite_output(output):
                    target = global_to_local(targets_global.detach(), auxiliary['pool_ids'], output['solver_mask'])
                    pseudo = F.cross_entropy(output['logits'].float(), target)
                    pseudo_value = float(pseudo.detach())
                    if math.isfinite(pseudo_value):
                        scaler.scale(pseudo).backward()
                    else:
                        failed_task = dict(task=task, problem_id=problem_id, phase='pseudo_loss')
                    del target, pseudo
                else:
                    failed_task = dict(task=task, problem_id=problem_id, phase='auxiliary_output')
            del output, auxiliary, targets_global
            if failed_task is not None:
                break
            gradient = layout.unscaled_buffer(scale)
            if not torch.isfinite(gradient).all():
                failed_task = dict(task=task, problem_id=problem_id, phase='gradient')
                del gradient
                break
            active_names.update(name for name, parameter in model.named_parameters() if parameter.grad is not None)
            gradients.append(gradient)
            task_values.append(dict(native_ce=native_values['ce'], kl=native_values['kl'],
                pseudo_ce=pseudo_value, loss=native_values['ce'] + consistency_weight * native_values['kl'] + pseudo_value))
        if failed_task is None:
            original = torch.stack(gradients)
            reduced = original.mean(0)
            if not torch.isfinite(reduced).all() or not torch.isfinite(reduced.norm()):
                failed_task = dict(phase='reduction')
        if failed_task is not None:
            optimizer.zero_grad(set_to_none=True)
            failures.append(dict(attempt=retry + 1, scale=scale, **failed_task))
            scaler.update(new_scale=scale * scaler.get_backoff_factor())
            growth_successes = 0
            print('[R73 retry] same whole triple; ' + str(failures[-1]), flush=True)
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
        rows = sum(len(batch['ind']) for batch in batches)
        return dict(**{key: sum(value[key] for value in task_values) / 3
                       for key in ('loss', 'native_ce', 'kl', 'pseudo_ce')},
            task_losses=task_values, full_task_grad_norms=original.norm(dim=1).tolist(),
            grad_norm=norm, clip_coefficient=min(1., 1. / (norm + 1e-6)),
            retries=retry, retry_attempts=failures, attempts=retry + 1,
            initial_scale=initial_scale, scale=float(scaler.get_scale()),
            scale_growth_successes=growth_successes, amp_enabled=scaler.is_enabled(),
            successful_native_view_rows=2 * rows, successful_auxiliary_view_rows=rows,
            attempted=attempted, active_gradient_names=sorted(active_names),
            optimizer_gradient_parameters=len(layout.parameters), finite_loss=True, finite_gradient=True,
            reduction='FP32 arithmetic mean of three combined native+auxiliary task gradients',
            optimizer_steps=1, native_graphs_released_before_auxiliary=True)
    restore_rng(rng)
    optimizer.zero_grad(set_to_none=True)
    error = RuntimeError('Twenty failed attempts of the same R73 whole update; no samples skipped')
    error.r73_update_receipt = dict(attempted=attempted, failures=failures,
                                    successful_update_not_advanced=successful_update)
    raise error
