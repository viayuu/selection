"""R70 clean plus candidate-aligned latent mixup; whole-triple AMP recovery."""

import hashlib
import math

import torch
import torch.nn.functional as F

from .r64_model import classification_objective
from .r69_gradients import GradientLayout, NonfiniteTask
from .r69_model import model_inputs
from .training_monitor import capture_rng, restore_rng


def auxiliary_seed(successful_update, problem_id):
    return 700002 + 18 * (successful_update - 1) + problem_id


def interpolate_features(features, partner, coefficient):
    return coefficient * features.float() + (1. - coefficient) * partner.float()


def mixed_cross_entropy(outputs, winner, permutation, coefficient):
    losses = []
    for output in outputs:
        mask = output['solver_mask']
        partner_winner = winner[permutation]
        if not (mask.gather(1, winner[:, None]).all()
                and mask.gather(1, partner_winner[:, None]).all()):
            raise ValueError('Both mixed targets must be legal in the aligned candidate pool')
        log_probs = F.log_softmax(output['logits'].float().masked_fill(~mask, -float('inf')), -1)
        left = log_probs.gather(1, winner[:, None]).squeeze(1)
        right = log_probs.gather(1, partner_winner[:, None]).squeeze(1)
        losses.append(-(coefficient * left + (1. - coefficient) * right).mean())
    return torch.stack(losses).mean()


def clean_views(branch, clean, amp, row_counts):
    features, outputs = [], []
    with torch.autocast(device_type=clean['node_mask'].device.type, dtype=torch.float16, enabled=amp):
        for _ in range(2):
            row_counts['encoder'] += len(clean['node_mask'])
            hidden, mask = branch.encode_pairs(clean)
            logits = branch.score_head(hidden).squeeze(-1).float().masked_fill(~mask, -float('inf'))
            features.append(hidden)
            outputs.append(dict(logits=logits, solver_mask=mask))
    return features, outputs


def mixup_views(branch, clean, features, outputs, successful_update, amp, row_counts):
    device = features[0].device
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == 'cuda' else []
    seed = auxiliary_seed(successful_update, clean['problem_id'])
    # Fork only the visible device and CPU RNG; auxiliary work cannot shift clean dropout.
    with torch.random.fork_rng(devices=devices):
        torch.random.default_generator.manual_seed(seed)
        if devices:
            torch.cuda.manual_seed(seed)
        permutation_cpu = torch.randperm(len(features[0]))
        coefficient = float(torch.rand((), dtype=torch.float32))
        permutation = permutation_cpu.to(device)
        mask = outputs[0]['solver_mask']
        if not torch.equal(mask, outputs[1]['solver_mask']) or not torch.equal(mask, mask[permutation]):
            raise ValueError('Latent mixup requires identical legal masks across the single-problem batch')
        ids = clean['pool_ids']
        if ids.ndim == 2 and not torch.equal(ids, ids[permutation]):
            raise ValueError('Latent mixup cannot permute the solver-ID alignment')
        if ids.ndim not in (1, 2) or ids.shape[-1] != mask.shape[-1]:
            raise ValueError('Candidate features must retain their original solver IDs')
        mixed = []
        for hidden in features:
            with torch.autocast(device_type=device.type, enabled=False):
                latent = interpolate_features(hidden, hidden[permutation], coefficient)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                row_counts['mixed_head'] += len(hidden)
                logits = branch.score_head(latent).squeeze(-1).float().masked_fill(~mask, -float('inf'))
            mixed.append(dict(logits=logits, solver_mask=mask))
    draw = dict(seed=seed, lambda_value=coefficient,
        permutation_sha256=hashlib.sha256(permutation_cpu.numpy().tobytes()).hexdigest(),
        shared_across_views=True, ordinary_rng_restored=True)
    return mixed, permutation, coefficient, draw


def disjoint_pass(model, optimizer, scaler, batches, consistency_weight, layout, row_counts, successful_update):
    gradients, values, norms, draws = [], [], [], []
    scale = float(scaler.get_scale())
    for position, batch in enumerate(batches):
        branch_id = model.branch_index(batch['problem_id'])
        branch = layout.branches[branch_id]
        optimizer.zero_grad(set_to_none=True)
        clean = model_inputs(batch)
        amp = scaler.is_enabled() and batch['ind'].device.type == 'cuda'
        features, outputs = clean_views(model.branches[branch_id], clean, amp, row_counts)
        mixed, permutation, coefficient, draw = mixup_views(
            model.branches[branch_id], clean, features, outputs, successful_update, amp, row_counts)
        with torch.autocast(device_type=batch['ind'].device.type, enabled=False):
            _, clean_parts = classification_objective(*outputs, batch['ind'], consistency_weight)
            mix_ce = mixed_cross_entropy(mixed, batch['ind'], permutation, coefficient)
            clean_ce, kl = clean_parts['ce'], clean_parts['kl']
            loss = .5 * clean_ce + .5 * mix_ce + consistency_weight * kl
        scaled_loss = scaler.scale(loss)
        item = {key: float(value.detach()) for key, value in
                dict(loss=loss, clean_ce=clean_ce, mix_ce=mix_ce, kl=kl).items()}
        finite_outputs = all(bool(torch.isfinite(out['logits'][out['solver_mask']]).all())
                             for out in outputs + mixed)
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
        draws.append(draw)
    optimizer.zero_grad(set_to_none=True)
    for branch_id, gradient, active in gradients:
        layout.branches[branch_id].assign_active(gradient / 3., active)
    return values, norms, gradients, draws


def update_triple(model, optimizer, scaler, batches, consistency_weight, layout, lr,
                  successful_update, growth_successes=0):
    branch_ids = [model.branch_index(batch['problem_id']) for batch in batches]
    if len(batches) != 3 or len(set(branch_ids)) != 3 or successful_update < 1:
        raise ValueError('R70 requires three distinct complete task batches and a fixed successful-update clock')
    rng = capture_rng()
    initial_scale, failures = float(scaler.get_scale()), []
    row_counts = dict(encoder=0, mixed_head=0)
    for retry in range(20):
        restore_rng(rng)
        scale = float(scaler.get_scale())
        try:
            task_values, task_norms, gradients, draws = disjoint_pass(
                model, optimizer, scaler, batches, consistency_weight, layout, row_counts, successful_update)
            norm = float(torch.nn.utils.clip_grad_norm_(layout.parameters, 1., error_if_nonfinite=False))
            if not math.isfinite(norm):
                raise NonfiniteTask('combined_clip_norm')
        except NonfiniteTask as error:
            restore_rng(rng)
            optimizer.zero_grad(set_to_none=True)
            failures.append(dict(attempt=retry + 1, task=error.task, scale=scale,
                                 branch_ids=branch_ids, pass_name='clean_plus_mixup'))
            if not scaler.is_enabled():
                raise RuntimeError('Nonfinite R70 FP32 update; RNG/gradients restored; no advancement') from error
            scaler.update(new_scale=scale * scaler.get_backoff_factor())
            growth_successes = 0
            print(f'[R70 AMP retry] whole same triple; task={error.task}; scale={scaler.get_scale()}', flush=True)
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
        rows = sum(len(batch['ind']) for batch in batches)
        return dict(**{key: sum(value[key] for value in task_values) / 3 for key in task_values[0]},
            task_losses=task_values, auxiliary_draws=draws, full_task_grad_norms=task_norms,
            divided_task_grad_norms=[value / 3 for value in task_norms],
            grad_norm=norm, clip_coefficient=min(1., 1. / (norm + 1e-6)),
            active_branch_ids=branch_ids,
            inactive_branch_ids=[index for index in range(18) if index not in branch_ids],
            active_gradient_elements=sum(sum(p.numel() for p, used in
                zip(layout.branches[index].parameters, active) if used)
                for index, _, active in gradients),
            unused_parameters_active_branches={str(index): [name for name, used in
                zip(layout.branches[index].names, active) if not used] for index, _, active in gradients},
            base_optimizer_steps=1, active_parameter_weight_decay_steps=1,
            inactive_parameter_steps=0, inactive_grad_none=True,
            successful_view_rows=2 * rows, attempted_view_rows=row_counts['encoder'],
            successful_mixed_head_rows=2 * rows, attempted_mixed_head_rows=row_counts['mixed_head'],
            retries=retry, retry_attempts=failures, initial_scale=initial_scale,
            scale=float(scaler.get_scale()), scale_growth_successes=growth_successes,
            finite_loss=True, finite_gradient=True,
            reduction='FP32 task gradient /3 in each active disjoint branch',
            global_clip_calls=1, projected=False, per_task_normalization=False)
    restore_rng(rng)
    optimizer.zero_grad(set_to_none=True)
    raise RuntimeError('Twenty R70 AMP failures on the same triple; no rows skipped: ' + repr(failures))
