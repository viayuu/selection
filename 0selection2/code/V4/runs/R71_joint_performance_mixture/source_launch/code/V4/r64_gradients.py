"""Three same-state task gradients, selective PCGrad, and whole-triple AMP retries."""

import math

import torch

from .pairwise_objective import inputs_only
from .r64_model import classification_objective
from .training_monitor import capture_rng, restore_rng


SHARED_PREFIXES = ('instance_encoder.layers', 'solver_encoder.feature_mlp',
                   'solver_encoder.norm', 'joint_layers', 'pool', 'decoder', 'score_head')


class GradientLayout:
    def __init__(self, model):
        self.named = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
        self.parameters = [p for _, p in self.named]
        self.slices, matches, offset = [], {prefix: [] for prefix in SHARED_PREFIXES}, 0
        shared = []
        for name, p in self.named:
            self.slices.append(slice(offset, offset + p.numel()))
            matched = [prefix for prefix in SHARED_PREFIXES if name.startswith(prefix + '.')]
            if len(matched) > 1:
                raise ValueError('Overlapping R64 projection prefixes: ' + name)
            if matched:
                matches[matched[0]].append(name)
            shared.append(torch.full((p.numel(),), bool(matched), dtype=torch.bool))
            offset += p.numel()
        if any(not names for names in matches.values()):
            raise ValueError('Empty R64 projection prefix: ' + repr(matches))
        self.shared = torch.cat(shared).to(self.parameters[0].device)
        if self.shared.all():
            raise ValueError('R64 embeddings and task/input-specific parameters must remain outside projection')
        self.receipt = dict(prefixes=list(SHARED_PREFIXES), matched_parameters=matches,
                            ordinary_mean_parameters=[n for n, _ in self.named
                                if not any(n.startswith(p + '.') for p in SHARED_PREFIXES)],
                            total_elements=offset, projected_elements=int(self.shared.sum()))

    def unscaled_buffer(self, scale):
        # Parameter grads are scaled FP32. Each task is divided exactly once here.
        return torch.cat([torch.zeros_like(p, dtype=torch.float32).reshape(-1) if p.grad is None
                          else p.grad.detach().float().reshape(-1) / scale for p in self.parameters])

    def assign(self, gradient):
        for p, section in zip(self.parameters, self.slices):
            p.grad = gradient[section].view_as(p).clone()


def project_shared(original, rng, enabled=True):
    if original.dtype != torch.float32 or original.ndim != 2 or original.shape[0] != 3:
        raise ValueError('R64 projection requires three unscaled FP32 task vectors')
    if not torch.isfinite(original).all():
        raise ValueError('Projection cannot receive nonfinite gradients')
    gram = original @ original.T
    norms = original.norm(dim=1)
    projected, orders = original.clone(), []
    for i in range(3):
        order = [j for j in range(3) if j != i]
        rng.shuffle(order)
        orders.append(order)
        if enabled:
            for j in order:
                denominator = gram[j, j]
                if float(denominator) == 0.:
                    continue
                dot = torch.dot(projected[i], original[j])
                projected[i].sub_(torch.minimum(dot, dot.new_zeros(())) / denominator * original[j])
    pairs = [(0, 1), (0, 2), (1, 2)]
    comparable = [float(norms[i] * norms[j]) > 0. for i, j in pairs]
    dots = [float(gram[i, j]) for i, j in pairs]
    cosines = [dots[k] / float(norms[i] * norms[j]) if comparable[k] else 0.
               for k, (i, j) in enumerate(pairs)]
    change = projected - original
    mean_change = float(change.mean(0).norm())
    mean_norm = float(original.mean(0).norm())
    return projected.mean(0), dict(
        shared_task_grad_norms=norms.tolist(), original_pair_dots=dots,
        original_pair_cosines=cosines, comparable_pairs=sum(comparable),
        conflict_pairs=sum(dot < 0 and valid for dot, valid in zip(dots, comparable)),
        projection_orders=orders, projection_task_change_norms=change.norm(dim=1).tolist(),
        projection_mean_change_norm=mean_change,
        projection_relative_mean_change=mean_change / max(mean_norm, 1e-30))


def reduce_gradients(original, shared, rng, projected):
    reduced = original.mean(0)
    core, diagnostics = project_shared(original[:, shared], rng, projected)
    reduced[shared] = core
    diagnostics['full_task_grad_norms'] = original.norm(dim=1).tolist()
    return reduced, diagnostics


def update_triple(model, optimizer, scaler, batches, consistency_weight, layout,
                  projection_rng, projected, growth_successes=0):
    if len(batches) != 3 or len({b['problem_id'] for b in batches}) != 3:
        raise ValueError('Each R64 update must contain three distinct tasks')
    rng, projection_state = capture_rng(), projection_rng.getstate()
    initial_scale, failures = float(scaler.get_scale()), []
    for retry in range(20):
        restore_rng(rng)
        projection_rng.setstate(projection_state)
        task_gradients, task_values = [], []
        scale, failed_task = float(scaler.get_scale()), None
        for task, batch in enumerate(batches):
            optimizer.zero_grad(set_to_none=True)
            clean = inputs_only(batch)
            with torch.autocast(device_type=batch['ind'].device.type, dtype=torch.float16,
                                enabled=scaler.is_enabled()):
                first, second = model(clean), model(clean)
            loss, parts = classification_objective(first, second, batch['ind'], consistency_weight)
            scaler.scale(loss).backward()
            gradient = layout.unscaled_buffer(scale)
            values = {k: float(v.detach()) for k, v in dict(loss=loss, **parts).items()}
            if not torch.isfinite(gradient).all() or not all(math.isfinite(v) for v in values.values()):
                failed_task = task
                break
            task_gradients.append(gradient)
            task_values.append(values)
        if failed_task is None:
            reduced, diagnostics = reduce_gradients(torch.stack(task_gradients), layout.shared,
                                                    projection_rng, projected)
            if not torch.isfinite(reduced).all() or not torch.isfinite(reduced.norm()):
                failed_task = 'reduction'
        if failed_task is not None:
            optimizer.zero_grad(set_to_none=True)
            if not scaler.is_enabled():
                raise RuntimeError('Nonfinite R64 FP32 gradient; no optimizer/sample advancement')
            failures.append(dict(attempt=retry + 1, task=failed_task, scale=scale))
            scaler.update(new_scale=scale * scaler.get_backoff_factor())
            growth_successes = 0
            print(f'[R64 AMP retry] whole triple; task={failed_task}; scale={scaler.get_scale()}', flush=True)
            continue
        layout.assign(reduced)
        norm = float(torch.nn.utils.clip_grad_norm_(layout.parameters, 1., error_if_nonfinite=True))
        # Buffers already are unscaled: never call scaler.unscale_ or scaler.step.
        optimizer.step()
        if scaler.is_enabled():
            growth_successes += 1
            if growth_successes == scaler.get_growth_interval():
                scale *= scaler.get_growth_factor()
                growth_successes = 0
            scaler.update(new_scale=scale)
        return dict(loss=sum(v['loss'] for v in task_values) / 3,
                    ce=sum(v['ce'] for v in task_values) / 3,
                    kl=sum(v['kl'] for v in task_values) / 3,
                    task_losses=task_values, grad_norm=norm,
                    clip_coefficient=min(1., 1. / (norm + 1e-6)),
                    retries=retry, retry_attempts=failures, initial_scale=initial_scale,
                    scale=float(scaler.get_scale()), scale_growth_successes=growth_successes,
                    finite_gradient=True, **diagnostics)
    restore_rng(rng)
    projection_rng.setstate(projection_state)
    optimizer.zero_grad(set_to_none=True)
    raise RuntimeError('Twenty AMP overflows on the same R64 triple; no samples skipped')
