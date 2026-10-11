"""One R43 update plus one observed ReLD comparison forward on MVRP."""

import torch
import torch.nn.functional as F

from code.unified_selector.registry import S2I
from .pairwise_objective import inputs_only, predicted_top3, single_objective
from .r42_experiment import symmetric_kl
from .r43_experiment import update_batch as original_update
from .training_monitor import capture_rng, restore_rng


AUXILIARY_WEIGHT = .35


def pair_objective(output, pool_ids, costs):
    costs = torch.as_tensor(costs)
    if costs.dtype != torch.float64 or costs.ndim != 2 or costs.shape[1] != 2:
        raise ValueError('Auxiliary comparisons require original FP64 [MOEL, MTL] costs')
    if not torch.isfinite(costs).all():
        raise ValueError('Auxiliary costs must be finite')
    ids = torch.as_tensor(pool_ids)
    if ids.ndim != 1 or len(ids) != 7 or not output['solver_mask'].all():
        raise ValueError('Auxiliary forward must retain the full seven-solver MVRP pool')
    positions = []
    for name in ('RELD_MOEL', 'RELD_MTL'):
        found = (ids == S2I[name]).nonzero().flatten()
        if len(found) != 1:
            raise ValueError('Missing or repeated ReLD identity in the full pool')
        positions.append(int(found[0]))
    margin = output['pair_margin'][:, positions[0], positions[1]].float()
    if len(margin) != len(costs):
        raise ValueError('Auxiliary inputs and pair costs are not aligned')
    valid = (costs[:, 0] != costs[:, 1]).to(margin.device)
    target = (costs[:, 0] < costs[:, 1]).to(device=margin.device, dtype=torch.float32)
    values = F.binary_cross_entropy_with_logits(margin, target, reduction='none')
    count = valid.sum()
    loss = values.masked_fill(~valid, 0.).sum() / count.clamp_min(1)
    correct = (((margin > 0) & (target == 1)) | ((margin < 0) & (target == 0))) & valid
    return loss, dict(extra_aux_samples=int(count), extra_aux_rows=len(costs),
                      extra_aux_correct=int(correct.sum()))


def update_batch(model, optimizer, scaler, batch, consistency_weight, extra=None):
    if extra is None:
        values = original_update(model, optimizer, scaler, batch, consistency_weight)
        return dict(values, base_loss=values['loss'], extra_aux_bce=0., extra_aux_accuracy=0.,
                    extra_aux_samples=0, extra_aux_rows=0, extra_aux_correct=0)
    clean, rng = inputs_only(batch), capture_rng()
    auxiliary, costs = extra
    auxiliary = inputs_only(auxiliary)
    if len(costs) != len(batch['ind']):
        raise ValueError('Original and auxiliary batches must have matching sizes, including tails')
    for retry in range(20):
        restore_rng(rng)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=batch['ind'].device.type, dtype=torch.float16,
                            enabled=scaler.is_enabled()):
            first, second = model(clean), model(clean)
            extra_output = model(auxiliary)
        focus = predicted_top3(first['logits'], second['logits'], batch['pool_ids'], first['solver_mask'])
        loss1, parts1 = single_objective(first, batch, focus)
        loss2, parts2 = single_objective(second, batch, focus)
        kl = symmetric_kl(first['logits'], second['logits'], first['solver_mask'])
        base = .5 * (loss1 + loss2) + consistency_weight * kl
        pair, diagnostics = pair_objective(extra_output, auxiliary['pool_ids'], costs)
        loss = base + AUXILIARY_WEIGHT * pair
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        old_scale = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        if scaler.get_scale() >= old_scale:
            return dict(loss=float(loss.detach()), base_loss=float(base.detach()), kl=float(kl.detach()),
                        **{k: float((parts1[k] + parts2[k]).detach() / 2) for k in parts1},
                        grad_norm=float(grad_norm), skipped=retry, extra_aux_bce=float(pair.detach()),
                        extra_aux_accuracy=diagnostics['extra_aux_correct'] / max(1, diagnostics['extra_aux_samples']),
                        **diagnostics)
        print(f'[AMP retry] same original+auxiliary batches/RNG; lower scale={scaler.get_scale()}', flush=True)
    raise RuntimeError('Repeated nonfinite gradients on the same R60 combined batch')
