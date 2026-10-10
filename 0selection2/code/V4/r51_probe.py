"""Native, exactly-ten-action ReLD probes and resumable solver states."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .multitask_probe import state_hash
from .r41_label_stability import relocated_reld_module
from .r48_common import PAIR
from .r50_solver_encoder import native_inputs
from .train import set_seed


STEPS = (1, 5, 10)
FIELDS = ('served_customer_fraction', 'served_demand_fraction', 'depot_returns',
          'open_distance_over_mean_distance', 'waiting_over_mean_travel_time',
          'remaining_capacity', 'feasible_fraction_of_unserved_customers',
          'normalized_action_entropy', 'top1_minus_top2_action_probability')
STATS = ('mean', 'std', 'q10', 'q50', 'q90')
FEATURE_NAMES = [f'{solver}/step{step}/{field}/{stat}' for solver in PAIR
                 for step in STEPS for field in FIELDS for stat in STATS]
BEHAVIOR_DIM = len(FEATURE_NAMES)


def build_solvers(provenance, device):
    module = relocated_reld_module(Path(provenance['solvers'][PAIR[0]]['original_entry']))
    utils, _, _ = module.import_reld_modules()
    set_seed(2)
    models = []
    for name, key in zip(PAIR, ('reld_moel', 'reld_mtl')):
        entry = provenance['solvers'][name]
        spec = module.MODEL_SPECS[key]
        params = module.model_init_params(spec, 'Train_ALL')
        if params != entry['runtime_params_contract']:
            raise ValueError('R48 native solver parameters changed')
        model = utils.get_model(spec['model_type'])(**dict(params, device=torch.device(device))).to(device)
        checkpoint = torch.load(entry['checkpoint']['path'], map_location=device, weights_only=False)
        if (checkpoint['epoch'], checkpoint['problem']) != (5000, 'Train_ALL'):
            raise ValueError('Wrong full solver checkpoint')
        model.load_state_dict(checkpoint['model_state_dict'], strict=True)
        model.requires_grad_(False).eval()
        models.append(model)
    return models, module


def input_length_scale(batch):
    xy = torch.cat(batch[:2], 1)
    distances = torch.cdist(xy, xy)
    n = xy.shape[1]
    return (distances.sum((1, 2)) / (n * (n - 1))).clamp_min(1e-8)


def open_segment_distance(previous_xy, selected_xy, selected):
    return (selected_xy - previous_xy).norm(dim=-1) * (selected != 0)


def action_statistics(probabilities, legal_mask):
    p = probabilities.float()
    legal = torch.isfinite(legal_mask)
    count = legal.sum(-1)
    entropy = -(p * p.clamp_min(1e-30).log()).sum(-1)
    entropy = torch.where(count > 1, entropy / count.clamp_min(2).float().log(), torch.zeros_like(entropy))
    top = p.topk(2, dim=-1).values
    return entropy, top[..., 0] - top[..., 1]


def summarize_pomo(values):
    # values: batch x all-customer POMO starts x fields. Population std is finite for one start.
    quantiles = torch.quantile(values, torch.tensor([.1, .5, .9], device=values.device), dim=1)
    return torch.stack((values.mean(1), values.std(1, correction=0), *quantiles.unbind(0)), -1)


@dataclass
class Prefix:
    env: object
    state: object
    effective_distance: torch.Tensor
    waiting_time: torch.Tensor
    depot_returns: torch.Tensor
    done: bool = False
    network_actions: int = 0


@torch.no_grad()
def initialize(model, module, batch, encode=True):
    size, count = batch[1].shape[1], batch[0].shape[0]
    env = module.get_env_class('OVRPTW')(problem_size=size, pomo_size=size, device=batch[0].device)
    env.load_problems(count, problems=batch, aug_factor=1)
    reset, _, _ = env.reset()
    if encode:
        model.pre_forward(reset)
    state, _, _ = env.pre_step()
    zeros = torch.zeros(count, size, device=batch[0].device)
    prefix = Prefix(env, state, zeros.clone(), zeros.clone(), zeros.clone())
    for forced in range(2):
        selected, _ = model(prefix.state)
        expected = torch.zeros_like(selected) if forced == 0 else prefix.state.START_NODE
        if not torch.equal(selected, expected):
            raise ValueError('Native forced depot/all-customer POMO initialization changed')
        advance(prefix, selected, count_return=False)
    return prefix


def advance(prefix, selected, count_return=True):
    env = prefix.env
    coords = env.depot_node_xy.gather(1, selected[..., None].expand(-1, -1, 2))
    distance = (coords - env.current_coord).norm(dim=-1)
    arrival = env.current_time + distance / env.speed
    tw_start = env.depot_node_tw_start.gather(1, selected)
    prefix.waiting_time += (tw_start - arrival).clamp_min(0) * (selected != 0)
    prefix.effective_distance += open_segment_distance(env.current_coord, coords, selected)
    if count_return:
        prefix.depot_returns += (selected == 0) & ~env.finished
    prefix.state, reward, done = env.step(selected)
    prefix.done = bool(done)
    return reward


def snapshot(prefix, entropy, probability_gap, length_scale):
    env = prefix.env
    served = torch.isneginf(env.visited_ninf_flag[..., 1:])
    remaining = (~served).sum(-1)
    demand = env.depot_node_demand[:, None, 1:]
    served_demand = (served * demand).sum(-1) / demand.sum(-1).clamp_min(1e-8)
    feasible = torch.isfinite(env.ninf_mask[..., 1:]) & ~served
    fraction = feasible.sum(-1) / remaining.clamp_min(1)
    scale = length_scale[:, None]
    return torch.stack((served.float().mean(-1), served_demand, prefix.depot_returns,
        prefix.effective_distance / scale, prefix.waiting_time / (scale / env.speed),
        env.load, fraction, entropy, probability_gap), -1)


@torch.no_grad()
def probe(model, module, batch, keep_witness=False):
    prefix = initialize(model, module, batch)
    scale = input_length_scale(batch)
    captured = []

    def capture(_module, _arguments, output):
        captured.append(output[0] if isinstance(output, tuple) else output)

    hook = model.decoder.register_forward_hook(capture)
    summary, raw_snapshots = [], []
    try:
        for step in range(1, 11):
            if prefix.done or bool(prefix.env.finished.any()):
                raise ValueError('A POMO trajectory completed within the fixed probe budget')
            captured.clear()
            legal = prefix.state.ninf_mask
            selected, _ = model(prefix.state)
            if len(captured) != 1:
                raise ValueError('Exactly one original decoder call is required per network action')
            probabilities = captured[0]
            if not torch.equal(selected, probabilities.argmax(-1)):
                raise ValueError('Probe no longer uses original greedy decoding')
            if step in STEPS:
                entropy, probability_gap = action_statistics(probabilities, legal)
            advance(prefix, selected)
            prefix.network_actions += 1
            if step in STEPS:
                values = snapshot(prefix, entropy, probability_gap, scale)
                summary.append(summarize_pomo(values))
                if keep_witness:
                    raw_snapshots.append(values.cpu())
    finally:
        hook.remove()
    features = torch.stack(summary, 1).flatten(1)
    if prefix.env.selected_count != 12 or prefix.network_actions != 10 or not torch.isfinite(features).all():
        raise ValueError('Invalid fixed-prefix feature output')
    witness = None
    if keep_witness:
        witness = dict(snapshots=torch.stack(raw_snapshots, 1),
            prefix_nodes=prefix.env.selected_node_list.cpu(), input_length_scale=scale.cpu())
    return features, prefix, witness


@torch.no_grad()
def finish(model, prefix, track_metrics=True):
    # Continue the live native environment AND decoder K/V, without replaying the prefix.
    start_count = prefix.env.selected_count
    reward = None
    maximum = 2 * prefix.env.problem_size + 5
    while not prefix.done:
        if prefix.env.selected_count > maximum:
            raise ValueError('Native solve exceeded the route-construction safety bound')
        selected, _ = model(prefix.state)
        if track_metrics:
            reward = advance(prefix, selected)
        else:
            # Deployment needs only the native continuation, not diagnostics after the probe.
            prefix.state, reward, done = prefix.env.step(selected)
            prefix.done = bool(done)
    if reward is None:
        reward = -prefix.env._get_travel_distance()
    distance = -reward
    # Independent incremental open-route accumulation must equal the native final objective.
    if track_metrics:
        torch.testing.assert_close(prefix.effective_distance, distance, rtol=2e-6, atol=3e-5)
    return distance.min(1).values, prefix.env.selected_count - start_count


@torch.no_grad()
def cache_behaviors(models, module, instances, sizes, progress=None):
    values = torch.empty(len(instances), BEHAVIOR_DIM)
    seen = np.zeros(len(instances), dtype=bool)
    hashes = [state_hash(model.state_dict()) for model in models]
    witnesses = []
    for size in np.unique(sizes):
        group = np.flatnonzero(sizes == size)
        for offset in range(0, len(group), 128):
            indices = group[offset:offset + 128]
            batch, _ = native_inputs(module, instances, indices.tolist(), 'cuda:0')
            parts = []
            for method, model in enumerate(models):
                features, prefix, witness = probe(model, module, batch, keep_witness=len(witnesses) < 2)
                parts.append(features)
                if witness is not None:
                    witnesses.append(dict(solver=PAIR[method], indices=indices.tolist(), **witness))
                del prefix
            values[indices] = torch.cat(parts, -1).cpu()
            seen[indices] = True
        if progress is not None:
            progress(int(seen.sum()), len(seen), int(size))
    if not seen.all() or hashes != [state_hash(model.state_dict()) for model in models]:
        raise ValueError('Missing behavior rows or changed frozen solver parameters')
    return values, witnesses
