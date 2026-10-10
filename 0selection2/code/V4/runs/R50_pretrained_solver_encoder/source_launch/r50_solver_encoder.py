"""Pinned R48 native inputs and encoder-only ReLD features, without route decoding."""

import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .multitask_probe import file_hash, state_hash
from .r41_label_stability import OVRPTW_FIELDS, relocated_reld_module
from .r48_common import PAIR
from .train import set_seed


R48_PLAN = Path('code/V4/runs/R48_core_diagnosis/audit_plan.json')
RELD_ROOT = Path('/public/home/shiys/reld-nco-main/Multi-Task')


def source_provenance():
    plan = json.loads(R48_PLAN.read_text())
    entries = plan['solvers']
    for entry in entries.values():
        if file_hash(entry['original_entry']) != entry['source_sha256']:
            raise ValueError('Pinned R48 native bridge changed')
        if file_hash(entry['checkpoint']['path']) != entry['checkpoint']['sha256']:
            raise ValueError('Pinned R48 solver weight changed')
    sources = [RELD_ROOT / 'models' / name for name in
               ('MTLModel.py', 'MOEModel_Light.py', 'MOELayer.py')]
    sources += [RELD_ROOT / name for name in ('utils.py', 'train.py', 'Trainer.py', 'envs/OVRPTWEnv.py')]
    return dict(solvers=entries, audit_plan_sha256=file_hash(R48_PLAN),
        source_files={str(path): file_hash(path) for path in sources},
        paper='https://arxiv.org/html/2503.00753v1',
        repository='https://github.com/ziweileonhuang/reld-nco',
        observed_checkpoint_metadata=dict(epoch=5000, problem='Train_ALL'),
        code_training_evidence=dict(
            entry=str(RELD_ROOT / 'Trainer.py'),
            sampling='Trainer._train_one_epoch calls env.get_random_problems for fresh online instances',
            tasks=['CVRP', 'OVRP', 'VRPB', 'VRPL', 'VRPTW', 'OVRPTW'],
            objective='POMO/REINFORCE expected route cost; MOE also uses its load-balancing loss'),
        limitations=[
            'Checkpoints contain epoch/problem and optimizer state, not a complete historical CLI or training-instance manifest.',
            'Online random generation is verified in available source, not proof of the exact historical training trajectory.',
            'No source evidence identifies these R49 holdouts as solver training data; absence of overlap cannot be certified.',
            'External solver pretraining is introduced; this is not an equal-total-pretraining-data comparison.',
            'The same current weight hashes were pinned in R48; missing historical deployment provenance is not silently reconstructed.'])


def native_inputs(module, instances, rows, device):
    if not rows or len({len(instances[int(i)]['node_xy']) for i in rows}) != 1:
        raise ValueError('Native encoders must receive a same-size batch, never padded graphs')
    for i in rows:
        if float(instances[int(i)].get('capacity', 1.)) != 1.:
            raise ValueError('OVRPTW native demand convention changed')
    batch, size = module.collate_problem_batch('OVRPTWtrain', instances, rows)
    if len(batch) != len(OVRPTW_FIELDS):
        raise ValueError('OVRPTW native input field contract changed')
    for field, value in zip(OVRPTW_FIELDS, batch):
        expected = torch.stack([torch.as_tensor(instances[int(i)][field], dtype=torch.float32) for i in rows])
        torch.testing.assert_close(value, expected, rtol=0, atol=0)
    return tuple(value.to(device) for value in batch), size


def raw_nodes(batch):
    depot, xy, demand, service, start, end = batch
    customers = torch.cat((xy, demand[..., None], service[..., None], start[..., None],
                           end[..., None], torch.zeros_like(demand[..., None])), -1)
    depot_node = torch.zeros(len(depot), 1, 7, device=depot.device, dtype=depot.dtype)
    depot_node[:, :, :2] = depot
    depot_node[:, :, -1] = 1
    return torch.cat((depot_node, customers), 1)


def encoder_arguments(batch):
    depot, xy, demand, _, start, end = batch
    return depot, torch.cat((xy, demand[..., None], start[..., None], end[..., None]), -1)


def encoder_output(encoder, batch):
    output = encoder(*encoder_arguments(batch))
    return output[0] if isinstance(output, tuple) else output


class FrozenSolverPair(nn.Module):
    def __init__(self, encoders):
        super().__init__()
        self.encoders = nn.ModuleList(encoders)
        self.requires_grad_(False)
        self.eval()

    def train(self, mode=True):
        # Frozen MoE gating must remain deterministic, including when a parent trains.
        return super().train(False)

    @torch.no_grad()
    def forward(self, batch):
        return torch.stack([encoder_output(encoder, batch) for encoder in self.encoders], 2)


def build_pair(pretrained, device, provenance, witness_instances=None):
    source = Path(provenance['solvers'][PAIR[0]]['original_entry'])
    module = relocated_reld_module(source)
    utils, _, _ = module.import_reld_modules()
    set_seed(2)
    encoders, checks, params_by_solver = [], {}, {}
    for name, key in zip(PAIR, ('reld_moel', 'reld_mtl')):
        entry = provenance['solvers'][name]
        spec = module.MODEL_SPECS[key]
        params = module.model_init_params(spec, 'Train_ALL')
        if params != entry['runtime_params_contract']:
            raise ValueError('Native model parameters no longer match R48')
        model = utils.get_model(spec['model_type'])(**dict(params, device=torch.device(device))).to(device)
        if pretrained:
            checkpoint = torch.load(entry['checkpoint']['path'], map_location=device, weights_only=False)
            if (checkpoint['epoch'], checkpoint['problem']) != (5000, 'Train_ALL'):
                raise ValueError('Unexpected pretrained checkpoint identity')
            model.load_state_dict(checkpoint['model_state_dict'], strict=True)
            del checkpoint
        model.requires_grad_(False).eval()
        if witness_instances is not None:
            rows = [0]
            batch, size = native_inputs(module, witness_instances, rows, device)
            env = module.get_env_class('OVRPTW')(problem_size=size, pomo_size=size, device=torch.device(device))
            env.load_problems(1, problems=batch, aug_factor=1)
            reset, _, _ = env.reset()
            with torch.no_grad():
                direct = encoder_output(model.encoder, batch)
                # This witness initializes decoder K/V but never selects or visits a node.
                model.pre_forward(reset)
            error = float((direct - model.encoded_nodes).abs().max())
            if error != 0 or not torch.isfinite(direct).all():
                raise ValueError('Encoder-only output differs from original pre_forward')
            if not torch.equal(reset.prob_emb.cpu(), torch.tensor([[1., 1., 0., 0., 1.]])):
                raise ValueError('Wrong native OVRPTW constraints')
            checks[name] = dict(pre_forward_max_abs_error=error, shape=list(direct.shape),
                native_fields=list(OVRPTW_FIELDS), depot_index=0, no_route_decoding=True,
                encoder_state_sha256=state_hash(model.encoder.state_dict()))
        encoders.append(model.encoder)
        params_by_solver[name] = params
    return FrozenSolverPair(encoders), module, dict(params=params_by_solver, checks=checks)


@torch.no_grad()
def cache_split(pair, module, instances, sizes, batch_size=128):
    count, maximum = len(instances), int(max(sizes)) + 1
    encoded = torch.zeros(count, maximum, 2, 128, dtype=torch.float32)
    raw = torch.zeros(count, maximum, 7, dtype=torch.float32)
    mask = torch.arange(maximum)[None] < torch.as_tensor(sizes + 1)[:, None]
    device = next(pair.parameters()).device
    seen = np.zeros(count, dtype=bool)
    for size in np.unique(sizes):
        rows = np.flatnonzero(sizes == size)
        for indices in torch.as_tensor(rows).split(batch_size):
            batch, actual = native_inputs(module, instances, indices.tolist(), device)
            values = pair(batch).cpu()
            if actual != size or values.shape != (len(indices), size + 1, 2, 128):
                raise ValueError('Native node order/shape contract changed')
            if not torch.isfinite(values).all():
                raise ValueError('Nonfinite frozen encoder representations')
            encoded[indices, :size + 1] = values
            raw[indices, :size + 1] = raw_nodes(batch).cpu()
            seen[indices] = True
    if not seen.all():
        raise ValueError('Missing cache rows')
    return dict(encoded=encoded, raw_node=raw, node_mask=mask)
