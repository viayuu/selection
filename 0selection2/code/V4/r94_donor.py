"""Frozen, input-only ReLD-MOEL nodes for R94 early (not pooled) fusion.

Audited against unified_selector/data.py::__getitem__, r60_data._node_input,
reld_bridge_common.collate_problem_batch and MOEModel_Light.pre_forward:
raw4 = [x, y, demand, depotflag] for CVRP;
raw8 = [x, y, demand, depotflag, L, service, TWstart, TWend] for MVRP.
Demand is already in native capacity-normalized units, including its sign.
Coordinates and enabled TW endpoints retain their supplied units. No scaling,
generator defaults, depot TW, environment, decoder or solver state is used.
L/service/open flags remain on the selector raw path, outside this encoder.
"""

import copy
import hashlib
import importlib.util
import json
import random
import sys
import types
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
from torch import nn


RELD_ROOT = Path('/public/home/shiys/reld-nco-main/Multi-Task')
DEFAULT_CHECKPOINT = RELD_ROOT / 'pretrained/reld_moe_light/epoch-5000.pt'
CHECKPOINT_SHA256 = '2ce5fb49950c4c9f51d2015a125744be9960aa0c60c609cfe235e74d44ad4be3'
MAX_CHECKPOINT_BYTES = 45 * 1024 * 1024
RANDOM_SEED = 940002
OUTPUT_DIM = 128
PROBLEM_NAMES = (
    'TSP', 'CVRP', 'ATSP', 'OVRP', 'VRPB', 'VRPL', 'VRPTW', 'OVRPTW',
    'OVRPB', 'OVRPL', 'VRPBL', 'VRPBTW', 'VRPLTW', 'OVRPBL', 'OVRPBTW',
    'OVRPLTW', 'VRPBLTW', 'OVRPBLTW',
)
SUPPORTED_PROBLEMS = tuple(p for p in PROBLEM_NAMES if p not in ('TSP', 'ATSP'))
INPUT_ALLOWLIST = ('problem_id', 'problem_name', 'kind', 'd_in', 'node', 'n', 'node_mask')
SPEC = dict(
    format='r94_static_reld_moel_v1', output_dim=OUTPUT_DIM,
    native_precision=dict(dtype='float32', autocast=False, matmul_tf32=False, cudnn_tf32=False),
    supported_problems=list(SUPPORTED_PROBLEMS),
    raw_cvrp_fields=['x', 'y', 'demand', 'depotflag'],
    raw_mvrp_fields=['x', 'y', 'demand', 'depotflag', 'L', 'service', 'TWstart', 'TWend'],
    native_fields=dict(depot_xy=['x', 'y'],
                       node_xy_demand_tw=['x', 'y', 'signed_demand', 'TWstart', 'TWend']),
    disabled_tw=[0., 0.], units='unchanged native input units; demand already capacity-normalized',
    extraction='exact valid N per row; depot first; preserve customer identities',
    fusion='raw plus frozen per-node residual before trainable inter-node layers',
    model_params=dict(
        embedding_dim=128, sqrt_embedding_dim=128 ** .5, encoder_layer_num=6,
        decoder_layer_num=1, qkv_dim=16, head_num=8, logit_clipping=10,
        ff_hidden_dim=512, num_experts=4, eval_type='argmax', norm='none',
        norm_loc='norm_last', expert_loc=[*(f'Enc{i}' for i in range(6)), 'Dec'],
        problem='Train_ALL', topk=2, routing_level='node',
        routing_method='input_choice', ffidt=True),
)


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _native_sources():
    files = {name: _sha256(RELD_ROOT / 'models' / name)
             for name in ('MOEModel_Light.py', 'MOELayer.py')}
    digest = hashlib.sha256(json.dumps(files, sort_keys=True).encode('ascii')).hexdigest()
    return digest, files


@contextmanager
def _native_precision(device_type):
    # Backend flags are configuration only; reading them does not open a GPU.
    matmul = torch.backends.cuda.matmul.allow_tf32
    cudnn = torch.backends.cudnn.allow_tf32
    precision = torch.get_float32_matmul_precision()
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        with torch.autocast(device_type=device_type, enabled=False):
            yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = matmul
        torch.set_float32_matmul_precision(precision)
        torch.backends.cudnn.allow_tf32 = cudnn


def _build_encoder():
    # A private namespace avoids executing models/__init__ (other solver models)
    # or importing native utils/envs, and never changes sys.path.
    package = '_r94_reld_static_native'
    if package not in sys.modules:
        module = types.ModuleType(package)
        module.__path__ = [str(RELD_ROOT / 'models')]
        sys.modules[package] = module
    name = package + '.MOEModel_Light'
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, RELD_ROOT / 'models/MOEModel_Light.py')
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        try:
            spec.loader.exec_module(module)
        except BaseException:
            sys.modules.pop(name, None)
            raise
    return sys.modules[name].MTL_Encoder(**copy.deepcopy(SPEC['model_params']))


def input_only(batch):
    """Copy only the donor's forward allowlist; never inspect supervision."""
    return {key: batch[key] for key in INPUT_ALLOWLIST if key in batch}


def _problem(batch):
    name = batch.get('problem_name')
    pid = batch.get('problem_id')
    if name is None:
        if torch.is_tensor(pid):
            if pid.numel() != 1:
                raise ValueError('Donor requires a homogeneous, scalar problem_id')
            pid = pid.item()
        if isinstance(pid, bool) or not isinstance(pid, (int, np.integer)) or not 0 <= pid < len(PROBLEM_NAMES):
            raise ValueError('Supply a canonical problem_name or scalar problem_id')
        name = PROBLEM_NAMES[int(pid)]
    if name not in SUPPORTED_PROBLEMS:
        raise ValueError(f'Unsupported static ReLD problem {name!r}; TSP/ATSP must use their raw paths')
    if pid is not None:
        if torch.is_tensor(pid) and pid.numel() == 1:
            pid = pid.item()
        if not isinstance(pid, (int, np.integer)) or pid != PROBLEM_NAMES.index(name):
            raise ValueError('problem_name and problem_id disagree')
    return name


def _validated(batch):
    problem = _problem(batch)
    if batch.get('kind', 'coord') != 'coord':
        raise ValueError('Static ReLD requires raw coordinate nodes')
    node, mask, lengths = batch['node'], batch['node_mask'], batch['n']
    width = 4 if problem == 'CVRP' else 8
    if (not torch.is_tensor(node) or node.ndim != 3 or not node.is_floating_point()
            or node.shape[-1] != width or not node.shape[0]
            or batch.get('d_in', width) != width):
        raise ValueError(f'{problem}: audited raw{width} layout required')
    if (not torch.is_tensor(mask) or mask.dtype != torch.bool or mask.shape != node.shape[:2]
            or mask.device != node.device or not torch.is_tensor(lengths)
            or lengths.shape != (node.shape[0],) or lengths.dtype not in (torch.int32, torch.int64)):
        raise ValueError('Invalid node_mask/n shape, dtype or device')
    lengths = lengths.to(device=node.device, dtype=torch.long)
    if (not torch.equal(mask.sum(1), lengths) or (lengths < 2).any()
            or not mask[:, 0].all()):
        raise ValueError('n must count valid nodes including depot and at least one customer')
    valid = node[mask]
    if not torch.isfinite(valid).all():
        raise ValueError('Nonfinite valid raw inputs')
    if (not (node[:, 0, 3] == 1).all() or not (node[:, 0, 2] == 0).all()
            or (node[..., 3][mask].sum() != node.shape[0])
            or not ((valid[:, 3] == 0) | (valid[:, 3] == 1)).all()):
        raise ValueError('Exactly one depot at original node 0 is required')
    if width == 8:
        if not (node[:, 0, 4:] == 0).all():
            raise ValueError('Raw depot L/service/TW fields must be zero, not fabricated')
        if 'L' not in problem and not (valid[:, 4] == 0).all():
            raise ValueError('Disabled L fields must be zero')
        if 'TW' not in problem and not (valid[:, 5:] == 0).all():
            raise ValueError('Disabled service/TW fields must be 0/0/0')
        if 'TW' in problem and ((valid[:, 6] > valid[:, 7]).any() or (valid[:, 5] < 0).any()):
            raise ValueError('Invalid supplied service/TW bounds')
    return problem, node, mask, lengths


def build_inputs(batch):
    """Native keyword arguments for ONE exact-N group, without padded tokens.

    ``depot_xy`` is [B,1,2]; ``node_xy_demand_tw`` is [B,N-1,5].
    Valid nodes are gathered in original order, even for non-prefix masks.
    L/service are audited raw fields but are not native encoder arguments.
    """
    problem, node, mask, lengths = _validated(input_only(batch))
    if not (lengths == lengths[0]).all():
        raise ValueError('build_inputs requires one exact valid-N group')
    raw = node[mask].reshape(len(node), int(lengths[0]), node.shape[-1]).float()
    customers = raw[:, 1:]
    tw = customers[..., 6:8] if problem != 'CVRP' else customers.new_zeros((*customers.shape[:2], 2))
    return dict(depot_xy=raw[:, :1, :2].contiguous(),
                node_xy_demand_tw=torch.cat((customers[..., :3], tw), -1).contiguous())


class StaticReLDDonor(nn.Module):
    """``(variant, checkpoint_path=None)`` -> frozen H[B,paddedN,128].

    ``provenance()`` returns a JSON-ready defensive copy. Hashes are computed
    once at construction, never on forward. A custom pretrained path must
    contain exactly the same R48-pinned checkpoint bytes, not another weight.
    """

    SPEC = SPEC
    SUPPORTED_PROBLEMS = SUPPORTED_PROBLEMS
    covered_problems = SUPPORTED_PROBLEMS
    build_inputs = staticmethod(build_inputs)

    def __init__(self, variant, checkpoint_path=None):
        super().__init__()
        if variant not in ('random', 'pretrained'):
            raise ValueError("variant must be 'random' or 'pretrained'")
        if variant == 'random' and checkpoint_path is not None:
            raise ValueError('Random donor does not accept checkpoint weights')
        self.variant = variant
        source_sha, source_files = _native_sources()
        py_state, np_state = random.getstate(), np.random.get_state()
        try:
            with torch.random.fork_rng(devices=[]):
                random.seed(RANDOM_SEED)
                np.random.seed(RANDOM_SEED)
                # Unlike torch.manual_seed, this does not touch CUDA RNGs.
                torch.random.default_generator.manual_seed(RANDOM_SEED)
                self.encoder = _build_encoder()
        finally:
            random.setstate(py_state)
            np.random.set_state(np_state)
        checkpoint_sha, path = None, None
        if variant == 'pretrained':
            path = Path(checkpoint_path or DEFAULT_CHECKPOINT).expanduser().resolve()
            if not 0 < path.stat().st_size <= MAX_CHECKPOINT_BYTES:
                raise ValueError('Pinned checkpoint exceeds the bounded weight-load allowance')
            checkpoint_sha = _sha256(path)
            if checkpoint_sha != CHECKPOINT_SHA256:
                raise ValueError('Checkpoint SHA256 does not match the exact R48 MOEL pin')
            # Only authenticated local bytes are unpickled; never a selector.
            checkpoint = torch.load(path, map_location='cpu', weights_only=False)
            if (checkpoint.get('epoch'), checkpoint.get('problem')) != (5000, 'Train_ALL'):
                raise ValueError('Pinned checkpoint epoch/problem contract changed')
            weights = {key[len('encoder.'):]: value
                       for key, value in checkpoint['model_state_dict'].items()
                       if key.startswith('encoder.')}
            self.encoder.load_state_dict(weights, strict=True)
            del weights, checkpoint
        self._provenance = dict(
            variant=variant, checkpoint_path=None if path is None else str(path),
            checkpoint_sha256=checkpoint_sha, native_source_sha256=source_sha,
            native_source_files=source_files, output_dim=OUTPUT_DIM,
            supported_problems=list(SUPPORTED_PROBLEMS),
            random_seed=RANDOM_SEED if variant == 'random' else None,
            checkpoint_pin=dict(path=str(DEFAULT_CHECKPOINT), sha256=CHECKPOINT_SHA256),
            spec=copy.deepcopy(SPEC),
            known_unknown_pretraining_source=dict(
                applies_to_loaded_weights=variant == 'pretrained',
                known='R48 pinned ReLD-MOEL epoch5000/Train_ALL; six source training tasks',
                training_tasks=['CVRP', 'OVRP', 'VRPB', 'VRPL', 'VRPTW', 'OVRPTW'],
                unknown=['historical CLI', 'training instance manifest', 'exact trajectory',
                         'overlap with selector inputs'],
                note='16-task input coverage is not 16-task pretraining or a feasibility certificate'),
        )
        self.requires_grad_(False)
        self.eval()

    def provenance(self):
        return copy.deepcopy(self._provenance)

    def train(self, mode=True):
        # Parent model.train() must never re-enable noisy MoE gating.
        return super().train(False)

    @torch.no_grad()
    def forward(self, batch):
        clean = input_only(batch)
        _, node, mask, lengths = _validated(clean)
        self.eval()
        parameter = next(self.encoder.parameters())
        if parameter.device != node.device or parameter.dtype != torch.float32:
            raise ValueError('Donor must be FP32 on the input device (cache output may be FP16)')
        output = node.new_zeros((*node.shape[:2], OUTPUT_DIM), dtype=torch.float32)
        # Attention has no padding mask. Group BEFORE encoding, not by cropping H.
        for length in torch.unique(lengths).tolist():
            rows = torch.nonzero(lengths == length, as_tuple=False).flatten()
            group = dict(clean, node=node[rows], node_mask=mask[rows], n=lengths[rows])
            positions = torch.nonzero(group['node_mask'], as_tuple=False)[:, 1].reshape(len(rows), length)
            with _native_precision(node.device.type):
                result = self.encoder(**build_inputs(group))
            hidden = result[0] if isinstance(result, tuple) else result
            if hidden.shape != (len(rows), length, OUTPUT_DIM) or not torch.isfinite(hidden).all():
                raise ValueError('Native encoder returned invalid per-node features')
            output[rows[:, None], positions] = hidden.detach()
        return output.detach()
