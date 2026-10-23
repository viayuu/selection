"""R94 coverage extension: frozen native ELG/MatNet plus identical R94B ReLD.

Only encoder modules are constructed. No environment, decoding, labels or
historical solver recipe is used. Native definitions are executed unchanged;
the MatNet backbone import is isolated from optional torch_cluster imports.
"""

import ast
import copy
import hashlib
import importlib.util
import json
import random
import sys
import types
from contextlib import contextmanager
from pathlib import Path
from threading import RLock

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from . import r94_donor as reld


HOME = Path('/public/home/shiys')
ELG_ROOT = HOME / 'methods\u8bba\u6587_mineru_output/source_code/ELG/TSP'
MATNET_ROOT = HOME / 'EasyNCO'
ELG_SOURCE = ELG_ROOT / 'models.py'
MATNET_SOURCE = MATNET_ROOT / 'neural_solvers/methods/matnet/matnet_encoder.py'
MATNET_BACKBONE = MATNET_ROOT / 'neural_solvers/backbones/Transformer/attr_component.py'
CHECKPOINTS = {
    'TSP': (ELG_ROOT / 'weights/ELG.pt',
            '94448cce3ba413388286449886eae05f27c03f94996ff45daea37ebdce88fcdf',
            'model_state_dict', 16 * 1024 * 1024),
    'ATSP': (MATNET_ROOT / 'pretrained/matnet_pretrain/matnet_atsp100.ckpt',
             'a86d1ab1d8db62088e6a9d54118db92ebe8472c316dc046fcf755ca61280b5ca',
             'state_dict', 32 * 1024 * 1024),
}
RANDOM_SEEDS = {'TSP': 1020002, 'ATSP': 1020005}
VARIANTS = {'A': 'extra_random', 'B': 'full_pretrained'}
PROBLEM_NAMES = reld.PROBLEM_NAMES
SUPPORTED_PROBLEMS = PROBLEM_NAMES
ROUTING_PROBLEMS = reld.SUPPORTED_PROBLEMS
WIDTH_BY_PROBLEM = {name: 512 if name == 'ATSP' else 128 for name in PROBLEM_NAMES}
INPUT_ALLOWLIST = (*reld.INPUT_ALLOWLIST, 'matrix')
ELG_PARAMS = dict(embedding_dim=128, encoder_layer_num=6, head_num=8,
                  qkv_dim=16, ff_hidden_dim=512)
MATNET_PARAMS = dict(embedding_dim=256, encoder_layer_num=5, head_num=16,
                     qkv_dim=16, ms_hidden_dim=16, ff_hidden_dim=512,
                     ms_layer1_init=(1 / 2) ** .5, ms_layer2_init=(1 / 16) ** .5)
ROLE_SPEC = dict(from_role='all zeros', to_role='identity onehot columns first N of 256',
                 fixed_for='all instances, arms and datasets', max_n=256,
                 historical_random_role_recipe_reproduced=False,
                 output='concatenate FROM256 then TO256, never sum or average')
SPEC = dict(format='r102_full_type_native_donor_v1',
            supported_problems=list(SUPPORTED_PROBLEMS), widths=WIDTH_BY_PROBLEM,
            native_precision=copy.deepcopy(reld.SPEC['native_precision']),
            grouping='exact valid N; preserve original node order; padded output zero',
            tsp_anchor='node 0 remains the selector anchor; not a depot',
            atsp_roles=ROLE_SPEC, no_finetuning=True, no_hint_loss=True,
            numerical_warning='MatNet FP32 CPU batch2 vs batch1 maxabs .00135 observed; no bitwise partition invariance promised',
            scope='R94 coverage extension; R77 objective and R84 initialization unchanged')
_IMPORT_LOCK = RLock()


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _digest(record):
    return hashlib.sha256(json.dumps(record, sort_keys=True,
        separators=(',', ':'), allow_nan=False).encode('utf-8')).hexdigest()


def component_for_problem(problem):
    if isinstance(problem, (int, np.integer)) and not isinstance(problem, bool):
        if not 0 <= problem < len(PROBLEM_NAMES):
            raise ValueError('Invalid canonical problem_id')
        problem = PROBLEM_NAMES[int(problem)]
    if problem not in SUPPORTED_PROBLEMS:
        raise ValueError(f'Unsupported R102 problem {problem!r}')
    return problem if problem in ('TSP', 'ATSP') else 'VRP'


def input_only(batch):
    return {name: batch[name] for name in INPUT_ALLOWLIST if name in batch}


def _problem(batch):
    name, pid = batch.get('problem_name'), batch.get('problem_id')
    if torch.is_tensor(pid):
        if pid.numel() != 1 or pid.dtype not in (torch.int32, torch.int64):
            raise ValueError('Require a homogeneous scalar integer problem_id')
        pid = pid.item()
    if name is None:
        component_for_problem(pid)
        name = PROBLEM_NAMES[int(pid)]
    component_for_problem(name)
    if pid is not None and (isinstance(pid, bool) or not isinstance(pid, (int, np.integer))
                            or pid != PROBLEM_NAMES.index(name)):
        raise ValueError('problem_name and problem_id disagree')
    return name


@contextmanager
def _seeded_cpu(seed):
    python, numpy = random.getstate(), np.random.get_state()
    try:
        with torch.random.fork_rng(devices=[]), torch.device('cpu'):
            random.seed(seed)
            np.random.seed(seed)
            torch.random.default_generator.manual_seed(seed)
            yield
    finally:
        random.setstate(python)
        np.random.set_state(numpy)


def _load_file(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _build_tsp():
    return _load_file(ELG_SOURCE, '_r102_elg_native').TSP_Encoder(**ELG_PARAMS)


def _build_atsp():
    names = {'Normalization', 'FeedForward', 'reshape_by_heads'}
    definitions = [node for node in ast.parse(MATNET_BACKBONE.read_text()).body
                   if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names]
    if len(definitions) != len(names) or {node.name for node in definitions} != names:
        raise ValueError('Native MatNet backbone definitions changed')
    scope = dict(__name__='_r102_matnet_backbone', torch=torch, nn=nn, F=F)
    exec(compile(ast.Module(body=definitions, type_ignores=[]), str(MATNET_BACKBONE), 'exec'), scope)
    namespace = 'EasyNCO.neural_solvers.backbones'
    stub = types.ModuleType(namespace)
    for name in names:
        setattr(stub, name, scope[name])
    # The temporary module is restored even when executing native code fails.
    with _IMPORT_LOCK:
        sentinel = object()
        original = sys.modules.get(namespace, sentinel)
        try:
            sys.modules[namespace] = stub
            native = _load_file(MATNET_SOURCE, '_r102_matnet_native')
        finally:
            if original is sentinel:
                sys.modules.pop(namespace, None)
            else:
                sys.modules[namespace] = original
    return native.MatNetEncoder(**MATNET_PARAMS)


def _component_provenance(component, arm):
    source_paths = [ELG_SOURCE] if component == 'TSP' else [MATNET_SOURCE, MATNET_BACKBONE]
    files = {str(path): _sha256(path) for path in source_paths}
    path, sha, key, limit = CHECKPOINTS[component]
    pretrained = arm == 'B'
    return dict(variant='pretrained' if pretrained else 'random',
        output_dim=128 if component == 'TSP' else 512,
        supported_problems=[component], native_source_files=files,
        native_source_sha256=_digest(files),
        checkpoint_pin=dict(path=str(path), sha256=sha, state_key=key, prefix='encoder.'),
        checkpoint_sha256=sha if pretrained else None,
        checkpoint_path=str(path) if pretrained else None,
        random_seed=None if pretrained else RANDOM_SEEDS[component],
        max_checkpoint_bytes=limit,
        model_params=copy.deepcopy(ELG_PARAMS if component == 'TSP' else MATNET_PARAMS),
        input_fields=['xy'] if component == 'TSP' else ['matrix', 'fixed identity roles'],
        roles='original node order; node 0 is not a depot' if component == 'TSP' else copy.deepcopy(ROLE_SPEC),
        known_unknown_pretraining_source=dict(
            applies_to_loaded_weights=pretrained,
            known='Pinned native ELG TSP encoder' if component == 'TSP' else 'Pinned native MatNet ATSP100 encoder',
            unknown=['historical CLI', 'training instance manifest', 'exact trajectory',
                     'overlap with selector inputs', 'historical solver cost recipe'],
            historical_cost_recipe_certified=False))


def _load_encoder_weights(encoder, component):
    path, sha, key, limit = CHECKPOINTS[component]
    if not 0 < path.stat().st_size <= limit:
        raise ValueError(f'{component}: native checkpoint exceeds bounded weight allowance')
    if _sha256(path) != sha:
        raise ValueError(f'{component}: pinned native checkpoint SHA256 differs')
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get(key), dict):
        raise ValueError(f'{component}: missing pinned encoder state container')
    state = {name[len('encoder.'):]: value for name, value in checkpoint[key].items()
             if name.startswith('encoder.')}
    if not state:
        raise ValueError(f'{component}: empty native encoder state')
    encoder.load_state_dict(state, strict=True)


def _routing_restore_placeholder():
    # Same native class/layout, with NO external weight I/O. Only the parent's
    # strict state_dict restore may supply the pinned pretrained tensors here.
    donor = reld.StaticReLDDonor('random')
    provenance = donor.provenance()
    provenance.update(variant='pretrained', checkpoint_path=str(reld.DEFAULT_CHECKPOINT.resolve()),
                      checkpoint_sha256=reld.CHECKPOINT_SHA256, random_seed=None)
    provenance['known_unknown_pretraining_source']['applies_to_loaded_weights'] = True
    donor.variant, donor._provenance = 'pretrained', provenance
    return donor


def _native_inputs(batch, problem):
    key, kind = ('matrix', 'matrix') if problem == 'ATSP' else ('node', 'coord')
    values, mask, lengths = batch[key], batch['node_mask'], batch['n']
    if (batch.get('kind', kind) != kind or not torch.is_tensor(values)
            or values.ndim != 3 or not values.is_floating_point() or values.shape[0] == 0
            or (problem == 'TSP' and (values.shape[2] != 2 or batch.get('d_in', 2) != 2))
            or (problem == 'ATSP' and values.shape[1] != values.shape[2])):
        raise ValueError(f'{problem}: require the native input-only {kind} layout')
    if (not torch.is_tensor(mask) or mask.dtype != torch.bool or mask.shape != values.shape[:2]
            or mask.device != values.device or not torch.is_tensor(lengths)
            or lengths.shape != (len(values),) or lengths.dtype not in (torch.int32, torch.int64)):
        raise ValueError('Invalid native node_mask/n shape, dtype or device')
    lengths = lengths.to(device=values.device, dtype=torch.long)
    if (not torch.equal(mask.sum(1), lengths) or (lengths < 2).any()
            or (problem == 'ATSP' and (lengths > 256).any())):
        raise ValueError('Invalid valid N (native ATSP supports N<=256)')
    return values, mask, lengths


class FullTypeDonor(nn.Module):
    """Arm A/B -> frozen native features, with unchanged pretrained routing.

    Restore mode never opens external weights. The caller must immediately
    restore the complete parent state dict strictly before using the donor.
    Provenance describes the saved identity, not placeholder initialization.
    """

    SUPPORTED_PROBLEMS = SUPPORTED_PROBLEMS
    WIDTH_BY_PROBLEM = WIDTH_BY_PROBLEM
    covered_problems = SUPPORTED_PROBLEMS
    component_for_problem = staticmethod(component_for_problem)
    SPEC = SPEC

    def __init__(self, arm, *, initialize_from_checkpoint=True, expected_provenance=None):
        super().__init__()
        if arm not in VARIANTS:
            raise ValueError('R102 donor arm must be A or B')
        if not isinstance(initialize_from_checkpoint, bool):
            raise TypeError('initialize_from_checkpoint must be boolean')
        if not initialize_from_checkpoint and expected_provenance is None:
            raise ValueError('Checkpoint restore requires expected_provenance')
        if expected_provenance is not None and not isinstance(expected_provenance, dict):
            raise TypeError('expected_provenance must be a saved donor dict')
        self.arm, self.variant = arm, VARIANTS[arm]
        components = {name: _component_provenance(name, arm) for name in ('TSP', 'ATSP')}
        with _seeded_cpu(RANDOM_SEEDS['TSP']):
            self.tsp = _build_tsp().float()
            if arm == 'B' and initialize_from_checkpoint:
                _load_encoder_weights(self.tsp, 'TSP')
        with _seeded_cpu(RANDOM_SEEDS['ATSP']):
            self.atsp = _build_atsp().float()
            if arm == 'B' and initialize_from_checkpoint:
                _load_encoder_weights(self.atsp, 'ATSP')
        with _seeded_cpu(reld.RANDOM_SEED):
            self.routing = (reld.StaticReLDDonor('pretrained') if initialize_from_checkpoint
                            else _routing_restore_placeholder()).float()
        components['VRP'] = self.routing.provenance()
        self._provenance = dict(format=SPEC['format'], arm=arm, variant=self.variant,
            supported_problems=list(SUPPORTED_PROBLEMS), widths=copy.deepcopy(WIDTH_BY_PROBLEM),
            external_pretraining_by_component=dict(TSP=arm == 'B', ATSP=arm == 'B', VRP=True),
            components=components, spec=copy.deepcopy(SPEC))
        if expected_provenance is not None and self._provenance != expected_provenance:
            raise ValueError('R102 donor source/weight/role provenance differs from the saved identity')
        self.requires_grad_(False)
        self.eval()

    def provenance(self):
        return copy.deepcopy(self._provenance)

    def train(self, mode=True):
        return super().train(False)

    @torch.no_grad()
    def forward(self, batch):
        clean = input_only(batch)
        problem = _problem(clean)
        self.eval()
        if component_for_problem(problem) == 'VRP':
            return self.routing(clean)
        values, mask, lengths = _native_inputs(clean, problem)
        encoder = self.tsp if problem == 'TSP' else self.atsp
        parameter = next(encoder.parameters())
        if parameter.device != values.device or any(p.dtype != torch.float32 for p in encoder.parameters()):
            raise ValueError('Native donor must be FP32 on the input device')
        width = WIDTH_BY_PROBLEM[problem]
        output = values.new_zeros((*mask.shape, width), dtype=torch.float32)
        for count in torch.unique(lengths).tolist():
            rows = torch.nonzero(lengths == count, as_tuple=False).flatten()
            positions = torch.nonzero(mask[rows], as_tuple=False)[:, 1].reshape(len(rows), count)
            if problem == 'TSP':
                raw = values[rows[:, None], positions].float().contiguous()
            else:
                raw = values[rows[:, None, None], positions[:, :, None], positions[:, None, :]].float().contiguous()
            if not torch.isfinite(raw).all():
                raise ValueError('Nonfinite valid native input')
            with reld._native_precision(values.device.type):
                if problem == 'TSP':
                    hidden = encoder(raw)
                else:
                    from_role = raw.new_zeros((len(rows), count, 256))
                    to_role = torch.eye(256, device=raw.device, dtype=torch.float32)[:count]
                    to_role = to_role.unsqueeze(0).expand(len(rows), -1, -1).contiguous()
                    roles = encoder(from_role, to_role, raw)
                    if (not isinstance(roles, (tuple, list)) or len(roles) != 2
                            or any(role.shape != (len(rows), count, 256) for role in roles)):
                        raise ValueError('MatNet must return both native FROM and TO roles')
                    hidden = torch.cat(tuple(roles), dim=-1)
            if hidden.shape != (len(rows), count, width) or not torch.isfinite(hidden).all():
                raise ValueError('Native encoder returned invalid node-aligned features')
            output[rows[:, None], positions] = hidden.detach()
        return output.detach()
