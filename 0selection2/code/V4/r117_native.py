"""Frozen, input-only R117 native computations, without actions or environments.

A/B share the SAME pinned pretrained static ReLD encoder. Only the added
LEHD models, ReLD decoder and MatNet encoder/decoder differ in weight origin.
Reference currents are independent input-defined probes, NOT visited prefixes,
feasible execution states, constraint simulation or a historical cost recipe.
"""

import ast
import copy
import sys
import types
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from . import r94_donor as reld
from . import r102_donor as native


HOME = Path('/public/home/shiys')
LEHD_ROOT = HOME / 'methods\u8bba\u6587_mineru_output/source_code/NCO_code/single_objective/LEHD'
LEHD_SOURCES = dict(TSP=LEHD_ROOT / 'TSP/TSPModel.py', CVRP=LEHD_ROOT / 'CVRP/VRPModel.py')
MATNET_DECODER_SOURCE = native.MATNET_ROOT / 'neural_solvers/methods/matnet/matnet_decoder.py'
CHECKPOINTS = {
    'TSP': (LEHD_ROOT / 'TSP/result/20230509_153705_train/checkpoint-150.pt',
            'ead8dc462d9b1f6a09fa77025ac8a0c594660f3f1f02fb0d104eeffe48fffa8c',
            'model_state_dict', 24 * 1024 * 1024),
    'CVRP': (LEHD_ROOT / 'CVRP/result/20230817_235537_train/checkpoint-40.pt',
             '7a3cd8359f94c07ee5674758ef2aa834890e9a3fcd9fc5d35bd6140be1a3551b',
             'model_state_dict', 24 * 1024 * 1024),
    'VRP': (reld.DEFAULT_CHECKPOINT, reld.CHECKPOINT_SHA256,
            'model_state_dict', reld.MAX_CHECKPOINT_BYTES),
    'ATSP': native.CHECKPOINTS['ATSP'],
}
RANDOM_SEEDS = dict(TSP=1170102, CVRP=1170103, VRP=1170104, ATSP=1170105)
VARIANTS = dict(A='added_random', B='added_pretrained')
PROBLEM_NAMES = reld.PROBLEM_NAMES
SUPPORTED_PROBLEMS = PROBLEM_NAMES
ROUTING_PROBLEMS = reld.SUPPORTED_PROBLEMS
WIDTH_BY_PROBLEM = {name: 256 if name == 'ATSP' else 128 for name in PROBLEM_NAMES}
INPUT_ALLOWLIST = ('node', 'matrix', 'n', 'node_mask', 'problem_id', 'problem_name')
LEHD_PARAMS = dict(embedding_dim=128, head_num=8, qkv_dim=16,
                   ff_hidden_dim=512, decoder_layer_num=6, mode='test')
MATNET_DECODER_PARAMS = dict(head_num=16, embedding_dim=256, qkv_dim=16, logit_clipping=10.)
SPEC = dict(
    format='r117_bounded_native_computation_v1', arms=VARIANTS,
    supported_problems=list(PROBLEM_NAMES), widths=WIDTH_BY_PROBLEM,
    common_base='same pinned pretrained ReLD-MOEL static128 for CVRP and 15 MVRP',
    input_allowlist=list(INPUT_ALLOWLIST),
    input_scopes=dict(TSP=['raw xy'], CVRP=['raw xy, normalized demand, depot flag'],
        VRP=['raw xy, normalized signed demand, depot flag, L, service, TWstart, TWend'],
        ATSP=['raw directed matrix', 'fixed zero FROM and identity TO roles']),
    native_precision=copy.deepcopy(reld.SPEC['native_precision']),
    base_quantization='static base -> float16 -> float32 BEFORE ReLD reference decoder, online AND cache',
    cache_contract='forward_extra returns only native_nodes; supplied base must match instance, mask and pinned identity',
    grouping='sorted exact valid N; gather in original node order; no padded native execution; zero output padding',
    first_physical_node=0,
    tsp_reference='only node0 selected; first=last=node0; endpoint embeddings plus all six native decoder layers; no Linear_final',
    cvrp_reference='empty selected customer list; capacity=remaining_capacity=1; first=last=depot; all six native layers; no Linear_final',
    lehd_alignment='customers retain physical identities; node0 is mean of final first/last endpoint representations',
    vrp_reference=dict(currents='every physical node independently, in parallel',
        load=1., time=0., length='raw per-node L if enabled, else 0', open="float('O' in problem_name)",
        available='all valid nodes except current self; diagonal-only mask',
        extraction='native feed_forward output[0] plus its residual input, before single-head score',
        feasible_execution_claim=False, constraint_simulation=False),
    atsp_reference=dict(from_role='zeros256', to_role='first N identity256 rows', min_n=3, max_n=256,
        currents='all encoded FROM nodes', first='encoded FROM physical node0', keys_values='encoded TO nodes',
        mask='exclude current diagonal and first node0; row0 excludes node0 only once',
        extraction='native decoder.multi_head_combine output256', historical_random_roles_reproduced=False),
    frozen=True, no_actions=True, no_environments=True, no_teacher_inputs=True,
    diffusion_covered=False, all_candidate_computations_covered=False,
    training_instance_manifest='unknown for external pretrained weights; overlap uncertified',
    historical_cost_recipe_certified=False,
    numerical_warning='native FP32 batch partitioning need not be bitwise invariant')


def input_only(batch):
    """Never enumerate, read or pass through any supervision fields."""
    return {key: batch[key] for key in INPUT_ALLOWLIST if key in batch}


def _inputs(batch):
    clean = input_only(batch)
    problem = native._problem(clean)
    if problem in ROUTING_PROBLEMS:
        _, values, mask, lengths = reld._validated(clean)
    else:
        values, mask, lengths = native._native_inputs(clean, problem)
    if not mask[:, 0].all():
        raise ValueError('Physical node0 must be valid for the fixed native reference')
    if problem == 'ATSP' and (lengths < 3).any():
        raise ValueError('ATSP first/current exclusion requires at least 3 valid nodes')
    return clean, problem, values, mask, lengths


def _build_lehd(problem):
    module = native._load_file(LEHD_SOURCES[problem], '_r117_lehd_' + problem.lower())
    cls = module.TSPModel if problem == 'TSP' else module.VRPModel
    return cls(**copy.deepcopy(LEHD_PARAMS))


def _build_atsp_decoder():
    # Execute only native attention helpers, bypassing optional backbone/env imports.
    names = {'reshape_by_heads', 'multi_head_attention'}
    definitions = [item for item in ast.parse(native.MATNET_BACKBONE.read_text()).body
                   if isinstance(item, ast.FunctionDef) and item.name in names]
    if len(definitions) != len(names) or {item.name for item in definitions} != names:
        raise ValueError('Native MatNet decoder helpers changed')
    scope = dict(__name__='_r117_matnet_helpers', torch=torch, nn=nn, F=F)
    exec(compile(ast.Module(body=definitions, type_ignores=[]),
                 str(native.MATNET_BACKBONE), 'exec'), scope)
    namespace = 'EasyNCO.neural_solvers.backbones'
    stub = types.ModuleType(namespace)
    for name in names:
        setattr(stub, name, scope[name])
    with native._IMPORT_LOCK:
        sentinel = object()
        original = sys.modules.get(namespace, sentinel)
        try:
            sys.modules[namespace] = stub
            module = native._load_file(MATNET_DECODER_SOURCE, '_r117_matnet_decoder')
        finally:
            if original is sentinel:
                sys.modules.pop(namespace, None)
            else:
                sys.modules[namespace] = original
    return module.ATSPDecoder(**MATNET_DECODER_PARAMS)


def _load_added(component, modules):
    path, sha, key, limit = CHECKPOINTS[component]
    if not 0 < path.stat().st_size <= limit:
        raise ValueError(f'{component}: checkpoint exceeds bounded weight allowance')
    if native._sha256(path) != sha:
        raise ValueError(f'{component}: native checkpoint SHA256 differs from pin')
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get(key), dict):
        raise ValueError(f'{component}: missing native state container')
    for prefix, module in modules.items():
        weights = {name[len(prefix):]: value for name, value in checkpoint[key].items()
                   if name.startswith(prefix)}
        module.load_state_dict(weights, strict=True)


def _component_provenance(component, arm):
    paths = ([LEHD_SOURCES[component]] if component in LEHD_SOURCES else
             [reld.RELD_ROOT / 'models/MOEModel_Light.py', reld.RELD_ROOT / 'models/MOELayer.py']
             if component == 'VRP' else
             [native.MATNET_SOURCE, native.MATNET_BACKBONE, MATNET_DECODER_SOURCE])
    files = {str(path): native._sha256(path) for path in paths}
    path, sha, key, limit = CHECKPOINTS[component]
    pretrained = arm == 'B'
    params = (LEHD_PARAMS if component in LEHD_SOURCES else reld.SPEC['model_params']
              if component == 'VRP' else dict(encoder=native.MATNET_PARAMS, decoder=MATNET_DECODER_PARAMS))
    return dict(variant='pretrained' if pretrained else 'random',
        output_dim=256 if component == 'ATSP' else 128,
        native_source_files=files, native_source_sha256=native._digest(files),
        checkpoint_pin=dict(path=str(path), sha256=sha, state_key=key,
            prefixes=['decoder.'] if component == 'VRP' else ['encoder.', 'decoder.']),
        checkpoint_sha256=sha if pretrained else None,
        checkpoint_path=str(path) if pretrained else None,
        max_checkpoint_bytes=limit, random_seed=None if pretrained else RANDOM_SEEDS[component],
        model_params=copy.deepcopy(params),
        input_scope=copy.deepcopy(SPEC['input_scopes'][component]),
        training_instance_manifest='unknown', historical_cost_recipe_certified=False)


def _quantized_base(base, values, mask):
    if (not torch.is_tensor(base) or not base.is_floating_point()
            or base.shape != (*mask.shape, 128) or base.device != values.device):
        raise ValueError('Cached base_nodes must align with physical nodes, width128 and input device')
    base = base.detach().to(torch.float16).to(torch.float32)
    if not torch.isfinite(base).all() or base[~mask].count_nonzero():
        raise ValueError('Quantized base_nodes must be finite with zero padding')
    return base


def _tensor_output(output):
    tensor = output[0] if isinstance(output, (tuple, list)) else output
    if not torch.is_tensor(tensor):
        raise ValueError('Native hook did not return a tensor activation')
    return tensor


class NativeComputation(nn.Module):
    """``(arm, restore=False, expected_provenance=None)`` -> frozen native module.

    ``forward(batch)`` returns base_nodes[B,P,128], native_nodes[B,P,D] and
    node_mask[B,P]. D=256 for ATSP, 128 otherwise; non-routing base is zero.
    ``forward_extra(batch, base_nodes=None)`` returns only native_nodes. Passing
    the shared cached base (FP16 or FP32) skips the ReLD encoder entirely.

    Restore constructs random placeholders and checks saved identity WITHOUT
    opening external weights. The parent MUST strictly restore its complete
    state_dict before any forward; provenance describes that saved identity.
    """

    SUPPORTED_PROBLEMS = SUPPORTED_PROBLEMS
    covered_problems = SUPPORTED_PROBLEMS
    WIDTH_BY_PROBLEM = WIDTH_BY_PROBLEM
    SPEC = SPEC

    def __init__(self, arm, restore=False, expected_provenance=None):
        super().__init__()
        if arm not in VARIANTS:
            raise ValueError('NativeComputation arm must be A or B')
        if not isinstance(restore, bool):
            raise TypeError('restore must be boolean')
        if restore and expected_provenance is None:
            raise ValueError('Restore requires expected_provenance from the saved parent')
        if expected_provenance is not None and not isinstance(expected_provenance, dict):
            raise TypeError('expected_provenance must be a saved native dict')
        self.arm, self.variant = arm, VARIANTS[arm]
        with native._seeded_cpu(reld.RANDOM_SEED):
            self.base = (native._routing_restore_placeholder() if restore
                         else reld.StaticReLDDonor('pretrained')).float()
        for problem in ('TSP', 'CVRP'):
            with native._seeded_cpu(RANDOM_SEEDS[problem]):
                model = _build_lehd(problem).float()
                if arm == 'B' and not restore:
                    _load_added(problem, {'': model})
            setattr(self, problem.lower(), model)
        with native._seeded_cpu(RANDOM_SEEDS['VRP']):
            module = sys.modules[type(self.base.encoder).__module__]
            self.mvrp_decoder = module.MTL_Decoder(**copy.deepcopy(reld.SPEC['model_params'])).float()
            if arm == 'B' and not restore:
                _load_added('VRP', {'decoder.': self.mvrp_decoder})
        with native._seeded_cpu(RANDOM_SEEDS['ATSP']):
            self.atsp_encoder = native._build_atsp().float()
            self.atsp_decoder = _build_atsp_decoder().float()
            if arm == 'B' and not restore:
                _load_added('ATSP', {'encoder.': self.atsp_encoder, 'decoder.': self.atsp_decoder})
        components = {name: _component_provenance(name, arm) for name in RANDOM_SEEDS}
        components['base'] = self.base.provenance()
        self._provenance = dict(format=SPEC['format'], arm=arm, variant=self.variant,
            components=components, random_initialization_seeds=copy.deepcopy(RANDOM_SEEDS),
            external_pretraining_by_component=dict(base=True, TSP=arm == 'B', CVRP=arm == 'B',
                                                   VRP=arm == 'B', ATSP=arm == 'B'),
            reused_adapter_source_files={str(Path(module.__file__)): native._sha256(module.__file__)
                                         for module in (reld, native)},
            spec=copy.deepcopy(SPEC))
        if expected_provenance is not None and self._provenance != expected_provenance:
            raise ValueError('R117 native source/weight/reference provenance differs from saved identity')
        self.requires_grad_(False)
        self.eval()

    def provenance(self):
        return copy.deepcopy(self._provenance)

    def train(self, mode=True):
        # Also freeze every native normalization running statistic and MoE gate.
        return super().train(False)

    def _check_precision(self, values):
        if any(p.dtype != torch.float32 or p.device != values.device for p in self.parameters()):
            raise ValueError('Native computations must be FP32 on the raw input device')

    def _lehd_nodes(self, raw, problem):
        model = self.tsp if problem == 'TSP' else self.cvrp
        encoded = model.encoder(raw) if problem == 'TSP' else model.encoder(raw, capacity=1.)
        decoder = model.decoder
        if problem == 'TSP':
            indices = torch.zeros((len(raw), 2), dtype=torch.long, device=raw.device)
            endpoints = decoder._get_encoding(encoded, indices)
            first = decoder.embedding_first_node(endpoints[:, 0]).unsqueeze(1)
            last = decoder.embedding_last_node(endpoints[:, 1]).unsqueeze(1)
        else:
            endpoint = encoded[:, [0]]
            endpoint = torch.cat((endpoint, endpoint.new_ones((*endpoint.shape[:2], 1))), -1)
            first = decoder.embedding_first_node(endpoint)
            last = decoder.embedding_last_node(endpoint)
        # Exact native initial-reference assembly, without CPU-only index helpers.
        out = torch.cat((first, encoded[:, 1:], last), 1)
        for layer in decoder.layers:
            out = layer(out)
        return torch.cat(((out[:, :1] + out[:, -1:]) * .5, out[:, 1:-1]), 1)

    def _mvrp_nodes(self, raw, base, problem):
        decoder = self.mvrp_decoder
        batch_size, count = raw.shape[:2]
        attrs = raw.new_zeros((batch_size, count, 4))
        attrs[..., 0] = 1.
        if 'L' in problem:
            attrs[..., 2] = raw[..., 4]
        attrs[..., 3] = float('O' in problem)
        diagonal = torch.eye(count, device=raw.device, dtype=torch.bool)[None]
        mask = raw.new_zeros((batch_size, count, count)).masked_fill(diagonal, float('-inf'))
        captured = []

        def capture(_, inputs, output):
            hidden = _tensor_output(output)
            if hidden.shape != inputs[0].shape:
                raise ValueError('Unexpected native ReLD feed_forward tensor shape')
            captured.append(hidden + inputs[0])

        hook = decoder.feed_forward.register_forward_hook(capture)
        try:
            decoder.set_kv(base)
            probabilities, _ = decoder(base, attrs, ninf_mask=mask)
            if not torch.isfinite(probabilities).all():
                raise ValueError('Nonfinite native ReLD reference probabilities')
        finally:
            hook.remove()
            decoder.k = decoder.v = decoder.single_head_key = None
        if len(captured) != 1:
            raise ValueError('Expected one native ReLD feed_forward invocation')
        return captured[0]

    def _atsp_nodes(self, matrix):
        batch_size, count = matrix.shape[:2]
        from_role = matrix.new_zeros((batch_size, count, 256))
        to_role = torch.eye(256, device=matrix.device, dtype=torch.float32)[:count]
        to_role = to_role[None].expand(batch_size, -1, -1).contiguous()
        roles = self.atsp_encoder(from_role, to_role, matrix)
        if (not isinstance(roles, (tuple, list)) or len(roles) != 2
                or any(role.shape != (batch_size, count, 256) for role in roles)):
            raise ValueError('MatNet encoder must return native FROM, TO tensors')
        rows, columns = roles
        diagonal = torch.eye(count, device=matrix.device, dtype=torch.bool)[None]
        mask = matrix.new_zeros((batch_size, count, count)).masked_fill(diagonal, float('-inf'))
        mask[:, 1:, 0] = float('-inf')  # Row0 already excludes first=current once.
        decoder, captured = self.atsp_decoder, []

        def capture(_, inputs, output):
            captured.append(_tensor_output(output))

        hook = decoder.multi_head_combine.register_forward_hook(capture)
        try:
            decoder.set_kv(columns)
            decoder.set_q1(rows[:, :1])
            probabilities = decoder(rows, ninf_mask=mask)
            if not torch.isfinite(probabilities).all():
                raise ValueError('Nonfinite native MatNet reference probabilities')
        finally:
            hook.remove()
            decoder.k = decoder.v = decoder.single_head_key = decoder.q1 = None
        if len(captured) != 1:
            raise ValueError('Expected one native MatNet multi_head_combine invocation')
        return captured[0]

    @torch.no_grad()
    def forward_extra(self, batch, base_nodes=None):
        """Only added native_nodes, optionally reusing the parent's shared base."""
        clean, problem, values, mask, lengths = _inputs(batch)
        self.eval()
        self._check_precision(values)
        base = None
        if problem in ROUTING_PROBLEMS and problem != 'CVRP':
            if base_nodes is None:
                base_nodes = self.base(clean)
            base = _quantized_base(base_nodes, values, mask)
        output = values.new_zeros((*mask.shape, WIDTH_BY_PROBLEM[problem]), dtype=torch.float32)
        for count in torch.unique(lengths, sorted=True).tolist():
            rows = torch.nonzero(lengths == count, as_tuple=False).flatten()
            positions = torch.nonzero(mask[rows], as_tuple=False)[:, 1].reshape(len(rows), count)
            if problem == 'ATSP':
                raw = values[rows[:, None, None], positions[:, :, None], positions[:, None, :]]
            else:
                raw = values[rows[:, None], positions]
            raw = raw.float().contiguous()
            if not torch.isfinite(raw).all():
                raise ValueError('Nonfinite valid raw native input')
            with reld._native_precision(values.device.type):
                if problem in ('TSP', 'CVRP'):
                    hidden = self._lehd_nodes(raw, problem)
                elif problem == 'ATSP':
                    hidden = self._atsp_nodes(raw)
                else:
                    hidden = self._mvrp_nodes(raw, base[rows[:, None], positions].contiguous(), problem)
            if (hidden.shape != (len(rows), count, WIDTH_BY_PROBLEM[problem])
                    or hidden.dtype != torch.float32 or not torch.isfinite(hidden).all()):
                raise ValueError('Invalid node-aligned native computation output')
            output[rows[:, None], positions] = hidden.detach()
        return output.detach()

    @torch.no_grad()
    def forward(self, batch):
        clean, problem, values, mask, _ = _inputs(batch)
        self.eval()
        self._check_precision(values)
        base = (self.base(clean) if problem in ROUTING_PROBLEMS
                else values.new_zeros((*mask.shape, 128), dtype=torch.float32))
        base = _quantized_base(base, values, mask)
        return dict(base_nodes=base, native_nodes=self.forward_extra(clean, base_nodes=base),
                    node_mask=mask.detach().clone())
