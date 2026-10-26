"""A selector that must execute its learned, typed process before deciding.

All arms deploy the same self-feedback model. Only TRAIN pretraining differs:
scratch, independent state snapshots, or autoregressive state transitions.
"""

import copy

import torch
from torch import nn

from .V4Model import _masked_max, _masked_mean
from .pairwise_selector import ScoreDifferenceSelector
from .r77_data import parameter_hash
from .r88_model import prepare_role_inputs
from .r113_process_data import (BINARY_FIELDS, GRAPH_FIELDS, PROBLEMS,
                               RAW_KEYS, SCALAR_FIELDS, initial_state)
from .r113_processor import TypedProcessOperator


ARMS = ('A_scratch', 'B_snapshot', 'C_transition')
PROCESS_SEED = 113002
SPEC = dict(experiment='R113', schema_version=1, node_stem_layers=2,
    hidden_dim=64, max_steps=12, neighbors=2, deployment='self_initialized_predicted_feedback',
    decision='candidate interaction with global and node trajectories plus final directed state; no old-score bypass',
    program_supervision='new TRAIN-only typed fragments, not native solver routes or winners',
    role_arm='B_execution_anchor', processor_seed=PROCESS_SEED,
    pretraining_modes={'A_scratch': None, 'B_snapshot': 'snapshot', 'C_transition': 'transition'},
    snapshot_definition='independent input/time-conditioned predictions; same target states and posthoc deltas',
    process_targets_in_query_forward=False)


class FeedbackProcessor(TypedProcessOperator):
    def __init__(self, hidden_dim=64):
        super().__init__(hidden_dim)
        # Independent snapshots must not be restricted to one propagation hop.
        # Both training modes and deployment use this same learned graph update.
        self.graph_type = nn.Embedding(len(GRAPH_FIELDS), 8)
        self.graph_state = nn.Sequential(nn.Linear(2 * hidden_dim + 12, hidden_dim),
            nn.SiLU(), nn.Linear(hidden_dim, 2))
        del self.trace_readout

    def _step(self, program, encoded, previous, step):
        state = super()._step(program, encoded, previous, step)
        b, k = program['node_mask'].shape
        g, d = len(GRAPH_FIELDS), self.hidden_dim
        node = encoded['nodes'][:, None].expand(b, g, k, d)
        pool = encoded['pool'][:, None, None].expand_as(node)
        types = self.graph_type.weight[None, :, None].expand(b, g, k, 8)
        fraction = node.new_full((b, g, k, 1), step + 1) / program['steps'][:, None, None, None]
        features = torch.cat((node, pool, types, previous['reached'][..., None],
            previous['frontier'][..., None], state['reached'][..., None], fraction), -1)
        predicted = self.graph_state(features).sigmoid()
        valid = program['graph_mask'][:, :, None] & program['node_mask'][:, None]
        state['reached'] = predicted[..., 0] * valid
        state['frontier'] = predicted[..., 1] * valid
        return state

    def run(self, program, context=None, mode='transition'):
        if mode not in ('transition', 'snapshot'):
            raise ValueError('Choose transition or independent snapshot pretraining')
        if mode == 'snapshot' and not self.training:
            raise PermissionError('All deployment uses self-feedback, never snapshot mode')
        encoded = self._encode(program)
        if context is not None:
            encoded['pool'] = encoded['pool'] + context
            encoded['nodes'] = encoded['nodes'] + context[:, None] * program['node_mask'][..., None]
            encoded['anchor'] = encoded['anchor'] + context
            encoded['pointer_keys'] = torch.cat((encoded['nodes'], encoded['anchor'][:, None]), 1)
        first = initial_state(program)
        previous = first
        history = {key: [value] for key, value in first.items()}
        for step in range(program['nodes'].size(1)):
            state_in = previous if mode == 'transition' else first
            prediction = self._step(program, encoded, state_in, step)
            active = step < program['steps']
            previous = {key: torch.where(active.reshape(-1, *([1] * (value.ndim - 1))),
                value, previous[key]) for key, value in prediction.items()}
            for key, value in previous.items():
                history[key].append(value)
        result = {key: torch.stack(value, 1) for key, value in history.items()}
        # Identical supervised transitions in both modes, without feeding them in.
        result['scalar_delta'] = result['scalar'][:, 1:] - result['scalar'][:, :-1]
        return result

    def forward(self, program):
        return self.run(program)


class ExecutingEncoder(nn.Module):
    def __init__(self, stem, **params):
        super().__init__()
        self.stem = stem
        self.stem.layers = nn.ModuleList(list(stem.layers[:2]))
        d, h = params['embedding_dim'], SPEC['hidden_dim']
        self.context = nn.Sequential(nn.Linear(2 * d, h), nn.GELU(), nn.Linear(h, h))
        self.processor = FeedbackProcessor(h)
        fields = len(SCALAR_FIELDS) + len(BINARY_FIELDS) + 3 * len(GRAPH_FIELDS) + 2
        self.state_projection = nn.Sequential(nn.Linear(fields, d), nn.GELU(), nn.Linear(d, d), nn.LayerNorm(d))
        self.node_projection = nn.Sequential(nn.Linear(26, d), nn.GELU(), nn.Linear(d, d), nn.LayerNorm(d))
        self.edge_projection = nn.Sequential(nn.Linear(34, d), nn.GELU(), nn.Linear(d, d), nn.LayerNorm(d))
        self.state_types = nn.Parameter(torch.randn(3, d) * .02)

    def process_state(self, batch):
        """Expose input-only predictions for post-lock process diagnostics."""
        from .r113_torch_inputs import compile_inputs_torch
        raw = {key: value for key, value in batch.items() if key in RAW_KEYS}
        problem = PROBLEMS[int(batch['problem_id'])]
        raw.pop('node' if problem == 'ATSP' else 'matrix', None)
        if problem == 'TSP':
            raw = dict(raw, node=raw['node'][..., :2])
        elif problem == 'CVRP':
            raw = dict(raw, node=raw['node'][..., :4])
        program = compile_inputs_torch(raw, problem)
        nodes, node_mask = self.stem(batch)
        context = self.context(torch.cat((_masked_mean(nodes, node_mask), _masked_max(nodes, node_mask)), -1))
        outputs = self.processor.run(program, context)
        return program, outputs

    def forward(self, batch):
        program, outputs = self.process_state(batch)
        b, t = outputs['scalar'].shape[:2]
        count = program['node_mask'].sum(-1).clamp_min(1)
        reached = outputs['reached'].sum(-1) / count[:, None, None]
        frontier = outputs['frontier'].sum(-1) / count[:, None, None]
        pairs = (count * (count - 1)).clamp_min(1)
        compatible = outputs['compatible'].sum((-1, -2)) / pairs[:, None, None]
        device, dtype = outputs['scalar'].device, outputs['scalar'].dtype
        time = torch.arange(t, device=device, dtype=dtype)[None, :, None] / program['steps'][:, None, None]
        time = time.expand(b, -1, -1)
        state = torch.cat((outputs['scalar'], outputs['binary'], reached, frontier, compatible,
                           outputs['pointer'][..., -1:], time), -1)
        mask = torch.arange(t, device=device)[None] <= program['steps'][:, None]
        globals = self.state_projection(state) + self.state_types[0]
        k = program['nodes'].size(1)
        matrix = outputs['compatible'].permute(0, 1, 3, 4, 2)
        degree = (count - 1).clamp_min(1)[:, None, None, None]
        node_state = torch.cat((program['nodes'][:, None].expand(b, t, k, 12),
            outputs['reached'].transpose(2, 3), outputs['frontier'].transpose(2, 3),
            matrix.sum(3) / degree, matrix.sum(2) / degree,
            outputs['pointer'][..., :k, None], time[:, :, None].expand(b, t, k, 1)), -1)
        node_tokens = self.node_projection(node_state) + self.state_types[1]
        node_valid = mask[:, :, None] & program['node_mask'][:, None]
        rows = torch.arange(b, device=device)
        final = matrix[rows, program['steps']]
        pointer = outputs['pointer'][rows, program['steps'], :k]
        edge_state = torch.cat((program['nodes'][:, :, None].expand(b, k, k, 12),
            program['nodes'][:, None].expand(b, k, k, 12), program['edge_features'],
            final, final.transpose(1, 2), pointer[:, :, None, None].expand(b, k, k, 1),
            pointer[:, None, :, None].expand(b, k, k, 1)), -1)
        edge_tokens = self.edge_projection(edge_state) + self.state_types[2]
        edge_valid = program['node_mask'][:, :, None] & program['node_mask'][:, None]
        edge_valid &= ~torch.eye(k, dtype=torch.bool, device=device)[None]
        tokens = torch.cat((globals, node_tokens.flatten(1, 2), edge_tokens.flatten(1, 2)), 1)
        valid = torch.cat((mask, node_valid.flatten(1, 2), edge_valid.flatten(1, 2)), 1)
        return tokens.masked_fill(~valid[..., None], 0.), valid


class ExecutingSelector(ScoreDifferenceSelector):
    def __init__(self, arm, **params):
        if arm not in ARMS:
            raise ValueError('Unknown R113 arm')
        params = dict(params, r113_arm=arm, r113_spec=copy.deepcopy(SPEC))
        super().__init__(**params)
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(PROCESS_SEED)
            self.instance_encoder = ExecutingEncoder(self.instance_encoder, **params)

    def forward(self, batch):
        return super().forward(prepare_role_inputs(batch, SPEC['role_arm']))


def make_model(params, arm, state_dict=None):
    with torch.device('cpu'), torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(2)
        model = ExecutingSelector(arm, **copy.deepcopy(params))
    if state_dict is not None:
        model.load_state_dict(state_dict, strict=True)
    return model


def initialize_shared(initial, geometry, arm, device='cpu'):
    if initial.get('seed') != 2 or initial.get('initialization') != 'scratch':
        raise ValueError('No trained selector can initialize R113')
    model = make_model(initial['model_params'], arm)
    actual, copied = dict(model.named_parameters()), {}
    with torch.no_grad():
        for name, value in actual.items():
            source = name.replace('instance_encoder.stem.', 'instance_encoder.')
            if source in initial['trainable']:
                value.copy_(initial['trainable'][source])
                copied[name] = value
        branch = model.instance_encoder.stem.geometry_residual
        branch.mean.copy_(torch.as_tensor(geometry['mean']))
        branch.std.copy_(torch.as_tensor(geometry['std']))
    receipt = dict(seed=2, fresh_initialization=True, historical_checkpoint_loaded=False,
        initialization='fresh_seed2_backbone_seed113002_process_before_new_TRAIN_pretraining',
        trainable_sha256=parameter_hash(actual), fresh_trainable_sha256=parameter_hash(actual),
        fresh_state_sha256=parameter_hash(model.state_dict()), common_parameter_sha256=parameter_hash(copied),
        common_source_trainable_sha256=initial.get('trainable_sha256'),
        parameter_count=sum(p.numel() for p in actual.values()), processor_pretraining_applied=False,
        r113_spec=copy.deepcopy(SPEC))
    return model.to(device), copy.deepcopy(model.params), receipt
