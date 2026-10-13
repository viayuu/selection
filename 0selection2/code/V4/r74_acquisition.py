"""Locked R74 input-only recipes; no completion or final-cost routine is called."""

import importlib
import importlib.util
import json
import sys
from contextlib import contextmanager
from importlib.machinery import ModuleSpec
from pathlib import Path

import numpy as np
import torch


HOME = Path('/public/home/shiys')
BQ_ROOT = HOME / 'methods论文_mineru_output/source_code/bq-nco'
RELD_ROOT = HOME / 'reld-nco-main'
EASY_ROOT = HOME / 'EasyNCO'
BRIDGE = HOME / 'easynco_v3_bridge/scripts'
SCALARS = ('selections_over_n', 'required_visited_over_n', 'partial_distance', 'last_probability',
           'normalized_entropy', 'probability_gap', 'legal_fraction', 'depot_returns_over_n',
           'remaining_load', 'current_time', 'current_route_length', 'accumulated_waiting')


def quarter_increment(n):
    h = (n // 4 - 1) // 2
    if n // 4 < 1 or h < 0 or 1+2*h > n//4:
        raise ValueError('True size has no admissible R74 quarter-instance prefix')
    return h


def guard_incomplete(n, visited, vrp, done=False, finished=None):
    required = visited[..., 1:] if vrp else visited
    maximum = int(required.sum(-1).max())
    if bool(torch.as_tensor(done).any()) or (finished is not None and bool(torch.as_tensor(finished).any())):
        raise ValueError('A trajectory completed; preserve failure without a completion call')
    if maximum > min(1+2*quarter_increment(n), n//4) or maximum >= n:
        raise ValueError('Required-node quarter boundary violated before solver completion')
    return maximum


def decision_statistics(probabilities, legal, selected):
    p = probabilities.float()
    if not legal.any(-1).all() or not torch.isfinite(p).all() or (p < 0).any():
        raise ValueError('Nonfinite/invalid incomplete decoder distribution')
    count = legal.sum(-1)
    entropy = -(p * p.clamp_min(1e-30).log()).sum(-1)
    entropy = torch.where(count > 1, entropy/count.clamp_min(2).float().log(), torch.zeros_like(entropy))
    top = p.topk(min(2, p.shape[-1]), -1).values
    gap = top[..., 0] - (top[..., 1] if top.shape[-1] == 2 else 0.)
    return p.gather(-1, selected[..., None]).squeeze(-1), entropy, gap, legal.float().mean(-1)


@contextmanager
def source_namespace(root, names):
    """The native repositories use conflicting top-level utils/model module names."""
    previous_path = list(sys.path)
    def belongs(name):
        return any(name == prefix or name.startswith(prefix + '.') for prefix in names)
    saved = {key: value for key, value in sys.modules.items() if belongs(key)}
    for key in saved:
        del sys.modules[key]
    sys.path.insert(0, str(root))
    try:
        yield
    finally:
        for key in list(sys.modules):
            if belongs(key):
                del sys.modules[key]
        sys.modules.update(saved)
        sys.path[:] = previous_path


def load_source(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@contextmanager
def matnet_namespace():
    """Bind real native MATNET exports without unrelated solver/env registries."""
    with source_namespace(HOME, ('EasyNCO', 'neural_solvers')), source_namespace(EASY_ROOT, ()):
        parent = importlib.import_module('EasyNCO.neural_solvers')
        for child in ('methods', 'envs'):
            name = 'EasyNCO.neural_solvers.'+child
            spec = ModuleSpec(name, loader=None, is_package=True)
            spec.submodule_search_locations = [str(EASY_ROOT / 'neural_solvers' / child)]
            package = importlib.util.module_from_spec(spec)
            sys.modules[name] = package
            setattr(parent, child, package)
        methods = sys.modules['EasyNCO.neural_solvers.methods']
        encoder = importlib.import_module('EasyNCO.neural_solvers.methods.matnet.matnet_encoder')
        decoder = importlib.import_module('EasyNCO.neural_solvers.methods.matnet.matnet_decoder')
        methods.MatNetEncoder = encoder.MatNetEncoder
        methods.ATSPDecoder, methods.FFSPDecoder = decoder.ATSPDecoder, decoder.FFSPDecoder
        yield


def recipe_records():
    return dict(
        BQ=dict(source=str(BQ_ROOT / 'learning/tsp/decoding.py'), symbol='greedy_decoding_step',
            checkpoint=str(BQ_ROOT / 'pretrained_models/tsp.best'),
            params=dict(dim_input_nodes=2, emb_size=192, dim_ff=512, activation_ff='relu',
                nb_layers_encoder=9, nb_heads=12, activation_attention='softmax', dropout=0., batchnorm=False, problem='tsp'),
            beam=1, knns=-1, initialization='original node0; append node0 only as closure placeholder'),
        RELD_CVRP=dict(source=str(RELD_ROOT / 'CVRP/CVRPModel.py'), symbol='one_step_rollout',
            checkpoint=str(RELD_ROOT / 'CVRP/weights/ReLD/model_epoch_90.pt'),
            config=str(RELD_ROOT / 'CVRP/config.yml'), forcing_first_step=False, starts='learned min(n,100)'),
        MATNET=dict(source=str(EASY_ROOT / 'neural_solvers/methods/matnet/policy.py'), symbol='ATSPPolicy',
            checkpoint=str(EASY_ROOT / 'pretrained/matnet_pretrain/matnet_atsp100.ckpt'),
            params=dict(embedding_dim=256, num_heads=16, qkv_dim=16, num_encoder_layers=5,
                feedforward_hidden=512, logit_clipping=10., ms_hidden_dim=16,
                ms_layer1_init=2**-.5, ms_layer2_init=16**-.5),
            starts='all n', column_embedding='CPU torch.randperm(n) with private generator seed2 per n; broadcast queries',
            environment='ATSPEnv method_name=matnet aug_factor=1; no NN initializer'),
        RELD_MTL=dict(source=str(RELD_ROOT / 'Multi-Task/models/MTLModel.py'),
            checkpoint=str(RELD_ROOT / 'Multi-Task/pretrained/reld_mtl/epoch-5000.pt'), problem='Train_ALL'),
        RELD_MOEL=dict(source=str(RELD_ROOT / 'Multi-Task/models/MOEModel_Light.py'),
            checkpoint=str(RELD_ROOT / 'Multi-Task/pretrained/reld_moe_light/epoch-5000.pt'), problem='Train_ALL'))


class ProbeRecipes:
    def __init__(self, device, r48_solvers, on_loaded):
        self.device = torch.device(device)
        self.r48_solvers, self.on_loaded = r48_solvers, on_loaded
        self.models, self.helpers, self.envs = {}, {}, {}
        self.recipes = recipe_records()

    def ensure(self, problem):
        if problem == 'TSP' and 0 not in self.models:
            with source_namespace(BQ_ROOT, ('model', 'utils', 'learning')):
                cls = importlib.import_module('model.model').BQModel
                decoder = load_source('r74_native_bq_decoding', BQ_ROOT / 'learning/tsp/decoding.py')
                model = cls(**self.recipes['BQ']['params'])
                state = torch.load(self.recipes['BQ']['checkpoint'], map_location='cpu', weights_only=False)
                model.load_state_dict(state['net'], strict=True)
            self.models[0], self.helpers[0] = model, decoder
            self._loaded(0, 'BQ')
        elif problem == 'CVRP' and 1 not in self.models:
            with source_namespace(RELD_ROOT / 'CVRP', ('utils', 'CVRPModel', 'CVRPEnv')):
                bridge = load_source('r74_native_cvrp_bridge', BRIDGE / 'run_reld_cvrp_single_bridge.py')
                model, params = bridge.instantiate_single_cvrp_model(self.device)
                required = dict(embedding_dim=128, encoder_layer_num=6, head_num=8, qkv_dim=16,
                    ff_hidden_dim=512, logit_clipping=50, forcing_first_step=False, eval_type='greedy')
                if any(params.get(key) != value for key, value in required.items()):
                    raise ValueError('The locked ReLD_CVRP native configuration differs')
                env_cls = importlib.import_module('CVRPEnv').CVRPEnv
            self.models[1], self.helpers[1], self.envs[1] = model, bridge, env_cls
            self.recipes['RELD_CVRP']['params'] = params
            self._loaded(1, 'RELD_CVRP')
        elif problem == 'ATSP' and 2 not in self.models:
            from .r57_probe import torchrl_compat
            torchrl_compat()
            with matnet_namespace():
                from EasyNCO.neural_solvers.methods.matnet.policy import ATSPPolicy
                from EasyNCO.neural_solvers.envs.ATSPEnv import ATSPEnv
                from EasyNCO.utils.utils import load_model
                class DeviceCompatibleATSPEnv(ATSPEnv):
                    @ATSPEnv.device.setter
                    def device(self, value):
                        current = ATSPEnv.device.fget(self)
                        if current is not None and value is not None and torch.device(current) == torch.device(value):
                            return
                        ATSPEnv.device.fset(self, value)

                    @property
                    def batch_size(self):
                        value = self.__dict__.get('_batch_size')
                        return torch.Size([]) if value is None else value

                    @batch_size.setter
                    def batch_size(self, value):
                        # Native MATNET states use this size without TorchRL spec resizing.
                        self.__dict__['_batch_size'] = torch.Size(value)
                model = ATSPPolicy(**self.recipes['MATNET']['params'])
                model = load_model(model, self.recipes['MATNET']['checkpoint'], device=1, model_name='matnet')
            model.set_decoder_strategy('greedy')
            self.models[2], self.envs[2] = model, DeviceCompatibleATSPEnv
            self._loaded(2, 'MATNET')
        elif problem not in ('TSP', 'CVRP', 'ATSP') and 3 not in self.models:
            from .r51_probe import build_solvers
            from code.unified_selector.registry import PROBLEMS
            with source_namespace(RELD_ROOT / 'Multi-Task', ('utils', 'models', 'envs')):
                models, bridge = build_solvers(dict(solvers=self.r48_solvers), self.device)
                utils, _, _ = bridge.import_reld_modules()
                envs = {task: utils.get_env(task)[0] for task in PROBLEMS[3:]}
            self.models[3], self.models[4] = models[1], models[0]
            self.helpers[3] = self.helpers[4] = bridge
            self.envs.update(envs)
            for probe, name in ((3, 'RELD_MTL'), (4, 'RELD_MOEL')):
                self.recipes[name]['params'] = self.r48_solvers[name]['runtime_params_contract']
                self._loaded(probe, name)

    def _loaded(self, probe, name):
        model = self.models[probe].to(self.device).float().requires_grad_(False).eval()
        self.on_loaded(name, model, self.recipes[name])

    def start(self, problem, instances, rows, probe):
        self.ensure(problem)
        return Trajectory(self, problem, instances, rows, probe)


class Trajectory:
    @torch.no_grad()
    def __init__(self, recipes, problem, instances, rows, probe):
        self.recipes, self.problem, self.probe = recipes, problem, probe
        self.model, self.device = recipes.models[probe], recipes.device
        self.vrp = problem not in ('TSP', 'ATSP')
        self.batch_size, self.done, self.env, self.distribution = len(rows), False, None, None
        samples = [instances[int(row)] for row in rows]
        if probe == 0:
            xy = torch.stack([torch.as_tensor(sample[0], dtype=torch.float32) for sample in samples]).to(self.device)
            self.n = xy.shape[1]
            self.distance_matrix = torch.cdist(xy, xy)
            self.starts = 1
            coords = torch.cat((xy, xy[:, :1]), 1)
            decoder = recipes.helpers[0]
            self.subproblem = decoder.DecodingSubPb(coords, torch.arange(self.n+1, device=self.device)[None].expand(len(rows), -1), None)
        elif probe == 2:
            matrix = torch.stack([torch.as_tensor(sample[0] if sample.ndim == 3 else sample, dtype=torch.float32)
                                  for sample in samples]).to(self.device)
            self.n, self.starts, self.distance_matrix = matrix.shape[1], matrix.shape[1], matrix
            with torch.device(self.device):
                self.env = recipes.envs[2](pomo_size=self.n, device=str(self.device), seed=2,
                                          method_name='matnet', aug_factor=1, init_solver=None)
                self.env.load_problems(matrix, batch_size=len(rows))
                reset = self.env._reset(None)
                permutation = torch.randperm(self.n, generator=torch.Generator(device='cpu').manual_seed(2), device='cpu').to(self.device)
                row_embedding = torch.zeros(len(rows), self.n, 256)
                column_embedding = torch.zeros_like(row_embedding)
                column_embedding[:, torch.arange(self.n), permutation] = 1.
                self.model.encoded_row, self.model.encoded_col = self.model.encoder(row_embedding, column_embedding, reset['problems'])
                self.model.decoder.set_kv(self.model.encoded_col)
                self.state = self.env.pre_step()
        else:
            if probe == 1:
                native, self.n = recipes.helpers[1].collate_batch(instances, rows)
                native = {key: value.to(self.device) for key, value in native.items()}
                self.starts = min(self.n, 100)
                with torch.device(self.device):
                    self.env = recipes.envs[1](multi_width=self.starts, problem_size=self.n, device=self.device)
                    self.env.load_random_problems(native, aug_factor=1)
            else:
                native, self.n = recipes.helpers[probe].collate_problem_batch(problem+'train', instances, rows)
                native = tuple(value.to(self.device) for value in native)
                if any(float(sample.get('capacity', 1.)) != 1. for sample in samples):
                    raise ValueError('Native MVRP normalized demand convention changed')
                with torch.device(self.device):
                    self.env = recipes.envs[problem](problem_size=self.n, pomo_size=self.n, device=self.device)
                    self.env.load_problems(len(rows), problems=native, aug_factor=1)
                self.starts = self.env.pomo_size
                if 'B' in problem and not (self.env.depot_node_demand.gather(1, self.env.step_state.START_NODE) > 0).all():
                    raise ValueError('Native backhaul starts must remain the actual positive-demand subset')
            with torch.device(self.device):
                reset, _, _ = self.env.reset()
                self.model.pre_forward(reset)
                self.state, _, self.done = self.env.pre_step()
            self.distance_matrix = torch.cdist(self.env.depot_node_xy, self.env.depot_node_xy)
        self.increment = quarter_increment(self.n)
        self.node_count = self.n + int(self.vrp)
        shape = (self.batch_size, self.starts)
        self.visited = torch.zeros(*shape, self.node_count, dtype=torch.bool, device=self.device)
        self.current = torch.zeros(shape, dtype=torch.long, device=self.device)
        self.start_node = None
        self.selections = self.network_steps = self.model_calls = self.decoder_calls = 0
        self.partial_distance = torch.zeros(shape, device=self.device)
        self.waiting = torch.zeros_like(self.partial_distance)
        self.depot_returns = torch.zeros_like(self.partial_distance)
        self.last_stats = tuple(torch.zeros_like(self.partial_distance) for _ in range(4))
        off_diagonal_sum = self.distance_matrix.sum((1, 2)) - self.distance_matrix.diagonal(dim1=1, dim2=2).sum(1)
        self.scale = (self.n * (off_diagonal_sum /
                     (self.node_count*(self.node_count-1))).clamp_min(1e-8))[:, None]
        if probe == 0:
            self.visited[:, :, 0] = True
            self.start_node = self.current.clone()
            self.selections = 1
        else:
            for _ in range(2 if self.vrp else 1):
                self.step()
            self.start_node = self.current.clone()
        self.check()

    def check(self, visited=None):
        finished = None if self.env is None else getattr(self.env, 'finished', None)
        return guard_incomplete(self.n, self.visited if visited is None else visited, self.vrp, self.done, finished)

    @torch.no_grad()
    def step(self):
        self.check()
        self.distribution = None
        def capture(_module, _args, output):
            probabilities = output[0] if isinstance(output, tuple) else output
            self.distribution = probabilities.detach().float()
        handle = (self.model if self.probe == 0 else self.model.decoder).register_forward_hook(capture)
        try:
            with torch.device(self.device):
                if self.probe == 0:
                    old = self.subproblem
                    selected, self.subproblem = self.recipes.helpers[0].greedy_decoding_step(old, self.model, -1)
                    selected = selected[:, None]
                    local = (old.original_idxs == selected).long().argmax(-1)[:, None]
                    p = self.distribution.softmax(-1)[:, None]
                    legal = torch.isfinite(self.distribution)[:, None]
                else:
                    before_mask = (self.state['ninf_mask'] if self.probe == 2 else self.state.ninf_mask).clone()
                    if self.probe == 1:
                        selected, _ = self.model.one_step_rollout(self.state, self.env.get_cur_feature(), eval_type='greedy')
                    elif self.probe == 2:
                        self.state = self.model(self.state)
                        selected = self.state['action']
                    else:
                        selected, _ = self.model(self.state)
                    local, legal = selected, torch.isfinite(before_mask)
                    if self.distribution is None:
                        p = torch.zeros_like(before_mask).scatter_(-1, selected[..., None], 1.)
                    else:
                        p = self.distribution
        finally:
            handle.remove()
        self.model_calls += 1
        self.decoder_calls += int(self.distribution is not None)
        if selected.shape != self.current.shape or (selected < 0).any() or (selected >= self.node_count).any():
            raise ValueError('Native selection/start identity is outside the original node set')
        next_visited = self.visited.clone().scatter_(-1, selected[..., None], True)
        self.check(next_visited)
        self.last_stats = decision_statistics(p, legal, local)
        previous = self.current
        batch_ids = torch.arange(self.batch_size, device=self.device)[:, None]
        distance = self.distance_matrix[batch_ids, previous, selected]
        if self.selections == 0:
            distance = torch.zeros_like(distance)
        self.partial_distance += distance
        if self.vrp:
            self.depot_returns += ((selected == 0) & (previous != 0)).float()
            if hasattr(self.env, 'depot_node_tw_start'):
                arrival = self.env.current_time + distance / getattr(self.env, 'speed', 1.)
                self.waiting += (self.env.depot_node_tw_start.gather(1, selected)-arrival).clamp_min(0)*(selected != 0)
        if self.probe != 0:
            with torch.device(self.device):
                if self.probe == 2:
                    self.state = self.env._step(self.state)
                    self.done = self.state['done']
                else:
                    self.state, reward, self.done = self.env.step(selected)
                    if reward is not None:
                        raise ValueError('Unexpected reward from an incomplete trajectory')
        self.visited, self.current = next_visited, selected
        self.selections += 1
        self.check()
        self.distribution = None

    @torch.no_grad()
    def acquire_increment(self, base_nodes):
        for _ in range(self.increment):
            self.step()
            self.network_steps += 1
        maximum = self.check()
        required_count = self.visited[..., 1:].sum(-1) if self.vrp else self.visited.sum(-1)
        zero = torch.zeros_like(self.partial_distance)
        load = getattr(self.env, 'load', zero) if self.vrp else zero
        current_time = getattr(self.env, 'current_time', zero) if self.vrp else zero
        route_length = getattr(self.env, 'length', zero) if self.vrp else zero
        scalar = torch.stack((zero+self.selections/self.n, required_count.float()/self.n,
            self.partial_distance/self.scale, *self.last_stats, self.depot_returns/self.n, load,
            current_time/self.scale, route_length/self.scale, self.waiting/self.scale), -1).float()
        visited_mean = torch.bmm(self.visited.float(), base_nodes.float()) / self.visited.sum(-1, keepdim=True)
        if scalar.shape[-1] != 12 or not torch.isfinite(scalar).all() or not torch.isfinite(visited_mean).all():
            raise ValueError('Nonfinite or malformed formal incomplete observation')
        return dict(start=self.start_node.clone(), current=self.current.clone(), scalars=scalar,
            visited_mean=visited_mean, visited_bits=np.packbits(self.visited.cpu().numpy(), axis=-1, bitorder='little'),
            receipt=dict(probe=self.probe, true_n=self.n, starts=self.starts, increment=self.increment,
                selections=self.selections, network_selections=self.network_steps,
                required_visited_max=maximum, quarter_limit=self.n//4, finished_trajectories=0,
                done=False, finite_features=True, decoder_forward_calls=self.decoder_calls,
                model_action_calls=self.model_calls,
                initialization_selections=2 if self.vrp else 1))
