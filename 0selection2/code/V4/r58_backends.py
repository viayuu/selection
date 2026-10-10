"""R58 deployments: existing fixed weights, explicitly new constraint adapters."""

import copy
import sys
from pathlib import Path

import numpy as np
import torch

from .r58_environments import RoutingState, routefinder_depot_clock, routefinder_mask
from .r58_scenario import check_route, fields, file_hash


BRIDGE = Path('/public/home/shiys/easynco_v3_bridge/scripts')
RF_METHODS = ('RouteFinder', 'MoSES_RF', 'MoSES_CaDA')
DIFFUSION_METHODS = ('DIFUSCO', 'DIFUSCO500', 'T2T', 'T2T500')
REVISED_RECIPE = 'r58_rf_aug1_diff20'


def independent_audit(problem, items, tours, costs):
    checks = [check_route(problem, item, tour, cost)
              for item, tour, cost in zip(items, tours, costs)]
    if len(checks) != len(items) or not checks:
        raise ValueError('Missing solver routes')
    failures = [(i, result) for i, result in enumerate(checks)
                if not result['feasible'] or not result['cost_matches']]
    if failures:
        raise ValueError(f'Independent scenario_v2 route check failed: {failures[:3]}')
    return dict(checked=len(checks), feasible=len(checks), cost_matches=len(checks),
                costs_fp64=[result['cost'] for result in checks])


def load_bridge():
    if str(BRIDGE) not in sys.path:
        sys.path.insert(0, str(BRIDGE))
    # PyG's optional accelerator wheel requires newer glibc than gpu03. Its
    # supported torch-sparse fallback is sufficient; never replace tensor ops.
    try:
        import pyg_lib  # noqa: F401
    except (ImportError, OSError):
        sys.modules['pyg_lib'] = None
    import train_100k_backends as base
    # Process-local replacement only. No original source or other process is changed.
    base.audit_tours = independent_audit
    return base


class AdaptedPOMO:
    def __init__(self, problem, method, device):
        self.problem, self.method, self.device = problem, method, torch.device(device)
        self.base = load_bridge()
        self.original = self.base.make_backend(problem, method, str(self.device))
        self.model = self.original.model.eval().requires_grad_(False)
        torch.set_float32_matmul_precision('highest')

    def profile(self):
        profile = self.original.profile()
        profile.update(scenario_deployment='original fixed weights + scenario_v2 unified-constraint adaptation',
            problem=self.problem, constraint_adapter=str(Path(__file__).with_name('r58_environments.py')),
            constraint_adapter_sha256=file_hash(Path(__file__).with_name('r58_environments.py')),
            normalization='original coordinates; demands divided by raw capacity; raw customer TW/service, '
                          'speed and L; raw depot departure; missing depot close=+infinity',
            budget=dict(starts='ALL N physical customer nodes, including pickups', augmentation=1,
                        decoding='greedy/argmax, explicitly selected', precision='FP32; FP64 independent cost',
                        selection='minimum complete-trajectory FP32 travel distance; stable first start on ties'),
            adaptation='separate delivered/picked capacity counters; no delivery after pickup within route; '
                       'phase-relative decoder load; input depot close else infinity; exact open and distance semantics',
            historical_equivalence=False)
        return profile

    @torch.inference_mode()
    def run(self, items):
        state = RoutingState(self.problem, items, self.device)
        if self.method.startswith('RELD_'):
            self.model.pre_forward(state.reld_reset())
        else:
            self.model.set_decoder_strategy('greedy')
            self.model.pre_forward(state.native_reset())
        done = False
        while not done:
            if self.method.startswith('RELD_'):
                selected, _ = self.model(state.reld_step())
            else:
                output = self.model(state.native_step())
                selected = output['action']
            done = state.step(selected)
        best = state.cost.argmin(-1)
        tours = state.tour[torch.arange(state.b, device=self.device), best].cpu().tolist()
        costs = state.cost[torch.arange(state.b, device=self.device), best].cpu().tolist()
        audit = independent_audit(self.problem, items, tours, costs)
        return self.base.InferenceResult(audit['costs_fp64'], tours, audit)


class AdaptedRouteFinder:
    def __init__(self, problem, method, device):
        self.problem, self.method, self.device = problem, method, torch.device(device)
        self.base = load_bridge()
        self.original = self.base.RouteFinderBackend(problem, method, str(self.device))

    def profile(self):
        profile = self.original.profile()
        profile.update(scenario_deployment='original fixed weights + scenario_v2 unified-constraint adaptation',
            problem=self.problem, historical_equivalence=False,
            normalization='original coordinates; demands divided by raw capacity; raw speed and L; '
                          'customer TW retained; missing depot close=+infinity',
            historical_adapter_note='Historical scalar columns are not certified for this deployment; '
                                    'scenario_v2 uses its explicit contract, not the inherited depot cutoff of3',
            backhaul_adapter='explicit classical class=1; separate capacity counters; pure pickup routes allowed',
            checkpoint_loading='strict full state; duplicate names restored only from identical registered parameter aliases',
            checkpoint_alias_loader_sha256=file_hash(BRIDGE / 'train_100k_alias_weights.py'),
            constraint_adapter_sha256=file_hash(Path(__file__).with_name('r58_environments.py')),
            r58_tensor_adapter_sha256=file_hash(__file__))
        profile['budget'] = dict(starts='ALL N physical customers', augmentation=1,
            orientation='original, untransformed',
            decoding='multistart_greedy, explicitly selected', precision='FP32, matmul_precision=highest',
            inference_batch_size=1,
            checkpoint_bucket='50 if true N<=75, otherwise100',
            selection='minimum over ALL N starts in original orientation; stable first tie')
        profile['budget_source'] = ('R58 pre-validation revised deployment: original orientation only, '
            'all customer starts, unchanged fixed checkpoint; NOT equivalent to the prior 8-augmentation recipe')
        profile['recipe_revision'] = REVISED_RECIPE
        return profile

    def tensor_dict(self, items):
        from tensordict import TensorDict
        specs = [fields(self.problem, item) for item in items]
        def tensor(key):
            return torch.as_tensor(np.stack([s[key] for s in specs]), dtype=torch.float32)
        capacity, demand = tensor('capacity')[:, None], tensor('demand')[:, 1:]
        count = len(items)
        return TensorDict(dict(locs=tensor('xy'), demand_linehaul=(demand / capacity).clamp_min(0),
            demand_backhaul=(-demand / capacity).clamp_min(0),
            distance_limit=tensor('limit')[:, None], time_windows=torch.stack((tensor('start'), tensor('end')), -1),
            service_time=tensor('service'), vehicle_capacity=torch.ones(count, 1),
            capacity_original=capacity, speed=tensor('speed')[:, None],
            open_route=torch.full((count, 1), specs[0]['opened'], dtype=torch.bool),
            backhaul_class=torch.ones(count, 1, dtype=torch.int64)), batch_size=[count])

    @torch.inference_mode()
    def run(self, items):
        if len(items) != 1:
            raise ValueError('RouteFinder-family scenario_v2 deployments require singleton inference')
        scale = fields(self.problem, items[0])['n']
        bucket = 50 if scale <= 75 else 100
        runner = self.original.pool.get(self.original.key, bucket)
        if bucket not in self.original.validated_buckets:
            checkpoint = torch.load(self.original.checkpoints[str(bucket)], map_location='cpu', weights_only=False)
            weights = {k[len('policy.'):]: v for k, v in checkpoint['state_dict'].items() if k.startswith('policy.')}
            if not weights:
                raise ValueError('Missing original policy weights')
            from train_100k_alias_weights import load_alias_complete_state
            load_alias_complete_state(runner.policy, weights)
            runner.policy.eval().requires_grad_(False)
            runner.env.get_action_mask = routefinder_mask
            original_step = runner.env._step
            runner.env._step = lambda td: routefinder_depot_clock(original_step(td))
            self.original.validated_buckets.add(bucket)
        torch.set_float32_matmul_precision('highest')
        batch = routefinder_depot_clock(runner.env.reset(self.tensor_dict(items)).to(self.device))
        expected = torch.as_tensor(fields(self.problem, items[0])['xy'], dtype=torch.float32,
                                   device=self.device)[None]
        if not torch.equal(batch['locs'], expected):
            raise ValueError('RF policy input changed original coordinate orientation')
        if torch.is_autocast_enabled() or any(p.dtype != torch.float32 for p in runner.policy.parameters()):
            raise ValueError('RF revised deployment requires FP32 inference and weights')
        output = runner.policy(batch, runner.env, phase='test', num_starts=scale,
                               decode_type='multistart_greedy', return_actions=True)
        rewards = output['reward'].reshape(1, scale)
        actions = output['actions'].reshape(1, scale, -1)
        best = rewards.argmax(-1)
        rows = torch.arange(len(items), device=self.device)
        costs = (-rewards[rows, best]).cpu().tolist()
        tours = actions[rows, best].cpu().tolist()
        # The original API omits the forced initial depot from its action output.
        tours = [[0, *tour] if not tour or tour[0] != 0 else tour for tour in tours]
        audit = independent_audit(self.problem, items, tours, costs)
        return self.base.InferenceResult(audit['costs_fp64'], tours, audit)


class AdaptedDiffusion:
    """New fixed compute budget; original checkpoints and actual guided rewrite."""

    def __init__(self, problem, method, device):
        self.original = load_bridge().make_backend(problem, method, device)
        self.method, self.problem = method, problem
        self.overrides = dict(inference_diffusion_steps=20, two_opt_iterations=100,
            rewrite=self.original.is_t2t, rewrite_steps=1 if self.original.is_t2t else 0,
            inference_steps=10, parallel_sampling=1, sequential_sampling=1)
        self.ratio = self.original.parameters['rewrite_ratio']
        for key, value in self.overrides.items():
            self.original.parameters[key] = value
            setattr(self.original.model.args, key, value)
        self._check_arguments()

    def _check_arguments(self):
        for key, value in self.overrides.items():
            if self.original.parameters[key] != value or getattr(self.original.model.args, key) != value:
                raise ValueError('Actual diffusion arguments differ from revised R58 recipe')
        if (self.original.parameters['rewrite_ratio'] != self.ratio or
                self.original.model.args.rewrite_ratio != self.ratio):
            raise ValueError('Original checkpoint-specific rewrite ratio changed')

    def profile(self):
        self._check_arguments()
        profile = copy.deepcopy(self.original.profile())
        profile.update(backend='original_diffusion_with_r58_fixed_budget',
            recipe_revision=REVISED_RECIPE, historical_equivalence=False,
            r58_adapter_sha256=file_hash(__file__),
            budget_source='R58 pre-validation revised budget: 20 denoising steps, 100 2-opt cap; '
                'T2T performs one genuine 10-step guided rewrite with original checkpoint/rewrite ratio. '
                'Not equivalent to the original 50-step/3-rewrite recipe.')
        profile['budget'].update(denoising_steps=20, two_opt_iterations=100,
            rewrite_steps=1 if self.original.is_t2t else 0,
            rewrite_denoising_steps=10 if self.original.is_t2t else 0)
        return profile

    def run(self, items):
        if len(items) != 1:
            raise ValueError('Revised diffusion deployments require singleton inference')
        self._check_arguments()
        model, module = self.original.model, self.original.module
        witnessed = dict(denoising_calls=0, guided_calls=0, two_opt_limits=[])
        originals = {}
        for name, counter in (('categorical_denoise_step', 'denoising_calls'),
                              ('guided_categorical_denoise_step', 'guided_calls')):
            if not hasattr(model, name):
                continue
            original = getattr(model, name)
            originals[name] = original
            def observe(*args, _original=original, _counter=counter, **kwargs):
                witnessed[_counter] += 1
                return _original(*args, **kwargs)
            setattr(model, name, observe)
        refine = module.batched_two_opt_torch
        def observe_refine(*args, **kwargs):
            witnessed['two_opt_limits'].append(kwargs.get('max_iterations'))
            return refine(*args, **kwargs)
        module.batched_two_opt_torch = observe_refine
        try:
            result = self.original.run(items)
            expected = dict(denoising_calls=20, guided_calls=10 if self.original.is_t2t else 0,
                two_opt_limits=[100, 100] if self.original.is_t2t else [100])
            if witnessed != expected:
                raise ValueError(f'Actual diffusion execution budget differs: {witnessed}; expected {expected}')
            result.audit['execution_witness'] = witnessed
            return result
        finally:
            for name, original in originals.items():
                setattr(model, name, original)
            module.batched_two_opt_torch = refine


def make_backend(problem, method, device='cuda'):
    if problem == 'TSP' and method in DIFFUSION_METHODS:
        return AdaptedDiffusion(problem, method, device)
    if problem not in ('TSP', 'CVRP', 'ATSP'):
        if method in ('RELD_MOEL', 'RELD_MTL', 'MTPOMO', 'MVMOE'):
            return AdaptedPOMO(problem, method, device)
        if method in ('RouteFinder', 'MoSES_RF', 'MoSES_CaDA'):
            return AdaptedRouteFinder(problem, method, device)
    if problem == 'CVRP' and method in ('RouteFinder', 'MoSES_RF', 'MoSES_CaDA'):
        return AdaptedRouteFinder(problem, method, device)
    return load_bridge().make_backend(problem, method, device)
