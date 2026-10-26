from __future__ import annotations

import ast
import importlib
import inspect
import sys
import types
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from train_100k_common import WORKSPACE, file_hash, sample_scale


SCRIPT_ROOT = Path(__file__).resolve().parent
SOURCE_ROOT = WORKSPACE / "methods论文_mineru_output/source_code"
SUPPORTED_METHODS = {"BQ", "DIFUSCO", "DIFUSCO500", "ELG", "LEHD", "OMNI", "T2T", "T2T500",
                     "ICAM", "MVMOE", "MoSES_CaDA", "MoSES_RF", "RELD_CVRP", "RouteFinder",
                     "GLOP", "ICAM_ATSP", "MATNET", "MATPOENET", "UNICO_MatPOENet", "MTPOMO",
                     "RELD_MOEL", "RELD_MTL"}


def supports_backend(problem, method):
    return method in SUPPORTED_METHODS


def relocated_module(name, path):
    if name in sys.modules:
        return sys.modules[name]
    path = Path(path)
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    old_root = "/public/home/zhoucl/shiys"
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and node.value.startswith(old_root):
            node.value = str(WORKSPACE) + node.value[len(old_root):]
    module = types.ModuleType(name)
    module.__file__ = str(path)
    sys.modules[name] = module
    exec(compile(tree, str(path), "exec"), module.__dict__)
    return module


def source_hashes(root):
    return {str(path): file_hash(path) for path in sorted(Path(root).rglob("*.py"))}


@dataclass
class InferenceResult:
    costs: list
    tours: list
    audit: dict


def audit_tours(problem, items, tours, costs):
    if problem == "CVRP":
        from code.V4.r114_tensor_boundary_recovery import cpu_numpy_view
        items = cpu_numpy_view(items)
    if len(items) != len(tours) or len(items) != len(costs):
        raise ValueError("Solver output count changed")
    maximum_cost_error = 0.0
    for item, raw_tour, reported in zip(items, tours, costs):
        scale = sample_scale(problem, item)
        nodes = np.asarray(raw_tour).reshape(-1)
        if nodes.dtype.kind not in "iuf" or not np.isfinite(nodes).all() or not np.equal(nodes, np.floor(nodes)).all():
            raise ValueError("Tour node ids must be finite integers")
        if (nodes < 0).any() or (nodes > scale).any():
            raise ValueError("Tour references a nonexistent node")
        if not np.isfinite(float(reported)) or float(reported) <= 0:
            raise ValueError("Reported route cost must be finite and positive")
        tour = nodes.astype(np.int64)
        if problem == "TSP":
            if not np.array_equal(np.sort(tour), np.arange(scale)):
                raise ValueError("TSP tour is not a permutation")
            locations = np.asarray(item[0], dtype=np.float64)
            ordered = locations[tour]
            actual = np.linalg.norm(ordered - np.roll(ordered, 1, axis=0), axis=-1).sum()
        elif problem == "ATSP":
            if not np.array_equal(np.sort(tour), np.arange(scale)):
                raise ValueError("ATSP tour is not a permutation")
            matrix = np.asarray(item[0], dtype=np.float64)
            actual = matrix[tour, np.roll(tour, -1)].sum()
        else:
            if problem == "CVRP":
                locations = np.concatenate((np.asarray(item["depot"][0]), np.asarray(item["loc"][0])))
                demands = np.asarray(item["demand"][0], dtype=np.float64)
            else:
                locations = np.concatenate((np.asarray(item["depot_xy"]), np.asarray(item["node_xy"])))
                demands = np.asarray(item["node_demand"], dtype=np.float64)
            if (tour < 0).any() or (tour > scale).any():
                raise ValueError("VRP tour references a nonexistent node")
            customers = tour[tour != 0]
            if not np.array_equal(np.sort(customers), np.arange(1, scale + 1)):
                raise ValueError("VRP customers are not visited exactly once")
            if len(tour) == 0 or tour[0] != 0:
                tour = np.concatenate(([0], tour))
            if tour[-1] != 0:
                tour = np.concatenate((tour, [0]))
            boundaries = np.flatnonzero(tour == 0)
            actual = 0.0
            remaining_linehauls = demands > 0
            for start, end in zip(boundaries[:-1], boundaries[1:]):
                route = tour[start:end + 1]
                if len(route) <= 2:
                    continue
                customer_ids = route[1:-1] - 1
                load = 1.0 if remaining_linehauls.any() or "B" not in problem else 0.0
                for customer_id in customer_ids:
                    load -= demands[customer_id]
                    if load < -1e-5 or load > 1.0 + 1e-5:
                        raise ValueError("VRP route violates vehicle capacity")
                    remaining_linehauls[customer_id] = False
                open_route = problem.startswith("OVRP")
                edges = np.linalg.norm(np.asarray(locations[route[:-1]], dtype=np.float64)
                                       - np.asarray(locations[route[1:]], dtype=np.float64), axis=-1)
                distance = edges[:-1].sum() if open_route else edges.sum()
                actual += distance
                if "L" in problem and distance > item["route_limit"] + 1e-4:
                    raise ValueError("VRP route violates distance limit")
                if "TW" in problem:
                    clock = 0.0
                    for position, node in enumerate(route[1:-1]):
                        clock = max(clock + edges[position], item["tw_start"][node - 1])
                        if clock > item["tw_end"][node - 1] + 1e-4:
                            raise ValueError("VRP route violates a customer time window")
                        clock += item["service_time"][node - 1]
                    if not open_route and clock + edges[-1] > 3.0 + 1e-4:
                        raise ValueError("VRP route misses the depot deadline")
        if not np.isfinite(float(actual)):
            raise ValueError("Recomputed route cost is not finite")
        error = abs(float(actual) - float(reported))
        maximum_cost_error = max(maximum_cost_error, error)
        if error > 2e-4 + 2e-5 * abs(float(actual)):
            raise ValueError(f"Solver objective mismatch: reported={reported}, recomputed={actual}")
    return dict(checked_instances=len(items), feasible=True, maximum_cost_error=maximum_cost_error)


class BQBackend:
    def __init__(self, problem, method, device):
        if problem not in {"TSP", "CVRP"} or method != "BQ":
            raise ValueError("BQ supports TSP and CVRP only")
        self.problem = problem
        self.method = method
        self.device = torch.device(device)
        self.repo = SOURCE_ROOT / "bq-nco"
        sys.path.insert(0, str(self.repo))
        from model.model import BQModel
        self.checkpoint = self.repo / "pretrained_models" / f"{problem.lower()}.best"
        self.model = BQModel(2 if problem == "TSP" else 4, 192, 512, "relu", 9, 12,
                             "softmax", 0.0, False, problem.lower()).to(self.device).eval()
        checkpoint = torch.load(self.checkpoint, map_location="cpu", weights_only=False)
        self.model.load_state_dict(checkpoint["net"], strict=True)
        self.solver = importlib.import_module(f"learning.{problem.lower()}.decoding")
        torch.set_float32_matmul_precision("highest")

    def profile(self):
        return dict(backend="original_bq", method=self.method, problem=self.problem,
                    checkpoint=str(self.checkpoint), checkpoint_sha256=file_hash(self.checkpoint),
                    source_sha256=source_hashes(self.repo), adapter_source=inspect.getsource(type(self)),
                    budget=dict(beam_size=1, knns=-1, augmentation=1, decoding="greedy", precision="float32"),
                    normalization="Original coordinates and normalized demands; capacity=1; append duplicate endpoint only.")

    def run(self, items):
        if self.problem == "TSP":
            coordinates = torch.cat(items, dim=0).to(self.device)
            coordinates = torch.cat((coordinates, coordinates[:, :1]), dim=1)
        else:
            coordinates = torch.cat([torch.cat((item["depot"], item["loc"], item["depot"]), dim=1)
                                     for item in items], dim=0).to(self.device)
        distances = torch.cdist(coordinates, coordinates, compute_mode="donot_use_mm_for_euclid_dist")
        with torch.no_grad():
            if self.problem == "TSP":
                tours, scores = self.solver.decode(coordinates, distances, self.model, beam_size=1, knns=-1)
                tours = tours.cpu().tolist()
            else:
                demands = torch.cat([torch.cat((torch.zeros(1, 1), item["demand"], torch.zeros(1, 1)), dim=1)
                                     for item in items], dim=0).to(self.device)
                capacities = torch.ones(len(items), device=self.device)
                scores, tours = self.solver.decode(coordinates, distances, demands, capacities, self.model,
                                                   beam_size=1, knns=-1, make_tours=True)
                endpoint = coordinates.shape[1] - 1
                tours = [[0 if node == endpoint else node for node in tour] for tour in tours]
        costs = scores.detach().cpu().tolist()
        return InferenceResult(costs, tours, audit_tours(self.problem, items, tours, costs))


class RELDBackend:
    def __init__(self, problem, method, device):
        self.problem = problem
        self.method = method
        self.device = torch.device(device)
        self.common = relocated_module("reld_bridge_common", SCRIPT_ROOT / "reld_bridge_common.py")
        torch.set_float32_matmul_precision("highest")
        if method == "RELD_CVRP":
            if problem != "CVRP":
                raise ValueError("RELD_CVRP supports CVRP only")
            self.module = relocated_module("train100k_reld_single", SCRIPT_ROOT / "run_reld_cvrp_single_bridge.py")
            self.model, self.params = self.module.instantiate_single_cvrp_model(self.device)
            self.repo = WORKSPACE / "reld-nco-main/CVRP"
            self.checkpoint = self.repo / "weights/ReLD/model_epoch_90.pt"
        else:
            self.key = {"RELD_MTL": "reld_mtl", "RELD_MOEL": "reld_moel"}[method]
            self.model, self.params = self.common.instantiate_reld_model(self.key, self.device)
            self.env_class = self.common.get_env_class(problem)
            self.repo = WORKSPACE / "reld-nco-main/Multi-Task"
            self.checkpoint = self.common.MODEL_SPECS[self.key]["checkpoint"]

    def profile(self):
        return dict(backend="original_reld", method=self.method, problem=self.problem,
                    checkpoint=str(self.checkpoint), checkpoint_sha256=file_hash(self.checkpoint),
                    source_sha256=source_hashes(self.repo), adapter_source=inspect.getsource(type(self)),
                    params=self.params,
                    budget=dict(starts="min(N,100)" if self.method == "RELD_CVRP" else "N customers",
                                augmentation=1, sample_size=1, decoding="greedy", precision="float32"),
                    normalization="Reuse original bridge collation without changing original instance attributes.")

    def run(self, items):
        captured = []
        scale = sample_scale(self.problem, items[0])
        if self.method == "RELD_CVRP":
            environment_module = importlib.import_module("CVRPEnv")
            original_class = environment_module.CVRPEnv
        else:
            original_class = self.env_class

        class WitnessEnv(original_class):
            def __init__(environment, *arguments, **keywords):
                super().__init__(*arguments, **keywords)
                captured.append(environment)

            def step(environment, *arguments, **keywords):
                result = super().step(*arguments, **keywords)
                if result[-1]:
                    environment.witness_reward = result[1].detach()
                return result

        indices = list(range(len(items)))
        if self.method == "RELD_CVRP":
            batch, _ = self.module.collate_batch(items, indices)
            environment_module.CVRPEnv = WitnessEnv
            try:
                scores = self.module.run_no_aug_scores(self.model, batch, scale, self.device)
            finally:
                environment_module.CVRPEnv = original_class
        else:
            batch, _ = self.common.collate_problem_batch(f"{self.problem}train", items, indices)
            scores = self.common._run_no_aug_batch(self.model, WitnessEnv, batch, scale, self.device)
        environment = captured[0]
        reward = environment.witness_reward
        selected = reward.argmax(dim=1)
        tours = environment.selected_node_list[torch.arange(len(items), device=self.device), selected].cpu().tolist()
        costs = scores.tolist()
        return InferenceResult(costs, tours, audit_tours(self.problem, items, tours, costs))


class OriginalELGBackend:
    def __init__(self, problem, method, device):
        import yaml
        if method != "ELG" or problem not in {"TSP", "CVRP"}:
            raise ValueError("Original ELG supports TSP/CVRP")
        self.problem = problem
        self.method = method
        self.device = torch.device(device)
        self.repo = SOURCE_ROOT / "ELG" / problem
        self.config = yaml.safe_load((self.repo / "config.yml").read_text())
        self.params = self.config["model_params"]
        self.utils = relocated_module(f"train100k_elg_{problem}_utils", self.repo / "utils.py")
        definitions = relocated_module(f"train100k_elg_{problem}_definitions", self.repo / "models.py")
        saved_modules = {name: sys.modules.get(name) for name in ("utils", "models")}
        try:
            sys.modules["utils"] = self.utils
            sys.modules["models"] = definitions
            model_module = relocated_module(f"train100k_elg_{problem}_model", self.repo / f"{problem}Model.py")
            environment_module = relocated_module(f"train100k_elg_{problem}_environment", self.repo / f"{problem}Env.py")
        finally:
            for name, module in saved_modules.items():
                if module is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = module
        self.model = getattr(model_module, f"{problem}Model")(**self.params).to(self.device)
        self.model.decoder.add_local_policy(self.device)
        self.checkpoint = self.repo / "weights" / ("ELG*.pt" if problem == "TSP" else "ELG.pt")
        checkpoint = torch.load(self.checkpoint, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        self.model.eval().requires_grad_(False)
        self.environment_class = getattr(environment_module, f"{problem}Env")
        torch.set_float32_matmul_precision("highest")

    def profile(self):
        return dict(backend="original_elg", method=self.method, problem=self.problem,
                    checkpoint=str(self.checkpoint), checkpoint_sha256=file_hash(self.checkpoint),
                    source_sha256=source_hashes(self.repo), adapter_source=inspect.getsource(type(self)),
                    model_params=self.params,
                    budget=dict(starts="min(N,100), original configured multiple_width", augmentation=8,
                                decoding="greedy", precision="float32"),
                    normalization="Original coordinates and normalized demands; no change to input attributes.")

    def run(self, items):
        scale = sample_scale(self.problem, items[0])
        starts = min(scale, int(self.config["params"]["multiple_width"]))
        environment = self.environment_class(multi_width=starts, device=str(self.device))
        if self.problem == "TSP":
            batch = torch.cat(items, dim=0).to(self.device)
        else:
            batch = {key: torch.cat([item[key] for item in items], dim=0).to(self.device)
                     for key in ("loc", "demand", "depot")}
        with torch.no_grad():
            environment.load_random_problems(batch, aug_factor=8)
            reset_state, _, _ = environment.reset()
            self.model.pre_forward(reset_state)
            solutions, _, reward = self.utils.rollout(self.model, environment, eval_type="greedy")
        rewards = reward.reshape(8, len(items), starts)
        costs, tours = [], []
        for index in range(len(items)):
            chosen = int(rewards[:, index].reshape(-1).argmax().item())
            augmentation, start = divmod(chosen, starts)
            tours.append(solutions[augmentation * len(items) + index, start].detach().cpu().tolist())
            costs.append(float(-rewards[augmentation, index, start].item()))
        return InferenceResult(costs, tours, audit_tours(self.problem, items, tours, costs))


def native_api_class(original_class):
    class NativeAPIEnv(original_class):
        @property
        def batch_size(environment):
            return environment._batch_size if environment._batch_size is not None else torch.Size([])

        @batch_size.setter
        def batch_size(environment, value):
            environment._batch_size = torch.Size(value)

        @property
        def device(environment):
            return environment._device

        @device.setter
        def device(environment, value):
            if environment._device != torch.device(value):
                environment.to(value)

        @property
        def output_spec(environment):
            return original_class.output_spec.fget(environment)

        @output_spec.setter
        def output_spec(environment, value):
            if value is not None:
                original_class.output_spec.fset(environment, value)

        def reset(environment, tensordict=None, **kwargs):
            return environment._reset(tensordict, **kwargs)

        def step(environment, tensordict):
            return environment._step(tensordict)
    return NativeAPIEnv


def restore_decoder_batch_axis(module, arguments):
    encoded, attributes, *remaining = arguments
    expected = (*attributes.shape[:-1], encoded.shape[-1])
    if encoded.shape != expected:
        if encoded.numel() != int(np.prod(expected)):
            raise ValueError("Unexpected MVMoE decoder encoding dimensions")
        encoded = encoded.reshape(expected)
    return (encoded, attributes, *remaining)


class NativeARBackend:
    def __init__(self, problem, method, device):
        if problem not in {"TSP", "CVRP"} or method not in {"ELG", "LEHD", "OMNI"}:
            raise ValueError("Native AR backend supports ELG/LEHD/OMNI on TSP/CVRP")
        self.problem = problem
        self.method = method
        self.device = torch.device(device)
        self.repo = WORKSPACE / "EasyNCO"
        self.compat = relocated_module("train100k_native_compat", SCRIPT_ROOT / "paper_source_routefinder_runner.py")
        self.compat._patch_torchrl_compat(include_top_level=True)
        sys.path.insert(0, str(WORKSPACE))
        sys.path.insert(0, str(self.repo))
        from hydra.utils import get_class, instantiate
        from omegaconf import OmegaConf
        from EasyNCO.utils.utils import load_model, set_device

        gpu = self.device.index if self.device.index is not None else torch.cuda.current_device()
        set_device(use_cuda=True, cuda_device=[gpu], auto_configure_ddp=False, matmul_precision="highest")
        settings = OmegaConf.load(self.repo / "settings" / f"{method.lower()}_settings.yaml")
        config = OmegaConf.create(dict(problem=problem.lower(), model=method.lower(), mode="test", scale=100))
        config.settings = settings
        settings = config.settings
        if method == "ELG" and problem == "TSP":
            settings.model.local_size = 30
        self.model = instantiate(settings.model).to(self.device)
        if method == "OMNI":
            self.checkpoint = self.repo / "pretrained/omni" / f"omni_{problem.lower()}_maml_fomaml.ckpt"
        else:
            directory = "ELG" if method == "ELG" else "lehd"
            self.checkpoint = self.repo / "pretrained" / directory / f"{method.lower()}_{problem.lower()}100.ckpt"
        load_model(self.model, str(self.checkpoint), device=[gpu], model_name=method.lower())
        self.model.eval()
        self.initialization = instantiate(settings.initialization, policy=self.model)
        env_params = OmegaConf.to_container(settings.env, resolve=True)
        env_params.pop("_target_")
        env_params["method_name"] = method.lower()
        original_class = get_class(f"EasyNCO.neural_solvers.envs.{problem}Env")

        NativeAPIEnv = native_api_class(original_class)
        self.env = NativeAPIEnv(device=str(self.device), **env_params)
        self.model_params = OmegaConf.to_container(settings.model, resolve=True)
        self.env_params = env_params
        self.api_adapter_source = inspect.getsource(NativeAPIEnv)

    def profile(self):
        return dict(backend="easynco_native_initialization", method=self.method, problem=self.problem,
                    checkpoint=str(self.checkpoint), checkpoint_sha256=file_hash(self.checkpoint),
                    source_sha256={**source_hashes(self.repo / "neural_solvers"),
                                   **source_hashes(self.repo / "utils")},
                    adapter_source=inspect.getsource(type(self)), api_adapter_source=self.api_adapter_source,
                    settings_sha256=file_hash(self.repo / "settings" / f"{self.method.lower()}_settings.yaml"),
                    model_params=self.model_params, env_params=self.env_params,
                    budget=dict(starts=1 if self.method == "LEHD" else "N", augmentation=8,
                                inference_batch_size=1, decoding="greedy", iteration_steps=0,
                                precision="float32, matmul_precision=highest", score_selection="Original orientation, maximum POMO reward"),
                    normalization="Original coordinates and normalized demand; capacity=1; no rescaling.",
                    compatibility="Forward reset/step to original _reset/_step; all original masks, transitions and rewards are unchanged.")

    def run(self, items):
        costs, tours = [], []
        for item in items:
            scale = sample_scale(self.problem, item)
            if self.problem == "TSP":
                batch = item.to(self.device)
            else:
                coordinates = torch.cat((item["depot"], item["loc"]), dim=1)
                demands = torch.cat((torch.zeros_like(item["demand"][:, :1]), item["demand"]), dim=1)
                batch = torch.cat((coordinates, demands[..., None]), dim=-1).to(self.device)
            self.env.problem_size = scale
            self.env.pomo_size = 1 if self.method == "LEHD" else scale
            with torch.inference_mode():
                self.env.load_problems(batch, batch_size=1)
                _, output = self.initialization.play_episode(self.env, "greedy")
            rewards = output["reward"].reshape(self.env.aug_factor, self.env.pomo_size)
            augmentation = 0
            start = int(rewards[augmentation].argmax().item())
            selected = self.env.selected_node_list[augmentation, start].detach().cpu().tolist()
            if self.problem == "CVRP" and self.method == "LEHD":
                tour = [0]
                for node, depot_flag in selected:
                    if depot_flag and tour[-1] != 0:
                        tour.append(0)
                    tour.append(node)
                tour.append(0)
            else:
                tour = selected
            tours.append(tour)
            costs.append(float(-rewards[augmentation, start].item()))
        return InferenceResult(costs, tours, audit_tours(self.problem, items, tours, costs))


class NativeMVRPBackend:
    def __init__(self, problem, method, device):
        if problem in {"TSP", "ATSP"} or method not in {"MTPOMO", "MVMOE"}:
            raise ValueError("Native multi-task backend supports MTPOMO/MVMOE on VRP variants")
        self.problem = problem
        self.method = method
        self.device = torch.device(device)
        self.repo = WORKSPACE / "EasyNCO"
        compat = relocated_module("train100k_native_compat", SCRIPT_ROOT / "paper_source_routefinder_runner.py")
        compat._patch_torchrl_compat(include_top_level=True)
        sys.path.insert(0, str(WORKSPACE))
        sys.path.insert(0, str(self.repo))
        from hydra.utils import instantiate
        from omegaconf import OmegaConf
        from EasyNCO.neural_solvers.envs import MVRPEnv
        from EasyNCO.utils.utils import load_model, set_device

        gpu = self.device.index if self.device.index is not None else torch.cuda.current_device()
        set_device(use_cuda=True, cuda_device=[gpu], auto_configure_ddp=False, matmul_precision="medium")
        config = OmegaConf.create(dict(problem=problem, scale=100))
        config.settings = OmegaConf.load(self.repo / "settings" / f"{method.lower()}_settings.yaml")
        settings = config.settings
        self.model = instantiate(settings.model).to(self.device)
        filename = "mtpomo_mvrp_100.ckpt" if method == "MTPOMO" else "mvmoe_mvrp100.ckpt"
        self.checkpoint = self.repo / "pretrained" / method.lower() / method.lower() / filename
        load_model(self.model, str(self.checkpoint), device=[gpu], model_name=method.lower())
        self.model.eval()
        if method == "MVMOE":
            self.model.decoder.register_forward_pre_hook(restore_decoder_batch_axis)
        self.initialization = instantiate(settings.initialization, policy=self.model)
        env_params = OmegaConf.to_container(settings.env, resolve=True)
        env_params.pop("_target_")
        env_params["mode"] = 1
        self.env = native_api_class(MVRPEnv)(device=str(self.device), **env_params)
        self.model_params = OmegaConf.to_container(settings.model, resolve=True)
        self.env_params = env_params

    def profile(self):
        return dict(backend="easynco_native_multitask", problem=self.problem, method=self.method,
                    checkpoint=str(self.checkpoint), checkpoint_sha256=file_hash(self.checkpoint),
                    source_sha256={**source_hashes(self.repo / "neural_solvers"),
                                   **source_hashes(self.repo / "utils")},
                    adapter_source=inspect.getsource(type(self)), api_adapter_source=inspect.getsource(native_api_class),
                    singleton_decoder_adapter=inspect.getsource(restore_decoder_batch_axis) if self.method == "MVMOE" else None,
                    model_params=self.model_params, env_params=self.env_params,
                    budget=dict(starts="Original MVRPEnv starts: N, or min(floor(0.8*N),N) for backhaul variants",
                                augmentation=1, decoding="greedy", precision="float32, matmul_precision=medium", iteration_steps=0),
                    normalization="Original coordinates, signed normalized demands, route_limit, service times and time windows; depot TW=[0,3].")

    def batch(self, items):
        scale = sample_scale(self.problem, items[0])
        if any(sample_scale(self.problem, item) != scale for item in items):
            raise ValueError("Multi-task inference batch mixes scales")
        count = len(items)
        if self.problem == "CVRP":
            coordinates = torch.cat([torch.cat((item["depot"], item["loc"]), dim=1) for item in items]).to(self.device)
            demand = torch.cat([item["demand"] for item in items]).to(self.device)
        else:
            coordinates = torch.tensor([item["depot_xy"] + item["node_xy"] for item in items], dtype=torch.float32, device=self.device)
            demand = torch.tensor([item["node_demand"] for item in items], dtype=torch.float32, device=self.device)
        service = torch.zeros(count, scale + 1, device=self.device)
        starts = torch.zeros_like(service)
        ends = torch.zeros_like(service)
        if "TW" in self.problem:
            service[:, 1:] = torch.tensor([item["service_time"] for item in items], device=self.device)
            starts[:, 1:] = torch.tensor([item["tw_start"] for item in items], device=self.device)
            ends[:, 1:] = torch.tensor([item["tw_end"] for item in items], device=self.device)
            ends[:, 0] = 3.0
        return dict(depot_node_xy=coordinates,
                    depot_node_demand=torch.cat((torch.zeros(count, 1, device=self.device), demand), dim=1)[..., None],
                    route_limit=torch.tensor([item.get("route_limit", 0.0) for item in items], device=self.device),
                    depot_node_service_time=service, depot_node_tw_start=starts, depot_node_tw_end=ends,
                    problem_type=[self.problem] * count)

    def run(self, items):
        scale = sample_scale(self.problem, items[0])
        self.env.problem_size = scale
        self.env.tem_pomo_size = scale
        with torch.inference_mode():
            self.env.load_problems(self.batch(items), batch_size=len(items))
            _, output = self.initialization.play_episode(self.env, "greedy")
        rewards = output["reward"].reshape(len(items), self.env.pomo_size)
        selected = rewards.argmax(dim=1)
        tours = self.env.selected_node_list[torch.arange(len(items), device=self.device), selected].detach().cpu().tolist()
        costs = (-rewards.max(dim=1).values).detach().cpu().tolist()
        return InferenceResult(costs, tours, audit_tours(self.problem, items, tours, costs))


class RouteFinderBackend:
    def __init__(self, problem, method, device):
        if problem in {"TSP", "ATSP"}:
            raise ValueError("RouteFinder family supports VRP problems only")
        self.problem = problem
        self.method = method
        self.device = torch.device(device)
        self.key = {"RouteFinder": "routefinder", "MoSES_RF": "moses_rf", "MoSES_CaDA": "moses_cada"}[method]
        self.bridge = relocated_module("train100k_routefinder_bridge", SCRIPT_ROOT / "paper_source_routefinder_runner.py")
        self.pool = self.bridge.RunnerPool(str(self.device))
        self.repo = SOURCE_ROOT / ("routefinder" if method == "RouteFinder" else "moses_vrp")
        self.checkpoints = {str(bucket): self.bridge.resolve_checkpoint(self.key, bucket) for bucket in (50, 100)}
        self.validated_buckets = set()

    def profile(self):
        return dict(backend="original_routefinder_family", method=self.method, problem=self.problem,
                    checkpoints={bucket: dict(path=str(path), sha256=file_hash(path))
                                 for bucket, path in self.checkpoints.items()},
                    source_sha256=source_hashes(self.repo), adapter_source=inspect.getsource(type(self)),
                    loader_sha256=file_hash(SCRIPT_ROOT / "paper_source_routefinder_runner.py"),
                    signed_demand_adapter_sha256=file_hash(SCRIPT_ROOT / "train_100k_routefinder.py"),
                    budget=dict(starts="Original env.get_num_starts(N)", augmentation=8, augment_fn="dihedral8",
                                decoding="Original phase=test", precision="Original CUDA autocast",
                                checkpoint_bucket="50 if N<=75, otherwise 100"),
                    normalization="Preserve coordinates and demands, vehicle capacity=1, speed=1, route limit=3, depot TW=[0,3].",
                    historical_adapter_note="Historical prepared TW files used an infinite depot deadline; new labels use the unchanged dataset deadline 3.",
                    backhaul_adapter="For B variants preserve original time/distance/visited/depot masks and use the existing EasyNCO mode=1 signed-demand load constraint, rather than RouteFinder's different classical backhaul definition. Keep the N-start budget with feasible delivery starts.")

    def tensor_dict(self, items):
        from tensordict import TensorDict
        scale = sample_scale(self.problem, items[0])
        if any(sample_scale(self.problem, item) != scale for item in items):
            raise ValueError("RouteFinder requires equal-sized instances within an inference batch")
        if self.problem == "CVRP":
            coordinates = torch.cat([torch.cat((item["depot"], item["loc"]), dim=1) for item in items])
            demands = torch.cat([item["demand"] for item in items])
        else:
            coordinates = torch.tensor([item["depot_xy"] + item["node_xy"] for item in items], dtype=torch.float32)
            demands = torch.tensor([item["node_demand"] for item in items], dtype=torch.float32)
        count = len(items)
        windows = torch.zeros(count, scale + 1, 2, dtype=torch.float32)
        windows[..., 1] = float("inf")
        service = torch.zeros(count, scale + 1, dtype=torch.float32)
        if "TW" in self.problem:
            windows[:, 0, 1] = 3.0
            windows[:, 1:, 0] = torch.tensor([item["tw_start"] for item in items])
            windows[:, 1:, 1] = torch.tensor([item["tw_end"] for item in items])
            service[:, 1:] = torch.tensor([item["service_time"] for item in items])
        return TensorDict(dict(locs=coordinates.float(), demand_linehaul=demands.clamp(min=0).float(),
                               demand_backhaul=(-demands).clamp(min=0).float(),
                               distance_limit=torch.full((count, 1), 3.0 if "L" in self.problem else float("inf")),
                               time_windows=windows, service_time=service,
                               vehicle_capacity=torch.ones(count, 1), capacity_original=torch.ones(count, 1),
                               speed=torch.ones(count, 1),
                               open_route=torch.full((count, 1), self.problem.startswith("OVRP"), dtype=torch.bool)),
                          batch_size=[count])

    def run(self, items):
        scale = sample_scale(self.problem, items[0])
        bucket = 50 if scale <= 75 else 100
        runner = self.pool.get(self.key, bucket)
        if bucket not in self.validated_buckets:
            checkpoint = torch.load(self.checkpoints[str(bucket)], map_location="cpu", weights_only=False)
            state = checkpoint["state_dict"]
            weights = {key[len("policy."):]: value for key, value in state.items() if key.startswith("policy.")}
            if not weights:
                raise ValueError("The original checkpoint has no policy weights")
            runner.policy.load_state_dict(weights, strict=True)
            if "B" in self.problem:
                from train_100k_routefinder import install_signed_demand_adapter
                install_signed_demand_adapter(runner.env)
            self.validated_buckets.add(bucket)
        batch = runner.env.reset(self.tensor_dict(items)).to(self.device)
        output = runner.module.test(runner.policy, batch, runner.env, device=str(self.device))
        costs = (-output["max_aug_reward"]).detach().cpu().tolist()
        tours = output["best_aug_actions"].reshape(len(items), -1).detach().cpu().tolist()
        return InferenceResult(costs, tours, audit_tours(self.problem, items, tours, costs))


def make_backend(problem, method, device="cuda"):
    if (problem, method) == ("CVRP", "LEHD"):
        from train_100k_nss_ar import OriginalLEHDCVRPBackend
        return OriginalLEHDCVRPBackend(problem, method, device)
    if method == "OMNI":
        if problem == "TSP":
            return NativeARBackend(problem, method, device)
        from train_100k_nss_ar import OriginalOmniCVRPBackend
        return OriginalOmniCVRPBackend(problem, method, device)
    if (problem, method) == ("CVRP", "MVMOE"):
        from train_100k_nss_ar import NSSMVmoeCVRPBackend
        return NSSMVmoeCVRPBackend(problem, method, device)
    if method in {"DIFUSCO", "DIFUSCO500", "T2T", "T2T500"}:
        from train_100k_diffusion import OriginalDiffusionBackend
        return OriginalDiffusionBackend(problem, method, device)
    if method == "ICAM":
        from train_100k_icam import OriginalICAMCVRPBackend
        return OriginalICAMCVRPBackend(problem, method, device)
    if method in {"MATNET", "MATPOENET"}:
        from train_100k_atsp import NativeATSPBackend
        return NativeATSPBackend(problem, method, device)
    if method == "GLOP":
        from train_100k_atsp import GLOPATSPBackend
        return GLOPATSPBackend(problem, method, device)
    if method == "ICAM_ATSP":
        from train_100k_atsp import ICAMATSPBackend
        return ICAMATSPBackend(problem, method, device)
    if method == "UNICO_MatPOENet":
        from train_100k_atsp import UnicoATSPBackend
        return UnicoATSPBackend(problem, method, device)
    if method == "BQ":
        return BQBackend(problem, method, device)
    if method in {"RELD_CVRP", "RELD_MTL", "RELD_MOEL"}:
        return RELDBackend(problem, method, device)
    if method == "ELG":
        return OriginalELGBackend(problem, method, device)
    if method in {"LEHD", "OMNI"}:
        return NativeARBackend(problem, method, device)
    if method in {"MTPOMO", "MVMOE"}:
        return NativeMVRPBackend(problem, method, device)
    if method in {"RouteFinder", "MoSES_RF", "MoSES_CaDA"}:
        return RouteFinderBackend(problem, method, device)
    raise NotImplementedError(f"Backend not registered yet: {problem}/{method}")
