from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Protocol
import time

import torch
import torch.nn.functional as F

from checkpoints import load_policy_checkpoint
from easynco_bootstrap import ensure_local_easynco
from tsp_utils import SolutionBatch, group_indices, perm_to_succ, succ_to_perm, tsp_succ_length, tsp_tour_length


class Initializer(Protocol):
    name: str

    def solve(self, coords: torch.Tensor) -> SolutionBatch:
        ...


class Operator(Protocol):
    name: str

    def apply(self, coords: torch.Tensor, solution: SolutionBatch) -> SolutionBatch:
        ...


class TorchNoGrad:
    def __enter__(self):
        self._cm = torch.inference_mode()
        return self._cm.__enter__()

    def __exit__(self, exc_type, exc, tb):
        return self._cm.__exit__(exc_type, exc, tb)


def run_initializers_by_action(
    coords: torch.Tensor,
    actions: torch.Tensor,
    zoo: list[Initializer],
    log_fn: Callable[[str], None] | None = None,
) -> SolutionBatch:
    batch_size, problem_size = coords.size(0), coords.size(1)
    out_tour = torch.zeros((batch_size, problem_size), dtype=torch.long, device=coords.device)
    out_len = torch.zeros((batch_size,), dtype=coords.dtype, device=coords.device)
    total_t0 = time.perf_counter()
    for action_id, idx in enumerate(group_indices(actions, len(zoo))):
        if idx.numel() == 0:
            continue
        solver = zoo[action_id]
        if log_fn is not None:
            log_fn(f"[solver:init] name={solver.name} count={idx.numel()} start")
        t0 = time.perf_counter()
        sub = solver.solve(coords[idx])
        elapsed = time.perf_counter() - t0
        if log_fn is not None:
            log_fn(f"[solver:init] name={solver.name} count={idx.numel()} done elapsed={elapsed:.2f}s")
        out_tour[idx] = sub.tour
        out_len[idx] = sub.length
    if log_fn is not None:
        log_fn(f"[solver:init] all done elapsed={time.perf_counter() - total_t0:.2f}s")
    return SolutionBatch(tour=out_tour, length=out_len)


def run_operators_by_action(coords: torch.Tensor, actions: torch.Tensor, solution: SolutionBatch, zoo: list[Operator]) -> SolutionBatch:
    batch_size, problem_size = coords.size(0), coords.size(1)
    out_tour = torch.zeros((batch_size, problem_size), dtype=torch.long, device=coords.device)
    out_len = torch.zeros((batch_size,), dtype=coords.dtype, device=coords.device)
    for action_id, idx in enumerate(group_indices(actions, len(zoo))):
        if idx.numel() == 0:
            continue
        sub = zoo[action_id].apply(coords[idx], SolutionBatch(tour=solution.tour[idx], length=solution.length[idx]))
        out_tour[idx] = sub.tour
        out_len[idx] = sub.length
    return SolutionBatch(tour=out_tour, length=out_len)


class NeuralARInitializer:
    def __init__(self, name: str, env, initialization, decoder_strategy: str = "greedy", solver_device: str = "cpu"):
        self.name = name
        self.env = env
        self.initialization = initialization
        self.decoder_strategy = decoder_strategy
        self.solver_device = solver_device

    def solve(self, coords: torch.Tensor) -> SolutionBatch:
        run_device = torch.device(self.solver_device)
        coords_run = coords.detach().to(run_device)
        with TorchNoGrad():
            state_td, _ = self.initialization.run(self.env, coords_run, self.decoder_strategy, "eval")
        reward = state_td.get("reward", None)
        if reward is None:
            raise RuntimeError(f"[{self.name}] initialization did not return reward")
        selected_node_list = state_td["next"]["selected_node_list"] if "next" in state_td.keys() else state_td["selected_node_list"]
        batch_index = torch.arange(selected_node_list.size(0), device=selected_node_list.device)
        best_idx = reward.max(dim=1).indices.to(selected_node_list.device)
        best_tour = selected_node_list[batch_index, best_idx]
        best_length = (-reward[batch_index, best_idx]).to(dtype=coords.dtype)
        return SolutionBatch(
            tour=best_tour.to(device=coords.device, dtype=torch.long),
            length=best_length.to(device=coords.device, dtype=coords.dtype),
        )


class DifuscoInitializer:
    def __init__(self, name: str, model, solver_device: str = "cpu"):
        self.name = name
        self.model = model
        self.solver_device = solver_device

    def solve(self, coords: torch.Tensor) -> SolutionBatch:
        run_device = torch.device(self.solver_device)
        coords_run = coords.detach().to(run_device)
        batch_size, problem_size, _ = coords_run.shape
        out_tour = torch.zeros((batch_size, problem_size), dtype=torch.long, device=run_device)
        out_len = torch.zeros((batch_size,), dtype=coords_run.dtype, device=run_device)
        self.model.eval()
        with torch.no_grad():
            for i in range(batch_size):
                nodes = coords_run[i : i + 1]
                batch_idx = torch.zeros((1,), dtype=torch.long, device=run_device)
                adj = torch.zeros((1, problem_size, problem_size), dtype=coords_run.dtype, device=run_device)
                gt_tour = torch.arange(problem_size, dtype=torch.long, device=run_device)[None, :]
                stacked_policy_dict = self.model.initialize((batch_idx, nodes, adj, gt_tour))
                candidates: list[list[int]] = []
                for item in stacked_policy_dict:
                    tours = item.get("tours", None)
                    if tours is None:
                        continue
                    for tour_like in tours:
                        seq = [int(x) for x in tour_like]
                        if len(seq) == problem_size + 1 and seq[0] == seq[-1]:
                            seq = seq[:-1]
                        if len(seq) != problem_size:
                            continue
                        if len(set(seq)) != problem_size:
                            continue
                        candidates.append(seq)
                if not candidates:
                    raise RuntimeError(f"[{self.name}] DIFUSCO produced no candidate tour")
                cand = torch.tensor(candidates, dtype=torch.long, device=run_device)
                cand_len = tsp_tour_length(nodes.expand(cand.size(0), -1, -1), cand)
                best_k = int(cand_len.argmin().item())
                out_tour[i] = cand[best_k]
                out_len[i] = cand_len[best_k]
        return SolutionBatch(
            tour=out_tour.to(device=coords.device, dtype=torch.long),
            length=out_len.to(device=coords.device, dtype=coords.dtype),
        )


class HeuristicInsertionInitializer:
    def __init__(
        self,
        name: str,
        strategy: str,
        restarts: int = 4,
        metric_strategy: str = "cartesian",
        metric_p: int = 2,
        regret_k: int = 1,
        solver_device: str = "cpu",
    ):
        self.name = name
        self.strategy = strategy
        self.restarts = max(int(restarts), 1)
        self.metric_strategy = metric_strategy
        self.metric_p = int(metric_p)
        self.regret_k = int(regret_k)
        self.solver_device = solver_device

    def _compute_metric(self, coords: torch.Tensor) -> torch.Tensor:
        if self.metric_strategy == "cartesian":
            return torch.cdist(coords, coords, p=float(self.metric_p))
        if self.metric_strategy == "polar":
            origins = coords[:, [0]]
            rel = coords - origins
            angles = torch.atan2(rel[:, :, 1], rel[:, :, 0])
            diff = torch.abs(angles[:, :, None] - angles[:, None, :])
            return torch.minimum(diff, 2 * torch.pi - diff)
        raise ValueError(f"Unknown heuristic metric strategy: {self.metric_strategy}")

    def _construct_single(self, distance_matrix: torch.Tensor, metric_matrix: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        problem_size = int(distance_matrix.size(0))
        device = distance_matrix.device
        unvisited = torch.ones(problem_size, device=device, dtype=torch.bool)
        start_node = int(torch.randint(0, problem_size, (1,), device=device).item())
        unvisited[start_node] = False
        tour = [start_node, start_node]

        while bool(unvisited.any().item()):
            mask = unvisited.nonzero(as_tuple=False).flatten()
            prev_cities = torch.tensor(tour[:-1], device=device, dtype=torch.long)
            next_cities = torch.tensor(tour[1:], device=device, dtype=torch.long)

            if self.strategy == "regret":
                insertion_costs = (
                    distance_matrix[mask][:, prev_cities]
                    + distance_matrix[mask][:, next_cities]
                    - distance_matrix[prev_cities, next_cities].unsqueeze(0)
                )
                sorted_costs, sorted_indices = torch.sort(insertion_costs, dim=1)
                best_costs = sorted_costs[:, 0]
                if insertion_costs.size(1) <= 1:
                    regret_values = torch.zeros_like(best_costs)
                else:
                    alt_idx = min(max(self.regret_k, 1), insertion_costs.size(1) - 1)
                    regret_values = sorted_costs[:, alt_idx] - best_costs
                selected_row = int(torch.argmax(regret_values).item())
                node_to_insert = int(mask[selected_row].item())
                best_pos = int(sorted_indices[selected_row, 0].item()) + 1
            else:
                current_tour = torch.tensor(tour, device=device, dtype=torch.long)
                dists_to_tour = metric_matrix[mask][:, current_tour].min(dim=1).values
                if self.strategy == "nearest":
                    selected_row = int(torch.argmin(dists_to_tour).item())
                elif self.strategy == "random":
                    selected_row = int(torch.randint(0, len(mask), (1,), device=device).item())
                else:
                    raise ValueError(f"Unknown heuristic insertion strategy: {self.strategy}")
                node_to_insert = int(mask[selected_row].item())
                insertion_costs = (
                    distance_matrix[prev_cities, node_to_insert]
                    + distance_matrix[node_to_insert, next_cities]
                    - distance_matrix[prev_cities, next_cities]
                )
                best_pos = int(torch.argmin(insertion_costs).item()) + 1

            tour.insert(best_pos, node_to_insert)
            unvisited[node_to_insert] = False

        tour_tensor = torch.tensor(tour[:-1], device=device, dtype=torch.long)
        next_tensor = tour_tensor.roll(shifts=-1, dims=0)
        length = distance_matrix[tour_tensor, next_tensor].sum()
        return tour_tensor, length

    def solve(self, coords: torch.Tensor) -> SolutionBatch:
        run_device = torch.device(self.solver_device)
        coords_run = coords.detach().to(run_device)
        batch_size, problem_size, _ = coords_run.shape
        distance_matrix_batch = torch.cdist(coords_run, coords_run, p=2)
        metric_batch = self._compute_metric(coords_run)

        out_tour = torch.empty((batch_size, problem_size), dtype=torch.long, device=run_device)
        out_len = torch.empty((batch_size,), dtype=coords_run.dtype, device=run_device)
        target = torch.arange(problem_size, device=run_device, dtype=torch.long)

        with TorchNoGrad():
            for batch_idx in range(batch_size):
                best_tour = None
                best_len = None
                for _ in range(self.restarts):
                    cand_tour, cand_len = self._construct_single(distance_matrix_batch[batch_idx], metric_batch[batch_idx])
                    if best_len is None or cand_len < best_len:
                        best_tour = cand_tour
                        best_len = cand_len
                if best_tour is None or best_len is None:
                    raise RuntimeError(f"[{self.name}] failed to build a heuristic tour")
                if not torch.equal(best_tour.sort().values, target):
                    raise RuntimeError(f"[{self.name}] produced an invalid tour")
                out_tour[batch_idx] = best_tour
                out_len[batch_idx] = best_len

        recalculated = tsp_tour_length(coords_run, out_tour).to(coords_run.dtype)
        if not torch.allclose(out_len, recalculated, atol=1e-5, rtol=1e-5):
            raise RuntimeError(f"[{self.name}] heuristic length mismatch with tsp_tour_length")

        return SolutionBatch(
            tour=out_tour.to(device=coords.device, dtype=torch.long),
            length=recalculated.to(device=coords.device, dtype=coords.dtype),
        )


class TwoOptOperator:
    def __init__(self, name: str = "two_opt", max_iterations: int = 1, device: str = "cpu"):
        self.name = name
        self.max_iterations = int(max_iterations)
        self.device = device

    def apply(self, coords: torch.Tensor, solution: SolutionBatch) -> SolutionBatch:
        import numpy as np

        ensure_local_easynco()
        from EasyNCO.neural_solvers.methods.difusco.util import two_opt_refine

        nodes_np = coords.detach().cpu().numpy()
        tour0_np = solution.tour.detach().cpu().numpy()
        refined_np = np.empty_like(tour0_np)
        for i in range(tour0_np.shape[0]):
            closed = np.concatenate([tour0_np[i], tour0_np[i][:1]], axis=0)[None, :]
            refined_i, _ = two_opt_refine(nodes_np[i].astype("float64", copy=False), closed, max_iteration=self.max_iterations, device=self.device)
            refined_np[i] = refined_i[0, :-1]
        tour = torch.from_numpy(refined_np).to(device=coords.device, dtype=torch.long)
        length = tsp_tour_length(coords, tour).to(coords.dtype)
        return SolutionBatch(tour=tour, length=length)


class LEHDRRCStepOperator:
    def __init__(self, name: str, env, policy, decoder_strategy: str = "greedy", solver_device: str = "cpu"):
        self.name = name
        self.env = env
        self.policy = policy
        self.decoder_strategy = decoder_strategy
        self.solver_device = solver_device

    def apply(self, coords: torch.Tensor, solution: SolutionBatch) -> SolutionBatch:
        run_device = torch.device(self.solver_device)
        coords_run = coords.detach().to(run_device)
        solution_run = SolutionBatch(
            tour=solution.tour.detach().to(run_device),
            length=solution.length.detach().to(run_device),
        )
        with TorchNoGrad():
            self.policy.eval()
            if self.policy.decoder_strategy is None:
                self.policy.set_decoder_strategy(self.decoder_strategy)
            best_selected_node_list = solution_run.tour[:, None, :]
            batch_size = coords_run.size(0)
            problem_size = coords_run.size(1)
            self.env.load_problems(coords_run, batch_size)
            partial_len, first_node_index, subpath_length, solution_copy = self._destroy_solution(coords_run, best_selected_node_list)
            current_step = 0
            reset_td = self.env.reset()
            self.policy.pre_forward(reset_td)
            state_td = self.env.pre_step()
            done = False
            next_td = state_td
            while not done:
                if current_step == 0:
                    next_td["action"] = self.env.solution[:, :, -1]
                elif current_step == 1:
                    next_td["action"] = self.env.solution[:, :, 0]
                else:
                    next_td = self.policy(state_td)
                current_step += 1
                state_td = self.env.step(next_td)
                done = state_td["done"].all()
            repaired_sub = torch.roll(self.env.selected_node_list, shifts=-1, dims=2)
            repaired_len = -state_td["reward"]
            repaired_full = self._accept_repaired_solution(repaired_sub, partial_len, repaired_len, first_node_index, subpath_length, solution_copy)
            tour = repaired_full.squeeze(1)
            length = self.env._get_travel_distance(problems=coords_run, selected_node_list=repaired_full, batch_size=batch_size, pomo_size=1).squeeze(1)
            if not ((tour.sort(dim=1).values == torch.arange(problem_size, device=tour.device).view(1, -1)).all(dim=1)).all():
                raise RuntimeError("LEHD RRC step produced an invalid tour")
            return SolutionBatch(
                tour=tour.to(device=coords.device, dtype=torch.long),
                length=length.to(device=coords.device, dtype=coords.dtype),
            )

    def _destroy_solution(self, coords: torch.Tensor, solution_3d: torch.Tensor):
        destroyed_problem, destroyed_solution, first_node_index, subpath_length, solution_copy = self._sampling_subpaths(coords, solution_3d, repair=True)
        self.env.solution = destroyed_solution
        self.env.problems = destroyed_problem
        partial_length = self.env._get_travel_distance(selected_node_list=destroyed_solution, problems=destroyed_problem)
        return partial_length, first_node_index, subpath_length, solution_copy

    def _sampling_subpaths(self, coords: torch.Tensor, solution: torch.Tensor, repair: bool = False):
        batch_size, problem_size, embedding_size = coords.shape
        first_node_index = int(torch.randint(low=0, high=problem_size, size=(1,), device=coords.device).item())
        subpath_length = int(torch.randint(low=4, high=problem_size + 1, size=(1,), device=coords.device).item())
        solution_copy = torch.cat([solution, solution], dim=-1)
        new_solution = solution_copy[:, :, first_node_index : first_node_index + subpath_length]
        _, rank = torch.sort(new_solution, dim=-1, descending=False)
        _, new_solution_rank = torch.sort(rank, dim=-1, descending=False)
        new_solution_copy = torch.cat([new_solution, new_solution], dim=-1).long()
        sorted_new_solution_index_pomo, _ = new_solution_copy.sort(dim=-1, descending=False)
        sorted_new_solution_index = sorted_new_solution_index_pomo.squeeze(1)
        global_index = torch.arange(batch_size, dtype=torch.long, device=coords.device)[:, None].expand(batch_size, sorted_new_solution_index.shape[1])
        node_index = torch.arange(embedding_size, dtype=torch.long, device=coords.device)[None, :].expand(batch_size, embedding_size)
        coordinate_index = node_index.repeat([1, subpath_length])
        new_problem = coords[global_index, sorted_new_solution_index, coordinate_index].view(batch_size, subpath_length, 2)
        if repair:
            return new_problem, new_solution_rank, first_node_index, subpath_length, solution_copy
        raise NotImplementedError

    def _accept_repaired_solution(self, repaired_solution, pre_length, after_repair_length, first_node_index, subpath_length, solution_copy):
        problem_size = solution_copy.shape[-1] // 2
        part1 = solution_copy[:, :, :first_node_index]
        part2 = solution_copy[:, :, first_node_index + subpath_length :]
        origin_sub_solution = solution_copy[:, :, first_node_index : first_node_index + subpath_length]
        sorted_solution_index, _ = torch.sort(origin_sub_solution, dim=2, descending=False)
        repaired_solution_index = sorted_solution_index.gather(dim=2, index=repaired_solution)
        if_repair = pre_length > after_repair_length
        solution_copy = solution_copy.clone()
        solution_copy[if_repair] = torch.cat((part1[if_repair], repaired_solution_index[if_repair], part2[if_repair]), dim=1)
        return solution_copy[:, :, first_node_index : first_node_index + problem_size]


class DACTTwoOptOperator:
    def __init__(self, name: str, policy, solver_device: str = "cpu", do_sample: bool = False):
        self.name = name
        self.policy = policy
        self.solver_device = solver_device
        self.do_sample = do_sample

    def _prepare_tsp_runtime(self, problem_size: int, device: torch.device) -> None:
        if getattr(self.policy, "env_name", None) != "tsp":
            return
        self.policy.problem_size = int(problem_size)
        self.policy.problem_size_extend = int(problem_size)
        self.policy.dummy_size = 0
        pattern = self.policy.encoder.embedder.Cyclic_Positional_Encoding(int(problem_size), self.policy.embed_dim)
        self.policy.encoder.embedder.pattern = pattern.to(device)

    def _forward_tsp_policy(self, td):
        bs, gs, _ = td["locs"].size()
        device = td["locs"].device
        h_em, g_em = self.policy.encoder(td)
        masks = torch.eye(gs, device=device, dtype=torch.bool).view(1, gs, gs).expand(bs, gs, gs)
        compatibility = torch.tanh(self.policy.decoder(h_em, g_em)) * self.policy.logit_clipping
        compatibility = compatibility.masked_fill(masks, -1e20)

        if not torch.all(td["exchange"] == 0).item():
            batch_index = torch.arange(bs, device=device)
            compatibility[batch_index, td["exchange"][:, 0], td["exchange"][:, 1]] = -1e20
            compatibility[batch_index, td["exchange"][:, 1], td["exchange"][:, 0]] = -1e20

        logits = compatibility.view(bs, -1)
        log_likelihood = F.log_softmax(logits, dim=-1)
        probs = F.softmax(logits, dim=-1)
        if self.do_sample:
            pair_index = probs.multinomial(1)
        else:
            pair_index = probs.max(dim=-1).indices.view(-1, 1)
        row_selected = pair_index // gs
        col_selected = pair_index % gs
        pair = torch.cat((row_selected, col_selected), dim=-1)
        td.update({
            "exchange": pair,
            "log_likelihood": log_likelihood.gather(1, pair_index).squeeze(-1),
        })
        return td

    def apply(self, coords: torch.Tensor, solution: SolutionBatch) -> SolutionBatch:
        ensure_local_easynco()
        from tensordict import TensorDict
        from EasyNCO.neural_solvers.methods.dact.iteration import Iteration_tool

        run_device = torch.device(self.solver_device)
        coords_run = coords.detach().to(run_device)
        succ0 = perm_to_succ(solution.tour.detach().to(run_device))
        self._prepare_tsp_runtime(problem_size=succ0.size(1), device=run_device)
        length0 = tsp_succ_length(coords_run, succ0).to(coords_run.dtype)
        tool = Iteration_tool(coords_run, "tsp")
        td = TensorDict(
            {
                "solution": succ0,
                "current_length": length0,
                "pre_length": length0,
                "exchange": torch.zeros((coords_run.size(0), 2), dtype=torch.long, device=run_device),
                "locs": coords_run,
            },
            batch_size=torch.Size([coords_run.size(0)]),
        )
        with TorchNoGrad():
            self.policy.eval()
            if getattr(self.policy, "env_name", None) == "tsp":
                td = self._forward_tsp_policy(td)
            else:
                td = self.policy(td, do_sample=self.do_sample)
            td = tool.operate(td)
        succ1 = td["solution"]
        tour1 = succ_to_perm(succ1, start_node=0).to(device=coords.device, dtype=torch.long)
        length1 = td["current_length"].to(device=coords.device, dtype=coords.dtype)
        return SolutionBatch(tour=tour1, length=length1)


class LIHTwoOptOperator:
    def __init__(self, name: str, policy):
        self.name = name
        self.policy = policy

    def apply(self, coords: torch.Tensor, solution: SolutionBatch) -> SolutionBatch:
        ensure_local_easynco()
        from tensordict import TensorDict
        from EasyNCO.neural_solvers.methods.lih.iteration import Iteration_tool

        if coords.device.type != "cuda":
            raise RuntimeError("LIH two-opt step requires CUDA because upstream code hard-codes .cuda().")
        tool = Iteration_tool(coords, "tsp")
        length0 = tsp_tour_length(coords, solution.tour).to(coords.dtype)
        td = TensorDict(
            {
                "solution": solution.tour,
                "current_length": length0,
                "pre_length": length0,
                "exchange": torch.zeros((coords.size(0), 2), dtype=torch.long, device=coords.device),
                "locs": coords,
            },
            batch_size=torch.Size([coords.size(0)]),
        )
        with TorchNoGrad():
            self.policy.eval()
            td["input_info"], td["pos_enc"] = self.policy.get_input_and_pe(td)
            td = self.policy(td)
            td = tool.operate(td)
        return SolutionBatch(tour=td["solution"].to(dtype=torch.long), length=td["current_length"].to(dtype=coords.dtype))


class GLOPSubproblemOperator:
    def __init__(
        self,
        name: str,
        env,
        policy,
        revision_len: int,
        revision_iters: int = 1,
        decoder_strategy: str = "greedy",
        solver_device: str = "cpu",
    ):
        self.name = name
        self.env = env
        self.policy = policy
        self.revision_len = int(revision_len)
        self.revision_iters = int(revision_iters)
        self.decoder_strategy = decoder_strategy
        self.solver_device = solver_device

    def _apply_revision(self, coords: torch.Tensor, tour: torch.Tensor, revision_len: int) -> torch.Tensor:
        ensure_local_easynco()
        from EasyNCO.neural_solvers.methods.glop.sub_tsp_utils import tsp_decompose_and_solve

        if revision_len <= 1 or revision_len > coords.size(1):
            return tour
        model_key = f"sub_{revision_len}"
        if model_key not in self.policy.lower_model:
            raise KeyError(f"GLOP lower model '{model_key}' not found in checkpoint-loaded policy.")
        self.env.problem_size = revision_len
        return tsp_decompose_and_solve(
            problem=coords,
            solution=tour,
            policy=self.policy.lower_model[model_key],
            env=self.env,
            revision_len=revision_len,
            iter=self.revision_iters,
            decoder_strategy=self.decoder_strategy,
        )

    def apply(self, coords: torch.Tensor, solution: SolutionBatch) -> SolutionBatch:
        run_device = torch.device(self.solver_device)
        coords_run = coords.detach().to(run_device)
        tour_run = solution.tour.detach().to(run_device)
        with TorchNoGrad():
            self.policy.eval()
            revised_tour = self._apply_revision(coords_run, tour_run.clone(), self.revision_len)
        length = tsp_tour_length(coords, revised_tour.to(coords.device)).to(coords.dtype)
        return SolutionBatch(tour=revised_tour.to(device=coords.device, dtype=torch.long), length=length)


class GLOPPassOperator(GLOPSubproblemOperator):
    def __init__(
        self,
        name: str,
        env,
        policy,
        revision_lens: tuple[int, ...] = (100, 50, 20),
        revision_iters: tuple[int, ...] = (1, 1, 1),
        decoder_strategy: str = "greedy",
    ):
        super().__init__(
            name=name,
            env=env,
            policy=policy,
            revision_len=max(revision_lens),
            revision_iters=1,
            decoder_strategy=decoder_strategy,
        )
        self.revision_lens = tuple(int(x) for x in revision_lens)
        self.revision_iters_seq = tuple(int(x) for x in revision_iters)
        if len(self.revision_lens) != len(self.revision_iters_seq):
            raise ValueError("revision_lens and revision_iters must have the same length.")

    def apply(self, coords: torch.Tensor, solution: SolutionBatch) -> SolutionBatch:
        run_device = torch.device(self.solver_device)
        coords_run = coords.detach().to(run_device)
        revised_tour = solution.tour.detach().to(run_device).clone()
        with TorchNoGrad():
            self.policy.eval()
            for revision_len, revision_iters in zip(self.revision_lens, self.revision_iters_seq):
                if revision_len > coords_run.size(1):
                    continue
                old_iters = self.revision_iters
                self.revision_iters = revision_iters
                revised_tour = self._apply_revision(coords_run, revised_tour, revision_len)
                self.revision_iters = old_iters
        length = tsp_tour_length(coords, revised_tour.to(coords.device)).to(coords.dtype)
        return SolutionBatch(tour=revised_tour.to(device=coords.device, dtype=torch.long), length=length)


def _auto_find_ckpt(base_dir: str | None, keyword: str, exts: tuple[str, ...] = (".ckpt", ".pt", ".pkl")) -> str | None:
    if base_dir is None:
        return None
    root = Path(base_dir)
    if not root.exists():
        return None
    keyword_l = keyword.lower()
    candidates = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in exts and keyword_l in p.name.lower()]
    candidates.sort()
    return str(candidates[0]) if candidates else None


def _require_ckpt(path: str | None, name: str) -> str:
    if path is None:
        raise FileNotFoundError(f"No checkpoint found for '{name}'. Please provide an explicit path.")
    return path


def build_initializer_zoo(cfg, log=print) -> list[Initializer]:
    ensure_local_easynco()
    from EasyNCO.neural_solvers.envs.TSPEnv import TSPEnv

    zoo: list[Initializer] = []
    pomo_size = getattr(cfg, "pomo_size", None) or cfg.problem_size
    solver_device = getattr(cfg, "easynco_solver_device", "cpu")

    for name in cfg.init_zoo:
        if name == "pomo":
            from EasyNCO.neural_solvers.methods.pomo.initialization import POMOInitialization
            from EasyNCO.neural_solvers.methods.pomo.policy import POMOPolicy
            env = TSPEnv(problem_size=cfg.problem_size, pomo_size=pomo_size, device=solver_device, aug_type=None, aug_factor=1)
            policy = POMOPolicy(env_name="tsp").to(solver_device)
            ckpt = cfg.pomo_ckpt or _auto_find_ckpt(cfg.solver_ckpt_dir, "pomo")
            if ckpt is not None:
                rep = load_policy_checkpoint(policy, ckpt, device=solver_device)
                log(f"[solver_load] pomo: {rep['path']} missing={len(rep['missing_keys'])} unexpected={len(rep['unexpected_keys'])}")
            else:
                log("[solver_load] pomo: no checkpoint found, using random weights")
            policy.requires_grad_(False)
            zoo.append(NeuralARInitializer(name="pomo", env=env, initialization=POMOInitialization(policy=policy), decoder_strategy=cfg.decoder_strategy, solver_device=solver_device))
        elif name == "am":
            from EasyNCO.neural_solvers.methods.am.policy import AttentionModelPolicy
            from EasyNCO.neural_solvers.pipeline.initialization import ARInitialization
            env = TSPEnv(problem_size=cfg.problem_size, pomo_size=1, device=solver_device, aug_type=None, aug_factor=1)
            policy = AttentionModelPolicy(env_name="tsp").to(solver_device)
            ckpt = cfg.am_ckpt or _auto_find_ckpt(cfg.solver_ckpt_dir, "am")
            if ckpt is not None:
                rep = load_policy_checkpoint(policy, ckpt, device=solver_device)
                log(f"[solver_load] am: {rep['path']} missing={len(rep['missing_keys'])} unexpected={len(rep['unexpected_keys'])}")
            else:
                log("[solver_load] am: no checkpoint found, using random weights")
            policy.requires_grad_(False)
            zoo.append(NeuralARInitializer(name="am", env=env, initialization=ARInitialization(policy=policy), decoder_strategy=cfg.decoder_strategy, solver_device=solver_device))
        elif name == "lehd":
            from EasyNCO.neural_solvers.methods.lehd.initialization import LEHDInitialization
            from EasyNCO.neural_solvers.methods.lehd.policy import LEHDPolicy
            env = TSPEnv(problem_size=cfg.problem_size, pomo_size=1, device=solver_device, aug_type=None, aug_factor=1, method_name="lehd")
            policy = LEHDPolicy(phase="test", env_name="tsp").to(solver_device)
            ckpt = _require_ckpt(cfg.lehd_ckpt or _auto_find_ckpt(cfg.solver_ckpt_dir, "lehd"), "lehd")
            rep = load_policy_checkpoint(policy, ckpt, device=solver_device)
            log(f"[solver_load] lehd: {rep['path']} missing={len(rep['missing_keys'])} unexpected={len(rep['unexpected_keys'])}")
            policy.requires_grad_(False)
            zoo.append(NeuralARInitializer(name="lehd", env=env, initialization=LEHDInitialization(policy=policy), decoder_strategy=cfg.decoder_strategy, solver_device=solver_device))
        elif name == "elg":
            from EasyNCO.neural_solvers.methods.elg.initialization import ELGInitialization
            from EasyNCO.neural_solvers.methods.elg.policy import ELGPolicy
            env = TSPEnv(problem_size=cfg.problem_size, pomo_size=pomo_size, device=solver_device, aug_type=None, aug_factor=1, method_name="elg")
            policy = ELGPolicy(
                env_name="tsp",
                embed_dim=128,
                num_heads=8,
                qkv_dim=16,
                num_encoder_layers=6,
                normalization="instance",
                feedforward_hidden=512,
                logit_clipping=50,
                use_graph_mean=False,
                am_mode=False,
                first_placeholder=False,
                first_mode="random",
                local_enable=False,
                local_size=40,
                xi=-1,
                euclidean=False,
            ).to(solver_device)
            ckpt = _require_ckpt(cfg.elg_ckpt or _auto_find_ckpt(cfg.solver_ckpt_dir, "elg"), "elg")
            rep = load_policy_checkpoint(policy, ckpt, device=solver_device)
            log(f"[solver_load] elg: {rep['path']} missing={len(rep['missing_keys'])} unexpected={len(rep['unexpected_keys'])}")
            policy.requires_grad_(False)
            zoo.append(NeuralARInitializer(name="elg", env=env, initialization=ELGInitialization(policy=policy), decoder_strategy=cfg.decoder_strategy, solver_device=solver_device))
        elif name == "difusco":
            from EasyNCO.neural_solvers.methods.difusco.policy import DIFUSCOPolicy
            policy = DIFUSCOPolicy(
                env_name="tsp",
                n_layers=12,
                hidden_dim=256,
                aggregation="sum",
                diffusion_type="categorical",
                diffusion_schedule="linear",
                diffusion_steps=1000,
                sparse_factor=-1,
                use_activation_checkpoint=False,
                parallel_sampling=1,
                sequential_sampling=1,
                inference_diffusion_steps=20,
                inference_schedule="cosine",
                inference_trick="ddim",
                node_feature_only=False,
            )
            model = policy._get_model().to(solver_device)
            ckpt = _require_ckpt(cfg.difusco_ckpt or _auto_find_ckpt(cfg.solver_ckpt_dir, "difusco"), "difusco")
            rep = load_policy_checkpoint(policy, ckpt, device=solver_device)
            log(f"[solver_load] difusco: {rep['path']} missing={len(rep['missing_keys'])} unexpected={len(rep['unexpected_keys'])}")
            model.requires_grad_(False)
            zoo.append(DifuscoInitializer(name="difusco", model=model, solver_device=solver_device))
        elif name in {"ins_nearest", "ins_random", "ins_regret"}:
            strategy = {
                "ins_nearest": "nearest",
                "ins_random": "random",
                "ins_regret": "regret",
            }[name]
            zoo.append(
                HeuristicInsertionInitializer(
                    name=name,
                    strategy=strategy,
                    restarts=getattr(cfg, "heuristic_init_restarts", 4),
                    metric_strategy=getattr(cfg, "heuristic_metric_strategy", "cartesian"),
                    metric_p=getattr(cfg, "heuristic_metric_p", 2),
                    regret_k=getattr(cfg, "heuristic_regret_k", 1),
                    solver_device=solver_device,
                )
            )
        else:
            raise ValueError(f"Unknown initializer '{name}'")
    return zoo


def build_operator_zoo(cfg, log=print) -> list[Operator]:
    ensure_local_easynco()
    from EasyNCO.neural_solvers.envs.TSPEnv import TSPEnv
    from EasyNCO.neural_solvers.methods.dact.policy import DACTPolicy
    from EasyNCO.neural_solvers.methods.lehd.policy import LEHDPolicy
    from EasyNCO.neural_solvers.methods.lih.policy import LIHPolicy

    solver_device = getattr(cfg, "easynco_solver_device", "cpu")
    glop_bundle = None

    def get_glop_bundle():
        nonlocal glop_bundle
        if glop_bundle is not None:
            return glop_bundle
        from EasyNCO.neural_solvers.methods.glop.policy import GLOPPolicy

        env = TSPEnv(
            problem_size=min(cfg.problem_size, 100),
            pomo_size=cfg.glop_pomo_size,
            device=solver_device,
            aug_type="pomo_aug",
            aug_factor=cfg.glop_aug_factor,
            method_name="glop",
        )
        policy = GLOPPolicy(
            problem="tsp",
            global_params={
                "width": 1,
                "n_subset": 1,
                "revision_lens": [100, 50, 20],
                "iter": [cfg.glop_revision_iters, cfg.glop_revision_iters, cfg.glop_revision_iters],
            },
            lower={
                "env_name": "tsp",
                "embed_dim": 128,
                "num_heads": 8,
                "qkv_dim": 16,
                "num_encoder_layers": 6,
                "normalization": "batch",
                "feedforward_hidden": 512,
                "logit_clipping": 10,
                "use_graph_mean": True,
                "am_mode": False,
                "first_placeholder": True,
                "sub_glop": True,
                "bias": False,
                "bias_k": False,
                "bias_v": False,
                "bias_combine": False,
            },
        ).to(solver_device)
        ckpt = _require_ckpt(
            cfg.glop_ckpt
            or _auto_find_ckpt(cfg.solver_ckpt_dir, "glop_policy_tsp")
            or _auto_find_ckpt(cfg.solver_ckpt_dir, "glop"),
            "glop",
        )
        rep = load_policy_checkpoint(policy, ckpt, device=solver_device)
        log(f"[solver_load] glop: {rep['path']} missing={len(rep['missing_keys'])} unexpected={len(rep['unexpected_keys'])}")
        policy.requires_grad_(False)
        glop_bundle = (policy, env)
        return glop_bundle

    zoo: list[Operator] = []
    for name in cfg.operator_zoo:
        if name == "two_opt":
            zoo.append(TwoOptOperator(name="two_opt", max_iterations=cfg.two_opt_iterations, device=cfg.dact_solver_device))
        # elif name == "difusco_2opt_step":
        #     zoo.append(TwoOptOperator(name="difusco_2opt_step", max_iterations=cfg.two_opt_iterations, device=cfg.dact_solver_device))
        elif name == "lehd_rrc_step":
            env = TSPEnv(problem_size=cfg.problem_size, pomo_size=1, device=solver_device, aug_type=None, aug_factor=1, method_name="lehd")
            policy = LEHDPolicy(phase="test", env_name="tsp").to(solver_device)
            ckpt = _require_ckpt(cfg.lehd_ckpt or _auto_find_ckpt(cfg.solver_ckpt_dir, "lehd"), "lehd")
            rep = load_policy_checkpoint(policy, ckpt, device=solver_device)
            log(f"[solver_load] lehd_rrc_step: {rep['path']} missing={len(rep['missing_keys'])} unexpected={len(rep['unexpected_keys'])}")
            policy.requires_grad_(False)
            zoo.append(LEHDRRCStepOperator(name="lehd_rrc_step", env=env, policy=policy, decoder_strategy=cfg.decoder_strategy, solver_device=solver_device))
        elif name == "dact_2opt_step":
            dact_solver_device = getattr(cfg, "dact_solver_device", "cpu")
            policy = DACTPolicy(env_name="tsp")
            policy.initial_problem_size(cfg.problem_size)
            policy.to(dact_solver_device)
            ckpt = _require_ckpt(cfg.dact_ckpt or _auto_find_ckpt(cfg.solver_ckpt_dir, "dact"), "dact")
            rep = load_policy_checkpoint(policy, ckpt, device=dact_solver_device)
            log(f"[solver_load] dact_2opt_step: {rep['path']} missing={len(rep['missing_keys'])} unexpected={len(rep['unexpected_keys'])}")
            policy.requires_grad_(False)
            zoo.append(DACTTwoOptOperator(name="dact_2opt_step", policy=policy, solver_device=dact_solver_device, do_sample=False))
        elif name == "lih_2opt_step":
            lih_solver_device = getattr(cfg, "easynco_solver_device", cfg.device)
            if not str(lih_solver_device).startswith("cuda"):
                raise RuntimeError("'lih_2opt_step' requires a CUDA device because upstream LIH hard-codes .cuda().")
            policy = LIHPolicy(env_name="tsp").to(lih_solver_device)
            ckpt = _require_ckpt(cfg.lih_ckpt or _auto_find_ckpt(cfg.solver_ckpt_dir, "lih"), "lih")
            rep = load_policy_checkpoint(policy, ckpt, device=lih_solver_device)
            log(f"[solver_load] lih_2opt_step: {rep['path']} missing={len(rep['missing_keys'])} unexpected={len(rep['unexpected_keys'])}")
            policy.requires_grad_(False)
            zoo.append(LIHTwoOptOperator(name="lih_2opt_step", policy=policy))
        elif name in {"glop_sub100_step", "glop_sub50_step", "glop_sub20_step", "glop_pass_step"}:
            policy, env = get_glop_bundle()
            if name == "glop_pass_step":
                zoo.append(
                    GLOPPassOperator(
                        name="glop_pass_step",
                        env=env,
                        policy=policy,
                        revision_lens=(100, 50, 20),
                        revision_iters=(cfg.glop_revision_iters, cfg.glop_revision_iters, cfg.glop_revision_iters),
                        decoder_strategy=cfg.decoder_strategy,
                        solver_device=solver_device,
                    )
                )
            else:
                revision_len = int(name.split("sub")[1].split("_")[0])
                zoo.append(
                    GLOPSubproblemOperator(
                        name=name,
                        env=env,
                        policy=policy,
                        revision_len=revision_len,
                        revision_iters=cfg.glop_revision_iters,
                        decoder_strategy=cfg.decoder_strategy,
                        solver_device=solver_device,
                    )
                )
        else:
            raise ValueError(f"Unknown operator '{name}'")
    return zoo


