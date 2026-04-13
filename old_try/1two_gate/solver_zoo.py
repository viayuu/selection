from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional


@dataclass
class SolutionBatch:
    tour: "object"  # torch.Tensor[B,N] (kept generic to avoid torch import at module import time)
    length: "object"  # torch.Tensor[B]

    @property
    def reward(self):
        # reward = -length
        return -self.length


class Initializer:
    name: str

    def solve(self, coords) -> SolutionBatch:
        raise NotImplementedError


class Iterator:
    name: str

    def improve(self, coords, sol0: SolutionBatch) -> SolutionBatch:
        raise NotImplementedError


def _gather_by_action(actions, num_actions: int):
    import torch

    groups = []
    for a in range(num_actions):
        idx = (actions == a).nonzero(as_tuple=False).flatten()
        groups.append(idx)
    return groups


def run_initializers_by_action(coords, actions, zoo: list[Initializer]) -> SolutionBatch:
    import torch

    batch_size = coords.size(0)
    problem_size = coords.size(1)
    device = coords.device

    out_tour = torch.zeros((batch_size, problem_size), dtype=torch.long, device=device)
    out_len = torch.zeros((batch_size,), dtype=coords.dtype, device=device)

    groups = _gather_by_action(actions, len(zoo))
    for action_id, idx in enumerate(groups):
        if idx.numel() == 0:
            continue
        sub = zoo[action_id].solve(coords[idx])
        out_tour[idx] = sub.tour
        out_len[idx] = sub.length

    return SolutionBatch(tour=out_tour, length=out_len)


def run_iterators_by_action(coords, actions, sol0: SolutionBatch, zoo: list[Iterator]) -> SolutionBatch:
    import torch

    batch_size = coords.size(0)
    problem_size = coords.size(1)
    device = coords.device

    out_tour = torch.zeros((batch_size, problem_size), dtype=torch.long, device=device)
    out_len = torch.zeros((batch_size,), dtype=coords.dtype, device=device)

    groups = _gather_by_action(actions, len(zoo))
    for action_id, idx in enumerate(groups):
        if idx.numel() == 0:
            continue
        sub0 = SolutionBatch(tour=sol0.tour[idx], length=sol0.length[idx])
        sub1 = zoo[action_id].improve(coords[idx], sub0)
        out_tour[idx] = sub1.tour
        out_len[idx] = sub1.length

    return SolutionBatch(tour=out_tour, length=out_len)


class TorchNoGrad:
    """
    Convenience context manager wrapper.
    """

    def __enter__(self):
        import torch

        self._cm = torch.inference_mode()
        return self._cm.__enter__()

    def __exit__(self, exc_type, exc, tb):
        return self._cm.__exit__(exc_type, exc, tb)


class NeuralARInitializer(Initializer):
    """
    Wrap a `final/` (EasyNCO) Initialization.run(...) and extract a single best tour + its length.
    """

    def __init__(
        self,
        *,
        name: str,
        env,
        initialization,
        decoder_strategy: str = "greedy",
    ):
        self.name = name
        self.env = env
        self.initialization = initialization
        self.decoder_strategy = decoder_strategy

    def solve(self, coords) -> SolutionBatch:
        import torch

        with TorchNoGrad():
            state_td, _out = self.initialization.run(
                self.env,
                coords,
                self.decoder_strategy,
                "eval",
            )

        # Extract env rollout results (reward and tours).
        reward = state_td.get("reward", None)
        if reward is None:
            raise RuntimeError(f"[{self.name}] initialization did not produce reward in state_td")

        if "next" in state_td.keys():
            selected_node_list = state_td["next"]["selected_node_list"]
        else:
            selected_node_list = state_td["selected_node_list"]

        # reward: [B, pomo], selected_node_list: [B, pomo, N]
        # Choose best (max reward = shortest length) for each instance.
        best_idx = reward.max(dim=1).indices  # [B]
        b = torch.arange(coords.size(0), device=coords.device)
        best_tour = selected_node_list[b, best_idx]  # [B,N]

        # Convert reward (-length) to length.
        best_reward = reward[b, best_idx]  # [B]
        best_length = (-best_reward).to(coords.dtype)

        return SolutionBatch(tour=best_tour, length=best_length)


def _tsp_tour_length(coords, tour):
    """
    Compute TSP tour length in pure torch (no env stepping).
    coords: Tensor[B,N,2]
    tour:   Tensor[B,N] permutation
    """
    ordered = coords.gather(1, tour.unsqueeze(-1).expand(-1, -1, 2))  # [B,N,2]
    rolled = ordered.roll(shifts=-1, dims=1)
    seg = ((ordered - rolled) ** 2).sum(dim=-1).sqrt()  # [B,N]
    return seg.sum(dim=1)  # [B]


def _tsp_succ_length(coords, succ):
    """
    Compute TSP tour length for a successor representation.
    coords: Tensor[B,N,2]
    succ:   Tensor[B,N] where succ[b, i] = next node of i
    """
    # Successor coordinates for each node i.
    succ_xy = coords.gather(1, succ.unsqueeze(-1).expand(-1, -1, 2))  # [B,N,2]
    seg = ((succ_xy - coords) ** 2).sum(dim=-1).sqrt()  # [B,N]
    return seg.sum(dim=1)  # [B]


def _perm_to_succ(tour):
    """
    Convert a permutation tour into successor representation.
    tour: Tensor[B,N] permutation
    returns: Tensor[B,N] succ where succ[node] = next node
    """
    import torch

    nxt = tour.roll(shifts=-1, dims=1)
    succ = torch.empty_like(tour)
    succ.scatter_(1, tour, nxt)
    return succ


def _succ_to_perm(succ, *, start_node: int = 0):
    """
    Convert successor representation into a permutation tour by following the cycle from `start_node`.
    succ: Tensor[B,N]
    """
    import torch

    bsz, n = succ.shape
    device = succ.device

    tour = torch.empty((bsz, n), dtype=torch.long, device=device)
    cur = torch.full((bsz,), int(start_node), dtype=torch.long, device=device)
    for i in range(n):
        tour[:, i] = cur
        cur = succ.gather(1, cur.view(-1, 1)).squeeze(1)

    # sanity: each tour must be a permutation
    for i in range(bsz):
        if torch.unique(tour[i]).numel() != n:
            raise RuntimeError("Invalid successor tour (not a single Hamiltonian cycle).")

    return tour


def _normalize_closed_tour(tour_like, n: int) -> list[int]:
    """
    Convert DIFUSCO-style tour (often length N+1 with repeated start) into a permutation of length N.
    """
    seq = [int(x) for x in tour_like]
    if len(seq) == n + 1 and seq[0] == seq[-1]:
        seq = seq[:-1]
    elif len(seq) == n + 1:
        seq = seq[:n]
    elif len(seq) != n:
        raise ValueError(f"Expected tour length {n} or {n+1}, got {len(seq)}")

    if min(seq) < 0 or max(seq) >= n:
        raise ValueError("Tour contains out-of-range node indices.")
    if len(set(seq)) != n:
        raise ValueError("Tour is not a permutation (duplicates/missing nodes).")
    return seq


class DifuscoInitializer(Initializer):
    """
    Wrap DIFUSCO TSP diffusion model to produce an initial tour.

    Notes:
    - DIFUSCO's `TSPDiffusionPolicy.initialize(...)` is effectively batch=1, so we loop over instances.
    - We keep it simple for correctness; performance is not the focus of this prototype.
    """

    def __init__(self, *, name: str, model):
        self.name = name
        self.model = model  # nn.Module (e.g., TSPDiffusionPolicy)

    def solve(self, coords) -> SolutionBatch:
        import torch

        batch_size, n, _ = coords.shape
        device = coords.device

        out_tour = torch.zeros((batch_size, n), dtype=torch.long, device=device)
        out_len = torch.zeros((batch_size,), dtype=coords.dtype, device=device)

        self.model.eval()
        with torch.no_grad():
            for i in range(batch_size):
                nodes = coords[i : i + 1]  # [1,N,2]
                batch_idx = torch.zeros((1,), dtype=torch.long, device=device)
                adj = torch.zeros((1, n, n), dtype=coords.dtype, device=device)
                gt_tour = torch.arange(n, dtype=torch.long, device=device)[None, :]

                stacked_policy_dict = self.model.initialize((batch_idx, nodes, adj, gt_tour))

                cand_tours: list[list[int]] = []
                for d in stacked_policy_dict:
                    tours = d.get("tours", None)
                    if tours is None:
                        continue
                    for t in tours:
                        cand_tours.append(_normalize_closed_tour(t, n))

                if not cand_tours:
                    raise RuntimeError(f"[{self.name}] DIFUSCO produced no candidate tours.")

                cand = torch.tensor(cand_tours, dtype=torch.long, device=device)  # [K,N]
                lengths = _tsp_tour_length(nodes.expand(cand.size(0), -1, -1), cand)  # [K]
                best_k = int(lengths.argmin().item())

                out_tour[i] = cand[best_k]
                out_len[i] = lengths[best_k].to(coords.dtype)

        return SolutionBatch(tour=out_tour, length=out_len)


class NoIterationIterator(Iterator):
    def __init__(self):
        self.name = "none"

    def improve(self, coords, sol0: SolutionBatch) -> SolutionBatch:
        return sol0


class TwoOptIterator(Iterator):
    """
    Use EasyNCO's difusco 2-opt utility (deterministic, no NN required).
    """

    def __init__(self, *, max_iterations: int = 1000, device: str = "cpu"):
        self.name = f"two_opt_{max_iterations}"
        self.max_iterations = int(max_iterations)
        self.device = device

    def improve(self, coords, sol0: SolutionBatch) -> SolutionBatch:
        import numpy as np
        import torch

        from EasyNCO.neural_solvers.methods.difusco.util import two_opt_refine

        # NOTE: EasyNCO's `two_opt_refine(nodes, tour, ...)` expects a *single* instance `nodes` with shape (N,2),
        # and a batch of tours for that same instance. Here we have a batch of different TSP instances, so we loop.
        nodes_np = coords.detach().cpu().numpy()  # [B,N,2]
        tour0 = sol0.tour.detach().cpu().numpy()  # [B,N]

        batch_size = int(tour0.shape[0])
        tour1_np = np.empty_like(tour0)
        for i in range(batch_size):
            nodes_i = nodes_np[i].astype("float64", copy=False)  # [N,2]
            tour_i = tour0[i].astype("int64", copy=False)  # [N]
            # close the tour: [N] -> [1,N+1]
            tour_i_closed = np.concatenate([tour_i, tour_i[:1]], axis=0)[None, :]
            refined_i, _cnt = two_opt_refine(
                nodes_i,
                tour_i_closed,
                max_iteration=self.max_iterations,
                device=self.device,
            )
            tour1_np[i] = refined_i[0, :-1]  # drop the closing city

        tour1 = torch.from_numpy(tour1_np).to(device=coords.device, dtype=torch.long)

        # compute length using EasyNCO's TSPEnv distance code (no need to step env)
        from EasyNCO.neural_solvers.envs.TSPEnv import TSPEnv

        env = TSPEnv(problem_size=coords.size(1), pomo_size=1, device=str(coords.device))
        length = env._get_travel_distance(
            problems=coords,
            selected_node_list=tour1[:, None, :],
            batch_size=coords.size(0),
            pomo_size=1,
        ).squeeze(1)

        return SolutionBatch(tour=tour1, length=length)


class RRCLIHStyleLEHDIterator(Iterator):
    """
    A minimal wrapper of LEHD's RRC idea for TSP.

    We keep the core logic close to `final/neural_solvers/methods/lehd/iteration.py:TSPLEHDIteration`,
    but return per-instance best tours + lengths (needed for RL reward).
    """

    def __init__(
        self,
        *,
        name: str,
        env,
        policy,
        max_steps: int = 10,
        decoder_strategy: str = "greedy",
    ):
        self.name = name
        self.env = env
        self.policy = policy
        self.max_steps = int(max_steps)
        self.decoder_strategy = decoder_strategy

    def improve(self, coords, sol0: SolutionBatch) -> SolutionBatch:
        import torch

        with TorchNoGrad():
            self.policy.eval()
            # Keep a single trajectory (pomo=1) for the RRC loop.
            best_selected_node_list = sol0.tour[:, None, :]  # [B,1,N]

            problems = coords
            batch_size = problems.size(0)
            problem_size = problems.size(1)

            if self.policy.decoder_strategy is None:
                self.policy.set_decoder_strategy(self.decoder_strategy)

            for _ in range(self.max_steps):
                # (Re-)load full problem first (RRC uses sub-problems internally).
                self.env.load_problems(problems, batch_size)

                # Randomly destroy & repair a subpath.
                partial_len, first_node_index, subpath_length, solution_copy = self._destroy_solution(
                    problems, best_selected_node_list
                )

                prev_length = partial_len
                current_step = 0

                reset_td = self.env.reset()
                self.policy.pre_forward(reset_td)
                state_td = self.env.pre_step()
                done = False
                next_td = state_td

                while not done:
                    if current_step == 0:
                        selected = self.env.solution[:, :, -1]
                        next_td["action"] = selected
                    elif current_step == 1:
                        selected = self.env.solution[:, :, 0]
                        next_td["action"] = selected
                    else:
                        next_td = self.policy(state_td)

                    current_step += 1
                    state_td = self.env.step(next_td)
                    done = state_td["done"].all()

                repaired_sub_solution = torch.roll(self.env.selected_node_list, shifts=-1, dims=2)
                repaired_length = -state_td["reward"]  # [B,1] positive length

                repaired_full_solution = self._accept_repaired_solution(
                    repaired_sub_solution,
                    prev_length,
                    repaired_length,
                    first_node_index,
                    subpath_length,
                    solution_copy,
                )

                best_selected_node_list = repaired_full_solution

            best_tour = best_selected_node_list.squeeze(1)  # [B,N]
            best_length = self.env._get_travel_distance(
                problems=problems,
                selected_node_list=best_selected_node_list,
                batch_size=batch_size,
                pomo_size=1,
            ).squeeze(1)

            # sanity: each tour must be a permutation
            for i in range(best_tour.size(0)):
                if torch.unique(best_tour[i]).numel() != problem_size:
                    raise RuntimeError("RRC produced an invalid tour (duplicate/missing nodes).")

            return SolutionBatch(tour=best_tour, length=best_length)

    def _destroy_solution(self, problems, solution_3d):
        destroyed_problem, destroyed_solution, first_node_index, subpath_length, solution_copy = self._sampling_subpaths(
            problems, solution_3d, mode="test", repair=True
        )

        # store into env for the repair phase
        self.env.solution = destroyed_solution
        self.env.problems = destroyed_problem

        partial_solution_length = self.env._get_travel_distance(
            selected_node_list=destroyed_solution, problems=destroyed_problem
        )

        return partial_solution_length, first_node_index, subpath_length, solution_copy

    def _sampling_subpaths(self, problems, solution, *, length_fix=False, mode="test", repair=False):
        import torch

        problem_size = problems.shape[1]
        batch_size = problems.shape[0]
        embedding_size = problems.shape[2]

        # Use Python ints for slicing (CUDA scalar tensors cannot be used as slice indices).
        first_node_index = int(torch.randint(low=0, high=problem_size, size=(1,)).item())

        if mode == "test":
            subpath_length = int(torch.randint(low=4, high=problem_size + 1, size=(1,)).item())
        else:
            subpath_length = problem_size if length_fix else torch.randint(
                low=4, high=problem_size + 1, size=(1,)
            ).item()
            subpath_length = int(subpath_length)

        solution_copy = torch.cat([solution, solution], dim=-1)  # (B,1,2N)
        new_solution = solution_copy[:, :, first_node_index : first_node_index + subpath_length]
        # rank -> permutation of 0..subpath_len-1
        new_solution_sort_ascending, rank = torch.sort(new_solution, dim=-1, descending=False)
        _, new_solution_rank = torch.sort(rank, dim=-1, descending=False)

        new_solution_copy = torch.cat([new_solution, new_solution], dim=-1).long()
        sorted_new_solution_index_pomo, _ = new_solution_copy.sort(dim=-1, descending=False)
        sorted_new_solution_index = sorted_new_solution_index_pomo.squeeze(1)  # (B,2*sub)

        global_index = torch.arange(batch_size, dtype=torch.long, device=problems.device)[:, None].expand(
            batch_size, sorted_new_solution_index.shape[1]
        )
        node_index = torch.arange(embedding_size, dtype=torch.long, device=problems.device)[None, :].expand(
            batch_size, embedding_size
        )
        coordinate_index = node_index.repeat([1, subpath_length])

        new_problem = problems[global_index, sorted_new_solution_index, coordinate_index].view(
            batch_size, subpath_length, 2
        )

        if repair:
            return new_problem, new_solution_rank, first_node_index, subpath_length, solution_copy

        raise NotImplementedError("Only repair=True is used in this wrapper.")

    def _accept_repaired_solution(
        self,
        repaired_solution,
        pre_length,
        after_repair_length,
        first_node_index,
        subpath_length,
        solution_copy,
    ):
        import torch

        problem_size = solution_copy.shape[-1] // 2

        part1 = solution_copy[:, :, :first_node_index]
        part2 = solution_copy[:, :, first_node_index + subpath_length :]
        origin_sub_solution = solution_copy[:, :, first_node_index : first_node_index + subpath_length]

        sorted_solution_index, _ = torch.sort(origin_sub_solution, dim=2, descending=False)

        # Map the repaired sub-solution (rank indices) back to original node indices.
        repaired_solution_index = sorted_solution_index.gather(dim=2, index=repaired_solution)

        if_repair = pre_length > after_repair_length

        solution_copy = solution_copy.clone()
        solution_copy[if_repair] = torch.cat(
            (part1[if_repair], repaired_solution_index[if_repair], part2[if_repair]), dim=1
        )

        after_repair_solution = solution_copy[:, :, first_node_index : first_node_index + problem_size]
        return after_repair_solution


class DACTIterator(Iterator):
    """
    Call EasyNCO's DACT iteration (policy-guided 2-opt) to improve an existing TSP tour.

    Notes:
    - EasyNCO's DACT implementation is not fully device-agnostic (several tensors are created on CPU).
      To keep this prototype robust, we run DACT on `solver_device` (default: cpu) and move outputs back.
    - We treat the *final* tour after `max_steps` as the output (fixed iteration budget).
    """

    def __init__(
        self,
        *,
        name: str,
        env,
        policy,
        iteration,
        max_steps: int,
        solver_device: str = "cpu",
    ):
        self.name = name
        self.env = env
        self.policy = policy
        self.iteration = iteration
        self.max_steps = int(max_steps)
        self.solver_device = solver_device

    def improve(self, coords, sol0: SolutionBatch) -> SolutionBatch:
        import torch
        from tensordict import TensorDict

        # Run DACT on a dedicated device for robustness.
        run_device = torch.device(self.solver_device)
        coords_run = coords.detach().to(run_device)
        tour0_run = sol0.tour.detach().to(run_device)

        succ0 = _perm_to_succ(tour0_run)
        length0 = _tsp_succ_length(coords_run, succ0).to(coords_run.dtype)

        batch_size = int(coords_run.size(0))
        self.env.load_problems(coords_run, batch_size)

        td0 = TensorDict(
            {
                "solution": succ0,
                "current_length": length0,
                "pre_length": length0,
                "exchange": torch.zeros((batch_size, 2), dtype=torch.long, device=run_device),
                "locs": coords_run,
            },
            batch_size=torch.Size([batch_size]),
        )

        initialization_out = {"locs": coords_run}

        with TorchNoGrad():
            self.policy.eval()
            self.iteration.run(td0, self.env, initialization_out, "eval", max_steps=self.max_steps)

        td_final = getattr(self.iteration, "td", None)
        if td_final is None:
            td_final = td0

        succ1 = td_final["solution"]
        length1 = td_final["current_length"].to(dtype=coords_run.dtype)
        tour1 = _succ_to_perm(succ1, start_node=0)

        return SolutionBatch(
            tour=tour1.to(device=coords.device, dtype=torch.long),
            length=length1.to(device=coords.device, dtype=coords.dtype),
        )


class LIHIterator(Iterator):
    """
    Call EasyNCO's LIH iteration (policy-guided 2-opt) to improve an existing TSP tour.

    Note: The current EasyNCO LIH implementation hard-codes CUDA in multiple places, so this iterator
    only supports `coords.device.type == 'cuda'`.
    """

    def __init__(
        self,
        *,
        name: str,
        env,
        policy,
        iteration,
        max_steps: int,
    ):
        self.name = name
        self.env = env
        self.policy = policy
        self.iteration = iteration
        self.max_steps = int(max_steps)

    def improve(self, coords, sol0: SolutionBatch) -> SolutionBatch:
        import torch
        from tensordict import TensorDict

        if coords.device.type != "cuda":
            raise RuntimeError("LIHIterator requires CUDA (EasyNCO LIH uses hard-coded .cuda()).")

        batch_size = int(coords.size(0))
        self.env.load_problems(coords, batch_size)

        tour0 = sol0.tour
        length0 = _tsp_tour_length(coords, tour0).to(coords.dtype)

        td0 = TensorDict(
            {
                "solution": tour0,
                "current_length": length0,
                "pre_length": length0,
                "exchange": torch.zeros((batch_size, 2), dtype=torch.long, device=coords.device),
                "locs": coords,
            },
            batch_size=torch.Size([batch_size]),
        )

        initialization_out = {"locs": coords}
        max_steps = {"test": self.max_steps}

        with TorchNoGrad():
            self.policy.eval()
            self.iteration.run(td0, self.env, initialization_out, "eval", max_steps=max_steps)

        td_final = getattr(self.iteration, "td", None)
        if td_final is None:
            td_final = td0

        tour1 = td_final["solution"].to(dtype=torch.long)
        length1 = td_final["current_length"].to(dtype=coords.dtype)

        return SolutionBatch(tour=tour1, length=length1)
