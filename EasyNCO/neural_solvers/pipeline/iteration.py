from abc import ABC, abstractmethod
from typing import Any, Tuple, Literal
import torch
import torch.nn as nn
from tensordict import TensorDict
from torchrl.envs import EnvBase
from torch import Tensor
from EasyNCO.utils.utils import *


def _extract_tsp_scores_from_batch_and_solution(batch: Tensor, solution: Tensor):
    """从 UDC TSP 的 init-only 输出中直接计算 no-aug / aug 分数。"""
    if solution.dim() == 2:
        solution = solution.unsqueeze(1)
    batch_size, aug_factor, problem_size = solution.size()
    gathering_index = solution.unsqueeze(-1).expand(batch_size, aug_factor, problem_size, batch.size(-1))
    ordered_seq = batch[:, None, :, :].expand(batch_size, aug_factor, problem_size, batch.size(-1)).gather(
        dim=2, index=gathering_index
    )
    rolled_seq = ordered_seq.roll(dims=2, shifts=-1)
    segment_lengths = ((ordered_seq - rolled_seq) ** 2).sum(3).sqrt().sum(2)
    return segment_lengths[:, 0], segment_lengths.min(dim=1).values


def _extract_tsp_scores_from_policy_dict_list(initialization_out):
    """从 DIFUSCO/T2T 这类 list[policy_dict] 的 init-only 输出中直接计算分数。"""
    if not isinstance(initialization_out, list) or not initialization_out:
        raise ValueError("initialization_out must be a non-empty list")

    tour_costs = []
    for policy_dict in initialization_out:
        if not isinstance(policy_dict, dict):
            raise TypeError(f"policy_dict must be a dict, got {type(policy_dict).__name__}")
        if "np_nodes" not in policy_dict or "tours" not in policy_dict:
            raise KeyError(f"policy_dict must contain ['np_nodes', 'tours'], got keys={list(policy_dict.keys())}")

        nodes = torch.as_tensor(policy_dict["np_nodes"], dtype=torch.float32)
        tours = torch.as_tensor(policy_dict["tours"], dtype=torch.long)
        if tours.ndim == 1:
            tours = tours.unsqueeze(0)

        ordered_nodes = nodes[tours]
        segment_lengths = ((ordered_nodes[:, 1:, :] - ordered_nodes[:, :-1, :]) ** 2).sum(dim=-1).sqrt().sum(dim=-1)
        tour_costs.append(segment_lengths)

    all_costs = torch.cat(tour_costs, dim=0).float()
    best_cost = all_costs.min()
    return best_cost, best_cost


def _extract_cvrp_scores_from_batch_solution_flag(batch: Tensor, solution: Tensor, solution_flag: Tensor):
    """从 UDC CVRP 的 init-only 输出中直接计算 no-aug / aug 分数。"""
    batch = batch.float()
    solution = solution.long()
    solution_flag = solution_flag.long()
    order_node = solution.clone()
    order_flag = solution_flag.clone().float()

    order_flag[order_flag <= 0.5] = order_node[order_flag <= 0.5].float()
    order_flag[solution_flag > 0.5] = 0.0
    merged_solution = torch.stack((order_node.float(), order_flag), dim=3).view(
        order_node.size(0), order_node.size(1), -1
    ).long()

    gathering_index = merged_solution.unsqueeze(3).expand(
        merged_solution.size(0), merged_solution.size(1), merged_solution.size(2), 2
    )
    ordered_seq = batch[:, None, :, :2].expand(
        merged_solution.size(0), merged_solution.size(1), batch.size(1), 2
    ).gather(dim=2, index=gathering_index)
    rolled_seq = ordered_seq.roll(dims=2, shifts=-1)
    segment_lengths = ((ordered_seq - rolled_seq) ** 2).sum(3).sqrt().sum(2)
    return segment_lengths[:, 0], segment_lengths.min(dim=1).values


class Iteration(ABC):
    """
    Base class for iteration.
    """
    def __init__(self, policy: nn.Module):
        super().__init__()
        self.policy = policy

    @abstractmethod
    def run(self,
            td: TensorDict,
            env: EnvBase,
            initialization_out: dict,
            phase: Literal["train", "eval"],
            max_steps: int = 0,
            **kwargs,
    ) -> dict:
        """
        This function is used to perform an iteration of the algorithm.
        args:
            - td: The TensorDict containing the state of the environment.
            - env: The environment to be used.
            - initialization_out: The output of the initialization phase.
            - phase: The mode of the iteration, can be 'train' or 'eval'.
            - max_steps: Maximum number of iterations.
        returns:
            - A dictionary containing the results of the iteration.
        """
        raise NotImplementedError("Implement me in subclass!")


class NoIteration(Iteration):
    """
    A no-operation iteration class that does nothing.
    """
    def __init__(self, policy: nn.Module, **kwargs):
        super().__init__(policy=policy)
        self.extra_kwargs = kwargs

    def run(self,
            td: TensorDict,
            env: EnvBase,
            initialization_out: dict,
            phase: Literal["train", "eval"],
            max_steps: int = 0,
            **kwargs,
    ) -> dict:
        """
        This function is used to perform an iteration of the algorithm.
        It simply returns the initialization output without any modifications.
        args:
            - td: The TensorDict containing the state of the environment.
            - env: The environment to be used.
            - initialization_out: The output of the initialization phase.
            - phase: The mode of the iteration, can be 'train' or 'eval'.
            - max_steps: Maximum number of iterations.
        returns:
            - A dictionary containing the results of the iteration, which is the same as initialization_out.
        """
        if (
            isinstance(initialization_out, dict)
            and isinstance(td, TensorDict)
            and getattr(env, "env_name", None) == "tsp"
            and "batch" in td.keys()
            and "solution" in td.keys()
            and "score" not in initialization_out
            and "no_aug_score" not in initialization_out
            and "aug_score" not in initialization_out
        ):
            no_aug_score, aug_score = _extract_tsp_scores_from_batch_and_solution(
                batch=td["batch"].float(),
                solution=td["solution"].long(),
            )
            return {
                "score": no_aug_score.mean(),
                "aug_score": aug_score.mean(),
            }

        if (
            isinstance(initialization_out, dict)
            and isinstance(td, TensorDict)
            and getattr(env, "env_name", None) == "cvrp"
            and "batch" in td.keys()
            and "solution" in td.keys()
            and "solution_flag" in td.keys()
            and "score" not in initialization_out
            and "no_aug_score" not in initialization_out
            and "aug_score" not in initialization_out
        ):
            no_aug_score, aug_score = _extract_cvrp_scores_from_batch_solution_flag(
                batch=td["batch"].float(),
                solution=td["solution"].long(),
                solution_flag=td["solution_flag"].long(),
            )
            return {
                "score": no_aug_score.mean(),
                "aug_score": aug_score.mean(),
            }

        if (
            isinstance(initialization_out, list)
            and getattr(env, "env_name", None) == "tsp"
        ):
            no_aug_score, aug_score = _extract_tsp_scores_from_policy_dict_list(initialization_out)
            return {
                "score": no_aug_score,
                "aug_score": aug_score,
            }

        if isinstance(initialization_out, TensorDict):
            # 兼容 LIH 这类“初始化阶段返回状态 TensorDict，而不是 score dict”的方法，
            # 使其也能走 initialization-only 的统一评测流程。
            if "current_length" in initialization_out.keys():
                current_length = initialization_out["current_length"].reshape(-1).float()
                aug_factor = getattr(env, "aug_factor", 1) if getattr(env, "aug_flag", False) else 1
                raw_problems = getattr(env, "raw_problems", None)
                raw_batch_size = None
                if torch.is_tensor(raw_problems):
                    raw_batch_size = raw_problems.size(0)

                if raw_batch_size is not None and current_length.numel() == raw_batch_size * aug_factor:
                    base_batch_size = raw_batch_size
                    current_length = current_length.reshape(aug_factor, base_batch_size)
                    return {
                        "no_aug_score": current_length[0].mean(),
                        "aug_score": current_length.min(dim=0).values.mean(),
                    }

                if raw_batch_size is not None and current_length.numel() == raw_batch_size:
                    return {
                        "no_aug_score": current_length.mean(),
                        "aug_score": current_length.mean(),
                    }

                if aug_factor > 1 and current_length.numel() % aug_factor == 0:
                    base_batch_size = current_length.size(0) // aug_factor
                    current_length = current_length.reshape(aug_factor, base_batch_size)
                    return {
                        "no_aug_score": current_length[0].mean(),
                        "aug_score": current_length.min(dim=0).values.mean(),
                    }

                return {
                    "no_aug_score": current_length.mean(),
                    "aug_score": current_length.mean(),
                }
        return initialization_out
