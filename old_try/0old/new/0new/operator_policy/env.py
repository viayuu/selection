from __future__ import annotations

from dataclasses import dataclass

import torch

from solver_zoo import run_initializers_by_action, run_operators_by_action
from tsp_utils import SolutionBatch, validate_tour


@dataclass
class SearchState:
    coords: torch.Tensor
    current: SolutionBatch
    best_tour: torch.Tensor
    best_length: torch.Tensor
    init_length: torch.Tensor
    init_action: torch.Tensor
    prev_operator: torch.Tensor
    step_index: torch.Tensor
    stagnation: torch.Tensor


class TSPImprovementEnv:
    def __init__(self, init_zoo, operator_zoo, rollout_steps: int):
        self.init_zoo = init_zoo
        self.operator_zoo = operator_zoo
        self.rollout_steps = int(rollout_steps)

    def reset(self, coords: torch.Tensor, init_actions: torch.Tensor, log_fn=None) -> SearchState:
        sol0 = run_initializers_by_action(coords, init_actions, self.init_zoo, log_fn=log_fn)
        return SearchState(
            coords=coords,
            current=sol0,
            best_tour=sol0.tour.clone(),
            best_length=sol0.length.clone(),
            init_length=sol0.length.clone(),
            init_action=init_actions.clone(),
            prev_operator=torch.full_like(init_actions, -1),
            step_index=torch.zeros_like(init_actions),
            stagnation=torch.zeros_like(init_actions),
        )

    def step(self, state: SearchState, operator_actions: torch.Tensor):
        next_solution = run_operators_by_action(state.coords, operator_actions, state.current, self.operator_zoo)
        valid = validate_tour(next_solution.tour) & torch.isfinite(next_solution.length)
        if not valid.all():
            safe_tour = next_solution.tour.clone()
            safe_length = next_solution.length.clone()
            safe_tour[~valid] = state.current.tour[~valid]
            safe_length[~valid] = state.current.length[~valid]
            next_solution = SolutionBatch(tour=safe_tour, length=safe_length)
        improved = next_solution.length < state.best_length
        next_best_tour = state.best_tour.clone()
        next_best_tour[improved] = next_solution.tour[improved]
        next_best_length = torch.minimum(state.best_length, next_solution.length)
        reward = state.best_length - next_best_length
        next_stagnation = torch.where(improved, torch.zeros_like(state.stagnation), state.stagnation + 1)
        next_step_index = state.step_index + 1
        next_state = SearchState(
            coords=state.coords,
            current=next_solution,
            best_tour=next_best_tour,
            best_length=next_best_length,
            init_length=state.init_length,
            init_action=state.init_action,
            prev_operator=operator_actions.clone(),
            step_index=next_step_index,
            stagnation=next_stagnation,
        )
        done = next_step_index >= self.rollout_steps
        return next_state, reward, done

