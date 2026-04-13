from EasyNCO.neural_solvers.pipeline import Initialization
from typing import Any, Tuple, Literal
from torchrl.envs import EnvBase
from tensordict import TensorDict
import torch
import numpy as np

class NLNSInitialization(Initialization):
    """
    A class for NLNS initialization that extends the Initialization class.
    It is used to create solutions for training in the NLNS framework.
    """
    def run(self,
            env: EnvBase,
            batch: int,
            strategy: str,
            phase: Literal["train", "eval"],
            **kwargs,
    ) -> Tuple[TensorDict, Any]:

        if phase == "train":
            self.policy.train()
            env.load_problems(batch, batch.size(0))
            tool = Initialization_tool(env.problems)
            solutions = tool.initial()

        else:
            env.load_problems(batch, batch.size(0))
            tool = Initialization_tool(env.problems)
            solutions = tool.initial()

        out = {
            'solution': solutions,
            'locs': env.problems,
            'strategy': strategy,
        }

        return out, out


class Initialization_tool():
    def __init__(self, problems):
        self.problems = problems
        # shape: (batch, 1 + problem, 3)


    def initial(self):

        round_error_epsilon = 0.00001
        capacity = 1.0 + round_error_epsilon
        solutions = self.create_initial_solutions(self.problems, capacity) # bacth_solution, list
        # shape: (batch,problem)

        return solutions


    def create_initial_solutions(self, data, capacity):
        # data: shape: (batch, 1 + problem, 3)
        # capacity: shape: (batch, pomo, 1)
        """Create initial solutions for a batch of instances using a greedy heuristic."""
        batch_size = data.size(0)
        problem_size = data.size(1) - 1
        depot_node_xy = data[:, :, :2] # shape: (batch, 1 + problem, 2)
        depot_node_demand = data[:, :, 2] # shape: (batch, 1 + problem)
        solutions = []

        for i in range(batch_size):
            locations = depot_node_xy[i] # shape: (problem+1, 2)
            demand = depot_node_demand[i] # shape: (problem+1, )
            solution = [[[0, 0, 0]], [[0, 0, 0]]]
            cur_load = capacity
            mask = np.array([True] * (problem_size + 1))
            mask[0] = False
            while mask.any():
                closest_customer_idx = self.get_n_closest_locations_to(locations, solution[-1][-1][0], mask, 1)[0]
                if demand[closest_customer_idx] <= cur_load:
                    mask[closest_customer_idx] = False
                    solution[-1].append([int(closest_customer_idx), demand[closest_customer_idx].item(), None])
                    cur_load -= demand[closest_customer_idx]
                else:
                    solution[-1].append([0, 0, 0])
                    solution.append([[0, 0, 0]])
                    cur_load = capacity
            solution[-1].append([0, 0, 0]) # solution_instance shape: [[[], []], [[], [], ...], ...]

            solutions.append(solution) # solutions_batch

        return solutions

    def get_n_closest_locations_to(self, locations, origin_location_id, mask, n):
        """Return the idx of the n closest locations sorted by distance."""
        distances = np.array([np.inf] * len(mask))
        distances[mask] = ((locations[mask] * locations[origin_location_id]) ** 2).sum(1).cpu().numpy()
        order = np.argsort(distances)
        return order[:n]

