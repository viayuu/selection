from EasyNCO.neural_solvers.pipeline import Iteration
from tensordict import TensorDict
from torchrl.envs import EnvBase
from typing import Any, Tuple, Literal

class DIFUSCOIteration(Iteration):
    """
    A class for DIFUSCO improvement by 2-opt/mcts,used to search for better solution based on the provided initial
    solution.
    """

    def run(self,
            td: dict,
            env: EnvBase,
            initialization_out: dict,
            phase: Literal["train", "eval"],
            max_steps: int = 0,
            **kwargs,
            ):

        metric = self.policy.model.iteration(initialization_out, max_steps)

        return metric
