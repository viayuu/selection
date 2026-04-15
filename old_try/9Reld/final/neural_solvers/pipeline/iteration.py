from abc import ABC, abstractmethod
from typing import Any, Tuple, Literal
import torch
import torch.nn as nn
from tensordict import TensorDict
from torchrl.envs import EnvBase
from torch import Tensor
from EasyNCO.utils.utils import *

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
        return initialization_out
