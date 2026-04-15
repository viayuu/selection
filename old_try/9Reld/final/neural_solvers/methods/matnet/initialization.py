from EasyNCO.neural_solvers.pipeline import ARInitialization
from typing import Any, Tuple, Literal
from tensordict import TensorDict
import torch
import torch.nn as nn
from torchrl.envs import EnvBase

class MatNetInitialization(ARInitialization):
    """
    Initialization class for MatNet model
    """
    def __init__(self, policy: nn.Module):
        super().__init__(policy.model)

    def run(self,
            env: EnvBase,
            batch: int,
            strategy: str,
            phase: Literal["train", "eval"],
            **kwargs,
    ) -> Tuple[TensorDict, Any]:
        pomo_size = kwargs.get("pomo_size")
        if pomo_size == 24:
            env.pomo_size = 24
        return super().run(env, batch, strategy, phase, **kwargs)
