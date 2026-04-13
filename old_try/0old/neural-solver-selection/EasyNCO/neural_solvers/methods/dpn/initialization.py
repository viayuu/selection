from EasyNCO.neural_solvers.pipeline import ARInitialization
from typing import Any, Tuple, Literal
from tensordict import TensorDict
import torch
from torchrl.envs import EnvBase

class DPNInitialization(ARInitialization):
    """
    A class for DPN initialization that extends the ARInitialization class.
    It is used to create initial solutions for training in the DPN framework.
    """
    def run(self,
            env: EnvBase,
            batch: int,
            strategy: str,
            phase: Literal["train", "eval"],
            **kwargs,
    ) -> Tuple[TensorDict, Any]:
        env.pomo_size = kwargs.get("pomo_size", 60)
        return super().run(env, batch, strategy, phase, **kwargs)
