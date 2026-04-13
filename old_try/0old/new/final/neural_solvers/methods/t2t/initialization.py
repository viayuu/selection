from EasyNCO.neural_solvers.pipeline import Initialization
from torchrl.envs import EnvBase
from typing import Any, Tuple, Literal

import torch

class T2TInitialization(Initialization):
    """
    A class for DIFUSCO initialization,used to create initial solutions for further improvement in the DIFUSCO framework.
    """
    def run(self,
            env: EnvBase,
            batch: torch.Tensor,
            strategy: str,
            phase: Literal["train", "eval"],
            **kwargs,):

        heatmap_path = kwargs.get('heatmap_path', None)

        stacked_policy_dict = self.policy.model.initialize(batch, heatmap_path = heatmap_path)
        td = None

        return td, stacked_policy_dict
