import torch
import os
from typing import Optional

from torch.utils.data import DataLoader
from torchrl.envs import EnvBase
from tensordict import TensorDict, TensorDictBase

from EasyNCO.utils.utils import getLogger
from EasyNCO.data.data_utils import *
from EasyNCO.data import JSSPGenerator
import re

class JSSPEnv(EnvBase):
    def __init__(self,
                 device: str = 'cpu',
                 seed: int = 2024,
                 **kwargs):
        super().__init__(device=device)

        self.num_job = kwargs.get('num_job', 10)
        self.num_machine = kwargs.get('num_machine', 10)
        self.time_low = kwargs.get("time_low", 1)
        self.time_high = kwargs.get("time_high", 99)

        self.device = device
        self.seed_value = seed
        self.env_name = "jssp"
        self.aug_type = None
        self.pomo_size = 1  # dummy
        self._set_seed(seed=self.seed_value)

    def _set_seed(self, seed: Optional[int]):
        rng = torch.manual_seed(seed)
        self.rng = rng

    def __getstate__(self):
        """Return the state of the environment. By default, we want to avoid pickling
        the random number generator directly as it is not allowed by `deepcopy`
        """
        state = self.__dict__.copy()
        state["rng"] = state["rng"].get_state()
        return state

    def __setstate__(self, state):
        """Set the state of the environment. By default, we want to avoid pickling
        the random number generator directly as it is not allowed by `deepcopy`
        """
        self.__dict__.update(state)
        self.rng = torch.manual_seed(self.seed_value)
        self.rng.set_state(state["rng"])

    def load_problems(self, dataset, batch_size: int = 64):
        self.problems = dataset
        self.env_batch_size = batch_size

    def _reset(self, tensordict: TensorDictBase, **kwargs) -> TensorDictBase:
        pass

    def _step(self, td: TensorDict) -> TensorDict:
        pass