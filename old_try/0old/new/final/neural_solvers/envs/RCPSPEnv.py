import torch
import os
from typing import Optional

from torch.utils.data import DataLoader
from torchrl.envs import EnvBase
from tensordict import TensorDict

from EasyNCO.utils.utils import getLogger
from EasyNCO.data.data_utils import *
from EasyNCO.data import TSPGenerator


logger = getLogger(__name__)

class RCPSPEnv(EnvBase):
    # batch_locked = False
    def __init__(self,
                 problem_size: int,
                 pomo_size: int = 1,  # multi_start trajectory, in AM-based models, it is 1 by default
                 device: str = 'cpu',
                 seed: int = 2024,
                 **kwargs):
        super().__init__(device=device)

        self.env_name = 'rcpsp'
        self.problem_size = problem_size
        self.pomo_size = pomo_size
        self.device = device

        self.problems = None

        self.selected_count = None
        self.current_node = None
        self.ninf_mask = None
        self.selected_node_list = None
        #self.first_node = None
        self.dummy_flag_bool = None
        self.dummy_flag_long = None

        self.seed_value = seed

        self._set_seed(seed=seed)
        self.method_name = kwargs.get('method_name')

        self.aug_type = kwargs.get('aug_type', None)
        self.aug_factor = kwargs.get('aug_factor',1)
        self.mix_prop = kwargs.get('mix_prop')
        self.aug_flag = self.aug_type is not None and self.aug_factor > 1
        self.distribution=kwargs.get("distribution","uniform")

    def _set_seed(self, seed: Optional[int]):
        rng = torch.manual_seed(seed)
        self.rng = rng

    def _reset(self,td: TensorDict,batch_size=None) -> TensorDict:
        pass

    def _step(self, td: TensorDict) -> TensorDict:
        pass

    def __getstate__(self):
        """Return the state of the environment. By default, we want to avoid pickling
        the random number generator directly as it is not allowed by `deepcopy`
        """
        pass

    def __setstate__(self, state):
        """Set the state of the environment. By default, we want to avoid pickling
        the random number generator directly as it is not allowed by `deepcopy`
        """
        pass









