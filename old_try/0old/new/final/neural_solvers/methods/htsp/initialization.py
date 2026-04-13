from EasyNCO.neural_solvers.pipeline import Initialization
from typing import Any, Tuple, Literal
from tensordict import TensorDict
import torch
from torchrl.envs import EnvBase
from torch import Tensor
from EasyNCO.utils.utils import *
from EasyNCO.neural_solvers.methods.htsp import utils_htsp as htsp_utils


class HTSPInitialization(Initialization):
    """
    A base initialization class that does nothing.
    """

    def run(self,
            env: EnvBase,
            batch: int,
            strategy: str,
            phase: Literal["train", "eval"],
            **kwargs,
    ) -> Tuple[TensorDict, Any]:
        """
        This function is used to create initial solution for training.
        It simply loads the problems into the environment and plays an episode
        using the provided policy and strategy.
        args:
            - policy: The policy network to be used.
            - env: The environment to be used.
            - batch: The batch size.
            - strategy: The strategy to decode the output of the policy network.
            - mode: The mode of the initialization, can be 'train' or 'eval'.
        returns:
            - state_td: The final state of the environment.
            - out: The output of the policy network or the scores for evaluation.
        """
        if phase == "train":
            self.policy.train()
            self.memory = kwargs["memory"]
            trajectory = self.policy.explore_trajectory(batch) ##!!  # batch
            num_steps, experience_reward = htsp_utils.update_buffer(self.memory, trajectory)
            # not using play_episode
            state_td, policy_out = self.play_episode(env, strategy)
            out = {'num_steps': num_steps, 'experience_reward': experience_reward}
        else:
            self.policy.eval()
            duration, length = self.policy.HTSP_test_step(batch)
            state_td = None

            out = {
                "score": length,
                "aug_socre": length,
            }

        return state_td, out

    def play_episode(
        self,
        env,
        decoder_strategy: str = "sampling",
    ) -> Tuple[TensorDict, dict]:
        """
        This function is used to play an episode of the environment.
        It doesn't exist in lightning, but it is necessary for the REINFORCE algorithm.
        The function returns the final state of the environment and the output of the policy network.

        - Args:

            - decoder_strategy: The strategy to decode the output of the policy network.
            It can be 'sampling' or 'greedy' in the current version.
        """
        return None, None
