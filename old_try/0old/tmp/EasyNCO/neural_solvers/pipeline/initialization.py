from abc import ABC, abstractmethod
from typing import Any, Tuple, Literal, Union
import torch
import torch.nn as nn
from tensordict import TensorDict
from torchrl.envs import EnvBase
from torch import Tensor
from EasyNCO.utils.utils import *


class Initialization(ABC):
    """
    Base class for initialization.
    """
    def __init__(self, policy: nn.Module):
        super().__init__()
        self.policy = policy

    @abstractmethod
    def run(
        self,
        env: EnvBase,
        batch: Union[Tensor, dict, list],
        strategy: str,
        phase: Literal["train", "eval"],
        **kwargs,
    ) -> Tuple[TensorDict, Any]:
        """
        This function is used to create initial solution for training.
        args:
            - env: The environment to be used.
            - batch: The batch of data to be used.
            - strategy: The strategy to decode the output of the policy network.
            - phase: The mode of the initialization, can be 'train' or 'eval'.
        returns:
            - state_td: The final state of the environment.
            - out: The output of the policy network or the scores for evaluation.
        """
        raise NotImplementedError("Implement me in subclass!")


class ARInitialization(Initialization):
    """
    A base initialization class that does nothing.
    """

    def run(self,
            env: EnvBase,
            batch: Union[Tensor, dict, list],
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
            - batch: The batch of data to be used.
            - strategy: The strategy to decode the output of the policy network.
            - mode: The mode of the initialization, can be 'train' or 'eval'.
        returns:
            - state_td: The final state of the environment.
            - out: The output of the policy network or the scores for evaluation.
        """
        if phase == "train":
            self.policy.train()
            if isinstance(batch,dict):
                batch_size = batch['depot_node_xy'].size(0)
            else:
                batch_size=batch.size(0)
            env.load_problems(batch, batch_size)
            state_td, policy_out = self.play_episode(env, strategy)
            out = policy_out
        else:
            self.policy.eval()
            with torch.inference_mode():
                if isinstance(batch,dict):
                    batch_size = batch['depot_node_xy'].size(0)
                else:
                    batch_size=batch.size(0)
                env.load_problems(batch, batch_size=batch_size)
                state_td, policy_out = self.play_episode(env, strategy)
            aug_reward = policy_out["reward"].reshape(
                env.aug_factor, batch_size, env.pomo_size
            )
            # shape: (augmentation, batch, pomo)

            max_pomo_reward, _ = aug_reward.max(dim=2)  # best result from pomo
            # shape: (augmentation, batch)

            no_aug_reward = (
                -max_pomo_reward[0, :].float().mean()
            )  # negative sign to make positive value

            max_aug_pomo_reward, _ = max_pomo_reward.max(dim=0)  # get best result from augmentation
            # shape (batch, )
            aug_score = (
                -max_aug_pomo_reward.float().mean()
            )  # negative sign to make positive value

            out = {
                "no_aug_score": no_aug_reward,
                "aug_score": aug_score,
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
        reset_td = env.reset()

        self.policy.set_decoder_strategy(decoder_strategy)
        self.policy.pre_forward(reset_td)

        likelihood = torch.zeros(size=(env.batch_size[0], env.pomo_size, 0))

        done = False
        reward = None
        state_td = env.pre_step()
        while not done:
            next_td = self.policy(state_td)
            prob = next_td["prob"]
            state_td = env.step(next_td)
            likelihood = torch.cat((likelihood, prob[:, :, None]), dim=2)

            reward = state_td["reward"]
            done = state_td["done"].all()

        policy_out = {
            "reward": reward,
            "likelihood": likelihood,
        }
        return state_td, policy_out
