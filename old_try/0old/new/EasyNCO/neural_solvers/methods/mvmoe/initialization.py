from EasyNCO.neural_solvers.pipeline import Initialization
from typing import Any, Tuple, Literal
from tensordict import TensorDict
import torch
from torchrl.envs import EnvBase
from torch import Tensor
from EasyNCO.utils.utils import *
from EasyNCO.neural_solvers.methods.htsp import utils_htsp as htsp_utils


class MVRPInitialization(Initialization):
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
            assert len(set(batch["problem_type"])) == 1
            batch_size = batch['depot_node_xy'].size(0)
            env.load_problems(batch, batch_size)
            state_td, policy_out = self.play_episode(env, strategy)
            out = policy_out

            return state_td, out

        else:
            self.policy.eval()
            with torch.inference_mode():
                env.load_problems(batch, batch_size=batch.size(0))
                state_td, policy_out = self.play_episode(env, strategy)
            aug_reward = policy_out["reward"].reshape(
                env.aug_factor, batch.size(0), env.pomo_size
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






def training_step(self, batch, batch_idx):
    """
    This function is called for each batch of data. It is used to calculate the loss and update the model.
    :param batch:
    :param batch_idx:
    :return: loss
    """


def play_episode(self, policy, env, decoder_strategy: str = "sampling", first_mode: str = None) -> Tuple[
    TensorDict, dict]:
    '''
    This function is used to play an episode of the environment.
    It doesn't exist in lightning, but it is necessary for the REINFORCE algorithm.
    The function returns the final state of the environment and the output of the policy network.

    - Args:

        - decoder_strategy: The strategy to decode the output of the policy network.
        It can be 'sampling' or 'greedy' in the current version.

        - first_mode: The mode to select the first node.

    Note that if first_mode is 'random', the first node is selected randomly in the order of the nodes,
    otherwise, the first node is selected based on the placeholder net.
    '''
    reset_td = env.reset()

    policy.set_decoder_strategy(decoder_strategy)
    policy.pre_forward(reset_td)

    log_likelihood = torch.zeros(size=(env.batch_size[0], env.pomo_size, 0))

    done = False
    reward = None
    state_td = env.pre_step()
    while not done:
        next_td = policy(state_td, first_mode=first_mode)
        prob = next_td.get("prob", None)
        state_td = env.step(next_td)
        log_likelihood = torch.cat((log_likelihood, prob[:, :, None]), dim=2)
        reward = state_td["reward"]
        done = state_td["done"].all()

    policy_out = {
        "reward": reward,
        "log_likelihood": log_likelihood,
    }
    return state_td, policy_out


def calculate_loss(self, policy_out: dict, **kwargs):
    """Calculate loss for REINFORCE algorithm.
    Args:
        policy_out: Output of the policy network
    """
    reward = policy_out["reward"]  # shape: (batch, pomo)
    log_likelihood = policy_out["log_likelihood"]  # shape: (batch, pomo, problem)
    log_prob = log_likelihood.log().sum(dim=2)
    # shape = (batch, pomo)
    self.env.aug_flag = False
    bl_value, bl_loss = (self.baseline.eval(self.env.problems, reward, env=self.env))
    self.env.aug_flag = self.env.aug_type is not None
    advantage = reward - bl_value
    reinforce_loss = -(log_prob * advantage).mean()

    task_loss = reinforce_loss + bl_loss

    if hasattr(self.policy, "aux_loss"):
        task_loss = task_loss + self.policy.aux_loss  # add aux(moe)_loss for load balancing (default coefficient: 1e-2)

    return task_loss