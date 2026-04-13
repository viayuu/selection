from EasyNCO.neural_solvers.pipeline import Initialization
from typing import Any, Tuple, Literal
from tensordict import TensorDict
from torchrl.envs import EnvBase
import torch

class LEHDInitialization(Initialization):
    """
    A class for LEHD initialization that extends the ARInitialization class.
    It is used to create initial solutions for training in the LEHD Model.
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
            env.load_problems(batch, batch.size(0))
            state_td, policy_out = self.play_episode(env, strategy)
            out = policy_out
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
                "batch" : batch,
            }

        return state_td, out

    def play_episode(
        self,
        env,
        decoder_strategy: str = "sampling",
    ) -> Tuple[TensorDict, dict]:
        reset_td = env.reset()

        self.policy.set_decoder_strategy(decoder_strategy)
        self.policy.pre_forward(reset_td)

        loss_list = []
        done = False
        reward = None
        state_td = env.pre_step()
        while not done:
            next_td = self.policy(state_td)
            prob = next_td["prob"]
            state_td = env.step(next_td)
            loss_list.append(prob)

            reward = state_td['reward']
            done = state_td['done'].all()
        policy_out = {
            'loss' : loss_list,
            'reward' : reward,
        }

        return state_td, policy_out