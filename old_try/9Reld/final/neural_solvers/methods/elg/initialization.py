from EasyNCO.neural_solvers.pipeline import ARInitialization
from typing import Any, Tuple
from tensordict import TensorDict
import torch

class ELGInitialization(ARInitialization):
    """
    A class for ELG initialization that extends the ARInitialization class.
    It is used to create initial solutions for training in the ELG framework.
    """
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
        """
        reset_td = env.reset()

        self.policy.set_decoder_strategy(decoder_strategy)
        self.policy.pre_forward(reset_td)


        likelihood = torch.zeros(size=(env.batch_size[0], env.pomo_size, 0))
        done = False
        reward = None
        state_td = env.pre_step()
        step = 0
        while not done:
            if step > 1:
                self.policy.enable_local_policy()
            next_td = self.policy(state_td)
            prob = next_td["prob"]
            state_td = env.step(next_td)
            likelihood = torch.cat((likelihood, prob[:, :, None]), dim=2)
            reward = state_td["reward"]
            done = state_td["done"].all()
            step += 1
        policy_out = {
            "reward": reward,
            "likelihood": likelihood,
        }
        return state_td, policy_out
