from EasyNCO.neural_solvers.pipeline import ARInitialization
from typing import Any, Tuple
from tensordict import TensorDict
import torch

class POMOInitialization(ARInitialization):
    """
    A class for POMO initialization that extends the ARInitialization class.
    It is used to create initial solutions for training in the POMO framework.
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
        if env.env_name == "kp":
            while not done:
                temp_td = self.policy(state_td)
                selected = temp_td.get("action", None)
                prob = temp_td.get("prob", None)

                action_w_finished = selected.clone()
                action_w_finished[temp_td.get("done", False)] = env.problem_size
                temp_td["action"] = action_w_finished
                next_td = env.step(temp_td)
                if prob is not None:
                    chosen_prob = prob.clone()
                    chosen_prob[next_td["done"]] = 1
                    likelihood = torch.cat(
                        (likelihood, chosen_prob[:, :, None]), dim=2
                    )
                reward = next_td["reward"]
                done = next_td["done"].all()
        else:
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
