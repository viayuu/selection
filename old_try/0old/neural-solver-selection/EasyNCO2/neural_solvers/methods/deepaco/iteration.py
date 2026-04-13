from EasyNCO.neural_solvers.pipeline import Iteration
from EasyNCO.neural_solvers.methods.deepaco.initialization import load_policy
from typing import Any, Tuple, Literal
from tensordict import TensorDict
from torchrl.envs import EnvBase
import torch

class DeepACOIteration(Iteration):
    """
    A class for DeepACO iteration that extends the Iteration class.
    It is used to create initial solutions for training in the DeepACO framework.
    """
    def run(self,
            td: TensorDict,
            env: EnvBase,
            initialization_out: dict,
            phase: Literal["train", "eval"],
            max_steps: int = 0,
            **kwargs,
    ) -> dict:
        """
        This function is used to perform an iteration of the algorithm.
        It simply returns the initialization output without any modifications.
        args:
            - td: The TensorDict containing the state of the environment.
            - env: The environment to be used.
            - initialization_out: The output of the initialization phase.
            - phase: The mode of the iteration, can be 'train' or 'eval'.
            - max_steps: Maximum number of iterations.
        returns:
            - A dictionary containing the results of the iteration, which is the same as initialization_out.
        """

        batch = initialization_out['batch']
        policy_param = kwargs.get('policy_param', {})
        problem_type = kwargs.get('problem_type', {})

        if phase == 'train':

            problem_size = batch.size(1)
            pomo_size = policy_param['n_ants']

            likelihood = torch.zeros(size=(0, pomo_size, problem_size-1))
            rewards = torch.zeros(size=(0, pomo_size))

            for data in batch:
                # batch: (batch_size, problem_size, features)
                # data： (problem_size, features)
                policy = load_policy(
                    model = self.policy,
                    data = data,
                    policy_param = policy_param,
                    env_name = problem_type,
                )
                reward, prob = policy.train_instance()
                # reward: (ants, )
                # prob: (problem-1, ants)

                reward = reward.unsqueeze(0) # reward: (1, pomo=ants)
                prob = prob.t().unsqueeze(0) # prob: (1, pomo=ants, problem-1)

                rewards = torch.cat((rewards, reward), dim=0)
                likelihood = torch.cat((likelihood, prob), dim=0)


            out = {
                'reward': rewards, # (batch, pomo=ants)
                'likelihood': likelihood, # (batch, pomo=ants, problem-1)
            }

        else:

            sum_results = torch.zeros(size=(max_steps,))
            # Only process a single instance.
            for data in batch:
                # Load the policy.
                policy = load_policy(
                    model = self.policy,
                    data = data,
                    policy_param = policy_param,
                    env_name = problem_type,
                )
                results = torch.zeros(size=(max_steps,))
                for i in range(max_steps):
                    # Generate the heatmap.
                    heatmap = policy.heatmap()
                    # Search based on the heatmap.
                    results[i] = policy.search(heatmap)

                sum_results += results

            avg_results = sum_results / len(batch)
            # The average output of the batch in each iteration.

            out = {
                'score': avg_results[-1],
                'aug_score': avg_results[-1],
            }



        return out
