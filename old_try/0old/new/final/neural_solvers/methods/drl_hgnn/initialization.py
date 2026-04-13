from typing import Any, Tuple, Literal
import torch
from tensordict import TensorDict
from torchrl.envs import EnvBase
from EasyNCO.utils.utils import *
from EasyNCO.neural_solvers.pipeline import Initialization


class Memory:
    def __init__(self):
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []
        self.action_indexes = []

        self.ope_ma_adj = []
        self.ope_pre_adj = []
        self.ope_sub_adj = []
        self.batch_idxes = []
        self.raw_opes = []
        self.raw_mas = []
        self.proc_time = []
        self.jobs_gather = []
        self.eligible = []
        self.nums_opes = []

    def clear_memory(self):
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]
        del self.action_indexes[:]

        del self.ope_ma_adj[:]
        del self.ope_pre_adj[:]
        del self.ope_sub_adj[:]
        del self.batch_idxes[:]
        del self.raw_opes[:]
        del self.raw_mas[:]
        del self.proc_time[:]
        del self.jobs_gather[:]
        del self.eligible[:]
        del self.nums_opes[:]


class DRL_HGNNInitialization(Initialization):
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
            # self.policy.train()
            # env.load_problems(batch, batch.size(0))
            # state_td, policy_out = self.play_episode(env, strategy)
            # out = policy_out
            raise NotImplementedError("hgnn_train not implemented")
        else:
            self.policy.eval()
            with torch.inference_mode():
                env.load_problems(batch, batch_size=batch.size(0))
                state_td, policy_out = self.play_episode(env, strategy)
            makespan = policy_out['reward']
            out={
                "no_aug_score": makespan.mean(),
                "aug_score": makespan.mean(),
            }

        return state_td, out


    def play_episode(
        self,
        env,
        decoder_strategy: str = "sampling",  #In drl_hgnn,sampling means contains multiple (=num_sample) copies of one instance
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
        likelihood = torch.zeros(size=(env.env_batch_size, env.num_job*env.num_machine, 0))

        done = False
        reward = None
        state_td = env.state
        memories = Memory()
        while not done:
            next_td = self.policy.act(state_td,memories)
            prob = next_td["prob"]
            state_td = env.step(next_td)
            likelihood = torch.cat((likelihood, prob[:, :, None]), dim=2)
            reward = state_td["reward"]
            done = state_td["done"].all()
            state_td = state_td['next']['state']

        # Verify the solution
        gantt_result = env.validate_gantt()[0]
        if not gantt_result:
            assert ("Scheduling Error！！！！！！")

        policy_out = {
            "reward": torch.min(env.makespan_batch),
            "likelihood": likelihood,
        }

        return state_td, policy_out