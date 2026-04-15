from EasyNCO.neural_solvers.pipeline import ARInitialization
from typing import Any, Tuple,Literal
from torchrl.envs import EnvBase
from tensordict import TensorDict
import torch
from einops import rearrange

class PSLInitialization(ARInitialization):
    """
    A class for PSL initialization that extends the ARInitialization class.
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
        self.policy.decoder.assign(env.pref)
        self.policy.set_decoder_strategy(decoder_strategy)
        self.policy.pre_forward(reset_td)

        likelihood = torch.zeros(size=(env.batch_size[0], env.pomo_size, 0))

        done = False
        reward = None
        state_td = env.pre_step()
        if env.env_name == "mokp":
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
        self.policy.num_target = env.num_target

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

            reward = - policy_out["reward"]
            # (batch_size, pomo_size, num_target)

            if env.num_target == 2:
                # z = torch.ones(reward.shape).cuda() * 0.0
                z = torch.ones(reward.shape) * 0.0
                tch_reward = torch.tensor(env.pref, device=reward.device) * (reward - z)
                tch_reward, _ = tch_reward.max(dim=2)
                # (batch, pomo)

                reward = -reward
                reward_obj1 = reward[:, :, 0].reshape(env.aug_factor, batch.size(0), env.pomo_size) # (aug, batch, pomo)
                reward_obj2 = reward[:, :, 1].reshape(env.aug_factor, batch.size(0), env.pomo_size) # (aug, batch, pomo)

                tch_reward = -tch_reward
                tch_reward = tch_reward.reshape(env.aug_factor, batch.size(0), env.pomo_size)
                # (augmentation, batch, pomo)
                tch_reward_no_aug = tch_reward[[0], :]
                # (1, batch, pomo)

                # aug
                tch_reward_aug = rearrange(tch_reward, 'c b h -> b (c h)')
                _, max_idx_aug = tch_reward_aug.max(dim=1)
                max_idx_aug = max_idx_aug.reshape(max_idx_aug.shape[0], 1)
                max_reward_obj1 = rearrange(reward_obj1,
                                            'c b h -> b (c h)').gather(1, max_idx_aug)
                max_reward_obj2 = rearrange(reward_obj2,
                                            'c b h -> b (c h)').gather(1, max_idx_aug)

                aug_score = []
                aug_score.append(-max_reward_obj1.float().mean())
                aug_score.append(-max_reward_obj2.float().mean())

                # no aug
                tch_reward_no_aug = rearrange(tch_reward_no_aug, 'c b h -> b (c h)')
                _, max_idx_no_aug = tch_reward_no_aug.max(dim=1)
                max_idx_no_aug = max_idx_no_aug.reshape(max_idx_no_aug.shape[0], 1)
                max_reward_obj1_no_aug = rearrange(reward_obj1[[0], :],
                                            'c b h -> b (c h)').gather(1, max_idx_no_aug)
                max_reward_obj2_no_aug = rearrange(reward_obj2[[0], :],
                                            'c b h -> b (c h)').gather(1, max_idx_no_aug)

                no_aug_score = []
                no_aug_score.append(-max_reward_obj1_no_aug.float().mean())
                no_aug_score.append(-max_reward_obj2_no_aug.float().mean())

                out = {
                    "no_aug_score": no_aug_score,
                    "aug_score": aug_score,
                }

            elif env.num_target == 3:
                if env.problem_size == 20:
                    ref_values = 15
                if env.problem_size == 50:
                    ref_values = 30
                if env.problem_size == 100:
                    ref_values = 45

                # # reward was negative, here we set it to positive to calculate TCH
                # reward = - reward
                # z = torch.ones(reward.shape).cuda() * ref_values
                z = torch.ones(reward.shape) * ref_values

                theta = 0.1

                pref = torch.tensor(env.pref, dtype=torch.float32)
                d1 = (pref * (z - reward)).sum(dim=2) / torch.norm(pref)
                d1_variant = pref[:, None, None] / torch.norm(pref) * d1
                d1_varaint = rearrange(d1_variant, 'c b h -> b h c')
                d2 = torch.norm(z - reward - d1_varaint)
                pbi_reward = d1 - theta * d2
                tch_reward = pbi_reward

                # reward = - reward
                reward_obj1 = reward[:, :, 0].reshape(env.aug_factor, batch.size(0), env.pomo_size) # (aug, batch, pomo)
                reward_obj2 = reward[:, :, 1].reshape(env.aug_factor, batch.size(0), env.pomo_size) # (aug, batch, pomo)
                reward_obj3 = reward[:, :, 2].reshape(env.aug_factor, batch.size(0), env.pomo_size) # (aug, batch, pomo)

                tch_reward = tch_reward.reshape(env.aug_factor, batch.size(0), env.pomo_size)
                # (aug, batch, pomo)
                tch_reward_no_aug = tch_reward[[0], :]
                # (1, batch, pomo)

                # aug
                tch_reward_aug = rearrange(tch_reward, 'c b h -> b (c h)')
                _, max_idx_aug = tch_reward_aug.max(dim=1)
                max_idx_aug = max_idx_aug.reshape(max_idx_aug.shape[0], 1)
                max_reward_obj1 = rearrange(reward_obj1,
                                            'c b h -> b (c h)').gather(1, max_idx_aug)
                max_reward_obj2 = rearrange(reward_obj2,
                                            'c b h -> b (c h)').gather(1, max_idx_aug)
                max_reward_obj3 = rearrange(reward_obj3,
                                            'c b h -> b (c h)').gather(1, max_idx_aug)

                aug_score = []
                aug_score.append(max_reward_obj1.float().mean())
                aug_score.append(max_reward_obj2.float().mean())
                aug_score.append(max_reward_obj3.float().mean())

                # no aug
                tch_reward_no_aug = rearrange(tch_reward_no_aug, 'c b h -> b (c h)')
                _, max_idx_no_aug = tch_reward_no_aug.max(dim=1)
                max_idx_no_aug = max_idx_no_aug.reshape(max_idx_no_aug.shape[0], 1)
                max_reward_obj1_no_aug = rearrange(reward_obj1[[0], :],
                                            'c b h -> b (c h)').gather(1, max_idx_no_aug)
                max_reward_obj2_no_aug = rearrange(reward_obj2[[0], :],
                                            'c b h -> b (c h)').gather(1, max_idx_no_aug)
                max_reward_obj3_no_aug = rearrange(reward_obj3[[0], :],
                                            'c b h -> b (c h)').gather(1, max_idx_no_aug)

                no_aug_score = []
                no_aug_score.append(max_reward_obj1_no_aug.float().mean())
                no_aug_score.append(max_reward_obj2_no_aug.float().mean())
                no_aug_score.append(max_reward_obj3_no_aug.float().mean())

                out = {
                    "no_aug_score": no_aug_score,
                    "aug_score": aug_score,
                }


        return state_td, out



