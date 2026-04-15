import abc
import copy
import importlib

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchrl.envs import EnvBase

from scipy.stats import ttest_rel
from tensordict import TensorDict
from torch.utils.data import DataLoader, Dataset
from torch import Tensor

from EasyNCO.utils.utils import getLogger

from EasyNCO.neural_solvers.methods.lih.critic_network import Critic_network as lih_critic_network
from EasyNCO.neural_solvers.methods.dact.critic_network import Critic_network as dact_critic_network
from EasyNCO.neural_solvers.methods.nlns.critic import CriticModel


logger = getLogger(__name__)


class REINFORCEBaseline(nn.Module, metaclass=abc.ABCMeta):
    """Base class for REINFORCE baselines"""

    def __init__(self, *args, **kw):
        super().__init__()
        pass

    @abc.abstractmethod
    def eval(self, batch: Tensor, reward: Tensor, env: EnvBase=None,**kwargs):
        """Evaluate baseline"""
        raise NotImplementedError

    def epoch_callback(self, *args, **kw):
        """Callback at the end of each epoch
        For example, update baseline parameters and obtain baseline values
        """
        pass

    def setup(self, *args, **kw):
        """To be called before training during setup phase
        This follow PyTorch Lightning's setup() convention
        """
        pass


class NoBaseline(REINFORCEBaseline):
    """No baseline: return 0 for baseline and neg_los"""

    def eval(self, batch, reward, env=None):
        return 0, 0  # No baseline, no neg_los


class SharedBaseline(REINFORCEBaseline):
    """
    Shared baseline: return mean of multi-trajectory reward as baseline.
    It is used for multi-trajectory training, such as POMO.
    """

    def eval(self, batch, reward, env=None, on_dim=1):  # e.g. [batch, pomo, ...]
        return reward.mean(dim=on_dim, keepdims=True), 0
        # shape: [batch, 1]


class ExponentialBaseline(REINFORCEBaseline):
    """Exponential baseline: return exponential moving average of reward as baseline

      The original AM paper:
      "With the rollout baseline, we use an exponential baseline (β = 0.8) during the first epoch, to stabilize initial learning.
      Although in many cases learning also succeeds without this ‘warmup’."
    Args:
        beta: Beta value for the exponential moving average，By default, it is 0.8
    """

    def __init__(self, beta=0.8):
        super(REINFORCEBaseline, self).__init__()

        self.beta = beta
        self.v = None

    def eval(self, batch, reward, env=None):
        if self.v is None:
            v = reward.mean()
        else:
            v = self.beta * self.v + (1.0 - self.beta) * reward.mean()
        self.v = v.detach()  # Detach since we never want to backprop
        return self.v, 0  # No loss


class MeanBaseline(REINFORCEBaseline):
    """
    Mean baseline: return mean of reward as baseline.
    However, it is not used for current NCO training.
    """

    def eval(self, batch, reward, env,**kwargs):
        return reward.mean(), 0


class WarmupBaseline(REINFORCEBaseline):
    """Warmup baseline: return convex combination of baseline and exponential baseline

    Args:
        baseline: Baseline to use after warmup, by default, it is the greedyrollout baseline
        n_epochs: Number of epochs to warmup
        warmup_exp_beta: Beta value for the exponential baseline during warmup
    """

    def __init__(self, baseline, n_epochs=1, warmup_exp_beta=0.8, problem_size=100):
        super(REINFORCEBaseline, self).__init__()

        self.baseline = baseline
        self.warmup_baseline = ExponentialBaseline(warmup_exp_beta)
        '''
        If alpha = 0, it means that we are in the warmup phase, so we use the ExponentialBaseline as the baseline.
        Otherwise, we use the greedyRollout baseline.
        '''
        self.alpha = 0

        assert n_epochs > 0, f"n_epochs to warmup must be positive, the got value ({n_epochs}) of n_epochs is non-positive."
        self.n_epochs = n_epochs

    def setup(self, *args, **kw):
        self.baseline.setup(*args, **kw)

    def eval(self, batch, reward, env=None):
        if self.alpha == 1:  # greedyRollout baseline
            return self.baseline.eval(batch, reward, env)
        if self.alpha == 0:  # ExponentialBaseline
            return self.warmup_baseline.eval(batch, reward, env)
        v_b, l_b = self.baseline.eval(batch, reward, env)
        v_wb, l_wb = self.warmup_baseline.eval(batch, reward, env)
        # Return convex combination of baseline and of loss
        return (
            self.alpha * v_b + (1 - self.alpha) * v_wb,  # reward
            self.alpha * l_b + (1 - self.alpha) * l_wb,  # loss
        )

    def epoch_callback(self, *args, **kw):
        # Need to call epoch callback of inner policy (also after first epoch if we have not used it)
        val_reward = self.baseline.epoch_callback(*args, **kw)
        if kw["epoch"] < self.n_epochs:  # only update alpha during warmup phase, that is the first epoch.
            self.alpha = (kw["epoch"] + 1) / float(self.n_epochs)
            logger.info("Set warmup alpha = {}".format(self.alpha))

        return val_reward


class CriticBaseline(REINFORCEBaseline):
    """Critic baseline: use critic network as baseline

    Args:
        critic: Critic network to use as baseline. If None, create a new critic network based on the environment
    """

    def __init__(self, critic):
        super(CriticBaseline, self).__init__()
        self.critic = critic
        self.critic_train = False

    def setup(self, *args, **kw):
        policy = kw['policy']
        self.critic_train = getattr(policy, 'critic_train', False)

        if self.critic_train:
            self.max_grad_norm = policy.critic_max_grad_norm
            self.critic_lr = policy.critic_lr
            self.critic_optim = optim.Adam(self.critic.parameters(), lr=self.critic_lr)
            self.critic.train()

    def eval(self, batch = None, reward = None, env = None, td :TensorDict = None ,**kwargs):
        # reward: shape: (batch, pomo=1)

        if self.critic_train:

            critic_est = - self.critic.forward(env).view(-1)
            # shape: (batch, )
            advantage = reward.squeeze(1) - critic_est # shape: (batch, )
            
            critic_loss = torch.mean(advantage ** 2)
            self.critic_optim.zero_grad()
            critic_loss.backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
            self.critic_optim.step()

            critic_est = critic_est.unsqueeze(1)
            # shape: (batch, pomo=1)
            return critic_est.detach(), 0
    
        else:
            v = self.critic(td,**kwargs)
            # Detach v since actor should not backprop through baseline, only for loss
            return v.detach().squeeze(), v.squeeze()


    def load_state_dict(self, state_dict):
        critic_state_dict = state_dict.get('critic', {})
        if not isinstance(critic_state_dict, dict):  # backwards compatibility
            critic_state_dict = critic_state_dict.state_dict()
        self.critic.load_state_dict({**self.critic.state_dict(), **critic_state_dict})



class RolloutBaseline(REINFORCEBaseline):
    """Rollout baseline: use greedy rollout as baseline.

    The baseline is used in the training of AM-based models.

    Args:
        bl_alpha: Alpha value for the baseline T-test. By default, it is 0.05.
    """

    def __init__(self, bl_alpha=0.05, **kw):
        super(RolloutBaseline, self).__init__()
        self.bl_alpha = bl_alpha
        self.device = None

    def setup(self, *args, **kw):
        self._update_policy(*args, **kw)

    def _update_policy(
        self, policy, env, batch_size=64, device="cpu", dataset_size=None, dataset=None
    ):
        """
        Update policy (=actor) and rollout baseline values.
        The baseline policy is updated if the candidate policy is better than the baseline policy.
        """
        self.policy = copy.deepcopy(policy).to(device)   # use deepcopy to avoid changing the original policy
        if dataset is None:
            logger.info("Creating evaluation dataset for rollout baseline")
            self.dataset_dl = env.generate_eval_instances(dataset_size, batch_size)

        logger.info("Evaluating baseline policy on evaluation dataset")
        self.bl_vals = (
            self.rollout(self.policy, env, device, self.dataset_dl).squeeze(-1).cpu().numpy()
        )
        self.mean = self.bl_vals.mean()
        self.device = device
        logger.info("The evaluation dataset and baseline policy have been updated successfully.")

    def eval(self, batch, reward, env):

        """
        Evaluate rollout baseline,using baseline model to evaluate the reward of the same dataset
        """
        rewards = (self.eval_policy(self.policy, env, batch_data=batch).detach())
        return rewards, 0

    def epoch_callback(self, policy, env, batch_size=64, device="cpu", epoch=None, dataset_size=None, dataset_dl=None):
        """Challenges the current baseline with the policy and replaces the baseline policy if it is improved"""
        logger.info("Evaluating candidate policy on evaluation dataset")
        candidate_vals = self.rollout(policy, env, device, dataset_dl).squeeze(-1).cpu().numpy()
        candidate_mean = candidate_vals.mean()

        logger.info(
            "Candidate mean: {:.3f}, Baseline mean: {:.3f}".format(
                -candidate_mean, -self.mean
            )
        )
        if candidate_mean - self.mean > 0:
            # Calc p value with inverse logic (costs)
            t, p = ttest_rel(-candidate_vals.reshape(-1), -self.bl_vals.reshape(-1))

            p_val = p / 2  # one-sided
            assert t < 0, "T-statistic should be negative"
            logger.info("p-value: {:.3f}".format(p_val))
            if p_val < self.bl_alpha:
                logger.info("Updating baseline")
                # 'dataset=None' to update the evaluation dataset
                self._update_policy(policy, env, batch_size, device, dataset_size)

        return candidate_mean

    def rollout(self, policy, env, device="cpu", dataset_dl: DataLoader = None):
        """Rollout the policy on the given dataset"""

        # if dataset is None, use the dataset of the baseline
        dataset_dl = self.dataset_dl if dataset_dl is None else dataset_dl

        policy.eval()
        policy = policy.to(device)

        rewards = torch.cat([self.eval_policy(policy, env, batch_data) for batch_data in dataset_dl], 0)
        # shape: (dataset_size,1)
        return rewards

    def eval_policy(self, policy, env, batch_data):
        '''
        Compare with torch.no_grad(), torch.inference_mode() has better performance
        and will force to close the gradient recording.
        And it cannot set gradient in the middle
        '''
        env.load_problems(batch_data, batch_size=batch_data.size(0))
        with torch.inference_mode():
            reset_td = env.reset()
            policy.set_decoder_strategy("greedy")
            policy.pre_forward(reset_td)

            done = False
            reward = None
            state_td = env.pre_step()
            while not done:
                next_td = policy(state_td)  # As long as it is not 'random', it is fine.
                state_td = env.step(next_td)
                reward = state_td["reward"]
                done = state_td["done"].all()

            return reward  # shape: (batch, 1)

    def __getstate__(self):
        """Do not include datasets in state to avoid pickling issues"""
        state = self.__dict__.copy()
        try:
            del state["dataset"]
        except KeyError:
            pass
        return state

    def __setstate__(self, state):
        """Restore datasets after unpickling. Will be restored in setup"""
        self.__dict__.update(state)
        self.dataset = None

REINFORCE_BASELINES_REGISTRY = {
    "no": NoBaseline,
    "shared": SharedBaseline,
    "exponential": ExponentialBaseline,
    "mean": MeanBaseline,
    "rollout": RolloutBaseline,
    "warmup": WarmupBaseline,
    "critic": CriticBaseline,
}

def get_reinforce_baseline(name,**kw):
    """Get a REINFORCE baseline by name
    The rollout baseline default to warmup baseline with one epoch of
    exponential baseline and the greedy rollout
    """
    if name == "warmup":
        inner_baseline = kw.get("baseline", "rollout")
        if not isinstance(inner_baseline, REINFORCEBaseline):
            inner_baseline = get_reinforce_baseline(inner_baseline, **kw)
        return WarmupBaseline(inner_baseline, **kw)
    elif name == "rollout":
        warmup_epochs = kw.get("n_epochs", 1)
        warmup_exp_beta = kw.get("exp_beta", 0.8)
        bl_alpha = kw.get("bl_alpha", 0.05)
        return WarmupBaseline(
            RolloutBaseline(bl_alpha=bl_alpha), warmup_epochs, warmup_exp_beta
        )
    elif name == "critic":
        method_name = kw.get('method_name')
        if method_name == 'lih':
            critic = CriticBaseline(lih_critic_network(env_name=kw.get("env_name")))
        elif method_name == 'dact':
            critic = CriticBaseline(dact_critic_network(env_name=kw.get("env_name")))
        elif method_name == 'nlns':
            critic = CriticBaseline(CriticModel())
        return critic

    if name is None:
        name = "no"  # default to no baseline
    baseline_cls = REINFORCE_BASELINES_REGISTRY.get(name, None)
    if baseline_cls is None:
        raise ValueError(
            f"Unknown baseline {baseline_cls}. Available baselines: {REINFORCE_BASELINES_REGISTRY.keys()}"
        )
    return baseline_cls(**kw)