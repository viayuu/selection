import copy

import torch.nn.functional as F
from lightning import LightningModule
from torch.optim import *
from torch.optim.lr_scheduler import *
from lightning.pytorch.utilities.types import EVAL_DATALOADERS, STEP_OUTPUT
from ot.backend import torch
from torch.optim import Adam as Optimizer
from torch.optim.lr_scheduler import MultiStepLR as Scheduler
from torch import nn
from torchrl.envs import EnvBase
from tensordict import TensorDict
from lightning.pytorch.callbacks import ModelCheckpoint, TQDMProgressBar
from typing import Tuple, Any
from torch.utils.data import DataLoader

from EasyNCO.methods.operators.neural_operators.difusco.schedule import two_opt_refine
from EasyNCO.phases.train.rl import get_reinforce_baseline
from EasyNCO.utils.pylogger import get_pylogger
from EasyNCO.utils.utils import *
# from lightning.pytorch.utilities.rank_zero import rank_zero_info

from EasyNCO.data import *

logger = get_pylogger(__name__)

class REINFORCELightning(LightningModule):
    def __init__(self,
                 policy: nn.Module, # In distillation model, it is student's policy.
                 env: EnvBase, # In distillation model, it is student's env.
                 baseline: str,
                 problem_size: int,
                 batch_size: int,
                 episodes: int,
                 first_mode: str,
                 optimizer_params: dict,
                 optimizer_type: str = 'Adam',
                 scheduler_type: str = None,
                 val_aug_flag: bool = False,
                 val_episode: int = 1000,
                 val_batch_size: int = 1000,
                 bl_data_size: int = 10000,  # only for evaluation of the baseline
                 bl_batch_size: int = 10000, # baseline update parameter, used for AM-style method
                 val_env: EnvBase = None,
                 **kwargs):
        super().__init__()
        self.policy = policy
        self.env = env
        self.model_name=kwargs.get('model_name')
        # train
        self.problem_size = problem_size
        self.batch_size = batch_size
        self.train_episodes = episodes
        self.optimizer_params = optimizer_params

        self.val_aug_flag = val_aug_flag # To indicate whether augment validation set.
        self.bl_data_size = bl_data_size
        self.bl_batch_size = bl_batch_size
        self.test_num_episodes = 0

        self.optimizer_type = optimizer_type
        self.scheduler_type = scheduler_type

        self.selected = None
        self.prob = None
        self.first_mode = first_mode

        # train AverageMeter
        self.avgScore_one_epoch = AverageMeter()
        self.avgLoss_one_epoch = AverageMeter()

        # test AverageMeter
        self.score_AM = AverageMeter()
        self.aug_score_AM = AverageMeter()
        self.aug_factor = kwargs.get('aug_factor', 1)

        self.time_estimator = TimeEstimator()  # Time estimator
        self.trained_num_episodes = 0

        self.training_step_outputs = []

        # automatically save hyperparameters that are transferred to the model's hparams attribute, except for the policy and env
        self.save_hyperparameters(logger=False,ignore=['policy', 'env'])

        # we use a single class to handle all baseline methods
        self.baseline_name = baseline
        self.baseline = get_reinforce_baseline(self.baseline_name)

        # ELG settings
        self.elg_start_step = kwargs.get('elg_start_step', None)

        # path of datasets
        self.train_data_path = kwargs.get('train_data_path', None)
        self.test_data_path = kwargs.get('test_data_path', None)
        self.val_data_path = kwargs.get('val_data_path', None)

        # test
        self.test_episodes = episodes
        self.test_batch_size = batch_size
        self.test_strategy = kwargs.get('test_strategy', None)
        self.do_val = kwargs.get('do_val', False)
        self.valid_episode = val_episode
        self.valid_batch_size = val_batch_size
        self.val_env = val_env

        # glop settings
        # train
        self.data_distribution = kwargs.get('data_distribution','uniform')
        # test
        self.glop_params=kwargs.get('glop_params')
        if self.model_name=="glop":
            self.before_score_AM = AverageMeter()
            self.revision_score_AM = AverageMeter()

        # AMDKD settings
        self.distillation = kwargs.get('distillation')
        self.teacher_policy = kwargs.get('teacher_policy')
        self.teacher_env = kwargs.get('teacher_env')
        self.rl_alpha = kwargs.get('rl_alpha')
        self.valid_path = kwargs.get('valid_path')
        self.valid_dataloader = {}
        self.train_data_params = kwargs.get('train_data_params', {})

    def on_fit_start(self) -> None:
        '''
        This function is called at the beginning of the training process.
        :return:
        '''
        self.time_estimator.reset(self.current_epoch+1)
        self.baseline_env = copy.deepcopy(self.env)
        setup_dict = {
            "policy": self.policy,
            "env": self.baseline_env,
            "batch_size": self.bl_batch_size,
            "device": get_lightning_device(self),
            "dataset_size": self.bl_data_size,
            "first_mode": self.first_mode
        }
        if not self.val_aug_flag:
            self.baseline_env.aug_type = None
        self.baseline.setup(**setup_dict)
        # if self.distillation:
        #     for key, value in self.valid_path.items():
        #         self.valid_dataloader[key] = self.env.generate_eval_instances(self.val_data_size, self.val_batch_size, )

    def on_test_start(self) -> None:
        self.time_estimator.reset(self.current_epoch + 1)

    def training_step(self, batch, batch_idx):
        """
        This function is called for each batch of data. It is used to calculate the loss and update the model.
        :param batch:
        :param batch_idx:
        :return: loss
        """

        if self.elg_start_step is not None and self.current_epoch == self.elg_start_step:
            self.policy.enable_local_policy()

        self.env.load_problems(batch, batch.size(0))
        self.trained_num_episodes += self.env.env_batch_size

        if self.distillation:
            state_td, policy_out = self.play_episode(self.policy, self.env, 'sampling', self.first_mode)
            self.teacher_policy.eval()
            with torch.no_grad():
                self.teacher_policy.label = self.env.selected_node_list
                self.teacher_env.load_problems(self.env.problems, self.env.env_batch_size)
                teacher_state_td, teacher_policy_out = self.play_episode(self.teacher_policy, self.teacher_env, 'sampling', self.first_mode)
            loss = self.calculate_loss(policy_out, teacher_policy_out=teacher_policy_out)

        else:
            state_td, policy_out = self.play_episode(self.policy, self.env, 'sampling', self.first_mode)
            loss = self.calculate_loss(policy_out)

        reward = policy_out["reward"]  # shape: (batch, pomo)
        # Score
        ###############################################
        max_pomo_reward, _ = reward.max(dim=1)  # get best results from pomo
        score_mean = -max_pomo_reward.float().mean()  # negative sign to make positive value

        # prog_bar: whether to show the metric on the progress bar
        self.log('batch_reward', score_mean, on_step=True, on_epoch=False, prog_bar=True, logger=True)
        self.log('batch_loss', loss, on_step=True, on_epoch=False, prog_bar=False, logger=True)

        self.avgScore_one_epoch.update(score_mean.item(), self.batch_size)
        self.avgLoss_one_epoch.update(loss.item(), self.batch_size)

        return loss

    def test_step(self, batch, batch_idx, env = None):
        """
        This function is called for each batch of data. Used to evaluate the performance on the given dataset
        :param batch:
        :param batch_idx:
        :return: performance metric
        """
        self.test_num_episodes += batch.size(0)
        if self.model_name=="glop":
            from EasyNCO.methods.main_loop.glop.test_utils import glop_test_one_batch
            before_score, revision_score = glop_test_one_batch(batch,self.env,self.policy,self.glop_params,self.test_strategy)
            self.before_score_AM.update(before_score, batch.size(0))
            self.revision_score_AM.update(revision_score, batch.size(0))

            # Return
            self.log_dict({'before_score': before_score,
                           'revision_score': revision_score})
        else:
            if env is None:
                env = self.env

            with torch.inference_mode():
                env.load_problems(batch, batch_size=batch.size(0))
            score, aug_score  = self.test_one_batch(batch.size(0), env)

            self.score_AM.update(score, batch.size(0))
            self.aug_score_AM.update(aug_score, batch.size(0))

            # Return
            self.log_dict({'no_aug_score' : score,
                               'aug_score' : aug_score})

    def validation_step(self, batch, batch_idx):
        if self.do_val:
            return self.test_step(batch, batch_idx, env = self.val_env)
        else:
            pass

    def test_one_batch(self, batch_size, env):

        self.policy.set_decoder_strategy(strategy = self.test_strategy)
        self.policy.eval()

        reset_state = env.reset()
        self.policy.pre_forward(reset_state)

        # POMO rollout
        state_td = env.pre_step()
        reward = state_td['reward']; done_all = state_td['done']
        done = done_all.all()

        if env.env_name == 'kp':
            while not done:
                next_td = self.policy(state_td)
                # shape: (batch, pomo)
                action_w_finished = next_td['action'].clone()
                action_w_finished[next_td['done']] = self.problem_size
                next_td['action'] = action_w_finished
                state_td = env.step(next_td)
                reward = state_td['reward']; done = state_td['done'].all()
        else:
            while not done:
                next_td = self.policy(state_td, self.first_mode)
                # shape: (batch, pomo)
                state_td = env.step(next_td)
                reward = state_td['reward']; done = state_td['done'].all()

        # Return
        aug_reward = reward.reshape(self.aug_factor, batch_size, self.env.pomo_size)
        # shape: (augmentation, batch, pomo)

        max_pomo_reward, _ = aug_reward.max(dim = 2) # best result from pomo
        # shape: (augmentation, batch)

        no_aug_reward = -max_pomo_reward[0, :].float().mean() # negative sign to make positive value

        max_aug_pomo_reward, _ = max_pomo_reward.max(dim = 0) # get best result from augmentation
        # shape (batch, )
        aug_score = -max_aug_pomo_reward.float().mean() # negative sign to make positive value

        return no_aug_reward, aug_score

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
        if self.distillation:
            student_log_probs = (policy_out["kd_probs"] + 1e-5).log()
            teacher_probs = kwargs["teacher_policy_out"]["kd_probs"] + 1e-5
            kd_loss = F.kl_div(student_log_probs, teacher_probs)
            return self.rl_alpha * task_loss + (1-self.rl_alpha) * kd_loss
        else:
            return task_loss

    def play_episode(self, policy, env, decoder_strategy: str = "sampling", first_mode: str = None) -> Tuple[TensorDict, dict]:
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
        kd_probs = torch.zeros(size=(env.batch_size[0], env.pomo_size, env.problems.size(-2), 0)) # used for distillation model

        done = False
        reward = None
        state_td = env.pre_step()
        if env.env_name == 'kp':
            while not done:
                temp_td = policy(state_td, first_mode = first_mode)
                selected = temp_td.get("action", None); prob = temp_td.get("prob", None)

                action_w_finished = selected.clone()
                action_w_finished[temp_td.get('done', False)] = self.problem_size
                temp_td['action'] = action_w_finished
                next_td = env.step(temp_td)

                chosen_prob = prob.clone()
                chosen_prob[next_td['done']] = 1
                log_likelihood = torch.cat((log_likelihood, chosen_prob[:, :, None]), dim = 2)
                reward = next_td["reward"]
                done = next_td["done"].all()
        else:
            while not done:
                next_td = policy(state_td, first_mode=first_mode)
                prob = next_td.get("prob", None)
                if self.distillation:
                    probs = next_td.get("probs", None)
                    kd_probs = torch.cat((kd_probs, probs[:, :, :, None]), dim=-1)
                state_td = env.step(next_td)
                log_likelihood = torch.cat((log_likelihood, prob[:, :, None]), dim=2)
                reward = state_td["reward"]
                done = state_td["done"].all()

        policy_out = {
            "reward": reward,
            "log_likelihood": log_likelihood,
            "kd_probs": kd_probs
        }
        return state_td, policy_out

    def configure_optimizers(self):
        '''
        This function is used to create the optimizer and scheduler.
        Note that there are some methods don't use the scheduler.
        '''
        Optimizer = {'Adam': Adam,
                     'AdamW': AdamW}
        Scheduler = {'MultiStepLR': MultiStepLR,
                     'ExponentialLR': ExponentialLR}

        optimizer = Optimizer[self.optimizer_type](self.policy.parameters(), **self.optimizer_params["optimizer"])
        if self.scheduler_type is not None:
            scheduler = Scheduler[self.scheduler_type](optimizer, **self.optimizer_params["scheduler"])
            return {"optimizer": optimizer, "lr_scheduler": scheduler}
        else:
            return {"optimizer": optimizer}

    def train_dataloader(self):
        '''
        This function is used to create the dataloader based on DataLoader.
        '''
        data_generator = {
            "TSPEnv": TSPGenerator,
            "CVRPEnv": CVRPGenerator,
            "KPEnv": KPGenerator,
            "ASHPPEnv": ATSPGenerator,
            "min_max_mTSPEnv": TSPGenerator,
            "min_max_mPDPEnv": TSPGenerator,
            "min_max_MDVRPEnv": TSPGenerator,
            "min_max_FMDVRPEnv": TSPGenerator,
            "ATSPEnv": ATSPGenerator,
            "FFSPEnv": FFSPGenerator,
        }
        env_name = self.env._get_name()
        if env_name not in data_generator.keys():
            raise ValueError(f"Environment name '{env_name}' is not supported. Available data generators: {data_generator.keys()}")

        if self.model_name=="glop_revision":

            return data_generator[env_name](self.train_episodes if self.env.aug_type is None else self.train_episodes // self.env.aug_factor,
                self.problem_size, self.batch_size, self.device, path=self.train_data_path,distribution=self.data_distribution)
        else:
            return data_generator[env_name](self.train_episodes if self.env.aug_type is None else self.train_episodes//self.env.aug_factor,
                                            self.problem_size, self.batch_size, **self.train_data_params)

    def test_dataloader(self):
        """
        This function is used to load test dataloader
        """
        data_generator = {
            "TSPEnv": TSPGenerator,
            "CVRPEnv": CVRPGenerator,
            "KPEnv": KPGenerator,
            "ATSPEnv": ATSPGenerator,
            "PCTSPEnv": PCTSPGenerator,
            "min_max_mTSPEnv": TSPGenerator,
            "min_max_mPDPEnv": TSPGenerator,
            "min_max_MDVRPEnv": TSPGenerator,
            "min_max_FMDVRPEnv": TSPGenerator,
            "FFSPEnv": FFSPGenerator,
        }
        if self.model_name == 'glop':
            env_name=f"{self.glop_params.main_problem}Env"
        else:
            env_name = self.env._get_name()
        if env_name not in data_generator.keys():
            raise ValueError(f"Environment name '{env_name}' is not supported. Available data generators: {data_generator.keys()}")

        return data_generator[env_name](self.test_episodes, self.problem_size, self.test_batch_size, self.device, self.test_data_path)

    def val_dataloader(self):
        """
        This function is used to load valid dataloader
        """
        data_generator = {
            "TSPEnv": TSPGenerator,
            "CVRPEnv": CVRPGenerator,
            "KPEnv": KPGenerator,
            "ASHPPEnv": ATSPGenerator,
            "min_max_mTSPEnv": TSPGenerator,
            "min_max_mPDPEnv": TSPGenerator,
            "min_max_MDVRPEnv": TSPGenerator,
            "min_max_FMDVRPEnv": TSPGenerator,
            "ATSPEnv": ATSPGenerator,
            "FFSPEnv": FFSPGenerator,
        }
        env_name = self.env._get_name()
        if env_name not in data_generator.keys():
            raise ValueError(f"Environment name '{env_name}' is not supported. Available data generators: {data_generator.keys()}")

        return data_generator[env_name](self.valid_episode, self.problem_size, self.valid_batch_size, self.device, self.val_data_path)

    def on_train_epoch_end(self):
        '''
        There is a bug in the current platform version that does not allow to record the reward and loss of the epoch.
        '''

        # self.log('epoch_score',self.avgScore_one_epoch.avg,on_step=False,on_epoch=True,prog_bar=True,logger=True)
        # self.log('epoch_loss',self.avgLoss_one_epoch.avg,on_step=False,on_epoch=True,prog_bar=True,logger=True)
        logger.info("Epoch {:3d}/{:3d}: Train {:3d}/{:3d}({:1.1f}%)  Score: {:.4f}, Loss: {:.4f}"
                    .format(self.current_epoch+1, self.trainer.max_epochs, self.trained_num_episodes, self.train_episodes,
                            100. * self.trained_num_episodes / self.train_episodes,
                            self.avgScore_one_epoch.avg, self.avgLoss_one_epoch.avg))
        # Reset the average meter
        self.avgScore_one_epoch.reset()
        self.avgLoss_one_epoch.reset()
        self.trained_num_episodes = 0
        elapsed_time_str, remain_time_str = self.time_estimator.get_est_string(self.current_epoch+1, self.trainer.max_epochs)
        logger.info("Epoch {:3d}/{:3d}: Time Est.: Elapsed[{}], Remain[{}]".format(
            self.current_epoch+1, self.trainer.max_epochs, elapsed_time_str, remain_time_str))

        """Callback for end of training epoch: we evaluate the baseline"""
        if self.baseline_name in ['warmup', 'rollout']:
            logger.info("Epoch {:3d}/{:3d}: Evaluate the baseline".format(self.current_epoch+1, self.trainer.max_epochs))

        if self.distillation:
            reward = torch.zeros(size=(3,), dtype=torch.float)
            i = 0
            for key in self.valid_dataloader.keys():
                valid_reward = self.baseline.epoch_callback(
                                    self.policy,
                                    env=self.baseline_env,
                                    batch_size=self.bl_batch_size,
                                    device=get_lightning_device(self),
                                    epoch=self.current_epoch,
                                    dataset_size=self.bl_data_size,
                                    dataset_dl=self.valid_dataloader[key]
                                    )
                reward[i] = -torch.tensor(valid_reward)
                i += 1
            self.teacher_policy.get_cur_teacher(self.current_epoch, reward)
            self.train_data_params['data_type'] = self.teacher_policy.cur_teacher

        else:
            _ = self.baseline.epoch_callback(
                    self.policy,
                    env=self.baseline_env,
                    batch_size=self.bl_batch_size,
                    device=get_lightning_device(self),
                    epoch=self.current_epoch,
                    dataset_size=self.bl_data_size,
                    )

    def on_test_end(self) -> None:
        # TODO: display test result
        self.score_AM.reset()
        self.aug_score_AM.reset()
        self.test_num_episodes = 0

'''
Note that the next two class/function are not used in the current version of the platform, but they are useful for future versions.
'''
class CustomProgressBar(TQDMProgressBar):
    def init_train_tqdm(self):
        bar = super().init_train_tqdm()
        bar.ncols = 200  # set the maximum width of the progress bar, but it is not working.
        return bar

def checkpoint_callback(dirpath: str,every_n_epochs:int,model_name:str) -> ModelCheckpoint:
    """
    This function returns a ModelCheckpoint callback that saves the model every n epochs.
    :param dirpath:
    :param every_n_epochs:
    :param model_name:
    :return: checkpoint_callback
    """
    checkpoint_callback = ModelCheckpoint(dirpath=dirpath,
                                        filename=model_name+'_ckpt_epoch-{epoch:02d}',  # filename
                                        save_top_k=-1,  # saving all checkpoints
                                        every_n_epochs=every_n_epochs  # save it every n epochs
    )
    return checkpoint_callback
