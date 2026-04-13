import copy

import torch
from lightning import LightningModule
from torch.optim import *
from torch.optim.lr_scheduler import *
from torch import nn, tensor
from torch.utils.data import DataLoader, TensorDataset
from torchrl.envs import EnvBase

from EasyNCO.phases import get_reinforce_baseline
from EasyNCO.utils.utils import *
from EasyNCO.data import generate_data
from EasyNCO.neural_solvers.pipeline import Initialization, Iteration
from EasyNCO.sandbox.test_batch import SandboxTestBatch


import time

logger = getLogger(__name__)

class ARREINFORCELightning(LightningModule):
    def __init__(self,
                 policy: nn.Module,
                 env: EnvBase,
                 problem_size: int,
                 initialization: Initialization,
                 iteration: Iteration,
                 sandbox_test_batch: SandboxTestBatch=None,
                 baseline: str='shared',
                 optimizer_params: dict=None,
                 optimizer_type: str = 'Adam',
                 batch_size: int = 128,
                 episodes: int = 1280,
                 scheduler_type: str = None,
                 val_aug_flag: bool = False,
                 val_episode: int = 10000,
                 val_batch_size: int = 10000,
                 bl_data_size: int = 10000,  # only for evaluation of the baseline
                 bl_batch_size: int = 10000, # baseline update parameter, used for AM-style method
                 val_env: EnvBase = None,
                 decoder_strategy: str = "sampling",
                 every_n_steps_output: int = 100,
                 **kwargs):
        super().__init__()
        self.policy = policy
        self.env = env
        self.method_name = kwargs.get('method_name',None)
        self.problem=kwargs.get('problem',None)
        self.decoder_strategy = decoder_strategy
        self.initialization = initialization
        self.iteration = iteration
        self.sandbox_test_batch = sandbox_test_batch
        self.pomo_size = self.env.pomo_size

        # train
        self.problem_size = problem_size
        self.batch_size = batch_size
        self.train_episodes = episodes
        self.optimizer_params = optimizer_params
        self.every_n_steps_output = every_n_steps_output

        self.val_aug_flag = val_aug_flag # To indicate whether augment validation set.
        self.bl_data_size = bl_data_size
        self.bl_batch_size = bl_batch_size
        self.test_num_episodes = 0

        self.optimizer_type = optimizer_type
        self.scheduler_type = scheduler_type

        self.selected = None
        self.prob = None

        # train AverageMeter
        self.avgScore_one_epoch = AverageMeter()
        self.avgLoss_one_epoch = AverageMeter()

        # test AverageMeter, also used for validation
        self.score = AverageMeter()
        self.aug_score = AverageMeter()

        self.gap = AverageMeter()
        self.aug_gap = AverageMeter()

        # for calculating standard deviation
        self.gap_list = []
        self.aug_gap_list = []

        self.time_estimator = TimeEstimator()  # Time estimator
        self.trained_num_episodes = 0

        self.training_step_outputs = []

        # automatically save hyperparameters that are transferred to the model's hparams attribute, except for the policy and env
        self.save_hyperparameters(logger=False,ignore=['policy', 'env', 'val_env'])

        # we use a single class to handle all baseline methods
        self.baseline_name = baseline

        if self.baseline_name == 'critic':
            self.baseline = get_reinforce_baseline(self.baseline_name, env_name=self.env.env_name, method_name=self.method_name)
        else:
            self.baseline = get_reinforce_baseline(self.baseline_name)

        # ELG settings
        self.elg_start_step = kwargs.get('elg_start_step', None)

        # path of datasets
        self.train_data_path = kwargs.get('train_data_path', None) #  RL training, do not need pre-generated data,glop stage2
        self.test_data_path = kwargs.get('test_data_path', None)
        self.val_data_path = kwargs.get('val_data_path', None)

        # test
        self.test_episodes = episodes
        self.test_batch_size = batch_size
        self.do_val = kwargs.get('do_val', False)
        self.valid_episode = val_episode
        self.valid_batch_size = val_batch_size
        self.val_env = val_env

        # glop settings
        # train
        self.data_distribution = kwargs.get('data_distribution', 'uniform')
        # test

        self.train_data_params = kwargs.get('train_data_params', {})
        self.test_data_params = kwargs.get('test_data_params', {})
        self.val_data_params = kwargs.get('val_data_params', {})
        self.varying_data_params = kwargs.get('varying_data_params', {})
        self.start_time=0

        # improve
        self.improve = kwargs.get('improve',False)
        if self.improve:
            self.automatic_optimization = False
            self.iteration.optimizers = self.optimizers
            self.iteration.manual_backward = self.manual_backward

        if self.method_name == 'lih':
            self.iteration.baseline = self.baseline
            if self.env.env_name == 'cvrp':
                self.iteration.old_policy = copy.deepcopy(self.policy)
                self.iteration.old_policy_state_dict = self.policy.state_dict()

        self.initialization_params = kwargs.get('initialization_params', {})
        self.iteration_params = kwargs.get('iteration_params', {})

        # omni
        if self.method_name == 'omni':
            self.fine_tune = kwargs.get('fine_tune')
            self.mode = kwargs.get('mode')
            if self.mode == 'train' and self.fine_tune.enable == False:
                self.automatic_optimization = False
                self.initialization.optimizers = self.optimizers
                self.initialization.optimizer_params = self.optimizer_params
                self.initialization.calculate_loss = self.calculate_loss
                self.initialization.set_parameter(**self.initialization_params)

        self.customized = kwargs.get(
            "customized", False
        )  # whether to compose initialization and iteration


        # PSL
        if self.method_name == 'psl':
            self.num_target = self.env.num_target
            self.score_psl = {}
            self.aug_score_psl = {}
            for i in range(self.num_target):
                self.score_psl[i] = AverageMeter()
                self.aug_score_psl[i] = AverageMeter()

    def on_fit_start(self) -> None:
        '''
        This function is called at the beginning of the fitting process.
        on_fit_start
        ├── on_train_start
        │   ├── on_train_epoch_start
        │   └── on_train_batch_start
        ├── on_validation_start
        └── on_test_start
        '''
        self.env.device = get_lightning_device(self)
        if self.do_val:
            self.val_env.device = get_lightning_device(self)
        self.time_estimator.reset(self.current_epoch+1)

    def on_train_start(self) -> None:
        """
        This function is called at the beginning of the training process.
        :return:
        """
        self.baseline_env = copy.deepcopy(self.env)
        setup_dict = {
            "policy": self.policy,
            "env": self.baseline_env,
            "batch_size": self.bl_batch_size,
            "device": get_lightning_device(self),
            "dataset_size": self.bl_data_size,
        }
        if not self.val_aug_flag:
            self.baseline_env.aug_type = None
        self.baseline.setup(**setup_dict)
        logger.info("RL Baseline ==> [{}] is already initialized.".format(self.baseline_name))

    def on_test_start(self) -> None:
        '''
        Note that we skip the fit function and directly execute the test function in inference process.
        '''
        self.env.device = get_lightning_device(self)
        self.time_estimator.reset(self.current_epoch + 1)
        self.start_time=time.time()

    def training_step(self, batch, batch_idx):
        """
        This function is called for each batch of data. It is used to calculate the loss and update the model.
        :param batch:
        :param batch_idx:
        :return: loss
        """
        if self.method_name == 'omni':
            self.initialization_params['current_epoch'] = self.current_epoch
        state_td, initialization_out = self.initialization.run(self.env, batch, self.decoder_strategy, "train",**self.initialization_params)
        iteration_out = self.iteration.run(state_td, self.env, initialization_out, "train", **self.iteration_params)
        self.trained_num_episodes += self.env.env_batch_size
        if self.method_name == 'omni':
            self.train_episodes = self.env.env_batch_size
        loss = self.calculate_loss(iteration_out)
        reward = iteration_out["reward"]  # shape: (batch, pomo)
        # Score
        ###############################################
        max_pomo_reward, _ = reward.max(dim=1)  # get best results from pomo
        score_mean = (
            -max_pomo_reward.float().mean()
        )  # negative sign to make positive value

        # prog_bar: whether to show the metric on the progress bar
        self.log('batch_reward', score_mean, on_step=True, on_epoch=False, prog_bar=True, logger=True)
        self.log('batch_loss', loss, on_step=True, on_epoch=False, prog_bar=False, logger=True)

        self.avgScore_one_epoch.update(score_mean.item(), self.batch_size)
        self.avgLoss_one_epoch.update(loss.item(), self.batch_size)

        if batch_idx+1 <= 5 or (batch_idx+1) % self.every_n_steps_output == 0 or batch_idx+1 == self.trainer.num_training_batches:
            logger.info("Train ==> Epoch {:3d}/{:3d}: Completed {:3d}/{:3d}({:1.1f}%)  Score: {:.4f}, Loss: {:.4f}".format(
                self.current_epoch+1, self.trainer.max_epochs, batch_idx+1, self.trainer.num_training_batches, 100. * (batch_idx+1) / self.trainer.num_training_batches,
                self.avgScore_one_epoch.avg, self.avgLoss_one_epoch.avg))

        return loss

    def test_step(self, batch, batch_idx, env = None, decoder_strategy: str = None):
        """
        This function is called for each batch of data.
        Used to evaluate the performance on the given dataset
        :param batch:
        :param batch_idx:
        :return: performance metric
        """
        if isinstance(batch, dict):
            if batch.get("depot_node_tw_start", None) is not None:
                optimal = solution = lib_data = name = edge_weight_type = None
                batch_size = batch["depot_node_tw_start"].size(0)
            else:
                optimal, solution, lib_data, name, edge_weight_type, batch_size = batch.get("optimal", None), batch.get("solution", None), batch.get("lib_data", None), batch.get("name", None), batch.get("edge_weight_type") ,batch.get("depot_node_xy", batch["data"]).size(0)
                if batch.get("num_sample", None) is not None:
                    self.test_episodes = batch.get("num_sample").item()
                num_job,num_machine,num_ope = batch.get("num_job",None),batch.get("num_machine",None),batch.get("num_ope",None)
                batch = batch.get("data", batch)
        elif isinstance(batch, list):
            optimal = solution = lib_data = name = edge_weight_type = None
            batch_size = batch[0].size(0)
        else:
            optimal = solution = lib_data = name = edge_weight_type = None
            batch_size = batch.size(0)
        self.test_num_episodes += batch_size

        if lib_data is not None:
            self.env.lib_data = {"data": lib_data, "edge_weight_type": edge_weight_type}

        if env is None:
            env = self.env

        if decoder_strategy is None:
            decoder_strategy = self.decoder_strategy

        if "vrp" in env.env_name and "mdvrp" not in env.env_name:
            if isinstance(batch, dict):
                env.problem_size  = batch["depot_node_xy"].size(1)-1
            else:
                env.problem_size = batch.size(1) -1
        else:
            env.problem_size = batch[0].size(1) if isinstance(batch, list) else batch.size(1)

        if self.pomo_size > 1:
            env.pomo_size = env.problem_size
        else:
            env.pomo_size = 1

        if "jsp" in env.env_name :
            env.num_job,env.num_machine,env.num_ope = num_job,num_machine,num_ope

        if self.sandbox_test_batch is not None:
            test_out = self.sandbox_test_batch.run(batch)
        else:
            state_td, initialization_out = self.initialization.run(env, batch, decoder_strategy, "eval", **self.initialization_params)
            if self.customized:
                env.pomo_size = 1 # reset pomo_size to 1 for iteration
                env.aug_factor = 1 # reset aug_factor to 1 for iteration
                env.aug_flag = False # reset aug_flag to False for iteration
            self.iteration_params["problems"] = batch
            self.iteration_params["decoder_strategy"] = decoder_strategy
            test_out = self.iteration.run(state_td, env, initialization_out, "eval", **self.iteration_params)

        score, aug_score = test_out.values()
        if isinstance(score, float):
            score = tensor(score, device=self.device)
            self.score.update(score.item(),batch_size)
        elif isinstance(score, list):
            for i in range(self.num_target):
                self.score_psl[i].update(score[i], batch_size)            
        
        if isinstance(aug_score, float):
            aug_score = tensor(aug_score, device=self.device)
            self.aug_score.update(aug_score.item(), batch_size)
        elif isinstance(aug_score, list):
            for i in range(self.num_target):
                self.aug_score_psl[i].update(aug_score[i], batch_size)

        elapsed_time_str, remain_time_str = self.time_estimator.get_est_string(self.test_num_episodes, self.test_episodes)
        logger.info("Eval ==> {:3d}/{:3d}({:1.2f}%)  Time Est.: Elapsed[{}], Remain[{}]".format(
             self.test_num_episodes, self.test_episodes, 100. * self.test_num_episodes / self.test_episodes, elapsed_time_str, remain_time_str))

        if self.method_name == 'psl':

            for i in range(self.num_target):
                logger.info("Eval ==> Problem[{}], Obj{}, Size[{}], {}Completed {:3d}/{:3d}({:1.2f}%), Score: {:.4f}, Augmented Score: {:.4f}{}".format(
                    env.env_name, i+1, env.problem_size, "Name[" + name[0] + "], " if name is not None else "",
                    self.test_num_episodes, self.test_episodes, 100. * self.test_num_episodes / self.test_episodes,
                score[i].item(), aug_score[i].item(), f", Gap: {gap_mean:.4f}%, Augmented Gap: {aug_gap_mean:.4f}%" if optimal is not None else "",))

                logger.info("Eval ==> Score Summary: Avg: {:.4f}, Avg Augmented: {:.4f}".format(self.score_psl[i].avg, self.aug_score_psl[i].avg))
        else:
            gap_mean, aug_gap_mean = None, None
            if optimal is not None:
                gap = 100 * ((score - optimal) / optimal)
                self.gap_list.extend(gap.tolist())
                gap_mean = gap.mean().item()
                aug_gap = 100 * ((aug_score - optimal) / optimal)
                self.aug_gap_list.extend(aug_gap.tolist())
                aug_gap_mean = aug_gap.mean().item()
                self.gap.update(gap_mean, batch.size(0))
                self.aug_gap.update(aug_gap_mean, batch.size(0))
                test_out.update({"gap": gap_mean, "aug_gap": aug_gap_mean})
            logger.info("Eval ==> Problem[{}], Size[{}], {}Completed {:3d}/{:3d}({:1.2f}%), Score: {:.4f}, Augmented Score: {:.4f}{}".format(
                env.env_name, env.problem_size, "Name[" + name[0] + "], " if name is not None else "",
                self.test_num_episodes, self.test_episodes, 100. * self.test_num_episodes / self.test_episodes,
            score.item(), aug_score.item(), f", Gap: {gap_mean:.4f}%, Augmented Gap: {aug_gap_mean:.4f}%" if optimal is not None else "",))

            logger.info("Eval ==> Score Summary: Avg: {:.4f}, Avg Augmented: {:.4f}".format(self.score.avg, self.aug_score.avg))
            if self.gap.avg > 0:
                logger.info("Eval ==> Gap Summary: Avg: {:.4f}%, Avg Augmented: {:.4f}%, Std: {:.4f}%, Std Augmented: {:.4f}%".format(
                        self.gap.avg, self.aug_gap.avg, np.std(self.gap_list), np.std(self.aug_gap_list)))
        
        logger.info("============================================================================")
        
        # Return
        if self.method_name == 'psl':
            self.log("score_obj1", self.score_psl[0].avg)
            self.log("aug_score_obj1", self.aug_score_psl[0].avg)
            if self.num_target >= 2:
                self.log("score_obj2", self.score_psl[1].avg)
                self.log("aug_score_obj2", self.aug_score_psl[1].avg)
            if self.num_target >= 3:
                self.log("score_obj3", self.score_psl[2].avg)
                self.log("aug_score_obj3", self.aug_score_psl[2].avg)
        else:
            self.log_dict(test_out)



    def validation_step(self, batch, batch_idx):
        if self.do_val:
            self.test_episodes = self.valid_episode
            return self.test_step(batch, batch_idx, env = self.val_env, decoder_strategy = "greedy")
        else:
            pass

    def calculate_loss(self, policy_out: dict, **kwargs):
        """Calculate loss for REINFORCE algorithm.
        Args:
            policy_out: Output of the policy network
        """
        if self.method_name in ['lih','omni','udc','glop','l2s']:
            if 'loss' in policy_out:
                return policy_out['loss']
        reward = policy_out["reward"]  # shape: (batch, pomo)
        likelihood = policy_out["likelihood"]  # shape: (batch, pomo, problem)
        log_prob = likelihood.log().sum(dim=2)
        # shape = (batch, pomo)
        self.env.aug_flag = False
        bl_value, bl_loss = self.baseline.eval(batch=self.env.problems, reward=reward, env=self.env)
        self.env.aug_flag = self.env.aug_type is not None
        advantage = reward - bl_value

        reinforce_loss = -(log_prob * advantage).mean()

        task_loss = reinforce_loss + bl_loss
        return task_loss

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

        problem_name = self.env._get_name().replace("Env", "") # remove "Env" from the name
        data_generator = generate_data(problem_name=problem_name)
        if self.method_name=="glop_revision":
            return data_generator(self.train_episodes if self.env.aug_type is None else self.train_episodes // self.env.aug_factor,
                self.problem_size, self.batch_size, self.device, path=self.train_data_path,distribution=self.data_distribution)
        elif self.method_name == 'omni' and self.fine_tune.enable :
            return data_generator(
                self.fine_tune.fine_tune_episodes if self.fine_tune.augmentation_enable is False else self.fine_tune.fine_tune_episodes // self.env.aug_factor,
                self.problem_size, self.fine_tune.fine_tune_batch_size, device=self.device,
                path=self.test_data_path,fine_tune_offset=self.fine_tune.fine_tune_episodes)
        elif self.method_name == 'mvmoe' or self.method_name=="mtpomo" :
            return data_generator(self.train_episodes if self.env.aug_type is None else self.train_episodes//self.env.aug_factor,
                                            self.problem_size, self.batch_size, **self.train_data_params,device=self.device,train_problems=self.env.train_problems)
        else:
            return data_generator(self.train_episodes if self.env.aug_type is None else self.train_episodes//self.env.aug_factor,
                                            self.problem_size, self.batch_size, **self.train_data_params,device=self.device)

    def test_dataloader(self):
        """
        This function is used to load test dataloader
        """
        if self.method_name == 'glop':
            problem_name=self.problem.upper()
        else:
            problem_name = self.env._get_name().replace("Env", "") # remove "Env" from the name
        if problem_name == 'DUMMY':
            problem_name = self.problem.upper()
        data_generator = generate_data(problem_name=problem_name)
        if self.method_name=="mvmoe" or self.method_name=='mtpomo':
            return data_generator(self.test_episodes, None, self.test_batch_size, self.device, self.test_data_path,test_problem=self.env.test_problem,
                                  **self.test_data_params)
        else:
            return data_generator(self.test_episodes, None, self.test_batch_size, self.device, self.test_data_path, **self.varying_data_params, **self.test_data_params)

    def val_dataloader(self):
        """
        This function is used to load valid dataloader
        """
        if self.do_val:
            problem_name = self.env._get_name().replace("Env", "") # remove "Env" from the name
            data_generator = generate_data(problem_name=problem_name)
            return data_generator(self.valid_episode, None, self.valid_batch_size, self.device, self.val_data_path)
        else:
            return []

    def on_train_epoch_end(self):

        self.log('epoch_score',self.avgScore_one_epoch.avg,on_step=False,on_epoch=True,prog_bar=True,logger=True)
        self.log('epoch_loss',self.avgLoss_one_epoch.avg,on_step=False,on_epoch=True,prog_bar=True,logger=True)
        logger.info("Train ==> Epoch {:3d}/{:3d}: Completed {:3d}/{:3d}({:1.1f}%)  Score: {:.4f}, Loss: {:.4f}"
                    .format(self.current_epoch+1, self.trainer.max_epochs, self.trained_num_episodes, self.train_episodes,
                            100. * self.trained_num_episodes / self.train_episodes,
                            self.avgScore_one_epoch.avg, self.avgLoss_one_epoch.avg))
        # Reset the average meter
        self.avgScore_one_epoch.reset()
        self.avgLoss_one_epoch.reset()
        self.trained_num_episodes = 0
        elapsed_time_str, remain_time_str = self.time_estimator.get_est_string(self.current_epoch+1, self.trainer.max_epochs)
        logger.info("Train ==> Epoch {:3d}/{:3d}: Time Est.: Elapsed[{}], Remain[{}]".format(
            self.current_epoch+1, self.trainer.max_epochs, elapsed_time_str, remain_time_str))

        """Callback for end of training epoch: we evaluate the baseline"""
        if self.baseline_name in ['warmup', 'rollout']:
            logger.info("Train ==> Epoch {:3d}/{:3d}: Evaluate the baseline".format(self.current_epoch+1, self.trainer.max_epochs))

        _ = self.baseline.epoch_callback(
                self.policy,
                env=self.baseline_env,
                batch_size=self.bl_batch_size,
                device=get_lightning_device(self),
                epoch=self.current_epoch,
                dataset_size=self.bl_data_size,
                )
        # Need to call super() for the dataset to be reset
        super().on_train_epoch_end()

    def on_validation_epoch_end(self) -> None:
        if self.do_val:
            self.log('val_aug_score', self.aug_score.avg, on_step=False, on_epoch=True, prog_bar=True, logger=True)
            self.log('val_no_aug_score', self.score.avg, on_step=False, on_epoch=True, prog_bar=True, logger=True)
            logger.info("Eval Done ==> Epoch {:3d}/{:3d}: Problem[{}], Size[{}], Value[{:4f}]".format(
                self.current_epoch + 1, self.trainer.max_epochs,
                self.env.env_name, self.problem_size, self.score.avg))
            # Reset the average meter
            self.score.reset()
            self.aug_score.reset()
            self.test_num_episodes = 0

    def on_test_end(self) -> None:
        # TODO: display test result
        total_time = time.time() - self.start_time
        logger.info(f"Total Testing Time: {total_time:.4f} seconds, {total_time/60:.4f} minutes, {total_time/3600:.4f} hours")  # Log testing time
        logger.info(f"Avg Time per Episode: {total_time / self.test_num_episodes:.4f} seconds (Episode: {self.test_num_episodes})")  # Log average time per episode
        elapsed_time_str, remain_time_str = self.time_estimator.get_est_string(self.test_num_episodes, self.test_episodes)
        logger.info("Eval Done ==> Tested {:3d}/{:3d}({:1.2f}%)  Time Est.: Elapsed[{}], Remain[{}]".format(
            self.test_num_episodes, self.test_episodes,
            100. * self.test_num_episodes / self.test_episodes, elapsed_time_str, remain_time_str))

        logger.info("Eval Done ==> Problem[{}], Score Summary: Avg: {:.4f}, Avg Augmented: {:.4f}".format(self.env.env_name, self.score.avg, self.aug_score.avg))
        if self.gap.avg > 0:
            logger.info("Eval Done ==> Problem[{}], Gap Summary: Avg: {:.4f}%, Avg Augmented: {:.4f}%, Std: {:.4f}%, Std Augmented: {:.4f}%".format(
                self.env.env_name, self.gap.avg, self.aug_gap.avg, np.std(self.gap_list), np.std(self.aug_gap_list)))