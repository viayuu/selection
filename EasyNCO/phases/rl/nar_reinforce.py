import torch
from lightning import LightningModule
from torch.optim import Adam as Optimizer
from torch.optim.lr_scheduler import MultiStepLR as Scheduler
from torch import nn
from torchrl.envs import EnvBase

from EasyNCO.data import *
from EasyNCO.neural_solvers.main_loop_old.DeepACO.load_policy import load_policy
from EasyNCO.utils.utils import *

logger = getLogger(__name__)

class NARREINFORCENARLightning(LightningModule):
    def __init__(self,
                 policy: nn.Module,
                 problem_type: str,
                 problem_size: int,
                 batch_size: int,
                 episodes: int,
                 policy_param: dict,
                 optimizer_params: dict = None,
                 iteration = 1,
                 val_episode: int = 1000,
                 val_batch_size: int = 1000,
                 every_n_steps_output: int = 100,
                 env: EnvBase=None,
                    **kwargs):
        super().__init__()
        self.policy = policy  # GNN
        self.env = env # can be None
        self.problem_type = problem_type
        self.problem_size = problem_size
        self.batch_size = batch_size
        self.train_episodes = episodes
        self.optimizer_params = optimizer_params
        self.policy_param = policy_param

        self.avgLoss_one_epoch = AverageMeter()
        self.sum_results = AverageMeter()
        self.every_n_steps_output = every_n_steps_output

        self.time_estimator = TimeEstimator()
        self.iteration = iteration

        self.trained_num_episodes = 0

        # path of datasets
        self.train_data_path = None  # RL training, do not need pre-generated data
        self.test_data_path = kwargs.get('test_data_path', None)
        self.val_data_path = kwargs.get('val_data_path', None)

        # validation
        self.valid_episode = val_episode
        self.valid_batch_size = val_batch_size
        self.test_num_episodes = 0

        # test
        self.test_episodes = episodes
        self.test_batch_size = batch_size



    def on_fit_start(self) -> None:
        '''
        This function is called at the beginning of the training process.
        on_fit_start
        ├── on_train_start
        │   ├── on_train_epoch_start
        │   └── on_train_batch_start
        ├── on_validation_start
        └── on_test_start
        '''
        if self.env is not None:
            self.env.device = get_lightning_device(self)
        self.time_estimator.reset(self.current_epoch+1)

    def on_test_start(self) -> None:
        '''
        Note that we skip the fit function and directly execute the test function in inference process.
        '''
        if self.env is not None:
            self.env.device = get_lightning_device(self)
        self.time_estimator.reset(self.current_epoch + 1)
        self.start_time = time.time()

    def training_step(self, batch, batch_idx):
        sum_loss = 0.0
        for data in batch:
            # batch: (batch_size, problem_size, features)
            # data： (problem_size, features)
            policy = load_policy(
                model = self.policy,
                data = data,
                policy_param = self.policy_param,
                env_name = self.problem_type,
            )
            reinforce_loss = policy.train_instance()
            sum_loss += reinforce_loss

        avg_loss = sum_loss / len(batch)
        self.trained_num_episodes += batch.size(0)
        self.log("train_loss", avg_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.avgLoss_one_epoch.update(sum_loss.item(), self.batch_size)

        if batch_idx+1 <= 5 or (batch_idx+1) % self.every_n_steps_output == 0 or batch_idx+1 == self.trainer.num_training_batches:
            logger.info("Epoch {:3d}/{:3d}: Train {:3d}/{:3d}({:1.1f}%)  Loss: {:.4f}".format(
                self.current_epoch+1, self.trainer.max_epochs, batch_idx+1, self.trainer.num_training_batches,
                batch_idx+1 / self.trainer.num_training_batches * 100, self.avgLoss_one_epoch.avg))

        return avg_loss


    def validation_step(self, batch, batch_idx):
        stats = []
        for data in batch:
            policy = load_policy(
                model = self.policy,
                data = data,
                policy_param = self.policy_param,
                env_name = self.problem_type,
            )
            stats.append(policy.train_instance(infer = True))

        avg_stats = [i.item() for i in np.stack(stats).mean(0)]
        # log the validation results: average cost, minimum (best) cost, result.
        val_results = {'loss_baseline': avg_stats[0],
                       'loss_best': avg_stats[1],
                       'result': avg_stats[2]}

        logger.info("Epoch {:3d}/{:3d}:  loss_baseline: {:.4f}, loss_best: {:.4f}, result: {:.4f}, "
                    .format(self.current_epoch + 1,
                            self.trainer.max_epochs,
                            avg_stats[0],
                            avg_stats[1],
                            avg_stats[2]))
        self.test_num_episodes += batch.size(0)
        return val_results


    def train_dataloader(self):
        '''
        This function is used to create the dataloader based on DataLoader.
        '''

        problem_name = self.problem_type.upper()
        data_generator = generate_data(problem_name=problem_name)
        return data_generator(self.train_episodes, self.problem_size, self.batch_size, self.device, self.train_data_path)
        # [batch_size, problem_size, 2] * steps


    def val_dataloader(self):
        '''
        This function is used to create the dataloader based on DataLoader.
        '''
        problem_name = self.problem_type.upper()
        data_generator = generate_data(problem_name=problem_name)
        return data_generator(self.valid_episode, self.problem_size, self.valid_batch_size, self.device, self.val_data_path)

    def test_dataloader(self):
        """
        This function is used to load test dataloader
        """
        problem_name = self.problem_type.upper()
        data_generator = generate_data(problem_name=problem_name)
        return data_generator(self.test_episodes, self.problem_size, self.test_batch_size, self.device, self.test_data_path)


    def configure_optimizers(self):
        '''
        This function is used to create the optimizer and scheduler.
        Note that there are some methods don't use the scheduler.
        '''
        optimizer = Optimizer(self.policy.parameters(), **self.optimizer_params["optimizer"])
        if "scheduler" in self.optimizer_params:
            scheduler = Scheduler(optimizer, **self.optimizer_params["scheduler"])
            return optimizer, scheduler
        else:
            return optimizer

    def on_train_epoch_end(self):
        '''
        There is a bug in the current platform version that does not allow to record the reward and loss of the epoch.
        '''
        self.log('epoch_loss',self.avgLoss_one_epoch.avg,on_step=False,on_epoch=True,prog_bar=True,logger=True)

        logger.info("Epoch {:3d}/{:3d}:  Train {:3d}/{:3d}({:1.1f}%)  Loss: {:.4f}".format(
            self.current_epoch+1, self.trainer.max_epochs, self.trained_num_episodes, self.train_episodes,
                            100. * self.trained_num_episodes / self.train_episodes, self.avgLoss_one_epoch.avg))
        # Reset the average meter
        self.avgLoss_one_epoch.reset()
        elapsed_time_str, remain_time_str = self.time_estimator.get_est_string(self.current_epoch+1, self.trainer.max_epochs)
        logger.info("Epoch {:3d}/{:3d}: Time Est.: Elapsed[{}], Remain[{}]".format(
            self.current_epoch+1, self.trainer.max_epochs, elapsed_time_str, remain_time_str))

        # Need to call super() for the dataset to be reset
        super().on_train_epoch_end()

    def on_validation_epoch_end(self) -> None:
        # Todo: add the validation results to the logger.
        self.test_num_episodes = 0

    def test_step(self, batch, batch_idx):
        sum_results = torch.zeros(size=(self.iteration,))
        # Only process a single instance.
        for data in batch:
            # Load the policy.
            policy = load_policy(
                model = self.policy,
                data = data,
                policy_param = self.policy_param,
                env_name = self.problem_type,
            )
            results = torch.zeros(size=(self.iteration,))
            for i in range(self.iteration):
                # Generate the heatmap. If some methods generate the heatmap only once, you can set the iteration to 1.
                heatmap = policy.heatmap()
                # Search based on the heatmap.
                results[i] = policy.search(heatmap)

            sum_results += results

        avg_results = sum_results / len(batch)
        self.sum_results.update(avg_results, self.batch_size)
        for i in range(self.iteration):
            logger.info("T={:d} results: {:.15f}".format(i+1, avg_results[i]))
        return avg_results

    def on_test_end(self) -> None:
        total_time = time.time() - self.start_time
        logger.info(f"Total Testing Time: {total_time:.4f} seconds")  # Log testing time
        for idx, value in enumerate(self.sum_results.avg):
            logger.info(f"The result of T={idx+1}: {value.item():.15f}")
        # Reset the average meter
        self.sum_results.reset()

