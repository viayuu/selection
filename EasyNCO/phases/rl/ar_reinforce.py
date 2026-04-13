import copy
import json
import os
import numbers

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


def _extract_eval_summary_scores(test_out: dict):
    """稳定提取评测摘要分数，避免依赖 dict.values() 的插入顺序。"""
    if not isinstance(test_out, dict):
        raise TypeError(f"test_out must be a dict, got {type(test_out).__name__}")

    score = test_out.get("no_aug_score", None)
    if score is None:
        score = test_out.get("score", None)
    aug_score = test_out.get("aug_score", None)

    if score is None or aug_score is None:
        raise KeyError(
            f"test_out must contain [('no_aug_score' or 'score'), 'aug_score'], got keys={list(test_out.keys())}"
        )

    return score, aug_score


def _filter_loggable_metrics(metrics: dict):
    """只保留 Lightning 可安全记录的标量指标，过滤掉 batch 等大张量。"""
    if not isinstance(metrics, dict):
        raise TypeError(f"metrics must be a dict, got {type(metrics).__name__}")

    filtered = {}
    for key, value in metrics.items():
        if isinstance(value, numbers.Number):
            filtered[key] = value
        elif torch.is_tensor(value) and value.numel() == 1:
            filtered[key] = value
    return filtered


def _extract_per_instance_scores_from_reward(
    reward: torch.Tensor,
    aug_factor: int,
    batch_size: int,
    pomo_size: int,
    score_summary=None,
    aug_score_summary=None,
):
    """从最终 reward 张量中恢复每个实例自己的 no-aug / aug 分数。"""
    aug_factor = 1 if aug_factor in (None, 0) else int(aug_factor)
    pomo_size = 1 if pomo_size in (None, 0) else int(pomo_size)
    expected_numel = aug_factor * batch_size * pomo_size
    if reward.numel() != expected_numel:
        base = aug_factor * batch_size
        if base <= 0 or reward.numel() % base != 0:
            raise ValueError(
                f"reward size mismatch: expected aug_factor*batch_size*pomo_size={expected_numel}, "
                f"got numel={reward.numel()}"
            )
        pomo_size = reward.numel() // base
    aug_reward = reward.reshape(aug_factor, batch_size, pomo_size)
    max_pomo_reward, _ = aug_reward.max(dim=2)
    base_no_aug_score = max_pomo_reward[0, :].float().detach().cpu()
    base_aug_score = max_pomo_reward.max(dim=0).values.float().detach().cpu()

    if score_summary is not None and aug_score_summary is not None:
        score_summary = float(torch.as_tensor(score_summary).item())
        aug_score_summary = float(torch.as_tensor(aug_score_summary).item())
        neg_error = abs((-base_no_aug_score.mean()).item() - score_summary) + abs((-base_aug_score.mean()).item() - aug_score_summary)
        pos_error = abs(base_no_aug_score.mean().item() - score_summary) + abs(base_aug_score.mean().item() - aug_score_summary)
        if neg_error <= pos_error:
            return -base_no_aug_score, -base_aug_score
        return base_no_aug_score, base_aug_score

    if base_no_aug_score.mean().item() < 0 or base_aug_score.mean().item() < 0:
        return -base_no_aug_score, -base_aug_score
    return base_no_aug_score, base_aug_score


def _extract_per_instance_scores_from_lengths(
    current_length: torch.Tensor,
    aug_factor: int,
    batch_size: int,
):
    """从 LIH 这类初始化状态里的 current_length 恢复每个实例的 no-aug / aug 分数。"""
    aug_factor = 1 if aug_factor in (None, 0) else int(aug_factor)
    current_length = current_length.reshape(-1).float()
    expected_aug_size = aug_factor * batch_size

    if current_length.numel() == expected_aug_size:
        aug_length = current_length.reshape(aug_factor, batch_size)
        base_no_aug_score = aug_length[0, :].float().detach().cpu()
        base_aug_score = aug_length.min(dim=0).values.float().detach().cpu()
        return base_no_aug_score, base_aug_score

    if current_length.numel() == batch_size:
        base_no_aug_score = current_length.float().detach().cpu()
        base_aug_score = current_length.float().detach().cpu()
        return base_no_aug_score, base_aug_score

    raise ValueError(
        f"current_length size mismatch: expected {batch_size} or {expected_aug_size}, got {current_length.numel()}"
    )


def _extract_per_instance_scores_from_tsp_solutions(
    batch: torch.Tensor,
    solution: torch.Tensor,
):
    """从 UDC TSP 的 batch + solution 状态恢复每个实例的 no-aug / aug 分数。"""
    if solution.dim() == 2:
        solution = solution.unsqueeze(1)
    batch = batch.float()
    solution = solution.long()
    batch_size, aug_factor, problem_size = solution.size()
    gathering_index = solution.unsqueeze(-1).expand(batch_size, aug_factor, problem_size, batch.size(-1))
    ordered_seq = batch[:, None, :, :].expand(batch_size, aug_factor, problem_size, batch.size(-1)).gather(
        dim=2, index=gathering_index
    )
    rolled_seq = ordered_seq.roll(dims=2, shifts=-1)
    segment_lengths = ((ordered_seq - rolled_seq) ** 2).sum(3).sqrt().sum(2)
    return segment_lengths[:, 0].detach().cpu(), segment_lengths.min(dim=1).values.detach().cpu()


def _extract_per_instance_scores_from_policy_dict_list(initialization_out):
    """从 DIFUSCO/T2T 的 list[policy_dict] 初始化输出恢复逐实例分数。"""
    if not isinstance(initialization_out, list) or not initialization_out:
        raise ValueError("initialization_out must be a non-empty list")

    no_aug_scores = []
    aug_scores = []
    for policy_dict in initialization_out:
        if not isinstance(policy_dict, dict):
            raise TypeError(f"policy_dict must be a dict, got {type(policy_dict).__name__}")
        if "np_nodes" not in policy_dict or "tours" not in policy_dict:
            raise KeyError(f"policy_dict must contain ['np_nodes', 'tours'], got keys={list(policy_dict.keys())}")

        nodes = torch.as_tensor(policy_dict["np_nodes"], dtype=torch.float32)
        tours = torch.as_tensor(policy_dict["tours"], dtype=torch.long)
        if tours.ndim == 1:
            tours = tours.unsqueeze(0)

        ordered_nodes = nodes[tours]
        costs = ((ordered_nodes[:, 1:, :] - ordered_nodes[:, :-1, :]) ** 2).sum(dim=-1).sqrt().sum(dim=-1).float()
        no_aug_scores.append(costs[0])
        aug_scores.append(costs.min())

    return torch.stack(no_aug_scores).detach().cpu(), torch.stack(aug_scores).detach().cpu()


def _extract_per_instance_scores_from_cvrp_solutions(
    batch: torch.Tensor,
    solution: torch.Tensor,
    solution_flag: torch.Tensor,
):
    """从 UDC CVRP 的 batch + solution + solution_flag 状态恢复逐实例分数。"""
    batch = batch.float()
    solution = solution.long()
    solution_flag = solution_flag.long()
    order_node = solution.clone()
    order_flag = solution_flag.clone().float()

    order_flag[order_flag <= 0.5] = order_node[order_flag <= 0.5].float()
    order_flag[solution_flag > 0.5] = 0.0
    merged_solution = torch.stack((order_node.float(), order_flag), dim=3).view(
        order_node.size(0), order_node.size(1), -1
    ).long()

    gathering_index = merged_solution.unsqueeze(3).expand(
        merged_solution.size(0), merged_solution.size(1), merged_solution.size(2), 2
    )
    ordered_seq = batch[:, None, :, :2].expand(
        merged_solution.size(0), merged_solution.size(1), batch.size(1), 2
    ).gather(dim=2, index=gathering_index)
    rolled_seq = ordered_seq.roll(dims=2, shifts=-1)
    segment_lengths = ((ordered_seq - rolled_seq) ** 2).sum(3).sqrt().sum(2)
    return segment_lengths[:, 0].detach().cpu(), segment_lengths.min(dim=1).values.detach().cpu()


def _build_eval_instance_records(start_index: int, no_aug_score: torch.Tensor, aug_score: torch.Tensor, optimal=None, names=None):
    """把单 batch 的逐实例结果整理成可落盘的记录列表。"""
    no_aug_score = no_aug_score.detach().cpu().reshape(-1)
    aug_score = aug_score.detach().cpu().reshape(-1)
    batch_size = no_aug_score.numel()

    if optimal is not None:
        optimal = torch.as_tensor(optimal).detach().cpu().reshape(-1)
        if optimal.numel() != batch_size:
            raise ValueError(f"optimal size mismatch: expected {batch_size}, got {optimal.numel()}")
    if names is None:
        names = [None] * batch_size
    elif isinstance(names, (str, bytes)):
        names = [names] * batch_size
    else:
        names = list(names)
        if len(names) != batch_size:
            names = (names + [None] * batch_size)[:batch_size]

    records = []
    for local_idx in range(batch_size):
        record = {
            "global_index": int(start_index + local_idx),
            "name": names[local_idx],
            "no_aug_score": float(no_aug_score[local_idx].item()),
            "aug_score": float(aug_score[local_idx].item()),
        }
        if optimal is not None:
            optimal_value = float(optimal[local_idx].item())
            record["optimal"] = optimal_value
            if abs(optimal_value) > 1e-12:
                record["gap"] = float((record["no_aug_score"] - optimal_value) / optimal_value * 100.0)
                record["aug_gap"] = float((record["aug_score"] - optimal_value) / optimal_value * 100.0)
            else:
                record["gap"] = None
                record["aug_gap"] = None
        records.append(record)
    return records

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
                 export_instance_results: bool = True,
                 export_instance_results_filename: str = "instance_results.jsonl",
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
        self.export_instance_results = export_instance_results
        self.export_instance_results_filename = export_instance_results_filename

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
        self.instance_result_records = []
        self.instance_result_export_active = False
        self.instance_result_skipped_batches = 0
        self.instance_result_export_path = None

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
        self.instance_result_records = []
        self.instance_result_skipped_batches = 0
        self.instance_result_export_active = self.export_instance_results and self.method_name != 'psl'
        self.instance_result_export_path = os.path.join(os.getcwd(), self.export_instance_results_filename)

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
            state_td = None
            initialization_out = None
            export_aug_factor = env.aug_factor
            export_pomo_size = env.pomo_size
        else:
            export_aug_factor = env.aug_factor
            export_pomo_size = env.pomo_size
            state_td, initialization_out = self.initialization.run(env, batch, decoder_strategy, "eval", **self.initialization_params)
            if self.customized:
                env.pomo_size = 1 # reset pomo_size to 1 for iteration
                env.aug_factor = 1 # reset aug_factor to 1 for iteration
                env.aug_flag = False # reset aug_flag to False for iteration
            self.iteration_params["problems"] = batch
            self.iteration_params["decoder_strategy"] = decoder_strategy
            test_out = self.iteration.run(state_td, env, initialization_out, "eval", **self.iteration_params)

        score, aug_score = _extract_eval_summary_scores(test_out)
        if isinstance(score, float):
            score = tensor(score, device=self.device)
        if isinstance(aug_score, float):
            aug_score = tensor(aug_score, device=self.device)

        per_instance_no_aug_score = None
        per_instance_aug_score = None
        if (
            self.instance_result_export_active
            and self.method_name != 'psl'
        ):
            try:
                if state_td is not None and "reward" in state_td.keys():
                    per_instance_no_aug_score, per_instance_aug_score = _extract_per_instance_scores_from_reward(
                        reward=state_td["reward"],
                        aug_factor=export_aug_factor,
                        batch_size=batch_size,
                        pomo_size=export_pomo_size,
                        score_summary=score if torch.is_tensor(score) and score.ndim == 0 else None,
                        aug_score_summary=aug_score if torch.is_tensor(aug_score) and aug_score.ndim == 0 else None,
                    )
                elif state_td is not None and "current_length" in state_td.keys():
                    per_instance_no_aug_score, per_instance_aug_score = _extract_per_instance_scores_from_lengths(
                        current_length=state_td["current_length"],
                        aug_factor=export_aug_factor,
                        batch_size=batch_size,
                    )
                elif (
                    state_td is not None
                    and
                    getattr(env, "env_name", None) == "tsp"
                    and "batch" in state_td.keys()
                    and "solution" in state_td.keys()
                ):
                    per_instance_no_aug_score, per_instance_aug_score = _extract_per_instance_scores_from_tsp_solutions(
                        batch=state_td["batch"],
                        solution=state_td["solution"],
                    )
                elif (
                    state_td is not None
                    and getattr(env, "env_name", None) == "cvrp"
                    and "batch" in state_td.keys()
                    and "solution" in state_td.keys()
                    and "solution_flag" in state_td.keys()
                ):
                    per_instance_no_aug_score, per_instance_aug_score = _extract_per_instance_scores_from_cvrp_solutions(
                        batch=state_td["batch"],
                        solution=state_td["solution"],
                        solution_flag=state_td["solution_flag"],
                    )
                elif (
                    isinstance(initialization_out, list)
                    and getattr(env, "env_name", None) == "tsp"
                ):
                    per_instance_no_aug_score, per_instance_aug_score = _extract_per_instance_scores_from_policy_dict_list(
                        initialization_out=initialization_out,
                    )
                else:
                    raise KeyError("unsupported per-instance export state")
            except Exception as exc:
                self.instance_result_skipped_batches += 1
                logger.warning(f"Eval export skipped for batch {batch_idx}: failed to extract per-instance scores ({exc})")
        elif self.instance_result_export_active and self.method_name != 'psl':
            self.instance_result_skipped_batches += 1
            logger.warning(f"Eval export skipped for batch {batch_idx}: per-instance state is unavailable for method [{self.method_name}]")

        if torch.is_tensor(score) and score.ndim == 0:
            self.score.update(score.item(),batch_size)
        elif isinstance(score, list):
            for i in range(self.num_target):
                self.score_psl[i].update(score[i], batch_size)            
        
        if torch.is_tensor(aug_score) and aug_score.ndim == 0:
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
                if per_instance_no_aug_score is not None and per_instance_aug_score is not None:
                    optimal_cpu = torch.as_tensor(optimal).detach().cpu().reshape(-1)
                    gap = 100 * ((per_instance_no_aug_score - optimal_cpu) / optimal_cpu)
                    aug_gap = 100 * ((per_instance_aug_score - optimal_cpu) / optimal_cpu)
                else:
                    gap = 100 * ((score - optimal) / optimal)
                    aug_gap = 100 * ((aug_score - optimal) / optimal)
                self.gap_list.extend(torch.as_tensor(gap).detach().cpu().reshape(-1).tolist())
                gap_mean = torch.as_tensor(gap).float().mean().item()
                self.aug_gap_list.extend(torch.as_tensor(aug_gap).detach().cpu().reshape(-1).tolist())
                aug_gap_mean = torch.as_tensor(aug_gap).float().mean().item()
                self.gap.update(gap_mean, batch_size)
                self.aug_gap.update(aug_gap_mean, batch_size)
                test_out.update({"gap": gap_mean, "aug_gap": aug_gap_mean})

            if self.instance_result_export_active and per_instance_no_aug_score is not None and per_instance_aug_score is not None:
                if per_instance_no_aug_score.numel() != batch_size or per_instance_aug_score.numel() != batch_size:
                    self.instance_result_skipped_batches += 1
                    logger.warning(
                        "Eval export skipped for batch %s: extracted per-instance score count mismatch "
                        "(expected %s, got no_aug=%s, aug=%s). "
                        "For DIFUSCO/T2T init-only export, please use batch_size=1.",
                        batch_idx,
                        batch_size,
                        per_instance_no_aug_score.numel(),
                        per_instance_aug_score.numel(),
                    )
                else:
                    start_index = self.test_num_episodes - batch_size
                    optimal_for_export = optimal if optimal is not None else None
                    self.instance_result_records.extend(
                        _build_eval_instance_records(
                            start_index=start_index,
                            no_aug_score=per_instance_no_aug_score,
                            aug_score=per_instance_aug_score,
                            optimal=optimal_for_export,
                            names=name,
                        )
                    )
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
            self.log_dict(_filter_loggable_metrics(test_out))



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
        if self.instance_result_export_active:
            if self.instance_result_records:
                with open(self.instance_result_export_path, "w", encoding="utf-8") as f:
                    for record in self.instance_result_records:
                        f.write(json.dumps(record, ensure_ascii=False) + "\n")
                logger.info(f"Eval Done ==> Per-instance results saved to {self.instance_result_export_path} ({len(self.instance_result_records)} instances)")
            else:
                logger.warning("Eval Done ==> Per-instance export is enabled but no records were collected.")
            if self.instance_result_skipped_batches > 0:
                logger.warning(f"Eval Done ==> Per-instance export skipped {self.instance_result_skipped_batches} batch(es).")
        self.instance_result_export_active = False
