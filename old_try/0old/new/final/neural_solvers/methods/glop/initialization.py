import numpy as np
from torchrl.envs import EnvBase
from typing import Literal
import torch.nn as nn
from typing import Any, Tuple
from tensordict import TensorDict
import torch
import random_insertion as insertion
import os
from EasyNCO.neural_solvers.pipeline import Initialization
from EasyNCO.neural_solvers.methods.glop.sub_tsp_utils import sub_tsp_len
from EasyNCO.neural_solvers.methods.glop.cvrp_heatmap import infer_heatmap_cvrp, Sampler_cvrp, cvrp_solution_heatmap, \
    cvrp_trans_tsp
from EasyNCO.neural_solvers.methods.glop.pctsp_heatmap import pctsp_solution_heatmap



class GLOPInitialization(Initialization):
    """
    A class for POMO initialization that extends the ARInitialization class.
    It is used to create initial solutions for training in the POMO framework.
    """
    def __init__(self, policy: nn.Module,**kwargs):
        self.problem=kwargs.get("problem")
        self.global_params=policy.global_params
        self.policy = policy


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
            pass
            # self.upper_model.train()
            # env.env_batch_size=batch.shape[0]
            # if self.problem=="cvrp":
            #     batch_routes=[]
            #     batch_probs=[]
            #
            #     for i in range(batch.shape[0]):
            #         current_inst=batch[i]
            #         routes1, probs = cvrp_solution_heatmap(current_inst, main_net=self.upper_model, n_subset=self.global_params["n_subset"],
            #                                                Train_flag=True)
            #
            #         batch_routes.append(routes1.unsqueeze(0))
            #         batch_probs.append(probs)
            #
            #     td = {
            #         "data": batch,
            #         "probs": batch_probs,
            #         "selected_node_list": batch_routes,
            #
            #     }
            #     out = {}
            #
            #
            # elif self.problem=="pctsp":
            #     batch_routes=[]
            #     batch_probs=[]
            #     batch_penaltys=[]
            #     for i in range(batch.shape[0]):
            #         current_inst=batch[i]
            #         pi, penalty, probs = pctsp_solution_heatmap(current_inst.unsqueeze(0),self.upper_model, n_subset=self.global_params["n_subset"],
            #                                                Train_flag=True)
            #         batch_probs.append(probs)
            #         batch_penaltys.append(penalty)
            #         batch_routes.append(pi)
            #     td = {
            #         "data": batch,
            #         "probs": batch_probs,
            #         "selected_node_list": batch_routes,
            #         "penalty":batch_penaltys
            #     }
            #     out = {}


        else:
            if self.problem == "atsp":
                batch_size=batch.shape[0]
                assert batch_size == 1, "Set eval_batch_size to 1 for ATSP!"
                batch0 = batch.squeeze().cpu().numpy()
                problem_size = batch0.shape[-1]
                # order=np.random.permutation(problem_size)
                order = torch.randperm(problem_size, device="cpu").numpy()
                tour, cost = insertion.atsp_random_insertion(batch0, order)
                pi = torch.tensor(tour.astype(np.int64))
                cost = torch.tensor(cost).unsqueeze(0)
                td = TensorDict({
                    "data": batch ,
                    "selected_node_list":pi.unsqueeze(0),
                    "reward":cost
                }, batch_size=[1])
                out = {
                    "no_aug_score": cost.item(),
                    "aug_score": cost.item(),
                }
            elif self.problem== "tsp":
                val_size, problem_size, problem_dim = batch.shape
                orders = np.array([np.random.permutation(problem_size) for _ in range(self.global_params.width)])
                coors = batch.cpu().numpy()
                pi_all = [insertion.tsp_random_insertion(instance, orders[order_id])[0] for instance in
                          coors for order_id in range(len(orders))]  # instance: (p_size, 2)
                pi_all = torch.tensor(np.array(pi_all).astype(np.int64)).reshape(val_size, self.global_params.width, -1)
                data0 = batch[:, None, :, :].repeat(1, self.global_params.width, 1, 1)
                new_data0 = data0.gather(2, pi_all.unsqueeze(-1).repeat(1, 1, 1, 2))  # 得到多个解决方案的序列化后的坐标
                cost_data = new_data0.view(-1, problem_size, problem_dim)
                cost = sub_tsp_len(cost_data)
                cost=cost.reshape(val_size,self.global_params.width)
                # pi_all = pi_all.view(-1, problem_size)
                mean_cost=cost.mean()
                min_cost=cost.min(dim=1)[0]
                aug_cost=min_cost.mean()
                td = TensorDict({
                    "data": batch ,
                    "selected_node_list":pi_all,
                    "reward":cost
                }, batch_size=[val_size])
                out = {
                    "no_aug_score": mean_cost,
                    "aug_score": aug_cost,
                }
            elif self.problem == "cvrp":
                batch_size=batch.shape[0]
                problem_size=batch.shape[1]-1
                assert batch_size == 1, "Set eval batch size to 1 for CVRP!"
                assert self.global_params["width"]==1,"Set width to 1 for CVRP!"
                assert self.global_params["n_subset"]==1,"Set n_subset to 1 for CVRP when testing!"
                model=self.policy.upper_model
                if problem_size<=1000:
                    k_sparse=self.policy.k_sparse_cvrp_leq1000
                else:
                    k_sparse = self.policy.k_sparse_cvrp_gt1000
                pi, _ = cvrp_solution_heatmap(batch,model, self.global_params.n_subset,k_sparse=k_sparse,Train_flag=False)
                pi_, _ = cvrp_trans_tsp(pi)
                batch0 = batch.squeeze()
                sub_problem = batch0[pi_][:, :, :2].cpu().numpy()
                sub_num, sub_problem_size, problem_dim = sub_problem.shape
                order = np.arange(sub_problem_size, dtype=np.uint32)
                pi_batch = [insertion.tsp_random_insertion(instance, order)[0] for instance in sub_problem]
                pi_batch = torch.tensor(np.array(pi_batch).astype(np.int64))
                init_pi = pi_.gather(1, pi_batch).reshape(-1)
                cost_data = batch0.gather(0, init_pi.unsqueeze(-1).repeat(1, 3))[:, 0:2]
                ori_cost = sub_tsp_len(cost_data[None, :, :])
                td = TensorDict({
                    "data": batch ,
                    "selected_node_list":init_pi.unsqueeze(0),
                    "reward":ori_cost,
                    "sub_problem_size": torch.tensor(sub_problem_size).unsqueeze(0),
                }, batch_size=[1])
                out = {
                    "no_aug_score": ori_cost,
                    "aug_score": ori_cost,
                }
            elif self.problem == "pctsp":
                assert self.global_params["width"]==1,"Set width to 1 for PCTSP!"
                batch_size=batch.shape[0]
                problem_size=batch.shape[0]-1
                model=self.policy.upper_model
                if problem_size<=500:
                    k_sparse=self.policy.k_sparse_pctsp_leq500
                elif problem_size<=1000:
                    k_sparse=self.policy.k_sparse_pctsp_leq1000
                else:
                    k_sparse = self.policy.k_sparse_pctsp_gt1000
                pi, penalty, _ = pctsp_solution_heatmap(batch,model, self.global_params.n_subset,k_sparse=k_sparse)
                coors = batch[:, :, 0:2]
                xy_data = torch.repeat_interleave(coors, self.global_params.n_subset, 0)  # n_val*n_subset, p+1, 2
                xy_data = xy_data.gather(1, pi.unsqueeze(-1).repeat(1, 1, 2)).reshape(
                    coors.shape[0] * self.global_params.n_subset, -1, 2)  # (n_val*n_subset, max_seq_len, 2)

                val_size, problem_size, problem_dim = xy_data.shape
                orders = np.array([np.random.permutation(problem_size) for _ in range(self.global_params.width)])
                orders_repeated = np.tile(orders, (val_size, 1))
                instances = xy_data.cpu().numpy()
                pi_all = insertion.tsp_random_insertion_parallel(instances, orders_repeated, threads=0)
                pi_all = torch.tensor(np.array(pi_all).astype(np.int64)).reshape(val_size, self.global_params.width, -1)
                data0 = xy_data[:, None, :, :].repeat(1, self.global_params.width, 1, 1)
                new_data0 = data0.gather(2, pi_all.unsqueeze(-1).repeat(1, 1, 1, 2))  # 得到多个解决方案的序列化后的坐标
                cost_data = new_data0.view(-1, problem_size, problem_dim)
                cost = sub_tsp_len(cost_data) + penalty.squeeze()
                pi_all = pi_all.view(-1, problem_size)
                really_pi = pi.gather(1, pi_all)
                solution=really_pi.reshape(batch_size,self.global_params.n_subset,-1)
                cost=cost.reshape(batch_size,self.global_params.n_subset)
                penalty=penalty.reshape(batch_size,self.global_params.n_subset)
                td = TensorDict({
                    "data": batch ,
                    "selected_node_list":solution,
                    "reward":cost,
                    "penalty": penalty,
                }, batch_size=[batch_size])

                mean_cost=cost.mean()
                aug_cost=cost.min(dim=1)[0].mean()
                out = {
                    "no_aug_score": mean_cost,
                    "aug_score": aug_cost,
                }


        return td,out


