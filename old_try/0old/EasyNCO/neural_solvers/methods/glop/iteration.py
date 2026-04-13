from abc import ABC, abstractmethod
from typing import Any, Tuple, Literal

import numpy as np
import torch
import torch.nn as nn
from tensordict import TensorDict
from torchrl.envs import EnvBase
from torch import Tensor
from EasyNCO.utils.utils import *

from EasyNCO.neural_solvers.pipeline.iteration import Iteration
import copy
import os

from EasyNCO.neural_solvers.methods.glop.sub_atsp_utils import atsp_decompose_and_solve
from EasyNCO.neural_solvers.methods.glop.sub_tsp_utils import tsp_decompose_and_solve, sub_tsp_len
import random_insertion as insertion

from EasyNCO.neural_solvers.methods.glop.cvrp_heatmap import cvrp_trans_tsp


def _tsplib_tour_len(coords: torch.Tensor, tour: torch.Tensor, edge_weight_type: str) -> torch.Tensor:
    ordered = coords.gather(1, tour.unsqueeze(-1).expand(-1, -1, coords.size(-1)))
    rolled = ordered.roll(dims=1, shifts=-1)
    seg = ((ordered - rolled) ** 2).sum(-1).sqrt()
    if edge_weight_type == "CEIL_2D":
        seg = torch.ceil(seg)
    elif edge_weight_type == "EUC_2D":
        seg = (seg + 0.5).floor()
    return seg.sum(1)


class GLOPIteration(Iteration):
    """
    A no-operation iteration class that does nothing.
    """
    def __init__(self, policy: nn.Module=None,**kwargs):
        self.global_params=policy.global_params
        self.policy=policy
        self.problem=kwargs.get("problem")
        if self.problem in ["atsp"]:
            self.env=kwargs.get('lower_env_atsp')
        else:
            self.env=kwargs.get('lower_env')
        self.lower_model_decoder_strategy=kwargs.get("lower_model_decoder_strategy")

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

        problem = td.get("data", None)
        if problem is None:
            problem = kwargs.get("problems", None)
        if problem is None and isinstance(initialization_out, dict):
            problem = initialization_out.get("batch", None)
        if problem is None:
            problem = getattr(env, "problems", None)
        if problem is None:
            raise KeyError(
                f'GLOPIteration requires problem data, but none was found. '
                f'td.keys={list(td.keys())}, kwargs.keys={list(kwargs.keys())}, '
                f'initialization_out.keys={list(initialization_out.keys()) if isinstance(initialization_out, dict) else type(initialization_out)}'
            )
        device = problem.device
        val_size, problem_size, problem_dim = problem.shape
        outer_lib_data = getattr(env, "lib_data", None)
        if outer_lib_data is not None:
            self.env.lib_data = outer_lib_data
        elif getattr(self.env, "lib_data", None) is None:
            self.env.lib_data = {
                "data": torch.empty((1, 0, 2), device=device),
                "edge_weight_type": "NONE",
            }
        self.env.device=device
        if phase=="train":
            pass
            # if self.problem == "cvrp":
            #     obj_ss=[]
            #     loss_list=[]
            #     for i in range(problem.shape[0]):
            #         current_inst = problem[i]
            #         routes1=td["selected_node_list"][i].squeeze(0)
            #         pi_, perIns_sub_num = cvrp_trans_tsp(routes1)
            #         with torch.no_grad():
            #             # #RI的初始解
            #             sub_problem = current_inst[pi_][:, :, :2]
            #             sub_num, sub_problem_size, problem_dim = sub_problem.shape
            #             order = np.arange(sub_problem_size, dtype=np.uint32)
            #             pi_batch = [insertion.tsp_random_insertion(instance.cpu().numpy(), order)[0] for instance in
            #                         sub_problem]
            #             pi_batch = torch.tensor(np.array(pi_batch).astype(np.int64))
            #             self.lower_model[f"sub{self.global_params['revision_lens'][0]}"].to(device)
            #             self.env.problem_size = self.global_params.revision_lens[0]
            #             solution = tsp_decompose_and_solve(sub_problem, pi_batch, self.lower_model[f"sub{self.global_params['revision_lens'][0]}"],
            #                                                self.env, self.global_params["revision_lens"][0], self.global_params["iter"][0],
            #                                                decoder_strategy="greedy")
            #             # 计算cost
            #             cost_data = sub_problem.gather(1, solution[:, :, None].repeat(1, 1, 2))
            #             rolled_seq = cost_data.roll(dims=1, shifts=-1)
            #             dist = ((cost_data - rolled_seq) ** 2).sum(2).sqrt().sum(1)
            #             # obj = sum(dist)
            #             start = 0
            #             ret = []
            #             for n in perIns_sub_num:
            #                 ret.append(dist[start: start + n].sum())
            #                 start += n
            #             obj = torch.stack(ret)
            #
            #         #计算loss
            #         baseline = obj.mean()
            #         log_probs=td["probs"][i]
            #         log_probs = log_probs.to(device)
            #         reinforce_loss = torch.sum((obj - baseline) * log_probs.sum(dim=1)) / 2
            #         loss_list.append(reinforce_loss)
            #         obj_ss.append(obj.unsqueeze(0))
            #             # batch_objs.append(objs)
            #     loss = sum(loss_list) / problem.shape[0]
            #     ave_obj=torch.cat(obj_ss,dim=0)
            #     initialization_out = {'loss':loss, 'reward':-ave_obj}
            # elif self.problem == "pctsp":
            #     obj_ss=[]
            #     loss_list=[]
            #     for i in range(problem.shape[0]):
            #         current_inst = problem[[i],:,:]
            #         pi=td['solution'][i]
            #         coors = current_inst[:, :, 0:2]
            #         xy_data = torch.repeat_interleave(coors, self.global_params["n_subset"], 0)  # n_val*n_subset, p+1, 2
            #         xy_data = xy_data.gather(1, pi.unsqueeze(-1).repeat(1, 1, 2)).reshape(coors.shape[0] *  self.global_params["n_subset"], -1, 2)
            #
            #         val_size, problem_size, problem_dim = xy_data.shape
            #         order = np.arange(problem_size, dtype=np.uint32)
            #         pi_all = [insertion.tsp_random_insertion(instance.cpu().numpy(), order)[0] for instance in xy_data]
            #         pi_all = torch.tensor(np.array(pi_all).astype(np.int64), device=device)  # width, p_size
            #
            #         self.lower_model[f"sub{self.global_params['revision_lens'][0]}"].to(device)
            #         self.env.problem_size = self.global_params.revision_lens[0]
            #         solution = tsp_decompose_and_solve(xy_data, pi_all, self.lower_model[f"sub{self.global_params['revision_lens'][0]}"],
            #                                            self.env, self.global_params["revision_lens"][0], self.global_params["iter"][0],
            #                                            decoder_strategy="greedy")
            #         # 计算cost
            #         cost_data = xy_data.gather(1, solution[:, :, None].repeat(1, 1, 2))
            #         rolled_seq = cost_data.roll(dims=1, shifts=-1)
            #         dist = ((cost_data - rolled_seq) ** 2).sum(2).sqrt().sum(1)
            #         penalty=td['penalty'][i]
            #         obj = dist + penalty.reshape(-1)
            #         baseline = obj.mean()
            #         log_probs=td["probs"][i]
            #         log_probs = log_probs.to(device)
            #         reinforce_loss = torch.sum((obj - baseline) * log_probs.sum(dim=1)) / 2
            #         loss_list.append(reinforce_loss)
            #         obj_ss.append(obj.unsqueeze(0))
            #     loss = sum(loss_list) / problem.shape[0]
            #     ave_obj=torch.cat(obj_ss,dim=0)
            #     initialization_out = {'loss':loss, 'reward':-ave_obj}

        else:
            solution = td.get("selected_node_list", None)
            if solution is None:
                next_td = td.get("next", None)
                if next_td is not None:
                    solution = next_td.get("selected_node_list", None)
            if solution is None:
                raise KeyError(
                    f'GLOPIteration requires "selected_node_list" in td (or td["next"]), '
                    f'but it was not found. td.keys={list(td.keys())}'
                )
            if self.problem == "atsp":

                problem = problem.squeeze()

                for i in range(len(self.global_params.revision_lens)):
                    self.env.problem_size=self.global_params.revision_lens[i]
                    model=self.policy.lower_model[f"sub_{self.global_params.revision_lens[i]}"]
                    solution = atsp_decompose_and_solve(problem, solution, model,
                                                        self.env,self.global_params.revision_lens[i], self.global_params.iter[i],
                                                        decoder_strategy=self.lower_model_decoder_strategy)

                fin_cost = problem[solution, torch.roll(solution, -1, -1)].sum()
                initialization_out = {'score':fin_cost, 'aug_score':fin_cost}

            elif self.problem == "tsp":
                problem = problem[:, None, :, :].repeat(1, self.global_params.width, 1, 1).reshape(-1, problem_size,
                                                                                            problem_dim)  # vaL_size*width，p_size,2
                solution = solution.reshape(-1, problem_size)
                for i in range(len(self.global_params.revision_lens)):
                    self.env.problem_size=self.global_params.revision_lens[i]
                    model=self.policy.lower_model[f"sub_{self.global_params.revision_lens[i]}"]
                    solution = tsp_decompose_and_solve(problem, solution, model,
                                                       self.env, self.global_params.revision_lens[i], self.global_params.iter[i],
                                                       decoder_strategy=self.lower_model_decoder_strategy)

                outer_lib = getattr(self.env, "lib_data", None)
                if (
                    outer_lib is not None
                    and isinstance(outer_lib, dict)
                    and "data" in outer_lib
                    and "edge_weight_type" in outer_lib
                    and outer_lib["data"].dim() == 3
                    and outer_lib["data"].size(1) == problem_size
                ):
                    coords = (
                        outer_lib["data"][:, None, :, :]
                        .repeat(1, self.global_params.width, 1, 1)
                        .reshape(-1, problem_size, outer_lib["data"].size(-1))
                        .to(solution.device)
                    )
                    cost = _tsplib_tour_len(coords, solution, outer_lib["edge_weight_type"])
                else:
                    cost_data = problem.gather(1, solution.unsqueeze(-1).expand_as(problem))
                    cost = sub_tsp_len(cost_data)
                solution = solution.reshape(-1,self.global_params.width, problem_size)
                cost=cost.reshape(-1, self.global_params.width)
                score=cost.mean()
                aug_score=cost.min(dim=1)[0].mean()
                initialization_out = {'score':score, 'aug_score':aug_score}

            elif self.problem== "pctsp":

                penaty = td["penalty"].reshape(val_size*self.global_params.n_subset)
                inin_solution=td["selected_node_list"].reshape(val_size*self.global_params.n_subset,-1)
                coors = problem[:, :, 0:2]

                coors = torch.repeat_interleave(coors, self.global_params.n_subset, 0)

                problem = coors.gather(1, inin_solution[:, :, None].repeat(1, 1, 2))

                solution = torch.arange(0, problem.shape[1])[None, :].repeat(inin_solution.shape[0], 1)

                for i in range(len(self.global_params.revision_lens)):
                    self.env.problem_size=self.global_params.revision_lens[i]
                    model=self.policy.lower_model[f"sub_{self.global_params.revision_lens[i]}"]

                    solution = tsp_decompose_and_solve(problem, solution, model,
                                                       self.env,self.global_params.revision_lens[i], self.global_params.iter[i],
                                                       decoder_strategy=self.lower_model_decoder_strategy)

                cost_data = problem.gather(1, solution.unsqueeze(-1).repeat(1, 1, 2))

                cost = sub_tsp_len(cost_data) + penaty

                real_solution = inin_solution.gather(1, solution)
                score=cost.mean()
                aug_score=cost.reshape(val_size,-1).min(1)[0].mean()
                initialization_out = {'score':score, 'aug_score':aug_score}

            elif self.problem == "cvrp":

                sub_problem_size = td["sub_problem_size"][0].item()
                inin_solution=td["selected_node_list"].squeeze(0)
                sub_solution = inin_solution.reshape(-1, sub_problem_size)

                solution = torch.arange(0, sub_problem_size)[None, :].expand_as(sub_solution)

                sub_problem = problem.squeeze().gather(0, inin_solution[:, None].repeat(1, 3))[:, 0:2]

                sub_problem = sub_problem.reshape(-1, sub_problem_size, 2)

                for i in range(len(self.global_params.revision_lens)):
                    self.env.problem_size=self.global_params.revision_lens[i]
                    model=self.policy.lower_model[f"sub_{self.global_params.revision_lens[i]}"]
                    solution = tsp_decompose_and_solve(sub_problem, solution, model,
                                                       self.env,self.global_params.revision_lens[i], self.global_params.iter[i],
                                                       decoder_strategy=self.lower_model_decoder_strategy)
                solution = sub_solution.gather(1, solution)
                solution0=solution.clone()
                is_zero = solution == 0
                solution0[is_zero]-=1
                is_not_zero = solution != 0
                is_zero_next_not_zero = is_zero & is_not_zero.roll(dims=-1, shifts=-1)
                solution0[is_zero_next_not_zero] += 1

                double_solution0=solution0.repeat(1,2)
                rows, cols = torch.where(is_zero_next_not_zero)
                idx=torch.arange(0,solution0.shape[1]).unsqueeze(0).repeat(solution0.shape[0],1)
                start=(idx*is_zero_next_not_zero).max(dim=1)[1]
                end=start+solution0.shape[1]
                double_idx=torch.arange(0,double_solution0.shape[1]).unsqueeze(0).repeat(double_solution0.shape[0],1)
                idx1=(double_idx>=start.unsqueeze(1))*(double_idx<=end.unsqueeze(1))
                new_soultion=double_solution0[idx1]
                new_soultion=new_soultion[new_soultion>=0]


                cost_data = problem.squeeze().gather(0, new_soultion[:, None].repeat(1, 3))[:, 0:2]

                final_cost = sub_tsp_len(cost_data[None, :, :])
                initialization_out = {'score':final_cost.item(), 'aug_score':final_cost.item()}


        return initialization_out
