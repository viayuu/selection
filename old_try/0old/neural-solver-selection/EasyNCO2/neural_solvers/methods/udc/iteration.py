from EasyNCO.neural_solvers.pipeline import Iteration
from typing import Any, Tuple, Literal
import torch
import torch.nn as nn
from tensordict import TensorDict
from torchrl.envs import EnvBase
from torch import Tensor
from EasyNCO.utils.utils import *
from EasyNCO.data.data_utils import glop_coordinate_transformation


logger = getLogger(__name__)

class UDCIteration(Iteration):
    def run(
        self,
        td: TensorDict,
        env: EnvBase,
        initialization_out: dict,
        phase: Literal["train", "eval"],
        max_steps: int = 2,
        **kwargs,
    ) -> dict:
        if phase == "train":
            if env.env_name == "tsp":
                return self._run_train_tsp(
                    td, env, initialization_out, max_steps, **kwargs
                )
            if env.env_name == "cvrp":
                return self._run_train_cvrp(
                    td, env, initialization_out, max_steps, **kwargs
                )
        else:
            if env.env_name == "tsp":
                return self._run_eval_tsp(
                    td, env, initialization_out, max_steps, **kwargs
                )
            if env.env_name == "cvrp":
                return self._run_eval_cvrp(
                    td, env, initialization_out, max_steps, **kwargs
                )

    def _run_train_tsp(
        self,
        td: TensorDict,
        env: EnvBase,
        initialization_out: dict,
        max_steps: int = 2,
        **kwargs,
    ) -> Tuple[TensorDict, Tensor]:
        batch_size = 1
        for i in range(max_steps):
            roll = env.sub_problem_size // 2
            solution = initialization_out["solution"].roll(dims=1, shifts=roll)
            n_tsps_per_route = solution.view(solution.size(0), -1, env.sub_problem_size)
            tsp_insts = (
                initialization_out["raw_problems"][:, None, :]
                .repeat(solution.size(0), n_tsps_per_route.size(1), 1, 1)
                .gather(-2, n_tsps_per_route.unsqueeze(-1).expand(-1, -1, -1, 2))
            )
            tsp_insts_now = tsp_insts.view(-1, tsp_insts.size(-2), tsp_insts.size(-1))
            solution_now = torch.arange(tsp_insts_now.size(-2))[None, :].expand(
                (tsp_insts_now.size(0), -1)
            )[:, None, :]
            new_batch_size = tsp_insts_now.size(0)
            tsp_insts_now_norm = glop_coordinate_transformation(tsp_insts_now)
            env.load_problems(tsp_insts_now_norm, new_batch_size)
            env.problem_size = env.sub_problem_size
            reward_now = env._get_travel_distance(
                tsp_insts_now, solution_now, open_flag=True
            )
            _, policy_out = self.play_episode(policy=self.policy.model_t, env=env)
            loss = self.calculate_loss(policy_out, env)
            self.policy.model_t.zero_grad()
            loss.backward()
            initialization_out["optimizer_t"].step()
            reward = env._get_travel_distance(
                tsp_insts_now, env.selected_node_list, open_flag=True
            )
            # Loss
            ###############################################
            tag = (
                reward.view(batch_size, env.sample_size, -1, env.pomo_size)
                .min(-1)[1][..., None, None]
                .expand(-1, -1, -1, -1, env.problem_size)
            )
            tag_solution = (
                env.selected_node_list.view(
                    batch_size,
                    env.sample_size,
                    -1,
                    env.pomo_size,
                    env.problem_size,
                )
                .gather(-2, tag)
                .squeeze()
            )
            reversed_tag_solution = torch.flip(tag_solution, dims=[2])
            tag_solution[tag.squeeze() >= env.pomo_size / 2] = reversed_tag_solution[
                tag.squeeze() >= env.pomo_size / 2
            ]
            r = (
                (reward.min(1)[0] > reward_now.squeeze())
                .view(env.sample_size, -1, 1)
                .expand((-1, -1, tsp_insts_now.size(-2)))
            )
            tag_solution[r] = solution_now.view(
                env.sample_size, -1, tsp_insts_now.size(-2)
            )[r]
            merge_solution = n_tsps_per_route.gather(-1, tag_solution).view(
                solution.size(0), -1
            )
            solution = merge_solution.clone()

        env.env_batch_size = batch_size
        env.problem_size = initialization_out["raw_problems"].size(1)
        env.pomo_size = env.sample_size
        merge_reward = -1 * env._get_travel_distance(
            initialization_out["raw_problems"], solution[None, :]
        )
        advantage2 = merge_reward - merge_reward.float().mean(dim=1, keepdims=True)
        # shape: (batch, pomo)
        loss_partition = (
            -advantage2 * initialization_out["logp"]
        ).mean()  # + (-advantage2 * logp_a).mean()

        # Step & Return
        ###############################################
        Iteration_out = {
            "reward": merge_reward,
            "loss": loss_partition,
        }
        return Iteration_out

    def _run_train_cvrp(
        self,
        td: TensorDict,
        env: EnvBase,
        initialization_out: dict,
        max_steps: int = 2,
        **kwargs,
    ) -> Tuple[TensorDict, Tensor]:
        batch_size = 1
        solution = initialization_out["solution"]
        solution_flag = initialization_out["solution_flag"]
        for i in range(2):
            solution, solution_flag = self.route_ranking(
                initialization_out["raw_problems"][:, :, :2], solution, solution_flag
            )
            roll = env.sub_problem_size // 2
            solution = solution.roll(dims=1, shifts=roll)
            solution_flag = solution_flag.roll(dims=1, shifts=roll)
            n_tsps_per_route = solution.view(solution.size(0), -1, env.sub_problem_size)
            n_tsps_per_route_flag = solution_flag.view(
                solution.size(0), -1, env.sub_problem_size
            )
            demand_per_route = (
                initialization_out["raw_problems"][:, :, 2][:, None, :]
                .repeat(solution.size(0), n_tsps_per_route.size(1), 1)
                .gather(-1, n_tsps_per_route)
            )
            capacity_now = torch.ones(
                (demand_per_route.size(0), demand_per_route.size(1)),
                device=demand_per_route.device,
            )
            tag = (
                (
                    n_tsps_per_route_flag
                    * (
                        n_tsps_per_route.size(-1)
                        - torch.arange(n_tsps_per_route.size(-1))
                    )
                )
                .max(-1)[1]
                .unsqueeze(-1)
            )
            capacity_now -= (
                torch.cumsum(demand_per_route, dim=-1).gather(-1, tag).squeeze()
            )
            capacity_end = torch.ones(
                (demand_per_route.size(0), demand_per_route.size(1)),
                device=demand_per_route.device,
            )
            tag = (
                (n_tsps_per_route_flag * (torch.arange(n_tsps_per_route.size(-1))))
                .max(-1)[1]
                .unsqueeze(-1)
            )
            capacity_end -= (
                torch.cumsum(demand_per_route, dim=-1)[:, :, -1]
                - torch.cumsum(demand_per_route, dim=-1).gather(-1, tag).squeeze()
            )
            tsp_insts = (
                initialization_out["raw_problems"][:, None, :]
                .repeat(solution.size(0), n_tsps_per_route.size(1), 1, 1)
                .gather(-2, n_tsps_per_route.unsqueeze(-1).expand(-1, -1, -1, 3))
            )
            customer_insts_now = tsp_insts.view(
                -1, tsp_insts.size(-2), tsp_insts.size(-1)
            )
            # following: normalization in conquering stage
            tsp_insts_now = torch.cat(
                (
                    initialization_out["raw_problems"][:, 0, :]
                    .unsqueeze(0)
                    .repeat(customer_insts_now.size(0), 1, 1),
                    customer_insts_now,
                ),
                dim=1,
            )
            solution_now = torch.arange(1, tsp_insts_now.size(-2))[None, :].expand(
                (tsp_insts_now.size(0), -1)
            )[:, None, :]
            reward_now = env.cal_open_length(
                tsp_insts_now[:, :, [0, 1]],
                solution_now,
                n_tsps_per_route_flag.view(-1, tsp_insts_now.size(-2) - 1)[:, None, :],
            )
            capacity_pair2 = capacity_end.clone().view(-1, 1)
            capacity_pair1 = capacity_now.clone().roll(dims=1, shifts=-1).view(-1, 1)
            first_demand = tsp_insts[:, :, 0, -1].roll(dims=1, shifts=1).view(-1, 1)
            capacity_pair = torch.cat((capacity_pair1, capacity_pair2), dim=-1)
            capacity_pair[:, 0][(capacity_pair[:, 1] == 1.0)] = 0.0
            tag = ((capacity_pair[:, 1] > 0.5) & (capacity_pair[:, 0] > 0.5)).clone()
            capacity_pair[:, 1][tag] = 0.5
            capacity_pair[:, 0][tag] = 0.5
            capacity_pair[:, 0][
                (capacity_pair[:, 0] > 0.5) & (capacity_pair[:, 1] <= 0.5)
            ] = (
                1
                - capacity_pair[:, 1][
                    (capacity_pair[:, 0] > 0.5) & (capacity_pair[:, 1] <= 0.5)
                ]
            )
            capacity_pair[:, 1][
                (capacity_pair[:, 1] > 0.5) & (capacity_pair[:, 0] <= 0.5)
            ] = (
                1
                - capacity_pair[:, 0][
                    (capacity_pair[:, 1] > 0.5) & (capacity_pair[:, 0] <= 0.5)
                ]
            )
            # capacity_pair[:, 1][(capacity_pair[:, 1] > 0.5) & (capacity_pair[:, 0] <= 0.5) & (1 - capacity_pair[:, 0] < first_demand.squeeze())] = 1.
            # capacity_pair[:, 0][(capacity_pair[:, 1] > 0.5) & (capacity_pair[:, 0] <= 0.5) & (1 - capacity_pair[:, 0] < first_demand.squeeze())] = 0.
            capacity_head = (
                capacity_pair[:, 1]
                .clone()
                .view(env.sample_size, -1)
                .roll(dims=1, shifts=1)
                .view(-1, 1)
            )
            capacity_tail = capacity_pair[:, 0].clone().view(-1, 1)
            new_batch_size = tsp_insts_now.size(0)
            tsp_insts_now_norm = glop_coordinate_transformation(tsp_insts_now)
            env.flag = n_tsps_per_route_flag[:, :, -1].clone().view(-1)
            env.load_problems(tsp_insts_now_norm, new_batch_size)
            env.load = torch.ones(size=(new_batch_size, env.pomo_size)) * capacity_head
            env.left = torch.ones(size=(new_batch_size, env.pomo_size)) * capacity_tail
            self.policy.model_t.left = env.left
            self.policy.model_t.flag_return = (
                n_tsps_per_route_flag[:, :, -1].clone().view(-1)
            )
            _, policy_out = self.play_episode(policy=self.policy.model_t, env=env)
            new_solution = torch.cat(
                (
                    env.udc_solution_list.unsqueeze(-1),
                    env.udc_solution_flag.unsqueeze(-1),
                ),
                dim=-1,
            )
            loss = self.calculate_loss(policy_out)
            self.policy.model_t.zero_grad()
            loss.backward()
            initialization_out["optimizer_t"].step()
            reward = env.cal_open_length(
                tsp_insts_now[:, :, [0, 1]],
                new_solution[:, :, :, 0],
                new_solution[:, :, :, 1],
            )
            tag = (
                reward.view(batch_size, env.sample_size, -1, env.pomo_size)
                .min(-1)[1][..., None, None]
                .expand(-1, -1, -1, -1, env.sub_problem_size)
            )
            tag_solution = (
                env.udc_solution_list.view(
                    batch_size,
                    env.sample_size,
                    -1,
                    env.pomo_size,
                    env.sub_problem_size,
                )
                .gather(-2, tag)
                .squeeze()
            )
            tag_solution_flag = (
                env.udc_solution_flag.view(
                    batch_size,
                    env.sample_size,
                    -1,
                    env.pomo_size,
                    env.sub_problem_size,
                )
                .gather(-2, tag)
                .squeeze()
            )
            r = (
                (reward.min(1)[0] > reward_now.squeeze())
                .view(env.sample_size, -1, 1)
                .expand((-1, -1, tsp_insts_now.size(-2) - 1))
            )
            tag_solution[r] = solution_now.view(
                env.sample_size, -1, tsp_insts_now.size(-2) - 1
            )[r]
            tag_solution_flag[r] = n_tsps_per_route_flag[r]
            solution = n_tsps_per_route.gather(-1, tag_solution - 1).view(
                solution.size(0), -1
            )
            solution_flag = tag_solution_flag.view(solution.size(0), -1)
        env.flag = None

        merge_reward = -1 * env.cal_open_length(
            initialization_out["raw_problems"][:, :, [0, 1]],
            solution,
            solution_flag,
            total_flag=True,
        )
        advantage2 = merge_reward - merge_reward.float().mean(dim=1, keepdims=True)
        # shape: (batch, pomo)
        loss_partition = (
            -advantage2 * initialization_out["logp"]
        ).mean()  # + (-advantage2 * logp_a).mean()
        # cal_leagal_cvrp(
        #     initialization_out["raw_problems"][:, :, -1], solution, solution_flag
        # )  # use to check legality

        # Step & Return
        ###############################################
        Iteration_out = {
            "reward": merge_reward,
            "loss": loss_partition,
        }
        return Iteration_out

    def _run_eval_tsp(
        self,
        td: TensorDict,
        env: EnvBase,
        initialization_out: dict,
        max_steps: int = 251,
        **kwargs,
    ) -> Tuple[TensorDict, Any]:
        k = 0
        solution = td["solution"]
        while k < max_steps:
            k += 1
            episode = 0
            score_AM = AverageMeter()
            aug_score_AM = AverageMeter()
            while episode < initialization_out["test_num_episode"]:
                remaining = initialization_out["test_num_episode"] - episode
                batch_size = min(kwargs["udc_batch_size"], remaining)
                solution, score, aug_score = self._tsp_eval_one_batch(
                    solution,
                    batch_size,
                    episode,
                    k,
                    aug_factor=env.aug_factor,
                    batch=td["batch"],
                    env=env,
                    policy=self.policy,
                    test_strategy=initialization_out["strategy"],
                )
                score_AM.update(score, batch_size)
                aug_score_AM.update(aug_score, batch_size)
                episode += batch_size
                logger.info(
                    "iter {:2d}, score:{:.3f}, aug_score:{:.3f}".format(
                        k,
                        score_AM.avg,
                        aug_score_AM.avg,
                    )
                )
        iteration_out = {
            "score": score_AM.avg,
            "aug_score": aug_score_AM.avg,
        }
        return iteration_out

    def _run_eval_cvrp(
        self,
        td: TensorDict,
        env: EnvBase,
        initialization_out: dict,
        max_steps: int = 50,
        **kwargs,
    ) -> Tuple[TensorDict, Any]:
        solution = td["solution"]
        solution_flag = td["solution_flag"]
        k = 0
        while k < max_steps:

            solution, solution_flag = self.route_ranking2(
                td["batch"], solution, solution_flag
            )
            k += 1
            episode = 0
            score_AM = AverageMeter()
            aug_score_AM = AverageMeter()
            while episode < initialization_out["test_num_episode"]:
                remaining = initialization_out["test_num_episode"] - episode
                batch_size = min(kwargs["udc_batch_size"], remaining)
                solution, solution_flag, score, aug_score = self._cvrp_test_one_batch(
                    solution,
                    solution_flag,
                    batch_size,
                    episode,
                    k,
                    aug_factor=env.aug_factor,
                    batch=td["batch"],
                    env=env,
                    policy=self.policy,
                    test_strategy=initialization_out["strategy"],
                )
                score_AM.update(score, batch_size)
                aug_score_AM.update(aug_score, batch_size)
                episode += batch_size
                logger.info(
                    "iter {:2d}, score:{:.3f}, aug_score:{:.3f}".format(
                        k,
                        score_AM.avg,
                        aug_score_AM.avg,
                    )
                )
        iteration_out = {
            "score": score_AM.avg,
            "aug_score": aug_score_AM.avg,
        }
        return iteration_out

    def _tsp_eval_one_batch(
        self,
        solution_gnn,
        batch_size,
        episode,
        k,
        aug_factor,
        batch,
        env,
        policy,
        test_strategy,
    ):
        solution = solution_gnn.clone()[episode : episode + batch_size]
        now_problem = batch.clone().float()[episode : episode + batch_size]
        # now_optimal_tour = self.tours.clone()[episode:episode + batch_size]
        if k == 1 or k == 2:
            roll = env.sub_problem_size // 2
        else:
            roll = random.randint(1, env.sub_problem_size)
        solution = solution.roll(dims=-1, shifts=roll)
        solving_length = (
            (solution.size(-1)) // env.sub_problem_size
        ) * env.sub_problem_size
        solution_cut = solution[:, :, :solving_length]
        n_tsps_per_route = solution_cut.clone().view(
            solution.size(0), -1, env.sub_problem_size
        )
        tsp_insts = (
            now_problem[:, None, :, :]
            .repeat(1, n_tsps_per_route.size(1), 1, 1)
            .gather(-2, n_tsps_per_route.unsqueeze(-1).expand(-1, -1, -1, 2))
        )
        tsp_insts_now = tsp_insts.view(-1, tsp_insts.size(-2), tsp_insts.size(-1))
        solution_now = torch.arange(tsp_insts_now.size(-2))[None, :].expand(
            (tsp_insts_now.size(0), -1)
        )[:, None, :]
        problem_size = env.problem_size
        env.problem_size = env.sub_problem_size
        reward_now = env._get_travel_distance(
            tsp_insts_now,
            solution_now,
            solution_now.size(0),
            solution_now.size(1),
            open_flag=True,
        )
        new_batch_size = tsp_insts_now.size(0)
        tsp_insts_now_norm = glop_coordinate_transformation(tsp_insts_now)
        env.load_problems(tsp_insts_now_norm, new_batch_size)
        self.play_episode(
            policy=policy.model_t, env=env, decoder_strategy=test_strategy
        )
        reward = env._get_travel_distance(
            tsp_insts_now, env.selected_node_list, open_flag=True
        )

        # Loss
        ###############################################
        tag = (
            reward.view(batch_size, aug_factor, -1, env.pomo_size)
            .min(-1)[1][..., None, None]
            .expand(-1, -1, -1, -1, env.sub_problem_size)
        )
        tag_solution = (
            env.selected_node_list.view(
                batch_size, aug_factor, -1, env.pomo_size, env.sub_problem_size
            )
            .gather(-2, tag)
            .squeeze(3)
        )
        reversed_tag_solution = torch.flip(tag_solution.clone(), dims=[-1])
        tag_solution[tag.squeeze(3) >= env.pomo_size / 2] = reversed_tag_solution[
            tag.squeeze(3) >= env.pomo_size / 2
        ]
        r = (
            (reward.min(1)[0] > reward_now.squeeze())
            .view(batch_size, aug_factor, -1, 1)
            .expand((-1, -1, -1, tsp_insts_now.size(-2)))
        )
        tag_solution[r] = solution_now.view(
            batch_size, aug_factor, -1, tsp_insts_now.size(-2)
        )[r]
        merge_solution = (
            n_tsps_per_route.view(batch_size, aug_factor, -1, tsp_insts_now.size(-2))
            .gather(-1, tag_solution)
            .view(solution.size(0), solution_cut.size(1), -1)
        )
        solution = torch.cat(
            (merge_solution.clone(), solution[:, :, solving_length:]), dim=-1
        )
        env.problem_size = problem_size
        merge_reward_0 = env._get_travel_distance(
            now_problem, solution, solution.size(0), solution.size(1)
        )
        solution_out = solution_gnn.clone()
        solution_out[episode : episode + batch_size] = solution
        # check if the selected node list is valid
        #cal_leagal_tsp(solution_out.size(2), solution_out)

        return (
            solution_out,
            merge_reward_0[:, 0].mean(0).item(),
            merge_reward_0.min(1)[0].mean(0).item(),
        )  # merge_reward_1.min(1)[0].mean(0).item()

    def _cvrp_test_one_batch(
        self,
        solution_gnn,
        solution_gnn_flag,
        batch_size,
        episode,
        k,
        aug_factor,
        batch,
        env,
        policy,
        test_strategy,
    ):
        solution = solution_gnn.clone()[episode : episode + batch_size]
        solution_flag = solution_gnn_flag.clone()[episode : episode + batch_size]
        now_problem = batch[episode : episode + batch_size]
        now_demand = batch[:, :, 2][episode : episode + batch_size]
        if k == 1 or k == 2:
            roll = env.sub_problem_size // 2
        else:
            roll = random.randint(1, env.sub_problem_size)
        solving_length = (
            (solution.size(-1)) // env.sub_problem_size
        ) * env.sub_problem_size
        padding_aim = (
            (solution.size(-1) - 1) // env.sub_problem_size + 1
        ) * env.sub_problem_size
        solution = solution.clone().roll(dims=-1, shifts=roll)
        solution_flag = solution_flag.clone().roll(dims=-1, shifts=roll)
        solution_cut = solution[:, :, :solving_length].clone()
        solution_cut_flag = solution_flag[:, :, :solving_length].clone()
        n_tsps_per_route = solution_cut.view(
            solution.size(0), solution.size(1), -1, env.sub_problem_size
        )
        n_tsps_per_route_flag = solution_cut_flag.view(
            solution.size(0), solution.size(1), -1, env.sub_problem_size
        )
        demand_per_route = (
            now_demand[:, None, None, :]
            .repeat(1, n_tsps_per_route.size(1), n_tsps_per_route.size(2), 1)
            .gather(-1, n_tsps_per_route)
        )
        n_tsps_per_route = n_tsps_per_route.view(
            -1, n_tsps_per_route.size(-2), env.sub_problem_size
        )
        n_tsps_per_route_flag = n_tsps_per_route_flag.view(
            -1, n_tsps_per_route.size(-2), env.sub_problem_size
        )
        demand_per_route = demand_per_route.view(
            -1, n_tsps_per_route.size(-2), env.sub_problem_size
        )

        real_demand = (
            now_demand.clone()[:, None, :]
            .repeat(1, solution.size(1), 1)
            .gather(-1, solution)
        )
        double_segment_sum = torch.cumsum(
            torch.cat((real_demand, real_demand), dim=-1).clone(), dim=-1
        )
        double_solution_flag = torch.cat((solution_flag, solution_flag), dim=-1).clone()
        solution_start = torch.cummax(
            double_solution_flag.roll(dims=-1, shifts=1)
            * torch.arange(solution.size(-1) * 2)[None, :],
            dim=-1,
        )[0]
        before = double_segment_sum.roll(dims=-1, shifts=1) - double_segment_sum.gather(
            -1, (solution_start - 1).clamp_min_(0)
        )
        before = before[:, :, solution.size(-1) :]
        solution_end = (
            2 * solution.size(-1)
            - 1
            - torch.flip(
                torch.cummax(
                    torch.flip(double_solution_flag, dims=[-1])
                    * torch.arange(solution.size(-1) * 2)[None, :],
                    dim=-1,
                )[0],
                dims=[-1],
            )
        )
        double_segment_sum[:, :, -1] = 0
        end = double_segment_sum.gather(-1, solution_end) - double_segment_sum.roll(
            dims=-1, shifts=1
        )
        end = end[:, :, : solution.size(-1)]

        if padding_aim > solution.size(-1):
            before_padding = before.roll(dims=-1, shifts=-1).clone()
            before_padding = torch.cat(
                (
                    before_padding,
                    before_padding[:, :, -1]
                    .unsqueeze(-1)
                    .repeat(1, 1, padding_aim - solution.size(-1)),
                ),
                dim=-1,
            )
            end_padding = torch.cat(
                (
                    end.clone(),
                    end[:, :, -1]
                    .unsqueeze(-1)
                    .repeat(1, 1, padding_aim - solution.size(-1)),
                ),
                dim=-1,
            )
        else:
            before_padding = before.roll(dims=-1, shifts=-1).clone()
            end_padding = end.clone()
        capacity_end = (
            1
            - before_padding.reshape(
                solution.size(0) * solution.size(1), -1, env.sub_problem_size
            )[:, :, -1]
        )
        capacity_now = (
            1
            - end_padding.reshape(
                solution.size(0) * solution.size(1), -1, env.sub_problem_size
            )[:, :, 0]
        )

        n_tsps_per_route = n_tsps_per_route.view(
            solution.size(0), solution.size(1), -1, env.sub_problem_size
        ).view(solution.size(0), -1, env.sub_problem_size)
        n_tsps_per_route_flag = n_tsps_per_route_flag.view(
            solution.size(0), solution.size(1), -1, env.sub_problem_size
        ).view(solution.size(0), -1, env.sub_problem_size)
        tsp_insts = (
            now_problem[:, None, :]
            .repeat(1, n_tsps_per_route.size(1), 1, 1)
            .gather(-2, n_tsps_per_route.unsqueeze(-1).expand(-1, -1, -1, 3))
        )
        add_depot_insts_now = torch.cat(
            (
                now_problem[:, 0, :][:, None, None, :].repeat(
                    1, tsp_insts.size(1), 1, 1
                ),
                tsp_insts,
            ),
            dim=2,
        )
        tsp_insts_now = add_depot_insts_now.view(
            -1, add_depot_insts_now.size(-2), add_depot_insts_now.size(-1)
        )
        solution_now = torch.arange(1, tsp_insts_now.size(-2))[None, :].expand(
            (tsp_insts_now.size(0), -1)
        )[:, None, :]
        reward_now = env.cal_open_length(
            tsp_insts_now[:, :, [0, 1]],
            solution_now,
            n_tsps_per_route_flag.view(-1, tsp_insts_now.size(-2) - 1)[:, None, :],
        )
        capacity_pair2 = capacity_end.clone().view(-1, 1)
        capacity_pair1 = capacity_now.clone().roll(dims=1, shifts=-1).view(-1, 1)
        capacity_pair = torch.cat((capacity_pair1, capacity_pair2), dim=-1)
        capacity_pair[:, 0][(capacity_pair[:, 1] == 1.0)] = 0.0
        tag = ((capacity_pair[:, 1] > 0.5) & (capacity_pair[:, 0] > 0.5)).clone()
        capacity_pair[:, 1][tag] = 0.5
        capacity_pair[:, 0][tag] = 0.5
        capacity_pair[:, 0][
            (capacity_pair[:, 0] > 0.5) & (capacity_pair[:, 1] <= 0.5)
        ] = (
            1
            - capacity_pair[:, 1][
                (capacity_pair[:, 0] > 0.5) & (capacity_pair[:, 1] <= 0.5)
            ]
        )
        capacity_pair[:, 1][
            (capacity_pair[:, 1] > 0.5) & (capacity_pair[:, 0] <= 0.5)
        ] = (
            1
            - capacity_pair[:, 0][
                (capacity_pair[:, 1] > 0.5) & (capacity_pair[:, 0] <= 0.5)
            ]
        )
        if padding_aim > solution.size(-1):
            capacity_head = (
                capacity_pair[:, 1]
                .clone()
                .view(batch_size * aug_factor, -1)
                .roll(dims=1, shifts=1)[:, :-1]
                .reshape(-1, 1)
            )
            capacity_tail = (
                capacity_pair[:, 0]
                .clone()
                .view(batch_size * aug_factor, -1)[:, :-1]
                .reshape(-1, 1)
            )
        else:
            capacity_head = (
                capacity_pair[:, 1]
                .clone()
                .view(batch_size * aug_factor, -1)
                .roll(dims=1, shifts=1)
                .view(-1, 1)
            )
            capacity_tail = capacity_pair[:, 0].clone().view(-1, 1)
        new_batch_size = tsp_insts_now.size(0)
        tsp_insts_now_norm = glop_coordinate_transformation(tsp_insts_now)
        env.flag = n_tsps_per_route_flag[:, :, -1].clone().view(-1)
        env.load_problems(tsp_insts_now_norm, new_batch_size)
        env.load = torch.ones(size=(new_batch_size, env.pomo_size)) * capacity_head
        env.left = torch.ones(size=(new_batch_size, env.pomo_size)) * capacity_tail
        policy.model_t.left = env.left
        policy.model_t.flag_return = n_tsps_per_route_flag[:, :, -1].clone().view(-1)
        self.play_episode(policy=policy.model_t, env=env, decoder_strategy=test_strategy)
        new_solution = torch.cat(
            (env.udc_solution_list.unsqueeze(-1), env.udc_solution_flag.unsqueeze(-1)),
            dim=-1,
        )
        reward = env.cal_open_length(
            tsp_insts_now[:, :, [0, 1]],
            new_solution[:, :, :, 0],
            new_solution[:, :, :, 1],
        )
        tag = (
            reward.view(batch_size, aug_factor, -1, env.pomo_size)
            .min(-1)[1][..., None, None]
            .expand(-1, -1, -1, -1, env.sub_problem_size)
        )
        tag_solution = (
            env.udc_solution_list.view(
                batch_size, aug_factor, -1, env.pomo_size, env.sub_problem_size
            )
            .gather(-2, tag)
            .squeeze(3)
        )
        tag_solution_flag = (
            env.udc_solution_flag.view(
                batch_size, aug_factor, -1, env.pomo_size, env.sub_problem_size
            )
            .gather(-2, tag)
            .squeeze(3)
        )
        r = (
            (reward.min(1)[0] > reward_now.squeeze())
            .view(solution.size(0), aug_factor, -1, 1)
            .expand((-1, -1, -1, tsp_insts_now.size(-2) - 1))
        )
        tag_solution[r] = solution_now.view(
            batch_size, aug_factor, -1, tsp_insts_now.size(-2) - 1
        )[r]
        tag_solution_flag[r] = n_tsps_per_route_flag.view(
            batch_size, aug_factor, -1, tsp_insts_now.size(-2) - 1
        )[r]
        merge_solution = (
            n_tsps_per_route.view(
                batch_size, aug_factor, -1, tsp_insts_now.size(-2) - 1
            )
            .gather(-1, tag_solution - 1)
            .view(solution.size(0), solution.size(1), -1)
        )
        merge_solution_flag = tag_solution_flag.view(
            solution.size(0), solution.size(1), -1
        )
        solution = torch.cat(
            (merge_solution.clone(), solution[:, :, solving_length:]), dim=-1
        )
        solution_flag = torch.cat(
            (merge_solution_flag.clone(), solution_flag[:, :, solving_length:]), dim=-1
        )
        #cal_leagal_cvrp(now_problem[:, :, -1], solution, solution_flag)
        solution_out = solution_gnn.clone()
        solution_out_flag = solution_gnn_flag.clone()
        solution_out[episode : episode + batch_size] = solution
        solution_out_flag[episode : episode + batch_size] = solution_flag
        merge_reward_0 = env.cal_open_length(
            now_problem[:, :, [0, 1]], solution, solution_flag, last_flag=True
        )
        env.flag = None

        return (
            solution_out,
            solution_out_flag,
            merge_reward_0[:, 0].mean(0).item(),
            merge_reward_0.min(1)[0].mean(0).item(),
        )  # merge_reward_1.min(1)[0].mean(0).item()

    def play_episode(
        self,
        policy: nn.Module,
        env,
        decoder_strategy: str = "sampling",
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

        policy.set_decoder_strategy(decoder_strategy)
        policy.pre_forward(reset_td)

        likelihood = torch.zeros(size=(env.batch_size[0], env.pomo_size, 0))

        done = False
        reward = None
        state_td = env.pre_step()
        while not done:
            next_td = policy(state_td)
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

    def calculate_loss(self, policy_out: dict, env, **kwargs):
        """Calculate loss for REINFORCE algorithm.
        Args:
            policy_out: Output of the policy network
        """
        reward = policy_out["reward"]  # shape: (batch, pomo)
        likelihood = policy_out["likelihood"]  # shape: (batch, pomo, problem)
        log_prob = likelihood.log().sum(dim=2)
        # shape = (batch, pomo)
        env.aug_flag = False
        from EasyNCO.phases import get_reinforce_baseline

        baseline = get_reinforce_baseline("shared")
        bl_value, bl_loss = baseline.eval(env.problems, reward, env=env)
        env.aug_flag = env.aug_type is not None
        advantage = reward - bl_value

        reinforce_loss = -(log_prob * advantage).mean()

        task_loss = reinforce_loss + bl_loss
        return task_loss

    def route_ranking(self, problem, solution, solution_flag):
        roll = (
            (solution_flag * torch.arange(solution.size(-1))[None, :]).max(-1)[1] + 1
        ) % solution.size(-1)
        roll_init = solution.size(-1) - roll[:, None]
        roll_diff = (
            torch.arange(solution.size(-1))[None, :].expand_as(solution) + roll[:, None]
        ) % solution.size(-1)
        now_solution = solution.gather(1, roll_diff)
        now_solution_flag = solution_flag.gather(1, roll_diff)
        solution = now_solution.clone()
        solution_flag = now_solution_flag.clone()
        vector = problem - problem[:, 0, :][:, None, :]
        vector_rank = vector.repeat(solution.size(0), 1, 1).gather(
            1, solution.unsqueeze(-1).expand(-1, -1, 2)
        )
        solution_start = torch.cummax(
            solution_flag.roll(dims=1, shifts=1)
            * torch.arange(solution.size(-1))[None, :],
            dim=-1,
        )[0]
        solution_end = (
            solution.size(-1)
            - 1
            - torch.flip(
                torch.cummax(
                    torch.flip(solution_flag, dims=[1])
                    * torch.arange(solution.size(-1))[None, :],
                    dim=-1,
                )[0],
                dims=[1],
            )
        )
        num_vector2 = solution_end - solution_start + 1
        cum_vr = torch.cumsum(vector_rank.clone(), dim=1)
        sum_vector2 = (
            cum_vr.clone().gather(1, solution_end.unsqueeze(-1).expand_as(vector_rank))
            - cum_vr.clone().gather(
                1, solution_start.unsqueeze(-1).expand_as(vector_rank)
            )
            + vector_rank.clone().gather(
                1, solution_start.unsqueeze(-1).expand_as(vector_rank)
            )
        )
        vector_angle = torch.atan2(
            sum_vector2[:, :, 1] / num_vector2, sum_vector2[:, :, 0] / num_vector2
        )
        total_indi = vector_angle
        total_rank = np.argsort(total_indi.cpu().numpy(), kind="stable")
        total_rank = torch.from_numpy(total_rank).cuda()
        roll = total_rank.min(-1)[1]
        roll_diff = (
            torch.arange(solution.size(-1))[None, :].expand_as(solution) + roll[:, None]
        ) % solution.size(-1)
        now_rank = total_rank.gather(1, roll_diff)
        solution_rank = solution.gather(1, now_rank)
        solution_flag_rank = solution_flag.gather(1, now_rank)
        roll_diff = (
            torch.arange(solution.size(-1))[None, :].expand_as(solution) + roll_init
        ) % solution.size(-1)
        solution_rank = solution_rank.gather(1, roll_diff)
        solution_flag_rank = solution_flag_rank.gather(1, roll_diff)
        return solution_rank, solution_flag_rank

    def route_ranking2(self, problem, solution, solution_flag):
        roll = (
            (solution_flag * torch.arange(solution.size(-1))[None, None, :]).max(-1)[1]
            + 1
        ) % solution.size(-1)
        roll_init = solution.size(-1) - roll[:, :, None]
        roll_diff = (
            torch.arange(solution.size(-1))[None, None, :].expand_as(solution)
            + roll[:, :, None]
        ) % solution.size(-1)
        now_solution = solution.gather(-1, roll_diff)
        now_solution_flag = solution_flag.gather(-1, roll_diff)
        solution = now_solution.clone()
        solution_flag = now_solution_flag.clone()
        vector = problem - problem[:, 0, :][:, None, :]
        vector_rank = (
            vector[:, None]
            .repeat(1, solution.size(1), 1, 1)
            .gather(2, solution.unsqueeze(-1).expand(-1, -1, -1, 2))
        )
        solution_start = torch.cummax(
            solution_flag.roll(dims=-1, shifts=1)
            * torch.arange(solution.size(-1))[None, :],
            dim=-1,
        )[0]
        solution_end = (
            solution.size(-1)
            - 1
            - torch.flip(
                torch.cummax(
                    torch.flip(solution_flag, dims=[-1])
                    * torch.arange(solution.size(-1))[None, :],
                    dim=-1,
                )[0],
                dims=[-1],
            )
        )
        num_vector2 = solution_end - solution_start + 1
        cum_vr = torch.cumsum(vector_rank.clone(), dim=-2)
        sum_vector2 = (
            cum_vr.clone().gather(2, solution_end.unsqueeze(-1).expand_as(vector_rank))
            - cum_vr.clone().gather(
                2, solution_start.unsqueeze(-1).expand_as(vector_rank)
            )
            + vector_rank.clone().gather(
                2, solution_start.unsqueeze(-1).expand_as(vector_rank)
            )
        )
        vector_angle = torch.atan2(
            sum_vector2[:, :, :, 1] / num_vector2, sum_vector2[:, :, :, 0] / num_vector2
        )
        total_indi = vector_angle
        total_rank = np.argsort(total_indi.cpu().numpy(), kind="stable")
        total_rank = torch.from_numpy(total_rank).cuda()
        roll = total_rank.min(-1)[1]
        roll_diff = (
            torch.arange(solution.size(-1))[None, None, :].expand_as(solution)
            + roll[:, :, None]
        ) % solution.size(-1)
        now_rank = total_rank.gather(-1, roll_diff)
        solution_rank = solution.gather(-1, now_rank)
        solution_flag_rank = solution_flag.gather(-1, now_rank)
        roll_diff = (
            torch.arange(solution.size(-1))[None, None, :].expand_as(solution)
            + roll_init
        ) % solution.size(-1)
        solution_rank = solution_rank.gather(-1, roll_diff)
        solution_flag_rank = solution_flag_rank.gather(-1, roll_diff)
        return solution_rank, solution_flag_rank

def cal_leagal_tsp(problem_size, solution):
    logger.info("!!!! Check legality of the solution, this function may be slow, if you want to skip it, please comment this function")
    for i in range(solution.size(0)):
        for j in range(solution.size(1)):
            unique_node_list_len = len(
                torch.unique(solution[i][j])
            )
            assert unique_node_list_len == problem_size, (
                f"rrc process error, unique_node_list_len:{unique_node_list_len}, problem_size:{problem_size}"
            )

def cal_leagal_cvrp(demand, order_node, order_flag):
    logger.info("!!!! Check legality of the solution, this function may be slow, if you want to skip it, please comment this function")
    order_node_ = order_node.clone()
    order_flag_ = order_flag.clone()
    demand_ = demand.clone()
    for i in range(order_node_.size(0)):
        for k in range(order_node_.size(1)):
            list_d = []
            now = 0
            for j in range(order_node_.size(2)):
                now += demand_[i, order_node_[i, k, j]]
                if order_flag_[i, k, j] == 1:
                    list_d.append(now)
                    now = 0
            list_demand = torch.stack(list_d, 0)
            list_demand[0] += now
            if (list_demand > 1 + 1e-5).any():
                # print("illeagal")
                logger.error("Illegal solution detected!")
            # else:
            #    print('leagal')
