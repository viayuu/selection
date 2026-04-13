from EasyNCO.neural_solvers.pipeline import Initialization
from typing import Any, Tuple, Literal
import torch
import torch.nn as nn
from tensordict import TensorDict
from torchrl.envs import EnvBase
from torch.optim import Adam as Optimizer
from EasyNCO.neural_solvers.backbones.GNN.DATATransform import gen_pyg_data_tsp
from EasyNCO.neural_solvers.methods.glop.cvrp_heatmap import (
    gen_pyg_data as gen_pyg_data_cvrp,
)
from torch.distributions import Categorical


class UDCInitialization(Initialization):
    '''
    Initialization class for UDC model
    '''
    def __init__(self, policy: nn.Module):
        super().__init__(policy.model)

    def run(
        self,
        env: EnvBase,
        batch: int,
        strategy: str,
        phase: Literal["train", "eval"],
        **kwargs,
    ) -> Tuple[TensorDict, Any]:
        env.pomo_size = kwargs.get("pomo_size", 50)
        if phase == "train":
            env.load_problems(batch, batch.size(0))
            if env.env_name == "tsp":
                return self._run_train_tsp(env, strategy, **kwargs)
            elif env.env_name == "cvrp":
                return self._run_train_cvrp(env, batch, strategy, **kwargs)
        else:
            if env.env_name == "tsp":
                return self._run_eval_tsp(env, batch, strategy, **kwargs)
            if env.env_name == "cvrp":
                return self._run_eval_cvrp(env, batch, strategy, **kwargs)

    def _run_train_tsp(
        self, env: EnvBase, strategy: str, **kwargs
    ) -> Tuple[TensorDict, Any]:
        optimizer_t = Optimizer(
            self.policy.model_t.parameters(), **kwargs["optimizer_t"]
        )
        self.policy.model_p.train()
        self.policy.model_t.train()
        raw_problems = env.problems
        assert raw_problems.shape[0] == 1, "Only support batch size of 1 for TSP"
        pyg_data = self._gen_pyg_data(
            data=raw_problems.squeeze(0), env_name=env.env_name
        )
        index = torch.randint(0, raw_problems.size(1), [env.sample_size])
        logp = torch.zeros(env.sample_size, dtype=torch.float32)
        visited = torch.zeros_like(index)[:, None].repeat(1, raw_problems.size(1))
        solution = index[:, None]
        visited = visited.scatter(-1, solution[:, 0:1], 1)
        selected = solution
        heatmap = None
        self.policy.model_p.pre(pyg_data)
        while solution.size(-1) < raw_problems.size(1):
            if (solution.size(-1) - 1) % env.sub_problem_size == 0:
                node_emb, heatmap = self.policy.model_p(
                    solution=solution, visited=visited
                )
                heatmap = heatmap / (heatmap.min() + 1e-5)
                heatmap = (
                    self.policy.model_p.reshape(pyg_data, heatmap, udc_flag=True) + 1e-5
                )
            row = (
                heatmap.gather(1, selected[:, None, :].expand(-1, -1, heatmap.size(-1)))
                .clone()
                .squeeze(1)
                * (1 - visited).clone()
            )
            dist = Categorical(row)
            item = dist.sample()  # row.argmax(dim=-1)  #
            log_prob = dist.log_prob(item)
            selected = item[
                :, None
            ]  # row.reshape(batch_size * row.size(1), -1).multinomial(1).squeeze(dim=1).reshape(batch_size, row.size(1))[:, :, None]
            logp += log_prob  # row.gather(2, selected).log().squeeze()
            visited = visited.scatter(-1, selected, 1)
            solution = torch.cat((solution, selected), dim=-1)
        out = {
            "raw_problems": raw_problems,
            "optimizer_t": optimizer_t,
            "logp": logp,
            "solution": solution,
        }
        return TensorDict({}, batch_size=[1]), out

    def _run_train_cvrp(
        self, env: EnvBase, batch: TensorDict, strategy: str, **kwargs
    ) -> Tuple[TensorDict, Any]:
        optimizer_t = Optimizer(
            self.policy.model_t.parameters(), **kwargs["optimizer_t"]
        )
        self.policy.model_p.train()
        self.policy.model_t.train()
        raw_problems = env.problems
        pyg_data = self._gen_pyg_data(
            data=raw_problems.squeeze(0), env_name=env.env_name
        )
        logp = torch.zeros(env.sample_size, dtype=torch.float32)
        index = torch.zeros(env.sample_size, dtype=torch.long, device=logp.device)
        vehicle_count = torch.zeros((env.sample_size,), device=logp.device)
        demand_count = torch.zeros((env.sample_size,), device=logp.device)
        visited = torch.zeros_like(index)[:, None].repeat(1, raw_problems.size(1))
        solution_raw = index[:, None]
        visited = visited.scatter(-1, solution_raw[:, 0:1], 1)
        selected = solution_raw
        self.policy.model_p.pre(pyg_data)
        max_vehicle = (raw_problems[:, :, 2].sum().ceil() + 1).item()
        total_demand = raw_problems[:, :, 2].sum()
        remaining_demand = total_demand.clone()
        solution = torch.zeros(
            (env.sample_size, raw_problems.size(1) - 1), dtype=torch.long
        )
        solution[:, 0] = solution_raw[:, 0]
        step = 0
        solution_flag = torch.zeros(
            (env.sample_size, raw_problems.size(1) - 1), dtype=torch.long
        )
        node_count = -1 * torch.ones((env.sample_size, 1), dtype=torch.long)
        capacity = torch.ones_like(index)[:, None].float()
        capacity -= raw_problems[:, :, 2].expand((env.sample_size, -1)).gather(-1, selected)
        for i in range(env.problem_size // env.sub_problem_size):
            node_emb, heatmap = self.policy.model_p(
                solution=solution, visited=visited, selected=selected
            )
            heatmap = heatmap / (heatmap.min() + 1e-5)
            heatmap = self.policy.model_p.reshape(pyg_data, heatmap, udc_flag=True) + 1e-5
            while (
                (visited.sum(-1) - i * env.sub_problem_size) < env.sub_problem_size
            ).any():
                step += 1
                capacity_mask = (raw_problems[:, :, 2] > capacity + 1e-5).long().squeeze()
                row = (
                    heatmap.gather(1, selected[:, None, :].expand(-1, -1, heatmap.size(-1)))
                    .clone()
                    .squeeze(1)
                    * (1 - visited).clone()
                    * (1 - capacity_mask).clone()
                )
                row[:, 0][selected[:, 0] == 0] = 0
                row[:, 0][remaining_demand > (max_vehicle - vehicle_count)] = 0
                row[:, 0][row[:, 1:].sum(-1) < 1e-8] = 1
                dist = Categorical(row)
                item = dist.sample()  # row.argmax(dim=-1)  #
                log_prob = dist.log_prob(item)
                selected = item[
                    :, None
                ]  # row.reshape(batch_size * row.size(1), -1).multinomial(1).squeeze(dim=1).reshape(batch_size, row.size(1))[:, :, None]
                logp += log_prob  # row.gather(2, selected).log().squeeze()
                demand_count += (
                    raw_problems[:, :, 2]
                    .expand(env.sample_size, -1)
                    .gather(1, selected)
                    .squeeze()
                )
                remaining_demand = total_demand - demand_count
                visited = visited.scatter(-1, selected, 1)
                visited[:, 0] = 0
                capacity -= (
                    raw_problems[:, :, 2].expand((env.sample_size, -1)).gather(-1, selected)
                )
                capacity[selected == 0] = 1
                vehicle_count[item == 0] += 1
                if step > 1:
                    solution_flag = solution_flag.scatter_add(
                        dim=-1, index=node_count, src=(selected == 0).long()
                    )
                node_count[selected != 0] += 1
                solution = solution.scatter_add(dim=-1, index=node_count, src=selected)
        solution_flag[:, -1] = 1
        out = {
            "raw_problems": raw_problems,
            "optimizer_t": optimizer_t,
            "logp": logp,
            "solution": solution,
            "solution_flag": solution_flag,
        }
        return TensorDict({}, batch_size=[1]), out

    def _run_eval_tsp(
        self, env: EnvBase, batch: TensorDict, strategy: str, **kwargs
    ) -> Tuple[TensorDict, Any]:
        solution_list = []
        test_num_episode = batch.size(0)
        for i in range(test_num_episode):
            if batch.size(1) < 100:
                self.policy.model_p.par_net_heu.k_sparse = batch.size(1) - 1
                k_sparse = batch.size(1) - 1
            else:
                self.policy.model_p.par_net_heu.k_sparse = 100
                k_sparse = 100
            env.load_problems(batch[i : i + 1].clone().detach().float(), 1)
            raw_problems = env.raw_problems
            if raw_problems.size(1) >= 2000:
                env.sub_problem_size = raw_problems.size(1) // 10
            elif raw_problems.size(1) <= 100:
                env.sub_problem_size = raw_problems.size(1)
            pyg_data = self._gen_pyg_data(raw_problems.squeeze(0), env_name=env.env_name, k_sparse=k_sparse)
            torch.manual_seed(i)
            index = torch.randint(0, raw_problems.size(1), [env.aug_factor])
            visited = torch.zeros_like(index)[:, None].repeat(1, raw_problems.size(1))
            solution = index[:, None]
            visited = visited.scatter(-1, solution[:, 0:1], 1)
            selected = solution
            heatmap = None
            self.policy.model_p.pre(pyg_data)
            while solution.size(-1) < raw_problems.size(1):
                if (solution.size(-1) - 1) % env.sub_problem_size == 0:
                    _, heatmap = self.policy.model_p(solution=solution, visited=visited)
                    heatmap = heatmap / (heatmap.min() + 1e-5)
                    heatmap = (
                        self.policy.model_p.reshape(pyg_data, heatmap, udc_flag=True) + 1e-5
                    )
                row = (
                    heatmap.gather(1, selected[:, None, :].expand(-1, -1, heatmap.size(-1)))
                    .clone()
                    .squeeze(1)
                    * (1 - visited).clone()
                )
                item = row.max(-1)[1]
                selected = item[:, None]
                visited = visited.scatter(-1, selected, 1)
                solution = torch.cat((solution, selected), dim=-1)
            solution_list.append(solution)
        solution = torch.stack(solution_list, dim=0)
        td = TensorDict(
            {"batch": batch, "solution": solution}, batch_size=[test_num_episode]
        )
        out = {
            "test_num_episode": test_num_episode,
            "strategy": strategy,
        }
        return td, out

    def _run_eval_cvrp(
        self, env: EnvBase, batch: TensorDict, strategy: str, **kwargs
    ) -> Tuple[TensorDict, Any]:
        env.pomo_size = 1
        self.policy.model_p.eval()
        self.policy.model_t.eval()
        self.policy.model_t.set_decoder_strategy("greedy")
        solution_list = []
        solution_flag_list = []

        pre_step = 0
        if env.aug_factor > 0:
            pre_step = 3
        test_num_episode = batch.size(0)
        for j in range(test_num_episode):
            if batch.size(1) < 100:
                self.policy.model_p.par_net_heu.k_sparse = batch.size(1) - 1
                k_sparse = batch.size(1) - 1
            else:
                self.policy.model_p.par_net_heu.k_sparse = 100
                k_sparse = 100
            env.load_problems(batch[j : j + 1].clone().detach().float(), 1)
            raw_problems = env.problems
            if raw_problems.size(1) >= 2000:
                env.sub_problem_size = raw_problems.size(1) // 10
            elif raw_problems.size(1) <= 100:
                env.sub_problem_size = raw_problems.size(1) - 1
            pyg_data = self._gen_pyg_data(data=raw_problems.squeeze(0), env_name=env.env_name, k_sparse=k_sparse)
            index = torch.zeros(env.aug_factor, dtype=torch.long, device=batch.device)
            vehicle_count = torch.zeros((env.aug_factor,), device=index.device)
            demand_count = torch.zeros((env.aug_factor,), device=index.device)
            visited = torch.zeros_like(index)[:, None].repeat(1, raw_problems.size(1))
            solution_raw = index[:, None]
            visited = visited.scatter(-1, solution_raw[:, 0:1], 1)
            selected = solution_raw
            self.policy.model_p.pre(pyg_data)
            max_vehicle = (env.depot_node_demand.sum().ceil() + 1).item()
            total_demand = env.depot_node_demand.sum()
            remaining_demand = total_demand.clone()
            solution = torch.zeros(
                (env.aug_factor, raw_problems.size(1) - 1), dtype=torch.long
            )
            solution[:, 0] = solution_raw[:, 0]
            step = 0
            solution_flag = torch.zeros(
                (env.aug_factor, raw_problems.size(1) - 1), dtype=torch.long
            )
            node_count = -1 * torch.ones((env.aug_factor, 1), dtype=torch.long)
            capacity = torch.ones_like(index)[:, None].float()
            capacity -= env.depot_node_demand.expand((env.aug_factor, -1)).gather(
                -1, selected
            )
            for i in range((batch.size(1) - 1) // env.sub_problem_size + 1):
                node_emb, heatmap = self.policy.model_p(
                    solution=solution, selected=selected, visited=visited
                )
                heatmap = heatmap / (heatmap.min() + 1e-5)
                heatmap = (
                    self.policy.model_p.reshape(pyg_data, heatmap, udc_flag=True) + 1e-5
                )
                while (
                    (visited.sum(-1) - i * env.sub_problem_size) < env.sub_problem_size
                ).any():
                    step += 1
                    capacity_mask = (
                        (env.depot_node_demand > capacity + 1e-5).long().squeeze()
                    )
                    row = (
                        heatmap.gather(
                            1, selected[:, None, :].expand(-1, -1, heatmap.size(-1))
                        )
                        .clone()
                        .squeeze(1)
                        * (1 - visited).clone()
                        * (1 - capacity_mask).clone()
                    )
                    row[:, 0][selected[:, 0] == 0] = 0
                    row[:, 0][remaining_demand > (max_vehicle - vehicle_count)] = 0
                    row[:, 0][row[:, 1:].sum(-1) < 1e-8] = 1
                    if step < pre_step:
                        dist = Categorical(row)
                        item = dist.sample()  #! row.argmax(dim=-1)
                        # item = row.argmax(dim=-1)
                        selected = item[:, None]
                    else:
                        selected = row.max(-1)[1][:, None]  # row.argmax(dim=-1)
                    demand_count += (
                        env.depot_node_demand.expand(env.aug_factor, -1)
                        .gather(1, selected)
                        .squeeze()
                    )
                    remaining_demand = total_demand - demand_count
                    visited = visited.scatter(-1, selected, 1)
                    visited[:, 0] = 0
                    capacity -= env.depot_node_demand.expand((env.aug_factor, -1)).gather(
                        -1, selected
                    )
                    capacity[selected == 0] = 1
                    vehicle_count[selected.squeeze() == 0] += 1
                    if step > 1:
                        solution_flag = solution_flag.scatter_add(
                            dim=-1, index=node_count, src=(selected == 0).long()
                        )
                    node_count[selected != 0] += 1
                    solution = solution.scatter_add(dim=-1, index=node_count, src=selected)
                    if (visited.sum(-1) == raw_problems.size(1) - 1).all():
                        break
            solution_flag[:, -1] = 1
            solution_list.append(solution)
            solution_flag_list.append(solution_flag)
        solution = torch.stack(solution_list, dim=0)
        solution_flag = torch.stack(solution_flag_list, dim=0)
        td = TensorDict(
            {
                "batch": batch,
                "solution": solution,
                "solution_flag": solution_flag,
            },
            batch_size=[test_num_episode],
        )
        out = {
            "test_num_episode": test_num_episode,
            "strategy": strategy,
        }
        return td, out

    def _gen_pyg_data(self, data, env_name, k_sparse):
        if env_name == "tsp":
            pyg_data, _ = gen_pyg_data_tsp(data, infinity=False, udc_flag=True, k_sparse=k_sparse)
        elif env_name == "cvrp":
            pyg_data = gen_pyg_data_cvrp(data[:, :2], data[:, 2], 1, k_sparse=k_sparse)
        return pyg_data
