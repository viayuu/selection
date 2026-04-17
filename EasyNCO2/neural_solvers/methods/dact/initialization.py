from neural_solvers.pipeline.initialization import Initialization
from typing import Any, Tuple, Literal
from torchrl.envs import EnvBase
from tensordict import TensorDict
import torch

class Initialization_tool():
    def __init__(self, problems, problems_type):
        self.problems = problems
        self.problems_type = problems_type
        if self.problems_type == 'tsp':
            self.problem_size = problems.shape[1]
        else:
            self.problem_size = problems.shape[1]-1

            if self.problem_size < 50:
                dummy_rate = 0.5
            elif self.problem_size < 100:
                dummy_rate = 0.4
            else:
                dummy_rate = 0.2

            self.problem_size_extend = int(self.problem_size * (1 + dummy_rate))
            self.dummy_size = self.problem_size_extend - self.problem_size
            extend_num = self.problem_size_extend - self.problem_size
            extend_depot = self.problems[:, 0, :].unsqueeze(1).expand(-1, extend_num - 1, -1)
            self.problems = torch.cat([extend_depot, self.problems], dim=1)

    def get_initial_solutions(self,batch,method):
        if self.problems_type == 'tsp':
            batch_size = batch.size(0)
            def get_solution(methods):
                if methods == 'random':
                    set = torch.rand(batch_size, self.problem_size).argsort().long()
                    rec = torch.zeros(batch_size, self.problem_size).long()
                    index = torch.zeros(batch_size, 1).long()

                    for i in range(self.problem_size - 1):
                        rec.scatter_(1, set.gather(1, index + i), set.gather(1, index + i + 1))
                    rec.scatter_(1, set[:, -1].view(-1, 1), set.gather(1, index))
                    return rec

                elif methods == 'greedy':
                    candidates = torch.ones(batch_size, self.problem_size).bool()
                    rec = torch.zeros(batch_size, self.problem_size).long()
                    selected_node = torch.zeros(batch_size, 1).long()
                    candidates.scatter_(1, selected_node, 0)

                    for i in range(self.problem_size - 1):
                        d1 = batch.gather(1, selected_node.unsqueeze(-1).expand(batch_size, self.problem_size, 2))
                        d2 = batch

                        dists = (d1 - d2).norm(p=2, dim=2)
                        dists[~candidates] = 1e5

                        next_selected_node = dists.min(-1)[1].view(-1, 1)
                        rec.scatter_(1, selected_node, next_selected_node)
                        candidates.scatter_(1, next_selected_node, 0)
                        selected_node = next_selected_node

                    return rec

                else:
                    raise NotImplementedError()
            return get_solution(method).expand(batch_size, self.problem_size).clone()
        else:
            batch_size = batch.size(0)
            p_size = self.problem_size
            if method == 'random':
                candidates = torch.ones(batch_size, self.problem_size_extend).bool()
                candidates[:, :self.dummy_size] = False

                rec = torch.zeros(batch_size, self.problem_size_extend).long()
                selected_node = torch.zeros(batch_size, 1).long()
                cum_demand = torch.zeros(batch_size, 2)

                demand = batch[:, :, -1].cpu()

                for i in range(self.problem_size_extend - 1):
                    dists = torch.arange(p_size).view(-1, p_size).repeat(batch_size, 1)

                    dists.scatter_(1, selected_node, 1e5)
                    dists[~candidates] = 1e5

                    dists[cum_demand[:, -1:] + demand > 1.] = 1e5
                    dists.scatter_(1, cum_demand[:, :-1].long() + 1, 1e4)

                    next_selected_node = dists.min(-1)[1].view(-1, 1)
                    selected_demand = demand.gather(1, next_selected_node)
                    cum_demand[:, -1:] = torch.where(selected_demand > 0, selected_demand + cum_demand[:, -1:],
                                                     0 * cum_demand[:, -1:])
                    cum_demand[:, :-1] = torch.where(selected_demand > 0, cum_demand[:, :-1], cum_demand[:, :-1] + 1)

                    rec.scatter_(1, selected_node, next_selected_node)
                    candidates.scatter_(1, next_selected_node, 0)
                    selected_node = next_selected_node

            elif method == 'greedy':

                candidates = torch.ones(batch_size, self.problem_size_extend).bool()
                candidates[:, :self.dummy_size] = False

                rec = torch.zeros(batch_size, self.problem_size_extend).long()
                selected_node = torch.zeros(batch_size, 1).long()
                cum_demand = torch.zeros(batch_size, 2)

                d2 = batch[:, :, 0:2]
                demand = batch[:, :, -1]

                for i in range(self.problem_size_extend - 1):
                    d1 = batch[:, :, 0:2].gather(1, selected_node.unsqueeze(-1).expand(batch_size,
                                                                                       self.problem_size_extend, 2))
                    dists = (d1 - d2).norm(p=2, dim=2)

                    dists.scatter_(1, selected_node, 1e5)
                    dists[~candidates] = 1e5

                    dists[cum_demand[:, -1:] + demand > 1.] = 1e5
                    dists.scatter_(1, cum_demand[:, :-1].long() + 1, 1e4)

                    next_selected_node = dists.min(-1)[1].view(-1, 1)
                    selected_demand = demand.gather(1, next_selected_node)
                    cum_demand[:, -1:] = torch.where(selected_demand > 0, selected_demand + cum_demand[:, -1:],
                                                     0 * cum_demand[:, -1:])
                    cum_demand[:, :-1] = torch.where(selected_demand > 0, cum_demand[:, :-1], cum_demand[:, :-1] + 1)

                    rec.scatter_(1, selected_node, next_selected_node)
                    candidates.scatter_(1, next_selected_node, 0)
                    selected_node = next_selected_node

            else:
                raise NotImplementedError()
            return rec.expand(batch_size, self.problem_size_extend).clone()


    def get_costs(self,batch,rec):
        batch_size, size = rec.size()
        d1 = batch[:, :, 0:2].gather(1, rec.long().unsqueeze(-1).expand(batch_size, size, 2))
        d2 = batch[:, :, 0:2]
        length = ((d1 - d2).norm(p=2, dim=2)).sum(1)
        return  length

    def initial(self, method):
        batch_size = self.problems.shape[0]
        self.solution = self.get_initial_solutions(self.problems, method)
        self.current_length = self.get_costs(self.problems, self.solution)
        self.pre_length = self.current_length
        # node pair #shape (batch,2)
        self.exchange = torch.zeros((batch_size, 2), dtype=torch.long)

        return TensorDict({
            'solution': self.solution,  # shape: (batch,problem)
            'current_length': self.current_length,  # shape: (batch,1)
            'pre_length': self.pre_length,  # shape : (batch,1)
            'exchange': self.exchange,  # shape :(batch,2)
            'locs': self.problems,
        }, batch_size=torch.Size([batch_size])
        )

class DACTInitialization(Initialization):
    def run(self,
            env: EnvBase,
            batch: int,
            strategy: str,
            phase: Literal["train", "eval"],
            **kwargs,
    ) -> Tuple[TensorDict, Any]:
        if env.env_name == 'tsp':
            problem_size = batch.shape[1]
        elif env.env_name == 'cvrp':
            problem_size = batch.shape[1] - 1
        self.policy.initial_problem_size(problem_size)

        if phase == "train":
            self.policy.train()
            env.load_problems(batch, batch.size(0))
            reset_td = env.reset()

        else:
            self.policy.eval()
            env.load_problems(batch, batch.size(0))
            tool = Initialization_tool(batch, env.env_name)
            reset_td = tool.initial(method='greedy')
            # reset_td = env.reset(method='greedy')

        return reset_td, reset_td
