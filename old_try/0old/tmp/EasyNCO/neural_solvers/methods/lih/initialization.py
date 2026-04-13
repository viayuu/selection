from neural_solvers.pipeline.initialization import Initialization
from typing import Any, Tuple, Literal
from torchrl.envs import EnvBase
from tensordict import TensorDict
import torch
from EasyNCO.neural_solvers.methods.lih.utils import lih_reset

class Initialization_tool():
    def __init__(self, problems, problems_type):
        self.problems = problems
        self.problems_type = problems_type

    def get_initial_solutions(self,batch):
        batch_size = batch.size(0)
        if self.problems_type == 'tsp':
            problem_size = batch.size(1)
        else:
            problem_size = batch.size(1) - 1

        def get_solution():
            set = torch.rand(batch_size, problem_size).argsort().long()
            rec = torch.zeros(batch_size, problem_size).long()
            index = torch.zeros(batch_size, 1).long()

            for i in range(problem_size - 1):
                rec.scatter_(1, set.gather(1, index + i), set.gather(1, index + i + 1))
            rec.scatter_(1, set[:, -1].view(-1, 1), set.gather(1, index))
            return rec

        return get_solution().expand(batch_size, problem_size).clone()

    def get_costs(self, batch, rec):
        if self.problems_type == 'tsp':
            dn = self.problems.gather(1, rec.long().unsqueeze(-1).expand_as(self.problems))
            length = (dn[:, 1:] - dn[:, :-1]).norm(p=2, dim=2).sum(1) + (dn[:, 0] - dn[:, -1]).norm(p=2, dim=1)
        else:
            pi = rec - 1
            d = batch[:, :, 0:2].gather(1, pi[..., None].expand(*pi.size(), batch[:, :, 0:2].size(-1)))
            length = (d[:, 1:] - d[:, :-1]).norm(p=2, dim=2).sum(1) + (d[:, 0] - d[:, -1]).norm(p=2, dim=1)

        return length
    
    
    def initial(self):
        if self.problems_type == 'tsp':
            batch_size = self.problems.shape[0]
            self.solution = self.get_initial_solutions(self.problems)
            self.current_length = self.get_costs(self.problems, self.solution)
            self.pre_length = self.current_length
            # node pair #shape (batch,2)
            self.exchange = torch.zeros((batch_size, 2), dtype=torch.long)
        else:
            self.problem_size = self.problems.shape[1]-1
            batch_size = self.problems.shape[0]
            if self.problem_size <= 50:
                problem_size_extend = self.problem_size * 2
            elif self.problem_size == 100:
                problem_size_extend = 128
            batch_e = lih_reset(self.problems,self.problem_size,batch_size)
            demand_extend = batch_e['demand']
            locs_extend = batch_e['loc']

            self.solution = torch.zeros(batch_size, problem_size_extend).long()
            self.solution[demand_extend != 0.] = torch.tensor([i + 1 for i in range(self.problem_size)] * batch_size)

            extra = problem_size_extend - self.problem_size
            self.solution[demand_extend == 0.] = torch.tensor(
                [i + self.problem_size + 1 for i in range(extra)] * batch_size)

            self.depot_node_demand = demand_extend.gather(1, self.solution.sort()[1])
            self.depot_node_xy = locs_extend.gather(1, self.solution.sort()[1][..., None].expand(
                *self.solution.sort()[1].size(), 2))
            self.problems = torch.cat((self.depot_node_xy, self.depot_node_demand.unsqueeze(2)), dim=2)

            self.exchange = torch.zeros((batch_size, 2), dtype=torch.long)
            self.current_length = self.get_costs(self.problems, self.solution)
            self.pre_length = self.current_length

        return TensorDict({
            'solution': self.solution,  # shape: (batch,problem)
            'current_length': self.current_length,  # shape: (batch,1)
            'pre_length': self.pre_length,  # shape : (batch,1)
            'exchange': self.exchange,  # shape :(batch,2)
            'locs': self.problems,
        }, batch_size=torch.Size([batch_size])
        )
    
class LIHInitialization(Initialization):
    def run(self,
            env: EnvBase,
            batch: int,
            strategy: str,
            phase: Literal["train", "eval"],
            **kwargs,
    ) -> Tuple[TensorDict, Any]:


        if phase == "train":
            self.policy.train()
            env.load_problems(batch, batch.size(0))
            tool = Initialization_tool(env.problems, env.env_name)
            reset_td = tool.initial()
            # reset_td = env.reset()
        else:
            self.policy.eval()
            env.load_problems(batch, batch.size(0))
            tool = Initialization_tool(env.problems, env.env_name)
            # # if env.env_name == 'cvrp':
            # #     env.problem_size = batch.shape[1] - 1
            reset_td = tool.initial()
            # reset_td1 = env.reset()


        return reset_td, reset_td
