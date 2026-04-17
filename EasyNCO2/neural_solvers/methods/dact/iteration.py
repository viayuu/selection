from neural_solvers.pipeline import Iteration
from typing import Any, Tuple, Literal
from torchrl.envs import EnvBase
from tensordict import TensorDict
import torch
from EasyNCO.neural_solvers.methods.lih.utils import get_all_2opt
from EasyNCO.utils.utils import getLogger

logger = getLogger(__name__)


class Iteration_tool():
    def __init__(self,problems,problems_type):
        self.problems = problems
        self.problems_type = problems_type
        if self.problems_type == 'tsp':
            self.problem_size = problems.shape[1]
        else:
            self.problem_size_extend = problems.shape[1]

    def get_costs(self,batch,rec):
        batch_size, size = rec.size()
        d1 = batch[:, :, 0:2].gather(1, rec.long().unsqueeze(-1).expand(batch_size, size, 2))
        d2 = batch[:, :, 0:2]
        length = ((d1 - d2).norm(p=2, dim=2)).sum(1)
        return  length

    def two_opt(self, solution, first, second):

        rec = solution.clone()

        # fix connection for first node
        argsort = solution.argsort()
        pre_first = argsort.gather(1, first)
        pre_first = torch.where(pre_first != second, pre_first, first)
        rec.scatter_(1, pre_first, second)

        # fix connection for second node
        post_second = solution.gather(1, second)
        post_second = torch.where(post_second != first, post_second, second)
        rec.scatter_(1, first, post_second)

        # reverse loop:
        cur = first
        if self.problems_type == 'cvrp':
            len = self.problem_size_extend
        else:
            len = self.problem_size

        for i in range(len):
            cur_next = solution.gather(1, cur)
            rec.scatter_(1, cur_next, torch.where(cur != second, cur, rec.gather(1, cur_next)))
            cur = torch.where(cur != second, cur_next, cur)

        return rec

    def operate(self, td: TensorDict) -> TensorDict:
        bs =td['exchange'].size(0)
        first = td['exchange'][:, 0].view(bs, 1)
        second = td['exchange'][:, 1].view(bs, 1)
        pre_bsf = td['pre_length'].view(bs, -1)
        next_state = self.two_opt(td['solution'], first, second)
        new_obj = self.get_costs(self.problems, next_state)
        now_bsf = torch.min(torch.cat((new_obj[:, None], pre_bsf[:, -1, None]), -1), -1)[0]
        reward = pre_bsf[:, -1] - now_bsf

        pre_length = td['current_length']
        now_length = new_obj
        rec_new = next_state

        next_one = {
            "solution":rec_new,
            "reward": reward,
        }

        next_td = td.clone()
        next_td.update(
            {
                'pre_length': pre_length,
                'current_length': now_length,
                'solution': rec_new,
                "next": next_one,
            }
        )

        return next_td



class DACTIteration(Iteration):
    def run(self,
            td: TensorDict,
            env: EnvBase,
            initialization_out: dict,
            phase: Literal["train", "eval"],
            max_steps: int = 0,
            **kwargs,
    ) -> dict:
        tool = Iteration_tool(initialization_out['locs'],env.env_name)
        if phase == "train":
            raise NotImplementedError("dact_train not implemented")
        else:
            batch_size = env.batch_size[0]
            no_aug_batch_size = env.batch_size[0] // env.aug_factor
            self.td = td
            self.do_perturb = True
            self.P = 250
            self.do_sample = True
            solving_state = torch.zeros(batch_size, 1).long()
            length_current = []
            best_solution = self.td['solution'].clone()
            if env.env_name == 'tsp':
                for T in range(max_steps):
                    self.td = self.policy(self.td, do_sample=self.do_sample)
                    # self.td = env.step(self.td)
                    self.td = tool.operate(self.td)
                    length_current.append(self.td['current_length'])
                    reward = torch.squeeze(self.td['next']['reward'])
                    # Record the number of times no improvement occurs;
                    # if no improvement happens for too long, a perturbation will be applied
                    if self.do_perturb:
                        solving_state[:, :1] = (1 - (reward > 0).view(-1, 1).long()) * (solving_state[:, :1] + 1)
                        perturb_index = (solving_state[:, :1] >= self.P).view(-1)
                        solving_state[:, :1][perturb_index.view(-1, 1)] *= 0
                        pertrb_cnt = perturb_index.sum().item()
                        if pertrb_cnt > 0:
                            self.td['solution'][perturb_index] = best_solution[perturb_index]

                        best_solution[reward > 0] = self.td['solution'][reward > 0]

                    if T % 100 == 0:
                        current_m = torch.stack(length_current, 0)
                        min = current_m.min(0)[0]
                        logger.info(f"T={T},length_current.mean={min.mean()}")

            elif env.env_name == 'cvrp':  # cvrp':
                for T in range(max_steps):
                    self.td = self.policy(self.td, do_sample=self.do_sample)
                    # self.td = env.step(self.td)
                    self.td = tool.operate(self.td)
                    length_current.append(self.td['current_length'])
                    reward = self.td['pre_length'] - self.td['current_length']
                    # Record the number of times no improvement occurs;
                    # if no improvement happens for too long, a perturbation will be applied
                    if self.do_perturb:
                        solving_state[:, :1] = (1 - (reward > 0).view(-1, 1).long()) * (solving_state[:, :1] + 1)
                        perturb_index = (solving_state[:, :1] >= self.P).view(-1)
                        solving_state[:, :1][perturb_index.view(-1, 1)] *= 0
                        pertrb_cnt = perturb_index.sum().item()
                        if pertrb_cnt > 0:
                            self.td['solution'][perturb_index] = best_solution[perturb_index]

                        best_solution[reward > 0] = self.td['solution'][reward > 0]

                    if T % 50 == 0:
                        current_m = torch.stack(length_current, 0)
                        min = current_m.min(0)[0]
                        logger.info(f"T={T},length_current.mean={min.mean()}")

            best_solution = []
            current_m = torch.stack(length_current, 0)
            best_solution.append(current_m.min(0)[0])
            best = torch.cat(best_solution, 0)
            ret = {
                "no_aug_score": best[:no_aug_batch_size].mean(),
                "aug_score": best.mean(),
            }
            return ret