from neural_solvers.pipeline import Iteration
from typing import Any, Tuple, Literal
from torchrl.envs import EnvBase
from tensordict import TensorDict
import torch
from EasyNCO.neural_solvers.methods.lih.utils import get_all_2opt
from EasyNCO.utils.utils import getLogger

logger = getLogger(__name__)

class Memory:
    def __init__(self):
        self.bl_r = []
        self.bl_detach_r = []
        self.log_like_r = []
        self.reward_r = []
        self.ret = []

    def clear_memory(self):
        del self.bl_r[:]
        del self.bl_detach_r[:]
        del self.log_like_r[:]
        del self.reward_r[:]
        del self.ret[:]

    def update_memory(self,eval,eval_det,reward,ll):
        self.bl_r.append(eval)
        self.bl_detach_r.append(eval_det)
        self.reward_r.append(reward)
        self.log_like_r.append(ll)








class Iteration_tool():
    def __init__(self,problems,problems_type):
        self.problems = problems
        self.problems_type = problems_type
   
        self.dic = None
        self.cl = None

    def get_2opt_mask_LIH(self, rec, dic, cl):
        demand = self.problems[:, :, 2]  # 需求量 (batch_size, problem_extend)
        batch_size, problem_extend = demand.size()
        comb_num, g2s = dic.size()

        def check_is_valid(input, rec):

            batch_size_multi_com, problem_extend = rec.size()
            batch_size = input.size(0)

            # input从 (batch,problem_extend)变为(batch * 128取2的组合数, problem_extend)
            expanded_demand = input[:, None, :].expand(batch_size, batch_size_multi_com // batch_size, problem_extend)
            expanded_demand = expanded_demand.reshape(-1, problem_extend)

            # 根据 rec 的下标提取需求
            reordered_demand = expanded_demand.gather(1, rec.long() - 1)

            # 计算累积需求
            cumulative_demand = reordered_demand.cumsum(dim=1)

            # 根据cvrp50和100分类从处理
            if problem_extend <= 100:
                # 找出 rec 中仓库节点的索引
                depot_indices = torch.nonzero(rec > problem_extend // 2)[:, 1].reshape(batch_size_multi_com, -1)

                # 将仓库节点与路径末尾节点索引组合
                depot_indices_with_end = torch.cat(
                    (
                        depot_indices,
                        (problem_extend - 1) * torch.ones(batch_size_multi_com, 1, dtype=torch.long).cuda()),
                    dim=1
                )

                # 提取与仓库节点相关的需求总和
                depot_demand = cumulative_demand.gather(1, depot_indices_with_end)

                # 计算路径需求差，首尾合并需求
                depot_demand[:, 1:] -= depot_demand[:, :-1]
                depot_demand[:, 0] += depot_demand[:, -1]
                depot_demand[:, -1] = depot_demand[:, 0]

                # 判断所有路径需求是否满足约束
                is_valid = (depot_demand <= 1.0 + 1e-5).all(dim=1)
            else:
                depot_indices = rec.sort(dim=1)[1][:, 100:].sort(dim=1)[0]

                # 将仓库节点索引与路径末尾节点索引组合
                depot_indices_with_end = torch.cat(
                    (
                        depot_indices,
                        (problem_extend - 1) * torch.ones(batch_size_multi_com, 1, dtype=torch.long).cuda()),
                    dim=1
                )

                # 提取仓库节点和路径末尾节点的需求总和
                depot_demand = cumulative_demand.gather(1, depot_indices_with_end)

                # 计算路径需求差
                depot_demand_diff = depot_demand - torch.cat(
                    (torch.zeros(batch_size_multi_com, 1, device=depot_demand.device),
                     cumulative_demand.gather(1, depot_indices)), dim=1
                )

                # 合并路径首尾需求
                final_route_demand = torch.cat(
                    ((depot_demand_diff[:, 0] + depot_demand_diff[:, -1]).unsqueeze(1), depot_demand_diff[:, 1:-1]),
                    dim=1
                )

                # 判断所有路径需求是否满足约束
                is_valid = (final_route_demand <= 1.0 + 1e-5).all(dim=1)

            return is_valid

        def generate_2opt_mask(rec_seq, demand, dic, cl):
            """
            根据当前路径序列和 2-opt 组合生成可行性 mask。
            """
            rec_rep = rec_seq[:, None, :].expand(batch_size, comb_num, problem_extend).reshape(-1, problem_extend)
            mov_set = rec_rep.gather(1, dic.repeat(batch_size, 1))
            che_mask = check_is_valid(demand, mov_set)

            # 根据 cl 提取和排序路径组合索引
            cor = rec_rep.gather(1, cl.repeat(batch_size, 1)).sort(1)[0]
            order = (cor[:, 0] * (g2s * 2) + cor[:, 1]).view(batch_size, -1).sort(1)[1]

            # 根据排序重新排列 mask
            cm = che_mask.view(batch_size, -1).gather(1, order)
            return cm

        cm = generate_2opt_mask(rec, demand, dic, cl)

        return cm

    def two_opt(self, solution, first, second):
    # 在DACT中，solution是边集,solution[i]=j表示点i与点j之间有边
    # 在LIH中， solution是节点访问顺序
    #     if self.linked_list:
    #         rec = solution.clone()
    #
    #         # fix connection for first node
    #         argsort = solution.argsort()
    #         pre_first = argsort.gather(1, first)
    #         pre_first = torch.where(pre_first != second, pre_first, first)
    #         rec.scatter_(1, pre_first, second)
    #
    #         # fix connection for second node
    #         post_second = solution.gather(1, second)
    #         post_second = torch.where(post_second != first, post_second, second)
    #         rec.scatter_(1, first, post_second)
    #
    #         # reverse loop:
    #         cur = first
    #         for i in range(self.problem_size):
    #             cur_next = solution.gather(1, cur)
    #             rec.scatter_(1, cur_next, torch.where(cur != second, cur, rec.gather(1, cur_next)))
    #             cur = torch.where(cur != second, cur_next, cur)
    #
    #         return rec
        if self.problems_type == 'tsp':
            loc_of_first = torch.nonzero(solution.long() == first)
            loc_of_second = torch.nonzero(solution.long() == second)
        else:
            loc_of_first = torch.nonzero(solution.long() == first+1)
            loc_of_second = torch.nonzero(solution.long() == second+1)

        exchange_for_now = torch.cat((loc_of_first[:, 1][:, None], loc_of_second[:, 1][:, None]), 1)

        rec = solution.clone().cpu()
        exchange_sort = exchange_for_now.sort(1)[0].cpu()
        batch_size = solution.size()[0]
        for i in range(batch_size):
            inter = torch.narrow(rec[i], 0, exchange_sort[i][0],
                                 exchange_sort[i][1] - exchange_sort[i][0] + 1).clone()
            rec[i][exchange_sort[i][0]:exchange_sort[i][1] + 1] = torch.flip(inter, [0])
        rec = rec.cuda()

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
        if self.problems_type == 'cvrp':
            che_mask = self.get_2opt_mask_LIH(next_state, self.dic, self.cl)
            next_td.update({
                'che_mask': che_mask.reshape(td['log_likelihood'].shape[0], -1),
            })

        return next_td

    def get_costs(self, batch, rec):
        batch_size, size = rec.size()

        # if self.linked_list:
        #     d1 = batch.gather(1, rec.long().unsqueeze(-1).expand(batch_size, size, 2))
        #     d2 = batch
        #     length = (d1 - d2).norm(p=2, dim=2).sum(1)
        # else:
        if self.problems_type == 'tsp':
            dn = self.problems.gather(1, rec.long().unsqueeze(-1).expand_as(self.problems))
            length = (dn[:, 1:] - dn[:, :-1]).norm(p=2, dim=2).sum(1) + (dn[:, 0] - dn[:, -1]).norm(p=2, dim=1)
        else:
            pi = rec - 1
            d = batch[:, :, 0:2].gather(1, pi[..., None].expand(*pi.size(), batch[:, :, 0:2].size(-1)))
            length = (d[:, 1:] - d[:, :-1]).norm(p=2, dim=2).sum(1) + (d[:, 0] - d[:, -1]).norm(p=2, dim=1)

        return length


class LIHIteration(Iteration):
    def run(self,
            td: TensorDict,
            env: EnvBase,
            initialization_out: dict,
            phase: Literal["train", "eval"],
            max_steps,
            **kwargs,
    ) -> dict:

        problems = initialization_out['locs']
        tool = Iteration_tool(problems,env.env_name)
        if phase == "train":
            n_step_return = kwargs.get('n_step_return')

            if env.problem_size < 50:
                max_steps = max_steps['train_le50']
                n_step_return = n_step_return['train_le50']
            else:
                max_steps = max_steps['train_geq50']
                n_step_return = n_step_return['train_geq50']

            T = max_steps // n_step_return
            loss_list =[]
            if env.env_name == 'tsp':
                Y = 0.99
                self.td = td
                for i in range(T):
                    memory = Memory()
                    for j in range(n_step_return):
                        eval_det, eval = self.baseline.eval(td=self.td)
                        self.td['input_info'], self.td['pos_enc'] = self.policy.get_input_and_pe(self.td)
                        self.td = self.policy(self.td)
                        self.td = tool.operate(self.td)

                        reward = self.td['next']['reward'].squeeze()
                        memory.update_memory(eval, eval_det, reward, self.td['log_likelihood'])

                    reward_r_r = memory.reward_r[::-1]
                    next_return, _ = self.baseline.eval(td=self.td)

                    for r in range(len(memory.reward_r)):
                        this_return = next_return * Y + reward_r_r[r]
                        memory.ret.append(this_return)
                        next_return = this_return

                    val_tru = torch.stack(memory.ret[::-1], 0)
                    val_est = torch.stack(memory.bl_r, 0)
                    val_est_det = torch.stack(memory.bl_detach_r, 0)

                    log_like = torch.stack(memory.log_like_r, 0)

                    bl_loss = (val_tru - val_est).pow(2).mean()

                    reinforce_loss = ((val_tru - val_est_det) * log_like).mean()

                    loss = bl_loss - reinforce_loss
                    loss_list.append(loss)

                    opt = self.optimizers()
                    opt.zero_grad()
                    self.manual_backward(loss)
                    torch.nn.utils.clip_grad_norm_(
                        parameters=[p for group in opt.param_groups for p in group['params']],
                        max_norm=1.0,
                        norm_type=2
                    )
                    opt.step()
                    memory.clear_memory()

            elif env.env_name == 'cvrp':
                if tool.dic == None:
                    tool.cl, tool.dic = get_all_2opt(env.problem_size)
                self.td = td
                batch_size = env.batch_size[0]
                for i in range(T):
                    Y = 0.996
                    reward_r = []
                    log_like_r = []
                    ret = []
                    rec_rec = []
                    rec_ii = []
                    rec_act = []
                    rec_pos = []
                    rec_mask = []
                    for j in range(n_step_return):
                        if T == 0 and i == 0:
                            self.td["che_mask"] = tool.get_2opt_mask_LIH(self.td['solution'], tool.dic, tool.cl)

                        self.td["input_info"], self.td["pos_enc"] = self.policy.get_input_and_pe(self.td)
                        rec_mask.append(self.td["che_mask"].view(batch_size, -1))
                        self.td = self.old_policy(self.td, policy_old=True)
                        self.td = tool.operate(self.td)

                        reward = self.td['next']['reward'].squeeze()

                        rec_act.append(self.td['action'])  # 按概率采样的动作
                        rec_pos.append(self.td["pos_enc"])
                        rec_rec.append(self.td['solution'])
                        rec_ii.append(self.td["input_info"])
                        reward_r.append(reward)
                        log_like_r.append(self.td['log_likelihood'].detach())

                    reward_r_r = reward_r[::-1]
                    self.td['input_info'], self.td['pos_enc'] = self.policy.get_input_and_pe(self.td)
                    next_return, _ = self.baseline.eval(td=self.td)
                    for r in range(len(reward_r)):
                        this_return = next_return * Y + reward_r_r[r]
                        ret.append(this_return)
                        next_return = this_return

                    rol_act = torch.cat(rec_act)
                    rol_input = torch.cat(rec_ii)
                    rol_pos = torch.cat(rec_pos)
                    rol_mask = torch.cat(rec_mask)
                    val_tru = torch.cat(ret[::-1])
                    log_like = torch.cat(log_like_r).detach()

                    from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
                    sampler = BatchSampler(
                        SubsetRandomSampler(range(len(rol_act))),
                        100,  # 需要根据episodes设置
                        drop_last=True)
                    self.td['exchange'] = torch.zeros((batch_size, 2), dtype=torch.long)
                    for k in sampler:
                        # td需要根据k进行修改
                        temp_td = {}

                        temp_td['input_info'] = rol_input[k]  # sample_num,extend_problem, node_embed=7
                        temp_td['pos_enc'] = rol_pos[k]  # sample_num,extend_problem, embed=128
                        temp_td['action'] = rol_act[k]  # sample_num,1
                        temp_td['che_mask'] = rol_mask[k].flatten()  # 2340=组合数*sampe_num

                        bl_val, val = self.baseline.eval(td=temp_td,already_embed = True)
                        # 用新的policy计算ll
                        ll = self.policy(temp_td, policy_old=False)
                        # Finding the ratio (pi_theta / pi_theta__old):
                        ratios = torch.exp(ll['log_likelihood'] - log_like[k])

                        # Finding Surrogate Loss:
                        advantages = val_tru[k] - bl_val
                        surr1 = ratios * advantages
                        surr2 = torch.clamp(ratios, 1 - 0.2, 1 + 0.2) * advantages
                        loss = -torch.min(surr1, surr2).mean() + 0.5 * (val - val_tru[k]).pow(2).mean()
                        loss_list.append(loss)

                        opt = self.optimizers()
                        opt.zero_grad()
                        self.manual_backward(loss)
                        torch.nn.utils.clip_grad_norm_(
                            parameters=[p for group in opt.param_groups for p in group['params']],
                            max_norm=1.0,
                            norm_type=2
                        )
                        opt.step()

                    self.old_policy.load_state_dict(self.policy.state_dict())

            min_length =  -self.td['current_length']
            total_loss = torch.stack(loss_list)
            loss = torch.mean(total_loss)
            ret = {}
            ret['reward'] = min_length.unsqueeze(-1)
            ret['loss'] = loss
            return ret
        else:
            self.td = td
            length_current = []
            no_aug_batch_size = env.batch_size[0] // env.aug_factor
            max_steps = max_steps['test']
            if env.env_name == 'tsp':
                for T in range(max_steps):
                    self.td['input_info'], self.td['pos_enc'] = self.policy.get_input_and_pe(self.td)
                    self.td = self.policy(self.td)
                    self.td = tool.operate(self.td)
                    length_current.append(self.td['current_length'])
                    reward = torch.squeeze(self.td['next']['reward'])

                    if T % 100 == 0:
                        current_m = torch.stack(length_current, 0)
                        min = current_m.min(0)[0]
                        logger.info(f"T={T},length_current.mean={min.mean()}")

            elif env.env_name == 'cvrp':
                if tool.dic == None:
                    tool.cl, tool.dic = get_all_2opt(env.problem_size)
                for T in range(max_steps):
                    if T == 0:
                        self.td["che_mask"] = tool.get_2opt_mask_LIH(self.td['solution'], tool.dic, tool.cl)
                    self.td["input_info"], self.td["pos_enc"] = self.policy.get_input_and_pe(self.td)
                    self.td = self.policy(self.td, policy_old=True)
                    self.td = tool.operate(self.td)
                    length_current.append(self.td["current_length"])

                    if T % 100 == 0:
                        logger.info(f"T={T},length_current.mean={length_current[T].mean()}")
            best_solution = []
            current_m = torch.stack(length_current, 0)
            best_solution.append(current_m.min(0)[0])
            best = torch.cat(best_solution, 0)
            ret = {
                "no_aug_score": best[:no_aug_batch_size].mean(),
                "aug_score": best.mean(),
            }
            return ret


