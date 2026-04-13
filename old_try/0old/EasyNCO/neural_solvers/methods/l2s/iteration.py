from neural_solvers.pipeline import Iteration
from typing import Any, Tuple, Literal
from torchrl.envs import EnvBase
from tensordict import TensorDict
import torch
from EasyNCO.utils.utils import getLogger
from EasyNCO.neural_solvers.methods.l2s.utils import BatchGraph
import numpy as np
import networkx as nx
from EasyNCO.neural_solvers.methods.l2s.utils import Evaluator,CPM_batch_G
logger = getLogger(__name__)


class Iteration_tool():
    def __init__(self,initialization_out,device,reward_type, n_job, n_mch, low, high, fea_norm_const=1000, evaluator_type='message-passing'):
        self.itr = 0
        self.instances = initialization_out['instances']
        self.tabu_lists = [[] for _ in range(self.instances.shape[0])]
        self.current_graphs = initialization_out['current_graphs']
        self.sub_graphs_mc = initialization_out['sub_graphs_mc']
        self.current_objs = initialization_out['current_objs']
        self.incumbent_objs = initialization_out['incumbent_objs']
        self.adj_mat_pc = initialization_out['adj_mat_pc']
        self.device = device
        self.reward_type = reward_type
        self.n_job = n_job
        self.n_mch = n_mch
        self.low = low
        self.high = high
        self.tabu_size = 1

        self.feasible_actions, flag = self._feasible_actions()
        self.dones = ~flag
        self.fea_norm_const = fea_norm_const
        self.evaluator_type = evaluator_type
        self.eva = Evaluator() if evaluator_type == 'message-passing' else CPM_batch_G



    def change_nxgraph_topology(self, actions):
        n_jobs, n_machines = self.instances[0][0].shape
        n_operations = n_jobs * n_machines

        for i, (action, G, G_mc, instance) in enumerate(zip(actions, self.current_graphs, self.sub_graphs_mc, self.instances)):
            if action == [0, 0]:  # if dummy action then do not transit
                pass
            else:  # change nx graph topology
                S = [s for s in G.predecessors(action[0]) if int((s - 1) // n_machines) != int((action[0] - 1) // n_machines) and s != 0]
                T = [t for t in G.successors(action[1]) if int((t - 1) // n_machines) != int((action[1] - 1) // n_machines) and t != n_operations + 1]
                s = S[0] if len(S) != 0 else None
                t = T[0] if len(T) != 0 else None

                if s is not None:  # connect s with action[1]
                    G.remove_edge(s, action[0])
                    G.add_edge(s, action[1], weight=np.take(instance[0], s - 1))
                    G_mc.remove_edge(s, action[0])
                    G_mc.add_edge(s, action[1], weight=np.take(instance[0], s - 1))
                else:
                    pass

                if t is not None:  # connect action[0] with t
                    G.remove_edge(action[1], t)
                    G.add_edge(action[0], t, weight=np.take(instance[0], action[0] - 1))
                    G_mc.remove_edge(action[1], t)
                    G_mc.add_edge(action[0], t, weight=np.take(instance[0], action[0] - 1))
                else:
                    pass

                # reverse edge connecting selected pair
                G.remove_edge(action[0], action[1])
                G.add_edge(action[1], action[0], weight=np.take(instance[0], action[1] - 1))
                G_mc.remove_edge(action[0], action[1])
                G_mc.add_edge(action[1], action[0], weight=np.take(instance[0], action[1] - 1))

    @staticmethod
    def _get_pairs(cb, cb_op, tabu_list=None):
        pairs = []
        rg = cb[:-1].shape[0]  # sliding window of 2
        for i in range(rg):
            if cb[i] == cb[i + 1]:  # find potential pair
                if i == 0:
                    if cb[i + 1] != cb[i + 2]:
                        if [cb_op[i], cb_op[i + 1]] not in tabu_list:
                            pairs.append([cb_op[i], cb_op[i + 1]])
                elif cb[i] != cb[i - 1]:
                    if [cb_op[i], cb_op[i + 1]] not in tabu_list:
                        pairs.append([cb_op[i], cb_op[i + 1]])
                elif i + 1 == rg:
                    if cb[i + 1] != cb[i]:
                        if [cb_op[i], cb_op[i + 1]] not in tabu_list:
                            pairs.append([cb_op[i], cb_op[i + 1]])
                elif cb[i + 1] != cb[i + 2]:
                    if [cb_op[i], cb_op[i + 1]] not in tabu_list:
                        pairs.append([cb_op[i], cb_op[i + 1]])
                else:
                    pass
        return pairs

    def _gen_moves(self, solution, mch_mat, tabu_list=None):
        """
        solution: networkx DAG conjunctive graph
        mch_mat: the same mch from our NeurIPS 2020 paper of solution
        """
        critical_path = nx.dag_longest_path(solution)[1:-1]
        critical_blocks_opr = np.array(critical_path)
        critical_blocks = mch_mat.take(critical_blocks_opr - 1)  # -1: ops id starting from 0
        pairs = self._get_pairs(critical_blocks, critical_blocks_opr, tabu_list)
        return pairs

    def _feasible_actions(self):
        actions = []
        feasible_actions_flag = []  # False for no feasible operation pairs
        for i, (G, instance, tabu_list) in enumerate(zip(self.current_graphs, self.instances, self.tabu_lists)):
            action = self._gen_moves(solution=G, mch_mat=instance[1], tabu_list=tabu_list)
            # print(action)
            if len(action) != 0:
                actions.append(action)
                feasible_actions_flag.append(True)
            else:  # if no feasible actions available append dummy actions [0, 0]
                actions.append([[0, 0]])
                feasible_actions_flag.append(False)
        return actions, torch.tensor(feasible_actions_flag, device=self.device).unsqueeze(1)


    def dag2pyg(self, instances, nx_graphs, device):
        n_jobs, n_machines = instances[0][0].shape
        n_operations = n_jobs * n_machines

        edge_indices_pc = []
        edge_indices_mc = []
        durations = []
        for i, (instance, G_mc) in enumerate(zip(instances, nx_graphs)):
            durations.append(np.pad(instance[0].reshape(-1), (1, 1), 'constant', constant_values=0))
            adj_mat_mc = nx.adjacency_matrix(G_mc, weight=None).todense()
            edge_indices_pc.append((torch.nonzero(torch.from_numpy(self.adj_mat_pc)).t().contiguous()) + (n_operations + 2) * i)
            edge_indices_mc.append((torch.nonzero(torch.from_numpy(adj_mat_mc)).t().contiguous()) + (n_operations + 2) * i)

        edge_indices_pc = torch.cat(edge_indices_pc, dim=-1).to(device)
        edge_indices_mc = torch.cat(edge_indices_mc, dim=-1).to(device)
        durations = torch.from_numpy(np.concatenate(durations)).reshape(-1, 1).to(device)
        if self.evaluator_type == 'message-passing':
            est, lst, make_span = self.eva.forward(edge_index=torch.cat([edge_indices_pc, edge_indices_mc], dim=-1), duration=durations, n_j=self.n_job, n_m=self.n_mch)
        else:
            est, lst, make_span = self.eva(self.current_graphs, dev=device)
        # prepare x
        x = torch.cat([durations / self.high, est / self.fea_norm_const, lst / self.fea_norm_const], dim=-1)
        # prepare batch
        batch = torch.from_numpy(np.repeat(np.arange(instances.shape[0], dtype=np.int64), repeats=n_jobs * n_machines + 2)).to(device)

        return x, edge_indices_pc, edge_indices_mc, batch, make_span


    def operate(self,actions):
        self.change_nxgraph_topology(actions)  # change graph topology
        x, edge_indices_pc, edge_indices_mc, batch, makespan = self.dag2pyg(self.instances, self.sub_graphs_mc,
                                                                            self.device)  # generate new state data
        if self.reward_type == 'consecutive':
            reward = self.current_objs - makespan
        elif self.reward_type == 'l2s':
            reward = torch.where(self.incumbent_objs - makespan > 0, self.incumbent_objs - makespan,
                                 torch.tensor(0, dtype=torch.float32, device=self.device))
        else:
            raise ValueError('reward type must be "l2s" or "consecutive".')

        self.incumbent_objs = torch.where(makespan - self.incumbent_objs < 0, makespan, self.incumbent_objs)
        self.current_objs = makespan

        # update tabu list
        if self.tabu_size != 0:
            action_reversed = [a[::-1] for a in actions]
            for i, action in enumerate(action_reversed):
                if action == [0, 0]:  # if dummy action, don't update tabu list
                    pass
                else:
                    if len(self.tabu_lists[i]) == self.tabu_size:
                        self.tabu_lists[i].pop(0)
                        self.tabu_lists[i].append(action)
                    else:
                        self.tabu_lists[i].append(action)

        self.itr = self.itr + 1

        self.feasible_actions, flag = self._feasible_actions()  # new feasible actions w.r.t updated tabu list
        self.dones = ~flag
        return {
            'states': {
                'x': x,
                'edge_indices_pc': edge_indices_pc,
                'edge_indices_mc': edge_indices_mc,
                'batch': batch
            },
            'rewards':reward,
        }



class L2SIteration(Iteration):

    def learn(self, rewards, log_probs, dones, optimizer):
        R = torch.zeros_like(rewards[0], dtype=torch.float, device=rewards[0].device)
        returns = []
        for r in rewards[::-1]:
            R = r + 1 * R     #R = r + args.gamma * R (l2s set gamma=1)
            returns.insert(0, R)
        returns = torch.cat(returns, dim=-1)
        dones = torch.cat(dones, dim=-1)
        log_probs = torch.cat(log_probs, dim=-1)

        losses = []
        self.eps = np.finfo(np.float32).eps.item()
        for b in range(returns.shape[0]):
            masked_R = torch.masked_select(returns[b], ~dones[b])
            masked_R = (masked_R - masked_R.mean()) / (torch.std(masked_R, unbiased=False) + self.eps)
            masked_log_prob = torch.masked_select(log_probs[b], ~dones[b])
            loss = (- masked_log_prob * masked_R).sum()
            losses.append(loss)

        optimizer.zero_grad()
        mean_loss = torch.stack(losses).mean()
        self.manual_backward(mean_loss)
        optimizer.step()
        return mean_loss.detach()



    def run(self,
            td: TensorDict,
            env: EnvBase,
            initialization_out: dict,
            phase: Literal["train", "eval"],
            max_steps: int = 0,
            **kwargs,
            ) -> dict:

        batch_data = BatchGraph()
        states = initialization_out['states']
        reward_type = kwargs['reward_type']
        tool = Iteration_tool(initialization_out,env.device,reward_type,
                              n_job=env.num_job,n_mch=env.num_machine,low=env.time_low,high=env.time_high)


        if phase == "train":
            rewards_buffer = []
            log_probs_buffer = []
            dones_buffer = [tool.dones]
            n_step_return = kwargs['n_step_return']
            batch_loss =[]

            for i in range(max_steps):
                batch_data.wrapper(states)
                actions, log_ps = self.policy(batch_data, tool.feasible_actions)
                op_dict = tool.operate(actions)
                states = op_dict['states']
                reward = op_dict['rewards']
                # print(tool.incumbent_objs.mean())
                # store training data
                rewards_buffer.append(reward)
                log_probs_buffer.append(log_ps)
                dones_buffer.append(tool.dones)

                if (i+1) % n_step_return ==0:
                    optimizer = self.optimizers()
                    loss = self.learn(rewards_buffer, log_probs_buffer, dones_buffer[:-1], optimizer)
                    batch_loss.append(loss)
                    # clean training data
                    rewards_buffer = []
                    log_probs_buffer = []
                    dones_buffer = [tool.dones]
            ret ={}
            ret['reward'] = -tool.incumbent_objs
            ret['loss'] = torch.stack(batch_loss).mean()  #这里返回的loss形式有待商榷
            return ret
        else:
            for i in range(1,max_steps+1):
                batch_data.wrapper(states)
                actions, _ = self.policy(batch_data, tool.feasible_actions)
                op_dict = tool.operate(actions)
                states = op_dict['states']
                if i %10 ==0:
                    logger.info(f"step={i},makespan={tool.incumbent_objs.tolist()}",)

            result = tool.incumbent_objs.cpu().squeeze().numpy()

            out = {
                "no_aug_score": result.mean(),
                "aug_score": result.mean(),
            }

            return out
