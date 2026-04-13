import numpy as np
from EasyNCO.neural_solvers.methods.l2s.utils import permissibleLeftShift,Evaluator,CPM_batch_G
import torch
from neural_solvers.pipeline.initialization import Initialization
from typing import Any, Tuple, Literal
from tensordict import TensorDict
from torchrl.envs import EnvBase
import networkx as nx

class Initialization_tool():
    def __init__(self, n_job, n_mch, low, high, fea_norm_const=1000, evaluator_type='message-passing'):
        self.n_job = n_job
        self.n_mch = n_mch
        self.n_oprs = self.n_job * self.n_mch
        self.low = low
        self.high = high
        # self.itr = 0
        self.instances = None
        self.sub_graphs_mc = None
        self.current_graphs = None
        self.current_objs = None
        self.tabu_size = 1
        self.tabu_lists = None
        self.incumbent_objs = None
        self.fea_norm_const = fea_norm_const
        self.evaluator_type = evaluator_type
        self.eva = Evaluator() if evaluator_type == 'message-passing' else CPM_batch_G
        self.adj_mat_pc = self._adj_mat_pc()

    def _p_list_solver(self, args):
        instances, priority_lists, device = args[0], args[1], args[2]

        edge_indices_pc = []
        edge_indices_mc = []
        durations = []
        current_graphs = []
        sub_graphs_mc = []
        for i, (instance, priority_list) in enumerate(zip(instances, priority_lists)):
            dur_mat, mch_mat = instance[0], instance[1]
            n_jobs = mch_mat.shape[0]
            n_machines = mch_mat.shape[1]
            n_operations = n_jobs * n_machines

            # prepare NIPS adj
            ops_mat = np.arange(0, n_operations).reshape(mch_mat.shape).tolist()  # Init operations mat
            list_for_latest_task_onMachine = [None] * n_machines  # Init list_for_latest_task_onMachine

            adj_mat_mc = np.zeros(shape=[n_operations, n_operations], dtype=int)  # Create adjacent matrix for machine clique
            # Construct NIPS adjacent matrix only for machine cliques
            for job_id in priority_list:
                op_id = ops_mat[job_id][0]
                m_id_for_action = mch_mat[op_id // n_machines, op_id % n_machines] - 1
                if list_for_latest_task_onMachine[m_id_for_action] is not None:
                    adj_mat_mc[op_id, list_for_latest_task_onMachine[m_id_for_action]] = 1
                list_for_latest_task_onMachine[m_id_for_action] = op_id
                ops_mat[job_id].pop(0)
            adj_mat_mc = np.pad(adj_mat_mc, ((1, 1), (1, 1)), 'constant', constant_values=0)  # add S and T to machine clique adj
            adj_mat_mc = np.transpose(adj_mat_mc)  # convert input adj from column pointing to row, to, row pointing to column

            adj_all = self.adj_mat_pc + adj_mat_mc
            dur_mat = np.pad(dur_mat.reshape(-1, 1), ((1, 1), (0, 0)), 'constant', constant_values=0).repeat(self.n_oprs + 2, axis=1)
            edge_weight = np.multiply(adj_all, dur_mat)
            G = nx.from_numpy_array(edge_weight, parallel_edges=False, create_using=nx.DiGraph)  # create nx.DiGraph
            G.add_weighted_edges_from([(0, i, 0) for i in range(1, self.n_oprs + 2 - 1, self.n_mch)])  # add release time, here all jobs are available at t=0. This is the only way to add release date. And if you do not add release date, startime computation will return wired value
            current_graphs.append(G)
            G_mc = nx.from_numpy_array(adj_mat_mc, parallel_edges=False, create_using=nx.DiGraph)  # create nx.DiGraph
            sub_graphs_mc.append(G_mc)

            edge_indices_pc.append((torch.nonzero(torch.from_numpy(self.adj_mat_pc)).t().contiguous()) + (n_operations + 2) * i)
            edge_indices_mc.append((torch.nonzero(torch.from_numpy(adj_mat_mc)).t().contiguous()) + (n_operations + 2) * i)
            durations.append(torch.from_numpy(dur_mat[:, 0]).to(device))

        edge_indices_pc = torch.cat(edge_indices_pc, dim=-1).to(device)
        edge_indices_mc = torch.cat(edge_indices_mc, dim=-1).to(device)

        durations = torch.cat(durations, dim=0).reshape(-1, 1)
        if self.evaluator_type == 'message-passing':
            est, lst, make_span = self.eva.forward(edge_index=torch.cat([edge_indices_pc, edge_indices_mc], dim=-1), duration=durations, n_j=self.n_job, n_m=self.n_mch)
        else:
            est, lst, make_span = self.eva(current_graphs, dev=device)

            # prepare x
        x = torch.cat([durations / self.high, est / self.fea_norm_const, lst / self.fea_norm_const], dim=-1)
        # prepare batch
        batch = torch.from_numpy(np.repeat(np.arange(instances.shape[0], dtype=np.int64), repeats=self.n_job * self.n_mch + 2)).to(device)

        return (x, edge_indices_pc, edge_indices_mc, batch), current_graphs, sub_graphs_mc, make_span

    def _rules_solver(self, args):
        instances, device, rule_type = args[0], args[1], args[2]

        edge_indices_pc = []
        edge_indices_mc = []
        durations = []
        current_graphs = []
        sub_graphs_mc = []
        for i, instance in enumerate(instances):
            dur_mat, dur_cp, mch_mat = instance[0], np.copy(instance[0]), instance[1]
            n_jobs, n_machines = dur_mat.shape[0], dur_mat.shape[1]
            n_operations = n_jobs * n_machines
            last_col = np.arange(start=0, stop=n_operations, step=1).reshape(n_jobs, -1)[:, -1]
            candidate_oprs = np.arange(start=0, stop=n_operations, step=1).reshape(n_jobs, -1)[:,0]  # initialize action space: [n_jobs, 1], the first column
            mask = np.zeros(shape=n_jobs, dtype=bool)  # initialize the mask: [n_jobs, 1]
            adj_mat_mc = np.zeros(shape=[n_operations, n_operations], dtype=int)  # Create adjacent matrix for machine clique

            gant_chart = -self.high * np.ones_like(dur_mat.transpose(), dtype=np.int32)
            opIDsOnMchs = -n_jobs * np.ones_like(dur_mat.transpose(), dtype=np.int32)
            finished_mark = np.zeros_like(mch_mat, dtype=np.int32)

            actions = []
            for _ in range(n_operations):

                if rule_type == 'spt':
                    candidate_masked = candidate_oprs[np.where(~mask)]
                    dur_candidate = np.take(dur_mat, candidate_masked)
                    idx = np.random.choice(np.where(dur_candidate == np.min(dur_candidate))[0])
                    action = candidate_masked[idx]
                elif rule_type == 'fdd-divide-mwkr':
                    candidate_masked = candidate_oprs[np.where(~mask)]
                    fdd = np.take(np.cumsum(dur_mat, axis=1), candidate_masked)
                    wkr = np.take(np.cumsum(np.multiply(dur_mat, 1 - finished_mark), axis=1), last_col[np.where(~mask)])
                    priority = fdd / wkr
                    idx = np.random.choice(np.where(priority == np.min(priority))[0])
                    action = candidate_masked[idx]
                else:
                    action = None
                actions.append(action)

                permissibleLeftShift(a=action, durMat=dur_mat, mchMat=mch_mat, mchsStartTimes=gant_chart,
                                     opIDsOnMchs=opIDsOnMchs)

                # update action space or mask
                if action not in last_col:
                    candidate_oprs[action // n_machines] += 1
                else:
                    mask[action // n_machines] = 1
                # update finished_mark:
                finished_mark[action // n_machines, action % n_machines] = 1

            for _ in range(opIDsOnMchs.shape[1] - 1):
                adj_mat_mc[opIDsOnMchs[:, _ + 1], opIDsOnMchs[:, _]] = 1

            # prepare augmented adj, augmented dur, and G
            adj_mat_mc = np.pad(adj_mat_mc, ((1, 1), (1, 1)), 'constant', constant_values=0)  # add S and T to machine clique adj
            adj_mat_mc = np.transpose(adj_mat_mc)  # convert input adj from column pointing to row, to, row pointing to column
            adj_all = self.adj_mat_pc + adj_mat_mc
            dur_mat = np.pad(dur_mat.reshape(-1, 1), ((1, 1), (0, 0)), 'constant', constant_values=0).repeat(self.n_oprs + 2, axis=1)
            edge_weight = np.multiply(adj_all, dur_mat)
            G = nx.from_numpy_array(edge_weight, parallel_edges=False, create_using=nx.DiGraph)  # create nx.DiGraph
            G.add_weighted_edges_from([(0, i, 0) for i in range(1, self.n_oprs + 2 - 1, self.n_mch)])  # add release time, here all jobs are available at t=0. This is the only way to add release date. And if you do not add release date, startime computation will return wired value
            current_graphs.append(G)
            G_mc = nx.from_numpy_array(adj_mat_mc, parallel_edges=False, create_using=nx.DiGraph)  # create nx.DiGraph
            sub_graphs_mc.append(G_mc)

            edge_indices_pc.append((torch.nonzero(torch.from_numpy(self.adj_mat_pc)).t().contiguous()) + (n_operations + 2) * i)
            edge_indices_mc.append((torch.nonzero(torch.from_numpy(adj_mat_mc)).t().contiguous()) + (n_operations + 2) * i)
            durations.append(torch.from_numpy(dur_mat[:, 0]).to(device))

        edge_indices_pc = torch.cat(edge_indices_pc, dim=-1).to(device)
        edge_indices_mc = torch.cat(edge_indices_mc, dim=-1).to(device)
        durations = torch.cat(durations, dim=0).reshape(-1, 1)
        if self.evaluator_type == 'message-passing':
            est, lst, make_span = self.eva.forward(edge_index=torch.cat([edge_indices_pc, edge_indices_mc], dim=-1), duration=durations, n_j=self.n_job, n_m=self.n_mch)
        else:
            est, lst, make_span = self.eva(current_graphs, dev=device)

        # prepare x
        x = torch.cat([durations / self.high, est / self.fea_norm_const, lst / self.fea_norm_const], dim=-1)
        # prepare batch
        batch = torch.from_numpy(np.repeat(np.arange(instances.shape[0], dtype=np.int64), repeats=self.n_job * self.n_mch + 2)).to(device)

        return (x, edge_indices_pc, edge_indices_mc, batch), current_graphs, sub_graphs_mc, make_span

    def _adj_mat_pc(self):
        adj_mat_pc = np.eye(self.n_oprs, k=-1, dtype=int)  # Create adjacent matrix for precedence constraints
        adj_mat_pc[np.arange(start=0, stop=self.n_oprs, step=1).reshape(self.n_job, -1)[:, 0]] = 0  # first column does not have upper stream conj_nei
        adj_mat_pc = np.pad(adj_mat_pc, 1, 'constant', constant_values=0)  # pad dummy S and T nodes
        adj_mat_pc[[i for i in range(1, self.n_job * self.n_mch + 2 - 1, self.n_mch)], 0] = 1  # connect S with 1st operation of each job
        adj_mat_pc[-1, [i for i in range(self.n_mch, self.n_job * self.n_mch + 2 - 1, self.n_mch)]] = 1  # connect last operation of each job to T
        adj_mat_pc = np.transpose(adj_mat_pc)  # convert input adj from column pointing to row, to, row pointing to column
        return adj_mat_pc

    def initial(self, instances, init_type, device):
        # self.instances_tensor_version = instances  #tensor_version
        self.instances = instances.cpu().numpy() #numpy_version

        if init_type == 'plist':
            plist = np.repeat(np.random.permutation(np.arange(self.n_job).repeat(self.n_mch)).reshape(1, -1), repeats=self.instances.shape[0], axis=0)  # fixed random plist for all instances
            (x, edge_indices_pc, edge_indices_mc, batch), current_graphs, sub_graphs_mc, make_span = self._p_list_solver(args=[self.instances, plist, device])
        elif init_type == 'spt':
            (x, edge_indices_pc, edge_indices_mc, batch), current_graphs, sub_graphs_mc, make_span = self._rules_solver(args=[self.instances, device, 'spt'])
        elif init_type == 'fdd-divide-mwkr':
            (x, edge_indices_pc, edge_indices_mc, batch), current_graphs, sub_graphs_mc, make_span = self._rules_solver(args=[self.instances, device, 'fdd-divide-mwkr'])
        else:
            assert False, 'Initial solution type = "plist", "spt", "fdd-divide-mwkr".'

        self.sub_graphs_mc = sub_graphs_mc
        self.current_graphs = current_graphs
        self.current_objs = make_span
        self.incumbent_objs = make_span

        return {
            'states': {
                'x': x,
                'edge_indices_pc': edge_indices_pc,
                'edge_indices_mc': edge_indices_mc,
                'batch': batch
            },
            'sub_graphs_mc': self.sub_graphs_mc,
            'current_graphs':self.current_graphs,
            'current_objs': self.current_objs,
            'incumbent_objs': self.incumbent_objs,
            'instances': self.instances,
            'adj_mat_pc':self.adj_mat_pc
        }





class L2SInitialization(Initialization):
    def run(self,
            env: EnvBase,
            batch: int,
            strategy: str,
            phase: Literal["train", "eval"],
            **kwargs,
    ) -> Tuple[TensorDict, Any]:

        init_type = kwargs.get('init_type', 'fdd-divide-mwkr')
        if phase == "train":
            self.policy.train()
            env.load_problems(batch, batch.size(0)) # batch,2,job,machine  (2 is time,machine)
            tool = Initialization_tool(env.num_job,env.num_machine,low=env.time_low,high=env.time_high)
            reset_td = tool.initial(env.problems,init_type=init_type,device=env.device)
        else:
            self.policy.train() # l2s uses train_mode to test, what can i say
            env.load_problems(batch, batch.size(0))
            tool = Initialization_tool(env.num_job,env.num_machine,low=env.time_low,high=env.time_high)
            reset_td = tool.initial(env.problems,init_type=init_type,device=env.device)


        return reset_td, reset_td