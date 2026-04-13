import numpy as np
from torch_geometric.nn.conv import MessagePassing
import torch
from torch import Tensor
from typing import Union
from torch_geometric.typing import OptPairTensor, Adj, Size
import networkx as nx

#permissible_LS
def permissibleLeftShift(a, durMat, mchMat, mchsStartTimes, opIDsOnMchs):
    jobRdyTime_a, mchRdyTime_a = calJobAndMchRdyTimeOfa(a, mchMat, durMat, mchsStartTimes, opIDsOnMchs)
    dur_a = np.take(durMat, a)
    mch_a = np.take(mchMat, a) - 1
    startTimesForMchOfa = mchsStartTimes[mch_a]
    opsIDsForMchOfa = opIDsOnMchs[mch_a]
    flag = False

    possiblePos = np.where(jobRdyTime_a < startTimesForMchOfa)[0]
    # print('possiblePos:', possiblePos)
    if len(possiblePos) == 0:
        startTime_a = putInTheEnd(a, jobRdyTime_a, mchRdyTime_a, startTimesForMchOfa, opsIDsForMchOfa)
    else:
        idxLegalPos, legalPos, endTimesForPossiblePos = calLegalPos(dur_a, jobRdyTime_a, durMat, possiblePos, startTimesForMchOfa, opsIDsForMchOfa)
        # print('legalPos:', legalPos)
        if len(legalPos) == 0:
            startTime_a = putInTheEnd(a, jobRdyTime_a, mchRdyTime_a, startTimesForMchOfa, opsIDsForMchOfa)
        else:
            flag = True
            startTime_a = putInBetween(a, idxLegalPos, legalPos, endTimesForPossiblePos, startTimesForMchOfa, opsIDsForMchOfa)
    return startTime_a, flag


def putInTheEnd(a, jobRdyTime_a, mchRdyTime_a, startTimesForMchOfa, opsIDsForMchOfa):
    # index = first position of -config.high in startTimesForMchOfa
    # print('Yes!OK!')
    # index = np.where(startTimesForMchOfa == -args.h)[0][0]
    index = np.where(startTimesForMchOfa == -99)[0][0]
    startTime_a = max(jobRdyTime_a, mchRdyTime_a)
    startTimesForMchOfa[index] = startTime_a
    opsIDsForMchOfa[index] = a
    return startTime_a


def calLegalPos(dur_a, jobRdyTime_a, durMat, possiblePos, startTimesForMchOfa, opsIDsForMchOfa):
    startTimesOfPossiblePos = startTimesForMchOfa[possiblePos]
    durOfPossiblePos = np.take(durMat, opsIDsForMchOfa[possiblePos])
    startTimeEarlst = max(jobRdyTime_a, startTimesForMchOfa[possiblePos[0]-1] + np.take(durMat, [opsIDsForMchOfa[possiblePos[0]-1]]))
    endTimesForPossiblePos = np.append(startTimeEarlst, (startTimesOfPossiblePos + durOfPossiblePos))[:-1]# end time for last ops don't care
    possibleGaps = startTimesOfPossiblePos - endTimesForPossiblePos
    idxLegalPos = np.where(dur_a <= possibleGaps)[0]
    legalPos = np.take(possiblePos, idxLegalPos)
    return idxLegalPos, legalPos, endTimesForPossiblePos


def putInBetween(a, idxLegalPos, legalPos, endTimesForPossiblePos, startTimesForMchOfa, opsIDsForMchOfa):
    earlstIdx = idxLegalPos[0]
    # print('idxLegalPos:', idxLegalPos)
    earlstPos = legalPos[0]
    startTime_a = endTimesForPossiblePos[earlstIdx]
    # print('endTimesForPossiblePos:', endTimesForPossiblePos)
    startTimesForMchOfa[:] = np.insert(startTimesForMchOfa, earlstPos, startTime_a)[:-1]
    opsIDsForMchOfa[:] = np.insert(opsIDsForMchOfa, earlstPos, a)[:-1]
    return startTime_a


def calJobAndMchRdyTimeOfa(a, mchMat, durMat, mchsStartTimes, opIDsOnMchs):
    mch_a = np.take(mchMat, a) - 1
    # cal jobRdyTime_a
    jobPredecessor = a - 1 if a % mchMat.shape[1] != 0 else None
    if jobPredecessor is not None:
        durJobPredecessor = np.take(durMat, jobPredecessor)
        mchJobPredecessor = np.take(mchMat, jobPredecessor) - 1
        jobRdyTime_a = (mchsStartTimes[mchJobPredecessor][np.where(opIDsOnMchs[mchJobPredecessor] == jobPredecessor)] + durJobPredecessor).item()
    else:
        jobRdyTime_a = 0
    # cal mchRdyTime_a
    mchPredecessor = opIDsOnMchs[mch_a][np.where(opIDsOnMchs[mch_a] >= 0)][-1] if len(np.where(opIDsOnMchs[mch_a] >= 0)[0]) != 0 else None
    if mchPredecessor is not None:
        durMchPredecessor = np.take(durMat, mchPredecessor)
        mchRdyTime_a = (mchsStartTimes[mch_a][np.where(mchsStartTimes[mch_a] >= 0)][-1] + durMchPredecessor).item()
    else:
        mchRdyTime_a = 0

    return jobRdyTime_a, mchRdyTime_a

#message_passing
class ForwardPass(MessagePassing):
    def __init__(self, **kwargs):
        kwargs.setdefault('aggr', 'max')
        super(ForwardPass, self).__init__(**kwargs)

    def forward(self,
                x: Union[Tensor, OptPairTensor],
                edge_index: Adj,
                size: Size = None) -> Tensor:
        """"""
        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)
        # propagate_type: (x: OptPairTensor)
        out = self.propagate(edge_index, x=x, size=size)
        return out


class BackwardPass(MessagePassing):
    def __init__(self, **kwargs):
        kwargs.setdefault('aggr', 'max')
        super(BackwardPass, self).__init__(**kwargs)

    def forward(self,
                x: Union[Tensor, OptPairTensor],
                edge_index: Adj,
                size: Size = None) -> Tensor:
        """"""
        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)
        out = self.propagate(edge_index, x=x, size=size)
        return out


class Evaluator:
    def __init__(self):
        self.forward_pass = ForwardPass(aggr='max', flow="source_to_target")
        self.backward_pass = BackwardPass(aggr='max', flow="target_to_source")

    def forward(self, edge_index, duration, n_j, n_m):
        """
        support batch version
        edge_index: [2, n_edges] tensor
        duration: [n_nodes, 1] tensor
        """
        n_nodes = duration.shape[0]
        n_nodes_each_graph = n_j * n_m + 2
        device = edge_index.device

        # forward pass...
        index_S = np.arange(n_nodes // n_nodes_each_graph, dtype=int) * n_nodes_each_graph
        earliest_start_time = torch.zeros_like(duration, dtype=torch.float32, device=device)
        mask_earliest_start_time = torch.ones_like(duration, dtype=torch.int8, device=device)
        mask_earliest_start_time[index_S] = 0
        for _ in range(n_nodes):
            if mask_earliest_start_time.sum() == 0:
                break
            x_forward = duration + earliest_start_time.masked_fill(mask_earliest_start_time.bool(), 0)
            earliest_start_time = self.forward_pass(x=x_forward, edge_index=edge_index)
            mask_earliest_start_time = self.forward_pass(x=mask_earliest_start_time, edge_index=edge_index)

        # backward pass...
        index_T = np.cumsum(np.ones(shape=[n_nodes // n_nodes_each_graph], dtype=int) * n_nodes_each_graph) - 1
        make_span = earliest_start_time[index_T]
        # latest_start_time = torch.zeros_like(duration, dtype=torch.float32, device=device)
        latest_start_time = - torch.ones_like(duration, dtype=torch.float32, device=device)
        latest_start_time[index_T] = - make_span
        mask_latest_start_time = torch.ones_like(duration, dtype=torch.int8, device=device)
        mask_latest_start_time[index_T] = 0
        for _ in range(n_nodes):
            if mask_latest_start_time.sum() == 0:
                break
            x_backward = latest_start_time.masked_fill(mask_latest_start_time.bool(), 0)
            latest_start_time = self.backward_pass(x=x_backward, edge_index=edge_index) + duration
            latest_start_time[index_T] = - make_span
            mask_latest_start_time = self.backward_pass(x=mask_latest_start_time, edge_index=edge_index)

        return earliest_start_time, torch.abs(latest_start_time), make_span


def processing_order_to_edge_index(order, instance):
    """
    order: [n_m, n_j] a numpy array specifying the processing order on each machine, each row is a machine
    instance: [1, n_j, n_m] an instance as numpy array
    RETURN: edge index: [2, n_j * n_m +2] tensor for the directed disjunctive graph
    """
    dur, mch = instance[0], instance[1]
    n_j, n_m = dur.shape[0], dur.shape[1]
    n_opr = n_j*n_m

    adj = np.eye(n_opr, k=-1, dtype=int)  # Create adjacent matrix for precedence constraints
    adj[np.arange(start=0, stop=n_opr, step=1).reshape(n_j, -1)[:, 0]] = 0  # first column does not have upper stream conj_nei
    adj = np.pad(adj, 1, 'constant', constant_values=0)  # pad dummy S and T nodes
    adj[[i for i in range(1, n_opr + 2 - 1, n_m)], 0] = 1  # connect S with 1st operation of each job
    adj[-1, [i for i in range(n_m, n_opr + 2 - 1, n_m)]] = 1  # connect last operation of each job to T
    adj = np.transpose(adj)

    # rollout ortools solution
    steps_basedon_sol = []
    for i in range(n_m):
        get_col_position_unsorted = np.argwhere(mch == (i + 1))
        get_col_position_sorted = get_col_position_unsorted[order[i]]
        sol_i = order[i] * n_m + get_col_position_sorted[:, 1]
        steps_basedon_sol.append(sol_i.tolist())

    for operations in steps_basedon_sol:
        for i in range(len(operations) - 1):
            adj[operations[i]+1][operations[i+1]+1] += 1

    return torch.nonzero(torch.from_numpy(adj)).t().contiguous()


def forward_pass(graph, topological_order=None):  # graph is a nx.DiGraph;
    # assert (graph.in_degree(topological_order[0]) == 0)
    earliest_ST = dict.fromkeys(graph.nodes, -float('inf'))
    if topological_order is None:
        topo_order = list(nx.topological_sort(graph))
    else:
        topo_order = topological_order
    earliest_ST[topo_order[0]] = 0.
    for n in topo_order:
        for s in graph.successors(n):
            if earliest_ST[s] < earliest_ST[n] + graph.edges[n, s]['weight']:
                earliest_ST[s] = earliest_ST[n] + graph.edges[n, s]['weight']
    # return is a dict where key is each node's ID, value is the length from source node s
    return earliest_ST


def backward_pass(graph, makespan, topological_order=None):
    if topological_order is None:
        reverse_order = list(reversed(list(nx.topological_sort(graph))))
    else:
        reverse_order = list(reversed(topological_order))
    latest_ST = dict.fromkeys(graph.nodes, float('inf'))
    latest_ST[reverse_order[0]] = float(makespan)
    for n in reverse_order:
        for p in graph.predecessors(n):
            if latest_ST[p] > latest_ST[n] - graph.edges[p, n]['weight']:
                # assert latest_ST[n] - graph.edges[p, n]['weight'] >= 0, 'latest start times should is negative, BUG!'  # latest start times should be non-negative
                latest_ST[p] = latest_ST[n] - graph.edges[p, n]['weight']
    return latest_ST


def forward_and_backward_pass(G):
    # calculate topological order
    topological_order = list(nx.topological_sort(G))
    # forward and backward pass
    est = np.fromiter(forward_pass(graph=G, topological_order=topological_order).values(), dtype=np.float32)
    lst = np.fromiter(backward_pass(graph=G, topological_order=topological_order, makespan=est[-1]).values(), dtype=np.float32)
    # assert np.where(est > lst)[0].shape[0] == 0, 'latest starting time is smaller than earliest starting time, bug!'  # latest starting time should be larger or equal to earliest starting time
    return est, lst, est[-1]


def CPM_batch_G(Gs, dev):
    multi_est = []
    multi_lst = []
    multi_makespan = []
    for G in Gs:
        est, lst, makespan = forward_and_backward_pass(G)
        multi_est.append(est)
        multi_lst.append(lst)
        multi_makespan.append([makespan])
    multi_est = torch.from_numpy(np.concatenate(multi_est, axis=0)).view(-1, 1).to(dev)
    multi_lst = torch.from_numpy(np.concatenate(multi_lst, axis=0)).view(-1, 1).to(dev)
    multi_makespan = torch.tensor(multi_makespan, device=dev)
    return multi_est, multi_lst, multi_makespan


class BatchGraph:
    def __init__(self):
        self.x = None
        self.edge_index_pc = None
        self.edge_index_mc = None
        self.batch = None

    def wrapper(self, states):
        self.x = states['x']
        self.edge_index_pc = states['edge_indices_pc']
        self.edge_index_mc = states['edge_indices_mc']
        self.batch = states['batch']

    def clean(self):
        self.x = None
        self.edge_index_pc = None
        self.edge_index_mc = None
        self.batch = None

