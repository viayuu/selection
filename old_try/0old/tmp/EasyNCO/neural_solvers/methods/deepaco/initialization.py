from EasyNCO.neural_solvers.methods.deepaco.aco_bpp import ACO_BPP
from EasyNCO.neural_solvers.methods.deepaco.aco_mkp import ACO_MKP
from EasyNCO.neural_solvers.methods.deepaco.aco_mkp_transformer import ACO_MKP_Transformer
from EasyNCO.neural_solvers.methods.deepaco.aco_rcpsp import ACO_RCPSP, Activity, RCPSPInstance
from EasyNCO.neural_solvers.methods.deepaco.aco_tsp import ACO_TSP
from EasyNCO.neural_solvers.methods.deepaco.aco_cvrp import ACO_CVRP
from EasyNCO.neural_solvers.methods.deepaco.aco_op import ACO_OP
from EasyNCO.neural_solvers.methods.deepaco.aco_pctsp import ACO_PCTSP
from EasyNCO.neural_solvers.methods.deepaco.aco_sop import ACO_SOP
from EasyNCO.neural_solvers.methods.deepaco.aco_smtwtp import ACO_SMTWTP
from EasyNCO.neural_solvers.backbones.GNN.DATATransform import *
from EasyNCO.neural_solvers.pipeline import Initialization
from typing import Any, Tuple, Literal
from torchrl.envs import EnvBase
from tensordict import TensorDict
import torch

class DeepACOInitialization(Initialization):
    """
    A class for DeepACO initialization that extends the Initialization class.
    It is used to create solutions for training in the DeepACO framework.
    """
    def run(self,
            env: EnvBase,
            batch: int,
            strategy: str,
            phase: Literal["train", "eval"],
            **kwargs,
    ) -> Tuple[TensorDict, Any]:
        """
        This function is used to create initial solution for training.
        It simply loads the problems into the environment and plays an episode
        using the provided policy and strategy.
        args:
            - policy: The policy network to be used.
            - env: The environment to be used.
            - batch: The batch size.
            - strategy: The strategy to decode the output of the policy network.
            - mode: The mode of the initialization, can be 'train' or 'eval'.
        returns:
            - state_td: The final state of the environment.
            - out: The output of the policy network or the scores for evaluation.
        """
        env.env_batch_size = batch.size(0)
        state_td = None

        out = {
            "batch": batch,
        }

        return state_td, out

def load_policy(model, data, policy_param, env_name):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if env_name == "tsp":
        EPS = 1e-10
        k_sparse = policy_param['k_sparse'] or len(data) // 10
        pyg_data, distances = gen_pyg_data_tsp(data, k_sparse, start_node=0)
        heu_mat = model.reshape(pyg_data, model(pyg_data)) + EPS
        aco = ACO_TSP(
            distances=distances,
            heuristic=heu_mat,
            device=device,
            n_ants=policy_param['n_ants'],
            decay=policy_param['decay'],
            alpha=policy_param['alpha'],
            beta=policy_param['beta'],
            elitist=policy_param['elitist'],
            min_max=policy_param['min_max'],
            min=policy_param['min'],
            two_opt=policy_param['two_opt'],
            local_search=policy_param['local_search']
        )

    elif env_name == 'cvrp':
        EPS = 1e-10

        position = data[:, :2]
        demand = data[:, -1]
        distances = gen_distance_matrix(position, infinity=False)
        pyg_data = gen_pyg_data_cvrp(demand, distances, device, k_sparse=max(len(data)//5, 4))
        heu_vec = model(pyg_data)
        heu_mat = model.reshape(pyg_data, heu_vec) + EPS

        aco = ACO_CVRP(
            distances=distances,
            demand=demand,
            heuristic=heu_mat,
            positions=position,
            device=device,
            n_ants=policy_param['n_ants'],
            decay=policy_param['decay'],
            alpha=policy_param['alpha'],
            beta=policy_param['beta'],
            elitist=policy_param['elitist'],
            min_max=policy_param['min_max'],
            min=policy_param['min'],
            adaptive=policy_param['adaptive'],
            swapstar=policy_param['swapstar']
        )

    elif env_name == 'op':
        EPS = 1e-10
        n = len(data)
        position = data[:, :2]
        prizes = data[:, -1]
        sparse_table = {
            100: 20,
            200: 50,
            300: 50
        }
        k_sparse = sparse_table[n]
        pyg_data, distances = gen_pyg_data_op(position, prizes, k_sparse)
        heu_mat = model.reshape(pyg_data, model(pyg_data)) + EPS
        max_len = {
            100: 4,
            200: 5,
            300: 6
        }
        aco = ACO_OP(
            distances=distances,
            prizes=prizes,
            heuristic=heu_mat,
            device=device,
            max_len=max_len[n],
            n_ants=policy_param['n_ants'],
            decay=policy_param['decay'],
            alpha=policy_param['alpha'],
            beta=policy_param['beta'],
            elitist=policy_param['elitist'],
            min_max=policy_param['min_max'],
            min=policy_param['min'],
        )

    elif env_name == 'pctsp':
        EPS = 1e-10

        nodes_pctsp = data[:, :2]
        distances = torch.norm(nodes_pctsp[:, None] - nodes_pctsp, dim=2, p=2)
        prizes = data[:, -2]
        penalties = data[:, -1]
        pyg_data = gen_pyg_data_pctsp(prizes, penalties, distances)
        heu_mat = model(pyg_data)
        heu_mat = (heu_mat / (heu_mat.min() + EPS) + EPS).reshape(len(data) , len(data))
        aco = ACO_PCTSP(
            distances=distances,
            prizes=prizes,
            penalties=penalties,
            heuristic=heu_mat,
            device=device,
            n_ants=policy_param['n_ants'],
            decay=policy_param['decay'],
            alpha=policy_param['alpha'],
            beta=policy_param['beta'],
            elitist=policy_param['elitist'],
            min_max=policy_param['min_max'],
            min=policy_param['min'],
        )

    elif env_name == 'sop':
        EPS = 1e-10

        distances = data[: int(len(data)/2)]
        adj_mat = data[int(len(data)/2) :]

        prec_cons = 1 - adj_mat
        prec_cons[torch.arange(int(len(data)/2)), torch.arange(int(len(data)/2))] = 0

        pyg_data = gen_pyg_data_sop(distances, adj_mat)
        heu_vec = model(pyg_data)
        heu_mat = model.reshape(pyg_data, heu_vec) + EPS

        aco = ACO_SOP(
            distances=distances,
            prec_cons=prec_cons,
            heuristic=heu_mat,
            device=device,
            n_ants=policy_param['n_ants'],
            decay=policy_param['decay'],
            alpha=policy_param['alpha'],
            beta=policy_param['beta'],
            elitist=policy_param['elitist'],
            min_max=policy_param['min_max'],
            min=policy_param['min'],
        )

    elif env_name == 'smtwtp':
        EPS = 1e-10

        due_time = data[:, 0]
        weights = data[:, 1]
        processing_time = data[:, 2]

        pyg_data = gen_pyg_data_smtwtp(due_time, weights, processing_time, device)
        heu_vec = model(pyg_data)
        heu_mat = model.reshape(pyg_data, heu_vec) + EPS

        aco = ACO_SMTWTP(
            due_time=due_time,
            weights=weights,
            processing_time=processing_time,
            heuristic=heu_mat,
            device=device,
            n_ants=policy_param['n_ants'],
            decay=policy_param['decay'],
            alpha=policy_param['alpha'],
            beta=policy_param['beta'],
            elitist=policy_param['elitist'],
            min_max=policy_param['min_max'],
            min=policy_param['min'],
        )

    elif env_name == 'rcpsp':
        # The following codes are used to convert the data from the dataset into the RCPSPInstance class.
        # RCPSPInstance denotes an instance of the RCPSP problem, which includes jobs and capacities of resources.
        # Jobs are denoted by Activity, including the duration, the demands of corresponding resources and the order of jobs.
        EPS = 1e-10
        n_jobs = data.size(0) - 1
        n_resources = data.size(1) - n_jobs - 2
        duration = data[1:, 0].T
        resource_capacity = data[0, 1:n_resources+1].tolist()
        resource = data[1:, 1:n_resources+1]
        succ_mat = data[1:, n_resources+2:]
        nodes = [Activity(i) for i in range(n_jobs)]
        for index, act in enumerate(nodes):
            act.duration = duration[index].item()
            act.resources = [resource[index][j].item() for j in range(n_resources)]
            successors = succ_mat[index, :]
            for k, value in enumerate(successors):
                if value == 1:
                    successor = nodes[k]
                    act.add_successor(successor)

        instance = RCPSPInstance(nodes, resource_capacity)
        pyg_data = instance.to_pyg_data(device=device)
        heu_vec = model(pyg_data)
        heu_mat = model.reshape(pyg_data, heu_vec) + EPS

        aco = ACO_RCPSP(
            rcpsp=instance,
            heuristic=heu_mat,
            device=device,
            n_ants=policy_param['n_ants'],
            elitist=policy_param['elitist'],
            min_max=policy_param['min_max'],
        )

    elif env_name == 'mkp':
        EPS = 1e-10
        # The shape of data from MKPGenerator is (n, m+1), and constraints are all 1.
        # Note that for transformer, constraints are all 1; for GNN, constraints are all n//2.
        if policy_param['model_name'] == 'transformer':
            prize = data[:, 0].T # (n, )
            weight = data[:, 1:].T # (m, n)
            src = reformat(prize, weight)
            heu_vec = model(src) + EPS

            aco = ACO_MKP_Transformer(
                price=prize,
                weight=weight,
                heuristic=heu_vec,
                device=device,
                n_ants=policy_param['n_ants'],
                decay=policy_param['decay'],
                alpha=policy_param['alpha'],
                beta=policy_param['beta'],
                elitist=policy_param['elitist'],
            )
        else:
            prize = data[:, 0] # (n, )
            weight = data[:, 1:] * (len(data)//2) # (n, m)
            src = gen_pyg_data_mkp(prize, weight)
            heu_mat = model(src).reshape((prize.size(0), prize.size(0)))
            heu_mat = heu_mat / (heu_mat.min() + EPS) + EPS

            aco = ACO_MKP(
                prize=prize,
                weight=weight,
                heuristic=heu_mat,
                device=device,
                n_ants=policy_param['n_ants'],
                decay=policy_param['decay'],
                alpha=policy_param['alpha'],
                beta=policy_param['beta'],
                elitist=policy_param['elitist'],
            )


    elif env_name == 'bpp':
        EPS = 1e-10

        pyg_data = gen_pyg_data_bpp(data, device)
        heu_vec = model(pyg_data)
        heu_mat = heu_vec.reshape((len(data), len(data))) + EPS

        aco = ACO_BPP(
            demand=data,
            heuristic=heu_mat,
            device=device,
            n_ants=policy_param['n_ants'],
            decay=policy_param['decay'],
            alpha=policy_param['alpha'],
            beta=policy_param['beta'],
            elitist=policy_param['elitist'],
        )


    else:
        raise ValueError("DeepACO does not support this problem.")

    return aco
