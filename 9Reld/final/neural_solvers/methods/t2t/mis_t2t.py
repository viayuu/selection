import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.sparse import coo_matrix
from torch_sparse import SparseTensor
from EasyNCO.neural_solvers.methods.difusco.util import InferenceSchedule
from EasyNCO.neural_solvers.backbones.Diffusion.diffusion import GaussianDiffusion, CategoricalDiffusion
from EasyNCO.neural_solvers.backbones.GNN.AGNNDifusco import GNNEncoder
from EasyNCO.utils.utils import getLogger

logger = getLogger(__name__)

class MIST2TPolicy(nn.Module):
    def __init__(self,
                 problem_type,
                 n_layers,
                 hidden_dim,
                 aggregation,
                 diffusion_type,
                 diffusion_schedule,
                 diffusion_steps,
                 sparse_factor,
                 use_activation_checkpoint,
                 parallel_sampling,
                 sequential_sampling,
                 inference_diffusion_steps,
                 inference_schedule,
                 inference_trick,
                 norm,
                 rewrite,
                 rewrite_steps,
                 rewrite_ratio,
                 rewrite_inference_steps,
                 node_feature_only = False):
        super().__init__()

        self.problem_type = problem_type
        self.diffusion_type = diffusion_type
        self.diffusion_schedule = diffusion_schedule
        self.diffusion_steps = diffusion_steps
        self.sparse = sparse_factor > 0 or node_feature_only
        self.sparse_factor = sparse_factor
        self.parallel_sampling = parallel_sampling
        self.sequential_sampling = sequential_sampling
        self.inference_diffusion_steps = inference_diffusion_steps
        self.inference_schedule = inference_schedule
        self.inference_trick = inference_trick
        self.norm = norm
        self.do_rewrite = rewrite
        self.rewrite_steps = rewrite_steps
        self.rewrite_ratio = rewrite_ratio
        self.rewrite_inference_steps = rewrite_inference_steps

        if self.diffusion_type == 'gaussian':
            out_channels = 1
            self.diffusion = GaussianDiffusion(
                T=self.diffusion_steps, schedule=self.diffusion_schedule)
        elif self.diffusion_type == 'categorical':
            out_channels = 2
            self.diffusion = CategoricalDiffusion(
                T=self.diffusion_steps, schedule=self.diffusion_schedule, sparse = self.sparse)
        else:
            raise ValueError(f"Unknown diffusion type {self.diffusion_type}")

        if hidden_dim is None:
            hidden_dim = 128

        self.model = GNNEncoder(
            n_layers = n_layers,
            hidden_dim = hidden_dim,
            out_channels = out_channels,
            aggregation = aggregation,
            sparse = self.sparse,
            use_activation_checkpoint = use_activation_checkpoint,
            node_feature_only = node_feature_only,
        )
        self.num_training_steps_cached = None

    def forward(self, x, t, edge_index):
        return self.model(x, t, edge_index = edge_index)

    def categorical_denoise_step(self, xt, t, device, edge_index = None, target_t = None):
        with torch.no_grad():
            t = torch.from_numpy(t).view(1)
            x0_pred = self.forward(
                xt.float().to(device),
                t.float().to(device),
                edge_index.long().to(device) if edge_index is not None else None,
            )
            x0_pred_prob = x0_pred.reshape((1, xt.shape[0], -1, 2)).softmax(dim = -1)
            xt, _ = self.diffusion.posterior(target_t, t, x0_pred_prob, xt)

            return xt

    def gaussian_denoise_step(self, xt, t, device, edge_index = None, target_t = None):
        with torch.no_grad():
            t = torch.from_numpy(t).view(1)
            pred = self.forward(
                xt.float().to(device),
                t.float().to(device),
                edge_index.long().to(device) if edge_index is not None else None
            )
            pred = pred.squeeze(1)
            xt = self.diffusion.posterior(target_t, t, pred, xt, self.inference_trick)

            return xt

    def guided_categorical_denoise_step(self, xt, t, device, edge_index = None, target_t = None):
        torch.set_grad_enabled(True)
        xt = xt.float()
        xt.requires_grad = True
        t = torch.from_numpy(t).view(1)
        # shape: (batch, 2, num_nodes, num_nodes)

        with torch.inference_mode(False):
            x0_pred = self.forward(
                xt.float().to(device),
                t.float().to(device),
                edge_index.long().to(device) if edge_index is not None else None
            )

            x0_pred_prob = x0_pred.reshape((1 , xt.shape[0], -1, 2)).softmax(dim = -1)
            num_nodes = xt.shape[0]
            adj_matrix = SparseTensor(
                row = edge_index[0],
                col = edge_index[1],
                value = torch.ones_like(edge_index[0].float()),
                sparse_sizes = (num_nodes, num_nodes),
            ).to_dense()
            adj_matrix.fill_diagonal_(0)

            pred_nodes = x0_pred_prob[..., 1].squeeze(0)
            f_mis = -pred_nodes.sum()
            g_mis = adj_matrix @ pred_nodes
            g_mis = (pred_nodes * g_mis).sum()
            cost_est = f_mis + 0.5 * g_mis
            cost_est.requires_grad_(True)
            cost_est.backward()

            assert xt.grad is not None
            if self.norm:
                xt.grad = F.normalize(xt.grad, p = 2, dim = -1)
            xt, _ = self.diffusion.posterior(target_t, t, x0_pred_prob, xt, guided = True, grad = xt.grad)

        return xt.detach()

    def initialize(self, batch, heatmap_path = None):
        self.device = batch[-1].device

        batch_idx, graph_data, nodes_indicator = batch
        nodes_labels = graph_data.x
        edge_index = graph_data.edge_index

        stacked_predict_labels = []
        edge_index = edge_index.to(self.device).reshape(2, -1)
        edge_index_np = edge_index.cpu().numpy()
        adj_mat = coo_matrix(
            (np.ones_like(edge_index_np[0]), (edge_index_np[0], edge_index_np[1]))
        )
        if self.parallel_sampling > 1:
            edge_index = self.duplicate_edge_index(edge_index, nodes_labels.shape[0], self.device)

        for _ in range(self.sequential_sampling):
            xt = torch.randn_like(nodes_labels.float())
            if self.parallel_sampling > 1:
                xt = xt.repeat(self.parallel_sampling, 1, 1)
                xt = torch.randn_like(xt)
            xt = (xt > 0).long()
            xt = xt.reshape(-1)

            batch_size = 1
            steps = self.inference_diffusion_steps
            time_schedule = InferenceSchedule(inference_schedule=self.inference_schedule,
                                              T=self.diffusion.T, inference_T=steps)
            for i in range(steps):
                t1, t2 = time_schedule(i)
                t1 = np.array([t1 for _ in range(batch_size)]).astype(int)
                t2 = np.array([t2 for _ in range(batch_size)]).astype(int)

                if self.diffusion_type == 'gaussian':
                    xt = self.gaussian_denoise_step(xt, t1, self.device, edge_index, target_t=t2)
                    # shape: (parallel_sampling * graph_size, )
                else:
                    xt = self.categorical_denoise_step(xt, t1, self.device, edge_index, target_t=t2)
                    # shape: (parallel_sampling * graph_size, )
            # generate node probability
            if self.diffusion_type == 'gaussian':
                predict_labels = xt.float().cpu().detach().numpy() * 0.5 + 0.5
            else:
                predict_labels = xt.float().cpu().detach().numpy() * 1e-6
            stacked_predict_labels.append(predict_labels)

        heatmap_dict = {
            'stacked_predict_labels' : stacked_predict_labels,
            'adj_mat' : adj_mat,
            'nodes_labels' : nodes_labels,
            'edge_index' : edge_index,
        }

        return heatmap_dict

    def iteration(self, heatmap_dict, max_step = 10):
        stacked_predict_labels = heatmap_dict['stacked_predict_labels']
        adj_mat = heatmap_dict['adj_mat']
        nodes_labels = heatmap_dict['nodes_labels']

        predict_labels = np.concatenate(stacked_predict_labels, axis = 0)
        all_sampling = self.sequential_sampling * self.parallel_sampling

        split_predict_labels = np.split(predict_labels, all_sampling)
        solved_solutions = [self.mis_decode_np(predict_labels, adj_mat) for predict_labels in split_predict_labels]
        solved_costs = [solved_solution.sum() for solved_solution in solved_solutions]
        best_cost = np.max(solved_costs)
        best_index = np.argmax(solved_costs)

        gt_cost = nodes_labels.cpu().numpy().sum()

        search_dict = {
            'best_cost': best_cost,
            'gt_cost' : gt_cost,
            'solved_cost' : solved_costs,
            'solved_best_solution' : solved_solutions[best_index]
        }

        metric = self.rewrite(heatmap_dict, search_dict, max_step)

        return metric

    def rewrite(self, heatmap_dict, search_dict, max_step):
        best_solution = search_dict['solved_best_solution']
        gt_cost = search_dict['gt_cost']
        edge_index = heatmap_dict['edge_index']
        adj_mat = heatmap_dict['adj_mat']
        guided_best_solved_cost = search_dict['best_cost']
        self.rewrite_inference_steps = max_step

        if self.do_rewrite:
            guided_best_solu = best_solution
            for _ in range(self.rewrite_steps):
                guided_stack_pred_labels = []
                g_x0 = torch.from_numpy(guided_best_solu).unsqueeze(0).to(self.device)
                g_x0 = F.one_hot(g_x0.long(), num_classes = 2).float()

                steps_T = int(self.diffusion_steps * self.rewrite_ratio)
                steps_inference = self.rewrite_inference_steps

                time_schedule = InferenceSchedule(inference_schedule=self.inference_schedule,
                                                  T=steps_T, inference_T=steps_inference)

                Q_bar = torch.from_numpy(self.diffusion.Q_bar[steps_T]).float().to(g_x0.device)
                g_xt_prob = torch.matmul(g_x0, Q_bar)  # [B, N, 2]
                g_xt = torch.bernoulli(g_xt_prob[..., 1].clamp(0, 1)).to(g_x0.device)  # [B, N]
                g_xt = g_xt * 2 - 1  # project to [-1, 1]
                g_xt = g_xt * (1.0 + 0.05 * torch.rand_like(g_xt))  # add noise

                if self.parallel_sampling > 1:
                    g_xt = g_xt.repeat(self.parallel_sampling, 1, 1)

                g_xt = (g_xt > 0).long().reshape(-1)
                for i in range(steps_inference):
                    t1, t2 = time_schedule(i)
                    t1 = np.array([t1]).astype(int)
                    t2 = np.array([t2]).astype(int)
                    g_xt = self.guided_categorical_denoise_step(g_xt, t1, self.device, edge_index, target_t=t2)

                g_predict_labels = g_xt.float().cpu().detach().numpy() + 1e-6
                guided_stack_pred_labels.append(g_predict_labels)
                g_predict_labels = np.concatenate(guided_stack_pred_labels, axis=0)

                g_split_predict_labels = np.split(g_predict_labels, self.parallel_sampling)
                g_solved_solutions = [self.mis_decode_np(g_predict_labels, adj_mat) for g_predict_labels in
                                      g_split_predict_labels]
                g_solved_costs = [g_solved_solution.sum() for g_solved_solution in g_solved_solutions]
                guided_best_solved_cost = np.max([guided_best_solved_cost, np.max(g_solved_costs)])
                g_best_solved_id = np.argmax(g_solved_costs)

                guided_best_solu = g_solved_solutions[g_best_solved_id]

            metric = {
                'no_aug_score' : guided_best_solved_cost,
                'aug_score': guided_best_solved_cost,
            }
            return metric


    def tour2adj(self, tour, points, sparse, sparse_factor, edge_index):
        if not sparse:
            adj_matrix = torch.zeros((points.shape[0], points.shape[0]))
            for i in range(tour.shape[0] - 1):
                adj_matrix[tour[i], tour[i + 1]] = 1
        else:
            adj_matrix = np.zeros(points.shape[0], dtype=np.int64)
            adj_matrix[tour[:-1]] = tour[1:]
            adj_matrix = torch.from_numpy(adj_matrix)
            adj_matrix = adj_matrix.reshape((-1, 1)).repeat(1, sparse_factor).reshape(-1)
            adj_matrix = torch.eq(edge_index[1].cpu(), adj_matrix).to(torch.int)
        return adj_matrix

    def nodes2adj(self, nodes):
        """
        return distance matrix
        Args:
          nodes: batch, num_nodes, 2
        Returns: batch, num_nodes, num_ndoes
        """
        assert nodes.dim() == 3
        return torch.sum((nodes.unsqueeze(2) - nodes.unsqueeze(1)) ** 2, dim=-1) ** 0.5

    def duplicate_edge_index(self, edge_index, num_nodes, device):
        """Duplicate the edge index (in sparse graphs) for parallel sampling."""
        edge_index = edge_index.reshape((2, 1, -1))
        edge_index_indent = torch.arange(0, self.parallel_sampling).view(1, -1, 1).to(device)
        edge_index_indent = edge_index_indent * num_nodes
        edge_index = edge_index + edge_index_indent
        edge_index = edge_index.reshape((2, -1))

        return edge_index

    def mis_decode_np(self, predictions, adj_matrix):
        """Decode the labels to the MIS."""
        solution = np.zeros_like(predictions.astype(int))
        sorted_predict_labels = np.argsort(- predictions)
        csr_adj_matrix = adj_matrix.tocsr()

        for i in sorted_predict_labels:
            next_node = i

            if solution[next_node] == -1:
                continue
            # no edges between these nodes
            solution[csr_adj_matrix[next_node].nonzero()[1]] = -1
            # nodes that in the independent set
            solution[next_node] = 1

        return (solution == 1).astype(int)

###############################################
# Training, same as DIFUSCO
###############################################

    def gaussian_training_step(self, batch):
        _, graph_data, point_indicator = batch
        t = np.random.randint(1, self.diffusion.T + 1, point_indicator.shape[0]).astype(int)
        node_labels = graph_data.x
        edge_index = graph_data.edge_index
        device = node_labels.device

        # Sample from diffusion
        node_labels = node_labels.float() * 2 - 1
        node_labels = node_labels * (1.0 + 0.05 * torch.rand_like(node_labels))
        node_labels = node_labels.unsqueeze(1).unsqueeze(1)

        t = torch.from_numpy(t).long()
        t = t.repeat_interleave(point_indicator.reshape(-1).cpu(), dim = 0).numpy()
        xt, epsilon = self.diffusion.sample(node_labels, t)

        t = torch.from_numpy(t).float()
        t = t.reshape(-1)
        xt = xt.reshape(-1)
        edge_index = edge_index.to(device).reshape(2, -1)
        epsilon = epsilon.reshape(-1)

        # Denoise
        epsilon_pred = self.forward(
            xt.float().to(device),
            t.float().to(device),
            edge_index,
        )
        epsilon_pred = epsilon_pred.squeeze(1)

        loss = F.mse_loss(epsilon_pred, epsilon.float())

        return loss

    def categorical_training_step(self, batch):
        _, graph_data, point_indicator = batch
        t = np.random.randint(1, self.diffusion.T + 1, point_indicator.shape[0]).astype(int)
        nodes_labels = graph_data.x
        edge_index = graph_data.edge_index
        device = nodes_labels.device

        # sample from diffusion
        node_labels_onehot = F.one_hot(nodes_labels.long(), num_classes = 2).float()
        node_labels_onehot = node_labels_onehot.unsqueeze(1).unsqueeze(1)

        t = torch.from_numpy(t).long()
        t = t.repeat_interleave(point_indicator.reshape(-1).cpu(), dim = 0).numpy()

        xt = self.diffusion.sample(node_labels_onehot, t)
        xt = xt * 2 - 1
        xt = xt * (1.0 + 0.05 * torch.rand_like(xt))

        t = torch.from_numpy(t).float()
        t = t.reshape(-1)
        xt = xt.reshape(-1)
        edge_index = edge_index.to(device).reshape(2, -1)

        # Denoise
        x0_pred = self.forward(
            xt.float().to(device),
            t.float().to(device),
            edge_index,
        )

        loss_func = nn.CrossEntropyLoss()
        loss = loss_func(x0_pred, nodes_labels)

        return loss

    def training_step(self, batch):
        if self.diffusion_type == 'gaussian':
            loss = self.gaussian_training_step(batch)
        else:
            loss = self.categorical_training_step(batch)
        return loss