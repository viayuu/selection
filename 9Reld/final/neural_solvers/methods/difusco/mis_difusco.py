import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.sparse import coo_matrix

from EasyNCO.neural_solvers.methods.difusco.util import InferenceSchedule
from EasyNCO.neural_solvers.backbones.Diffusion.diffusion import GaussianDiffusion, CategoricalDiffusion
from EasyNCO.neural_solvers.backbones.GNN.AGNNDifusco import GNNEncoder
from EasyNCO.utils.utils import getLogger

logger = getLogger(__name__)

class MISDiffusionPolicy(nn.Module):
    def __init__(self,
                 env_name,
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
                 node_feature_only = True):
        super().__init__()

        self.problem_type = env_name
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

        if self.diffusion_type == 'gaussian':
            out_channel = 1
            self.diffusion = GaussianDiffusion(T = self.diffusion_steps, schedule = self.diffusion_schedule)
        elif self.diffusion_type == 'categorical':
            out_channel = 2
            self.diffusion = CategoricalDiffusion(T = self.diffusion_steps, schedule = self.diffusion_schedule, sparse = self.sparse)
        else:
            raise ValueError(f"Unknown diffusion type {self.diffusion_type}")

        self.model = GNNEncoder(
            n_layers = n_layers,
            hidden_dim = hidden_dim,
            out_channels = out_channel,
            aggregation = aggregation,
            sparse = self.sparse,
            use_activation_checkpoint = use_activation_checkpoint,
            node_feature_only = node_feature_only
        )
        self.num_training_steps_cached = None

    def forward(self, x, t, edge_index):
        return self.model(x, t, edge_index = edge_index)

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
        # sample solution in the latent space
        for _ in range(self.sequential_sampling):
            xt = torch.randn_like(nodes_labels.float())
            if self.parallel_sampling > 1:
                xt = xt.repeat(self.parallel_sampling, 1, 1)
                xt = torch.randn_like(xt)
            if self.diffusion_type == 'gaussian':
                xt.requires_grad = True
            else:
                xt = (xt > 0).long()
            xt = xt.reshape(-1)

            if self.parallel_sampling > 1:
                edge_index = self.duplicate_edge_index(edge_index, nodes_labels.shape[0], self.device)
            batch_size = 1
            steps = self.inference_diffusion_steps
            time_schedule = InferenceSchedule(inference_schedule = self.inference_schedule,
                                              T = self.diffusion.T,
                                              inference_T = steps)
            # denoise step
            for i in range(steps):
                t1, t2 = time_schedule(i)
                t1 = np.array([t1 for _ in range(batch_size)]).astype(int)
                t2 = np.array([t2 for _ in range(batch_size)]).astype(int)

                if self.diffusion_type == 'gaussian':
                    xt = self.gaussian_denoise_step(xt, t1, self.device, edge_index, target_t = t2)
                    # shape: (parallel_sampling * graph_size, )
                else:
                    xt = self.categorical_denoise_step(xt, t1, self.device, edge_index, target_t = t2)
                    # shape: (parallel_sampling * graph_size, )
            # generate node probability
            if self.diffusion_type == 'gaussian':
                predict_labels = xt.float().cpu().detach().numpy() * 0.5 + 0.5
            else:
                predict_labels = xt.float().cpu().detach().numpy() * 1e-6
            stacked_predict_labels.append(predict_labels)

        policy_dict = {
            'stacked_predict_labels' : stacked_predict_labels,
            'adj_mat' : adj_mat,
            'nodes_labels' : nodes_labels,
        }



        return policy_dict

    def iteration(self, policy_dict, max_steps):
        stacked_predict_labels = policy_dict['stacked_predict_labels']
        adj_mat = policy_dict['adj_mat']
        nodes_labels = policy_dict['nodes_labels']

        predict_labels = np.concatenate(stacked_predict_labels, axis = 0)
        all_sampling = self.sequential_sampling * self.parallel_sampling

        # extract MIS node from the heatmap
        split_predict_labels = np.split(predict_labels, all_sampling)
        solved_solutions = [self.mis_decode_np(predict_labels, adj_mat) for predict_labels in split_predict_labels]
        solved_costs = [solved_solution.sum() for solved_solution in solved_solutions]
        best_cost = np.max(solved_costs)

        gt_cost = nodes_labels.cpu().numpy().sum()

        metric = {
            'score': best_cost,
            'aug_score': best_cost,
        }

        return metric

    def training_step(self, batch):
        if self.diffusion_type == 'gaussian':
            loss = self.gaussian_training_step(batch)
        else:
            loss = self.categorical_training_step(batch)
        return loss

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