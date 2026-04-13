import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from EasyNCO.neural_solvers.methods.difusco.util import InferenceSchedule, merge_tour, two_opt_refine, evaluate_tsp_tour
from EasyNCO.neural_solvers.backbones.GNN.AGNNDifusco import GNNEncoder
from EasyNCO.neural_solvers.backbones.Diffusion.diffusion import GaussianDiffusion, CategoricalDiffusion
from EasyNCO.utils.utils import getLogger

logger = getLogger(__name__)

class TSPDiffusionPolicy(nn.Module):
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
                 node_feature_only = False):
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
            out_channels = 1
            self.diffusion = GaussianDiffusion(
                T=self.diffusion_steps, schedule=self.diffusion_schedule)
        elif self.diffusion_type == 'categorical':
            out_channels = 2
            self.diffusion = CategoricalDiffusion(
                T=self.diffusion_steps, schedule=self.diffusion_schedule, sparse = self.sparse)
        else:
            raise ValueError(f"Unknown diffusion type {self.diffusion_type}")


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

    def forward(self, x, adj, t, edge_index):
        return self.model(x, t, adj, edge_index)


    def categorical_denoise_step(self, nodes, xt, t, device, edge_index = None, target_t = None):
        with torch.no_grad():
            t = torch.from_numpy(t).view(1)
            x0_pred = self.forward(
                nodes.float().to(device),
                xt.float().to(device),
                t.float().to(device),
                edge_index.long().to(device) if edge_index is not None else None
            )

            if not self.sparse:
                x0_pred_prob = x0_pred.permute((0, 2, 3, 1)).contiguous().softmax(dim = -1)
            else:
                x0_pred_prob = x0_pred.reshape((1, nodes.shape[0], -1, 2)).softmax(dim = -1)

            xt, _ = self.diffusion.posterior(target_t, t, x0_pred_prob, xt)
            return xt

    def gaussian_denoise_step(self, nodes, xt, t, device, edge_index = None, target_t = None):
        with torch.no_grad():
            t = torch.from_numpy(t).view(1)
            pred = self.forward(
                nodes.float().to(device),
                xt.float().to(device),
                t.float().to(device),
                edge_index.long().to(device) if edge_index is not None else None,
            )
            pred = pred.squeeze(1)

            xt = self.diffusion.posterior(target_t, t, pred, xt, self.inference_trick)

            return xt

    def categorical_training_step(self, batch):
        edge_index = None
        if not self.sparse:
            _, points, adj_matrix, _ = batch
            t = np.random.randint(1, self.diffusion.T + 1, points.shape[0]).astype(int)
        else:
            _, graph_data, point_indicator, edge_indicator, _ = batch
            t = np.random.randint(1, self.diffusion.T + 1, point_indicator.shape[0]).astype(int)
            route_edge_flags = graph_data.edge_attr
            points = graph_data.x
            edge_index = graph_data.edge_index
            num_edges = edge_index.shape[1]
            batch_size = point_indicator.shape[0]
            adj_matrix = route_edge_flags.reshape((batch_size, num_edges // batch_size))

        # Sample from diffusion
        adj_matrix_onehot = F.one_hot(adj_matrix.long(), num_classes=2).float()
        if self.sparse:
            adj_matrix_onehot = adj_matrix_onehot.unsqueeze(1)

        xt = self.diffusion.sample(adj_matrix_onehot, t)
        xt = xt * 2 - 1
        xt = xt * (1.0 + 0.05 * torch.rand_like(xt))

        if self.sparse:
            t = torch.from_numpy(t).float()
            t = t.reshape(-1, 1).repeat(1, adj_matrix.shape[1]).reshape(-1)
            xt = xt.reshape(-1)
            adj_matrix = adj_matrix.reshape(-1)
            points = points.reshape(-1, 2)
            edge_index = edge_index.float().to(adj_matrix.device).reshape(2, -1)
        else:
            t = torch.from_numpy(t).float().view(adj_matrix.shape[0])

        # Denoise
        x0_pred = self.forward(
            points.float().to(adj_matrix.device),
            xt.float().to(adj_matrix.device),
            t.float().to(adj_matrix.device),
            edge_index,
        )

        # Compute loss
        loss_func = nn.CrossEntropyLoss()
        loss = loss_func(x0_pred, adj_matrix.long())
        return loss

    def gaussian_training_step(self, batch):
        if self.sparse:
            raise ValueError("DIFUSCO with sparse graphs are not supported for Gaussian diffusion")
        _, points, adj_matrix, _ = batch

        adj_matrix = adj_matrix * 2 - 1
        adj_matrix = adj_matrix * (1.0 + 0.05 * torch.rand_like(adj_matrix))
        # Sample from diffusion
        t = np.random.randint(1, self.diffusion.T + 1, adj_matrix.shape[0]).astype(int)
        xt, epsilon = self.diffusion.sample(adj_matrix, t)

        t = torch.from_numpy(t).float().view(adj_matrix.shape[0])
        # Denoise
        epsilon_pred = self.forward(
            points.float().to(adj_matrix.device),
            xt.float().to(adj_matrix.device),
            t.float().to(adj_matrix.device),
            None,
        )
        epsilon_pred = epsilon_pred.squeeze(1)

        # Compute loss
        loss = F.mse_loss(epsilon_pred, epsilon.float())
        return loss

    def initialize(self, batch, heatmap_path = None):
        """
        Used to generate the heatmap with diffusion
        """
        edge_index = None
        np_edge_index = None
        self.device = batch[-1].device
        if not self.sparse:
            batch_idx, nodes, adj_matrix, gt_tour = batch
            np_nodes = nodes.cpu().numpy()[0]
            np_gt_tour = gt_tour.cpu().numpy()[0]
        else:
            batch_idx, graph_data, nodes_indicator, edge_indicator, gt_tour = batch
            route_edge_flags = graph_data.edge_attr
            nodes = graph_data.x.reshape((-1, 2))
            edge_index = graph_data.edge_index.reshape((2, -1))
            num_edges = edge_index.shape[1]
            batch_size = nodes_indicator.shape[0]
            adj_matrix = route_edge_flags.reshape((batch_size, num_edges // batch_size))
            np_nodes = nodes.cpu().numpy()
            np_gt_tour = gt_tour.cpu().numpy().reshape(-1)
            np_edge_index = edge_index.cpu().numpy()

        if self.parallel_sampling > 1:
            if not self.sparse:
                nodes = nodes.repeat(self.parallel_sampling, 1, 1)
            else:
                nodes = nodes.repeat(self.parallel_sampling, 1)
                edge_index = self.duplicate_edge_index(edge_index, np_nodes.shape[0], self.device)

        # sample solution in the latent space
        stacked_policy_dict = []
        for _ in range(self.sequential_sampling):
            xt = torch.randn_like(adj_matrix.float())
            if self.parallel_sampling > 1:
                if not self.sparse:
                    xt = xt.repeat(self.parallel_sampling, 1, 1)
                else:
                    xt = xt.repeat(self.parallel_sampling, 1)
                xt = torch.randn_like(xt)

            if self.diffusion_type == 'gaussian':
                xt.requires_grad = True
            else:
                xt = (xt > 0).long()

            if self.sparse:
                xt = xt.reshape(-1)

            steps = self.inference_diffusion_steps
            time_schedule = InferenceSchedule(inference_schedule=self.inference_schedule,
                                              T=self.diffusion.T,
                                              inference_T=steps)
            # denoise iteration
            for i in range(steps):
                t1, t2 = time_schedule(i)
                t1 = np.array([t1]).astype(int)
                t2 = np.array([t2]).astype(int)

                if self.diffusion_type == 'gaussian':
                    xt = self.gaussian_denoise_step(nodes, xt, t1, self.device, edge_index, target_t=t2)
                else:
                    xt = self.categorical_denoise_step(nodes, xt, t1, self.device, edge_index, target_t=t2)

            if self.diffusion_type == 'gaussian':
                adj_mat = xt.cpu().detach().numpy() * 0.5 + 0.5
            else:
                adj_mat = xt.float().cpu().detach().numpy() + 1e-6
            if self.parallel_sampling == 1 and heatmap_path is not None:
                self.save_numpy_heatmap(adj_mat, batch_idx, split='test', heatmap_path=heatmap_path)

            for i in range(self.sequential_sampling):

                torus, merge_iterations = merge_tour(
                    adj_mat, np_nodes, np_edge_index,
                    sparse_graph = self.sparse,
                    parallel_sampling = self.parallel_sampling
                    )

            policyDict = {
                'np_nodes' : np_nodes,
                'np_edge_index' : np_edge_index,
                'heatmap' : adj_mat,
                'tours' : torus,
                'np_gt_tour' : np_gt_tour,
            }
            stacked_policy_dict.append(policyDict)

        return stacked_policy_dict

    def iteration(self, stacked_policy_dict, max_step):
        stacked_tours = []
        for i in range(self.sequential_sampling):
            policy_dict = stacked_policy_dict[i]
            np_nodes = policy_dict['np_nodes']
            tours = policy_dict['tours']

            # Refine using 2-opt
            refined_tours, ns = two_opt_refine(
                np_nodes.astype('float64'),
                np.array(tours).astype('int64'),
                max_iteration = max_step,
                device = self.device,
            )
            stacked_tours.append(refined_tours)

        solved_tours = np.concatenate(stacked_tours, axis = 0)

        total_sampling = self.parallel_sampling * self.sequential_sampling
        total_solved_cost = [evaluate_tsp_tour(np_nodes, solved_tours[i]) for i in range(total_sampling)]
        best_solved_cost = np.min(total_solved_cost)

        metrics = {
            'no_aug_score' : best_solved_cost,
            'aug_score' : best_solved_cost,
        }

        return metrics

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

    def save_numpy_heatmap(self, adj_mat, batch_idx, split, heatmap_path):
        if self.parallel_sampling > 1 or self.sequential_sampling > 1:
            raise NotImplementedError("Save numpy heatmap only support single sampling")

        logger.info(f"Saving heatmap tp {heatmap_path}")
        os.makedirs(heatmap_path, exist_ok = True)
        batch_idx = batch_idx.cpu().numpy().reshape(-1)[0]
        np.save(os.path.join(heatmap_path, f'{split}-heatmap-{batch_idx}.npy'), adj_mat)


