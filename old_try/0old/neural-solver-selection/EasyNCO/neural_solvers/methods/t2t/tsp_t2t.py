import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from EasyNCO.neural_solvers.methods.difusco.util import InferenceSchedule, merge_tour, two_opt_refine, evaluate_tsp_tour
from EasyNCO.neural_solvers.backbones.Diffusion.diffusion import GaussianDiffusion, CategoricalDiffusion
from EasyNCO.neural_solvers.backbones.GNN.AGNNDifusco import GNNEncoder
from EasyNCO.utils.utils import getLogger

logger = getLogger(__name__)

class TSPT2TPolicy(nn.Module):
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

        hidden_dim = 256 if hidden_dim is None else hidden_dim

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

    def guided_categorical_denoise_step(self, nodes, xt, t, device, edge_index = None, target_t = None):
        torch.set_grad_enabled(True)
        xt = xt.float()
        xt.requires_grad = True
        t = torch.from_numpy(t).view(1)

        with torch.inference_mode(False):
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

            if not self.sparse:
                dis_matrix = self.nodes2adj(nodes)
                cost_est = (dis_matrix * x0_pred_prob[..., 1]).sum()
                cost_est.requires_grad_(True)
                cost_est.backward()
            else:
                dis_matrix = torch.sqrt(torch.sum((nodes[edge_index.T[:, 0]] - nodes[edge_index[:, 1]])**2, dim = 1))
                dis_matrix = dis_matrix.reshape((1, nodes.shape[0], -1))
                cost_est = (dis_matrix * x0_pred_prob[..., 1]).sum()
                cost_est.requires_grad_(True)
                cost_est.backward()
            assert xt.grad is not None

            if self.norm:
                xt.grad = F.normalize(xt.grad, p = 2, dim = -1)
            with torch.no_grad():
                xt, _ = self.diffusion.posterior(target_t, t, x0_pred_prob, xt, guided = True, grad = xt.grad)

            return xt.detach()

    def initialize(self, batch, heatmap_path = None):
        edge_index = None
        original_edge_index = None
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
            original_edge_index = edge_index.clone()
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
        for i in range(self.sequential_sampling):
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
                self.save_numpy_heatmap(adj_mat, split='test', heatmap_path=heatmap_path)

            for i in range(self.sequential_sampling):
                # Extract tours from given heatmap
                tours, merge_iterations = merge_tour(
                    adj_mat, np_nodes, np_edge_index,
                    sparse_graph=self.sparse,
                    parallel_sampling=self.parallel_sampling
                )

            policyDict = {
                'np_nodes' : np_nodes,
                'nodes' : nodes,
                'edge_index' : edge_index,
                'np_edge_index' : np_edge_index,
                'heatmap' : adj_mat,
                'np_gt_tour' : np_gt_tour,
                'original_edge_index' : original_edge_index,
                'tours': tours
            }
            stacked_policy_dict.append(policyDict)

        return stacked_policy_dict

    def iteration(self, stacked_policy_dict, max_steps, heatmap_path = None):
        stacked_tours = []

        for i in range(self.sequential_sampling):
            policy_dict = stacked_policy_dict[i]
            # np_edge_index = policy_dict['np_edge_index']
            np_nodes = policy_dict['np_nodes']
            np_gt_tour = policy_dict['np_gt_tour']
            tours = policy_dict['tours']

            # Refine using 2-opt
            refined_tours, ns = two_opt_refine(
                np_nodes.astype('float64'),
                np.array(tours).astype('int64'),
                max_iteration=max_steps,
                device=self.device,
            )
            stacked_tours.append(refined_tours)

        solved_tours = np.concatenate(stacked_tours, axis=0)
        gt_cost = evaluate_tsp_tour(np_nodes, np_gt_tour)
        total_sampling = self.parallel_sampling * self.sequential_sampling
        total_solved_cost = [evaluate_tsp_tour(np_nodes, solved_tours[i]) for i in range(total_sampling)]
        best_solved_cost = np.min(total_solved_cost)
        optimal_id = np.argmin(total_solved_cost)
        best_tour = solved_tours[optimal_id]

        nodes = stacked_policy_dict[optimal_id]['nodes']
        np_edge_index = stacked_policy_dict[optimal_id]['np_edge_index']
        edge_index = stacked_policy_dict[optimal_id]['edge_index']
        original_edge_index = stacked_policy_dict[optimal_id]['original_edge_index']

        if self.do_rewrite:
            guided_best_solved_cost = best_solved_cost

            for _ in range(self.rewrite_steps):
                guided_stacked_tours = []
                # optimal adjacent matrix
                g_x0 = self.tour2adj(best_tour, np_nodes, self.sparse, self.sparse_factor, original_edge_index)
                g_x0 = g_x0.unsqueeze(0).to(self.device)
                if self.parallel_sampling > 1:
                    if not self.sparse:
                        g_x0 = g_x0.repeat(self.parallel_sampling, 1, 1)
                        # shape: (1 ~ batch, N, N)
                    else:
                        g_x0 = g_x0.repeat(self.parallel_sampling, 1)

                if self.sparse:
                    g_x0 = g_x0.reshape(-1)

                g_x0_onehot = F.one_hot(g_x0.long(), num_classes = 2).float()
                # shape: (batch, N, N, 2)
                steps_T = int(self.diffusion_steps * self.rewrite_ratio)
                steps_inference = self.rewrite_inference_steps
                time_schedule = InferenceSchedule(inference_schedule=self.inference_schedule,
                                                  T = steps_T,
                                                  inference_T = steps_inference)
                Q_bar = torch.from_numpy(self.diffusion.Q_bar[steps_T]).float().to(g_x0_onehot.device)
                g_xt_prob = torch.matmul(g_x0_onehot, Q_bar)
                # shape: (batch, N, N, 2)

                # add noise for the steps_T samples, namely rewrite
                g_xt = torch.bernoulli(g_xt_prob[..., 1].clamp(0, 1))
                # shape: (batch, N, N)
                g_xt = g_xt * 2 - 1
                # project to [-1, 1]
                g_xt = g_xt * (1.0 + 0.05 * torch.rand_like(g_xt))  # add noise
                g_xt = (g_xt > 0).long()

                for i in range(steps_inference):
                    t1, t2 = time_schedule(i)
                    t1 = np.array([t1]).astype(int)
                    t2 = np.array([t2]).astype(int)

                    g_xt = self.guided_categorical_denoise_step(nodes, g_xt, t1, self.device, edge_index, target_t = t2)
                g_adj_mat = g_xt.float().cpu().detach().numpy() + 1e-6

                if heatmap_path is not None:
                    self.save_numpy_heatmap(g_adj_mat, split = 'test', heatmap_path=heatmap_path)

                g_tours, g_merge_iterations = merge_tour(
                    g_adj_mat, np_nodes, np_edge_index,
                    sparse_graph=self.sparse,
                    parallel_sampling=self.parallel_sampling,
                )
                g_solved_tours, g_ns = two_opt_refine(
                    np_nodes.astype("float64"), np.array(g_tours).astype('int64'),
                    max_iteration=max_steps, device=self.device)
                guided_stacked_tours.append(g_solved_tours)

                guided_solved_tours = np.concatenate(guided_stacked_tours, axis=0)
                guided_total_sampling = self.parallel_sampling
                guided_solved_costs = [evaluate_tsp_tour(np_nodes, guided_solved_tours[i]) for i in range(guided_total_sampling)]

                guided_best_cost_tmp = np.min(guided_solved_costs)
                guided_best_index = np.argmin(guided_solved_costs)
                guided_best_solved_cost = min(guided_best_solved_cost, guided_best_cost_tmp)

                guided_gap = (guided_best_solved_cost - gt_cost) / gt_cost * 100
                guided_best_tour = guided_solved_tours[guided_best_index]

                metrics = {
                    'no_aug_score': guided_best_solved_cost,
                    'aug_score' : guided_best_solved_cost,
                }

                return metrics

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

    def save_numpy_heatmap(self, adj_mat, split, heatmap_path):
        if self.parallel_sampling > 1 or self.sequential_sampling > 1:
            raise NotImplementedError("Save numpy heatmap only support single sampling")

        logger.info(f"Saving heatmap tp {heatmap_path}")
        os.makedirs(heatmap_path, exist_ok = True)
        np.save(os.path.join(heatmap_path, f'{split}-heatmap.npy'), adj_mat)
###############################################
# Training, same as DIFUSCO
###############################################
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

    def training_step(self, batch):
        if self.diffusion_type == 'gaussian':
            loss = self.gaussian_training_step(batch)
        else:
            loss = self.categorical_training_step(batch)
        return loss
