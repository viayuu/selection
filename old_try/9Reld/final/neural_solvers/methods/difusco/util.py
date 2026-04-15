"""Schedulers for Denoising Diffusion Probabilistic Models"""
import warnings

import math

import numpy as np
import torch
from scipy.sparse import coo_matrix
from scipy.spatial import distance_matrix
from multiprocessing import Pool



class InferenceSchedule(object):
  def __init__(self, inference_schedule="linear", T=1000, inference_T=1000):
    self.inference_schedule = inference_schedule
    self.T = T
    self.inference_T = inference_T

  def __call__(self, i):
    assert 0 <= i < self.inference_T

    if self.inference_schedule == "linear":
      t1 = self.T - int((float(i) / self.inference_T) * self.T)
      t1 = np.clip(t1, 1, self.T)

      t2 = self.T - int((float(i + 1) / self.inference_T) * self.T)
      t2 = np.clip(t2, 0, self.T - 1)
      return t1, t2
    elif self.inference_schedule == "cosine":
      t1 = self.T - int(
          np.sin((float(i) / self.inference_T) * np.pi / 2) * self.T)
      t1 = np.clip(t1, 1, self.T)

      t2 = self.T - int(
          np.sin((float(i + 1) / self.inference_T) * np.pi / 2) * self.T)
      t2 = np.clip(t2, 0, self.T - 1)
      return t1, t2
    else:
      raise ValueError("Unknown inference schedule: {}".format(self.inference_schedule))

##############################################################
# Util Functions
##############################################################
def merge_tour(adj_mat, np_points, edge_index_np, sparse_graph = False, parallel_sampling = 1):
  """
  To extract a tour from the inferred adjacency matrix A, we used the following greedy edge insertion
  procedure.
  • Initialize extracted tour with an empty graph with N vertices.
  • Sort all the possible edges (i, j) in decreasing order of Aij/kvi − vjk (i.e., the inverse edge weight,
  multiplied by inferred likelihood). Call the resulting edge list (i1, j1),(i2, j2), . . . .
  • For each edge (i, j) in the list:
    – If inserting (i, j) into the graph results in a complete tour, insert (i, j) and terminate.
    – If inserting (i, j) results in a graph with cycles (of length < N), continue.
    – Otherwise, insert (i, j) into the tour.
  • Return the extracted tour.
  """
  split_adj_mat = np.split(adj_mat, parallel_sampling, axis = 0)

  if not sparse_graph:
    split_adj_mat = [
      adj_mat[0] + adj_mat[0].T for adj_mat in split_adj_mat
    ]
  else:
    split_adj_mat = [
      coo_matrix(
          (adj_mat, (edge_index_np[0], edge_index_np[1])),
      ).toarray() +
      coo_matrix(
        (adj_mat, (edge_index_np[1], edge_index_np[0])),
      ).toarray() for adj_mat in split_adj_mat
    ]

  split_points = [
    np_points for _ in range(parallel_sampling)
  ]

  if np_points.shape[0] > 1000 and parallel_sampling > 1:
    with Pool(parallel_sampling) as p:
      result = p.starmap(
        cython_merge,
        zip(split_points, split_adj_mat)
      )
  else:
    result = [
              cython_merge(tmp_np_points, tmp_adj_mat) for tmp_np_points, tmp_adj_mat in zip(split_points, split_adj_mat)
    ]

  split_real_adj_mat, split_merge_iteration = zip(*result)
  tours = []

  for i in range(parallel_sampling):
    tour = [0]
    while len(tour) < split_adj_mat[i].shape[0] + 1:
      n = np.nonzero(split_real_adj_mat[i][tour[-1]])[0]
      if len(tour) > 1:
        n = n[n != tour[-2]]
      tour.append(n.max())
    tours.append(tour)

  merge_iterations = np.mean(split_merge_iteration)

  return tours, merge_iterations


def cython_merge(nodes, adj_mat):
  from EasyNCO.neural_solvers.methods.difusco.merge.cython_merge import merge_cython
  with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    real_adj_mat, merge_iterations = merge_cython(nodes.astype('double'), adj_mat.astype('double'))
    real_adj_mat = np.asarray(real_adj_mat)

  return real_adj_mat, merge_iterations

def evaluate_tsp_tour(nodes, route):
  dist_mat = distance_matrix(nodes, nodes)

  total_cost = 0
  for i in range(len(route) - 1):
    total_cost += dist_mat[route[i], route[i + 1]]

  return total_cost

def two_opt_refine(nodes, tour, max_iteration = 1000, device = 'cpu'):
  counter = 0
  tour = tour.copy()
  with torch.inference_mode():
    cuda_nodes = torch.from_numpy(nodes).to(device)
    cuda_tour = torch.from_numpy(tour).to(device)
    batch_size = cuda_tour.shape[0]
    min_change = -1.0

    while min_change < 0.0:
      nodes_i = cuda_nodes[cuda_tour[:, : -1].reshape(-1)].reshape((batch_size, -1, 1, 2))
      nodes_j = cuda_nodes[cuda_tour[:, : -1].reshape(-1)].reshape((batch_size, 1, -1, 2))
      nodes_i_plus_1 = cuda_nodes[cuda_tour[:, 1 :].reshape(-1)].reshape((batch_size, -1, 1, 2))
      nodes_j_plus_1 = cuda_nodes[cuda_tour[:, 1 :].reshape(-1)].reshape((batch_size, 1, -1, 2))

      A_ij = torch.sqrt(torch.sum((nodes_i - nodes_j) ** 2, dim = -1))
      A_i_plus_1_j_plus_1 = torch.sqrt(torch.sum((nodes_i_plus_1 - nodes_j_plus_1) ** 2, dim = -1))
      A_i_i_plus_1 = torch.sqrt(torch.sum((nodes_i - nodes_i_plus_1) ** 2, dim = -1))
      A_j_j_plus_1 = torch.sqrt(torch.sum((nodes_j - nodes_j_plus_1) ** 2, dim = -1))

      change = A_ij + A_i_plus_1_j_plus_1 - A_i_i_plus_1 - A_j_j_plus_1
      valid_change = torch.triu(change, diagonal = 2)

      min_change = torch.min(valid_change)
      flatten_argmin_index = torch.argmin(valid_change.reshape(batch_size, -1), dim = -1)
      min_i = torch.div(flatten_argmin_index, len(nodes), rounding_mode = 'floor')
      min_j = torch.remainder(flatten_argmin_index, len(nodes))

      if min_change < -1e-6:
        for i in range(batch_size):
          cuda_tour[i, min_i[i] + 1 : min_j[i] + 1] = torch.flip(cuda_tour[i, min_i[i] + 1 : min_j[i] + 1], dims = (0, ))
        counter += 1
      else:
        break

      if counter >= max_iteration:
        break
    tour = cuda_tour.cpu().numpy()

  return tour, counter
