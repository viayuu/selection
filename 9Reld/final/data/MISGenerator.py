import os

import torch
import numpy as np
import pickle
import glob

from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data as GraphData
from torch_geometric.data import DataLoader as GraphDataLoader

from EasyNCO.utils.utils import getLogger

logger = getLogger(__name__)

####################################
# DATA
####################################
def MISGenerator(data_size, problem_size, batch_size, device = 'cpu', path = None, graph = False, **kwargs):
    if path is None:
        raise NotImplementedError('Random generation for MIS is not supported currently')
    else:
        dataset = cutomized_mis_loader(num_sample = data_size,
                                       num_nodes = problem_size,
                                       device = device,
                                       path = path,
                                       graph = graph)
    data_loader = GraphDataLoader(dataset,
                             batch_size = batch_size,
                             shuffle = False,
                             num_workers = 0,
                             collate_fn = None)

    return data_loader

class cutomized_mis_loader(Dataset):
    def __init__(self, num_sample, num_nodes, device = 'cpu', path = None, graph = False, **kwargs):
        self.num_sample = num_sample
        self.num_nodes = num_nodes
        self.device = device
        self.path = path
        self.file_type = os.path.splitext(self.path)[-1]
        self.data = None
        self.graph = graph

        # self.file_list = glob.glob(path)
        self.file_list = os.listdir(path)

    def __getitem__(self, index):
        full_path = os.path.join(self.path, self.file_list[index])
        with open(full_path, 'rb') as f:
            graph = pickle.load(f)
        num_nodes = graph.number_of_nodes()

        node_labels = [_[1] for _ in graph.nodes(data = 'label')]
        if node_labels is not None and node_labels[0] is not None:
            node_labels = np.array(node_labels, dtype = np.int64)
        else:
            node_labels = np.zeros(num_nodes, dtype = np.int64)

        edge = np.array(graph.edges, dtype = np.int64)
        edge = np.concatenate([edge, edge[:, ::-1]], axis = 0)

        self_loop = np.arange(num_nodes).reshape(-1, 1).repeat(2, axis = 1)
        edges = np.concatenate([edge, self_loop], axis = 0)
        edges = edges.T

        graph_data = GraphData(x = torch.from_numpy(node_labels),
                               edge_index = torch.from_numpy(edges))

        node_indicator = np.array([num_nodes], dtype = np.int64)

        return (
            torch.LongTensor(np.array([index], dtype = np.int64)),
            graph_data,
            torch.from_numpy(node_indicator).long()
        )

    def __len__(self):

        return self.num_sample

