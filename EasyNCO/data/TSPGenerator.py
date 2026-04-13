import os
import torch
import numpy as np
import random

from torch.utils.data import Dataset, DataLoader
from sklearn.neighbors import KDTree
from torch_geometric.data import Data as GraphData
import ast


from EasyNCO.utils.utils import getLogger
from EasyNCO.data.data_utils import get_gaussian_mixture,generate_by_distribution,generate_explosion_instance,generate_implosion_instance
from EasyNCO.data.LIBUtils import TSPLIBReader, tsplib_cost
from EasyNCO.data.data_utils import normalize_to_unit_board


logger = getLogger(__name__)

####################################
# DATA
####################################
def TSPGenerator(data_size, problem_size, batch_size,device='cpu', path=None, partial_updater=None,distribution="uniform",graph = False, **kwargs):
    scale_range = kwargs.get("scale_range", None)
    if isinstance(scale_range,str):
        scale_range = ast.literal_eval(scale_range)
        kwargs["scale_range"] = scale_range
    if scale_range is not None and scale_range[0] != scale_range[1]:
        batch_size = 1
    if path is None:
        dataset = random_tsp_generator(num_sample=data_size,
                                  num_nodes=problem_size,
                                  device=device,distribution=distribution,
                                  **kwargs)
    elif path.endswith('tsplib'):
        batch_size = 1
        dataset = tsplib_loader(device=device,
                                 path=path,
                                 **kwargs)
    elif scale_range is not None:
        dataset = varying_tsp_loader(
            num_sample=data_size, device=device, path=path, **kwargs
        )
    else:
        dataset = customized_tsp_loader(num_sample=data_size,
                                        num_nodes=problem_size,
                                        device=device,
                                        path=path,
                                        graph = graph,
                                        **kwargs)
        if partial_updater is not None:
            dataset.data['to_DataLoader'] = partial_updater.load_problem_one_epoch(dataset)
            batch_size = 1
    data_loader = DataLoader(dataset=dataset,
                             batch_size=batch_size,
                             shuffle=False,
                             num_workers=0,
                             collate_fn=None)
    return data_loader


class random_tsp_generator(Dataset):
    '''
    The random_tsp_generator is used to generate the random TSP data.
    The data is generated randomly with the size of [num_sample, num_nodes, 2]
    By default, the data is generated with the uniform distribution.
    # num_modes and cdist are used for gaussian_mixture
    # explosion_args include [range_min , range_max , rate]
    # implosion_args include [range_min , range_max]
    '''
    def __init__(self, num_sample, num_nodes, device='cpu',distribution="uniform",**kwargs):
        self.num_sample = num_sample
        self.num_nodes = num_nodes
        self.device = device
        self.distribution=distribution
        self.num_modes=kwargs.get('num_modes',0)
        self.cdist=kwargs.get('cdist',0)
        self.explosion_args=kwargs.get('explosion_args',None)
        self.implosion_args=kwargs.get('implosion_args',None)
        self.problem_size_low = kwargs.get('problem_size_low', None)
        self.problem_size_high = kwargs.get('problem_size_high', None)

    def __getitem__(self, index):
        if self.problem_size_low: # udc
            num_nodes = (
                np.random.randint(
                    self.problem_size_low // self.num_nodes,
                    self.problem_size_high // self.num_nodes + 1,
                )
                * self.num_nodes
            )
        else:
            num_nodes = self.num_nodes
        if self.distribution=="uniform":
            node_xy_data = torch.rand(num_nodes, 2).to(self.device)
        elif self.distribution=="glop_scale":
            x = torch.rand(num_nodes, 1).to(self.device)
            y = torch.rand(num_nodes, 1).to(self.device) * torch.rand(1)
            node_xy_data = torch.cat((x, y), dim=1)
        elif self.distribution=="gaussian_mixture":
            node_xy_data = get_gaussian_mixture(self.num_nodes, self.num_modes, self.cdist).to(self.device)
        elif self.distribution in ["uniform_rectangle","gaussian","cluster","diagonal"]:
            #Generate data using the method in Omni
            node_xy_data = generate_by_distribution(self.num_nodes,self.distribution).to(self.device)
        elif self.distribution == "explosion":
            #Generate data using the method in Invit
            if self.explosion_args is not None:
                range_min,range_max,rate = self.explosion_args['range_min'],self.explosion_args['range_max'],self.explosion_args['rate']
                node_xy_data = generate_explosion_instance(self.num_nodes, range_min, range_max, rate).to(self.device)
            else:
                node_xy_data = generate_explosion_instance(self.num_nodes).to(self.device)
        elif self.distribution == "implosion":
            # Generate data using the method in Invit
            if self.implosion_args is not None:
                range_min,range_max = self.implosion_args['range_min'],self.explosion_args['range_max']
                node_xy_data = generate_implosion_instance(self.num_nodes,range_min,range_max).to(self.device)
            else:
                node_xy_data = generate_implosion_instance(self.num_nodes).to(self.device)
        else:
            raise ValueError(f"Unsupported distribution type: {self.distribution}")

        return node_xy_data

    def __len__(self):
        return self.num_sample


class customized_tsp_loader(Dataset):
    '''
    The customized_tsp_loader is used to load the customized TSP data from the file.
    The file should be saved in the format of .pkl or .pt.
    And the data should be saved in the format of {'node_xy': shape (sample_num, problem, 2),
                                                    (if learning paradigm is sl, 'label' will be contained)}
    In the future, we will support the more general type of data.
    '''
    def __init__(self, num_sample, num_nodes, device='cpu', path=None, graph = False, **kwargs):
        self.num_sample = num_sample
        self.num_nodes = num_nodes
        self.device = device
        self.path = path
        self.file_type = os.path.splitext(self.path)[-1] # maybe .pkl or .pt
        self.data = None
        self.graph = graph
        self.sparse_factor = kwargs.get('sparse_factor', -1)
        self.fine_tune_offset = kwargs.get('fine_tune_offset', None)

        self.udc_flag = kwargs.get('udc_flag', False)

        if self.file_type == '.pkl':
            import pickle
            with open(self.path, 'rb') as f:
                if self.fine_tune_offset is not None:
                    self.data = torch.tensor(pickle.load(f)[self.fine_tune_offset:self.fine_tune_offset+self.num_sample], dtype=torch.float32)
                else:
                    self.data = torch.tensor(pickle.load(f)[:self.num_sample], dtype=torch.float32)

        elif self.file_type == '.pt':
            self.data = torch.load(self.path, map_location=self.device)
            if 'label' in self.data.keys():
                for key, value in self.data.items():
                    self.data[key] = value[:self.num_sample]
            else:
                self.data = self.data['node_xy'][:self.num_sample]
        elif self.file_type == '.txt':
            self.file_lines = open(self.path).read().splitlines()
            self.data = {}
        else:
            raise NotImplementedError(f"The file type ({self.file_type}) is not supported in current version")

    def __getitem__(self, index):
        if not self.graph:
            if isinstance(self.data, dict):
                return self.data['to_DataLoader'][index]
            elif not isinstance(self.data, dict):
                return self.data[index]
        else:
            if self.file_type == '.txt':
                nodes, tours = self.get_txt_file(index)
            elif self.file_type == '.pt':
                nodes = self.data[index]
                tours = torch.randint(low = 0, high = nodes.shape[0], size = (nodes.shape[0], ), dtype = torch.int64)
                start_node = torch.tensor(0, dtype = torch.int64).reshape(1, )
                tours = torch.cat((tours, start_node), dim = 0)
            if self.udc_flag:
                return getTSPGraphItem(nodes, tours, self.sparse_factor, index)[1]
            return getTSPGraphItem(nodes, tours, self.sparse_factor,index)

    def __len__(self):
        if isinstance(self.data, dict) and 'to_DataLoader' in self.data.keys():
            return len(self.data['to_DataLoader'])
        else:
            return self.num_sample

    def get_txt_file(self, idx_):

        line = self.file_lines[idx_]
        line = line.strip()

        nodes = line.split(' output ')[0]
        nodes = nodes.split(' ')
        num_nodes = len(nodes)
        nodes = np.array([[float(nodes[i]), float(nodes[i + 1])] for i in range(0, num_nodes, 2)])

        tours = line.split(' output ')[1]
        tours = tours.split(' ')
        tours = np.array([int(t) for t in tours])
        tours -= 1

        data_nodes = torch.tensor(nodes)
        data_tour = torch.tensor(tours)

        return data_nodes, data_tour

class varying_tsp_loader(Dataset):

    def __init__(
        self, num_sample, device="cpu", path=None, **kwargs
    ):
        self.num_sample = num_sample
        self.device = device
        self.path = path
        self.file_type = os.path.splitext(self.path)[-1]  # maybe .pkl or .pt
        self.data = None
        self.scale_range = kwargs.get("scale_range")  # e.g. [500, 800]
        self.distribution = kwargs.get(
            "distribution_list", None
        )  # e.g. ['uniform', 'cluster']

        if self.file_type == ".pt":
            self.data = torch.load(self.path, map_location=self.device)
            scale_list = self.data["scale_list"]
            distribution_list = self.data["distribution_list"]

            # Find indices that satisfy both scale range and distribution conditions
            self.valid_indices = []
            for i, (scale, dist) in enumerate(zip(scale_list, distribution_list)):
                valid = True
                # Check scale range condition
                if not (self.scale_range[0] <= scale < self.scale_range[1]):
                    valid = False

                # Check distribution condition
                if (self.distribution is not None and dist not in self.distribution):
                    valid = False

                if valid:
                    self.valid_indices.append(i)

            # random sample
            if self.num_sample < len(self.valid_indices):
                self.valid_indices = random.sample(
                    self.valid_indices, self.num_sample
                )
            else:
                self.num_sample = len(self.valid_indices)

        else:
            raise NotImplementedError(
                f"The file type ({self.file_type}) is not supported in current version"
            )

    def __getitem__(self, index):
        index = self.valid_indices[index]
        sample = self.data["dataset"][f"num{index}"]
        item = {
            "data": torch.tensor(sample["node_xy"], dtype=torch.float32).to(self.device),
            "num_sample": self.num_sample,
        }
        if "cost" in sample:
            item["optimal"] = torch.tensor(
                sample["cost"], dtype=torch.float32
            ).to(self.device)
        if "solution" in sample:
            item["solution"] = torch.tensor(
                sample["solution"], dtype=torch.int64
            ).to(self.device)
        if "name" in sample:
            item["name"] = sample["name"]
        return item

    def __len__(self):
        return self.num_sample


class tsplib_loader(Dataset):
    def __init__(self, device="cpu", path=None, **kwargs):
        self.device = device
        self.path = path
        self.scale_range = kwargs.get("scale_range", None)  # e.g. [500, 800]
        self.data = {}
        self.num_sample = 0
        self.scale_min_max = [float("inf"), float("-inf")]
        for root, dirs, files in os.walk(self.path):
            for file in files:
                if file.endswith(".tsp"):
                    name, dimension, locs, edge_weight_type = TSPLIBReader(os.path.join(root, file))
                    if name is None:
                        continue
                    if self.scale_range is not None and not (self.scale_range[0] <= dimension < self.scale_range[1]):
                        continue
                    self.data[name] = {"dimension": dimension, "locations": locs, "edge_weight_type": edge_weight_type}
                    self.num_sample += 1
                    self.scale_min_max[0] = min(self.scale_min_max[0], dimension)
                    self.scale_min_max[1] = max(self.scale_min_max[1], dimension)
        if self.num_sample == 0:
            raise ValueError(f"No TSP files found in {self.path} within the specified scale range {self.scale_range}, the scale range is {self.scale_min_max}")

    def __getitem__(self, index):
        name = list(self.data.keys())[index]
        edge_weight_type = self.data[name]["edge_weight_type"]
        original_data = torch.tensor(self.data[name]["locations"], dtype=torch.float32).to(self.device)
        data, _ = normalize_to_unit_board(original_data)
        return {
            "name": name,
            "edge_weight_type": edge_weight_type,
            "lib_data": original_data,
            "data": data,
            "optimal": torch.tensor(tsplib_cost.get(name, 0), dtype=torch.float32).to(
                self.device
            ),
            "num_sample": self.num_sample,
        }
    def __len__(self):
        return self.num_sample


def getTSPGraphItem(nodes, tours, sparse_factor, idx):

    if sparse_factor <= 0:
        # Return a densely connected graph
        adj_matrix = np.zeros((nodes.shape[0], nodes.shape[0]))
        for i in range(tours.shape[0] - 1):
            adj_matrix[tours[i], tours[i + 1]] = 1
        # return nodes, adj_matrix, tour
        return (
            torch.LongTensor(np.array([idx], dtype=np.int64)),
            nodes,
            torch.from_numpy(adj_matrix).float(),
            tours,
        )
    else:
        # Return a sparse graph where each node is connected to its k nearest neighbors
        # k = self.sparse_factor
        sparse_factor = sparse_factor
        kdt = KDTree(nodes, leaf_size=30, metric='euclidean')
        dis_knn, idx_knn = kdt.query(nodes, k=sparse_factor, return_distance=True)

        edge_index_0 = torch.arange(nodes.shape[0]).reshape((-1, 1)).repeat(1, sparse_factor).reshape(-1)
        edge_index_1 = torch.from_numpy(idx_knn.reshape(-1))


        edge_index = torch.stack([edge_index_0, edge_index_1], dim=0)

        tour_edges = np.zeros(nodes.shape[0], dtype=np.int64)
        tour_edges[tours[:-1]] = tours[1:]
        tour_edges = torch.from_numpy(tour_edges)
        tour_edges = tour_edges.reshape((-1, 1)).repeat(1, sparse_factor).reshape(-1)
        tour_edges = torch.eq(edge_index_1, tour_edges).reshape(-1, 1)
        graph_data = GraphData(x=torch.from_numpy(nodes).float(),
        edge_index=edge_index,
        edge_attr=tour_edges)

        point_indicator = np.array([nodes.shape[0]], dtype=np.int64)
        edge_indicator = np.array([edge_index.shape[1]], dtype=np.int64)

        return (
            torch.LongTensor(np.array([idx], dtype=np.int64)),
            graph_data,
            torch.from_numpy(point_indicator).long(),
            torch.from_numpy(edge_indicator).long(),
            torch.from_numpy(tours).long(),
        )
