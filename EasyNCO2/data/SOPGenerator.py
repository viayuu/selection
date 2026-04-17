import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

from EasyNCO.utils.utils import getLogger

logger = getLogger(__name__)

####################################
# DATA
####################################
def SOPGenerator(data_size, problem_size, batch_size, device='cpu', path=None, **kwargs):
    if path is None:
        dataset = random_sop_generator(num_sample=data_size,
                                  num_nodes=problem_size,
                                  device=device,)

    else:
        dataset = customized_sop_loader(num_sample=data_size,
                                        num_nodes=problem_size,
                                        device=device,
                                        path=path)

    data_loader = DataLoader(dataset=dataset,
                             batch_size=batch_size,
                             shuffle=False,
                             num_workers=0,
                             collate_fn=None)
    return data_loader


class random_sop_generator(Dataset):
    '''
    The random_sop_generator is used to generate the random SOP data.
    The data is generated randomly with the size of [num_sample, num_nodes, 2]
    By default, the data is generated with the uniform distribution.
    '''
    def __init__(self, num_sample, num_nodes, device='cpu'):
        self.num_sample = num_sample
        self.num_nodes = num_nodes
        self.device = device


    def __getitem__(self, index):
        distance = cost_mat_gen(self.num_nodes)
        r = ordering_constraint_gen(self.num_nodes)
        adj_mat = adjacency_mat_gen(self.num_nodes, r).to(self.device)

        data = torch.cat((distance, adj_mat), dim=0)
        return data

    def __len__(self):
        return self.num_sample


def ordering_constraint_gen(n, rand=0.2):
    r = []
    for i in range(1, n):
        r.append((0, i))

    a = [i for i in range(1, n)]
    precede = [set() for i in range(1, n)]
    for i in range(n - 3, -1, -1):
        for j in range(i + 1, n - 1):
            if torch.rand(size=(1,)) > rand:
                continue
            precede[i].add(j)
            for k in precede[j]:
                precede[i].add(k)

        for j in precede[i]:
            r.append((a[i], a[j]))
    return r

def cost_mat_gen(n):
    distances = torch.rand(size=(n, n))
    job_processing_cost = distances[0, :]
    distances[1:, :] += job_processing_cost
    return distances

def adjacency_mat_gen(n, r):
    c = torch.ones(size=(n, n))
    c[torch.arange(n), torch.arange(n)] = 0
    for i, j in r: # i precedes j
        c[j][i] = 0
    return c

class customized_sop_loader(Dataset):
    '''
    The customized_sop_loader is used to load the customized SOP data from the file.
    The file should be saved in the format of .pkl or .pt. And the data should be saved in the format of [num_sample, num_nodes, 2]
    In the future, we will support the more general type of data.
    '''
    def __init__(self, num_sample, num_nodes, device='cpu', path=None):
        self.num_sample = num_sample
        self.num_nodes = num_nodes
        self.device = device
        self.path = path
        self.file_type = os.path.splitext(self.path)[-1] # maybe .pkl or .pt
        self.data = None

        if self.file_type == '.pkl':
            import pickle
            with open(self.path, 'rb') as f:
                self.data = pickle.load(f)[:self.num_sample]
                # self.data = torch.tensor(pickle.load(f)[:self.num_sample], dtype=torch.float32)
        elif self.file_type == '.pt':
            # self.data = torch.load(self.path,map_location=self.device)['node_xy'][:self.num_sample]
            self.data = torch.load(self.path)
        else:
            raise NotImplementedError(f"The file type ({self.file_type}) is not supported in current version")

    def __getitem__(self, index):
        data = self.data[index]
        distance = data[0]
        adj_mat = data[1]
        mask = data[2]
        data_sop = torch.cat((distance, adj_mat), dim=0)
        return data_sop

    def __len__(self):
        return self.num_sample


