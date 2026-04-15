import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

from EasyNCO.utils.utils import getLogger

logger = getLogger(__name__)

####################################
# DATA
####################################
def MKPGenerator(data_size, problem_size, batch_size, device='cpu', path=None, **kwargs):
    if path is None:
        dataset = random_mkp_generator(num_sample=data_size,
                                  num_nodes=problem_size,
                                  device=device,)

    else:
        dataset = customized_mkp_loader(num_sample=data_size,
                                        num_nodes=problem_size,
                                        device=device,
                                        path=path)

    data_loader = DataLoader(dataset=dataset,
                             batch_size=batch_size,
                             shuffle=False,
                             num_workers=0,
                             collate_fn=None)
    return data_loader

class random_mkp_generator(Dataset):
    '''
    The random_mkp_generator is used to generate the random MKP data.
    The data is generated randomly with the size of [num_sample, n, m+1]
    By default, the data is generated with the uniform distribution.
    '''
    def __init__(self, num_sample, num_nodes, device='cpu'):
        self.num_sample = num_sample
        self.knapsacks = num_nodes
        self.device = device
        self.constraints = 5


    def __getitem__(self, index):
        '''
        Generate *well-stated* MKP instances
        Args:
            self.knapsacks: # n of knapsacks
            self.constraints: # m of constraints, a.k.a., the problem dimensionality
        '''
        prize = torch.rand(size=(self.knapsacks,), device=self.device)
        weight_matrix = torch.rand(size=(self.knapsacks, self.constraints), device=self.device)
        max_weight, _ = torch.max(weight_matrix, dim=0)
        sum_weight = torch.sum(weight_matrix, dim=0)
        constraints = []
        for idx in range(self.constraints):
            constraint = np.random.uniform(low=max_weight[idx].item(), high=sum_weight[idx].item())
            constraints.append(constraint)
        constraints = torch.tensor(constraints, device=self.device)
        # after norm, constraints are all 1
        weight_matrix = weight_matrix / constraints.unsqueeze(0)

        data = torch.cat((prize.unsqueeze(1), weight_matrix), dim=1)
        # prize: (n, ); weight_matrix: (n, m)
        return data # (n, m+1)

    def __len__(self):
        return self.num_sample

class customized_mkp_loader(Dataset):
    '''
    The customized_mkp_loader is used to load the customized MKP data from the file.
    The file should be saved in the format of .pkl or .pt. And the data should be saved in the format of [num_sample, n, m+1]
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

        return data

    def __len__(self):
        return self.num_sample
