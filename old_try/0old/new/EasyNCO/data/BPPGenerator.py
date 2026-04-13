import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

from EasyNCO.utils.utils import getLogger

logger = getLogger(__name__)

####################################
# DATA
####################################
def BPPGenerator(data_size, problem_size, batch_size,device='cpu', path=None, **kwargs):
    if path is None:
        dataset = random_bpp_generator(num_sample=data_size,
                                  num_nodes=problem_size,
                                  device=device,)
    else:
        dataset = customized_bpp_loader(num_sample=data_size,
                                        num_nodes=problem_size,
                                        device=device,
                                        path=path)

    data_loader = DataLoader(dataset=dataset,
                             batch_size=batch_size,
                             shuffle=False,
                             num_workers=0,
                             collate_fn=None)
    return data_loader


class random_bpp_generator(Dataset):
    '''
    The random_bpp_generator is used to generate the random BPP data.
    The data is generated randomly with the size of [num_sample, num_nodes+1]
    By default, the data is generated with the uniform distribution.
    '''
    def __init__(self, num_sample, num_nodes, device='cpu'):
        self.num_sample = num_sample
        self.num_nodes = num_nodes
        self.device = device


    def __getitem__(self, index):
        DEMAND_LOW = 20
        DEMAND_HIGH = 100
        demands = torch.randint(low=DEMAND_LOW, high=DEMAND_HIGH + 1, size=(self.num_nodes,), device=self.device)
        all_demands = torch.cat((torch.zeros((1,), device=self.device), demands))

        return all_demands # (n+1,)

    def __len__(self):
        return self.num_sample


class customized_bpp_loader(Dataset):
    '''
    The customized_bpp_loader is used to load the customized BPP data from the file.
    The file should be saved in the format of .pkl or .pt. And the data should be saved in the format of [num_sample, num_nodes+1]
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
        demands = self.data[index]

        return demands # (n+1,)

    def __len__(self):
        return self.num_sample


