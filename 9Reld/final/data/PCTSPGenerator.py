import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from EasyNCO.utils.utils import getLogger

logger = getLogger(__name__)

K_n = {
    20: 2,
    50: 3,
    100: 4,
    500: 9,
    1000: 12,
    2000: 15,
    5000: 20,
    10000: 38
}


def PCTSPGenerator(data_size, problem_size, batch_size,device='cpu', path=None,distribution="uniform",**kwargs):
    if path is None:
        dataset = random_pctsp_generator(num_sample=data_size,
                                       num_nodes=problem_size,
                                       device=device)
    else:
        dataset = customized_pctsp_loader(num_sample=data_size,
                                        num_nodes=problem_size,
                                        device=device,
                                        path=path)

    data_loader = DataLoader(dataset=dataset,
                             batch_size=batch_size,
                             shuffle=False,
                             num_workers=0,
                             collate_fn=None)

    return data_loader

class random_pctsp_generator(Dataset):
    '''
    The random_vrp_generator is used to generate the random PCTSP data.
    The data is generated randomly with the size of [num_sample, num_nodes,4]
    By default, the data is generated with the uniform distribution.
    '''
    def __init__(self,num_sample,num_nodes,device):
        self.num_sample = num_sample
        self.num_nodes = num_nodes
        self.device = device

    def __getitem__(self, index):

        problems = torch.rand(size=(self.num_nodes+1, 2))

        prizes = torch.rand(self.num_nodes ) * 4 / self.num_nodes
        prize = torch.cat((torch.zeros((1)), prizes), dim=0)

        if K_n.get(self.num_nodes) is not None:
            K = K_n[self.num_nodes]
        else:
            K = np.random.randint(9, 12 + 1)  # one scalar
        beta = torch.rand(self.num_nodes) * 3 * K / self.num_nodes
        c = torch.cat((torch.zeros((1)), beta), dim=0)  # (n+1,)
        problems = torch.cat((problems, prize.unsqueeze(-1), c.unsqueeze(-1)), dim=1)
        # problems.shape: (problem, 4)
        return problems

    def __len__(self):
        return self.num_sample

class customized_pctsp_loader(Dataset):
    '''
    The customized_tsp_loader is used to load the customized PCTSP data from the file.
    The file should be saved in the format of .pkl or .pt. And the data should be saved in the format of [num_sample, num_nodes, 4]
    In the future, we will support the more general type of data.
    '''
    def __init__(self, num_sample, num_nodes, device='cpu', path=None):
        self.num_sample = num_sample
        self.num_nodes = num_nodes
        self.device = device
        self.path = path
        self.file_type = os.path.splitext(self.path)[-1]  # maybe .pkl or .pt
        self.data = None


        if self.file_type == '.pkl':
            import pickle
            with open(self.path, 'rb') as f:
                self.data = torch.tensor(pickle.load(f)[:self.num_sample], dtype=torch.float32)
        elif self.file_type == '.pt':
            self.data = torch.load(self.path)
        else:
            raise NotImplementedError(f"The file type ({self.file_type}) is not supported in current version")




    def __getitem__(self, index):
        data = self.data[index]

        return data

    def __len__(self):
        return self.num_sample


