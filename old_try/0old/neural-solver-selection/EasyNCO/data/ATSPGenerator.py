import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

from EasyNCO.utils.utils import getLogger

logger = getLogger(__name__)


####################################
# DATA
####################################
def ATSPGenerator(data_size, problem_size, batch_size,device='cpu',path=None,int_min=0, int_max=1000000,scaler=1000000,**kwargs):
    if path is None:
        dataset = random_atsp_generator(num_sample=data_size,
                                  num_nodes=problem_size,
                                  int_min=int_min, int_max=int_max,scaler=scaler,device=device)

    else:
        dataset = customized_atsp_loader(num_sample=data_size,
                                        num_nodes=problem_size,
                                        device=device,
                                        path=path)

    data_loader = DataLoader(dataset=dataset,
                             batch_size=batch_size,
                             shuffle=False,
                             num_workers=0,
                             collate_fn=None)
    return data_loader


class random_atsp_generator(Dataset):
    '''
    The random_tsp_generator is used to generate the random ATSP data.
    The data is generated randomly with the size of [num_sample, num_nodes, num_nodes]
    '''
    def __init__(self, num_sample, num_nodes,int_min=0, int_max=1000000,scaler=1000000,device='cpu'):
        self.num_sample = num_sample
        self.num_nodes = num_nodes
        self.int_min = int_min
        self.int_max=int_max
        self.scaler=scaler
        self.device = device

    def __getitem__(self, index):
        problems = torch.randint(
            low=self.int_min,
            high=self.int_max,
            size=(self.num_nodes, self.num_nodes),
            device=self.device,
        )
        # shape: (node, node)
        problems[torch.arange(self.num_nodes), torch.arange(self.num_nodes)] = 0  # 对角矩阵为0

        while True:
            old_problems = problems.clone()

            problems, _ = (problems[ :, None, :] + problems[None, :, :].transpose(1, 2)).min(dim=2)
            # shape: (batch, node, node)

            if (problems == old_problems).all():
                break

        # Scale
        scaled_problems = problems.float() / self.scaler
        return scaled_problems

    def __len__(self):
        return self.num_sample


class customized_atsp_loader(Dataset):
    '''
    The customized_tsp_loader is used to load the customized ATSP data from the file.
    The file should be saved in the format of .pkl or .pt. And the data should be saved in the format of [num_sample, num_nodes, num_nodes]
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
            raise NotImplementedError(f"The file type ({self.file_type}) is not supported in current version")

        elif self.file_type == '.pt':
            self.data = torch.load(self.path,map_location=self.device)[:self.num_sample]
        else:
            raise NotImplementedError(f"The file type ({self.file_type}) is not supported in current version")

    def __getitem__(self, index):
        data = self.data[index]

        return data

    def __len__(self):
        return self.num_sample


