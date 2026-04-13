import os
import torch
from torch.utils.data import Dataset, DataLoader

from EasyNCO.utils.utils import getLogger

logger = getLogger(__name__)

####################################
# DATA
####################################
def OPGenerator(data_size, problem_size, batch_size, device='cpu', path=None, **kwargs):
    if path is None:
        dataset = random_op_generator(num_sample=data_size,
                                  num_nodes=problem_size,
                                  device=device,)
    else:
        dataset = customized_op_loader(num_sample=data_size,
                                        num_nodes=problem_size,
                                        device=device,
                                        path=path)

    data_loader = DataLoader(dataset=dataset,
                             batch_size=batch_size,
                             shuffle=False,
                             num_workers=0,
                             collate_fn=None)
    return data_loader


class random_op_generator(Dataset):
    '''
    The random_op_generator is used to generate the random OP data.
    The data is generated randomly with the size of [num_sample, num_nodes, 3]
    By default, the data is generated with the uniform distribution.
    '''
    def __init__(self, num_sample, num_nodes, device='cpu'):
        self.num_sample = num_sample
        self.num_nodes = num_nodes
        self.device = device


    def __getitem__(self, index):
        node_xy_data = torch.rand(size = (self.num_nodes, 2)).to(self.device)

        distances = (node_xy_data - node_xy_data[0]).norm(p=2, dim=-1)
        prizes = 1 + torch.floor(99 * distances / distances.max())
        prizes /= prizes.max()

        prizes = prizes.unsqueeze(1)
        #shape (problem, 1)

        node_op_data = torch.cat([node_xy_data, prizes], dim = 1)

        return node_op_data
        # shape: (num_nodes, 3): x,y,prize, including depot and customer nodes

    def __len__(self):
        return self.num_sample


class customized_op_loader(Dataset):
    '''
    The customized_op_loader is used to load the customized OP data from the file.
    The file should be saved in the format of .pkl or .pt. And the data should be saved in the format of [num_sample, num_nodes, 3]
    In the future, we will support the more general type of data.
    '''
    def __init__(self, num_sample, num_nodes, device='cpu', path=None):
        self.num_sample = num_sample
        self.num_nodes = num_nodes
        self.device = device
        self.path = path
        self.file_type = os.path.splitext(self.path)[-1] # maybe .pkl or .pt
        self.data = None

        # OP can use the dataset of TSP.
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
        node_xy_data = self.data[index]
        distances = (node_xy_data - node_xy_data[0]).norm(p=2, dim=-1)
        prizes = 1 + torch.floor(99 * distances / distances.max())
        prizes /= prizes.max()

        prizes = prizes.unsqueeze(1)
        # shape (problem, 1)

        node_op_data = torch.cat([node_xy_data, prizes], dim=1)
        return node_op_data
        # shape: (num_nodes, 3)

    def __len__(self):
        return self.num_sample


