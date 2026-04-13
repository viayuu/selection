import os

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from EasyNCO.utils.utils import getLogger

logger = getLogger(__name__)

####################################
# DATA
####################################

def KPGenerator(data_size, problem_size, batch_size, device = 'cpu', path = None):
    if path is None:
        dataset = random_kp_generator(num_sample = data_size,
                                      num_items = problem_size,
                                      device = device)
    else:
        dataset = customized_kp_loader(num_sample = data_size,
                                       num_items = problem_size,
                                       device = device)

    data_loader = DataLoader(dataset = dataset,
                             batch_size = batch_size,
                             shuffle = False,
                             num_workers = 0,
                             collate_fn = None)

    return data_loader



class random_kp_generator(Dataset):
    '''
    The random_kp_generator is used to generate the random Knapsack data.
    The data is generated randomly with the size of [num_sample, num_nodes, demand, 2]
    By default, the data is generated with the uniform distribution.
    '''

    def __init__(self, num_sample, num_items, device = 'cpu'):
        self.num_sample = num_sample
        self.num_items = num_items
        self.device = device

    def __getitem__(self, index):
        data = torch.rand(self.num_items, 2)

        return data

    def __len__(self):

        return self.num_sample


class customized_kp_loader(Dataset):
    '''
    The customized_kp_loader is used to load the customized KP data from the file.
    The file should be saved in the format of .pkl or .pt. And the data should be saved in the format of [num_sample, num_items, 2]
    In the future, we will support the more general type of data.
    '''
    def __init__(self, num_sample, num_items, device = 'cpu', path = None):
        self.num_sample = num_sample
        self.num_items = num_items
        self.device = device
        self.path = path
        self.file_type = os.path.splitext(self.path)[-1] # maybe .pkl or .pt
        self.data = None

        if self.file_type == '.pkl':
            import pickle
            with open(self.path, 'rb') as f:
                self.data = torch.tensor(pickle.load(f)[:self.num_sample], dtype = torch.float32)

            with open(self.path, 'rb') as f:
                out = np.array(pickle.load(f)[:self.num_sample], dtype = object)
                saved_item_data = torch.tensor(out.tolist(), dtype = torch.float32).to(device)
        elif self.file_type == '.pt':
            loaded_dict = torch.load(self.path, map_location = self.device)
            saved_item_data = loaded_dict['saved_item_data'][:self.num_sample]

        else:
            raise NotImplementedError(f"The file type ({self.file_type}) is not supported in current version")

        self.data = saved_item_data

    def __getitem__(self, index):
        item_data = self.data[index]

        return item_data

    def __len__(self):

        return self.num_sample