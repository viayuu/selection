import os
import torch
import numpy as np

from torch.utils.data import Dataset, DataLoader
from sklearn.neighbors import KDTree


from EasyNCO.utils.utils import getLogger
from EasyNCO.data.data_utils import get_gaussian_mixture,generate_by_distribution,generate_explosion_instance,generate_implosion_instance


logger = getLogger(__name__)

####################################
# DATA
####################################
def MOTSPGenerator(data_size, problem_size, batch_size,device='cpu', path=None, partial_updater=None,distribution="uniform",**kwargs):
    if path is None:
        dataset = random_motsp_generator(num_sample=data_size,
                                  num_nodes=problem_size,
                                  device=device,distribution=distribution,
                                  **kwargs)
    else:
        dataset = customized_motsp_loader(num_sample=data_size,
                                        num_nodes=problem_size,
                                        device=device,
                                        path=path,
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


class random_motsp_generator(Dataset):
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
        self.num_target = kwargs.get('num_target',2)
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
            node_xy_data = torch.rand(num_nodes, 2 * self.num_target).to(self.device)
        else:
            raise ValueError(f"Unsupported distribution type: {self.distribution}")

        return node_xy_data

    def __len__(self):
        return self.num_sample


class customized_motsp_loader(Dataset):
    '''
    The customized_tsp_loader is used to load the customized TSP data from the file.
    The file should be saved in the format of .pkl or .pt.
    And the data should be saved in the format of {'node_xy': shape (sample_num, problem, 2),
                                                    (if learning paradigm is sl, 'label' will be contained)}
    In the future, we will support the more general type of data.
    '''
    def __init__(self, num_sample, num_nodes, device='cpu', path=None, **kwargs):
        self.num_sample = num_sample
        self.num_nodes = num_nodes
        self.device = device
        self.path = path
        self.file_type = os.path.splitext(self.path)[-1] # maybe .pkl or .pt
        self.data = None
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
        if isinstance(self.data, dict):
            return self.data['to_DataLoader'][index]
        elif not isinstance(self.data, dict):
            return self.data[index]

    def __len__(self):
        if isinstance(self.data, dict) and 'to_DataLoader' in self.data.keys():
            return len(self.data['to_DataLoader'])
        else:
            return self.num_sample






