import os
import torch
import glob
from torch.utils.data import Dataset, DataLoader
from typing import List

from EasyNCO.utils.utils import getLogger

logger = getLogger(__name__)

####################################
# DATA
####################################
def RCPSPGenerator(data_size, problem_size, batch_size, device='cpu', path=None, **kwargs):
    if path is None:
        dataset = random_rcpsp_generator(num_sample=data_size,
                                         num_nodes=problem_size,
                                         device=device,)
    else:
        dataset = customized_rcpsp_loader(num_sample=data_size,
                                          num_nodes=problem_size,
                                          device=device,
                                          path=path)
    data_loader = DataLoader(dataset=dataset,
                             batch_size=batch_size,
                             shuffle=False,
                             num_workers=0,
                             collate_fn=None)
    return data_loader


class random_rcpsp_generator(Dataset):
    '''
    The random_rcpsp_generator is used to generate the random RCPSP data.
    Due to the complexity of generating rspsp, we are currently directly using the existing dataset.
    In the future, we will make further improvements.
    '''
    def __init__(self, num_sample, num_nodes, device='cpu'):
        self.num_sample = num_sample
        self.num_nodes = num_nodes
        self.device = device


    def __getitem__(self, index):
        pass


    def __len__(self):
        return self.num_sample


class customized_rcpsp_loader(Dataset):
    '''
    The customized_rcpsp_loader is used to load the customized RCPSP data from the file.
    The file should be saved in the format of .pkl or .pt. And the data should be saved in the format of [num_sample, num_nodes, 2]
    In the future, we will support the more general type of data.

    In psplib.tar.gz are some RCPSP instances downloaded from https://www.om-db.wi.tum.de/psplib/data.html, which is a subset of PSPLIB dataset [1]. 
    This tarball should be extracted before running our program.
    [1] Kolisch, Rainer, and Arno Sprecher. “PSPLIB - A Project Scheduling Problem Library: OR Software - ORSEP Operations Research Software Exchange Program.” European Journal of Operational Research 96, no. 1 (January 10, 1997): 205–16. https://doi.org/10.1016/S0377-2217(96)00170-1.
    '''
    def __init__(self, num_sample, num_nodes, device='cpu', path=None):
        self.num_sample = num_sample
        self.num_nodes = num_nodes
        self.device = device
        self.path = path


    def __getitem__(self, index):
        """Load a set of RCP files from a folder.
         Only the first {test_size} files (in lexicographic order) are included in the testset.

         Args:
             directory (str)
             test_size (int, optional): Size of testset. Defaults to 100.
         Returns:
             trainset (list[RCPSPInstance])
             testset (list[RCPSPInstance])
         """
        current_file_path = os.path.abspath(__file__)
        current_dir = os.path.dirname(current_file_path)
        data_path = os.path.join(current_dir, self.path)
        files = glob.glob(os.path.join(data_path, "*.RCP"))
        files.sort()

        resource_capacity, duration, resources, succ_mat = read_RCPfile(files[index])
        part1 = torch.cat([torch.zeros(1), duration], dim=0)
        part2 = torch.cat([resource_capacity.unsqueeze(0), resources], dim=0)

        data = torch.cat([part1.unsqueeze(1), part2, succ_mat], dim=1)

        return data

    def __len__(self):
        return self.num_sample


def readints(f) -> List[int]:
    return list(map(int, f.readline().strip().split()))

def read_RCPfile(filepath):
    '''

    Args:
        filepath:

    Returns:
        resource_capacity: shape:(n_resources, )
        duration: shape:(n_jobs, )
        resources: shape:(n_jobs, n_resources)
        succ_mat: shape:(n_jobs+1, n_jobs+1)

    '''
    with open(filepath) as f:
        n_jobs, n_resources = readints(f)
        resource_capacity = readints(f)
        assert len(resource_capacity) == n_resources
        precede = []
        duration = []
        resources = []
        for i in range(1, n_jobs+1):
            line = iter(readints(f))
            duration.append(next(line))
            resources.append([next(line) for _ in range(n_resources)])
            n_successors = next(line)
            successors = list(line)  # consume all items remaining in `line`
            assert len(successors) == n_successors
            for succ_index in successors:  # connect the nodes
                # The index in RCP file starts from 1, so the shape of succ_mat is (n_jobs+1, n_jobs+1)
                precede.append((i, succ_index))
    succ_mat = successor_mat_gen(n_jobs, precede)
    return torch.tensor(resource_capacity), torch.tensor(duration), torch.tensor(resources), torch.tensor(succ_mat)

def successor_mat_gen(n, r):
    '''
    The successor nodes of node i are marked with 1 in prec_mat[i, :]
    We set up a dummy whose index is 0, so the shape of succ_mat is (n_jobs+1, n_jobs+1)
    '''
    succ_mat = torch.zeros(size=(n+1, n+1))
    for i, j in r:
        succ_mat[i, j] = 1
    return succ_mat