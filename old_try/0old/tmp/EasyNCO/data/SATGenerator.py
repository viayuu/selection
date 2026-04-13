import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

import glob

from EasyNCO.utils.utils import getLogger

logger = getLogger(__name__)


####################################
# DATA
####################################
def SATGenerator(data_size, problem_size, batch_size,device='cpu',path=None,int_min=0, int_max=1000000,scaler=1000000, **kwargs):
    
    if path is None:
        raise NotImplementedError("HCP problem generator has not suppported yet.")
    
    else:
        
        if os.path.isdir(path):
            dataset = variale_sat_loader(num_sample=data_size,
                                         device=device,
                                         path=path)
        else:
            dataset = customized_sat_loader(num_sample=data_size,
                                            num_nodes=problem_size,
                                            device=device,
                                            path=path)

    data_loader = DataLoader(dataset=dataset,
                             batch_size=batch_size,
                             shuffle=False,
                             num_workers=0,
                             collate_fn=None)
    return data_loader



class variale_sat_loader(Dataset):
    
    def __init__(self,num_sample, device='cpu', path=None):
        self.num_sample = num_sample
        self.device = device
        self.path = path # dir type
        self.data = None
           
        search_pattern = os.path.join(self.path, '*.atsp')
        self.file_paths = sorted(glob.glob(search_pattern))
        
        if num_sample is not None:
            if num_sample > len(self.file_paths):
                raise NotImplementedError("episode is larger than num_sample under this path")
            self.file_paths = self.file_paths[:num_sample]

        if not self.file_paths:
            raise NotImplementedError("cannot find valid .atsp file under this path")

           
    def __len__(self):
        
        return len(self.file_paths)
        
    def __getitem__(self, index):
        
        file_path = self.file_paths[index]
        data = self._load_single_problem(file_path)
                
        return data
    
    def _load_single_problem(self, file_path,scaler=1e6): 

        with open(file_path, 'r') as f:
            lines = [line.strip() for line in f.readlines()]

        node_cnt = 0
        in_matrix_section = False
        data_start_index = 0
        line_idx = -1 
        for line in lines:
            line_idx += 1
            if line.startswith('EDGE_WEIGHT_SECTION'):
                in_matrix_section = True
                data_start_index = line_idx + 1
                continue
                    
            if in_matrix_section:
                if line.upper() == 'EOF':
                    break
                node_cnt += 1
        if node_cnt == 0:
            raise NotImplementedError(f"The EDGE_WEIGHT_SECTIOIN is not found in file")
                    

        # find the start line in matrix
        problem = torch.empty(size=(node_cnt, node_cnt), dtype=torch.float32)
        matrix_lines = lines[data_start_index : data_start_index + node_cnt]
        for i, line in enumerate(matrix_lines):
            weights = list(map(int, line.split()))
            problem[i] = torch.tensor(weights, dtype=torch.float32)
            
        problem.fill_diagonal_(0)     
        # normlization
        scaled_problem = problem / scaler

        return scaled_problem
        

class customized_sat_loader(Dataset):
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
            if os.path.isdir(self.path):
               self.data = self.load_batch_problem_from_file(scaler = 1e-6) 
            else:
                raise NotImplementedError(f"The file type ({self.file_type}) is not supported in current version")

    def __getitem__(self, index):
        data = self.data[index]

        return data

    def __len__(self):
        return self.num_sample
    
    
    def load_batch_problem_from_file(self, scaler):
        
        # search all files in data path
        file_list = sorted(glob.glob(os.path.join(self.path, '*.atsp')))
        if self.num_sample is not None:
            file_list = file_list[:self.num_sample]
        if not file_list:
            raise NotImplementedError(f"The file type ({self.file_type}) is not found in data path")
        
        
        
        # each data is saved in list first
        batch_problems = []
        # add matrix line into problem
        for filename in file_list:
            try:
                with open(filename, 'r') as f:
                    lines = [line.strip() for line in f.readlines()]

                node_cnt = 0
                in_matrix_section = False
                for line in lines:
                    if line.startswith('EDGE_WEIGHT_SECTION'):
                        in_matrix_section = True
                        continue
                    
                    if in_matrix_section:
                        if line.upper() == 'EOF' or not line.split()[0].isdigit():
                            break
                        node_cnt += 1
            
                if node_cnt is None:
                    raise NotImplementedError(f"The DIMENSION is not found in file")
                    

                # find the start line in matrix
                problem = torch.empty(size=(node_cnt, node_cnt), dtype=torch.float32)
                data_start_index = -1
                for i, line in enumerate(lines):
                    if line.startswith('EDGE_WEIGHT_SECTION'):
                        data_start_index = i + 1
                        break
            
                if data_start_index == -1:
                    raise NotImplementedError(f"The EDGE_WEIGHT_SECTION is not found in file")


                matrix_lines = lines[data_start_index : data_start_index + node_cnt]
                for i, line in enumerate(matrix_lines):
                    weights = list(map(int, line.split()))
                    problem[i] = torch.tensor(weights, dtype=torch.float32)
            
            
                problem.fill_diagonal_(0)     
                # normlization
                scaled_problem = problem / scaler
                batch_problems.append(scaled_problem)

            except Exception as e:
                raise NotImplementedError(f"Unexpected error in loading {filename}")

        if not batch_problems:
            raise NotImplementedError(f"cannot find valid data in {self.path}")

        return torch.stack(batch_problems, dim=0)





