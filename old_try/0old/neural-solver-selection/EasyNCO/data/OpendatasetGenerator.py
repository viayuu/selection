import torch
from torch.utils.data import Dataset, DataLoader

class TSPLibDataset(Dataset):
    def __init__(self, data):
        # data's format: {'name': <list of string>, 'node_num': <tensor> 'problem': <list of tensor>, 'optimal': <tensor>}
        # The meaning of TSPLib's keys:
        #       name: The name of each problem.
        #       node_num: The number of nodes in each problem.
        #       problem: cvrp problems, the first node is depot.
        #       optimal: The optimal cost of each problem.
        self.data = data
        self.instance_num = 93
        self.current_problem_name = None # Show which problems in running.

        self.scaled_data = None
        self.scale_limit_max = None
        self.scale_limit_min = None
        self.scale_instance_num = None

    def __getitem__(self, index):
        if self.scaled_data is not None:
            self.current_problem_name = self.scaled_data['name'][index]
            data = self.scaled_data['problem'][index]
            optimal = self.scaled_data['optimal'][index]
        else:
            self.current_problem_name = self.data['name'][index]
            data = self.data['problem'][index]
            optimal = self.data['optimal'][index]
        return self.current_problem_name, data, optimal

    def __len__(self):
        return self.instance_num if self.scaled_data is None else self.scale_instance_num

    def get_scaled_problem(self, scale_limit_max: int, scale_limit_min: int = 0):
        scale_index = ((self.data['node_num'] <= scale_limit_max) & (self.data['node_num'] >= scale_limit_min)).nonzero().squeeze()
        if scale_index.size == 0:
            raise ValueError(f'TSPLib doesn\'t contain problems on a scale of {scale_limit_min} to {scale_limit_max}.')
        self.scaled_data = {}
        self.scale_limit_max = scale_limit_max
        self.scale_limit_min = scale_limit_min

        self.scaled_data['node_num'] = self.data['node_num'][scale_index]
        self.scaled_data['optimal'] = self.data['optimal'][scale_index]
        self.scaled_data['name'] = [self.data['name'][index] for index in scale_index.tolist()]
        self.scaled_data['problem'] = [self.data['problem'][index] for index in scale_index.tolist()]

    def get_all_problem(self):
        self.scaled_data = None
        self.scale_limit_max = None
        self.scale_limit_min = None
        self.scale_instance_num = None

class NationalTSPDataset(Dataset):
    def __init__(self, data):
        # NationalTSP only supports one data format: 'coordination'.
        # data's format: {'name': <list of string>, 'node_num':<tensor>, 'problem': <list of tensor>, 'optimal': <tensor>}
        # The meaning of NationalTSP's keys:
        #       name: The name of each problem.
        #       node_num: The number of nodes in each problem.
        #       problem: cvrp problems, the first node is depot.
        #       optimal: The optimal cost of each problem.
        self.data = data
        self.instance_num = 29
        self.current_problem_name = None # Show which problems in running.

        self.scaled_data = None
        self.scale_limit_max = None
        self.scale_limit_min = None
        self.scale_instance_num = None

    def __getitem__(self, index):

        if self.scaled_data is not None:
            self.current_problem_name = self.scaled_data['name'][index]
            data = self.scaled_data['problem'][index]
            optimal = self.scaled_data['optimal'][index]
        else:
            self.current_problem_name = self.data['name'][index]
            data = self.data['problem'][index]
            optimal = self.data['optimal'][index]
        return self.current_problem_name, data, optimal

    def __len__(self):
        return self.instance_num if self.scaled_data is None else self.scale_instance_num

    def get_scaled_problem(self, scale_limit_max: int, scale_limit_min: int = 0):
        scale_index = ((self.data['node_num'] <= scale_limit_max) & (self.data['node_num'] >= scale_limit_min)).nonzero().squeeze()
        if scale_index.size == 0:
            raise ValueError(f'NationalTSP doesn\'t contain problems on a scale of {scale_limit_min} to {scale_limit_max}.')
        self.scaled_data = {}
        self.scale_limit_max = scale_limit_max
        self.scale_limit_min = scale_limit_min

        self.scaled_data['node_num'] = self.data['node_num'][scale_index]
        self.scaled_data['optimal'] = self.data['optimal'][scale_index]
        self.scaled_data['name'] = [self.data['name'][index] for index in scale_index.tolist()]
        self.scaled_data['problem'] = [self.data['problem'][index] for index in scale_index.tolist()]

    def get_all_problem(self):
        self.scaled_data = None
        self.scale_limit_max = None
        self.scale_limit_min = None
        self.scale_instance_num = None

class CVRPLibDataset(Dataset):
    def __init__(self, data):
        # data's format in CVRPLib: {'name': <list of string>, 'node_num': <tensor>, 'problem': <list of tensor>,
        #                            'capacity': <tensor>, 'optimal': <tensor>, 'tour': <list of tensor>}
        # The meaning of CVRPLib's keys:
        #       name: The name of each problem.
        #       node_num: The number of nodes in each problem.
        #       problem: cvrp problems, the first node is depot.
        #       capacity: The capacity of each problem.
        #       optimal: The optimal cost of each problem.
        #       tour: The optimal solution of each problem.
        self.data = data
        self.instance_num = 253
        self.current_problem_name = None # Show which problems in running.

        self.scaled_data = None
        self.scale_limit_max = None
        self.scale_limit_min = None
        self.scale_instance_num = None

    def __getitem__(self, index):
        if self.scaled_data is not None:
            self.current_problem_name = self.scaled_data['name'][index]
            problem = self.scaled_data['problem'][index]
            capacity = self.scaled_data['capacity'][index]
            optimal = self.scaled_data['optimal'][index]
            tour = self.scaled_data['tour'][index]

            problem[:, 2] = problem[:, 2] / capacity
        else:
            self.current_problem_name = self.data['name'][index]
            problem = self.data['problem'][index]
            capacity = self.data['capacity'][index]
            optimal = self.data['optimal'][index]
            tour = self.data['tour'][index]

            problem[:, 2] = problem[:, 2]/capacity
        return self.current_problem_name, problem, optimal, tour

    def __len__(self):
        return self.instance_num if self.scaled_data is None else self.scale_instance_num

    def get_scaled_problem(self, scale_limit_max: int, scale_limit_min: int = 0):
        scale_index = ((self.data['node_num'] <= scale_limit_max) & (self.data['node_num'] >= scale_limit_min)).nonzero().squeeze()
        if scale_index.size == 0:
            raise ValueError(f'CVRPLib doesn\'t contain problems on a scale of {scale_limit_min} to {scale_limit_max}.' )
        self.scaled_data = {}
        self.scale_limit_max = scale_limit_max
        self.scale_limit_min = scale_limit_min

        self.scaled_data['node_num'] = self.data['node_num'][scale_index]
        self.scaled_data['capacity'] = self.data['capacity'][scale_index]
        self.scaled_data['optimal'] = self.data['optimal'][scale_index]
        self.scaled_data['name'] = [self.data['name'][index] for index in scale_index.tolist()]
        self.scaled_data['problem'] = [self.data['problem'][index] for index in scale_index.tolist()]
        self.scaled_data['tour'] = [self.data['tour'][index] for index in scale_index.tolist()]

    def get_all_problem(self):
        self.scaled_data = None
        self.scale_limit_max = None
        self.scale_limit_min = None
        self.scale_instance_num = None

def get_public_dataset(name_of_public_dataset: str) -> DataLoader:
    return torch.load('./datasets/open_dataset/' + name_of_public_dataset + '_dataloader.pt')