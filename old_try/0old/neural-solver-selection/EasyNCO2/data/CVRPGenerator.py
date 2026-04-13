import os

import numpy as np
import random
import torch
from torch.utils.data import Dataset, DataLoader
from EasyNCO.utils.utils import getLogger
from EasyNCO.data.data_utils import get_gaussian_mixture,generate_by_distribution,generate_explosion_instance,generate_implosion_instance
from EasyNCO.data.LIBUtils import CVRPLIBReader
from EasyNCO.data.data_utils import normalize_to_unit_board

logger = getLogger(__name__)

####################################
# DATA
####################################
def CVRPGenerator(data_size, problem_size, batch_size, device = 'cpu', path = None, demand_scaler=None, partial_updater=None,**kwargs):
    scale_range = kwargs.get("scale_range", None)
    use_triangular_capacity = kwargs.get("use_triangular_capacity", False)
    triangular_params = kwargs.get("triangular_params", [3, 6, 25])
    
    # 🔥 ReLD移植：支持data_distribution参数解析
    data_distribution = kwargs.get("data_distribution", None)
    if data_distribution is not None:
        # 解析论文数据分布设置，例如："uniform_40_100" -> [40, 100]
        if "_" in data_distribution:
            parts = data_distribution.split("_")
            if len(parts) >= 3 and parts[1] == "40" and parts[2] == "100":
                scale_range = [int(parts[1]), int(parts[2])]
                logger.info(f"🔥 Parsed data_distribution: {data_distribution} -> scale_range: {scale_range}")
            else:
                logger.warning(f"⚠️ Unsupported data_distribution format: {data_distribution}")
        else:
            # 保持原有的scale_range逻辑
            pass
    

    if scale_range is not None and scale_range[0] != scale_range[1]:
        batch_size = 1
    if path is None:
        dataset = random_vrp_generator(num_sample = data_size,
                                           num_nodes = problem_size,
                                           demand_scaler = demand_scaler,
                                           device = device,
                                           **kwargs)
    elif "cvrplib" in path:
        dataset = cvrplib_loader(device=device,
                                    path=path,
                                    **kwargs)
    elif scale_range is not None:
        dataset = varying_vrp_loader(
            num_sample=data_size, device=device, path=path, **kwargs
        )
    else:
        dataset = customized_vrp_loader(num_sample = data_size,
                                        num_nodes = problem_size,
                                        device = device,
                                        path=path)
        if partial_updater is not None:
            dataset.data['to_DataLoader'] = partial_updater.load_problem_one_epoch(dataset)
            batch_size = 1
    data_loader = DataLoader(dataset = dataset,
                             batch_size = batch_size,
                             shuffle = False,
                             num_workers = 0,
                             collate_fn = None)

    return data_loader

class random_vrp_generator(Dataset):
    '''
    The random_vrp_generator is used to generate the random VRP data.
    The data is generated randomly with the size of [num_sample, num_nodes, demand, 3]
    By default, the data is generated with the uniform distribution.
    # num_modes and cdist are used for gaussian_mixture
    # explosion_args include [range_min , range_max , rate]
    # implosion_args include [range_min , range_max]
    '''
    def __init__(self, num_sample, num_nodes, demand_scaler=None, device='cpu',use_triangular_capacity=False, triangular_params=[3, 6, 25],**kwargs):
        self.num_sample = num_sample
        self.num_nodes = num_nodes
        self.device = device
        self.demand_scaler = demand_scaler
        self.use_triangular_capacity = use_triangular_capacity
        self.triangular_params = triangular_params
        self.distribution = kwargs.get('distribution','uniform')
        self.num_modes = kwargs.get('num_modes',0)
        self.cdist = kwargs.get('cdist',0)
        self.explosion_args = kwargs.get('explosion_args',None)
        self.implosion_args = kwargs.get('implosion_args',None)
        self.problem_size_low = kwargs.get("problem_size_low", None)
        self.problem_size_high = kwargs.get("problem_size_high", None)

    def __getitem__(self, index):
        if self.problem_size_low:
            num_nodes = (
                np.random.randint(
                    self.problem_size_low // self.num_nodes,
                    self.problem_size_high // self.num_nodes + 1,
                )
                * self.num_nodes
            )
            self.demand_scaler = np.random.randint(100, 200 + 1)  # one scalar
        else:
            num_nodes = self.num_nodes
        depot_xy = torch.rand(size = (1, 2)).to(self.device)
        if self.distribution == 'uniform':
            node_xy_data = torch.rand(size = (num_nodes, 2)).to(self.device)
        elif self.distribution == "gaussian_mixture":
            node_xy_data = get_gaussian_mixture(self.num_nodes, self.num_modes, self.cdist).to(self.device)
        elif self.distribution in ["uniform_rectangle","gaussian","cluster","diagonal"]:
            # Generate data using the method in omni
            self.demand_scaler = 30 + self.num_nodes/5 if self.num_nodes >= 20 else 20
            node_xy_data = generate_by_distribution(self.num_nodes, self.distribution).to(self.device)
        elif self.distribution == "explosion":
            #Generate data using the method in Invit
            if self.explosion_args is not None:
                range_min,range_max,rate = self.explosion_args['range_min'],self.explosion_args['range_max'],self.explosion_args['rate']
                depot_xy, node_xy_data = generate_explosion_instance(self.num_nodes, range_min, range_max, rate,depot_xy=depot_xy).to(self.device)
            else:
                depot_xy, node_xy_data = generate_explosion_instance(self.num_nodes,depot_xy=depot_xy).to(self.device)
        elif self.distribution == "implosion":
            # Generate data using the method in Invit
            if self.implosion_args is not None:
                range_min,range_max = self.implosion_args['range_min'],self.explosion_args['range_max']
                depot_xy, node_xy_data = generate_implosion_instance(self.num_nodes,range_min,range_max,depot_xy=depot_xy).to(self.device)
            else:
                depot_xy, node_xy_data = generate_implosion_instance(self.num_nodes,depot_xy=depot_xy).to(self.device)
        else:
            raise ValueError(f"Unsupported distribution type: {self.distribution}")

        if self.use_triangular_capacity:
            # T(3,6,25)三角分布实现
            route_length = np.random.triangular(*self.triangular_params)
            demand_raw = torch.randint(1, 10, size = (num_nodes + 1,))
            capacity = torch.ceil(torch.tensor(route_length) * demand_raw.sum() / num_nodes)
            demand = demand_raw / capacity
            demand[0] = 0
        else:
            if self.demand_scaler is not None:
                demand_scaler = self.demand_scaler
            else:
                if self.num_nodes == 20:
                    demand_scaler = 30
                elif self.num_nodes == 50:
                    demand_scaler = 40
                elif self.num_nodes == 100:
                    demand_scaler = 50
                elif self.num_nodes == 200:
                    demand_scaler = 70
                elif self.num_nodes == 300:
                    demand_scaler = 90
                elif self.num_nodes == 500:
                    demand_scaler = 130
                elif self.num_nodes == 1000:
                    demand_scaler = 200
                elif self.num_nodes == 2000:
                    demand_scaler = 300
                elif self.num_nodes == 5000:
                    demand_scaler = 300
                elif self.num_nodes == 7000:
                    demand_scaler = 300
                else:
                    raise ValueError(f"The default demand_scaler is not supported for the problem size of {self.num_nodes}, " +
                                    f"please specify the demand_scaler")

            demand = torch.randint(1, 10, size = (num_nodes + 1,)) / float(demand_scaler)
            demand[0] = 0
        demand = demand.unsqueeze(1).to(self.device)
        # shape (problem, 1)

        node_vrp_data = torch.cat([depot_xy, node_xy_data], dim = 0)
        node_vrp_data = torch.cat([node_vrp_data, demand], dim = 1)
        # shape:(problem+1,3) x,y,demand

        return node_vrp_data
        # shape: (problem+1, 3), including depot and customer nodes

    def __len__(self):
        return self.num_sample

class customized_vrp_loader(Dataset):
    '''
    The customized_vrp_loader is used to load the customized VRP data from the file.
    The file should be saved in the format of .pkl or .pt.
    And the data should be saved in the format of {'depot_xy': shape (sample_num, 1, 2),
                                                    'node_xy': shape (sample_num, problem, 2),
                                                    'node_demand': shape (sample_num, problem),
                                                    (if learning paradigm is sl, 'label' will be contained)}
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
                out = np.array(pickle.load(f)[:self.num_sample], dtype=object)
                saved_depot_xy = torch.tensor(out[:, 0].tolist(), dtype=torch.float32).to(device)
                if saved_depot_xy.shape == (self.num_sample, 2):
                    saved_depot_xy = saved_depot_xy[:, None, :]
                # shape: (batch, 1, 2)
                saved_node_xy = torch.tensor(out[:, 1].tolist(), dtype=torch.float32).to(device) # shape: (batch, problem, 2)
                demand = torch.tensor(out[:, 2].tolist(), dtype=torch.float32) # shape: (batch, problem)
                capacity = float(out[0, 3])
                saved_node_demand = demand / capacity

        elif self.file_type == '.pt':
            loaded_dict = torch.load(self.path, map_location=self.device)
            saved_depot_xy = loaded_dict['depot_xy'][:self.num_sample]  # shape: (batch, 1, 2)
            saved_node_xy = loaded_dict['node_xy'][:self.num_sample]  # shape: (batch, problem, 2)
            saved_node_demand = loaded_dict['node_demand'][:self.num_sample]  # shape: (batch, problem)
            if 'label' in loaded_dict.keys():
                self.data = {'label': loaded_dict['label'][:self.num_sample]}
        else:
            raise NotImplementedError(f"The file type ({self.file_type}) is not supported in current version")

        depot_node_demand = torch.cat((torch.zeros(size=(self.num_sample, 1)), saved_node_demand), dim=1)  # shape: (batch, 1+num_nodes)
        depot_node_xy = torch.cat((saved_depot_xy, saved_node_xy), dim=1)  # shape: (batch, 1+num_nodes, 2)
        data = torch.cat((depot_node_xy, depot_node_demand[:, :, None]), dim=2)
        if isinstance(self.data, dict) and 'label' in self.data:
            self.data['node_xy_demand'] = torch.cat((depot_node_xy, depot_node_demand[:, :, None]), dim=2)
        else:
            self.data = torch.cat((depot_node_xy, depot_node_demand[:, :, None]), dim=2)  # shape: (batch, 1+num_nodes, 3)

    def __getitem__(self, index):
        if isinstance(self.data, dict) and 'to_DataLoader' in self.data.keys():
            return self.data['to_DataLoader'][index]
        else:
            return self.data[index]

    def __len__(self):
        if isinstance(self.data, dict) and 'to_DataLoader' in self.data.keys():
            return len(self.data['to_DataLoader'])
        else:
            return self.num_sample


class varying_vrp_loader(Dataset):

    def __init__(self, num_sample, device="cpu", path=None, **kwargs):
        self.num_sample = num_sample
        self.device = device
        self.path = path
        self.file_type = os.path.splitext(self.path)[-1]  # maybe .pkl or .pt
        self.data = None
        self.scale_range = kwargs.get("scale_range")  # e.g. [500, 800]
        
        # 🔥 ReLD移植：优先使用data_distribution解析的scale_range
        if self.scale_range is None and hasattr(self, 'data_distribution') and self.data_distribution is not None:
            # 如果data_distribution被解析，使用解析出的scale_range
            self.scale_range = self.data_distribution_parsed_range
            logger.info(f"🔥 Using parsed scale_range from data_distribution: {self.scale_range}")
        elif self.scale_range is None:
            # 如果没有scale_range，尝试从data_distribution解析
            if hasattr(self, 'data_distribution') and self.data_distribution is not None:
                self._parse_and_store_data_distribution(kwargs.get("data_distribution", None))
        
        self.distribution = kwargs.get(
            "distribution_list", None
        )  # e.g. ['uniform', 'cluster']
        self.capacity_range = kwargs.get("capacity_range", None)  # e.g. [100, 200]

        if self.file_type == ".pt":
            self.data = torch.load(self.path, map_location=self.device)
            
            # 🔥 ReLD移植：处理不同格式的测试数据文件
            if "scale_list" in self.data:
                # 格式1：包含scale_list, distribution_list, capacity_list的格式
                scale_list = self.data["scale_list"]
                distribution_list = self.data["distribution_list"]
                capacity_list = self.data["capacity_list"]

                # Find indices that satisfy both scale range and distribution conditions
                self.valid_indices = []
                for i, (scale, dist, capa) in enumerate(
                    zip(scale_list, distribution_list, capacity_list)
                ):
                    valid = True
                    # Check scale range condition
                    if not (self.scale_range[0] <= scale <= self.scale_range[1]):
                        valid = False

                    # Check distribution condition
                    if self.distribution is not None and dist not in self.distribution:
                        valid = False

                    # Check capacity range condition
                    if self.capacity_range is not None and not (
                        self.capacity_range[0] <= capa <= self.capacity_range[1]
                    ):
                        valid = False

                    if valid:
                        self.valid_indices.append(i)
                # random sample
                if self.num_sample < len(self.valid_indices):
                    self.valid_indices = random.sample(self.valid_indices, self.num_sample)
                else:
                    self.num_sample = len(self.valid_indices)
            elif "depot_xy" in self.data or "node_xy" in self.data:
                # 格式2：标准的CVRP测试数据格式（depot_xy, node_xy, node_demand）
                logger.info(f"🔥 Using standard CVRP test data format from {self.path}")
                self.valid_indices = list(range(min(self.num_sample, self._get_data_length())))
                self.use_standard_format = True
            else:
                # 格式3：其他格式，尝试直接使用
                logger.warning(f"⚠️ Unknown data format in {self.path}, attempting to use directly")
                self.valid_indices = list(range(min(self.num_sample, self._get_data_length())))
                self.use_standard_format = True
        else:
            raise NotImplementedError(
                f"The file type ({self.file_type}) is not supported in current version"
            )

    def _get_data_length(self):
        """获取数据长度，支持不同格式的数据文件"""
        if hasattr(self, 'use_standard_format') and self.use_standard_format:
            # 标准格式：检查数据的第一维
            if "depot_xy" in self.data:
                return len(self.data["depot_xy"])
            elif "node_xy" in self.data:
                return len(self.data["node_xy"])
            elif isinstance(self.data, torch.Tensor):
                return self.data.shape[0]
            else:
                return 1  # 默认返回1
        elif "dataset" in self.data:
            # 旧格式：dataset字典
            return len(self.data["dataset"])
        else:
            return 1  # 默认返回1

    def __getitem__(self, index):
        if hasattr(self, 'use_standard_format') and self.use_standard_format:
            # 🔥 ReLD移植：处理标准CVRP测试数据格式
            return self._get_standard_format_item(index)
        else:
            # 原有的varying_vrp_loader格式
            index = self.valid_indices[index]
            depot_node_xy = torch.tensor(
                        [self.data["dataset"][f"num{index}"]["depot_xy"]]
                        + self.data["dataset"][f"num{index}"]["node_xy"],
                        dtype=torch.float32,
                    )
            depot_node_demand = torch.cat(
                (
                    torch.zeros(size=(1, 1), dtype=torch.float32),
                    torch.tensor(
                        self.data["dataset"][f"num{index}"]["node_demand"],
                        dtype=torch.float32,
                    )[:, None],
                ),
                dim=0,
            ) / self.data["dataset"][f"num{index}"]["capacity"]
            return {
                "data": torch.cat(
                    (depot_node_xy, depot_node_demand), dim=1
                ),  # shape: (1+num_nodes, 3)
                "optimal": torch.tensor(
                    self.data["dataset"][f"num{index}"]["cost"], dtype=torch.float32
                ),
                "solution": torch.tensor(
                    self.data["dataset"][f"num{index}"]["solution"], dtype=torch.int64
                ),
                "num_sample": self.num_sample,
            }

    def _get_standard_format_item(self, index):
        """处理标准格式的CVRP数据"""
        if "depot_xy" in self.data and "node_xy" in self.data and "node_demand" in self.data:
            # 标准格式：depot_xy, node_xy, node_demand
            depot_xy = self.data["depot_xy"][index].to(self.device)
            node_xy = self.data["node_xy"][index].to(self.device)
            node_demand = self.data["node_demand"][index].to(self.device)
            
            # 假设capacity为50（根据配置文件中的test_vrp100_capacity50）
            capacity = 50.0
            
            # 构建完整数据
            depot_node_xy = torch.cat([depot_xy, node_xy], dim=0)
            depot_node_demand = torch.cat([
                torch.zeros(1, dtype=torch.float32, device=self.device),
                node_demand
            ], dim=0) / capacity
            
            return torch.cat([depot_node_xy, depot_node_demand.unsqueeze(1)], dim=1)
        elif isinstance(self.data, torch.Tensor):
            # 张量格式：直接返回
            return self.data[index].to(self.device)
        else:
            raise ValueError(f"Unsupported data format for standard CVRP test data")

    def __len__(self):
        return self.num_sample


class cvrplib_loader(Dataset):
    def __init__(self, device="cpu", path=None, **kwargs):
        self.device = device
        self.path = path
        self.data = {}
        self.scale_range = kwargs.get("scale_range", None)  # e.g. [500, 800]
        self.capacity_range = kwargs.get("capacity_range", None)  # e.g. [100, 200]
        self.scale_min_max = [float("inf"), float("-inf")]
        self.num_sample = 0
        for root, dirs, files in os.walk(self.path):
            for file in files:
                if file.endswith(".vrp"):
                    name, dimension, locs, demand, capacity, cost = CVRPLIBReader(
                        os.path.join(root, file)
                    )
                    if name is None:
                        continue
                    if self.scale_range is not None and not (self.scale_range[0] <= dimension <= self.scale_range[1]):
                        continue
                    if self.capacity_range is not None and not (self.capacity_range[0] <= capacity <= self.capacity_range[1]):
                        continue
                    self.data[name] = {"dimension": dimension, "locations": locs, "demand": demand, "capacity": capacity, "cost": cost}
                    self.num_sample += 1
                    self.scale_min_max[0] = min(self.scale_min_max[0], dimension)
                    self.scale_min_max[1] = max(self.scale_min_max[1], dimension)
        if self.num_sample == 0:
            raise ValueError(f"No VRP files found in {self.path} within the specified scale range {self.scale_range} and capacity range {self.capacity_range}")

    def __getitem__(self, index):
        name = list(self.data.keys())[index]
        data, factor = normalize_to_unit_board(
            torch.tensor(self.data[name]["locations"], dtype=torch.float32).to(
                self.device
            )
        )
        demand = (
            torch.tensor(self.data[name]["demand"], dtype=torch.float32).to(self.device)
            / self.data[name]["capacity"]
        )

        return {
            "name": name,
            "data": torch.cat(
                (data, demand[:, None]), dim=1
            ),  # shape: (num_nodes, 3) including x, y, demand
            "factor": factor,
            "optimal": torch.tensor(self.data[name]["cost"], dtype=torch.float32).to(
                self.device
            ),
            "num_sample": self.num_sample,
        }

    def __len__(self):
        return self.num_sample
