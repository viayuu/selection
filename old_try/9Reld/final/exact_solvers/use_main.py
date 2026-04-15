import os
import sys
import hydra
from omegaconf import DictConfig, OmegaConf

root_dir = os.path.dirname(os.path.abspath(__file__))  # '{your_path}/EasyNCO/exact_solvers'
normalized_path = os.path.normpath(root_dir)  # Normalize the path to avoid issues with different separators
split_path = normalized_path.split(os.sep) # Split the path using the system-specific separator
insert_dir = os.sep.join(split_path[:-2])  # your_path
sys.path.insert(0, insert_dir) # insert the path to sys.path and make sure it is the first one
insert_dir = os.sep.join(split_path[:-1])  # your_path/EasyNCO
sys.path.insert(0, insert_dir) # insert the path to sys.path and make sure it is the first one
dataset_path = os.sep.join([root_dir, "data", "datasets"])

from EasyNCO.data import CVRPGenerator
from EasyNCO.data import TSPGenerator
from EasyNCO.data.ATSPGenerator import ATSPGenerator
from EasyNCO.utils.utils import *
from EasyNCO.exact_solvers.exact_solver_main import exact_solver_main

config = {}

@hydra.main(config_path=root_dir, config_name='configs.yaml')
def get_params(cfg : DictConfig):
    global config 
    config = cfg

if __name__ == "__main__":
    get_params()
    cfg = config
    if cfg.solver_mode == 'multi':
        # INPUT: dataloader
        data_list = []
        if cfg.ptype == 'tsp':
            data_loader = TSPGenerator(data_size=cfg.problem_number, problem_size=cfg.problem_scale,
                                        batch_size=cfg.batch_size, device=cfg.device, path=None)
            for batch_idx, batch in enumerate(data_loader):
                for instance in batch:
                    node_coords = [(xy[0].item(), xy[1].item()) for xy in instance]
                    data_list.append(node_coords)
        elif cfg.ptype == 'cvrp':
            data_loader = CVRPGenerator(data_size=cfg.problem_number, problem_size=cfg.problem_scale,
                                        batch_size=cfg.batch_size, device=cfg.device, path=None,
                                        demand_scaler=cfg.demand_scale)
            for batch_idx, batch in enumerate(data_loader):
                for instance in batch:
                    depot = instance[0, 0:2]
                    locs = instance[1:, 0:2]
                    demand = instance[1:, 2]
                    capacity = cfg.capacity

                    demand = demand.tolist()
                    locs = locs.tolist()
                    depot = depot.tolist()
                    data_list.append([depot, locs, demand, capacity])

        elif cfg.ptype == 'atsp':
            data_loader = ATSPGenerator(data_size=cfg.problem_number, problem_size=cfg.problem_scale,
                                        batch_size=cfg.batch_size, device=cfg.device, path=None)
            for batch_idx, batch in enumerate(data_loader):
                for instance in batch:
                    node_matrix = instance
                    data_list.append(node_matrix)
        exact_solver_main(cfg, data_list=data_list)

    elif cfg.solver_mode == 'single':
        # INPUT: node coords. Using node coords directly rather than dataloader!
        if cfg.ptype == 'tsp':
            data_loader = TSPGenerator(data_size=cfg.problem_number, problem_size=cfg.problem_scale,
                                       batch_size=cfg.batch_size, device=cfg.device, path=None)
            for batch_idx, batch in enumerate(data_loader):
                node_coords = batch[0, :, 0:2]
            node_coords = node_coords.tolist()

            exact_solver_main(cfg, node_coords = node_coords)

        elif cfg.ptype == 'cvrp':
            data_loader = CVRPGenerator(data_size=cfg.problem_number, problem_size=cfg.problem_scale,
                                        batch_size=cfg.batch_size, device=cfg.device, path=None,
                                        demand_scaler=cfg.demand_scale)
            for batch_idx, batch in enumerate(data_loader):
                depot = batch[0, 0, 0:2]
                locs = batch[0, 1:, 0:2]
                import torch
                demand = batch[0, 1:, 2]
                num_nodes = len(locs)
                capacity = cfg.capacity
            demand = demand.tolist()
            locs = locs.tolist()
            depot = depot.tolist()
            exact_solver_main(cfg, depot=depot, locs=locs, demand=demand, capacity=capacity)
        
        elif cfg.ptype == 'atsp':
            data_loader = ATSPGenerator(data_size=cfg.problem_number, problem_size=cfg.problem_scale,
                                        batch_size=cfg.batch_size, device=cfg.device, path=None)
            for batch_idx, batch in enumerate(data_loader):
                for instance in batch:
                    node_matrix = instance
            exact_solver_main(cfg, node_matrix=node_matrix)