import time

import numpy as np
import torch

import os
from logging import getLogger

from tqdm import tqdm

# from Env.CVRPLIBEnv import CVRPEnv as Env
from Env.UNIEnv import UniEnv as Env
from Model import Model as Model

from utils.utils import *
from multi_hot_set import *
from ProblemDef import augment_xy_data_by_8_fold
import math

class CVRPTWLibTester:
    def __init__(self,
                 env_params,
                 model_params,
                 tester_params):

        # save arguments
        self.env_params = env_params
        self.model_params = model_params
        self.tester_params = tester_params

        # result folder, logger
        self.logger = getLogger(name='trainer')
        self.result_folder = get_result_folder()

        self.problem_attributes_set = get_problem_attributes_set()

        # cuda
        USE_CUDA = self.tester_params['use_cuda']
        if USE_CUDA:
            cuda_device_num = self.tester_params['cuda_device_num']
            torch.cuda.set_device(cuda_device_num)
            device = torch.device('cuda', cuda_device_num)
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            device = torch.device('cpu')
            torch.set_default_tensor_type('torch.FloatTensor')
        self.device = device

        # ENV and MODEL
        self.env = Env(**self.env_params)
        self.model = Model(**self.model_params)

        # Restore
        model_load = tester_params['model_load']
        if "epoch" in model_load.keys():
            checkpoint_fullname = '{path}/checkpoint-{epoch}.pt'.format(**model_load)
        else:
            checkpoint_fullname = '{path}/{name}.pt'.format(**model_load)

        checkpoint = torch.load(checkpoint_fullname, map_location=device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.logger.info("Model loaded successfully!!!")
        self.logger.info("Model loaded from: {}".format(checkpoint_fullname))

        self.path_list = [os.path.join(tester_params['cvrptwlib_dir'], f) for f in
                          sorted(os.listdir(tester_params['cvrptwlib_dir']))] \
            if os.path.isdir(tester_params['cvrptwlib_dir']) else [tester_params['cvrptwlib_dir']]
        assert self.path_list[-1].endswith(".vrp") or self.path_list[-1].endswith(".txt"), "Unsupported file types."

        # utility
        self.time_estimator = TimeEstimator()


    def run(self):
        for path in self.path_list:
            self._solve_cvrptwlib(path)



    #from mvmoe
    def _solve_cvrptwlib(self,path):
        """
            Solving one instance with VRPTW benchmark (e.g., Solomon) format.
        """
        model = self.model


        file = open(path, "r")
        lines = [ll.strip() for ll in file]
        i = 0
        while i < len(lines):
            line = lines[i]
            if line.startswith("NUMBER"):
                line = lines[i + 1]
                vehicle_number = int(line.split(' ')[0])  # TODO: check vehicle number constraint
                capacity = int(line.split(' ')[-1])
            elif line.startswith("CUST NO."):
                data = np.loadtxt(lines[i + 1:], dtype=int)
                break
            i += 1
        original_locations = data[:, 1:3]
        original_locations = np.expand_dims(original_locations, axis=0)  # [1, n+1, 2]


        # norm 1
        # node_coord = torch.Tensor(original_locations)
        # xy_max = torch.max(node_coord, dim=1, keepdim=True).values
        # xy_min = torch.min(node_coord, dim=1, keepdim=True).values
        # # shape: (1, 1, 2)
        # scaler = torch.max((xy_max - xy_min), dim=-1, keepdim=True).values
        # scaler[scaler == 0] = 1
        # # shape: (1, 1, 1)
        # nodes_xy_normalized = (node_coord - xy_min) / scaler.expand(-1, 1, 2)
        # # shape: (1, dimension+1,2)
        # depot_xy, node_xy = nodes_xy_normalized[:, :1, :],nodes_xy_normalized[:, 1:, :]
        # node_demand = torch.Tensor(data[1:, 3].reshape((1, -1))) / capacity  # [1, n]
        #
        # scaler = scaler[0,0,0]
        # service_time = torch.Tensor(data[1:, -1].reshape((1, -1))) / scaler  # [1, n]
        # tw_start = torch.Tensor(data[1:, 4].reshape((1, -1))) / scaler  # [1, n]
        # tw_end = torch.Tensor(data[1:, 5].reshape((1, -1))) / scaler   # [1, n]


        # norm 2
        node_coord = torch.Tensor(original_locations)
        instance_xy = original_locations.flatten()
        max_value = np.max(instance_xy)
        min_value = np.min(instance_xy)
        nodes_xy_normalized = node_coord / max_value
        scaler = max_value
        depot_xy,node_xy = nodes_xy_normalized[:, :1, :],nodes_xy_normalized[:, 1:, :]
        node_demand = torch.Tensor(data[1:, 3].reshape((1, -1))) / capacity  # [1, n]
        service_time = torch.Tensor(data[1:, -1].reshape((1, -1))) / scaler  # [1, n]
        tw_start = torch.Tensor(data[1:, 4].reshape((1, -1))) / scaler  # [1, n]
        tw_end = torch.Tensor(data[1:, 5].reshape((1, -1))) / scaler  # [1, n]


        #norm 3
        # spatial_width = original_locations[:, :, 0].max() - original_locations[:, :, 0].min()
        # spatial_height = original_locations[:, :, 1].max() - original_locations[:, :, 1].min()
        # spatial_scale = (spatial_width + spatial_height) / 2
        # tw_scale = data[0, 5] / 3.
        # scaler = max(spatial_scale, tw_scale)


        # norm 4
        # coords = torch.tensor(original_locations.reshape(-1, 2), dtype=torch.float32)
        # cov = torch.cov(coords.T)
        # scale = torch.sqrt(torch.det(cov))  # 空间分布体积尺度
        # tw_scale = data[0, 5] / 3.
        # scaler = float(max(scale, tw_scale))



        # norm 5
        # coords = torch.tensor(original_locations.reshape(-1, 2), dtype=torch.float32)
        # spatial_scale = torch.sqrt((coords ** 2).mean())
        # tw_scale = data[0, 5] / 3.
        # scaler = max(spatial_scale, tw_scale)

        # norm 6
        # coords = torch.tensor(original_locations.reshape(-1), dtype=torch.float32)
        # spatial_scale = torch.quantile(coords, 0.95)
        # tw_scale = data[0, 5] / 3.
        #
        # scaler = float(max(spatial_scale, tw_scale))
        # locations = original_locations / scaler
        # depot_xy, node_xy = torch.Tensor(locations[:, :1, :]), torch.Tensor(locations[:, 1:, :])
        # node_demand = torch.Tensor(data[1:, 3].reshape((1, -1))) / capacity  # [1, n]
        # service_time = torch.Tensor(data[1:, -1].reshape((1, -1))) / scaler  # [1, n]
        # tw_start = torch.Tensor(data[1:, 4].reshape((1, -1))) / scaler  # [1, n]
        # tw_end = torch.Tensor(data[1:, 5].reshape((1, -1))) / scaler  # [1, n]



        # mvmoe_norm
        # scaler = max(original_locations.max(), data[0, 5] / 3.)  # we set the time window of the depot node as [0, 3]
        #
        # assert original_locations.max() <= scaler, ">> Scaler is too small for {}".format(path)
        # locations = original_locations / scaler  # [1, n+1, 2]: Scale location coordinates to [0, 1]
        # depot_xy, node_xy = torch.Tensor(locations[:, :1, :]), torch.Tensor(locations[:, 1:, :])
        # node_demand = torch.Tensor(data[1:, 3].reshape((1, -1))) / capacity  # [1, n]
        # service_time = torch.Tensor(data[1:, -1].reshape((1, -1))) / scaler  # [1, n]
        # tw_start = torch.Tensor(data[1:, 4].reshape((1, -1))) / scaler  # [1, n]
        # tw_end = torch.Tensor(data[1:, 5].reshape((1, -1))) / scaler  # [1, n]
        #
        env_params = {'problem_size': node_xy.size(1), 'pomo_size': node_xy.size(1), 'loc_scaler': scaler,
                      'device': self.device}

        # print(env_params['problem_size'])

        self.env.depot_end = data[0, 5] / scaler
        print(path)
        print(self.env.depot_end)
        self.env.loc_scaler = scaler
        data_dict = {
            'depot_xy':depot_xy,
            'node_xy':node_xy,
            'node_demand':node_demand,
            'service_time':service_time,
            'tw_start':tw_start,
            'tw_end':tw_end,
        }

        no_aug_score, aug_score = self._test_one_batch(1,data_dict,env_params) #batch_size = 1



        # Check distance
        original_locations = torch.Tensor(original_locations)
        depot_xy = augment_xy_data_by_8_fold(original_locations[:, :1, :])
        node_xy = augment_xy_data_by_8_fold(original_locations[:, 1:, :])
        original_locations = torch.cat((depot_xy, node_xy), dim=1)
        gathering_index = self.env.selected_node_list[:, :, :, None].expand(-1, -1, -1, 2)
        all_xy = original_locations[:, None, :, :].expand(-1, env_params["pomo_size"], -1, -1)
        ordered_seq = all_xy.gather(dim=2, index=gathering_index)
        rolled_seq = ordered_seq.roll(dims=2, shifts=-1)
        segment_lengths = ((ordered_seq - rolled_seq) ** 2).sum(3).sqrt()
        travel_distances = segment_lengths.sum(2)
        # print(travel_distances.min())

        no_aug_score = torch.round(no_aug_score * scaler).long()
        aug_score = torch.round(aug_score * scaler).long()

        # print(self.env.tw_end[0][0])
        print(">> Finish solving {} -> no_aug: {} aug: {} real: {}".format(path, no_aug_score, aug_score,
                                                                           travel_distances.min()))


        return no_aug_score, aug_score

    def _test_one_batch(self, batch_size, dict_instance_info,env_params):

        # Augmentation
        ###############################################
        problem_size = env_params['problem_size']
        if self.tester_params['augmentation_enable']:
            aug_factor = self.tester_params['aug_factor']
            pomo_size = env_params['pomo_size']
        else:
            aug_factor = 1
            pomo_size = env_params['pomo_size']

        problem_name = 'cvrptw'
        problem_representation = torch.tensor(self.problem_attributes_set[problem_name], dtype=torch.float32,
                                              device=self.device)

        # Ready
        ###############################################
        self.model.eval()
        with torch.no_grad():

            self.env.load_problems(batch_size, problem_size, pomo_size=pomo_size,
                                   lib_data=dict_instance_info,
                                   aug_factor=aug_factor, device=self.device,
                                   problem_name=problem_name)
            reset_state, _, _ = self.env.reset()

            self.model.decoder.assign(problem_representation)
            self.model.pre_forward(reset_state, problem_name, problem_representation)

            ###############################################
            state, reward, done = self.env.pre_step()
            with tqdm(total=0) as pbar:
                while not done:
                    cur_dist = self.env.get_local_feature()
                    selected, _ = self.model(state, cur_dist)
                    # shape: (batch, pomo)
                    state, reward, done = self.env.step(selected, lib_mode=True)
                    pbar.total += 1
                    pbar.update(1)

        # Return
        ###############################################
        aug_reward = reward.reshape(aug_factor, batch_size, self.env.pomo_size)
        # shape: (augmentation, batch, pomo)

        max_pomo_reward, _ = aug_reward.max(dim=2)  # get best results from pomo
        # shape: (augmentation, batch)
        no_aug_score = -max_pomo_reward[0, :].float().mean()  # negative sign to make positive value

        max_aug_pomo_reward, _ = max_pomo_reward.max(dim=0)  # get best results from augmentation
        # shape: (batch,)
        aug_score = -max_aug_pomo_reward.float().mean()  # negative sign to make positive value

        return no_aug_score, aug_score