
import argparse
import sys

import numpy as np
import torch
import os
from logging import warning

from torch.utils.data import DataLoader

current_dir = os.path.dirname(os.path.abspath(__file__))  # '/public/home/zhoucl/000NCO_Codes_v20230715/EasyNCO/phases/single_obj/train/rl'
split_path = current_dir.split("/")
root_dir = "/".join(split_path[:-1]) # '/public/home/zhoucl/000_NCO_Codes_v20240822/EasyNCO'
insert_dir = "/".join(split_path[:-2])  # '/public/home/zhoucl/000_NCO_Codes_v20240822'
sys.path.insert(0, insert_dir)
config_path = f"{root_dir}/configs"

from tqdm import tqdm
from EasyNCO.utils.utils import set_device, seed_everything
from EasyNCO.data.data_utils import glop_coordinate_transformation
from EasyNCO.methods.main_loop.initialization.insertion import random_insertion
from EasyNCO.methods.operators.neural_operators.am.policy import AttentionModelPolicy
from EasyNCO.methods.envs import TSPEnv

seed = 1235
gpu_device = 2

device = set_device(use_cuda=True,
                    gpu_device=gpu_device)

def decomposition(data0, coordinate_dim, revision_len, offset, shift_len = 1):
    # change decomposition point
    # data0 = torch.cat([data0[:, shift_len:],data0[:, :shift_len]], 1)
    data0=data0.roll(dims=1, shifts=-shift_len)
    if offset!=0:
        decomposed_seeds = data0[:, :-offset]
        offset_seeds = data0[:,-offset:]
    else:
        decomposed_seeds = data0
        offset_seeds = None
    # decompose original seeds
    decomposed_seeds = decomposed_seeds.reshape(-1, revision_len, coordinate_dim)#
    return decomposed_seeds, offset_seeds


def solve_and_decompose(instance,target_len):
    # instance shape: (problem size, 2); dtype: torch tensor
    problem_size, _ = instance.shape
    pi,cost=random_insertion(instance)
    instance = instance.unsqueeze(0)
    pi = torch.tensor(pi.astype(np.int64))
    seed = instance.gather(1, pi.unsqueeze(-1).expand_as(instance))

    batch_size, num_nodes, coordinate_dim = seed.shape
    revision_len = revision_iter = target_len
    offset = num_nodes % revision_len  # 剩余的子问题
    dataset = None
    for shift in range(revision_iter):
        decomposed_seeds, offset_seed = decomposition(seed,
                                                      coordinate_dim,
                                                      revision_len,
                                                      offset,
                                                      shift)
        if dataset is None:
            dataset = decomposed_seeds
        else:
            dataset = torch.cat((dataset, decomposed_seeds), dim=0)
    # print('dataset_ shape:', dataset.shape)
    return dataset

def generate_RI_data(dataset_size,original_size, target_problem_size):
    batch_size = (original_size // target_problem_size) * target_problem_size
    dataset = None
    for _ in tqdm(range(dataset_size // batch_size)):
        instance = torch.rand((original_size, 2))
        dataset_ = solve_and_decompose(instance,target_problem_size)
        dataset_=glop_coordinate_transformation(dataset_)
        # coordinate transformation
        if dataset is None:
            dataset = dataset_
        else:
            dataset = torch.cat((dataset, dataset_), dim=0)
        # return shape: (dataset size, problem size, 2);
    print('dataset shape:', dataset.shape)
    return dataset


def main():
    parser = argparse.ArgumentParser(description="stage2_using_Random_insertert")
    # 通用参数
    parser.add_argument('--save_dir', type=str, default=f'{root_dir}/data/glop_stage2_train_data',
                        help="save_dir")
    parser.add_argument('--dataset_size', type=int, default=1000000,
                        help="dataset_size")
    parser.add_argument('--ori_problem_size', type=int, default=500,
                        help="stage2 original_size")
    parser.add_argument('--target_problem_szie', type=int, default=100,
                        help="stage2 target_problem_szie")
    args = parser.parse_args()

    data = generate_RI_data(args.dataset_size, args.ori_problem_size, args.target_problem_szie)
    file_name = f"glop_stage2_{args.ori_problem_size}to{args.target_problem_szie}(RI)_{args.dataset_size}.pkl"

    import pickle
    os.makedirs(args.save_dir, exist_ok=True)
    file_path = os.path.join(args.save_dir, file_name)
    with open(file_path, 'wb') as f:
        pickle.dump(data.cpu().numpy(), f)


if __name__ == "__main__":
    main()
