
import argparse
import math
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
from EasyNCO.data.TSPGenerator import TSPGenerator
from EasyNCO.methods.operators.neural_operators.glop.sub_tsp_utils import tsp_decompose_and_solve
seed = 1235
gpu_device = 2

device = set_device(use_cuda=True,
                    gpu_device=gpu_device)

def generate_RG_data(model_path,data_path,dataset_size,original_size,batch_size,target_sub_size):
    perIns_genNum=original_size//target_sub_size
    num=math.ceil(dataset_size/perIns_genNum)
    dataloader=TSPGenerator(num, original_size, batch_size, device=device, path=data_path)
    sub_policy=AttentionModelPolicy(env_name='tsp',num_heads= 8,qkv_dim=16,num_encoder_layers= 6,normalization='batch',feedforward_hidden= 512,
                                    logit_clipping=10,use_graph_mean= True,am_mode=False,first_placeholder=True,sub_glop=True)
    sub_policy.load_state_dict(torch.load(model_path, map_location=device))
    sub_policy.to(device)
    env = TSPEnv(problem_size=original_size, pomo_size=2,device=device,aug_type='pomo_aug', aug_factor=4)
    new_datas=None
    for batch in tqdm(dataloader, desc="Processing batches", unit="batch"):
        problem=batch.clone()
        solution=torch.arange(original_size)[None,:].repeat(batch_size,1)
        solution=tsp_decompose_and_solve(problem, solution, sub_policy, env, original_size, iter=1, decoder_strategy="greedy")
        #reshape
        offset=original_size%target_sub_size
        if offset:
            new_data=problem.gather(1,solution[:,:-offset,None].repeat(1,1,2))
        else:
            new_data=problem.gather(1,solution[:,:,None].repeat(1,1,2))

        new_data=new_data.reshape(-1,target_sub_size,2)
        new_data=glop_coordinate_transformation(new_data)
        if new_datas==None:
            new_datas=new_data
        else:
            new_datas = torch.cat((new_datas, new_data), dim=0)

    print(new_datas.shape)
    return new_datas



def main():
    parser = argparse.ArgumentParser(description="stage2_using_sub_TSP_model")

    parser.add_argument('--save_dir', type=str, default=f'{root_dir}/data/glop_stage2_train_data',help="save_dir")
    parser.add_argument('--dataset_size', type=int, default=1000000,help="stage2 data_size")
    parser.add_argument('--ori_problem_size', type=int, default=50,help="stage2 original_size")
    parser.add_argument('--target_problem_size', type=int, default=20,help="stage2 target_problem_size")
    parser.add_argument('--batch_size', type=int, default=100,help="batch_size")
    parser.add_argument('--model_path', type=str, default=f"{root_dir}/pretrain/glop_revision/revision_50.pth",
                        help="model path")
    parser.add_argument('--stage2_load_data_path', type=str, default=f"{root_dir}/data/glop_stage2_train_data/glop_stage2_100to50(RG)_1000000.pkl",
                        help="stage2_load_data_path")

    args = parser.parse_args()
    data = generate_RG_data(args.model_path, args.stage2_load_data_path, args.dataset_size, args.ori_problem_size,
                            args.batch_size, args.target_problem_size)
    file_name = f"glop_stage2_{args.ori_problem_size}to{args.target_problem_size}(RG)_{args.dataset_size}.pkl"

    import pickle
    os.makedirs(args.save_dir, exist_ok=True)
    file_path = os.path.join(args.save_dir, file_name)
    with open(file_path, 'wb') as f:
        pickle.dump(data.cpu().numpy(), f)



if __name__ == "__main__":
    main()
