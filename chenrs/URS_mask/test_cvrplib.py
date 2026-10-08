##########################################################################################
# Machine Environment Config
import argparse
import random

import numpy as np
import torch


from multi_hot_set import get_problem_list


##########################################################################################
# Path Config
import os
import sys

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "..")  # for utils


##########################################################################################
# import

import logging
from utils.utils import create_logger, copy_all_src

from CVRPTester_SetXXL import CVRPTester as Tester


def _print_config():
    logger = logging.getLogger('root')
    logger.info('USE_CUDA: {}, CUDA_DEVICE_NUM: {}'.format(USE_CUDA, CUDA_DEVICE_NUM))
    [logger.info(g_key + "{}".format(globals()[g_key])) for g_key in globals().keys() if g_key.endswith('params')]


def seed_everything(seed=3407):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", type=int, default=0, help="CUDA device number to use")
    parser.add_argument("--seed", type=int, default=3407, help="Random seed for reproducibility")

    # environment parameters
    parser.add_argument("--problem_size_list", type=int, nargs='+', default=[100], help="List of problem sizes to consider")
    parser.add_argument("--pomo_size", type=int, default=100, help="POMO size (number of starting points)")

    # model parameters
    parser.add_argument("--embedding_dim", type=int, default=128, help="Embedding dimension for the model")
    parser.add_argument("--encoder_layer_num", type=int, default=12, help="Number of encoder layers in the model")
    parser.add_argument("--ff_hidden_dim", type=int, default=512, help="Hidden dimension for feed-forward layer in the model")
    parser.add_argument("--logit_clipping", type=float, default=50, help="Logit clipping value for the model")
    parser.add_argument("--dist_norm_style", type=str, choices=["all_max", "sep_max", "nonorm"], default="all_max",
                        help="Normalization style for distance in the model")
    parser.add_argument("--eval_type", type=str, default="greedy", help="Evaluation type for the model",choices=['sampling', 'greedy'])
    parser.add_argument("--no_com_bias", action="store_true", help="Whether to use bias in the compatibility layer")
    parser.add_argument("--demand_max1", action="store_false", help="Whether to normalize demand to max 1")

    # Inference parameters
    parser.add_argument("--test_episodes", type=int, default=1000, help="Number of test episodes")
    parser.add_argument("--test_batch_size", type=int, default=1000, help="Batch size for testing")
    parser.add_argument("--use_aug", action="store_true", help="Enable data augmentation during testing")
    parser.add_argument("--aug_factor", type=int, default=128, help="Augmentation factor for testing")
    parser.add_argument("--aug_batch_size", type=int, default=8, help="Batch size for augmented testing")

    opts = parser.parse_args()
    ##########################################################################################

    a= '/public/home/zhoucl/000_NCO_Codes_v20241213/CrossProblem_HeavyEncoder/'
    b = 'a6_AddACVRP_ClearXY_RelationNoScale_Universal_Neural_Routing_Solver/result_models_icam_11Tasks/20250912_013518_demand_max1_0001_11tasks_AddACVRP_ClearXY_c50_encoder12_epoch500_bs128_batch2000'

    # model_path = a+b
    model_path = "./result_models_icam_11Tasks"

    problem_type = "cvrplib"
    opts.aug_factor = 8
    opts.aug_batch_size = 1
    test_problem_list = ['cvrp']



    USE_CUDA = True
    CUDA_DEVICE_NUM = opts.cuda

    # env_params = {
    #     'problem_size_list': [100]
    # }
    env_params = {
        'problem_size': None,
        'pomo_size': None,
    }


    model_params = {
        'embedding_dim': opts.embedding_dim,
        'encoder_layer_num': opts.encoder_layer_num,
        'ff_hidden_dim': opts.ff_hidden_dim,
        'logit_clipping': opts.logit_clipping,
        'dist_norm_style': opts.dist_norm_style,
        'com_bias': not opts.no_com_bias,
        'eval_type': opts.eval_type,
        'demand_max1': True,#opts.demand_max1,
    }
    # global_path = '/public/home/zhoucl/000_NCO_Codes_v20241213/CrossProblem_HeavyEncoder/'
    # saved_folder_1 = 'a3_ClearXY_RelationNoScale_Universal_Neural_Routing_Solver/result_models_icam_10Tasks/'





    tester_params = {
        'use_cuda': USE_CUDA,
        'cuda_device_num': CUDA_DEVICE_NUM,
        'model_load': {
            'path':  model_path,# directory path of pre-trained model and log files saved.
            # 'epoch': 500,  # epoch version of pre-trained model to load.
            'name': "cvrp_checkpoint-300"
        },
        'test_episodes': opts.test_episodes,
        'test_batch_size': opts.test_batch_size,
        'augmentation_enable':  True, #opts.use_aug,
        'aug_factor': opts.aug_factor,
        'aug_batch_size': opts.aug_batch_size,
        'filename': "/home/zhengyp/ych/AAA/a6_20250912/lib_data/CVRPLIB_XXL.txt",
        "detailed_log": True,
    }

    if tester_params['augmentation_enable']:
        tester_params['test_batch_size'] = tester_params['aug_batch_size']
        highlight = f'aug{tester_params["aug_factor"]}'
    else:
        highlight = 'no_aug'

    import pytz
    from datetime import datetime
    process_start_time = datetime.now(pytz.timezone("Asia/Shanghai"))

    com_bias_str = 'com_bias' if not opts.no_com_bias else 'no_com_bias'
    # test_problem_len = len(test_problem_list)
    extra_str = f"{com_bias_str}_{problem_type}tasks"

    # epoch = tester_params['model_load']['epoch']
    epoch = 300
    print(f"使用的model:{tester_params['model_load']['name']}")

    logger_params = {
        'log_file': {
            'desc': f"{highlight}_c{opts.logit_clipping}_{extra_str}_{opts.dist_norm_style}_epoch{epoch}_bs{opts.aug_batch_size}",
            'filename': 'run_log.txt',
            'filepath': './result_test_models_icam/' + process_start_time.strftime("%Y%m%d_%H%M%S") + '{desc}'
        }
    }

    ##########################################################################################
    # main
    seed_everything(opts.seed)
    create_logger(**logger_params)
    _print_config()
    tester = Tester(env_params=env_params,
                      model_params=model_params,
                      tester_params=tester_params,
                      )
    # copy_all_src(tester.result_folder)
    tester.run_lib()
