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

from Tester import Tester as Tester


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
    parser.add_argument("--aug_factor", type=int, default=1, help="Augmentation factor for testing")
    parser.add_argument("--aug_batch_size", type=int, default=1000, help="Batch size for augmented testing")

    opts = parser.parse_args()
    ##########################################################################################

    a= '/public/home/zhoucl/000_NCO_Codes_v20241213/CrossProblem_HeavyEncoder/'
    b = 'a6_AddACVRP_ClearXY_RelationNoScale_Universal_Neural_Routing_Solver/result_models_icam_11Tasks/20250912_013518_demand_max1_0001_11tasks_AddACVRP_ClearXY_c50_encoder12_epoch500_bs128_batch2000'

    # model_path = a+b
    model_path = "./result_models_icam_11Tasks"

    problem_type = "8mdvrp_bp"
    opts.aug_factor = 8
    opts.aug_batch_size = 125
    test_problem_list = ["mdvrpbp", "mdovrpbp", "mdvrpbpl", "mdovrpbptw", "mdvrpbpltw", "mdvrpbptw", "mdovrpbpl", "mdovrpbpltw",]

    # problem_type = "8amdvrp_bp"
    # opts.aug_factor = 128
    # opts.aug_batch_size = 8
    # test_problem_list = ["amdvrpbp", "amdovrpbp", "amdvrpbpl", "amdovrpbptw", "amdvrpbpltw", "amdvrpbptw", "amdovrpbpl",
    #                       "amdovrpbpltw",]

    # problem_type = "24vrp"
    # opts.aug_factor = 8
    # opts.aug_batch_size = 1000
    #
    # test_problem_list = ['vrpb', 'vrpl', 'cvrptw', 'ovrp', 'ovrptw', 'ovrpb', 'vrpbl', 'vrpltw', 'ovrpbtw', 'vrpbltw', 'ovrpl', 'vrpbtw',
    #  'ovrpbl', 'ovrpltw', 'ovrpbltw',"vrpbp", "ovrpbp", "vrpbpl", "ovrpbptw", "vrpbpltw", "vrpbptw", "ovrpbpl", "ovrpbpltw","cvrp"]

    # problem_type = "24avrp"
    # opts.aug_factor = 128
    # opts.aug_batch_size = 50
    # test_problem_list = ['avrpb', 'avrpl', 'acvrptw', 'aovrp', 'aovrptw', 'aovrpb', 'avrpbl', 'avrpltw', 'aovrpbtw', 'avrpbltw', 'aovrpl', 'avrpbtw',
    #  'aovrpbl', 'aovrpltw', 'aovrpbltw',"avrpbp", "aovrpbp", "avrpbpl", "aovrpbptw", "avrpbpltw", "avrpbptw", "aovrpbpl", "aovrpbpltw","acvrp"]

    # problem_type = "amdcvrp"
    # opts.aug_factor = 128
    # opts.aug_batch_size = 8
    # test_problem_list = ['amdcvrp']

    # problem_type = "mdcvrp"
    # opts.aug_factor = 8
    # opts.aug_batch_size = 125
    # test_problem_list = ['mdcvrp']

    # problem_type = "apdp"
    # opts.aug_factor = 128
    # opts.aug_batch_size = 50
    # test_problem_list = ['apdp']

    # problem_type = "in_domain_exclude_A_Scale500"
    # opts.aug_factor = 8
    # opts.aug_batch_size = 32
    #'atsp', 'acvrp'
    # test_problem_list = ['tsp', 'op', 'pctsp', 'cvrp', 'vrpb', 'cvrptw', 'ovrp', 'ovrptw', 'pdp']


    # problem_type = "in_domain_exclude_A_Scale500"
    # opts.aug_factor = 8
    # opts.aug_batch_size = 100
    # test_problem_list = ['tsp', 'atsp', 'cvrp', 'acvrp', 'op', 'pctsp', 'cvrptw', 'ovrp', 'vrpb', 'ovrptw', 'pdp']
    
    # problem_type = "test"
    # opts.aug_factor = 8
    # opts.aug_batch_size = 100
    # test_problem_list = ['atsp','acvrp','tsp','op','pctsp','spctsp', 'pdp','cvrp'] + ['vrpb','vrpl','cvrptw','ovrp','ovrptw', 'ovrpb','vrpbl','vrpltw','ovrpbtw','vrpbltw','ovrpl','vrpbtw','ovrpbl','ovrpltw','ovrpbltw']

    # problem_type = "L0.6_A"
    # opts.aug_factor = 128
    # opts.aug_batch_size = 50
    # test_problem_list = ["avrpl", 'avrpbl', 'avrpltw','avrpbltw', 'aovrpl', 'aovrpbl', 'aovrpltw', 'aovrpbltw',
    #                     "avrpbpl", "avrpbpltw", "aovrpbpl","aovrpbpltw"]

    # problem_type = "L0.6_AMD"
    # opts.aug_factor = 128
    # opts.aug_batch_size = 8
    # test_problem_list = ["amdvrpl", 'amdvrpbl', 'amdvrpltw',
    #                 'amdvrpbltw', 'amdovrpl', 'amdovrpbl', 'amdovrpltw', 'amdovrpbltw',
    #                 "amdvrpbpl", "amdvrpbpltw", "amdovrpbpl",
    #                 "amdovrpbpltw"]

    # problem_type = "apdcvrp"
    # opts.aug_factor = 8
    # opts.aug_batch_size = 1000
    # test_problem_list = ['apdcvrp']

    # problem_type = "sdcvrp_spctsp_pdcvrp_opdcvrp"
    # opts.aug_factor = 8
    # opts.aug_batch_size = 1000
    # test_problem_list = ['sdvrp','spctsp','pdcvrp','opdcvrp']

    # problem_type = "cvrptw"
    # opts.aug_factor = 128
    # opts.aug_batch_size = 50
    # test_problem_list = ['apdp','apdcvrp','aopdcvrp']
    # opts.aug_factor = 1
    # opts.aug_batch_size = 16
    # test_problem_list = ['pctsp']



    USE_CUDA = True
    CUDA_DEVICE_NUM = opts.cuda

    env_params = {
        'problem_size_list': [100]
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
            'path':  '/public/home/chenrs/project/Mask/URS_mask/result_models_icam_11Tasks/20251202_124132_demand_max1_0001_11tasks_AddACVRP_ClearXY_c50_encoder12_epoch500_bs128_batch2000',# directory path of pre-trained model and log files saved.
            # 'epoch': 500,  # epoch version of pre-trained model to load.
            'name': "checkpoint-500"
        },
        'test_episodes': opts.test_episodes,
        'test_batch_size': opts.test_batch_size,
        'augmentation_enable':  True, #opts.use_aug,
        'aug_factor': opts.aug_factor,
        'aug_batch_size': opts.aug_batch_size,
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
    trainer = Tester(env_params=env_params,
                      model_params=model_params,
                      tester_params=tester_params,
                      test_problem_list=test_problem_list,
                      use_mask_embedding=True)
    # copy_all_src(trainer.result_folder)
    trainer.run()
