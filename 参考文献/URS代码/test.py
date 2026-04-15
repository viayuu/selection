##########################################################################################
# Machine Environment Config
import argparse
import random

import numpy as np
import torch

from 参考文献.URS代码.multi_hot_set import get_problem_list


##########################################################################################
# Path Config
import os
import sys

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "..")  # for utils


##########################################################################################
# import

import logging
from 参考文献.URS代码.utils.utils import create_logger, copy_all_src
from 参考文献.URS代码.args import obtain_all_settings

def _print_config(CUDA_DEVICE_NUM=0):
    USE_CUDA = True if CUDA_DEVICE_NUM >=0 and torch.cuda.is_available() else False
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

    parser = argparse.ArgumentParser(description="Test a unified model for multiple VRP variants.")
    obtain_all_settings(parser)
    opts = parser.parse_args()
    ##########################################################################################

    # a= '/public/home/zhoucl/000_NCO_Codes_v20241213/CrossProblem_HeavyEncoder/'
    # b = 'a6_AddACVRP_ClearXY_RelationNoScale_Universal_Neural_Routing_Solver/result_models_icam_11Tasks/20250912_013518_demand_max1_0001_11tasks_AddACVRP_ClearXY_c50_encoder12_epoch500_bs128_batch2000'

    # model_path = a+b
    model_path = '/public/home/zhoucl/000_NCO_Codes_v20241213/CrossProblem_HeavyEncoder/a7_SpecificTrain_AddACVRP_ClearXY_RelationNoScale/STL_Models'#"./result_models_icam_11Tasks"

    problem_type = opts.problem_type
    if problem_type.endswith('_list'): # detailed problem list can be found in multi_hot_set.py
        from 参考文献.URS代码.Tester import Tester as Tester
        test_problem_list = get_problem_list(problem_type)
    elif problem_type == 'tsplib':
        from 参考文献.URS代码.TSPTester_LIB import TSPTester as Tester
        opts.aug_batch_size = 1
        test_problem_list = ['tsp']
    elif problem_type == 'cvrplib_xxl':
        from 参考文献.URS代码.CVRPTester_SetXXL import CVRPTester as Tester
        opts.aug_batch_size = 1
        test_problem_list = ['cvrp']
    else:
        # customized problem set, not belonging to prepared problem sets, the name can be freely defined.
        from 参考文献.URS代码.Tester import Tester as Tester
        test_problem_list = ['acvrp']

    env_params = {
        'problem_size_list': opts.problem_size_list,
    }

    model_params = {
        'embedding_dim': opts.embedding_dim,
        'encoder_layer_num': opts.encoder_layer_num,
        'ff_hidden_dim': opts.ff_hidden_dim,
        'logit_clipping': opts.logit_clipping,
        'eval_type': opts.eval_type,
        'demand_max1': not opts.no_demand_max1,
    }

    tester_params = {
        'use_cuda': True if opts.cuda >= 0 and torch.cuda.is_available() else False,
        'cuda_device_num': opts.cuda,
        'model_load': {
            'path':  model_path,# directory path of pre-trained model and log files saved.
            #'epoch': 500,  # epoch version of pre-trained model to load.
            'name': "atsp_checkpoint-300"
        },
        'test_episodes': opts.test_episodes,
        'test_batch_size': opts.test_batch_size,
        'augmentation_enable':  not opts.disable_aug,
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

    test_problem_len = len(test_problem_list)
    extra_str = f"{problem_type}_{test_problem_len}tasks"

    logger_params = {
        'log_file': {
            'desc': f"{highlight}_c{opts.logit_clipping}_{extra_str}_bs{opts.aug_batch_size}",
            'filename': 'run_log.txt',
            'filepath': './result_test_urs/' + process_start_time.strftime("%Y%m%d_%H%M%S") + '{desc}'
        }
    }

    ##########################################################################################
    # main
    seed_everything(opts.seed)
    create_logger(**logger_params)
    _print_config(CUDA_DEVICE_NUM=opts.cuda)
    trainer = Tester(env_params=env_params,
                      model_params=model_params,
                      tester_params=tester_params,
                      test_problem_list=test_problem_list)
    copy_all_src(trainer.result_folder)
    trainer.run()
