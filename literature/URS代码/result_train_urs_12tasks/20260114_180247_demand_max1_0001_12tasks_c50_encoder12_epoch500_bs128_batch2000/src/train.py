##########################################################################################
# Machine Environment Config
import argparse
import random

import numpy as np
import torch



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

from Trainer import Trainer as Trainer
from args import obtain_all_settings

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

    parser = argparse.ArgumentParser(description="Train a unified model for multiple VRP variants.")
    obtain_all_settings(parser)
    opts = parser.parse_args()

    ##########################################################################################
    # 11 training tasks, ensuring each attribute is covered at least twice, except for 'penalty', which is covered once.
    # Because we observe that the model can generalize well to penalty-related tasks (e.g., PCTSP, SPCTSP) when "penalty" is covered once.
    train_problem_list = ['atsp','acvrp','tsp','op','pctsp','cvrp', 'cvrpb', 'cvrptw', 'ocvrp', 'ocvrptw', 'pdtsp','acvrpbtw'] # 11 tasks
    test_problem_list = train_problem_list[:]
    # 11 additional tasks, following MVMoE
    multi_task_list_num15 = ['cvrpl','ocvrpb','cvrpbl','cvrpltw','ocvrpbtw','cvrpbltw','ocvrpl','cvrpbtw','ocvrpbl','ocvrpltw','ocvrpbltw']
    test_problem_list.extend(multi_task_list_num15)
    # train_problem_list = ['acvrp'] # for debugging
    # test_problem_list = train_problem_list


    env_params = {
        'problem_size': opts.problem_size,
        'capacity': opts.capacity,
    }

    model_params = {
        'embedding_dim': opts.embedding_dim,
        'encoder_layer_num': opts.encoder_layer_num,
        'ff_hidden_dim': opts.ff_hidden_dim,
        'logit_clipping': opts.logit_clipping,
        'demand_max1': not opts.no_demand_max1,
        'eval_type': opts.eval_type,

    }

    optimizer_params = {
        'optimizer_type': opts.optimizer_type,  # 'AdamW'
        'optimizer': {
            'lr': opts.optimizer_lr,
            'weight_decay': opts.weight_decay,
        },
        'lr_decay_epoch': opts.lr_decay_epoch,
    }

    trainer_params = {
        'use_cuda': True if opts.cuda >= 0 and torch.cuda.is_available() else False,
        'cuda_device_num': opts.cuda,
        'epochs': opts.training_epochs,
        'batches_per_epoch': opts.batches_per_epoch,
        'batch_size': opts.batch_size,
        'logging': {
            'model_save_interval': 1, # save model per epoch
            'log_image_params_1': {
                'json_foldername': 'log_image_style',
                'filename': 'style_score.json'
            },
            'log_image_params_2': {
                'json_foldername': 'log_image_style',
                'filename': 'style_loss.json'
            },
        },
        'model_load': {
            'enable': False,  # enable loading pre-trained model
            #'path': '',  # directory path of pre-trained model and log files saved.
            #'epoch': ,  # epoch version of pre-trained model to load.
        },
    }


    import pytz
    from datetime import datetime
    process_start_time = datetime.now(pytz.timezone("Asia/Shanghai"))

    lr_str = str(opts.optimizer_lr).replace('0.', '')
    demand_max_str = 'demand_max1' if model_params['demand_max1'] else 'no_demand_max1'
    train_tasks_len = len(train_problem_list)
    extra_str = f"{demand_max_str}_{lr_str}_{train_tasks_len}tasks"

    logger_params = {
        'log_file': {
            'desc': f"{extra_str}_c{opts.logit_clipping}_encoder{opts.encoder_layer_num}_epoch{opts.training_epochs}_bs{opts.batch_size}_batch{opts.batches_per_epoch}",
            'filename': 'run_log.txt',
            'filepath': f'./result_train_urs_{train_tasks_len}tasks/' + process_start_time.strftime("%Y%m%d_%H%M%S") + '{desc}'
        }
    }


    ##########################################################################################
    # main
    seed_everything(opts.seed)
    create_logger(**logger_params)
    _print_config(CUDA_DEVICE_NUM=opts.cuda)
    trainer = Trainer(env_params=env_params,
                      model_params=model_params,
                      optimizer_params=optimizer_params,
                      trainer_params=trainer_params,
                      train_problem_list=train_problem_list,
                      test_problem_list=test_problem_list)
    copy_all_src(trainer.result_folder)
    trainer.run()
