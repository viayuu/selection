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
from utils.utils import create_logger, copy_all_src

from Trainer_DAR import Trainer as Trainer

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
    parser.add_argument("--min_problem_size", type=int, default=100, help="Minimum problem size for training")
    parser.add_argument("--max_problem_size", type=int, default=500, help="Maximum problem size for training")
    parser.add_argument("--min_capacity", type=int, default=50, help="Minimum capacity for training")
    parser.add_argument("--max_capacity", type=int, default=50, help="Maximum capacity for training")

    # model parameters
    parser.add_argument("--embedding_dim", type=int, default=128, help="Embedding dimension for the model")
    parser.add_argument("--encoder_layer_num", type=int, default=12, help="Number of encoder layers in the model")
    parser.add_argument("--ff_hidden_dim", type=int, default=512, help="Hidden dimension for feed-forward layer in the model")
    parser.add_argument("--logit_clipping", type=float, default=50, help="Logit clipping value for the model")
    # parser.add_argument("--dist_norm_style", type=str, choices=["all_max", "sep_max", "nonorm"], default="all_max",
    #                     help="Normalization style for distance in the model")
    parser.add_argument("--eval_type", type=str, default="sampling", help="Evaluation type for the model",choices=['sampling', 'greedy'])
    # parser.add_argument("--relation_scale", action="store_true", help="Whether to use additional linear for three attns")
    parser.add_argument("--demand_max1", action="store_true", help="Whether to normalize demand to max 1")

    # optimizer parameters
    parser.add_argument("--optimizer_type", type=str, default="AdamW", help="Optimizer type for the model",choices=['AdamW', 'Adam'])
    parser.add_argument("--optimizer_lr", type=float, default=1e-4, help="Learning rate for the optimizer")
    parser.add_argument("--weight_decay", type=float, default=1e-6, help="Weight decay for the optimizer")
    parser.add_argument("--lr_decay_epoch", type=int, nargs='+', default=[451], help="Epochs at which to decay the learning rate")

    # Training parameters
    parser.add_argument("--training_epochs", type=int, default=500, help="Total epochs for training")
    parser.add_argument("--batches_per_epoch", type=int, default=2000, help="Steps per epoch")
    parser.add_argument("--stage1_epochs", type=int, default=400, help="Epochs for stage 1")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for training in stage 1")
    parser.add_argument("--vst_base_bs", type=int, default=None, help="Base batch size for varying-scale Training, can be adjusted based on GPU memory")

    opts = parser.parse_args()
    ##########################################################################################
    train_problem_list = ['atsp','acvrp','tsp','op','pctsp','cvrp', 'vrpb', 'cvrptw', 'ovrp', 'ovrptw', 'pdp']
    test_problem_list = ['atsp','acvrp','tsp','op','pctsp','spctsp', 'pdp','cvrp','sdvrp']
    multi_task_list_num15 = ['vrpb','vrpl','cvrptw','ovrp','ovrptw', 'ovrpb','vrpbl','vrpltw','ovrpbtw','vrpbltw','ovrpl','vrpbtw','ovrpbl','ovrpltw','ovrpbltw']
    test_problem_list.extend(multi_task_list_num15)
    # train_problem_list = ['cvrp'] # for debugging
    # test_problem_list = ['cvrp']
    USE_CUDA = True
    CUDA_DEVICE_NUM = opts.cuda
    assert opts.max_problem_size >= opts.min_problem_size, "max_problem_size must be greater than or equal to min_problem_size"
    # assert opts.stage1_epochs + opts.stage3_epochs <= opts.training_epochs, "stage1_epochs + stage3_epochs must be less than or equal to training_epochs"

    env_params = {
        'min_problem_size': opts.min_problem_size,
        'max_problem_size': opts.max_problem_size,

        'min_capacity': opts.min_capacity,
        'max_capacity': opts.max_capacity,
    }

    model_params = {
        'embedding_dim': opts.embedding_dim,
        'encoder_layer_num': opts.encoder_layer_num,
        'ff_hidden_dim': opts.ff_hidden_dim,
        'logit_clipping': opts.logit_clipping,
        'demand_max1': opts.demand_max1,
        'eval_type': opts.eval_type,

    }

    optimizer_params = {
        'optimizer_type': opts.optimizer_type,  # 'AdamW' or 'Adam'
        'optimizer': {
            'lr': opts.optimizer_lr,
            'weight_decay': opts.weight_decay,
        },
        'lr_decay_epoch': opts.lr_decay_epoch,
    }

    trainer_params = {
        'use_cuda': USE_CUDA,
        'cuda_device_num': CUDA_DEVICE_NUM,
        'epochs': opts.training_epochs,
        'batches_per_epoch': opts.batches_per_epoch,
        'stage1_epochs': opts.stage1_epochs,
        'stage1_batch_size': opts.batch_size,
        'vst_base_batch_size': opts.vst_base_bs,
        'logging': {
            'model_save_interval': 50, # save model per epoch
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
            'desc': f"{extra_str}_AddACVRP_ClearXY_c{opts.logit_clipping}_encoder{opts.encoder_layer_num}_epoch{opts.training_epochs}_bs{opts.batch_size}_batch{opts.batches_per_epoch}",
            'filename': 'run_log.txt',
            'filepath': './result_models_icam_11Tasks/' + process_start_time.strftime("%Y%m%d_%H%M%S") + '{desc}'
        }
    }


    ##########################################################################################
    # main
    seed_everything(opts.seed)
    create_logger(**logger_params)
    _print_config()
    trainer = Trainer(env_params=env_params,
                      model_params=model_params,
                      optimizer_params=optimizer_params,
                      trainer_params=trainer_params,
                      train_problem_list=train_problem_list,
                      test_problem_list=test_problem_list)
    copy_all_src(trainer.result_folder)
    trainer.run()
