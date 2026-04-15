import hydra
import os
import sys
import pytz

from omegaconf import DictConfig
from ot.backend import torch

from EasyNCO.data import *
from EasyNCO.phases.train.train_tsp_difusco import data_path
from EasyNCO.utils.utils import set_device, seed_everything
from EasyNCO.phases.train.sl.nar_supervise import SUPERVISELightningNAR
from datetime import datetime


seed = 2024
gpu_device = 0

device = set_device(use_cuda=True,
                    gpu_device=gpu_device)
seed_everything(seed=seed)

current_dir = os.path.dirname(os.path.abspath(__file__))
split_path = current_dir.split("/")
root_dir = "/".join(split_path[:-2])
insert_dir = "/".join(split_path[:-5])
sys.path.insert(0, insert_dir)
config_path = f"{root_dir}/configs"
first_mode = 'random'
CUDA_LAUNCH_BLOCKING=1
process_start_time = datetime.now(pytz.timezone("Asia/Seoul"))

@hydra.main(config_path = config_path, config_name = 'tsp_t2t_config.yaml')
def main(cfg : DictConfig):
    model_path = f"{cfg.test_loader.model_dirpath}{cfg.test_loader.model_filename}"
    full_model_path = f"{root_dir}{model_path}"
    tester = hydra.utils.instantiate(cfg.trainer, devices=[gpu_device])
    policy = hydra.utils.instantiate(cfg.model)
    path = cfg.test_loader.data_path
    data_path = f"{root_dir}{path}"
    heatmap_path = f'{root_dir}{cfg.test_loader.heatmap_path}{process_start_time.strftime("%Y%m%d_%H%M%S")}'
    other_params = {'heatmap_path' : heatmap_path}

    policy = load_model(policy, full_model_path)
    module = hydra.utils.instantiate(cfg.module,
                                     policy = policy,
                                     batch_size = cfg.test_loader.test_params['test_batch_size'],
                                     episodes = cfg.test_loader.test_params['test_episodes'],
                                     data_path = data_path,
                                     other_params = other_params,)

    tester.test(module)

def load_model(model, ckpt_path):
    ckpt = torch.load(ckpt_path)
    model_state_dict = model.state_dict()

    for name, param in ckpt['state_dict'].items():
        if name in model_state_dict:
            if model_state_dict[name].shape == param.shape:
                model_state_dict[name].copy_(param)
            else:
                print(f"Shape mismatch for {name}: "
                      f"model {model_state_dict[name].shape}, ckpt {param.shape}")
        else:
            print(f"Key {name} not found in the model's state_dict.")
    model.load_state_dict(model_state_dict)

    return model


if __name__ == '__main__':
    main()