import hydra
import os
import sys
import pytz

from omegaconf import DictConfig
import torch


from datetime import datetime

current_dir = os.path.dirname(os.path.abspath(__file__))
split_path = current_dir.split("/")
root_dir = "/".join(split_path[:-2])
insert_dir = "/".join(split_path[:-3])
sys.path.insert(0, insert_dir)
config_path = f"{root_dir}/configs"
from EasyNCO.data import *
from EasyNCO.utils.utils import set_device, seed_everything


seed = 2024
gpu_device = 2

device = set_device(use_cuda=True,
                    gpu_device=gpu_device)
seed_everything(seed=seed)
first_mode = 'random'
CUDA_LAUNCH_BLOCKING=1
process_start_time = datetime.now(pytz.timezone("Asia/Seoul"))

@hydra.main(config_path = config_path, config_name = 'tsp_pointerformer_config.yaml')
def main(cfg : DictConfig):
    env = hydra.utils.instantiate(cfg.env, device = device, seed = seed)
    tester = hydra.utils.instantiate(cfg.trainer, devices=[gpu_device], )
    policy = hydra.utils.instantiate(cfg.model)
    data_path = cfg.test_loader.data_path
    full_data_path = f"{root_dir}{data_path}"
    model_path = f"{cfg.test_loader.model_dirpath}{cfg.test_loader.model_filename}"
    full_model_path = f"{root_dir}{model_path}"
    policy.load_state_dict(torch.load(full_model_path, map_location=device))

    module = hydra.utils.instantiate(cfg.module,
                                     env = env,
                                     policy = policy,
                                     test_data_path = full_data_path)

    tester.test(module)


if __name__ == '__main__':
    main()