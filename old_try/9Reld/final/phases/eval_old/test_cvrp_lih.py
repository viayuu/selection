import hydra
import os
import sys
import pytz
import re
from omegaconf import DictConfig
import torch

current_dir = os.path.dirname(os.path.abspath(__file__))
split_path = current_dir.split("\\")
root_dir = "/".join(split_path[:-2])
insert_dir = "/".join(split_path[:-3])
sys.path.insert(0, insert_dir)
config_path = f"{root_dir}/configs"


from EasyNCO.data import *
from EasyNCO.utils.utils import set_device, seed_everything
from EasyNCO.phases.train.rl.ar_reinforce import REINFORCELightning
from datetime import datetime


seed = 2024
gpu_device = 0

device = set_device(use_cuda=True,
                    gpu_device=gpu_device)
seed_everything(seed=seed)


first_mode = 'random'
CUDA_LAUNCH_BLOCKING=1
process_start_time = datetime.now(pytz.timezone("Asia/Seoul"))

@hydra.main(config_path = config_path, config_name = 'cvrp_lih_config.yaml')
def main(cfg : DictConfig):
    env = hydra.utils.instantiate(cfg.env, device = device, seed = seed)
    tester = hydra.utils.instantiate(cfg.trainer, devices=[gpu_device], )
    policy = hydra.utils.instantiate(cfg.model)

    data_path = cfg.test_loader.data_path
    full_data_path = f"{root_dir}{data_path}"
    model_path = f"{cfg.test_loader.model_dirpath}{cfg.test_loader.model_filename}"
    full_model_path = f"{root_dir}{model_path}"

    policy = load_model(policy, full_model_path)
    module = hydra.utils.instantiate(cfg.module,
                                     env = env,
                                     policy = policy,
                                     test_data_path = None)

    tester.test(module)

def load_model(model, ckpt_path):
    ckpt = torch.load(ckpt_path)
    model_state_dict = model.state_dict()
    ckpt_dict = ckpt['state_dict']

    for name, param in ckpt_dict.items():
        name = re.sub(r"policy\.", "", name)
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