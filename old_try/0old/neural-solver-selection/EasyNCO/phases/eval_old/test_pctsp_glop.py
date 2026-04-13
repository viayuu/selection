import hydra
import os
import sys
import pytz

from omegaconf import DictConfig
import torch

current_dir = os.path.dirname(os.path.abspath(__file__))
split_path = current_dir.split("/")
root_dir = "/".join(split_path[:-2])
insert_dir = "/".join(split_path[:-3])
sys.path.insert(0, insert_dir)
config_path = f"{root_dir}/configs"

from EasyNCO.data import *
from EasyNCO.utils.utils import set_device, seed_everything
from EasyNCO.phases.train.rl.ar_reinforce import REINFORCELightning
from datetime import datetime
from EasyNCO.methods.main_loop.glop.test_utils import glop_cheak_param


seed = 2024
gpu_device = 2

device = set_device(use_cuda=True,
                    gpu_device=gpu_device)
seed_everything(seed=seed)


first_mode = 'random'
CUDA_LAUNCH_BLOCKING=1
process_start_time = datetime.now(pytz.timezone("Asia/Seoul"))

@hydra.main(config_path = config_path, config_name = 'pctsp_glop_test_config.yaml')
def main(cfg : DictConfig):
    envs = {}
    policys = {}
    glop_cheak_param(cfg)

    for i in cfg.module.glop_params['revision_lens']:
        env_key = f"env{i}"
        env = hydra.utils.instantiate(cfg[env_key], device=device, seed=seed,_batch_locked=False)
        envs[env_key] = env

        revision = hydra.utils.instantiate(cfg.revision_model)
        path = root_dir + getattr(cfg.test_loader.revision_path, f"re{i}_path")
        revision.load_state_dict(torch.load(path, map_location=device))
        policys[f"re{i}"] = revision
    main_net = hydra.utils.instantiate(cfg.main_net)
    main_net.load_state_dict(torch.load(root_dir + cfg.test_loader.main_net_path, map_location=device))
    policys["main_net"]=main_net
    tester = hydra.utils.instantiate(cfg.trainer, devices=[gpu_device],)
    data_path = cfg.test_loader.data_path
    full_data_path = f"{root_dir}{data_path}"
    module = hydra.utils.instantiate(cfg.module,
                                     env = envs,
                                     policy = policys,
                                     test_data_path = full_data_path)
    tester.test(module)




# def load_main_net(cfg):
#     if cfg.module.glop_params.main_problem=="TSP":
#         return None
#     elif cfg.module.glop_params.main_problem=="ATSP":
#         return None
#     elif cfg.module.glop_params.main_problem=="CVRP":
#
#         net=hydra.utils.instantiate(cfg.main_net)
#         net.load_state_dict(torch.load(root_dir + cfg.test_loader.main_net_path, map_location=device))
#         return net
#     elif cfg.module.glop_params.main_problem=="PCTSP":
#         net = hydra.utils.instantiate(cfg.main_net)
#         net.load_state_dict(torch.load(root_dir + cfg.test_loader.main_net_path, map_location=device))
#         return net


if __name__ == '__main__':
    main()