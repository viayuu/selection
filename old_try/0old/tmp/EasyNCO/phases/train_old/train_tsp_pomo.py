import torch
import hydra
import os
import sys
from omegaconf import DictConfig

import pyrootutils

current_dir = os.path.dirname(os.path.abspath(__file__))  # '/public/home/zhoucl/000NCO_Codes_v20230715/EasyNCO/phases/single_obj/train/rl'
split_path = current_dir.split("\\")
root_dir = "/".join(split_path[:-2]) # '/public/home/zhoucl/000_NCO_Codes_v20240822/EasyNCO'
insert_dir = "/".join(split_path[:-3])  # '/public/home/zhoucl/000_NCO_Codes_v20240822'
sys.path.insert(0, insert_dir)
config_path = f"{root_dir}/configs"
val_data_path = f'{root_dir}/data/datasets/test_dataset_tsp_uniform/test_tsp100_nums10000_uniform.pt'

from EasyNCO.utils.utils import set_device, seed_everything


seed = 2024
gpu_device = 0

device = set_device(use_cuda=True,
                    gpu_device=gpu_device)
seed_everything(seed=seed)

@hydra.main(config_path=config_path, config_name="tsp_pomo_config.yaml")
def train(cfg: DictConfig):
    env = hydra.utils.instantiate(cfg.env, device=device, seed=seed)
    val_env = hydra.utils.instantiate(cfg.env, device = device, seed = seed)
    model = hydra.utils.instantiate(cfg.model)
    module = hydra.utils.instantiate(cfg.module, env=env, policy=model, do_val = True, val_data_path = val_data_path, val_env = val_env)
    logger = hydra.utils.instantiate(cfg.logger)
    checkpoint_callback = hydra.utils.instantiate(cfg.model_checkpoint)

    trainer = hydra.utils.instantiate(
        cfg.trainer,
        devices=[gpu_device],
        # devices = 1,
        callbacks=[checkpoint_callback],
        logger=logger
    )

    trainer.fit(model=module)


if __name__ == "__main__":
    train()
