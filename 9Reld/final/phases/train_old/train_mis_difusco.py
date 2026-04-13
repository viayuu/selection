import hydra
import os

import torch
from omegaconf import DictConfig
import lightning as L
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))  # '/public/home/liuyang/NCO/EasyNCO/phases/single_obj/train'
split_path = current_dir.split("/")
root_dir = "/".join(split_path[:-2])
insert_dir = "/".join(split_path[:-3])
sys.path.insert(0, insert_dir)
config_path = f"{root_dir}/configs"
data_path = f"{root_dir}/data/datasets/test_dataset_mis/er_test/*gpickle"
# data_path = f"{root_dir}/data/datasets/test_dataset_tsp_uniform/test_tsp100_nums10000_uniform.pt"

from EasyNCO.utils.utils import set_device, seed_everything



# pyrootutils.setup_root(__file__, indicator=".gitignore", pythonpath=True)


seed = 2024
gpu_device = 0

device = set_device(use_cuda=False,
                    gpu_device=gpu_device)
L.seed_everything(seed=seed, workers=True)

@hydra.main(config_path=config_path, config_name="mis_difusco_config.yaml")
def train(cfg: DictConfig):
    cfg.module.optimizer_params['scheduler']['milestones'] = [i for i in range(1, cfg.trainer.max_epochs)]
    model = hydra.utils.instantiate(cfg.model)
    module = hydra.utils.instantiate(cfg.module, policy=model, data_path=data_path)
    logger = hydra.utils.instantiate(cfg.logger)
    checkpoint_callback = hydra.utils.instantiate(cfg.model_checkpoint)

    trainer = hydra.utils.instantiate(
        cfg.trainer,
        devices=[gpu_device],
        callbacks=[checkpoint_callback],
        logger=logger
    )

    trainer.fit(model=module, ckpt_path=cfg.get('saved_ckpt'))

if __name__ == "__main__":
    train()