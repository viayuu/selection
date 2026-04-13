import hydra
import os
import sys
from omegaconf import DictConfig

current_dir = os.path.dirname(os.path.abspath(__file__))  # '/public/home/zhoucl/000NCO_Codes_v20230715/EasyNCO/phases/single_obj/train/rl'
split_path = current_dir.split("/")
root_dir = "/".join(split_path[:-2])
insert_dir = "/".join(split_path[:-3])
sys.path.insert(0, insert_dir)
config_path = f"{root_dir}/configs"

from EasyNCO.utils.utils import set_device, seed_everything

seed = 2024
gpu_device = 0

device = set_device(use_cuda=True,
                    gpu_device=gpu_device)
seed_everything(seed=seed)

@hydra.main(config_path=config_path, config_name="pctsp_deepaco_config.yaml")
def train(cfg: DictConfig):
    # env = hydra.utils.instantiate(cfg.env, device=device, seed=seed)
    model = hydra.utils.instantiate(cfg.model)
    module = hydra.utils.instantiate(cfg.module, policy=model)
    logger = hydra.utils.instantiate(cfg.logger)
    checkpoint_callback = hydra.utils.instantiate(cfg.model_checkpoint)

    trainer = hydra.utils.instantiate(
        cfg.trainer,
        devices=[gpu_device],
        callbacks=[checkpoint_callback],
        logger=logger
    )

    trainer.fit(model=module)


if __name__ == "__main__":
    train()
