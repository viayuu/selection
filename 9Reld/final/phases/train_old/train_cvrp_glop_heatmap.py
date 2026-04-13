import torch
import hydra
import os
from omegaconf import DictConfig
import lightning as L
import sys
import pyrootutils

current_dir = os.path.dirname(os.path.abspath(__file__))  # '/public/home/zhoucl/000NCO_Codes_v20230715/EasyNCO/phases/single_obj/train/rl'
split_path = current_dir.split("/")
root_dir = "/".join(split_path[:-2]) # '/public/home/zhoucl/000_NCO_Codes_v20240822/EasyNCO'
insert_dir = "/".join(split_path[:-3])  # '/public/home/zhoucl/000_NCO_Codes_v20240822'
sys.path.insert(0, insert_dir)
config_path = f"{root_dir}/configs"

from EasyNCO.utils.utils import set_device, seed_everything



# pyrootutils.setup_root(__file__, indicator=".gitignore", pythonpath=True)


seed = 2024
gpu_device = 2

device = set_device(use_cuda=True,
                    gpu_device=gpu_device)
# seed_everything(seed=seed)
L.seed_everything(seed=seed, workers=True)

@hydra.main(config_path=config_path, config_name="cvrp_heatmap_glop_train.yaml")
def train(cfg: DictConfig):
    main_net = hydra.utils.instantiate(cfg.main_net)
    revision_model= hydra.utils.instantiate(cfg.revision_model)
    path = root_dir + getattr(cfg.policy_path, f"re{cfg.module.revision_len}_path")
    revision_model.load_state_dict(torch.load(path, map_location=device))
    env = hydra.utils.instantiate(cfg.env, device=device, seed=seed)

    module = hydra.utils.instantiate(cfg.module,env=env,main_net=main_net,revision_model=revision_model)
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
