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

from EasyNCO.phases.train.rl.ar_reinforce import REINFORCELightning


# pyrootutils.setup_root(__file__, indicator=".gitignore", pythonpath=True)


seed = 2024
gpu_device = 2

device = set_device(use_cuda=True,
                    gpu_device=gpu_device)
# seed_everything(seed=seed)
L.seed_everything(seed=seed, workers=True)

@hydra.main(config_path=config_path, config_name="sub_tsp_glop_train_stage2.yaml")
def train(cfg: DictConfig):
    env = hydra.utils.instantiate(cfg.env, device=device, seed=seed)
    model = hydra.utils.instantiate(cfg.model)
    model=load_model(model,root_dir+cfg.module.ckpt_path)

    # module = hydra.utils.instantiate(cfg.module, env=env, policy=model)
    module = hydra.utils.instantiate(cfg.module, env=env, policy=model,train_data_path=root_dir+cfg.module.train_data_path)
    logger = hydra.utils.instantiate(cfg.logger)
    checkpoint_callback = hydra.utils.instantiate(cfg.model_checkpoint)

    trainer = hydra.utils.instantiate(
        cfg.trainer,
        devices=[gpu_device],
        callbacks=[checkpoint_callback],
        logger=logger,
    )

    trainer.fit(model=module, ckpt_path=cfg.get('saved_ckpt'))


def load_model(model, ckpt_path):
    ckpt = torch.load(ckpt_path)
    model_state_dict = model.state_dict()
    ckpt_dict = ckpt['state_dict']

    for name, param in model_state_dict.items():
        if 'policy.' + name in ckpt_dict:
            if model_state_dict[name].shape == param.shape:
                model_state_dict[name].copy_(param)
            else:
                print(f"Shape mismatch for {name}: "
                      f"model {model_state_dict[name].shape}, ckpt {param.shape}")
        else:
            print(f"Key {name} not found in the model's state_dict.")


    model.load_state_dict(model_state_dict)

    return model

if __name__ == "__main__":
    train()
