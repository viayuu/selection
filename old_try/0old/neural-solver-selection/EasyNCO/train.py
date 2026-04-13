import torch
import hydra
import os
import sys
import types
from omegaconf import DictConfig
import lightning as L

root_dir = os.path.dirname(os.path.abspath(__file__))  # '{your_path}/EasyNCO'

# Ensure `import EasyNCO.*` resolves to this `final/` folder (repo also has a top-level `EasyNCO/`).
local_pkg = types.ModuleType("EasyNCO")
local_pkg.__path__ = [root_dir]
sys.modules["EasyNCO"] = local_pkg

normalized_path = os.path.normpath(root_dir)  # Normalize the path to avoid issues with different separators
split_path = normalized_path.split(os.sep) # Split the path using the system-specific separator
insert_dir = os.sep.join(split_path[:-1])  # your_path
sys.path.insert(0, insert_dir) # insert the path to sys.path and make sure it is the first one
dataset_path = os.sep.join([root_dir, "data", "datasets"])

from EasyNCO.utils.utils import (set_device,
                                 print_configs,
                                 warnings_filter)

@hydra.main(config_path=root_dir, config_name="configs.yaml")
def train(cfg: DictConfig):
    seed = cfg.seed
    # set a random seed for reproducibility
    L.seed_everything(seed=seed, workers=True)

    # set the device to use for training
    use_cuda = True if not cfg.disable_gpu and torch.cuda.is_available() else False
    cuda_device = cfg.cuda if use_cuda else 1 # 1 denotes CPU device is used
    auto_configure_ddp = True if isinstance(cuda_device, list) and len(cuda_device) > 1 else False
    devices, strategy = set_device(use_cuda=use_cuda,
                                   cuda_device=cuda_device,
                                   auto_configure_ddp=auto_configure_ddp,
                                   strategy=cfg.strategy,
                                   matmul_precision=cfg.matmul_precision)

    model_settings = cfg.settings

    model_settings.env._target_ = f"EasyNCO.neural_solvers.envs.{cfg.problem.upper()}Env"
    try:
        env = hydra.utils.instantiate(model_settings.env, seed = seed)
    except ImportError:
        print(f'Warning: {model_settings.env._target_} is not supported, using DUMMYEnv instead.')
        model_settings.env._target_ = "EasyNCO.neural_solvers.envs.DUMMYEnv"
        env = hydra.utils.instantiate(model_settings.env, seed = seed)
        
    do_val = True if cfg.val_data_path is not None else False
    if do_val:
        cfg.val_data_path = os.sep.join([dataset_path, cfg.val_data_path])
        val_env = hydra.utils.instantiate(model_settings.env, seed=seed)
    else:
        val_env = None

    print_configs(cfg, exclude_keys=["tester","test_loader"])

    model = hydra.utils.instantiate(model_settings.model)
    initialization = hydra.utils.instantiate(model_settings.initialization, policy=model)
    iteration = hydra.utils.instantiate(model_settings.iteration, policy=model)
    module = hydra.utils.instantiate(
        model_settings.module,
        env=env,
        policy=model,
        initialization=initialization,
        iteration=iteration,
        do_val=do_val,
        val_data_path=cfg.val_data_path,
        val_env=val_env,
    )
    logger = hydra.utils.instantiate(cfg.logger)
    checkpoint_callback = hydra.utils.instantiate(model_settings.model_checkpoint)

    trainer = hydra.utils.instantiate(
        model_settings.trainer,
        devices=devices,
        num_nodes=len(devices) if isinstance(devices, list) else 1,
        accelerator="gpu" if use_cuda else "cpu",
        callbacks=[checkpoint_callback],
        logger=logger,
        strategy=strategy,
    )

    # In our platform, we load the data in LightningModule directly, rather than using "trainer.fit".
    trainer.fit(model=module,
                train_dataloaders=None,
                val_dataloaders=None,
                datamodule=None,
                ckpt_path=None,
                )


'''
The format of training command is:
python train.py settings={model}_settings mode=train model={model} problem={problem} cuda=[{cuda}] ....,etc.

For more settings, please refer to the base.yaml file

A example command to run the code:
python train.py settings=pomo_settings model=pomo problem=tsp cuda=[0]

or only add "settings" and "mode", if you don't want to change the default settings:
python train.py settings=pomo_settings mode=train

or you can also run directly if you already override the default settings in configs.yaml:
python train.py

'''
if __name__ == "__main__":
    warnings_filter()
    train()
