import hydra
import os
import sys
import logging
import torch

from EasyNCO.utils.utils import *



def omni_finetune_eval(cfg : DictConfig, dataset_path,model_path):
    seed = cfg.seed
    # set a random seed for reproducibility
    L.seed_everything(seed=seed, workers=True)
    full_data_path = os.sep.join([dataset_path, cfg.test_data_path])  # load the test data

    # set the device to use for training
    use_cuda = True if not cfg.disable_gpu and torch.cuda.is_available() else False
    cuda_device = cfg.cuda if use_cuda else 1  # 1 denotes CPU device is used
    auto_configure_ddp = True if isinstance(cuda_device, list) and len(cuda_device) > 1 else False


    devices, strategy = set_device(use_cuda=use_cuda,
                                   cuda_device=cuda_device,
                                   auto_configure_ddp=auto_configure_ddp,
                                   strategy=cfg.strategy,
                                   matmul_precision="medium")
    model_settings = cfg.settings
    cfg.decoder_strategy = 'sampling'

    model_settings.trainer.max_epochs = model_settings.module.fine_tune.k
    model_settings.module.optimizer_params.optimizer = model_settings.module.fine_tune.optimizer
    if model_settings.module.fine_tune.augmentation_enable == False:
        temp_aug_type = model_settings.env.aug_type
        temp_aug_factor = model_settings.env.aug_factor
        model_settings.env.aug_type  = 'pomo_aug'
        model_settings.env.aug_factor = 1
        model_settings.module.aug_type = 'pomo_aug'

    env = hydra.utils.instantiate(model_settings.env, seed=seed)
    model = hydra.utils.instantiate(model_settings.model)
    model = load_model(model, model_path)

    initialization = hydra.utils.instantiate(
        model_settings.fine_tune_initialization, policy=model
    )
    iteration = hydra.utils.instantiate(model_settings.iteration, policy=model)

    module = hydra.utils.instantiate(model_settings.module, env=env, policy=model,initialization=initialization,iteration=iteration,
                                     test_data_path=full_data_path)
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
    trainer.fit(model=module,
                train_dataloaders=None,
                val_dataloaders=None,
                datamodule=None,
                ckpt_path=None,
                )

    logging.info('--------------finish fine_tune--------------------')
    cfg.decoder_strategy = 'greedy'
    if model_settings.module.fine_tune.augmentation_enable == False:
        env.aug_factor = temp_aug_factor
        env.aug_type = temp_aug_type
    policy = hydra.utils.instantiate(model_settings.model)
    policy = load_model(policy, checkpoint_callback.best_model_path)

    initialization = hydra.utils.instantiate(
        model_settings.initialization, policy=policy
    )
    iteration = hydra.utils.instantiate(model_settings.iteration, policy=policy)

    module = hydra.utils.instantiate(model_settings.module,
                                     env=env,
                                     policy=policy,
                                     initialization=initialization,
                                     iteration=iteration,
                                     test_data_path=full_data_path)
    trainer.test(module)