import hydra
import os
import sys

from omegaconf import OmegaConf

root_dir = os.path.dirname(os.path.abspath(__file__))  # '{your_path}/EasyNCO'
normalized_path = os.path.normpath(root_dir)  # Normalize the path to avoid issues with different separators
split_path = normalized_path.split(os.sep) # Split the path using the system-specific separator
insert_dir = os.sep.join(split_path[:-1])  # your_path
sys.path.insert(0, insert_dir) # insert the path to sys.path and make sure it is the first one
dataset_path = os.sep.join([root_dir, "data", "datasets"])

from EasyNCO.utils.utils import *
from EasyNCO.phases.specific_eval.omni_finetune_eval import omni_finetune_eval


@hydra.main(config_path=root_dir, config_name='configs.yaml')
def main(cfg : DictConfig):

    if cfg.model == 'omni' and cfg.settings.module.fine_tune.enable:
        model_settings = cfg.settings
        model_settings.env._target_ = f"EasyNCO.neural_solvers.envs.{cfg.problem.upper()}Env"

        test_loader = model_settings.test_loader
        model_path = os.sep.join([root_dir, test_loader.model_dirpath,
                     test_loader.model_filename])
        omni_finetune_eval(cfg,dataset_path,model_path)
    else:
        seed = cfg.seed

        # set a random seed for reproducibility
        L.seed_everything(seed=seed, workers=True)
        full_data_path = os.sep.join([dataset_path, cfg.test_data_path]) # load the test data

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
        if cfg.model in ["mtpomo","mvmoe"]:
            model_settings.env._target_ = f"EasyNCO.neural_solvers.envs.MVRPEnv"
            OmegaConf.set_struct(cfg, False)  # 允许修改结构
            model_settings.env["test_problem"]=cfg.problem.upper()
            model_settings.env = OmegaConf.merge({"test_problem":cfg.problem.upper()}, model_settings.env)
        else:
            model_settings.env._target_ = f"EasyNCO.neural_solvers.envs.{cfg.problem.upper()}Env"

        print_configs(cfg, exclude_keys=["trainer"])

        try:
            env = hydra.utils.instantiate(model_settings.env, seed = seed)
        except ImportError:
            print(f'Warning: {model_settings.env._target_} is not supported, using DUMMYEnv instead.')
            model_settings.env._target_ = "EasyNCO.neural_solvers.envs.DUMMYEnv"
            env = hydra.utils.instantiate(model_settings.env, seed = seed)
        
        logger = hydra.utils.instantiate(cfg.logger)

        tester = hydra.utils.instantiate(model_settings.trainer,
                                         devices=devices,
                                         num_nodes=len(devices) if isinstance(devices, list) else 1,
                                         accelerator="gpu" if use_cuda else "cpu",
                                         strategy=strategy,
                                         logger=logger)

        def _load_policy(*, model_cfg: DictConfig, loader_cfg: DictConfig, model_name: str):
            full_model_path = os.sep.join([root_dir, loader_cfg.model_dirpath, loader_cfg.model_filename])
            pol = hydra.utils.instantiate(model_cfg)
            pol = load_model(pol, full_model_path, device=cuda_device, model_name=model_name) if cfg.sandbox == False else pol
            return pol, full_model_path

        # Support a minimal "two-policy composition" mode:
        # - init_model/init_test_loader -> policy for Initialization
        # - iter_model/iter_test_loader -> policy for Iteration
        # This enables experiments like "LEHD init + GLOP iter" without rewriting the pipeline.
        composed_mode = ("init_model" in model_settings) and ("iter_model" in model_settings)
        if composed_mode:
            init_model_name = model_settings.get("init_model_name", None) or cfg.model
            iter_model_name = model_settings.get("iter_model_name", None) or cfg.model

            init_policy, init_path = _load_policy(
                model_cfg=model_settings.init_model,
                loader_cfg=model_settings.init_test_loader,
                model_name=init_model_name,
            )
            iter_policy, iter_path = _load_policy(
                model_cfg=model_settings.iter_model,
                loader_cfg=model_settings.iter_test_loader,
                model_name=iter_model_name,
            )

            # logger.info(f"[compose] init_policy={init_model_name} ckpt={init_path}")
            # logger.info(f"[compose] iter_policy={iter_model_name} ckpt={iter_path}")

            initialization = hydra.utils.instantiate(model_settings.initialization, policy=init_policy)
            iteration = hydra.utils.instantiate(model_settings.iteration, policy=iter_policy)
            # The LightningModule still expects a single `policy` field.
            # For test-only use cases, it is safe to pass the iteration policy.
            policy = iter_policy
        else:
            test_loader = model_settings.test_loader
            full_model_path = os.sep.join([root_dir, test_loader.model_dirpath, test_loader.model_filename])

            policy = hydra.utils.instantiate(model_settings.model)
            policy = load_model(policy, full_model_path, device=cuda_device, model_name=cfg.model) if cfg.sandbox == False else policy

            initialization = hydra.utils.instantiate(model_settings.initialization, policy=policy)
            iteration = hydra.utils.instantiate(model_settings.iteration, policy=policy)
        sandbox_test_batch = hydra.utils.instantiate(cfg.sandbox_test_batch) if cfg.sandbox else None

        module = hydra.utils.instantiate(
            model_settings.module,
            env=env,
            policy=policy,
            initialization=initialization,
            iteration=iteration,
            sandbox_test_batch=sandbox_test_batch,
            test_data_path=full_data_path,
            varying_data_params=cfg.varying_data_params,
        )

        tester.test(module)


'''
The format of training command is:
python eval.py settings={model}_settings mode=test model={model} problem={problem} cuda=[{cuda}] ... .,etc.

For more settings, please refer to the base.yaml file

A example command to run the code:
python eval.py settings=pomo_settings model=pomo problem=tsp cuda=[0] mode=test

or only add "settings" and "mode", if you don't want to change the default settings:
python eval.py settings=pomo_settings mode=test

or you can also run directly if you already override the default settings in configs.yaml:
python eval.py mode=test

'''
if __name__ == '__main__':
    warnings_filter()
    main()
