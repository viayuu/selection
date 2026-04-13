import hydra
import os
import sys

from omegaconf import DictConfig


current_dir = os.path.dirname(os.path.abspath(__file__))
split_path = current_dir.split("/")
root_dir = "/".join(split_path[:-2])
insert_dir = "/".join(split_path[:-5])
sys.path.insert(0, insert_dir)
config_path = f"{root_dir}/configs"
first_mode = 'random'


from EasyNCO.data import *
from EasyNCO.utils.utils import set_device, seed_everything
from EasyNCO.phases.eval.Evaluate import EVALUATELightning


seed = 2024
gpu_device = 0

device = set_device(use_cuda=True,
                    gpu_device=gpu_device)
seed_everything(seed=seed)


@hydra.main(config_path = config_path, config_name = 'tsp_pomo_config.yaml')
def main(cfg : DictConfig):
    model_path = f"{cfg.test_loader.model_dirpath}{cfg.test_loader.model_filename}"
    absolute_path = f"{root_dir}{model_path}"
    tester = hydra.utils.instantiate(cfg.trainer, devices=[gpu_device], )

    env = hydra.utils.instantiate(cfg.test_env, device = device, seed = seed)
    policy = hydra.utils.instantiate(cfg.model)
    model = EVALUATELightning.load_from_checkpoint(absolute_path,
                                                  env = env,
                                                  policy = policy,
                                                  test_params = cfg.test_loader.test_params,
                                                  first_mode = first_mode)

    data_dict = cfg.test_loader.data_dict
    if data_dict['use_customized_data']:
        path = data_dict['data_path']
        data_path = f"{root_dir}{path}"
    else:
        data_path = None

    data_loader = test_DataLoader(env.env_name,
                                      {
                                          'data_size': cfg.test_loader.test_params["test_episodes"] \
                                                if cfg.test_loader.test_params["augmentation"]["aug_type"] is None \
                                                else int(cfg.test_loader.test_params["test_episodes"] / cfg.test_loader.test_params["augmentation"]["aug_factor"]),
                                          'problem_size': env.problem_size,
                                          'batch_size': cfg.test_loader.test_params["test_batch_size"],
                                          'device' : device,
                                          'path' : data_path,
                                      })

    tester.test(model, dataloaders = data_loader)


def test_DataLoader(env_name: str, config: dict):
    dataloader = {
        'cvrp': CVRPGenerator,
        'tsp': TSPGenerator,
        'kp': KPGenerator,
    }

    return dataloader[env_name](**config)


if __name__ == '__main__':
    main()