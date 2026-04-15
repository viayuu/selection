'''
utils.py
This file contains utility functions for the EasyNCO project, mainly including:
- get_encoding: obtain the encoding of the nodes to be picked
- seed_everything: set the seed for all the libraries, including numpy, torch, and random. Note that this function is replaced by lightning.seed_everything(seed) in lightning.pytorch.utilities.rank_zero.py
- set_device: set the device for the training
- get_lightning_device: get the device of the Lightning module
- getLogger: set up the logger level for the Lightning module
- print_configs: print the configs in a readable format
- warnings_filter: ignore specific warnings, such as the warning about the dataloader
- AverageMeter: a class to calculate the average of a list of numbers
- TimeEstimator: a class to estimate the time
'''
import glob
import io
import os
import pickle
import random
import time
import types
from typing import Dict, Iterable, List, Optional, Union

import numpy as np
import torch
import lightning as L

import logging
from lightning.pytorch.utilities.rank_zero import rank_zero_only
import warnings

from lightning.fabric.accelerators.cuda import num_cuda_devices
from lightning.pytorch.strategies import DDPStrategy, Strategy
from omegaconf import DictConfig

logger = logging.getLogger(__name__)

def _get_encoding(encoded_nodes, node_index_to_pick):
    # encoded_nodes.shape: (batch, node_num, embedding) or (batch, 1, node_num, embedding) or (batch, pomo, node_num, embedding)
    # node_index_to_pick.shape: (batch, pomo, num_to_pick)

    pomo_size = node_index_to_pick.size(1)
    embedding_dim = encoded_nodes.size(-1)
    if encoded_nodes.dim() == node_index_to_pick.dim():
        encoded_nodes_ = encoded_nodes.clone()[:, None, :, :].expand(-1, pomo_size, -1, -1)
    elif encoded_nodes.dim() > node_index_to_pick.dim() and encoded_nodes.size(1) != pomo_size:
        encoded_nodes_ = encoded_nodes.clone().expand(-1, pomo_size, -1, -1)
    else:
        encoded_nodes_ = encoded_nodes.clone()
    gathering_index = node_index_to_pick[:, :, :, None].expand(-1, -1, -1, embedding_dim)
    # shape: (batch, pomo, num_to_pick, embedding)
    picked_nodes = encoded_nodes_.gather(dim=2, index=gathering_index)
    # shape: (batch, pomo, num_to_pick, embedding)

    return picked_nodes

def seed_everything(seed=1234):
    """
    you can also use lighting.seed_everything(seed) to set the seed for all the libraries
    """
    random.seed(seed)
    np.random.seed(seed)
    # To prevent hash randomization, making the experiment reproducible.
    os.environ['PYTHONHASHSEED'] = str(seed)

    # Set the random seed manually for GPU environment
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)

def set_device(use_cuda: bool = True,
               cuda_device: Union[List[int], str] = "auto",
               disable_profiling_executor: bool=True,
               auto_configure_ddp: bool=True,
               strategy: Union[str, Strategy]="auto",
               matmul_precision: str="medium"):
    if not use_cuda or not torch.cuda.is_available():
        torch.device("cpu")
        torch.set_default_tensor_type("torch.FloatTensor")
        device = 1
    else:
        # Configure the default tensor type
        torch.set_default_tensor_type(torch.cuda.FloatTensor)
        # Configure DDP automatically if multiple GPUs are available
        if auto_configure_ddp and strategy == "auto":
            if cuda_device == "auto":
                n_devices = num_cuda_devices()
            else:
                n_devices = len(cuda_device)
            if n_devices > 1:
                logger.info("Configuring DDP strategy automatically with {} GPUs".format(n_devices))
                strategy = DDPStrategy(
                    find_unused_parameters=True,  # We set to True due to RL envs
                    gradient_as_bucket_view=True,
                    # https://pytorch-lightning.readthedocs.io/en/stable/advanced/advanced_gpu.html#ddp-optimizations
                )
        else:
            torch.cuda.set_device(cuda_device[0])
        device = cuda_device
    # Disable JIT profiling executor. This reduces memory and increases speed.
    # Reference: https://github.com/HazyResearch/safari/blob/111d2726e7e2b8d57726b7a8b932ad8a4b2ad660/train.py#LL124-L129C17
    if disable_profiling_executor:
        try:
            torch._C._jit_set_profiling_executor(False)
            torch._C._jit_set_profiling_mode(False)
        except AttributeError:
            pass

    # Configure the precision
    torch.set_float32_matmul_precision(matmul_precision)
    return device, strategy

def get_lightning_device(lit_module: L.LightningModule) -> torch.device:
    """
    Get the device of the Lightning module before setup is called
    See device setting issue in setup https://github.com/Lightning-AI/lightning/issues/2638
    """
    try:
        if lit_module.trainer.strategy.root_device != lit_module.device:
            return lit_module.trainer.strategy.root_device
        return lit_module.device
    except Exception:
        return lit_module.device

def getLogger(name=__name__) -> logging.Logger:
    """
    Initializes multi-GPU-friendly python command line logger.
    """

    logger = logging.getLogger(name)

    # this ensures all logging levels get marked with the rank zero decorator
    # otherwise logs would get multiplied for each GPU process in multi-GPU setup
    logging_levels = (
        "debug",
        "info",
        "warning",
        "error",
        "exception",
        "fatal",
        "critical",
    )

    for level in logging_levels:
        setattr(logger, level, rank_zero_only(getattr(logger, level)))

    return logger

def print_configs(cfg: DictConfig,
                  exclude_keys: Optional[List[str]] = None):
    """
    Print the configs in a readable format
    """
    for key, value in cfg.items():
        if key == "settings":
            for k, v in value.items():
                if exclude_keys is not None and k not in exclude_keys:
                    logger.info(f"{key}.{k}: {v}")
        else:
            logger.info(f"{key}: {value}")

def warnings_filter():
    """
    Ignore specific warnings, which include:
    1. UserWarning: The 'train_dataloader' does not have many workers.
    2. xxx (you can add more warnings here if needed)
    """
    # ignore the warning about the dataloader
    warnings.filterwarnings(
        "ignore",
        message=".*The .*_dataloader' does not have many workers.*",
        category=UserWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message=".*has been moved to cryptography.hazmat.decrepit.*",
        category=DeprecationWarning,
    )



class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.sum += (val * n)
        self.count += n

    @property
    def avg(self):
        return self.sum / self.count if self.count > 0 else 0

class TimeEstimator:
    def __init__(self):
        self.start_time = time.time()
        self.count_zero = 0

    def reset(self, count=1):
        self.start_time = time.time()
        self.count_zero = count-1

    def get_est(self, count, total):
        curr_time = time.time()
        elapsed_time = curr_time - self.start_time
        remain = total-count
        remain_time = elapsed_time * remain / (count - self.count_zero)

        elapsed_time /= 3600.0
        remain_time /= 3600.0

        return elapsed_time, remain_time

    def get_est_string(self, count, total):
        elapsed_time, remain_time = self.get_est(count, total)

        elapsed_time_str = "{:.2f}h".format(elapsed_time) if elapsed_time > 1.0 else "{:.2f}m".format(elapsed_time*60)
        remain_time_str = "{:.2f}h".format(remain_time) if remain_time > 1.0 else "{:.2f}m".format(remain_time*60)

        return elapsed_time_str, remain_time_str

from lightning.pytorch.callbacks import ModelCheckpoint
import re

class CustomModelCheckpoint(ModelCheckpoint):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    def _format_checkpoint_name(
            cls,
            filename: Optional[str],
            metrics: Dict[str, torch.Tensor],
            prefix: str = "",
            auto_insert_metric_name: bool = True,
    ) -> str:

        formatted_name = super(CustomModelCheckpoint, cls)._format_checkpoint_name(
            filename, metrics, prefix, auto_insert_metric_name
        )

        # use regular expressions to remove the 'epoch=' and 'step=' parts
        formatted_name = re.sub(r'(epoch)=', r'\1', formatted_name)
        formatted_name = re.sub(r'(step)=', r'\1', formatted_name)

        return formatted_name



def load_model(model, path, device, model_name):
    class Placeholder:
        def __init__(self, *args, **kwargs):
            pass
        def __getattr__(self, name):
            return lambda *args, **kwargs: None

    class CustomUnpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if module.startswith('EasyNCO'):
                return Placeholder
            return super().find_class(module, name)

    custom_pickle_module = types.ModuleType('custom_pickle_module')
    custom_pickle_module.Unpickler = CustomUnpickler

    load_device = "cpu" if device == 1 else f"cuda:{device[0]}"

    if model_name != 'nlns':
        try:
            # 🔥 ReLD移植：尝试不同的权重文件结构
            loaded_data = torch.load(path, pickle_module=custom_pickle_module, map_location=load_device)
            
            # 检查权重文件的结构
            if 'model_state_dict' in loaded_data:
                # ReLD作者权重文件格式：{'model_state_dict': {...}, 'step': ..., 'optimizer_state_dict': ...}
                ckpt_dict = loaded_data['model_state_dict']
                logger.info(f"🔥 Loaded ReLD author format: model_state_dict found with {len(ckpt_dict)} parameters")
            elif 'state_dict' in loaded_data:
                # 平台权重文件格式：{'state_dict': {...}}
                ckpt_dict = loaded_data['state_dict']
                logger.info(f"📦 Loaded platform format: state_dict found with {len(ckpt_dict)} parameters")
            else:
                # 直接是参数字典格式：{param_name: tensor, ...}
                ckpt_dict = loaded_data
                logger.info(f"📄 Loaded direct format: {len(ckpt_dict)} parameters found")
                
        except Exception as e:
            logger.error(f"❌ Failed to load checkpoint: {e}")
            ckpt_dict = torch.load(path, pickle_module=custom_pickle_module,map_location=load_device)
        try:
            model_state_dict = model.state_dict()

        except AttributeError as e:
            model_state_dict = model.model.state_dict()

        for name, param in model_state_dict.items():
            ckpt_name = "policy." + name if "policy." + name in ckpt_dict else name
            if ckpt_name in ckpt_dict:
                ckpt_param = ckpt_dict[ckpt_name]
                if ckpt_param.shape == param.shape:
                    model_state_dict[name].copy_(ckpt_param)
                else:
                    logger.info(f"Shape mismatch for {name}: "
                        f"model {param.shape}, ckpt {ckpt_param.shape}")
                    assert False, f"Shape mismatch for {name}: model {param.shape}, ckpt {ckpt_param.shape}"
            else:
                # 🔥 ReLD移植：多策略参数匹配
                # 尝试多种匹配策略来解决平台与作者代码的参数名差异
                possible_names = [
                    name,                                                    # 原始名称
                    name.replace('encoder.', ''),                              # 移除encoder前缀 (ReLD关键)
                    name.replace('decoder.', ''),                               # 移除decoder前缀
                    "policy." + name,                                         # 添加policy前缀
                    "policy." + name.replace('encoder.', ''),                    # policy + 移除encoder
                    "policy." + name.replace('decoder.', ''),                     # policy + 移除decoder
                    name.replace('encoder.layers.', 'layers.'),                   # encoder.layers -> layers
                    name.replace('decoder.feed_forward.', 'feed_forward.'),        # decoder.feed_forward -> feed_forward
                    "policy." + name.replace('encoder.layers.', 'layers.'),         # policy + encoder.layers -> layers
                    "policy." + name.replace('decoder.feed_forward.', 'feed_forward.'), # policy + decoder.feed_forward -> feed_forward
                ]
                
                found_match = False
                matched_name = None
                
                # 尝试每种可能的参数名
                for possible_name in possible_names:
                    if possible_name in ckpt_dict:
                        ckpt_param = ckpt_dict[possible_name]
                        if ckpt_param.shape == param.shape:
                            logger.info(f"✅ Successfully matched: {name} -> {possible_name}")
                            model_state_dict[name].copy_(ckpt_param)
                            found_match = True
                            matched_name = possible_name
                            break
                        else:
                            logger.warning(f"⚠️ Shape mismatch for {name} -> {possible_name}: "
                                       f"model {param.shape}, ckpt {ckpt_param.shape}")
                
                if found_match:
                    logger.info(f"🎯 Loaded parameter: {name} (matched as {matched_name})")
                else:
                    # 记录所有尝试的匹配策略
                    logger.error(f"❌ Failed to match parameter: {name}")
                    logger.error(f"   Attempted matches: {possible_names}")
                    logger.error(f"   Available checkpoint keys: {list(ckpt_dict.keys())[:10]}...")  # 只显示前10个
                    
                    # 对于ReLD关键参数，提供更详细的错误信息
                    if 'embedding_depot' in name or 'embedding_node' in name or 'feed_forward' in name:
                        logger.error(f"🚨 Critical ReLD parameter failed to load: {name}")
                        logger.error(f"   This suggests a fundamental architecture mismatch")
                    
                    assert False, f"Key {name} not found in checkpoint after trying all match strategies."
        try:
            model.load_state_dict(model_state_dict)
        except AttributeError as e:
            model.model.load_state_dict(model_state_dict)
    else:
        # This is for loading the NLNS models.
        # NLNS needs to load multiple models, multiple destroy strategies, and their different parameters.
        model_paths = glob.glob(os.path.join(path, '*.pt'))

        models  = []
        destroy_operations = []
        p_destructions = []
        for model_path in model_paths:
            model_data = torch.load(model_path, map_location=load_device)
            model_para = model_data['parameters']
            new_state_dict = {}
            
            for key in model_para.keys():
                if key in model.state_dict():
                    new_state_dict[key] = model_para[key]
                else:
                    new_key = "actor." + key if "actor." + key in model.state_dict() else None
                    if new_key:
                        new_state_dict[new_key] = model_para[key]
                    else:
                        raise ValueError(f"Skipping unmatched key: {key}")
            
            model.load_state_dict(new_state_dict)
            models.append(model)
            destroy_operations.append(model_data['destroy_operation'])
            p_destructions.append(model_data['p_destruction'])

        model = {
            'models': models,
            'destroy_operations': destroy_operations,
            'p_destructions': p_destructions
        }

    return model