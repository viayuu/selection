from typing import Literal
import torch.nn as nn
from torch import Tensor
import torch
from tensordict import TensorDict
from torch import nn
from tensordict import TensorDict

from .reld_encoder import ReLDEncoder
from .reld_decoder import ReLDDecoder
from EasyNCO.neural_solvers.utils import get_post_search_strategy
from EasyNCO.utils.utils import getLogger
from EasyNCO.utils.utils import _get_encoding

logger = getLogger(__name__)


class ReLDPolicy(nn.Module):
    def __init__(self,
                 env_name: str = "cvrp",
                 embed_dim: int = 128,
                 num_heads: int = 8,
                 qkv_dim: int = 16,
                 num_encoder_layers: int = 6,
                 num_decoder_layers: int = None,
                 normalization: str = None,
                 feedforward_hidden: int = 512,
                 logit_clipping: float = 50,  # ReLD使用更大的截断值
                 use_graph_mean: bool = False,
                 am_mode: bool = True,
                 first_placeholder: bool = False,
                 first_mode: Literal["random","placeholder"] = "random",
                 **kwargs):
        super().__init__()
        
        # 🔥 ReLD移植：将参数转换为字典格式以匹配新的ReLDEncoder
        model_params = {
            'embedding_dim': embed_dim,
            'head_num': num_heads,
            'qkv_dim': qkv_dim,
            'encoder_layer_num': num_encoder_layers,
            'ff_hidden_dim': feedforward_hidden,
            # 添加其他可能的参数
            **kwargs
        }
        
        # ReLD编码器（与作者架构一致）
        self.encoder = ReLDEncoder(**model_params)
        
        # ReLD解码器（增强版，包含IDT、FF、距离启发式）
        self.decoder = ReLDDecoder(
            env_name=env_name,
            embed_dim=embed_dim,
            num_heads=num_heads,
            qkv_dim=qkv_dim,
            logit_clipping=logit_clipping,
            use_graph_mean=use_graph_mean,
            am_mode=am_mode,
            first_placeholder=first_placeholder,
            feedforward_hidden=feedforward_hidden,
        )
        
        self.decoder_strategy = None
        self.use_graph_mean = use_graph_mean
        self.first_mode = first_mode

    def set_decoder_strategy(self, strategy: str="sampling"):
        self.decoder_strategy = strategy

    def pre_forward(self, td: TensorDict):
        self.encoded_nodes, _ = self.encoder(td)  # 修改：作为类的成员变量
        self.decoder.set_kv(self.encoded_nodes)
        if self.use_graph_mean:
            self.decoder.set_graph_mean(self.encoded_nodes)

    def forward(self, td: TensorDict, cur_dist: Tensor = None, **kwargs) -> TensorDict:
        # 获取当前距离信息
        if cur_dist is not None:
            kwargs['cur_dist'] = cur_dist

        batch_size = td.batch_size[0]
        pomo_size = td.batch_size[1]

        if (td['next']['selected_count'] == 0).all():
            selected = torch.zeros(size=(batch_size, pomo_size), dtype=torch.long)
            prob = torch.ones(size=(batch_size, pomo_size))
        elif (td['next']['selected_count'] == 1).all():
            selected = torch.arange(start=1, end=pomo_size + 1)[None, :].expand(batch_size, pomo_size)
            # selected=td["start_node"]
            prob = torch.ones(size=(batch_size, pomo_size))
        else:
            # 只有在else分支中才执行神经网络计算
            encoded_last_node = _get_encoding(self.encoded_nodes, td["action"].unsqueeze(-1))
            
            pomo_selected, probs = self.decoder(encoded_last_node, td["load"], td, first_mode=self.first_mode, ninf_mask=td["next"]["ninf_mask"], **kwargs)

            if pomo_selected is not None:
                selected = pomo_selected
                prob = probs
            else:
                selected, prob = get_post_search_strategy(self.decoder_strategy, probs)

        td.set("action", selected)
        td.set("prob", prob)

        return td