import torch.nn as nn
import torch
from tensordict import TensorDict

from EasyNCO.neural_solvers.methods import PointerformerEncoder, PointerformerDecoder
from EasyNCO.neural_solvers.utils import get_post_search_strategy
from EasyNCO.utils.utils import getLogger

from EasyNCO.data.data_utils import augment_pomo

logger = getLogger(__name__)

class PointerformerPolicy(nn.Module):
    def __init__(self,
                 embedding_dim: int = 128,
                 input_dim: int = 24,
                 num_heads: int = 8,
                 num_encoder_layers: int = 6,
                 intermediate_dim: int=512,
                 add_init_projection:bool=True,
                 tanh_clipping: int=10.0,
                 multi_pointer: int=8,
                 multi_pointer_level=1,
                 add_more_query: bool = True,
                 env_name: str ="tsp"
                 ):
        super().__init__()
        self.encoder = PointerformerEncoder(
                        n_layers=num_encoder_layers,
                        n_heads=num_heads,
                        embedding_dim=embedding_dim,
                        input_dim=input_dim,
                        intermediate_dim=intermediate_dim,
                        add_init_projection = add_init_projection,
        )
        self.decoder = PointerformerDecoder(
                        embedding_dim=embedding_dim,
                        n_heads=num_heads,
                        tanh_clipping=tanh_clipping,
                        multi_pointer=multi_pointer,
                        multi_pointer_level=multi_pointer_level,
                        add_more_query=add_more_query,
                        env_name=env_name,
                        )
        self.decoder_strategy = None

    def set_decoder_strategy(self, strategy: str = "sampling"):
        self.decoder_strategy = strategy

    def pre_forward(self, td:TensorDict):
        problems=td["locs"]
        aug_problems=augment_pomo(problems=problems, aug_factor=8)
        data1, data2, data3, data4, data5, data6, data7, data8 = torch.chunk(aug_problems, 8, dim=0)
        aug_problems=torch.cat((data1, data2, data3, data4, data5, data6, data7, data8), dim=2)
        theta = []
        for i in range(8):
            theta.append(torch.atan(aug_problems[:, :, i * 2 + 1] / aug_problems[:, :, i * 2]).unsqueeze(-1))  # aug8后的y/x
        theta.append(aug_problems)
        problems = torch.cat(theta, dim=2)  # 再把这些进行拼接
        encoded_nodes= self.encoder(problems)
        self.decoder.set_kv(problems,encoded_nodes,problems.shape[1])


    def forward(self, td: TensorDict, first_mode: str = None) -> TensorDict:

        pomo_selected, probs = self.decoder(td, first_mode=first_mode)

        if pomo_selected is not None:
            selected = pomo_selected
            prob = probs
        else:
            selected, prob = get_post_search_strategy(self.decoder_strategy, probs)

        td.set("action", selected)
        td.set("prob", prob)

        return td

