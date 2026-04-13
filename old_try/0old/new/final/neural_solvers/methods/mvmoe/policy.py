import torch.nn as nn
import torch
from tensordict import TensorDict

from EasyNCO.neural_solvers.utils import get_post_search_strategy
from EasyNCO.neural_solvers.methods import MTMoEEncoder
from EasyNCO.neural_solvers.methods import MTMoEDecoder
from EasyNCO.utils.utils import _get_encoding

class MoEPolicy(nn.Module):
    """
        MOE implementations:
            (1) with tutel, ref to "https://github.com/microsoft/tutel"
            (2) with "https://github.com/davidmrau/mixture-of-experts"
    """
    def __init__(self,embedding_dim,ff_hidden_dim,encoder_layer_num,num_heads=8,qkv_dim=16,logit_clipping=10.0,normalization="batch",norm_loc='norm_last',MoE_param=None):
        super().__init__()

        self.aux_loss = 0
        self.light=MoE_param['light']
        if self.light:
            self.T=1.0
        self.sqrt_embedding_dim=embedding_dim**(1/2)

        self.encoder = MTMoEEncoder(embedding_dim,ff_hidden_dim,encoder_layer_num,num_heads,qkv_dim,normalization,norm_loc,MoE_param)
        self.decoder = MTMoEDecoder(embedding_dim,num_heads,qkv_dim,self.sqrt_embedding_dim,logit_clipping,MoE_param)
        self.encoded_nodes = None  # shape: (batch, problem+1, EMBEDDING_DIM)
        self.decoder_strategy = None
        # self.device = torch.device('cuda', torch.cuda.current_device()) if 'device' not in model_params.keys() else model_params['device']

    def pre_forward(self, reset_state):
        depot_node_xy=reset_state["depot_node_xy"]
        depot_node_demand=reset_state["depot_node_demand"]
        depot_node_tw_start=reset_state["depot_node_tw_start"]
        depot_node_tw_end=reset_state["depot_node_tw_end"]

        node_xy_demand_tw = torch.cat((depot_node_xy[:,1:,:], depot_node_demand[:, 1:, :], depot_node_tw_start[:, 1:, None], depot_node_tw_end[:, 1:, None]), dim=2)
        depot_xy=depot_node_xy[:,[0],:]
        self.encoded_nodes, moe_loss = self.encoder(depot_xy, node_xy_demand_tw)
        self.aux_loss = moe_loss
        # shape: (batch, problem+1, embedding)
        self.decoder.set_kv(self.encoded_nodes)

    def set_decoder_strategy(self, strategy: str = "sampling"):
        self.decoder_strategy = strategy

    def forward(self, state: TensorDict, first_mode: str = 'random') -> TensorDict:
        batch_size = state.batch_size[0]
        pomo_size = state.batch_size[1]
        selected=None
        prob=None
        if (state['next']['selected_count'] == 0).all():
            selected = torch.zeros(size=(batch_size, pomo_size), dtype=torch.long)
            prob = torch.ones(size=(batch_size, pomo_size))
        elif (state['next']['selected_count'] == 1).all():
            # selected = torch.arange(start=1, end=pomo_size + 1)[None, :].expand(batch_size, pomo_size)
            selected=state["start_node"]
            prob = torch.ones(size=(batch_size, pomo_size))

        else:
            encoded_last_node = _get_encoding(self.encoded_nodes, state['current_node'].unsqueeze(-1)).squeeze()
            # shape: (batch, pomo, embedding)
            attr = torch.cat((state['load'][:, :, None], state['current_time'][:, :, None], state['length'][:, :, None], state['open'][:, :, None]), dim=2)
            # shape: (batch, pomo, 4)
            if self.light:
                probs, moe_loss = self.decoder(encoded_last_node, attr, ninf_mask=state['next']['ninf_mask'], T=self.T,
                                               step=state["next"]["selected_count"][0,0])
            else:
                probs, moe_loss = self.decoder(encoded_last_node, attr, ninf_mask=state['next']['ninf_mask'])
            self.aux_loss += moe_loss

            selected, prob = get_post_search_strategy(self.decoder_strategy, probs)
        state.set("action", selected)
        state.set("prob", prob)
        return state