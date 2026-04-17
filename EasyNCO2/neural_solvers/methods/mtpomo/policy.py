import torch
from tensordict import TensorDict
from torch import nn
from EasyNCO.neural_solvers.methods import MTPOMOEncoder,MTPOMODecoder
from EasyNCO.neural_solvers.utils import get_post_search_strategy

from EasyNCO.utils.utils import _get_encoding


class MTPOMOPolicy(nn.Module):

    def __init__(self, embedding_dim,ff_hidden_dim,encoder_layer_num,head_num,qkv_dim,logit_clipping,normalization="batch"):
        super().__init__()
        self.sqrt_embedding_dim=embedding_dim**(1/2)
        self.encoder = MTPOMOEncoder(embedding_dim,ff_hidden_dim,encoder_layer_num,head_num=head_num,qkv_dim=qkv_dim,normalization=normalization)
        self.decoder = MTPOMODecoder(embedding_dim,head_num,qkv_dim,self.sqrt_embedding_dim,logit_clipping)
        self.encoded_nodes = None
        # shape: (batch, problem+1, EMBEDDING_DIM)

    def pre_forward(self, reset_state):
        # depot_node_xy_demand=reset_state["locs"]
        # depot_xy=depot_node_xy_demand[:,0:1,:2]
        # node_xy_demand=depot_node_xy_demand[:,1:,:]
        # node_earlyTW = reset_state["node_tw_start"]
        # # shape: (batch, problem)
        # node_lateTW = reset_state["node_tw_start"]
        # # shape: (batch, problem)
        # node_TW = torch.cat((node_earlyTW[:, :, None],node_lateTW[:, :, None]),dim=2)
        # # shape: (batch, problem, 2)
        # node_xy_demand_TW = torch.cat((node_xy_demand,node_TW),dim=2)
        # # shape: (batch, problem, 5)

        depot_node_xy=reset_state["depot_node_xy"]
        depot_node_demand=reset_state["depot_node_demand"]
        depot_node_tw_start=reset_state["depot_node_tw_start"]
        depot_node_tw_end=reset_state["depot_node_tw_end"]

        node_xy_demand_tw = torch.cat((depot_node_xy[:,1:,:], depot_node_demand[:, 1:, :], depot_node_tw_start[:, 1:, None], depot_node_tw_end[:, 1:, None]), dim=2)
        depot_xy=depot_node_xy[:,[0],:]



        self.encoded_nodes = self.encoder(depot_xy, node_xy_demand_tw)
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
            encoded_last_node = _get_encoding(self.encoded_nodes, state["current_node"].unsqueeze(-1))
            # shape: (batch, pomo, embedding)

            probs = self.decoder(encoded_last_node, state["load"], state["current_time"],state["length"],state["open"], ninf_mask=state["next"]["ninf_mask"])
            # shape: (batch, pomo, problem+1)
            #print(probs.shape)

            selected, prob = get_post_search_strategy(self.decoder_strategy, probs)


        state.set("action", selected)
        state.set("prob", prob)
        return state