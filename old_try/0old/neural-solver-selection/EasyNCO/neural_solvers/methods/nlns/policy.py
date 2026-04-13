import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tensordict import TensorDict

from EasyNCO.neural_solvers.methods.nlns.actor import ActorModel
from EasyNCO.neural_solvers.utils import get_post_search_strategy
from EasyNCO.utils.utils import getLogger

logger = getLogger(__name__)

class NLNSPolicy(nn.Module):
    def __init__(self,
                 env_name,
                 pointer_hidden_size,
                 critic_train,
                 critic_max_grad_norm,
                 critic_lr,
                 ):
        super().__init__()
        self.env_name = env_name
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.actor = ActorModel(self.device, hidden_size=pointer_hidden_size).to(self.device)
        self.decoder_strategy = None
        self.critic_train = critic_train
        self.critic_max_grad_norm = critic_max_grad_norm
        self.critic_lr = critic_lr

    def set_decoder_strategy(self, strategy: str="sampling"):
        self.decoder_strategy = strategy

    def pre_forward(self, td:TensorDict):

        self.batch_size = td['locs'].size(0)
        self.vehicle_capacity = td['vehicle_capacity'][0].item()
        # self.vehicle_capacity = td['vehicle_capacity'][0][0]


    def forward(self, td: TensorDict) -> TensorDict:
        next = td['next']

        static_input = next['static_input'] # shape: (batch, nb_input, 2)
        dynamic_input = next['dynamic_input'] # shape: (batch, nb_input, 2)
        origin_idx = next['origin_idx'] # shape: (batch, )
        mask = next['mask'] # shape: (batch, num_input)

        # Rescale customer demand based on vehicle capacity
        dynamic_input_float = dynamic_input.float()
        # dynamic_input_float[:, :, 0] = dynamic_input_float[:, :, 0] / float(self.vehicle_capacity)

        origin_static_input = static_input[torch.arange(self.batch_size), origin_idx]
        origin_dynamic_input_float = dynamic_input_float[torch.arange(self.batch_size), origin_idx]

        # print(static_input[0])
        # print(dynamic_input_float[0])
        # print(origin_static_input[0])
        # print(origin_dynamic_input_float[0])

        # Forward pass. Returns a probability distribution over the point (tour end or depot) that origin should be connected to
        probs = self.actor.forward(static_input, dynamic_input_float, origin_static_input, origin_dynamic_input_float, mask)
        probs = F.softmax(probs + mask.log(), dim=1)  # Set prob of masked tour ends to zero
        # shape: (batch, num_input)
        probs = probs.unsqueeze(1)
        # shape: (batch, 1, num_input)

        selected, prob = get_post_search_strategy(self.decoder_strategy, probs)
        # shape: (batch, 1)

        td["selected"] = selected
        td["prob"] = prob

        return td




