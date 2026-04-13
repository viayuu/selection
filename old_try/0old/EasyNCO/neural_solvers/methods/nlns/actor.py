# Parts of this code are based on https://github.com/mveres01/pytorch-drl4vrp/blob/master/model.py

import torch
import torch.nn as nn
import torch.nn.functional as F

from EasyNCO.neural_solvers.methods.ptr_nets.policy import Pointer

class Encoder(nn.Module):

    def __init__(self, input_size, hidden_size):
        super(Encoder, self).__init__()
        self.embed = nn.Linear(input_size, hidden_size)
        self.embed_2 = nn.Linear(hidden_size, hidden_size)

    def forward(self, input):
        output = F.relu(self.embed(input))
        output = self.embed_2(output)
        return output


class ActorModel(nn.Module):

    def __init__(self, device, hidden_size=128):
        super(ActorModel, self).__init__()

        self.all_embed = Encoder(4, hidden_size)
        self.pointer = Pointer(device, hidden_size)
        self.origin_embed = Encoder(4, hidden_size)

        for p in self.parameters():
            if len(p.shape) > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, static_input, dynamic_input_float, origin_static_input, origin_dynamic_input_float, mask):
        # Set the input feature values of already visited customers (demand == 0) to zero
        active_inputs = dynamic_input_float[:, 1:, 1] > 0
        static_input[:, 1:, :] = static_input[:, 1:, :] * active_inputs.unsqueeze(2).float()

        # Embed inputs
        all_hidden = self.all_embed.forward(
            torch.cat((static_input, dynamic_input_float), dim=2))
        origin_hidden = self.origin_embed.forward(
            torch.cat((origin_static_input.unsqueeze(1), origin_dynamic_input_float.unsqueeze(1)), dim=2))

        probs = self.pointer.forward(all_hidden.permute(0, 2, 1), origin_hidden.permute(0, 2, 1))
        return probs
