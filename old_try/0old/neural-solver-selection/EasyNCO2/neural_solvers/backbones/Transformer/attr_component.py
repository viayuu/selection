import inspect
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from EasyNCO.utils.utils import getLogger

logger = getLogger(__name__)

def reshape_by_heads(qkv, head_num):
    # q.shape: (batch, pomo, head_num*key_dim) or (batch, pomo, node_num, head_num*key_dim)
    # kv.shape: (batch, node_num, head_num*key_dim) or (batch, pomo, node_num, head_num*key_dim)

    if len(qkv.shape) == 4:
        # q.shape: (batch, pomo, node_num, head_num * key_dim)
        batch_size = qkv.size(0)
        pomo = qkv.size(1)
        node_num = qkv.size(2)
        q_reshaped = qkv.reshape(batch_size, pomo, node_num, head_num, -1)
        # shape: (batch, pomo, node_num, head_num, key_dim)
        q_transposed = q_reshaped.transpose(2, 3)
        # shape: (batch, pomo, head_num, node_num, key_dim)
    else:
        batch_size = qkv.size(0)
        node_num = qkv.size(1)
        q_reshaped = qkv.reshape(batch_size, node_num, head_num, -1)
        # shape: q:(batch, pomo, head_num, key_dim) kv:(batch, node_num, head_num, key_dim)
        q_transposed = q_reshaped.transpose(1, 2)
        # shape: q:(batch, head_num, pomo, key_dim) kv:(batch, head_num, node_num, key_dim)

    return q_transposed

def multi_head_attention(q, k, v, mask=None, sharp=False,score_dim=None):
    # q.shape: (batch, head_num, pomo, key_dim) or (batch, pomo, head_num, node_num_q, key_dim)
    # k,v shape: (batch, head_num, node_num_kv, key_dim) or (batch, pomo, head_num, node_num_kv, key_dim)
    # mask.shape: (batch, problem)  AM use it
    # mask.shape: (batch, pomo, problem) POMO use it

    batch_size = q.size(0)
    head_num = q.size(-3)
    n = q.size(-2) # pomo or node_num_q
    key_dim = q.size(-1)

    score = torch.matmul(q, k.transpose(-2, -1))
    if sharp == False:
        score_scaled = score / torch.sqrt(
            torch.tensor(key_dim if score_dim is None else score_dim, dtype=torch.float)
        )
    else:
        score_scaled = score
    # shape: (batch, head_num, pomo, node_num_kv) or (batch, pomo, head_num, node_num_q, node_num_kv)

    if mask is not None:
        if q.dim() == 4:
            score_scaled = score_scaled + mask[:, None, :, :].expand(-1, head_num, n, -1)
        elif q.dim() == 5:
            score_scaled = score_scaled + mask[:, :, None, None, :].expand(-1, -1, head_num, n, -1)
        else:
            raise RuntimeError(f"Invalid mask dimension: {mask.dim()}. It should be either 2 or 3.")

    weights = torch.softmax(score_scaled, dim=-1)
    # shape (batch, head_num, pomo, node_num_kv) or (batch, pomo, head_num, node_num_q, node_num_kv)

    assert not torch.isnan(weights).any(), "Attention weights contain NaNs, but it should not have any nans."

    out = torch.matmul(weights, v)
    # shape (batch, head_num, pomo, key_dim) or (batch, pomo, head_num, node_num_q, key_dim)
    out_transposed = out.transpose(-3, -2)
    # shape (batch, pomo, head_num, key_dim) or (batch, pomo, node_num_q, head_num, key_dim)
    if q.dim() == 4:
        out_concat = out_transposed.reshape(batch_size, n, head_num * key_dim)
    elif q.dim() == 5:
        out_concat = out_transposed.reshape(batch_size, -1, n, head_num * key_dim)
    else:
        raise RuntimeError(f"Invalid q dimension: {q.dim()}. It should be either 4 or 5.")

    return out_concat

def positional_encoding_DIFUSCO(x, embed_dim, max_timescale, min_timescale):
    """
    :param x: shape: (batch, pomo, local, 2)
    :return: positional encoding: (local, embed_dim)
    """
    device = x.device
    num_timescale = embed_dim // 2

    log_timescale_increment = (
        np.log(float(max_timescale) / float(min_timescale)) / num_timescale
    )

    inv_timescales = min_timescale * torch.exp(
        torch.arange(num_timescale, dtype = torch.float32) * -log_timescale_increment
    ).to(device)

    if len(x.size()) > 1:
        max_length = x.size()[2]
    elif len(x.size()) == 1:
        max_length = x.size()[-1]
    position = torch.tensor(x, dtype = torch.float32, device = device)
    scaled_time = position.unsqueeze(1) * inv_timescales.unsqueeze(0)
    # signal = torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim = 1)
    signal = torch.cat([torch.cos(scaled_time), torch.sin(scaled_time)], dim = 1)
    signal = F.pad(signal, (0, 0, 0, embed_dim % 2))
    # signal = signal.view(1, max_length, embed_dim)

    return signal

def positional_encoding_ELG(x, embed_dim, max_timescale, min_timescale):
    """
    :param x: shape: (batch, pomo, local, 2)
    :return: positional encoding: (local, embed_dim)
    """
    device = x.device
    num_timescale = embed_dim // 2

    log_timescale_increment = (
        np.log(float(max_timescale) / float(min_timescale)) /
        max(num_timescale - 1, 1)
    )

    inv_timescales = min_timescale * torch.exp(
        torch.arange(num_timescale, dtype = torch.float32) * -log_timescale_increment
    ).to(device)

    if len(x.size()) > 1:
        max_length = x.size()[2]
    elif len(x.size()) == 1:
        max_length = x.size()[-1]

    position = torch.arange(max_length, dtype = torch.float32, device = device)
    scaled_time = position.unsqueeze(1) * inv_timescales.unsqueeze(0)
    signal = torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim = 1)
    signal = F.pad(signal, (0, 0, 0, embed_dim % 2))
    # signal = signal.view(1, max_length, embed_dim)

    return signal

def positional_encoding_init(n_position, emb_dim):
    ''' Init the sinusoid position encoding table '''

    # keep dim 0 for padding token position encoding zero vector
    position_enc = np.array([
        [pos / np.power(10000, 2 * (j // 2) / emb_dim) for j in range(emb_dim)]
        if pos != 0 else np.zeros(emb_dim) for pos in range(n_position)])

    position_enc[1:, 0::2] = np.sin(position_enc[1:, 0::2])  # dim 2i
    position_enc[1:, 1::2] = np.cos(position_enc[1:, 1::2])  # dim 2i+1
    return torch.from_numpy(position_enc).type(torch.FloatTensor)




class SkipConnection(nn.Module):
    """
    The skip connection module which can contain other needed variable beside input.

    For example, if the layers in skip connection is layer1 whose variable is only input, layer2 whose variables are input
    and masks, layer3 whose variable is only input, layer4 whose variables are input and query. Suppose layer2 don't need
    masks when training, then the params in skipconnection.forward() should be [{}, {'query': query}].

    That means if one of the layer in skipconnection need extra variables beside input when training, the length of the
    list should be the number of layers that have extra variables. But the dict of the layer that doesn't need extra variables
    when training could be empty. The sort of dicts in the list must follow the layers that have extra variables in skipconnection.
    """
    def __init__(self, *args):
        super().__init__()
        self.params = {}
        for idx, module in enumerate(args):
            self.add_module(str(idx), module)

            if isinstance(module, SkipConnection):
                self.params[idx] = module.params
            else:
                module_params = inspect.signature(module.forward).parameters
                i = 0
                for name, parameter in module_params.items():
                    if i==0:
                        i+=1
                        pass
                    elif module_params[name].default == inspect.Parameter.empty:
                        if idx not in self.params.keys():
                            self.params[idx] = {}
                        self.params[idx][name] = None
                    else:
                        if idx not in self.params.keys():
                            self.params[idx] = {}
                        self.params[idx][name] = parameter.default

    def forward(self, x, param=[], weights=None):
        out = x
        for layer_order, module in self._modules.items():
            layer_order = int(layer_order)
            if isinstance(module, SkipConnection):
                out = module.forward(out, param)
            else:
                if layer_order in self.params.keys():
                    if len(param) != 0 and len(param[0]) != 0:
                        for key, value in param[0].items():
                            self.params[layer_order][key] = value
                        param.pop(0)
                    if weights is not None:
                        self.params[layer_order]['weights'] = weights
                    out = module(out, **(self.params[layer_order]))
                else:
                    out = module(out)
        out = out + x

        return out

class Normalization(nn.Module):
    def __init__(self, embed_dim, normalization="batch", **kwargs):
        super(Normalization, self).__init__()

        normalizer_class = {
            "batch": nn.BatchNorm1d,
            "batch_no_track": nn.BatchNorm1d,
            "instance": nn.InstanceNorm1d,
            "layer": nn.LayerNorm,
        }.get(normalization, None) # None is for no normalization

        if normalizer_class is None:
            self.normalizer = None
        else:
            if normalization == "layer":
                self.normalizer = normalizer_class(embed_dim, eps = 1e-5 if kwargs.get('eps') is None else kwargs.get('eps'),
                                                   elementwise_affine = True if kwargs.get('elementwise_affine') is None else kwargs.get('elementwise_affine') )
            elif normalization == "batch_no_track":
                self.normalizer = normalizer_class(embed_dim, affine=True,track_running_stats=False)
            else:
                self.normalizer = normalizer_class(embed_dim, affine=True) #bug, affine=True

    def forward(self, x, weights=None):
        if x.dim() == 3:
            x_ = x.unsqueeze(1)
        elif x.dim() == 4:
            x_ = x

        if weights is None:
            if isinstance(self.normalizer, nn.BatchNorm1d):
                out = self.normalizer(x_.view(-1, x.size(-1))).view(*x_.size())
            elif isinstance(self.normalizer, nn.InstanceNorm1d):
                out = self.normalizer(x_.reshape(-1, x_.size(-2), x_.size(-1)).permute(0, 2, 1)).permute(0, 2, 1).reshape(*x_.size())
            elif isinstance(self.normalizer, nn.LayerNorm):
                out = self.normalizer(x_.reshape(-1, x_.size(-2), x_.size(-1))).reshape(*x_.size())
            else:
                out = x
        else:
            if isinstance(self.normalizer, nn.BatchNorm1d):
                out = F.batch_norm(x_.view(-1, x.size(-1)),
                                          running_mean=self.normalizer.running_mean, running_var=self.normalizer.running_var,
                                          weight=weights['weight'], bias=weights['bias'], training=True).view(*x_.size())
            elif isinstance(self.normalizer, nn.InstanceNorm1d):
                out = F.instance_norm(x_.reshape(-1, x_.size(-2), x_.size(-1)).permute(0, 2, 1), weight=weights['weight'], bias=weights['bias']).permute(0, 2, 1).reshape(*x_.size())
            else:
                out = x

        if x.dim() == 3 and x.shape[1] != 1:
            out = out.squeeze(1)

        return out

class FeedForward(nn.Module):
    def __init__(self, embedding_dim,ff_hidden_dim,activation="relu",inplace = False):
        super().__init__()

        self.W1 = nn.Linear(embedding_dim, ff_hidden_dim)
        self.W2 = nn.Linear(ff_hidden_dim, embedding_dim)
        self.activation_type=activation
        if activation=="Gelu":
            self.activation=nn.GELU()
        self.inplace = inplace
    def forward(self, input1,weights=None):
        # input.shape: (batch, pomo, problem, embedding)
        if self.activation_type=="Gelu":
            return self.W2(self.activation(self.W1(input1)))
        else:
            if weights is not None:
                output = F.relu(F.linear(input1, weights['weight1'], bias=weights['bias1']))
                return F.linear(output, weights['weight2'], bias=weights['bias2'])
            else:
                return self.W2(F.relu(self.W1(input1),inplace=self.inplace))
