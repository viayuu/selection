import torch.nn as nn
import torch
import torch.nn.functional as F
import math
import numpy as np

from EasyNCO.neural_solvers.utils import GraphMeanEmbedding
from EasyNCO.neural_solvers.backbones import TransformerNet,FeedForward,Normalization



def dynamic_layer_norm(x, eps=1e-5):
    # 计算最后两个维度上的 mean/std
    mean = x.mean(dim=(-2, -1), keepdim=True)
    std = x.std(dim=(-2, -1), keepdim=True, unbiased=False)
    return (x - mean) / (std + eps)

class MutilInputSequential(nn.Sequential):
    def forward(self, *inputs):
        for module in self._modules.values():
            if type(inputs) == tuple:
                inputs = module(*inputs)
            else:
                inputs = module(inputs)
        return inputs

class FFandNormsubLayer(nn.Module):
    def __init__(
            self,
            embed_dim: int = 128,
            feedforward_hidden: int = 512,
            normalization: str = "layer",
    ):
        super(FFandNormsubLayer, self).__init__()

        self.FF1 = FeedForward(embed_dim,feedforward_hidden,inplace=True)
        self.FF2 = FeedForward(embed_dim, feedforward_hidden, inplace=True)

    def forward(self, input1, input2):
        # FF and Residual connection
        out1 = self.FF1(input1)
        out2 = self.FF2(input2)

        return dynamic_layer_norm(out1 + input1), dynamic_layer_norm(out2 + input2)

# implements the propsoed Multi-head DAC-Att module
class DACT_Att(nn.Module):
    def __init__(
            self,
            n_heads,
            input_dim,
            embed_dim=None,
            val_dim=None,
            key_dim=None
    ):
        super(DACT_Att, self).__init__()

        self.n_heads = n_heads

        self.key_dim = self.val_dim = embed_dim // n_heads
        self.input_dim = input_dim
        self.embed_dim = embed_dim

        self.norm_factor = 1 / math.sqrt(1 * self.key_dim)

        # W_h^Q in the paper
        self.W_query_node = nn.Parameter(torch.Tensor(n_heads, self.input_dim, self.key_dim))
        # W_g^Q in the paper
        self.W_query_pos = nn.Parameter(torch.Tensor(n_heads, self.input_dim, self.key_dim))
        # W_h^K in the paper
        self.W_key_node = nn.Parameter(torch.Tensor(n_heads, self.input_dim, self.key_dim))
        # W_g^K in the paper
        self.W_key_pos = nn.Parameter(torch.Tensor(n_heads, self.input_dim, self.key_dim))

        # W_h^V and W_h^Vref in the paper
        self.W_val_node = nn.Parameter(torch.Tensor(2 * n_heads, self.input_dim, self.val_dim))
        # W_g^V and W_g^Vref in the paper
        self.W_val_pos = nn.Parameter(torch.Tensor(2 * n_heads, self.input_dim, self.val_dim))

        # W_h^O and W_g^O in the paper
        if embed_dim is not None:
            self.W_out_node = nn.Parameter(torch.Tensor(n_heads, 2 * self.key_dim, embed_dim))
            self.W_out_pos = nn.Parameter(torch.Tensor(n_heads, 2 * self.key_dim, embed_dim))

        self.init_parameters()

    def init_parameters(self):

        for param in self.parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(self, h_node_in, h_pos_in):  # input (NFEs, PFEs)

        # h,g should be (batch_size, graph_size, input_dim)
        batch_size, graph_size, input_dim = h_node_in.size()

        shp = (self.n_heads, batch_size, graph_size, -1)
        shp_v = (2, self.n_heads, batch_size, graph_size, -1)

        h_node = h_node_in.contiguous().view(-1, input_dim)
        h_pos = h_pos_in.contiguous().view(-1, input_dim)

        Q_node = torch.matmul(h_node, self.W_query_node).view(shp)
        Q_pos = torch.matmul(h_pos, self.W_query_pos).view(shp)

        K_node = torch.matmul(h_node, self.W_key_node).view(shp)
        K_pos = torch.matmul(h_pos, self.W_key_pos).view(shp)

        V_node = torch.matmul(h_node, self.W_val_node).view(shp_v)
        V_pos = torch.matmul(h_pos, self.W_val_pos).view(shp_v)

        # Get attention correlations and norm by softmax
        node_correlations = self.norm_factor * torch.matmul(Q_node, K_node.transpose(2, 3))
        pos_correlations = self.norm_factor * torch.matmul(Q_pos, K_pos.transpose(2, 3))
        attn1 = F.softmax(node_correlations, dim=-1)  # head, bs, n, n
        attn2 = F.softmax(pos_correlations, dim=-1)  # head, bs, n, n

        heads_node_1 = torch.matmul(attn1, V_node[0])  # self-attn
        heads_node_2 = torch.matmul(attn2, V_node[1])  # cross-aspect ref attn

        heads_pos_1 = torch.matmul(attn1, V_pos[0])  # cross-aspect ref attn
        heads_pos_2 = torch.matmul(attn2, V_pos[1])  # self-attn

        heads_node = torch.cat((heads_node_1, heads_node_2), -1)
        heads_pos = torch.cat((heads_pos_1, heads_pos_2), -1)

        # get output
        out_node = torch.mm(
            heads_node.permute(1, 2, 0, 3).contiguous().view(-1, self.n_heads * 2 * self.val_dim),
            self.W_out_node.view(-1, self.embed_dim)
        ).view(batch_size, graph_size, self.embed_dim)

        out_pos = torch.mm(
            heads_pos.permute(1, 2, 0, 3).contiguous().view(-1, self.n_heads * 2 * self.val_dim),
            self.W_out_pos.view(-1, self.embed_dim)
        ).view(batch_size, graph_size, self.embed_dim)

        return out_node, out_pos  # dual-aspect representation (NFEs, PFEs)

class DACT_ATTsublayer(nn.Module):
    def __init__(
            self,
            n_heads,
            embed_dim,
            feed_forward_hidden,
            normalization='layer',
    ):
        super(DACT_ATTsublayer, self).__init__()

        self.MHA = DACT_Att(
            n_heads,
            input_dim=embed_dim,
            embed_dim=embed_dim
        )

    def forward(self, input1, input2):
        # Attention and Residual connection
        out1, out2 = self.MHA(input1, input2)

        # Normalization
        return dynamic_layer_norm(out1 + input1), dynamic_layer_norm(out2 + input2)

class DACT_EncoderLayer(nn.Module):
    def __init__(
            self,
            embed_dim: int = 128,
            num_heads: int = 8,
            normalization: str = "layer",  # norm can be None, batch, or instance
            feedforward_hidden: int = 512,
    ):
        super(DACT_EncoderLayer, self).__init__()


        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.normalization = normalization
        self.feedforward_hidden = feedforward_hidden

        self.MHA_sublayer = DACT_ATTsublayer(
            num_heads,
            embed_dim,
            feedforward_hidden,
            normalization=normalization,
        )

        self.FFandNorm_sublayer = FFandNormsubLayer(
            embed_dim,
            feedforward_hidden,
            normalization=normalization,
        )

    def forward(self, input1, input2):
        out1, out2 = self.MHA_sublayer(input1, input2)
        return self.FFandNorm_sublayer(out1, out2)

# implements the initilization of the NFEs and the PFEs
class EmbeddingNet(nn.Module):

    def __init__(
            self,
            node_dim,
            embedding_dim,
    ):
        super(EmbeddingNet, self).__init__()
        self.node_dim = node_dim
        self.embedding_dim = embedding_dim
        self.embedder = nn.Linear(node_dim, embedding_dim, bias=False)
        self.pattern = None
        # Two ways for generalizing CPEs:
        # -- 1. Use the target size CPE directly (default)
        # self.pattern = self.Cyclic_Positional_Encoding(seq_length, embedding_dim)
        # -- 2. Use the original size CPE: reuse the wavelength of the original size but make it compatible with the target size (by duplicating or discarding)
        # way 1 works well for most of cases
        # original_size, target_size = 100, 150
        # self.pattern = self.Cyclic_Positional_Encoding(original_size, embedding_dim, target_size=target_size)

    def init_parameters(self):

        for param in self.parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def basesin(self, x, T, fai=0):
        return np.sin(2 * np.pi / T * np.abs(np.mod(x, 2 * T) - T) + fai)

    def basecos(self, x, T, fai=0):
        return np.cos(2 * np.pi / T * np.abs(np.mod(x, 2 * T) - T) + fai)

    # implements the CPE
    def Cyclic_Positional_Encoding(self, n_position, emb_dim, mean_pooling=True, target_size=None):

        Td_set = np.linspace(np.power(n_position, 1 / (emb_dim // 2)), n_position, emb_dim // 2, dtype='int')
        x = np.zeros((n_position, emb_dim))

        for i in range(emb_dim):
            Td = Td_set[i // 3 * 3 + 1] if (i // 3 * 3 + 1) < (emb_dim // 2) else Td_set[-1]
            fai = 0 if i <= (emb_dim // 2) else 2 * np.pi * ((-i + (emb_dim // 2)) / (emb_dim // 2))
            longer_pattern = np.arange(0, np.ceil((n_position) / Td) * Td, 0.01)
            if i % 2 == 1:
                x[:, i] = self.basecos(longer_pattern, Td, fai)[
                    np.linspace(0, len(longer_pattern), n_position, dtype='int', endpoint=False)]
            else:
                x[:, i] = self.basesin(longer_pattern, Td, fai)[
                    np.linspace(0, len(longer_pattern), n_position, dtype='int', endpoint=False)]

        pattern = torch.from_numpy(x).type(torch.FloatTensor)
        pattern_sum = torch.zeros_like(pattern)

        # for generalization (way 2): reuse the wavelength of the original size but make it compatible with the target size (by duplicating or discarding)
        if target_size is not None:
            pattern = pattern[np.ceil(np.linspace(0, n_position - 1, target_size))]
            pattern_sum = torch.zeros_like(pattern)
            n_position = target_size

        # averaging the adjacient embeddings if needed (optional, almost the same performance)
        arange = torch.arange(n_position,device="cpu")
        pooling = [0] if not mean_pooling else [-2, -1, 0, 1, 2]
        time = 0
        for i in pooling:
            time += 1
            index = (arange + i + n_position) % n_position
            pattern_sum += pattern.gather(0, index.view(-1, 1).expand_as(pattern))
        pattern = 1. / time * pattern_sum - pattern.mean(0)

        return pattern

    def position_encoding(self, solutions, embedding_dim, visited_time):

        # batch: batch_size, problem_size, dim
        batch_size, seq_length = solutions.size()
        arange = torch.arange(batch_size)

        # expand for every batch
        CPE_embeddings = self.pattern.expand(batch_size, seq_length, embedding_dim).clone().to(solutions.device)

        # get index according to the solutions
        if visited_time is None:
            visited_time = torch.zeros((batch_size, seq_length), device=solutions.device)
            pre = torch.zeros((batch_size), device=solutions.device).long()
            for i in range(seq_length):
                visited_time[arange, solutions[arange, pre]] = i + 1
                pre = solutions[arange, pre]
        index = (visited_time % seq_length).long().unsqueeze(-1).expand(batch_size, seq_length, embedding_dim)

        return torch.gather(CPE_embeddings, 1, index), visited_time.long()

    def forward(self, x, solutions, visited_time=None):
        PFEs, visited_time = self.position_encoding(solutions, self.embedding_dim, visited_time)
        NFEs = self.embedder(x)
        return NFEs, PFEs, visited_time

class DACTCritic_Encoder(nn.Module):
    def __init__(
            self,
            env_name: str = "tsp",  # tsp or cvrp
            embed_dim: int = 128,
            num_heads: int = 8,
            qkv_dim: int = 128,
            num_encoder_layers: int = 3,
            normalization: str = "batch",  # norm can be None, batch, or instance
            feedforward_hidden: int = 512,
            net: nn.Module = None,
    ):
        super(DACTCritic_Encoder, self).__init__()

        self.env_name = env_name
        self.embed_dim = embed_dim
        self.qkv_dim = qkv_dim
        self.num_encoder_layers = num_encoder_layers
        self.num_heads = num_heads
        self.normalization = normalization
        self.feedforward_hidden = feedforward_hidden

        self.net = (TransformerNet(
            num_encoder_layers,
            num_heads,
            qkv_dim,
            embed_dim * 2,
            normalization,
            feedforward_hidden * 2,
        ) if net is None
                    else net)

    def forward(self, input):
        # get concatenated input
        h_features = torch.cat(input, -1).detach()

        # pass through encoder
        h_em = self.net(h_features)

        return h_em

class DACT_Encoder(nn.Module):
    def __init__(
            self,
            env_name: str = "tsp",  # tsp or cvrp
            embed_dim: int = 128,
            node_dim: int =2,
            num_heads: int = 8,
            qkv_dim: int = 128,
            num_encoder_layers: int = 3,
            normalization: str = "batch",  # norm can be None, batch, or instance
            feedforward_hidden: int = 512,
    ):
        super(DACT_Encoder, self).__init__()

        self.env_name = env_name
        self.embed_dim = embed_dim
        self.qkv_dim = qkv_dim
        self.num_encoder_layers = num_encoder_layers
        self.num_heads = num_heads
        self.normalization = normalization
        self.feedforward_hidden = feedforward_hidden
        self.node_dim = node_dim




        self.embedder = EmbeddingNet(
            self.node_dim,
            self.embed_dim,
            )

        self.encoder = MutilInputSequential(*(
            DACT_EncoderLayer(num_heads=self.num_heads,
                             embed_dim=self.embed_dim,
                             feedforward_hidden=self.feedforward_hidden,
                             normalization=self.normalization,
                             )
            for _ in range(self.num_encoder_layers)))  # stack L layers


    def forward(self,td,**kwargs):
        if self.env_name == 'tsp':
            NFE, PFE, visited_time = self.embedder(td['locs'], td['solution'], None)

        elif self.env_name == 'cvrp':
            x_in = kwargs.get('input')
            visited_time = kwargs.get('visited_time')
            NFE, PFE, visited_time = self.embedder(x_in, td['solution'], visited_time=visited_time)

        h_em, g_em = self.encoder(NFE,PFE)
        return h_em, g_em