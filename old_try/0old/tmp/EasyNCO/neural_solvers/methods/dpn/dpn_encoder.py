import math
import torch.nn as nn
from tensordict import TensorDict
import torch
import numpy as np
from EasyNCO.neural_solvers.backbones import MultiHeadAttentionLayer
from EasyNCO.neural_solvers.backbones.Transformer.attr_component import (
    reshape_by_heads,
    FeedForward,
    multi_head_attention,
)

from EasyNCO.utils.utils import getLogger


logger = getLogger(__name__)


class RotatePostionalEncoding(nn.Module):
    """
    compute sinusoid encoding.
    """

    def __init__(self, d_model, max_len):
        """
        constructor of sinusoid encoding class
        :param d_model: dimension of model
        :param max_len: max sequence length
        """
        super(RotatePostionalEncoding, self).__init__()

        # same size with input matrix (for adding with input matrix)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(-1)
        # (output_dim//2)
        ids = torch.arange(0, d_model // 2, dtype=torch.float)
        theta = torch.pow(1000, -2 * ids / d_model)

        # (max_len, output_dim//2)
        embeddings = position * theta

        # (max_len, output_dim//2, 2)
        self.cos_embeddings = torch.sin(embeddings)
        self.sin_embeddings = torch.cos(embeddings)

    def forward(self, input):
        # self.encoding
        # [max_len = 512, d_model = 512]

        _, seq_len, _ = input.size()
        cos_pos = (
            self.cos_embeddings[None, :seq_len, :]
            .repeat_interleave(2, dim=-1)
            .to(input.device)
        )
        sin_pos = (
            self.sin_embeddings[None, :seq_len, :]
            .repeat_interleave(2, dim=-1)
            .to(input.device)
        )

        # q,k: (bs, head, max_len, output_dim)
        input2 = torch.stack([-input[..., 1::2], input[..., ::2]], dim=-1)
        input2 = input2.reshape(input.shape)

        output = input * cos_pos + input2 * sin_pos
        # [seq_len = 30, d_model = 512]
        # it will add with tok_emb : [128, 30, 512]
        return output


class DPNEncoder(nn.Module):

    def __init__(
        self,
        env_name: str = "mtsp",
        embed_dim: int = 128,
        num_heads: int = 8,
        qkv_dim: int = 16,  # in the original code, num_heads is val_dim = embed_dim // n_heads
        num_layers: int = 6,
        feedforward_hidden: int = 512,
        positional_encoding: nn.Module = None,
    ):
        super(DPNEncoder, self).__init__()
        self.env_name = env_name
        self.init_embed_depot = nn.Linear(2, embed_dim)
        self.init_embed_agent = nn.Linear(2, embed_dim)
        self.init_embed = nn.Linear(2, embed_dim)
        self.positional_encoding = (
            RotatePostionalEncoding(embed_dim, 10000)
            if positional_encoding is None
            else positional_encoding
        )
        self.alpha = nn.Parameter(torch.Tensor([1]))
        self.pos_emb_proj = nn.Sequential(nn.Linear(embed_dim, embed_dim, bias=False))

        if self.env_name == "mtsp":
            self.layers = nn.Sequential(
                *(
                    Partition_Navigation_Encoder_TSP(
                        num_heads=num_heads,
                        embed_dim=embed_dim,
                        qkv_dim=qkv_dim,
                        feed_forward_hidden=feedforward_hidden,
                    )
                    for _ in range(num_layers)
                )
            )
        elif self.env_name == "mpdp":
            self.init_embed_pick = nn.Linear(4, embed_dim)
            self.init_embed_delivery = nn.Linear(2, embed_dim)
            self.layers = nn.Sequential(
                *(
                    Partition_Navigation_Encoder_PDP(
                        num_heads=num_heads,
                        embed_dim=embed_dim,
                        qkv_dim=qkv_dim,
                        feed_forward_hidden=feedforward_hidden,
                    )
                    for _ in range(num_layers)
                )
            )
        elif self.env_name == "mdvrp" or self.env_name == "fmdvrp":
            self.beta = nn.Parameter(torch.Tensor(embed_dim))
            self.layers = nn.Sequential(
                *(
                    Partition_Navigation_Encoder_MDVRRP(
                        num_heads=num_heads,
                        embed_dim=embed_dim,
                        qkv_dim=qkv_dim,
                        feed_forward_hidden=feedforward_hidden,
                    )
                    for _ in range(num_layers)
                )
            )

    def forward(self, td: TensorDict):
        if self.env_name == "mtsp":
            self.agent_num = int(td["agent_num"][0])
            depot_embeddings = self.init_embed_depot(td["locs"][:, :1, :])
            place_embeddings = self.init_embed_agent(td["locs"][:, :1, :])
            depot_embeddings = depot_embeddings.repeat(1, self.agent_num, 1)
            place_embeddings = place_embeddings.repeat(1, self.agent_num, 1)
            positional_embeddings = self.positional_encoding(place_embeddings)
            positional_embeddings = self.alpha * self.pos_emb_proj(
                positional_embeddings
            )
            agent_embeddings = depot_embeddings + positional_embeddings
            node_embeddings = self.init_embed(td["locs"][:, self.agent_num :, :])

            for layer in self.layers:
                agent_embeddings, node_embeddings = layer(
                    agent_embeddings, node_embeddings
                )
            h = torch.cat([agent_embeddings, node_embeddings], dim=1)
            return (
                h,  # (batch_size, graph_size, embed_dim)
                h.mean(
                    dim=1
                ),  # average to get embedding of graph, (batch_size, embed_dim)
            )
        elif self.env_name == "mpdp":
            self.agent_num = int(td["agent_num"][0])
            n_loc = td["locs"].size(1) - self.agent_num
            new_depot = td["locs"][:, : self.agent_num, :]
            new_input = td["locs"][:, self.agent_num :, :]
            depot_embeddings = self.init_embed_depot(new_depot)
            place_embeddings = self.init_embed_agent(new_depot)
            self.num_request = n_loc // 2
            positional_embeddings = self.positional_encoding(place_embeddings)
            positional_embeddings = self.alpha * self.pos_emb_proj(
                positional_embeddings
            )
            agent_embeddings = depot_embeddings + positional_embeddings
            feature_pick = torch.cat(
                [new_input[:, : n_loc // 2, :], new_input[:, n_loc // 2 :, :]], -1
            )
            feature_delivery = new_input[
                :, n_loc // 2 :, :
            ]  # [batch_size, graph_size//2, 2]
            embed_pick = self.init_embed_pick(feature_pick)
            embed_delivery = self.init_embed_delivery(feature_delivery)
            node_embeddings = torch.cat([embed_pick, embed_delivery], 1)
            for layer in self.layers:
                agent_embeddings, node_embeddings = layer(
                    agent_embeddings, node_embeddings
                )
            h = torch.cat([agent_embeddings, node_embeddings], dim=1)
            return (
                h,  # (batch_size, graph_size, embed_dim)
                h.mean(
                    dim=1
                ),  # average to get embedding of graph, (batch_size, embed_dim)
            )
        elif self.env_name == "mdvrp" or self.env_name == "fmdvrp":
            self.agent_num = int(td["agent_num"][0])
            self.depot_num = int(td["depot_num"][0])
            # Embedding of depot
            depot_embeddings = self.init_embed_depot(td["locs"][:, : self.depot_num, :])
            # Make the depot embedding the same for all agents
            agent_embeddings = self.beta[None, None, :].repeat(
                depot_embeddings.size(0), self.agent_num, 1
            )
            positional_embeddings = self.positional_encoding(agent_embeddings)
            agent_embeddings = self.alpha * self.pos_emb_proj(positional_embeddings)
            node_embeddings = self.init_embed(td["locs"][:, self.depot_num :, :])
            for layer in self.layers:
                agent_embeddings, depot_embeddings, node_embeddings = layer(
                    agent_embeddings, depot_embeddings, node_embeddings
                )
            h = torch.cat((depot_embeddings, node_embeddings), dim=1)
            return (
                agent_embeddings,  # (batch_size, graph_size, embed_dim)
                h,  # average to get embedding of graph, (batch_size, embed_dim)
            )


class Partition_Navigation_Encoder_MDVRRP(nn.Module):
    def __init__(
        self,
        num_heads,
        embed_dim,
        qkv_dim,
        feed_forward_hidden=512,
    ):
        super(Partition_Navigation_Encoder_MDVRRP, self).__init__()
        self.num_heads = num_heads
        self.MHA_layers = nn.ModuleList(
            [
                MultiHeadAttentionLayer(embed_dim, num_heads, qkv_dim, bias_combine=True)
                for _ in range(3)
            ]
        )
        self.FF_layers = nn.ModuleList(
            [Feed_Forward_Module(embed_dim, feed_forward_hidden) for _ in range(7)]
        )
        self.addAndNormalization = nn.ModuleList(
            [Add_And_Normalization_Module(embed_dim) for _ in range(7)]
        )
        self.RoPE = RotatePostionalEncoding(embed_dim, 10000)
        self.Wq_ad = nn.Linear(embed_dim, num_heads * qkv_dim, bias=False)
        self.Wk_ad = nn.Linear(embed_dim, num_heads * qkv_dim, bias=False)
        self.Wv_ad = nn.Linear(embed_dim, num_heads * qkv_dim, bias=False)
        self.Wq2_ad = nn.Linear(embed_dim, num_heads * qkv_dim, bias=False)
        self.Wk2_ad = nn.Linear(embed_dim, num_heads * qkv_dim, bias=False)
        self.Wv2_ad = nn.Linear(embed_dim, num_heads * qkv_dim, bias=False)
        self.multi_head_combine_ad = nn.Linear(num_heads * qkv_dim, embed_dim)
        self.multi_head_combine2_ad = nn.Linear(num_heads * qkv_dim, embed_dim)
        self.Wq_an = nn.Linear(embed_dim, num_heads * qkv_dim, bias=False)
        self.Wk_an = nn.Linear(embed_dim, num_heads * qkv_dim, bias=False)
        self.Wv_an = nn.Linear(embed_dim, num_heads * qkv_dim, bias=False)
        self.Wq2_an = nn.Linear(embed_dim, num_heads * qkv_dim, bias=False)
        self.Wk2_an = nn.Linear(embed_dim, num_heads * qkv_dim, bias=False)
        self.Wv2_an = nn.Linear(embed_dim, num_heads * qkv_dim, bias=False)
        self.multi_head_combine_an = nn.Linear(num_heads * qkv_dim, embed_dim)
        self.multi_head_combine2_an = nn.Linear(num_heads * qkv_dim, embed_dim)

    def forward(self, agent_emb, depot_emb, node_emb):
        # self
        multi_head_out_s = self.MHA_layers[0](node_emb)
        node_hidden_2 = self.addAndNormalization[0](node_emb, multi_head_out_s)
        node_out_2 = self.FF_layers[0](node_hidden_2)
        # dn
        multi_head_out1_dn = self.MHA_layers[1](depot_emb, kv_input=node_out_2)
        depot_hidden = self.addAndNormalization[1](depot_emb, multi_head_out1_dn)
        depot_out_ = self.FF_layers[1](depot_hidden)
        multi_head_out2_dn = self.MHA_layers[2](
            node_out_2, kv_input=depot_out_, sharp=True
        )
        node_hidden = self.addAndNormalization[2](node_out_2, multi_head_out2_dn)
        node_out_ = self.FF_layers[2](node_hidden)
        q_ad = reshape_by_heads(
            self.RoPE(self.Wq_ad(agent_emb)), head_num=self.num_heads
        )
        k_ad = reshape_by_heads(self.Wk_ad(depot_out_), head_num=self.num_heads)
        v_ad = reshape_by_heads(self.Wv_ad(depot_out_), head_num=self.num_heads)
        # q shape: (batch, HEAD_NUM, problem, KEY_DIM)
        out_concat_ad = multi_head_attention(
            q_ad, k_ad, v_ad, sharp=True
        )  # shape: (B, n, head_num*key_dim)
        multi_head_out1_ad = self.multi_head_combine_ad(
            out_concat_ad
        )  # shape: (B, n, embedding_dim)
        agent_hidden = self.addAndNormalization[3](agent_emb, multi_head_out1_ad)
        agent_out_ = self.FF_layers[3](agent_hidden)
        q2_ad = reshape_by_heads(self.Wq2_ad(depot_out_), head_num=self.num_heads)
        k2_ad = reshape_by_heads(
            self.RoPE(self.Wk2_ad(agent_out_)), head_num=self.num_heads
        )
        v2_ad = reshape_by_heads(
            self.RoPE(self.Wv2_ad(agent_out_)), head_num=self.num_heads
        )
        # k = reshape_by_heads(input2, head_num=head_num)
        out_concat2_ad = multi_head_attention(
            q2_ad, k2_ad, v2_ad
        )  # shape: (B, n, head_num*key_dim)
        multi_head_out2_ad = self.multi_head_combine2_ad(
            out_concat2_ad
        )  # shape: (B, n, embedding_dim)
        depot_hidden = self.addAndNormalization[4](depot_out_, multi_head_out2_ad)
        depot_out = self.FF_layers[4](depot_hidden)
        q_an = reshape_by_heads(
            self.RoPE(self.Wq_an(agent_out_)), head_num=self.num_heads
        )
        k_an = reshape_by_heads(self.Wk_an(node_out_), head_num=self.num_heads)
        v_an = reshape_by_heads(self.Wv_an(node_out_), head_num=self.num_heads)
        # q shape: (batch, HEAD_NUM, problem, KEY_DIM)
        out_concat_an = multi_head_attention(
            q_an, k_an, v_an
        )  # shape: (B, n, head_num*key_dim)
        multi_head_out1_an = self.multi_head_combine_an(
            out_concat_an
        )  # shape: (B, n, embedding_dim)
        agent_hidden = self.addAndNormalization[5](agent_out_, multi_head_out1_an)
        agent_out = self.FF_layers[5](agent_hidden)
        q2_an = reshape_by_heads(self.Wq2_an(node_out_), head_num=self.num_heads)
        k2_an = reshape_by_heads(
            self.RoPE(self.Wk2_an(agent_out)), head_num=self.num_heads
        )
        v2_an = reshape_by_heads(
            self.RoPE(self.Wv2_an(agent_out)), head_num=self.num_heads
        )
        out_concat2_an = multi_head_attention(
            q2_an, k2_an, v2_an, sharp=True
        )  # shape: (B, n, head_num*key_dim)
        multi_head_out2_an = self.multi_head_combine2_an(
            out_concat2_an
        )  # shape: (B, n, embedding_dim)
        node_hidden = self.addAndNormalization[6](node_out_, multi_head_out2_an)
        node_out = self.FF_layers[6](node_hidden)
        return agent_out, depot_out, node_out


class Partition_Navigation_Encoder_TSP(nn.Module):

    def __init__(
        self,
        num_heads,
        embed_dim,
        qkv_dim,
        feed_forward_hidden=512,
    ):
        super(Partition_Navigation_Encoder_TSP, self).__init__()

        self.MHA_layers = nn.ModuleList(
            [
                MultiHeadAttentionLayer(embed_dim, num_heads, qkv_dim, bias_combine=True)
                for _ in range(3)
            ]
        )
        self.FF_layers = nn.ModuleList(
            [Feed_Forward_Module(embed_dim, feed_forward_hidden) for _ in range(3)]
        )
        self.addAndNormalization = nn.ModuleList(
            [Add_And_Normalization_Module(embed_dim) for _ in range(3)]
        )

    def forward(self, agents, nodes):

        nodes_out = self.MHA_layers[2](nodes)
        nodes_out = self.addAndNormalization[2](nodes, nodes_out)
        nodes_out = self.FF_layers[2](nodes_out)
        agents_out = self.MHA_layers[0](agents, kv_input=nodes_out)
        agents_out = self.addAndNormalization[0](agents, agents_out)
        agents_out = self.FF_layers[0](agents_out)
        nodes_out_multi = self.MHA_layers[1](nodes_out, kv_input=agents_out, sharp=True)
        nodes_out = self.addAndNormalization[1](nodes_out, nodes_out_multi)
        nodes_out = self.FF_layers[1](nodes_out)
        return agents_out, nodes_out


class Partition_Navigation_Encoder_PDP(nn.Module):

    def __init__(
        self,
        num_heads,
        embed_dim,
        qkv_dim,
        feed_forward_hidden=512,
    ):
        super(Partition_Navigation_Encoder_PDP, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.W_query = nn.Parameter(torch.Tensor(num_heads, embed_dim, qkv_dim))
        self.W_key = nn.Parameter(torch.Tensor(num_heads, embed_dim, qkv_dim))
        self.W_val = nn.Parameter(torch.Tensor(num_heads, embed_dim, qkv_dim))
        # pickup
        self.W1_query = nn.Parameter(torch.Tensor(num_heads, embed_dim, qkv_dim))
        self.W2_query = nn.Parameter(torch.Tensor(num_heads, embed_dim, qkv_dim))
        self.W3_query = nn.Parameter(torch.Tensor(num_heads, embed_dim, qkv_dim))
        # delivery
        self.W4_query = nn.Parameter(torch.Tensor(num_heads, embed_dim, qkv_dim))
        self.W5_query = nn.Parameter(torch.Tensor(num_heads, embed_dim, qkv_dim))
        self.W6_query = nn.Parameter(torch.Tensor(num_heads, embed_dim, qkv_dim))
        self.multi_head_combine = nn.ModuleList(
            [nn.Linear(num_heads * qkv_dim, embed_dim) for _ in range(2)]
        )
        self.Wq_2 = nn.Linear(embed_dim, num_heads * qkv_dim, bias=False)
        self.Wq_2_pickup = nn.Linear(embed_dim, num_heads * qkv_dim, bias=False)
        self.Wq_2_delivery = nn.Linear(embed_dim, num_heads * qkv_dim, bias=False)
        self.Wk_2 = nn.Linear(embed_dim, num_heads * qkv_dim, bias=False)
        self.Wv_2 = nn.Linear(embed_dim, num_heads * qkv_dim, bias=False)
        self.addAndNormalization = nn.ModuleList(
            [Add_And_Normalization_Module(embed_dim) for _ in range(3)]
        )
        self.FF_layers = nn.ModuleList(
            [Feed_Forward_Module(embed_dim, feed_forward_hidden) for _ in range(3)]
        )
        self.addAndNormalization = nn.ModuleList(
            [Add_And_Normalization_Module(embed_dim) for _ in range(3)]
        )
        self.MHA_layers = MultiHeadAttentionLayer(
            embed_dim, num_heads, qkv_dim, bias_combine=True
        )
        self.norm_factor = 1 / math.sqrt(qkv_dim)  # See Attention is all you need

    def forward(self, agent_emb, node_emb):
        multi_head_out_3 = self.multi_head_combine[0](
            self.ha_encoder(node_emb)
        )  # shape: (B, n, embedding_dim)
        node_hidden_2 = self.addAndNormalization[0](node_emb, multi_head_out_3)
        node_out_2 = self.FF_layers[0](node_hidden_2)
        multi_head_out = self.MHA_layers(agent_emb, kv_input=node_out_2)
        agent_hidden = self.addAndNormalization[1](agent_emb, multi_head_out)
        agent_out = self.FF_layers[1](agent_hidden)
        pickup = node_out_2[:, : node_out_2.size(1) // 2, :]
        delivery = node_out_2[:, node_out_2.size(1) // 2 :, :]
        q_2 = reshape_by_heads(self.Wq_2(node_out_2), head_num=self.num_heads)
        q_2_p = reshape_by_heads(self.Wq_2_pickup(pickup), head_num=self.num_heads)
        q_2_d = reshape_by_heads(self.Wq_2_delivery(delivery), head_num=self.num_heads)
        q_2 += torch.cat((q_2_d, q_2_p), dim=2)
        k_2 = reshape_by_heads(self.Wk_2(agent_out), head_num=self.num_heads)
        v_2 = reshape_by_heads(self.Wv_2(agent_out), head_num=self.num_heads)
        # k = reshape_by_heads(input2, head_num=head_num)
        out_concat_2 = multi_head_attention(
            q_2, k_2, v_2, sharp=True
        )  # shape: (B, n, head_num*key_dim)
        multi_head_out_2 = self.multi_head_combine[1](
            out_concat_2
        )  # shape: (B, n, embedding_dim)
        node_hidden = self.addAndNormalization[2](node_out_2, multi_head_out_2)
        node_out = self.FF_layers[2](node_hidden)
        return agent_out, node_out

    def ha_encoder(self, q):
        """
        :param q: queries (batch_size, n_query, input_dim)
        """
        h = q  # compute self-attention

        # h should be (batch_size, graph_size, input_dim)
        batch_size, graph_size, input_dim = h.size()
        n_query = q.size(1)
        assert q.size(0) == batch_size
        assert q.size(2) == input_dim
        assert input_dim == self.embed_dim, "Wrong embedding dimension of input"
        hflat = h.contiguous().view(
            -1, input_dim
        )  # [batch_size * graph_size, embed_dim]
        qflat = q.contiguous().view(-1, input_dim)  # [batch_size * n_query, embed_dim]
        # last dimension can be different for keys and values
        shp = (self.num_heads, batch_size, graph_size, -1)
        shp_q = (self.num_heads, batch_size, n_query, -1)
        # pickup -> its delivery attention
        n_pick = (graph_size) // 2
        shp_delivery = (self.num_heads, batch_size, n_pick, -1)
        shp_q_pick = (self.num_heads, batch_size, n_pick, -1)
        # pickup -> all pickups attention
        shp_allpick = (self.num_heads, batch_size, n_pick, -1)
        shp_q_allpick = (self.num_heads, batch_size, n_pick, -1)
        # pickup -> all pickups attention
        shp_alldelivery = (self.num_heads, batch_size, n_pick, -1)
        shp_q_alldelivery = (self.num_heads, batch_size, n_pick, -1)
        # Calculate queries, (n_heads, n_query, graph_size, key/val_size)
        Q = torch.matmul(qflat, self.W_query).view(shp_q)
        # Calculate keys and values (n_heads, batch_size, graph_size, key/val_size)
        K = torch.matmul(hflat, self.W_key).view(shp)
        V = torch.matmul(hflat, self.W_val).view(shp)
        # pickup -> its delivery
        pick_flat = (
            h[:, :n_pick, :].contiguous().view(-1, input_dim)
        )  # [batch_size * n_pick, embed_dim]
        delivery_flat = (
            h[:, n_pick:, :].contiguous().view(-1, input_dim)
        )  # [batch_size * n_pick, embed_dim]
        # pickup -> its delivery attention
        Q_pick = torch.matmul(pick_flat, self.W1_query).view(
            shp_q_pick
        )  # (self.n_heads, batch_size, n_pick, key_size)
        K_delivery = torch.matmul(delivery_flat, self.W_key).view(
            shp_delivery
        )  # (self.n_heads, batch_size, n_pick, -1)
        V_delivery = torch.matmul(delivery_flat, self.W_val).view(
            shp_delivery
        )  # (n_heads, batch_size, n_pick, key/val_size)
        # pickup -> all pickups attention
        Q_pick_allpick = torch.matmul(pick_flat, self.W2_query).view(
            shp_q_allpick
        )  # (self.n_heads, batch_size, n_pick, -1)
        K_allpick = torch.matmul(pick_flat, self.W_key).view(
            shp_allpick
        )  # [self.n_heads, batch_size, n_pick, key_size]
        V_allpick = torch.matmul(pick_flat, self.W_val).view(
            shp_allpick
        )  # [self.n_heads, batch_size, n_pick, key_size]
        # pickup -> all delivery
        Q_pick_alldelivery = torch.matmul(pick_flat, self.W3_query).view(
            shp_q_alldelivery
        )  # (self.n_heads, batch_size, n_pick, key_size)
        K_alldelivery = torch.matmul(delivery_flat, self.W_key).view(
            shp_alldelivery
        )  # (self.n_heads, batch_size, n_pick, -1)
        V_alldelivery = torch.matmul(delivery_flat, self.W_val).view(
            shp_alldelivery
        )  # (n_heads, batch_size, n_pick, key/val_size)
        # pickup -> its delivery
        V_additional_delivery = torch.cat(
            [  # [n_heads, batch_size, graph_size, key_size]
                V_delivery,  # [n_heads, batch_size, n_pick, key/val_size]
                torch.zeros(
                    self.num_heads,
                    batch_size,
                    n_pick,
                    self.embed_dim // self.num_heads,
                    dtype=V.dtype,
                    device=V.device,
                ),
            ],
            2,
        )
        # delivery -> its pickup attention
        Q_delivery = torch.matmul(delivery_flat, self.W4_query).view(
            shp_delivery
        )  # (self.n_heads, batch_size, n_pick, key_size)
        K_pick = torch.matmul(pick_flat, self.W_key).view(
            shp_q_pick
        )  # (self.n_heads, batch_size, n_pick, -1)
        V_pick = torch.matmul(pick_flat, self.W_val).view(
            shp_q_pick
        )  # (n_heads, batch_size, n_pick, key/val_size)
        # delivery -> all delivery attention
        Q_delivery_alldelivery = torch.matmul(delivery_flat, self.W5_query).view(
            shp_alldelivery
        )  # (self.n_heads, batch_size, n_pick, -1)
        K_alldelivery2 = torch.matmul(delivery_flat, self.W_key).view(
            shp_alldelivery
        )  # [self.n_heads, batch_size, n_pick, key_size]
        V_alldelivery2 = torch.matmul(delivery_flat, self.W_val).view(
            shp_alldelivery
        )  # [self.n_heads, batch_size, n_pick, key_size]
        # delivery -> all pickup
        Q_delivery_allpickup = torch.matmul(delivery_flat, self.W6_query).view(
            shp_alldelivery
        )  # (self.n_heads, batch_size, n_pick, key_size)
        K_allpickup2 = torch.matmul(pick_flat, self.W_key).view(
            shp_q_alldelivery
        )  # (self.n_heads, batch_size, n_pick, -1)
        V_allpickup2 = torch.matmul(pick_flat, self.W_val).view(
            shp_q_alldelivery
        )  # (n_heads, batch_size, n_pick, key/val_size)
        # delivery -> its pick up
        V_additional_pick = torch.cat(
            [  # [n_heads, batch_size, graph_size, key_size]
                torch.zeros(
                    self.num_heads,
                    batch_size,
                    n_pick,
                    self.embed_dim // self.num_heads,
                    dtype=V.dtype,
                    device=V.device,
                ),
                V_pick,  # [n_heads, batch_size, n_pick, key/val_size]
            ],
            2,
        )
        # Calculate compatibility (n_heads, batch_size, n_query, graph_size)
        compatibility = self.norm_factor * torch.matmul(Q, K.transpose(2, 3))
        ##Pick up
        compatibility_pick_delivery = self.norm_factor * torch.sum(
            Q_pick * K_delivery, -1
        )  # element_wise, [n_heads, batch_size, n_pick]
        # [n_heads, batch_size, n_pick, n_pick]
        compatibility_pick_allpick = self.norm_factor * torch.matmul(
            Q_pick_allpick, K_allpick.transpose(2, 3)
        )  # [n_heads, batch_size, n_pick, n_pick]
        compatibility_pick_alldelivery = self.norm_factor * torch.matmul(
            Q_pick_alldelivery, K_alldelivery.transpose(2, 3)
        )  # [n_heads, batch_size, n_pick, n_pick]

        ##Deliver
        compatibility_delivery_pick = self.norm_factor * torch.sum(
            Q_delivery * K_pick, -1
        )  # element_wise, [n_heads, batch_size, n_pick]
        compatibility_delivery_alldelivery = self.norm_factor * torch.matmul(
            Q_delivery_alldelivery, K_alldelivery2.transpose(2, 3)
        )  # [n_heads, batch_size, n_pick, n_pick]
        compatibility_delivery_allpick = self.norm_factor * torch.matmul(
            Q_delivery_allpickup, K_allpickup2.transpose(2, 3)
        )  # [n_heads, batch_size, n_pick, n_pick]
        ##Pick up->
        # compatibility_additional?pickup????delivery????attention(size 1),1:n_pick+1??attention,depot?delivery??
        compatibility_additional_delivery = torch.cat(
            [  # [n_heads, batch_size, graph_size, 1]
                compatibility_pick_delivery,  # [n_heads, batch_size, n_pick]
                -np.inf
                * torch.ones(
                    self.num_heads,
                    batch_size,
                    n_pick,
                    dtype=compatibility.dtype,
                    device=compatibility.device,
                ),
            ],
            -1,
        ).view(self.num_heads, batch_size, graph_size, 1)
        compatibility_additional_allpick = torch.cat(
            [  # [n_heads, batch_size, graph_size, n_pick]
                compatibility_pick_allpick,  # [n_heads, batch_size, n_pick, n_pick]
                -np.inf
                * torch.ones(
                    self.num_heads,
                    batch_size,
                    n_pick,
                    n_pick,
                    dtype=compatibility.dtype,
                    device=compatibility.device,
                ),
            ],
            2,
        ).view(self.num_heads, batch_size, graph_size, n_pick)
        compatibility_additional_alldelivery = torch.cat(
            [  # [n_heads, batch_size, graph_size, n_pick]
                compatibility_pick_alldelivery,  # [n_heads, batch_size, n_pick, n_pick]
                -np.inf
                * torch.ones(
                    self.num_heads,
                    batch_size,
                    n_pick,
                    n_pick,
                    dtype=compatibility.dtype,
                    device=compatibility.device,
                ),
            ],
            2,
        ).view(self.num_heads, batch_size, graph_size, n_pick)
        # [n_heads, batch_size, n_query, graph_size+1+n_pick+n_pick]
        ##Delivery->
        compatibility_additional_pick = torch.cat(
            [  # [n_heads, batch_size, graph_size, 1]
                -np.inf
                * torch.ones(
                    self.num_heads,
                    batch_size,
                    n_pick,
                    dtype=compatibility.dtype,
                    device=compatibility.device,
                ),
                compatibility_delivery_pick,  # [n_heads, batch_size, n_pick]
            ],
            -1,
        ).view(self.num_heads, batch_size, graph_size, 1)
        compatibility_additional_alldelivery2 = torch.cat(
            [  # [n_heads, batch_size, graph_size, n_pick]
                -np.inf
                * torch.ones(
                    self.num_heads,
                    batch_size,
                    n_pick,
                    n_pick,
                    dtype=compatibility.dtype,
                    device=compatibility.device,
                ),
                compatibility_delivery_alldelivery,  # [n_heads, batch_size, n_pick, n_pick]
            ],
            2,
        ).view(self.num_heads, batch_size, graph_size, n_pick)

        compatibility_additional_allpick2 = torch.cat(
            [  # [n_heads, batch_size, graph_size, n_pick]
                -np.inf
                * torch.ones(
                    self.num_heads,
                    batch_size,
                    n_pick,
                    n_pick,
                    dtype=compatibility.dtype,
                    device=compatibility.device,
                ),
                compatibility_delivery_allpick,  # [n_heads, batch_size, n_pick, n_pick]
            ],
            2,
        ).view(self.num_heads, batch_size, graph_size, n_pick)

        compatibility = torch.cat(
            [
                compatibility,
                compatibility_additional_delivery,
                compatibility_additional_allpick,
                compatibility_additional_alldelivery,
                compatibility_additional_pick,
                compatibility_additional_alldelivery2,
                compatibility_additional_allpick2,
            ],
            dim=-1,
        )
        # Optionally apply mask to prevent attention
        attn = torch.softmax(
            compatibility, dim=-1
        )  # [n_heads, batch_size, n_query, graph_size+1+n_pick*2] (graph_size include depot)
        # heads: [n_heads, batrch_size, n_query, val_size], attn????pick?deliver?attn
        heads = torch.matmul(
            attn[:, :, :, :graph_size], V
        )  # V: (self.n_heads, batch_size, graph_size, val_size)
        # heads??pick -> its delivery
        heads = (
            heads
            + attn[:, :, :, graph_size].view(self.num_heads, batch_size, graph_size, 1)
            * V_additional_delivery
        )  # V_addi:[n_heads, batch_size, graph_size, key_size]
        # heads??pick -> otherpick, V_allpick: # [n_heads, batch_size, n_pick, key_size]
        # heads: [n_heads, batch_size, graph_size, key_size]
        heads = heads + torch.matmul(
            attn[:, :, :, graph_size + 1 : graph_size + 1 + n_pick].view(
                self.num_heads, batch_size, graph_size, n_pick
            ),
            V_allpick,
        )
        # V_alldelivery: # (n_heads, batch_size, n_pick, key/val_size)
        heads = heads + torch.matmul(
            attn[:, :, :, graph_size + 1 + n_pick : graph_size + 1 + 2 * n_pick].view(
                self.num_heads, batch_size, graph_size, n_pick
            ),
            V_alldelivery,
        )
        # delivery
        heads = (
            heads
            + attn[:, :, :, graph_size + 1 + 2 * n_pick].view(
                self.num_heads, batch_size, graph_size, 1
            )
            * V_additional_pick
        )
        heads = heads + torch.matmul(
            attn[
                :,
                :,
                :,
                graph_size + 1 + 2 * n_pick + 1 : graph_size + 1 + 3 * n_pick + 1,
            ].view(self.num_heads, batch_size, graph_size, n_pick),
            V_alldelivery2,
        )
        heads = heads + torch.matmul(
            attn[:, :, :, graph_size + 1 + 3 * n_pick + 1 :].view(
                self.num_heads, batch_size, graph_size, n_pick
            ),
            V_allpickup2,
        )
        out = heads.permute(1, 2, 0, 3).reshape(
            batch_size, n_query, self.embed_dim
        )  # shape: (B, n, head_num, key_dim)
        return out


class Add_And_Normalization_Module(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.alpha = nn.Parameter(torch.Tensor([0]))

    def forward(self, input1, input2):
        return input1 + input2 * self.alpha


class Feed_Forward_Module(nn.Module):
    def __init__(self, emb_dim, ff_dim):
        super().__init__()
        embedding_dim = emb_dim
        ff_hidden_dim = ff_dim

        self.alpha = nn.Parameter(torch.Tensor([0]))
        self.FeedForward = FeedForward(embedding_dim, ff_hidden_dim)

    def forward(self, input):
        # input.shape: (batch, problem, embedding)

        return input + self.FeedForward(input) * self.alpha
