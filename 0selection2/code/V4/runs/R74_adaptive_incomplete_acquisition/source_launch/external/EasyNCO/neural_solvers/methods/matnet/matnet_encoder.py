import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from EasyNCO.neural_solvers.backbones import (
    Normalization,
    reshape_by_heads,
    FeedForward,
)




class MatNetEncoder(nn.Module):

    def __init__(
        self,
        encoder_layer_num: int = 5,
        embedding_dim: int = 256,
        head_num: int = 16,
        qkv_dim: int = 16,
        ms_hidden_dim: int = 16,
        ms_layer1_init: float = (1 / 2) ** (1 / 2),
        ms_layer2_init: float = (1 / 16) ** (1 / 2),
        ff_hidden_dim: int = 512,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [EncoderLayer(
                embedding_dim=embedding_dim,
                head_num=head_num,
                qkv_dim=qkv_dim,
                ms_hidden_dim=ms_hidden_dim,
                ms_layer1_init=ms_layer1_init,
                ms_layer2_init=ms_layer2_init,
                ff_hidden_dim=ff_hidden_dim,
                ) for _ in range(encoder_layer_num)]
        )

    def forward(self, row_emb, col_emb, cost_mat):
        # col_emb.shape: (batch, col_cnt, embedding)
        # row_emb.shape: (batch, row_cnt, embedding)
        # cost_mat.shape: (batch, row_cnt, col_cnt)

        for layer in self.layers:
            row_emb, col_emb = layer(row_emb, col_emb, cost_mat)

        return row_emb, col_emb


class EncoderLayer(nn.Module):

    def __init__(
        self,
        embedding_dim: int = 256,
        head_num: int = 16,
        qkv_dim: int = 16,
        ms_hidden_dim: int = 16,
        ms_layer1_init: float = 1 / math.sqrt(2),
        ms_layer2_init: float = 0.25,
        ff_hidden_dim: int = 512,
    ):
        super().__init__()
        self.row_encoding_block = EncodingBlock(
            embedding_dim=embedding_dim,
            head_num=head_num,
            qkv_dim=qkv_dim,
            ms_hidden_dim=ms_hidden_dim,
            ms_layer1_init=ms_layer1_init,
            ms_layer2_init=ms_layer2_init,
            ff_hidden_dim=ff_hidden_dim,
        )
        self.col_encoding_block = EncodingBlock(
            embedding_dim=embedding_dim,
            head_num=head_num,
            qkv_dim=qkv_dim,
            ms_hidden_dim=ms_hidden_dim,
            ms_layer1_init=ms_layer1_init,
            ms_layer2_init=ms_layer2_init,
            ff_hidden_dim=ff_hidden_dim,
        )

    def forward(self, row_emb, col_emb, cost_mat):
        # row_emb.shape: (batch, row_cnt, embedding)
        # col_emb.shape: (batch, col_cnt, embedding)
        # cost_mat.shape: (batch, row_cnt, col_cnt)
        row_emb_out = self.row_encoding_block(row_emb, col_emb, cost_mat)
        col_emb_out = self.col_encoding_block(col_emb, row_emb, cost_mat.transpose(1, 2))

        return row_emb_out, col_emb_out


class EncodingBlock(nn.Module):
    def __init__(
        self,
        embedding_dim: int = 256,
        head_num: int = 16,
        qkv_dim: int = 16,
        ms_hidden_dim: int = 16,
        ms_layer1_init: float = 1 / math.sqrt(2),
        ms_layer2_init: float = 0.25,
        ff_hidden_dim: int = 512,
    ):
        super().__init__()

        self.Wq = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wk = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wv = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.mixed_score_MHA = MixedScore_MultiHeadAttention(
            head_num=head_num,
            ms_hidden_dim=ms_hidden_dim,
            mix1_init=ms_layer1_init,
            mix2_init=ms_layer2_init,
        )
        self.multi_head_combine = nn.Linear(head_num * qkv_dim, embedding_dim)

        self.normalization_1 = Normalization(embed_dim=embedding_dim, normalization="instance")
        self.feed_forward = FeedForward(
            embedding_dim=embedding_dim, ff_hidden_dim=ff_hidden_dim
        )
        self.normalization_2 = Normalization(embed_dim=embedding_dim, normalization="instance")
        self.head_num = head_num

    def forward(self, row_emb, col_emb, cost_mat):
        # NOTE: row and col can be exchanged, if cost_mat.transpose(1,2) is used
        # input1.shape: (batch, row_cnt, embedding)
        # input2.shape: (batch, col_cnt, embedding)
        # cost_mat.shape: (batch, row_cnt, col_cnt)

        q = reshape_by_heads(self.Wq(row_emb), head_num=self.head_num)
        # q shape: (batch, head_num, row_cnt, qkv_dim)
        k = reshape_by_heads(self.Wk(col_emb), head_num=self.head_num)
        v = reshape_by_heads(self.Wv(col_emb), head_num=self.head_num)
        # kv shape: (batch, head_num, col_cnt, qkv_dim)

        out_concat = self.mixed_score_MHA(q, k, v, cost_mat)
        # shape: (batch, row_cnt, head_num*qkv_dim)

        multi_head_out = self.multi_head_combine(out_concat)
        # shape: (batch, row_cnt, embedding)

        out1 = self.normalization_1(multi_head_out + row_emb)
        out2 = self.feed_forward(out1)
        out3 = self.normalization_2(out1 + out2)

        return out3
        # shape: (batch, row_cnt, embedding)





class MixedScore_MultiHeadAttention(nn.Module):
    def __init__(
        self,
        head_num: int = 16,
        ms_hidden_dim: int = 16,
        mix1_init: float = 1 / math.sqrt(2),
        mix2_init: float = 0.25,
        qkv_dim: int = 16,
    ):
        super().__init__()

        # self.sqrt_qkv_dim=sqrt_qkv_dim
        # self.ms_hidden_dim=ms_hidden_dim
        # self.ms_layer1_init=ms_layer1_init
        # self.ms_layer2_init=ms_layer2_init
        # mix1_init = ms_layer1_init
        # mix2_init = ms_layer2_init

        mix1_weight = torch.torch.distributions.Uniform(
            low=-mix1_init, high=mix1_init
        ).sample((head_num, 2, ms_hidden_dim))
        mix1_bias = torch.torch.distributions.Uniform(
            low=-mix1_init, high=mix1_init
        ).sample((head_num, ms_hidden_dim))
        self.mix1_weight = nn.Parameter(mix1_weight)
        # shape: (head, 2, ms_hidden)
        self.mix1_bias = nn.Parameter(mix1_bias)
        # shape: (head, ms_hidden)

        mix2_weight = torch.torch.distributions.Uniform(
            low=-mix2_init, high=mix2_init
        ).sample((head_num, ms_hidden_dim, 1))
        mix2_bias = torch.torch.distributions.Uniform(
            low=-mix2_init, high=mix2_init
        ).sample((head_num, 1))
        self.mix2_weight = nn.Parameter(mix2_weight)
        # shape: (head, ms_hidden, 1)
        self.mix2_bias = nn.Parameter(mix2_bias)
        # shape: (head, 1)
        self.head_num = head_num
        self.qkv_dim = qkv_dim

    def forward(self, q, k, v, cost_mat):
        # q shape: (batch, head_num, row_cnt, qkv_dim)
        # k,v shape: (batch, head_num, col_cnt, qkv_dim)
        # cost_mat.shape: (batch, row_cnt, col_cnt)
        sqrt_qkv_dim = math.sqrt(self.qkv_dim)

        batch_size = q.size(0)
        row_cnt = q.size(2)
        col_cnt = k.size(2)

        dot_product = torch.matmul(q, k.transpose(2, 3))
        # shape: (batch, head_num, row_cnt, col_cnt)

        dot_product_score = dot_product / sqrt_qkv_dim
        # shape: (batch, head_num, row_cnt, col_cnt)

        cost_mat_score = cost_mat[:, None, :, :].expand(
            batch_size, self.head_num, row_cnt, col_cnt
        )
        # shape: (batch, head_num, row_cnt, col_cnt)

        two_scores = torch.stack((dot_product_score, cost_mat_score), dim=4)
        # shape: (batch, head_num, row_cnt, col_cnt, 2)

        two_scores_transposed = two_scores.transpose(1, 2)
        # shape: (batch, row_cnt, head_num, col_cnt, 2)

        ms1 = torch.matmul(two_scores_transposed, self.mix1_weight)
        # shape: (batch, row_cnt, head_num, col_cnt, ms_hidden_dim)

        ms1 = ms1 + self.mix1_bias[None, None, :, None, :]
        # shape: (batch, row_cnt, head_num, col_cnt, ms_hidden_dim)

        ms1_activated = F.relu(ms1)

        ms2 = torch.matmul(ms1_activated, self.mix2_weight)
        # shape: (batch, row_cnt, head_num, col_cnt, 1)

        ms2 = ms2 + self.mix2_bias[None, None, :, None, :]
        # shape: (batch, row_cnt, head_num, col_cnt, 1)

        mixed_scores = ms2.transpose(1, 2)
        # shape: (batch, head_num, row_cnt, col_cnt, 1)

        mixed_scores = mixed_scores.squeeze(4)
        # shape: (batch, head_num, row_cnt, col_cnt)

        weights = nn.Softmax(dim=3)(mixed_scores)
        # shape: (batch, head_num, row_cnt, col_cnt)

        out = torch.matmul(weights, v)
        # shape: (batch, head_num, row_cnt, qkv_dim)

        out_transposed = out.transpose(1, 2)
        # shape: (batch, row_cnt, head_num, qkv_dim)

        out_concat = out_transposed.reshape(
            batch_size, row_cnt, self.head_num * self.qkv_dim
        )
        # shape: (batch, row_cnt, head_num*qkv_dim)

        return out_concat
