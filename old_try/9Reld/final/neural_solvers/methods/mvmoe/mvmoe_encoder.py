import torch
import torch.nn as nn

from EasyNCO.neural_solvers.backbones import FeedForward, reshape_by_heads, multi_head_attention
from EasyNCO.neural_solvers.methods.mvmoe.moe_layer  import MoE, Add_And_Normalization_Module


class MTMoEEncoder(nn.Module):
    def __init__(self, embedding_dim,ff_hidden_dim,encoder_layer_num,head_num=8,qkv_dim=16,normalization="batch",norm_loc='norm_last',MoE_param=None):
        super().__init__()
        self.MoE_param=MoE_param
        # [Option 1]: Use MoEs in Raw Features
        if MoE_param['num_experts'] > 1 and "Raw" in MoE_param['expert_loc']:
            self.embedding_depot = MoE(input_size=2, output_size=embedding_dim, num_experts=MoE_param['num_experts'],
                                       k=MoE_param['topk'], T=1.0, noisy_gating=True, routing_level=MoE_param['routing_level'],
                                       routing_method=MoE_param['routing_method'], moe_model="Linear")
            self.embedding_node = MoE(input_size=5, output_size=embedding_dim, num_experts=MoE_param['num_experts'],
                                      k=MoE_param['topk'], T=1.0, noisy_gating=True, routing_level=MoE_param['routing_level'],
                                      routing_method=MoE_param['routing_method'], moe_model="Linear")
        else:
            self.embedding_depot = nn.Linear(2, embedding_dim)
            self.embedding_node = nn.Linear(5, embedding_dim)
        self.layers = nn.ModuleList([EncoderLayer(i, embedding_dim,ff_hidden_dim,head_num,qkv_dim,normalization,norm_loc,MoE_param) for i in range(encoder_layer_num)])

    def forward(self, depot_xy, node_xy_demand_tw):
        # depot_xy.shape: (batch, 1, 2)
        # node_xy_demand_tw.shape: (batch, problem, 5)
        # prob_emb: (1, embedding)

        moe_loss = 0
        if isinstance(self.embedding_depot, MoE) or isinstance(self.embedding_node, MoE):#检查节点网络是不是MoE
            embedded_depot, loss_depot = self.embedding_depot(depot_xy)
            embedded_node, loss_node = self.embedding_node(node_xy_demand_tw)
            moe_loss = moe_loss + loss_depot + loss_node
        else:
            embedded_depot = self.embedding_depot(depot_xy)
            # shape: (batch, 1, embedding)
            embedded_node = self.embedding_node(node_xy_demand_tw)
            # shape: (batch, problem, embedding)

        out = torch.cat((embedded_depot, embedded_node), dim=1)
        # shape: (batch, problem+1, embedding)

        for layer in self.layers:
            out, loss = layer(out)
            moe_loss = moe_loss + loss

        return out, moe_loss
        # shape: (batch, problem+1, embedding)
class EncoderLayer(nn.Module):
    def __init__(self, depth=0,embedding_dim=128,ff_hidden_dim=512,head_num=8,qkv_dim=16,normalization='batch',norm_loc="norm_last",MoE_param=None):
        super().__init__()

        self.head_num=head_num
        self.norm_loc=norm_loc
        self.Wq = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wk = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wv = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.multi_head_combine = nn.Linear(head_num * qkv_dim, embedding_dim)

        self.addAndNormalization1 = Add_And_Normalization_Module(embedding_dim,norm=normalization,norm_loc=norm_loc)
        # [Option 2]: Use MoEs in Encoder
        if MoE_param['num_experts'] > 1 and "Enc{}".format(depth) in MoE_param['expert_loc']:
            # TODO: enabling parallelism
            # (1) MOE with tutel, ref to "https://github.com/microsoft/tutel"
            """
            assert self.model_params['routing_level'] == "node", "Tutel only supports node-level routing!"
            self.feedForward = tutel_moe.moe_layer(
                gate_type={'type': 'top', 'k': self.model_params['topk']},
                model_dim=embedding_dim,
                experts={'type': 'ffn', 'count_per_node': self.model_params['num_experts'],
                         'hidden_size_per_expert': self.model_params['ff_hidden_dim'],
                         'activation_fn': lambda x: F.relu(x)},
            )
            """
            # (2) MOE with "https://github.com/davidmrau/mixture-of-experts"
            self.feedForward = MoE(input_size=embedding_dim, output_size=embedding_dim, num_experts=MoE_param['num_experts'],
                                   hidden_size=ff_hidden_dim, k=MoE_param['topk'], T=1.0, noisy_gating=True,
                                   routing_level=MoE_param['routing_level'], routing_method=MoE_param['routing_method'], moe_model="MLP")
        else:
            self.feedForward = FeedForward(embedding_dim,ff_hidden_dim)
        self.addAndNormalization2 = Add_And_Normalization_Module(embedding_dim,norm=normalization,norm_loc=norm_loc)

    def forward(self, input1):
        """
        Two implementations:
            norm_last: the original implementation of AM/POMO: MHA -> Add & Norm -> FFN/MOE -> Add & Norm
            norm_first: the convention in NLP: Norm -> MHA -> Add -> Norm -> FFN/MOE -> Add
        """
        # input.shape: (batch, problem, EMBEDDING_DIM)
        head_num, moe_loss = self.head_num, 0

        q = reshape_by_heads(self.Wq(input1), head_num=head_num)
        k = reshape_by_heads(self.Wk(input1), head_num=head_num)
        v = reshape_by_heads(self.Wv(input1), head_num=head_num)
        # q shape: (batch, HEAD_NUM, problem, KEY_DIM)

        if self.norm_loc == "norm_last":
            out_concat = multi_head_attention(q, k, v)  # (batch, problem, HEAD_NUM*KEY_DIM)
            multi_head_out = self.multi_head_combine(out_concat)  # (batch, problem, EMBEDDING_DIM)
            out1 = self.addAndNormalization1(input1, multi_head_out)
            out2, moe_loss = self.feedForward(out1)
            out3 = self.addAndNormalization2(out1, out2)  # (batch, problem, EMBEDDING_DIM)
        else:#先归一化，在注意力和FF /MoE
            out1 = self.addAndNormalization1(None, input1)
            multi_head_out = self.multi_head_combine(out1)
            input2 = input1 + multi_head_out
            out2 = self.addAndNormalization2(None, input2)
            out2, moe_loss = self.feedForward(out2)
            out3 = input2 + out2

        return out3, moe_loss