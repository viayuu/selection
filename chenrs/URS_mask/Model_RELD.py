from Model_LIB import *
from Model import Encoder



class Model(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params

        self.encoder = Encoder(**model_params)
        self.decoder = RELD_Decoder(**model_params)

        embedding_dim = self.model_params['embedding_dim']
        self.position_embedding  = nn.Linear(3, embedding_dim, bias=False) # coordinate embedding
        self.attribute_embedding = nn.Linear(6, embedding_dim, bias=False)  # attribute embedding
        self.node_type_embedding = nn.Linear(5, embedding_dim, bias=False)  # node embedding

        self.problem_name = None
        self.problem_representation = None

        self.encoded_nodes = None
        # shape: (batch, node, embedding)

    def set_decoder_type(self,decoder_type):
        self.model_params['eval_type'] = decoder_type

    def pre_forward(self, reset_state, problem_name,problem_representation):

        self.problem_name = problem_name
        self.problem_representation = problem_representation
        dist = reset_state.dist

        log_scale = reset_state.log_scale
        self.decoder.log_scale = log_scale # for decoder

        # distance normalization
        ##################################################
        dist_normed = distance_normalization(dist, dist_norm_style="all_max")
        dist_normed_transpose = distance_normalization(dist.transpose(1, 2), dist_norm_style="all_max")

        negative_scale_dist = -1 * log_scale * dist_normed
        negative_scale_dist_transpose = -1 * log_scale * dist_normed_transpose
        # shape: (batch, problem, problem)

        #relation matrix
        ##################################################
        relation = reset_state.relation
        negative_scale_relation = None
        if relation is not None:
            assert problem_name in ['pdp'], "relation matrix is only for pdp problem"  # only for pdp problem
            negative_scale_relation = -1 * relation

        # unified node feature extraction
        ##################################################
        # position embedding
        position_features = unified_node_position_construction(reset_state.problems, problem_name)
        # shape: (batch, problem, 3)  # 3: random identifier, x, y
        position_embedded = self.position_embedding(position_features)
        # shape: (batch, problem, embedding)

        attribute_node_type_features = unified_node_attribute_construction(reset_state.problems, problem_name,demand_max1=self.model_params['demand_max1'])
        attribute_embedded = self.attribute_embedding(attribute_node_type_features[:, :, :6])
        # shape: (batch, problem, embedding), attribute embedded
        node_type_embedded = self.node_type_embedding(attribute_node_type_features[:, :, 6:])
        # shape: (batch, problem, embedding), node type embedded

        init_emb = position_embedded + attribute_embedded + node_type_embedded
        # shape: (batch, problem, embedding)

        self.encoded_nodes = self.encoder(init_emb,
                                          negative_scale_dist,
                                          negative_scale_dist_transpose,
                                          negative_scale_relation,
                                          problem_representation)
        # shape: (batch, problem, embedding_dim)
        self.decoder.set_kv(self.encoded_nodes)


    def forward(self, state,cur_dist):
        batch_size = state.batch_size
        pomo_size = state.pomo_size
        if state.selected_count == 0:  # First Move, depot
            if self.problem_name in ['cvrp','op','pctsp','sdvrp','spctsp','cvrptw','ovrp','vrpl','vrpb','ovrptw','pdp','acvrp',
                                     'ovrpb','vrpbl','vrpltw','ovrpbtw','vrpbltw','ovrpl','vrpbtw','ovrpbl','ovrpltw','ovrpbltw']:
                # For CVRP, we need to select the depot node first
                selected = torch.zeros(size=(batch_size, pomo_size), dtype=torch.long)
                prob = torch.ones(size=(batch_size, pomo_size))
            elif self.problem_name in ['tsp','atsp']:
                selected = torch.arange(pomo_size)[None, :].expand(batch_size, pomo_size)
                prob = torch.ones(size=(batch_size, pomo_size))
            else:
                raise NotImplementedError(f"problem_name: {self.problem_name} is not implemented!")

            encoded_first_node = get_encoding(self.encoded_nodes, selected)
            # shape: (batch, pomo, embedding)
            self.decoder.set_q1(encoded_first_node)

        elif state.selected_count == 1 and pomo_size > 1 and self.problem_name not in ['tsp','atsp']:  # Second Move, POMO
            if self.problem_name in ['vrpb','ovrpb', 'vrpbl', 'ovrpbtw', 'vrpbltw', 'vrpbtw', 'ovrpbl', 'ovrpbltw']:
                #For VRPB, node with negative demand can not be selected as second Move
                selected = state.START_NODE
                prob = torch.ones(size=(batch_size, pomo_size))
            else:
                selected = torch.arange(start=1, end=pomo_size+1)[None, :].expand(batch_size, pomo_size)
                prob = torch.ones(size=(batch_size, pomo_size))
        else:
            encoded_last_node = get_encoding(self.encoded_nodes, state.current_node)
            # shape: (batch, pomo, embedding)

            if self.problem_name in ['tsp','atsp','pdp']:
                constraint = None
            elif self.problem_name in ['cvrp','sdvrp','vrpb','ovrp','cvrptw','ovrptw','vrpl','acvrp',
                                       'ovrpb','vrpbl','vrpltw','ovrpbtw','vrpbltw','ovrpl','vrpbtw','ovrpbl','ovrpltw','ovrpbltw']:
                constraint = state.load.unsqueeze(-1)  # 目前剩余的容量，可补充 and 闭环路径表示
                # shape: (batch, pomo, 1)
            elif self.problem_name in ['op']:
                constraint = (state.load / 4.0).unsqueeze(-1)  # 还能走多远，不可补充
            elif self.problem_name in ['pctsp', 'spctsp']:
                constraint = (1.0 - state.load).unsqueeze(-1)  # 还差多少就能出去
            else:
                raise NotImplementedError(f"problem_name: {self.problem_name} is not implemented!")

            probs = self.decoder(encoded_last_node,
                                 cur_dist,
                                 ninf_mask=state.ninf_mask,
                                 constraint=constraint)
            # shape: (batch, pomo, problem)
            selected, prob = select_next_node(probs, decoding_strategy=self.model_params['eval_type'])

        return selected, prob





class RELD_Decoder(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params['embedding_dim']
        ff_hidden_dim = self.model_params['ff_hidden_dim']

        hyper_input_dim = 13
        hyper_hidden_embd_dim = 256
        self.embd_dim = hyper_input_dim  # 这个维度 应该是固定的
        self.hyper_output_dim = 10 * self.embd_dim
        self.head_num = 8
        self.qkv_dim = embedding_dim//self.head_num


        self.hyper_fc1 = nn.Linear(hyper_input_dim, hyper_hidden_embd_dim, bias=True)  # problem_type_num -> 256
        self.hyper_fc2 = nn.Linear(hyper_hidden_embd_dim, hyper_hidden_embd_dim, bias=True)  # 256->256
        self.hyper_fc3 = nn.Linear(hyper_hidden_embd_dim, self.hyper_output_dim, bias=True)  # 256-> 45

        self.hyper_Wq_first = nn.Linear(self.embd_dim, embedding_dim * embedding_dim, bias=False)
        self.hyper_Wq_last = nn.Linear(self.embd_dim, embedding_dim * embedding_dim, bias=False)
        self.hyper_Wk = nn.Linear(self.embd_dim, embedding_dim * embedding_dim, bias=False)
        self.hyper_Wv = nn.Linear(self.embd_dim, embedding_dim * embedding_dim, bias=False)
        self.hyper_Wcombine = nn.Linear(self.embd_dim, self.head_num * self.qkv_dim * embedding_dim, bias=False)

        self.hyper_Wattr_mapping  = nn.Linear(self.embd_dim, 1 * embedding_dim, bias=False)
        self.hyper_ff_W1 = nn.Linear(self.embd_dim, embedding_dim * ff_hidden_dim, bias=False)
        self.hyper_ff_W2 = nn.Linear(self.embd_dim, ff_hidden_dim * embedding_dim, bias=False)
        self.hyper_ff_bias1 = nn.Linear(self.embd_dim, ff_hidden_dim, bias=False)
        self.hyper_ff_bias2 = nn.Linear(self.embd_dim, embedding_dim, bias=False)


        self.Wq_first_para = None
        self.Wq_last_para = None
        self.Wk_para = None
        self.Wv_para = None
        self.Wcombine_para = None
        self.Wattr_mapping = None
        self.ff_W1 = None
        self.ff_W2 = None
        self.ff_bias1 = None
        self.ff_bias2 = None


        self.k = None  # saved key, for multi-head attention
        self.v = None  # saved value, for multi-head_attention
        self.single_head_key = None  # saved, for single-head attention
        self.q_first = None  # saved q1, for multi-head attention

        self.logit_clipping = self.model_params['logit_clipping']

    def assign(self, problem_representation):  # assign->pre_forward->forward
        embedding_dim = self.model_params['embedding_dim']
        ff_hidden_dim = self.model_params['ff_hidden_dim']

        hyper_embd = self.hyper_fc1(problem_representation)
        hyper_embd = self.hyper_fc2(hyper_embd)
        mid_embd = self.hyper_fc3(hyper_embd)
        # mid_embd.shape: (hyper_output_dim,)

        self.Wq_first_para = self.hyper_Wq_first(mid_embd[:self.embd_dim]).reshape(embedding_dim, embedding_dim)
        self.Wq_last_para = self.hyper_Wq_last(mid_embd[self.embd_dim: 2 * self.embd_dim]).reshape(embedding_dim,embedding_dim)
        self.Wk_para = self.hyper_Wk(mid_embd[2 * self.embd_dim: 3 * self.embd_dim]).reshape(embedding_dim,embedding_dim)
        self.Wv_para = self.hyper_Wv(mid_embd[3 * self.embd_dim: 4 * self.embd_dim]).reshape(embedding_dim,embedding_dim)
        # Note that F.linear execute calculation in the form of xW^T + b, so we need to transpose the weight matrix.
        self.Wcombine_para = self.hyper_Wcombine(mid_embd[4 * self.embd_dim: 5 * self.embd_dim]).reshape(embedding_dim, self.head_num*self.qkv_dim)
        self.Wattr_mapping = self.hyper_Wattr_mapping(mid_embd[5 * self.embd_dim: 6 * self.embd_dim]).reshape(embedding_dim, 1)
        self.ff_W1 = self.hyper_ff_W1(mid_embd[6 * self.embd_dim: 7 * self.embd_dim]).reshape(ff_hidden_dim, embedding_dim)
        self.ff_W2 = self.hyper_ff_W2(mid_embd[7 * self.embd_dim: 8 * self.embd_dim]).reshape(embedding_dim, ff_hidden_dim)
        self.ff_bias1 = self.hyper_ff_bias1(mid_embd[8 * self.embd_dim: 9 * self.embd_dim]).reshape(ff_hidden_dim)
        self.ff_bias2 = self.hyper_ff_bias2(mid_embd[9 * self.embd_dim: 10 * self.embd_dim]).reshape(embedding_dim)



    def set_kv(self, encoded_nodes):
        # encoded_nodes.shape: (batch, problem, embedding)

        self.k = reshape_by_heads(F.linear(encoded_nodes, self.Wk_para),head_num=self.head_num)
        self.v = reshape_by_heads(F.linear(encoded_nodes, self.Wv_para),head_num=self.head_num)
        # shape: (batch, problem, embedding)
        self.single_head_key = encoded_nodes.transpose(1, 2)
        # shape: (batch, embedding, problem)

    def set_q1(self, encoded_q1):
        # encoded_q.shape: (batch, n, embedding)  # n can be 1 or pomo
        self.q_first = reshape_by_heads(F.linear(encoded_q1, self.Wq_first_para),head_num=self.head_num)
        # shape: (batch, head_num, pomo, qkv_dim)

    def ff_hypernet_ver(self,input1):
        input1 = F.linear(input1, self.ff_W1, self.ff_bias1)
        input1 = F.relu(input1)
        input1 = F.linear(input1, self.ff_W2, self.ff_bias2)
        return input1

    def forward(self, encoded_last_node, cur_dist, ninf_mask, constraint=None):
        q_last = reshape_by_heads(F.linear(encoded_last_node, self.Wq_last_para),head_num=self.head_num)
        #加log_dist和不加log_dist
        cur_dist = distance_normalization(cur_dist, dist_norm_style="sep_max")
        negative_dist = -torch.log(cur_dist)


        q = self.q_first + q_last

        out_concat = multi_head_attention(q, self.k, self.v, rank3_ninf_mask=ninf_mask)

        mh_atten_out = F.linear(out_concat, self.Wcombine_para)

        if constraint is None:
            mh_atten_out = mh_atten_out + encoded_last_node
        else:
            mh_atten_out = mh_atten_out + encoded_last_node + F.linear(constraint.clone(), self.Wattr_mapping)
        mh_atten_out = self.ff_hypernet_ver(mh_atten_out) + mh_atten_out

        #  Single-Head Attention, for probability calculation
        #######################################################
        score = torch.matmul(mh_atten_out, self.single_head_key)
        # shape: (batch, pomo, problem)
        score_scaled = score / torch.sqrt(torch.tensor(self.model_params['embedding_dim'], dtype=torch.float))
        # shape: (batch, pomo, problem)
        if self.model_params['use_log_dist']:
            score_scaled = score_scaled + negative_dist

        score_clipped = self.logit_clipping * torch.tanh(score_scaled)

        score_masked = score_clipped + ninf_mask

        probs = F.softmax(score_masked, dim=2)
        # shape: (batch, pomo, problem)

        return probs