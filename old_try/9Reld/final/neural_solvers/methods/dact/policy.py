from torch import nn
import torch
import torch.nn.functional as F
from torch.distributions import Categorical

from EasyNCO.neural_solvers.methods.dact.encoder import DACTCritic_Encoder,EmbeddingNet,DACT_Encoder
from EasyNCO.neural_solvers.methods.dact.decoder import DACTCritic_Decoder,DACT_Decoder
from tensordict import TensorDict


class DACTPolicy(nn.Module):
    def __init__(self,
                 env_name: str = "tsp",
                 embed_dim: int = 64,
                 feedforward_hidden: int = 64,
                 qkv_dim: int = 16,
                 num_heads: int = 4,
                 num_encoder_layers: int = 3,
                 normalization: str = "layer",
                 logit_clipping: float = 6.0,
                 **kwargs
                 ):
        super(DACTPolicy, self).__init__()

        self.env_name = env_name
        self.embed_dim = embed_dim
        self.feedforward_hidden = feedforward_hidden
        self.num_heads = num_heads
        self.qkv_dim = qkv_dim
        self.num_encoder_layers = num_encoder_layers
        self.normalization = normalization
        self.logit_clipping = logit_clipping

        if self.env_name == 'tsp':
            self.node_dim = 2
        elif self.env_name == 'cvrp':
            self.node_dim = 7


        self.encoder = DACT_Encoder(
            env_name=self.env_name,
            embed_dim=self.embed_dim,
            qkv_dim=self.qkv_dim,
            num_encoder_layers=self.num_encoder_layers,
            num_heads=self.num_heads,
            normalization=self.normalization,
            feedforward_hidden=self.feedforward_hidden,
            node_dim=self.node_dim,
        )

        self.decoder = DACT_Decoder(
            num_heads=self.num_heads,
            input_dim=self.embed_dim,
            embed_dim=self.embed_dim
        )


    def initial_problem_size(self,problem_size):
        self.problem_size = problem_size
        if self.problem_size < 50:
            dummy_rate = 0.5
        elif self.problem_size < 100:
            dummy_rate = 0.4
        else:
            dummy_rate = 0.2
        self.problem_size_extend = int(problem_size * (1 + dummy_rate))
        self.dummy_size = self.problem_size_extend - problem_size
        self.encoder.embedder.pattern = self.encoder.embedder.Cyclic_Positional_Encoding(self.problem_size_extend, self.embed_dim)

    #  (x-1,x,x+1,x.demand)组成一个超点, 7-dim
    def prepare_cvrp(self,solution,x_in,bs):
        argsort = solution.argsort()
        loc = x_in[:, :, :2]
        post = loc.gather(1, solution.view(bs, -1, 1).expand_as(loc))
        pre = loc.gather(1, argsort.view(bs, -1, 1).expand_as(loc))
        post = torch.norm(post - loc, 2, -1, True)
        pre = torch.norm(pre - loc, 2, -1, True)

        x_in = torch.cat((loc, pre, post, x_in[:, :, -1:]), -1)
        del post, pre, argsort
        return x_in


    def set_decoder_strategy(self, strategy: str = "sampling"):
        self.decoder_strategy = strategy

    def get_2opt_mask_DACT(self, rec, batch):
        bs, gs = rec.size()
        selfmask = torch.eye(gs, device=rec.device).view(1, gs, gs)

        real_mask, contex, to_actor = self.get_real_mask(rec, batch)
        masks = (real_mask) + selfmask.expand(bs, gs, gs).bool()

        return masks, contex, to_actor

    def get_real_mask(self, rec, batch):
        CAPACITIES = {
            10: 20.,
            20: 30.,
            50: 40.,
            100: 50.,
        }

        # get mixed contex: 1000 * route_plan + 1 * visited_time + 0.5 * cu_demand
        # (e.g., 1000 + 34 + 0.05 means the node is the 34th node in route 1 and the cum demand before the node is 0.1)
        contex, patial_sum = self.preprocessing(rec, batch)
        # only allow in-route 2-opt
        route_plan = (contex // 1000).long() % self.dummy_size
        mask_in = route_plan.view(-1, self.problem_size_extend, 1) != route_plan.view(-1, 1, self.problem_size_extend)
        mask_in[:, :self.dummy_size, :] = True
        mask_in[:, :, :self.dummy_size] = True
        # special case
        mask_special1 = mask_in.clone() & False
        mask_special1[:, self.dummy_size:, :] = True
        mask_special1 |= ((route_plan.view(-1, self.problem_size_extend, 1) - 1) % self.dummy_size) != route_plan.view(-1, 1,
                                                                                                        self.problem_size_extend)
        mask_special2 = mask_in.clone() & False
        mask_special2[:, :, self.dummy_size:] = True
        mask_special2 |= ((route_plan.view(-1, self.problem_size_extend, 1)) % self.dummy_size) != route_plan.view(-1, 1,
                                                                                                    self.problem_size_extend)

        # further allow btw-route 2-opt
        demand = batch[:,:,-1] if isinstance(batch, dict) else batch[:, :, -1]
        cum_demand = ((contex % 1) * 2)
        total = patial_sum.gather(-1, route_plan)
        cor = (demand != 0).float()
        pi = cum_demand.view(-1, self.problem_size_extend, 1)
        pj = (cor * cum_demand).view(-1, 1, self.problem_size_extend)
        qi = (cor * (total - cum_demand)).view(-1, self.problem_size_extend, 1)
        qj = (total - cor * cum_demand).view(-1, 1, self.problem_size_extend)
        corj = demand.view(-1, 1, self.problem_size_extend)
        mask_btw = ((pi + pj + corj) > (1. + 0.1 / CAPACITIES[self.problem_size])) | (
                    (qi + qj - corj) > (1. + 0.1 / CAPACITIES[self.problem_size]))
        mask = ~(~mask_in + ~mask_btw + ~mask_special1 + ~mask_special2)
        mask[:, :self.dummy_size, :self.dummy_size] = False

        return mask, contex, torch.cat((cum_demand.view(-1, self.problem_size_extend, 1),
                                        demand.view(-1, self.problem_size_extend, 1),
                                        (total - cor.view(-1, self.problem_size_extend) * cum_demand).view(-1, self.problem_size_extend, 1),
                                        ), -1
                                       )

    def preprocessing(self, solutions, batch):
        #dact
        batch_size, seq_length = solutions.size()
        demand = batch[:, :, -1]
        arange = torch.arange(batch_size)

        pre = torch.zeros(batch_size, device=solutions.device).long() #从node_0开始
        route = torch.zeros(batch_size, device=solutions.device)
        partial_sum = torch.zeros((batch_size, self.dummy_size), device=solutions.device)
        route_plan1000_visited_time1_dot_demand = torch.zeros((batch_size, seq_length), device=solutions.device)
        assert seq_length < 1000
        for i in range(seq_length):
            next_ = solutions[arange, pre]
            index = next_ < self.dummy_size
            route = torch.where(index, route + 1, route)
            cu_demand = torch.where(index,
                                    partial_sum[arange, (route.long() - 1) % self.dummy_size],
                                    partial_sum[arange, route.long() % self.dummy_size])
            route_plan1000_visited_time1_dot_demand[arange, next_] = i + 1 + route * 1000 + cu_demand * 0.5
            partial_sum[arange, route.long() % self.dummy_size] += demand[arange, next_]
            pre = next_


        return route_plan1000_visited_time1_dot_demand, partial_sum

    def forward(self,td: TensorDict,**kwargs):
        #for train
        fixed_action = kwargs.get('fixed_action')
        do_sample = kwargs.get('do_sample',False)
        require_entropy = kwargs.get('require_entropy',False)
        to_critic = kwargs.get('to_critic',False)
        only_critic =kwargs.get('only_critic',False)

        bs, gs, in_d = td['locs'].size()
        #  x-1,x,x+1组成一个超点
        if self.env_name == 'cvrp':
            x_in = self.prepare_cvrp(td['solution'],td['locs'],bs)

            masks, contex, to_actor = self.get_2opt_mask_DACT(td['solution'], x_in)
            # concate the 7-dim features x_in
            x_in = torch.cat((x_in[:, :, :4], to_actor), -1)
            del to_actor
            contex = contex % 1000 // 1
            h_em, g_em = self.encoder(td,input=x_in,visited_time=contex)

        elif self.env_name == 'tsp':
            selfmask = torch.eye(gs).view(1, gs, gs)
            masks = selfmask.expand(bs, gs, gs).bool().cpu()
            h_em, g_em = self.encoder(td)

        if only_critic:
            return (h_em, g_em)

        compatibility = torch.tanh(self.decoder(h_em, g_em)) *  self.logit_clipping

        compatibility[masks] = -1e20
        del masks

        is_all_zero = torch.all(td['exchange'] == 0).item()
        if not is_all_zero:
            compatibility[torch.arange(bs), td['exchange'][:,0], td['exchange'][:,1]] = -1e20
            compatibility[torch.arange(bs), td['exchange'][:,1], td['exchange'][:,0]] = -1e20


        im = compatibility.view(bs, -1)

        # softmax
        log_likelihood = F.log_softmax(im,dim = -1)
        M_table = F.softmax(im,dim = -1)

        # fixed action for PPO training if needed
        if fixed_action is not None:
            row_selected = fixed_action[:, 0]
            col_selected = fixed_action[:, 1]
            pair_index = row_selected * gs + col_selected
            pair_index = pair_index.view(-1, 1)
            pair = fixed_action

        else:
            # sample one action
            if do_sample:
                pair_index = M_table.multinomial(1)
            else:
                pair_index = M_table.max(-1)[1].view(-1, 1)

            # from action (selected node pair)
            row_selected = pair_index // gs
            col_selected = pair_index % gs
            pair = torch.cat((row_selected, col_selected), -1)  # pair: no_head bs, 2

        selected_log_likelihood = log_likelihood.gather(1, pair_index)

        if require_entropy:
            dist = Categorical(M_table, validate_args=False)  # for logging only
            entropy = dist.entropy()  # for logging only
            td['entropy'] = entropy

        if to_critic:
            td['h_em'] = h_em
            td['g_em'] = g_em
        td.update(
            {
                'exchange': pair,
                'log_likelihood':selected_log_likelihood.squeeze(),
            }
        )

        return td