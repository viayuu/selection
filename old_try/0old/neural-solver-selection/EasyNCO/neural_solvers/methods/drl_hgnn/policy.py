import torch
import torch.nn.functional as F
import torch.nn as nn
import copy
from torch.distributions.categorical import Categorical
from EasyNCO.neural_solvers.methods.drl_hgnn.models import GATedge,MLPActor,MLPCritic,MLPsim,MLPs
from tensordict import TensorDict

class DRL_HGNNPolicy(nn.Module):
    def __init__(self,
                 in_size_ma=3,
                 out_size_ma=8,
                 in_size_ope=6,
                 out_size_ope=8,
                 hidden_size_ope=128,
                 n_latent_actor=64,
                 n_latent_critic=64,
                 n_hidden_actor=3,
                 n_hidden_critic=3,
                 action_dim=1,
                 num_heads=[1,1],
                 dropout=0.0):
        super(DRL_HGNNPolicy, self).__init__()
        actor_in_dim = out_size_ma * 2 + out_size_ope * 2
        critic_in_dim = out_size_ma + out_size_ope

        self.in_size_ma = in_size_ma  # Dimension of the raw feature vectors of machine nodes
        self.out_size_ma = out_size_ma  # Dimension of the embedding of machine nodes
        self.in_size_ope = in_size_ope  # Dimension of the raw feature vectors of operation nodes
        self.out_size_ope = out_size_ope  # Dimension of the embedding of operation nodes
        self.hidden_size_ope = hidden_size_ope  # Hidden dimensions of the MLPs
        self.actor_dim = actor_in_dim  # Input dimension of actor
        self.critic_dim = critic_in_dim  # Input dimension of critic
        self.n_latent_actor = n_latent_actor  # Hidden dimensions of the actor
        self.n_latent_critic = n_latent_critic  # Hidden dimensions of the critic
        self.n_hidden_actor = n_hidden_actor  # Number of layers in actor
        self.n_hidden_critic = n_hidden_critic  # Number of layers in critic
        self.action_dim = action_dim  # Output dimension of actor

        # len() means of the number of HGNN iterations
        # and the element means the number of heads of each HGNN (=1 in final experiment)
        self.num_heads = num_heads
        self.dropout = dropout


        # Machine node embedding
        self.get_machines = nn.ModuleList()
        self.get_machines.append(GATedge((self.in_size_ope, self.in_size_ma), self.out_size_ma, self.num_heads[0],
                                    self.dropout, self.dropout, activation=F.elu))
        for i in range(1,len(self.num_heads)):
            self.get_machines.append(GATedge((self.out_size_ope, self.out_size_ma), self.out_size_ma, self.num_heads[i],
                                    self.dropout, self.dropout, activation=F.elu))

        # Operation node embedding
        self.get_operations = nn.ModuleList()
        self.get_operations.append(MLPs([self.out_size_ma, self.in_size_ope, self.in_size_ope, self.in_size_ope],
                                        self.hidden_size_ope, self.out_size_ope, self.num_heads[0], self.dropout))
        for i in range(len(self.num_heads)-1):
            self.get_operations.append(MLPs([self.out_size_ma, self.out_size_ope, self.out_size_ope, self.out_size_ope],
                                            self.hidden_size_ope, self.out_size_ope, self.num_heads[i], self.dropout))

        self.actor = MLPActor(self.n_hidden_actor, self.actor_dim, self.n_latent_actor, self.action_dim)
        self.critic = MLPCritic(self.n_hidden_critic, self.critic_dim, self.n_latent_critic, 1)

        self.decoder_strategy = None



    def set_decoder_strategy(self, strategy: str = "sampling"):
        self.decoder_strategy = strategy


    def forward(self):
        '''
        Replaced by separate act and evaluate functions
        '''
        raise NotImplementedError

    def feature_normalize(self, data):
        return (data - torch.mean(data)) / ((data.std() + 1e-5))

    '''
        raw_opes: shape: [len(batch_idxes), max(num_opes), in_size_ope]
        raw_mas: shape: [len(batch_idxes), num_mas, in_size_ma]
        proc_time: shape: [len(batch_idxes), max(num_opes), num_mas]
    '''
    def get_normalized(self, raw_opes, raw_mas, proc_time, batch_idxes, nums_opes, flag_sample=False, flag_train=False):
        '''
        :param raw_opes: Raw feature vectors of operation nodes
        :param raw_mas: Raw feature vectors of machines nodes
        :param proc_time: Processing time
        :param batch_idxes: Uncompleted instances
        :param nums_opes: The number of operations for each instance
        :param flag_sample: Flag for DRL-S
        :param flag_train: Flag for training
        :return: Normalized feats, including operations, machines and edges
        '''
        batch_size = batch_idxes.size(0)  # number of uncompleted instances

        # There may be different operations for each instance, which cannot be normalized directly by the matrix
        if not flag_sample and not flag_train:
            mean_opes = []
            std_opes = []
            for i in range(batch_size):
                mean_opes.append(torch.mean(raw_opes[i, :nums_opes[i], :], dim=-2, keepdim=True))
                std_opes.append(torch.std(raw_opes[i, :nums_opes[i], :], dim=-2, keepdim=True))
                proc_idxes = torch.nonzero(proc_time[i])
                proc_values = proc_time[i, proc_idxes[:, 0], proc_idxes[:, 1]]
                proc_norm = self.feature_normalize(proc_values)
                proc_time[i, proc_idxes[:, 0], proc_idxes[:, 1]] = proc_norm
            mean_opes = torch.stack(mean_opes, dim=0)
            std_opes = torch.stack(std_opes, dim=0)
            mean_mas = torch.mean(raw_mas, dim=-2, keepdim=True)
            std_mas = torch.std(raw_mas, dim=-2, keepdim=True)
            proc_time_norm = proc_time
        # DRL-S and scheduling during training have a consistent number of operations
        else:
            mean_opes = torch.mean(raw_opes, dim=-2, keepdim=True)  # shape: [len(batch_idxes), 1, in_size_ope]
            mean_mas = torch.mean(raw_mas, dim=-2, keepdim=True)  # shape: [len(batch_idxes), 1, in_size_ma]
            std_opes = torch.std(raw_opes, dim=-2, keepdim=True)  # shape: [len(batch_idxes), 1, in_size_ope]
            std_mas = torch.std(raw_mas, dim=-2, keepdim=True)  # shape: [len(batch_idxes), 1, in_size_ma]
            proc_time_norm = self.feature_normalize(proc_time)  # shape: [len(batch_idxes), num_opes, num_mas]
        return ((raw_opes - mean_opes) / (std_opes + 1e-5), (raw_mas - mean_mas) / (std_mas + 1e-5),
                proc_time_norm)

    def get_action_prob(self, state, memories, flag_sample=False, flag_train=False):
        '''
        Get the probability of selecting each action in decision-making
        '''
        # Uncompleted instances
        batch_idxes = state["batch_idxes"]
        # Raw feats
        raw_opes = state["feat_opes_batch"].transpose(1, 2)[batch_idxes]
        raw_mas = state["feat_mas_batch"].transpose(1, 2)[batch_idxes]
        proc_time = state["proc_times_batch"][batch_idxes]
        # Normalize
        nums_opes = state["nums_opes_batch"][batch_idxes]
        features = self.get_normalized(raw_opes, raw_mas, proc_time, batch_idxes, nums_opes, flag_sample, flag_train)
        norm_opes = copy.deepcopy(features[0])
        norm_mas = copy.deepcopy(features[1])
        norm_proc = copy.deepcopy(features[2])

        # L iterations of the HGNN
        for i in range(len(self.num_heads)):
            h_mas = self.get_machines[i](state["ope_ma_adj_batch"], state["batch_idxes"], features)
            features = (features[0], h_mas, features[2])

            h_opes = self.get_operations[i](
                state["ope_ma_adj_batch"], state["ope_pre_adj_batch"], state["ope_sub_adj_batch"],
                state["batch_idxes"], features)
            features = (h_opes, features[1], features[2])

        h_mas_pooled = h_mas.mean(dim=-2)

        if not flag_sample and not flag_train:
            h_opes_pooled = []
            for i in range(len(batch_idxes)):
                h_opes_pooled.append(torch.mean(h_opes[i, :nums_opes[i], :], dim=-2))
            h_opes_pooled = torch.stack(h_opes_pooled)
        else:
            h_opes_pooled = h_opes.mean(dim=-2)

        # Detect eligible O-M pairs
        ope_step_batch = torch.where(
            state["ope_step_batch"] > state["end_ope_biases_batch"],
            state["end_ope_biases_batch"],
            state["ope_step_batch"]
        )
        jobs_gather = ope_step_batch[..., :, None].expand(-1, -1, h_opes.size(-1))[batch_idxes]
        h_jobs = h_opes.gather(1, jobs_gather)

        eligible_proc = state["ope_ma_adj_batch"][batch_idxes].gather(
            1,
            ope_step_batch[..., :, None].expand(-1, -1, state["ope_ma_adj_batch"].size(-1))[batch_idxes]
        )

        h_jobs_padding = h_jobs.unsqueeze(-2).expand(-1, -1, state["proc_times_batch"].size(-1), -1)
        h_mas_padding = h_mas.unsqueeze(-3).expand_as(h_jobs_padding)
        h_mas_pooled_padding = h_mas_pooled[:, None, None, :].expand_as(h_jobs_padding)
        h_opes_pooled_padding = h_opes_pooled[:, None, None, :].expand_as(h_jobs_padding)

        ma_eligible = ~state["mask_ma_procing_batch"][batch_idxes].unsqueeze(1).expand_as(h_jobs_padding[..., 0])
        job_eligible = ~(
                                state["mask_job_procing_batch"][batch_idxes] +
                                state["mask_job_finish_batch"][batch_idxes]
                        )[:, :, None].expand_as(h_jobs_padding[..., 0])
        eligible = job_eligible & ma_eligible & (eligible_proc == 1)

        if (~eligible).all():
            print("No eligible O-M pair!")
            return

        h_actions = torch.cat(
            (h_jobs_padding, h_mas_padding, h_opes_pooled_padding, h_mas_pooled_padding),
            dim=-1
        ).transpose(1, 2)
        h_pooled = torch.cat((h_opes_pooled, h_mas_pooled), dim=-1)
        mask = eligible.transpose(1, 2).flatten(1)

        scores = self.actor(h_actions).flatten(1)
        scores[~mask] = float('-inf')
        action_probs = F.softmax(scores, dim=1)

        if flag_train:
            memories.ope_ma_adj.append(copy.deepcopy(state["ope_ma_adj_batch"]))
            memories.ope_pre_adj.append(copy.deepcopy(state["ope_pre_adj_batch"]))
            memories.ope_sub_adj.append(copy.deepcopy(state["ope_sub_adj_batch"]))
            memories.batch_idxes.append(copy.deepcopy(state["batch_idxes"]))
            memories.raw_opes.append(copy.deepcopy(norm_opes))
            memories.raw_mas.append(copy.deepcopy(norm_mas))
            memories.proc_time.append(copy.deepcopy(norm_proc))
            memories.nums_opes.append(copy.deepcopy(nums_opes))
            memories.jobs_gather.append(copy.deepcopy(jobs_gather))
            memories.eligible.append(copy.deepcopy(eligible))

        return action_probs, ope_step_batch, h_pooled

    def act(self, state, memories):
        # Get probability of actions and the id of the current operation (be waiting to be processed) of each job
        if self.decoder_strategy == 'sampling':
            flag_sample = True
        else:
            flag_sample = False
        flag_train = self.training
        action_probs, ope_step_batch, _ = self.get_action_prob(state, memories, flag_sample, flag_train=flag_train)

        # DRL-S, sampling actions following \pi
        if flag_sample:
            dist = Categorical(action_probs)
            action_indexes = dist.sample()
        # DRL-G, greedily picking actions with the maximum probability
        else:
            action_indexes = action_probs.argmax(dim=1)

        # Calculate the machine, job and operation index based on the action index
        mas = (action_indexes / state["mask_job_finish_batch"].size(1)).long()
        jobs = (action_indexes % state["mask_job_finish_batch"].size(1)).long()
        opes = ope_step_batch[state["batch_idxes"], jobs]

        # Store data in memory during training
        if flag_train == True:
            # memories.states.append(copy.deepcopy(state))
            memories.logprobs.append(dist.log_prob(action_indexes))
            memories.action_indexes.append(action_indexes)


        action = torch.stack((opes, mas, jobs), dim=1)
        td = TensorDict({
            'action': action,
            'prob': action_probs
        }, batch_size=action_probs.shape[0])

        return td

    def evaluate(self, ope_ma_adj, ope_pre_adj, ope_sub_adj, raw_opes, raw_mas, proc_time,
                 jobs_gather, eligible, action_envs, flag_sample=False):
        batch_idxes = torch.arange(0, ope_ma_adj.size(-3)).long()
        features = (raw_opes, raw_mas, proc_time)

        # L iterations of the HGNN
        for i in range(len(self.num_heads)):
            h_mas = self.get_machines[i](ope_ma_adj, batch_idxes, features)
            features = (features[0], h_mas, features[2])
            h_opes = self.get_operations[i](ope_ma_adj, ope_pre_adj, ope_sub_adj, batch_idxes, features)
            features = (h_opes, features[1], features[2])

        # Stacking and pooling
        h_mas_pooled = h_mas.mean(dim=-2)
        h_opes_pooled = h_opes.mean(dim=-2)

        # Detect eligible O-M pairs (eligible actions) and generate tensors for critic calculation
        h_jobs = h_opes.gather(1, jobs_gather)
        h_jobs_padding = h_jobs.unsqueeze(-2).expand(-1, -1, proc_time.size(-1), -1)
        h_mas_padding = h_mas.unsqueeze(-3).expand_as(h_jobs_padding)
        h_mas_pooled_padding = h_mas_pooled[:, None, None, :].expand_as(h_jobs_padding)
        h_opes_pooled_padding = h_opes_pooled[:, None, None, :].expand_as(h_jobs_padding)

        h_actions = torch.cat((h_jobs_padding, h_mas_padding, h_opes_pooled_padding, h_mas_pooled_padding),
                              dim=-1).transpose(1, 2)
        h_pooled = torch.cat((h_opes_pooled, h_mas_pooled), dim=-1)
        scores = self.actor(h_actions).flatten(1)
        mask = eligible.transpose(1, 2).flatten(1)

        scores[~mask] = float('-inf')
        action_probs = F.softmax(scores, dim=1)
        state_values = self.critic(h_pooled)
        dist = Categorical(action_probs.squeeze())
        action_logprobs = dist.log_prob(action_envs)
        dist_entropys = dist.entropy()
        return action_logprobs, state_values.squeeze().double(), dist_entropys