import torch.nn as nn
from tensordict import TensorDict
import torch
import numpy as np
import time
from torchrl.envs import EnvBase
from numba import jit
from typing import Callable, List, Optional, Set, Tuple, Union



from EasyNCO.utils.utils import getLogger


from EasyNCO.neural_solvers.methods.htsp import models
from EasyNCO.neural_solvers.methods.htsp.utils_htsp import *
from EasyNCO.data.data_utils import augment_pomo
from EasyNCO.data import TSPGenerator
from EasyNCO.data.TSPGenerator import random_tsp_generator
from EasyNCO.neural_solvers.methods.pomo.htsp_pomo_policy import Policy_SHPP


htsp_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(htsp_dir)))



DISTANCE_SCALE = 1000
STATE_KEYS= ['x', 'device', 'graph_size', 'k', 'dist_matrix', 'knn_neighbor', 
               'numpy_knn_neighbor', 'selected_mask', 'available_mask', 'neighbor_coord', 
               'current_tour', 'current_num_cities', 'current_tour_len']


logger = getLogger(__name__)

class HTSPPolicy(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

        # initiate the lower level model
        model_path = root_dir + kwargs["low_level"]["lower_path"]
        model_ckpt = torch.load(model_path)
        self.low_level_model = Policy_SHPP()
        self.low_level_model.load_state_dict(model_ckpt['state_dict'])

        self.low_level_type = kwargs["low_level_type"]


        if self.low_level_type  == "pomo":
            self.lower_solver = RLSolver(self.low_level_model)

        elif self.low_level_type == "lkh":
            self.lower_solver = LKHSolver()

        else:
            logger.warning(f"{self.low_level_type} is not support!")


        # set encoder for PPO
        self.encoder = models.IMPALAEncoder(**kwargs["Encoder"])
        self.encoder_target = models.IMPALAEncoder(**kwargs["Encoder"]) 

        # set actor for PPO
        self.actor = models.ActorPPO(**kwargs["Actor"])
        
        # set crtic for PPO
        self.critic = models.CriticPPO(**kwargs["Critic"])
        self.critic_target = models.CriticPPO(**kwargs["Critic"])

        # parameters
        self.target_step = kwargs.get("target_step",None)
        self.reward_scale = kwargs.get("reward_scale", None)
        self.gamma = kwargs.get("gamma",None)
        self.experience_items = kwargs.get("experience_items",None)
        self.env_num = kwargs.get("env_num", None)
        self.grad_method = kwargs.get("grad_method" , None)
        self.lambda_GAE = kwargs.get("lambda_GAE" , None)
        self.data_augment = kwargs.get("data_augment", None)
        self.kwargs = kwargs

        # state
        self.states = None

        # env
        self.env  = HTSPEnv(**kwargs["low_env"])

        # Memory
        self.trajectory_list = [[[] for _ in range(self.experience_items)] for _ in range(self.env_num)]  # (env_num,item_num)

        # fragment for low_level
        self.frag_buffer = FragmentBuffer(kwargs["low_level"]["low_level_buffer_size"], 
                                                     kwargs["low_level"]["frag_len"], 
                                                     kwargs["low_level"]["node_dim"])
        
        self.frag_buffer_eval = FragmentBuffer(kwargs["low_level"]["low_level_buffer_size"], 
                                                     kwargs["low_level"]["frag_len"], 
                                                     kwargs["low_level"]["node_dim"])
        
        # deivce
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")




    def forward(self, state):

        with torch.no_grad():
            # in this case, device == "GPU"
            state = state.to(self.device) 
            # encode the graph
            graph_feature = self.encoder(state)
            # get action from actor network
            action, noise = self.actor.get_action(graph_feature)

        return action.detach(), noise.detach()
    

    def get_value(self, state, is_target = True):

        # get value from critic target
        if is_target == True:
           graph_feature = self.encoder_target(state) 
           value = self.critic_target(graph_feature)
        # get value from critic
        else:
            graph_feature = self.encoder(state)
            value = self.critic(graph_feature)

        return value
    


    @torch.no_grad() # in case of OMM
    def explore_trajectory(self, TSP_data):
        
        if self.states is None:
            self.states = self.env.reset(TSP_data)
            # states:tensors(env_num, dims)
        self.env.to_device_(self.device) # to "gpu"

        step = 0
        states = self.states
        last_dones = torch.zeros(self.env_num, dtype=torch.int, device=self.device)
        #last_dones:(env_num,)
        # env_num = batch_size
        trajectory_list = []
        # trajectory_list:(step_num, item_num=6, env_num)
        while step < self.target_step:
            actions, noises = self.forward(states)
            next_states, rewards, dones, info = self.env.step(actions.tanh(),solver=self.lower_solver, frag_buffer=self.frag_buffer)
            trajectory_list.append((states.clone(), rewards.clone(), dones.clone(),actions,noises,info["start_city"]))
            states = next_states
            step += 1
            last_dones[torch.where(dones)[0]] = step
        

        self.states = states

        buffer_traj = list(map(list, zip(*trajectory_list)))
        del trajectory_list[:]

        buffer_traj[0] = torch.stack(buffer_traj[0]) #states
        buffer_traj[1] = (torch.stack(buffer_traj[1]) * self.reward_scale).unsqueeze(2) # rewards
        buffer_traj[2] = ((1 - torch.stack(buffer_traj[2])) * self.gamma).unsqueeze(2)  # dones
        buffer_traj[3] = torch.stack(buffer_traj[3]) # actions
        buffer_traj[4] = torch.stack(buffer_traj[4]) # noises
        buffer_traj[5] = torch.stack(buffer_traj[5]) # start_city

        experiences = []
        experiences.append(self.splice_trajectory(buffer_traj, last_dones)[0]) #(experience_num, sum_of_steps)
        self.env.to_device_("cpu")
        return experiences
    

    def splice_trajectory(self, buffer_traj, last_dones):

        experiences = [] #(item_num,[sums of valid steps, dims])
        for idx in range(self.experience_items): # experiences_items: item_num

            current_buffer_item = buffer_traj.pop(0) #(step_num,env_num)
            current_item = [] # for one specific experience item (length:env_num)

            for env_idx in range(self.env_num): #  env_num: the number of upper environment
                last_step = last_dones[env_idx]

                previous_buffer_item = self.trajectory_list[env_idx][idx]
                if len(previous_buffer_item):
                    current_item.append(previous_buffer_item)

                current_item.append(current_buffer_item[:last_step,env_idx])
                self.trajectory_list[env_idx][idx] = current_buffer_item[last_step:,env_idx]

            experiences.append(torch.vstack(current_item).detach().cpu())

        return [experiences]
    

    def collect_experience(self,batch,num_steps): 
        """This function is not usable, but is the template for PPO"""
         
        states, actions, rewards, old_log_probs = [], [], [], []

        for _ in range(num_steps):
            state = self.env.reset(batch)
            action, log_prob = self.policy(state)
            next_state, reward, done, _ = self.env.step(action, solver=self.lower_solver,frag_buffer=self.frag_buffer)
        
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            old_log_probs.append(log_prob)
        
            if done:
                continue
            else:
                state = next_state
        return states, actions, rewards, old_log_probs

        
    
    def low_level_trainning(self, optimizer:torch.optim.Optimizer):
        """This function is for joint training"""

        for _ in range(self.kwargs["low_level"]["update_time"]):
            data = self.frag_buffer.sample_batch(self.kwargs["low_level"]["batch_size"]).to(self.device)
            outputs = self.low_level_model.low_level_training(data,norm_reward = True,
                                                                           group_size = self.kwargs["low_level"]["group_size"])
            self.optimizer_update(optimizer, outputs["loss"])




    def optimizer_update(self,optimizer:torch.optim.Optimizer, objective: torch.Tensor) -> None:

        optimizer.zero_grad()
        self.manual_backward(objective)
        for param_group in optimizer.param_groups:
            if self.grad_method == "norm":
                nn.utils.clip_grad_norm_(parameters=param_group["params"], max_norm=self.grad_clip)
            elif self.grad_method == "value":
                nn.utils.clip_grad_value_(parameters=param_group["params"], clip_value=self.grad_clip)
            else:
                raise ValueError(f"gradient method could not be {self.grad_method}, can only be norm and value.")
        optimizer.step()




    def get_GAE_reward(self, memory_len, rewards, masks, values):

        rewards_sum = torch.empty(memory_len, dtype = torch.float32, device=rewards.device)
        advantages = torch.empty(memory_len, dtype = torch.float32, device=rewards.device)

        previous_reward_sum =  0
        previous_advantge = 0

        mask_bool = torch.not_equal(masks, 0).float()
        for idx in range(memory_len-1, -1, -1):

            rewards_sum[idx]  = rewards[idx] + mask_bool[idx] * previous_reward_sum
            previous_reward_sum = rewards_sum[idx]
            # td_error
            advantages[idx] = rewards[idx] + mask_bool[idx] * (previous_advantge - values[idx])
            previous_advantge = values[idx] + advantages[idx] * self.lambda_GAE

        return rewards_sum, advantages
   



    def HTSP_test_step(self,data):

        if self.data_augment == True:

            TSP_data = {"problem":data,"aug_factor":8}
            data = augment_pomo(**TSP_data)


        val_env = HTSPEnv(**self.kwargs["low_env"])
        # k, frag_len, max_new_nodes, max_improvement_steps

        result_length = np.array([]) # record the output of length
        start_time = time.time()
        batch_time = time.time() # the start time for a batch
        batch_time_limit = self.kwargs["eval"]["time_limit"] * self.kwargs["eval"]["batch_size"]

        # output state from env
        states = val_env.reset(data.to(self.device))

        while not val_env.done:
            
            graph_feature = self.encoder(states)
            actions = self.actor(graph_feature).detach()

            states, rewards, dones, info = val_env.step(actions, solver=self.lower_solver, frag_buffer=self.frag_buffer_eval)


        if self.kwargs["eval"]["improvement_step"] > 0:
            for env in val_env.envs:
                env.max_improvement_step = self.kwargs["eval"]["improvement_step"]
            while not val_env.done:
                if time.time() - batch_time > batch_time_limit:
                    break
                action = val_env.random_action().to(self.device)
                state, reward, done, info = val_env.step(action, solver=self.lower_solver, frag_buffer=self.frag_buffer_eval)

        #average length
        length = np.array([val_state.current_tour_len.item() for val_state in val_env.states])      
        result_length = np.concatenate((result_length, length))

        duration = time.time() - start_time

        if self.data_augment == True:
            result_length = result_length.reshape(8, -1).min(axis=0)

        return duration, result_length.mean()



    def HTSP_validation_step(self,data):

        Env_param = self.kwargs["low_env"]
        Env_param["auto_reset"] = False
        self.env_val = HTSPEnv(**Env_param)
        
        states = self.env_val.reset(data)
        rewards_list = [] 

        while not self.env_val.done:

            with torch.no_grad():
                actions = self.actor(self.encoder(states)).detach().squeeze()

            states, rewards, dones, info = self.env_val.step(actions, solver=self.lower_solver, frag_buffer=self.frag_buffer_eval )
            rewards_list.append(rewards)

        average_length = np.mean([env_state.current_tour_len.item() for env_state in self.env_val.states])    
        discounted_rewards = torch.zeros_like(rewards_list[0])
        for reward in rewards_list[::1]:
            discounted_rewards = discounted_rewards * self.gamma + reward  

        del self.env_val

        return average_length, discounted_rewards.mean()



class HTSPEnv():

    def __init__(self, 
                 k: int,
                 frag_len: int = 200,
                 max_new_nodes: int = 160,
                 max_improvement_step: int = 5,
                 auto_reset=False,
                 no_depot=False,
                 solver: type = None,
    **kwargs):

        self.k = kwargs.get("k", k)
        self.frag_len = kwargs.get("frag_len", frag_len)
        self.max_new_nodes = kwargs.get("max_new_nodes", max_new_nodes)
        self.max_improvement_step = kwargs.get("max_improvement_step", max_improvement_step)
        self.auto_reset = kwargs.get("auto_reset", auto_reset)
        self.no_depot = kwargs.get("no_depot", no_depot)

    @property
    def done(self):
        return all(state.done for state in self.states)


    def reset(self, x: torch.Tensor):

        self.x = x
        self.device = x.device
        self.batch_size = x.shape[0]
        self.graph_size = x.shape[1]
        self.node_dim = x.shape[2]

        states = []
        states_tensor = []
        # reset function from Env(original)
        for i in  range (self.batch_size):
            x_data = x[i].type(torch.float32)
            dist_matrix = torch.cdist(x_data.type(torch.float64) * DISTANCE_SCALE,
                                      x_data.type(torch.float64) * DISTANCE_SCALE).type(torch.float32)
            dist_matrix.fill_diagonal_(float("inf"))
            knn_neighbor = dist_matrix.topk(k = self.k, largest= False).indices
            if self.no_depot:
                init_tour = []
            else:
                depot = 0
                init_tour = [depot, knn_neighbor[depot][0].item()]
            state = LargeState(x = x_data,
                                k = self.k,
                                init_tour = init_tour,
                                dist_matrix = dist_matrix,
                                knn_neighbor = knn_neighbor,
                                max_improvement_step = self.max_improvement_step,
                                max_new_nodes = self.max_new_nodes,
                                frag_len = self.frag_len)
            state_tensor = state.to_tensor()
            state._improvement_steps = 0
            states.append(state)
            states_tensor.append(state_tensor)

        self.states = states
        self.states_tensor = states_tensor
        return torch.stack(states_tensor).type(torch.float32)
    
    
    def step(self,
        predict_coords: torch.Tensor,
        solver: type,
        greedy_reward: bool = False,
        average_reward: bool = False,
        frag_buffer: FragmentBuffer = None):


        actions = [self.scale_action(coord) for coord in predict_coords]  # treat predict_coord as actions
        if not self.auto_reset:
            # for eval
            active_idx = [i for i in range(self.batch_size) if not self.states[i].done]
            active_state = [self.states[i] for i in active_idx]
            active_acts = [actions[i] for i in active_idx]
        else:
            # for train
            active_idx = list(range(self.batch_size))
            active_state = self.states
            active_acts = actions

        fragments = []
        new_cities = []
        start_cities = []
        for state, action in zip(active_state, active_acts):
            """frag:
               new_city:
               start_city: """
            frag, new_city, start_city = state.get_fragment_knn(action)
            fragments.append(frag)
            new_cities.append(new_city)
            start_cities.append(start_city) 

        if isinstance(solver, RLSolver):
            new_paths, _ = solver.solve(self.x[active_idx], torch.stack(fragments), frag_buffer)
        elif isinstance(solver, LKHSolver):
            np_fragments = [frag.cpu().numpy() for frag in fragments]
            new_paths, _ = solver.solve(self.x[active_idx].cpu().numpy(), np.stack(np_fragments))
        
        outputs = []
        for state, path in zip(active_state, new_paths):
            # result [1:state 2:reward 3:self.done] 
            result = state._step(path, greedy_reward, average_reward)
            outputs.append(result)

        if not self.auto_reset:
            # for eval
            states_tensor = [None] * self.batch_size
            rewards = [0.0] * self.batch_size
            dones = [False] * self.batch_size
            count = 0
            for i in range(self.batch_size):
                if i in active_idx:
                    states_tensor[i] = outputs[count][0]
                    rewards[i] = outputs[count][1]
                    dones[i] = outputs[count][2]
                    count += 1
                    
        else:
            # for train
            states_tensor = [output[0] for output in outputs]
            rewards = [output[1] for output in outputs]
            dones = [output[2] for output in outputs]
            for i in range(self.batch_size):
                state = self.states[i]
                if state.done:
                    data = random_tsp_generator(num_sample=1,num_nodes=self.x.shape[1], device=self.device, distribution="uniform")
                    tsp_data = data[0].squeeze()
                    # if done, reload data
                    state_curr_tensor = state._reset(tsp_data,no_depot = self.no_depot)
                    self.x[i] = tsp_data
                    states_tensor[i] = state_curr_tensor
        
        for i in active_idx:
            self.states_tensor[i] = states_tensor[i]


        return (torch.stack(self.states_tensor).type(torch.float32),
                torch.tensor(rewards, dtype=torch.float32, device=self.device),
                torch.tensor(dones, dtype=torch.float32, device=self.device),
                {"fragments": fragments,"new_cities": new_cities,
                 "start_city": torch.stack(start_cities),})



    @staticmethod
    def unscale_action(a):
        """unscale action from [0, 1] to [-1, 1]"""
        return a * 2 - 1


    @staticmethod
    def scale_action(a):
        """scale action from [-1, 1] to [0, 1]"""
        return a * 0.5 + 0.5


    def to_device_(self, device: Optional[Union[torch.device, str]] = None):
        for state in self.states:
            state.to_device_(device)


    def random_action(self):
        return self.unscale_action(torch.randn(size=(self.batch_size, self.node_dim)))
    
    def heuristic_action(self):
        return torch.stack([state.heuristic_action() if not state.done else state.random_action()
                for state in self.states],dim=0,)


class LargeState():

    def __init__(self,
                x: torch.Tensor,
                k: int,
                init_tour: List[int],
                dist_matrix: torch.Tensor = None,
                knn_neighbor: torch.Tensor = None,
                max_improvement_step: int = 5,
                max_new_nodes: int = 160,
                frag_len: int = 200):
        
        self.x = x
        self.device = x.device
        self.graph_size = x.shape[0]
        # k for k-Nearest-Neighbor
        self.k = k
        self._improvement_steps = 0 # set to zero when first initialization
        self.max_improvement_step = max_improvement_step
        self.max_new_nodes = max_new_nodes
        self.frag_len = frag_len

        if dist_matrix is None:
            self.dist_matrix = torch.cdist(x.type(torch.float64) * DISTANCE_SCALE,
                                           x.type(torch.float64) * DISTANCE_SCALE).type(torch.float32)
        else:
            self.dist_matrix = dist_matrix

        if knn_neighbor is None:
            # the cloest neigbor is itself here
            self.knn_neighbor = self.dist_matrix.topk(k =self.k+1 , largest= False).indices[:,1:]
        else:
            self.knn_neighbor = knn_neighbor
        self.numpy_knn_neighbor = self.knn_neighbor.cpu().numpy()

        self.selected_mask = torch.zeros(x.shape[0], dtype = torch.bool, device= self.device) 
        self.available_mask = torch.zeros(x.shape[0], dtype = torch.bool, device = self.device)
        self.neighbor_coord = torch.zeros((x.shape[0], 4), dtype = torch.float32, device = self.device)

        # start with a 2-city-tour or empty tour
        self.current_tour = init_tour.copy()
        self.current_num_cities = len(self.current_tour)
        # to make final return equals to tour length
        self.current_tour_len = (self.get_tour_distance(tour = init_tour, dist_matrix = self.dist_matrix) / DISTANCE_SCALE)
        self.update_neighbor_coord_(self.neighbor_coord, self.current_tour, self.x)
        self.mask(self.current_tour)



    def get_tour_distance(self,tour: List[int],dist_matrix: torch.Tensor):
        """This is length of partial solution(tour) , original version from utils"""
        _tour = torch.tensor(tour, dtype = torch.int64, device = dist_matrix.device)
        _tour_offset = torch.tensor(tour[1:] + tour[0:1], dtype = torch.int64, device=dist_matrix.device)
        edge_list = torch.stack([_tour, _tour_offset],dim = 1)
        tour_length = (dist_matrix.gather(index=edge_list[:,0][:,None].expand(-1, dist_matrix.shape[0]),dim = 0)
                       .gather(index = edge_list[:,1][:,None], dim = 1).sum())
        return tour_length
    


    def update_neighbor_coord_(self, neighbor_corrd:torch.Tensor,
                               current_tour: List[int], nodes_coord:torch.Tensor):
        """orginal version from utils"""
        # neighbor_corrd.shape[num_nodes,4] , nodes_coord.shape[num_nodes,2]
        num_nodes = nodes_coord.shape[0]
        node_dim = nodes_coord.shape[1]
        tour  = torch.tensor(current_tour,dtype=torch.int64, device = nodes_coord.device)
        curr_pre_next = torch.stack([tour, torch.roll(tour, -1),torch.roll(tour,1)],dim=1)
        pre_corrd = nodes_coord.gather(index = curr_pre_next[:,1][:,None].expand(-1, node_dim), dim=0)
        next_corrd = nodes_coord.gather(index = curr_pre_next[:,2][:,None].expand(-1, node_dim), dim=0)
        pre_next_corrd = torch.cat([pre_corrd, next_corrd], dim =-1)
        neighbor_corrd.scatter_(index=curr_pre_next[:,0][:,None].expand(-1,2 * node_dim),
                                src = pre_next_corrd,
                                dim = 0)
        
    

    def mask(self, new_path:List[int]):
        self.selected_mask[new_path] = True
        # update mask of available starting cities for k-NN process
        self.available_mask[new_path] = True
        avaliable_cities = torch.where(self.available_mask == True)
        available_status = torch.logical_not(
                            self.selected_mask[self.knn_neighbor[avaliable_cities]].all(dim=1))
        self.available_mask[avaliable_cities] = available_status


    
    def get_fragment_knn(self,predict_coord: torch.Tensor):
        if self.done:
            return []
        
        if not self._construction_done:
            nearest_new_city = self.get_nearest_new_city_idx(predict_coord)
            if self.current_num_cities != 0:
                start_city = self.get_nearest_old_city_idx(self.x[nearest_new_city])
                new_city_start = self.get_nearest_new_city_idx(self.x[start_city])
            else:
                start_city = None
                new_city_start = nearest_new_city
            # search for new cities with k-NN heuristic
            # max_cities: the maximum capacity for fragment 
            max_cities = max(self.max_new_nodes, self.frag_len - self.current_num_cities)    
            new_cities = []
            knn_deque = []
            selected_set = set()
            knn_deque.append(new_city_start)
            selected_set.add(new_city_start)
            selected_mask = self.selected_mask.cpu().numpy()
            while len(knn_deque) != 0 and len(new_cities) < max_cities:
                node = knn_deque.pop(0)
                new_cities.append(node)
                new_nodes = self.append_selected(self.numpy_knn_neighbor[node], selected_set, selected_mask)
                knn_deque.extend(new_nodes)
                selected_set.update(new_nodes)
        else:
            nearest_old_city = self.get_nearest_old_city_idx(predict_coord)
            new_cities = []

        if self.current_num_cities != 0:
            fragment = self._extend_fragment(start_city, new_cities)
        else:
            fragment = new_cities # initialzation
            
        for i in fragment:
            if i in new_cities:
                assert not self.selected_mask[i]
            else:
                assert self.selected_mask[i]

        return (torch.tensor(fragment, device=self.device),
                torch.tensor(new_cities, device = self.device),
                HTSPEnv.unscale_action(self.x[start_city]))
    


    def _extend_fragment(self, nearest_city:int , new_cities: List[int]):

        # nearest_idx: What position is it in the current_tour
        nearest_idx = self.current_tour.index(nearest_city)
        #total_extend_len: the number of old cities contained in frag_len
        total_extend_len = self.frag_len - len(new_cities)

        offset = nearest_idx - (total_extend_len // 2)
        reorder_tour = self.current_tour[offset:] + self.current_tour[:offset]
        fragment = reorder_tour[:(total_extend_len // 2)] +\
                    new_cities + reorder_tour[(total_extend_len // 2):total_extend_len]
        return fragment
    

        

    def _step(self, new_path:List[int], 
              greedy_reward: bool = False, 
              average_reward: bool = False):
        
        """This function is from Env"""
        if self._construction_done and not self._improvement_done:
            self._improvement_steps += 1
        
        # record old states
        old_len = self.current_tour_len
        old_num_cities = self.current_num_cities
        # update state
        self.move_to(new_path)
        self.current_tour_len = (self.get_tour_distance(self.current_tour,
                                                         self.dist_matrix)/ DISTANCE_SCALE)
        if greedy_reward:
            len_rl = (self.get_tour_distance(new_path, self.dist_matrix)
                        - self.dist_matrix[new_path[0], new_path[-1]]) / DISTANCE_SCALE
            _, len_greedy = GreedySolver().solve(self.x, new_path)
            reward = len_greedy - len_rl
        else:
            reward = old_len - self.current_tour_len
        if average_reward:
            added_num_cities = self.current_num_cities - old_num_cities
            reward /= added_num_cities

        return self.to_tensor(), reward, self.done         


    def move_to(self, new_path:List[int]):
        """add new_path to current_tour"""
        if self.current_num_cities == 0:
            self.current_tour = new_path
        else:
            start_idx = self.current_tour.index(new_path[0])
            end_idx = self.current_tour.index(new_path[-1])
            if end_idx > start_idx:
                self.current_tour = (self.current_tour[:start_idx]+ new_path
                                    + self.current_tour[end_idx + 1 :])
            else:
                self.current_tour = (self.current_tour[end_idx + 1 : start_idx] + new_path)

        self.current_num_cities = len(self.current_tour)
        self.current_tour_len = (self.get_tour_distance(self.current_tour, self.dist_matrix) /DISTANCE_SCALE)
        self.update_neighbor_coord_(self.neighbor_coord, self.current_tour, self.x)
        # update two masks
        self.mask(new_path) 


    def to_device_(self, device):
        self.x = self.x.to(device)
        self.device = self.x.device
        self.dist_matrix = self.dist_matrix.to(device)
        self.knn_neighbor = self.knn_neighbor.to(device)
        self.selected_mask = self.selected_mask.to(device)
        self.available_mask = self.available_mask.to(device)
        self.neighbor_coord = self.neighbor_coord.to(device)


    def _reset(self, x, no_depot):
        self.x = x.type(torch.float32)
        self.dist_matrix = torch.cdist(self.x.type(torch.float64) * DISTANCE_SCALE,
                                      self.x.type(torch.float64) * DISTANCE_SCALE).type(torch.float32)
        self.dist_matrix.fill_diagonal_(float("inf"))
        self.knn_neighbor = self.dist_matrix.topk(k = self.k, largest= False).indices
        if no_depot:
            self.init_tour = []
        else:
            depot = 0
            self.init_tour = [depot, self.knn_neighbor[depot][0].item()]

        self.selected_mask = torch.zeros(x.shape[0], dtype = torch.bool, device= self.device) 
        self.available_mask = torch.zeros(x.shape[0], dtype = torch.bool, device = self.device)
        self.neighbor_coord = torch.zeros((x.shape[0], 4), dtype = torch.float32, device = self.device)

        # start with a 2-city-tour or empty tour
        self.current_tour = self.init_tour.copy()
        self.current_num_cities = len(self.current_tour)
        # to make final return equals to tour length
        self.current_tour_len = (self.get_tour_distance(tour = self.init_tour, dist_matrix = self.dist_matrix) / DISTANCE_SCALE)
        self.update_neighbor_coord_(self.neighbor_coord, self.current_tour, self.x)
        self.mask(self.current_tour)
        

        return self.to_tensor()



    @staticmethod
    @jit(nopython=True)
    def append_selected(neighbor: np.ndarray, selected_set: Set[int], selected_mask: np.ndarray):
        result = []
        for n in neighbor:
            if n not in selected_set and not selected_mask[n]:
                result.append(n)
        return result



    def get_nearest_new_city_idx(self,predict_coord: torch.Tensor):
        """orginal version from utils"""
        masked_cities = torch.where(~self.selected_mask == True)[0]
        dist_matrix = torch.cdist(self.x[masked_cities].type(torch.float64),
                                  predict_coord[None,:].type(torch.float64)).type(self.x.dtype)
        city_idx = masked_cities[dist_matrix.argmin()].item()
        return city_idx 


    def get_nearest_old_city_idx(self,predict_coord: torch.Tensor):
        """orginal version from utils"""
        masked_cities = torch.where(self.selected_mask == True)[0]
        dist_matrix = torch.cdist(self.x[masked_cities].type(torch.float64),
                                  predict_coord[None,:].type(torch.float64)).type(self.x.dtype)
        city_idx = masked_cities[dist_matrix.argmin()].item()
        return city_idx
    

    def random_action(self):
        action = torch.randn(size=(self.node_dim,), device=self.device)
        return self.unscale_action(action)


    def heuristic_action(self):
        non_selected_idx = torch.where(self.state.selected_mask == False)[0]
        random_choice = np.random.randint(0, non_selected_idx.shape[0])
        new_idx = non_selected_idx[random_choice]
        action = self.x[new_idx]
        return self.unscale_action(action)



    @property
    def _construction_done(self):
        return self.current_num_cities == self.graph_size

    @property
    def _improvement_done(self):
        return self._improvement_steps >= self.max_improvement_step

    @property
    def done(self):
        return self._construction_done and self._improvement_done


    def to_tensor(self):
        """concatenate `x`, `selected_mask` to produce an state"""
        available_mask = self.available_mask
        return torch.cat([self.x, self.selected_mask[:, None],
                          available_mask[:, None], self.neighbor_coord],dim=1)







        