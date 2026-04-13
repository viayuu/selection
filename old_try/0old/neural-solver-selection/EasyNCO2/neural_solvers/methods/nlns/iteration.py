import math
import time
import torch
import numpy as np
from copy import deepcopy
from EasyNCO.neural_solvers.pipeline import Iteration
from typing import Literal, Optional
from tensordict import TensorDict
from torchrl.envs import EnvBase

EMA_ALPHA = 0.2

class NLNSIteration(Iteration):
    """
    A class for NLNS iteration that extends the Iteration class.
    It is used to create initial solutions for training in the NLNS framework.
    """
    def __init__(self, policy, **kwargs):
        super().__init__(policy)
        self.solutions_batches = []
        self.batch_id = -1
        self.nb_train_batches = int(kwargs.get('episodes') / kwargs.get('batch_size')) + 1


    def run(self,
            td: TensorDict,
            env: EnvBase,
            initialization_out: dict,
            phase: Literal["train", "eval"],
            max_steps: int = 0,
            **kwargs,
    ) -> dict:
        
        solution = td['solution']
        decoder_strategy = td['strategy']

        tool_params = kwargs.get('tool_params')
        nlns_phase = kwargs.get('nlns_phase')
        split_delivery = kwargs.get('split_delivery')
        tool = Iteration_tool(
            problems=env.problems,
            nlns_phase=nlns_phase,
            split_delivery=split_delivery,
            nb_train_batches=self.nb_train_batches,
            **tool_params
        )

        if phase == 'train':
            self.batch_id += 1
            reset_td = tool.reset(solution, batch_size=env.env_batch_size)

            self.policy.set_decoder_strategy(decoder_strategy)
            self.policy.pre_forward(reset_td)

            likelihood = torch.zeros(size=(env.env_batch_size, env.pomo_size, 0))
            done = False
            reward = None
            state_td = tool.pre_step()

            while not done:
                next_td = self.policy(state_td)
                state_td = tool.step(next_td)
                prob = state_td.get("prob", None)
                # shape: (batch, pomo)
                likelihood = torch.cat((likelihood, prob[:, :, None]), dim=2)

                reward = state_td["reward"]
                done = state_td["done"].all()

            solution = state_td["solution"]
            if self.batch_id >= self.nb_train_batches:
                self.solutions_batches[self.batch_id % self.nb_train_batches] = solution
            else:
                self.solutions_batches.append(solution)

            reward = reward.unsqueeze(1)
            # reward: shape: (batch, pomo)
            # likelihood: shape: (batch, pomo, num_input)

            policy_out = {
                "reward": reward,
                "likelihood": likelihood,
            }

            out = policy_out

        else:

            timelimit_batch = tool.timelimit_batch
            lns_reheating_nb = tool.lns_reheating_nb

            policy_dict = self.policy
            # 'self.policy' here is a dictionary that contains:
            # models (list), destruction strategies (list), and different parameters (list)
            # In testing phase, the model and destruction strategy along with its parameter
            # are selected from the lists either randomly or based on some strategies.

            policy_pairs = policy_dict['models']
            destroy_operations_pairs = policy_dict['destroy_operations']
            p_destructions_pairs = policy_dict['p_destructions']

            tool.destroy_procedure = destroy_operations_pairs
            tool.destruction_p = p_destructions_pairs

            with torch.inference_mode():
                
                iteration = 0
                if tool.phase == 'test_batch':
                    timelimit = timelimit_batch
                else:
                    timelimit = timelimit_batch / lns_reheating_nb
                    solution = [solution[0] for _ in range(tool_params['lns_batch_size'])]

                reset_state = tool.reset(solution, batch_size=env.env_batch_size)
                start_time = time.time()

                results = []

                while (iteration < max_steps) and (time.time() - start_time <= timelimit):
                    iteration += 1
                    
                    state_td = tool.pre_step()
                    done_all = state_td['done']
                    done = done_all.all()

                    policy = policy_pairs[int(tool.selected_id)]

                    policy.set_decoder_strategy(strategy = 'greedy')
                    policy.eval()
                    policy.pre_forward(reset_state)

                    while not done:
                        next_td = policy(state_td)
                        state_td = tool.step(next_td)
                        reward = state_td['reward']
                        # shape: (batch, )
                        done = state_td['done'].all()

                    results.append(reward.mean())
                    # print(reward)

            # Return
            no_aug_score = results[-1]
            aug_score = results[-1]

            out = {
                "no_aug_score": no_aug_score,
                "aug_score": aug_score,
            }

        return out


class Iteration_tool():
    def __init__(self, problems, nlns_phase, split_delivery, nb_train_batches, seed=2024, **kwargs):

        self.problems = problems
        self.problem_size = problems.size(1) - 1
        self.depot_node_xy = None
        self.capacity = None

        # Dynamic
        self.selected_count = None
        self.current_node = None
        self.selected_node_list = None

        self.at_the_depot = None
        self.load = None
        self.ninf_mask = None
        self.visited_ninf_flag = None
        self.finished = None

        self.dummy_flag_bool = None
        self.dummy_flag_long = None

        self.seed_value = seed
        self._set_seed(seed = seed)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # NLNS
        self.solutions = None
        self.incomplete_tours = None
        self.nn_input_idx_to_tour = None
        self.open_nn_input_idx = None

        self.dynamic_input_critic = None
        self.static_input_critic = None

        self.phase = nlns_phase
        self.costs_memory = None

        self.destroy_procedure = kwargs.get('destroy_procedure')
        self.destruction_p = kwargs.get('destruction_p')
        self.round = kwargs.get('round', False)
        self.split_delivery = split_delivery

        # train
        self.nb_train_batches = nb_train_batches
        self.solutions_batches =[]
        # test_batch
        self.lns_adaptive_search = kwargs.get('adaptive_search', False)
        # test_single
        self.timelimit_batch = kwargs.get('timelimit_batch')
        self.lns_reheating_nb = kwargs.get('lns_reheating_nb')
        self.lns_batch_size = kwargs.get('lns_batch_size', 10)
        self.lns_Z_param = kwargs.get('Z_param')
        self.T_min = kwargs.get('T_min')

    def _set_seed(self, seed: Optional[int]):
        rng = torch.manual_seed(seed)
        self.rng = rng

    def reset(self, solution, batch_size = None) -> TensorDict:

        if self.phase == 'test_single':
            batch_size = self.lns_batch_size
            self.problems = self.problems.repeat(batch_size, 1, 1)
            # single instance repeated batch_size, shape: (batch, 1 + problem, 3)
        else:
            self.problems = self.problems
            # shape: (batch, 1 + problem, 3)

        self.env_batch_size = batch_size
        self.batch_size = torch.Size([batch_size])

        self.depot_node_xy = self.problems[:, :, :2]
        # shape: (batch, 1 + problem, 2)
        self.depot_node_demand = self.problems[:, :, 2]
        # shape: (batch, 1 + problem)
        round_error_epsilon = 0.00001
        self.capacity = torch.ones(size = (self.env_batch_size, 1, 1)) + round_error_epsilon
        # shape: (batch, pomo=1, 1)
        self.vehicle_capacity = 1.0 + round_error_epsilon

        if self.phase == 'train' and self.batch_id >= self.nb_train_batches:
            self.solutions = self.solutions_batches[self.batch_id % self.nb_train_batches]
        else:
            self.solutions = solution # bacth_solution, list
        
        if self.phase == 'train':
            self.costs = self.get_costs(self.round) # list, shape: (batch, )
        else:
            self.costs_memory = []
            for _ in range(self.env_batch_size):
                self.costs_memory.append(np.full((self.problem_size + 1, self.problem_size + 1), np.nan, dtype="float"))
            self.costs = self.get_costs_memory(self.round)

        assert self.env_batch_size == self.depot_node_xy.size(0), \
            ('batch_size and the first dimension of problems should be the same. ' +
             f'Expected batch_size: {self.depot_node_xy.size(0)}, got: {self.env_batch_size}')
        

        if self.phase == 'test_batch':
            self.performance_EMA = [np.inf] * len(self.destroy_procedure)
            # Exponential moving average of avg. improvement in last iterations

        elif self.phase == 'test_single':
            self.start_time_reheating = time.time()
            self.single_solution = self.solutions[0] # current single solution
            self.incumbent_solution = deepcopy(self.single_solution) # best single solution
            self.costs = self.costs[0] # float
            self.iter = -1
            self.cur_cost = np.inf

        return TensorDict({
            'locs' : self.problems,
            'vehicle_capacity': self.capacity,
        }, batch_size = torch.Size([self.env_batch_size])
        )
    

    def pre_step(self) -> TensorDict:
        self.open_nn_input_idx = []
        self.nn_input_idx_to_tour = []

        if self.phase == 'test_batch':
            self.mean_cost_before_iteration = np.mean(self.costs)
            self.solutions_copies = self.get_solution_copy()

            if self.lns_adaptive_search:
                self.selected_id = np.argmax(self.performance_EMA)
                # select operator pair with the best EMA
            else:
                self.selected_id = np.random.randint(0, len(self.destroy_procedure))
                # select operator pair at random
            destroy_operator = DestroyOperator(self.solutions, self.problems, self.destroy_procedure[int(self.selected_id)], self.destruction_p[int(self.selected_id)])
        
        elif self.phase == 'test_single':
            self.iter += 1
            for i in range(int(self.lns_Z_param * self.env_batch_size)):
                self.solutions[i] = deepcopy(self.single_solution)

            self.selected_id = np.random.randint(0, len(self.destroy_procedure))
            destroy_operator = DestroyOperator(self.solutions, self.problems, self.destroy_procedure[int(self.selected_id)], self.destruction_p[int(self.selected_id)])
        
        else:
            destroy_operator = DestroyOperator(self.solutions, self.problems, self.destroy_procedure, self.destruction_p)

        self.start_time_destroy = time.time()
        self.solutions, self.incomplete_tours= destroy_operator.destroy_instances()

        if self.phase == 'train':
            self.costs_destroyed = self.get_costs_incomplete(self.round) # list, shape: (bacth, )
        
        self.nb_input_points = max([get_max_nb_input_points(incomplete_tour) for incomplete_tour in self.incomplete_tours])  # Max. input points of batch
      
        # Create batch input
        self.static_input = np.zeros((self.env_batch_size, self.nb_input_points, 2))
        # array, shape: (batch, num_input, 2)
        self.dynamic_input = np.zeros((self.env_batch_size, self.nb_input_points, 2))
        # array, shape: (batch, num_input, 2)

        for i in range(self.env_batch_size):
            static_nn_input, dynamic_nn_input = self.get_network_input(self.nb_input_points, i)
            self.static_input[i] = static_nn_input
            self.dynamic_input[i] = dynamic_nn_input
        # update self.open_nn_input_idx, self.nn_input_idx_to_tour, in 'self.get_network_input()'

        self.static_input = torch.from_numpy(self.static_input).to(self.device).float()
        # tensor, shape: (batch, num_input, 2)
        self.dynamic_input = torch.from_numpy(self.dynamic_input).to(self.device).float()
        # tensor, shape: (batch, num_input, 2)

        self.dynamic_input_critic = deepcopy(self.dynamic_input)
        self.static_input_critic = deepcopy(self.static_input)

        self.instance_repaired = np.zeros(self.env_batch_size)
        # array, shape: (batch, )
        self.origin_idx = np.zeros((self.env_batch_size), dtype=int)
        # array, shape: (batch, )

        # if origin_idx == 0 select the next tour end that serves as the origin at random
        for i in range(self.env_batch_size):
            if self.origin_idx[i] == 0 and not self.instance_repaired[i]:
                self.origin_idx[i] = np.random.choice(self.open_nn_input_idx[i], 1).item()

        self.mask = get_mask(self.origin_idx, self.dynamic_input, self.nn_input_idx_to_tour, self.split_delivery, self.vehicle_capacity).to(self.device).float()
        # tensor, shape: (batch, num_input)
        origin_idx = torch.from_numpy(self.origin_idx)
        # tensor, shape: (batch, )

        self.dummy_flag_bool = torch.zeros((self.env_batch_size), dtype = torch.bool) # shape: (batch, )
        self.dummy_flag_long = torch.zeros((self.env_batch_size), dtype = torch.long) - 1 # shape: (batch, )
               
        next_state = {
            'mask' : self.mask,
            'static_input': self.static_input,
            'dynamic_input': self.dynamic_input,
            'origin_idx': origin_idx,
            "reward": self.dummy_flag_long,
            "done": self.dummy_flag_bool
        }

        out = TensorDict({
            'next' : next_state,
            'vehicle_capacity' : self.capacity,
            'reward' : self.dummy_flag_long,
            'done' : self.dummy_flag_bool,
        }, batch_size = self.batch_size
        )

        return out

    def step(self, td : TensorDict) -> TensorDict:
        ptr = td["selected"] # tensor, shape: (batch, )
        prob = td['prob'] # tensor, shape: (batch, )

        ptr_np = ptr.cpu().numpy() # array, shape: (batch, )
        nn_input_updates = []
        
        for i in range(self.env_batch_size):
            idx_from = self.origin_idx[i].item()
            idx_to = ptr_np[i].item()
            if idx_from == 0 and idx_to == 0:  # No need to update in this case
                continue

            nn_input_update, cur_nn_input_idx = self.do_action(idx_from, idx_to, i)
            # Connect origin to select point
            for s in nn_input_update:
                s.insert(0, i)
                nn_input_updates.append(s)

            # Update origin
            if len(self.open_nn_input_idx[i]) == 0:
                self.instance_repaired[i] = 1
                self.origin_idx[i] = 0  # If instance is repaired set origin to 0
            else:
                self.origin_idx[i] = cur_nn_input_idx  # Otherwise, set to tour end of the connect tour

        # Update network input
        nn_input_idx = np.array(nn_input_updates, dtype=int)
        nn_input_idx = torch.from_numpy(nn_input_idx).to(self.device).long()

        nn_input_update = torch.tensor(nn_input_updates, dtype=torch.float32).to(self.device)

        self.dynamic_input[nn_input_idx[:, 0], nn_input_idx[:, 1]] = nn_input_update[:, 2:]

        prob = prob.squeeze(1) * (1. - torch.from_numpy(self.instance_repaired).float().to(self.device))
        # shape: (batch, )
        prob = torch.where(prob == 0, torch.tensor(1), prob)
        prob = prob.unsqueeze(1) # shape: (batch, pomo)

        done_all = self.instance_repaired.all()

        if done_all:
            if self.phase == 'train':
                costs_repaired = self.get_costs(self.round) # shape: (batch, )
                # Reward/Advantage computation
                reward = np.array(costs_repaired) - np.array(self.costs_destroyed)
                reward = - torch.from_numpy(reward).float().to(self.device)
                # shape: (batch, )


            elif self.phase == 'test_batch':
                destroy_repair_duration = time.time() - self.start_time_destroy
                costs = self.get_costs_memory(self.round)
                for i in range(self.env_batch_size):
                    if self.costs[i] < costs[i]:
                        self.solutions[i] = self.solutions_copies[i]
                    else:
                        self.costs[i] = costs[i]
                # update solutions and cost

                # If adaptive search is used, update performance scores
                if self.lns_adaptive_search:
                    delta = (self.mean_cost_before_iteration - np.mean(self.costs)) / destroy_repair_duration
                    if self.performance_EMA[self.selected_id] == np.inf:
                        self.performance_EMA[self.selected_id] = delta
                    self.performance_EMA[self.selected_id] = self.performance_EMA[self.selected_id] * (
                                1 - EMA_ALPHA) + delta * EMA_ALPHA
                    
                reward = torch.tensor(self.costs).float().to(self.device)

            elif self.phase == 'test_single':
                costs = self.get_costs_memory(self.round) # list, (batch, )

                # Calculate the T_max and T_factor values for simulated annealing in the first iteration
                if self.iter == 0:
                    q75, q25 = np.percentile(costs, [75, 25])
                    self.T_max = q75 - q25
                    self.T_factor = -math.log(self.T_max / self.T_min)

                min_costs = min(costs) # float

                # Update incumbent if a new best solution is found
                if min_costs <= self.costs:
                    self.incumbent_solution = deepcopy(self.solutions[np.argmin(costs)])
                    self.costs = min_costs

                # Calculate simulated annealing temperature
                T = self.T_max * math.exp(
                    self.T_factor * (time.time() - self.start_time_reheating) / (self.timelimit_batch / self.lns_reheating_nb))

                # Accept a solution if the acceptance criteria is fulfilled
                if min_costs <= self.cur_cost or np.random.rand() < math.exp(-(min(costs) - self.cur_cost) / T):
                    self.single_solution = self.solutions[np.argmin(costs)]
                    self.cur_cost = min_costs

                reward = torch.full((self.env_batch_size,), self.costs, dtype=torch.float, device=self.device)
                # (batch, )
                
            
        else:
            reward = self.dummy_flag_long

            for i in range(self.env_batch_size):
                if self.origin_idx[i] == 0 and not self.instance_repaired[i]:
                    self.origin_idx[i] = np.random.choice(self.open_nn_input_idx[i], 1).item()

            self.mask = get_mask(self.origin_idx, self.dynamic_input, self.nn_input_idx_to_tour, self.split_delivery, self.vehicle_capacity).to(self.device).float()
        
        origin_idx = torch.from_numpy(self.origin_idx)
        instance_repaired = torch.from_numpy(self.instance_repaired)
        # shape: (batch, )
        
        next_state = {
            'mask' : self.mask,
            'static_input': self.static_input,
            'dynamic_input': self.dynamic_input,
            'origin_idx': origin_idx,
            "reward": reward,
            "done": instance_repaired,
        }

        out = {
            'next' : next_state,
            'vehicle_capacity' : self.capacity,
            'reward' : reward,
            'done' : instance_repaired,
            'prob': prob,
            'solution': self.solutions
        }

        return out


    def get_costs_incomplete(self, round):
        """Return the cost of the current bacth incomplete solutions."""
        cost = []
        for k in range(self.env_batch_size):
            solution = self.solutions[k]
            original_depot_node_xy = self.depot_node_xy[k].cpu().numpy()
            c = 0
            for tour in solution:
                if len(tour) <= 1:
                    continue
                for i in range(0, len(tour) - 1):
                    cc = np.sqrt((original_depot_node_xy[tour[i][0], 0] - original_depot_node_xy[tour[i + 1][0], 0]) ** 2
                                + (original_depot_node_xy[tour[i][0], 1] - original_depot_node_xy[tour[i + 1][0], 1]) ** 2)
                    if round:
                        cc = np.round(cc)
                    c += cc
            cost.append(c)
        return cost


    def get_costs(self, round):
        """Return the costs of the batch complete solution."""
        cost = []
        for j in range(self.env_batch_size):
            solution = self.solutions[j]
            original_depot_node_xy = self.depot_node_xy[j].cpu().numpy()

            """the cost of the current complete solution."""
            c = 0
            for t in solution:
                if t[0][0] != 0 or t[-1][0] != 0:
                    raise Exception("Incomplete solution.")
                for i in range(0, len(t) - 1):
                    cc = np.sqrt((original_depot_node_xy[t[i][0], 0] - original_depot_node_xy[t[i + 1][0], 0]) ** 2
                                + (original_depot_node_xy[t[i][0], 1] - original_depot_node_xy[t[i + 1][0], 1]) ** 2)
                    if round:
                        cc = np.round(cc)
                    c += cc
            cost.append(c)

        return cost

    def get_costs_memory(self, round):
        """Return the costs of the batch complete solution."""
        cost = []
        for j in range(self.env_batch_size):
            solution = self.solutions[j]
            original_depot_node_xy = self.depot_node_xy[j].cpu().numpy()
            costs_memory = self.costs_memory[j]

            """Return the cost of the current complete solution. Uses a memory to improve performance."""
            c = 0
            for t in solution:
                if t[0][0] != 0 or t[-1][0] != 0:
                    raise Exception("Incomplete solution.")
                for i in range(0, len(t) - 1):
                    from_idx = t[i][0]
                    to_idx = t[i + 1][0]
                    if np.isnan(costs_memory[from_idx, to_idx]):
                        cc = np.sqrt((original_depot_node_xy[from_idx, 0] - original_depot_node_xy[to_idx, 0]) ** 2
                                    + (original_depot_node_xy[from_idx, 1] - original_depot_node_xy[to_idx, 1]) ** 2)
                        if round:
                            cc = np.round(cc)
                        costs_memory[from_idx, to_idx] = cc
                        c += cc
                    else:
                        c += costs_memory[from_idx, to_idx]

            cost.append(c)
            self.costs_memory[j] = costs_memory
        return cost


    def get_max_nb_input_points(self):
        incomplete_tours = self.incomplete_tours
        nb = 1  # input point for the depot
        for tour in incomplete_tours:
            if len(tour) == 1:
                nb += 1
            else:
                if tour[0][0] != 0:
                    nb += 1
                if tour[-1][0] != 0:
                    nb += 1
        return nb


    def get_network_input(self, input_size, index):
        """Generate the tensor representation of an incomplete solution (i.e, a representation of the repair problem).
         The input size must be provided so that the representations of all inputs of the batch have the same size.

        [:, 0] x-coordinates for all points
        [:, 1] y-coordinates for all points
        [:, 2] demand values for all points
        [:, 3] state values for all points

        """
        depot_node_xy = self.depot_node_xy[index].cpu().numpy()
        solution = self.solutions[index]
        incomplete_tours = self.incomplete_tours[index]
        capacity = self.vehicle_capacity

        nn_input = np.zeros((input_size, 4))
        nn_input[0, :2] = depot_node_xy[0]  # Depot location
        nn_input[0, 2] = -1 * capacity  # Depot demand
        nn_input[0, 3] = -1  # Depot state
        network_input_idx_to_tour = [None] * input_size
        network_input_idx_to_tour[0] = [solution[0], 0]
        i = 1
        destroyed_location_idx = []

        for tour in incomplete_tours:
            # Create an input for a tour consisting of a single customer
            if len(tour) == 1:
                nn_input[i, :2] = depot_node_xy[tour[0][0]]
                nn_input[i, 2] = tour[0][1]
                nn_input[i, 3] = 1
                tour[0][2] = i
                network_input_idx_to_tour[i] = [tour, 0]
                destroyed_location_idx.append(tour[0][0])
                i += 1
            else:
                # Create an input for the first location in an incomplete tour if the location is not the depot
                if tour[0][0] != 0:
                    nn_input[i, :2] = depot_node_xy[tour[0][0]]
                    nn_input[i, 2] = sum(l[1] for l in tour)
                    network_input_idx_to_tour[i] = [tour, 0]
                    if tour[-1][0] == 0:
                        nn_input[i, 3] = 3
                    else:
                        nn_input[i, 3] = 2
                    tour[0][2] = i
                    destroyed_location_idx.append(tour[0][0])
                    i += 1
                # Create an input for the last location in an incomplete tour if the location is not the depot
                if tour[-1][0] != 0:
                    nn_input[i, :2] = depot_node_xy[tour[-1][0]]
                    nn_input[i, 2] = sum(l[1] for l in tour)
                    network_input_idx_to_tour[i] = [tour, len(tour) - 1]
                    tour[-1][2] = i
                    if tour[0][0] == 0:
                        nn_input[i, 3] = 3
                    else:
                        nn_input[i, 3] = 2
                    destroyed_location_idx.append(tour[-1][0])
                    i += 1


        self.open_nn_input_idx.append(list(range(1, i))) # list
        self.nn_input_idx_to_tour.append(network_input_idx_to_tour) # list
        return nn_input[:, :2], nn_input[:, 2:]

    def _get_network_input_update_for_tour(self, tour, new_demand, index):
        """Returns an nn_input update for the tour tour. The demand of the tour is updated to new_demand"""
        nn_input_idx_start = tour[0][2]  # Idx of the nn_input for the first location in tour
        nn_input_idx_end = tour[-1][2]  # Idx of the nn_input for the last location in tour

        # If the tour stars and ends at the depot, no update is required
        if nn_input_idx_start == 0 and nn_input_idx_end == 0:
            return []

        nn_input_update = []
        # Tour with a single location
        if len(tour) == 1:
            if tour[0][0] != 0:
                nn_input_update.append([nn_input_idx_end, new_demand, 1])
                self.nn_input_idx_to_tour[index][nn_input_idx_end] = [tour, 0]
        else:
            # Tour contains the depot
            if tour[0][0] == 0 or tour[-1][0] == 0:
                # First location in the tour is not the depot
                if tour[0][0] != 0:
                    nn_input_update.append([nn_input_idx_start, new_demand, 3])
                    # update first location
                    self.nn_input_idx_to_tour[index][nn_input_idx_start] = [tour, 0]
                # Last location in the tour is not the depot
                elif tour[-1][0] != 0:
                    nn_input_update.append([nn_input_idx_end, new_demand, 3])
                    # update last location
                    self.nn_input_idx_to_tour[index][nn_input_idx_end] = [tour, len(tour) - 1]
            # Tour does not contain the depot
            else:
                # update first and last location of the tour
                nn_input_update.append([nn_input_idx_start, new_demand, 2])
                self.nn_input_idx_to_tour[index][nn_input_idx_start] = [tour, 0]
                nn_input_update.append([nn_input_idx_end, new_demand, 2])
                self.nn_input_idx_to_tour[index][nn_input_idx_end] = [tour, len(tour) - 1]
        return nn_input_update

    def do_action(self, id_from, id_to, index):
        """Performs an action. The tour end represented by input with the id id_from is connected to the tour end
         presented by the input with id id_to."""

        tour_from = self.nn_input_idx_to_tour[index][id_from][0]  # Tour that should be connected
        tour_to = self.nn_input_idx_to_tour[index][id_to][0]  # to this tour.
        pos_from = self.nn_input_idx_to_tour[index][id_from][1]  # Position of the location that should be connected in tour_from
        pos_to = self.nn_input_idx_to_tour[index][id_to][1]  # Position of the location that should be connected in tour_to

        nn_input_update = []  # Instead of recalculating the tensor representation, we only compute an update description.
        # This improves performance.

        # Exchange tour_from with tour_to or invert order of the tours. This reduces the number of cases that need
        # to be considered in the following.
        if len(tour_from) > 1 and len(tour_to) > 1:
            if pos_from > 0 and pos_to > 0:
                tour_to.reverse()
            elif pos_from == 0 and pos_to == 0:
                tour_from.reverse()
            elif pos_from == 0 and pos_to > 0:
                tour_from, tour_to = tour_to, tour_from
        elif len(tour_to) > 1:
            if pos_to == 0:
                tour_to.reverse()
            tour_from, tour_to = tour_to, tour_from
        elif len(tour_from) > 1 and pos_from == 0:
            tour_from.reverse()

        # Now we only need to consider two cases 1) Connecting an incomplete tour with more than one location
        # to an incomplete tour with more than one location 2) Connecting an incomplete tour (single
        # or multiple locations) to incomplete tour consisting of a single location

        # Case 1
        if len(tour_from) > 1 and len(tour_to) > 1:
            combined_demand = sum(l[1] for l in tour_from) + sum(l[1] for l in tour_to)
            assert combined_demand <= self.vehicle_capacity  # This is ensured by the masking schema

            # The two incomplete tours are combined to one (in)complete tour. All network inputs associated with the
            # two connected tour ends are set to 0
            nn_input_update.append([tour_from[-1][2], 0, 0])
            nn_input_update.append([tour_to[0][2], 0, 0])
            tour_from.extend(tour_to)
            self.solutions[index].remove(tour_to)
            nn_input_update.extend(self._get_network_input_update_for_tour(tour_from, combined_demand, index))

        # Case 2
        if len(tour_to) == 1:
            demand_from = sum(l[1] for l in tour_from)
            combined_demand = demand_from + sum(l[1] for l in tour_to)
            unfulfilled_demand = combined_demand - self.vehicle_capacity

            # The new tour has a total demand that is smaller than or equal to the vehicle capacity
            if unfulfilled_demand <= 0:
                if len(tour_from) > 1:
                    nn_input_update.append([tour_from[-1][2], 0, 0])
                # Update solution
                tour_from.extend(tour_to)
                self.solutions[index].remove(tour_to)
                # Generate input update
                nn_input_update.extend(self._get_network_input_update_for_tour(tour_from, combined_demand, index))
            # The new tour has a total demand that is larger than the vehicle capacity
            else:
                nn_input_update.append([tour_from[-1][2], 0, 0])
                if len(tour_from) > 1 and tour_from[0][0] != 0:
                    nn_input_update.append([tour_from[0][2], 0, 0])

                # Update solution
                tour_from.append([tour_to[0][0], tour_to[0][1], tour_to[0][2]])  # deepcopy of tour_to
                tour_from[-1][1] = self.vehicle_capacity - demand_from
                tour_from.append([0, 0, 0])
                if tour_from[0][0] != 0:
                    tour_from.insert(0, [0, 0, 0])
                tour_to[0][1] = unfulfilled_demand  # Update demand of tour_to

                nn_input_update.extend(self._get_network_input_update_for_tour(tour_to, unfulfilled_demand, index))

        # Add depot tour to the solution tours if it was removed
        if self.solutions[index][0] != [[0, 0, 0]]:
            self.solutions[index].insert(0, [[0, 0, 0]])
            self.nn_input_idx_to_tour[index][0] = [self.solutions[index][0], 0]

        for update in nn_input_update:
            if update[2] == 0 and update[0] != 0:
                self.open_nn_input_idx[index].remove(update[0])

        return nn_input_update, tour_from[-1][2]

    def verify_solution(self, config):
        """Verify that a feasible solution has been found."""
        d = np.zeros((self.problem_size + 1), dtype=int)
        for i in range(len(self.solutions)):
            for ii in range(len(self.solutions[i])):
                d[self.solutions[i][ii][0]] += self.solutions[i][ii][1]
        if (self.demand != d).any():
            raise Exception('Solution could not be verified.')

        for tour in self.solutions:
            if sum([t[1] for t in tour]) > self.vehicle_capacity:
                raise Exception('Solution could not be verified.')

        if not config.split_delivery:
            customers = []
            for tour in self.solutions:
                for c in tour:
                    if c[0] != 0:
                        customers.append(c[0])

            if len(customers) > len(set(customers)):
                raise Exception('Solution could not be verified.')

    def get_solution_copy(self):
        """ Returns a copy of self.solutions"""
        solutions_copy = []
        for solution in self.solutions:
            solution_copy = []
            for tour in solution:
                solution_copy.append([x[:] for x in tour]) # Fastest way to make a deep copy
            solutions_copy.append(solution_copy)

        return solutions_copy
    

def get_mask(origin_nn_input_idx, dynamic_input, nn_input_idx_to_tour, split_delivery, capacity):
    """ Returns a mask for the current nn_input"""
    batch_size = origin_nn_input_idx.shape[0]

    # Start with all used input positions
    mask = (dynamic_input[:, :, 1] != 0).cpu().long().numpy()

    for i in range(batch_size):
        idx_from = origin_nn_input_idx[i]
        origin_tour = nn_input_idx_to_tour[i][idx_from][0]
        origin_pos = nn_input_idx_to_tour[i][idx_from][1]

        # Find the start of the tour in the nn input
        # e.g. for the tour [2, 3] two entries in nn input exists
        if origin_pos == 0:
            idx_same_tour = origin_tour[-1][2]
        else:
            idx_same_tour = origin_tour[0][2]

        mask[i, idx_same_tour] = 0

        # Do not allow origin location = destination location
        mask[i, idx_from] = 0

    mask = torch.from_numpy(mask)

    origin_tour_demands = dynamic_input[torch.arange(batch_size), origin_nn_input_idx, 0]
    combined_demand = origin_tour_demands.unsqueeze(1).expand(batch_size, dynamic_input.shape[1]) + dynamic_input[:, :,
                                                                                                    0]

    if split_delivery:
        multiple_customer_tour = (dynamic_input[torch.arange(batch_size), origin_nn_input_idx, 1] > 1).unsqueeze(1).expand(
            batch_size, dynamic_input.shape[1])

        # If the origin tour consists of multiple customers mask all tours with multiple customers where
        # the combined demand is > 1
        mask[multiple_customer_tour & (combined_demand > capacity) & (dynamic_input[:, :, 1] > 1)] = 0

        # If the origin tour consists of a single customer mask all tours with demand is >= 1
        mask[(~multiple_customer_tour) & (dynamic_input[:, :, 0] >= capacity)] = 0
    else:
        mask[combined_demand > capacity] = 0

    mask[:, 0] = 1  # Always allow to go to the depot

    return mask


def get_max_nb_input_points(incomplete_tours):
    nb = 1  # input point for the depot
    for tour in incomplete_tours:
        if len(tour) == 1:
            nb += 1
        else:
            if tour[0][0] != 0:
                nb += 1
            if tour[-1][0] != 0:
                nb += 1
    return nb
   



class DestroyOperator():
    def __init__(self, solutions, problem, destroy_procedure=None, destruction_p=None):
        self.solutions = solutions
        self.locations = problem[:, :, :2]
        self.demands = problem[:, :, 2]
        self.batch_size = problem.size(0)
        self.nb_customers = problem.size(1) - 1
        self.destroy_procedure = destroy_procedure
        self.destruction_p = destruction_p


    def destroy_instances(self):
        solution_batch = []
        incomplete_tours_batch =[]

        for i in range(self.batch_size):
            self.solution = self.solutions[i] # list
            self.location = self.locations[i].cpu().numpy() # tensor, shape: (1 + problem, 2)
            self.demand = self.demands[i]

            if self.destroy_procedure == "R":
                self.destroy_random(self.destruction_p)
            elif self.destroy_procedure == "P":
                self.destroy_point_based(self.destruction_p)
            elif self.destroy_procedure == "T":
                self.destroy_tour_based(self.destruction_p)

            solution_batch.append(self.solution) # list
            incomplete_tours_batch.append(self.incomplete_tours) # list

        return solution_batch, incomplete_tours_batch


    def destroy_random(self, p):
        """Random destroy. Select customers that should be removed at random and remove them from tours."""
        customers_to_remove_idx = np.random.choice(range(1, self.nb_customers + 1), int(self.nb_customers * p),
                                                    replace=False)
        self.destroy(customers_to_remove_idx)

    def destroy_point_based(self, p):
        """Point based destroy. Select customers that should be removed based on their distance to a random point
            and remove them from tours."""
        nb_customers_to_remove = int(self.nb_customers * p)
        random_point = np.random.rand(1, 2)
        dist = np.sum((self.location[1:] - random_point) ** 2, axis=1)
        closest_customers_idx = np.argsort(dist)[:nb_customers_to_remove] + 1
        self.destroy(closest_customers_idx)


    def destroy(self, customers_to_remove_idx):
        """Remove the customers with the given idx from their tours. This creates an incomplete solution."""
        self.incomplete_tours = []
        st = []  # solution tours

        removed_customer_idx = []

        for tour in self.solution:
            last_split_idx = 0
            for i in range(1, len(tour) - 1):
                if tour[i][0] in customers_to_remove_idx:
                    # Create two new tours:
                    # The first consisting of the tour from the depot or from the last removed customer to the
                    # customer that should be removed
                    if i > last_split_idx and i > 1:
                        new_tour_pre = tour[last_split_idx:i]
                        st.append(new_tour_pre)
                        self.incomplete_tours.append(new_tour_pre)

                    # The second consisting of only the customer to be removed
                    customer_idx = tour[i][0]
                    if customer_idx not in removed_customer_idx:  # make sure the customer has not already been
                        # extracted from a different tour
                        demand = self.demand[customer_idx].item()
                        new_tour = [[customer_idx, demand, None]]
                        st.append(new_tour)
                        self.incomplete_tours.append(new_tour)
                        removed_customer_idx.append(customer_idx)
                    last_split_idx = i + 1

            if last_split_idx > 0:
                # Create another new tour consisting of the remaining part of the original tour
                if last_split_idx < len(tour) - 1:
                    new_tour_post = tour[last_split_idx:]
                    st.append(new_tour_post)
                    self.incomplete_tours.append(new_tour_post)
            else:  # add unchanged tour
                st.append(tour)
                # st代表断过的路程

        self.solution = st

    def destroy_tour_based(self, p):
        """Tour based destroy. Remove all tours closest to a randomly selected point from a solution."""

        # Make a dictionary that maps customers to tours
        customer_to_tour = {}
        for i, tour in enumerate(self.solution[1:]):
            for e in tour[1:-1]:
                if e[0] in customer_to_tour:
                    customer_to_tour[e[0]].append(i + 1)
                else:
                    customer_to_tour[e[0]] = [i + 1]

        nb_customers_to_remove = int(self.nb_customers * p)  # Number of customer that should be removed
        nb_removed_customers = 0
        tours_to_remove_idx = []
        random_point = np.random.rand(1, 2)  # Randomly selected point
        dist = np.sum((self.location[1:] - random_point) ** 2, axis=1)
        closest_customers_idx = np.argsort(dist) + 1

        # Iterate over customers starting with the customer closest to the random point.
        for customer_idx in closest_customers_idx:
            # Iterate over the tours of the customer
            for i in customer_to_tour[customer_idx]:
                # and if the tour is not yet marked for removal
                if i not in tours_to_remove_idx:
                    # mark it for removal
                    tours_to_remove_idx.append(i)
                    nb_removed_customers += len(self.solution[i])

            # Stop once enough tours are marked for removal
            if nb_removed_customers >= nb_customers_to_remove and len(tours_to_remove_idx) > 1:
                break

        # Create the new tours that all consist of only a single customer
        new_tours = []
        removed_customer_idx = []
        for i in tours_to_remove_idx:
            tour = self.solution[i]
            for e in tour[1:-1]:
                if e[0] in removed_customer_idx:
                    for new_tour in new_tours:
                        if new_tour[0][0] == e[0]:
                            new_tour[0][1] += e[1]
                            break
                else:
                    new_tours.append([e])
                    removed_customer_idx.append(e[0])

        # Remove the tours that are marked for removal from the solution
        for index in sorted(tours_to_remove_idx, reverse=True):
            del self.solution[index]

        self.solution.extend(new_tours)  # Add new tours to solution
        self.incomplete_tours = new_tours
