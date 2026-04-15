import torch
from typing import Optional

from torch.utils.data import DataLoader
from torchrl.envs import EnvBase
from tensordict import TensorDict

from EasyNCO.utils.utils import getLogger
from EasyNCO.data import PCTSPGenerator

logger = getLogger(__name__)

class PCTSPEnv(EnvBase):
    def __init__(self,
                 problem_size : int,
                 pomo_size : int = 1,
                 device : str = 'cpu',
                 seed : int = 2024,
                 **kwargs):
        super().__init__(device = device)

        self.env_name = 'pctsp'
        self.problem_size = problem_size
        self.pomo_size = pomo_size
        self.device = device

        self.problems = None
        self.FLAG__use_saved_problems = False
        self.aug_type=None
        self.aug_factor=1
        # Dynamic
        self.selected_count = None
        self.current_node = None
        self.ninf_mask = None
        self.selected_node_list = None
        self.first_node = None
        self.dummy_flag_bool = None
        self.dummy_flag_long = None

        self.load = None


        self.seed_value = seed
        self._set_seed(seed=seed)

    def _set_seed(self, seed: Optional[int]):
        rng = torch.manual_seed(seed)
        self.rng = rng

    def use_saved_problems_test(self,filename):
        pass

    def generate_eval_instances(self, num_instances: int, batch_size: int) -> DataLoader:
        '''
        Note that this function is necessary for the AM-based models.
        If you are using the AM-based models, you should implement this function according to specific CO problems.
        The function generates the evaluation instances for the CO problem.
        It is used to evaluate the trained model using greedyRollout baseline in AM-based models.
        :param num_instances:
        :param batch_size:
        :return: DataLoader
        '''
        eval_dataset = PCTSPGenerator(num_instances, self.problem_size, batch_size, self.device)
        return eval_dataset



    def load_problems(self, dataset, batch_size: int = 64 ):
        self.problems = dataset
        # problems.shape: (batch, problem, 4)  (x,y,prize,beta)
        self.env_batch_size = batch_size
        self.batch_size = torch.Size([batch_size, self.pomo_size])

        # if aug_factor > 1:
        #     if aug_factor == 8:
        #         self.batch_size = torch.Size([self.env_batch_size * aug_factor, self.pomo_size])
        #         self.env_batch_size = self.env_batch_size * aug_factor
        #         self.problems = augment_xy_data_by_8_fold(self.problems)
        #         # shape: (8*batch, problem, 2)
        #     else:
        #         raise NotImplementedError(f'aug_factor={aug_factor} is not supported, expected values are 1 or 8.')

        assert self.env_batch_size == self.problems.size(0), \
            ('batch_size and the first dimension of problems should be the same. ' +
             f'Expected batch_size: {self.problems.size(0)}, got: {self.env_batch_size}')

    def _reset(self,td: TensorDict,batch_size=None) -> TensorDict:
        self.output_spec = None  # Solve the inconsistent batch size
        self.selected_count = torch.zeros((self.env_batch_size, self.pomo_size), dtype=torch.long)
        self.current_node = None
        # shape: (batch, pomo)
        self.selected_node_list = torch.zeros((self.env_batch_size, self.pomo_size, 0), dtype=torch.long)
        # shape: (batch, pomo, 0~problem)
        self.ninf_mask = torch.zeros((self.env_batch_size, self.pomo_size, self.problem_size))
        # for done
        self.dummy_flag_bool = torch.zeros((self.env_batch_size, self.pomo_size), dtype=torch.bool)
        # for action, -1 represents the dummy action, which is used to select the first node
        self.dummy_flag_long = torch.zeros((self.env_batch_size, self.pomo_size), dtype=torch.long) - 1
        #collect prize
        self.load = torch.zeros((self.env_batch_size, self.pomo_size), dtype=torch.float)

        return TensorDict({
            'action': self.dummy_flag_long,  # 'action': 'int',  # shape: (batch, pomo)
            'locs': self.problems,
            'load' : self.load,
            'ninf_mask': self.ninf_mask,
            "selected_count": self.selected_count,
            "selected_node_list": self.selected_node_list,
            "reward": self.dummy_flag_long,
            "done": self.dummy_flag_bool,
        }, batch_size=torch.Size([self.env_batch_size])
        )



    def pre_step(self) -> TensorDict:

        next_state = {
            'ninf_mask': self.ninf_mask,
            'selected_count': self.selected_count,
            'selected_node_list': self.selected_node_list,
        }
        out = TensorDict({
            'action' : self.dummy_flag_long,
            'first_node' : self.dummy_flag_long,
            'load' : self.load,
            'next' : next_state,
            'reward' : self.dummy_flag_long,
            'done' : self.dummy_flag_bool,
        }, batch_size = self.batch_size
        )
        return out

    def _step(self,td : TensorDict) -> TensorDict:
        self.selected_count += 1
        self.current_node = td['action'] #(batch,pomo)
        selected = self.current_node
        # shape: (batch, pomo)
        self.selected_node_list = torch.cat((self.selected_node_list, self.current_node[:, :, None]), dim=2)

        self.ninf_mask.scatter_(dim=-1, index=self.current_node.unsqueeze(-1), value=float('-inf'))
        self.load += self.problems[:, :, 2][:, None, :].expand(-1, self.pomo_size, -1).gather(-1, selected[:, :,None]).squeeze()
        done = (self.load + 1e-5 > 1.) | (self.selected_count == self.problem_size)
        done_all = done.all()

        #batch,pomo,problems
        if done_all:
            # judge whether solution is valid.
            temp = self.ninf_mask.gather(-1, self.selected_node_list)
            assert (temp == float('-inf')).all(), \
                'The selected nodes should be masked with -inf. It means that the certain solutions are not valid.'

            reward = -self._get_travel_distance()  # note the minus sign!
        else:
            reward = self.dummy_flag_long

        next_state = {
            'ninf_mask': self.ninf_mask,
            "selected_count": self.selected_count,
            "selected_node_list": self.selected_node_list,
            "reward": reward,
            "done": done,
        }
        out = TensorDict({
            'action': self.current_node,  # 'action': 'int',  # shape: (batch, pomo)
            #"first_node": self.first_node,  # "first_node": "int",  # shape: (batch, pomo)
            "load" : self.load,
            "next": next_state,
            "reward": reward,
            "done": done,
        }, batch_size=self.batch_size
        )

        return out







    def _get_travel_distance(self):
        solution = self.selected_node_list.clone()
        return_to_depot = torch.zeros(self.batch_size,self.pomo_size)
        solution = torch.cat((solution, return_to_depot[:, :, None]), dim=2)

        visited = torch.ones((solution.size(0), solution.size(1), self.problems.size(-2)))
        visited = visited.scatter(-1, solution, 0)
        penalty = (visited * self.problems[:, :, -1][:, None, :].expand(-1, solution.size(1), -1)).sum(-1)
        batch_size = solution.size(0)
        pomo_size = solution.size(1)
        gathering_index = solution.unsqueeze(3).expand(batch_size, pomo_size, -1, 2)
        # shape: (batch, pomo, problem, 2)
        seq_expanded = self.problems[:, None, :, :2].expand(batch_size, pomo_size, -1, 2)

        ordered_seq = seq_expanded.gather(dim=2, index=gathering_index)
        # shape: (batch, pomo, problem, 2)

        rolled_seq = ordered_seq.roll(dims=2, shifts=-1)
        segment_lengths = ((ordered_seq - rolled_seq) ** 2).sum(3).sqrt()
        # shape: (batch, pomo, problem)
        travel_distances = segment_lengths[:, :, :-1].sum(2)
        # shape: (batch, pomo)
        return travel_distances + penalty


    def __getstate__(self):
        """
        Return the state of the environment. By default, we want to avoid pickling
        the random number generator directly as it is not allowed by `deepcopy`
        """
        state = self.__dict__.copy()
        state['rng'] = state['rng'].get_state()

        return state

    def __setstate__(self, state):
        """
        Set the state of the environment. By default, we want to avoid pickling
        the random number generator directly as it is not allowed by `deepcopy`
        """
        self.__dict__.update(state)
        self.rng = torch.manual_seed(self.seed_value)
        self.rng.set_state(state['rng'])