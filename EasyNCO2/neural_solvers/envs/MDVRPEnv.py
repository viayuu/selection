import torch
from typing import Optional
import random
from torchrl.envs import EnvBase
from tensordict import TensorDict

from EasyNCO.utils.utils import getLogger
from EasyNCO.data.data_utils import augment_pomo


logger = getLogger(__name__)


class MDVRPEnv(EnvBase):
    def __init__(
        self,
        problem_size: int,
        pomo_size: int = 60,
        device: str = "cpu",
        seed: int = 2024,
        agent_min: int = 3,
        agent_max: int = 3,
        depot_min: int = 5,
        depot_max: int = 5,
        aug_factor: int = 1,
        env_name: str = "mdvrp",
        **kwargs,
    ):
        super().__init__(device=device)
        self.env_name = env_name
        self.problem_size = problem_size
        self.pomo_size = pomo_size
        self.agent_min = agent_min
        self.agent_max = agent_max
        self.depot_min = depot_min
        self.depot_max = depot_max
        self.agent_num = None
        self.device = device

        self.saved_node_xy = None
        self.saved_index = None
        self.FLAG__use_saved_problems = False

        self.problems = None
        self.selected_count = None
        self.current_node = None
        self.dummy_flag_bool = None
        self.dummy_flag_long = None

        self.seed_value = seed
        self.aug_type = kwargs.get("aug_type", None)
        self.aug_factor = aug_factor
        self.mix_prop = kwargs.get("mix_prop")
        self.aug_flag = (
            self.aug_type is not None
        )  # Used in baseline to prevent data from augment.

        self._set_seed(seed=self.seed_value)

    def _set_seed(self, seed: Optional[int]):
        rng = torch.manual_seed(seed)
        self.rng = rng

    def use_saved_problems_test(self, filename):
        self.FLAG__use_saved_problems = True

        loaded_dict = torch.load(filename, map_location=self.device)
        self.saved_node_xy = loaded_dict["node_xy"]
        self.saved_index = 0

    def __getstate__(self):
        """Return the state of the environment. By default, we want to avoid pickling
        the random number generator directly as it is not allowed by `deepcopy`
        """
        state = self.__dict__.copy()
        state["rng"] = state["rng"].get_state()
        return state

    def __setstate__(self, state):
        """Set the state of the environment. By default, we want to avoid pickling
        the random number generator directly as it is not allowed by `deepcopy`
        """
        self.__dict__.update(state)
        self.rng = torch.manual_seed(self.seed_value)
        self.rng.set_state(state["rng"])

    def load_problems(self, dataset, batch_size: int = 64, *kwargs):
        self.problems = dataset
        # problems.shape: (batch, problem, 2)
        self.raw_problems = dataset
        self.env_batch_size = batch_size
        self.batch_size = torch.Size([batch_size, self.pomo_size])

        if self.aug_flag is True:
            self.batch_size = torch.Size(
                [self.env_batch_size * self.aug_factor, self.pomo_size]
            )
            self.env_batch_size = self.env_batch_size * self.aug_factor
            aug_operator = {
                "pomo_aug": augment_pomo,
            }
            if self.aug_type not in aug_operator:
                raise NotImplementedError(
                    f"Augment type {self.aug_type} is not supported."
                )
            self.problems = aug_operator[self.aug_type](
                problems=self.problems,
                aug_factor=self.aug_factor,
                mix_prop=self.mix_prop,
            )
            if self.aug_type != "pomo_aug":
                self.problems[: self.batch_size[0] // self.aug_factor] = (
                    self.raw_problems
                )

        assert self.env_batch_size == self.problems.size(0), (
            "batch_size and the first dimension of problems should be the same. "
            + f"Expected batch_size: {self.problems.size(0)}, got: {self.env_batch_size}"
        )

    def _reset(self, td: TensorDict, batch_size=None) -> TensorDict:
        self.output_spec = None  # Solve the inconsistent batch size
        self.agent_num = random.sample(range(self.agent_min, self.agent_max + 1), 1)[0]
        self.depot_num = random.sample(range(self.depot_min, self.depot_max + 1), 1)[0]
        self.loc = self.problems
        # for done
        self.dummy_flag_bool = torch.zeros(
            (self.env_batch_size, self.pomo_size), dtype=torch.bool
        )
        # for action, -1 represents the dummy action, which is used to select the first node
        self.dummy_flag_long = (
            torch.zeros(
                (self.env_batch_size, self.pomo_size),
                dtype=torch.long,
            )
            - 1
        )
        self.selected_count = torch.zeros(
            (self.env_batch_size, self.pomo_size), dtype=torch.long
        )
        return TensorDict(
            {
                "locs": self.loc,
                "agent_num": torch.tensor([self.agent_num] * self.env_batch_size),
                "depot_num": torch.tensor([self.depot_num] * self.env_batch_size),
            },
            batch_size=torch.Size([self.env_batch_size]),
        )

    def pre_step(self) -> TensorDict:
        self.agent_per = torch.arange(self.agent_num, device=self.device)[
            None, :
        ].expand(
            self.pomo_size, -1
        )  # pomo_size就是k，现在在做k个agent的排列
        self._set_seed(seed=self.seed_value)
        if self.pomo_size > 1:
            for i in range(100):
                a = torch.randint(0, self.agent_num, (self.pomo_size,))
                b = torch.randint(0, self.agent_num, (self.pomo_size,))
                p = self.agent_per[torch.arange(self.pomo_size), a].clone()
                q = self.agent_per[torch.arange(self.pomo_size), b].clone()
                self.agent_per = self.agent_per.scatter(
                    dim=1, index=b[:, None], src=p[:, None]
                )
                self.agent_per = self.agent_per.scatter(
                    dim=1, index=a[:, None], src=q[:, None]
                )
            self.agent_per[0] = torch.arange(self.agent_num)
        left_city = self.problem_size - self.depot_num
        batch_size = self.loc.size(0)
        loc = self.loc[:, None, :, :].expand(-1, self.pomo_size, -1, -1)
        prev_a = self.agent_per[None, :, 0:1].expand(batch_size, -1, -1)
        depot_distance = torch.cdist(loc, loc, p=2)
        depot_distance = depot_distance[:, :, : self.depot_num, :].min(dim=-2)[0]
        max_distance = depot_distance.max(dim=-1, keepdim=True)[0]
        current = loc.gather(2, prev_a[..., None].expand(-1, -1, -1, 2))
        cur_dist = torch.cdist(self.loc, self.loc, p=2)  # 计算两个矩阵之间的欧式距离
        max_dist = cur_dist.max(-1)[0].unsqueeze(-1)  # shape: (batch, pomo, 1)
        cur_dist_out = cur_dist / max_dist  # shape: (batch, pomo, problem)
        dist_out = -1 * cur_dist_out
        next_state = {
            "selected_count": self.selected_count,
        }
        out = TensorDict(
            {
                "action": self.dummy_flag_long,
                "next": next_state,
                "reward": self.dummy_flag_long,
                "done": self.dummy_flag_bool,
                "agent_idx": self.agent_per[None, :, 0:1].expand(batch_size, -1, -1),
                "agent_per": self.agent_per[None, :, :].expand(batch_size, -1, -1),
                "loc": loc,
                "to_assign": torch.ones(
                    batch_size, self.pomo_size, dtype=torch.bool, device=loc.device
                ),
                "dist": (loc[:, :, None, :] - loc[:, None, :, :]).norm(p=2, dim=-1),
                "prev_a": prev_a,
                # Keep visited with depot so we can scatter efficiently (if there is an action for depot)
                "visited_": torch.zeros(
                    batch_size,
                    self.pomo_size,
                    left_city + self.depot_num,
                    dtype=torch.uint8,
                    device=loc.device,
                ),
                "assign": torch.zeros(
                    batch_size,
                    self.pomo_size,
                    self.agent_num,
                    dtype=torch.int64,
                    device=loc.device,
                ),
                "lengths": torch.zeros(
                    batch_size, self.pomo_size, self.agent_num, device=loc.device
                ),
                "cur_coord": current,
                "count_depot": torch.zeros(
                    batch_size, self.pomo_size, 1, dtype=torch.int64, device=loc.device
                ),
                "left_city": left_city
                * torch.ones(
                    batch_size, self.pomo_size, 1, dtype=torch.long, device=loc.device
                ),
                "remain_max_distance": max_distance,
                "max_distance": max_distance,
                "depot_distance": depot_distance,
                "dist_out": dist_out[:, None, :, :].expand(-1, self.pomo_size, -1, -1),
            },
            batch_size=self.batch_size,
        )
        return out

    def _step(self, td: TensorDict) -> TensorDict:
        td = td.clone()
        self.selected_count += 1
        self.current_node = td["action"]
        # shape: (batch, pomo)

        # Update the state

        prev_a = self.current_node[:, :, None]  # Add dimension for step
        to_assign = ~td["to_assign"] & (self.current_node < self.depot_num)
        # City idices starts from agent_num + 1
        is_city = self.current_node >= self.depot_num

        td["left_city"][is_city] -= 1
        # If agent move to other city, then, the distance between visited city and depot is 0
        depot_distance = td["depot_distance"].scatter(-1, prev_a, 0)
        remain_max_distance = td["depot_distance"].max(dim=-1, keepdim=True)[0]

        cur_coord = td["loc"].gather(2, prev_a[..., None].expand(-1, -1, -1, 2))

        path_lengths = (cur_coord - td["cur_coord"]).norm(p=2, dim=-1)  # (batch_dim, 1)
        path_lengths[:, :, 0][td["to_assign"]] = 0
        assign = td["assign"]
        assign[td["to_assign"], :] = td["assign"][td["to_assign"], :].scatter(
            -1, td["count_depot"][td["to_assign"], :], prev_a[td["to_assign"], :]
        )
        lengths = td["lengths"].scatter_add(-1, td["count_depot"], path_lengths)
        if self.env_name == "mdvrp":
            index = (
                prev_a == td["assign"].gather(2, td["count_depot"])
            ) & to_assign.unsqueeze(-1)
        elif self.env_name == "fmdvrp":
            index = (prev_a < self.depot_num) & to_assign.unsqueeze(-1)
        # Current agent comes back to depot when it selects its own index
        td["count_depot"][index] += torch.ones(
            td["count_depot"][index].shape,
            dtype=torch.int64,
            device=td["count_depot"].device,
        )
        agent_idx = td["agent_idx"]
        # agent_idx is added by 1 if the current agent comes back to depot
        if td["visited_"].dtype == torch.uint8:
            # Add one dimension since we write a single value
            visited_ = td["visited_"].scatter(-1, prev_a, 1)
        else:
            visited_ = self.mask_long_scatter(td["visited_"], prev_a)
        if ~(
            (is_city == False).all()
            and (td["count_depot"] == self.agent_per.size(1)).all()
        ):
            agent_idx = (
                td["agent_per"][:1, :, :]
                .expand(td["count_depot"].size(0), -1, -1)
                .gather(2, td["count_depot"])
            )
        done = td["count_depot"] == lengths.size(2)
        done_all = done.all()
        if done_all:
            reward = -lengths.max(-1)[0]
        else:
            reward = self.dummy_flag_long
        next_state = {
            "selected_count": self.selected_count,
            "reward": reward,
            "done": done,
        }
        td.update(
            {
                "action": self.current_node,
                "next": next_state,
                "reward": reward,
                "done": done,
                "prev_a": prev_a,
                "visited_": visited_,
                "lengths": lengths,
                "cur_coord": cur_coord,
                "agent_idx": agent_idx,
                "depot_distance": depot_distance,
                "to_assign": to_assign,
                "assign": assign,
                "remain_max_distance": remain_max_distance,
            }
        )
        return td

    def mask_long_scatter(mask, values, check_unset=True):
        """
        Sets values in mask in dimension -1 with arbitrary batch dimensions
        If values contains -1, nothing is set
        Note: does not work for setting multiple values at once (like normal scatter)
        """
        assert mask.size()[:-1] == values.size()
        rng = torch.arange(mask.size(-1), out=mask.new())
        values_ = values[..., None]  # Need to broadcast up do mask dim
        # This indicates in which value of the mask a bit should be set
        where = (values_ >= (rng * 64)) & (values_ < ((rng + 1) * 64))
        # Optional: check that bit is not already set
        assert not (
            check_unset and ((mask & (where.long() << (values_ % 64))) > 0).any()
        )
        # Set bit by shifting a 1 to the correct position
        # (% not strictly necessary as bitshift is cyclic)
        # since where is 0 if no value needs to be set, the bitshift has no effect
        return mask | (where.long() << (values_ % 64))
