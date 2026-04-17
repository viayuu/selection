import torch
from typing import Optional
import random
from torchrl.envs import EnvBase
from tensordict import TensorDict

from EasyNCO.utils.utils import getLogger
from EasyNCO.data.data_utils import augment_pomo


logger = getLogger(__name__)


class MPDPEnv(EnvBase):

    def __init__(
        self,
        problem_size: int,
        pomo_size: int = 60,
        device: str = "cpu",
        seed: int = 2024,
        agent_min: int = 3,
        agent_max: int = 3,
        aug_factor: int = 1,
        **kwargs,
    ):
        super().__init__(device=device)
        self.env_name = "min-max mpdp"
        self.problem_size = problem_size + 1  # +1 for depot
        assert self.problem_size % 2 == 1, "The problem size should be odd."
        self.pomo_size = pomo_size
        self.agent_min = agent_min
        self.agent_max = agent_max
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

        self._set_seed(seed=seed)

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
        self.loc = torch.cat(
            (
                self.problems[:, :1, :].repeat(1, self.agent_num, 1),
                self.problems[:, 1:, :],
            ),
            dim=1,
        )
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
        depot = self.loc[:, : self.agent_num, :]
        loc = self.loc[:, self.agent_num :, :]
        left_request = loc.size(1) // 2
        whole_instance = self.loc
        # Distance from all nodes between each other
        distance = torch.cdist(whole_instance, whole_instance, p=2)
        # Distance paires between pickup and delivery nodes
        pickup_delivery_distance = distance[
            :, self.agent_num : self.agent_num + left_request, self.agent_num :
        ]
        index = torch.arange(left_request, 2 * left_request, device=depot.device)[
            None, :, None
        ]
        index = index.repeat(pickup_delivery_distance.shape[0], 1, 1)
        add_pd_distance = pickup_delivery_distance.gather(-1, index)
        add_pd_distance = add_pd_distance.squeeze(-1)
        remain_pickup_max_distance = distance[
            :, 0, : self.agent_num + left_request
        ].max(dim=-1, keepdim=True)[0]
        remain_delivery_max_distance = distance[
            :, 0, self.agent_num + left_request :
        ].max(dim=-1, keepdim=True)[0]
        remain_sum_paired_distance = add_pd_distance.sum(dim=-1, keepdim=True)
        # Distance from depot to all nodes
        # Delivery nodes should consider the sum of distance from depot to paired pickup nodes and pickup nodes to delivery nodes
        distance[:, 0, self.agent_num : self.agent_num + left_request] = (
            distance[:, 0, self.agent_num : self.agent_num + left_request]
            + distance[:, 0, self.agent_num + left_request :]
        )
        # Distance from depot to all nodes
        depot_distance = distance[:, 0, :]
        depot_distance[:, self.agent_num : self.agent_num + left_request] = (
            depot_distance[:, self.agent_num : self.agent_num + left_request]
        )  # + add_pd_distance
        batch_size, n_loc, _ = loc.size()
        to_delivery = torch.cat(
            [
                torch.ones(
                    batch_size,
                    1,
                    n_loc // 2 + self.agent_num,
                    dtype=torch.uint8,
                    device=loc.device,
                ),
                torch.zeros(
                    batch_size, 1, n_loc // 2, dtype=torch.uint8, device=loc.device
                ),
            ],
            dim=-1,
        )  # [batch_size, 1, graph_size+1], [1,1...1, 0...0]
        whole_instance = whole_instance[:, None, :, :].expand(
            -1, self.pomo_size, -1, -1
        )
        to_delivery = to_delivery.repeat(1, self.pomo_size, 1)
        remain_pickup_max_distance = remain_pickup_max_distance[:, None, :].repeat(
            1, self.pomo_size, 1
        )
        remain_sum_paired_distance = remain_sum_paired_distance[:, None, :].repeat(
            1, self.pomo_size, 1
        )
        remain_delivery_max_distance = remain_delivery_max_distance[:, None, :].repeat(
            1, self.pomo_size, 1
        )
        depot_distance = depot_distance[:, None, :].repeat(1, self.pomo_size, 1)
        add_pd_distance = add_pd_distance[:, None, :].repeat(1, self.pomo_size, 1)
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
                # "coords": self.loc,
                "pds": torch.arange(
                    self.pomo_size, dtype=torch.int64, device=loc.device
                )[None, :, None].expand(
                    batch_size, -1, -1
                ),  # Add steps dimension
                "prev_a": self.agent_per[None, :, 0:1].repeat(batch_size, 1, 1),
                # Keep visited with depot so we can scatter efficiently (if there is an action for depot)
                "visited_": torch.zeros(
                    batch_size,
                    self.pomo_size,
                    n_loc + self.agent_num,
                    dtype=torch.uint8,
                    device=self.loc.device,
                ),  # Visited as mask is easier to understand, as long more memory efficient
                "lengths": torch.zeros(
                    batch_size, self.pomo_size, self.agent_num, device=self.loc.device
                ),
                "longest_lengths": torch.zeros(
                    batch_size, self.pomo_size, self.agent_num, device=loc.device
                ),
                "cur_coord": self.loc[:, None, :1, :].expand(
                    -1, self.pomo_size, -1, -1
                ),
                "count_depot": torch.zeros(
                    batch_size,
                    self.pomo_size,
                    1,
                    dtype=torch.int64,
                    device=self.loc.device,
                ),
                "to_delivery": to_delivery,
                "left_request": left_request
                * torch.ones(
                    batch_size, self.pomo_size, 1, dtype=torch.long, device=loc.device
                ),
                "remain_pickup_max_distance": remain_pickup_max_distance,
                "remain_sum_paired_distance": remain_sum_paired_distance,
                "remain_delivery_max_distance": remain_delivery_max_distance,
                "depot_distance": depot_distance,
                "add_pd_distance": add_pd_distance,
                "dist_out": dist_out[:, None, :, :].expand(-1, self.pomo_size, -1, -1),
            },
            batch_size=self.batch_size,
        )
        return out

    def _step(self, td: TensorDict) -> TensorDict:
        td = td.clone()
        self.selected_count += 1
        self.current_node = td["action"]

        # returning values
        done = self.selected_count == self.problem_size + self.agent_num - 1
        done_all = done.all()

        # Update the state
        n_loc = td["to_delivery"].size(-1) - self.agent_num  # number of customers
        new_to_delivery = (self.current_node + n_loc // 2) % (
            n_loc + self.agent_num
        )  # the pair node of selected node
        new_to_delivery = new_to_delivery[:, :, None]
        selected = self.current_node[:, :, None]  # Add dimension for step
        prev_a = selected
        # if selected node is pickup, is_request = True
        is_request = (prev_a >= self.agent_num) & (prev_a < self.agent_num + n_loc // 2)
        td["left_request"][is_request] -= 1
        depot_distance = td["depot_distance"].scatter(-1, prev_a, 0)
        add_pd = td["add_pd_distance"][is_request.squeeze(-1), :].gather(
            -1, prev_a[is_request.squeeze(-1), :] - self.agent_num
        )
        td["longest_lengths"][is_request.squeeze(-1), :] = td["longest_lengths"][
            is_request.squeeze(-1), :
        ].scatter_add(-1, td["count_depot"][is_request.squeeze(-1), :], add_pd)
        td["add_pd_distance"][is_request.squeeze(-1), :] = torch.scatter(
            td["add_pd_distance"][is_request.squeeze(-1), :],
            -1,
            prev_a[is_request.squeeze(-1), :] - self.agent_num,
            0,
        )
        remain_sum_paired_distance = td["add_pd_distance"].sum(-1, keepdim=True)
        remain_pickup_max_distance = depot_distance[
            :, :, : self.agent_num + n_loc // 2
        ].max(dim=-1, keepdim=True)[0]
        remain_delivery_max_distance = depot_distance[
            :, :, self.agent_num + n_loc // 2 :
        ].max(dim=-1, keepdim=True)[0]
        cur_coord = (
            self.loc[:, None, :, :]
            .expand(-1, depot_distance.size(1), -1, -1)
            .gather(2, selected[..., None].expand(-1, -1, -1, self.loc.size(-1)))
        )
        # To calculate makespan
        path_lengths = (cur_coord - td["cur_coord"]).norm(p=2, dim=-1)  # (batch_dim, 1)
        lengths = td["lengths"].scatter_add(-1, td["count_depot"], path_lengths)
        if done_all:
            reward = -lengths.max(-1)[0]
        else:
            reward = self.dummy_flag_long
        # if visit depot then plus one to count_depot
        td["count_depot"][
            (selected == td["agent_idx"]) & (td["count_depot"] < self.agent_num)
        ] += torch.ones(
            td["count_depot"][
                (selected == td["agent_idx"]) & (td["count_depot"] < self.agent_num)
            ].shape,
            dtype=torch.int64,
            device=td["count_depot"].device,
        )
        # Note: here we do not subtract one as we have to scatter so the first column allows scattering depot
        # Add one dimension since we write a single value
        visited_ = td["visited_"].scatter(-1, prev_a, 1)
        to_delivery = td["to_delivery"].scatter(-1, new_to_delivery, 1)
        agent_idx = td["agent_idx"]
        if ~((td["count_depot"] == td["agent_per"].size(2)).all()):
            # agent_idx is added by 1 if the current agent comes back to depot
            agent_idx = (
                td["agent_per"][:1, :, :]
                .expand(td["count_depot"].size(0), -1, -1)
                .gather(2, td["count_depot"])
            )
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
                "to_delivery": to_delivery,
                "agent_idx": agent_idx,
                "depot_distance": depot_distance,
                "remain_pickup_max_distance": remain_pickup_max_distance,
                "remain_sum_paired_distance": remain_sum_paired_distance,
                "remain_delivery_max_distance": remain_delivery_max_distance,
            }
        )
        return td
