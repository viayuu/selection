import torch
from typing import Optional
from torchrl.envs import EnvBase
from tensordict import TensorDict
import itertools  # for permutation list

from EasyNCO.utils.utils import getLogger

logger = getLogger(__name__)


class FFSPEnv(EnvBase):

    def __init__(
        self,
        pomo_size: int = 20,
        device: str = "cpu",
        seed: int = 0,
        machine_cnt_list=[4, 4, 4],
        job_cnt=20,
        **kwargs,
    ):
        super().__init__(device=device)
        self.env_name = "ffsp"
        self.pomo_size = pomo_size
        self.device = device
        self.aug_type = None

        self.seed_value = seed
        self.BATCH_IDX = None
        self.POMO_IDX = None
        # IDX.shape: (batch, pomo)
        self.problems = None
        # shape: (batch, node, node)

        self.machine_cnt_list = machine_cnt_list
        self.sm_indexer = _Stage_N_Machine_Index_Converter(self)

        self.selected_count = None
        self.current_node = None
        # shape: (batch, pomo)
        self.selected_node_list = None
        # shape: (batch, pomo, 0~)

        self._set_seed(seed=self.seed_value)
        self.aug_factor = kwargs.get("aug_factor")

    def _set_seed(self, seed: Optional[int]):
        rng = torch.manual_seed(seed)
        self.rng = rng

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

    def load_problems(self, dataset, batch_size: int = 64):
        self.problems = dataset
        # shape: (batch, job_cnt, sum(machine_cnt_list))
        self.job_cnt = self.problems.size(1)
        self.env_batch_size = batch_size
        self.batch_size = torch.Size([batch_size, self.pomo_size])
        if self.aug_factor != 1:
            self.batch_size = torch.Size(
                [self.env_batch_size * self.aug_factor, self.pomo_size]
            )
            self.env_batch_size = self.env_batch_size * self.aug_factor
            self.problems = self.problems.repeat(self.aug_factor, 1, 1)
        self.BATCH_IDX = torch.arange(self.env_batch_size)[:, None].expand(
            self.env_batch_size, self.pomo_size
        )
        self.POMO_IDX = torch.arange(self.pomo_size)[None, :].expand(
            self.env_batch_size, self.pomo_size
        )
        assert self.env_batch_size == self.problems.size(0), (
            "batch_size and the first dimension of problems should be the same. "
            + f"Expected batch_size: {self.problems.size(0)}, got: {self.env_batch_size}"
        )

        self.job_durations = torch.empty(
            size=(self.env_batch_size, self.job_cnt + 1, sum(self.machine_cnt_list)),
            dtype=torch.long,
        )
        # shape: (batch, job+1, total_machine)
        self.job_durations[:, : self.job_cnt, :] = self.problems[:, :, :]
        self.job_durations[:, self.job_cnt, :] = 0

    def _reset(self, td: TensorDict, batch_size=None) -> TensorDict:
        self.output_spec = None  # Solve the inconsistent batch size
        self.time_idx = torch.zeros(
            size=(self.env_batch_size, self.pomo_size), dtype=torch.long
        )
        # shape: (batch, pomo)
        self.sub_time_idx = torch.zeros(
            size=(self.env_batch_size, self.pomo_size), dtype=torch.long
        )
        # shape: (batch, pomo)
        self.machine_idx = self.sm_indexer.get_machine_index(
            self.POMO_IDX, self.sub_time_idx
        )
        # shape: (batch, pomo)

        self.schedule = torch.full(
            size=(
                self.env_batch_size,
                self.pomo_size,
                sum(self.machine_cnt_list),
                self.job_cnt + 1,
            ),
            dtype=torch.long,
            fill_value=-999999,
        )
        # shape: (batch, pomo, machine, job+1)
        self.machine_wait_step = torch.zeros(
            size=(self.env_batch_size, self.pomo_size, sum(self.machine_cnt_list)),
            dtype=torch.long,
        )
        # shape: (batch, pomo, machine)
        self.job_location = torch.zeros(
            size=(self.env_batch_size, self.pomo_size, self.job_cnt + 1),
            dtype=torch.long,
        )
        # shape: (batch, pomo, job+1)
        self.job_wait_step = torch.zeros(
            size=(self.env_batch_size, self.pomo_size, self.job_cnt + 1),
            dtype=torch.long,
        )
        # shape: (batch, pomo, job+1)
        self.finished = torch.full(
            size=(self.env_batch_size, self.pomo_size),
            dtype=torch.bool,
            fill_value=False,
        )
        # shape: (batch, pomo)
        # for done
        self.dummy_flag_bool = torch.zeros(
            (self.env_batch_size, self.pomo_size), dtype=torch.bool
        )
        # for action, -1 represents the dummy action, which is used to select the first node
        self.dummy_flag_long = (
            torch.zeros((self.env_batch_size, self.pomo_size), dtype=torch.long) - 1
        )

        self.job_ninf_mask = None
        return TensorDict(
            {
                "problems": self.problems,
            },
            batch_size=torch.Size([self.env_batch_size]),
        )

    def pre_step(self) -> TensorDict:
        self.step_cnt = 1
        self.stage_idx = self.sm_indexer.get_stage_index(self.sub_time_idx)
        self.stage_machine_idx = self.sm_indexer.get_stage_machine_index(
            self.POMO_IDX, self.sub_time_idx
        )
        job_loc = self.job_location[:, :, : self.job_cnt]
        # shape: (batch, pomo, job)
        job_wait_t = self.job_wait_step[:, :, : self.job_cnt]
        # shape: (batch, pomo, job)

        job_in_stage = job_loc == self.stage_idx[:, :, None]
        # shape: (batch, pomo, job)
        job_not_waiting = job_wait_t == 0
        # shape: (batch, pomo, job)
        job_available = job_in_stage & job_not_waiting
        # shape: (batch, pomo, job)

        job_in_previous_stages = (job_loc < self.stage_idx[:, :, None]).any(dim=2)
        # shape: (batch, pomo)
        job_waiting_in_stage = (job_in_stage & (job_wait_t > 0)).any(dim=2)
        # shape: (batch, pomo)
        wait_allowed = job_in_previous_stages + job_waiting_in_stage + self.finished
        # shape: (batch, pomo)

        self.job_ninf_mask = torch.full(
            size=(self.env_batch_size, self.pomo_size, self.job_cnt + 1),
            fill_value=float("-inf"),
        )
        # shape: (batch, pomo, job+1)
        job_enable = torch.cat((job_available, wait_allowed[:, :, None]), dim=2)
        # shape: (batch, pomo, job+1)
        self.job_ninf_mask[job_enable] = 0
        # shape: (batch, pomo, job+1)
        self.step_cnt = 0
        return TensorDict(
            {
                "stage_idx": self.stage_idx,
                "stage_machine_idx": self.stage_machine_idx,
                "job_ninf_mask": self.job_ninf_mask,
                "finished": self.finished,
                "batch_idx": self.BATCH_IDX,
                "pomo_idx": self.POMO_IDX,
                "reward": self.dummy_flag_long,
                "done": self.dummy_flag_bool,
            },
            batch_size=self.batch_size,
        )

    def _step(self, td: TensorDict) -> TensorDict:
        job_idx = td["action"]
        # job_idx.shape: (batch, pomo)

        self.schedule[self.BATCH_IDX, self.POMO_IDX, self.machine_idx, job_idx] = (
            self.time_idx
        )

        job_length = self.job_durations[self.BATCH_IDX, job_idx, self.machine_idx]
        # shape: (batch, pomo)
        self.machine_wait_step[self.BATCH_IDX, self.POMO_IDX, self.machine_idx] = (
            job_length
        )
        # shape: (batch, pomo, machine)
        self.job_location[self.BATCH_IDX, self.POMO_IDX, job_idx] += 1
        # shape: (batch, pomo, job+1)
        self.job_wait_step[self.BATCH_IDX, self.POMO_IDX, job_idx] = job_length
        # shape: (batch, pomo, job+1)
        self.finished = (self.job_location[:, :, : self.job_cnt] == len(self.machine_cnt_list)).all(
            dim=2
        )
        # shape: (batch, pomo)

        if self.finished.all():
            pass  # do nothing. do not update step_state, because it won't be used anyway
        else:
            self._move_to_next_machine()
            self.step_cnt += 1

            self.stage_idx = self.sm_indexer.get_stage_index(self.sub_time_idx)
            # shape: (batch, pomo)
            self.stage_machine_idx = self.sm_indexer.get_stage_machine_index(
                self.POMO_IDX, self.sub_time_idx
            )
            # shape: (batch, pomo)

            job_loc = self.job_location[:, :, : self.job_cnt]
            # shape: (batch, pomo, job)
            job_wait_t = self.job_wait_step[:, :, : self.job_cnt]
            # shape: (batch, pomo, job)

            job_in_stage = job_loc == self.stage_idx[:, :, None]
            # shape: (batch, pomo, job)
            job_not_waiting = job_wait_t == 0
            # shape: (batch, pomo, job)
            job_available = job_in_stage & job_not_waiting
            # shape: (batch, pomo, job)

            job_in_previous_stages = (job_loc < self.stage_idx[:, :, None]).any(dim=2)
            # shape: (batch, pomo)
            job_waiting_in_stage = (job_in_stage & (job_wait_t > 0)).any(dim=2)
            # shape: (batch, pomo)
            wait_allowed = job_in_previous_stages + job_waiting_in_stage + self.finished
            # shape: (batch, pomo)

            self.job_ninf_mask = torch.full(
                size=(self.env_batch_size, self.pomo_size, self.job_cnt + 1),
                fill_value=float("-inf"),
            )
            # shape: (batch, pomo, job+1)
            job_enable = torch.cat((job_available, wait_allowed[:, :, None]), dim=2)
            # shape: (batch, pomo, job+1)
            self.job_ninf_mask[job_enable] = 0
            # shape: (batch, pomo, job+1)

        if self.finished.all():
            reward = (
                -self._get_makespan()
            )  # Note the MINUS Sign ==> We want to MAXIMIZE reward
            # shape: (batch, pomo)
        else:
            reward = self.dummy_flag_long
        next_state = {
            "reward": reward,
            "done": self.finished,
        }
        return TensorDict(
            {
                "next": next_state,
                "reward": reward,
                "done": self.finished,
                "stage_idx": self.stage_idx,
                "stage_machine_idx": self.stage_machine_idx,
                "job_ninf_mask": self.job_ninf_mask,
                "finished": self.finished,
                "batch_idx": self.BATCH_IDX,
                "pomo_idx": self.POMO_IDX,
            },
            batch_size=self.batch_size,
        )

    def _move_to_next_machine(self):

        b_idx = torch.flatten(self.BATCH_IDX)
        # shape: (batch*pomo,) == (not_ready_cnt,)
        p_idx = torch.flatten(self.POMO_IDX)
        # shape: (batch*pomo,) == (not_ready_cnt,)
        ready = torch.flatten(self.finished)
        # shape: (batch*pomo,) == (not_ready_cnt,)

        b_idx = b_idx[~ready]
        # shape: ( (NEW) not_ready_cnt,)
        p_idx = p_idx[~ready]
        # shape: ( (NEW) not_ready_cnt,)

        while ~ready.all():
            new_sub_time_idx = self.sub_time_idx[b_idx, p_idx] + 1
            # shape: (not_ready_cnt,)
            step_time_required = new_sub_time_idx == sum(self.machine_cnt_list)
            # shape: (not_ready_cnt,)
            self.time_idx[b_idx, p_idx] += step_time_required.long()
            new_sub_time_idx[step_time_required] = 0
            self.sub_time_idx[b_idx, p_idx] = new_sub_time_idx
            new_machine_idx = self.sm_indexer.get_machine_index(p_idx, new_sub_time_idx)
            self.machine_idx[b_idx, p_idx] = new_machine_idx

            machine_wait_steps = self.machine_wait_step[b_idx, p_idx, :]
            # shape: (not_ready_cnt, machine)
            machine_wait_steps[step_time_required, :] -= 1
            machine_wait_steps[machine_wait_steps < 0] = 0
            self.machine_wait_step[b_idx, p_idx, :] = machine_wait_steps

            job_wait_steps = self.job_wait_step[b_idx, p_idx, :]
            # shape: (not_ready_cnt, job+1)
            job_wait_steps[step_time_required, :] -= 1
            job_wait_steps[job_wait_steps < 0] = 0
            self.job_wait_step[b_idx, p_idx, :] = job_wait_steps

            machine_ready = self.machine_wait_step[b_idx, p_idx, new_machine_idx] == 0
            # shape: (not_ready_cnt,)

            new_stage_idx = self.sm_indexer.get_stage_index(new_sub_time_idx)
            # shape: (not_ready_cnt,)
            job_ready_1 = (
                self.job_location[b_idx, p_idx, : self.job_cnt]
                == new_stage_idx[:, None]
            )
            # shape: (not_ready_cnt, job)
            job_ready_2 = self.job_wait_step[b_idx, p_idx, : self.job_cnt] == 0
            # shape: (not_ready_cnt, job)
            job_ready = (job_ready_1 & job_ready_2).any(dim=1)
            # shape: (not_ready_cnt,)

            ready = machine_ready & job_ready
            # shape: (not_ready_cnt,)

            b_idx = b_idx[~ready]
            # shape: ( (NEW) not_ready_cnt,)
            p_idx = p_idx[~ready]
            # shape: ( (NEW) not_ready_cnt,)

    def _get_makespan(self):

        job_durations_perm = self.job_durations.permute(0, 2, 1)
        # shape: (batch, machine, job+1)
        end_schedule = self.schedule + job_durations_perm[:, None, :, :]
        # shape: (batch, pomo, machine, job+1)

        end_time_max, _ = end_schedule[:, :, :, : self.job_cnt].max(dim=3)
        # shape: (batch, pomo, machine)
        end_time_max, _ = end_time_max.max(dim=2)
        # shape: (batch, pomo)
        end_time_max = end_time_max.type(torch.float)

        return end_time_max


class _Stage_N_Machine_Index_Converter:
    def __init__(self, env):
        assert env.machine_cnt_list == [4, 4, 4]
        assert env.pomo_size == 24

        machine_SUBindex_0 = torch.tensor(list(itertools.permutations([0, 1, 2, 3])))
        machine_SUBindex_1 = torch.tensor(list(itertools.permutations([0, 1, 2, 3])))
        machine_SUBindex_2 = torch.tensor(list(itertools.permutations([0, 1, 2, 3])))
        self.machine_SUBindex_table = torch.cat(
            (machine_SUBindex_0, machine_SUBindex_1, machine_SUBindex_2), dim=1
        )
        # machine_SUBindex_table.shape: (pomo, total_machine)

        starting_SUBindex = [0, 4, 8]
        machine_order_0 = machine_SUBindex_0 + starting_SUBindex[0]
        machine_order_1 = machine_SUBindex_1 + starting_SUBindex[1]
        machine_order_2 = machine_SUBindex_2 + starting_SUBindex[2]
        self.machine_table = torch.cat(
            (machine_order_0, machine_order_1, machine_order_2), dim=1
        )
        # machine_table.shape: (pomo, total_machine)

        # assert env.pomo_size == 1
        # self.machine_SUBindex_table = torch.tensor([[0,1,2,3,0,1,2,3,0,1,2,3]])
        # self.machine_table = torch.tensor([[0,1,2,3,4,5,6,7,8,9,10,11]])

        self.stage_table = torch.tensor(
            [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2], dtype=torch.long
        )

    def get_stage_index(self, sub_time_idx):
        return self.stage_table[sub_time_idx]

    def get_machine_index(self, POMO_IDX, sub_time_idx):
        # POMO_IDX.shape: (batch, pomo)
        # sub_time_idx.shape: (batch, pomo)
        return self.machine_table[POMO_IDX, sub_time_idx]
        # shape: (batch, pomo)

    def get_stage_machine_index(self, POMO_IDX, sub_time_idx):
        return self.machine_SUBindex_table[POMO_IDX, sub_time_idx]
