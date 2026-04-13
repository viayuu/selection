from EasyNCO.neural_solvers.pipeline import Iteration
from tensordict import TensorDict
from torchrl.envs import EnvBase
from typing import Any, Tuple, Literal
from EasyNCO.utils.utils import getLogger
import torch
import numpy as np
from EasyNCO.neural_solvers.utils import get_best_solution

logger = getLogger(__name__)

class LEHDIteration(Iteration):

    def run(self,
            td: TensorDict,
            env: EnvBase,
            initialization_out: dict,
            phase: Literal["train", "eval"],
            max_steps: int = 100,
            **kwargs,
            ):
        if env._get_name() == 'TSPEnv':
            problems = kwargs.get('problems', None)
            iteration = TSPLEHDIteration(self.policy, problems)
            iteration_out = iteration.run(td, env, initialization_out, phase, max_steps)
        else:
            iteration = CVRPLEHDIteration(self.policy)
            iteration_out = iteration.run(td, env, initialization_out, phase, max_steps)



        return iteration_out

class TSPLEHDIteration:
    """
    Implementation of Random Re-Construction (RRC) proposed in LEHD. RRC iteratively refine a solution by randomly destroy
    and repair a partial route of a complete solution
    """
    def __init__(self, policy, problems):
        self.policy = policy
        self.problems = problems

    def run(self,
            td: TensorDict,
            env: EnvBase,
            initialization_out: dict,
            phase: Literal["train", "eval"],
            max_steps: int = 100,
            **kwargs,
    ) -> dict:
        batch = self.problems
        problems = batch
        self.env = env
        self.problem_size = batch.size(1)

        current_node_list = td['next']['selected_node_list'] if 'next' in td.keys() else td['selected_node_list']
        current_length, current_node_list = get_best_solution(reward=td["reward"], selected_node_list=current_node_list,batch_size= batch.size(0))
        # current_length = env._get_travel_distance(problems=problems,selected_node_list=current_node_list,batch_size= batch.size(0))
        mean_length = current_length.mean()
        best_selected_node_list = current_node_list
        if max_steps > 0:
            logger.info(f'RRC start, initial length: {current_length.mean().item()}')
        if self.policy.decoder_strategy is None:
            self.policy.set_decoder_strategy(kwargs.get("decoder_strategy", "greedy"))
        for bbbb in range(max_steps):
            env.load_problems(batch, batch.size(0))

            # Randomly sample the partial solution
            # random reverse
            if_inverse = True
            inverse_index = torch.randint(low = 0, high = 100, size=[1])[0] # [4, N]
            if inverse_index < 50:
                if_inverse = False
            if if_inverse:
                current_node_list = torch.flip(current_node_list, dims = [1])
            # sample partial solution

            partial_solution_length, first_node_index, subpath_length, solution_copy = self.destroy_solution(
                problems, current_node_list
            )

            prev_length = partial_solution_length
            current_step = 0

            reset_td = env.reset()
            self.policy.pre_forward(reset_td)
            state_td = env.pre_step()
            done = False
            next_td = state_td

            # Re-Construct subprobelm
            while not done:
                if current_step == 0:
                    selected = self.env.solution[:, :, -1]
                    next_td['action'] = selected
                elif current_step == 1:
                    selected = self.env.solution[:, :, 0]
                    next_td['action'] = selected
                else:
                    next_td = self.policy(state_td)

                current_step += 1
                state_td = env.step(next_td)
                done = state_td['done'].all()

            after_repair_sub_solution = torch.roll(env.selected_node_list, shifts = -1, dims = 2)
            # after_repair_sub_solution = env.selected_node_list
            after_repair_length = -state_td['reward']

            # Decide whether to accept the reconstructed partial solution
            after_repair_solution = self.accept_repaired_solution(after_repair_sub_solution, prev_length,
                                                                  after_repair_length, first_node_index, subpath_length,
                                                                  solution_copy)
            best_selected_node_list = after_repair_solution

            current_best_length = env._get_travel_distance(selected_node_list=best_selected_node_list, problems=problems)
            mean_length = current_best_length.mean()
            logger.info(f"RRC step{bbbb}, length {mean_length}")
            current_node_list = best_selected_node_list

        for i in range(best_selected_node_list.shape[0]):
            unique_node_list_len = len(torch.unique(best_selected_node_list[i]))
            assert unique_node_list_len == self.problem_size, \
                f"rrc process error, unique_node_list_len:{unique_node_list_len}, problem_size:{self.problem_size}"

        out = {
            'no_aug_score' : mean_length,
            'aug_score' : mean_length
        }

        return out


    def accept_repaired_solution(self, repaired_solution, pre_length, after_repair_length,
                                 first_node_index, subpath_length, solution_copy):
        problem_size = self.problem_size
        pomo_size = self.env.pomo_size
        batch_size = repaired_solution.shape[0]

        part1 = solution_copy[:, :, : first_node_index]
        part2 = solution_copy[:, :, first_node_index + subpath_length :]
        origin_sub_solution = solution_copy[:, :, first_node_index : first_node_index + subpath_length]

        sorted_solution_index, _ = torch.sort(origin_sub_solution, dim = 2, descending = False)

        index = torch.arange(sorted_solution_index.shape[0])[:, None, None].repeat(1, 1, sorted_solution_index.shape[-1])

        pomo_index = torch.arange(pomo_size, dtype = torch.long).repeat(batch_size, pomo_size, subpath_length)

        repaired_solution_index = sorted_solution_index[index, pomo_index, repaired_solution]

        if_repair = pre_length > after_repair_length

        solution_copy[if_repair] = torch.cat((part1[if_repair],
                                              repaired_solution_index[if_repair],
                                              part2[if_repair]), dim = 1)

        after_repair_solution = solution_copy[:, :, first_node_index : first_node_index + problem_size]

        return after_repair_solution


    def destroy_solution(self, problem, solution):
        destroyed_problem, destroyed_solution, first_node_index, subpath_length, solution_copy = self.sampling_subpaths(
            problem, solution, mode = 'test', repair = True
        )

        self.env.solution = destroyed_solution
        self.env.problems = destroyed_problem

        partial_solution_length = self.env._get_travel_distance(selected_node_list=destroyed_solution, problems=destroyed_problem)

        return partial_solution_length, first_node_index, subpath_length, solution_copy


    def sampling_subpaths(self, problems, solution, length_fix = False, mode = 'test', repair = False):
        problem_size = problems.shape[1]
        batch_size = problems.shape[0]
        embedding_size = problems.shape[2]

        first_node_index = torch.randint(low = 0, high = problem_size, size = [1])[0] # [0, N]

        # subpath length
        if mode == 'test':
            subpath_length = torch.randint(low = 4, high = problem_size + 1, size = [1])[0] # [4, N]
        else:
            if length_fix:
                subpath_length = problem_size
            else:
                subpath_length = torch.randint(low = 4, high = problem_size + 1, size = [1])[0]

        # Generate new solution
        solution_copy = torch.cat([solution, solution], dim = -1)
        # shape: (batch, pomo, N * 2)
        new_solution = solution_copy[:, :, first_node_index : first_node_index + subpath_length]
        # shape: (batch, pomo, subpath_length)
        # )
        new_solution_sort_ascending, rank = torch.sort(new_solution, dim = -1, descending = False)
        _, new_solution_rank = torch.sort(rank, dim = -1, descending = False)
        # shape: (batch, pomo, subpath_length)
        # problem without partial length
        new_solution_copy = torch.cat([new_solution, new_solution], dim = -1).type(torch.long)
        # Index2
        sorted_new_solution_index_pomo, _ = new_solution_copy.sort(dim = -1, descending = False)
        # shape: (batch, pomo, 2 * subpath_length)
        sorted_new_solution_index = sorted_new_solution_index_pomo.squeeze(1)
        # shape: (batch, 2 * subpath_length)
        # Index1
        global_index = torch.arange(batch_size, dtype = torch.long)[:, None].expand(batch_size, sorted_new_solution_index.shape[1])
        # shape: (batch, 2 * subpath_length)
        node_index = torch.arange((embedding_size), dtype = torch.long)[None, :].expand(batch_size, embedding_size)
        # shape: (batch, 2 * subpath_length)
        # Index3
        coordinate_index = node_index.repeat([1, subpath_length])

        new_problem = problems[global_index, sorted_new_solution_index, coordinate_index].view(batch_size, subpath_length, 2)

        if repair:
            return new_problem, new_solution_rank, first_node_index, subpath_length, solution_copy
        else:
            return new_problem, new_solution_rank

class CVRPLEHDIteration:

    def __init__(self, policy):
        self.policy = policy

    def run(self,
            td: TensorDict,
            env: EnvBase,
            initialization_out: dict,
            phase: Literal["train", "eval"],
            max_steps: int = 100,
            **kwargs,
    ) -> dict:
        batch = initialization_out['batch']
        problems = batch
        depot_node_xy = problems[:, :, [0, 1]]
        self.env = env
        self.problem_size = batch.size(1) - 1
        self.batch_size = batch.size(0)
        env.rrc = True

        current_solution = td['next']['selected_node_list'] if 'next' in td.keys() else td['selected_node_list']
        # prev_length = self.env._get_travel_distance_2(current_solution.squeeze(1), depot_node_xy)
        # TODO: with flag and without flag, flexible transformation, note that pomo technique is not used (not realized yet)
        prev_length = self.env._get_travel_distance_use_flag(current_solution, depot_node_xy)

        env.validate_solution_legal(problems, current_solution.squeeze(1))
        # shape: (batch, pomo, problem_size, 2)
        mean_length = prev_length.mean()
        best_selected_node_list = current_solution
        if max_steps > 0:
            logger.info(f'RRC start, initial length: {prev_length.mean().item()}')
        before_repair_length = prev_length
        before_repair_solution = current_solution

        for bbbb in range(max_steps):
            env.load_problems(batch, batch.size(0))

            current_solution = self.vrp_inverse(current_solution)
            # prev_length = self.env._get_travel_distance_2(current_solution.squeeze(1), depot_node_xy)
            prev_length = self.env._get_travel_distance_use_flag(current_solution, depot_node_xy)


            partial_solution_length, first_node_index, subpath_length, solution_copy, partial_solution = \
                self.destroy_solution(problems, current_solution)

            before_repair_sub_solution = self.env.solution

            before_reward = partial_solution_length
            current_step = 0

            reset_td = self.env.reset()
            self.policy.pre_forward(reset_td)
            state_td = env.pre_step()
            done = False
            next_td = state_td

            while not done:
                if current_step == 0:
                    selected_node = self.env.solution[:, :, 0, 0]
                    selected_flag = self.env.solution[:, :, 0, 1]
                    selected = torch.cat((selected_node, selected_flag), dim = -1)
                    selected = selected.unsqueeze(1)
                    next_td["action"] = selected
                else:
                    next_td = self.policy(state_td)

                current_step += 1
                state_td = env.step(next_td)
                done = state_td['done'].all()

            after_repair_sub_solution = self.env.selected_node_list
            after_reward = -state_td['reward']

            after_repair_solution, if_repair = self.accept_repaired_solution(after_repair_sub_solution, before_reward, after_reward,
                                                                  first_node_index, subpath_length, solution_copy)

            env.validate_solution_legal(problems, after_repair_solution.squeeze(1))
            current_solution = after_repair_solution

            # after_repair_length = env._get_travel_distance_2(after_repair_solution.squeeze(1), depot_node_xy)
            after_repair_length = env._get_travel_distance_use_flag(after_repair_solution, depot_node_xy)

            mean_length = after_repair_length.mean()
            logger.info(f"RRC step{bbbb}, length {mean_length}")
            before_repair_length = after_repair_length
            before_repair_solution = after_repair_solution

        out = {
            'no_aug_score' : mean_length,
            'aug_score' : mean_length
        }
        env.rrc = False
        return out

    def decide_whether_to_repair_solution(self,
                                          after_repair_sub_solution, before_reward, after_reward,
                                          first_node_index, length_of_subpath, double_solution):

        # TODO: squeeze pomo
        after_repair_sub_solution = after_repair_sub_solution.squeeze(1)
        before_reward = before_reward.squeeze(1)
        after_reward = after_reward.squeeze(1)
        double_solution = double_solution.squeeze(1)
        first_node_index = first_node_index.squeeze(1)

        the_whole_problem_size = int(double_solution.shape[1] / 2)
        batch_size = len(double_solution)
        temp = torch.arange(double_solution.shape[1])

        x3 = temp >= first_node_index[:, None].long()
        x4 = temp < (first_node_index[:, None] + length_of_subpath).long()
        x5 = x3 * x4

        origin_sub_solution = double_solution[x5.unsqueeze(2).repeat(1, 1, 2)].reshape(batch_size, length_of_subpath, 2)

        jjj, _ = torch.sort(origin_sub_solution[:, :, 0], dim=1, descending=False)

        index = torch.arange(batch_size)[:, None].repeat(1, jjj.shape[1])

        kkk_2 = jjj[index, after_repair_sub_solution[:, :, 0] - 1]

        after_repair_sub_solution[:, :, 0] = kkk_2

        if_repair = before_reward > after_reward

        need_to_repari_double_solution = double_solution[if_repair]
        need_to_repari_double_solution[x5[if_repair].unsqueeze(2).repeat(1, 1, 2)] = after_repair_sub_solution[if_repair].ravel()
        double_solution[if_repair] = need_to_repari_double_solution

        x6 = temp >= (first_node_index[:, None] + length_of_subpath - the_whole_problem_size).long()

        x7 = temp < (first_node_index[:, None] + length_of_subpath).long()

        x8 = x6 * x7

        after_repair_complete_solution = double_solution[x8.unsqueeze(2).repeat(1, 1, 2)].reshape(batch_size, the_whole_problem_size, -1)

        after_repair_complete_solution = after_repair_complete_solution.unsqueeze(1)

        return after_repair_complete_solution, if_repair

    def accept_repaired_solution(self, after_repair_sub_solution, before_reward, after_reward,
                                 first_node_index, subpath_length, solution_copy):
        problem_size = self.problem_size
        pomo_size = self.env.pomo_size
        batch_size = after_repair_sub_solution.shape[0]

        # TODO: squeeze pomo
        after_repair_sub_solution = after_repair_sub_solution.squeeze(1)
        before_reward = before_reward.squeeze(1)
        after_reward = after_reward.squeeze(1)
        solution_copy = solution_copy.squeeze(1)
        first_node_index = first_node_index.squeeze(1)

        temp = torch.arange(solution_copy.shape[1])
        x3 = temp >= first_node_index[:, None].long()
        x4 = temp < (first_node_index[:, None] + subpath_length).long()
        x5 = x3 * x4

        origin_sub_solution = solution_copy[x5.unsqueeze(2).repeat(1, 1, 2)].reshape(batch_size, subpath_length, 2)

        origin_sub_solution_index, _ = torch.sort(origin_sub_solution[:, :, 0], dim = 1, descending = False)
        batch_index = torch.arange(batch_size)[:, None].repeat(1, origin_sub_solution_index.shape[1])
        pomo_index = torch.arange(pomo_size, dtype = torch.long).repeat(batch_size, pomo_size, subpath_length)

        repaired_solution_index = origin_sub_solution_index[batch_index, after_repair_sub_solution[:, :, 0] - 1]

        after_repair_sub_solution[:, :, 0] = repaired_solution_index

        if_repair = before_reward > after_reward

        update_repaired_solution = solution_copy[if_repair]
        update_repaired_solution[x5[if_repair].unsqueeze(2).repeat(1, 1, 2)] = after_repair_sub_solution[if_repair].ravel()
        solution_copy[if_repair] = update_repaired_solution

        x6 = temp >= (first_node_index[:, None] + subpath_length - problem_size).long()
        x7 = temp < (first_node_index[:, None] + subpath_length).long()

        x8 = x6 * x7

        after_repair_solution = solution_copy[x8.unsqueeze(2).repeat(1, 1, 2)].reshape(batch_size, problem_size, -1)

        after_repair_solution = after_repair_solution.unsqueeze(1)
        return after_repair_solution, if_repair

    def destroy_solution(self, problem, solution):
        # destroyed_problem, partial_solution, first_node_index, subpath_length, solution_copy, remaining_capacity = self.sampling_subpath(
        #     problem, solution, mode = 'test'
        # )
        # capacity = torch.ones((self.batch_size, self.problem_size + 1))
        # problem = torch.cat((problem, capacity[:, :, None]), dim = 2)

        destroyed_problem, partial_solution, first_node_index, subpath_length, solution_copy, remaining_capacity = self.sampling_subpath(
            problem, solution, mode = 'test'
        )

        # solution_copy = solution_copy.unsqueeze(1)
        # partial_solution = partial_solution.unsqueeze(1)
        # first_node_index = first_node_index.unsqueeze(1)

        self.env.problems = destroyed_problem
        self.env.solution = partial_solution
        self.env.load = remaining_capacity
        self.env.depot_node_demand = destroyed_problem[:, :, 2]

        # partial_solution_length = self.env._get_travel_distance_2(partial_solution.squeeze(1), destroyed_problem[:, :, :2])
        partial_solution_length = self.env._get_travel_distance_use_flag(partial_solution, destroyed_problem[:, :, :2])

        return partial_solution_length, first_node_index, subpath_length, solution_copy, partial_solution

    def vrp_inverse(self, input_solution):
        clockwise = torch.rand(1)[0]
        current_solution = input_solution.clone().detach()

        if clockwise >= 0.5:
            current_solution = torch.flip(current_solution, dims = [2])
            index = torch.arange(current_solution.shape[2]).roll(shifts = 1)
            current_solution[:, :, :, 1] = current_solution[:, :, index, 1]

        # Find:
        # the number of sub-tours in each instance
        # the number of sub-paths in all instances
        # the longest length in a sub-path among all instances
        batch_size = current_solution.shape[0]
        problem_size = current_solution.shape[2]
        pomo_size = current_solution.shape[1]

        visit_depot_num = torch.sum(current_solution[:, :, :, 1], dim = 2)
        sub_tour_num = torch.sum(visit_depot_num)

        fake_solution = torch.cat(([current_solution[:, :, :, 1], torch.ones(batch_size)[:, None, None]]), dim = 2)
        start_from_depot = fake_solution.nonzero()
        start_from_depot_1 = start_from_depot[:, 2]
        start_from_depot_2 = torch.roll(start_from_depot_1, shifts = -1)
        sub_tour_length = start_from_depot_2 - start_from_depot_1
        max_sub_tour_length = torch.max(sub_tour_length)

        # For each sub-path, align the solution length by padding 0
        start_from_depot_3 = current_solution[:, :, :, 1].nonzero()
        start_from_depot_4 = current_solution[:, :, :, 1].roll(shifts = -1, dims = 2).nonzero()

        # repeat_solution_node = current_solution[:, :, :, 0].repeat_interleave(visit_depot_num, dim = 0)
        batch_indices = torch.arange(batch_size, dtype = torch.long)
        repeat_indices = batch_indices.repeat_interleave(visit_depot_num.squeeze(1), dim = 0)
        repeat_solution_node = current_solution[repeat_indices, :, :, 0]
        repeat_solution_node_copy = repeat_solution_node.repeat(1, 1, 2)

        index_1 = torch.arange(repeat_solution_node_copy.shape[2])[None, None, :].repeat(len(repeat_solution_node), pomo_size, 1) \
                   >= start_from_depot_3[:, 2][:, None, None]
        index_2 = torch.arange(repeat_solution_node_copy.shape[2])[None, None, :].repeat(len(repeat_solution_node), pomo_size, 1) \
                   <= start_from_depot_4[:, 2][:, None, None]
        index_3 = (index_1 * index_2).long()

        sub_tours = repeat_solution_node_copy * index_3

        index_4 = torch.arange(repeat_solution_node_copy.shape[2])[None, None, :].repeat(len(repeat_solution_node), pomo_size, 1) \
                  < (start_from_depot_3[:, 2][:, None, None] + max_sub_tour_length)

        index_5 = index_1 * index_4

        sub_tours_padding = sub_tours[index_5].reshape(sub_tour_num, pomo_size, max_sub_tour_length)

        # Inverse the tour based on random number

        clockwise = torch.rand(len(sub_tours_padding))
        clockwise_bool = clockwise.le(0.5)

        sub_tours_padding[clockwise_bool] = torch.flip(sub_tours_padding[clockwise_bool], dims = [2])

        # Reshape the sub_tour back to the original solution
        sub_tours_origin = sub_tours
        sub_tours_origin[index_5] = sub_tours_padding.ravel()
        solution_node_flip = sub_tours_origin[sub_tours_origin.gt(0.1)].reshape(batch_size, pomo_size, problem_size)
        solution_flip = torch.cat((solution_node_flip.unsqueeze(3), current_solution[:, :, :, 1].unsqueeze(3)), dim = 3)

        return solution_flip

    def sampling_subpath(self, problems, solution, length_fix = False, mode = 'test', repair = True):
        batch_size = problems.shape[0]
        problem_size = problems.shape[1] - 1
        embedding_size = problems.shape[2]
        pomo_size = solution.shape[1]

        # Uniformly sampling the first node of sub_path
        subpath_length = torch.randint(low = 4, high = problem_size + 1, size = [1])[0]

        start_from_depot = solution[:, :, :, 1].nonzero()

        end_with_depot = start_from_depot
        end_with_depot[:, 2] = end_with_depot[:, 2] - 1
        end_with_depot[end_with_depot.le(-0.5)] = solution.shape[2] - 1

        visit_depot_num = torch.sum(solution[:, :, :, 1], dim = 2)
        # p = torch.rand(len(visit_depot_num))
        p = torch.rand((visit_depot_num.shape[0], visit_depot_num.shape[1]))
        temp_end_with_depot_index = torch.floor(p * visit_depot_num).long()

        # TODO: Squeeze pomo
        visit_depot_num = visit_depot_num.squeeze(1)
        temp_end_with_depot_index = temp_end_with_depot_index.squeeze(1)

        temp_tri = np.triu(np.ones((len(visit_depot_num), len(visit_depot_num))), k = 1)
        visit_depot_num_numpy = visit_depot_num.clone().cpu().numpy()

        temp_index = np.dot(visit_depot_num_numpy, temp_tri)
        temp_index_torch = torch.from_numpy(temp_index).long().cuda()
        # temp_index_torch = torch.from_numpy(temp_index).long()

        end_with_depot_index = temp_end_with_depot_index + temp_index_torch
        end_with_depot_node = end_with_depot[end_with_depot_index, 2]

        solution_copy = torch.cat((solution, solution), dim = 2)
        end_with_depot_node = end_with_depot_node + problem_size

        subpath_index = torch.arange(subpath_length).repeat(batch_size, pomo_size, 1)
        offset = end_with_depot_node - subpath_length + 1

        solution_index = subpath_index + offset[:, None, None]

        batch_index = torch.arange(batch_size)[:, None, None].repeat(1, 1, 2 * subpath_length)
        # shape: (batch, pomo, 2 * subpath_length)
        pomo_index = torch.arange(pomo_size)[:, None, None].repeat(batch_size, 1, 2 * subpath_length)
        # shape: (batch, pomo, 2 * subpath_length)
        node_index = solution_index.repeat_interleave(2, dim = 2)
        # shape: (batch, pomo, 2 * subpath_length)
        flag_index = torch.arange(solution_copy.shape[3])[None, None, :].repeat(batch_size, pomo_size, subpath_length)
        # shape: (batch, pomo, 2 * subpath_length)
        sub_solution = solution_copy[batch_index, pomo_index, node_index, flag_index].reshape(batch_size, pomo_size,
                                                                                              subpath_length, 2)

        offset_index = problems.shape[0]
        start_index = solution_index[:, :, 0]

        x1 = torch.arange(solution_copy[: offset_index, :, :, 1].shape[2]) <= start_index[: offset_index, :][:, None]

        before_is_via_depot_all = solution_copy[: offset_index, :, :, 1] * x1
        before_is_via_depot = before_is_via_depot_all.nonzero()

        visit_depot_num_2 = torch.sum(before_is_via_depot_all, dim = 2)
        end_with_depot_node_index_2 = visit_depot_num_2 - 1

        # TODO squeeze pomo
        visit_depot_num_2 = visit_depot_num_2.squeeze(1)
        end_with_depot_node_index_2 = end_with_depot_node_index_2.squeeze(1)

        temp_tri_2 = np.triu(np.ones((len(visit_depot_num_2), len(visit_depot_num_2))), k = 1)
        visit_depot_num_numpy_2 = visit_depot_num_2.clone().cpu().numpy()

        temp_index_2 = np.dot(visit_depot_num_numpy_2, temp_tri_2)
        temp_index_torch_2 = torch.from_numpy(temp_index_2).long().cuda()
        # temp_index_torch_2 = torch.from_numpy(temp_index_2).long()

        end_with_depot_index_2 = end_with_depot_node_index_2 + temp_index_torch_2
        before_is_via_depot_index = before_is_via_depot[end_with_depot_index_2]

        before_start_index = before_is_via_depot_index[:, 2]
        x2 = torch.arange(solution_copy[:offset_index, :, :, 1].shape[2]) < start_index[:offset_index, :][:, None]
        x3 = torch.arange(solution_copy[:offset_index, :, :, 1].shape[2]) >= before_start_index[:, None, None]
        x4 = x2 * x3
        # TODO: squeeze pomo in solution_copy
        solution_copy = solution_copy.squeeze(1)
        x4 = x4.squeeze(1)

        double_solution_demand = problems[:offset_index,:,2][
            torch.arange(offset_index)[:, None].repeat(1, solution_copy.shape[1]), solution_copy[:offset_index, :, 0]]
        before_demand = double_solution_demand * x4
        satisfy_demand = before_demand.sum(1)

        # TODO: unsqueeze pomo in solution_copy
        solution_copy = solution_copy.unsqueeze(1)

        # TODO: Due to the capacity is 1 in EasyNCO, currently restrict to pomo=1
        capacity = torch.ones((offset_index, pomo_size))
        remaining_capacity = capacity - satisfy_demand[:, None]
        # problems[:offset_index, :, 3] = problems[:offset_index, :, 3] - self.satisfy_demand[:, None]

        # Update sub_tour index
        sub_solution_node = sub_solution[:, :, :, 0]

        new_solution_ascending, rank = torch.sort(sub_solution_node, dim = -1, descending = False)
        _, new_solution_rank = torch.sort(rank, dim = -1, descending = False)
        sub_solution[:, :, :, 0] = new_solution_rank + 1

        # TODO: Squeeze pomo
        new_solution_ascending = new_solution_ascending.squeeze(1)
        index_2, _ = torch.cat((new_solution_ascending, new_solution_ascending, new_solution_ascending,
                                ), dim = 1).type(torch.long).sort(dim = 1, descending = False)

        index_1 = torch.arange(batch_size, dtype = torch.long)[:, None].expand(batch_size, index_2.shape[1])
        temp = torch.arange((embedding_size), dtype = torch.long)[None, :].expand(batch_size, embedding_size)
        index_3 = temp.repeat([1, subpath_length])

        new_data = problems[index_1, index_2, index_3].view(batch_size, subpath_length, embedding_size)
        new_data = torch.cat((problems[:, 0, :].unsqueeze(1), new_data), dim = 1)

        if repair == True:
            return new_data, sub_solution, start_index, subpath_length, solution_copy, remaining_capacity
        else:
            return new_data, sub_solution, remaining_capacity

    def sampling_subpaths_repair(self, problems, solution, length_fix=False, mode='test', repair=True):
        # problems shape (B,V+1,4)
        # solution shape (B,V,2) index从1开始

        problems_size = problems.shape[1] - 1
        # print('problems_size',problems_size)
        batch_size = problems.shape[0]
        embedding_size = problems.shape[2]

        # the first node of subpath: uniform sampling, from 0 to N
        # 1.1

        length_of_subpath = torch.randint(low=4, high=problems_size+1 , size=[1])[0]  # in [4,N]
        # length_of_subpath = int(7)


        start_from_depot = solution[:, :, 1].nonzero()

        end_with_depot = start_from_depot
        end_with_depot[:, 1] = end_with_depot[:, 1] - 1
        end_with_depot[end_with_depot.le(-0.5)] = solution.shape[1] - 1

        # 1.4
        visit_depot_num = torch.sum(solution[:, :, 1], dim=1)

        p = torch.rand(len(visit_depot_num))
        select_end_with_depot_node_index = p * visit_depot_num
        select_end_with_depot_node_index = torch.floor(select_end_with_depot_node_index).long()

        temp_tri = np.triu(np.ones((len(visit_depot_num), len(visit_depot_num))), k=1)
        visit_depot_num_numpy = visit_depot_num.clone().cpu().numpy()

        temp_index = np.dot(visit_depot_num_numpy, temp_tri)
        temp_index_torch = torch.from_numpy(temp_index).long().cuda()

        select_end_with_depot_node_index_ = select_end_with_depot_node_index + temp_index_torch

        select_end_with_depot_node = end_with_depot[select_end_with_depot_node_index_, 1]
        # 1.5
        double_solution = torch.cat((solution, solution), dim=1)

        select_end_with_depot_node = select_end_with_depot_node + problems_size

        indexx = torch.arange(length_of_subpath).repeat(batch_size, 1)
        offset = select_end_with_depot_node - length_of_subpath + 1

        indexxxx = indexx + offset[:, None]


        sub_solu_index1 = torch.arange(batch_size)[:,None].repeat(1,2*length_of_subpath)
        sub_solu_index2 =indexxxx.repeat_interleave(2,dim=1)
        sub_solu_index3 = torch.arange(double_solution.shape[2])[None,:].repeat(batch_size,length_of_subpath)
        sub_solution = double_solution[sub_solu_index1,sub_solu_index2,sub_solu_index3].reshape(batch_size,length_of_subpath,2)

        offset_index = problems.shape[0]
        start_index = indexxxx[:, 0]


        x1 = torch.arange(double_solution[:offset_index, :, 1].shape[1]) <= start_index[:offset_index][:, None]

        start_capacity = 0
        before_is_via_depot_all = double_solution[:offset_index, :, 1] * x1
        before_is_via_depot = before_is_via_depot_all.nonzero()

        visit_depot_num_2 = torch.sum(before_is_via_depot_all, dim=1)

        select_end_with_depot_node_index_2 = visit_depot_num_2 - 1

        temp_tri_2 = np.triu(np.ones((len(visit_depot_num_2), len(visit_depot_num_2))), k=1)
        visit_depot_num_numpy_2 = visit_depot_num_2.clone().cpu().numpy()

        temp_index_2 = np.dot(visit_depot_num_numpy_2, temp_tri_2)
        temp_index_torch_2 = torch.from_numpy(temp_index_2).long().cuda()

        select_end_with_depot_node_index_2 = select_end_with_depot_node_index_2 + temp_index_torch_2
        before_is_via_depot_index = before_is_via_depot[select_end_with_depot_node_index_2]

        before_start_index = before_is_via_depot_index[:, 1]
        x2 = torch.arange(double_solution[:offset_index, :, 1].shape[1]) < start_index[:offset_index][:, None]
        x3 = torch.arange(double_solution[:offset_index, :, 1].shape[1]) >= before_start_index[:, None]
        x4 = x2 * x3
        double_solution_demand = problems[:offset_index, :, 2][
            torch.arange(offset_index)[:, None].repeat(1, double_solution.shape[1]), double_solution[:offset_index, :, 0]]

        before_demand = double_solution_demand * x4

        self.satisfy_demand = before_demand.sum(1)

        problems[:offset_index, :, 3] = problems[:offset_index, :, 3] - self.satisfy_demand[:, None]

        # -----------------------------
        # 2.
        # -----------------------------
        # 2.1
        sub_solution_node = sub_solution[:, :, 0]

        origin_sub_solution = sub_solution.clone()

        new_sulution_ascending, rank = torch.sort(sub_solution_node, dim=-1, descending=False)  # 升序
        _, new_sulution_rank = torch.sort(rank, dim=-1, descending=False)  # 升序
        sub_solution[:, :, 0] = new_sulution_rank + 1
        # 2.2
        index_2, _ = torch.cat((new_sulution_ascending, new_sulution_ascending, new_sulution_ascending, new_sulution_ascending), dim=1). \
            type(torch.long).sort(dim=-1, descending=False)

        index_1 = torch.arange(batch_size, dtype=torch.long)[:, None].expand(batch_size, index_2.shape[1])  # shape: [B, 2current_step]
        temp = torch.arange((embedding_size), dtype=torch.long)[None, :].expand(batch_size, embedding_size)  # shape: [B, current_step]
        index_3 = temp.repeat([1, length_of_subpath])

        new_data = problems[index_1, index_2, index_3].view(batch_size, length_of_subpath, embedding_size)
        new_data = torch.cat((problems[:, 0, :].unsqueeze(dim=1), new_data), dim=1)

        remaining_capacity = new_data[:, :, 3]
        if repair == True:
            return new_data, sub_solution,start_index,length_of_subpath,double_solution, remaining_capacity, origin_sub_solution
        else:
            return new_data, sub_solution

    def vrp_whole_and_solution_subrandom_inverse(self, solution):
        solution = solution.squeeze(1)

        clockwise_or_not = torch.rand(1)[0]

        if clockwise_or_not >= 0.5:
            solution = torch.flip(solution, dims=[1])
            index = torch.arange(solution.shape[1]).roll(shifts=1)
            solution[:, :, 1] = solution[:, index, 1]

        # 1.
        # find the number of subtours in each instance.
        # the total number of subpaths in all instances:     all_subtour_num，
        # The longest length in a subpath among all instances:  max_subtour_length
        batch_size = solution.shape[0]
        problem_size = solution.shape[1]

        visit_depot_num = torch.sum(solution[:, :, 1], dim=1)
        all_subtour_num = torch.sum(visit_depot_num)

        fake_solution = torch.cat((solution[:, :, 1], torch.ones(batch_size)[:, None]), dim=1)

        start_from_depot = fake_solution.nonzero()

        start_from_depot_1 = start_from_depot[:, 1]

        start_from_depot_2 = torch.roll(start_from_depot_1, shifts=-1)

        sub_tours_length = start_from_depot_2 - start_from_depot_1

        max_subtour_length = torch.max(sub_tours_length)

        # 2。
        # For each subpath, take it out separately, pandding 0 to length max_subtour_length
        #For each instance, padding 0 to max_subtour_num number of subpaths
        # 3.
        # Put all subpaths of all instances into the same array

        start_from_depot2 = solution[:, :, 1].nonzero()
        start_from_depot3 = solution[:, :, 1].roll(shifts=-1, dims=1).nonzero()

        repeat_solutions_node = solution[:, :, 0].repeat_interleave(visit_depot_num, dim=0)
        double_repeat_solution_node = repeat_solutions_node.repeat(1, 2)

        x1 = torch.arange(double_repeat_solution_node.shape[1])[None, :].repeat(len(repeat_solutions_node), 1) \
             >= start_from_depot2[:, 1][:, None]
        x2 = torch.arange(double_repeat_solution_node.shape[1])[None, :].repeat(len(repeat_solutions_node), 1) \
             <= start_from_depot3[:, 1][:, None]

        x3 = (x1 * x2).long()

        sub_tourss = double_repeat_solution_node * x3

        x4 = torch.arange(double_repeat_solution_node.shape[1])[None, :].repeat(len(repeat_solutions_node), 1) \
             < (start_from_depot2[:, 1][:, None] + max_subtour_length)

        x5 = x1 * x4

        sub_tours_padding = sub_tourss[x5].reshape(all_subtour_num, max_subtour_length)

        # 4.
        # For each row, a random number of [0,100] is generated, greater than 50 is positive and less than 50 is inverse

        clockwise_or_not = torch.rand(len(sub_tours_padding))

        clockwise_or_not_bool = clockwise_or_not.le(0.5)

        # 5.
        # For each row, randomly flip

        sub_tours_padding[clockwise_or_not_bool] = torch.flip(sub_tours_padding[clockwise_or_not_bool], dims=[1])

        # 6。
        # Map the subtours to the original solution matrix dimension
        sub_tourss_back = sub_tourss

        sub_tourss_back[x5] = sub_tours_padding.ravel()

        solution_node_flip = sub_tourss_back[sub_tourss_back.gt(0.1)].reshape(batch_size, problem_size)

        solution_flip = torch.cat((solution_node_flip.unsqueeze(2), solution[:, :, 1].unsqueeze(2)), dim=2)

        solution_flip.unsqueeze(1)
        return solution_flip

    def node_flag_tran_to_(self, node_flag):
        '''
        :param node_list: [B, V, 2]
        :return: [B, V+n]
        '''

        batch_size = node_flag.shape[0]
        problem_size = node_flag.shape[1]
        node = node_flag[:, :, 0]
        flag = node_flag[:, :, 1]
        depot_num = flag.sum(1)

        max_length = torch.max(depot_num)

        store_1 = torch.ones(size=(batch_size, problem_size + max_length), dtype=torch.long)

        where_is_depot_0, where_is_depot_1 = torch.where(flag == 1)

        temp1 = torch.arange(max_length)[None, :].repeat(batch_size, 1)
        temp2 = temp1 < depot_num[:, None]
        temp3 = temp1[temp2]
        where_is_depot_1 = where_is_depot_1 + temp3

        store_1[where_is_depot_0, where_is_depot_1] = 0

        mask = torch.arange(problem_size + max_length)[None, :].repeat(batch_size, 1)
        nodesss = problem_size + depot_num
        mask2 = (mask < nodesss[:, None]).long()
        store_2 = store_1 * mask2

        store_2[store_2.gt(0.1)] = node.ravel()

        zeros = torch.zeros(size=(batch_size, 1), dtype=torch.long)

        result = torch.cat((store_2, zeros), dim=1)

        return result

    def tran_to_node_flag(self, node_list):
        '''
        :param node_list: [B, V+n]
        :return: [B, V, 2]
        '''

        batch_size = node_list.shape[0]

        index_smaller_0_shift = torch.roll(torch.le(node_list, 0), shifts=1, dims=1).long()
        index_smaller_0_shift[:, 0] = 0
        index_bigger_0 = torch.gt(node_list, 0).long()

        flag_index = index_smaller_0_shift * index_bigger_0

        save_index = torch.gt(node_list, 0.1)

        save_node = node_list[save_index].reshape(batch_size, -1)
        save_flag = flag_index[save_index].reshape(batch_size, -1)

        node_flag_1 = torch.cat((save_node.unsqueeze(2), save_flag.unsqueeze(2)), dim=2)

        return node_flag_1

    def calculate_distance_by_hand(self, distance_matrix, solution, skip_first = False):
        total_distance = 0
        depot_node_index = 0
        first = True

        last_node_index = depot_node_index
        for row in solution:
            current_node_index = row[0]
            is_new_route_flag = row[1]

            # 如果是新路径的起点 (flag == 1)
            if is_new_route_flag == 1:
                # 在开始一条新路径之前，必须先将上一条路径的最后一个节点返回仓库
                # (此条件防止在处理第一条路径时，错误地添加 0->0 的距离)
                if last_node_index != depot_node_index:
                    # 计算上一辆车返回仓库的距离
                    distance_back_to_depot = torch.linalg.norm(
                        distance_matrix[last_node_index] - distance_matrix[depot_node_index])
                    # print(
                    #     f"  上一路径结束: {last_node_index} -> 仓库 {depot_node_index} | 距离: {distance_back_to_depot:.4f}")
                    total_distance += distance_back_to_depot

                last_node_index = depot_node_index
            if skip_first and first:
                distance_segment = 0
                first = False
            else:
                distance_segment = torch.linalg.norm(distance_matrix[last_node_index] - distance_matrix[current_node_index])
            segment_type = "新路径" if is_new_route_flag == 1 else "路径延续"
            # print(f"- {segment_type}: {last_node_index} -> {current_node_index} | 距离: {distance_segment:.4f}")
            total_distance += distance_segment

            last_node_index = current_node_index

        distance_final_return = torch.linalg.norm(distance_matrix[last_node_index] - distance_matrix[depot_node_index])
        # print(
        #     f"  最后路径结束: {last_node_index} -> 仓库 {depot_node_index} | 距离: {distance_final_return:.4f}")
        total_distance += distance_final_return
        # print("-" * 30)
        print(f"计算出的总路径长度为: {total_distance:.4f}")

        return total_distance



