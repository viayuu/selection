import math
import torch

from EasyNCO.data.data_utils import glop_coordinate_transformation


def tsp_decompose_and_solve(problem,solution,policy, env,revision_len, iter,decoder_strategy="greedy"):
    """
    Divide and conquer the TSP problem by dividing it into subproblems of length 'revision_len',
    and solve each subproblem iteratively 'iter_count' times using the 'policy'.
    """
    device=problem.device
    batch_size, problem_size, pro_dim = problem.shape
    if revision_len > problem_size:
        return solution
    offset_counts = problem_size % revision_len
    offset_solution=None
    shift_len = max(revision_len // iter, 1)
    for i in range(iter):
        data0=problem.clone()
        solution=solution.roll(dims=-1,shifts=shift_len*i)
        if offset_counts:
            offset_solution=solution[:,-offset_counts:]
        div_solution=solution[:,0:(problem_size-offset_counts)]
        data0=data0.gather(1, div_solution.unsqueeze(-1).repeat(1, 1, 2))
        decomposed_seeds=data0.reshape(-1, revision_len, pro_dim)#
        original_subtour = torch.arange(0, revision_len, dtype=torch.long).to(device)

        ori_cost = (decomposed_seeds[:, 1:] - decomposed_seeds[:, :-1]).norm(p=2, dim=2).sum(1)
        # env = TSPEnv(problem_size=revision_len, pomo_size=2, aug_type='pomo_aug', aug_factor=4, device=device)
        trans_data = glop_coordinate_transformation(decomposed_seeds)
        env.load_problems(trans_data, batch_size=trans_data.size(0))
        sub_tour, reward = sub_tsp_solution(env, policy, decoder_strategy)
        new_data = decomposed_seeds.gather(1, sub_tour.unsqueeze(-1).expand_as(decomposed_seeds))  # policy的排序后的矩阵
        policy_cost = (new_data[:, 1:] - new_data[:, :-1]).norm(p=2, dim=2).sum(1)
        sub_tour[policy_cost > ori_cost] = original_subtour
        sub_tour=sub_tour.reshape(batch_size, -1)
        for i in range(math.floor(problem_size/revision_len)):
            sub_tour[:,i*revision_len:(i+1)*revision_len]+=(revision_len)*i

        solution=solution.gather(1,sub_tour)
        if offset_counts :
            solution = torch.cat([solution, offset_solution], dim=1)
    return solution

@torch.no_grad()
def sub_tsp_solution(env,policy,decoder_strategy):
    """
    Solve the TSP subproblem using the given policy, returning the reconstructed cost and path.
    """
    policy.eval()
    reset_td = env.reset()
    policy.set_decoder_strategy(decoder_strategy)
    policy.pre_forward(reset_td)
    done=False
    state_td = env.pre_step()

    while not done:
        next_td = policy(state_td, first_mode='placeholder')
        state_td = env.step(next_td)
        done = state_td["done"].all()
    reward=state_td["next"]["reward"].squeeze(-1)
    solution=state_td["next"]["selected_node_list"]
    solution[:,1,:]=torch.flip(solution[:,1,:], dims=(-1,))
    reward = reward.reshape(env.aug_factor, -1, env.pomo_size).transpose(0,1)
    solution = solution.reshape(env.aug_factor, -1, env.pomo_size, env.problem_size).transpose(0,1)
    ori_batch_size=reward.shape[0]
    reward=reward.reshape(ori_batch_size,-1)
    solution=solution.reshape(ori_batch_size,-1,env.problem_size)
    max_reward, max_reward_idx = reward.max(dim=1)
    optimal_solution = solution[torch.arange(ori_batch_size), max_reward_idx]
    return optimal_solution,reward
def sub_tsp_len(data):
    rolled_seq = data.roll(dims=1, shifts=-1)
    dist = ((data - rolled_seq) ** 2).sum(2).sqrt().sum(1)
    return dist