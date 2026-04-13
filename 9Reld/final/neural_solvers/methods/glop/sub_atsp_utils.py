import torch


def sub_atsp_len(tour, dist):
    cost = dist[tour[:-1], tour[1:]].sum()
    cost = cost.unsqueeze(0)
    return cost


def atsp_decompose_and_solve(problem,solution,policy, env,revision_len, iter,decoder_strategy="sampling"):
    '''
    Divide and conquer the complete ATSP problem by dividing it into subproblems of length 'revision_len',
    and solve each subproblem iteratively 'iter' times using the 'policy'.
    '''
    N_SHIFTS=revision_len//iter
    for p in range(iter):
        sub_tours = solution.reshape(-1, revision_len)  # shape: (batch, revision_len)
        sub_insts = [problem[sub_tour][:, sub_tour] for sub_tour in sub_tours]
        original_scores = torch.tensor(
            [sub_atsp_len(sub_tour, problem) for sub_tour in sub_tours])  # note that original_scores are positive values

        scale_coef = [sub_inst.max() for sub_inst in sub_insts]
        sub_insts = torch.stack(sub_insts)
        sub_insts_scaled = sub_insts / torch.tensor(scale_coef)[:, None, None]
        # Main part of the revision
        env.load_problems(sub_insts_scaled,sub_insts_scaled.shape[0])
        policy_cost,policy_solution=sub_atsp_solution(env,policy,decoder_strategy)
        revised_scores = - policy_cost * torch.tensor(scale_coef)
        improved_scores = original_scores - revised_scores
        # subtours should be aranged in the same order as the original tours, if the improved_scores <= 0
        policy_solution[improved_scores <= 0] = torch.arange(sub_tours.shape[1])
        # Gather the subtours according to the solutions
        revised_tours = sub_tours.gather(1, policy_solution)
        # Flatten the revised_tours
        tour = revised_tours.reshape(-1)  # shape: (batch * revision_len) i.e. (node_cnt,)
        solution = torch.roll(tour, shifts=N_SHIFTS, dims=-1)
    return solution


@torch.no_grad()
def sub_atsp_solution(env,policy,decoder_strategy):
    """
    Solve the ATSP subproblem using the given policy, returning the reconstructed cost and path.
    """
    policy.eval()
    reset_td = env.reset()
    policy.set_decoder_strategy(decoder_strategy)
    policy.pre_forward(reset_td)
    done=False
    state = env.pre_step()
    while not done:
        next_state = policy(state)
        # shape: (batch, pomo)
        state = env.step(next_state)
        done = state["done"].all()
        reward = state["reward"]
    aug_reward = reward.reshape(env.batch_size, env.pomo_size)
    solutions = env.selected_node_list
    max_pomo_reward, max_pomo_reward_idx = aug_reward.max(dim=1)
    optimal_solution = solutions[torch.arange(env.env_batch_size), max_pomo_reward_idx]
    return max_pomo_reward.float(), optimal_solution