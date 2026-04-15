

def get_best_solution(reward, selected_node_list,batch_size):
    # reward.shape: (batch, n_start)
    # selected_node_list.shape: (batch, n_start, n_node)

    if reward[0,0] > 0:
        reward = - reward

    pomo_size = selected_node_list.shape[1]
    aug_factor = reward.shape[0] // batch_size
    assert aug_factor * batch_size == reward.shape[0], "Augmentation factor does not match batch size."
    aug_reward = reward.reshape(aug_factor, batch_size, pomo_size)
    # shape: (augmentation, batch, pomo)
    max_pomo_reward, max_pomo_reward_index = aug_reward.max(dim=2)  # get best results from pomo
    # shape: (augmentation, batch)
    max_aug_pomo_reward, max_aug_pomo_reward_index = max_pomo_reward.max(dim=0)  # get best results from augmentation
    # shape: (batch,)

    # get best sequence
    ###############################################
    selected_node_list = selected_node_list.reshape(aug_factor, batch_size, pomo_size, -1)
    # shape: (1, batch, pomo, problem)
    sequence_length = selected_node_list.shape[-1]
    max_pomo_reward_node_list = selected_node_list.gather(dim=2, index=max_pomo_reward_index[:, :, None, None].
                                                          expand(-1, -1, -1, sequence_length)).squeeze(2).transpose(0, 1)
    # shape: (batch,aug,problem)
    best_aug_pomo_selected_node_list = max_pomo_reward_node_list.gather(dim=1,
                                     index=max_aug_pomo_reward_index[:, None, None].expand(-1, -1, sequence_length))
    # shape: (batch,1,sequence)

    max_aug_pomo_reward = - max_aug_pomo_reward.unsqueeze(1)
    # shape: (batch,1)
    return max_aug_pomo_reward, best_aug_pomo_selected_node_list