import torch
import numpy as np
from torch.distributions import Exponential

def augment_pomo(**kwargs):
    # problems.shape: (batch, problem, 2)
    x = kwargs['problems'][:, :, [0]]
    y = kwargs['problems'][:, :, [1]]
    # x,y shape: (batch, problem, 1)
    aug_factor=kwargs['aug_factor']
    assert aug_factor in [1, 4, 6, 8], f"aug_factor should be in [1, 4, 6, 8], but got {aug_factor}"
    dat1 = torch.cat((x, y), dim=2)
    dat2 = torch.cat((1 - x, y), dim=2)
    dat3 = torch.cat((x, 1 - y), dim=2)
    dat4 = torch.cat((1 - x, 1 - y), dim=2)
    dat5 = torch.cat((y, x), dim=2)
    dat6 = torch.cat((1 - y, x), dim=2)
    dat7 = torch.cat((y, 1 - x), dim=2)
    dat8 = torch.cat((1 - y, 1 - x), dim=2)
    if aug_factor==1:
        aug_problems = dat1
    elif aug_factor==4:
        aug_problems = torch.cat((dat1, dat2, dat3, dat4), dim=0)
    elif aug_factor==6:
        aug_problems = torch.cat((dat1, dat2, dat3, dat4,dat5,dat6), dim=0)
    else:
        aug_problems = torch.cat((dat1, dat2, dat3, dat4, dat5, dat6, dat7, dat8), dim=0)
    # shape: (8*batch, problem, 2)

    return aug_problems

def augment_rotate(**kwargs):
    batch_size = kwargs['problems'].shape[0]
    x = kwargs['problems'][:, :, [0]].repeat(kwargs['aug_factor'], 1, 1)
    y = kwargs['problems'][:, :, [1]].repeat(kwargs['aug_factor'], 1, 1)

    theta = torch.rand((batch_size * kwargs['aug_factor'], 1, 1)) * 2 * torch.pi
    # Let the nodes' coordinates be positive
    temp_x = x * torch.cos(theta) - y * torch.sin(theta) + 2
    temp_y = x * torch.sin(theta) + y * torch.cos(theta) + 2
    temp_problems = torch.cat((temp_x, temp_y), dim=-1)
    new_problems = coordinate_norm(temp_problems)

    return new_problems

def augment_reflect(**kwargs):
    batch_size = kwargs['problems'].shape[0]
    x = kwargs['problems'][:, :, [0]].repeat(kwargs['aug_factor'], 1, 1)
    y = kwargs['problems'][:, :, [1]].repeat(kwargs['aug_factor'], 1, 1)

    theta = torch.rand((batch_size * kwargs['aug_factor'], 1, 1)) * 2 * torch.pi
    # Let the nodes' coordinates be positive
    temp_x = x * torch.cos(2 * theta) + y * torch.sin(2 * theta) + 2
    temp_y = x * torch.sin(2 * theta) - y * torch.cos(2 * theta) + 2
    temp_problems = torch.cat((temp_x, temp_y), dim=-1)
    new_problems = coordinate_norm(temp_problems)

    return new_problems

def augment_mix(**kwargs):
    problems = kwargs['problems']
    rotate_factor = int(kwargs['aug_factor'] * kwargs['mix_prop'])
    reflect_factor = int(kwargs['aug_factor'] * (1 - kwargs['mix_prop']))
    rotate_problems = augment_rotate(problems=problems, aug_factor=rotate_factor)
    reflect_problems = augment_reflect(problems=problems, aug_factor=reflect_factor)
    new_problems = torch.cat((rotate_problems, reflect_problems), dim=0)

    return new_problems
def augment_noise(**kwargs):
    problems = kwargs['problems'].repeat(kwargs['aug_factor'], 1, 1)
    noise = torch.rand(problems.size()) * 1e-5
    new_problems = problems + noise

    return new_problems

def augment_xy_data_by_64_fold_2obj(**kwargs):
    xy_data = kwargs['problems']
    x1 = xy_data[:, :, [0]]
    y1 = xy_data[:, :, [1]]
    x2 = xy_data[:, :, [2]]
    y2 = xy_data[:, :, [3]]

    dat1 = {}
    dat2 = {}

    dat_aug = []

    dat1[0] = torch.cat((x1, y1), dim=2)
    dat1[1] = torch.cat((1 - x1, y1), dim=2)
    dat1[2] = torch.cat((x1, 1 - y1), dim=2)
    dat1[3] = torch.cat((1 - x1, 1 - y1), dim=2)
    dat1[4] = torch.cat((y1, x1), dim=2)
    dat1[5] = torch.cat((1 - y1, x1), dim=2)
    dat1[6] = torch.cat((y1, 1 - x1), dim=2)
    dat1[7] = torch.cat((1 - y1, 1 - x1), dim=2)

    dat2[0] = torch.cat((x2, y2), dim=2)
    dat2[1] = torch.cat((1 - x2, y2), dim=2)
    dat2[2] = torch.cat((x2, 1 - y2), dim=2)
    dat2[3] = torch.cat((1 - x2, 1 - y2), dim=2)
    dat2[4] = torch.cat((y2, x2), dim=2)
    dat2[5] = torch.cat((1 - y2, x2), dim=2)
    dat2[6] = torch.cat((y2, 1 - x2), dim=2)
    dat2[7] = torch.cat((1 - y2, 1 - x2), dim=2)

    for i in range(8):
        for j in range(8):
            dat = torch.cat((dat1[i], dat2[j]), dim=2)
            dat_aug.append(dat)

    aug_problems = torch.cat(dat_aug, dim=0)

    return aug_problems


def augment_xy_data_by_n_fold_3obj(**kwargs):
    xy_data = kwargs['problems']
    size = kwargs['aug_factor']

    x1 = xy_data[:, :, [0]]
    y1 = xy_data[:, :, [1]]
    x2 = xy_data[:, :, [2]]
    y2 = xy_data[:, :, [3]]
    x3 = xy_data[:, :, [4]]
    y3 = xy_data[:, :, [5]]

    dat1 = {}
    dat2 = {}
    dat3 = {}

    dat_aug = []

    dat1[0] = torch.cat((x1, y1), dim=2)
    dat1[1] = torch.cat((1 - x1, y1), dim=2)
    dat1[2] = torch.cat((x1, 1 - y1), dim=2)
    dat1[3] = torch.cat((1 - x1, 1 - y1), dim=2)
    dat1[4] = torch.cat((y1, x1), dim=2)
    dat1[5] = torch.cat((1 - y1, x1), dim=2)
    dat1[6] = torch.cat((y1, 1 - x1), dim=2)
    dat1[7] = torch.cat((1 - y1, 1 - x1), dim=2)

    dat2[0] = torch.cat((x2, y2), dim=2)
    dat2[1] = torch.cat((1 - x2, y2), dim=2)
    dat2[2] = torch.cat((x2, 1 - y2), dim=2)
    dat2[3] = torch.cat((1 - x2, 1 - y2), dim=2)
    dat2[4] = torch.cat((y2, x2), dim=2)
    dat2[5] = torch.cat((1 - y2, x2), dim=2)
    dat2[6] = torch.cat((y2, 1 - x2), dim=2)
    dat2[7] = torch.cat((1 - y2, 1 - x2), dim=2)

    dat3[0] = torch.cat((x3, y3), dim=2)
    dat3[1] = torch.cat((1 - x3, y3), dim=2)
    dat3[2] = torch.cat((x3, 1 - y3), dim=2)
    dat3[3] = torch.cat((1 - x3, 1 - y3), dim=2)
    dat3[4] = torch.cat((y3, x3), dim=2)
    dat3[5] = torch.cat((1 - y3, x3), dim=2)
    dat3[6] = torch.cat((y3, 1 - x3), dim=2)
    dat3[7] = torch.cat((1 - y3, 1 - x3), dim=2)

    all_idx = [[i, j, k] for i in range(8) for j in range(8) for k in range(8)]
    item_list = list(range(1, 512))
    np.random.shuffle(item_list)
    item_list = [0] + item_list

    for i in range(size):
        idx = all_idx[item_list[i]]
        dat = torch.cat((dat1[idx[0]], dat2[idx[1]], dat3[idx[2]]), dim=2)
        dat_aug.append(dat)
    aug_problems = torch.cat(dat_aug, dim=0)

    return aug_problems

def coordinate_norm(problems):
    xy_min = problems.min(dim=1)[0]
    xy_max = problems.max(dim=1)[0]
    ratio = torch.max((xy_max - xy_min), dim=-1)[0].reshape((-1, 1, 1))
    ratio[ratio == 0] = 1
    new_problems = (problems-xy_min[:, None, :].expand(-1, problems.shape[1], -1)) / ratio

    return new_problems
def glop_coordinate_transformation(x):
    '''
    glop的坐标变换，为了维持子问题的一致性

    '''
    input = x.clone()
    max_x, indices_max_x = input[:, :, 0].max(dim=1)
    max_y, indices_max_y = input[:, :, 1].max(dim=1)
    min_x, indices_min_x = input[:, :, 0].min(dim=1)
    min_y, indices_min_y = input[:, :, 1].min(dim=1)
    # shapes: (batch_size, ); (batch_size, )

    diff_x = max_x - min_x
    diff_y = max_y - min_y
    xy_exchanged = diff_y > diff_x

    # shift to zero
    input[:, :, 0] -= (min_x).unsqueeze(-1)
    input[:, :, 1] -= (min_y).unsqueeze(-1)

    # exchange coordinates for those diff_y > diff_x
    input[xy_exchanged, :, 0], input[xy_exchanged, :, 1] = input[xy_exchanged, :, 1], input[xy_exchanged, :, 0]

    # scale to (0, 1)
    scale_degree = torch.max(diff_x, diff_y)
    scale_degree = scale_degree.view(input.shape[0], 1, 1)
    if x.size(-1) == 3:
        input[:, :, :2] /= scale_degree + 1e-10
    else:
        input /= scale_degree + 1e-10
    return input


# data generation from omni
def get_gaussian_mixture(graph_size=100, num_modes=0, cdist=1):
    '''
    GMM create one instance of TSP-100, using cdist
    num_modes is used to specify the number of mixed Gaussian distributions
    the generated points range from [0,cdist]
    '''
    from sklearn.preprocessing import MinMaxScaler
    nums = np.random.multinomial(graph_size, np.ones(num_modes) / num_modes)
    xy = []
    for num in nums:
        center = np.random.uniform(0, cdist, size=(1, 2))
        nxy = np.random.multivariate_normal(mean=center.squeeze(), cov=np.eye(2, 2), size=(num,))
        xy.extend(nxy)
    xy = np.array(xy)
    xy = MinMaxScaler().fit_transform(xy)
    return torch.Tensor(xy)

def generate_by_distribution(num_nodes, distribution):
    if distribution == "uniform_rectangle":
        width = np.random.uniform(0, 1)
        x1 = np.random.uniform(0, 1, [num_nodes, 1])
        x2 = np.random.uniform(0.5 - width / 2, 0.5 + width / 2, [num_nodes, 1])
        if np.random.randint(2) == 0:
            data = np.concatenate([x1, x2], 1)
        else:
            data = np.concatenate([x2, x1], 1)
        node_xy_data = torch.Tensor(data)
    elif distribution == "gaussian":
        mean = [0.5, 0.5]
        cov = np.random.uniform(0, 1)
        cov = [[1.0, cov], [cov, 1.0]]
        data = np.random.multivariate_normal(mean, cov, [num_nodes])
        node_xy_data = torch.Tensor(data)
    elif distribution == "cluster":
        loc = []
        n_cluster = np.random.randint(low=2, high=9)
        loc.append(np.random.randint(1000, size=[1, n_cluster, 2]))
        prob = np.zeros((1000, 1000))
        coord = np.concatenate([np.tile(np.arange(1000).reshape(-1, 1, 1), [1, 1000, 1]),
                                np.tile(np.arange(1000).reshape(1, -1, 1), [1000, 1, 1])], -1)
        for j in range(n_cluster):
            dist = np.sqrt(np.sum((coord - loc[-1][0, j, :]) ** 2, -1))
            dist = np.exp(-dist / 40)
            prob += dist
        for j in range(n_cluster):
            prob[loc[-1][0, j, 0], loc[-1][0, j, 1]] = 0
        prob = prob / prob.sum()
        index = np.random.choice(1000000, num_nodes - n_cluster, replace=False, p=prob.reshape(-1))
        coord = coord[index // 1000, index % 1000]
        loc.append(coord.reshape(1, -1, 2))
        loc = np.concatenate(loc, 1)
        node_xy_data = torch.Tensor(loc).squeeze() / 1000

    elif distribution == "diagonal":
        x = np.random.uniform(low=0, high=1, size=(1, num_nodes, 1))
        r = np.random.uniform(low=0, high=1)
        if np.random.randint(4) == 0:
            x = np.concatenate([x, x * r + (1 - r) / 2], 2)
        elif np.random.randint(4) == 1:
            x = np.concatenate([x, (1 - x) * r + (1 - r) / 2], 2)
        elif np.random.randint(4) == 2:
            x = np.concatenate([x * r + (1 - r) / 2, x], 2)
        else:
            x = np.concatenate([(1 - x) * r + (1 - r) / 2, x], 2)
        width = np.random.uniform(low=0.05, high=0.2)
        x += np.random.uniform(low=-width / 2, high=width / 2, size=(1, num_nodes, 2))
        node_xy_data = torch.Tensor(x).squeeze()

    if distribution != "uniform_rectangle":
        # Normalize to [0,1] and shift the data to center it
        xy_min, _ = node_xy_data.min(dim=0, keepdim=True)
        xy_max, _ = node_xy_data.max(dim=0, keepdim=True)
        node_xy_data = node_xy_data - xy_min
        node_xy_data = node_xy_data / (xy_max - xy_min).amax(dim=-1, keepdim=True)
        node_xy_data = node_xy_data + (1 - node_xy_data.max(dim=0, keepdim=True)[0]) / 2

    return node_xy_data


# data generation from Invit
def normalize_to_unit_board(node_coord):
    """
    normalize a tsp instance to a [0, 1]^2 unit board, prefer to have points on both x=0 and y=0
    :param tsp_instance: a (num_nodes, 2) tensor
    :return: a (num_nodes, 2) tensor, a normalized tsp instance
    """
    normalized_instance = node_coord.clone()
    normalization_factor = (normalized_instance.max(dim=0).values - normalized_instance.min(dim=0).values).max()
    normalized_instance = (normalized_instance - normalized_instance.min(dim=0).values) / normalization_factor
    return normalized_instance, normalization_factor

def unified_min_max_normalize(node_coord):
    # normalize data to [0,1] using min-max normalization
    ################################################################
    max_value = node_coord.max()
    min_value = node_coord.min()
    normalization_factor = max_value - min_value
    nodes_xy_normalized = (node_coord - min_value) / normalization_factor

    return nodes_xy_normalized, normalization_factor


def generate_explosion_instance(num_nodes, range_min=0.1, range_max=0.5, rate=10,depot_xy=None):
    """
    first generate uniformly i.i.d. coordinates of data points
    select an explosion center and expel all data points in a range
    :param num_nodes: the number of nodes for tsp instances
    :param range_min: the minimum range of explosion
    :param range_max: the maximum range of explosion
    :param rate: rate of exponential distribution, for random extra movement our of range
    :return: a (num_nodes, 2) tensor, a tsp instance following explosion distribution
    """
    tsp_instance = torch.rand((num_nodes, 2))
    explosion_center = torch.rand(2, )
    pointer_vector = tsp_instance - explosion_center
    explosion_range = (range_max - range_min) * torch.rand((1, )) + range_min #爆炸半径,用于判断哪些点会受影响
    exploded = pointer_vector.norm(dim=1) < explosion_range   #判断哪些点会受影响
    explosion_factor = explosion_range + Exponential(rate=rate).sample((num_nodes, 1)) #被炸出去的距离
    directional_vector = pointer_vector / pointer_vector.norm(dim=1).unsqueeze(dim=1) #方向向量，炸出去的方向
    explosion_movement = directional_vector * explosion_factor
    tsp_instance[exploded] = explosion_center + explosion_movement[exploded]
    if depot_xy is not None:
        tsp_instance = torch.cat((depot_xy,tsp_instance), dim=0)
        result, _ = normalize_to_unit_board(tsp_instance)
        return result[0].unsqueeze(dim=0),result[1:]
    else:
        result, _ = normalize_to_unit_board(tsp_instance)
        return result

def generate_implosion_instance(num_nodes, range_min=0.1, range_max=0.5,depot_xy=None):
    """
    first generate uniformly i.i.d. coordinates of data points
    select an implosion center and attracts all data points in a range
    :param num_nodes: the number of nodes for tsp instances
    :param range_min: the minimum range of implosion
    :param range_max: the maximum range of implosion
    :return: a (num_nodes, 2) tensor, a tsp instance following implosion distribution
    """
    tsp_instance = torch.rand((num_nodes, 2))
    implosion_center = torch.rand(2, )
    pointer_vector = tsp_instance - implosion_center
    implosion_range = (range_max - range_min) * torch.rand((1, )) + range_min
    imploded = pointer_vector.norm(dim=1) < implosion_range
    implosion_factor = min(implosion_range, torch.normal(0, 1, (1, ))) #坍塌的距离
    implosion_movement = pointer_vector * implosion_factor
    tsp_instance[imploded] = implosion_center + implosion_movement[imploded]
    if depot_xy is not None:
        tsp_instance = torch.cat((depot_xy,tsp_instance), dim=0)
        result, _ = normalize_to_unit_board(tsp_instance)
        return result[0].unsqueeze(dim=0),result[1:]
    else:
        result, _ = normalize_to_unit_board(tsp_instance)
        return result
