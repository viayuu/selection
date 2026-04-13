import numpy as np
import torch
from itertools import combinations
from EasyNCO.utils.utils import getLogger
logger = getLogger(__name__)
class VrpNode(object):
	"""
	Class to represent each node for vehicle routing.
	"""
	def __init__(self, x, y, demand, px, py, capacity, dis, embedding=None):
		self.x = x
		self.y = y
		self.demand = demand
		self.px = px
		self.py = py
		self.capacity = capacity
		self.dis = dis
		if embedding is None:
			self.embedding = None
		else:
			self.embedding = embedding.copy()

class SeqManager(object):
	"""
	Base class for sequential input data. Can be used for vehicle routing.
	"""
	def __init__(self):
		self.nodes = []
		self.num_nodes = 0


	def get_node(self, idx):
		return self.nodes[idx]


class VrpManager(SeqManager):
	"""
	The class to maintain the state for vehicle routing.
	"""
	def __init__(self, capacity):
		super(VrpManager, self).__init__()
		self.capacity = capacity
		self.route = []
		self.vehicle_state = []
		self.tot_dis = []
		self.encoder_outputs = None


	def clone(self):
		res = VrpManager(self.capacity)
		res.nodes = []
		for i, node in enumerate(self.nodes):
			res.nodes.append(VrpNode(x=node.x, y=node.y, demand=node.demand, px=node.px, py=node.py, capacity=node.capacity, dis=node.dis, embedding=node.embedding))
		res.num_nodes = self.num_nodes
		res.route = self.route[:]
		res.vehicle_state = self.vehicle_state[:]
		res.tot_dis = self.tot_dis[:]
		res.encoder_outputs = self.encoder_outputs.clone()
		return res


	def get_dis(self, node_1, node_2):
		return np.sqrt((node_1.x - node_2.x) ** 2 + (node_1.y - node_2.y) ** 2)


	def get_neighbor_idxes(self, route_idx):
		neighbor_idxes = []
		route_node_idx = self.vehicle_state[route_idx][0]
		pre_node_idx, pre_capacity = self.vehicle_state[route_idx - 1]
		for i in range(1, len(self.vehicle_state) - 1):
			cur_node_idx = self.vehicle_state[i][0]
			if route_node_idx == cur_node_idx:
				continue
			if pre_node_idx == 0 and cur_node_idx == 0:
				continue
			cur_node = self.get_node(cur_node_idx)
			if route_node_idx == 0 and i > route_idx and cur_node.demand > pre_capacity:
				continue
			neighbor_idxes.append(i)
		return neighbor_idxes


	def add_route_node(self, node_idx):
		node = self.get_node(node_idx)
		if len(self.vehicle_state) == 0:
			pre_node_idx = 0
			pre_capacity = self.capacity
		else:
			pre_node_idx, pre_capacity = self.vehicle_state[-1]
		pre_node = self.get_node(pre_node_idx)
		if node_idx > 0:
			self.vehicle_state.append((node_idx, pre_capacity - self.nodes[node_idx].demand))
		else:
			self.vehicle_state.append((node_idx, self.capacity))
		cur_dis = self.get_dis(node, pre_node)
		if len(self.tot_dis) == 0:
			self.tot_dis.append(cur_dis)
		else:
			self.tot_dis.append(self.tot_dis[-1] + cur_dis)
		new_node = VrpNode(x=node.x, y=node.y, demand=node.demand, px=pre_node.x, py=pre_node.y, capacity=pre_capacity, dis=cur_dis)
		if new_node.capacity == 0:
			new_node.embedding = [new_node.x, new_node.y, new_node.demand * 1.0 / self.capacity, new_node.px, new_node.py, 0.0, new_node.dis]
		else:
			new_node.embedding = [new_node.x, new_node.y, new_node.demand * 1.0 / self.capacity, new_node.px, new_node.py, new_node.demand * 1.0 / new_node.capacity, new_node.dis]
		self.nodes[node_idx] = new_node
		self.route.append(new_node.embedding[:])


class vrpParser(object):
	def parse(self, problem, debug=False):
		self.is_debug = debug
		dm = VrpManager(problem['capacity'])
		dm.nodes.append(VrpNode(x=problem['depot'][0], y=problem['depot'][1], demand=0, px=problem['depot'][0], py=problem['depot'][1], capacity=problem['capacity'], dis=0.0))
		for customer in problem['customers']:
			dm.nodes.append(VrpNode(x=customer['position'][0], y=customer['position'][1], demand=customer['demand'], px=customer['position'][0], py=customer['position'][1], capacity=problem['capacity'], dis=0.0))
		dm.num_nodes = len(dm.nodes)
		cur_capacity = problem['capacity']
		pending_nodes = [i for i in range(0, dm.num_nodes)]
		dm.add_route_node(0)
		cur_capacity = dm.vehicle_state[-1][1]


		while len(pending_nodes) > 1:
			dis = []
			demands = []
			pre_node_idx = dm.vehicle_state[-1][0]
			pre_node = dm.get_node(pre_node_idx)
			for i in pending_nodes:
				cur_node = dm.get_node(i)
				dis.append(dm.get_dis(pre_node, cur_node))
				demands.append(cur_node.demand)
			for i in range(len(pending_nodes)):
				for j in range(i + 1, len(pending_nodes)):
					if dis[i] > dis[j] or dis[i] == dis[j] and demands[i] > demands[j]:
						pending_nodes[i], pending_nodes[j] = pending_nodes[j], pending_nodes[i]
						dis[i], dis[j] = dis[j], dis[i]
						demands[i], demands[j] = demands[j], demands[i]
			for i in pending_nodes:
				if i == 0:
					if cur_capacity == problem['capacity']:
						continue
					dm.add_route_node(0)
					break
				else:
					cur_node = dm.get_node(i)
					if cur_node.demand > cur_capacity:
						continue
					dm.add_route_node(i)
					pending_nodes.remove(i)
					break
			cur_capacity = dm.vehicle_state[-1][1]

		dm.add_route_node(0)
		return dm


def get_all_2opt(problem_size):
	'''
	返回所有2-opt点对以及交换后的路径序列

	对于cvrp问题，LIH将解的长度统一，如果 problem_size < 100 ,problem_extend = problem_size *2
	若 problem_size = 100, problem_extend = 128
	comb_num为 problem_extend取2的组合数

	return CL (comb_num,2) 所有2-opt点对  即(0,1),(0,2) ...
		   dic(comb_num,problem_extend)  交换后的路径序列 如(1,0,...,problem_extend)
	'''


	if problem_size < 100:
		o = torch.arange(problem_size * 2)  # 生成范围 [0, 1, ..., batch.size(1) - 1]
	elif problem_size == 100:
		o = torch.arange(128)
	# 生成所有两两组合
	CL = torch.tensor(list(combinations(o.tolist(), 2))).cuda()
	# 生成所有 2-opt 交换后的结果
	lis = []
	for c in CL:
		swapped = o.clone()
		swapped[c[0]:c[1] + 1] = torch.flip(swapped[c[0]:c[1] + 1], dims=[0])  # 翻转子区间
		lis.append(swapped)
	# 将所有结果堆叠起来
	dic = torch.stack(lis).cuda()
	return CL,dic

def lih_reset(problems,problem_size,batch_size):
	list_of_data = [
		{
			'depot': problems[batch_idx, 0].tolist(),
			'customers': [
				{
					'position': problems[batch_idx, i, :2].tolist(),
					'demand': problems[batch_idx, i, 2].item()
				}
				for i in range(1, problems.shape[1])
			],
			'capacity': 1
		}
		for batch_idx in range(problems.shape[0])
	]

	batch_data = []
	parser = vrpParser()
	for batch_idx in range(0, batch_size):
		problem = list_of_data[batch_idx]
		dm = parser.parse(problem)
		batch_data.append(dm)

	if problem_size <= 50:
		problem_size_extend = problem_size * 2
	elif problem_size == 100:
		problem_size_extend = 128
	data = [
		{
			'loc': torch.cat((torch.FloatTensor(batch_data[i].route)[:, 0:2],
							  torch.FloatTensor(batch_data[i].route)[0, 0:2].repeat(
								  problem_size_extend - len(batch_data[i].route),
								  1)), 0),
			'demand': torch.cat(
				(torch.FloatTensor(batch_data[i].route)[:, 2].float(),
				 torch.zeros(problem_size_extend - len(batch_data[i].route)).to("cpu")),
				0),
			'depot': torch.FloatTensor(batch_data[i].route)[0, 0:2]
		}
		for i in range(batch_size)
	]
	# batch,problem,2
	locs_extend = torch.cat([item['loc'] for item in data], dim=0).reshape(batch_size, -1, 2).to('cuda')
	demand_extend = torch.cat([item['demand'] for item in data], dim=0).reshape(batch_size, -1).to('cuda')
	depot_extend = torch.cat([item['depot'] for item in data], dim=0).reshape(batch_size, -1).to('cuda')
	batch_e = {
		'loc': locs_extend,
		'demand': demand_extend,
		'depot': depot_extend,
	}
	return batch_e
