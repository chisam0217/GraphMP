import math
from dataclasses import dataclass
from typing import Optional, List
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import Sequential as Seq, Linear as Lin, ReLU
from torch_scatter import scatter_mean, scatter_max, scatter_add
from torch_geometric.nn import voxel_grid, radius_graph
from torch_geometric.nn.pool.consecutive import consecutive_cluster
from torch_geometric.nn.pool import knn
from torch_geometric.utils import grid, add_self_loops, remove_self_loops, softmax
from torch_geometric.nn.conv import MessagePassing, GCNConv
from torch.nn import BatchNorm1d
from torch.autograd import Variable
from torch_geometric.nn import knn_graph, GraphConv
# from nets import ResConv, EdgePooling, ASAPooling, SAModule, FPModule, MLP, PointConv
from torch_sparse import coalesce
import math
from heuristic_encoder import Explorer
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


@dataclass
class AstarOutput:
	"""
	Output structure of A* search planners
	"""
	histories: torch.tensor
	paths: torch.tensor
	intermediate_results: Optional[List[dict]] = None


# def get_heuristic(loc: torch.tensor, goal_n: torch.tensor,
# 				  tb_factor: float = 0.001) -> torch.tensor:
# 	goal_loc = loc[goal_n]
# 	diff = loc[:,:2] - goal_loc[:2]
# 	h = torch.norm(diff, dim = 1)
# 	return h


#Manhattan + l2norm
# def get_heuristic(loc: torch.tensor, goal_n: torch.tensor,
# 				  tb_factor: float = 0.001) -> torch.tensor:
# 	goal_loc = loc[goal_n]
# 	diff = torch.abs(loc - goal_loc)
# 	h = torch.sum(diff, dim=1) + torch.norm(diff, dim = 1)
# 	return h

# #Manhattan
# def get_heuristic(loc: torch.tensor, goal_n: torch.tensor,
# 				  tb_factor: float = 0.001) -> torch.tensor:
# 	goal_loc = loc[goal_n]
# 	diff = torch.abs(loc - goal_loc)
# 	h = torch.sum(diff, dim=1)
# 	return h

def get_heuristic(loc: torch.tensor, goal_n: torch.tensor,
				  tb_factor: float = 0.001) -> torch.tensor:	
	# print ('loc', loc.size())
	goal_loc = loc[goal_n, :]
	# print ('goal_loc', goal_loc.size())
	diff = loc - goal_loc
	# print ('diff', diff.size())
	# h = torch.sum(diff, dim=1)
	return torch.norm(diff, dim=1)


#L2-norm
# def get_heuristic(loc: torch.tensor, goal_n: torch.tensor,
# 				  tb_factor: float = 0.001) -> torch.tensor:
# 	goal_loc = loc[goal_n]
# 	diff = loc - goal_loc
# 	h = torch.norm(diff, dim = 1)
# 	return h


def _st_softmax_noexp(val: torch.tensor) -> torch.tensor:
	"""
	Softmax + discretized activation
	Used a detach() trick as done in straight-through softmax

	Args:
		val (torch.tensor): exponential of inputs.

	Returns:
		torch.tensor: one-hot matrices for input argmax.
	"""

	val_ = val
	y = val_ / (val_.sum(dim=0, keepdim=True))
	_, ind = y.max(dim=-1)
	y_hard = torch.zeros_like(y)
	y_hard[ind] = 1
	# y_hard = y_hard.reshape_as(val)
	# y = y.reshape_as(val)
	return (y_hard - y).detach() + y



def backtrack(start_n: torch.tensor, goal_n: torch.tensor,
			  parents: torch.tensor, current_t: int) -> torch.tensor:
	"""
	Backtrack the search results to obtain paths
	Args:
		start_maps (torch.tensor): one-hot matrices for start locations
		goal_maps (torch.tensor): one-hot matrices for goal locations
		parents (torch.tensor): parent nodes
		current_t (int): current time step
	Returns:
		torch.tensor: solution paths
	"""

	parents = parents.long()
	# print ('parents', parents)
	goal_n = goal_n.type(torch.long)
	path_maps = goal_n.type(torch.long)

	loc = (parents * goal_n).sum(0)
	# path_order = [loc]
	# loc = loc.long()
	# print ('loc', loc)
	for _ in range(current_t):
		# loc = loc.long()
		# print ('the loc is', loc)
		path_maps[loc] = 1
		loc = parents[loc]
		# path_order.append(loc)
	return path_maps



def get_path(start_index, goal_index, parents, current_t):
	parents = parents.long()
	path = []
	loc = goal_index
	for _ in range(current_t):
		loc = parents[loc]
		path.insert(0, loc)
		if loc == start_index:
			break
	return path


class DifferentiableAstar(nn.Module):
	def __init__(self, g_ratio: float = 0.5, Tmax: float = 0.25):
		"""
		Differentiable A* module

		Args:
			g_ratio (float, optional): ratio between g(v) + h(v). Set 0 to perform as best-first search. Defaults to 0.5.
			Tmax (float, optional): how much of the map the planner explores during training. Defaults to 0.25.
		"""

		super().__init__()
		self.get_heuristic = get_heuristic

		self.g_ratio = g_ratio
		assert (Tmax > 0) & (Tmax <= 1), "Tmax must be within (0, 1]"
		self.Tmax = Tmax

	def forward(self, 
				start_index: torch.tensor,
				goal_index: torch.tensor,
				cost_maps: torch.tensor,
				nodes: torch.tensor,
				adj: torch.tensor,
				weighted_adj: torch.tensor,
				store_intermediate_results: bool = False) -> AstarOutput:
		adj.fill_diagonal_(0)
		weighted_adj[weighted_adj==float("inf")] = 0
		weighted_adj.fill_diagonal_(0)

		size = nodes.size(0)
		open_maps = torch.zeros(nodes.size(0)).to(device)
		open_maps[start_index] = 1

		start_maps = torch.zeros(nodes.size(0)).to(device)
		start_maps[start_index] = 1

		goal_maps = torch.zeros(nodes.size(0)).to(device)
		goal_maps[goal_index] = 1

		histories = torch.zeros(nodes.size(0)).to(device)
		intermediate_results = []

		h = cost_maps
		g = weighted_adj[start_index]

		parents = (torch.ones(nodes.size(0)).to(device) * goal_maps.max(-1, keepdim=True)[1]) #XXXX
		
		Tmax = self.Tmax if self.training else 1.
		Tmax = int(Tmax * size)
		# print ('Tmax', Tmax)

		old_parents = start_index

		for t in range(Tmax):
			# select the node that minimizes cost equation (3)
			f = self.g_ratio * g + (1 - self.g_ratio) * h
			f_exp = torch.exp(-1 * f /math.sqrt(cost_maps.shape[-1]))
			f_exp = f_exp * open_maps 

			#Important
			if not self.training:
				f_exp = f_exp * (1-histories) 
				if not torch.sum(f_exp):
					break

			selected_node_maps = _st_softmax_noexp(f_exp) 
			snm = selected_node_maps
			new_parents = snm.max(0, keepdim=True)[1]

			if store_intermediate_results:
					intermediate_results.append({
						"histories":
						histories.unsqueeze(1).detach(),
						"paths":
						selected_node_maps.unsqueeze(1).detach()
					})

			
			dist_to_goal = (selected_node_maps * goal_maps).sum(0)
			is_unsolved = (dist_to_goal < 1e-8).float()

			########################
			open_maps = open_maps - selected_node_maps
			open_maps = torch.clamp(open_maps, 0, 1) 
			########################
			histories = histories + selected_node_maps 
			histories = torch.clamp(histories, 0, 1)

			neighbor_nodes = torch.mm(torch.unsqueeze(selected_node_maps, 0), adj) * (1 - open_maps) * (1 - histories) #XXXX
			neighbor_nodes = torch.squeeze(neighbor_nodes, 0)

			# update g if one of the following conditions is met
			# 1) neighbor is not in the close list (1 - histories) nor in the open list (1 - open_maps)
			# 2) neighbor is in the open list but g < g2
			g2 = g[new_parents] + torch.squeeze(weighted_adj[new_parents], 0)

			idx = (1 - open_maps) * (1 - histories) + open_maps * (g > g2)
			idx = idx * neighbor_nodes
			idx = idx.detach()
			g = g2 * idx + g * (1 - idx)
			# g = (g[old_parents] + weighted_adj[old_parents, new_parents] + torch.squeeze(weighted_adj[new_parents], 0)) * idx + g * (1 - idx)
			g = g.detach()

			# update open maps
			open_maps = torch.clamp(open_maps + idx, 0, 1)
			open_maps = open_maps.detach()
			parents = new_parents * idx + parents * (1 - idx)

			old_parents = new_parents

			if torch.all(is_unsolved.flatten() == 0):
				# print ('path is found')
				break

		path_maps = backtrack(start_maps, goal_maps, parents, t) 
		if not self.training:
			ordered_path = get_path(start_index, goal_index, parents, t)

		if store_intermediate_results:
			intermediate_results.append({
				"histories":
				histories.unsqueeze(1).detach(),
				"paths":
				path_maps.unsqueeze(1).detach()
			})
		if not self.training:
			return open_maps, histories, path_maps, ordered_path
			# return histories, path_maps, ordered_path

		return histories, path_maps



class VanillaAstar(nn.Module):
	def __init__(
		self,
		g_ratio: float = 0.5,
		Tmax: float = 1,
		encoder_depth: int = 4,
		config_size=2, 
		embed_size=32, 
		obs_size=2
	):
		super().__init__()
		self.astar = DifferentiableAstar(
			g_ratio=g_ratio,
			Tmax=Tmax,
		)

	def forward(self,
				start_index: torch.tensor,
				goal_index: torch.tensor,
				nodes: torch.tensor,
				edge_idx: torch.tensor,
				edges: torch.tensor,
				weighted_edges: torch.tensor,
				loop: np.array,
				labels,
				store_intermediate_results: bool = False) -> AstarOutput:
		
		pred_cost_maps = get_heuristic(nodes, goal_index)

		astar_outputs = self.astar(start_index, goal_index, pred_cost_maps, nodes,
								   edges, weighted_edges, store_intermediate_results)

		return astar_outputs




class HeuristicNeuralAstar(nn.Module):
	def __init__(
		self,
		g_ratio: float = 0.5,
		Tmax: float = 0.25,
		encoder_depth: int = 4,
		config_size=2, 
		embed_size=128, 
		obs_size=2
	):
		"""
		Neural A* search

		Args:
			g_ratio (float, optional): ratio between g(v) + h(v). Set 0 to perform as best-first search. Defaults to 0.5.
			Tmax (float, optional): how much of the map the model explores during training. Defaults to 0.25.
		"""
		super().__init__()
		self.astar = DifferentiableAstar(
			g_ratio=g_ratio,
			Tmax=Tmax,
		)
		self.encoder = Explorer(config_size, embed_size, obs_size)

	def forward(self,
				start_index: torch.tensor,
				goal_index: torch.tensor,
				nodes: torch.tensor,
				edge_idx: torch.tensor,
				edges: torch.tensor,
				weighted_edges: torch.tensor,
				loop,
				labels,
				search_path: bool = True,
				store_intermediate_results: bool = False) -> AstarOutput:
		
		# pred_cost_maps = self.encoder(nodes, edge_idx, goal_index, loop, labels)
		pred_cost_maps = self.encoder(nodes, edge_idx, loop, labels)
		pred_cost_maps = torch.squeeze(pred_cost_maps, 1)

		if search_path:
			return self.astar(start_index, goal_index, pred_cost_maps, nodes,
								   edges, weighted_edges, store_intermediate_results)
		else:
			return pred_cost_maps







class DifferentiableAstar_backup(nn.Module):
	def __init__(self, g_ratio: float = 0.5, Tmax: float = 0.25):
		"""
		Differentiable A* module

		Args:
			g_ratio (float, optional): ratio between g(v) + h(v). Set 0 to perform as best-first search. Defaults to 0.5.
			Tmax (float, optional): how much of the map the planner explores during training. Defaults to 0.25.
		"""

		super().__init__()
		self.get_heuristic = get_heuristic

		self.g_ratio = g_ratio
		assert (Tmax > 0) & (Tmax <= 1), "Tmax must be within (0, 1]"
		self.Tmax = Tmax

	def forward(self, 
				start_index: torch.tensor,
				goal_index: torch.tensor,
				cost_maps: torch.tensor,
				nodes: torch.tensor,
				adj: torch.tensor,
				weighted_adj: torch.tensor,
				store_intermediate_results: bool = False) -> AstarOutput:
		
		adj.fill_diagonal_(0)
		weighted_adj[weighted_adj==float("inf")] = 0
		weighted_adj.fill_diagonal_(0)

		size = nodes.size(0)
		open_maps = torch.zeros(nodes.size(0)).to(device)
		open_maps[start_index] = 1

		start_maps = torch.zeros(nodes.size(0)).to(device)
		start_maps[start_index] = 1

		goal_maps = torch.zeros(nodes.size(0)).to(device)
		goal_maps[goal_index] = 1

		histories = torch.zeros(nodes.size(0)).to(device)
		intermediate_results = []

		h = cost_maps
		g = weighted_adj[start_index]

		parents = (torch.ones(nodes.size(0)).to(device) * goal_maps.max(-1, keepdim=True)[1]) #XXXX
		
		Tmax = self.Tmax if self.training else 1.
		Tmax = int(Tmax * size)
		# print ('Tmax', Tmax)

		old_parents = start_index

		for t in range(Tmax):
			# select the node that minimizes cost equation (3)
			f = self.g_ratio * g + (1 - self.g_ratio) * h
			f_exp = torch.exp(-1 * f /math.sqrt(cost_maps.shape[-1]))
			f_exp = f_exp * open_maps 

			#Important
			if not self.training:
				f_exp = f_exp * (1-histories) 
				if not torch.sum(f_exp):
					break

			selected_node_maps = _st_softmax_noexp(f_exp) 
			snm = selected_node_maps
			new_parents = snm.max(0, keepdim=True)[1]

			if store_intermediate_results:
					intermediate_results.append({
						"histories":
						histories.unsqueeze(1).detach(),
						"paths":
						selected_node_maps.unsqueeze(1).detach()
					})

			
			dist_to_goal = (selected_node_maps * goal_maps).sum(0)
			is_unsolved = (dist_to_goal < 1e-8).float()
			histories = histories + selected_node_maps 
			histories = torch.clamp(histories, 0, 1)

			neighbor_nodes = torch.mm(torch.unsqueeze(selected_node_maps, 0), adj)
			neighbor_nodes = torch.squeeze(neighbor_nodes, 0)

			# update g if one of the following conditions is met
			# 1) neighbor is not in the close list (1 - histories) nor in the open list (1 - open_maps)
			# 2) neighbor is in the open list but g < g2
			g2 = g[new_parents] + torch.squeeze(weighted_adj[new_parents], 0)

			idx = (1 - open_maps) * (1 - histories) + open_maps * (g > g2)
			idx = idx * neighbor_nodes
			idx = idx.detach()
			g = g2 * idx + g * (1 - idx)
			# g = (g[old_parents] + weighted_adj[old_parents, new_parents] + torch.squeeze(weighted_adj[new_parents], 0)) * idx + g * (1 - idx)
			g = g.detach()

			# update open maps
			open_maps = torch.clamp(open_maps + idx, 0, 1)
			open_maps = open_maps.detach()
			parents = new_parents * idx + parents * (1 - idx)

			old_parents = new_parents

			if torch.all(is_unsolved.flatten() == 0):
				break

		path_maps = backtrack(start_maps, goal_maps, parents, t) 
		if not self.training:
			ordered_path = get_path(start_index, goal_index, parents, t)

		if store_intermediate_results:
			intermediate_results.append({
				"histories":
				histories.unsqueeze(1).detach(),
				"paths":
				path_maps.unsqueeze(1).detach()
			})
		if not self.training:
			return histories, path_maps, ordered_path

		return histories, path_maps

