#Verison 1.
import torch
import numpy as np
from torch_geometric.data import Data
from tqdm import tqdm as tqdm
import pickle
from time import time
from algorithm.dijkstra import dijkstra
from diff_astar import HeuristicNeuralAstar
from collision_net import CollisionNet
import torch.nn as nn
from environment import MazeEnv
from torch_geometric.nn import knn_graph
from torch_sparse import coalesce
from collision_net import CollisionNet
import argparse
import random
from str2name import str2name
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


parser = argparse.ArgumentParser()
parser.add_argument('--is_training', action='store_true', default=True, help='Training or Testing.')
parser.add_argument('--seed', type=int, default=123, help='Random seed.')
parser.add_argument('--lr', type=float, default=1e-3, help='Initial learning rate.')#1e-3
parser.add_argument('--weight_decay', type=float, default=0, help='Weight decay (L2 loss on parameters).')#
parser.add_argument('--epoch', type=int, default=500, help='The number of epochs.')
parser.add_argument('--batch_size', type=int, default=16, help='The batch size.') #8
parser.add_argument('--hidden', type=int, default=128, help='Number of hidden units.')
parser.add_argument('--schedule', type=bool, default=False, help='Whether to turn on optimizer scheduler.') #True
parser.add_argument('--finetune', type=bool, default=False, help='Whether to finetune the model.')

parser.add_argument('--robot', type=str, default="ur5", help='[maze2_easy, maze2_hard, ur5, snake7, kuka7, kuka13, kuka14]')
parser.add_argument('--n_sample', type=str, default='random', help='The number of samples.')
parser.add_argument('--k', type=str, default='random', help='The number of neighbors.')
parser.add_argument('--theta', type=float, default=0.8, help='The threshold of in-search collision check.')

args = parser.parse_args()
def set_random_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
set_random_seed(args.seed)
INFINITY = float('inf')

env, model_astar, model_astar_path, model_coll, model_coll_path, _ = str2name(args.robot, args.n_sample, args.k)
model_astar.load_state_dict(torch.load(model_astar_path, map_location=device))
model_coll.load_state_dict(torch.load(model_coll_path, map_location=device))

model_astar = model_astar.to(device)
model_coll = model_coll.to(device)


def construct_graph(env, points, k=5, check_collision=True):
    points = np.array(points)
    edge_index = knn_graph(torch.FloatTensor(points), k=k, loop=True)
    edge_index = torch.cat((edge_index, edge_index.flip(0)), dim=-1)
    edge_index_torch, _ = coalesce(edge_index, None, len(points), len(points))
    
    edge_index = edge_index_torch.data.cpu().numpy().T
    points1 = points[edge_index[:, 0]]
    points2 = points[edge_index[:, 1]]
    edge_cost = np.linalg.norm(points1 - points2, axis = 1)
    return edge_cost, edge_index, points


    
def backtrack(start_n, goal_n, prev_node, num_nodes):
    # print ('start_n', start_n)
    # print ('goal_n', goal_n)
    path = torch.zeros(num_nodes).to(device)
    path[goal_n] = 1
    path[start_n] = 1
    loc = start_n
    path_order = [loc]
    while loc != goal_n:
        # print ('loc', loc)
        loc = prev_node[loc]
        path[loc] = 1
        path_order.append(loc)
    return path, path_order


def compute_pathcost(path, states):
    path = path.astype(np.int)
    cost = 0
    for i in range(0, path.shape[0]-1):
        cost += np.linalg.norm(states[path[i]] - states[path[i+1]])
    return cost
        

set_random_seed(1234)

model_astar.eval()
model_coll.eval()
# criterion = nn.L1Loss()
success = 0
n_tasks = 0
# indexes = np.random.permutation(epoch)


total_history = 0
total_AstarPath = 0
total_OraclePath = 0
total_pathcost_astar = []
total_pathcost_oracle = 0

plan_time = 0
oracle_time = 0

indexes = np.arange(len(env.problems))[-1000:]
pbar = tqdm(indexes)

for index in pbar:
    pb = env.init_new_problem(index)

    n_sample = 300
    n_neighbor = 10

    env.init_new_problem(index)
    points = env.sample_n_points(n=n_sample-2)
    all_states = np.copy(points)
    points.insert(0, env.init_state)
    points.insert(0, env.goal_state)
    start_index = 1
    goal_index = 0
    edge_cost, edge_index, points = construct_graph(env, points, n_neighbor)
    edge_cost = torch.FloatTensor(edge_cost)

    edge_index = torch.LongTensor(edge_index.T)
    edge_index = edge_index.to(device)

    points = torch.FloatTensor(points).to(device)
    obs = torch.FloatTensor(env.obstacles).to(device)

    pred_edges = model_coll(points, edge_index, obs)
    pred_free = pred_edges[:, 1] > 0.5 # Edges that are predicted as collision-free

    edge_free_index = edge_index[:, pred_free] # Retrieve collision-free edges

    node_free = torch.zeros(points.size(0), points.size(0))
    node_free[edge_free_index[0, :], edge_free_index[1, :]] = 1.0

    edge_free_cost = edge_cost[pred_free].to(device) # Retrieve the weights of collision-free edges
    cost_maps = torch.zeros(points.size(0), points.size(0)).to(device)
    cost_maps[edge_free_index[0, :], edge_free_index[1, :]] = edge_free_cost
    
    node_free = node_free.to(device)
    cost_maps = cost_maps.to(device)
    edge_index = edge_index.to(device)

    labels = torch.zeros(len(points), 2)
    labels[:, 0] = 1 #nodes in the free space
    labels[goal_index, 0] = 0
    labels[goal_index, 1] = 1 #goal node
    labels = labels.to(device)

    current_loop = 3

    open_list, pred_history, pred_path, pred_ordered_path = model_astar(start_index, goal_index, points, edge_free_index, node_free, cost_maps, current_loop, labels)
    if pred_ordered_path: #if the results are not empty
        pred_ordered_path = torch.stack(pred_ordered_path).cpu().detach().numpy()
    pred_ordered_path = np.concatenate((pred_ordered_path, np.array([goal_index])), axis=0) 
    print ('The computed path is: ', pred_ordered_path)

    if pred_path[start_index]:
        path_cost_astar = compute_pathcost(pred_ordered_path, all_states)
        total_pathcost_astar.append(path_cost_astar)

print ('The average path cost is: ', sum(total_pathcost_astar)/len(total_pathcost_astar))