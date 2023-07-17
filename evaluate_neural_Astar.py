import torch
import numpy as np
from torch_geometric.data import Data
from tqdm import tqdm as tqdm
from tensorboardX import SummaryWriter
import pickle
from time import time
from algorithm.dijkstra import dijkstra
from diff_astar import VanillaAstar, HeuristicNeuralAstar
import torch.nn as nn
from environment import KukaEnv, MazeEnv, SnakeEnv, Kuka2Env
from torch_geometric.nn import knn_graph
from torch_sparse import coalesce
import random
import argparse
from str2name import str2name
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")


def construct_graph(env, points, k=5):
    edge_index = knn_graph(torch.FloatTensor(points), k=k, loop=True)
    edge_index = torch.cat((edge_index, edge_index.flip(0)), dim=-1)
    edge_index_torch, _ = coalesce(edge_index, None, len(points), len(points))
    points = np.array(points)
    edge_index = edge_index_torch.data.cpu().numpy().T
    edge_cost = np.linalg.norm(points[edge_index[:, 1]]-points[edge_index[:, 0]], axis=1) 
    return points, edge_cost, edge_index


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

def lvc(path, environment, states):
    for i in range(0,len(path)-1):
        for j in range(len(path)-1,i+1,-1):
            if environment._edge_fp(states[path[i]], states[path[j]]):
                pc=[]
                for k in range(0,i+1):
                    pc.append(path[k])
                for k in range(j,len(path)):
                    pc.append(path[k])

                return lvc(pc, environment, states)
                
    return path
    

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=1234, help='Random seed.')
parser.add_argument('--hidden', type=int, default=128, help='Number of hidden units.')

parser.add_argument('--model_name', type=str, default="NeuralAstar", help='VanillaAstar/VanillaAstar')
parser.add_argument('--robot', type=str, default="maze2", help='The robot system.')
parser.add_argument('--n_sample', type=int, default=300, help='The number of samples.')
parser.add_argument('--k', type=int, default=20, help='The number of neighbors.')


args = parser.parse_args()
def set_random_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
set_random_seed(args.seed)
INFINITY = float('inf')
writer = SummaryWriter()

env, model_astar, weights_astar, model_coll, weights_coll, data_path = str2name(args.robot, args.n_sample, args.k)
# weights_name = "Astar_weights/{}_{}_{}.pt".format(args.robot, args.n_sample, args.k)
model_astar = model_astar.to(device)
model_coll = model_coll.to(device)

# weights_astar = "Astar_weights/{}_{}_{}_batch64.pt".format(args.robot, args.n_sample, args.k)
model_astar.load_state_dict(torch.load(weights_astar, map_location=device))
model_coll.load_state_dict(torch.load(weights_coll, map_location=device))

model_astar.eval()
model_coll.eval()

with open(data_path, 'rb') as f:
    graphs = pickle.load(f)



success = 0
n_tasks = 0
# indexes = np.random.permutation(epoch)
# indexes = np.arange(epoch)
indexes = np.arange(2000, 3000)
pbar = tqdm(indexes)

total_history = 0
total_AstarPath = 0
total_OraclePath = 0
total_pathcost_astar = 0
total_pathcost_oracle = 0

oracle_success = 0
all_plan_time = 0
all_construct_time = 0
all_search_time = 0

for index in pbar:
    s_time0 = time()
    env.init_new_problem(index)
    points = env.sample_n_points(n=args.n_sample-2)
    points.insert(0, env.init_state)
    points.insert(0, env.goal_state)
    # points, neighbors, edge_cost, edge_index, edge_free, fake_edge_cost = graphs[index]
    points, edge_cost, edge_index = construct_graph(env, points, args.k)
    all_states = np.copy(points)
    points = torch.FloatTensor(points).to(device)
    obs = torch.FloatTensor(env.obstacles).to(device)

    start_index = 1
    goal_index = 0

    edge_index = torch.LongTensor(edge_index.T)
    edge_index = edge_index.to(device)

    #vanilla A star changes the following edge_cost and function construct_graph
    cost_maps = torch.zeros(points.size(0), points.size(0))
    cost_maps[edge_index[0, :], edge_index[1, :]] = torch.FloatTensor(edge_cost).squeeze() 
    cost_maps = cost_maps.to(device)


    pred_edges = model_coll(points, edge_index, obs)
    # pred_edges = torch.argmax(pred_edges, dim=1)
    pred_edges = pred_edges[:,1]>0.5
    # correct = np.sum(pred_edges.detach().cpu().numpy() == np.array(edge_free))
    pred_adj_free = torch.zeros(len(points), len(points)).to(device)
    pred_adj_free[edge_index[0, :], edge_index[1, :]] = pred_edges.float()#.squeeze()
    construct_time = time() - s_time0

    ###### A star part
    s_time1 = time()

    # labels = torch.zeros(len(points), 2)
    # labels[:, 0] = 1 #nodes in the free space
    # labels[goal_index, 0] = 0 #goal node
    # labels[goal_index, 1] = 1 #goal node
    # labels = labels.to(device)

    current_loop = 5
    pred_history, pred_path, pred_ordered_path = \
            model_astar(start_index, goal_index, points, edge_index, pred_adj_free, cost_maps, current_loop)
        
    if pred_ordered_path: #if the results are not empty
        pred_ordered_path = torch.stack(pred_ordered_path).cpu().detach().numpy()
    pred_ordered_path = np.concatenate((pred_ordered_path, np.array([goal_index])), axis=0)

    for k in range(10):
        collision_checks = True
        # if not (start_index == goal_index).all():
        if pred_ordered_path.shape[0] > 1 and pred_path[start_index]: #the start is different from goal
            path_states = all_states[pred_ordered_path]
        
            for n in range(pred_ordered_path.shape[0]-1):
                if not env._edge_fp(path_states[n], path_states[n+1]):
                    # print ('Collision between {} and {} happens in {}th iteration'.format(path_states[n], path_states[n+1], n))
                    collision_checks = False
                    break
        else:
            break

        if collision_checks: 
            break
        else:
            pred_adj_free[pred_ordered_path[n], pred_ordered_path[n+1]] = 0
            pred_adj_free[pred_ordered_path[n+1], pred_ordered_path[n]] = 0
            pred_history, pred_path, pred_ordered_path = \
                model_astar(start_index, goal_index, points, edge_index, pred_adj_free, cost_maps, current_loop)

            pred_ordered_path = torch.stack(pred_ordered_path).cpu().detach().numpy()
            pred_ordered_path = np.concatenate((pred_ordered_path, np.array([goal_index])), axis=0)
    
    
    search_time = time() - s_time1

    if pred_path[start_index] and collision_checks:
        rewired_path = np.array(lvc(pred_ordered_path, env, all_states))
        success += 1
        # print ('success!!')
        path_cost_astar = compute_pathcost(rewired_path, all_states)
        # print ('pred_ordered_path', pred_ordered_path)
        # print ('path_cost_astar', path_cost_astar)
        total_pathcost_astar += path_cost_astar
        total_AstarPath += torch.where(pred_path>0)[0].size(0)
        total_history += torch.where(pred_history>0)[0].size(0)
        all_plan_time += construct_time + search_time
        all_construct_time += construct_time
        all_search_time += search_time

    n_tasks += 1

print ('*********************************************')
print ('The total number of nodes in Neural A star paths is', total_AstarPath/success)
print ('The total number of nodes in Neural A star search history is', total_history/success)
print ('The total path cost of Neural A star is', total_pathcost_astar/success)
print ('The success rate of Neural A star is:', success/n_tasks)
print ('The average time of Neural A star is', all_plan_time/success)
print ('The average time of construcing collision-free graph is', all_construct_time/success)
print ('The average time of A star search is', all_search_time/success)

# print ('*********************************************')
# print ('The average number of nodes in oracle path is', total_OraclePath/n_tasks)
# print ('The average path cost of oracle is', total_pathcost_oracle/n_tasks)
# print ('The average time of oracle is', oracle_time/n_tasks)

