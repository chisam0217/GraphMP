import torch
import numpy as np
from torch_geometric.data import Data
from tqdm import tqdm as tqdm
from tensorboardX import SummaryWriter
import pickle
from time import time
from algorithm.dijkstra import dijkstra
from diff_astar import HeuristicNeuralAstar
import torch.nn as nn
from environment import KukaEnv, MazeEnv, SnakeEnv, Kuka2Env
import argparse
import random
from str2name import str2name
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def backtrack(start_n, goal_n, prev_node, num_nodes):
    path = torch.zeros(num_nodes).to(device)
    path[goal_n] = 1
    path[start_n] = 1
    loc = start_n
    while loc != goal_n:
        loc = prev_node[loc]
        path[loc] = 1
    return path

def compute_pathcost(path, states):
    path = path.astype(np.int)
    cost = 0
    for i in range(0, path.shape[0]-1):
        cost += np.linalg.norm(states[path[i]] - states[path[i+1]])
    return cost

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--is_training', action='store_true', default=True, help='Training or Testing.')
parser.add_argument('--seed', type=int, default=1234, help='Random seed.')
parser.add_argument('--batch_size', type=int, default=8, help='The batch size.')#8
parser.add_argument('--hidden', type=int, default=32, help='Number of hidden units.')

parser.add_argument('--epoch', type=int, default=1000, help='The number of epochs.')
parser.add_argument('--lr', type=float, default=1e-3, help='Initial learning rate.')
parser.add_argument('--schedule', type=bool, default=False, help='Whether to turn on optimizer scheduler.')
parser.add_argument('--finetune', type=bool, default=False, help='Whether to finetune the model.')

parser.add_argument('--robot', type=str, default="ur5", help='[maze2_easy, maze2_hard, ur5, snake7, kuka7, kuka13, kuka14]')
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

env, model, weights_name, _, _, data_path = str2name(args.robot, args.n_sample, args.k)
model = model.to(device)
# weights_name = "Astar_weights/{}_{}_{}_batch64.pt".format(args.robot, args.n_sample, args.k)

if args.finetune:
    model.load_state_dict(torch.load(weights_name, map_location=device))
    weights_name = "Astar_weights/{}_{}_{}_finetune.pt".format(args.robot, args.n_sample, args.k)

with open(data_path, 'rb') as f:
    graphs = pickle.load(f)

loop = 10
T = 0



# optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4) #XXXXX

# if args.robot in ["maze2", "kuka13"]:
#     if not args.finetune:
#         optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
#     else:
#         optimizer = torch.optim.SGD(model.parameters(), lr=5e-3, momentum=0.9)
# else:
#     if not args.finetune:
#         optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
#     else:
#         optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)


def test_planning():
    if args.robot == "maze2_easy" or args.robot == "maze2_hard":
        indexes = np.arange(1000)
    else:
        indexes = np.arange(2000, 3000)
    pbar = tqdm(indexes)
    total_pathcost_astar = []
    model.eval()
    for index in pbar:
        pb = env.init_new_problem(index)
        points, neighbors, edge_cost, edge_index, edge_free, _ = graphs[index]
        all_states = np.copy(points)
        goal_index = 1
        dist, prev = dijkstra(list(range(len(points))), neighbors, edge_cost, goal_index)
        start_index = 0

        edge_index = torch.LongTensor(edge_index.T)
        temp_tensor = torch.FloatTensor()
        node_free = temp_tensor.new_zeros(len(points), len(points))
        node_free[edge_index[0, :], edge_index[1, :]] = torch.FloatTensor(edge_free).squeeze()

        edge_cost = sum(edge_cost.values(), [])
        edge_cost = np.array(edge_cost)
        cost_maps = temp_tensor.new_zeros(len(points), len(points))
        cost_maps[edge_index[0, :], edge_index[1, :]] = torch.FloatTensor(edge_cost).squeeze()
        
        diag_node_free = torch.diag(node_free, 0)
        node_free = node_free.to(device)
        cost_maps = cost_maps.to(device)
        points = torch.FloatTensor(np.array(points)).to(device)
        edge_index = edge_index.to(device)

        labels = torch.zeros(len(points), 2)
        labels[:, 0] = 1 #nodes in the free space
        labels[goal_index, 0] = 0
        labels[goal_index, 1] = 1 #goal node
        labels = labels.to(device)

        current_loop = 3

        open_list, pred_history, pred_path, pred_ordered_path = model(start_index, goal_index, points, edge_index, node_free, cost_maps, current_loop, labels)
        if pred_ordered_path: #if the results are not empty
            pred_ordered_path = torch.stack(pred_ordered_path).cpu().detach().numpy()
        pred_ordered_path = np.concatenate((pred_ordered_path, np.array([goal_index])), axis=0)

        if pred_path[start_index]:
            path_cost_astar = compute_pathcost(pred_ordered_path, all_states)
            total_pathcost_astar.append(path_cost_astar)
    return np.mean(total_pathcost_astar)


# optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)#, weight_decay=5e-4)
optimizer.zero_grad()
if args.schedule:
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)

criterion = nn.L1Loss()

s_time = time()
losses = 0
iteration = 0
min_test_path_cost = 100

for iter_i in range(args.epoch):
    model.train()
    # indexes = np.random.permutation(2000)
    if args.robot == "maze2_easy" or args.robot == "maze2_hard":
            indexes = np.random.permutation(1000)
    else:
        indexes = np.random.permutation(2000)
    pbar = tqdm(indexes)
    for index in pbar:
        pb = env.init_new_problem(index)

        points, neighbors, edge_cost, edge_index, edge_free, _ = graphs[index]
        goal_index = np.random.choice(len(points))

        dist, prev = dijkstra(list(range(len(points))), neighbors, edge_cost, goal_index)
        prev[goal_index] = goal_index
        valid_node = (np.array(list(dist.values())) != INFINITY)
        if sum(valid_node) == 1:
            continue
        start_index = np.random.choice(np.arange(len(valid_node))[valid_node])#random select a start node

        edge_index = torch.LongTensor(edge_index.T)
        temp_tensor = torch.FloatTensor()
        node_free = temp_tensor.new_zeros(len(points), len(points))
        node_free[edge_index[0, :], edge_index[1, :]] = torch.FloatTensor(edge_free).squeeze()

        edge_cost = sum(edge_cost.values(), [])
        edge_cost = np.array(edge_cost)

        cost_maps = temp_tensor.new_zeros(len(points), len(points))
        cost_maps[edge_index[0, :], edge_index[1, :]] = torch.FloatTensor(edge_cost).squeeze()

        labels = torch.zeros(len(points), 2)
        labels[:, 0] = 1 #nodes in the free space
        labels[goal_index, 0] = 0
        labels[goal_index, 1] = 1 #goal node

        # start_index = np.random.choice(np.arange(len(valid_node))[valid_node], size=batch_size, replace=True) #random select a start node
        node_free = node_free.to(device)
        cost_maps = cost_maps.to(device)
        labels = labels.to(device)
        points = np.stack(points)
        points = torch.FloatTensor(points).to(device)
        edge_index = edge_index.to(device)

        # loop = 10
        # current_loop = np.random.randint(1, loop)
        current_loop = 3

        pred_history, pred_path = model(start_index, goal_index, points, edge_index, node_free, cost_maps, current_loop, labels)
        oracle_path = backtrack(start_index, goal_index, prev, points.size(0))
        # print (oracle_path)
        loss = criterion(pred_history, oracle_path) /points.size(0) #+ 0.01 * torch.abs(pred_history[goal_index] - 1)
        loss.backward()
        losses += loss.item()

        if T % args.batch_size == 0:
            iteration += 1
            optimizer.step()
            optimizer.zero_grad()
            curr_loss = losses / iteration
            writer.add_scalar('{}_{}_{}/total_loss'.format(args.robot, args.n_sample, args.k), curr_loss, iteration)
            
        T += 1
    mean_test_path_cost = test_planning()
    if mean_test_path_cost < min_test_path_cost:
        min_test_path_cost = mean_test_path_cost
        print ('The average path cost is:', min_test_path_cost)
        torch.save(model.state_dict(), weights_name)
    if args.schedule:
        scheduler.step()

print ('The total training time is', time() - s_time)
writer.close()


#batch = 2, lr=0.001, weight_decay=2e-4
#batch = 8, lr=0.001, weight_decay=5e-4