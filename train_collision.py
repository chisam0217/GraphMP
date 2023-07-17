import torch
import numpy as np
from torch_geometric.data import Data
from tqdm import tqdm as tqdm
from tensorboardX import SummaryWriter
import pickle
from time import time
from algorithm.dijkstra import dijkstra
from collision_net import CollisionNet
import torch.nn as nn
from environment import KukaEnv, MazeEnv, SnakeEnv, Kuka2Env
import argparse
import random
from str2name import str2name
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--is_training', action='store_true', default=False, help='Training or Testing.')
parser.add_argument('--seed', type=int, default=123, help='Random seed.')
parser.add_argument('--lr', type=float, default=1e-3, help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--epoch', type=int, default=500, help='The number of epochs.')
parser.add_argument('--batch_size', type=int, default=1, help='The batch size.') #8
parser.add_argument('--hidden', type=int, default=64, help='Number of hidden units.')
parser.add_argument('--schedule', type=bool, default=True, help='Whether to turn on optimizer scheduler.')
parser.add_argument('--finetune', type=bool, default=False, help='Whether to finetune the model.')

parser.add_argument('--robot', type=str, default="maze2_hard", help='[maze2_easy, maze2_hard, ur5, snake7, kuka7, kuka13, kuka14]')
parser.add_argument('--n_sample', type=int, default=50, help='The number of samples.')
parser.add_argument('--k', type=int, default=10, help='The number of neighbors.')

args = parser.parse_args()
def set_random_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
set_random_seed(args.seed)
INFINITY = float('inf')
writer = SummaryWriter()

env, _, _, model, weights_name, data_path = str2name(args.robot, args.n_sample, args.k)
if not args.is_training:
    model.load_state_dict(torch.load(weights_name, map_location=device))

model = model.to(device)

if args.finetune:
    model.load_state_dict(torch.load(weights_name, map_location=device))
    weights_name = "collision_weights/{}_{}_{}_finetune.pt".format(args.robot, args.n_sample, args.k)


with open(data_path, 'rb') as f:
    graphs = pickle.load(f)


# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=200*2000, gamma=0.5)

def train_collision_net(model, graphs):
    T = 0
    model.train()
    # optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    if not args.finetune:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)

    optimizer.zero_grad()
    criterion = torch.nn.BCELoss()
    iteration = 0
    min_loss = INFINITY

    if args.schedule:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)

    # writer = SummaryWriter()
    for iter_i in range(args.epoch):
        scheduler.step()
        if args.robot == "maze2_easy" or args.robot == "maze2_hard":
            indexes = np.random.permutation(1000)
        else:
            indexes = np.random.permutation(2000)
        pbar = tqdm(indexes)
        losses = 0

        for index in pbar:
        # for index in indexes:
            pb = env.init_new_problem(index)
            points, neighbors, edge_cost, edge_index, edge_free, _ = graphs[index]

            edge_index = torch.LongTensor(edge_index.T)
            temp_tensor = torch.FloatTensor()

            node_free = temp_tensor.new_zeros(len(points), len(points))
            node_free[edge_index[0, :], edge_index[1, :]] = torch.FloatTensor(edge_free).squeeze()
            node_free = torch.diag(node_free, 0)

            points = np.stack(points)
            points = torch.FloatTensor(points).to(device)

            edge_index = edge_index.to(device)
            obs = torch.FloatTensor(env.obstacles).to(device)

            edge_free = np.array(edge_free).astype(np.int32)

            one_hot_edge_free = np.zeros((edge_free.shape[0], 2))
            one_hot_edge_free[np.arange(edge_free.shape[0]), edge_free] = 1

            one_hot_edge_free = torch.FloatTensor(one_hot_edge_free).to(device)    
            pred = model(points, edge_index, obs)

            # loss = torch.nn.CrossEntropyLoss(pred, edge_free)
            loss = criterion(pred, one_hot_edge_free)/edge_index.size(0)
            loss.backward()
            losses += loss.item()

            if T % args.batch_size == 0:
                optimizer.step()
                optimizer.zero_grad()
                iteration += 1
                curr_loss = losses / iteration
                # curr_loss = losses/(T+1)
                # writer.add_scalar('2dmaze_collision/total_loss', curr_loss, T)
                writer.add_scalar('{}_{}_{}_collision/total_loss'.format(args.robot, args.n_sample, args.k), curr_loss, iteration)
                # if curr_loss < min_loss:
                #     min_loss = curr_loss
                # print ('loss', loss)
            T += 1
        torch.save(model.state_dict(), weights_name)
            
    writer.close()


#Evaluating
# with open('data_gen/data/maze_prm_2_test_200_10.pkl', 'rb') as f:
# # with open('data_gen/data/maze_prm_2_200_10.pkl', 'rb') as f:
#     graphs = pickle.load(f)

def test_collision_net(model, graphs):
    all_pred = 0
    correct_pred = 0
    if args.robot == 'maze2_easy' or args.robot == 'maze2_hard':
        indexes = np.arange(1000)
    else:
        indexes = np.arange(2000, 3000)
    pbar = tqdm(indexes)
    model.eval()

    FP = 0
    time0 = time()
    collision_time = 0
    confidence = []
    for index in pbar:
        pb = env.init_new_problem(index)
        time0 = time()
        points, neighbors, edge_cost, edge_index, edge_free, _ = graphs[index]
        edge_index = torch.LongTensor(edge_index.T)
        temp_tensor = torch.FloatTensor()

        node_free = temp_tensor.new_zeros(len(points), len(points))
        node_free[edge_index[0, :], edge_index[1, :]] = torch.FloatTensor(edge_free).squeeze()
        node_free = torch.diag(node_free, 0)
        points = np.stack(points)

        points = torch.FloatTensor(points).to(device)
        edge_index = edge_index.to(device)
        obs = torch.FloatTensor(env.obstacles).to(device)

        edge_free = np.array(edge_free).astype(np.int32)


        outputs = model(points, edge_index, obs)
        pred = np.argmax(outputs.detach().cpu().numpy(), axis=1)
        collision_time += time() - time0

        correct = np.sum(pred == edge_free)
        correct_pred += correct
        all_pred += edge_free.shape[0]
        confidence.append(np.mean(np.max(outputs.detach().cpu().numpy(), axis=1)))

        for j in range(pred.shape[0]):
            if pred[j] == 1 and edge_free[j] == 0:
                FP += 1
        # FP += np.sum(all(pred==1 and edge_free==0))
    print ('The time cost is: ', collision_time/indexes.shape[0])
    print ('Test Accuracy:', correct_pred/all_pred)
    print ('The False Positive:', FP/all_pred)
    print ('confidence', np.mean(confidence), np.std(confidence))


if args.is_training: 
    train_collision_net(model, graphs)
test_collision_net(model, graphs)


# Test Accuracy: 0.9715029009502486
# The False Positive: 0.024331808337614242
