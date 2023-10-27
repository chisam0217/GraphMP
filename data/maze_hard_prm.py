import sys
sys.path.append("./")
import numpy as np
import torch
from environment import KukaEnv
from torch_geometric.nn import knn_graph
from collections import defaultdict
from time import time
import pickle
from tqdm import tqdm
from torch_sparse import coalesce
from algorithm.bit_star import BITStar
from environment import MazeEnv
from torch_geometric.data import Data
import random
INFINITY = float('inf')


def construct_graph(env, points, k=5, check_collision=True):
    edge_index = knn_graph(torch.FloatTensor(points), k=k, loop=True)
    edge_index = torch.cat((edge_index, edge_index.flip(0)), dim=-1)
    edge_index_torch, _ = coalesce(edge_index, None, len(points), len(points))
    # print ('edge_index_torch', edge_index_torch.shape[1])
    
    edge_index = edge_index_torch.data.cpu().numpy().T
    edge_cost = defaultdict(list)
    fake_edge_cost = defaultdict(list)
    edge_free = []
    neighbors = defaultdict(list)
    for i, edge in enumerate(edge_index):
        fake_edge_cost[edge[1]].append(np.linalg.norm(points[edge[1]]-points[edge[0]]))
        if env._edge_fp(points[edge[0]], points[edge[1]]):
            edge_cost[edge[1]].append(np.linalg.norm(points[edge[1]]-points[edge[0]]))
            edge_free.append(True)
        else:
            edge_cost[edge[1]].append(INFINITY)
            edge_free.append(False)
        neighbors[edge[1]].append(edge[0])
    # print ('edge_free', np.sum(edge_free))
    return edge_cost, neighbors, edge_index, edge_free, fake_edge_cost


def min_dist(q, dist):
    """
    Returns the node with the smallest distance in q.
    Implemented to keep the main algorithm clean.
    """
    min_node = None
    for node in q:
        if min_node is None:
            min_node = node
        elif dist[node] < dist[min_node]:
            min_node = node

    return min_node


def dijkstra(nodes, edges, costs, source):
    q = set()
    dist = {}
    prev = {}

    for v in nodes:       # initialization
        dist[v] = INFINITY      # unknown distance from source to v
        prev[v] = INFINITY      # previous node in optimal path from source
        q.add(v)                # all nodes initially in q (unvisited nodes)

    # distance from source to source
    dist[source] = 0

    while q:
        # node with the least distance selected first
        u = min_dist(q, dist)

        q.remove(u)

        for index, v in enumerate(edges[u]):
            alt = dist[u] + costs[u][index]
            if alt < dist[v]:
                # a shorter path to v has been found
                dist[v] = alt
                prev[v] = u

    return dist, prev


if __name__ == "__main__":
    data = []
    env = MazeEnv(dim=2, map_file="maze_files/mazes_hard.npz")

    def set_random_seed(seed):
        np.random.seed(seed)
        torch.manual_seed(seed)
        random.seed(seed)

    set_random_seed(4123)

    time0 = time()
    results = []
    n_sample = 300
    n_neighbor = 20

    for problem_index in tqdm(range(1000)):
        for i in range(10):
            env.init_new_problem(problem_index)
            n_sample = random.randint(1, 3) * 100
            n_neighbor = random.randint(1, 3) * 10

            points = env.sample_n_points(n=n_sample)
            # points.insert(0, env.init_state)
            # points.insert(0, env.goal_state)

            edge_cost, neighbors, edge_index, edge_free, fake_edge_cost = construct_graph(env, points, n_neighbor)

            data.append((points, neighbors, edge_cost, edge_index, edge_free, fake_edge_cost, problem_index))

            # dist, prev = dijkstra(list(range(len(points))), neighbors, edge_cost, 0)
            # valid_goal = np.logical_and(np.array(list(dist.values())) != INFINITY, np.array(list(dist.values()))!=0)
            # goal_index = np.random.choice(len(valid_goal), p=valid_goal.astype(float)/sum(valid_goal))
            #
            # print(time()-time0)
            # print('yes')
    with open('data/own_pkl/maze2_hard_prm_random_random.pkl', 'wb') as f:
        pickle.dump(data, f, pickle.DEFAULT_PROTOCOL)

    # with open('data/own_pkl/maze2_hard_prm_{}_{}.pkl'.format(n_sample, n_neighbor), 'wb') as f:
    #     pickle.dump(data, f, pickle.DEFAULT_PROTOCOL)