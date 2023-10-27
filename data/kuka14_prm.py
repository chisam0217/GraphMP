import sys
sys.path.append("./")
import numpy as np
import torch
from environment import KukaEnv, MazeEnv, SnakeEnv, UR5Env
from environment import Kuka2Env
from torch_geometric.nn import knn_graph
from collections import defaultdict
from time import time
import pickle
from tqdm import tqdm
from torch_sparse import coalesce
import random

INFINITY = float('inf')


def construct_graph(env, points, k=5, check_collision=True):
    edge_index = knn_graph(torch.FloatTensor(points), k=k, loop=True)
    edge_index = torch.cat((edge_index, edge_index.flip(0)), dim=-1)
    edge_index_torch, _ = coalesce(edge_index, None, len(points), len(points))
    
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
    prev[source] = source

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
    problems = []
    # n_sample = [50, 200, 1000]
    n_sample = 300
    n_neighbor = 20

    def set_random_seed(seed):
        np.random.seed(seed)
        torch.manual_seed(seed)
        random.seed(seed)

    set_random_seed(4123)
    env = Kuka2Env()
    # for n in n_sample:\
    pbar = tqdm(range(2000))
    for problem_index in pbar:
        for _ in range(5):
            n_sample = random.randint(1, 3) * 100
            n_neighbor = random.randint(1, 3) * 10

            env.init_new_problem(problem_index)
            points = env.sample_n_points(n=n_sample)
            # points.insert(0, env.init_state)
            # points.insert(0, env.goal_state)
            edge_cost, neighbors, edge_index, edge_free, fake_edge_cost = construct_graph(env, points, n_neighbor)
            data.append((points, neighbors, edge_cost, edge_index, edge_free, fake_edge_cost, problem_index))

    
    with open('data/own_pkl/kuka14_prm_random_random.pkl'.format(n_sample, n_neighbor), 'wb') as f:
        pickle.dump(data, f, pickle.DEFAULT_PROTOCOL)



