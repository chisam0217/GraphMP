import torch
from environment import MazeEnv, KukaEnv, Kuka2Env, SnakeEnv, UR5Env
import numpy as np
from diff_astar import HeuristicNeuralAstar
from collision_net import CollisionNet

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def str2name(robot, n_sample, k, use_obstacles=True):
    # if robot == "maze2_easy" or robot == "maze2_hard":
    #     model_astar_path = "Astar_weights/maze2_{}_{}.pt".format(n_sample, k)
    # else:
    model_astar_path = "Astar_weights/{}_{}_{}.pt".format(robot, n_sample, k)
    model_coll_path = "collision_weights/{}_{}_{}_collision.pt".format(robot, n_sample, k)
    data_path = "data/own_pkl/{}_prm_{}_{}.pkl".format(robot, n_sample, k)

    if robot == 'maze2_easy':
        env = MazeEnv(dim=2, map_file='maze_files/mazes_easy.npz')
        model_astar = HeuristicNeuralAstar(Tmax=1, config_size=2, obs_size=2, embed_size=64) #64
        model_coll = CollisionNet(config_size=2, embed_size=64, obs_size=2, use_obstacles=True)
        # data_path = 'data/own_pkl/maze_prm_4000.pkl'
    elif robot == 'maze2_hard':
        env = MazeEnv(dim=2, map_file='maze_files/mazes_hard.npz')
        model_astar = HeuristicNeuralAstar(Tmax=1, config_size=2, obs_size=2, embed_size=32) #64
        model_coll = CollisionNet(config_size=2, embed_size=64, obs_size=2, use_obstacles=True)
        # data_path = 'data/own_pkl/maze_prm_4000.pkl'

    elif robot == 'maze3':
        env = MazeEnv(dim=3)
        model_explore = EncoderProcessDecoder(config_size=3, embed_size=32, obs_size=2).to(device)
        model_smooth = ModelSmoother(config_size=env.config_dim, embed_size=64, obs_size=6).to(device)
        # data_path = 'data/pkl/maze_prm_3.pkl'

    elif robot == 'ur5':
        env = UR5Env()
        model_astar = HeuristicNeuralAstar(Tmax=1, config_size=6, obs_size=6, embed_size=32) #64
        model_coll = CollisionNet(config_size=6, embed_size=64, obs_size=6, use_obstacles=True)
        # data_path = 'data/pkl/ur5_prm_3000.pkl'

    elif robot == 'snake7':
        env = SnakeEnv(map_file='maze_files/snakes_15_2_3000.npz')
        model_astar = HeuristicNeuralAstar(Tmax=1, config_size=7, obs_size=2, embed_size=32)
        model_coll = CollisionNet(config_size=7, embed_size=64, obs_size=2, use_obstacles=True)
        # data_path = 'data/pkl/snake_prm_3000.pkl'

    elif robot == 'kuka7':
        env = KukaEnv()
        model_astar = HeuristicNeuralAstar(Tmax=1, config_size=7, obs_size=6, embed_size=32)
        model_coll = CollisionNet(config_size=7, embed_size=64, obs_size=6, use_obstacles=True)

    elif robot == 'kuka13':
        env = KukaEnv(kuka_file="kuka_iiwa/model_3.urdf", map_file="maze_files/kukas_13_3000.pkl")
        model_astar = HeuristicNeuralAstar(Tmax=1, config_size=13, obs_size=6, embed_size=32)
        model_coll = CollisionNet(config_size=13, embed_size=64, obs_size=6, use_obstacles=True)
        # data_path = 'data/pkl/kuka_prm_13.pkl'

    elif robot == 'kuka14':
        env = Kuka2Env()
        model_astar = HeuristicNeuralAstar(Tmax=1, config_size=14, obs_size=6, embed_size=32)
        model_coll = CollisionNet(config_size=14, embed_size=64, obs_size=6, use_obstacles=True)


    # if load:
    #     model_explore.load_state_dict(torch.load(model_explore_path, map_location=device))
    #     model_explore.to(device)

    #     model_smooth.load_state_dict(torch.load(model_smooth_path, map_location=device))
    #     model_smooth.to(device)

    return env, model_astar, model_astar_path, model_coll, model_coll_path, data_path
