import torch
import numpy as np
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, Sigmoid, Dropout
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
from torch import nn
from torch_sparse import coalesce
import math

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class MPNN(MessagePassing):
    def __init__(self, embed_size, aggr: str = 'max', **kwargs):
        super(MPNN, self).__init__(aggr=aggr, **kwargs)
        self.fx = Seq(Lin(embed_size * 4, embed_size), ReLU(), Lin(embed_size, embed_size))

    def forward(self, x, edge_index, edge_attr):
        """"""
        # propagate_type: (x: PairTensor, edge_attr: OptTensor)
        out = self.propagate(edge_index, x=(x, x), edge_attr=edge_attr)
        return torch.max(x, out)

    def message(self, x_i, x_j, edge_attr):
        z = torch.cat([x_j - x_i, x_j, x_i, edge_attr], dim=-1)
        values = self.fx(z)
        return values

    def __repr__(self):
        return '{}({}, dim={})'.format(self.__class__.__name__, self.channels,
                                       self.dim)


# class MPNN(MessagePassing):
#     def __init__(self, embed_size, aggr: str = 'max', batch_norm: bool = False, **kwargs):
#         super(MPNN, self).__init__(aggr=aggr, **kwargs)
#         self.batch_norm = batch_norm
#         self.lin_0 = Seq(Lin(embed_size * 5, embed_size), ReLU(), Lin(embed_size, embed_size))
#         self.lin_1 = Lin(embed_size * 2, embed_size)
#         self.bn = BatchNorm1d(embed_size)

#     def forward(self, x, edge_index, edge_attr):
#         """"""
#         # propagate_type: (x: PairTensor, edge_attr: OptTensor)
#         out = self.propagate(edge_index, x=(x, x), edge_attr=edge_attr)
#         out = self.bn(out) if self.batch_norm else out

#         return self.lin_1(torch.cat((x, out), dim=-1))

#     def message(self, x_i, x_j, edge_attr):
#         z = torch.cat([x_j - x_i, x_j, x_i, edge_attr], dim=-1)
#         values = self.lin_0(z)
#         return values

#     def __repr__(self):
#         return '{}({}, dim={})'.format(self.__class__.__name__, self.channels,
#                                        self.dim)



class Explorer(torch.nn.Module):
    def __init__(self, config_size, embed_size, obs_size):
        super(Explorer, self).__init__()

        self.config_size = config_size
        self.embed_size = embed_size
        self.obs_size = obs_size

        self.hx = Seq(Lin(config_size*4+8, embed_size), ReLU(),
#                           BatchNorm1d(embed_size, track_running_stats=False),
                          Lin(embed_size, embed_size))
        self.hy = Seq(Lin(config_size*3+6, embed_size), ReLU(),
#                           BatchNorm1d(embed_size, track_running_stats=True),
                          Lin(embed_size, embed_size))
        self.mpnn = MPNN(embed_size)
        self.fy = Seq(Lin(embed_size*3, embed_size), ReLU(),
                          Lin(embed_size, embed_size))

        # self.feta = Seq(Lin(embed_size, embed_size), ReLU(), #Dropout(p=0.5),
        #                   Lin(embed_size, 1, bias=False))

        # self.feta = Seq(Lin(embed_size, embed_size), ReLU(), #Dropout(p=0.4),
        #                 Lin(embed_size, 64), ReLU(), #Dropout(p=0.4),
        #                 Lin(64, 32), ReLU(),
        #                   Lin(32, 1))

        self.feta = Seq(Lin(embed_size, embed_size), ReLU(),# Dropout(p=0.5),
                        Lin(embed_size, embed_size), ReLU(), #Dropout(p=0.5),
                        Lin(embed_size, 1, bias=False))

    def reset_parameters(self):
        self.encoder.reset_parameters()
        self.decoder.reset_parameters()
        for op in self.ops:
            op.reset_parameters()
        self.node_feature.reset_parameters()
        self.edge_feature.reset_parameters()

    def forward(self, v, edge_index, loop, labels):
        self.labels = labels
        v = torch.cat((v, labels), dim=-1)
        goal = v[labels[:,1]==1].view(1, -1)
        x = self.hx(torch.cat((v, goal.repeat(len(v), 1), v-goal, (v-goal)**2), dim=-1))
        vi, vj = v[edge_index[0, :]], v[edge_index[1, :]]
        y = self.hy(torch.cat((vj-vi, vj, vi), dim=-1))

        # During training, we iterate x and y over a random number of loops between 1 and 10. Intuitively, taking
        # random loops encourages the GNN to converge faster, which also helps propagating the gradient.
        # During evaluation, the GNN explorer will output x and y after 10 loops. For loops larger than 10.
        for _ in range(loop):
            x = self.mpnn(x, edge_index, y)
            xi, xj = x[edge_index[0, :]], x[edge_index[1, :]]
            y = torch.max(y, self.fy(torch.cat((xj-xi, xj, xi), dim=-1)))

        heuristic = self.feta(x)
        return heuristic



# class Explorer(torch.nn.Module):
#     def __init__(self, config_size, embed_size, obs_size):
#         super(Explorer, self).__init__()
#         self.config_size = config_size
#         self.obs_size = obs_size

#         self.embed_size = embed_size

#         self.node_code = Seq(Lin(config_size*4, embed_size), ReLU(), Lin(embed_size, embed_size))
#         self.edge_code = Seq(Lin(config_size*2, embed_size), ReLU(), Lin(embed_size, embed_size))

#         self.obs_node_code = Seq(Lin(obs_size, embed_size), ReLU(), Lin(embed_size, embed_size))
#         self.obs_edge_code = Seq(Lin(obs_size, embed_size), ReLU(), Lin(embed_size, embed_size))
#         self.free_code = Seq(Lin(config_size, embed_size), ReLU(), Lin(embed_size, embed_size))
#         self.collided_code = Seq(Lin(config_size, embed_size), ReLU(), Lin(embed_size, embed_size))

#         self.env_code = Seq(Lin(embed_size*3, embed_size), ReLU(), Lin(embed_size, embed_size))

#         self.node_free_code = Seq(Lin(config_size, embed_size),
#                                   ReLU(), Lin(embed_size, embed_size))
#         self.edge_free_code = Seq(Lin(config_size * 2, embed_size),
#                                   ReLU(), Lin(embed_size, embed_size))

#         self.goal_encoder = nn.Parameter(torch.rand(embed_size))

#         self.encoder = Lin(embed_size * 4, embed_size)
#         self.process = MPNN(embed_size, aggr='max')

#         self.decoder = Lin(embed_size * 2, embed_size)

#         self.node_free = Lin(embed_size, 1)

#     def reset_parameters(self):
#         self.encoder.reset_parameters()
#         self.decoder.reset_parameters()
#         for op in self.ops:
#             op.reset_parameters()
#         self.node_feature.reset_parameters()
#         self.edge_feature.reset_parameters()


#     def forward(self, v, edge_index, goal_index, loop):
#     # def forward(self, v, goal, obstacles, labels, edge_index, loop, k=10, **kwargs):
#         goal = v[goal_index]

#         node_code = self.node_code(torch.cat((v, goal.repeat(len(v), 1), (v-goal)**2, v-goal), dim=-1))

#         edge_code = self.edge_code(torch.cat((v[edge_index[0, :]], v[edge_index[1, :]]), dim=-1))
#         node_free_code = self.node_free_code(v)
#         edge_free_code = self.edge_free_code(torch.cat((v[edge_index[0, :]], v[edge_index[1, :]]), dim=-1))
#         h_0 = node_code.new_zeros(len(node_code), self.embed_size)
#         h_0[goal_index, :] = h_0[goal_index, :] + self.goal_encoder
#         h_i = h_0

#         for i in range(loop):
#             encode = self.encoder(torch.cat((node_code, node_free_code.detach(), h_0, h_i), dim=-1))
#             h_i = self.process(encode, edge_index, torch.cat((edge_free_code.detach(), edge_code), dim=-1))
#         return self.node_free(h_i)

#         # edge_code = self.edge_code(torch.cat((v[edge_index[0, :]], v[edge_index[1, :]]), dim=-1))
#         # node_free_code = self.node_free_code(v)
#         # edge_free_code = self.edge_free_code(torch.cat((v[edge_index[0, :]], v[edge_index[1, :]]), dim=-1))

#         # goal_index = knn(v, goal.view(-1, self.config_size), k=1)[1]
#         # h_0 = node_code.new_zeros(len(node_code), self.embed_size)
#         # h_0[goal_index, :] = h_0[goal_index, :] + self.goal_encoder
#         # h_i = h_0

#         # for i in range(loop):
#         #     encode = self.encoder(torch.cat((node_code, node_free_code.detach(), h_0, h_i), dim=-1))
#         #     h_i = self.process(encode, edge_index, torch.cat((edge_free_code.detach(), edge_code), dim=-1))
#         #     decode = self.decoder(torch.cat((node_code, h_i), dim=-1))
#         # return self.node_free(decode)




     