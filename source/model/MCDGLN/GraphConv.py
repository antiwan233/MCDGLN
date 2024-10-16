import torch
import torch.nn as nn
from torch_geometric.nn import global_max_pool, Sequential,global_add_pool, global_mean_pool,LayerNorm, GCNConv
from torch_geometric.nn.pool import SAGPooling
from omegaconf import DictConfig


# 带残差连接的GCN-MLP块
class GCBLock(nn.Module):

    def __init__(self,
                 in_dim,
                 out_dim,
                 ):

        super(GCBLock, self).__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim

        self.activation = nn.Tanh()

        self.ConvBlock = Sequential('x, edge_index, edge_attr', [
            (GCNConv(in_channels=in_dim,
                     out_channels=out_dim,
                     add_self_loops=True),'x, edge_index -> x'),
            (self.activation, 'x->x'),
        ])

        self.lin = torch.nn.Sequential(
            nn.Linear(in_dim + out_dim, out_dim),
            self.activation,
            nn.Linear(out_dim, out_dim),
            self.activation
        )

        self.init_weight()

    def init_weight(self):

        for i in self.modules():
            if isinstance(i, nn.Linear):
                nn.init.xavier_uniform_(i.weight)
                nn.init.zeros_(i.bias)

    def forward(self, x, edge_index, edge_attr):

        x1 = self.ConvBlock(x, edge_index, edge_attr)

        return self.lin(torch.cat((x, x1), dim=1))


class GraphConv(nn.Module):

    def __init__(self,cfg: DictConfig):

        super().__init__()

        self.in_dim = cfg.model.in_dim
        self.hidden_dim = cfg.model.hidden_dim

        self.dropout = cfg.model.dropout
        self.batchsize = cfg.training.batch_size
        self.nrois = cfg.dataset.nrois
        self.SAGRatio = cfg.model.SAGRatio

        self.sagpooling = SAGPooling(in_channels=self.hidden_dim, ratio=self.SAGRatio)
        self.gcn_layers = cfg.model.gcn_layers

        self.mha = nn.MultiheadAttention(self.hidden_dim*self.gcn_layers,
                                         cfg.model.num_heads,
                                         dropout=cfg.model.dropout,
                                         batch_first=True)

        self.modlist = nn.ModuleList()

        for i in range(self.gcn_layers):

            if i == 0:
                self.modlist.append(GCBLock(in_dim=self.in_dim, out_dim=self.hidden_dim))
            else:
                self.modlist.append(GCBLock(in_dim=self.hidden_dim, out_dim=self.hidden_dim))

        self.lin = torch.nn.Sequential(
            nn.Linear(self.hidden_dim * self.gcn_layers, self.hidden_dim),
            nn.Tanh(),
        )

    def forward(self, src, edge_index, edge_attr, batch):

        z = src

        # 保留每一层residual GCN的输出
        torch_list = []

        for i in range(self.gcn_layers):

            if i == 0:
                z = self.modlist[i](z, edge_index, edge_attr)
            else:
                z = self.modlist[i](z, edge_index, edge_attr)

            z_pad = self.pooling(z, edge_index, batch)

            torch_list.append(z_pad)

        x = torch.cat(torch_list, dim=1)

        x = x.resize(self.batchsize, self.nrois, -1)

        x, attn_weights = self.SelfAttention(x, x, x, mask=None)

        # add & norm
        # x = att_x.reshape(self.batchsize*self.nrois, -1) + x

        x = self.readout(x.reshape(self.batchsize * self.nrois, -1), batch)

        x = self.lin(x.squeeze())

        return x, attn_weights

    def pooling(self, z, edge_index, batch):

        # 应用SAGPooling，给被mask的部分填充0，使得其维度与原始的x_in一致
        z_pad = torch.zeros_like(z)
        z_pooling, _, _, _, perm, _ = self.sagpooling(z, edge_index, batch=batch)
        mask = torch.zeros(z.size(0), dtype=torch.bool)
        mask[perm.long()] = True
        z_pad[mask] = z_pooling

        return z_pad

    def readout(self, x, batch):
        # 一次性的readout，还可以尝试hirarchical readout
        if self.args.gnn_pooling == 'mean':
            out = global_mean_pool(x, batch)
        elif self.args.gnn_pooling == 'max':
            # out = global_max_pool(x, batch, size=8)
            out = global_max_pool(x, batch)
        elif self.args.gnn_pooling == 'mean+max':
            out = global_mean_pool(x, batch) + global_max_pool(x, batch)
        else:
            out = global_add_pool(x, batch)

        return out
