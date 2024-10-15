from torch_geometric.nn.conv import GCNConv
from torch_geometric.nn.pool import global_max_pool, global_mean_pool, global_add_pool
import torch
from torch.nn import functional as F
from omegaconf import DictConfig


class GCN(torch.nn.Module):

    def __init__(self, cfg: DictConfig):

        super(GCN, self).__init__()
        self.in_dim = cfg.model.in_dim
        self.hidden_dim = cfg.model.hidden_dim
        self.convs = torch.nn.ModuleList()

        self.num_layers = 3
        self.pooling = cfg.model.gnn_pooling
        self.num_classes = 2

        for i in range(self.num_layers):
            if i == 0:
                conv = GCNConv(self.in_dim, self.hidden_dim)
            elif i != self.num_layers - 1:
                conv = GCNConv(self.hidden_dim, self.hidden_dim)
            else:
                conv = GCNConv(self.hidden_dim, self.num_classes)
            self.convs.append(conv)

    def forward(self, x, edge_index, batch):
        z = x
        for i, conv in enumerate(self.convs):
            z = conv(z, edge_index)
            if i != len(self.convs) - 1:
                z = F.relu(z)
                z = F.dropout(z, training=self.training)
            if self.pooling == 'sum':
                g = global_add_pool(z, batch)
            elif self.pooling == 'mean':
                g = global_mean_pool(z, batch)
            else:
                g = global_max_pool(z, batch)

        return g

    def pooling(self, x, batch):

        # 一次性的readout，还可以尝试hierarchical readout
        if self.args.gnn_pooling == 'mean':
            out = global_mean_pool(x, batch)
        elif self.args.gnn_pooling == 'max':
            # out = global_max_pool(x, batch, size=8)
            out = global_max_pool(x, batch)
        else:
            out = global_add_pool(x, batch)

        return out
