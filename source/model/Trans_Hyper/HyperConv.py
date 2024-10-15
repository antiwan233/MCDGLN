import torch
from torch_geometric.nn import HypergraphConv, BatchNorm
from omegaconf import DictConfig
from torch_geometric.nn.pool import global_max_pool, global_mean_pool, global_add_pool, SAGPooling
from torch.nn import functional as F
from torch import Tensor


class SingleHyperConv(torch.nn.Module):

    def __init__(self, in_channels, out_channels, dropout):

        super(SingleHyperConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv = HypergraphConv(in_channels, out_channels, dropout=dropout)
        self.bn = BatchNorm(out_channels)
        self.activate = torch.nn.LeakyReLU()

        self.lin1 = torch.nn.Linear(in_channels, out_channels)

        self.pooling = SAGPooling(out_channels, ratio=0.5)

    def forward(self,
                x: Tensor,
                hyperedge_index: Tensor,
                hyperedge_weight: Tensor,
                batch):

        z = self.conv(x, hyperedge_index, hyperedge_weight, batch)
        z = self.bn(z)
        z = self.activate(z)

        res = self.lin1(x)
        res = self.activate(res)

        z = z + res

        z, pooled_edge_index, _, pooled_batch, _, _ = self.pooling(z, hyperedge_index, batch=batch)

        return z, pooled_edge_index, pooled_batch


class HyperConv(torch.nn.Module):

    def __init__(self, cfg:DictConfig):
        super(HyperConv, self).__init__()

        self.num_classes = cfg.model.num_classes
        self.num_layers = cfg.model.num_layers
        self.pooling = cfg.model.gnn_pooling

        self.conv = torch.nn.ModuleList()

        for i in range(self.num_layers):
            if i == 0:
                conv = SingleHyperConv(cfg.model.in_channels, cfg.model.hidden_channels, cfg.model.dropout)
            elif i != self.num_layers - 1:
                conv = SingleHyperConv(cfg.model.hidden_channels, cfg.model.hidden_channels, cfg.model.dropout)
            else:
                conv = SingleHyperConv(cfg.model.hidden_channels, cfg.model.hidden_channels, cfg.model.dropout)
            self.conv.append(conv)

        self._initialize_weights()

        self.lin = torch.nn.Sequential(
            torch.nn.Linear(cfg.model.hidden_channels, cfg.model.hidden_channels // 2),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(cfg.model.hidden_channels // 2, self.num_classes)
        )

    def _initialize_weights(self):
        """
        初始化模型中的所有可学习参数。
        """
        for module in self.modules():
            if isinstance(module, SingleHyperConv):
                for param in module.parameters():
                    if param.dim() > 1:  # 如果是权重参数
                        torch.nn.init.kaiming_uniform_(param)  # 使用 Kaiming 初始化
                    else:  # 如果是偏置参数
                        torch.nn.init.zeros_(param)  # 初始化为零

    def forward(self, x, hyperedge_index, hyperedge_weight, batch):

        # try:
        #     self.check_edge_index(hyperedge_index)
        #     print("Hyperedge index is valid")
        # except IndexError as e:
        #     print("Hyperedge index is invalid")
        #     print(e)

        z = x
        for i, conv in enumerate(self.conv):

            if i == 0:
                z, pooled_hyperedge_index, pooled_batch = conv(z, hyperedge_index, hyperedge_weight, batch)

                out = self.readout(z, pooled_batch)
            else:
                z, pooled_hyperedge_index, pooled_batch = conv(z, pooled_hyperedge_index, hyperedge_weight, pooled_batch)

                out = out + self.readout(z, pooled_batch)

        return self.lin(out)

    def readout(self, z, batch):

        if self.pooling == 'sum':
            g = global_add_pool(z, batch)
        elif self.pooling == 'mean':
            g = global_mean_pool(z, batch)
        elif self.pooling == 'max':
            g = global_max_pool(z, batch)
        elif self.pooling == 'mean+max':
            g = global_mean_pool(z, batch) + global_max_pool(z, batch)
        elif self.pooling == 'mean+max+sum':
            g = global_mean_pool(z, batch) + global_max_pool(z, batch) + global_add_pool(z, batch)
        else:
            g = global_max_pool(z, batch)

        return g

    def check_edge_index(self, edge_index):

        """
                检查边索引是否越界。

                参数:
                - edge_index (torch.LongTensor): 形状为 [2, E] 的张量，每一列代表一条边的两个端点索引。
                - num_nodes (int): 图中的节点数量。

                返回:
                - bool: 如果所有索引都在有效范围内则返回 True，否则抛出 IndexError。
                """
        # 检查是否存在负数索引
        if edge_index.numel() > 0 and edge_index.min() < 0:
            raise IndexError(
                f"Found negative indices in 'edge_index' (got {edge_index.min().item()}). "
                f"Please ensure that all indices in 'edge_index' point to valid indices "
                f"in the interval [0, 1600) in your node feature matrix and try again."
            )

        # 检查索引是否超过节点数量
        if edge_index.numel() > 0 and edge_index.max() >= 1600:
            raise IndexError(
                f"Found indices in 'edge_index' that exceed the number of nodes (got {edge_index.max().item()}). "
                f"Please ensure that all indices in 'edge_index' point to valid indices "
                f"in the interval [0, 1600) in your node feature matrix and try again."
            )

        return True
