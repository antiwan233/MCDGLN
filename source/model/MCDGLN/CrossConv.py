import torch
import torch.nn as nn
from torch_geometric.utils import dense_to_sparse
from .MCCrossConv import MCCrossConv
from .SEAttention import SEAttention


class CrossConvBlock(torch.nn.Module):

    def __init__(self,
                 nrois: int,
                 in_channels: int,
                 out_channels: int):

        super().__init__()

        self.nrois = nrois
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.out_channels_list = torch.tensor([16, 8, 4, 4, 4, 4, 4, 4], dtype=torch.int8)
        self.cross_conv_layers = 1
        self.modlist = nn.ModuleList()

        for i in range(self.cross_conv_layers):

            if i == 0:
                self.modlist.append(MCCrossConv(in_channels=self.in_channels,
                                                out_channels=self.out_channels_list[i],
                                                nrois=self.nrois))
            else:
                self.modlist.append(MCCrossConv(in_channels=self.out_channels_list[i - 1],
                                                out_channels=self.out_channels_list[i],
                                                nrois=self.nrois))

        self.activation = nn.Tanh()

        self.se = SEAttention(channels=self.out_channels_list[self.cross_conv_layers - 1] + 1, reduction=2)

    def forward(self, windows, batch, batchsize, *args):

        # 输入的windows的形状为(batchsize, amount_windows/in_channels, V, V)
        adj_batch = windows.clone().contiguous().view(-1, self.in_channels, self.nrois, self.nrois)

        # 对Modulelist中的每一个CrossConv进行前向传播
        for i in range(self.cross_conv_layers):
            adj_batch = self.modlist[i](adj_batch)

        # 将origin_fc也引入SE模块
        origin_fc = args[0]
        origin_fc = origin_fc.unsqueeze(1)  # batchsize*1*V*V

        sum_adj = torch.cat((adj_batch, origin_fc), dim=1)  # batchsize*(out_channels+1)*V*V

        # 在这里对一个样本的out_adj进行一个（注意力机制）的融合
        sum_adj = self.se(sum_adj)
        out_adj = torch.sum(sum_adj, dim=1)  # batchsize*V*V

        out_adj = self.activation(out_adj)  # batchsize*V*V

        # 将out_adj中的负值去除(去除抑制连接)， 然后转置相加，使其强制变成对称矩阵
        # out_adj小于0.2的地方置为0，大于0的地方置为其原本的值

        # 20240521修改
        out_adj = torch.where(torch.abs(out_adj) < 0.3, torch.zeros_like(out_adj), out_adj)

        # 使得out_adj强制对称
        out_adj = (out_adj + out_adj.transpose(1, 2)) / 2  # batchsize*V*V
        # new_edge_index, new_edge_attr = dense_to_sparse(out_adj)

        # 20240521修改
        new_edge_index, new_edge_attr = dense_to_sparse(torch.where(out_adj >= 0.3, out_adj, torch.zeros_like(out_adj)))

        return new_edge_index, new_edge_attr, out_adj
