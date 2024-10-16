import torch.nn as nn
import torch
import torch.nn.init as init
import torch.nn.functional as F
from torch.nn import BatchNorm2d


class MCCrossConv(nn.Module):

    def __init__(self, in_channels, out_channels, nrois):

        super(MCCrossConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.nrois = nrois

        self.weights = nn.Parameter(torch.empty((self.out_channels, self.in_channels, self.nrois, self.nrois)))
        self.bias = nn.Parameter(torch.empty((self.out_channels, self.in_channels, self.nrois, self.nrois)))

        self.norm = BatchNorm2d(self.out_channels)

        self.init_parameters()

    def init_parameters(self) -> None:

        init.uniform_(self.weights, 0, 1)

        if self.bias is not None:
            init.uniform_(self.bias, 0, 1)

    def forward(self, x):

        # 多个batch共用一个权重池，所以利用广播，一次将所有的权重矩阵乘以邻接矩阵，接下来要做的就是行列求和
        expanded_adj = x.unsqueeze(1).repeat(1, self.out_channels, 1, 1, 1)
        temp_adj = torch.sum(expanded_adj * self.weights + self.bias, dim=2)  # batchsize*out_channels*V*V
        row_sum = torch.sum(temp_adj, dim=2)  # batchsize*out_channels*V
        column_sum = torch.sum(temp_adj, dim=3)  # batchsize*out_channels*V
        row_sum = row_sum.unsqueeze(3).expand(-1, self.out_channels, self.nrois, self.nrois).transpose(2,3)
        # batchsize*out_channels*V*V
        column_sum = column_sum.unsqueeze(2).expand(-1, self.out_channels, self.nrois, self.nrois)
        # batchsize*out_channels*V*V
        sum_adj = row_sum + column_sum  # batchsize*out_channels*V*V

        # 是否过一次激活函数补充非线性，过了性能更好，因为会增加其非线性，从而表征能力更强
        sum_adj = F.tanh(sum_adj)  # batchsize*out_channels*V*V

        return sum_adj
