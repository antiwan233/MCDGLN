import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


class ChannelConv(nn.Module):

    def __init__(self, num_layers, num_heads, num_nodes):
        super(ChannelConv, self).__init__()

        # shape [batchsize, num_layers, num_nodes]

        self.num_layers = num_layers
        self.num_heads = num_heads
        self.num_nodes = num_nodes

        self.depthwise1 = nn.Conv2d(in_channels=self.num_heads,
                                   out_channels=self.num_heads*4,
                                   kernel_size=(1, 1),
                                   groups=self.num_heads)

        self.pointwise1 = nn.Conv2d(in_channels=self.num_heads*4,
                                   out_channels=1,
                                   kernel_size=(1, 1))

        self.Conv2Vector = nn.Sequential(
            self.depthwise1,
            nn.LayerNorm(self.num_nodes),
            # nn.BatchNorm2d(self.num_heads*4),
            nn.LeakyReLU(),

            self.pointwise1,
            nn.LayerNorm(self.num_nodes),
            # nn.BatchNorm2d(1),
            nn.LeakyReLU()
        )

        self.depthwise2 = nn.Conv1d(in_channels=self.num_layers,
                                    out_channels=self.num_layers*4,
                                    kernel_size=1,
                                    groups=self.num_layers)

        self.pointwise2 = nn.Conv1d(in_channels=self.num_layers*4,
                                    out_channels=1,
                                    kernel_size=1)

        self.Conv2Scalar = nn.Sequential(
            self.depthwise2,
            # nn.BatchNorm1d(self.num_layers*4),
            nn.LayerNorm(self.num_nodes),
            nn.LeakyReLU(),

            self.pointwise2,
            nn.LayerNorm(self.num_nodes),
            # nn.BatchNorm1d(1),
            nn.LeakyReLU()
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # 使用 Kaiming He 正态分布初始化
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # 假设输入形状为 [batchsize, num_layers, num_heads, num_nodes]
        batchsize, num_layers, num_heads, num_nodes = x.size()

        z = x.permute(0, 2, 1, 3)

        z1 = self.Conv2Vector(z).squeeze()

        z2 = self.Conv2Scalar(z1).squeeze()

        return z2, z1
