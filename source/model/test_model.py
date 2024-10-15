import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        # 第一层全连接层
        self.fc1 = nn.Linear(input_size, hidden_size)
        # 第二层全连接层
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        # 输出层
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # 输入层到隐藏层，使用ReLU激活函数
        x = F.relu(self.fc1(x))
        # 隐藏层到隐藏层，再次使用ReLU激活函数
        x = F.relu(self.fc2(x))
        # 隐藏层到输出层
        x = self.fc3(x)
        return x