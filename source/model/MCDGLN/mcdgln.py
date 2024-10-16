import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig


class MCDGLN(torch.nn.Module):

    def __init__(self, cfg: DictConfig):

        super().__init__()

        self._init_weights()

    def _init_weights(self):

        for module in self.modules():
            if isinstance(module, MCDGLN):
                for param in module.parameters():
                    if param.dim() > 1:  # 如果是权重参数
                        torch.nn.init.kaiming_uniform_(param)  # 使用 Kaiming 初始化
                    else:  # 如果是偏置参数
                        torch.nn.init.zeros_(param)  # 初始化为零

    def forward(self, src, windows, batch):

        pass

