import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig
from .CrossConv import CrossConvBlock
from .GraphConv import GraphConv


class MCDGLN(torch.nn.Module):

    def __init__(self, cfg: DictConfig):

        super().__init__()

        self.nrois = cfg.dataset.nrois
        self.bs = cfg.training.batch_size

        self.CrossConv = CrossConvBlock(nrois=cfg.dataset.nrois,
                                        in_channels=cfg.model.windows_amount,
                                        out_channels=cfg.model.E2E_channels)

        self.GraphConv = GraphConv(cfg)

        self.activation = nn.Tanh()
        self.lin1 = torch.nn.Sequential(
            nn.Linear(cfg.model.hidden_dim * 2, 256),
            self.activation,
            nn.Dropout(cfg.model.dropout),
            nn.Linear(256, 128),
            self.activation,
            nn.Dropout(cfg.model.dropout),
            nn.Linear(128, 32),
            self.activation,
            nn.Dropout(cfg.model.dropout),
            nn.Linear(32, 2),
        )

        self.lin2 = torch.nn.Sequential(
            nn.Linear(cfg.dataset.nrois * cfg.dataset.nrois // 2 - cfg.dataset.nrois // 2, cfg.dataset.nrois * 20),
            self.activation,
            nn.Linear(cfg.dataset.nrois * 20, cfg.dataset.nrois * 10),
            self.activation,
            nn.Linear(cfg.dataset.nrois * 10, cfg.dataset.nrois * 5),
            self.activation,
            nn.Linear(cfg.dataset.nrois * 5, cfg.dataset.nrois),
        )

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

        z = src

        static_fc = z.contiguous().view(-1, self.nrois, self.nrois)

        dynamic_edge_index, dynamic_edge_attr, dynamic_fc = self.CrossConv(windows, batch, self.bs, static_fc)
        graph_out, attn_weights = self.GraphConv(z, dynamic_edge_index, dynamic_edge_attr)

        # 使用dynamic 对static fc进行遮蔽
        # 使用attn_weights对static fc进行增强

        mask = (dynamic_fc <= 0)
        masked_fc = static_fc.masked_fill(mask, 0)

        row_index, col_index = torch.triu_indices(self.nrois, self.nrois, offset=1)
        att_fc = attn_weights * masked_fc
        flatten_fc = att_fc[:, row_index, col_index]

        static_out = self.lin2(flatten_fc)

        z = torch.cat((graph_out, static_out), dim=1)
        z = self.lin1(z)

        return z
