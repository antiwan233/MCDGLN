import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from .CustomTransEncoderLayer import CustomTransformerEncoderLayer
from .GCNBlock import GCNBlock
from .CustomTransfoerEncoder import CustomTransformerEncoder
from torch_geometric.utils import to_dense_adj
from .HyperConv import HyperConv
from .ChannelConv import ChannelConv
from .LstmBlock import LstmBlock


class Trans_HyperGraph(torch.nn.Module):

    def __init__(self, cfg: DictConfig):
        super().__init__()

        self.activation = torch.nn.LeakyReLU()
        self.dropout = cfg.model.dropout

        self.transformer_encoder = CustomTransformerEncoder(cfg)

        self.channel_conv = ChannelConv(cfg.model.num_transformer_encoder_layers,
                                        cfg.model.nhead,
                                        cfg.dataset.nrois,
                                        )
        #
        self.hyper_dependency_conv = HyperConv(cfg)
        self.hyper_importance_conv = HyperConv(cfg)

        # self.lstm_dependency = LstmBlock(cfg)
        # self.lstm_importance = LstmBlock(cfg)

        # self.hyper_conv = HyperConv(cfg)

    def forward(self, x, edge_index, edge_weight, batch, mask=None, src_key_padding_mask=None):

        z = x

        z, dependency, importance = self.transformer_encoder(z, mask, src_key_padding_mask)

        z = z.reshape(-1,x.shape[1])

        dependency, sequence_dependency = self.channel_conv(dependency)
        importance, sequence_importance = self.channel_conv(importance)

        dependency = dependency.reshape(-1)
        importance = importance.reshape(-1)
        #
        # # dependency = torch.mean(torch.mean(dependency, dim=1), dim=1).reshape(-1)
        # # importance = torch.mean(torch.mean(importance, dim=1), dim=1).reshape(-1)
        #
        edge_dependency = dependency[edge_index[0]]
        edge_importance = importance[edge_index[0]]

        edge_importance = edge_importance.cuda()
        edge_dependency = edge_dependency.cuda()
        #
        z1 = self.hyper_dependency_conv(z, edge_index, edge_dependency, batch)
        z2 = self.hyper_importance_conv(z, edge_index, edge_importance, batch)

        # output = self.hyper_conv(z, edge_index, edge_weight, batch)
        # z2 = self.hyper_conv(z, edge_index, edge_importance, batch)

        # c1 = self.lstm_dependency(sequence_dependency)
        # c2 = self.lstm_importance(sequence_importance)

        output = z1 + z2

        return output
