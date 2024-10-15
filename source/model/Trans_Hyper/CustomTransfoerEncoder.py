import torch
import torch.nn as nn
from omegaconf import DictConfig
from .CustomTransEncoderLayer import CustomTransformerEncoderLayer


class CustomTransformerEncoder(nn.Module):

    def __init__(self, cfg: DictConfig):
        super(CustomTransformerEncoder, self).__init__()

        self.batch_size = cfg.training.batch_size
        self.encoder_layer = CustomTransformerEncoderLayer(cfg.model.transformer_d_model,
                                                           cfg.model.nhead,
                                                           cfg.model.d_feedforward,
                                                           cfg.model.trans_dropout,
                                                           batch_first=True)
        self.layers = self._get_clones(self.encoder_layer, cfg.model.num_transformer_encoder_layers)

    def forward(self, src, mask=None, src_key_padding_mask=None):
        output = src.reshape(self.batch_size, -1, src.shape[1])
        dependency_heads_weights = []
        importance_heads_weights = []
        for layer in self.layers:
            output, attn_weight_before_soft = layer(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)
            attn_weight_before_soft = attn_weight_before_soft.reshape(self.batch_size,
                                                                      -1,
                                                                      attn_weight_before_soft.shape[2],
                                                                      attn_weight_before_soft.shape[2])

            dependency_heads_weights.append(torch.sum(attn_weight_before_soft,dim=3))
            importance_heads_weights.append(torch.sum(attn_weight_before_soft, dim=2))

        # shape [batch_size, num_layers, num_heads,num_nodes]
        dependency = torch.stack(dependency_heads_weights, dim=0).transpose(0,1)
        importance = torch.stack(importance_heads_weights, dim=0).transpose(0,1)

        return output, dependency, importance

    def _get_clones(self, module, n):
        return nn.ModuleList([module for i in range(n)])
