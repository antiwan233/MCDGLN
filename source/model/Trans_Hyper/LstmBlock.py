import torch
import torch.nn as nn
from omegaconf import DictConfig


class LstmBlock(nn.Module):

    def __init__(self,
                 cfg: DictConfig):

        super().__init__()

        self.input_size = cfg.model.lstm_input_size
        self.hidden_size = cfg.model.lstm_hidden_size
        self.num_layers = cfg.model.lstm_num_layers

        self.lstm = nn.LSTM(input_size=self.input_size,
                            hidden_size=self.hidden_size,
                            num_layers=self.num_layers,
                            batch_first=True,
                            dropout=0.5 if self.num_layers > 1 else 0)

        self.lin1 = nn.Linear(self.hidden_size, 2)

    def forward(self, x):

        z = x

        h_0 = torch.zeros(self.num_layers, z.size(0), self.hidden_size).cuda()
        c_0 = torch.zeros(self.num_layers, z.size(0), self.hidden_size).cuda()

        output, _ = self.lstm(z, (h_0, c_0))
        output = self.lin1(output[:,-1,:])

        return output
