import torch
from torch import nn
from arch import BaseModule

class FreTSDec(BaseModule):
    def __init__(self, input_shape, pred_len, n_vars, hidden_size=None, **kwargs):
        super(FreTSDec, self).__init__(input_shape)
        C, L, D = input_shape
        if hidden_size is None:
            hidden_size = 2*D
        self.flatten = nn.Flatten(start_dim=-2)    # (B, C, L, D) -> (B, C, L*D)
        self.model = nn.Sequential(
            nn.Linear(L * D, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, pred_len),        # (B, C, pred_len)
        )
        self.set_output_shape_template((n_vars, pred_len, 1))
    def forward(self, x):
        x = self.flatten(x)
        x = self.model(x)
        x = x.unsqueeze(-1)   # (B, C, pred_len, 1)
        return x