import torch
from torch import nn
from arch import BaseModule

class CDLinearHead(BaseModule):
    def __init__(self, input_shape, pred_len, n_vars, **kwargs):
        super(CDLinearHead, self).__init__(input_shape)
        C, L, D = input_shape
        self.flatten = nn.Flatten(start_dim=1)  # (B, C, L, D) -> (B, C*L*D)
        self.head = nn.Linear(C*L*D, pred_len*n_vars)
        
        self.pred_len, self.n_vars = pred_len, n_vars
        self.set_output_shape_template((n_vars, pred_len, 1))

    def forward(self, x):
        x = self.flatten(x)           # (B, C*L*D)
        x = self.head(x)              # (B, pred_len*n_vars)
        x = x.reshape(-1, self.n_vars, self.pred_len, 1)
        return x