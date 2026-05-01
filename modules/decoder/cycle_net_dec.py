import torch
from torch import nn
from arch import BaseModule

class Head(BaseModule):
    def __init__(self, input_shape, seq_len, pred_len, n_vars, d_model, dropout, **kwargs):
        assert input_shape[-1] == 1, "The D_model dimension of input_shape must be 1."
        super(Head, self).__init__(input_shape)
        self.model = nn.Sequential(
            nn.Linear(seq_len, d_model),
            nn.ReLU(),
            nn.Linear(d_model, pred_len)
        )
        self.dropout = nn.Dropout(dropout)
        self.set_output_shape_template((n_vars, pred_len, 1))
    
    def forward(self, x):
        x = x.squeeze(-1)  # (B, C, seq_len)
        x = self.model(x)  # (B, C, pred_len)
        x = x.unsqueeze(-1)  # (B, C, pred_len, 1)
        x = self.dropout(x)
        return x