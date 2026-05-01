from typing_extensions import Self
import torch
from torch import nn
from arch import BaseModule

class ConvLinear(BaseModule):
    def __init__(self, input_shape, n_vars, pred_len, kernel_size=8, **kwargs):
        super(ConvLinear, self).__init__(input_shape)
        C, L, D = input_shape
        assert C == n_vars, "In ConvLinear, input channels must be equal to number of variables."
        self.flatten = nn.Flatten(start_dim=-2) # (B, C, L, D) -> (B, C, L*D)
        self.fuse_proj = nn.Conv2d(
            in_channels=1,
            out_channels=1,
            kernel_size=(4+C, kernel_size),
            padding='same'
        )
        self.linear = nn.Linear(L*D, pred_len)
        self.set_output_shape_template((n_vars, pred_len, 1))
    
    def forward(self, x):
        x = self.flatten(x.contiguous()).unsqueeze(1)  # (B, 1, C, L*D)
        x = self.fuse_proj(x)                          # (B, 1, C, L*D)
        x = x.squeeze(1)                              # (B, C, L*D)
        x = self.linear(x)                            # (B, C, pred_len)
        x = x.unsqueeze(-1)                          # (B, C, pred_len, 1)
        return x