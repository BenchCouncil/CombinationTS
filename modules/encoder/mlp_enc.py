import torch
from torch import nn
from arch import BaseModule


class EncLayer(nn.Module):
    def __init__(self, d_model, c_dim, dropout, channel_independence):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff1 = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.ff2 = nn.Sequential(
            nn.Linear(c_dim, c_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.channel_independence = channel_independence
    
    def forward(self, x):
        B, C, L, D = x.shape
        if self.channel_independence:
            x = x.reshape(B * C, L, D)
        else:
            x = x.reshape(B, C * L, D)
        y_0 = self.ff1(x)
        y_0 = y_0 + x
        y_0 = self.norm1(y_0)
        y_1 = y_0.permute(0, 2, 1)
        y_1 = self.ff2(y_1)
        y_1 = y_1.permute(0, 2, 1)
        y_2 = y_1 * y_0 + x
        y_2 = self.norm2(y_2)
        y_2 = y_2.reshape(B, C, L, D)
        return y_2


class MLPEnc(BaseModule):
    def __init__(self, input_shape, e_layers, dropout, channel_independence, **kwargs):
        super().__init__(input_shape)
        C, L, D = input_shape
        self.channel_independence = channel_independence
        if self.channel_independence:
            c_dim = L
        else:
            c_dim = C * L
        self.d_model = D
        self.enc_in = C
        self.layers = nn.ModuleList([
            EncLayer(d_model=D, c_dim=c_dim, dropout=dropout, channel_independence=channel_independence)
            for _ in range(e_layers)
        ])
        self.set_output_shape_template((C, L, D))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
