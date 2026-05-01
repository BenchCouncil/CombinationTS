import torch
from torch import nn
from arch import BaseModule

class LinearHead(BaseModule):
    def __init__(self, input_shape, **kwargs):
        super(LinearHead, self).__init__(input_shape)
        separate_length_proj, pred_len, n_vars = kwargs['separate_length_proj'], kwargs['pred_len'], kwargs['n_vars']
        C, L, D = input_shape
        self.separate_length_proj = separate_length_proj
        self.channel_mixing = C == 1 # we force 'C' to represent channel number, so if C==1, it means channel_mixing is True
        if self.separate_length_proj and self.channel_mixing:
            self.model = LinearHead_wiLengthProj_wiChannelMixing(input_shape, **kwargs)
        elif self.separate_length_proj and not self.channel_mixing:
            self.model = LinearHead_wiLengthProj_woChannelMixing(input_shape, **kwargs)
        elif not self.separate_length_proj and self.channel_mixing:
            self.model = LinearHead_woLengthProj_wiChannelMixing(input_shape, **kwargs)
        else: # not self.separate_length_proj and not self.channel_mixing:
            self.model = LinearHead_woLengthProj_woChannelMixing(input_shape, **kwargs)
        self.set_output_shape_template((n_vars, pred_len, 1))
    def forward(self, x):
        return self.model(x)

class LinearHead_wiLengthProj_woChannelMixing(nn.Module):
    def __init__(self, input_shape, **kwargs):
        super().__init__()
        pred_len, n_vars, dropout, init_parameters = kwargs['pred_len'], kwargs['n_vars'], kwargs['dropout'], kwargs['init_parameters']
        C, L, D = input_shape
        self.pred_len, self.n_vars = pred_len, n_vars
        self.length_projector = nn.Linear(L, pred_len)
        self.feature_projector = nn.Linear(D, 1)
        if init_parameters:
            self.length_projector.weight = nn.Parameter((1 / L) * torch.ones([pred_len, L]))
            self.feature_projector.weight = nn.Parameter((1 / D) * torch.ones([1, D]))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):                   # (B, C, L, D)
        x = x.permute(0, 1, 3, 2)           # (B, C, D, L)
        x = self.length_projector(x)        # (B, C, D, pred_len)
        x = x.permute(0, 1, 3, 2)           # (B, C, pred_len, D)
        x = self.feature_projector(x)       # (B, C, pred_len, 1)
        x = self.dropout(x)
        return x

class LinearHead_woLengthProj_woChannelMixing(nn.Module):
    def __init__(self, input_shape, **kwargs):
        super().__init__()
        pred_len, n_vars, dropout, init_parameters = kwargs['pred_len'], kwargs['n_vars'], kwargs['dropout'], kwargs['init_parameters']
        C, L, D = input_shape
        self.pred_len, self.n_vars = pred_len, n_vars
        self.flatten = nn.Flatten(start_dim=-2) # (B, C, L, D) -> (B, C, L*D)
        self.linear = nn.Linear(L * D, pred_len)
        if init_parameters:
            self.linear.weight = nn.Parameter((1 / (L*D)) * torch.ones([pred_len, L*D]))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):                   # (B, C, L, D)
        x = self.flatten(x.contiguous())    # (B, C, L*D)
        x = self.linear(x)                  # (B, C, pred_len)
        x = x.unsqueeze(-1)               # (B, C, pred_len, 1)
        x = self.dropout(x)
        return x

class LinearHead_wiLengthProj_wiChannelMixing(nn.Module):
    def __init__(self, input_shape, **kwargs):
        super().__init__()
        pred_len, n_vars, dropout, init_parameters = kwargs['pred_len'], kwargs['n_vars'], kwargs['dropout'], kwargs['init_parameters']
        C, L, D = input_shape
        self.pred_len, self.n_vars = pred_len, n_vars
        self.length_projector = nn.Linear(L, pred_len)
        self.feature_projector = nn.Linear(D, n_vars)
        if init_parameters:
            self.length_projector.weight = nn.Parameter((1 / L) * torch.ones([pred_len, L]))
            self.feature_projector.weight = nn.Parameter((1 / D) * torch.ones([n_vars, D]))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):   # (B, 1, L, D)
        x = x.squeeze(1).permute(0, 2, 1)   # (B, D, L)
        x = self.length_projector(x)        # (B, D, pred_len)
        x = x.permute(0, 2, 1)              # (B, pred_len, D)
        x = self.feature_projector(x)       # (B, pred_len, n_vars)
        x = x.permute(0, 2, 1).unsqueeze(-1)# (B, n_vars, pred_len, 1)
        x = self.dropout(x)
        return x

class LinearHead_woLengthProj_wiChannelMixing(nn.Module):
    def __init__(self, input_shape, **kwargs):
        super().__init__()
        pred_len, n_vars, dropout, init_parameters = kwargs['pred_len'], kwargs['n_vars'], kwargs['dropout'], kwargs['init_parameters']
        C, L, D = input_shape
        self.pred_len, self.n_vars = pred_len, n_vars
        self.flatten = nn.Flatten(start_dim=-3)  # (B, C, L, D) -> (B, C*L*D)
        self.linear = nn.Linear(C*L*D, n_vars*pred_len)
        if init_parameters:
            self.linear.weight = nn.Parameter((1 / (C*L*D)) * torch.ones([C*pred_len, C*L*D]))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):                   # (B, C, L, D)
        x = self.flatten(x.contiguous())    # (B, C*L*D)
        x = self.linear(x)                  # (B, n_vars*pred_len)
        x = x.reshape(-1, self.pred_len, self.n_vars)  # (B, pred_len, n_vars)
        x = x.permute(0, 2, 1).unsqueeze(-1)  # (B, n_vars, pred_len, 1)
        x = self.dropout(x)
        return x