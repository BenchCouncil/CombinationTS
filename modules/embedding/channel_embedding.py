import torch
from torch import nn
from arch import BaseModule

class ChannelEmbedding(BaseModule):
    def __init__(self, input_shape, d_model, dropout, channel_independence, **kwargs):
        super(ChannelEmbedding, self).__init__(input_shape)
        n_vars, seq_len, _ = input_shape
        assert not channel_independence, "ChannelEmbedding does not support channel_independence!"
        self.value_embedding = nn.Linear(seq_len, d_model)
        self.dropout = nn.Dropout(p=dropout)
        self.set_output_shape_template((n_vars, 1, d_model))

    def forward(self, x, x_mark=None):
        x = x.squeeze(-1)
        x = self.value_embedding(x)
        x = self.dropout(x)
        x = x.unsqueeze(2)
        return x