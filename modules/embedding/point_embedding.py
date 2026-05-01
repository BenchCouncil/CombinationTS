import torch
from torch import nn
import math
from arch import BaseModule

class PositionalEmbedding(nn.Module):
    pe: torch.Tensor
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.requires_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]

class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                   kernel_size=3, padding=padding, padding_mode='circular', bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x

class FixedEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(FixedEmbedding, self).__init__()

        w = torch.zeros(c_in, d_model).float()
        w.requires_grad = False

        position = torch.arange(0, c_in).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()

        w[:, 0::2] = torch.sin(position * div_term)
        w[:, 1::2] = torch.cos(position * div_term)

        self.emb = nn.Embedding(c_in, d_model)
        self.emb.weight = nn.Parameter(w, requires_grad=False)

    def forward(self, x):
        return self.emb(x).detach()

class TemporalEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='fixed', freq='h'):
        super(TemporalEmbedding, self).__init__()

        minute_size = 4
        hour_size = 24
        weekday_size = 7
        day_size = 32
        month_size = 13

        Embed = FixedEmbedding if embed_type == 'fixed' else nn.Embedding
        if freq == 't':
            self.minute_embed = Embed(minute_size, d_model)
        self.hour_embed = Embed(hour_size, d_model)
        self.weekday_embed = Embed(weekday_size, d_model)
        self.day_embed = Embed(day_size, d_model)
        self.month_embed = Embed(month_size, d_model)

    def forward(self, x):
        x = x.long()
        minute_x = self.minute_embed(x[:, :, 4]) if hasattr(
            self, 'minute_embed') else 0.
        hour_x = self.hour_embed(x[:, :, 3])
        weekday_x = self.weekday_embed(x[:, :, 2])
        day_x = self.day_embed(x[:, :, 1])
        month_x = self.month_embed(x[:, :, 0])

        return hour_x + weekday_x + day_x + month_x + minute_x

class TimeFeatureEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='timeF', freq='h'):
        super(TimeFeatureEmbedding, self).__init__()

        freq_map = {'h': 4, 't': 5, 's': 6,
                    'm': 1, 'a': 1, 'w': 2, 'd': 3, 'b': 3}
        d_inp = freq_map[freq]
        self.embed = nn.Linear(d_inp, d_model, bias=False)

    def forward(self, x):
        return self.embed(x)

class PointEmbedding(BaseModule):
    def __init__(self, input_shape, d_model, dropout,
                    channel_independence=True, use_positional=True,
                    use_temporal=True, temporal_embed_type='timeF',
                    freq='h', channel_mixing=False,
                 **kwargs):
        super(PointEmbedding, self).__init__(input_shape)
        n_vars, seq_len, _ = input_shape

        if channel_mixing is None:
            channel_mixing = not channel_independence
        assert not (channel_mixing and channel_independence), "DataEmbedding with channel_mixing must set channel_independence to False!"

        self.use_positional = use_positional
        self.use_temporal = use_temporal
        self.channel_independence = channel_independence
        self.channel_mixing = channel_mixing
        self.d_model = d_model

        if self.channel_mixing:
            self.value_embedding = TokenEmbedding(c_in=n_vars, d_model=d_model)
        else:
            self.value_embedding = TokenEmbedding(c_in=1, d_model=d_model)

        if self.use_positional:
            self.position_embedding = PositionalEmbedding(d_model=d_model)
        if self.use_temporal:
            assert freq is not None, "freq must be specified when use_temporal is True!"
            self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=temporal_embed_type, freq=freq) if temporal_embed_type != 'timeF' \
                else TimeFeatureEmbedding(d_model=d_model, embed_type=temporal_embed_type, freq=freq)
        self.dropout = nn.Dropout(p=dropout)

        self.set_output_shape_template((1, -1, d_model) if self.channel_mixing else (n_vars, -1, d_model))
        
    def forward(self, x, x_mark):
        # x (B, C, L, 1)
        x_mark = x_mark.squeeze(-1).permute(0, 2, 1).contiguous() if x_mark is not None else None  # x_mark: (B, L, C2)
        if self.channel_mixing:
            x = x.squeeze(-1).permute(0, 2, 1)  # x: (B, L, C)
            x = self.value_embedding(x) + \
                (self.temporal_embedding(x_mark) if self.use_temporal else 0) + \
                (self.position_embedding(x) if self.use_positional else 0 )
            x = x.unsqueeze(1)  # (B, 1, L, D)
        else:
            # x: (B, C, L, 1) -> (B, C, L)
            B, N, T, _ = x.shape
            x = x.reshape(B * N, T, 1)
            x_mark = x_mark.repeat(N, 1, 1) if x_mark is not None else None
            x = self.value_embedding(x) + \
                (self.temporal_embedding(x_mark) if self.use_temporal else 0) + \
                (self.position_embedding(x) if self.use_positional else 0 )
            x = x.reshape(-1, N, T, self.d_model) # (B, C, L, D)
        return self.dropout(x)