import torch
from torch import nn
import torch.nn.functional as F
import layers
from layers.self_attention_family import FullAttention, AttentionLayer
from layers.swt_attention_family import GeomAttention, GeomAttentionLayer
from arch import BaseModule

class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        new_x, attn = self.attention(
            x, x, x,
            attn_mask=attn_mask,
            tau=tau, delta=delta
        )
        x = x + self.dropout(new_x)

        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm2(x + y), attn

class Encoder(nn.Module):
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = attn_layers if isinstance(attn_layers, nn.ModuleList) else nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        # x [B, L, D]
        attns = []
        if self.conv_layers is not None:
            for i, (attn_layer, conv_layer) in enumerate(zip(self.attn_layers, self.conv_layers)):
                delta = delta if i == 0 else None
                x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
                x = conv_layer(x)
                attns.append(attn)
            x, attn = self.attn_layers[-1](x, tau=tau, delta=None)
            attns.append(attn)
        else:
            for attn_layer in self.attn_layers:
                x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
                attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)

        return x, attns

class Transpose(nn.Module):
    def __init__(self, *dims, contiguous=False): 
        super().__init__()
        self.dims, self.contiguous = dims, contiguous
    def forward(self, x):
        if self.contiguous: return x.transpose(*self.dims).contiguous()
        else: return x.transpose(*self.dims)

class TransformerEnc(BaseModule):
    def __init__(self, input_shape, **kwargs):
        super(TransformerEnc, self).__init__(input_shape)
        factor, dropout, d_ff, activation, e_layers, channel_independence = \
            kwargs['factor'], kwargs['dropout'], kwargs['d_ff'], kwargs['activation'], kwargs['e_layers'], kwargs['channel_independence']
        # for full attention
        n_heads = kwargs.get('n_heads', 2)
        # for geom attention
        alpha = kwargs.get('alpha', 0.3)
        requires_grad = kwargs.get('requires_grad', True)
        wv = kwargs.get('wv', 'db1')
        m = kwargs.get('m', 3)
        kernel_size = kwargs.get('kernel_size', None)
        geomattn_dropout = kwargs.get('geomattn_dropout', 0.5)

        attention = kwargs.get('attention', 'full_attention')

        C, L, D = input_shape
        self.channel_independence = channel_independence

        # build encoder layers
        if attention == 'full_attention':
            assert D % n_heads == 0, "D should be divided by n_heads for FullAttention."
            self.layers = nn.ModuleList(
                [
                    EncoderLayer(
                        AttentionLayer(
                            FullAttention(False, factor, attention_dropout=dropout,
                                          output_attention=False), D, n_heads),
                        D,
                        d_ff,
                        dropout=dropout,
                        activation=activation
                    ) for l in range(e_layers)
                ],
            )
        elif attention == 'geom_attention':
            d_channel = L if self.channel_independence else C * L
            self.layers = nn.ModuleList(
                [
                    EncoderLayer(
                        GeomAttentionLayer(
                            GeomAttention(
                                False, factor, attention_dropout=dropout,
                                output_attention=False, alpha=alpha
                            ),
                            D,
                            requires_grad=requires_grad,
                            wv=wv,
                            m=m,
                            d_channel=d_channel,
                            kernel_size=kernel_size,
                            geomattn_dropout=geomattn_dropout
                        ),
                        D,
                        d_ff,
                        dropout=dropout,
                        activation=activation,
                    ) for l in range(e_layers)
                ]
            )
        else:
            raise ValueError(f"Unknown attention type: {attention}")


        self.encoder = Encoder(
            self.layers,
            norm_layer=nn.Sequential(Transpose(1,2), nn.BatchNorm1d(D), Transpose(1,2))
        )
        self.set_output_shape_template((C, -1, D))

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        B, C, L, D = x.shape
        reshape = (B * C, L, D) if self.channel_independence else (B, C * L, D)
        x = torch.reshape(x, reshape)
        x, attn = self.encoder(x, attn_mask=attn_mask, tau=tau, delta=delta)
        x = torch.reshape(x, (B, C, L, D))
        # [B, C, L, D]
        return x
