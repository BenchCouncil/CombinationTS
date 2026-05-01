import torch
from torch import nn
from torch.nn import functional as F
from layers.wavelet import WaveletEmbedding
from math import sqrt


class GeomAttentionLayer(nn.Module):
    def __init__(self, attention, d_model, d_channel,
                 requires_grad=True, wv='db2', m=2, kernel_size=None,
                 geomattn_dropout=0.5,):
        super(GeomAttentionLayer, self).__init__()

        self.d_channel = d_channel
        self.inner_attention = attention
        
        self.swt = WaveletEmbedding(d_channel=self.d_channel, swt=True, requires_grad=requires_grad, wv=wv, m=m, kernel_size=kernel_size)
        self.query_projection = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Dropout(geomattn_dropout)
        )
        self.key_projection = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Dropout(geomattn_dropout)
        )
        self.value_projection = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Dropout(geomattn_dropout)
        )
        self.out_projection = nn.Sequential(
            nn.Linear(d_model, d_model),
            WaveletEmbedding(d_channel=self.d_channel, swt=False, requires_grad=requires_grad, wv=wv, m=m, kernel_size=kernel_size),
        )
        
    def forward(self, queries, keys, values, attn_mask=None, tau=None, delta=None):
        queries = self.swt(queries)
        keys = self.swt(keys)
        values = self.swt(values)

        queries = self.query_projection(queries).permute(0,3,2,1)
        keys = self.key_projection(keys).permute(0,3,2,1)
        values = self.value_projection(values).permute(0,3,2,1)

        out, attn = self.inner_attention(
            queries,
            keys,
            values,
        )

        out = self.out_projection(out.permute(0,3,2,1))

        return out, attn


class GeomAttention(nn.Module):
    def __init__(self, mask_flag=False, factor=5, scale=None, attention_dropout=0.1, 
                 output_attention=False,
                 alpha=1.,):
        super(GeomAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)
        
        self.alpha = alpha 

    def forward(self, queries, keys, values, attn_mask=None):
        B, L, H, E = queries.shape
        _, S, _, _ = values.shape
        scale = self.scale or 1. / sqrt(E)

        dot_product = torch.einsum("blhe,bshe->bhls", queries, keys)

        queries_norm2 = torch.sum(queries**2, dim=-1)
        keys_norm2 = torch.sum(keys**2, dim=-1)
        queries_norm2 = queries_norm2.permute(0, 2, 1).unsqueeze(-1)         # (B, H, L, 1)
        keys_norm2 = keys_norm2.permute(0, 2, 1).unsqueeze(-2)               # (B, H, 1, S)
        wedge_norm2 = queries_norm2 * keys_norm2 - dot_product ** 2          # (B, H, L, S)
        wedge_norm2 = F.relu(wedge_norm2)
        wedge_norm = torch.sqrt(wedge_norm2 + 1e-8)

        scores = (1 - self.alpha) * dot_product + self.alpha * wedge_norm
        scores = scores * scale

        if self.mask_flag:
            if attn_mask is None:
                attn_mask = torch.tril(torch.ones(L, S)).to(scores.device)
            scores.masked_fill_(attn_mask.unsqueeze(1).unsqueeze(2) == 0, float('-inf'))

        A = self.dropout(torch.softmax(scores, dim=-1)) 

        V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            return V.contiguous()
        else:
            return (V.contiguous(), scores.abs().mean())

