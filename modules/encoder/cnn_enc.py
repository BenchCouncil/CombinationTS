import torch
from torch import nn
from torch.nn import functional as F
from layers.Conv_Blocks import Inception_Block_V1
from arch import BaseModule

def FFT_for_Period(x, k=2):
    # [B, T, C]
    xf = torch.fft.rfft(x, dim=1)
    # find period by amplitudes
    frequency_list = abs(xf).mean(0).mean(-1)
    frequency_list[0] = 0
    _, top_list = torch.topk(frequency_list, k)
    top_list = top_list.detach().cpu().numpy()
    period = x.shape[1] // top_list
    return period, abs(xf).mean(-1)[:, top_list]

class TimesBlock(nn.Module):
    def __init__(self, d_model, d_ff, num_kernels, top_k):
        self.d_model, self.d_ff, self.num_kernels, self.top_k = d_model, d_ff, num_kernels, top_k
        super(TimesBlock, self).__init__()
        self.conv = nn.Sequential(
            Inception_Block_V1(d_model, d_ff, num_kernels),
            nn.GELU(),
            Inception_Block_V1(d_ff, d_model, num_kernels)
        )
    
    def forward(self, x):
        B, L, D = x.shape
        period_list, period_weight = FFT_for_Period(x, self.top_k)
        res = []
        for i in range(self.top_k):
            period = period_list[i]
            # padding
            if (L) % period != 0:
                length = ((L // period) + 1) * period
                padding = torch.zeros([x.shape[0], (length - L), x.shape[2]]).to(x.device)
                out = torch.cat([x, padding], dim=1)
            else:
                length = L
                out = x
            # reshape
            out = out.reshape(B, length // period, period,
                              D).permute(0, 3, 1, 2).contiguous()
            # 2D conv: from 1d Variation to 2d Variation
            out = self.conv(out)
            # reshape back
            out = out.permute(0, 2, 3, 1).reshape(B, -1, D)
            res.append(out[:, :L, :])
        res = torch.stack(res, dim=-1)
        # adaptive aggregation
        period_weight = F.softmax(period_weight, dim=1)
        period_weight = period_weight.unsqueeze(
            1).unsqueeze(1).repeat(1, L, D, 1)
        res = torch.sum(res * period_weight, -1)
        # residual connection
        res = res + x
        return res
        

class TimesNetEnc(BaseModule):
    def __init__(self, input_shape, **kwargs):
        super(TimesNetEnc, self).__init__(input_shape)
        seq_len, pred_len, top_k, d_ff, num_kernels = \
            kwargs['seq_len'], kwargs['pred_len'], kwargs['top_k'], kwargs['d_ff'], kwargs['num_kernels']
        self.channel_independence = kwargs['channel_independence']
        e_layers = kwargs['e_layers']
        C,L,d_model = input_shape
        self.layers = nn.ModuleList(
            [TimesBlock(d_model=d_model, d_ff=d_ff, num_kernels=num_kernels, top_k=top_k) for l in range(e_layers)]
        )
        self.layer_nom = nn.LayerNorm(d_model)
        self.set_output_shape_template(tuple(input_shape))
    
    def forward(self, x):
        # x: (B, C, L, D)
        B, C, L, D = x.shape
        x = x.reshape(B * C, L, D)
        for i in range(len(self.layers)):
            x = self.layer_nom(self.layers[i](x))
        x = x.reshape(B, C, L, D)
        return x