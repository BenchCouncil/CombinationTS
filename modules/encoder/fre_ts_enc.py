import torch
from torch import nn
from arch import BaseModule
from torch.nn import functional as F

class FreTSEnc(BaseModule):
    def __init__(self, input_shape, channel_independence=True, scale=0.02, sparsity_threshold=0.01, **kwargs):
        super(FreTSEnc, self).__init__(input_shape)
        C, L, D = input_shape
        self.channel_independence = channel_independence
        self.embed_size = D
        self.feature_size = C
        self.length = L
        self.scale = scale
        self.sparsity_threshold = sparsity_threshold
        if not self.channel_independence:
            self.r1 = nn.Parameter(scale * torch.randn(D, D))
            self.i1 = nn.Parameter(scale * torch.randn(D, D))
            self.rb1 = nn.Parameter(scale * torch.randn(D))
            self.ib1 = nn.Parameter(scale * torch.randn(D))
        
        self.r2 = nn.Parameter(scale * torch.randn(D, D))
        self.i2 = nn.Parameter(scale * torch.randn(D, D))
        self.rb2 = nn.Parameter(scale * torch.randn(D))
        self.ib2 = nn.Parameter(scale * torch.randn(D))

        self.set_output_shape_template((C, L, D))

    def FreMLP(self, B, nd, dimension, x, r, i, rb, ib):
        o1_real = F.relu(
            torch.einsum('bijd,dd->bijd', x.real, r) - \
            torch.einsum('bijd,dd->bijd', x.imag, i) + \
            rb
        )

        o1_imag = F.relu(
            torch.einsum('bijd,dd->bijd', x.imag, r) + \
            torch.einsum('bijd,dd->bijd', x.real, i) + \
            ib
        )

        y = torch.stack([o1_real, o1_imag], dim=-1)
        y = F.softshrink(y, lambd=self.sparsity_threshold)
        y = torch.view_as_complex(y)
        return y

    def MLP_temporal(self, x, B, N, L):
        # [B, N, T, D]
        x = torch.fft.rfft(x, dim=2, norm='ortho')  # FFT on L dimension
        y = self.FreMLP(B, N, L, x, self.r2, self.i2, self.rb2, self.ib2)
        x = torch.fft.irfft(y, n=self.length, dim=2, norm="ortho")
        return x
    
    def MLP_channel(self, x, B, N, L):
        # [B, N, T, D]
        x = x.permute(0, 2, 1, 3)
        # [B, T, N, D]
        x = torch.fft.rfft(x, dim=2, norm='ortho')  # FFT on N dimension
        y = self.FreMLP(B, L, N, x, self.r1, self.i1, self.rb1, self.ib1)
        x = torch.fft.irfft(y, n=self.feature_size, dim=2, norm="ortho")
        x = x.permute(0, 2, 1, 3)
        # [B, N, T, D]
        return x

    def forward(self, x):
        # x: [B, C, L, D]
        B, N, L, D = x.shape
        bias = x
        if not self.channel_independence:
            x = self.MLP_channel(x, B, N, L)
        x = self.MLP_temporal(x, B, N, L)
        x = x + bias
        return x