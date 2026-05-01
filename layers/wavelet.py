import torch
from torch import nn
import torch.nn.functional as F
import pywt

class WaveletEmbedding(nn.Module):
    def __init__(self, d_channel=16, swt=True, requires_grad=False, wv='db2', m=2,
                 kernel_size=None):
        super().__init__()

        self.swt = swt
        self.d_channel = d_channel
        self.m = m  # Number of decomposition levels of detailed coefficients
        
        if kernel_size is None:
            self.wavelet = pywt.Wavelet(wv) # type: ignore
            if self.swt:
                h0 = torch.tensor(self.wavelet.dec_lo[::-1], dtype=torch.float32)
                h1 = torch.tensor(self.wavelet.dec_hi[::-1], dtype=torch.float32)
            else:
                h0 = torch.tensor(self.wavelet.rec_lo[::-1], dtype=torch.float32)
                h1 = torch.tensor(self.wavelet.rec_hi[::-1], dtype=torch.float32)
            self.h0 = nn.Parameter(torch.tile(h0[None, None, :], [self.d_channel, 1, 1]), requires_grad=requires_grad)
            self.h1 = nn.Parameter(torch.tile(h1[None, None, :], [self.d_channel, 1, 1]), requires_grad=requires_grad)
            self.kernel_size = self.h0.shape[-1]
        else:
            self.kernel_size = kernel_size
            self.h0 = nn.Parameter(torch.Tensor(self.d_channel, 1, self.kernel_size), requires_grad=requires_grad)
            self.h1 = nn.Parameter(torch.Tensor(self.d_channel, 1, self.kernel_size), requires_grad=requires_grad)
            nn.init.xavier_uniform_(self.h0)
            nn.init.xavier_uniform_(self.h1)
        
            with torch.no_grad():
                self.h0.data = self.h0.data / torch.norm(self.h0.data, dim=-1, keepdim=True)
                self.h1.data = self.h1.data / torch.norm(self.h1.data, dim=-1, keepdim=True)


    def forward(self, x):
        if self.swt:
            coeffs = self.swt_decomposition(x, self.h0, self.h1, self.m, self.kernel_size)
        else:
            coeffs = self.swt_reconstruction(x, self.h0, self.h1, self.m, self.kernel_size)
        return coeffs

    def swt_decomposition(self, x, h0, h1, depth, kernel_size):
        approx_coeffs = x
        coeffs = []
        dilation = 1
        for _ in range(depth):
            padding = dilation * (kernel_size - 1)
            padding_r = (kernel_size * dilation) // 2
            pad = (padding - padding_r, padding_r)
            approx_coeffs_pad = F.pad(approx_coeffs, pad, "circular")
            detail_coeff = F.conv1d(approx_coeffs_pad, h1, dilation=dilation, groups=x.shape[1])
            approx_coeffs = F.conv1d(approx_coeffs_pad, h0, dilation=dilation, groups=x.shape[1])
            coeffs.append(detail_coeff)
            dilation *= 2
        coeffs.append(approx_coeffs)

        return torch.stack(list(reversed(coeffs)), -2)

    def swt_reconstruction(self, coeffs, g0, g1, m, kernel_size):
        dilation = 2 ** (m - 1)
        approx_coeff = coeffs[:,:,0,:]
        detail_coeffs = coeffs[:,:,1:,:]
        
        for i in range(m):
            detail_coeff = detail_coeffs[:,:,i,:]
            padding = dilation * (kernel_size - 1)
            padding_l = (dilation * kernel_size) // 2
            pad = (padding_l, padding - padding_l)
            approx_coeff_pad = F.pad(approx_coeff, pad, "circular")
            detail_coeff_pad = F.pad(detail_coeff, pad, "circular")
            
            y = F.conv1d(approx_coeff_pad, g0, groups=approx_coeff.shape[1], dilation=dilation) + \
                F.conv1d(detail_coeff_pad, g1, groups=detail_coeff.shape[1], dilation=dilation)
            approx_coeff = y / 2
            dilation //= 2
            
        return approx_coeff
