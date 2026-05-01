from typing_extensions import Self
import torch
from torch import nn
from utils.tools import is_list

class Normalize(nn.Module):
    def __init__(self):
        super(Normalize, self).__init__()

    def _normalize(self, x, dim=2):
        means = x.mean(dim, keepdim=True).detach()
        x = x - means
        stdev = torch.sqrt(
            torch.var(x, dim=dim, keepdim=True, unbiased=False) + 1e-5)
        x /= stdev
        return x, means, stdev

    def forward(self, x, type='norm'):
        if is_list(x) and len(x) == 1:
            x = x[0]
        if type == 'norm':
            x, self.means, self.stdev = self._normalize(x)
        elif type == 'denorm':
            x = self._denorm(x)
        else:
            raise ValueError(f"Unsupported type: {type}")
        return x
    
    def _denorm(self, x):
        x = x * self.stdev + self.means
        return x

class MultiStreamNorm(nn.Module):
    def __init__(self, input_shape, **kwargs):
        super(MultiStreamNorm, self).__init__()
        use_multi_stream = kwargs.get('use_multi_stream', False)
        self.stream_num = len(input_shape) if use_multi_stream else 1
        self.norm_layers = nn.ModuleList([Normalize() for _ in range(self.stream_num)])
        self.output_shape = input_shape
    
    def forward(self, x_list, type='norm'):
        if type == 'norm':
            normed_x_list = []
            for i in range(self.stream_num):
                normed_x = self.norm_layers[i](x_list[i])
                normed_x_list.append(normed_x)
            return normed_x_list
        elif type == 'denorm':
            return self._denorm(x_list)
        else:
            raise ValueError(f"Unsupported type: {type}")
    
    def _denorm(self, x):
        # only use the first stream's norm parameters to denorm
        x = [x] if not is_list(x) else x
        denormed_x_list = []
        for i in range(len(x)):
            denormed_x = self.norm_layers[i](x[i], 'denorm')
            denormed_x_list.append(denormed_x)
        return denormed_x_list