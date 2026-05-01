import torch
from torch import nn
from arch import BaseModule

class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """

    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class TSDecomp(nn.Module):
    def __init__(self, kernel_size):
        super(TSDecomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean


class TrendSeasonalDecomp(BaseModule):
    """
    Series decomposition block
    """

    def __init__(self, kernel_size, input_shape, **kwargs):
        super(TrendSeasonalDecomp, self).__init__(input_shape)
        self.moving_avg = moving_avg(kernel_size, stride=1)
        n_vars, seq_len, _ = input_shape
        output_shape = [(n_vars, seq_len, 1)] * 2
        self.set_output_shape_template(output_shape)

    def forward(self, x, x_mark=None):
        x = x.squeeze(-1).permute(0, 2, 1)  # (B, N, L, 1) -> (B, L, N)
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        seasonal, trend = \
            res.permute(0, 2, 1).unsqueeze(-1), moving_mean.permute(0, 2, 1).unsqueeze(-1)
        x_list = [seasonal, trend]
        x_mark_list = [x_mark, x_mark] if x_mark is not None else [None, None]
        return x_list, x_mark_list


class DFT_series_decomp(nn.Module):
    """
    Series decomposition block
    """

    def __init__(self, top_k: int = 5):
        super(DFT_series_decomp, self).__init__()
        self.top_k = top_k

    def forward(self, x):
        xf = torch.fft.rfft(x)
        freq = abs(xf)
        freq[0] = 0
        top_k_freq, top_list = torch.topk(freq, k=self.top_k)
        xf[freq <= top_k_freq.min()] = 0
        x_season = torch.fft.irfft(xf)
        x_trend = x - x_season
        return x_season, x_trend


class MultiScaleDecomp(BaseModule):
    def __init__(self, input_shape, **kwargs):
        super(MultiScaleDecomp, self).__init__(input_shape)
        n_vars, seq_len, _ = input_shape
        down_sampling_method, down_sampling_window, down_sampling_layers = \
            kwargs['down_sampling_method'], kwargs['down_sampling_window'], kwargs['down_sampling_layers']
        if down_sampling_method == 'max':
            down_pool = nn.MaxPool1d(down_sampling_window, return_indices=False)
        elif down_sampling_method == 'avg':
            down_pool = nn.AvgPool1d(down_sampling_window)
        elif down_sampling_method == 'conv':
            down_pool = nn.Conv1d(in_channels=n_vars, out_channels=n_vars, kernel_size=down_sampling_window, stride=down_sampling_window)
        else:
            raise ValueError("Unsupported down_sampling_method. Choose from 'max', 'avg', or 'conv'.")
        self.down_pool = down_pool
        self.down_sampling_layers = down_sampling_layers
        self.down_sampling_window = down_sampling_window
        output_shape = [(n_vars, seq_len // (down_sampling_window ** i), 1) for i in range(down_sampling_layers + 1)]
        self.set_output_shape_template(output_shape)

    def forward(self, x, x_mark):
        x_sampling = x
        x_mark_sampling = x_mark
        x_sampling_list = []
        x_mark_sampling_list = []
        x_sampling_list.append(x_sampling)
        x_mark_sampling_list.append(x_mark_sampling)
        x_sampling = x_sampling.squeeze(-1) # (B, N, L, 1) -> (B, N, L)

        for _ in range(self.down_sampling_layers):
            x_sampling = self.down_pool(x_sampling)
            x_sampling_list.append(x_sampling.unsqueeze(-1))

            if x_mark is not None:
                x_mark_sampling = x_mark_sampling[:, :, ::self.down_sampling_window, :]
                x_mark_sampling_list.append(x_mark_sampling)
        return x_sampling_list, x_mark_sampling_list

