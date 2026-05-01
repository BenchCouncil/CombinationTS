import torch
from torch import nn
from layers.decompsition import TSDecomp, DFT_series_decomp
from utils.tools import is_list
from arch import BaseModule

class MultiStreamMixing(nn.Module):
    def __init__(self, seq_list, reverse=False):
        assert len(seq_list) > 1, "MultiStreamMixing requires more than one input"
        super(MultiStreamMixing, self).__init__()
        self.reverse = reverse
        if self.reverse:
            seq_list.reverse()
        self.layers = torch.nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(in_features=seq_list[i], out_features=seq_list[i + 1]),
                    nn.GELU(),
                    nn.Linear(in_features=seq_list[i + 1], out_features=seq_list[i + 1]),
                )
                for i in range(len(seq_list) - 1)
            ]
        )
    def forward(self, x):
        # x: (B, L, D) list
        if self.reverse:
            x.reverse()
        x = [item.permute(0, 2, 1) for item in x]  # (B, D, L)
        out_low = x[0]
        out_high = x[1]
        out_list = [out_low.permute(0, 2, 1)]
        for i in range(len(x) - 1):
            out_high_res = self.layers[i](out_low)
            out_high = out_high + out_high_res
            out_low = out_high
            if i + 2 <= len(x) - 1:
                out_high = x[i + 2]
            out_list.append(out_low.permute(0, 2, 1)) # (B, L, D)
        if self.reverse:
            out_list.reverse()
        return out_list

class MultiScaleSeasonMixing(nn.Module):
    """
    Bottom-up mixing season pattern
    """
    def __init__(self, seq_len, down_sampling_window, down_sampling_layers):
        super(MultiScaleSeasonMixing, self).__init__()
        self.down_sampling_layers = torch.nn.ModuleList(
            [
                nn.Sequential(
                    torch.nn.Linear(
                        seq_len // (down_sampling_window ** i),
                        seq_len // (down_sampling_window ** (i + 1)),
                    ),
                    nn.GELU(),
                    torch.nn.Linear(
                        seq_len // (down_sampling_window ** (i + 1)),
                        seq_len // (down_sampling_window ** (i + 1)),
                    ),
                )
                for i in range(down_sampling_layers)
            ]
        )

    def forward(self, season_list):
        # mixing high->low
        out_high = season_list[0]
        out_low = season_list[1]
        out_season_list = [out_high.permute(0, 2, 1)]

        for i in range(len(season_list) - 1):
            out_low_res = self.down_sampling_layers[i](out_high)
            out_low = out_low + out_low_res
            out_high = out_low
            if i + 2 <= len(season_list) - 1:
                out_low = season_list[i + 2]
            out_season_list.append(out_high.permute(0, 2, 1))

        return out_season_list

class MultiScaleTrendMixing(nn.Module):
    """
    Top-down mixing trend pattern
    """
    def __init__(self, seq_len, down_sampling_window, down_sampling_layers):
        super(MultiScaleTrendMixing, self).__init__()
        self.up_sampling_layers = torch.nn.ModuleList(
            [
                nn.Sequential(
                    torch.nn.Linear(
                        seq_len // (down_sampling_window ** (i + 1)),
                        seq_len // (down_sampling_window ** i),
                    ),
                    nn.GELU(),
                    torch.nn.Linear(
                        seq_len // (down_sampling_window ** i),
                        seq_len // (down_sampling_window ** i),
                    ),
                )
                for i in reversed(range(down_sampling_layers))
            ])

    def forward(self, trend_list):
        # mixing low->high
        trend_list_reverse = trend_list.copy()
        trend_list_reverse.reverse()
        out_low = trend_list_reverse[0]
        out_high = trend_list_reverse[1]
        out_trend_list = [out_low.permute(0, 2, 1)]

        for i in range(len(trend_list_reverse) - 1):
            out_high_res = self.up_sampling_layers[i](out_low)
            out_high = out_high + out_high_res
            out_low = out_high
            if i + 2 <= len(trend_list_reverse) - 1:
                out_high = trend_list_reverse[i + 2]
            out_trend_list.append(out_low.permute(0, 2, 1))

        out_trend_list.reverse()
        return out_trend_list

class PDMEncLayer(nn.Module):
    def __init__(self, input_shape, **kwargs):
        super(PDMEncLayer, self).__init__()
        dropout, channel_independence = \
            kwargs['dropout'], kwargs['channel_independence']
        self.channel_independence = channel_independence
        seq_len, d_model, d_ff = kwargs['seq_len'], kwargs['d_model'], kwargs['d_ff']
        decomp_method = kwargs.get('decomp_method', 'moving_avg')
        self.d_model = d_model
        # down_sampling_window = kwargs['down_sampling_window']
        # down_sampling_layers = kwargs['down_sampling_layers']
        length_list = [length for _, length, _ in input_shape]
        
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
        if decomp_method == 'moving_avg':
            moving_avg = kwargs.get('moving_avg', 25)
            self.decompsition = TSDecomp(moving_avg)
        elif decomp_method == 'dft_decomp':
            top_k = kwargs.get('top_k', 10)
            self.decompsition = DFT_series_decomp(top_k)
        else:
            raise ValueError('decompsition is error')
        
        if not channel_independence:
            self.cross_layer = nn.Sequential(
                nn.Linear(in_features=d_model, out_features=d_ff),
                nn.GELU(),
                nn.Linear(in_features=d_ff, out_features=d_model),
            )
        
        # Mixing season
        self.mixing_multi_scale_season = MultiStreamMixing(length_list, reverse=False)
        # Mixing trend
        self.mixing_multi_scale_trend = MultiStreamMixing(length_list, reverse=True)

        self.out_cross_layer = nn.Sequential(
            nn.Linear(in_features=d_model, out_features=d_ff),
            nn.GELU(),
            nn.Linear(in_features=d_ff, out_features=d_model),
        )

    def forward(self, x_list, x_mark=None):
        length_list = [length for _,_,length,_ in [x.shape for x in x_list]]
        batch,channel,_,dimension = x_list[0].shape

        # Decompose to obtain the season and trend
        season_list = []
        trend_list = []
        for i, x in enumerate(x_list):
            B,C,L,D = x.shape
            x = x.reshape(B*C, L, D)    # support both channel_independence and channel_dependence
            x_list[i] = x
            season, trend = self.decompsition(x)
            if not self.channel_independence:
                season = self.cross_layer(season)
                trend = self.cross_layer(trend)
            season_list.append(season)
            trend_list.append(trend)

        # bottom-up season mixing
        out_season_list = self.mixing_multi_scale_season(season_list)
        # top-down trend mixing
        out_trend_list = self.mixing_multi_scale_trend(trend_list)

        out_list = []
        for ori, out_season, out_trend, length in zip(x_list, out_season_list, out_trend_list,
                                                      length_list):
            out = out_season + out_trend
            if self.channel_independence:
                out = ori + self.out_cross_layer(out)
            out_list.append(out[:, :length, :].reshape(batch, channel, length, dimension))
        return out_list

class PDMEnc(BaseModule):
    def __init__(self, input_shape, **kwargs):
        super(PDMEnc, self).__init__(input_shape)
        e_layers = kwargs['e_layers']
        self.enc_layers = nn.ModuleList(
            [PDMEncLayer(input_shape, **kwargs) for _ in range(e_layers)]
        )
        self.set_output_shape_template(input_shape)

    def forward(self, x_list, x_mark=None):
        assert type(x_list) is list, "PDMEncoder is a component of TimeMixer, Input should be a list"
        for enc_layer in self.enc_layers:
            x_list = enc_layer(x_list, x_mark=x_mark)
        return x_list