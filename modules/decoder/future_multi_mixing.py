import torch
from torch import nn
from arch import BaseModule

class FutureMultiMixing(BaseModule):
    def __init__(self, input_shape, **kwargs):
        super(FutureMultiMixing, self).__init__(input_shape)
        assert len(input_shape) > 1, "FutureMultiMixing is a component of TimeMixer, Input shape should be more than 1"
        self.channel_independence = kwargs['channel_independence']
        d_model = input_shape[0][2]     # (C, L1, D), (C, L2, D), ...
        self.pred_len, self.n_vars = kwargs['pred_len'], kwargs['n_vars']
        self.predict_layers = nn.ModuleList([nn.Linear(length, self.pred_len,) for _, length, _ in input_shape])

        if self.channel_independence:
            self.projection_layer = nn.Linear(d_model, 1, bias=True)
        else:
            self.projection_layer = nn.Linear(d_model, self.n_vars, bias=True)
            self.out_res_layers = nn.ModuleList([nn.Linear(length, length) for _, length, _ in input_shape])
            self.regression_layers= nn.ModuleList([nn.Linear(length, self.pred_len) for _, length, _ in input_shape])
        
        self.set_output_shape_template((self.pred_len, self.n_vars))
    
    def out_projection(self, dec_out, i, out_res):
        dec_out = self.projection_layer(dec_out)
        out_res = out_res.squeeze(-1)  # B, N, L
        out_res = self.out_res_layers[i](out_res)
        out_res = self.regression_layers[i](out_res).permute(0, 2, 1)
        dec_out = dec_out + out_res
        return dec_out
    
    def forward(self, enc_out_list, trend_list=None):
        # B, C, L, D
        if isinstance(trend_list, list):
            trend_list = None if len(trend_list) == 0 else trend_list
        B,C,_,D = enc_out_list[0].shape
        for i, enc_out in enumerate(enc_out_list):
            # B, C, L, D -> B*C, L, D, support both channel_independence and channel_dependence
            enc_out_list[i] = enc_out.reshape(B*C, -1, D)
        dec_out_list = []
        if self.channel_independence:
            for i, enc_out in enumerate(enc_out_list):
                dec_out = self.predict_layers[i](enc_out.permute(0, 2, 1)).permute(0, 2, 1)  # align temporal dimension
                dec_out = self.projection_layer(dec_out)
                dec_out = dec_out.reshape(B, self.n_vars, self.pred_len, 1).contiguous()
                dec_out_list.append(dec_out)
        else:
            assert trend_list is not None, "trend_list should not be None in channel_dependence mode"
            for i, enc_out, out_res in zip(range(len(enc_out_list)), enc_out_list, trend_list):
                dec_out = self.predict_layers[i](enc_out.permute(0, 2, 1)).permute(0, 2, 1)  # align temporal dimension
                dec_out = self.out_projection(dec_out, i, out_res)
                dec_out_list.append(dec_out.permute(0, 2, 1).unsqueeze(-1).contiguous())  # B, N, L, 1

        dec_out = torch.stack(dec_out_list, dim=-1).sum(-1)
        return dec_out