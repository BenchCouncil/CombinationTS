import torch
from torch import nn
from arch import BaseModule

class NoneEnc(BaseModule):
    def __init__(self, input_shape, **kwargs):
        super(NoneEnc, self).__init__(input_shape)
        e_layers = kwargs.get('e_layers', 0)
        self.set_output_shape_template(-1)
        self.layers = nn.ModuleList([nn.Identity() for _ in range(e_layers)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
