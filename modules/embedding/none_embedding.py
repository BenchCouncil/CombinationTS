import torch
from torch import nn
import torch.nn.functional as F
import math
from arch import BaseModule

class NoneEmbedding(BaseModule):
    def __init__(self, input_shape, embed_dim=None, **kwargs):
        super(NoneEmbedding, self).__init__(input_shape)
        C, L, _ = input_shape  # C: num_vars, L: seq_len
        if embed_dim is None:
            self.set_output_shape_template(-1)
            self.forward = lambda x, x_mark=None: x
        elif embed_dim == 'C':
            self.forward = lambda x, x_mark=None: x.permute(0, 3, 2, 1)
            self.set_output_shape_template((1, L, C))
        elif embed_dim == 'L':
            self.forward = lambda x, x_mark=None: x.permute(0, 1, 3, 2)
            self.set_output_shape_template((C, 1, L))
        else:
            raise ValueError("embed_dim must be None, 'C', or 'L'")

    # def forward(self, x, x_mark=None):
    #     return x