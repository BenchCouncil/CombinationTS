import torch
from torch import nn
import torch.nn.functional as F
import math
from arch import BaseModule

class PositionalEmbedding(nn.Module):
    pe: torch.Tensor
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.requires_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]

class PatchEmbedding(BaseModule):
    def __init__(self, input_shape, patch_len, stride, padding, d_model, dropout, **kwargs):
        super(PatchEmbedding, self).__init__(input_shape)
        # Patching
        n_vars, seq_len, _ = input_shape
        self.patch_len, self.stride = patch_len, stride

        self.padding_patch_layer = nn.ReplicationPad1d((0, padding))

        # Backbone, Input encoding: projection of feature vectors onto a d-dim vector space
        self.value_embedding = nn.Linear(self.patch_len, d_model, bias=False)

        # Positional embedding
        self.position_embedding = PositionalEmbedding(d_model)

        # Residual dropout
        self.dropout = nn.Dropout(dropout)
        assert (seq_len - self.patch_len + padding) % self.stride == 0, \
            "Error: (seq_len - patch_len + padding) must be divisible by stride"
        patch_num = int((seq_len - self.patch_len + padding) / self.stride) + 1
        self.set_output_shape_template((n_vars, patch_num, d_model))
    
    def get_output_shape(self, input_shape: tuple | list[tuple] | None) -> tuple | list[tuple]:
        input_shapes: list[tuple]
        is_list = isinstance(input_shape, list)
        input_shapes = input_shape if is_list else [input_shape]
        output_shapes = []
        for shape in input_shapes:
            n_vars, l, _ = shape
            assert (l - self.patch_len + (self.padding_patch_layer.padding[1])) % self.stride == 0, \
                "Error: (seq_len - patch_len + padding) must be divisible by stride"
            patch_num = int((l - self.patch_len + (self.padding_patch_layer.padding[1])) / self.stride) + 1
            output_shapes.append((n_vars, patch_num, self.value_embedding.out_features))
        return output_shapes if is_list else output_shapes[0]
        

    def forward(self, x, x_mark=None):
        x = x.squeeze(-1)
        # do patching
        x = self.padding_patch_layer(x)
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        B, C, N, L = x.shape
        x = torch.reshape(x, (B * C, N, L))
        # embedding
        x = self.value_embedding(x) + self.position_embedding(x)
        x = self.dropout(x)
        x = torch.reshape(x, (B, C, N, -1))
        return x
