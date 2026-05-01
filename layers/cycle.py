import torch
import torch.nn as nn
# from Base import BaseModule

class RecurrentCycle(torch.nn.Module):
    # Thanks for the contribution of wayhoww.
    # The new implementation uses index arithmetic with modulo to directly gather cyclic data in a single operation,
    # while the original implementation manually rolls and repeats the data through looping.
    # It achieves a significant speed improvement (2x ~ 3x acceleration).
    # See https://github.com/ACAT-SCUT/CycleNet/pull/4 for more details.
    def __init__(self, cycle_len, n_vars, seq_len, pred_len, **kwargs):
        super(RecurrentCycle, self).__init__()
        self.cycle_len, self.channel_size, self.seq_len, self.pred_len = \
            cycle_len, n_vars, seq_len, pred_len
        self.data = torch.nn.Parameter(torch.zeros(cycle_len, n_vars), requires_grad=True)
        # self.set_output_shape_template(-1)

    def forward(self, x, cycle_index, type='cycle'):
        return_list = False
        if isinstance(x, list):
            return_list = True
            assert len(x) == 1, "Only single input is supported for cycle adjustment."
            x = x[0]

        if type == 'cycle':
            cycle = self.cycle(cycle_index, self.seq_len).permute(0, 2, 1).unsqueeze(-1)  # (B, C, L, 1)
            return x - cycle if not return_list else [x - cycle]

        elif type == 'forecast':
            cycle_idx = (cycle_index + self.seq_len) % self.cycle_len
            cycle = self.cycle(cycle_idx, self.pred_len).permute(0, 2, 1).unsqueeze(-1)  # (B, C, L, 1)
            return x + cycle if not return_list else [x + cycle]

        raise ValueError("type must be 'cycle' or 'forecast'")
    
    def cycle(self, cycle_index, length):
        gather_index = (cycle_index.view(-1, 1) + torch.arange(length, device=cycle_index.device).view(1, -1)) % self.cycle_len
        cycle = self.data[gather_index] # (B, L, C)
        return cycle