import torch.nn as nn


class LinearInterpolantDT(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x0, x1, time_steps):
        dt_xt = x1 - x0
        return dt_xt[:,None,:].repeat(1,time_steps, 1)