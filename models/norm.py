import torch
import torch.nn as nn

class SNorm(nn.Module):
    def __init__(self, feat_size, bias=False):
        super().__init__()
        self.bias = bias
        if bias:
            self.beta = nn.Parameter(torch.zeros(1, 1, 1, feat_size))
            self.gamma = nn.Parameter(torch.ones(1, 1, 1, feat_size))

    def forward(self, x):
        mean = x.mean(-2, keepdims=True)
        var = x.var(-2, keepdims=True, unbiased=True)
        x_norm = (x - mean) / (var + 1e-6) ** 0.5
        if self.bias: x_norm = x_norm * self.gamma + self.beta
        return x_norm

class TNorm(nn.Module):
    def __init__(self, feat_size, bias=False):
        super().__init__()
        self.bias = bias
        if bias:
            self.beta = nn.Parameter(torch.zeros(1, 1, 1, feat_size))
            self.gamma = nn.Parameter(torch.ones(1, 1, 1, feat_size))

    def forward(self, x):
        mean = x.mean(-3, keepdims=True)
        var = x.var(-3, keepdims=True, unbiased=True)
        x_norm = (x - mean) / (var + 1e-6) ** 0.5
        if self.bias: x_norm = x_norm * self.gamma + self.beta
        return x_norm
