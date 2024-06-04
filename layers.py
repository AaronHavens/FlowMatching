import torch
from torch import nn
import torch.nn.functional as F
import numpy as np


class SDPLin(nn.Module):

    def __init__(self, cin, cout, gamma=1.0, epsilon=1e-6, bias=True):
        super(SDPLin, self).__init__()

        self.cout = cout
        self.weight = nn.Parameter(torch.empty(1, self.cout, cin))
        nn.init.xavier_normal_(self.weight)
        if bias:
            self.bias = nn.Parameter(torch.empty(cout))
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / np.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
        else: self.bias = None
        self.q = nn.Parameter(torch.rand(1, cin))

        self.cout = cout
        self.cin = cin
        self.W = None
        self.gamma = nn.Parameter(torch.tensor([gamma]), requires_grad=False)

    def forward(self, x):
        if self.training or self.W is None:
    
            q = torch.exp(self.q)[:,None,:]
            q_inv = torch.exp(-self.q)[:, :, None]
            T = 1/torch.abs(q_inv*torch.transpose(self.weight,1,2)@self.weight*q/self.gamma**2).sum(2)
            W_ = (self.weight@torch.diag_embed(torch.sqrt(T))).view(self.cout, self.cin)
            if self.training:
                self.W = W_
            else:
                self.W = nn.Parameter(W_.detach()) 
        W = self.W #if self.training else nn.Parameter(self.W.detach(), requires_grad=True)

        out =  nn.functional.linear(x, W, self.bias)
        return out


class SLLRes(nn.Module):

  def __init__(self, cin, cout, epsilon=1e-6):
    super(SLLRes, self).__init__()

    self.activation = nn.ReLU(inplace=False)
    self.weights = nn.Parameter(torch.empty(cout, cin))
    self.bias = nn.Parameter(torch.empty(cout))
    self.q = nn.Parameter(torch.rand(cout))

    nn.init.xavier_normal_(self.weights)
    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weights)
    bound = 1 / np.sqrt(fan_in)
    nn.init.uniform_(self.bias, -bound, bound)  # bias init

    self.epsilon = epsilon

    #self.W = None

  def forward(self, x):
    #if self.training or self.W is None:
    res = nn.functional.linear(x, self.weights, self.bias)
    res = self.activation(res)
    q = torch.exp(self.q)[None, :]
    q_inv = torch.exp(-self.q)[:, None]
    T = 2/torch.abs(q_inv * self.weights @ self.weights.t() * q).sum(1)
    res = T * res
    res = nn.functional.linear(res, self.weights.t())
    out = x - res
    return out
