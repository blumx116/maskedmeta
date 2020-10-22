import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class GatedLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        
        self.bias = bias
        self.in_features = in_features
        self.out_features = out_features
        
        # Use Kaiming initialization for weights
        # https://arxiv.org/abs/1502.01852
        # multiply by 2 to account for masks
        self.WW, self.bW = GatedLinear.kaiming_initialize(
            (out_features, in_features), bias)
        # self.WW : torch.Tensor[float32], (out_features, in_features)
        # self.bW : torch.Tensor[float32], (out_features)
        with torch.no_grad():
            self.WW *= 2
            if self.bias:
                self.bW *= 2
        
        self.WM, self.bM = GatedLinear.constant_initialize(
            (out_features, in_features), bias, constant=0)
        # self.WW : torch.Tensor[float32], (in_features, out_features)
        # self.bW : torch.Tensor[float32], (out_features)
        
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        W = GatedLinear.gated(self.WW, self.WM)
        b = GatedLinear.gated(self.bW, self.bM) if self.bias else None
        return F.linear(input, W, b)
    
    @staticmethod
    def gated(W: torch.Tensor, M: torch.Tensor) -> torch.Tensor:
        # W and M should be of the same shape
        return torch.sigmoid(M) * W
    
    @staticmethod
    def kaiming_initialize(
            shape: Tuple[int, int],
            bias: bool) -> Tuple[torch.Tensor, torch.Tensor]:
        # basically copied from here:
        # https://pytorch.org/docs/stable/_modules/torch/nn/modules/linear.html#Linear
        W = nn.Parameter(torch.Tensor(*shape))
        nn.init.kaiming_uniform_(W, a=math.sqrt(5))
        if bias:
            b = nn.Parameter(torch.Tensor(shape[0]))
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(W)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(b, -bound, bound)
        else:
            b = None
        return W, b

    @staticmethod
    def constant_initialize(
            shape: Tuple[int, int],
            bias: bool,
            constant: float) -> Tuple[torch.Tensor, torch.Tensor]:
        W = nn.Parameter(torch.Tensor(*shape))
        nn.init.constant_(W, constant)
        if bias:
            b = nn.Parameter(torch.Tensor(shape[0]))
            nn.init.constant_(b, constant)
        else:
            b = None
        return W, b

    def copy_weights(self, other: "GatedLinear") -> None:
        assert self.in_features == other.in_features
        assert self.out_features == other.out_features
        assert self.bias == other.bias
        self.WW = other.WW
        self.bW = other.bW
