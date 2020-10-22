import math
from typing import Tuple

import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.common_types import _size_2_t
from torch.nn.utils import _pair
from torch.nn.modules.conv import _ConvNd
from torch import Tensor

from utils import _pair

class GatedConv(_ConvNd):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t,
        stride: _size_2_t = 1,
        padding: _size_2_t = 0,
        dilation: _size_2_t = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = 'zeros'  # TODO: refine this type
    ):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(GatedConv, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _pair(0), groups, bias, padding_mode)    #note: _ConvNd should initialize self.weight and self.bias

        self.weightMask, self.biasMask = GatedConv.constant_initialize((out_channels, in_channels), self.bias, constant=0)
    
    @staticmethod
    def gated(W: torch.Tensor, M: torch.Tensor) -> torch.Tensor:
        # W and M should be of the same shape
        return torch.sigmoid(M) * W

    #TODO: so we're always starting our masks out at all 0's?
    @staticmethod
    def constant_initialize(shape: Tuple[int, int], bias: bool, constant: float) -> Tuple[torch.Tensor, torch.Tensor]:
        W = nn.Parameter(torch.Tensor(*shape))
        nn.init.constant_(W, constant)
        if bias:
            b = nn.Parameter(torch.Tensor(shape[0]))
            nn.init.constant_(b, constant)
        else:
            b = None
        return W, b

    #taken from nn.Conv2d
    def _conv_forward(self, input, weight):
        #add masking to bias
        maskedBias = GatedConv.gated(self.bias, self.biasMask) if self.bias else None

        if self.padding_mode != 'zeros':
            return F.conv2d(F.pad(input, self._reversed_padding_repeated_twice, mode=self.padding_mode),
                            weight, maskedBias, self.stride,
                            _pair(0), self.dilation, self.groups)
        return F.conv2d(input, weight, maskedBias, self.stride,
                        self.padding, self.dilation, self.groups)

    #taken from nn.Conv2d
    def forward(self, input: Tensor) -> Tensor:
        #add masking to weight
        maskedWeight = GatedConv.gated(self.weight, self.weightMask)
        return self._conv_forward(input, maskedWeight)

    #TODO: is this func necessary, and if so are we doing the right checks?
    def copy_weights(self, other: "GatedConv") -> None:
        assert self.in_channels == other.in_channels
        assert self.out_channels == other.out_channels
        assert self.bias == other.bias
        self.weight = other.weight
        self.bias = other.bias