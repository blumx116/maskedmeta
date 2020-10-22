import math
from typing import Tuple

import torch 
import torch.nn as nn
import torch.nn.functional as F

#TODO: import GatedLinear file to access class GatedLinear?

# Define model
class GatedModel(nn.Module):
    def __init__(self):
        super(TheModelClass, self).__init__()
        self.conv1 = Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = Conv2d(6, 16, 5)
        #self.fc1 = nn.Linear(16 * 5 * 5, 120)
        #self.fc2 = nn.Linear(120, 84)
        # self.fc3 = nn.Linear(84, 10)
        self.fc1 = GatedLinear(16 * 5 * 5, 120)
        self.fc2 = GatedLinear(120, 84)
        self.fc3 = GatedLinear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class Conv2d(nn._ConvNd):
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
        super(Conv2d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _pair(0), groups, bias, padding_mode)    #note: _ConvNd should initialize self.weight and self.bias

        #TODO: Is it okay that self.bias is either False (boolean) or a Tensor? This probably holds under if(self.bias)
        self.weightMask, self.biasMask = Conv2d.constant_initialize((out_channels, in_channels), self.bias, constant=0)
    
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
        maskedBias = Conv2d.gated(self.bias, self.biasMask) if self.bias else None

        if self.padding_mode != 'zeros':
            return F.conv2d(F.pad(input, self._reversed_padding_repeated_twice, mode=self.padding_mode),
                            weight, maskedBias, self.stride,
                            _pair(0), self.dilation, self.groups)
        return F.conv2d(input, weight, maskedBias, self.stride,
                        self.padding, self.dilation, self.groups)

    #taken from nn.Conv2d
    def forward(self, input: Tensor) -> Tensor:
        #add masking to weight
        maskedWeight = Conv2d.gated(self.weight, self.weightMask)
        return self._conv_forward(input, maskedWeight)

    #TODO: is this func necessary, and if so are we doing the right checks?
    def copy_weights(self, other: "Conv2d") -> None:
        assert self.in_channels == other.in_channels
        assert self.out_channels == other.out_channels
        assert self.bias == other.bias
        self.weight = other.weight
        self.bias = other.bias