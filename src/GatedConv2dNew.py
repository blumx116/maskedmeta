import math
from typing import Optional, List, Tuple

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from torch.nn.common_types import _size_1_t, _size_2_t, _size_3_t
#from torch.nn.modules.conv import _ConvNd
from torch import Tensor

from .utils import _single, _pair, _triple, _reverse_repeat_tuple
from .MetaModel import MetaModel


class GatedConvNd(nn.Module):
    __constants__ = ['stride', 'padding', 'dilation', 'groups',
                     'padding_mode', 'output_padding', 'in_channels',
                     'out_channels', 'kernel_size']
    __annotations__ = {'bW': Optional[torch.Tensor]}

    _in_channels: int
    out_channels: int
    kernel_size: Tuple[int, ...]
    stride: Tuple[int, ...]
    padding: Tuple[int, ...]
    dilation: Tuple[int, ...]
    transposed: bool
    output_padding: Tuple[int, ...]
    groups: int
    padding_mode: str
    WW: Tensor
    bW: Optional[Tensor]

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: _size_1_t,
                 stride: _size_1_t,
                 padding: _size_1_t,
                 dilation: _size_1_t,
                 transposed: bool,
                 output_padding: _size_1_t,
                 groups: int,
                 bW: Optional[Tensor],
                 padding_mode: str) -> None:
        super(GatedConvNd, self).__init__()
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        valid_padding_modes = {'zeros', 'reflect', 'replicate', 'circular'}
        if padding_mode not in valid_padding_modes:
            raise ValueError("padding_mode must be one of {}, but got padding_mode='{}'".format(
                valid_padding_modes, padding_mode))
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.transposed = transposed
        self.output_padding = output_padding
        self.groups = groups
        self.padding_mode = padding_mode
        # `_reversed_padding_repeated_twice` is the padding to be passed to
        # `F.pad` if needed (e.g., for non-zero padding types that are
        # implemented as two ops: padding + conv). `F.pad` accepts paddings in
        # reverse order than the dimension.
        self._reversed_padding_repeated_twice = _reverse_repeat_tuple(self.padding, 2)
        if transposed:
            self.WW = Parameter(torch.Tensor(
                in_channels, out_channels // groups, *kernel_size))
        else:
            self.WW = Parameter(torch.Tensor(
                out_channels, in_channels // groups, *kernel_size))
        if bW:
            self.bW = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bW', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.WW, a=math.sqrt(5))
        self.WW = nn.Parameter(self.WW * 2.)
        if self.bW is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.WW)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bW, -bound, bound)
            self.bW = nn.Parameter(self.bW * 2.)

    def extra_repr(self):
        s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.output_padding != (0,) * len(self.output_padding):
            s += ', output_padding={output_padding}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bW is None:
            s += ', bW=False'
        if self.padding_mode != 'zeros':
            s += ', padding_mode={padding_mode}'
        return s.format(**self.__dict__)

    def __setstate__(self, state):
        super(GatedConvNd, self).__setstate__(state)
        if not hasattr(self, 'padding_mode'):
            self.padding_mode = 'zeros'


class GatedConv2d(GatedConvNd, MetaModel):
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
            padding_mode: str = 'zeros',  # TODO: refine this type
            n_tasks: int = 1
    ):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)

        assert in_channels > 0
        assert out_channels > 0
        assert n_tasks > 0

        super(GatedConv2d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _pair(0), groups, bias, padding_mode)
        # note: GatedConvNd should initialize self.WW, self.bW

        self.in_channels: int = in_channels
        self.out_channels: int = out_channels
        self.bias = bias
        self.n_tasks: int = 0
        self.cur_task_idx: Optional[int] = None

        self.WMs: nn.ParameterList = nn.ParameterList()
        self.bMs: Optional[nn.ParameterList] = nn.ParameterList() if self.bias else None
        self.add_tasks(n_tasks)

    def add_tasks(self,
                  n_tasks: int) -> None:
        assert n_tasks > 0

        for _ in range(n_tasks):
            self.WMs.append(self._make_WM_(self.in_channels, self.out_channels, self.kernel_size, self.groups))
            if self.bias:
                self.bMs.append(self._make_bM_(self.in_channels, self.out_channels))

        self.n_tasks += n_tasks

    def shared_parameters(self) -> nn.ParameterList:
        result: nn.ParameterList = nn.ParameterList()
        result.append(self.WW)
        if self.bias:
            result.append(self.bW)
        return result

    def task_parameters_for(self,
                            task_idx: int) -> nn.ParameterList:
        assert task_idx < self.n_tasks
        result: nn.ParameterList = nn.ParameterList()
        result.append(self.WMs[task_idx])
        if self.bias:
            result.append(self.bMs[task_idx])
        return result

    def forward(self,
                inputs: torch.Tensor) -> torch.Tensor:
        return F.conv2d(inputs, self.get_current_W(), self.get_current_b(), self.stride, self.padding, self.dilation,
                        self.groups)

    def get_current_W(self) -> torch.Tensor:
        assert self.cur_task_idx is not None, \
            "Must call set_task() before forward() or get_current_W()"

        return self._gated_(self.WW, self.WMs[self.cur_task_idx])

    def get_current_b(self) -> Optional[torch.Tensor]:
        assert self.cur_task_idx is not None, \
            "Must call set_task() before forward() or get_current_b()"
        if self.bias:
            return self._gated_(self.bW, self.bMs[self.cur_task_idx])
        # implicit else
        return None

    @staticmethod
    def _gated_(
            weights: torch.Tensor,
            masks: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(masks) * weights

    def set_task(self,
                 task_idx: int) -> None:
        assert 0 <= task_idx < self.n_tasks
        self.cur_task_idx = task_idx

    def get_task(self) -> int:
        return self.cur_task_idx

    '''def _make_WW_(self,
                  in_channels: int,
                  out_channels: int,
                  kernel_size: _size_2_t,
                  groups: int) -> nn.Parameter:
        W: torch.Tensor = torch.Tensor(out_channels, in_channels // groups, *kernel_size)
        nn.init.kaiming_uniform_(W, a=math.sqrt(5))
        return nn.Parameter(W * 2.)

    def _make_bW_(self,
                  in_channels: int,
                  out_channels: int) -> torch.Tensor:
        b: torch.Tensor = torch.Tensor(out_channels)
        fan_in: float = float(in_channels)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(b, -bound, bound)
        return nn.Parameter(b * 2.)'''

    def _make_WM_(self,
                  in_channels: int,
                  out_channels: int,
                  kernel_size: _size_2_t,
                  groups: int) -> torch.Tensor:
        return nn.Parameter(torch.zeros(out_channels, in_channels // groups, *kernel_size))

    def _make_bM_(self,
                  in_channels: int,
                  out_channels: int) -> torch.Tensor:
        return nn.Parameter(torch.zeros(out_channels))
