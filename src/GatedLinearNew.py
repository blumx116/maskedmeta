import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .MetaModel import MetaModel


class GatedLinear(nn.Module, MetaModel):
    def __init__(self,
            in_features: int,
            out_features: int,
            bias: bool = True,
            n_tasks: int = 1):
        assert in_features > 0
        assert out_features > 0
        assert n_tasks > 0

        super().__init__()

        self.in_features: int = in_features
        self.out_features: int = out_features
        self.bias = bias
        self.n_tasks: int = 0
        self.cur_task_idx: Optional[int] = None

        self.WW: nn.Parameter = self._make_WW_(in_features, out_features)
        self.bW: Optional[nn.Parameter] = self._make_bW_(in_features, out_features) if self.bias else None

        self.WMs: nn.ParameterList = nn.ParameterList()
        self.bMs: Optional[nn.ParameterList] = nn.ParameterList() if self.bias else None
        self.add_tasks(n_tasks)

    def add_tasks(self,
            n_tasks: int) -> None:
        assert n_tasks > 0

        for _ in range(n_tasks):
            self.WMs.append(self._make_WM_(self.in_features, self.out_features))
            if self.bias:
                self.bMs.append(self._make_bM_(self.in_features, self.out_features))

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
        return F.linear(inputs, self.get_current_W(), self.get_current_b())

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

    def _make_WW_(self,
            in_features: int,
            out_features: int) -> nn.Parameter:
        W: torch.Tensor = torch.Tensor(out_features, in_features)
        nn.init.kaiming_uniform_(W, a=math.sqrt(5))
        return nn.Parameter(W * 2.)

    def _make_bW_(self,
                  in_features: int,
                  out_features: int) -> torch.Tensor:
        b: torch.Tensor = torch.Tensor(out_features)
        fan_in: float = float(in_features)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(b, -bound, bound)
        return nn.Parameter(b * 2.)

    def _make_WM_(self,
            in_features: int,
            out_features: int) -> torch.Tensor:
        return nn.Parameter(torch.zeros(out_features, in_features))

    def _make_bM_(self,
                  in_features: int,
                  out_features: int) -> torch.Tensor:
        return nn.Parameter(torch.zeros(out_features))
