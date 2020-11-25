from typing import Optional

import torch
import torch.nn as nn

from .MetaModel import MetaModel


class SharedModel(nn.Module, MetaModel):
    def __init__(self,
            inner: nn.Module,
            n_tasks: int = 1):
        assert n_tasks > 0

        super().__init__()

        self.inner: nn.Module = inner
        self.n_tasks: int = n_tasks
        self.cur_task_idx: Optional[int] = None

    def forward(self,
            inputs: torch.Tensor) -> torch.Tensor:
        return self.inner.forward(inputs)

    def add_tasks(self,
            n_tasks: int) -> None:
        assert n_tasks > 0

        self.n_tasks += n_tasks

    def set_task(self,
            task_idx: int) -> None:
        assert 0 <= task_idx < self.n_tasks

        self.cur_task_idx = task_idx

    def get_task(self) -> Optional[int]:
        return self.cur_task_idx

    def shared_parameters(self) -> nn.ParameterList:
        return nn.ParameterList(list(self.inner.parameters()))

    def task_parameters_for(self,
            task_idx: int) -> nn.ParameterList:
        assert 0 <= task_idx < self.n_tasks

        return nn.ParameterList()
