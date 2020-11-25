from typing import Optional, Callable

import torch
import torch.nn as nn

from .MetaModel import MetaModel


class SeparateModels(nn.Module, MetaModel):
    def __init__(self,
            make_model: Callable[[], nn.Module],
            n_tasks: int = 1):
        self._make_model_: Callable[[], nn.Module] = make_model
        self.models: nn.ModuleList = nn.ModuleList()
        self.n_tasks: int = 0
        self.cur_task_idx: Optional[int] = None

        self.add_tasks(n_tasks)

    def set_task(self,
            task_idx: int) -> None:
        assert 0 <= task_idx < self.n_tasks

        self.cur_task_idx = None

    def get_task(self) -> Optional[int]:
        return self.cur_task_idx

    def shared_parameters(self) -> nn.ParameterList:
        return nn.ParameterList()

    def task_parameters_for(self, task_idx: int) -> nn.ParameterList:
        assert 0 <= task_idx < self.n_tasks

        return nn.ParameterList(list(self.models[task_idx].parameters()))

    def add_tasks(self, n_tasks: int) -> None:
        assert n_tasks > 0

        for _ in range(n_tasks):
            self.models.append(self._make_model_())

        self.n_tasks += n_tasks
