from abc import ABC
from typing import Optional

import torch.nn as nn


class MetaModel(ABC):
    def add_tasks(self,
            n_tasks: int) -> None:
        pass

    def set_task(self,
            task_idx: int) -> None:
        pass

    def get_task(self) -> Optional[int]:
        pass

    def shared_parameters(self) -> nn.ParameterList:
        pass

    def task_parameters_for(self,
            task_idx: int) -> nn.ParameterList:
        pass