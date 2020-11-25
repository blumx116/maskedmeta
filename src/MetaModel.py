from abc import ABC, abstractmethod
from typing import Optional

import torch.nn as nn


class MetaModel(ABC):
    @abstractmethod
    def add_tasks(self,
            n_tasks: int) -> None:
        pass

    @abstractmethod
    def set_task(self,
            task_idx: int) -> None:
        pass

    @abstractmethod
    def get_task(self) -> Optional[int]:
        pass

    @abstractmethod
    def shared_parameters(self) -> nn.ParameterList:
        pass

    @abstractmethod
    def task_parameters_for(self,
            task_idx: int) -> nn.ParameterList:
        pass
