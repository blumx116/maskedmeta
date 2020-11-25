from itertools import islice
from typing import Callable, List, Tuple, Union, NamedTuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim.optimizer import Optimizer
from tqdm.autonotebook import tqdm

from .MetaModel import MetaModel
from .utils import sample

"""
def copy_weights(From: nn.Module, To: nn.Module) -> None:
    assert type(From) == type(To)
    if isinstance(From, nn.Sequential):
        for module1, module2 in zip(From.modules(), To.modules()):
            copy_weights(From=module1, To=module2)
    elif isinstance(From, GatedLinear):
        assert From.in_features == To.in_features
        assert From.out_features == To.out_features
        assert From.bias == To.bias
        To.WW = From.WW
        To.bW = From.bW
    elif isinstance(From, GatedConv2d):
        raise Exception("Ashley you haven't added GatedConv2d to copy_weights")
    else:
        To.load_state_dict(From.state_dict())
"""

def set_task(
        module: nn.Module,
        task_idx: int) -> None:
    if isinstance(module, nn.Sequential):
        for submodule in module.modules():
            set_task(submodule, task_idx)
    elif isinstance(module, MetaModel):
        module.set_task(task_idx)
    else:
        pass


def get_shared_parameters(
        module: nn.Module) -> nn.ParameterList:
    result: nn.ParameterList = nn.ParameterList()
    if isinstance(module, nn.Sequential):
        for submodule in islice(module.modules(), 1, None):
            # skip the first element of module.modules(), because it's recursively the nn.Sequential
            result.extend(get_shared_parameters(submodule))
    elif isinstance(module, MetaModel):
        return module.shared_parameters()
    else:
        result.extend(module.parameters())
    return result


def get_task_parameters(
        module: nn.Module,
        task_idx: int) -> nn.ParameterList:
    assert task_idx >= 0

    result: nn.ParameterList = nn.ParameterList()
    if isinstance(module, nn.Sequential):
        for submodule in islice(module.modules(), 1, None):
            result.extend(get_task_parameters(submodule, task_idx))
    elif isinstance(module, MetaModel):
        result.extend(module.task_parameters_for(task_idx))
    # nothing to do for other module types, give empty list
    return result


class TrainResult(NamedTuple):
    model: Union[MetaModel, nn.Module]
    shared_optim: Optimizer
    task_optims: List[Optimizer]
    losses: List[List[float]]


def train(
        make_model: Callable[[int], Union[MetaModel, nn.Module]],
        make_shared_optim: Callable[[nn.ParameterList], Optimizer],
        make_task_optim: Callable[[nn.ParameterList, int], Optimizer],
        tasks: List[Tuple[torch.Tensor, torch.Tensor]],
        criterion: nn.Module,
        batch_size: int=32,
        test_hook=None,
        n_epochs: int = 10000) -> TrainResult:
    assert len(tasks) > 0

    n_tasks: int = len(tasks)
    model: Union[MetaModel, nn.Module] = make_model(n_tasks)
    shared_optim: Optimizer = make_shared_optim(get_shared_parameters(model))
    task_optims: List[Optimizer] = [make_task_optim(get_task_parameters(model, idx), idx) for idx in range(n_tasks)]

    losses: List[List[float]] = [[] for _ in range(n_tasks)]

    for _ in tqdm(range(n_epochs)):
        task_idx: int = np.random.randint(0, n_tasks)

        cur_task_optim: Optimizer = task_optims[task_idx]
        model.set_task(task_idx)

        x: torch.Tensor
        y: torch.Tensor
        x, y = sample(*tasks[task_idx], batch_size=batch_size)

        yhat: torch.Tensor = model(x)
        loss: torch.Tensor = criterion(yhat, y)

        shared_optim.zero_grad()
        cur_task_optim.zero_grad()
        loss.backward()
        shared_optim.step()
        cur_task_optim.step()

        losses[task_idx].append(loss.detach().cpu().item())

    return TrainResult(
        model=model,
        shared_optim=shared_optim,
        task_optims=task_optims,
        losses=losses)
