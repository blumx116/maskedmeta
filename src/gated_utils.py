from itertools import islice, chain
from typing import Callable, List, Tuple, Union, NamedTuple, Iterable, Optional, Dict

import numpy as np
import torch
import torch.nn as nn
from torch.optim.optimizer import Optimizer
from tqdm.autonotebook import tqdm

from .MetaModel import MetaModel
from .utils import sample

from src.LeNet import LeNet5

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
    elif isinstance(module, LeNet5):
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
    elif (isinstance(module, LeNet5) and module.isGated()):
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
    elif (isinstance(module, LeNet5) and module.isGated()):
        result.extend(module.task_parameters_for(task_idx))
    # nothing to do for other module types, give empty list
    return result


class TrainResult(NamedTuple):
    model: Union[MetaModel, nn.Module]
    optim: Optimizer
    losses: List[List[float]]
    grads: List[torch.Tensor]
    weights: List[torch.Tensor]
    accuracy: List[List[float]]


def train(
        make_model: Callable[[int], nn.Module], # (n_tasks) => model
        make_optim: Callable[[Iterable[nn.Parameter], Iterable[nn.Parameter]], Optimizer], # (weight params, mask params) => optimizer
        tasks: List[Tuple[torch.Tensor, torch.Tensor]],
        criterion: nn.Module,
        batch_size: int=32,
        test_hooks: Optional[List[Tuple[int, Callable[[nn.Module, int], None]]]] = None,
        n_epochs: int = 10000) -> TrainResult:
    assert len(tasks) > 0

    n_tasks: int = len(tasks)
    model: nn.Module = make_model(n_tasks)
    task_params_l: List[nn.ParameterList] = [get_task_parameters(model, idx) for idx in range(n_tasks)]
    task_params: nn.ParameterList = nn.ParameterList(chain(*task_params_l))
    optim: Optimizer = make_optim(get_shared_parameters(model), task_params)
    grads = []
    weights = []
    losses: List[List[float]] = [[] for _ in range(n_tasks)]
    accuracy: List[List[float]] = [[] for _ in range(n_tasks)]

    for t in tqdm(range(n_epochs)):
        optim.zero_grad()
        task_idx: int = np.random.randint(0, n_tasks)
        # task_idx: int = 1

        set_task(model, task_idx)

        x: torch.Tensor
        y: torch.Tensor
        x, y = sample(*tasks[task_idx], batch_size=batch_size)

        yhat: torch.Tensor = model(x)
        loss: torch.Tensor = criterion(yhat, y)

        loss.backward()
        optim.step()

        losses[task_idx].append(loss.item())

        if (type(model) == LeNet5):
            with torch.no_grad():
                model.eval()
                x, y_true = tasks[task_idx]
                y_prob = model(x)
                _, predicted_labels = torch.max(y_prob, 1)

                n = y_true.size(0)
                correct_pred = (predicted_labels == y_true).sum()

            accuracy[task_idx].append(correct_pred.float() / n)

        if test_hooks is not None:
            frequency: int
            fn: Callable[[nn.Module, int], None]
            for frequency, fn in test_hooks:
                if t % frequency == 0:
                    fn(model, t)


    return TrainResult(
        model=model,
        optim=optim,
        losses=losses,
        grads=grads,
        weights=weights,
        accuracy=accuracy)
