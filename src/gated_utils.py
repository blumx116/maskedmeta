from itertools import islice, chain
from typing import Callable, List, Tuple, Union, NamedTuple, Iterable, Optional, Dict

import numpy as np
import torch
import torch.nn as nn
from torch.optim.optimizer import Optimizer
from tqdm.autonotebook import tqdm

from .MetaModel import MetaModel
from .utils import sample, sampleLenet

from src.LeNet import LeNet5
from torch.utils.data import DataLoader
from src.GatedLinearNew import GatedLinear
from src.GatedConv2dNew import GatedConv2d



def gaussian_nonzero_mask(weights: nn.Parameter) -> torch.Tensor:
    return torch.sum(torch.exp(-(weights ** 2)))


def abs_nonzero_mask(weights: nn.Parameter) -> torch.Tensor:
    return torch.sum(torch.exp(-torch.abs(weights)))

def mask_regularizer(
        model: nn.Module,
        fn: Callable[[nn.Parameter], torch.Tensor] = gaussian_nonzero_mask) -> torch.Tensor:
    # NOTE: this only regularizes THE CURRENT TASK
    result: torch.Tensor = torch.from_numpy(np.asarray(0.))
    if isinstance(model, (GatedLinear, GatedConv2d)):
        result += fn(model.WMs[model.cur_task_idx])
        result += fn(model.bMs[model.cur_task_idx])
    elif isinstance(model, (nn.Sequential, LeNet5)):
        module: nn.Module
        for module in model.modules():
            result += mask_regularizer(module, fn)
    return result


def get_mask_regularizer(
        name: str,
        weight: float = 1.) \
        -> Optional[Callable[[nn.Module], torch.Tensor]]:
    name = name.lower()
    assert name in ['none', 'gaussian', 'abs']
    if name == 'none':
        return None
    lookup: Dict[str, Callable[[nn.Parameter], torch.Tensor]] = {
        'gaussian': gaussian_nonzero_mask,
        'abs': abs_nonzero_mask}
    return lambda model: weight * mask_regularizer(model, fn=lookup[name])


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
        n_epochs: int = 10000,
        regularizer: Optional[Callable[[nn.Module], torch.Tensor]] = None) -> TrainResult:
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
        if regularizer is not None:
            loss += regularizer(model)

        loss.backward()
        optim.step()

        losses[task_idx].append(loss.item())

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

def trainLenet(
        make_model: Callable[[int], nn.Module], # (n_tasks) => model
        make_optim: Callable[[Iterable[nn.Parameter], Iterable[nn.Parameter]], Optimizer], # (weight params, mask params) => optimizer
        tasks,
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
        x, y = sampleLenet(tasks[task_idx], batch_size)

        yhat: torch.Tensor = model(x)
        loss: torch.Tensor = criterion(yhat, y)

        loss.backward()
        optim.step()

        losses[task_idx].append(loss.item())

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
