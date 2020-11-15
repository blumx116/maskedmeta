from itertools import repeat

import numpy as np
import torch
import torch.nn as nn
from torch._six import container_abcs

from tqdm.notebook import tqdm

from src.GatedLinear import GatedLinear
from src.GatedConv2d import GatedConv2d


def seed(rand_seed):
    np.random.seed(rand_seed)
    torch.manual_seed(rand_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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


# check that initial parameters are all different
def forall(fn, params1, params2):
    return np.all(
        list(map(
            lambda param_tup: np.all(fn(param_tup[0], param_tup[1])),
            zip(params1, params2))))


# ----------------------- #
# -- NEEDS TO BE FIXED -- #
# ----------------------- #
def sample(x: torch.Tensor, y: torch.Tensor, batch_size: int):
    n_datapoints = x.shape[0]
    assert n_datapoints == y.shape[0] 
    # both should have the same number of samples
    selected = np.random.choice(n_datapoints, size=batch_size)
    # sample with replacement
    return x[selected, :], y[selected]


# ----------------------- #
# -- NEEDS TO BE FIXED -- #
# ----------------------- #
def train(
        make_model,
        make_optim,
        tasks,
        criterion,
        batch_size=32,
        test_hook=None,
        n_epochs: int = 10000):
    """
        make_model: () -> nn.Module
        make_optim: (params) -> torch.optim
        tasks: [(trainx, trainy)]
        criterion: (pred, correct) -> torch.Tensor
        batch_size: int = 32
        test_hook: [nn.Module], int, int -> None
            runs tests on any or all of the modules
            also given info about epoch number
        n_epochs: int > 0

        returns:
            trained_models: [nn.Module]
            losses: [float]
    """
    n_tasks = len(tasks)
    models = [make_model() for _ in n_tasks]
    for model in models[1:]:
        copy_weights(From=model[0], To=model)
        # share the weights of the first model among all models
    optims = [make_optim(model.parameters()) for model in models]
    losses = [[] for _ in models]

    prev_model = None

    for _ in tqdm(range(n_epochs), "Training"):

        task_index = np.random.randint(0, high=n_tasks)
        x, y = sample(*tasks[task_index], batch_size)
        optim = optims[task_index]
        model = models[task_index]

        if prev_model is not None:
            copy_weights(From=prev_model, To=model)
        prev_model = model

        y_hat = model(x)
        loss = criterion(y_hat, y)

        optim.zero_grad()
        loss.backward()
        optim.step()

        losses[task_index].append(loss.item())

    return models, optims, losses


def _ntuple(n):
    def parse(x):
        if isinstance(x, container_abcs.Iterable):
            return x
        return tuple(repeat(x, n))
    return parse


_pair = _ntuple(2)
