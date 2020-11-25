import numpy as np
import torch.nn as nn
from tqdm.notebook import tqdm

from .GatedConv2d import GatedConv2d
from .GatedLinear import GatedLinear
from .utils import sample


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
    models = [make_model() for _ in range(n_tasks)]
    for model in models[1:]:
        copy_weights(From=models[0], To=model)
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

