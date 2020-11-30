from itertools import zip_longest, chain
from typing import Iterable, List, Tuple, Callable

import torch
import torch.nn as nn
import torch.optim
from torch.optim.optimizer import Optimizer

from src.utils import seed
from src.gated_utils import get_shared_parameters, get_task_parameters, train, TrainResult
from src.GatedLinearNew import GatedLinear


def check_param_eq(
        first: Iterable[nn.Parameter],
        second: Iterable[nn.Parameter],
        message: str = "") -> None:
    for a, b in zip_longest(first, second):
        assert a is not None and b is not None, message
        assert torch.all(a == b), message


def test_set_task():
    assert False


def test_get_shared_parameters():
    seed(0)
    linear: nn.Module = nn.Linear(100, 100)
    check_param_eq(
        linear.parameters(),
        get_shared_parameters(linear),
        message="shared parameters did not return the correct parameters for nn.Linear")

    gated_linear: GatedLinear = GatedLinear(50, 100, bias=True)
    check_param_eq(
        [gated_linear.WW, gated_linear.bW],
        get_shared_parameters(gated_linear),
        message="shared parameters did not return the correct parameters for GatedLinear(bias=True)")

    gated_linear_no_b: GatedLinear = GatedLinear(20, 50, bias=False)
    check_param_eq(
        [gated_linear_no_b.WW],
        get_shared_parameters(gated_linear_no_b),
        message="shared parameters did not return the correct parameters for GatedLinear(bias=False)")

    sequential: nn.Sequential = nn.Sequential(nn.Linear(100, 100), nn.ReLU(), nn.Linear(100, 20))
    check_param_eq(
        sequential.parameters(),
        get_shared_parameters(sequential),
        message="shared parameters mismatch for normal sequential model")

    s1: GatedLinear = GatedLinear(100, 100)
    s2: nn.Linear = nn.Linear(100, 20)
    sequential_mix: nn.Sequential = nn.Sequential(s1, nn.ReLU(), s2)
    check_param_eq(
        chain(s1.parameters(), s2.parameters()),
        get_shared_parameters(sequential_mix),
        message="shared parameters mismatch for mixed sequential model")


def test_get_task_parameters():
    seed(0)
    linear: nn.Module = nn.Linear(100, 100)
    check_param_eq(
        [],
        get_task_parameters(linear, 0),
        message="should not have any task parameters for nn.Linear")
    check_param_eq(
        [],
        get_task_parameters(linear, 1),
        message="should not have any task parameters for any task of nn.Linear")

    gated_linear: GatedLinear = GatedLinear(50, 100, bias=True)
    check_param_eq(
        [gated_linear.WMs[0], gated_linear.bMs[0]],
        get_task_parameters(gated_linear, 0),
        message="GatedLinear should return masks only")

    gated_linear_no_b: GatedLinear = GatedLinear(20, 50, bias=False, n_tasks=3)
    gated_linear_no_b.WMs[1] = nn.Parameter(torch.randn(*gated_linear_no_b.WMs[1].shape))
    gated_linear_no_b.WMs[2] = nn.Parameter(torch.randn(*gated_linear_no_b.WMs[2].shape))
    for i in range(3):
        check_param_eq(
            [gated_linear_no_b.WMs[i]],
            get_task_parameters(gated_linear_no_b, i),
            message="GatedLinear not working for multiple masks")

    seq_gl: GatedLinear = GatedLinear(20, 50, bias=True, n_tasks=2)
    seq_l: nn.Linear = nn.Linear(50, 3, bias=True)
    seq: nn.Sequential = nn.Sequential(seq_gl, nn.ReLU(), seq_l)
    for i in range(2):
        check_param_eq(
            [seq_gl.WMs[i], seq_gl.bMs[i]],
            get_task_parameters(seq, i),
            message=f"Sequential get_task_parameters failed for task {i}")


def test_train_linear():
    seed(0)
    n_tasks: int = 2
    input_dim: int = 2
    output_dim: int = 1
    # weights: torch.Tensor = torch.normal(0, 1, size=(input_dim,))
    weights = torch.Tensor([2, 3])
    ws: List[torch.Tensor] = []
    for i in range(n_tasks):
        w = weights.detach().clone()
        w[i] = 0.
        ws.append(w)

    xs: torch.Tensor = torch.normal(1, 2, (300, input_dim))
    ys: List[torch.Tensor] = []
    for i in range(n_tasks):
        ys.append(xs @ ws[i])

    tasks: List[Tuple[torch.Tensor, torch.Tensor]] = [(xs, ys[i]) for i in range(n_tasks)]
    make_optim: Callable[[Iterable[nn.Parameter], Iterable[nn.Parameter]], Optimizer] = \
        lambda shared, task_weights: torch.optim.Adam(
            [{'params': shared, 'lr': 3e-2},
             {'params': task_weights, 'lr': 6e-1}])
    make_model: Callable[[int], GatedLinear] = lambda n: GatedLinear(input_dim, output_dim, bias=False, n_tasks=2)

    results: TrainResult = train(
        make_model,
        make_optim=make_optim,
        tasks=tasks,
        criterion=nn.MSELoss(),
        batch_size=1,
        n_epochs=10000)
    import sys
    if isinstance(results.model, nn.Linear):
        print("learned weights: ", results.model.weight, file=sys.stderr)
    else:
        print("learned weights: ", [(torch.sigmoid(results.model.WMs[i]) * results.model.WW) for i in range(n_tasks)], file=sys.stderr)
        print("learned masks: ", [torch.sigmoid(results.model.WMs[i]) for i in range(n_tasks)], file=sys.stderr)
    print("correct weights: ", [ws], file=sys.stderr)
    import matplotlib.pyplot as plt
    from src.plotting import exponential_average
    for i in range(n_tasks):
        if len(results.losses[i]) > 0:
            plt.plot(exponential_average(results.losses[i], gamma=0.95))
    plt.title("plot of losses")
    plt.show()

    # task 1 : y = ax_1
    # task 2 : y = bx_2
    # ideal WW : [a, b]
    # idea WM[0] = [-++
    plt.title("plot of gradients")
    grads: torch.Tensor = torch.stack(list(map(lambda g: g[0], results.grads)), dim=0)
    for i in range(grads.shape[1]):
        plt.plot(exponential_average(grads[:, i].numpy(), gamma=0.95))
    plt.show()

    plt.title("plot of weights")
    grads: torch.Tensor = torch.stack(list(map(lambda g: g[0], results.weights)), dim=0)
    for i in range(grads.shape[1]):
        plt.plot(exponential_average(grads[:, i].numpy(), gamma=0.95))
    plt.show()

    xs: torch.Tensor = torch.normal(1, 2, size=(300, 3))

def test_train_shared():
    assert False

def test_train_separate():
    assert False
