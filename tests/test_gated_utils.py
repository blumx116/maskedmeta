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
    n_tasks: int = 3
    input_dim: int = 3
    output_dim: int = 1
    weights: torch.Tensor = torch.normal(0, 1, size=(input_dim,))
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
    make_optim: Callable[[Iterable[nn.Parameter]], Optimizer] = \
        lambda params, idx=None: torch.optim.Adam(params, lr=3e-2)
    make_model: Callable[[int], GatedLinear] = lambda n: GatedLinear(input_dim, output_dim, bias=True, n_tasks=n)

    results: TrainResult = train(
        make_model,
        make_shared_optim=make_optim,
        make_task_optim=make_optim,
        tasks=tasks,
        criterion=nn.MSELoss(),
        batch_size=32,
        n_epochs=10000)

    xs: torch.Tensor = torch.normal(1, 2, size=(300, 3))

def test_train_shared():
    assert False

def test_train_separate():
    assert False
