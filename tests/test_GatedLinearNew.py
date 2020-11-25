import sys
sys.path.append('../src')

from typing import List

import pytest
import torch
import torch.nn as nn


from src.utils import seed, forall
from src.gated_utils import copy_weights
from src.GatedLinearNew import GatedLinear

def save_W_params(model: GatedLinear) -> List[nn.Parameter]:
    result: List[nn.Parameter] = [model.WW.detach().numpy().copy()]
    if model.bias:
        result.append(model.bW.detach().numpy().copy())
    return result

def save_M_params(model: GatedLinear) -> List[nn.Parameter]:
    result: List[nn.Parameter] = [model.WMs.detach().numpy().copy()]
    if model.bias:
        result.append(model.bMs.detach().numpy().copy())
    return result

def test_gated_linear_initialization():
    """
    For maximal fairness in testing, we would like a model initialized with nn.Linear
    to have the exact same output as one initialized with GatedLinear, given the same random seed
    This test creates one of each with random seed of 0 and checks that it has the same out
    when a large number of vectors are put in.
    """
    seed(0)
    glin: GatedLinear = GatedLinear(100, 100, bias=True)
    glin.set_task(0)

    seed(0)
    lin: nn.Linear = nn.Linear(100, 100, bias=True)

    randvec: torch.Tensor = torch.rand((100, 100))
    assert torch.all(torch.eq(glin(randvec), lin(randvec))), \
        "At initialization, GatedLinear() should always return same result as nn.Linear"

def test_gated_linear_initialization2():
    """
    Sames as test_gated_linear_initialization except that input and output dimensions are different
    Also, GatedLinear has multiple tasks, all of which must be equivalent to nn.Linear at initialization
    """
    seed(0)
    glin: GatedLinear = GatedLinear(50, 101, bias=True, n_tasks=3)
    glin.set_task(0)

    seed(0)
    lin: nn.Linear = nn.Linear(50, 101, bias=True)

    for task in range(3):
        randvec: torch.Tensor = torch.rand((200, 50))
        glin.set_task(task)
        assert torch.all(torch.eq(glin(randvec), lin(randvec))), \
            "At initialization, GatedLinear() should always return same result as nn.Linear"

    glin.add_tasks(4)
    for task in range(3+4):
        randvec: torch.Tensor = torch.rand((200, 50))
        glin.set_task(task)
        assert torch.all(torch.eq(glin(randvec), lin(randvec))), \
            "At initialization, GatedLinear() should always return same result as nn.Linear"

