import sys
sys.path.append('../src')

from typing import List

import pytest
import torch
import torch.nn as nn

from src.utils import seed, forall
from src.GatedConv2dNew import GatedConv2d

def save_W_params(model: GatedConv2d) -> List[nn.Parameter]:
    result: List[nn.Parameter] = [model.WW.detach().numpy().copy()]
    if model.biasBool:
        result.append(model.bW.detach().numpy().copy())
    return result

def save_M_params(model: GatedConv2d) -> List[nn.Parameter]:
    result: List[nn.Parameter] = [model.WMs.detach().numpy().copy()]
    if model.biasBool:
        result.append(model.bMs.detach().numpy().copy())
    return result

def test_gated_conv2d_initialization():
    """
    For maximal fairness in testing, we would like a model initialized with nn.Conv2d
    to have the exact same output as one initialized with GatedConv2d, given the same random seed
    This test creates one of each with random seed of 0 and checks that it has the same out
    when a large number of vectors are put in.
    """
    seed(0)
    gconv: GatedConv2d = GatedConv2d(16, 33, (3, 5), stride=(2, 1), bias=True)
    gconv.set_task(0)

    seed(0)
    conv: nn.Conv2d = nn.Conv2d(16, 33, (3, 5), stride=(2, 1), bias=True)

    randvec: torch.Tensor = torch.randn((20, 16, 50, 100))
    
    assert torch.all(torch.eq(gconv(randvec), conv(randvec))), \
        "At initialization, GatedConv2d() should always return same result as nn.Conv2d"

def test_gated_conv2d_initialization2():
    """
    Sames as test_gated_conv2d_initialization except that input and output dimensions are different
    Also, GatedConv2d has multiple tasks, all of which must be equivalent to nn.Conv2d at initialization
    """
    seed(0)
    gconv: GatedConv2d = GatedConv2d(15, 32, (2, 4), bias=True, n_tasks=3)
    gconv.set_task(0)

    seed(0)
    conv: nn.Conv2d = nn.Conv2d(15, 32, (2, 4), bias=True)

    for task in range(3):
        randvec: torch.Tensor = torch.rand((200, 15, 60, 120))
        gconv.set_task(task)
        assert torch.all(torch.eq(gconv(randvec), conv(randvec))), \
            "At initialization, GatedConv2d() should always return same result as nn.Conv2d"

    gconv.add_tasks(4)
    for task in range(3+4):
        randvec: torch.Tensor = torch.rand((200, 15, 60, 120))
        gconv.set_task(task)
        assert torch.all(torch.eq(gconv(randvec), conv(randvec))), \
            "At initialization, GatedConv2d() should always return same result as nn.Conv2d"
