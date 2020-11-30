from itertools import repeat

import numpy as np
import torch
from torch._six import container_abcs




def seed(rand_seed):
    np.random.seed(rand_seed)
    torch.manual_seed(rand_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False




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


def _ntuple(n):
    def parse(x):
        if isinstance(x, container_abcs.Iterable):
            return x
        return tuple(repeat(x, n))
    return parse

_single = _ntuple(1)
_pair = _ntuple(2)
_triple = _ntuple(3)
_quadruple = _ntuple(4)


def _reverse_repeat_tuple(t, n):
    r"""Reverse the order of `t` and repeat each element for `n` times.
    This can be used to translate padding arg used by Conv and Pooling modules
    to the ones used by `F.pad`.
    """
    return tuple(x for x in reversed(t) for _ in range(n))