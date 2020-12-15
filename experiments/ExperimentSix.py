from datetime import datetime
from typing import Tuple, Optional, List, Dict, Callable

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.optim import Adam

from src.GatedConv2dNew import GatedConv2d
from src.SeparateModels import SeparateModels
from src.SharedModel import SharedModel
from src.gated_utils import train, set_task, TrainResult
from src.plotting import exponential_average
from src.utils import seed

from src.LeNet import LeNet5

import argparse
from torchvision import datasets, transforms
from torch import nn, optim, autograd
import matplotlib.pyplot as plt
import torchvision

#train_mnist = datasets.MNIST('~/datasets/mnist', transform=transforms.ToTensor(), train=True, download=True)
#test_mnist = datasets.MNIST('~/datasets/mnist', transform=transforms.ToTensor(), train=False, download=True)

class ExperimentSixGenerator:
    def __init__(self,
            in_channels: int,
            out_channels: int):
        self.in_channels = in_channels
        self.out_channels = out_channels

    def create_task(self):
        #use this later to differentiate tasks
        a = 1


def run_experiment(
            n_inputs: int,
            n_outputs: int,
            n_tasks: int,
            n_samples: int,
            batch_size: int,
            n_epochs: int,
            model: str,
            learning_rate: 3e-4,
            task_speedup: float = None,
            mask_probability: float = 0.5,
            random_seed: int = 0,
            noise_scale: float = 0.,
            name: str = None,
            additional_test_hooks: Optional[List[Tuple[int, Callable[[nn.Module, int], None]]]] = [])\
            -> (TrainResult, SummaryWriter):
    if random_seed is not None:
        seed(random_seed)

    if name is None:
        dt: datetime = datetime.now()
        name = dt.strftime("%Y-%m-%d@%H-%M")

    if task_speedup is None:
        task_speedup = n_tasks * 3

    assert n_inputs > 0
    assert n_outputs > 0
    assert n_tasks > 0
    assert n_epochs > 0
    assert n_samples > 0
    assert learning_rate > 0
    assert task_speedup > 0
    assert mask_probability >= 0
    assert noise_scale >= 0

    assert model in ['gated', 'shared', 'separate']

    if name not in ['debug', '']:
        writer: SummaryWriter = SummaryWriter("../results/experiment_six/" + name)
    else:
        writer: SummaryWriter


    generator: ExperimentSixGenerator = ExperimentSixGenerator(
        n_inputs,
        n_outputs)

    #tasks: List[ExperimentSixTask] = [generator.create_task() for _ in range(n_tasks)]

    train_mnist = datasets.MNIST('~/datasets/mnist', transform=transforms.ToTensor(), train=True, download=True)
    train_x: torch.Tensor = torch.stack([x for x, y in train_mnist], dim=0)
    train_y: torch.Tensor = torch.stack([torch.from_numpy(np.asarray(y)) for x, y in train_mnist], dim=0)
    test_mnist = datasets.MNIST('~/datasets/mnist', transform=transforms.ToTensor(), train=False, download=True)
    test_x: torch.Tensor = torch.stack([x for x, y in test_mnist], dim=0)
    test_y: torch.Tensor = torch.stack([torch.from_numpy(np.asarray(y)) for x, y in test_mnist], dim=0)
    task_test_data: List[Tuple[torch.Tensor, torch.Tensor]] = [(test_x, test_y)]

    def test_fn(model: nn.Module, epoch: int):
        task_idx: int
        x: torch.Tensor  # torch.Tensor[float32, cpu] (300, input_dimi)
        y: torch.Tensor  # torch.Tensor[float32, cpu] (300, output_dim)
        for task_idx, (x, y) in enumerate(task_test_data):
            set_task(model, task_idx)

            yhat: torch.Tensor = model(x)
            # torch.Tensor[flaot32, cpu] (300, output_dim)
            loss = nn.MSELoss()(yhat, y)

            writer.add_scalar(
                tag=f"Test Loss: {task_idx}",
                scalar_value=loss.item(),
                global_step=epoch)

    model_maker_lookup: Dict[str, Callable[[int], nn.Module]] = {
        'gated': lambda n_tasks: SeparateModels(lambda: LeNet5(gated=True), n_tasks=n_tasks),
        'separate': lambda n_tasks: SeparateModels(lambda: LeNet5(gated=False), n_tasks=n_tasks),
        'shared': lambda n_tasks: SharedModel(LeNet5(gated=False), n_tasks=n_tasks)
    }
    # n_tasks here is shadowed, but it shouldn't matter because it's the exact same value

    results = train(
        make_model=model_maker_lookup[model],
        make_optim=lambda shared_params, task_params: Adam([
            {'params': shared_params, 'lr': learning_rate},
            {'params': task_params, 'lr': learning_rate * task_speedup}]),
        tasks=[(train_x, train_y)],
        criterion=nn.CrossEntropyLoss(),
        test_hooks=[(10, test_fn)] + additional_test_hooks,
        n_epochs=n_epochs)

    writer.add_hparams(hparam_dict={
        'n_inputs': N_INPUTS,
        'n_outputs': N_OUTPUTS,
        'n_tasks': N_TASKS,
        'seed': SEED,
        'num_samples': N_SAMPLES,
        'mask_proba': MASK_PROBABILITY,
        'lr': LEARNING_RATE,
        'task multiplier': TASK_SPEEDUP,
        'batch_size': BATCH_SIZE,
        'n_epochs': N_EPOCHS,
        'model': MODEL
    }, metric_dict={
        f'MSE:({task_idx})': np.mean(loss[-10:]) for task_idx, loss in enumerate(results.losses) if len(loss) > 10
    })

    return results, writer


if __name__ == "__main__":
    N_INPUTS: int = 11
    N_OUTPUTS: int = 11
    N_TASKS: int = 1
    SEED: int = 1
    N_SAMPLES: int = 51
    MASK_PROBABILITY: float = 0.75
    LEARNING_RATE: float = 0.001
    TASK_SPEEDUP: float = N_TASKS * 3
    BATCH_SIZE: int = 32
    N_EPOCHS: int = 20001
    MODEL: str = 'shared'  # (gated, shared, separate)

    name: str = input("Enter model name: ")

    results, writer = run_experiment(
        n_inputs=N_INPUTS,
        n_outputs=N_OUTPUTS,
        n_tasks=N_TASKS,
        mask_probability=MASK_PROBABILITY,
        random_seed=SEED,
        batch_size=BATCH_SIZE,
        task_speedup=TASK_SPEEDUP,
        learning_rate=LEARNING_RATE,
        model=MODEL,
        noise_scale=0.,
        n_samples=N_SAMPLES,
        n_epochs=N_EPOCHS,
        name=name)

    for i in range(len(results.losses)):
        plt.plot(exponential_average(results.losses[i], gamma=0.98))
    plt.title("Loss by task during training")
    plt.legend([i for i in range(len(results.losses))])
