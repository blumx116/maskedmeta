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
from src.gated_utils import train, set_task, TrainResult, trainLenet
from src.plotting import plot_dataset_loss
from src.utils import seed

from src.LeNet import LeNet5
from src.LeNet import transform_grayscale_to_RGB, transform_grayscale_to_RGB_inverse
from torch.utils.data import DataLoader

from time import time

import argparse
from torchvision import datasets, transforms
from torch import nn, optim, autograd
import matplotlib.pyplot as plt
import torchvision

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

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

    colors = [[0.5, 0, 0], [0, 0.75, 0], [1, 1, 1]]
    inverses = [False, False, False]
    assert(len(inverses) == len(colors))
    task_train_data = []
    task_test_data = []
    task_train_loader = []
    task_test_loader = []
    for c in range(len(colors)):
        if inverses[c]:
            sizetransforms = transforms.Compose(
                [transforms.Resize((32, 32)), transforms.ToTensor(), transform_grayscale_to_RGB_inverse(colors[c])])
        else:
            sizetransforms = transforms.Compose(
                [transforms.Resize((32, 32)), transforms.ToTensor(), transform_grayscale_to_RGB(colors[c])])
        #download and create datasets
        train_mnist = datasets.MNIST('~/datasets/mnist', transform=sizetransforms, train=True, download=True)
        test_mnist = datasets.MNIST('~/datasets/mnist', transform=sizetransforms, train=False, download=True)
        #define data loaders
        train_loader = DataLoader(dataset=train_mnist,
                              batch_size=len(train_mnist),
                              shuffle=True)
        test_loader = DataLoader(dataset=test_mnist,
                                  batch_size=len(test_mnist),
                                  shuffle=True)
        task_train_data.append(train_mnist)
        task_test_data.append(test_mnist)
        task_train_loader.append(train_loader)
        task_test_loader.append(test_loader)

    def test_fn(model: nn.Module, epoch: int):
        task_idx: int
        criterion = nn.CrossEntropyLoss()

        for task_idx, tr_loader in enumerate(task_train_loader):
            set_task(model, task_idx)
            model.eval()
            running_loss = 0
            correct_pred = 0
            n = 0

            for X, y_true in tr_loader:
                # Forward pass and record loss
                y_hat = model(X)
                loss = criterion(y_hat, y_true)
                running_loss += loss.item() * X.size(0)
                _, predicted_labels = torch.max(y_hat, 1)
                n += y_true.size(0)
                correct_pred += (predicted_labels == y_true).sum()

            train_loss = running_loss / len(tr_loader.dataset)
            train_accuracy = correct_pred.float() / n
            writer.add_scalar(
                tag="Train Loss: " + str(task_idx),
                scalar_value=train_loss,
                global_step=epoch)
            writer.add_scalar(
                tag="Training Accuracy " + str(task_idx),
                scalar_value=train_accuracy,
                global_step=epoch)
        for task_idx, ts_loader in enumerate(task_test_loader):
            set_task(model, task_idx)
            model.eval()
            running_loss = 0
            correct_pred = 0
            n = 0

            for X, y_true in ts_loader:
                # Forward pass and record loss
                y_hat = model(X)
                loss = criterion(y_hat, y_true)
                running_loss += loss.item() * X.size(0)
                _, predicted_labels = torch.max(y_hat, 1)
                n += y_true.size(0)
                correct_pred += (predicted_labels == y_true).sum()

            test_loss = running_loss / len(ts_loader.dataset)
            test_accuracy = correct_pred.float() / n
            writer.add_scalar(
                tag="Test Loss: " + str(task_idx),
                scalar_value=test_loss,
                global_step=epoch)
            writer.add_scalar(
                tag="Testing Accuracy " + str(task_idx),
                scalar_value=test_accuracy,
                global_step=epoch)


    model_maker_lookup: Dict[str, Callable[[int], nn.Module]] = {
        'gated': lambda n_tasks: LeNet5(gated=True, n_tasks=n_tasks),
        'separate': lambda n_tasks: SeparateModels(lambda: LeNet5(gated=False), n_tasks=n_tasks),
        'shared': lambda n_tasks: SharedModel(LeNet5(gated=False), n_tasks=n_tasks)
    }
    # n_tasks here is shadowed, but it shouldn't matter because it's the exact same value

    results = trainLenet(
        make_model=model_maker_lookup[model],
        make_optim=lambda shared_params, task_params: Adam([
            {'params': shared_params, 'lr': learning_rate},
            {'params': task_params, 'lr': learning_rate * task_speedup}]),
        tasks=task_train_data,
        criterion=nn.CrossEntropyLoss(),
        test_hooks=[(10, test_fn)] + additional_test_hooks,
        n_epochs=n_epochs)

    writer.add_hparams(hparam_dict={
        'n_inputs': N_INPUTS,
        'n_outputs': N_OUTPUTS,
        'n_tasks': N_TASKS,
        'seed': SEED,
        'num_samples': N_SAMPLES,
        'lr': LEARNING_RATE,
        'task multiplier': TASK_SPEEDUP,
        'batch_size': BATCH_SIZE,
        'n_epochs': N_EPOCHS,
        'model': MODEL
    }, metric_dict={
        f'CrossEntropy:({task_idx})': np.mean(loss[-10:]) for task_idx, loss in enumerate(results.losses) if len(loss) > 10
    })

    return results, writer


if __name__ == "__main__":
    N_INPUTS: int = 11
    N_OUTPUTS: int = 11
    N_TASKS: int = 3
    SEED: int = 1
    N_SAMPLES: int = 51
    MASK_PROBABILITY: float = 0.75
    LEARNING_RATE: float = 0.001
    TASK_SPEEDUP: float = N_TASKS * 3
    BATCH_SIZE: int = 32
    N_EPOCHS: int = 1000#20001
    MODEL: str = 'shared'  # (gated, shared, separate)

    print(MODEL + " for " + str(N_TASKS) + " tasks")

    name: str = input("Enter model name: ")

    start = time()

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

    stop = time()
    print(str(stop-start)+'seconds')

    '''plt.figure()
    for i in range(len(results.losses)):
        #plt.plot(exponential_average(results.losses[i], gamma=0.98))
        plt.plot(results.losses[i])
    plt.title("Loss by task during training")
    plt.legend([i for i in range(len(results.losses))])
    plt.show()'''

    '''plt.figure()
    for i in range(len(results.accuracy)):
        plt.plot(results.accuracy[i])
    plt.title("Accuracy")
    plt.legend([i for i in range(len(results.accuracy))])
    plt.show()'''