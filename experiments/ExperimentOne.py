from datetime import datetime
from typing import Tuple, Optional, List, Dict, Callable

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.optim import Adam

from src.GatedLinearNew import GatedLinear
from src.SeparateModels import SeparateModels
from src.SharedModel import SharedModel
from src.gated_utils import train, set_task, TrainResult
from src.plotting import exponential_average, Axes, Figure, mpl_to_tensorboard, plot_dataset_loss
from src.utils import seed


class ExperimentOneTask:
    def __init__(self,
            weights: torch.Tensor,
            bias: Optional[torch.Tensor] = None,
            input_range: Tuple[int, int] =None,
            noise_scale: float = 0.):
        assert noise_scale >= 0
        self.in_features: int = weights.shape[1]
        self.out_features: int = weights.shape[0]
        self.weights: torch.Tensor = weights
        # torch.Tensor[float32, cpu] (in_features, out_features)
        self.bias: torch.Tensor = bias
        # torch.Tensor[float32, cpu] (out_features, )
        self.input_range: Tuple[int, int] = input_range
        self.noise_scale = noise_scale

    def sample(self, count: int) -> (torch.Tensor, torch.Tensor):
        with torch.no_grad():
            xs: torch.Tensor = torch.randn(count, self.in_features).float()
            # torch.Tensor[float32, cpu] (count, in_features)
            if self.input_range is not None:
                xs = torch.clamp(xs, *self.input_range)
                # TODO: this isn't great because it might result on a bunch of things
                # at the edges
            ys: torch.Tensor = F.linear(xs, self.weights, self.bias)
            ys += torch.normal(mean=0, std=self.noise_scale, size=ys.shape)
            # torch.Tensor[float32, cpu] (count, out_features)
            return xs, ys


class ExperimentOneGenerator:
    def __init__(self,
            in_features: int,
            out_features: int,
            input_range: Tuple[int, int] = None,
            bias: bool=True,
            mask_proba: float=0.5,
            noise_scale: float = 0.0):
        self.in_features: int = in_features
        self.out_features: int = out_features
        self.input_range: Optional[Tuple[int, int]] = input_range
        self.has_bias: bool = bias
        self.mask_proba: float = mask_proba
        self.noise_scale: float = noise_scale

        self.weights: np.ndarray = np.random.uniform(size=(self.out_features, self.in_features))
        # np.ndarray[float] (out_features, in_features)
        self.bias: Optional[np.ndarray] = np.random.uniform(size=(self.out_features,)) \
            if self.has_bias else None
        # np.ndarray[float] (out_features, )

    def create_task(self):
        weight_mask: np.ndarray = np.random.uniform(size=(self.out_features, self.in_features)) < self.mask_proba
        # np.ndarray[bool] (out_features, in_features)
        task_weights: torch.Tensor = torch.from_numpy(self.weights * weight_mask).float()
        # torch.Tensor[float32, cpu] (out_features, in_features)

        if self.has_bias:
            bias_mask: np.ndarray = np.random.uniform(size=(self.out_features, )) < self.mask_proba
            # np.ndarray [bool] (out_features, )
            task_bias: Optional[torch.Tensor] = torch.from_numpy(self.bias * bias_mask).float()
            # torch.Tensor[float32, cpu] (out_features, )
        else:
            task_bias: Optional[torch.Tensor] = None

        return ExperimentOneTask(task_weights, task_bias, self.input_range, self.noise_scale)


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
            device: str = 'cpu',
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
        writer: SummaryWriter = SummaryWriter("../results/experiment_one/" + name)
    else:
        writer: SummaryWriter


    generator: ExperimentOneGenerator = ExperimentOneGenerator(
        n_inputs,
        n_outputs,
        mask_proba=mask_probability,
        noise_scale=noise_scale)

    tasks: List[ExperimentOneTask] = [generator.create_task() for _ in range(n_tasks)]

    task_train_data: List[Tuple[torch.Tensor, torch.Tensor]] = [task.sample(n_samples) for task in tasks]
    task_train_data = [(x.to(device), y.to(device)) for x, y in task_train_data]
    task_test_data: List[Tuple[torch.Tensor, torch.Tensor]] = [task.sample(300) for task in tasks]
    task_test_data = [(x.to(device), y.to(device)) for x, y in task_test_data]

    def test_fn(model: nn.Module, epoch: int):
        plot_dataset_loss(model, epoch, task_test_data,
            loss_fn=nn.MSELoss(), writer=writer)

        plot_dataset_loss(model, epoch, task_train_data,
            loss_fn=nn.MSELoss(), writer=writer, name="Train Loss: ")


    model_maker_lookup: Dict[str, Callable[[int], nn.Module]] = {
        'gated': lambda n_tasks: GatedLinear(n_inputs, n_outputs, n_tasks=n_tasks).to(device),
        'separate': lambda n_tasks: SeparateModels(lambda: nn.Linear(n_inputs, n_outputs), n_tasks=n_tasks).to(device),
        'shared': lambda n_tasks: SharedModel(nn.Linear(n_inputs, n_outputs), n_tasks=n_tasks).to(device)
    }
    # n_tasks here is shadowed, but it shouldn't matter because it's the exact same value

    results = train(
        make_model=model_maker_lookup[model],
        make_optim=lambda shared_params, task_params: Adam([
            {'params': shared_params, 'lr': learning_rate},
            {'params': task_params, 'lr': learning_rate * task_speedup}]),
        tasks=task_train_data,
        criterion=nn.MSELoss(),
        batch_size=batch_size,
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

    fig: Figure
    ax: Axes
    fig, ax = plt.subplots(1, 1)
    for i in range(len(results.losses)):
        ax.plot(exponential_average(results.losses[i], gamma=0.98))
    ax.set_title("Loss by task during training")
    fig.legend([i for i in range(len(results.losses))])
    writer.add_figure("Train Loss by task during training", fig)

    return results, writer


if __name__ == "__main__":
    N_INPUTS: int = 10
    N_OUTPUTS: int = 10
    N_TASKS: int = 10
    SEED: int = 0
    N_SAMPLES: int = 32
    MASK_PROBABILITY: float = 0.75
    LEARNING_RATE: float = 3e-5
    TASK_SPEEDUP: float = N_TASKS * 3
    BATCH_SIZE: int = 16
    N_EPOCHS: int = 100000
    NOISE_SCALE: float = 0.4
    MODEL: str = 'gated'  # (gated, shared, separate)
    DEVICE: str = 'cpu'

    name: str = input("Enter model name: ")

    results, writer = run_experiment(
        n_inputs=N_INPUTS,
        n_outputs=N_OUTPUTS,
        n_tasks=N_TASKS,
        mask_probability=MASK_PROBABILITY,
        random_seed=SEED,
        batch_size=BATCH_SIZE,
        device=DEVICE,
        task_speedup=TASK_SPEEDUP,
        learning_rate=LEARNING_RATE,
        model=MODEL,
        noise_scale=NOISE_SCALE,
        n_samples=N_SAMPLES,
        n_epochs=N_EPOCHS,
        name=name)

    for i in range(len(results.losses)):
        plt.plot(exponential_average(results.losses[i], gamma=0.98))
    plt.title("Loss by task during training")
    plt.legend([i for i in range(len(results.losses))])
    plt.show()

    print("done")
