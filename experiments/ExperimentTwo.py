from typing import Tuple, Callable, Dict, Optional

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.optim import Adam

from experiments.ExperimentOne import ExperimentOneTask, ExperimentOneGenerator

class ExperimentTwoTask:
    def __init__(self,
            inner: ExperimentOneTask):
        self.inner: ExperimentOneTask = inner

    def sample(self, count: int) -> (torch.Tensor, torch.Tensor):
        xs: torch.Tensor # torch.Tensor[float32, cpu] (count, in_features)
        ys: torch.Tensor # torch.Tensor[float32, cpu] (count, out_features)
        xs, ys = self.inner.sample(count)
        ys = torch.sin(ys)
        return xs, ys

class ExperimentTwoGenerator:
    def __init__(self,
            in_features: int,
            out_features: int,
            input_range: Tuple[int, int] = None,
            bias: bool = True,
            mask_proba: float = 0.5,
            noise_scale: float = 0.0):

        self.inner: ExperimentOneGenerator = ExperimentOneGenerator(
            in_features=in_features,
            out_features=out_features,
            input_range=input_range,
            bias=bias,
            mask_proba=mask_proba,
            noise_scale=noise_scale)

    def create_task(self) -> ExperimentTwoTask:
        inner_task: ExperimentOneTask = self.inner.create_task()
        return ExperimentTwoTask(inner_task)

if __name__ == "__main__":
    generator: ExperimentTwoGenerator = ExperimentTwoGenerator(2, 1, input_range=[-10, 10])

    tasks = [generator.create_task() for _ in range(2)]
    data = [task.sample(100) for task in tasks]

    for task_idx in range(2):
        x, y = data[task_idx]
        for i in range(2):
            plt.scatter(x[:,i], y[:, 0])
        plt.title(f"Plot of data for task {task_idx}")
        plt.legend([1, 2])
        plt.show()

        plt.bar(list(range(2)), tasks[task_idx].inner.weights[0,:].numpy())
        plt.title(f"Plot of weights for task {task_idx}")
        plt.show()