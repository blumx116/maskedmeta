import io
from io import BytesIO
from typing import List, Tuple, Callable

from matplotlib.figure import Figure
from matplotlib.axes._axes import Axes
from torch.utils.tensorboard import SummaryWriter
import tensorflow as tf
import torch
import torch.nn as nn

from src.gated_utils import set_task


def exponential_average(vals, gamma=0.99):
    result = [vals[0]]
    for v in vals[1:]:
        result.append((gamma * result[-1]) + ((1-gamma) * v))
    return result


def mpl_to_tensorboard(fig: Figure) -> torch.Tensor:
    buf: BytesIO = BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    img: tf.Tensor = tf.image.decode_png(buf.getvalue(), channels=4)
    img = tf.expand_dims(img, 0)
    return torch.from_numpy(img.numpy())


def plot_dataset_loss(model: nn.Module, epoch: int,
                      dataset: List[Tuple[torch.Tensor, torch.Tensor]],
                      loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
                      writer: SummaryWriter, name="Test Loss: ") -> None:
    task_idx: int
    x: torch.Tensor  # torch.Tensor[float32, cpu] (300, input_dimi)
    y: torch.Tensor  # torch.Tensor[float32, cpu] (300, output_dim)

    for task_idx, (x, y) in enumerate(dataset):
        set_task(model, task_idx)

        yhat: torch.Tensor = model(x)
        # torch.Tensor[flaot32, cpu] (300, output_dim)
        loss: torch.Tensor = loss_fn(yhat, y)

        writer.add_scalar(
            tag=name + " " + str(task_idx),
            scalar_value=loss.item(),
            global_step=epoch)
