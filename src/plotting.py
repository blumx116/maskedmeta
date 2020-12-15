import io
from io import BytesIO

from matplotlib.figure import Figure
from matplotlib.axes._axes import Axes
from torch.utils.tensorboard import SummaryWriter
import tensorflow as tf
import torch

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
