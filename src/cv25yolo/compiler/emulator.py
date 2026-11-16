"""A float emulator of the primitives, used to measure value ranges during calibration.

Tensors are numpy arrays of shape (1, channels, height, width).
"""

import numpy as np

from cv25yolo.compiler.records import Primitive


def convolve(
    x: np.ndarray, kernel: np.ndarray, stride: int, padding: int, out_shape: tuple
) -> np.ndarray:
    """Convolve one image with a kernel.

    Args:
        x: Input of shape (1, inputs, height, width).
        kernel: Kernel of shape (outputs, inputs, size, size).
        stride: Step between two windows.
        padding: Number of zero rows and columns before the first pixel.
        out_shape: Shape of the result (1, outputs, height, width).

    Returns:
        The result, without bias.
    """
    outputs, inputs, size, _ = kernel.shape
    _, _, out_height, out_width = out_shape
    height, width = x.shape[2:]
    need_height, need_width = (out_height - 1) * stride + size, (out_width - 1) * stride + size
    padded = np.zeros(
        (inputs, max(need_height, height + padding), max(need_width, width + padding))
    )
    padded[:, padding : padding + height, padding : padding + width] = x[0]
    padded = padded[:, :need_height, :need_width]
    # One column per kernel tap, then a single matrix product.
    columns = np.empty((inputs, size, size, out_height, out_width))
    for i in range(size):
        for j in range(size):
            columns[:, i, j] = padded[
                :, i : i + stride * out_height : stride, j : j + stride * out_width : stride
            ]
    flat = kernel.reshape(outputs, inputs * size * size) @ columns.reshape(inputs * size * size, -1)
    return flat.reshape(1, outputs, out_height, out_width)


def max_pool(x: np.ndarray, size: int, padding: int) -> np.ndarray:
    """Max pooling with stride 1. The border is padded with minus infinity."""
    _, channels, height, width = x.shape
    padded = np.full((1, channels, height + 2 * padding, width + 2 * padding), -np.inf)
    padded[:, :, padding : padding + height, padding : padding + width] = x
    out = np.full(x.shape, -np.inf)
    for i in range(size):
        for j in range(size):
            out = np.maximum(out, padded[:, :, i : i + height, j : j + width])
    return out


def run(primitives: list[Primitive], image: np.ndarray) -> dict[str, np.ndarray]:
    """Run the primitives on one image in floating point.

    Args:
        primitives: The primitives in an order that computes inputs before their
            users.
        image: Planar uint8 image of shape (channels, height, width).

    Returns:
        The output tensor of every primitive, by name.
    """
    raise NotImplementedError("not published")
