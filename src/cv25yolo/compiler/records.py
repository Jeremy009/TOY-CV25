"""The records that the front end of the compiler passes from step to step.

The ONNX file is first read into a list of `Layer` records (convolution, SiLU,
add, ...). The layers are then lowered into a list of `Primitive` records: the
operations the accelerator executes. The quantizer fills in the number formats and
the integer parameters of the convolutions.
"""

from dataclasses import dataclass, field
from typing import NamedTuple

import numpy as np


class Format(NamedTuple):
    """Number format of a tensor or a constant.

    Attributes:
        signed: 1 when the integers are signed, 0 when they are unsigned.
        wide: 0 for 8-bit integers, 1 for 16-bit integers.
        fraction: Number of fraction bits. The real value is
            integer / 2**fraction.
    """

    signed: int
    wide: int
    fraction: int


@dataclass
class Layer:
    """One node of the network as it is read from the ONNX file.

    Attributes:
        id: Position of the layer in the layer list.
        op: One of input, conv, silu, add, concat, maxpool, upsample, copy, output.
        inputs: Ids of the layers whose results this layer reads.
        name: Name of the ONNX node (for a convolution also the key of its weights).
        attributes: kernel, stride and padding for conv and maxpool, scale for upsample,
            split_part for a convolution that is one part of a split convolution.
    """

    id: int
    op: str
    inputs: list[int]
    name: str
    attributes: dict[str, int] = field(default_factory=dict)


@dataclass
class Constant:
    """A constant operand of a primitive.

    Attributes:
        values: The real (float) values.
        format: The number format they are stored in. None until the quantizer
            has run.
    """

    values: np.ndarray
    format: Format | None = None


@dataclass
class ConvIntegers:
    """The integer parameters of one convolution.

    Attributes:
        weights: Integer weights, shape (outputs, inputs, kernel, kernel).
        shifts: Shift per output channel.
        bias: Integer bias per output channel.
        wide: True when the output is 16 bit.
    """

    weights: np.ndarray
    shifts: np.ndarray
    bias: np.ndarray
    wide: bool


@dataclass
class Primitive:
    """One operation of the accelerator, with one output tensor.

    Attributes:
        index: Position in program order, starting at 1.
        op: Name of the operation.
        name: Name of the output tensor.
        shape: Shape of the output tensor (1, channels, height, width).
        inputs: Names of the input tensors. None marks an unused optional input.
        constants: Constant operands by role, such as the weights and the bias of
            a convolution.
        attributes: Attributes of the operation, such as the kernel size.
        format: Number format of the output tensor. None until the quantizer has
            run.
        integers: Integer parameters of a convolution. None until the quantizer
            has run.
    """

    index: int
    op: str
    name: str
    shape: tuple[int, int, int, int]
    inputs: list[str | None]
    constants: dict[str, Constant] = field(default_factory=dict)
    attributes: dict[str, float] = field(default_factory=dict)
    format: Format | None = None
    integers: ConvIntegers | None = None
