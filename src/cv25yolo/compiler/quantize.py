"""Step 4 of the pipeline: choose a fixed-point format for every tensor and constant.

The accelerator computes with integers. A tensor is stored as 8-bit or 16-bit
integers with a fixed number of fraction bits: value = integer / 2**fraction (see
`records.Format`). More fraction bits give a finer step but a smaller range, so the
format follows from the range of values the tensor takes. Calibration measures
that range by running the network on sample images.

On top of that, the accelerator puts its own constraints on the formats of some
tensors. Those rules are not published.
"""

from collections.abc import Iterable

import numpy as np

from cv25yolo.compiler import emulator
from cv25yolo.compiler.records import Format, Primitive

PERCENTILE = 95  # share of the calibration images whose range a tensor's format must cover
Ranges = dict[str, tuple[float, float]]


def calibrate(primitives: list[Primitive], images: Iterable[np.ndarray]) -> Ranges:
    """Measure the range of every tensor on a set of images.

    Per image the minimum and the maximum of each tensor are recorded. The range of a tensor
    is the 5th percentile of the minima and the 95th percentile of the maxima, so that a few
    unusual images do not cost every other image precision.

    Args:
        primitives: The primitives, without formats.
        images: Planar uint8 images of the network's input shape.

    Returns:
        The range (low, high) of every tensor, by name.
    """
    seen = {}
    for image in images:
        for name, tensor in emulator.run(primitives, image).items():
            seen.setdefault(name, []).append((float(tensor.min()), float(tensor.max())))
    ranges = {}
    for name, values in seen.items():
        values = np.array(values)
        low = float(np.percentile(values[:, 0], 100 - PERCENTILE))
        high = float(np.percentile(values[:, 1], PERCENTILE))
        ranges[name] = (low, high)
    return ranges


def fit(low: float, high: float, wide: int = 0, fraction_cap: int | None = None) -> Format:
    """Return the format with the most fraction bits that still holds the range.

    Args:
        low: Smallest value.
        high: Largest value.
        wide: 0 to try 8 bit first, 1 for 16 bit.
        fraction_cap: Upper limit for the number of fraction bits.

    Returns:
        The format. An 8-bit format that would need a negative number of fraction bits
        becomes 16 bit instead.
    """
    signed = int(low < 0)
    largest = max(abs(low), abs(high), 1e-12)
    bits = 16 if wide else 8
    largest_integer = (1 << (bits - signed)) - 1
    fraction = int(np.floor(np.log2(largest_integer / largest)))
    if fraction_cap is not None:
        fraction = min(fraction, fraction_cap)
    if not wide and fraction < 0:
        return fit(low, high, 1, fraction_cap)
    return Format(signed, wide, fraction)


def tensor_formats(
    primitives: list[Primitive], ranges: Ranges, readers: dict[str, list[str]]
) -> dict[str, Format]:
    """Choose a format for every tensor from its range and the operation that makes it.

    Args:
        primitives: The primitives in program order.
        ranges: The calibrated range per tensor name.
        readers: The operations that read each tensor, by tensor name.

    Returns:
        The format per tensor name.
    """
    raise NotImplementedError("not published")


def silu_aware_range(
    p: Primitive, ranges: Ranges, readers: dict[str, list[str]]
) -> tuple[float, float]:
    """Return the calibrated range of a tensor, ignoring values that a SiLU maps to zero.

    SiLU(x) is smaller than 2e-6 in size for x < -16. For a convolution that is
    read only by its SiLU, values below -16 therefore do not matter. Without this,
    one very negative outlier can force a 16-bit output.

    Args:
        p: The primitive that makes the tensor.
        ranges: The calibrated range per tensor name.
        readers: The operations that read each tensor, by tensor name.

    Returns:
        The range (low, high) to choose the format from.
    """
    raise NotImplementedError("not published")


def align_formats(primitives: list[Primitive], formats: dict[str, Format]) -> list[tuple]:
    """Make the formats agree where an operation needs its tensors in one format.

    Args:
        primitives: The primitives in program order.
        formats: The format per tensor name. Updated in place.

    Returns:
        The conversions to insert, as tuples (source tensor name, format,
        primitive that reads the converted tensor).
    """
    raise NotImplementedError("not published")


def set_conv_integers(p: Primitive, input_format: Format) -> None:
    """Compute the integer weights, shifts and bias of a convolution.

    Args:
        p: The conv primitive, with its output format set. Its `integers` are
            filled in.
        input_format: Format of the convolution's input tensor.
    """
    raise NotImplementedError("not published")


def quantize(primitives: list[Primitive], ranges: Ranges) -> list[Primitive]:
    """Set the format of every tensor and constant, and the integers of every convolution.

    Args:
        primitives: The primitives in program order, without formats.
        ranges: The calibrated range per tensor name.

    Returns:
        The primitives with formats, numbered again from 1.
    """
    raise NotImplementedError("not published")
