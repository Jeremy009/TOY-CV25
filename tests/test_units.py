"""Small tests of single functions."""

import numpy as np
from decode import iou, select

from cv25yolo.compiler import emulator
from cv25yolo.compiler.quantize import fit
from cv25yolo.compiler.records import Format


def test_fit_uses_all_bits_of_the_range():
    """The chosen format has the most fraction bits that still hold the range."""
    assert fit(0.0, 1.0) == Format(signed=0, wide=0, fraction=7)
    assert fit(-1.0, 1.0) == Format(signed=1, wide=0, fraction=6)
    assert fit(-3.0, 100.0) == Format(signed=1, wide=0, fraction=0)


def test_fit_widens_a_range_that_8_bits_cannot_hold():
    """A range beyond 127 becomes 16 bit instead of saturating."""
    assert fit(-300.0, 300.0) == Format(signed=1, wide=1, fraction=6)


def test_convolve_matches_a_direct_sum():
    """The emulator's convolution equals the definition, also with stride and padding."""
    random = np.random.default_rng(0)
    x = random.normal(size=(1, 2, 6, 6))
    kernel = random.normal(size=(3, 2, 3, 3))
    got = emulator.convolve(x, kernel, stride=2, padding=1, out_shape=(1, 3, 3, 3))
    padded = np.pad(x, ((0, 0), (0, 0), (1, 1), (1, 1)))
    for channel in range(3):
        for row in range(3):
            for column in range(3):
                window = padded[0, :, 2 * row : 2 * row + 3, 2 * column : 2 * column + 3]
                assert np.isclose(got[0, channel, row, column], (window * kernel[channel]).sum())


def test_max_pool_keeps_the_shape():
    """Max pooling with stride 1 and padding returns the largest neighbour of every pixel."""
    x = np.arange(16.0).reshape(1, 1, 4, 4)
    pooled = emulator.max_pool(x, size=3, padding=1)
    assert pooled.shape == x.shape
    assert pooled[0, 0, 0, 0] == 5 and pooled[0, 0, 3, 3] == 15


def test_select_drops_overlapping_boxes():
    """Of two boxes of one class on the same spot only the better one is kept."""
    rows = np.array(
        [
            [50, 50, 20, 20, 0.9, 0.9],
            [51, 50, 20, 20, 0.8, 0.9],
            [150, 150, 20, 20, 0.7, 0.9],
            [250, 250, 20, 20, 0.1, 0.9],
        ]
    )
    kept = select(rows)
    assert len(kept) == 2
    assert np.allclose(kept[0, :4], [40, 40, 60, 60])
    assert iou(kept[0, :4], kept[1, :4]) == 0
