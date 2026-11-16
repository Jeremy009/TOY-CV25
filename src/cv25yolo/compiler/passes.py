"""Step 2 of the pipeline: split the layers that are too large for the accelerator.

Two kinds of layers can be too large to run in one go: some convolutions, and
the SPPF block. Each is replaced by smaller layers that compute the same
result.
"""

import dataclasses

import numpy as np

from cv25yolo.compiler.records import Layer

Weights = dict[str, tuple[np.ndarray, np.ndarray]]


def renumber(layers: list[Layer]) -> list[Layer]:
    """Give the layers ids that match their position in the list."""
    new_id = {layer.id: k for k, layer in enumerate(layers)}
    return [
        dataclasses.replace(layer, id=new_id[layer.id], inputs=[new_id[i] for i in layer.inputs])
        for layer in layers
    ]


def users_of(layers: list[Layer]) -> dict[int, list[int]]:
    """Return, for every layer id, the ids of the layers that read it."""
    users = {}
    for layer in layers:
        for i in layer.inputs:
            users.setdefault(i, []).append(layer.id)
    return users


def split_convs(layers: list[Layer], weights: Weights) -> tuple[list[Layer], Weights, list[str]]:
    """Split every convolution that is too large into parts.

    Each part computes some of the output channels, and a concat puts them back
    together.

    Args:
        layers: The layers of the network.
        weights: Kernel and bias per convolution name.

    Returns:
        The new layers, the new weights and a description of each split.

    Raises:
        ValueError: If a convolution is too large even when split.
    """
    raise NotImplementedError("not published")


def split_sppf(layers: list[Layer], weights: Weights) -> tuple[list[Layer], Weights, list[str]]:
    """Split an SPPF block that is too wide into two blocks of half the width.

    The block is x = silu(cv1(input)), three chained max pools of x, a concat of
    the four tensors, and cv2 on the concat. cv2 is linear, so it can be computed
    as the sum of two convolutions, one per half of the channels:

        half h:  x_h = silu(cv1 part h), its three pools, concat_h
        result:  cv2_half0(concat_0) + cv2_half1(concat_1), with the bias in half 0

    Args:
        layers: The layers of the network.
        weights: Kernel and bias per convolution name.

    Returns:
        The new layers, the new weights and the names of the blocks that were
        split.
    """
    raise NotImplementedError("not published")
