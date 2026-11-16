"""Step 3 of the pipeline: lower the layers into the accelerator's operations.

Each layer becomes one or more primitives. A layer that the accelerator has no
operation for is built from simpler ones. The primitives are listed in program
order: the order in which the accelerator will execute them.
"""

import numpy as np

from cv25yolo.compiler.records import Constant, Layer, Primitive


def emit(
    primitives: list[Primitive],
    op: str,
    inputs: list[Primitive | None],
    shape: tuple[int, ...],
    name: str | None = None,
    constants: dict[str, float | np.ndarray] | None = None,
    attributes: dict[str, float] | None = None,
) -> Primitive:
    """Append a primitive to the list and return it.

    Args:
        primitives: The list to append to.
        op: Name of the operation.
        inputs: The primitives whose outputs this one reads. None is an unused optional input.
        shape: Shape of the output tensor.
        name: Name of the output tensor. By default t<index>_<op>.
        constants: Real values of the constant operands, by role.
        attributes: Attributes of the operation.

    Returns:
        The new primitive.
    """
    index = len(primitives) + 1
    primitive = Primitive(
        index=index,
        op=op,
        name=name or f"t{index}_{op}",
        shape=tuple(shape),
        inputs=[None if x is None else x.name for x in inputs],
        constants={k: Constant(np.asarray(v, np.float64)) for k, v in (constants or {}).items()},
        attributes=dict(attributes or {}),
    )
    primitives.append(primitive)
    return primitive


def lower_layer(
    primitives: list[Primitive],
    layer: Layer,
    inputs: list[Primitive],
    weights: dict[str, tuple[np.ndarray, np.ndarray]],
    input_shape: tuple[int, int, int],
) -> Primitive:
    """Emit the primitives of one layer.

    Args:
        primitives: The list to append to.
        layer: The layer to lower.
        inputs: The primitives that compute the layer's inputs.
        weights: Kernel and bias per convolution name.
        input_shape: Shape of the network input (channels, height, width).

    Returns:
        The primitive whose output is the layer's result.
    """
    raise NotImplementedError("not published")


def program_order(layers: list[Layer]) -> list[Layer]:
    """Return the layers in the order the accelerator will execute them.

    Args:
        layers: The layers in the order of the ONNX file.

    Returns:
        The same layers in program order.
    """
    raise NotImplementedError("not published")


def lower(
    layers: list[Layer],
    weights: dict[str, tuple[np.ndarray, np.ndarray]],
    input_shape: tuple[int, int, int],
    in_program_order: bool,
) -> list[Primitive]:
    """Lower all layers into primitives.

    The tensor names are the same for both orders, so that the value ranges measured during
    calibration (which runs in file order) can be looked up by name afterwards.

    Args:
        layers: The layers of the network.
        weights: Kernel and bias per convolution name.
        input_shape: Shape of the network input (channels, height, width).
        in_program_order: True to list the primitives in the order of `program_order`, False
            to keep the order of the ONNX file.

    Returns:
        The primitives, with index 1, 2, 3, ... in list order.
    """
    primitives: list[Primitive] = []
    result = {}
    layer_of = {}  # name of a primitive's output -> id of its layer
    for layer in layers:
        first = len(primitives)
        result[layer.id] = lower_layer(
            primitives, layer, [result[i] for i in layer.inputs], weights, input_shape
        )
        for primitive in primitives[first:]:
            layer_of[primitive.name] = layer.id
    if in_program_order:
        rank = {layer.id: k for k, layer in enumerate(program_order(layers))}
        primitives.sort(key=lambda p: (rank[layer_of[p.name]], p.index))
        for k, primitive in enumerate(primitives):
            primitive.index = k + 1
    return primitives
