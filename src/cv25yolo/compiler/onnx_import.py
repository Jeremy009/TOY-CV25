"""Step 1 of the pipeline: read a YOLOv5 ONNX export into a list of layers.

Only the body of the network is imported: everything that can be reached from the input
through the operations below. The Detect head's reshapes, sigmoids and box arithmetic (and a
built-in NMS, if any) are left out. They run on the CPU, see examples/decode.py. The tensors
where the body ends, the three Detect convolutions, become the network outputs.

    ONNX                                  layer
    Conv (batch norm already folded in)   conv
    Sigmoid(x) followed by Mul(x, ...)    silu
    Add                                   add
    Concat on the channel axis            concat
    MaxPool                               maxpool
    Resize (nearest, integer factor)      upsample
"""

import collections

import numpy as np
import onnx
import onnx.helper
import onnx.numpy_helper
import onnxruntime

from cv25yolo.compiler.records import Layer


def attributes_of(node: onnx.NodeProto) -> dict:
    """Return the attributes of an ONNX node as a dictionary."""
    return {a.name: onnx.helper.get_attribute_value(a) for a in node.attribute}


def read_anchors(model: onnx.ModelProto) -> list[np.ndarray]:
    """Read the anchor boxes of the Detect head from the ONNX file.

    The export stores them as tensors of shape (1, anchors, height, width, 2) that are constant
    over height and width.

    Args:
        model: The loaded ONNX model.

    Returns:
        One array of shape (anchors, 2) per output scale, in pixels, finest scale first.
    """
    arrays = [onnx.numpy_helper.to_array(t) for t in model.graph.initializer]
    for node in model.graph.node:
        if node.op_type == "Constant" and node.attribute:
            if node.attribute[0].type == onnx.AttributeProto.TENSOR:
                arrays.append(onnx.numpy_helper.to_array(node.attribute[0].t))
    found = []
    for a in arrays:
        if a.ndim == 5 and a.shape[-1] == 2 and a.shape[0] == 1:
            if np.allclose(a, a[:, :, :1, :1, :]) and a.min() > 0:
                found.append((a.shape[2], a[0, :, 0, 0, :].astype(np.float64)))
    found.sort(key=lambda item: -item[0])
    return [anchors for _, anchors in found]


def computed_resize_scales(model: onnx.ModelProto, shape: list[int]) -> dict[str, np.ndarray]:
    """Work out the scale factors of Resize nodes whose output size is computed in the graph.

    Some exports compute the size with Shape, Slice and Concat nodes. The input shape is
    fixed, so one run of the model on a zero image gives the sizes.

    Args:
        model: The loaded ONNX model.
        shape: Shape of the network input (batch, channels, height, width).

    Returns:
        The scale factors (batch, channels, height, width) by the name of the Resize node's
        `sizes` input.
    """
    probe = onnx.ModelProto()
    probe.CopyFrom(model)
    resizes = [
        n for n in probe.graph.node if n.op_type == "Resize" and len(n.input) > 3 and n.input[3]
    ]
    wanted = []
    for node in resizes:
        wanted += [node.input[3], node.input[0]]
    existing = {o.name for o in probe.graph.output}
    for name in dict.fromkeys(wanted):
        if name not in existing:
            probe.graph.output.append(onnx.ValueInfoProto(name=name))
    session = onnxruntime.InferenceSession(
        probe.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    zeros = np.zeros([d if d > 0 else 1 for d in shape], np.float32)
    names = [o.name for o in session.get_outputs()]
    values = dict(zip(names, session.run(None, {session.get_inputs()[0].name: zeros})))
    scales = {}
    for node in resizes:
        size, source = values[node.input[3]].ravel(), values[node.input[0]].shape
        scales[node.input[3]] = np.array(
            [float(o) / float(i) for o, i in zip(size, source)], np.float32
        )
    return scales


def read_layers(
    model: onnx.ModelProto,
) -> tuple[list[Layer], dict[str, tuple[np.ndarray, np.ndarray]], tuple[int, int, int]]:
    """Read the body of a YOLOv5 ONNX export.

    Args:
        model: The loaded ONNX model.

    Returns:
        The layers in execution order (the first is the input, the last lists the outputs),
        the weights as {conv name: (kernel, bias)}, and the input shape (channels, height,
        width).

    Raises:
        ValueError: If the model uses something this converter does not support.
    """
    graph = model.graph
    constants = {t.name: onnx.numpy_helper.to_array(t) for t in graph.initializer}
    initializer_names = set(constants)
    for node in graph.node:
        if node.op_type == "Constant":
            constants[node.output[0]] = onnx.numpy_helper.to_array(attributes_of(node)["value"])
    users = collections.defaultdict(list)
    for node in graph.node:
        for name in node.input:
            users[name].append(node)
    graph_inputs = [i for i in graph.input if i.name not in initializer_names]
    if len(graph_inputs) != 1:
        raise ValueError(f"the model must have one input, found {len(graph_inputs)}")
    shape = [d.dim_value for d in graph_inputs[0].type.tensor_type.shape.dim]
    if len(shape) != 4 or shape[1] != 3 or min(shape[2:]) <= 0:
        raise ValueError(
            f"the input must have a fixed shape (batch, 3, height, width), found {shape}"
        )
    if shape[2] % 32 or shape[3] % 32:
        raise ValueError(
            f"input height and width must be multiples of 32, found {shape[2]} x {shape[3]}"
        )

    layers: list[Layer] = []
    weights = {}
    layer_of = {}  # ONNX tensor name -> id of the layer that computes it
    silu_input = {}  # output of a Sigmoid -> its input tensor
    resize_scales = None

    def add(op: str, inputs: list[int], name: str, **attributes: int) -> int:
        layers.append(Layer(len(layers), op, list(inputs), name, attributes))
        return len(layers) - 1

    layer_of[graph_inputs[0].name] = add("input", [], "images")
    for node in graph.node:
        op = node.op_type
        inputs = [name for name in node.input if name]
        attributes = attributes_of(node)
        name = node.name or node.output[0]
        if op == "Identity":  # an alias of a constant or of a tensor
            if inputs[0] in constants:
                constants[node.output[0]] = constants[inputs[0]]
            elif inputs[0] in layer_of:
                layer_of[node.output[0]] = layer_of[inputs[0]]
        elif op == "Conv" and inputs[0] in layer_of:
            kernel = constants[inputs[1]].astype(np.float32)
            if len(inputs) > 2:
                bias = constants[inputs[2]].astype(np.float32)
            else:
                bias = np.zeros(kernel.shape[0], np.float32)
            strides = list(attributes.get("strides", [1, 1]))
            pads = list(attributes.get("pads", [0, 0, 0, 0]))
            if int(attributes.get("group", 1)) != 1:
                raise ValueError(f"{name}: grouped convolutions are not supported")
            if list(attributes.get("dilations", [1, 1])) != [1, 1]:
                raise ValueError(f"{name}: dilated convolutions are not supported")
            if (
                kernel.shape[2] != kernel.shape[3]
                or strides[0] != strides[1]
                or len(set(pads)) != 1
            ):
                raise ValueError(
                    f"{name}: only square kernels, equal strides and equal pads are supported"
                )
            layer_of[node.output[0]] = add(
                "conv",
                [layer_of[inputs[0]]],
                name,
                kernel=int(kernel.shape[2]),
                stride=int(strides[0]),
                padding=int(pads[0]),
            )
            weights[name] = (kernel, bias)
        elif op == "Sigmoid" and inputs[0] in layer_of:
            silu_input[node.output[0]] = inputs[0]
        elif op == "Mul":
            sigmoids = [x for x in inputs if x in silu_input]
            if len(inputs) == 2 and sigmoids and silu_input[sigmoids[0]] in inputs:
                layer_of[node.output[0]] = add("silu", [layer_of[silu_input[sigmoids[0]]]], name)
        elif op == "Add":
            if len(inputs) == 2 and all(x in layer_of for x in inputs):
                layer_of[node.output[0]] = add("add", [layer_of[x] for x in inputs], name)
        elif op == "Concat":
            if all(x in layer_of for x in inputs) and attributes.get("axis") in (1, -3):
                layer_of[node.output[0]] = add("concat", [layer_of[x] for x in inputs], name)
        elif op == "MaxPool" and inputs[0] in layer_of:
            kernel_shape = list(attributes["kernel_shape"])
            strides = list(attributes.get("strides", [1, 1]))
            pads = list(attributes.get("pads", [0] * 4))
            if kernel_shape[0] != kernel_shape[1] or strides != [1, 1] or len(set(pads)) != 1:
                raise ValueError(f"{name}: only square max pooling with stride 1 is supported")
            layer_of[node.output[0]] = add(
                "maxpool",
                [layer_of[inputs[0]]],
                name,
                kernel=int(kernel_shape[0]),
                padding=int(pads[0]),
            )
        elif op == "Resize" and inputs[0] in layer_of:
            mode = attributes.get("mode", b"nearest")
            mode = mode.decode() if isinstance(mode, bytes) else mode
            if mode != "nearest":
                raise ValueError(
                    f"{name}: only nearest-neighbour upsampling is supported, found {mode}"
                )
            scales = None
            if len(node.input) > 2 and node.input[2] in constants:
                scales = constants[node.input[2]]
            if (scales is None or scales.size == 0) and len(node.input) > 3 and node.input[3]:
                if resize_scales is None:
                    resize_scales = computed_resize_scales(model, shape)
                scales = resize_scales.get(node.input[3])
            if scales is None or scales.size == 0:
                raise ValueError(f"{name}: the upsampling factor could not be determined")
            scales = [float(v) for v in scales.ravel()]
            if scales[2] != scales[3] or scales[2] != int(scales[2]):
                raise ValueError(
                    f"{name}: only integer upsampling factors are supported, found {scales}"
                )
            layer_of[node.output[0]] = add(
                "upsample", [layer_of[inputs[0]]], name, scale=int(scales[2])
            )

    # The network outputs: body tensors that are read by an operation outside the body. A Shape
    # node only reads the (fixed) dimensions. A Sigmoid counts unless it is part of a SiLU.
    outputs = []
    graph_outputs = {o.name for o in graph.output}
    for tensor, layer_id in layer_of.items():
        if layers[layer_id].op == "input":
            continue
        outside = [
            u
            for u in users[tensor]
            if not any(o in layer_of for o in u.output) and u.op_type not in ("Sigmoid", "Shape")
        ]
        for user in users[tensor]:
            if user.op_type == "Sigmoid":
                in_body = {
                    x
                    for v in users[user.output[0]]
                    for x in v.input
                    if v.output and v.output[0] in layer_of
                }
                if user.output[0] not in in_body:
                    outside.append(user)
        if outside or tensor in graph_outputs:
            outputs.append(layer_id)
    add("output", sorted(outputs), "output")

    # Drop the layers that no output needs and number the rest again.
    needed, stack = set(), [len(layers) - 1]
    while stack:
        layer_id = stack.pop()
        if layer_id not in needed:
            needed.add(layer_id)
            stack += layers[layer_id].inputs
    kept = [layer for layer in layers if layer.id in needed]
    new_id = {layer.id: k for k, layer in enumerate(kept)}
    layers = [
        Layer(new_id[x.id], x.op, [new_id[i] for i in x.inputs], x.name, x.attributes) for x in kept
    ]
    weights = {x.name: weights[x.name] for x in layers if x.op == "conv"}

    output_ops = [layers[i].op for i in layers[-1].inputs]
    if not output_ops or any(op != "conv" for op in output_ops):
        found = sorted({n.op_type for n in graph.node})
        raise ValueError(
            "this does not look like a YOLOv5 export with SiLU activations: the network body "
            f"must end at the Detect convolutions, but it ends at {output_ops}. "
            f"Operations in the file: {found}"
        )
    return layers, weights, tuple(shape[1:])
