"""A tiny YOLOv5-like network for the tests: two convolutions with SiLU and one Detect scale."""

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
from PIL import Image

SIZE = 64  # input height and width
WIDTH = 16  # output channels of the first convolution, as in YOLOv5n
ANCHORS = [[10.0, 13.0], [16.0, 30.0], [33.0, 23.0]]
CLASSES = 1


def make_model(path: str, seed: int = 0) -> None:
    """Write the tiny network to an ONNX file.

    The body is conv 6x6 stride 2 + SiLU, conv 3x3 stride 2 + SiLU and a 1x1 Detect
    convolution. A Sigmoid after it stands for the rest of the Detect head.
    """
    random = np.random.default_rng(seed)
    nodes, initializers = [], []

    def conv(
        name: str, source: str, inputs: int, outputs: int, kernel: int, stride: int, pad: int
    ) -> str:
        weights = random.normal(
            0, 1.5 / np.sqrt(inputs * kernel * kernel), (outputs, inputs, kernel, kernel)
        )
        initializers.append(numpy_helper.from_array(weights.astype(np.float32), name + ".weight"))
        bias = random.normal(0, 0.2, outputs).astype(np.float32)
        initializers.append(numpy_helper.from_array(bias, name + ".bias"))
        nodes.append(
            helper.make_node(
                "Conv",
                [source, name + ".weight", name + ".bias"],
                [name + "_out"],
                name=name,
                kernel_shape=[kernel, kernel],
                strides=[stride, stride],
                pads=[pad] * 4,
            )
        )
        return name + "_out"

    def silu(name: str, source: str) -> str:
        nodes.append(
            helper.make_node("Sigmoid", [source], [name + "_sigmoid"], name=name + "_sigmoid")
        )
        nodes.append(
            helper.make_node("Mul", [source, name + "_sigmoid"], [name + "_out"], name=name)
        )
        return name + "_out"

    x = silu("act0", conv("conv0", "images", 3, WIDTH, 6, 2, 2))
    x = silu("act1", conv("conv1", x, WIDTH, 32, 3, 2, 1))
    x = conv("detect", x, 32, len(ANCHORS) * (5 + CLASSES), 1, 1, 0)
    nodes.append(helper.make_node("Sigmoid", [x], ["output0"], name="head"))
    cells = SIZE // 4
    grid = np.broadcast_to(
        np.array(ANCHORS, np.float32)[None, :, None, None, :], (1, 3, cells, cells, 2)
    )
    initializers.append(numpy_helper.from_array(np.ascontiguousarray(grid), "anchor_grid"))
    graph = helper.make_graph(
        nodes,
        "tiny",
        [helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 3, SIZE, SIZE])],
        [helper.make_tensor_value_info("output0", TensorProto.FLOAT, [1, 18, cells, cells])],
        initializers,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    onnx.save(model, path)


def make_images(directory: str, count: int = 4, seed: int = 0) -> list[str]:
    """Write a few random images and return their paths."""
    random = np.random.default_rng(seed)
    paths = []
    for k in range(count):
        blocks = random.integers(0, 256, (8, 8, 3), dtype=np.uint8)
        image = Image.fromarray(blocks).resize((SIZE, SIZE), Image.BILINEAR)
        paths.append(f"{directory}/image{k}.jpg")
        image.save(paths[-1])
    return paths
