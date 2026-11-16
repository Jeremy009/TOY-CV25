"""Writes the quantized network as a standard ONNX file.

Every primitive becomes standard ONNX nodes that compute it in floating point,
followed by a QuantizeLinear / DequantizeLinear pair that rounds and saturates the
result to the tensor's fixed-point format. Plain onnxruntime can run the file,
which makes it a quick preview of what quantization does to the detections, see
examples/check.py. It is a close model of the accelerator, not an exact one.
"""

from cv25yolo.compiler.records import Primitive


def export(primitives: list[Primitive], path: str) -> list[str]:
    """Write the quantized primitives to an ONNX file.

    The graph input is named "input" and takes the same planar uint8 image as the
    network binary.

    Args:
        primitives: The quantized primitives in program order.
        path: Path of the ONNX file to write.

    Returns:
        The names of the graph outputs: the network's output tensors, as float
        values.
    """
    raise NotImplementedError("not published")
