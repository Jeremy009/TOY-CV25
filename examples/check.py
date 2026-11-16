"""Example: see on the PC what quantization does to the detections.

The converter writes model_quantized.onnx next to model.bin: the network as the chip computes
it, in a form that plain onnxruntime can run. This script runs the original (float) ONNX file
and the quantized one on the same images and compares the detections.

    python examples/check.py yolov5n.onnx out/ photo1.jpg photo2.jpg ...

The quantized ONNX file is a close model of the chip, not an exact one.
"""

import json
import os
import sys

import numpy as np
import onnx
import onnxruntime
from decode import decode, iou, select

from cv25yolo.image import load_image


def float_session(
    onnx_path: str,
    output_names: list[str],
) -> tuple[onnxruntime.InferenceSession, list[str]]:
    """Open the float model so that it returns the Detect convolutions.

    A YOLOv5 export returns decoded boxes. To compare like with like, the outputs of the
    Detect convolutions are added to the graph outputs and decoded with the same code as the
    chip's outputs.

    Args:
        onnx_path: Path of the float ONNX file.
        output_names: Names of the Detect convolutions, from model.json.

    Returns:
        The session and the names of the tensors to fetch.
    """
    model = onnx.load(onnx_path)
    tensor_of = {(node.name or node.output[0]): node.output[0] for node in model.graph.node}
    tensors = [tensor_of[name] for name in output_names]
    existing = {output.name for output in model.graph.output}
    for tensor in tensors:
        if tensor not in existing:
            model.graph.output.append(onnx.ValueInfoProto(name=tensor))
    session = onnxruntime.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    return session, tensors


def count_matches(
    a: np.ndarray,
    b: np.ndarray,
) -> int:
    """Count the detections of `a` that have a partner in `b`: same class, IoU of 0.5 or more."""
    free = list(range(len(b)))
    matches = 0
    for box in a:
        for j in free:
            if box[5] == b[j][5] and iou(box[:4], b[j][:4]) >= 0.5:
                free.remove(j)
                matches += 1
                break
    return matches


def main() -> None:
    """Compare the float and the quantized model on the images given on the command line."""
    if len(sys.argv) < 4:
        sys.exit(__doc__)
    onnx_path, directory, image_paths = sys.argv[1], sys.argv[2], sys.argv[3:]
    with open(os.path.join(directory, "model.json")) as file:
        description = json.load(file)
    _, _, height, width = description["input"]["shape"]
    stretch = description["input"]["preprocessing"] == "stretch"
    names = [output["name"] for output in description["outputs"]]
    float_model, float_tensors = float_session(onnx_path, names)
    float_input = float_model.get_inputs()[0].name
    quantized_model = onnxruntime.InferenceSession(
        os.path.join(directory, "model_quantized.onnx"), providers=["CPUExecutionProvider"]
    )
    totals = np.zeros(3, int)
    for path in image_paths:
        image = load_image(path, height, width, stretch)
        as_float = image[None].astype(np.float32) / 255
        float_outputs = float_model.run(float_tensors, {float_input: as_float})
        quantized_outputs = quantized_model.run(names, {"input": image[None]})
        float_boxes = select(decode([t[0] for t in float_outputs], description))
        quantized_boxes = select(decode([t[0] for t in quantized_outputs], description))
        matches = count_matches(quantized_boxes, float_boxes)
        totals += (len(float_boxes), len(quantized_boxes), matches)
        counts = f"float {len(float_boxes)}, quantized {len(quantized_boxes)}, in both {matches}"
        print(f"{os.path.basename(path)}: {counts}")
    print(f"total: float {totals[0]} detections, quantized {totals[1]}, in both {totals[2]}")


if __name__ == "__main__":
    main()
