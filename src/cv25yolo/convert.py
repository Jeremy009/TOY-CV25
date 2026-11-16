"""The whole pipeline: from a YOLOv5 ONNX file to a network binary for the CV25.

1. read the ONNX file into layers                 compiler/onnx_import.py
2. split the layers that are too large            compiler/passes.py
3. lower the layers into the chip's primitives    compiler/lower.py
4. calibrate and choose fixed-point formats       compiler/quantize.py
5. cut into pieces and generate the binary        compiler/codegen.py
"""

import json
import os
import warnings
from collections.abc import Callable

import onnx

from cv25yolo.compiler import codegen, lower, onnx_import, passes, quantize, quantized_onnx
from cv25yolo.image import load_image

FIRST_CONV_CHANNELS = (16, 32)  # output channels of the first convolution of YOLOv5n and YOLOv5s
LARGEST_TESTED = (960, 544)  # the largest input that was run on a device (either orientation)


def convert(
    onnx_path: str,
    output_directory: str,
    calibration_images: list[str],
    stretch: bool = False,
    log: Callable[[str], None] = print,
) -> dict:
    """Convert a YOLOv5n or YOLOv5s ONNX export into a network binary.

    Three files are written to the output directory:

    - model.bin: the network binary, which larod loads on the camera. It takes a planar
      RGB uint8 image and returns the three Detect convolutions as integer tensors.
    - model_quantized.onnx: the same quantized network as a standard ONNX file.
    - model.json: what an application needs to use the binary: the input size, and per
      output its shape, number format, stride and anchors.

    Args:
        onnx_path: Path of the ONNX file, exported with a fixed input size.
        output_directory: Directory for the three files. It is created if needed.
        calibration_images: Paths of images like the ones the model will see in use. About
            64 is a good number.
        stretch: Set this when the application resizes camera frames to the input size
            without keeping the aspect ratio. By default the images are letterboxed.
        log: Called with progress messages.

    Returns:
        The content of model.json.

    Raises:
        ValueError: If the model is not supported or does not fit the chip.
    """
    if not calibration_images:
        raise ValueError("at least one calibration image is needed")
    model = onnx.load(onnx_path)
    layers, weights, input_shape = onnx_import.read_layers(model)
    anchors = onnx_import.read_anchors(model)
    output_count = len(layers[-1].inputs)
    if len(anchors) != output_count:
        raise ValueError(
            f"found {output_count} output convolutions but {len(anchors)} anchor sets: "
            "only anchor-based YOLOv5 Detect heads are supported"
        )
    first_conv = next(layer for layer in layers if layer.op == "conv")
    first_channels = weights[first_conv.name][0].shape[0]
    if first_channels not in FIRST_CONV_CHANNELS:
        raise ValueError(
            "only YOLOv5n and YOLOv5s are supported: their first convolution has 16 or 32 "
            f"output channels, this model has {first_channels}"
        )
    _, height, width = input_shape
    if height * width > LARGEST_TESTED[0] * LARGEST_TESTED[1]:
        warnings.warn(
            f"input {width} x {height} is larger than the largest tested input "
            f"({LARGEST_TESTED[0]} x {LARGEST_TESTED[1]}); the binary may not run",
            stacklevel=2,
        )
    log(f"read {os.path.basename(onnx_path)}: {len(layers)} layers, input {width} x {height}")

    layers, weights, split_blocks = passes.split_sppf(layers, weights)
    layers, weights, split_convolutions = passes.split_convs(layers, weights)
    for name in split_blocks:
        log(f"split the SPPF block at {name} into two halves")
    for description in split_convolutions:
        log(f"split convolution {description}")

    # Calibration runs on the primitives in file order. The tensor names are the same in
    # program order, so the measured ranges carry over.
    log(f"calibrating on {len(calibration_images)} images")
    images = (load_image(path, height, width, stretch) for path in calibration_images)
    ranges = quantize.calibrate(
        lower.lower(layers, weights, input_shape, in_program_order=False), images
    )
    primitives = lower.lower(layers, weights, input_shape, in_program_order=True)
    primitives = quantize.quantize(primitives, ranges)
    wide = sum(bool(p.format.wide) for p in primitives if p.op == "conv")
    log(f"quantized: {len(primitives)} primitives, {wide} convolutions with a 16-bit output")

    os.makedirs(output_directory, exist_ok=True)
    quantized_onnx.export(primitives, os.path.join(output_directory, "model_quantized.onnx"))
    piece_count = codegen.generate(primitives, os.path.join(output_directory, "model.bin"), log)

    # The outputs, finest scale first, like the anchors.
    by_name = {p.name: p for p in primitives}
    outputs = sorted((by_name[name] for name in primitives[-1].inputs), key=lambda p: -p.shape[2])
    anchor_count = len(anchors[0])
    description = {
        "source": os.path.basename(onnx_path),
        "input": {
            "shape": [1, *input_shape],
            "layout": "planar RGB, uint8, values 0 to 255 (the binary divides by 255 itself)",
            "preprocessing": "stretch" if stretch else "letterbox",
        },
        "classes": outputs[0].shape[1] // anchor_count - 5,
        "outputs": [
            {
                "name": p.name,
                "shape": list(p.shape),
                "stride": height // p.shape[2],
                "anchors": scale_anchors.tolist(),
                "signed": bool(p.format.signed),
                "bits": 16 if p.format.wide else 8,
                "fraction_bits": p.format.fraction,
                # The size of the file that the device writes for this output. Not published.
                "file_bytes": ...,
            }
            for p, scale_anchors in zip(outputs, anchors)
        ],
        "pieces": piece_count,
    }
    with open(os.path.join(output_directory, "model.json"), "w") as file:
        json.dump(description, file, indent=1)
    log(f"wrote model.bin ({piece_count} pieces), model_quantized.onnx and model.json")
    return description
